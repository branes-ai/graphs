"""
Hierarchical Table Formatter for Graph Partitioner
===================================================

Creates fvcore-style hierarchical tables showing module structure,
parameters, FLOPs, and memory usage.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class ModuleStats:
    """Statistics for a module or submodule"""
    name: str
    level: int  # Indentation level
    parameters: int  # Total parameters
    parameter_shapes: List[Tuple[str, tuple]]  # [(param_name, shape), ...]
    flops: int
    macs: int
    input_bytes: int
    output_bytes: int
    weight_bytes: int
    total_memory: int  # input + output + weight
    order: int = 0  # Execution order from FX graph (for sorting)
    tensor_shape: Optional[tuple] = None  # Output tensor shape (for all operations)


class HierarchicalTableFormatter:
    """Format partitioner results as hierarchical table"""

    def __init__(self):
        self.module_stats: Dict[str, ModuleStats] = {}

    def format_table(self, fx_graph, report, show_shapes: bool = False) -> str:
        """
        Generate hierarchical table from partition report

        Args:
            fx_graph: PyTorch FX graph
            report: PartitionReport from graph partitioner
            show_shapes: If True, show parameter shapes; if False, hide them (default)

        Returns:
            Formatted markdown table string
        """
        # Build module hierarchy from FX graph and partition report
        self._build_module_hierarchy(fx_graph, report)

        # Sort modules by hierarchy
        sorted_modules = self._sort_by_hierarchy()

        # Format as table
        return self._generate_table(sorted_modules, show_shapes)

    def _build_module_hierarchy(self, fx_graph, report):
        """Build hierarchical module statistics"""
        from torch.fx import GraphModule

        # Build node order mapping from FX graph (execution order)
        node_order = {}
        for idx, node in enumerate(fx_graph.graph.nodes):
            node_order[node.name] = idx

        # Get model parameters
        total_params = sum(p.numel() for p in fx_graph.parameters())

        # Create root entry
        total_flops = sum(sg.flops for sg in report.subgraphs)
        total_macs = sum(sg.macs for sg in report.subgraphs)
        total_input = sum(sg.total_input_bytes for sg in report.subgraphs)
        total_output = sum(sg.total_output_bytes for sg in report.subgraphs)
        total_weight = sum(sg.total_weight_bytes for sg in report.subgraphs)

        self.module_stats['model'] = ModuleStats(
            name='model',
            level=0,
            parameters=total_params,
            parameter_shapes=[],
            flops=total_flops,
            macs=total_macs,
            input_bytes=total_input,
            output_bytes=total_output,
            weight_bytes=total_weight,
            total_memory=total_input + total_output + total_weight,
            order=0
        )

        # Process each subgraph
        for sg in report.subgraphs:
            # Find the node in FX graph to get module target and tensor shape
            module_path = None
            param_count = 0
            param_shapes = []
            tensor_shape = None

            try:
                if hasattr(fx_graph, 'get_submodule'):
                    # Find the node in the graph
                    for node in fx_graph.graph.nodes:
                        if node.name == sg.node_name:
                            # Get tensor shape from metadata
                            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                                tensor_meta = node.meta['tensor_meta']
                                if hasattr(tensor_meta, 'shape'):
                                    tensor_shape = tuple(tensor_meta.shape)

                            if node.op == 'call_module':
                                # Use the target path, but append call suffix if present
                                # e.g., "layer1_0_relu_1" -> target "layer1.0.relu" + suffix "_1"
                                module_path = node.target

                                # Check if node name has a call suffix (ends with _N)
                                parts = sg.node_name.split('_')
                                if len(parts) >= 2 and parts[-1].isdigit():
                                    # Check if this is a call suffix by seeing if target matches
                                    # the node name without the suffix
                                    expected_base = '.'.join(parts[:-1])
                                    if expected_base.replace('_', '.') == node.target:
                                        # It's a call suffix, append it
                                        module_path = f"{node.target}_{parts[-1]}"

                                module = fx_graph.get_submodule(node.target)
                                param_count = sum(p.numel() for p in module.parameters())
                                param_shapes = [(name, tuple(p.shape)) for name, p in module.named_parameters()]
                            else:
                                # call_function - parse node name with suffix handling
                                module_path = self._node_name_to_module_path(sg.node_name)
                            break
            except:
                pass

            # Fallback: parse node name if target not found
            if module_path is None:
                module_path = self._node_name_to_module_path(sg.node_name)

            # Create stats for this module
            # Level is dot count + 1 (to indent under 'model' which is level 0)
            self.module_stats[module_path] = ModuleStats(
                name=module_path,
                level=module_path.count('.') + 1,
                parameters=param_count,
                parameter_shapes=param_shapes,
                flops=sg.flops,
                macs=sg.macs,
                input_bytes=sg.total_input_bytes,
                output_bytes=sg.total_output_bytes,
                weight_bytes=sg.total_weight_bytes,
                total_memory=sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes,
                order=node_order.get(sg.node_name, 999999),  # Use execution order from FX graph
                tensor_shape=tensor_shape  # Output tensor shape
            )

        # Aggregate parent statistics
        self._aggregate_parent_stats()

    def _node_name_to_module_path(self, node_name: str) -> str:
        """Convert FX node name to module path

        Handles two types of numeric parts:
        1. Module indices (middle): layer1_0_conv1 -> layer1.0.conv1
        2. FX call suffixes (end): layer1_0_relu_1 -> layer1.0.relu_1

        The distinction: if last part is digit AND previous part is non-digit,
        it's a call suffix (keep underscore). Otherwise it's a module index (use dot).
        """
        parts = node_name.split('_')

        if len(parts) == 1:
            return parts[0]

        # Check if last part is a digit (potential FX call suffix)
        if parts[-1].isdigit() and len(parts) >= 2 and not parts[-2].isdigit():
            # Last part is FX call suffix (e.g., "relu_1"), keep underscore
            # Convert all but last part to dots, then append suffix with underscore
            module_path = '.'.join(parts[:-1])
            return f"{module_path}_{parts[-1]}"
        else:
            # Regular module hierarchy, use dots throughout
            return '.'.join(parts)

    def _aggregate_parent_stats(self):
        """Aggregate statistics for parent modules"""
        # Get all unique parent paths
        parent_paths = set()
        for module_path in list(self.module_stats.keys()):  # Use list() to avoid modification during iteration
            if module_path == 'model':
                continue

            parts = module_path.split('.')
            for i in range(1, len(parts)):
                parent_path = '.'.join(parts[:i])
                parent_paths.add(parent_path)

        # Create parent entries and aggregate
        # Sort by depth (deepest first) so "layer1.0" is created before "layer1"
        for parent_path in sorted(parent_paths, key=lambda p: (-p.count('.'), p)):
            # Level is dot count + 1 (to indent under 'model' which is level 0)
            level = parent_path.count('.') + 1

            # Find all children (children have one more dot than parent)
            children = [path for path in self.module_stats.keys()
                       if path.startswith(parent_path + '.') and
                       path.count('.') == parent_path.count('.') + 1]

            if children:
                total_params = sum(self.module_stats[c].parameters for c in children)
                total_flops = sum(self.module_stats[c].flops for c in children)
                total_macs = sum(self.module_stats[c].macs for c in children)
                total_input = sum(self.module_stats[c].input_bytes for c in children)
                total_output = sum(self.module_stats[c].output_bytes for c in children)
                total_weight = sum(self.module_stats[c].weight_bytes for c in children)
                # Parent uses minimum order of children (appears before first child)
                min_order = min(self.module_stats[c].order for c in children)

                self.module_stats[parent_path] = ModuleStats(
                    name=parent_path,
                    level=level,
                    parameters=total_params,
                    parameter_shapes=[],
                    flops=total_flops,
                    macs=total_macs,
                    input_bytes=total_input,
                    output_bytes=total_output,
                    weight_bytes=total_weight,
                    total_memory=total_input + total_output + total_weight,
                    order=min_order - 0.5  # Place parent just before first child
                )

    def _sort_by_hierarchy(self) -> List[str]:
        """Sort modules by execution order from FX graph (depth-first)"""
        # Build sort key that encodes full ancestor path with execution orders
        # This ensures proper depth-first traversal
        def sort_key(path):
            if path == 'model':
                return (0,)

            # Build key from all ancestors + self
            parts = path.split('.')
            key = [0]  # Start with model order

            # Add order for each ancestor in the path
            for i in range(1, len(parts) + 1):
                ancestor_path = '.'.join(parts[:i])
                if ancestor_path in self.module_stats:
                    key.append(self.module_stats[ancestor_path].order)
                else:
                    # Should not happen, but handle gracefully
                    key.append(999999)

            return tuple(key)

        return sorted(self.module_stats.keys(), key=sort_key)

    def _format_number(self, num: int, suffix: str = '') -> str:
        """Format large numbers with K/M/G suffix"""
        if num == 0:
            return ''

        if num >= 1e9:
            return f"{num / 1e9:.3f}G{suffix}"
        elif num >= 1e6:
            return f"{num / 1e6:.3f}M{suffix}"
        elif num >= 1e3:
            return f"{num / 1e3:.3f}K{suffix}"
        else:
            return f"{num}{suffix}"

    def _format_memory(self, bytes: int) -> str:
        """Format memory in bytes with KB/MB/GB suffix"""
        if bytes == 0:
            return ''

        if bytes >= 1e9:
            return f"{bytes / 1e9:.2f}GB"
        elif bytes >= 1e6:
            return f"{bytes / 1e6:.2f}MB"
        elif bytes >= 1e3:
            return f"{bytes / 1e3:.2f}KB"
        else:
            return f"{bytes}B"

    def _generate_table(self, sorted_modules: List[str], show_shapes: bool) -> str:
        """Generate formatted markdown table"""
        lines = []

        # Header - Tensor Shape column right after #Parameters (operator info grouped together)
        if show_shapes:
            lines.append(f"| {'Module':<35} | {'#Parameters':<20} | {'Tensor Shape':<20} | {'MACs':<12} | {'FLOPs':<12} | {'Memory':<12} |")
            lines.append(f"|:{'-'*36}|:{'-'*21}|:{'-'*21}|:{'-'*13}|:{'-'*13}|:{'-'*13}|")
        else:
            lines.append(f"| {'Module':<35} | {'#Parameters':<20} | {'MACs':<12} | {'FLOPs':<12} | {'Memory':<12} |")
            lines.append(f"|:{'-'*36}|:{'-'*21}|:{'-'*13}|:{'-'*13}|:{'-'*13}|")

        # Rows
        for module_path in sorted_modules:
            stats = self.module_stats[module_path]

            # Indentation based on level
            indent = ' ' * stats.level

            # Display name (show more context for ambiguous names)
            parts = module_path.split('.')
            if len(parts) > 1 and parts[-1].isdigit() and stats.level > 0:
                # For numeric names like "0", "1", show parent context
                # "layer1.0" -> show as "0" but we want to show what layer
                if len(parts) >= 2:
                    name = indent + '.'.join(parts[-2:])  # e.g., "layer1.0"
                else:
                    name = indent + parts[-1]
            else:
                name = indent + parts[-1]

            # Format parameters
            if stats.parameter_shapes:
                # Show shape for leaf modules with parameters
                param_str = self._format_number(stats.parameters, '')
            elif stats.parameters > 0:
                param_str = self._format_number(stats.parameters, '')
            else:
                param_str = ''

            # Format compute - separate MACs and FLOPs columns
            macs_str = self._format_number(stats.macs, '')
            flops_str = self._format_number(stats.flops, '')

            # Format memory
            memory_str = self._format_memory(stats.total_memory)

            # Format tensor shape (after #Parameters, before MACs)
            if show_shapes:
                shape_str = str(stats.tensor_shape) if stats.tensor_shape else ''
                lines.append(f"| {name:<35} | {param_str:<20} | {shape_str:<20} | {macs_str:<12} | {flops_str:<12} | {memory_str:<12} |")
            else:
                lines.append(f"| {name:<35} | {param_str:<20} | {macs_str:<12} | {flops_str:<12} | {memory_str:<12} |")

            # Add parameter shapes for leaf modules (only if show_shapes is True)
            if show_shapes and stats.parameter_shapes:
                param_indent = ' ' * (stats.level + 1)  # One level deeper than parent
                # Get the display name (same logic as module name)
                if len(parts) > 1 and parts[-1].isdigit() and stats.level > 0:
                    if len(parts) >= 2:
                        module_display = '.'.join(parts[-2:])
                    else:
                        module_display = parts[-1]
                else:
                    module_display = parts[-1]

                for param_name, shape in stats.parameter_shapes:
                    param_shape_str = str(shape)
                    # Show parameter relative to display name, not full path
                    param_display_name = f"{param_indent}{module_display}.{param_name}"
                    # Parameter rows have shape in Tensor Shape column (3rd column)
                    lines.append(f"| {param_display_name:<35} | {'':<20} | {param_shape_str:<20} | {'':<12} | {'':<12} | {'':<12} |")

        # Add footnote explaining compute metrics and shape distinctions
        lines.append("")
        lines.append("Compute Metrics:")
        lines.append("  - Conv2d/Linear: MACs (multiply-accumulate operations)")
        lines.append("  - BatchNorm: 5 FLOPs/element (normalize + scale + shift)")
        lines.append("  - ReLU: 1 FLOP/element (max(0,x) comparison)")
        lines.append("  - Add/Mul/Sub/Div: 1 FLOP/element (elementwise operation)")
        lines.append("  - MaxPool/AdaptiveAvgPool: 0 FLOPs (comparison-based, matches fvcore)")
        lines.append("")
        lines.append("Shape Information (shown with --showshape):")
        lines.append("  - Parameters: Learnable weights/biases (e.g., conv.weight shape)")
        lines.append("  - Tensor Shape: Output tensor dimensions during forward pass (e.g., [1, 64, 56, 56])")
        lines.append("  - Operations without parameters (ReLU, MaxPool) only show Tensor Shape")

        return '\n'.join(lines)


def format_partition_table(fx_graph, report, show_shapes: bool = False) -> str:
    """
    Convenience function to format partition report as hierarchical table

    Args:
        fx_graph: PyTorch FX graph
        report: PartitionReport from graph partitioner
        show_shapes: If True, show parameter shapes; if False, hide them (default)

    Returns:
        Formatted markdown table string
    """
    formatter = HierarchicalTableFormatter()
    return formatter.format_table(fx_graph, report, show_shapes)
