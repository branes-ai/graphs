"""
Mermaid diagram generator for FX graphs and hardware analysis.

This module converts PyTorch FX graphs and analysis results into Mermaid diagrams
for visualization in GitHub markdown and other markdown viewers.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import torch.fx as fx

from ..ir.structures import PartitionReport, SubgraphDescriptor


class ColorScheme(Enum):
    """Color scheme presets for visualization."""
    BOTTLENECK = 'bottleneck'      # Color by compute vs memory bound
    UTILIZATION = 'utilization'    # Color by hardware utilization
    OP_TYPE = 'op_type'            # Color by operation type
    DEFAULT = 'default'            # Minimal coloring


@dataclass
class MermaidStyle:
    """Styling configuration for Mermaid nodes and edges."""
    fill_color: str = '#808080'      # Node fill color (medium gray)
    stroke_color: str = '#000000'    # Node border color
    stroke_width: str = '2px'        # Border width
    text_color: str = '#000000'      # Text color

    def to_mermaid(self, node_id: str) -> str:
        """Convert style to Mermaid style syntax."""
        return f"style {node_id} fill:{self.fill_color},stroke:{self.stroke_color},stroke-width:{self.stroke_width}"


class ColorSchemeManager:
    """Manages color schemes for different visualization types."""

    # Color scheme: Bottleneck type (high contrast)
    BOTTLENECK_COLORS = {
        'compute_bound': '#228B22',    # Forest green (good with black text)
        'memory_bound': '#DC143C',     # Crimson red (good with white text)
        'balanced': '#FF8C00',         # Dark orange (good with black text)
        'idle': '#696969',             # Dim gray (good with white text)
        'unknown': '#808080',          # Gray (medium contrast)
    }

    # Color scheme: Utilization level (high contrast)
    UTILIZATION_COLORS = {
        'very_high': '#006400',    # Dark green (>80%) - white text
        'high': '#228B22',         # Forest green (60-80%) - white text
        'medium': '#FF8C00',       # Dark orange (40-60%) - black text
        'low': '#FFA500',          # Orange (20-40%) - black text
        'very_low': '#DC143C',     # Crimson red (<20%) - white text
        'idle': '#696969',         # Dim gray (0%) - white text
    }

    # Color scheme: Operation type (high contrast)
    OP_TYPE_COLORS = {
        'conv': '#1E90FF',         # Dodger blue (good contrast)
        'matmul': '#8A2BE2',       # Blue violet (good contrast)
        'linear': '#8A2BE2',       # Blue violet (good contrast)
        'activation': '#228B22',   # Forest green (good contrast)
        'relu': '#228B22',         # Forest green (good contrast)
        'normalization': '#DAA520', # Goldenrod (better than gold)
        'batchnorm': '#DAA520',    # Goldenrod (better than gold)
        'pooling': '#FF8C00',      # Dark orange (good contrast)
        'elementwise': '#008B8B',  # Dark cyan (better than light sea green)
        'add': '#008B8B',          # Dark cyan (better than light sea green)
        'mul': '#008B8B',          # Dark cyan (better than light sea green)
        'default': '#808080',      # Medium gray (better contrast)
    }

    @staticmethod
    def get_bottleneck_color(compute_pct: float, memory_pct: float) -> str:
        """Get color based on bottleneck type."""
        if compute_pct > 70 and memory_pct < 30:
            return ColorSchemeManager.BOTTLENECK_COLORS['compute_bound']
        elif memory_pct > 70 and compute_pct < 30:
            return ColorSchemeManager.BOTTLENECK_COLORS['memory_bound']
        elif compute_pct > 40 and memory_pct > 40:
            return ColorSchemeManager.BOTTLENECK_COLORS['balanced']
        else:
            return ColorSchemeManager.BOTTLENECK_COLORS['unknown']

    @staticmethod
    def get_utilization_color(utilization_pct: float) -> str:
        """Get color based on utilization percentage."""
        if utilization_pct >= 80:
            return ColorSchemeManager.UTILIZATION_COLORS['very_high']
        elif utilization_pct >= 60:
            return ColorSchemeManager.UTILIZATION_COLORS['high']
        elif utilization_pct >= 40:
            return ColorSchemeManager.UTILIZATION_COLORS['medium']
        elif utilization_pct >= 20:
            return ColorSchemeManager.UTILIZATION_COLORS['low']
        elif utilization_pct > 0:
            return ColorSchemeManager.UTILIZATION_COLORS['very_low']
        else:
            return ColorSchemeManager.UTILIZATION_COLORS['idle']

    @staticmethod
    def get_op_type_color(op_name: str) -> str:
        """Get color based on operation type."""
        op_lower = op_name.lower()
        for key, color in ColorSchemeManager.OP_TYPE_COLORS.items():
            if key in op_lower:
                return color
        return ColorSchemeManager.OP_TYPE_COLORS['default']


class MermaidGenerator:
    """
    Generate Mermaid diagrams from FX graphs and analysis results.

    Supports:
    - FX graph structure visualization
    - Partitioned/fused graph visualization
    - Hardware mapping visualization
    - Bottleneck analysis visualization
    - Multi-architecture comparison
    """

    def __init__(self, style: str = 'default'):
        """
        Initialize Mermaid generator.

        Args:
            style: Visualization style preset
                - 'default': Clean, minimal
                - 'detailed': Show all metadata
                - 'compact': Minimal labels
                - 'colorful': Color-coded by operation type
        """
        self.style = style
        self.color_manager = ColorSchemeManager()

    def generate_fx_graph(
        self,
        graph_module: fx.GraphModule,
        direction: str = 'TD',
        max_nodes: int = 100,
        show_shapes: bool = True,
        show_types: bool = True,
    ) -> str:
        """
        Generate Mermaid diagram from FX graph.

        Args:
            graph_module: PyTorch FX GraphModule
            direction: 'TD' (top-down) or 'LR' (left-right)
            max_nodes: Maximum nodes to show (for large graphs)
            show_shapes: Show tensor shapes in labels
            show_types: Show operation types

        Returns:
            Mermaid diagram syntax as string
        """
        lines = [f"graph {direction}"]

        # Track node count
        node_count = 0
        node_ids = {}

        # Generate nodes
        for node in graph_module.graph.nodes:
            if node_count >= max_nodes:
                lines.append(f"    Truncated[... {len(list(graph_module.graph.nodes)) - max_nodes} more nodes ...]")
                break

            node_id = f"N{node_count}"
            node_ids[node.name] = node_id

            # Build label
            label_parts = [node.name]
            if show_types and node.op != 'placeholder':
                label_parts.append(f"〈{node.op}〉")  # Use angle brackets instead of []
            if show_shapes and hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                shape = node.meta['tensor_meta'].shape
                label_parts.append(f"shape: {list(shape)}")

            label = self._sanitize_label("<br/>".join(label_parts))
            lines.append(f"    {node_id}[{label}]")

            # Apply color based on op type if colorful style
            if self.style == 'colorful':
                color = self.color_manager.get_op_type_color(node.name)
                lines.append(f"    style {node_id} fill:{color}")

            node_count += 1

        # Generate edges
        for node in graph_module.graph.nodes:
            if node.name not in node_ids:
                break

            src_id = node_ids[node.name]
            for user in node.users:
                if user.name in node_ids:
                    dst_id = node_ids[user.name]
                    lines.append(f"    {src_id} --> {dst_id}")

        return "\n".join(lines)

    def generate_partitioned_graph(
        self,
        partition_report,  # PartitionReport or FusionReport
        direction: str = 'TD',
        color_by: str = 'bottleneck',
        show_metrics: bool = True,
        max_subgraphs: int = 50,
    ) -> str:
        """
        Generate Mermaid diagram showing partitioned/fused graph.

        Args:
            partition_report: Partitioning analysis results (PartitionReport or FusionReport)
            direction: 'TD' (top-down) or 'LR' (left-right)
            color_by: 'bottleneck', 'utilization', 'op_type'
            show_metrics: Show FLOPs, memory, latency metrics
            max_subgraphs: Maximum subgraphs to show

        Returns:
            Mermaid diagram syntax as string
        """
        lines = [f"graph {direction}"]
        lines.append("")

        # Add start node
        lines.append("    Start([Input])")
        lines.append("    style Start fill:#ADD8E6,stroke:#000080,stroke-width:3px")
        lines.append("")

        # Generate subgraphs - handle both PartitionReport and FusionReport
        if hasattr(partition_report, 'fused_subgraphs'):
            subgraphs = partition_report.fused_subgraphs[:max_subgraphs]
        else:
            subgraphs = partition_report.subgraphs[:max_subgraphs]

        for idx, sg in enumerate(subgraphs):
            sg_id = f"SG{idx}"

            # Build subgraph header
            header_parts = [f"Subgraph {idx}"]

            # Add operation list
            ops_list = self._get_ops_list(sg)
            if ops_list:
                op_names = [op.split('.')[-1] for op in ops_list[:3]]  # First 3 ops
                if len(ops_list) > 3:
                    op_names.append(f"... +{len(ops_list)-3} more")
                header_parts.append(" → ".join(op_names))

            # Add metrics if requested
            if show_metrics:
                metrics_parts = []
                # Handle both 'flops' and 'total_flops' attributes
                flops = getattr(sg, 'total_flops', getattr(sg, 'flops', 0))
                if flops > 0:
                    flops_str = self._format_flops(flops)
                    metrics_parts.append(flops_str)

                # Add bottleneck type
                if hasattr(sg, 'compute_bound_pct') and hasattr(sg, 'memory_bound_pct'):
                    if sg.compute_bound_pct > 70:
                        metrics_parts.append("Compute-bound")
                    elif sg.memory_bound_pct > 70:
                        metrics_parts.append("Memory-bound")
                    else:
                        metrics_parts.append("Balanced")

                if metrics_parts:
                    header_parts.append(", ".join(metrics_parts))

            header = self._sanitize_label("<br/>".join(header_parts))

            # Create subgraph with single node
            lines.append(f"    subgraph {sg_id}[\"{header}\"]")
            node_id = f"{sg_id}_exec"

            # Build node label
            node_label_parts = []
            flops = getattr(sg, 'total_flops', getattr(sg, 'flops', 0))
            if show_metrics and flops > 0:
                node_label_parts.append(f"{self._format_flops(flops)}")
            # Handle both 'memory_bytes' and various memory attributes
            memory_bytes = getattr(sg, 'memory_bytes',
                                   getattr(sg, 'total_input_bytes', 0) +
                                   getattr(sg, 'total_output_bytes', 0))
            if show_metrics and memory_bytes > 0:
                node_label_parts.append(f"{self._format_memory(memory_bytes)}")

            if node_label_parts:
                node_label = self._sanitize_label("<br/>".join(node_label_parts))
            else:
                node_label = "Execute"

            # Add invisible spacer to prevent node from covering subgraph label
            lines.append(f"        {sg_id}_spacer[ ]")
            lines.append(f"        {sg_id}_spacer --> {node_id}")
            lines.append(f"        {node_id}[{node_label}]")
            lines.append(f"    end")
            lines.append("")

            # Apply color scheme to subgraph
            color = self._get_subgraph_color(sg, color_by)
            lines.append(f"    style {sg_id} fill:{color},stroke:#000000,stroke-width:2px")

            # Hide the spacer node
            lines.append(f"    style {sg_id}_spacer fill:none,stroke:none")

        # Add truncation notice if needed
        total_subgraphs = len(partition_report.fused_subgraphs) if hasattr(partition_report, 'fused_subgraphs') else len(partition_report.subgraphs)
        if total_subgraphs > max_subgraphs:
            lines.append(f"    Truncated[... {total_subgraphs - max_subgraphs} more subgraphs ...]")
            lines.append("    style Truncated fill:#696969")

        # Add end node
        lines.append("    End([Output])")
        lines.append("    style End fill:#228B22,stroke:#006400,stroke-width:3px")
        lines.append("")

        # Generate edges (sequential flow)
        lines.append("    Start --> SG0_exec")
        for idx in range(len(subgraphs) - 1):
            src = f"SG{idx}_exec"
            dst = f"SG{idx+1}_exec"
            lines.append(f"    {src} --> {dst}")

        if subgraphs:
            lines.append(f"    SG{len(subgraphs)-1}_exec --> End")

        return "\n".join(lines)

    def generate_hardware_mapping(
        self,
        partition_report,  # PartitionReport or FusionReport
        hardware_name: str,
        peak_compute_units: int,
        direction: str = 'TD',
        show_allocation: bool = True,
        show_utilization: bool = True,
        max_subgraphs: int = 20,
    ) -> str:
        """
        Generate Mermaid diagram showing hardware resource mapping.

        Args:
            partition_report: Partitioning analysis results (PartitionReport or FusionReport)
            hardware_name: Name of hardware (e.g., "H100 GPU")
            peak_compute_units: Total available compute units
            direction: 'TD' (top-down) or 'LR' (left-right)
            show_allocation: Show resource allocation per subgraph
            show_utilization: Show utilization percentages
            max_subgraphs: Maximum subgraphs to show

        Returns:
            Mermaid diagram syntax as string
        """
        lines = [f"graph {direction}"]
        lines.append("")

        # Hardware overview
        lines.append(f"    HW[{hardware_name}<br/>{peak_compute_units} Compute Units]")
        lines.append("    style HW fill:#87CEEB,stroke:#000080,stroke-width:4px")
        lines.append("")

        # Handle both PartitionReport and FusionReport
        if hasattr(partition_report, 'fused_subgraphs'):
            subgraphs = partition_report.fused_subgraphs[:max_subgraphs]
        else:
            subgraphs = partition_report.subgraphs[:max_subgraphs]

        # Calculate allocations and utilizations
        allocations = []
        utilizations = []

        for sg in subgraphs:
            # Get allocated units (from parallelism descriptor if available)
            allocated = 0
            utilization = 0.0

            if hasattr(sg, 'parallelism') and sg.parallelism:
                # Try to extract allocated resources
                if hasattr(sg.parallelism, 'num_blocks'):
                    allocated = sg.parallelism.num_blocks
                elif hasattr(sg.parallelism, 'num_cores'):
                    allocated = sg.parallelism.num_cores

            # Calculate utilization (allocated / peak)
            if peak_compute_units > 0:
                utilization = (allocated / peak_compute_units) * 100.0

            allocations.append(allocated)
            utilizations.append(utilization)

        # Generate subgraph visualizations
        for idx, sg in enumerate(subgraphs):
            sg_id = f"SG{idx}"
            allocated = allocations[idx]
            util = utilizations[idx]

            # Build label
            label_parts = [f"Subgraph {idx}"]

            ops_list = self._get_ops_list(sg)
            if ops_list:
                op_names = [op.split('.')[-1] for op in ops_list[:2]]
                if len(ops_list) > 2:
                    op_names.append(f"+{len(ops_list)-2}")
                label_parts.append(" → ".join(op_names))

            if show_allocation and allocated > 0:
                label_parts.append(f"{allocated} units")

            if show_utilization:
                label_parts.append(f"{util:.1f}% util")

            # Add latency if available
            if hasattr(sg, 'latency_ms') and sg.latency_ms > 0:
                label_parts.append(f"{sg.latency_ms:.3f}ms")

            # Add bottleneck warning
            if hasattr(sg, 'memory_bound_pct') and sg.memory_bound_pct > 70:
                label_parts.append("⚠️ Memory-bound")

            label = self._sanitize_label("<br/>".join(label_parts))

            lines.append(f"    {sg_id}[{label}]")

            # Color by utilization
            color = self.color_manager.get_utilization_color(util)
            lines.append(f"    style {sg_id} fill:{color},stroke:#000000,stroke-width:2px")

        # Show idle resources if significant
        if subgraphs:
            avg_utilization = sum(utilizations) / len(utilizations)
            idle_pct = 100.0 - avg_utilization

            if idle_pct > 20:
                lines.append("")
                idle_units = int(peak_compute_units * (idle_pct / 100.0))
                lines.append(f"    Idle[IDLE RESOURCES<br/>{idle_units} units<br/>{idle_pct:.1f}% of hardware]")
                lines.append("    style Idle fill:#DC143C,stroke:#8B0000,stroke-width:3px")

        # Add truncation notice
        total_subgraphs = len(partition_report.fused_subgraphs) if hasattr(partition_report, 'fused_subgraphs') else len(partition_report.subgraphs)
        if total_subgraphs > max_subgraphs:
            lines.append("")
            lines.append(f"    Truncated[... {total_subgraphs - max_subgraphs} more subgraphs ...]")
            lines.append("    style Truncated fill:#696969")

        # Generate edges (sequential execution)
        lines.append("")
        lines.append("    HW --> SG0")
        for idx in range(len(subgraphs) - 1):
            lines.append(f"    SG{idx} --> SG{idx+1}")

        return "\n".join(lines)

    def generate_architecture_comparison(
        self,
        partition_reports: List[Tuple[str, PartitionReport]],  # [(arch_name, report), ...]
        peak_compute_units: List[int],
        layout: str = 'side_by_side',
        max_subgraphs: int = 10,
    ) -> str:
        """
        Generate side-by-side comparison of 2-3 architectures.

        Args:
            partition_reports: List of (architecture_name, partition_report) tuples
            peak_compute_units: List of peak compute units for each architecture
            layout: 'side_by_side' or 'stacked'
            max_subgraphs: Maximum subgraphs to show per architecture

        Returns:
            Mermaid diagram syntax as string
        """
        if layout == 'side_by_side':
            direction = 'LR'
        else:
            direction = 'TD'

        lines = [f"graph {direction}"]
        lines.append("")

        # Generate subgraph for each architecture
        for arch_idx, ((arch_name, report), peak_units) in enumerate(zip(partition_reports, peak_compute_units)):
            arch_id = f"ARCH{arch_idx}"

            # Use ~ instead of : to avoid Mermaid parse errors
            lines.append(f"    subgraph {arch_id}[\"{arch_name} ~ {peak_units} units\"]")

            # Handle both PartitionReport and FusionReport
            if hasattr(report, 'fused_subgraphs'):
                subgraphs = report.fused_subgraphs[:max_subgraphs]
            else:
                subgraphs = report.subgraphs[:max_subgraphs]

            for sg_idx, sg in enumerate(subgraphs):
                node_id = f"{arch_id}_SG{sg_idx}"

                # Build label
                label_parts = []

                ops_list = self._get_ops_list(sg)
                if ops_list:
                    op_name = ops_list[0].split('.')[-1]
                    label_parts.append(op_name)
                else:
                    label_parts.append(f"SG{sg_idx}")

                # Calculate utilization
                allocated = 0
                if hasattr(sg, 'parallelism') and sg.parallelism:
                    if hasattr(sg.parallelism, 'num_blocks'):
                        allocated = sg.parallelism.num_blocks
                    elif hasattr(sg.parallelism, 'num_cores'):
                        allocated = sg.parallelism.num_cores

                util = (allocated / peak_units * 100.0) if peak_units > 0 else 0.0
                label_parts.append(f"{util:.0f}%")

                # Add latency if available
                if hasattr(sg, 'latency_ms') and sg.latency_ms > 0:
                    label_parts.append(f"{sg.latency_ms:.2f}ms")

                label = self._sanitize_label("<br/>".join(label_parts))
                lines.append(f"        {node_id}[{label}]")

                # Color by utilization
                color = self.color_manager.get_utilization_color(util)
                lines.append(f"        style {node_id} fill:{color}")

                # Add edges within architecture
                if sg_idx > 0:
                    prev_node = f"{arch_id}_SG{sg_idx-1}"
                    lines.append(f"        {prev_node} --> {node_id}")

            lines.append("    end")
            lines.append("")

        return "\n".join(lines)

    def generate_bottleneck_analysis(
        self,
        partition_report,  # PartitionReport or FusionReport
        threshold: float = 0.2,
        direction: str = 'TD',
        max_subgraphs: int = 30,
    ) -> str:
        """
        Generate diagram highlighting bottleneck operations.

        Args:
            partition_report: Partitioning analysis results (PartitionReport or FusionReport)
            threshold: Highlight subgraphs using >threshold of total time
            direction: 'TD' (top-down) or 'LR' (left-right)
            max_subgraphs: Maximum subgraphs to show

        Returns:
            Mermaid diagram syntax as string
        """
        lines = [f"graph {direction}"]
        lines.append("")

        # Handle both PartitionReport and FusionReport
        if hasattr(partition_report, 'fused_subgraphs'):
            all_subgraphs = partition_report.fused_subgraphs
        else:
            all_subgraphs = partition_report.subgraphs

        # Calculate total latency
        total_latency = 0.0
        for sg in all_subgraphs:
            if hasattr(sg, 'latency_ms'):
                total_latency += sg.latency_ms

        if total_latency == 0:
            total_latency = 1.0  # Avoid division by zero

        lines.append(f"    Start[Total Latency: {total_latency:.2f}ms]")
        lines.append("    style Start fill:#87CEEB,stroke:#000080,stroke-width:3px")
        lines.append("")

        subgraphs = all_subgraphs[:max_subgraphs]

        # Generate nodes with bottleneck highlighting
        for idx, sg in enumerate(subgraphs):
            node_id = f"SG{idx}"

            # Calculate percentage of total time
            latency = sg.latency_ms if hasattr(sg, 'latency_ms') else 0.0
            pct = (latency / total_latency * 100.0) if total_latency > 0 else 0.0

            # Build label
            label_parts = [f"Subgraph {idx}"]

            ops_list = self._get_ops_list(sg)
            if ops_list:
                op_name = ops_list[0].split('.')[-1]
                label_parts.append(op_name)

            label_parts.append(f"{latency:.2f}ms ~ {pct:.0f}%")  # Use ~ instead of ()

            # Add bottleneck indicator
            is_bottleneck = pct >= (threshold * 100)
            if is_bottleneck:
                label_parts.append("🔥 BOTTLENECK")

            label = self._sanitize_label("<br/>".join(label_parts))
            lines.append(f"    {node_id}[{label}]")

            # Color based on percentage
            if pct >= 20:
                # Critical bottleneck
                lines.append(f"    style {node_id} fill:#FF0000,stroke:#8B0000,stroke-width:4px")
            elif pct >= 15:
                # Significant contributor
                lines.append(f"    style {node_id} fill:#DC143C,stroke:#8B0000,stroke-width:2px")
            elif pct >= 10:
                # Moderate contributor
                lines.append(f"    style {node_id} fill:#FF8C00,stroke:#000000,stroke-width:2px")
            else:
                # Minor contributor
                lines.append(f"    style {node_id} fill:#808080,stroke:#000000,stroke-width:1px")

        # Generate edges
        lines.append("")
        lines.append("    Start --> SG0")
        for idx in range(len(subgraphs) - 1):
            lines.append(f"    SG{idx} --> SG{idx+1}")

        return "\n".join(lines)

    def generate_legend(self, scheme: ColorScheme) -> str:
        """
        Generate a legend explaining the color scheme.

        Args:
            scheme: Color scheme being used

        Returns:
            Markdown text with legend
        """
        if scheme == ColorScheme.BOTTLENECK:
            return """
**Legend** (High Contrast Colors):
- 🟢 **Forest Green**: Compute-bound (efficient use of compute resources)
- 🔴 **Crimson Red**: Memory-bound (bottlenecked by memory bandwidth)
- 🟠 **Dark Orange**: Balanced (mixed compute and memory bound)
- ⚫ **Dim Gray**: Unknown or idle
"""
        elif scheme == ColorScheme.UTILIZATION:
            return """
**Legend** (High Contrast Colors):
- 🟢 **Dark Green**: Very high utilization (>80%)
- 🟢 **Forest Green**: High utilization (60-80%)
- 🟠 **Dark Orange**: Medium utilization (40-60%)
- 🟠 **Orange**: Low utilization (20-40%)
- 🔴 **Crimson**: Very low utilization (<20%)
- ⚫ **Dim Gray**: Idle (0%)
"""
        elif scheme == ColorScheme.OP_TYPE:
            return """
**Legend** (High Contrast Colors):
- 🔵 **Dodger Blue**: Convolution operations
- 🟣 **Blue Violet**: Matrix multiplication / Linear layers
- 🟢 **Forest Green**: Activation functions
- 🟡 **Goldenrod**: Normalization layers
- 🟠 **Dark Orange**: Pooling operations
- 🔷 **Dark Cyan**: Element-wise operations
"""
        else:
            return ""

    # Helper methods

    def _sanitize_label(self, label: str) -> str:
        """
        Sanitize label text for Mermaid compatibility.

        Mermaid has issues with:
        - Square brackets [] (node syntax)
        - Parentheses () (can cause parse issues)
        - Colons : (subgraph syntax)
        """
        # Replace square brackets with angle brackets
        label = label.replace('[', '〈').replace(']', '〉')
        # Replace colons with tildes in contexts where they might cause issues
        # But preserve them in simple time formats like "0.10ms"
        # This is conservative - we keep colons only in simple numeric contexts
        return label

    def _get_ops_list(self, sg):
        """Get operations list from subgraph, handling different attribute names."""
        if hasattr(sg, 'ops') and sg.ops:
            return sg.ops
        elif hasattr(sg, 'node_names') and sg.node_names:
            return sg.node_names
        elif hasattr(sg, 'operation_types') and sg.operation_types:
            return [str(op) for op in sg.operation_types]
        return None

    def _get_subgraph_color(self, sg: SubgraphDescriptor, color_by: str) -> str:
        """Get color for a subgraph based on color scheme."""
        if color_by == 'bottleneck':
            compute_pct = sg.compute_bound_pct if hasattr(sg, 'compute_bound_pct') else 50.0
            memory_pct = sg.memory_bound_pct if hasattr(sg, 'memory_bound_pct') else 50.0
            return self.color_manager.get_bottleneck_color(compute_pct, memory_pct)

        elif color_by == 'utilization':
            # Would need utilization data - default to medium
            return self.color_manager.UTILIZATION_COLORS['medium']

        elif color_by == 'op_type':
            ops_list = self._get_ops_list(sg)
            if ops_list:
                return self.color_manager.get_op_type_color(ops_list[0])
            return self.color_manager.OP_TYPE_COLORS['default']

        else:
            return '#808080'  # Default medium gray

    def _format_flops(self, flops: float) -> str:
        """Format FLOP count in human-readable form."""
        if flops >= 1e9:
            return f"{flops/1e9:.1f}G FLOPs"
        elif flops >= 1e6:
            return f"{flops/1e6:.1f}M FLOPs"
        elif flops >= 1e3:
            return f"{flops/1e3:.1f}K FLOPs"
        else:
            return f"{int(flops)} FLOPs"

    def _format_memory(self, bytes: float) -> str:
        """Format memory in human-readable form."""
        if bytes >= 1e9:
            return f"{bytes/1e9:.1f} GB"
        elif bytes >= 1e6:
            return f"{bytes/1e6:.1f} MB"
        elif bytes >= 1e3:
            return f"{bytes/1e3:.1f} KB"
        else:
            return f"{int(bytes)} B"
