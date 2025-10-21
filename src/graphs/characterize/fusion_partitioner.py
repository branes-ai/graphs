"""
Fusion-Based Graph Partitioner

This module implements TRUE graph partitioning by fusing multiple operators
into coarse-grained subgraphs suitable for hardware mapping.

Unlike the basic GraphPartitioner (which creates one subgraph per operator),
this partitioner aggregates operators to minimize data movement.
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.fx import GraphModule
from collections import defaultdict

from .graph_structures import (
    OperationType,
    BottleneckType,
    ParallelismDescriptor,
    SubgraphDescriptor
)

from .visualization import (
    detect_terminal_capability,
    get_box_chars,
    get_bottleneck_color,
    colorize,
    create_legend,
    export_to_dot,
    TerminalCapability,
    ANSIColor,
)


@dataclass
class FusedSubgraph:
    """A fused subgraph containing multiple operators"""
    subgraph_id: int
    node_ids: List[str]  # List of FX node IDs in this subgraph
    node_names: List[str]  # Human-readable names
    operation_types: List[OperationType]  # Operation types in order

    # Computed properties
    total_flops: int
    total_macs: int
    total_input_bytes: int  # Only external inputs (not intermediate)
    total_output_bytes: int  # Only external outputs
    internal_bytes: int  # Intermediate tensors that don't leave subgraph
    total_weight_bytes: int

    # Parallelism (of dominant operation or merged)
    parallelism: Optional[ParallelismDescriptor]

    # Fusion info
    fusion_pattern: str  # e.g., "Conv_BN_ReLU", "Conv_BN", "Add_ReLU"
    num_operators: int  # Number of operators fused

    # Dependencies
    depends_on: List[int]  # Other subgraph IDs this depends on

    # Characterization
    arithmetic_intensity: float  # FLOPs / external_bytes_transferred
    recommended_bottleneck: BottleneckType

    def data_movement_reduction(self) -> float:
        """
        Calculate reduction in data movement vs unfused

        Unfused: Each intermediate tensor written/read from global memory
        Fused: Intermediate tensors stay in cache/registers
        """
        if self.num_operators <= 1:
            return 0.0

        # Savings = internal bytes that don't touch global memory
        external_bytes = self.total_input_bytes + self.total_output_bytes + self.total_weight_bytes
        total_without_fusion = external_bytes + self.internal_bytes

        if total_without_fusion == 0:
            return 0.0

        return self.internal_bytes / total_without_fusion


@dataclass
class FusionReport:
    """Report of fusion-based partitioning"""
    fused_subgraphs: List[FusedSubgraph]
    total_subgraphs: int
    original_operators: int

    # Aggregated stats
    total_flops: int
    total_memory_traffic_fused: int  # With fusion
    total_memory_traffic_unfused: int  # Without fusion
    data_movement_reduction: float  # Fraction of memory traffic eliminated

    # Fusion stats
    fusion_patterns: Dict[str, int]  # Pattern → count
    avg_fusion_size: float  # Average ops per subgraph
    max_fusion_size: int

    def summary_stats(self) -> str:
        """Generate summary statistics"""

        lines = [
            "Fusion-Based Partition Summary",
            "=" * 50,
            f"Original operators: {self.original_operators}",
            f"Fused subgraphs: {self.total_subgraphs}",
            f"Reduction: {self.original_operators / max(1, self.total_subgraphs):.1f}× fewer execution units",
            f"",
            f"Fusion Statistics:",
            f"  Average operators per subgraph: {self.avg_fusion_size:.1f}",
            f"  Largest fused subgraph: {self.max_fusion_size} operators",
            f"",
            f"Compute:",
            f"  Total FLOPs: {self.total_flops / 1e9:.2f} G",
            f"",
            f"Memory Traffic:",
            f"  With fusion: {self.total_memory_traffic_fused / 1e6:.2f} MB",
            f"  Without fusion: {self.total_memory_traffic_unfused / 1e6:.2f} MB",
            f"  Reduction: {self.data_movement_reduction * 100:.1f}%",
            f"  Savings: {(self.total_memory_traffic_unfused - self.total_memory_traffic_fused) / 1e6:.2f} MB",
            f"",
            f"Fusion Patterns:",
        ]

        for pattern, count in sorted(self.fusion_patterns.items(), key=lambda x: x[1], reverse=True):
            pct = count / self.total_subgraphs * 100
            lines.append(f"  {pattern}: {count} ({pct:.1f}%)")

        return "\n".join(lines)


class FusionBasedPartitioner:
    """
    True graph partitioner using operator fusion

    Algorithm:
    1. Build dependency graph from FX nodes
    2. Greedily fuse operators along sequential paths
    3. Stop fusion at boundaries (fork, join, resource limits)
    4. Compute statistics for each fused subgraph
    """

    def __init__(self):
        self.next_subgraph_id = 0
        self.fused_subgraphs: List[FusedSubgraph] = []  # Store for visualization
        self.fx_graph_cached: Optional[GraphModule] = None  # Store FX graph for visualization

    def partition(self, fx_graph: GraphModule) -> FusionReport:
        """
        Partition FX graph into fused subgraphs

        Args:
            fx_graph: Traced PyTorch FX graph with shape propagation

        Returns:
            FusionReport with fused subgraphs and statistics
        """

        # Step 1: Extract nodes and build dependency graph
        nodes = self._extract_nodes(fx_graph)
        consumers = self._build_consumer_map(fx_graph, nodes)
        producers = self._build_producer_map(fx_graph, nodes)

        # Step 2: Perform fusion
        fused_subgraphs = self._fuse_operators(fx_graph, nodes, consumers, producers)

        # Step 3: Store results for visualization
        self.fused_subgraphs = fused_subgraphs
        self.fx_graph_cached = fx_graph

        # Step 4: Compute statistics
        report = self._generate_report(fused_subgraphs, len(nodes))

        return report

    def _extract_nodes(self, fx_graph: GraphModule) -> List:
        """Extract call_module and call_function nodes"""
        nodes = []
        for node in fx_graph.graph.nodes:
            if node.op in ['call_module', 'call_function']:
                nodes.append(node)
        return nodes

    def _build_consumer_map(self, fx_graph: GraphModule, nodes: List) -> Dict:
        """Build map of node → consumers (nodes that use this node's output)"""
        consumers = defaultdict(list)

        for node in fx_graph.graph.nodes:
            for input_node in node.all_input_nodes:
                if input_node in nodes:
                    consumers[input_node].append(node)

        return consumers

    def _build_producer_map(self, fx_graph: GraphModule, nodes: List) -> Dict:
        """Build map of node → producers (nodes that produce this node's inputs)"""
        producers = defaultdict(list)

        for node in nodes:
            for input_node in node.all_input_nodes:
                if input_node in nodes:
                    producers[node].append(input_node)

        return producers

    def _fuse_operators(self, fx_graph: GraphModule, nodes: List,
                       consumers: Dict, producers: Dict) -> List[FusedSubgraph]:
        """
        Greedy fusion algorithm

        Start from each unvisited node, fuse sequentially until hitting boundary
        """
        visited = set()
        fused_subgraphs = []

        # Process nodes in topological order
        for start_node in nodes:
            if start_node in visited:
                continue

            # Start new subgraph
            subgraph_nodes = [start_node]
            visited.add(start_node)
            current = start_node

            # Greedily fuse forward
            while True:
                next_node = self._get_fusible_successor(
                    current, consumers, producers, visited, nodes
                )

                if next_node is None:
                    break

                subgraph_nodes.append(next_node)
                visited.add(next_node)
                current = next_node

            # Create fused subgraph descriptor
            fused = self._create_fused_subgraph(
                fx_graph, subgraph_nodes, consumers, producers
            )
            fused_subgraphs.append(fused)

        return fused_subgraphs

    def _get_fusible_successor(self, node, consumers: Dict, producers: Dict,
                               visited: Set, all_nodes: List) -> Optional:
        """
        Find the next node to fuse (if any)

        Returns None if fusion should stop (boundary reached)
        """

        # Get consumers of current node
        node_consumers = [c for c in consumers.get(node, []) if c in all_nodes]

        # Boundary 1: Fork (multiple consumers)
        if len(node_consumers) != 1:
            return None

        next_node = node_consumers[0]

        # Boundary 2: Already visited
        if next_node in visited:
            return None

        # Boundary 3: Join (multiple producers)
        node_producers = [p for p in producers.get(next_node, []) if p in all_nodes]
        if len(node_producers) > 1:
            # Exception: SE block mul can have multiple producers (sigmoid + features)
            # Allow fusion if current node is Sigmoid and next is mul (SE block output)
            current_type = self._get_node_type(node)
            next_type = self._get_node_type(next_node)
            if current_type == 'Sigmoid' and next_type == 'mul':
                # This is an SE block output, allow fusion
                pass
            else:
                return None  # Stop before join

        # Boundary 4: Check if operations are fusible
        if not self._is_fusible(node, next_node):
            return None

        return next_node

    def _is_fusible(self, node1, node2) -> bool:
        """
        Check if two nodes can be fused

        Fusible patterns:
        - Conv2d → BatchNorm2d
        - BatchNorm2d → ReLU/SiLU/Swish
        - Conv2d → ReLU/SiLU/Swish
        - Linear → ReLU/GELU/SiLU
        - Add → ReLU/SiLU (residual)
        - SE blocks: AdaptiveAvgPool2d → Conv2d → SiLU → Conv2d → Sigmoid → mul
        - Any element-wise chain
        """

        type1 = self._get_node_type(node1)
        type2 = self._get_node_type(node2)

        # Define fusible patterns
        fusible_patterns = [
            # Standard conv/bn patterns
            ('Conv2d', 'BatchNorm2d'),
            ('Conv2d', 'ReLU'),
            ('Conv2d', 'ReLU6'),
            ('Conv2d', 'SiLU'),  # EfficientNet
            ('Conv2d', 'Swish'),  # Alternative name for SiLU
            ('BatchNorm2d', 'ReLU'),
            ('BatchNorm2d', 'ReLU6'),
            ('BatchNorm2d', 'SiLU'),  # EfficientNet
            ('BatchNorm2d', 'Swish'),
            ('BatchNorm2d', 'Hardswish'),

            # Linear patterns
            ('Linear', 'BatchNorm1d'),
            ('Linear', 'ReLU'),
            ('Linear', 'GELU'),
            ('Linear', 'SiLU'),
            ('Linear', 'Dropout'),

            # Residual connections
            ('add', 'ReLU'),
            ('add', 'ReLU6'),
            ('add', 'SiLU'),

            # SE block patterns (Squeeze-Excitation)
            ('AdaptiveAvgPool2d', 'Conv2d'),  # SE: pool → fc1
            ('SiLU', 'Conv2d'),  # SE: activation → fc2 (also general pattern)
            ('Sigmoid', 'mul'),  # SE: sigmoid → scale
            ('Conv2d', 'Sigmoid'),  # SE: fc2 → sigmoid

            # Element-wise operations can fuse
            ('ReLU', 'Dropout'),
            ('GELU', 'Dropout'),
            ('SiLU', 'Dropout'),

            # Transformer patterns (ViT, Swin, BERT, etc.)
            # Attention block
            ('LayerNorm', 'MultiheadAttention'),
            ('MultiheadAttention', 'Dropout'),
            ('Dropout', 'add'),  # Post-attention residual

            # Feed-Forward Network (FFN)
            ('LayerNorm', 'Linear'),
            ('Linear', 'GELU'),  # FFN first layer
            ('GELU', 'Linear'),  # FFN second layer (after dropout)
            ('Linear', 'Dropout'),  # After FFN layers

            # Stochastic depth (drop path) for transformers
            ('StochasticDepth', 'add'),
            ('stochastic_depth', 'add'),
        ]

        for pattern in fusible_patterns:
            if (type1, type2) == pattern:
                return True

        return False

    def _get_node_type(self, node) -> str:
        """Get node type as string"""
        if node.op == 'call_module':
            module = node.graph.owning_module.get_submodule(node.target)
            return type(module).__name__
        elif node.op == 'call_function':
            if hasattr(node.target, '__name__'):
                return node.target.__name__
            else:
                return str(node.target).split('.')[-1]
        else:
            return node.op

    def _create_fused_subgraph(self, fx_graph: GraphModule, nodes: List,
                               consumers: Dict, producers: Dict) -> FusedSubgraph:
        """
        Create a FusedSubgraph descriptor from a list of nodes

        Computes FLOPs, memory traffic, and fusion statistics
        """

        subgraph_id = self.next_subgraph_id
        self.next_subgraph_id += 1

        # Collect basic info
        node_ids = [n.name for n in nodes]
        node_names = [n.name for n in nodes]
        operation_types = [self._classify_operation(fx_graph, n) for n in nodes]

        # Compute FLOPs and memory for each node
        total_flops = 0
        total_macs = 0
        total_weight_bytes = 0

        node_flops = {}
        node_memory = {}

        for node in nodes:
            flops, macs = self._compute_flops(fx_graph, node)
            memory = self._compute_memory(fx_graph, node)

            total_flops += flops
            total_macs += macs
            total_weight_bytes += memory['weights']

            node_flops[node] = flops
            node_memory[node] = memory

        # Determine external vs internal data transfers
        external_inputs = []
        external_outputs = []
        internal_tensors = []

        node_set = set(nodes)

        for node in nodes:
            # Check inputs
            for input_node in node.all_input_nodes:
                if input_node not in node_set:
                    # External input (comes from outside subgraph)
                    external_inputs.append(input_node)
                else:
                    # Internal (produced within subgraph)
                    internal_tensors.append(input_node)

            # Check outputs
            node_consumers = consumers.get(node, [])
            has_external_consumer = any(c not in node_set for c in node_consumers)

            if has_external_consumer or len(node_consumers) == 0:
                # External output (goes outside subgraph)
                external_outputs.append(node)

        # Calculate bytes transferred
        total_input_bytes = sum(self._get_tensor_size(n) for n in external_inputs)
        total_output_bytes = sum(self._get_tensor_size(n) for n in external_outputs)
        internal_bytes = sum(self._get_tensor_size(n) for n in internal_tensors)

        # External bytes = what actually crosses subgraph boundary
        external_bytes = total_input_bytes + total_output_bytes + total_weight_bytes

        # Arithmetic intensity (based on external transfers only)
        arithmetic_intensity = total_flops / max(1, external_bytes)

        # Classify bottleneck
        if arithmetic_intensity > 50:
            bottleneck = BottleneckType.COMPUTE_BOUND
        elif arithmetic_intensity > 10:
            bottleneck = BottleneckType.BALANCED
        elif arithmetic_intensity > 1:
            bottleneck = BottleneckType.MEMORY_BOUND
        else:
            bottleneck = BottleneckType.BANDWIDTH_BOUND

        # Determine fusion pattern
        fusion_pattern = self._identify_fusion_pattern(nodes)

        # Get parallelism from dominant operation (usually first conv/linear)
        parallelism = self._compute_parallelism(fx_graph, nodes[0])

        # Dependencies (subgraphs that must execute before this one)
        # For now, we'll compute this later
        depends_on = []

        return FusedSubgraph(
            subgraph_id=subgraph_id,
            node_ids=node_ids,
            node_names=node_names,
            operation_types=operation_types,
            total_flops=total_flops,
            total_macs=total_macs,
            total_input_bytes=total_input_bytes,
            total_output_bytes=total_output_bytes,
            internal_bytes=internal_bytes,
            total_weight_bytes=total_weight_bytes,
            parallelism=parallelism,
            fusion_pattern=fusion_pattern,
            num_operators=len(nodes),
            depends_on=depends_on,
            arithmetic_intensity=arithmetic_intensity,
            recommended_bottleneck=bottleneck
        )

    def _identify_fusion_pattern(self, nodes: List) -> str:
        """Identify the fusion pattern from node types"""
        if len(nodes) == 1:
            return "Unfused"

        types = [self._get_node_type(n) for n in nodes]
        pattern = "_".join(types[:3])  # First 3 ops

        if len(types) > 3:
            pattern += f"_+{len(types) - 3}more"

        return pattern

    def _classify_operation(self, fx_graph: GraphModule, node) -> OperationType:
        """Classify operation type (reuse from GraphPartitioner)"""
        if node.op == 'call_module':
            module = fx_graph.get_submodule(node.target)
            module_type = type(module).__name__

            if module_type == 'Conv2d':
                # Check if depthwise or pointwise
                if hasattr(module, 'groups'):
                    if module.groups == module.in_channels and module.groups > 1:
                        return OperationType.CONV2D_DEPTHWISE
                    elif module.kernel_size == (1, 1):
                        return OperationType.CONV2D_POINTWISE
                return OperationType.CONV2D
            elif module_type == 'Linear':
                return OperationType.LINEAR
            elif module_type in ['BatchNorm2d', 'BatchNorm1d']:
                return OperationType.BATCHNORM
            elif module_type == 'ReLU':
                return OperationType.RELU
            elif module_type == 'ReLU6':
                return OperationType.RELU6
            elif module_type in ['SiLU', 'Swish']:
                return OperationType.SILU
            elif module_type == 'GELU':
                return OperationType.GELU
            elif module_type == 'Hardswish':
                return OperationType.HARDSWISH
            elif module_type == 'Sigmoid':
                return OperationType.SIGMOID
            elif module_type in ['MaxPool2d', 'MaxPool1d']:
                return OperationType.MAXPOOL
            elif module_type in ['AvgPool2d', 'AvgPool1d']:
                return OperationType.AVGPOOL
            elif module_type in ['AdaptiveAvgPool2d', 'AdaptiveAvgPool1d']:
                return OperationType.ADAPTIVEAVGPOOL
            elif module_type == 'LayerNorm':
                return OperationType.LAYERNORM
            elif module_type == 'MultiheadAttention':
                return OperationType.MULTIHEAD_ATTENTION
            elif module_type == 'Dropout':
                return OperationType.DROPOUT
            elif module_type in ['StochasticDepth', 'stochastic_depth']:
                return OperationType.STOCHASTIC_DEPTH

        return OperationType.UNKNOWN

    def _compute_flops(self, fx_graph: GraphModule, node) -> Tuple[int, int]:
        """Compute FLOPs and MACs for a node (simplified)"""
        # This is a simplified version - reuse GraphPartitioner logic for full version

        if node.op != 'call_module':
            return 0, 0

        module = fx_graph.get_submodule(node.target)

        if not hasattr(node, 'meta') or 'tensor_meta' not in node.meta:
            return 0, 0

        meta = node.meta['tensor_meta']

        if isinstance(module, torch.nn.Conv2d):
            batch = meta.shape[0]
            out_channels = meta.shape[1]
            out_h, out_w = meta.shape[2], meta.shape[3]
            in_channels = module.in_channels
            k_h, k_w = module.kernel_size

            if module.groups == in_channels and module.groups > 1:
                # Depthwise
                macs = batch * in_channels * out_h * out_w * k_h * k_w
            else:
                # Standard or pointwise
                macs = batch * out_channels * out_h * out_w * in_channels * k_h * k_w

            flops = 2 * macs
            return flops, macs

        elif isinstance(module, torch.nn.Linear):
            batch = meta.shape[0] if len(meta.shape) > 1 else 1
            out_features = module.out_features
            in_features = module.in_features

            macs = batch * out_features * in_features
            flops = 2 * macs
            return flops, macs

        return 0, 0

    def _compute_memory(self, fx_graph: GraphModule, node) -> Dict[str, int]:
        """Compute memory usage for a node"""
        if node.op != 'call_module':
            return {'weights': 0, 'input': 0, 'output': 0}

        module = fx_graph.get_submodule(node.target)

        # Count parameters
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())

        return {'weights': param_bytes, 'input': 0, 'output': 0}

    def _get_tensor_size(self, node) -> int:
        """Get tensor size in bytes"""
        if not hasattr(node, 'meta') or 'tensor_meta' not in node.meta:
            return 0

        meta = node.meta['tensor_meta']
        if hasattr(meta, 'shape'):
            numel = 1
            for dim in meta.shape:
                numel *= dim
            return numel * 4  # Assume float32

        return 0

    def _compute_parallelism(self, fx_graph: GraphModule, node) -> Optional[ParallelismDescriptor]:
        """Compute parallelism for a node"""
        if not hasattr(node, 'meta') or 'tensor_meta' not in node.meta:
            return None

        meta = node.meta['tensor_meta']
        if not hasattr(meta, 'shape') or len(meta.shape) < 2:
            return None

        # Simple version: batch * channels * spatial
        if len(meta.shape) == 4:  # Conv output
            batch = meta.shape[0]
            channels = meta.shape[1]
            spatial = meta.shape[2] * meta.shape[3]
            total_threads = batch * channels * spatial

            return ParallelismDescriptor(
                batch=batch,
                channels=channels,
                spatial=spatial,
                total_threads=total_threads,
                is_depthwise=False,
                is_grouped=False,
                can_split_batch=True,
                can_split_spatial=True,
                can_split_channels=True
            )

        return None

    def _generate_report(self, fused_subgraphs: List[FusedSubgraph],
                        original_operators: int) -> FusionReport:
        """Generate fusion report with statistics"""

        total_flops = sum(sg.total_flops for sg in fused_subgraphs)

        # Memory traffic with fusion
        total_memory_fused = sum(
            sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
            for sg in fused_subgraphs
        )

        # Memory traffic without fusion (all intermediate tensors touch memory)
        total_memory_unfused = total_memory_fused + sum(
            sg.internal_bytes for sg in fused_subgraphs
        )

        # Data movement reduction
        if total_memory_unfused > 0:
            data_movement_reduction = (total_memory_unfused - total_memory_fused) / total_memory_unfused
        else:
            data_movement_reduction = 0.0

        # Fusion pattern counts
        fusion_patterns = {}
        for sg in fused_subgraphs:
            fusion_patterns[sg.fusion_pattern] = fusion_patterns.get(sg.fusion_pattern, 0) + 1

        # Fusion size stats
        fusion_sizes = [sg.num_operators for sg in fused_subgraphs]
        avg_fusion_size = sum(fusion_sizes) / len(fusion_sizes) if fusion_sizes else 1
        max_fusion_size = max(fusion_sizes) if fusion_sizes else 1

        return FusionReport(
            fused_subgraphs=fused_subgraphs,
            total_subgraphs=len(fused_subgraphs),
            original_operators=original_operators,
            total_flops=total_flops,
            total_memory_traffic_fused=total_memory_fused,
            total_memory_traffic_unfused=total_memory_unfused,
            data_movement_reduction=data_movement_reduction,
            fusion_patterns=fusion_patterns,
            avg_fusion_size=avg_fusion_size,
            max_fusion_size=max_fusion_size
        )

    def visualize_partitioning(self, fx_graph: GraphModule, max_nodes: Optional[int] = None) -> str:
        """
        Create side-by-side visualization of FX graph and fused subgraphs.

        Shows original graph on left, fused subgraphs on right with visual grouping.

        Args:
            fx_graph: The FX graph that was partitioned
            max_nodes: Maximum number of nodes to show (None for all)

        Returns:
            String containing the formatted visualization
        """
        import torch.nn as nn

        # Build mapping from node_id to fused subgraph
        node_to_fused_subgraph = {}
        node_position_in_subgraph = {}  # Track position: (subgraph, index, total)

        for fused_sg in self.fused_subgraphs:
            for idx, node_id in enumerate(fused_sg.node_ids):
                node_to_fused_subgraph[node_id] = fused_sg
                node_position_in_subgraph[node_id] = (fused_sg, idx, len(fused_sg.node_ids))

        # Collect all nodes in execution order
        all_nodes = list(fx_graph.graph.nodes)

        if max_nodes:
            all_nodes = all_nodes[:max_nodes]

        # Build visualization
        lines = []

        # Header
        left_width = 50
        right_width = 60
        total_width = left_width + 4 + right_width

        lines.append("=" * total_width)
        lines.append("FUSION-BASED GRAPH PARTITIONING VISUALIZATION")
        lines.append("=" * total_width)
        lines.append("")

        # Column headers
        header_left = "FX Graph (Execution Order)".ljust(left_width)
        header_right = "Fused Subgraphs"
        lines.append(f"{header_left}    {header_right}")
        lines.append("-" * left_width + "    " + "-" * right_width)
        lines.append("")

        # Process each node
        subgraph_counter = 1
        current_subgraph_id = None

        for idx, node in enumerate(all_nodes, 1):
            node_id = node.name  # Use node name, not id()

            # LEFT SIDE: FX Node info
            left_lines = self._format_fx_node(node, fx_graph, idx)

            # RIGHT SIDE: Fused subgraph info with grouping
            right_lines = []

            if node_id in node_to_fused_subgraph:
                fused_sg, node_idx, total_nodes = node_position_in_subgraph[node_id]
                is_first = (node_idx == 0)
                is_last = (node_idx == total_nodes - 1)

                # New subgraph starting
                if fused_sg.subgraph_id != current_subgraph_id:
                    current_subgraph_id = fused_sg.subgraph_id

                    if is_first:
                        # Header for new fused subgraph
                        header_lines = self._format_fused_subgraph_header(fused_sg, subgraph_counter)
                        right_lines.extend(header_lines)
                        subgraph_counter += 1

                # Operator within fused subgraph
                op_lines = self._format_fused_operator(node, fx_graph, is_first, is_last)
                right_lines.extend(op_lines)

                # Footer for completed subgraph
                if is_last:
                    footer_lines = self._format_fused_subgraph_footer(fused_sg)
                    right_lines.extend(footer_lines)
                    current_subgraph_id = None
            else:
                # Node not in any fused subgraph
                right_lines = self._format_not_fused(node)

            # Combine left and right, ensuring same number of lines
            max_lines = max(len(left_lines), len(right_lines))
            for i in range(max_lines):
                left = left_lines[i] if i < len(left_lines) else ""
                right = right_lines[i] if i < len(right_lines) else ""
                lines.append(f"{left.ljust(left_width)}    {right}")

            # Add spacing between nodes
            lines.append("")

        # Footer
        if max_nodes and len(fx_graph.graph.nodes) > max_nodes:
            lines.append(f"... ({len(fx_graph.graph.nodes) - max_nodes} more nodes not shown)")
            lines.append("")

        lines.append("=" * total_width)
        lines.append(f"Total FX nodes: {len(fx_graph.graph.nodes)}")
        lines.append(f"Fused subgraphs: {len(self.fused_subgraphs)}")
        lines.append(f"Reduction: {len(fx_graph.graph.nodes) / max(1, len(self.fused_subgraphs)):.1f}× fewer execution units")

        if self.fused_subgraphs:
            avg_fusion = sum(sg.num_operators for sg in self.fused_subgraphs) / len(self.fused_subgraphs)
            lines.append(f"Average fusion size: {avg_fusion:.1f} operators/subgraph")

        lines.append("=" * total_width)

        return "\n".join(lines)

    def _format_fx_node(self, node, graph: GraphModule, idx: int) -> List[str]:
        """Format FX node information for left column"""
        import torch.nn as nn

        lines = []

        # Node header
        lines.append(f"{idx}. [{node.op}] {node.name}")

        # Add details based on node type
        if node.op == 'call_module':
            try:
                module = graph.get_submodule(node.target)
                module_type = type(module).__name__

                # Add module-specific details
                if isinstance(module, nn.Conv2d):
                    details = f"   Conv2d({module.in_channels}->{module.out_channels}, "
                    details += f"k={module.kernel_size}, s={module.stride})"
                    lines.append(details)
                elif isinstance(module, nn.Linear):
                    details = f"   Linear({module.in_features}->{module.out_features})"
                    lines.append(details)
                elif isinstance(module, nn.BatchNorm2d):
                    details = f"   BatchNorm2d({module.num_features})"
                    lines.append(details)
                elif isinstance(module, (nn.ReLU, nn.ReLU6)):
                    lines.append(f"   {module_type}")
                elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                    lines.append(f"   {module_type}")
                else:
                    lines.append(f"   {module_type}")
            except:
                lines.append(f"   {node.target}")

        elif node.op == 'call_function':
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            lines.append(f"   Function: {func_name}")

        elif node.op == 'call_method':
            lines.append(f"   Method: {node.target}")

        return lines

    def _format_fused_subgraph_header(self, fused_sg: FusedSubgraph, counter: int) -> List[str]:
        """Format header for a fused subgraph"""
        lines = []

        # Top border with subgraph info
        lines.append(f"┌─ SUBGRAPH #{counter} ─────────────────────────")
        lines.append(f"│  Pattern: {fused_sg.fusion_pattern}")
        lines.append(f"│  Operators: {fused_sg.num_operators}")
        lines.append(f"│")

        return lines

    def _format_fused_operator(self, node, graph: GraphModule, is_first: bool, is_last: bool) -> List[str]:
        """Format a single operator within a fused subgraph"""
        import torch.nn as nn

        lines = []

        # Get operator type
        op_type = "unknown"
        if node.op == 'call_module':
            try:
                module = graph.get_submodule(node.target)
                op_type = type(module).__name__
            except:
                op_type = str(node.target)
        elif node.op == 'call_function':
            op_type = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        elif node.op == 'call_method':
            op_type = node.target

        # Format operator line
        lines.append(f"│  • {node.name} ({op_type})")

        return lines

    def _format_fused_subgraph_footer(self, fused_sg: FusedSubgraph) -> List[str]:
        """Format footer with metrics for a fused subgraph"""
        lines = []

        lines.append(f"│")

        # Compute metrics
        flops_str = self._format_number(fused_sg.total_flops, 'FLOPs')
        macs_str = self._format_number(fused_sg.total_macs, 'MACs')
        lines.append(f"│  Compute: {macs_str}, {flops_str}")

        # Memory metrics
        external_bytes = fused_sg.total_input_bytes + fused_sg.total_output_bytes + fused_sg.total_weight_bytes
        external_str = self._format_bytes(external_bytes)
        lines.append(f"│  Memory: {external_str} (external)")

        if fused_sg.internal_bytes > 0:
            internal_str = self._format_bytes(fused_sg.internal_bytes)
            reduction_pct = fused_sg.data_movement_reduction() * 100
            lines.append(f"│  Saved: {internal_str} internal ({reduction_pct:.1f}% reduction)")

        # Arithmetic Intensity
        ai_str = f"{fused_sg.arithmetic_intensity:.1f}"
        bottleneck = fused_sg.recommended_bottleneck.value.upper()
        lines.append(f"│  AI: {ai_str} FLOPs/byte [{bottleneck}]")

        # Bottom border
        lines.append(f"└─────────────────────────────────────────────")

        return lines

    def _format_not_fused(self, node) -> List[str]:
        """Format info for nodes that weren't fused"""
        lines = []
        lines.append("   (not fused)")

        if node.op == 'placeholder':
            lines.append("   Reason: input placeholder")
        elif node.op == 'get_attr':
            lines.append("   Reason: attribute access")
        elif node.op == 'output':
            lines.append("   Reason: output node")
        else:
            lines.append("   Reason: not included in fusion")

        return lines

    def _format_number(self, num: int, suffix: str = '') -> str:
        """Format large numbers with K/M/G suffix"""
        if num == 0:
            return f"0 {suffix}"

        if num >= 1e9:
            return f"{num / 1e9:.2f}G{suffix}"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M{suffix}"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K{suffix}"
        else:
            return f"{num} {suffix}"

    def _format_bytes(self, bytes: int) -> str:
        """Format memory in bytes with KB/MB/GB suffix"""
        if bytes == 0:
            return "0 B"

        if bytes >= 1e9:
            return f"{bytes / 1e9:.2f}GB"
        elif bytes >= 1e6:
            return f"{bytes / 1e6:.2f}MB"
        elif bytes >= 1e3:
            return f"{bytes / 1e3:.2f}KB"
        else:
            return f"{bytes}B"

    def _categorize_single_ops(self) -> Dict[str, List]:
        """
        Categorize single-op subgraphs into structural vs potentially fusible.

        Structural operations (expected to be unfused):
        - Control flow: getitem, getattr, eq, _assert
        - Shape manipulation: reshape, permute, transpose, view
        - Placeholders: input, output, get_attr
        - Methods: method calls that don't do computation

        Returns:
            Dict with 'structural' and 'fusible' lists of subgraphs
        """
        structural_ops = {
            # Control flow
            'getitem', 'getattr', 'eq', '_assert', 'floordiv',
            # Shape manipulation
            'reshape', 'permute', 'transpose', 'view', 'flatten', 'squeeze', 'unsqueeze',
            'expand', 'contiguous', 'select', 'slice',
            # Tensor creation/manipulation
            'cat', 'stack', 'split', 'chunk',
            # Size/shape queries
            'size', 'dim', 'numel',
            # Other
            'clone', 'detach', 'to',
        }

        single_op_subgraphs = [sg for sg in self.fused_subgraphs if sg.num_operators == 1]

        structural = []
        fusible = []

        for sg in single_op_subgraphs:
            # Check if it's a structural operation
            pattern_lower = sg.fusion_pattern.lower()

            is_structural = False

            # Check for placeholder/output
            if 'placeholder' in pattern_lower or 'output' in pattern_lower:
                is_structural = True
            # Check for get_attr
            elif 'get_attr' in pattern_lower or 'getattr' in pattern_lower:
                is_structural = True
            # Check for method calls (reshape, permute, etc.)
            elif any(op in pattern_lower for op in structural_ops):
                is_structural = True
            # Check operation types
            elif sg.operation_types and sg.operation_types[0].value == 'unknown':
                is_structural = True

            if is_structural:
                structural.append(sg)
            else:
                fusible.append(sg)

        return {'structural': structural, 'fusible': fusible}

    def _detect_fusion_opportunities(self) -> List[Dict]:
        """
        Detect missed fusion opportunities by analyzing adjacent unfused operations.

        Returns:
            List of missed opportunities with details
        """
        opportunities = []

        # Build node-to-subgraph mapping
        node_to_subgraph = {}
        for sg in self.fused_subgraphs:
            for node_id in sg.node_ids:
                node_to_subgraph[node_id] = sg

        # Analyze graph for sequential operations that could fuse
        for node in self.fx_graph_cached.graph.nodes:
            if node.name not in node_to_subgraph:
                continue

            current_sg = node_to_subgraph[node.name]

            # Only look at single-op subgraphs (potential opportunities)
            if current_sg.num_operators != 1:
                continue

            # Check consumers
            for consumer in node.users:
                if consumer.name not in node_to_subgraph:
                    continue

                consumer_sg = node_to_subgraph[consumer.name]

                # If both are single-op and adjacent, check if they could fuse
                if consumer_sg.num_operators == 1:
                    current_type = self._get_node_type(node)
                    consumer_type = self._get_node_type(consumer)

                    # Check if this pattern is fusible
                    if self._is_fusible(node, consumer):
                        opportunities.append({
                            'op1': current_type,
                            'op2': consumer_type,
                            'pattern': f"{current_type} → {consumer_type}",
                            'subgraph1_id': current_sg.subgraph_id,
                            'subgraph2_id': consumer_sg.subgraph_id,
                            'reason': 'Adjacent fusible operations not fused'
                        })

        # Deduplicate opportunities
        seen_patterns = set()
        unique_opportunities = []
        for opp in opportunities:
            if opp['pattern'] not in seen_patterns:
                seen_patterns.add(opp['pattern'])
                unique_opportunities.append(opp)

        return unique_opportunities

    def _calculate_sequential_fusion_baseline(self) -> Dict[str, int]:
        """
        Calculate baseline fusion assuming simple sequential-only strategy.

        This provides a conservative baseline that only fuses operations in
        simple sequential chains (single producer/consumer). It breaks at:
        - Join points (multiple producers)
        - Fork points (multiple consumers)

        The actual fusion partitioner can do better by fusing through join
        points (e.g., add, concat) and using more sophisticated fusion rules.

        Returns:
            Dict with baseline fusion metrics
        """
        if not self.fx_graph_cached:
            return {'max_fusible_chains': 0, 'max_fusion_efficiency': 1.0}

        # Build dependency graph
        compute_nodes = [n for n in self.fx_graph_cached.graph.nodes
                        if n.op in ['call_module', 'call_function']]

        # Find longest sequential chains
        from collections import defaultdict
        producers = defaultdict(list)
        consumers = defaultdict(list)

        for node in compute_nodes:
            for input_node in node.all_input_nodes:
                if input_node in compute_nodes:
                    producers[node].append(input_node)
                    consumers[input_node].append(node)

        # Count chains (sequences with single producer/consumer)
        chains = []
        visited = set()

        def follow_chain(start_node):
            chain = [start_node]
            current = start_node
            visited.add(current)

            # Follow forward while single consumer
            while len(consumers.get(current, [])) == 1:
                next_node = consumers[current][0]
                if next_node in visited:
                    break
                if len(producers.get(next_node, [])) > 1:
                    break  # Join point
                chain.append(next_node)
                visited.add(next_node)
                current = next_node

            return chain

        for node in compute_nodes:
            if node not in visited:
                # Start chain from nodes with no producer or multiple producers
                if len(producers.get(node, [])) != 1:
                    chain = follow_chain(node)
                    if len(chain) > 1:
                        chains.append(chain)

        # Calculate baseline (simple sequential fusion)
        total_ops = len(compute_nodes)
        num_chains = len(chains)
        single_ops = total_ops - sum(len(c) for c in chains)
        baseline_subgraphs = num_chains + single_ops

        baseline_efficiency = total_ops / max(1, baseline_subgraphs)

        return {
            'total_compute_ops': total_ops,
            'sequential_chains': num_chains,
            'longest_chain': max(len(c) for c in chains) if chains else 1,
            'baseline_subgraphs': baseline_subgraphs,
            'baseline_efficiency': baseline_efficiency,
            'single_ops_in_baseline': single_ops,
        }

    def analyze_balance(self) -> str:
        """
        Analyze balance and quality of fusion partitioning.

        Provides insights into:
        - Distribution of fusion sizes (histogram)
        - Categorized single-op analysis (structural vs fusible)
        - Detection of overly large fusions (potential issues)
        - Top fusion patterns
        - Bottleneck distribution
        - Missed fusion opportunity detection
        - Comparison to sequential fusion baseline

        Returns:
            Formatted report string with analysis and recommendations
        """
        if not self.fused_subgraphs:
            return "No fused subgraphs to analyze"

        if not self.fx_graph_cached:
            return "FX graph not cached - cannot perform detailed analysis"

        lines = []
        lines.append("=" * 100)
        lines.append("FUSION BALANCE ANALYSIS")
        lines.append("=" * 100)
        lines.append("")

        # Basic statistics
        sizes = [sg.num_operators for sg in self.fused_subgraphs]
        total_operators = sum(sizes)

        lines.append(f"Total Fused Subgraphs: {len(self.fused_subgraphs)}")
        lines.append(f"Total Operators: {total_operators}")
        lines.append(f"Fusion Efficiency: {total_operators / len(self.fused_subgraphs):.2f}× "
                    f"({len(self.fused_subgraphs)} execution units vs {total_operators} original ops)")
        lines.append("")

        # Fusion size distribution
        lines.append("─" * 100)
        lines.append("FUSION SIZE DISTRIBUTION")
        lines.append("─" * 100)
        lines.append(f"  Min:    {min(sizes)} operator{'s' if min(sizes) != 1 else ''}")
        lines.append(f"  Max:    {max(sizes)} operators")
        lines.append(f"  Mean:   {sum(sizes) / len(sizes):.2f} operators")

        sorted_sizes = sorted(sizes)
        median = sorted_sizes[len(sorted_sizes)//2]
        lines.append(f"  Median: {median} operator{'s' if median != 1 else ''}")
        lines.append("")

        # Histogram
        from collections import Counter
        histogram = Counter(sizes)

        lines.append("Fusion Size Histogram:")
        max_count = max(histogram.values())
        bar_scale = min(50, max_count)  # Limit bar width to 50 chars

        for size in sorted(histogram.keys()):
            count = histogram[size]
            pct = count / len(self.fused_subgraphs) * 100
            bar_width = int(count / max_count * bar_scale)
            bar = "█" * bar_width
            lines.append(f"  {size:3d} op{'s' if size != 1 else ' '}: {bar:<50} {count:4d} ({pct:5.1f}%)")
        lines.append("")

        # Identify potential issues
        lines.append("─" * 100)
        lines.append("FUSION QUALITY ANALYSIS")
        lines.append("─" * 100)

        # Single-operator subgraphs (categorized analysis)
        categorized = self._categorize_single_ops()
        structural_ops = categorized['structural']
        fusible_ops = categorized['fusible']
        total_single_op = len(structural_ops) + len(fusible_ops)

        if total_single_op > 0:
            pct = total_single_op / len(self.fused_subgraphs) * 100
            lines.append(f"Single-Operator Subgraphs: {total_single_op} ({pct:.1f}%)")
            lines.append("")

            # Structural operations (expected)
            if structural_ops:
                struct_pct = len(structural_ops) / len(self.fused_subgraphs) * 100
                lines.append(f"  ✓  Structural Operations: {len(structural_ops)} ({struct_pct:.1f}%)")
                lines.append("      Expected unfused (control flow, shape manipulation, placeholders)")
                pattern_counts = Counter(sg.fusion_pattern for sg in structural_ops)
                lines.append("      Top patterns:")
                for pattern, count in pattern_counts.most_common(5):
                    lines.append(f"        • {pattern}: {count}")
                lines.append("")

            # Potentially fusible operations (opportunities)
            if fusible_ops:
                fusible_pct = len(fusible_ops) / len(self.fused_subgraphs) * 100
                lines.append(f"  ⚠️  Potentially Fusible Operations: {len(fusible_ops)} ({fusible_pct:.1f}%)")
                lines.append("      These might be fusion opportunities")
                pattern_counts = Counter(sg.fusion_pattern for sg in fusible_ops)
                lines.append("      Top patterns:")
                for pattern, count in pattern_counts.most_common(5):
                    # Find example for this pattern
                    example = next(sg for sg in fusible_ops if sg.fusion_pattern == pattern)
                    op_type = example.operation_types[0].value if example.operation_types else "unknown"
                    lines.append(f"        • {pattern}: {count} (type: {op_type})")
                lines.append("")
            else:
                lines.append("  ✓  No potentially fusible single-ops (excellent!)")
                lines.append("")
        else:
            lines.append("✓  No single-operator subgraphs (perfect fusion coverage)")
            lines.append("")

        # Large fusions (>10 operators)
        large_fusions = [sg for sg in self.fused_subgraphs if sg.num_operators > 10]
        if large_fusions:
            lines.append(f"⚠️  Large Fusions (>10 ops): {len(large_fusions)}")
            lines.append("    → May indicate overly aggressive fusion")
            lines.append("    → Could cause register pressure or reduced parallelism")
            lines.append("")

            # Show details for largest
            sorted_large = sorted(large_fusions, key=lambda x: x.num_operators, reverse=True)
            lines.append("    Largest fusions:")
            for sg in sorted_large[:5]:
                lines.append(f"      • Subgraph #{sg.subgraph_id}: {sg.num_operators} ops "
                           f"({sg.fusion_pattern[:40]}...)" if len(sg.fusion_pattern) > 40
                           else f"({sg.fusion_pattern})")
            lines.append("")
        else:
            lines.append("✓  No excessively large fusions (well-balanced)")
            lines.append("")

        # Very large fusions (>20 operators) - critical warning
        very_large = [sg for sg in self.fused_subgraphs if sg.num_operators > 20]
        if very_large:
            lines.append(f"🔴 CRITICAL: Very Large Fusions (>20 ops): {len(very_large)}")
            lines.append("    → High risk of performance degradation")
            lines.append("    → Consider revising fusion strategy")
            lines.append("")

        # Fusion pattern analysis
        lines.append("─" * 100)
        lines.append("TOP FUSION PATTERNS")
        lines.append("─" * 100)

        pattern_counts = Counter(sg.fusion_pattern for sg in self.fused_subgraphs)
        pattern_stats = []
        for pattern, count in pattern_counts.items():
            # Get average metrics for this pattern
            pattern_sgs = [sg for sg in self.fused_subgraphs if sg.fusion_pattern == pattern]
            avg_ops = sum(sg.num_operators for sg in pattern_sgs) / len(pattern_sgs)
            avg_ai = sum(sg.arithmetic_intensity for sg in pattern_sgs) / len(pattern_sgs)
            avg_reduction = sum(sg.data_movement_reduction() for sg in pattern_sgs) / len(pattern_sgs) * 100

            pattern_stats.append((pattern, count, avg_ops, avg_ai, avg_reduction))

        # Sort by count
        pattern_stats.sort(key=lambda x: x[1], reverse=True)

        lines.append(f"{'Pattern':<40} {'Count':>8} {'Avg Ops':>10} {'Avg AI':>10} {'Mem Save':>10}")
        lines.append("-" * 100)

        for pattern, count, avg_ops, avg_ai, avg_reduction in pattern_stats[:15]:
            pct = count / len(self.fused_subgraphs) * 100
            pattern_display = pattern[:37] + "..." if len(pattern) > 40 else pattern
            lines.append(f"{pattern_display:<40} {count:>4} ({pct:4.1f}%) {avg_ops:>8.1f} "
                        f"{avg_ai:>10.1f} {avg_reduction:>9.1f}%")

        if len(pattern_stats) > 15:
            lines.append(f"... and {len(pattern_stats) - 15} more patterns")
        lines.append("")

        # Bottleneck distribution
        lines.append("─" * 100)
        lines.append("BOTTLENECK DISTRIBUTION")
        lines.append("─" * 100)

        bottleneck_counts = Counter(sg.recommended_bottleneck.value for sg in self.fused_subgraphs)

        for bottleneck, count in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(self.fused_subgraphs) * 100
            bar_width = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_width
            lines.append(f"  {bottleneck.upper():<20} {bar:<50} {count:4d} ({pct:5.1f}%)")
        lines.append("")

        # Data movement savings summary
        lines.append("─" * 100)
        lines.append("DATA MOVEMENT SAVINGS")
        lines.append("─" * 100)

        total_external = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                            for sg in self.fused_subgraphs)
        total_internal = sum(sg.internal_bytes for sg in self.fused_subgraphs)
        total_unfused = total_external + total_internal

        if total_unfused > 0:
            overall_reduction = total_internal / total_unfused * 100
        else:
            overall_reduction = 0.0

        lines.append(f"  Memory Traffic (with fusion):    {self._format_bytes(total_external)}")
        lines.append(f"  Memory Traffic (without fusion): {self._format_bytes(total_unfused)}")
        lines.append(f"  Internal Bytes Saved:            {self._format_bytes(total_internal)}")
        lines.append(f"  Overall Reduction:               {overall_reduction:.1f}%")
        lines.append("")

        # Fusion opportunity detection
        lines.append("─" * 100)
        lines.append("MISSED FUSION OPPORTUNITIES")
        lines.append("─" * 100)

        opportunities = self._detect_fusion_opportunities()
        if opportunities:
            lines.append(f"  Found {len(opportunities)} potential fusion pattern(s) not currently fused:")
            lines.append("")
            for i, opp in enumerate(opportunities[:10], 1):
                lines.append(f"  {i}. {opp['pattern']}")
                lines.append(f"     Reason: {opp['reason']}")
            if len(opportunities) > 10:
                lines.append(f"  ... and {len(opportunities) - 10} more")
            lines.append("")
            lines.append("  These patterns are fusible but currently execute as separate subgraphs.")
            lines.append("  This may indicate:")
            lines.append("    • Greedy fusion stopped early (hit a boundary)")
            lines.append("    • Join/fork points preventing fusion")
            lines.append("    • Operations in different execution paths")
        else:
            lines.append("  ✓ No obvious missed fusion opportunities detected")
            lines.append("  All adjacent fusible operations are properly fused")
        lines.append("")

        # Comparison to sequential fusion baseline
        lines.append("─" * 100)
        lines.append("FUSION STRATEGY COMPARISON")
        lines.append("─" * 100)
        lines.append("")
        lines.append("Comparing actual fusion to a simple sequential-only baseline:")
        lines.append("(Baseline only fuses simple chains, breaks at join/fork points)")
        lines.append("")

        baseline = self._calculate_sequential_fusion_baseline()
        actual_efficiency = sum(sizes) / len(self.fused_subgraphs)

        lines.append(f"  Total Compute Operations:        {baseline['total_compute_ops']}")
        lines.append(f"  Sequential Chains Found:         {baseline['sequential_chains']}")
        lines.append(f"  Longest Sequential Chain:        {baseline['longest_chain']} ops")
        lines.append("")
        lines.append(f"  Baseline (Sequential Only):      {baseline['baseline_subgraphs']} subgraphs, "
                    f"{baseline['baseline_efficiency']:.2f}× efficiency")
        lines.append(f"  Actual (Smart Fusion):           {len(self.fused_subgraphs)} subgraphs, "
                    f"{actual_efficiency:.2f}× efficiency")
        lines.append("")

        # Calculate improvement over baseline
        if len(self.fused_subgraphs) <= baseline['baseline_subgraphs']:
            improvement = (baseline['baseline_subgraphs'] - len(self.fused_subgraphs)) / baseline['baseline_subgraphs'] * 100
            lines.append(f"  Improvement Over Baseline:       {improvement:.1f}% fewer subgraphs")
            lines.append(f"                                   ({baseline['baseline_subgraphs'] - len(self.fused_subgraphs)} fewer kernel launches)")
        else:
            regression = (len(self.fused_subgraphs) - baseline['baseline_subgraphs']) / baseline['baseline_subgraphs'] * 100
            lines.append(f"  ⚠️  Regression:                   {regression:.1f}% more subgraphs than baseline")
            lines.append(f"                                   (This suggests the fusion strategy may have issues)")

        lines.append("")

        efficiency_ratio = actual_efficiency / baseline['baseline_efficiency'] if baseline['baseline_efficiency'] > 0 else 1.0
        lines.append(f"  Fusion Efficiency Gain:          {efficiency_ratio:.2f}× vs baseline")
        lines.append("")

        if efficiency_ratio >= 1.5:
            lines.append("  ✓ Excellent! Fusion is significantly better than sequential-only strategy")
            lines.append("    → Successfully fusing through join points and complex patterns")
        elif efficiency_ratio >= 1.2:
            lines.append("  ✓ Good! Fusion improves over sequential-only baseline")
            lines.append("    → Taking advantage of some cross-branch fusion opportunities")
        elif efficiency_ratio >= 1.0:
            lines.append("  ✓ Modest improvement over sequential fusion")
            lines.append("    → May have opportunities to fuse through more join points")
        else:
            lines.append("  ⚠️  Performing worse than sequential baseline")
            lines.append("     → This suggests fusion strategy may be creating unnecessary boundaries")
        lines.append("")

        # Recommendations
        lines.append("─" * 100)
        lines.append("RECOMMENDATIONS")
        lines.append("─" * 100)

        recommendations = []

        # Enhanced recommendations based on categorized analysis
        if len(fusible_ops) > len(self.fused_subgraphs) * 0.15:
            recommendations.append("⚠️  High number of potentially fusible single-ops (>15%)")
            recommendations.append("   → Review fusion heuristics to increase fusion coverage")
            if opportunities:
                recommendations.append(f"   → {len(opportunities)} specific fusion patterns detected (see above)")
        elif len(fusible_ops) > 0:
            recommendations.append(f"ℹ️  {len(fusible_ops)} potentially fusible single-ops detected")
            recommendations.append("   → These may be edge cases or structural constraints")

        if len(large_fusions) > 0:
            recommendations.append("⚠️  Some large fusions detected")
            recommendations.append("   → Monitor for register pressure and reduced parallelism")
            recommendations.append("   → Consider fusion size limits in the strategy")

        if overall_reduction < 20:
            recommendations.append("⚠️  Low data movement reduction (<20%)")
            recommendations.append("   → Fusion may not be providing significant benefit")
            recommendations.append("   → Review operator compatibility and fusion boundaries")
        elif overall_reduction > 50:
            recommendations.append("✓  Excellent data movement reduction (>50%)")
            recommendations.append("   → Fusion strategy is highly effective")

        compute_bound_pct = bottleneck_counts.get('compute_bound', 0) / len(self.fused_subgraphs) * 100
        if compute_bound_pct > 70:
            recommendations.append("✓  Majority compute-bound (good for GPU/TPU)")
        elif compute_bound_pct < 30:
            recommendations.append("⚠️  Majority memory-bound")
            recommendations.append("   → May benefit from increased fusion to improve AI")

        if recommendations:
            for rec in recommendations:
                lines.append(f"  {rec}")
        else:
            lines.append("  ✓ No significant issues detected")
            lines.append("  ✓ Fusion strategy appears well-balanced")

        lines.append("")
        lines.append("=" * 100)

        return "\n".join(lines)

    def visualize_partitioning_colored(self, fx_graph: GraphModule,
                                      max_nodes: Optional[int] = None,
                                      use_color: Optional[bool] = None) -> str:
        """
        Create color-coded visualization of fusion partitioning.

        Color codes subgraphs by bottleneck type:
        - Green: Compute-bound (good for accelerators)
        - Cyan: Balanced
        - Yellow: Memory-bound
        - Red: Bandwidth-bound

        Args:
            fx_graph: The FX graph that was partitioned
            max_nodes: Maximum number of nodes to show (None for all)
            use_color: Force color on/off (None for auto-detect)

        Returns:
            String containing the formatted colored visualization
        """
        import torch.nn as nn

        # Detect terminal capability
        if use_color is None:
            capability = detect_terminal_capability()
        else:
            capability = TerminalCapability.COLOR if use_color else TerminalCapability.BASIC

        box = get_box_chars(capability)

        # Build mapping from node_id to fused subgraph
        node_to_fused_subgraph = {}
        node_position_in_subgraph = {}

        for fused_sg in self.fused_subgraphs:
            for idx, node_id in enumerate(fused_sg.node_ids):
                node_to_fused_subgraph[node_id] = fused_sg
                node_position_in_subgraph[node_id] = (fused_sg, idx, len(fused_sg.node_ids))

        # Collect nodes
        all_nodes = list(fx_graph.graph.nodes)
        if max_nodes:
            all_nodes = all_nodes[:max_nodes]

        # Build visualization
        lines = []

        # Header
        left_width = 50
        right_width = 70
        total_width = left_width + 4 + right_width

        lines.append(box['heavy_horizontal'] * total_width)
        title = "FUSION-BASED PARTITIONING (Color-Coded by Bottleneck)"
        lines.append(colorize(title, ANSIColor.BOLD, capability))
        lines.append(box['heavy_horizontal'] * total_width)
        lines.append("")

        # Add legend if color is supported
        if capability in [TerminalCapability.COLOR, TerminalCapability.TRUECOLOR]:
            lines.append(create_legend(capability))
            lines.append("")

        # Column headers
        header_left = "FX Graph (Execution Order)".ljust(left_width)
        header_right = "Fused Subgraphs (Color-Coded)"
        lines.append(f"{header_left}    {header_right}")
        lines.append(box['horizontal'] * left_width + "    " + box['horizontal'] * right_width)
        lines.append("")

        # Process each node
        subgraph_counter = 1
        current_subgraph_id = None

        for idx, node in enumerate(all_nodes, 1):
            node_id = node.name

            # LEFT SIDE: FX Node info
            left_lines = self._format_fx_node_colored(node, fx_graph, idx, capability, box)

            # RIGHT SIDE: Fused subgraph info
            right_lines = []

            if node_id in node_to_fused_subgraph:
                fused_sg, node_idx, total_nodes = node_position_in_subgraph[node_id]
                is_first = (node_idx == 0)
                is_last = (node_idx == total_nodes - 1)

                if fused_sg.subgraph_id != current_subgraph_id:
                    current_subgraph_id = fused_sg.subgraph_id

                    if is_first:
                        header_lines = self._format_fused_subgraph_header_colored(
                            fused_sg, subgraph_counter, capability, box
                        )
                        right_lines.extend(header_lines)
                        subgraph_counter += 1

                op_lines = self._format_fused_operator_colored(node, fx_graph, is_first, is_last, capability, box)
                right_lines.extend(op_lines)

                if is_last:
                    footer_lines = self._format_fused_subgraph_footer_colored(fused_sg, capability, box)
                    right_lines.extend(footer_lines)
                    current_subgraph_id = None
            else:
                right_lines = self._format_not_fused_colored(node, capability, box)

            # Combine left and right
            max_lines = max(len(left_lines), len(right_lines))
            for i in range(max_lines):
                left = left_lines[i] if i < len(left_lines) else ""
                right = right_lines[i] if i < len(right_lines) else ""
                lines.append(f"{left.ljust(left_width)}    {right}")

            lines.append("")

        # Footer
        if max_nodes and len(fx_graph.graph.nodes) > max_nodes:
            lines.append(f"... ({len(fx_graph.graph.nodes) - max_nodes} more nodes not shown)")
            lines.append("")

        lines.append(box['heavy_horizontal'] * total_width)
        lines.append(f"Total FX nodes: {len(fx_graph.graph.nodes)}")
        lines.append(f"Fused subgraphs: {len(self.fused_subgraphs)}")
        lines.append(f"Reduction: {len(fx_graph.graph.nodes) / max(1, len(self.fused_subgraphs)):.1f}× fewer execution units")

        if self.fused_subgraphs:
            avg_fusion = sum(sg.num_operators for sg in self.fused_subgraphs) / len(self.fused_subgraphs)
            lines.append(f"Average fusion size: {avg_fusion:.1f} operators/subgraph")

        lines.append(box['heavy_horizontal'] * total_width)

        return "\n".join(lines)

    def _format_fx_node_colored(self, node, graph: GraphModule, idx: int,
                               capability: TerminalCapability, box: dict) -> List[str]:
        """Format FX node with color support"""
        import torch.nn as nn

        lines = []
        lines.append(f"{idx}. [{node.op}] {node.name}")

        if node.op == 'call_module':
            try:
                module = graph.get_submodule(node.target)
                module_type = type(module).__name__
                lines.append(f"   {module_type}")
            except:
                lines.append(f"   {node.target}")
        elif node.op == 'call_function':
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            lines.append(f"   Function: {func_name}")

        return lines

    def _format_fused_subgraph_header_colored(self, fused_sg: FusedSubgraph, counter: int,
                                             capability: TerminalCapability, box: dict) -> List[str]:
        """Format fused subgraph header with color coding"""
        lines = []

        # Get color for bottleneck type
        color_start, color_end = get_bottleneck_color(
            fused_sg.recommended_bottleneck.value, capability
        )

        # Header with color
        header_text = f"SUBGRAPH #{counter}"
        colored_header = f"{color_start}{header_text}{color_end}"
        lines.append(f"{box['top_left']}{box['horizontal']} {colored_header} {box['horizontal'] * 30}")

        lines.append(f"{box['vertical']}  Pattern: {fused_sg.fusion_pattern[:40]}")
        lines.append(f"{box['vertical']}  Operators: {fused_sg.num_operators}")

        # Bottleneck type with color
        bottleneck_text = f"Type: {fused_sg.recommended_bottleneck.value.upper()}"
        colored_bottleneck = f"{color_start}{bottleneck_text}{color_end}"
        lines.append(f"{box['vertical']}  {colored_bottleneck}")
        lines.append(f"{box['vertical']}")

        return lines

    def _format_fused_operator_colored(self, node, graph: GraphModule, is_first: bool,
                                      is_last: bool, capability: TerminalCapability, box: dict) -> List[str]:
        """Format operator within fused subgraph with color"""
        import torch.nn as nn

        lines = []

        op_type = "unknown"
        if node.op == 'call_module':
            try:
                module = graph.get_submodule(node.target)
                op_type = type(module).__name__
            except:
                op_type = str(node.target)
        elif node.op == 'call_function':
            op_type = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)

        lines.append(f"{box['vertical']}  {box['vertical_right']} {node.name} ({op_type})")

        return lines

    def _format_fused_subgraph_footer_colored(self, fused_sg: FusedSubgraph,
                                             capability: TerminalCapability, box: dict) -> List[str]:
        """Format subgraph footer with metrics"""
        lines = []

        lines.append(f"{box['vertical']}")

        # Get color for bottleneck
        color_start, color_end = get_bottleneck_color(
            fused_sg.recommended_bottleneck.value, capability
        )

        # Metrics
        flops_str = self._format_number(fused_sg.total_flops, 'FLOPs')
        lines.append(f"{box['vertical']}  Compute: {flops_str}")

        external_bytes = fused_sg.total_input_bytes + fused_sg.total_output_bytes + fused_sg.total_weight_bytes
        external_str = self._format_bytes(external_bytes)
        lines.append(f"{box['vertical']}  Memory: {external_str}")

        if fused_sg.internal_bytes > 0:
            internal_str = self._format_bytes(fused_sg.internal_bytes)
            reduction_pct = fused_sg.data_movement_reduction() * 100
            lines.append(f"{box['vertical']}  Saved: {internal_str} ({reduction_pct:.1f}%)")

        ai_text = f"AI: {fused_sg.arithmetic_intensity:.1f} FLOPs/byte"
        colored_ai = f"{color_start}{ai_text}{color_end}"
        lines.append(f"{box['vertical']}  {colored_ai}")

        lines.append(f"{box['bottom_left']}{box['horizontal'] * 45}")

        return lines

    def _format_not_fused_colored(self, node, capability: TerminalCapability, box: dict) -> List[str]:
        """Format non-fused node"""
        lines = []
        dim_color = ANSIColor.DIM if capability in [TerminalCapability.COLOR, TerminalCapability.TRUECOLOR] else ""
        reset = ANSIColor.RESET if capability in [TerminalCapability.COLOR, TerminalCapability.TRUECOLOR] else ""

        lines.append(f"{dim_color}(not fused){reset}")

        if node.op == 'placeholder':
            lines.append(f"{dim_color}Reason: input placeholder{reset}")
        elif node.op == 'output':
            lines.append(f"{dim_color}Reason: output node{reset}")
        else:
            lines.append(f"{dim_color}Reason: structural operation{reset}")

        return lines

    def export_to_graphviz(self, fx_graph: GraphModule, output_file: str = "fusion_graph.dot"):
        """
        Export fusion partitioning to DOT format for Graphviz visualization.

        Creates a visual graph showing fused subgraphs as nodes, colored by
        bottleneck type, with edges showing data dependencies.

        Args:
            fx_graph: The FX graph that was partitioned
            output_file: Path to output .dot file

        Example:
            partitioner.export_to_graphviz(fx_graph, "resnet_fusion.dot")
            # Then generate PNG: dot -Tpng resnet_fusion.dot -o resnet_fusion.png
        """
        export_to_dot(self.fused_subgraphs, fx_graph, output_file)
