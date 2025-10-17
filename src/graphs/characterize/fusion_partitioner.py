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

        # Step 3: Compute statistics
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
        - BatchNorm2d → ReLU
        - Conv2d → ReLU
        - Linear → ReLU/GELU
        - Add → ReLU (residual)
        - Any element-wise chain
        """

        type1 = self._get_node_type(node1)
        type2 = self._get_node_type(node2)

        # Define fusible patterns
        fusible_patterns = [
            ('Conv2d', 'BatchNorm2d'),
            ('Conv2d', 'ReLU'),
            ('Conv2d', 'ReLU6'),
            ('BatchNorm2d', 'ReLU'),
            ('BatchNorm2d', 'ReLU6'),
            ('BatchNorm2d', 'Hardswish'),
            ('Linear', 'BatchNorm1d'),
            ('Linear', 'ReLU'),
            ('Linear', 'GELU'),
            ('Linear', 'Dropout'),
            ('add', 'ReLU'),  # Residual connection
            ('add', 'ReLU6'),
            # Element-wise operations can fuse
            ('ReLU', 'Dropout'),
            ('GELU', 'Dropout'),
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
