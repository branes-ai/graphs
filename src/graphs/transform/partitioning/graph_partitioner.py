"""
Graph Partitioner: Decomposes computation graph into subgraphs with full statistics.

This module implements the first phase of realistic performance modeling:
- Partition FX graph into executable subgraphs (kernels)
- Analyze parallelism dimensions for each subgraph
- Compute memory traffic and arithmetic intensity
- Build dependency graph for concurrency analysis
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule
from typing import List, Dict, Tuple, Optional
import networkx as nx

from graphs.ir.structures import (
    SubgraphDescriptor,
    ParallelismDescriptor,
    TensorDescriptor,
    PartitionReport,
    OperationType,
    BottleneckType,
    PartitionReason,
    create_tensor_descriptor,
    classify_operation_type
)


class GraphPartitioner:
    """Partition FX graph into subgraphs with detailed statistics"""

    def __init__(self):
        self.subgraphs: List[SubgraphDescriptor] = []
        self.dependency_graph: Optional[nx.DiGraph] = None

    @staticmethod
    def _get_shape(meta):
        """
        Safely extract shape from tensor metadata.

        Handles different metadata formats:
        1. TensorMetadata object with .shape attribute (most common)
        2. Plain tuple representing the shape directly
        3. Other tensor-like objects with .shape

        Note: Check for .shape FIRST because TensorMetadata is a namedtuple,
        so isinstance(meta, tuple) is True, but we want meta.shape not the full tuple.
        """
        if hasattr(meta, 'shape'):
            return meta.shape
        elif isinstance(meta, tuple):
            return meta
        else:
            raise ValueError(f"Unexpected meta type: {type(meta)}, value: {meta}")

    @staticmethod
    def _compute_total_elements(shape) -> int:
        """
        Safely compute total elements from shape, handling None/dynamic dimensions.

        Args:
            shape: Tuple of dimensions (may contain None for dynamic dims)

        Returns:
            Total number of elements (treats None as 1)
        """
        total = 1
        for dim in shape:
            if dim is not None and isinstance(dim, int):
                total *= dim
            elif dim is not None:
                # Handle torch.SymInt or other symbolic types
                try:
                    total *= int(dim)
                except (TypeError, ValueError):
                    # Can't convert to int, treat as 1
                    pass
        return total

    @staticmethod
    def _safe_dim(shape, index: int, default: int = 1) -> int:
        """
        Safely extract a dimension from shape, handling None/dynamic/missing dims.

        Args:
            shape: Tuple of dimensions
            index: Index to extract
            default: Default value if index is out of bounds or value is None

        Returns:
            Integer dimension value
        """
        if index >= len(shape):
            return default

        dim = shape[index]
        if dim is None:
            return default

        if isinstance(dim, int):
            return dim

        # Try to convert (e.g., torch.SymInt)
        try:
            return int(dim)
        except (TypeError, ValueError):
            return default

    def partition(self, fx_graph: GraphModule) -> PartitionReport:
        """
        Partition graph into subgraphs

        Args:
            fx_graph: PyTorch FX traced graph with shape propagation

        Returns:
            PartitionReport with statistics and subgraph list
        """
        self.subgraphs = []

        # Extract call_module nodes (actual operations)
        call_module_nodes = [node for node in fx_graph.graph.nodes
                            if node.op == 'call_module']

        # Also extract call_function nodes (like add, flatten, etc.)
        call_function_nodes = [node for node in fx_graph.graph.nodes
                              if node.op == 'call_function']

        # Analyze call_module nodes
        for node in call_module_nodes:
            subgraph = self._analyze_node(node, fx_graph)
            if subgraph:
                self.subgraphs.append(subgraph)

        # Analyze call_function nodes
        for node in call_function_nodes:
            subgraph = self._analyze_function_node(node, fx_graph)
            if subgraph:
                self.subgraphs.append(subgraph)

        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()

        # Generate report
        report = self._generate_report()

        return report

    def _analyze_node(self, node, graph: GraphModule) -> Optional[SubgraphDescriptor]:
        """Analyze a single FX node to create SubgraphDescriptor"""

        try:
            # Get module
            module = graph.get_submodule(node.target)

            # Get tensor metadata (from shape propagation)
            meta = node.meta.get('tensor_meta')
            if not meta:
                print(f"Warning: No tensor metadata for {node.name}")
                return None

            # Classify operation
            op_type = self._classify_operation(module)

            # Compute parallelism
            parallelism = self._compute_parallelism(node, meta, module)

            # Compute FLOPs and memory
            flops, macs = self._compute_flops(node, meta, module, op_type)
            input_bytes, output_bytes, weight_bytes = self._compute_memory(node, meta, module)

            # Find dependencies
            depends_on = self._find_dependencies(node)

            # Determine partition reasoning
            partition_reason, partition_criteria, fusion_candidates = self._determine_partition_reason(
                node, graph, flops, input_bytes + output_bytes + weight_bytes
            )

            # Create descriptor (unified interface - single-op subgraph)
            subgraph = SubgraphDescriptor(
                subgraph_id=abs(hash(str(id(node)))) % (2**31),  # Deterministic numeric ID
                node_ids=[str(id(node))],  # List with single element
                node_names=[node.name],  # List with single element
                operation_types=[op_type],  # List with single element
                fusion_pattern=self._infer_fusion_pattern(node, graph),
                total_flops=flops,  # Use unified name
                total_macs=macs,  # Use unified name
                total_input_bytes=input_bytes,
                total_output_bytes=output_bytes,
                total_weight_bytes=weight_bytes,
                internal_bytes=0,  # No fusion, so no internal bytes
                num_operators=1,  # Single operator (unfused)
                parallelism=parallelism,
                depends_on=[],  # Will be filled in later with numeric IDs
                input_tensors=[],  # Could populate with TensorDescriptor but optional for now
                output_tensors=[],
                weight_tensors=[],
                partition_reason=partition_reason,
                partition_criteria=partition_criteria,
                fusion_candidates=fusion_candidates
            )

            return subgraph

        except Exception as e:
            print(f"Error analyzing node {node.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_function_node(self, node, graph: GraphModule) -> Optional[SubgraphDescriptor]:
        """Analyze a call_function node (like add, flatten, etc.)"""

        try:
            # Get tensor metadata
            meta = node.meta.get('tensor_meta')
            if not meta:
                # Skip nodes without tensor metadata
                return None

            # Classify function
            import torch
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)

            # Determine operation type based on function
            if func_name in ['add', 'mul', 'sub', 'div']:
                op_type = OperationType.ELEMENTWISE
            elif func_name == 'flatten':
                # Flatten is just a view operation, no FLOPs
                return None
            else:
                # Skip unknown functions
                return None

            # Compute FLOPs for elementwise operations
            total_elements = self._compute_total_elements(self._get_shape(meta))

            flops = total_elements if op_type == OperationType.ELEMENTWISE else 0
            macs = 0

            # Compute memory traffic
            # Input: sum of all input tensors
            input_bytes = 0
            for arg in node.args:
                if hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                    arg_meta = arg.meta['tensor_meta']
                    size = self._compute_total_elements(self._get_shape(arg_meta))
                    input_bytes += size * 4  # float32

            # Output
            output_bytes = total_elements * 4

            # No weights for elementwise ops
            weight_bytes = 0

            # Compute parallelism (simple for elementwise)
            shape = self._get_shape(meta)
            B = self._safe_dim(shape, 0)
            C = self._safe_dim(shape, 1)

            parallelism = ParallelismDescriptor(
                batch=B,
                channels=C,
                spatial=total_elements // (B * C),
                total_threads=total_elements,
                vectorizable_dim='all'
            )

            # Find dependencies
            depends_on = self._find_dependencies(node)

            # Determine partition reasoning
            partition_reason, partition_criteria, fusion_candidates = self._determine_partition_reason(
                node, graph, flops, input_bytes + output_bytes + weight_bytes
            )

            # Create descriptor (unified interface - single-op subgraph)
            subgraph = SubgraphDescriptor(
                subgraph_id=abs(hash(str(id(node)))) % (2**31),
                node_ids=[str(id(node))],
                node_names=[node.name],
                operation_types=[op_type],
                fusion_pattern=node.name,
                total_flops=flops,
                total_macs=macs,
                total_input_bytes=input_bytes,
                total_output_bytes=output_bytes,
                total_weight_bytes=weight_bytes,
                internal_bytes=0,
                num_operators=1,
                parallelism=parallelism,
                depends_on=[],
                partition_reason=partition_reason,
                partition_criteria=partition_criteria,
                fusion_candidates=fusion_candidates
            )

            return subgraph

        except Exception as e:
            print(f"Error analyzing function node {node.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _classify_operation(self, module) -> OperationType:
        """Classify module type to operation type"""
        module_type = type(module)

        if module_type == nn.Conv2d:
            # Check for depthwise or pointwise
            if module.groups == module.in_channels == module.out_channels and module.groups > 1:
                return OperationType.CONV2D_DEPTHWISE
            elif module.kernel_size == (1, 1) or module.kernel_size == 1:
                return OperationType.CONV2D_POINTWISE
            else:
                return OperationType.CONV2D
        else:
            return classify_operation_type(module_type)

    def _compute_parallelism(self, node, meta, module) -> ParallelismDescriptor:
        """Compute available parallelism dimensions"""

        if isinstance(module, nn.Conv2d):
            shape = self._get_shape(meta)
            B = self._safe_dim(shape, 0)
            C_in = self._safe_dim(shape, 1)
            H = self._safe_dim(shape, 2)
            W = self._safe_dim(shape, 3)

            # Output dimensions
            C_out = module.out_channels
            K_h, K_w = (module.kernel_size if isinstance(module.kernel_size, tuple)
                       else (module.kernel_size, module.kernel_size))
            S_h, S_w = (module.stride if isinstance(module.stride, tuple)
                       else (module.stride, module.stride))
            P = module.padding if isinstance(module.padding, int) else module.padding[0]

            H_out = (H + 2 * P - K_h) // S_h + 1
            W_out = (W + 2 * P - K_w) // S_w + 1

            # Validate output dimensions - can be 0 or negative if input shape was defaulted
            if H_out <= 0 or W_out <= 0:
                print(f"WARNING: Conv2d produced invalid output dimensions:")
                print(f"  Node: {node.name}")
                print(f"  Input shape: {shape}")
                print(f"  H={H}, W={W}, K_h={K_h}, K_w={K_w}, S_h={S_h}, S_w={S_w}, P={P}")
                print(f"  H_out={H_out}, W_out={W_out}")
                print(f"  Using H_out=W_out=1 as fallback")
                H_out = max(1, H_out)
                W_out = max(1, W_out)

            # Check if depthwise
            is_depthwise = (module.groups == C_in == C_out and module.groups > 1)
            is_grouped = (module.groups > 1)

            return ParallelismDescriptor(
                batch=B,
                channels=C_out,
                spatial=H_out * W_out,
                total_threads=B * C_out * H_out * W_out,
                is_depthwise=is_depthwise,
                is_grouped=is_grouped,
                num_groups=module.groups,
                vectorizable_dim='channels' if not is_depthwise else 'spatial',
                can_split_channels=not is_depthwise
            )

        elif isinstance(module, nn.Linear):
            # Assume input is (batch, features)
            shape = self._get_shape(meta)
            if len(shape) == 2:
                B = self._safe_dim(shape, 0)
                D_in = self._safe_dim(shape, 1)
            else:
                # Flatten all but last dimension
                B = self._compute_total_elements(shape[:-1])
                D_in = self._safe_dim(shape, -1)

            D_out = module.out_features

            # Validate dimensions
            if B <= 0:
                print(f"WARNING: Linear layer has invalid batch size B={B}, using B=1")
                B = 1

            return ParallelismDescriptor(
                batch=B,
                channels=D_out,
                spatial=1,
                total_threads=B * D_out,
                vectorizable_dim='channels'
            )

        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.Hardswish)):
            # Elementwise operations
            shape = self._get_shape(meta)
            B = self._safe_dim(shape, 0)
            C = self._safe_dim(shape, 1)
            total_elements = self._compute_total_elements(shape)

            # Validate division to ensure spatial is positive
            spatial = max(1, total_elements // (B * C)) if (B * C) > 0 else 1

            return ParallelismDescriptor(
                batch=B,
                channels=C,
                spatial=spatial,
                total_threads=max(1, total_elements),
                vectorizable_dim='all'
            )

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            # Pooling operations
            # Output shape from meta (since pooling reduces spatial dimensions)
            shape = self._get_shape(meta)
            B = self._safe_dim(shape, 0)
            C = self._safe_dim(shape, 1)
            H = self._safe_dim(shape, 2)
            W = self._safe_dim(shape, 3)

            return ParallelismDescriptor(
                batch=B,
                channels=C,
                spatial=H * W,
                total_threads=B * C * H * W,
                vectorizable_dim='channels'
            )

        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            # Normalization operations
            shape = self._get_shape(meta)
            B = self._safe_dim(shape, 0)
            C = self._safe_dim(shape, 1)
            total_elements = self._compute_total_elements(shape)

            # Validate division to ensure spatial is positive
            spatial = max(1, total_elements // (B * C)) if (B * C) > 0 else 1

            return ParallelismDescriptor(
                batch=B,
                channels=C,
                spatial=spatial,
                total_threads=max(1, total_elements),
                vectorizable_dim='all'
            )

        else:
            # Default for unknown operations
            shape = self._get_shape(meta)
            total_elements = self._compute_total_elements(shape)

            return ParallelismDescriptor(
                batch=1,
                channels=1,
                spatial=max(1, total_elements),
                total_threads=max(1, total_elements)
            )

    def _compute_flops(self, node, meta, module, op_type: OperationType) -> Tuple[int, int]:
        """Compute FLOPs and MACs for operation"""

        if op_type in [OperationType.CONV2D, OperationType.CONV2D_DEPTHWISE, OperationType.CONV2D_POINTWISE]:
            # Get INPUT shape from node arguments, not output metadata
            # self._get_shape(meta) is the OUTPUT shape, but we need INPUT shape for FLOP calculation
            if node.args and hasattr(node.args[0], 'meta') and 'tensor_meta' in node.args[0].meta:
                input_meta = node.args[0].meta['tensor_meta']
                input_shape = self._get_shape(input_meta)
                B = self._safe_dim(input_shape, 0)
                C_in = self._safe_dim(input_shape, 1)
                H = self._safe_dim(input_shape, 2)
                W = self._safe_dim(input_shape, 3)
            else:
                # Fallback: use module parameters (less accurate for batch size)
                B = 1
                C_in = module.in_channels
                H = W = 224  # Assume default, will be inaccurate

            C_out = module.out_channels
            K_h, K_w = (module.kernel_size if isinstance(module.kernel_size, tuple)
                       else (module.kernel_size, module.kernel_size))
            S_h, S_w = (module.stride if isinstance(module.stride, tuple)
                       else (module.stride, module.stride))
            P = module.padding if isinstance(module.padding, int) else module.padding[0]
            groups = module.groups

            H_out = (H + 2 * P - K_h) // S_h + 1
            W_out = (W + 2 * P - K_w) // S_w + 1

            # Depthwise convolution
            if groups == C_in == C_out and groups > 1:
                # Each output channel depends only on corresponding input channel
                macs = B * C_out * H_out * W_out * K_h * K_w
            else:
                # Standard or grouped convolution
                C_in_per_group = C_in // groups
                macs = B * C_out * H_out * W_out * C_in_per_group * K_h * K_w

            # FLOPs = 2 × MACs (multiply + add)
            flops = 2 * macs

            return flops, macs

        elif op_type == OperationType.LINEAR:
            # Get INPUT shape from node arguments
            if node.args and hasattr(node.args[0], 'meta') and 'tensor_meta' in node.args[0].meta:
                input_meta = node.args[0].meta['tensor_meta']
                input_shape = self._get_shape(input_meta)

                if len(input_shape) == 2:
                    B = self._safe_dim(input_shape, 0)
                    D_in = self._safe_dim(input_shape, 1)
                else:
                    # Flatten all but last dimension
                    B = self._compute_total_elements(input_shape[:-1])
                    D_in = self._safe_dim(input_shape, -1)
            else:
                # Fallback
                B = 1
                D_in = module.in_features

            D_out = module.out_features

            macs = B * D_in * D_out
            flops = 2 * macs

            return flops, macs

        elif op_type in [OperationType.RELU, OperationType.RELU6, OperationType.GELU, OperationType.HARDSWISH]:
            # Activation functions: ~1 FLOP per element
            total_elements = self._compute_total_elements(self._get_shape(meta))

            flops = total_elements
            macs = 0

            return flops, macs

        elif op_type == OperationType.BATCHNORM:
            # BatchNorm: mean, variance, normalize, scale, shift
            # ~4-5 ops per element
            total_elements = self._compute_total_elements(self._get_shape(meta))

            flops = total_elements * 5
            macs = 0

            return flops, macs

        elif op_type in [OperationType.MAXPOOL, OperationType.AVGPOOL]:
            # Pooling: comparisons (MaxPool) or additions/divisions (AvgPool)
            # FVCore doesn't count these, so we return 0 to match
            # (Technically MaxPool does comparisons, AvgPool does adds/divs)
            return 0, 0

        elif op_type == OperationType.ADAPTIVEAVGPOOL:
            # Adaptive pooling: FVCore doesn't count these
            return 0, 0

        elif op_type == OperationType.ELEMENTWISE:
            # Element-wise operations (add, mul, etc.): 1 FLOP per element
            total_elements = self._compute_total_elements(self._get_shape(meta))

            flops = total_elements
            macs = 0

            return flops, macs

        else:
            # Unknown operation
            return 0, 0

    def _compute_memory(self, node, meta, module) -> Tuple[int, int, int]:
        """Compute memory traffic: input, output, weights"""

        # Output tensor
        output_bytes = self._compute_total_elements(self._get_shape(meta)) * 4  # float32

        # Input tensors
        input_bytes = 0
        for arg in node.args:
            if hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                input_meta = arg.meta['tensor_meta']
                size = self._compute_total_elements(self._get_shape(input_meta))
                input_bytes += size * 4

        # Weight tensors
        weight_bytes = 0
        if isinstance(module, nn.Conv2d):
            # Weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
            weight_bytes = (module.out_channels *
                          (module.in_channels // module.groups) *
                          module.kernel_size[0] *
                          module.kernel_size[1] * 4)
            if module.bias is not None:
                weight_bytes += module.out_channels * 4

        elif isinstance(module, nn.Linear):
            # Weight: [out_features, in_features]
            weight_bytes = module.out_features * module.in_features * 4
            if module.bias is not None:
                weight_bytes += module.out_features * 4

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: weight, bias, running_mean, running_var
            weight_bytes = module.num_features * 4 * 4

        return input_bytes, output_bytes, weight_bytes

    def _find_dependencies(self, node) -> List[str]:
        """Find nodes that this node depends on"""
        depends_on = []

        for arg in node.args:
            if hasattr(arg, 'op') and arg.op == 'call_module':
                depends_on.append(str(id(arg)))

        return depends_on

    def _infer_fusion_pattern(self, node, graph) -> str:
        """Infer fusion pattern (e.g., conv_bn_relu)"""
        # Simplified: just return the node name for now
        # In a full implementation, would look at adjacent nodes
        return node.name

    def _determine_partition_reason(
        self, node, graph: GraphModule, flops: int, total_bytes: int
    ) -> Tuple[PartitionReason, Dict[str, any], List[str]]:
        """
        Determine why this node is partitioned separately.

        Returns:
            (partition_reason, partition_criteria, fusion_candidates)
        """
        # Default thresholds (can be made configurable)
        MAX_FUSION_FLOPS = 10e9  # 10 GFLOPs
        MAX_FUSION_MEMORY = 100e6  # 100 MB

        fusion_candidates = []
        partition_criteria = {
            'flops': flops,
            'total_bytes': total_bytes,
            'arithmetic_intensity': flops / total_bytes if total_bytes > 0 else 0.0
        }

        # Find adjacent nodes that could potentially be fused
        # Look at users (nodes that consume this node's output)
        users = list(node.users.keys())
        fusable_users = []

        for user in users:
            if user.op == 'call_module':
                try:
                    user_module = graph.get_submodule(user.target)
                    # Check if fusion is theoretically possible
                    if self._can_fuse_operations(node, user, graph):
                        fusable_users.append(user.name)
                        fusion_candidates.append(str(id(user)))
                except:
                    pass

        # Determine partition reason
        if flops > MAX_FUSION_FLOPS:
            partition_criteria['threshold_flops'] = MAX_FUSION_FLOPS
            return PartitionReason.COMPUTE_THRESHOLD_EXCEEDED, partition_criteria, fusion_candidates

        if total_bytes > MAX_FUSION_MEMORY:
            partition_criteria['threshold_memory'] = MAX_FUSION_MEMORY
            return PartitionReason.MEMORY_LIMIT_EXCEEDED, partition_criteria, fusion_candidates

        # Check if there are fusion opportunities
        if fusion_candidates:
            partition_criteria['fusion_candidates'] = fusable_users
            return PartitionReason.FUSION_OPPORTUNITY, partition_criteria, fusion_candidates

        # Check data dependencies
        if len(list(node.users.keys())) > 1:
            # Multiple consumers - harder to fuse
            partition_criteria['num_consumers'] = len(list(node.users.keys()))
            return PartitionReason.DATA_DEPENDENCY, partition_criteria, fusion_candidates

        # Default: operation boundary
        return PartitionReason.OPERATION_BOUNDARY, partition_criteria, fusion_candidates

    def _can_fuse_operations(self, node1, node2, graph: GraphModule) -> bool:
        """
        Check if two operations can theoretically be fused.

        This is a simplified check - in practice, fusion depends on:
        - Operation types (conv+bn, conv+relu, etc.)
        - Data flow patterns
        - Hardware capabilities
        """
        try:
            module1 = graph.get_submodule(node1.target)
            module2 = graph.get_submodule(node2.target)

            # Common fusion patterns
            fusable_pairs = [
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv2d, nn.ReLU),
                (nn.Conv2d, nn.ReLU6),
                (nn.BatchNorm2d, nn.ReLU),
                (nn.Linear, nn.ReLU),
                (nn.Linear, nn.GELU),
            ]

            for type1, type2 in fusable_pairs:
                if isinstance(module1, type1) and isinstance(module2, type2):
                    return True

            return False
        except:
            return False

    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build directed graph of dependencies"""
        G = nx.DiGraph()

        # Add nodes
        for subgraph in self.subgraphs:
            G.add_node(subgraph.node_id, subgraph=subgraph)

        # Add edges (dependencies)
        for subgraph in self.subgraphs:
            for dep_id in subgraph.depends_on:
                if dep_id in G.nodes:
                    G.add_edge(dep_id, subgraph.node_id)

        return G

    def _generate_report(self) -> PartitionReport:
        """Generate comprehensive partition report"""

        if not self.subgraphs:
            return PartitionReport(
                subgraphs=[],
                total_subgraphs=0,
                total_flops=0,
                total_macs=0,
                total_memory_traffic=0,
                average_arithmetic_intensity=0.0,
                min_arithmetic_intensity=0.0,
                max_arithmetic_intensity=0.0
            )

        # Compute totals
        total_flops = sum(sg.flops for sg in self.subgraphs)
        total_macs = sum(sg.macs for sg in self.subgraphs)
        total_memory = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                          for sg in self.subgraphs)

        # Arithmetic intensity stats
        ai_values = [sg.arithmetic_intensity for sg in self.subgraphs if sg.arithmetic_intensity > 0]
        avg_ai = sum(ai_values) / len(ai_values) if ai_values else 0.0
        min_ai = min(ai_values) if ai_values else 0.0
        max_ai = max(ai_values) if ai_values else 0.0

        # Operation type distribution
        op_type_counts = {}
        for sg in self.subgraphs:
            op_name = sg.operation_type.value
            op_type_counts[op_name] = op_type_counts.get(op_name, 0) + 1

        # Bottleneck distribution
        bottleneck_counts = {}
        for sg in self.subgraphs:
            bt_name = sg.recommended_bottleneck.value
            bottleneck_counts[bt_name] = bottleneck_counts.get(bt_name, 0) + 1

        # Partition reason distribution
        partition_reason_counts = {}
        for sg in self.subgraphs:
            pr_name = sg.partition_reason.value
            partition_reason_counts[pr_name] = partition_reason_counts.get(pr_name, 0) + 1

        # Parallelism distribution
        parallelism_dist = {'<1K': 0, '1K-10K': 0, '10K-100K': 0, '100K-1M': 0, '>1M': 0}
        for sg in self.subgraphs:
            if sg.parallelism:
                threads = sg.parallelism.total_threads
                if threads < 1000:
                    parallelism_dist['<1K'] += 1
                elif threads < 10000:
                    parallelism_dist['1K-10K'] += 1
                elif threads < 100000:
                    parallelism_dist['10K-100K'] += 1
                elif threads < 1000000:
                    parallelism_dist['100K-1M'] += 1
                else:
                    parallelism_dist['>1M'] += 1

        # Critical path (simplified: longest path in dependency graph)
        critical_path = []
        if self.dependency_graph and len(self.dependency_graph.nodes) > 0:
            try:
                critical_path = nx.dag_longest_path(self.dependency_graph)
            except:
                critical_path = list(self.dependency_graph.nodes)[:10]  # fallback

        return PartitionReport(
            subgraphs=self.subgraphs,
            total_subgraphs=len(self.subgraphs),
            total_flops=total_flops,
            total_macs=total_macs,
            total_memory_traffic=total_memory,
            average_arithmetic_intensity=avg_ai,
            min_arithmetic_intensity=min_ai,
            max_arithmetic_intensity=max_ai,
            operation_type_counts=op_type_counts,
            bottleneck_distribution=bottleneck_counts,
            partition_reason_distribution=partition_reason_counts,
            parallelism_distribution=parallelism_dist,
            critical_path_subgraphs=critical_path
        )

    def visualize_partitioning(self, fx_graph: GraphModule,
                              start: Optional[int] = None,
                              end: Optional[int] = None) -> str:
        """
        Create side-by-side visualization of FX graph and partitioned subgraphs.

        Shows original graph on left, partitioned subgraphs on right, aligned vertically.

        Args:
            fx_graph: The FX graph that was partitioned
            start: Starting node index (0-based, inclusive), None for beginning
            end: Ending node index (0-based, exclusive), None for end

        Returns:
            String containing the formatted visualization
        """
        # Build mapping from node_id to subgraph
        node_to_subgraph = {}
        for sg in self.subgraphs:
            node_to_subgraph[sg.node_id] = sg

        # Collect all nodes in execution order
        all_nodes = list(fx_graph.graph.nodes)
        total_nodes = len(all_nodes)

        # Determine range to display
        start = start if start is not None else 0
        end = end if end is not None else total_nodes

        # Clamp to valid range
        start = max(0, min(start, total_nodes))
        end = max(start, min(end, total_nodes))

        # Slice to requested range
        nodes_to_show = all_nodes[start:end]

        # Build visualization
        lines = []

        # Header
        left_width = 50
        right_width = 60
        total_width = left_width + 4 + right_width

        lines.append("=" * total_width)
        lines.append("GRAPH PARTITIONING VISUALIZATION")
        lines.append("=" * total_width)
        lines.append("")

        # Column headers
        header_left = "FX Graph (Execution Order)".ljust(left_width)
        header_right = "Partitioned Subgraphs"
        lines.append(f"{header_left}    {header_right}")
        lines.append("-" * left_width + "    " + "-" * right_width)
        lines.append("")

        # Process each node
        subgraph_counter = 1
        for idx, node in enumerate(nodes_to_show, start + 1):
            node_id = str(id(node))

            # LEFT SIDE: FX Node info
            left_lines = self._format_fx_node(node, fx_graph, idx)

            # RIGHT SIDE: Subgraph info (if exists)
            if node_id in node_to_subgraph:
                sg = node_to_subgraph[node_id]
                right_lines = self._format_subgraph(sg, subgraph_counter)
                subgraph_counter += 1
            else:
                right_lines = self._format_not_partitioned(node)

            # Combine left and right, ensuring same number of lines
            max_lines = max(len(left_lines), len(right_lines))
            for i in range(max_lines):
                left = left_lines[i] if i < len(left_lines) else ""
                right = right_lines[i] if i < len(right_lines) else ""
                lines.append(f"{left.ljust(left_width)}    {right}")

            # Add spacing between nodes
            lines.append("")

        # Footer
        nodes_shown = end - start
        if nodes_shown < total_nodes:
            nodes_not_shown = total_nodes - nodes_shown
            if start > 0 and end < total_nodes:
                lines.append(f"... ({start} nodes before, {total_nodes - end} nodes after not shown)")
            elif start > 0:
                lines.append(f"... ({start} nodes before not shown)")
            else:
                lines.append(f"... ({nodes_not_shown} more nodes not shown)")
            lines.append("")

        lines.append("=" * total_width)
        lines.append(f"Total FX nodes: {total_nodes}")
        lines.append(f"Partitioned subgraphs: {len(self.subgraphs)}")
        lines.append(f"Nodes not partitioned: {total_nodes - len(self.subgraphs)}")
        lines.append("=" * total_width)

        return "\n".join(lines)

    def _format_fx_node(self, node, graph: GraphModule, idx: int) -> List[str]:
        """Format FX node information for left column"""
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
                    details += f"{module.kernel_size}, stride={module.stride})"
                    lines.append(details)
                elif isinstance(module, nn.Linear):
                    details = f"   Linear({module.in_features}->{module.out_features})"
                    lines.append(details)
                elif isinstance(module, nn.BatchNorm2d):
                    details = f"   BatchNorm2d({module.num_features})"
                    lines.append(details)
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

    def _format_subgraph(self, sg: SubgraphDescriptor, counter: int) -> List[str]:
        """Format subgraph information for right column"""
        lines = []

        # Subgraph header with connector
        lines.append(f">> Subgraph #{counter}: {sg.node_name}")

        # Operation type and compute
        flops_str = f"{sg.flops / 1e9:.3f}G" if sg.flops > 1e9 else f"{sg.flops / 1e6:.1f}M"
        lines.append(f"   Type: {sg.operation_type.value}, FLOPs: {flops_str}")

        # Arithmetic intensity and bottleneck
        lines.append(f"   Arithmetic Intensity: {sg.arithmetic_intensity:.1f} FLOPs/byte, "
                    f"Bottleneck: {sg.recommended_bottleneck.value}")

        # Partition reason
        reason_line = f"   Partition: {sg.partition_reason.value}"
        if sg.fusion_candidates:
            candidates = sg.partition_criteria.get('fusion_candidates', [])
            if candidates:
                reason_line += f" ({len(candidates)} fusion candidates)"
        lines.append(reason_line)

        return lines

    def _format_not_partitioned(self, node) -> List[str]:
        """Format info for nodes that weren't partitioned"""
        lines = []
        lines.append("   (not partitioned)")

        if node.op == 'call_function':
            lines.append("   Reason: call_function not yet supported")
        elif node.op == 'call_method':
            lines.append("   Reason: call_method not yet supported")
        elif node.op in ['placeholder', 'get_attr', 'output']:
            lines.append("   Reason: graph structure node")

        return lines
