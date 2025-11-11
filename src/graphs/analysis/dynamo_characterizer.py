"""
Dynamo-Based Workload Characterizer

Uses PyTorch Dynamo (torch.compile) to extract and analyze computational graphs,
separating operations into MACs, FLOPs, and IntOps for accurate hardware mapping.

Key Features:
- Handles control flow and dynamic shapes (via Dynamo)
- Analyzes aten-level operations (lower-level than torch.nn.functional)
- Supports graph breaks (multiple subgraphs)
- Maps operations to hardware execution units

Usage:
    from graphs.analysis.dynamo_characterizer import characterize_with_dynamo

    model = MyModel()
    input_data = torch.randn(1, 256)
    workload = characterize_with_dynamo(model, input_data)
"""

import torch
import torch.fx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from graphs.analysis.workload_characterization import (
    WorkloadCharacterization,
    OperationBreakdown,
    create_simple_workload
)


# ============================================================
# ATEN OPERATION CLASSIFICATION
# ============================================================

@dataclass
class AtenOpInfo:
    """Information about an aten operation"""
    op_type: str  # 'mac', 'flop', 'intop', 'mixed'
    mac_formula: Optional[str] = None
    flop_formula: Optional[str] = None
    intop_formula: Optional[str] = None
    description: str = ""


# Comprehensive mapping of aten ops to operation types
ATEN_OP_CATALOG: Dict[str, AtenOpInfo] = {
    # ============================================================
    # MATRIX OPERATIONS (MACs)
    # ============================================================
    'aten.mm': AtenOpInfo(
        op_type='mac',
        mac_formula='M * K * N',
        description='Matrix multiplication [M×K] @ [K×N]'
    ),
    'aten.matmul': AtenOpInfo(
        op_type='mac',
        mac_formula='M * K * N',
        description='Matrix multiplication with broadcasting'
    ),
    'aten.bmm': AtenOpInfo(
        op_type='mac',
        mac_formula='B * M * K * N',
        description='Batched matrix multiplication'
    ),
    'aten.addmm': AtenOpInfo(
        op_type='mixed',
        mac_formula='M * K * N',
        flop_formula='M * N',
        description='Matrix multiply + bias: out = beta*bias + alpha*(A @ B)'
    ),
    'aten.baddbmm': AtenOpInfo(
        op_type='mixed',
        mac_formula='B * M * K * N',
        flop_formula='B * M * N',
        description='Batched matrix multiply + bias'
    ),
    'aten.linear': AtenOpInfo(
        op_type='mixed',
        mac_formula='M * K * N',
        flop_formula='M * N',
        description='Linear layer: out = input @ weight.T + bias (if bias exists)'
    ),

    # ============================================================
    # CONVOLUTION OPERATIONS (MACs)
    # ============================================================
    'aten.conv1d': AtenOpInfo(
        op_type='mac',
        mac_formula='Cout * L * Cin * K / groups',
        description='1D convolution'
    ),
    'aten.conv2d': AtenOpInfo(
        op_type='mac',
        mac_formula='Cout * H * W * Cin * Kh * Kw / groups',
        description='2D convolution'
    ),
    'aten.conv3d': AtenOpInfo(
        op_type='mac',
        mac_formula='Cout * D * H * W * Cin * Kd * Kh * Kw / groups',
        description='3D convolution'
    ),
    'aten.conv_transpose2d': AtenOpInfo(
        op_type='mac',
        mac_formula='Cout * H * W * Cin * Kh * Kw / groups',
        description='2D transposed convolution'
    ),

    # ============================================================
    # ACTIVATION FUNCTIONS (FLOPs)
    # ============================================================
    'aten.relu': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='ReLU activation (1 comparison per element)'
    ),
    'aten.relu_': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='In-place ReLU'
    ),
    'aten.gelu': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 8',
        description='GELU activation (~8 FLOPs per element)'
    ),
    'aten.hardtanh': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 2',
        description='Hard tanh activation (2 comparisons per element)'
    ),
    'aten.hardtanh_': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 2',
        description='In-place hard tanh activation'
    ),
    'aten.silu': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 5',
        description='SiLU/Swish activation (~5 FLOPs per element)'
    ),
    'aten.sigmoid': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Sigmoid activation (~4 FLOPs per element)'
    ),
    'aten.tanh': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Tanh activation (~4 FLOPs per element)'
    ),

    # ============================================================
    # NORMALIZATION (FLOPs)
    # ============================================================
    'aten.batch_norm': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Batch normalization (4 FLOPs per element)'
    ),
    'aten.layer_norm': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Layer normalization (4 FLOPs per element)'
    ),
    'aten.group_norm': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Group normalization (4 FLOPs per element)'
    ),
    'aten.instance_norm': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Instance normalization (4 FLOPs per element)'
    ),

    # ============================================================
    # SOFTMAX (FLOPs)
    # ============================================================
    'aten.softmax': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 10',
        description='Softmax (~10 FLOPs per element: exp + sum + div)'
    ),
    'aten._softmax': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 10',
        description='Internal softmax'
    ),
    'aten.log_softmax': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 12',
        description='Log softmax (~12 FLOPs per element)'
    ),

    # ============================================================
    # ELEMENT-WISE OPERATIONS (FLOPs)
    # ============================================================
    'aten.add': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='Element-wise addition'
    ),
    'aten.add_': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='In-place element-wise addition'
    ),
    'aten.iadd': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='In-place addition (iadd)'
    ),
    'aten.sub': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='Element-wise subtraction'
    ),
    'aten.mul': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='Element-wise multiplication'
    ),
    'aten.div': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 1',
        description='Element-wise division'
    ),
    'aten.pow': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Element-wise power (~4 FLOPs per element)'
    ),
    'aten.exp': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 8',
        description='Element-wise exponential (~8 FLOPs per element)'
    ),
    'aten.sqrt': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Element-wise square root (~4 FLOPs per element)'
    ),
    'aten.rsqrt': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 4',
        description='Reciprocal square root (~4 FLOPs per element)'
    ),

    # ============================================================
    # REDUCTION OPERATIONS (FLOPs)
    # ============================================================
    'aten.sum': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements - 1',
        description='Sum reduction (N-1 additions)'
    ),
    'aten.mean': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements',
        description='Mean reduction (sum + division)'
    ),
    'aten.var': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements * 3',
        description='Variance reduction (~3 FLOPs per element)'
    ),
    'aten.max': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements - 1',
        description='Max reduction (N-1 comparisons)'
    ),
    'aten.min': AtenOpInfo(
        op_type='flop',
        flop_formula='num_elements - 1',
        description='Min reduction (N-1 comparisons)'
    ),

    # ============================================================
    # QUANTIZATION OPERATIONS (IntOps)
    # ============================================================
    'aten.quantize_per_tensor': AtenOpInfo(
        op_type='intop',
        intop_formula='num_elements',
        description='Quantize FP → INT (per-tensor scale)'
    ),
    'aten.quantize_per_channel': AtenOpInfo(
        op_type='intop',
        intop_formula='num_elements',
        description='Quantize FP → INT (per-channel scale)'
    ),
    'aten.dequantize': AtenOpInfo(
        op_type='intop',
        intop_formula='num_elements',
        description='Dequantize INT → FP'
    ),

    # ============================================================
    # INDEXING OPERATIONS (IntOps)
    # ============================================================
    'aten.embedding': AtenOpInfo(
        op_type='intop',
        intop_formula='num_embeddings * embedding_dim',
        description='Embedding lookup'
    ),
    'aten.index': AtenOpInfo(
        op_type='intop',
        intop_formula='num_elements',
        description='Tensor indexing'
    ),
    'aten.index_select': AtenOpInfo(
        op_type='intop',
        intop_formula='num_elements',
        description='Index selection'
    ),

    # ============================================================
    # NO-OP / MEMORY OPERATIONS (0 compute)
    # ============================================================
    'aten.view': AtenOpInfo(
        op_type='noop',
        description='Reshape (no compute)'
    ),
    'aten.reshape': AtenOpInfo(
        op_type='noop',
        description='Reshape (no compute)'
    ),
    'aten.permute': AtenOpInfo(
        op_type='noop',
        description='Permute dimensions (no compute)'
    ),
    'aten.transpose': AtenOpInfo(
        op_type='noop',
        description='Transpose (no compute)'
    ),
    'aten.contiguous': AtenOpInfo(
        op_type='noop',
        description='Make contiguous (memory layout)'
    ),
    'aten.clone': AtenOpInfo(
        op_type='noop',
        description='Clone tensor (memory copy)'
    ),
}


class DynamoWorkloadCharacterizer:
    """
    Analyzes FX graphs produced by PyTorch Dynamo to extract workload characteristics.

    This characterizer walks through the computational graph captured by Dynamo
    and classifies operations into MACs, FLOPs, and IntOps based on their
    execution on different hardware units.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the characterizer.

        Args:
            verbose: If True, print detailed analysis during characterization
        """
        self.verbose = verbose
        self.aten_op_catalog = ATEN_OP_CATALOG

    def characterize_graph(
        self,
        fx_graph: torch.fx.Graph,
        example_inputs: Optional[List[torch.Tensor]] = None
    ) -> WorkloadCharacterization:
        """
        Analyze an FX graph and extract workload characteristics.

        Args:
            fx_graph: FX Graph produced by Dynamo
            example_inputs: Example inputs for shape inference (optional)

        Returns:
            WorkloadCharacterization with MAC/FLOP/IntOp counts
        """
        macs = 0
        flops = 0
        intops = 0
        breakdown = OperationBreakdown()
        bytes_transferred = 0

        for node in fx_graph.nodes:
            if node.op == 'call_function':
                # Extract operation name (e.g., aten.mm, aten.add)
                target_name = self._get_target_name(node.target)

                if target_name in self.aten_op_catalog:
                    op_info = self.aten_op_catalog[target_name]

                    # Count operations based on type
                    mac_count, flop_count, intop_count = self._count_operation(
                        node, op_info
                    )

                    macs += mac_count
                    flops += flop_count
                    intops += intop_count

                    # Update detailed breakdown
                    self._update_breakdown(
                        breakdown, target_name, mac_count, flop_count, intop_count
                    )

                    if self.verbose and (mac_count > 0 or flop_count > 0 or intop_count > 0):
                        print(f"{target_name}: MACs={mac_count:,}, FLOPs={flop_count:,}, IntOps={intop_count:,}")

                elif self.verbose:
                    print(f"Unknown aten op: {target_name}")

            # Estimate memory traffic from node
            bytes_transferred += self._estimate_memory_traffic(node)

        return WorkloadCharacterization(
            macs=macs,
            flops=flops,
            intops=intops,
            bytes_transferred=bytes_transferred,
            breakdown=breakdown,
            mac_precision='fp32',  # TODO: detect from graph
            flop_precision='fp32',
            intop_precision='int32'
        )

    def _get_target_name(self, target) -> str:
        """Extract operation name from target"""
        if hasattr(target, '__name__'):
            return f"aten.{target.__name__}"
        elif hasattr(target, '__module__') and hasattr(target, '__qualname__'):
            return f"{target.__module__}.{target.__qualname__}"
        else:
            return str(target)

    def _count_operation(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """
        Count MACs/FLOPs/IntOps for a node.

        Returns:
            (mac_count, flop_count, intop_count)
        """
        # Note: We need metadata to extract shapes, but we check in _get_shape()
        # rather than here, since 'tensor_meta' or 'example_value' may be present

        target_name = self._get_target_name(node.target)

        try:
            # Dispatch to operation-specific counter
            if 'linear' in target_name:
                return self._count_linear(node, op_info)
            elif 'mm' in target_name or 'matmul' in target_name or 'bmm' in target_name or 'addmm' in target_name:
                return self._count_matmul(node, op_info)
            elif 'conv' in target_name:
                return self._count_conv(node, op_info)
            elif target_name in ['aten.add', 'aten.sub', 'aten.mul', 'aten.div']:
                return self._count_elementwise(node, op_info)
            elif 'relu' in target_name or 'gelu' in target_name or 'silu' in target_name:
                return self._count_activation(node, op_info)
            elif 'norm' in target_name:
                return self._count_normalization(node, op_info)
            elif 'softmax' in target_name:
                return self._count_softmax(node, op_info)
            elif 'sum' in target_name or 'mean' in target_name or 'max' in target_name:
                return self._count_reduction(node, op_info)
            else:
                # Generic count based on op_info
                return self._count_generic(node, op_info)

        except Exception as e:
            if self.verbose:
                print(f"Error counting {target_name}: {e}")
            return (0, 0, 0)

    def _count_linear(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """
        Count linear layer operations.

        Linear: out = input @ weight.T + bias
        Args: (input, weight, bias?)
        Shapes: input=[M, K], weight=[N, K], output=[M, N]
        """
        args = node.args
        if len(args) < 2:
            return (0, 0, 0)

        # Extract shapes
        input_shape = self._get_shape(args[0])
        weight_shape = self._get_shape(args[1])

        if not input_shape or not weight_shape:
            return (0, 0, 0)

        # Input: [..., K], Weight: [N, K]
        # Output: [..., N]
        K = int(input_shape[-1])
        N = int(weight_shape[0])  # Out features

        # Calculate batch size (all dims except last)
        M = 1
        for dim in input_shape[:-1]:
            M *= int(dim)

        # MACs from matmul: M * K * N
        mac_count = int(M * K * N)

        # FLOPs from bias (if present)
        flop_count = 0
        has_bias = len(args) >= 3 and args[2] is not None
        if has_bias:
            flop_count = int(M * N)  # Bias addition

        return (mac_count, flop_count, 0)

    def _count_matmul(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Count matmul operations (M×K×N MACs)"""
        # Get input shapes
        args = node.args
        if len(args) < 2:
            return (0, 0, 0)

        # Extract shapes from metadata
        input0_shape = self._get_shape(args[0])
        input1_shape = self._get_shape(args[1])

        if not input0_shape or not input1_shape:
            return (0, 0, 0)

        # Calculate MACs (simplified for 2D matmul)
        M = int(input0_shape[-2]) if len(input0_shape) > 1 else 1
        K = int(input0_shape[-1])
        N = int(input1_shape[-1])

        # Handle batch dimensions
        batch_size = 1
        for dim in input0_shape[:-2]:
            batch_size *= int(dim)

        mac_count = int(batch_size * M * K * N)

        # Check if this is addmm (matmul + bias)
        target_name = self._get_target_name(node.target)
        if 'addmm' in target_name or 'baddbmm' in target_name:
            flop_count = int(batch_size * M * N)  # Bias addition
            return (mac_count, flop_count, 0)
        else:
            return (mac_count, 0, 0)

    def _count_conv(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """
        Count convolution operations.

        Conv2D MAC formula:
        MACs = B × C_out × H_out × W_out × (C_in × K_h × K_w) / groups

        Where:
        - Output dims: H_out = (H_in + 2*P - K_h) / stride + 1
        - Input shape: [B, C_in, H_in, W_in]
        - Weight shape: [C_out, C_in/groups, K_h, K_w]
        """
        args = node.args

        # Extract shapes
        input_shape = self._get_shape(args[0])
        weight_shape = self._get_shape(args[1])

        if not input_shape or not weight_shape:
            return (0, 0, 0)

        # Extract convolution parameters
        # Input: [B, C_in, H_in, W_in]
        B = int(input_shape[0])
        C_in = int(input_shape[1])
        H_in = int(input_shape[2])
        W_in = int(input_shape[3])

        # Weight: [C_out, C_in/groups, K_h, K_w]
        C_out = int(weight_shape[0])
        K_h = int(weight_shape[2])
        K_w = int(weight_shape[3])

        # Get stride, padding, dilation, groups
        # torch.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        # They can be in args (positional) or kwargs (keyword)
        kwargs = node.kwargs

        # Try to get from args first (positions 3, 4, 5, 6)
        stride = [1, 1]
        padding = [0, 0]
        dilation = [1, 1]
        groups = 1

        if len(args) > 3:  # stride is args[3]
            stride = args[3]
        elif 'stride' in kwargs:
            stride = kwargs['stride']

        if len(args) > 4:  # padding is args[4]
            padding = args[4]
        elif 'padding' in kwargs:
            padding = kwargs['padding']

        if len(args) > 5:  # dilation is args[5]
            dilation = args[5]
        elif 'dilation' in kwargs:
            dilation = kwargs['dilation']

        if len(args) > 6:  # groups is args[6]
            groups = args[6]
        elif 'groups' in kwargs:
            groups = kwargs['groups']

        # Handle scalar stride/padding (convert to list)
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]

        stride_h, stride_w = int(stride[0]), int(stride[1])
        padding_h, padding_w = int(padding[0]), int(padding[1])
        dilation_h, dilation_w = int(dilation[0]), int(dilation[1])
        groups = int(groups)

        # Calculate output spatial dimensions
        # H_out = (H_in + 2*P - D*(K-1) - 1) / S + 1
        K_h_dilated = dilation_h * (K_h - 1) + 1
        K_w_dilated = dilation_w * (K_w - 1) + 1

        H_out = (H_in + 2*padding_h - K_h_dilated) // stride_h + 1
        W_out = (W_in + 2*padding_w - K_w_dilated) // stride_w + 1

        # Calculate MACs
        # Each output element requires (C_in/groups × K_h × K_w) MACs
        macs_per_output = (C_in // groups) * K_h * K_w
        total_outputs = B * C_out * H_out * W_out
        mac_count = int(total_outputs * macs_per_output)

        # Check for bias
        flop_count = 0
        has_bias = len(args) >= 3 and args[2] is not None
        if has_bias:
            # Bias addition: one add per output element
            flop_count = int(total_outputs)

        return (mac_count, flop_count, 0)

    def _count_elementwise(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Count element-wise operations"""
        output_shape = self._get_output_shape(node)
        if not output_shape:
            return (0, 0, 0)

        num_elements = 1
        for dim in output_shape:
            num_elements *= dim

        return (0, num_elements, 0)  # Element-wise = FLOPs

    def _count_activation(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Count activation function FLOPs"""
        output_shape = self._get_output_shape(node)
        if not output_shape:
            return (0, 0, 0)

        num_elements = 1
        for dim in output_shape:
            num_elements *= dim

        # Extract FLOPs per element from op_info
        target_name = self._get_target_name(node.target)
        flops_per_element = {
            'aten.relu': 1,
            'aten.relu_': 1,
            'aten.gelu': 8,
            'aten.silu': 5,
            'aten.sigmoid': 4,
            'aten.tanh': 4,
            'aten.hardtanh': 2,
            'aten.hardtanh_': 2,
        }.get(target_name, 1)

        return (0, num_elements * flops_per_element, 0)

    def _count_normalization(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Count normalization FLOPs (~4 per element)"""
        output_shape = self._get_output_shape(node)
        if not output_shape:
            return (0, 0, 0)

        num_elements = 1
        for dim in output_shape:
            num_elements *= dim

        return (0, num_elements * 4, 0)

    def _count_softmax(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Count softmax FLOPs (~10 per element)"""
        output_shape = self._get_output_shape(node)
        if not output_shape:
            return (0, 0, 0)

        num_elements = 1
        for dim in output_shape:
            num_elements *= dim

        return (0, num_elements * 10, 0)

    def _count_reduction(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Count reduction FLOPs"""
        # Simplified: assume num_elements - 1
        input_shape = self._get_shape(node.args[0]) if node.args else None
        if not input_shape:
            return (0, 0, 0)

        num_elements = 1
        for dim in input_shape:
            num_elements *= dim

        return (0, max(1, num_elements - 1), 0)

    def _count_generic(
        self,
        node: torch.fx.Node,
        op_info: AtenOpInfo
    ) -> Tuple[int, int, int]:
        """Generic counter based on op_info"""
        return (0, 0, 0)  # Default: no ops

    def _get_shape(self, arg) -> Optional[List[int]]:
        """Extract shape from argument"""
        if isinstance(arg, torch.fx.Node):
            if hasattr(arg, 'meta'):
                # Try 'example_value' first (Dynamo metadata)
                if 'example_value' in arg.meta:
                    ex_val = arg.meta['example_value']
                    if hasattr(ex_val, 'shape'):
                        return list(ex_val.shape)
                # Fall back to 'tensor_meta' (older FX metadata)
                elif 'tensor_meta' in arg.meta:
                    return list(arg.meta['tensor_meta'].shape)
        elif isinstance(arg, torch.Tensor):
            return list(arg.shape)
        return None

    def _get_output_shape(self, node: torch.fx.Node) -> Optional[List[int]]:
        """Extract output shape from node"""
        if hasattr(node, 'meta'):
            # Try 'example_value' first (Dynamo metadata)
            if 'example_value' in node.meta:
                ex_val = node.meta['example_value']
                if hasattr(ex_val, 'shape'):
                    return list(ex_val.shape)
            # Fall back to 'tensor_meta' (older FX metadata)
            elif 'tensor_meta' in node.meta:
                return list(node.meta['tensor_meta'].shape)
        return None

    def _estimate_memory_traffic(self, node: torch.fx.Node) -> int:
        """Estimate bytes transferred for a node"""
        # Simplified: sum of input and output bytes
        bytes_total = 0

        # Output bytes
        output_shape = self._get_output_shape(node)
        if output_shape:
            num_elements = 1
            for dim in output_shape:
                num_elements *= dim
            bytes_total += num_elements * 4  # Assume FP32

        return bytes_total

    def _update_breakdown(
        self,
        breakdown: OperationBreakdown,
        target_name: str,
        mac_count: int,
        flop_count: int,
        intop_count: int
    ):
        """Update detailed operation breakdown"""
        # Map aten ops to breakdown fields
        if 'linear' in target_name:
            breakdown.matmul_macs += mac_count
            if flop_count > 0:
                breakdown.bias_flops += flop_count
        elif 'mm' in target_name or 'matmul' in target_name:
            breakdown.matmul_macs += mac_count
        elif 'bmm' in target_name:
            breakdown.bmm_macs += mac_count
        elif 'conv2d' in target_name:
            breakdown.conv2d_macs += mac_count
        elif 'addmm' in target_name:
            breakdown.matmul_macs += mac_count
            breakdown.bias_flops += flop_count
        elif 'relu' in target_name:
            breakdown.relu_flops += flop_count
        elif 'gelu' in target_name:
            breakdown.gelu_flops += flop_count
        elif 'silu' in target_name:
            breakdown.silu_flops += flop_count
        elif 'sigmoid' in target_name:
            breakdown.sigmoid_flops += flop_count
        elif 'hardtanh' in target_name:
            breakdown.hardtanh_flops += flop_count
        elif 'tanh' in target_name and 'hardtanh' not in target_name:
            breakdown.tanh_flops += flop_count
        elif 'batch_norm' in target_name:
            breakdown.batchnorm_flops += flop_count
        elif 'layer_norm' in target_name:
            breakdown.layernorm_flops += flop_count
        elif 'softmax' in target_name:
            breakdown.softmax_flops += flop_count
        elif target_name in ['aten.add', 'aten.add_', 'aten.iadd']:
            breakdown.elementwise_add_flops += flop_count
        elif target_name == 'aten.sub':
            breakdown.elementwise_sub_flops += flop_count
        elif target_name == 'aten.mul':
            breakdown.elementwise_mul_flops += flop_count
        elif target_name == 'aten.div':
            breakdown.elementwise_div_flops += flop_count
        else:
            # Catch any uncategorized operations
            if mac_count > 0:
                breakdown.other_macs += mac_count
            if flop_count > 0:
                breakdown.other_flops += flop_count
            if intop_count > 0:
                breakdown.other_intops += intop_count


def characterize_with_dynamo(
    model: torch.nn.Module,
    example_input,  # Can be Tensor or tuple/list of Tensors
    verbose: bool = False
) -> WorkloadCharacterization:
    """
    Characterize a model using Dynamo graph capture.

    Args:
        model: PyTorch model to characterize
        example_input: Example input(s) for tracing (Tensor or tuple of Tensors)
        verbose: Print detailed analysis

    Returns:
        WorkloadCharacterization

    Example:
        >>> model = torch.nn.Linear(256, 256)
        >>> input_data = torch.randn(1, 256)
        >>> workload = characterize_with_dynamo(model, input_data)
        >>> print(f"MACs: {workload.macs}, FLOPs: {workload.flops}")
    """
    characterizer = DynamoWorkloadCharacterizer(verbose=verbose)
    captured_workload = None

    def characterizer_backend(gm: torch.fx.GraphModule, example_inputs):
        """Custom Dynamo backend that captures workload"""
        nonlocal captured_workload
        captured_workload = characterizer.characterize_graph(gm.graph, example_inputs)
        return gm  # Return original for execution

    # Compile with custom backend
    compiled_model = torch.compile(model, backend=characterizer_backend)

    # Run to trigger graph capture
    with torch.no_grad():
        if isinstance(example_input, (tuple, list)):
            _ = compiled_model(*example_input)
        else:
            _ = compiled_model(example_input)

    if captured_workload is None:
        # Fallback: create empty workload
        captured_workload = create_simple_workload(0, 0, 0)

    return captured_workload
