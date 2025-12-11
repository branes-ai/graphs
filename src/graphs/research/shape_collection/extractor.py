"""
Tensor Shape Extractor

Extract tensor shapes from FX-traced DNN models and convert to matmul dimensions
(M, K, N) for systolic array utilization analysis.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp


@dataclass
class TensorShapeRecord:
    """
    Single tensor shape record with full metadata.

    Captures both raw tensor shapes and derived matmul dimensions (M, K, N)
    for systolic array mapping.
    """
    # Model identification
    model_name: str
    model_class: str  # CNN, Encoder, Decoder, FullTransformer

    # Layer identification
    layer_name: str
    layer_index: int
    op_type: str  # conv2d, linear, matmul, attention_qkv, layernorm, etc.

    # Input tensor shape
    input_shape: Tuple[int, ...]  # (N, C, H, W) for conv, (B, S, D) for transformer
    input_dtype: str

    # Output tensor shape
    output_shape: Tuple[int, ...]
    output_dtype: str

    # Weight tensor shape (if applicable)
    weight_shape: Optional[Tuple[int, ...]] = None

    # Derived matmul dimensions for systolic array mapping
    # For Conv2D: M = B * H_out * W_out, K = C_in * K_h * K_w, N = C_out
    # For Linear: M = B * seq_len (or batch), K = in_features, N = out_features
    M: int = 0  # Output rows (batch * spatial or batch * seq_len)
    K: int = 0  # Reduction dimension (input channels * kernel or hidden_dim)
    N: int = 0  # Output columns (output channels or output features)

    # Computation metrics
    flops: int = 0
    macs: int = 0

    # Memory metrics (bytes)
    input_bytes: int = 0
    weight_bytes: int = 0
    output_bytes: int = 0

    # Precision
    precision: str = 'float32'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'model_name': self.model_name,
            'model_class': self.model_class,
            'layer_name': self.layer_name,
            'layer_index': self.layer_index,
            'op_type': self.op_type,
            'input_shape': str(self.input_shape),
            'output_shape': str(self.output_shape),
            'weight_shape': str(self.weight_shape) if self.weight_shape else '',
            'M': self.M,
            'K': self.K,
            'N': self.N,
            'flops': self.flops,
            'macs': self.macs,
            'input_bytes': self.input_bytes,
            'weight_bytes': self.weight_bytes,
            'output_bytes': self.output_bytes,
            'precision': self.precision,
            'input_dtype': self.input_dtype,
            'output_dtype': self.output_dtype,
        }


def _get_dtype_bytes(dtype: torch.dtype) -> int:
    """Get bytes per element for dtype."""
    dtype_sizes = {
        torch.float64: 8,
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return dtype_sizes.get(dtype, 4)


def _dtype_to_string(dtype: torch.dtype) -> str:
    """Convert torch dtype to string."""
    dtype_names = {
        torch.float64: 'float64',
        torch.float32: 'float32',
        torch.float16: 'float16',
        torch.bfloat16: 'bfloat16',
        torch.int64: 'int64',
        torch.int32: 'int32',
        torch.int16: 'int16',
        torch.int8: 'int8',
        torch.uint8: 'uint8',
        torch.bool: 'bool',
    }
    return dtype_names.get(dtype, 'float32')


class ShapeExtractor:
    """
    Extract tensor shapes from FX-traced models.

    Supports:
    - Conv2D layers (all variants: regular, depthwise, grouped)
    - Linear/Dense layers
    - Attention operations (Q, K, V projections, output projection)
    - Normalization layers (LayerNorm, BatchNorm)
    - Pooling layers
    - Activation functions
    """

    # Operations that map to matmul-like compute on systolic arrays
    MATMUL_OPS = {'conv2d', 'linear', 'matmul', 'bmm', 'attention_qkv', 'attention_proj'}

    def __init__(self):
        self.records: List[TensorShapeRecord] = []

    def extract_from_model(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        model_name: str,
        model_class: str,
    ) -> List[TensorShapeRecord]:
        """
        Trace model and extract all tensor shapes.

        Args:
            model: PyTorch model to analyze
            input_tensor: Example input tensor
            model_name: Name of the model (e.g., 'resnet18')
            model_class: DNN class (CNN, Encoder, Decoder, FullTransformer)

        Returns:
            List of TensorShapeRecord for each layer
        """
        self.records = []
        model.eval()

        # FX symbolic trace
        try:
            traced = torch.fx.symbolic_trace(model)
        except Exception as e:
            # Try with torch.compile for models that don't trace directly
            try:
                traced = torch.compile(model, backend='eager')
                traced = torch.fx.symbolic_trace(traced)
            except Exception:
                raise RuntimeError(f"Failed to trace model {model_name}: {e}")

        # Shape propagation
        ShapeProp(traced).propagate(input_tensor)

        # Extract shapes from each node
        layer_index = 0
        for node in traced.graph.nodes:
            if node.op == 'call_module':
                record = self._extract_module_shape(
                    node, traced, model_name, model_class, layer_index
                )
                if record:
                    self.records.append(record)
                    layer_index += 1
            elif node.op == 'call_function':
                record = self._extract_function_shape(
                    node, traced, model_name, model_class, layer_index
                )
                if record:
                    self.records.append(record)
                    layer_index += 1

        return self.records

    def _get_shape(self, meta) -> Tuple[int, ...]:
        """Safely extract shape from tensor metadata."""
        def safe_int(d):
            """Convert dimension to int, handling SymInt and other types."""
            if d is None:
                return 1
            if isinstance(d, int):
                return d
            # Handle torch.SymInt or other symbolic types
            try:
                return int(d)
            except (TypeError, ValueError):
                # If we can't convert, use a default
                return 1

        if hasattr(meta, 'shape'):
            return tuple(safe_int(d) for d in meta.shape)
        elif isinstance(meta, tuple):
            return tuple(safe_int(d) for d in meta)
        else:
            return (1,)

    def _get_dtype(self, meta) -> torch.dtype:
        """Extract dtype from tensor metadata."""
        if hasattr(meta, 'dtype'):
            return meta.dtype
        return torch.float32

    def _extract_module_shape(
        self,
        node,
        graph: GraphModule,
        model_name: str,
        model_class: str,
        layer_index: int,
    ) -> Optional[TensorShapeRecord]:
        """Extract shape information from a call_module node."""

        # Get module
        try:
            module = graph.get_submodule(node.target)
        except AttributeError:
            return None

        # Get tensor metadata
        meta = node.meta.get('tensor_meta')
        if meta is None:
            return None

        # Get input metadata
        if node.args:
            input_node = node.args[0]
            input_meta = input_node.meta.get('tensor_meta') if hasattr(input_node, 'meta') else None
        else:
            input_meta = None

        output_shape = self._get_shape(meta)
        output_dtype = self._get_dtype(meta)

        if input_meta:
            input_shape = self._get_shape(input_meta)
            input_dtype = self._get_dtype(input_meta)
        else:
            input_shape = output_shape
            input_dtype = output_dtype

        # Determine operation type and extract matmul dimensions
        op_type, M, K, N, weight_shape, flops, macs = self._analyze_module(
            module, input_shape, output_shape
        )

        # Skip trivial operations (no matmul mapping)
        if op_type in {'identity', 'flatten', 'dropout'}:
            return None

        # Calculate memory
        elem_size = _get_dtype_bytes(input_dtype)
        input_bytes = math.prod(input_shape) * elem_size
        output_bytes = math.prod(output_shape) * elem_size
        weight_bytes = math.prod(weight_shape) * elem_size if weight_shape else 0

        return TensorShapeRecord(
            model_name=model_name,
            model_class=model_class,
            layer_name=node.target,
            layer_index=layer_index,
            op_type=op_type,
            input_shape=input_shape,
            input_dtype=_dtype_to_string(input_dtype),
            output_shape=output_shape,
            output_dtype=_dtype_to_string(output_dtype),
            weight_shape=weight_shape,
            M=M,
            K=K,
            N=N,
            flops=flops,
            macs=macs,
            input_bytes=input_bytes,
            weight_bytes=weight_bytes,
            output_bytes=output_bytes,
            precision=_dtype_to_string(input_dtype),
        )

    def _analyze_module(
        self,
        module: nn.Module,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Optional[Tuple[int, ...]], int, int]:
        """
        Analyze module and extract matmul dimensions.

        Returns:
            (op_type, M, K, N, weight_shape, flops, macs)
        """
        # Conv2d
        if isinstance(module, nn.Conv2d):
            return self._analyze_conv2d(module, input_shape, output_shape)

        # Linear
        elif isinstance(module, nn.Linear):
            return self._analyze_linear(module, input_shape, output_shape)

        # BatchNorm
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            return self._analyze_batchnorm(module, input_shape, output_shape)

        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            return self._analyze_layernorm(module, input_shape, output_shape)

        # Pooling
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return self._analyze_pooling(module, input_shape, output_shape)

        # Activation
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
            return self._analyze_activation(module, input_shape, output_shape)

        # Embedding
        elif isinstance(module, nn.Embedding):
            return self._analyze_embedding(module, input_shape, output_shape)

        # MultiheadAttention
        elif isinstance(module, nn.MultiheadAttention):
            return self._analyze_multihead_attention(module, input_shape, output_shape)

        # Dropout - skip
        elif isinstance(module, nn.Dropout):
            return ('dropout', 0, 0, 0, None, 0, 0)

        # Identity - skip
        elif isinstance(module, nn.Identity):
            return ('identity', 0, 0, 0, None, 0, 0)

        # Flatten - skip
        elif isinstance(module, nn.Flatten):
            return ('flatten', 0, 0, 0, None, 0, 0)

        else:
            # Unknown module - return basic info
            return ('unknown', 0, 0, 0, None, 0, 0)

    def _analyze_conv2d(
        self,
        module: nn.Conv2d,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Tuple[int, ...], int, int]:
        """
        Analyze Conv2d and extract matmul dimensions.

        Conv2d can be viewed as im2col + matmul:
        - M = B * H_out * W_out (output spatial locations)
        - K = C_in/groups * K_h * K_w (filter size)
        - N = C_out (number of filters)
        """
        B = input_shape[0] if len(input_shape) >= 1 else 1
        C_in = input_shape[1] if len(input_shape) >= 2 else module.in_channels
        H_in = input_shape[2] if len(input_shape) >= 3 else 1
        W_in = input_shape[3] if len(input_shape) >= 4 else 1

        C_out = module.out_channels
        K_h, K_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        S_h, S_w = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        P_h, P_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        groups = module.groups

        # Output spatial dimensions
        H_out = (H_in + 2 * P_h - K_h) // S_h + 1
        W_out = (W_in + 2 * P_w - K_w) // S_w + 1

        # Matmul dimensions
        M = B * H_out * W_out
        K = (C_in // groups) * K_h * K_w
        N = C_out

        # Weight shape
        weight_shape = (C_out, C_in // groups, K_h, K_w)

        # MACs and FLOPs
        macs = B * C_out * H_out * W_out * (C_in // groups) * K_h * K_w
        flops = 2 * macs

        # Determine specific conv type
        if groups == C_in and groups == C_out:
            op_type = 'conv2d_depthwise'
        elif groups > 1:
            op_type = 'conv2d_grouped'
        else:
            op_type = 'conv2d'

        return (op_type, M, K, N, weight_shape, flops, macs)

    def _analyze_linear(
        self,
        module: nn.Linear,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Tuple[int, ...], int, int]:
        """
        Analyze Linear layer and extract matmul dimensions.

        Linear: Y = X @ W.T + b
        - M = batch_size * sequence_length (or just batch for CNNs)
        - K = in_features
        - N = out_features
        """
        # M is product of all dimensions except the last
        M = math.prod(input_shape[:-1]) if len(input_shape) > 1 else input_shape[0]
        K = module.in_features
        N = module.out_features

        weight_shape = (N, K)

        macs = M * K * N
        flops = 2 * macs

        return ('linear', M, K, N, weight_shape, flops, macs)

    def _analyze_batchnorm(
        self,
        module: nn.Module,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Optional[Tuple[int, ...]], int, int]:
        """Analyze BatchNorm - element-wise operations."""
        total_elements = math.prod(input_shape)
        # BatchNorm: ~4 ops per element (sub mean, div std, scale, shift)
        flops = 4 * total_elements
        return ('batchnorm', 0, 0, 0, None, flops, 0)

    def _analyze_layernorm(
        self,
        module: nn.LayerNorm,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Optional[Tuple[int, ...]], int, int]:
        """Analyze LayerNorm - reduction + element-wise."""
        total_elements = math.prod(input_shape)
        normalized_shape = module.normalized_shape
        if isinstance(normalized_shape, int):
            norm_elements = normalized_shape
        else:
            norm_elements = math.prod(normalized_shape)

        # LayerNorm: mean reduction + variance reduction + normalize + scale/shift
        flops = 5 * total_elements
        return ('layernorm', 0, 0, 0, None, flops, 0)

    def _analyze_pooling(
        self,
        module: nn.Module,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Optional[Tuple[int, ...]], int, int]:
        """Analyze pooling layers."""
        # Get kernel size
        if hasattr(module, 'kernel_size'):
            ks = module.kernel_size
            if isinstance(ks, tuple):
                k_h, k_w = ks
            else:
                k_h = k_w = ks
        else:
            # Adaptive pooling
            k_h = k_w = input_shape[2] if len(input_shape) > 2 else 1

        output_elements = math.prod(output_shape)
        flops = output_elements * k_h * k_w

        if isinstance(module, nn.MaxPool2d):
            return ('maxpool2d', 0, 0, 0, None, flops, 0)
        elif isinstance(module, nn.AvgPool2d):
            return ('avgpool2d', 0, 0, 0, None, flops, 0)
        else:
            return ('adaptive_avgpool', 0, 0, 0, None, flops, 0)

    def _analyze_activation(
        self,
        module: nn.Module,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Optional[Tuple[int, ...]], int, int]:
        """Analyze activation functions."""
        total_elements = math.prod(input_shape)

        if isinstance(module, nn.ReLU):
            op_type = 'relu'
            flops = total_elements  # 1 comparison per element
        elif isinstance(module, nn.GELU):
            op_type = 'gelu'
            flops = 8 * total_elements  # Approximation uses ~8 ops
        elif isinstance(module, nn.SiLU):
            op_type = 'silu'
            flops = 4 * total_elements  # sigmoid + multiply
        elif isinstance(module, nn.Sigmoid):
            op_type = 'sigmoid'
            flops = 4 * total_elements
        elif isinstance(module, nn.Tanh):
            op_type = 'tanh'
            flops = 6 * total_elements
        elif isinstance(module, nn.Softmax):
            op_type = 'softmax'
            flops = 5 * total_elements  # exp, sum, div
        else:
            op_type = 'activation'
            flops = total_elements

        return (op_type, 0, 0, 0, None, flops, 0)

    def _analyze_embedding(
        self,
        module: nn.Embedding,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Tuple[int, ...], int, int]:
        """Analyze Embedding layer - lookup table."""
        num_lookups = math.prod(input_shape)
        embedding_dim = module.embedding_dim

        # No FLOPs, just memory access
        weight_shape = (module.num_embeddings, embedding_dim)
        flops = 0

        return ('embedding', 0, 0, 0, weight_shape, flops, 0)

    def _analyze_multihead_attention(
        self,
        module: nn.MultiheadAttention,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[str, int, int, int, Optional[Tuple[int, ...]], int, int]:
        """
        Analyze MultiheadAttention.

        MHA consists of:
        1. Q, K, V projections (3 linear layers)
        2. Attention scores (Q @ K.T)
        3. Softmax
        4. Weighted sum (scores @ V)
        5. Output projection
        """
        # Assume input is (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim)
        if len(input_shape) >= 3:
            if input_shape[0] < input_shape[1]:  # (seq_len, batch, dim)
                seq_len = input_shape[0]
                batch = input_shape[1]
            else:  # (batch, seq_len, dim)
                batch = input_shape[0]
                seq_len = input_shape[1]
            embed_dim = input_shape[2]
        else:
            seq_len = input_shape[0] if len(input_shape) >= 1 else 1
            batch = 1
            embed_dim = module.embed_dim

        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        # Total FLOPs for MHA
        # Q, K, V projections: 3 * (batch * seq_len * embed_dim * embed_dim)
        proj_flops = 3 * 2 * batch * seq_len * embed_dim * embed_dim

        # Attention scores: batch * num_heads * seq_len * seq_len * head_dim
        attn_flops = 2 * batch * num_heads * seq_len * seq_len * head_dim

        # Softmax: ~5 * batch * num_heads * seq_len * seq_len
        softmax_flops = 5 * batch * num_heads * seq_len * seq_len

        # Weighted sum: batch * num_heads * seq_len * seq_len * head_dim
        weighted_flops = 2 * batch * num_heads * seq_len * head_dim * seq_len

        # Output projection: batch * seq_len * embed_dim * embed_dim
        out_proj_flops = 2 * batch * seq_len * embed_dim * embed_dim

        total_flops = proj_flops + attn_flops + softmax_flops + weighted_flops + out_proj_flops

        # For systolic mapping, use Q/K/V projection dimensions
        M = batch * seq_len
        K = embed_dim
        N = embed_dim

        return ('multihead_attention', M, K, N, None, total_flops, total_flops // 2)

    def _extract_function_shape(
        self,
        node,
        graph: GraphModule,
        model_name: str,
        model_class: str,
        layer_index: int,
    ) -> Optional[TensorShapeRecord]:
        """Extract shape information from a call_function node."""

        # Get tensor metadata
        meta = node.meta.get('tensor_meta')
        if meta is None:
            return None

        output_shape = self._get_shape(meta)
        output_dtype = self._get_dtype(meta)

        # Get input metadata
        if node.args:
            input_node = node.args[0]
            input_meta = input_node.meta.get('tensor_meta') if hasattr(input_node, 'meta') else None
        else:
            input_meta = None

        if input_meta:
            input_shape = self._get_shape(input_meta)
            input_dtype = self._get_dtype(input_meta)
        else:
            input_shape = output_shape
            input_dtype = output_dtype

        # Determine function type
        func_name = str(node.target)

        # Skip common non-compute operations
        skip_ops = {'getattr', 'getitem', 'cat', 'chunk', 'split', 'view',
                    'reshape', 'transpose', 'permute', 'contiguous', 'flatten'}
        for skip in skip_ops:
            if skip in func_name.lower():
                return None

        # Matmul operations
        if 'matmul' in func_name.lower() or 'bmm' in func_name.lower():
            return self._extract_matmul_shape(
                node, input_shape, output_shape, input_dtype, output_dtype,
                model_name, model_class, layer_index
            )

        # Add operations (element-wise)
        if 'add' in func_name.lower():
            total_elements = math.prod(output_shape)
            elem_size = _get_dtype_bytes(input_dtype)
            return TensorShapeRecord(
                model_name=model_name,
                model_class=model_class,
                layer_name=f"add_{layer_index}",
                layer_index=layer_index,
                op_type='add',
                input_shape=input_shape,
                input_dtype=_dtype_to_string(input_dtype),
                output_shape=output_shape,
                output_dtype=_dtype_to_string(output_dtype),
                M=0, K=0, N=0,
                flops=total_elements,
                macs=0,
                input_bytes=math.prod(input_shape) * elem_size,
                output_bytes=math.prod(output_shape) * elem_size,
                precision=_dtype_to_string(input_dtype),
            )

        # Multiply operations
        if 'mul' in func_name.lower():
            total_elements = math.prod(output_shape)
            elem_size = _get_dtype_bytes(input_dtype)
            return TensorShapeRecord(
                model_name=model_name,
                model_class=model_class,
                layer_name=f"mul_{layer_index}",
                layer_index=layer_index,
                op_type='mul',
                input_shape=input_shape,
                input_dtype=_dtype_to_string(input_dtype),
                output_shape=output_shape,
                output_dtype=_dtype_to_string(output_dtype),
                M=0, K=0, N=0,
                flops=total_elements,
                macs=0,
                input_bytes=math.prod(input_shape) * elem_size,
                output_bytes=math.prod(output_shape) * elem_size,
                precision=_dtype_to_string(input_dtype),
            )

        return None

    def _extract_matmul_shape(
        self,
        node,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        model_name: str,
        model_class: str,
        layer_index: int,
    ) -> TensorShapeRecord:
        """Extract matmul/bmm shape information."""

        # Get both input shapes
        if len(node.args) >= 2:
            arg1 = node.args[0]
            arg2 = node.args[1]

            meta1 = arg1.meta.get('tensor_meta') if hasattr(arg1, 'meta') else None
            meta2 = arg2.meta.get('tensor_meta') if hasattr(arg2, 'meta') else None

            if meta1 and meta2:
                shape1 = self._get_shape(meta1)
                shape2 = self._get_shape(meta2)

                # For matmul A @ B:
                # A: (..., M, K), B: (..., K, N) -> (..., M, N)
                if len(shape1) >= 2 and len(shape2) >= 2:
                    M = shape1[-2]
                    K = shape1[-1]
                    N = shape2[-1]

                    # Account for batch dimensions
                    batch = math.prod(shape1[:-2]) if len(shape1) > 2 else 1
                    M = batch * M
                else:
                    M = shape1[-1] if shape1 else 1
                    K = 1
                    N = shape2[-1] if shape2 else 1
            else:
                M = K = N = 1
        else:
            M = K = N = 1

        macs = M * K * N
        flops = 2 * macs

        elem_size = _get_dtype_bytes(input_dtype)

        return TensorShapeRecord(
            model_name=model_name,
            model_class=model_class,
            layer_name=f"matmul_{layer_index}",
            layer_index=layer_index,
            op_type='matmul',
            input_shape=input_shape,
            input_dtype=_dtype_to_string(input_dtype),
            output_shape=output_shape,
            output_dtype=_dtype_to_string(output_dtype),
            M=M, K=K, N=N,
            flops=flops,
            macs=macs,
            input_bytes=math.prod(input_shape) * elem_size,
            output_bytes=math.prod(output_shape) * elem_size,
            precision=_dtype_to_string(input_dtype),
        )
