"""
Architectural Modifiers for Operator-Level EDP (Phase 2)

This module defines architecture-specific energy/delay modifiers for different
operator types, revealing how fusion and architectural characteristics impact
per-operator EDP.

Key Insight:
- CPU/GPU (STORED_PROGRAM, DATA_PARALLEL): Lightweight ops (ReLU, Bias) incur
  separate kernel overhead (~3.0× modifier)
- TPU/KPU (SYSTOLIC_ARRAY, DOMAIN_FLOW): Lightweight ops hidden in dataflow
  (~0.05× modifier)

Example:
    Linear→Bias→ReLU on KPU (fused):
      Linear: 95.0% (1.0× modifier - dominates)
      Bias:    2.5% (0.05× modifier - hidden in dataflow)
      ReLU:    2.5% (0.05× modifier - hidden in dataflow)

    Same on GPU (separate kernels):
      Linear: 32% (1.0× modifier)
      Bias:   34% (3.0× modifier - separate kernel overhead)
      ReLU:   34% (3.0× modifier - separate kernel overhead)
"""

from enum import Enum
from typing import Dict, Tuple
from graphs.hardware.architectural_energy import ArchitectureClass


# ==============================================================================
# Architectural Modifier Tables
# ==============================================================================

# Modifier format: (fused_modifier, separate_modifier)
# - fused_modifier: When operator is fused with others in subgraph
# - separate_modifier: When operator executes as separate kernel

ARCHITECTURAL_MODIFIERS: Dict[ArchitectureClass, Dict[str, Tuple[float, float]]] = {
    # -------------------------------------------------------------------------
    # STORED_PROGRAM (CPU): Multi-core + SIMD
    # -------------------------------------------------------------------------
    ArchitectureClass.STORED_PROGRAM: {
        # Compute-intensive ops (baseline)
        "Linear": (1.0, 1.0),       # Dominates in both scenarios
        "addmm": (1.0, 1.0),        # Matrix multiply (Linear backend)
        "matmul": (1.0, 1.0),       # General matrix multiply
        "bmm": (1.0, 1.0),          # Batch matrix multiply
        "Conv2d": (1.0, 1.0),       # Convolution dominates
        "conv2d": (1.0, 1.0),       # aten::conv2d

        # Lightweight ops (separate kernel overhead on CPU)
        "Bias": (0.8, 3.0),         # Fused: moderate, Separate: kernel overhead
        "add": (0.5, 3.0),          # Elementwise add (includes bias, residual)
        "ReLU": (0.5, 3.0),         # Activation function
        "relu": (0.5, 3.0),         # aten::relu
        "GELU": (0.8, 3.0),         # More complex activation
        "gelu": (0.8, 3.0),         # aten::gelu
        "Sigmoid": (0.8, 3.0),      # Activation
        "sigmoid": (0.8, 3.0),      # aten::sigmoid
        "Tanh": (0.8, 3.0),         # Activation
        "tanh": (0.8, 3.0),         # aten::tanh

        # Normalization ops (memory-bound, moderate fusion benefit)
        "BatchNorm2d": (1.0, 1.5),  # Fused with Conv: baseline, Separate: moderate overhead
        "batch_norm": (1.0, 1.5),   # aten::batch_norm
        "LayerNorm": (1.0, 1.5),    # Similar to BatchNorm
        "layer_norm": (1.0, 1.5),   # aten::layer_norm

        # Pooling ops (memory-bound)
        "MaxPool2d": (1.0, 1.2),    # Slight overhead when separate
        "max_pool2d": (1.0, 1.2),   # aten::max_pool2d
        "AvgPool2d": (1.0, 1.2),    # Similar to MaxPool
        "avg_pool2d": (1.0, 1.2),   # aten::avg_pool2d
        "AdaptiveAvgPool2d": (1.0, 1.2),
        "adaptive_avg_pool2d": (1.0, 1.2),

        # Softmax (complex, architecture-dependent)
        "Softmax": (1.2, 1.5),      # Moderate overhead in both cases
        "softmax": (1.2, 1.5),      # aten::softmax

        # View/reshape ops (metadata-only, minimal cost)
        "view": (0.01, 0.05),       # Metadata operation
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default for unknown ops
        "default": (1.0, 1.0),
    },

    # -------------------------------------------------------------------------
    # DATA_PARALLEL (GPU): SIMT, warp-based execution
    # -------------------------------------------------------------------------
    ArchitectureClass.DATA_PARALLEL: {
        # Compute-intensive ops (baseline)
        "Linear": (1.0, 1.0),
        "addmm": (1.0, 1.0),
        "matmul": (1.0, 1.0),
        "bmm": (1.0, 1.0),
        "Conv2d": (1.0, 1.0),
        "conv2d": (1.0, 1.0),

        # Lightweight ops (kernel launch overhead on GPU)
        "Bias": (0.8, 3.0),         # Separate kernel has launch overhead
        "add": (0.5, 3.0),
        "ReLU": (0.5, 3.0),         # GPU kernel launch is expensive
        "relu": (0.5, 3.0),
        "GELU": (0.8, 3.0),
        "gelu": (0.8, 3.0),
        "Sigmoid": (0.8, 3.0),
        "sigmoid": (0.8, 3.0),
        "Tanh": (0.8, 3.0),
        "tanh": (0.8, 3.0),

        # Normalization ops
        "BatchNorm2d": (1.0, 1.5),
        "batch_norm": (1.0, 1.5),
        "LayerNorm": (1.0, 1.5),
        "layer_norm": (1.0, 1.5),

        # Pooling ops
        "MaxPool2d": (1.0, 1.2),
        "max_pool2d": (1.0, 1.2),
        "AvgPool2d": (1.0, 1.2),
        "avg_pool2d": (1.0, 1.2),
        "AdaptiveAvgPool2d": (1.0, 1.2),
        "adaptive_avg_pool2d": (1.0, 1.2),

        # Softmax (GPU handles well with parallel reduction)
        "Softmax": (1.0, 1.2),
        "softmax": (1.0, 1.2),

        # View/reshape ops
        "view": (0.01, 0.05),
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default
        "default": (1.0, 1.0),
    },

    # -------------------------------------------------------------------------
    # SYSTOLIC_ARRAY (TPU): Weight-stationary dataflow
    # -------------------------------------------------------------------------
    ArchitectureClass.SYSTOLIC_ARRAY: {
        # Compute-intensive ops (systolic array optimized)
        "Linear": (1.0, 1.0),       # Optimal for matmul
        "addmm": (1.0, 1.0),
        "matmul": (1.0, 1.0),
        "bmm": (1.0, 1.0),
        "Conv2d": (1.0, 1.0),       # Im2col → matmul
        "conv2d": (1.0, 1.0),

        # Lightweight ops (HIDDEN in dataflow pipeline)
        "Bias": (0.05, 1.0),        # Fused: hidden in output pipeline, Separate: full cost
        "add": (0.05, 1.0),         # Hidden in accumulator pipeline
        "ReLU": (0.05, 1.0),        # Hidden in activation pipeline
        "relu": (0.05, 1.0),
        "GELU": (0.1, 1.2),         # More complex, but still pipelined
        "gelu": (0.1, 1.2),
        "Sigmoid": (0.1, 1.2),
        "sigmoid": (0.1, 1.2),
        "Tanh": (0.1, 1.2),
        "tanh": (0.1, 1.2),

        # Normalization ops (can be fused with systolic output)
        "BatchNorm2d": (0.1, 1.5),  # Fused: in output pipeline, Separate: more expensive
        "batch_norm": (0.1, 1.5),
        "LayerNorm": (0.1, 1.5),
        "layer_norm": (0.1, 1.5),

        # Pooling ops (systolic not optimal, but can pipeline)
        "MaxPool2d": (0.8, 1.0),
        "max_pool2d": (0.8, 1.0),
        "AvgPool2d": (0.8, 1.0),
        "avg_pool2d": (0.8, 1.0),
        "AdaptiveAvgPool2d": (0.8, 1.0),
        "adaptive_avg_pool2d": (0.8, 1.0),

        # Softmax (harder on systolic, requires full reduce)
        "Softmax": (1.5, 2.0),      # Systolic not ideal for softmax
        "softmax": (1.5, 2.0),

        # View/reshape ops (metadata only)
        "view": (0.01, 0.05),
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default
        "default": (1.0, 1.0),
    },

    # -------------------------------------------------------------------------
    # DOMAIN_FLOW (KPU): Tile-based spatial dataflow
    # -------------------------------------------------------------------------
    ArchitectureClass.DOMAIN_FLOW: {
        # Compute-intensive ops (tile-based execution)
        "Linear": (1.0, 1.0),
        "addmm": (1.0, 1.0),
        "matmul": (1.0, 1.0),
        "bmm": (1.0, 1.0),
        "Conv2d": (1.0, 1.0),
        "conv2d": (1.0, 1.0),

        # Lightweight ops (HIDDEN in stream processing pipeline)
        "Bias": (0.05, 1.0),        # Fused: hidden in dataflow, Separate: full tile overhead
        "add": (0.2, 1.0),          # Residual connections can be pipelined
        "ReLU": (0.05, 1.0),        # Hidden in activation stage
        "relu": (0.05, 1.0),
        "GELU": (0.1, 1.2),
        "gelu": (0.1, 1.2),
        "Sigmoid": (0.1, 1.2),
        "sigmoid": (0.1, 1.2),
        "Tanh": (0.1, 1.2),
        "tanh": (0.1, 1.2),

        # Normalization ops (can stream with Conv tiles)
        "BatchNorm2d": (0.1, 1.5),  # Fused with Conv tiles
        "batch_norm": (0.1, 1.5),
        "LayerNorm": (0.1, 1.5),
        "layer_norm": (0.1, 1.5),

        # Pooling ops (tile-friendly)
        "MaxPool2d": (0.8, 1.0),
        "max_pool2d": (0.8, 1.0),
        "AvgPool2d": (0.8, 1.0),
        "avg_pool2d": (0.8, 1.0),
        "AdaptiveAvgPool2d": (0.8, 1.0),
        "adaptive_avg_pool2d": (0.8, 1.0),

        # Softmax (harder with tile-based execution)
        "Softmax": (1.5, 2.0),
        "softmax": (1.5, 2.0),

        # View/reshape ops
        "view": (0.01, 0.05),
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default
        "default": (1.0, 1.0),
    },

    # -------------------------------------------------------------------------
    # DATA_FLOW_MACHINE: Dataflow execution
    # -------------------------------------------------------------------------
    ArchitectureClass.DATA_FLOW_MACHINE: {
        # Compute-intensive ops (tensor units)
        "Linear": (1.0, 1.0),
        "addmm": (1.0, 1.0),
        "matmul": (1.0, 1.0),
        "bmm": (1.0, 1.0),
        "Conv2d": (1.0, 1.0),
        "conv2d": (1.0, 1.0),

        # Lightweight ops (vector units can fuse)
        "Bias": (0.5, 2.0),
        "add": (0.5, 2.0),
        "ReLU": (0.5, 2.0),
        "relu": (0.5, 2.0),
        "GELU": (0.8, 2.0),
        "gelu": (0.8, 2.0),
        "Sigmoid": (0.8, 2.0),
        "sigmoid": (0.8, 2.0),
        "Tanh": (0.8, 2.0),
        "tanh": (0.8, 2.0),

        # Normalization ops
        "BatchNorm2d": (0.8, 1.5),
        "batch_norm": (0.8, 1.5),
        "LayerNorm": (0.8, 1.5),
        "layer_norm": (0.8, 1.5),

        # Pooling ops
        "MaxPool2d": (0.9, 1.2),
        "max_pool2d": (0.9, 1.2),
        "AvgPool2d": (0.9, 1.2),
        "avg_pool2d": (0.9, 1.2),
        "AdaptiveAvgPool2d": (0.9, 1.2),
        "adaptive_avg_pool2d": (0.9, 1.2),

        # Softmax
        "Softmax": (1.2, 1.5),
        "softmax": (1.2, 1.5),

        # View/reshape ops
        "view": (0.01, 0.05),
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default
        "default": (1.0, 1.0),
    },

    # -------------------------------------------------------------------------
    # ADAPTIVE_DATAPATH (DPU/FPGA): Configurable dataflow
    # -------------------------------------------------------------------------
    ArchitectureClass.ADAPTIVE_DATAPATH: {
        # Compute-intensive ops (configurable for specific kernels)
        "Linear": (1.0, 1.0),
        "addmm": (1.0, 1.0),
        "matmul": (1.0, 1.0),
        "bmm": (1.0, 1.0),
        "Conv2d": (1.0, 1.0),
        "conv2d": (1.0, 1.0),

        # Lightweight ops (can configure for fusion)
        "Bias": (0.1, 1.2),         # Reconfigurable can fuse efficiently
        "add": (0.1, 1.2),
        "ReLU": (0.1, 1.2),
        "relu": (0.1, 1.2),
        "GELU": (0.2, 1.5),
        "gelu": (0.2, 1.5),
        "Sigmoid": (0.2, 1.5),
        "sigmoid": (0.2, 1.5),
        "Tanh": (0.2, 1.5),
        "tanh": (0.2, 1.5),

        # Normalization ops
        "BatchNorm2d": (0.2, 1.5),
        "batch_norm": (0.2, 1.5),
        "LayerNorm": (0.2, 1.5),
        "layer_norm": (0.2, 1.5),

        # Pooling ops
        "MaxPool2d": (0.8, 1.0),
        "max_pool2d": (0.8, 1.0),
        "AvgPool2d": (0.8, 1.0),
        "avg_pool2d": (0.8, 1.0),
        "AdaptiveAvgPool2d": (0.8, 1.0),
        "adaptive_avg_pool2d": (0.8, 1.0),

        # Softmax
        "Softmax": (1.2, 1.5),
        "softmax": (1.2, 1.5),

        # View/reshape ops
        "view": (0.01, 0.05),
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default
        "default": (1.0, 1.0),
    },

    # -------------------------------------------------------------------------
    # SPATIAL_PARTITION: Coarse-Grained Reconfigurable Array
    # -------------------------------------------------------------------------
    ArchitectureClass.SPATIAL_PARTITION: {
        # Compute-intensive ops (spatial mapping)
        "Linear": (1.0, 1.0),
        "addmm": (1.0, 1.0),
        "matmul": (1.0, 1.0),
        "bmm": (1.0, 1.0),
        "Conv2d": (1.0, 1.0),
        "conv2d": (1.0, 1.0),

        # Lightweight ops (can map spatially)
        "Bias": (0.1, 1.2),
        "add": (0.1, 1.2),
        "ReLU": (0.1, 1.2),
        "relu": (0.1, 1.2),
        "GELU": (0.2, 1.5),
        "gelu": (0.2, 1.5),
        "Sigmoid": (0.2, 1.5),
        "sigmoid": (0.2, 1.5),
        "Tanh": (0.2, 1.5),
        "tanh": (0.2, 1.5),

        # Normalization ops
        "BatchNorm2d": (0.2, 1.5),
        "batch_norm": (0.2, 1.5),
        "LayerNorm": (0.2, 1.5),
        "layer_norm": (0.2, 1.5),

        # Pooling ops
        "MaxPool2d": (0.8, 1.0),
        "max_pool2d": (0.8, 1.0),
        "AvgPool2d": (0.8, 1.0),
        "avg_pool2d": (0.8, 1.0),
        "AdaptiveAvgPool2d": (0.8, 1.0),
        "adaptive_avg_pool2d": (0.8, 1.0),

        # Softmax
        "Softmax": (1.2, 1.5),
        "softmax": (1.2, 1.5),

        # View/reshape ops
        "view": (0.01, 0.05),
        "reshape": (0.01, 0.05),
        "transpose": (0.01, 0.05),
        "permute": (0.01, 0.05),
        "contiguous": (0.01, 0.05),

        # Default
        "default": (1.0, 1.0),
    },
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_architectural_modifier(
    operator_type: str,
    arch_class: ArchitectureClass,
    is_fused: bool
) -> float:
    """
    Get architectural modifier for an operator based on architecture and fusion status.

    Args:
        operator_type: Operator type (e.g., "Linear", "ReLU", "add")
        arch_class: Architecture class (STORED_PROGRAM, DATA_PARALLEL, etc.)
        is_fused: True if operator is fused with others in subgraph

    Returns:
        Modifier value (e.g., 0.05 for hidden ReLU on KPU, 3.0 for separate kernel on GPU)

    Example:
        >>> get_architectural_modifier("ReLU", ArchitectureClass.DOMAIN_FLOW, is_fused=True)
        0.05  # Hidden in dataflow on KPU

        >>> get_architectural_modifier("ReLU", ArchitectureClass.DATA_PARALLEL, is_fused=False)
        3.0   # Separate kernel overhead on GPU
    """
    # Get modifiers for this architecture
    arch_modifiers = ARCHITECTURAL_MODIFIERS.get(arch_class, ARCHITECTURAL_MODIFIERS[ArchitectureClass.STORED_PROGRAM])

    # Get operator-specific modifiers (fused, separate)
    if operator_type in arch_modifiers:
        fused_mod, separate_mod = arch_modifiers[operator_type]
    else:
        # Try lowercase version
        op_lower = operator_type.lower()
        if op_lower in arch_modifiers:
            fused_mod, separate_mod = arch_modifiers[op_lower]
        else:
            # Use default
            fused_mod, separate_mod = arch_modifiers["default"]

    # Return appropriate modifier based on fusion status
    return fused_mod if is_fused else separate_mod


def get_fusion_benefit(
    operator_type: str,
    arch_class: ArchitectureClass
) -> float:
    """
    Calculate EDP benefit from fusing this operator (ratio: separate/fused).

    Args:
        operator_type: Operator type
        arch_class: Architecture class

    Returns:
        Fusion benefit ratio (>1.0 means fusion helps, <1.0 means fusion hurts)

    Example:
        >>> get_fusion_benefit("ReLU", ArchitectureClass.DOMAIN_FLOW)
        20.0  # Fusing ReLU on KPU provides 20× benefit (0.05 vs 1.0)

        >>> get_fusion_benefit("Linear", ArchitectureClass.DATA_PARALLEL)
        1.0   # Linear not affected by fusion on GPU
    """
    fused_mod = get_architectural_modifier(operator_type, arch_class, is_fused=True)
    separate_mod = get_architectural_modifier(operator_type, arch_class, is_fused=False)

    # Avoid division by zero
    if fused_mod == 0:
        return float('inf')

    return separate_mod / fused_mod


def explain_modifier(
    operator_type: str,
    arch_class: ArchitectureClass,
    is_fused: bool
) -> str:
    """
    Generate human-readable explanation for why this modifier value is used.

    Args:
        operator_type: Operator type
        arch_class: Architecture class
        is_fused: Fusion status

    Returns:
        Explanation string

    Example:
        >>> explain_modifier("ReLU", ArchitectureClass.DOMAIN_FLOW, is_fused=True)
        "Hidden in dataflow pipeline (0.05× modifier on KPU)"
    """
    modifier = get_architectural_modifier(operator_type, arch_class, is_fused)

    # Architecture-specific explanations
    if arch_class in [ArchitectureClass.SYSTOLIC_ARRAY, ArchitectureClass.DOMAIN_FLOW]:
        if is_fused and modifier < 0.2:
            return f"Hidden in dataflow pipeline ({modifier:.2f}× modifier)"
        elif not is_fused and modifier > 1.0:
            return f"Separate tile/systolic overhead ({modifier:.2f}× modifier)"
        else:
            return f"Standard execution ({modifier:.2f}× modifier)"

    elif arch_class in [ArchitectureClass.STORED_PROGRAM, ArchitectureClass.DATA_PARALLEL]:
        if not is_fused and modifier > 2.0:
            return f"Separate kernel launch overhead ({modifier:.2f}× modifier)"
        elif is_fused and modifier < 0.8:
            return f"Fused with compute kernel ({modifier:.2f}× modifier)"
        else:
            return f"Standard execution ({modifier:.2f}× modifier)"

    else:
        return f"Architecture-specific modifier ({modifier:.2f}×)"
