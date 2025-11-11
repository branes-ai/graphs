"""
Workload Characterization: MACs/FLOPs/IntOps Separation

This module provides data structures for accurately characterizing neural network
workloads by separating operations into categories that map to different hardware
execution units:

- MACs (Multiply-Accumulate): Matrix/convolution operations → Tensor Cores, Systolic Arrays, AMX
- FLOPs (Floating-Point Ops): Bias, activation, elementwise → CUDA cores, SIMD units
- IntOps (Integer Ops): Quantization, indexing → Integer ALUs

This separation enables accurate hardware mapping and energy modeling.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class Precision(Enum):
    """Numerical precision types"""
    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"  # Mixed precision


@dataclass
class OperationBreakdown:
    """
    Detailed breakdown of operation types within a workload.

    This provides fine-grained visibility into what operations are present,
    enabling detailed analysis and optimization opportunities.
    """

    # ============================================================
    # MAC OPERATIONS (Matrix/Convolution)
    # ============================================================

    # Matrix multiplication operations
    matmul_macs: int = 0
    """Matrix multiplication MACs (M×K×N for [M×K] @ [K×N])"""

    bmm_macs: int = 0
    """Batched matrix multiplication MACs"""

    # Convolution operations
    conv1d_macs: int = 0
    """1D convolution MACs"""

    conv2d_macs: int = 0
    """2D convolution MACs (standard convolution)"""

    conv3d_macs: int = 0
    """3D convolution MACs"""

    depthwise_conv_macs: int = 0
    """Depthwise separable convolution MACs"""

    pointwise_conv_macs: int = 0
    """Pointwise (1×1) convolution MACs"""

    # ============================================================
    # FLOP OPERATIONS (Non-Matrix)
    # ============================================================

    # Vector/bias operations
    bias_flops: int = 0
    """Bias addition FLOPs (vector addition)"""

    # Activation functions
    relu_flops: int = 0
    """ReLU activation FLOPs"""

    gelu_flops: int = 0
    """GELU activation FLOPs (~8 FLOPs per element)"""

    silu_flops: int = 0
    """SiLU/Swish activation FLOPs (~5 FLOPs per element)"""

    sigmoid_flops: int = 0
    """Sigmoid activation FLOPs (~4 FLOPs per element)"""

    tanh_flops: int = 0
    """Tanh activation FLOPs (~4 FLOPs per element)"""

    # Normalization operations
    batchnorm_flops: int = 0
    """Batch normalization FLOPs (~4 FLOPs per element)"""

    layernorm_flops: int = 0
    """Layer normalization FLOPs (~4 FLOPs per element)"""

    groupnorm_flops: int = 0
    """Group normalization FLOPs"""

    instancenorm_flops: int = 0
    """Instance normalization FLOPs"""

    # Attention operations
    softmax_flops: int = 0
    """Softmax FLOPs (~10 FLOPs per element)"""

    attention_flops: int = 0
    """Attention mechanism FLOPs (non-matmul part)"""

    # Element-wise operations
    elementwise_add_flops: int = 0
    """Element-wise addition FLOPs"""

    elementwise_sub_flops: int = 0
    """Element-wise subtraction FLOPs"""

    elementwise_mul_flops: int = 0
    """Element-wise multiplication FLOPs"""

    elementwise_div_flops: int = 0
    """Element-wise division FLOPs"""

    # Reduction operations
    reduction_sum_flops: int = 0
    """Sum reduction FLOPs"""

    reduction_mean_flops: int = 0
    """Mean reduction FLOPs"""

    reduction_max_flops: int = 0
    """Max reduction FLOPs"""

    reduction_min_flops: int = 0
    """Min reduction FLOPs"""

    # Other FLOPs
    other_flops: int = 0
    """Other unclassified FLOPs"""

    # ============================================================
    # INTEGER OPERATIONS
    # ============================================================

    # Quantization operations
    quantize_intops: int = 0
    """Quantization operations (FP → INT)"""

    dequantize_intops: int = 0
    """Dequantization operations (INT → FP)"""

    # Indexing/addressing operations
    indexing_intops: int = 0
    """Indexing operations (gather, scatter, lookup)"""

    embedding_intops: int = 0
    """Embedding lookup operations"""

    # Comparison operations
    comparison_intops: int = 0
    """Comparison operations (<, >, ==, !=)"""

    # Logical operations
    logical_intops: int = 0
    """Logical operations (AND, OR, NOT, XOR)"""

    # Bitwise operations
    bitwise_intops: int = 0
    """Bitwise operations"""

    # Other integer ops
    other_intops: int = 0
    """Other unclassified integer operations"""

    def total_macs(self) -> int:
        """Total MAC operations"""
        return (
            self.matmul_macs +
            self.bmm_macs +
            self.conv1d_macs +
            self.conv2d_macs +
            self.conv3d_macs +
            self.depthwise_conv_macs +
            self.pointwise_conv_macs
        )

    def total_flops(self) -> int:
        """Total FLOP operations"""
        return (
            self.bias_flops +
            self.relu_flops +
            self.gelu_flops +
            self.silu_flops +
            self.sigmoid_flops +
            self.tanh_flops +
            self.batchnorm_flops +
            self.layernorm_flops +
            self.groupnorm_flops +
            self.instancenorm_flops +
            self.softmax_flops +
            self.attention_flops +
            self.elementwise_add_flops +
            self.elementwise_sub_flops +
            self.elementwise_mul_flops +
            self.elementwise_div_flops +
            self.reduction_sum_flops +
            self.reduction_mean_flops +
            self.reduction_max_flops +
            self.reduction_min_flops +
            self.other_flops
        )

    def total_intops(self) -> int:
        """Total integer operations"""
        return (
            self.quantize_intops +
            self.dequantize_intops +
            self.indexing_intops +
            self.embedding_intops +
            self.comparison_intops +
            self.logical_intops +
            self.bitwise_intops +
            self.other_intops
        )


@dataclass
class WorkloadCharacterization:
    """
    Comprehensive workload characterization with operation type separation.

    This structure separates operations by type (MAC/FLOP/IntOp) to enable
    accurate mapping to hardware execution units and energy modeling.

    Key Design Principles:
    - MACs (Multiply-Accumulate) → Specialized matrix units (Tensor Cores, Systolic Arrays)
    - FLOPs (Floating-Point Ops) → General compute units (CUDA cores, SIMD)
    - IntOps (Integer Ops) → Integer ALUs

    Example for 256×256 MLP with bias and ReLU:
        macs = 65,536              # 256×256 matrix multiplication
        flops = 512                # 256 bias + 256 ReLU
        intops = 0                 # No quantization
        mac_precision = 'fp32'
        flop_precision = 'fp32'
    """

    # ============================================================
    # CORE OPERATION COUNTS
    # ============================================================

    macs: int
    """
    Total multiply-accumulate operations (matrix/convolution).
    Maps to: Tensor Cores, Systolic Arrays, AMX, BLAS tiles.
    """

    flops: int
    """
    Total floating-point operations (bias, activation, elementwise).
    Maps to: CUDA cores, SIMD units, Vector ALUs.
    """

    intops: int
    """
    Total integer operations (quantization, indexing, comparisons).
    Maps to: Integer ALUs, Special Function Units (SFU).
    """

    # ============================================================
    # PRECISION CONFIGURATION
    # ============================================================

    mac_precision: str = "fp32"
    """Precision for MAC operations (fp32, fp16, bf16, int8)"""

    flop_precision: str = "fp32"
    """Precision for FLOP operations (fp32, fp16, bf16)"""

    intop_precision: str = "int32"
    """Precision for integer operations (int8, int16, int32)"""

    accumulator_precision: Optional[str] = None
    """
    Accumulator precision (if different from MAC precision).
    Example: INT8 MACs with FP32 accumulation (Tensor Core mixed precision).
    """

    # ============================================================
    # MEMORY CHARACTERISTICS
    # ============================================================

    bytes_transferred: int = 0
    """Total bytes transferred (input + weights + output)"""

    input_bytes: int = 0
    """Input activation bytes"""

    weight_bytes: int = 0
    """Weight/parameter bytes"""

    output_bytes: int = 0
    """Output activation bytes"""

    # ============================================================
    # DERIVED METRICS
    # ============================================================

    def total_ops(self) -> int:
        """Total operations (MACs + FLOPs + IntOps)"""
        return self.macs + self.flops + self.intops

    def arithmetic_intensity_macs(self) -> float:
        """Arithmetic intensity for MAC operations (MACs/byte)"""
        if self.bytes_transferred == 0:
            return 0.0
        return self.macs / self.bytes_transferred

    def arithmetic_intensity_total(self) -> float:
        """Arithmetic intensity for all operations (ops/byte)"""
        if self.bytes_transferred == 0:
            return 0.0
        return self.total_ops() / self.bytes_transferred

    def mac_percentage(self) -> float:
        """Percentage of operations that are MACs"""
        total = self.total_ops()
        if total == 0:
            return 0.0
        return 100.0 * self.macs / total

    def flop_percentage(self) -> float:
        """Percentage of operations that are FLOPs"""
        total = self.total_ops()
        if total == 0:
            return 0.0
        return 100.0 * self.flops / total

    def intop_percentage(self) -> float:
        """Percentage of operations that are IntOps"""
        total = self.total_ops()
        if total == 0:
            return 0.0
        return 100.0 * self.intops / total

    def is_compute_bound(self, peak_bandwidth_bytes_per_sec: float,
                        peak_compute_ops_per_sec: float) -> bool:
        """
        Determine if workload is compute-bound vs memory-bound.

        Args:
            peak_bandwidth_bytes_per_sec: Peak memory bandwidth (bytes/sec)
            peak_compute_ops_per_sec: Peak compute throughput (ops/sec)

        Returns:
            True if compute-bound, False if memory-bound
        """
        ai = self.arithmetic_intensity_total()
        roofline_threshold = peak_compute_ops_per_sec / peak_bandwidth_bytes_per_sec
        return ai >= roofline_threshold

    # ============================================================
    # DETAILED BREAKDOWN (OPTIONAL)
    # ============================================================

    breakdown: Optional[OperationBreakdown] = None
    """Detailed breakdown of operation types (optional)"""

    # ============================================================
    # METADATA
    # ============================================================

    model_name: Optional[str] = None
    """Model name (e.g., 'resnet18', 'bert-base')"""

    batch_size: int = 1
    """Batch size"""

    input_shape: Optional[tuple] = None
    """Input tensor shape"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def __post_init__(self):
        """Validation after initialization"""
        if self.macs < 0:
            raise ValueError(f"macs must be non-negative, got {self.macs}")
        if self.flops < 0:
            raise ValueError(f"flops must be non-negative, got {self.flops}")
        if self.intops < 0:
            raise ValueError(f"intops must be non-negative, got {self.intops}")
        if self.bytes_transferred < 0:
            raise ValueError(f"bytes_transferred must be non-negative, got {self.bytes_transferred}")

        # If breakdown is provided, validate totals match
        if self.breakdown:
            breakdown_macs = self.breakdown.total_macs()
            breakdown_flops = self.breakdown.total_flops()
            breakdown_intops = self.breakdown.total_intops()

            if breakdown_macs != self.macs:
                raise ValueError(
                    f"Breakdown MACs ({breakdown_macs}) doesn't match total MACs ({self.macs})"
                )
            if breakdown_flops != self.flops:
                raise ValueError(
                    f"Breakdown FLOPs ({breakdown_flops}) doesn't match total FLOPs ({self.flops})"
                )
            if breakdown_intops != self.intops:
                raise ValueError(
                    f"Breakdown IntOps ({breakdown_intops}) doesn't match total IntOps ({self.intops})"
                )

    def __str__(self) -> str:
        """Human-readable string representation"""
        lines = []
        if self.model_name:
            lines.append(f"Model: {self.model_name}")
        lines.append(f"Batch size: {self.batch_size}")
        lines.append("")
        lines.append("Operations:")
        lines.append(f"  MACs:    {self.macs:,}  ({self.mac_percentage():.1f}%)")
        lines.append(f"  FLOPs:   {self.flops:,}  ({self.flop_percentage():.1f}%)")
        lines.append(f"  IntOps:  {self.intops:,}  ({self.intop_percentage():.1f}%)")
        lines.append(f"  Total:   {self.total_ops():,}")
        lines.append("")
        lines.append("Precision:")
        lines.append(f"  MACs:    {self.mac_precision}")
        lines.append(f"  FLOPs:   {self.flop_precision}")
        lines.append(f"  IntOps:  {self.intop_precision}")
        if self.accumulator_precision:
            lines.append(f"  Accum:   {self.accumulator_precision}")
        lines.append("")
        lines.append("Memory:")
        lines.append(f"  Transferred: {self.bytes_transferred:,} bytes")
        lines.append(f"  AI (MACs):   {self.arithmetic_intensity_macs():.2f} MACs/byte")
        lines.append(f"  AI (Total):  {self.arithmetic_intensity_total():.2f} ops/byte")

        return "\n".join(lines)


def create_simple_workload(
    macs: int,
    flops: int = 0,
    intops: int = 0,
    bytes_transferred: int = 0,
    mac_precision: str = "fp32",
    flop_precision: str = "fp32",
    intop_precision: str = "int32",
    batch_size: int = 1,
    model_name: Optional[str] = None
) -> WorkloadCharacterization:
    """
    Factory function for creating simple workloads without detailed breakdown.

    Args:
        macs: Total MAC operations
        flops: Total FLOP operations
        intops: Total integer operations
        bytes_transferred: Total bytes transferred
        mac_precision: MAC precision
        flop_precision: FLOP precision
        intop_precision: Integer operation precision
        batch_size: Batch size
        model_name: Model name

    Returns:
        WorkloadCharacterization instance

    Example:
        >>> workload = create_simple_workload(
        ...     macs=65536,
        ...     flops=512,
        ...     bytes_transferred=264192,
        ...     model_name="mlp_256x256"
        ... )
    """
    return WorkloadCharacterization(
        macs=macs,
        flops=flops,
        intops=intops,
        mac_precision=mac_precision,
        flop_precision=flop_precision,
        intop_precision=intop_precision,
        bytes_transferred=bytes_transferred,
        batch_size=batch_size,
        model_name=model_name
    )
