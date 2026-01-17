"""
Data structures for graph partitioning and analysis.

This module defines the core data structures used for realistic performance modeling:
- SubgraphDescriptor: Complete description of a computational subgraph
- ParallelismDescriptor: Available parallelism dimensions
- TensorDescriptor: Tensor shape and memory info
- ConcurrencyDescriptor: Graph-level concurrency analysis
- PartitionReport: Complete partition statistics
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


class OperationType(Enum):
    """Types of operations in neural networks"""
    CONV2D = "conv2d"
    CONV2D_DEPTHWISE = "conv2d_depthwise"
    CONV2D_POINTWISE = "conv2d_pointwise"
    LINEAR = "linear"
    MATMUL = "matmul"
    BATCHNORM = "batchnorm"
    LAYERNORM = "layernorm"
    RELU = "relu"
    RELU6 = "relu6"
    SILU = "silu"  # Swish activation (EfficientNet)
    SWISH = "swish"  # Alternative name for SiLU
    GELU = "gelu"
    HARDSWISH = "hardswish"
    SIGMOID = "sigmoid"  # For SE blocks
    MAXPOOL = "maxpool"
    AVGPOOL = "avgpool"
    ADAPTIVEAVGPOOL = "adaptiveavgpool"
    ELEMENTWISE = "elementwise"
    SQUEEZE_EXCITE = "squeeze_excite"

    # Transformer operations
    MULTIHEAD_ATTENTION = "multihead_attention"
    DROPOUT = "dropout"
    STOCHASTIC_DEPTH = "stochastic_depth"  # Drop path

    UNKNOWN = "unknown"


class BottleneckType(Enum):
    """Performance bottleneck types"""
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    BANDWIDTH_BOUND = "bandwidth_bound"
    BALANCED = "balanced"


class PartitionReason(Enum):
    """Reasons for creating a partition boundary"""
    OPERATION_BOUNDARY = "operation_boundary"  # Default: each operation analyzed separately
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"  # Would exceed memory threshold if fused
    COMPUTE_THRESHOLD_EXCEEDED = "compute_threshold_exceeded"  # Would exceed compute threshold if fused
    FUSION_INCOMPATIBLE = "fusion_incompatible"  # Operations cannot be fused together
    DATA_DEPENDENCY = "data_dependency"  # Data dependencies prevent fusion
    FUSION_OPPORTUNITY = "fusion_opportunity"  # Could be fused but policy keeps separate
    PARALLEL_SCHEDULE_DIVERGENCE = "parallel_schedule_divergence"  # KPU scheduling constraints prevent fusion


@dataclass
class TensorDescriptor:
    """Tensor shape and memory information"""
    shape: Tuple[int, ...]
    dtype: str  # 'float32', 'bfloat16', 'int8', etc.
    size_bytes: int
    layout: str = "NCHW"  # NCHW, NHWC, etc.

    def __post_init__(self):
        """Validate tensor descriptor"""
        if self.size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {self.size_bytes}")


@dataclass
class ParallelismDescriptor:
    """Available parallelism dimensions within an operation"""

    # Basic dimensions
    batch: int
    channels: int  # output channels or features
    spatial: int  # height × width for convs, 1 for linear
    total_threads: int  # batch × channels × spatial

    # Special characteristics
    is_depthwise: bool = False
    is_grouped: bool = False
    num_groups: int = 1

    # Vectorization potential (for CPU)
    vectorizable_dim: Optional[str] = None  # 'channels', 'spatial', etc.
    vector_width: int = 8  # lanes available (8 for AVX2, 16 for AVX-512)

    # Data parallelism capabilities
    can_split_batch: bool = True
    can_split_spatial: bool = True
    can_split_channels: bool = True

    def __post_init__(self):
        """Validate parallelism descriptor"""
        if self.batch <= 0 or self.channels <= 0 or self.spatial <= 0:
            raise ValueError("All parallelism dimensions must be positive")

        # Depthwise convolutions have limited channel parallelism
        if self.is_depthwise:
            self.can_split_channels = False


@dataclass
class SubgraphDescriptor:
    """
    Unified description of a computational subgraph (single or fused operators).

    This class supports both:
    - Single-operator subgraphs (from GraphPartitioner or unfused operations)
    - Multi-operator fused subgraphs (from FusionBasedPartitioner)

    For single-op subgraphs: node_ids/node_names/operation_types have single element, num_operators=1
    For fused subgraphs: node_ids/node_names/operation_types have multiple elements, num_operators>1
    """

    # Identity (unified for single and multi-op)
    subgraph_id: int  # Numeric ID (required for all subgraphs)
    node_ids: List[str]  # List of FX node IDs (single-element for unfused)
    node_names: List[str]  # Human-readable names (single-element for unfused)
    operation_types: List[OperationType]  # Operation types in order (single-element for unfused)
    fusion_pattern: str  # 'Conv_BN_ReLU', 'Linear', 'Unfused', etc.

    # Computation (unified naming - total_* for consistency)
    total_flops: int
    total_macs: int  # multiply-accumulate operations

    # Memory (external vs internal distinction)
    total_input_bytes: int  # External inputs (cross subgraph boundary)
    total_output_bytes: int  # External outputs (cross subgraph boundary)
    total_weight_bytes: int  # Weights/parameters
    internal_bytes: int = 0  # Intermediate tensors (saved by fusion, 0 for unfused)
    arithmetic_intensity: float = 0.0  # FLOPs / external_bytes

    # Fusion metadata
    num_operators: int = 1  # Number of operators fused (1 for unfused)

    # Parallelism (merged for fused ops)
    parallelism: Optional[ParallelismDescriptor] = None

    # Dependencies (numeric IDs for subgraph dependencies)
    depends_on: List[int] = field(default_factory=list)  # Other subgraph IDs
    dependency_type: str = "sequential"  # 'sequential', 'independent', 'partial'

    # Hardware hints
    recommended_bottleneck: BottleneckType = BottleneckType.BALANCED

    # Optional: Detailed tensor info (primarily for single-op analysis)
    input_tensors: List[TensorDescriptor] = field(default_factory=list)
    output_tensors: List[TensorDescriptor] = field(default_factory=list)
    weight_tensors: List[TensorDescriptor] = field(default_factory=list)

    # Optional: Partition metadata (primarily for unfused/single-op)
    partition_reason: Optional[PartitionReason] = None
    partition_criteria: Dict[str, any] = field(default_factory=dict)
    fusion_candidates: List[str] = field(default_factory=list)  # Node IDs that could have been fused

    def __post_init__(self):
        """Compute derived fields"""
        # Calculate arithmetic intensity based on external bytes only
        external_bytes = self.total_input_bytes + self.total_output_bytes + self.total_weight_bytes
        if self.total_flops > 0 and external_bytes > 0:
            self.arithmetic_intensity = self.total_flops / external_bytes

            # Classify bottleneck based on arithmetic intensity
            if self.arithmetic_intensity > 50:
                self.recommended_bottleneck = BottleneckType.COMPUTE_BOUND
            elif self.arithmetic_intensity > 10:
                self.recommended_bottleneck = BottleneckType.BALANCED
            elif self.arithmetic_intensity > 1:
                self.recommended_bottleneck = BottleneckType.MEMORY_BOUND
            else:
                self.recommended_bottleneck = BottleneckType.BANDWIDTH_BOUND

    def data_movement_reduction(self) -> float:
        """
        Calculate reduction in data movement due to fusion.

        Returns:
            Fraction of memory traffic eliminated (0.0 for unfused, >0.0 for fused)
        """
        if self.num_operators <= 1:
            return 0.0

        # Savings = internal bytes that don't touch global memory due to fusion
        external_bytes = self.total_input_bytes + self.total_output_bytes + self.total_weight_bytes
        total_without_fusion = external_bytes + self.internal_bytes

        if total_without_fusion == 0:
            return 0.0

        return self.internal_bytes / total_without_fusion

    def partition_reasoning_summary(self) -> str:
        """Generate human-readable explanation of partition reasoning"""
        if self.partition_reason is None:
            return "Partition Reason: Not specified (fused subgraph)\n"

        summary = f"Partition Reason: {self.partition_reason.value}\n"

        if self.partition_reason == PartitionReason.OPERATION_BOUNDARY:
            summary += "  This operation is analyzed as a separate subgraph (default behavior).\n"

        elif self.partition_reason == PartitionReason.MEMORY_LIMIT_EXCEEDED:
            summary += f"  Memory usage ({self.partition_criteria.get('total_bytes', 0) / 1e6:.2f} MB) "
            summary += f"exceeds fusion threshold ({self.partition_criteria.get('threshold_memory', 0) / 1e6:.2f} MB).\n"

        elif self.partition_reason == PartitionReason.COMPUTE_THRESHOLD_EXCEEDED:
            summary += f"  Compute load ({self.partition_criteria.get('flops', 0) / 1e9:.2f} GFLOPs) "
            summary += f"exceeds fusion threshold ({self.partition_criteria.get('threshold_flops', 0) / 1e9:.2f} GFLOPs).\n"

        elif self.partition_reason == PartitionReason.FUSION_OPPORTUNITY:
            candidates = self.partition_criteria.get('fusion_candidates', [])
            summary += f"  Could be fused with: {', '.join(candidates)}\n"
            summary += "  Kept separate by partitioning policy.\n"

        elif self.partition_reason == PartitionReason.DATA_DEPENDENCY:
            num_consumers = self.partition_criteria.get('num_consumers', 0)
            summary += f"  Multiple data consumers ({num_consumers}) make fusion difficult.\n"

        elif self.partition_reason == PartitionReason.FUSION_INCOMPATIBLE:
            summary += "  Operation type cannot be fused with neighbors.\n"

        elif self.partition_reason == PartitionReason.PARALLEL_SCHEDULE_DIVERGENCE:
            summary += "  KPU scheduling constraints prevent fusion with neighbors.\n"

        return summary

    # Backward compatibility properties for legacy code expecting single-op interface
    @property
    def node_id(self) -> str:
        """Legacy: First node ID (for single-op compatibility)"""
        return self.node_ids[0] if self.node_ids else ""

    @property
    def node_name(self) -> str:
        """Legacy: First node name (for single-op compatibility)"""
        return self.node_names[0] if self.node_names else ""

    @property
    def operation_type(self) -> OperationType:
        """Legacy: First operation type (for single-op compatibility)"""
        return self.operation_types[0] if self.operation_types else OperationType.UNKNOWN

    @property
    def flops(self) -> int:
        """Legacy: Alias for total_flops"""
        return self.total_flops

    @property
    def macs(self) -> int:
        """Legacy: Alias for total_macs"""
        return self.total_macs


@dataclass
class SubgraphConcurrency:
    """Concurrency available within a single subgraph"""

    # Thread-level parallelism
    total_threads: int
    independent_threads: int  # truly independent (no data sharing)

    # Instruction-level parallelism
    independent_operations: int  # ops with no dependencies
    dependency_chains: int  # longest dependency chain length

    # Data parallelism splits
    can_split_batch: bool = True
    can_split_spatial: bool = True
    can_split_channels: bool = True

    # Hardware requirements
    min_hardware_units: int = 1  # minimum cores/SMs needed
    optimal_hardware_units: int = 8  # for best efficiency
    max_hardware_units: int = 16  # diminishing returns beyond this

    # Efficiency estimate
    parallelism_efficiency: float = 1.0  # how well parallelism can be exploited


@dataclass
class ConcurrencyDescriptor:
    """Analysis of available concurrency in entire computation graph"""

    # Graph-level concurrency
    total_subgraphs: int
    independent_subgraphs: int  # can execute in parallel
    sequential_subgraphs: int  # must execute serially

    # Critical path
    critical_path_length: int  # number of sequential ops on longest path
    critical_path_flops: int  # FLOPs on critical path
    parallelizable_flops: int  # FLOPs that can run concurrently

    # Batching potential
    batch_size: int
    batch_parallelism: int  # independent samples
    max_theoretical_speedup: float  # with infinite batch

    # Stages (groups that can run in parallel)
    num_stages: int
    max_parallel_ops_per_stage: int
    stages: List[List[str]] = field(default_factory=list)  # [[node_ids in stage 1], [stage 2], ...]

    # Per-subgraph concurrency
    subgraph_concurrency: Dict[str, SubgraphConcurrency] = field(default_factory=dict)

    # Validation metrics
    concurrency_utilization: float = 0.0  # actual / theoretical
    parallelism_efficiency: float = 0.0  # how well we use hardware

    # Human-readable explanation
    explanation: str = ""


@dataclass
class PartitionReport:
    """
    Unified statistics from graph partitioning (single-op or fused).

    Supports both:
    - GraphPartitioner output (unfused single-op subgraphs)
    - FusionBasedPartitioner output (multi-op fused subgraphs)
    """

    # Subgraphs (unified list of SubgraphDescriptor)
    subgraphs: List[SubgraphDescriptor]
    total_subgraphs: int

    # Computation totals
    total_flops: int
    total_macs: int
    total_memory_traffic: int  # External memory traffic (with fusion)

    # Fusion metrics (NEW - for measuring fusion benefit)
    original_operators: int = 0  # Number of operators before fusion
    total_memory_traffic_unfused: int = 0  # What memory traffic would be without fusion
    data_movement_reduction: float = 0.0  # Fraction of memory traffic saved by fusion
    avg_fusion_size: float = 1.0  # Average operators per subgraph
    max_fusion_size: int = 1  # Largest fused subgraph

    # Arithmetic intensity distribution
    average_arithmetic_intensity: float = 0.0
    min_arithmetic_intensity: float = 0.0
    max_arithmetic_intensity: float = 0.0

    # Subgraph type distribution
    operation_type_counts: Dict[str, int] = field(default_factory=dict)
    fusion_pattern_counts: Dict[str, int] = field(default_factory=dict)

    # Parallelism distribution
    parallelism_distribution: Dict[str, int] = field(default_factory=dict)  # '<1K', '1K-10K', etc.

    # Bottleneck analysis
    bottleneck_distribution: Dict[str, int] = field(default_factory=dict)

    # Partition reasoning (optional - primarily for unfused)
    partition_reason_distribution: Dict[str, int] = field(default_factory=dict)

    # Concurrency analysis (optional)
    concurrency: Optional[ConcurrencyDescriptor] = None

    # Critical path (optional)
    critical_path_subgraphs: List[str] = field(default_factory=list)

    # Backward compatibility alias for hardware mappers
    @property
    def fused_subgraphs(self) -> List[SubgraphDescriptor]:
        """Alias for backward compatibility with code expecting FusionReport"""
        return self.subgraphs

    def summary_stats(self) -> str:
        """Generate human-readable summary"""
        fusion_stats = ""
        if self.original_operators > self.total_subgraphs:
            # Fusion occurred
            fusion_stats = f"""
Fusion Statistics:
  Original operators: {self.original_operators}
  Fused subgraphs: {self.total_subgraphs}
  Reduction: {self.original_operators / max(1, self.total_subgraphs):.1f}× fewer execution units
  Average fusion size: {self.avg_fusion_size:.1f} ops/subgraph
  Max fusion size: {self.max_fusion_size} ops
  Data movement reduction: {self.data_movement_reduction * 100:.1f}%
  Memory saved: {(self.total_memory_traffic_unfused - self.total_memory_traffic) / 1e6:.2f} MB
"""

        return f"""
Graph Partition Summary
=======================
Total subgraphs: {self.total_subgraphs}
Total FLOPs: {self.total_flops / 1e9:.2f} G
Total memory traffic: {self.total_memory_traffic / 1e6:.2f} MB
Average arithmetic intensity: {self.average_arithmetic_intensity:.2f} FLOPs/byte
{fusion_stats}
Operation types:
{self._format_dict(self.operation_type_counts)}

Bottleneck distribution:
{self._format_dict(self.bottleneck_distribution)}

Partition reasons:
{self._format_dict(self.partition_reason_distribution)}

Parallelism distribution:
{self._format_dict(self.parallelism_distribution)}
"""

    def _format_dict(self, d: Dict[str, int]) -> str:
        """Format dictionary for display"""
        if not d:
            return "  (none)"
        return "\n".join(f"  {k}: {v} ({v/self.total_subgraphs*100:.0f}%)"
                        for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True))


# Helper functions for creating descriptors

def create_tensor_descriptor(shape: Tuple[int, ...], dtype: str = "float32") -> TensorDescriptor:
    """Create tensor descriptor with automatic size calculation"""
    dtype_bytes = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
        'int32': 4,
        'int64': 8
    }

    element_size = dtype_bytes.get(dtype, 4)
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    size_bytes = num_elements * element_size

    return TensorDescriptor(
        shape=shape,
        dtype=dtype,
        size_bytes=size_bytes,
        layout="NCHW"  # default
    )


def classify_operation_type(module_type: type) -> OperationType:
    """Classify PyTorch module type to operation type"""
    import torch.nn as nn

    if module_type == nn.Conv2d:
        return OperationType.CONV2D
    elif module_type == nn.Linear:
        return OperationType.LINEAR
    elif module_type == nn.BatchNorm2d:
        return OperationType.BATCHNORM
    elif module_type == nn.LayerNorm:
        return OperationType.LAYERNORM
    elif module_type == nn.ReLU:
        return OperationType.RELU
    elif module_type == nn.ReLU6:
        return OperationType.RELU6
    elif module_type == nn.GELU:
        return OperationType.GELU
    elif module_type == nn.Hardswish:
        return OperationType.HARDSWISH
    elif module_type == nn.MaxPool2d:
        return OperationType.MAXPOOL
    elif module_type == nn.AvgPool2d:
        return OperationType.AVGPOOL
    elif module_type == nn.AdaptiveAvgPool2d:
        return OperationType.ADAPTIVEAVGPOOL
    else:
        return OperationType.UNKNOWN
