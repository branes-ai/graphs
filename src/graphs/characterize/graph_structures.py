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
    """Complete description of a computational subgraph (kernel/operation)"""

    # Identity
    node_id: str
    node_name: str
    operation_type: OperationType
    fusion_pattern: str  # 'conv_bn_relu', 'conv_relu6', 'linear', etc.

    # Computation
    flops: int
    macs: int  # multiply-accumulate operations

    # Memory
    input_tensors: List[TensorDescriptor] = field(default_factory=list)
    output_tensors: List[TensorDescriptor] = field(default_factory=list)
    weight_tensors: List[TensorDescriptor] = field(default_factory=list)
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    total_weight_bytes: int = 0
    arithmetic_intensity: float = 0.0  # FLOPs / total_bytes

    # Parallelism
    parallelism: Optional[ParallelismDescriptor] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # node_ids
    dependency_type: str = "sequential"  # 'sequential', 'independent', 'partial'

    # Hardware hints
    recommended_bottleneck: BottleneckType = BottleneckType.BALANCED

    # Partition reasoning
    partition_reason: PartitionReason = PartitionReason.OPERATION_BOUNDARY
    partition_criteria: Dict[str, any] = field(default_factory=dict)  # Supporting data for the decision
    fusion_candidates: List[str] = field(default_factory=list)  # Node IDs that could have been fused

    def __post_init__(self):
        """Compute derived fields"""
        if self.flops > 0 and (self.total_input_bytes + self.total_output_bytes + self.total_weight_bytes) > 0:
            total_bytes = self.total_input_bytes + self.total_output_bytes + self.total_weight_bytes
            self.arithmetic_intensity = self.flops / total_bytes

            # Classify bottleneck based on arithmetic intensity
            if self.arithmetic_intensity > 50:
                self.recommended_bottleneck = BottleneckType.COMPUTE_BOUND
            elif self.arithmetic_intensity > 10:
                self.recommended_bottleneck = BottleneckType.BALANCED
            elif self.arithmetic_intensity > 1:
                self.recommended_bottleneck = BottleneckType.MEMORY_BOUND
            else:
                self.recommended_bottleneck = BottleneckType.BANDWIDTH_BOUND

    def partition_reasoning_summary(self) -> str:
        """Generate human-readable explanation of partition reasoning"""
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
    """Complete statistics from graph partitioning"""

    # Subgraphs
    subgraphs: List[SubgraphDescriptor]
    total_subgraphs: int

    # Computation totals
    total_flops: int
    total_macs: int
    total_memory_traffic: int  # input + output + weights

    # Arithmetic intensity distribution
    average_arithmetic_intensity: float
    min_arithmetic_intensity: float
    max_arithmetic_intensity: float

    # Subgraph type distribution
    operation_type_counts: Dict[str, int] = field(default_factory=dict)
    fusion_pattern_counts: Dict[str, int] = field(default_factory=dict)

    # Parallelism distribution
    parallelism_distribution: Dict[str, int] = field(default_factory=dict)  # '<1K', '1K-10K', etc.

    # Bottleneck analysis
    bottleneck_distribution: Dict[str, int] = field(default_factory=dict)

    # Partition reasoning
    partition_reason_distribution: Dict[str, int] = field(default_factory=dict)

    # Concurrency analysis
    concurrency: Optional[ConcurrencyDescriptor] = None

    # Critical path
    critical_path_subgraphs: List[str] = field(default_factory=list)

    def summary_stats(self) -> str:
        """Generate human-readable summary"""
        return f"""
Graph Partition Summary
=======================
Total subgraphs: {self.total_subgraphs}
Total FLOPs: {self.total_flops / 1e9:.2f} G
Total memory traffic: {self.total_memory_traffic / 1e6:.2f} MB
Average arithmetic intensity: {self.average_arithmetic_intensity:.2f} FLOPs/byte

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
