"""
TPU Hardware Mapper - Maps fused subgraphs to TPU systolic arrays

This module implements realistic TPU mapping considering:
- Systolic array allocation (128×128 matrix units)
- Matrix vs vector operation routing
- High batch size optimization
- BF16/INT8 quantization (2× speedup for INT8)
- Deep pipeline depth for matrix operations

Key differences from GPU:
- Fewer compute units (2 TensorCores vs 132 SMs)
- Massive systolic arrays (128×128 = 16K MACs)
- Optimized for matrix multiplication (Conv, Linear)
- Less flexible than GPU (no dynamic branching)
- Best performance at large batch sizes (64+)
- BF16 is the native precision (FP32 is emulated)

Example:
  ResNet-18 Conv layer with 64 channels:
  - Matrix op: Use systolic array (275 TFLOPS BF16)
  - Element-wise (ReLU): Use vector units (~10% of systolic array performance)
  - Batch=1: Low utilization (~10-20%)
  - Batch=64: High utilization (~80-90%)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from ...resource_model import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport
from graphs.ir.structures import BottleneckType


@dataclass
class TPUAllocation:
    """
    TPU allocation analysis for an operation.

    TPU routes operations to either:
    - Systolic array: Matrix operations (Conv, Linear, MatMul)
    - Vector units: Element-wise operations (ReLU, Add, etc.)
    """
    uses_systolic_array: bool  # True if matrix op (Conv, Linear, MatMul)
    systolic_array_utilization: float  # 0.0 to 1.0 (128×128 array usage)

    # For systolic array ops
    matrix_rows: int  # Rows in matrix operation
    matrix_cols: int  # Columns in matrix operation
    matrix_depth: int  # Depth (K dimension for A×B where A is M×K, B is K×N)

    # Batch size dependency
    batch_scaling_factor: float  # How much batch size helps (1.0 = no help, 2.0 = 2× better)

    # Pipeline depth
    pipeline_depth: int  # Systolic array pipeline depth (128 typical)
    pipeline_fill_overhead: float  # Overhead to fill/drain pipeline


class TPUMapper(HardwareMapper):
    """
    TPU hardware mapper using systolic array + vector unit allocation.

    Implements realistic TPU mapping considering:
    - Systolic array allocation for matrix ops
    - Vector unit allocation for element-wise ops
    - Batch size scaling (TPU loves large batches)
    - BF16/INT8 quantization benefits
    - Pipeline depth overhead
    """

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        thermal_profile: str = None
    ):
        """
        Initialize TPU mapper.

        Args:
            resource_model: TPU resource model
            thermal_profile: Thermal profile name (if applicable)
                           If None, uses default from resource model
        """
        super().__init__(resource_model, thermal_profile=thermal_profile)

        # Validate this is a TPU model
        if resource_model.hardware_type.value != "tpu":
            raise ValueError(f"TPUMapper requires TPU resource model, got {resource_model.hardware_type}")

        # TPU-specific parameters
        self.num_tensor_cores = resource_model.compute_units  # 2 TensorCores
        self.systolic_array_size = resource_model.warp_size  # 128 (128×128 array)
        self.threads_per_core = resource_model.threads_per_unit  # 16,384 (128×128)

    def _analyze_operation_type(
        self,
        subgraph: FusedSubgraph,
    ) -> TPUAllocation:
        """
        Analyze if operation uses systolic array or vector units.

        Args:
            subgraph: Fused subgraph to analyze

        Returns:
            TPUAllocation with routing analysis
        """
        # Check if this is a matrix operation
        is_matrix_op = any(op.value in ['conv2d', 'linear', 'matmul', 'mm', 'bmm']
                          for op in subgraph.operation_types)

        # Systolic array is for matrix ops
        uses_systolic_array = is_matrix_op

        if uses_systolic_array:
            # Matrix operations: Estimate matrix dimensions
            # For Conv2D: M = batch * out_h * out_w, N = out_channels, K = in_channels * kernel_h * kernel_w
            # For Linear: M = batch, N = out_features, K = in_features

            # Rough estimate from FLOPS
            # FLOPS = 2 * M * N * K (for matrix multiply)
            # We'll estimate dimensions assuming square-ish matrices
            total_ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2

            # Estimate: Assume M ≈ N ≈ sqrt(ops / (2 * K))
            # For Conv2D, K is typically 9-576 (3×3 to 24×24 kernel × channels)
            estimated_K = 64  # Conservative estimate for typical ResNet

            if total_ops > 0:
                matrix_size_product = total_ops / (2 * estimated_K)
                matrix_size = int(math.sqrt(max(1, matrix_size_product)))
            else:
                matrix_size = 128  # Default

            matrix_rows = matrix_size
            matrix_cols = matrix_size
            matrix_depth = estimated_K

            # Systolic array utilization depends on matrix size
            # 128×128 array is fully utilized if M ≥ 128 and N ≥ 128
            utilization_M = min(1.0, matrix_rows / self.systolic_array_size)
            utilization_N = min(1.0, matrix_cols / self.systolic_array_size)
            systolic_array_utilization = utilization_M * utilization_N

            # Pipeline depth (128 for systolic array)
            pipeline_depth = self.systolic_array_size

            # Pipeline fill overhead: small matrices have higher overhead
            # Overhead = pipeline_depth / (matrix_depth + pipeline_depth)
            if matrix_depth > 0:
                pipeline_fill_overhead = pipeline_depth / (matrix_depth + pipeline_depth)
            else:
                pipeline_fill_overhead = 0.5  # Conservative

            # Batch scaling: TPU benefits from large batches
            # Assume batch=1 → 20% utilization, batch=64 → 80% utilization
            # batch_scaling_factor = 1.0 + log2(batch) * 0.15
            batch_scaling_factor = 1.0  # Will be updated based on actual batch
        else:
            # Vector operations: Lower performance (~10% of systolic array)
            matrix_rows = 0
            matrix_cols = 0
            matrix_depth = 0
            systolic_array_utilization = 0.0
            pipeline_depth = 1
            pipeline_fill_overhead = 0.0
            batch_scaling_factor = 1.0

        return TPUAllocation(
            uses_systolic_array=uses_systolic_array,
            systolic_array_utilization=systolic_array_utilization,
            matrix_rows=matrix_rows,
            matrix_cols=matrix_cols,
            matrix_depth=matrix_depth,
            batch_scaling_factor=batch_scaling_factor,
            pipeline_depth=pipeline_depth,
            pipeline_fill_overhead=pipeline_fill_overhead,
        )

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.BF16  # TPU default is BF16
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to TPU.

        Algorithm:
        1. Determine if operation uses systolic array or vector units
        2. Calculate systolic array utilization (depends on matrix size)
        3. Account for batch size scaling
        4. Apply pipeline fill overhead for small matrices
        5. Calculate latency using roofline model
        """
        # Analyze operation type
        tpu_alloc = self._analyze_operation_type(subgraph)

        # Get parallelism
        if subgraph.parallelism is None:
            # Fallback: assume minimal parallelism
            threads_required = self.threads_per_core
            cores_allocated = self.num_tensor_cores
        else:
            # TPU parallelism is limited by TensorCore count
            parallelism = subgraph.parallelism.total_threads

            # Calculate cores needed
            threads_per_core = self.threads_per_core
            cores_needed = math.ceil(parallelism / threads_per_core)

            # Allocate up to all cores
            cores_allocated = min(cores_needed, self.num_tensor_cores)

        cores_allocated = max(1, cores_allocated)  # At least 1 core
        threads_required = cores_allocated * self.threads_per_core

        # Calculate occupancy
        # For systolic array ops, occupancy depends on matrix size
        if tpu_alloc.uses_systolic_array:
            occupancy = tpu_alloc.systolic_array_utilization
        else:
            # Vector ops: standard occupancy
            occupancy = cores_allocated / self.num_tensor_cores

        # Calculate utilization
        utilization = cores_allocated / self.num_tensor_cores

        # Calculate latency using roofline model
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        )

        # Adjust ops for operation type
        effective_ops = ops
        if not tpu_alloc.uses_systolic_array:
            # Vector ops are ~10× slower than systolic array
            # (not using specialized hardware)
            effective_ops = ops * 10.0

        # Apply pipeline fill overhead for systolic array
        if tpu_alloc.uses_systolic_array:
            # Pipeline overhead makes small matrices less efficient
            overhead_factor = 1.0 + tpu_alloc.pipeline_fill_overhead
            effective_ops = int(effective_ops * overhead_factor)

        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=int(effective_ops),
            bytes_transferred=bytes_transferred,
            allocated_units=cores_allocated,
            occupancy=occupancy,
            precision=precision
        )

        estimated_latency = max(compute_time, memory_time)

        # Calculate energy
        compute_energy, memory_energy = self._calculate_energy(
            ops=ops,
            bytes_transferred=bytes_transferred,
            precision=precision
        )
        total_energy = compute_energy + memory_energy

        # Check if this can run in parallel with others
        is_parallel = concurrent_subgraphs > 1

        return HardwareAllocation(
            subgraph_id=str(subgraph.subgraph_id),
            subgraph_name=", ".join(subgraph.node_names[:2]),
            precision=precision,
            threads_required=threads_required,
            warps_required=0,  # TPU doesn't use warps like GPU
            compute_units_allocated=cores_allocated,
            compute_units_ideal=cores_allocated,
            occupancy=occupancy,
            utilization=utilization,
            bottleneck=bottleneck,
            compute_time=compute_time,
            memory_time=memory_time,
            estimated_latency=estimated_latency,
            compute_energy=compute_energy,
            memory_energy=memory_energy,
            total_energy=total_energy,
            execution_stage=execution_stage,
            is_parallel=is_parallel,
        )

    def map_graph(
        self,
        fusion_report: FusionReport,
        execution_stages: List[List[int]],
        batch_size: int = 1,
        precision: Precision = Precision.BF16
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to TPU.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: Execution stages with subgraph indices
            batch_size: Batch size (scales parallelism, critical for TPU!)
            precision: Numerical precision (BF16/INT8 preferred)

        Returns:
            Complete hardware allocation
        """
        subgraph_allocations: List[HardwareAllocation] = []
        latency_breakdown: Dict[int, float] = {}

        peak_cores_used = 0
        total_cores_used = 0
        total_cores_samples = 0

        # Process each execution stage
        for stage_id, subgraph_indices in enumerate(execution_stages):
            stage_allocations = []
            concurrent_subgraphs = len(subgraph_indices)

            # Map each subgraph in this stage
            for subgraph_idx in subgraph_indices:
                if subgraph_idx >= len(fusion_report.fused_subgraphs):
                    continue

                subgraph = fusion_report.fused_subgraphs[subgraph_idx]

                # Scale parallelism by batch size
                if subgraph.parallelism and batch_size > 1:
                    import copy
                    subgraph = copy.copy(subgraph)
                    subgraph.parallelism = copy.copy(subgraph.parallelism)
                    subgraph.parallelism.batch *= batch_size
                    subgraph.parallelism.total_threads *= batch_size

                allocation = self.map_subgraph(
                    subgraph=subgraph,
                    execution_stage=stage_id,
                    concurrent_subgraphs=concurrent_subgraphs,
                    precision=precision
                )
                stage_allocations.append(allocation)
                subgraph_allocations.append(allocation)

            # For parallel subgraphs: max cores used, max latency
            if stage_allocations:
                stage_cores_used = max(a.compute_units_allocated for a in stage_allocations)
                stage_latency = max(a.estimated_latency for a in stage_allocations)

                peak_cores_used = max(peak_cores_used, stage_cores_used)
                total_cores_used += stage_cores_used
                total_cores_samples += 1

                latency_breakdown[stage_id] = stage_latency

        # Calculate aggregate metrics
        total_subgraphs = len(subgraph_allocations)
        total_execution_stages = len(execution_stages)
        average_cores_used = total_cores_used / total_cores_samples if total_cores_samples > 0 else 0

        total_cores = self.resource_model.compute_units
        peak_utilization = peak_cores_used / total_cores
        average_utilization = average_cores_used / total_cores

        # Total latency = sum of stage latencies (sequential)
        total_latency = sum(latency_breakdown.values())

        # Total energy
        total_energy = sum(a.total_energy for a in subgraph_allocations)

        # Naive latency (assuming 100% utilization)
        total_ops = fusion_report.total_flops
        peak_ops_per_sec = self.resource_model.get_peak_ops(precision)
        naive_latency = total_ops / peak_ops_per_sec if peak_ops_per_sec > 0 else 0

        # Correction factor
        latency_correction_factor = total_latency / naive_latency if naive_latency > 0 else 1.0

        # Bottleneck analysis
        compute_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.COMPUTE_BOUND)
        memory_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.MEMORY_BOUND)
        bandwidth_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BANDWIDTH_BOUND)
        balanced_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BALANCED)

        return GraphHardwareAllocation(
            model_name="Unknown",
            hardware_name=self.resource_model.name,
            batch_size=batch_size,
            model_precision=precision,
            subgraph_allocations=subgraph_allocations,
            total_subgraphs=total_subgraphs,
            total_execution_stages=total_execution_stages,
            peak_compute_units_used=peak_cores_used,
            average_compute_units_used=average_cores_used,
            peak_utilization=peak_utilization,
            average_utilization=average_utilization,
            total_latency=total_latency,
            latency_breakdown=latency_breakdown,
            total_energy=total_energy,
            naive_latency=naive_latency,
            latency_correction_factor=latency_correction_factor,
            compute_bound_count=compute_bound_count,
            memory_bound_count=memory_bound_count,
            bandwidth_bound_count=bandwidth_bound_count,
            balanced_count=balanced_count,
        )


def create_tpu_v4_mapper(thermal_profile: str = None) -> TPUMapper:
    """
    Create TPU mapper for Google TPU v4.

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        TPUMapper configured for TPU v4
    """
    from ...resource_model import tpu_v4_resource_model

    model = tpu_v4_resource_model()
    return TPUMapper(model, thermal_profile=thermal_profile)


def create_coral_edge_tpu_mapper(thermal_profile: str = None) -> TPUMapper:
    """
    Create TPU mapper for Google Coral Edge TPU.

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        TPUMapper configured for Coral Edge TPU (ultra-low-power edge AI)
    """
    from ...resource_model import coral_edge_tpu_resource_model

    model = coral_edge_tpu_resource_model()
    return TPUMapper(model, thermal_profile=thermal_profile)
