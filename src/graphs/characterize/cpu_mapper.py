"""
CPU Hardware Mapper - Maps fused subgraphs to CPU cores with SIMD/Vector units

This module implements realistic CPU mapping considering:
- Multi-core allocation (8-16 cores typical)
- SIMD/Vector units (AVX-2, AVX-512, ARM NEON)
- Advanced Matrix Extensions (AMX) for newer Intel CPUs
- Cache hierarchy (L1/L2/L3)
- Memory bandwidth limitations (DDR4/DDR5)

Key differences from GPU:
- Fewer cores (8-16 vs 132 SMs)
- Lower parallelism per core (SIMD 8-16 wide vs 2048 threads/SM)
- Deeper cache hierarchy (L1/L2/L3 vs L1/L2)
- Much lower memory bandwidth (80 GB/s vs 2 TB/s)
- More flexible (can run irregular workloads)

Example:
  ResNet-18 Conv layer with 200K operations:
  - 200K ops / 16 cores = 12.5K ops/core
  - With AVX-512 (16-wide SIMD): 12.5K / 16 = 781 vector ops/core
  - Memory bandwidth often limits performance (not compute)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from .hardware_mapper import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
)
from .fusion_partitioner import FusedSubgraph, FusionReport
from .graph_structures import BottleneckType


@dataclass
class CPUVectorization:
    """
    CPU vectorization analysis for an operation.

    CPUs use SIMD (Single Instruction Multiple Data) to process
    multiple elements in parallel within a single core.
    """
    simd_width: int  # Elements processed in parallel (8 for AVX-2, 16 for AVX-512)
    vector_operations: int  # Number of SIMD operations needed
    scalar_operations: int  # Remaining scalar operations (not vectorized)
    vectorization_efficiency: float  # 0.0 to 1.0 (how well vectorized)

    # Special accelerators
    uses_amx: bool = False  # Intel Advanced Matrix Extensions (BF16/INT8 matrix ops)
    uses_vnni: bool = False  # Vector Neural Network Instructions (INT8 dot products)


class CPUMapper(HardwareMapper):
    """
    CPU hardware mapper using core + SIMD/vector unit allocation.

    Implements realistic CPU mapping considering:
    - Core allocation based on parallelism
    - SIMD vectorization (AVX-2, AVX-512)
    - Cache hierarchy effects
    - Memory bandwidth constraints
    - Threading overhead
    """

    def __init__(self, resource_model: HardwareResourceModel):
        super().__init__(resource_model)

        # Validate this is a CPU model
        if resource_model.hardware_type.value != "cpu":
            raise ValueError(f"CPUMapper requires CPU resource model, got {resource_model.hardware_type}")

        # CPU-specific parameters
        self.simd_width = resource_model.warp_size  # Reuse warp_size for SIMD width
        self.cores = resource_model.compute_units
        self.threads_per_core = resource_model.threads_per_unit  # SMT/HyperThreading

    def _analyze_vectorization(
        self,
        subgraph: FusedSubgraph,
        precision: Precision
    ) -> CPUVectorization:
        """
        Analyze how well an operation vectorizes on CPU SIMD units.

        Args:
            subgraph: Fused subgraph to analyze
            precision: Numerical precision (affects SIMD width)

        Returns:
            CPUVectorization with vectorization analysis
        """
        # Get effective SIMD width for this precision
        # AVX-512: 16 FP32, 32 FP16, 64 INT8
        # AVX-2:   8 FP32, 16 FP16, 32 INT8
        bytes_per_element = self._get_bytes_per_element(precision)
        vector_register_bytes = 64 if self.simd_width >= 16 else 32  # 512-bit or 256-bit
        effective_simd_width = vector_register_bytes // bytes_per_element

        # For matrix operations (Conv, Linear), vectorization is good
        # For element-wise ops (ReLU, Add), vectorization is excellent
        is_matrix_op = any(op.value in ['conv2d', 'linear', 'matmul']
                          for op in subgraph.operation_types)
        is_elementwise = any(op.value in ['relu', 'relu6', 'gelu', 'sigmoid']
                            for op in subgraph.operation_types)

        # Estimate vectorization efficiency
        if is_elementwise:
            vectorization_efficiency = 0.95  # Element-wise ops vectorize very well
        elif is_matrix_op:
            vectorization_efficiency = 0.80  # Matrix ops have some overhead
        else:
            vectorization_efficiency = 0.70  # Conservative estimate

        # Calculate vector operations
        total_ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        vectorizable_ops = int(total_ops * vectorization_efficiency)
        vector_operations = math.ceil(vectorizable_ops / effective_simd_width)
        scalar_operations = total_ops - vectorizable_ops

        # Check for special accelerators (AMX for matrix ops, VNNI for INT8)
        uses_amx = is_matrix_op and precision in [Precision.BF16, Precision.INT8] and self.simd_width >= 16
        uses_vnni = is_matrix_op and precision == Precision.INT8 and self.simd_width >= 8

        return CPUVectorization(
            simd_width=effective_simd_width,
            vector_operations=vector_operations,
            scalar_operations=scalar_operations,
            vectorization_efficiency=vectorization_efficiency,
            uses_amx=uses_amx,
            uses_vnni=uses_vnni,
        )

    def _get_bytes_per_element(self, precision: Precision) -> int:
        """Get bytes per element for a precision"""
        bytes_map = {
            Precision.FP64: 8,
            Precision.FP32: 4,
            Precision.FP16: 2,
            Precision.BF16: 2,
            Precision.FP8_E4M3: 1,
            Precision.FP8_E5M2: 1,
            Precision.INT32: 4,
            Precision.INT16: 2,
            Precision.INT8: 1,
            Precision.INT4: 0.5,  # Packed
        }
        return bytes_map.get(precision, 4)

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.FP32
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to CPU cores.

        Algorithm:
        1. Analyze vectorization potential (SIMD width, efficiency)
        2. Determine core allocation based on parallelism
        3. Calculate occupancy (limited by core count)
        4. Calculate latency using roofline model
        5. Account for threading overhead
        """
        # Analyze vectorization
        vectorization = self._analyze_vectorization(subgraph, precision)

        # Get parallelism
        if subgraph.parallelism is None:
            # Fallback: assume minimal parallelism
            threads_required = self.cores
            cores_allocated = self.cores
        else:
            # CPU parallelism is limited by core count
            # Can't exceed cores (even with SMT, doesn't help much for compute)
            parallelism = subgraph.parallelism.total_threads

            # Heuristic: Each core handles batch_size * (channels/cores) * spatial
            # For CPU, batch parallelism is primary
            batch = subgraph.parallelism.batch

            # Allocate up to all cores, but don't over-allocate
            cores_allocated = min(batch, self.cores)

            # If batch < cores, can still use multiple cores for spatial/channel parallelism
            if cores_allocated < self.cores:
                # Try to use more cores for intra-op parallelism
                # But effectiveness diminishes (Amdahl's law)
                extra_parallelism = min(
                    self.cores - cores_allocated,
                    subgraph.parallelism.channels // 4  # Need enough work per core
                )
                cores_allocated = min(cores_allocated + extra_parallelism, self.cores)

        cores_allocated = max(1, cores_allocated)  # At least 1 core
        threads_required = cores_allocated  # 1 thread per core (SMT doesn't help compute)

        # Calculate occupancy (what fraction of cores are busy)
        occupancy = cores_allocated / self.cores

        # Calculate utilization (cores used / total cores)
        utilization = cores_allocated / self.cores

        # Calculate latency using roofline model
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        )

        # Adjust ops for vectorization
        # Vectorized ops are faster, but not as dramatic as GPU
        effective_ops = ops
        if vectorization.vectorization_efficiency > 0:
            # SIMD gives speedup, but not full width due to overhead
            simd_speedup = vectorization.simd_width * 0.7  # 70% efficiency
            effective_ops = ops / simd_speedup

        # Special accelerator boost
        if vectorization.uses_amx:
            # AMX can provide 2-4× speedup for matrix ops
            effective_ops = ops / 3.0
        elif vectorization.uses_vnni:
            # VNNI provides ~2× speedup for INT8 dot products
            effective_ops = ops / 2.0

        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=int(effective_ops),
            bytes_transferred=bytes_transferred,
            allocated_units=cores_allocated,
            occupancy=occupancy,
            precision=precision
        )

        # Add threading overhead (OpenMP-style parallelism has overhead)
        # More cores = more overhead for synchronization
        threading_overhead = 1.0 + (cores_allocated - 1) * 0.02  # 2% overhead per additional core
        compute_time *= threading_overhead

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
            warps_required=0,  # N/A for CPU
            compute_units_allocated=cores_allocated,
            compute_units_ideal=cores_allocated,  # No quantization on CPU
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
        precision: Precision = Precision.FP32
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to CPU.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: Execution stages with subgraph indices
            batch_size: Batch size (scales parallelism)
            precision: Numerical precision

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


def create_intel_cpu_mapper(simd_type: str = "avx512") -> CPUMapper:
    """
    Create CPU mapper for Intel CPU with specified SIMD support.

    Args:
        simd_type: "avx2", "avx512", or "amx" (includes AVX-512)

    Returns:
        CPUMapper configured for Intel CPU
    """
    from .hardware_mapper import cpu_x86_resource_model

    model = cpu_x86_resource_model()

    # Adjust SIMD width based on type
    if simd_type == "avx2":
        model.warp_size = 8  # 8-wide SIMD (256-bit)
    elif simd_type == "avx512":
        model.warp_size = 16  # 16-wide SIMD (512-bit)
    elif simd_type == "amx":
        model.warp_size = 16  # AMX includes AVX-512
        # AMX already modeled in precision profiles

    return CPUMapper(model)


def create_amd_cpu_mapper() -> CPUMapper:
    """Create CPU mapper for AMD Ryzen CPU (AVX-2)"""
    from .hardware_mapper import cpu_x86_resource_model

    model = cpu_x86_resource_model()
    model.name = "AMD-Ryzen-7-16core"
    model.warp_size = 8  # AVX-2 (256-bit)
    model.peak_flops = 1.0e12  # 1.0 TFLOPS (slightly lower than Intel)

    return CPUMapper(model)
