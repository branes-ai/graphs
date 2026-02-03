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

from ..resource_model import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport
from graphs.core.structures import BottleneckType


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

    # Idle power fraction (nanoscale transistor leakage)
    # Modern datacenter CPUs consume ~50% TDP at idle
    IDLE_POWER_FRACTION = 0.5

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        thermal_profile: str = None,
        calibration = None  # Type hint: Optional[HardwareCalibration]
    ):
        super().__init__(resource_model, thermal_profile=thermal_profile)

        # Validate this is a CPU model
        if resource_model.hardware_type.value != "cpu":
            raise ValueError(f"CPUMapper requires CPU resource model, got {resource_model.hardware_type}")

        # CPU-specific parameters
        self.simd_width = resource_model.warp_size  # Reuse warp_size for SIMD width
        self.cores = resource_model.compute_units
        self.threads_per_core = resource_model.threads_per_unit  # SMT/HyperThreading

        # Calibration data (if provided)
        self.calibration = calibration
        self.default_efficiency = 0.20  # Fallback if no calibration available

    def compute_energy_with_idle_power(
        self,
        latency: float,
        dynamic_energy: float
    ) -> Tuple[float, float]:
        """
        Compute total energy including idle power consumption.

        Modern datacenter CPUs consume significant power even at idle due to:
        - Transistor leakage in nanoscale processes (7nm, 5nm, 3nm)
        - Always-on circuitry (memory controllers, PCIe, interconnects)
        - Typical idle consumption: ~50% of TDP

        Power model:
        P_total = P_idle + P_dynamic
        P_idle = TDP × 0.5 (constant, independent of frequency/DVFS)
        P_dynamic = dynamic_energy / latency

        Args:
            latency: Total execution time (seconds)
            dynamic_energy: Energy from computation and memory transfers (Joules)

        Returns:
            (total_energy, average_power)
        """
        if latency <= 0:
            return dynamic_energy, 0.0

        # Get TDP from thermal operating point if available
        # Datacenter CPUs have multiple thermal profiles (different core counts, frequencies)
        tdp_watts = None
        if self.thermal_profile and self.resource_model.thermal_operating_points:
            thermal_point = self.resource_model.thermal_operating_points.get(self.thermal_profile)
            if thermal_point:
                tdp_watts = thermal_point.tdp_watts

        # If no thermal profile specified, try to use the first available one
        if tdp_watts is None and self.resource_model.thermal_operating_points:
            # Try "default" first
            default_thermal = self.resource_model.thermal_operating_points.get("default")
            if default_thermal:
                tdp_watts = default_thermal.tdp_watts
            else:
                # Use the first available thermal profile
                first_profile = next(iter(self.resource_model.thermal_operating_points.values()), None)
                if first_profile:
                    tdp_watts = first_profile.tdp_watts

        # If still no TDP, estimate from dynamic power
        # For datacenter CPUs: dynamic power is typically ~40-60% of TDP at peak
        if tdp_watts is None:
            dynamic_power = dynamic_energy / latency if latency > 0 else 0
            tdp_watts = dynamic_power * 2.0

        # Idle power: ~50% TDP constantly consumed (leakage + always-on circuits)
        # Note: This does NOT scale with DVFS - leakage is largely independent of frequency
        idle_power = tdp_watts * self.IDLE_POWER_FRACTION
        idle_energy = idle_power * latency

        # Total energy = idle + dynamic
        total_energy = idle_energy + dynamic_energy

        # Average power during execution
        average_power = total_energy / latency

        return total_energy, average_power

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

    def _classify_operation(self, subgraph: FusedSubgraph) -> Tuple[str, Dict]:
        """
        Classify the dominant operation type in a subgraph.

        This method analyzes the operations in a subgraph and determines
        the primary operation type for calibration lookup.

        Args:
            subgraph: Fused subgraph to classify

        Returns:
            (operation_type, extra_params) tuple for calibration lookup
        """
        # Extract operation types
        ops = [op.value for op in subgraph.operation_types]

        # Matrix multiplication (highest priority for compute-bound ops)
        if any(op in ['matmul', 'linear', 'addmm', 'bmm'] for op in ops):
            # Estimate matrix size from memory footprint
            # For matmul: C = A @ B where A is [M, K], B is [K, N], C is [M, N]
            # Memory = M*K + K*N + M*N elements
            # Approximate N ≈ sqrt(memory / (3 * bytes_per_element))
            total_elements = subgraph.total_input_bytes + subgraph.total_output_bytes
            total_elements //= 4  # FP32
            matrix_size = int((total_elements / 3) ** 0.5)  # Rough estimate

            # Categorize by size
            if matrix_size < 768:
                size_category = 'small'
            elif matrix_size < 3072:
                size_category = 'medium'
            else:
                size_category = 'large'

            return 'matmul', {'matrix_size': matrix_size}

        # Convolution
        if any('conv' in op for op in ops):
            return 'conv2d', {}

        # Element-wise operations (memory-bound)
        if any(op in ['relu', 'relu6', 'gelu', 'sigmoid', 'add', 'mul', 'sub'] for op in ops):
            # Use 'add' as proxy for memory-bound elementwise ops
            return 'add', {}

        # Pooling
        if any('pool' in op for op in ops):
            return 'maxpool', {}

        # Default: unknown operation
        return 'unknown', {}

    def _get_calibrated_efficiency(self, subgraph: FusedSubgraph) -> float:
        """
        Get efficiency for this subgraph using calibration data.

        Args:
            subgraph: Fused subgraph to analyze

        Returns:
            Efficiency (0.0 to 1.0)
        """
        if not self.calibration:
            # No calibration: use conservative default
            return self.default_efficiency

        # Classify operation
        op_type, extra_params = self._classify_operation(subgraph)

        # Query calibration
        efficiency = self.calibration.get_efficiency(op_type, **extra_params)

        # Sanity check
        if efficiency <= 0 or efficiency > 1.0:
            # Invalid efficiency, use default
            return self.default_efficiency

        return efficiency

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

        # Apply calibration correction if available
        # Calibration provides operation-specific efficiency that overrides
        # the generic efficiency factors from the resource model
        if self.calibration:
            # Compute efficiency correction
            calibrated_efficiency = self._get_calibrated_efficiency(subgraph)
            base_efficiency = self.default_efficiency
            efficiency_correction = base_efficiency / calibrated_efficiency
            compute_time *= efficiency_correction

            # Memory bandwidth correction
            # If measured bandwidth is lower than theoretical, memory ops are slower
            theoretical_bandwidth = self.resource_model.peak_bandwidth
            measured_bandwidth = self.calibration.measured_bandwidth_gbps * 1e9  # Convert GB/s to bytes/s
            bandwidth_correction = theoretical_bandwidth / measured_bandwidth
            memory_time *= bandwidth_correction

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

        # Total energy: dynamic energy from compute + memory, plus idle power
        dynamic_energy = sum(a.total_energy for a in subgraph_allocations)
        total_energy, average_power = self.compute_energy_with_idle_power(
            total_latency, dynamic_energy
        )

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
    from ..models.datacenter.cpu_x86 import cpu_x86_resource_model

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
    from ..models.datacenter.cpu_x86 import cpu_x86_resource_model

    model = cpu_x86_resource_model()
    model.name = "AMD-Ryzen-7-16core"
    model.warp_size = 8  # AVX-2 (256-bit)
    model.peak_flops = 1.0e12  # 1.0 TFLOPS (slightly lower than Intel)

    return CPUMapper(model)


def create_i7_12700k_mapper() -> CPUMapper:
    """
    Create CPU mapper for Intel Core i7-12700K (12th Gen Alder Lake).

    HYBRID ARCHITECTURE (Performance + Efficiency cores):
    - 8 P-cores @ 5.0 GHz (16 threads with HT)
    - 4 E-cores @ 3.8 GHz (4 threads, no HT)
    - Total: 12 cores, 20 threads
    - Effective performance: ~10 P-core equivalents (E-cores ≈ 0.6× P-cores)

    KEY DIFFERENCES vs Datacenter Xeon:
    - AVX2 only (NO AVX-512) → 8-wide FP32 SIMD
    - Hybrid scheduling overhead → lower utilization
    - Consumer DDR5 bandwidth → ~75 GB/s (vs 80+ GB/s server)
    - Smaller L2 cache per core
    - Much lower efficiency_factor for small batches (~12-15% vs 70%)

    CALIBRATION DATA (from empirical sweep on tiny MLPs):
    - Tiny MLPs (batch 1-32): 57.8% MAPE with efficiency_factor=0.12
    - Final calibrated values (compromise):
      - efficiency_factor: 0.20 (20% of peak)
      - memory_bottleneck_factor: 0.25 (tiny models are memory-starved)
    - These values optimize for SMALL models at LOW batch sizes

    ⚠ ACCURACY TRADE-OFF WARNING:
    These coefficients are tuned for tiny MLPs (batch 1-32). For different
    model types, accuracy will vary:

    Model Type                  | Expected MAPE | Why
    ----------------------------|---------------|---------------------------
    Tiny MLPs (batch 1-32)      | 10-20%        | ✓ Optimized for this
    Medium CNNs (batch 16-64)   | 15-25%        | Moderate (more compute-bound)
    Large Transformers (batch≥64)| 25-40%       | ⚠ Over-pessimistic (different bottleneck)
    Vision models (ResNet, etc) | 10-30%        | Good (mixed workload)

    REASON: Large transformers are compute-bound with high arithmetic intensity,
    so low memory_bottleneck_factor (0.25) over-estimates memory impact.

    SOLUTION: Create separate mappers for different workload classes:
      - create_i7_12700k_tiny_mapper() → Small models, batch 1-32
      - create_i7_12700k_large_mapper() → Large models, batch≥64

    Use Cases:
    - Edge AI with small batch sizes
    - Real-time inference (batch=1)
    - Consumer hardware benchmarking
    - Laptop/desktop deployment
    """
    from ..resource_model import (
        HardwareResourceModel,
        HardwareType,
        Precision,
        PrecisionProfile,
        ClockDomain,
        ComputeResource,
        ThermalOperatingPoint,
        PerformanceCharacteristics,
    )

    # ========================================================================
    # HYBRID CORE MODELING
    # ========================================================================
    # P-cores: 8 cores @ 5 GHz, 2 threads each
    # E-cores: 4 cores @ 3.8 GHz, 1 thread each
    # Effective cores for performance modeling: 8 + (4 * 0.6) = 10.4 ≈ 10
    p_cores = 8
    e_cores = 4
    e_core_efficiency = 0.6  # E-cores are ~60% of P-core performance
    effective_cores = p_cores + int(e_cores * e_core_efficiency)  # 10 effective cores

    # ========================================================================
    # CLOCK DOMAIN (P-cores dominate)
    # ========================================================================
    clock = ClockDomain(
        base_clock_hz=3.6e9,       # 3.6 GHz base (P-cores)
        max_boost_clock_hz=5.0e9,  # 5.0 GHz max boost (single P-core)
        sustained_clock_hz=4.5e9,  # 4.5 GHz all-core sustained
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE (AVX2 - 8-wide FP32)
    # ========================================================================
    # AVX2: 8 FP32 ops/cycle per core (FMA: 2 ops × 8 lanes × 2 units = 32 ops/cycle peak)
    # But realistic is ~16 ops/cycle (1 FMA unit active, not both)
    ops_per_core_per_cycle_fp32 = 16  # Conservative: 1 FMA unit
    ops_per_core_per_cycle_int8 = 32  # VNNI: better INT8 throughput

    avx2_compute = ComputeResource(
        resource_type="Intel-P-Core-AVX2",
        num_units=effective_cores,
        ops_per_unit_per_clock={
            Precision.FP32: ops_per_core_per_cycle_fp32,
            Precision.FP16: ops_per_core_per_cycle_fp32,  # Emulated (no native FP16)
            Precision.INT8: ops_per_core_per_cycle_int8,  # VNNI
        },
        clock_domain=clock,
    )

    # Peak FP32: 10 cores × 16 ops/cycle × 4.5 GHz = 720 GFLOPS sustained
    # Peak INT8: 10 cores × 32 ops/cycle × 4.5 GHz = 1.44 TOPS sustained

    # ========================================================================
    # THERMAL PROFILE (Consumer CPU - realistic for continuous workload)
    # ========================================================================
    thermal_profile = ThermalOperatingPoint(
        name="consumer-continuous",
        tdp_watts=125.0,  # PL1 (long-term)
        cooling_solution="tower-cooler",
        performance_specs={
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=avx2_compute,
                instruction_efficiency=0.65,  # Lower than Xeon (hybrid scheduling)
                memory_bottleneck_factor=0.25,
                # Measured efficiency across ResNet/ViT workloads: median ~50% of
                # theoretical peak (range 37-63%). CPUs have no tile quantization
                # so tile_utilization=1.0; efficiency_factor captures AVX2 ISA
                # overhead, cache miss rate, and hybrid core scheduling.
                efficiency_factor=0.50,
                tile_utilization=1.0,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=avx2_compute,
                efficiency_factor=0.25,  # Emulated, worse than FP32
                tile_utilization=1.0,
                native_acceleration=False,
            ),
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=avx2_compute,
                efficiency_factor=0.45,  # Better with VNNI
                tile_utilization=1.0,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    model = HardwareResourceModel(
        name="Intel-i7-12700K-Alder-Lake",
        hardware_type=HardwareType.CPU,
        compute_units=effective_cores,  # 10 effective cores
        threads_per_unit=2,  # Weighted average (P-cores dominate)
        warps_per_unit=1,
        warp_size=8,  # AVX2 (8-wide)

        # Precision profiles (for legacy compatibility)
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=720e9,  # 720 GFLOPS sustained
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=1.44e12,  # 1.44 TOPS sustained (VNNI)
                tensor_core_supported=True,  # VNNI counts as "tensor core"
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        # ====================================================================
        # MEMORY HIERARCHY - i7-12700K Cache Structure
        # ====================================================================
        # Physical cache hierarchy:
        #   L1: 32 KB data per core (private, fastest)
        #   L2: ~1.25 MB per P-core, ~2 MB per E-core (private, core-attached)
        #   L3: 25 MB shared across all cores (LLC - Last-Level Cache)
        #
        # PERFORMANCE MODELING RATIONALE:
        # The HardwareResourceModel parameter 'l2_cache_total' represents the
        # Last-Level Cache (LLC), NOT the physical L2 cache. For modern CPUs,
        # the LLC is L3, which is the shared cache that determines:
        #   1. When data spills to main memory (DRAM)
        #   2. Cache coherency overhead across cores
        #   3. Memory bandwidth pressure
        #
        # For tiling and partitioning decisions, the LLC size is the critical
        # metric because:
        #   - Working set fits in LLC → low latency (~50 cycles)
        #   - Working set spills to DRAM → high latency (~200+ cycles)
        #
        # Therefore: l2_cache_total = L3 size (25 MB) = LLC
        # ====================================================================
        peak_bandwidth=75e9,  # 75 GB/s DDR5 (consumer grade)
        l1_cache_per_unit=32 * 1024,  # 32 KB L1 data per core
        l2_cache_total=25 * 1024 * 1024,  # 25 MB L3 shared (LLC)
        main_memory=64 * 1024**3,  # 64 GB typical

        # Energy (consumer CPU)
        energy_per_flop_fp32=1.5e-12,  # Higher than server (less efficient)
        energy_per_byte=25e-12,

        # Scheduling
        min_occupancy=0.4,  # Hybrid scheduling challenges
        max_concurrent_kernels=20,  # 20 threads total
        wave_quantization=1,

        # Thermal profiles
        thermal_operating_points={
            "consumer-continuous": thermal_profile,
        },
        default_thermal_profile="consumer-continuous",
    )

    return CPUMapper(model)


def create_i7_12700k_large_mapper() -> CPUMapper:
    """
    Create CPU mapper for Intel Core i7-12700K optimized for LARGE models.

    SAME HARDWARE as create_i7_12700k_mapper():
    - 8 P-cores @ 5.0 GHz + 4 E-cores @ 3.8 GHz
    - AVX2 (8-wide FP32 SIMD)
    - 25 MB L3 cache (LLC)
    - 75 GB/s DDR5 bandwidth

    DIFFERENT CALIBRATION TARGET:
    This mapper is tuned for large models with high arithmetic intensity:
    - Large transformers (BERT, GPT, etc) at batch≥32
    - Deep CNNs (ResNet-50/101, EfficientNet) at batch≥16
    - Large vision models (ViT, DeiT) at batch≥32
    - Any model where working set >> L3 cache but compute dominates

    KEY DIFFERENCES vs create_i7_12700k_mapper() (tiny model variant):

    Parameter                      | Tiny Models | Large Models | Why Different
    -------------------------------|-------------|--------------|------------------
    efficiency_factor (FP32)       | 0.20        | 0.60         | Better amortization of overhead
    memory_bottleneck_factor       | 0.25        | 0.65         | Compute-bound vs memory-bound
    tile_utilization               | 0.50        | 0.80         | Better core saturation
    instruction_efficiency         | 0.65        | 0.80         | Better compiler optimization

    EXPECTED ACCURACY:

    Model Type                  | Expected MAPE | Why
    ----------------------------|---------------|---------------------------
    Tiny MLPs (batch 1-32)      | 50-80%        | ✗ Over-optimistic (wrong mapper!)
    Medium CNNs (batch 16-64)   | 15-25%        | ✓ Good fit
    Large Transformers (batch≥64)| 10-20%       | ✓ Excellent fit
    Vision models (ResNet-50+)  | 12-22%        | ✓ Good fit
    Batch matmuls (large tiles) | 8-15%         | ✓ Excellent fit

    REASON FOR IMPROVEMENT:
    Large models have:
    1. Higher arithmetic intensity → less memory bottleneck
    2. Longer execution times → overhead amortization
    3. Better SIMD utilization → wider vectors, fewer edge cases
    4. Better cache behavior → more reuse, less thrashing
    5. Better thread utilization → all cores busy

    WHEN TO USE WHICH MAPPER:
    - Use create_i7_12700k_mapper() for:
      * Real-time inference (batch=1)
      * Edge AI with tiny models
      * Rapid prototyping with small models

    - Use create_i7_12700k_large_mapper() for:
      * Training large models
      * Batch inference (batch≥32)
      * Large transformer deployment
      * Throughput-oriented workloads

    CALIBRATION STATUS:
    ⚠ These coefficients are ESTIMATED based on:
    - Extrapolation from tiny model calibration
    - Expected scaling behavior from architecture analysis
    - Typical performance characteristics of large models on similar CPUs

    TODO: Run empirical calibration sweep on large models to refine these values!
    """
    from ..resource_model import (
        HardwareResourceModel,
        HardwareType,
        Precision,
        PrecisionProfile,
        ClockDomain,
        ComputeResource,
        ThermalOperatingPoint,
        PerformanceCharacteristics,
    )

    # ========================================================================
    # HYBRID CORE MODELING (same as tiny model variant)
    # ========================================================================
    p_cores = 8
    e_cores = 4
    e_core_efficiency = 0.6
    effective_cores = p_cores + int(e_cores * e_core_efficiency)  # 10 effective cores

    # ========================================================================
    # CLOCK DOMAIN (same - hardware doesn't change)
    # ========================================================================
    clock = ClockDomain(
        base_clock_hz=3.6e9,
        max_boost_clock_hz=5.0e9,
        sustained_clock_hz=4.5e9,  # 4.5 GHz all-core sustained
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE (same - AVX2 8-wide FP32)
    # ========================================================================
    ops_per_core_per_cycle_fp32 = 16  # 1 FMA unit
    ops_per_core_per_cycle_int8 = 32  # VNNI

    avx2_compute = ComputeResource(
        resource_type="Intel-P-Core-AVX2",
        num_units=effective_cores,
        ops_per_unit_per_clock={
            Precision.FP32: ops_per_core_per_cycle_fp32,
            Precision.FP16: ops_per_core_per_cycle_fp32,  # Emulated
            Precision.INT8: ops_per_core_per_cycle_int8,  # VNNI
        },
        clock_domain=clock,
    )

    # Peak FP32: 10 cores × 16 ops/cycle × 4.5 GHz = 720 GFLOPS sustained

    # ========================================================================
    # THERMAL PROFILE - LARGE MODEL TUNING
    # ========================================================================
    # KEY CHANGE: efficiency_factor values tuned for large, compute-bound models
    thermal_profile = ThermalOperatingPoint(
        name="consumer-continuous-large",
        tdp_watts=125.0,
        cooling_solution="tower-cooler",
        performance_specs={
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=avx2_compute,
                instruction_efficiency=0.80,  # ↑ from 0.65 (better loop optimization)
                memory_bottleneck_factor=0.65,  # ↑↑ from 0.25 (compute-bound!)
                efficiency_factor=0.60,  # ↑↑↑ from 0.20 (better amortization)
                tile_utilization=0.80,  # ↑ from 0.50 (better core saturation)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=avx2_compute,
                instruction_efficiency=0.70,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.30,  # ↑ from 0.10 (emulated but better amortized)
                tile_utilization=0.75,
                native_acceleration=False,
            ),
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=avx2_compute,
                instruction_efficiency=0.85,  # VNNI is efficient
                memory_bottleneck_factor=0.70,  # INT8 → less memory pressure
                efficiency_factor=0.65,  # ↑ from 0.18 (VNNI + large models)
                tile_utilization=0.85,  # Better utilization with VNNI
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL (same physical hardware)
    # ========================================================================
    model = HardwareResourceModel(
        name="Intel-i7-12700K-Alder-Lake-Large",
        hardware_type=HardwareType.CPU,
        compute_units=effective_cores,
        threads_per_unit=2,
        warps_per_unit=1,
        warp_size=8,  # AVX2

        # Precision profiles
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=720e9,  # 720 GFLOPS sustained
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=1.44e12,  # 1.44 TOPS sustained
                tensor_core_supported=True,  # VNNI
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        # Memory hierarchy (same hardware)
        peak_bandwidth=75e9,  # 75 GB/s DDR5
        l1_cache_per_unit=32 * 1024,  # 32 KB L1 data
        l2_cache_total=25 * 1024 * 1024,  # 25 MB L3 (LLC)
        main_memory=64 * 1024**3,

        # Energy
        energy_per_flop_fp32=1.5e-12,
        energy_per_byte=25e-12,

        # Scheduling
        min_occupancy=0.6,  # ↑ from 0.4 (large models keep cores busy)
        max_concurrent_kernels=20,
        wave_quantization=1,

        # Thermal profiles
        thermal_operating_points={
            "consumer-continuous-large": thermal_profile,
        },
        default_thermal_profile="consumer-continuous-large",
    )

    return CPUMapper(model)


def create_ampere_ampereone_192_mapper() -> CPUMapper:
    """
    Create CPU mapper for Ampere AmpereOne 192-core ARM Server Processor.

    ARCHITECTURE:
    - 192 ARM v8.6+ cores @ up to 3.6 GHz
    - 2×128-bit SIMD units per core (NEON + SVE)
    - Cloud-native server processor (5nm TSMC)
    - Single thread per core (no SMT)

    PERFORMANCE:
    - Peak FP32: 5.5 TFLOPS
    - Peak FP16/BF16: 11.1 TFLOPS
    - Peak INT8: 22.1 TOPS
    - Memory: 332.8 GB/s (8-channel DDR5-5200)

    POWER:
    - TDP: 283W (A192-32X flagship SKU)

    USE CASES:
    - Cloud-native workloads (microservices, containers)
    - AI inference at scale (cloud servers)
    - High-performance computing (HPC)
    - Hyperscale datacenter deployments

    KEY ADVANTAGES vs x86:
    - 12× more cores than typical server x86 (192 vs 16-32)
    - Better power efficiency for cloud workloads
    - Native ARM architecture (no legacy x86 baggage)
    - Superior AI inference performance (native FP16/BF16/INT8)
    - Ampere AIO (AI Optimizer) for ML frameworks

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Ampere published specs

    REFERENCES:
    - Ampere AmpereOne Family Product Brief (2024)
    - ARM v8.6+ Architecture Reference Manual

    Args:
        None

    Returns:
        CPUMapper configured for Ampere AmpereOne 192-core
    """
    from ..models.datacenter.ampere_ampereone_192 import ampere_ampereone_192_resource_model

    model = ampere_ampereone_192_resource_model()
    return CPUMapper(model)


def create_intel_xeon_platinum_8490h_mapper() -> CPUMapper:
    """
    Create CPU mapper for Intel Xeon Platinum 8490H (Sapphire Rapids).

    ARCHITECTURE:
    - 60 Golden Cove cores @ 2.0-3.5 GHz
    - Intel 7 process (10nm Enhanced SuperFin)
    - HyperThreading (120 threads total)
    - Monolithic die design

    PERFORMANCE:
    - Peak FP32: 2.78 TFLOPS
    - Peak INT8: 88.7 TOPS (AMX)
    - Peak BF16: 44.4 TFLOPS (AMX)
    - Memory: 307 GB/s (8-channel DDR5-4800)

    POWER:
    - TDP: 350W

    AI ACCELERATION:
    - AMX (Advanced Matrix Extensions) for INT8/BF16
    - VNNI (Vector Neural Network Instructions)
    - Deep Learning Boost
    - Much better AI performance than previous Xeon generations

    USE CASES:
    - AI training and inference (AMX acceleration)
    - HPC workloads (AVX-512)
    - Database servers (high thread count)
    - Virtualization hosts (many cores)

    KEY ADVANTAGES vs Consumer CPUs:
    - 60 cores (vs 12-16 for consumer)
    - AMX for AI acceleration (vs basic VNNI)
    - 8-channel DDR5 (vs 2-channel)
    - Enterprise features (RAS, security)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Intel published specs

    REFERENCES:
    - Intel Xeon Scalable Processors (4th Gen) Product Brief

    Args:
        None

    Returns:
        CPUMapper configured for Intel Xeon Platinum 8490H
    """
    from ..models.datacenter.intel_xeon_platinum_8490h import intel_xeon_platinum_8490h_resource_model
    from ..architectural_energy import StoredProgramEnergyModel
    from ..technology_profile import DATACENTER_7NM_DDR5

    model = intel_xeon_platinum_8490h_resource_model()

    # Configure architectural energy model for CPU (STORED_PROGRAM)
    # Energy parameters derived from DATACENTER_7NM_DDR5 profile
    # (Intel 4 process is approximately 7nm equivalent)
    model.architecture_energy_model = StoredProgramEnergyModel(
        tech_profile=DATACENTER_7NM_DDR5
    )

    return CPUMapper(model)


def create_amd_epyc_9654_mapper() -> CPUMapper:
    """
    Create CPU mapper for AMD EPYC 9654 (Genoa).

    ARCHITECTURE:
    - 96 Zen 4 cores @ 2.4-3.7 GHz
    - TSMC 5nm process
    - SMT (192 threads total)
    - Chiplet design (12× 8-core CCDs)

    PERFORMANCE:
    - Peak FP32: 1.84 TFLOPS
    - Peak INT8: 7.37 TOPS
    - Peak FP16: 3.69 TFLOPS
    - Memory: 460.8 GB/s (12-channel DDR5-4800)

    POWER:
    - TDP: 360W (tunable up to 400W)

    AI ACCELERATION:
    - AVX-512 support (double-pumped 256-bit)
    - AVX2 for compatibility
    - No dedicated AI accelerator (unlike Intel AMX)

    USE CASES:
    - Cloud computing (highest core density)
    - Virtualization (massive thread count)
    - Database servers (many cores)
    - Scientific computing

    KEY ADVANTAGES vs Intel Xeon:
    - 60% more cores (96 vs 60)
    - 50% more memory bandwidth (460 GB/s vs 307 GB/s)
    - TSMC 5nm (vs Intel 10nm)
    - Chiplet design (better yields)

    DISADVANTAGES vs Intel Xeon:
    - No AMX (weaker AI performance)
    - Double-pumped AVX-512 (slower than native)
    - Lower per-core performance

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on AMD published specs

    REFERENCES:
    - AMD EPYC 9004 Series Processors Product Brief

    Args:
        None

    Returns:
        CPUMapper configured for AMD EPYC 9654
    """
    from ..models.datacenter.amd_epyc_9654 import amd_epyc_9654_resource_model

    model = amd_epyc_9654_resource_model()
    return CPUMapper(model)


def create_amd_epyc_9754_mapper() -> CPUMapper:
    """
    Create CPU mapper for AMD EPYC 9754 (Genoa) - Flagship 128-core.

    ARCHITECTURE:
    - 128 Zen 4 cores @ 2.25-3.1 GHz
    - TSMC 5nm process
    - SMT (256 threads total)
    - Chiplet design (16× 8-core CCDs)
    - 33% more cores than 9654

    PERFORMANCE:
    - Peak FP32: 2.30 TFLOPS
    - Peak INT8: 9.22 TOPS
    - Peak FP16: 4.61 TFLOPS
    - Memory: 460.8 GB/s (12-channel DDR5-4800, same as 9654)

    POWER:
    - TDP: 360W (same as 9654)

    AI ACCELERATION:
    - AVX-512 support (double-pumped 256-bit)
    - AVX2 for compatibility
    - No dedicated AI accelerator (unlike Intel AMX)

    USE CASES:
    - Cloud computing (extreme core density)
    - Virtualization (256 threads!)
    - Database servers (massive concurrent connections)
    - Scientific computing (massively parallel workloads)

    KEY ADVANTAGES vs EPYC 9654:
    - 33% more cores (128 vs 96)
    - Same power envelope (360W)
    - Same memory bandwidth (460.8 GB/s)
    - More L3 cache (512 MB vs 384 MB)

    TRADE-OFFS:
    - Slightly lower clocks (2.25 GHz vs 2.4 GHz base)
    - More cores share same bandwidth

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on AMD published specs

    REFERENCES:
    - AMD EPYC 9004 Series Processors Product Brief

    Args:
        None

    Returns:
        CPUMapper configured for AMD EPYC 9754
    """
    from ..models.datacenter.amd_epyc_9754 import amd_epyc_9754_resource_model

    model = amd_epyc_9754_resource_model()
    return CPUMapper(model)


def create_intel_xeon_platinum_8592plus_mapper() -> CPUMapper:
    """
    Create CPU mapper for Intel Xeon Platinum 8592+ (Sapphire Rapids) - Flagship 64-core.

    ARCHITECTURE:
    - 64 Golden Cove cores @ 1.9-3.9 GHz
    - Intel 7 process (10nm Enhanced SuperFin)
    - HyperThreading (128 threads total)
    - Monolithic die design
    - 7% more cores than 8490H

    PERFORMANCE:
    - Peak FP32: 3.07 TFLOPS
    - Peak INT8: 98.3 TOPS (AMX)
    - Peak BF16: 49.2 TFLOPS (AMX)
    - Memory: 307.2 GB/s (8-channel DDR5-4800)

    POWER:
    - TDP: 350W (same as 8490H)

    AI ACCELERATION:
    - AMX (Advanced Matrix Extensions): INT8, BF16 matrix operations
    - VNNI (Vector Neural Network Instructions): INT8 dot products
    - Deep Learning Boost
    - Highest AMX performance in Sapphire Rapids lineup

    USE CASES:
    - AI training and inference (flagship AI SKU)
    - HPC workloads
    - Database servers
    - Virtualization hosts

    KEY ADVANTAGES vs 8490H:
    - 7% more cores (64 vs 60)
    - Higher all-core boost (3.0 GHz vs 2.9 GHz)
    - Slightly more L2/L3 cache

    KEY ADVANTAGES vs AMD EPYC:
    - AMX provides 4-10× speedup for CNNs
    - Native AVX-512 (not double-pumped)
    - Higher per-core performance

    DISADVANTAGES vs AMD EPYC:
    - Half the cores (64 vs 128)
    - 33% less memory bandwidth (307 GB/s vs 461 GB/s)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Intel published specs

    REFERENCES:
    - Intel Xeon Scalable Processors (4th Gen) Product Brief

    Args:
        None

    Returns:
        CPUMapper configured for Intel Xeon Platinum 8592+
    """
    from ..models.datacenter.intel_xeon_platinum_8592plus import intel_xeon_platinum_8592plus_resource_model

    model = intel_xeon_platinum_8592plus_resource_model()
    return CPUMapper(model)


def create_ampere_ampereone_128_mapper() -> CPUMapper:
    """
    Create CPU mapper for Ampere AmpereOne 128-core (A128-30X) - ARM mid-tier.

    ARCHITECTURE:
    - 128 ARM v8.6+ cores @ 3.6 GHz
    - TSMC 5nm process
    - No SMT (128 threads total, single-threaded per core)
    - Coherent mesh interconnect
    - 67% of 192-core flagship

    PERFORMANCE:
    - Peak FP32: 3.69 TFLOPS
    - Peak INT8: 14.75 TOPS
    - Peak FP16/BF16: 7.37 TFLOPS
    - Memory: 332.8 GB/s (8-channel DDR5-5200)

    POWER:
    - TDP: 210W (A128-30X, lower than 192-core)

    AI ACCELERATION:
    - Native FP16/BF16 support (2×128-bit SIMD)
    - Native INT8/INT16 support
    - Ampere AIO (AI Optimizer) for ML frameworks

    USE CASES:
    - Cloud-native workloads (microservices, containers)
    - AI inference at scale (cloud servers)
    - High-performance computing (HPC)
    - Cost-effective datacenter deployments

    KEY ADVANTAGES vs 192-core:
    - Lower power (210W vs 283W)
    - Better TCO for non-max-core workloads
    - Same per-core performance

    KEY ADVANTAGES vs x86:
    - Lower power per core
    - Better power efficiency
    - Native ARM ecosystem

    DISADVANTAGES vs Intel:
    - No AMX (weaker CNN performance)
    - ARM software ecosystem (smaller vs x86)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Ampere published specs

    REFERENCES:
    - Ampere AmpereOne Family Product Brief (2024)

    Args:
        None

    Returns:
        CPUMapper configured for Ampere AmpereOne 128-core
    """
    from ..models.datacenter.ampere_ampereone_128 import ampere_ampereone_128_resource_model

    model = ampere_ampereone_128_resource_model()
    return CPUMapper(model)


def create_intel_granite_rapids_mapper() -> CPUMapper:
    """
    Create CPU mapper for Intel Xeon Granite Rapids (Next-Gen 2024-2025) - 128-core.

    ARCHITECTURE:
    - 128 Redwood Cove P-cores @ 2.0-3.8 GHz
    - Intel 3 process (Enhanced FinFET)
    - HyperThreading (256 threads total)
    - Tile-based chiplet design (new for Intel)
    - 2× core count vs Sapphire Rapids flagship

    PERFORMANCE:
    - Peak FP32: 6.55 TFLOPS
    - Peak INT8: 209.7 TOPS (Enhanced AMX)
    - Peak BF16: 104.9 TFLOPS (Enhanced AMX)
    - Memory: 358.4 GB/s (8-channel DDR5-5600, up to 537.6 GB/s with 12-channel)

    POWER:
    - TDP: 500W (high core count)

    AI ACCELERATION:
    - Enhanced AMX with INT4, FP8 support
    - Sparsity acceleration (structured sparsity)
    - VNNI improvements for INT8
    - Better AMX efficiency than Sapphire Rapids

    USE CASES:
    - Large-scale AI training and inference
    - HPC workloads
    - Cloud computing at scale
    - Next-generation datacenter deployments

    KEY ADVANTAGES vs Sapphire Rapids:
    - 2× core count (128 vs 64)
    - Enhanced AMX (INT4, FP8, sparsity)
    - 17% higher memory bandwidth (DDR5-5600 vs 4800)
    - Tile-based design (better scaling)

    KEY ADVANTAGES vs AMD Turin:
    - Enhanced AMX (4-10× faster for CNNs)
    - Higher clocks (3.2 GHz all-core vs 2.5 GHz)
    - Sparsity acceleration

    DISADVANTAGES:
    - 50% higher power (500W vs AMD's 500W but fewer cores)
    - Lower memory bandwidth than AMD (358 GB/s vs 576 GB/s)

    CALIBRATION STATUS:
    ⚠ PROJECTED - Based on Intel roadmap and industry estimates
    - Not yet shipping (2024-2025 timeline)

    REFERENCES:
    - Intel Xeon Roadmap 2024
    - Intel 3 Process Technology Brief

    Args:
        None

    Returns:
        CPUMapper configured for Intel Granite Rapids
    """
    from ..models.datacenter.intel_granite_rapids import intel_granite_rapids_resource_model

    model = intel_granite_rapids_resource_model()
    return CPUMapper(model)


def create_amd_epyc_turin_mapper() -> CPUMapper:
    """
    Create CPU mapper for AMD EPYC Turin (Zen 5, Next-Gen 2024-2025) - 192-core.

    ARCHITECTURE:
    - 192 Zen 5 cores @ 2.5-3.8 GHz
    - TSMC 3nm process (N3)
    - SMT (384 threads total)
    - Chiplet design (24× 8-core CCDs + I/O die)
    - 50% more cores than EPYC 9754

    PERFORMANCE:
    - Peak FP32: 3.84 TFLOPS
    - Peak INT8: 15.36 TOPS
    - Peak FP16: 7.68 TFLOPS
    - Memory: 576 GB/s (12-channel DDR5-6000, 25% more than 9000 series)

    POWER:
    - TDP: 500W (higher core count)

    AI ACCELERATION:
    - Native AVX-512 support (improved from double-pumped)
    - AVX2 for compatibility
    - Possible AI matrix accelerator (rumored, not confirmed)
    - Better INT8 performance than Zen 4

    USE CASES:
    - Cloud computing (extreme core density)
    - Virtualization (384 threads!)
    - Database servers (massive concurrent connections)
    - Large-scale AI inference

    KEY ADVANTAGES vs EPYC 9754 (Zen 4):
    - 50% more cores (192 vs 128)
    - Native AVX-512 (not double-pumped)
    - 25% more memory bandwidth (576 GB/s vs 461 GB/s)
    - 3nm process (improved efficiency)
    - Larger L1 cache (48 KB vs 32 KB per core)

    KEY ADVANTAGES vs Intel Granite Rapids:
    - 50% more cores (192 vs 128)
    - 61% more memory bandwidth (576 GB/s vs 358 GB/s)
    - More threads (384 vs 256)

    DISADVANTAGES vs Intel:
    - No AMX-equivalent (weaker CNN performance)
    - Lower per-core clocks

    CALIBRATION STATUS:
    ⚠ PROJECTED - Based on AMD roadmap and industry estimates
    - Not yet shipping (2024-2025 timeline)

    REFERENCES:
    - AMD EPYC Roadmap 2024
    - AMD Zen 5 Architecture Disclosures

    Args:
        None

    Returns:
        CPUMapper configured for AMD EPYC Turin
    """
    from ..models.datacenter.amd_epyc_turin import amd_epyc_turin_resource_model

    model = amd_epyc_turin_resource_model()
    return CPUMapper(model)


def create_jetson_orin_agx_cpu_mapper(thermal_profile: str = None) -> CPUMapper:
    """
    Create CPU mapper for ARM Cortex-A78AE in Jetson Orin AGX (12-core).

    ARCHITECTURE:
    - 12x ARM Cortex-A78AE cores @ 2.2 GHz sustained
    - ARMv8.2-A with NEON Advanced SIMD
    - NEON dotprod extension for INT8 acceleration
    - Out-of-order execution, branch prediction

    SIMD CAPABILITIES:
    - INT8: 32 ops/cycle/core (NEON dotprod)
    - FP16: 8 ops/cycle/core (NEON FP16)
    - FP32: 8 ops/cycle/core (NEON FMA)

    MEMORY:
    - 64 GB LPDDR5 (shared with GPU)
    - 51.2 GB/s bandwidth (CPU allocation)
    - 64 KB L1, 512 KB L2 per core, 2 MB L3 shared

    POWER: 30W TDP (CPU-only mode for fair comparison)

    PERFORMANCE @ 30W:
    - INT8 peak: 844.8 GOPS (12 x 32 x 2.2 GHz)
    - INT8 effective: ~506 GOPS (60% efficiency)
    - FP32 peak: 211.2 GFLOPS (12 x 8 x 2.2 GHz)
    - FP32 effective: ~105 GFLOPS (50% efficiency)

    USE CASE:
    - Host CPU for robotics and edge AI
    - Lightweight inference (batch=1)
    - Control loops and real-time tasks
    - Architecture comparison studies (Stored-Program vs Data-Parallel vs Domain-Flow)

    KEY CHARACTERISTICS:
    - Stored Program Architecture (instruction fetch overhead)
    - MIMD execution (12-way parallelism)
    - Less efficient than spatial architectures (GPU/KPU) for AI workloads
    - But: lowest idle power, best for irregular control tasks

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on ARM Cortex-A78 specs and Jetson docs

    REFERENCES:
    - ARM Cortex-A78 Technical Reference Manual (ARM DDI 0593)
    - NVIDIA Jetson Orin Series Datasheet (DA-10676-001)
    - ARM NEON Programmer's Guide

    Args:
        thermal_profile: Thermal profile name (e.g., "30W")
                        If None, uses default ("30W")

    Returns:
        CPUMapper configured for Jetson Orin AGX CPU
    """
    from ..models.edge.jetson_orin_agx_cpu import jetson_orin_agx_cpu_resource_model
    from ..architectural_energy import StoredProgramEnergyModel
    from ..technology_profile import EDGE_8NM_LPDDR5

    model = jetson_orin_agx_cpu_resource_model()

    # Configure enhanced Stored Program Architecture energy model for ARM CPU
    # Energy parameters derived from EDGE_8NM_LPDDR5 profile (Jetson Orin class)
    model.architecture_energy_model = StoredProgramEnergyModel(
        tech_profile=EDGE_8NM_LPDDR5
    )

    return CPUMapper(model, thermal_profile=thermal_profile)
