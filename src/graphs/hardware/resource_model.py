"""
Hardware Mapping - Phase 2 of Realistic Performance Modeling

This module maps fused subgraphs from Phase 1 to actual hardware resources,
calculating realistic utilization and latency estimates.

The problem we're solving:
- Phase 1 gives us fused subgraphs with parallelism info (thread counts)
- But assuming 100% hardware utilization leads to 1000× errors
- Phase 2 maps subgraphs to actual HW resources (SMs, tiles, cores)
  to get realistic utilization percentages

Example:
  H100 has 132 SMs, but ResNet-18 at batch=1 has only 12 parallel ops
  → Only ~18% utilization (24/132 SMs), not 100%!
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from enum import Enum

from graphs.ir.structures import (
    SubgraphDescriptor,
    ParallelismDescriptor,
    BottleneckType
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport

if TYPE_CHECKING:
    from graphs.hardware.architectural_energy import ArchitecturalEnergyModel


class HardwareType(Enum):
    """Supported hardware types"""
    GPU = "gpu"
    CPU = "cpu"
    TPU = "tpu"
    KPU = "kpu"
    DPU = "dpu"  # Xilinx Vitis AI (FPGA-based accelerator)
    CGRA = "cgra"  # Stanford Plasticine (spatial dataflow accelerator)
    DSP = "dsp"  # Digital Signal Processors (Qualcomm Hexagon, TI C7x, etc.)


class Precision(Enum):
    """Numerical precision types"""
    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"  # 4-bit exponent, 3-bit mantissa
    FP8_E5M2 = "fp8_e5m2"  # 5-bit exponent, 2-bit mantissa
    FP4 = "fp4"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    INT4 = "int4"


# ============================================================================
# NEW: DVFS-Aware Performance Modeling with Heterogeneous Compute Resources
# ============================================================================

@dataclass
class ClockDomain:
    """
    Clock frequency specifications with DVFS (Dynamic Voltage and Frequency Scaling).

    Modern SoCs don't run at constant frequency - they dynamically adjust voltage
    and frequency based on thermal constraints:
    - base_clock_hz: Guaranteed minimum (always sustainable)
    - max_boost_clock_hz: Maximum burst (datasheet spec, short duration)
    - sustained_clock_hz: Actual clock under sustained load (empirical)
    - thermal_throttle_factor: sustained/boost (how much DVFS reduces performance)

    Example:
        Jetson Orin @ 15W: 306 MHz base, 1.02 GHz boost, 400 MHz sustained
        → thermal_throttle_factor = 0.39 (severe throttling!)
    """
    base_clock_hz: float          # Minimum guaranteed frequency
    max_boost_clock_hz: float     # Maximum burst frequency (datasheet)
    sustained_clock_hz: float     # Actual frequency under thermal load (empirical)
    dvfs_enabled: bool = True     # Dynamic voltage/frequency scaling support

    @property
    def thermal_throttle_factor(self) -> float:
        """How much DVFS reduces clocks: sustained/boost"""
        if self.max_boost_clock_hz > 0:
            return self.sustained_clock_hz / self.max_boost_clock_hz
        return 1.0


@dataclass
class ComputeResource:
    """
    Physical compute units and their capabilities (for homogeneous architectures).

    Calculates peak/sustained performance from first principles:
        peak_ops = num_units × ops_per_unit_per_clock × max_boost_clock_hz

    Example:
        16 ALUs × 4 INT8 ops/ALU/clock × 1.5 GHz = 96 GOPS INT8
    """
    resource_type: str            # "Ampere-SM", "Systolic-Array", "AVX512-Core"
    num_units: int                # Count of compute units (SMs, cores, tiles)
    ops_per_unit_per_clock: Dict[Precision, int]  # SIMD width per precision
    clock_domain: ClockDomain     # Frequency specifications

    def calc_peak_ops(self, precision: Precision) -> float:
        """Calculate peak from first principles (datasheet number)"""
        ops_per_clock = self.ops_per_unit_per_clock.get(precision, 0)
        return (self.num_units *
                ops_per_clock *
                self.clock_domain.max_boost_clock_hz)

    def calc_sustained_ops(self, precision: Precision) -> float:
        """Sustained performance under thermal load (DVFS throttled)"""
        ops_per_clock = self.ops_per_unit_per_clock.get(precision, 0)
        return (self.num_units *
                ops_per_clock *
                self.clock_domain.sustained_clock_hz)


@dataclass
class TileSpecialization:
    """
    KPU-specific: A pool of compute tiles optimized for specific arithmetic types.

    KPU tiles contain array processors that execute SURE (Systems of Uniform
    Recurrence Equations) with heterogeneous PEs (Processing Elements).

    Unlike homogeneous architectures (GPU: all SMs identical), KPU allocates
    silicon budget across specialized tile types:
    - 70 tiles: INT8-optimized (vision, detection)
    - 20 tiles: BF16-optimized (normalization, attention)
    - 10 tiles: Matrix units (large matmuls)

    All precisions are native (no emulation) - just on different tile types.
    """
    tile_type: str                # "INT8-primary", "BF16-primary", "Matrix-8x8"
    num_tiles: int                # Count of tiles with this specialization

    # Performance characteristics per precision
    # (all precisions are native, but some are more optimized)
    ops_per_tile_per_clock: Dict[Precision, int]

    # Silicon optimization level for each precision (0.0-1.0)
    # 1.0 = fully optimized PEs, 0.25 = supported but not optimal
    optimization_level: Dict[Precision, float]

    clock_domain: ClockDomain

    # Array processor characteristics
    array_dimensions: Tuple[int, int] = (16, 8)  # e.g., 16×8 systolic array
    pe_configuration: str = "Mixed"              # "INT8-MAC", "BF16-FMA", "Mixed"


@dataclass
class KPUComputeResource:
    """
    KPU-specific compute model with heterogeneous tile allocation.

    Goal: Characterize workload → recommend tile allocation → build optimal KPU.

    Example silicon allocation for embodied AI:
        - 70% INT8 tiles (Conv, detection)
        - 20% BF16 tiles (normalization, attention)
        - 10% Matrix tiles (large matmuls)
    """
    total_tiles: int
    tile_specializations: List[TileSpecialization]

    def get_tiles_for_precision(self, precision: Precision) -> List[TileSpecialization]:
        """Find which tile types support this precision natively"""
        return [ts for ts in self.tile_specializations
                if precision in ts.ops_per_tile_per_clock]

    def calc_peak_ops(self, precision: Precision) -> float:
        """
        Calculate peak performance across all tiles supporting this precision.

        Example: INT8 performance =
            70 INT8-tiles × 128 ops/tile/clock × 1 GHz +
            20 BF16-tiles × 64 ops/tile/clock × 1 GHz +
            10 Matrix-tiles × 512 ops/tile/clock × 1 GHz
        """
        total_ops = 0.0
        for ts in self.get_tiles_for_precision(precision):
            ops_per_clock = ts.ops_per_tile_per_clock[precision]
            clock_hz = ts.clock_domain.max_boost_clock_hz
            opt_level = ts.optimization_level.get(precision, 1.0)
            total_ops += ts.num_tiles * ops_per_clock * clock_hz * opt_level
        return total_ops

    def calc_sustained_ops(self, precision: Precision) -> float:
        """Sustained performance under thermal load"""
        total_ops = 0.0
        for ts in self.get_tiles_for_precision(precision):
            ops_per_clock = ts.ops_per_tile_per_clock[precision]
            sustained_clock = ts.clock_domain.sustained_clock_hz
            opt_level = ts.optimization_level.get(precision, 1.0)
            total_ops += ts.num_tiles * ops_per_clock * sustained_clock * opt_level
        return total_ops

    def get_silicon_allocation(self) -> Dict[str, float]:
        """Show silicon budget allocation across tile types"""
        return {ts.tile_type: ts.num_tiles / self.total_tiles
                for ts in self.tile_specializations}


@dataclass
class PerformanceCharacteristics:
    """
    Performance data for a specific precision at a thermal operating point.

    Key metrics:
    - peak_ops_per_sec: Datasheet theoretical maximum (boost clock)
    - sustained_ops_per_sec: DVFS-throttled performance
    - effective_ops_per_sec: Actual achieved (with all efficiency factors applied)

    Efficiency factors (all are MULTIPLIERS in range 0.0-1.0):
    - instruction_efficiency: Compiler/ISA efficiency (0.0-1.0)
    - memory_bottleneck_factor: Memory system limits (0.0-1.0)
    - efficiency_factor: Combined measured performance (actual/sustained)

      IMPORTANT: efficiency_factor represents what FRACTION of sustained
      performance you actually achieve:
        - 1.0 = 100% efficiency (theoretical maximum)
        - 0.70 = 70% efficiency (common for datacenter GPUs)
        - 0.12 = 12% efficiency (tiny models on consumer CPUs)

      This is NOT a reduction factor! Higher values = better performance.
    """
    precision: Precision
    compute_resource: Optional[Union[ComputeResource, KPUComputeResource]] = None

    # Microarchitectural efficiency factors (all are multipliers 0.0-1.0)
    instruction_efficiency: float = 0.85     # Compiler/ISA efficiency
    memory_bottleneck_factor: float = 0.75   # Memory system limits
    tile_utilization: float = 1.0            # For KPU: fraction of tiles used

    # Hardware support
    native_acceleration: bool = True         # True = HW accelerated, False = emulated
    emulation_penalty: float = 0.01          # 100× slowdown if not native

    # Combined efficiency factor (measured on real hardware)
    # efficiency_factor = empirical_performance / sustained_performance
    # Example: 0.60 means you achieve 60% of sustained throughput
    efficiency_factor: float = 0.60

    # Optional: Direct measurement overrides calculation
    measured_ops_per_sec: Optional[float] = None

    @property
    def peak_ops_per_sec(self) -> float:
        """Datasheet theoretical maximum (boost clock)"""
        if self.compute_resource:
            return self.compute_resource.calc_peak_ops(self.precision)
        return 0.0

    @property
    def sustained_ops_per_sec(self) -> float:
        """Sustained performance (DVFS throttled)"""
        if self.compute_resource:
            return self.compute_resource.calc_sustained_ops(self.precision)
        return 0.0

    @property
    def effective_ops_per_sec(self) -> float:
        """
        Actual achieved performance (with all efficiency factors applied).

        Calculation:
        1. Start with sustained_ops_per_sec (DVFS-throttled)
        2. Apply emulation penalty if not native
        3. Apply efficiency_factor (measured performance achieved)
        4. For KPU: Apply tile utilization
        """
        # If we have measured data, use it directly
        if self.measured_ops_per_sec:
            return self.measured_ops_per_sec

        # Otherwise calculate from sustained with derates
        base_perf = self.sustained_ops_per_sec

        # Apply emulation penalty if not native
        if not self.native_acceleration:
            base_perf *= self.emulation_penalty

        # Apply tile utilization (for KPU)
        base_perf *= self.tile_utilization

        # Apply efficiency factor (measured performance achieved)
        return base_perf * self.efficiency_factor


@dataclass
class ThermalOperatingPoint:
    """
    Hardware configuration at a specific power/thermal envelope.

    Modern edge devices support multiple power modes:
    - 15W passive: Severe DVFS throttling (sustained = 39% of boost)
    - 30W active: Moderate throttling (sustained = 60% of boost)
    - 60W active: Light throttling (sustained = 77% of boost)

    Each thermal point has different clock behavior and per-precision performance.
    """
    name: str                     # "15W-passive", "60W-active"
    tdp_watts: float              # Thermal Design Power
    cooling_solution: str         # "passive-heatsink", "active-fan", "liquid"

    # Per-precision performance characteristics at this thermal point
    performance_specs: Dict[Precision, PerformanceCharacteristics] = field(default_factory=dict)

    def get_effective_ops(self, precision: Precision) -> float:
        """Get actual achieved performance for a precision"""
        perf_spec = self.performance_specs.get(precision)
        if not perf_spec:
            return 0.0
        return perf_spec.effective_ops_per_sec


# ============================================================================
# OLD: Legacy PrecisionProfile (kept for backward compatibility)
# ============================================================================

@dataclass
class PrecisionProfile:
    """
    Performance characteristics for a specific numerical precision.

    Modern accelerators have vastly different peak performance at different
    precisions. For example, H100 has:
    - FP64: 60 TFLOPS
    - FP32: 60 TFLOPS (without Tensor Cores)
    - BF16: 750 TFLOPS (with Tensor Cores, 12.5x faster!)
    - FP8: 1.5 PFLOPS (with Tensor Cores, 25x faster!)
    """
    precision: Precision
    peak_ops_per_sec: float  # Operations per second at this precision
    tensor_core_supported: bool = False  # Uses specialized matrix units?
    relative_speedup: float = 1.0  # Relative to FP32

    # Memory characteristics
    bytes_per_element: int = 4  # For weights and activations

    # Accumulation precision (can be higher than operand precision)
    accumulator_precision: Optional[Precision] = None


@dataclass
class HardwareResourceModel:
    """
    Hardware resource specification with precision-aware performance.

    This defines the physical resources available on a hardware accelerator,
    including precision-specific peak performance.
    """
    # Required fields (no defaults)
    name: str
    hardware_type: HardwareType
    compute_units: int  # SMs (GPU), cores (CPU), tiles (KPU), arrays (TPU)
    threads_per_unit: int  # Max threads per SM/core/tile
    warps_per_unit: int  # GPU: warps per SM, KPU: vectors per tile
    peak_bandwidth: float  # Memory bandwidth (bytes/sec)
    l1_cache_per_unit: int  # bytes
    l2_cache_total: int  # bytes
    main_memory: int  # bytes (HBM for GPU, DDR for CPU)
    energy_per_flop_fp32: float  # Joules per FLOP at FP32 (baseline)
    energy_per_byte: float  # Joules per byte transferred

    # Optional fields (with defaults)

    # NEW: Architectural energy modeling
    # Captures architecture-specific energy events (instruction fetch, coherence, etc.)
    architecture_energy_model: Optional['ArchitecturalEnergyModel'] = None
    warp_size: int = 32  # Threads per warp (32 for NVIDIA, varies for others)

    # Precision-specific performance
    # Key: Precision, Value: PrecisionProfile
    precision_profiles: Dict[Precision, PrecisionProfile] = field(default_factory=dict)

    # Default precision for mixed-precision models
    default_precision: Precision = Precision.FP32

    # Energy scaling factors by precision (relative to FP32)
    energy_scaling: Dict[Precision, float] = field(default_factory=lambda: {
        Precision.FP64: 2.0,    # 2× energy of FP32
        Precision.FP32: 1.0,    # Baseline
        Precision.FP16: 0.5,    # Half energy
        Precision.BF16: 0.5,
        Precision.FP8_E4M3: 0.25,
        Precision.FP8_E5M2: 0.25,
        Precision.FP4: 0.125,
        Precision.INT32: 0.5,
        Precision.INT16: 0.25,
        Precision.INT8: 0.125,
        Precision.INT4: 0.0625,
    })

    # Scheduling characteristics
    min_occupancy: float = 0.25  # Minimum occupancy for efficiency
    max_concurrent_kernels: int = 1  # Can run multiple kernels?
    wave_quantization: int = 4  # Units allocated in groups (e.g., 4 SMs/wave)

    # NEW: Multi-power-profile support with DVFS modeling
    thermal_operating_points: Optional[Dict[str, ThermalOperatingPoint]] = None
    default_thermal_profile: Optional[str] = None

    # GPU Microarchitecture (for accurate compute modeling)
    # These parameters define the actual hardware implementation
    cuda_cores_per_sm: Optional[int] = None           # 64 (Pascal-Turing), 128 (Ampere-Hopper)
    ops_per_clock_per_core: Optional[float] = 2.0     # FMA: 2 ops/clock for FP32
    sm_boost_clock_hz: Optional[float] = None         # Maximum boost frequency (short bursts)
    sm_sustained_clock_hz: Optional[float] = None     # Sustained frequency under thermal load

    # Tensor Core microarchitecture (for matrix operations)
    tensor_cores_per_sm: Optional[int] = None         # 4 (Volta/Turing/Ampere/Hopper)
    tensor_core_ops_per_clock: Optional[float] = None # Varies by precision and generation

    def get_peak_ops(self, precision: Precision) -> float:
        """
        Get peak operations per second for a given precision.

        Returns:
            Peak ops/sec, or falls back to FP32 if precision not available
        """
        if precision in self.precision_profiles:
            return self.precision_profiles[precision].peak_ops_per_sec

        # Fallback to default precision
        if self.default_precision in self.precision_profiles:
            return self.precision_profiles[self.default_precision].peak_ops_per_sec

        # Should never reach here if properly configured
        raise ValueError(f"No precision profile available for {precision} or {self.default_precision}")

    def get_precision_profile(self, precision: Precision) -> PrecisionProfile:
        """Get precision profile, with fallback to default"""
        if precision in self.precision_profiles:
            return self.precision_profiles[precision]
        return self.precision_profiles[self.default_precision]

    def effective_compute_units(self, occupancy: float) -> float:
        """
        Calculate effective compute units given occupancy.

        Occupancy < min_occupancy may lead to underutilization.
        """
        if occupancy < self.min_occupancy:
            # Penalty for very low occupancy
            penalty = occupancy / self.min_occupancy
            return self.compute_units * occupancy * penalty
        return self.compute_units * occupancy


@dataclass
class HardwareAllocation:
    """
    Result of mapping a single subgraph to hardware resources.

    This describes how one fused subgraph actually executes on hardware.
    """
    subgraph_id: str
    subgraph_name: str
    precision: Precision  # Numerical precision for this operation

    # Resource allocation
    threads_required: int  # From parallelism analysis
    warps_required: int  # threads_required / warp_size
    compute_units_allocated: int  # SMs, cores, tiles actually used
    compute_units_ideal: int  # Ideal without quantization
    occupancy: float  # 0.0 to 1.0

    # Utilization analysis
    utilization: float  # 0.0 to 1.0 (actual / peak resources)
    bottleneck: BottleneckType

    # Timing
    compute_time: float  # seconds (ops / effective_ops_per_sec)
    memory_time: float  # seconds (bytes / bandwidth)
    estimated_latency: float  # seconds (max of compute_time, memory_time)

    # Energy
    compute_energy: float  # Joules
    memory_energy: float  # Joules
    total_energy: float  # Joules

    # Context
    execution_stage: int  # Which execution stage (from concurrency analysis)
    is_parallel: bool  # Can run in parallel with others in same stage


@dataclass
class GraphHardwareAllocation:
    """
    Complete hardware allocation for an entire computation graph.

    This is the final output of Phase 2: realistic utilization estimates.
    """
    model_name: str
    hardware_name: str
    batch_size: int
    model_precision: Precision  # Overall model precision

    # Per-subgraph allocations
    subgraph_allocations: List[HardwareAllocation]

    # Aggregate metrics
    total_subgraphs: int
    total_execution_stages: int

    # Utilization analysis
    peak_compute_units_used: int  # Max SMs used at any time
    average_compute_units_used: float  # Average across execution
    peak_utilization: float  # peak_used / total_available
    average_utilization: float  # average_used / total_available

    # Timing
    total_latency: float  # seconds (sum across stages, max within stage)
    latency_breakdown: Dict[int, float]  # stage_id → latency

    # Energy
    total_energy: float  # Joules

    # Comparison to naive
    naive_latency: float  # Assuming 100% utilization
    latency_correction_factor: float  # naive / actual (should be ~5-20×)

    # Bottleneck analysis
    compute_bound_count: int
    memory_bound_count: int
    bandwidth_bound_count: int
    balanced_count: int

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Hardware Allocation: {self.model_name} on {self.hardware_name}",
            f"Batch Size: {self.batch_size}",
            f"Precision: {self.model_precision.value}",
            f"",
            f"Subgraphs: {self.total_subgraphs}",
            f"Execution Stages: {self.total_execution_stages}",
            f"",
            f"Utilization:",
            f"  Peak: {self.peak_utilization:.1%} ({self.peak_compute_units_used} units)",
            f"  Average: {self.average_utilization:.1%} ({self.average_compute_units_used:.1f} units)",
            f"",
            f"Latency:",
            f"  Total: {self.total_latency*1000:.3f} ms",
            f"  Naive (100% util): {self.naive_latency*1000:.3f} ms",
            f"  Correction factor: {self.latency_correction_factor:.1f}× slower than naive",
            f"",
            f"Energy: {self.total_energy:.3f} Joules",
            f"",
            f"Bottlenecks:",
            f"  Compute-bound: {self.compute_bound_count} ({self.compute_bound_count/self.total_subgraphs:.1%})",
            f"  Memory-bound: {self.memory_bound_count} ({self.memory_bound_count/self.total_subgraphs:.1%})",
            f"  Bandwidth-bound: {self.bandwidth_bound_count} ({self.bandwidth_bound_count/self.total_subgraphs:.1%})",
            f"  Balanced: {self.balanced_count} ({self.balanced_count/self.total_subgraphs:.1%})",
        ]
        return "\n".join(lines)


class HardwareMapper(ABC):
    """
    Base class for hardware-specific mappers.

    Each hardware type (GPU, CPU, TPU, KPU) implements this interface.
    """

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        thermal_profile: Optional[str] = None
    ):
        """
        Initialize hardware mapper.

        Args:
            resource_model: Hardware resource model
            thermal_profile: Thermal profile name (e.g., "15W", "30W").
                           If None, uses default_thermal_profile from resource_model.
        """
        self.resource_model = resource_model

        # Select thermal profile
        if resource_model.thermal_operating_points:
            if thermal_profile is None:
                thermal_profile = resource_model.default_thermal_profile

            # Only validate if a thermal_profile was specified or found
            if thermal_profile is not None:
                if thermal_profile not in resource_model.thermal_operating_points:
                    available = list(resource_model.thermal_operating_points.keys())
                    raise ValueError(
                        f"Thermal profile '{thermal_profile}' not found. "
                        f"Available: {available}"
                    )
            self.thermal_profile = thermal_profile
        else:
            self.thermal_profile = None

    @abstractmethod
    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.FP32
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to hardware resources.

        Args:
            subgraph: Fused subgraph from Phase 1
            execution_stage: Which execution stage (from concurrency analysis)
            concurrent_subgraphs: How many subgraphs run in parallel in this stage
            precision: Numerical precision for this operation

        Returns:
            HardwareAllocation with resource allocation and timing
        """
        pass

    @abstractmethod
    def map_graph(
        self,
        fusion_report: FusionReport,
        execution_stages: List[List[int]],
        batch_size: int = 1,
        precision: Precision = Precision.FP32
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to hardware.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: List of execution stages, each containing subgraph indices
                             e.g., [[0, 1, 2], [3], [4, 5]] means:
                                   - Stage 0: subgraphs 0,1,2 can run in parallel
                                   - Stage 1: subgraph 3 (sequential)
                                   - Stage 2: subgraphs 4,5 can run in parallel
            batch_size: Batch size for scaling parallelism
            precision: Default numerical precision for the model

        Returns:
            Complete hardware allocation with utilization and latency
        """
        pass

    def _calculate_latency(
        self,
        ops: int,  # Operations (not FLOPs, could be INT8 ops, etc.)
        bytes_transferred: int,
        allocated_units: int,
        occupancy: float,
        precision: Precision
    ) -> Tuple[float, float, BottleneckType]:
        """
        Calculate latency for an operation using roofline model.

        Uses thermal operating points (with DVFS and empirical derates) when available,
        otherwise falls back to legacy peak ops calculation.

        Args:
            ops: Number of operations (precision-agnostic count)
            bytes_transferred: Bytes read/written to main memory
            allocated_units: Compute units allocated
            occupancy: Occupancy fraction
            precision: Numerical precision

        Returns:
            (compute_time, memory_time, bottleneck)
        """
        # Get effective ops/sec for this precision
        if self.thermal_profile and self.resource_model.thermal_operating_points:
            # NEW: Use thermal operating point with DVFS and empirical derates
            thermal_point = self.resource_model.thermal_operating_points[self.thermal_profile]

            if precision in thermal_point.performance_specs:
                perf_spec = thermal_point.performance_specs[precision]
                # Use effective ops/sec (includes DVFS throttling + empirical derate)
                base_ops_per_sec = perf_spec.effective_ops_per_sec
            else:
                # Precision not supported at this thermal point
                # Fall back to peak with massive penalty
                base_ops_per_sec = self.resource_model.get_peak_ops(precision) * 0.01
        else:
            # LEGACY: Use old peak ops approach
            base_ops_per_sec = self.resource_model.get_peak_ops(precision)

        # Apply hardware utilization
        effective_ops_per_sec = (
            base_ops_per_sec *
            (allocated_units / self.resource_model.compute_units) *
            occupancy
        )
        compute_time = ops / effective_ops_per_sec if effective_ops_per_sec > 0 else 0

        # Memory time (precision-independent for bandwidth)
        memory_time = bytes_transferred / self.resource_model.peak_bandwidth

        # Determine bottleneck
        if compute_time > memory_time * 1.5:
            bottleneck = BottleneckType.COMPUTE_BOUND
        elif memory_time > compute_time * 1.5:
            bottleneck = BottleneckType.BANDWIDTH_BOUND
        else:
            bottleneck = BottleneckType.BALANCED

        return compute_time, memory_time, bottleneck

    def _calculate_energy(
        self,
        ops: int,
        bytes_transferred: int,
        precision: Precision
    ) -> Tuple[float, float]:
        """
        Calculate energy consumption.

        Args:
            ops: Number of operations
            bytes_transferred: Bytes read/written
            precision: Numerical precision

        Returns:
            (compute_energy, memory_energy) in Joules
        """
        # Get energy scaling factor for this precision
        energy_scale = self.resource_model.energy_scaling.get(precision, 1.0)
        energy_per_op = self.resource_model.energy_per_flop_fp32 * energy_scale

        compute_energy = ops * energy_per_op
        memory_energy = bytes_transferred * self.resource_model.energy_per_byte
        return compute_energy, memory_energy

    def _calculate_energy_with_architecture(
        self,
        ops: int,
        bytes_transferred: int,
        precision: Precision,
        execution_context: Optional[Dict] = None
    ) -> Tuple[float, float, Optional['ArchitecturalEnergyBreakdown']]:
        """
        Calculate energy consumption WITH architectural overhead.

        This method computes baseline energy (compute + memory) and then
        adds architecture-specific energy events if an architectural energy
        model is configured.

        Args:
            ops: Number of operations
            bytes_transferred: Bytes read/written
            precision: Numerical precision
            execution_context: Additional context (threads, batch size, etc.)

        Returns:
            (compute_energy_total, memory_energy_total, architectural_breakdown)
            where architectural_breakdown is None if no model is configured
        """
        # Baseline energy (as before)
        compute_energy, memory_energy = self._calculate_energy(
            ops, bytes_transferred, precision
        )

        # Add architectural energy if model available
        if self.resource_model.architecture_energy_model:
            from graphs.hardware.architectural_energy import ArchitecturalEnergyBreakdown

            if execution_context is None:
                execution_context = {}

            arch_breakdown = self.resource_model.architecture_energy_model.compute_architectural_energy(
                ops=ops,
                bytes_transferred=bytes_transferred,
                compute_energy_baseline=compute_energy,
                memory_energy_baseline=memory_energy,
                execution_context=execution_context
            )

            # Apply architectural overheads
            compute_energy += arch_breakdown.compute_overhead
            memory_energy += arch_breakdown.memory_overhead

            return compute_energy, memory_energy, arch_breakdown
        else:
            # Legacy path - no architectural energy model
            return compute_energy, memory_energy, None


# Pre-defined hardware resource models


# ==============================================================================
# Hardware Resource Models
# ==============================================================================
#
# Individual hardware resource models have been moved to the models/ subdirectory
# for better organization and maintainability.
#
# Import models from:
#   from graphs.hardware.models import <model_name>_resource_model
#
# Example:
#   from graphs.hardware.models import jetson_thor_resource_model
#   model = jetson_thor_resource_model()
#
# Models are organized by category:
#   - models/datacenter/    : High-end GPUs, TPUs, and server CPUs
#   - models/edge/          : Edge AI accelerators and SBCs  
#   - models/automotive/    : Automotive-grade SoCs
#   - models/mobile/        : Mobile GPUs and SoCs
#   - models/accelerators/  : Fixed-function and reconfigurable accelerators
#
# ==============================================================================
