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
from typing import List, Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum

from graphs.ir.structures import (
    SubgraphDescriptor,
    ParallelismDescriptor,
    BottleneckType
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport


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


# Pre-defined hardware resource models

def h100_pcie_resource_model() -> HardwareResourceModel:
    """
    NVIDIA H100 PCIe (80GB) resource model.

    Key characteristics:
    - 132 SMs with 4th gen Tensor Cores
    - Massive speedup for low-precision (BF16: 12.5×, FP8: 25× vs FP32)
    - 2 TB/s HBM2e bandwidth
    """
    return HardwareResourceModel(
        name="H100-PCIe-80GB",
        hardware_type=HardwareType.GPU,
        compute_units=132,  # SMs (Streaming Multiprocessors)
        threads_per_unit=2048,  # Max threads per SM
        warps_per_unit=64,  # Max warps per SM (2048 / 32)
        warp_size=32,

        # Precision profiles (NVIDIA H100 specifications)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=60e12,  # 60 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=60e12,  # 60 TFLOPS (without Tensor Cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=750e12,  # 750 TFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=12.5,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=750e12,  # 750 TFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=12.5,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E4M3: PrecisionProfile(
                precision=Precision.FP8_E4M3,
                peak_ops_per_sec=1500e12,  # 1.5 PFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E5M2: PrecisionProfile(
                precision=Precision.FP8_E5M2,
                peak_ops_per_sec=1500e12,  # 1.5 PFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=1500e12,  # 1.5 POPS
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=2e12,  # 2 TB/s HBM2e
        l1_cache_per_unit=256 * 1024,  # 256 KB per SM
        l2_cache_total=50 * 1024 * 1024,  # 50 MB
        main_memory=80 * 1024**3,  # 80 GB HBM2e
        energy_per_flop_fp32=0.501e-12,  # ~0.5 pJ/FLOP at FP32
        energy_per_byte=15e-12,  # ~15 pJ/byte
        min_occupancy=0.25,
        max_concurrent_kernels=128,  # Can run many kernels concurrently
        wave_quantization=4,  # Launch in waves of 4 SMs

        # Hopper microarchitecture (for compute modeling)
        cuda_cores_per_sm=128,          # Doubled from Turing (64→128)
        ops_per_clock_per_core=2.0,     # FMA: 2 ops/clock
        sm_boost_clock_hz=1980e6,       # 1980 MHz boost
        sm_sustained_clock_hz=1830e6,   # 1830 MHz sustained (92% of boost)

        # Tensor Core details
        tensor_cores_per_sm=4,          # 4th gen Tensor Cores
        tensor_core_ops_per_clock=512,  # 512 FP16 FMAs per clock per TC
    )


def v100_sxm2_resource_model() -> HardwareResourceModel:
    """
    NVIDIA V100 SXM2 (32GB) resource model - Volta generation (2017).

    ARCHITECTURE:
    - First generation with Tensor Cores
    - 80 SMs with 64 CUDA cores each (5,120 CUDA cores total)
    - 8 Tensor Cores per SM (640 total)
    - Volta microarchitecture

    PERFORMANCE:
    - FP32: 15.7 TFLOPS (CUDA cores)
    - FP16: 31.4 TFLOPS (CUDA cores, 2× FP32)
    - FP16 (Tensor Cores): 125 TFLOPS (8× CUDA cores)
    - Boost clock: 1530 MHz

    MEMORY:
    - 32 GB HBM2
    - 900 GB/s bandwidth

    POWER:
    - 300W TDP

    USE CASE:
    - Training pioneer (first with Tensor Cores)
    - DGX-1 V100, Cloud instances (AWS P3, GCP)
    """
    return HardwareResourceModel(
        name="V100-SXM2-32GB",
        hardware_type=HardwareType.GPU,
        compute_units=80,  # SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,

        # Precision profiles
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=7.8e12,  # 7.8 TFLOPS
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=15.7e12,  # 15.7 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=125e12,  # 125 TFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=7.96,  # 125 / 15.7
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=125e12,  # Same as FP16 Tensor Cores
                tensor_core_supported=True,
                relative_speedup=7.96,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=900e9,  # 900 GB/s HBM2
        l1_cache_per_unit=128 * 1024,  # 128 KB per SM
        l2_cache_total=6 * 1024 * 1024,  # 6 MB
        main_memory=32 * 1024**3,  # 32 GB HBM2
        energy_per_flop_fp32=0.64e-12,  # ~0.64 pJ/FLOP (300W / 15.7 TFLOPS / efficiency)
        energy_per_byte=12e-12,  # ~12 pJ/byte
        min_occupancy=0.25,
        max_concurrent_kernels=32,
        wave_quantization=4,

        # Volta microarchitecture
        cuda_cores_per_sm=64,           # First with 64 cores/SM
        ops_per_clock_per_core=2.0,     # FMA
        sm_boost_clock_hz=1530e6,       # 1530 MHz boost
        sm_sustained_clock_hz=1400e6,   # 1400 MHz sustained (~91% of boost)

        # Tensor Core details (1st generation)
        tensor_cores_per_sm=8,          # 8 TCs per SM (first generation)
        tensor_core_ops_per_clock=256,  # 256 FP16 FMAs per clock per TC
    )


def t4_resource_model() -> HardwareResourceModel:
    """
    NVIDIA T4 resource model - Turing generation (2018).

    ARCHITECTURE:
    - Inference-optimized GPU (low power, high INT8 throughput)
    - 40 SMs with 64 CUDA cores each (2,560 CUDA cores total)
    - 8 Tensor Cores per SM (2nd gen, improved INT8)
    - Turing microarchitecture

    PERFORMANCE:
    - FP32: 8.1 TFLOPS (CUDA cores)
    - FP16 (Tensor Cores): 65 TFLOPS
    - INT8 (Tensor Cores): 130 TOPS (2× FP16)
    - INT4: 260 TOPS
    - Boost clock: 1590 MHz

    MEMORY:
    - 16 GB GDDR6
    - 320 GB/s bandwidth

    POWER:
    - 70W TDP (inference-optimized!)

    USE CASE:
    - Inference-optimized (low latency, high throughput)
    - Cloud inference (AWS G4, GCP T4)
    - Edge servers
    """
    return HardwareResourceModel(
        name="T4",
        hardware_type=HardwareType.GPU,
        compute_units=40,  # SMs
        threads_per_unit=1024,  # Reduced from V100
        warps_per_unit=32,
        warp_size=32,

        # Precision profiles (inference-optimized)
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=8.1e12,  # 8.1 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=65e12,  # 65 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=130e12,  # 130 TOPS (2× FP16)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=260e12,  # 260 TOPS (2× INT8)
                tensor_core_supported=True,
                relative_speedup=32.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=320e9,  # 320 GB/s GDDR6
        l1_cache_per_unit=96 * 1024,  # 96 KB per SM
        l2_cache_total=4 * 1024 * 1024,  # 4 MB
        main_memory=16 * 1024**3,  # 16 GB GDDR6
        energy_per_flop_fp32=0.29e-12,  # ~0.29 pJ/FLOP (70W / 8.1 TFLOPS / efficiency)
        energy_per_byte=8e-12,  # ~8 pJ/byte
        min_occupancy=0.25,
        max_concurrent_kernels=32,
        wave_quantization=4,

        # Turing microarchitecture (same core count as Volta)
        cuda_cores_per_sm=64,           # Same as Volta
        ops_per_clock_per_core=2.0,     # FMA
        sm_boost_clock_hz=1590e6,       # 1590 MHz boost
        sm_sustained_clock_hz=1470e6,   # 1470 MHz sustained (~92% of boost)

        # Tensor Core details (2nd generation, improved INT8)
        tensor_cores_per_sm=8,          # 8 TCs per SM
        tensor_core_ops_per_clock=256,  # Similar to Volta
    )


def a100_sxm4_80gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA A100 SXM4 (80GB) resource model - Ampere generation (2020).

    ARCHITECTURE:
    - Ampere microarchitecture (GA100 die)
    - 108 SMs with 128 CUDA cores each (6,912 CUDA cores total)
    - 4 Tensor Cores per SM (3rd gen, added TF32, BF16, FP64 TC support)
    - Doubled CUDA cores per SM (64→128)

    PERFORMANCE:
    - FP32: 19.5 TFLOPS (CUDA cores)
    - TF32 (Tensor Cores): 156 TFLOPS (new in Ampere)
    - BF16 (Tensor Cores): 312 TFLOPS
    - FP16 (Tensor Cores): 312 TFLOPS
    - INT8 (Tensor Cores): 624 TOPS
    - Boost clock: 1410 MHz

    MEMORY:
    - 80 GB HBM2e
    - 2 TB/s bandwidth (same as H100)

    POWER:
    - 400W TDP

    USE CASE:
    - Training standard (DGX A100, cloud)
    - First with TF32 and BF16 support
    - Strong balance of training and inference
    """
    return HardwareResourceModel(
        name="A100-SXM4-80GB",
        hardware_type=HardwareType.GPU,
        compute_units=108,  # SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,

        # Precision profiles
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=9.7e12,  # 9.7 TFLOPS
                tensor_core_supported=True,  # New in Ampere!
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=19.5e12,  # 19.5 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            # TF32: New in Ampere (FP32 range, FP16 precision)
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=312e12,  # 312 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=312e12,  # 312 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=624e12,  # 624 TOPS (2× FP16)
                tensor_core_supported=True,
                relative_speedup=32.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=2e12,  # 2 TB/s HBM2e (same as H100)
        l1_cache_per_unit=192 * 1024,  # 192 KB per SM
        l2_cache_total=40 * 1024 * 1024,  # 40 MB
        main_memory=80 * 1024**3,  # 80 GB HBM2e
        energy_per_flop_fp32=0.69e-12,  # ~0.69 pJ/FLOP (400W / 19.5 TFLOPS / efficiency)
        energy_per_byte=14e-12,  # ~14 pJ/byte
        min_occupancy=0.25,
        max_concurrent_kernels=128,
        wave_quantization=4,

        # Ampere microarchitecture (doubled CUDA cores!)
        cuda_cores_per_sm=128,          # Doubled from Volta/Turing (64→128)
        ops_per_clock_per_core=2.0,     # FMA
        sm_boost_clock_hz=1410e6,       # 1410 MHz boost
        sm_sustained_clock_hz=1300e6,   # 1300 MHz sustained (~92% of boost)

        # Tensor Core details (3rd generation, added TF32/BF16/FP64)
        tensor_cores_per_sm=4,          # Reduced from 8 (but much more capable)
        tensor_core_ops_per_clock=512,  # 512 ops/clock (doubled throughput)
    )


def tpu_v4_resource_model() -> HardwareResourceModel:
    """
    Google TPU v4 resource model.

    Key characteristics:
    - Optimized for BF16 and INT8
    - 2× INT8 performance vs BF16
    - Very energy efficient
    """
    return HardwareResourceModel(
        name="TPU-v4",
        hardware_type=HardwareType.TPU,
        compute_units=2,  # 2 TensorCores
        threads_per_unit=128 * 128,  # 128×128 systolic array
        warps_per_unit=128,  # rows in systolic array
        warp_size=128,  # columns in systolic array

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=137.5e12,  # Half of BF16 (not native)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=275e12,  # 275 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=550e12,  # 550 TOPS (2× BF16)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.BF16,

        peak_bandwidth=1.2e12,  # 1.2 TB/s HBM2e
        l1_cache_per_unit=16 * 1024 * 1024,  # 16 MB per core
        l2_cache_total=32 * 1024 * 1024,  # 32 MB
        main_memory=32 * 1024**3,  # 32 GB HBM2e
        energy_per_flop_fp32=0.4e-12,  # Very efficient (assuming FP32 equiv)
        energy_per_byte=10e-12,
        min_occupancy=0.5,  # Systolic arrays need high utilization
        max_concurrent_kernels=1,  # Typically runs one large batch
        wave_quantization=1,
    )


def kpu_t64_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T64 with 64 heterogeneous tiles for embodied AI / edge devices.

    ============================================================================
    EDGE AI / EMBODIED AI WORKLOAD ALLOCATION (64 tiles)
    ============================================================================

    Target: Battery-powered drones, robots, edge AI devices
    - Same 70/20/10 ratio as T100/T300, scaled to 64 tiles
    - 44 INT8 tiles (69%): Computer vision, object detection
    - 13 BF16 tiles (20%): Sensor fusion, lightweight transformers
    - 7 Matrix tiles (11%): Classification heads, embeddings

    Architecture: 8×8 Checkerboard
    - 64 compute tiles arranged in 8×8 grid
    - 64 L3 memory tiles (256KB each) for distributed memory
    - Low-latency interconnect for tile-to-tile communication
    - Power-optimized for 3-6W operation

    Power Profiles for Edge AI:
    - 3W: Ultra-low power (battery-powered drones)
    - 6W: Standard edge AI (default)
    - 10W: High performance edge (max sustainable)

    Key Advantages vs Jetson Orin Nano:
    ✓ Better TOPS/W efficiency (10.6 vs 2.7 TOPS/W)
    ✓ Higher efficiency_factor (65-70% vs 4-10%)
    ✓ No DVFS throttling (well-designed thermal)
    ✓ Distributed on-chip memory (lower latency)
    """
    # Clock domain for T64 (power-optimized)
    t64_clock = ClockDomain(
        base_clock_hz=800e6,
        max_boost_clock_hz=900e6,
        sustained_clock_hz=850e6,  # 94% of boost at 3W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T64 TILE ALLOCATION (64 tiles total, ~70/20/10 ratio)
    # ========================================================================
    t64_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=44,  # 69% of 64
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t64_clock,
    )

    t64_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=13,  # 20% of 64
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t64_clock,
    )

    t64_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=7,  # 11% of 64
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t64_clock,
    )

    t64_compute = KPUComputeResource(
        total_tiles=64,
        tile_specializations=[t64_int8_tiles, t64_bf16_tiles, t64_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T64 (Edge AI SKUs)
    # ========================================================================

    # 3W Profile: Ultra-low power (battery-powered drones, wearables)
    thermal_3w = ThermalOperatingPoint(
        name="3W-battery",
        tdp_watts=3.0,
        cooling_solution="passive-heatsink-tiny",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t64_compute,
                efficiency_factor=0.60,  # 60%! (vs Jetson Orin Nano's 4%)
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t64_compute,
                efficiency_factor=0.55,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t64_compute,
                efficiency_factor=0.65,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
        }
    )

    # 6W Profile: Standard edge AI (default)
    t64_clock_6w = ClockDomain(
        base_clock_hz=850e6,
        max_boost_clock_hz=950e6,
        sustained_clock_hz=900e6,  # 95% of boost
        dvfs_enabled=True,
    )
    t64_compute_6w = KPUComputeResource(
        total_tiles=64,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=44, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t64_clock_6w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=13, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t64_clock_6w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=7, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t64_clock_6w,
            ),
        ],
    )

    thermal_6w = ThermalOperatingPoint(
        name="6W-standard",
        tdp_watts=6.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t64_compute_6w,
                efficiency_factor=0.65,  # Excellent efficiency
                tile_utilization=0.93,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t64_compute_6w,
                efficiency_factor=0.60,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t64_compute_6w,
                efficiency_factor=0.70,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
        }
    )

    # 10W Profile: High performance edge (active cooling)
    t64_clock_10w = ClockDomain(
        base_clock_hz=900e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=950e6,  # 95% of boost
        dvfs_enabled=True,
    )
    t64_compute_10w = KPUComputeResource(
        total_tiles=64,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=44, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t64_clock_10w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=13, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t64_clock_10w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=7, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t64_clock_10w,
            ),
        ],
    )

    thermal_10w = ThermalOperatingPoint(
        name="10W-performance",
        tdp_watts=10.0,
        cooling_solution="active-fan-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t64_compute_10w,
                efficiency_factor=0.70,  # Peak efficiency with cooling
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t64_compute_10w,
                efficiency_factor=0.65,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t64_compute_10w,
                efficiency_factor=0.75,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Stillwater KPU-T64",
        hardware_type=HardwareType.KPU,
        compute_units=64,
        threads_per_unit=256,
        warps_per_unit=0,  # KPU uses tiles, not warps
        warp_size=0,

        # Thermal operating points
        thermal_operating_points={
            "3W": thermal_3w,   # Battery-powered
            "6W": thermal_6w,   # Standard edge AI
            "10W": thermal_10w,  # High performance
        },
        default_thermal_profile="6W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=6.9e12,  # 6.9 TOPS @ 900 MHz
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=3.3e12,  # 3.3 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=8.8e12,  # 8.8 TOPS
                tensor_core_supported=True,
                relative_speedup=2.5,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=64e9,  # 64 GB/s LPDDR5
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=4 * 1024 * 1024,  # 4 MB shared L2
        main_memory=8 * 1024**3,  # 8 GB LPDDR5
        energy_per_flop_fp32=0.10e-12,
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=2,
        wave_quantization=1,
    )


def kpu_t256_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T256 with 256 heterogeneous tiles for high-performance edge/datacenter AI.

    ============================================================================
    HIGH-PERFORMANCE EDGE AI WORKLOAD ALLOCATION (256 tiles)
    ============================================================================

    Target: High-throughput edge servers, datacenter inference, autonomous vehicles
    - Same 70/20/10 ratio as T100/T300, scaled to 256 tiles
    - 179 INT8 tiles (70%): Massive parallel vision processing
    - 51 BF16 tiles (20%): Multi-modal fusion, transformers
    - 26 Matrix tiles (10%): Large-scale embeddings, LLM inference

    Architecture: 16×16 Checkerboard
    - 256 compute tiles arranged in 16×16 grid
    - 256 L3 memory tiles (256KB each) for distributed memory
    - High-bandwidth interconnect (2D torus)
    - Power profiles: 15W, 30W, 50W

    Power Profiles:
    - 15W: Efficient mode (edge servers)
    - 30W: Balanced mode (datacenter inference)
    - 50W: Performance mode (max throughput)

    Key Advantages vs Jetson Orin AGX:
    ✓ 2.56× more tiles than T100
    ✓ Higher efficiency_factor (70-80% vs 5-12%)
    ✓ Better memory hierarchy (distributed L3)
    ✓ Predictable performance (no DVFS throttling)
    """
    # Clock domain for T256
    t256_clock = ClockDomain(
        base_clock_hz=900e6,
        max_boost_clock_hz=1.05e9,
        sustained_clock_hz=1.0e9,  # 95% of boost at 15W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T256 TILE ALLOCATION (256 tiles total, 70/20/10 ratio)
    # ========================================================================
    t256_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=179,  # 70% of 256
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t256_clock,
    )

    t256_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=51,  # 20% of 256
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t256_clock,
    )

    t256_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=26,  # 10% of 256
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t256_clock,
    )

    t256_compute = KPUComputeResource(
        total_tiles=256,
        tile_specializations=[t256_int8_tiles, t256_bf16_tiles, t256_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T256
    # ========================================================================

    # 15W Profile: Efficient mode
    thermal_15w = ThermalOperatingPoint(
        name="15W-efficient",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink-large",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t256_compute,
                efficiency_factor=0.68,
                tile_utilization=0.93,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t256_compute,
                efficiency_factor=0.63,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t256_compute,
                efficiency_factor=0.73,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
        }
    )

    # 30W Profile: Balanced mode
    t256_clock_30w = ClockDomain(
        base_clock_hz=950e6,
        max_boost_clock_hz=1.1e9,
        sustained_clock_hz=1.05e9,  # 95% of boost
        dvfs_enabled=True,
    )
    t256_compute_30w = KPUComputeResource(
        total_tiles=256,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=179, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t256_clock_30w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=51, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t256_clock_30w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=26, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t256_clock_30w,
            ),
        ],
    )

    thermal_30w = ThermalOperatingPoint(
        name="30W-balanced",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t256_compute_30w,
                efficiency_factor=0.73,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t256_compute_30w,
                efficiency_factor=0.68,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t256_compute_30w,
                efficiency_factor=0.78,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
        }
    )

    # 50W Profile: Performance mode
    t256_clock_50w = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.15e9,
        sustained_clock_hz=1.1e9,  # 96% of boost
        dvfs_enabled=True,
    )
    t256_compute_50w = KPUComputeResource(
        total_tiles=256,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=179, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t256_clock_50w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=51, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t256_clock_50w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=26, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t256_clock_50w,
            ),
        ],
    )

    thermal_50w = ThermalOperatingPoint(
        name="50W-performance",
        tdp_watts=50.0,
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t256_compute_50w,
                efficiency_factor=0.78,  # Peak efficiency
                tile_utilization=0.97,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t256_compute_50w,
                efficiency_factor=0.73,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t256_compute_50w,
                efficiency_factor=0.83,
                tile_utilization=0.98,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Stillwater KPU-T256",
        hardware_type=HardwareType.KPU,
        compute_units=256,
        threads_per_unit=256,
        warps_per_unit=0,
        warp_size=0,

        # Thermal operating points
        thermal_operating_points={
            "15W": thermal_15w,
            "30W": thermal_30w,
            "50W": thermal_50w,
        },
        default_thermal_profile="30W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=33.8e12,  # 33.8 TOPS @ 1.05 GHz
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=16.3e12,  # 16.3 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=43.3e12,  # 43.3 TOPS
                tensor_core_supported=True,
                relative_speedup=2.5,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=256e9,  # 256 GB/s (4×LPDDR5 or DDR5)
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=16 * 1024 * 1024,  # 16 MB shared L2
        main_memory=32 * 1024**3,  # 32 GB
        energy_per_flop_fp32=0.09e-12,
        energy_per_byte=11e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,
    )


def kpu_t768_resource_model() -> HardwareResourceModel:
    """
    Stillwater KPU-T768 with 768 heterogeneous tiles for datacenter AI inference.

    ============================================================================
    DATACENTER AI INFERENCE WORKLOAD ALLOCATION (768 tiles)
    ============================================================================

    Target: Datacenter inference, high-throughput AI serving, LLM inference
    - Same 70/20/10 ratio as T64/T256, scaled to 768 tiles
    - 537 INT8 tiles (70%): Massive parallel vision, object detection
    - 154 BF16 tiles (20%): Transformer layers, multi-modal fusion
    - 77 Matrix tiles (10%): Large matmuls, LLM token generation

    Architecture: 32×24 Grid (768 tiles)
    - 768 compute tiles arranged in optimized grid
    - 768 L3 memory tiles (256KB each) for distributed memory
    - Ultra-high-bandwidth interconnect (2D torus + express channels)
    - Power profiles: 30W, 60W, 100W

    Power Profiles:
    - 30W: Efficient datacenter (max efficiency)
    - 60W: Balanced datacenter (default)
    - 100W: Performance mode (max throughput)

    Key Advantages:
    ✓ 3× more tiles than T256
    ✓ Datacenter-class throughput (100+ TOPS effective)
    ✓ Excellent efficiency_factor (75-85%)
    ✓ Distributed memory architecture
    ✓ Optimized for batch inference
    """
    # Clock domain for T768 (datacenter-optimized)
    t768_clock = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.2e9,
        sustained_clock_hz=1.1e9,  # 92% of boost at 30W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T768 TILE ALLOCATION (768 tiles total, 70/20/10 ratio)
    # ========================================================================
    t768_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=537,  # 70% of 768
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t768_clock,
    )

    t768_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=154,  # 20% of 768
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t768_clock,
    )

    t768_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=77,  # 10% of 768
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t768_clock,
    )

    t768_compute = KPUComputeResource(
        total_tiles=768,
        tile_specializations=[t768_int8_tiles, t768_bf16_tiles, t768_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T768 (Datacenter SKUs)
    # ========================================================================

    # 30W Profile: Efficient datacenter
    thermal_30w = ThermalOperatingPoint(
        name="30W-efficient",
        tdp_watts=30.0,
        cooling_solution="active-datacenter-1U",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t768_compute,
                efficiency_factor=0.75,  # 75% efficiency
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t768_compute,
                efficiency_factor=0.70,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t768_compute,
                efficiency_factor=0.78,
                tile_utilization=0.96,
                native_acceleration=True,
            ),
        }
    )

    # 60W Profile: Balanced datacenter (default)
    t768_clock_60w = ClockDomain(
        base_clock_hz=1.1e9,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.2e9,  # 92% of boost
        dvfs_enabled=True,
    )
    t768_compute_60w = KPUComputeResource(
        total_tiles=768,
        tile_specializations=[
            TileSpecialization("INT8-primary", 537, (16, 8), "INT8-MAC",
                             {Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                             {Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                             t768_clock_60w),
            TileSpecialization("BF16-primary", 154, (16, 8), "BF16-FMA",
                             {Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                             {Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                             t768_clock_60w),
            TileSpecialization("Matrix-8x8", 77, (8, 8), "Mixed-INT8-BF16-Matrix",
                             {Precision.INT8: 512, Precision.BF16: 256},
                             {Precision.INT8: 1.0, Precision.BF16: 1.0},
                             t768_clock_60w),
        ],
    )

    thermal_60w = ThermalOperatingPoint(
        name="60W-balanced",
        tdp_watts=60.0,
        cooling_solution="active-datacenter-2U",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t768_compute_60w,
                efficiency_factor=0.80,  # 80% efficiency at higher power
                tile_utilization=0.96,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t768_compute_60w,
                efficiency_factor=0.75,
                tile_utilization=0.94,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t768_compute_60w,
                efficiency_factor=0.82,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
        }
    )

    # 100W Profile: Performance mode
    t768_clock_100w = ClockDomain(
        base_clock_hz=1.2e9,
        max_boost_clock_hz=1.4e9,
        sustained_clock_hz=1.35e9,  # 96% of boost
        dvfs_enabled=True,
    )
    t768_compute_100w = KPUComputeResource(
        total_tiles=768,
        tile_specializations=[
            TileSpecialization("INT8-primary", 537, (16, 8), "INT8-MAC",
                             {Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                             {Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                             t768_clock_100w),
            TileSpecialization("BF16-primary", 154, (16, 8), "BF16-FMA",
                             {Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                             {Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                             t768_clock_100w),
            TileSpecialization("Matrix-8x8", 77, (8, 8), "Mixed-INT8-BF16-Matrix",
                             {Precision.INT8: 512, Precision.BF16: 256},
                             {Precision.INT8: 1.0, Precision.BF16: 1.0},
                             t768_clock_100w),
        ],
    )

    thermal_100w = ThermalOperatingPoint(
        name="100W-performance",
        tdp_watts=100.0,
        cooling_solution="active-datacenter-2U-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t768_compute_100w,
                efficiency_factor=0.85,  # 85% efficiency at max power
                tile_utilization=0.98,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t768_compute_100w,
                efficiency_factor=0.80,
                tile_utilization=0.96,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t768_compute_100w,
                efficiency_factor=0.87,
                tile_utilization=0.99,
                native_acceleration=True,
            ),
        }
    )

    # Build resource model
    return HardwareResourceModel(
        name="Stillwater KPU-T768",
        hardware_type=HardwareType.KPU,
        compute_units=768,
        threads_per_unit=256,
        warps_per_unit=0,  # KPU uses tiles, not warps
        warp_size=0,

        thermal_operating_points={
            "30W": thermal_30w,
            "60W": thermal_60w,
            "100W": thermal_100w,
        },
        default_thermal_profile="60W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=130.1e12,  # 130.1 TOPS @ 60W
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=48.9e12,  # 48.9 TFLOPS @ 60W
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=260.2e12,  # 260.2 TOPS @ 60W
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=512e9,  # 512 GB/s (8×DDR5 or HBM3)
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=32 * 1024 * 1024,  # 32 MB shared L2
        main_memory=64 * 1024**3,  # 64 GB
        energy_per_flop_fp32=0.08e-12,
        energy_per_byte=10e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=8,
        wave_quantization=2,
    )


def cpu_x86_resource_model() -> HardwareResourceModel:
    """
    Generic high-end x86 CPU resource model (Intel/AMD).

    Key characteristics:
    - Decent FP32/FP64 with AVX-512
    - Limited INT8 acceleration (VNNI on newer CPUs)
    - Much slower than GPUs but flexible
    """
    return HardwareResourceModel(
        name="CPU-x86-16core",
        hardware_type=HardwareType.CPU,
        compute_units=16,  # 16 cores
        threads_per_unit=2,  # SMT/HyperThreading
        warps_per_unit=1,  # No warp concept
        warp_size=1,

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=750e9,  # 750 GFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1.5e12,  # 1.5 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=3e12,  # 3 TFLOPS (with AMX on newer CPUs)
                tensor_core_supported=True,  # AMX
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=6e12,  # 6 TOPS (with VNNI/AMX)
                tensor_core_supported=True,  # VNNI/AMX
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=80e9,  # 80 GB/s DDR5
        l1_cache_per_unit=32 * 1024,  # 32 KB per core
        l2_cache_total=16 * 1024 * 1024,  # 16 MB total L2
        main_memory=64 * 1024**3,  # 64 GB DDR5
        energy_per_flop_fp32=1.004e-12,  # Less efficient
        energy_per_byte=20e-12,
        min_occupancy=0.5,
        max_concurrent_kernels=16,  # One per core
        wave_quantization=1,
    )


def ampere_ampereone_192_resource_model() -> HardwareResourceModel:
    """
    Ampere AmpereOne 192-core ARM Server Processor (A192-32X flagship).

    ARCHITECTURE:
    - 192 Ampere 64-bit ARM v8.6+ cores
    - 2×128-bit SIMD units per core (NEON + SVE)
    - Coherent mesh interconnect with distributed snoop filtering
    - TSMC 5nm process

    PERFORMANCE:
    - Clock: Up to 3.6 GHz (consistent across all cores)
    - Peak FP32: ~5.5 TFLOPS (192 cores × 8 ops/cycle × 3.6 GHz)
    - Peak FP16/BF16: ~11.1 TFLOPS (192 cores × 16 ops/cycle × 3.6 GHz)
    - Peak INT8: ~22.1 TOPS (192 cores × 32 ops/cycle × 3.6 GHz)

    CACHE HIERARCHY:
    - L1 Data: 64 KB per core (12.3 MB total)
    - L1 Instruction: 16 KB per core (3.1 MB total)
    - L2: 2 MB per core (384 MB total)
    - System Cache (L3-like): 64 MB shared

    MEMORY:
    - 8-channel DDR5-5200 (up to 4TB)
    - Peak bandwidth: 332.8 GB/s (8 × 41.6 GB/s)

    POWER:
    - TDP: 283W (A192-32X at max performance)
    - Idle: ~50W (estimate)
    - Dynamic: ~233W

    AI ACCELERATION:
    - Native FP16/BF16 support (2×128-bit SIMD)
    - Native INT8/INT16 support
    - Ampere AIO (AI Optimizer) for ML frameworks
    - Better than x86 for AI inference (wider SIMD for low precision)

    CONNECTIVITY:
    - 128 lanes PCIe 5.0 with 32 controllers

    USE CASES:
    - Cloud-native workloads (microservices, containers)
    - AI inference at scale (cloud servers)
    - High-performance computing (HPC)
    - Hyperscale datacenter deployments

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Ampere published specs
    - SIMD calculations based on ARM v8.6+ NEON/SVE
    - Memory bandwidth from DDR5-5200 8-channel configuration

    REFERENCES:
    - Ampere AmpereOne Family Product Brief (2024)
    - ARM v8.6+ Architecture Reference Manual
    - DDR5-5200 specifications
    """
    # Performance calculations
    num_cores = 192
    clock_hz = 3.6e9  # 3.6 GHz
    simd_units_per_core = 2  # 2×128-bit SIMD units

    # SIMD operations per cycle per core
    # 128-bit SIMD: FP32=4, FP16=8, INT8=16 ops per unit
    fp32_ops_per_core = simd_units_per_core * 4  # 8 FP32 ops/cycle
    fp16_ops_per_core = simd_units_per_core * 8  # 16 FP16/BF16 ops/cycle
    int8_ops_per_core = simd_units_per_core * 16  # 32 INT8 ops/cycle

    # Peak performance
    peak_fp32 = num_cores * fp32_ops_per_core * clock_hz  # 5.53 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * clock_hz  # 11.06 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * clock_hz  # 22.12 TOPS

    # Memory bandwidth (8-channel DDR5-5200)
    channels = 8
    ddr5_rate = 5200e6  # 5200 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 332.8 GB/s

    # Power and energy
    tdp = 283.0  # Watts (A192-32X)
    idle_power = 50.0  # Estimated
    dynamic_power = tdp - idle_power  # 233W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~42 pJ/FLOP
    energy_per_byte = 25e-12  # 25 pJ/byte (server-class, more efficient than desktop)

    return HardwareResourceModel(
        name="Ampere-AmpereOne-192core",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 192 cores
        threads_per_unit=1,  # No SMT/HyperThreading (single thread per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # Effective SIMD width for INT8 (per unit)

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 2.77 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 5.53 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_fp16,  # 11.06 TFLOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 11.06 TFLOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=peak_fp16,  # 11.06 TOPS (same as FP16)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8,  # 22.12 TOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 332.8 GB/s (8-channel DDR5-5200)
        l1_cache_per_unit=64 * 1024,  # 64 KB L1D per core
        l2_cache_total=384 * 1024 * 1024,  # 384 MB total L2 (2 MB × 192 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~42 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 25 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def intel_xeon_platinum_8490h_resource_model() -> HardwareResourceModel:
    """
    Intel Xeon Platinum 8490H (Sapphire Rapids) - Datacenter Server Processor.

    ARCHITECTURE:
    - 60 Golden Cove cores (120 threads with HyperThreading)
    - Intel 7 process (10nm Enhanced SuperFin)
    - Monolithic die design

    PERFORMANCE:
    - Clock: 2.0 GHz base, 3.5 GHz single-core boost, 2.9 GHz all-core boost
    - Peak FP32: ~2.78 TFLOPS @ 2.9 GHz all-core (60 cores × 16 ops/cycle)
    - Peak INT8: ~88.7 TOPS with AMX (60 cores × 2 tiles × 256 ops/cycle @ 2.9 GHz)
    - AMX: Advanced Matrix Extensions for AI acceleration

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (2.9 MB total)
    - L1 Instruction: 32 KB per core (1.9 MB total)
    - L2: 2 MB per core (120 MB total)
    - L3 (LLC): 112.5 MB shared

    MEMORY:
    - 8-channel DDR5-4800
    - Up to 4TB capacity
    - Peak bandwidth: 307 GB/s (8 × 38.4 GB/s)

    POWER:
    - TDP: 350W
    - Idle: ~80W (estimate)
    - Dynamic: ~270W

    AI ACCELERATION:
    - AMX (Advanced Matrix Extensions): INT8, BF16 matrix operations
    - VNNI (Vector Neural Network Instructions): INT8 dot products
    - Deep Learning Boost
    - Better AI performance than previous Xeon generations

    CONNECTIVITY:
    - PCIe 5.0: 80 lanes
    - CXL 1.1 support

    USE CASES:
    - AI training and inference
    - HPC workloads
    - Database servers
    - Virtualization hosts

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Intel published specs
    - AMX performance is theoretical peak
    - Sustained performance depends on thermal limits

    REFERENCES:
    - Intel Xeon Scalable Processors (4th Gen) Product Brief
    - Intel AMX Architecture Guide
    """
    # Performance calculations
    num_cores = 60
    base_clock_hz = 2.0e9  # 2.0 GHz
    boost_clock_hz = 3.5e9  # 3.5 GHz (single core)
    all_core_boost_hz = 2.9e9  # 2.9 GHz (realistic sustained)

    # AVX-512: 16 FP32 ops/cycle per core
    fp32_ops_per_core = 16  # AVX-512
    fp16_ops_per_core = 32  # AVX-512 FP16

    # AMX: 2 tiles per core, each tile 16×16 INT8
    amx_tiles_per_core = 2
    amx_ops_per_tile = 256  # 16×16 matrix

    # Peak performance (all-core boost)
    peak_fp32 = num_cores * fp32_ops_per_core * all_core_boost_hz  # 2.78 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * all_core_boost_hz  # 5.57 TFLOPS
    peak_int8_amx = num_cores * amx_tiles_per_core * amx_ops_per_tile * all_core_boost_hz  # 88.7 TOPS

    # Memory bandwidth (8-channel DDR5-4800)
    channels = 8
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 307.2 GB/s

    # Power and energy
    tdp = 350.0  # Watts
    idle_power = 80.0  # Estimated
    dynamic_power = tdp - idle_power  # 270W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~97 pJ/FLOP
    energy_per_byte = 30e-12  # 30 pJ/byte (datacenter-class)

    return HardwareResourceModel(
        name="Intel-Xeon-Platinum-8490H",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 60 cores
        threads_per_unit=2,  # HyperThreading (SMT)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # AVX-512 width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.39 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 2.78 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_int8_amx / 2,  # 44.4 TFLOPS (AMX BF16)
                tensor_core_supported=True,  # AMX
                relative_speedup=16.0,  # AMX is much faster than SIMD
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 5.57 TFLOPS (AVX-512 FP16)
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8_amx,  # 88.7 TOPS (AMX INT8)
                tensor_core_supported=True,  # AMX
                relative_speedup=32.0,  # AMX is very fast for INT8
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 307.2 GB/s (8-channel DDR5-4800)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core
        l2_cache_total=120 * 1024 * 1024,  # 120 MB total L2 (2 MB × 60 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~97 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 30 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def amd_epyc_9654_resource_model() -> HardwareResourceModel:
    """
    AMD EPYC 9654 (Genoa) - Datacenter Server Processor.

    ARCHITECTURE:
    - 96 Zen 4 cores (192 threads with SMT)
    - TSMC 5nm process
    - Chiplet design (12× 8-core CCDs + I/O die)

    PERFORMANCE:
    - Clock: 2.4 GHz base, 3.7 GHz boost
    - Peak FP32: ~1.84 TFLOPS @ 2.4 GHz base (96 cores × 8 effective ops/cycle)
    - AVX-512: Double-pumped 256-bit (not native 512-bit)
    - Peak INT8: ~7.37 TOPS (96 cores × 32 ops/cycle @ 2.4 GHz)

    CACHE HIERARCHY:
    - L1 Data: 32 KB per core (3.1 MB total)
    - L1 Instruction: 32 KB per core (3.1 MB total)
    - L2: 1 MB per core (96 MB total)
    - L3: 384 MB shared (32 MB per CCD × 12 CCDs)

    MEMORY:
    - 12-channel DDR5-4800
    - Up to 6TB capacity
    - Peak bandwidth: 460.8 GB/s (12 × 38.4 GB/s)

    POWER:
    - TDP: 360W (can be tuned up to 400W)
    - Idle: ~90W (estimate)
    - Dynamic: ~270W

    AI ACCELERATION:
    - AVX-512 support (double-pumped, slower than native)
    - AVX2 for wider compatibility
    - No dedicated AI accelerator (unlike Intel AMX)

    CONNECTIVITY:
    - PCIe 5.0: 128 lanes
    - CXL 1.1+ support

    USE CASES:
    - Cloud computing (highest core density)
    - Virtualization (many VMs)
    - Database servers
    - Scientific computing

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on AMD published specs
    - AVX-512 is double-pumped, so effective throughput is lower

    REFERENCES:
    - AMD EPYC 9004 Series Processors Product Brief
    - AMD Zen 4 Architecture
    """
    # Performance calculations
    num_cores = 96
    base_clock_hz = 2.4e9  # 2.4 GHz
    boost_clock_hz = 3.7e9  # 3.7 GHz (single core)

    # AVX-512 (double-pumped): Effective 8 FP32 ops/cycle
    # Native 256-bit × 2 ops = 8 FP32 per cycle (but takes 2 cycles for 512-bit)
    fp32_ops_per_core = 8  # Effective (double-pumped AVX-512)
    fp16_ops_per_core = 16  # Effective
    int8_ops_per_core = 32  # Effective

    # Peak performance (base clock, conservative)
    peak_fp32 = num_cores * fp32_ops_per_core * base_clock_hz  # 1.84 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * base_clock_hz  # 3.69 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * base_clock_hz  # 7.37 TOPS

    # Memory bandwidth (12-channel DDR5-4800)
    channels = 12
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 460.8 GB/s

    # Power and energy
    tdp = 360.0  # Watts
    idle_power = 90.0  # Estimated
    dynamic_power = tdp - idle_power  # 270W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~147 pJ/FLOP
    energy_per_byte = 28e-12  # 28 pJ/byte (datacenter-class)

    return HardwareResourceModel(
        name="AMD-EPYC-9654-Genoa",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 96 cores
        threads_per_unit=2,  # SMT (2 threads per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=8,  # Effective SIMD width for FP32 (double-pumped)

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 0.92 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 1.84 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 3.69 TFLOPS
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8,  # 7.37 TOPS
                tensor_core_supported=False,  # No AMX equivalent
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 460.8 GB/s (12-channel DDR5-4800)
        l1_cache_per_unit=32 * 1024,  # 32 KB L1D per core
        l2_cache_total=96 * 1024 * 1024,  # 96 MB total L2 (1 MB × 96 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 6TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~147 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 28 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def amd_epyc_9754_resource_model() -> HardwareResourceModel:
    """
    AMD EPYC 9754 (Genoa) - Flagship 128-core Datacenter Processor.

    ARCHITECTURE:
    - 128 Zen 4 cores (256 threads with SMT)
    - TSMC 5nm process
    - Chiplet design (16× 8-core CCDs + I/O die)
    - 33% more cores than EPYC 9654

    PERFORMANCE:
    - Clock: 2.25 GHz base, 3.1 GHz boost
    - Peak FP32: ~2.30 TFLOPS @ 2.25 GHz base (128 cores × 8 effective ops/cycle)
    - AVX-512: Double-pumped 256-bit (not native 512-bit)
    - Peak INT8: ~9.22 TOPS (128 cores × 32 ops/cycle @ 2.25 GHz)

    CACHE HIERARCHY:
    - L1 Data: 32 KB per core (4 MB total)
    - L1 Instruction: 32 KB per core (4 MB total)
    - L2: 1 MB per core (128 MB total)
    - L3: 512 MB shared (32 MB per CCD × 16 CCDs)

    MEMORY:
    - 12-channel DDR5-4800
    - Up to 6TB capacity
    - Peak bandwidth: 460.8 GB/s (12 × 38.4 GB/s)

    POWER:
    - TDP: 360W (same as 9654)
    - Idle: ~95W (estimate, more cores)
    - Dynamic: ~265W

    AI ACCELERATION:
    - AVX-512 support (double-pumped, slower than native)
    - AVX2 for wider compatibility
    - No dedicated AI accelerator (unlike Intel AMX)

    CONNECTIVITY:
    - PCIe 5.0: 128 lanes
    - CXL 1.1+ support

    USE CASES:
    - Cloud computing (highest core density)
    - Virtualization (256 threads!)
    - Database servers (many concurrent connections)
    - Scientific computing (massively parallel workloads)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on AMD published specs
    - Slightly lower clocks than 9654 due to higher core count

    REFERENCES:
    - AMD EPYC 9004 Series Processors Product Brief
    - AMD Zen 4 Architecture
    """
    # Performance calculations
    num_cores = 128
    base_clock_hz = 2.25e9  # 2.25 GHz (lower than 9654 due to more cores)
    boost_clock_hz = 3.1e9  # 3.1 GHz (single core)

    # AVX-512 (double-pumped): Effective 8 FP32 ops/cycle
    fp32_ops_per_core = 8  # Effective (double-pumped AVX-512)
    fp16_ops_per_core = 16  # Effective
    int8_ops_per_core = 32  # Effective

    # Peak performance (base clock, conservative)
    peak_fp32 = num_cores * fp32_ops_per_core * base_clock_hz  # 2.30 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * base_clock_hz  # 4.61 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * base_clock_hz  # 9.22 TOPS

    # Memory bandwidth (12-channel DDR5-4800, same as 9654)
    channels = 12
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 460.8 GB/s

    # Power and energy
    tdp = 360.0  # Watts (same as 9654)
    idle_power = 95.0  # Estimated (more cores)
    dynamic_power = tdp - idle_power  # 265W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~115 pJ/FLOP
    energy_per_byte = 28e-12  # 28 pJ/byte (datacenter-class)

    return HardwareResourceModel(
        name="AMD-EPYC-9754-Genoa",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 128 cores
        threads_per_unit=2,  # SMT (2 threads per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=8,  # Effective SIMD width for FP32 (double-pumped)

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.15 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 2.30 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 4.61 TFLOPS
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8,  # 9.22 TOPS
                tensor_core_supported=False,  # No AMX equivalent
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 460.8 GB/s (12-channel DDR5-4800)
        l1_cache_per_unit=32 * 1024,  # 32 KB L1D per core
        l2_cache_total=128 * 1024 * 1024,  # 128 MB total L2 (1 MB × 128 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 6TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~115 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 28 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def intel_xeon_platinum_8592plus_resource_model() -> HardwareResourceModel:
    """
    Intel Xeon Platinum 8592+ (Sapphire Rapids) - Flagship 64-core Datacenter Processor.

    ARCHITECTURE:
    - 64 Golden Cove cores (128 threads with HyperThreading)
    - Intel 7 process (10nm Enhanced SuperFin)
    - Monolithic die design
    - 7% more cores than 8490H

    PERFORMANCE:
    - Clock: 1.9 GHz base, 3.9 GHz single-core boost, 3.0 GHz all-core boost
    - Peak FP32: ~3.07 TFLOPS @ 3.0 GHz all-core (64 cores × 16 ops/cycle)
    - Peak INT8: ~98.3 TOPS with AMX (64 cores × 2 tiles × 256 ops/cycle @ 3.0 GHz)
    - AMX: Advanced Matrix Extensions for AI acceleration

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (3.1 MB total)
    - L1 Instruction: 32 KB per core (2.0 MB total)
    - L2: 2 MB per core (128 MB total)
    - L3 (LLC): 120 MB shared

    MEMORY:
    - 8-channel DDR5-4800
    - Up to 4TB capacity
    - Peak bandwidth: 307.2 GB/s (8 × 38.4 GB/s)

    POWER:
    - TDP: 350W (same as 8490H)
    - Idle: ~85W (estimate)
    - Dynamic: ~265W

    AI ACCELERATION:
    - AMX (Advanced Matrix Extensions): INT8, BF16 matrix operations
    - VNNI (Vector Neural Network Instructions): INT8 dot products
    - Deep Learning Boost
    - Highest AMX performance in Sapphire Rapids lineup

    CONNECTIVITY:
    - PCIe 5.0: 80 lanes
    - CXL 1.1 support

    USE CASES:
    - AI training and inference (flagship AI SKU)
    - HPC workloads
    - Database servers
    - Virtualization hosts

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Intel published specs
    - AMX performance is theoretical peak
    - Sustained performance depends on thermal limits

    REFERENCES:
    - Intel Xeon Scalable Processors (4th Gen) Product Brief
    - Intel AMX Architecture Guide
    """
    # Performance calculations
    num_cores = 64
    base_clock_hz = 1.9e9  # 1.9 GHz
    boost_clock_hz = 3.9e9  # 3.9 GHz (single core)
    all_core_boost_hz = 3.0e9  # 3.0 GHz (realistic sustained, higher than 8490H)

    # AVX-512: 16 FP32 ops/cycle per core
    fp32_ops_per_core = 16  # AVX-512
    fp16_ops_per_core = 32  # AVX-512 FP16

    # AMX: 2 tiles per core, each tile 16×16 INT8
    amx_tiles_per_core = 2
    amx_ops_per_tile = 256  # 16×16 matrix

    # Peak performance (all-core boost)
    peak_fp32 = num_cores * fp32_ops_per_core * all_core_boost_hz  # 3.07 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * all_core_boost_hz  # 6.14 TFLOPS
    peak_int8_amx = num_cores * amx_tiles_per_core * amx_ops_per_tile * all_core_boost_hz  # 98.3 TOPS

    # Memory bandwidth (8-channel DDR5-4800, same as 8490H)
    channels = 8
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 307.2 GB/s

    # Power and energy
    tdp = 350.0  # Watts (same as 8490H)
    idle_power = 85.0  # Estimated
    dynamic_power = tdp - idle_power  # 265W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~86 pJ/FLOP
    energy_per_byte = 30e-12  # 30 pJ/byte (datacenter-class)

    return HardwareResourceModel(
        name="Intel-Xeon-Platinum-8592+",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 64 cores
        threads_per_unit=2,  # HyperThreading (SMT)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # AVX-512 width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.54 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 3.07 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_int8_amx / 2,  # 49.2 TFLOPS (AMX BF16)
                tensor_core_supported=True,  # AMX
                relative_speedup=16.0,  # AMX is much faster than SIMD
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 6.14 TFLOPS (AVX-512 FP16)
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8_amx,  # 98.3 TOPS (AMX INT8)
                tensor_core_supported=True,  # AMX
                relative_speedup=32.0,  # AMX is very fast for INT8
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 307.2 GB/s (8-channel DDR5-4800)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core
        l2_cache_total=128 * 1024 * 1024,  # 128 MB total L2 (2 MB × 64 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~86 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 30 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def ampere_ampereone_128_resource_model() -> HardwareResourceModel:
    """
    Ampere AmpereOne 128-core ARM Server Processor (A128-30X mid-tier).

    ARCHITECTURE:
    - 128 Ampere 64-bit ARM v8.6+ cores
    - 2×128-bit SIMD units per core (NEON + SVE)
    - Coherent mesh interconnect with distributed snoop filtering
    - TSMC 5nm process
    - 67% of 192-core flagship

    PERFORMANCE:
    - Clock: Up to 3.6 GHz (consistent across all cores)
    - Peak FP32: ~3.69 TFLOPS (128 cores × 8 ops/cycle × 3.6 GHz)
    - Peak FP16/BF16: ~7.37 TFLOPS (128 cores × 16 ops/cycle × 3.6 GHz)
    - Peak INT8: ~14.75 TOPS (128 cores × 32 ops/cycle × 3.6 GHz)

    CACHE HIERARCHY:
    - L1 Data: 64 KB per core (8 MB total)
    - L1 Instruction: 16 KB per core (2 MB total)
    - L2: 2 MB per core (256 MB total)
    - System Cache (L3-like): 48 MB shared

    MEMORY:
    - 8-channel DDR5-5200 (up to 4TB)
    - Peak bandwidth: 332.8 GB/s (8 × 41.6 GB/s)

    POWER:
    - TDP: 210W (A128-30X)
    - Idle: ~40W (estimate)
    - Dynamic: ~170W

    AI ACCELERATION:
    - Native FP16/BF16 support (2×128-bit SIMD)
    - Native INT8/INT16 support
    - Ampere AIO (AI Optimizer) for ML frameworks
    - Better than x86 for AI inference (wider SIMD for low precision)

    CONNECTIVITY:
    - 128 lanes PCIe 5.0 with 32 controllers

    USE CASES:
    - Cloud-native workloads (microservices, containers)
    - AI inference at scale (cloud servers)
    - High-performance computing (HPC)
    - Cost-effective datacenter deployments

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Ampere published specs
    - SIMD calculations based on ARM v8.6+ NEON/SVE
    - Memory bandwidth from DDR5-5200 8-channel configuration

    REFERENCES:
    - Ampere AmpereOne Family Product Brief (2024)
    - ARM v8.6+ Architecture Reference Manual
    - DDR5-5200 specifications
    """
    # Performance calculations
    num_cores = 128
    clock_hz = 3.6e9  # 3.6 GHz
    simd_units_per_core = 2  # 2×128-bit SIMD units

    # SIMD operations per cycle per core
    # 128-bit SIMD: FP32=4, FP16=8, INT8=16 ops per unit
    fp32_ops_per_core = simd_units_per_core * 4  # 8 FP32 ops/cycle
    fp16_ops_per_core = simd_units_per_core * 8  # 16 FP16/BF16 ops/cycle
    int8_ops_per_core = simd_units_per_core * 16  # 32 INT8 ops/cycle

    # Peak performance
    peak_fp32 = num_cores * fp32_ops_per_core * clock_hz  # 3.69 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * clock_hz  # 7.37 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * clock_hz  # 14.75 TOPS

    # Memory bandwidth (8-channel DDR5-5200, same as 192-core)
    channels = 8
    ddr5_rate = 5200e6  # 5200 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 332.8 GB/s

    # Power and energy
    tdp = 210.0  # Watts (A128-30X, lower than 192-core)
    idle_power = 40.0  # Estimated
    dynamic_power = tdp - idle_power  # 170W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~46 pJ/FLOP
    energy_per_byte = 25e-12  # 25 pJ/byte (server-class)

    return HardwareResourceModel(
        name="Ampere-AmpereOne-128core",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 128 cores
        threads_per_unit=1,  # No SMT/HyperThreading (single thread per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # Effective SIMD width for INT8 (per unit)

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.84 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 3.69 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_fp16,  # 7.37 TFLOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 7.37 TFLOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=peak_fp16,  # 7.37 TOPS (same as FP16)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8,  # 14.75 TOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 332.8 GB/s (8-channel DDR5-5200)
        l1_cache_per_unit=64 * 1024,  # 64 KB L1D per core
        l2_cache_total=256 * 1024 * 1024,  # 256 MB total L2 (2 MB × 128 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~46 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 25 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def intel_granite_rapids_resource_model() -> HardwareResourceModel:
    """
    Intel Xeon Granite Rapids (Next-Gen 2024-2025) - 128-core Datacenter Processor.

    ARCHITECTURE:
    - 128 Redwood Cove P-cores (256 threads with HyperThreading)
    - Intel 3 process (Enhanced FinFET)
    - Tile-based chiplet design (new for Intel)
    - 2× core count vs Sapphire Rapids flagship

    PERFORMANCE:
    - Clock: 2.0 GHz base, 3.8 GHz single-core boost, 3.2 GHz all-core boost (estimated)
    - Peak FP32: ~6.55 TFLOPS @ 3.2 GHz all-core (128 cores × 16 ops/cycle)
    - Peak INT8: ~209 TOPS with AMX (128 cores × 2 tiles × 256 ops/cycle @ 3.2 GHz)
    - Enhanced AMX with sparsity acceleration

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (6.1 MB total)
    - L1 Instruction: 32 KB per core (4.1 MB total)
    - L2: 2 MB per core (256 MB total)
    - L3 (LLC): 320 MB shared (distributed across tiles)

    MEMORY:
    - 8-channel DDR5-5600 (improved from DDR5-4800)
    - 12-channel DDR5-5600 on HBM SKUs
    - Peak bandwidth: 358.4 GB/s (8 × 44.8 GB/s) or 537.6 GB/s (12-channel)

    POWER:
    - TDP: 500W (high core count)
    - Idle: ~100W (estimate)
    - Dynamic: ~400W

    AI ACCELERATION:
    - Enhanced AMX with INT4, FP8 support
    - Sparsity acceleration (structured sparsity)
    - VNNI improvements for INT8
    - Better AMX efficiency than Sapphire Rapids

    CONNECTIVITY:
    - PCIe 6.0: 96 lanes
    - CXL 2.0 support
    - UPI (Ultra Path Interconnect) for multi-socket

    USE CASES:
    - Large-scale AI training and inference
    - HPC workloads
    - Cloud computing at scale
    - Next-generation datacenter deployments

    CALIBRATION STATUS:
    ⚠ PROJECTED - Based on Intel roadmap and industry estimates
    - Not yet shipping (2024-2025 timeline)
    - Specs are projections based on Intel disclosures

    REFERENCES:
    - Intel Xeon Roadmap 2024
    - Intel 3 Process Technology Brief
    - Industry analyst projections
    """
    # Performance calculations
    num_cores = 128
    base_clock_hz = 2.0e9  # 2.0 GHz
    boost_clock_hz = 3.8e9  # 3.8 GHz (single core, estimated)
    all_core_boost_hz = 3.2e9  # 3.2 GHz (estimated, higher than Sapphire Rapids)

    # AVX-512: 16 FP32 ops/cycle per core
    fp32_ops_per_core = 16  # AVX-512
    fp16_ops_per_core = 32  # AVX-512 FP16

    # Enhanced AMX: 2 tiles per core, each tile 16×16 INT8
    amx_tiles_per_core = 2
    amx_ops_per_tile = 256  # 16×16 matrix

    # Peak performance (all-core boost)
    peak_fp32 = num_cores * fp32_ops_per_core * all_core_boost_hz  # 6.55 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * all_core_boost_hz  # 13.11 TFLOPS
    peak_int8_amx = num_cores * amx_tiles_per_core * amx_ops_per_tile * all_core_boost_hz  # 209.7 TOPS

    # Memory bandwidth (8-channel DDR5-5600, improved)
    channels = 8
    ddr5_rate = 5600e6  # 5600 MT/s (up from 4800)
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 358.4 GB/s

    # Power and energy
    tdp = 500.0  # Watts (higher due to 128 cores)
    idle_power = 100.0  # Estimated
    dynamic_power = tdp - idle_power  # 400W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~61 pJ/FLOP
    energy_per_byte = 28e-12  # 28 pJ/byte (improved process)

    return HardwareResourceModel(
        name="Intel-Xeon-Granite-Rapids",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 128 cores
        threads_per_unit=2,  # HyperThreading (SMT)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # AVX-512 width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 3.28 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 6.55 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_int8_amx / 2,  # 104.9 TFLOPS (Enhanced AMX BF16)
                tensor_core_supported=True,  # Enhanced AMX
                relative_speedup=16.0,  # AMX is much faster than SIMD
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 13.11 TFLOPS (AVX-512 FP16)
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8_amx,  # 209.7 TOPS (Enhanced AMX INT8)
                tensor_core_supported=True,  # Enhanced AMX
                relative_speedup=32.0,  # AMX is very fast for INT8
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=peak_int8_amx * 2,  # 419.4 TOPS (Enhanced AMX INT4)
                tensor_core_supported=True,  # Enhanced AMX with INT4 support
                relative_speedup=64.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 358.4 GB/s (8-channel DDR5-5600)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core
        l2_cache_total=256 * 1024 * 1024,  # 256 MB total L2 (2 MB × 128 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~61 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 28 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def amd_epyc_turin_resource_model() -> HardwareResourceModel:
    """
    AMD EPYC Turin (Zen 5, Next-Gen 2024-2025) - 192-core Datacenter Processor.

    ARCHITECTURE:
    - 192 Zen 5 cores (384 threads with SMT)
    - TSMC 3nm process (N3)
    - Chiplet design (24× 8-core CCDs + I/O die)
    - 50% more cores than EPYC 9754

    PERFORMANCE:
    - Clock: 2.5 GHz base, 3.8 GHz boost (estimated)
    - Peak FP32: ~3.84 TFLOPS @ 2.5 GHz base (192 cores × 8 effective ops/cycle)
    - AVX-512: Native 512-bit (improved from double-pumped Zen 4)
    - Peak INT8: ~15.36 TOPS (192 cores × 32 ops/cycle @ 2.5 GHz)

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (9.2 MB total, increased from 32 KB)
    - L1 Instruction: 32 KB per core (6.1 MB total)
    - L2: 1 MB per core (192 MB total)
    - L3: 768 MB shared (32 MB per CCD × 24 CCDs)

    MEMORY:
    - 12-channel DDR5-6000 (up from DDR5-4800)
    - Up to 6TB capacity
    - Peak bandwidth: 576 GB/s (12 × 48 GB/s)
    - 25% more bandwidth than EPYC 9000 series

    POWER:
    - TDP: 500W (higher core count)
    - Idle: ~120W (estimate)
    - Dynamic: ~380W

    AI ACCELERATION:
    - Native AVX-512 support (improved from double-pumped)
    - AVX2 for compatibility
    - Possible AI matrix accelerator (rumored, not confirmed)
    - Better INT8 performance than Zen 4

    CONNECTIVITY:
    - PCIe 6.0: 160 lanes
    - CXL 2.0 support

    USE CASES:
    - Cloud computing (extreme core density)
    - Virtualization (384 threads!)
    - Database servers (massive concurrent connections)
    - Large-scale AI inference

    CALIBRATION STATUS:
    ⚠ PROJECTED - Based on AMD roadmap and industry estimates
    - Not yet shipping (2024-2025 timeline)
    - Specs are projections based on AMD disclosures

    REFERENCES:
    - AMD EPYC Roadmap 2024
    - AMD Zen 5 Architecture Disclosures
    - Industry analyst projections
    """
    # Performance calculations
    num_cores = 192
    base_clock_hz = 2.5e9  # 2.5 GHz (estimated)
    boost_clock_hz = 3.8e9  # 3.8 GHz (single core, estimated)

    # Native AVX-512: Effective 8 FP32 ops/cycle (improved throughput)
    fp32_ops_per_core = 8  # Native AVX-512 (not double-pumped)
    fp16_ops_per_core = 16  # Native AVX-512 FP16
    int8_ops_per_core = 32  # Native AVX-512 INT8

    # Peak performance (base clock, conservative)
    peak_fp32 = num_cores * fp32_ops_per_core * base_clock_hz  # 3.84 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * base_clock_hz  # 7.68 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * base_clock_hz  # 15.36 TOPS

    # Memory bandwidth (12-channel DDR5-6000, improved)
    channels = 12
    ddr5_rate = 6000e6  # 6000 MT/s (up from 4800)
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 576 GB/s

    # Power and energy
    tdp = 500.0  # Watts (higher due to 192 cores)
    idle_power = 120.0  # Estimated
    dynamic_power = tdp - idle_power  # 380W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~99 pJ/FLOP
    energy_per_byte = 26e-12  # 26 pJ/byte (improved 3nm process)

    return HardwareResourceModel(
        name="AMD-EPYC-Turin-Zen5",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 192 cores
        threads_per_unit=2,  # SMT (2 threads per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=8,  # SIMD width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.92 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 3.84 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 7.68 TFLOPS
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8,  # 15.36 TOPS
                tensor_core_supported=False,  # No AMX-like accelerator yet
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 576 GB/s (12-channel DDR5-6000)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core (increased from 32 KB)
        l2_cache_total=192 * 1024 * 1024,  # 192 MB total L2 (1 MB × 192 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 6TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~99 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 26 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


def xilinx_vitis_ai_dpu_resource_model() -> HardwareResourceModel:
    """
    Xilinx Vitis AI DPU (Deep Processing Unit) resource model.

    Configuration: B4096 (4096 MACs) on Versal VE2302 (embodied AI target)
    Architecture: AIE-ML v1 with 2D array of INT8 ALUs

    Key characteristics:
    - FPGA-based, reconfigurable
    - Native INT8 support (best performance)
    - Power efficient: 15-20W (embodied AI sweet spot)
    - Scratchpad-based memory hierarchy (similar to KPU)
    - 75% realistic efficiency (per specification)

    References:
    - Versal VE2302: 15-20W, edge-optimized
    - AIE-ML v1: 512 INT8 ops/clock @ 1.25 GHz
    - B4096: 4096 MACs
    - Realistic peak: 10.24 TOPS × 0.75 = 7.68 TOPS INT8
    """
    # Calculate theoretical and realistic peak performance
    mac_units = 4096
    clock_freq = 1.25e9  # 1.25 GHz (confirmed from Versal docs)
    ops_per_mac = 2  # Multiply + Accumulate
    efficiency = 0.75  # User-specified efficiency

    theoretical_tops = mac_units * ops_per_mac * clock_freq  # 10.24 TOPS
    realistic_tops = theoretical_tops * efficiency  # 7.68 TOPS

    # Power profile (VE2302: 15-20W)
    power_avg = 17.5  # Watts
    idle_power = 3.0  # Estimated idle
    dynamic_power = power_avg - idle_power  # 14.5W

    # Energy per operation
    energy_per_int8_op = dynamic_power / realistic_tops  # 1.89e-12 J/op
    # Convert to FP32 equivalent (INT8 is ~4× more efficient)
    energy_per_flop_fp32 = energy_per_int8_op * 4  # 7.56e-12 J/FLOP

    return HardwareResourceModel(
        name="DPU-Vitis-AI-B4096",
        hardware_type=HardwareType.DPU,
        compute_units=64,  # Tiles (estimate for B4096)
        threads_per_unit=64,  # Operations per tile
        warps_per_unit=8,  # Vector lanes per tile
        warp_size=8,

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=realistic_tops / 8,  # 0.96 TFLOPS (not native)
                tensor_core_supported=False,
                relative_speedup=0.125,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=realistic_tops / 4,  # 1.92 TFLOPS
                tensor_core_supported=True,  # AIE support
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=realistic_tops,  # 7.68 TOPS (native, best)
                tensor_core_supported=True,  # Native INT8 MACs
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=50e9,  # 50 GB/s DDR4 (edge device)
        l1_cache_per_unit=64 * 1024,  # 64 KB scratchpad per tile
        l2_cache_total=4 * 1024 * 1024,  # 4 MB (estimate)
        main_memory=8 * 1024**3,  # 8 GB DDR4 (edge deployment)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~7.56 pJ/FLOP
        energy_per_byte=15e-12,  # Similar to GPU (FPGA I/O)
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,  # Tiles allocated in pairs
    )


def stanford_plasticine_cgra_resource_model() -> HardwareResourceModel:
    """
    Stanford Plasticine CGRA (Coarse-Grained Reconfigurable Architecture) resource model.

    Configuration: Hypothetical Plasticine-v2 (newer generation)
    Architecture: Spatial dataflow with medium-granularity PCUs

    Key characteristics:
    - Spatial dataflow execution (fundamentally different from temporal)
    - Medium-grained PCUs (balanced coverage vs overhead)
    - Reconfigurable fabric (like DPU but different execution model)
    - Conservative reconfiguration overhead (1000 cycles - Achilles heel)
    - Power budget: 15W (embodied AI range: 10-25W)

    Design Trade-offs:
    - PCU granularity: Medium (not too fine, not too coarse)
      - Too fine: Reconfigurable fabric overhead kills cost/power
      - Too coarse: Becomes multi-core CPU, loses flexibility
    - 32 PCUs covers most DNN operations (Conv, MatMul, etc.)
    - Each PCU: ~8 MACs, ~16 GOPS INT8

    Performance:
    - Peak: 10 TOPS INT8 theoretical
    - Realistic @ 60% efficiency: 6 TOPS INT8
    - Similar to DPU (7.68 TOPS) but different trade-offs

    References:
    - Stanford Plasticine architecture (Prabhakar et al.)
    - Hypothetical v2: 32 PCUs, medium granularity
    - Power: 15W (embodied AI target)
    - Reconfiguration: 1000 cycles (conservative modeling)
    """
    # Calculate theoretical and realistic peak performance
    num_pcus = 32  # Pattern Compute Units
    macs_per_pcu = 8  # Medium granularity
    clock_freq = 1.0e9  # 1 GHz typical for CGRAs
    ops_per_mac = 2  # Multiply + Accumulate
    efficiency = 0.60  # 60% efficiency (fabric overhead)

    # Theoretical peak
    theoretical_tops = num_pcus * macs_per_pcu * ops_per_mac * clock_freq  # 10.24 TOPS
    realistic_tops = theoretical_tops * efficiency  # 6.14 TOPS

    # Power profile (embodied AI range: 10-25W, use 15W midpoint)
    power_avg = 15.0  # Watts
    idle_power = 2.0  # Estimated idle (lower than DPU due to simpler fabric)
    dynamic_power = power_avg - idle_power  # 13W

    # Energy per operation
    energy_per_int8_op = dynamic_power / realistic_tops  # 2.12e-12 J/op
    # Convert to FP32 equivalent (INT8 is ~4× more efficient)
    energy_per_flop_fp32 = energy_per_int8_op * 4  # 8.48e-12 J/FLOP

    return HardwareResourceModel(
        name="CGRA-Plasticine-v2",
        hardware_type=HardwareType.CGRA,
        compute_units=32,  # PCUs (Pattern Compute Units)
        threads_per_unit=8,  # Operations per PCU
        warps_per_unit=1,  # Spatial execution (no warp concept)
        warp_size=1,

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=realistic_tops / 8,  # 0.77 TFLOPS (not native)
                tensor_core_supported=False,
                relative_speedup=0.125,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=realistic_tops / 4,  # 1.54 TFLOPS
                tensor_core_supported=False,
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=realistic_tops,  # 6.14 TOPS (best)
                tensor_core_supported=False,  # Spatial execution, not tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=40e9,  # 40 GB/s on-chip interconnect
        l1_cache_per_unit=64 * 1024,  # 64 KB scratchpad per PCU
        l2_cache_total=2 * 1024 * 1024,  # 2 MB shared
        main_memory=4 * 1024**3,  # 4 GB DDR4 (edge device)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~8.48 pJ/FLOP
        energy_per_byte=12e-12,  # Similar to KPU (on-chip network)
        min_occupancy=0.3,
        max_concurrent_kernels=1,  # Spatial execution (entire graph mapped)
        wave_quantization=1,
    )


def jetson_orin_agx_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin AGX with realistic DVFS-aware multi-power-profile modeling.

    Configuration: AGX variant (2048 CUDA cores, 32 Ampere SMs, 64 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim: 275 TOPS INT8 (sparse networks, all engines: GPU+DLA+PVA)
    - Dense networks total: 138 TOPS INT8 (GPU + 2×DLA)
      - GPU only (dense): 85 TOPS INT8 ← Relevant for PyTorch workloads
      - 2×DLA (dense): 52.5 TOPS INT8
    - GPU sparse: 170 TOPS INT8 (requires specially-prepared sparse networks)
    - Customer empirical data: 2-4% of peak at typical power budgets
    - Root cause: Severe DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    15W Mode (Passive Cooling - What Customers Actually Deploy):
    - Base clock: 306 MHz (guaranteed minimum)
    - Boost clock: 1.02 GHz (datasheet spec, rarely sustained)
    - Sustained clock: 400 MHz (empirical under thermal load)
    - Thermal throttle factor: 39% (severe throttling!)
    - Effective INT8: ~5 TOPS (6% of 85 TOPS GPU dense peak)
    - Use case: Battery-powered robots, drones (must avoid thermal shutdown)

    30W Mode (Active Cooling - Better but Still Throttles):
    - Sustained clock: 650 MHz (64% of boost)
    - Effective INT8: ~17 TOPS (20% of 85 TOPS GPU dense peak)
    - Use case: Tethered robots with active cooling

    60W Mode (Max Performance - Unrealistic for Embodied AI):
    - Sustained clock: 1.0 GHz (98% of boost)
    - Effective INT8: ~51 TOPS (60% of 85 TOPS GPU dense peak)
    - Use case: Benchtop testing only (too hot for deployment!)

    References:
    - Jetson Orin AGX Datasheet: NVIDIA Technical Brief
    - Empirical measurements: Customer lab data (2-4% of peak @ 15W)
    - DVFS behavior: Observed clock throttling under sustained load
    """
    # Physical hardware specs (constant across power modes)
    num_sms = 32
    cuda_cores_per_sm = 64
    int8_ops_per_sm_per_clock = 512  # Tensor Core capability: 64 × 8
    fp32_ops_per_sm_per_clock = 64   # CUDA core capability
    fp16_ops_per_sm_per_clock = 512  # Tensor Core FP16

    # ========================================================================
    # 15W MODE: Realistic deployment configuration (passive cooling)
    # ========================================================================
    clock_15w = ClockDomain(
        base_clock_hz=306e6,         # 306 MHz guaranteed minimum
        max_boost_clock_hz=1.02e9,   # 1.02 GHz datasheet boost
        sustained_clock_hz=400e6,    # 400 MHz empirical (39% throttle!)
        dvfs_enabled=True,
    )

    compute_resource_15w_int8 = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_15w,
    )

    # Peak INT8: 32 SMs × 512 ops/SM/clock × 1.02 GHz = 16.7 TOPS (simplified model)
    # NOTE: Actual GPU dense peak is 85 TOPS at 1.3 GHz (using all 64 Tensor Cores)
    # Sustained INT8: 32 × 512 × 400 MHz = 6.5 TOPS
    # Effective INT8: 6.5 TOPS × 0.47 empirical derate = 3.1 TOPS (3.6% of 85 TOPS GPU dense)

    thermal_15w = ThermalOperatingPoint(
        name="15W-passive",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w_int8,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.47,  # 47% of sustained (3% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w_int8,
                efficiency_factor=0.40,  # Worse (more memory bound)
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w_int8,
                efficiency_factor=0.25,  # Much worse
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 30W MODE: Balanced configuration (active fan cooling)
    # ========================================================================
    clock_30w = ClockDomain(
        base_clock_hz=612e6,
        max_boost_clock_hz=1.15e9,
        sustained_clock_hz=650e6,    # 650 MHz sustained (57% throttle)
        dvfs_enabled=True,
    )

    compute_resource_30w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_30w,
    )

    # Sustained INT8: 32 × 512 × 650 MHz = 10.6 TOPS
    # Effective: 10.6 × 0.60 = 6.4 TOPS (7.5% of 85 TOPS GPU dense peak)

    thermal_30w = ThermalOperatingPoint(
        name="30W-active",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.60,  # Better (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.35,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 60W MODE: Max performance (unrealistic for robots - benchtop only!)
    # ========================================================================
    clock_60w = ClockDomain(
        base_clock_hz=918e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.0e9,    # 1.0 GHz sustained (77% of boost)
        dvfs_enabled=True,
    )

    compute_resource_60w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_60w,
    )

    # Sustained INT8: 32 × 512 × 1.0 GHz = 16.4 TOPS
    # Effective: 16.4 × 0.75 = 12.3 TOPS (7.2% of peak)

    thermal_60w = ThermalOperatingPoint(
        name="60W-max",
        tdp_watts=60.0,
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.75,  # Best case (still only 30% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.65,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # Hardware Resource Model (uses NEW thermal operating points)
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-AGX",
        hardware_type=HardwareType.GPU,
        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # NEW: Thermal operating points with DVFS modeling
        thermal_operating_points={
            "15W": thermal_15w,  # Realistic deployment
            "30W": thermal_30w,  # Balanced
            "60W": thermal_60w,  # Max performance (unrealistic)
        },
        default_thermal_profile="15W",  # Most realistic for embodied AI

        # Legacy precision profiles (for backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=85e12,  # GPU dense (PyTorch workloads; 170 TOPS sparse)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=204.8e9,
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=4 * 1024 * 1024,
        main_memory=64 * 1024**3,
        energy_per_flop_fp32=1.0e-12,
        energy_per_byte=15e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=8,
        wave_quantization=4,
    )


def jetson_orin_nano_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin Nano (8GB variant) with realistic DVFS-aware power modeling.

    Configuration: Nano variant (1024 CUDA cores, 16 Ampere SMs, 32 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim (Super): 67 TOPS INT8 (sparse, all engines)
    - Marketing claim (original): 40 TOPS INT8 (sparse, all engines)
    - Dense networks GPU only: ~21 TOPS INT8 (16 SMs × 512 ops/SM/clock × 650 MHz)
    - Customer empirical data: 2-4% of peak at typical power budgets
    - Root cause: Same as AGX - severe DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    7W Mode (Low Power - Battery-Powered Drones/Robots):
    - Base clock: 204 MHz (minimum)
    - Boost clock: 918 MHz (datasheet)
    - Sustained clock: 300 MHz (empirical under thermal load)
    - Thermal throttle factor: 33% (severe throttling!)
    - Effective INT8: ~1.5 TOPS (7% of 21 TOPS GPU dense peak)
    - Use case: Battery-powered drones, small robots (avoid thermal shutdown)

    15W Mode (Balanced - Typical Edge AI Deployment):
    - Sustained clock: 500 MHz (54% of boost)
    - Effective INT8: ~4 TOPS (19% of 21 TOPS GPU dense peak)
    - Use case: Edge AI devices with passive cooling

    References:
    - Jetson Orin Nano Super Specs: 67 TOPS, 102 GB/s, 15W max
    - Jetson Orin Nano 8GB Specs: 40 TOPS, 68 GB/s, 7-15W
    - TechPowerUp GPU Database: 1024 CUDA cores, 32 Tensor cores
    """
    # Physical hardware specs (Nano has half the SMs of AGX)
    num_sms = 16  # 1024 CUDA cores / 64 cores per SM
    cuda_cores_per_sm = 64
    int8_ops_per_sm_per_clock = 512  # Tensor Core: 64 × 8
    fp32_ops_per_sm_per_clock = 64   # CUDA core
    fp16_ops_per_sm_per_clock = 512  # Tensor Core FP16

    # ========================================================================
    # 7W MODE: Low power deployment (battery-powered devices)
    # ========================================================================
    clock_7w = ClockDomain(
        base_clock_hz=204e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=300e6,  # 33% throttle
        dvfs_enabled=True,
    )

    compute_resource_7w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_7w,
    )

    # Sustained INT8: 16 SMs × 512 ops/SM/clock × 300 MHz = 2.46 TOPS
    # Effective: 2.46 × 0.40 = 0.98 TOPS (4.7% of 21 TOPS dense peak)

    thermal_7w = ThermalOperatingPoint(
        name="7W-battery",
        tdp_watts=7.0,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.55,
                efficiency_factor=0.40,  # 40% of sustained (4% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_7w,
                efficiency_factor=0.35,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_7w,
                efficiency_factor=0.20,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 15W MODE: Balanced configuration (typical edge deployment)
    # ========================================================================
    clock_15w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=500e6,  # 54% of boost
        dvfs_enabled=True,
    )

    compute_resource_15w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_15w,
    )

    # Sustained INT8: 16 × 512 × 500 MHz = 4.1 TOPS
    # Effective: 4.1 × 0.50 = 2.05 TOPS (9.7% of 21 TOPS dense peak)

    thermal_15w = ThermalOperatingPoint(
        name="15W-standard",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.50,  # 50% of sustained (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.30,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-Nano",
        hardware_type=HardwareType.GPU,
        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # Thermal operating points with DVFS modeling
        thermal_operating_points={
            "7W": thermal_7w,   # Battery-powered devices
            "15W": thermal_15w,  # Standard edge AI deployment
        },
        default_thermal_profile="7W",  # Most realistic for embodied AI

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=21e12,  # 16 SMs × 512 ops/clock × 650 MHz (realistic peak)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=68e9,  # 68 GB/s (original) or 102 GB/s (Super)
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=2 * 1024 * 1024,  # 2 MB (half of AGX)
        main_memory=8 * 1024**3,  # 8 GB
        energy_per_flop_fp32=1.2e-12,  # Slightly worse efficiency than AGX
        energy_per_byte=18e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,  # Fewer than AGX
        wave_quantization=2,  # Smaller wave size
    )


def jetson_thor_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Thor with realistic DVFS modeling (Next-gen edge AI, 2025+).

    Configuration: Blackwell-based GPU, 64 SMs, 1000 TOPS INT8 peak (actual datapath)

    CRITICAL REALITY CHECK (Projected based on Orin empirical data):
    - NVIDIA claims: 2000 TOPS INT8 (includes sparsity - workload dependent!)
    - Actual datapath: 1000 TOPS INT8 (speed-of-light without sparsity)
    - Expected reality: 3-5% of peak at deployable power budgets
    - Improved thermal design vs Orin, but still throttles significantly

    Power Profiles (Projected):
    ==========================

    30W Mode (Typical Deployment - Autonomous Vehicles):
    - Better thermal design than Orin
    - Sustained clock: 750 MHz (58% of boost)
    - Effective INT8: ~30 TOPS (3% of peak)
    - Use case: Autonomous vehicles with active cooling

    60W Mode (Max Performance - High-end Robotics):
    - Sustained clock: 1.1 GHz (85% of boost)
    - Effective INT8: ~60 TOPS (6% of peak)
    - Use case: Humanoid robots, industrial AGVs

    100W Mode (Benchtop/Development Only):
    - Sustained clock: 1.25 GHz (96% of boost)
    - Effective INT8: ~100 TOPS (10% of peak)
    - Use case: Development workstations (not deployable)
    """
    # Physical hardware (constant across power modes)
    num_sms = 64  # Estimated for 1000 TOPS actual datapath
    int8_ops_per_sm_per_clock = 256  # Actual datapath (not sparsity-inflated)
    fp32_ops_per_sm_per_clock = 128  # Wider SMs
    fp16_ops_per_sm_per_clock = 256  # Match INT8 without sparsity

    # ========================================================================
    # 30W MODE: Typical deployment (autonomous vehicles)
    # ========================================================================
    clock_30w = ClockDomain(
        base_clock_hz=500e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=750e6,  # 58% of boost (better than Orin!)
        dvfs_enabled=True,
    )

    compute_resource_30w = ComputeResource(
        resource_type="Blackwell-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_30w,
    )

    # Sustained: 64 × 256 × 750 MHz = 12.3 TOPS
    # Effective: 12.3 × 0.50 = 6.1 TOPS (0.6% of peak!)

    thermal_30w = ThermalOperatingPoint(
        name="30W-active",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.50,  # 50% of sustained (3% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.30,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 60W MODE: High-performance robotics
    # ========================================================================
    clock_60w = ClockDomain(
        base_clock_hz=800e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.1e9,  # 85% of boost
        dvfs_enabled=True,
    )

    compute_resource_60w = ComputeResource(
        resource_type="Blackwell-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_60w,
    )

    # Sustained: 64 × 256 × 1.1 GHz = 18.0 TOPS
    # Effective: 18.0 × 0.65 = 11.7 TOPS (1.2% of peak)

    thermal_60w = ThermalOperatingPoint(
        name="60W-active",
        tdp_watts=60.0,
        cooling_solution="active-fan-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.65,  # Better (6% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.60,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_60w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 100W MODE: Development/benchtop only
    # ========================================================================
    clock_100w = ClockDomain(
        base_clock_hz=1.0e9,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.25e9,  # 96% of boost
        dvfs_enabled=True,
    )

    compute_resource_100w = ComputeResource(
        resource_type="Blackwell-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_100w,
    )

    # Sustained: 64 × 256 × 1.25 GHz = 20.5 TOPS
    # Effective: 20.5 × 0.80 = 16.4 TOPS (1.6% of peak)

    thermal_100w = ThermalOperatingPoint(
        name="100W-max",
        tdp_watts=100.0,
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_100w,
                efficiency_factor=0.80,  # Best case (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_100w,
                efficiency_factor=0.70,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_100w,
                efficiency_factor=0.55,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Jetson-Thor",
        hardware_type=HardwareType.GPU,
        compute_units=num_sms,
        threads_per_unit=128,
        warps_per_unit=4,
        warp_size=32,

        # NEW: Thermal operating points
        thermal_operating_points={
            "30W": thermal_30w,  # Typical deployment
            "60W": thermal_60w,  # High-performance
            "100W": thermal_100w,  # Development only
        },
        default_thermal_profile="30W",

        # Legacy for backward compat
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=2000e12,
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=450e9,
        l1_cache_per_unit=256 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=128 * 1024**3,
        energy_per_flop_fp32=0.8e-12,
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=16,
        wave_quantization=4,
    )


def coral_edge_tpu_resource_model() -> HardwareResourceModel:
    """
    Google Coral Edge TPU resource model.

    Configuration: Single Edge TPU chip (USB/M.2/PCIe variants)
    Architecture: Scaled-down systolic array from Google TPU

    Key characteristics:
    - Ultra-low power edge AI accelerator (0.5-2W)
    - 4 TOPS INT8 (much smaller than datacenter TPU v4)
    - INT8 quantization required (no FP16/FP32 support)
    - Perfect for IoT, embedded systems, battery-powered devices
    - Cost-effective: ~$25-75 depending on form factor

    References:
    - Coral Edge TPU: 4 TOPS @ INT8 only
    - Power: 0.5W idle, 2W peak (USB variant)
    - Target: Ultra-low-power edge inference (IoT, cameras, drones)
    - Limitation: Requires TensorFlow Lite models with full INT8 quantization

    Note: This is NOT the datacenter TPU v4 - it's designed for
    battery-powered edge devices where power is more critical than performance.
    """
    # Performance specs
    int8_tops = 4e12  # 4 TOPS INT8 (only mode supported)
    efficiency = 0.85  # 85% efficiency (well-optimized systolic array)
    effective_tops = int8_tops * efficiency  # 3.4 TOPS effective

    # Power profile (very low power)
    power_avg = 2.0  # Watts (peak during inference)

    # Energy per operation (ultra-efficient due to low power)
    # 2W / 3.4 TOPS = 0.59 pJ/op
    energy_per_flop_fp32 = 0.6e-12  # ~0.6 pJ/FLOP (most efficient!)
    energy_per_byte = 20e-12  # USB bandwidth limited

    return HardwareResourceModel(
        name="Coral-Edge-TPU",
        hardware_type=HardwareType.TPU,
        compute_units=1,  # Single systolic array
        threads_per_unit=256,  # Systolic array dimension (estimated)
        warps_per_unit=1,
        warp_size=1,

        precision_profiles={
            # Edge TPU ONLY supports INT8 - no FP32/FP16
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=int8_tops,  # 4 TOPS INT8
                tensor_core_supported=True,  # Systolic array acts like tensor cores
                relative_speedup=1.0,  # Only mode available
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=4e9,  # ~4 GB/s (USB 3.0 or PCIe limited)
        l1_cache_per_unit=512 * 1024,  # ~512 KB on-chip memory (estimated)
        l2_cache_total=0,  # No L2, uses host memory
        main_memory=0,  # Uses host CPU memory
        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=energy_per_byte,
        energy_scaling={
            Precision.INT8: 1.0,  # Base (only mode)
        },
        min_occupancy=1.0,  # Always fully utilized
        max_concurrent_kernels=1,  # Single model at a time
        wave_quantization=1,
    )


def qrb5165_resource_model() -> HardwareResourceModel:
    """
    Qualcomm QRB5165 (Dragonwing) Robotics Platform with Hexagon 698 DSP.

    Based on Snapdragon 865 SoC, optimized for robotics and edge AI applications.

    ARCHITECTURE:
    - Heterogeneous compute: Kryo 585 CPU + Adreno 650 GPU + Hexagon 698 DSP
    - Primary AI accelerator: Hexagon 698 DSP with Tensor Accelerator (HTA)
    - Dataflow-style execution with vector extensions (HVX)
    - Process: 7nm TSMC FinFET

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim: 15 TOPS INT8 (5th gen AI Engine, all engines combined)
    - Hexagon DSP breakdown:
      * Hexagon Vector eXtensions (HVX): Vector/tensor operations
      * Hexagon Tensor Accelerator (HTA): Dedicated tensor core-like units
      * Hexagon Scalar Accelerator: Control/scalar ops
    - Expected effective: ~5-6 TOPS INT8 under sustained 7W operation
    - Root cause: Thermal throttling + memory bandwidth bottleneck

    CPU CONFIGURATION:
    - 1× Kryo 585 Prime (Cortex-A77) @ 2.84 GHz
    - 3× Kryo 585 Gold (Cortex-A77) @ 2.42 GHz
    - 4× Kryo 585 Silver (Cortex-A55) @ 1.81 GHz

    GPU (Adreno 650):
    - ~1.0 TFLOPS FP32
    - OpenCL, Vulkan compute support
    - Useful for graphics + light AI workloads

    DSP (Hexagon 698):
    - 15 TOPS INT8 (combined with HVX + HTA)
    - INT8/INT16 native support
    - FP16 supported but slower
    - 4× HVX 1024-bit vector units

    Power Profile:
    ============

    7W Mode (Typical Robotics Deployment):
    - Sustained DSP clock: ~900 MHz (throttled from peak)
    - Effective INT8: ~6 TOPS (40% of 15 TOPS peak)
    - Use case: Battery-powered robots, drones, edge devices
    - Thermal throttle factor: ~60% (moderate throttling)

    Memory:
    - LPDDR5 @ 2750 MHz (quad-channel)
    - Bandwidth: 44 GB/s
    - Capacity: Up to 16GB

    References:
    - Qualcomm QRB5165 Product Brief (87-28730-1 REV D)
    - Snapdragon 865 specifications
    - Hexagon 698 DSP architecture
    """
    # ========================================================================
    # HEXAGON DSP ARCHITECTURE MODELING
    # ========================================================================
    # Hexagon 698 has:
    # - 4× HVX (Hexagon Vector eXtensions) units: 1024-bit SIMD
    # - Tensor Accelerator (HTA): Dedicated for matrix operations
    # - We model as equivalent "DSP cores" for compatibility

    # 15 TOPS INT8 @ ~1.5 GHz peak
    # → 15e12 ops/sec / 1.5e9 Hz = 10,000 ops/cycle
    # If we model as 32 "DSP processing elements":
    # → 10,000 / 32 = 312.5 ops/cycle/unit

    num_dsp_units = 32  # Equivalent processing elements (HVX + HTA combined)

    # ========================================================================
    # CLOCK DOMAIN - 7W Thermal Envelope
    # ========================================================================
    clock_7w = ClockDomain(
        base_clock_hz=800e6,        # 800 MHz minimum
        max_boost_clock_hz=1.5e9,   # 1.5 GHz peak
        sustained_clock_hz=900e6,   # 900 MHz sustained @ 7W (60% throttle)
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE
    # ========================================================================
    compute_resource_7w = ComputeResource(
        resource_type="Hexagon-698-DSP-HVX-HTA",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 312,    # 312 INT8 ops/cycle/unit (optimized)
            Precision.INT16: 156,   # 156 INT16 ops/cycle/unit (0.5× INT8)
            Precision.FP16: 78,     # 78 FP16 ops/cycle/unit (slower, not native)
            Precision.INT4: 624,    # 624 INT4 ops/cycle/unit (2× INT8, experimental)
        },
        clock_domain=clock_7w,
    )

    # Peak INT8: 32 units × 312 ops/cycle × 1.5 GHz = 14.98 TOPS ≈ 15 TOPS ✓
    # Sustained @ 7W: 32 × 312 × 900 MHz = 8.99 TOPS
    # Effective: 8.99 × 0.60 = 5.4 TOPS (36% of 15 TOPS peak)

    # ========================================================================
    # THERMAL PROFILE (7W Robotics/Edge Deployment)
    # ========================================================================
    thermal_7w = ThermalOperatingPoint(
        name="7W-robotics",
        tdp_watts=7.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.85,  # Good DSP efficiency
                memory_bottleneck_factor=0.70,  # 44 GB/s is limiting for 15 TOPS
                efficiency_factor=0.60,  # 60% effective (realistic for sustained load)
                tile_utilization=0.80,  # Good HVX utilization
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.82,
                memory_bottleneck_factor=0.65,  # More bandwidth per op
                efficiency_factor=0.55,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.70,  # Not as optimized as INT8
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.45,  # Lower efficiency for FP
                tile_utilization=0.70,
                native_acceleration=False,  # Emulated on INT/fixed hardware
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.75,  # Less bandwidth needed
                efficiency_factor=0.65,  # Slightly better than INT8
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    return HardwareResourceModel(
        name="Qualcomm-QRB5165-Hexagon698",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=4,  # HVX units per "processing element"
        warps_per_unit=1,
        warp_size=32,  # Vector lane width (approximation)

        # Thermal operating points
        thermal_operating_points={
            "7W": thermal_7w,
        },
        default_thermal_profile="7W",

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=15e12,  # 15 TOPS INT8
                tensor_core_supported=True,  # HTA acts like tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=7.5e12,  # 7.5 TOPS INT16 (0.5× INT8)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=3.75e12,  # 3.75 TFLOPS FP16 (estimated)
                tensor_core_supported=False,
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
        },
        default_precision=Precision.INT8,

        # ====================================================================
        # MEMORY HIERARCHY - LPDDR5 External Memory
        # ====================================================================
        # QRB5165 uses LPDDR5 (not on-chip like Hailo)
        # - 4 channels × 16-bit × 2750 MHz × 2 (DDR) = 44 GB/s
        # - Hexagon has local caches/scratchpad
        # ====================================================================
        peak_bandwidth=44e9,  # 44 GB/s LPDDR5 @ 2750 MHz
        l1_cache_per_unit=128 * 1024,  # 128 KB L1 per DSP unit (estimated)
        l2_cache_total=4 * 1024 * 1024,  # 4 MB shared L2 (estimated)
        main_memory=16 * 1024**3,  # Up to 16 GB LPDDR5

        # Energy (edge-optimized)
        energy_per_flop_fp32=1.5e-12,  # 1.5 pJ/FLOP (FP32 baseline)
        energy_per_byte=15e-12,  # 15 pJ/byte (LPDDR5)
        energy_scaling={
            Precision.INT8: 0.15,   # 15% of FP32 energy
            Precision.INT16: 0.25,  # 25% of FP32 energy
            Precision.FP16: 0.50,   # 50% of FP32 energy
            Precision.INT4: 0.08,   # 8% of FP32 energy
        },

        # Scheduling (DSP has sophisticated scheduling)
        min_occupancy=0.60,
        max_concurrent_kernels=8,  # Can run multiple layers concurrently
        wave_quantization=4,  # Process in groups of 4 units
    )


def ti_tda4vm_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4VM (Jacinto 7) Automotive ADAS Processor.

    Based on TI's Jacinto 7 family for automotive advanced driver assistance systems.

    ARCHITECTURE:
    - Heterogeneous compute: Cortex-A72 CPU + C7x DSP + MMA
    - Primary AI accelerator: C7x DSP with Matrix Multiply Accelerator (MMA)
    - Automotive-grade: ASIL-D/SIL-3 safety certification
    - Process: 16nm FinFET (automotive qualified)
    - Temperature: -40°C to 125°C (automotive grade AEC-Q100)

    CRITICAL REALITY CHECK - Performance Specifications:
    - C7x DSP: 80 GFLOPS FP32, 256 GOPS INT16 @ 1.0 GHz
    - MMA (Matrix Multiply Accelerator): 8 TOPS INT8 @ 1.0 GHz
    - Expected effective: ~4-5 TOPS INT8 under sustained 10W operation
    - Root cause: Automotive thermal constraints + memory bandwidth

    CPU CONFIGURATION:
    - 2× Cortex-A72 @ 2.0 GHz (application processing)
    - R5F safety cores for ASIL-D compliance

    DSP (C7x):
    - 1.0 GHz peak clock
    - 80 GFLOPS FP32
    - 256 GOPS INT16
    - Vector processing: 512-bit SIMD
    - L1D: 48 KB (32 KB cache + 16 KB SRAM)

    MMA (Matrix Multiply Accelerator):
    - 8 TOPS INT8 @ 1.0 GHz
    - Integrated with C7x DSP
    - INT8/INT16 native support
    - Optimized for CNNs

    Power Profiles:
    ==============

    10W Mode (Typical Front Camera ADAS):
    - Sustained DSP clock: ~850 MHz (85% of 1.0 GHz)
    - Effective INT8: ~5 TOPS (62% of 8 TOPS peak)
    - Use case: Front camera, lane detection, object detection
    - Thermal: Automotive passive cooling

    20W Mode (Full ADAS System):
    - Sustained DSP clock: ~950 MHz (95% of 1.0 GHz)
    - Effective INT8: ~6.5 TOPS (81% of 8 TOPS peak)
    - Use case: Multi-camera (4-6 cameras), radar fusion, parking assist
    - Thermal: Active cooling in vehicle

    Memory:
    - LPDDR4x @ 3733 MT/s (dual-channel)
    - Bandwidth: ~60 GB/s
    - Capacity: Up to 8GB
    - MSMC: 8 MB on-chip SRAM for DSP

    References:
    - TI TDA4VM Product Brief
    - Jacinto 7 Architecture Overview
    - C7x DSP Core Manual
    - Automotive ADAS specifications
    """
    # ========================================================================
    # C7x DSP + MMA ARCHITECTURE MODELING
    # ========================================================================
    # C7x has:
    # - Vector DSP core: 512-bit SIMD, 1.0 GHz
    # - Matrix Multiply Accelerator (MMA): Dedicated for matrix ops
    # - We model as equivalent "DSP processing elements"

    # 8 TOPS INT8 @ 1.0 GHz
    # → 8e12 ops/sec / 1.0e9 Hz = 8,000 ops/cycle
    # If we model as 32 "DSP processing elements":
    # → 8,000 / 32 = 250 ops/cycle/unit

    num_dsp_units = 32  # Equivalent processing elements (C7x + MMA combined)

    # ========================================================================
    # CLOCK DOMAIN - 10W Automotive Thermal Envelope
    # ========================================================================
    clock_10w = ClockDomain(
        base_clock_hz=600e6,        # 600 MHz minimum
        max_boost_clock_hz=1.0e9,   # 1.0 GHz peak
        sustained_clock_hz=850e6,   # 850 MHz sustained @ 10W (85% of peak)
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE - 10W Profile
    # ========================================================================
    compute_resource_10w = ComputeResource(
        resource_type="TI-C7x-DSP-MMA",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 250,    # 250 INT8 ops/cycle/unit (MMA optimized)
            Precision.INT16: 125,   # 125 INT16 ops/cycle/unit (0.5× INT8)
            Precision.FP16: 62,     # 62 FP16 ops/cycle/unit (slower)
            Precision.FP32: 31,     # 31 FP32 ops/cycle/unit (C7x baseline)
        },
        clock_domain=clock_10w,
    )

    # Peak INT8: 32 units × 250 ops/cycle × 1.0 GHz = 8.0 TOPS ✓
    # Sustained @ 10W: 32 × 250 × 850 MHz = 6.8 TOPS
    # Effective: 6.8 × 0.70 = 4.76 TOPS (60% of 8 TOPS peak)

    # ========================================================================
    # THERMAL PROFILE (10W Front Camera ADAS)
    # ========================================================================
    thermal_10w = ThermalOperatingPoint(
        name="10W-front-camera-ADAS",
        tdp_watts=10.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.90,  # Automotive optimized
                memory_bottleneck_factor=0.75,  # 60 GB/s for 8 TOPS
                efficiency_factor=0.70,  # 70% effective (conservative for automotive)
                tile_utilization=0.85,  # Good MMA utilization
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.75,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.55,
                tile_utilization=0.75,
                native_acceleration=False,  # Emulated via C7x
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.50,
                tile_utilization=0.70,
                native_acceleration=True,  # C7x native FP32
            ),
        }
    )

    # ========================================================================
    # CLOCK DOMAIN - 20W Full ADAS System
    # ========================================================================
    clock_20w = ClockDomain(
        base_clock_hz=700e6,        # 700 MHz minimum
        max_boost_clock_hz=1.0e9,   # 1.0 GHz peak
        sustained_clock_hz=950e6,   # 950 MHz sustained @ 20W (95% of peak)
        dvfs_enabled=True,
    )

    compute_resource_20w = ComputeResource(
        resource_type="TI-C7x-DSP-MMA",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 250,
            Precision.INT16: 125,
            Precision.FP16: 62,
            Precision.FP32: 31,
        },
        clock_domain=clock_20w,
    )

    # Sustained @ 20W: 32 × 250 × 950 MHz = 7.6 TOPS
    # Effective: 7.6 × 0.80 = 6.08 TOPS (76% of 8 TOPS peak)

    thermal_20w = ThermalOperatingPoint(
        name="20W-full-ADAS-system",
        tdp_watts=20.0,
        cooling_solution="automotive-active",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.92,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.80,  # Better at higher power
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.90,
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.75,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=False,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.60,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    return HardwareResourceModel(
        name="TI-TDA4VM-C7x-DSP",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=4,  # Vector lanes per processing element
        warps_per_unit=1,
        warp_size=16,  # SIMD width approximation

        # Thermal operating points
        thermal_operating_points={
            "10W": thermal_10w,   # Front camera ADAS
            "20W": thermal_20w,   # Full multi-camera system
        },
        default_thermal_profile="10W",  # Most common automotive deployment

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=8e12,  # 8 TOPS INT8
                tensor_core_supported=True,  # MMA acts like tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=4e12,  # 4 TOPS INT16 (0.5× INT8)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=80e9,  # 80 GFLOPS FP32
                tensor_core_supported=False,  # C7x vector, not MMA
                relative_speedup=0.01,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        # ====================================================================
        # MEMORY HIERARCHY - LPDDR4x External Memory
        # ====================================================================
        # TDA4VM uses LPDDR4x (automotive grade)
        # - Dual channel × 32-bit × 3733 MT/s = ~60 GB/s
        # - MSMC: 8 MB on-chip SRAM dedicated to C7x DSP
        # ====================================================================
        peak_bandwidth=60e9,  # 60 GB/s LPDDR4x @ 3733 MT/s
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per C7x (32 KB cache + 16 KB SRAM)
        l2_cache_total=8 * 1024 * 1024,  # 8 MB MSMC SRAM
        main_memory=8 * 1024**3,  # Up to 8 GB LPDDR4x

        # Energy (automotive-optimized, conservative)
        energy_per_flop_fp32=2.0e-12,  # 2.0 pJ/FLOP (automotive grade, higher than mobile)
        energy_per_byte=20e-12,  # 20 pJ/byte (LPDDR4x automotive)
        energy_scaling={
            Precision.INT8: 0.15,   # 15% of FP32 energy
            Precision.INT16: 0.25,  # 25% of FP32 energy
            Precision.FP16: 0.50,   # 50% of FP32 energy
            Precision.FP32: 1.0,    # Baseline
        },

        # Scheduling (automotive deterministic scheduling)
        min_occupancy=0.70,  # Automotive requires high utilization
        max_concurrent_kernels=4,  # Limited for determinism
        wave_quantization=4,
    )


def ti_tda4vl_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4VL (Jacinto 7 Entry-Level) Automotive ADAS Processor.

    ARCHITECTURE:
    - Entry-level ADAS: Lower cost, lower performance than TDA4VM
    - Heterogeneous: Cortex-A72 + C7x DSP + MMAv2 (newer generation)
    - Automotive-grade: ASIL-B/C (lower safety level than TDA4VM)
    - Temperature: -40°C to 125°C

    KEY DIFFERENCES FROM TDA4VM:
    - Half the AI performance: 4 TOPS INT8 vs 8 TOPS
    - Lower CPU frequency: A72 @ 1.2 GHz vs 2.0 GHz
    - Newer MMAv2 architecture (more efficient than MMAv1 in TDA4VM)
    - Lower power envelope: 7-12W typical

    PERFORMANCE:
    - MMAv2: 4 TOPS INT8 @ 1.0 GHz (half of TDA4VM)
    - C7x DSP: 40 GFLOPS FP32 @ 1.0 GHz
    - Expected effective: ~2-3 TOPS INT8 under sustained operation

    CPU:
    - 2× Cortex-A72 @ 1.2 GHz
    - R5F safety cores for ASIL-B/C

    MEMORY:
    - LPDDR4x @ 3733 MT/s
    - Bandwidth: ~60 GB/s
    - Capacity: Up to 4GB

    Power Profiles:
    - 7W Mode: Entry-level ADAS (single camera, lane detection)
    - 12W Mode: Multi-function ADAS (front camera + side cameras)

    USE CASES:
    - Entry-level ADAS (Lane Keep, TSR, basic ACC)
    - Cost-sensitive automotive markets
    - Single front-facing camera systems
    """
    num_dsp_units = 16  # Half of TDA4VM (4 TOPS vs 8 TOPS)

    clock_7w = ClockDomain(
        base_clock_hz=500e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=750e6,  # 75% sustained @ 7W
        dvfs_enabled=True,
    )

    compute_resource_7w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 250,
            Precision.INT16: 125,
            Precision.FP16: 62,
            Precision.FP32: 31,
        },
        clock_domain=clock_7w,
    )

    thermal_7w = ThermalOperatingPoint(
        name="7W-entry-ADAS",
        tdp_watts=7.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.92,  # MMAv2 more efficient
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.72,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    clock_12w = ClockDomain(
        base_clock_hz=700e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=900e6,
        dvfs_enabled=True,
    )

    compute_resource_12w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_12w,
    )

    thermal_12w = ThermalOperatingPoint(
        name="12W-multi-function-ADAS",
        tdp_watts=12.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_12w,
                instruction_efficiency=0.93,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.78,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="TI-TDA4VL-C7x-MMAv2",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=250,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=4.0e12,  # 4 TOPS INT8
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=40e9,  # 40 GFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=0.01,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=60e9,
        l1_cache_per_unit=48 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=4 * 1024**3,
        energy_per_flop_fp32=2.0e-12,
        energy_per_byte=20e-12,
        energy_scaling={Precision.INT8: 0.15, Precision.INT16: 0.25, Precision.FP16: 0.50, Precision.FP32: 1.0},
        min_occupancy=0.70,
        max_concurrent_kernels=4,
        wave_quantization=4,
        thermal_operating_points={"7W": thermal_7w, "12W": thermal_12w},
        default_thermal_profile="7W",
    )


def ti_tda4al_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4AL (Jacinto 7 Advanced Low-Power) Automotive ADAS Processor.

    ARCHITECTURE:
    - Mid-range ADAS: Similar AI performance to TDA4VM but newer architecture
    - Heterogeneous: Dual A72 @ 2.0 GHz + C7x DSP + MMAv2
    - Automotive-grade: ASIL-D/SIL-3
    - Process: Newer node, better power efficiency than TDA4VM

    KEY DIFFERENCES FROM TDA4VM:
    - Same AI performance: 8 TOPS INT8
    - MMAv2 architecture (more efficient than MMAv1)
    - Higher CPU frequency: A72 @ 2.0 GHz (vs TDA4VM's 2.0 GHz)
    - Better power efficiency: 10-18W range

    PERFORMANCE:
    - MMAv2: 8 TOPS INT8 @ 1.0 GHz
    - C7x DSP: 80 GFLOPS FP32 @ 1.0 GHz
    - Expected effective: ~5-6 TOPS INT8 sustained

    MEMORY:
    - LPDDR4x @ 3733 MT/s
    - Bandwidth: ~60 GB/s
    - Capacity: Up to 8GB

    Power Profiles:
    - 10W Mode: Front camera ADAS
    - 18W Mode: Multi-camera ADAS (better than TDA4VM @ 20W due to MMAv2)

    USE CASES:
    - ADAS Level 2-3 (similar to TDA4VM but more efficient)
    - Replaces TDA4VM in newer designs
    """
    num_dsp_units = 32  # Same as TDA4VM

    clock_10w = ClockDomain(base_clock_hz=600e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=880e6, dvfs_enabled=True)
    compute_resource_10w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_10w,
    )
    thermal_10w = ThermalOperatingPoint(
        name="10W-front-camera-ADAS",
        tdp_watts=10.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.94,  # MMAv2 improvement
                memory_bottleneck_factor=0.78,
                efficiency_factor=0.75,  # Better than TDA4VM's 0.70
                tile_utilization=0.88,
                native_acceleration=True,
            ),
        }
    )

    clock_18w = ClockDomain(base_clock_hz=700e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=980e6, dvfs_enabled=True)
    compute_resource_18w = ComputeResource(
        resource_type="TI-C7x-DSP-MMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_18w,
    )
    thermal_18w = ThermalOperatingPoint(
        name="18W-multi-camera-ADAS",
        tdp_watts=18.0,
        cooling_solution="automotive-active",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_18w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.82,
                efficiency_factor=0.82,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="TI-TDA4AL-C7x-MMAv2",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=250,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.INT8: PrecisionProfile(precision=Precision.INT8, peak_ops_per_sec=8.0e12, tensor_core_supported=False, relative_speedup=1.0, bytes_per_element=1, accumulator_precision=Precision.INT32),
            Precision.FP32: PrecisionProfile(precision=Precision.FP32, peak_ops_per_sec=80e9, tensor_core_supported=False, relative_speedup=0.01, bytes_per_element=4),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=60e9,
        l1_cache_per_unit=48 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=8 * 1024**3,
        energy_per_flop_fp32=1.8e-12,  # 10% better than TDA4VM
        energy_per_byte=18e-12,
        energy_scaling={Precision.INT8: 0.15, Precision.INT16: 0.25, Precision.FP16: 0.50, Precision.FP32: 1.0},
        min_occupancy=0.70,
        max_concurrent_kernels=4,
        wave_quantization=4,
        thermal_operating_points={"10W": thermal_10w, "18W": thermal_18w},
        default_thermal_profile="10W",
    )


def ti_tda4vh_resource_model() -> HardwareResourceModel:
    """
    Texas Instruments TDA4VH (Jacinto 7 Very High Performance) Automotive ADAS Processor.

    ARCHITECTURE:
    - High-performance ADAS for Level 3-4 autonomous driving
    - 8× Cortex-A72 @ 2.0 GHz (vs 2× in TDA4VM)
    - 4× C7x DSP cores @ 1.0 GHz (vs 1× in TDA4VM)
    - 4× MMAv2 accelerators (vs 1× MMAv1 in TDA4VM)
    - Automotive-grade: ASIL-D/SIL-3

    PERFORMANCE:
    - 4× MMAv2: 32 TOPS INT8 @ 1.0 GHz (4× TDA4VM)
    - 4× C7x DSP: 320 GFLOPS FP32
    - Expected effective: ~20-25 TOPS INT8 sustained

    CPU:
    - 8× Cortex-A72 @ 2.0 GHz
    - Multiple R5F cores for safety

    MEMORY:
    - LPDDR5 @ 6400 MT/s
    - Bandwidth: ~100 GB/s (higher than TDA4VM)
    - Capacity: Up to 16GB

    Power Profiles:
    - 20W Mode: Multi-camera Level 2+ ADAS
    - 35W Mode: Full Level 3-4 autonomy stack

    USE CASES:
    - Advanced ADAS Level 3-4
    - 8-12 camera surround view
    - Lidar + radar + camera fusion
    - Highway pilot, urban pilot
    """
    num_dsp_units = 128  # 4× TDA4VM (4× MMAv2)

    clock_20w = ClockDomain(base_clock_hz=700e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=850e6, dvfs_enabled=True)
    compute_resource_20w = ComputeResource(
        resource_type="TI-4xC7x-DSP-4xMMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_20w,
    )
    thermal_20w = ThermalOperatingPoint(
        name="20W-multi-camera-L2+-ADAS",
        tdp_watts=20.0,
        cooling_solution="automotive-active",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_20w,
                instruction_efficiency=0.93,
                memory_bottleneck_factor=0.75,
                efficiency_factor=0.70,  # Lower due to multi-accelerator coordination
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    clock_35w = ClockDomain(base_clock_hz=800e6, max_boost_clock_hz=1.0e9, sustained_clock_hz=950e6, dvfs_enabled=True)
    compute_resource_35w = ComputeResource(
        resource_type="TI-4xC7x-DSP-4xMMAv2",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={Precision.INT8: 250, Precision.INT16: 125, Precision.FP16: 62, Precision.FP32: 31},
        clock_domain=clock_35w,
    )
    thermal_35w = ThermalOperatingPoint(
        name="35W-full-L3-4-autonomy",
        tdp_watts=35.0,
        cooling_solution="automotive-active-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_35w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.78,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="TI-TDA4VH-4xC7x-4xMMAv2",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=250,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.INT8: PrecisionProfile(precision=Precision.INT8, peak_ops_per_sec=32.0e12, tensor_core_supported=False, relative_speedup=1.0, bytes_per_element=1, accumulator_precision=Precision.INT32),
            Precision.FP32: PrecisionProfile(precision=Precision.FP32, peak_ops_per_sec=320e9, tensor_core_supported=False, relative_speedup=0.01, bytes_per_element=4),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=100e9,  # LPDDR5 @ 6400 MT/s
        l1_cache_per_unit=48 * 1024,
        l2_cache_total=16 * 1024 * 1024,  # Larger cache for multiple accelerators
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=1.8e-12,
        energy_per_byte=15e-12,  # LPDDR5 more efficient
        energy_scaling={Precision.INT8: 0.15, Precision.INT16: 0.25, Precision.FP16: 0.50, Precision.FP32: 1.0},
        min_occupancy=0.60,  # Lower due to multi-accelerator complexity
        max_concurrent_kernels=8,  # 4× accelerators allow more parallelism
        wave_quantization=8,
        thermal_operating_points={"20W": thermal_20w, "35W": thermal_35w},
        default_thermal_profile="20W",
    )


# ============================================================================
# CEVA NeuPro Neural Processing IP
# ============================================================================

def ceva_neupro_npm11_resource_model() -> HardwareResourceModel:
    """
    CEVA NeuPro-M NPM11 Neural Processing IP Core.

    ARCHITECTURE:
    - Licensable NPU IP core for edge AI acceleration
    - Single NeuPro-M engine configuration
    - Heterogeneous: Tensor + Vector + Scalar units
    - Designed for SoC integration (mobile, automotive, IoT)

    PERFORMANCE:
    - Peak: 20 TOPS INT8 @ 1.25 GHz
    - Scalable architecture (2-256 TOPS range)
    - Optimized for CNNs, RNNs, and transformer models

    PRECISION SUPPORT:
    - INT8: Native, primary mode
    - INT16: Native support
    - FP16: Supported
    - INT4: Supported (2× INT8 throughput)

    MEMORY:
    - Configurable local memory (SRAM)
    - External DRAM access via SoC interconnect
    - Typical: 2-4 MB local SRAM
    - Bandwidth depends on SoC integration

    POWER:
    - 2W typical for NPM11 @ 1.0 GHz
    - ~10 TOPS/W efficiency
    - Highly power-efficient for edge AI

    USE CASES:
    - Mobile devices (always-on AI)
    - Automotive ADAS (sensor fusion)
    - IoT devices (edge AI)
    - Smart cameras and drones

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on CEVA published specs
    - Need empirical benchmarking on actual silicon implementations
    - Performance varies based on SoC integration and memory configuration

    REFERENCES:
    - CEVA NeuPro-M Product Brief
    - NPM11 configuration specifications
    - CEVA press releases (2021-2024)
    """
    # Model as 64 equivalent processing elements
    # 20 TOPS @ 1.25 GHz → 16e12 ops/sec → 250 ops/cycle → 64 units × 312.5 ops/unit/cycle
    num_npu_units = 64

    clock = ClockDomain(
        base_clock_hz=800e6,       # 800 MHz minimum
        max_boost_clock_hz=1.25e9, # 1.25 GHz peak
        sustained_clock_hz=1.0e9,  # 1.0 GHz sustained
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="CEVA-NeuPro-M-NPM11",
        num_units=num_npu_units,
        ops_per_unit_per_clock={
            Precision.INT8: 312,   # 312 INT8 MACs/cycle/unit
            Precision.INT16: 156,  # 156 INT16 MACs/cycle/unit
            Precision.FP16: 156,   # 156 FP16 MACs/cycle/unit
            Precision.INT4: 624,   # 624 INT4 MACs/cycle/unit (2× INT8)
        },
        clock_domain=clock,
    )

    # Peak INT8: 64 units × 312 ops/cycle × 1.25 GHz = 25 TOPS (conservative vs 20 TOPS spec)
    # This accounts for realistic efficiency

    thermal_2w = ThermalOperatingPoint(
        name="2W-edge-ai",
        tdp_watts=2.0,
        cooling_solution="passive-mobile",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.90,  # High NPU efficiency
                memory_bottleneck_factor=0.75,  # Depends on SoC integration
                efficiency_factor=0.70,  # 70% effective utilization
                tile_utilization=0.85,  # Good tensor utilization
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.60,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource,
                instruction_efficiency=0.92,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.75,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="CEVA-NeuPro-M-NPM11",
        hardware_type=HardwareType.DSP,
        compute_units=num_npu_units,
        threads_per_unit=4,
        warps_per_unit=1,
        warp_size=32,

        thermal_operating_points={
            "2W": thermal_2w,
        },
        default_thermal_profile="2W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=20e12,  # 20 TOPS INT8
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=10e12,  # 10 TOPS INT16
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=10e12,  # 10 TFLOPS FP16
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
        },
        default_precision=Precision.INT8,

        # Memory (depends on SoC integration, typical values)
        peak_bandwidth=50e9,  # 50 GB/s (typical for mobile SoC integration)
        l1_cache_per_unit=64 * 1024,  # 64 KB per unit
        l2_cache_total=2 * 1024 * 1024,  # 2 MB shared cache
        main_memory=8 * 1024**3,  # Up to 8 GB

        # Energy (edge-optimized)
        energy_per_flop_fp32=1.2e-12,  # 1.2 pJ/FLOP
        energy_per_byte=12e-12,  # 12 pJ/byte
        energy_scaling={
            Precision.INT8: 0.12,
            Precision.INT16: 0.20,
            Precision.FP16: 0.40,
            Precision.INT4: 0.06,
        },

        min_occupancy=0.70,
        max_concurrent_kernels=8,
        wave_quantization=4,
    )


# ============================================================================
# Cadence Tensilica Vision DSP IP
# ============================================================================

def cadence_vision_q8_resource_model() -> HardwareResourceModel:
    """
    Cadence Tensilica Vision Q8 DSP IP Core (7th Generation).

    ARCHITECTURE:
    - Licensable vision DSP IP core for SoC integration
    - 7th generation Tensilica Vision DSP (flagship)
    - 1024-bit SIMD engine for vision processing
    - Heterogeneous: Vector + Scalar units

    PERFORMANCE:
    - Peak: 3.8 TOPS (INT8/INT16)
    - 129 GFLOPS FP32
    - 2× performance of Vision Q7 DSP

    PRECISION SUPPORT:
    - INT8/INT16: Native, optimized for vision
    - FP32: 129 GFLOPS
    - FP16: Supported

    MEMORY:
    - Configurable local memory
    - External memory via SoC interconnect
    - Typical: 512 KB - 2 MB local SRAM

    POWER:
    - 0.5-1W typical @ 1.0 GHz
    - ~3-7 TOPS/W efficiency
    - Power-efficient for always-on vision

    USE CASES:
    - Automotive vision (ADAS cameras)
    - Mobile device cameras (ISP + AI)
    - Surveillance cameras
    - AR/VR vision processing

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Cadence published specs
    - Need empirical benchmarking on actual silicon

    REFERENCES:
    - Cadence Tensilica Vision Q8 Product Brief (2021)
    - Tensilica Vision DSP Family specifications
    """
    # Model as 32 equivalent processing elements
    # 3.8 TOPS @ 1.0 GHz → 3.8e12 ops/sec → 118.75 ops/cycle → 32 units × 118.75 ops/unit/cycle
    num_dsp_units = 32

    clock = ClockDomain(
        base_clock_hz=600e6,      # 600 MHz minimum
        max_boost_clock_hz=1.2e9, # 1.2 GHz max
        sustained_clock_hz=1.0e9, # 1.0 GHz sustained
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="Cadence-Tensilica-Vision-Q8",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 119,   # ~119 INT8 ops/cycle/unit
            Precision.INT16: 119,  # Same for INT16 (vision optimized)
            Precision.FP32: 4,     # 129 GFLOPS / 32 units / 1.0 GHz
            Precision.FP16: 8,     # 2× FP32
        },
        clock_domain=clock,
    )

    thermal_1w = ThermalOperatingPoint(
        name="1W-vision",
        tdp_watts=1.0,
        cooling_solution="passive-mobile",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,  # Good vision DSP efficiency
                memory_bottleneck_factor=0.70,  # Vision workloads are bandwidth-sensitive
                efficiency_factor=0.65,  # 65% effective
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.60,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Cadence-Tensilica-Vision-Q8",
        hardware_type=HardwareType.DSP,
        compute_units=num_dsp_units,
        threads_per_unit=4,
        warps_per_unit=1,
        warp_size=32,

        thermal_operating_points={
            "1W": thermal_1w,
        },
        default_thermal_profile="1W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=3.8e12,  # 3.8 TOPS
                tensor_core_supported=False,  # Vector DSP, not tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=3.8e12,  # 3.8 TOPS (same as INT8)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=129e9,  # 129 GFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=0.034,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        # Memory
        peak_bandwidth=40e9,  # 40 GB/s (typical SoC integration)
        l1_cache_per_unit=32 * 1024,  # 32 KB per unit
        l2_cache_total=1 * 1024 * 1024,  # 1 MB shared cache
        main_memory=4 * 1024**3,  # Up to 4 GB

        # Energy
        energy_per_flop_fp32=1.5e-12,  # 1.5 pJ/FLOP
        energy_per_byte=12e-12,  # 12 pJ/byte
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.INT16: 0.15,
            Precision.FP32: 1.0,
            Precision.FP16: 0.50,
        },

        min_occupancy=0.70,
        max_concurrent_kernels=4,
        wave_quantization=4,
    )


# ============================================================================
# Synopsys ARC EV Embedded Vision Processor IP
# ============================================================================

def synopsys_arc_ev7x_resource_model() -> HardwareResourceModel:
    """
    Synopsys ARC EV7x Embedded Vision Processor IP Core.

    ARCHITECTURE:
    - Licensable embedded vision processor IP for SoC integration
    - Heterogeneous: 1-4 Vector Processing Units (VPUs) + DNN accelerator
    - Each VPU: 512-bit wide vector DSP
    - DNN accelerator: 880-14,080 MACs (scalable)
    - ARCv2 RISC ISA base

    PERFORMANCE:
    - Peak: Up to 35 TOPS INT8 @ 16nm FinFET
    - 4× performance of ARC EV6x
    - Configurable 1-4 core

    PRECISION SUPPORT:
    - INT8: Native via DNN accelerator (primary mode)
    - INT16: Native via VPUs
    - INT32: Supported
    - FP32: Via VPU FPU

    MEMORY:
    - Configurable local memory
    - External memory via AXI interconnect
    - Typical: 2-8 MB local memory

    POWER:
    - 3-5W typical for full EV7x @ 1.0 GHz
    - ~7-10 TOPS/W efficiency
    - Automotive-grade power management

    USE CASES:
    - Automotive ADAS (camera processing)
    - Surveillance systems
    - Drone vision
    - AR/VR applications

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Synopsys published specs
    - Need empirical benchmarking on actual silicon

    REFERENCES:
    - Synopsys ARC EV7x Product Brief (2019)
    - EE Times coverage (2019)
    - Synopsys DesignWare IP catalog
    """
    # Model 4-core configuration with full DNN accelerator
    # 35 TOPS @ 1.0 GHz → 35e12 ops/sec → 35,000 ops/cycle
    # Model as 128 equivalent processing elements: 35,000 / 128 = 273 ops/cycle/unit
    num_ev_units = 128

    clock = ClockDomain(
        base_clock_hz=600e6,      # 600 MHz minimum
        max_boost_clock_hz=1.2e9, # 1.2 GHz max
        sustained_clock_hz=1.0e9, # 1.0 GHz sustained
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="Synopsys-ARC-EV7x-4core",
        num_units=num_ev_units,
        ops_per_unit_per_clock={
            Precision.INT8: 273,   # 273 INT8 MACs/cycle/unit
            Precision.INT16: 136,  # 136 INT16 MACs/cycle/unit
            Precision.INT32: 68,   # 68 INT32 MACs/cycle/unit
            Precision.FP32: 17,    # ~2.2 GFLOPS per core × 4 cores
        },
        clock_domain=clock,
    )

    # Peak INT8: 128 units × 273 ops/cycle × 1.0 GHz = 34.94 TOPS ≈ 35 TOPS ✓

    thermal_5w = ThermalOperatingPoint(
        name="5W-automotive-vision",
        tdp_watts=5.0,
        cooling_solution="automotive-passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.90,  # Efficient DNN accelerator
                memory_bottleneck_factor=0.72,  # Automotive workloads
                efficiency_factor=0.68,  # 68% effective
                tile_utilization=0.82,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.60,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="Synopsys-ARC-EV7x-4core",
        hardware_type=HardwareType.DSP,
        compute_units=num_ev_units,
        threads_per_unit=4,
        warps_per_unit=1,
        warp_size=32,

        thermal_operating_points={
            "5W": thermal_5w,
        },
        default_thermal_profile="5W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=35e12,  # 35 TOPS INT8
                tensor_core_supported=True,  # DNN accelerator
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=17.5e12,  # 17.5 TOPS INT16
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=8.8e9,  # ~8.8 GFLOPS FP32 (4 cores × 2.2 GFLOPS)
                tensor_core_supported=False,
                relative_speedup=0.00025,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        # Memory
        peak_bandwidth=60e9,  # 60 GB/s (automotive SoC integration)
        l1_cache_per_unit=32 * 1024,  # 32 KB per unit
        l2_cache_total=4 * 1024 * 1024,  # 4 MB shared cache
        main_memory=8 * 1024**3,  # Up to 8 GB

        # Energy (automotive-optimized)
        energy_per_flop_fp32=1.4e-12,  # 1.4 pJ/FLOP
        energy_per_byte=14e-12,  # 14 pJ/byte
        energy_scaling={
            Precision.INT8: 0.14,
            Precision.INT16: 0.22,
            Precision.FP32: 1.0,
            Precision.INT4: 0.07,
        },

        min_occupancy=0.70,
        max_concurrent_kernels=8,  # 4 VPUs allow good parallelism
        wave_quantization=4,
    )


# ============================================================================
# ARM Mali GPU IP
# ============================================================================

def arm_mali_g78_mp20_resource_model() -> HardwareResourceModel:
    """
    ARM Mali-G78 MP20 GPU IP Core.

    ARCHITECTURE:
    - Licensable mobile GPU IP core for SoC integration
    - 2nd generation Valhall architecture
    - 20 shader cores (MP20 configuration, max 24 cores)
    - Unified shader architecture (compute + graphics)
    - Warp width: 16 threads

    PERFORMANCE:
    - Graphics: ~1.94 TFLOPS FP32 @ 848 MHz (~97 GFLOPS per core)
    - Compute: ~2 TOPS INT8 (estimated, not optimized for AI)
    - FP16: ~3.88 TFLOPS (2× FP32)
    - Primarily a graphics GPU with compute capabilities

    PRECISION SUPPORT:
    - FP32: Native, primary mode for graphics
    - FP16: Native, 2× throughput vs FP32
    - INT8: Supported but not optimized (no tensor cores)
    - Note: Mali GPUs are graphics-focused, not AI-optimized

    MEMORY:
    - Configurable L2 cache (512 KB - 2 MB typical)
    - External memory via SoC interconnect
    - Bandwidth depends on SoC integration (20-50 GB/s typical)

    POWER:
    - 3-5W typical TDP @ 848 MHz
    - Power-efficient for mobile graphics
    - DVFS for dynamic power management

    USE CASES:
    - Mobile gaming (flagship smartphones)
    - Computational photography
    - Light AI inference (alongside dedicated NPU)
    - AR/VR applications
    - UI rendering and composition

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on ARM published specs and Google Tensor SoC
    - Used in Google Tensor (Pixel 6/6 Pro)
    - Graphics-optimized, not AI-optimized
    - For serious AI workloads, pair with dedicated NPU

    REFERENCES:
    - ARM Mali-G78 Product Brief (2020)
    - Google Tensor SoC specifications
    - AnandTech Mali-G78 analysis
    - Typical configuration: 848 MHz, 20 cores
    """
    # Model as 20 shader cores
    # Each core: 97 GFLOPS FP32 @ 848 MHz
    # Total: 1.94 TFLOPS FP32
    num_cores = 20

    clock = ClockDomain(
        base_clock_hz=400e6,      # 400 MHz minimum
        max_boost_clock_hz=950e6, # 950 MHz max
        sustained_clock_hz=848e6, # 848 MHz typical (Google Tensor)
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="ARM-Mali-G78-MP20",
        num_units=num_cores,
        ops_per_unit_per_clock={
            # Each core: 97 GFLOPS @ 848 MHz → 114 ops/cycle
            Precision.FP32: 114,   # 114 FP32 ops/cycle/core
            Precision.FP16: 228,   # 228 FP16 ops/cycle/core (2× FP32)
            Precision.INT8: 114,   # ~114 INT8 ops/cycle/core (not optimized)
            Precision.INT16: 114,  # Similar to FP16 throughput
        },
        clock_domain=clock,
    )

    # Peak FP32: 20 cores × 114 ops/cycle × 848 MHz = 1.93 TFLOPS ✓

    thermal_5w = ThermalOperatingPoint(
        name="5W-mobile-gaming",
        tdp_watts=5.0,
        cooling_solution="passive-mobile",
        performance_specs={
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,  # Graphics GPU efficiency
                memory_bottleneck_factor=0.65,  # Mobile bandwidth constraints
                efficiency_factor=0.60,  # Graphics workload optimized
                tile_utilization=0.75,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.70,  # Not optimized for INT8 AI
                memory_bottleneck_factor=0.65,
                efficiency_factor=0.50,  # Lower for AI workloads
                tile_utilization=0.70,
                native_acceleration=False,  # No tensor cores
            ),
        }
    )

    return HardwareResourceModel(
        name="ARM-Mali-G78-MP20",
        hardware_type=HardwareType.GPU,
        compute_units=num_cores,
        threads_per_unit=256,  # Warp size 16 × 16 execution lanes
        warps_per_unit=16,
        warp_size=16,

        thermal_operating_points={
            "5W": thermal_5w,
        },
        default_thermal_profile="5W",

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1.94e12,  # 1.94 TFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=3.88e12,  # 3.88 TFLOPS FP16
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=1.94e12,  # ~2 TOPS INT8 (not optimized)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.FP16,  # Mobile GPU default

        # Memory (typical mobile SoC integration)
        peak_bandwidth=40e9,  # 40 GB/s (typical for mobile SoC)
        l1_cache_per_unit=32 * 1024,  # 32 KB per core
        l2_cache_total=2 * 1024 * 1024,  # 2 MB shared L2
        main_memory=8 * 1024**3,  # Up to 8 GB

        # Energy (mobile-optimized)
        energy_per_flop_fp32=2.0e-12,  # 2.0 pJ/FLOP
        energy_per_byte=15e-12,  # 15 pJ/byte (mobile DRAM)
        energy_scaling={
            Precision.INT8: 0.20,
            Precision.INT16: 0.30,
            Precision.FP16: 0.50,
            Precision.FP32: 1.0,
        },

        min_occupancy=0.60,
        max_concurrent_kernels=32,  # High parallelism for graphics
        wave_quantization=16,  # Process in groups of 16 threads (warp size)
    )
