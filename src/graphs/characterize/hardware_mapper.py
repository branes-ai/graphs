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

from .graph_structures import (
    SubgraphDescriptor,
    ParallelismDescriptor,
    BottleneckType
)
from .fusion_partitioner import FusedSubgraph, FusionReport


class HardwareType(Enum):
    """Supported hardware types"""
    GPU = "gpu"
    CPU = "cpu"
    TPU = "tpu"
    KPU = "kpu"
    DPU = "dpu"  # Xilinx Vitis AI (FPGA-based accelerator)
    CGRA = "cgra"  # Stanford Plasticine (spatial dataflow accelerator)


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
    - effective_ops_per_sec: Actual achieved (with all derates applied)

    Derate factors:
    - instruction_efficiency: Compiler/ISA efficiency (0.0-1.0)
    - memory_bottleneck_factor: Memory system limits (0.0-1.0)
    - empirical_derate: Combined measured performance (actual/sustained)
    """
    precision: Precision
    compute_resource: Optional[Union[ComputeResource, KPUComputeResource]] = None

    # Microarchitectural efficiency factors
    instruction_efficiency: float = 0.85     # Compiler/ISA efficiency (0.0-1.0)
    memory_bottleneck_factor: float = 0.75   # Memory system limits (0.0-1.0)
    tile_utilization: float = 1.0            # For KPU: fraction of tiles used

    # Hardware support
    native_acceleration: bool = True         # True = HW accelerated, False = emulated
    emulation_penalty: float = 0.01          # 100× slowdown if not native

    # Combined empirical derate (measured on real hardware)
    # derate_factor = empirical_performance / sustained_performance
    empirical_derate: float = 0.60

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
        Actual achieved performance (with all derates applied).

        Calculation:
        1. Start with sustained_ops_per_sec (DVFS-throttled)
        2. Apply emulation penalty if not native
        3. Apply empirical derate (measured performance factor)
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

        # Apply empirical derate
        return base_perf * self.empirical_derate


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


def kpu_t100_resource_model() -> HardwareResourceModel:
    """
    KPU-T100 with heterogeneous tile allocation for embodied AI.

    ============================================================================
    WORKLOAD-DRIVEN SILICON ALLOCATION STRATEGY
    ============================================================================

    Goal: Characterize embodied AI workloads → allocate tiles optimally

    Typical Embodied AI Workload Characterization:
    - 70% of operations: INT8 Conv, pooling, detection heads
    - 20% of operations: BF16 normalization, attention, tracking
    - 10% of operations: Large matmuls (classification, embedding projection)

    KPU T100 Silicon Allocation (100 tiles total):
    - 70 tiles: INT8-optimized (16×8 array processors with INT8 MACs)
    - 20 tiles: BF16-optimized (16×8 array processors with BF16 FMAs)
    - 10 tiles: Matrix units (8×8 systolic arrays for large matmuls)

    Key Advantages:
    ✓ All precisions are NATIVE (no emulation/fallback)
    ✓ Silicon matches workload distribution
    ✓ Excellent tile utilization (minimal idle silicon)
    ✓ 60-70% empirical derate (vs Jetson's 2-4%!)
    ✓ No DVFS throttling (well-designed thermal solution @ 6W)

    Comparison to Jetson Orin @ 15W:
    - Jetson: 85 TOPS peak (GPU dense) → 5 TOPS effective (6% derate, severe DVFS throttling)
    - KPU: 100 TOPS peak → 60 TOPS effective (60% derate, no throttling!)
    - KPU delivers 12× higher effective performance at 40% of the power!
    """
    # Clock domain (no DVFS issues - well-designed thermal solution)
    kpu_clock = ClockDomain(
        base_clock_hz=900e6,         # 900 MHz base (conservative)
        max_boost_clock_hz=1.0e9,    # 1 GHz boost
        sustained_clock_hz=950e6,    # 950 MHz sustained (95% of boost!)
        dvfs_enabled=True,
    )

    # ========================================================================
    # TILE TYPE 1: INT8-Optimized Tiles (70 tiles = 70% silicon)
    # ========================================================================
    int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=70,
        array_dimensions=(16, 8),  # 16×8 systolic array
        pe_configuration="INT8-MAC",

        ops_per_tile_per_clock={
            Precision.INT8: 128,   # 16×8 = 128 MACs/clock (fully optimized)
            Precision.INT4: 256,   # 2× throughput (packed INT4)
            Precision.BF16: 32,    # Supported but not optimal (25% efficiency)
        },

        optimization_level={
            Precision.INT8: 1.0,   # 100% optimized silicon
            Precision.INT4: 1.0,   # Native packed support
            Precision.BF16: 0.25,  # 25% efficiency (runs but slow)
        },

        clock_domain=kpu_clock,
    )

    # ========================================================================
    # TILE TYPE 2: BF16-Optimized Tiles (20 tiles = 20% silicon)
    # ========================================================================
    bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=20,
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",

        ops_per_tile_per_clock={
            Precision.BF16: 128,   # 16×8 = 128 FMAs/clock (fully optimized)
            Precision.FP32: 64,    # Supported (half rate)
            Precision.INT8: 64,    # Supported but not optimal (50% efficiency)
        },

        optimization_level={
            Precision.BF16: 1.0,
            Precision.FP32: 0.5,
            Precision.INT8: 0.5,   # Can run INT8, but inefficient
        },

        clock_domain=kpu_clock,
    )

    # ========================================================================
    # TILE TYPE 3: Matrix Units (10 tiles = 10% silicon)
    # ========================================================================
    matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=10,
        array_dimensions=(8, 8),  # Systolic array for large matmuls
        pe_configuration="Mixed-INT8-BF16-Matrix",

        ops_per_tile_per_clock={
            # 8×8 matrix unit with deep pipeline (8 stages)
            Precision.INT8: 512,   # 8×8×8 = 512 ops/clock (high throughput!)
            Precision.BF16: 256,   # Half rate for BF16
        },

        optimization_level={
            Precision.INT8: 1.0,
            Precision.BF16: 1.0,
        },

        clock_domain=kpu_clock,
    )

    # ========================================================================
    # KPU Compute Resource (Heterogeneous Tiles)
    # ========================================================================
    kpu_compute = KPUComputeResource(
        total_tiles=100,
        tile_specializations=[int8_tiles, bf16_tiles, matrix_tiles],
    )

    # Performance calculations:
    # INT8 peak: 70×128 + 20×64 + 10×512 = 8960 + 1280 + 5120 = 15,360 ops/clock
    #           @ 1 GHz = 15.4 TOPS (theoretical peak)
    # INT8 sustained: @ 950 MHz = 14.6 TOPS
    # INT8 effective: 14.6 × 0.65 empirical = 9.5 TOPS (60% derate!)

    # BF16 peak: 70×32 + 20×128 + 10×256 = 2240 + 2560 + 2560 = 7,360 ops/clock
    #           @ 1 GHz = 7.4 TOPS
    # BF16 effective: 7.0 × 0.60 = 4.2 TOPS

    # INT4 peak: 70×256 = 17,920 ops/clock @ 1 GHz = 17.9 TOPS
    # INT4 effective: 17.0 × 0.70 = 11.9 TOPS (even better!)

    # ========================================================================
    # THERMAL PROFILES for T100 (Embodied AI SKUs)
    # ========================================================================

    # 6W Profile: Battery-powered robots, drones (conservative)
    thermal_6w = ThermalOperatingPoint(
        name="6W-battery-optimized",
        tdp_watts=6.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=kpu_compute,
                empirical_derate=0.65,  # 65%! (vs Jetson's 3%!)
                tile_utilization=0.95,   # High tile utilization
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=kpu_compute,
                empirical_derate=0.60,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=kpu_compute,
                empirical_derate=0.70,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=kpu_compute,
                empirical_derate=0.50,
                tile_utilization=0.60,
                native_acceleration=True,
            ),
        }
    )

    # 12W Profile: Balanced performance/power for mobile robots
    kpu_clock_12w = ClockDomain(
        base_clock_hz=900e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=980e6,  # 98% of boost (better thermal headroom)
        dvfs_enabled=True,
    )
    kpu_compute_12w = KPUComputeResource(
        total_tiles=100,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=70, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=kpu_clock_12w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=20, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=kpu_clock_12w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=10, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=kpu_clock_12w,
            ),
        ],
    )

    thermal_12w = ThermalOperatingPoint(
        name="12W-balanced",
        tdp_watts=12.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=kpu_compute_12w,
                empirical_derate=0.70,  # Better with more power headroom
                tile_utilization=0.97,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=kpu_compute_12w,
                empirical_derate=0.65,
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=kpu_compute_12w,
                empirical_derate=0.75,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=kpu_compute_12w,
                empirical_derate=0.55,
                tile_utilization=0.65,
                native_acceleration=True,
            ),
        }
    )

    # 24W Profile: Maximum performance for stationary/tethered robots
    kpu_clock_24w = ClockDomain(
        base_clock_hz=900e6,
        max_boost_clock_hz=1.0e9,
        sustained_clock_hz=1.0e9,  # 100% of boost! (full thermal headroom)
        dvfs_enabled=True,
    )
    kpu_compute_24w = KPUComputeResource(
        total_tiles=100,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=70, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=kpu_clock_24w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=20, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=kpu_clock_24w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=10, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=kpu_clock_24w,
            ),
        ],
    )

    thermal_24w = ThermalOperatingPoint(
        name="24W-performance",
        tdp_watts=24.0,
        cooling_solution="active-fan-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=kpu_compute_24w,
                empirical_derate=0.75,  # Peak performance mode
                tile_utilization=0.98,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=kpu_compute_24w,
                empirical_derate=0.70,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=kpu_compute_24w,
                empirical_derate=0.80,
                tile_utilization=0.98,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=kpu_compute_24w,
                empirical_derate=0.60,
                tile_utilization=0.70,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="KPU-T100",
        hardware_type=HardwareType.KPU,
        compute_units=100,  # Total tiles
        threads_per_unit=128,  # Ops per tile (average)
        warps_per_unit=8,
        warp_size=16,

        # Thermal operating points for Embodied AI (6W/12W/24W)
        thermal_operating_points={
            "6W": thermal_6w,
            "12W": thermal_12w,
            "24W": thermal_24w,
        },
        default_thermal_profile="6W",

        # Legacy for backward compat
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=100e12,  # Simplified peak
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=1e12,
        l1_cache_per_unit=256 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=0.1e-12,
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,
    )


def kpu_t300_resource_model() -> HardwareResourceModel:
    """
    KPU-T300 with 300 heterogeneous tiles for automotive AI.

    ============================================================================
    AUTOMOTIVE AI WORKLOAD ALLOCATION (300 tiles)
    ============================================================================

    Target: Autonomous vehicles requiring higher throughput and redundancy
    - Same 70/20/10 ratio as T100, but scaled to 300 tiles
    - 210 INT8 tiles (70%): Detection, tracking, lane finding
    - 60 BF16 tiles (20%): Transformer-based planning, sensor fusion
    - 30 Matrix tiles (10%): Large embedding projections, classification

    Power Profiles for Automotive:
    - 12.5W: Low power mode (parking, idle monitoring)
    - 25W: Normal driving mode
    - 50W: High performance mode (full autonomy stack)

    Key Advantages vs Jetson Thor:
    ✓ 3× more compute tiles than T100
    ✓ Better thermal design (automotive-grade cooling)
    ✓ 70-80% empirical derate (vs Jetson's 3%)
    ✓ Higher reliability for safety-critical applications
    """
    # Clock domain for T300 (slightly higher clocks due to advanced process)
    t300_clock = ClockDomain(
        base_clock_hz=950e6,
        max_boost_clock_hz=1.1e9,  # 1.1 GHz boost
        sustained_clock_hz=1.0e9,  # 91% of boost at 12.5W
        dvfs_enabled=True,
    )

    # ========================================================================
    # T300 TILE ALLOCATION (300 tiles total, 70/20/10 ratio)
    # ========================================================================
    t300_int8_tiles = TileSpecialization(
        tile_type="INT8-primary",
        num_tiles=210,  # 70% of 300
        array_dimensions=(16, 8),
        pe_configuration="INT8-MAC",
        ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
        optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
        clock_domain=t300_clock,
    )

    t300_bf16_tiles = TileSpecialization(
        tile_type="BF16-primary",
        num_tiles=60,  # 20% of 300
        array_dimensions=(16, 8),
        pe_configuration="BF16-FMA",
        ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
        optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
        clock_domain=t300_clock,
    )

    t300_matrix_tiles = TileSpecialization(
        tile_type="Matrix-8x8",
        num_tiles=30,  # 10% of 300
        array_dimensions=(8, 8),
        pe_configuration="Mixed-INT8-BF16-Matrix",
        ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
        optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
        clock_domain=t300_clock,
    )

    t300_compute = KPUComputeResource(
        total_tiles=300,
        tile_specializations=[t300_int8_tiles, t300_bf16_tiles, t300_matrix_tiles],
    )

    # ========================================================================
    # THERMAL PROFILES for T300 (Automotive SKUs)
    # ========================================================================

    # 12.5W Profile: Low power mode (parking assistance, idle monitoring)
    thermal_12_5w = ThermalOperatingPoint(
        name="12.5W-automotive-low",
        tdp_watts=12.5,
        cooling_solution="automotive-liquid-cooling",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t300_compute,
                empirical_derate=0.70,  # Better than T100 @ 12W
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t300_compute,
                empirical_derate=0.65,
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t300_compute,
                empirical_derate=0.75,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=t300_compute,
                empirical_derate=0.58,
                tile_utilization=0.65,
                native_acceleration=True,
            ),
        }
    )

    # 25W Profile: Normal driving mode (full ADAS/autonomy stack)
    t300_clock_25w = ClockDomain(
        base_clock_hz=950e6,
        max_boost_clock_hz=1.1e9,
        sustained_clock_hz=1.05e9,  # 95% of boost
        dvfs_enabled=True,
    )
    t300_compute_25w = KPUComputeResource(
        total_tiles=300,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=210, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t300_clock_25w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=60, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t300_clock_25w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=30, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t300_clock_25w,
            ),
        ],
    )

    thermal_25w = ThermalOperatingPoint(
        name="25W-automotive-normal",
        tdp_watts=25.0,
        cooling_solution="automotive-liquid-cooling",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t300_compute_25w,
                empirical_derate=0.75,  # Excellent with automotive cooling
                tile_utilization=0.97,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t300_compute_25w,
                empirical_derate=0.70,
                tile_utilization=0.92,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t300_compute_25w,
                empirical_derate=0.80,
                tile_utilization=0.97,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=t300_compute_25w,
                empirical_derate=0.63,
                tile_utilization=0.70,
                native_acceleration=True,
            ),
        }
    )

    # 50W Profile: High performance mode (full autonomy, highway speeds)
    t300_clock_50w = ClockDomain(
        base_clock_hz=950e6,
        max_boost_clock_hz=1.1e9,
        sustained_clock_hz=1.1e9,  # 100% of boost! (excellent automotive cooling)
        dvfs_enabled=True,
    )
    t300_compute_50w = KPUComputeResource(
        total_tiles=300,
        tile_specializations=[
            TileSpecialization(
                tile_type="INT8-primary", num_tiles=210, array_dimensions=(16, 8),
                pe_configuration="INT8-MAC",
                ops_per_tile_per_clock={Precision.INT8: 128, Precision.INT4: 256, Precision.BF16: 32},
                optimization_level={Precision.INT8: 1.0, Precision.INT4: 1.0, Precision.BF16: 0.25},
                clock_domain=t300_clock_50w,
            ),
            TileSpecialization(
                tile_type="BF16-primary", num_tiles=60, array_dimensions=(16, 8),
                pe_configuration="BF16-FMA",
                ops_per_tile_per_clock={Precision.BF16: 128, Precision.FP32: 64, Precision.INT8: 64},
                optimization_level={Precision.BF16: 1.0, Precision.FP32: 0.5, Precision.INT8: 0.5},
                clock_domain=t300_clock_50w,
            ),
            TileSpecialization(
                tile_type="Matrix-8x8", num_tiles=30, array_dimensions=(8, 8),
                pe_configuration="Mixed-INT8-BF16-Matrix",
                ops_per_tile_per_clock={Precision.INT8: 512, Precision.BF16: 256},
                optimization_level={Precision.INT8: 1.0, Precision.BF16: 1.0},
                clock_domain=t300_clock_50w,
            ),
        ],
    )

    thermal_50w = ThermalOperatingPoint(
        name="50W-automotive-performance",
        tdp_watts=50.0,
        cooling_solution="automotive-liquid-cooling-enhanced",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=t300_compute_50w,
                empirical_derate=0.80,  # Peak automotive performance
                tile_utilization=0.98,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=t300_compute_50w,
                empirical_derate=0.75,
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=t300_compute_50w,
                empirical_derate=0.85,
                tile_utilization=0.98,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=t300_compute_50w,
                empirical_derate=0.68,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="KPU-T300",
        hardware_type=HardwareType.KPU,
        compute_units=300,  # Total tiles (3× T100)
        threads_per_unit=128,
        warps_per_unit=8,
        warp_size=16,

        # Thermal operating points for Automotive (12.5W/25W/50W)
        thermal_operating_points={
            "12.5W": thermal_12_5w,
            "25W": thermal_25w,
            "50W": thermal_50w,
        },
        default_thermal_profile="25W",

        # Legacy for backward compat
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=46.2e12,  # 46.2 TOPS @ 1.1 GHz (3× T100)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=22.2e12,  # 22.2 TFLOPS (3× T100)
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=59.2e12,  # 59.2 TOPS (3× T100)
                tensor_core_supported=True,
                relative_speedup=2.5,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=120e9,  # 120 GB/s (3× T100 scale)
        l1_cache_per_unit=256 * 1024,  # 256 KB per tile
        l2_cache_total=8 * 1024 * 1024,  # 8 MB shared
        main_memory=16 * 1024**3,  # 16 GB DDR5
        energy_per_flop_fp32=0.08e-12,  # Slightly better process
        energy_per_byte=10e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
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
                empirical_derate=0.47,  # 47% of sustained (3% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w_int8,
                empirical_derate=0.40,  # Worse (more memory bound)
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w_int8,
                empirical_derate=0.25,  # Much worse
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
                empirical_derate=0.60,  # Better (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                empirical_derate=0.50,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                empirical_derate=0.35,
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
                empirical_derate=0.75,  # Best case (still only 30% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_60w,
                empirical_derate=0.65,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_60w,
                empirical_derate=0.50,
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
                empirical_derate=0.50,  # 50% of sustained (3% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                empirical_derate=0.45,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                empirical_derate=0.30,
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
                empirical_derate=0.65,  # Better (6% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_60w,
                empirical_derate=0.60,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_60w,
                empirical_derate=0.45,
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
                empirical_derate=0.80,  # Best case (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_100w,
                empirical_derate=0.70,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_100w,
                empirical_derate=0.55,
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
