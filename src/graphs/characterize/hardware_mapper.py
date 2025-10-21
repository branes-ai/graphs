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
from typing import List, Dict, Optional, Tuple
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


@dataclass
class PrecisionProfile:
    """
    Performance characteristics for a specific numerical precision.

    Modern accelerators have vastly different peak performance at different
    precisions. For example, H100 has:
    - FP64: 60 TFLOPS
    - FP32: 60 TFLOPS (without Tensor Cores)
    - BF16: 750 TFLOPS (with Tensor Cores, 12.5× faster!)
    - FP8: 1.5 PFLOPS (with Tensor Cores, 25× faster!)
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

    def __init__(self, resource_model: HardwareResourceModel):
        self.resource_model = resource_model

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

        Args:
            ops: Number of operations (precision-agnostic count)
            bytes_transferred: Bytes read/written to main memory
            allocated_units: Compute units allocated
            occupancy: Occupancy fraction
            precision: Numerical precision

        Returns:
            (compute_time, memory_time, bottleneck)
        """
        # Get peak ops/sec for this precision
        peak_ops_per_sec = self.resource_model.get_peak_ops(precision)

        # Effective ops/sec with utilization
        effective_ops_per_sec = (
            peak_ops_per_sec *
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
    KPU-T100 (high-performance edge) resource model.

    Key characteristics:
    - Optimized for INT8 inference (10× faster than FP32)
    - Good FP16/BF16 support
    - Very energy efficient
    """
    return HardwareResourceModel(
        name="KPU-T100",
        hardware_type=HardwareType.KPU,
        compute_units=64,  # 64 tiles
        threads_per_unit=256,  # 256 threads per tile
        warps_per_unit=8,  # 8 vector units per tile
        warp_size=32,

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=10e12,  # 10 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=50e12,  # 50 TFLOPS (5× FP32)
                tensor_core_supported=True,
                relative_speedup=5.0,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=50e12,  # 50 TFLOPS
                tensor_core_supported=True,
                relative_speedup=5.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=100e12,  # 100 TOPS (10× FP32)
                tensor_core_supported=True,
                relative_speedup=10.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=200e12,  # 200 TOPS (20× FP32)
                tensor_core_supported=True,
                relative_speedup=20.0,
                bytes_per_element=0.5,  # Packed 2 per byte
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=1e12,  # 1 TB/s HBM
        l1_cache_per_unit=256 * 1024,  # 256 KB scratchpad per tile
        l2_cache_total=8 * 1024 * 1024,  # 8 MB
        main_memory=16 * 1024**3,  # 16 GB HBM
        energy_per_flop_fp32=0.1e-12,  # 10× more efficient than CPU
        energy_per_byte=12e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,  # Tiles allocated in pairs
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
