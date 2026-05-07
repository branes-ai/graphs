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

from graphs.core.structures import (
    SubgraphDescriptor,
    ParallelismDescriptor,
    BottleneckType,
    PartitionReport,
)
from graphs.core.confidence import ConfidenceLevel, EstimationConfidence

# Aliases for backward compatibility (these were originally in transform.partitioning)
# Importing directly from ir.structures avoids pulling in torch dependency
FusedSubgraph = SubgraphDescriptor
FusionReport = PartitionReport

if TYPE_CHECKING:
    from graphs.hardware.architectural_energy import ArchitecturalEnergyModel
    from graphs.hardware.fabric_model import SoCFabricModel


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

    FP64 = "fp64"  # IEEE Double Precision, 64-bit, 1 sign, 11 exponent, 52 mantissa
    FP32 = "fp32"  # IEEE Single Precision, 32-bit, 1 sign, 8 exponent, 23 mantissa
    TF32 = "tf32"  # NVIDIA TensorFloat-32, 19-bit (1 sign, 8 exp, 10 mantissa), Tensor Cores only
    FP16 = "fp16"  # IEEE Half Precision, 16-bit, 1 sign, 5 exponent, 10 mantissa
    FP8 = "fp8"  # IEEE FP8 (generic), 1 sign, 3 exponent, 4 mantissa
    FP8_E4M3 = "fp8_e4m3"  # 4-bit exponent, 3-bit mantissa
    FP8_E5M2 = "fp8_e5m2"  # 5-bit exponent, 2-bit mantissa
    FP4 = "fp4"  # 4-bit floating point, 1 sign, 2 exponent, 1 mantissa
    BF16 = "bf16"  # Brain Floating Point, 16-bit, 1 sign, 8 exponent, 7 mantissa
    INT64 = "int64"  # 64-bit integer
    INT32 = "int32"  # 32-bit integer
    INT16 = "int16"  # 16-bit integer
    INT8 = "int8"  # 8-bit integer
    INT4 = "int4"  # 4-bit integer


# Canonical operand byte width per Precision (used for weight/activation sizing).
# Sub-byte precisions are returned as floats (packed storage); callers should
# round when converting to integer byte counts.
_PRECISION_BYTES_PER_ELEMENT: Dict["Precision", float] = {}


def precision_bytes_per_element(precision: "Precision") -> float:
    """Return bytes per element for a numerical precision.

    Used by the partitioner and tensor-size accounting so that weight and
    activation byte counts scale with the requested analysis precision rather
    than defaulting to fp32. Sub-byte precisions (int4, fp4) return 0.5.
    """
    if not _PRECISION_BYTES_PER_ELEMENT:
        _PRECISION_BYTES_PER_ELEMENT.update(
            {
                Precision.FP64: 8,
                Precision.FP32: 4,
                Precision.TF32: 4,  # stored as fp32; only the multiplier is narrower
                Precision.FP16: 2,
                Precision.BF16: 2,
                Precision.INT64: 8,
                Precision.INT32: 4,
                Precision.INT16: 2,
                Precision.INT8: 1,
                Precision.FP8: 1,
                Precision.FP8_E4M3: 1,
                Precision.FP8_E5M2: 1,
                Precision.INT4: 0.5,
                Precision.FP4: 0.5,
            }
        )
    return _PRECISION_BYTES_PER_ELEMENT.get(precision, 4)


# ============================================================================
# Physics-Based Energy Model
# ============================================================================

# Standard cell FP32 ALU energy by process node (Joules per operation)
# Based on: Energy = Capacitance × Voltage² per switch
# Frequency does NOT affect energy per operation (only power)
PROCESS_NODE_ENERGY = {
    3: 1.2e-12,  # 1.2 pJ @ 3nm (Intel 18A, TSMC N3, AMD Zen 5)
    4: 1.3e-12,  # 1.3 pJ @ 4nm (TSMC N4/N4P)
    5: 1.5e-12,  # 1.5 pJ @ 5nm (TSMC N5, Samsung 5LPE)
    6: 1.65e-12,  # 1.65 pJ @ 6nm (TSMC N6 - 7nm extension with slightly better density)
    7: 1.8e-12,  # 1.8 pJ @ 7nm (TSMC N7, Samsung 7LPP)
    8: 1.9e-12,  # 1.9 pJ @ 8nm (Samsung 8LPP)
    10: 2.1e-12,  # 2.1 pJ @ 10nm (Intel 10nm/7)
    12: 2.5e-12,  # 2.5 pJ @ 12nm (TSMC 12FFC)
    14: 2.6e-12,  # 2.6 pJ @ 14nm (Intel 14nm, Samsung 14LPP)
    16: 2.7e-12,  # 2.7 pJ @ 16nm (TSMC 16FFC)
    28: 4.0e-12,  # 4.0 pJ @ 28nm (TSMC 28HPC+)
}

# Circuit type multipliers (relative to standard cell baseline)
# Captures layout efficiency and parallelism benefits
CIRCUIT_TYPE_MULTIPLIER = {
    "standard_cell": 1.0,  # Baseline: Standard cell library ALU
    "tensor_core": 0.85,  # 15% more efficient: Amortized control, fused MAC+accumulate
    "simd_packed": 0.90,  # 10% more efficient: Packed operations (AVX-512, NEON)
    "custom_datacenter": 2.75,  # 2.75× higher: 5+ GHz custom circuits, wide datapath, extra pipeline
}


def get_base_alu_energy(
    process_node_nm: int, circuit_type: str = "standard_cell"
) -> float:
    """
    Calculate base ALU energy per FP32 operation.

    Args:
        process_node_nm: Process node in nanometers (4, 5, 7, 16, 28, etc.)
        circuit_type: Circuit implementation type

    Returns:
        Energy per FP32 operation in Joules

    Example:
        >>> get_base_alu_energy(5, 'standard_cell')
        1.5e-12  # 1.5 pJ
        >>> get_base_alu_energy(5, 'tensor_core')
        1.275e-12  # 1.28 pJ (15% better)
    """
    base_energy = PROCESS_NODE_ENERGY.get(process_node_nm)
    if base_energy is None:
        # Interpolate for missing nodes
        nodes = sorted(PROCESS_NODE_ENERGY.keys())
        if process_node_nm < nodes[0]:
            base_energy = PROCESS_NODE_ENERGY[nodes[0]]
        elif process_node_nm > nodes[-1]:
            base_energy = PROCESS_NODE_ENERGY[nodes[-1]]
        else:
            # Linear interpolation
            for i in range(len(nodes) - 1):
                if nodes[i] <= process_node_nm <= nodes[i + 1]:
                    e1, e2 = (
                        PROCESS_NODE_ENERGY[nodes[i]],
                        PROCESS_NODE_ENERGY[nodes[i + 1]],
                    )
                    t = (process_node_nm - nodes[i]) / (nodes[i + 1] - nodes[i])
                    base_energy = e1 + t * (e2 - e1)
                    break

    multiplier = CIRCUIT_TYPE_MULTIPLIER.get(circuit_type, 1.0)
    return base_energy * multiplier


# ============================================================================
# Compute Fabric Model - Multi-Fabric Hardware Support
# ============================================================================


@dataclass
class ComputeFabric:
    """
    A specific type of compute unit with its own energy characteristics.

    Modern accelerators have multiple fabric types:
      - GPU: CUDA cores (standard) + Tensor Cores (tensor_core)
      - KPU: INT8 tiles (standard) + Matrix tiles (tensor_core)
      - CPU: Scalar ALUs (standard/custom) + SIMD units (simd_packed)

    Each fabric has:
      - Different energy per operation (based on circuit type)
      - Different peak throughput
      - Different precision support

    This enables workload-aware fabric selection during graph partitioning.

    Example:
        CUDA Core fabric (H100):
          - circuit_type: 'standard_cell'
          - energy: 1.5 pJ @ 5nm
          - ops_per_clock: {FP32: 2, FP64: 2}

        Tensor Core fabric (H100):
          - circuit_type: 'tensor_core'
          - energy: 1.28 pJ @ 5nm (15% more efficient)
          - ops_per_clock: {BF16: 512, FP8: 1024}
    """

    fabric_type: (
        str  # "cuda_core", "tensor_core", "int8_tile", "matrix_tile", "avx512", "neon"
    )
    circuit_type: (
        str  # "standard_cell", "tensor_core", "simd_packed", "custom_datacenter"
    )
    num_units: int  # Count of this fabric type
    ops_per_unit_per_clock: Dict[Precision, int]  # Peak throughput
    core_frequency_hz: float  # Operating frequency (for power calculation)

    # Energy model (process + circuit type)
    process_node_nm: int  # Process node (4, 5, 7, 16, etc.)
    energy_per_flop_fp32: float  # Base energy (calculated from process + circuit)

    # Precision-specific energy scaling
    energy_scaling: Dict[Precision, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate energy_per_flop_fp32 if not provided"""
        if self.energy_per_flop_fp32 == 0:
            self.energy_per_flop_fp32 = get_base_alu_energy(
                self.process_node_nm, self.circuit_type
            )

    def get_energy_per_op(self, precision: Precision) -> float:
        """Get energy per operation for a specific precision"""
        base = self.energy_per_flop_fp32
        scaling = self.energy_scaling.get(precision, 1.0)
        return base * scaling

    def get_peak_ops_per_sec(self, precision: Precision) -> float:
        """Calculate peak operations per second for a precision"""
        ops_per_clock = self.ops_per_unit_per_clock.get(precision, 0)
        return self.num_units * ops_per_clock * self.core_frequency_hz

    def get_peak_power(self, precision: Precision) -> float:
        """Calculate peak power (Watts) for sustained operation at this precision"""
        ops_per_sec = self.get_peak_ops_per_sec(precision)
        energy_per_op = self.get_energy_per_op(precision)
        return ops_per_sec * energy_per_op


# ============================================================================
# BOM Cost Modeling for Market Analysis
# ============================================================================


@dataclass
class BOMCostProfile:
    """
    Bill of Materials (BOM) cost breakdown for hardware accelerators.

    Used for market positioning analysis and TCO (Total Cost of Ownership) studies.
    All costs in USD.

    Example:
        KPU-T64 @ 10K units:
        - Silicon die: $75 (16nm TSMC)
        - Package: $15 (flip-chip BGA)
        - Memory: $20 (2GB LPDDR4X on-package)
        - PCB assembly: $8
        - Thermal: $2 (small heatsink)
        → Total BOM: $120
        → Retail (2.5x margin): $299
    """

    # Component costs
    silicon_die_cost: float  # Die fabrication cost (process node dependent)
    package_cost: float  # Package cost (flip-chip BGA, etc.)
    memory_cost: float  # On-package/on-module DRAM cost
    pcb_assembly_cost: float  # PCB, passives, assembly labor
    thermal_solution_cost: float  # Heatsink, thermal interface materials
    other_costs: float = 0.0  # Connectors, housing, testing, etc.

    # Totals and pricing
    total_bom_cost: float = 0.0  # Sum of all component costs (auto-calculated if 0)
    margin_multiplier: float = 2.5  # Typical margin: retail = BOM × margin
    retail_price: float = 0.0  # Customer-facing price (if known)

    # Context
    volume_tier: str = "10K+"  # Volume pricing tier ("1K+", "10K+", "100K+", "1M+")
    process_node: str = "16nm"  # Fabrication process (affects die cost)
    year: int = 2025  # Year of pricing (inflation adjustments)

    # Notes
    notes: str = ""  # Additional context or assumptions

    def __post_init__(self):
        """Calculate total BOM if not provided"""
        if self.total_bom_cost == 0:
            self.total_bom_cost = (
                self.silicon_die_cost
                + self.package_cost
                + self.memory_cost
                + self.pcb_assembly_cost
                + self.thermal_solution_cost
                + self.other_costs
            )

        # Estimate retail if not provided
        if self.retail_price == 0:
            self.retail_price = self.total_bom_cost * self.margin_multiplier

    def cost_per_tops(self, tops: float) -> float:
        """Calculate BOM cost per TOPS (INT8)"""
        if tops > 0:
            return self.total_bom_cost / tops
        return 0.0

    def cost_per_watt(self, tdp_watts: float) -> float:
        """Calculate BOM cost per Watt of TDP"""
        if tdp_watts > 0:
            return self.total_bom_cost / tdp_watts
        return 0.0


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

    base_clock_hz: float  # Minimum guaranteed frequency
    max_boost_clock_hz: float  # Maximum burst frequency (datasheet)
    sustained_clock_hz: float  # Actual frequency under thermal load (empirical)
    dvfs_enabled: bool = True  # Dynamic voltage/frequency scaling support

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
        peak_ops = num_units x ops_per_unit_per_clock x max_boost_clock_hz

    Example:
        16 ALUs x 4 INT8 ops/ALU/clock x 1.5 GHz = 96 GOPS INT8
    """

    resource_type: str  # "Ampere-SM", "Systolic-Array", "AVX512-Core"
    num_units: int  # Count of compute units (SMs, cores, tiles)
    ops_per_unit_per_clock: Dict[Precision, int]  # SIMD width per precision
    clock_domain: ClockDomain  # Frequency specifications

    def calc_peak_ops(self, precision: Precision) -> float:
        """Calculate peak from first principles (datasheet number)"""
        ops_per_clock = self.ops_per_unit_per_clock.get(precision, 0)
        return self.num_units * ops_per_clock * self.clock_domain.max_boost_clock_hz

    def calc_sustained_ops(self, precision: Precision) -> float:
        """Sustained performance under thermal load (DVFS throttled)"""
        ops_per_clock = self.ops_per_unit_per_clock.get(precision, 0)
        return self.num_units * ops_per_clock * self.clock_domain.sustained_clock_hz


class TileScheduleClass(Enum):
    """
    Scheduling discipline that governs how a tile's fill/drain overhead
    composes across an N-tile workload.

    This is the key lever that distinguishes the KPU's domain-flow advantage
    from a systolic weight-stationary architecture. The KPU is a distributed
    domain-flow machine capable of direct execution of systems of affine
    recurrence equations; the chart-4 "effective pipeline utilization vs.
    workload tile count" plot is driven by this enum.

    See ``docs/hardware/kpu_domainflow_tile_model.md``.

    Values:
        OUTPUT_STATIONARY: KPU. Fill and drain of tile N overlap with
            fill and drain of tile N+1 on the fabric; the workload pays
            fill+drain once across N tiles. Effective utilization
            saturates near 1.0 at ~12+ tiles.

        WEIGHT_STATIONARY: TPU / classic systolic. Weights are held in
            the PE array while inputs stream through. Modern designs
            (TPU v1 onward) double-buffer weights so fill/drain per
            tile is largely amortized (Jouppi et al., ISCA 2017, sec. 2).
            The dominant utilization loss is *shape/tile mismatch*
            against the fixed PE dimensions and bandwidth-bound layers,
            NOT fill/drain. We model the combined effect as a flat
            floor; numerical values published for real workloads vary
            widely (10-55% of peak depending on design and workload
            shape; Jouppi ISCA 2017 Table 3; DeepEdgeBench 2021).

        ROW_STATIONARY: Reserved for Eyeriss-style row-stationary
            dataflow schedules; modeled like OUTPUT_STATIONARY for M0.5.

        SIMT_DATA_PARALLEL: NVIDIA/AMD GPU and GPU-style SIMT data-
            parallel execution. Not a spatial dataflow fabric; CUDA
            cores execute the instruction stream cycle-by-cycle, with
            operands supplied from registers/shared memory. Utilization
            is capped by warp divergence, warp occupancy, and memory
            coherence traffic, and *does not amortize* with workload
            tile count. Distinct from the naive-CUDA-GEMM software-
            level "one thread per output" pattern, which is output-
            stationary at the register level but runs on SIMT hardware.

        UNSPECIFIED: No pipeline model applied (e.g., CPU SIMD, DSP).
            Effective utilization is treated as 1.0; other utilization
            penalties are modeled elsewhere.
    """

    OUTPUT_STATIONARY = "output_stationary"
    WEIGHT_STATIONARY = "weight_stationary"
    ROW_STATIONARY = "row_stationary"
    SIMT_DATA_PARALLEL = "simt_data_parallel"
    UNSPECIFIED = "unspecified"


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
    - 10 tiles: TC32 units (large matmuls)

    All precisions are native (no emulation) - just on different tile types.

    Domain-flow-tile fields (added M0.5):
        schedule_class: scheduling discipline governing fill/drain composition.
        pipeline_fill_cycles: cycles for a wavefront to propagate through
            the PE array (one-time per pipeline start).
        pipeline_drain_cycles: cycles to drain the final wavefront.
        pe_mac_energy_pj_steady_state: optional per-PE MAC energy in the
            steady-state pipelined regime (pJ). If None, the architectural
            energy model derives it from CIRCUIT_TYPE_MULTIPLIER and the
            precision-specific energy_scaling.
    """

    tile_type: str  # "INT8-primary", "BF16-primary", "Matrix-8x8"
    num_tiles: int  # Count of tiles with this specialization

    # Performance characteristics per precision
    # (all precisions are native, but some are more optimized)
    ops_per_tile_per_clock: Dict[Precision, int]

    # Silicon optimization level for each precision (0.0-1.0)
    # 1.0 = fully optimized PEs, 0.25 = supported but not optimal
    optimization_level: Dict[Precision, float]

    clock_domain: ClockDomain

    # Array processor characteristics
    array_dimensions: Tuple[int, int] = (16, 8)  # e.g., 16×8 systolic array
    pe_configuration: str = "Mixed"  # "INT8-MAC", "BF16-FMA", "Mixed"

    # Domain-flow-tile scheduling parameters (M0.5)
    schedule_class: TileScheduleClass = TileScheduleClass.UNSPECIFIED
    pipeline_fill_cycles: int = 0
    pipeline_drain_cycles: int = 0
    pe_mac_energy_pj_steady_state: Optional[float] = None

    # SIMT_DATA_PARALLEL parameters (GPU Tensor Core, warp-level execution)
    # Defaults are neutral (no penalty) for non-SIMT tiles.
    warp_divergence_rate: float = 0.0  # fraction of warp-issue cycles with divergence
    warp_occupancy: float = 1.0  # achieved warps / theoretical max warps
    coherence_efficiency: float = 1.0  # memory coherence / reuse efficiency

    @property
    def pe_count(self) -> int:
        """Total processing elements in this tile's PE array."""
        rows, cols = self.array_dimensions
        return rows * cols

    def effective_pipeline_utilization(
        self,
        num_tiles_in_workload: int,
        steady_cycles_per_tile: int = 128,
    ) -> float:
        """
        Effective pipeline utilization given the workload's tile-count shape.

        Args:
            num_tiles_in_workload: how many tiles the workload decomposes
                into along the sequential pipeline.
            steady_cycles_per_tile: steady-state wavefront duration per
                tile, in cycles. For a GEMM tile this is the reduction
                dimension K; default 128 is representative of a mid-size
                model's inner-dimension.

        Output-stationary (KPU): fill and drain amortize across N tiles.
            total = fill + N * steady + drain
            useful = N * steady
            -> util = N / (N + (fill+drain)/steady)
            Saturates near 1.0 as N grows; the KPU's signature advantage.

        Weight-stationary (TPU / systolic): approximated as a flat
            utilization floor representing the combined effect of
            shape/tile mismatch, bandwidth-bound layers, and any
            residual fill/drain overhead after double-buffering.
            Published values range from 10-55% on real workloads
            (Jouppi ISCA 2017 Table 3 reports 10-25% on TPU v1
            production workloads). Modeled as
            steady/(steady+fill+drain) using effective fill/drain that
            represents the combined loss mechanism, not literal
            wavefront propagation cycles.

        SIMT data-parallel (GPU Tensor Core): flat utilization capped
            by warp divergence, warp occupancy, and memory coherence.
            util = (1 - warp_divergence_rate * 0.5)
                   * warp_occupancy * coherence_efficiency
            Does not amortize with workload tile count.

        Unspecified: no pipeline model; returns 1.0.
        """
        if num_tiles_in_workload <= 0:
            return 0.0
        fill = int(self.pipeline_fill_cycles)
        drain = int(self.pipeline_drain_cycles)
        steady = max(int(steady_cycles_per_tile), 1)

        if (
            self.schedule_class == TileScheduleClass.OUTPUT_STATIONARY
            or self.schedule_class == TileScheduleClass.ROW_STATIONARY
        ):
            total = num_tiles_in_workload * steady + fill + drain
            useful = num_tiles_in_workload * steady
            return useful / total if total > 0 else 1.0

        if self.schedule_class == TileScheduleClass.WEIGHT_STATIONARY:
            total = num_tiles_in_workload * (steady + fill + drain)
            useful = num_tiles_in_workload * steady
            return useful / total if total > 0 else 1.0

        if self.schedule_class == TileScheduleClass.SIMT_DATA_PARALLEL:
            # Divergence: a divergent warp serializes 2 code paths, so
            # the fractional cost is ~0.5 * divergence_rate.
            divergence_penalty = 1.0 - 0.5 * max(
                0.0, min(1.0, self.warp_divergence_rate)
            )
            occ = max(0.0, min(1.0, self.warp_occupancy))
            coh = max(0.0, min(1.0, self.coherence_efficiency))
            return divergence_penalty * occ * coh

        # UNSPECIFIED: pipeline model not applicable for this tile class
        return 1.0


@dataclass
class KPUComputeResource:
    """
    KPU-specific compute model with heterogeneous tile allocation.

    Goal: Characterize workload -> recommend tile allocation -> build optimal KPU.

    Example silicon allocation for embodied AI:
        - 70% INT8 tiles (Conv, detection)
        - 20% BF16 tiles (normalization, attention)
        - 10% TC32 tiles (large matmuls)  tensorcore processing elements
    """

    total_tiles: int
    tile_specializations: List[TileSpecialization]

    def get_tiles_for_precision(self, precision: Precision) -> List[TileSpecialization]:
        """Find which tile types support this precision natively"""
        return [
            ts
            for ts in self.tile_specializations
            if precision in ts.ops_per_tile_per_clock
        ]

    def calc_peak_ops(self, precision: Precision) -> float:
        """
        Calculate peak performance across all tiles supporting this precision.

        Example: INT8 performance =
            70 INT8-tiles x 512 ops/tile/clock x 1 GHz +
            20 BF16-tiles x 256 ops/tile/clock x 1 GHz +
            10 TC32-tiles x 4096 ops/tile/clock x 1 GHz
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
        return {
            ts.tile_type: ts.num_tiles / self.total_tiles
            for ts in self.tile_specializations
        }


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
    instruction_efficiency: float = 0.85  # Compiler/ISA efficiency
    memory_bottleneck_factor: float = 0.75  # Memory system limits
    tile_utilization: float = 1.0  # For KPU: fraction of tiles used

    # Hardware support
    native_acceleration: bool = True  # True = HW accelerated, False = emulated
    emulation_penalty: float = 0.01  # 100× slowdown if not native

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

    name: str  # "15W-passive", "60W-active"
    tdp_watts: float  # Thermal Design Power
    cooling_solution: str  # "passive-heatsink", "active-fan", "liquid"

    # Per-precision performance characteristics at this thermal point
    performance_specs: Dict[Precision, PerformanceCharacteristics] = field(
        default_factory=dict
    )

    def get_effective_ops(self, precision: Precision) -> float:
        """Get actual achieved performance for a precision"""
        perf_spec = self.performance_specs.get(precision)
        if not perf_spec:
            return 0.0
        return perf_spec.effective_ops_per_sec


# ============================================================================
# V5-1: Memory hierarchy as a queueing-theory tier list
# ============================================================================
# See docs/plans/v5-memory-hierarchy-rewrite-plan.md.
#
# Each tier is a "server" in the operational-analysis sense -- capacity
# plus a service rate. The analyzer's tier_picker (V5-3) walks the list
# innermost-out to find the binding bottleneck for a given (op, shape).
# This dataclass is V5-1 scaffolding; nothing in the analyzer reads it
# yet.


@dataclass(frozen=True)
class MemoryTier:
    """One level of the memory hierarchy, as a queueing-theory server.

    Constructed by ``HardwareResourceModel.memory_hierarchy`` from
    existing fields (l1_cache_per_unit + l1_bandwidth_per_unit_bps,
    l2_cache_total + l2_bandwidth_bps, l3_cache_total + l3_bandwidth_bps,
    main_memory + peak_bandwidth, plus per-tier access_latency_ns
    overrides).

    Per-unit tiers (``is_per_unit=True``): ``capacity_bytes`` is the
    per-compute-unit value (e.g., 32 KB per CPU core, 256 KB per SM).
    The aggregate capacity across the chip is exposed via the
    ``total_capacity_bytes`` property.

    Aggregate tiers (``is_per_unit=False``, the default): capacity and
    BW are already the chip-wide totals. ``num_units`` is 1 for these.

    The ``access_latency_ns`` field is the per-request first-stream
    startup cost. For a kernel issuing N independent requests, the
    latency-bound floor is roughly ``startup + per_request * N`` --
    dominant for vector / matvec ops where bandwidth-limited service
    time is comparable to the startup. The analyzer's tier picker
    (V5-3) uses this as a memory-side analog of LAUNCH_BOUND.

    The ``achievable_fraction`` field defaults to 1.0; it's the place
    V5-5 calibration hangs the per-(hardware, tier) measured-vs-peak
    ratio (currently captured in the scalar bw_efficiency_scale).
    """

    name: str  # "L1", "L2", "L3", "DRAM", "scratchpad"
    capacity_bytes: int  # per-unit if is_per_unit else aggregate
    is_per_unit: bool
    num_units: int  # compute_units when is_per_unit=True; else 1
    peak_bandwidth_bps: float  # aggregate, for both per-unit and shared tiers
    # (per-unit BW is multiplied by num_units when
    # the property derives this -- the field is
    # always the aggregate the kernel sees)
    access_latency_ns: float  # first-request startup latency for this tier
    achievable_fraction: float = 1.0  # V5-5 calibration knob; default = ideal

    def __post_init__(self) -> None:
        """Validate physical invariants. Without this, V5-3's tier picker
        could silently produce garbage memory_time math from physically
        nonsensical inputs (negative BW, fraction > 1.0, etc.). Fail
        loudly at construction so calibration mistakes get caught at
        the source instead of as confusing analyzer drift downstream."""
        if not self.name:
            raise ValueError("MemoryTier.name must be non-empty")
        if self.capacity_bytes < 0:
            raise ValueError(
                f"MemoryTier({self.name!r}).capacity_bytes must be >= 0; "
                f"got {self.capacity_bytes}"
            )
        if self.num_units < 1:
            raise ValueError(
                f"MemoryTier({self.name!r}).num_units must be >= 1; "
                f"got {self.num_units}"
            )
        if self.peak_bandwidth_bps < 0:
            raise ValueError(
                f"MemoryTier({self.name!r}).peak_bandwidth_bps must be >= 0; "
                f"got {self.peak_bandwidth_bps}"
            )
        if self.access_latency_ns < 0:
            raise ValueError(
                f"MemoryTier({self.name!r}).access_latency_ns must be >= 0; "
                f"got {self.access_latency_ns}"
            )
        if not (0.0 <= self.achievable_fraction <= 1.0):
            raise ValueError(
                f"MemoryTier({self.name!r}).achievable_fraction must be in "
                f"[0.0, 1.0]; got {self.achievable_fraction}"
            )

    @property
    def total_capacity_bytes(self) -> int:
        """Aggregate capacity across the whole chip."""
        if self.is_per_unit:
            return self.capacity_bytes * self.num_units
        return self.capacity_bytes

    @property
    def effective_bandwidth_bps(self) -> float:
        """Achievable BW = peak * calibrated fraction. V5-5 will drive
        the fraction down from 1.0 to per-(hw, tier) measured values."""
        return self.peak_bandwidth_bps * self.achievable_fraction


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
    architecture_energy_model: Optional["ArchitecturalEnergyModel"] = None
    warp_size: int = 32  # Threads per warp (32 for NVIDIA, varies for others)

    # NEW: Multi-fabric support (CUDA + Tensor Cores, INT8 + Matrix tiles, Scalar + SIMD)
    # If specified, this overrides legacy energy_per_flop_fp32 and precision_profiles
    compute_fabrics: Optional[List[ComputeFabric]] = None

    # Precision-specific performance
    # Key: Precision, Value: PrecisionProfile
    precision_profiles: Dict[Precision, PrecisionProfile] = field(default_factory=dict)

    # Default precision for mixed-precision models
    default_precision: Precision = Precision.FP32

    # Energy scaling factors by precision (relative to FP32)
    energy_scaling: Dict[Precision, float] = field(
        default_factory=lambda: {
            Precision.FP64: 2.0,  # 2× energy of FP32
            Precision.FP32: 1.0,  # Baseline
            Precision.FP16: 0.5,  # Half energy
            Precision.BF16: 0.5,
            Precision.FP8_E4M3: 0.25,
            Precision.FP8_E5M2: 0.25,
            Precision.FP4: 0.125,
            Precision.INT32: 0.5,
            Precision.INT16: 0.25,
            Precision.INT8: 0.125,
            Precision.INT4: 0.0625,
        }
    )

    # Scheduling characteristics
    min_occupancy: float = 0.25  # Minimum occupancy for efficiency
    max_concurrent_kernels: int = 1  # Can run multiple kernels?
    wave_quantization: int = 4  # Units allocated in groups (e.g., 4 SMs/wave)

    # NEW: Multi-power-profile support with DVFS modeling
    thermal_operating_points: Optional[Dict[str, ThermalOperatingPoint]] = None
    default_thermal_profile: Optional[str] = None

    # GPU Microarchitecture (for accurate compute modeling)
    # These parameters define the actual hardware implementation
    cuda_cores_per_sm: Optional[int] = None  # 64 (Pascal-Turing), 128 (Ampere-Hopper)
    ops_per_clock_per_core: Optional[float] = 2.0  # FMA: 2 ops/clock for FP32
    sm_boost_clock_hz: Optional[float] = None  # Maximum boost frequency (short bursts)
    sm_sustained_clock_hz: Optional[float] = (
        None  # Sustained frequency under thermal load
    )

    # Tensor Core microarchitecture (for matrix operations)
    tensor_cores_per_sm: Optional[int] = None  # 4 (Volta/Turing/Ampere/Hopper)
    tensor_core_ops_per_clock: Optional[float] = (
        None  # Varies by precision and generation
    )

    # NEW: BOM cost modeling for market analysis
    bom_cost_profile: Optional[BOMCostProfile] = None

    # M2 Layer 2: per-SKU SIMD-vectorization efficiency.
    # Maps an op-kind label ('elementwise', 'matrix', 'default') to the
    # fraction of theoretical SIMD throughput that survives ISA overhead,
    # alignment, and tail-loop costs. Populated by CPU/DSP SKUs; left
    # None on architectures without a SIMD ISA (KPU, TPU, GPU shader
    # cores). Provenance lives in field_provenance under the key
    # ``simd_efficiency.<op_kind>``.
    simd_efficiency: Optional[Dict[str, float]] = None

    # M2 Layer 2: per-SKU systolic / wavefront pipeline-fill overhead.
    # The fraction of cycles spent filling and draining the pipeline
    # before sustained throughput is reached. Populated on TPU SKUs;
    # KPU SKUs read fill/drain from the attached
    # tile_energy_model / TileSpecialization (no double-counting).
    # Provenance: ``pipeline_fill_overhead``.
    pipeline_fill_overhead: Optional[float] = None

    # M3 Layer 3: storage kind for the innermost on-chip memory.
    # ``"cache"`` -- hardware-managed (CPU L1, GPU shared mem / L1).
    # Has a hit rate that depends on access pattern.
    # ``"scratchpad"`` -- software-managed (KPU tile-local SRAM,
    # TPU unified buffer, Hailo on-chip memory). Hit rate is
    # deterministic 1.0 by design; the cost lives in the host
    # software's tiling decisions, not in the runtime.
    # Provenance: ``l1_storage_kind``.
    l1_storage_kind: Optional[str] = None

    # M4 Layer 4: physical L2 capacity per unit. Distinct from the
    # legacy ``l2_cache_total`` field, which (per M1 schema convention)
    # holds the LLC -- on x86 CPUs that's L3, on GPUs / scratchpad
    # accelerators that's the L2 itself. ``l2_cache_per_unit`` is
    # always physical L2, so the Layer 4 panel can compare apples to
    # apples across architectures.
    # Provenance: ``l2_cache_per_unit``.
    l2_cache_per_unit: Optional[int] = None

    # M4 Layer 4: L2 topology classifier.
    # ``"per-unit"`` -- private L2 per core / SM / tile (CPU L2 caches,
    # KPU tile-local L2, Hailo per-unit L2).
    # ``"shared"``   -- L2 is a single shared structure (GPU L2 on
    # discrete GPUs with a distinct L3, multi-GPU systems).
    # ``"shared-llc"`` -- L2 is shared AND is the last-level cache
    # (no distinct L3): Ampere SoC GPUs, TPU unified buffer collapse.
    # Provenance: ``l2_topology``.
    l2_topology: Optional[str] = None

    # M5 Layer 5: explicit presence flag for the L3 / LLC layer.
    # True  -- the SKU has a distinct L3 cache (CPU LLC).
    # False -- the SKU has no L3 layer; either L2 is the LLC
    # (GPU SoCs) or the on-chip SRAM model has no inter-cluster
    # cache layer (KPU dataflow, TPU UB, Hailo).
    # Provenance: ``l3_present``.
    l3_present: Optional[bool] = None

    # M5 Layer 5: physical L3 capacity in bytes. Zero / None when
    # ``l3_present`` is False. Distinct from the legacy
    # ``l2_cache_total`` field (which carries the LLC value, == L3
    # on x86 by M1 schema convention; the new ``l3_cache_total`` is
    # explicit so the Layer 5 panel does not have to disambiguate).
    # Provenance: ``l3_cache_total``.
    l3_cache_total: Optional[int] = None

    # On-chip bandwidth peaks (issue #61). Optional so all 45+ existing
    # mappers continue to load without modification; populated mappers
    # unlock the v4 classifier's L1_BOUND vs L2_BOUND regime split.
    # All values in bytes/sec.
    #
    # ``l1_bandwidth_per_unit_bps`` -- per-SM / per-tile / per-core L1
    # read+write peak bandwidth. Aggregate L1 BW across the chip is
    # ``l1_bandwidth_per_unit_bps * compute_units``. Sites that need
    # the value MUST check for None and fall back to capacity-only
    # behavior (the v4-1 default). Provenance: ``l1_bandwidth_per_unit_bps``.
    l1_bandwidth_per_unit_bps: Optional[float] = None

    # ``l2_bandwidth_bps`` -- shared L2 (or LLC, per M1 schema convention)
    # peak bandwidth, aggregate across the chip. Optional for accelerators
    # whose on-chip memory hierarchy doesn't have a distinct L2 layer.
    # Provenance: ``l2_bandwidth_bps``.
    l2_bandwidth_bps: Optional[float] = None

    # ``l3_bandwidth_bps`` -- shared L3 peak bandwidth (CPU only). Zero /
    # None when ``l3_present`` is False. On x86, L3 is the LLC and is
    # what ``l2_cache_total`` already holds (per M1 convention); this
    # field is the matching bandwidth so the M5 Layer 5 panel can model
    # L3 as a distinct hop. Provenance: ``l3_bandwidth_bps``.
    l3_bandwidth_bps: Optional[float] = None

    # Per-tier access latency (V5-1, plan: docs/plans/v5-memory-hierarchy-rewrite-plan.md).
    # All values in nanoseconds, Optional so unaugmented mappers fall
    # through to typical-tier-class defaults in MemoryTier.
    #
    # The latency floor matters for memory-bound operators where the
    # single-stream startup dominates total time -- vector add, matvec,
    # and small-matrix ops where the per-request latency at the binding
    # tier is comparable to or larger than the bandwidth-limited service
    # time. This is the analog of LAUNCH_BOUND for the memory side.
    #
    # Typical published values (only as defaults if mapper doesn't set):
    #   L1:  1-3 ns   (4-12 cycles at 3 GHz)
    #   L2:  5-15 ns  (Skylake L2 ~12 cycles)
    #   L3:  20-40 ns (LLC depends on slice / ring topology)
    #   DRAM: 80-200 ns DDR / LPDDR; 100-300 ns HBM
    l1_access_latency_ns: Optional[float] = None
    l2_access_latency_ns: Optional[float] = None
    l3_access_latency_ns: Optional[float] = None
    dram_access_latency_ns: Optional[float] = None

    # V5-5: per-tier achievable_fraction overrides for the V5-3b
    # tier-aware roofline path. Maps tier name ("L1", "L2", "L3", "DRAM")
    # to the calibrated achievable BW fraction (peak * fraction =
    # effective). Empty default -> every tier defaults to 1.0 (ideal),
    # which is what V5-1 / V5-3b shipped with.
    #
    # V5-5 only calibrates DRAM from vector_add baselines; L1 / L2 / L3
    # entries stay absent until matmul-anchored calibration lands as a
    # V5-5 follow-up. Per-mapper overrides are set in the factory
    # functions, not here.
    tier_achievable_fractions: Dict[str, float] = field(default_factory=dict)

    # M5 Layer 5: cache-coherence protocol class.
    # ``"snoopy_mesi"`` -- snoopy MESI / MOESI on shared bus or
    # ring (CPU multi-core, single-socket).
    # ``"directory"``   -- directory-based coherence
    # (multi-socket NUMA).
    # ``"none"``        -- no inter-core coherence; SIMT / dataflow
    # / systolic architectures route data via shared memory or
    # explicit NoC tokens, not a coherence protocol.
    # Layer 5 owns the PROTOCOL energy cost (snoop messages, state
    # transitions); Layer 6 owns the TRANSPORT cost (NoC hops).
    # Provenance: ``coherence_protocol``.
    coherence_protocol: Optional[str] = None

    # M6 Layer 6: on-chip fabric / NoC topology + characteristics.
    # Captures the TRANSPORT cost of routing packets between cores,
    # caches, and memory controllers (NoC hops, bisection bandwidth,
    # per-flit energy). The PROTOCOL cost lives at Layer 5.
    # Provenance: ``soc_fabric``.
    soc_fabric: Optional["SoCFabricModel"] = None

    # M7 Layer 7: explicit external memory technology naming.
    # Examples: "DDR5", "LPDDR5", "LPDDR5x", "HBM3", "HBM3e",
    # "GDDR6", "on-chip" (for accelerators with no DRAM).
    # Provenance: ``memory_technology``.
    memory_technology: Optional[str] = None

    # M7 Layer 7: read / write energy asymmetry. Modern DRAM costs
    # ~1.2 - 1.5x more per byte for writes than reads (write-buffer
    # tail-end energy + bank precharge). Defaults to None when only
    # the legacy symmetric ``energy_per_byte`` is populated; the
    # Layer 7 panel falls back to that field when these are unset.
    # Values are picojoules per byte (panel-friendly units).
    # Provenance: ``memory_read_energy_per_byte_pj``,
    # ``memory_write_energy_per_byte_pj``.
    memory_read_energy_per_byte_pj: Optional[float] = None
    memory_write_energy_per_byte_pj: Optional[float] = None

    # Provenance of individual resource-model fields.
    # Maps field name (e.g., "peak_bandwidth", "energy_per_flop_fp32") to
    # the EstimationConfidence that describes where that value came from.
    # Unannotated fields default to UNKNOWN via ``get_provenance``. The
    # bottom-up benchmark suite (see docs/plans/bottom-up-microbenchmark-plan.md)
    # populates this map as each layer's fitter upgrades a field from
    # THEORETICAL to INTERPOLATED or CALIBRATED.
    field_provenance: Dict[str, EstimationConfidence] = field(default_factory=dict)

    def _unsupported_precision_error(self, precision: Precision) -> ValueError:
        supported = sorted(p.name.lower() for p in self.precision_profiles)
        return ValueError(
            f"{self.name} does not support {precision.name.lower()} -- "
            f"supported precisions are: {', '.join(supported) if supported else '(none)'}"
        )

    @property
    def memory_hierarchy(self) -> list["MemoryTier"]:
        """Derived view of the memory hierarchy as an ordered list of
        ``MemoryTier`` objects, innermost (smallest, fastest) to outermost.

        V5-1 scaffolding (plan: docs/plans/v5-memory-hierarchy-rewrite-plan.md).
        Builds the tier list from existing fields:

        * L1 if ``l1_cache_per_unit`` and ``l1_bandwidth_per_unit_bps`` are
          both set (per-unit tier; aggregate = capacity * compute_units).
        * L2 if ``l2_bandwidth_bps`` is set (the M1 schema convention puts
          the LLC capacity in ``l2_cache_total``; on x86 that's L3, on
          GPUs that's L2 -- see ``l2_topology``).
        * L3 if ``l3_cache_total`` and ``l3_bandwidth_bps`` are both set.
        * DRAM is always present from ``main_memory`` + ``peak_bandwidth``.

        Mappers that don't populate the on-chip BW peaks (most of the 45+
        existing mappers as of V5-1) return a hierarchy with only the
        DRAM tier. Backward compat: nothing in V5-1 reads this property
        in the analyzer; that lands in V5-3.

        Tier ``access_latency_ns`` falls back to typical published values
        per tier class when the mapper doesn't override (1.5 ns L1,
        10 ns L2, 30 ns L3, 100 ns DRAM). These defaults match consumer
        x86 / Ampere ballparks; specific mappers should override.

        V5-5: ``tier_achievable_fractions`` overrides MemoryTier's default
        ``achievable_fraction = 1.0`` per tier name. Mappers calibrated
        against the V5-2b vector_add baselines set DRAM here (i7 = 0.47,
        Jetson Orin Nano = 0.55 as of V5-5 PR). L1 / L2 / L3 entries
        stay absent until the V5-5 follow-up calibrates them from
        matmul data.
        """
        tiers: list["MemoryTier"] = []
        # Per-tier achievable_fraction lookup; default to ideal (1.0)
        # so existing un-calibrated mappers behave exactly as before.
        af = self.tier_achievable_fractions

        # L1 (per-unit capacity, per-unit BW). Only emitted when both
        # capacity and BW are set. peak_bandwidth_bps stored on the tier
        # is the AGGREGATE (per-unit BW * compute_units) so callers can
        # treat all tiers uniformly; the per-unit value is recoverable
        # via tier.peak_bandwidth_bps / tier.num_units.
        if (
            self.l1_cache_per_unit
            and self.l1_bandwidth_per_unit_bps
            and self.l1_bandwidth_per_unit_bps > 0
        ):
            tiers.append(
                MemoryTier(
                    name="L1",
                    capacity_bytes=self.l1_cache_per_unit,
                    is_per_unit=True,
                    num_units=self.compute_units,
                    peak_bandwidth_bps=(
                        self.l1_bandwidth_per_unit_bps * self.compute_units
                    ),
                    access_latency_ns=(
                        self.l1_access_latency_ns
                        if self.l1_access_latency_ns is not None
                        else 1.5
                    ),
                    achievable_fraction=af.get("L1", 1.0),
                )
            )

        # L2. The M1 convention: ``l2_cache_total`` may carry either L2 (on
        # GPUs that have a distinct L2 -- e.g., H100) or LLC (on x86, where
        # the LLC IS L3 and there's no separate L2 capacity at the schema
        # level). The hierarchy emits this as "L2" when ``l2_bandwidth_bps``
        # is set; if the mapper has both ``l2_bandwidth_bps`` and
        # ``l3_bandwidth_bps``, the L3 tier is emitted as a separate hop
        # below.
        if self.l2_cache_total and self.l2_bandwidth_bps and self.l2_bandwidth_bps > 0:
            tiers.append(
                MemoryTier(
                    name="L2",
                    capacity_bytes=self.l2_cache_total,
                    is_per_unit=False,
                    num_units=1,
                    peak_bandwidth_bps=self.l2_bandwidth_bps,
                    access_latency_ns=(
                        self.l2_access_latency_ns
                        if self.l2_access_latency_ns is not None
                        else 10.0
                    ),
                    achievable_fraction=af.get("L2", 1.0),
                )
            )

        # L3 / LLC (CPU only, typically). Distinct from L2 only on x86
        # where ``l2_cache_total`` carries the LLC value but we want a
        # separate hop. On i7-12700K post-#94: l3_bandwidth_bps=200e9
        # but no l2_bandwidth_bps, so the hierarchy is L1 -> L3 -> DRAM
        # (no distinct L2 hop).
        if self.l3_cache_total and self.l3_bandwidth_bps and self.l3_bandwidth_bps > 0:
            tiers.append(
                MemoryTier(
                    name="L3",
                    capacity_bytes=self.l3_cache_total,
                    is_per_unit=False,
                    num_units=1,
                    peak_bandwidth_bps=self.l3_bandwidth_bps,
                    access_latency_ns=(
                        self.l3_access_latency_ns
                        if self.l3_access_latency_ns is not None
                        else 30.0
                    ),
                    achievable_fraction=af.get("L3", 1.0),
                )
            )
        elif (
            self.l2_cache_total
            and not self.l2_bandwidth_bps
            and self.l3_bandwidth_bps
            and self.l3_bandwidth_bps > 0
        ):
            # x86 fallback: ``l2_cache_total`` carries the LLC (which IS L3
            # on x86), and ``l3_bandwidth_bps`` is the matching BW. Emit
            # this as a single L3 tier using the L2_total capacity.
            tiers.append(
                MemoryTier(
                    name="L3",
                    capacity_bytes=self.l2_cache_total,
                    is_per_unit=False,
                    num_units=1,
                    peak_bandwidth_bps=self.l3_bandwidth_bps,
                    access_latency_ns=(
                        self.l3_access_latency_ns
                        if self.l3_access_latency_ns is not None
                        else 30.0
                    ),
                    achievable_fraction=af.get("L3", 1.0),
                )
            )

        # DRAM is always present (every mapper sets peak_bandwidth +
        # main_memory; these are required fields).
        tiers.append(
            MemoryTier(
                name="DRAM",
                capacity_bytes=self.main_memory,
                is_per_unit=False,
                num_units=1,
                peak_bandwidth_bps=self.peak_bandwidth,
                access_latency_ns=(
                    self.dram_access_latency_ns
                    if self.dram_access_latency_ns is not None
                    else 100.0
                ),
                achievable_fraction=af.get("DRAM", 1.0),
            )
        )

        return tiers

    def get_peak_ops(self, precision: Precision) -> float:
        """
        Get peak operations per second for a given precision.

        Raises:
            ValueError: if ``precision`` is not in ``precision_profiles``.
            Previously this fell back to ``default_precision`` silently, which
            hid analyzer-precision-plumbing bugs (issue #53). Hardware that
            genuinely supports a precision must list it explicitly.
        """
        if precision in self.precision_profiles:
            return self.precision_profiles[precision].peak_ops_per_sec
        raise self._unsupported_precision_error(precision)

    def get_precision_profile(self, precision: Precision) -> PrecisionProfile:
        """Get precision profile.

        Raises:
            ValueError: if ``precision`` is not in ``precision_profiles``.
            See :meth:`get_peak_ops` for rationale.
        """
        if precision in self.precision_profiles:
            return self.precision_profiles[precision]
        raise self._unsupported_precision_error(precision)

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

    # ------------------------------------------------------------------
    # Field provenance API
    # ------------------------------------------------------------------
    # Every field of a HardwareResourceModel originates from one of
    #   - a datasheet (THEORETICAL)
    #   - an interpolation across two measured points (INTERPOLATED)
    #   - a direct measurement on this silicon (CALIBRATED)
    #   - unknown / unannotated (UNKNOWN)
    # The ``field_provenance`` map records this per-field. Bottom-up
    # benchmark fitters use these setters so callers can ask "what is
    # the confidence on the L3 cache energy for this SKU?" without
    # scraping source comments.

    def set_provenance(
        self,
        field_name: str,
        confidence: EstimationConfidence,
    ) -> None:
        """
        Record the provenance of a named resource-model field.

        ``field_name`` is free-form so it can cover nested or derived
        quantities (e.g., ``"thermal.maxn.efficiency_factor"``).
        """
        self.field_provenance[field_name] = confidence

    def get_provenance(self, field_name: str) -> EstimationConfidence:
        """
        Return the provenance for a field, or UNKNOWN if unrecorded.

        Returning UNKNOWN (rather than None) keeps callers from having
        to branch on absence: ``model.get_provenance(x).level`` is
        always safe.
        """
        return self.field_provenance.get(field_name, EstimationConfidence.unknown())

    def aggregate_confidence(self) -> EstimationConfidence:
        """
        Return the weakest per-field confidence across the model.

        Estimation rules require aggregate confidence to be the minimum
        across the analysis chain. If any field is UNKNOWN or
        THEORETICAL, the aggregate demotes to match. Empty provenance
        maps to UNKNOWN.
        """
        if not self.field_provenance:
            return EstimationConfidence.unknown()

        # Ordering: CALIBRATED > INTERPOLATED > THEORETICAL > UNKNOWN.
        rank = {
            ConfidenceLevel.CALIBRATED: 3,
            ConfidenceLevel.INTERPOLATED: 2,
            ConfidenceLevel.THEORETICAL: 1,
            ConfidenceLevel.UNKNOWN: 0,
        }
        worst = min(self.field_provenance.values(), key=lambda c: rank[c.level])
        return worst


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
        thermal_profile: Optional[str] = None,
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
        precision: Precision = Precision.FP32,
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
        precision: Precision = Precision.FP32,
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
        precision: Precision,
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
            thermal_point = self.resource_model.thermal_operating_points[
                self.thermal_profile
            ]

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
            base_ops_per_sec
            * (allocated_units / self.resource_model.compute_units)
            * occupancy
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
        self, ops: int, bytes_transferred: int, precision: Precision
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
        execution_context: Optional[Dict] = None,
    ) -> Tuple[float, float, Optional["ArchitecturalEnergyBreakdown"]]:
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
            from graphs.hardware.architectural_energy import (
                ArchitecturalEnergyBreakdown,
            )

            if execution_context is None:
                execution_context = {}

            arch_breakdown = self.resource_model.architecture_energy_model.compute_architectural_energy(
                ops=ops,
                bytes_transferred=bytes_transferred,
                compute_energy_baseline=compute_energy,
                data_movement_energy_baseline=memory_energy,
                execution_context=execution_context,
            )

            # Apply architectural overheads
            compute_energy += arch_breakdown.compute_overhead
            memory_energy += arch_breakdown.data_movement_overhead

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
