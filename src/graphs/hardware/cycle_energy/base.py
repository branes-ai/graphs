"""
Base Classes for Cycle-Level Energy Models

This module provides the foundational data structures and enumerations
used across all architecture-specific energy models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class CyclePhase(Enum):
    """
    Phases of processor execution cycles.

    Common phases for stored program machines:
    - INSTRUCTION_FETCH, INSTRUCTION_DECODE, OPERAND_FETCH, EXECUTE, WRITEBACK

    GPU-specific (SIMT overhead):
    - SIMT_FIXED_OVERHEAD: Fixed infrastructure cost (kernel launch, SM activation)
    - SIMT_THREAD_MGMT: Warp scheduling, thread context
    - SIMT_COHERENCE: Cache coherence machinery
    - SIMT_SYNC: Barriers, divergence handling

    TPU-specific (Systolic array):
    - SYSTOLIC_DATA_LOAD: Loading data into systolic array
    - SYSTOLIC_WEIGHT_LOAD: Loading weights (weight-stationary)
    - SYSTOLIC_COMPUTE: Matrix multiply in systolic array
    - SYSTOLIC_DRAIN: Draining results from array

    KPU-specific (Spatial dataflow):
    - SPATIAL_CONFIG: Configuration/reconfiguration overhead
    - SPATIAL_STREAM: Data streaming through tiles
    - SPATIAL_COMPUTE: Compute in spatial tiles
    - SPATIAL_INTERCONNECT: On-chip network energy

    Memory hierarchy (all architectures):
    - MEM_L1, MEM_L2, MEM_L3, MEM_DRAM
    """
    # Common phases (stored program machines)
    INSTRUCTION_FETCH = "instruction_fetch"
    INSTRUCTION_DECODE = "instruction_decode"
    OPERAND_FETCH = "operand_fetch"
    EXECUTE = "execute"
    WRITEBACK = "writeback"

    # GPU-specific phases (SIMT overhead)
    SIMT_FIXED_OVERHEAD = "simt_fixed_overhead"
    SIMT_THREAD_MGMT = "simt_thread_mgmt"
    SIMT_COHERENCE = "simt_coherence"
    SIMT_SYNC = "simt_sync"

    # TPU-specific phases (Systolic array)
    SYSTOLIC_DATA_LOAD = "systolic_data_load"
    SYSTOLIC_WEIGHT_LOAD = "systolic_weight_load"
    SYSTOLIC_COMPUTE = "systolic_compute"
    SYSTOLIC_DRAIN = "systolic_drain"
    SYSTOLIC_CONTROL = "systolic_control"

    # KPU-specific phases (Spatial dataflow)
    SPATIAL_CONFIG = "spatial_config"
    SPATIAL_STREAM = "spatial_stream"
    SPATIAL_COMPUTE = "spatial_compute"
    SPATIAL_INTERCONNECT = "spatial_interconnect"

    # KPU EDDO Scratchpad Hierarchy (software-managed, NOT caches)
    # These are directly-addressed SRAM banks with no tag lookups, no coherence.
    # Data placement is compiler-directed (Explicit Data Distribution & Orchestration).
    EDDO_TILE_SCRATCHPAD = "eddo_tile_scratchpad"      # Per-tile local SRAM (like L1)
    EDDO_GLOBAL_SCRATCHPAD = "eddo_global_scratchpad"  # Shared SRAM (like L2)
    EDDO_STREAMING_BUFFER = "eddo_streaming_buffer"    # DMA staging (like L3)
    EDDO_DMA_SETUP = "eddo_dma_setup"                  # DMA descriptor setup

    # Memory hierarchy (all architectures)
    MEMORY_ACCESS = "memory_access"  # Parent category
    MEM_L1 = "mem_l1"
    MEM_L2 = "mem_l2"
    MEM_L3 = "mem_l3"
    MEM_DRAM = "mem_dram"
    MEM_HBM = "mem_hbm"
    MEM_SRAM = "mem_sram"  # For TPU/KPU on-chip SRAM


class OperatingMode(Enum):
    """
    Operating modes based on where the working set resides.

    Each mode represents a scenario where data primarily lives at a specific
    level of the memory hierarchy. This enables apples-to-apples comparison
    across architectures.
    """
    L1_RESIDENT = "l1"      # Working set fits in L1/shared/scratchpad/SRAM
    L2_RESIDENT = "l2"      # Working set fits in L2, L1 acts as cache
    L3_RESIDENT = "l3"      # Working set fits in L3/LLC (CPU only)
    DRAM_RESIDENT = "dram"  # Working set streams from off-chip memory


class OperatorType(Enum):
    """
    DNN operator types classified by their cache behavior.

    This classification determines L2 hit ratios based on data reuse patterns:

    HIGH_REUSE (MatMul/Conv):
        - Tiled execution with O(n) reuse per element
        - L2 hit ratio depends on tile size vs L2 size
        - Example: GEMM with 128x128 tiles, each element reused 128 times

    LOW_REUSE (MatVec, reductions):
        - Limited reuse, typically O(1) to O(sqrt(n))
        - Partial L2 benefit if working set fits
        - Example: MatVec - each matrix row used once

    STREAMING (elementwise, activations):
        - O(1) reuse - each element touched exactly once
        - Flushes L2 contents, nearly 0% hit ratio
        - Example: ReLU, BatchNorm, Softmax

    The key insight: A STREAMING operator between two HIGH_REUSE operators
    will flush the L2, forcing the second operator to reload from HBM.
    """
    HIGH_REUSE = "high_reuse"      # MatMul, Conv2D (tiled, high reuse)
    LOW_REUSE = "low_reuse"        # MatVec, pooling, reductions
    STREAMING = "streaming"        # Activation, BatchNorm, Softmax, elementwise


@dataclass
class HitRatios:
    """
    Cache hit ratios for a given operating mode.

    These ratios determine what fraction of memory accesses are served
    by each level of the cache hierarchy.
    """
    l1_hit: float = 0.0   # Fraction served by L1
    l2_hit: float = 0.0   # Fraction of L1 misses served by L2
    l3_hit: float = 0.0   # Fraction of L2 misses served by L3
    # Remainder goes to DRAM/HBM

    def __post_init__(self):
        """Validate hit ratios are in [0, 1]."""
        for name, val in [('l1_hit', self.l1_hit), ('l2_hit', self.l2_hit),
                          ('l3_hit', self.l3_hit)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


# Default hit ratios for each operating mode
DEFAULT_HIT_RATIOS: Dict[OperatingMode, HitRatios] = {
    OperatingMode.L1_RESIDENT: HitRatios(l1_hit=1.0, l2_hit=0.0, l3_hit=0.0),
    OperatingMode.L2_RESIDENT: HitRatios(l1_hit=0.90, l2_hit=1.0, l3_hit=0.0),
    OperatingMode.L3_RESIDENT: HitRatios(l1_hit=0.85, l2_hit=0.90, l3_hit=1.0),
    OperatingMode.DRAM_RESIDENT: HitRatios(l1_hit=0.80, l2_hit=0.85, l3_hit=0.90),
}


def get_mode_description(mode: OperatingMode) -> str:
    """Get human-readable description of an operating mode."""
    descriptions = {
        OperatingMode.L1_RESIDENT: "L1-Resident (on-chip fast memory)",
        OperatingMode.L2_RESIDENT: "L2-Resident (L1 as cache)",
        OperatingMode.L3_RESIDENT: "L3-Resident (L1+L2 as cache, CPU/KPU)",
        OperatingMode.DRAM_RESIDENT: "DRAM-Resident (off-chip streaming)",
    }
    return descriptions[mode]


# =============================================================================
# Cache/Memory Sizes by Architecture
# =============================================================================
# These sizes determine hit ratios for implicit (hardware-managed) caches.
# For explicit (software-managed) memories like KPU scratchpads, these
# represent capacity limits but hit ratios are determined by compiler.
#
# Memory Hierarchy Mapping:
#   CPU: L1 (per-core) -> L2 (per-core) -> L3/LLC (shared) -> DRAM
#   GPU: Shared Mem/L1 -> L2 (shared)   -> [none]          -> HBM
#   TPU: VMEM (per-core) -> SRAM (shared) -> [none]        -> HBM
#   KPU: Streaming buffer -> Tile staging -> Global scratchpad -> DRAM

# -----------------------------------------------------------------------------
# GPU Memory Hierarchy
# -----------------------------------------------------------------------------
# L1: Shared memory / L1 cache (per-SM, ~128-256 KB)
# L2: Shared L2 cache (chip-wide)
# No L3 - misses go to HBM

GPU_L2_CACHE_SIZES = {
    'h100': 50 * 1024 * 1024,        # 50 MB L2 (H100 SXM)
    'a100': 40 * 1024 * 1024,        # 40 MB L2 (A100)
    'v100': 6 * 1024 * 1024,         # 6 MB L2 (V100)
    'jetson_orin': 4 * 1024 * 1024,  # 4 MB L2 (Jetson Orin)
    'default': 40 * 1024 * 1024,     # Default to 40 MB
}

# -----------------------------------------------------------------------------
# CPU Memory Hierarchy
# -----------------------------------------------------------------------------
# L1: Per-core L1 (32-64 KB)
# L2: Per-core or per-cluster L2 (256 KB - 2 MB)
# L3: Shared LLC (last-level cache)

CPU_L3_CACHE_SIZES = {
    'xeon_w9': 105 * 1024 * 1024,    # 105 MB (Xeon W9-3595X)
    'epyc_9654': 384 * 1024 * 1024,  # 384 MB (AMD EPYC 9654)
    'i9_14900k': 36 * 1024 * 1024,   # 36 MB (Intel i9-14900K)
    'apple_m3_max': 48 * 1024 * 1024, # 48 MB (Apple M3 Max)
    'default': 32 * 1024 * 1024,     # Default to 32 MB
}

# -----------------------------------------------------------------------------
# TPU Memory Hierarchy
# -----------------------------------------------------------------------------
# L1: VMEM - per-core vector memory (closest to MXU)
# L2: Shared SRAM - on-chip memory shared across cores
# No L3 - misses go to HBM

TPU_L2_SRAM_SIZES = {
    'tpu_v4': 32 * 1024 * 1024,      # 32 MB shared SRAM per chip
    'tpu_v5': 48 * 1024 * 1024,      # 48 MB (estimate)
    'coral_edge': 8 * 1024 * 1024,   # 8 MB (Edge TPU)
    'default': 32 * 1024 * 1024,     # Default to 32 MB
}

# -----------------------------------------------------------------------------
# KPU Memory Hierarchy (EDDO - Explicit Data Distribution & Orchestration)
# -----------------------------------------------------------------------------
# L1: Streaming buffer - bridges data into processing elements
# L2: Tile staging area - bridges cycle time between L3 and compute fabric
# L3: Global tile scratchpad - shared across all tiles
#
# Note: KPU uses explicit (software-managed) memory, so hit ratios are
# compiler-determined. These sizes affect whether data FITS on-chip.

KPU_L1_STREAMING_BUFFER_SIZES = {
    'default': 64 * 1024,            # 64 KB per-tile streaming buffer
}

KPU_L2_TILE_STAGING_SIZES = {
    'default': 256 * 1024,           # 256 KB per-tile staging area
}

KPU_L3_GLOBAL_SCRATCHPAD_SIZES = {
    'default': 8 * 1024 * 1024,      # 8 MB global scratchpad
}

# Backward compatibility alias
DEFAULT_L2_CACHE_SIZES = GPU_L2_CACHE_SIZES


class MemoryType(Enum):
    """
    Memory hierarchy level for cache hit ratio computation.

    Used to specify which level of the hierarchy we're computing hit ratios for.
    The behavior is similar across architectures but the naming differs.
    """
    L2 = "l2"  # GPU L2, TPU shared SRAM, KPU tile staging
    L3 = "l3"  # CPU LLC, KPU global scratchpad


def compute_cache_hit_ratio(
    operator_type: OperatorType,
    working_set_bytes: int,
    cache_size_bytes: int,
    cache_is_cold: bool = False,
    is_explicit_memory: bool = False,
) -> float:
    """
    Compute cache hit ratio based on operator type and working set size.

    This models cache behavior for implicit (hardware-managed) caches like
    CPU L3, GPU L2, and TPU shared SRAM. The hit ratio depends on:
    1. Operator type (determines reuse pattern)
    2. Working set size relative to cache size
    3. Whether cache was flushed by a previous streaming operator

    For explicit (software-managed) memory like KPU scratchpads, the model
    is different: data either fits (100% hit) or doesn't (0% hit).

    Args:
        operator_type: Type of DNN operator (HIGH_REUSE, LOW_REUSE, STREAMING)
        working_set_bytes: Size of data accessed by this operator
        cache_size_bytes: Size of the cache/scratchpad
        cache_is_cold: True if cache was flushed by previous operator
        is_explicit_memory: True for KPU (compiler-managed, not hardware-managed)

    Returns:
        Hit ratio in [0.0, 1.0]

    Model for Implicit Caches (CPU L3, GPU L2, TPU SRAM):
        HIGH_REUSE (MatMul/Conv with tiling):
            - If tiles fit: high hit ratio (0.85-0.95)
            - Tiles designed to maximize reuse within cache
            - Cold start penalty if cache was flushed

        LOW_REUSE (MatVec, reductions):
            - Limited reuse, depends on working set fit
            - If working_set < cache: ~50% hit ratio
            - If working_set > cache: ~10% hit ratio

        STREAMING (elementwise, activations):
            - Each element accessed once, no reuse
            - Nearly 0% hit ratio regardless of size
            - Flushes cache for next operator

    Model for Explicit Memory (KPU scratchpads):
        - Compiler places data explicitly
        - If data fits: 100% hit (no capacity misses)
        - If data doesn't fit: spills to next level (0% hit at this level)
        - No "cold" penalty - compiler handles placement
    """
    # Explicit memory model (KPU)
    if is_explicit_memory:
        # Compiler-managed: data either fits or it doesn't
        # No partial hits, no cold penalties
        if working_set_bytes <= cache_size_bytes:
            return 1.0  # Data fits, compiler placed it correctly
        else:
            return 0.0  # Data doesn't fit, spills to next level

    # Implicit cache model (CPU L3, GPU L2, TPU SRAM)
    ws_ratio = working_set_bytes / cache_size_bytes

    if operator_type == OperatorType.HIGH_REUSE:
        # MatMul/Conv with tiled execution
        # Tiles are sized to fit in cache, so we expect high reuse
        # But if cache is cold (previous streaming op), first pass has misses

        if ws_ratio <= 0.5:
            # Working set easily fits - excellent reuse
            base_hit_ratio = 0.95
        elif ws_ratio <= 1.0:
            # Working set fits with some pressure
            base_hit_ratio = 0.85
        elif ws_ratio <= 2.0:
            # Working set slightly exceeds cache - tile spilling
            base_hit_ratio = 0.70
        else:
            # Working set much larger than cache - capacity limited
            # Still get some hits from tile reuse within cache capacity
            base_hit_ratio = 0.50

        # Cold start penalty: first tile load misses cache
        if cache_is_cold:
            # Roughly 1/reuse_factor of accesses are compulsory misses
            # For MatMul with 128x128 tiles, reuse ~128x, so ~1% penalty
            # Model as 10-15% reduction for cold start
            cold_penalty = 0.15
            return max(0.0, base_hit_ratio - cold_penalty)
        return base_hit_ratio

    elif operator_type == OperatorType.LOW_REUSE:
        # MatVec, pooling, reductions
        # Limited reuse - each input element used O(1) to O(sqrt(n)) times

        if ws_ratio <= 0.5:
            # Fits easily - get some temporal reuse
            return 0.60
        elif ws_ratio <= 1.0:
            # Fits with pressure - reduced reuse
            return 0.40
        else:
            # Exceeds cache - mostly capacity misses
            # Small hit ratio from LRU keeping recent data
            return 0.10

    elif operator_type == OperatorType.STREAMING:
        # Elementwise ops: ReLU, BatchNorm, Softmax, add, mul
        # Each element accessed exactly once - no temporal reuse
        # Spatial locality gives small benefit from cache line fetches

        if ws_ratio <= 0.1:
            # Very small working set - might stay in cache
            return 0.20
        else:
            # Streaming through - minimal hits
            # Only benefit is prefetching within cache lines
            return 0.05

    # Default fallback
    return 0.50


def will_flush_cache(
    operator_type: OperatorType,
    working_set_bytes: int,
    cache_size_bytes: int,
    is_explicit_memory: bool = False,
) -> bool:
    """
    Determine if an operator will flush a cache level.

    Streaming operators with working set > cache size will evict all cached
    data, leaving the cache "cold" for the next operator.

    For explicit memory (KPU), this concept doesn't apply - the compiler
    manages data placement explicitly.

    Args:
        operator_type: Type of DNN operator
        working_set_bytes: Size of data accessed
        cache_size_bytes: Size of the cache
        is_explicit_memory: True for KPU (compiler-managed)

    Returns:
        True if this operator will flush cache contents
    """
    # Explicit memory doesn't have "flushing" - compiler manages placement
    if is_explicit_memory:
        return False

    if operator_type == OperatorType.STREAMING:
        # Streaming ops with working set > 50% of cache will effectively flush it
        return working_set_bytes > (cache_size_bytes * 0.5)

    if operator_type == OperatorType.LOW_REUSE:
        # Low reuse ops with working set > cache will also flush
        return working_set_bytes > cache_size_bytes

    # High reuse ops (tiled MatMul) are designed to NOT flush cache
    return False


# Backward compatibility aliases
def compute_l2_hit_ratio(
    operator_type: OperatorType,
    working_set_bytes: int,
    l2_cache_bytes: int = GPU_L2_CACHE_SIZES['default'],
    l2_is_cold: bool = False,
) -> float:
    """Compute L2 hit ratio (backward compatibility wrapper)."""
    return compute_cache_hit_ratio(
        operator_type=operator_type,
        working_set_bytes=working_set_bytes,
        cache_size_bytes=l2_cache_bytes,
        cache_is_cold=l2_is_cold,
        is_explicit_memory=False,
    )


def will_flush_l2(
    operator_type: OperatorType,
    working_set_bytes: int,
    l2_cache_bytes: int = GPU_L2_CACHE_SIZES['default'],
) -> bool:
    """Determine if operator will flush L2 (backward compatibility wrapper)."""
    return will_flush_cache(
        operator_type=operator_type,
        working_set_bytes=working_set_bytes,
        cache_size_bytes=l2_cache_bytes,
        is_explicit_memory=False,
    )


@dataclass
class EnergyEvent:
    """A single energy event in the execution cycle."""
    phase: CyclePhase
    description: str
    energy_pj: float  # Energy in picojoules per event
    count: int = 1    # Number of occurrences

    @property
    def total_energy_pj(self) -> float:
        return self.energy_pj * self.count


@dataclass
class CycleEnergyBreakdown:
    """Complete energy breakdown for an architecture's execution."""
    architecture_name: str
    architecture_class: str
    events: List[EnergyEvent] = field(default_factory=list)

    # Cycle counts for normalization
    num_cycles: int = 1
    ops_per_cycle: int = 1

    def add_event(self, phase: CyclePhase, description: str,
                  energy_pj: float, count: int = 1):
        """Add an energy event to the breakdown."""
        self.events.append(EnergyEvent(phase, description, energy_pj, count))

    @property
    def total_energy_pj(self) -> float:
        """Total energy for all events."""
        return sum(e.total_energy_pj for e in self.events)

    @property
    def energy_per_cycle_pj(self) -> float:
        """Energy per cycle."""
        return self.total_energy_pj / max(1, self.num_cycles)

    def get_phase_energy(self, phase: CyclePhase) -> float:
        """Get total energy for a specific phase."""
        # Handle MEMORY_ACCESS as parent category
        if phase == CyclePhase.MEMORY_ACCESS:
            mem_phases = (CyclePhase.MEM_L1, CyclePhase.MEM_L2,
                         CyclePhase.MEM_L3, CyclePhase.MEM_DRAM,
                         CyclePhase.MEM_HBM, CyclePhase.MEM_SRAM)
            return sum(e.total_energy_pj for e in self.events
                      if e.phase in mem_phases)
        return sum(e.total_energy_pj for e in self.events if e.phase == phase)

    def get_simt_overhead(self) -> float:
        """Get total SIMT overhead (GPU-specific)."""
        simt_phases = (CyclePhase.SIMT_FIXED_OVERHEAD,
                       CyclePhase.SIMT_THREAD_MGMT,
                       CyclePhase.SIMT_COHERENCE,
                       CyclePhase.SIMT_SYNC)
        return sum(e.total_energy_pj for e in self.events
                  if e.phase in simt_phases)

    def get_systolic_overhead(self) -> float:
        """Get total systolic array overhead (TPU-specific)."""
        systolic_phases = (CyclePhase.SYSTOLIC_DATA_LOAD,
                          CyclePhase.SYSTOLIC_WEIGHT_LOAD,
                          CyclePhase.SYSTOLIC_COMPUTE,
                          CyclePhase.SYSTOLIC_DRAIN,
                          CyclePhase.SYSTOLIC_CONTROL)
        return sum(e.total_energy_pj for e in self.events
                  if e.phase in systolic_phases)

    def get_spatial_overhead(self) -> float:
        """Get total spatial dataflow overhead (KPU-specific)."""
        spatial_phases = (CyclePhase.SPATIAL_CONFIG,
                         CyclePhase.SPATIAL_STREAM,
                         CyclePhase.SPATIAL_COMPUTE,
                         CyclePhase.SPATIAL_INTERCONNECT)
        return sum(e.total_energy_pj for e in self.events
                  if e.phase in spatial_phases)

    def get_compute_energy(self) -> float:
        """Get pure compute energy (ALU/MAC operations only).

        This is the actual useful work - the compute operations themselves.
        Does NOT include instruction fetch/decode, control, or data movement.

        For each architecture:
        - CPU: EXECUTE (ALU/FPU operations)
        - GPU: EXECUTE (CUDA/Tensor Core MACs)
        - TPU: SYSTOLIC_COMPUTE (systolic MACs + accumulation + activation)
        - KPU: SPATIAL_COMPUTE (domain flow MACs + accumulation + activation)
        """
        compute_phases = (
            CyclePhase.EXECUTE,
            CyclePhase.SYSTOLIC_COMPUTE,
            CyclePhase.SPATIAL_COMPUTE,
        )
        return sum(e.total_energy_pj for e in self.events
                  if e.phase in compute_phases)

    def get_control_overhead_energy(self) -> float:
        """Get control overhead energy (instruction handling, scheduling, config).

        This is the overhead for managing execution - NOT useful work.

        For each architecture:
        - CPU: INSTRUCTION_FETCH, INSTRUCTION_DECODE, OPERAND_FETCH, WRITEBACK
               (full Von Neumann instruction cycle overhead)
        - GPU: INSTRUCTION_FETCH, INSTRUCTION_DECODE, OPERAND_FETCH,
               SIMT_FIXED_OVERHEAD, SIMT_THREAD_MGMT, SIMT_SYNC
               (instruction overhead + SIMT scheduling overhead)
        - TPU: SYSTOLIC_CONTROL
               (minimal - just configuration, no per-op instruction fetch)
        - KPU: SPATIAL_CONFIG
               (minimal - domain program loaded once per layer)
        """
        control_phases = (
            # Stored program instruction cycle
            CyclePhase.INSTRUCTION_FETCH,
            CyclePhase.INSTRUCTION_DECODE,
            CyclePhase.OPERAND_FETCH,
            CyclePhase.WRITEBACK,
            # GPU SIMT scheduling overhead
            CyclePhase.SIMT_FIXED_OVERHEAD,
            CyclePhase.SIMT_THREAD_MGMT,
            CyclePhase.SIMT_SYNC,
            # TPU/KPU configuration
            CyclePhase.SYSTOLIC_CONTROL,
            CyclePhase.SPATIAL_CONFIG,
        )
        return sum(e.total_energy_pj for e in self.events
                  if e.phase in control_phases)

    def get_data_movement_energy(self) -> float:
        """Get data movement energy (memory access, transfers, coherence).

        This is the overhead for moving data - NOT useful compute work.

        For each architecture:
        - CPU: MEM_L1, MEM_L2, MEM_L3, MEM_DRAM
               (cache hierarchy access energy)
        - GPU: MEM_L1, MEM_L2, MEM_HBM, SIMT_COHERENCE
               (memory access + coherence protocol overhead)
        - TPU: SYSTOLIC_WEIGHT_LOAD, SYSTOLIC_DATA_LOAD, SYSTOLIC_DRAIN,
               MEM_SRAM, MEM_HBM
               (systolic data movement + memory access)
        - KPU: SPATIAL_STREAM, SPATIAL_INTERCONNECT, EDDO_*, MEM_DRAM
               (streaming dataflow + scratchpad access)
        """
        data_movement_phases = (
            # Memory hierarchy (all architectures)
            CyclePhase.MEM_L1,
            CyclePhase.MEM_L2,
            CyclePhase.MEM_L3,
            CyclePhase.MEM_DRAM,
            CyclePhase.MEM_HBM,
            CyclePhase.MEM_SRAM,
            # GPU coherence (part of data movement overhead)
            CyclePhase.SIMT_COHERENCE,
            # TPU systolic data movement
            CyclePhase.SYSTOLIC_WEIGHT_LOAD,
            CyclePhase.SYSTOLIC_DATA_LOAD,
            CyclePhase.SYSTOLIC_DRAIN,
            # KPU streaming dataflow
            CyclePhase.SPATIAL_STREAM,
            CyclePhase.SPATIAL_INTERCONNECT,
            # KPU EDDO scratchpads
            CyclePhase.EDDO_TILE_SCRATCHPAD,
            CyclePhase.EDDO_GLOBAL_SCRATCHPAD,
            CyclePhase.EDDO_STREAMING_BUFFER,
            CyclePhase.EDDO_DMA_SETUP,
        )
        return sum(e.total_energy_pj for e in self.events
                  if e.phase in data_movement_phases)

    def get_energy_categories(self) -> Dict[str, float]:
        """Get energy breakdown by category (compute, control, data_movement).

        Returns dict with:
        - 'compute': Pure compute energy (the useful work)
        - 'control': Control overhead (instruction handling, scheduling)
        - 'data_movement': Data movement overhead (memory, coherence, transfers)
        - 'total': Sum of all categories
        """
        compute = self.get_compute_energy()
        control = self.get_control_overhead_energy()
        data_movement = self.get_data_movement_energy()
        return {
            'compute': compute,
            'control': control,
            'data_movement': data_movement,
            'total': compute + control + data_movement,
        }
