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
