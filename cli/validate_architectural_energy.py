#!/usr/bin/env python3
"""
Architectural Energy Model Validation Tool

This tool validates and compares the energy models for different architectures
by building up a sequence of energy events that reflect each architecture's
operating cycle. The goal is to:

1. Define the basic cycle for each architecture class
2. Accumulate energy events through the cycle
3. Calculate energy per operation for fair comparison

Architecture Classes (Stored Program Machines):
- CPU: MIMD Stored Program Machine (multi-core + SIMD)
- GPU: SIMT Data Parallel (warps of 32 threads lockstep)
- DSP: VLIW with heterogeneous vector/tensor units

All three are "stored program machines" - they execute instructions
from memory, with the key difference being HOW they manage parallelism
and resource contention.

Usage:
    # Compare all stored program architectures
    ./cli/validate_architectural_energy.py

    # Compare with specific workload size
    ./cli/validate_architectural_energy.py --ops 1000 --bytes 4096

    # Show detailed cycle breakdown
    ./cli/validate_architectural_energy.py --verbose

    # Compare at different operation scales
    ./cli/validate_architectural_energy.py --sweep
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.hardware.architectural_energy import (
    StoredProgramEnergyModel,
    DataParallelEnergyModel,
    ArchitecturalEnergyBreakdown,
)


# =============================================================================
# Basic Cycle Definitions for Stored Program Machines
# =============================================================================

class CyclePhase(Enum):
    """Phases of a stored program machine's basic cycle."""
    INSTRUCTION_FETCH = "instruction_fetch"
    INSTRUCTION_DECODE = "instruction_decode"
    OPERAND_FETCH = "operand_fetch"
    EXECUTE = "execute"
    WRITEBACK = "writeback"
    # GPU-specific phases (SIMT overhead)
    SIMT_FIXED_OVERHEAD = "simt_fixed_overhead"  # Fixed infrastructure cost (SMs, schedulers)
    SIMT_THREAD_MGMT = "simt_thread_mgmt"        # Thread scheduling, warp management
    SIMT_COHERENCE = "simt_coherence"            # Cache coherence machinery
    SIMT_SYNC = "simt_sync"                      # Synchronization barriers, divergence
    # Memory access hierarchy (subcategories)
    MEMORY_ACCESS = "memory_access"              # Parent category (not used directly)
    MEM_L1 = "mem_l1"                            # L1 cache / Shared memory / Scratchpad
    MEM_L2 = "mem_l2"                            # L2 cache
    MEM_L3 = "mem_l3"                            # L3 cache (CPU only)
    MEM_DRAM = "mem_dram"                        # DRAM / HBM


class OperatingMode(Enum):
    """
    Operating modes based on where the working set resides.

    Each mode represents a scenario where data primarily lives at a specific
    level of the memory hierarchy. This enables apples-to-apples comparison
    across architectures.
    """
    L1_RESIDENT = "l1"      # Working set fits in L1/shared/scratchpad
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
    # Remainder goes to DRAM

    def __post_init__(self):
        """Validate hit ratios are in [0, 1]."""
        for name, val in [('l1_hit', self.l1_hit), ('l2_hit', self.l2_hit),
                          ('l3_hit', self.l3_hit)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


# Default hit ratios for each operating mode
DEFAULT_HIT_RATIOS = {
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
        OperatingMode.L3_RESIDENT: "L3-Resident (L1+L2 as cache, CPU only)",
        OperatingMode.DRAM_RESIDENT: "DRAM-Resident (off-chip streaming)",
    }
    return descriptions[mode]


def get_mode_memory_sizes() -> Dict[OperatingMode, Dict[str, str]]:
    """Get typical memory sizes for each mode by architecture."""
    return {
        OperatingMode.L1_RESIDENT: {
            "CPU": "32-48 KB (L1 D$)",
            "GPU": "128-228 KB (Shared Mem)",
            "DSP": "256-512 KB (Scratchpad)",
        },
        OperatingMode.L2_RESIDENT: {
            "CPU": "256 KB - 2 MB (L2)",
            "GPU": "4-60 MB (L2)",
            "DSP": "N/A (DMA prefetch)",
        },
        OperatingMode.L3_RESIDENT: {
            "CPU": "8-64 MB (LLC)",
            "GPU": "N/A",
            "DSP": "N/A",
        },
        OperatingMode.DRAM_RESIDENT: {
            "CPU": "DDR4/5 (50-100 GB/s)",
            "GPU": "HBM2/3 (2-5 TB/s)",
            "DSP": "LPDDR4/5 (25-50 GB/s)",
        },
    }


@dataclass
class EnergyEvent:
    """A single energy event in the execution cycle."""
    phase: CyclePhase
    description: str
    energy_pj: float  # Energy in picojoules
    count: int = 1    # Number of occurrences

    @property
    def total_energy_pj(self) -> float:
        return self.energy_pj * self.count


@dataclass
class CycleEnergyBreakdown:
    """Complete energy breakdown for an architecture's basic cycle."""
    architecture_name: str
    architecture_class: str
    events: List[EnergyEvent] = field(default_factory=list)

    # Cycle counts for normalization
    num_cycles: int = 1
    ops_per_cycle: int = 1

    def add_event(self, phase: CyclePhase, description: str,
                  energy_pj: float, count: int = 1):
        """Add an energy event to the cycle."""
        self.events.append(EnergyEvent(phase, description, energy_pj, count))

    @property
    def total_energy_pj(self) -> float:
        """Total energy for all events."""
        return sum(e.total_energy_pj for e in self.events)

    @property
    def energy_per_cycle_pj(self) -> float:
        """Energy per cycle."""
        return self.total_energy_pj / max(1, self.num_cycles)

    @property
    def energy_per_op_pj(self) -> float:
        """Energy per operation."""
        total_ops = self.num_cycles * self.ops_per_cycle
        return self.total_energy_pj / max(1, total_ops)

    def get_phase_energy(self, phase: CyclePhase) -> float:
        """Get total energy for a specific phase."""
        # Handle MEMORY_ACCESS as parent category for MEM_L1, MEM_L2, MEM_L3, MEM_DRAM
        if phase == CyclePhase.MEMORY_ACCESS:
            return sum(e.total_energy_pj for e in self.events
                      if e.phase in (CyclePhase.MEM_L1, CyclePhase.MEM_L2,
                                    CyclePhase.MEM_L3, CyclePhase.MEM_DRAM))
        return sum(e.total_energy_pj for e in self.events if e.phase == phase)


# =============================================================================
# CPU Basic Cycle Energy Model
# =============================================================================

def build_cpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the CPU basic cycle energy breakdown.

    Args:
        num_ops: Number of operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1, L2, L3, or DRAM resident)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        verbose: Enable verbose output

    CPU Basic Cycle (Stored Program Machine):

    +-------------------+     +------------------+     +------------------+
    |  INSTRUCTION      |     |  INSTRUCTION     |     |  DISPATCH        |
    |  FETCH            |---->|  DECODE          |---->|  (Control Sigs)  |
    |  (I-cache read)   |     |  (Logic)         |     |                  |
    |  ~1.5 pJ          |     |  ~0.8 pJ         |     |  ~0.5 pJ         |
    +-------------------+     +------------------+     +------------------+
                                                                |
                                                                v
    +-------------------+     +------------------+     +------------------+
    |  WRITEBACK        |     |  EXECUTE         |     |  OPERAND FETCH   |
    |  (Register Write) |<----|  (ALU ops)       |<----|  (Register Read) |
    |  ~3.0 pJ          |     |  ~4.0 pJ         |     |  ~2.5 pJ x 2     |
    +-------------------+     +------------------+     +------------------+
            |
            v
    +-------------------+
    |  MEMORY ACCESS    |
    |  L1 -> L2 -> L3   |
    |  -> DRAM          |
    +-------------------+

    Energy per cycle: ~15-20 pJ (varies with cache hierarchy)

    The "memory wall" is the fundamental bottleneck:
    - Register file energy is comparable to ALU energy
    - Cache hierarchy adds significant energy overhead
    - DRAM access is 20x more expensive than L1

    Memory access energy depends on operating mode:
    - L1-Resident: 100% L1 hits
    - L2-Resident: 90% L1 hits, 100% L2 hits for misses
    - L3-Resident: 85% L1 hits, 90% L2 hits, 100% L3 hits
    - DRAM-Resident: Full hierarchy traversal with hit ratios
    """
    # Get hit ratios for this mode
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

    breakdown = CycleEnergyBreakdown(
        architecture_name="CPU (Intel Xeon / AMD EPYC)",
        architecture_class="Stored Program Machine (MIMD)"
    )

    # Use the existing energy model for reference values
    cpu_model = StoredProgramEnergyModel()

    # Calculate number of instructions (assume 2 instructions per op)
    num_instructions = int(num_ops * cpu_model.instructions_per_op)
    breakdown.num_cycles = num_instructions
    breakdown.ops_per_cycle = 1  # Simplified: 1 op per cycle average

    # ==========================================================================
    # Phase 1: INSTRUCTION FETCH (from I-cache)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        "I-cache read (64B line)",
        cpu_model.instruction_fetch_energy * 1e12,  # Convert to pJ
        num_instructions
    )

    # ==========================================================================
    # Phase 2: INSTRUCTION DECODE
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        "Decode logic (x86-64 variable length)",
        cpu_model.instruction_decode_energy * 1e12,
        num_instructions
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Register File Reads)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register file read (source operand 1)",
        cpu_model.register_file_read_energy * 1e12,
        num_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register file read (source operand 2)",
        cpu_model.register_file_read_energy * 1e12,
        num_instructions
    )

    # ==========================================================================
    # Phase 4: EXECUTE (ALU operations)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "ALU/FPU operation",
        cpu_model.alu_energy_per_op * 1e12,
        num_ops
    )

    # ==========================================================================
    # Phase 5: WRITEBACK (Register File Write)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Register file write (destination)",
        cpu_model.register_file_write_energy * 1e12,
        num_instructions
    )

    # ==========================================================================
    # Phase 6: MEMORY ACCESS (Mode-dependent cache hierarchy)
    # Energy depends on which level the working set resides at
    # ==========================================================================
    cache_line_size = 64
    num_accesses = (bytes_transferred + cache_line_size - 1) // cache_line_size

    # Energy per access at each level (in pJ)
    l1_energy_per_access = cpu_model.l1_cache_energy_per_byte * cache_line_size * 1e12
    l2_energy_per_access = cpu_model.l2_cache_energy_per_byte * cache_line_size * 1e12
    l3_energy_per_access = cpu_model.l3_cache_energy_per_byte * cache_line_size * 1e12
    dram_energy_per_access = cpu_model.dram_energy_per_byte * cache_line_size * 1e12

    # Calculate accesses at each level based on hit ratios
    # All accesses go through L1
    l1_accesses = num_accesses
    l1_hits = int(l1_accesses * ratios.l1_hit)
    l1_misses = l1_accesses - l1_hits

    # L1 misses go to L2
    l2_accesses = l1_misses
    l2_hits = int(l2_accesses * ratios.l2_hit)
    l2_misses = l2_accesses - l2_hits

    # L2 misses go to L3
    l3_accesses = l2_misses
    l3_hits = int(l3_accesses * ratios.l3_hit)
    l3_misses = l3_accesses - l3_hits

    # L3 misses go to DRAM
    dram_accesses = l3_misses

    # Add events for each level (only if there are accesses)
    if l1_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_L1,
            f"L1 D-cache ({l1_hits} hits, {l1_misses} misses)",
            l1_energy_per_access,
            l1_accesses
        )

    if l2_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_L2,
            f"L2 cache ({l2_hits} hits, {l2_misses} misses)",
            l2_energy_per_access,
            l2_accesses
        )

    if l3_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_L3,
            f"L3 cache ({l3_hits} hits, {l3_misses} misses)",
            l3_energy_per_access,
            l3_accesses
        )

    if dram_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_DRAM,
            f"DRAM ({dram_accesses} accesses)",
            dram_energy_per_access,
            dram_accesses
        )

    return breakdown


# =============================================================================
# GPU Basic Cycle Energy Model
# =============================================================================

def build_gpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    concurrent_threads: int = 200_000,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the GPU basic cycle energy breakdown.

    Args:
        num_ops: Number of operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1, L2, or DRAM resident - GPU has no L3)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        concurrent_threads: Number of concurrent GPU threads
        verbose: Enable verbose output

    GPU Basic Cycle (SIMT Data Parallel):

    The GPU is STILL a stored program machine, but with SIMT execution:
    - One instruction controls 32 threads (warp)
    - Thousands of warps execute concurrently
    - Memory accesses must be coalesced for efficiency

    CRITICAL: SIMT overhead varies by operating mode!
    - L1-Resident (Shared Memory): Minimal coherence (per-SM, not coherent)
    - L2+: Full coherence machinery engaged (L2 is coherent across SMs)

    +-------------------+     +------------------+     +------------------+
    |  INSTRUCTION      |     |  INSTRUCTION     |     |  WARP            |
    |  FETCH            |---->|  DECODE          |---->|  SCHEDULING      |
    |  (per-warp)       |     |  (SIMT logic)    |     |  (~1 pJ/thread)  |
    |  ~2.0 pJ          |     |  ~0.5 pJ         |     |                  |
    +-------------------+     +------------------+     +------------------+
                                                                |
                                                                v
    +-------------------+     +------------------+     +------------------+
    |  COHERENCE        |     |  EXECUTE         |     |  REGISTER FILE   |
    |  MACHINERY        |<----|  (CUDA/Tensor)   |<----|  ACCESS          |
    |  ~5 pJ/request    |     |  ~0.3-0.8 pJ     |     |  ~0.6 pJ         |
    |  *** DOMINANT *** |     |                  |     |                  |
    +-------------------+     +------------------+     +------------------+
            |
            v
    +-------------------+
    |  MEMORY ACCESS    |
    |  Shared/L1 -> L2  |
    |  -> HBM/GDDR      |
    +-------------------+

    KEY INSIGHT: Coherence machinery dominates at small batch sizes!
    The GPU burns massive energy managing thousands of concurrent memory requests.
    """
    # Get hit ratios for this mode
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

    # GPU has no L3, so L3-resident mode falls back to DRAM-resident behavior
    # (L3 hit ratio is ignored)

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU (NVIDIA H100 / Jetson)",
        architecture_class="SIMT Data Parallel"
    )

    # Use the existing energy model for reference values
    gpu_model = DataParallelEnergyModel()

    # ==========================================================================
    # FIXED INFRASTRUCTURE OVERHEAD
    # ==========================================================================
    # GPUs have significant fixed costs that don't scale with workload:
    # - SM idle/leakage power (all 132 SMs on H100 are powered)
    # - Memory controller infrastructure (8 HBM stacks, always active)
    # - Clock distribution network
    # - Warp scheduler baseline (4 schedulers per SM, always running)
    # - L2 cache tag arrays and coherence directory (always maintained)
    #
    # This models the "GPU tax" - you pay for the infrastructure regardless
    # of how many operations you actually execute.
    #
    # For a typical kernel launch on H100:
    # - Kernel launch overhead: ~5-10 us
    # - At 1.8 GHz, that's ~9000-18000 cycles of fixed overhead
    # - At ~50 pJ/cycle for infrastructure, that's 450K-900K pJ minimum
    #
    # We model a conservative fixed overhead based on minimum kernel execution

    # Fixed overhead per kernel invocation (minimum cost to use the GPU)
    # This represents: kernel launch, SM activation, memory controller setup
    kernel_launch_overhead_pj = 100_000.0  # ~100 nJ minimum per kernel
    sm_activation_energy_pj = 5000.0       # Per-SM activation (132 SMs = 660K pJ)
    num_active_sms = min(132, max(1, num_ops // 100))  # Scale SMs with workload, min 1

    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Kernel launch overhead",
        kernel_launch_overhead_pj,
        1  # Once per kernel
    )
    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        f"SM activation ({num_active_sms} SMs)",
        sm_activation_energy_pj,
        num_active_sms
    )

    # Memory controller fixed overhead (HBM PHY, refresh, etc.)
    memory_controller_overhead_pj = 50_000.0  # ~50 nJ for memory subsystem
    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Memory controller infrastructure",
        memory_controller_overhead_pj,
        1
    )

    # Calculate execution parameters
    warp_size = 32

    # Scale thread count with workload
    # Even for small workloads, we have minimum infrastructure cost above
    effective_threads = min(concurrent_threads, num_ops * 32)  # At most 32 threads per op (1 warp)
    num_warps = max(1, effective_threads // warp_size)
    num_instructions = int(num_ops * gpu_model.instructions_per_op)

    breakdown.num_cycles = num_instructions
    breakdown.ops_per_cycle = concurrent_threads // num_warps  # Ops per warp

    # ==========================================================================
    # Phase 1: INSTRUCTION FETCH (shared across warp)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        "I-cache read (per warp, shared by 32 threads)",
        gpu_model.instruction_fetch_energy * 1e12,
        num_instructions
    )

    # ==========================================================================
    # Phase 2: INSTRUCTION DECODE (SIMT logic)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        "SIMT decode + predication",
        gpu_model.instruction_decode_energy * 1e12,
        num_instructions
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Register File)
    # GPUs have massive register files (256KB per SM on H100)
    # ==========================================================================
    num_register_accesses = num_ops * 2  # 2 reads per op
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register file access (256KB/SM)",
        gpu_model.register_file_energy_per_access * 1e12,
        num_register_accesses
    )

    # ==========================================================================
    # Phase 4: EXECUTE (CUDA Cores + Tensor Cores)
    # ==========================================================================
    # Assume 80% of MACs go to Tensor Cores (for GEMM-like ops)
    macs = num_ops
    tensor_core_macs = int(macs * gpu_model.tensor_core_utilization)
    cuda_core_macs = macs - tensor_core_macs

    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Tensor Core MACs (FP16/BF16, 4x4x4)",
        gpu_model.tensor_core_mac_energy * 1e12,
        tensor_core_macs
    )
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "CUDA Core MACs (FP32)",
        gpu_model.cuda_core_mac_energy * 1e12,
        cuda_core_macs
    )

    # ==========================================================================
    # Phase 5: SIMT THREAD MANAGEMENT
    # Managing thousands of concurrent threads has significant overhead
    # ==========================================================================

    # 5a. Warp Scheduler Energy
    # Each SM has 4 warp schedulers that select warps for execution each cycle
    # Energy: selecting from pool of eligible warps + issuing instructions
    warp_scheduler_energy_pj = 0.5  # ~0.5 pJ per warp scheduling decision
    num_scheduling_decisions = num_warps * max(1, num_instructions // num_warps)
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Warp scheduler (select eligible warps)",
        warp_scheduler_energy_pj,
        num_scheduling_decisions
    )

    # 5b. Thread State Management
    # Each thread has state: PC, registers, predicates
    # Maintaining/switching thread context has energy cost
    thread_context_energy_pj = 0.2  # ~0.2 pJ per thread context access
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Thread context management",
        thread_context_energy_pj,
        effective_threads
    )

    # 5c. Scoreboard / Dependency Tracking
    # Track which warps are ready (all operands available)
    scoreboard_energy_pj = 0.3  # ~0.3 pJ per dependency check
    num_dependency_checks = num_warps * 2  # Check before and after each op
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Scoreboard (dependency tracking)",
        scoreboard_energy_pj,
        num_dependency_checks
    )

    # ==========================================================================
    # Phase 6: SIMT COHERENCE MACHINERY
    # This is THE critical energy component for GPUs
    # Must track and order thousands of concurrent memory requests
    #
    # CRITICAL: Coherence overhead depends on operating mode!
    # - L1-Resident (Shared Memory): Minimal coherence (per-SM, not coherent)
    # - L2+: Full coherence machinery engaged (L2 is coherent across SMs)
    # ==========================================================================

    cache_line_size = 128  # H100 uses 128B cache lines
    num_memory_ops = max(1, bytes_transferred // cache_line_size)
    num_memory_requests = num_warps * num_memory_ops

    # In L1-resident mode (shared memory), coherence overhead is minimal
    # because shared memory is per-SM and not coherent across SMs
    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

    if is_l1_resident:
        # L1-Resident: Shared memory has minimal coherence overhead
        # Only need address banking logic (not full coherence)
        bank_conflict_energy_pj = 0.3  # ~0.3 pJ per bank conflict check
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Shared memory bank conflict check",
            bank_conflict_energy_pj,
            num_memory_requests
        )
    else:
        # L2+ modes: Full coherence machinery engaged

        # 6a. Memory Request Queuing
        # Each warp's memory requests go into queues
        request_queue_energy_pj = 1.0  # ~1 pJ per request enqueue/dequeue
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Memory request queuing (per warp)",
            request_queue_energy_pj,
            num_memory_requests
        )

        # 6b. Address Coalescing Logic
        # Combine adjacent addresses from threads in a warp
        coalesce_energy_pj = 0.8  # ~0.8 pJ per coalescing check
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Address coalescing logic",
            coalesce_energy_pj,
            num_memory_requests
        )

        # 6c. L1 Tag Check (per warp, parallel across threads)
        l1_tag_energy_pj = 0.5  # ~0.5 pJ per tag lookup
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "L1 cache tag lookup",
            l1_tag_energy_pj,
            num_memory_requests
        )

        # 6d. L2 Coherence Directory
        # Track which L1 caches have copies of each line
        l2_directory_energy_pj = 1.5  # ~1.5 pJ per directory lookup
        l1_miss_rate = 1.0 - ratios.l1_hit
        l2_directory_lookups = int(num_memory_requests * l1_miss_rate)
        if l2_directory_lookups > 0:
            breakdown.add_event(
                CyclePhase.SIMT_COHERENCE,
                "L2 coherence directory lookup",
                l2_directory_energy_pj,
                l2_directory_lookups
            )

        # 6e. Memory Ordering / Fence Logic
        # Ensure memory operations complete in correct order
        ordering_energy_pj = 0.3  # ~0.3 pJ per ordering check
        num_ordering_checks = num_memory_requests // 4  # One per group of 4 requests
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Memory ordering/fence logic",
            ordering_energy_pj,
            max(1, num_ordering_checks)
        )

    # ==========================================================================
    # Phase 7: SIMT SYNCHRONIZATION
    # Barriers, divergence handling, atomics
    # ==========================================================================

    # 7a. Warp Divergence Penalty
    # When threads in a warp take different branches, must serialize
    num_divergent = int(num_ops * gpu_model.warp_divergence_rate)
    divergence_mask_energy_pj = 1.0  # ~1 pJ per divergence mask update
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Warp divergence (predication masks)",
        divergence_mask_energy_pj,
        num_divergent
    )

    # 7b. Reconvergence Stack
    # Track where divergent warps should reconverge
    reconverge_energy_pj = 2.0  # ~2 pJ per reconvergence point
    num_reconverge = num_divergent // 2  # Reconverge after each divergence
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Reconvergence stack operations",
        reconverge_energy_pj,
        max(1, num_reconverge)
    )

    # 7c. Block-level Barriers (__syncthreads)
    # Synchronize all threads in a thread block
    barrier_energy_pj = 10.0  # ~10 pJ per barrier (expensive!)
    num_barriers = max(1, num_ops // 1000)  # ~1 barrier per 1000 ops
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Thread block barriers (__syncthreads)",
        barrier_energy_pj,
        num_barriers
    )

    # 7d. Atomic Operations (if any)
    # Atomics require exclusive access, very expensive
    atomic_energy_pj = 5.0  # ~5 pJ per atomic
    num_atomics = max(1, num_ops // 100)  # ~1% of ops are atomics (conservative)
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Atomic operations (exclusive access)",
        atomic_energy_pj,
        num_atomics
    )

    # ==========================================================================
    # Phase 8: MEMORY ACCESS (Mode-dependent GPU Memory Hierarchy)
    # Shared Memory/L1 (unified) -> L2 -> HBM
    # GPU has no L3, so L3 hit ratio is ignored
    # ==========================================================================
    num_accesses = max(1, bytes_transferred // 4)  # 4-byte elements

    # Energy per access at each level (in pJ, for 4-byte element)
    l1_energy_per_access = gpu_model.shared_memory_l1_unified_energy_per_byte * 4 * 1e12
    l2_energy_per_access = gpu_model.l2_cache_energy_per_byte * 4 * 1e12
    dram_energy_per_access = gpu_model.dram_energy_per_byte * 4 * 1e12

    # Calculate accesses at each level based on hit ratios
    l1_accesses = num_accesses
    l1_hits = int(l1_accesses * ratios.l1_hit)
    l1_misses = l1_accesses - l1_hits

    # L1 misses go to L2
    l2_accesses = l1_misses
    l2_hits = int(l2_accesses * ratios.l2_hit)
    l2_misses = l2_accesses - l2_hits

    # L2 misses go directly to HBM (GPU has no L3)
    dram_accesses = l2_misses

    # Shared Memory / L1
    if l1_accesses > 0:
        mem_name = "Shared Memory" if is_l1_resident else "L1 cache"
        breakdown.add_event(
            CyclePhase.MEM_L1,
            f"{mem_name} ({l1_hits} hits, {l1_misses} misses)",
            l1_energy_per_access,
            l1_accesses
        )

    # L2 cache
    if l2_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_L2,
            f"L2 cache ({l2_hits} hits, {l2_misses} misses)",
            l2_energy_per_access,
            l2_accesses
        )

    # HBM/GDDR
    if dram_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_DRAM,
            f"HBM/GDDR ({dram_accesses} accesses)",
            dram_energy_per_access,
            dram_accesses
        )

    return breakdown


# =============================================================================
# DSP (VLIW) Basic Cycle Energy Model
# =============================================================================

def build_dsp_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the DSP (VLIW) basic cycle energy breakdown.

    Args:
        num_ops: Number of operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1 or DRAM resident - DSP has no L2/L3)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        verbose: Enable verbose output

    DSP Basic Cycle (VLIW Stored Program Machine):

    VLIW = Very Long Instruction Word
    - Compiler schedules multiple operations per instruction
    - No dynamic scheduling (unlike superscalar CPU)
    - Heterogeneous units: Vector, Tensor, Scalar

    DSP Memory Model:
    - Uses software-managed scratchpad (SRAM), not hardware cache
    - No L2/L3 - data is either in scratchpad or must come from DRAM
    - L2-resident mode emulates DMA double-buffering from DRAM

    +-------------------+     +------------------+     +------------------+
    |  VLIW INSTRUCTION |     |  PARALLEL        |     |  OPERAND FETCH   |
    |  FETCH            |---->|  DECODE          |---->|  (Multi-port RF) |
    |  (256-512 bit)    |     |  (Multi-slot)    |     |                  |
    |  ~2.5 pJ          |     |  ~0.5 pJ         |     |  ~0.4 pJ x 4     |
    +-------------------+     +------------------+     +------------------+
                                                                |
                                                                v
    +-------------------+     +------------------+     +------------------+
    |  WRITEBACK        |     |  EXECUTE         |     |  (PARALLEL       |
    |  (Multi-port)     |<----|  (Vector/Tensor/ |<----|   EXECUTION)     |
    |  ~0.5 pJ x 4      |     |   Scalar)        |     |                  |
    +-------------------+     |  ~0.5-1.5 pJ     |     +------------------+
            |                 +------------------+
            v
    +-------------------+
    |  MEMORY ACCESS    |
    |  Scratchpad/SRAM  |
    |  (No cache!)      |
    |  ~1.0-2.0 pJ/B    |
    +-------------------+

    KEY INSIGHT: DSP trades dynamic flexibility for energy efficiency.
    - No branch prediction overhead (compiler static scheduling)
    - No cache coherence (scratchpad memories)
    - Lower clock frequencies (1-2 GHz vs 2-4 GHz)
    - But limited flexibility for irregular workloads
    """
    # Get hit ratios for this mode
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

    # DSP has no L2/L3, so only L1 (scratchpad) and DRAM matter
    # L2/L3 hit ratios are ignored

    breakdown = CycleEnergyBreakdown(
        architecture_name="DSP (Qualcomm Hexagon / TI C7x)",
        architecture_class="VLIW Stored Program Machine"
    )

    # DSP-specific energy values
    #
    # IMPORTANT: DSP is NOT fundamentally more efficient per operation!
    # It is designed for lower performance and lower power envelope.
    #
    # Key differences from CPU:
    # - Lower clock (1.0-1.5 GHz vs 3-4 GHz) -> ~2x lower dynamic power
    # - Lower voltage (0.7V vs 0.9V) -> ~1.5x lower power (V^2)
    # - Simpler decode (no OoO, no rename) -> saves some energy
    # - But VLIW needs MORE register ports (8+ read, 4 write)
    #
    # Net effect: DSP is ~2-3x more power efficient per Hz, but runs at
    # ~2-3x lower frequency, so energy per op is SIMILAR to CPU.
    #
    # The DSP advantage comes from:
    # 1. Software-managed scratchpad (no tag lookup energy)
    # 2. No speculation/misprediction energy
    # 3. Deterministic execution (no cache miss stalls burning power)

    vliw_width = 4  # 4 slots per VLIW bundle
    ops_per_vliw = 4  # Can issue 4 ops per cycle (ideal)

    # Energy values - CORRECTED to be realistic
    # DSP runs at ~1 GHz vs CPU at ~3.5 GHz, but VLIW has more ports
    instruction_fetch_energy_pj = 2.5   # VLIW bundles are larger (256-512 bit)
    instruction_decode_energy_pj = 0.8  # Simpler than x86 but still 4-wide
    register_read_energy_pj = 1.5       # VLIW needs 8+ read ports! Not cheaper than CPU
    register_write_energy_pj = 1.8      # 4 write ports for VLIW
    vector_op_energy_pj = 0.8           # Vector unit at lower frequency
    tensor_op_energy_pj = 0.4           # Tensor/MAC unit (optimized datapath)
    scalar_op_energy_pj = 2.0           # Scalar ops
    scratchpad_energy_per_byte_pj = 0.8 # SRAM scratchpad (no tag overhead - real advantage!)
    dram_energy_per_byte_pj = 15.0      # LPDDR (less bandwidth than DDR5)

    # Calculate execution parameters
    num_vliw_bundles = (num_ops + ops_per_vliw - 1) // ops_per_vliw

    breakdown.num_cycles = num_vliw_bundles
    breakdown.ops_per_cycle = ops_per_vliw

    # Assume typical AI workload: 70% MACs, 20% vector, 10% scalar
    mac_ops = int(num_ops * 0.70)
    vector_ops = int(num_ops * 0.20)
    scalar_ops = num_ops - mac_ops - vector_ops

    # ==========================================================================
    # Phase 1: VLIW INSTRUCTION FETCH
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        "VLIW bundle fetch (256-512 bit)",
        instruction_fetch_energy_pj,
        num_vliw_bundles
    )

    # ==========================================================================
    # Phase 2: PARALLEL DECODE (VLIW - all slots in parallel)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        "Multi-slot parallel decode (4 slots)",
        instruction_decode_energy_pj,
        num_vliw_bundles
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Multi-port Register File)
    # ==========================================================================
    # Each VLIW slot reads operands (assume 2 reads per op)
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Multi-port register file read",
        register_read_energy_pj,
        num_ops * 2
    )

    # ==========================================================================
    # Phase 4: EXECUTE (Heterogeneous Units)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Tensor/MAC unit (INT8/INT16)",
        tensor_op_energy_pj,
        mac_ops
    )
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Vector unit (SIMD)",
        vector_op_energy_pj,
        vector_ops
    )
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Scalar ALU",
        scalar_op_energy_pj,
        scalar_ops
    )

    # ==========================================================================
    # Phase 5: WRITEBACK (Multi-port Register File)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Multi-port register file write",
        register_write_energy_pj,
        num_ops
    )

    # ==========================================================================
    # Phase 6: MEMORY ACCESS (Mode-dependent Scratchpad/DRAM)
    # ==========================================================================
    # DSPs use software-managed scratchpads, not hardware caches
    # This eliminates tag/coherence overhead but requires explicit DMA
    #
    # DSP Memory Model:
    # - L1-Resident: 100% scratchpad (data fits in SRAM)
    # - L2-Resident: Emulated via DMA double-buffering from DRAM
    #   (prefetch next block while processing current)
    # - L3-Resident: N/A (DSP has no L3, falls back to DRAM behavior)
    # - DRAM-Resident: Streaming from DRAM with DMA prefetch

    # For DSP, we interpret hit ratios differently:
    # - l1_hit = fraction served from scratchpad (already in SRAM)
    # - Everything else comes from DRAM via DMA
    # DSP has no L2/L3, so we use l1_hit directly

    scratchpad_bytes = int(bytes_transferred * ratios.l1_hit)
    dram_bytes = bytes_transferred - scratchpad_bytes

    # Determine mode-specific description
    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

    if scratchpad_bytes > 0:
        if is_l1_resident:
            desc = "Scratchpad SRAM (100% resident)"
        else:
            desc = f"Scratchpad SRAM ({int(ratios.l1_hit * 100)}% hit, DMA prefetch)"
        breakdown.add_event(
            CyclePhase.MEM_L1,
            desc,
            scratchpad_energy_per_byte_pj,
            scratchpad_bytes
        )

    if dram_bytes > 0:
        if mode == OperatingMode.L2_RESIDENT:
            desc = f"DRAM via DMA double-buffer ({dram_bytes} bytes)"
        else:
            desc = f"DRAM/DDR streaming ({dram_bytes} bytes)"
        breakdown.add_event(
            CyclePhase.MEM_DRAM,
            desc,
            dram_energy_per_byte_pj,
            dram_bytes
        )

    return breakdown


# =============================================================================
# Reporting and Comparison
# =============================================================================

def format_phase_breakdown(breakdown: CycleEnergyBreakdown) -> str:
    """Format the phase-by-phase energy breakdown."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {breakdown.architecture_name}")
    lines.append(f"  Class: {breakdown.architecture_class}")
    lines.append(f"{'='*70}")

    # Group events by phase
    phases = [
        CyclePhase.INSTRUCTION_FETCH,
        CyclePhase.INSTRUCTION_DECODE,
        CyclePhase.OPERAND_FETCH,
        CyclePhase.EXECUTE,
        CyclePhase.WRITEBACK,
        CyclePhase.SIMT_THREAD_MGMT,
        CyclePhase.SIMT_COHERENCE,
        CyclePhase.SIMT_SYNC,
        CyclePhase.MEMORY_ACCESS,
    ]

    for phase in phases:
        phase_events = [e for e in breakdown.events if e.phase == phase]
        if not phase_events:
            continue

        phase_total = sum(e.total_energy_pj for e in phase_events)
        lines.append(f"\n  {phase.value.upper().replace('_', ' ')} PHASE: {phase_total:.2f} pJ")
        lines.append(f"  {'-'*50}")

        for event in phase_events:
            energy_str = f"{event.total_energy_pj:.2f} pJ"
            lines.append(f"    {event.description:<45} {energy_str:>12}")
            if event.count > 1:
                lines.append(f"      ({event.count:,} x {event.energy_pj:.3f} pJ)")

    lines.append(f"\n  {'='*50}")
    lines.append(f"  TOTAL ENERGY: {breakdown.total_energy_pj:.2f} pJ")
    lines.append(f"  CYCLES: {breakdown.num_cycles:,}")
    lines.append(f"  ENERGY/CYCLE: {breakdown.energy_per_cycle_pj:.3f} pJ")
    lines.append(f"  ENERGY/OP: {breakdown.energy_per_op_pj:.4f} pJ")

    return "\n".join(lines)


def format_comparison_table(breakdowns: List[CycleEnergyBreakdown],
                           mode: Optional[OperatingMode] = None,
                           num_ops: int = 1000) -> str:
    """Format a comparison table across architectures."""
    lines = []
    lines.append("\n" + "="*90)
    if mode:
        lines.append(f"  STORED PROGRAM MACHINE ENERGY COMPARISON - {mode.value.upper()} Mode")
    else:
        lines.append("  STORED PROGRAM MACHINE ENERGY COMPARISON")
    lines.append("="*90)

    # Header
    lines.append(f"\n  {'Architecture':<25} {'Total (pJ)':<15} {'Per Cycle':<15} {'Per Op':<15} {'Relative':<12}")
    lines.append(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")

    # Find baseline (CPU) for relative comparison
    baseline_energy = breakdowns[0].total_energy_pj if breakdowns else 1.0

    for breakdown in breakdowns:
        total = breakdown.total_energy_pj
        per_cycle = breakdown.energy_per_cycle_pj
        # Use actual ops count for per-op calculation (not the derived cycle-based value)
        per_op = total / num_ops
        relative = total / baseline_energy

        lines.append(f"  {breakdown.architecture_name[:25]:<25} "
                    f"{total:>12.2f} pJ "
                    f"{per_cycle:>12.3f} pJ "
                    f"{per_op:>12.4f} pJ "
                    f"{relative:>10.2f}x")

    # Phase breakdown comparison
    lines.append(f"\n  PHASE BREAKDOWN")
    lines.append(f"  {'-'*78}")

    # Column width for each architecture's data
    COL_WIDTH = 22

    # Define phases with indentation level
    # (display_name, phase_or_None, indent_level, is_parent_category)
    # phase_or_None: CyclePhase for leaf nodes, None for parent categories
    # is_parent_category: True means calculate sum of children
    phases = [
        ("Instruction Fetch",  CyclePhase.INSTRUCTION_FETCH,  0, False),
        ("Instruction Decode", CyclePhase.INSTRUCTION_DECODE, 0, False),
        ("Operand Fetch",      CyclePhase.OPERAND_FETCH,      0, False),
        ("Execute",            CyclePhase.EXECUTE,            0, False),
        ("Writeback",          CyclePhase.WRITEBACK,          0, False),
        ("SIMT Overhead",      "SIMT_TOTAL",                  0, True),
        ("Fixed Infra",        CyclePhase.SIMT_FIXED_OVERHEAD, 1, False),
        ("Thread Mgmt",        CyclePhase.SIMT_THREAD_MGMT,   1, False),
        ("Coherence",          CyclePhase.SIMT_COHERENCE,     1, False),
        ("Sync",               CyclePhase.SIMT_SYNC,          1, False),
        ("Memory Access",      CyclePhase.MEMORY_ACCESS,      0, True),
        ("L1/Scratchpad",      CyclePhase.MEM_L1,             1, False),
        ("L2",                 CyclePhase.MEM_L2,             1, False),
        ("L3",                 CyclePhase.MEM_L3,             1, False),
        ("DRAM/HBM",           CyclePhase.MEM_DRAM,           1, False),
    ]

    # Header row
    header = f"  {'Phase':<22}"
    for b in breakdowns:
        arch_name = b.architecture_name.split()[0]
        header += f" {arch_name:>{COL_WIDTH}}"
    lines.append(header)

    # Separator
    sep = f"  {'-'*22}"
    for _ in breakdowns:
        sep += f" {'-'*COL_WIDTH}"
    lines.append(sep)

    def format_energy_cell(energy_pj: float, total_pj: float) -> str:
        """Format a single cell with energy and percentage."""
        if energy_pj == 0:
            return "n/a"
        pct = (energy_pj / total_pj * 100) if total_pj > 0 else 0
        return f"{energy_pj:,.0f} pJ ({pct:4.1f}%)"

    for phase_name, phase, indent, is_parent in phases:
        # Build display name with indentation
        indent_str = "  " * indent
        display_name = f"{indent_str}{phase_name}"

        # Build row
        row = f"  {display_name:<22}"

        for breakdown in breakdowns:
            # Calculate energy for this phase
            if phase == "SIMT_TOTAL":
                # Special case: sum of all SIMT phases (including fixed overhead)
                energy = (
                    breakdown.get_phase_energy(CyclePhase.SIMT_FIXED_OVERHEAD) +
                    breakdown.get_phase_energy(CyclePhase.SIMT_THREAD_MGMT) +
                    breakdown.get_phase_energy(CyclePhase.SIMT_COHERENCE) +
                    breakdown.get_phase_energy(CyclePhase.SIMT_SYNC)
                )
            elif isinstance(phase, CyclePhase):
                energy = breakdown.get_phase_energy(phase)
            else:
                energy = 0

            cell = format_energy_cell(energy, breakdown.total_energy_pj)
            row += f" {cell:>{COL_WIDTH}}"

        lines.append(row)

    return "\n".join(lines)


def format_key_insights(breakdowns: List[CycleEnergyBreakdown]) -> str:
    """Format key insights from the comparison."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("  KEY INSIGHTS")
    lines.append("="*70)

    if len(breakdowns) >= 3:
        cpu, gpu, dsp = breakdowns[0], breakdowns[1], breakdowns[2]

        # Calculate GPU SIMT overhead breakdown
        simt_fixed = gpu.get_phase_energy(CyclePhase.SIMT_FIXED_OVERHEAD)
        simt_thread_mgmt = gpu.get_phase_energy(CyclePhase.SIMT_THREAD_MGMT)
        simt_coherence = gpu.get_phase_energy(CyclePhase.SIMT_COHERENCE)
        simt_sync = gpu.get_phase_energy(CyclePhase.SIMT_SYNC)
        simt_total = simt_fixed + simt_thread_mgmt + simt_coherence + simt_sync
        simt_pct = (simt_total / gpu.total_energy_pj * 100) if gpu.total_energy_pj > 0 else 0

        lines.append(f"""
  1. CPU (Stored Program Machine):
     - High instruction fetch/decode overhead per operation
     - Cache hierarchy adds significant memory access energy
     - Register file energy is comparable to ALU energy
     - Good for: General-purpose computing, irregular workloads

  2. GPU (SIMT Data Parallel):
     - Instruction overhead amortized across 32 threads (warp)
     - BUT: SIMT overhead is THE dominant energy cost ({simt_pct:.1f}% of total!)

     GPU SIMT OVERHEAD BREAKDOWN:
       Fixed Infra:       {simt_fixed:>12,.1f} pJ (kernel launch, SM activation, mem ctrl)
       Thread Management: {simt_thread_mgmt:>12,.1f} pJ (warp scheduling, scoreboard)
       Coherence:         {simt_coherence:>12,.1f} pJ (queuing, coalescing, tags, directory)
       Synchronization:   {simt_sync:>12,.1f} pJ (divergence, barriers, atomics)
       ---------------------------------------------------------
       TOTAL SIMT:        {simt_total:>12,.1f} pJ

     Compare to CPU writeback: {cpu.get_phase_energy(CyclePhase.WRITEBACK):.1f} pJ
     Good for: Large batch sizes where SIMT overhead is amortized

  3. DSP (VLIW):
     - VLIW bundles amortize instruction fetch across 4 ops
     - No dynamic scheduling = simpler, lower energy
     - Scratchpad memories eliminate cache tag overhead
     - Good for: Signal processing, known data access patterns

  FUNDAMENTAL INSIGHT:
  All three are stored program machines, but they trade off flexibility
  vs. efficiency differently:

    CPU: Maximum flexibility, highest overhead per op
    GPU: Data parallelism amortizes instruction cost, but SIMT overhead dominates
    DSP: Static scheduling + scratchpads = lowest overhead, least flexible
""")

    return "\n".join(lines)


def print_architecture_diagrams() -> None:
    """Print ASCII art diagrams showing the basic cycle for each architecture."""
    print("""
================================================================================
  STORED PROGRAM MACHINE BASIC CYCLES
================================================================================

  All three architectures are "stored program machines" that execute
  instructions from memory. The key difference is HOW they manage
  parallelism and resource contention.

--------------------------------------------------------------------------------
  CPU Basic Cycle (MIMD Stored Program Machine)
--------------------------------------------------------------------------------

  +-------------------+     +------------------+     +------------------+
  |  INSTRUCTION      |     |  INSTRUCTION     |     |  DISPATCH        |
  |  FETCH            |---->|  DECODE          |---->|  (Control Sigs)  |
  |  (I-cache read)   |     |  (x86-64 logic)  |     |                  |
  |  ~1.5 pJ          |     |  ~0.8 pJ         |     |  ~0.5 pJ         |
  +-------------------+     +------------------+     +------------------+
                                                              |
                                                              v
  +-------------------+     +------------------+     +------------------+
  |  WRITEBACK        |     |  EXECUTE         |     |  OPERAND FETCH   |
  |  (Register Write) |<----|  (ALU/FPU)       |<----|  (Register Read) |
  |  ~3.0 pJ          |     |  ~4.0 pJ         |     |  ~2.5 pJ x 2     |
  +-------------------+     +------------------+     +------------------+
          |
          v
  +-----------------------------------------------+
  |  MEMORY ACCESS (Cache Hierarchy)              |
  |  L1 (1.0 pJ/B) -> L2 (2.5 pJ/B) -> L3 (5 pJ/B)|
  |                 -> DRAM (20 pJ/B)              |
  +-----------------------------------------------+

  KEY: ~15-20 pJ per cycle (dominated by memory hierarchy)

--------------------------------------------------------------------------------
  GPU Basic Cycle (SIMT Data Parallel)
--------------------------------------------------------------------------------

  +-------------------+     +------------------+     +------------------+
  |  INSTRUCTION      |     |  INSTRUCTION     |     |  WARP            |
  |  FETCH            |---->|  DECODE          |---->|  SCHEDULING      |
  |  (per-warp)       |     |  (SIMT logic)    |     |  (~1 pJ/thread)  |
  |  ~2.0 pJ          |     |  ~0.5 pJ         |     |  HUGE @ 200K!    |
  +-------------------+     +------------------+     +------------------+
                                                              |
                                                              v
  +-------------------+     +------------------+     +------------------+
  |  COHERENCE        |     |  EXECUTE         |     |  REGISTER FILE   |
  |  MACHINERY        |<----|  (CUDA/Tensor)   |<----|  ACCESS          |
  |  ~5 pJ/request    |     |  ~0.3-0.8 pJ     |     |  ~0.6 pJ         |
  |  *** DOMINANT *** |     |                  |     |  (256KB/SM)      |
  +-------------------+     +------------------+     +------------------+
          |
          v
  +-----------------------------------------------+
  |  MEMORY ACCESS (GPU Hierarchy)                |
  |  Shared/L1 (0.25 pJ/B) -> L2 (0.8 pJ/B)       |
  |                       -> HBM (10 pJ/B)        |
  +-----------------------------------------------+

  KEY: Coherence machinery dominates at small batch sizes!
       GPU burns massive energy managing concurrent memory requests.

--------------------------------------------------------------------------------
  DSP Basic Cycle (VLIW Stored Program Machine)
--------------------------------------------------------------------------------

  +-------------------+     +------------------+     +------------------+
  |  VLIW INSTRUCTION |     |  PARALLEL        |     |  OPERAND FETCH   |
  |  FETCH            |---->|  DECODE          |---->|  (Multi-port RF) |
  |  (256-512 bit)    |     |  (4 slots)       |     |                  |
  |  ~2.5 pJ          |     |  ~0.5 pJ         |     |  ~0.4 pJ x 4     |
  +-------------------+     +------------------+     +------------------+
                                                              |
                                                              v
                            +----------------------------------+
                            |  PARALLEL EXECUTE (4 slots)      |
                            |  +--------+ +--------+ +-------+ |
                            |  | Tensor | | Vector | | Scalar| |
                            |  | 0.3 pJ | | 0.5 pJ | | 1.5 pJ| |
                            |  +--------+ +--------+ +-------+ |
                            +----------------------------------+
                                          |
                                          v
  +-----------------------------------------------+
  |  MEMORY ACCESS (Scratchpad - No Cache!)       |
  |  Scratchpad SRAM (~1.0 pJ/B) -> DRAM (~15 pJ/B)|
  |  No tag overhead, software-managed            |
  +-----------------------------------------------+

  KEY: VLIW + scratchpad = lowest overhead, but least flexible

================================================================================
  WHY THESE DIFFERENCES MATTER
================================================================================

  1. INSTRUCTION OVERHEAD:
     CPU: 1 instruction per ~0.5 ops (high decode complexity)
     GPU: 1 instruction per 32 threads (warp), but coherence dominates
     DSP: 1 instruction per 4 ops (VLIW parallelism)

  2. MEMORY ACCESS:
     CPU: Hardware caches (tag lookup + coherence)
     GPU: Massive coherence machinery (thousands of concurrent requests)
     DSP: Software scratchpads (no tag overhead, explicit DMA)

  3. SCHEDULING:
     CPU: Out-of-order dynamic scheduling (complex, energy hungry)
     GPU: Hardware thread scheduling + warp divergence penalties
     DSP: Compiler static scheduling (simple, energy efficient)

================================================================================
""")


def run_sweep(mode: OperatingMode = OperatingMode.DRAM_RESIDENT, verbose: bool = False) -> None:
    """Run a sweep across different operation scales."""
    print("\n" + "="*100)
    print(f"  ENERGY SCALING ANALYSIS (Stored Program Machines) - {mode.value.upper()} Mode")
    print("="*100)

    scales = [100, 1000, 10000, 100000, 1000000]
    bytes_per_op = 4  # 4 bytes per operation (typical)

    # Collect results for both tables
    results = []
    for ops in scales:
        bytes_transferred = ops * bytes_per_op

        cpu = build_cpu_cycle_energy(ops, bytes_transferred, mode=mode)
        gpu = build_gpu_cycle_energy(ops, bytes_transferred, mode=mode)
        dsp = build_dsp_cycle_energy(ops, bytes_transferred, mode=mode)

        # Calculate actual energy per operation (total / actual ops requested)
        # NOT using the internal energy_per_op_pj which uses derived cycle counts
        results.append({
            'ops': ops,
            'cpu_total': cpu.total_energy_pj,
            'gpu_total': gpu.total_energy_pj,
            'dsp_total': dsp.total_energy_pj,
            'cpu_per_op': cpu.total_energy_pj / ops,
            'gpu_per_op': gpu.total_energy_pj / ops,
            'dsp_per_op': dsp.total_energy_pj / ops,
        })

    # ==========================================================================
    # TABLE 1: AMORTIZED ENERGY PER OPERATION (the key metric!)
    # ==========================================================================
    print(f"\n  TABLE 1: AMORTIZED ENERGY PER OPERATION")
    print(f"  {'-'*90}")
    print(f"  {'Operations':<12} {'CPU (pJ/op)':<15} {'GPU (pJ/op)':<15} {'DSP (pJ/op)':<15} {'Best':<12}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")

    for r in results:
        energies = {'CPU': r['cpu_per_op'], 'GPU': r['gpu_per_op'], 'DSP': r['dsp_per_op']}
        best = min(energies, key=energies.get)

        print(f"  {r['ops']:<12,} {r['cpu_per_op']:>12.2f} pJ "
              f"{r['gpu_per_op']:>12.2f} pJ "
              f"{r['dsp_per_op']:>12.2f} pJ "
              f"{best:>10}")

    # ==========================================================================
    # TABLE 2: TOTAL ENERGY
    # ==========================================================================
    print(f"\n  TABLE 2: TOTAL ENERGY")
    print(f"  {'-'*90}")
    print(f"  {'Operations':<12} {'CPU (pJ)':<15} {'GPU (pJ)':<15} {'DSP (pJ)':<15} {'GPU/CPU':<10} {'DSP/CPU':<10}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*10} {'-'*10}")

    for r in results:
        gpu_ratio = r['gpu_total'] / r['cpu_total']
        dsp_ratio = r['dsp_total'] / r['cpu_total']

        print(f"  {r['ops']:<12,} {r['cpu_total']:>12,.0f} pJ "
              f"{r['gpu_total']:>12,.0f} pJ "
              f"{r['dsp_total']:>12,.0f} pJ "
              f"{gpu_ratio:>8.2f}x "
              f"{dsp_ratio:>8.2f}x")

    print(f"""

  OBSERVATION:
  - GPU energy/op is HIGH at small scales due to fixed SIMT overhead
    (warp schedulers, coherence machinery, memory controllers run regardless)
  - GPU energy/op improves at large scales as overhead amortizes across more ops
  - DSP maintains consistent low energy/op (VLIW + scratchpad = minimal overhead)
  - CPU has moderate, consistent energy/op across all scales

  KEY INSIGHT: GPU is INEFFICIENT for small workloads!
  The massive parallel machinery (132 SMs on H100, 16K+ CUDA cores) consumes
  energy even when only a few operations are needed. This is the "GPU tax"
  for data parallelism - you pay for the infrastructure whether you use it or not.
""")


def run_mode_sweep(num_ops: int = 1000, bytes_transferred: int = 4096,
                   verbose: bool = False) -> None:
    """Run a sweep across all operating modes to compare architectures."""
    print("\n" + "="*100)
    print("  OPERATING MODE COMPARISON")
    print("  Comparing energy across L1-Resident, L2-Resident, L3-Resident, and DRAM-Resident modes")
    print("="*100)
    print(f"  Workload: {num_ops:,} ops, {bytes_transferred:,} bytes")
    print()

    modes = [
        OperatingMode.L1_RESIDENT,
        OperatingMode.L2_RESIDENT,
        OperatingMode.L3_RESIDENT,
        OperatingMode.DRAM_RESIDENT,
    ]

    # Header
    print(f"  {'Mode':<20} {'CPU (pJ)':<18} {'GPU (pJ)':<18} {'DSP (pJ)':<18} {'Notes':<30}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18} {'-'*30}")

    # Collect results for analysis
    results = []

    for mode in modes:
        cpu = build_cpu_cycle_energy(num_ops, bytes_transferred, mode=mode)
        gpu = build_gpu_cycle_energy(num_ops, bytes_transferred, mode=mode)
        dsp = build_dsp_cycle_energy(num_ops, bytes_transferred, mode=mode)

        # Determine notes based on mode
        notes = ""
        if mode == OperatingMode.L1_RESIDENT:
            notes = "Best case: all data on-chip"
        elif mode == OperatingMode.L2_RESIDENT:
            notes = "GPU coherence active"
        elif mode == OperatingMode.L3_RESIDENT:
            notes = "CPU only (GPU/DSP->DRAM)"
        else:
            notes = "Streaming from memory"

        # Mark N/A for modes that don't apply
        cpu_str = f"{cpu.total_energy_pj:>14,.0f} pJ"
        gpu_str = f"{gpu.total_energy_pj:>14,.0f} pJ"
        dsp_str = f"{dsp.total_energy_pj:>14,.0f} pJ"

        # L3 mode doesn't really apply to GPU/DSP (they go to DRAM)
        if mode == OperatingMode.L3_RESIDENT:
            gpu_str = f"{gpu.total_energy_pj:>14,.0f} pJ*"
            dsp_str = f"{dsp.total_energy_pj:>14,.0f} pJ*"

        print(f"  {get_mode_description(mode):<20} {cpu_str:<18} {gpu_str:<18} {dsp_str:<18} {notes:<30}")

        results.append({
            'mode': mode,
            'cpu': cpu.total_energy_pj,
            'gpu': gpu.total_energy_pj,
            'dsp': dsp.total_energy_pj,
        })

    print()
    print("  * GPU and DSP have no L3 cache - L3 mode uses same hit ratios but data goes to DRAM")
    print()

    # Analysis: which architecture wins in each mode?
    print("  WINNER BY MODE:")
    print(f"  {'-'*60}")
    for r in results:
        energies = {'CPU': r['cpu'], 'GPU': r['gpu'], 'DSP': r['dsp']}
        winner = min(energies, key=energies.get)
        winner_energy = energies[winner]
        runner_up = sorted(energies.values())[1]
        ratio = runner_up / winner_energy if winner_energy > 0 else 0

        print(f"  {get_mode_description(r['mode']):<30} -> {winner} ({ratio:.1f}x more efficient)")

    print()
    print(f"""
  KEY INSIGHTS:
  1. L1-Resident Mode: This is where GPU shared memory shines - minimal coherence
  2. L2-Resident Mode: GPU coherence overhead becomes significant
  3. L3-Resident Mode: Only CPU has L3 cache - advantage to CPU
  4. DRAM-Resident Mode: GPU's HBM bandwidth helps, but coherence still hurts

  RECOMMENDATION:
  - Small kernels (L1-fitting): Consider GPU shared memory approach
  - Medium kernels (L2-fitting): CPU may be competitive due to lower coherence
  - Large kernels (L3-fitting): CPU has clear advantage (only one with L3)
  - Streaming (DRAM): Depends on memory bandwidth requirements
""")


def main():
    parser = argparse.ArgumentParser(
        description='Validate and compare architectural energy models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (DRAM-resident mode, default)
  %(prog)s

  # Compare in L1-resident mode (all data on-chip)
  %(prog)s --mode l1

  # Compare in L2-resident mode (with cache hit ratios)
  %(prog)s --mode l2

  # Custom workload size
  %(prog)s --ops 10000 --bytes 40960

  # Show architecture diagrams
  %(prog)s --diagram

  # Detailed breakdown
  %(prog)s --verbose

  # Operation scaling sweep
  %(prog)s --sweep

  # Mode comparison (all modes side-by-side)
  %(prog)s --mode-sweep

  # Custom L1 hit ratio
  %(prog)s --mode l2 --l1-hit-rate 0.92

  # JSON output
  %(prog)s --output results.json
"""
    )

    parser.add_argument('--ops', type=int, default=1000,
                        help='Number of operations (default: 1000)')
    parser.add_argument('--bytes', type=int, default=4096,
                        help='Bytes transferred (default: 4096)')
    parser.add_argument('--threads', type=int, default=200_000,
                        help='GPU concurrent threads (default: 200000)')
    parser.add_argument('--mode', type=str, default='dram',
                        choices=['l1', 'l2', 'l3', 'dram'],
                        help='Operating mode (default: dram)')
    parser.add_argument('--l1-hit-rate', type=float,
                        help='Custom L1 hit rate (0.0-1.0)')
    parser.add_argument('--l2-hit-rate', type=float,
                        help='Custom L2 hit rate (0.0-1.0)')
    parser.add_argument('--l3-hit-rate', type=float,
                        help='Custom L3 hit rate (0.0-1.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed phase breakdown')
    parser.add_argument('--diagram', '-d', action='store_true',
                        help='Show architecture cycle diagrams')
    parser.add_argument('--sweep', action='store_true',
                        help='Run scaling sweep across operation counts')
    parser.add_argument('--mode-sweep', action='store_true',
                        help='Compare energy across all operating modes')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (JSON format)')

    args = parser.parse_args()

    # Parse operating mode
    mode_map = {
        'l1': OperatingMode.L1_RESIDENT,
        'l2': OperatingMode.L2_RESIDENT,
        'l3': OperatingMode.L3_RESIDENT,
        'dram': OperatingMode.DRAM_RESIDENT,
    }
    mode = mode_map[args.mode]

    # Build custom hit ratios if specified
    hit_ratios = None
    if args.l1_hit_rate is not None or args.l2_hit_rate is not None or args.l3_hit_rate is not None:
        defaults = DEFAULT_HIT_RATIOS[mode]
        hit_ratios = HitRatios(
            l1_hit=args.l1_hit_rate if args.l1_hit_rate is not None else defaults.l1_hit,
            l2_hit=args.l2_hit_rate if args.l2_hit_rate is not None else defaults.l2_hit,
            l3_hit=args.l3_hit_rate if args.l3_hit_rate is not None else defaults.l3_hit,
        )

    # Show diagrams if requested
    if args.diagram:
        print_architecture_diagrams()

    # Run mode sweep if requested
    if args.mode_sweep:
        run_mode_sweep(args.ops, args.bytes, args.verbose)
        return

    print("="*70)
    print("  ARCHITECTURAL ENERGY MODEL VALIDATION")
    print("  Stored Program Machines: CPU, GPU, DSP (VLIW)")
    print("="*70)
    print(f"\n  Workload: {args.ops:,} operations, {args.bytes:,} bytes")
    print(f"  Mode: {get_mode_description(mode)}")
    if hit_ratios:
        print(f"  Custom hit ratios: L1={hit_ratios.l1_hit:.0%}, L2={hit_ratios.l2_hit:.0%}, L3={hit_ratios.l3_hit:.0%}")
    else:
        defaults = DEFAULT_HIT_RATIOS[mode]
        print(f"  Default hit ratios: L1={defaults.l1_hit:.0%}, L2={defaults.l2_hit:.0%}, L3={defaults.l3_hit:.0%}")

    # Build cycle energy breakdowns
    cpu_breakdown = build_cpu_cycle_energy(args.ops, args.bytes, mode=mode, hit_ratios=hit_ratios, verbose=args.verbose)
    gpu_breakdown = build_gpu_cycle_energy(args.ops, args.bytes, mode=mode, hit_ratios=hit_ratios,
                                            concurrent_threads=args.threads, verbose=args.verbose)
    dsp_breakdown = build_dsp_cycle_energy(args.ops, args.bytes, mode=mode, hit_ratios=hit_ratios, verbose=args.verbose)

    breakdowns = [cpu_breakdown, gpu_breakdown, dsp_breakdown]

    # Print detailed breakdown if verbose
    if args.verbose:
        for breakdown in breakdowns:
            print(format_phase_breakdown(breakdown))

    # Print comparison table
    print(format_comparison_table(breakdowns, mode=mode, num_ops=args.ops))

    # Print insights
    print(format_key_insights(breakdowns))

    # Run sweep if requested
    if args.sweep:
        run_sweep(mode=mode, verbose=args.verbose)

    # Output JSON if requested
    if args.output:
        import json

        # Get the effective hit ratios
        effective_ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

        output_data = {
            "workload": {
                "ops": args.ops,
                "bytes": args.bytes,
            },
            "mode": {
                "name": mode.value,
                "description": get_mode_description(mode),
                "hit_ratios": {
                    "l1_hit": effective_ratios.l1_hit,
                    "l2_hit": effective_ratios.l2_hit,
                    "l3_hit": effective_ratios.l3_hit,
                }
            },
            "architectures": []
        }

        for breakdown in breakdowns:
            arch_data = {
                "name": breakdown.architecture_name,
                "class": breakdown.architecture_class,
                "total_energy_pj": breakdown.total_energy_pj,
                "energy_per_cycle_pj": breakdown.energy_per_cycle_pj,
                "energy_per_op_pj": breakdown.energy_per_op_pj,
                "phases": {}
            }

            for phase in CyclePhase:
                phase_energy = breakdown.get_phase_energy(phase)
                arch_data["phases"][phase.value] = phase_energy

            output_data["architectures"].append(arch_data)

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Output saved to: {args.output}")


if __name__ == '__main__':
    main()
