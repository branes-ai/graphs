"""
CPU Cycle-Level Energy Model

Models energy consumption for a stored program machine (MIMD) architecture
like Intel Xeon or AMD EPYC processors.

CPU Basic Cycle:
  INSTRUCTION FETCH -> DECODE -> OPERAND FETCH -> EXECUTE -> WRITEBACK
                                                      |
                                                      v
                                              MEMORY ACCESS
                                          (L1 -> L2 -> L3 -> DRAM)

Key characteristics:
- Complex x86-64 decode (variable length instructions)
- Out-of-order execution with register renaming
- Deep cache hierarchy (L1/L2/L3)
- High frequency (3-4 GHz) but high per-op energy
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graphs.hardware.technology_profile import TechnologyProfile

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
)


# NOTE: Energy parameters are now REQUIRED via TechnologyProfile.
# The hardcoded CPU_ENERGY_PARAMS dict has been removed to eliminate
# dual sources of energy definitions. Use a TechnologyProfile instance
# (e.g., DEFAULT_PROFILE from technology_profile.py) for all energy values.


def build_cpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    tech_profile: 'TechnologyProfile' = None,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the CPU basic cycle energy breakdown.

    Args:
        num_ops: Number of operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1, L2, L3, or DRAM resident)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    CPU Energy Model:
    - Instruction overhead scales with 2 instructions per op (x86 CISC)
    - Register file energy is significant (multi-port, high frequency)
    - Cache hierarchy provides energy filtering (hit ratios)
    - Memory access energy depends on operating mode

    Technology Profile (REQUIRED):
        A TechnologyProfile must be provided to specify energy parameters.
        Use DEFAULT_PROFILE for typical datacenter values.

        Example:
            from graphs.hardware.technology_profile import DATACENTER_4NM_HBM3
            breakdown = build_cpu_cycle_energy(
                num_ops=1000,
                tech_profile=DATACENTER_4NM_HBM3
            )

    Raises:
        ValueError: If tech_profile is not provided.
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )
    # Get hit ratios for this mode
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

    breakdown = CycleEnergyBreakdown(
        architecture_name="CPU (Intel Xeon / AMD EPYC)",
        architecture_class="Stored Program Machine (MIMD)"
    )

    # Derive all energy parameters from technology profile
    params = {
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'instructions_per_op': 2.0,  # x86 CISC overhead (fixed)
        'register_read_pj': tech_profile.register_read_energy_pj,
        'register_write_pj': tech_profile.register_write_energy_pj,
        'alu_energy_pj': tech_profile.base_alu_energy_pj,
        'l1_cache_pj_per_byte': tech_profile.l1_cache_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'l3_cache_pj_per_byte': tech_profile.l3_cache_energy_per_byte_pj,
        'dram_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,
    }

    # Calculate number of instructions (x86 CISC overhead)
    num_instructions = int(num_ops * params['instructions_per_op'])
    breakdown.num_cycles = num_instructions
    breakdown.ops_per_cycle = 1

    # ==========================================================================
    # Phase 1: INSTRUCTION FETCH (from I-cache)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        "I-cache read (64B line)",
        params['instruction_fetch_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 2: INSTRUCTION DECODE
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        "Decode logic (x86-64 variable length)",
        params['instruction_decode_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Register File Reads)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register file read (source operand 1)",
        params['register_read_pj'],
        num_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register file read (source operand 2)",
        params['register_read_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 4: EXECUTE (ALU operations)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "ALU/FPU operation",
        params['alu_energy_pj'],
        num_ops
    )

    # ==========================================================================
    # Phase 5: WRITEBACK (Register File Write)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Register file write (destination)",
        params['register_write_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 6: MEMORY ACCESS (Mode-dependent cache hierarchy)
    # ==========================================================================
    cache_line_size = 64
    num_accesses = (bytes_transferred + cache_line_size - 1) // cache_line_size

    # Energy per access at each level (in pJ)
    l1_energy_per_access = params['l1_cache_pj_per_byte'] * cache_line_size
    l2_energy_per_access = params['l2_cache_pj_per_byte'] * cache_line_size
    l3_energy_per_access = params['l3_cache_pj_per_byte'] * cache_line_size
    dram_energy_per_access = params['dram_pj_per_byte'] * cache_line_size

    # Calculate accesses at each level based on hit ratios
    l1_accesses = num_accesses
    l1_hits = int(l1_accesses * ratios.l1_hit)
    l1_misses = l1_accesses - l1_hits

    l2_accesses = l1_misses
    l2_hits = int(l2_accesses * ratios.l2_hit)
    l2_misses = l2_accesses - l2_hits

    l3_accesses = l2_misses
    l3_hits = int(l3_accesses * ratios.l3_hit)
    l3_misses = l3_accesses - l3_hits

    dram_accesses = l3_misses

    # Add events for each level
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
