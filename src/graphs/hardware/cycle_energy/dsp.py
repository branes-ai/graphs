"""
DSP (VLIW) Cycle-Level Energy Model

Models energy consumption for VLIW (Very Long Instruction Word) DSP
architectures like Qualcomm Hexagon or TI C7x.

DSP Basic Cycle (VLIW):
  VLIW INSTRUCTION FETCH -> PARALLEL DECODE -> OPERAND FETCH
            |                    |                   |
            v                    v                   v
       (256-512 bit)        (4 slots)         (Multi-port RF)
                                                    |
                                                    v
                                    PARALLEL EXECUTE (4 slots)
                                    [Tensor] [Vector] [Scalar]
                                                    |
                                                    v
                                            MEMORY ACCESS
                                       (Scratchpad -> DRAM)

Key characteristics:
- VLIW bundles contain 4 operations per instruction
- Compiler does scheduling (no dynamic scheduling)
- Software-managed scratchpad (no cache tags)
- Lower frequency (1.0-1.5 GHz) and voltage (~0.7V)
- No L2/L3 cache hierarchy

DSP is NOT fundamentally more efficient per operation - it trades
performance for power by running at lower voltage/frequency.
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
# The hardcoded DSP_ENERGY_PARAMS dict has been removed to eliminate
# dual sources of energy definitions. Use a TechnologyProfile instance
# (e.g., DEFAULT_PROFILE from technology_profile.py) for all energy values.


def build_dsp_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    tech_profile: 'TechnologyProfile' = None,
    verbose: bool = False
) -> CycleEnergyBreakdown:
    """
    Build the DSP (VLIW) basic cycle energy breakdown.

    Args:
        num_ops: Number of operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1 or DRAM resident - DSP has no L2/L3)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    DSP Energy Model:
    - VLIW amortizes instruction fetch across 4 ops
    - Lower voltage/frequency reduces dynamic power
    - BUT VLIW needs more register ports
    - Software scratchpad eliminates tag lookup energy
    - No L2/L3 means DRAM penalty for large working sets

    Technology Profile (REQUIRED):
        A TechnologyProfile must be provided to specify energy parameters.
        Use DEFAULT_PROFILE for typical datacenter values.

    Raises:
        ValueError: If tech_profile is not provided.
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

    # Derive all energy parameters from technology profile
    process_scale = tech_profile.process_node_nm / 7.0  # 7nm is baseline for DSP
    params = {
        # VLIW instruction overhead
        'vliw_fetch_pj': tech_profile.instruction_fetch_energy_pj * 1.25,  # Wider instruction
        'vliw_decode_pj': tech_profile.instruction_decode_energy_pj,
        'vliw_width': 4,  # Operations per VLIW bundle

        # Register file (VLIW needs many ports: 8+ read, 4 write)
        'register_read_pj': tech_profile.register_read_energy_pj * 0.6,  # Lower freq
        'register_write_pj': tech_profile.register_write_energy_pj * 0.6,

        # Execution units
        'tensor_mac_pj': tech_profile.tensor_core_mac_energy_pj,
        'vector_op_pj': tech_profile.simd_mac_energy_pj,
        'scalar_op_pj': tech_profile.base_alu_energy_pj,

        # Workload mix (typical AI workload)
        'mac_fraction': 0.70,
        'vector_fraction': 0.20,
        'scalar_fraction': 0.10,

        # Memory (software-managed scratchpad)
        'scratchpad_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'dram_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="DSP (Qualcomm Hexagon / TI C7x)",
        architecture_class="VLIW Stored Program Machine"
    )

    vliw_width = params['vliw_width']
    num_vliw_bundles = (num_ops + vliw_width - 1) // vliw_width

    breakdown.num_cycles = num_vliw_bundles
    breakdown.ops_per_cycle = vliw_width

    # Workload breakdown
    mac_ops = int(num_ops * params['mac_fraction'])
    vector_ops = int(num_ops * params['vector_fraction'])
    scalar_ops = num_ops - mac_ops - vector_ops

    # ==========================================================================
    # Phase 1: VLIW INSTRUCTION FETCH
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        "VLIW bundle fetch (256-512 bit)",
        params['vliw_fetch_pj'],
        num_vliw_bundles
    )

    # ==========================================================================
    # Phase 2: PARALLEL DECODE
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        "Multi-slot parallel decode (4 slots)",
        params['vliw_decode_pj'],
        num_vliw_bundles
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Multi-port Register File)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Multi-port register file read",
        params['register_read_pj'],
        num_ops * 2  # 2 reads per op
    )

    # ==========================================================================
    # Phase 4: EXECUTE (Heterogeneous Units)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Tensor/MAC unit (INT8/INT16)",
        params['tensor_mac_pj'],
        mac_ops
    )
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Vector unit (SIMD)",
        params['vector_op_pj'],
        vector_ops
    )
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Scalar ALU",
        params['scalar_op_pj'],
        scalar_ops
    )

    # ==========================================================================
    # Phase 5: WRITEBACK
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Multi-port register file write",
        params['register_write_pj'],
        num_ops
    )

    # ==========================================================================
    # Phase 6: MEMORY ACCESS (Scratchpad + DRAM)
    # ==========================================================================
    # DSP uses l1_hit as scratchpad hit rate (no L2/L3)
    scratchpad_bytes = int(bytes_transferred * ratios.l1_hit)
    dram_bytes = bytes_transferred - scratchpad_bytes

    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

    if scratchpad_bytes > 0:
        if is_l1_resident:
            desc = "Scratchpad SRAM (100% resident)"
        else:
            desc = f"Scratchpad SRAM ({int(ratios.l1_hit * 100)}% hit, DMA prefetch)"
        breakdown.add_event(
            CyclePhase.MEM_SRAM,
            desc,
            params['scratchpad_pj_per_byte'],
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
            params['dram_pj_per_byte'],
            dram_bytes
        )

    return breakdown
