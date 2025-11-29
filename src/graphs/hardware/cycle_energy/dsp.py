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

from typing import Optional

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
)


# DSP Energy Parameters (based on Qualcomm Hexagon / TI C7x)
# Process: 5-7nm
# Frequency: 1.0-1.5 GHz
# Voltage: ~0.7V

DSP_ENERGY_PARAMS = {
    # VLIW instruction overhead
    'vliw_fetch_pj': 2.5,             # VLIW bundle (256-512 bit)
    'vliw_decode_pj': 0.8,            # 4-wide parallel decode
    'vliw_width': 4,                  # Operations per VLIW bundle

    # Register file (VLIW needs many ports: 8+ read, 4 write)
    'register_read_pj': 1.5,          # Per read (more ports than CPU!)
    'register_write_pj': 1.8,         # Per write

    # Execution units
    'tensor_mac_pj': 0.4,             # Tensor/MAC unit (INT8/INT16)
    'vector_op_pj': 0.8,              # Vector unit (SIMD)
    'scalar_op_pj': 2.0,              # Scalar ALU

    # Workload mix (typical AI workload)
    'mac_fraction': 0.70,             # 70% MACs
    'vector_fraction': 0.20,          # 20% vector ops
    'scalar_fraction': 0.10,          # 10% scalar ops

    # Memory (software-managed scratchpad)
    'scratchpad_pj_per_byte': 0.8,    # SRAM scratchpad (no tag overhead!)
    'dram_pj_per_byte': 15.0,         # LPDDR4/5
}


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

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    DSP Energy Model:
    - VLIW amortizes instruction fetch across 4 ops
    - Lower voltage/frequency reduces dynamic power
    - BUT VLIW needs more register ports
    - Software scratchpad eliminates tag lookup energy
    - No L2/L3 means DRAM penalty for large working sets
    """
    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]
    params = DSP_ENERGY_PARAMS

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
