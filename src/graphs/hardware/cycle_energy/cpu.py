"""
CPU Cycle-Level Energy Model

Models energy consumption for a stored program machine (MIMD) architecture
like Intel Xeon, AMD EPYC, or ARM Cortex processors.

CPU Basic Cycle:
  INSTRUCTION FETCH -> DECODE -> OPERAND FETCH -> EXECUTE -> WRITEBACK
                                                      |
                                                      v
                                              MEMORY ACCESS
                                          (L1 -> L2 -> L3 -> DRAM)

Key characteristics:
- Complex instruction decode (x86-64 variable length, ARM fixed length)
- Out-of-order execution with register renaming
- Deep cache hierarchy (L1/L2/L3)
- High frequency (3-4 GHz) but high per-op energy

SIMD Amortization:
- Modern CPUs use SIMD/vector instructions to amortize instruction overhead
- x86 AVX-512: 512 bits / 32 bits = 16 FP32 ops per instruction
- x86 AVX-256: 256 bits / 32 bits = 8 FP32 ops per instruction
- ARM SVE/SVE2: Variable length up to 2048 bits (1-64 FP32 ops per instruction)
- ARM NEON: 128 bits / 32 bits = 4 FP32 ops per instruction
- Without SIMD: 2 instructions per FMA (mul + add) for scalar execution

The model supports a `simd_width` parameter to control amortization:
- simd_width=1: Scalar execution (2 instructions per op)
- simd_width=4: ARM NEON (1 instruction per 4 ops = 0.25 instructions/op)
- simd_width=8: AVX-256 (1 instruction per 8 ops = 0.125 instructions/op)
- simd_width=16: AVX-512 (1 instruction per 16 ops = 0.0625 instructions/op)
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graphs.hardware.technology_profile import TechnologyProfile

from .base import (
    CyclePhase,
    OperatingMode,
    OperatorType,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CPU_L3_CACHE_SIZES,
    CycleEnergyBreakdown,
    compute_cache_hit_ratio,
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
    verbose: bool = False,
    # New parameters for working-set-based L3 modeling
    operator_type: OperatorType = OperatorType.HIGH_REUSE,
    working_set_bytes: Optional[int] = None,
    l3_cache_bytes: int = CPU_L3_CACHE_SIZES['default'],
    l3_is_cold: bool = False,
    # SIMD amortization parameter
    simd_width: int = 16,  # Default AVX-512 (16 FP32 ops per instruction)
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
        operator_type: Type of operator (HIGH_REUSE, LOW_REUSE, STREAMING)
                      Determines L3 hit ratio based on data reuse pattern
        working_set_bytes: Size of working set for L3 hit ratio calculation
                          If None, defaults to bytes_transferred
        l3_cache_bytes: Size of L3/LLC cache (default 32MB)
        l3_is_cold: True if L3 was flushed by previous streaming operator
        simd_width: Number of FP32 ops per SIMD instruction (amortization factor)
                   - 1: Scalar execution (2 instructions per op)
                   - 4: ARM NEON (128-bit vectors)
                   - 8: x86 AVX-256
                   - 16: x86 AVX-512 (default)
                   - 32+: ARM SVE with wider vectors

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    CPU Energy Model:
    - Instruction overhead is AMORTIZED across SIMD width
    - With simd_width=16 (AVX-512): 1 instruction controls 16 FP32 ops
    - Register file energy scales with instruction count (not op count)
    - Cache hierarchy provides energy filtering (hit ratios)
    - Memory access energy depends on operating mode

    L3 Cache Model:
        The CPU L3 (LLC) is implicit (hardware-managed). Hit ratio depends on:
        - operator_type: MatMul (HIGH_REUSE) vs ReLU (STREAMING)
        - working_set_bytes: Whether data fits in L3
        - l3_is_cold: Whether previous op flushed L3

        IMPORTANT: Streaming operators (activation, BatchNorm, Softmax)
        will flush L3 contents, causing the next MatMul to start cold.

    Technology Profile (REQUIRED):
        A TechnologyProfile must be provided to specify energy parameters.
        Use DEFAULT_PROFILE for typical datacenter values.

        Example:
            from graphs.hardware.technology_profile import DATACENTER_4NM_HBM3
            breakdown = build_cpu_cycle_energy(
                num_ops=1000,
                tech_profile=DATACENTER_4NM_HBM3,
                operator_type=OperatorType.HIGH_REUSE,
                working_set_bytes=16*1024*1024,
            )

    Raises:
        ValueError: If tech_profile is not provided.
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )

    # Compute L3 hit ratio based on operator type and working set
    # This models the implicit (hardware-managed) L3/LLC cache behavior
    if hit_ratios is not None:
        # User provided explicit hit ratios - use them
        ratios = hit_ratios
    elif mode in (OperatingMode.L1_RESIDENT, OperatingMode.L2_RESIDENT):
        # Data fits in L1 or L2 - use default ratios (high hit rates)
        ratios = DEFAULT_HIT_RATIOS[mode]
    else:
        # L3 or DRAM resident - compute L3 hit ratio from operator type
        ws_bytes = working_set_bytes if working_set_bytes is not None else bytes_transferred
        l3_hit = compute_cache_hit_ratio(
            operator_type=operator_type,
            working_set_bytes=ws_bytes,
            cache_size_bytes=l3_cache_bytes,
            cache_is_cold=l3_is_cold,
            is_explicit_memory=False,
        )
        # L1/L2 hit ratios from default, L3 computed from operator type
        default_ratios = DEFAULT_HIT_RATIOS[mode]
        ratios = HitRatios(
            l1_hit=default_ratios.l1_hit,
            l2_hit=default_ratios.l2_hit,
            l3_hit=l3_hit,  # Computed from operator type
        )
        if verbose:
            print(f"  L3 hit ratio: {l3_hit:.1%} (operator={operator_type.value}, "
                  f"ws={ws_bytes/1024/1024:.1f}MB, L3={l3_cache_bytes/1024/1024:.0f}MB, "
                  f"cold={l3_is_cold})")

    breakdown = CycleEnergyBreakdown(
        architecture_name="CPU (Intel Xeon / AMD EPYC)",
        architecture_class="Stored Program Machine (MIMD)"
    )

    # Derive all energy parameters from technology profile
    #
    # SIMD Amortization Model:
    # - Scalar: 2 instructions per op (load/store + compute)
    # - SIMD: 1 instruction controls simd_width ops
    # - instructions_per_op = 2 / simd_width for SIMD execution
    # - With AVX-512 (simd_width=16): 2/16 = 0.125 instructions per op
    #
    # This models the key advantage of SIMD: instruction fetch/decode is
    # amortized across the vector width, similar to how GPU amortizes
    # across 32 threads in a warp.
    #
    if simd_width < 1:
        simd_width = 1  # Minimum is scalar

    # For SIMD, one instruction handles simd_width ops
    # Base scalar overhead is 2 instructions per op (typical CISC)
    instructions_per_op = 2.0 / simd_width

    params = {
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'instructions_per_op': instructions_per_op,
        'register_read_pj': tech_profile.register_read_energy_pj,
        'register_write_pj': tech_profile.register_write_energy_pj,
        'alu_energy_pj': tech_profile.base_alu_energy_pj,
        'l1_cache_pj_per_byte': tech_profile.l1_cache_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'l3_cache_pj_per_byte': tech_profile.l3_cache_energy_per_byte_pj,
        'dram_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,
        'simd_width': simd_width,
    }

    # Calculate number of instructions with SIMD amortization
    # With AVX-512 (simd_width=16): 1000 ops -> 125 instructions
    num_instructions = max(1, int(num_ops * params['instructions_per_op']))
    breakdown.num_cycles = num_instructions
    breakdown.ops_per_cycle = simd_width  # Each instruction produces simd_width ops

    # ==========================================================================
    # Phase 1: INSTRUCTION FETCH (from I-cache)
    # ==========================================================================
    # SIMD amortization: one fetch serves simd_width ops
    if simd_width > 1:
        fetch_desc = f"I-cache read (SIMD: {simd_width} ops/instr)"
    else:
        fetch_desc = "I-cache read (scalar)"
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        fetch_desc,
        params['instruction_fetch_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 2: INSTRUCTION DECODE
    # ==========================================================================
    if simd_width > 1:
        decode_desc = f"Decode (SIMD: {simd_width} ops/instr)"
    else:
        decode_desc = "Decode logic (scalar)"
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        decode_desc,
        params['instruction_decode_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Register File Reads)
    # ==========================================================================
    # Register reads scale with instructions (SIMD reads vector regs once per instr)
    if simd_width > 1:
        reg_read_desc = f"Vector register read (SIMD {simd_width}-wide)"
    else:
        reg_read_desc = "Register file read"
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"{reg_read_desc} (src 1)",
        params['register_read_pj'],
        num_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"{reg_read_desc} (src 2)",
        params['register_read_pj'],
        num_instructions
    )

    # ==========================================================================
    # Phase 4: EXECUTE (ALU operations)
    # ==========================================================================
    # Compute energy scales with actual operations (not instructions)
    # Each lane in the SIMD unit consumes energy for its computation
    if simd_width > 1:
        exec_desc = f"SIMD ALU/FPU ({simd_width}-wide vector)"
    else:
        exec_desc = "ALU/FPU operation (scalar)"
    breakdown.add_event(
        CyclePhase.EXECUTE,
        exec_desc,
        params['alu_energy_pj'],
        num_ops  # Energy per actual operation
    )

    # ==========================================================================
    # Phase 5: WRITEBACK (Register File Write)
    # ==========================================================================
    # Writeback scales with instructions (SIMD writes vector reg once per instr)
    if simd_width > 1:
        writeback_desc = f"Vector register write (SIMD {simd_width}-wide)"
    else:
        writeback_desc = "Register file write"
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        writeback_desc,
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
