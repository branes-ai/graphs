"""
GPU Cycle-Level Energy Model (SM-Centric)

Models energy consumption for SIMT (Single Instruction Multiple Thread)
data parallel architectures like NVIDIA H100 or Jetson GPUs.

SM-Centric Model:
==================
A GPU is a collection of Streaming Multiprocessors (SMs). The energy per
operation is determined by what happens on a SINGLE SM. A bigger GPU (more
SMs) increases throughput but NOT energy per op.

Single SM Architecture (H100-class):
  - 128 CUDA Cores (FP32)
  - 4 Tensor Core units (4x4x4 matrix ops)
  - 64 max resident warps (2048 threads)
  - 256KB register file
  - 192KB L1/Shared memory (configurable)

Warp Execution Model:
  - One warp = 32 threads executing in SIMT lockstep
  - One instruction controls all 32 threads in a warp
  - Instruction fetch/decode is amortized across 32 threads (like SIMD)
  - Register access is per-instruction (vector register read)
  - Compute energy is per-op (each lane does 1 MAC)

Memory Hierarchy:
  - L1/Shared Memory: 192KB per SM (configurable split)
  - L2 Cache: Shared across all SMs (50MB on H100)
  - HBM: Off-chip high-bandwidth memory (80GB on H100)

Key insight: Energy per op is CONSTANT regardless of GPU size.
The number of SMs only affects throughput (ops/second), not efficiency (pJ/op).

GPU Basic Cycle (SIMT):
  INSTRUCTION FETCH -> DECODE -> WARP SCHEDULING -> REGISTER ACCESS
         |                            |
         v                            v
    (per warp)              COHERENCE MACHINERY
                                   |
                                   v
                           EXECUTE (CUDA/Tensor)
                                   |
                                   v
                           MEMORY ACCESS
                       (Shared/L1 -> L2 -> HBM)
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
    DEFAULT_L2_CACHE_SIZES,
    CycleEnergyBreakdown,
    compute_l2_hit_ratio,
)


# =============================================================================
# SM Configuration Constants
# =============================================================================
# These define the fixed hardware resources of a single SM.
# Energy per op is determined by these constants, not by workload size.

SM_CONFIG = {
    # Warp configuration
    'warp_size': 32,              # Threads per warp (SIMT width)
    'max_warps_per_sm': 64,       # Max resident warps
    'max_threads_per_sm': 2048,   # Max resident threads (64 warps * 32)

    # Compute resources per SM
    'cuda_cores': 128,            # FP32 CUDA cores
    'tensor_cores': 4,            # Tensor Core units

    # Memory per SM
    'register_file_kb': 256,      # Register file size
    'l1_shared_kb': 192,          # L1/Shared memory (configurable)
}


def build_gpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    tech_profile: 'TechnologyProfile' = None,
    verbose: bool = False,
    # Working-set-based L2 modeling
    operator_type: OperatorType = OperatorType.HIGH_REUSE,
    working_set_bytes: Optional[int] = None,
    l2_cache_bytes: int = DEFAULT_L2_CACHE_SIZES['default'],
    l2_is_cold: bool = False,
    # Tensor core utilization
    tensor_core_utilization: float = 0.8,
) -> CycleEnergyBreakdown:
    """
    Build the GPU basic cycle energy breakdown using SM-centric model.

    The model calculates energy based on a SINGLE SM's perspective:
    - Instruction overhead is amortized across warp (32 threads)
    - Register access is per-instruction (vector register operations)
    - Compute energy is per-op (each CUDA/Tensor core lane)
    - Memory energy depends on hierarchy (L1/Shared -> L2 -> HBM)

    A larger GPU (more SMs) processes more ops in parallel but doesn't
    change the energy per op - it only increases throughput.

    Args:
        num_ops: Number of operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1, L2, or DRAM resident - GPU has no L3)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output
        operator_type: Type of operator (HIGH_REUSE, LOW_REUSE, STREAMING)
                      Determines L2 hit ratio based on data reuse pattern
        working_set_bytes: Size of working set for L2 hit ratio calculation
        l2_cache_bytes: Size of L2 cache (default 50MB for H100-class)
        l2_is_cold: True if L2 was flushed by previous streaming operator
        tensor_core_utilization: Fraction of ops using tensor cores (0.0-1.0)

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    SM-Centric Energy Model:
        1. Instruction overhead: Amortized across warp (32 threads)
           - num_instructions = num_ops / warp_size
           - Each instruction fetch/decode serves 32 ops

        2. Register access: Per-instruction (vector registers)
           - 2 source vector register reads per instruction
           - 1 destination vector register write per instruction

        3. Warp scheduling: Per-instruction
           - Scheduler selects one warp per cycle per scheduler
           - SM has 4 warp schedulers, but we model per-instruction

        4. Compute: Per-op
           - Each CUDA core lane or Tensor Core element does 1 MAC
           - Energy scales linearly with num_ops

        5. Memory: Per-byte with hierarchy
           - L1/Shared: On-SM, very low latency
           - L2: Shared across SMs, higher latency
           - HBM: Off-chip, highest latency and energy

    Raises:
        ValueError: If tech_profile is not provided.
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )

    # GPU has no L3 cache - L3 mode should behave like DRAM mode
    effective_mode = mode
    if mode == OperatingMode.L3_RESIDENT:
        effective_mode = OperatingMode.DRAM_RESIDENT

    # Compute L2 hit ratio based on operator type and working set
    if hit_ratios is not None:
        ratios = hit_ratios
    elif effective_mode == OperatingMode.L1_RESIDENT:
        ratios = DEFAULT_HIT_RATIOS[effective_mode]
    else:
        ws_bytes = working_set_bytes if working_set_bytes is not None else bytes_transferred
        l2_hit = compute_l2_hit_ratio(
            operator_type=operator_type,
            working_set_bytes=ws_bytes,
            l2_cache_bytes=l2_cache_bytes,
            l2_is_cold=l2_is_cold,
        )
        default_ratios = DEFAULT_HIT_RATIOS[effective_mode]
        ratios = HitRatios(
            l1_hit=default_ratios.l1_hit,
            l2_hit=l2_hit,
            l3_hit=0.0,  # GPU has no L3
        )
        if verbose:
            print(f"  L2 hit ratio: {l2_hit:.1%} (operator={operator_type.value}, "
                  f"ws={ws_bytes/1024/1024:.1f}MB, L2={l2_cache_bytes/1024/1024:.0f}MB, "
                  f"cold={l2_is_cold})")

    # ==========================================================================
    # Energy Parameters (from TechnologyProfile)
    # ==========================================================================
    process_scale = tech_profile.process_node_nm / 4.0  # 4nm baseline

    # SM configuration
    warp_size = SM_CONFIG['warp_size']

    # Energy parameters
    params = {
        # Per-instruction costs (amortized across warp)
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'warp_schedule_pj': 0.5 * process_scale,  # Scheduler decision per instruction

        # Per-instruction register access (vector register, serves 32 threads)
        # GPU register files are simpler than CPU (no renaming, in-order within warp)
        'vector_register_read_pj': tech_profile.register_read_energy_pj * 0.25,
        'vector_register_write_pj': tech_profile.register_write_energy_pj * 0.25,

        # Per-op compute energy
        # Tensor cores are ~0.85x energy of baseline ALU (specialized MAC array)
        # CUDA cores run at baseline ALU energy
        'tensor_core_mac_pj': tech_profile.tensor_core_mac_energy_pj,
        'cuda_core_mac_pj': tech_profile.base_alu_energy_pj * 0.85,  # GPU ALU is simpler than CPU

        # Fixed overhead (one-time per kernel, amortizes with scale)
        'kernel_launch_pj': 100_000.0 * process_scale,

        # Per-byte memory costs
        'l1_shared_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'hbm_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # Coherence/synchronization (minimal overhead)
        'coalesce_pj': 0.8 * process_scale,  # Per memory instruction
        'divergence_pj': tech_profile.warp_divergence_energy_pj,
        'divergence_rate': 0.05,  # 5% of instructions have divergence
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU (NVIDIA H100 / Jetson)",
        architecture_class="SIMT Data Parallel"
    )

    # ==========================================================================
    # SM-CENTRIC CALCULATIONS
    # ==========================================================================
    # Key insight: One instruction controls 32 threads (one warp).
    # Instruction overhead is amortized across warp, just like CPU SIMD.
    #
    # num_instructions = num_ops / warp_size
    # This is the GPU's "SIMD amortization" - identical concept to AVX-512.

    num_instructions = max(1, num_ops // warp_size)

    breakdown.num_cycles = num_instructions
    breakdown.ops_per_cycle = warp_size  # Each instruction produces warp_size ops

    # ==========================================================================
    # FIXED OVERHEAD (One-time per kernel)
    # ==========================================================================
    # Kernel launch is the only true fixed cost. It amortizes with scale.
    # We don't model "SM activation" separately because:
    # - SMs are always powered when GPU is active
    # - The per-op energy already accounts for SM power consumption

    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Kernel launch overhead",
        params['kernel_launch_pj'],
        1
    )

    # ==========================================================================
    # INSTRUCTION FETCH (amortized across warp)
    # ==========================================================================
    # One fetch serves 32 threads. This is GPU's SIMT amortization.
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        f"I-cache read (1 instr -> {warp_size} ops)",
        params['instruction_fetch_pj'],
        num_instructions
    )

    # ==========================================================================
    # INSTRUCTION DECODE (amortized across warp)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        f"SIMT decode + predication ({warp_size} threads)",
        params['instruction_decode_pj'],
        num_instructions
    )

    # ==========================================================================
    # WARP SCHEDULING (per instruction)
    # ==========================================================================
    # Each instruction dispatch requires a scheduling decision.
    # The SM's warp schedulers select which warp to issue.
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Warp scheduler decision",
        params['warp_schedule_pj'],
        num_instructions
    )

    # ==========================================================================
    # REGISTER ACCESS (per instruction, vector registers)
    # ==========================================================================
    # Each instruction reads 2 source vector registers and writes 1 destination.
    # A "vector register" holds 32 values (one per thread in warp).
    # This is analogous to AVX-512 reading/writing 512-bit registers.
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"Vector register read (src1, {warp_size}-wide)",
        params['vector_register_read_pj'],
        num_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"Vector register read (src2, {warp_size}-wide)",
        params['vector_register_read_pj'],
        num_instructions
    )
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        f"Vector register write (dst, {warp_size}-wide)",
        params['vector_register_write_pj'],
        num_instructions
    )

    # ==========================================================================
    # COMPUTE (per op)
    # ==========================================================================
    # Each of the 32 lanes does one MAC. Energy scales with actual ops.
    tensor_core_ops = int(num_ops * tensor_core_utilization)
    cuda_core_ops = num_ops - tensor_core_ops

    if tensor_core_ops > 0:
        breakdown.add_event(
            CyclePhase.EXECUTE,
            "Tensor Core MACs (FP16/BF16)",
            params['tensor_core_mac_pj'],
            tensor_core_ops
        )

    if cuda_core_ops > 0:
        breakdown.add_event(
            CyclePhase.EXECUTE,
            "CUDA Core MACs (FP32)",
            params['cuda_core_mac_pj'],
            cuda_core_ops
        )

    # ==========================================================================
    # SIMT OVERHEAD (minimal per-instruction costs)
    # ==========================================================================
    # Coalescing: Per memory instruction (not per op)
    # Estimate 1 memory instruction per 10 compute instructions
    num_mem_instructions = max(1, num_instructions // 10)
    breakdown.add_event(
        CyclePhase.SIMT_COHERENCE,
        "Address coalescing (per mem instr)",
        params['coalesce_pj'],
        num_mem_instructions
    )

    # Divergence: Only for divergent instructions (5% default)
    num_divergent = int(num_instructions * params['divergence_rate'])
    if num_divergent > 0:
        breakdown.add_event(
            CyclePhase.SIMT_SYNC,
            "Warp divergence handling",
            params['divergence_pj'],
            num_divergent
        )

    # ==========================================================================
    # MEMORY ACCESS (per byte, through hierarchy)
    # ==========================================================================
    # Memory model: L1/Shared -> L2 -> HBM
    # Energy is charged per byte at each level based on hit ratios.
    #
    # This is consistent with CPU/TPU/KPU models:
    # - bytes_transferred is the external memory footprint
    # - Hit ratios determine how much goes to each level

    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

    # L1/Shared memory access
    l1_bytes = int(bytes_transferred * ratios.l1_hit)
    l2_bytes = int((bytes_transferred - l1_bytes) * ratios.l2_hit)
    hbm_bytes = bytes_transferred - l1_bytes - l2_bytes

    if l1_bytes > 0:
        mem_name = "Shared Memory" if is_l1_resident else "L1 cache"
        breakdown.add_event(
            CyclePhase.MEM_L1,
            f"{mem_name} ({l1_bytes} bytes)",
            params['l1_shared_pj_per_byte'],
            l1_bytes
        )

    if l2_bytes > 0:
        breakdown.add_event(
            CyclePhase.MEM_L2,
            f"L2 cache ({l2_bytes} bytes)",
            params['l2_cache_pj_per_byte'],
            l2_bytes
        )

    if hbm_bytes > 0:
        breakdown.add_event(
            CyclePhase.MEM_HBM,
            f"HBM ({hbm_bytes} bytes)",
            params['hbm_pj_per_byte'],
            hbm_bytes
        )

    return breakdown
