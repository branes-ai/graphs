"""
GPU Cycle-Level Energy Models (SM-Centric)

Models energy consumption for NVIDIA GPUs with two execution modes:
1. CUDA Core model: For scalar/vector workloads (128 MACs/SM-cycle)
2. TensorCore model: For matrix workloads (256 MACs/SM-cycle)

SM Architecture (4 Partitions):
===============================
A single SM is divided into 4 independent processing partitions (sub-cores).
This partitioning exists because a single warp scheduler for 128 CUDA cores
would be infeasible - the scheduling logic doesn't scale.

  +------------------+------------------+------------------+------------------+
  |   Partition 0    |   Partition 1    |   Partition 2    |   Partition 3    |
  +------------------+------------------+------------------+------------------+
  | Warp Scheduler   | Warp Scheduler   | Warp Scheduler   | Warp Scheduler   |
  | Dispatch Unit    | Dispatch Unit    | Dispatch Unit    | Dispatch Unit    |
  +------------------+------------------+------------------+------------------+
  | 32 CUDA Cores    | 32 CUDA Cores    | 32 CUDA Cores    | 32 CUDA Cores    |
  | 1 TensorCore     | 1 TensorCore     | 1 TensorCore     | 1 TensorCore     |
  | Register Bank    | Register Bank    | Register Bank    | Register Bank    |
  +------------------+------------------+------------------+------------------+
                              |
                    Shared L1/Shared Memory (192KB)
                              |
                    L2 Cache (50MB shared across SMs)
                              |
                    HBM (80GB)

To fully utilize an SM:
- CUDA mode: Need 4 warps active (one per partition) = 4 x 32 = 128 ops/cycle
- TensorCore mode: Need 4 MMA instructions (one per partition) = 4 x 64 = 256 MACs/cycle

Native Execution Units:
-----------------------
  GPU (CUDA):      4 warp instructions -> 128 MACs (4 partitions x 32 cores)
  GPU (TensorCore): 4 MMA instructions -> 256 MACs (4 partitions x 64 MACs)

Comparison with other architectures:
  CPU (AVX-512):   1 instruction       ->  16 MACs
  GPU (CUDA):      4 warp instructions -> 128 MACs
  GPU (TensorCore): 4 MMA instructions -> 256 MACs  <- Same as KPU tile!
  TPU (Systolic):  1 tile cycle        -> 16,384 MACs (128x128)
  KPU (Domain):    1 tile cycle        -> 256 MACs (16x16)

Memory Hierarchy:
  - L1/Shared Memory: 192KB per SM (configurable split)
  - L2 Cache: Shared across all SMs (50MB on H100)
  - HBM: Off-chip high-bandwidth memory (80GB on H100)
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
# SM Configuration Constants (H100-class)
# =============================================================================

SM_CONFIG = {
    # Partition configuration (SM is divided into 4 partitions)
    'num_partitions': 4,              # Independent sub-cores per SM
    'warp_schedulers_per_sm': 4,      # One per partition

    # Warp configuration
    'warp_size': 32,                  # Threads per warp (SIMT width)
    'max_warps_per_sm': 64,           # Max resident warps
    'max_threads_per_sm': 2048,       # Max resident threads (64 warps * 32)

    # CUDA cores (per SM, distributed across partitions)
    'cuda_cores_per_sm': 128,         # Total CUDA cores
    'cuda_cores_per_partition': 32,   # 128 / 4 partitions

    # TensorCores (per SM, one per partition)
    'tensor_cores_per_sm': 4,         # One per partition
    'tensor_core_macs_per_cycle': 64, # 4x4x4 MMA operation

    # Derived: MACs per SM-cycle
    'cuda_macs_per_sm_cycle': 128,    # 4 partitions x 32 cores
    'tc_macs_per_sm_cycle': 256,      # 4 partitions x 64 MACs

    # Memory per SM
    'register_file_kb': 256,          # Register file size
    'l1_shared_kb': 192,              # L1/Shared memory (configurable)
}


# =============================================================================
# GPU CUDA Core Model
# =============================================================================

def build_gpu_cuda_cycle_energy(
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
) -> CycleEnergyBreakdown:
    """
    Build the GPU CUDA Core cycle energy breakdown.

    Models scalar/vector workloads executing on CUDA cores.

    SM Partition Model:
        The SM has 4 partitions, each with:
        - 1 warp scheduler
        - 32 CUDA cores
        - 1/4 of the register file

        To fully utilize an SM, you need 4 warps active (one per partition).
        Native unit: 4 warp instructions = 4 x 32 = 128 MACs per SM-cycle.

    Args:
        num_ops: Number of FP32 operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1, L2, or DRAM resident)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output
        operator_type: Type of operator (HIGH_REUSE, LOW_REUSE, STREAMING)
        working_set_bytes: Size of working set for L2 hit ratio calculation
        l2_cache_bytes: Size of L2 cache (default 50MB for H100-class)
        l2_is_cold: True if L2 was flushed by previous streaming operator

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    Native Unit:
        4 warp instructions (one per partition) = 128 MACs
        - 4 instruction fetches (one per warp scheduler)
        - 4 instruction decodes
        - 4 warp scheduling decisions
        - 128 CUDA core operations (32 per partition)
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
    # Energy Parameters
    # ==========================================================================
    process_scale = tech_profile.process_node_nm / 4.0  # 4nm baseline

    # SM configuration
    num_partitions = SM_CONFIG['num_partitions']
    warp_size = SM_CONFIG['warp_size']
    cuda_cores_per_partition = SM_CONFIG['cuda_cores_per_partition']
    macs_per_sm_cycle = SM_CONFIG['cuda_macs_per_sm_cycle']  # 128

    params = {
        # Per-warp instruction costs (each partition has its own warp scheduler)
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'warp_schedule_pj': 0.5 * process_scale,

        # Per-instruction register access (vector register, serves 32 threads)
        'vector_register_read_pj': tech_profile.register_read_energy_pj * 0.25,
        'vector_register_write_pj': tech_profile.register_write_energy_pj * 0.25,

        # Per-op compute energy (CUDA cores)
        'cuda_core_mac_pj': tech_profile.base_alu_energy_pj * 0.85,  # GPU ALU simpler than CPU

        # Fixed overhead (one-time per kernel)
        'kernel_launch_pj': 100_000.0 * process_scale,

        # Per-byte memory costs
        'l1_shared_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'hbm_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # SIMT overhead
        'coalesce_pj': 0.8 * process_scale,
        'divergence_pj': tech_profile.warp_divergence_energy_pj,
        'divergence_rate': 0.05,
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU CUDA (4 partitions x 32 cores)",
        architecture_class="SIMT Data Parallel (CUDA Cores)"
    )

    # ==========================================================================
    # SM-CYCLE CALCULATIONS
    # ==========================================================================
    # To fully utilize an SM, we need 4 warps (one per partition).
    # Each SM-cycle: 4 warp instructions x 32 threads = 128 ops
    #
    # num_sm_cycles = num_ops / 128
    # num_warp_instructions = num_sm_cycles * 4 (one per partition)

    num_sm_cycles = max(1, (num_ops + macs_per_sm_cycle - 1) // macs_per_sm_cycle)
    num_warp_instructions = num_sm_cycles * num_partitions  # 4 warps per SM-cycle

    breakdown.num_cycles = num_sm_cycles
    breakdown.ops_per_cycle = macs_per_sm_cycle  # 128 ops per SM-cycle

    # ==========================================================================
    # FIXED OVERHEAD (One-time per kernel)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Kernel launch overhead",
        params['kernel_launch_pj'],
        1
    )

    # ==========================================================================
    # INSTRUCTION FETCH (per warp instruction, 4 per SM-cycle)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        f"I-cache read ({num_partitions} warps x {warp_size} threads)",
        params['instruction_fetch_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # INSTRUCTION DECODE (per warp instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        f"SIMT decode ({num_partitions} partitions)",
        params['instruction_decode_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # WARP SCHEDULING (4 schedulers per SM, one decision per warp)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        f"Warp scheduler ({num_partitions} per SM)",
        params['warp_schedule_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # REGISTER ACCESS (per warp instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"Vector register read (src1, {warp_size}-wide)",
        params['vector_register_read_pj'],
        num_warp_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"Vector register read (src2, {warp_size}-wide)",
        params['vector_register_read_pj'],
        num_warp_instructions
    )
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        f"Vector register write (dst, {warp_size}-wide)",
        params['vector_register_write_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # COMPUTE (per op, 128 CUDA cores active per SM-cycle)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        f"CUDA Core MACs ({macs_per_sm_cycle}/SM-cycle)",
        params['cuda_core_mac_pj'],
        num_ops
    )

    # ==========================================================================
    # SIMT OVERHEAD
    # ==========================================================================
    num_mem_instructions = max(1, num_warp_instructions // 10)
    breakdown.add_event(
        CyclePhase.SIMT_COHERENCE,
        "Address coalescing (per mem instr)",
        params['coalesce_pj'],
        num_mem_instructions
    )

    num_divergent = int(num_warp_instructions * params['divergence_rate'])
    if num_divergent > 0:
        breakdown.add_event(
            CyclePhase.SIMT_SYNC,
            "Warp divergence handling",
            params['divergence_pj'],
            num_divergent
        )

    # ==========================================================================
    # MEMORY ACCESS
    # ==========================================================================
    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

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


# =============================================================================
# GPU TensorCore Model
# =============================================================================

def build_gpu_tensorcore_cycle_energy(
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
) -> CycleEnergyBreakdown:
    """
    Build the GPU TensorCore cycle energy breakdown.

    Models matrix workloads executing on TensorCores (MMA operations).

    SM Partition Model:
        The SM has 4 partitions, each with:
        - 1 warp scheduler
        - 1 TensorCore (4x4x4 = 64 MACs per MMA instruction)
        - 1/4 of the register file

        To fully utilize an SM, you need 4 warps issuing MMA instructions.
        Native unit: 4 MMA instructions = 4 x 64 = 256 MACs per SM-cycle.

    TensorCore Operation (H100):
        - Input: Two 4x4 matrices (FP16/BF16)
        - Output: One 4x4 matrix (FP32 accumulator)
        - MACs: 4 x 4 x 4 = 64 MACs per MMA instruction
        - Total: 4 TensorCores x 64 = 256 MACs per SM-cycle

    Args:
        num_ops: Number of MAC operations to execute
        bytes_transferred: Total bytes of data accessed
        mode: Operating mode (L1, L2, or DRAM resident)
        hit_ratios: Custom hit ratios (uses defaults for mode if None)
        tech_profile: REQUIRED TechnologyProfile for energy parameters
        verbose: Enable verbose output
        operator_type: Type of operator (HIGH_REUSE, LOW_REUSE, STREAMING)
        working_set_bytes: Size of working set for L2 hit ratio calculation
        l2_cache_bytes: Size of L2 cache (default 50MB for H100-class)
        l2_is_cold: True if L2 was flushed by previous streaming operator

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    Native Unit:
        4 MMA instructions (one per partition) = 256 MACs
        - 4 instruction fetches
        - 4 instruction decodes
        - 4 warp scheduling decisions
        - 4 TensorCore MMA operations (64 MACs each)

    Comparison:
        This 256 MACs/SM-cycle is identical to KPU's 16x16 tile (256 MACs).
        The key difference is:
        - GPU: 4 separate 4x4x4 operations, need 4 warp schedulers
        - KPU: 1 unified 16x16 systolic array, 1 domain tracker
    """
    if tech_profile is None:
        raise ValueError(
            "tech_profile is required. Use DEFAULT_PROFILE from "
            "graphs.hardware.technology_profile or provide a specific profile."
        )

    # GPU has no L3 cache
    effective_mode = mode
    if mode == OperatingMode.L3_RESIDENT:
        effective_mode = OperatingMode.DRAM_RESIDENT

    # Compute L2 hit ratio
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
            l3_hit=0.0,
        )
        if verbose:
            print(f"  L2 hit ratio: {l2_hit:.1%} (operator={operator_type.value}, "
                  f"ws={ws_bytes/1024/1024:.1f}MB, L2={l2_cache_bytes/1024/1024:.0f}MB, "
                  f"cold={l2_is_cold})")

    # ==========================================================================
    # Energy Parameters
    # ==========================================================================
    process_scale = tech_profile.process_node_nm / 4.0

    num_partitions = SM_CONFIG['num_partitions']
    tc_macs_per_op = SM_CONFIG['tensor_core_macs_per_cycle']  # 64 MACs per TensorCore
    macs_per_sm_cycle = SM_CONFIG['tc_macs_per_sm_cycle']  # 256 MACs per SM-cycle

    params = {
        # Per-MMA instruction costs
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'warp_schedule_pj': 0.5 * process_scale,

        # MMA instructions use matrix registers (larger than vector registers)
        # But the access pattern is more efficient (bulk load)
        'matrix_register_read_pj': tech_profile.register_read_energy_pj * 0.4,
        'matrix_register_write_pj': tech_profile.register_write_energy_pj * 0.4,

        # Per-op compute energy (TensorCore MAC)
        # TensorCores are highly optimized for matrix multiply
        'tensor_core_mac_pj': tech_profile.tensor_core_mac_energy_pj,

        # Fixed overhead
        'kernel_launch_pj': 100_000.0 * process_scale,

        # Per-byte memory costs
        'l1_shared_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'hbm_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # Minimal SIMT overhead for TensorCore workloads
        # (Less divergence since matrix ops are uniform)
        'coalesce_pj': 0.8 * process_scale,
        'divergence_pj': tech_profile.warp_divergence_energy_pj,
        'divergence_rate': 0.01,  # Much lower for matrix ops
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU TensorCore (4 partitions x 64 MACs)",
        architecture_class="SIMT Data Parallel (TensorCores)"
    )

    # ==========================================================================
    # SM-CYCLE CALCULATIONS
    # ==========================================================================
    # To fully utilize TensorCores, we need 4 MMA instructions per SM-cycle.
    # Each MMA: 4x4x4 = 64 MACs
    # Total: 4 x 64 = 256 MACs per SM-cycle
    #
    # num_sm_cycles = num_ops / 256
    # num_mma_instructions = num_sm_cycles * 4

    num_sm_cycles = max(1, (num_ops + macs_per_sm_cycle - 1) // macs_per_sm_cycle)
    num_mma_instructions = num_sm_cycles * num_partitions

    breakdown.num_cycles = num_sm_cycles
    breakdown.ops_per_cycle = macs_per_sm_cycle  # 256 MACs per SM-cycle

    # ==========================================================================
    # FIXED OVERHEAD
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Kernel launch overhead",
        params['kernel_launch_pj'],
        1
    )

    # ==========================================================================
    # INSTRUCTION FETCH (per MMA instruction, 4 per SM-cycle)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        f"I-cache read ({num_partitions} MMA instrs)",
        params['instruction_fetch_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # INSTRUCTION DECODE (per MMA instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        f"MMA decode ({num_partitions} partitions)",
        params['instruction_decode_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # WARP SCHEDULING (4 schedulers issue MMA instructions)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        f"Warp scheduler MMA ({num_partitions} per SM)",
        params['warp_schedule_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # REGISTER ACCESS (matrix fragments for MMA)
    # ==========================================================================
    # MMA uses matrix fragments: A (4x4), B (4x4), C (4x4 accumulator)
    # Each MMA reads 2 input matrices, writes 1 output matrix
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Matrix register read (A fragment, 4x4)",
        params['matrix_register_read_pj'],
        num_mma_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Matrix register read (B fragment, 4x4)",
        params['matrix_register_read_pj'],
        num_mma_instructions
    )
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Matrix register write (C accumulator, 4x4)",
        params['matrix_register_write_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # COMPUTE (TensorCore MMA operations)
    # ==========================================================================
    # Each TensorCore does 64 MACs per MMA instruction
    # 4 TensorCores x 64 = 256 MACs per SM-cycle
    breakdown.add_event(
        CyclePhase.EXECUTE,
        f"TensorCore MACs ({tc_macs_per_op} per TC, {num_partitions} TCs)",
        params['tensor_core_mac_pj'],
        num_ops
    )

    # ==========================================================================
    # SIMT OVERHEAD (minimal for matrix ops)
    # ==========================================================================
    num_mem_instructions = max(1, num_mma_instructions // 16)  # Less frequent
    breakdown.add_event(
        CyclePhase.SIMT_COHERENCE,
        "Address coalescing (matrix tiles)",
        params['coalesce_pj'],
        num_mem_instructions
    )

    num_divergent = int(num_mma_instructions * params['divergence_rate'])
    if num_divergent > 0:
        breakdown.add_event(
            CyclePhase.SIMT_SYNC,
            "Warp divergence (minimal for MMA)",
            params['divergence_pj'],
            num_divergent
        )

    # ==========================================================================
    # MEMORY ACCESS
    # ==========================================================================
    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

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


# =============================================================================
# Legacy API (for backwards compatibility)
# =============================================================================

def build_gpu_cycle_energy(
    num_ops: int = 1000,
    bytes_transferred: int = 4096,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
    tech_profile: 'TechnologyProfile' = None,
    verbose: bool = False,
    operator_type: OperatorType = OperatorType.HIGH_REUSE,
    working_set_bytes: Optional[int] = None,
    l2_cache_bytes: int = DEFAULT_L2_CACHE_SIZES['default'],
    l2_is_cold: bool = False,
    tensor_core_utilization: float = 0.8,
) -> CycleEnergyBreakdown:
    """
    Legacy GPU energy model (deprecated).

    For new code, use:
    - build_gpu_cuda_cycle_energy() for scalar/vector workloads
    - build_gpu_tensorcore_cycle_energy() for matrix workloads

    This function provides backwards compatibility by blending CUDA and TensorCore
    models based on tensor_core_utilization.
    """
    if tensor_core_utilization >= 0.5:
        # Primarily matrix workload - use TensorCore model
        return build_gpu_tensorcore_cycle_energy(
            num_ops=num_ops,
            bytes_transferred=bytes_transferred,
            mode=mode,
            hit_ratios=hit_ratios,
            tech_profile=tech_profile,
            verbose=verbose,
            operator_type=operator_type,
            working_set_bytes=working_set_bytes,
            l2_cache_bytes=l2_cache_bytes,
            l2_is_cold=l2_is_cold,
        )
    else:
        # Primarily scalar/vector workload - use CUDA model
        return build_gpu_cuda_cycle_energy(
            num_ops=num_ops,
            bytes_transferred=bytes_transferred,
            mode=mode,
            hit_ratios=hit_ratios,
            tech_profile=tech_profile,
            verbose=verbose,
            operator_type=operator_type,
            working_set_bytes=working_set_bytes,
            l2_cache_bytes=l2_cache_bytes,
            l2_is_cold=l2_is_cold,
        )
