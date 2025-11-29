"""
GPU Cycle-Level Energy Model

Models energy consumption for SIMT (Single Instruction Multiple Thread)
data parallel architectures like NVIDIA H100 or Jetson GPUs.

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

Key characteristics:
- One instruction controls 32 threads (warp)
- Massive parallelism (thousands of concurrent threads)
- SIMT overhead dominates for small workloads:
  - Fixed infrastructure (kernel launch, SM activation)
  - Thread management (warp scheduling, context)
  - Coherence machinery (for L2+ modes)
  - Synchronization (barriers, divergence)
"""

from typing import Optional

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
)


# GPU Energy Parameters (based on NVIDIA H100 / modern datacenter GPUs)
# Process: 4nm (TSMC)
# Frequency: 1.8-2.0 GHz
# Voltage: ~0.8V

GPU_ENERGY_PARAMS = {
    # Instruction overhead (amortized across warp)
    'instruction_fetch_pj': 2.0,      # Per warp (shared by 32 threads)
    'instruction_decode_pj': 0.5,     # SIMT decode + predication
    'instructions_per_op': 0.1,       # Amortized across 32 threads

    # Register file (256KB per SM, banked)
    'register_access_pj': 0.6,        # Per access (banked structure)

    # Execution units
    'tensor_core_mac_pj': 0.3,        # Tensor Core MAC (FP16/BF16)
    'cuda_core_mac_pj': 0.8,          # CUDA Core MAC (FP32)
    'tensor_core_utilization': 0.8,   # Fraction using Tensor Cores

    # SIMT Fixed Overhead (per kernel invocation)
    'kernel_launch_pj': 100_000.0,    # ~100 nJ per kernel launch
    'sm_activation_pj': 5_000.0,      # Per SM activation
    'memory_controller_pj': 50_000.0, # Memory subsystem setup

    # SIMT Variable Overhead
    'warp_scheduler_pj': 0.5,         # Per scheduling decision
    'thread_context_pj': 0.2,         # Per thread context access
    'scoreboard_pj': 0.3,             # Per dependency check

    # Coherence (L2+ modes only)
    'request_queue_pj': 1.0,          # Per request enqueue
    'coalesce_pj': 0.8,               # Address coalescing
    'l1_tag_pj': 0.5,                 # L1 tag lookup
    'l2_directory_pj': 1.5,           # L2 coherence directory
    'ordering_pj': 0.3,               # Memory ordering

    # Coherence (L1 mode - shared memory)
    'bank_conflict_pj': 0.3,          # Bank conflict check

    # Synchronization
    'divergence_mask_pj': 1.0,        # Warp divergence
    'reconverge_pj': 2.0,             # Reconvergence stack
    'barrier_pj': 10.0,               # Thread block barrier
    'atomic_pj': 5.0,                 # Atomic operation

    # Memory hierarchy (per byte)
    'shared_l1_pj_per_byte': 0.25,    # Shared memory / L1 unified
    'l2_cache_pj_per_byte': 0.8,      # L2 cache
    'hbm_pj_per_byte': 10.0,          # HBM2/HBM3

    # Divergence rate
    'warp_divergence_rate': 0.05,     # 5% of ops cause divergence
}


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

    Returns:
        CycleEnergyBreakdown with detailed energy breakdown

    GPU Energy Model:
    - Fixed infrastructure overhead (kernel launch, SM activation)
    - SIMT overhead scales with thread count
    - Coherence overhead depends on mode (minimal in L1/shared memory)
    - Memory access through Shared/L1 -> L2 -> HBM hierarchy
    """
    # GPU has no L3 cache - L3 mode should behave like DRAM mode
    # (L2 misses go directly to HBM)
    effective_mode = mode
    if mode == OperatingMode.L3_RESIDENT:
        effective_mode = OperatingMode.DRAM_RESIDENT

    ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[effective_mode]
    params = GPU_ENERGY_PARAMS

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU (NVIDIA H100 / Jetson)",
        architecture_class="SIMT Data Parallel"
    )

    # ==========================================================================
    # FIXED INFRASTRUCTURE OVERHEAD
    # ==========================================================================
    # GPUs have significant fixed costs that don't scale with workload
    num_active_sms = min(132, max(1, num_ops // 100))

    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Kernel launch overhead",
        params['kernel_launch_pj'],
        1
    )
    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        f"SM activation ({num_active_sms} SMs)",
        params['sm_activation_pj'],
        num_active_sms
    )
    breakdown.add_event(
        CyclePhase.SIMT_FIXED_OVERHEAD,
        "Memory controller infrastructure",
        params['memory_controller_pj'],
        1
    )

    # Calculate execution parameters
    warp_size = 32
    effective_threads = min(concurrent_threads, num_ops * 32)
    num_warps = max(1, effective_threads // warp_size)
    num_instructions = int(num_ops * params['instructions_per_op'])

    breakdown.num_cycles = max(1, num_instructions)
    breakdown.ops_per_cycle = concurrent_threads // max(1, num_warps)

    # ==========================================================================
    # Phase 1: INSTRUCTION FETCH (shared across warp)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        "I-cache read (per warp, shared by 32 threads)",
        params['instruction_fetch_pj'],
        max(1, num_instructions)
    )

    # ==========================================================================
    # Phase 2: INSTRUCTION DECODE (SIMT logic)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        "SIMT decode + predication",
        params['instruction_decode_pj'],
        max(1, num_instructions)
    )

    # ==========================================================================
    # Phase 3: OPERAND FETCH (Register File)
    # ==========================================================================
    num_register_accesses = num_ops * 2
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register file access (256KB/SM)",
        params['register_access_pj'],
        num_register_accesses
    )

    # ==========================================================================
    # Phase 4: EXECUTE (CUDA Cores + Tensor Cores)
    # ==========================================================================
    tensor_core_macs = int(num_ops * params['tensor_core_utilization'])
    cuda_core_macs = num_ops - tensor_core_macs

    breakdown.add_event(
        CyclePhase.EXECUTE,
        "Tensor Core MACs (FP16/BF16, 4x4x4)",
        params['tensor_core_mac_pj'],
        tensor_core_macs
    )
    breakdown.add_event(
        CyclePhase.EXECUTE,
        "CUDA Core MACs (FP32)",
        params['cuda_core_mac_pj'],
        cuda_core_macs
    )

    # ==========================================================================
    # Phase 5: SIMT THREAD MANAGEMENT
    # ==========================================================================
    num_scheduling_decisions = num_warps * max(1, num_instructions // max(1, num_warps))
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Warp scheduler (select eligible warps)",
        params['warp_scheduler_pj'],
        num_scheduling_decisions
    )

    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Thread context management",
        params['thread_context_pj'],
        effective_threads
    )

    num_dependency_checks = num_warps * 2
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Scoreboard (dependency tracking)",
        params['scoreboard_pj'],
        num_dependency_checks
    )

    # ==========================================================================
    # Phase 6: SIMT COHERENCE MACHINERY
    # ==========================================================================
    # Memory requests scale with UNIQUE cache lines accessed, not with warps.
    # Warps share data through caching - they don't each generate independent
    # requests for all memory. Coalescing combines requests from threads in a warp.
    #
    # Model: Each unique cache line generates coherence overhead once.
    # Additional overhead for warp-level coalescing decisions.

    cache_line_size = 128
    num_cache_lines = max(1, bytes_transferred // cache_line_size)

    # Coalescing overhead: one decision per warp per memory instruction
    # Estimate ~1 memory instruction per 10 ops
    num_coalesce_decisions = max(1, num_warps * (num_ops // (num_warps * 10) + 1))

    is_l1_resident = (mode == OperatingMode.L1_RESIDENT)

    if is_l1_resident:
        # L1-Resident: Shared memory has minimal coherence overhead
        # Bank conflict checks happen per-warp access
        num_shared_accesses = max(1, num_cache_lines * 2)  # Read + write
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Shared memory bank conflict check",
            params['bank_conflict_pj'],
            num_shared_accesses
        )
    else:
        # L2+ modes: Coherence machinery for unique cache lines
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Memory request queuing",
            params['request_queue_pj'],
            num_cache_lines
        )
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Address coalescing logic (per warp)",
            params['coalesce_pj'],
            num_coalesce_decisions
        )
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "L1 cache tag lookup",
            params['l1_tag_pj'],
            num_cache_lines
        )

        l1_miss_rate = 1.0 - ratios.l1_hit
        l2_directory_lookups = int(num_cache_lines * l1_miss_rate)
        if l2_directory_lookups > 0:
            breakdown.add_event(
                CyclePhase.SIMT_COHERENCE,
                "L2 coherence directory lookup",
                params['l2_directory_pj'],
                l2_directory_lookups
            )

        num_ordering_checks = max(1, num_cache_lines // 4)
        breakdown.add_event(
            CyclePhase.SIMT_COHERENCE,
            "Memory ordering/fence logic",
            params['ordering_pj'],
            num_ordering_checks
        )

    # ==========================================================================
    # Phase 7: SIMT SYNCHRONIZATION
    # ==========================================================================
    num_divergent = int(num_ops * params['warp_divergence_rate'])
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Warp divergence (predication masks)",
        params['divergence_mask_pj'],
        num_divergent
    )

    num_reconverge = max(1, num_divergent // 2)
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Reconvergence stack operations",
        params['reconverge_pj'],
        num_reconverge
    )

    num_barriers = max(1, num_ops // 1000)
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Thread block barriers (__syncthreads)",
        params['barrier_pj'],
        num_barriers
    )

    num_atomics = max(1, num_ops // 100)
    breakdown.add_event(
        CyclePhase.SIMT_SYNC,
        "Atomic operations (exclusive access)",
        params['atomic_pj'],
        num_atomics
    )

    # ==========================================================================
    # Phase 8: MEMORY ACCESS
    # ==========================================================================
    num_accesses = max(1, bytes_transferred // 4)

    l1_energy_per_access = params['shared_l1_pj_per_byte'] * 4
    l2_energy_per_access = params['l2_cache_pj_per_byte'] * 4
    hbm_energy_per_access = params['hbm_pj_per_byte'] * 4

    l1_accesses = num_accesses
    l1_hits = int(l1_accesses * ratios.l1_hit)
    l1_misses = l1_accesses - l1_hits

    l2_accesses = l1_misses
    l2_hits = int(l2_accesses * ratios.l2_hit)
    l2_misses = l2_accesses - l2_hits

    hbm_accesses = l2_misses  # GPU has no L3

    if l1_accesses > 0:
        mem_name = "Shared Memory" if is_l1_resident else "L1 cache"
        breakdown.add_event(
            CyclePhase.MEM_L1,
            f"{mem_name} ({l1_hits} hits, {l1_misses} misses)",
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

    if hbm_accesses > 0:
        breakdown.add_event(
            CyclePhase.MEM_HBM,
            f"HBM ({hbm_accesses} accesses)",
            hbm_energy_per_access,
            hbm_accesses
        )

    return breakdown
