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
  | Scoreboard       | Scoreboard       | Scoreboard       | Scoreboard       |
  | Operand Collector| Operand Collector| Operand Collector| Operand Collector|
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

GPU Request/Reply Execution Cycle (Stored Program Machine):
===========================================================
For EACH operation, the GPU must:
  1. Scoreboard lookup - Check RAW/WAW/WAR hazards for this warp
  2. Generate register addresses - Decode source/destination register IDs
  3. Operand collector - Gather operands from banked register file
  4. Bank arbitration - Resolve conflicts when multiple threads access same bank
  5. Send operands to ALU - Route data through operand network
  6. ALU execution - Perform the actual computation
  7. Write result back - Route result back to register file

This request/reply overhead is FUNDAMENTAL to stored program machines.
Each instruction must explicitly specify WHERE its operands come from
and WHERE its result goes.

Contrast with KPU (Spatial Dataflow):
=====================================
In KPU, operands ARRIVE at the PE based on SURE dependency vectors:
  1. Operands arrive from neighboring PE (no address lookup)
  2. ALU executes (same as GPU)
  3. Result written to LOCAL register (next PE's input, no arbitration)

NO scoreboard, NO operand collector, NO register arbitration!
The routing is determined at COMPILE TIME and baked into the spatial layout.

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

    # Register file per SM
    'register_file_kb': 256,          # Register file size
    'register_banks': 32,             # Number of banks (for conflict modeling)
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

    Models scalar/vector workloads executing on CUDA cores, including the
    full request/reply cycle overhead inherent to stored program machines.

    Request/Reply Cycle (per warp instruction):
        1. Scoreboard lookup - Check dependencies (RAW/WAW/WAR hazards)
        2. Register address generation - Decode src1, src2, dst addresses
        3. Operand collector - Gather operands from banked register file
        4. Bank arbitration - Resolve bank conflicts
        5. Operand routing - Send operands to execution units
        6. ALU execution - Perform computation
        7. Result writeback - Route result back to register file

    This overhead is UNAVOIDABLE in stored program machines. Each operation
    must explicitly look up where its data comes from and goes to.

    SM Partition Model:
        The SM has 4 partitions, each with:
        - 1 warp scheduler + scoreboard
        - 1 operand collector
        - 32 CUDA cores
        - 1/4 of the register file (banked)

        To fully utilize an SM, you need 4 warps active (one per partition).
        Native unit: 4 warp instructions = 4 x 32 = 128 MACs per SM-cycle.
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
            l3_hit=0.0,
        )
        if verbose:
            print(f"  L2 hit ratio: {l2_hit:.1%} (operator={operator_type.value}, "
                  f"ws={ws_bytes/1024/1024:.1f}MB, L2={l2_cache_bytes/1024/1024:.0f}MB, "
                  f"cold={l2_is_cold})")

    # ==========================================================================
    # Energy Parameters
    # ==========================================================================
    process_scale = tech_profile.process_node_nm / 4.0  # 4nm baseline

    num_partitions = SM_CONFIG['num_partitions']
    warp_size = SM_CONFIG['warp_size']
    macs_per_sm_cycle = SM_CONFIG['cuda_macs_per_sm_cycle']  # 128

    params = {
        # === CONTROL OVERHEAD (per warp instruction) ===
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'warp_schedule_pj': 0.5 * process_scale,

        # === REQUEST/REPLY CYCLE OVERHEAD (per warp instruction) ===
        # These are the costs of the stored-program execution model
        #
        # Scoreboard: Track dependencies for all active warps
        # ~0.3 pJ per lookup at 4nm (CAM-like structure)
        'scoreboard_lookup_pj': 0.3 * process_scale,

        # Register address generation: Decode register IDs for 32 threads
        # ~0.2 pJ per address at 4nm
        'reg_addr_gen_pj': 0.2 * process_scale,

        # Operand collector: Gather operands from banked register file
        # This is the main cost - must buffer operands until all are ready
        # ~0.8 pJ at 4nm (includes arbitration logic)
        'operand_collector_pj': 0.8 * process_scale,

        # Bank arbitration: Resolve conflicts when threads access same bank
        # Average ~10% conflicts, but arbitration logic always runs
        # ~0.3 pJ at 4nm
        'bank_arbitration_pj': 0.3 * process_scale,

        # Operand routing: Crossbar to route operands to execution units
        # ~0.4 pJ at 4nm for 32-wide routing
        'operand_routing_pj': 0.4 * process_scale,

        # Result routing: Route results back to register file
        # ~0.3 pJ at 4nm
        'result_routing_pj': 0.3 * process_scale,

        # === REGISTER FILE ACCESS (per warp instruction) ===
        # Vector register read/write for 32 threads
        'vector_register_read_pj': tech_profile.register_read_energy_pj * 0.25,
        'vector_register_write_pj': tech_profile.register_write_energy_pj * 0.25,

        # === COMPUTE (per op) ===
        'cuda_core_mac_pj': tech_profile.base_alu_energy_pj * 0.85,

        # === FIXED OVERHEAD ===
        'kernel_launch_pj': 100_000.0 * process_scale,

        # === MEMORY (per byte) ===
        'l1_shared_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'hbm_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # === SIMT OVERHEAD ===
        'coalesce_pj': 0.8 * process_scale,
        'divergence_pj': tech_profile.warp_divergence_energy_pj,
        'divergence_rate': 0.05,
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU CUDA (4 partitions x 32 cores)",
        architecture_class="SIMT Stored Program Machine"
    )

    # ==========================================================================
    # SM-CYCLE CALCULATIONS
    # ==========================================================================
    num_sm_cycles = max(1, (num_ops + macs_per_sm_cycle - 1) // macs_per_sm_cycle)
    num_warp_instructions = num_sm_cycles * num_partitions

    breakdown.num_cycles = num_sm_cycles
    breakdown.ops_per_cycle = macs_per_sm_cycle

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
    # INSTRUCTION FETCH/DECODE (per warp instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        f"I-cache read ({num_partitions} warps)",
        params['instruction_fetch_pj'],
        num_warp_instructions
    )

    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        f"SIMT decode ({num_partitions} partitions)",
        params['instruction_decode_pj'],
        num_warp_instructions
    )

    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        f"Warp scheduler ({num_partitions} per SM)",
        params['warp_schedule_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # REQUEST/REPLY CYCLE OVERHEAD (per warp instruction)
    # This is the fundamental cost of stored-program execution!
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Scoreboard lookup (RAW/WAW/WAR)",
        params['scoreboard_lookup_pj'],
        num_warp_instructions
    )

    # Register address generation for src1, src2, dst (3 addresses per instruction)
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register address generation (3 regs)",
        params['reg_addr_gen_pj'] * 3,
        num_warp_instructions
    )

    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Operand collector (gather operands)",
        params['operand_collector_pj'],
        num_warp_instructions
    )

    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Bank arbitration (conflict resolution)",
        params['bank_arbitration_pj'],
        num_warp_instructions
    )

    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Operand routing (crossbar to ALU)",
        params['operand_routing_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # REGISTER FILE ACCESS (per warp instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"Register file read src1 ({warp_size}-wide)",
        params['vector_register_read_pj'],
        num_warp_instructions
    )
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        f"Register file read src2 ({warp_size}-wide)",
        params['vector_register_read_pj'],
        num_warp_instructions
    )

    # ==========================================================================
    # COMPUTE (per op)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        f"CUDA Core MACs ({macs_per_sm_cycle}/SM-cycle)",
        params['cuda_core_mac_pj'],
        num_ops
    )

    # ==========================================================================
    # WRITEBACK (per warp instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Result routing (ALU to register file)",
        params['result_routing_pj'],
        num_warp_instructions
    )

    breakdown.add_event(
        CyclePhase.WRITEBACK,
        f"Register file write ({warp_size}-wide)",
        params['vector_register_write_pj'],
        num_warp_instructions
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
    # EXTERNAL MEMORY ACCESS (L1/L2/HBM)
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

    TensorCores still have request/reply overhead, but it's more efficient:
    - Matrix fragments are loaded in bulk (less address generation overhead)
    - Operand collector handles 4x4 tiles instead of individual scalars
    - Less bank conflicts due to structured access pattern

    However, the fundamental stored-program overhead remains:
    - Still need scoreboard for MMA instruction dependencies
    - Still need operand collector to gather matrix fragments
    - Still need to route results back to register file

    TensorCore Operation (H100):
        - Input: Two 4x4 matrices (FP16/BF16)
        - Output: One 4x4 matrix (FP32 accumulator)
        - MACs: 4 x 4 x 4 = 64 MACs per MMA instruction
        - Total: 4 TensorCores x 64 = 256 MACs per SM-cycle
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
        # === CONTROL OVERHEAD (per MMA instruction) ===
        'instruction_fetch_pj': tech_profile.instruction_fetch_energy_pj,
        'instruction_decode_pj': tech_profile.instruction_decode_energy_pj,
        'warp_schedule_pj': 0.5 * process_scale,

        # === REQUEST/REPLY CYCLE OVERHEAD (per MMA instruction) ===
        # TensorCore has reduced overhead vs CUDA cores due to bulk access
        #
        # Scoreboard: Still need to track MMA dependencies
        'scoreboard_lookup_pj': 0.3 * process_scale,

        # Register address generation: Fewer addresses (matrix fragments vs scalars)
        # ~0.15 pJ per MMA (vs 0.2 for CUDA)
        'reg_addr_gen_pj': 0.15 * process_scale,

        # Operand collector: Bulk load of 4x4 tiles is more efficient
        # ~0.5 pJ at 4nm (vs 0.8 for CUDA)
        'operand_collector_pj': 0.5 * process_scale,

        # Bank arbitration: Structured access pattern reduces conflicts
        # ~0.2 pJ at 4nm (vs 0.3 for CUDA)
        'bank_arbitration_pj': 0.2 * process_scale,

        # Operand routing: Route matrix tiles to TensorCore
        # ~0.3 pJ at 4nm (vs 0.4 for CUDA)
        'operand_routing_pj': 0.3 * process_scale,

        # Result routing: Route 4x4 result tile back
        # ~0.25 pJ at 4nm
        'result_routing_pj': 0.25 * process_scale,

        # === REGISTER FILE ACCESS ===
        # Matrix registers (larger but more efficient access pattern)
        'matrix_register_read_pj': tech_profile.register_read_energy_pj * 0.4,
        'matrix_register_write_pj': tech_profile.register_write_energy_pj * 0.4,

        # === COMPUTE (per op) ===
        'tensor_core_mac_pj': tech_profile.tensor_core_mac_energy_pj,

        # === FIXED OVERHEAD ===
        'kernel_launch_pj': 100_000.0 * process_scale,

        # === MEMORY (per byte) ===
        'l1_shared_pj_per_byte': tech_profile.sram_energy_per_byte_pj,
        'l2_cache_pj_per_byte': tech_profile.l2_cache_energy_per_byte_pj,
        'hbm_pj_per_byte': tech_profile.offchip_energy_per_byte_pj,

        # === SIMT OVERHEAD (minimal for matrix ops) ===
        'coalesce_pj': 0.8 * process_scale,
        'divergence_pj': tech_profile.warp_divergence_energy_pj,
        'divergence_rate': 0.01,
    }

    breakdown = CycleEnergyBreakdown(
        architecture_name="GPU TensorCore (4 partitions x 64 MACs)",
        architecture_class="SIMT Stored Program Machine (TensorCores)"
    )

    # ==========================================================================
    # SM-CYCLE CALCULATIONS
    # ==========================================================================
    num_sm_cycles = max(1, (num_ops + macs_per_sm_cycle - 1) // macs_per_sm_cycle)
    num_mma_instructions = num_sm_cycles * num_partitions

    breakdown.num_cycles = num_sm_cycles
    breakdown.ops_per_cycle = macs_per_sm_cycle

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
    # INSTRUCTION FETCH/DECODE (per MMA instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.INSTRUCTION_FETCH,
        f"I-cache read ({num_partitions} MMA instrs)",
        params['instruction_fetch_pj'],
        num_mma_instructions
    )

    breakdown.add_event(
        CyclePhase.INSTRUCTION_DECODE,
        f"MMA decode ({num_partitions} partitions)",
        params['instruction_decode_pj'],
        num_mma_instructions
    )

    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        f"Warp scheduler MMA ({num_partitions} per SM)",
        params['warp_schedule_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # REQUEST/REPLY CYCLE OVERHEAD (per MMA instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.SIMT_THREAD_MGMT,
        "Scoreboard lookup (MMA dependencies)",
        params['scoreboard_lookup_pj'],
        num_mma_instructions
    )

    # Register address generation for A, B, C fragments
    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Register address gen (A, B, C fragments)",
        params['reg_addr_gen_pj'] * 3,
        num_mma_instructions
    )

    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Operand collector (matrix tiles)",
        params['operand_collector_pj'],
        num_mma_instructions
    )

    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Bank arbitration (tile access)",
        params['bank_arbitration_pj'],
        num_mma_instructions
    )

    breakdown.add_event(
        CyclePhase.OPERAND_FETCH,
        "Operand routing (tiles to TensorCore)",
        params['operand_routing_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # REGISTER FILE ACCESS (per MMA instruction)
    # ==========================================================================
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

    # ==========================================================================
    # COMPUTE (TensorCore MMA operations)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.EXECUTE,
        f"TensorCore MACs ({tc_macs_per_op} per TC, {num_partitions} TCs)",
        params['tensor_core_mac_pj'],
        num_ops
    )

    # ==========================================================================
    # WRITEBACK (per MMA instruction)
    # ==========================================================================
    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Result routing (TensorCore to regfile)",
        params['result_routing_pj'],
        num_mma_instructions
    )

    breakdown.add_event(
        CyclePhase.WRITEBACK,
        "Matrix register write (C accumulator, 4x4)",
        params['matrix_register_write_pj'],
        num_mma_instructions
    )

    # ==========================================================================
    # SIMT OVERHEAD (minimal for matrix ops)
    # ==========================================================================
    num_mem_instructions = max(1, num_mma_instructions // 16)
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
    # EXTERNAL MEMORY ACCESS
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

    This function provides backwards compatibility by selecting CUDA or TensorCore
    models based on tensor_core_utilization.
    """
    if tensor_core_utilization >= 0.5:
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
