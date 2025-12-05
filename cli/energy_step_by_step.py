#!/usr/bin/env python
"""
Energy Step-by-Step: Follow Along Educational Walkthrough

This script walks through a small operation (e.g., 4x4 matrix multiply) step by step,
showing EXACTLY what happens at each stage and the energy consumed.

Unlike energy_walkthrough.py which shows summary tables, this script is designed
for teaching - you can follow along and understand where every picojoule goes.

Usage:
    ./cli/energy_step_by_step.py                    # Default: 4x4 matmul on all architectures
    ./cli/energy_step_by_step.py --arch cpu         # CPU only
    ./cli/energy_step_by_step.py --arch gpu         # GPU (CUDA cores) only
    ./cli/energy_step_by_step.py --arch gpu-tc      # GPU (TensorCores) only
    ./cli/energy_step_by_step.py --arch tpu         # TPU only
    ./cli/energy_step_by_step.py --arch kpu         # KPU only
    ./cli/energy_step_by_step.py --size 8           # 8x8 matrix multiply
    ./cli/energy_step_by_step.py --ops 64           # Explicit 64 MACs
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphs.hardware.technology_profile import ARCH_COMPARISON_8NM_X86
from graphs.hardware.operand_fetch import (
    CPUOperandFetchModel,
    GPUOperandFetchModel,
    TPUOperandFetchModel,
    KPUOperandFetchModel,
    compare_operand_fetch_energy,
    format_comparison_table,
)


# =============================================================================
# Energy Constants (from technology profile, ~8nm process)
# =============================================================================

# These are approximate values for educational purposes
# Real values come from the technology profile

ENERGY_CONSTANTS = {
    # Instruction handling (stored program machines)
    'instruction_fetch_pj': 25.0,       # Fetch from I-cache
    'instruction_decode_pj': 15.0,      # Decode logic
    'operand_fetch_pj': 10.0,           # Register file read
    'writeback_pj': 8.0,                # Register file write

    # Compute (per MAC operation)
    'fp32_mac_pj': 4.0,                 # FP32 multiply-accumulate
    'fp16_mac_pj': 1.0,                 # FP16 multiply-accumulate
    'int8_mac_pj': 0.2,                 # INT8 multiply-accumulate

    # Memory access (per byte)
    'register_pj_per_byte': 0.1,        # Register file
    'l1_cache_pj_per_byte': 0.5,        # L1 cache / shared memory
    'l2_cache_pj_per_byte': 2.0,        # L2 cache
    'l3_cache_pj_per_byte': 5.0,        # L3 cache (CPU only)
    'dram_pj_per_byte': 20.0,           # DRAM/HBM access

    # GPU-specific
    'warp_scheduler_pj': 50.0,          # Warp scheduler decision
    'simt_stack_pj': 5.0,               # Divergence tracking

    # TPU-specific
    'systolic_weight_load_pj': 0.5,     # Load weight into PE
    'systolic_data_shift_pj': 0.1,      # Shift data between PEs
    'systolic_accumulator_pj': 0.3,     # Accumulator update

    # KPU-specific
    'domain_config_pj': 20.0,           # Domain program config (amortized)
    'stream_token_pj': 0.05,            # Stream a data token
    'tile_local_pj_per_byte': 0.3,      # Tile-local scratchpad
}


# =============================================================================
# Architecture Timing Parameters
# =============================================================================
# Each architecture has different clock frequencies and MACs per cycle.
# These determine the TIME to execute operations, which combined with
# energy gives us POWER.

ARCH_TIMING = {
    'cpu': {
        'name': 'CPU (AVX-512)',
        'frequency_ghz': 3.5,           # Typical x86 boost clock
        'macs_per_cycle': 16,           # AVX-512: 16 FP32 MACs
        'description': 'x86-64 with AVX-512 @ 3.5 GHz',
    },
    'gpu_cuda': {
        'name': 'GPU (CUDA)',
        'frequency_ghz': 1.5,           # Typical SM clock
        'macs_per_cycle': 128,          # 4 partitions x 32 cores
        'description': 'NVIDIA SM with CUDA cores @ 1.5 GHz',
    },
    'gpu_tc': {
        'name': 'GPU (TensorCore)',
        'frequency_ghz': 1.5,           # Same SM clock
        'macs_per_cycle': 256,          # 4 partitions x 64 MACs (4x4x4)
        'description': 'NVIDIA SM with TensorCores @ 1.5 GHz',
    },
    'tpu': {
        'name': 'TPU (Systolic)',
        'frequency_ghz': 1.0,           # TPU typically runs slower
        'macs_per_cycle': 16384,        # 128x128 systolic array
        'description': 'Google TPU systolic array @ 1.0 GHz',
    },
    'kpu': {
        'name': 'KPU (Domain Flow)',
        'frequency_ghz': 1.0,           # Similar to TPU
        'macs_per_cycle': 16384,        # 64 tiles x 256 MACs
        'description': 'Domain-flow accelerator @ 1.0 GHz',
    },
}


def calculate_power_at_1tops(energy_per_mac_pj: float, macs_per_cycle: int, freq_ghz: float) -> float:
    """
    Calculate power consumed when delivering 1 TOPS of throughput.

    Power = Energy / Time

    For 1 TOPS (10^12 MACs/sec):
        Energy for 1 TOPS work = 10^12 MACs × E_mac (pJ) × 10^-12 J/pJ = E_mac (J)
        Time to do 1 TOPS work = 10^12 MACs / (macs_per_cycle × freq_hz)
                               = 10^12 / (macs_per_cycle × freq_ghz × 10^9) seconds

        Power = E_mac (J) / Time (s)
              = E_mac × macs_per_cycle × freq_ghz × 10^9 / 10^12
              = E_mac × macs_per_cycle × freq_ghz / 1000  (Watts)

    Args:
        energy_per_mac_pj: Energy per MAC in picojoules
        macs_per_cycle: Number of MACs executed per cycle (concurrency)
        freq_ghz: Clock frequency in GHz

    Returns:
        Power in Watts to deliver 1 TOPS
    """
    return energy_per_mac_pj * macs_per_cycle * freq_ghz / 1000.0


def calculate_power_metrics(energy_pj: float, num_macs: int, arch_key: str) -> dict:
    """
    Calculate power metrics for an architecture.

    Args:
        energy_pj: Total energy in picojoules
        num_macs: Number of MAC operations
        arch_key: Architecture key (cpu, gpu_cuda, gpu_tc, tpu, kpu)

    Returns:
        dict with power and throughput metrics
    """
    arch = ARCH_TIMING[arch_key]
    freq_ghz = arch['frequency_ghz']
    macs_per_cycle = arch['macs_per_cycle']

    # Energy per MAC (the fundamental metric from the walkthrough)
    energy_per_mac_pj = energy_pj / num_macs

    # Throughput of this architecture
    # macs_per_cycle × freq_ghz × 10^9 = MACs/sec
    # Divide by 10^12 to get TOPS
    throughput_tops = macs_per_cycle * freq_ghz / 1000.0  # TOPS

    # Power to deliver 1 TOPS of work
    power_for_1tops_w = calculate_power_at_1tops(energy_per_mac_pj, macs_per_cycle, freq_ghz)

    return {
        'energy_per_mac_pj': energy_per_mac_pj,
        'macs_per_cycle': macs_per_cycle,
        'frequency_ghz': freq_ghz,
        'throughput_tops': throughput_tops,
        'power_for_1tops_w': power_for_1tops_w,
    }


def format_energy(pj: float) -> str:
    """Format energy with appropriate units."""
    if pj >= 1_000_000:
        return f"{pj/1_000_000:,.2f} uJ"
    elif pj >= 1_000:
        return f"{pj/1_000:,.2f} nJ"
    elif pj >= 0.01:
        return f"{pj:,.2f} pJ"
    else:
        return f"{pj*1000:,.2f} fJ"


def print_step(step_num: int, description: str, energy_pj: float,
               details: str = "", running_total: float = 0):
    """Print a single step with energy accounting."""
    total_str = f"[Running total: {format_energy(running_total + energy_pj)}]"
    print(f"    Step {step_num}: {description}")
    if details:
        print(f"           {details}")
    print(f"           Energy: +{format_energy(energy_pj):>12}  {total_str}")
    print()
    return running_total + energy_pj


def print_divider(char="-", width=80):
    print(f"    {char * width}")


# =============================================================================
# CPU Walkthrough
# =============================================================================

def walkthrough_cpu(num_macs: int, matrix_dim: int = None):
    """
    Step-by-step walkthrough of CPU execution.

    CPU executes as a stored program machine with SIMD (AVX-512):
    - Each instruction: Fetch -> Decode -> Execute -> Writeback
    - AVX-512: 16 FP32 MACs per instruction
    """
    print()
    print("=" * 90)
    print("  CPU (x86-64 with AVX-512) STEP-BY-STEP WALKTHROUGH")
    print("=" * 90)
    print()

    if matrix_dim:
        print(f"  Workload: {matrix_dim}x{matrix_dim} matrix multiply = {num_macs:,} MACs")
    else:
        print(f"  Workload: {num_macs:,} MAC operations")
    print()

    print("  Architecture Overview:")
    print("  -----------------------")
    print("  - Stored Program Machine (Von Neumann architecture)")
    print("  - AVX-512: 512-bit SIMD = 16 FP32 values per instruction")
    print("  - Each VFMADD instruction: 16 parallel multiply-accumulates")
    print()

    # Calculate number of SIMD instructions needed
    simd_width = 16
    num_instructions = (num_macs + simd_width - 1) // simd_width

    print(f"  Execution Plan:")
    print(f"  - Total MACs needed: {num_macs:,}")
    print(f"  - MACs per AVX-512 instruction: {simd_width}")
    print(f"  - Instructions needed: {num_instructions:,}")
    print()

    # Show detailed walkthrough for first few instructions
    show_detailed = min(3, num_instructions)

    print("  DETAILED EXECUTION (first few instructions):")
    print_divider("=")
    print()

    running_total = 0.0
    step = 1

    for i in range(show_detailed):
        print(f"  --- Instruction {i+1} of {num_instructions}: VFMADD231PS (Fused Multiply-Add) ---")
        print()

        # Step 1: Instruction Fetch
        running_total = print_step(
            step, "Instruction Fetch",
            ENERGY_CONSTANTS['instruction_fetch_pj'],
            "Fetch VFMADD231PS from L1 I-cache (32 bytes)",
            running_total
        )
        step += 1

        # Step 2: Instruction Decode
        running_total = print_step(
            step, "Instruction Decode",
            ENERGY_CONSTANTS['instruction_decode_pj'],
            "Decode AVX-512 instruction, identify operands (ZMM0, ZMM1, ZMM2)",
            running_total
        )
        step += 1

        # Step 3: Operand Fetch (read 3 registers)
        operand_bytes = 3 * 64  # 3 ZMM registers, 64 bytes each
        operand_energy = operand_bytes * ENERGY_CONSTANTS['register_pj_per_byte']
        running_total = print_step(
            step, "Operand Fetch",
            operand_energy + ENERGY_CONSTANTS['operand_fetch_pj'],
            f"Read ZMM0, ZMM1, ZMM2 ({operand_bytes} bytes from register file)",
            running_total
        )
        step += 1

        # Step 4: Execute (16 FP32 MACs in parallel)
        compute_energy = simd_width * ENERGY_CONSTANTS['fp32_mac_pj']
        running_total = print_step(
            step, "Execute (SIMD Compute)",
            compute_energy,
            f"16 parallel FP32 MACs: ZMM0 = ZMM1 * ZMM2 + ZMM0",
            running_total
        )
        step += 1

        # Step 5: Writeback
        writeback_bytes = 64  # 1 ZMM register
        writeback_energy = writeback_bytes * ENERGY_CONSTANTS['register_pj_per_byte']
        running_total = print_step(
            step, "Writeback",
            writeback_energy + ENERGY_CONSTANTS['writeback_pj'],
            f"Write result to ZMM0 ({writeback_bytes} bytes to register file)",
            running_total
        )
        step += 1

        print()

    # Calculate remaining instructions
    remaining_instructions = num_instructions - show_detailed
    if remaining_instructions > 0:
        energy_per_instruction = running_total / show_detailed
        remaining_energy = remaining_instructions * energy_per_instruction
        print(f"  ... {remaining_instructions:,} more instructions (same pattern) ...")
        print(f"      Additional energy: +{format_energy(remaining_energy)}")
        running_total += remaining_energy
        print()

    # Summary
    print_divider("=")
    print()
    print("  ENERGY SUMMARY (CPU):")
    print_divider("-")

    # Recalculate components
    instruction_overhead = num_instructions * (
        ENERGY_CONSTANTS['instruction_fetch_pj'] +
        ENERGY_CONSTANTS['instruction_decode_pj']
    )
    operand_overhead = num_instructions * (
        ENERGY_CONSTANTS['operand_fetch_pj'] +
        3 * 64 * ENERGY_CONSTANTS['register_pj_per_byte']
    )
    compute_energy = num_macs * ENERGY_CONSTANTS['fp32_mac_pj']
    writeback_overhead = num_instructions * (
        ENERGY_CONSTANTS['writeback_pj'] +
        64 * ENERGY_CONSTANTS['register_pj_per_byte']
    )

    total_control = instruction_overhead + operand_overhead + writeback_overhead
    total_compute = compute_energy
    total_energy = total_control + total_compute

    print(f"    Control Overhead:     {format_energy(total_control):>12} ({total_control/total_energy*100:.1f}%)")
    print(f"      - Instruction F/D:  {format_energy(instruction_overhead):>12}")
    print(f"      - Operand Fetch:    {format_energy(operand_overhead):>12}")
    print(f"      - Writeback:        {format_energy(writeback_overhead):>12}")
    print(f"    Compute (MACs):       {format_energy(total_compute):>12} ({total_compute/total_energy*100:.1f}%)")
    print_divider("-")
    print(f"    TOTAL:                {format_energy(total_energy):>12}")
    print(f"    Energy per MAC:       {format_energy(total_energy/num_macs):>12}")
    print()

    return total_energy


# =============================================================================
# GPU CUDA Walkthrough
# =============================================================================

def walkthrough_gpu_cuda(num_macs: int, matrix_dim: int = None):
    """
    Step-by-step walkthrough of GPU CUDA core execution.

    GPU executes with SIMT (Single Instruction Multiple Thread):
    - Warps of 32 threads execute in lockstep
    - Each SM has 4 partitions, each with 32 CUDA cores
    - Full SM: 128 CUDA cores = 128 MACs per cycle
    """
    print()
    print("=" * 90)
    print("  GPU (NVIDIA CUDA Cores) STEP-BY-STEP WALKTHROUGH")
    print("=" * 90)
    print()

    if matrix_dim:
        print(f"  Workload: {matrix_dim}x{matrix_dim} matrix multiply = {num_macs:,} MACs")
    else:
        print(f"  Workload: {num_macs:,} MAC operations")
    print()

    print("  Architecture Overview:")
    print("  -----------------------")
    print("  - SIMT: Single Instruction Multiple Thread")
    print("  - Warp: 32 threads executing same instruction in lockstep")
    print("  - SM has 4 partitions, each with:")
    print("      - 1 Warp Scheduler")
    print("      - 32 CUDA cores (32 FP32 MACs/cycle)")
    print("      - Portion of register file and shared memory")
    print("  - Full SM throughput: 4 warps x 32 cores = 128 MACs/cycle")
    print()

    # Calculate execution
    macs_per_warp = 32
    macs_per_sm_cycle = 128  # 4 partitions x 32 cores
    num_warp_instructions = (num_macs + macs_per_warp - 1) // macs_per_warp
    num_sm_cycles = (num_macs + macs_per_sm_cycle - 1) // macs_per_sm_cycle

    print(f"  Execution Plan:")
    print(f"  - Total MACs needed: {num_macs:,}")
    print(f"  - MACs per warp instruction: {macs_per_warp}")
    print(f"  - MACs per SM cycle (4 warps): {macs_per_sm_cycle}")
    print(f"  - Warp instructions needed: {num_warp_instructions:,}")
    print(f"  - SM cycles needed: {num_sm_cycles:,}")
    print()

    print("  DETAILED EXECUTION (first SM cycle):")
    print_divider("=")
    print()

    running_total = 0.0
    step = 1

    # Show one complete SM cycle (4 partitions working in parallel)
    print("  --- SM Cycle 1: 4 Partitions Execute in Parallel ---")
    print()

    # Each partition has its own warp scheduler
    for partition in range(4):
        print(f"  [Partition {partition}] (32 CUDA cores)")
        print()

        # Warp Scheduler
        running_total = print_step(
            step, f"Warp Scheduler (Partition {partition})",
            ENERGY_CONSTANTS['warp_scheduler_pj'],
            f"Select ready warp, issue FFMA instruction to 32 cores",
            running_total
        )
        step += 1

        # Instruction Fetch/Decode (shared per warp)
        running_total = print_step(
            step, f"Instruction Fetch/Decode",
            ENERGY_CONSTANTS['instruction_fetch_pj'] + ENERGY_CONSTANTS['instruction_decode_pj'],
            f"Fetch FFMA.F32 from I-cache, decode for 32 threads",
            running_total
        )
        step += 1

        # Operand Fetch (32 threads x 3 operands x 4 bytes)
        operand_bytes = 32 * 3 * 4  # 32 threads, 3 FP32 operands each
        operand_energy = operand_bytes * ENERGY_CONSTANTS['register_pj_per_byte']
        running_total = print_step(
            step, f"Operand Fetch (32 threads)",
            operand_energy,
            f"Read operands from register file ({operand_bytes} bytes for warp)",
            running_total
        )
        step += 1

        # Execute (32 MACs in parallel)
        compute_energy = 32 * ENERGY_CONSTANTS['fp32_mac_pj']
        running_total = print_step(
            step, f"Execute (32 CUDA Cores)",
            compute_energy,
            f"32 parallel FP32 MACs (one per thread in warp)",
            running_total
        )
        step += 1

        # Writeback
        writeback_bytes = 32 * 4  # 32 results x 4 bytes
        writeback_energy = writeback_bytes * ENERGY_CONSTANTS['register_pj_per_byte']
        running_total = print_step(
            step, f"Writeback",
            writeback_energy,
            f"Write 32 results to register file ({writeback_bytes} bytes)",
            running_total
        )
        step += 1

        print()

    # Calculate remaining cycles
    remaining_cycles = num_sm_cycles - 1
    if remaining_cycles > 0:
        energy_per_cycle = running_total
        remaining_energy = remaining_cycles * energy_per_cycle
        print(f"  ... {remaining_cycles:,} more SM cycles (same pattern) ...")
        print(f"      Additional energy: +{format_energy(remaining_energy)}")
        running_total += remaining_energy
        print()

    # Summary
    print_divider("=")
    print()
    print("  ENERGY SUMMARY (GPU CUDA):")
    print_divider("-")

    # Recalculate components
    warp_scheduler_energy = num_warp_instructions * ENERGY_CONSTANTS['warp_scheduler_pj']
    instruction_energy = num_warp_instructions * (
        ENERGY_CONSTANTS['instruction_fetch_pj'] +
        ENERGY_CONSTANTS['instruction_decode_pj']
    )
    operand_energy = num_warp_instructions * 32 * 3 * 4 * ENERGY_CONSTANTS['register_pj_per_byte']
    compute_energy = num_macs * ENERGY_CONSTANTS['fp32_mac_pj']
    writeback_energy = num_warp_instructions * 32 * 4 * ENERGY_CONSTANTS['register_pj_per_byte']

    total_control = warp_scheduler_energy + instruction_energy + operand_energy + writeback_energy
    total_compute = compute_energy
    total_energy = total_control + total_compute

    print(f"    Control Overhead:     {format_energy(total_control):>12} ({total_control/total_energy*100:.1f}%)")
    print(f"      - Warp Scheduling:  {format_energy(warp_scheduler_energy):>12}")
    print(f"      - Instruction F/D:  {format_energy(instruction_energy):>12}")
    print(f"      - Operand/Writeback:{format_energy(operand_energy + writeback_energy):>12}")
    print(f"    Compute (MACs):       {format_energy(total_compute):>12} ({total_compute/total_energy*100:.1f}%)")
    print_divider("-")
    print(f"    TOTAL:                {format_energy(total_energy):>12}")
    print(f"    Energy per MAC:       {format_energy(total_energy/num_macs):>12}")
    print()

    return total_energy


# =============================================================================
# GPU TensorCore Walkthrough
# =============================================================================

def walkthrough_gpu_tensorcore(num_macs: int, matrix_dim: int = None):
    """
    Step-by-step walkthrough of GPU TensorCore execution.

    TensorCores do matrix operations directly:
    - Each TensorCore: 4x4x4 = 64 MACs per instruction (FP16 or mixed precision)
    - SM partition has 1 TensorCore
    - Full SM: 4 TensorCores = 256 MACs per cycle
    """
    print()
    print("=" * 90)
    print("  GPU (NVIDIA TensorCores) STEP-BY-STEP WALKTHROUGH")
    print("=" * 90)
    print()

    if matrix_dim:
        print(f"  Workload: {matrix_dim}x{matrix_dim} matrix multiply = {num_macs:,} MACs")
    else:
        print(f"  Workload: {num_macs:,} MAC operations")
    print()

    print("  Architecture Overview:")
    print("  -----------------------")
    print("  - TensorCore: Hardware matrix multiply unit")
    print("  - Each MMA instruction: D = A x B + C")
    print("      - A: 4x4 matrix, B: 4x4 matrix, C: 4x4 accumulator")
    print("      - 4x4x4 = 64 MACs per TensorCore instruction")
    print("  - SM has 4 partitions, each with 1 TensorCore")
    print("  - Full SM throughput: 4 x 64 = 256 MACs/cycle")
    print()

    # Calculate execution
    macs_per_mma = 64  # 4x4x4 matrix multiply
    macs_per_sm_cycle = 256  # 4 TensorCores
    num_mma_instructions = (num_macs + macs_per_mma - 1) // macs_per_mma
    num_sm_cycles = (num_macs + macs_per_sm_cycle - 1) // macs_per_sm_cycle

    print(f"  Execution Plan:")
    print(f"  - Total MACs needed: {num_macs:,}")
    print(f"  - MACs per MMA instruction: {macs_per_mma}")
    print(f"  - MACs per SM cycle (4 TensorCores): {macs_per_sm_cycle}")
    print(f"  - MMA instructions needed: {num_mma_instructions:,}")
    print(f"  - SM cycles needed: {num_sm_cycles:,}")
    print()

    print("  DETAILED EXECUTION (first SM cycle):")
    print_divider("=")
    print()

    running_total = 0.0
    step = 1

    print("  --- SM Cycle 1: 4 TensorCores Execute in Parallel ---")
    print()

    # Each partition executes one MMA instruction
    for partition in range(4):
        print(f"  [Partition {partition}] (1 TensorCore)")
        print()

        # Warp Scheduler
        running_total = print_step(
            step, f"Warp Scheduler",
            ENERGY_CONSTANTS['warp_scheduler_pj'],
            f"Issue MMA instruction to TensorCore",
            running_total
        )
        step += 1

        # Load A matrix (4x4 = 16 FP16 values = 32 bytes)
        a_bytes = 16 * 2  # 4x4 FP16
        running_total = print_step(
            step, f"Load Matrix A (4x4)",
            a_bytes * ENERGY_CONSTANTS['register_pj_per_byte'],
            f"Load 4x4 FP16 matrix from registers ({a_bytes} bytes)",
            running_total
        )
        step += 1

        # Load B matrix
        b_bytes = 16 * 2
        running_total = print_step(
            step, f"Load Matrix B (4x4)",
            b_bytes * ENERGY_CONSTANTS['register_pj_per_byte'],
            f"Load 4x4 FP16 matrix from registers ({b_bytes} bytes)",
            running_total
        )
        step += 1

        # Load C accumulator (4x4 FP32 = 64 bytes for accumulator precision)
        c_bytes = 16 * 4  # 4x4 FP32 accumulator
        running_total = print_step(
            step, f"Load Accumulator C (4x4)",
            c_bytes * ENERGY_CONSTANTS['register_pj_per_byte'],
            f"Load 4x4 FP32 accumulator ({c_bytes} bytes)",
            running_total
        )
        step += 1

        # Execute MMA (64 MACs in systolic-like operation)
        # TensorCores use lower energy per MAC due to specialized design
        compute_energy = 64 * ENERGY_CONSTANTS['fp16_mac_pj'] * 1.5  # Slightly higher for mixed precision
        running_total = print_step(
            step, f"Execute MMA (TensorCore)",
            compute_energy,
            f"4x4x4 = 64 MACs: D = A x B + C (mixed precision)",
            running_total
        )
        step += 1

        # Store D result
        d_bytes = 16 * 4  # 4x4 FP32 result
        running_total = print_step(
            step, f"Store Result D (4x4)",
            d_bytes * ENERGY_CONSTANTS['register_pj_per_byte'],
            f"Store 4x4 FP32 result ({d_bytes} bytes)",
            running_total
        )
        step += 1

        print()

    # Calculate remaining
    remaining_cycles = num_sm_cycles - 1
    if remaining_cycles > 0:
        energy_per_cycle = running_total
        remaining_energy = remaining_cycles * energy_per_cycle
        print(f"  ... {remaining_cycles:,} more SM cycles (same pattern) ...")
        print(f"      Additional energy: +{format_energy(remaining_energy)}")
        running_total += remaining_energy
        print()

    # Summary
    print_divider("=")
    print()
    print("  ENERGY SUMMARY (GPU TensorCore):")
    print_divider("-")

    # Recalculate
    warp_scheduler_energy = num_mma_instructions * ENERGY_CONSTANTS['warp_scheduler_pj']
    data_load_energy = num_mma_instructions * (16*2 + 16*2 + 16*4) * ENERGY_CONSTANTS['register_pj_per_byte']
    compute_energy = num_macs * ENERGY_CONSTANTS['fp16_mac_pj'] * 1.5
    writeback_energy = num_mma_instructions * 16 * 4 * ENERGY_CONSTANTS['register_pj_per_byte']

    total_control = warp_scheduler_energy + data_load_energy + writeback_energy
    total_compute = compute_energy
    total_energy = total_control + total_compute

    print(f"    Control Overhead:     {format_energy(total_control):>12} ({total_control/total_energy*100:.1f}%)")
    print(f"      - Warp Scheduling:  {format_energy(warp_scheduler_energy):>12}")
    print(f"      - Data Load/Store:  {format_energy(data_load_energy + writeback_energy):>12}")
    print(f"    Compute (MACs):       {format_energy(total_compute):>12} ({total_compute/total_energy*100:.1f}%)")
    print_divider("-")
    print(f"    TOTAL:                {format_energy(total_energy):>12}")
    print(f"    Energy per MAC:       {format_energy(total_energy/num_macs):>12}")
    print()

    return total_energy


# =============================================================================
# TPU Walkthrough
# =============================================================================

def walkthrough_tpu(num_macs: int, matrix_dim: int = None):
    """
    Step-by-step walkthrough of TPU systolic array execution.

    TPU uses a weight-stationary systolic array:
    - 128x128 = 16,384 MACs per cycle
    - Weights are loaded once and stay in place
    - Data flows through the array
    """
    print()
    print("=" * 90)
    print("  TPU (Google Systolic Array) STEP-BY-STEP WALKTHROUGH")
    print("=" * 90)
    print()

    if matrix_dim:
        print(f"  Workload: {matrix_dim}x{matrix_dim} matrix multiply = {num_macs:,} MACs")
    else:
        print(f"  Workload: {num_macs:,} MAC operations")
    print()

    print("  Architecture Overview:")
    print("  -----------------------")
    print("  - Weight-Stationary Systolic Array")
    print("  - 128 x 128 = 16,384 Processing Elements (PEs)")
    print("  - Each PE: 1 MAC unit + local register for weight")
    print("  - Execution phases:")
    print("      1. Load weights into array (weights stay in place)")
    print("      2. Stream activations through array")
    print("      3. Results accumulate and drain out")
    print("  - NO per-operation instruction fetch/decode!")
    print()

    # Calculate execution
    tile_size = 128
    macs_per_tile = tile_size * tile_size  # 16,384

    # For matrix multiply, we need to tile
    if matrix_dim:
        tiles_per_dim = (matrix_dim + tile_size - 1) // tile_size
        num_tiles = tiles_per_dim * tiles_per_dim * tiles_per_dim  # For AxB
    else:
        num_tiles = (num_macs + macs_per_tile - 1) // macs_per_tile

    print(f"  Execution Plan:")
    print(f"  - Total MACs needed: {num_macs:,}")
    print(f"  - MACs per tile (128x128): {macs_per_tile:,}")
    print(f"  - Tile operations needed: {num_tiles:,}")
    print()

    print("  DETAILED EXECUTION (first tile operation):")
    print_divider("=")
    print()

    running_total = 0.0
    step = 1

    print("  --- Tile Operation 1: 128x128 Systolic Multiply ---")
    print()

    # Phase 1: Configuration (once per tile)
    running_total = print_step(
        step, "Tile Configuration",
        ENERGY_CONSTANTS['domain_config_pj'],
        "Configure tile dimensions, accumulator mode (minimal - no instruction stream)",
        running_total
    )
    step += 1

    # Phase 2: Weight Loading
    weight_bytes = tile_size * tile_size * 4  # 128x128 FP32 weights
    weight_load_energy = weight_bytes * ENERGY_CONSTANTS['systolic_weight_load_pj']
    running_total = print_step(
        step, "Load Weights into Array",
        weight_load_energy,
        f"Load 128x128 weight matrix into PEs ({weight_bytes/1024:.1f} KB)",
        running_total
    )
    step += 1

    print(f"         - Each PE receives 1 weight value")
    print(f"         - Weights loaded via systolic shift (row by row)")
    print(f"         - Weights remain stationary during compute")
    print()

    # Phase 3: Activation Streaming (128 cycles to fill + 128 cycles to drain)
    cycles_to_compute = tile_size + tile_size  # Fill + drain
    activation_bytes = tile_size * tile_size * 4
    stream_energy = activation_bytes * ENERGY_CONSTANTS['systolic_data_shift_pj']
    running_total = print_step(
        step, f"Stream Activations ({cycles_to_compute} cycles)",
        stream_energy,
        f"Stream 128x128 activation matrix through array ({activation_bytes/1024:.1f} KB)",
        running_total
    )
    step += 1

    # Phase 4: Compute (happens during streaming)
    compute_energy = macs_per_tile * ENERGY_CONSTANTS['fp32_mac_pj'] * 0.8  # Systolic is more efficient
    running_total = print_step(
        step, "Compute (16,384 MACs)",
        compute_energy,
        f"All 16,384 PEs compute in parallel as data flows through",
        running_total
    )
    step += 1

    print(f"         - Each PE: multiply local weight with incoming activation")
    print(f"         - Add to partial sum from above PE")
    print(f"         - Pass partial sum to PE below")
    print()

    # Phase 5: Drain results
    result_bytes = tile_size * tile_size * 4
    drain_energy = result_bytes * ENERGY_CONSTANTS['systolic_data_shift_pj']
    running_total = print_step(
        step, "Drain Results",
        drain_energy,
        f"Accumulated results flow out of array ({result_bytes/1024:.1f} KB)",
        running_total
    )
    step += 1

    # Calculate remaining tiles
    remaining_tiles = num_tiles - 1
    if remaining_tiles > 0:
        energy_per_tile = running_total
        remaining_energy = remaining_tiles * energy_per_tile
        print(f"  ... {remaining_tiles:,} more tile operations (same pattern) ...")
        print(f"      Additional energy: +{format_energy(remaining_energy)}")
        running_total += remaining_energy
        print()

    # Summary
    print_divider("=")
    print()
    print("  ENERGY SUMMARY (TPU Systolic):")
    print_divider("-")

    # Recalculate
    config_energy = num_tiles * ENERGY_CONSTANTS['domain_config_pj']
    weight_energy = num_tiles * tile_size * tile_size * 4 * ENERGY_CONSTANTS['systolic_weight_load_pj']
    stream_energy = num_tiles * tile_size * tile_size * 4 * ENERGY_CONSTANTS['systolic_data_shift_pj']
    compute_energy = num_macs * ENERGY_CONSTANTS['fp32_mac_pj'] * 0.8
    drain_energy = num_tiles * tile_size * tile_size * 4 * ENERGY_CONSTANTS['systolic_data_shift_pj']

    total_control = config_energy
    total_data_movement = weight_energy + stream_energy + drain_energy
    total_compute = compute_energy
    total_energy = total_control + total_data_movement + total_compute

    print(f"    Control (Config):     {format_energy(total_control):>12} ({total_control/total_energy*100:.1f}%)")
    print(f"    Data Movement:        {format_energy(total_data_movement):>12} ({total_data_movement/total_energy*100:.1f}%)")
    print(f"      - Weight Load:      {format_energy(weight_energy):>12}")
    print(f"      - Stream + Drain:   {format_energy(stream_energy + drain_energy):>12}")
    print(f"    Compute (MACs):       {format_energy(total_compute):>12} ({total_compute/total_energy*100:.1f}%)")
    print_divider("-")
    print(f"    TOTAL:                {format_energy(total_energy):>12}")
    print(f"    Energy per MAC:       {format_energy(total_energy/num_macs):>12}")
    print()

    print("  KEY INSIGHT: No per-operation instruction fetch/decode!")
    print("  - Configuration happens once per tile (16,384 MACs)")
    print("  - vs CPU: 1 instruction per 16 MACs")
    print("  - vs GPU: 1 warp instruction per 32 MACs")
    print()

    return total_energy


# =============================================================================
# KPU Walkthrough
# =============================================================================

def walkthrough_kpu(num_macs: int, matrix_dim: int = None):
    """
    Step-by-step walkthrough of KPU spatial dataflow execution.

    KPU uses domain-flow architecture:
    - 64 tiles of 16x16 = 16,384 MACs per cycle (same as TPU)
    - BUT: smaller tiles = better mapping to irregular sizes
    - Domain program configures entire layer at once
    """
    print()
    print("=" * 90)
    print("  KPU (Spatial Dataflow) STEP-BY-STEP WALKTHROUGH")
    print("=" * 90)
    print()

    if matrix_dim:
        print(f"  Workload: {matrix_dim}x{matrix_dim} matrix multiply = {num_macs:,} MACs")
    else:
        print(f"  Workload: {num_macs:,} MAC operations")
    print()

    print("  Architecture Overview:")
    print("  -----------------------")
    print("  - Spatial Dataflow with Domain Programming")
    print("  - 64 tiles, each 16x16 = 256 MACs")
    print("  - Total: 64 x 256 = 16,384 MACs per cycle (same as TPU)")
    print("  - BUT: Finer granularity (16x16 vs 128x128)")
    print("  - Domain Program: Configure entire layer in one instruction")
    print("  - EDDO: Explicit Data Distribution & Orchestration")
    print()

    # Calculate
    tile_size = 16
    num_tiles = 64
    macs_per_tile = tile_size * tile_size  # 256
    macs_per_cycle = num_tiles * macs_per_tile  # 16,384

    num_cycles = (num_macs + macs_per_cycle - 1) // macs_per_cycle

    print(f"  Execution Plan:")
    print(f"  - Total MACs needed: {num_macs:,}")
    print(f"  - MACs per tile (16x16): {macs_per_tile}")
    print(f"  - MACs per cycle (64 tiles): {macs_per_cycle:,}")
    print(f"  - Cycles needed: {num_cycles:,}")
    print()

    print("  DETAILED EXECUTION:")
    print_divider("=")
    print()

    running_total = 0.0
    step = 1

    # Phase 1: Domain Program Load (ONCE for entire operation)
    print("  --- Phase 1: Domain Program Configuration (ONE TIME) ---")
    print()

    running_total = print_step(
        step, "Load Domain Program",
        ENERGY_CONSTANTS['domain_config_pj'],
        "Configure all 64 tiles for matrix multiply dataflow",
        running_total
    )
    step += 1

    print(f"         - Domain program specifies data flow between tiles")
    print(f"         - Each tile knows its role in the computation")
    print(f"         - NO per-operation instructions after this point!")
    print()

    # Phase 2: DMA Setup
    data_bytes = 3 * tile_size * tile_size * 4 * num_tiles  # A, B, C for all tiles
    dma_energy = data_bytes * 0.01  # DMA setup is cheap
    running_total = print_step(
        step, "DMA Setup",
        dma_energy,
        f"Configure DMA to stream data to tiles ({data_bytes/1024:.1f} KB total)",
        running_total
    )
    step += 1

    print()
    print("  --- Phase 2: Execution (Data Flows Through Tiles) ---")
    print()

    # Show one cycle in detail
    print(f"  [Cycle 1 of {num_cycles}] All 64 tiles compute in parallel:")
    print()

    # Data streaming
    stream_bytes_per_cycle = macs_per_cycle * 8  # Rough: 2 inputs per MAC
    stream_energy = stream_bytes_per_cycle * ENERGY_CONSTANTS['stream_token_pj']
    running_total = print_step(
        step, "Stream Data Tokens",
        stream_energy,
        f"Data tokens flow through interconnect to tiles",
        running_total
    )
    step += 1

    # Compute (all tiles in parallel)
    compute_energy = macs_per_cycle * ENERGY_CONSTANTS['fp32_mac_pj'] * 0.6  # Very efficient
    running_total = print_step(
        step, f"Compute ({macs_per_cycle:,} MACs)",
        compute_energy,
        f"64 tiles x 256 MACs each = 16,384 parallel MACs",
        running_total
    )
    step += 1

    # Tile-local accumulation
    accum_energy = num_tiles * macs_per_tile * ENERGY_CONSTANTS['systolic_accumulator_pj']
    running_total = print_step(
        step, "Local Accumulation",
        accum_energy,
        f"Each tile accumulates 256 partial products",
        running_total
    )
    step += 1

    # Stream results
    result_energy = macs_per_cycle * 4 * ENERGY_CONSTANTS['stream_token_pj']
    running_total = print_step(
        step, "Stream Results",
        result_energy,
        f"Results flow to next tiles or output",
        running_total
    )
    step += 1

    # Remaining cycles
    remaining_cycles = num_cycles - 1
    if remaining_cycles > 0:
        cycle_energy = stream_energy + compute_energy + accum_energy + result_energy
        remaining_energy = remaining_cycles * cycle_energy
        print(f"  ... {remaining_cycles:,} more cycles (same pattern, NO new configuration) ...")
        print(f"      Additional energy: +{format_energy(remaining_energy)}")
        running_total += remaining_energy
        print()

    # Summary
    print_divider("=")
    print()
    print("  ENERGY SUMMARY (KPU Domain Flow):")
    print_divider("-")

    # Recalculate
    config_energy = ENERGY_CONSTANTS['domain_config_pj']  # Only once!
    stream_energy_total = num_cycles * stream_bytes_per_cycle * ENERGY_CONSTANTS['stream_token_pj']
    compute_energy_total = num_macs * ENERGY_CONSTANTS['fp32_mac_pj'] * 0.6
    accum_energy_total = num_cycles * num_tiles * macs_per_tile * ENERGY_CONSTANTS['systolic_accumulator_pj']
    result_energy_total = num_cycles * macs_per_cycle * 4 * ENERGY_CONSTANTS['stream_token_pj']

    total_control = config_energy
    total_data_movement = stream_energy_total + result_energy_total
    total_compute = compute_energy_total + accum_energy_total
    total_energy = total_control + total_data_movement + total_compute

    print(f"    Control (Config):     {format_energy(total_control):>12} ({total_control/total_energy*100:.1f}%)")
    print(f"    Data Movement:        {format_energy(total_data_movement):>12} ({total_data_movement/total_energy*100:.1f}%)")
    print(f"    Compute + Accum:      {format_energy(total_compute):>12} ({total_compute/total_energy*100:.1f}%)")
    print_divider("-")
    print(f"    TOTAL:                {format_energy(total_energy):>12}")
    print(f"    Energy per MAC:       {format_energy(total_energy/num_macs):>12}")
    print()

    print("  KEY INSIGHT: Domain program amortizes control over ENTIRE layer!")
    print("  - 1 configuration for potentially millions of MACs")
    print("  - vs TPU: 1 config per 16K MACs (tile)")
    print("  - vs CPU: 1 instruction per 16 MACs")
    print()
    print("  BONUS: 16x16 tiles map better to irregular sizes")
    print("  - 300x300 matrix: 95.5% utilization (vs TPU's 51%)")
    print()

    return total_energy


# =============================================================================
# Comparison Summary
# =============================================================================

def format_power(watts: float) -> str:
    """Format power with appropriate units."""
    if watts >= 1000:
        return f"{watts/1000:,.1f} kW"
    elif watts >= 1:
        return f"{watts:,.2f} W"
    elif watts >= 0.001:
        return f"{watts*1000:,.2f} mW"
    else:
        return f"{watts*1e6:,.2f} uW"


def print_comparison_summary(results: dict, num_macs: int):
    """Print side-by-side comparison with power metrics."""
    print()
    print("=" * 110)
    print("  ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 110)
    print()
    print(f"  Workload: {num_macs:,} MACs")
    print()

    # Map result keys to arch timing keys
    arch_key_map = {
        'CPU (AVX-512)': 'cpu',
        'GPU (CUDA)': 'gpu_cuda',
        'GPU (TensorCore)': 'gpu_tc',
        'TPU (Systolic)': 'tpu',
        'KPU (Domain Flow)': 'kpu',
    }

    # Calculate metrics for all architectures
    metrics = {}
    for arch_name, energy in results.items():
        arch_key = arch_key_map.get(arch_name)
        if arch_key:
            metrics[arch_name] = calculate_power_metrics(energy, num_macs, arch_key)

    # Part 1: Energy comparison
    print("  ENERGY COMPARISON:")
    print("  -------------------")
    print(f"  {'Architecture':<20} {'Energy/MAC':>12} {'vs Best':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10}")

    best_energy = min(results.values())
    for arch, energy in sorted(results.items(), key=lambda x: x[1]):
        per_mac = energy / num_macs
        vs_best = energy / best_energy
        print(f"  {arch:<20} {format_energy(per_mac):>12} {vs_best:>9.2f}x")

    # Part 2: Timing and throughput
    print()
    print("  TIMING & THROUGHPUT:")
    print("  ---------------------")
    print(f"  {'Architecture':<20} {'Frequency':>10} {'MACs/cyc':>10} {'Throughput':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")

    for arch_name in sorted(results.keys(), key=lambda x: results[x]):
        m = metrics.get(arch_name)
        if m:
            freq_str = f"{m['frequency_ghz']:.1f} GHz"
            mpc_str = f"{m['macs_per_cycle']:,}"
            tput_str = f"{m['throughput_tops']:.3f} TOPS"
            print(f"  {arch_name:<20} {freq_str:>10} {mpc_str:>10} {tput_str:>12}")

    # Part 3: Power analysis - THE KEY METRIC
    print()
    print("  POWER TO DELIVER 1 TOPS:")
    print("  --------------------------")
    print()
    print("  Formula: Power = Energy/MAC × Concurrency × Clock / 1000")
    print()
    print(f"  {'Architecture':<20} {'Energy/MAC':>10} {'Concurrency':>12} {'Clock':>10} {'Power@1TOPS':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")

    for arch_name in sorted(results.keys(), key=lambda x: metrics.get(x, {}).get('power_for_1tops_w', float('inf'))):
        m = metrics.get(arch_name)
        if m:
            e_str = f"{m['energy_per_mac_pj']:.2f} pJ"
            c_str = f"{m['macs_per_cycle']:,}"
            f_str = f"{m['frequency_ghz']:.1f} GHz"
            p_str = format_power(m['power_for_1tops_w'])
            print(f"  {arch_name:<20} {e_str:>10} {c_str:>12} {f_str:>10} {p_str:>12}")

    # Explanation
    print()
    print("  UNDERSTANDING 'POWER @ 1 TOPS':")
    print("  ---------------------------------")
    print("  This answers: 'How much power to deliver 1 TOPS of work?'")
    print()
    print("  Derivation:")
    print("    Work = 10^12 MACs")
    print("    Energy = 10^12 × E_mac(pJ) × 10^-12 = E_mac (Joules)")
    print("    Time = 10^12 / (Concurrency × Clock_Hz) seconds")
    print("    Power = Energy / Time")
    print("         = E_mac(pJ) × Concurrency × Clock(GHz) / 1000  (Watts)")
    print()
    print("  Lower is better - less power needed for same throughput.")
    print()

    # Context
    print("  POWER BUDGET CONTEXT:")
    print("  ----------------------")
    print("    Mobile/Edge:     1-10 W   -> Need < 0.1 W/TOPS for 100 TOPS")
    print("    Laptop:         15-45 W   -> Need < 0.5 W/TOPS for 100 TOPS")
    print("    Desktop GPU:   200-450 W  -> Can afford ~2-4 W/TOPS")
    print("    Datacenter:    300-700 W  -> Can afford ~3-7 W/TOPS")
    print()

    print("  WHAT MAKES THE DIFFERENCE:")
    print("  ---------------------------")
    print("  1. CONTROL OVERHEAD: Instructions per MAC")
    print("     - CPU: 1 instruction / 16 MACs (Von Neumann tax)")
    print("     - GPU: 1 warp instruction / 32 MACs (SIMT overhead)")
    print("     - TPU: 1 config / 16,384 MACs (systolic efficiency)")
    print("     - KPU: 1 domain program / millions of MACs (best amortization)")
    print()
    print("  2. COMPUTE EFFICIENCY: Energy per MAC operation")
    print("     - Specialized units (TensorCore, Systolic) are 2-5x more efficient")
    print("     - General-purpose units (CUDA cores, ALU) have more overhead")
    print()
    print("  3. DATA MOVEMENT: Registers vs Caches vs DRAM")
    print("     - Registers: ~0.1 pJ/byte")
    print("     - L1 cache:  ~0.5 pJ/byte")
    print("     - DRAM:      ~20 pJ/byte (200x more!)")
    print()
    print("  4. OPERAND FETCH: Register-to-ALU delivery (the key differentiator!)")
    print("     - Pure ALU energy: ~1.8 pJ per FP32 MAC (same for all architectures)")
    print("     - CPU operand fetch: ~3.4 pJ per op (register file reads/writes)")
    print("     - GPU operand fetch: ~2.7 pJ per op (operand collectors + crossbar)")
    print("     - TPU operand fetch: ~0.002 pJ per op (PE forwarding with 16K reuse)")
    print("     - KPU operand fetch: ~0.4 pJ per op (domain flow forwarding with 64x reuse)")
    print()
    print("     For CPU/GPU, operand fetch dominates (60-90% of operation energy)")
    print("     For TPU/KPU, ALU dominates (operand fetch is amortized over spatial reuse)")
    print()


# =============================================================================
# Operand Fetch Energy Analysis
# =============================================================================

def print_operand_fetch_comparison(num_macs: int):
    """
    Show operand fetch energy comparison across architectures.

    This is THE key insight: the ALU energy is nearly identical (~1.8 pJ)
    across all architectures at the same process node. What differs is
    the energy to DELIVER operands to the ALU.
    """
    print()
    print("=" * 110)
    print("  OPERAND FETCH ENERGY: The Key Architectural Differentiator")
    print("=" * 110)
    print()
    print("  The ALU circuit (ADD/MUL/FMA) consumes nearly identical energy across architectures")
    print("  at the same process node (~1.8 pJ for FP32 MAC at 8nm).")
    print()
    print("  What differs dramatically is the OPERAND FETCH infrastructure:")
    print("  - CPU/GPU: Every operation reads from register file (no spatial reuse)")
    print("  - TPU/KPU: Operands forwarded PE-to-PE (massive spatial reuse)")
    print()

    # Get technology profile for energy values
    comparison = ARCH_COMPARISON_8NM_X86
    tech_profile = comparison.cpu_profile

    # Create operand fetch models
    cpu_model = CPUOperandFetchModel(tech_profile=tech_profile)
    gpu_model = GPUOperandFetchModel(tech_profile=tech_profile)
    tpu_model = TPUOperandFetchModel(tech_profile=tech_profile)
    kpu_model = KPUOperandFetchModel(tech_profile=tech_profile)

    # Compute operand fetch energy for each architecture
    cpu_fetch = cpu_model.compute_operand_fetch_energy(num_macs)
    gpu_fetch = gpu_model.compute_operand_fetch_energy(num_macs)
    tpu_fetch = tpu_model.compute_operand_fetch_energy(num_macs)
    kpu_fetch = kpu_model.compute_operand_fetch_energy(num_macs)

    # Pure ALU energy (same for all architectures)
    pure_alu_energy_pj = 1.8  # pJ per FP32 MAC at 8nm
    total_alu_energy = num_macs * pure_alu_energy_pj * 1e-12  # J

    # Print comparison table
    print(f"  Workload: {num_macs:,} MAC operations")
    print(f"  Pure ALU energy: {total_alu_energy * 1e6:.3f} uJ ({pure_alu_energy_pj} pJ/op)")
    print()
    print(f"  {'Architecture':<20} {'Fetch Energy':>14} {'pJ/op':>10} {'Reuse':>10} {'ALU/Fetch':>12} {'Bottleneck':<15}")
    print(f"  {'-'*20} {'-'*14} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")

    for name, fetch in [('CPU (Stored Program)', cpu_fetch),
                        ('GPU (SIMT)', gpu_fetch),
                        ('TPU (Systolic)', tpu_fetch),
                        ('KPU (Domain Flow)', kpu_fetch)]:
        fetch_energy_j = fetch.total_fetch_energy
        fetch_per_op_pj = fetch.energy_per_operation * 1e12

        # Format fetch energy
        if fetch_energy_j >= 1e-6:
            fetch_str = f"{fetch_energy_j * 1e6:.3f} uJ"
        elif fetch_energy_j >= 1e-9:
            fetch_str = f"{fetch_energy_j * 1e9:.3f} nJ"
        else:
            fetch_str = f"{fetch_energy_j * 1e12:.3f} pJ"

        reuse = fetch.operand_reuse_factor
        reuse_str = f"{reuse:.1f}x"

        # ALU/Fetch ratio
        if fetch_energy_j > 0:
            alu_fetch_ratio = total_alu_energy / fetch_energy_j
        else:
            alu_fetch_ratio = float('inf')

        # Determine bottleneck
        if alu_fetch_ratio < 1.0:
            bottleneck = "Fetch-dominated"
        else:
            bottleneck = "ALU-dominated"

        print(f"  {name:<20} {fetch_str:>14} {fetch_per_op_pj:>9.2f} {reuse_str:>10} {alu_fetch_ratio:>11.2f} {bottleneck:<15}")

    print()
    print("  KEY INSIGHT:")
    print("  -----------")
    print("    - ALU/Fetch < 1.0: Operand fetch energy EXCEEDS ALU energy (inefficient)")
    print("    - ALU/Fetch > 1.0: ALU energy exceeds operand fetch (efficient)")
    print()
    print("    CPU/GPU are FETCH-DOMINATED: Most energy goes to moving data to/from registers")
    print("    TPU/KPU are ALU-DOMINATED: Most energy goes to actual computation")
    print()
    print("    This is why spatial architectures achieve 10-100x better TOPS/W!")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step-by-Step Energy Walkthrough",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--arch', type=str,
                        choices=['cpu', 'gpu', 'gpu-tc', 'tpu', 'kpu', 'all'],
                        default='all',
                        help='Architecture to walk through (default: all)')
    parser.add_argument('--size', type=int, default=4,
                        help='Matrix dimension for NxN matmul (default: 4)')
    parser.add_argument('--ops', type=int, default=None,
                        help='Explicit number of MACs (overrides --size)')
    parser.add_argument('--operand-fetch', action='store_true',
                        help='Show operand fetch energy comparison (ALU vs operand delivery)')

    args = parser.parse_args()

    # Calculate number of MACs
    if args.ops:
        num_macs = args.ops
        matrix_dim = None
    else:
        matrix_dim = args.size
        num_macs = matrix_dim * matrix_dim * matrix_dim

    print()
    print("=" * 90)
    print("  ENERGY STEP-BY-STEP: Follow Along Educational Walkthrough")
    print("=" * 90)
    print()
    print("  This walkthrough shows EXACTLY where energy is spent in each architecture.")
    print("  Follow along to understand the fundamental differences in execution models.")
    print()

    results = {}

    if args.arch in ['cpu', 'all']:
        results['CPU (AVX-512)'] = walkthrough_cpu(num_macs, matrix_dim)

    if args.arch in ['gpu', 'all']:
        results['GPU (CUDA)'] = walkthrough_gpu_cuda(num_macs, matrix_dim)

    if args.arch in ['gpu-tc', 'all']:
        results['GPU (TensorCore)'] = walkthrough_gpu_tensorcore(num_macs, matrix_dim)

    if args.arch in ['tpu', 'all']:
        results['TPU (Systolic)'] = walkthrough_tpu(num_macs, matrix_dim)

    if args.arch in ['kpu', 'all']:
        results['KPU (Domain Flow)'] = walkthrough_kpu(num_macs, matrix_dim)

    if len(results) > 1:
        print_comparison_summary(results, num_macs)

    # Show operand fetch comparison if requested or always for 'all' architecture
    if args.operand_fetch or args.arch == 'all':
        print_operand_fetch_comparison(num_macs)

    print("=" * 90)
    print("  END OF WALKTHROUGH")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
