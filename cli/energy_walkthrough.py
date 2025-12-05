#!/usr/bin/env python
"""
Energy Walkthrough: Step-by-Step Execution Comparison

This script demonstrates architectural energy differences through TWO analyses:

PART 1: NATIVE OPERATION ENERGY
  Compare each architecture at its natural execution granularity:
  - CPU:          1 AVX-512 instruction  ->  16 MACs
  - GPU (CUDA):   4 warp instructions    -> 128 MACs (4 partitions x 32 cores)
  - GPU (TC):     4 MMA instructions     -> 256 MACs (4 partitions x 64 MACs)
  - TPU:          1 systolic tile cycle  -> 16,384 MACs (128x128)
  - KPU:          1 tile cycle           -> 256 MACs (16x16, 64 tiles = 16,384 total)

  This answers: "What's the fundamental energy cost per native operation?"

PART 2: REAL WORKLOAD MAPPING
  Map the same workload (e.g., 300x300 MatMul = 27M MACs) to each architecture:
  - Shows tiling/partitioning overhead
  - Shows utilization efficiency (300 doesn't divide evenly into 128 or 16)
  - Shows partial tile waste

  This answers: "How efficiently can each architecture execute real workloads?"

Key insights:
  - GPU TensorCore (256 MACs/SM) matches KPU tile (256 MACs) - direct comparison!
  - TPU and KPU have same total MACs/cycle (16,384), but different granularity
  - GPU SM has 4 partitions because a single warp scheduler for 128 cores is infeasible
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphs.hardware.cycle_energy import (
    build_cpu_cycle_energy,
    build_gpu_cuda_cycle_energy,
    build_gpu_tensorcore_cycle_energy,
    build_tpu_cycle_energy,
    build_kpu_cycle_energy,
)
from graphs.hardware.cycle_energy.base import OperatingMode, OperatorType
from graphs.hardware.technology_profile import ARCH_COMPARISON_8NM_X86
from graphs.hardware.operand_fetch import (
    CPUOperandFetchModel,
    GPUOperandFetchModel,
    TPUOperandFetchModel,
    KPUOperandFetchModel,
)


def format_energy(pj: float) -> str:
    """Format energy in appropriate units."""
    if pj >= 1_000_000:
        return f"{pj/1_000_000:.2f} uJ"
    elif pj >= 1_000:
        return f"{pj/1_000:.2f} nJ"
    else:
        return f"{pj:.2f} pJ"


def format_energy_fixed(pj: float, width: int = 12) -> str:
    """Format energy with fixed width for alignment."""
    if pj == 0:
        return "-".center(width)
    elif pj >= 1_000_000:
        return f"{pj/1_000_000:>{width-3}.2f} uJ"
    elif pj >= 1_000:
        return f"{pj/1_000:>{width-3}.2f} nJ"
    elif pj >= 0.1:
        return f"{pj:>{width-3}.2f} pJ"
    else:
        return f"{pj*1000:>{width-3}.2f} fJ"


def print_header(title: str, width: int = 140):
    """Print a section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step_row(step: str, cpu_pj: float, gpu_cuda_pj: float, gpu_tc_pj: float,
                   tpu_pj: float, kpu_pj: float, col_width: int = 15):
    """Print a row comparing energy across architectures."""
    cpu_str = format_energy_fixed(cpu_pj, col_width-2) if cpu_pj > 0 else "-".center(col_width-2)
    gpu_cuda_str = format_energy_fixed(gpu_cuda_pj, col_width-2) if gpu_cuda_pj > 0 else "-".center(col_width-2)
    gpu_tc_str = format_energy_fixed(gpu_tc_pj, col_width-2) if gpu_tc_pj > 0 else "-".center(col_width-2)
    tpu_str = format_energy_fixed(tpu_pj, col_width-2) if tpu_pj > 0 else "-".center(col_width-2)
    kpu_str = format_energy_fixed(kpu_pj, col_width-2) if kpu_pj > 0 else "-".center(col_width-2)

    print(f"  {step:<35} {cpu_str:>{col_width}} {gpu_cuda_str:>{col_width}} {gpu_tc_str:>{col_width}} {tpu_str:>{col_width}} {kpu_str:>{col_width}}")


# =============================================================================
# PART 1: NATIVE OPERATION ENERGY
# =============================================================================

def part1_native_operation_energy(comparison, verbose: bool = False):
    """
    Part 1: Compare each architecture at its NATIVE execution granularity.

    This is an apples-to-apples circuit comparison asking:
    "What's the fundamental energy cost of ONE native operation?"

    Native operations:
      CPU:          1 AVX-512 instruction  = 16 FP32 MACs
      GPU (CUDA):   4 warp instructions    = 128 MACs (4 partitions x 32 cores)
      GPU (TC):     4 MMA instructions     = 256 MACs (4 partitions x 64 MACs)
      TPU:          1 systolic tile cycle  = 16,384 MACs (128x128 array)
      KPU:          1 tile cycle           = 256 MACs (16x16 tile)

    Key comparison: GPU TensorCore (256 MACs) = KPU tile (256 MACs)
    """
    print_header("PART 1: NATIVE OPERATION ENERGY (Apples-to-Apples Circuit Comparison)")
    print()
    print("  Each architecture has a NATIVE execution unit that amortizes control overhead:")
    print()
    print("    Architecture      Native Unit                   MACs/Unit    Notes")
    print("    ----------------  ----------------------------  ------------ ----------------------------------")
    print("    CPU (AVX-512)     1 vector instruction          16           FP32, 512-bit registers")
    print("    GPU (CUDA)        4 warp instructions (1 SM)    128          4 partitions x 32 CUDA cores")
    print("    GPU (TensorCore)  4 MMA instructions (1 SM)     256          4 partitions x 64 MACs (4x4x4)")
    print("    TPU (Systolic)    1 tile cycle (128x128)        16,384       Weight-stationary dataflow")
    print("    KPU (Domain)      1 tile cycle (16x16)          256          Per-tile; 64 tiles total")
    print()
    print("  Key insight: GPU TensorCore (256) = KPU tile (256) - direct comparison!")
    print()

    # Define native unit sizes
    cpu_ops = 16         # AVX-512: 16 FP32 MACs per vector instruction
    gpu_cuda_ops = 128   # 4 partitions x 32 CUDA cores
    gpu_tc_ops = 256     # 4 partitions x 64 MACs (TensorCores)
    tpu_ops = 16384      # 128x128 systolic array
    kpu_ops = 256        # Single 16x16 tile

    # Minimal bytes - just enough for the native operation (L1 resident)
    cpu_bytes = cpu_ops * 8
    gpu_cuda_bytes = gpu_cuda_ops * 8
    gpu_tc_bytes = gpu_tc_ops * 8
    tpu_bytes = tpu_ops * 8
    kpu_bytes = kpu_ops * 8

    # Build breakdowns - L1 resident to focus on compute, not memory
    cpu = build_cpu_cycle_energy(
        num_ops=cpu_ops, bytes_transferred=cpu_bytes,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.cpu_profile,
        simd_width=16
    )
    gpu_cuda = build_gpu_cuda_cycle_energy(
        num_ops=gpu_cuda_ops, bytes_transferred=gpu_cuda_bytes,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.gpu_profile,
    )
    gpu_tc = build_gpu_tensorcore_cycle_energy(
        num_ops=gpu_tc_ops, bytes_transferred=gpu_tc_bytes,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.gpu_profile,
    )
    tpu = build_tpu_cycle_energy(
        num_ops=tpu_ops, bytes_transferred=tpu_bytes,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.tpu_profile,
    )
    kpu = build_kpu_cycle_energy(
        num_ops=kpu_ops, bytes_transferred=kpu_bytes,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.kpu_profile,
        num_layers=1,
    )

    col_w = 15
    print(f"  {'METRIC':<35} {'CPU (16)':>{col_w}} {'GPU-CUDA(128)':>{col_w}} {'GPU-TC (256)':>{col_w}} {'TPU (16K)':>{col_w}} {'KPU (256)':>{col_w}}")
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    # Energy breakdown by category
    cpu_ctrl = cpu.get_control_overhead_energy()
    gpu_cuda_ctrl = gpu_cuda.get_control_overhead_energy()
    gpu_tc_ctrl = gpu_tc.get_control_overhead_energy()
    tpu_ctrl = tpu.get_control_overhead_energy()
    kpu_ctrl = kpu.get_control_overhead_energy()

    cpu_compute = cpu.get_compute_energy()
    gpu_cuda_compute = gpu_cuda.get_compute_energy()
    gpu_tc_compute = gpu_tc.get_compute_energy()
    tpu_compute = tpu.get_compute_energy()
    kpu_compute = kpu.get_compute_energy()

    cpu_data = cpu.get_data_movement_energy()
    gpu_cuda_data = gpu_cuda.get_data_movement_energy()
    gpu_tc_data = gpu_tc.get_data_movement_energy()
    tpu_data = tpu.get_data_movement_energy()
    kpu_data = kpu.get_data_movement_energy()

    print_step_row("Control Overhead", cpu_ctrl, gpu_cuda_ctrl, gpu_tc_ctrl, tpu_ctrl, kpu_ctrl)
    print_step_row("Compute (MACs)", cpu_compute, gpu_cuda_compute, gpu_tc_compute, tpu_compute, kpu_compute)
    print_step_row("Data Movement (L1)", cpu_data, gpu_cuda_data, gpu_tc_data, tpu_data, kpu_data)
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("TOTAL (native unit)", cpu.total_energy_pj, gpu_cuda.total_energy_pj,
                   gpu_tc.total_energy_pj, tpu.total_energy_pj, kpu.total_energy_pj)

    # Per-MAC energy (the key metric!)
    print()
    print(f"  {'ENERGY PER MAC (pJ)':<35} {'CPU':>{col_w}} {'GPU-CUDA':>{col_w}} {'GPU-TC':>{col_w}} {'TPU':>{col_w}} {'KPU':>{col_w}}")
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("Total / num_MACs",
                   cpu.total_energy_pj / cpu_ops,
                   gpu_cuda.total_energy_pj / gpu_cuda_ops,
                   gpu_tc.total_energy_pj / gpu_tc_ops,
                   tpu.total_energy_pj / tpu_ops,
                   kpu.total_energy_pj / kpu_ops)
    print_step_row("Control / num_MACs",
                   cpu_ctrl / cpu_ops,
                   gpu_cuda_ctrl / gpu_cuda_ops,
                   gpu_tc_ctrl / gpu_tc_ops,
                   tpu_ctrl / tpu_ops,
                   kpu_ctrl / kpu_ops)
    print_step_row("Compute / num_MACs",
                   cpu_compute / cpu_ops,
                   gpu_cuda_compute / gpu_cuda_ops,
                   gpu_tc_compute / gpu_tc_ops,
                   tpu_compute / tpu_ops,
                   kpu_compute / kpu_ops)

    # Show control overhead percentage
    print()
    print("  CONTROL OVERHEAD ANALYSIS:")
    print(f"    CPU:        {cpu_ctrl/cpu.total_energy_pj*100:5.1f}% (1 fetch/decode per 16 MACs)")
    print(f"    GPU-CUDA:   {gpu_cuda_ctrl/gpu_cuda.total_energy_pj*100:5.1f}% (4 warp schedulers per 128 MACs)")
    print(f"    GPU-TC:     {gpu_tc_ctrl/gpu_tc.total_energy_pj*100:5.1f}% (4 warp schedulers per 256 MACs)")
    print(f"    TPU:        {tpu_ctrl/tpu.total_energy_pj*100:5.1f}% (1 tile config per 16K MACs)")
    print(f"    KPU:        {kpu_ctrl/kpu.total_energy_pj*100:5.1f}% (1 domain tracker per 256 MACs)")
    print()
    print("  KEY OBSERVATIONS:")
    print("    1. GPU-TC and KPU have SAME native unit size (256 MACs) - direct comparison!")
    print("    2. GPU needs 4 warp schedulers; KPU needs 1 domain tracker")
    print("    3. TPU amortizes control over 16K MACs but has poor mapping for irregular sizes")


# =============================================================================
# PART 2: REAL WORKLOAD MAPPING
# =============================================================================

def calculate_tiling_efficiency(matrix_dim: int, tile_dim: int) -> dict:
    """Calculate tiling efficiency for a matrix operation."""
    full_tiles = matrix_dim // tile_dim
    remainder = matrix_dim % tile_dim

    total_tiles_needed = full_tiles * full_tiles
    partial_tiles = 0
    if remainder > 0:
        partial_tiles = 2 * full_tiles + 1

    if remainder > 0:
        partial_utilization = (remainder * remainder) / (tile_dim * tile_dim)
    else:
        partial_utilization = 1.0

    total_tiles = total_tiles_needed + partial_tiles
    if total_tiles > 0:
        full_tile_ops = total_tiles_needed * (tile_dim * tile_dim)
        partial_tile_ops = partial_tiles * (remainder * remainder if remainder > 0 else 0)
        actual_ops = full_tile_ops + partial_tile_ops
        potential_ops = total_tiles * (tile_dim * tile_dim)
        avg_utilization = actual_ops / potential_ops if potential_ops > 0 else 0
    else:
        avg_utilization = 1.0

    return {
        'full_tiles': total_tiles_needed,
        'partial_tiles': partial_tiles,
        'total_tiles': total_tiles,
        'partial_utilization': partial_utilization,
        'avg_utilization': avg_utilization,
        'remainder': remainder,
    }


def part2_real_workload_mapping(comparison, matrix_dim: int = 300, verbose: bool = False):
    """
    Part 2: Map a real workload to each architecture.

    This shows how each architecture handles a REAL workload:
    - Matrix multiply: C = A x B where A, B are (matrix_dim x matrix_dim)
    - Total MACs = matrix_dim^3 (for square matrix multiply)
    """
    total_macs = matrix_dim * matrix_dim * matrix_dim
    bytes_transferred = 3 * matrix_dim * matrix_dim * 4

    print_header(f"PART 2: REAL WORKLOAD MAPPING ({matrix_dim}x{matrix_dim} MatMul = {total_macs/1e6:.1f}M MACs)")
    print()
    print(f"  Workload: C = A x B where A, B are {matrix_dim}x{matrix_dim} FP32 matrices")
    print(f"  Total MACs: {total_macs:,}")
    print(f"  Data Size: {bytes_transferred/1024/1024:.1f} MB (3 matrices x 4 bytes)")
    print()

    # Calculate tiling efficiency
    tpu_tile = 128
    kpu_tile = 16

    tpu_eff = calculate_tiling_efficiency(matrix_dim, tpu_tile)
    kpu_eff = calculate_tiling_efficiency(matrix_dim, kpu_tile)

    print("  TILING ANALYSIS:")
    print("  ----------------")
    print(f"    TPU (128x128 tiles):")
    print(f"      {matrix_dim} = {matrix_dim // tpu_tile} x {tpu_tile} + {matrix_dim % tpu_tile} remainder")
    print(f"      Full tiles: {tpu_eff['full_tiles']}, Partial tiles: {tpu_eff['partial_tiles']}")
    print(f"      Average utilization: {tpu_eff['avg_utilization']*100:.1f}%")
    print()
    print(f"    KPU (16x16 tiles):")
    print(f"      {matrix_dim} = {matrix_dim // kpu_tile} x {kpu_tile} + {matrix_dim % kpu_tile} remainder")
    print(f"      Full tiles: {kpu_eff['full_tiles']}, Partial tiles: {kpu_eff['partial_tiles']}")
    print(f"      Average utilization: {kpu_eff['avg_utilization']*100:.1f}%")
    print()

    # Build energy breakdowns
    cpu = build_cpu_cycle_energy(
        num_ops=total_macs, bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.cpu_profile,
        simd_width=16,
        operator_type=OperatorType.HIGH_REUSE,
    )
    gpu_cuda = build_gpu_cuda_cycle_energy(
        num_ops=total_macs, bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.gpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
    )
    gpu_tc = build_gpu_tensorcore_cycle_energy(
        num_ops=total_macs, bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.gpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
    )
    tpu = build_tpu_cycle_energy(
        num_ops=total_macs, bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.tpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
    )
    kpu = build_kpu_cycle_energy(
        num_ops=total_macs, bytes_transferred=bytes_transferred,
        mode=OperatingMode.DRAM_RESIDENT,
        tech_profile=comparison.kpu_profile,
        operator_type=OperatorType.HIGH_REUSE,
        num_layers=1,
    )

    col_w = 15
    print(f"  {'EXECUTION METRICS':<35} {'CPU (AVX512)':>{col_w}} {'GPU (CUDA)':>{col_w}} {'GPU (TC)':>{col_w}} {'TPU (128x128)':>{col_w}} {'KPU (16x16)':>{col_w}}")
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    # Show execution parameters
    cpu_instrs = total_macs // 16
    gpu_cuda_cycles = (total_macs + 127) // 128
    gpu_tc_cycles = (total_macs + 255) // 256
    tpu_cycles = (total_macs + 16383) // 16384
    kpu_cycles = (total_macs + 16383) // 16384  # 64 tiles x 256 = 16384

    print(f"  {'Total MACs':<35} {total_macs:>{col_w},} {total_macs:>{col_w},} {total_macs:>{col_w},} {total_macs:>{col_w},} {total_macs:>{col_w},}")
    print(f"  {'Instructions/Cycles':<35} {cpu_instrs:>{col_w},} {gpu_cuda_cycles:>{col_w},} {gpu_tc_cycles:>{col_w},} {tpu_cycles:>{col_w},} {kpu_cycles:>{col_w},}")
    print(f"  {'MACs per cycle':<35} {16:>{col_w}} {128:>{col_w}} {256:>{col_w}} {16384:>{col_w}} {16384:>{col_w}}")
    print()

    # Energy breakdown
    print(f"  {'ENERGY BREAKDOWN':<35} {'CPU':>{col_w}} {'GPU-CUDA':>{col_w}} {'GPU-TC':>{col_w}} {'TPU':>{col_w}} {'KPU':>{col_w}}")
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    cpu_ctrl = cpu.get_control_overhead_energy()
    gpu_cuda_ctrl = gpu_cuda.get_control_overhead_energy()
    gpu_tc_ctrl = gpu_tc.get_control_overhead_energy()
    tpu_ctrl = tpu.get_control_overhead_energy()
    kpu_ctrl = kpu.get_control_overhead_energy()

    cpu_compute = cpu.get_compute_energy()
    gpu_cuda_compute = gpu_cuda.get_compute_energy()
    gpu_tc_compute = gpu_tc.get_compute_energy()
    tpu_compute = tpu.get_compute_energy()
    kpu_compute = kpu.get_compute_energy()

    cpu_data = cpu.get_data_movement_energy()
    gpu_cuda_data = gpu_cuda.get_data_movement_energy()
    gpu_tc_data = gpu_tc.get_data_movement_energy()
    tpu_data = tpu.get_data_movement_energy()
    kpu_data = kpu.get_data_movement_energy()

    print_step_row("Control Overhead", cpu_ctrl, gpu_cuda_ctrl, gpu_tc_ctrl, tpu_ctrl, kpu_ctrl)
    print_step_row("Compute (MACs)", cpu_compute, gpu_cuda_compute, gpu_tc_compute, tpu_compute, kpu_compute)
    print_step_row("Data Movement", cpu_data, gpu_cuda_data, gpu_tc_data, tpu_data, kpu_data)
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("TOTAL ENERGY", cpu.total_energy_pj, gpu_cuda.total_energy_pj,
                   gpu_tc.total_energy_pj, tpu.total_energy_pj, kpu.total_energy_pj)

    # Per-MAC efficiency
    print()
    print(f"  {'EFFICIENCY METRICS':<35} {'CPU':>{col_w}} {'GPU-CUDA':>{col_w}} {'GPU-TC':>{col_w}} {'TPU':>{col_w}} {'KPU':>{col_w}}")
    print(f"  {'-'*35} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("Energy per MAC (pJ)",
                   cpu.total_energy_pj / total_macs,
                   gpu_cuda.total_energy_pj / total_macs,
                   gpu_tc.total_energy_pj / total_macs,
                   tpu.total_energy_pj / total_macs,
                   kpu.total_energy_pj / total_macs)

    best_total = min(cpu.total_energy_pj, gpu_cuda.total_energy_pj, gpu_tc.total_energy_pj,
                     tpu.total_energy_pj, kpu.total_energy_pj)
    print(f"  {'vs Best':<35} {cpu.total_energy_pj/best_total:>{col_w}.2f}x {gpu_cuda.total_energy_pj/best_total:>{col_w}.2f}x {gpu_tc.total_energy_pj/best_total:>{col_w}.2f}x {tpu.total_energy_pj/best_total:>{col_w}.2f}x {kpu.total_energy_pj/best_total:>{col_w}.2f}x")
    print(f"  {'Control %':<35} {cpu_ctrl/cpu.total_energy_pj*100:>{col_w-1}.1f}% {gpu_cuda_ctrl/gpu_cuda.total_energy_pj*100:>{col_w-1}.1f}% {gpu_tc_ctrl/gpu_tc.total_energy_pj*100:>{col_w-1}.1f}% {tpu_ctrl/tpu.total_energy_pj*100:>{col_w-1}.1f}% {kpu_ctrl/kpu.total_energy_pj*100:>{col_w-1}.1f}%")
    print(f"  {'Data Movement %':<35} {cpu_data/cpu.total_energy_pj*100:>{col_w-1}.1f}% {gpu_cuda_data/gpu_cuda.total_energy_pj*100:>{col_w-1}.1f}% {gpu_tc_data/gpu_tc.total_energy_pj*100:>{col_w-1}.1f}% {tpu_data/tpu.total_energy_pj*100:>{col_w-1}.1f}% {kpu_data/kpu.total_energy_pj*100:>{col_w-1}.1f}%")

    print()
    print("  KEY OBSERVATIONS:")
    print(f"    1. Data movement dominates for DRAM-resident workloads ({bytes_transferred/1024/1024:.1f} MB)")
    print(f"    2. GPU-TC has 2x throughput vs GPU-CUDA (256 vs 128 MACs/cycle)")
    print(f"    3. TPU/KPU have same total throughput (16,384 MACs/cycle)")
    print(f"    4. KPU's finer tiles ({kpu_tile}x{kpu_tile}) map better to {matrix_dim}x{matrix_dim}")
    print(f"       - TPU waste: {(1-tpu_eff['avg_utilization'])*100:.1f}% of tiles underutilized")
    print(f"       - KPU waste: {(1-kpu_eff['avg_utilization'])*100:.1f}% of tiles underutilized")


# =============================================================================
# DETAILED EXECUTION TRACE
# =============================================================================

def execution_trace(comparison, num_ops: int = 10000, verbose: bool = False):
    """Show detailed execution trace for each architecture."""
    print_header(f"DETAILED EXECUTION TRACE: {num_ops:,} operations")
    print()
    print("  This shows every energy-consuming event in each architecture.")
    print()

    bytes_transferred = num_ops * 4

    cpu = build_cpu_cycle_energy(
        num_ops=num_ops, bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.cpu_profile,
        simd_width=16,
    )
    gpu_cuda = build_gpu_cuda_cycle_energy(
        num_ops=num_ops, bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.gpu_profile,
    )
    gpu_tc = build_gpu_tensorcore_cycle_energy(
        num_ops=num_ops, bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.gpu_profile,
    )
    tpu = build_tpu_cycle_energy(
        num_ops=num_ops, bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.tpu_profile,
    )
    kpu = build_kpu_cycle_energy(
        num_ops=num_ops, bytes_transferred=bytes_transferred,
        mode=OperatingMode.L2_RESIDENT,
        tech_profile=comparison.kpu_profile,
    )

    def print_trace(name: str, breakdown):
        print(f"\n  {name}")
        print(f"  {'-'*100}")

        total = 0
        for event in breakdown.events:
            event_total = event.energy_pj * event.count
            total += event_total
            desc = event.description[:55] if len(event.description) > 55 else event.description
            print(f"    {desc:<55} {event.count:>10,} x {event.energy_pj:>8.2f} pJ = {format_energy_fixed(event_total, 12)}")

        print(f"  {'-'*100}")
        print(f"    {'TOTAL':<55} {'':>10} {'':>12}   {format_energy_fixed(total, 12)}")
        print(f"    {'Per operation':<55} {'':>10} {'':>12}   {format_energy_fixed(total/num_ops, 12)}")

    print_trace("CPU (AVX-512)", cpu)
    print_trace("GPU CUDA (4 partitions x 32 cores = 128 MACs/SM)", gpu_cuda)
    print_trace("GPU TensorCore (4 partitions x 64 MACs = 256 MACs/SM)", gpu_tc)
    print_trace("TPU (Systolic 128x128)", tpu)
    print_trace("KPU (Domain Flow 64x16x16)", kpu)


# =============================================================================
# PART 3: OPERAND FETCH ENERGY ANALYSIS
# =============================================================================

def part3_operand_fetch_analysis(comparison, num_ops: int = 10000, verbose: bool = False):
    """
    Part 3: Operand Fetch Energy - The Key Architectural Differentiator.

    This section separates ALU energy from operand fetch energy to show
    WHY spatial architectures are fundamentally more efficient.

    Key insight: Pure ALU energy (~1.8 pJ for FP32 MAC) is nearly identical
    across all architectures. What differs is operand fetch:
    - CPU/GPU: Every operation reads from register file (no spatial reuse)
    - TPU/KPU: Operands forwarded PE-to-PE (massive spatial reuse)
    """
    print_header("PART 3: OPERAND FETCH ENERGY (ALU vs Register-to-ALU Delivery)")
    print()
    print("  The ALU circuit (ADD/MUL/FMA) has nearly identical energy across architectures")
    print("  at the same process node. What differs dramatically is OPERAND FETCH:")
    print()
    print("    - CPU/GPU: Every operation independently fetches operands from register file")
    print("    - TPU/KPU: Operands forwarded PE-to-PE with massive spatial reuse")
    print()
    print("  This is the fundamental reason spatial architectures achieve 10-100x better TOPS/W")
    print()

    # Get technology profile
    tech_profile = comparison.cpu_profile

    # Create operand fetch models
    cpu_model = CPUOperandFetchModel(tech_profile=tech_profile)
    gpu_model = GPUOperandFetchModel(tech_profile=tech_profile)
    tpu_model = TPUOperandFetchModel(tech_profile=tech_profile)
    kpu_model = KPUOperandFetchModel(tech_profile=tech_profile)

    # Compute operand fetch energy
    cpu_fetch = cpu_model.compute_operand_fetch_energy(num_ops)
    gpu_fetch = gpu_model.compute_operand_fetch_energy(num_ops)
    tpu_fetch = tpu_model.compute_operand_fetch_energy(num_ops)
    kpu_fetch = kpu_model.compute_operand_fetch_energy(num_ops)

    # Pure ALU energy (same for all architectures at same process node)
    pure_alu_energy_pj = 1.8  # pJ per FP32 MAC at 8nm
    total_alu_energy = num_ops * pure_alu_energy_pj * 1e-12  # Convert to J

    print(f"  Workload: {num_ops:,} MAC operations")
    print(f"  Pure ALU Energy: {total_alu_energy * 1e6:.3f} uJ ({pure_alu_energy_pj} pJ/op - same for all)")
    print()

    # Detailed comparison table
    col_w = 12
    print(f"  {'ARCHITECTURE':<18} {'ALU (pJ)':>{col_w}} {'Fetch (pJ)':>{col_w}} {'Total (pJ)':>{col_w}} {'Reuse':>{col_w}} {'ALU/Fetch':>{col_w}} {'Bottleneck':>{col_w}}")
    print(f"  {'-'*18} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    for name, fetch in [('CPU (Stored)', cpu_fetch),
                        ('GPU (SIMT)', gpu_fetch),
                        ('TPU (Systolic)', tpu_fetch),
                        ('KPU (Domain)', kpu_fetch)]:
        fetch_per_op_pj = fetch.energy_per_operation * 1e12
        total_per_op_pj = pure_alu_energy_pj + fetch_per_op_pj
        reuse = fetch.operand_reuse_factor

        # ALU/Fetch ratio
        if fetch_per_op_pj > 0:
            alu_fetch_ratio = pure_alu_energy_pj / fetch_per_op_pj
        else:
            alu_fetch_ratio = float('inf')

        # Determine bottleneck
        bottleneck = "Fetch" if alu_fetch_ratio < 1.0 else "ALU"

        print(f"  {name:<18} {pure_alu_energy_pj:>{col_w}.2f} {fetch_per_op_pj:>{col_w}.2f} {total_per_op_pj:>{col_w}.2f} {reuse:>{col_w-1}.1f}x {alu_fetch_ratio:>{col_w}.2f} {bottleneck:>{col_w}}")

    print()
    print("  KEY OBSERVATION:")
    print("  -----------------")
    print("    ALU/Fetch < 1.0: Fetch-dominated (CPU/GPU) - energy goes to moving data")
    print("    ALU/Fetch > 1.0: ALU-dominated (TPU/KPU) - energy goes to computation")
    print()
    print("  CPU/GPU OVERHEAD BREAKDOWN:")
    print("    - Register file read:    2x operands per op")
    print("    - Register file write:   1x result per op")
    print("    - GPU adds: operand collectors + crossbar + bank conflicts")
    print()
    print("  TPU/KPU EFFICIENCY:")
    print("    - Weights loaded ONCE, reused across entire input batch")
    print("    - Inputs/outputs forwarded PE-to-PE (0.1 pJ short wire)")
    print("    - Reuse factor 64-16,000x reduces fetch energy proportionally")
    print()


# =============================================================================
# INSIGHTS
# =============================================================================

def print_insights(comparison):
    """Print key insights about architectural energy differences."""
    print_header("KEY INSIGHTS: Understanding Architectural Energy Differences")
    print()
    print("  1. SM PARTITION ARCHITECTURE (Why GPU has 4 partitions)")
    print("     " + "-"*90)
    print("     A single warp scheduler for 128 CUDA cores would be infeasible.")
    print("     Solution: Divide SM into 4 independent partitions, each with:")
    print("       - 1 warp scheduler")
    print("       - 32 CUDA cores OR 1 TensorCore")
    print("       - 1/4 of register file")
    print()
    print("     Full SM utilization requires 4 active warps (one per partition).")
    print()

    print("  2. NATIVE EXECUTION UNIT COMPARISON")
    print("     " + "-"*90)
    print("     Architecture      Native Unit                 MACs    Control Units")
    print("     ----------------  --------------------------  ------  -----------------")
    print("     CPU (AVX-512)     1 instruction               16      1 fetch/decode")
    print("     GPU (CUDA)        4 warp instructions (SM)    128     4 warp schedulers")
    print("     GPU (TensorCore)  4 MMA instructions (SM)     256     4 warp schedulers")
    print("     TPU (Systolic)    1 tile cycle                16,384  1 tile config")
    print("     KPU (Domain)      1 tile cycle (16x16)        256     1 domain tracker")
    print()
    print("     GPU-TC (256) = KPU tile (256) -> Direct comparison!")
    print("     GPU needs 4x control units; KPU needs 1 domain tracker")
    print()

    print("  3. CONTROL OVERHEAD AMORTIZATION")
    print("     " + "-"*90)
    print("     Architecture      MACs per control decision")
    print("     ----------------  -------------------------")
    print("     CPU (AVX-512)     16  (1 instruction -> 16 MACs)")
    print("     GPU (CUDA)        32  (1 warp -> 32 MACs, but 4 warps needed)")
    print("     GPU (TensorCore)  64  (1 MMA -> 64 MACs, but 4 MMA needed)")
    print("     TPU (Systolic)    16,384 (1 tile config -> 16K MACs)")
    print("     KPU (Domain)      256-millions (1 domain program -> entire layer)")
    print()
    print("     TPU/KPU amortize control over 100-1000x more MACs than GPU!")
    print()

    print("  4. OPERAND FETCH: THE KEY DIFFERENTIATOR")
    print("     " + "-"*90)
    print("     Pure ALU energy is ~1.8 pJ per FP32 MAC (same across architectures)")
    print("     What differs is operand fetch (register-to-ALU delivery):")
    print()
    print("     Architecture      Fetch/Op    Reuse      Bottleneck")
    print("     ----------------  ----------  ---------  -----------")
    print("     CPU               ~5.4 pJ     1x         Fetch (75%)")
    print("     GPU               ~2.2 pJ     1x         Fetch (55%)")
    print("     TPU               ~0.002 pJ   16,000x    ALU (99.9%)")
    print("     KPU               ~0.4 pJ     64x        ALU (80%)")
    print()
    print("     This explains 10-100x TOPS/W advantage of spatial architectures!")
    print()

    print("  5. WHEN EACH ARCHITECTURE WINS")
    print("     " + "-"*90)
    print("     CPU:       Small irregular workloads, branch-heavy code")
    print("     GPU-CUDA:  Element-wise ops, reductions, non-matrix kernels")
    print("     GPU-TC:    Matrix multiply, convolutions (when batch size is large)")
    print("     TPU:       Large power-of-2 matrices, batch inference, training")
    print("     KPU:       Streaming workloads, irregular sizes, batch=1 inference")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Energy Walkthrough: Architecture Comparison with GPU CUDA/TensorCore Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Run all parts with defaults
  %(prog)s --matrix 512         # Use 512x512 matrix for Part 2
  %(prog)s --trace              # Include detailed execution trace
  %(prog)s --part1              # Only run Part 1 (native operation)
  %(prog)s --part2              # Only run Part 2 (workload mapping)
  %(prog)s --part3              # Only run Part 3 (operand fetch analysis)
"""
    )
    parser.add_argument('--matrix', type=int, default=300,
                        help='Matrix dimension for Part 2 (default: 300)')
    parser.add_argument('--ops', type=int, default=10000,
                        help='Number of operations for Part 3 (default: 10000)')
    parser.add_argument('--trace', action='store_true',
                        help='Show detailed execution trace')
    parser.add_argument('--part1', action='store_true',
                        help='Only run Part 1 (native operation energy)')
    parser.add_argument('--part2', action='store_true',
                        help='Only run Part 2 (workload mapping)')
    parser.add_argument('--part3', action='store_true',
                        help='Only run Part 3 (operand fetch energy)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # If no specific part selected, run all
    any_part_selected = args.part1 or args.part2 or args.part3
    run_part1 = args.part1 or not any_part_selected
    run_part2 = args.part2 or not any_part_selected
    run_part3 = args.part3 or not any_part_selected

    comparison = ARCH_COMPARISON_8NM_X86

    print("=" * 140)
    print("  ENERGY WALKTHROUGH: Understanding Architectural Energy Differences")
    print("  Including: GPU CUDA (128 MACs/SM), GPU TensorCore (256 MACs/SM), and Operand Fetch Analysis")
    print("=" * 140)
    print()
    print(f"  Technology: {comparison.name}")
    print(f"  Process: {comparison.process_node_nm}nm")
    print(f"  Memory: {comparison.memory_type.value}")

    if run_part1:
        part1_native_operation_energy(comparison, args.verbose)

    if run_part2:
        part2_real_workload_mapping(comparison, args.matrix, args.verbose)

    if run_part3:
        part3_operand_fetch_analysis(comparison, args.ops, args.verbose)

    if args.trace:
        execution_trace(comparison, 10000, args.verbose)

    print_insights(comparison)

    print()
    print("=" * 140)
    print("  END OF WALKTHROUGH")
    print("=" * 140)


if __name__ == "__main__":
    main()
