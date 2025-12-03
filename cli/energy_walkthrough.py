#!/usr/bin/env python3
"""
Energy Walkthrough: Step-by-Step Execution Comparison

This script demonstrates architectural energy differences through TWO analyses:

PART 1: NATIVE OPERATION ENERGY
  Compare each architecture at its natural execution granularity:
  - CPU:  1 AVX-512 instruction  ->  16 MACs
  - GPU:  1 warp instruction     ->  32 MACs
  - TPU:  1 systolic tile cycle  ->  16,384 MACs (128x128)
  - KPU:  1 tile cycle           ->  256 MACs (16x16, but 64 tiles = 16,384 total)

  This answers: "What's the fundamental energy cost per native operation?"

PART 2: REAL WORKLOAD MAPPING
  Map the same workload (e.g., 300x300 MatMul = 27M MACs) to each architecture:
  - Shows tiling/partitioning overhead
  - Shows utilization efficiency (300 doesn't divide evenly into 128 or 16)
  - Shows partial tile waste

  This answers: "How efficiently can each architecture execute real workloads?"

Key insight: TPU and KPU have the same total MACs/cycle (16,384), but:
  - TPU: Single 128x128 array (poor mapping for 300x300)
  - KPU: 64 x 16x16 tiles (better mapping for irregular sizes)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphs.hardware.cycle_energy import (
    build_cpu_cycle_energy,
    build_gpu_cycle_energy,
    build_tpu_cycle_energy,
    build_kpu_cycle_energy,
)
from graphs.hardware.cycle_energy.base import OperatingMode, OperatorType
from graphs.hardware.technology_profile import ARCH_COMPARISON_8NM_X86


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


def print_header(title: str, width: int = 120):
    """Print a section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step_row(step: str, cpu_pj: float, gpu_pj: float, tpu_pj: float, kpu_pj: float,
                   col_width: int = 18):
    """Print a row comparing energy across architectures."""
    cpu_str = format_energy_fixed(cpu_pj, col_width-2) if cpu_pj > 0 else "-".center(col_width-2)
    gpu_str = format_energy_fixed(gpu_pj, col_width-2) if gpu_pj > 0 else "-".center(col_width-2)
    tpu_str = format_energy_fixed(tpu_pj, col_width-2) if tpu_pj > 0 else "-".center(col_width-2)
    kpu_str = format_energy_fixed(kpu_pj, col_width-2) if kpu_pj > 0 else "-".center(col_width-2)

    print(f"  {step:<40} {cpu_str:>{col_width}} {gpu_str:>{col_width}} {tpu_str:>{col_width}} {kpu_str:>{col_width}}")


# =============================================================================
# PART 1: NATIVE OPERATION ENERGY
# =============================================================================

def part1_native_operation_energy(comparison, verbose: bool = False):
    """
    Part 1: Compare each architecture at its NATIVE execution granularity.

    This is an apples-to-apples circuit comparison asking:
    "What's the fundamental energy cost of ONE native operation?"

    Native operations:
      CPU:  1 AVX-512 instruction  = 16 FP32 MACs
      GPU:  1 warp instruction     = 32 FP32 MACs
      TPU:  1 systolic tile cycle  = 16,384 MACs (128x128 array)
      KPU:  1 tile cycle           = 256 MACs (16x16 tile)

    Note: KPU has 64 tiles, so 1 "full cycle" = 64 x 256 = 16,384 MACs,
    but we show SINGLE TILE to compare the fundamental execution units.
    """
    print_header("PART 1: NATIVE OPERATION ENERGY (Apples-to-Apples Circuit Comparison)")
    print()
    print("  Each architecture has a NATIVE execution unit that amortizes control overhead:")
    print()
    print("    Architecture    Native Unit               MACs/Unit    Notes")
    print("    --------------- ------------------------- ------------ ---------------------------")
    print("    CPU (AVX-512)   1 vector instruction      16           FP32, 512-bit registers")
    print("    GPU (SIMT)      1 warp instruction        32           32 threads in lockstep")
    print("    TPU (Systolic)  1 tile cycle (128x128)    16,384       Weight-stationary dataflow")
    print("    KPU (Domain)    1 tile cycle (16x16)      256          Per-tile; 64 tiles total")
    print()
    print("  Question: What's the fundamental energy cost per native operation?")
    print()

    # Define native unit sizes
    cpu_ops = 16       # AVX-512: 16 FP32 MACs per vector instruction
    gpu_ops = 32       # Warp: 32 threads in lockstep
    tpu_ops = 16384    # 128x128 systolic array
    kpu_ops = 256      # Single 16x16 tile (not all 64 tiles)

    # Minimal bytes - just enough for the native operation (L1 resident)
    cpu_bytes = cpu_ops * 8    # 2 operands x 4 bytes
    gpu_bytes = gpu_ops * 8
    tpu_bytes = tpu_ops * 8
    kpu_bytes = kpu_ops * 8

    # Build breakdowns - L1 resident to focus on compute, not memory
    cpu = build_cpu_cycle_energy(
        num_ops=cpu_ops, bytes_transferred=cpu_bytes,
        mode=OperatingMode.L1_RESIDENT,
        tech_profile=comparison.cpu_profile,
        simd_width=16
    )
    gpu = build_gpu_cycle_energy(
        num_ops=gpu_ops, bytes_transferred=gpu_bytes,
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

    col_w = 18
    print(f"  {'METRIC':<40} {'CPU (16 MACs)':>{col_w}} {'GPU (32 MACs)':>{col_w}} {'TPU (16K MACs)':>{col_w}} {'KPU (256 MACs)':>{col_w}}")
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    # Energy breakdown by category
    cpu_ctrl = cpu.get_control_overhead_energy()
    gpu_ctrl = gpu.get_control_overhead_energy()
    tpu_ctrl = tpu.get_control_overhead_energy()
    kpu_ctrl = kpu.get_control_overhead_energy()

    cpu_compute = cpu.get_compute_energy()
    gpu_compute = gpu.get_compute_energy()
    tpu_compute = tpu.get_compute_energy()
    kpu_compute = kpu.get_compute_energy()

    cpu_data = cpu.get_data_movement_energy()
    gpu_data = gpu.get_data_movement_energy()
    tpu_data = tpu.get_data_movement_energy()
    kpu_data = kpu.get_data_movement_energy()

    print_step_row("Control Overhead", cpu_ctrl, gpu_ctrl, tpu_ctrl, kpu_ctrl)
    print_step_row("Compute (MACs)", cpu_compute, gpu_compute, tpu_compute, kpu_compute)
    print_step_row("Data Movement (L1)", cpu_data, gpu_data, tpu_data, kpu_data)
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("TOTAL (native unit)", cpu.total_energy_pj, gpu.total_energy_pj, tpu.total_energy_pj, kpu.total_energy_pj)

    # Per-MAC energy (the key metric!)
    print()
    print(f"  {'ENERGY PER MAC (pJ)':<40} {'CPU':>{col_w}} {'GPU':>{col_w}} {'TPU':>{col_w}} {'KPU':>{col_w}}")
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("Total / num_MACs",
                   cpu.total_energy_pj / cpu_ops,
                   gpu.total_energy_pj / gpu_ops,
                   tpu.total_energy_pj / tpu_ops,
                   kpu.total_energy_pj / kpu_ops)
    print_step_row("Control / num_MACs",
                   cpu_ctrl / cpu_ops,
                   gpu_ctrl / gpu_ops,
                   tpu_ctrl / tpu_ops,
                   kpu_ctrl / kpu_ops)
    print_step_row("Compute / num_MACs",
                   cpu_compute / cpu_ops,
                   gpu_compute / gpu_ops,
                   tpu_compute / tpu_ops,
                   kpu_compute / kpu_ops)

    # Show control overhead percentage
    print()
    print("  CONTROL OVERHEAD ANALYSIS:")
    print(f"    CPU:  {cpu_ctrl/cpu.total_energy_pj*100:5.1f}% of energy is control (fetch/decode per 16 MACs)")
    print(f"    GPU:  {gpu_ctrl/gpu.total_energy_pj*100:5.1f}% of energy is control (warp scheduler per 32 MACs)")
    print(f"    TPU:  {tpu_ctrl/tpu.total_energy_pj*100:5.1f}% of energy is control (config per 16K MACs)")
    print(f"    KPU:  {kpu_ctrl/kpu.total_energy_pj*100:5.1f}% of energy is control (domain tracker per 256 MACs)")
    print()
    print("  -> Systolic/Domain architectures amortize control over 100-1000x more ops!")


# =============================================================================
# PART 2: REAL WORKLOAD MAPPING
# =============================================================================

def calculate_tiling_efficiency(matrix_dim: int, tile_dim: int) -> dict:
    """Calculate tiling efficiency for a matrix operation."""
    # How many full tiles fit?
    full_tiles = matrix_dim // tile_dim
    remainder = matrix_dim % tile_dim

    # For matrix multiply C = A x B (MxN = MxK * KxN)
    # We tile along all dimensions
    total_tiles_needed = full_tiles * full_tiles
    partial_tiles = 0
    if remainder > 0:
        # Partial tiles on edges
        partial_tiles = 2 * full_tiles + 1  # Two edges + corner

    # Utilization within partial tiles
    if remainder > 0:
        partial_utilization = (remainder * remainder) / (tile_dim * tile_dim)
    else:
        partial_utilization = 1.0

    # Average utilization across all tiles
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

    Key insight: 300x300 doesn't map cleanly to:
    - TPU 128x128 tiles (300 = 2*128 + 44, so ~35% of last tile is wasted)
    - KPU 16x16 tiles (300 = 18*16 + 12, so ~75% of last tile is wasted)

    But KPU's finer granularity (16x16 vs 128x128) wastes less total area.
    """
    total_macs = matrix_dim * matrix_dim * matrix_dim
    bytes_transferred = 3 * matrix_dim * matrix_dim * 4  # A, B, C matrices in FP32

    print_header(f"PART 2: REAL WORKLOAD MAPPING ({matrix_dim}x{matrix_dim} MatMul = {total_macs/1e6:.1f}M MACs)")
    print()
    print(f"  Workload: C = A x B where A, B are {matrix_dim}x{matrix_dim} FP32 matrices")
    print(f"  Total MACs: {total_macs:,}")
    print(f"  Data Size: {bytes_transferred/1024/1024:.1f} MB (3 matrices x 4 bytes)")
    print()

    # Calculate tiling efficiency for each architecture
    tpu_tile = 128
    kpu_tile = 16

    tpu_eff = calculate_tiling_efficiency(matrix_dim, tpu_tile)
    kpu_eff = calculate_tiling_efficiency(matrix_dim, kpu_tile)

    print("  TILING ANALYSIS:")
    print("  ----------------")
    print(f"    TPU (128x128 tiles):")
    print(f"      {matrix_dim} = {matrix_dim // tpu_tile} x {tpu_tile} + {matrix_dim % tpu_tile} remainder")
    print(f"      Full tiles: {tpu_eff['full_tiles']}, Partial tiles: {tpu_eff['partial_tiles']}")
    print(f"      Partial tile utilization: {tpu_eff['partial_utilization']*100:.1f}%")
    print(f"      Average utilization: {tpu_eff['avg_utilization']*100:.1f}%")
    print()
    print(f"    KPU (16x16 tiles):")
    print(f"      {matrix_dim} = {matrix_dim // kpu_tile} x {kpu_tile} + {matrix_dim % kpu_tile} remainder")
    print(f"      Full tiles: {kpu_eff['full_tiles']}, Partial tiles: {kpu_eff['partial_tiles']}")
    print(f"      Partial tile utilization: {kpu_eff['partial_utilization']*100:.1f}%")
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
    gpu = build_gpu_cycle_energy(
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

    col_w = 18
    print(f"  {'EXECUTION METRICS':<40} {'CPU (AVX-512)':>{col_w}} {'GPU (SIMT)':>{col_w}} {'TPU (128x128)':>{col_w}} {'KPU (16x16x64)':>{col_w}}")
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    # Show execution parameters
    cpu_instrs = total_macs // 16  # AVX-512
    gpu_instrs = total_macs // 32  # Warp
    tpu_cycles = (total_macs + 16383) // 16384  # 128x128 tiles
    kpu_cycles = (total_macs + 16383) // 16384  # 64 x 16x16 tiles = 16384 total

    print(f"  {'Total MACs':<40} {total_macs:>{col_w},} {total_macs:>{col_w},} {total_macs:>{col_w},} {total_macs:>{col_w},}")
    print(f"  {'Instructions/Cycles needed':<40} {cpu_instrs:>{col_w},} {gpu_instrs:>{col_w},} {tpu_cycles:>{col_w},} {kpu_cycles:>{col_w},}")
    print(f"  {'Tiling utilization':<40} {'100%':>{col_w}} {'100%':>{col_w}} {tpu_eff['avg_utilization']*100:>{col_w-1}.1f}% {kpu_eff['avg_utilization']*100:>{col_w-1}.1f}%")
    print()

    # Energy breakdown
    print(f"  {'ENERGY BREAKDOWN':<40} {'CPU':>{col_w}} {'GPU':>{col_w}} {'TPU':>{col_w}} {'KPU':>{col_w}}")
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    cpu_ctrl = cpu.get_control_overhead_energy()
    gpu_ctrl = gpu.get_control_overhead_energy()
    tpu_ctrl = tpu.get_control_overhead_energy()
    kpu_ctrl = kpu.get_control_overhead_energy()

    cpu_compute = cpu.get_compute_energy()
    gpu_compute = gpu.get_compute_energy()
    tpu_compute = tpu.get_compute_energy()
    kpu_compute = kpu.get_compute_energy()

    cpu_data = cpu.get_data_movement_energy()
    gpu_data = gpu.get_data_movement_energy()
    tpu_data = tpu.get_data_movement_energy()
    kpu_data = kpu.get_data_movement_energy()

    print_step_row("Control Overhead", cpu_ctrl, gpu_ctrl, tpu_ctrl, kpu_ctrl)
    print_step_row("Compute (MACs)", cpu_compute, gpu_compute, tpu_compute, kpu_compute)
    print_step_row("Data Movement", cpu_data, gpu_data, tpu_data, kpu_data)
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("TOTAL ENERGY", cpu.total_energy_pj, gpu.total_energy_pj, tpu.total_energy_pj, kpu.total_energy_pj)

    # Per-MAC efficiency
    print()
    print(f"  {'EFFICIENCY METRICS':<40} {'CPU':>{col_w}} {'GPU':>{col_w}} {'TPU':>{col_w}} {'KPU':>{col_w}}")
    print(f"  {'-'*40} {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w}")
    print_step_row("Energy per MAC (pJ)",
                   cpu.total_energy_pj / total_macs,
                   gpu.total_energy_pj / total_macs,
                   tpu.total_energy_pj / total_macs,
                   kpu.total_energy_pj / total_macs)

    best_total = min(cpu.total_energy_pj, gpu.total_energy_pj, tpu.total_energy_pj, kpu.total_energy_pj)
    print(f"  {'vs Best':<40} {cpu.total_energy_pj/best_total:>{col_w}.2f}x {gpu.total_energy_pj/best_total:>{col_w}.2f}x {tpu.total_energy_pj/best_total:>{col_w}.2f}x {kpu.total_energy_pj/best_total:>{col_w}.2f}x")
    print(f"  {'Control %':<40} {cpu_ctrl/cpu.total_energy_pj*100:>{col_w-1}.1f}% {gpu_ctrl/gpu.total_energy_pj*100:>{col_w-1}.1f}% {tpu_ctrl/tpu.total_energy_pj*100:>{col_w-1}.1f}% {kpu_ctrl/kpu.total_energy_pj*100:>{col_w-1}.1f}%")
    print(f"  {'Data Movement %':<40} {cpu_data/cpu.total_energy_pj*100:>{col_w-1}.1f}% {gpu_data/gpu.total_energy_pj*100:>{col_w-1}.1f}% {tpu_data/tpu.total_energy_pj*100:>{col_w-1}.1f}% {kpu_data/kpu.total_energy_pj*100:>{col_w-1}.1f}%")

    print()
    print("  KEY OBSERVATIONS:")
    print(f"    1. Data movement dominates for DRAM-resident workloads ({bytes_transferred/1024/1024:.1f} MB)")
    print(f"    2. TPU and KPU have same total throughput (16,384 MACs/cycle)")
    print(f"    3. KPU's finer tiles ({kpu_tile}x{kpu_tile}) map better to {matrix_dim}x{matrix_dim}")
    print(f"       - TPU waste: {(1-tpu_eff['avg_utilization'])*100:.1f}% of partial tiles")
    print(f"       - KPU waste: {(1-kpu_eff['avg_utilization'])*100:.1f}% of partial tiles")


# =============================================================================
# DETAILED EXECUTION TRACE
# =============================================================================

def execution_trace(comparison, num_ops: int = 10000, verbose: bool = False):
    """
    Show detailed execution trace for each architecture.
    """
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
    gpu = build_gpu_cycle_energy(
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
        """Print execution trace for one architecture."""
        print(f"\n  {name}")
        print(f"  {'-'*90}")

        total = 0
        for event in breakdown.events:
            event_total = event.energy_pj * event.count
            total += event_total
            desc = event.description[:55] if len(event.description) > 55 else event.description
            print(f"    {desc:<55} {event.count:>10,} x {event.energy_pj:>8.2f} pJ = {format_energy_fixed(event_total, 12)}")

        print(f"  {'-'*90}")
        print(f"    {'TOTAL':<55} {'':>10} {'':>12}   {format_energy_fixed(total, 12)}")
        print(f"    {'Per operation':<55} {'':>10} {'':>12}   {format_energy_fixed(total/num_ops, 12)}")

    print_trace("CPU (AVX-512)", cpu)
    print_trace("GPU (SIMT)", gpu)
    print_trace("TPU (Systolic 128x128)", tpu)
    print_trace("KPU (Domain Flow 64x16x16)", kpu)


# =============================================================================
# INSIGHTS
# =============================================================================

def print_insights(comparison):
    """Print key insights about architectural energy differences."""
    print_header("KEY INSIGHTS: Understanding Architectural Energy Differences")
    print()
    print("  1. CONTROL OVERHEAD AMORTIZATION")
    print("     " + "-"*70)
    print("     Architecture     Control Amortization    Reason")
    print("     --------------- ----------------------- -----------------------------")
    print("     CPU (AVX-512)   1 fetch/decode per 16   Stored-program; every instr")
    print("     GPU (SIMT)      1 fetch/decode per 32   Warp broadcasts instruction")
    print("     TPU (Systolic)  1 config per 16,384     Load tile weights once")
    print("     KPU (Domain)    1 program per layer     SURE program, not instructions")
    print()
    print("     -> TPU/KPU amortize control over 500-1000x more MACs!")
    print()

    print("  2. NATIVE EXECUTION UNIT SIZE")
    print("     " + "-"*70)
    print("     Architecture     Native Unit     MACs      Silicon Efficiency")
    print("     --------------- --------------- -------- -------------------------")
    print("     CPU (AVX-512)   1 instruction   16       General-purpose, flexible")
    print("     GPU (SIMT)      1 warp          32       SIMD + shared memory")
    print("     TPU (Systolic)  1 tile 128x128  16,384   Fixed matrix dataflow")
    print("     KPU (Domain)    1 tile 16x16    256      Programmable dataflow")
    print()
    print("     KPU has 64 tiles -> same 16,384 MACs/cycle as TPU")
    print("     But finer granularity (16x16 vs 128x128) maps better to irregular sizes")
    print()

    print("  3. WORKLOAD MAPPING EFFICIENCY")
    print("     " + "-"*70)
    print("     For 300x300 matrix multiply:")
    print("       - CPU/GPU: No tiling waste (scalar or warp granularity)")
    print("       - TPU: 300 = 2x128 + 44, so partial tiles are 34% utilized")
    print("       - KPU: 300 = 18x16 + 12, so partial tiles are 56% utilized")
    print()
    print("     For 1024x1024 matrix multiply:")
    print("       - TPU: 1024 = 8x128, perfect fit!")
    print("       - KPU: 1024 = 64x16, perfect fit!")
    print()
    print("     -> TPU excels at power-of-2 sizes; KPU handles irregular sizes better")
    print()

    print("  4. WHEN EACH ARCHITECTURE WINS")
    print("     " + "-"*70)
    print("     CPU:  Small irregular workloads, branch-heavy code, single-threaded")
    print("     GPU:  Large batches (>10K ops) where kernel launch amortizes")
    print("     TPU:  Large power-of-2 matrices, batch inference, training")
    print("     KPU:  Streaming workloads, irregular sizes, low-batch inference")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Energy Walkthrough: Two-Part Architecture Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Run both parts with defaults
  %(prog)s --matrix 512         # Use 512x512 matrix for Part 2
  %(prog)s --trace              # Include detailed execution trace
  %(prog)s --part1              # Only run Part 1 (native operation)
  %(prog)s --part2              # Only run Part 2 (workload mapping)
"""
    )
    parser.add_argument('--matrix', type=int, default=300,
                        help='Matrix dimension for Part 2 (default: 300)')
    parser.add_argument('--trace', action='store_true',
                        help='Show detailed execution trace')
    parser.add_argument('--part1', action='store_true',
                        help='Only run Part 1 (native operation energy)')
    parser.add_argument('--part2', action='store_true',
                        help='Only run Part 2 (workload mapping)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Default to both parts if neither specified
    run_part1 = args.part1 or not (args.part1 or args.part2)
    run_part2 = args.part2 or not (args.part1 or args.part2)

    comparison = ARCH_COMPARISON_8NM_X86

    print("=" * 120)
    print("  ENERGY WALKTHROUGH: Understanding Architectural Energy Differences")
    print("=" * 120)
    print()
    print(f"  Technology: {comparison.name}")
    print(f"  Process: {comparison.process_node_nm}nm")
    print(f"  Memory: {comparison.memory_type.value}")

    if run_part1:
        part1_native_operation_energy(comparison, args.verbose)

    if run_part2:
        part2_real_workload_mapping(comparison, args.matrix, args.verbose)

    if args.trace:
        execution_trace(comparison, 10000, args.verbose)

    print_insights(comparison)

    print()
    print("=" * 120)
    print("  END OF WALKTHROUGH")
    print("=" * 120)


if __name__ == "__main__":
    main()
