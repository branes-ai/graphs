#!/usr/bin/env python3
"""
Architecture Energy Comparison Tool

Compare energy consumption across different processor architecture classes:
- Stored Program Machines: CPU (MIMD), GPU (SIMT)
- Dataflow Architectures: TPU (Systolic Array), KPU (Domain Flow / SURE)

This tool focuses on comparing general-purpose processors (CPU, GPU) against
dataflow accelerators (TPU, KPU) to understand the energy trade-offs
between flexibility and efficiency.

KPU is the Stillwater Supercomputing KPU, a Domain Flow Architecture that
implements direct execution of Systems of Uniform Recurrence Equations (SURE).

Usage:
    # Compare all four architectures
    ./cli/compare_architecture_energy.py

    # Compare with specific workload size
    ./cli/compare_architecture_energy.py --ops 10000 --bytes 40960

    # Show detailed cycle breakdown
    ./cli/compare_architecture_energy.py --verbose

    # Compare at different operation scales
    ./cli/compare_architecture_energy.py --sweep

    # Compare across operating modes
    ./cli/compare_architecture_energy.py --mode-sweep
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Import from the shared cycle_energy package
from graphs.hardware.cycle_energy import (
    # Base classes
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
    get_mode_description,
    # Architecture models
    build_cpu_cycle_energy,
    build_gpu_cycle_energy,
    build_tpu_cycle_energy,
    build_kpu_cycle_energy,
    # Comparison utilities
    format_energy,
    format_phase_breakdown,
    format_comparison_table,
    format_key_insights,
)


def format_architecture_comparison_table(
    breakdowns: List[CycleEnergyBreakdown],
    mode: Optional[OperatingMode] = None,
    num_ops: int = 1000
) -> str:
    """Format a comparison table for CPU/GPU/TPU/KPU."""
    lines = []
    lines.append("\n" + "="*100)
    if mode:
        lines.append(f"  ARCHITECTURE ENERGY COMPARISON - {mode.value.upper()} Mode")
    else:
        lines.append("  ARCHITECTURE ENERGY COMPARISON")
    lines.append("="*100)

    # Header
    lines.append(f"\n  {'Architecture':<35} {'Total (pJ)':<15} {'Per Op':<15} {'Relative':<12}")
    lines.append(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*12}")

    # Find baseline (CPU) for relative comparison
    baseline_energy = breakdowns[0].total_energy_pj if breakdowns else 1.0

    for breakdown in breakdowns:
        total = breakdown.total_energy_pj
        per_op = total / num_ops
        relative = total / baseline_energy

        lines.append(f"  {breakdown.architecture_name[:35]:<35} "
                    f"{format_energy(total):>12} "
                    f"{format_energy(per_op):>12} "
                    f"{relative:>10.2f}x")

    # Add architecture class summary
    lines.append(f"\n  ARCHITECTURE CLASSES")
    lines.append(f"  {'-'*60}")
    for breakdown in breakdowns:
        lines.append(f"  {breakdown.architecture_name[:25]:<25} -> {breakdown.architecture_class}")

    return "\n".join(lines)


def format_detailed_phase_comparison(
    breakdowns: List[CycleEnergyBreakdown],
    num_ops: int = 1000
) -> str:
    """Format detailed phase comparison showing all architecture phases."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("  DETAILED PHASE BREAKDOWN")
    lines.append("="*100)

    COL_WIDTH = 20

    # Group phases by category
    phase_groups = [
        ("STORED PROGRAM OVERHEAD", [
            ("Instruction Fetch", CyclePhase.INSTRUCTION_FETCH),
            ("Instruction Decode", CyclePhase.INSTRUCTION_DECODE),
            ("Operand Fetch", CyclePhase.OPERAND_FETCH),
            ("Execute", CyclePhase.EXECUTE),
            ("Writeback", CyclePhase.WRITEBACK),
        ]),
        ("GPU SIMT OVERHEAD", [
            ("Fixed Infrastructure", CyclePhase.SIMT_FIXED_OVERHEAD),
            ("Thread Management", CyclePhase.SIMT_THREAD_MGMT),
            ("Coherence", CyclePhase.SIMT_COHERENCE),
            ("Synchronization", CyclePhase.SIMT_SYNC),
        ]),
        ("TPU SYSTOLIC ARRAY", [
            ("Control", CyclePhase.SYSTOLIC_CONTROL),
            ("Weight Load", CyclePhase.SYSTOLIC_WEIGHT_LOAD),
            ("Data Load", CyclePhase.SYSTOLIC_DATA_LOAD),
            ("Compute", CyclePhase.SYSTOLIC_COMPUTE),
            ("Drain", CyclePhase.SYSTOLIC_DRAIN),
        ]),
        ("KPU SPATIAL DATAFLOW", [
            ("Configuration", CyclePhase.SPATIAL_CONFIG),
            ("Streaming", CyclePhase.SPATIAL_STREAM),
            ("Compute", CyclePhase.SPATIAL_COMPUTE),
            ("Interconnect", CyclePhase.SPATIAL_INTERCONNECT),
        ]),
        ("MEMORY ACCESS", [
            ("L1/SRAM", CyclePhase.MEM_L1),
            ("SRAM (on-chip)", CyclePhase.MEM_SRAM),
            ("L2 Cache", CyclePhase.MEM_L2),
            ("L3 Cache", CyclePhase.MEM_L3),
            ("DRAM", CyclePhase.MEM_DRAM),
            ("HBM", CyclePhase.MEM_HBM),
        ]),
    ]

    # Header row
    header = f"  {'Phase':<25}"
    for b in breakdowns:
        arch_name = b.architecture_name.split()[0][:COL_WIDTH-2]
        header += f" {arch_name:>{COL_WIDTH}}"
    lines.append(header)

    # Separator
    sep = f"  {'-'*25}"
    for _ in breakdowns:
        sep += f" {'-'*COL_WIDTH}"
    lines.append(sep)

    def format_energy_cell(energy_pj: float, total_pj: float) -> str:
        if energy_pj == 0:
            return "n/a"
        pct = (energy_pj / total_pj * 100) if total_pj > 0 else 0
        if energy_pj < 1000:
            return f"{energy_pj:.0f}pJ ({pct:.0f}%)"
        elif energy_pj < 1_000_000:
            return f"{energy_pj/1000:.1f}nJ ({pct:.0f}%)"
        else:
            return f"{energy_pj/1_000_000:.1f}uJ ({pct:.0f}%)"

    for group_name, phases in phase_groups:
        # Check if any architecture has energy in this group
        group_has_energy = False
        for _, phase in phases:
            for bd in breakdowns:
                if bd.get_phase_energy(phase) > 0:
                    group_has_energy = True
                    break
            if group_has_energy:
                break

        if not group_has_energy:
            continue

        # Group header
        lines.append(f"\n  {group_name}")
        lines.append(f"  {'-'*25}" + (" " + "-"*COL_WIDTH) * len(breakdowns))

        for phase_name, phase in phases:
            row = f"    {phase_name:<23}"
            has_any = False

            for bd in breakdowns:
                energy = bd.get_phase_energy(phase)
                if energy > 0:
                    has_any = True
                    cell = format_energy_cell(energy, bd.total_energy_pj)
                else:
                    cell = "n/a"
                row += f" {cell:>{COL_WIDTH}}"

            if has_any:
                lines.append(row)

    # Total row
    lines.append(f"\n  {'-'*25}" + (" " + "-"*COL_WIDTH) * len(breakdowns))
    row = f"  {'TOTAL':<25}"
    for bd in breakdowns:
        row += f" {format_energy(bd.total_energy_pj):>{COL_WIDTH}}"
    lines.append(row)

    return "\n".join(lines)


def format_architecture_insights(breakdowns: List[CycleEnergyBreakdown], num_ops: int) -> str:
    """Generate insights comparing stored program vs dataflow architectures."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("  KEY INSIGHTS: STORED PROGRAM vs DATAFLOW ARCHITECTURES")
    lines.append("="*100)

    if len(breakdowns) >= 4:
        cpu, gpu, tpu, kpu = breakdowns[0], breakdowns[1], breakdowns[2], breakdowns[3]

        # Calculate overhead ratios
        cpu_instr = (cpu.get_phase_energy(CyclePhase.INSTRUCTION_FETCH) +
                    cpu.get_phase_energy(CyclePhase.INSTRUCTION_DECODE))
        cpu_instr_pct = (cpu_instr / cpu.total_energy_pj * 100) if cpu.total_energy_pj > 0 else 0

        gpu_simt = gpu.get_simt_overhead()
        gpu_simt_pct = (gpu_simt / gpu.total_energy_pj * 100) if gpu.total_energy_pj > 0 else 0

        tpu_systolic = tpu.get_systolic_overhead()
        tpu_systolic_pct = (tpu_systolic / tpu.total_energy_pj * 100) if tpu.total_energy_pj > 0 else 0

        kpu_spatial = kpu.get_spatial_overhead()
        kpu_spatial_pct = (kpu_spatial / kpu.total_energy_pj * 100) if kpu.total_energy_pj > 0 else 0

        # Energy per op
        cpu_per_op = cpu.total_energy_pj / num_ops
        gpu_per_op = gpu.total_energy_pj / num_ops
        tpu_per_op = tpu.total_energy_pj / num_ops
        kpu_per_op = kpu.total_energy_pj / num_ops

        lines.append(f"""
  STORED PROGRAM MACHINES (Instruction-driven)
  ==============================================

  1. CPU (MIMD Stored Program):
     - Architecture: {cpu.architecture_class}
     - Energy/op: {format_energy(cpu_per_op)}
     - Instruction overhead: {format_energy(cpu_instr)} ({cpu_instr_pct:.1f}%)
     - Trade-off: Maximum flexibility, highest per-op energy

  2. GPU (SIMT Data Parallel):
     - Architecture: {gpu.architecture_class}
     - Energy/op: {format_energy(gpu_per_op)}
     - SIMT overhead: {format_energy(gpu_simt)} ({gpu_simt_pct:.1f}%)
     - Trade-off: Parallel flexibility, but fixed infrastructure cost

  DATAFLOW ARCHITECTURES (No instruction fetch per op)
  ====================================================

  3. TPU (Systolic Array):
     - Architecture: {tpu.architecture_class}
     - Energy/op: {format_energy(tpu_per_op)}
     - Systolic energy: {format_energy(tpu_systolic)} ({tpu_systolic_pct:.1f}%)
     - Trade-off: Highly efficient MACs, but data movement overhead

  4. KPU (Domain Flow Architecture):
     - Architecture: {kpu.architecture_class}
     - Energy/op: {format_energy(kpu_per_op)}
     - Domain flow energy: {format_energy(kpu_spatial)} ({kpu_spatial_pct:.1f}%)
     - Trade-off: SURE execution efficient, near 100% util at batch=1

  COMPARISON SUMMARY
  ==================

  Energy/Op Rankings (lower is better):
  """)

        # Sort by energy per op
        rankings = [
            ("CPU", cpu_per_op),
            ("GPU", gpu_per_op),
            ("TPU", tpu_per_op),
            ("KPU", kpu_per_op),
        ]
        rankings.sort(key=lambda x: x[1])

        for i, (name, energy) in enumerate(rankings, 1):
            ratio = energy / rankings[0][1]
            lines.append(f"    {i}. {name}: {format_energy(energy)} ({ratio:.1f}x vs best)")

        # Key insight
        best = rankings[0][0]
        worst = rankings[-1][0]
        efficiency_ratio = rankings[-1][1] / rankings[0][1]

        lines.append(f"""

  KEY INSIGHT:
  - {best} is most energy-efficient at this scale ({format_energy(rankings[0][1])}/op)
  - {worst} is least efficient ({format_energy(rankings[-1][1])}/op)
  - Efficiency ratio: {efficiency_ratio:.1f}x between best and worst

  FUNDAMENTAL TRADE-OFF:
  ----------------------
  Stored Program (CPU/GPU):
    + Flexible - can run any algorithm
    + Programmable - just load new instructions
    - Pay instruction fetch/decode per operation
    - GPU: Pay SIMT overhead (warp scheduling, coherence)

  Systolic Array (TPU):
    + No instruction fetch per operation
    + Fixed datapath = highly optimized energy for matrix ops
    - Fixed function - limited to matrix operations (GEMM)
    - Fill/drain overhead, poor utilization at small batch sizes

  Domain Flow Machine (KPU):
    + No instruction fetch per operation
    + PROGRAMMABLE - executes ANY system of uniform recurrence equations
    + Covers signal processing, linear algebra, constraint solvers, optimizers
    + Distributed CAM across PEs solves dataflow scalability problem
    + Near 100% utilization at any batch size
    - Domain program configuration overhead when switching algorithms

  KEY INNOVATION (Stillwater KPU):
  Traditional dataflow machines have a centralized CAM (Content-Addressable
  Memory) that becomes a bottleneck, forcing reduced cycle times as the
  machine scales. The KPU distributes the CAM across processing elements,
  enabling scalability without cycle time reduction - solving the
  quintessential problem that limited dataflow machines in general
  parallel computing.
""")

    return "\n".join(lines)


def run_architecture_sweep(mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
                           verbose: bool = False) -> None:
    """Run a sweep across different operation scales."""
    print("\n" + "="*100)
    print(f"  ENERGY SCALING ANALYSIS (All Architectures) - {mode.value.upper()} Mode")
    print("="*100)

    scales = [100, 1000, 10000, 100000, 1000000]
    bytes_per_op = 4

    results = []
    for ops in scales:
        bytes_transferred = ops * bytes_per_op

        cpu = build_cpu_cycle_energy(ops, bytes_transferred, mode=mode)
        gpu = build_gpu_cycle_energy(ops, bytes_transferred, mode=mode)
        tpu = build_tpu_cycle_energy(ops, bytes_transferred, mode=mode)
        kpu = build_kpu_cycle_energy(ops, bytes_transferred, mode=mode)

        results.append({
            'ops': ops,
            'cpu_per_op': cpu.total_energy_pj / ops,
            'gpu_per_op': gpu.total_energy_pj / ops,
            'tpu_per_op': tpu.total_energy_pj / ops,
            'kpu_per_op': kpu.total_energy_pj / ops,
            'cpu_total': cpu.total_energy_pj,
            'gpu_total': gpu.total_energy_pj,
            'tpu_total': tpu.total_energy_pj,
            'kpu_total': kpu.total_energy_pj,
        })

    # TABLE 1: AMORTIZED ENERGY PER OPERATION
    print(f"\n  TABLE 1: AMORTIZED ENERGY PER OPERATION")
    print(f"  {'-'*100}")
    print(f"  {'Operations':<12} {'CPU':<14} {'GPU':<14} {'TPU':<14} {'KPU':<14} {'Best':<10}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

    for r in results:
        energies = {
            'CPU': r['cpu_per_op'],
            'GPU': r['gpu_per_op'],
            'TPU': r['tpu_per_op'],
            'KPU': r['kpu_per_op']
        }
        best = min(energies, key=energies.get)

        print(f"  {r['ops']:<12,} "
              f"{format_energy(r['cpu_per_op']):>12} "
              f"{format_energy(r['gpu_per_op']):>12} "
              f"{format_energy(r['tpu_per_op']):>12} "
              f"{format_energy(r['kpu_per_op']):>12} "
              f"{best:>8}")

    # TABLE 2: TOTAL ENERGY
    print(f"\n  TABLE 2: TOTAL ENERGY")
    print(f"  {'-'*100}")
    print(f"  {'Operations':<12} {'CPU':<16} {'GPU':<16} {'TPU':<16} {'KPU':<16}")
    print(f"  {'-'*12} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")

    for r in results:
        print(f"  {r['ops']:<12,} "
              f"{format_energy(r['cpu_total']):>14} "
              f"{format_energy(r['gpu_total']):>14} "
              f"{format_energy(r['tpu_total']):>14} "
              f"{format_energy(r['kpu_total']):>14}")

    print(f"""

  OBSERVATIONS:
  -------------
  1. GPU has HIGH energy/op at small scales due to fixed SIMT infrastructure
     (kernel launch, SM activation, memory controllers)

  2. TPU has LOW energy/op due to:
     - No instruction fetch per operation
     - Systolic MACs are extremely efficient (0.1 pJ each)
     - BUT: Fill/drain overhead, 10-20% utilization at batch=1

  3. KPU (Domain Flow) has LOWEST energy/op due to:
     - No instruction fetch (SURE-based execution)
     - Domain program loaded once per layer
     - Near 100% utilization even at batch=1
     - Heterogeneous tiles (INT8/BF16/Matrix)

  4. CPU has consistent energy/op across all scales (no fixed infrastructure)

  CROSSOVER POINTS:
  - GPU becomes competitive with CPU at ~100K+ ops (amortizes fixed overhead)
  - TPU is efficient for large matrix operations (batch > 8)
  - KPU dominates at all scales due to SURE execution model
""")


def run_architecture_mode_sweep(num_ops: int = 10000, bytes_transferred: int = 40960,
                                 verbose: bool = False) -> None:
    """Run a sweep across operating modes for all architectures."""
    print("\n" + "="*100)
    print("  OPERATING MODE COMPARISON (All Architectures)")
    print("="*100)
    print(f"  Workload: {num_ops:,} ops, {bytes_transferred:,} bytes")
    print()

    modes = [
        OperatingMode.L1_RESIDENT,
        OperatingMode.L2_RESIDENT,
        OperatingMode.L3_RESIDENT,
        OperatingMode.DRAM_RESIDENT,
    ]

    # Header
    print(f"  {'Mode':<15} {'CPU':<18} {'GPU':<18} {'TPU':<18} {'KPU':<18} {'Notes':<20}")
    print(f"  {'-'*15} {'-'*18} {'-'*18} {'-'*18} {'-'*18} {'-'*20}")

    results = []

    for mode in modes:
        cpu = build_cpu_cycle_energy(num_ops, bytes_transferred, mode=mode)
        gpu = build_gpu_cycle_energy(num_ops, bytes_transferred, mode=mode)
        tpu = build_tpu_cycle_energy(num_ops, bytes_transferred, mode=mode)
        kpu = build_kpu_cycle_energy(num_ops, bytes_transferred, mode=mode)

        cpu_str = format_energy(cpu.total_energy_pj)
        gpu_str = format_energy(gpu.total_energy_pj)
        tpu_str = format_energy(tpu.total_energy_pj)
        kpu_str = format_energy(kpu.total_energy_pj)

        # Notes about L3 availability
        if mode == OperatingMode.L3_RESIDENT:
            notes = "CPU/KPU only"
        elif mode == OperatingMode.L1_RESIDENT:
            notes = "On-chip SRAM"
        elif mode == OperatingMode.L2_RESIDENT:
            notes = "Cache hierarchy"
        else:
            notes = "Off-chip memory"

        print(f"  {mode.value:<15} {cpu_str:>16} {gpu_str:>16} {tpu_str:>16} {kpu_str:>16} {notes:<20}")

        results.append({
            'mode': mode,
            'cpu': cpu.total_energy_pj,
            'gpu': gpu.total_energy_pj,
            'tpu': tpu.total_energy_pj,
            'kpu': kpu.total_energy_pj,
        })

    print()
    print("  WINNER BY MODE:")
    print(f"  {'-'*60}")
    for r in results:
        energies = {'CPU': r['cpu'], 'GPU': r['gpu'], 'TPU': r['tpu'], 'KPU': r['kpu']}
        winner = min(energies, key=energies.get)
        winner_energy = energies[winner]
        runner_up = sorted(energies.values())[1]
        ratio = runner_up / winner_energy if winner_energy > 0 else 0

        print(f"  {r['mode'].value:<15} -> {winner} ({ratio:.1f}x more efficient than runner-up)")

    print(f"""

  KEY INSIGHTS BY MODE:
  ---------------------
  L1-Resident (On-chip SRAM/Scratchpad):
    - All architectures have fast on-chip memory
    - KPU distributed scratchpads (256KB/tile) excel here
    - GPU shared memory competitive if used properly

  L2-Resident (Cache hierarchy):
    - GPU coherence overhead becomes significant
    - CPU and KPU benefit from larger working sets

  L3-Resident (CPU and KPU only):
    - CPU: 8-64MB LLC filters DRAM accesses
    - KPU: 4-16MB shared L3 filters LPDDR accesses
    - GPU/TPU: No L3 - go directly to HBM from L2
    - This mode shows CPU/KPU advantage for medium working sets

  DRAM-Resident (Off-chip streaming):
    - Memory bandwidth dominates energy
    - HBM (GPU/TPU) more efficient per byte than DDR/LPDDR
    - But L3 filtering (CPU/KPU) reduces DRAM access frequency
""")


def print_architecture_diagrams() -> None:
    """Print ASCII art diagrams for all four architectures."""
    print("""
================================================================================
  ARCHITECTURE COMPARISON: STORED PROGRAM vs DATAFLOW
================================================================================

  STORED PROGRAM MACHINES execute instructions from memory.
  DATAFLOW ARCHITECTURES have fixed datapaths - no instruction fetch per op.

--------------------------------------------------------------------------------
  CPU: MIMD Stored Program Machine (Intel Xeon / AMD EPYC)
--------------------------------------------------------------------------------

  INSTRUCTION FETCH -> DECODE -> OPERAND FETCH -> EXECUTE -> WRITEBACK
         |                                            |
         v                                            v
    (I-cache)                                   MEMORY ACCESS
                                            (L1 -> L2 -> L3 -> DRAM)

  Key: Flexible, but high per-op instruction overhead

--------------------------------------------------------------------------------
  GPU: SIMT Data Parallel (NVIDIA H100 / Jetson)
--------------------------------------------------------------------------------

  INSTRUCTION FETCH -> DECODE -> WARP SCHEDULING -> EXECUTE (32 threads)
         |                           |                    |
         v                           v                    v
    (per warp)             COHERENCE MACHINERY      MEMORY ACCESS
                           (queue, coalesce, tags)  (Shared/L1 -> L2 -> HBM)

  Key: Amortizes instruction cost across 32 threads, but SIMT overhead dominates

--------------------------------------------------------------------------------
  TPU: Systolic Array (Google TPU v4 / Coral Edge TPU)
--------------------------------------------------------------------------------

  WEIGHT LOAD -> DATA LOAD -> SYSTOLIC COMPUTE (128x128) -> DRAIN
       |             |                |                       |
       v             v                v                       v
  (weight-stationary)  (input edge)  [MAC][MAC][MAC]...  (output edge)
                                     [MAC][MAC][MAC]...
                                     [MAC][MAC][MAC]...

  Key: No instruction fetch! Extremely efficient MACs (0.1 pJ each)
       But: Data movement overhead (load/drain)

--------------------------------------------------------------------------------
  KPU: Domain Flow Architecture (Stillwater Supercomputing)
--------------------------------------------------------------------------------

  Implements direct execution of Systems of Uniform Recurrence Equations (SURE)

  DOMAIN PROGRAM -> DOMAIN TRACKER -> TILE ARRAY (8x8) -> OUTPUT AGGREGATION
       |                 |                 |                    |
       v                 v                 v                    v
  (load once)    (configure tiles)   [INT8][BF16][MTX]   (streaming)
                                     [INT8][INT8][BF16]
                                     (heterogeneous)

  Key: NO instruction fetch per op! Data-driven SURE execution
       - PROGRAMMABLE: Executes ANY system of uniform recurrence equations
       - Covers: signal processing, linear algebra, solvers, optimizers
       - Domain program loaded once (not per operation)
       - Distributed CAM solves dataflow scalability problem
       - Near 100% utilization even at batch=1 (vs TPU's 10-20%)

  Theoretical Foundation:
       - Karp-Miller-Winograd SURE theory
       - Omtzigt Computational Space-Times (1994)

================================================================================
  ENERGY TRADE-OFFS
================================================================================

  Architecture      | Flexibility | Per-Op Overhead | Best For
  ------------------|-------------|-----------------|-------------------------
  CPU (MIMD)        | Maximum     | Highest         | General purpose, irregular
  GPU (SIMT)        | High        | High (SIMT)     | Large parallel workloads
  TPU (Systolic)    | Low         | Very Low        | Matrix operations (GEMM)
  KPU (Domain Flow) | High        | Very Low        | SURE algorithms, edge AI

  KPU vs TPU:
  - KPU: PROGRAMMABLE (any SURE), ~100% util at batch=1, distributed CAM
  - TPU: Fixed systolic schedule, 10-20% util at batch=1, fill/drain overhead

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description='Compare energy across CPU, GPU, TPU, and KPU architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (DRAM-resident mode, default)
  %(prog)s

  # Compare in L1-resident mode (all data on-chip)
  %(prog)s --mode l1

  # Custom workload size
  %(prog)s --ops 10000 --bytes 40960

  # Show architecture diagrams
  %(prog)s --diagram

  # Detailed phase breakdown
  %(prog)s --verbose

  # Operation scaling sweep
  %(prog)s --sweep

  # Mode comparison (all modes side-by-side)
  %(prog)s --mode-sweep

  # JSON output
  %(prog)s --output results.json
"""
    )

    parser.add_argument('--ops', type=int, default=10000,
                        help='Number of operations (default: 10000)')
    parser.add_argument('--bytes', type=int, default=40960,
                        help='Bytes transferred (default: 40960)')
    parser.add_argument('--threads', type=int, default=200_000,
                        help='GPU concurrent threads (default: 200000)')
    parser.add_argument('--mode', type=str, default='dram',
                        choices=['l1', 'l2', 'l3', 'dram'],
                        help='Operating mode: l1, l2, l3 (CPU/KPU only), dram (default: dram)')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of neural network layers for KPU (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed phase breakdown')
    parser.add_argument('--diagram', '-d', action='store_true',
                        help='Show architecture diagrams')
    parser.add_argument('--sweep', action='store_true',
                        help='Run scaling sweep across operation counts')
    parser.add_argument('--mode-sweep', action='store_true',
                        help='Compare energy across all operating modes')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (JSON format)')

    args = parser.parse_args()

    # Parse operating mode
    # CPU and KPU have L3 caches that filter DRAM accesses
    # GPU and TPU do not have L3 - they go directly to HBM from L2
    mode_map = {
        'l1': OperatingMode.L1_RESIDENT,
        'l2': OperatingMode.L2_RESIDENT,
        'l3': OperatingMode.L3_RESIDENT,
        'dram': OperatingMode.DRAM_RESIDENT,
    }
    mode = mode_map[args.mode]

    # Show diagrams if requested
    if args.diagram:
        print_architecture_diagrams()

    # Run mode sweep if requested
    if args.mode_sweep:
        run_architecture_mode_sweep(args.ops, args.bytes, args.verbose)
        return

    print("="*80)
    print("  ARCHITECTURE ENERGY COMPARISON")
    print("  CPU (MIMD) vs GPU (SIMT) vs TPU (Systolic) vs KPU (Domain Flow)")
    print("="*80)
    print(f"\n  Workload: {args.ops:,} operations, {args.bytes:,} bytes")
    print(f"  Mode: {get_mode_description(mode)}")
    defaults = DEFAULT_HIT_RATIOS[mode]
    print(f"  Hit ratios: L1={defaults.l1_hit:.0%}, L2={defaults.l2_hit:.0%}")

    # Build cycle energy breakdowns
    cpu_breakdown = build_cpu_cycle_energy(args.ops, args.bytes, mode=mode, verbose=args.verbose)
    gpu_breakdown = build_gpu_cycle_energy(args.ops, args.bytes, mode=mode,
                                            concurrent_threads=args.threads, verbose=args.verbose)
    tpu_breakdown = build_tpu_cycle_energy(args.ops, args.bytes, mode=mode, verbose=args.verbose)
    kpu_breakdown = build_kpu_cycle_energy(args.ops, args.bytes, mode=mode,
                                            num_layers=args.layers, verbose=args.verbose)

    breakdowns = [cpu_breakdown, gpu_breakdown, tpu_breakdown, kpu_breakdown]

    # Print detailed breakdown if verbose
    if args.verbose:
        for breakdown in breakdowns:
            print(format_phase_breakdown(breakdown))

    # Print comparison table
    print(format_architecture_comparison_table(breakdowns, mode=mode, num_ops=args.ops))

    # Print detailed phase comparison
    print(format_detailed_phase_comparison(breakdowns, num_ops=args.ops))

    # Print insights
    print(format_architecture_insights(breakdowns, args.ops))

    # Run sweep if requested
    if args.sweep:
        run_architecture_sweep(mode=mode, verbose=args.verbose)

    # Output JSON if requested
    if args.output:
        import json

        output_data = {
            "workload": {
                "ops": args.ops,
                "bytes": args.bytes,
            },
            "mode": {
                "name": mode.value,
                "description": get_mode_description(mode),
            },
            "architectures": []
        }

        for breakdown in breakdowns:
            arch_data = {
                "name": breakdown.architecture_name,
                "class": breakdown.architecture_class,
                "total_energy_pj": breakdown.total_energy_pj,
                "energy_per_op_pj": breakdown.total_energy_pj / args.ops,
                "phases": {}
            }

            # Add all phases
            for phase in CyclePhase:
                phase_energy = breakdown.get_phase_energy(phase)
                if phase_energy > 0:
                    arch_data["phases"][phase.value] = phase_energy

            output_data["architectures"].append(arch_data)

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Output saved to: {args.output}")


if __name__ == '__main__':
    main()
