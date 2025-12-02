#!/usr/bin/env python3
"""
Stored Program Machine (SPM) Architectural Energy Model Comparison

This tool calculates and compares the energy models for different architectures
by building up a sequence of energy events that reflect each architecture's
operating cycle. The goal is to:

1. Define the basic cycle for each architecture class
2. Accumulate energy events through the cycle
3. Calculate energy per operation for fair comparison

Architecture Classes (Stored Program Machines):
- CPU: MIMD Stored Program Machine (multi-core + SIMD)
- GPU: SIMT Data Parallel (warps of 32 threads lockstep)
- DSP: VLIW with heterogeneous vector/tensor units

All three are "stored program machines" - they execute instructions
from memory, with the key difference being HOW they manage parallelism
and resource contention.

Usage:
    # Compare all stored program architectures
    ./cli/validate_architectural_energy.py

    # Compare with specific workload size
    ./cli/validate_architectural_energy.py --ops 1000 --bytes 4096

    # Show detailed cycle breakdown
    ./cli/validate_architectural_energy.py --verbose

    # Compare at different operation scales
    ./cli/validate_architectural_energy.py --sweep
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

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
    build_dsp_cycle_energy,
    # Comparison utilities
    format_energy,
    format_phase_breakdown,
    format_comparison_table,
    format_key_insights,
    run_sweep,
    run_mode_sweep,
    format_sweep_table,
    format_mode_comparison_table,
    SweepResult,
)
from graphs.hardware.technology_profile import DEFAULT_PROFILE


def get_mode_memory_sizes():
    """Get typical memory sizes for each mode by architecture."""
    return {
        OperatingMode.L1_RESIDENT: {
            "CPU": "32-48 KB (L1 D$)",
            "GPU": "128-228 KB (Shared Mem)",
            "DSP": "256-512 KB (Scratchpad)",
        },
        OperatingMode.L2_RESIDENT: {
            "CPU": "256 KB - 2 MB (L2)",
            "GPU": "4-60 MB (L2)",
            "DSP": "N/A (DMA prefetch)",
        },
        OperatingMode.L3_RESIDENT: {
            "CPU": "8-64 MB (LLC)",
            "GPU": "N/A",
            "DSP": "N/A",
        },
        OperatingMode.DRAM_RESIDENT: {
            "CPU": "DDR4/5 (50-100 GB/s)",
            "GPU": "HBM2/3 (2-5 TB/s)",
            "DSP": "LPDDR4/5 (25-50 GB/s)",
        },
    }


def format_stored_program_comparison_table(
    breakdowns: list,
    mode: Optional[OperatingMode] = None,
    num_ops: int = 1000
) -> str:
    """Format a comparison table specific to stored program machines (CPU/GPU/DSP)."""
    lines = []
    lines.append("\n" + "="*90)
    if mode:
        lines.append(f"  STORED PROGRAM MACHINE ENERGY COMPARISON - {mode.value.upper()} Mode")
    else:
        lines.append("  STORED PROGRAM MACHINE ENERGY COMPARISON")
    lines.append("="*90)

    # Header
    lines.append(f"\n  {'Architecture':<25} {'Total (pJ)':<15} {'Per Cycle':<15} {'Per Op':<15} {'Relative':<12}")
    lines.append(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")

    # Find baseline (CPU) for relative comparison
    baseline_energy = breakdowns[0].total_energy_pj if breakdowns else 1.0

    for breakdown in breakdowns:
        total = breakdown.total_energy_pj
        per_cycle = breakdown.energy_per_cycle_pj
        per_op = total / num_ops
        relative = total / baseline_energy

        lines.append(f"  {breakdown.architecture_name[:25]:<25} "
                    f"{total:>12.2f} pJ "
                    f"{per_cycle:>12.3f} pJ "
                    f"{per_op:>12.4f} pJ "
                    f"{relative:>10.2f}x")

    # Phase breakdown comparison
    lines.append(f"\n  PHASE BREAKDOWN")
    lines.append(f"  {'-'*78}")

    COL_WIDTH = 22

    phases = [
        ("Instruction Fetch",  CyclePhase.INSTRUCTION_FETCH,  0, False),
        ("Instruction Decode", CyclePhase.INSTRUCTION_DECODE, 0, False),
        ("Operand Fetch",      CyclePhase.OPERAND_FETCH,      0, False),
        ("Execute",            CyclePhase.EXECUTE,            0, False),
        ("Writeback",          CyclePhase.WRITEBACK,          0, False),
        ("SIMT Overhead",      "SIMT_TOTAL",                  0, True),
        ("  Fixed Infra",      CyclePhase.SIMT_FIXED_OVERHEAD, 1, False),
        ("  Thread Mgmt",      CyclePhase.SIMT_THREAD_MGMT,   1, False),
        ("  Coherence",        CyclePhase.SIMT_COHERENCE,     1, False),
        ("  Sync",             CyclePhase.SIMT_SYNC,          1, False),
        ("Memory Access",      CyclePhase.MEMORY_ACCESS,      0, True),
        ("  L1/Scratchpad",    CyclePhase.MEM_L1,             1, False),
        ("  L2",               CyclePhase.MEM_L2,             1, False),
        ("  L3",               CyclePhase.MEM_L3,             1, False),
        ("  DRAM/HBM",         CyclePhase.MEM_DRAM,           1, False),
    ]

    # Header row
    header = f"  {'Phase':<22}"
    for b in breakdowns:
        arch_name = b.architecture_name.split()[0]
        header += f" {arch_name:>{COL_WIDTH}}"
    lines.append(header)

    # Separator
    sep = f"  {'-'*22}"
    for _ in breakdowns:
        sep += f" {'-'*COL_WIDTH}"
    lines.append(sep)

    def format_energy_cell(energy_pj: float, total_pj: float) -> str:
        if energy_pj == 0:
            return "n/a"
        pct = (energy_pj / total_pj * 100) if total_pj > 0 else 0
        return f"{energy_pj:,.0f} pJ ({pct:4.1f}%)"

    for phase_name, phase, indent, is_parent in phases:
        row = f"  {phase_name:<22}"

        for breakdown in breakdowns:
            if phase == "SIMT_TOTAL":
                energy = (
                    breakdown.get_phase_energy(CyclePhase.SIMT_FIXED_OVERHEAD) +
                    breakdown.get_phase_energy(CyclePhase.SIMT_THREAD_MGMT) +
                    breakdown.get_phase_energy(CyclePhase.SIMT_COHERENCE) +
                    breakdown.get_phase_energy(CyclePhase.SIMT_SYNC)
                )
            elif isinstance(phase, CyclePhase):
                energy = breakdown.get_phase_energy(phase)
            else:
                energy = 0

            cell = format_energy_cell(energy, breakdown.total_energy_pj)
            row += f" {cell:>{COL_WIDTH}}"

        lines.append(row)

    return "\n".join(lines)


def format_stored_program_insights(breakdowns: list) -> str:
    """Format key insights specific to stored program machines."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("  KEY INSIGHTS")
    lines.append("="*70)

    if len(breakdowns) >= 3:
        cpu, gpu, dsp = breakdowns[0], breakdowns[1], breakdowns[2]

        simt_fixed = gpu.get_phase_energy(CyclePhase.SIMT_FIXED_OVERHEAD)
        simt_thread_mgmt = gpu.get_phase_energy(CyclePhase.SIMT_THREAD_MGMT)
        simt_coherence = gpu.get_phase_energy(CyclePhase.SIMT_COHERENCE)
        simt_sync = gpu.get_phase_energy(CyclePhase.SIMT_SYNC)
        simt_total = simt_fixed + simt_thread_mgmt + simt_coherence + simt_sync
        simt_pct = (simt_total / gpu.total_energy_pj * 100) if gpu.total_energy_pj > 0 else 0

        lines.append(f"""
  1. CPU (Stored Program Machine):
     - High instruction fetch/decode overhead per operation
     - Cache hierarchy adds significant memory access energy
     - Register file energy is comparable to ALU energy
     - Good for: General-purpose computing, irregular workloads

  2. GPU (SIMT Data Parallel):
     - Instruction overhead amortized across 32 threads (warp)
     - BUT: SIMT overhead is THE dominant energy cost ({simt_pct:.1f}% of total!)

     GPU SIMT OVERHEAD BREAKDOWN:
       Fixed Infra:       {simt_fixed:>12,.1f} pJ (kernel launch, SM activation, mem ctrl)
       Thread Management: {simt_thread_mgmt:>12,.1f} pJ (warp scheduling, scoreboard)
       Coherence:         {simt_coherence:>12,.1f} pJ (queuing, coalescing, tags, directory)
       Synchronization:   {simt_sync:>12,.1f} pJ (divergence, barriers, atomics)
       ---------------------------------------------------------
       TOTAL SIMT:        {simt_total:>12,.1f} pJ

     Compare to CPU writeback: {cpu.get_phase_energy(CyclePhase.WRITEBACK):.1f} pJ
     Good for: Large batch sizes where SIMT overhead is amortized

  3. DSP (VLIW):
     - VLIW bundles amortize instruction fetch across 4 ops
     - No dynamic scheduling = simpler, lower energy
     - Scratchpad memories eliminate cache tag overhead
     - Good for: Signal processing, known data access patterns

  FUNDAMENTAL INSIGHT:
  All three are stored program machines, but they trade off flexibility
  vs. efficiency differently:

    CPU: Maximum flexibility, highest overhead per op
    GPU: Data parallelism amortizes instruction cost, but SIMT overhead dominates
    DSP: Static scheduling + scratchpads = lowest overhead, least flexible
""")

    return "\n".join(lines)


def print_architecture_diagrams() -> None:
    """Print ASCII art diagrams showing the basic cycle for each architecture."""
    print("""
================================================================================
  STORED PROGRAM MACHINE BASIC CYCLES
================================================================================

  All three architectures are "stored program machines" that execute
  instructions from memory. The key difference is HOW they manage
  parallelism and resource contention.

--------------------------------------------------------------------------------
  CPU Basic Cycle (MIMD Stored Program Machine)
--------------------------------------------------------------------------------

  +-------------------+     +------------------+     +------------------+
  |  INSTRUCTION      |     |  INSTRUCTION     |     |  DISPATCH        |
  |  FETCH            |---->|  DECODE          |---->|  (Control Sigs)  |
  |  (I-cache read)   |     |  (x86-64 logic)  |     |                  |
  |  ~1.5 pJ          |     |  ~0.8 pJ         |     |  ~0.5 pJ         |
  +-------------------+     +------------------+     +------------------+
                                                              |
                                                              v
  +-------------------+     +------------------+     +------------------+
  |  WRITEBACK        |     |  EXECUTE         |     |  OPERAND FETCH   |
  |  (Register Write) |<----|  (ALU/FPU)       |<----|  (Register Read) |
  |  ~3.0 pJ          |     |  ~4.0 pJ         |     |  ~2.5 pJ x 2     |
  +-------------------+     +------------------+     +------------------+
          |
          v
  +-----------------------------------------------+
  |  MEMORY ACCESS (Cache Hierarchy)              |
  |  L1 (1.0 pJ/B) -> L2 (2.5 pJ/B) -> L3 (5 pJ/B)|
  |                 -> DRAM (20 pJ/B)              |
  +-----------------------------------------------+

  KEY: ~15-20 pJ per cycle (dominated by memory hierarchy)

--------------------------------------------------------------------------------
  GPU Basic Cycle (SIMT Data Parallel)
--------------------------------------------------------------------------------

  +-------------------+     +------------------+     +------------------+
  |  INSTRUCTION      |     |  INSTRUCTION     |     |  WARP            |
  |  FETCH            |---->|  DECODE          |---->|  SCHEDULING      |
  |  (per-warp)       |     |  (SIMT logic)    |     |  (~1 pJ/thread)  |
  |  ~2.0 pJ          |     |  ~0.5 pJ         |     |  HUGE @ 200K!    |
  +-------------------+     +------------------+     +------------------+
                                                              |
                                                              v
  +-------------------+     +------------------+     +------------------+
  |  COHERENCE        |     |  EXECUTE         |     |  REGISTER FILE   |
  |  MACHINERY        |<----|  (CUDA/Tensor)   |<----|  ACCESS          |
  |  ~5 pJ/request    |     |  ~0.3-0.8 pJ     |     |  ~0.6 pJ         |
  |  *** DOMINANT *** |     |                  |     |  (256KB/SM)      |
  +-------------------+     +------------------+     +------------------+
          |
          v
  +-----------------------------------------------+
  |  MEMORY ACCESS (GPU Hierarchy)                |
  |  Shared/L1 (0.25 pJ/B) -> L2 (0.8 pJ/B)       |
  |                       -> HBM (10 pJ/B)        |
  +-----------------------------------------------+

  KEY: Coherence machinery dominates at small batch sizes!
       GPU burns massive energy managing concurrent memory requests.

--------------------------------------------------------------------------------
  DSP Basic Cycle (VLIW Stored Program Machine)
--------------------------------------------------------------------------------

  +-------------------+     +------------------+     +------------------+
  |  VLIW INSTRUCTION |     |  PARALLEL        |     |  OPERAND FETCH   |
  |  FETCH            |---->|  DECODE          |---->|  (Multi-port RF) |
  |  (256-512 bit)    |     |  (4 slots)       |     |                  |
  |  ~2.5 pJ          |     |  ~0.5 pJ         |     |  ~1.5 pJ x 2     |
  +-------------------+     +------------------+     +------------------+
                                                              |
                                                              v
                            +----------------------------------+
                            |  PARALLEL EXECUTE (4 slots)      |
                            |  +--------+ +--------+ +-------+ |
                            |  | Tensor | | Vector | | Scalar| |
                            |  | 0.4 pJ | | 0.8 pJ | | 2.0 pJ| |
                            |  +--------+ +--------+ +-------+ |
                            +----------------------------------+
                                          |
                                          v
  +-----------------------------------------------+
  |  MEMORY ACCESS (Scratchpad - No Cache!)       |
  |  Scratchpad SRAM (~0.8 pJ/B) -> DRAM (~15 pJ/B)|
  |  No tag overhead, software-managed            |
  +-----------------------------------------------+

  KEY: VLIW + scratchpad = lowest overhead, but least flexible

================================================================================
  WHY THESE DIFFERENCES MATTER
================================================================================

  1. INSTRUCTION OVERHEAD:
     CPU: 1 instruction per ~0.5 ops (high decode complexity)
     GPU: 1 instruction per 32 threads (warp), but coherence dominates
     DSP: 1 instruction per 4 ops (VLIW parallelism)

  2. MEMORY ACCESS:
     CPU: Hardware caches (tag lookup + coherence)
     GPU: Massive coherence machinery (thousands of concurrent requests)
     DSP: Software scratchpads (no tag overhead, explicit DMA)

  3. SCHEDULING:
     CPU: Out-of-order dynamic scheduling (complex, energy hungry)
     GPU: Hardware thread scheduling + warp divergence penalties
     DSP: Compiler static scheduling (simple, energy efficient)

================================================================================
""")


def run_stored_program_sweep(mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
                             verbose: bool = False) -> None:
    """Run a sweep across different operation scales for stored program machines."""
    print("\n" + "="*100)
    print(f"  ENERGY SCALING ANALYSIS (Stored Program Machines) - {mode.value.upper()} Mode")
    print("="*100)

    scales = [100, 1000, 10000, 100000, 1000000]
    bytes_per_op = 4

    results = []
    for ops in scales:
        bytes_transferred = ops * bytes_per_op

        cpu = build_cpu_cycle_energy(ops, bytes_transferred, mode=mode, tech_profile=DEFAULT_PROFILE)
        gpu = build_gpu_cycle_energy(ops, bytes_transferred, mode=mode, tech_profile=DEFAULT_PROFILE)
        dsp = build_dsp_cycle_energy(ops, bytes_transferred, mode=mode, tech_profile=DEFAULT_PROFILE)

        results.append({
            'ops': ops,
            'cpu_total': cpu.total_energy_pj,
            'gpu_total': gpu.total_energy_pj,
            'dsp_total': dsp.total_energy_pj,
            'cpu_per_op': cpu.total_energy_pj / ops,
            'gpu_per_op': gpu.total_energy_pj / ops,
            'dsp_per_op': dsp.total_energy_pj / ops,
        })

    # TABLE 1: AMORTIZED ENERGY PER OPERATION
    print(f"\n  TABLE 1: AMORTIZED ENERGY PER OPERATION")
    print(f"  {'-'*90}")
    print(f"  {'Operations':<12} {'CPU (pJ/op)':<15} {'GPU (pJ/op)':<15} {'DSP (pJ/op)':<15} {'Best':<12}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")

    for r in results:
        energies = {'CPU': r['cpu_per_op'], 'GPU': r['gpu_per_op'], 'DSP': r['dsp_per_op']}
        best = min(energies, key=energies.get)

        print(f"  {r['ops']:<12,} {r['cpu_per_op']:>12.2f} pJ "
              f"{r['gpu_per_op']:>12.2f} pJ "
              f"{r['dsp_per_op']:>12.2f} pJ "
              f"{best:>10}")

    # TABLE 2: TOTAL ENERGY
    print(f"\n  TABLE 2: TOTAL ENERGY")
    print(f"  {'-'*90}")
    print(f"  {'Operations':<12} {'CPU (pJ)':<15} {'GPU (pJ)':<15} {'DSP (pJ)':<15} {'GPU/CPU':<10} {'DSP/CPU':<10}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*10} {'-'*10}")

    for r in results:
        gpu_ratio = r['gpu_total'] / r['cpu_total']
        dsp_ratio = r['dsp_total'] / r['cpu_total']

        print(f"  {r['ops']:<12,} {r['cpu_total']:>12,.0f} pJ "
              f"{r['gpu_total']:>12,.0f} pJ "
              f"{r['dsp_total']:>12,.0f} pJ "
              f"{gpu_ratio:>8.2f}x "
              f"{dsp_ratio:>8.2f}x")

    print(f"""

  OBSERVATION:
  - GPU energy/op is HIGH at small scales due to fixed SIMT overhead
    (warp schedulers, coherence machinery, memory controllers run regardless)
  - GPU energy/op improves at large scales as overhead amortizes across more ops
  - DSP maintains consistent low energy/op (VLIW + scratchpad = minimal overhead)
  - CPU has moderate, consistent energy/op across all scales

  KEY INSIGHT: GPU is INEFFICIENT for small workloads!
  The massive parallel machinery (132 SMs on H100, 16K+ CUDA cores) consumes
  energy even when only a few operations are needed. This is the "GPU tax"
  for data parallelism - you pay for the infrastructure whether you use it or not.
""")


def run_stored_program_mode_sweep(num_ops: int = 1000, bytes_transferred: int = 4096,
                                   verbose: bool = False) -> None:
    """Run a sweep across all operating modes for stored program machines."""
    print("\n" + "="*100)
    print("  OPERATING MODE COMPARISON")
    print("  Comparing energy across L1-Resident, L2-Resident, L3-Resident, and DRAM-Resident modes")
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
    print(f"  {'Mode':<20} {'CPU (pJ)':<18} {'GPU (pJ)':<18} {'DSP (pJ)':<18} {'Notes':<30}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18} {'-'*30}")

    results = []

    for mode in modes:
        cpu = build_cpu_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=DEFAULT_PROFILE)
        gpu = build_gpu_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=DEFAULT_PROFILE)
        dsp = build_dsp_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=DEFAULT_PROFILE)

        notes = ""
        if mode == OperatingMode.L1_RESIDENT:
            notes = "Best case: all data on-chip"
        elif mode == OperatingMode.L2_RESIDENT:
            notes = "GPU coherence active"
        elif mode == OperatingMode.L3_RESIDENT:
            notes = "CPU only (GPU/DSP->DRAM)"
        else:
            notes = "Streaming from memory"

        cpu_str = f"{cpu.total_energy_pj:>14,.0f} pJ"
        gpu_str = f"{gpu.total_energy_pj:>14,.0f} pJ"
        dsp_str = f"{dsp.total_energy_pj:>14,.0f} pJ"

        if mode == OperatingMode.L3_RESIDENT:
            gpu_str = f"{gpu.total_energy_pj:>14,.0f} pJ*"
            dsp_str = f"{dsp.total_energy_pj:>14,.0f} pJ*"

        print(f"  {get_mode_description(mode):<20} {cpu_str:<18} {gpu_str:<18} {dsp_str:<18} {notes:<30}")

        results.append({
            'mode': mode,
            'cpu': cpu.total_energy_pj,
            'gpu': gpu.total_energy_pj,
            'dsp': dsp.total_energy_pj,
        })

    print()
    print("  * GPU and DSP have no L3 cache - L3 mode uses same hit ratios but data goes to DRAM")
    print()

    print("  WINNER BY MODE:")
    print(f"  {'-'*60}")
    for r in results:
        energies = {'CPU': r['cpu'], 'GPU': r['gpu'], 'DSP': r['dsp']}
        winner = min(energies, key=energies.get)
        winner_energy = energies[winner]
        runner_up = sorted(energies.values())[1]
        ratio = runner_up / winner_energy if winner_energy > 0 else 0

        print(f"  {get_mode_description(r['mode']):<30} -> {winner} ({ratio:.1f}x more efficient)")

    print()
    print(f"""
  KEY INSIGHTS:
  1. L1-Resident Mode: This is where GPU shared memory shines - minimal coherence
  2. L2-Resident Mode: GPU coherence overhead becomes significant
  3. L3-Resident Mode: Only CPU has L3 cache - advantage to CPU
  4. DRAM-Resident Mode: GPU's HBM bandwidth helps, but coherence still hurts

  RECOMMENDATION:
  - Small kernels (L1-fitting): Consider GPU shared memory approach
  - Medium kernels (L2-fitting): CPU may be competitive due to lower coherence
  - Large kernels (L3-fitting): CPU has clear advantage (only one with L3)
  - Streaming (DRAM): Depends on memory bandwidth requirements
""")


def main():
    parser = argparse.ArgumentParser(
        description='Validate and compare architectural energy models for stored program machines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (DRAM-resident mode, default)
  %(prog)s

  # Compare in L1-resident mode (all data on-chip)
  %(prog)s --mode l1

  # Compare in L2-resident mode (with cache hit ratios)
  %(prog)s --mode l2

  # Custom workload size
  %(prog)s --ops 10000 --bytes 40960

  # Show architecture diagrams
  %(prog)s --diagram

  # Detailed breakdown
  %(prog)s --verbose

  # Operation scaling sweep
  %(prog)s --sweep

  # Mode comparison (all modes side-by-side)
  %(prog)s --mode-sweep

  # Custom L1 hit ratio
  %(prog)s --mode l2 --l1-hit-rate 0.92

  # JSON output
  %(prog)s --output results.json
"""
    )

    parser.add_argument('--ops', type=int, default=1000,
                        help='Number of operations (default: 1000)')
    parser.add_argument('--bytes', type=int, default=4096,
                        help='Bytes transferred (default: 4096)')
    parser.add_argument('--tensor-core-util', type=float, default=0.8,
                        help='GPU tensor core utilization 0.0-1.0 (default: 0.8)')
    parser.add_argument('--mode', type=str, default='dram',
                        choices=['l1', 'l2', 'l3', 'dram'],
                        help='Operating mode (default: dram)')
    parser.add_argument('--l1-hit-rate', type=float,
                        help='Custom L1 hit rate (0.0-1.0)')
    parser.add_argument('--l2-hit-rate', type=float,
                        help='Custom L2 hit rate (0.0-1.0)')
    parser.add_argument('--l3-hit-rate', type=float,
                        help='Custom L3 hit rate (0.0-1.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed phase breakdown')
    parser.add_argument('--diagram', '-d', action='store_true',
                        help='Show architecture cycle diagrams')
    parser.add_argument('--sweep', action='store_true',
                        help='Run scaling sweep across operation counts')
    parser.add_argument('--mode-sweep', action='store_true',
                        help='Compare energy across all operating modes')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (JSON format)')

    args = parser.parse_args()

    # Parse operating mode
    mode_map = {
        'l1': OperatingMode.L1_RESIDENT,
        'l2': OperatingMode.L2_RESIDENT,
        'l3': OperatingMode.L3_RESIDENT,
        'dram': OperatingMode.DRAM_RESIDENT,
    }
    mode = mode_map[args.mode]

    # Build custom hit ratios if specified
    hit_ratios = None
    if args.l1_hit_rate is not None or args.l2_hit_rate is not None or args.l3_hit_rate is not None:
        defaults = DEFAULT_HIT_RATIOS[mode]
        hit_ratios = HitRatios(
            l1_hit=args.l1_hit_rate if args.l1_hit_rate is not None else defaults.l1_hit,
            l2_hit=args.l2_hit_rate if args.l2_hit_rate is not None else defaults.l2_hit,
            l3_hit=args.l3_hit_rate if args.l3_hit_rate is not None else defaults.l3_hit,
        )

    # Show diagrams if requested
    if args.diagram:
        print_architecture_diagrams()

    # Run mode sweep if requested
    if args.mode_sweep:
        run_stored_program_mode_sweep(args.ops, args.bytes, args.verbose)
        return

    print("="*70)
    print("  ARCHITECTURAL ENERGY MODEL VALIDATION")
    print("  Stored Program Machines: CPU, GPU, DSP (VLIW)")
    print("="*70)
    print(f"\n  Workload: {args.ops:,} operations, {args.bytes:,} bytes")
    print(f"  Mode: {get_mode_description(mode)}")
    if hit_ratios:
        print(f"  Custom hit ratios: L1={hit_ratios.l1_hit:.0%}, L2={hit_ratios.l2_hit:.0%}, L3={hit_ratios.l3_hit:.0%}")
    else:
        defaults = DEFAULT_HIT_RATIOS[mode]
        print(f"  Default hit ratios: L1={defaults.l1_hit:.0%}, L2={defaults.l2_hit:.0%}, L3={defaults.l3_hit:.0%}")

    # Build cycle energy breakdowns using shared code
    cpu_breakdown = build_cpu_cycle_energy(args.ops, args.bytes, mode=mode, hit_ratios=hit_ratios,
                                            tech_profile=DEFAULT_PROFILE, verbose=args.verbose)
    gpu_breakdown = build_gpu_cycle_energy(args.ops, args.bytes, mode=mode, hit_ratios=hit_ratios,
                                            tensor_core_utilization=args.tensor_core_util,
                                            tech_profile=DEFAULT_PROFILE, verbose=args.verbose)
    dsp_breakdown = build_dsp_cycle_energy(args.ops, args.bytes, mode=mode, hit_ratios=hit_ratios,
                                            tech_profile=DEFAULT_PROFILE, verbose=args.verbose)

    breakdowns = [cpu_breakdown, gpu_breakdown, dsp_breakdown]

    # Print detailed breakdown if verbose
    if args.verbose:
        for breakdown in breakdowns:
            print(format_phase_breakdown(breakdown))

    # Print comparison table
    print(format_stored_program_comparison_table(breakdowns, mode=mode, num_ops=args.ops))

    # Print insights
    print(format_stored_program_insights(breakdowns))

    # Run sweep if requested
    if args.sweep:
        run_stored_program_sweep(mode=mode, verbose=args.verbose)

    # Output JSON if requested
    if args.output:
        import json

        effective_ratios = hit_ratios if hit_ratios else DEFAULT_HIT_RATIOS[mode]

        output_data = {
            "workload": {
                "ops": args.ops,
                "bytes": args.bytes,
            },
            "mode": {
                "name": mode.value,
                "description": get_mode_description(mode),
                "hit_ratios": {
                    "l1_hit": effective_ratios.l1_hit,
                    "l2_hit": effective_ratios.l2_hit,
                    "l3_hit": effective_ratios.l3_hit,
                }
            },
            "architectures": []
        }

        for breakdown in breakdowns:
            arch_data = {
                "name": breakdown.architecture_name,
                "class": breakdown.architecture_class,
                "total_energy_pj": breakdown.total_energy_pj,
                "energy_per_cycle_pj": breakdown.energy_per_cycle_pj,
                "energy_per_op_pj": breakdown.total_energy_pj / args.ops,
                "phases": {}
            }

            for phase in CyclePhase:
                phase_energy = breakdown.get_phase_energy(phase)
                arch_data["phases"][phase.value] = phase_energy

            output_data["architectures"].append(arch_data)

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Output saved to: {args.output}")


if __name__ == '__main__':
    main()
