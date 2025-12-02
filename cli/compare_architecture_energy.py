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
    OperatorType,
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
    # Consistent-scale formatting
    determine_common_scale,
    format_energy_with_scale,
    # 3-category energy breakdown
    format_energy_categories_table,
    format_energy_categories_per_op_table,
)
from graphs.hardware.technology_profile import (
    DEFAULT_PROFILE,
    TechnologyProfile,
    ArchitectureComparisonSet,
    get_architecture_comparison_set,
    create_architecture_comparison_set,
    ARCHITECTURE_COMPARISON_SETS,
    CIRCUIT_TYPE_MULTIPLIER,
    MemoryType,
    # Legacy aliases
    ProductCategoryProfiles,
    get_product_category,
    PRODUCT_CATEGORIES,
)


def format_comparison_set_header(comp_set: ArchitectureComparisonSet) -> str:
    """Format header showing architecture comparison set details."""
    cpu_mult = CIRCUIT_TYPE_MULTIPLIER.get(comp_set.cpu_circuit_type, 1.0)
    lines = []
    lines.append("=" * 100)
    lines.append(f"  ARCHITECTURE COMPARISON: {comp_set.name}")
    lines.append("=" * 100)
    lines.append(f"  {comp_set.description}")
    lines.append("")
    lines.append(f"  Process Node: {comp_set.process_node_nm}nm")
    lines.append(f"  Memory Type:  {comp_set.memory_type.value}")
    lines.append(f"  Base ALU:     {comp_set.base_alu_energy_pj:.2f} pJ (standard cell @ {comp_set.process_node_nm}nm)")
    lines.append("")
    lines.append("  Circuit Type Multipliers (relative to base ALU):")
    lines.append(f"    CPU ({comp_set.cpu_circuit_type}): {cpu_mult:.2f}x -> {comp_set.cpu_profile.base_alu_energy_pj:.2f} pJ/op")
    lines.append(f"    GPU (tensor_core):    {CIRCUIT_TYPE_MULTIPLIER['tensor_core']:.2f}x -> {comp_set.gpu_profile.tensor_core_mac_energy_pj:.2f} pJ/op")
    lines.append(f"    TPU (systolic_mac):   {CIRCUIT_TYPE_MULTIPLIER['systolic_mac']:.2f}x -> {comp_set.tpu_profile.systolic_mac_energy_pj:.2f} pJ/op")
    lines.append(f"    KPU (domain_flow):    {CIRCUIT_TYPE_MULTIPLIER['domain_flow']:.2f}x -> {comp_set.kpu_profile.domain_flow_mac_energy_pj:.2f} pJ/op")
    return "\n".join(lines)


# Legacy alias
def format_product_category_header(category: ProductCategoryProfiles) -> str:
    """Legacy alias for format_comparison_set_header."""
    return format_comparison_set_header(category)


def format_architecture_comparison_table(
    breakdowns: List[CycleEnergyBreakdown],
    mode: Optional[OperatingMode] = None,
    num_ops: int = 1000
) -> str:
    """Format a comparison table for CPU/GPU/TPU/KPU with consistent units."""
    lines = []
    lines.append("\n" + "="*100)
    if mode:
        lines.append(f"  ARCHITECTURE ENERGY COMPARISON - {mode.value.upper()} Mode")
    else:
        lines.append("  ARCHITECTURE ENERGY COMPARISON")
    lines.append("="*100)

    # Find baseline (CPU) for relative comparison
    baseline_energy = breakdowns[0].total_energy_pj if breakdowns else 1.0

    # Collect all energy values to determine common scale
    total_energies = [b.total_energy_pj for b in breakdowns]
    per_op_energies = [b.total_energy_pj / num_ops for b in breakdowns]

    # Determine common scale for total energy column
    total_unit, total_divisor = determine_common_scale(total_energies)

    # Determine common scale for per-op energy column
    per_op_unit, per_op_divisor = determine_common_scale(per_op_energies)

    # Header with consistent units
    lines.append(f"\n  {'Architecture':<35} {'Total (' + total_unit + ')':<15} {'Per Op (' + per_op_unit + ')':<15} {'Relative':<12}")
    lines.append(f"  {'-'*35} {'-'*15} {'-'*15} {'-'*12}")

    for breakdown in breakdowns:
        total = breakdown.total_energy_pj
        per_op = total / num_ops
        relative = total / baseline_energy

        # Format with consistent scale (no unit suffix, just the number)
        total_str = format_energy_with_scale(total, total_divisor)
        per_op_str = format_energy_with_scale(per_op, per_op_divisor)

        lines.append(f"  {breakdown.architecture_name[:35]:<35} "
                    f"{total_str:>12} "
                    f"{per_op_str:>12} "
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
        ("KPU EDDO SCRATCHPADS", [
            ("Tile Scratchpad", CyclePhase.EDDO_TILE_SCRATCHPAD),
            ("Global Scratchpad", CyclePhase.EDDO_GLOBAL_SCRATCHPAD),
            ("Streaming Buffer", CyclePhase.EDDO_STREAMING_BUFFER),
            ("DMA Setup", CyclePhase.EDDO_DMA_SETUP),
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

    # Total row with consistent units across all architectures
    lines.append(f"\n  {'-'*25}" + (" " + "-"*COL_WIDTH) * len(breakdowns))

    # Determine common scale for totals
    totals = [bd.total_energy_pj for bd in breakdowns]
    total_unit, total_divisor = determine_common_scale(totals)

    row = f"  {'TOTAL (' + total_unit + ')':<25}"
    for bd in breakdowns:
        total_str = format_energy_with_scale(bd.total_energy_pj, total_divisor)
        row += f" {total_str:>{COL_WIDTH}}"
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
                           verbose: bool = False,
                           category: ArchitectureComparisonSet = None,
                           operator_type: OperatorType = OperatorType.HIGH_REUSE,
                           working_set_mb: float = 20.0) -> None:
    """Run a sweep across different operation scales.

    Args:
        mode: Operating mode (L1, L2, L3, or DRAM resident)
        verbose: Enable verbose output
        category: Architecture comparison set (profiles for each arch)
        operator_type: Type of operator (HIGH_REUSE, LOW_REUSE, STREAMING)
        working_set_mb: Working set size in MB for cache hit ratio calculation
    """
    # Get profiles from category or use defaults
    if category:
        cpu_profile = category.cpu_profile
        gpu_profile = category.gpu_profile
        tpu_profile = category.tpu_profile
        kpu_profile = category.kpu_profile
        # Support both old ProductCategoryProfiles (.category) and new ArchitectureComparisonSet (.name)
        cat_name = getattr(category, 'name', getattr(category, 'category', 'CUSTOM')).upper()
    else:
        cpu_profile = gpu_profile = tpu_profile = kpu_profile = DEFAULT_PROFILE
        cat_name = "DEFAULT"

    working_set_bytes = int(working_set_mb * 1024 * 1024)
    op_type_desc = {
        OperatorType.HIGH_REUSE: "HIGH_REUSE (MatMul/Conv)",
        OperatorType.LOW_REUSE: "LOW_REUSE (MatVec/Pool)",
        OperatorType.STREAMING: "STREAMING (ReLU/BatchNorm)",
    }

    print("\n" + "="*100)
    print(f"  ENERGY SCALING ANALYSIS ({cat_name}) - {mode.value.upper()} Mode")
    print("="*100)
    print(f"  Operator Type: {op_type_desc.get(operator_type, operator_type.value)}")
    print(f"  Working Set: {working_set_mb:.0f} MB")
    print()

    scales = [100, 1000, 10000, 100000, 1000000]
    bytes_per_op = 4

    # For streaming operators, cache is "cold" (flushed by previous streaming op)
    cache_is_cold = (operator_type == OperatorType.STREAMING)

    results = []
    for ops in scales:
        bytes_transferred = ops * bytes_per_op

        cpu = build_cpu_cycle_energy(
            ops, bytes_transferred, mode=mode, tech_profile=cpu_profile,
            operator_type=operator_type, working_set_bytes=working_set_bytes,
            l3_is_cold=cache_is_cold)
        gpu = build_gpu_cycle_energy(
            ops, bytes_transferred, mode=mode, tech_profile=gpu_profile,
            operator_type=operator_type, working_set_bytes=working_set_bytes,
            l2_is_cold=cache_is_cold)
        tpu = build_tpu_cycle_energy(
            ops, bytes_transferred, mode=mode, tech_profile=tpu_profile,
            operator_type=operator_type, working_set_bytes=working_set_bytes,
            sram_is_cold=cache_is_cold)
        kpu = build_kpu_cycle_energy(
            ops, bytes_transferred, mode=mode, tech_profile=kpu_profile,
            operator_type=operator_type, working_set_bytes=working_set_bytes)

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
    print(f"\n  TABLE 1: AMORTIZED ENERGY PER OPERATION ({operator_type.value})")
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

    # Print operator-specific observations
    if operator_type == OperatorType.HIGH_REUSE:
        print(f"""

  OBSERVATIONS (HIGH_REUSE - MatMul/Conv operators):
  --------------------------------------------------
  High data reuse enables efficient caching. Results assume warm caches.

  1. TPU excels at large MatMul due to systolic array efficiency
     - Weight-stationary dataflow maximizes weight reuse
     - Very efficient for batch > 1

  2. GPU benefits from high L2 hit ratio (~95% with good tiling)
     - Tensor cores provide efficient MACs
     - Energy/op stabilizes once hardware is saturated

  3. KPU provides consistent efficiency via explicit memory (EDDO)
     - Compiler pre-stages data, no cache misses
     - Near 100% utilization even at batch=1

  4. CPU has highest control overhead (instruction fetch/decode per op)
""")
    elif operator_type == OperatorType.STREAMING:
        print(f"""

  OBSERVATIONS (STREAMING - ReLU/BatchNorm/Softmax operators):
  ------------------------------------------------------------
  Each element accessed once - no data reuse. Flushes implicit caches.

  1. KPU wins for streaming ops due to explicit memory (EDDO)
     - No cache flush penalty - compiler manages data placement
     - Same energy regardless of operator sequence

  2. TPU suffers because SRAM has low hit ratio (~5%)
     - Streaming ops don't benefit from weight-stationary dataflow
     - Data flows through without reuse

  3. GPU L2 provides minimal benefit (~5% hit ratio)
     - Static power becomes dominant overhead

  4. CPU L3 similarly ineffective for streaming access patterns

  KEY INSIGHT: Streaming operators between MatMuls flush caches,
  causing "cold start" penalties for subsequent high-reuse operators.
  KPU's explicit memory model avoids this problem entirely.
""")
    else:  # LOW_REUSE
        print(f"""

  OBSERVATIONS (LOW_REUSE - MatVec/Pooling operators):
  ----------------------------------------------------
  Limited data reuse. Moderate cache benefit when data fits.

  1. Cache hit ratios depend on working set vs cache size
     - ~60% hit if working set < 50% of cache
     - ~10% hit if working set > cache size

  2. KPU maintains consistent efficiency via explicit memory
     - Binary behavior: data fits (100%) or spills (0%)
     - No partial cache benefit, but predictable performance
""")


def run_architecture_mode_sweep(num_ops: int = 10000, bytes_transferred: int = 40960,
                                 verbose: bool = False,
                                 category: ArchitectureComparisonSet = None) -> None:
    """Run a sweep across operating modes for all architectures."""
    # Get profiles from category or use defaults
    if category:
        cpu_profile = category.cpu_profile
        gpu_profile = category.gpu_profile
        tpu_profile = category.tpu_profile
        kpu_profile = category.kpu_profile
        # Support both old ProductCategoryProfiles (.category) and new ArchitectureComparisonSet (.name)
        cat_name = getattr(category, 'name', getattr(category, 'category', 'CUSTOM')).upper()
    else:
        cpu_profile = gpu_profile = tpu_profile = kpu_profile = DEFAULT_PROFILE
        cat_name = "All Architectures"

    print("\n" + "="*100)
    print(f"  OPERATING MODE COMPARISON ({cat_name})")
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
        cpu = build_cpu_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=cpu_profile)
        gpu = build_gpu_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=gpu_profile)
        tpu = build_tpu_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=tpu_profile)
        kpu = build_kpu_cycle_energy(num_ops, bytes_transferred, mode=mode, tech_profile=kpu_profile)

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


def run_comparison_sweep(num_ops: int = 10000, bytes_transferred: int = 40960,
                         mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
                         verbose: bool = False) -> None:
    """Compare energy across all predefined architecture comparison sets."""
    print("\n" + "="*100)
    print("  ARCHITECTURE COMPARISON SWEEP")
    print("  Same workload, different process nodes and CPU types")
    print("="*100)
    print(f"  Workload: {num_ops:,} ops, {bytes_transferred:,} bytes")
    print(f"  Mode: {mode.value}")
    print()

    # Collect results for each comparison set
    results = {}
    for set_name, comp_set in ARCHITECTURE_COMPARISON_SETS.items():
        cpu = build_cpu_cycle_energy(num_ops, bytes_transferred, mode=mode,
                                     tech_profile=comp_set.cpu_profile)
        gpu = build_gpu_cycle_energy(num_ops, bytes_transferred, mode=mode,
                                     tech_profile=comp_set.gpu_profile)
        tpu = build_tpu_cycle_energy(num_ops, bytes_transferred, mode=mode,
                                     tech_profile=comp_set.tpu_profile)
        kpu = build_kpu_cycle_energy(num_ops, bytes_transferred, mode=mode,
                                     tech_profile=comp_set.kpu_profile)

        results[set_name] = {
            'comp_set': comp_set,
            'cpu': cpu.total_energy_pj,
            'gpu': gpu.total_energy_pj,
            'tpu': tpu.total_energy_pj,
            'kpu': kpu.total_energy_pj,
        }

    # Print table
    print(f"  {'Set':<18} {'Node':<6} {'CPU Type':<18} {'CPU':<14} {'GPU':<14} {'TPU':<14} {'KPU':<14} {'Best':<6}")
    print(f"  {'-'*18} {'-'*6} {'-'*18} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*6}")

    for set_name, data in results.items():
        cs = data['comp_set']
        energies = {'CPU': data['cpu'], 'GPU': data['gpu'], 'TPU': data['tpu'], 'KPU': data['kpu']}
        best = min(energies, key=energies.get)

        print(f"  {set_name:<18} {cs.process_node_nm}nm    {cs.cpu_circuit_type:<18} "
              f"{format_energy(data['cpu']):>12} "
              f"{format_energy(data['gpu']):>12} "
              f"{format_energy(data['tpu']):>12} "
              f"{format_energy(data['kpu']):>12} "
              f"{best:>4}")

    # Summary table: Circuit multipliers
    print()
    print("  CIRCUIT TYPE MULTIPLIERS (relative to base ALU energy)")
    print(f"  {'-'*90}")
    print(f"  {'Set':<18} {'Base ALU':<10} {'CPU Mult':<10} {'GPU':<10} {'TPU':<10} {'KPU':<10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for set_name, data in results.items():
        cs = data['comp_set']
        cpu_mult = CIRCUIT_TYPE_MULTIPLIER.get(cs.cpu_circuit_type, 1.0)
        print(f"  {set_name:<18} "
              f"{cs.base_alu_energy_pj:>8.2f}pJ "
              f"{cpu_mult:>8.2f}x "
              f"{CIRCUIT_TYPE_MULTIPLIER['tensor_core']:>8.2f}x "
              f"{CIRCUIT_TYPE_MULTIPLIER['systolic_mac']:>8.2f}x "
              f"{CIRCUIT_TYPE_MULTIPLIER['domain_flow']:>8.2f}x")

    print(f"""

  KEY INSIGHTS:
  -------------
  1. x86 performance CPUs (i7-class) use 2.5x more energy per op than baseline
     due to aggressive 5GHz clocking and full IEEE-754 compliance.

  2. ARM efficiency CPUs (Cortex-A class) use baseline energy (1.0x)
     with simpler pipelines and lower voltage/frequency.

  3. At the SAME process node, architecture rankings are consistent:
     - KPU (domain_flow, 0.75x): Most efficient - no instruction fetch
     - TPU (systolic_mac, 0.80x): Very efficient - weight-stationary
     - GPU (tensor_core, 0.85x): Efficient MACs, but SIMT overhead
     - CPU: Depends on circuit type (1.0x to 2.5x)

  4. For embodied AI hw/sw design:
     - 8nm-x86: Intel i7 NUC-class comparison
     - 8nm-arm: ARM efficiency core comparison
     - Use same process node for fair architectural comparison
""")


# Legacy alias
def run_category_sweep(num_ops: int = 10000, bytes_transferred: int = 40960,
                       mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
                       verbose: bool = False) -> None:
    """Legacy alias for run_comparison_sweep."""
    run_comparison_sweep(num_ops, bytes_transferred, mode, verbose)


def main():
    # Build comparison set choices string for help
    arch_set_choices = list(ARCHITECTURE_COMPARISON_SETS.keys())
    arch_set_help = "\n".join([
        f"    {k}: {v.description}"
        for k, v in ARCHITECTURE_COMPARISON_SETS.items()
    ])

    # CPU circuit types
    cpu_types = ['x86_performance', 'x86_efficiency', 'arm_performance', 'arm_efficiency']
    cpu_type_help = ", ".join(cpu_types)

    parser = argparse.ArgumentParser(
        description='Compare energy across CPU, GPU, TPU, and KPU architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Architecture Comparison Sets (fair comparison at same process node):
{arch_set_help}

CPU Circuit Types:
    x86_performance: 2.50x - Intel i7/i9, AMD Ryzen 9 (5GHz, full IEEE-754)
    x86_efficiency:  1.50x - Intel E-cores (lower clocks, simpler pipeline)
    arm_performance: 1.80x - ARM Cortex-X4, Apple Firestorm
    arm_efficiency:  1.00x - ARM Cortex-A520, Apple Icestorm (baseline)

Examples:
  # Fair comparison at 8nm with x86 performance CPU
  %(prog)s --arch-comparison 8nm-x86

  # Fair comparison at 8nm with ARM efficiency CPU
  %(prog)s --arch-comparison 8nm-arm

  # 3-category breakdown (compute, control, data movement) - RECOMMENDED
  %(prog)s --categories --arch-comparison 8nm-x86

  # Custom comparison: 7nm with x86 efficiency CPU
  %(prog)s --process-node 7 --cpu-type x86_efficiency --memory lpddr5

  # Compare in L1-resident mode (all data on-chip)
  %(prog)s --mode l1 --arch-comparison 8nm-x86

  # Show architecture diagrams
  %(prog)s --diagram

  # Detailed phase breakdown
  %(prog)s --verbose --arch-comparison 4nm-datacenter

  # Operation scaling sweep with 3-category breakdown
  %(prog)s --categories --sweep --arch-comparison 8nm-x86

  # Mode comparison (all modes side-by-side)
  %(prog)s --mode-sweep --arch-comparison 8nm-arm

  # Compare all predefined comparison sets
  %(prog)s --comparison-sweep

  # JSON output with categories
  %(prog)s --output results.json --arch-comparison 8nm-x86
"""
    )

    parser.add_argument('--ops', type=int, default=10000,
                        help='Number of operations (default: 10000)')
    parser.add_argument('--bytes', type=int, default=40960,
                        help='Bytes transferred (default: 40960)')
    parser.add_argument('--tensor-core-util', type=float, default=0.8,
                        help='GPU tensor core utilization 0.0-1.0 (default: 0.8)')
    parser.add_argument('--mode', type=str, default='dram',
                        choices=['l1', 'l2', 'l3', 'dram'],
                        help='Operating mode: l1, l2, l3 (CPU/KPU only), dram (default: dram)')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of neural network layers for KPU (default: 1)')

    # Architecture comparison options
    parser.add_argument('--arch-comparison', '-a', type=str,
                        choices=arch_set_choices,
                        help='Predefined architecture comparison set (8nm-x86, 8nm-arm, 4nm-datacenter, 4nm-mobile)')
    parser.add_argument('--process-node', type=int,
                        help='Custom process node in nm (requires --cpu-type)')
    parser.add_argument('--cpu-type', type=str,
                        choices=cpu_types,
                        help='CPU circuit type (x86_performance, x86_efficiency, arm_performance, arm_efficiency)')
    parser.add_argument('--memory', type=str,
                        choices=['lpddr4', 'lpddr5', 'lpddr5x', 'ddr5', 'hbm3'],
                        help='Memory type for custom comparison')
    parser.add_argument('--comparison-sweep', action='store_true',
                        help='Compare energy across all predefined comparison sets')

    # Legacy (deprecated)
    parser.add_argument('--product-category', '-p', type=str,
                        choices=list(PRODUCT_CATEGORIES.keys()),
                        help='(DEPRECATED) Use --arch-comparison instead')
    parser.add_argument('--category-sweep', action='store_true',
                        help='(DEPRECATED) Use --comparison-sweep instead')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed phase breakdown')
    parser.add_argument('--categories', '-c', action='store_true',
                        help='Show clean 3-category breakdown (compute/control/data movement)')
    parser.add_argument('--diagram', '-d', action='store_true',
                        help='Show architecture diagrams')
    parser.add_argument('--sweep', action='store_true',
                        help='Run scaling sweep across operation counts')
    parser.add_argument('--mode-sweep', action='store_true',
                        help='Compare energy across all operating modes')
    parser.add_argument('--operator', type=str, default='high_reuse',
                        choices=['high_reuse', 'low_reuse', 'streaming', 'all'],
                        help='Operator type for cache hit ratio calculation '
                             '(high_reuse=MatMul, streaming=ReLU, all=compare both)')
    parser.add_argument('--working-set', type=float, default=20.0,
                        help='Working set size in MB (default: 20)')
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

    # Memory type mapping
    memory_map = {
        'lpddr4': MemoryType.LPDDR4,
        'lpddr5': MemoryType.LPDDR5,
        'lpddr5x': MemoryType.LPDDR5X,
        'ddr5': MemoryType.DDR5,
        'hbm3': MemoryType.HBM3,
    }

    # Determine comparison set
    comp_set = None

    # Priority 1: New --arch-comparison argument
    if args.arch_comparison:
        comp_set = get_architecture_comparison_set(args.arch_comparison)

    # Priority 2: Custom comparison with --process-node and --cpu-type
    elif args.process_node and args.cpu_type:
        mem_type = memory_map.get(args.memory, MemoryType.LPDDR5)
        comp_set = create_architecture_comparison_set(
            process_node_nm=args.process_node,
            memory_type=mem_type,
            cpu_circuit_type=args.cpu_type,
        )

    # Priority 3: Legacy --product-category (deprecated)
    elif args.product_category:
        print("WARNING: --product-category is deprecated. Use --arch-comparison instead.")
        comp_set = get_product_category(args.product_category)

    # Show diagrams if requested
    if args.diagram:
        print_architecture_diagrams()

    # Run comparison sweep if requested
    if args.comparison_sweep or args.category_sweep:
        run_comparison_sweep(args.ops, args.bytes, mode, args.verbose)
        return

    # Run mode sweep if requested
    if args.mode_sweep:
        run_architecture_mode_sweep(args.ops, args.bytes, args.verbose, category=comp_set)
        return

    # Get profiles from comparison set or use defaults
    if comp_set:
        cpu_profile = comp_set.cpu_profile
        gpu_profile = comp_set.gpu_profile
        tpu_profile = comp_set.tpu_profile
        kpu_profile = comp_set.kpu_profile
        print(format_comparison_set_header(comp_set))
    else:
        cpu_profile = gpu_profile = tpu_profile = kpu_profile = DEFAULT_PROFILE
        print("="*100)
        print("  ARCHITECTURE ENERGY COMPARISON")
        print("  CPU (MIMD) vs GPU (SIMT) vs TPU (Systolic) vs KPU (Domain Flow)")
        print("="*100)
        print("  WARNING: Using DEFAULT_PROFILE for all architectures.")
        print("  For FAIR comparison, use --arch-comparison to specify a comparison set.")
        print("  Example: --arch-comparison 8nm-x86")

    print(f"\n  Workload: {args.ops:,} operations, {args.bytes:,} bytes")
    print(f"  Mode: {get_mode_description(mode)}")
    defaults = DEFAULT_HIT_RATIOS[mode]
    print(f"  Hit ratios: L1={defaults.l1_hit:.0%}, L2={defaults.l2_hit:.0%}")

    # Build cycle energy breakdowns
    cpu_breakdown = build_cpu_cycle_energy(args.ops, args.bytes, mode=mode,
                                            tech_profile=cpu_profile, verbose=args.verbose)
    gpu_breakdown = build_gpu_cycle_energy(args.ops, args.bytes, mode=mode,
                                            tensor_core_utilization=args.tensor_core_util,
                                            tech_profile=gpu_profile, verbose=args.verbose)
    tpu_breakdown = build_tpu_cycle_energy(args.ops, args.bytes, mode=mode,
                                            tech_profile=tpu_profile, verbose=args.verbose)
    kpu_breakdown = build_kpu_cycle_energy(args.ops, args.bytes, mode=mode,
                                            num_layers=args.layers,
                                            tech_profile=kpu_profile, verbose=args.verbose)

    breakdowns = [cpu_breakdown, gpu_breakdown, tpu_breakdown, kpu_breakdown]

    # Print detailed breakdown if verbose
    if args.verbose:
        for breakdown in breakdowns:
            print(format_phase_breakdown(breakdown))

    # Print 3-category breakdown if requested (or by default when --categories is used)
    if args.categories:
        # Clean 3-category output - this is the main view when --categories is specified
        print(format_energy_categories_table(breakdowns, num_ops=args.ops))
        print(format_energy_categories_per_op_table(breakdowns, num_ops=args.ops))
    else:
        # Standard output with detailed phase breakdown
        # Print comparison table
        print(format_architecture_comparison_table(breakdowns, mode=mode, num_ops=args.ops))

        # Print detailed phase comparison
        print(format_detailed_phase_comparison(breakdowns, num_ops=args.ops))

        # Print insights
        print(format_architecture_insights(breakdowns, args.ops))

    # Run sweep if requested
    if args.sweep:
        # Parse operator type
        operator_map = {
            'high_reuse': OperatorType.HIGH_REUSE,
            'low_reuse': OperatorType.LOW_REUSE,
            'streaming': OperatorType.STREAMING,
        }

        if args.operator == 'all':
            # Run sweep for both HIGH_REUSE and STREAMING to show contrast
            for op_type in [OperatorType.HIGH_REUSE, OperatorType.STREAMING]:
                run_architecture_sweep(
                    mode=mode, verbose=args.verbose, category=comp_set,
                    operator_type=op_type, working_set_mb=args.working_set)
        else:
            op_type = operator_map.get(args.operator, OperatorType.HIGH_REUSE)
            run_architecture_sweep(
                mode=mode, verbose=args.verbose, category=comp_set,
                operator_type=op_type, working_set_mb=args.working_set)

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
            # Get energy categories
            categories = breakdown.get_energy_categories()

            arch_data = {
                "name": breakdown.architecture_name,
                "class": breakdown.architecture_class,
                "total_energy_pj": breakdown.total_energy_pj,
                "energy_per_op_pj": breakdown.total_energy_pj / args.ops,
                "categories": {
                    "compute_pj": categories['compute'],
                    "control_pj": categories['control'],
                    "data_movement_pj": categories['data_movement'],
                    "compute_percent": (categories['compute'] / categories['total'] * 100)
                                       if categories['total'] > 0 else 0,
                    "control_percent": (categories['control'] / categories['total'] * 100)
                                       if categories['total'] > 0 else 0,
                    "data_movement_percent": (categories['data_movement'] / categories['total'] * 100)
                                             if categories['total'] > 0 else 0,
                },
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
