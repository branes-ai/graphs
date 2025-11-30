"""
Comparison Utilities for Cycle-Level Energy Models

Provides formatting and analysis utilities for comparing energy
consumption across different processor architectures.
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from .base import (
    CyclePhase,
    OperatingMode,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    CycleEnergyBreakdown,
    get_mode_description,
)


# Column width for formatted tables
COL_WIDTH = 22


def format_energy(energy_pj: float, show_na: bool = True) -> str:
    """Format energy value with appropriate unit (auto-scaling)."""
    if energy_pj == 0.0 and show_na:
        return "n/a"
    elif energy_pj < 1000:
        return f"{energy_pj:.1f} pJ"
    elif energy_pj < 1_000_000:
        return f"{energy_pj/1000:.2f} nJ"
    elif energy_pj < 1_000_000_000:
        return f"{energy_pj/1_000_000:.2f} uJ"
    else:
        return f"{energy_pj/1_000_000_000:.2f} mJ"


# Energy scale definitions (all relative to pJ)
ENERGY_SCALES = [
    ('fJ', 1e-3),      # femtojoules
    ('pJ', 1.0),       # picojoules (base)
    ('nJ', 1e3),       # nanojoules
    ('uJ', 1e6),       # microjoules
    ('mJ', 1e9),       # millijoules
    ('J',  1e12),      # joules
]


def determine_common_scale(values_pj: List[float]) -> tuple:
    """
    Determine the best common scale for a list of energy values.

    Returns:
        Tuple of (unit_name, divisor) where divisor converts pJ to the unit.

    Strategy: Find the scale where the smallest non-zero value is >= 1.0
    and the largest value is reasonably displayed (< 10000 preferred).
    """
    # Filter out zeros and get min/max
    non_zero = [v for v in values_pj if v > 0]
    if not non_zero:
        return ('pJ', 1.0)

    min_val = min(non_zero)
    max_val = max(non_zero)

    # Find the scale where min_val >= 0.1 (so we get at least one significant digit)
    best_scale = ('pJ', 1.0)
    for unit_name, divisor in ENERGY_SCALES:
        scaled_min = min_val / divisor
        scaled_max = max_val / divisor
        # We want min >= 0.1 and max < 100000 for readable output
        if scaled_min >= 0.1 and scaled_max < 100000:
            best_scale = (unit_name, divisor)
            break

    return best_scale


def format_energy_with_scale(energy_pj: float, divisor: float, decimals: int = 2) -> str:
    """Format energy value using a specific scale (no unit suffix)."""
    if energy_pj == 0.0:
        return "0"
    scaled = energy_pj / divisor
    if scaled >= 100:
        return f"{scaled:.0f}"
    elif scaled >= 10:
        return f"{scaled:.1f}"
    else:
        return f"{scaled:.{decimals}f}"


def format_phase_breakdown(
    breakdown: CycleEnergyBreakdown,
    indent: int = 2
) -> str:
    """Format detailed phase breakdown for a single architecture."""
    lines = []
    lines.append(f"\n{breakdown.architecture_name}")
    lines.append(f"  Class: {breakdown.architecture_class}")
    lines.append(f"  Cycles: {breakdown.num_cycles:,}, Ops/cycle: {breakdown.ops_per_cycle}")
    lines.append("")

    # Group events by phase
    phase_events: Dict[CyclePhase, List] = {}
    for event in breakdown.events:
        if event.phase not in phase_events:
            phase_events[event.phase] = []
        phase_events[event.phase].append(event)

    # Print each phase
    for phase in CyclePhase:
        if phase not in phase_events:
            continue
        events = phase_events[phase]
        phase_total = sum(e.total_energy_pj for e in events)
        pct = 100.0 * phase_total / breakdown.total_energy_pj if breakdown.total_energy_pj > 0 else 0

        lines.append(f"  {phase.value}: {format_energy(phase_total)} ({pct:.1f}%)")
        for event in events:
            lines.append(f"    - {event.description}: {format_energy(event.total_energy_pj)}")
            lines.append(f"      ({event.count:,} x {event.energy_pj:.2f} pJ)")

    lines.append("")
    lines.append(f"  TOTAL: {format_energy(breakdown.total_energy_pj)}")
    lines.append(f"  Energy/cycle: {format_energy(breakdown.energy_per_cycle_pj)}")

    return "\n".join(lines)


def format_comparison_table(
    breakdowns: List[CycleEnergyBreakdown],
    title: str = "Energy Comparison",
    show_per_op: bool = False,
    num_ops: int = 1000
) -> str:
    """
    Format a comparison table across multiple architectures.

    Args:
        breakdowns: List of CycleEnergyBreakdown objects to compare
        title: Table title
        show_per_op: If True, show energy per operation instead of total
        num_ops: Number of operations (for per-op calculation)

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"{title}")
    lines.append(f"{'=' * 80}")

    # Build header
    header = f"{'Phase':<{COL_WIDTH}}"
    for bd in breakdowns:
        # Shorten name for column header
        short_name = bd.architecture_name.split('(')[0].strip()
        header += f"{short_name:>{COL_WIDTH}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Define phase categories with subcategories
    phase_categories = [
        ("Instruction Overhead", [
            ("  Fetch", CyclePhase.INSTRUCTION_FETCH),
            ("  Decode", CyclePhase.INSTRUCTION_DECODE),
        ]),
        ("Operand/Register", [
            ("  Operand Fetch", CyclePhase.OPERAND_FETCH),
            ("  Writeback", CyclePhase.WRITEBACK),
        ]),
        ("Execute", [
            ("  ALU/Compute", CyclePhase.EXECUTE),
        ]),
        ("GPU SIMT Overhead", [
            ("  Fixed Infrastructure", CyclePhase.SIMT_FIXED_OVERHEAD),
            ("  Thread Management", CyclePhase.SIMT_THREAD_MGMT),
            ("  Coherence", CyclePhase.SIMT_COHERENCE),
            ("  Synchronization", CyclePhase.SIMT_SYNC),
        ]),
        ("TPU Systolic", [
            ("  Control", CyclePhase.SYSTOLIC_CONTROL),
            ("  Weight Load", CyclePhase.SYSTOLIC_WEIGHT_LOAD),
            ("  Data Load", CyclePhase.SYSTOLIC_DATA_LOAD),
            ("  Compute", CyclePhase.SYSTOLIC_COMPUTE),
            ("  Drain", CyclePhase.SYSTOLIC_DRAIN),
        ]),
        ("KPU Spatial", [
            ("  Configuration", CyclePhase.SPATIAL_CONFIG),
            ("  Streaming", CyclePhase.SPATIAL_STREAM),
            ("  Compute", CyclePhase.SPATIAL_COMPUTE),
            ("  Interconnect", CyclePhase.SPATIAL_INTERCONNECT),
        ]),
        ("Memory Hierarchy", [
            ("  L1/Shared/SRAM", CyclePhase.MEM_L1),
            ("  SRAM (on-chip)", CyclePhase.MEM_SRAM),
            ("  L2 Cache", CyclePhase.MEM_L2),
            ("  L3/LLC", CyclePhase.MEM_L3),
            ("  DRAM", CyclePhase.MEM_DRAM),
            ("  HBM", CyclePhase.MEM_HBM),
        ]),
    ]

    for category_name, subcategories in phase_categories:
        # Check if any architecture has energy in this category
        category_has_energy = False
        for _, phase in subcategories:
            for bd in breakdowns:
                if bd.get_phase_energy(phase) > 0:
                    category_has_energy = True
                    break
            if category_has_energy:
                break

        if not category_has_energy:
            continue

        # Category header
        lines.append(f"{category_name:<{COL_WIDTH}}" + " " * (COL_WIDTH * len(breakdowns)))

        # Subcategories
        for subcat_name, phase in subcategories:
            row = f"{subcat_name:<{COL_WIDTH}}"
            has_any = False
            for bd in breakdowns:
                energy = bd.get_phase_energy(phase)
                if show_per_op and num_ops > 0:
                    energy = energy / num_ops
                if energy > 0:
                    has_any = True
                    row += f"{format_energy(energy):>{COL_WIDTH}}"
                else:
                    row += f"{'n/a':>{COL_WIDTH}}"
            if has_any:
                lines.append(row)

    # Total row
    lines.append("-" * len(header))
    row = f"{'TOTAL':<{COL_WIDTH}}"
    for bd in breakdowns:
        energy = bd.total_energy_pj
        if show_per_op and num_ops > 0:
            energy = energy / num_ops
        row += f"{format_energy(energy):>{COL_WIDTH}}"
    lines.append(row)

    return "\n".join(lines)


def format_key_insights(
    breakdowns: List[CycleEnergyBreakdown],
    num_ops: int = 1000
) -> str:
    """Generate key insights from architecture comparison."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 80)

    if not breakdowns:
        return "\n".join(lines)

    # Find most and least efficient
    by_total = sorted(breakdowns, key=lambda b: b.total_energy_pj)
    most_efficient = by_total[0]
    least_efficient = by_total[-1]

    lines.append(f"\n1. Most efficient: {most_efficient.architecture_name}")
    lines.append(f"   Total energy: {format_energy(most_efficient.total_energy_pj)}")
    lines.append(f"   Energy/op: {format_energy(most_efficient.total_energy_pj / num_ops)}")

    lines.append(f"\n2. Least efficient: {least_efficient.architecture_name}")
    lines.append(f"   Total energy: {format_energy(least_efficient.total_energy_pj)}")
    lines.append(f"   Energy/op: {format_energy(least_efficient.total_energy_pj / num_ops)}")

    if len(breakdowns) > 1:
        ratio = least_efficient.total_energy_pj / most_efficient.total_energy_pj
        lines.append(f"\n3. Efficiency ratio: {ratio:.1f}x difference")

    # Architecture-specific insights
    for bd in breakdowns:
        simt_overhead = bd.get_simt_overhead()
        if simt_overhead > 0:
            pct = 100.0 * simt_overhead / bd.total_energy_pj
            lines.append(f"\n4. {bd.architecture_name} SIMT overhead: {format_energy(simt_overhead)} ({pct:.1f}%)")

        systolic_energy = bd.get_systolic_overhead()
        if systolic_energy > 0:
            pct = 100.0 * systolic_energy / bd.total_energy_pj
            lines.append(f"\n5. {bd.architecture_name} systolic energy: {format_energy(systolic_energy)} ({pct:.1f}%)")

        spatial_energy = bd.get_spatial_overhead()
        if spatial_energy > 0:
            pct = 100.0 * spatial_energy / bd.total_energy_pj
            lines.append(f"\n6. {bd.architecture_name} spatial energy: {format_energy(spatial_energy)} ({pct:.1f}%)")

    return "\n".join(lines)


@dataclass
class SweepResult:
    """Result from a single sweep point."""
    num_ops: int
    bytes_transferred: int
    architecture: str
    total_energy_pj: float
    energy_per_op_pj: float
    mode: OperatingMode


def run_sweep(
    build_funcs: Dict[str, Callable],
    ops_range: List[int],
    bytes_per_op: int = 4,
    mode: OperatingMode = OperatingMode.DRAM_RESIDENT,
    hit_ratios: Optional[HitRatios] = None,
) -> List[SweepResult]:
    """
    Run a parameter sweep across architectures.

    Args:
        build_funcs: Dict mapping architecture name to build function
        ops_range: List of operation counts to test
        bytes_per_op: Bytes transferred per operation
        mode: Operating mode
        hit_ratios: Custom hit ratios (uses defaults if None)

    Returns:
        List of SweepResult objects
    """
    results = []

    for num_ops in ops_range:
        bytes_transferred = num_ops * bytes_per_op

        for arch_name, build_func in build_funcs.items():
            breakdown = build_func(
                num_ops=num_ops,
                bytes_transferred=bytes_transferred,
                mode=mode,
                hit_ratios=hit_ratios,
            )

            results.append(SweepResult(
                num_ops=num_ops,
                bytes_transferred=bytes_transferred,
                architecture=arch_name,
                total_energy_pj=breakdown.total_energy_pj,
                energy_per_op_pj=breakdown.total_energy_pj / num_ops,
                mode=mode,
            ))

    return results


def run_mode_sweep(
    build_funcs: Dict[str, Callable],
    num_ops: int = 10000,
    bytes_per_op: int = 4,
    modes: Optional[List[OperatingMode]] = None,
) -> List[SweepResult]:
    """
    Run a sweep across operating modes for each architecture.

    Args:
        build_funcs: Dict mapping architecture name to build function
        num_ops: Number of operations
        bytes_per_op: Bytes transferred per operation
        modes: List of modes to test (defaults to all modes)

    Returns:
        List of SweepResult objects
    """
    if modes is None:
        modes = list(OperatingMode)

    results = []
    bytes_transferred = num_ops * bytes_per_op

    for mode in modes:
        for arch_name, build_func in build_funcs.items():
            breakdown = build_func(
                num_ops=num_ops,
                bytes_transferred=bytes_transferred,
                mode=mode,
            )

            results.append(SweepResult(
                num_ops=num_ops,
                bytes_transferred=bytes_transferred,
                architecture=arch_name,
                total_energy_pj=breakdown.total_energy_pj,
                energy_per_op_pj=breakdown.total_energy_pj / num_ops,
                mode=mode,
            ))

    return results


def format_sweep_table(
    results: List[SweepResult],
    show_per_op: bool = True,
    title: str = "Energy Scaling Analysis"
) -> str:
    """
    Format sweep results as a table.

    Args:
        results: List of SweepResult objects
        show_per_op: If True, show energy per operation
        title: Table title

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"{title}")
    lines.append(f"{'=' * 80}")

    # Get unique architectures and operation counts
    architectures = sorted(set(r.architecture for r in results))
    ops_counts = sorted(set(r.num_ops for r in results))

    # Build header
    header = f"{'Operations':<15}"
    for arch in architectures:
        short_name = arch.split('(')[0].strip() if '(' in arch else arch
        header += f"{short_name:>{COL_WIDTH}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Build rows
    for num_ops in ops_counts:
        row = f"{num_ops:<15,}"
        for arch in architectures:
            # Find matching result
            matching = [r for r in results if r.architecture == arch and r.num_ops == num_ops]
            if matching:
                r = matching[0]
                energy = r.energy_per_op_pj if show_per_op else r.total_energy_pj
                row += f"{format_energy(energy):>{COL_WIDTH}}"
            else:
                row += f"{'n/a':>{COL_WIDTH}}"
        lines.append(row)

    return "\n".join(lines)


def format_mode_comparison_table(
    results: List[SweepResult],
    show_per_op: bool = True,
    title: str = "Operating Mode Comparison"
) -> str:
    """
    Format mode sweep results as a table.

    Args:
        results: List of SweepResult objects from mode sweep
        show_per_op: If True, show energy per operation
        title: Table title

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"{title}")
    lines.append(f"{'=' * 80}")

    # Get unique architectures and modes
    architectures = sorted(set(r.architecture for r in results))
    modes = sorted(set(r.mode for r in results), key=lambda m: m.value)

    # Build header
    header = f"{'Mode':<25}"
    for arch in architectures:
        short_name = arch.split('(')[0].strip() if '(' in arch else arch
        header += f"{short_name:>{COL_WIDTH}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Build rows
    for mode in modes:
        row = f"{get_mode_description(mode):<25}"
        for arch in architectures:
            # Find matching result
            matching = [r for r in results if r.architecture == arch and r.mode == mode]
            if matching:
                r = matching[0]
                energy = r.energy_per_op_pj if show_per_op else r.total_energy_pj
                row += f"{format_energy(energy):>{COL_WIDTH}}"
            else:
                row += f"{'n/a':>{COL_WIDTH}}"
        lines.append(row)

    return "\n".join(lines)
