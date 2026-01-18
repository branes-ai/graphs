#!/usr/bin/env python3
"""
Compare Power Allocations CLI Tool

Compare different power allocation strategies and their impact on mission capabilities.
Shows tradeoffs between perception, control, and movement subsystems.

Usage:
    # Compare allocation strategies
    ./cli/compare_power_allocations.py --total-budget 50 --strategies perception-heavy balanced control-heavy

    # For specific application
    ./cli/compare_power_allocations.py --application warehouse-amr --total-budget 30

    # With mission profile
    ./cli/compare_power_allocations.py --profile drone-inspection --total-budget 15

    # JSON output
    ./cli/compare_power_allocations.py --total-budget 50 --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.capability_tiers import (
    TierName,
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.power_allocation import (
    PowerAllocation,
    SubsystemType,
    get_typical_allocation,
    get_application_allocation,
    get_allocation_strategy,
    list_allocation_strategies,
    ALLOCATION_PERCEPTION_HEAVY,
    ALLOCATION_BALANCED,
    ALLOCATION_CONTROL_HEAVY,
    ALLOCATION_MOVEMENT_HEAVY,
    ALLOCATION_STATIONARY,
    APPLICATION_ALLOCATIONS,
)
from graphs.mission.mission_profiles import (
    MissionProfile,
    MISSION_PROFILES,
    get_mission_profile,
)


@dataclass
class AllocationAnalysis:
    """Analysis of a single allocation strategy."""
    name: str
    description: str
    allocation: PowerAllocation

    # Absolute power values
    perception_power_w: float
    control_power_w: float
    movement_power_w: float
    overhead_power_w: float
    total_power_w: float

    # Derived capabilities (estimated)
    estimated_inference_fps: float = 0.0
    estimated_control_hz: float = 0.0
    movement_capability: str = "standard"

    # Mission impact
    mission_suitability: str = "general"
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "ratios": {
                "perception": self.allocation.perception_ratio,
                "control": self.allocation.control_ratio,
                "movement": self.allocation.movement_ratio,
                "overhead": self.allocation.overhead_ratio,
            },
            "power_watts": {
                "perception": self.perception_power_w,
                "control": self.control_power_w,
                "movement": self.movement_power_w,
                "overhead": self.overhead_power_w,
                "total": self.total_power_w,
            },
            "capabilities": {
                "estimated_inference_fps": self.estimated_inference_fps,
                "estimated_control_hz": self.estimated_control_hz,
                "movement_capability": self.movement_capability,
            },
            "mission_suitability": self.mission_suitability,
            "limitations": self.limitations,
        }


@dataclass
class AllocationComparisonResult:
    """Result of comparing multiple allocation strategies."""
    total_budget_w: float
    strategies: List[AllocationAnalysis]
    application: Optional[str]
    profile: Optional[str]

    # Recommendations
    best_for_perception: Optional[str] = None
    best_for_control: Optional[str] = None
    best_for_movement: Optional[str] = None
    most_balanced: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_budget_w": self.total_budget_w,
            "application": self.application,
            "profile": self.profile,
            "strategies": [s.to_dict() for s in self.strategies],
            "recommendations": {
                "best_for_perception": self.best_for_perception,
                "best_for_control": self.best_for_control,
                "best_for_movement": self.best_for_movement,
                "most_balanced": self.most_balanced,
            },
        }


def estimate_capabilities(power_w: float, subsystem: str) -> Dict[str, Any]:
    """Estimate capabilities based on power allocation."""
    if subsystem == "perception":
        # Rough estimate: 1W of perception power ~ 10-20 FPS inference
        fps = power_w * 15
        return {"inference_fps": fps}
    elif subsystem == "control":
        # Rough estimate: 1W of control power ~ 100-500 Hz control loop
        hz = power_w * 200
        return {"control_hz": hz}
    elif subsystem == "movement":
        # Categorize movement capability
        if power_w < 5:
            capability = "minimal"
        elif power_w < 15:
            capability = "light"
        elif power_w < 30:
            capability = "standard"
        elif power_w < 60:
            capability = "dynamic"
        else:
            capability = "high-performance"
        return {"capability": capability}

    return {}


def analyze_allocation(
    name: str,
    allocation: PowerAllocation,
    total_budget_w: float,
    description: str = "",
) -> AllocationAnalysis:
    """Analyze a single allocation strategy."""
    # Calculate absolute power values
    powers = allocation.allocate_power(total_budget_w)

    perception_power = powers[SubsystemType.PERCEPTION]
    control_power = powers[SubsystemType.CONTROL]
    movement_power = powers[SubsystemType.MOVEMENT]
    overhead_power = powers[SubsystemType.OVERHEAD]

    # Estimate capabilities
    perception_caps = estimate_capabilities(perception_power, "perception")
    control_caps = estimate_capabilities(control_power, "control")
    movement_caps = estimate_capabilities(movement_power, "movement")

    # Determine mission suitability and limitations
    limitations = []
    if perception_power < 3:
        limitations.append("Limited perception capability (<3W)")
    if control_power < 2:
        limitations.append("Basic control only (<2W)")
    if movement_power < 5:
        limitations.append("Minimal movement capability (<5W)")

    # Mission suitability
    if allocation.perception_ratio >= 0.5:
        suitability = "perception-intensive (inspection, monitoring)"
    elif allocation.movement_ratio >= 0.4:
        suitability = "movement-intensive (locomotion, flight)"
    elif allocation.control_ratio >= 0.3:
        suitability = "control-intensive (manipulation, planning)"
    else:
        suitability = "general-purpose"

    return AllocationAnalysis(
        name=name,
        description=description or allocation.description,
        allocation=allocation,
        perception_power_w=perception_power,
        control_power_w=control_power,
        movement_power_w=movement_power,
        overhead_power_w=overhead_power,
        total_power_w=total_budget_w,
        estimated_inference_fps=perception_caps.get("inference_fps", 0),
        estimated_control_hz=control_caps.get("control_hz", 0),
        movement_capability=movement_caps.get("capability", "unknown"),
        mission_suitability=suitability,
        limitations=limitations,
    )


def compare_power_allocations(
    total_budget_w: float,
    strategy_names: Optional[List[str]] = None,
    application: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> AllocationComparisonResult:
    """
    Compare multiple power allocation strategies.

    Args:
        total_budget_w: Total power budget in watts
        strategy_names: List of strategy names to compare
        application: Application type for context
        profile_name: Mission profile name for context

    Returns:
        AllocationComparisonResult with detailed comparison
    """
    strategies = []

    # Get strategies to compare
    if strategy_names:
        for name in strategy_names:
            alloc = get_allocation_strategy(name)
            if alloc:
                analysis = analyze_allocation(name, alloc, total_budget_w)
                strategies.append(analysis)
    else:
        # Default: compare all standard strategies
        all_strategies = [
            ("perception-heavy", ALLOCATION_PERCEPTION_HEAVY),
            ("balanced", ALLOCATION_BALANCED),
            ("control-heavy", ALLOCATION_CONTROL_HEAVY),
            ("movement-heavy", ALLOCATION_MOVEMENT_HEAVY),
            ("stationary", ALLOCATION_STATIONARY),
        ]

        for name, alloc in all_strategies:
            analysis = analyze_allocation(name, alloc, total_budget_w)
            strategies.append(analysis)

    # Add application-specific allocation if specified
    if application:
        app_alloc = get_application_allocation(application)
        if app_alloc:
            analysis = analyze_allocation(f"app:{application}", app_alloc, total_budget_w)
            strategies.append(analysis)

    # Add profile-specific allocation if specified
    profile = None
    if profile_name:
        profile = get_mission_profile(profile_name)
        if profile:
            # Derive allocation from profile duty cycles
            tier = get_tier_for_power(total_budget_w)
            if tier:
                tier_alloc = get_typical_allocation(tier.name)
                analysis = analyze_allocation(f"profile:{profile_name}", tier_alloc, total_budget_w)
                strategies.append(analysis)

    # Determine recommendations
    if strategies:
        best_perception = max(strategies, key=lambda x: x.perception_power_w).name
        best_control = max(strategies, key=lambda x: x.control_power_w).name
        best_movement = max(strategies, key=lambda x: x.movement_power_w).name

        # Most balanced = smallest max deviation from 0.25 each
        def balance_score(s):
            ratios = [
                s.allocation.perception_ratio,
                s.allocation.control_ratio,
                s.allocation.movement_ratio,
                s.allocation.overhead_ratio,
            ]
            return max(abs(r - 0.25) for r in ratios)

        most_balanced = min(strategies, key=balance_score).name
    else:
        best_perception = None
        best_control = None
        best_movement = None
        most_balanced = None

    return AllocationComparisonResult(
        total_budget_w=total_budget_w,
        strategies=strategies,
        application=application,
        profile=profile_name,
        best_for_perception=best_perception,
        best_for_control=best_control,
        best_for_movement=best_movement,
        most_balanced=most_balanced,
    )


def format_comparison(result: AllocationComparisonResult, verbose: bool = False) -> str:
    """Format comparison result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  POWER ALLOCATION COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append(f"  Total Power Budget: {result.total_budget_w:.0f}W")
    if result.application:
        lines.append(f"  Application: {result.application}")
    if result.profile:
        lines.append(f"  Mission Profile: {result.profile}")
    lines.append("")

    # Strategy comparison table
    lines.append("  Allocation Strategies:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Strategy':<18} {'Percep':>8} {'Control':>8} {'Move':>8} {'Overhead':>8} {'Total':>8}")
    lines.append("  " + "-" * 76)

    for s in result.strategies:
        lines.append(
            f"    {s.name:<18} {s.perception_power_w:>7.1f}W {s.control_power_w:>7.1f}W "
            f"{s.movement_power_w:>7.1f}W {s.overhead_power_w:>7.1f}W {s.total_power_w:>7.1f}W"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Visual comparison
    lines.append("  Visual Comparison (power distribution):")
    lines.append("")

    bar_width = 50
    for s in result.strategies:
        # Create stacked bar
        p_len = int(s.allocation.perception_ratio * bar_width)
        c_len = int(s.allocation.control_ratio * bar_width)
        m_len = int(s.allocation.movement_ratio * bar_width)
        o_len = bar_width - p_len - c_len - m_len

        bar = "P" * p_len + "C" * c_len + "M" * m_len + "O" * o_len
        lines.append(f"    {s.name:<18} [{bar}]")

    lines.append("")
    lines.append("    Legend: P=Perception, C=Control, M=Movement, O=Overhead")
    lines.append("")

    # Capability estimates
    lines.append("  Estimated Capabilities:")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Strategy':<18} {'Inference':>12} {'Control':>12} {'Movement':>12}")
    lines.append("  " + "-" * 60)

    for s in result.strategies:
        fps_str = f"{s.estimated_inference_fps:.0f} FPS"
        hz_str = f"{s.estimated_control_hz:.0f} Hz"
        lines.append(
            f"    {s.name:<18} {fps_str:>12} {hz_str:>12} {s.movement_capability:>12}"
        )

    lines.append("  " + "-" * 60)
    lines.append("")

    # Recommendations
    lines.append("  Recommendations:")
    lines.append(f"    Best for Perception: {result.best_for_perception}")
    lines.append(f"    Best for Control:    {result.best_for_control}")
    lines.append(f"    Best for Movement:   {result.best_for_movement}")
    lines.append(f"    Most Balanced:       {result.most_balanced}")
    lines.append("")

    # Limitations (if verbose)
    if verbose:
        lines.append("  Strategy Limitations:")
        for s in result.strategies:
            if s.limitations:
                lines.append(f"    {s.name}:")
                for lim in s.limitations:
                    lines.append(f"      - {lim}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare power allocation strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare standard strategies
  ./cli/compare_power_allocations.py --total-budget 50

  # Compare specific strategies
  ./cli/compare_power_allocations.py --total-budget 30 --strategies perception-heavy balanced movement-heavy

  # For specific application
  ./cli/compare_power_allocations.py --total-budget 30 --application warehouse-amr

  # With mission profile
  ./cli/compare_power_allocations.py --total-budget 15 --profile drone-inspection

  # JSON output
  ./cli/compare_power_allocations.py --total-budget 50 --format json
"""
    )

    parser.add_argument(
        "--total-budget", "-b",
        type=float,
        required=True,
        help="Total power budget in watts"
    )
    parser.add_argument(
        "--strategies", "-s",
        type=str,
        nargs="+",
        choices=list_allocation_strategies(),
        help="Strategies to compare (default: all)"
    )
    parser.add_argument(
        "--application", "-a",
        type=str,
        choices=list(APPLICATION_ALLOCATIONS.keys()),
        help="Application type for context"
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        choices=list(MISSION_PROFILES.keys()),
        help="Mission profile for context"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including limitations"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available allocation strategies"
    )
    parser.add_argument(
        "--list-applications",
        action="store_true",
        help="List available application types"
    )

    args = parser.parse_args()

    # Handle list options
    if args.list_strategies:
        print("Available allocation strategies:")
        for name in list_allocation_strategies():
            alloc = get_allocation_strategy(name)
            if alloc:
                print(f"  {name:<18} {alloc.description}")
        return 0

    if args.list_applications:
        print("Available application types:")
        for name in sorted(APPLICATION_ALLOCATIONS.keys()):
            alloc = APPLICATION_ALLOCATIONS[name]
            print(f"  {name:<25} {alloc.description}")
        return 0

    # Run comparison
    result = compare_power_allocations(
        total_budget_w=args.total_budget,
        strategy_names=args.strategies,
        application=args.application,
        profile_name=args.profile,
    )

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_comparison(result, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
