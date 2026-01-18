#!/usr/bin/env python3
"""
Explore Power Allocation CLI Tool

Analyze how compute power should be distributed across subsystems
(perception, control, movement, overhead) for different capability tiers
and mission profiles.

Usage:
    # Show typical allocation for a tier
    ./cli/explore_power_allocation.py --tier micro-autonomy

    # Explore allocation for a specific mission profile
    ./cli/explore_power_allocation.py --mission drone-inspection

    # Compare allocations across all tiers
    ./cli/explore_power_allocation.py --compare-tiers

    # Custom allocation analysis
    ./cli/explore_power_allocation.py --tier embodied-ai --perception 40 --control 25 --movement 30

    # JSON output for integration
    ./cli/explore_power_allocation.py --tier micro-autonomy --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.capability_tiers import (
    TierName,
    CAPABILITY_TIERS,
    CAPABILITY_TIERS_ORDERED,
    get_tier_by_name,
    list_tier_names,
)
from graphs.mission.power_allocation import (
    PowerAllocation,
    SubsystemType,
    TYPICAL_ALLOCATIONS,
    APPLICATION_ALLOCATIONS,
    get_typical_allocation,
)
from graphs.mission.mission_profiles import (
    MissionProfile,
    MISSION_PROFILES,
    get_mission_profile,
    list_mission_profiles,
    get_profiles_for_tier,
)


def format_allocation_bar(ratio: float, width: int = 30) -> str:
    """Format a ratio as a visual bar."""
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_allocation_details(
    allocation: PowerAllocation,
    total_power_w: Optional[float] = None,
    show_bars: bool = True
) -> str:
    """Format power allocation as detailed text."""
    lines = []

    if show_bars:
        lines.append("  Subsystem    | Ratio |  %    | Distribution")
        lines.append("  " + "-" * 60)
        lines.append(f"  Perception   | {allocation.perception_ratio:5.2f} | {allocation.perception_ratio*100:5.1f}% | {format_allocation_bar(allocation.perception_ratio)}")
        lines.append(f"  Control      | {allocation.control_ratio:5.2f} | {allocation.control_ratio*100:5.1f}% | {format_allocation_bar(allocation.control_ratio)}")
        lines.append(f"  Movement     | {allocation.movement_ratio:5.2f} | {allocation.movement_ratio*100:5.1f}% | {format_allocation_bar(allocation.movement_ratio)}")
        lines.append(f"  Overhead     | {allocation.overhead_ratio:5.2f} | {allocation.overhead_ratio*100:5.1f}% | {format_allocation_bar(allocation.overhead_ratio)}")
    else:
        lines.append("  Subsystem    | Ratio |  %")
        lines.append("  " + "-" * 30)
        lines.append(f"  Perception   | {allocation.perception_ratio:5.2f} | {allocation.perception_ratio*100:5.1f}%")
        lines.append(f"  Control      | {allocation.control_ratio:5.2f} | {allocation.control_ratio*100:5.1f}%")
        lines.append(f"  Movement     | {allocation.movement_ratio:5.2f} | {allocation.movement_ratio*100:5.1f}%")
        lines.append(f"  Overhead     | {allocation.overhead_ratio:5.2f} | {allocation.overhead_ratio*100:5.1f}%")

    if total_power_w:
        lines.append("")
        lines.append(f"  At {total_power_w:.1f}W total power budget:")
        lines.append(f"    Perception: {allocation.perception_ratio * total_power_w:6.2f}W")
        lines.append(f"    Control:    {allocation.control_ratio * total_power_w:6.2f}W")
        lines.append(f"    Movement:   {allocation.movement_ratio * total_power_w:6.2f}W")
        lines.append(f"    Overhead:   {allocation.overhead_ratio * total_power_w:6.2f}W")

    return "\n".join(lines)


def format_tier_allocation(tier_name: TierName, show_missions: bool = False) -> str:
    """Format power allocation for a capability tier."""
    tier = CAPABILITY_TIERS[tier_name]
    allocation = get_typical_allocation(tier_name)

    lines = []
    lines.append("=" * 80)
    lines.append(f"  POWER ALLOCATION FOR {tier.display_name.upper()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Power Envelope: {tier.power_range_str}")
    lines.append(f"  Typical Power:  {tier.typical_power_w:.1f}W")
    lines.append("")
    lines.append(format_allocation_details(allocation, tier.typical_power_w))
    lines.append("")

    # Allocation rationale
    lines.append("  Allocation Rationale:")
    if tier_name == TierName.WEARABLE_AI:
        lines.append("    - Minimal movement (haptic feedback only)")
        lines.append("    - Perception dominated by sensors")
        lines.append("    - Control handles simple inference tasks")
        lines.append("    - Higher overhead due to small battery management")
    elif tier_name == TierName.MICRO_AUTONOMY:
        lines.append("    - Movement significant (flight, locomotion)")
        lines.append("    - Perception for navigation + task execution")
        lines.append("    - Control handles path planning + flight control")
        lines.append("    - Standard overhead for embedded systems")
    elif tier_name == TierName.INDUSTRIAL_EDGE:
        lines.append("    - Perception heavy (cameras, LIDAR for SLAM)")
        lines.append("    - Movement for wheels/arms")
        lines.append("    - Control for fleet coordination + safety")
        lines.append("    - Lower overhead with AC power available")
    elif tier_name == TierName.EMBODIED_AI:
        lines.append("    - High perception for world modeling")
        lines.append("    - Movement dominated by actuators (legged robots)")
        lines.append("    - Complex control for balance + manipulation")
        lines.append("    - Standard overhead")
    elif tier_name == TierName.AUTOMOTIVE_AI:
        lines.append("    - Perception dominates (sensor fusion)")
        lines.append("    - Movement minimal (compute only, not propulsion)")
        lines.append("    - High control for safety-critical decisions")
        lines.append("    - Lower overhead with vehicle power supply")
    lines.append("")

    # Show mission profiles if requested
    if show_missions:
        profiles = get_profiles_for_tier(tier_name)
        if profiles:
            lines.append("  Mission Profiles for this Tier:")
            lines.append("  " + "-" * 40)
            for profile in profiles:
                avg_mult = profile.estimate_average_power_multiplier()
                lines.append(f"    {profile.display_name}:")
                lines.append(f"      Duration: {profile.typical_duration_hours:.1f}h")
                lines.append(f"      Avg power multiplier: {avg_mult:.2f}x")
                lines.append(f"      Duty cycles:")
                lines.append(f"        Perception: {profile.perception_duty.active_ratio*100:.0f}% active")
                lines.append(f"        Control:    {profile.control_duty.active_ratio*100:.0f}% active")
                lines.append(f"        Movement:   {profile.movement_duty.active_ratio*100:.0f}% active")
                lines.append("")

    return "\n".join(lines)


def format_mission_allocation(profile: MissionProfile) -> str:
    """Format power allocation for a specific mission profile."""
    tier = CAPABILITY_TIERS[profile.tier]
    base_allocation = get_typical_allocation(profile.tier)

    lines = []
    lines.append("=" * 80)
    lines.append(f"  POWER ALLOCATION FOR {profile.display_name.upper()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Capability Tier: {tier.display_name}")
    lines.append(f"  Power Envelope:  {tier.power_range_str}")
    lines.append(f"  Mission Duration: {profile.typical_duration_hours:.1f} hours")
    lines.append(f"  Environment:     {profile.environment}")
    lines.append("")
    lines.append("  Description:")
    # Word wrap description
    desc_words = profile.description.split()
    line = "    "
    for word in desc_words:
        if len(line) + len(word) > 76:
            lines.append(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line)
    lines.append("")

    # Base allocation
    lines.append("  Base Power Allocation (tier default):")
    lines.append(format_allocation_details(base_allocation, show_bars=False))
    lines.append("")

    # Duty cycle analysis
    lines.append("  Mission Duty Cycles:")
    lines.append("  " + "-" * 50)
    lines.append(f"    Perception: {profile.perception_duty.active_ratio*100:5.1f}% active, {profile.perception_duty.peak_ratio*100:5.1f}% at peak")
    lines.append(f"    Control:    {profile.control_duty.active_ratio*100:5.1f}% active, {profile.control_duty.peak_ratio*100:5.1f}% at peak")
    lines.append(f"    Movement:   {profile.movement_duty.active_ratio*100:5.1f}% active, {profile.movement_duty.peak_ratio*100:5.1f}% at peak")
    lines.append("")

    # Effective power
    avg_mult = profile.estimate_average_power_multiplier()
    effective_power = tier.typical_power_w * avg_mult
    lines.append(f"  Power Analysis:")
    lines.append(f"    Average power multiplier: {avg_mult:.2f}x")
    lines.append(f"    Effective average power:  {effective_power:.1f}W (vs {tier.typical_power_w:.1f}W peak)")
    lines.append("")

    # Constraints
    if profile.constraints:
        lines.append("  Operational Constraints:")
        for constraint in profile.constraints:
            lines.append(f"    - {constraint}")
        lines.append("")

    return "\n".join(lines)


def format_tier_comparison() -> str:
    """Format a comparison of power allocations across all tiers."""
    lines = []
    lines.append("=" * 80)
    lines.append("  POWER ALLOCATION COMPARISON ACROSS TIERS")
    lines.append("=" * 80)
    lines.append("")

    # Table header
    lines.append("  Tier                | Power    | Perception | Control | Movement | Overhead")
    lines.append("  " + "-" * 75)

    for tier in CAPABILITY_TIERS_ORDERED:
        allocation = get_typical_allocation(tier.name)
        lines.append(
            f"  {tier.display_name:<19} | {tier.power_range_str:>8} | "
            f"{allocation.perception_ratio*100:>9.0f}% | {allocation.control_ratio*100:>6.0f}% | "
            f"{allocation.movement_ratio*100:>7.0f}% | {allocation.overhead_ratio*100:>7.0f}%"
        )

    lines.append("")
    lines.append("  Key Observations:")
    lines.append("    - Perception is highest in automotive (sensor fusion demands)")
    lines.append("    - Movement dominates in embodied AI (legged locomotion)")
    lines.append("    - Control increases with safety-critical requirements")
    lines.append("    - Overhead decreases as power budgets increase")
    lines.append("")

    # Visual comparison
    lines.append("  Visual Comparison (Perception/Control/Movement/Overhead):")
    lines.append("")
    for tier in CAPABILITY_TIERS_ORDERED:
        allocation = get_typical_allocation(tier.name)
        p_bar = "#" * int(allocation.perception_ratio * 20)
        c_bar = "=" * int(allocation.control_ratio * 20)
        m_bar = "*" * int(allocation.movement_ratio * 20)
        o_bar = "." * int(allocation.overhead_ratio * 20)
        lines.append(f"  {tier.display_name:<19}")
        lines.append(f"    P: [{p_bar:<20}] {allocation.perception_ratio*100:.0f}%")
        lines.append(f"    C: [{c_bar:<20}] {allocation.control_ratio*100:.0f}%")
        lines.append(f"    M: [{m_bar:<20}] {allocation.movement_ratio*100:.0f}%")
        lines.append(f"    O: [{o_bar:<20}] {allocation.overhead_ratio*100:.0f}%")
        lines.append("")

    return "\n".join(lines)


def format_custom_allocation(
    tier_name: TierName,
    perception: float,
    control: float,
    movement: float
) -> str:
    """Format analysis of a custom power allocation."""
    tier = CAPABILITY_TIERS[tier_name]
    typical = get_typical_allocation(tier_name)

    # Calculate overhead as remainder
    overhead = max(0.0, 1.0 - perception - control - movement)
    total = perception + control + movement + overhead

    # Normalize if needed
    if total > 0:
        perception /= total
        control /= total
        movement /= total
        overhead /= total

    custom = PowerAllocation(
        tier=tier_name,
        perception_ratio=perception,
        control_ratio=control,
        movement_ratio=movement,
        overhead_ratio=overhead,
    )

    lines = []
    lines.append("=" * 80)
    lines.append(f"  CUSTOM POWER ALLOCATION ANALYSIS FOR {tier.display_name.upper()}")
    lines.append("=" * 80)
    lines.append("")

    lines.append("  Custom Allocation:")
    lines.append(format_allocation_details(custom, tier.typical_power_w))
    lines.append("")

    lines.append("  Comparison with Typical Allocation:")
    lines.append("  " + "-" * 50)
    lines.append(f"  Subsystem    | Custom | Typical | Difference")
    lines.append(f"  Perception   | {custom.perception_ratio*100:5.1f}% | {typical.perception_ratio*100:6.1f}% | {(custom.perception_ratio-typical.perception_ratio)*100:+6.1f}%")
    lines.append(f"  Control      | {custom.control_ratio*100:5.1f}% | {typical.control_ratio*100:6.1f}% | {(custom.control_ratio-typical.control_ratio)*100:+6.1f}%")
    lines.append(f"  Movement     | {custom.movement_ratio*100:5.1f}% | {typical.movement_ratio*100:6.1f}% | {(custom.movement_ratio-typical.movement_ratio)*100:+6.1f}%")
    lines.append(f"  Overhead     | {custom.overhead_ratio*100:5.1f}% | {typical.overhead_ratio*100:6.1f}% | {(custom.overhead_ratio-typical.overhead_ratio)*100:+6.1f}%")
    lines.append("")

    # Warnings/recommendations
    lines.append("  Analysis:")
    warnings = []
    if custom.perception_ratio > typical.perception_ratio * 1.5:
        warnings.append("    ! High perception allocation may require specialized vision processors")
    if custom.perception_ratio < typical.perception_ratio * 0.5:
        warnings.append("    ! Low perception allocation may limit situational awareness")
    if custom.control_ratio > typical.control_ratio * 1.5:
        warnings.append("    ! High control allocation suggests complex decision making")
    if custom.control_ratio < typical.control_ratio * 0.5:
        warnings.append("    ! Low control allocation may limit autonomy capabilities")
    if custom.movement_ratio > typical.movement_ratio * 1.5:
        warnings.append("    ! High movement allocation typical for highly dynamic systems")
    if custom.movement_ratio < typical.movement_ratio * 0.5 and tier_name != TierName.WEARABLE_AI:
        warnings.append("    ! Low movement allocation suggests limited mobility")
    if custom.overhead_ratio < 0.05:
        warnings.append("    ! Very low overhead may be unrealistic - consider thermal management")
    if custom.overhead_ratio > 0.25:
        warnings.append("    ! High overhead indicates potential for optimization")

    if warnings:
        for w in warnings:
            lines.append(w)
    else:
        lines.append("    Allocation appears reasonable for this tier.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Explore power allocation across subsystems for embodied AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show allocation for a tier
  ./cli/explore_power_allocation.py --tier micro-autonomy

  # Show tier with mission profiles
  ./cli/explore_power_allocation.py --tier embodied-ai --show-missions

  # Explore a specific mission
  ./cli/explore_power_allocation.py --mission drone-inspection

  # Compare all tiers
  ./cli/explore_power_allocation.py --compare-tiers

  # Custom allocation
  ./cli/explore_power_allocation.py --tier micro-autonomy --perception 40 --control 30 --movement 20

  # JSON output
  ./cli/explore_power_allocation.py --tier micro-autonomy --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=list_tier_names(),
        help="Capability tier to analyze"
    )
    parser.add_argument(
        "--mission", "-m",
        type=str,
        choices=list(MISSION_PROFILES.keys()),
        help="Mission profile to analyze"
    )
    parser.add_argument(
        "--compare-tiers",
        action="store_true",
        help="Compare power allocations across all tiers"
    )
    parser.add_argument(
        "--show-missions",
        action="store_true",
        help="Show mission profiles for the tier"
    )
    parser.add_argument(
        "--perception",
        type=float,
        help="Custom perception allocation (0-100%%)"
    )
    parser.add_argument(
        "--control",
        type=float,
        help="Custom control allocation (0-100%%)"
    )
    parser.add_argument(
        "--movement",
        type=float,
        help="Custom movement allocation (0-100%%)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.tier and not args.mission and not args.compare_tiers:
        parser.print_help()
        print("\nError: Specify --tier, --mission, or --compare-tiers", file=sys.stderr)
        return 1

    # Handle custom allocation
    if args.perception is not None or args.control is not None or args.movement is not None:
        if not args.tier:
            print("Error: --tier required for custom allocation", file=sys.stderr)
            return 1
        tier_enum = TierName(args.tier)
        perception = (args.perception or 0) / 100.0
        control = (args.control or 0) / 100.0
        movement = (args.movement or 0) / 100.0

        if args.format == "json":
            overhead = max(0.0, 1.0 - perception - control - movement)
            output = {
                "tier": args.tier,
                "custom_allocation": {
                    "perception": perception,
                    "control": control,
                    "movement": movement,
                    "overhead": overhead,
                },
                "typical_allocation": get_typical_allocation(tier_enum).to_dict(),
            }
            print(json.dumps(output, indent=2))
        else:
            print(format_custom_allocation(tier_enum, perception, control, movement))
        return 0

    # Handle compare tiers
    if args.compare_tiers:
        if args.format == "json":
            output = {
                "tiers": [
                    {
                        "name": tier.name.value,
                        "display_name": tier.display_name,
                        "power_range": tier.power_range_str,
                        "allocation": get_typical_allocation(tier.name).to_dict(),
                    }
                    for tier in CAPABILITY_TIERS_ORDERED
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            print(format_tier_comparison())
        return 0

    # Handle mission profile
    if args.mission:
        profile = get_mission_profile(args.mission)
        if not profile:
            print(f"Error: Unknown mission profile '{args.mission}'", file=sys.stderr)
            return 1

        if args.format == "json":
            output = profile.to_dict()
            output["base_allocation"] = get_typical_allocation(profile.tier).to_dict()
            print(json.dumps(output, indent=2))
        else:
            print(format_mission_allocation(profile))
        return 0

    # Handle tier
    if args.tier:
        tier_enum = TierName(args.tier)
        if args.format == "json":
            tier = CAPABILITY_TIERS[tier_enum]
            output = {
                "tier": tier.to_dict(),
                "allocation": get_typical_allocation(tier_enum).to_dict(),
            }
            if args.show_missions:
                profiles = get_profiles_for_tier(tier_enum)
                output["mission_profiles"] = [p.to_dict() for p in profiles]
            print(json.dumps(output, indent=2))
        else:
            print(format_tier_allocation(tier_enum, show_missions=args.show_missions))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
