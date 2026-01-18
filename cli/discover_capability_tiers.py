#!/usr/bin/env python3
"""
Discover Capability Tiers CLI Tool

List and explore capability tiers for embodied AI systems. Each tier
represents a class of systems with similar power envelopes and application
domains.

Capability Tiers:
    - Wearable AI (0.1-1W): Smartwatches, AR glasses, health monitors
    - Micro-Autonomy (1-10W): Drones, handheld scanners, smart IoT
    - Industrial Edge (10-30W): Factory AMRs, cobots, automated sorting
    - Embodied AI (30-100W+): Quadrupeds, humanoids, world-model simulation
    - Automotive AI (100-500W): L3/L3+/L4 autonomous driving

Usage:
    # List all tiers
    ./cli/discover_capability_tiers.py

    # Show details for specific tier
    ./cli/discover_capability_tiers.py --tier embodied-ai

    # List platforms in a tier
    ./cli/discover_capability_tiers.py --tier industrial-edge --show-platforms

    # List mission profiles for a tier
    ./cli/discover_capability_tiers.py --tier micro-autonomy --show-missions

    # JSON output
    ./cli/discover_capability_tiers.py --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.capability_tiers import (
    CapabilityTier,
    CAPABILITY_TIERS_ORDERED,
    get_tier_by_name,
    list_tier_names,
)
from graphs.mission.mission_profiles import (
    get_profiles_for_tier,
    MissionProfile,
)
from graphs.mission.power_allocation import (
    get_typical_allocation,
    PowerAllocation,
)
from graphs.mission.battery import (
    find_batteries_for_tier,
    BatteryConfiguration,
)


def format_tier_summary(tier: CapabilityTier) -> str:
    """Format a single tier as a summary line."""
    return (
        f"  {tier.display_name:<20} | {tier.power_range_str:>12} | "
        f"{len(tier.typical_applications)} applications"
    )


def format_tier_details(tier: CapabilityTier, show_platforms: bool = False,
                        show_missions: bool = False, show_batteries: bool = False) -> str:
    """Format detailed information about a tier."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  {tier.display_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Power Envelope:    {tier.power_range_str}")
    lines.append(f"  Typical Power:     {tier.typical_power_w:.1f}W")
    lines.append(f"  Mission Duration:  {tier.typical_mission_hours[0]:.1f}-{tier.typical_mission_hours[1]:.1f} hours")
    lines.append("")
    lines.append("  Description:")
    # Word wrap description
    desc_words = tier.description.split()
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

    # Typical applications
    lines.append("  Typical Applications:")
    for app in tier.typical_applications:
        lines.append(f"    - {app}")
    lines.append("")

    # Thermal constraints
    lines.append("  Thermal Constraints:")
    lines.append(f"    Max Ambient:     {tier.thermal.max_ambient_temp_c}C")
    lines.append(f"    Max Junction:    {tier.thermal.max_junction_temp_c}C")
    lines.append(f"    Typical Cooling: {tier.thermal.typical_cooling}")
    lines.append(f"    Sustained Power: {tier.thermal.sustained_power_derating*100:.0f}% of peak")
    lines.append("")

    # Power allocation
    allocation = get_typical_allocation(tier.name)
    lines.append("  Typical Power Allocation:")
    lines.append(f"    Perception: {allocation.perception_ratio*100:5.1f}%")
    lines.append(f"    Control:    {allocation.control_ratio*100:5.1f}%")
    lines.append(f"    Movement:   {allocation.movement_ratio*100:5.1f}%")
    lines.append(f"    Overhead:   {allocation.overhead_ratio*100:5.1f}%")
    lines.append("")

    # Example platforms
    if show_platforms and tier.example_platforms:
        lines.append("  Example Platforms:")
        for platform in tier.example_platforms:
            lines.append(f"    - {platform}")
        lines.append("")

    # Mission profiles
    if show_missions:
        profiles = get_profiles_for_tier(tier.name)
        if profiles:
            lines.append("  Mission Profiles:")
            for profile in profiles:
                lines.append(f"    - {profile.display_name} ({profile.typical_duration_hours:.1f}h)")
                lines.append(f"      {profile.description[:60]}...")
            lines.append("")

    # Battery options
    if show_batteries:
        batteries = find_batteries_for_tier(tier.name)
        if batteries:
            lines.append("  Battery Options:")
            for battery in batteries:
                runtime = battery.estimate_runtime_hours(tier.typical_power_w)
                lines.append(f"    - {battery.name}: {battery.capacity_wh:.0f}Wh, {battery.weight_kg:.2f}kg")
                lines.append(f"      ~{runtime:.1f}h at typical power")
            lines.append("")

    return "\n".join(lines)


def format_all_tiers_table() -> str:
    """Format all tiers as a summary table."""
    lines = []
    lines.append("=" * 80)
    lines.append("  CAPABILITY TIERS FOR EMBODIED AI")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Tier                 | Power Range  | Applications")
    lines.append("  " + "-" * 60)

    for tier in CAPABILITY_TIERS_ORDERED:
        lines.append(format_tier_summary(tier))

    lines.append("")
    lines.append("  Use --tier <name> for details, e.g.:")
    lines.append("    ./cli/discover_capability_tiers.py --tier micro-autonomy")
    lines.append("")

    return "\n".join(lines)


def format_json_output(tier: Optional[CapabilityTier] = None) -> str:
    """Format output as JSON."""
    if tier:
        data = tier.to_dict()
        # Add additional context
        allocation = get_typical_allocation(tier.name)
        data["typical_power_allocation"] = allocation.to_dict()

        profiles = get_profiles_for_tier(tier.name)
        data["mission_profiles"] = [p.name for p in profiles]

        batteries = find_batteries_for_tier(tier.name)
        data["battery_options"] = [b.name for b in batteries]
    else:
        data = {
            "capability_tiers": [t.to_dict() for t in CAPABILITY_TIERS_ORDERED]
        }

    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Discover capability tiers for embodied AI systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capability Tiers:
  wearable-ai      0.1-1W    Smartwatches, AR glasses, health monitors
  micro-autonomy   1-10W     Drones, handheld scanners, smart IoT
  industrial-edge  10-30W    Factory AMRs, cobots, automated sorting
  embodied-ai      30-100W+  Quadrupeds, humanoids, world-model simulation
  automotive-ai    100-500W  L3/L3+/L4 autonomous driving

Examples:
  # List all tiers
  ./cli/discover_capability_tiers.py

  # Show details for a tier
  ./cli/discover_capability_tiers.py --tier micro-autonomy

  # Show with platforms and missions
  ./cli/discover_capability_tiers.py --tier embodied-ai --show-platforms --show-missions

  # JSON output for integration
  ./cli/discover_capability_tiers.py --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=list_tier_names(),
        help="Show details for a specific tier"
    )
    parser.add_argument(
        "--show-platforms", "-p",
        action="store_true",
        help="Show example platforms for the tier"
    )
    parser.add_argument(
        "--show-missions", "-m",
        action="store_true",
        help="Show mission profiles for the tier"
    )
    parser.add_argument(
        "--show-batteries", "-b",
        action="store_true",
        help="Show battery options for the tier"
    )
    parser.add_argument(
        "--show-all", "-a",
        action="store_true",
        help="Show all details (platforms, missions, batteries)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Handle --show-all
    if args.show_all:
        args.show_platforms = True
        args.show_missions = True
        args.show_batteries = True

    # Generate output
    if args.format == "json":
        tier = get_tier_by_name(args.tier) if args.tier else None
        print(format_json_output(tier))
    elif args.tier:
        tier = get_tier_by_name(args.tier)
        if tier:
            print(format_tier_details(
                tier,
                show_platforms=args.show_platforms,
                show_missions=args.show_missions,
                show_batteries=args.show_batteries
            ))
        else:
            print(f"Error: Unknown tier '{args.tier}'", file=sys.stderr)
            print(f"Available tiers: {', '.join(list_tier_names())}", file=sys.stderr)
            return 1
    else:
        print(format_all_tiers_table())

    return 0


if __name__ == "__main__":
    sys.exit(main())
