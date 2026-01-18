#!/usr/bin/env python3
"""
Explore Mission Profiles CLI Tool

Interactive exploration of mission profiles and their characteristics.
Shows duty cycles, power requirements, and suitable platforms.

Usage:
    # List all mission profiles
    ./cli/explore_mission_profiles.py --list

    # Show detailed profile
    ./cli/explore_mission_profiles.py --profile drone-inspection

    # Find profiles for a tier
    ./cli/explore_mission_profiles.py --tier micro-autonomy

    # Compare profiles
    ./cli/explore_mission_profiles.py --compare drone-inspection warehouse-amr

    # JSON output
    ./cli/explore_mission_profiles.py --profile drone-inspection --format json
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
    get_tier_by_name,
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.mission_profiles import (
    MissionProfile,
    DutyCycle,
    MISSION_PROFILES,
    get_mission_profile,
    get_profiles_for_tier,
)
from graphs.mission.power_allocation import (
    PowerAllocation,
    get_typical_allocation,
)


@dataclass
class ProfileAnalysis:
    """Analysis of a mission profile."""
    name: str
    profile: MissionProfile

    # Power analysis
    recommended_tier: str
    min_power_w: float
    typical_power_w: float
    max_power_w: float

    # Duty cycle analysis
    perception_duty: float
    control_duty: float
    movement_duty: float
    average_duty: float

    # Characteristics
    power_intensity: str  # low, medium, high
    mission_type: str  # inspection, navigation, manipulation, etc.
    suitable_platforms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.profile.description,
            "power": {
                "recommended_tier": self.recommended_tier,
                "min_w": self.min_power_w,
                "typical_w": self.typical_power_w,
                "max_w": self.max_power_w,
            },
            "duty_cycles": {
                "perception": self.perception_duty,
                "control": self.control_duty,
                "movement": self.movement_duty,
                "average": self.average_duty,
            },
            "characteristics": {
                "power_intensity": self.power_intensity,
                "mission_type": self.mission_type,
                "suitable_platforms": self.suitable_platforms,
            },
        }


def analyze_profile(name: str, profile: MissionProfile) -> ProfileAnalysis:
    """Analyze a mission profile."""
    # Get duty cycles
    perception_duty = profile.duty_cycle.perception
    control_duty = profile.duty_cycle.control
    movement_duty = profile.duty_cycle.movement
    average_duty = (perception_duty + control_duty + movement_duty) / 3

    # Determine power intensity
    if average_duty < 0.3:
        power_intensity = "low"
    elif average_duty < 0.6:
        power_intensity = "medium"
    else:
        power_intensity = "high"

    # Determine mission type based on dominant duty cycle
    if movement_duty > perception_duty and movement_duty > control_duty:
        mission_type = "locomotion-intensive"
    elif perception_duty > control_duty:
        mission_type = "perception-intensive"
    else:
        mission_type = "control-intensive"

    # Get recommended tier
    recommended_tier = profile.tier_name if hasattr(profile, 'tier_name') else "micro-autonomy"
    tier = get_tier_by_name(recommended_tier)

    if tier:
        min_power = tier.power_range[0]
        typical_power = tier.typical_power_w
        max_power = tier.power_range[1]
    else:
        min_power = 5.0
        typical_power = 15.0
        max_power = 30.0

    # Determine suitable platforms based on tier
    suitable_platforms = []
    if recommended_tier == "wearable-ai":
        suitable_platforms = ["smartwatch", "AR glasses", "hearing aid"]
    elif recommended_tier == "micro-autonomy":
        suitable_platforms = ["nano-drone", "inspection robot", "small AGV"]
    elif recommended_tier == "industrial-edge":
        suitable_platforms = ["warehouse AMR", "collaborative robot", "inspection drone"]
    elif recommended_tier == "embodied-ai":
        suitable_platforms = ["humanoid robot", "quadruped", "heavy-lift drone"]
    elif recommended_tier == "automotive-ai":
        suitable_platforms = ["autonomous vehicle", "heavy equipment", "delivery robot"]

    return ProfileAnalysis(
        name=name,
        profile=profile,
        recommended_tier=recommended_tier,
        min_power_w=min_power,
        typical_power_w=typical_power,
        max_power_w=max_power,
        perception_duty=perception_duty,
        control_duty=control_duty,
        movement_duty=movement_duty,
        average_duty=average_duty,
        power_intensity=power_intensity,
        mission_type=mission_type,
        suitable_platforms=suitable_platforms,
    )


def format_profile_list() -> str:
    """Format list of all profiles."""
    lines = []
    lines.append("=" * 80)
    lines.append("  AVAILABLE MISSION PROFILES")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"  {'Profile':<25} {'Tier':<18} {'Type':<20}")
    lines.append("  " + "-" * 70)

    for name, profile in sorted(MISSION_PROFILES.items()):
        analysis = analyze_profile(name, profile)
        lines.append(
            f"  {name:<25} {analysis.recommended_tier:<18} {analysis.mission_type:<20}"
        )

    lines.append("  " + "-" * 70)
    lines.append(f"  Total: {len(MISSION_PROFILES)} profiles")
    lines.append("")

    return "\n".join(lines)


def format_profile_detail(analysis: ProfileAnalysis) -> str:
    """Format detailed profile analysis."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"  MISSION PROFILE: {analysis.name.upper()}")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"  Description: {analysis.profile.description}")
    lines.append("")

    # Power requirements
    lines.append("  Power Requirements:")
    lines.append(f"    Recommended Tier:  {analysis.recommended_tier}")
    lines.append(f"    Power Range:       {analysis.min_power_w:.1f}W - {analysis.max_power_w:.1f}W")
    lines.append(f"    Typical Power:     {analysis.typical_power_w:.1f}W")
    lines.append("")

    # Duty cycles
    lines.append("  Duty Cycles:")
    lines.append(f"    Perception:  {analysis.perception_duty*100:>5.1f}%  {'#' * int(analysis.perception_duty * 30)}")
    lines.append(f"    Control:     {analysis.control_duty*100:>5.1f}%  {'#' * int(analysis.control_duty * 30)}")
    lines.append(f"    Movement:    {analysis.movement_duty*100:>5.1f}%  {'#' * int(analysis.movement_duty * 30)}")
    lines.append(f"    Average:     {analysis.average_duty*100:>5.1f}%")
    lines.append("")

    # Characteristics
    lines.append("  Characteristics:")
    lines.append(f"    Power Intensity:   {analysis.power_intensity}")
    lines.append(f"    Mission Type:      {analysis.mission_type}")
    lines.append("")

    # Suitable platforms
    lines.append("  Suitable Platforms:")
    for platform in analysis.suitable_platforms:
        lines.append(f"    - {platform}")
    lines.append("")

    return "\n".join(lines)


def format_profile_comparison(analyses: List[ProfileAnalysis]) -> str:
    """Format comparison of multiple profiles."""
    lines = []

    lines.append("=" * 80)
    lines.append("  MISSION PROFILE COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Power comparison
    lines.append("  Power Requirements:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Profile':<25} {'Tier':<18} {'Typical Power':>12}")
    lines.append("  " + "-" * 70)
    for a in analyses:
        lines.append(f"    {a.name:<25} {a.recommended_tier:<18} {a.typical_power_w:>10.1f}W")
    lines.append("  " + "-" * 70)
    lines.append("")

    # Duty cycle comparison
    lines.append("  Duty Cycles:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Profile':<25} {'Percep':>8} {'Control':>8} {'Move':>8} {'Avg':>8}")
    lines.append("  " + "-" * 70)
    for a in analyses:
        lines.append(
            f"    {a.name:<25} {a.perception_duty*100:>7.0f}% {a.control_duty*100:>7.0f}% "
            f"{a.movement_duty*100:>7.0f}% {a.average_duty*100:>7.0f}%"
        )
    lines.append("  " + "-" * 70)
    lines.append("")

    # Visual comparison
    lines.append("  Visual Comparison (duty cycle distribution):")
    lines.append("")
    bar_width = 40
    for a in analyses:
        p_len = int(a.perception_duty * bar_width)
        c_len = int(a.control_duty * bar_width)
        m_len = int(a.movement_duty * bar_width)
        bar = "P" * p_len + "C" * c_len + "M" * m_len
        bar = bar[:bar_width].ljust(bar_width, ".")
        lines.append(f"    {a.name:<25} [{bar}]")
    lines.append("")
    lines.append("    Legend: P=Perception, C=Control, M=Movement, .=Idle")
    lines.append("")

    # Mission type comparison
    lines.append("  Mission Types:")
    for a in analyses:
        lines.append(f"    {a.name:<25} {a.mission_type}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Explore mission profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all profiles
  ./cli/explore_mission_profiles.py --list

  # Show detailed profile
  ./cli/explore_mission_profiles.py --profile drone-inspection

  # Find profiles for a tier
  ./cli/explore_mission_profiles.py --tier micro-autonomy

  # Compare profiles
  ./cli/explore_mission_profiles.py --compare drone-inspection warehouse-amr

  # JSON output
  ./cli/explore_mission_profiles.py --profile drone-inspection --format json
"""
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available profiles"
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        choices=list(MISSION_PROFILES.keys()),
        help="Show detailed profile information"
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Find profiles suitable for a tier"
    )
    parser.add_argument(
        "--compare", "-c",
        type=str,
        nargs="+",
        help="Compare multiple profiles"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list:
        if args.format == "json":
            result = {
                "profiles": [
                    analyze_profile(name, profile).to_dict()
                    for name, profile in MISSION_PROFILES.items()
                ]
            }
            print(json.dumps(result, indent=2))
        else:
            print(format_profile_list())
        return 0

    # Handle profile detail
    if args.profile:
        profile = get_mission_profile(args.profile)
        if profile is None:
            print(f"Error: Unknown profile: {args.profile}", file=sys.stderr)
            return 1

        analysis = analyze_profile(args.profile, profile)

        if args.format == "json":
            print(json.dumps(analysis.to_dict(), indent=2))
        else:
            print(format_profile_detail(analysis))
        return 0

    # Handle tier filter
    if args.tier:
        profiles = get_profiles_for_tier(args.tier)
        if args.format == "json":
            result = {
                "tier": args.tier,
                "profiles": [
                    analyze_profile(name, profile).to_dict()
                    for name, profile in profiles.items()
                ]
            }
            print(json.dumps(result, indent=2))
        else:
            lines = []
            lines.append("=" * 80)
            lines.append(f"  PROFILES FOR TIER: {args.tier.upper()}")
            lines.append("=" * 80)
            lines.append("")

            if profiles:
                for name, profile in profiles.items():
                    analysis = analyze_profile(name, profile)
                    lines.append(f"  {name}:")
                    lines.append(f"    Type: {analysis.mission_type}")
                    lines.append(f"    Avg Duty: {analysis.average_duty*100:.0f}%")
                    lines.append("")
            else:
                lines.append("  No profiles found for this tier.")
                lines.append("")

            print("\n".join(lines))
        return 0

    # Handle comparison
    if args.compare:
        analyses = []
        for name in args.compare:
            profile = get_mission_profile(name)
            if profile is None:
                print(f"Error: Unknown profile: {name}", file=sys.stderr)
                return 1
            analyses.append(analyze_profile(name, profile))

        if args.format == "json":
            result = {
                "comparison": [a.to_dict() for a in analyses]
            }
            print(json.dumps(result, indent=2))
        else:
            print(format_profile_comparison(analyses))
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
