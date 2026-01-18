#!/usr/bin/env python3
"""
Compare Mission Configurations CLI Tool

Compare different hardware/model configurations for the same mission requirements.
Shows tradeoffs between runtime, weight, and capability.

Usage:
    # Auto-generate and compare configurations for a mission profile
    ./cli/compare_mission_configurations.py --profile warehouse-amr --tier industrial-edge

    # Compare specific platforms for a mission
    ./cli/compare_mission_configurations.py --profile drone-inspection \\
        --platforms Jetson-Orin-Nano-8GB Hailo-8 Coral-Edge-TPU

    # With specific model
    ./cli/compare_mission_configurations.py --profile quadruped-patrol \\
        --tier embodied-ai --model yolov8n

    # JSON output
    ./cli/compare_mission_configurations.py --profile warehouse-amr --tier industrial-edge --format json
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

from graphs.hardware.mappers import (
    list_all_mappers,
    get_mapper_info,
    list_mappers_by_tdp_range,
)
from graphs.mission.capability_tiers import (
    TierName,
    get_tier_by_name,
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.power_allocation import (
    get_typical_allocation,
)
from graphs.mission.battery import (
    BatteryConfiguration,
    BATTERY_CONFIGURATIONS,
    find_batteries_for_mission,
    find_batteries_for_tier,
)
from graphs.mission.mission_profiles import (
    MissionProfile,
    MISSION_PROFILES,
    get_mission_profile,
    get_profiles_for_tier,
)


@dataclass
class ConfigurationAnalysis:
    """Analysis of a single mission configuration."""
    platform_name: str
    model_name: str
    battery_name: str

    # Platform specs
    platform_tdp_w: float
    platform_memory_gb: float

    # Power estimates
    estimated_power_w: float
    perception_power_w: float
    movement_power_w: float

    # Mission performance
    estimated_runtime_hours: float
    battery_capacity_wh: float
    battery_weight_kg: float

    # Capability scores (0-100)
    perception_score: float = 0.0
    runtime_score: float = 0.0
    weight_score: float = 0.0
    overall_score: float = 0.0

    # Ranking
    rank: int = 0

    # Notes
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "model": self.model_name,
            "battery": self.battery_name,
            "specs": {
                "platform_tdp_w": self.platform_tdp_w,
                "platform_memory_gb": self.platform_memory_gb,
                "estimated_power_w": self.estimated_power_w,
                "battery_capacity_wh": self.battery_capacity_wh,
                "battery_weight_kg": self.battery_weight_kg,
            },
            "performance": {
                "estimated_runtime_hours": self.estimated_runtime_hours,
                "perception_power_w": self.perception_power_w,
            },
            "scores": {
                "perception": self.perception_score,
                "runtime": self.runtime_score,
                "weight": self.weight_score,
                "overall": self.overall_score,
            },
            "rank": self.rank,
            "warnings": self.warnings,
        }


@dataclass
class MissionComparisonResult:
    """Result of comparing configurations for a mission."""
    profile_name: str
    profile_description: str
    target_duration_hours: float
    tier_name: str

    configurations: List[ConfigurationAnalysis]

    # Best options
    best_overall: Optional[str] = None
    best_runtime: Optional[str] = None
    lightest_weight: Optional[str] = None
    best_perception: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": {
                "profile": self.profile_name,
                "description": self.profile_description,
                "target_duration_hours": self.target_duration_hours,
                "tier": self.tier_name,
            },
            "configurations": [c.to_dict() for c in self.configurations],
            "recommendations": {
                "best_overall": self.best_overall,
                "best_runtime": self.best_runtime,
                "lightest_weight": self.lightest_weight,
                "best_perception": self.best_perception,
            },
        }


def estimate_power(platform_name: str, model_name: str, profile: MissionProfile) -> Dict[str, float]:
    """Estimate power consumption for a configuration."""
    info = get_mapper_info(platform_name)
    if info is None:
        return {"total_w": 0.0, "perception_w": 0.0}

    tdp = info["default_tdp_w"]

    # Model complexity factor
    model_lower = model_name.lower()
    if "resnet50" in model_lower:
        complexity = 0.70
    elif "resnet18" in model_lower:
        complexity = 0.45
    elif "yolov8n" in model_lower:
        complexity = 0.30
    elif "yolov8s" in model_lower:
        complexity = 0.40
    elif "mobilenet" in model_lower:
        complexity = 0.25
    else:
        complexity = 0.50

    # Compute power
    compute_power = tdp * 0.7 * complexity

    # Get tier allocation
    tier = get_tier_for_power(tdp)
    if tier:
        allocation = get_typical_allocation(tier.name)
        movement_power = tier.typical_power_w * allocation.movement_ratio
    else:
        movement_power = tdp * 0.3

    # Apply duty cycles
    effective_perception = compute_power * profile.perception_duty.effective_ratio
    effective_movement = movement_power * profile.movement_duty.effective_ratio
    overhead = compute_power * 0.12

    total_power = effective_perception + effective_movement + overhead

    return {
        "total_w": total_power,
        "perception_w": effective_perception,
        "movement_w": effective_movement,
    }


def analyze_configuration(
    platform_name: str,
    model_name: str,
    profile: MissionProfile,
    battery: Optional[BatteryConfiguration] = None,
) -> ConfigurationAnalysis:
    """Analyze a single configuration for a mission."""
    info = get_mapper_info(platform_name)
    if info is None:
        raise ValueError(f"Unknown platform: {platform_name}")

    # Estimate power
    power = estimate_power(platform_name, model_name, profile)

    # Find suitable battery if not specified
    if battery is None:
        batteries = find_batteries_for_tier(profile.tier)
        if batteries:
            battery = batteries[0]  # Take lightest suitable battery
        else:
            # Create a synthetic battery estimate
            required_wh = power["total_w"] * profile.typical_duration_hours * 1.2
            battery = BatteryConfiguration(
                name="estimated",
                chemistry=None,
                capacity_wh=required_wh,
                voltage_nominal=24.0,
                weight_kg=required_wh / 150,  # Assume 150 Wh/kg
                volume_cm3=required_wh * 3,
            )

    # Calculate runtime
    if power["total_w"] > 0:
        runtime = battery.estimate_runtime_hours(power["total_w"], safety_margin=0.9)
    else:
        runtime = 0.0

    # Warnings
    warnings = []
    if runtime < profile.typical_duration_hours:
        deficit = profile.typical_duration_hours - runtime
        warnings.append(f"Runtime {deficit:.1f}h short of target")
    if power["total_w"] > info["default_tdp_w"]:
        warnings.append("Power exceeds platform TDP")

    return ConfigurationAnalysis(
        platform_name=platform_name,
        model_name=model_name,
        battery_name=battery.name,
        platform_tdp_w=info["default_tdp_w"],
        platform_memory_gb=info["memory_gb"],
        estimated_power_w=power["total_w"],
        perception_power_w=power["perception_w"],
        movement_power_w=power["movement_w"],
        estimated_runtime_hours=runtime,
        battery_capacity_wh=battery.capacity_wh,
        battery_weight_kg=battery.weight_kg,
        warnings=warnings,
    )


def compare_mission_configurations(
    profile_name: str,
    tier_name: Optional[str] = None,
    platform_names: Optional[List[str]] = None,
    model_name: str = "yolov8n",
) -> MissionComparisonResult:
    """
    Compare configurations for a mission profile.

    Args:
        profile_name: Mission profile name
        tier_name: Capability tier (for auto-selecting platforms)
        platform_names: Specific platforms to compare
        model_name: Model to use for estimation

    Returns:
        MissionComparisonResult with ranked configurations
    """
    profile = get_mission_profile(profile_name)
    if profile is None:
        raise ValueError(f"Unknown mission profile: {profile_name}")

    # Determine tier
    if tier_name:
        tier = get_tier_by_name(tier_name)
    else:
        tier = CAPABILITY_TIERS.get(profile.tier)

    if tier is None:
        raise ValueError(f"Could not determine tier for profile")

    # Get platforms to compare
    if platform_names:
        platforms = platform_names
    else:
        # Auto-select platforms from tier
        platforms = list_mappers_by_tdp_range(tier.power_min_w, tier.power_max_w)[:8]  # Top 8

    # Analyze each configuration
    configurations = []
    for platform in platforms:
        try:
            config = analyze_configuration(platform, model_name, profile)
            configurations.append(config)
        except ValueError:
            continue

    if not configurations:
        return MissionComparisonResult(
            profile_name=profile_name,
            profile_description=profile.description,
            target_duration_hours=profile.typical_duration_hours,
            tier_name=tier.name.value,
            configurations=[],
        )

    # Calculate scores (0-100)
    max_runtime = max(c.estimated_runtime_hours for c in configurations)
    min_weight = min(c.battery_weight_kg for c in configurations)
    max_perception = max(c.perception_power_w for c in configurations)

    for c in configurations:
        # Runtime score: how well does it meet target duration?
        if profile.typical_duration_hours > 0:
            runtime_ratio = c.estimated_runtime_hours / profile.typical_duration_hours
            c.runtime_score = min(100, runtime_ratio * 100)
        else:
            c.runtime_score = 100 if c.estimated_runtime_hours > 0 else 0

        # Weight score: lighter is better (inverse)
        if min_weight > 0:
            c.weight_score = (min_weight / c.battery_weight_kg) * 100
        else:
            c.weight_score = 100

        # Perception score: more perception power is better
        if max_perception > 0:
            c.perception_score = (c.perception_power_w / max_perception) * 100
        else:
            c.perception_score = 0

        # Overall score: weighted average
        c.overall_score = (
            c.runtime_score * 0.4 +
            c.weight_score * 0.3 +
            c.perception_score * 0.3
        )

    # Rank by overall score
    configurations.sort(key=lambda x: -x.overall_score)
    for i, c in enumerate(configurations):
        c.rank = i + 1

    # Determine best options
    best_overall = configurations[0].platform_name if configurations else None
    best_runtime = max(configurations, key=lambda x: x.estimated_runtime_hours).platform_name
    lightest = min(configurations, key=lambda x: x.battery_weight_kg).platform_name
    best_perception = max(configurations, key=lambda x: x.perception_power_w).platform_name

    return MissionComparisonResult(
        profile_name=profile_name,
        profile_description=profile.description,
        target_duration_hours=profile.typical_duration_hours,
        tier_name=tier.name.value,
        configurations=configurations,
        best_overall=best_overall,
        best_runtime=best_runtime,
        lightest_weight=lightest,
        best_perception=best_perception,
    )


def format_comparison(result: MissionComparisonResult, verbose: bool = False) -> str:
    """Format comparison result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  MISSION CONFIGURATION COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Mission info
    lines.append("  Mission Profile:")
    lines.append(f"    Name:            {result.profile_name}")
    lines.append(f"    Description:     {result.profile_description[:60]}...")
    lines.append(f"    Target Duration: {result.target_duration_hours:.1f} hours")
    lines.append(f"    Tier:            {result.tier_name}")
    lines.append("")

    if not result.configurations:
        lines.append("  No configurations found for this mission.")
        lines.append("")
        return "\n".join(lines)

    # Configuration comparison table
    lines.append("  Configuration Comparison:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Rank':>4} {'Platform':<22} {'Runtime':>8} {'Battery':>10} {'Score':>8} {'Warnings':>10}")
    lines.append("  " + "-" * 76)

    for c in result.configurations:
        runtime_str = f"{c.estimated_runtime_hours:.1f}h"
        battery_str = f"{c.battery_capacity_wh:.0f}Wh"
        score_str = f"{c.overall_score:.0f}"
        warn_str = f"{len(c.warnings)}" if c.warnings else "-"

        lines.append(
            f"    #{c.rank:<3} {c.platform_name:<22} {runtime_str:>8} {battery_str:>10} {score_str:>8} {warn_str:>10}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Score breakdown
    lines.append("  Score Breakdown (0-100):")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Platform':<22} {'Runtime':>10} {'Weight':>10} {'Percep':>10} {'Overall':>10}")
    lines.append("  " + "-" * 60)

    for c in result.configurations[:5]:  # Top 5
        lines.append(
            f"    {c.platform_name:<22} {c.runtime_score:>9.0f} {c.weight_score:>9.0f} "
            f"{c.perception_score:>9.0f} {c.overall_score:>9.0f}"
        )

    lines.append("  " + "-" * 60)
    lines.append("")

    # Recommendations
    lines.append("  Recommendations:")
    lines.append(f"    Best Overall:     {result.best_overall}")
    lines.append(f"    Best Runtime:     {result.best_runtime}")
    lines.append(f"    Lightest Weight:  {result.lightest_weight}")
    lines.append(f"    Best Perception:  {result.best_perception}")
    lines.append("")

    # Warnings (if verbose)
    if verbose:
        configs_with_warnings = [c for c in result.configurations if c.warnings]
        if configs_with_warnings:
            lines.append("  Configuration Warnings:")
            for c in configs_with_warnings:
                lines.append(f"    {c.platform_name}:")
                for w in c.warnings:
                    lines.append(f"      - {w}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare configurations for a mission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare configurations for warehouse AMR
  ./cli/compare_mission_configurations.py --profile warehouse-amr --tier industrial-edge

  # Compare specific platforms
  ./cli/compare_mission_configurations.py --profile drone-inspection \\
      --platforms Jetson-Orin-Nano-8GB Hailo-8

  # With specific model
  ./cli/compare_mission_configurations.py --profile quadruped-patrol \\
      --tier embodied-ai --model resnet18

  # JSON output
  ./cli/compare_mission_configurations.py --profile warehouse-amr --format json
"""
    )

    parser.add_argument(
        "--profile", "-p",
        type=str,
        required=True,
        choices=list(MISSION_PROFILES.keys()),
        help="Mission profile"
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Capability tier for platform selection"
    )
    parser.add_argument(
        "--platforms",
        type=str,
        nargs="+",
        help="Specific platforms to compare"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n",
        help="Model for estimation (default: yolov8n)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including warnings"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available mission profiles"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list_profiles:
        print("Available mission profiles:")
        for name, profile in MISSION_PROFILES.items():
            print(f"  {name:<25} {profile.tier.value:<18} {profile.typical_duration_hours:.1f}h")
        return 0

    # Run comparison
    try:
        result = compare_mission_configurations(
            profile_name=args.profile,
            tier_name=args.tier,
            platform_names=args.platforms,
            model_name=args.model,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_comparison(result, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
