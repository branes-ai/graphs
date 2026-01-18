#!/usr/bin/env python3
"""
Estimate Battery Requirements CLI Tool

Estimate the battery size needed to meet mission duration requirements.
Finds suitable batteries from the database or calculates custom requirements.

Usage:
    # Basic battery sizing
    ./cli/estimate_battery_requirements.py --platform Jetson-Orin-Nano-8GB --model yolov8n --mission-hours 4

    # With safety margin
    ./cli/estimate_battery_requirements.py --platform Jetson-Orin-NX-16GB --model resnet18 \\
        --mission-hours 4 --safety-margin 1.2

    # With weight constraint
    ./cli/estimate_battery_requirements.py --tier micro-autonomy --mission-hours 2 --max-weight-kg 0.5

    # With mission profile
    ./cli/estimate_battery_requirements.py --platform Jetson-Orin-NX-16GB --model yolov8n \\
        --mission-hours 8 --profile warehouse-amr
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
)
from graphs.mission.capability_tiers import (
    TierName,
    get_tier_by_name,
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.power_allocation import (
    get_typical_allocation,
    ALLOCATION_BALANCED,
)
from graphs.mission.battery import (
    BatteryConfiguration,
    BATTERY_CONFIGURATIONS,
    find_batteries_for_mission,
    estimate_battery_weight,
)
from graphs.mission.mission_profiles import (
    MissionProfile,
    MISSION_PROFILES,
    get_mission_profile,
)


@dataclass
class BatteryRequirement:
    """Battery requirement estimate result."""
    platform_name: str
    model_name: str
    mission_hours: float

    # Power estimates
    perception_power_w: float
    control_power_w: float
    movement_power_w: float
    overhead_power_w: float
    total_power_w: float
    effective_power_w: float  # After duty cycle

    # Battery requirements
    required_capacity_wh: float
    recommended_capacity_wh: float  # With safety margin
    safety_margin: float

    # Weight/volume estimates
    estimated_weight_kg: float
    estimated_volume_cm3: float

    # Recommendations
    recommended_batteries: List[Dict[str, Any]] = field(default_factory=list)

    # Mission profile
    profile_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "model": self.model_name,
            "mission_hours": self.mission_hours,
            "power": {
                "perception_w": self.perception_power_w,
                "control_w": self.control_power_w,
                "movement_w": self.movement_power_w,
                "overhead_w": self.overhead_power_w,
                "total_w": self.total_power_w,
                "effective_w": self.effective_power_w,
            },
            "battery": {
                "required_capacity_wh": self.required_capacity_wh,
                "recommended_capacity_wh": self.recommended_capacity_wh,
                "safety_margin": self.safety_margin,
                "estimated_weight_kg": self.estimated_weight_kg,
                "estimated_volume_cm3": self.estimated_volume_cm3,
            },
            "recommendations": self.recommended_batteries,
            "mission_profile": self.profile_name,
        }


def estimate_compute_power(platform_name: str, model_name: str) -> Dict[str, float]:
    """Estimate compute power for a model on a platform."""
    info = get_mapper_info(platform_name)
    if info is None:
        return {"compute_power_w": 5.0, "tdp_w": 10.0}  # Fallback

    tdp = info["default_tdp_w"]

    # Typical utilization
    if tdp <= 30:
        typical_ratio = 0.75
    elif tdp <= 100:
        typical_ratio = 0.60
    else:
        typical_ratio = 0.50

    # Model complexity
    model_lower = model_name.lower()
    if "resnet50" in model_lower or "resnet101" in model_lower:
        complexity = 0.70
    elif "resnet18" in model_lower or "resnet34" in model_lower:
        complexity = 0.45
    elif "yolo" in model_lower:
        complexity = 0.40
    elif "mobilenet" in model_lower:
        complexity = 0.25
    else:
        complexity = 0.50

    return {
        "compute_power_w": tdp * typical_ratio * complexity,
        "tdp_w": tdp,
    }


def estimate_battery_requirements(
    platform_name: str,
    model_name: str,
    mission_hours: float,
    profile: Optional[MissionProfile] = None,
    movement_power_w: float = 0.0,
    safety_margin: float = 1.2,
    max_weight_kg: Optional[float] = None,
) -> BatteryRequirement:
    """
    Estimate battery requirements for a mission.

    Args:
        platform_name: Hardware platform name
        model_name: Model name
        mission_hours: Required mission duration
        profile: Optional mission profile for duty cycles
        movement_power_w: External movement power
        safety_margin: Capacity multiplier for safety (default 1.2 = 20% extra)
        max_weight_kg: Optional weight constraint

    Returns:
        BatteryRequirement with capacity and recommendations
    """
    # Get power estimates
    power_info = estimate_compute_power(platform_name, model_name)
    compute_power = power_info["compute_power_w"]
    tdp = power_info["tdp_w"]

    # Get allocation
    tier = get_tier_for_power(tdp)
    allocation = get_typical_allocation(tier.name) if tier else ALLOCATION_BALANCED

    # Split compute power
    perception_ratio = allocation.perception_ratio / (allocation.perception_ratio + allocation.control_ratio)
    control_ratio = allocation.control_ratio / (allocation.perception_ratio + allocation.control_ratio)

    perception_power = compute_power * perception_ratio
    control_power = compute_power * control_ratio
    overhead_power = compute_power * 0.12

    # Movement power
    if movement_power_w <= 0 and tier:
        movement_power_w = tier.typical_power_w * allocation.movement_ratio

    # Total peak power
    total_power = perception_power + control_power + movement_power_w + overhead_power

    # Apply duty cycles if profile provided
    if profile:
        effective_power = (
            perception_power * profile.perception_duty.effective_ratio +
            control_power * profile.control_duty.effective_ratio +
            movement_power_w * profile.movement_duty.effective_ratio +
            overhead_power
        )
        profile_name = profile.name
    else:
        effective_power = total_power
        profile_name = None

    # Calculate required capacity
    required_capacity = effective_power * mission_hours
    recommended_capacity = required_capacity * safety_margin

    # Estimate weight and volume
    # Use typical energy densities
    energy_density_wh_kg = 150.0  # Conservative Li-ion
    energy_density_wh_l = 300.0   # Wh/L

    estimated_weight = recommended_capacity / energy_density_wh_kg
    estimated_volume = (recommended_capacity / energy_density_wh_l) * 1000  # cm3

    # Find suitable batteries from database
    tier_for_search = tier.name if tier else None
    matching_batteries = find_batteries_for_mission(
        mission_hours=mission_hours,
        average_power_w=effective_power,
        tier=tier_for_search,
        max_weight_kg=max_weight_kg,
        safety_margin=1.0 / safety_margin,  # Adjust for our safety margin
    )

    # If no matches in tier, search all
    if not matching_batteries:
        matching_batteries = find_batteries_for_mission(
            mission_hours=mission_hours,
            average_power_w=effective_power,
            tier=None,
            max_weight_kg=max_weight_kg,
            safety_margin=1.0 / safety_margin,
        )

    # Format recommendations
    recommendations = []
    for battery in matching_batteries[:5]:  # Top 5
        runtime = battery.estimate_runtime_hours(effective_power, safety_margin=1.0/safety_margin)
        recommendations.append({
            "name": battery.name,
            "capacity_wh": battery.capacity_wh,
            "weight_kg": battery.weight_kg,
            "chemistry": battery.chemistry.value,
            "estimated_runtime_hours": runtime,
            "surplus_capacity_pct": (battery.capacity_wh / recommended_capacity - 1) * 100 if recommended_capacity > 0 else 0,
        })

    return BatteryRequirement(
        platform_name=platform_name,
        model_name=model_name,
        mission_hours=mission_hours,
        perception_power_w=perception_power,
        control_power_w=control_power,
        movement_power_w=movement_power_w,
        overhead_power_w=overhead_power,
        total_power_w=total_power,
        effective_power_w=effective_power,
        required_capacity_wh=required_capacity,
        recommended_capacity_wh=recommended_capacity,
        safety_margin=safety_margin,
        estimated_weight_kg=estimated_weight,
        estimated_volume_cm3=estimated_volume,
        recommended_batteries=recommendations,
        profile_name=profile_name,
    )


def format_battery_requirements(req: BatteryRequirement, verbose: bool = False) -> str:
    """Format battery requirements as text output."""
    lines = []

    lines.append("=" * 70)
    lines.append("  BATTERY REQUIREMENTS ESTIMATE")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    lines.append("  Mission Configuration:")
    lines.append(f"    Platform:     {req.platform_name}")
    lines.append(f"    Model:        {req.model_name}")
    lines.append(f"    Duration:     {req.mission_hours:.1f} hours")
    if req.profile_name:
        lines.append(f"    Profile:      {req.profile_name}")
    lines.append("")

    # Power breakdown
    lines.append("  Power Consumption:")
    lines.append("  " + "-" * 40)
    lines.append(f"    Perception:   {req.perception_power_w:6.1f}W")
    lines.append(f"    Control:      {req.control_power_w:6.1f}W")
    lines.append(f"    Movement:     {req.movement_power_w:6.1f}W")
    lines.append(f"    Overhead:     {req.overhead_power_w:6.1f}W")
    lines.append("  " + "-" * 40)
    lines.append(f"    Total Peak:   {req.total_power_w:6.1f}W")
    if req.profile_name:
        lines.append(f"    Effective:    {req.effective_power_w:6.1f}W")
    lines.append("")

    # Battery requirements
    lines.append("  Battery Requirements:")
    lines.append("  " + "-" * 40)
    lines.append(f"    Minimum Capacity:     {req.required_capacity_wh:6.0f}Wh")
    lines.append(f"    Recommended:          {req.recommended_capacity_wh:6.0f}Wh (+{(req.safety_margin-1)*100:.0f}% margin)")
    lines.append("")
    lines.append(f"    Estimated Weight:     {req.estimated_weight_kg:6.2f}kg")
    lines.append(f"    Estimated Volume:     {req.estimated_volume_cm3:6.0f}cm3 ({req.estimated_volume_cm3/1000:.2f}L)")
    lines.append("")

    # Recommendations
    if req.recommended_batteries:
        lines.append("  Recommended Batteries:")
        lines.append("  " + "-" * 60)
        lines.append(f"    {'Name':<22} {'Capacity':>8} {'Weight':>8} {'Runtime':>8} {'Surplus':>8}")
        lines.append("  " + "-" * 60)

        for bat in req.recommended_batteries:
            name = bat["name"][:20]
            cap = f"{bat['capacity_wh']:.0f}Wh"
            weight = f"{bat['weight_kg']:.2f}kg"
            runtime = f"{bat['estimated_runtime_hours']:.1f}h"
            surplus = f"+{bat['surplus_capacity_pct']:.0f}%"
            lines.append(f"    {name:<22} {cap:>8} {weight:>8} {runtime:>8} {surplus:>8}")

        lines.append("")
    else:
        lines.append("  No matching batteries in database.")
        lines.append("  Consider custom battery with specifications above.")
        lines.append("")

    # Energy calculation breakdown if verbose
    if verbose:
        lines.append("  Calculation:")
        lines.append(f"    {req.effective_power_w:.1f}W x {req.mission_hours:.1f}h = {req.required_capacity_wh:.0f}Wh (minimum)")
        lines.append(f"    {req.required_capacity_wh:.0f}Wh x {req.safety_margin:.1f} = {req.recommended_capacity_wh:.0f}Wh (with margin)")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate battery requirements for mission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic battery sizing
  ./cli/estimate_battery_requirements.py --platform Jetson-Orin-Nano-8GB --model yolov8n --mission-hours 4

  # With safety margin
  ./cli/estimate_battery_requirements.py --platform Jetson-Orin-NX-16GB --model resnet18 \\
      --mission-hours 4 --safety-margin 1.3

  # With mission profile
  ./cli/estimate_battery_requirements.py --platform Jetson-Orin-NX-16GB --model yolov8n \\
      --mission-hours 8 --profile warehouse-amr

  # With weight constraint
  ./cli/estimate_battery_requirements.py --platform Jetson-Orin-Nano-8GB --model yolov8n \\
      --mission-hours 2 --max-weight-kg 0.5

  # JSON output
  ./cli/estimate_battery_requirements.py --platform Jetson-Orin-Nano-8GB --model yolov8n \\
      --mission-hours 4 --format json
"""
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        required=True,
        help="Hardware platform name from registry"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name (e.g., resnet18, yolov8n)"
    )
    parser.add_argument(
        "--mission-hours",
        type=float,
        required=True,
        help="Required mission duration in hours"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(MISSION_PROFILES.keys()),
        help="Mission profile for duty cycle adjustment"
    )
    parser.add_argument(
        "--movement-power",
        type=float,
        default=0.0,
        help="External movement power in watts"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.2,
        help="Capacity safety margin multiplier (default: 1.2 = 20%% extra)"
    )
    parser.add_argument(
        "--max-weight-kg",
        type=float,
        help="Maximum battery weight constraint in kg"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed calculations"
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

    # Validate platform
    if get_mapper_info(args.platform) is None:
        print(f"Error: Unknown platform '{args.platform}'", file=sys.stderr)
        print("Use --list-platforms in estimate_power_consumption.py to see options", file=sys.stderr)
        return 1

    # Get mission profile
    profile = None
    if args.profile:
        profile = get_mission_profile(args.profile)
        if profile is None:
            print(f"Error: Unknown mission profile '{args.profile}'", file=sys.stderr)
            return 1

    # Estimate requirements
    requirements = estimate_battery_requirements(
        platform_name=args.platform,
        model_name=args.model,
        mission_hours=args.mission_hours,
        profile=profile,
        movement_power_w=args.movement_power,
        safety_margin=args.safety_margin,
        max_weight_kg=args.max_weight_kg,
    )

    # Output
    if args.format == "json":
        print(json.dumps(requirements.to_dict(), indent=2))
    else:
        print(format_battery_requirements(requirements, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
