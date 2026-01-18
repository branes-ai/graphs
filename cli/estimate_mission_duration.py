#!/usr/bin/env python3
"""
Estimate Mission Duration CLI Tool

Estimate how long a mission can run given hardware, model, and battery configuration.
Uses power consumption estimates and mission profiles to calculate runtime.

Usage:
    # Basic estimation
    ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n --battery-wh 100

    # With mission profile
    ./cli/estimate_mission_duration.py --platform Jetson-Orin-NX-16GB --model resnet18 \\
        --battery-wh 500 --profile warehouse-amr

    # With movement power
    ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n \\
        --battery-wh 100 --movement-power 15 --movement-duty-cycle 0.7

    # Using predefined battery
    ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n --battery drone-medium
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.hardware.mappers import (
    list_all_mappers,
    get_mapper_info,
)
from graphs.mission.capability_tiers import (
    TierName,
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
    get_battery_by_name,
)
from graphs.mission.mission_profiles import (
    MissionProfile,
    MISSION_PROFILES,
    get_mission_profile,
    list_mission_profiles,
)


@dataclass
class MissionDurationEstimate:
    """Mission duration estimate result."""
    platform_name: str
    model_name: str
    battery_name: str
    battery_capacity_wh: float

    # Power breakdown
    perception_power_w: float
    control_power_w: float
    movement_power_w: float
    overhead_power_w: float
    total_power_w: float

    # Duration estimates
    mission_duration_hours: float
    mission_duration_minutes: float

    # Duty cycle adjusted
    effective_power_w: float  # After duty cycle adjustment
    adjusted_duration_hours: float

    # Safety margins
    safety_margin: float
    usable_capacity_wh: float

    # Mission profile info
    profile_name: Optional[str]
    profile_description: Optional[str]

    # Confidence
    confidence: str  # "high", "medium", "low"
    limiting_factor: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "model": self.model_name,
            "battery": {
                "name": self.battery_name,
                "capacity_wh": self.battery_capacity_wh,
                "usable_capacity_wh": self.usable_capacity_wh,
                "safety_margin": self.safety_margin,
            },
            "power": {
                "perception_w": self.perception_power_w,
                "control_w": self.control_power_w,
                "movement_w": self.movement_power_w,
                "overhead_w": self.overhead_power_w,
                "total_w": self.total_power_w,
                "effective_w": self.effective_power_w,
            },
            "duration": {
                "hours": self.mission_duration_hours,
                "minutes": self.mission_duration_minutes,
                "adjusted_hours": self.adjusted_duration_hours,
            },
            "mission_profile": self.profile_name,
            "confidence": self.confidence,
            "limiting_factor": self.limiting_factor,
        }


def estimate_model_power(platform_name: str, model_name: str) -> Dict[str, float]:
    """Estimate compute power for a model on a platform."""
    info = get_mapper_info(platform_name)
    if info is None:
        return {"compute_power_w": 0.0}

    tdp = info["default_tdp_w"]

    # Typical ratio based on platform size
    if tdp <= 30:
        typical_ratio = 0.75
    elif tdp <= 100:
        typical_ratio = 0.60
    else:
        typical_ratio = 0.50

    # Model complexity adjustment
    model_lower = model_name.lower()
    if "resnet50" in model_lower or "resnet101" in model_lower:
        complexity = 0.70
    elif "resnet18" in model_lower or "resnet34" in model_lower:
        complexity = 0.45
    elif "yolo" in model_lower:
        if "yolov8x" in model_lower or "yolov8l" in model_lower:
            complexity = 0.60
        elif "yolov8m" in model_lower:
            complexity = 0.45
        else:
            complexity = 0.35
    elif "mobilenet" in model_lower:
        complexity = 0.25
    elif "efficientnet" in model_lower:
        complexity = 0.35
    else:
        complexity = 0.50

    compute_power = tdp * typical_ratio * complexity

    return {
        "compute_power_w": compute_power,
        "tdp_w": tdp,
    }


def estimate_mission_duration(
    platform_name: str,
    model_name: str,
    battery_wh: float,
    battery_name: str = "custom",
    profile: Optional[MissionProfile] = None,
    movement_power_w: float = 0.0,
    movement_duty_cycle: float = 1.0,
    safety_margin: float = 0.9,
) -> Optional[MissionDurationEstimate]:
    """
    Estimate mission duration for a configuration.

    Args:
        platform_name: Hardware platform name
        model_name: Model name
        battery_wh: Battery capacity in watt-hours
        battery_name: Name of battery configuration
        profile: Optional mission profile for duty cycle adjustments
        movement_power_w: External movement power
        movement_duty_cycle: Fraction of time movement is active
        safety_margin: Usable battery fraction (default 0.9 = 90%)

    Returns:
        MissionDurationEstimate with detailed breakdown
    """
    # Get platform info
    info = get_mapper_info(platform_name)
    if info is None:
        return None

    # Estimate compute power
    power_info = estimate_model_power(platform_name, model_name)
    compute_power = power_info["compute_power_w"]
    tdp = power_info["tdp_w"]

    # Get allocation for this tier
    tier = get_tier_for_power(tdp)
    allocation = get_typical_allocation(tier.name) if tier else ALLOCATION_BALANCED

    # Split compute power between perception and control
    perception_ratio = allocation.perception_ratio / (allocation.perception_ratio + allocation.control_ratio)
    control_ratio = allocation.control_ratio / (allocation.perception_ratio + allocation.control_ratio)

    perception_power = compute_power * perception_ratio
    control_power = compute_power * control_ratio

    # Overhead
    overhead_power = compute_power * 0.12  # ~12% overhead

    # Movement power (if not specified, estimate from tier)
    if movement_power_w <= 0 and tier:
        movement_power_w = tier.typical_power_w * allocation.movement_ratio

    # Apply duty cycles from mission profile if provided
    if profile:
        perception_effective = perception_power * profile.perception_duty.effective_ratio
        control_effective = control_power * profile.control_duty.effective_ratio
        movement_effective = movement_power_w * profile.movement_duty.effective_ratio
        overhead_effective = overhead_power * 1.0  # Always on

        effective_power = perception_effective + control_effective + movement_effective + overhead_effective
        profile_name = profile.name
        profile_desc = profile.description
    else:
        movement_effective = movement_power_w * movement_duty_cycle
        effective_power = perception_power + control_power + movement_effective + overhead_power
        profile_name = None
        profile_desc = None

    # Total peak power (without duty cycle)
    total_power = perception_power + control_power + movement_power_w + overhead_power

    # Calculate durations
    usable_capacity = battery_wh * safety_margin

    # Peak power duration
    if total_power > 0:
        peak_duration_hours = usable_capacity / total_power
    else:
        peak_duration_hours = float('inf')

    # Effective power duration (with duty cycles)
    if effective_power > 0:
        effective_duration_hours = usable_capacity / effective_power
    else:
        effective_duration_hours = float('inf')

    # Determine confidence and limiting factor
    if effective_power > battery_wh * 0.5:  # Power > 0.5C discharge
        confidence = "medium"
        limiting_factor = "high discharge rate"
    elif movement_power_w > compute_power * 2:
        confidence = "medium"
        limiting_factor = "movement power dominates"
    elif profile is None:
        confidence = "medium"
        limiting_factor = "no mission profile (using peak power)"
    else:
        confidence = "high"
        limiting_factor = "battery capacity"

    return MissionDurationEstimate(
        platform_name=platform_name,
        model_name=model_name,
        battery_name=battery_name,
        battery_capacity_wh=battery_wh,
        perception_power_w=perception_power,
        control_power_w=control_power,
        movement_power_w=movement_power_w,
        overhead_power_w=overhead_power,
        total_power_w=total_power,
        mission_duration_hours=peak_duration_hours,
        mission_duration_minutes=peak_duration_hours * 60,
        effective_power_w=effective_power,
        adjusted_duration_hours=effective_duration_hours,
        safety_margin=safety_margin,
        usable_capacity_wh=usable_capacity,
        profile_name=profile_name,
        profile_description=profile_desc,
        confidence=confidence,
        limiting_factor=limiting_factor,
    )


def format_duration(hours: float) -> str:
    """Format duration as human-readable string."""
    if hours >= 24:
        days = int(hours / 24)
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours:.1f}h"
    elif hours >= 1:
        return f"{hours:.1f}h"
    else:
        return f"{hours * 60:.0f}min"


def format_mission_estimate(estimate: MissionDurationEstimate, verbose: bool = False) -> str:
    """Format mission estimate as text output."""
    lines = []

    lines.append("=" * 70)
    lines.append("  MISSION DURATION ESTIMATE")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    lines.append("  Configuration:")
    lines.append(f"    Platform:     {estimate.platform_name}")
    lines.append(f"    Model:        {estimate.model_name}")
    lines.append(f"    Battery:      {estimate.battery_name} ({estimate.battery_capacity_wh:.0f}Wh)")
    if estimate.profile_name:
        lines.append(f"    Profile:      {estimate.profile_name}")
    lines.append("")

    # Power breakdown
    lines.append("  Power Consumption:")
    lines.append("  " + "-" * 40)
    lines.append(f"    Perception:   {estimate.perception_power_w:6.1f}W")
    lines.append(f"    Control:      {estimate.control_power_w:6.1f}W")
    lines.append(f"    Movement:     {estimate.movement_power_w:6.1f}W")
    lines.append(f"    Overhead:     {estimate.overhead_power_w:6.1f}W")
    lines.append("  " + "-" * 40)
    lines.append(f"    Peak Total:   {estimate.total_power_w:6.1f}W")
    if estimate.profile_name:
        lines.append(f"    Effective:    {estimate.effective_power_w:6.1f}W (with duty cycles)")
    lines.append("")

    # Duration estimates
    lines.append("  Estimated Duration:")
    lines.append("  " + "-" * 40)

    # Visual bar for duration
    max_hours = 12.0  # Reference for bar
    bar_width = 30

    # Peak power duration
    peak_bar_len = min(int((estimate.mission_duration_hours / max_hours) * bar_width), bar_width)
    peak_bar = "#" * peak_bar_len
    lines.append(f"    Peak Power:   [{peak_bar:<{bar_width}}]")
    lines.append(f"                  {format_duration(estimate.mission_duration_hours)}")

    # Adjusted duration (if profile)
    if estimate.profile_name and estimate.adjusted_duration_hours != estimate.mission_duration_hours:
        adj_bar_len = min(int((estimate.adjusted_duration_hours / max_hours) * bar_width), bar_width)
        adj_bar = "#" * adj_bar_len
        lines.append(f"    With Profile: [{adj_bar:<{bar_width}}]")
        lines.append(f"                  {format_duration(estimate.adjusted_duration_hours)}")
    lines.append("")

    # Battery info
    lines.append("  Battery Details:")
    lines.append(f"    Capacity:     {estimate.battery_capacity_wh:.0f}Wh")
    lines.append(f"    Safety Margin: {estimate.safety_margin * 100:.0f}%")
    lines.append(f"    Usable:       {estimate.usable_capacity_wh:.0f}Wh")
    lines.append("")

    # Confidence
    confidence_indicators = {"high": "***", "medium": "**", "low": "*"}
    lines.append(f"  Confidence: {confidence_indicators.get(estimate.confidence, '?')} {estimate.confidence.upper()}")
    lines.append(f"  Limiting Factor: {estimate.limiting_factor}")
    lines.append("")

    if verbose and estimate.profile_description:
        lines.append(f"  Profile: {estimate.profile_description}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate mission duration for hardware/model/battery configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic estimation with battery capacity
  ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n --battery-wh 100

  # Using predefined battery
  ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n --battery drone-medium

  # With mission profile
  ./cli/estimate_mission_duration.py --platform Jetson-Orin-NX-16GB --model resnet18 \\
      --battery-wh 500 --profile warehouse-amr

  # With movement power
  ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n \\
      --battery-wh 100 --movement-power 15 --movement-duty-cycle 0.7

  # JSON output
  ./cli/estimate_mission_duration.py --platform Jetson-Orin-Nano-8GB --model yolov8n \\
      --battery-wh 100 --format json
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

    # Battery options (one required)
    battery_group = parser.add_mutually_exclusive_group(required=True)
    battery_group.add_argument(
        "--battery-wh",
        type=float,
        help="Battery capacity in watt-hours"
    )
    battery_group.add_argument(
        "--battery",
        type=str,
        choices=list(BATTERY_CONFIGURATIONS.keys()),
        help="Predefined battery configuration"
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
        "--movement-duty-cycle",
        type=float,
        default=1.0,
        help="Movement duty cycle (0.0-1.0, default: 1.0)"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.9,
        help="Usable battery fraction (default: 0.9 = 90%%)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
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
        help="List available mission profiles and exit"
    )
    parser.add_argument(
        "--list-batteries",
        action="store_true",
        help="List available battery configurations and exit"
    )

    args = parser.parse_args()

    # Handle list options
    if args.list_profiles:
        print("Available mission profiles:")
        for name, profile in MISSION_PROFILES.items():
            print(f"  {name:<25} {profile.tier.value:<18} {profile.typical_duration_hours:.1f}h")
        return 0

    if args.list_batteries:
        print("Available battery configurations:")
        for name, battery in BATTERY_CONFIGURATIONS.items():
            print(f"  {name:<25} {battery.capacity_wh:>6.0f}Wh  {battery.weight_kg:.2f}kg")
        return 0

    # Validate platform
    if get_mapper_info(args.platform) is None:
        print(f"Error: Unknown platform '{args.platform}'", file=sys.stderr)
        return 1

    # Get battery configuration
    if args.battery:
        battery_config = get_battery_by_name(args.battery)
        if battery_config is None:
            print(f"Error: Unknown battery '{args.battery}'", file=sys.stderr)
            return 1
        battery_wh = battery_config.capacity_wh
        battery_name = battery_config.name
    else:
        battery_wh = args.battery_wh
        battery_name = "custom"

    # Get mission profile
    profile = None
    if args.profile:
        profile = get_mission_profile(args.profile)
        if profile is None:
            print(f"Error: Unknown mission profile '{args.profile}'", file=sys.stderr)
            return 1

    # Estimate mission duration
    estimate = estimate_mission_duration(
        platform_name=args.platform,
        model_name=args.model,
        battery_wh=battery_wh,
        battery_name=battery_name,
        profile=profile,
        movement_power_w=args.movement_power,
        movement_duty_cycle=args.movement_duty_cycle,
        safety_margin=args.safety_margin,
    )

    if estimate is None:
        print("Error: Could not estimate mission duration", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(estimate.to_dict(), indent=2))
    else:
        print(format_mission_estimate(estimate, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
