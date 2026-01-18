#!/usr/bin/env python3
"""
Estimate Operational Range CLI Tool

Estimate operational range (distance, area coverage) for mobile platforms
based on battery capacity, power consumption, and mission profile.

Usage:
    # Estimate range for a configuration
    ./cli/estimate_operational_range.py --battery small-lipo-2s --platform jetson-orin-nano

    # With specific speed
    ./cli/estimate_operational_range.py --battery medium-lipo-4s --platform jetson-orin-nano --speed 1.0

    # For drone (3D coverage)
    ./cli/estimate_operational_range.py --battery drone-6s-10ah --platform jetson-orin-nano --mode aerial

    # JSON output
    ./cli/estimate_operational_range.py --battery small-lipo-2s --platform jetson-orin-nano --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.capability_tiers import (
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.battery import (
    BatteryConfiguration,
    BATTERY_CONFIGURATIONS,
    get_battery,
)
from graphs.mission.hardware_mapper import (
    get_platform,
    list_platforms,
)
from graphs.mission.mission_profiles import (
    MISSION_PROFILES,
    get_mission_profile,
)


class OperatingMode(Enum):
    """Operating mode for range calculation."""
    GROUND = "ground"  # Ground robot, 2D movement
    AERIAL = "aerial"  # Drone, 3D movement
    STATIONARY = "stationary"  # Fixed position, coverage by sensing


# Typical speeds by mode (m/s)
DEFAULT_SPEEDS = {
    OperatingMode.GROUND: 0.5,    # Walking pace
    OperatingMode.AERIAL: 5.0,    # Moderate drone speed
    OperatingMode.STATIONARY: 0.0,
}

# Power overhead factors by mode
POWER_OVERHEAD = {
    OperatingMode.GROUND: 1.5,    # Motors, sensors
    OperatingMode.AERIAL: 3.0,    # High propulsion power
    OperatingMode.STATIONARY: 1.1,
}


@dataclass
class RangeEstimate:
    """Operational range estimate."""
    battery_name: str
    platform_name: str
    mode: str

    # Battery specs
    battery_capacity_wh: float
    usable_capacity_wh: float  # After safety margin

    # Power consumption
    compute_power_w: float
    total_power_w: float  # Including movement/overhead

    # Time estimates
    runtime_hours: float
    effective_runtime_hours: float  # With duty cycle

    # Range estimates
    speed_m_s: float
    max_range_km: float
    round_trip_range_km: float

    # Coverage estimates
    coverage_area_km2: float  # Assuming patrol pattern
    sensing_radius_m: float

    # Mission context
    profile_name: Optional[str] = None
    duty_cycle_avg: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration": {
                "battery": self.battery_name,
                "platform": self.platform_name,
                "mode": self.mode,
                "profile": self.profile_name,
            },
            "battery": {
                "capacity_wh": self.battery_capacity_wh,
                "usable_capacity_wh": self.usable_capacity_wh,
            },
            "power": {
                "compute_w": self.compute_power_w,
                "total_w": self.total_power_w,
            },
            "time": {
                "runtime_hours": self.runtime_hours,
                "effective_runtime_hours": self.effective_runtime_hours,
            },
            "range": {
                "speed_m_s": self.speed_m_s,
                "max_range_km": self.max_range_km,
                "round_trip_range_km": self.round_trip_range_km,
            },
            "coverage": {
                "area_km2": self.coverage_area_km2,
                "sensing_radius_m": self.sensing_radius_m,
            },
        }


@dataclass
class RangeAnalysis:
    """Complete range analysis with scenarios."""
    base_estimate: RangeEstimate

    # Scenario variations
    conservative_estimate: RangeEstimate  # 70% efficiency
    optimistic_estimate: RangeEstimate    # 90% efficiency

    # Speed sensitivity
    speed_vs_range: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base_estimate.to_dict(),
            "conservative": self.conservative_estimate.to_dict(),
            "optimistic": self.optimistic_estimate.to_dict(),
            "speed_sensitivity": self.speed_vs_range,
        }


def calculate_range(
    battery_capacity_wh: float,
    compute_power_w: float,
    mode: OperatingMode,
    speed_m_s: float,
    safety_margin: float = 0.8,
    duty_cycle: float = 1.0,
) -> Dict[str, float]:
    """Calculate operational range."""
    # Usable capacity after safety margin
    usable_capacity = battery_capacity_wh * safety_margin

    # Total power with mode overhead
    total_power = compute_power_w * POWER_OVERHEAD[mode]

    # Runtime
    runtime_hours = usable_capacity / total_power if total_power > 0 else 0
    effective_runtime = runtime_hours * duty_cycle

    # Range (one-way)
    if speed_m_s > 0:
        max_range_m = effective_runtime * 3600 * speed_m_s
        max_range_km = max_range_m / 1000
        round_trip_km = max_range_km / 2
    else:
        max_range_km = 0
        round_trip_km = 0

    # Coverage area (assuming patrol pattern)
    # For ground: rectangular patrol with width = sensing_radius
    # For aerial: similar but can cover more efficiently
    sensing_radius_m = 50 if mode == OperatingMode.AERIAL else 20

    if mode == OperatingMode.STATIONARY:
        # Circular coverage
        coverage_km2 = 3.14159 * (sensing_radius_m / 1000) ** 2
    else:
        # Linear patrol with sensing width
        coverage_km2 = max_range_km * (sensing_radius_m / 1000) * 2

    return {
        "usable_capacity_wh": usable_capacity,
        "total_power_w": total_power,
        "runtime_hours": runtime_hours,
        "effective_runtime_hours": effective_runtime,
        "max_range_km": max_range_km,
        "round_trip_range_km": round_trip_km,
        "coverage_area_km2": coverage_km2,
        "sensing_radius_m": sensing_radius_m,
    }


def estimate_operational_range(
    battery_name: str,
    platform_name: str,
    mode: str = "ground",
    speed_m_s: Optional[float] = None,
    profile_name: Optional[str] = None,
) -> RangeAnalysis:
    """Estimate operational range for a configuration."""
    battery = get_battery(battery_name)
    if battery is None:
        raise ValueError(f"Unknown battery: {battery_name}")

    platform = get_platform(platform_name)
    if platform is None:
        raise ValueError(f"Unknown platform: {platform_name}")

    op_mode = OperatingMode(mode)
    if speed_m_s is None:
        speed_m_s = DEFAULT_SPEEDS[op_mode]

    # Get duty cycle from profile if specified
    duty_cycle = 1.0
    if profile_name:
        profile = get_mission_profile(profile_name)
        if profile:
            duty_cycle = (
                profile.duty_cycle.perception +
                profile.duty_cycle.control +
                profile.duty_cycle.movement
            ) / 3

    # Base estimate (80% efficiency)
    base_result = calculate_range(
        battery.capacity_wh,
        platform.tdp_w,
        op_mode,
        speed_m_s,
        safety_margin=0.8,
        duty_cycle=duty_cycle,
    )

    base_estimate = RangeEstimate(
        battery_name=battery_name,
        platform_name=platform_name,
        mode=mode,
        battery_capacity_wh=battery.capacity_wh,
        usable_capacity_wh=base_result["usable_capacity_wh"],
        compute_power_w=platform.tdp_w,
        total_power_w=base_result["total_power_w"],
        runtime_hours=base_result["runtime_hours"],
        effective_runtime_hours=base_result["effective_runtime_hours"],
        speed_m_s=speed_m_s,
        max_range_km=base_result["max_range_km"],
        round_trip_range_km=base_result["round_trip_range_km"],
        coverage_area_km2=base_result["coverage_area_km2"],
        sensing_radius_m=base_result["sensing_radius_m"],
        profile_name=profile_name,
        duty_cycle_avg=duty_cycle,
    )

    # Conservative estimate (70% efficiency)
    conservative_result = calculate_range(
        battery.capacity_wh,
        platform.tdp_w,
        op_mode,
        speed_m_s,
        safety_margin=0.7,
        duty_cycle=duty_cycle,
    )

    conservative_estimate = RangeEstimate(
        battery_name=battery_name,
        platform_name=platform_name,
        mode=mode,
        battery_capacity_wh=battery.capacity_wh,
        usable_capacity_wh=conservative_result["usable_capacity_wh"],
        compute_power_w=platform.tdp_w,
        total_power_w=conservative_result["total_power_w"],
        runtime_hours=conservative_result["runtime_hours"],
        effective_runtime_hours=conservative_result["effective_runtime_hours"],
        speed_m_s=speed_m_s,
        max_range_km=conservative_result["max_range_km"],
        round_trip_range_km=conservative_result["round_trip_range_km"],
        coverage_area_km2=conservative_result["coverage_area_km2"],
        sensing_radius_m=conservative_result["sensing_radius_m"],
        profile_name=profile_name,
        duty_cycle_avg=duty_cycle,
    )

    # Optimistic estimate (90% efficiency)
    optimistic_result = calculate_range(
        battery.capacity_wh,
        platform.tdp_w,
        op_mode,
        speed_m_s,
        safety_margin=0.9,
        duty_cycle=duty_cycle,
    )

    optimistic_estimate = RangeEstimate(
        battery_name=battery_name,
        platform_name=platform_name,
        mode=mode,
        battery_capacity_wh=battery.capacity_wh,
        usable_capacity_wh=optimistic_result["usable_capacity_wh"],
        compute_power_w=platform.tdp_w,
        total_power_w=optimistic_result["total_power_w"],
        runtime_hours=optimistic_result["runtime_hours"],
        effective_runtime_hours=optimistic_result["effective_runtime_hours"],
        speed_m_s=speed_m_s,
        max_range_km=optimistic_result["max_range_km"],
        round_trip_range_km=optimistic_result["round_trip_range_km"],
        coverage_area_km2=optimistic_result["coverage_area_km2"],
        sensing_radius_m=optimistic_result["sensing_radius_m"],
        profile_name=profile_name,
        duty_cycle_avg=duty_cycle,
    )

    # Speed sensitivity analysis
    speed_vs_range = []
    for speed_factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        test_speed = speed_m_s * speed_factor
        result = calculate_range(
            battery.capacity_wh,
            platform.tdp_w,
            op_mode,
            test_speed,
            safety_margin=0.8,
            duty_cycle=duty_cycle,
        )
        speed_vs_range.append({
            "speed_m_s": test_speed,
            "range_km": result["max_range_km"],
            "runtime_hours": result["runtime_hours"],
        })

    return RangeAnalysis(
        base_estimate=base_estimate,
        conservative_estimate=conservative_estimate,
        optimistic_estimate=optimistic_estimate,
        speed_vs_range=speed_vs_range,
    )


def format_range_analysis(analysis: RangeAnalysis) -> str:
    """Format range analysis as text."""
    lines = []
    e = analysis.base_estimate

    lines.append("=" * 80)
    lines.append("  OPERATIONAL RANGE ESTIMATE")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("  Configuration:")
    lines.append(f"    Battery:    {e.battery_name}")
    lines.append(f"    Platform:   {e.platform_name}")
    lines.append(f"    Mode:       {e.mode}")
    if e.profile_name:
        lines.append(f"    Profile:    {e.profile_name}")
    lines.append(f"    Speed:      {e.speed_m_s:.1f} m/s ({e.speed_m_s * 3.6:.1f} km/h)")
    lines.append("")

    # Battery and power
    lines.append("  Energy & Power:")
    lines.append(f"    Battery Capacity:   {e.battery_capacity_wh:.0f} Wh")
    lines.append(f"    Usable (80%):       {e.usable_capacity_wh:.0f} Wh")
    lines.append(f"    Compute Power:      {e.compute_power_w:.1f} W")
    lines.append(f"    Total Power:        {e.total_power_w:.1f} W (incl. {e.mode} overhead)")
    lines.append("")

    # Runtime
    lines.append("  Runtime:")
    lines.append(f"    Max Runtime:        {e.runtime_hours:.2f} hours ({e.runtime_hours*60:.0f} min)")
    if e.duty_cycle_avg < 1.0:
        lines.append(f"    Duty Cycle:         {e.duty_cycle_avg*100:.0f}%")
        lines.append(f"    Effective Runtime:  {e.effective_runtime_hours:.2f} hours")
    lines.append("")

    # Range estimates
    lines.append("  Range Estimates:")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Scenario':<15} {'Max Range':>12} {'Round Trip':>12} {'Coverage':>12}")
    lines.append("  " + "-" * 60)

    scenarios = [
        ("Conservative", analysis.conservative_estimate),
        ("Base (80%)", analysis.base_estimate),
        ("Optimistic", analysis.optimistic_estimate),
    ]

    for name, est in scenarios:
        lines.append(
            f"    {name:<15} {est.max_range_km:>10.2f} km {est.round_trip_range_km:>10.2f} km "
            f"{est.coverage_area_km2:>10.3f} km2"
        )

    lines.append("  " + "-" * 60)
    lines.append("")

    # Visual range comparison
    lines.append("  Visual Range Comparison:")
    max_range = max(analysis.optimistic_estimate.max_range_km, 0.1)
    bar_width = 40
    for name, est in scenarios:
        bar_len = int((est.max_range_km / max_range) * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {name:<15} [{bar}] {est.max_range_km:.2f}km")
    lines.append("")

    # Speed sensitivity
    lines.append("  Speed vs Range Trade-off:")
    lines.append("  " + "-" * 50)
    lines.append(f"    {'Speed (m/s)':>12} {'Speed (km/h)':>12} {'Range (km)':>12}")
    lines.append("  " + "-" * 50)
    for point in analysis.speed_vs_range:
        lines.append(
            f"    {point['speed_m_s']:>12.1f} {point['speed_m_s']*3.6:>12.1f} {point['range_km']:>12.2f}"
        )
    lines.append("  " + "-" * 50)
    lines.append("")

    # Coverage analysis
    lines.append("  Coverage Analysis:")
    lines.append(f"    Sensing Radius:     {e.sensing_radius_m:.0f} m")
    lines.append(f"    Coverage Area:      {e.coverage_area_km2:.3f} km2 ({e.coverage_area_km2*100:.1f} hectares)")
    if e.mode != "stationary":
        lines.append(f"    Patrol Width:       {e.sensing_radius_m*2:.0f} m")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate operational range for mobile platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ground robot range
  ./cli/estimate_operational_range.py --battery small-lipo-2s --platform jetson-orin-nano

  # Drone range
  ./cli/estimate_operational_range.py --battery drone-6s-10ah --platform jetson-orin-nano --mode aerial

  # With specific speed
  ./cli/estimate_operational_range.py --battery medium-lipo-4s --platform jetson-orin-nano --speed 1.0

  # With mission profile
  ./cli/estimate_operational_range.py --battery medium-lipo-4s --platform jetson-orin-nano --profile drone-inspection

  # JSON output
  ./cli/estimate_operational_range.py --battery small-lipo-2s --platform jetson-orin-nano --format json
"""
    )

    parser.add_argument(
        "--battery", "-b",
        type=str,
        required=True,
        help="Battery configuration name"
    )
    parser.add_argument(
        "--platform", "-p",
        type=str,
        required=True,
        help="Hardware platform name"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["ground", "aerial", "stationary"],
        default="ground",
        help="Operating mode (default: ground)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="Travel speed in m/s (default: mode-specific)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(MISSION_PROFILES.keys()),
        help="Mission profile for duty cycle"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-batteries",
        action="store_true",
        help="List available batteries"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available platforms"
    )

    args = parser.parse_args()

    # Handle list options
    if args.list_batteries:
        print("Available batteries:")
        for name, battery in sorted(BATTERY_CONFIGURATIONS.items()):
            print(f"  {name:<25} {battery.capacity_wh:>6.0f}Wh  {battery.weight_kg:.2f}kg")
        return 0

    if args.list_platforms:
        print("Available platforms:")
        for name in sorted(list_platforms()):
            platform = get_platform(name)
            if platform:
                print(f"  {name:<30} TDP: {platform.tdp_w:>6.1f}W")
        return 0

    # Run analysis
    try:
        analysis = estimate_operational_range(
            battery_name=args.battery,
            platform_name=args.platform,
            mode=args.mode,
            speed_m_s=args.speed,
            profile_name=args.profile,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print(format_range_analysis(analysis))

    return 0


if __name__ == "__main__":
    sys.exit(main())
