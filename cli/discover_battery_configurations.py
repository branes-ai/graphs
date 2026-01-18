#!/usr/bin/env python3
"""
Discover Battery Configurations CLI Tool

List battery options for different mission durations and platforms.
Shows capacity, weight, volume, and estimated runtime.

Usage:
    # Find batteries for 2-hour drone mission
    ./cli/discover_battery_configurations.py --tier micro-autonomy --mission-hours 2

    # Find batteries for 8-hour warehouse shift
    ./cli/discover_battery_configurations.py --tier industrial-edge --mission-hours 8

    # With weight constraint
    ./cli/discover_battery_configurations.py --tier embodied-ai --mission-hours 4 --max-weight-kg 5

    # JSON output
    ./cli/discover_battery_configurations.py --tier micro-autonomy --format json
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
    CAPABILITY_TIERS,
)
from graphs.mission.battery import (
    BatteryConfiguration,
    BatteryChemistry,
    BATTERY_CONFIGURATIONS,
    find_batteries_for_mission,
    find_batteries_for_tier,
    estimate_battery_weight,
)


@dataclass
class BatteryOption:
    """A battery option with fitness analysis."""
    name: str
    chemistry: str
    capacity_wh: float
    weight_kg: float
    volume_cm3: float
    energy_density_wh_kg: float
    energy_density_wh_l: float

    # Runtime estimates
    estimated_runtime_hours: float
    meets_duration: bool
    runtime_margin_pct: float

    # Fitness
    fits_weight: bool
    weight_margin_pct: float

    # Ranking
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "chemistry": self.chemistry,
            "capacity_wh": self.capacity_wh,
            "weight_kg": self.weight_kg,
            "volume_cm3": self.volume_cm3,
            "energy_density": {
                "wh_per_kg": self.energy_density_wh_kg,
                "wh_per_l": self.energy_density_wh_l,
            },
            "runtime": {
                "estimated_hours": self.estimated_runtime_hours,
                "meets_duration": self.meets_duration,
                "margin_pct": self.runtime_margin_pct,
            },
            "weight": {
                "fits_constraint": self.fits_weight,
                "margin_pct": self.weight_margin_pct,
            },
            "rank": self.rank,
        }


@dataclass
class BatteryDiscoveryResult:
    """Result of battery discovery."""
    tier_name: str
    mission_hours: Optional[float]
    average_power_w: float
    max_weight_kg: Optional[float]

    options: List[BatteryOption]
    options_meeting_duration: int
    options_meeting_weight: int

    # Recommendations
    best_overall: Optional[str] = None
    lightest_fitting: Optional[str] = None
    highest_capacity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": {
                "tier": self.tier_name,
                "mission_hours": self.mission_hours,
                "average_power_w": self.average_power_w,
                "max_weight_kg": self.max_weight_kg,
            },
            "summary": {
                "options_found": len(self.options),
                "meeting_duration": self.options_meeting_duration,
                "meeting_weight": self.options_meeting_weight,
            },
            "recommendations": {
                "best_overall": self.best_overall,
                "lightest_fitting": self.lightest_fitting,
                "highest_capacity": self.highest_capacity,
            },
            "options": [o.to_dict() for o in self.options],
        }


def discover_battery_configurations(
    tier_name: str,
    mission_hours: Optional[float] = None,
    average_power_w: Optional[float] = None,
    max_weight_kg: Optional[float] = None,
) -> BatteryDiscoveryResult:
    """
    Discover battery configurations for a tier/mission.

    Args:
        tier_name: Capability tier name
        mission_hours: Required mission duration
        average_power_w: Average power consumption (auto-estimated if not provided)
        max_weight_kg: Maximum weight constraint

    Returns:
        BatteryDiscoveryResult with ranked options
    """
    tier = get_tier_by_name(tier_name)
    if tier is None:
        raise ValueError(f"Unknown tier: {tier_name}")

    # Estimate average power if not provided
    if average_power_w is None:
        average_power_w = tier.typical_power_w

    # Get all batteries
    options = []
    for name, battery in BATTERY_CONFIGURATIONS.items():
        # Calculate runtime
        runtime = battery.estimate_runtime_hours(average_power_w, safety_margin=0.9)

        # Check duration constraint
        meets_duration = True
        runtime_margin = 0.0
        if mission_hours:
            meets_duration = runtime >= mission_hours
            runtime_margin = ((runtime - mission_hours) / mission_hours) * 100 if mission_hours > 0 else 0

        # Check weight constraint
        fits_weight = True
        weight_margin = 0.0
        if max_weight_kg:
            fits_weight = battery.weight_kg <= max_weight_kg
            weight_margin = ((max_weight_kg - battery.weight_kg) / max_weight_kg) * 100 if max_weight_kg > 0 else 0

        option = BatteryOption(
            name=name,
            chemistry=battery.chemistry.value if battery.chemistry else "unknown",
            capacity_wh=battery.capacity_wh,
            weight_kg=battery.weight_kg,
            volume_cm3=battery.volume_cm3,
            energy_density_wh_kg=battery.energy_density_wh_per_kg,
            energy_density_wh_l=battery.energy_density_wh_per_l,
            estimated_runtime_hours=runtime,
            meets_duration=meets_duration,
            runtime_margin_pct=runtime_margin,
            fits_weight=fits_weight,
            weight_margin_pct=weight_margin,
        )
        options.append(option)

    # Sort by capacity (ascending for lighter options first among those that fit)
    options.sort(key=lambda x: x.capacity_wh)

    # Assign ranks (among those that fit all constraints)
    fitting = [o for o in options if o.meets_duration and o.fits_weight]
    for i, opt in enumerate(fitting):
        opt.rank = i + 1

    # Count statistics
    meeting_duration = sum(1 for o in options if o.meets_duration)
    meeting_weight = sum(1 for o in options if o.fits_weight)

    # Recommendations
    if fitting:
        best_overall = fitting[0].name  # Lightest that fits
        lightest_fitting = min(fitting, key=lambda x: x.weight_kg).name
        highest_capacity = max(fitting, key=lambda x: x.capacity_wh).name
    else:
        best_overall = None
        lightest_fitting = None
        highest_capacity = None

    return BatteryDiscoveryResult(
        tier_name=tier_name,
        mission_hours=mission_hours,
        average_power_w=average_power_w,
        max_weight_kg=max_weight_kg,
        options=options,
        options_meeting_duration=meeting_duration,
        options_meeting_weight=meeting_weight,
        best_overall=best_overall,
        lightest_fitting=lightest_fitting,
        highest_capacity=highest_capacity,
    )


def format_discovery(result: BatteryDiscoveryResult, verbose: bool = False) -> str:
    """Format discovery result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  BATTERY CONFIGURATION DISCOVERY")
    lines.append("=" * 80)
    lines.append("")

    # Constraints
    lines.append("  Constraints:")
    lines.append(f"    Tier:           {result.tier_name}")
    if result.mission_hours:
        lines.append(f"    Mission:        {result.mission_hours:.1f} hours")
    lines.append(f"    Avg Power:      {result.average_power_w:.1f}W")
    if result.max_weight_kg:
        lines.append(f"    Max Weight:     {result.max_weight_kg:.1f}kg")
    lines.append("")

    # Summary
    lines.append("  Summary:")
    lines.append(f"    Total Options:      {len(result.options)}")
    lines.append(f"    Meet Duration:      {result.options_meeting_duration}")
    if result.max_weight_kg:
        lines.append(f"    Meet Weight:        {result.options_meeting_weight}")
    lines.append("")

    # Recommendations
    if result.best_overall:
        lines.append("  Recommendations:")
        lines.append(f"    Best Overall:       {result.best_overall}")
        lines.append(f"    Lightest Fitting:   {result.lightest_fitting}")
        lines.append(f"    Highest Capacity:   {result.highest_capacity}")
        lines.append("")

    # Options table
    lines.append("  Battery Options:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Name':<22} {'Capacity':>10} {'Weight':>8} {'Runtime':>10} {'Fit':>6}")
    lines.append("  " + "-" * 76)

    for o in result.options:
        cap_str = f"{o.capacity_wh:.0f}Wh"
        weight_str = f"{o.weight_kg:.2f}kg"
        runtime_str = f"{o.estimated_runtime_hours:.1f}h"
        fit_str = "YES" if (o.meets_duration and o.fits_weight) else "no"

        lines.append(
            f"    {o.name:<22} {cap_str:>10} {weight_str:>8} {runtime_str:>10} {fit_str:>6}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Detailed info (if verbose)
    if verbose:
        fitting = [o for o in result.options if o.meets_duration and o.fits_weight]
        if fitting:
            lines.append("  Fitting Options Detail:")
            for o in fitting[:5]:
                lines.append(f"    {o.name}:")
                lines.append(f"      Chemistry: {o.chemistry}")
                lines.append(f"      Density: {o.energy_density_wh_kg:.0f} Wh/kg, {o.energy_density_wh_l:.0f} Wh/L")
                lines.append(f"      Volume: {o.volume_cm3:.0f} cm3 ({o.volume_cm3/1000:.2f}L)")
                if result.mission_hours:
                    lines.append(f"      Runtime Margin: +{o.runtime_margin_pct:.0f}%")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Discover battery configurations for missions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find batteries for micro-autonomy tier
  ./cli/discover_battery_configurations.py --tier micro-autonomy

  # For 2-hour mission
  ./cli/discover_battery_configurations.py --tier micro-autonomy --mission-hours 2

  # With weight constraint
  ./cli/discover_battery_configurations.py --tier embodied-ai --mission-hours 4 --max-weight-kg 5

  # JSON output
  ./cli/discover_battery_configurations.py --tier industrial-edge --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        required=True,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Capability tier"
    )
    parser.add_argument(
        "--mission-hours", "-m",
        type=float,
        help="Required mission duration in hours"
    )
    parser.add_argument(
        "--power", "-p",
        type=float,
        help="Average power consumption in watts (auto-estimated if not provided)"
    )
    parser.add_argument(
        "--max-weight-kg", "-w",
        type=float,
        help="Maximum battery weight constraint"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all battery configurations"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list_all:
        print("Available battery configurations:")
        for name, battery in sorted(BATTERY_CONFIGURATIONS.items()):
            chem = battery.chemistry.value if battery.chemistry else "?"
            print(f"  {name:<25} {battery.capacity_wh:>6.0f}Wh  {battery.weight_kg:.2f}kg  {chem}")
        return 0

    # Run discovery
    try:
        result = discover_battery_configurations(
            tier_name=args.tier,
            mission_hours=args.mission_hours,
            average_power_w=args.power,
            max_weight_kg=args.max_weight_kg,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_discovery(result, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
