#!/usr/bin/env python3
"""
Compare Battery Strategies CLI Tool

Compare different battery selection strategies for missions.
Analyzes tradeoffs between weight, capacity, chemistry, and cost.

Usage:
    # Compare strategies for a mission
    ./cli/compare_battery_strategies.py --tier micro-autonomy --mission-hours 2

    # With weight constraint
    ./cli/compare_battery_strategies.py --tier industrial-edge --mission-hours 4 --max-weight 3

    # Compare specific batteries
    ./cli/compare_battery_strategies.py --batteries small-lipo-2s medium-lipo-4s large-lipo-6s

    # JSON output
    ./cli/compare_battery_strategies.py --tier micro-autonomy --format json
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
    get_battery,
)


# Strategy definitions
STRATEGIES = {
    "minimum-weight": {
        "description": "Minimize weight at cost of runtime",
        "weight_priority": 1.0,
        "capacity_priority": 0.2,
        "density_priority": 0.5,
    },
    "maximum-runtime": {
        "description": "Maximize runtime regardless of weight",
        "weight_priority": 0.2,
        "capacity_priority": 1.0,
        "density_priority": 0.3,
    },
    "balanced": {
        "description": "Balance weight and runtime",
        "weight_priority": 0.5,
        "capacity_priority": 0.5,
        "density_priority": 0.5,
    },
    "high-density": {
        "description": "Prioritize energy density",
        "weight_priority": 0.3,
        "capacity_priority": 0.3,
        "density_priority": 1.0,
    },
    "compact": {
        "description": "Minimize volume",
        "weight_priority": 0.4,
        "capacity_priority": 0.3,
        "density_priority": 0.8,
    },
}


@dataclass
class BatteryScore:
    """Battery scoring for a strategy."""
    battery_name: str
    chemistry: str
    capacity_wh: float
    weight_kg: float
    volume_cm3: float
    energy_density_wh_kg: float
    energy_density_wh_l: float

    # Runtime for mission
    runtime_hours: float
    meets_duration: bool

    # Scores
    weight_score: float
    capacity_score: float
    density_score: float
    total_score: float

    # Ranking
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery": self.battery_name,
            "chemistry": self.chemistry,
            "specs": {
                "capacity_wh": self.capacity_wh,
                "weight_kg": self.weight_kg,
                "volume_cm3": self.volume_cm3,
                "energy_density_wh_kg": self.energy_density_wh_kg,
                "energy_density_wh_l": self.energy_density_wh_l,
            },
            "runtime_hours": self.runtime_hours,
            "meets_duration": self.meets_duration,
            "scores": {
                "weight": self.weight_score,
                "capacity": self.capacity_score,
                "density": self.density_score,
                "total": self.total_score,
            },
            "rank": self.rank,
        }


@dataclass
class StrategyComparison:
    """Comparison of battery strategies."""
    strategy_name: str
    strategy_description: str
    tier_name: Optional[str]
    mission_hours: Optional[float]
    average_power_w: float
    max_weight_kg: Optional[float]

    batteries: List[BatteryScore]
    recommended_battery: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "name": self.strategy_name,
                "description": self.strategy_description,
            },
            "constraints": {
                "tier": self.tier_name,
                "mission_hours": self.mission_hours,
                "average_power_w": self.average_power_w,
                "max_weight_kg": self.max_weight_kg,
            },
            "batteries": [b.to_dict() for b in self.batteries],
            "recommended": self.recommended_battery,
        }


@dataclass
class StrategyComparisonResult:
    """Result comparing multiple strategies."""
    tier_name: Optional[str]
    mission_hours: Optional[float]
    average_power_w: float
    max_weight_kg: Optional[float]

    strategy_results: Dict[str, StrategyComparison]

    # Cross-strategy recommendations
    best_per_strategy: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": {
                "tier": self.tier_name,
                "mission_hours": self.mission_hours,
                "average_power_w": self.average_power_w,
                "max_weight_kg": self.max_weight_kg,
            },
            "strategies": {
                name: comp.to_dict()
                for name, comp in self.strategy_results.items()
            },
            "best_per_strategy": self.best_per_strategy,
        }


def score_battery(
    battery: BatteryConfiguration,
    name: str,
    average_power_w: float,
    mission_hours: Optional[float],
    max_weight_kg: Optional[float],
    strategy: Dict[str, float],
    all_batteries: Dict[str, BatteryConfiguration],
) -> BatteryScore:
    """Score a battery for a strategy."""
    # Calculate runtime
    runtime = battery.estimate_runtime_hours(average_power_w, safety_margin=0.9)

    # Check constraints
    meets_duration = mission_hours is None or runtime >= mission_hours
    fits_weight = max_weight_kg is None or battery.weight_kg <= max_weight_kg

    # Normalize scores (0-1 based on range across all batteries)
    all_weights = [b.weight_kg for b in all_batteries.values()]
    all_capacities = [b.capacity_wh for b in all_batteries.values()]
    all_densities = [b.energy_density_wh_per_kg for b in all_batteries.values()]

    # Weight score (lower is better, so invert)
    weight_range = max(all_weights) - min(all_weights) if len(all_weights) > 1 else 1
    weight_score = 1.0 - (battery.weight_kg - min(all_weights)) / weight_range if weight_range > 0 else 0.5

    # Capacity score (higher is better)
    capacity_range = max(all_capacities) - min(all_capacities) if len(all_capacities) > 1 else 1
    capacity_score = (battery.capacity_wh - min(all_capacities)) / capacity_range if capacity_range > 0 else 0.5

    # Density score (higher is better)
    density_range = max(all_densities) - min(all_densities) if len(all_densities) > 1 else 1
    density_score = (battery.energy_density_wh_per_kg - min(all_densities)) / density_range if density_range > 0 else 0.5

    # Apply penalties for not meeting constraints
    if not meets_duration:
        capacity_score *= 0.5
    if not fits_weight:
        weight_score *= 0.5

    # Calculate total score
    total_score = (
        weight_score * strategy["weight_priority"] +
        capacity_score * strategy["capacity_priority"] +
        density_score * strategy["density_priority"]
    )

    # Normalize total
    total_priority = sum(strategy.values())
    if total_priority > 0:
        total_score /= total_priority

    return BatteryScore(
        battery_name=name,
        chemistry=battery.chemistry.value if battery.chemistry else "unknown",
        capacity_wh=battery.capacity_wh,
        weight_kg=battery.weight_kg,
        volume_cm3=battery.volume_cm3,
        energy_density_wh_kg=battery.energy_density_wh_per_kg,
        energy_density_wh_l=battery.energy_density_wh_per_l,
        runtime_hours=runtime,
        meets_duration=meets_duration,
        weight_score=weight_score,
        capacity_score=capacity_score,
        density_score=density_score,
        total_score=total_score,
    )


def compare_strategies(
    tier_name: Optional[str] = None,
    mission_hours: Optional[float] = None,
    average_power_w: Optional[float] = None,
    max_weight_kg: Optional[float] = None,
    battery_names: Optional[List[str]] = None,
) -> StrategyComparisonResult:
    """Compare battery selection strategies."""
    # Get tier info
    if tier_name:
        tier = get_tier_by_name(tier_name)
        if tier and average_power_w is None:
            average_power_w = tier.typical_power_w

    if average_power_w is None:
        average_power_w = 15.0  # Default

    # Get batteries to compare
    if battery_names:
        batteries = {
            name: BATTERY_CONFIGURATIONS[name]
            for name in battery_names
            if name in BATTERY_CONFIGURATIONS
        }
    else:
        batteries = BATTERY_CONFIGURATIONS

    # Compare each strategy
    strategy_results = {}
    best_per_strategy = {}

    for strategy_name, strategy_params in STRATEGIES.items():
        scores = []
        for name, battery in batteries.items():
            score = score_battery(
                battery, name, average_power_w,
                mission_hours, max_weight_kg,
                strategy_params, batteries
            )
            scores.append(score)

        # Sort by score and assign ranks
        scores.sort(key=lambda x: x.total_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1

        # Get recommended (highest scoring that meets constraints)
        recommended = None
        for score in scores:
            if score.meets_duration and (max_weight_kg is None or score.weight_kg <= max_weight_kg):
                recommended = score.battery_name
                break

        comparison = StrategyComparison(
            strategy_name=strategy_name,
            strategy_description=strategy_params["description"],
            tier_name=tier_name,
            mission_hours=mission_hours,
            average_power_w=average_power_w,
            max_weight_kg=max_weight_kg,
            batteries=scores,
            recommended_battery=recommended,
        )
        strategy_results[strategy_name] = comparison

        if recommended:
            best_per_strategy[strategy_name] = recommended

    return StrategyComparisonResult(
        tier_name=tier_name,
        mission_hours=mission_hours,
        average_power_w=average_power_w,
        max_weight_kg=max_weight_kg,
        strategy_results=strategy_results,
        best_per_strategy=best_per_strategy,
    )


def format_strategy_comparison(result: StrategyComparisonResult) -> str:
    """Format strategy comparison as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  BATTERY STRATEGY COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Constraints
    lines.append("  Constraints:")
    if result.tier_name:
        lines.append(f"    Tier:           {result.tier_name}")
    if result.mission_hours:
        lines.append(f"    Mission:        {result.mission_hours:.1f} hours")
    lines.append(f"    Avg Power:      {result.average_power_w:.1f}W")
    if result.max_weight_kg:
        lines.append(f"    Max Weight:     {result.max_weight_kg:.1f}kg")
    lines.append("")

    # Summary table
    lines.append("  Strategy Recommendations:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Strategy':<20} {'Recommended Battery':<25} {'Runtime':>10}")
    lines.append("  " + "-" * 70)

    for name, comparison in result.strategy_results.items():
        if comparison.recommended_battery:
            rec = comparison.recommended_battery
            battery_scores = {b.battery_name: b for b in comparison.batteries}
            runtime = battery_scores[rec].runtime_hours if rec in battery_scores else 0
            lines.append(f"    {name:<20} {rec:<25} {runtime:>9.1f}h")
        else:
            lines.append(f"    {name:<20} {'(none fit)':^25} {'-':>10}")

    lines.append("  " + "-" * 70)
    lines.append("")

    # Detailed scores for balanced strategy
    balanced = result.strategy_results.get("balanced")
    if balanced:
        lines.append("  Balanced Strategy Details:")
        lines.append("  " + "-" * 76)
        lines.append(f"    {'Battery':<22} {'Wt Score':>9} {'Cap Score':>10} {'Dens Score':>11} {'Total':>8} {'Fit':>5}")
        lines.append("  " + "-" * 76)

        for b in balanced.batteries[:8]:
            fits = "YES" if b.meets_duration else "no"
            lines.append(
                f"    {b.battery_name:<22} {b.weight_score:>8.2f} {b.capacity_score:>10.2f} "
                f"{b.density_score:>11.2f} {b.total_score:>8.2f} {fits:>5}"
            )

        lines.append("  " + "-" * 76)
        lines.append("")

    # Visual comparison
    lines.append("  Visual Score Comparison (top 5 balanced):")
    lines.append("")

    if balanced:
        bar_width = 30
        for b in balanced.batteries[:5]:
            score_len = int(b.total_score * bar_width)
            bar = "#" * score_len + "." * (bar_width - score_len)
            lines.append(f"    {b.battery_name:<22} [{bar}] {b.total_score:.2f}")

    lines.append("")

    # Strategy descriptions
    lines.append("  Strategy Descriptions:")
    for name, params in STRATEGIES.items():
        lines.append(f"    {name:<20}: {params['description']}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare battery selection strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies for a tier
  ./cli/compare_battery_strategies.py --tier micro-autonomy

  # With mission duration
  ./cli/compare_battery_strategies.py --tier micro-autonomy --mission-hours 2

  # With weight constraint
  ./cli/compare_battery_strategies.py --tier industrial-edge --mission-hours 4 --max-weight 3

  # Compare specific batteries
  ./cli/compare_battery_strategies.py --batteries small-lipo-2s medium-lipo-4s large-lipo-6s

  # JSON output
  ./cli/compare_battery_strategies.py --tier micro-autonomy --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Capability tier (sets typical power)"
    )
    parser.add_argument(
        "--mission-hours", "-m",
        type=float,
        help="Required mission duration in hours"
    )
    parser.add_argument(
        "--power", "-p",
        type=float,
        help="Average power consumption in watts"
    )
    parser.add_argument(
        "--max-weight", "-w",
        type=float,
        help="Maximum battery weight in kg"
    )
    parser.add_argument(
        "--batteries", "-b",
        type=str,
        nargs="+",
        help="Specific batteries to compare"
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
        "--list-strategies",
        action="store_true",
        help="List available strategies"
    )

    args = parser.parse_args()

    # Handle list options
    if args.list_batteries:
        print("Available batteries:")
        for name, battery in sorted(BATTERY_CONFIGURATIONS.items()):
            chem = battery.chemistry.value if battery.chemistry else "?"
            print(f"  {name:<25} {battery.capacity_wh:>6.0f}Wh  {battery.weight_kg:.2f}kg  {chem}")
        return 0

    if args.list_strategies:
        print("Available strategies:")
        for name, params in STRATEGIES.items():
            print(f"  {name:<20}: {params['description']}")
        return 0

    # Run comparison
    result = compare_strategies(
        tier_name=args.tier,
        mission_hours=args.mission_hours,
        average_power_w=args.power,
        max_weight_kg=args.max_weight,
        battery_names=args.batteries,
    )

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_strategy_comparison(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
