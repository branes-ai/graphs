#!/usr/bin/env python3
"""
Explore Battery Life CLI Tool

Understand factors affecting battery life and explore tradeoffs.
Provides sensitivity analysis and optimization suggestions.

Usage:
    # Explore battery life for configuration
    ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 --model resnet18

    # Show sensitivity analysis
    ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 --sensitivity

    # Compare duty cycle impacts
    ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 \\
        --duty-cycles 0.5 0.75 1.0

    # Compare models
    ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 \\
        --models yolov8n yolov8s yolov8m
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
    get_tier_for_power,
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
)


@dataclass
class BatteryLifeExploration:
    """Battery life exploration results."""
    platform_name: str
    battery_wh: float
    base_model: str

    # Base case
    base_runtime_hours: float
    base_power_w: float

    # Sensitivity results
    duty_cycle_analysis: List[Dict[str, Any]] = field(default_factory=list)
    model_comparison: List[Dict[str, Any]] = field(default_factory=list)
    power_sensitivity: List[Dict[str, Any]] = field(default_factory=list)

    # Optimization suggestions
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "battery_wh": self.battery_wh,
            "base_model": self.base_model,
            "base_case": {
                "runtime_hours": self.base_runtime_hours,
                "power_w": self.base_power_w,
            },
            "duty_cycle_analysis": self.duty_cycle_analysis,
            "model_comparison": self.model_comparison,
            "power_sensitivity": self.power_sensitivity,
            "suggestions": self.suggestions,
        }


def estimate_power_for_model(platform_name: str, model_name: str) -> float:
    """Estimate power consumption for a model on a platform."""
    info = get_mapper_info(platform_name)
    if info is None:
        return 5.0

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
    elif "yolov8x" in model_lower or "yolov8l" in model_lower:
        complexity = 0.60
    elif "yolov8m" in model_lower:
        complexity = 0.45
    elif "yolov8s" in model_lower:
        complexity = 0.35
    elif "yolov8n" in model_lower:
        complexity = 0.28
    elif "yolo" in model_lower:
        complexity = 0.40
    elif "mobilenet" in model_lower:
        complexity = 0.25
    elif "efficientnet_b0" in model_lower:
        complexity = 0.28
    elif "efficientnet" in model_lower:
        complexity = 0.38
    else:
        complexity = 0.50

    compute_power = tdp * typical_ratio * complexity

    # Add overhead
    tier = get_tier_for_power(tdp)
    if tier:
        allocation = get_typical_allocation(tier.name)
        # Add movement (typical) and overhead
        movement_power = tier.typical_power_w * allocation.movement_ratio
        overhead_power = compute_power * 0.12
        total_power = compute_power + movement_power + overhead_power
    else:
        total_power = compute_power * 1.15  # 15% overhead

    return total_power


def calculate_runtime(battery_wh: float, power_w: float, safety_margin: float = 0.9) -> float:
    """Calculate runtime in hours."""
    if power_w <= 0:
        return float('inf')
    usable = battery_wh * safety_margin
    return usable / power_w


def explore_battery_life(
    platform_name: str,
    battery_wh: float,
    base_model: str = "resnet18",
    models: Optional[List[str]] = None,
    duty_cycles: Optional[List[float]] = None,
    sensitivity: bool = False,
) -> BatteryLifeExploration:
    """
    Explore battery life factors and tradeoffs.

    Args:
        platform_name: Hardware platform name
        battery_wh: Battery capacity
        base_model: Base model for comparison
        models: List of models to compare
        duty_cycles: List of duty cycles to analyze
        sensitivity: Whether to run sensitivity analysis

    Returns:
        BatteryLifeExploration with detailed analysis
    """
    # Base case
    base_power = estimate_power_for_model(platform_name, base_model)
    base_runtime = calculate_runtime(battery_wh, base_power)

    # Initialize results
    exploration = BatteryLifeExploration(
        platform_name=platform_name,
        battery_wh=battery_wh,
        base_model=base_model,
        base_runtime_hours=base_runtime,
        base_power_w=base_power,
    )

    # Duty cycle analysis
    if duty_cycles:
        for dc in duty_cycles:
            effective_power = base_power * dc
            runtime = calculate_runtime(battery_wh, effective_power)
            improvement = (runtime / base_runtime - 1) * 100 if base_runtime > 0 else 0

            exploration.duty_cycle_analysis.append({
                "duty_cycle": dc,
                "effective_power_w": effective_power,
                "runtime_hours": runtime,
                "improvement_pct": improvement,
            })

    # Model comparison
    if models:
        for model in models:
            power = estimate_power_for_model(platform_name, model)
            runtime = calculate_runtime(battery_wh, power)
            diff = (runtime / base_runtime - 1) * 100 if base_runtime > 0 else 0

            exploration.model_comparison.append({
                "model": model,
                "power_w": power,
                "runtime_hours": runtime,
                "difference_pct": diff,
            })

    # Sensitivity analysis
    if sensitivity:
        # Power sensitivity: what if power changes by 10%, 20%, 30%?
        for pct_change in [-30, -20, -10, 0, 10, 20, 30]:
            adjusted_power = base_power * (1 + pct_change / 100)
            runtime = calculate_runtime(battery_wh, adjusted_power)

            exploration.power_sensitivity.append({
                "power_change_pct": pct_change,
                "power_w": adjusted_power,
                "runtime_hours": runtime,
                "runtime_change_pct": (runtime / base_runtime - 1) * 100 if base_runtime > 0 else 0,
            })

    # Generate suggestions
    suggestions = []

    # Check if duty cycle optimization would help
    if not duty_cycles:
        # Calculate potential with 75% duty cycle
        dc75_runtime = calculate_runtime(battery_wh, base_power * 0.75)
        improvement = (dc75_runtime / base_runtime - 1) * 100
        if improvement > 20:
            suggestions.append(
                f"Reducing duty cycle to 75% could extend runtime by {improvement:.0f}% "
                f"(+{dc75_runtime - base_runtime:.1f}h)"
            )

    # Check if lighter model would help
    light_models = ["yolov8n", "mobilenet_v2", "efficientnet_b0"]
    best_light = None
    best_improvement = 0
    for model in light_models:
        if model.lower() != base_model.lower():
            power = estimate_power_for_model(platform_name, model)
            runtime = calculate_runtime(battery_wh, power)
            improvement = (runtime / base_runtime - 1) * 100
            if improvement > best_improvement:
                best_improvement = improvement
                best_light = (model, runtime)

    if best_light and best_improvement > 15:
        suggestions.append(
            f"Switching to {best_light[0]} could extend runtime by {best_improvement:.0f}% "
            f"(+{best_light[1] - base_runtime:.1f}h)"
        )

    # Battery capacity suggestions
    if base_runtime < 2.0:
        double_capacity = calculate_runtime(battery_wh * 2, base_power)
        suggestions.append(
            f"Doubling battery to {battery_wh*2:.0f}Wh would give {double_capacity:.1f}h runtime"
        )

    # Power gating suggestion
    if base_power > 20:
        suggestions.append(
            "Consider power gating inactive subsystems between inference cycles"
        )

    exploration.suggestions = suggestions

    return exploration


def format_exploration(exploration: BatteryLifeExploration, verbose: bool = False) -> str:
    """Format exploration results as text."""
    lines = []

    lines.append("=" * 70)
    lines.append("  BATTERY LIFE EXPLORATION")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    lines.append("  Configuration:")
    lines.append(f"    Platform:     {exploration.platform_name}")
    lines.append(f"    Battery:      {exploration.battery_wh:.0f}Wh")
    lines.append(f"    Base Model:   {exploration.base_model}")
    lines.append("")

    # Base case
    lines.append("  Base Case:")
    lines.append(f"    Power:        {exploration.base_power_w:.1f}W")
    lines.append(f"    Runtime:      {exploration.base_runtime_hours:.1f}h ({exploration.base_runtime_hours*60:.0f}min)")
    lines.append("")

    # Duty cycle analysis
    if exploration.duty_cycle_analysis:
        lines.append("  Duty Cycle Analysis:")
        lines.append("  " + "-" * 55)
        lines.append(f"    {'Duty Cycle':>12} {'Power (W)':>12} {'Runtime':>12} {'Change':>12}")
        lines.append("  " + "-" * 55)

        for dc in exploration.duty_cycle_analysis:
            dc_str = f"{dc['duty_cycle']*100:.0f}%"
            power_str = f"{dc['effective_power_w']:.1f}W"
            runtime_str = f"{dc['runtime_hours']:.1f}h"
            change_str = f"{dc['improvement_pct']:+.0f}%"
            lines.append(f"    {dc_str:>12} {power_str:>12} {runtime_str:>12} {change_str:>12}")
        lines.append("")

    # Model comparison
    if exploration.model_comparison:
        lines.append("  Model Comparison:")
        lines.append("  " + "-" * 60)
        lines.append(f"    {'Model':<20} {'Power (W)':>12} {'Runtime':>12} {'vs Base':>12}")
        lines.append("  " + "-" * 60)

        for mc in exploration.model_comparison:
            model_str = mc['model'][:18]
            power_str = f"{mc['power_w']:.1f}W"
            runtime_str = f"{mc['runtime_hours']:.1f}h"
            diff_str = f"{mc['difference_pct']:+.0f}%"
            lines.append(f"    {model_str:<20} {power_str:>12} {runtime_str:>12} {diff_str:>12}")
        lines.append("")

    # Power sensitivity
    if exploration.power_sensitivity:
        lines.append("  Power Sensitivity:")
        lines.append("  " + "-" * 55)

        # Find the baseline (0% change)
        baseline_idx = None
        for i, ps in enumerate(exploration.power_sensitivity):
            if ps['power_change_pct'] == 0:
                baseline_idx = i
                break

        max_runtime = max(ps['runtime_hours'] for ps in exploration.power_sensitivity)

        for ps in exploration.power_sensitivity:
            change_pct = ps['power_change_pct']
            runtime = ps['runtime_hours']

            # Visual bar
            bar_len = int((runtime / max_runtime) * 25) if max_runtime > 0 else 0
            bar = "#" * bar_len

            sign = "+" if change_pct > 0 else ""
            marker = " <-- base" if change_pct == 0 else ""

            lines.append(
                f"    {sign}{change_pct:3.0f}% power: [{bar:<25}] {runtime:.1f}h{marker}"
            )
        lines.append("")

    # Suggestions
    if exploration.suggestions:
        lines.append("  Optimization Suggestions:")
        lines.append("  " + "-" * 55)
        for i, suggestion in enumerate(exploration.suggestions, 1):
            # Wrap long suggestions
            wrapped = suggestion
            if len(suggestion) > 60:
                words = suggestion.split()
                wrapped = ""
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > 60:
                        wrapped += line + "\n      "
                        line = word
                    else:
                        line = line + " " + word if line else word
                wrapped += line
            lines.append(f"    {i}. {wrapped}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Explore battery life factors and tradeoffs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic exploration
  ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 --model resnet18

  # With sensitivity analysis
  ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 --sensitivity

  # Compare duty cycles
  ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 \\
      --duty-cycles 0.5 0.75 1.0

  # Compare models
  ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 \\
      --models yolov8n yolov8s yolov8m resnet18

  # Using predefined battery
  ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery drone-medium

  # JSON output
  ./cli/explore_battery_life.py --platform Jetson-Orin-Nano-8GB --battery-wh 100 --format json
"""
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        required=True,
        help="Hardware platform name"
    )

    # Battery options
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
        "--model", "-m",
        type=str,
        default="resnet18",
        help="Base model for analysis (default: resnet18)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of models to compare"
    )
    parser.add_argument(
        "--duty-cycles",
        type=float,
        nargs="+",
        help="List of duty cycles to analyze (0.0-1.0)"
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run power sensitivity analysis"
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

    args = parser.parse_args()

    # Validate platform
    if get_mapper_info(args.platform) is None:
        print(f"Error: Unknown platform '{args.platform}'", file=sys.stderr)
        return 1

    # Get battery capacity
    if args.battery:
        battery_config = get_battery_by_name(args.battery)
        if battery_config is None:
            print(f"Error: Unknown battery '{args.battery}'", file=sys.stderr)
            return 1
        battery_wh = battery_config.capacity_wh
    else:
        battery_wh = args.battery_wh

    # Validate duty cycles
    if args.duty_cycles:
        for dc in args.duty_cycles:
            if not 0.0 <= dc <= 1.0:
                print(f"Error: Duty cycle must be 0.0-1.0, got {dc}", file=sys.stderr)
                return 1

    # Run exploration
    exploration = explore_battery_life(
        platform_name=args.platform,
        battery_wh=battery_wh,
        base_model=args.model,
        models=args.models,
        duty_cycles=args.duty_cycles,
        sensitivity=args.sensitivity,
    )

    # Output
    if args.format == "json":
        print(json.dumps(exploration.to_dict(), indent=2))
    else:
        print(format_exploration(exploration, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
