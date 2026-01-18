#!/usr/bin/env python3
"""
Benchmark Thermal Sustained Performance CLI Tool

Benchmark sustained performance under thermal constraints.
Models thermal buildup, throttling, and steady-state performance.

Usage:
    # Benchmark sustained performance
    ./cli/benchmark_thermal_sustained.py --platform jetson-orin-nano --ambient 25

    # Extended duration test
    ./cli/benchmark_thermal_sustained.py --platform jetson-orin-nano --duration 60 --ambient 35

    # Compare platforms
    ./cli/benchmark_thermal_sustained.py --compare jetson-orin-nano jetson-orin-agx --ambient 30

    # JSON output
    ./cli/benchmark_thermal_sustained.py --platform jetson-orin-nano --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import math

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.hardware_mapper import (
    get_platform,
    list_platforms,
)


@dataclass
class ThermalPoint:
    """A point in the thermal simulation."""
    time_minutes: float
    junction_temp_c: float
    power_w: float
    throttle_pct: float  # 0-100, 100 = no throttling
    performance_pct: float  # Relative to peak

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_minutes": self.time_minutes,
            "junction_temp_c": self.junction_temp_c,
            "power_w": self.power_w,
            "throttle_pct": self.throttle_pct,
            "performance_pct": self.performance_pct,
        }


@dataclass
class ThermalBenchmarkResult:
    """Result of thermal sustained performance benchmark."""
    platform_name: str
    tdp_w: float

    # Test conditions
    ambient_temp_c: float
    duration_minutes: float

    # Thermal characteristics (estimated)
    thermal_resistance_c_per_w: float
    thermal_capacitance_j_per_c: float
    max_junction_temp_c: float

    # Timeline
    thermal_timeline: List[ThermalPoint]

    # Summary
    peak_power_w: float
    sustained_power_w: float
    time_to_throttle_minutes: float
    steady_state_temp_c: float
    steady_state_performance_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "tdp_w": self.tdp_w,
            "conditions": {
                "ambient_temp_c": self.ambient_temp_c,
                "duration_minutes": self.duration_minutes,
            },
            "thermal_characteristics": {
                "thermal_resistance_c_per_w": self.thermal_resistance_c_per_w,
                "thermal_capacitance_j_per_c": self.thermal_capacitance_j_per_c,
                "max_junction_temp_c": self.max_junction_temp_c,
            },
            "timeline": [p.to_dict() for p in self.thermal_timeline],
            "summary": {
                "peak_power_w": self.peak_power_w,
                "sustained_power_w": self.sustained_power_w,
                "time_to_throttle_minutes": self.time_to_throttle_minutes,
                "steady_state_temp_c": self.steady_state_temp_c,
                "steady_state_performance_pct": self.steady_state_performance_pct,
            },
        }


def estimate_thermal_params(tdp_w: float) -> Dict[str, float]:
    """Estimate thermal parameters based on TDP."""
    if tdp_w < 10:
        # Passive cooling
        return {
            "thermal_resistance": 15.0,
            "thermal_capacitance": 50.0,
            "max_junction": 95.0,
        }
    elif tdp_w < 30:
        # Small heatsink
        return {
            "thermal_resistance": 8.0,
            "thermal_capacitance": 100.0,
            "max_junction": 100.0,
        }
    elif tdp_w < 100:
        # Active cooling
        return {
            "thermal_resistance": 3.0,
            "thermal_capacitance": 200.0,
            "max_junction": 105.0,
        }
    else:
        # Advanced cooling
        return {
            "thermal_resistance": 1.0,
            "thermal_capacitance": 500.0,
            "max_junction": 110.0,
        }


def simulate_thermal(
    tdp_w: float,
    thermal_resistance: float,
    thermal_capacitance: float,
    max_junction: float,
    ambient_c: float,
    duration_minutes: float,
    time_step_seconds: float = 10.0,
) -> List[ThermalPoint]:
    """Simulate thermal behavior over time."""
    points = []

    # Initial conditions
    junction_temp = ambient_c
    current_power = tdp_w  # Start at full power

    # Thermal time constant (seconds)
    tau = thermal_resistance * thermal_capacitance

    total_seconds = duration_minutes * 60
    current_time = 0

    while current_time <= total_seconds:
        # Calculate equilibrium temperature at current power
        equilibrium_temp = ambient_c + current_power * thermal_resistance

        # Update junction temperature (exponential approach)
        delta_t = time_step_seconds
        junction_temp = junction_temp + (equilibrium_temp - junction_temp) * (1 - math.exp(-delta_t / tau))

        # Check for throttling
        throttle_pct = 100.0
        if junction_temp >= max_junction:
            # Throttle to maintain max junction temp
            allowable_power = (max_junction - ambient_c) / thermal_resistance
            throttle_pct = (allowable_power / tdp_w) * 100 if tdp_w > 0 else 0
            current_power = allowable_power
            junction_temp = max_junction
        elif junction_temp > max_junction - 5:
            # Start soft throttling as we approach limit
            margin = (max_junction - junction_temp) / 5
            throttle_pct = 90 + margin * 10
            current_power = tdp_w * (throttle_pct / 100)

        # Performance is proportional to power
        performance_pct = (current_power / tdp_w) * 100 if tdp_w > 0 else 0

        points.append(ThermalPoint(
            time_minutes=current_time / 60,
            junction_temp_c=junction_temp,
            power_w=current_power,
            throttle_pct=throttle_pct,
            performance_pct=performance_pct,
        ))

        current_time += time_step_seconds

    return points


def benchmark_thermal_sustained(
    platform_name: str,
    ambient_c: float = 25.0,
    duration_minutes: float = 30.0,
) -> ThermalBenchmarkResult:
    """Benchmark sustained thermal performance."""
    platform = get_platform(platform_name)
    if platform is None:
        raise ValueError(f"Unknown platform: {platform_name}")

    # Get thermal parameters
    params = estimate_thermal_params(platform.tdp_w)

    # Run simulation
    timeline = simulate_thermal(
        platform.tdp_w,
        params["thermal_resistance"],
        params["thermal_capacitance"],
        params["max_junction"],
        ambient_c,
        duration_minutes,
    )

    # Analyze results
    peak_power = max(p.power_w for p in timeline)

    # Find time to throttle
    time_to_throttle = duration_minutes
    for p in timeline:
        if p.throttle_pct < 100:
            time_to_throttle = p.time_minutes
            break

    # Steady state (last 20% of simulation)
    steady_start = int(len(timeline) * 0.8)
    steady_points = timeline[steady_start:]
    steady_power = sum(p.power_w for p in steady_points) / len(steady_points) if steady_points else peak_power
    steady_temp = sum(p.junction_temp_c for p in steady_points) / len(steady_points) if steady_points else ambient_c
    steady_perf = sum(p.performance_pct for p in steady_points) / len(steady_points) if steady_points else 100

    return ThermalBenchmarkResult(
        platform_name=platform_name,
        tdp_w=platform.tdp_w,
        ambient_temp_c=ambient_c,
        duration_minutes=duration_minutes,
        thermal_resistance_c_per_w=params["thermal_resistance"],
        thermal_capacitance_j_per_c=params["thermal_capacitance"],
        max_junction_temp_c=params["max_junction"],
        thermal_timeline=timeline,
        peak_power_w=peak_power,
        sustained_power_w=steady_power,
        time_to_throttle_minutes=time_to_throttle,
        steady_state_temp_c=steady_temp,
        steady_state_performance_pct=steady_perf,
    )


def format_thermal_result(result: ThermalBenchmarkResult) -> str:
    """Format thermal benchmark result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"  THERMAL SUSTAINED PERFORMANCE: {result.platform_name.upper()}")
    lines.append("=" * 80)
    lines.append("")

    # Platform specs
    lines.append("  Platform Specifications:")
    lines.append(f"    TDP:                   {result.tdp_w:.1f}W")
    lines.append(f"    Thermal Resistance:    {result.thermal_resistance_c_per_w:.1f} C/W")
    lines.append(f"    Max Junction Temp:     {result.max_junction_temp_c:.0f}C")
    lines.append("")

    # Test conditions
    lines.append("  Test Conditions:")
    lines.append(f"    Ambient Temperature:   {result.ambient_temp_c:.0f}C")
    lines.append(f"    Test Duration:         {result.duration_minutes:.0f} minutes")
    lines.append("")

    # Results summary
    lines.append("  Performance Summary:")
    lines.append(f"    Peak Power:            {result.peak_power_w:.1f}W")
    lines.append(f"    Sustained Power:       {result.sustained_power_w:.1f}W ({result.sustained_power_w/result.tdp_w*100:.0f}% TDP)")
    lines.append(f"    Time to Throttle:      {result.time_to_throttle_minutes:.1f} min")
    lines.append(f"    Steady State Temp:     {result.steady_state_temp_c:.1f}C")
    lines.append(f"    Steady State Perf:     {result.steady_state_performance_pct:.0f}%")
    lines.append("")

    # Thermal timeline
    lines.append("  Thermal Timeline:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Time':>8} {'Temp':>8} {'Power':>10} {'Throttle':>10} {'Perf':>8}")
    lines.append("  " + "-" * 70)

    # Sample points
    sample_indices = [0]
    sample_indices.extend(range(len(result.thermal_timeline)//10, len(result.thermal_timeline), len(result.thermal_timeline)//10))
    sample_indices.append(len(result.thermal_timeline) - 1)

    for i in sorted(set(sample_indices)):
        if i < len(result.thermal_timeline):
            p = result.thermal_timeline[i]
            lines.append(
                f"    {p.time_minutes:>7.1f}m {p.junction_temp_c:>7.1f}C {p.power_w:>9.1f}W "
                f"{p.throttle_pct:>9.0f}% {p.performance_pct:>7.0f}%"
            )

    lines.append("  " + "-" * 70)
    lines.append("")

    # Visual temperature curve
    lines.append("  Temperature Over Time:")
    lines.append("")
    bar_width = 40
    temp_range = result.max_junction_temp_c - result.ambient_temp_c

    for p in result.thermal_timeline[::max(1, len(result.thermal_timeline)//10)]:
        temp_frac = (p.junction_temp_c - result.ambient_temp_c) / temp_range if temp_range > 0 else 0
        bar_len = int(temp_frac * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        throttle_marker = "!" if p.throttle_pct < 100 else " "
        lines.append(f"    {p.time_minutes:>5.0f}m{throttle_marker}[{bar}] {p.junction_temp_c:.0f}C")

    lines.append("")
    lines.append("    ! = Throttling active")
    lines.append("")

    # Performance curve
    lines.append("  Performance Over Time:")
    lines.append("")
    for p in result.thermal_timeline[::max(1, len(result.thermal_timeline)//10)]:
        bar_len = int(p.performance_pct / 100 * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {p.time_minutes:>5.0f}m [{bar}] {p.performance_pct:.0f}%")
    lines.append("")

    return "\n".join(lines)


def format_thermal_comparison(results: List[ThermalBenchmarkResult]) -> str:
    """Format comparison of multiple platforms."""
    lines = []

    lines.append("=" * 80)
    lines.append("  THERMAL SUSTAINED PERFORMANCE COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    ambient = results[0].ambient_temp_c if results else 25
    lines.append(f"  Test Conditions: {ambient:.0f}C ambient")
    lines.append("")

    # Summary table
    lines.append("  Performance Summary:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Platform':<25} {'TDP':>8} {'Sustained':>10} {'Time2Thrt':>10} {'SS Perf':>10}")
    lines.append("  " + "-" * 76)

    for r in sorted(results, key=lambda x: x.steady_state_performance_pct, reverse=True):
        lines.append(
            f"    {r.platform_name:<25} {r.tdp_w:>7.1f}W {r.sustained_power_w:>9.1f}W "
            f"{r.time_to_throttle_minutes:>9.1f}m {r.steady_state_performance_pct:>9.0f}%"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Visual comparison
    lines.append("  Sustained Performance Comparison:")
    lines.append("")
    max_perf = 100
    bar_width = 40
    for r in sorted(results, key=lambda x: x.steady_state_performance_pct, reverse=True):
        bar_len = int((r.steady_state_performance_pct / max_perf) * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {r.platform_name:<25} [{bar}] {r.steady_state_performance_pct:.0f}%")
    lines.append("")

    # Thermal headroom comparison
    lines.append("  Thermal Headroom (at steady state):")
    for r in results:
        headroom = r.max_junction_temp_c - r.steady_state_temp_c
        lines.append(f"    {r.platform_name:<25} {headroom:>5.1f}C headroom (Tj: {r.steady_state_temp_c:.0f}C)")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sustained thermal performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark sustained performance
  ./cli/benchmark_thermal_sustained.py --platform jetson-orin-nano --ambient 25

  # Extended test
  ./cli/benchmark_thermal_sustained.py --platform jetson-orin-nano --duration 60 --ambient 35

  # Compare platforms
  ./cli/benchmark_thermal_sustained.py --compare jetson-orin-nano jetson-orin-agx --ambient 30

  # JSON output
  ./cli/benchmark_thermal_sustained.py --platform jetson-orin-nano --format json
"""
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        help="Platform to benchmark"
    )
    parser.add_argument(
        "--ambient", "-a",
        type=float,
        default=25.0,
        help="Ambient temperature in Celsius (default: 25)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Test duration in minutes (default: 30)"
    )
    parser.add_argument(
        "--compare", "-c",
        type=str,
        nargs="+",
        help="Compare multiple platforms"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available platforms"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list_platforms:
        print("Available platforms:")
        for name in sorted(list_platforms()):
            platform = get_platform(name)
            if platform:
                print(f"  {name:<30} TDP: {platform.tdp_w:>6.1f}W")
        return 0

    # Handle comparison
    if args.compare:
        results = []
        for name in args.compare:
            try:
                result = benchmark_thermal_sustained(name, args.ambient, args.duration)
                results.append(result)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        if args.format == "json":
            output = {"platforms": [r.to_dict() for r in results]}
            print(json.dumps(output, indent=2))
        else:
            print(format_thermal_comparison(results))
        return 0

    # Handle single platform
    if args.platform:
        try:
            result = benchmark_thermal_sustained(args.platform, args.ambient, args.duration)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_thermal_result(result))
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
