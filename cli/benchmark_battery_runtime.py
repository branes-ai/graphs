#!/usr/bin/env python3
"""
Benchmark Battery Runtime CLI Tool

Benchmark battery runtime under various power loads and conditions.
Models discharge curves, temperature effects, and degradation.

Usage:
    # Benchmark battery runtime
    ./cli/benchmark_battery_runtime.py --battery medium-lipo-4s --power 15

    # With temperature consideration
    ./cli/benchmark_battery_runtime.py --battery large-lipo-6s --power 25 --temp 35

    # Compare batteries
    ./cli/benchmark_battery_runtime.py --compare small-lipo-2s medium-lipo-4s large-lipo-6s --power 20

    # JSON output
    ./cli/benchmark_battery_runtime.py --battery medium-lipo-4s --power 15 --format json
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

from graphs.mission.battery import (
    BatteryConfiguration,
    BatteryChemistry,
    BATTERY_CONFIGURATIONS,
    get_battery,
)


# Temperature derating factors
TEMP_DERATING = {
    -20: 0.60,  # 60% capacity at -20C
    -10: 0.75,
    0: 0.85,
    10: 0.92,
    20: 0.98,
    25: 1.00,  # Nominal
    30: 0.99,
    35: 0.97,
    40: 0.93,
    45: 0.88,
    50: 0.80,
}

# Chemistry-specific characteristics
CHEMISTRY_CHARACTERISTICS = {
    BatteryChemistry.LIPO: {
        "nominal_voltage": 3.7,
        "discharge_curve": "flat",  # Relatively flat discharge
        "temp_sensitivity": "medium",
        "cycle_life": 500,
        "self_discharge_pct_month": 5,
    },
    BatteryChemistry.LIION: {
        "nominal_voltage": 3.6,
        "discharge_curve": "sloped",
        "temp_sensitivity": "medium",
        "cycle_life": 800,
        "self_discharge_pct_month": 3,
    },
    BatteryChemistry.LIFE: {
        "nominal_voltage": 3.2,
        "discharge_curve": "very_flat",
        "temp_sensitivity": "low",
        "cycle_life": 2000,
        "self_discharge_pct_month": 2,
    },
    BatteryChemistry.NIMH: {
        "nominal_voltage": 1.2,
        "discharge_curve": "sloped",
        "temp_sensitivity": "high",
        "cycle_life": 1000,
        "self_discharge_pct_month": 20,
    },
}


@dataclass
class DischargePoint:
    """A point on the discharge curve."""
    time_minutes: float
    capacity_remaining_pct: float
    voltage_pct: float  # Relative to nominal
    power_w: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_minutes": self.time_minutes,
            "capacity_remaining_pct": self.capacity_remaining_pct,
            "voltage_pct": self.voltage_pct,
            "power_w": self.power_w,
        }


@dataclass
class RuntimeBenchmarkResult:
    """Battery runtime benchmark result."""
    battery_name: str
    chemistry: str
    capacity_wh: float
    weight_kg: float

    # Test conditions
    power_draw_w: float
    temperature_c: float
    temp_derating: float

    # Results
    effective_capacity_wh: float
    runtime_hours: float
    runtime_minutes: float
    discharge_curve: List[DischargePoint]

    # Efficiency metrics
    wh_per_kg: float
    runtime_per_kg_hours: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery": {
                "name": self.battery_name,
                "chemistry": self.chemistry,
                "capacity_wh": self.capacity_wh,
                "weight_kg": self.weight_kg,
            },
            "conditions": {
                "power_draw_w": self.power_draw_w,
                "temperature_c": self.temperature_c,
                "temp_derating": self.temp_derating,
            },
            "results": {
                "effective_capacity_wh": self.effective_capacity_wh,
                "runtime_hours": self.runtime_hours,
                "runtime_minutes": self.runtime_minutes,
            },
            "efficiency": {
                "wh_per_kg": self.wh_per_kg,
                "runtime_per_kg_hours": self.runtime_per_kg_hours,
            },
            "discharge_curve": [p.to_dict() for p in self.discharge_curve],
        }


def get_temp_derating(temperature_c: float) -> float:
    """Get temperature derating factor."""
    # Interpolate between known points
    temps = sorted(TEMP_DERATING.keys())
    if temperature_c <= temps[0]:
        return TEMP_DERATING[temps[0]]
    if temperature_c >= temps[-1]:
        return TEMP_DERATING[temps[-1]]

    for i in range(len(temps) - 1):
        if temps[i] <= temperature_c <= temps[i + 1]:
            t1, t2 = temps[i], temps[i + 1]
            d1, d2 = TEMP_DERATING[t1], TEMP_DERATING[t2]
            ratio = (temperature_c - t1) / (t2 - t1)
            return d1 + ratio * (d2 - d1)

    return 1.0


def generate_discharge_curve(
    capacity_wh: float,
    power_w: float,
    chemistry: BatteryChemistry,
    num_points: int = 20,
) -> List[DischargePoint]:
    """Generate a discharge curve."""
    points = []

    # Get chemistry characteristics
    char = CHEMISTRY_CHARACTERISTICS.get(chemistry, CHEMISTRY_CHARACTERISTICS[BatteryChemistry.LIPO])

    runtime_hours = capacity_wh / power_w if power_w > 0 else 0
    runtime_minutes = runtime_hours * 60

    for i in range(num_points + 1):
        time_frac = i / num_points
        time_minutes = time_frac * runtime_minutes

        # Capacity remaining (linear approximation)
        capacity_remaining = 100 * (1 - time_frac)

        # Voltage curve depends on chemistry
        if char["discharge_curve"] == "very_flat":
            # LiFe has very flat curve
            if time_frac < 0.9:
                voltage_pct = 100 - time_frac * 5
            else:
                voltage_pct = 95 - (time_frac - 0.9) * 50
        elif char["discharge_curve"] == "flat":
            # LiPo is relatively flat
            if time_frac < 0.8:
                voltage_pct = 100 - time_frac * 10
            else:
                voltage_pct = 92 - (time_frac - 0.8) * 40
        else:
            # Sloped discharge
            voltage_pct = 100 - time_frac * 30

        points.append(DischargePoint(
            time_minutes=time_minutes,
            capacity_remaining_pct=capacity_remaining,
            voltage_pct=voltage_pct,
            power_w=power_w,
        ))

    return points


def benchmark_battery_runtime(
    battery_name: str,
    power_w: float,
    temperature_c: float = 25.0,
) -> RuntimeBenchmarkResult:
    """Benchmark battery runtime."""
    battery = get_battery(battery_name)
    if battery is None:
        raise ValueError(f"Unknown battery: {battery_name}")

    # Get temperature derating
    temp_derating = get_temp_derating(temperature_c)

    # Calculate effective capacity
    effective_capacity = battery.capacity_wh * temp_derating * 0.9  # 90% depth of discharge

    # Calculate runtime
    runtime_hours = effective_capacity / power_w if power_w > 0 else 0
    runtime_minutes = runtime_hours * 60

    # Generate discharge curve
    chemistry = battery.chemistry or BatteryChemistry.LIPO
    discharge_curve = generate_discharge_curve(
        effective_capacity, power_w, chemistry
    )

    # Efficiency metrics
    wh_per_kg = battery.capacity_wh / battery.weight_kg if battery.weight_kg > 0 else 0
    runtime_per_kg = runtime_hours / battery.weight_kg if battery.weight_kg > 0 else 0

    return RuntimeBenchmarkResult(
        battery_name=battery_name,
        chemistry=chemistry.value if chemistry else "unknown",
        capacity_wh=battery.capacity_wh,
        weight_kg=battery.weight_kg,
        power_draw_w=power_w,
        temperature_c=temperature_c,
        temp_derating=temp_derating,
        effective_capacity_wh=effective_capacity,
        runtime_hours=runtime_hours,
        runtime_minutes=runtime_minutes,
        discharge_curve=discharge_curve,
        wh_per_kg=wh_per_kg,
        runtime_per_kg_hours=runtime_per_kg,
    )


def format_runtime_result(result: RuntimeBenchmarkResult) -> str:
    """Format runtime benchmark result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"  BATTERY RUNTIME BENCHMARK: {result.battery_name.upper()}")
    lines.append("=" * 80)
    lines.append("")

    # Battery specs
    lines.append("  Battery Specifications:")
    lines.append(f"    Chemistry:      {result.chemistry}")
    lines.append(f"    Capacity:       {result.capacity_wh:.0f} Wh")
    lines.append(f"    Weight:         {result.weight_kg:.2f} kg")
    lines.append(f"    Energy Density: {result.wh_per_kg:.0f} Wh/kg")
    lines.append("")

    # Test conditions
    lines.append("  Test Conditions:")
    lines.append(f"    Power Draw:     {result.power_draw_w:.1f} W")
    lines.append(f"    Temperature:    {result.temperature_c:.0f}C")
    lines.append(f"    Temp Derating:  {result.temp_derating*100:.0f}%")
    lines.append("")

    # Results
    lines.append("  Runtime Results:")
    lines.append(f"    Effective Capacity: {result.effective_capacity_wh:.1f} Wh")
    lines.append(f"    Runtime:            {result.runtime_hours:.2f} hours ({result.runtime_minutes:.0f} min)")
    lines.append(f"    Runtime/kg:         {result.runtime_per_kg_hours:.2f} hours/kg")
    lines.append("")

    # Discharge curve
    lines.append("  Discharge Curve:")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Time':>8} {'Capacity':>12} {'Voltage':>10}")
    lines.append("  " + "-" * 60)

    # Sample points
    for p in result.discharge_curve[::max(1, len(result.discharge_curve)//10)]:
        lines.append(
            f"    {p.time_minutes:>7.0f}m {p.capacity_remaining_pct:>11.0f}% {p.voltage_pct:>9.0f}%"
        )

    lines.append("  " + "-" * 60)
    lines.append("")

    # Visual discharge curve
    lines.append("  Visual Discharge Curve:")
    lines.append("")
    bar_width = 40
    for p in result.discharge_curve[::max(1, len(result.discharge_curve)//8)]:
        bar_len = int(p.capacity_remaining_pct / 100 * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {p.time_minutes:>5.0f}m [{bar}] {p.capacity_remaining_pct:.0f}%")
    lines.append("")

    return "\n".join(lines)


def format_comparison(results: List[RuntimeBenchmarkResult]) -> str:
    """Format comparison of multiple batteries."""
    lines = []

    lines.append("=" * 80)
    lines.append("  BATTERY RUNTIME COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    power = results[0].power_draw_w if results else 0
    temp = results[0].temperature_c if results else 25
    lines.append(f"  Test Conditions: {power:.1f}W @ {temp:.0f}C")
    lines.append("")

    # Summary table
    lines.append("  Runtime Comparison:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Battery':<22} {'Capacity':>10} {'Runtime':>10} {'Wh/kg':>10} {'h/kg':>10}")
    lines.append("  " + "-" * 76)

    for r in sorted(results, key=lambda x: x.runtime_hours, reverse=True):
        lines.append(
            f"    {r.battery_name:<22} {r.capacity_wh:>9.0f}Wh {r.runtime_hours:>9.2f}h "
            f"{r.wh_per_kg:>10.0f} {r.runtime_per_kg_hours:>10.2f}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Visual comparison
    lines.append("  Visual Runtime Comparison:")
    lines.append("")
    max_runtime = max(r.runtime_hours for r in results)
    bar_width = 40
    for r in sorted(results, key=lambda x: x.runtime_hours, reverse=True):
        bar_len = int((r.runtime_hours / max_runtime) * bar_width) if max_runtime > 0 else 0
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {r.battery_name:<22} [{bar}] {r.runtime_hours:.1f}h")
    lines.append("")

    # Efficiency comparison
    lines.append("  Efficiency Ranking (runtime per kg):")
    for i, r in enumerate(sorted(results, key=lambda x: x.runtime_per_kg_hours, reverse=True), 1):
        lines.append(f"    {i}. {r.battery_name:<25} {r.runtime_per_kg_hours:.2f} h/kg")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark battery runtime under various conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single battery
  ./cli/benchmark_battery_runtime.py --battery medium-lipo-4s --power 15

  # With temperature
  ./cli/benchmark_battery_runtime.py --battery large-lipo-6s --power 25 --temp 35

  # Compare batteries
  ./cli/benchmark_battery_runtime.py --compare small-lipo-2s medium-lipo-4s large-lipo-6s --power 20

  # JSON output
  ./cli/benchmark_battery_runtime.py --battery medium-lipo-4s --power 15 --format json
"""
    )

    parser.add_argument(
        "--battery", "-b",
        type=str,
        help="Battery to benchmark"
    )
    parser.add_argument(
        "--power", "-p",
        type=float,
        required=True,
        help="Power draw in watts"
    )
    parser.add_argument(
        "--temp", "-t",
        type=float,
        default=25.0,
        help="Temperature in Celsius (default: 25)"
    )
    parser.add_argument(
        "--compare", "-c",
        type=str,
        nargs="+",
        help="Compare multiple batteries"
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

    args = parser.parse_args()

    # Handle list option
    if args.list_batteries:
        print("Available batteries:")
        for name, battery in sorted(BATTERY_CONFIGURATIONS.items()):
            chem = battery.chemistry.value if battery.chemistry else "?"
            print(f"  {name:<25} {battery.capacity_wh:>6.0f}Wh  {battery.weight_kg:.2f}kg  {chem}")
        return 0

    # Handle comparison
    if args.compare:
        results = []
        for name in args.compare:
            try:
                result = benchmark_battery_runtime(name, args.power, args.temp)
                results.append(result)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        if args.format == "json":
            output = {"batteries": [r.to_dict() for r in results]}
            print(json.dumps(output, indent=2))
        else:
            print(format_comparison(results))
        return 0

    # Handle single battery
    if args.battery:
        try:
            result = benchmark_battery_runtime(args.battery, args.power, args.temp)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_runtime_result(result))
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
