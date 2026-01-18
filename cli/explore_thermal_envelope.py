#!/usr/bin/env python3
"""
Explore Thermal Envelope CLI Tool

Analyze thermal constraints and their impact on sustained performance.
Shows how thermal limits affect power budgets over time.

Usage:
    # Explore thermal envelope for a platform
    ./cli/explore_thermal_envelope.py --platform jetson-orin-nano

    # With specific ambient temperature
    ./cli/explore_thermal_envelope.py --platform jetson-orin-nano --ambient 35

    # Compare platforms
    ./cli/explore_thermal_envelope.py --compare jetson-orin-nano jetson-orin-agx

    # JSON output
    ./cli/explore_thermal_envelope.py --platform jetson-orin-nano --format json
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
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.hardware_mapper import (
    HardwarePlatform,
    get_platform,
    list_platforms,
    HARDWARE_PLATFORMS,
)


@dataclass
class ThermalPoint:
    """A point on the thermal derating curve."""
    ambient_temp_c: float
    max_sustained_power_w: float
    derating_factor: float
    thermal_headroom_c: float
    operating_regime: str  # nominal, throttled, shutdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ambient_temp_c": self.ambient_temp_c,
            "max_sustained_power_w": self.max_sustained_power_w,
            "derating_factor": self.derating_factor,
            "thermal_headroom_c": self.thermal_headroom_c,
            "operating_regime": self.operating_regime,
        }


@dataclass
class ThermalEnvelopeAnalysis:
    """Complete thermal envelope analysis."""
    platform_name: str
    tdp_w: float
    max_junction_temp_c: float

    # Thermal characteristics
    thermal_resistance_c_per_w: float
    cooling_type: str

    # Derating curve
    thermal_points: List[ThermalPoint]

    # Operating limits
    nominal_ambient_max_c: float
    throttle_ambient_c: float
    shutdown_ambient_c: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "specs": {
                "tdp_w": self.tdp_w,
                "max_junction_temp_c": self.max_junction_temp_c,
                "thermal_resistance_c_per_w": self.thermal_resistance_c_per_w,
                "cooling_type": self.cooling_type,
            },
            "limits": {
                "nominal_ambient_max_c": self.nominal_ambient_max_c,
                "throttle_ambient_c": self.throttle_ambient_c,
                "shutdown_ambient_c": self.shutdown_ambient_c,
            },
            "derating_curve": [p.to_dict() for p in self.thermal_points],
        }


def estimate_thermal_characteristics(platform: HardwarePlatform) -> Dict[str, Any]:
    """Estimate thermal characteristics based on platform specs."""
    tdp = platform.tdp_w

    # Estimate thermal resistance based on TDP class
    if tdp < 10:
        thermal_resistance = 15.0  # Passive cooling
        cooling = "passive"
        max_junction = 95.0
    elif tdp < 30:
        thermal_resistance = 8.0  # Small heatsink
        cooling = "heatsink"
        max_junction = 100.0
    elif tdp < 100:
        thermal_resistance = 3.0  # Active fan
        cooling = "active-fan"
        max_junction = 105.0
    else:
        thermal_resistance = 1.0  # Advanced cooling
        cooling = "liquid/advanced"
        max_junction = 110.0

    return {
        "thermal_resistance": thermal_resistance,
        "cooling": cooling,
        "max_junction": max_junction,
    }


def calculate_derating(
    tdp_w: float,
    thermal_resistance: float,
    max_junction_c: float,
    ambient_c: float,
) -> ThermalPoint:
    """Calculate thermal derating at a given ambient temperature."""
    # Calculate max sustainable power at this ambient
    # P_max = (T_junction_max - T_ambient) / R_thermal
    max_power = (max_junction_c - ambient_c) / thermal_resistance

    # Clamp to TDP
    max_power = min(max_power, tdp_w)
    max_power = max(max_power, 0)

    derating = max_power / tdp_w if tdp_w > 0 else 0
    headroom = max_junction_c - ambient_c - (max_power * thermal_resistance)

    # Determine operating regime
    if derating >= 0.95:
        regime = "nominal"
    elif derating > 0.5:
        regime = "throttled"
    elif derating > 0:
        regime = "heavily-throttled"
    else:
        regime = "shutdown"

    return ThermalPoint(
        ambient_temp_c=ambient_c,
        max_sustained_power_w=max_power,
        derating_factor=derating,
        thermal_headroom_c=headroom,
        operating_regime=regime,
    )


def analyze_thermal_envelope(
    platform_name: str,
) -> ThermalEnvelopeAnalysis:
    """Analyze thermal envelope for a platform."""
    platform = get_platform(platform_name)
    if platform is None:
        raise ValueError(f"Unknown platform: {platform_name}")

    # Get thermal characteristics
    thermal = estimate_thermal_characteristics(platform)

    # Generate derating curve
    points = []
    for temp in range(-20, 71, 5):
        point = calculate_derating(
            platform.tdp_w,
            thermal["thermal_resistance"],
            thermal["max_junction"],
            temp,
        )
        points.append(point)

    # Find operating limits
    nominal_max = None
    throttle_start = None
    shutdown_temp = None

    for p in points:
        if p.derating_factor >= 0.95 and nominal_max is None:
            nominal_max = p.ambient_temp_c
        elif p.derating_factor < 0.95 and nominal_max is not None and throttle_start is None:
            throttle_start = p.ambient_temp_c
        if p.derating_factor <= 0 and shutdown_temp is None:
            shutdown_temp = p.ambient_temp_c

    # Set defaults if not found
    if nominal_max is None:
        nominal_max = 25.0
    if throttle_start is None:
        throttle_start = 40.0
    if shutdown_temp is None:
        shutdown_temp = thermal["max_junction"]

    return ThermalEnvelopeAnalysis(
        platform_name=platform_name,
        tdp_w=platform.tdp_w,
        max_junction_temp_c=thermal["max_junction"],
        thermal_resistance_c_per_w=thermal["thermal_resistance"],
        cooling_type=thermal["cooling"],
        thermal_points=points,
        nominal_ambient_max_c=nominal_max,
        throttle_ambient_c=throttle_start,
        shutdown_ambient_c=shutdown_temp,
    )


def format_thermal_analysis(analysis: ThermalEnvelopeAnalysis, ambient: Optional[float] = None) -> str:
    """Format thermal analysis as text."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"  THERMAL ENVELOPE: {analysis.platform_name.upper()}")
    lines.append("=" * 80)
    lines.append("")

    # Platform specs
    lines.append("  Platform Specifications:")
    lines.append(f"    TDP:                {analysis.tdp_w:.1f}W")
    lines.append(f"    Max Junction:       {analysis.max_junction_temp_c:.0f}C")
    lines.append(f"    Thermal Resistance: {analysis.thermal_resistance_c_per_w:.1f} C/W")
    lines.append(f"    Cooling:            {analysis.cooling_type}")
    lines.append("")

    # Operating limits
    lines.append("  Operating Limits:")
    lines.append(f"    Nominal (>95% TDP): up to {analysis.nominal_ambient_max_c:.0f}C ambient")
    lines.append(f"    Throttling starts:  {analysis.throttle_ambient_c:.0f}C ambient")
    lines.append(f"    Shutdown:           {analysis.shutdown_ambient_c:.0f}C ambient")
    lines.append("")

    # Specific ambient analysis
    if ambient is not None:
        point = calculate_derating(
            analysis.tdp_w,
            analysis.thermal_resistance_c_per_w,
            analysis.max_junction_temp_c,
            ambient,
        )
        lines.append(f"  At {ambient:.0f}C Ambient:")
        lines.append(f"    Max Sustained:    {point.max_sustained_power_w:.1f}W ({point.derating_factor*100:.0f}% of TDP)")
        lines.append(f"    Thermal Headroom: {point.thermal_headroom_c:.1f}C")
        lines.append(f"    Operating Regime: {point.operating_regime}")
        lines.append("")

    # Derating curve
    lines.append("  Thermal Derating Curve:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Ambient':>8} {'Max Power':>12} {'Derating':>10} {'Headroom':>10} {'Regime':>15}")
    lines.append("  " + "-" * 70)

    # Show subset of points
    for p in analysis.thermal_points:
        if p.ambient_temp_c % 10 == 0 or p.ambient_temp_c == 25:
            lines.append(
                f"    {p.ambient_temp_c:>7.0f}C {p.max_sustained_power_w:>11.1f}W "
                f"{p.derating_factor*100:>9.0f}% {p.thermal_headroom_c:>9.1f}C {p.operating_regime:>15}"
            )

    lines.append("  " + "-" * 70)
    lines.append("")

    # Visual derating curve
    lines.append("  Visual Derating Curve (power vs ambient temp):")
    lines.append("")
    bar_width = 40
    for p in analysis.thermal_points:
        if p.ambient_temp_c % 10 == 0:
            filled = int(p.derating_factor * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)
            regime_marker = "*" if p.operating_regime == "nominal" else " "
            lines.append(f"    {p.ambient_temp_c:>3.0f}C {regime_marker}[{bar}] {p.derating_factor*100:>3.0f}%")
    lines.append("")
    lines.append("    Legend: # = available power, . = derated, * = nominal regime")
    lines.append("")

    return "\n".join(lines)


def format_thermal_comparison(analyses: List[ThermalEnvelopeAnalysis], ambient: float = 25.0) -> str:
    """Format comparison of multiple platforms."""
    lines = []

    lines.append("=" * 80)
    lines.append("  THERMAL ENVELOPE COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Reference Ambient: {ambient:.0f}C")
    lines.append("")

    # Platform specs comparison
    lines.append("  Platform Specifications:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Platform':<25} {'TDP':>8} {'Max Tj':>8} {'R_th':>8} {'Cooling':>15}")
    lines.append("  " + "-" * 70)
    for a in analyses:
        lines.append(
            f"    {a.platform_name:<25} {a.tdp_w:>7.1f}W {a.max_junction_temp_c:>7.0f}C "
            f"{a.thermal_resistance_c_per_w:>7.1f} {a.cooling_type:>15}"
        )
    lines.append("  " + "-" * 70)
    lines.append("")

    # Performance at reference ambient
    lines.append(f"  Performance at {ambient:.0f}C:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Platform':<25} {'Sustained':>12} {'Derating':>10} {'Headroom':>10} {'Regime':>12}")
    lines.append("  " + "-" * 70)
    for a in analyses:
        point = calculate_derating(
            a.tdp_w,
            a.thermal_resistance_c_per_w,
            a.max_junction_temp_c,
            ambient,
        )
        lines.append(
            f"    {a.platform_name:<25} {point.max_sustained_power_w:>11.1f}W "
            f"{point.derating_factor*100:>9.0f}% {point.thermal_headroom_c:>9.1f}C {point.operating_regime:>12}"
        )
    lines.append("  " + "-" * 70)
    lines.append("")

    # Operating range comparison
    lines.append("  Operating Temperature Range:")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Platform':<25} {'Nominal Max':>12} {'Throttle':>10} {'Shutdown':>10}")
    lines.append("  " + "-" * 60)
    for a in analyses:
        lines.append(
            f"    {a.platform_name:<25} {a.nominal_ambient_max_c:>11.0f}C "
            f"{a.throttle_ambient_c:>9.0f}C {a.shutdown_ambient_c:>9.0f}C"
        )
    lines.append("  " + "-" * 60)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Explore thermal envelope and derating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore thermal envelope
  ./cli/explore_thermal_envelope.py --platform jetson-orin-nano

  # At specific ambient temperature
  ./cli/explore_thermal_envelope.py --platform jetson-orin-nano --ambient 35

  # Compare platforms
  ./cli/explore_thermal_envelope.py --compare jetson-orin-nano jetson-orin-agx

  # JSON output
  ./cli/explore_thermal_envelope.py --platform jetson-orin-nano --format json

  # List platforms
  ./cli/explore_thermal_envelope.py --list-platforms
"""
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        help="Platform to analyze"
    )
    parser.add_argument(
        "--ambient", "-a",
        type=float,
        help="Ambient temperature in Celsius"
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

    # Handle list platforms
    if args.list_platforms:
        print("Available platforms:")
        for name in sorted(list_platforms()):
            platform = get_platform(name)
            if platform:
                print(f"  {name:<30} TDP: {platform.tdp_w:>6.1f}W")
        return 0

    # Handle comparison
    if args.compare:
        analyses = []
        for name in args.compare:
            try:
                analysis = analyze_thermal_envelope(name)
                analyses.append(analysis)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        if args.format == "json":
            result = {
                "ambient_c": args.ambient or 25.0,
                "platforms": [a.to_dict() for a in analyses]
            }
            print(json.dumps(result, indent=2))
        else:
            print(format_thermal_comparison(analyses, args.ambient or 25.0))
        return 0

    # Handle single platform analysis
    if args.platform:
        try:
            analysis = analyze_thermal_envelope(args.platform)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(analysis.to_dict(), indent=2))
        else:
            print(format_thermal_analysis(analysis, args.ambient))
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
