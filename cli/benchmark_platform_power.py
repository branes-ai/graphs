#!/usr/bin/env python3
"""
Benchmark Platform Power CLI Tool

Benchmark and characterize platform power consumption under different workloads.
Simulates various operating conditions to estimate power profiles.

Usage:
    # Benchmark a platform
    ./cli/benchmark_platform_power.py --platform jetson-orin-nano

    # With specific workloads
    ./cli/benchmark_platform_power.py --platform jetson-orin-nano --workloads idle inference full

    # Compare platforms
    ./cli/benchmark_platform_power.py --compare jetson-orin-nano jetson-orin-agx

    # JSON output
    ./cli/benchmark_platform_power.py --platform jetson-orin-nano --format json
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

from graphs.mission.hardware_mapper import (
    get_platform,
    list_platforms,
    HARDWARE_PLATFORMS,
)


class WorkloadType(Enum):
    """Workload types for benchmarking."""
    IDLE = "idle"                # System idle, minimal processing
    LIGHT = "light"              # Light processing (sensor polling)
    INFERENCE = "inference"      # ML inference workload
    CONTROL = "control"          # Control loop processing
    FULL = "full"               # Full utilization
    BURST = "burst"             # Peak burst load


# Workload power profiles (as fraction of TDP)
WORKLOAD_PROFILES = {
    WorkloadType.IDLE: {
        "description": "System idle, minimal activity",
        "power_fraction": 0.15,
        "variance": 0.02,
    },
    WorkloadType.LIGHT: {
        "description": "Light processing (sensors, comms)",
        "power_fraction": 0.30,
        "variance": 0.05,
    },
    WorkloadType.INFERENCE: {
        "description": "ML inference workload",
        "power_fraction": 0.65,
        "variance": 0.10,
    },
    WorkloadType.CONTROL: {
        "description": "Control loop processing",
        "power_fraction": 0.45,
        "variance": 0.08,
    },
    WorkloadType.FULL: {
        "description": "Full system utilization",
        "power_fraction": 0.90,
        "variance": 0.05,
    },
    WorkloadType.BURST: {
        "description": "Peak burst load",
        "power_fraction": 1.0,
        "variance": 0.02,
    },
}


@dataclass
class WorkloadResult:
    """Result of a single workload benchmark."""
    workload: str
    description: str

    # Power measurements (simulated)
    avg_power_w: float
    min_power_w: float
    max_power_w: float
    variance_w: float

    # Utilization
    power_fraction: float  # Fraction of TDP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload": self.workload,
            "description": self.description,
            "power": {
                "avg_w": self.avg_power_w,
                "min_w": self.min_power_w,
                "max_w": self.max_power_w,
                "variance_w": self.variance_w,
            },
            "power_fraction": self.power_fraction,
        }


@dataclass
class PlatformBenchmarkResult:
    """Complete platform benchmark result."""
    platform_name: str
    tdp_w: float
    vendor: str
    category: str

    workload_results: List[WorkloadResult]

    # Summary statistics
    idle_power_w: float
    typical_power_w: float
    peak_power_w: float
    power_efficiency: float  # Compute per watt metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "specs": {
                "tdp_w": self.tdp_w,
                "vendor": self.vendor,
                "category": self.category,
            },
            "workloads": [w.to_dict() for w in self.workload_results],
            "summary": {
                "idle_power_w": self.idle_power_w,
                "typical_power_w": self.typical_power_w,
                "peak_power_w": self.peak_power_w,
                "power_efficiency": self.power_efficiency,
            },
        }


def benchmark_workload(
    platform_tdp_w: float,
    workload: WorkloadType,
) -> WorkloadResult:
    """Benchmark a single workload (simulated)."""
    profile = WORKLOAD_PROFILES[workload]

    avg_power = platform_tdp_w * profile["power_fraction"]
    variance = platform_tdp_w * profile["variance"]

    min_power = max(0, avg_power - 2 * variance)
    max_power = min(platform_tdp_w, avg_power + 2 * variance)

    return WorkloadResult(
        workload=workload.value,
        description=profile["description"],
        avg_power_w=avg_power,
        min_power_w=min_power,
        max_power_w=max_power,
        variance_w=variance,
        power_fraction=profile["power_fraction"],
    )


def benchmark_platform(
    platform_name: str,
    workloads: Optional[List[str]] = None,
) -> PlatformBenchmarkResult:
    """Benchmark a platform under various workloads."""
    platform = get_platform(platform_name)
    if platform is None:
        raise ValueError(f"Unknown platform: {platform_name}")

    # Determine workloads to run
    if workloads:
        workload_types = [WorkloadType(w) for w in workloads]
    else:
        workload_types = list(WorkloadType)

    # Run benchmarks
    results = []
    for workload in workload_types:
        result = benchmark_workload(platform.tdp_w, workload)
        results.append(result)

    # Calculate summary statistics
    idle_result = next((r for r in results if r.workload == "idle"), None)
    inference_result = next((r for r in results if r.workload == "inference"), None)
    burst_result = next((r for r in results if r.workload == "burst"), None)

    idle_power = idle_result.avg_power_w if idle_result else platform.tdp_w * 0.15
    typical_power = inference_result.avg_power_w if inference_result else platform.tdp_w * 0.65
    peak_power = burst_result.avg_power_w if burst_result else platform.tdp_w

    # Power efficiency (estimated TOPS/W or similar)
    power_efficiency = 1.0 / typical_power if typical_power > 0 else 0

    return PlatformBenchmarkResult(
        platform_name=platform_name,
        tdp_w=platform.tdp_w,
        vendor=platform.vendor,
        category=platform.category,
        workload_results=results,
        idle_power_w=idle_power,
        typical_power_w=typical_power,
        peak_power_w=peak_power,
        power_efficiency=power_efficiency,
    )


def format_benchmark_result(result: PlatformBenchmarkResult) -> str:
    """Format benchmark result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"  PLATFORM POWER BENCHMARK: {result.platform_name.upper()}")
    lines.append("=" * 80)
    lines.append("")

    # Platform specs
    lines.append("  Platform Specifications:")
    lines.append(f"    TDP:      {result.tdp_w:.1f}W")
    lines.append(f"    Vendor:   {result.vendor}")
    lines.append(f"    Category: {result.category}")
    lines.append("")

    # Workload results
    lines.append("  Workload Power Profile:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Workload':<12} {'Avg':>8} {'Min':>8} {'Max':>8} {'Var':>8} {'%TDP':>8}")
    lines.append("  " + "-" * 70)

    for w in result.workload_results:
        lines.append(
            f"    {w.workload:<12} {w.avg_power_w:>7.1f}W {w.min_power_w:>7.1f}W "
            f"{w.max_power_w:>7.1f}W {w.variance_w:>7.2f}W {w.power_fraction*100:>7.0f}%"
        )

    lines.append("  " + "-" * 70)
    lines.append("")

    # Visual power profile
    lines.append("  Visual Power Profile:")
    lines.append("")
    bar_width = 40
    for w in result.workload_results:
        bar_len = int(w.power_fraction * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {w.workload:<12} [{bar}] {w.avg_power_w:.1f}W")
    lines.append("")

    # Summary
    lines.append("  Summary:")
    lines.append(f"    Idle Power:     {result.idle_power_w:.1f}W ({result.idle_power_w/result.tdp_w*100:.0f}% TDP)")
    lines.append(f"    Typical Power:  {result.typical_power_w:.1f}W ({result.typical_power_w/result.tdp_w*100:.0f}% TDP)")
    lines.append(f"    Peak Power:     {result.peak_power_w:.1f}W ({result.peak_power_w/result.tdp_w*100:.0f}% TDP)")
    lines.append("")

    # Power states diagram
    lines.append("  Power States:")
    lines.append(f"    IDLE ----[{result.idle_power_w:.1f}W]---- TYPICAL ----[{result.typical_power_w:.1f}W]---- PEAK ----[{result.peak_power_w:.1f}W]")
    lines.append("")

    return "\n".join(lines)


def format_benchmark_comparison(results: List[PlatformBenchmarkResult]) -> str:
    """Format comparison of multiple platforms."""
    lines = []

    lines.append("=" * 80)
    lines.append("  PLATFORM POWER COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Summary comparison
    lines.append("  Power Summary:")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'Platform':<25} {'TDP':>8} {'Idle':>8} {'Typical':>10} {'Peak':>8}")
    lines.append("  " + "-" * 70)

    for r in results:
        lines.append(
            f"    {r.platform_name:<25} {r.tdp_w:>7.1f}W {r.idle_power_w:>7.1f}W "
            f"{r.typical_power_w:>9.1f}W {r.peak_power_w:>7.1f}W"
        )

    lines.append("  " + "-" * 70)
    lines.append("")

    # Workload comparison (inference)
    lines.append("  Inference Workload Comparison:")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Platform':<25} {'Power':>10} {'%TDP':>8} {'Efficiency':>12}")
    lines.append("  " + "-" * 60)

    for r in results:
        inference = next((w for w in r.workload_results if w.workload == "inference"), None)
        if inference:
            efficiency = 1.0 / inference.avg_power_w if inference.avg_power_w > 0 else 0
            lines.append(
                f"    {r.platform_name:<25} {inference.avg_power_w:>9.1f}W "
                f"{inference.power_fraction*100:>7.0f}% {efficiency:>12.3f}"
            )

    lines.append("  " + "-" * 60)
    lines.append("")

    # Visual comparison
    lines.append("  Visual Power Comparison (typical workload):")
    lines.append("")
    max_power = max(r.typical_power_w for r in results)
    bar_width = 40
    for r in results:
        bar_len = int((r.typical_power_w / max_power) * bar_width) if max_power > 0 else 0
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        lines.append(f"    {r.platform_name:<25} [{bar}] {r.typical_power_w:.1f}W")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark platform power consumption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a platform
  ./cli/benchmark_platform_power.py --platform jetson-orin-nano

  # Specific workloads
  ./cli/benchmark_platform_power.py --platform jetson-orin-nano --workloads idle inference full

  # Compare platforms
  ./cli/benchmark_platform_power.py --compare jetson-orin-nano jetson-orin-agx raspberry-pi-5

  # JSON output
  ./cli/benchmark_platform_power.py --platform jetson-orin-nano --format json
"""
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        help="Platform to benchmark"
    )
    parser.add_argument(
        "--workloads", "-w",
        type=str,
        nargs="+",
        choices=["idle", "light", "inference", "control", "full", "burst"],
        help="Workloads to benchmark"
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
    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List available workloads"
    )

    args = parser.parse_args()

    # Handle list options
    if args.list_platforms:
        print("Available platforms:")
        for name in sorted(list_platforms()):
            platform = get_platform(name)
            if platform:
                print(f"  {name:<30} TDP: {platform.tdp_w:>6.1f}W  ({platform.category})")
        return 0

    if args.list_workloads:
        print("Available workloads:")
        for workload, profile in WORKLOAD_PROFILES.items():
            print(f"  {workload.value:<12}: {profile['description']}")
        return 0

    # Handle comparison
    if args.compare:
        results = []
        for name in args.compare:
            try:
                result = benchmark_platform(name, args.workloads)
                results.append(result)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        if args.format == "json":
            output = {"platforms": [r.to_dict() for r in results]}
            print(json.dumps(output, indent=2))
        else:
            print(format_benchmark_comparison(results))
        return 0

    # Handle single platform
    if args.platform:
        try:
            result = benchmark_platform(args.platform, args.workloads)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_benchmark_result(result))
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
