#!/usr/bin/env python3
"""
Compare Tier Platforms CLI Tool

Compare all hardware platforms within a capability tier.
Shows efficiency rankings, power/performance tradeoffs, and recommendations.

Usage:
    # Compare micro-autonomy platforms
    ./cli/compare_tier_platforms.py --tier micro-autonomy

    # Compare with specific workload
    ./cli/compare_tier_platforms.py --tier industrial-edge --model yolov8n

    # Rank by efficiency
    ./cli/compare_tier_platforms.py --tier embodied-ai --rank-by efficiency

    # JSON output
    ./cli/compare_tier_platforms.py --tier micro-autonomy --format json
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
    list_mappers_by_tdp_range,
)
from graphs.mission.capability_tiers import (
    TierName,
    get_tier_by_name,
    CAPABILITY_TIERS,
)


@dataclass
class PlatformComparison:
    """Comparison data for a single platform."""
    name: str
    vendor: str
    category: str
    tdp_w: float
    memory_gb: float

    # Computed metrics
    estimated_throughput_fps: float = 0.0
    estimated_power_w: float = 0.0
    efficiency_fps_per_watt: float = 0.0

    # Ranking
    rank_by_tdp: int = 0
    rank_by_memory: int = 0
    rank_by_efficiency: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "category": self.category,
            "tdp_w": self.tdp_w,
            "memory_gb": self.memory_gb,
            "estimated_throughput_fps": self.estimated_throughput_fps,
            "estimated_power_w": self.estimated_power_w,
            "efficiency_fps_per_watt": self.efficiency_fps_per_watt,
            "rankings": {
                "by_tdp": self.rank_by_tdp,
                "by_memory": self.rank_by_memory,
                "by_efficiency": self.rank_by_efficiency,
            },
        }


@dataclass
class TierComparisonResult:
    """Complete tier comparison result."""
    tier_name: str
    tier_display_name: str
    power_range: str
    model_name: Optional[str]
    platforms: List[PlatformComparison]

    # Summary stats
    platform_count: int = 0
    avg_tdp_w: float = 0.0
    avg_memory_gb: float = 0.0

    # Recommendations
    best_efficiency: Optional[str] = None
    best_memory: Optional[str] = None
    lowest_power: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": {
                "name": self.tier_name,
                "display_name": self.tier_display_name,
                "power_range": self.power_range,
            },
            "model": self.model_name,
            "summary": {
                "platform_count": self.platform_count,
                "avg_tdp_w": self.avg_tdp_w,
                "avg_memory_gb": self.avg_memory_gb,
            },
            "recommendations": {
                "best_efficiency": self.best_efficiency,
                "best_memory": self.best_memory,
                "lowest_power": self.lowest_power,
            },
            "platforms": [p.to_dict() for p in self.platforms],
        }


def estimate_model_performance(platform_name: str, model_name: str) -> Dict[str, float]:
    """Estimate model performance on a platform."""
    info = get_mapper_info(platform_name)
    if info is None:
        return {"throughput_fps": 0.0, "power_w": 0.0}

    tdp = info["default_tdp_w"]

    # Base throughput estimation from TDP (simplified model)
    # Higher TDP generally means more compute capability
    if tdp <= 10:
        base_throughput = tdp * 8  # ~80 FPS at 10W for light models
    elif tdp <= 30:
        base_throughput = tdp * 6  # ~180 FPS at 30W
    elif tdp <= 100:
        base_throughput = tdp * 4  # ~400 FPS at 100W
    else:
        base_throughput = tdp * 2  # Datacenter scales differently

    # Model complexity factor
    model_lower = model_name.lower()
    if "resnet50" in model_lower or "resnet101" in model_lower:
        complexity_factor = 0.3
    elif "resnet18" in model_lower or "resnet34" in model_lower:
        complexity_factor = 0.6
    elif "yolov8x" in model_lower or "yolov8l" in model_lower:
        complexity_factor = 0.25
    elif "yolov8m" in model_lower:
        complexity_factor = 0.4
    elif "yolov8s" in model_lower:
        complexity_factor = 0.55
    elif "yolov8n" in model_lower:
        complexity_factor = 0.75
    elif "mobilenet" in model_lower:
        complexity_factor = 0.9
    elif "efficientnet_b0" in model_lower:
        complexity_factor = 0.8
    else:
        complexity_factor = 0.5

    throughput = base_throughput * complexity_factor

    # Power estimation (typical is 60-80% of TDP under load)
    power = tdp * 0.7

    return {
        "throughput_fps": throughput,
        "power_w": power,
    }


def compare_tier_platforms(
    tier_name: str,
    model_name: Optional[str] = None,
    rank_by: str = "efficiency",
) -> TierComparisonResult:
    """
    Compare all platforms within a capability tier.

    Args:
        tier_name: Capability tier name
        model_name: Optional model for performance estimation
        rank_by: Ranking criteria (efficiency, tdp, memory)

    Returns:
        TierComparisonResult with platform comparisons
    """
    tier = get_tier_by_name(tier_name)
    if tier is None:
        raise ValueError(f"Unknown tier: {tier_name}")

    # Get platforms in tier's power range
    platform_names = list_mappers_by_tdp_range(tier.power_min_w, tier.power_max_w)

    # Build comparison data
    platforms = []
    for name in platform_names:
        info = get_mapper_info(name)
        if info is None:
            continue

        platform = PlatformComparison(
            name=name,
            vendor=info["vendor"],
            category=info["category"],
            tdp_w=info["default_tdp_w"],
            memory_gb=info["memory_gb"],
        )

        # Estimate performance if model specified
        if model_name:
            perf = estimate_model_performance(name, model_name)
            platform.estimated_throughput_fps = perf["throughput_fps"]
            platform.estimated_power_w = perf["power_w"]
            if perf["power_w"] > 0:
                platform.efficiency_fps_per_watt = perf["throughput_fps"] / perf["power_w"]
        else:
            # Use TDP-based efficiency estimate
            platform.efficiency_fps_per_watt = 1.0 / platform.tdp_w  # Inverse TDP as proxy

        platforms.append(platform)

    # Assign rankings
    # By TDP (lower is better for power-constrained)
    for i, p in enumerate(sorted(platforms, key=lambda x: x.tdp_w)):
        p.rank_by_tdp = i + 1

    # By memory (higher is better)
    for i, p in enumerate(sorted(platforms, key=lambda x: -x.memory_gb)):
        p.rank_by_memory = i + 1

    # By efficiency (higher is better)
    for i, p in enumerate(sorted(platforms, key=lambda x: -x.efficiency_fps_per_watt)):
        p.rank_by_efficiency = i + 1

    # Sort by requested criteria
    if rank_by == "tdp":
        platforms.sort(key=lambda x: x.tdp_w)
    elif rank_by == "memory":
        platforms.sort(key=lambda x: -x.memory_gb)
    else:  # efficiency
        platforms.sort(key=lambda x: -x.efficiency_fps_per_watt)

    # Calculate summary stats
    if platforms:
        avg_tdp = sum(p.tdp_w for p in platforms) / len(platforms)
        avg_memory = sum(p.memory_gb for p in platforms) / len(platforms)

        best_efficiency = min(platforms, key=lambda x: x.rank_by_efficiency).name
        best_memory = min(platforms, key=lambda x: x.rank_by_memory).name
        lowest_power = min(platforms, key=lambda x: x.rank_by_tdp).name
    else:
        avg_tdp = 0.0
        avg_memory = 0.0
        best_efficiency = None
        best_memory = None
        lowest_power = None

    return TierComparisonResult(
        tier_name=tier_name,
        tier_display_name=tier.display_name,
        power_range=tier.power_range_str,
        model_name=model_name,
        platforms=platforms,
        platform_count=len(platforms),
        avg_tdp_w=avg_tdp,
        avg_memory_gb=avg_memory,
        best_efficiency=best_efficiency,
        best_memory=best_memory,
        lowest_power=lowest_power,
    )


def format_comparison(result: TierComparisonResult, verbose: bool = False) -> str:
    """Format comparison result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"  PLATFORM COMPARISON: {result.tier_display_name.upper()}")
    lines.append(f"  Power Range: {result.power_range}")
    if result.model_name:
        lines.append(f"  Workload: {result.model_name}")
    lines.append("=" * 80)
    lines.append("")

    if not result.platforms:
        lines.append("  No platforms found in this tier's power range.")
        lines.append("")
        return "\n".join(lines)

    # Summary
    lines.append("  Summary:")
    lines.append(f"    Platforms Found:    {result.platform_count}")
    lines.append(f"    Average TDP:        {result.avg_tdp_w:.1f}W")
    lines.append(f"    Average Memory:     {result.avg_memory_gb:.1f}GB")
    lines.append("")

    # Recommendations
    lines.append("  Recommendations:")
    lines.append(f"    Best Efficiency:    {result.best_efficiency}")
    lines.append(f"    Most Memory:        {result.best_memory}")
    lines.append(f"    Lowest Power:       {result.lowest_power}")
    lines.append("")

    # Platform table
    lines.append("  Platform Comparison:")
    lines.append("  " + "-" * 76)

    if result.model_name:
        lines.append(f"    {'Platform':<25} {'TDP':>7} {'Memory':>8} {'FPS':>8} {'Eff.':>10} {'Rank':>6}")
        lines.append("  " + "-" * 76)

        for p in result.platforms:
            eff_str = f"{p.efficiency_fps_per_watt:.2f}"
            rank_str = f"#{p.rank_by_efficiency}"
            lines.append(
                f"    {p.name:<25} {p.tdp_w:>6.1f}W {p.memory_gb:>7.1f}GB "
                f"{p.estimated_throughput_fps:>7.0f} {eff_str:>10} {rank_str:>6}"
            )
    else:
        lines.append(f"    {'Platform':<25} {'Vendor':<12} {'Category':<10} {'TDP':>7} {'Memory':>8}")
        lines.append("  " + "-" * 76)

        for p in result.platforms:
            lines.append(
                f"    {p.name:<25} {p.vendor:<12} {p.category:<10} {p.tdp_w:>6.1f}W {p.memory_gb:>7.1f}GB"
            )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Vendor breakdown
    vendors = {}
    for p in result.platforms:
        vendors[p.vendor] = vendors.get(p.vendor, 0) + 1

    if len(vendors) > 1:
        lines.append("  Vendor Breakdown:")
        for vendor, count in sorted(vendors.items(), key=lambda x: -x[1]):
            pct = count / len(result.platforms) * 100
            bar = "#" * int(pct / 5)
            lines.append(f"    {vendor:<15} {count:>2} [{bar:<20}] {pct:.0f}%")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare platforms within a capability tier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare micro-autonomy platforms
  ./cli/compare_tier_platforms.py --tier micro-autonomy

  # Compare with specific workload
  ./cli/compare_tier_platforms.py --tier industrial-edge --model yolov8n

  # Rank by memory
  ./cli/compare_tier_platforms.py --tier embodied-ai --rank-by memory

  # JSON output
  ./cli/compare_tier_platforms.py --tier micro-autonomy --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        required=True,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Capability tier to compare"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model for performance estimation (e.g., yolov8n, resnet18)"
    )
    parser.add_argument(
        "--rank-by", "-r",
        type=str,
        choices=["efficiency", "tdp", "memory"],
        default="efficiency",
        help="Ranking criteria (default: efficiency)"
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

    # Run comparison
    try:
        result = compare_tier_platforms(
            tier_name=args.tier,
            model_name=args.model,
            rank_by=args.rank_by,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_comparison(result, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
