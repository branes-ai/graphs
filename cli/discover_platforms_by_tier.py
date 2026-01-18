#!/usr/bin/env python3
"""
Discover Platforms by Capability Tier CLI Tool

Find hardware platforms that fit within a capability tier's power envelope.
Integrates with the hardware mapper registry to show available platforms
with their specifications.

Usage:
    # Find all platforms for micro-autonomy (1-10W)
    ./cli/discover_platforms_by_tier.py --tier micro-autonomy

    # Find platforms with specific constraints
    ./cli/discover_platforms_by_tier.py --tier embodied-ai --min-tops 20 --max-tdp 75

    # Filter by vendor
    ./cli/discover_platforms_by_tier.py --tier industrial-edge --vendor nvidia

    # Show detailed specs
    ./cli/discover_platforms_by_tier.py --tier micro-autonomy --verbose

    # JSON output for integration
    ./cli/discover_platforms_by_tier.py --tier micro-autonomy --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.mission.capability_tiers import (
    CapabilityTier,
    TierName,
    CAPABILITY_TIERS,
    get_tier_by_name,
    list_tier_names,
)

# Import hardware mappers
from graphs.hardware.mappers import (
    list_all_mappers,
    get_mapper_info,
)


@dataclass
class PlatformInfo:
    """Information about a hardware platform."""
    name: str
    display_name: str
    vendor: str
    category: str  # gpu, cpu, tpu, kpu, etc.
    tdp_w: float
    memory_gb: float
    tier_fit: TierName
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "category": self.category,
            "tdp_w": self.tdp_w,
            "memory_gb": self.memory_gb,
            "tier_fit": self.tier_fit.value,
            "description": self.description,
        }


def get_platform_info_from_registry(mapper_name: str) -> Optional[PlatformInfo]:
    """Extract platform info from the hardware mapper registry."""
    info = get_mapper_info(mapper_name)
    if info is None:
        return None

    tdp = info["default_tdp_w"]

    # Determine tier fit based on TDP
    if tdp <= 1.0:
        tier_fit = TierName.WEARABLE_AI
    elif tdp <= 10.0:
        tier_fit = TierName.MICRO_AUTONOMY
    elif tdp <= 30.0:
        tier_fit = TierName.INDUSTRIAL_EDGE
    elif tdp <= 100.0:
        tier_fit = TierName.EMBODIED_AI
    else:
        tier_fit = TierName.AUTOMOTIVE_AI

    return PlatformInfo(
        name=mapper_name,
        display_name=mapper_name,  # Use mapper name as display name
        vendor=info["vendor"],
        category=info["category"],
        tdp_w=tdp,
        memory_gb=info["memory_gb"],
        tier_fit=tier_fit,
        description=info["description"],
    )


def get_all_platforms() -> List[PlatformInfo]:
    """Get info for all available hardware platforms."""
    platforms = []
    for mapper_name in list_all_mappers():
        info = get_platform_info_from_registry(mapper_name)
        if info:
            platforms.append(info)
    return platforms


def filter_platforms(
    platforms: List[PlatformInfo],
    tier: Optional[TierName] = None,
    vendor: Optional[str] = None,
    category: Optional[str] = None,
    max_tdp: Optional[float] = None,
    min_memory_gb: Optional[float] = None,
) -> List[PlatformInfo]:
    """Filter platforms by various criteria."""
    result = platforms

    if tier:
        # Filter by tier's power range
        tier_obj = CAPABILITY_TIERS.get(tier)
        if tier_obj:
            result = [p for p in result if tier_obj.power_min_w <= p.tdp_w <= tier_obj.power_max_w]

    if vendor:
        vendor_lower = vendor.lower()
        result = [p for p in result if vendor_lower in p.vendor.lower()]

    if category:
        category_lower = category.lower()
        result = [p for p in result if p.category == category_lower]

    if max_tdp is not None:
        result = [p for p in result if p.tdp_w <= max_tdp]

    if min_memory_gb is not None:
        result = [p for p in result if p.memory_gb >= min_memory_gb]

    return result


def format_platforms_table(platforms: List[PlatformInfo], verbose: bool = False) -> str:
    """Format platforms as a table."""
    if not platforms:
        return "  No platforms found matching criteria.\n"

    lines = []

    if verbose:
        lines.append("  Platform                      | TDP (W) | Memory   | Category    | Vendor")
        lines.append("  " + "-" * 80)
        for p in sorted(platforms, key=lambda x: x.tdp_w):
            mem_str = f"{p.memory_gb:.0f}GB" if p.memory_gb >= 1 else f"{p.memory_gb*1024:.0f}MB"
            lines.append(
                f"  {p.display_name:<29} | {p.tdp_w:>7.1f} | {mem_str:>8} | {p.category:<11} | {p.vendor}"
            )
    else:
        lines.append("  Platform                      | TDP (W) | Memory   | Tier Fit")
        lines.append("  " + "-" * 70)
        for p in sorted(platforms, key=lambda x: x.tdp_w):
            tier_name = p.tier_fit.value.replace("-", " ").title()
            mem_str = f"{p.memory_gb:.0f}GB" if p.memory_gb >= 1 else f"{p.memory_gb*1024:.0f}MB"
            lines.append(
                f"  {p.display_name:<29} | {p.tdp_w:>7.1f} | {mem_str:>8} | {tier_name}"
            )

    return "\n".join(lines)


def format_tier_platforms(tier: CapabilityTier, platforms: List[PlatformInfo], verbose: bool = False) -> str:
    """Format platforms for a specific tier."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  PLATFORMS FOR {tier.display_name.upper()} ({tier.power_range_str})")
    lines.append("=" * 80)
    lines.append("")

    if platforms:
        lines.append(format_platforms_table(platforms, verbose))
        lines.append("")
        lines.append(f"  Found {len(platforms)} platform(s) in power range {tier.power_range_str}")

        # Summary stats
        if len(platforms) > 1:
            avg_tdp = sum(p.tdp_w for p in platforms) / len(platforms)
            max_memory = max(p.memory_gb for p in platforms)
            lines.append(f"  Average TDP: {avg_tdp:.1f}W")
            lines.append(f"  Maximum memory: {max_memory:.1f}GB")

        # Show vendor breakdown
        vendors = {}
        for p in platforms:
            vendors[p.vendor] = vendors.get(p.vendor, 0) + 1
        if len(vendors) > 1:
            vendor_str = ", ".join(f"{v}: {c}" for v, c in sorted(vendors.items(), key=lambda x: -x[1]))
            lines.append(f"  Vendors: {vendor_str}")
    else:
        lines.append("  No platforms found in this power range.")
        lines.append("")
        lines.append("  This may indicate:")
        lines.append("    - The tier's power envelope is very constrained")
        lines.append("    - Hardware mappers need to be added for this tier")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find hardware platforms by capability tier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find platforms for micro-autonomy tier
  ./cli/discover_platforms_by_tier.py --tier micro-autonomy

  # Find platforms with memory constraint
  ./cli/discover_platforms_by_tier.py --tier industrial-edge --min-memory 8

  # Filter by vendor
  ./cli/discover_platforms_by_tier.py --tier embodied-ai --vendor nvidia

  # Show all platforms with details
  ./cli/discover_platforms_by_tier.py --all --verbose

  # JSON output
  ./cli/discover_platforms_by_tier.py --tier micro-autonomy --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=list_tier_names(),
        help="Capability tier to search"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all platforms (no tier filter)"
    )
    parser.add_argument(
        "--vendor", "-V",
        type=str,
        help="Filter by vendor (e.g., nvidia, google, intel)"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["gpu", "cpu", "tpu", "kpu", "dsp", "dpu", "cgra", "accelerator", "dfm"],
        help="Filter by hardware category"
    )
    parser.add_argument(
        "--max-tdp",
        type=float,
        help="Maximum TDP in watts"
    )
    parser.add_argument(
        "--min-memory",
        type=float,
        help="Minimum memory in GB"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed platform information"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--rank-by", "-r",
        choices=["tdp", "memory", "vendor"],
        default="tdp",
        help="Ranking criteria (default: tdp)"
    )

    args = parser.parse_args()

    if not args.tier and not args.all:
        parser.print_help()
        print("\nError: Either --tier or --all is required", file=sys.stderr)
        return 1

    # Get all platforms
    all_platforms = get_all_platforms()

    # Apply filters
    tier_enum = None
    if args.tier:
        try:
            tier_enum = TierName(args.tier)
        except ValueError:
            pass

    filtered = filter_platforms(
        all_platforms,
        tier=tier_enum,
        vendor=args.vendor,
        category=args.category,
        max_tdp=args.max_tdp,
        min_memory_gb=args.min_memory,
    )

    # Sort by ranking criteria
    sort_key = {
        "tdp": lambda p: p.tdp_w,
        "memory": lambda p: -p.memory_gb,
        "vendor": lambda p: p.vendor,
    }[args.rank_by]
    filtered = sorted(filtered, key=sort_key)

    # Output
    if args.format == "json":
        output = {
            "tier": args.tier,
            "filters": {
                "vendor": args.vendor,
                "category": args.category,
                "max_tdp": args.max_tdp,
                "min_memory_gb": args.min_memory,
            },
            "platforms": [p.to_dict() for p in filtered],
            "count": len(filtered),
        }
        print(json.dumps(output, indent=2))
    else:
        if args.tier:
            tier = get_tier_by_name(args.tier)
            print(format_tier_platforms(tier, filtered, args.verbose))
        else:
            print("=" * 80)
            print("  ALL AVAILABLE PLATFORMS")
            print("=" * 80)
            print("")
            print(format_platforms_table(filtered, args.verbose))
            print("")
            print(f"  Total: {len(filtered)} platform(s)")
            print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
