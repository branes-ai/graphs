#!/usr/bin/env python3
"""
Estimate Power Consumption CLI Tool

Estimate total system power for a given hardware/model configuration.
Breaks down power by subsystem (perception, control, movement, overhead).

Usage:
    # Estimate for model + platform
    ./cli/estimate_power_consumption.py --platform Jetson-Orin-AGX-64GB --model resnet50

    # With batch size
    ./cli/estimate_power_consumption.py --platform Jetson-Orin-Nano-8GB --model yolov8n --batch-size 1

    # Using capability tier typical allocation
    ./cli/estimate_power_consumption.py --platform Jetson-Orin-NX-16GB --model resnet18 --tier industrial-edge

    # JSON output
    ./cli/estimate_power_consumption.py --platform H100-SXM5-80GB --model resnet50 --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.hardware.mappers import (
    list_all_mappers,
    get_mapper_info,
    get_mapper_by_name,
)
from graphs.mission.capability_tiers import (
    TierName,
    get_tier_by_name,
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.power_allocation import (
    PowerAllocation,
    SubsystemType,
    get_typical_allocation,
    get_application_allocation,
    get_allocation_strategy,
    ALLOCATION_BALANCED,
)


@dataclass
class PowerEstimate:
    """Power consumption estimate for a configuration."""
    platform_name: str
    model_name: str
    batch_size: int

    # Platform specs
    platform_tdp_w: float
    platform_typical_w: float  # Typical operating power

    # Subsystem breakdown
    perception_power_w: float
    control_power_w: float
    movement_power_w: float
    overhead_power_w: float

    # Totals
    compute_only_power_w: float  # Perception + Control (no movement)
    total_system_power_w: float

    # Utilization
    compute_utilization: float  # Fraction of TDP used for compute

    # Tier info
    tier_name: str
    allocation_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "model": self.model_name,
            "batch_size": self.batch_size,
            "platform_tdp_w": self.platform_tdp_w,
            "platform_typical_w": self.platform_typical_w,
            "subsystem_power": {
                "perception_w": self.perception_power_w,
                "control_w": self.control_power_w,
                "movement_w": self.movement_power_w,
                "overhead_w": self.overhead_power_w,
            },
            "compute_only_power_w": self.compute_only_power_w,
            "total_system_power_w": self.total_system_power_w,
            "compute_utilization": self.compute_utilization,
            "tier": self.tier_name,
            "allocation_strategy": self.allocation_strategy,
        }


def get_platform_power_specs(platform_name: str) -> Optional[Dict[str, float]]:
    """Get power specifications for a platform."""
    info = get_mapper_info(platform_name)
    if info is None:
        return None

    tdp = info["default_tdp_w"]

    # Estimate typical operating power (usually 60-80% of TDP)
    # Edge devices run closer to TDP, datacenter GPUs have more headroom
    if tdp <= 30:
        typical_ratio = 0.85  # Edge devices run near TDP
    elif tdp <= 100:
        typical_ratio = 0.75  # Medium systems
    else:
        typical_ratio = 0.65  # Datacenter has more headroom

    return {
        "tdp_w": tdp,
        "typical_w": tdp * typical_ratio,
        "idle_w": tdp * 0.15,  # Idle is typically 15% of TDP
    }


def estimate_model_power(
    platform_name: str,
    model_name: str,
    batch_size: int = 1,
) -> Optional[Dict[str, float]]:
    """
    Estimate power consumption for running a model on a platform.

    Uses the UnifiedAnalyzer if available, otherwise estimates from TDP.
    """
    platform_specs = get_platform_power_specs(platform_name)
    if platform_specs is None:
        return None

    # Try to use UnifiedAnalyzer for more accurate estimates
    try:
        from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig

        analyzer = UnifiedAnalyzer()
        config = AnalysisConfig(
            run_roofline=True,
            run_energy=True,
            run_hardware_mapping=True,
        )

        # Map platform name to hardware name format
        hardware_name = platform_name.replace("-", "_")
        result = analyzer.analyze_model(model_name, hardware_name, batch_size=batch_size, config=config)

        if result and result.energy_report:
            # Get energy per inference and convert to power at typical throughput
            energy_j = result.energy_report.total_energy_j
            latency_s = result.derived_metrics.get("latency_ms", 10.0) / 1000.0

            # Power = Energy / Time
            model_power_w = energy_j / latency_s if latency_s > 0 else platform_specs["typical_w"] * 0.5

            return {
                "model_power_w": min(model_power_w, platform_specs["tdp_w"]),
                "from_analysis": True,
            }
    except Exception:
        pass

    # Fallback: estimate based on model complexity heuristics
    model_lower = model_name.lower()

    # Simple complexity-based power estimate
    if "resnet50" in model_lower or "resnet101" in model_lower:
        complexity = 0.7
    elif "resnet18" in model_lower or "resnet34" in model_lower:
        complexity = 0.4
    elif "yolo" in model_lower:
        if "yolov8x" in model_lower or "yolov8l" in model_lower:
            complexity = 0.6
        elif "yolov8m" in model_lower:
            complexity = 0.45
        else:  # yolov8n, yolov8s
            complexity = 0.3
    elif "mobilenet" in model_lower:
        complexity = 0.25
    elif "efficientnet" in model_lower:
        complexity = 0.35
    elif "vit" in model_lower or "transformer" in model_lower:
        complexity = 0.65
    else:
        complexity = 0.5  # Default middle ground

    # Batch size increases power (sublinear)
    batch_factor = 1.0 + 0.1 * (batch_size - 1) ** 0.5

    model_power = platform_specs["typical_w"] * complexity * batch_factor

    return {
        "model_power_w": min(model_power, platform_specs["tdp_w"]),
        "from_analysis": False,
    }


def estimate_power_consumption(
    platform_name: str,
    model_name: str,
    batch_size: int = 1,
    allocation: Optional[PowerAllocation] = None,
    movement_power_w: float = 0.0,
) -> Optional[PowerEstimate]:
    """
    Estimate power consumption for a configuration.

    Args:
        platform_name: Hardware platform name from registry
        model_name: Model name (e.g., resnet18, yolov8n)
        batch_size: Inference batch size
        allocation: Power allocation strategy (default: balanced)
        movement_power_w: External movement power (motors, etc.)

    Returns:
        PowerEstimate with breakdown by subsystem
    """
    # Get platform specs
    platform_specs = get_platform_power_specs(platform_name)
    if platform_specs is None:
        return None

    # Get model power estimate
    model_power = estimate_model_power(platform_name, model_name, batch_size)
    if model_power is None:
        return None

    # Use provided allocation or determine from tier
    if allocation is None:
        tier = get_tier_for_power(platform_specs["tdp_w"])
        if tier:
            allocation = get_typical_allocation(tier.name)
            tier_name = tier.name.value
        else:
            allocation = ALLOCATION_BALANCED
            tier_name = "unknown"
    else:
        tier = get_tier_for_power(platform_specs["tdp_w"])
        tier_name = tier.name.value if tier else "unknown"

    # Compute power is the model inference power
    compute_power = model_power["model_power_w"]

    # Allocate compute power between perception and control
    # Perception gets the DNN inference, control gets a fraction for planning
    perception_ratio = allocation.perception_ratio / (allocation.perception_ratio + allocation.control_ratio)
    control_ratio = allocation.control_ratio / (allocation.perception_ratio + allocation.control_ratio)

    perception_power = compute_power * perception_ratio
    control_power = compute_power * control_ratio

    # Overhead is proportional to total compute
    overhead_power = compute_power * (allocation.overhead_ratio / (1 - allocation.movement_ratio - allocation.overhead_ratio))

    # Movement power is external (motors, actuators)
    # If not provided, estimate from allocation and tier typical power
    if movement_power_w <= 0:
        tier = get_tier_for_power(platform_specs["tdp_w"])
        if tier:
            tier_typical = tier.typical_power_w
            movement_power_w = tier_typical * allocation.movement_ratio
        else:
            movement_power_w = 0.0

    # Total power
    compute_only = perception_power + control_power
    total_power = perception_power + control_power + movement_power_w + overhead_power

    # Compute utilization
    utilization = compute_power / platform_specs["tdp_w"]

    return PowerEstimate(
        platform_name=platform_name,
        model_name=model_name,
        batch_size=batch_size,
        platform_tdp_w=platform_specs["tdp_w"],
        platform_typical_w=platform_specs["typical_w"],
        perception_power_w=perception_power,
        control_power_w=control_power,
        movement_power_w=movement_power_w,
        overhead_power_w=overhead_power,
        compute_only_power_w=compute_only,
        total_system_power_w=total_power,
        compute_utilization=utilization,
        tier_name=tier_name,
        allocation_strategy=allocation.description if allocation.description else "custom",
    )


def format_power_estimate(estimate: PowerEstimate, verbose: bool = False) -> str:
    """Format power estimate as text output."""
    lines = []

    lines.append("=" * 70)
    lines.append("  POWER CONSUMPTION ESTIMATE")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    lines.append("  Configuration:")
    lines.append(f"    Platform:     {estimate.platform_name}")
    lines.append(f"    Model:        {estimate.model_name}")
    lines.append(f"    Batch Size:   {estimate.batch_size}")
    lines.append(f"    Tier:         {estimate.tier_name.replace('-', ' ').title()}")
    lines.append("")

    # Platform specs
    lines.append("  Platform Power Specs:")
    lines.append(f"    TDP:          {estimate.platform_tdp_w:.1f}W")
    lines.append(f"    Typical:      {estimate.platform_typical_w:.1f}W")
    lines.append("")

    # Subsystem breakdown
    lines.append("  Subsystem Power Breakdown:")
    lines.append("  " + "-" * 40)

    # Bar chart visualization
    max_power = max(estimate.perception_power_w, estimate.control_power_w,
                   estimate.movement_power_w, estimate.overhead_power_w, 0.1)
    bar_width = 25

    subsystems = [
        ("Perception", estimate.perception_power_w, "P"),
        ("Control", estimate.control_power_w, "C"),
        ("Movement", estimate.movement_power_w, "M"),
        ("Overhead", estimate.overhead_power_w, "O"),
    ]

    for name, power, symbol in subsystems:
        bar_len = int((power / max_power) * bar_width) if max_power > 0 else 0
        bar = symbol * bar_len
        pct = (power / estimate.total_system_power_w * 100) if estimate.total_system_power_w > 0 else 0
        lines.append(f"    {name:12} {power:6.1f}W  [{bar:<{bar_width}}] {pct:5.1f}%")

    lines.append("  " + "-" * 40)
    lines.append(f"    {'Compute Only':12} {estimate.compute_only_power_w:6.1f}W  (perception + control)")
    lines.append(f"    {'TOTAL':12} {estimate.total_system_power_w:6.1f}W")
    lines.append("")

    # Utilization
    lines.append("  Compute Utilization:")
    util_pct = estimate.compute_utilization * 100
    util_bar = "#" * int(util_pct / 5)
    lines.append(f"    [{util_bar:<20}] {util_pct:.1f}% of TDP")
    lines.append("")

    # Strategy info
    if verbose:
        lines.append(f"  Allocation Strategy: {estimate.allocation_strategy}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate power consumption for hardware/model configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic estimation
  ./cli/estimate_power_consumption.py --platform Jetson-Orin-Nano-8GB --model resnet18

  # With tier-specific allocation
  ./cli/estimate_power_consumption.py --platform Jetson-Orin-NX-16GB --model yolov8n --tier industrial-edge

  # With allocation strategy
  ./cli/estimate_power_consumption.py --platform Jetson-Orin-AGX-64GB --model resnet50 --allocation perception-heavy

  # With external movement power
  ./cli/estimate_power_consumption.py --platform Jetson-Orin-NX-16GB --model yolov8n --movement-power 30

  # JSON output
  ./cli/estimate_power_consumption.py --platform H100-SXM5-80GB --model resnet50 --format json
"""
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        required=True,
        help="Hardware platform name from registry"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name (e.g., resnet18, yolov8n, mobilenet_v2)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Inference batch size (default: 1)"
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Use tier-specific power allocation"
    )
    parser.add_argument(
        "--allocation", "-a",
        type=str,
        choices=["perception-heavy", "balanced", "control-heavy", "movement-heavy", "stationary"],
        help="Power allocation strategy"
    )
    parser.add_argument(
        "--movement-power",
        type=float,
        default=0.0,
        help="External movement power in watts (motors, actuators)"
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
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available platforms and exit"
    )

    args = parser.parse_args()

    # Handle list platforms
    if args.list_platforms:
        print("Available platforms:")
        for name in sorted(list_all_mappers()):
            info = get_mapper_info(name)
            if info:
                print(f"  {name:<30} {info['default_tdp_w']:>6.1f}W  {info['vendor']}")
        return 0

    # Validate platform exists
    if get_mapper_info(args.platform) is None:
        print(f"Error: Unknown platform '{args.platform}'", file=sys.stderr)
        print("Use --list-platforms to see available options", file=sys.stderr)
        return 1

    # Determine allocation
    allocation = None
    if args.tier:
        tier_name = TierName(args.tier)
        allocation = get_typical_allocation(tier_name)
    elif args.allocation:
        allocation = get_allocation_strategy(args.allocation)

    # Estimate power
    estimate = estimate_power_consumption(
        platform_name=args.platform,
        model_name=args.model,
        batch_size=args.batch_size,
        allocation=allocation,
        movement_power_w=args.movement_power,
    )

    if estimate is None:
        print(f"Error: Could not estimate power for configuration", file=sys.stderr)
        return 1

    # Output
    if args.format == "json":
        print(json.dumps(estimate.to_dict(), indent=2))
    else:
        print(format_power_estimate(estimate, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
