#!/usr/bin/env python3
"""
Estimate Perception Budget CLI Tool

Estimate the power budget needed for perception requirements.
Shows how different inference rates map to power consumption.

Usage:
    # Estimate budget for target FPS
    ./cli/estimate_perception_budget.py --target-fps 30

    # For specific model complexity
    ./cli/estimate_perception_budget.py --target-fps 30 --model-complexity high

    # With specific tier
    ./cli/estimate_perception_budget.py --tier micro-autonomy --target-fps 15

    # JSON output
    ./cli/estimate_perception_budget.py --target-fps 30 --format json
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

from graphs.mission.capability_tiers import (
    TierName,
    get_tier_by_name,
    CAPABILITY_TIERS,
)
from graphs.mission.hardware_mapper import (
    get_platform,
    list_platforms,
    HARDWARE_PLATFORMS,
)


class ModelComplexity(Enum):
    """Model complexity levels."""
    MINIMAL = "minimal"  # MobileNet-V3-small, ~50 MFLOPs
    LOW = "low"          # MobileNet-V2, ~300 MFLOPs
    MEDIUM = "medium"    # EfficientNet-B0, ~400 MFLOPs
    HIGH = "high"        # ResNet-50, ~4 GFLOPs
    VERY_HIGH = "very-high"  # ViT-B, ~17 GFLOPs


# Power per FPS estimates (W per FPS) for different complexity levels
# Based on typical edge inference benchmarks
POWER_PER_FPS = {
    ModelComplexity.MINIMAL: 0.02,    # 20mW per FPS
    ModelComplexity.LOW: 0.05,        # 50mW per FPS
    ModelComplexity.MEDIUM: 0.08,     # 80mW per FPS
    ModelComplexity.HIGH: 0.3,        # 300mW per FPS
    ModelComplexity.VERY_HIGH: 1.0,   # 1W per FPS
}

# Example models per complexity
MODEL_EXAMPLES = {
    ModelComplexity.MINIMAL: ["MobileNet-V3-Small", "SqueezeNet", "NanoDet"],
    ModelComplexity.LOW: ["MobileNet-V2", "MobileNet-V3-Large", "YOLO-Nano"],
    ModelComplexity.MEDIUM: ["EfficientNet-B0", "ResNet-18", "YOLOv5s"],
    ModelComplexity.HIGH: ["ResNet-50", "EfficientNet-B3", "YOLOv5m"],
    ModelComplexity.VERY_HIGH: ["ViT-B/16", "ResNet-101", "YOLOv5l"],
}


@dataclass
class PerceptionBudgetEstimate:
    """Perception power budget estimate."""
    target_fps: float
    model_complexity: str
    model_examples: List[str]

    # Power estimates
    perception_power_w: float
    power_per_fps: float

    # With overhead
    total_system_power_w: float
    perception_ratio: float

    # Tier mapping
    recommended_tier: str
    fits_tier: bool

    # Platform suggestions
    suitable_platforms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirements": {
                "target_fps": self.target_fps,
                "model_complexity": self.model_complexity,
                "model_examples": self.model_examples,
            },
            "power_estimate": {
                "perception_power_w": self.perception_power_w,
                "power_per_fps": self.power_per_fps,
                "total_system_power_w": self.total_system_power_w,
                "perception_ratio": self.perception_ratio,
            },
            "tier_mapping": {
                "recommended_tier": self.recommended_tier,
                "fits_tier": self.fits_tier,
            },
            "suitable_platforms": self.suitable_platforms,
        }


@dataclass
class PerceptionBudgetAnalysis:
    """Complete perception budget analysis."""
    target_fps: float
    tier_name: Optional[str]
    tier_power_budget: Optional[float]

    estimates: List[PerceptionBudgetEstimate]

    # Recommendations
    recommended_complexity: Optional[str] = None
    max_feasible_fps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": {
                "target_fps": self.target_fps,
                "tier": self.tier_name,
                "tier_power_budget_w": self.tier_power_budget,
            },
            "estimates": [e.to_dict() for e in self.estimates],
            "recommendations": {
                "recommended_complexity": self.recommended_complexity,
                "max_feasible_fps": self.max_feasible_fps,
            },
        }


def get_recommended_tier(power_w: float) -> str:
    """Get recommended tier for a power level."""
    for tier in CAPABILITY_TIERS.values():
        if tier.power_range[0] <= power_w <= tier.power_range[1]:
            return tier.name
    if power_w < 0.1:
        return "wearable-ai"
    return "automotive-ai"


def find_suitable_platforms(power_w: float, margin: float = 0.2) -> List[str]:
    """Find platforms suitable for the power requirement."""
    suitable = []
    for name in list_platforms():
        platform = get_platform(name)
        if platform and platform.tdp_w >= power_w * (1 + margin):
            suitable.append(name)
    return sorted(suitable, key=lambda n: get_platform(n).tdp_w)[:5]


def estimate_perception_budget(
    target_fps: float,
    tier_name: Optional[str] = None,
) -> PerceptionBudgetAnalysis:
    """Estimate perception power budget for target FPS."""
    # Get tier constraints if specified
    tier_power = None
    if tier_name:
        tier = get_tier_by_name(tier_name)
        if tier:
            tier_power = tier.typical_power_w

    estimates = []
    recommended_complexity = None
    max_feasible_fps = None

    for complexity in ModelComplexity:
        power_per_fps = POWER_PER_FPS[complexity]
        perception_power = target_fps * power_per_fps

        # Estimate total system power (perception typically 30-50% of total)
        total_power = perception_power / 0.4  # Assuming 40% for perception

        # Check tier fit
        fits_tier = True
        if tier_power:
            fits_tier = total_power <= tier_power

        rec_tier = get_recommended_tier(total_power)
        platforms = find_suitable_platforms(total_power)

        estimate = PerceptionBudgetEstimate(
            target_fps=target_fps,
            model_complexity=complexity.value,
            model_examples=MODEL_EXAMPLES[complexity],
            perception_power_w=perception_power,
            power_per_fps=power_per_fps,
            total_system_power_w=total_power,
            perception_ratio=0.4,
            recommended_tier=rec_tier,
            fits_tier=fits_tier,
            suitable_platforms=platforms,
        )
        estimates.append(estimate)

        # Track recommended complexity (highest that fits)
        if fits_tier or tier_name is None:
            recommended_complexity = complexity.value

        # Calculate max feasible FPS for this complexity within tier
        if tier_power and fits_tier:
            perception_budget = tier_power * 0.4
            feasible_fps = perception_budget / power_per_fps
            if max_feasible_fps is None or feasible_fps > max_feasible_fps:
                max_feasible_fps = feasible_fps

    return PerceptionBudgetAnalysis(
        target_fps=target_fps,
        tier_name=tier_name,
        tier_power_budget=tier_power,
        estimates=estimates,
        recommended_complexity=recommended_complexity,
        max_feasible_fps=max_feasible_fps,
    )


def format_perception_budget(analysis: PerceptionBudgetAnalysis) -> str:
    """Format perception budget analysis as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  PERCEPTION POWER BUDGET ESTIMATE")
    lines.append("=" * 80)
    lines.append("")

    # Requirements
    lines.append("  Requirements:")
    lines.append(f"    Target Inference Rate: {analysis.target_fps:.0f} FPS")
    if analysis.tier_name:
        lines.append(f"    Tier Constraint:       {analysis.tier_name}")
        lines.append(f"    Tier Power Budget:     {analysis.tier_power_budget:.1f}W")
    lines.append("")

    # Estimates by complexity
    lines.append("  Power Estimates by Model Complexity:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Complexity':<12} {'Percep W':>10} {'Total W':>10} {'Tier':>18} {'Fits':>6}")
    lines.append("  " + "-" * 76)

    for e in analysis.estimates:
        fits_str = "YES" if e.fits_tier else "no"
        marker = " <--" if e.model_complexity == analysis.recommended_complexity else ""
        lines.append(
            f"    {e.model_complexity:<12} {e.perception_power_w:>9.2f}W {e.total_system_power_w:>9.1f}W "
            f"{e.recommended_tier:>18} {fits_str:>6}{marker}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Model examples
    lines.append("  Model Examples by Complexity:")
    for e in analysis.estimates:
        examples = ", ".join(e.model_examples[:3])
        lines.append(f"    {e.model_complexity:<12}: {examples}")
    lines.append("")

    # Recommendations
    lines.append("  Recommendations:")
    if analysis.recommended_complexity:
        lines.append(f"    Recommended Complexity: {analysis.recommended_complexity}")
    if analysis.max_feasible_fps and analysis.tier_name:
        lines.append(f"    Max FPS in {analysis.tier_name}: {analysis.max_feasible_fps:.0f} FPS")
    lines.append("")

    # Suitable platforms for recommended
    rec_estimate = next(
        (e for e in analysis.estimates if e.model_complexity == analysis.recommended_complexity),
        analysis.estimates[0]
    )
    if rec_estimate.suitable_platforms:
        lines.append("  Suitable Platforms:")
        for platform in rec_estimate.suitable_platforms[:5]:
            p = get_platform(platform)
            if p:
                lines.append(f"    - {platform:<30} (TDP: {p.tdp_w:.1f}W)")
    lines.append("")

    # Visual power breakdown
    lines.append("  Power Breakdown (recommended complexity):")
    bar_width = 50
    perception_len = int(rec_estimate.perception_ratio * bar_width)
    other_len = bar_width - perception_len
    bar = "P" * perception_len + "O" * other_len
    lines.append(f"    [{bar}]")
    lines.append(f"    P=Perception ({rec_estimate.perception_ratio*100:.0f}%), O=Other subsystems")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate perception power budget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate budget for 30 FPS
  ./cli/estimate_perception_budget.py --target-fps 30

  # Within tier constraint
  ./cli/estimate_perception_budget.py --tier micro-autonomy --target-fps 15

  # For specific complexity
  ./cli/estimate_perception_budget.py --target-fps 30 --model-complexity high

  # JSON output
  ./cli/estimate_perception_budget.py --target-fps 30 --format json
"""
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        required=True,
        help="Target inference rate in FPS"
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Constrain to tier power budget"
    )
    parser.add_argument(
        "--model-complexity",
        type=str,
        choices=["minimal", "low", "medium", "high", "very-high"],
        help="Filter to specific complexity level"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Run analysis
    analysis = estimate_perception_budget(
        target_fps=args.target_fps,
        tier_name=args.tier,
    )

    # Filter by complexity if specified
    if args.model_complexity:
        analysis.estimates = [
            e for e in analysis.estimates
            if e.model_complexity == args.model_complexity
        ]

    # Output
    if args.format == "json":
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print(format_perception_budget(analysis))

    return 0


if __name__ == "__main__":
    sys.exit(main())
