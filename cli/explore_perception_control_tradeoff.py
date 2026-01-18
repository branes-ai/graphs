#!/usr/bin/env python3
"""
Explore Perception-Control Tradeoff CLI Tool

Analyze the tradeoff between perception and control subsystems
within a power budget. Shows how allocating more power to perception
affects control capabilities and vice versa.

Usage:
    # Explore tradeoffs for a power budget
    ./cli/explore_perception_control_tradeoff.py --budget 20

    # For specific tier
    ./cli/explore_perception_control_tradeoff.py --tier micro-autonomy

    # With specific perception requirement
    ./cli/explore_perception_control_tradeoff.py --budget 30 --min-inference-fps 30

    # JSON output
    ./cli/explore_perception_control_tradeoff.py --budget 20 --format json
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
    get_tier_for_power,
    CAPABILITY_TIERS,
)
from graphs.mission.power_allocation import (
    PowerAllocation,
    SubsystemType,
    get_typical_allocation,
    ALLOCATION_PERCEPTION_HEAVY,
    ALLOCATION_BALANCED,
    ALLOCATION_CONTROL_HEAVY,
)


@dataclass
class TradeoffPoint:
    """A single point on the perception-control tradeoff curve."""
    perception_ratio: float
    control_ratio: float
    movement_ratio: float
    overhead_ratio: float

    perception_power_w: float
    control_power_w: float
    movement_power_w: float

    # Estimated capabilities
    estimated_inference_fps: float
    estimated_control_hz: float
    movement_capability: str

    # Quality metrics
    perception_quality: str  # minimal, basic, good, excellent
    control_quality: str  # minimal, basic, good, excellent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratios": {
                "perception": self.perception_ratio,
                "control": self.control_ratio,
                "movement": self.movement_ratio,
                "overhead": self.overhead_ratio,
            },
            "power_watts": {
                "perception": self.perception_power_w,
                "control": self.control_power_w,
                "movement": self.movement_power_w,
            },
            "capabilities": {
                "inference_fps": self.estimated_inference_fps,
                "control_hz": self.estimated_control_hz,
                "movement": self.movement_capability,
            },
            "quality": {
                "perception": self.perception_quality,
                "control": self.control_quality,
            },
        }


@dataclass
class TradeoffAnalysis:
    """Complete tradeoff analysis."""
    total_budget_w: float
    tier_name: Optional[str]
    min_inference_fps: Optional[float]
    min_control_hz: Optional[float]

    tradeoff_points: List[TradeoffPoint]

    # Recommendations
    best_balanced: Optional[int] = None  # Index into tradeoff_points
    best_perception: Optional[int] = None
    best_control: Optional[int] = None
    meets_requirements: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": {
                "total_budget_w": self.total_budget_w,
                "tier": self.tier_name,
                "min_inference_fps": self.min_inference_fps,
                "min_control_hz": self.min_control_hz,
            },
            "tradeoff_points": [p.to_dict() for p in self.tradeoff_points],
            "recommendations": {
                "best_balanced_idx": self.best_balanced,
                "best_perception_idx": self.best_perception,
                "best_control_idx": self.best_control,
                "meeting_requirements": self.meets_requirements,
            },
        }


def estimate_inference_fps(perception_power_w: float) -> float:
    """Estimate inference FPS based on perception power."""
    # Rough model: 1W ~ 10-20 FPS for typical edge inference
    return perception_power_w * 15


def estimate_control_hz(control_power_w: float) -> float:
    """Estimate control loop frequency based on control power."""
    # Rough model: 1W ~ 100-500 Hz control loop
    return control_power_w * 200


def get_movement_capability(movement_power_w: float) -> str:
    """Categorize movement capability based on power."""
    if movement_power_w < 2:
        return "stationary"
    elif movement_power_w < 5:
        return "minimal"
    elif movement_power_w < 15:
        return "light"
    elif movement_power_w < 30:
        return "standard"
    elif movement_power_w < 60:
        return "dynamic"
    else:
        return "high-performance"


def get_perception_quality(fps: float) -> str:
    """Categorize perception quality based on FPS."""
    if fps < 5:
        return "minimal"
    elif fps < 15:
        return "basic"
    elif fps < 30:
        return "good"
    else:
        return "excellent"


def get_control_quality(hz: float) -> str:
    """Categorize control quality based on frequency."""
    if hz < 100:
        return "minimal"
    elif hz < 500:
        return "basic"
    elif hz < 1000:
        return "good"
    else:
        return "excellent"


def generate_tradeoff_points(
    total_budget_w: float,
    movement_ratio: float = 0.3,
    overhead_ratio: float = 0.1,
    num_points: int = 11,
) -> List[TradeoffPoint]:
    """Generate tradeoff points by varying perception/control split."""
    points = []

    # Available power after movement and overhead
    available_ratio = 1.0 - movement_ratio - overhead_ratio
    movement_power = total_budget_w * movement_ratio
    overhead_power = total_budget_w * overhead_ratio

    for i in range(num_points):
        # Vary perception ratio from 0.1 to 0.9 of available
        perception_frac = 0.1 + (i / (num_points - 1)) * 0.8
        control_frac = 1.0 - perception_frac

        perception_ratio = perception_frac * available_ratio
        control_ratio = control_frac * available_ratio

        perception_power = total_budget_w * perception_ratio
        control_power = total_budget_w * control_ratio

        inference_fps = estimate_inference_fps(perception_power)
        control_hz = estimate_control_hz(control_power)
        movement_cap = get_movement_capability(movement_power)

        point = TradeoffPoint(
            perception_ratio=perception_ratio,
            control_ratio=control_ratio,
            movement_ratio=movement_ratio,
            overhead_ratio=overhead_ratio,
            perception_power_w=perception_power,
            control_power_w=control_power,
            movement_power_w=movement_power,
            estimated_inference_fps=inference_fps,
            estimated_control_hz=control_hz,
            movement_capability=movement_cap,
            perception_quality=get_perception_quality(inference_fps),
            control_quality=get_control_quality(control_hz),
        )
        points.append(point)

    return points


def analyze_tradeoffs(
    total_budget_w: float,
    tier_name: Optional[str] = None,
    min_inference_fps: Optional[float] = None,
    min_control_hz: Optional[float] = None,
    movement_ratio: float = 0.3,
) -> TradeoffAnalysis:
    """Analyze perception-control tradeoffs."""
    # Get tier info if provided
    if tier_name:
        tier = get_tier_by_name(tier_name)
        if tier:
            total_budget_w = tier.typical_power_w

    # Generate tradeoff points
    points = generate_tradeoff_points(
        total_budget_w,
        movement_ratio=movement_ratio,
        num_points=11,
    )

    # Find recommendations
    best_balanced = len(points) // 2  # Middle point

    best_perception = max(range(len(points)), key=lambda i: points[i].perception_power_w)
    best_control = max(range(len(points)), key=lambda i: points[i].control_power_w)

    # Find points meeting requirements
    meets_requirements = []
    for i, p in enumerate(points):
        meets_fps = min_inference_fps is None or p.estimated_inference_fps >= min_inference_fps
        meets_hz = min_control_hz is None or p.estimated_control_hz >= min_control_hz
        if meets_fps and meets_hz:
            meets_requirements.append(i)

    return TradeoffAnalysis(
        total_budget_w=total_budget_w,
        tier_name=tier_name,
        min_inference_fps=min_inference_fps,
        min_control_hz=min_control_hz,
        tradeoff_points=points,
        best_balanced=best_balanced,
        best_perception=best_perception,
        best_control=best_control,
        meets_requirements=meets_requirements,
    )


def format_tradeoff_analysis(analysis: TradeoffAnalysis) -> str:
    """Format tradeoff analysis as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  PERCEPTION-CONTROL TRADEOFF ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Configuration
    lines.append("  Configuration:")
    lines.append(f"    Total Budget:     {analysis.total_budget_w:.1f}W")
    if analysis.tier_name:
        lines.append(f"    Tier:             {analysis.tier_name}")
    if analysis.min_inference_fps:
        lines.append(f"    Min Inference:    {analysis.min_inference_fps:.0f} FPS")
    if analysis.min_control_hz:
        lines.append(f"    Min Control:      {analysis.min_control_hz:.0f} Hz")
    lines.append("")

    # Tradeoff curve
    lines.append("  Tradeoff Curve:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Percep%':>8} {'Ctrl%':>8} {'Percep W':>10} {'Ctrl W':>10} {'FPS':>8} {'Hz':>8} {'Meet':>6}")
    lines.append("  " + "-" * 76)

    for i, p in enumerate(analysis.tradeoff_points):
        meets = "YES" if i in analysis.meets_requirements else "no"
        marker = ""
        if i == analysis.best_balanced:
            marker = " <-- balanced"
        elif i == analysis.best_perception:
            marker = " <-- best percep"
        elif i == analysis.best_control:
            marker = " <-- best ctrl"

        lines.append(
            f"    {p.perception_ratio*100:>7.0f}% {p.control_ratio*100:>7.0f}% "
            f"{p.perception_power_w:>9.1f}W {p.control_power_w:>9.1f}W "
            f"{p.estimated_inference_fps:>7.0f} {p.estimated_control_hz:>7.0f} {meets:>6}{marker}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Visual tradeoff
    lines.append("  Visual Tradeoff (P=Perception, C=Control):")
    lines.append("")
    bar_width = 50
    for i, p in enumerate(analysis.tradeoff_points):
        p_len = int(p.perception_ratio / (p.perception_ratio + p.control_ratio) * bar_width)
        c_len = bar_width - p_len
        bar = "P" * p_len + "C" * c_len
        marker = "*" if i in analysis.meets_requirements else " "
        lines.append(f"    {marker} [{bar}]")
    lines.append("")

    # Quality mapping
    lines.append("  Quality Levels at Each Point:")
    lines.append("  " + "-" * 50)
    lines.append(f"    {'Point':>6} {'Perception':>15} {'Control':>15}")
    lines.append("  " + "-" * 50)
    for i, p in enumerate(analysis.tradeoff_points):
        lines.append(f"    {i:>6} {p.perception_quality:>15} {p.control_quality:>15}")
    lines.append("  " + "-" * 50)
    lines.append("")

    # Recommendations
    if analysis.meets_requirements:
        lines.append("  Recommendations:")
        lines.append(f"    Points meeting requirements: {len(analysis.meets_requirements)}")
        if analysis.best_balanced in analysis.meets_requirements:
            bp = analysis.tradeoff_points[analysis.best_balanced]
            lines.append(f"    Recommended (balanced): Point {analysis.best_balanced}")
            lines.append(f"      Perception: {bp.perception_power_w:.1f}W ({bp.estimated_inference_fps:.0f} FPS)")
            lines.append(f"      Control: {bp.control_power_w:.1f}W ({bp.estimated_control_hz:.0f} Hz)")
    else:
        lines.append("  WARNING: No points meet the specified requirements!")
        lines.append("  Consider:")
        lines.append("    - Increasing power budget")
        lines.append("    - Relaxing FPS or Hz requirements")
        lines.append("    - Reducing movement power allocation")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Explore perception-control power tradeoffs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore tradeoffs for 20W budget
  ./cli/explore_perception_control_tradeoff.py --budget 20

  # For specific tier
  ./cli/explore_perception_control_tradeoff.py --tier micro-autonomy

  # With minimum FPS requirement
  ./cli/explore_perception_control_tradeoff.py --budget 30 --min-inference-fps 30

  # With minimum control frequency
  ./cli/explore_perception_control_tradeoff.py --budget 30 --min-control-hz 500

  # JSON output
  ./cli/explore_perception_control_tradeoff.py --budget 20 --format json
"""
    )

    parser.add_argument(
        "--budget", "-b",
        type=float,
        help="Total power budget in watts"
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Use typical power for this tier"
    )
    parser.add_argument(
        "--min-inference-fps",
        type=float,
        help="Minimum required inference FPS"
    )
    parser.add_argument(
        "--min-control-hz",
        type=float,
        help="Minimum required control frequency Hz"
    )
    parser.add_argument(
        "--movement-ratio",
        type=float,
        default=0.3,
        help="Fraction of budget for movement (default: 0.3)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Require either budget or tier
    if not args.budget and not args.tier:
        parser.error("Either --budget or --tier is required")

    total_budget = args.budget or 15.0  # Default if tier is used

    # Run analysis
    analysis = analyze_tradeoffs(
        total_budget_w=total_budget,
        tier_name=args.tier,
        min_inference_fps=args.min_inference_fps,
        min_control_hz=args.min_control_hz,
        movement_ratio=args.movement_ratio,
    )

    # Output
    if args.format == "json":
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print(format_tradeoff_analysis(analysis))

    return 0


if __name__ == "__main__":
    sys.exit(main())
