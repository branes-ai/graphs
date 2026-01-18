#!/usr/bin/env python3
"""
Compare Perception Pipelines CLI Tool

Compare different perception pipeline configurations and their
resource requirements. Shows tradeoffs between accuracy, latency, and power.

Usage:
    # Compare pipelines for a tier
    ./cli/compare_perception_pipelines.py --tier micro-autonomy

    # With FPS requirement
    ./cli/compare_perception_pipelines.py --tier industrial-edge --min-fps 30

    # Compare specific models
    ./cli/compare_perception_pipelines.py --models mobilenet-v2 efficientnet-b0 resnet-18

    # JSON output
    ./cli/compare_perception_pipelines.py --tier micro-autonomy --format json
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


class TaskType(Enum):
    """Perception task types."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    TRACKING = "tracking"


# Perception model database
PERCEPTION_MODELS = {
    "mobilenet-v3-small": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.67,  # Top-1 ImageNet
        "mflops": 56,
        "params_m": 2.5,
        "typical_fps_per_watt": 50,
        "input_resolution": (224, 224),
    },
    "mobilenet-v2": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.72,
        "mflops": 300,
        "params_m": 3.4,
        "typical_fps_per_watt": 20,
        "input_resolution": (224, 224),
    },
    "mobilenet-v3-large": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.75,
        "mflops": 219,
        "params_m": 5.4,
        "typical_fps_per_watt": 25,
        "input_resolution": (224, 224),
    },
    "efficientnet-b0": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.77,
        "mflops": 390,
        "params_m": 5.3,
        "typical_fps_per_watt": 15,
        "input_resolution": (224, 224),
    },
    "efficientnet-b1": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.79,
        "mflops": 700,
        "params_m": 7.8,
        "typical_fps_per_watt": 10,
        "input_resolution": (240, 240),
    },
    "resnet-18": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.70,
        "mflops": 1800,
        "params_m": 11.7,
        "typical_fps_per_watt": 8,
        "input_resolution": (224, 224),
    },
    "resnet-50": {
        "task": TaskType.CLASSIFICATION,
        "accuracy": 0.76,
        "mflops": 4100,
        "params_m": 25.6,
        "typical_fps_per_watt": 3,
        "input_resolution": (224, 224),
    },
    "yolo-nano": {
        "task": TaskType.DETECTION,
        "accuracy": 0.23,  # mAP
        "mflops": 500,
        "params_m": 1.9,
        "typical_fps_per_watt": 30,
        "input_resolution": (416, 416),
    },
    "yolov5s": {
        "task": TaskType.DETECTION,
        "accuracy": 0.37,
        "mflops": 7800,
        "params_m": 7.2,
        "typical_fps_per_watt": 5,
        "input_resolution": (640, 640),
    },
    "yolov5m": {
        "task": TaskType.DETECTION,
        "accuracy": 0.45,
        "mflops": 21200,
        "params_m": 21.2,
        "typical_fps_per_watt": 2,
        "input_resolution": (640, 640),
    },
    "ssd-mobilenet-v2": {
        "task": TaskType.DETECTION,
        "accuracy": 0.22,
        "mflops": 1500,
        "params_m": 4.3,
        "typical_fps_per_watt": 15,
        "input_resolution": (300, 300),
    },
    "deeplabv3-mobilenet": {
        "task": TaskType.SEGMENTATION,
        "accuracy": 0.72,  # mIoU
        "mflops": 8900,
        "params_m": 5.8,
        "typical_fps_per_watt": 3,
        "input_resolution": (513, 513),
    },
    "midas-small": {
        "task": TaskType.DEPTH,
        "accuracy": 0.85,  # Relative depth accuracy
        "mflops": 2000,
        "params_m": 3.5,
        "typical_fps_per_watt": 8,
        "input_resolution": (256, 256),
    },
}


@dataclass
class PipelineConfig:
    """A perception pipeline configuration."""
    model_name: str
    task: str
    accuracy: float
    mflops: int
    params_m: float
    input_resolution: tuple

    # Performance estimates
    fps_per_watt: float
    achievable_fps: float
    required_power_w: float

    # Efficiency metrics
    accuracy_per_watt: float
    accuracy_per_flop: float

    # Fit analysis
    fits_power_budget: bool
    meets_fps_requirement: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task,
            "specs": {
                "accuracy": self.accuracy,
                "mflops": self.mflops,
                "params_m": self.params_m,
                "input_resolution": list(self.input_resolution),
            },
            "performance": {
                "fps_per_watt": self.fps_per_watt,
                "achievable_fps": self.achievable_fps,
                "required_power_w": self.required_power_w,
            },
            "efficiency": {
                "accuracy_per_watt": self.accuracy_per_watt,
                "accuracy_per_flop": self.accuracy_per_flop,
            },
            "fit": {
                "fits_power_budget": self.fits_power_budget,
                "meets_fps_requirement": self.meets_fps_requirement,
            },
        }


@dataclass
class PipelineComparisonResult:
    """Result of pipeline comparison."""
    tier_name: Optional[str]
    perception_budget_w: float
    min_fps: Optional[float]
    task_filter: Optional[str]

    pipelines: List[PipelineConfig]

    # Recommendations
    best_accuracy: Optional[str] = None
    best_efficiency: Optional[str] = None
    best_fps: Optional[str] = None
    recommended: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": {
                "tier": self.tier_name,
                "perception_budget_w": self.perception_budget_w,
                "min_fps": self.min_fps,
                "task_filter": self.task_filter,
            },
            "pipelines": [p.to_dict() for p in self.pipelines],
            "recommendations": {
                "best_accuracy": self.best_accuracy,
                "best_efficiency": self.best_efficiency,
                "best_fps": self.best_fps,
                "recommended": self.recommended,
            },
        }


def analyze_pipeline(
    model_name: str,
    model_info: Dict[str, Any],
    perception_budget_w: float,
    min_fps: Optional[float] = None,
) -> PipelineConfig:
    """Analyze a single pipeline configuration."""
    fps_per_watt = model_info["typical_fps_per_watt"]

    # Achievable FPS within budget
    achievable_fps = fps_per_watt * perception_budget_w

    # Required power for min FPS
    if min_fps:
        required_power = min_fps / fps_per_watt if fps_per_watt > 0 else float('inf')
    else:
        required_power = perception_budget_w

    # Efficiency metrics
    accuracy_per_watt = model_info["accuracy"] * fps_per_watt
    accuracy_per_flop = model_info["accuracy"] / model_info["mflops"] * 1000  # per GFLOP

    # Fit analysis
    fits_budget = required_power <= perception_budget_w
    meets_fps = min_fps is None or achievable_fps >= min_fps

    return PipelineConfig(
        model_name=model_name,
        task=model_info["task"].value,
        accuracy=model_info["accuracy"],
        mflops=model_info["mflops"],
        params_m=model_info["params_m"],
        input_resolution=model_info["input_resolution"],
        fps_per_watt=fps_per_watt,
        achievable_fps=achievable_fps,
        required_power_w=required_power,
        accuracy_per_watt=accuracy_per_watt,
        accuracy_per_flop=accuracy_per_flop,
        fits_power_budget=fits_budget,
        meets_fps_requirement=meets_fps,
    )


def compare_perception_pipelines(
    tier_name: Optional[str] = None,
    perception_budget_w: Optional[float] = None,
    min_fps: Optional[float] = None,
    model_names: Optional[List[str]] = None,
    task_filter: Optional[str] = None,
) -> PipelineComparisonResult:
    """Compare perception pipeline configurations."""
    # Determine perception budget
    if perception_budget_w is None:
        if tier_name:
            tier = get_tier_by_name(tier_name)
            if tier:
                # Assume ~40% of tier power for perception
                perception_budget_w = tier.typical_power_w * 0.4
            else:
                perception_budget_w = 5.0
        else:
            perception_budget_w = 5.0

    # Filter models
    if model_names:
        models = {
            name: PERCEPTION_MODELS[name]
            for name in model_names
            if name in PERCEPTION_MODELS
        }
    else:
        models = PERCEPTION_MODELS

    if task_filter:
        task_enum = TaskType(task_filter)
        models = {
            name: info for name, info in models.items()
            if info["task"] == task_enum
        }

    # Analyze each pipeline
    pipelines = []
    for name, info in models.items():
        config = analyze_pipeline(name, info, perception_budget_w, min_fps)
        pipelines.append(config)

    # Sort by accuracy
    pipelines.sort(key=lambda x: x.accuracy, reverse=True)

    # Find recommendations
    fitting = [p for p in pipelines if p.fits_power_budget and p.meets_fps_requirement]

    best_accuracy = max(fitting, key=lambda x: x.accuracy).model_name if fitting else None
    best_efficiency = max(fitting, key=lambda x: x.accuracy_per_watt).model_name if fitting else None
    best_fps = max(fitting, key=lambda x: x.achievable_fps).model_name if fitting else None

    # Recommended: best accuracy among fitting
    recommended = best_accuracy

    return PipelineComparisonResult(
        tier_name=tier_name,
        perception_budget_w=perception_budget_w,
        min_fps=min_fps,
        task_filter=task_filter,
        pipelines=pipelines,
        best_accuracy=best_accuracy,
        best_efficiency=best_efficiency,
        best_fps=best_fps,
        recommended=recommended,
    )


def format_pipeline_comparison(result: PipelineComparisonResult) -> str:
    """Format pipeline comparison as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  PERCEPTION PIPELINE COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Constraints
    lines.append("  Constraints:")
    if result.tier_name:
        lines.append(f"    Tier:               {result.tier_name}")
    lines.append(f"    Perception Budget:  {result.perception_budget_w:.1f}W")
    if result.min_fps:
        lines.append(f"    Min FPS:            {result.min_fps:.0f}")
    if result.task_filter:
        lines.append(f"    Task Filter:        {result.task_filter}")
    lines.append("")

    # Pipeline comparison table
    lines.append("  Pipeline Comparison:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Model':<22} {'Task':<12} {'Acc':>6} {'MFLOP':>7} {'FPS/W':>7} {'FPS':>6} {'Fit':>5}")
    lines.append("  " + "-" * 76)

    for p in result.pipelines:
        fits = "YES" if (p.fits_power_budget and p.meets_fps_requirement) else "no"
        marker = ""
        if p.model_name == result.recommended:
            marker = " <--"
        lines.append(
            f"    {p.model_name:<22} {p.task:<12} {p.accuracy:>5.0%} {p.mflops:>7} "
            f"{p.fps_per_watt:>7.1f} {p.achievable_fps:>6.0f} {fits:>5}{marker}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Efficiency analysis
    lines.append("  Efficiency Analysis:")
    lines.append("  " + "-" * 60)
    lines.append(f"    {'Model':<22} {'Acc/W':>12} {'Acc/GFLOP':>12} {'Params':>10}")
    lines.append("  " + "-" * 60)

    for p in result.pipelines[:8]:
        lines.append(
            f"    {p.model_name:<22} {p.accuracy_per_watt:>12.2f} {p.accuracy_per_flop:>12.4f} "
            f"{p.params_m:>9.1f}M"
        )

    lines.append("  " + "-" * 60)
    lines.append("")

    # Visual accuracy vs FPS tradeoff
    lines.append("  Accuracy vs FPS Trade-off:")
    lines.append("")

    max_fps = max(p.achievable_fps for p in result.pipelines)
    bar_width = 30
    for p in result.pipelines[:8]:
        acc_bar_len = int(p.accuracy * bar_width)
        fps_bar_len = int((p.achievable_fps / max_fps) * bar_width) if max_fps > 0 else 0

        acc_bar = "#" * acc_bar_len + "." * (bar_width - acc_bar_len)
        fps_bar = "#" * fps_bar_len + "." * (bar_width - fps_bar_len)

        fits_marker = "*" if (p.fits_power_budget and p.meets_fps_requirement) else " "
        lines.append(f"    {p.model_name:<20}")
        lines.append(f"      {fits_marker} Acc: [{acc_bar}] {p.accuracy:.0%}")
        lines.append(f"        FPS: [{fps_bar}] {p.achievable_fps:.0f}")

    lines.append("")
    lines.append("    * = Meets constraints")
    lines.append("")

    # Recommendations
    if result.recommended:
        lines.append("  Recommendations:")
        lines.append(f"    Recommended:        {result.recommended}")
        lines.append(f"    Best Accuracy:      {result.best_accuracy}")
        lines.append(f"    Best Efficiency:    {result.best_efficiency}")
        lines.append(f"    Best FPS:           {result.best_fps}")
    else:
        lines.append("  WARNING: No pipelines meet constraints!")
        lines.append("  Consider:")
        lines.append("    - Increasing perception power budget")
        lines.append("    - Reducing FPS requirements")
        lines.append("    - Using lighter models")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare perception pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare pipelines for a tier
  ./cli/compare_perception_pipelines.py --tier micro-autonomy

  # With FPS requirement
  ./cli/compare_perception_pipelines.py --tier industrial-edge --min-fps 30

  # Filter by task
  ./cli/compare_perception_pipelines.py --tier micro-autonomy --task detection

  # Compare specific models
  ./cli/compare_perception_pipelines.py --models mobilenet-v2 efficientnet-b0 resnet-18

  # JSON output
  ./cli/compare_perception_pipelines.py --tier micro-autonomy --format json
"""
    )

    parser.add_argument(
        "--tier", "-t",
        type=str,
        choices=["wearable-ai", "micro-autonomy", "industrial-edge", "embodied-ai", "automotive-ai"],
        help="Capability tier (sets perception budget)"
    )
    parser.add_argument(
        "--budget", "-b",
        type=float,
        help="Perception power budget in watts"
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        help="Minimum required inference FPS"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "detection", "segmentation", "depth", "tracking"],
        help="Filter by task type"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        help="Specific models to compare"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list_models:
        print("Available perception models:")
        for name, info in sorted(PERCEPTION_MODELS.items()):
            print(f"  {name:<22} {info['task'].value:<12} {info['mflops']:>6} MFLOP  {info['accuracy']:.0%} acc")
        return 0

    # Run comparison
    result = compare_perception_pipelines(
        tier_name=args.tier,
        perception_budget_w=args.budget,
        min_fps=args.min_fps,
        model_names=args.models,
        task_filter=args.task,
    )

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_pipeline_comparison(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
