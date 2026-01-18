#!/usr/bin/env python3
"""
Discover Models for Budget CLI Tool

Find DNN models that can run within a power and/or latency budget.
Shows model recommendations ranked by efficiency.

Usage:
    # Find models for 5W perception budget at 30fps
    ./cli/discover_models_for_budget.py --power-budget 5 --target-fps 30 --task detection

    # Find models for specific platform
    ./cli/discover_models_for_budget.py --platform Jetson-Orin-Nano-8GB --power-budget 8 --task segmentation

    # Multi-task discovery
    ./cli/discover_models_for_budget.py --power-budget 15 --tasks detection segmentation

    # JSON output
    ./cli/discover_models_for_budget.py --power-budget 10 --task detection --format json
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
    get_mapper_info,
)
from graphs.mission.capability_tiers import (
    get_tier_for_power,
)


# Model database with power/performance characteristics
# In practice, this would come from benchmarks or a model registry
MODEL_DATABASE = {
    # Detection models
    "yolov8n": {
        "task": "detection",
        "family": "yolo",
        "params_m": 3.2,
        "flops_g": 8.7,
        "typical_power_w": 2.5,
        "typical_fps_at_10w": 45,
        "accuracy_map": 37.3,
        "description": "YOLOv8 Nano - ultralight detection",
    },
    "yolov8s": {
        "task": "detection",
        "family": "yolo",
        "params_m": 11.2,
        "flops_g": 28.6,
        "typical_power_w": 4.0,
        "typical_fps_at_10w": 28,
        "accuracy_map": 44.9,
        "description": "YOLOv8 Small - balanced detection",
    },
    "yolov8m": {
        "task": "detection",
        "family": "yolo",
        "params_m": 25.9,
        "flops_g": 78.9,
        "typical_power_w": 7.0,
        "typical_fps_at_10w": 15,
        "accuracy_map": 50.2,
        "description": "YOLOv8 Medium - high accuracy detection",
    },
    "yolov8l": {
        "task": "detection",
        "family": "yolo",
        "params_m": 43.7,
        "flops_g": 165.2,
        "typical_power_w": 12.0,
        "typical_fps_at_10w": 8,
        "accuracy_map": 52.9,
        "description": "YOLOv8 Large - maximum accuracy",
    },
    "rt-detr-s": {
        "task": "detection",
        "family": "detr",
        "params_m": 20.0,
        "flops_g": 60.0,
        "typical_power_w": 6.0,
        "typical_fps_at_10w": 18,
        "accuracy_map": 48.1,
        "description": "RT-DETR Small - transformer detection",
    },

    # Classification models
    "mobilenet_v2": {
        "task": "classification",
        "family": "mobilenet",
        "params_m": 3.5,
        "flops_g": 0.3,
        "typical_power_w": 1.0,
        "typical_fps_at_10w": 180,
        "accuracy_top1": 71.9,
        "description": "MobileNetV2 - efficient classification",
    },
    "efficientnet_b0": {
        "task": "classification",
        "family": "efficientnet",
        "params_m": 5.3,
        "flops_g": 0.4,
        "typical_power_w": 1.5,
        "typical_fps_at_10w": 120,
        "accuracy_top1": 77.1,
        "description": "EfficientNet-B0 - balanced efficiency",
    },
    "resnet18": {
        "task": "classification",
        "family": "resnet",
        "params_m": 11.7,
        "flops_g": 1.8,
        "typical_power_w": 2.5,
        "typical_fps_at_10w": 80,
        "accuracy_top1": 69.8,
        "description": "ResNet-18 - lightweight ResNet",
    },
    "resnet50": {
        "task": "classification",
        "family": "resnet",
        "params_m": 25.6,
        "flops_g": 4.1,
        "typical_power_w": 5.0,
        "typical_fps_at_10w": 40,
        "accuracy_top1": 76.1,
        "description": "ResNet-50 - standard ResNet",
    },

    # Segmentation models
    "fastsam_s": {
        "task": "segmentation",
        "family": "sam",
        "params_m": 11.8,
        "flops_g": 42.4,
        "typical_power_w": 5.0,
        "typical_fps_at_10w": 20,
        "accuracy_miou": 63.0,
        "description": "FastSAM-S - fast segment anything",
    },
    "mobilesam": {
        "task": "segmentation",
        "family": "sam",
        "params_m": 9.7,
        "flops_g": 39.4,
        "typical_power_w": 4.5,
        "typical_fps_at_10w": 22,
        "accuracy_miou": 62.0,
        "description": "MobileSAM - mobile segment anything",
    },
    "deeplabv3_mobilenet": {
        "task": "segmentation",
        "family": "deeplab",
        "params_m": 11.0,
        "flops_g": 25.0,
        "typical_power_w": 4.0,
        "typical_fps_at_10w": 25,
        "accuracy_miou": 68.0,
        "description": "DeepLabV3 MobileNet - efficient segmentation",
    },

    # Depth estimation
    "depth_anything_small": {
        "task": "depth",
        "family": "depth_anything",
        "params_m": 24.8,
        "flops_g": 52.0,
        "typical_power_w": 5.5,
        "typical_fps_at_10w": 18,
        "accuracy_rmse": 0.25,
        "description": "Depth Anything Small - monocular depth",
    },
    "midas_small": {
        "task": "depth",
        "family": "midas",
        "params_m": 21.0,
        "flops_g": 45.0,
        "typical_power_w": 5.0,
        "typical_fps_at_10w": 20,
        "accuracy_rmse": 0.28,
        "description": "MiDaS Small - depth estimation",
    },

    # Pose estimation
    "yolov8n_pose": {
        "task": "pose",
        "family": "yolo",
        "params_m": 3.3,
        "flops_g": 9.2,
        "typical_power_w": 2.8,
        "typical_fps_at_10w": 42,
        "accuracy_ap": 50.4,
        "description": "YOLOv8n Pose - ultralight pose",
    },
    "movenet_lightning": {
        "task": "pose",
        "family": "movenet",
        "params_m": 2.0,
        "flops_g": 2.5,
        "typical_power_w": 1.5,
        "typical_fps_at_10w": 90,
        "accuracy_ap": 45.0,
        "description": "MoveNet Lightning - fastest pose",
    },
}


@dataclass
class ModelRecommendation:
    """A model recommendation with fitness scores."""
    name: str
    task: str
    family: str
    description: str

    # Specs
    params_m: float
    flops_g: float
    typical_power_w: float

    # Performance estimates
    estimated_fps: float
    estimated_power_w: float

    # Fitness for budget
    fits_power_budget: bool
    fits_fps_target: bool
    power_margin_pct: float
    fps_margin_pct: float

    # Efficiency score
    efficiency_score: float  # FPS per watt
    overall_score: float

    # Accuracy (task-dependent)
    accuracy_metric: str
    accuracy_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "family": self.family,
            "description": self.description,
            "specs": {
                "params_m": self.params_m,
                "flops_g": self.flops_g,
                "typical_power_w": self.typical_power_w,
            },
            "estimates": {
                "fps": self.estimated_fps,
                "power_w": self.estimated_power_w,
            },
            "fitness": {
                "fits_power_budget": self.fits_power_budget,
                "fits_fps_target": self.fits_fps_target,
                "power_margin_pct": self.power_margin_pct,
                "fps_margin_pct": self.fps_margin_pct,
            },
            "scores": {
                "efficiency": self.efficiency_score,
                "overall": self.overall_score,
            },
            "accuracy": {
                "metric": self.accuracy_metric,
                "value": self.accuracy_value,
            },
        }


@dataclass
class ModelDiscoveryResult:
    """Result of model discovery."""
    power_budget_w: float
    target_fps: Optional[float]
    tasks: List[str]
    platform: Optional[str]

    recommendations: List[ModelRecommendation]
    models_evaluated: int
    models_fitting: int

    # Best picks
    best_efficiency: Optional[str] = None
    best_accuracy: Optional[str] = None
    best_balanced: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": {
                "power_budget_w": self.power_budget_w,
                "target_fps": self.target_fps,
                "tasks": self.tasks,
                "platform": self.platform,
            },
            "summary": {
                "models_evaluated": self.models_evaluated,
                "models_fitting": self.models_fitting,
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
            "best_picks": {
                "best_efficiency": self.best_efficiency,
                "best_accuracy": self.best_accuracy,
                "best_balanced": self.best_balanced,
            },
        }


def estimate_performance(
    model_info: Dict[str, Any],
    power_budget_w: float,
    platform_tdp_w: float = 10.0,
) -> Dict[str, float]:
    """Estimate model performance for a given power budget."""
    # Scale from reference (10W) to actual budget
    base_fps = model_info["typical_fps_at_10w"]
    base_power = model_info["typical_power_w"]

    # Power scales roughly with TDP available
    power_scale = power_budget_w / 10.0

    # FPS scales sublinearly with power (diminishing returns)
    fps_scale = power_scale ** 0.7

    estimated_fps = base_fps * fps_scale
    estimated_power = min(base_power * power_scale, power_budget_w)

    return {
        "fps": estimated_fps,
        "power_w": estimated_power,
    }


def discover_models_for_budget(
    power_budget_w: float,
    target_fps: Optional[float] = None,
    tasks: Optional[List[str]] = None,
    platform: Optional[str] = None,
) -> ModelDiscoveryResult:
    """
    Discover models that fit within power/fps budget.

    Args:
        power_budget_w: Maximum power budget in watts
        target_fps: Target frames per second (optional)
        tasks: Filter by task type (detection, classification, etc.)
        platform: Platform name for more accurate estimates

    Returns:
        ModelDiscoveryResult with ranked recommendations
    """
    # Get platform TDP for scaling
    platform_tdp = 10.0  # Default
    if platform:
        info = get_mapper_info(platform)
        if info:
            platform_tdp = info["default_tdp_w"]

    # Filter by task if specified
    if tasks:
        task_set = set(t.lower() for t in tasks)
        candidates = {
            name: info for name, info in MODEL_DATABASE.items()
            if info["task"] in task_set
        }
    else:
        candidates = MODEL_DATABASE
        tasks = list(set(m["task"] for m in MODEL_DATABASE.values()))

    recommendations = []
    for name, info in candidates.items():
        # Estimate performance
        perf = estimate_performance(info, power_budget_w, platform_tdp)

        # Check fitness
        fits_power = perf["power_w"] <= power_budget_w
        fits_fps = perf["fps"] >= target_fps if target_fps else True

        # Calculate margins
        power_margin = ((power_budget_w - perf["power_w"]) / power_budget_w) * 100 if power_budget_w > 0 else 0
        fps_margin = ((perf["fps"] - target_fps) / target_fps) * 100 if target_fps and target_fps > 0 else 0

        # Efficiency score
        efficiency = perf["fps"] / perf["power_w"] if perf["power_w"] > 0 else 0

        # Get accuracy metric
        accuracy_metric = "map" if "accuracy_map" in info else \
                         "top1" if "accuracy_top1" in info else \
                         "miou" if "accuracy_miou" in info else \
                         "rmse" if "accuracy_rmse" in info else \
                         "ap" if "accuracy_ap" in info else "unknown"
        accuracy_value = info.get(f"accuracy_{accuracy_metric}", 0)

        # Overall score (weighted combination)
        # Penalize if doesn't fit budget
        fit_penalty = 1.0 if (fits_power and fits_fps) else 0.5

        overall_score = (
            efficiency * 0.4 +
            (accuracy_value / 100 if accuracy_metric != "rmse" else (1 - accuracy_value) * 4) * 0.3 +
            power_margin * 0.15 +
            fps_margin * 0.15
        ) * fit_penalty

        rec = ModelRecommendation(
            name=name,
            task=info["task"],
            family=info["family"],
            description=info["description"],
            params_m=info["params_m"],
            flops_g=info["flops_g"],
            typical_power_w=info["typical_power_w"],
            estimated_fps=perf["fps"],
            estimated_power_w=perf["power_w"],
            fits_power_budget=fits_power,
            fits_fps_target=fits_fps,
            power_margin_pct=power_margin,
            fps_margin_pct=fps_margin,
            efficiency_score=efficiency,
            overall_score=overall_score,
            accuracy_metric=accuracy_metric,
            accuracy_value=accuracy_value,
        )
        recommendations.append(rec)

    # Sort by overall score
    recommendations.sort(key=lambda x: -x.overall_score)

    # Filter to those that fit
    fitting = [r for r in recommendations if r.fits_power_budget and r.fits_fps_target]

    # Determine best picks
    if fitting:
        best_efficiency = max(fitting, key=lambda x: x.efficiency_score).name
        best_accuracy = max(fitting, key=lambda x: x.accuracy_value if x.accuracy_metric != "rmse" else -x.accuracy_value).name
        best_balanced = fitting[0].name  # Already sorted by overall
    else:
        best_efficiency = None
        best_accuracy = None
        best_balanced = None

    return ModelDiscoveryResult(
        power_budget_w=power_budget_w,
        target_fps=target_fps,
        tasks=tasks,
        platform=platform,
        recommendations=recommendations[:10],  # Top 10
        models_evaluated=len(candidates),
        models_fitting=len(fitting),
        best_efficiency=best_efficiency,
        best_accuracy=best_accuracy,
        best_balanced=best_balanced,
    )


def format_discovery(result: ModelDiscoveryResult, verbose: bool = False) -> str:
    """Format discovery result as text."""
    lines = []

    lines.append("=" * 80)
    lines.append("  MODEL DISCOVERY FOR BUDGET")
    lines.append("=" * 80)
    lines.append("")

    # Constraints
    lines.append("  Constraints:")
    lines.append(f"    Power Budget:   {result.power_budget_w:.1f}W")
    if result.target_fps:
        lines.append(f"    Target FPS:     {result.target_fps:.0f}")
    lines.append(f"    Tasks:          {', '.join(result.tasks)}")
    if result.platform:
        lines.append(f"    Platform:       {result.platform}")
    lines.append("")

    # Summary
    lines.append("  Summary:")
    lines.append(f"    Models Evaluated: {result.models_evaluated}")
    lines.append(f"    Models Fitting:   {result.models_fitting}")
    lines.append("")

    if not result.recommendations:
        lines.append("  No models found matching criteria.")
        lines.append("")
        return "\n".join(lines)

    # Recommendations table
    lines.append("  Model Recommendations:")
    lines.append("  " + "-" * 76)
    lines.append(f"    {'Model':<20} {'Task':<12} {'Power':>7} {'FPS':>7} {'Eff':>7} {'Fit':>5}")
    lines.append("  " + "-" * 76)

    for r in result.recommendations:
        power_str = f"{r.estimated_power_w:.1f}W"
        fps_str = f"{r.estimated_fps:.0f}"
        eff_str = f"{r.efficiency_score:.1f}"
        fit_str = "YES" if (r.fits_power_budget and r.fits_fps_target) else "no"

        lines.append(
            f"    {r.name:<20} {r.task:<12} {power_str:>7} {fps_str:>7} {eff_str:>7} {fit_str:>5}"
        )

    lines.append("  " + "-" * 76)
    lines.append("")

    # Best picks
    if result.best_balanced:
        lines.append("  Best Picks (among fitting models):")
        lines.append(f"    Best Balanced:    {result.best_balanced}")
        lines.append(f"    Best Efficiency:  {result.best_efficiency}")
        lines.append(f"    Best Accuracy:    {result.best_accuracy}")
        lines.append("")

    # Detailed info (if verbose)
    if verbose:
        lines.append("  Detailed Model Info:")
        for r in result.recommendations[:5]:
            lines.append(f"    {r.name}:")
            lines.append(f"      {r.description}")
            lines.append(f"      Params: {r.params_m:.1f}M, FLOPs: {r.flops_g:.1f}G")
            lines.append(f"      {r.accuracy_metric.upper()}: {r.accuracy_value:.1f}")
            if r.fits_power_budget and r.fits_fps_target:
                lines.append(f"      Margins: Power +{r.power_margin_pct:.0f}%, FPS +{r.fps_margin_pct:.0f}%")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Discover models for power/fps budget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find models for 5W at 30fps
  ./cli/discover_models_for_budget.py --power-budget 5 --target-fps 30 --task detection

  # Find models for platform
  ./cli/discover_models_for_budget.py --platform Jetson-Orin-Nano-8GB --power-budget 8

  # Multi-task discovery
  ./cli/discover_models_for_budget.py --power-budget 15 --tasks detection segmentation

  # JSON output
  ./cli/discover_models_for_budget.py --power-budget 10 --task detection --format json
"""
    )

    parser.add_argument(
        "--power-budget", "-p",
        type=float,
        required=True,
        help="Maximum power budget in watts"
    )
    parser.add_argument(
        "--target-fps", "-f",
        type=float,
        help="Target frames per second"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        choices=["detection", "classification", "segmentation", "depth", "pose"],
        help="Single task filter"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["detection", "classification", "segmentation", "depth", "pose"],
        help="Multiple task filter"
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Platform for performance scaling"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed model information"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models"
    )

    args = parser.parse_args()

    # Handle list option
    if args.list_models:
        print("Available models:")
        for name, info in sorted(MODEL_DATABASE.items()):
            print(f"  {name:<22} {info['task']:<14} {info['description']}")
        return 0

    # Determine tasks
    tasks = None
    if args.task:
        tasks = [args.task]
    elif args.tasks:
        tasks = args.tasks

    # Run discovery
    result = discover_models_for_budget(
        power_budget_w=args.power_budget,
        target_fps=args.target_fps,
        tasks=tasks,
        platform=args.platform,
    )

    # Output
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_discovery(result, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
