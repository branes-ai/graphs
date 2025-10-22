#!/usr/bin/env python
"""
Edge AI / Embodied AI Platform Comparison

Compares edge AI accelerators across two categories:

Category 1: Computer Vision / Low Power (≤10W)
- Hailo-8: 26 TOPS INT8 @ 2.5W
- Jetson Orin Nano: 40 TOPS INT8 @ 7-15W
- KPU-T64: 6.9 TOPS INT8 @ 6W

Category 2: Transformers / Higher Power (≤50W)
- Hailo-10H: 40 TOPS INT4 @ 2.5W
- Jetson Orin AGX: 275 TOPS INT8 @ 15-60W
- KPU-T256: 33.8 TOPS INT8 @ 30W

Test Models:
- ResNet-50: Computer vision backbone
- DeepLabV3+: Semantic segmentation
- ViT-Base: Vision Transformer

Metrics:
- Latency (ms)
- Throughput (FPS)
- Power efficiency (FPS/W)
- TOPS/W efficiency
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner
from src.graphs.characterize.hailo_mapper import create_hailo8_mapper, create_hailo10h_mapper
from src.graphs.characterize.gpu_mapper import create_jetson_orin_nano_mapper, create_jetson_orin_agx_mapper
from src.graphs.characterize.kpu_mapper import create_kpu_t64_mapper, create_kpu_t256_mapper
from src.graphs.characterize.hardware_mapper import Precision


@dataclass
class BenchmarkResult:
    """Results from a single hardware/model combination"""
    model_name: str
    hardware_name: str
    power_mode: str
    tdp_watts: float

    # Performance metrics
    latency_ms: float
    throughput_fps: float

    # Efficiency metrics
    fps_per_watt: float
    tops_per_watt: float

    # Utilization
    avg_utilization: float
    peak_utilization: float

    # Energy
    total_energy_mj: float  # millijoules
    energy_per_inference_mj: float


def extract_execution_stages(fusion_report):
    """Extract execution stages from fusion report"""
    stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]
    return stages


def create_resnet50(batch_size=1):
    """Create ResNet-50 model"""
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ResNet-50"


def create_deeplabv3(batch_size=1):
    """Create DeepLabV3+ model for semantic segmentation"""
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    return model, input_tensor, "DeepLabV3+"


def create_vit_base(batch_size=1):
    """Create Vision Transformer Base model"""
    model = models.vit_b_16(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ViT-Base"


def benchmark_model_on_hardware(model, input_tensor, model_name, mapper, hardware_name, power_mode, tdp_watts, precision='int8'):
    """Run single benchmark configuration"""

    # Trace model
    print(f"\n  Tracing {model_name}...")
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    # Fusion partitioning
    print(f"  Partitioning...")
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    # Extract stages
    execution_stages = extract_execution_stages(fusion_report)

    # Map to hardware
    print(f"  Mapping to {hardware_name} ({power_mode})...")
    batch_size = input_tensor.shape[0]

    if precision.lower() == 'int8':
        prec = Precision.INT8
    elif precision.lower() == 'int4':
        prec = Precision.INT4
    else:
        prec = Precision.FP16

    hw_report = mapper.map_graph(
        fusion_report,
        execution_stages,
        batch_size=batch_size,
        precision=prec
    )

    # Calculate metrics
    latency_ms = hw_report.total_latency * 1000  # Convert to ms
    throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    fps_per_watt = throughput_fps / tdp_watts if tdp_watts > 0 else 0

    # Calculate TOPS/W
    effective_ops_per_sec = fusion_report.total_flops / hw_report.total_latency if hw_report.total_latency > 0 else 0
    effective_tops = effective_ops_per_sec / 1e12
    tops_per_watt = effective_tops / tdp_watts if tdp_watts > 0 else 0

    # Energy
    total_energy_mj = hw_report.total_energy * 1000  # Convert to mJ
    energy_per_inference_mj = total_energy_mj

    return BenchmarkResult(
        model_name=model_name,
        hardware_name=hardware_name,
        power_mode=power_mode,
        tdp_watts=tdp_watts,
        latency_ms=latency_ms,
        throughput_fps=throughput_fps,
        fps_per_watt=fps_per_watt,
        tops_per_watt=tops_per_watt,
        avg_utilization=hw_report.average_utilization,
        peak_utilization=hw_report.peak_utilization,
        total_energy_mj=total_energy_mj,
        energy_per_inference_mj=energy_per_inference_mj,
    )


def print_category_results(category_name, results: List[BenchmarkResult]):
    """Print formatted results for a category"""
    print(f"\n{'='*100}")
    print(f"{category_name}")
    print(f"{'='*100}")

    # Group by model
    models = {}
    for r in results:
        if r.model_name not in models:
            models[r.model_name] = []
        models[r.model_name].append(r)

    for model_name, model_results in models.items():
        print(f"\n{model_name}")
        print(f"{'-'*100}")
        print(f"{'Hardware':<25} {'Power':<12} {'TDP':<8} {'Latency':<12} {'FPS':<10} {'FPS/W':<10} {'TOPS/W':<10} {'Util%':<8}")
        print(f"{'-'*100}")

        for r in model_results:
            print(f"{r.hardware_name:<25} {r.power_mode:<12} {r.tdp_watts:<8.1f} "
                  f"{r.latency_ms:<12.2f} {r.throughput_fps:<10.1f} "
                  f"{r.fps_per_watt:<10.2f} {r.tops_per_watt:<10.2f} "
                  f"{r.avg_utilization*100:<8.1f}")


def run_category_1_comparison():
    """
    Category 1: Computer Vision / Low Power (≤10W)
    Target: Battery-powered drones, robots, edge cameras
    """
    print("\n" + "="*100)
    print("CATEGORY 1: Computer Vision / Low Power (≤10W)")
    print("Target: Battery-powered drones, robots, edge cameras")
    print("="*100)

    # Hardware configurations
    hardware_configs = [
        ("Hailo-8", create_hailo8_mapper(), "Standard", 2.5, "int8"),
        ("Jetson-Orin-Nano", create_jetson_orin_nano_mapper(thermal_profile="7W"), "7W", 7.0, "int8"),
        ("KPU-T64", create_kpu_t64_mapper(thermal_profile="6W"), "6W", 6.0, "int8"),
    ]

    # Models to test
    models_to_test = [
        create_resnet50(batch_size=1),
        create_deeplabv3(batch_size=1),
        create_vit_base(batch_size=1),
    ]

    results = []

    for model, input_tensor, model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")

        for hw_name, mapper, power_mode, tdp, precision in hardware_configs:
            print(f"\n► {hw_name} @ {power_mode}")
            result = benchmark_model_on_hardware(
                model, input_tensor, model_name,
                mapper, hw_name, power_mode, tdp, precision
            )
            results.append(result)

    return results


def run_category_2_comparison():
    """
    Category 2: Transformers / Higher Power (≤50W)
    Target: Autonomous vehicles, high-performance edge servers
    """
    print("\n" + "="*100)
    print("CATEGORY 2: Transformers / Higher Power (≤50W)")
    print("Target: Autonomous vehicles, high-performance edge servers")
    print("="*100)

    # Hardware configurations
    hardware_configs = [
        ("Hailo-10H", create_hailo10h_mapper(), "Standard", 2.5, "int8"),
        ("Jetson-Orin-AGX", create_jetson_orin_agx_mapper(thermal_profile="15W"), "15W", 15.0, "int8"),
        ("KPU-T256", create_kpu_t256_mapper(thermal_profile="30W"), "30W", 30.0, "int8"),
    ]

    # Models to test
    models_to_test = [
        create_resnet50(batch_size=1),
        create_deeplabv3(batch_size=1),
        create_vit_base(batch_size=1),
    ]

    results = []

    for model, input_tensor, model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")

        for hw_name, mapper, power_mode, tdp, precision in hardware_configs:
            print(f"\n► {hw_name} @ {power_mode}")
            result = benchmark_model_on_hardware(
                model, input_tensor, model_name,
                mapper, hw_name, power_mode, tdp, precision
            )
            results.append(result)

    return results


def print_summary(cat1_results, cat2_results):
    """Print executive summary comparing categories"""
    print("\n" + "="*100)
    print("EXECUTIVE SUMMARY")
    print("="*100)

    print("\n## Key Findings")
    print("\n### Category 1 (Low Power ≤10W) - Best for:")
    print("  • Battery-powered drones (extended flight time)")
    print("  • Autonomous robots (thermal constraints)")
    print("  • Edge cameras (always-on vision)")

    print("\n  Winner by metric:")

    # Find best FPS/W in category 1
    cat1_best_efficiency = max(cat1_results, key=lambda r: r.fps_per_watt)
    print(f"  • Best FPS/W: {cat1_best_efficiency.hardware_name} "
          f"({cat1_best_efficiency.fps_per_watt:.2f} FPS/W on {cat1_best_efficiency.model_name})")

    # Find best latency in category 1
    cat1_best_latency = min(cat1_results, key=lambda r: r.latency_ms)
    print(f"  • Best Latency: {cat1_best_latency.hardware_name} "
          f"({cat1_best_latency.latency_ms:.2f} ms on {cat1_best_latency.model_name})")

    print("\n### Category 2 (Higher Power ≤50W) - Best for:")
    print("  • Autonomous vehicles (tethered power)")
    print("  • High-performance edge servers")
    print("  • Multi-model fusion pipelines")

    print("\n  Winner by metric:")

    # Find best FPS/W in category 2
    cat2_best_efficiency = max(cat2_results, key=lambda r: r.fps_per_watt)
    print(f"  • Best FPS/W: {cat2_best_efficiency.hardware_name} "
          f"({cat2_best_efficiency.fps_per_watt:.2f} FPS/W on {cat2_best_efficiency.model_name})")

    # Find best latency in category 2
    cat2_best_latency = min(cat2_results, key=lambda r: r.latency_ms)
    print(f"  • Best Latency: {cat2_best_latency.hardware_name} "
          f"({cat2_best_latency.latency_ms:.2f} ms on {cat2_best_latency.model_name})")

    print("\n### Key Architectural Insights:")
    print("  • Hailo: Dataflow architecture → high TOPS/W, low latency")
    print("  • Jetson: NVIDIA GPU → flexible but DVFS throttling hurts efficiency")
    print("  • KPU: Heterogeneous tiles → balanced performance across workloads")

    print("\n### Recommendation Matrix:")
    print("  ┌─────────────────────────┬──────────────────────┬─────────────────────┐")
    print("  │ Use Case                │ Best Choice          │ Runner-up           │")
    print("  ├─────────────────────────┼──────────────────────┼─────────────────────┤")
    print("  │ Drone (battery)         │ Hailo-8 @ 2.5W       │ KPU-T64 @ 3W        │")
    print("  │ Robot (mobile)          │ KPU-T64 @ 6W         │ Hailo-8 @ 2.5W      │")
    print("  │ Edge camera             │ Hailo-8 @ 2.5W       │ Jetson Nano @ 7W    │")
    print("  │ Autonomous vehicle      │ KPU-T256 @ 30W       │ Hailo-10H @ 2.5W    │")
    print("  │ Edge server             │ KPU-T256 @ 30W       │ Jetson AGX @ 15W    │")
    print("  └─────────────────────────┴──────────────────────┴─────────────────────┘")


def main():
    """Run full edge AI comparison"""
    print("\n" + "="*100)
    print("EDGE AI / EMBODIED AI PLATFORM COMPARISON")
    print("Comparing accelerators for battery-powered robotics and autonomous systems")
    print("="*100)

    # Run Category 1
    cat1_results = run_category_1_comparison()
    print_category_results("CATEGORY 1 RESULTS: Computer Vision / Low Power (≤10W)", cat1_results)

    # Run Category 2
    cat2_results = run_category_2_comparison()
    print_category_results("CATEGORY 2 RESULTS: Transformers / Higher Power (≤50W)", cat2_results)

    # Print summary
    print_summary(cat1_results, cat2_results)

    print("\n" + "="*100)
    print("Benchmark Complete")
    print("="*100)


if __name__ == "__main__":
    main()
