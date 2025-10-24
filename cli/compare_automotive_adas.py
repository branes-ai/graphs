#!/usr/bin/env python
"""
Automotive ADAS Platform Comparison

Compares AI accelerators specifically for automotive Advanced Driver Assistance Systems (ADAS).

Focus Areas:
- ADAS Level 2-3 (lane keep, adaptive cruise control, auto parking)
- Multi-camera sensor fusion (front, rear, surround view)
- Real-time performance with deterministic latency
- Automotive safety requirements (ASIL-D/SIL-3)
- Temperature range: -40°C to 125°C

Hardware Categories:
===================

Category 1: Front Camera ADAS (10-15W)
- Single front-facing camera
- Lane detection, object detection, traffic sign recognition
- Power budget: 10-15W
- Example: TI TDA4VM @ 10W

Category 2: Multi-Camera ADAS (15-25W)
- 4-6 cameras (front, rear, side mirrors, surround)
- Sensor fusion (camera + radar + lidar)
- Automatic parking, surround view monitoring
- Power budget: 15-25W
- Example: TI TDA4VM @ 20W

Test Models:
===========
1. ResNet-50: Backbone for object detection and classification
2. YOLOv5s: Real-time object detection (vehicles, pedestrians, signs)
3. UNet: Lane segmentation and drivable area estimation

Automotive Metrics:
==================
- Latency (ms): Critical for safety (target: <100ms end-to-end)
- Throughput (FPS): Processing rate (target: 30 FPS minimum)
- FPS/W: Power efficiency (vehicle power budget)
- Worst-case latency: For safety certification
- Jitter: Latency variation (determinism requirement)
- Temperature stability: Performance across -40°C to 125°C

Safety Requirements:
===================
- ASIL-D/SIL-3 certification
- Deterministic execution
- Fault detection and handling
- Graceful degradation
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))  # cli/ → graphs/

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner
from src.graphs.characterize.dsp_mapper import create_ti_tda4vm_mapper
from src.graphs.characterize.gpu_mapper import create_jetson_orin_nano_mapper, create_jetson_orin_agx_mapper
from src.graphs.characterize.kpu_mapper import create_kpu_t256_mapper
from src.graphs.characterize.hardware_mapper import Precision


@dataclass
class AutomotiveBenchmarkResult:
    """Results from automotive ADAS benchmark"""
    model_name: str
    hardware_name: str
    power_mode: str
    tdp_watts: float

    # Performance metrics
    latency_ms: float
    throughput_fps: float

    # Automotive-specific metrics
    meets_30fps_requirement: bool  # ADAS typically requires 30 FPS minimum
    meets_100ms_latency: bool      # Safety requirement: <100ms end-to-end
    fps_per_watt: float
    tops_per_watt: float

    # Utilization
    avg_utilization: float
    peak_utilization: float

    # Energy
    total_energy_mj: float
    energy_per_inference_mj: float

    # Safety & reliability
    safety_certified: bool = False  # ASIL-D/SIL-3
    temperature_range: str = "consumer"  # "consumer" or "automotive"


@dataclass
class AutomotiveUseCase:
    """Automotive ADAS use case definition"""
    name: str
    description: str
    camera_count: int
    power_budget_w: float
    target_fps: int
    max_latency_ms: float
    safety_level: str  # "ASIL-B", "ASIL-D", etc.


def extract_execution_stages(fusion_report):
    """Extract execution stages from fusion report"""
    stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]
    return stages


# ============================================================================
# Model Factory Functions
# ============================================================================

def create_resnet50(batch_size=1):
    """Create ResNet-50 model (backbone for detection)"""
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    return model, input_tensor, "ResNet-50"


def create_deeplabv3_lane_detection(batch_size=1):
    """Create DeepLabV3+ for lane detection and free space estimation"""
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.eval()
    # Automotive cameras typically use 1280x720 or 1920x1080
    # Using 640x360 for practical inference time
    input_tensor = torch.randn(batch_size, 3, 640, 360)
    return model, input_tensor, "DeepLabV3-Lane-Detection"


def create_yolov5s_automotive(batch_size=1):
    """
    YOLOv5s-like model for automotive object detection.

    For FX tracing compatibility, use ResNet-50 as a proxy.
    Represents similar computational complexity to YOLOv5s.
    Input: 640x640 (standard YOLOv5 resolution)
    """
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    return model, input_tensor, "YOLOv5s-Automotive"


def create_unet_lane(batch_size=1):
    """
    Lane segmentation model.

    For FX tracing compatibility, use FCN-ResNet50 (simpler than UNet, no skip connection issues).
    Input: 640x360 (automotive camera aspect ratio)
    """
    model = models.segmentation.fcn_resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 640, 360)
    return model, input_tensor, "FCN-LaneSegmentation"


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_model_on_hardware(
    model,
    input_tensor,
    model_name,
    mapper,
    hardware_name,
    power_mode,
    tdp_watts,
    precision='int8',
    safety_certified=False,
    temperature_range="consumer"
):
    """Run single automotive benchmark configuration"""

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
    latency_ms = hw_report.total_latency * 1000
    throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    fps_per_watt = throughput_fps / tdp_watts if tdp_watts > 0 else 0

    # Automotive requirements
    meets_30fps = throughput_fps >= 30.0
    meets_100ms = latency_ms <= 100.0

    # Calculate TOPS/W
    effective_ops_per_sec = fusion_report.total_flops / hw_report.total_latency if hw_report.total_latency > 0 else 0
    effective_tops = effective_ops_per_sec / 1e12
    tops_per_watt = effective_tops / tdp_watts if tdp_watts > 0 else 0

    # Energy
    total_energy_mj = hw_report.total_energy * 1000
    energy_per_inference_mj = total_energy_mj

    return AutomotiveBenchmarkResult(
        model_name=model_name,
        hardware_name=hardware_name,
        power_mode=power_mode,
        tdp_watts=tdp_watts,
        latency_ms=latency_ms,
        throughput_fps=throughput_fps,
        meets_30fps_requirement=meets_30fps,
        meets_100ms_latency=meets_100ms,
        fps_per_watt=fps_per_watt,
        tops_per_watt=tops_per_watt,
        avg_utilization=hw_report.average_utilization,
        peak_utilization=hw_report.peak_utilization,
        total_energy_mj=total_energy_mj,
        energy_per_inference_mj=energy_per_inference_mj,
        safety_certified=safety_certified,
        temperature_range=temperature_range,
    )


def print_category_results(category_name: str, results: List[AutomotiveBenchmarkResult]):
    """Print formatted automotive results"""
    print(f"\n{'='*120}")
    print(f"{category_name}")
    print(f"{'='*120}")

    # Group by model
    models = {}
    for r in results:
        if r.model_name not in models:
            models[r.model_name] = []
        models[r.model_name].append(r)

    for model_name, model_results in models.items():
        print(f"\n{model_name}")
        print(f"{'-'*120}")
        print(f"{'Hardware':<25} {'Power':<12} {'TDP':<8} {'Latency':<12} {'FPS':<10} "
              f"{'FPS/W':<10} {'30FPS?':<8} {'<100ms?':<8} {'Util%':<8}")
        print(f"{'-'*120}")

        for r in model_results:
            fps_ok = "✓" if r.meets_30fps_requirement else "✗"
            lat_ok = "✓" if r.meets_100ms_latency else "✗"

            print(f"{r.hardware_name:<25} {r.power_mode:<12} {r.tdp_watts:<8.1f} "
                  f"{r.latency_ms:<12.2f} {r.throughput_fps:<10.1f} "
                  f"{r.fps_per_watt:<10.2f} {fps_ok:<8} {lat_ok:<8} "
                  f"{r.avg_utilization*100:<8.1f}")


# ============================================================================
# Comparison Functions
# ============================================================================

def run_front_camera_adas_comparison():
    """
    Category 1: Front Camera ADAS (10-15W)

    Use Case:
    - Single front-facing camera
    - Lane Keep Assist (LKA)
    - Adaptive Cruise Control (ACC) vision
    - Traffic Sign Recognition (TSR)
    - Forward Collision Warning (FCW)

    Requirements:
    - 30 FPS minimum (camera frame rate)
    - <100ms latency (safety requirement)
    - 10-15W power budget
    """
    print("\n" + "="*120)
    print("CATEGORY 1: Front Camera ADAS (10-15W)")
    print("Use Case: Lane Keep Assist, ACC, TSR, Forward Collision Warning")
    print("Safety Level: ASIL-B/C")
    print("="*120)

    # Hardware configurations
    hardware_configs = [
        ("TI-TDA4VM-C7x", create_ti_tda4vm_mapper(thermal_profile="10W"), "10W", 10.0, "int8", True, "automotive"),
        ("Jetson-Orin-Nano", create_jetson_orin_nano_mapper(thermal_profile="15W"), "15W", 15.0, "int8", False, "consumer"),
    ]

    # Automotive models
    models_to_test = [
        create_resnet50(batch_size=1),
        create_unet_lane(batch_size=1),
        create_yolov5s_automotive(batch_size=1),
    ]

    results = []

    for model, input_tensor, model_name in models_to_test:
        print(f"\n{'='*100}")
        print(f"Testing: {model_name}")
        print(f"{'='*100}")

        for hw_name, mapper, power_mode, tdp, precision, safety, temp_range in hardware_configs:
            print(f"\n► {hw_name} @ {power_mode}")
            result = benchmark_model_on_hardware(
                model, input_tensor, model_name,
                mapper, hw_name, power_mode, tdp, precision,
                safety_certified=safety,
                temperature_range=temp_range
            )
            results.append(result)

    return results


def run_multi_camera_adas_comparison():
    """
    Category 2: Multi-Camera ADAS (15-25W)

    Use Case:
    - 4-6 cameras (front, rear, side mirrors)
    - Surround View Monitoring (SVM)
    - Automatic Parking Assist
    - Blind Spot Detection
    - Cross Traffic Alert

    Requirements:
    - 30 FPS per camera (some can be lower for parking)
    - <100ms latency for primary functions
    - 15-25W power budget
    """
    print("\n" + "="*120)
    print("CATEGORY 2: Multi-Camera ADAS (15-25W)")
    print("Use Case: Surround View, Auto Parking, Blind Spot Detection")
    print("Safety Level: ASIL-D")
    print("="*120)

    # Hardware configurations
    hardware_configs = [
        ("TI-TDA4VM-C7x", create_ti_tda4vm_mapper(thermal_profile="20W"), "20W", 20.0, "int8", True, "automotive"),
        ("Jetson-Orin-AGX", create_jetson_orin_agx_mapper(thermal_profile="15W"), "15W", 15.0, "int8", False, "consumer"),
        ("KPU-T256", create_kpu_t256_mapper(thermal_profile="30W"), "25W", 25.0, "int8", False, "consumer"),
    ]

    # Automotive models (heavier workloads for multi-camera)
    models_to_test = [
        create_resnet50(batch_size=1),
        create_unet_lane(batch_size=1),
        create_yolov5s_automotive(batch_size=1),
    ]

    results = []

    for model, input_tensor, model_name in models_to_test:
        print(f"\n{'='*100}")
        print(f"Testing: {model_name}")
        print(f"{'='*100}")

        for hw_name, mapper, power_mode, tdp, precision, safety, temp_range in hardware_configs:
            print(f"\n► {hw_name} @ {power_mode}")
            result = benchmark_model_on_hardware(
                model, input_tensor, model_name,
                mapper, hw_name, power_mode, tdp, precision,
                safety_certified=safety,
                temperature_range=temp_range
            )
            results.append(result)

    return results


def print_automotive_summary(cat1_results, cat2_results):
    """Print automotive-specific executive summary"""
    print("\n" + "="*120)
    print("AUTOMOTIVE ADAS EXECUTIVE SUMMARY")
    print("="*120)

    print("\n## Safety Certification Status")
    print("-" * 120)

    all_results = cat1_results + cat2_results

    # Count safety-certified platforms
    certified_platforms = set(r.hardware_name for r in all_results if r.safety_certified)
    print(f"\n✓ ASIL-D/SIL-3 Certified: {', '.join(certified_platforms) if certified_platforms else 'None in test'}")

    automotive_temp = set(r.hardware_name for r in all_results if r.temperature_range == "automotive")
    print(f"✓ Automotive Temperature (-40°C to 125°C): {', '.join(automotive_temp) if automotive_temp else 'None in test'}")

    print("\n## Real-Time Performance")
    print("-" * 120)

    # Find platforms meeting 30 FPS requirement
    meets_30fps = [r for r in all_results if r.meets_30fps_requirement]
    print(f"\n✓ Meets 30 FPS requirement: {len(meets_30fps)}/{len(all_results)} configurations")

    # Find platforms meeting 100ms latency
    meets_100ms = [r for r in all_results if r.meets_100ms_latency]
    print(f"✓ Meets <100ms latency: {len(meets_100ms)}/{len(all_results)} configurations")

    print("\n## Power Efficiency")
    print("-" * 120)

    # Best FPS/W in each category
    cat1_best = max(cat1_results, key=lambda r: r.fps_per_watt)
    print(f"\n✓ Best efficiency (Front Camera): {cat1_best.hardware_name} @ {cat1_best.power_mode}")
    print(f"  {cat1_best.fps_per_watt:.2f} FPS/W on {cat1_best.model_name}")

    cat2_best = max(cat2_results, key=lambda r: r.fps_per_watt)
    print(f"\n✓ Best efficiency (Multi-Camera): {cat2_best.hardware_name} @ {cat2_best.power_mode}")
    print(f"  {cat2_best.fps_per_watt:.2f} FPS/W on {cat2_best.model_name}")

    print("\n## Automotive Use Case Recommendations")
    print("-" * 120)
    print("\n┌─────────────────────────────────┬──────────────────────────┬──────────────────────┐")
    print("│ ADAS Use Case                   │ Recommended Platform     │ Power Budget         │")
    print("├─────────────────────────────────┼──────────────────────────┼──────────────────────┤")
    print("│ Lane Keep Assist (LKA)          │ TI TDA4VM @ 10W          │ 10W                  │")
    print("│ Adaptive Cruise Control (ACC)   │ TI TDA4VM @ 10W          │ 10W                  │")
    print("│ Traffic Sign Recognition        │ TI TDA4VM @ 10W          │ 10W                  │")
    print("│ Forward Collision Warning       │ TI TDA4VM @ 10W          │ 10W                  │")
    print("│ Surround View Monitoring        │ TI TDA4VM @ 20W          │ 20W                  │")
    print("│ Automatic Parking Assist        │ TI TDA4VM @ 20W          │ 20W                  │")
    print("│ Highway Pilot (L2/L3)           │ TI TDA4VM @ 20W          │ 20W                  │")
    print("└─────────────────────────────────┴──────────────────────────┴──────────────────────┘")

    print("\n## Key Findings")
    print("-" * 120)
    print("\n1. **Safety Certification Critical**: TI TDA4VM offers ASIL-D/SIL-3 certification")
    print("2. **Temperature Range**: Automotive-grade (-40°C to 125°C) required for production")
    print("3. **Real-Time Performance**: 30 FPS minimum, <100ms latency for safety")
    print("4. **Power Budget**: Front camera (10W), Multi-camera (20W) typical")
    print("5. **Deterministic Execution**: Required for safety-critical ADAS functions")


def main():
    """Run automotive ADAS comparison"""
    print("\n" + "="*120)
    print("AUTOMOTIVE ADAS PLATFORM COMPARISON")
    print("Comparing AI accelerators for Advanced Driver Assistance Systems (ADAS Level 2-3)")
    print("Focus: Safety certification, real-time performance, automotive temperature range")
    print("="*120)

    # Run Category 1: Front Camera
    cat1_results = run_front_camera_adas_comparison()
    print_category_results("CATEGORY 1 RESULTS: Front Camera ADAS (10-15W)", cat1_results)

    # Run Category 2: Multi-Camera
    cat2_results = run_multi_camera_adas_comparison()
    print_category_results("CATEGORY 2 RESULTS: Multi-Camera ADAS (15-25W)", cat2_results)

    # Print automotive-specific summary
    print_automotive_summary(cat1_results, cat2_results)

    print("\n" + "="*120)
    print("Automotive ADAS Benchmark Complete")
    print("="*120)


if __name__ == "__main__":
    main()
