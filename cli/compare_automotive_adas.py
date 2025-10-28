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
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.hardware.mappers.dsp import create_ti_tda4vm_mapper
from graphs.hardware.mappers.gpu import create_jetson_orin_nano_mapper, create_jetson_orin_agx_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.resource_model import Precision


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
    """Automotive ADAS use case definition with realistic performance requirements"""
    name: str
    description: str
    camera_count: int
    power_budget_w: float
    target_fps: int
    max_latency_ms: float
    safety_level: str  # "ASIL-B", "ASIL-D", etc.
    min_tops_required: float  # Minimum TOPS needed for this use case
    autonomy_level: str  # "L1", "L2", "L3", "L4"


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


# ============================================================================
# ADAS Use Case Definitions (Based on Industry Standards)
# ============================================================================

def get_adas_use_cases():
    """
    Define realistic ADAS use cases with actual TOPS requirements.

    Performance requirements based on:
    - SAE J3016 autonomy levels
    - Industry benchmarks (Tesla, Mobileye, NVIDIA)
    - Academic research on DNN compute requirements
    """
    return [
        AutomotiveUseCase(
            name="Lane Keep Assist (LKA)",
            description="Single front camera, lane detection",
            camera_count=1,
            power_budget_w=15.0,
            target_fps=30,
            max_latency_ms=100.0,
            safety_level="ASIL-B",
            min_tops_required=5.0,  # Simple lane detection + tracking
            autonomy_level="L1"
        ),
        AutomotiveUseCase(
            name="Adaptive Cruise Control (ACC)",
            description="Front camera + radar fusion, object detection",
            camera_count=1,
            power_budget_w=15.0,
            target_fps=30,
            max_latency_ms=100.0,
            safety_level="ASIL-B",
            min_tops_required=8.0,  # Object detection + tracking + fusion
            autonomy_level="L1"
        ),
        AutomotiveUseCase(
            name="Traffic Sign Recognition (TSR)",
            description="Front camera, sign detection and classification",
            camera_count=1,
            power_budget_w=10.0,
            target_fps=10,  # Can be lower FPS
            max_latency_ms=200.0,
            safety_level="ASIL-A",
            min_tops_required=3.0,  # Lightweight detection + classification
            autonomy_level="L1"
        ),
        AutomotiveUseCase(
            name="Forward Collision Warning (FCW)",
            description="Front camera, real-time collision detection",
            camera_count=1,
            power_budget_w=15.0,
            target_fps=30,
            max_latency_ms=50.0,  # Critical safety function
            safety_level="ASIL-D",
            min_tops_required=10.0,  # Real-time detection + trajectory prediction
            autonomy_level="L1"
        ),
        AutomotiveUseCase(
            name="Surround View Monitoring (SVM)",
            description="4-6 cameras, 360° view stitching",
            camera_count=4,
            power_budget_w=20.0,
            target_fps=30,
            max_latency_ms=100.0,
            safety_level="ASIL-B",
            min_tops_required=20.0,  # Multi-camera processing
            autonomy_level="L2"
        ),
        AutomotiveUseCase(
            name="Automatic Parking Assist",
            description="Surround view + object detection + path planning",
            camera_count=4,
            power_budget_w=25.0,
            target_fps=15,  # Lower FPS acceptable for parking
            max_latency_ms=150.0,
            safety_level="ASIL-C",
            min_tops_required=30.0,  # Multi-camera + planning
            autonomy_level="L2"
        ),
        AutomotiveUseCase(
            name="Highway Pilot (L2/L3)",
            description="Multi-sensor fusion (cameras + radar + lidar)",
            camera_count=8,
            power_budget_w=60.0,
            target_fps=30,
            max_latency_ms=100.0,
            safety_level="ASIL-D",
            min_tops_required=300.0,  # Full sensor fusion + path planning + redundancy
            autonomy_level="L3"
        ),
    ]


@dataclass
class PlatformScore:
    """Score for a platform on a specific use case"""
    use_case: str
    hardware_name: str
    power_mode: str
    tdp_watts: float

    # Requirements check
    meets_tops_requirement: bool
    meets_power_budget: bool
    meets_latency_requirement: bool
    meets_fps_requirement: bool

    # Performance metrics
    effective_tops: float
    latency_ms: float
    fps: float
    fps_per_watt: float

    # Overall score (0-100)
    total_score: float

    # Warnings
    warnings: list = field(default_factory=list)


def calculate_platform_score(
    use_case: AutomotiveUseCase,
    result: AutomotiveBenchmarkResult
) -> PlatformScore:
    """
    Calculate how well a platform meets a use case's requirements.

    Scoring methodology:
    - Performance (50%): Must meet minimum TOPS requirement
    - Efficiency (20%): FPS per watt within power budget
    - Latency (20%): Must meet real-time requirements
    - Safety (10%): Bonus for certification
    """
    warnings = []

    # Check requirements
    meets_tops = result.tops_per_watt * result.tdp_watts >= use_case.min_tops_required
    meets_power = result.tdp_watts <= use_case.power_budget_w
    meets_latency = result.latency_ms <= use_case.max_latency_ms
    meets_fps = result.throughput_fps >= use_case.target_fps

    # Calculate effective TOPS (TOPS/W × actual power)
    effective_tops = result.tops_per_watt * result.tdp_watts

    # Performance score (0-50 points)
    if not meets_tops:
        performance_score = 0.0
        warnings.append(f"INSUFFICIENT PERFORMANCE: {effective_tops:.1f} TOPS < {use_case.min_tops_required:.1f} TOPS required")
    else:
        # Scale based on how much over requirement
        tops_ratio = effective_tops / use_case.min_tops_required
        performance_score = min(50.0, 30.0 + 20.0 * (tops_ratio - 1.0))

    # Efficiency score (0-20 points)
    if not meets_power:
        efficiency_score = 0.0
        warnings.append(f"POWER BUDGET EXCEEDED: {result.tdp_watts:.1f}W > {use_case.power_budget_w:.1f}W budget")
    else:
        # Higher FPS/W is better
        efficiency_score = min(20.0, result.fps_per_watt * 2.0)

    # Latency score (0-20 points)
    if not meets_latency:
        latency_score = 0.0
        warnings.append(f"LATENCY TOO HIGH: {result.latency_ms:.1f}ms > {use_case.max_latency_ms:.1f}ms max")
    else:
        # Scale based on margin below requirement
        latency_margin = (use_case.max_latency_ms - result.latency_ms) / use_case.max_latency_ms
        latency_score = 10.0 + 10.0 * latency_margin

    # Safety certification bonus (0-10 points)
    safety_score = 0.0
    if result.safety_certified:
        safety_score += 5.0
    if result.temperature_range == "automotive":
        safety_score += 5.0

    # Total score
    total_score = performance_score + efficiency_score + latency_score + safety_score

    return PlatformScore(
        use_case=use_case.name,
        hardware_name=result.hardware_name,
        power_mode=result.power_mode,
        tdp_watts=result.tdp_watts,
        meets_tops_requirement=meets_tops,
        meets_power_budget=meets_power,
        meets_latency_requirement=meets_latency,
        meets_fps_requirement=meets_fps,
        effective_tops=effective_tops,
        latency_ms=result.latency_ms,
        fps=result.throughput_fps,
        fps_per_watt=result.fps_per_watt,
        total_score=total_score,
        warnings=warnings,
    )


def recommend_platforms_for_use_cases(all_results: List[AutomotiveBenchmarkResult]):
    """
    Generate data-driven recommendations for each ADAS use case.

    For each use case, scores all tested platforms and recommends the best.
    """
    use_cases = get_adas_use_cases()
    recommendations = {}

    for use_case in use_cases:
        # Score all platforms for this use case
        scores = []
        for result in all_results:
            score = calculate_platform_score(use_case, result)
            scores.append(score)

        # Sort by total score (descending)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Best recommendation is highest score
        recommendations[use_case.name] = {
            'use_case': use_case,
            'scores': scores,
            'best': scores[0] if scores else None,
        }

    return recommendations


def print_automotive_summary(cat1_results, cat2_results):
    """Print automotive-specific executive summary with data-driven recommendations"""
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

    # Generate data-driven recommendations
    all_results = cat1_results + cat2_results
    recommendations = recommend_platforms_for_use_cases(all_results)

    print("\n## Automotive Use Case Recommendations (Data-Driven)")
    print("-" * 120)
    print("\n┌─────────────────────────────────┬──────────────────────────┬──────────┬──────────┬──────────────────────────────────┐")
    print("│ ADAS Use Case                   │ Recommended Platform     │ Power    │ Eff TOPS │ Status / Warnings                │")
    print("├─────────────────────────────────┼──────────────────────────┼──────────┼──────────┼──────────────────────────────────┤")

    for use_case_name, rec_data in recommendations.items():
        best = rec_data['best']
        use_case = rec_data['use_case']

        if best:
            platform_str = f"{best.hardware_name} @ {best.power_mode}"
            power_str = f"{best.tdp_watts:.0f}W"
            tops_str = f"{best.effective_tops:.1f}"

            # Status indicator
            if best.meets_tops_requirement and best.meets_power_budget:
                status = "✓ MEETS REQUIREMENTS"
            elif not best.meets_tops_requirement:
                status = f"✗ INSUFFICIENT ({best.effective_tops:.1f}/{use_case.min_tops_required:.0f} TOPS)"
            elif not best.meets_power_budget:
                status = f"⚠ OVER BUDGET ({best.tdp_watts:.0f}/{use_case.power_budget_w:.0f}W)"
            else:
                status = "⚠ CHECK WARNINGS"

            # Truncate for table width
            uc_str = use_case_name[:31].ljust(31)
            plat_str = platform_str[:24].ljust(24)
            power_str = power_str[:8].ljust(8)
            tops_str = tops_str[:8].ljust(8)
            status_str = status[:32].ljust(32)

            print(f"│ {uc_str} │ {plat_str} │ {power_str} │ {tops_str} │ {status_str} │")
        else:
            print(f"│ {use_case_name[:31].ljust(31)} │ {'NO DATA':24} │ {'N/A':8} │ {'N/A':8} │ {'NO PLATFORMS TESTED':32} │")

    print("└─────────────────────────────────┴──────────────────────────┴──────────┴──────────┴──────────────────────────────────┘")

    # Print detailed warnings for problematic recommendations
    print("\n## Performance Warnings")
    print("-" * 120)

    critical_warnings_found = False
    for use_case_name, rec_data in recommendations.items():
        best = rec_data['best']
        use_case = rec_data['use_case']

        if best and best.warnings:
            critical_warnings_found = True
            print(f"\n⚠ {use_case_name} ({use_case.autonomy_level}):")
            for warning in best.warnings:
                print(f"  - {warning}")

            # Additional context for L3/L4
            if use_case.autonomy_level in ["L3", "L4"]:
                print(f"  ⚠ CRITICAL: {use_case.autonomy_level} autonomy requires {use_case.min_tops_required:.0f} TOPS minimum")
                print(f"  → Industry examples: Tesla FSD (~1000 TOPS), Waymo (~2000 TOPS)")

    if not critical_warnings_found:
        print("\n✓ All recommended platforms meet minimum requirements for their use cases")

    print("\n## Key Findings (Data-Driven Analysis)")
    print("-" * 120)

    # Analyze which platforms are suitable for which autonomy levels
    l1_suitable = []
    l2_suitable = []
    l3_suitable = []

    for use_case_name, rec_data in recommendations.items():
        best = rec_data['best']
        use_case = rec_data['use_case']

        if best and best.meets_tops_requirement and best.meets_power_budget:
            if use_case.autonomy_level == "L1":
                l1_suitable.append(best.hardware_name)
            elif use_case.autonomy_level == "L2":
                l2_suitable.append(best.hardware_name)
            elif use_case.autonomy_level == "L3":
                l3_suitable.append(best.hardware_name)

    # Count unique platforms per level
    l1_platforms = set(l1_suitable)
    l2_platforms = set(l2_suitable)
    l3_platforms = set(l3_suitable)

    print("\n1. **Autonomy Level Suitability:**")
    print(f"   - L1 ADAS (Lane Keep, ACC, FCW): {len(l1_platforms)} platforms meet requirements")
    if l1_platforms:
        print(f"     → Suitable: {', '.join(l1_platforms)}")
    print(f"   - L2 ADAS (Surround View, Parking): {len(l2_platforms)} platforms meet requirements")
    if l2_platforms:
        print(f"     → Suitable: {', '.join(l2_platforms)}")
    print(f"   - L3 Highway Pilot (300 TOPS): {len(l3_platforms)} platforms meet requirements")
    if l3_platforms:
        print(f"     → Suitable: {', '.join(l3_platforms)}")
    else:
        print(f"     → ⚠ CRITICAL: None of the tested platforms meet L3 requirements!")

    print("\n2. **Performance vs Safety Certification Trade-off:**")
    certified_platforms = [r for r in all_results if r.safety_certified]
    high_perf_platforms = [r for r in all_results if r.tops_per_watt * r.tdp_watts >= 50.0]

    if certified_platforms:
        avg_tops_certified = sum(r.tops_per_watt * r.tdp_watts for r in certified_platforms) / len(certified_platforms)
        print(f"   - ASIL-D certified platforms: {len(certified_platforms)} tested (avg {avg_tops_certified:.1f} TOPS)")
    else:
        print(f"   - ASIL-D certified platforms: 0 tested")

    if high_perf_platforms:
        print(f"   - High-performance (>50 TOPS): {len(high_perf_platforms)} platforms")
        print(f"   → Modern L3 systems prioritize performance + system-level safety over chip certification")

    print("\n3. **Real-Time Performance Requirements:**")
    fps_ok = sum(1 for r in all_results if r.meets_30fps_requirement)
    latency_ok = sum(1 for r in all_results if r.meets_100ms_latency)
    print(f"   - {fps_ok}/{len(all_results)} configurations meet 30 FPS requirement")
    print(f"   - {latency_ok}/{len(all_results)} configurations meet <100ms latency requirement")

    print("\n4. **Power Efficiency Analysis:**")
    best_efficiency = max(all_results, key=lambda r: r.fps_per_watt)
    print(f"   - Best FPS/W: {best_efficiency.hardware_name} @ {best_efficiency.power_mode}")
    print(f"     → {best_efficiency.fps_per_watt:.2f} FPS/W ({best_efficiency.model_name})")

    print("\n5. **Industry Reality Check:**")
    print(f"   - Tesla FSD Computer: ~144 TOPS (2× redundant 72 TOPS chips)")
    print(f"   - NVIDIA DRIVE Orin: ~254 TOPS (full SoC, all accelerators)")
    print(f"   - L3 systems typically: 300-500 TOPS for full sensor fusion")
    print(f"   - L4 urban autonomy: 1000-2000 TOPS (POPS scale)")
    print(f"   → Performance gap is the primary bottleneck, not certification status")


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
