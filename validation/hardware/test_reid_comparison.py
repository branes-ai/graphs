#!/usr/bin/env python3
"""
Quick Re-ID comparison test

Tests ResNet-18 (Re-ID proxy) on all available entry-level hardware
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import torch.fx
import torchvision.models as tv_models
from torch.fx.passes.shape_prop import ShapeProp
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision
from graphs.hardware.mappers.gpu import (
    create_jetson_orin_nano_8gb_mapper,
)
from graphs.hardware.mappers.accelerators.tpu import (
    create_coral_edge_tpu_mapper,
)
from graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
)
from graphs.hardware.mappers.accelerators.hailo import (
    create_hailo8_mapper,
    create_hailo10h_mapper,
)
from graphs.hardware.mappers.dsp import (
    create_qrb5165_mapper,
)

def _test_hardware_helper(hardware_name, mapper_func):
    """Helper function to test Re-ID on a single hardware platform"""
    print(f"  Testing {hardware_name}...", end=" ", flush=True)

    try:
        # Create model
        model = tv_models.resnet18(weights=None)
        model.eval()

        # Re-ID input size
        example_input = torch.randn(1, 3, 256, 128)

        # Warm-up
        with torch.no_grad():
            _ = model(example_input)

        # Create mapper
        hardware_mapper = mapper_func()

        # Analyze
        analyzer = UnifiedAnalyzer(verbose=False)
        config = AnalysisConfig(
            run_hardware_mapping=True,
            run_roofline=True,
            run_energy=True,
            run_memory=False,
            run_concurrency=False,
        )

        result = analyzer.analyze_model_with_custom_hardware(
            model=model,
            input_tensor=example_input,
            model_name="ResNet18-ReID",
            hardware_mapper=hardware_mapper,
            precision=Precision.INT8,
            config=config
        )

        # Extract metrics
        latency_ms = result.roofline_report.total_latency * 1000.0
        fps = 1000.0 / latency_ms
        energy_mj = result.energy_report.total_energy_j * 1000.0
        avg_power_w = energy_mj / latency_ms

        # BOM cost
        hardware = hardware_mapper.resource_model
        if hasattr(hardware, 'bom_cost_profile') and hardware.bom_cost_profile:
            bom_cost = hardware.bom_cost_profile.total_bom_cost
        else:
            bom_cost = 0.0

        cost_per_fps = bom_cost / fps if fps > 0 else 0.0
        fps_per_watt = fps / avg_power_w if avg_power_w > 0 else 0.0

        print(f"✓")
        print(f"    {latency_ms:.2f}ms, {fps:.0f} FPS, {energy_mj:.1f}mJ, {avg_power_w:.2f}W")
        print(f"    BOM: ${bom_cost:.0f}, ${cost_per_fps:.2f}/FPS, {fps_per_watt:.1f} FPS/W")

        return {
            'hardware': hardware_name,
            'latency_ms': latency_ms,
            'fps': fps,
            'energy_mj': energy_mj,
            'power_w': avg_power_w,
            'bom_cost': bom_cost,
            'cost_per_fps': cost_per_fps,
            'fps_per_watt': fps_per_watt,
        }

    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 80)
    print("Re-ID (ResNet-18 @ 256×128) Hardware Comparison")
    print("=" * 80)
    print()

    hardware_configs = [
        ("KPU-T64", create_kpu_t64_mapper),
        ("Hailo-8", create_hailo8_mapper),
        ("Hailo-10H", create_hailo10h_mapper),
        ("Coral Edge TPU", create_coral_edge_tpu_mapper),
        ("Qualcomm QRB5165", create_qrb5165_mapper),
        ("Jetson Orin Nano", create_jetson_orin_nano_8gb_mapper),
    ]

    results = []
    for hw_name, mapper_func in hardware_configs:
        result = _test_hardware_helper(hw_name, mapper_func)
        if result:
            results.append(result)
        print()

    # Summary table
    if results:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print(f"{'Hardware':<20} {'Latency':<10} {'FPS':<8} {'Power':<8} {'BOM':<8} {'$/FPS':<10} {'FPS/W':<8}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x['latency_ms']):
            print(f"{r['hardware']:<20} {r['latency_ms']:>6.2f} ms {r['fps']:>6.0f}   "
                  f"{r['power_w']:>6.2f} W ${r['bom_cost']:>6.0f}  ${r['cost_per_fps']:>8.2f}  "
                  f"{r['fps_per_watt']:>6.1f}")

        print()

if __name__ == "__main__":
    main()
