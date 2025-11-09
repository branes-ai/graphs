#!/usr/bin/env python3
"""
Quick test of Embodied AI comparison infrastructure

Tests a single workload on a single hardware platform to verify
the integration with UnifiedAnalyzer works correctly.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper

def test_resnet18_on_kpu():
    """Test ResNet-18 (re-ID proxy) on KPU-T64"""
    print("=" * 80)
    print("Testing ResNet-18 (256×128 Re-ID) on KPU-T64")
    print("=" * 80)
    print()

    # Create model
    print("[1/5] Creating ResNet-18 model...")
    import torchvision.models as models
    model = models.resnet18(weights=None)
    model.eval()

    # Create input (re-ID typical size: 256×128)
    print("[2/5] Creating example input (1, 3, 256, 128)...")
    example_input = torch.randn(1, 3, 256, 128)

    # Warm-up
    print("[3/5] Warming up model...")
    with torch.no_grad():
        _ = model(example_input)

    # Create hardware mapper
    print("[4/5] Creating KPU-T64 mapper...")
    hardware_mapper = create_kpu_t64_mapper()

    # Create analyzer
    print("[5/5] Running analysis...")
    analyzer = UnifiedAnalyzer(verbose=True)
    config = AnalysisConfig(
        run_hardware_mapping=True,
        run_roofline=True,
        run_energy=True,
        run_memory=True,
        run_concurrency=False,
    )

    result = analyzer.analyze_model_with_custom_hardware(
        model=model,
        input_tensor=example_input,
        model_name="ResNet-18-ReID",
        hardware_mapper=hardware_mapper,
        precision=Precision.INT8,
        config=config
    )

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    if result.roofline_report:
        latency_ms = result.roofline_report.total_latency * 1000.0  # seconds to ms
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        print(f"Latency:    {latency_ms:.2f} ms")
        print(f"Throughput: {fps:.1f} FPS")

    if result.energy_report:
        energy_mj = result.energy_report.total_energy_j * 1000.0
        print(f"Energy:     {energy_mj:.1f} mJ/inference")

        if latency_ms > 0:
            avg_power_w = energy_mj / latency_ms  # mJ / ms = W
            print(f"Avg Power:  {avg_power_w:.2f} W")

    if result.memory_report:
        peak_mb = result.memory_report.peak_memory_bytes / (1024 * 1024)
        print(f"Peak Memory: {peak_mb:.1f} MB")

    # BOM cost
    hardware = hardware_mapper.resource_model
    if hasattr(hardware, 'bom_cost_profile') and hardware.bom_cost_profile:
        bom_cost = hardware.bom_cost_profile.total_bom_cost
        print(f"BOM Cost:   ${bom_cost:.0f}")

        if fps > 0:
            cost_per_fps = bom_cost / fps
            print(f"Cost/FPS:   ${cost_per_fps:.2f}")

    print()
    print("✓ Test completed successfully!")
    print()

if __name__ == "__main__":
    test_resnet18_on_kpu()
