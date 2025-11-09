#!/usr/bin/env python3
"""
Embodied AI Hardware Comparison Test

Compares all 12 hardware platforms on Embodied AI workloads:
- Object Detection: YOLO (YOLOv8n, YOLOv8m)
- Segmentation: DeepLabV3+ MobileNetV3
- Re-Identification: ResNet-18 (256×128 input)

Generates comparison tables showing:
- Latency (ms/inference)
- Energy (mJ/inference)
- FPS (frames/sec)
- Power efficiency (FPS/W)
- BOM cost per FPS

Hardware platforms tested:
  Entry: KPU-T64, Hailo-8, Hailo-10H, Coral TPU, QCS6490, Jetson Nano
  Mid: KPU-T256, SA8775P, Jetson AGX
  High: KPU-T768, Snapdragon Ride, Jetson Thor
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator

# Hardware configurations to test
HARDWARE_CONFIGS = {
    # Entry-level (1-10W)
    "entry": [
        ("KPU-T64", "6W"),           # Stillwater KPU-T64 @ 6W
        ("Hailo-8", "2.5W"),          # Hailo-8 @ 2.5W
        ("Hailo-10H", "2.5W"),        # Hailo-10H @ 2.5W
        ("Coral-Edge-TPU", "2W"),     # Google Coral @ 2W
        ("Qualcomm-QCS6490", "10W"),  # Qualcomm QCS6490 @ 10W
        ("Jetson-Orin-Nano", "7W"),   # NVIDIA Jetson Orin Nano @ 7W
    ],

    # Mid-range (15-60W)
    "mid": [
        ("KPU-T256", "30W"),          # Stillwater KPU-T256 @ 30W
        ("Hailo-10H", "2.5W"),        # Hailo-10H (also entry)
        ("Qualcomm-SA8775P", "30W"),  # Qualcomm SA8775P @ 30W
        ("Jetson-Orin-AGX", "15W"),   # NVIDIA Jetson Orin AGX @ 15W
    ],

    # High-end (60-130W)
    "high": [
        ("KPU-T768", "80W"),                      # Stillwater KPU-T768 @ 80W
        ("Qualcomm-Snapdragon-Ride", "100W"),     # Qualcomm Snapdragon Ride @ 100W
        ("Jetson-Thor", "100W"),                  # NVIDIA Jetson Thor @ 100W
    ],
}

# Embodied AI workloads
EMBODIED_AI_WORKLOADS = {
    "yolov8n": {
        "description": "YOLOv8 Nano - Object Detection",
        "model": "yolov8n",  # Ultralytics will download if needed
        "input_shape": (1, 3, 640, 640),
        "params_m": 3.2,
        "gflops": 8.7,
    },
    "yolov8m": {
        "description": "YOLOv8 Medium - Object Detection",
        "model": "yolov8m",  # Ultralytics will download if needed
        "input_shape": (1, 3, 640, 640),
        "params_m": 25.9,
        "gflops": 78.9,
    },
    "deeplabv3": {
        "description": "DeepLabV3+ MobileNetV3 - Segmentation",
        "model": "deeplabv3_mobilenet_v3_large",
        "input_shape": (1, 3, 512, 512),
        "params_m": 5.8,
        "gflops": 17.3,
    },
    "reid": {
        "description": "ResNet-18 - Re-Identification",
        "model": "resnet18",
        "input_shape": (1, 3, 256, 128),
        "params_m": 11.7,
        "gflops": 1.8,
    },
}


def load_and_trace_model(workload_info: Dict) -> Tuple[torch.nn.Module, torch.fx.GraphModule]:
    """
    Load and trace a model based on workload info.

    Returns:
        (original_model, traced_graph_module)
    """
    import torchvision.models as tv_models

    model_spec = workload_info["model"]
    input_shape = workload_info["input_shape"]

    # Create example input
    example_input = torch.randn(*input_shape)

    # Load model
    if model_spec.endswith(".pt") or model_spec.lower().startswith(('yolov', 'yolo')):
        # YOLO model - can be a file path or model name
        # Ultralytics YOLO() handles both:
        # - Model names like 'yolov8n' (downloads if needed)
        # - File paths like 'yolov8n.pt' (loads from disk)
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("YOLO models require ultralytics. Install with: pip install ultralytics")

        yolo = YOLO(model_spec)
        model = yolo.model.eval()
    else:
        # TorchVision model
        if model_spec == "deeplabv3_mobilenet_v3_large":
            model = tv_models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)
        elif model_spec == "resnet18":
            model = tv_models.resnet18(weights=None)
        else:
            raise ValueError(f"Unknown model: {model_spec}")

        model.eval()

    # Warm-up
    with torch.no_grad():
        try:
            _ = model(example_input)
        except Exception:
            pass

    # Trace with Dynamo export (state-of-the-art, more reliable than symbolic_trace)
    # Warm-up first (important for lazy initialization like YOLO)
    with torch.no_grad():
        try:
            _ = model(example_input)
        except Exception:
            pass  # Some models fail warm-up, continue anyway

    try:
        exported_program = torch.export.export(model, (example_input,))
        traced = exported_program.module()
    except Exception as e:
        raise RuntimeError(f"Failed to trace model with Dynamo: {e}")

    # Run shape propagation
    ShapeProp(traced).propagate(example_input)

    return model, traced


def get_hardware_mapper(hardware_name: str):
    """Create hardware mapper for the specified platform"""
    from graphs.hardware.mappers.gpu import (
        create_jetson_orin_nano_mapper,
        create_jetson_orin_agx_mapper,
        create_jetson_thor_mapper,
    )
    from graphs.hardware.mappers.accelerators.tpu import (
        create_coral_edge_tpu_mapper,
    )
    from graphs.hardware.mappers.accelerators.kpu import (
        create_kpu_t64_mapper,
        create_kpu_t256_mapper,
        create_kpu_t768_mapper,
    )
    from graphs.hardware.mappers.accelerators.hailo import (
        create_hailo8_mapper,
        create_hailo10h_mapper,
    )
    from graphs.hardware.mappers.dsp import (
        create_qrb5165_mapper,
        create_qualcomm_sa8775p_mapper,
        create_qualcomm_snapdragon_ride_mapper,
    )

    mapper_funcs = {
        # Entry-level
        "KPU-T64": create_kpu_t64_mapper,
        "Hailo-8": create_hailo8_mapper,
        "Hailo-10H": create_hailo10h_mapper,
        "Coral-Edge-TPU": create_coral_edge_tpu_mapper,
        "Qualcomm-QCS6490": create_qrb5165_mapper,  # Use QRB5165 as proxy
        "Jetson-Orin-Nano": create_jetson_orin_nano_mapper,

        # Mid-range
        "KPU-T256": create_kpu_t256_mapper,
        "Qualcomm-SA8775P": create_qualcomm_sa8775p_mapper,
        "Jetson-Orin-AGX": create_jetson_orin_agx_mapper,

        # High-end
        "KPU-T768": create_kpu_t768_mapper,
        "Qualcomm-Snapdragon-Ride": create_qualcomm_snapdragon_ride_mapper,
        "Jetson-Thor": create_jetson_thor_mapper,
    }

    if hardware_name not in mapper_funcs:
        raise ValueError(f"No mapper available for {hardware_name}")

    return mapper_funcs[hardware_name]()


def test_hardware_on_workload(
    hardware_name: str,
    thermal_profile: str,
    workload_name: str,
    workload_info: Dict
) -> Dict:
    """
    Test a single hardware configuration on a workload.

    Returns:
        Dictionary with latency, energy, FPS, cost metrics
    """
    print(f"  Testing {hardware_name} @ {thermal_profile} on {workload_name}...")

    try:
        # Load and trace model
        model, traced_model = load_and_trace_model(workload_info)

        # Create example input
        example_input = torch.randn(*workload_info["input_shape"])

        # Get hardware mapper
        hardware_mapper = get_hardware_mapper(hardware_name)

        # Create analyzer
        analyzer = UnifiedAnalyzer(verbose=False)
        config = AnalysisConfig(
            run_hardware_mapping=True,
            run_roofline=True,
            run_energy=True,
            run_memory=True,
            run_concurrency=False,
        )

        # Run analysis
        from graphs.hardware.resource_model import Precision
        result_obj = analyzer.analyze_model_with_custom_hardware(
            model=model,
            input_tensor=example_input,
            model_name=workload_name,
            hardware_mapper=hardware_mapper,
            precision=Precision.INT8,  # Embodied AI typically uses INT8
            config=config
        )

        # Extract metrics from result
        if result_obj.roofline_report:
            latency_ms = result_obj.roofline_report.total_latency * 1000.0  # seconds to ms
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        else:
            latency_ms = 0.0
            fps = 0.0

        if result_obj.energy_report:
            energy_mj = result_obj.energy_report.total_energy_j * 1000.0
        else:
            energy_mj = 0.0

        # Get BOM cost from hardware model
        hardware = hardware_mapper.resource_model
        if hasattr(hardware, 'bom_cost_profile') and hardware.bom_cost_profile:
            bom_cost = hardware.bom_cost_profile.total_bom_cost
        else:
            bom_cost = 0.0

        # Calculate derived metrics
        avg_power_w = (energy_mj / latency_ms) if latency_ms > 0 else 0.0
        fps_per_watt = fps / avg_power_w if avg_power_w > 0 else 0.0
        cost_per_fps = bom_cost / fps if fps > 0 else 0.0

        result = {
            "hardware": hardware_name,
            "thermal_profile": thermal_profile,
            "workload": workload_name,
            "latency_ms": latency_ms,
            "energy_mj": energy_mj,
            "fps": fps,
            "fps_per_watt": fps_per_watt,
            "bom_cost": bom_cost,
            "cost_per_fps": cost_per_fps,
            "status": "PASS",
        }

        print(f"    ✓ {latency_ms:.2f}ms, {fps:.1f} FPS, {energy_mj:.1f}mJ")
        return result

    except Exception as e:
        import traceback
        print(f"    ❌ Failed: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return {
            "hardware": hardware_name,
            "thermal_profile": thermal_profile,
            "workload": workload_name,
            "status": "FAIL",
            "error": str(e),
        }


def run_comparison(tier: str = "all") -> Dict[str, List[Dict]]:
    """
    Run Embodied AI comparison across hardware platforms.

    Args:
        tier: "entry", "mid", "high", or "all"

    Returns:
        Dictionary mapping workload name to list of results
    """
    # Determine which hardware to test
    if tier == "all":
        hardware_list = []
        for tier_hw in HARDWARE_CONFIGS.values():
            hardware_list.extend(tier_hw)
    else:
        hardware_list = HARDWARE_CONFIGS.get(tier, [])

    if not hardware_list:
        raise ValueError(f"Unknown tier: {tier}. Choose from: entry, mid, high, all")

    print("=" * 100)
    print("EMBODIED AI HARDWARE COMPARISON")
    print("=" * 100)
    print()
    print(f"Testing {len(hardware_list)} hardware configurations on {len(EMBODIED_AI_WORKLOADS)} workloads")
    print()

    # Run tests
    results_by_workload = {}

    for workload_name, workload_info in EMBODIED_AI_WORKLOADS.items():
        print(f"\n{'='*100}")
        print(f"Workload: {workload_info['description']}")
        print(f"  Model: {workload_info['model']}")
        print(f"  Input: {workload_info['input_shape']}")
        print(f"  Params: {workload_info['params_m']:.1f}M")
        print(f"  FLOPs: {workload_info['gflops']:.1f} GFLOPs")
        print(f"{'='*100}")

        results = []
        for hardware_name, thermal_profile in hardware_list:
            result = test_hardware_on_workload(
                hardware_name, thermal_profile, workload_name, workload_info
            )
            results.append(result)

        results_by_workload[workload_name] = results

    return results_by_workload


def generate_comparison_table(results_by_workload: Dict[str, List[Dict]]) -> str:
    """Generate markdown comparison table"""

    table = []
    table.append("\n" + "=" * 100)
    table.append("EMBODIED AI HARDWARE COMPARISON SUMMARY")
    table.append("=" * 100)
    table.append("")

    for workload_name, results in results_by_workload.items():
        workload_info = EMBODIED_AI_WORKLOADS[workload_name]

        table.append(f"\n## {workload_info['description']}")
        table.append(f"**Model**: {workload_info['model']} | "
                    f"**FLOPs**: {workload_info['gflops']:.1f} GFLOPs | "
                    f"**Params**: {workload_info['params_m']:.1f}M")
        table.append("")

        # Table header
        table.append("| Hardware | Power | Latency (ms) | Energy (mJ) | FPS | FPS/W | BOM $/FPS | Status |")
        table.append("|----------|-------|--------------|-------------|-----|-------|-----------|--------|")

        # Sort by latency (ascending = better)
        sorted_results = sorted(results, key=lambda r: r.get("latency_ms", float('inf')))

        for result in sorted_results:
            hw = result["hardware"]
            pwr = result["thermal_profile"]
            lat = result.get("latency_ms", 0)
            eng = result.get("energy_mj", 0)
            fps = result.get("fps", 0)
            fps_w = result.get("fps_per_watt", 0)
            cost = result.get("cost_per_fps", 0)
            status = result.get("status", "TODO")

            if status == "TODO":
                table.append(f"| {hw} | {pwr} | TODO | TODO | TODO | TODO | TODO | ⚠️ {status} |")
            elif status == "FAIL":
                table.append(f"| {hw} | {pwr} | - | - | - | - | - | ❌ {status} |")
            else:
                table.append(f"| {hw} | {pwr} | {lat:.2f} | {eng:.1f} | {fps:.1f} | {fps_w:.2f} | ${cost:.2f} | ✅ {status} |")

        table.append("")

    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(
        description="Embodied AI Hardware Comparison Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tier",
        choices=["entry", "mid", "high", "all"],
        default="all",
        help="Hardware tier to test (default: all)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for comparison table (markdown format)"
    )

    args = parser.parse_args()

    # Run comparison
    results = run_comparison(tier=args.tier)

    # Generate table
    table = generate_comparison_table(results)
    print(table)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(table)
        print(f"\n✓ Comparison table saved to {output_path}")

    # Summary
    print("\n" + "=" * 100)
    print("TEST STATUS")
    print("=" * 100)
    print()
    print("⚠️  This is a TEMPLATE implementation")
    print()
    print("To complete the implementation:")
    print("  1. Download models: python scripts/download_embodied_ai_models.py")
    print("  2. Profile models: python cli/profile_graph.py --model yolov8n.pt")
    print("  3. Integrate with UnifiedAnalyzer to get actual latency/energy")
    print("  4. Add BOM cost lookup from hardware resource models")
    print()
    print("Once models are profiled, this script will generate accurate comparisons")
    print("showing latency, energy, FPS, and cost metrics for all 12 hardware platforms.")
    print()


if __name__ == "__main__":
    main()
