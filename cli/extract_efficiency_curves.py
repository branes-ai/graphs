#!/usr/bin/env python3
"""
Extract Hard-Coded Efficiency Curves from roofline.py

This script extracts the efficiency scaling factors currently embedded in
roofline.py and outputs them as structured JSON files for each hardware target.

This is a ONE-TIME migration script to externalize hard-coded values.

Usage:
    ./cli/extract_efficiency_curves.py --output hardware_registry/

Output files created:
    hardware_registry/gpu/jetson_orin_agx/efficiency_curves.json
    hardware_registry/cpu/i7-12700K/efficiency_curves.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def create_data_point(flops: float, efficiency: float, description: str = "",
                      source_models: list = None) -> dict:
    """Create an efficiency data point with uncertainty estimates.

    Since legacy values don't have statistical properties, we use
    conservative estimates: 20% relative uncertainty.
    """
    std_estimate = efficiency * 0.20
    return {
        "flops": flops,
        "efficiency_mean": efficiency,
        "efficiency_std": std_estimate,
        "efficiency_min": efficiency * 0.8,
        "efficiency_max": efficiency * 1.2,
        "ci_lower": max(0.0, efficiency - 1.96 * std_estimate),
        "ci_upper": efficiency + 1.96 * std_estimate,
        "num_observations": 1,
        "source": "legacy_hardcoded",
        "source_subgraphs": [description] if description else [],
        "source_models": source_models or [],
    }


def create_fitted_curve(breakpoints: list) -> dict:
    """Create fitted curve parameters from breakpoints."""
    return {
        "curve_type": "piecewise_linear",
        "breakpoints": breakpoints,
        "r_squared": 0.0,  # Unknown for legacy values
        "residual_std": 0.0,
        "max_residual": 0.0,
    }


def extract_gpu_efficiency_curves() -> dict:
    """
    Extract GPU efficiency curves from roofline.py hard-coded values.

    Source: roofline.py lines 537-827
    Calibration: Jetson Orin AGX 50W, FP32, 2026-02-04
    """

    # GPU Compute Efficiency Curves
    # From _get_compute_efficiency_scale() in roofline.py

    curves = {}

    # 1. Depthwise Convolution (lines 647-650)
    # Severely inefficient: 3% of standard conv efficiency
    curves["conv2d_depthwise"] = {
        "operation_type": "conv2d_depthwise",
        "description": "Depthwise separable convolution - severely memory-bound on GPU",
        "data_points": [
            create_data_point(1e6, 0.03, "depthwise any size", ["mobilenet_v2"]),
            create_data_point(10e6, 0.03, "depthwise any size", ["mobilenet_v2"]),
            create_data_point(100e6, 0.03, "depthwise any size", ["mobilenet_v2"]),
        ],
        "fitted_curve": create_fitted_curve([
            (1e6, 0.03),
            (100e6, 0.03),
        ]),
        "min_flops": 1e6,
        "max_flops": 100e6,
        "calibration_models": ["mobilenet_v2", "efficientnet_b0"],
        "notes": "Depthwise gets 3-80 GFLOPS vs 968 GFLOPS for standard conv = ~3% efficiency",
    }

    # 2. MBConv-style blocks (lines 665-683)
    # EfficientNet/MobileNet fused blocks
    curves["mbconv"] = {
        "operation_type": "mbconv",
        "description": "MBConv-style fused blocks (pointwise + depthwise)",
        "data_points": [
            create_data_point(5e6, 0.025, "very small MBConv <10M", ["efficientnet_b0"]),
            create_data_point(10e6, 0.025, "small MBConv 10M", ["efficientnet_b0"]),
            create_data_point(20e6, 0.030, "small MBConv 20M", ["efficientnet_b0"]),
            create_data_point(30e6, 0.035, "medium MBConv 30M", ["efficientnet_b1"]),
            create_data_point(40e6, 0.045, "medium MBConv 40M", ["efficientnet_b1"]),
            create_data_point(50e6, 0.055, "medium MBConv 50M", ["efficientnet_b2"]),
            create_data_point(75e6, 0.075, "large MBConv 75M", ["efficientnet_b2"]),
            create_data_point(100e6, 0.10, "large MBConv 100M+", ["efficientnet_b2"]),
        ],
        "fitted_curve": create_fitted_curve([
            (5e6, 0.025),
            (10e6, 0.025),
            (30e6, 0.035),
            (50e6, 0.055),
            (100e6, 0.10),
        ]),
        "min_flops": 5e6,
        "max_flops": 100e6,
        "calibration_models": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenet_v2"],
        "notes": "MBConv blocks achieve much lower efficiency than standard Conv2d+BN",
    }

    # 3. Tiny operations (lines 685-696)
    curves["generic_tiny"] = {
        "operation_type": "generic_tiny",
        "description": "Tiny operations (<10M FLOPs) - kernel launch dominated",
        "data_points": [
            create_data_point(0.5e6, 0.01, "tiny <1M", []),
            create_data_point(1e6, 0.02, "tiny 1M", []),
            create_data_point(5e6, 0.04, "small 5M", []),
            create_data_point(10e6, 0.06, "small 10M", []),
        ],
        "fitted_curve": create_fitted_curve([
            (0.5e6, 0.01),
            (1e6, 0.02),
            (10e6, 0.06),
        ]),
        "min_flops": 0.5e6,
        "max_flops": 10e6,
        "calibration_models": [],
        "notes": "Kernel launch overhead dominates for tiny operations",
    }

    # 4. Conv2D + BatchNorm pattern (lines 706-715)
    # BatchNorm fusion factor = 0.67
    curves["conv2d_batchnorm"] = {
        "operation_type": "conv2d_batchnorm",
        "description": "Conv2D fused with BatchNorm (BN adds memory overhead)",
        "data_points": [
            create_data_point(10e6, 0.06, "small-med 10M", ["resnet18"]),
            create_data_point(50e6, 0.23, "medium 50M (0.35*0.67)", ["resnet18"]),
            create_data_point(100e6, 0.40, "medium 100M", ["resnet18"]),
            create_data_point(200e6, 0.54, "medium 200M (0.80*0.67)", ["resnet18"]),
            create_data_point(231e6, 0.73, "ResNet-18 conv layer 231M", ["resnet18"]),
            create_data_point(500e6, 0.80, "large 500M", ["resnet50"]),
            create_data_point(1e9, 2.50, "very large 1G", ["vgg16"]),
            create_data_point(3.7e9, 4.50, "VGG-style 3.7G", ["vgg16"]),
            create_data_point(5e9, 5.00, "huge >5G", ["vgg19"]),
        ],
        "fitted_curve": create_fitted_curve([
            (10e6, 0.06),
            (50e6, 0.23),
            (200e6, 0.54),
            (500e6, 0.80),
            (5e9, 5.00),
        ]),
        "min_flops": 10e6,
        "max_flops": 10e9,
        "calibration_models": ["resnet18", "resnet50", "vgg16", "vgg19"],
        "notes": "BatchNorm fusion factor 0.67 derived from ResNet-18: 720 GFLOPS with BN vs 1083 GFLOPS without",
    }

    # 5. Conv2D only (no BatchNorm) (lines 717-719)
    curves["conv2d"] = {
        "operation_type": "conv2d",
        "description": "Conv2D without BatchNorm - higher efficiency",
        "data_points": [
            create_data_point(50e6, 0.60, "medium 50M", ["custom"]),
            create_data_point(200e6, 1.13, "medium 200M", ["resnet18"]),
            create_data_point(231e6, 1.13, "Conv2D+ReLU 231M no BN", ["resnet18"]),
            create_data_point(500e6, 1.30, "large 500M", ["vgg16"]),
            create_data_point(3.7e9, 5.40, "VGG-style 3.7G", ["vgg16"]),
            create_data_point(5e9, 5.60, "huge >5G", ["vgg19"]),
        ],
        "fitted_curve": create_fitted_curve([
            (50e6, 0.60),
            (200e6, 1.13),
            (500e6, 1.30),
            (5e9, 5.60),
        ]),
        "min_flops": 50e6,
        "max_flops": 10e9,
        "calibration_models": ["resnet18", "vgg16", "vgg19"],
        "notes": "Conv2D+ReLU (no BN): 1083 GFLOPS measured at 231M = 1.13x of 958 base",
    }

    # 6. MatMul / Attention (lines 721-722, 757-786)
    curves["matmul"] = {
        "operation_type": "matmul",
        "description": "Matrix multiplication and attention operations",
        "data_points": [
            create_data_point(10e6, 0.20, "small 10M", ["vit_b_16"]),
            create_data_point(50e6, 0.35, "medium 50M", ["vit_b_16"]),
            create_data_point(200e6, 0.60, "large 200M", ["vit_b_16"]),
            create_data_point(500e6, 0.90, "large 500M", ["vit_l_16"]),
            create_data_point(1e9, 1.15, "very large 1G", ["vit_l_16"]),
            create_data_point(5e9, 1.40, "huge 5G", ["vit_h_14"]),
            create_data_point(10e9, 1.50, "massive >5G", ["vit_h_14"]),
        ],
        "fitted_curve": create_fitted_curve([
            (10e6, 0.20),
            (50e6, 0.35),
            (200e6, 0.60),
            (500e6, 0.90),
            (5e9, 1.40),
            (10e9, 1.50),
        ]),
        "min_flops": 10e6,
        "max_flops": 20e9,
        "calibration_models": ["vit_b_16", "vit_l_16", "vit_h_14"],
        "notes": "MatMul efficiency scales strongly with size; ViT-B/16 avg 558M FLOPs/sg -> 67% eff",
    }

    return curves


def extract_gpu_bandwidth_curves() -> dict:
    """
    Extract GPU bandwidth efficiency curves from roofline.py.

    Source: roofline.py lines 829-933
    """
    curves = {}

    # 1. Depthwise bandwidth (lines 883-887)
    curves["conv2d_depthwise"] = {
        "operation_type": "conv2d_depthwise",
        "description": "Depthwise convolution bandwidth - scattered access patterns",
        "data_points": [
            create_data_point(1e6, 0.02, "depthwise any size", ["mobilenet_v2"]),
            create_data_point(10e6, 0.02, "depthwise any size", ["mobilenet_v2"]),
        ],
        "fitted_curve": create_fitted_curve([
            (1e6, 0.02),
            (10e6, 0.02),
        ]),
        "min_flops": 1e6,
        "max_flops": 10e6,
        "calibration_models": ["mobilenet_v2"],
        "notes": "~4 GB/s on 204 GB/s peak = 2% bandwidth efficiency",
    }

    # 2. Generic bandwidth (lines 905-925)
    curves["generic"] = {
        "operation_type": "generic",
        "description": "Generic bandwidth efficiency for memory-bound operations",
        "data_points": [
            create_data_point(10e3, 0.30, "<10KB", []),
            create_data_point(100e3, 0.35, "10KB-100KB", []),
            create_data_point(1e6, 0.50, "1MB", []),
            create_data_point(10e6, 0.60, "10MB", []),
            create_data_point(100e6, 0.70, ">10MB streaming", []),
        ],
        "fitted_curve": create_fitted_curve([
            (10e3, 0.30),
            (1e6, 0.50),
            (10e6, 0.60),
            (100e6, 0.70),
        ]),
        "min_flops": 10e3,
        "max_flops": 100e6,
        "calibration_models": [],
        "notes": "Bandwidth efficiency ranges 30-70% depending on access patterns and size",
    }

    return curves


def extract_cpu_efficiency_curves() -> dict:
    """
    Extract CPU efficiency curves from roofline.py.

    Source: roofline.py lines 788-823
    Calibration: i7-12700K, FP32
    """
    curves = {}

    # CPU efficiency model (lines 799-823)
    curves["generic"] = {
        "operation_type": "generic",
        "description": "Generic CPU efficiency - operation size dependent",
        "data_points": [
            create_data_point(0.5e6, 0.15, "tiny <1M", []),
            create_data_point(1e6, 0.15, "tiny 1M", []),
            create_data_point(5e6, 0.20, "small 5M", []),
            create_data_point(9e6, 0.17, "MobileNet avg 9M", ["mobilenet_v2"]),
            create_data_point(10e6, 0.25, "small 10M", []),
            create_data_point(50e6, 0.47, "medium 50M", []),
            create_data_point(100e6, 0.70, "large 100M", []),
            create_data_point(113e6, 0.85, "ResNet avg 113M", ["resnet18"]),
            create_data_point(225e6, 1.00, "ViT avg 225M", ["vit_b_16"]),
            create_data_point(500e6, 1.00, "very large 500M", []),
        ],
        "fitted_curve": create_fitted_curve([
            (1e6, 0.15),
            (10e6, 0.25),
            (100e6, 0.70),
            (500e6, 1.00),
        ]),
        "min_flops": 1e6,
        "max_flops": 1e9,
        "calibration_models": ["mobilenet_v2", "resnet18", "vit_b_16"],
        "notes": "CPU efficiency: MobileNet (9M avg) -> 0.17, ResNet (113M avg) -> 0.85, ViT (225M avg) -> 1.0",
    }

    return curves


def create_efficiency_profile(
    hardware_id: str,
    hardware_name: str,
    device_type: str,
    precision: str,
    compute_curves: dict,
    bandwidth_curves: dict,
    notes: str = "",
) -> dict:
    """Create a complete efficiency profile."""
    return {
        "schema_version": "1.0",
        "hardware_id": hardware_id,
        "hardware_name": hardware_name,
        "device_type": device_type,
        "precision": precision,
        "calibration_date": datetime.now().isoformat(),
        "calibration_tool_version": "extract_efficiency_curves.py v1.0",
        "curves": compute_curves,
        "bandwidth_curves": bandwidth_curves,
        "validation": None,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract hard-coded efficiency curves from roofline.py to JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="hardware_registry",
        help="Output directory (default: hardware_registry)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output without writing files",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    print("=" * 70)
    print("  EXTRACTING EFFICIENCY CURVES FROM ROOFLINE.PY")
    print("=" * 70)
    print()

    # Extract GPU curves (Jetson Orin AGX 50W)
    print("Extracting GPU efficiency curves (Jetson Orin AGX 50W)...")
    gpu_compute = extract_gpu_efficiency_curves()
    gpu_bandwidth = extract_gpu_bandwidth_curves()

    gpu_profile = create_efficiency_profile(
        hardware_id="jetson-orin-agx-50w",
        hardware_name="NVIDIA Jetson AGX Orin 64GB (50W)",
        device_type="gpu",
        precision="FP32",
        compute_curves=gpu_compute,
        bandwidth_curves=gpu_bandwidth,
        notes="Extracted from roofline.py hard-coded values. "
              "Calibrated on Jetson Orin AGX 50W with TF32 enabled (cuDNN default). "
              "Base peak: 958 GFLOPS FP32. "
              "Statistical uncertainty is estimated (20% relative) since original values lacked statistics.",
    )

    print(f"  - {len(gpu_compute)} compute curves")
    print(f"  - {len(gpu_bandwidth)} bandwidth curves")

    # Extract CPU curves (i7-12700K)
    print("Extracting CPU efficiency curves (i7-12700K)...")
    cpu_compute = extract_cpu_efficiency_curves()

    cpu_profile = create_efficiency_profile(
        hardware_id="i7-12700K",
        hardware_name="Intel Core i7-12700K (Alder Lake)",
        device_type="cpu",
        precision="FP32",
        compute_curves=cpu_compute,
        bandwidth_curves={},  # CPU bandwidth curves not yet extracted
        notes="Extracted from roofline.py hard-coded values. "
              "Calibrated on i7-12700K with MobileNet, ResNet, ViT reference models. "
              "Statistical uncertainty is estimated (20% relative) since original values lacked statistics.",
    )

    print(f"  - {len(cpu_compute)} compute curves")
    print()

    # Output files
    gpu_output_path = output_dir / "gpu" / "jetson_orin_agx" / "efficiency_curves.json"
    cpu_output_path = output_dir / "cpu" / "i7-12700K" / "efficiency_curves.json"

    if args.dry_run:
        print("DRY RUN - Would write to:")
        print(f"  {gpu_output_path}")
        print(f"  {cpu_output_path}")
        print()
        print("GPU Profile Preview:")
        print(json.dumps(gpu_profile, indent=2)[:2000] + "...")
        print()
        print("CPU Profile Preview:")
        print(json.dumps(cpu_profile, indent=2)[:2000] + "...")
    else:
        # Write GPU profile
        gpu_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gpu_output_path, 'w') as f:
            json.dump(gpu_profile, f, indent=2)
        print(f"Written: {gpu_output_path}")

        # Write CPU profile
        cpu_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cpu_output_path, 'w') as f:
            json.dump(cpu_profile, f, indent=2)
        print(f"Written: {cpu_output_path}")

    print()
    print("=" * 70)
    print("  EXTRACTION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review extracted curves for accuracy")
    print("  2. Run calibration to add measured statistical properties")
    print("  3. Update roofline.py to load curves from these files")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
