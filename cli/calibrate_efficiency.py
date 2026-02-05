#!/usr/bin/env python3
"""
Efficiency Calibration Workflow

Runs efficiency measurements for all supported models on the current hardware
and aggregates results into calibration curves.

Usage:
    # Run full calibration on CPU (i7-12700K)
    ./cli/calibrate_efficiency.py --hardware i7-12700K --device cpu

    # Run full calibration on Jetson GPU
    ./cli/calibrate_efficiency.py --hardware jetson-orin-agx --device cuda --thermal-profile 50W

    # Run quick calibration (fewer models, fewer runs)
    ./cli/calibrate_efficiency.py --hardware i7-12700K --device cpu --quick

    # Run specific models only
    ./cli/calibrate_efficiency.py --hardware i7-12700K --device cpu --models resnet18,vgg16

    # Resume from where you left off (skip existing measurements)
    ./cli/calibrate_efficiency.py --hardware i7-12700K --device cpu --resume

Directory Structure:
    calibration_data/
      <hardware>/
        measurements/
          resnet18.json
          resnet50.json
          ...
        efficiency_curves.json   (aggregated after all measurements)
        calibration_report.md    (summary report)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


# ============================================================================
# Model Configuration
# ============================================================================

# Full model set for comprehensive calibration
# Organized by category for balanced coverage of operation types
CALIBRATION_MODELS = {
    # ResNet family - standard Conv2D+BN+ReLU patterns
    "resnet": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        # "resnet152",  # Very large, optional
    ],

    # MobileNet family - depthwise separable, inverted residuals
    "mobilenet": [
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
    ],

    # EfficientNet family - MBConv blocks, squeeze-excite
    "efficientnet": [
        "efficientnet_b0",
        "efficientnet_b1",
        # "efficientnet_b2",  # Similar to B1, optional
    ],

    # VGG family - large standard convolutions
    "vgg": [
        "vgg11",
        "vgg16",
        # "vgg19",  # Similar to VGG16, optional
    ],

    # Vision Transformers - MatMul heavy, attention
    "vit": [
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        # "vit_l_32",  # Similar to L/16, optional
        # "vit_h_14",  # Very large, optional
    ],

    # Other architectures
    "other": [
        "maxvit_t",
        # "deeplabv3_resnet50",  # Segmentation, different patterns
    ],
}

# Quick calibration set (subset for fast testing)
QUICK_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "efficientnet_b0",
    "vgg16",
    "vit_b_16",
]

# Hardware configurations
HARDWARE_CONFIGS = {
    # Intel CPUs
    "i7-12700K": {
        "device": "cpu",
        "hardware_arg": "i7-12700K",
        "thermal_profile": None,
        "description": "Intel Core i7-12700K (Alder Lake)",
    },

    # AMD CPUs
    "ryzen-7-7800x3d": {
        "device": "cpu",
        "hardware_arg": "Ryzen",  # Uses generic Ryzen mapper
        "thermal_profile": None,
        "description": "AMD Ryzen 7 7800X3D",
    },
    "ryzen-9-7950x": {
        "device": "cpu",
        "hardware_arg": "Ryzen",
        "thermal_profile": None,
        "description": "AMD Ryzen 9 7950X",
    },

    # Jetson Orin AGX (64GB)
    "jetson-orin-agx-15w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "15W",
        "description": "NVIDIA Jetson AGX Orin 64GB (15W)",
    },
    "jetson-orin-agx-30w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "30W",
        "description": "NVIDIA Jetson AGX Orin 64GB (30W)",
    },
    "jetson-orin-agx-50w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "50W",
        "description": "NVIDIA Jetson AGX Orin 64GB (50W)",
    },
    "jetson-orin-agx-maxn": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "MAXN",
        "description": "NVIDIA Jetson AGX Orin 64GB (MAXN)",
    },

    # Jetson Orin Nano (8GB)
    "jetson-orin-nano-7w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-Nano",
        "thermal_profile": "7W",
        "description": "NVIDIA Jetson Orin Nano 8GB (7W)",
    },
    "jetson-orin-nano-15w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-Nano",
        "thermal_profile": "15W",
        "description": "NVIDIA Jetson Orin Nano 8GB (15W)",
    },

    # Jetson Orin NX (16GB)
    "jetson-orin-nx-10w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-NX",
        "thermal_profile": "10W",
        "description": "NVIDIA Jetson Orin NX 16GB (10W)",
    },
    "jetson-orin-nx-15w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-NX",
        "thermal_profile": "15W",
        "description": "NVIDIA Jetson Orin NX 16GB (15W)",
    },
    "jetson-orin-nx-25w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-NX",
        "thermal_profile": "25W",
        "description": "NVIDIA Jetson Orin NX 16GB (25W)",
    },
}


# ============================================================================
# Calibration Functions
# ============================================================================

def get_all_models() -> List[str]:
    """Get flat list of all calibration models."""
    models = []
    for category_models in CALIBRATION_MODELS.values():
        models.extend(category_models)
    return models


def get_calibration_dir(hardware_id: str) -> Path:
    """Get calibration data directory for hardware."""
    return repo_root / "calibration_data" / hardware_id


def get_measurements_dir(hardware_id: str) -> Path:
    """Get measurements directory for hardware."""
    return get_calibration_dir(hardware_id) / "measurements"


def run_measurement(
    model: str,
    hardware_id: str,
    hardware_config: dict,
    output_path: Path,
    warmup_runs: int = 10,
    timing_runs: int = 50,
    quiet: bool = False
) -> bool:
    """Run efficiency measurement for a single model.

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable,
        str(repo_root / "cli" / "measure_efficiency.py"),
        "--model", model,
        "--hardware", hardware_config["hardware_arg"],
        "--device", hardware_config["device"],
        "--warmup-runs", str(warmup_runs),
        "--timing-runs", str(timing_runs),
        "--output", str(output_path),
    ]

    if hardware_config.get("thermal_profile"):
        cmd.extend(["--thermal-profile", hardware_config["thermal_profile"]])

    if quiet:
        cmd.append("--quiet")

    try:
        result = subprocess.run(cmd, capture_output=quiet, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_aggregation(
    hardware_id: str,
    measurements_dir: Path,
    output_path: Path,
    hardware_config: dict
) -> bool:
    """Run aggregation to create efficiency curves.

    Returns:
        True if successful, False otherwise
    """
    # Map 'cuda' to 'gpu' for device-type
    device_type = "gpu" if hardware_config["device"] == "cuda" else hardware_config["device"]

    cmd = [
        sys.executable,
        str(repo_root / "cli" / "aggregate_efficiency.py"),
        "--input-dir", str(measurements_dir),
        "--output", str(output_path),
        "--hardware", hardware_id,
        "--hardware-name", hardware_config["description"],
        "--device-type", device_type,
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def generate_calibration_report(
    hardware_id: str,
    hardware_config: dict,
    models_measured: List[str],
    models_failed: List[str],
    output_path: Path
):
    """Generate markdown calibration report."""
    report = f"""# Calibration Report: {hardware_config['description']}

## Hardware Configuration

- **Hardware ID**: {hardware_id}
- **Device**: {hardware_config['device']}
- **Thermal Profile**: {hardware_config.get('thermal_profile', 'default')}
- **Calibration Date**: {datetime.now().isoformat()}

## Models Calibrated

| Model | Status |
|-------|--------|
"""
    for model in models_measured:
        report += f"| {model} | OK |\n"
    for model in models_failed:
        report += f"| {model} | FAILED |\n"

    report += f"""
## Summary

- **Total Models**: {len(models_measured) + len(models_failed)}
- **Successful**: {len(models_measured)}
- **Failed**: {len(models_failed)}

## Files Generated

- `measurements/` - Per-model measurement JSON files
- `efficiency_curves.json` - Aggregated efficiency curves
- `calibration_report.md` - This report

## Usage

To use these calibration curves:

```python
# Load calibration data
import json
with open('calibration_data/{hardware_id}/efficiency_curves.json') as f:
    curves = json.load(f)

# Or copy to hardware_registry
cp calibration_data/{hardware_id}/efficiency_curves.json \\
   hardware_registry/{hardware_config['device']}/{hardware_id}/efficiency_curves.json
```
"""

    with open(output_path, 'w') as f:
        f.write(report)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run efficiency calibration workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --hardware i7-12700K --device cpu
  %(prog)s --hardware jetson-orin-agx-50w --device cuda
  %(prog)s --hardware i7-12700K --device cpu --quick
  %(prog)s --hardware i7-12700K --device cpu --models resnet18,vgg16
  %(prog)s --hardware i7-12700K --device cpu --resume
  %(prog)s --list-hardware
  %(prog)s --list-models
""")
    parser.add_argument('--hardware', type=str,
                        help='Hardware identifier (e.g., i7-12700K, jetson-orin-agx-50w)')
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        help='Device for measurement (overrides hardware config)')
    parser.add_argument('--thermal-profile', type=str,
                        help='Thermal/power profile (overrides hardware config)')
    parser.add_argument('--models', type=str,
                        help='Comma-separated list of models to run')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick calibration (fewer models, fewer runs)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip models that already have measurements')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Warmup runs per model (default: 10)')
    parser.add_argument('--timing-runs', type=int, default=50,
                        help='Timing runs per model (default: 50)')
    parser.add_argument('--skip-aggregate', action='store_true',
                        help='Skip aggregation step (measurements only)')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only run aggregation (skip measurements)')
    parser.add_argument('--list-hardware', action='store_true',
                        help='List available hardware configurations')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # List modes
    if args.list_hardware:
        print("\nAvailable Hardware Configurations:")
        print("-" * 70)
        for hw_id, config in sorted(HARDWARE_CONFIGS.items()):
            profile = config.get('thermal_profile') or 'default'
            print(f"  {hw_id:<25} {config['device']:<5} {profile:<8} {config['description']}")
        return 0

    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 40)
        for category, models in CALIBRATION_MODELS.items():
            print(f"\n{category.upper()}:")
            for model in models:
                quick = " (quick)" if model in QUICK_MODELS else ""
                print(f"  - {model}{quick}")
        print(f"\nQuick calibration models: {', '.join(QUICK_MODELS)}")
        return 0

    # Validate hardware
    if not args.hardware:
        print("Error: --hardware is required")
        parser.print_help()
        return 1

    # Get or create hardware config
    if args.hardware in HARDWARE_CONFIGS:
        hardware_config = HARDWARE_CONFIGS[args.hardware].copy()
    else:
        # Create custom config
        if not args.device:
            print(f"Error: Unknown hardware '{args.hardware}'. Specify --device.")
            return 1
        hardware_config = {
            "device": args.device,
            "hardware_arg": args.hardware,
            "thermal_profile": args.thermal_profile,
            "description": f"Custom: {args.hardware}",
        }

    # Override with command-line args
    if args.device:
        hardware_config["device"] = args.device
    if args.thermal_profile:
        hardware_config["thermal_profile"] = args.thermal_profile

    # Determine models to run
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
    elif args.quick:
        models = QUICK_MODELS
    else:
        models = get_all_models()

    # Quick mode also reduces runs
    if args.quick:
        warmup_runs = 5
        timing_runs = 20
    else:
        warmup_runs = args.warmup_runs
        timing_runs = args.timing_runs

    # Setup directories
    calibration_dir = get_calibration_dir(args.hardware)
    measurements_dir = get_measurements_dir(args.hardware)
    measurements_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("  EFFICIENCY CALIBRATION WORKFLOW")
    print("=" * 70)
    print(f"  Hardware:    {args.hardware}")
    print(f"  Description: {hardware_config['description']}")
    print(f"  Device:      {hardware_config['device']}")
    print(f"  Thermal:     {hardware_config.get('thermal_profile', 'default')}")
    print(f"  Models:      {len(models)}")
    print(f"  Runs:        {warmup_runs} warmup + {timing_runs} timed")
    print(f"  Output:      {calibration_dir}")
    print("=" * 70)

    models_measured = []
    models_failed = []
    models_skipped = []

    # Run measurements
    if not args.aggregate_only:
        print(f"\nPhase 1: Running measurements for {len(models)} models...")
        print("-" * 70)

        for i, model in enumerate(models, 1):
            output_path = measurements_dir / f"{model}.json"

            # Check for resume
            if args.resume and output_path.exists():
                print(f"  [{i}/{len(models)}] {model}: skipped (exists)")
                models_skipped.append(model)
                models_measured.append(model)  # Count as measured for aggregation
                continue

            print(f"  [{i}/{len(models)}] {model}...", end=" ", flush=True)

            success = run_measurement(
                model=model,
                hardware_id=args.hardware,
                hardware_config=hardware_config,
                output_path=output_path,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
                quiet=True
            )

            if success:
                print("OK")
                models_measured.append(model)
            else:
                print("FAILED")
                models_failed.append(model)

        print("-" * 70)
        print(f"  Measured: {len(models_measured)}, Skipped: {len(models_skipped)}, Failed: {len(models_failed)}")

    # Run aggregation
    if not args.skip_aggregate:
        print(f"\nPhase 2: Aggregating measurements into efficiency curves...")
        print("-" * 70)

        curves_path = calibration_dir / "efficiency_curves.json"
        success = run_aggregation(
            hardware_id=args.hardware,
            measurements_dir=measurements_dir,
            output_path=curves_path,
            hardware_config=hardware_config
        )

        if success:
            print(f"  Written: {curves_path}")
        else:
            print("  Aggregation FAILED")

    # Generate report
    report_path = calibration_dir / "calibration_report.md"
    generate_calibration_report(
        hardware_id=args.hardware,
        hardware_config=hardware_config,
        models_measured=models_measured,
        models_failed=models_failed,
        output_path=report_path
    )
    print(f"  Report:  {report_path}")

    print()
    print("=" * 70)
    print("  CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"  Calibration data: {calibration_dir}")
    print()
    print("  To install calibration curves:")
    device_type = hardware_config['device']
    print(f"    cp {calibration_dir}/efficiency_curves.json \\")
    print(f"       hardware_registry/{device_type}/{args.hardware}/efficiency_curves.json")
    print()

    return 0 if not models_failed else 1


if __name__ == '__main__':
    sys.exit(main())
