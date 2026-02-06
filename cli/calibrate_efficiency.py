#!/usr/bin/env python3
"""
Efficiency Calibration Workflow

Runs efficiency measurements for all supported models on the current hardware
and aggregates results into calibration curves.

Usage:
    # Auto-detect hardware and run calibration (recommended)
    ./cli/calibrate_efficiency.py

    # Auto-detect but force CPU (even if GPU available)
    ./cli/calibrate_efficiency.py --device cpu

    # Use pre-configured hardware profile
    ./cli/calibrate_efficiency.py --id jetson_orin_agx_50w

    # Custom hardware ID with explicit mapper
    ./cli/calibrate_efficiency.py --id my_custom_cpu --hardware Ryzen --device cpu

    # Quick calibration (fewer models, fewer runs)
    ./cli/calibrate_efficiency.py --quick

    # Run specific models only
    ./cli/calibrate_efficiency.py --models resnet18,vgg16

    # Resume from where you left off (skip existing measurements)
    ./cli/calibrate_efficiency.py --resume

Directory Structure:
    calibration_data/
      <hardware_id>/
        measurements/
          resnet18.json
          resnet50.json
          ...
        efficiency_curves.json   (aggregated after all measurements)
        calibration_report.md    (summary report)
"""

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch


# ============================================================================
# Hardware Auto-Detection
# ============================================================================

def sanitize_hardware_id(name: str) -> str:
    """Convert hardware name to valid ID (lowercase, underscores)."""
    # Replace spaces, dashes, special chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', name)
    # Remove leading/trailing underscores, collapse multiple
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized.lower()


def detect_cpu_info() -> Tuple[str, str]:
    """Detect CPU and return (hardware_id, mapper_name).

    Returns:
        Tuple of (sanitized_id, mapper_name)
    """
    cpu_name = platform.processor()

    # Try to get more detailed info
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        cpu_name = info.get('brand_raw', cpu_name)
    except ImportError:
        # Fallback to platform
        if not cpu_name or cpu_name == 'x86_64':
            # Try /proc/cpuinfo on Linux
            try:
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_name = line.split(':')[1].strip()
                            break
            except:
                pass

    # Determine mapper based on CPU name
    cpu_lower = cpu_name.lower()

    if 'intel' in cpu_lower:
        if 'i7-12700' in cpu_lower:
            mapper = 'i7-12700K'
        elif 'xeon' in cpu_lower:
            mapper = 'Xeon'
        else:
            mapper = 'i7-12700K'  # Default Intel mapper
    elif 'amd' in cpu_lower or 'ryzen' in cpu_lower:
        if 'epyc' in cpu_lower:
            mapper = 'EPYC'
        else:
            mapper = 'Ryzen'
    elif 'ampere' in cpu_lower:
        mapper = 'Ampere-One'
    else:
        mapper = 'i7-12700K'  # Generic fallback

    hardware_id = sanitize_hardware_id(cpu_name)
    return hardware_id, mapper


def detect_gpu_info() -> Tuple[str, str]:
    """Detect GPU and return (hardware_id, mapper_name).

    Returns:
        Tuple of (sanitized_id, mapper_name) or (None, None) if no GPU
    """
    if not torch.cuda.is_available():
        return None, None

    gpu_name = torch.cuda.get_device_name(0)
    gpu_lower = gpu_name.lower()

    # Determine mapper based on GPU name
    if 'h100' in gpu_lower:
        mapper = 'H100'
    elif 'a100' in gpu_lower:
        mapper = 'A100'
    elif 'v100' in gpu_lower:
        mapper = 'V100'
    elif 'orin' in gpu_lower:
        if 'agx' in gpu_lower:
            mapper = 'Jetson-Orin-AGX'
        elif 'nx' in gpu_lower:
            mapper = 'Jetson-Orin-NX'
        elif 'nano' in gpu_lower:
            mapper = 'Jetson-Orin-Nano'
        else:
            mapper = 'Jetson-Orin-AGX'  # Default Orin
    elif 'jetson' in gpu_lower:
        mapper = 'Jetson-Orin-AGX'  # Default Jetson
    else:
        mapper = 'A100'  # Generic GPU fallback

    hardware_id = sanitize_hardware_id(gpu_name)
    return hardware_id, mapper


def find_existing_calibration_id(detected_id: str, mapper_name: str, device: str) -> Optional[str]:
    """Check if there's an existing calibration ID that matches the detected hardware.

    Args:
        detected_id: Auto-generated verbose ID (e.g., "12th_gen_intel_r_core_tm_i7_12700k")
        mapper_name: Hardware mapper name (e.g., "i7-12700K", "Ryzen", "Jetson-Orin-AGX")
        device: Device type ("cpu" or "cuda")

    Returns:
        Existing calibration ID if found, None otherwise
    """
    calibration_dir = repo_root / "calibration_data"
    if not calibration_dir.exists():
        return None

    # Get existing calibration IDs
    existing_ids = [d.name for d in calibration_dir.iterdir() if d.is_dir()]

    # Also check HARDWARE_CONFIGS keys
    existing_ids.extend(HARDWARE_CONFIGS.keys())
    existing_ids = list(set(existing_ids))  # Remove duplicates

    # Extract key identifiers from detected hardware for matching
    detected_lower = detected_id.lower()

    # Define patterns to extract from detected ID and match against existing
    # Format: (pattern_in_detected, pattern_to_find_in_existing)
    match_patterns = []

    # Intel patterns
    if 'i7_12700' in detected_lower or 'i7-12700' in mapper_name.lower():
        match_patterns.append('i7_12700')
    if 'i9_' in detected_lower:
        match_patterns.append('i9_')
    if 'xeon' in detected_lower:
        match_patterns.append('xeon')

    # AMD patterns
    if 'ryzen' in detected_lower:
        # Extract model number like "ryzen_7_8845" or "ryzen_9_7950"
        match = re.search(r'ryzen_(\d+_\d+)', detected_lower)
        if match:
            match_patterns.append(f'ryzen_{match.group(1)}')
        match_patterns.append('ryzen')
    if 'epyc' in detected_lower:
        match_patterns.append('epyc')

    # Jetson patterns
    if 'orin' in detected_lower:
        if 'agx' in detected_lower:
            match_patterns.append('orin_agx')
        elif 'nx' in detected_lower:
            match_patterns.append('orin_nx')
        elif 'nano' in detected_lower:
            match_patterns.append('orin_nano')

    # Try to find a matching existing ID
    for existing_id in existing_ids:
        existing_lower = existing_id.lower()
        for pattern in match_patterns:
            if pattern in existing_lower:
                # Found a match - prefer shorter/cleaner IDs
                return existing_id

    return None


def auto_detect_hardware(prefer_gpu: bool = True) -> dict:
    """Auto-detect hardware and return configuration.

    Args:
        prefer_gpu: If True and GPU available, use GPU config

    Returns:
        Hardware config dict with device, hardware_arg, description, hardware_id
    """
    gpu_id, gpu_mapper = detect_gpu_info()
    cpu_id, cpu_mapper = detect_cpu_info()

    if prefer_gpu and gpu_id:
        device = "cuda"
        detected_id = gpu_id
        mapper = gpu_mapper
        description = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        detected_id = cpu_id
        mapper = cpu_mapper
        try:
            import cpuinfo
            description = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
        except:
            description = platform.processor()

    # Check for existing calibration ID that matches this hardware
    existing_id = find_existing_calibration_id(detected_id, mapper, device)
    if existing_id:
        hardware_id = existing_id
    else:
        hardware_id = detected_id

    return {
        "device": device,
        "hardware_arg": mapper,
        "thermal_profile": None,
        "description": f"Auto-detected: {description}",
        "hardware_id": hardware_id,
    }


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
# IDs use underscores to match hardware_registry/ naming convention
HARDWARE_CONFIGS = {
    # Intel CPUs
    "i7_12700K": {
        "device": "cpu",
        "hardware_arg": "i7-12700K",
        "thermal_profile": None,
        "description": "Intel Core i7-12700K (Alder Lake)",
    },

    # AMD CPUs
    "ryzen_7_7800x3d": {
        "device": "cpu",
        "hardware_arg": "Ryzen",  # Uses generic Ryzen mapper
        "thermal_profile": None,
        "description": "AMD Ryzen 7 7800X3D",
    },
    "ryzen_9_7950x": {
        "device": "cpu",
        "hardware_arg": "Ryzen",
        "thermal_profile": None,
        "description": "AMD Ryzen 9 7950X",
    },

    # Jetson Orin AGX (64GB)
    "jetson_orin_agx_15w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "15W",
        "description": "NVIDIA Jetson AGX Orin 64GB (15W)",
    },
    "jetson_orin_agx_30w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "30W",
        "description": "NVIDIA Jetson AGX Orin 64GB (30W)",
    },
    "jetson_orin_agx_50w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "50W",
        "description": "NVIDIA Jetson AGX Orin 64GB (50W)",
    },
    "jetson_orin_agx_maxn": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-AGX",
        "thermal_profile": "MAXN",
        "description": "NVIDIA Jetson AGX Orin 64GB (MAXN)",
    },

    # Jetson Orin Nano (8GB)
    "jetson_orin_nano_7w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-Nano",
        "thermal_profile": "7W",
        "description": "NVIDIA Jetson Orin Nano 8GB (7W)",
    },
    "jetson_orin_nano_15w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-Nano",
        "thermal_profile": "15W",
        "description": "NVIDIA Jetson Orin Nano 8GB (15W)",
    },

    # Jetson Orin NX (16GB)
    "jetson_orin_nx_10w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-NX",
        "thermal_profile": "10W",
        "description": "NVIDIA Jetson Orin NX 16GB (10W)",
    },
    "jetson_orin_nx_15w": {
        "device": "cuda",
        "hardware_arg": "Jetson-Orin-NX",
        "thermal_profile": "15W",
        "description": "NVIDIA Jetson Orin NX 16GB (15W)",
    },
    "jetson_orin_nx_25w": {
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


def get_calibration_dir(hardware_id: str, precision: str = "fp32") -> Path:
    """Get calibration data directory for hardware and precision."""
    return repo_root / "calibration_data" / hardware_id / precision


def get_measurements_dir(hardware_id: str, precision: str = "fp32") -> Path:
    """Get measurements directory for hardware and precision."""
    return get_calibration_dir(hardware_id, precision) / "measurements"


def run_measurement(
    model: str,
    hardware_id: str,
    hardware_config: dict,
    output_path: Path,
    precision: str = "fp32",
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
        "--id", hardware_id,  # Calibration ID for aggregation filtering
        "--device", hardware_config["device"],
        "--precision", precision,
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
    hardware_config: dict,
    precision: str = "fp32"
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
        "--precision", precision,
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
    output_path: Path,
    precision: str = "fp32"
):
    """Generate markdown calibration report."""
    report = f"""# Calibration Report: {hardware_config['description']}

## Hardware Configuration

- **Hardware ID**: {hardware_id}
- **Device**: {hardware_config['device']}
- **Precision**: {precision.upper()}
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
with open('calibration_data/{hardware_id}/{precision}/efficiency_curves.json') as f:
    curves = json.load(f)

# Or copy to hardware_registry
cp calibration_data/{hardware_id}/{precision}/efficiency_curves.json \\
   hardware_registry/{hardware_config['device']}/{hardware_id}/{precision}/efficiency_curves.json
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
  # Auto-detect hardware (recommended)
  %(prog)s                              # Detect and calibrate
  %(prog)s --device cpu                 # Force CPU even if GPU available
  %(prog)s --quick                      # Quick calibration (fewer models)

  # Pre-configured hardware profiles
  %(prog)s --id jetson_orin_agx_50w
  %(prog)s --id jetson_orin_agx_50w --precision fp16

  # Custom hardware ID
  %(prog)s --id my_ryzen --hardware Ryzen --device cpu

  # Other options
  %(prog)s --models resnet18,vgg16      # Specific models only
  %(prog)s --resume                     # Skip existing measurements
  %(prog)s --list-hardware              # Show pre-configured hardware
  %(prog)s --list-models                # Show available models
""")
    parser.add_argument('--id', type=str,
                        help='Hardware identifier (e.g., i7_12700K, jetson_orin_agx_50w)')
    parser.add_argument('--hardware', type=str,
                        help='Hardware mapper name for custom --id (e.g., Ryzen, i7-12700K)')
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
    parser.add_argument('--precision',
                        choices=['fp32', 'fp16', 'bf16', 'tf32', 'int8'],
                        default='fp32',
                        help='Precision: fp32, fp16, bf16, tf32, int8 (default: fp32)')
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

    # Auto-detect or validate hardware
    auto_detected = False
    if not args.id:
        # Auto-detect hardware
        prefer_gpu = args.device != 'cpu'  # Prefer GPU unless explicitly --device cpu
        auto_config = auto_detect_hardware(prefer_gpu=prefer_gpu)
        args.id = auto_config['hardware_id']
        auto_detected = True
        print(f"\nAuto-detected hardware: {auto_config['description']}")
        print(f"  Hardware ID: {args.id}")
        print(f"  Mapper: {auto_config['hardware_arg']}")
        print(f"  Device: {auto_config['device']}")

    # Get or create hardware config
    if auto_detected:
        hardware_config = auto_config
    elif args.id in HARDWARE_CONFIGS:
        hardware_config = HARDWARE_CONFIGS[args.id].copy()
    else:
        # Create custom config
        if not args.device:
            print(f"Error: Unknown hardware '{args.id}'. Specify --device.")
            return 1
        if not args.hardware:
            print(f"Error: Unknown hardware '{args.id}'. Specify --hardware mapper name.")
            print("  Examples: --hardware Ryzen, --hardware i7-12700K, --hardware Jetson-Orin-AGX")
            return 1
        hardware_config = {
            "device": args.device,
            "hardware_arg": args.hardware,
            "thermal_profile": args.thermal_profile,
            "description": f"Custom: {args.id}",
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

    # Setup directories (include precision in path)
    precision = args.precision
    calibration_dir = get_calibration_dir(args.id, precision)
    measurements_dir = get_measurements_dir(args.id, precision)
    measurements_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("  EFFICIENCY CALIBRATION WORKFLOW")
    print("=" * 70)
    print(f"  Hardware:    {args.id}")
    print(f"  Description: {hardware_config['description']}")
    print(f"  Device:      {hardware_config['device']}")
    print(f"  Precision:   {precision.upper()}")
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
                hardware_id=args.id,
                hardware_config=hardware_config,
                output_path=output_path,
                precision=precision,
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
            hardware_id=args.id,
            measurements_dir=measurements_dir,
            output_path=curves_path,
            hardware_config=hardware_config,
            precision=precision
        )

        if success:
            print(f"  Written: {curves_path}")
        else:
            print("  Aggregation FAILED")

    # Generate report
    report_path = calibration_dir / "calibration_report.md"
    generate_calibration_report(
        hardware_id=args.id,
        hardware_config=hardware_config,
        models_measured=models_measured,
        models_failed=models_failed,
        output_path=report_path,
        precision=precision
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
    print(f"       hardware_registry/{device_type}/{args.id}/{precision}/efficiency_curves.json")
    print()

    return 0 if not models_failed else 1


if __name__ == '__main__':
    sys.exit(main())
