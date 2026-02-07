#!/usr/bin/env python3
"""
Full Hardware Calibration Pipeline

Automates the complete calibration data collection and efficiency curve generation:
1. Detects hardware (GPU model, power mode, clock speeds)
2. Runs micro-kernel benchmarks (BLAS, STREAM)
3. Measures per-subgraph efficiency across multiple models
4. Aggregates measurements into efficiency curves
5. Validates the generated curves

Usage:
    # Full calibration on GPU (recommended - takes 2-4 hours)
    ./cli/run_full_calibration.py --device cuda

    # Quick calibration (fewer models/configs, ~30 min)
    ./cli/run_full_calibration.py --device cuda --quick

    # Specific precisions only
    ./cli/run_full_calibration.py --device cuda --precisions fp32,fp16

    # CPU calibration
    ./cli/run_full_calibration.py --device cpu

    # Dry run (show what would be done)
    ./cli/run_full_calibration.py --device cuda --dry-run

Output:
    calibration_data/<hardware_id>/<precision>/efficiency_curves.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repo to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Models to benchmark for efficiency curve generation
# Format: (model_name, batch_sizes)
FULL_MODEL_CONFIGS = [
    # Standard CNNs
    ("resnet18", [1, 4, 16, 64]),
    ("resnet34", [1, 4, 16]),
    ("resnet50", [1, 4, 16]),
    ("resnet101", [1, 4]),

    # Efficient architectures (depthwise convs)
    ("mobilenet_v2", [1, 4, 16, 64]),
    ("mobilenet_v3_small", [1, 4, 16, 64]),
    ("mobilenet_v3_large", [1, 4, 16]),

    # EfficientNet family
    ("efficientnet_b0", [1, 4, 16]),
    ("efficientnet_b1", [1, 4]),

    # Other architectures
    ("squeezenet1_0", [1, 4, 16, 64]),
    ("densenet121", [1, 4]),
    ("shufflenet_v2_x1_0", [1, 4, 16, 64]),

    # VGG (for large FC layer calibration)
    ("vgg16", [1, 4]),
]

QUICK_MODEL_CONFIGS = [
    ("resnet18", [1, 16]),
    ("mobilenet_v2", [1, 16]),
    ("efficientnet_b0", [1, 4]),
    ("squeezenet1_0", [1, 16]),
]

# Precisions to calibrate
ALL_PRECISIONS = ["fp32", "fp16", "bf16"]
QUICK_PRECISIONS = ["fp32", "fp16"]

# Timing configuration
WARMUP_RUNS = 10
TIMING_RUNS = 50
QUICK_WARMUP = 5
QUICK_TIMING = 20


# =============================================================================
# HARDWARE DETECTION
# =============================================================================

@dataclass
class HardwareInfo:
    """Detected hardware information."""
    device_type: str  # 'cuda' or 'cpu'
    device_name: str
    hardware_id: str  # e.g., 'jetson_orin_agx_50w'
    power_mode: Optional[str] = None
    clock_mhz: Optional[int] = None
    memory_gb: Optional[float] = None
    cuda_cores: Optional[int] = None
    tensor_cores: Optional[int] = None
    compute_capability: Optional[str] = None


def detect_hardware(device: str) -> HardwareInfo:
    """Detect hardware configuration."""
    if device == "cuda":
        return _detect_gpu()
    else:
        return _detect_cpu()


def _detect_gpu() -> HardwareInfo:
    """Detect GPU hardware."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    # Detect power mode for Jetson
    power_mode = None
    try:
        from src.graphs.calibration.gpu_clock import get_jetson_power_mode
        power_mode = get_jetson_power_mode()
        if power_mode and power_mode.upper().startswith('MODE_'):
            power_mode = power_mode[5:]  # Strip MODE_ prefix
    except ImportError:
        pass

    # Build hardware ID
    gpu_lower = gpu_name.lower()
    if 'orin' in gpu_lower:
        if 'agx' in gpu_lower or '64gb' in gpu_lower:
            base_id = "jetson_orin_agx"
        elif 'nx' in gpu_lower:
            base_id = "jetson_orin_nx"
        elif 'nano' in gpu_lower:
            base_id = "jetson_orin_nano"
        else:
            base_id = "jetson_orin"

        if power_mode:
            hardware_id = f"{base_id}_{power_mode.lower()}"
        else:
            hardware_id = base_id
    elif 'h100' in gpu_lower:
        hardware_id = "nvidia_h100"
    elif 'a100' in gpu_lower:
        hardware_id = "nvidia_a100"
    elif 'rtx' in gpu_lower:
        # Extract RTX model
        import re
        match = re.search(r'rtx\s*(\d+)', gpu_lower)
        if match:
            hardware_id = f"nvidia_rtx_{match.group(1)}"
        else:
            hardware_id = "nvidia_rtx"
    else:
        # Generic GPU ID
        hardware_id = gpu_name.lower().replace(' ', '_').replace('-', '_')

    # Estimate CUDA/Tensor cores from compute capability
    major, minor = props.major, props.minor
    cuda_cores = props.multi_processor_count * _cuda_cores_per_sm(major, minor)
    tensor_cores = props.multi_processor_count * _tensor_cores_per_sm(major, minor)

    return HardwareInfo(
        device_type="cuda",
        device_name=gpu_name,
        hardware_id=hardware_id,
        power_mode=power_mode,
        memory_gb=props.total_memory / (1024**3),
        cuda_cores=cuda_cores,
        tensor_cores=tensor_cores,
        compute_capability=f"{major}.{minor}",
    )


def _cuda_cores_per_sm(major: int, minor: int) -> int:
    """Get CUDA cores per SM for compute capability."""
    cores = {
        (7, 0): 64,   # Volta
        (7, 5): 64,   # Turing
        (8, 0): 64,   # Ampere A100
        (8, 6): 128,  # Ampere consumer
        (8, 7): 128,  # Jetson Orin
        (8, 9): 128,  # Ada Lovelace
        (9, 0): 128,  # Hopper
    }
    return cores.get((major, minor), 64)


def _tensor_cores_per_sm(major: int, minor: int) -> int:
    """Get Tensor cores per SM for compute capability."""
    if major < 7:
        return 0
    elif major == 7:
        return 8  # Volta/Turing
    else:
        return 4  # Ampere and later


def _detect_cpu() -> HardwareInfo:
    """Detect CPU hardware."""
    import platform

    cpu_name = platform.processor() or "Unknown CPU"

    # Try to get more detailed info
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        cpu_name = info.get('brand_raw', cpu_name)
    except ImportError:
        pass

    # Build hardware ID
    cpu_lower = cpu_name.lower()
    if 'i7' in cpu_lower or 'i9' in cpu_lower or 'i5' in cpu_lower:
        import re
        match = re.search(r'i[579]-(\d+)', cpu_lower)
        if match:
            hardware_id = f"intel_core_i7_{match.group(1)}"
        else:
            hardware_id = "intel_core"
    elif 'ryzen' in cpu_lower:
        hardware_id = "amd_ryzen"
    elif 'xeon' in cpu_lower:
        hardware_id = "intel_xeon"
    elif 'arm' in cpu_lower or 'aarch64' in cpu_lower:
        hardware_id = "arm_cpu"
    else:
        hardware_id = cpu_name.lower().replace(' ', '_').replace('-', '_')[:30]

    return HardwareInfo(
        device_type="cpu",
        device_name=cpu_name,
        hardware_id=hardware_id,
    )


# =============================================================================
# CALIBRATION PIPELINE
# =============================================================================

@dataclass
class CalibrationRun:
    """Record of a calibration run."""
    hardware_info: HardwareInfo
    precisions: List[str]
    models: List[str]
    start_time: str
    end_time: Optional[str] = None
    measurements_dir: Optional[str] = None
    output_dir: Optional[str] = None
    status: str = "pending"
    errors: List[str] = field(default_factory=list)


def run_full_calibration(
    device: str,
    precisions: List[str],
    model_configs: List[Tuple[str, List[int]]],
    output_base: Path,
    warmup_runs: int = WARMUP_RUNS,
    timing_runs: int = TIMING_RUNS,
    dry_run: bool = False,
    verbose: bool = True,
) -> CalibrationRun:
    """
    Run the full calibration pipeline.

    Steps:
    1. Detect hardware
    2. Create output directories
    3. Run measurements for each model/precision/batch combination
    4. Aggregate measurements into efficiency curves
    5. Validate generated curves
    """
    # Detect hardware
    print("=" * 70)
    print("HARDWARE DETECTION")
    print("=" * 70)
    hw = detect_hardware(device)
    print(f"Device:      {hw.device_name}")
    print(f"Hardware ID: {hw.hardware_id}")
    if hw.power_mode:
        print(f"Power Mode:  {hw.power_mode}")
    if hw.cuda_cores:
        print(f"CUDA Cores:  {hw.cuda_cores}")
    if hw.tensor_cores:
        print(f"Tensor Cores: {hw.tensor_cores}")
    if hw.memory_gb:
        print(f"Memory:      {hw.memory_gb:.1f} GB")
    print()

    # Create run record
    run = CalibrationRun(
        hardware_info=hw,
        precisions=precisions,
        models=[m[0] for m in model_configs],
        start_time=datetime.now().isoformat(),
    )

    # Setup directories
    measurements_dir = output_base / "measurements" / hw.hardware_id
    calibration_dir = output_base / "calibration_data" / hw.hardware_id

    run.measurements_dir = str(measurements_dir)
    run.output_dir = str(calibration_dir)

    if not dry_run:
        measurements_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CALIBRATION PLAN")
    print("=" * 70)
    total_runs = sum(len(batches) for _, batches in model_configs) * len(precisions)
    print(f"Models:      {len(model_configs)}")
    print(f"Precisions:  {', '.join(precisions)}")
    print(f"Total runs:  {total_runs}")
    print(f"Warmup:      {warmup_runs} runs")
    print(f"Timing:      {timing_runs} runs")
    print()
    print(f"Measurements: {measurements_dir}")
    print(f"Output:       {calibration_dir}")
    print()

    if dry_run:
        print("[DRY RUN] Would run the following:")
        for model, batches in model_configs:
            for precision in precisions:
                for batch in batches:
                    print(f"  {model} @ {precision}, batch={batch}")
        return run

    # Run measurements
    print("=" * 70)
    print("RUNNING MEASUREMENTS")
    print("=" * 70)

    run_count = 0
    for model, batches in model_configs:
        for precision in precisions:
            for batch_size in batches:
                run_count += 1
                output_file = measurements_dir / precision / f"{model}_b{batch_size}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n[{run_count}/{total_runs}] {model} @ {precision}, batch={batch_size}")

                success = _run_measurement(
                    model=model,
                    hardware=hw,
                    precision=precision,
                    batch_size=batch_size,
                    output_file=output_file,
                    warmup_runs=warmup_runs,
                    timing_runs=timing_runs,
                    verbose=verbose,
                )

                if not success:
                    run.errors.append(f"{model}@{precision}@b{batch_size}")

    # Aggregate into efficiency curves
    print()
    print("=" * 70)
    print("AGGREGATING EFFICIENCY CURVES")
    print("=" * 70)

    for precision in precisions:
        print(f"\nAggregating {precision}...")
        precision_measurements = measurements_dir / precision
        precision_output = calibration_dir / precision
        precision_output.mkdir(parents=True, exist_ok=True)

        success = _aggregate_measurements(
            input_dir=precision_measurements,
            output_file=precision_output / "efficiency_curves.json",
            hardware_id=hw.hardware_id,
            hardware_name=hw.device_name,
            precision=precision,
            verbose=verbose,
        )

        if not success:
            run.errors.append(f"aggregate_{precision}")

    # Save run metadata
    run.end_time = datetime.now().isoformat()
    run.status = "completed" if not run.errors else "completed_with_errors"

    metadata_file = calibration_dir / "calibration_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(asdict(run), f, indent=2, default=str)

    # Summary
    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Status:       {run.status}")
    print(f"Duration:     {_format_duration(run.start_time, run.end_time)}")
    print(f"Measurements: {measurements_dir}")
    print(f"Curves:       {calibration_dir}")
    if run.errors:
        print(f"Errors:       {len(run.errors)}")
        for err in run.errors[:5]:
            print(f"  - {err}")
        if len(run.errors) > 5:
            print(f"  ... and {len(run.errors) - 5} more")

    return run


def _run_measurement(
    model: str,
    hardware: HardwareInfo,
    precision: str,
    batch_size: int,
    output_file: Path,
    warmup_runs: int,
    timing_runs: int,
    verbose: bool,
) -> bool:
    """Run a single measurement using measure_efficiency.py."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cli" / "measure_efficiency.py"),
        "--model", model,
        "--hardware", hardware.device_name.split()[0],  # First word
        "--id", hardware.hardware_id,
        "--device", hardware.device_type,
        "--precision", precision,
        "--batch-size", str(batch_size),
        "--warmup-runs", str(warmup_runs),
        "--timing-runs", str(timing_runs),
        "--output", str(output_file),
    ]

    if hardware.power_mode:
        cmd.extend(["--thermal-profile", hardware.power_mode])

    if not verbose:
        cmd.append("--quiet")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=600,  # 10 minute timeout per model
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {model}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def _aggregate_measurements(
    input_dir: Path,
    output_file: Path,
    hardware_id: str,
    hardware_name: str,
    precision: str,
    verbose: bool,
) -> bool:
    """Aggregate measurements into efficiency curves."""
    if not input_dir.exists():
        print(f"  No measurements found in {input_dir}")
        return False

    cmd = [
        sys.executable,
        str(REPO_ROOT / "cli" / "aggregate_efficiency.py"),
        "--input-dir", str(input_dir),
        "--output", str(output_file),
        "--hardware-id", hardware_id,
        "--hardware-name", hardware_name,
        "--precision", precision,
    ]

    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"  Created: {output_file}")
            return True
        else:
            print(f"  Failed: {result.stderr if result.stderr else 'Unknown error'}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def _format_duration(start: str, end: str) -> str:
    """Format duration between two ISO timestamps."""
    from datetime import datetime
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    delta = end_dt - start_dt
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_calibration(calibration_dir: Path, verbose: bool = True) -> bool:
    """Validate generated calibration data."""
    print()
    print("=" * 70)
    print("VALIDATING CALIBRATION DATA")
    print("=" * 70)

    all_valid = True

    for precision_dir in calibration_dir.iterdir():
        if not precision_dir.is_dir():
            continue

        curves_file = precision_dir / "efficiency_curves.json"
        if not curves_file.exists():
            print(f"  MISSING: {curves_file}")
            all_valid = False
            continue

        try:
            with open(curves_file) as f:
                data = json.load(f)

            curves = data.get("curves", {})
            print(f"\n  {precision_dir.name}:")
            print(f"    Operations: {len(curves)}")

            for op_type, op_data in curves.items():
                fitted = op_data.get("fitted_curve", {})
                breakpoints = fitted.get("breakpoints", [])
                print(f"      {op_type}: {len(breakpoints)} breakpoints")

                # Validate breakpoints
                if breakpoints:
                    flops_range = (breakpoints[0][0], breakpoints[-1][0])
                    eff_range = (
                        min(bp[1] for bp in breakpoints),
                        max(bp[1] for bp in breakpoints)
                    )
                    print(f"        FLOPs: {flops_range[0]:.0f} - {flops_range[1]:.0f}")
                    print(f"        Efficiency: {eff_range[0]:.4f} - {eff_range[1]:.4f}")

                    # Check for anomalies
                    if eff_range[1] > 1.0:
                        print(f"        WARNING: Efficiency > 100%")
                    if eff_range[0] < 0:
                        print(f"        WARNING: Negative efficiency")

        except Exception as e:
            print(f"  ERROR reading {curves_file}: {e}")
            all_valid = False

    return all_valid


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full Hardware Calibration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full GPU calibration (2-4 hours)
    %(prog)s --device cuda

    # Quick calibration (~30 min)
    %(prog)s --device cuda --quick

    # FP32 only
    %(prog)s --device cuda --precisions fp32

    # CPU calibration
    %(prog)s --device cpu

    # Preview without running
    %(prog)s --device cuda --dry-run

Output Structure:
    calibration_data/
      <hardware_id>/
        fp32/
          efficiency_curves.json
        fp16/
          efficiency_curves.json
        calibration_metadata.json
"""
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to calibrate (default: cuda)"
    )
    parser.add_argument(
        "--precisions",
        default=None,
        help="Comma-separated precisions (default: fp32,fp16,bf16 or fp32,fp16 for quick)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick calibration (fewer models/configs)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT,
        help="Output base directory (default: repo root)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without running"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing calibration data"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.quick:
        model_configs = QUICK_MODEL_CONFIGS
        precisions = args.precisions.split(",") if args.precisions else QUICK_PRECISIONS
        warmup = QUICK_WARMUP
        timing = QUICK_TIMING
    else:
        model_configs = FULL_MODEL_CONFIGS
        precisions = args.precisions.split(",") if args.precisions else ALL_PRECISIONS
        warmup = WARMUP_RUNS
        timing = TIMING_RUNS

    # Check device availability
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("ERROR: CUDA not available. Use --device cpu or check CUDA installation.")
                return 1
        except ImportError:
            print("ERROR: PyTorch not installed. Install with: pip install torch")
            return 1

    # Validate only mode
    if args.validate_only:
        hw = detect_hardware(args.device)
        calibration_dir = args.output / "calibration_data" / hw.hardware_id
        if calibration_dir.exists():
            valid = validate_calibration(calibration_dir, verbose=not args.quiet)
            return 0 if valid else 1
        else:
            print(f"No calibration data found at {calibration_dir}")
            return 1

    # Run calibration
    print()
    print("=" * 70)
    print("FULL HARDWARE CALIBRATION PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    run = run_full_calibration(
        device=args.device,
        precisions=precisions,
        model_configs=model_configs,
        output_base=args.output,
        warmup_runs=warmup,
        timing_runs=timing,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    # Validate if not dry run
    if not args.dry_run and run.output_dir:
        validate_calibration(Path(run.output_dir), verbose=not args.quiet)

    return 0 if run.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
