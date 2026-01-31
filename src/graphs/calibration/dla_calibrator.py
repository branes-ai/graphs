"""
DLA Calibrator

Orchestrates DLA benchmarking via TensorRT, collecting synthetic layer
and reference model performance data for each DLA core and precision.

Results are saved as timestamped JSON files in the calibration profiles
directory, following the same conventions as GPU/CPU calibration.
"""

import sys
import platform
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from .schema import (
    DLACalibrationData,
    DLALayerBenchmark,
    DLAModelBenchmark,
)


def _detect_power_mode():
    """Detect Jetson nvpmodel power mode (if available)."""
    import subprocess
    try:
        output = subprocess.check_output(
            ['sudo', 'nvpmodel', '-q'], stderr=subprocess.DEVNULL, text=True
        )
        mode_id = None
        mode_name = None
        for line in output.splitlines():
            if 'NV Power Mode' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    mode_name = parts[-1].strip()
            if 'POWER_MODEL_ID' in line:
                try:
                    mode_id = int(line.split('=')[-1].strip())
                except ValueError:
                    pass
            # Fallback: bare number line
            stripped = line.strip()
            if mode_id is None and stripped.isdigit():
                mode_id = int(stripped)
        return mode_id, mode_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _detect_clock_policy():
    """Detect if clocks are locked or using DVFS."""
    try:
        gov_path = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
        with open(gov_path) as f:
            governor = f.read().strip()
        if governor == 'performance':
            return 'locked'
    except (IOError, OSError):
        pass
    return 'dvfs'


def _get_hardware_name():
    """Detect Jetson hardware name from /proc/device-tree/model."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return f.read().strip().rstrip('\x00')
    except (IOError, OSError):
        return platform.node()


def calibrate_dla(
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    run_synthetic: bool = True,
    run_models: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    output_dir: Optional[Path] = None,
) -> DLACalibrationData:
    """
    Run DLA calibration benchmarks and save results.

    Args:
        dla_core: DLA core to benchmark (0 or 1).
        precision: "fp16" or "int8".
        gpu_fallback: Allow GPU fallback for unsupported layers.
        run_synthetic: Run synthetic single-layer benchmarks.
        run_models: Run reference model benchmarks.
        batch_size: Inference batch size.
        warmup: Warmup iterations per benchmark.
        iterations: Timed iterations per benchmark.
        output_dir: Directory to save results (auto-detected if None).

    Returns:
        DLACalibrationData with all benchmark results.
    """
    from ..benchmarks.tensorrt_benchmarks.trt_utils import (
        check_trt_available, get_dla_core_count, TRT_VERSION,
    )

    check_trt_available()

    num_cores = get_dla_core_count()
    if num_cores == 0:
        raise RuntimeError("No DLA cores found on this platform")
    if dla_core >= num_cores:
        raise RuntimeError(
            f"DLA core {dla_core} requested but only {num_cores} available"
        )

    # Detect system state
    mode_id, mode_name = _detect_power_mode()
    clock_policy = _detect_clock_policy()
    hardware_name = _get_hardware_name()
    timestamp = datetime.now(timezone.utc)

    print(f"\n{'=' * 70}")
    print(f"  DLA Calibration")
    print(f"  Hardware:   {hardware_name}")
    print(f"  DLA Core:   {dla_core} (of {num_cores})")
    print(f"  Precision:  {precision.upper()}")
    print(f"  Fallback:   {'GPU' if gpu_fallback else 'None (strict DLA)'}")
    if mode_name:
        print(f"  Power Mode: {mode_name} (ID: {mode_id})")
    print(f"  Clocks:     {clock_policy}")
    print(f"  TensorRT:   {TRT_VERSION}")
    print(f"{'=' * 70}\n")

    # Initialize calibration data
    cal_data = DLACalibrationData(
        hardware_name=hardware_name,
        calibration_date=timestamp.isoformat(),
        dla_core=dla_core,
        precision=precision,
        framework="tensorrt",
        tensorrt_version=TRT_VERSION,
        gpu_fallback=gpu_fallback,
        python_version=platform.python_version(),
        platform_architecture=platform.machine(),
        nvpmodel_mode=mode_id,
        power_mode_name=mode_name,
        clock_policy=clock_policy,
    )

    total = 0
    success = 0
    failed = 0

    # Run synthetic layer benchmarks
    if run_synthetic:
        print("Running synthetic layer benchmarks...")
        from ..benchmarks.tensorrt_benchmarks.dla_synthetic import run_all_synthetic

        synthetic_results = run_all_synthetic(
            dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, batch_size=batch_size,
            warmup=warmup, iterations=iterations,
        )

        for layer_type, results in synthetic_results.items():
            for r in results:
                total += 1
                lb = DLALayerBenchmark(
                    layer_type=r['layer_type'],
                    config=r['config'],
                    params=r['params'],
                    input_shape=r['input_shape'],
                    batch_size=r['batch_size'],
                    flops=r['flops'],
                    precision=r['precision'],
                    dla_core=r['dla_core'],
                    status=r['status'],
                    latency_ms=r.get('latency_ms'),
                    mean_ms=r.get('mean_ms'),
                    min_ms=r.get('min_ms'),
                    max_ms=r.get('max_ms'),
                    std_ms=r.get('std_ms'),
                    tflops=r.get('tflops'),
                    on_dla=r.get('on_dla', False),
                    dla_layer_count=r.get('dla_layer_count', 0),
                    gpu_layer_count=r.get('gpu_layer_count', 0),
                    error=r.get('error'),
                )
                cal_data.layer_benchmarks.append(lb)
                if r['status'] == 'success':
                    success += 1
                    status_str = f"{r['latency_ms']:.3f}ms"
                    if r.get('on_dla'):
                        status_str += " [DLA]"
                    else:
                        status_str += " [GPU fallback]"
                    print(f"    {r['config']:<35} {status_str}")
                else:
                    failed += 1
                    print(f"    {r['config']:<35} FAILED: {r.get('error', '?')}")

    # Run reference model benchmarks
    if run_models:
        print("\nRunning reference model benchmarks...")
        from ..benchmarks.tensorrt_benchmarks.dla_models import benchmark_all_models

        model_results = benchmark_all_models(
            dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, batch_size=batch_size,
            warmup=warmup, iterations=iterations,
        )

        for r in model_results:
            total += 1
            mb = DLAModelBenchmark(
                model=r['model'],
                precision=r['precision'],
                dla_core=r['dla_core'],
                gpu_fallback=r.get('gpu_fallback', gpu_fallback),
                batch_size=r['batch_size'],
                input_shape=r['input_shape'],
                approx_flops=r['approx_flops'],
                status=r['status'],
                latency_ms=r.get('latency_ms'),
                mean_ms=r.get('mean_ms'),
                min_ms=r.get('min_ms'),
                max_ms=r.get('max_ms'),
                std_ms=r.get('std_ms'),
                throughput_fps=r.get('throughput_fps'),
                tflops=r.get('tflops'),
                total_layers=r.get('total_layers', 0),
                dla_layer_count=r.get('dla_layer_count', 0),
                gpu_layer_count=r.get('gpu_layer_count', 0),
                dla_percentage=r.get('dla_percentage', 0),
                error=r.get('error'),
            )
            cal_data.model_benchmarks.append(mb)
            if r['status'] == 'success':
                success += 1
            else:
                failed += 1

    cal_data.total_benchmarks = total
    cal_data.successful_benchmarks = success
    cal_data.failed_benchmarks = failed

    # Save results
    if output_dir is None:
        output_dir = _get_default_output_dir(hardware_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    ts_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    power_tag = mode_name.lower().replace(' ', '_') if mode_name else "unknown"
    filename = f"dla{dla_core}_{precision}_{power_tag}_{ts_str}.json"
    output_path = output_dir / filename

    cal_data.save(output_path)

    print(f"\n{'=' * 70}")
    print(f"  DLA Calibration Complete")
    print(f"  Results: {success}/{total} succeeded, {failed} failed")
    print(f"  Saved:   {output_path}")
    print(f"{'=' * 70}\n")

    return cal_data


def _get_default_output_dir(hardware_name: str) -> Path:
    """Get default output directory for DLA calibration profiles.

    Follows the hardware_registry convention:
      hardware_registry/accelerator/nvidia_dla_orin/calibrations/
    """
    # Navigate from src/graphs/calibration/ up to repo root, then into hardware_registry
    repo_root = Path(__file__).parent.parent.parent.parent
    return repo_root / 'hardware_registry' / 'accelerator' / 'nvidia_dla_orin' / 'calibrations'


def calibrate_all_dla_cores(
    precision: str = "fp16",
    gpu_fallback: bool = True,
    run_synthetic: bool = True,
    run_models: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    output_dir: Optional[Path] = None,
) -> List[DLACalibrationData]:
    """
    Run calibration on all available DLA cores.

    Returns list of DLACalibrationData, one per core.
    """
    from ..benchmarks.tensorrt_benchmarks.trt_utils import (
        check_trt_available, get_dla_core_count,
    )

    check_trt_available()
    num_cores = get_dla_core_count()

    if num_cores == 0:
        raise RuntimeError("No DLA cores found on this platform")

    print(f"Found {num_cores} DLA core(s)")
    results = []
    for core in range(num_cores):
        result = calibrate_dla(
            dla_core=core, precision=precision,
            gpu_fallback=gpu_fallback,
            run_synthetic=run_synthetic, run_models=run_models,
            batch_size=batch_size, warmup=warmup, iterations=iterations,
            output_dir=output_dir,
        )
        results.append(result)

    return results
