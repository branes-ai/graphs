#!/usr/bin/env python
"""
Benchmark Sweep CLI Tool

Comprehensive benchmarking tool that collects empirical performance data
across three layers (micro-kernels, proxy workloads, full models) and stores
results in a normalized SQLite database for agentic tool queries.

Usage:
    # Quick micro-kernel sweep (10 minutes)
    ./cli/benchmark_sweep.py --layers micro --quick

    # Full sweep with all layers (2+ hours)
    ./cli/benchmark_sweep.py --layers micro,proxy,models

    # GPU calibration with multiple precisions
    ./cli/benchmark_sweep.py --hardware-id h100_sxm5 --precisions fp32,fp16,int8

    # Import existing calibrations into database
    ./cli/benchmark_sweep.py --import-registry

    # Query for similar hardware
    ./cli/benchmark_sweep.py --find-similar --target-gops 100 --target-bw 200

Layers:
    micro:  BLAS + STREAM micro-kernels (10-30 min)
    proxy:  MLP parameter sweep (30-60 min)
    models: Full model benchmarks - ResNet18, MobileNetV2 (30+ min)

Output:
    Results are stored in SQLite database (default: calibrations.db)
    with normalized schema for fast queries and similarity matching.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.calibration.calibration_db import (
    CalibrationDB, CalibrationPoint, HypotheticalArchitecture
)


# =============================================================================
# LAYER IMPLEMENTATIONS
# =============================================================================

def run_micro_kernel_layer(
    hardware_id: str,
    precisions: List[str],
    device: str,
    quick: bool = False,
    force: bool = False,
    db: Optional[CalibrationDB] = None
) -> List[CalibrationPoint]:
    """
    Run micro-kernel benchmarks (BLAS + STREAM).

    This layer measures fundamental hardware characteristics:
    - BLAS Level 1/2/3 performance across sizes
    - STREAM memory bandwidth (copy, scale, add, triad)
    - Multi-precision performance (FP32, FP16, INT8, etc.)

    Returns:
        List of CalibrationPoint objects with results.
    """
    from graphs.hardware.calibration.calibrator import calibrate_hardware, select_framework

    print()
    print("=" * 80)
    print("LAYER 1: Micro-Kernel Benchmarks")
    print("=" * 80)
    print()

    # Use hardware_id as the name (spec lookup disabled due to schema issues)
    # TODO: Re-enable when hardware_registry schema is updated
    hardware_spec = None
    hardware_name = hardware_id
    vendor = _detect_vendor(hardware_id)
    theoretical_peaks = {}
    peak_bandwidth = 100.0  # Default estimate (will be measured)
    peak_gflops = 100.0     # Default estimate (will be measured)

    print(f"Hardware: {hardware_name}" + (f" ({vendor})" if vendor != "Unknown" else ""))
    print(f"Device:   {device.upper()}")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"Note: Using measured peaks (empirical calibration)")
    print()

    # Run calibration
    try:
        calibration = calibrate_hardware(
            hardware_name=hardware_name,
            theoretical_peak_gflops=peak_gflops,
            theoretical_bandwidth_gbps=peak_bandwidth,
            theoretical_peaks=theoretical_peaks,
            device=device,
            operations=['blas', 'stream'],
            quick=quick,
            skip_preflight=force,
        )

        # Convert to CalibrationPoints
        points = []
        device_type = (hardware_spec.device_type if hardware_spec else None) or ('gpu' if device == 'cuda' else 'cpu')

        # Extract BLAS results
        for op_key, op_cal in calibration.operation_profiles.items():
            if 'blas' in op_cal.operation_type:
                if op_cal.precision_results:
                    for prec, result in op_cal.precision_results.items():
                        if result.supported and result.measured_gops:
                            point = CalibrationPoint(
                                hardware_id=hardware_id,
                                vendor=vendor,
                                architecture=_detect_architecture(hardware_id),
                                device_type=device_type,
                                precision=prec,
                                power_mode=_get_power_mode(calibration),
                                clock_mhz=_get_clock_mhz(calibration),
                                gemm_peak_gops=result.measured_gops,
                                gemm_efficiency=result.efficiency or 0.0,
                                bandwidth_gbps=calibration.measured_bandwidth_gbps,
                                bandwidth_efficiency=calibration.bandwidth_efficiency,
                                theoretical_peak_gops=theoretical_peaks.get(prec, peak_gflops),
                                theoretical_bandwidth_gbps=peak_bandwidth,
                                arithmetic_intensity_transition=(
                                    theoretical_peaks.get(prec, peak_gflops) / peak_bandwidth
                                    if peak_bandwidth > 0 else 0.0
                                ),
                                blas1_gops=_get_blas_level_gops(calibration, 1, prec),
                                blas2_gops=_get_blas_level_gops(calibration, 2, prec),
                                blas3_gops=_get_blas_level_gops(calibration, 3, prec),
                                calibration_date=calibration.metadata.calibration_date,
                                framework=calibration.metadata.framework,
                                framework_version=(
                                    calibration.metadata.numpy_version or
                                    calibration.metadata.pytorch_version or ""
                                ),
                            )
                            points.append(point)

                            # Add to database
                            if db:
                                db.add_calibration(point)

        print()
        print(f"Micro-kernel layer complete: {len(points)} calibration points collected")
        return points

    except Exception as e:
        print(f"[!] Error in micro-kernel layer: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_proxy_workload_layer(
    hardware_id: str,
    device: str,
    quick: bool = False,
    db: Optional[CalibrationDB] = None
) -> Dict[str, float]:
    """
    Run proxy workload benchmarks (MLP sweep).

    This layer measures real-world performance on small neural networks
    with varying configurations to capture:
    - Batch size scaling
    - Memory hierarchy effects (L1/L2/L3/DRAM)
    - Efficiency factor calibration

    Returns:
        Dictionary with summary statistics.
    """
    print()
    print("=" * 80)
    print("LAYER 2: Proxy Workload Benchmarks")
    print("=" * 80)
    print()

    try:
        # Import MLP sweep functionality
        sys.path.insert(0, str(Path(__file__).parent.parent / "validation" / "empirical"))
        from sweep_mlp import run_sweep, QUICK_SWEEP, FULL_SWEEP

        sweep_params = QUICK_SWEEP if quick else FULL_SWEEP

        # Run the sweep
        output_dir = Path(__file__).parent.parent / "results" / "proxy_workloads"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"mlp_sweep_{hardware_id}_{device}.csv"

        results = run_sweep(
            sweep_params=sweep_params,
            device=device,
            output_file=str(output_file)
        )

        if results:
            # Compute summary statistics
            errors = [r['time_error_pct'] for r in results]
            empirical_times = [r['empirical_time_ms'] for r in results]
            analytical_times = [r['analytical_time_ms'] for r in results]

            summary = {
                'num_configs': len(results),
                'mape': sum(errors) / len(errors),
                'min_error': min(errors),
                'max_error': max(errors),
                'mean_empirical_ms': sum(empirical_times) / len(empirical_times),
                'mean_analytical_ms': sum(analytical_times) / len(analytical_times),
            }

            print()
            print(f"Proxy workload layer complete:")
            print(f"  Configurations tested: {summary['num_configs']}")
            print(f"  MAPE: {summary['mape']:.1f}%")
            print(f"  Results saved to: {output_file}")

            return summary

    except ImportError as e:
        print(f"[!] Proxy workload layer requires sweep_mlp: {e}")
        print("    Run from repo root or check PYTHONPATH")
    except Exception as e:
        print(f"[!] Error in proxy workload layer: {e}")
        import traceback
        traceback.print_exc()

    return {}


def run_full_model_layer(
    hardware_id: str,
    device: str,
    batch_sizes: List[int],
    db: Optional[CalibrationDB] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run full model benchmarks (ResNet18, MobileNetV2, etc.).

    This layer measures end-to-end performance on production models
    to validate analytical estimates against real execution.

    Returns:
        Dictionary of {model_name: {batch_size: latency_ms}}.
    """
    import torch
    import torch.nn as nn

    print()
    print("=" * 80)
    print("LAYER 3: Full Model Benchmarks")
    print("=" * 80)
    print()

    results = {}

    # Check for torchvision
    try:
        import torchvision.models as models
        TORCHVISION_AVAILABLE = True
    except ImportError:
        print("[!] torchvision not available. Skipping full model layer.")
        return {}

    # Models to benchmark
    model_configs = [
        ('resnet18', lambda: models.resnet18(weights=None), (3, 224, 224)),
        ('mobilenet_v2', lambda: models.mobilenet_v2(weights=None), (3, 224, 224)),
    ]

    for model_name, model_fn, input_shape in model_configs:
        print(f"\nBenchmarking {model_name}...")
        results[model_name] = {}

        try:
            model = model_fn()
            model = model.to(device)
            model.eval()

            for batch_size in batch_sizes:
                input_tensor = torch.randn(batch_size, *input_shape).to(device)

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(input_tensor)

                if device == 'cuda':
                    torch.cuda.synchronize()

                # Benchmark
                times = []
                with torch.no_grad():
                    for _ in range(20):
                        start = time.perf_counter()
                        _ = model(input_tensor)
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        end = time.perf_counter()
                        times.append((end - start) * 1000)

                mean_time = sum(times) / len(times)
                results[model_name][batch_size] = mean_time
                throughput = batch_size / (mean_time / 1000)

                print(f"  Batch {batch_size:3d}: {mean_time:8.2f} ms ({throughput:.1f} samples/sec)")

        except Exception as e:
            print(f"  [!] Error benchmarking {model_name}: {e}")
            continue

    print()
    print(f"Full model layer complete: {len(results)} models benchmarked")
    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_vendor(hardware_id: str) -> str:
    """Detect vendor from hardware ID."""
    id_lower = hardware_id.lower()
    if 'intel' in id_lower or 'i7' in id_lower or 'i9' in id_lower or 'xeon' in id_lower:
        return 'Intel'
    elif 'amd' in id_lower or 'ryzen' in id_lower or 'epyc' in id_lower:
        return 'AMD'
    elif 'nvidia' in id_lower or 'h100' in id_lower or 'a100' in id_lower or 'jetson' in id_lower:
        return 'NVIDIA'
    elif 'ampere' in id_lower:
        return 'Ampere Computing'
    elif 'qualcomm' in id_lower:
        return 'Qualcomm'
    elif 'arm' in id_lower or 'mali' in id_lower:
        return 'ARM'
    elif 'google' in id_lower or 'tpu' in id_lower:
        return 'Google'
    elif 'hailo' in id_lower:
        return 'Hailo'
    return 'Unknown'


def _detect_architecture(hardware_id: str) -> str:
    """Detect architecture from hardware ID."""
    id_lower = hardware_id.lower()
    if 'h100' in id_lower:
        return 'hopper'
    elif 'a100' in id_lower:
        return 'ampere'
    elif 'v100' in id_lower:
        return 'volta'
    elif 'jetson' in id_lower and 'orin' in id_lower:
        return 'ampere'
    elif '12700' in id_lower or '12th' in id_lower:
        return 'alder_lake'
    elif 'ryzen' in id_lower:
        return 'zen4' if '8' in id_lower else 'zen3'
    elif 'xeon' in id_lower:
        return 'sapphire_rapids'
    return 'unknown'


def _get_power_mode(calibration) -> str:
    """Extract power mode from calibration."""
    if calibration.metadata.gpu_clock and calibration.metadata.gpu_clock.power_mode_name:
        return calibration.metadata.gpu_clock.power_mode_name
    elif calibration.metadata.cpu_clock and calibration.metadata.cpu_clock.governor:
        return calibration.metadata.cpu_clock.governor
    return "default"


def _get_clock_mhz(calibration) -> int:
    """Extract clock MHz from calibration."""
    if calibration.metadata.gpu_clock and calibration.metadata.gpu_clock.sm_clock_mhz:
        return calibration.metadata.gpu_clock.sm_clock_mhz
    elif calibration.metadata.cpu_clock and calibration.metadata.cpu_clock.current_freq_mhz:
        return int(calibration.metadata.cpu_clock.current_freq_mhz)
    return 0


def _get_blas_level_gops(calibration, level: int, precision: str) -> float:
    """Extract BLAS level GOPS from calibration."""
    best_gops = 0.0
    level_ops = {
        1: ['dot', 'axpy'],
        2: ['gemv'],
        3: ['gemm'],
    }

    for op_key, op_cal in calibration.operation_profiles.items():
        blas_level = op_cal.extra_params.get('blas_level', 0)
        if blas_level == level:
            if op_cal.precision_results:
                result = op_cal.precision_results.get(precision)
                if result and result.supported and result.measured_gops:
                    best_gops = max(best_gops, result.measured_gops)

    return best_gops


def print_summary_table(db: CalibrationDB):
    """Print summary of calibration database."""
    summary = db.get_summary()

    print()
    print("=" * 80)
    print("CALIBRATION DATABASE SUMMARY")
    print("=" * 80)
    print()
    print(f"Total calibrations:  {summary['total_calibrations']}")
    print(f"Unique hardware:     {summary['unique_hardware']}")
    print()

    if summary['by_device_type']:
        print("By device type:")
        for device_type, count in sorted(summary['by_device_type'].items()):
            print(f"  {device_type:15s} {count:5d}")
        print()

    if summary['by_precision']:
        print("By precision:")
        for prec, count in sorted(summary['by_precision'].items()):
            print(f"  {prec:15s} {count:5d}")
        print()

    if summary['by_vendor']:
        print("By vendor:")
        for vendor, count in sorted(summary['by_vendor'].items()):
            print(f"  {vendor:15s} {count:5d}")


def find_similar_hardware(
    db: CalibrationDB,
    target_gops: float,
    target_bw: float,
    target_tdp: float = 0.0,
    device_type: Optional[str] = None,
    n: int = 5
):
    """Find and display similar hardware."""
    target = HypotheticalArchitecture(
        peak_gops=target_gops,
        bandwidth_gbps=target_bw,
        tdp_watts=target_tdp,
        device_type=device_type or "",
    )

    similar = db.find_comparable(target, n=n)

    print()
    print("=" * 80)
    print("SIMILAR HARDWARE SEARCH RESULTS")
    print("=" * 80)
    print()
    print(f"Target: {target_gops:.1f} GOPS, {target_bw:.1f} GB/s" +
          (f", {target_tdp:.0f}W" if target_tdp > 0 else ""))
    print()

    if not similar:
        print("No similar hardware found in database.")
        print("Run benchmark sweeps to populate the database.")
        return

    print(f"{'Rank':<6} {'Hardware ID':<35} {'Similarity':<12} {'GOPS':<10} {'BW (GB/s)':<10} {'Eff':<8}")
    print("-" * 85)

    for i, (cal, similarity) in enumerate(similar, 1):
        print(f"{i:<6} {cal.hardware_id:<35} {similarity:>10.3f}  {cal.gemm_peak_gops:>8.1f}  {cal.bandwidth_gbps:>8.1f}  {cal.gemm_efficiency:>6.1%}")

    # Show efficiency estimate
    mean_eff, std_eff, sources = db.get_efficiency_estimate(target, n_samples=n)
    print()
    print(f"Estimated efficiency factor: {mean_eff:.2f} +/- {std_eff:.2f}")
    print(f"Based on: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Sweep CLI - Collect and query hardware calibration data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Hardware specification
    parser.add_argument("--hardware-id", type=str,
                       help="Hardware ID from database (e.g., intel_12th_gen_intelr_coretm_i7_12700k)")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cpu',
                       help="Device type (default: cpu)")

    # Benchmark layers
    parser.add_argument("--layers", type=str, default="micro",
                       help="Comma-separated layers to run: micro,proxy,models (default: micro)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks (fewer sizes/configs)")
    parser.add_argument("--precisions", type=str, default="fp32",
                       help="Comma-separated precisions to test (default: fp32)")
    parser.add_argument("--batch-sizes", type=str, default="1,8,64",
                       help="Comma-separated batch sizes for full model layer")
    parser.add_argument("--force", action="store_true",
                       help="Force benchmarks even if pre-flight checks fail")

    # Database options
    parser.add_argument("--db", type=str, default="calibrations.db",
                       help="SQLite database path (default: calibrations.db)")
    parser.add_argument("--import-registry", action="store_true",
                       help="Import existing calibrations from hardware_registry/")

    # Query options
    parser.add_argument("--find-similar", action="store_true",
                       help="Find similar hardware (requires --target-gops and --target-bw)")
    parser.add_argument("--target-gops", type=float, default=0.0,
                       help="Target GOPS for similarity search")
    parser.add_argument("--target-bw", type=float, default=0.0,
                       help="Target bandwidth (GB/s) for similarity search")
    parser.add_argument("--target-tdp", type=float, default=0.0,
                       help="Target TDP (watts) for similarity search")
    parser.add_argument("--target-device-type", type=str,
                       help="Filter similarity search by device type")
    parser.add_argument("--n-similar", type=int, default=5,
                       help="Number of similar hardware to return")

    # Export options
    parser.add_argument("--export-json", type=str,
                       help="Export database to JSON file")
    parser.add_argument("--export-parquet", type=str,
                       help="Export database to Parquet file")
    parser.add_argument("--summary", action="store_true",
                       help="Show database summary")

    args = parser.parse_args()

    # Determine database path
    db_dir = Path(__file__).parent.parent / "results" / "calibration_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / args.db

    print(f"Calibration database: {db_path}")

    # Open database
    db = CalibrationDB(str(db_path))

    # Handle import from hardware_registry
    if args.import_registry:
        registry_path = Path(__file__).parent.parent / "hardware_registry"
        print(f"Importing from: {registry_path}")
        imported = db.import_from_hardware_registry(registry_path)
        print(f"Imported {imported} calibration points")
        print_summary_table(db)
        return 0

    # Handle similarity search
    if args.find_similar:
        if args.target_gops <= 0 or args.target_bw <= 0:
            print("Error: --find-similar requires --target-gops and --target-bw")
            return 1
        find_similar_hardware(
            db,
            args.target_gops,
            args.target_bw,
            args.target_tdp,
            args.target_device_type,
            args.n_similar
        )
        return 0

    # Handle exports
    if args.export_json:
        if db.export_json(args.export_json):
            print(f"Exported to: {args.export_json}")
        return 0

    if args.export_parquet:
        if db.export_parquet(args.export_parquet):
            print(f"Exported to: {args.export_parquet}")
        return 0

    # Handle summary
    if args.summary:
        print_summary_table(db)
        return 0

    # Require hardware-id for benchmark runs
    if not args.hardware_id:
        # Try to generate a hardware ID from system info
        try:
            import platform
            import psutil

            # Generate a descriptive hardware ID
            cpu_model = platform.processor() or "unknown_cpu"
            # Clean up the model name to make a valid ID
            args.hardware_id = cpu_model.lower().replace(' ', '_').replace('(r)', '').replace('(tm)', '')
            args.hardware_id = ''.join(c for c in args.hardware_id if c.isalnum() or c == '_')

            print(f"Auto-generated hardware ID: {args.hardware_id}")
            print(f"(Use --hardware-id for a specific name)")

        except Exception as e:
            print("Error: --hardware-id required")
            print(f"Could not auto-detect: {e}")
            print("Example: --hardware-id my_test_system")
            return 1

    # Parse options
    layers = [l.strip() for l in args.layers.split(',')]
    precisions = [p.strip() for p in args.precisions.split(',')]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]

    print()
    print("=" * 80)
    print("BENCHMARK SWEEP")
    print("=" * 80)
    print(f"Hardware:    {args.hardware_id}")
    print(f"Device:      {args.device}")
    print(f"Layers:      {', '.join(layers)}")
    print(f"Precisions:  {', '.join(precisions)}")
    print(f"Quick mode:  {args.quick}")
    print("=" * 80)

    start_time = time.time()

    # Run requested layers
    if 'micro' in layers:
        run_micro_kernel_layer(
            args.hardware_id,
            precisions,
            args.device,
            args.quick,
            args.force,
            db
        )

    if 'proxy' in layers:
        run_proxy_workload_layer(
            args.hardware_id,
            args.device,
            args.quick,
            db
        )

    if 'models' in layers:
        run_full_model_layer(
            args.hardware_id,
            args.device,
            batch_sizes,
            db
        )

    elapsed = time.time() - start_time

    # Print final summary
    print_summary_table(db)

    print()
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Results stored in: {db_path}")
    print()
    print("Next steps:")
    print(f"  # View summary")
    print(f"  ./cli/benchmark_sweep.py --db {args.db} --summary")
    print()
    print(f"  # Find similar hardware for a new target")
    print(f"  ./cli/benchmark_sweep.py --db {args.db} --find-similar --target-gops 100 --target-bw 200")
    print()
    print(f"  # Export to JSON for analysis")
    print(f"  ./cli/benchmark_sweep.py --db {args.db} --export-json calibrations.json")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
