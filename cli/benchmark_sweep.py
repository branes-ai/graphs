#!/usr/bin/env python
"""
Benchmark Sweep CLI Tool - Self-Organizing Hardware Calibration

Comprehensive benchmarking tool that:
1. Auto-detects hardware and software stack (NO user-provided IDs)
2. Collects empirical performance data across multiple layers
3. Stores results in append-only, versioned database
4. Enables post-silicon dynamics tracking
5. Supports both CPU and CUDA GPU calibration

Usage:
    # Run CPU calibration (hardware auto-detected)
    ./cli/benchmark_sweep.py

    # Run GPU calibration
    ./cli/benchmark_sweep.py --device cuda

    # Quick calibration with force (skip preflight checks)
    ./cli/benchmark_sweep.py --quick --force

    # Run all benchmark layers
    ./cli/benchmark_sweep.py --layers micro,proxy,models

    # Show detected hardware context
    ./cli/benchmark_sweep.py --show-context

    # View database summary
    ./cli/benchmark_sweep.py --summary

    # Query for similar hardware
    ./cli/benchmark_sweep.py --find-similar --target-gops 100 --target-bw 200

    # Check for regressions
    ./cli/benchmark_sweep.py --detect-regressions

    # Show efficiency trajectory for current hardware
    ./cli/benchmark_sweep.py --show-trajectory

Devices:
    cpu:   Calibrate CPU using NumPy/BLAS (default)
    cuda:  Calibrate NVIDIA GPU using PyTorch CUDA

Layers:
    micro:  BLAS + STREAM micro-kernels (10-30 min)
    proxy:  MLP parameter sweep (30-60 min)
    models: Full model benchmarks - ResNet18, MobileNetV2 (30+ min)

Output:
    Results are stored in append-only SQLite database with complete
    provenance (hardware fingerprint, software stack, timestamp).
    Every run creates a NEW record - no updates, no overwrites.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.calibration.auto_detect import (
    CalibrationContext, HardwareIdentity, SoftwareStack, detect_all
)
from graphs.calibration.calibration_db import (
    CalibrationDB, CalibrationRun, RegressionAlert
)


# =============================================================================
# LAYER IMPLEMENTATIONS
# =============================================================================

def run_micro_kernel_layer(
    context: CalibrationContext,
    precisions: List[str],
    device: str,
    quick: bool = False,
    force: bool = False,
    db: Optional[CalibrationDB] = None
) -> List[CalibrationRun]:
    """
    Run micro-kernel benchmarks (BLAS + STREAM).

    This layer measures fundamental hardware characteristics:
    - BLAS Level 1/2/3 performance across sizes
    - STREAM memory bandwidth (copy, scale, add, triad)
    - Multi-precision performance (FP32, FP16, INT8, etc.)

    Returns:
        List of CalibrationRun objects with results.
    """
    from graphs.calibration.calibrator import calibrate_hardware

    print()
    print("=" * 70)
    print("LAYER 1: Micro-Kernel Benchmarks")
    print("=" * 70)
    print()

    hw = context.hardware
    sw = context.software

    # Select hardware target based on device
    if device == 'cuda' and hw.gpu:
        hardware_name = hw.gpu.model
        print(f"GPU: {hw.gpu.model}")
        print(f"  CUDA Cores: {hw.gpu.cuda_cores}, Tensor Cores: {hw.gpu.tensor_cores}")
        print(f"  Compute Capability: {hw.gpu.compute_capability}")
        print(f"  Memory: {hw.gpu.memory_mb} MB")

        # Estimate theoretical peaks from GPU specs
        peak_gflops = hw.gpu.estimate_theoretical_peak_gflops("fp32")
        peak_bandwidth = hw.gpu.estimate_theoretical_bandwidth_gbps()

        # Build per-precision theoretical peaks for GPU
        theoretical_peaks = {}
        for prec in precisions:
            theoretical_peaks[prec] = hw.gpu.estimate_theoretical_peak_gflops(prec)
    else:
        hardware_name = hw.cpu.model
        print(f"CPU: {hw.cpu.model}")

        # Estimate theoretical peaks from CPU specs
        peak_gflops = hw.cpu.estimate_theoretical_peak_gflops("fp32")
        peak_bandwidth = hw.cpu.estimate_theoretical_bandwidth_gbps()

        # Build per-precision theoretical peaks for CPU
        theoretical_peaks = {}
        for prec in precisions:
            theoretical_peaks[prec] = hw.cpu.estimate_theoretical_peak_gflops(prec)

    print(f"HW Fingerprint: {hw.fingerprint}")
    print(f"SW Fingerprint: {sw.fingerprint}")
    print(f"Device: {device.upper()}")
    print(f"Precisions: {', '.join(precisions)}")
    print(f"Theoretical Peak: {peak_gflops:.1f} GFLOPS (FP32), {peak_bandwidth:.1f} GB/s")
    print()

    runs = []

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

        # Extract STREAM results
        stream_results = {
            'copy_gbps': 0.0,
            'scale_gbps': 0.0,
            'add_gbps': 0.0,
            'triad_gbps': 0.0,
            'stream_best_gbps': calibration.measured_bandwidth_gbps,
        }

        for op_key, op_cal in calibration.operation_profiles.items():
            if 'stream' in op_cal.operation_type:
                kernel = op_cal.extra_params.get('kernel', '')
                bw = op_cal.achieved_bandwidth_gbps or 0
                if kernel == 'copy':
                    stream_results['copy_gbps'] = max(stream_results['copy_gbps'], bw)
                elif kernel == 'scale':
                    stream_results['scale_gbps'] = max(stream_results['scale_gbps'], bw)
                elif kernel == 'add':
                    stream_results['add_gbps'] = max(stream_results['add_gbps'], bw)
                elif kernel == 'triad':
                    stream_results['triad_gbps'] = max(stream_results['triad_gbps'], bw)

        # Extract BLAS results per precision
        for precision in precisions:
            blas_results = {
                'blas1_gops': 0.0,
                'blas2_gops': 0.0,
                'blas3_gops': 0.0,
            }

            for op_key, op_cal in calibration.operation_profiles.items():
                if 'blas' in op_cal.operation_type:
                    blas_level = op_cal.extra_params.get('blas_level', 0)

                    # Check precision results
                    if op_cal.precision_results:
                        prec_result = op_cal.precision_results.get(precision)
                        if prec_result and prec_result.supported and prec_result.measured_gops:
                            gops = prec_result.measured_gops
                            if blas_level == 1:
                                blas_results['blas1_gops'] = max(blas_results['blas1_gops'], gops)
                            elif blas_level == 2:
                                blas_results['blas2_gops'] = max(blas_results['blas2_gops'], gops)
                            elif blas_level == 3:
                                blas_results['blas3_gops'] = max(blas_results['blas3_gops'], gops)

            # Create CalibrationRun for this precision
            # Use our estimated theoretical peak for this precision
            prec_theoretical_peak = theoretical_peaks.get(precision, peak_gflops)
            run = CalibrationRun.from_context_and_results(
                context=context,
                precision=precision,
                device=device,
                stream_results=stream_results,
                blas_results=blas_results,
                theoretical_peak=prec_theoretical_peak,
                preflight_passed=calibration.metadata.preflight.passed if calibration.metadata.preflight else True,
                forced=force,
                notes=f"Quick={quick}" if quick else "",
            )

            runs.append(run)

            # Add to database
            if db:
                db.add_run(run)

        print()
        print(f"Micro-kernel layer complete: {len(runs)} calibration runs recorded")

        # Show summary
        for run in runs:
            print(f"  {run.precision}: {run.peak_measured_gops:.1f} GOPS, "
                  f"{run.stream_best_gbps:.1f} GB/s, "
                  f"{run.efficiency*100:.1f}% efficiency")

        return runs

    except Exception as e:
        print(f"[!] Error in micro-kernel layer: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_proxy_workload_layer(
    context: CalibrationContext,
    device: str,
    quick: bool = False,
    db: Optional[CalibrationDB] = None
) -> Dict[str, float]:
    """
    Run proxy workload benchmarks (MLP sweep).

    Returns:
        Dictionary with summary statistics.
    """
    print()
    print("=" * 70)
    print("LAYER 2: Proxy Workload Benchmarks")
    print("=" * 70)
    print()

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "validation" / "empirical"))
        from sweep_mlp import run_sweep, QUICK_SWEEP, FULL_SWEEP

        sweep_params = QUICK_SWEEP if quick else FULL_SWEEP

        output_dir = Path(__file__).parent.parent / "results" / "proxy_workloads"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"mlp_sweep_{context.hardware.fingerprint}_{device}.csv"

        results = run_sweep(
            sweep_params=sweep_params,
            device=device,
            output_file=str(output_file)
        )

        if results:
            errors = [r['time_error_pct'] for r in results]
            summary = {
                'num_configs': len(results),
                'mape': sum(errors) / len(errors),
                'min_error': min(errors),
                'max_error': max(errors),
            }

            print()
            print(f"Proxy workload layer complete:")
            print(f"  Configurations tested: {summary['num_configs']}")
            print(f"  MAPE: {summary['mape']:.1f}%")
            print(f"  Results saved to: {output_file}")

            return summary

    except ImportError as e:
        print(f"[!] Proxy workload layer requires sweep_mlp: {e}")
    except Exception as e:
        print(f"[!] Error in proxy workload layer: {e}")
        import traceback
        traceback.print_exc()

    return {}


def run_full_model_layer(
    context: CalibrationContext,
    device: str,
    batch_sizes: List[int],
    db: Optional[CalibrationDB] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run full model benchmarks (ResNet18, MobileNetV2, etc.).

    Returns:
        Dictionary of {model_name: {batch_size: latency_ms}}.
    """
    import torch

    print()
    print("=" * 70)
    print("LAYER 3: Full Model Benchmarks")
    print("=" * 70)
    print()

    results = {}

    try:
        import torchvision.models as models
    except ImportError:
        print("[!] torchvision not available. Skipping full model layer.")
        return {}

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

                with torch.no_grad():
                    for _ in range(5):
                        _ = model(input_tensor)

                if device == 'cuda':
                    torch.cuda.synchronize()

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
# ANALYSIS FUNCTIONS
# =============================================================================

def show_trajectory(db: CalibrationDB, hardware_fingerprint: str, precision: str = "fp32"):
    """Show efficiency trajectory for hardware."""
    trajectory = db.get_trajectory(hardware_fingerprint, precision)

    if not trajectory:
        print(f"No calibration history for hardware {hardware_fingerprint}")
        return

    print()
    print("=" * 70)
    print(f"EFFICIENCY TRAJECTORY: {hardware_fingerprint[:8]}...")
    print("=" * 70)
    print()

    print(f"{'Date':<20} {'SW Stack':<12} {'GOPS':<12} {'BW (GB/s)':<12} {'Efficiency':<10}")
    print("-" * 70)

    for run in trajectory:
        date = run.timestamp[:10]
        sw = run.software_fingerprint[:8]
        print(f"{date:<20} {sw:<12} {run.peak_measured_gops:>10.1f}  {run.stream_best_gbps:>10.1f}  {run.efficiency*100:>8.1f}%")

    # Show improvement rate
    rate = db.get_improvement_rate(hardware_fingerprint, precision)
    if rate:
        print()
        print(f"Improvement rate: {rate:+.1f}% per month")

    # Show time to 90% efficiency
    time_to_90 = db.get_time_to_milestone(hardware_fingerprint, 0.9, precision)
    if time_to_90:
        print(f"Time to 90% efficiency: {time_to_90.days} days")
    else:
        latest = trajectory[-1] if trajectory else None
        if latest and latest.efficiency < 0.9:
            print(f"Current efficiency: {latest.efficiency*100:.1f}% (90% not yet reached)")


def show_regressions(db: CalibrationDB, threshold: float = 5.0):
    """Show detected regressions."""
    alerts = db.detect_regressions(threshold)

    print()
    print("=" * 70)
    print(f"REGRESSION DETECTION (threshold: {threshold}%)")
    print("=" * 70)
    print()

    if not alerts:
        print("No regressions detected.")
        return

    for alert in alerts:
        print(alert.summary())
        print()


def find_similar_hardware(
    db: CalibrationDB,
    target_gops: float,
    target_bw: float,
    device: str = "cpu",
    n: int = 5
):
    """Find and display similar hardware."""
    similar = db.find_comparable(target_gops, target_bw, device=device, n=n)

    print()
    print("=" * 70)
    print("SIMILAR HARDWARE SEARCH RESULTS")
    print("=" * 70)
    print()
    print(f"Target: {target_gops:.1f} GOPS, {target_bw:.1f} GB/s")
    print()

    if not similar:
        print("No similar hardware found in database.")
        print("Run benchmark sweeps to populate the database.")
        return

    print(f"{'Rank':<6} {'HW Fingerprint':<18} {'CPU Model':<30} {'Sim':<8} {'GOPS':<10} {'BW':<10}")
    print("-" * 90)

    for i, (run, similarity) in enumerate(similar, 1):
        cpu_short = run.cpu_model[:28] if len(run.cpu_model) > 28 else run.cpu_model
        print(f"{i:<6} {run.hardware_fingerprint:<18} {cpu_short:<30} {similarity:>6.3f}  {run.peak_measured_gops:>8.1f}  {run.stream_best_gbps:>8.1f}")


def print_summary_table(db: CalibrationDB):
    """Print summary of calibration database."""
    summary = db.get_summary()

    print()
    print("=" * 70)
    print("CALIBRATION DATABASE SUMMARY")
    print("=" * 70)
    print()
    print(f"Total runs:          {summary['total_runs']}")
    print(f"Unique hardware:     {summary['unique_hardware']}")
    print(f"Unique SW stacks:    {summary['unique_software']}")

    if summary.get('date_range'):
        print(f"Date range:          {summary['date_range'].get('first', 'N/A')[:10]} to {summary['date_range'].get('last', 'N/A')[:10]}")

    print()

    if summary['by_device']:
        print("By device:")
        for device, count in sorted(summary['by_device'].items()):
            print(f"  {device:15s} {count:5d}")
        print()

    if summary['by_precision']:
        print("By precision:")
        for prec, count in sorted(summary['by_precision'].items()):
            print(f"  {prec:15s} {count:5d}")
        print()

    if summary['by_cpu_vendor']:
        print("By CPU vendor:")
        for vendor, count in sorted(summary['by_cpu_vendor'].items()):
            print(f"  {vendor:15s} {count:5d}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Sweep CLI - Self-Organizing Hardware Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Benchmark options (NO --hardware-id!)
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cpu',
                       help="Device type (default: cpu)")
    parser.add_argument("--layers", type=str, default="micro",
                       help="Comma-separated layers to run: micro,proxy,models (default: micro)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks (fewer sizes/configs)")
    parser.add_argument("--force", action="store_true",
                       help="Force benchmarks even if preflight checks fail")
    parser.add_argument("--precisions", type=str, default="fp32",
                       help="Comma-separated precisions to test (default: fp32)")
    parser.add_argument("--batch-sizes", type=str, default="1,8,64",
                       help="Comma-separated batch sizes for full model layer")

    # Database options
    parser.add_argument("--db", type=str, default="calibrations_v2.db",
                       help="SQLite database path (default: calibrations_v2.db)")

    # Query options
    parser.add_argument("--summary", action="store_true",
                       help="Show database summary")
    parser.add_argument("--show-trajectory", action="store_true",
                       help="Show efficiency trajectory for current hardware")
    parser.add_argument("--detect-regressions", action="store_true",
                       help="Detect performance regressions")
    parser.add_argument("--regression-threshold", type=float, default=5.0,
                       help="Regression threshold percentage (default: 5.0)")

    # Similarity search
    parser.add_argument("--find-similar", action="store_true",
                       help="Find similar hardware (requires --target-gops and --target-bw)")
    parser.add_argument("--target-gops", type=float, default=0.0,
                       help="Target GOPS for similarity search")
    parser.add_argument("--target-bw", type=float, default=0.0,
                       help="Target bandwidth (GB/s) for similarity search")
    parser.add_argument("--n-similar", type=int, default=5,
                       help="Number of similar hardware to return")

    # Export options
    parser.add_argument("--export-json", type=str,
                       help="Export database to JSON file")
    parser.add_argument("--export-parquet", type=str,
                       help="Export database to Parquet file")

    # Info
    parser.add_argument("--show-context", action="store_true",
                       help="Show auto-detected hardware/software context and exit")

    args = parser.parse_args()

    # Database setup
    db_dir = Path(__file__).parent.parent / "results" / "calibration_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / args.db

    # Auto-detect context FIRST (before any operations)
    print("Auto-detecting hardware and software stack...")
    context = detect_all()

    # Show context and exit if requested
    if args.show_context:
        print(context.summary())
        return 0

    print(f"Hardware fingerprint: {context.hardware.fingerprint}")
    print(f"Software fingerprint: {context.software.fingerprint}")
    print(f"Database: {db_path}")

    # Open database
    db = CalibrationDB(str(db_path))

    # Handle query-only operations
    if args.summary:
        print_summary_table(db)
        return 0

    if args.show_trajectory:
        show_trajectory(db, context.hardware.fingerprint)
        return 0

    if args.detect_regressions:
        show_regressions(db, args.regression_threshold)
        return 0

    if args.find_similar:
        if args.target_gops <= 0 or args.target_bw <= 0:
            print("Error: --find-similar requires --target-gops and --target-bw")
            return 1
        find_similar_hardware(db, args.target_gops, args.target_bw, args.device, args.n_similar)
        return 0

    if args.export_json:
        if db.export_json(args.export_json):
            print(f"Exported to: {args.export_json}")
        return 0

    if args.export_parquet:
        if db.export_parquet(args.export_parquet):
            print(f"Exported to: {args.export_parquet}")
        return 0

    # Parse benchmark options
    layers = [l.strip() for l in args.layers.split(',')]
    precisions = [p.strip() for p in args.precisions.split(',')]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]

    # Print benchmark configuration
    print()
    print("=" * 70)
    print("BENCHMARK SWEEP")
    print("=" * 70)
    print()
    print(f"Run ID:      {context.run_id}")
    print(f"Timestamp:   {context.timestamp.isoformat()}")

    # Show target hardware based on device
    if args.device == 'cuda' and context.hardware.gpu:
        gpu = context.hardware.gpu
        print(f"GPU:         {gpu.model}")
        print(f"  CUDA Cores: {gpu.cuda_cores}, Tensor Cores: {gpu.tensor_cores}")
        print(f"  Memory:    {gpu.memory_mb} MB, Compute: {gpu.compute_capability}")
    else:
        print(f"CPU:         {context.hardware.cpu.model}")

    print(f"HW Finger:   {context.hardware.fingerprint}")
    print(f"SW Finger:   {context.software.fingerprint}")
    print(f"Device:      {args.device}")
    print(f"Layers:      {', '.join(layers)}")
    print(f"Precisions:  {', '.join(precisions)}")
    print(f"Quick mode:  {args.quick}")
    print(f"Force mode:  {args.force}")
    print("=" * 70)

    start_time = time.time()

    # Run requested layers
    if 'micro' in layers:
        run_micro_kernel_layer(
            context,
            precisions,
            args.device,
            args.quick,
            args.force,
            db
        )

    if 'proxy' in layers:
        run_proxy_workload_layer(
            context,
            args.device,
            args.quick,
            db
        )

    if 'models' in layers:
        run_full_model_layer(
            context,
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
    print(f"  ./cli/benchmark_sweep.py --summary")
    print()
    print(f"  # Show efficiency trajectory for this hardware")
    print(f"  ./cli/benchmark_sweep.py --show-trajectory")
    print()
    print(f"  # Find similar hardware for a target spec")
    print(f"  ./cli/benchmark_sweep.py --find-similar --target-gops 100 --target-bw 200")
    print()
    print(f"  # Detect regressions")
    print(f"  ./cli/benchmark_sweep.py --detect-regressions")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
