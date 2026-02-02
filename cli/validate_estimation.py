#!/usr/bin/env python
"""
Estimation Validation Tool

Runs actual model inference on the current hardware, then compares measured
latency/throughput/memory against the roofline estimates from the unified
analyzer. This validates that the estimation model produces accurate
predictions for real workloads.

Usage:
    # Single model validation
    ./cli/validate_estimation.py --model resnet18 --hardware i7-12700K

    # Batch size sweep
    ./cli/validate_estimation.py --model resnet18 --hardware Jetson-Orin-AGX --batch-size 1 4 8 16

    # Multi-model comparison
    ./cli/validate_estimation.py --model resnet18,resnet50,mobilenet_v2 --hardware i7-12700K

    # CUDA inference
    ./cli/validate_estimation.py --model resnet18 --hardware Jetson-Orin-AGX --device cuda

    # CSV output
    ./cli/validate_estimation.py --model resnet18,resnet50 --hardware i7-12700K --output results.csv
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch

from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision


# ============================================================================
# Measurement
# ============================================================================

def get_memory_usage_cpu() -> float:
    """Get current CPU memory usage in MB (RSS)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def measure_inference(model, input_tensor, device, num_warmup=10, num_runs=50):
    """Measure actual inference latency and memory.

    Args:
        model: PyTorch model (already on correct device)
        input_tensor: Input tensor (already on correct device)
        device: 'cpu' or 'cuda'
        num_warmup: Warmup iterations
        num_runs: Measurement iterations

    Returns:
        Dict with mean_ms, median_ms, min_ms, std_ms, peak_memory_mb, throughput_fps
    """
    model.eval()

    if device == 'cuda':
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Baseline memory
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_before = get_memory_usage_cpu()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

    # Memory
    if device == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = max(get_memory_usage_cpu() - mem_before, 0.0)

    batch_size = input_tensor.shape[0]
    mean_ms = statistics.mean(times)

    return {
        'mean_ms': mean_ms,
        'median_ms': statistics.median(times),
        'min_ms': min(times),
        'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
        'peak_memory_mb': peak_memory_mb,
        'throughput_fps': batch_size / (mean_ms / 1000.0) if mean_ms > 0 else 0.0,
    }


# ============================================================================
# Estimation
# ============================================================================

def run_estimation(analyzer, model_name, hardware_name, precision, batch_size):
    """Run the unified analyzer estimation.

    Returns:
        Dict with latency_ms, throughput_fps, peak_memory_mb, or None on failure.
    """
    try:
        result = analyzer.analyze_model(
            model_name=model_name,
            hardware_name=hardware_name,
            precision=precision,
            batch_size=batch_size,
        )
        return {
            'latency_ms': result.total_latency_ms,
            'throughput_fps': result.throughput_fps,
            'peak_memory_mb': result.peak_memory_mb,
        }
    except Exception as e:
        print(f"  Estimation failed: {e}")
        return None


# ============================================================================
# Comparison
# ============================================================================

def compute_error(estimated, measured):
    """Compute percentage error: (estimated - measured) / measured * 100."""
    if measured == 0:
        return float('inf') if estimated != 0 else 0.0
    return (estimated - measured) / measured * 100.0


def quality_rating(latency_error_pct):
    """Rate estimation quality based on absolute latency error."""
    abs_err = abs(latency_error_pct)
    if abs_err < 10:
        return "EXCELLENT"
    elif abs_err < 25:
        return "GOOD"
    elif abs_err < 50:
        return "FAIR"
    else:
        return "POOR"


def print_comparison(model_name, hardware_name, precision_str, batch_size,
                     estimated, measured):
    """Print side-by-side comparison table."""
    print()
    title = f"Estimation Validation: {model_name} on {hardware_name} ({precision_str}, batch={batch_size})"
    print(title)
    print("=" * len(title))
    print()

    header = f"  {'Metric':<22} {'Estimated':>12} {'Measured':>12} {'Error':>10}"
    print(header)
    print("  " + "-" * 58)

    lat_err = compute_error(estimated['latency_ms'], measured['median_ms'])
    thr_err = compute_error(estimated['throughput_fps'], measured['throughput_fps'])
    mem_err = compute_error(estimated['peak_memory_mb'], measured['peak_memory_mb'])

    print(f"  {'Latency (ms)':<22} {estimated['latency_ms']:>12.2f} {measured['median_ms']:>12.2f} {lat_err:>+9.1f}%")
    print(f"  {'Throughput (fps)':<22} {estimated['throughput_fps']:>12.1f} {measured['throughput_fps']:>12.1f} {thr_err:>+9.1f}%")

    if measured['peak_memory_mb'] > 0:
        print(f"  {'Peak Memory (MB)':<22} {estimated['peak_memory_mb']:>12.1f} {measured['peak_memory_mb']:>12.1f} {mem_err:>+9.1f}%")
    else:
        print(f"  {'Peak Memory (MB)':<22} {estimated['peak_memory_mb']:>12.1f} {'N/A':>12}")

    print()
    rating = quality_rating(lat_err)
    print(f"  Estimation quality: {rating} (latency error {lat_err:+.1f}%)")

    # Measurement stats
    print()
    print(f"  Measurement stats: median={measured['median_ms']:.2f} ms, "
          f"mean={measured['mean_ms']:.2f} ms, min={measured['min_ms']:.2f} ms, "
          f"std={measured['std_ms']:.2f} ms")

    return {
        'model': model_name,
        'hardware': hardware_name,
        'precision': precision_str,
        'batch_size': batch_size,
        'est_latency_ms': estimated['latency_ms'],
        'est_throughput_fps': estimated['throughput_fps'],
        'est_memory_mb': estimated['peak_memory_mb'],
        'meas_latency_ms': measured['median_ms'],
        'meas_mean_ms': measured['mean_ms'],
        'meas_min_ms': measured['min_ms'],
        'meas_std_ms': measured['std_ms'],
        'meas_throughput_fps': measured['throughput_fps'],
        'meas_memory_mb': measured['peak_memory_mb'],
        'latency_error_pct': lat_err,
        'throughput_error_pct': thr_err,
        'memory_error_pct': mem_err if measured['peak_memory_mb'] > 0 else None,
        'quality': rating,
    }


# ============================================================================
# Summary table for multi-model/batch sweeps
# ============================================================================

def print_summary_table(rows):
    """Print a summary table across all runs."""
    if len(rows) <= 1:
        return

    print()
    print()
    print("Summary")
    print("=" * 90)
    print(f"  {'Model':<20} {'Batch':>5} {'Est (ms)':>10} {'Meas (ms)':>10} {'Error':>8} {'Quality':<10}")
    print("  " + "-" * 70)

    for r in rows:
        print(f"  {r['model']:<20} {r['batch_size']:>5} "
              f"{r['est_latency_ms']:>10.2f} {r['meas_latency_ms']:>10.2f} "
              f"{r['latency_error_pct']:>+7.1f}% {r['quality']:<10}")

    # Aggregate stats
    errors = [abs(r['latency_error_pct']) for r in rows]
    print("  " + "-" * 70)
    print(f"  {'MAPE':<20} {'':>5} {'':>10} {'':>10} {statistics.mean(errors):>7.1f}%")
    print()


def write_csv(rows, output_path):
    """Write results to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

HARDWARE_CHOICES = [
    'H100', 'A100', 'V100', 'Jetson-Orin-AGX', 'Jetson-Orin-Nano',
    'TPU-v4', 'Coral-Edge-TPU', 'KPU-T768', 'KPU-T256', 'KPU-T64',
    'EPYC', 'Xeon', 'Ampere-One', 'i7-12700K', 'Ryzen',
    'QRB5165', 'TI-TDA4VM', 'DPU', 'CGRA',
]

# Hardware names that imply CUDA device
GPU_HARDWARE = {
    'h100', 'a100', 'v100',
    'jetson-orin-agx', 'jetson-orin-nano',
}


def main():
    parser = argparse.ArgumentParser(
        description='Validate estimation accuracy by comparing against actual inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model resnet18 --hardware i7-12700K
  %(prog)s --model resnet18,resnet50 --hardware Jetson-Orin-AGX --device cuda
  %(prog)s --model resnet18 --hardware i7-12700K --batch-size 1 4 8 16
  %(prog)s --model resnet18,mobilenet_v2 --hardware i7-12700K --output results.csv
""")
    parser.add_argument('--model', required=True,
                        help='Model name(s), comma-separated (e.g., resnet18,resnet50,mobilenet_v2)')
    parser.add_argument('--hardware', required=True, choices=HARDWARE_CHOICES,
                        help='Target hardware for estimation')
    parser.add_argument('--batch-size', type=int, nargs='+', default=[1],
                        help='Batch size(s) to test (default: 1)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device for measurement (default: auto from hardware)')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32',
                        help='Precision (default: fp32)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Measurement iterations (default: 50)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Parse models
    model_names = [m.strip() for m in args.model.split(',')]

    # Auto-detect device
    device = args.device
    if device is None:
        if args.hardware.lower() in GPU_HARDWARE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cpu'

    # Precision
    precision_map = {'fp32': Precision.FP32, 'fp16': Precision.FP16}
    precision = precision_map[args.precision]

    print()
    print("=" * 70)
    print("  ESTIMATION VALIDATION")
    print("=" * 70)
    print(f"  Hardware (estimation): {args.hardware}")
    print(f"  Device (measurement):  {device}")
    print(f"  Precision:             {args.precision.upper()}")
    print(f"  Models:                {', '.join(model_names)}")
    print(f"  Batch sizes:           {args.batch_size}")
    print(f"  Warmup / iterations:   {args.warmup} / {args.iterations}")
    print("=" * 70)

    analyzer = UnifiedAnalyzer(verbose=not args.quiet)
    all_rows = []

    for model_name in model_names:
        for batch_size in args.batch_size:
            if not args.quiet:
                print(f"\n--- {model_name}, batch={batch_size} ---")

            # Step 1: Estimation
            if not args.quiet:
                print(f"  Running estimation...", flush=True)
            estimated = run_estimation(analyzer, model_name, args.hardware,
                                       precision, batch_size)
            if estimated is None:
                print(f"  Skipping {model_name} batch={batch_size} (estimation failed)")
                continue

            # Step 2: Create model and measure
            if not args.quiet:
                print(f"  Running inference ({args.iterations} iterations on {device})...",
                      flush=True)
            try:
                model, input_tensor, display_name = analyzer._create_model(model_name, batch_size)
            except ValueError as e:
                print(f"  Skipping {model_name}: {e}")
                continue

            model.eval()
            if args.precision == 'fp16':
                model = model.half()
                input_tensor = input_tensor.half()

            measured = measure_inference(
                model, input_tensor, device,
                num_warmup=args.warmup, num_runs=args.iterations,
            )

            # Step 3: Compare
            row = print_comparison(
                display_name, args.hardware, args.precision.upper(),
                batch_size, estimated, measured,
            )
            all_rows.append(row)

            # Free memory between models
            del model, input_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Summary
    print_summary_table(all_rows)

    # CSV output
    if args.output and all_rows:
        write_csv(all_rows, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
