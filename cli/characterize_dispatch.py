#!/usr/bin/env python3
"""
CUDA Kernel Dispatch Latency Characterization

Measures the baseline kernel dispatch overhead on CUDA devices by timing
many minimal kernel launches. This helps understand the fixed overhead
that dominates small operation latency.

Usage:
    ./cli/characterize_dispatch.py
    ./cli/characterize_dispatch.py --iterations 10000
    ./cli/characterize_dispatch.py --output dispatch_stats.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def measure_dispatch_latencies(
    iterations: int = 5000,
    warmup: int = 500,
    kernel_type: str = 'empty'
) -> List[float]:
    """Measure kernel dispatch latencies.

    Args:
        iterations: Number of timed kernel launches
        warmup: Number of warmup iterations (discarded)
        kernel_type: Type of kernel to launch ('empty', 'small_add', 'small_matmul')

    Returns:
        List of latencies in milliseconds
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device('cuda')

    # Create minimal tensors for different kernel types
    if kernel_type == 'empty':
        # Synchronize-only (measures sync overhead)
        tensor = None
    elif kernel_type == 'small_add':
        # Tiny element-wise add (minimal compute)
        tensor = torch.ones(1, device=device)
    elif kernel_type == 'small_matmul':
        # Tiny matmul (1x1)
        tensor = torch.ones(1, 1, device=device)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # Pre-create events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations + warmup)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations + warmup)]

    # Initial sync
    torch.cuda.synchronize()

    # Run all iterations
    for i in range(warmup + iterations):
        start_events[i].record()

        if kernel_type == 'empty':
            # Just record events, no kernel
            pass
        elif kernel_type == 'small_add':
            _ = tensor + tensor
        elif kernel_type == 'small_matmul':
            _ = torch.matmul(tensor, tensor)

        end_events[i].record()

    # Sync and collect times
    torch.cuda.synchronize()

    latencies = []
    for i in range(warmup, warmup + iterations):
        latencies.append(start_events[i].elapsed_time(end_events[i]))

    return latencies


def compute_statistics(latencies: List[float]) -> Dict[str, float]:
    """Compute statistical summary of latencies."""
    if not latencies:
        return {}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    mean = sum(latencies) / n
    variance = sum((x - mean) ** 2 for x in latencies) / n
    std = math.sqrt(variance)

    def percentile(p):
        idx = int(p / 100 * (n - 1))
        return sorted_lat[idx]

    return {
        'count': n,
        'min': sorted_lat[0],
        'max': sorted_lat[-1],
        'mean': mean,
        'std': std,
        'p50': percentile(50),
        'p90': percentile(90),
        'p95': percentile(95),
        'p99': percentile(99),
        'p99_9': percentile(99.9),
    }


def build_histogram(latencies: List[float], bins: int = 20) -> List[Dict[str, Any]]:
    """Build histogram of latencies."""
    if not latencies:
        return []

    min_val = min(latencies)
    max_val = max(latencies)

    # Add small buffer to include max value
    bin_width = (max_val - min_val) / bins
    if bin_width == 0:
        bin_width = 0.001

    counts = [0] * bins
    for lat in latencies:
        bin_idx = min(int((lat - min_val) / bin_width), bins - 1)
        counts[bin_idx] += 1

    histogram = []
    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        count = counts[i]
        pct = count / len(latencies) * 100
        histogram.append({
            'bin_start': bin_start,
            'bin_end': bin_end,
            'count': count,
            'percentage': pct,
        })

    return histogram


def print_histogram(histogram: List[Dict], max_bar_width: int = 50):
    """Print ASCII histogram."""
    if not histogram:
        return

    max_pct = max(h['percentage'] for h in histogram)
    if max_pct == 0:
        max_pct = 1

    print("\nHistogram:")
    for h in histogram:
        bar_len = int(h['percentage'] / max_pct * max_bar_width)
        bar = '#' * bar_len
        print(f"  {h['bin_start']:6.3f}-{h['bin_end']:6.3f} ms: {bar} ({h['percentage']:5.1f}%)")


def print_statistics(stats: Dict[str, float], kernel_type: str):
    """Print statistics summary."""
    print(f"\nStatistics ({kernel_type} kernel):")
    print(f"  Count:  {stats['count']:,}")
    print(f"  Min:    {stats['min']:.4f} ms")
    print(f"  p50:    {stats['p50']:.4f} ms")
    print(f"  p90:    {stats['p90']:.4f} ms")
    print(f"  p95:    {stats['p95']:.4f} ms")
    print(f"  p99:    {stats['p99']:.4f} ms")
    print(f"  p99.9:  {stats['p99_9']:.4f} ms")
    print(f"  Max:    {stats['max']:.4f} ms")
    print(f"  Mean:   {stats['mean']:.4f} ms")
    print(f"  Std:    {stats['std']:.4f} ms")


def main():
    parser = argparse.ArgumentParser(
        description='Characterize CUDA kernel dispatch latency',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --iterations 10000
  %(prog)s --kernel-types empty small_add small_matmul
  %(prog)s --output dispatch_stats.json
""")
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Number of timed iterations (default: 5000)')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Number of warmup iterations (default: 500)')
    parser.add_argument('--kernel-types', nargs='+',
                        default=['empty', 'small_add', 'small_matmul'],
                        choices=['empty', 'small_add', 'small_matmul'],
                        help='Kernel types to test')
    parser.add_argument('--bins', type=int, default=20,
                        help='Number of histogram bins (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress histogram output')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return 1

    device_name = torch.cuda.get_device_name(0)

    print("=" * 70)
    print("  CUDA KERNEL DISPATCH LATENCY CHARACTERIZATION")
    print("=" * 70)
    print(f"  Device:     {device_name}")
    print(f"  Iterations: {args.iterations:,} (+ {args.warmup:,} warmup)")
    print(f"  Kernels:    {', '.join(args.kernel_types)}")
    print("=" * 70)

    results = {
        'device': device_name,
        'timestamp': datetime.now().isoformat(),
        'iterations': args.iterations,
        'warmup': args.warmup,
        'kernel_types': {},
    }

    for kernel_type in args.kernel_types:
        print(f"\nMeasuring {kernel_type} kernel dispatch...", flush=True)

        latencies = measure_dispatch_latencies(
            iterations=args.iterations,
            warmup=args.warmup,
            kernel_type=kernel_type
        )

        stats = compute_statistics(latencies)
        histogram = build_histogram(latencies, bins=args.bins)

        results['kernel_types'][kernel_type] = {
            'statistics': stats,
            'histogram': histogram,
            'raw_latencies': latencies if args.output else None,
        }

        print_statistics(stats, kernel_type)

        if not args.quiet:
            print_histogram(histogram)

    # Summary comparison
    if len(args.kernel_types) > 1:
        print("\n" + "=" * 70)
        print("  SUMMARY COMPARISON")
        print("=" * 70)
        print(f"  {'Kernel Type':<15} {'Min':>10} {'p50':>10} {'p99':>10} {'Max':>10}")
        print("  " + "-" * 55)
        for kt in args.kernel_types:
            s = results['kernel_types'][kt]['statistics']
            print(f"  {kt:<15} {s['min']:>10.4f} {s['p50']:>10.4f} {s['p99']:>10.4f} {s['max']:>10.4f}")

    # Save results
    if args.output:
        # Remove raw latencies if too large
        for kt in results['kernel_types']:
            if results['kernel_types'][kt]['raw_latencies']:
                # Keep only if explicitly saving
                pass

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
