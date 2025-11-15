"""
Matrix multiplication calibration benchmark.

This benchmark measures real matmul performance using both:
1. NumPy/BLAS (upper bound - heavily optimized)
2. Naive implementation (lower bound - baseline)

The goal is to establish realistic performance expectations for
matrix multiplication operations of various sizes.
"""

import numpy as np
import time
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

from ..schema import OperationCalibration, OperationType


def benchmark_numpy_matmul(N: int, dtype=np.float32, num_trials: int = 10, num_warmup: int = 3) -> Dict:
    """
    Benchmark NumPy matrix multiplication (represents best-case performance).

    Args:
        N: Matrix dimension (N×N @ N×N)
        dtype: Data type
        num_trials: Number of measurement runs
        num_warmup: Number of warmup runs

    Returns:
        Dict with timing and performance metrics
    """
    # Create random matrices
    A = np.random.rand(N, N).astype(dtype)
    B = np.random.rand(N, N).astype(dtype)

    # Warmup
    for _ in range(num_warmup):
        C = A @ B

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    mean_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)

    # Calculate GFLOPS
    flops = 2.0 * N * N * N  # N^3 multiply-add operations
    mean_gflops = flops / (mean_time_ms / 1000.0) / 1e9

    # Memory traffic (for arithmetic intensity calculation)
    bytes_transferred = (2 * N * N + N * N) * dtype().itemsize  # Read A, B, write C
    arithmetic_intensity = flops / bytes_transferred

    return {
        'N': N,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'max_latency_ms': max_time_ms,
        'gflops': mean_gflops,
        'arithmetic_intensity': arithmetic_intensity,
        'num_trials': num_trials,
    }


def benchmark_cpp_matmul(N: int, num_trials: int = 10) -> Dict:
    """
    Benchmark C++ matmul implementation (if available).

    Attempts to run the C++ benchmark from src/matmul if it exists.

    Returns:
        Dict with performance metrics, or None if C++ benchmark not available
    """
    # Path to C++ benchmark (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    cpp_benchmark = project_root / "src" / "matmul" / "build" / "matmul_benchmark_v2"

    if not cpp_benchmark.exists():
        return None

    try:
        # Run C++ benchmark
        result = subprocess.run(
            [str(cpp_benchmark), "--size", str(N)],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse output to extract GFLOPS
        # Format: "Throughput: XXX.XX GFLOPS"
        for line in result.stdout.split('\n'):
            if 'Throughput:' in line and 'GFLOPS' in line:
                parts = line.split()
                gflops_idx = parts.index('GFLOPS') - 1
                gflops = float(parts[gflops_idx])

                # Find time
                for time_line in result.stdout.split('\n'):
                    if 'Time:' in time_line and 'ms' in time_line:
                        time_parts = time_line.split()
                        ms_idx = time_parts.index('ms') - 1
                        time_ms = float(time_parts[ms_idx])

                        return {
                            'N': N,
                            'mean_latency_ms': time_ms,
                            'gflops': gflops,
                            'source': 'cpp_v2'
                        }

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def calibrate_matmul(
    sizes: List[int] = [512, 1024, 2048, 4096],
    theoretical_peak_gflops: float = 1000.0,
    theoretical_bandwidth_gbps: float = 75.0,
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Calibrate matrix multiplication performance at various sizes.

    Args:
        sizes: List of matrix dimensions to test
        theoretical_peak_gflops: Theoretical peak GFLOPS (for efficiency calculation)
        theoretical_bandwidth_gbps: Theoretical memory bandwidth
        num_trials: Number of trials per size

    Returns:
        List of OperationCalibration objects, one per size
    """
    calibrations = []

    for N in sizes:
        print(f"Calibrating matmul {N}×{N} (FP32)...", end=" ", flush=True)

        # Benchmark with NumPy (best case)
        numpy_result = benchmark_numpy_matmul(N, num_trials=num_trials)

        # Try C++ benchmark (our implementation)
        cpp_result = benchmark_cpp_matmul(N)

        # Use NumPy as primary measurement (represents achievable peak)
        result = numpy_result

        # Determine if memory-bound or compute-bound
        # Rough heuristic: if AI < 10, likely memory-bound
        memory_bound = result['arithmetic_intensity'] < 10.0
        compute_bound = not memory_bound

        # Estimate bandwidth usage
        bytes_per_op = (3 * N * N * 4) / (2 * N * N * N)  # (A+B+C bytes) / ops
        achieved_bandwidth_gbps = (result['gflops'] * 1e9 * bytes_per_op) / 1e9

        # Create calibration object
        calibration = OperationCalibration(
            operation_type=OperationType.MATMUL.value,
            measured_gflops=result['gflops'],
            efficiency=result['gflops'] / theoretical_peak_gflops,
            achieved_bandwidth_gbps=achieved_bandwidth_gbps,
            memory_bound=memory_bound,
            compute_bound=compute_bound,
            arithmetic_intensity=result['arithmetic_intensity'],
            batch_size=1,
            input_shape=(N, N),
            output_shape=(N, N),
            mean_latency_ms=result['mean_latency_ms'],
            std_latency_ms=result['std_latency_ms'],
            min_latency_ms=result['min_latency_ms'],
            max_latency_ms=result['max_latency_ms'],
            num_trials=result['num_trials'],
            extra_params={
                'matrix_size': N,
                'implementation': 'numpy_blas',
                'cpp_gflops': cpp_result['gflops'] if cpp_result else None,
            }
        )

        calibrations.append(calibration)

        print(f"{result['gflops']:.1f} GFLOPS ({calibration.efficiency*100:.1f}% efficiency)")

    return calibrations


if __name__ == "__main__":
    # Standalone test
    print("Matrix Multiplication Calibration")
    print("=" * 60)

    calibrations = calibrate_matmul(
        sizes=[512, 1024, 2048, 4096],
        theoretical_peak_gflops=1000.0
    )

    print("\nResults:")
    print(f"{'Size':<10} {'GFLOPS':>10} {'Efficiency':>12} {'AI':>8} {'Bound':>10}")
    print("-" * 55)

    for cal in calibrations:
        bound = "Memory" if cal.memory_bound else "Compute"
        size = cal.extra_params['matrix_size']
        print(f"{size:<10} {cal.measured_gflops:>10.1f} {cal.efficiency*100:>11.1f}% "
              f"{cal.arithmetic_intensity:>8.2f} {bound:>10}")
