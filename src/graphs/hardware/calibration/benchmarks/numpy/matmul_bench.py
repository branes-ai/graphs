"""
NumPy Matrix Multiplication Benchmark (CPU-only)

Pure NumPy implementation for CPU performance characterization.
Important for understanding NumPy performance in real Embodied AI applications.
"""

import numpy as np
import time
from typing import List, Dict, Optional

from ...schema import PrecisionTestResult
from ....resource_model import Precision


# Precision to NumPy dtype mappings
NUMPY_DTYPE_MAP = {
    Precision.FP64: np.float64,
    Precision.FP32: np.float32,
    Precision.FP16: np.float16,
    Precision.INT32: np.int32,
    Precision.INT16: np.int16,
    Precision.INT8: np.int8,
}


def benchmark_numpy_matmul(
    N: int,
    dtype,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    Benchmark NumPy matmul at specific precision.

    Args:
        N: Matrix dimension
        dtype: NumPy dtype
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

    # Calculate GFLOPS/GIOPS
    ops = 2.0 * N * N * N  # N^3 multiply-add operations
    mean_gops = ops / (mean_time_ms / 1000.0) / 1e9

    # Memory traffic (for arithmetic intensity)
    bytes_per_element = dtype().itemsize
    bytes_transferred = (2 * N * N + N * N) * bytes_per_element  # Read A, B, write C
    arithmetic_intensity = ops / bytes_transferred

    return {
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'max_latency_ms': max_time_ms,
        'gops': mean_gops,  # GFLOPS for float, GIOPS for int
        'arithmetic_intensity': arithmetic_intensity,
    }


def calibrate_matmul_numpy(
    sizes: List[int],
    precisions: List[Precision],
    theoretical_peaks: Dict[str, float],
    num_trials: int = 10,
    min_useful_throughput: float = 50.0
) -> Dict[int, Dict[str, PrecisionTestResult]]:
    """
    Calibrate NumPy matmul across precisions at multiple sizes.

    Args:
        sizes: Matrix sizes to test
        precisions: List of Precision enums to test
        theoretical_peaks: Dict mapping precision name -> theoretical GFLOPS/GIOPS
        num_trials: Trials per test
        min_useful_throughput: Skip precision for larger sizes if throughput < this (GOPS)

    Returns:
        Dict mapping size -> Dict mapping precision name -> PrecisionTestResult
    """
    results = {}
    skip_precisions = set()

    print("Framework: NumPy (CPU-only)")
    print()

    for N in sizes:
        print(f"Calibrating matmul {N}×{N} across {len(precisions)} precisions...")

        precision_results = {}
        fp32_latency = None

        for precision in precisions:
            # Skip if this precision was too slow on smaller size
            if precision in skip_precisions:
                print(f"  {precision.value:8s}... SKIPPED (< {min_useful_throughput} GOPS on smaller size)")
                precision_results[precision.value] = PrecisionTestResult(
                    precision=precision.value,
                    supported=False,
                    failure_reason=f"Skipped: throughput <{min_useful_throughput} GOPS on smaller matrix",
                    test_size=N,
                    num_trials=0
                )
                continue

            # Check if NumPy supports this precision
            if precision not in NUMPY_DTYPE_MAP:
                print(f"  {precision.value:8s}... [X] UNSUPPORTED (not in NumPy)")
                precision_results[precision.value] = PrecisionTestResult(
                    precision=precision.value,
                    supported=False,
                    failure_reason=f"{precision.value} not available in NumPy",
                    test_size=N,
                    num_trials=0
                )
                continue

            dtype = NUMPY_DTYPE_MAP[precision]

            # Print status
            print(f"  {precision.value:8s}...", end=" ", flush=True)

            # Run benchmark
            try:
                result = benchmark_numpy_matmul(N, dtype, num_trials)

                # Calculate efficiency vs theoretical peak
                theoretical_peak = theoretical_peaks.get(precision.value, None)
                efficiency = None
                if theoretical_peak and theoretical_peak > 0:
                    efficiency = result['gops'] / theoretical_peak

                # Determine units: GFLOPS for float, GIOPS for integer
                is_int = precision in [Precision.INT32, Precision.INT16, Precision.INT8]
                units = "GIOPS" if is_int else "GFLOPS"

                # Format latency
                latency_ms = result['mean_latency_ms']
                if latency_ms >= 1000:
                    latency_str = f"{latency_ms/1000:5.1f}s"
                else:
                    latency_str = f"{latency_ms:6.1f}ms"

                print(f"[OK] {result['gops']:7.1f} {units} ({latency_str})", end="")
                if efficiency:
                    print(f" {efficiency*100:5.1f}% eff", end="")
                    # Flag anomalous efficiency
                    if efficiency > 1.10:  # >110% indicates turbo boost or measurement issue
                        print(" [!] ABOVE THEORETICAL")
                    else:
                        print()
                else:
                    print()

                # Check if unusable throughput
                if result['gops'] < min_useful_throughput:
                    skip_precisions.add(precision)
                    print(f"    [!] Warning: Throughput <{min_useful_throughput} GOPS, will skip for larger sizes")

                # Explain high efficiency
                if efficiency and efficiency > 1.10:
                    print(f"    Note: Likely caused by Turbo Boost, optimized BLAS, or conservative theoretical peak")

                # Track FP32 for speedup calculations
                if precision == Precision.FP32:
                    fp32_latency = result['mean_latency_ms']

                # Create result object
                precision_results[precision.value] = PrecisionTestResult(
                    precision=precision.value,
                    supported=True,
                    failure_reason=None,
                    measured_gops=result['gops'],
                    efficiency=efficiency,
                    mean_latency_ms=result['mean_latency_ms'],
                    std_latency_ms=result['std_latency_ms'],
                    min_latency_ms=result['min_latency_ms'],
                    max_latency_ms=result['max_latency_ms'],
                    speedup_vs_fp32=None,  # Calculated later
                    test_size=N,
                    num_trials=num_trials,
                    arithmetic_intensity=result['arithmetic_intensity'],
                )

            except Exception as e:
                print(f"[X] FAIL: {str(e)[:60]}")
                precision_results[precision.value] = PrecisionTestResult(
                    precision=precision.value,
                    supported=False,
                    failure_reason=f"Runtime error: {str(e)[:60]}",
                    test_size=N,
                    num_trials=0
                )

        # Calculate speedups relative to FP32
        if fp32_latency and fp32_latency > 0:
            for prec_name, result in precision_results.items():
                if result.supported and result.mean_latency_ms:
                    result.speedup_vs_fp32 = fp32_latency / result.mean_latency_ms

        results[N] = precision_results

    return results


if __name__ == "__main__":
    # Standalone test
    print("NumPy Matrix Multiplication Calibration (CPU-only)")
    print("=" * 70)

    precisions_to_test = [
        Precision.FP64,
        Precision.FP32,
        Precision.FP16,
        Precision.INT32,
        Precision.INT16,
        Precision.INT8,
    ]

    theoretical_peaks = {
        'fp64': 360.0,
        'fp32': 720.0,
        'fp16': 720.0,
        'int32': 360.0,
        'int16': 720.0,
        'int8': 1440.0,
    }

    results = calibrate_matmul_numpy(
        sizes=[1024],
        precisions=precisions_to_test,
        theoretical_peaks=theoretical_peaks,
        num_trials=5
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for size, prec_results in results.items():
        print(f"\nSize {size}×{size}:")
        for prec_name, result in prec_results.items():
            if result.supported:
                is_int = prec_name in ['int32', 'int16', 'int8']
                units = "GIOPS" if is_int else "GFLOPS"
                print(f"  {prec_name:8s}: {result.measured_gops:7.1f} {units}")
            else:
                print(f"  {prec_name:8s}: UNSUPPORTED - {result.failure_reason}")
