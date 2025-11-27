"""
Multi-Precision Matrix Multiplication Benchmark

Tests matmul performance across all numerical precisions (FP64, FP32, FP16, INT8, etc.)
Reports FAIL for unsupported precisions with clear failure reasons.
"""

import numpy as np
import time
from typing import List, Dict, Optional
from pathlib import Path

from ..schema import PrecisionTestResult, OperationType
from ..precision_detector import get_precision_capabilities
from ...resource_model import Precision


# Precision to dtype mappings
NUMPY_DTYPE_MAP = {
    Precision.FP64: np.float64,
    Precision.FP32: np.float32,
    Precision.FP16: np.float16,
    Precision.INT32: np.int32,
    Precision.INT16: np.int16,
    Precision.INT8: np.int8,
}

TORCH_DTYPE_MAP = {
    Precision.FP64: 'torch.float64',
    Precision.FP32: 'torch.float32',
    Precision.TF32: 'torch.float32',  # TF32 uses float32 dtype with Tensor Core truncation
    Precision.FP16: 'torch.float16',
    Precision.BF16: 'torch.bfloat16',
    Precision.INT32: 'torch.int32',
    Precision.INT16: 'torch.int16',
    Precision.INT8: 'torch.int8',
}


def benchmark_numpy_matmul_precision(
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

    # Calculate GFLOPS
    flops = 2.0 * N * N * N  # N^3 multiply-add operations
    mean_gflops = flops / (mean_time_ms / 1000.0) / 1e9

    # Memory traffic (for arithmetic intensity)
    bytes_per_element = dtype().itemsize
    bytes_transferred = (2 * N * N + N * N) * bytes_per_element  # Read A, B, write C
    arithmetic_intensity = flops / bytes_transferred

    return {
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'max_latency_ms': max_time_ms,
        'gflops': mean_gflops,
        'arithmetic_intensity': arithmetic_intensity,
    }


def benchmark_matmul_single_precision(
    N: int,
    precision: Precision,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> PrecisionTestResult:
    """
    Benchmark matmul at a single precision.

    Args:
        N: Matrix dimension
        precision: Precision to test
        device: 'cpu' or 'cuda'
        num_trials: Number of trials
        num_warmup: Number of warmup runs

    Returns:
        PrecisionTestResult with performance or failure reason
    """
    # Check if precision is supported
    supported_precisions, _ = get_precision_capabilities(device)

    if precision not in supported_precisions:
        return PrecisionTestResult(
            precision=precision.value,
            supported=False,
            failure_reason=f"{precision.value} not supported on {device}",
            test_size=N,
            num_trials=0
        )

    # Get dtype
    if precision in NUMPY_DTYPE_MAP:
        numpy_dtype = NUMPY_DTYPE_MAP[precision]
    else:
        return PrecisionTestResult(
            precision=precision.value,
            supported=False,
            failure_reason=f"{precision.value} requires PyTorch (not in NumPy)",
            test_size=N,
            num_trials=0
        )

    # Run benchmark
    try:
        result = benchmark_numpy_matmul_precision(N, numpy_dtype, num_trials, num_warmup)

        return PrecisionTestResult(
            precision=precision.value,
            supported=True,
            failure_reason=None,
            measured_gops=result['gflops'],  # Generic: GFLOPS for float, GIOPS for int
            efficiency=None,  # Calculated later against theoretical peak
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
        return PrecisionTestResult(
            precision=precision.value,
            supported=False,
            failure_reason=f"Runtime error: {str(e)[:60]}",
            test_size=N,
            num_trials=0
        )


def calibrate_matmul_all_precisions(
    sizes: List[int],
    precisions: List[Precision],
    theoretical_peaks: Dict[str, float],
    device: str = 'cpu',
    num_trials: int = 10,
    min_useful_throughput: float = 50.0  # Skip precision for larger sizes if < this GOPS
) -> Dict[int, Dict[str, PrecisionTestResult]]:
    """
    Calibrate matmul across all precisions at multiple sizes.

    Args:
        sizes: Matrix sizes to test
        precisions: List of Precision enums to test
        theoretical_peaks: Dict mapping precision name -> theoretical GFLOPS/GIOPS
        device: 'cpu' or 'cuda'
        num_trials: Trials per test
        min_useful_throughput: Skip precision for larger sizes if throughput < this (GOPS)
                               Default 50 GOPS - below this is not useful for Embodied AI

    Returns:
        Dict mapping size -> Dict mapping precision name -> PrecisionTestResult
    """
    results = {}
    skip_precisions = set()  # Precisions to skip for larger sizes

    # Detect actual execution device
    actual_device = 'cpu'  # Currently only NumPy is implemented, which always uses CPU
    if device == 'cuda':
        # Check if CUDA is actually available
        try:
            import torch
            if torch.cuda.is_available():
                # NOTE: Current implementation uses NumPy (CPU only)
                # TODO: Add PyTorch/CUDA path for GPU benchmarks
                print(f"⚠ NOTE: CUDA requested but benchmarks use NumPy (CPU-only)")
                print(f"         Results will reflect CPU performance, not GPU")
                print()
        except ImportError:
            pass

    for N in sizes:
        print(f"\nCalibrating matmul {N}×{N} across {len(precisions)} precisions...")

        precision_results = {}
        fp32_latency = None

        for precision in precisions:
            # Skip if this precision was too slow on smaller size
            if precision in skip_precisions:
                print(f"  {precision.value:8s}... ⊘ SKIPPED (< {min_useful_throughput} GOPS on smaller size)")
                # Create skipped result
                precision_results[precision.value] = PrecisionTestResult(
                    precision=precision.value,
                    supported=False,
                    failure_reason=f"Skipped: throughput <{min_useful_throughput} GOPS on smaller matrix",
                    test_size=N,
                    num_trials=0
                )
                continue

            # Print status
            print(f"  {precision.value:8s}...", end=" ", flush=True)

            # Run benchmark
            result = benchmark_matmul_single_precision(N, precision, device, num_trials)

            if result.supported:
                # Calculate efficiency vs theoretical peak
                theoretical_peak = theoretical_peaks.get(precision.value, None)
                if theoretical_peak and theoretical_peak > 0:
                    result.efficiency = result.measured_gops / theoretical_peak

                # Determine units: GFLOPS for float, GIOPS for integer
                is_int = precision in [Precision.INT32, Precision.INT16, Precision.INT8, Precision.INT4]
                units = "GIOPS" if is_int else "GFLOPS"

                # Format latency: show seconds if >1000ms, else milliseconds
                latency_ms = result.mean_latency_ms
                if latency_ms >= 1000:
                    latency_str = f"{latency_ms/1000:5.1f}s"
                else:
                    latency_str = f"{latency_ms:6.1f}ms"

                print(f"✓ {result.measured_gops:7.1f} {units} ({latency_str})", end="")
                if result.efficiency:
                    print(f" {result.efficiency*100:5.1f}% eff")
                else:
                    print()

                # Check if this precision has unusable throughput (skip for larger sizes)
                # Below 50 GOPS is not useful for Embodied AI applications
                if result.measured_gops < min_useful_throughput:
                    skip_precisions.add(precision)
                    print(f"    ⚠ Warning: Throughput <{min_useful_throughput} GOPS, will skip this precision for larger sizes")

                # Track FP32 for speedup calculations
                if precision == Precision.FP32:
                    fp32_latency = result.mean_latency_ms
            else:
                print(f"✗ FAIL: {result.failure_reason}")

            precision_results[precision.value] = result

        # Calculate speedups relative to FP32
        if fp32_latency and fp32_latency > 0:
            for prec_name, result in precision_results.items():
                if result.supported and result.mean_latency_ms:
                    result.speedup_vs_fp32 = fp32_latency / result.mean_latency_ms

        results[N] = precision_results

    return results


if __name__ == "__main__":
    # Standalone test
    print("Multi-Precision Matrix Multiplication Calibration")
    print("=" * 70)

    # Test precisions
    precisions_to_test = [
        Precision.FP64,
        Precision.FP32,
        Precision.FP16,
        Precision.INT32,
        Precision.INT16,
        Precision.INT8,
    ]

    # Dummy theoretical peaks
    theoretical_peaks = {
        'fp64': 360.0,
        'fp32': 720.0,
        'fp16': 720.0,
        'int32': 360.0,
        'int16': 720.0,
        'int8': 1440.0,
    }

    results = calibrate_matmul_all_precisions(
        sizes=[1024, 2048],
        precisions=precisions_to_test,
        theoretical_peaks=theoretical_peaks,
        device='cpu',
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
                # Determine units: GFLOPS for float, GIOPS for integer
                is_int = prec_name in ['int32', 'int16', 'int8', 'int4']
                units = "GIOPS" if is_int else "GFLOPS"
                print(f"  {prec_name:8s}: {result.measured_gops:7.1f} {units} "
                      f"({result.efficiency*100:5.1f}% eff, {result.speedup_vs_fp32:.2f}× vs FP32)")
            else:
                print(f"  {prec_name:8s}: FAIL - {result.failure_reason}")
