"""
NumPy BLAS Benchmark Suite (CPU-only)

Complete BLAS Level 1, 2, and 3 benchmarks for CPU compute characterization.
Measures performance of vector-vector, matrix-vector, and matrix-matrix operations
across different sizes to characterize the compute hierarchy.

BLAS Levels:
  Level 1: Vector-Vector (O(n))     - DOT, AXPY, SCAL
  Level 2: Matrix-Vector (O(n²))    - GEMV, GER
  Level 3: Matrix-Matrix (O(n³))    - GEMM (matmul)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...schema import OperationCalibration, OperationType, PrecisionTestResult
from ....resource_model import Precision


# Precision to NumPy dtype mappings
# Canonical order: fp64, fp32, fp16, fp8, fp4, bf16, int64, int32, int16, int8, int4
NUMPY_PRECISION_MAP = {
    'fp64': (Precision.FP64, np.float64),
    'fp32': (Precision.FP32, np.float32),
    'fp16': (Precision.FP16, np.float16),
    'fp8': (Precision.FP8_E4M3, None),  # NumPy doesn't have native fp8
    'fp4': (Precision.FP4, None),  # NumPy doesn't have native fp4
    'bf16': (Precision.BF16, None),  # NumPy doesn't have native bfloat16
    'int64': (Precision.INT32, np.int64),  # Use INT32 enum for now (no INT64 in Precision enum)
    'int32': (Precision.INT32, np.int32),
    'int16': (Precision.INT16, np.int16),
    'int8': (Precision.INT8, np.int8),
    'int4': (Precision.INT4, None),  # NumPy doesn't have native int4
}


@dataclass
class BlasKernelSpec:
    """Specification for a BLAS kernel"""
    name: str                      # "dot", "axpy", "gemv", "gemm", etc.
    level: int                     # BLAS level: 1, 2, or 3
    operation_type: OperationType  # Enum value
    complexity: str                # "O(n)", "O(n²)", or "O(n³)"
    description: str               # Human-readable description


# ===================================
# BLAS Level 1: Vector-Vector (O(n))
# ===================================

def benchmark_blas1_dot_numpy(
    size: int,
    dtype=np.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 1: DOT product
    result = dot(x, y) = sum(x[i] * y[i])

    2n FLOPs, 2n elements read
    Arithmetic intensity: 1.0 FLOPs/byte (for FP32)
    """
    # Allocate vectors
    x = np.random.rand(size).astype(dtype)
    y = np.random.rand(size).astype(dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = np.dot(x, y)

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = np.dot(x, y)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Statistics
    mean_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)

    # Calculate metrics
    flops = 2 * size  # n multiplies + n adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    bytes_transferred = 2 * size * dtype().itemsize  # Read x and y
    bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9
    arithmetic_intensity = flops / bytes_transferred

    return {
        'operation': 'dot',
        'level': 1,
        'size': size,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'gflops': gflops,
        'bandwidth_gbps': bandwidth_gbps,
        'arithmetic_intensity': arithmetic_intensity,
        'num_trials': num_trials,
    }


def benchmark_blas1_axpy_numpy(
    size: int,
    dtype=np.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 1: AXPY
    y = alpha * x + y

    2n FLOPs, 3n elements (read x, read y, write y)
    Arithmetic intensity: 0.67 FLOPs/byte (for FP32)
    """
    # Allocate vectors
    alpha = 2.5
    x = np.random.rand(size).astype(dtype)
    y = np.random.rand(size).astype(dtype)

    # Warmup
    for _ in range(num_warmup):
        y[:] = alpha * x + y

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        y[:] = alpha * x + y
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Statistics
    mean_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)

    # Calculate metrics
    flops = 2 * size  # n multiplies + n adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    bytes_transferred = 3 * size * dtype().itemsize  # Read x, y, write y
    bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9
    arithmetic_intensity = flops / bytes_transferred

    return {
        'operation': 'axpy',
        'level': 1,
        'size': size,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'gflops': gflops,
        'bandwidth_gbps': bandwidth_gbps,
        'arithmetic_intensity': arithmetic_intensity,
        'num_trials': num_trials,
    }


# =====================================
# BLAS Level 2: Matrix-Vector (O(n²))
# =====================================

def benchmark_blas2_gemv_numpy(
    size: int,
    dtype=np.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 2: GEMV (General Matrix-Vector multiply)
    y = alpha * A @ x + beta * y

    For n×n matrix: 2n² FLOPs
    Arithmetic intensity: ~2.0 FLOPs/byte (for FP32, large n)
    """
    # Allocate matrix and vectors
    alpha = 1.0
    beta = 0.0
    A = np.random.rand(size, size).astype(dtype)
    x = np.random.rand(size).astype(dtype)
    y = np.zeros(size, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        y[:] = alpha * (A @ x) + beta * y

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        y[:] = alpha * (A @ x) + beta * y
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Statistics
    mean_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)

    # Calculate metrics
    flops = 2 * size * size  # n² multiplies + n² adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    # Read A (n²), read x (n), write y (n)
    bytes_transferred = (size * size + 2 * size) * dtype().itemsize
    bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9
    arithmetic_intensity = flops / bytes_transferred

    return {
        'operation': 'gemv',
        'level': 2,
        'size': size,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'gflops': gflops,
        'bandwidth_gbps': bandwidth_gbps,
        'arithmetic_intensity': arithmetic_intensity,
        'num_trials': num_trials,
    }


# =====================================
# BLAS Level 3: Matrix-Matrix (O(n³))
# =====================================

def benchmark_blas3_gemm_numpy(
    size: int,
    dtype=np.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 3: GEMM (General Matrix-Matrix multiply)
    C = alpha * A @ B + beta * C

    For n×n matrices: 2n³ FLOPs
    Arithmetic intensity: ~n/2 FLOPs/byte (grows with n!)
    """
    # Allocate matrices
    alpha = 1.0
    beta = 0.0
    A = np.random.rand(size, size).astype(dtype)
    B = np.random.rand(size, size).astype(dtype)
    C = np.zeros((size, size), dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        C[:] = alpha * (A @ B) + beta * C

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        C[:] = alpha * (A @ B) + beta * C
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Statistics
    mean_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)

    # Calculate metrics
    flops = 2 * size ** 3  # n³ multiplies + n³ adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    # Read A (n²), read B (n²), write C (n²)
    bytes_transferred = 3 * size * size * dtype().itemsize
    bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9
    arithmetic_intensity = flops / bytes_transferred

    return {
        'operation': 'gemm',
        'level': 3,
        'size': size,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'gflops': gflops,
        'bandwidth_gbps': bandwidth_gbps,
        'arithmetic_intensity': arithmetic_intensity,
        'num_trials': num_trials,
    }


# ===================================
# BLAS Suite Orchestrator
# ===================================

def calibrate_blas_suite_numpy(
    operations: List[str] = ['dot', 'axpy', 'gemv', 'gemm'],
    sizes: Dict[str, List[int]] = None,
    precisions: List[str] = None,
    theoretical_peak_gflops: float = 720.0,
    precision_peaks: Dict[str, float] = None,
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Run full BLAS benchmark suite (NumPy, CPU-only) across multiple precisions.

    Args:
        operations: List of BLAS operations to benchmark
        sizes: Dict mapping operation -> list of sizes
               If None, uses defaults:
               - Level 1 (dot, axpy): [1K, 10K, 100K, 1M, 10M, 100M]
               - Level 2 (gemv): [32, 64, 128, 256, 512, 1024, 2048]
               - Level 3 (gemm): [32, 64, 128, 256, 512, 1024, 2048]
        precisions: List of precisions to test (e.g., ['fp64', 'fp32', 'fp16', 'int8'])
                   If None, defaults to ['fp32']
        theoretical_peak_gflops: Theoretical peak GFLOPS for FP32
        precision_peaks: Dict mapping precision -> theoretical peak GFLOPS
                        If None, uses theoretical_peak_gflops for all
        num_trials: Number of trials per operation/size/precision

    Returns:
        List of OperationCalibration objects
    """
    # Default sizes if not specified
    if sizes is None:
        sizes = {
            # Level 1: Vector operations (use larger sizes)
            'dot':  [1000, 10000, 100000, 1000000, 10000000],
            'axpy': [1000, 10000, 100000, 1000000, 10000000],
            # Level 2: Matrix-vector (moderate sizes)
            'gemv': [32, 64, 128, 256, 512, 1024, 2048],
            # Level 3: Matrix-matrix (same as current matmul)
            'gemm': [32, 64, 128, 256, 512, 1024, 2048],
        }

    # Default precisions if not specified
    if precisions is None:
        precisions = ['fp32']

    # Default precision peaks if not specified
    if precision_peaks is None:
        precision_peaks = {prec: theoretical_peak_gflops for prec in precisions}

    # Operation dispatch table
    benchmarks = {
        'dot': (benchmark_blas1_dot_numpy, OperationType.BLAS1_DOT, 1),
        'axpy': (benchmark_blas1_axpy_numpy, OperationType.BLAS1_AXPY, 1),
        'gemv': (benchmark_blas2_gemv_numpy, OperationType.BLAS2_GEMV, 2),
        'gemm': (benchmark_blas3_gemm_numpy, OperationType.BLAS3_GEMM, 3),
    }

    calibrations = []

    print("Framework: NumPy (CPU-only)")
    print()
    print("BLAS Benchmark Suite:")
    print("=" * 90)

    for op_name in operations:
        if op_name not in benchmarks:
            print(f"⚠ Warning: Unknown operation '{op_name}', skipping")
            continue

        bench_fn, op_type, level = benchmarks[op_name]
        op_sizes = sizes.get(op_name, [])

        print(f"\nBLAS Level {level}: {op_name.upper()}")
        print("-" * 90)

        for size in op_sizes:
            # Format size for display
            if size >= 1000000:
                size_str = f"{size // 1000000}M"
            elif size >= 1000:
                size_str = f"{size // 1000}K"
            else:
                size_str = str(size)

            # Test each precision for this operation/size
            precision_results = {}
            fp32_gflops = None  # For speedup calculation

            for prec_name in precisions:
                if prec_name not in NUMPY_PRECISION_MAP:
                    # Unknown precision, mark as N/A
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason="Unknown precision"
                    )
                    continue

                prec_enum, dtype = NUMPY_PRECISION_MAP[prec_name]

                if dtype is None:
                    # Unsupported by NumPy (bf16, fp8)
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason="NumPy does not support this precision"
                    )
                    continue

                # Try to run benchmark for this precision
                try:
                    result = bench_fn(size, dtype=dtype, num_trials=num_trials)

                    # Get theoretical peak for this precision
                    peak_gflops = precision_peaks.get(prec_name, theoretical_peak_gflops)
                    efficiency = result['gflops'] / peak_gflops if peak_gflops > 0 else 0.0

                    # Track FP32 for speedup calculation
                    if prec_name == 'fp32':
                        fp32_gflops = result['gflops']

                    # Calculate speedup vs FP32
                    speedup = None
                    if fp32_gflops is not None and fp32_gflops > 0:
                        speedup = result['gflops'] / fp32_gflops

                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=True,
                        measured_gops=result['gflops'],
                        efficiency=efficiency,
                        mean_latency_ms=result['mean_latency_ms'],
                        std_latency_ms=result['std_latency_ms'],
                        min_latency_ms=result['min_latency_ms'],
                        max_latency_ms=result['mean_latency_ms'] + result['std_latency_ms'],
                        speedup_vs_fp32=speedup,
                        test_size=size,
                        num_trials=num_trials,
                        arithmetic_intensity=result['arithmetic_intensity'],
                        achieved_bandwidth_gbps=result['bandwidth_gbps']
                    )

                except Exception as e:
                    # Benchmark failed, mark as unsupported
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason=f"{type(e).__name__}: {str(e)[:50]}"
                    )

            # Find best precision result for summary display
            supported_results = [(prec, res) for prec, res in precision_results.items() if res.supported]
            if not supported_results:
                print(f"  Size {size_str:>6}... ALL PRECISIONS FAILED")
                continue

            # Use the highest GFLOPS as the "best" result
            best_prec, best_result = max(supported_results, key=lambda x: x[1].measured_gops or 0)

            # Create calibration object with precision results
            calibration = OperationCalibration(
                operation_type=op_type.value,
                measured_gflops=best_result.measured_gops,
                efficiency=best_result.efficiency,
                achieved_bandwidth_gbps=best_result.achieved_bandwidth_gbps,
                memory_bound=best_result.arithmetic_intensity < 4.0,  # Rule of thumb
                compute_bound=best_result.arithmetic_intensity >= 4.0,
                arithmetic_intensity=best_result.arithmetic_intensity,
                batch_size=1,
                input_shape=(size,) if level == 1 else (size, size),
                output_shape=(size,) if level <= 2 else (size, size),
                mean_latency_ms=best_result.mean_latency_ms,
                std_latency_ms=best_result.std_latency_ms,
                min_latency_ms=best_result.min_latency_ms,
                max_latency_ms=best_result.max_latency_ms,
                num_trials=num_trials,
                precision_results=precision_results,  # Store all precision results
                extra_params={
                    'size': size,
                    'operation': op_name,
                    'blas_level': level,
                    'framework': 'numpy',
                    'best_precision': best_prec,
                }
            )

            calibrations.append(calibration)

            # Print results (show best precision)
            num_supported = len(supported_results)
            num_total = len(precisions)
            gflops_str = f"{best_result.measured_gops:>8.1f} GFLOPS"
            lat_str = f"{best_result.mean_latency_ms:>8.2f} ms"
            ai_str = f"AI={best_result.arithmetic_intensity:>6.2f}"
            eff_str = f"({best_result.efficiency*100:>5.1f}%)"
            prec_str = f"[{best_prec}, {num_supported}/{num_total} precisions]"

            print(f"  Size {size_str:>6}... {gflops_str} {lat_str}  {ai_str}  {eff_str}  {prec_str}")

    print()
    print("=" * 90)
    print(f"BLAS Suite Complete: {len(calibrations)} calibrations")
    print("=" * 90)

    return calibrations


if __name__ == "__main__":
    # Standalone test
    print("NumPy BLAS Benchmark Suite (CPU-only)")
    print("=" * 90)
    print()

    # Quick test with smaller sizes
    calibrations = calibrate_blas_suite_numpy(
        operations=['dot', 'axpy', 'gemv', 'gemm'],
        sizes={
            'dot':  [1000, 10000, 100000, 1000000],
            'axpy': [1000, 10000, 100000, 1000000],
            'gemv': [64, 128, 256, 512, 1024],
            'gemm': [64, 128, 256, 512, 1024],
        },
        theoretical_peak_gflops=720.0
    )

    print("\nSummary by BLAS Level:")
    print("-" * 90)

    # Group by level
    by_level = {}
    for cal in calibrations:
        level = cal.extra_params['blas_level']
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(cal)

    for level in sorted(by_level.keys()):
        cals = by_level[level]
        best_gflops = max(c.measured_gflops for c in cals)
        best_ai = max(c.arithmetic_intensity for c in cals)
        print(f"Level {level}: Best GFLOPS = {best_gflops:.1f}, Best AI = {best_ai:.2f}")
