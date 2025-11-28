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
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...schema import OperationCalibration, OperationType, PrecisionTestResult
from ....resource_model import Precision


class TimeoutError(Exception):
    """Raised when a benchmark trial exceeds the timeout"""
    pass


# =============================================================================
# Cross-Platform Benchmark Timeout using Multiprocessing
# =============================================================================
# The previous implementation used signal.SIGALRM which is Unix-only.
# This implementation uses multiprocessing.Process which works on all platforms
# (Linux, macOS, Windows) and can forcibly terminate stuck computations.
# =============================================================================

def _gemm_worker(size: int, dtype_name: str, num_trials: int, num_warmup: int,
                 result_queue: mp.Queue):
    """
    Worker function for GEMM benchmark that runs in a separate process.
    Results are sent back via a Queue.

    Args:
        size: Matrix dimension (N for NxN matrices)
        dtype_name: NumPy dtype name as string (e.g., 'float32', 'int8')
        num_trials: Number of benchmark trials
        num_warmup: Number of warmup iterations
        result_queue: Queue to send results back to parent process
    """
    # Map dtype name to actual dtype
    dtype_map = {
        'float64': np.float64,
        'float32': np.float32,
        'float16': np.float16,
        'int64': np.int64,
        'int32': np.int32,
        'int16': np.int16,
        'int8': np.int8,
    }
    dtype = dtype_map.get(dtype_name, np.float32)

    try:
        # Allocate matrices
        alpha = 1.0 if not np.issubdtype(dtype, np.integer) else 1
        beta = 0.0 if not np.issubdtype(dtype, np.integer) else 0

        if np.issubdtype(dtype, np.integer):
            A = np.random.randint(1, 100, size=(size, size), dtype=dtype)
            B = np.random.randint(1, 100, size=(size, size), dtype=dtype)
        else:
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
            times.append((end - start) * 1000)  # ms

        # Calculate metrics
        mean_time_ms = float(np.mean(times))
        std_time_ms = float(np.std(times))
        min_time_ms = float(np.min(times))

        flops = 2 * size ** 3  # n³ multiplies + n³ adds
        gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

        # Read A (n²), read B (n²), write C (n²)
        bytes_transferred = 3 * size * size * dtype().itemsize
        bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9
        arithmetic_intensity = flops / bytes_transferred

        result_queue.put({
            'success': True,
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
        })

    except Exception as e:
        result_queue.put({
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}"
        })


def run_benchmark_with_timeout(worker_func, args: tuple, timeout_seconds: int = 5) -> Dict:
    """
    Run a benchmark in a separate process with a timeout circuit breaker.

    This is cross-platform and works on Linux, macOS, and Windows.
    If the benchmark exceeds the timeout, the process is forcibly terminated.

    Args:
        worker_func: The worker function to run (must accept result_queue as last arg)
        args: Arguments to pass to worker_func (excluding result_queue)
        timeout_seconds: Maximum seconds before killing the process

    Returns:
        Dict with 'success' key and either results or error info
    """
    result_queue = mp.Queue()

    # Create and start the worker process
    process = mp.Process(
        target=worker_func,
        args=args + (result_queue,)
    )
    process.start()

    # Wait for completion or timeout
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # Timeout - kill the process
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            'success': False,
            'error': f'Timeout after {timeout_seconds}s',
            'timed_out': True
        }

    # Process completed - get result from queue
    if not result_queue.empty():
        return result_queue.get()
    else:
        return {
            'success': False,
            'error': 'No result returned from worker process'
        }


# Precision to NumPy dtype mappings
# Canonical order: fp64, fp32, tf32, fp16, fp8, fp4, bf16, int64, int32, int16, int8, int4
NUMPY_PRECISION_MAP = {
    'fp64': (Precision.FP64, np.float64),
    'fp32': (Precision.FP32, np.float32),
    'tf32': (Precision.TF32, None),  # TF32 is NVIDIA Tensor Core only, not available in NumPy
    'fp16': (Precision.FP16, np.float16),
    'fp8': (Precision.FP8_E4M3, None),  # NumPy doesn't have native fp8
    'fp4': (Precision.FP4, None),  # NumPy doesn't have native fp4
    'bf16': (Precision.BF16, None),  # NumPy doesn't have native bfloat16
    'int64': (Precision.INT64, np.int64),
    'int32': (Precision.INT32, np.int32),
    'int16': (Precision.INT16, np.int16),
    'int8': (Precision.INT8, np.int8),
    'int4': (Precision.INT4, None),  # NumPy doesn't have native int4
}


def _generate_random_array(shape, dtype):
    """
    Generate random array with appropriate values for dtype.

    For float types: values in [0.0, 1.0)
    For integer types: values in [1, 100] to avoid overflow and ensure non-zero results
    """
    if np.issubdtype(dtype, np.integer):
        # Integer types: use randint with reasonable range
        return np.random.randint(1, 100, size=shape, dtype=dtype)
    else:
        # Float types: use rand
        if isinstance(shape, tuple):
            return np.random.rand(*shape).astype(dtype)
        else:
            return np.random.rand(shape).astype(dtype)


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
    x = _generate_random_array(size, dtype)
    y = _generate_random_array(size, dtype)

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
    alpha = 2.5 if not np.issubdtype(dtype, np.integer) else 2
    x = _generate_random_array(size, dtype)
    y = _generate_random_array(size, dtype)

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
    alpha = 1.0 if not np.issubdtype(dtype, np.integer) else 1
    beta = 0.0 if not np.issubdtype(dtype, np.integer) else 0
    A = _generate_random_array((size, size), dtype)
    x = _generate_random_array(size, dtype)
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
    num_warmup: int = 3,
    timeout_seconds: int = 5
) -> Dict:
    """
    BLAS Level 3: GEMM (General Matrix-Matrix multiply)
    C = alpha * A @ B + beta * C

    For n×n matrices: 2n³ FLOPs
    Arithmetic intensity: ~n/2 FLOPs/byte (grows with n!)

    This function uses a cross-platform circuit breaker that runs the benchmark
    in a separate process. If the benchmark exceeds timeout_seconds, the process
    is forcibly terminated. This works on Linux, macOS, and Windows.

    Args:
        timeout_seconds: Maximum seconds for the entire benchmark (0 = no timeout)
    """
    # Convert dtype to string for pickling across process boundary
    dtype_name = np.dtype(dtype).name

    # Run benchmark in subprocess with timeout
    result = run_benchmark_with_timeout(
        _gemm_worker,
        args=(size, dtype_name, num_trials, num_warmup),
        timeout_seconds=timeout_seconds
    )

    if not result['success']:
        # Propagate timeout or error as exception
        if result.get('timed_out'):
            raise TimeoutError(f"GEMM benchmark exceeded {timeout_seconds}s timeout")
        else:
            raise RuntimeError(result.get('error', 'Unknown error in benchmark worker'))

    return result


# ===================================
# BLAS Suite Orchestrator
# ===================================

def calibrate_blas_suite_numpy(
    operations: List[str] = ['dot', 'axpy', 'gemv', 'gemm'],
    sizes: Dict[str, List[int]] = None,
    precisions: List[str] = None,
    theoretical_peak_gflops: float = 720.0,
    precision_peaks: Dict[str, float] = None,
    num_trials: int = 10,
    min_useful_gflops: float = 1.0
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

        # Track which precisions to skip for remaining sizes (due to poor performance)
        # Only enable early termination for BLAS Level 3 (GEMM) to avoid long runtimes
        # BUT only after reaching a minimum size where timing overhead doesn't dominate
        skipped_precisions = {}  # {prec_name: (reason, skip_after_size)}
        enable_early_termination = (level == 3)  # Only for GEMM
        # Minimum size for early termination - small matrices have high overhead
        # 256x256 GEMM = 33M FLOPs, gives meaningful GFLOPS measurement
        min_early_termination_size = 256

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
                # Check if this precision was skipped due to poor performance (only for GEMM)
                if enable_early_termination and prec_name in skipped_precisions:
                    reason, skip_size = skipped_precisions[prec_name]
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason=f"Skipped (poor performance at size {skip_size}: {reason})"
                    )
                    continue

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
                    # Add timeout for GEMM (level 3) to avoid very slow integer matmuls
                    if level == 3:
                        result = bench_fn(size, dtype=dtype, num_trials=num_trials, timeout_seconds=5)
                    else:
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

                    # Check for poor performance - skip this precision for larger sizes (GEMM only)
                    # Only trigger early termination after reaching minimum size threshold
                    # Small matrices have high overhead and naturally low GFLOPS
                    if (enable_early_termination and
                        size >= min_early_termination_size and
                        result['gflops'] < min_useful_gflops and
                        prec_name not in skipped_precisions):
                        size_display = f"{size_str}"
                        skipped_precisions[prec_name] = (f"{result['gflops']:.1f} GFLOPS < {min_useful_gflops} GFLOPS", size_display)

                except TimeoutError as e:
                    # Benchmark timed out - mark as skipped for remaining sizes (GEMM only)
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason=f"Timeout: {str(e)}"
                    )
                    if enable_early_termination and prec_name not in skipped_precisions:
                        skipped_precisions[prec_name] = (f"Timeout (>{5}s)", size_str)

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

            # Print results for all precisions (show calibration for each precision)
            print(f"  Size {size_str:>6}:")
            from ...schema import CANONICAL_PRECISION_ORDER
            for prec_name in [p for p in CANONICAL_PRECISION_ORDER if p in precision_results]:
                result = precision_results[prec_name]
                if result.supported:
                    gflops = result.measured_gops
                    latency_ms = result.mean_latency_ms
                    # Format latency
                    if latency_ms >= 1000:
                        lat_str = f"{latency_ms/1000:>6.2f}s"
                    else:
                        lat_str = f"{latency_ms:>6.2f}ms"

                    # Determine unit based on precision type
                    unit = "GIOPS" if prec_name.startswith('int') else "GFLOPS"

                    # Highlight if performance is poor
                    if gflops < min_useful_gflops:
                        status = f"⚠ SLOW ({gflops:>6.1f} {unit} < {min_useful_gflops} GFLOPS threshold)"
                    else:
                        status = f"{gflops:>6.1f} {unit}"
                    print(f"    {prec_name:<6} {status:>50}  {lat_str}")
                else:
                    # Show why it failed/was skipped
                    reason = result.failure_reason or "Unknown"
                    print(f"    {prec_name:<6} {'SKIPPED':>50}  ({reason})")

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
