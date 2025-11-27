"""
PyTorch BLAS Benchmark Suite (CPU or GPU)

Complete BLAS Level 1, 2, and 3 benchmarks with GPU support.
Measures performance of vector-vector, matrix-vector, and matrix-matrix operations
across different sizes to characterize the compute hierarchy.

BLAS Levels:
  Level 1: Vector-Vector (O(n))     - DOT, AXPY, SCAL
  Level 2: Matrix-Vector (O(n²))    - GEMV, GER
  Level 3: Matrix-Matrix (O(n³))    - GEMM (matmul)
"""

import time
import signal
from typing import Dict, List, Tuple, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ...schema import OperationCalibration, OperationType, PrecisionTestResult
from ....resource_model import Precision


class TimeoutError(Exception):
    """Raised when a benchmark trial exceeds the timeout"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Benchmark trial exceeded timeout")


class BenchmarkTimeout:
    """Context manager for benchmark timeouts using SIGALRM"""
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds > 0:
            signal.alarm(0)  # Cancel the alarm
        return False  # Don't suppress exceptions


# Precision to PyTorch dtype mappings
# Canonical order: fp64, fp32, tf32, fp16, fp8, fp4, bf16, int64, int32, int16, int8, int4
# Note: TF32 uses float32 dtype but with Tensor Core truncation (19-bit mantissa)
if TORCH_AVAILABLE:
    PYTORCH_PRECISION_MAP = {
        'fp64': (Precision.FP64, torch.float64),
        'fp32': (Precision.FP32, torch.float32),
        'tf32': (Precision.TF32, torch.float32),  # Same dtype, different Tensor Core mode
        'fp16': (Precision.FP16, torch.float16),
        'fp8': (Precision.FP8_E4M3, None),  # PyTorch 2.1+ has experimental fp8, but not widely supported
        'fp4': (Precision.FP4, None),  # PyTorch doesn't have native fp4
        'bf16': (Precision.BF16, torch.bfloat16),
        'int64': (Precision.INT64, torch.int64),
        'int32': (Precision.INT32, torch.int32),
        'int16': (Precision.INT16, torch.int16),
        'int8': (Precision.INT8, torch.int8),
        'int4': (Precision.INT4, None),  # PyTorch doesn't have native int4
    }
else:
    PYTORCH_PRECISION_MAP = {}


def _generate_random_tensor(shape, dtype, device='cpu'):
    """
    Generate random tensor with appropriate values for dtype.

    For float types: values in [0.0, 1.0)
    For integer types: values in [1, 100] to avoid overflow and ensure non-zero results
    """
    if dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # Integer types: use randint with reasonable range
        if isinstance(shape, tuple):
            return torch.randint(1, 100, shape, dtype=dtype, device=device)
        else:
            return torch.randint(1, 100, (shape,), dtype=dtype, device=device)
    else:
        # Float types: use rand
        if isinstance(shape, tuple):
            return torch.rand(*shape, dtype=dtype, device=device)
        else:
            return torch.rand(shape, dtype=dtype, device=device)


# ===================================
# BLAS Level 1: Vector-Vector (O(n))
# ===================================

def benchmark_blas1_dot_pytorch(
    size: int,
    device: str = 'cpu',
    dtype=torch.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 1: DOT product
    result = dot(x, y) = sum(x[i] * y[i])

    2n FLOPs, 2n elements read
    Arithmetic intensity: 1.0 FLOPs/byte (for FP32)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    torch_device = torch.device(device)

    # Allocate vectors
    x = _generate_random_tensor(size, dtype, device=torch_device)
    y = _generate_random_tensor(size, dtype, device=torch_device)

    # Warmup
    for _ in range(num_warmup):
        _ = torch.dot(x, y)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = torch.dot(x, y)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Statistics
    import statistics
    mean_time_ms = statistics.mean(times)
    std_time_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time_ms = min(times)

    # Calculate metrics
    flops = 2 * size  # n multiplies + n adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    item_size = 4 if dtype == torch.float32 else 2
    bytes_transferred = 2 * size * item_size  # Read x and y
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


def benchmark_blas1_axpy_pytorch(
    size: int,
    device: str = 'cpu',
    dtype=torch.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 1: AXPY
    y = alpha * x + y

    2n FLOPs, 3n elements (read x, read y, write y)
    Arithmetic intensity: 0.67 FLOPs/byte (for FP32)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    torch_device = torch.device(device)

    # Allocate vectors
    alpha = 2.5 if dtype not in [torch.int64, torch.int32, torch.int16, torch.int8] else 2
    x = _generate_random_tensor(size, dtype, device=torch_device)
    y = _generate_random_tensor(size, dtype, device=torch_device)

    # Warmup
    for _ in range(num_warmup):
        y[:] = alpha * x + y
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        y[:] = alpha * x + y

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Statistics
    import statistics
    mean_time_ms = statistics.mean(times)
    std_time_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time_ms = min(times)

    # Calculate metrics
    flops = 2 * size  # n multiplies + n adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    item_size = 4 if dtype == torch.float32 else 2
    bytes_transferred = 3 * size * item_size  # Read x, y, write y
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

def benchmark_blas2_gemv_pytorch(
    size: int,
    device: str = 'cpu',
    dtype=torch.float32,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    BLAS Level 2: GEMV (General Matrix-Vector multiply)
    y = alpha * A @ x + beta * y

    For n×n matrix: 2n² FLOPs
    Arithmetic intensity: ~2.0 FLOPs/byte (for FP32, large n)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    torch_device = torch.device(device)

    # Allocate matrix and vectors
    alpha = 1.0 if dtype not in [torch.int64, torch.int32, torch.int16, torch.int8] else 1
    beta = 0.0 if dtype not in [torch.int64, torch.int32, torch.int16, torch.int8] else 0
    A = _generate_random_tensor((size, size), dtype, device=torch_device)
    x = _generate_random_tensor(size, dtype, device=torch_device)
    y = torch.zeros(size, dtype=dtype, device=torch_device)

    # Warmup
    for _ in range(num_warmup):
        y[:] = alpha * (A @ x) + beta * y
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        y[:] = alpha * (A @ x) + beta * y

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Statistics
    import statistics
    mean_time_ms = statistics.mean(times)
    std_time_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time_ms = min(times)

    # Calculate metrics
    flops = 2 * size * size  # n² multiplies + n² adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    item_size = 4 if dtype == torch.float32 else 2
    # Read A (n²), read x (n), write y (n)
    bytes_transferred = (size * size + 2 * size) * item_size
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

def benchmark_blas3_gemm_pytorch(
    size: int,
    device: str = 'cpu',
    dtype=torch.float32,
    num_trials: int = 10,
    num_warmup: int = 3,
    timeout_seconds: int = 5
) -> Dict:
    """
    BLAS Level 3: GEMM (General Matrix-Matrix multiply)
    C = alpha * A @ B + beta * C

    For n×n matrices: 2n³ FLOPs
    Arithmetic intensity: ~n/2 FLOPs/byte (grows with n!)

    Args:
        timeout_seconds: Maximum seconds per trial (0 = no timeout)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    torch_device = torch.device(device)

    # Allocate matrices
    alpha = 1.0 if dtype not in [torch.int64, torch.int32, torch.int16, torch.int8] else 1
    beta = 0.0 if dtype not in [torch.int64, torch.int32, torch.int16, torch.int8] else 0
    A = _generate_random_tensor((size, size), dtype, device=torch_device)
    B = _generate_random_tensor((size, size), dtype, device=torch_device)
    C = torch.zeros(size, size, dtype=dtype, device=torch_device)

    # Warmup (with timeout)
    try:
        with BenchmarkTimeout(timeout_seconds):
            for _ in range(num_warmup):
                C[:] = alpha * (A @ B) + beta * C
                if device == 'cuda':
                    torch.cuda.synchronize()
    except TimeoutError:
        raise TimeoutError(f"Warmup exceeded {timeout_seconds}s timeout")

    # Benchmark
    times = []
    for trial_idx in range(num_trials):
        try:
            with BenchmarkTimeout(timeout_seconds):
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()
                C[:] = alpha * (A @ B) + beta * C

                if device == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)
        except TimeoutError:
            raise TimeoutError(f"Trial {trial_idx+1}/{num_trials} exceeded {timeout_seconds}s timeout")

    # Statistics
    import statistics
    mean_time_ms = statistics.mean(times)
    std_time_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time_ms = min(times)

    # Calculate metrics
    flops = 2 * size ** 3  # n³ multiplies + n³ adds
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9

    item_size = 4 if dtype == torch.float32 else 2
    # Read A (n²), read B (n²), write C (n²)
    bytes_transferred = 3 * size * size * item_size
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

def calibrate_blas_suite_pytorch(
    operations: List[str] = ['dot', 'axpy', 'gemv', 'gemm'],
    sizes: Dict[str, List[int]] = None,
    precisions: List[str] = None,
    theoretical_peak_gflops: float = 720.0,
    precision_peaks: Dict[str, float] = None,
    device: str = 'cpu',
    num_trials: int = 10,
    min_useful_gflops: float = 1.0
) -> List[OperationCalibration]:
    """
    Run full BLAS benchmark suite (PyTorch, CPU or GPU) across multiple precisions.

    Args:
        operations: List of BLAS operations to benchmark
        sizes: Dict mapping operation -> list of sizes
               If None, uses defaults
        precisions: List of precisions to test (e.g., ['fp64', 'fp32', 'fp16', 'int8'])
                   If None, defaults to ['fp32']
        theoretical_peak_gflops: Theoretical peak GFLOPS for FP32
        precision_peaks: Dict mapping precision -> theoretical peak GFLOPS
                        If None, uses theoretical_peak_gflops for all
        device: 'cpu' or 'cuda'
        num_trials: Number of trials per operation/size/precision

    Returns:
        List of OperationCalibration objects
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PyTorch benchmarks")

    # Verify device is available
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    # Default sizes if not specified
    if sizes is None:
        sizes = {
            # Level 1: Vector operations (use larger sizes)
            'dot':  [1000, 10000, 100000, 1000000, 10000000],
            'axpy': [1000, 10000, 100000, 1000000, 10000000],
            # Level 2: Matrix-vector (moderate sizes)
            'gemv': [32, 64, 128, 256, 512, 1024, 2048],
            # Level 3: Matrix-matrix
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
        'dot': (benchmark_blas1_dot_pytorch, OperationType.BLAS1_DOT, 1),
        'axpy': (benchmark_blas1_axpy_pytorch, OperationType.BLAS1_AXPY, 1),
        'gemv': (benchmark_blas2_gemv_pytorch, OperationType.BLAS2_GEMV, 2),
        'gemm': (benchmark_blas3_gemm_pytorch, OperationType.BLAS3_GEMM, 3),
    }

    calibrations = []

    # Print framework info
    print(f"Framework: PyTorch {torch.__version__}")
    if device == 'cuda':
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA:   {torch.version.cuda}")
    else:
        print(f"  Device: CPU")
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
        skipped_precisions = {}  # {prec_name: (reason, skip_after_size)}
        enable_early_termination = (level == 3)  # Only for GEMM

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

                if prec_name not in PYTORCH_PRECISION_MAP:
                    # Unknown precision, mark as N/A
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason="Unknown precision"
                    )
                    continue

                prec_enum, dtype = PYTORCH_PRECISION_MAP[prec_name]

                if dtype is None:
                    # Unsupported by PyTorch (fp8)
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason="PyTorch does not support this precision"
                    )
                    continue

                # Try to run benchmark for this precision
                try:
                    # Handle TF32 mode for CUDA devices
                    # TF32 uses FP32 tensors but truncates mantissa on Tensor Cores (19-bit)
                    tf32_was_enabled = None
                    if device == 'cuda' and TORCH_AVAILABLE:
                        tf32_was_enabled = torch.backends.cuda.matmul.allow_tf32
                        if prec_name == 'tf32':
                            # Enable TF32 for TF32 benchmark
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
                        elif prec_name == 'fp32':
                            # Disable TF32 for true FP32 benchmark
                            torch.backends.cuda.matmul.allow_tf32 = False
                            torch.backends.cudnn.allow_tf32 = False

                    # Add timeout for GEMM (level 3) to avoid very slow integer matmuls
                    if level == 3:
                        result = bench_fn(size, device=device, dtype=dtype, num_trials=num_trials, timeout_seconds=5)
                    else:
                        result = bench_fn(size, device=device, dtype=dtype, num_trials=num_trials)

                    # Restore TF32 setting
                    if tf32_was_enabled is not None:
                        torch.backends.cuda.matmul.allow_tf32 = tf32_was_enabled
                        torch.backends.cudnn.allow_tf32 = tf32_was_enabled

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
                    if enable_early_termination and result['gflops'] < min_useful_gflops and prec_name not in skipped_precisions:
                        size_display = f"{size_str}"
                        skipped_precisions[prec_name] = (f"{result['gflops']:.1f} GFLOPS < {min_useful_gflops} GFLOPS", size_display)

                except TimeoutError as e:
                    # Restore TF32 setting on error
                    if tf32_was_enabled is not None:
                        torch.backends.cuda.matmul.allow_tf32 = tf32_was_enabled
                        torch.backends.cudnn.allow_tf32 = tf32_was_enabled
                    # Benchmark timed out - mark as skipped for remaining sizes (GEMM only)
                    precision_results[prec_name] = PrecisionTestResult(
                        precision=prec_name,
                        supported=False,
                        failure_reason=f"Timeout: {str(e)}"
                    )
                    if enable_early_termination and prec_name not in skipped_precisions:
                        skipped_precisions[prec_name] = (f"Timeout (>{5}s)", size_str)

                except Exception as e:
                    # Restore TF32 setting on error
                    if tf32_was_enabled is not None:
                        torch.backends.cuda.matmul.allow_tf32 = tf32_was_enabled
                        torch.backends.cudnn.allow_tf32 = tf32_was_enabled
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
                    'framework': 'pytorch',
                    'device': device,
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
    import sys

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed")
        sys.exit(1)

    print("PyTorch BLAS Benchmark Suite")
    print("=" * 90)
    print()

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Quick test with smaller sizes
    calibrations = calibrate_blas_suite_pytorch(
        operations=['dot', 'axpy', 'gemv', 'gemm'],
        sizes={
            'dot':  [1000, 10000, 100000, 1000000],
            'axpy': [1000, 10000, 100000, 1000000],
            'gemv': [64, 128, 256, 512, 1024],
            'gemm': [64, 128, 256, 512, 1024],
        },
        theoretical_peak_gflops=720.0,
        device=device
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
