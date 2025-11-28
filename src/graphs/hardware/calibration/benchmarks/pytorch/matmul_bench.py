"""
PyTorch Matrix Multiplication Benchmark (CPU or GPU)

PyTorch implementation with GPU support for performance characterization.
Reflects actual PyTorch/DL framework performance in Embodied AI applications.
"""

import time
from typing import List, Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ...schema import PrecisionTestResult
from ....resource_model import Precision


# Precision to PyTorch dtype mappings
# Note: TF32 uses float32 dtype but with Tensor Core truncation (19-bit)
# TF32 is enabled/disabled via torch.backends.cuda.matmul.allow_tf32
TORCH_DTYPE_MAP = {
    Precision.FP64: torch.float64 if TORCH_AVAILABLE else None,
    Precision.FP32: torch.float32 if TORCH_AVAILABLE else None,
    Precision.TF32: torch.float32 if TORCH_AVAILABLE else None,  # Same dtype, different mode
    Precision.FP16: torch.float16 if TORCH_AVAILABLE else None,
    Precision.BF16: torch.bfloat16 if TORCH_AVAILABLE else None,
    Precision.INT64: torch.int64 if TORCH_AVAILABLE else None,
    Precision.INT32: torch.int32 if TORCH_AVAILABLE else None,
    Precision.INT16: torch.int16 if TORCH_AVAILABLE else None,
    Precision.INT8: torch.int8 if TORCH_AVAILABLE else None,
}


def benchmark_torch_matmul(
    N: int,
    dtype,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    Benchmark PyTorch matmul at specific precision and device.

    Args:
        N: Matrix dimension
        dtype: PyTorch dtype
        device: 'cpu' or 'cuda'
        num_trials: Number of measurement runs
        num_warmup: Number of warmup runs

    Returns:
        Dict with timing and performance metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    # Create random matrices on device
    torch_device = torch.device(device)
    A = torch.randn(N, N, dtype=dtype, device=torch_device)
    B = torch.randn(N, N, dtype=dtype, device=torch_device)

    # Warmup
    for _ in range(num_warmup):
        C = torch.matmul(A, B)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        C = torch.matmul(A, B)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    import statistics
    mean_time_ms = statistics.mean(times)
    std_time_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time_ms = min(times)
    max_time_ms = max(times)

    # Calculate GFLOPS/GIOPS
    ops = 2.0 * N * N * N  # N^3 multiply-add operations
    mean_gops = ops / (mean_time_ms / 1000.0) / 1e9

    # Memory traffic (for arithmetic intensity)
    bytes_per_element = A.element_size()
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


def calibrate_matmul_pytorch(
    sizes: List[int],
    precisions: List[Precision],
    theoretical_peaks: Dict[str, float],
    device: str = 'cpu',
    num_trials: int = 10,
    min_useful_throughput: float = 50.0
) -> Dict[int, Dict[str, PrecisionTestResult]]:
    """
    Calibrate PyTorch matmul across precisions at multiple sizes.

    Args:
        sizes: Matrix sizes to test
        precisions: List of Precision enums to test
        theoretical_peaks: Dict mapping precision name -> theoretical GFLOPS/GIOPS
        device: 'cpu' or 'cuda'
        num_trials: Trials per test
        min_useful_throughput: Skip precision for larger sizes if throughput < this (GOPS)

    Returns:
        Dict mapping size -> Dict mapping precision name -> PrecisionTestResult
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PyTorch benchmarks")

    # Verify device is available
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Install PyTorch with CUDA support.")

    results = {}
    skip_precisions = set()

    # Print framework info
    print(f"Framework: PyTorch {torch.__version__}")
    if device == 'cuda':
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA:   {torch.version.cuda}")
    else:
        print(f"  Device: CPU")
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

            # Check if PyTorch supports this precision
            if precision not in TORCH_DTYPE_MAP or TORCH_DTYPE_MAP[precision] is None:
                print(f"  {precision.value:8s}... [X] UNSUPPORTED (not in PyTorch)")
                precision_results[precision.value] = PrecisionTestResult(
                    precision=precision.value,
                    supported=False,
                    failure_reason=f"{precision.value} not available in PyTorch",
                    test_size=N,
                    num_trials=0
                )
                continue

            dtype = TORCH_DTYPE_MAP[precision]

            # Print status
            print(f"  {precision.value:8s}...", end=" ", flush=True)

            # Run benchmark
            try:
                # Handle TF32 mode for CUDA devices
                # TF32 uses FP32 tensors but truncates mantissa on Tensor Cores
                tf32_was_enabled = None
                if device == 'cuda' and TORCH_AVAILABLE:
                    tf32_was_enabled = torch.backends.cuda.matmul.allow_tf32
                    if precision == Precision.TF32:
                        # Enable TF32 for TF32 benchmark
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    elif precision == Precision.FP32:
                        # Disable TF32 for true FP32 benchmark
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False

                result = benchmark_torch_matmul(N, dtype, device, num_trials)

                # Restore TF32 setting
                if tf32_was_enabled is not None:
                    torch.backends.cuda.matmul.allow_tf32 = tf32_was_enabled
                    torch.backends.cudnn.allow_tf32 = tf32_was_enabled

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
                    print(f"    Note: Likely caused by Turbo Boost, GPU boost clocks, or conservative theoretical peak")

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
    import sys

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed")
        sys.exit(1)

    print("PyTorch Matrix Multiplication Calibration")
    print("=" * 70)

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    precisions_to_test = [
        Precision.FP32,
        Precision.FP16,
    ]

    theoretical_peaks = {
        'fp32': 1280.0,  # Example: Jetson Orin Nano
        'fp16': 2560.0,
    }

    results = calibrate_matmul_pytorch(
        sizes=[1024],
        precisions=precisions_to_test,
        theoretical_peaks=theoretical_peaks,
        device=device,
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
