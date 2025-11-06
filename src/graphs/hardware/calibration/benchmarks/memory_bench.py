"""
Memory bandwidth calibration benchmark.

Measures achievable memory bandwidth using simple copy operations.
This establishes the baseline for memory-bound operation performance.
"""

import numpy as np
import time
from typing import Dict, List

from ..schema import OperationCalibration, OperationType


def benchmark_memory_copy(size_mb: int, num_trials: int = 10, num_warmup: int = 3) -> Dict:
    """
    Benchmark memory copy bandwidth.

    Args:
        size_mb: Size of array to copy (in megabytes)
        num_trials: Number of measurement runs
        num_warmup: Number of warmup runs

    Returns:
        Dict with bandwidth and timing metrics
    """
    # Create arrays
    num_elements = (size_mb * 1024 * 1024) // 4  # Float32 = 4 bytes
    src = np.random.rand(num_elements).astype(np.float32)
    dst = np.zeros(num_elements, dtype=np.float32)

    # Warmup
    for _ in range(num_warmup):
        np.copyto(dst, src)

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        np.copyto(dst, src)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Statistics
    mean_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)

    # Bandwidth (read + write)
    bytes_transferred = 2 * num_elements * 4  # Read src, write dst
    bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9

    return {
        'size_mb': size_mb,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'max_latency_ms': max_time_ms,
        'bandwidth_gbps': bandwidth_gbps,
        'num_trials': num_trials,
    }


def calibrate_memory_bandwidth(
    sizes_mb: List[int] = [64, 128, 256, 512],
    theoretical_bandwidth_gbps: float = 75.0,
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Calibrate memory bandwidth at various buffer sizes.

    Args:
        sizes_mb: List of buffer sizes to test (in MB)
        theoretical_bandwidth_gbps: Theoretical peak bandwidth
        num_trials: Number of trials per size

    Returns:
        List of OperationCalibration objects for memory operations
    """
    calibrations = []

    for size_mb in sizes_mb:
        print(f"Calibrating memory copy {size_mb} MB...", end=" ", flush=True)

        result = benchmark_memory_copy(size_mb, num_trials=num_trials)

        # Create calibration (using ADD as representative elementwise op)
        calibration = OperationCalibration(
            operation_type=OperationType.ADD.value,
            measured_gflops=0.0,  # Not compute-intensive
            efficiency=result['bandwidth_gbps'] / theoretical_bandwidth_gbps,
            achieved_bandwidth_gbps=result['bandwidth_gbps'],
            memory_bound=True,
            compute_bound=False,
            arithmetic_intensity=0.0,  # Pure memory operation
            batch_size=1,
            input_shape=(size_mb * 1024 * 256,),  # Float32 count
            output_shape=(size_mb * 1024 * 256,),
            mean_latency_ms=result['mean_latency_ms'],
            std_latency_ms=result['std_latency_ms'],
            min_latency_ms=result['min_latency_ms'],
            max_latency_ms=result['max_latency_ms'],
            num_trials=result['num_trials'],
            extra_params={
                'size_mb': size_mb,
                'operation': 'memory_copy',
            }
        )

        calibrations.append(calibration)

        print(f"{result['bandwidth_gbps']:.1f} GB/s "
              f"({calibration.efficiency*100:.1f}% efficiency)")

    return calibrations


if __name__ == "__main__":
    # Standalone test
    print("Memory Bandwidth Calibration")
    print("=" * 60)

    calibrations = calibrate_memory_bandwidth(
        sizes_mb=[64, 128, 256, 512],
        theoretical_bandwidth_gbps=75.0
    )

    print("\nResults:")
    print(f"{'Size (MB)':<12} {'Bandwidth':>12} {'Efficiency':>12}")
    print("-" * 40)

    for cal in calibrations:
        size = cal.extra_params['size_mb']
        print(f"{size:<12} {cal.achieved_bandwidth_gbps:>11.1f} GB/s "
              f"{cal.efficiency*100:>11.1f}%")
