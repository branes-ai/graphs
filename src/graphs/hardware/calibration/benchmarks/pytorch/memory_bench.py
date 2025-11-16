"""
PyTorch Memory Bandwidth Benchmark (CPU or GPU)

PyTorch implementation with GPU support for memory bandwidth characterization.
Reflects actual PyTorch/DL framework memory performance.
"""

import time
from typing import Dict, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ...schema import OperationCalibration, OperationType


def benchmark_memory_copy_pytorch(
    size_mb: int,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    Benchmark PyTorch memory copy bandwidth (CPU or GPU).

    Args:
        size_mb: Size of array to copy (in megabytes)
        device: 'cpu' or 'cuda'
        num_trials: Number of measurement runs
        num_warmup: Number of warmup runs

    Returns:
        Dict with bandwidth and timing metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    # Create arrays on device
    torch_device = torch.device(device)
    num_elements = (size_mb * 1024 * 1024) // 4  # Float32 = 4 bytes
    src = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    dst = torch.zeros(num_elements, dtype=torch.float32, device=torch_device)

    # Warmup
    for _ in range(num_warmup):
        dst.copy_(src)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        dst.copy_(src)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Statistics
    import statistics
    mean_time_ms = statistics.mean(times)
    std_time_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time_ms = min(times)
    max_time_ms = max(times)

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


def calibrate_memory_bandwidth_pytorch(
    sizes_mb: List[int] = [64, 128, 256, 512],
    theoretical_bandwidth_gbps: float = 75.0,
    device: str = 'cpu',
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Calibrate PyTorch memory bandwidth at various buffer sizes (CPU or GPU).

    Args:
        sizes_mb: List of buffer sizes to test (in MB)
        theoretical_bandwidth_gbps: Theoretical peak bandwidth
        device: 'cpu' or 'cuda'
        num_trials: Number of trials per size

    Returns:
        List of OperationCalibration objects for memory operations
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PyTorch benchmarks")

    # Verify device is available
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Install PyTorch with CUDA support.")

    calibrations = []

    # Print framework info
    print(f"Framework: PyTorch {torch.__version__}")
    if device == 'cuda':
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA:   {torch.version.cuda}")
    else:
        print(f"  Device: CPU")
    print()

    for size_mb in sizes_mb:
        print(f"Calibrating memory copy {size_mb} MB...", end=" ", flush=True)

        result = benchmark_memory_copy_pytorch(size_mb, device=device, num_trials=num_trials)

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
                'framework': 'pytorch',
                'device': device,
            }
        )

        calibrations.append(calibration)

        print(f"{result['bandwidth_gbps']:.1f} GB/s "
              f"({calibration.efficiency*100:.1f}% efficiency)")

    return calibrations


if __name__ == "__main__":
    # Standalone test
    import sys

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed")
        sys.exit(1)

    print("PyTorch Memory Bandwidth Calibration")
    print("=" * 60)

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    calibrations = calibrate_memory_bandwidth_pytorch(
        sizes_mb=[64, 128, 256],
        theoretical_bandwidth_gbps=68.0,  # Example: Jetson Orin Nano
        device=device
    )

    print("\nResults:")
    print(f"{'Size (MB)':<12} {'Bandwidth':>12} {'Efficiency':>12}")
    print("-" * 40)

    for cal in calibrations:
        size = cal.extra_params['size_mb']
        print(f"{size:<12} {cal.achieved_bandwidth_gbps:>11.1f} GB/s "
              f"{cal.efficiency*100:>11.1f}%")
