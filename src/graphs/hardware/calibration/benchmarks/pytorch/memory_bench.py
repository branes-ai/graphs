"""
PyTorch STREAM Benchmark Suite (CPU or GPU)

Complete STREAM benchmark implementation with GPU support for memory bandwidth characterization.
Includes all four standard STREAM kernels:
  - Copy:  a[i] = b[i]
  - Scale: a[i] = q * b[i]
  - Add:   a[i] = b[i] + c[i]
  - Triad: a[i] = b[i] + q * c[i]

Reflects actual PyTorch/DL framework memory performance.
"""

import time
from typing import Dict, List, Callable
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ...schema import OperationCalibration, OperationType


@dataclass
class StreamKernelSpec:
    """Specification for a STREAM kernel"""
    name: str                      # "copy", "scale", "add", "triad"
    operation_type: OperationType  # Enum value
    num_arrays: int                # Number of input/output arrays
    num_memory_ops: int            # Total memory operations (reads + writes)
    num_flops: int                 # FLOPs per element
    description: str               # Human-readable description


# STREAM kernel specifications (identical to NumPy version)
STREAM_KERNELS = {
    'copy': StreamKernelSpec(
        name='copy',
        operation_type=OperationType.STREAM_COPY,
        num_arrays=2,          # src, dst
        num_memory_ops=2,      # 1 read, 1 write
        num_flops=0,           # No FLOPs
        description='a[i] = b[i]'
    ),
    'scale': StreamKernelSpec(
        name='scale',
        operation_type=OperationType.STREAM_SCALE,
        num_arrays=2,          # src, dst
        num_memory_ops=2,      # 1 read, 1 write
        num_flops=1,           # 1 multiply per element
        description='a[i] = q * b[i]'
    ),
    'add': StreamKernelSpec(
        name='add',
        operation_type=OperationType.STREAM_ADD,
        num_arrays=3,          # src1, src2, dst
        num_memory_ops=3,      # 2 reads, 1 write
        num_flops=1,           # 1 add per element
        description='a[i] = b[i] + c[i]'
    ),
    'triad': StreamKernelSpec(
        name='triad',
        operation_type=OperationType.STREAM_TRIAD,
        num_arrays=3,          # src1, src2, dst
        num_memory_ops=3,      # 2 reads, 1 write
        num_flops=2,           # 1 multiply + 1 add per element
        description='a[i] = b[i] + q * c[i]'
    ),
}


def _benchmark_kernel(
    kernel_fn: Callable,
    kernel_spec: StreamKernelSpec,
    size_mb: int,
    device: str,
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    Generic benchmark runner for STREAM kernels.

    Args:
        kernel_fn: Function that executes the kernel
        kernel_spec: Kernel specification
        size_mb: Size of arrays to allocate (in megabytes)
        device: 'cpu' or 'cuda'
        num_trials: Number of measurement runs
        num_warmup: Number of warmup runs

    Returns:
        Dict with timing and bandwidth metrics
    """
    # Warmup
    for _ in range(num_warmup):
        kernel_fn()
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        kernel_fn()

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

    # Calculate bandwidth
    num_elements = (size_mb * 1024 * 1024) // 4  # Float32 = 4 bytes
    bytes_transferred = kernel_spec.num_memory_ops * num_elements * 4
    bandwidth_gbps = (bytes_transferred / (mean_time_ms / 1000.0)) / 1e9

    # Calculate FLOPs (if applicable)
    if kernel_spec.num_flops > 0:
        total_flops = kernel_spec.num_flops * num_elements
        gflops = (total_flops / (mean_time_ms / 1000.0)) / 1e9
    else:
        gflops = 0.0

    # Arithmetic intensity (FLOPs per byte)
    arithmetic_intensity = (kernel_spec.num_flops * 4) / (kernel_spec.num_memory_ops * 4) if kernel_spec.num_memory_ops > 0 else 0.0

    return {
        'kernel_name': kernel_spec.name,
        'size_mb': size_mb,
        'mean_latency_ms': mean_time_ms,
        'std_latency_ms': std_time_ms,
        'min_latency_ms': min_time_ms,
        'max_latency_ms': max_time_ms,
        'bandwidth_gbps': bandwidth_gbps,
        'gflops': gflops,
        'arithmetic_intensity': arithmetic_intensity,
        'num_trials': num_trials,
        'num_arrays': kernel_spec.num_arrays,
        'num_memory_ops': kernel_spec.num_memory_ops,
        'num_flops_per_element': kernel_spec.num_flops,
    }


def benchmark_stream_copy_pytorch(
    size_mb: int,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    STREAM Copy: a[i] = b[i]

    Pure memory bandwidth test with no arithmetic.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    kernel_spec = STREAM_KERNELS['copy']
    num_elements = (size_mb * 1024 * 1024) // 4

    # Allocate arrays on device
    torch_device = torch.device(device)
    src = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    dst = torch.zeros(num_elements, dtype=torch.float32, device=torch_device)

    # Kernel function
    def kernel():
        dst.copy_(src)

    return _benchmark_kernel(kernel, kernel_spec, size_mb, device, num_trials, num_warmup)


def benchmark_stream_scale_pytorch(
    size_mb: int,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    STREAM Scale: a[i] = q * b[i]

    Memory bandwidth with scalar multiplication.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    kernel_spec = STREAM_KERNELS['scale']
    num_elements = (size_mb * 1024 * 1024) // 4
    scalar = 3.14159  # Standard STREAM scalar

    # Allocate arrays on device
    torch_device = torch.device(device)
    src = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    dst = torch.zeros(num_elements, dtype=torch.float32, device=torch_device)

    # Kernel function
    def kernel():
        torch.mul(src, scalar, out=dst)

    return _benchmark_kernel(kernel, kernel_spec, size_mb, device, num_trials, num_warmup)


def benchmark_stream_add_pytorch(
    size_mb: int,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    STREAM Add: a[i] = b[i] + c[i]

    Memory bandwidth with vector addition (3 arrays).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    kernel_spec = STREAM_KERNELS['add']
    num_elements = (size_mb * 1024 * 1024) // 4

    # Allocate arrays on device
    torch_device = torch.device(device)
    src1 = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    src2 = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    dst = torch.zeros(num_elements, dtype=torch.float32, device=torch_device)

    # Kernel function
    def kernel():
        torch.add(src1, src2, out=dst)

    return _benchmark_kernel(kernel, kernel_spec, size_mb, device, num_trials, num_warmup)


def benchmark_stream_triad_pytorch(
    size_mb: int,
    device: str = 'cpu',
    num_trials: int = 10,
    num_warmup: int = 3
) -> Dict:
    """
    STREAM Triad: a[i] = b[i] + q * c[i]

    Memory bandwidth with fused multiply-add (most complex STREAM kernel).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    kernel_spec = STREAM_KERNELS['triad']
    num_elements = (size_mb * 1024 * 1024) // 4
    scalar = 3.14159  # Standard STREAM scalar

    # Allocate arrays on device
    torch_device = torch.device(device)
    src1 = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    src2 = torch.randn(num_elements, dtype=torch.float32, device=torch_device)
    dst = torch.zeros(num_elements, dtype=torch.float32, device=torch_device)

    # Kernel function
    def kernel():
        # a = b + q * c
        # torch.addcmul(input, tensor1, tensor2, value=scalar) computes: input + value * tensor1 * tensor2
        # We want: src1 + scalar * src2, so we need a workaround
        # Using direct computation for clarity and correctness
        torch.mul(src2, scalar, out=dst)  # dst = scalar * src2
        torch.add(src1, dst, out=dst)     # dst = src1 + dst

    return _benchmark_kernel(kernel, kernel_spec, size_mb, device, num_trials, num_warmup)


def calibrate_stream_bandwidth_pytorch(
    kernels: List[str] = ['copy', 'scale', 'add', 'triad'],
    sizes_mb: List[int] = [8, 16, 32, 64, 128, 256, 512],
    theoretical_bandwidth_gbps: float = 75.0,
    device: str = 'cpu',
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Run full STREAM benchmark suite (PyTorch, CPU or GPU).

    Args:
        kernels: List of STREAM kernels to run (default: all)
        sizes_mb: List of buffer sizes to test (in MB)
        theoretical_bandwidth_gbps: Theoretical peak memory bandwidth
        device: 'cpu' or 'cuda'
        num_trials: Number of trials per kernel/size

    Returns:
        List of OperationCalibration objects for each kernel/size combination
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PyTorch benchmarks")

    # Verify device is available
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Install PyTorch with CUDA support.")

    # Kernel dispatch table
    kernel_functions = {
        'copy': benchmark_stream_copy_pytorch,
        'scale': benchmark_stream_scale_pytorch,
        'add': benchmark_stream_add_pytorch,
        'triad': benchmark_stream_triad_pytorch,
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

    print("STREAM Benchmark Suite:")
    print("-" * 80)

    for kernel_name in kernels:
        if kernel_name not in kernel_functions:
            print(f"[!] Warning: Unknown kernel '{kernel_name}', skipping")
            continue

        kernel_spec = STREAM_KERNELS[kernel_name]
        kernel_fn = kernel_functions[kernel_name]

        print(f"\n{kernel_name.upper()} ({kernel_spec.description}):")
        print(f"  Memory ops: {kernel_spec.num_memory_ops}, FLOPs/element: {kernel_spec.num_flops}")

        for size_mb in sizes_mb:
            print(f"  Size {size_mb:>5} MB...", end=" ", flush=True)

            result = kernel_fn(size_mb, device=device, num_trials=num_trials)

            # Create calibration object
            calibration = OperationCalibration(
                operation_type=kernel_spec.operation_type.value,
                measured_gflops=result['gflops'],
                efficiency=result['bandwidth_gbps'] / theoretical_bandwidth_gbps,
                achieved_bandwidth_gbps=result['bandwidth_gbps'],
                memory_bound=True,
                compute_bound=False,
                arithmetic_intensity=result['arithmetic_intensity'],
                batch_size=1,
                input_shape=(size_mb * 1024 * 256,),  # Float32 element count
                output_shape=(size_mb * 1024 * 256,),
                mean_latency_ms=result['mean_latency_ms'],
                std_latency_ms=result['std_latency_ms'],
                min_latency_ms=result['min_latency_ms'],
                max_latency_ms=result['max_latency_ms'],
                num_trials=result['num_trials'],
                extra_params={
                    'size_mb': size_mb,
                    'kernel': kernel_name,
                    'framework': 'pytorch',
                    'device': device,
                    'num_arrays': result['num_arrays'],
                    'num_memory_ops': result['num_memory_ops'],
                    'flops_per_element': result['num_flops_per_element'],
                }
            )

            calibrations.append(calibration)

            # Print results with latency
            latency_ms = result['mean_latency_ms']
            bw_str = f"{result['bandwidth_gbps']:.1f} GB/s"
            lat_str = f"{latency_ms:.2f} ms"
            eff_str = f"({calibration.efficiency*100:.1f}%)"

            if result['gflops'] > 0:
                flops_str = f"{result['gflops']:.1f} GFLOPS"
                print(f"{bw_str:>12} {eff_str:>8}  {lat_str:>9}  |  {flops_str}")
            else:
                print(f"{bw_str:>12} {eff_str:>8}  {lat_str:>9}")

    print()

    # Print STREAM summary
    if calibrations:
        print("=" * 80)
        print("STREAM Summary:")
        print("-" * 80)

        # Group by kernel
        by_kernel = {}
        for cal in calibrations:
            kernel = cal.extra_params['kernel']
            if kernel not in by_kernel:
                by_kernel[kernel] = []
            by_kernel[kernel].append(cal)

        # Print best bandwidth for each kernel with latency at best bandwidth
        print(f"{'Kernel':<12} {'Best BW':>12} {'Latency':>10} {'Efficiency':>12} {'Description'}")
        print("-" * 90)

        for kernel_name in ['copy', 'scale', 'add', 'triad']:
            if kernel_name in by_kernel:
                kernel_cals = by_kernel[kernel_name]
                # Find calibration with best bandwidth
                best_cal = max(kernel_cals, key=lambda c: c.achieved_bandwidth_gbps)
                best_bw = best_cal.achieved_bandwidth_gbps
                best_eff = best_cal.efficiency
                best_lat = best_cal.mean_latency_ms
                spec = STREAM_KERNELS[kernel_name]

                print(f"{kernel_name.upper():<12} {best_bw:>10.1f} GB/s {best_lat:>8.2f} ms {best_eff*100:>10.1f}%  {spec.description}")

        # STREAM score (minimum bandwidth across all kernels)
        all_bandwidths = [c.achieved_bandwidth_gbps for c in calibrations]
        stream_score = min(all_bandwidths)

        print()
        print(f"STREAM Score (minimum): {stream_score:.1f} GB/s")
        print("=" * 80)

    return calibrations


# Backward compatibility: keep old function name
def calibrate_memory_bandwidth_pytorch(
    sizes_mb: List[int] = [8, 16, 32, 64, 128, 256, 512],
    theoretical_bandwidth_gbps: float = 75.0,
    device: str = 'cpu',
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Legacy function for backward compatibility.
    Now runs full STREAM suite instead of just copy.
    """
    return calibrate_stream_bandwidth_pytorch(
        kernels=['copy', 'scale', 'add', 'triad'],
        sizes_mb=sizes_mb,
        theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
        device=device,
        num_trials=num_trials
    )


if __name__ == "__main__":
    # Standalone test
    import sys

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed")
        sys.exit(1)

    print("PyTorch STREAM Benchmark")
    print("=" * 80)
    print()

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    calibrations = calibrate_stream_bandwidth_pytorch(
        kernels=['copy', 'scale', 'add', 'triad'],
        sizes_mb=[8, 16, 32, 64, 128, 256],
        theoretical_bandwidth_gbps=68.0,  # Example: Jetson Orin Nano
        device=device
    )

    print("\nDetailed Results:")
    print(f"{'Kernel':<12} {'Size (MB)':<12} {'Bandwidth':>12} {'Latency':>10} {'Efficiency':>12}")
    print("-" * 70)

    for cal in calibrations:
        kernel = cal.extra_params['kernel']
        size = cal.extra_params['size_mb']
        print(f"{kernel.upper():<12} {size:<12} {cal.achieved_bandwidth_gbps:>11.1f} GB/s "
              f"{cal.mean_latency_ms:>8.2f} ms {cal.efficiency*100:>11.1f}%")
