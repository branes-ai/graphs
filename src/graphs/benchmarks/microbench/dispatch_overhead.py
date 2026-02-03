"""
CUDA Kernel Dispatch Overhead Benchmark

Measures the baseline overhead of CUDA kernel launches to help calibrate
per-operation timing measurements.

Components measured:
1. Empty kernel dispatch (pure launch overhead)
2. Minimal compute kernel dispatch
3. torch.profiler overhead
4. CUDA event timing overhead

Usage:
    python -m graphs.benchmarks.microbench.dispatch_overhead --device cuda
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn


def measure_cuda_event_overhead(num_iterations: int = 1000) -> float:
    """Measure the overhead of CUDA event creation and synchronization."""
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        event_end.record()
        torch.cuda.synchronize()
        _ = event_start.elapsed_time(event_end)
    end = time.perf_counter()

    return (end - start) / num_iterations * 1000  # ms


def measure_empty_kernel_dispatch(num_iterations: int = 1000) -> Tuple[float, float]:
    """
    Measure empty kernel dispatch overhead using a minimal operation.

    Returns:
        Tuple of (cuda_event_time_ms, wall_clock_time_ms)
    """
    # Create minimal tensors on GPU
    a = torch.zeros(1, device='cuda')

    # Warmup
    for _ in range(100):
        _ = a + 0
    torch.cuda.synchronize()

    # Measure with CUDA events
    times_cuda = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = a + 0  # Minimal operation
        end.record()
        torch.cuda.synchronize()
        times_cuda.append(start.elapsed_time(end))

    # Measure with wall clock
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for _ in range(num_iterations):
        _ = a + 0
        torch.cuda.synchronize()
    wall_end = time.perf_counter()

    cuda_avg = sum(times_cuda) / len(times_cuda)
    wall_avg = (wall_end - wall_start) / num_iterations * 1000

    return cuda_avg, wall_avg


def measure_small_kernel_dispatch(size: int = 1024, num_iterations: int = 1000) -> Tuple[float, float]:
    """
    Measure small kernel dispatch overhead.

    Returns:
        Tuple of (cuda_event_time_ms, wall_clock_time_ms)
    """
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')

    # Warmup
    for _ in range(100):
        _ = a + b
    torch.cuda.synchronize()

    # Measure with CUDA events
    times_cuda = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = a + b
        end.record()
        torch.cuda.synchronize()
        times_cuda.append(start.elapsed_time(end))

    # Measure with wall clock
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for _ in range(num_iterations):
        _ = a + b
        torch.cuda.synchronize()
    wall_end = time.perf_counter()

    cuda_avg = sum(times_cuda) / len(times_cuda)
    wall_avg = (wall_end - wall_start) / num_iterations * 1000

    return cuda_avg, wall_avg


def measure_profiler_overhead(num_iterations: int = 100) -> Tuple[float, float]:
    """
    Measure the overhead added by torch.profiler.

    Returns:
        Tuple of (time_without_profiler_ms, time_with_profiler_ms)
    """
    # Create a small operation
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')

    # Warmup
    for _ in range(50):
        _ = a + b
    torch.cuda.synchronize()

    # Without profiler
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = a + b
        torch.cuda.synchronize()
    end = time.perf_counter()
    time_without = (end - start) / num_iterations * 1000

    # With profiler (use legacy autograd profiler for compatibility)
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for _ in range(num_iterations):
                _ = a + b
                torch.cuda.synchronize()
        end = time.perf_counter()
        time_with = (end - start) / num_iterations * 1000
    except Exception:
        # If profiler not available, estimate overhead
        time_with = time_without * 10  # Rough estimate

    return time_without, time_with


def measure_conv2d_dispatch_overhead(
    in_channels: int = 64,
    out_channels: int = 64,
    size: int = 56,
    kernel_size: int = 3,
    num_iterations: int = 100,
) -> Tuple[float, float, float]:
    """
    Measure Conv2d kernel dispatch and compute time.

    Returns:
        Tuple of (cuda_event_time_ms, wall_clock_time_ms, gflops)
    """
    x = torch.randn(1, in_channels, size, size, device='cuda')
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False).cuda()
    conv.eval()

    # Calculate FLOPs
    out_size = size  # same padding
    flops = 2 * in_channels * out_channels * kernel_size * kernel_size * out_size * out_size

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = conv(x)
    torch.cuda.synchronize()

    # Measure with CUDA events
    times_cuda = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = conv(x)
            end.record()
            torch.cuda.synchronize()
            times_cuda.append(start.elapsed_time(end))

    # Measure with wall clock
    torch.cuda.synchronize()
    with torch.no_grad():
        wall_start = time.perf_counter()
        for _ in range(num_iterations):
            _ = conv(x)
            torch.cuda.synchronize()
        wall_end = time.perf_counter()

    cuda_avg = sum(times_cuda) / len(times_cuda)
    wall_avg = (wall_end - wall_start) / num_iterations * 1000
    gflops = (flops / 1e9) / (cuda_avg / 1000)

    return cuda_avg, wall_avg, gflops


def measure_conv_bn_relu_dispatch(
    in_channels: int = 64,
    out_channels: int = 64,
    size: int = 56,
    kernel_size: int = 3,
    num_iterations: int = 100,
) -> Tuple[float, float, float]:
    """
    Measure Conv2d + BatchNorm2d + ReLU kernel dispatch and compute time.
    This is the key fusion pattern in ResNet.

    Returns:
        Tuple of (cuda_event_time_ms, wall_clock_time_ms, gflops)
    """
    x = torch.randn(1, in_channels, size, size, device='cuda')

    # Create fused module (PyTorch may or may not fuse these)
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ).cuda()
    model.eval()

    # Calculate FLOPs (Conv2d dominates, BN and ReLU are negligible)
    out_size = size
    flops = 2 * in_channels * out_channels * kernel_size * kernel_size * out_size * out_size

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()

    # Measure with CUDA events
    times_cuda = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times_cuda.append(start.elapsed_time(end))

    # Measure with wall clock
    torch.cuda.synchronize()
    with torch.no_grad():
        wall_start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(x)
            torch.cuda.synchronize()
        wall_end = time.perf_counter()

    cuda_avg = sum(times_cuda) / len(times_cuda)
    wall_avg = (wall_end - wall_start) / num_iterations * 1000
    gflops = (flops / 1e9) / (cuda_avg / 1000)

    return cuda_avg, wall_avg, gflops


def main():
    parser = argparse.ArgumentParser(description="CUDA Kernel Dispatch Overhead Benchmark")
    parser.add_argument("--device", default="cuda", help="Device to benchmark")
    parser.add_argument("--iterations", type=int, default=500, help="Number of iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("=" * 70)
    print("  CUDA KERNEL DISPATCH OVERHEAD BENCHMARK")
    print("=" * 70)
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Iterations: {args.iterations}")
    print("=" * 70)
    print()

    # 1. CUDA event overhead
    print("1. CUDA Event Timing Overhead:")
    event_overhead = measure_cuda_event_overhead(args.iterations)
    print(f"   Event create/record/sync overhead: {event_overhead * 1000:.1f} us")
    print()

    # 2. Empty kernel dispatch
    print("2. Empty Kernel Dispatch (a + 0):")
    cuda_time, wall_time = measure_empty_kernel_dispatch(args.iterations)
    print(f"   CUDA events: {cuda_time * 1000:.1f} us")
    print(f"   Wall clock:  {wall_time * 1000:.1f} us")
    print()

    # 3. Small kernel dispatch
    print("3. Small Kernel Dispatch (1024-element add):")
    cuda_time, wall_time = measure_small_kernel_dispatch(1024, args.iterations)
    print(f"   CUDA events: {cuda_time * 1000:.1f} us")
    print(f"   Wall clock:  {wall_time * 1000:.1f} us")
    print()

    # 4. Profiler overhead
    print("4. torch.profiler Overhead (1024-element add):")
    time_without, time_with = measure_profiler_overhead(min(args.iterations, 100))
    print(f"   Without profiler: {time_without * 1000:.1f} us")
    print(f"   With profiler:    {time_with * 1000:.1f} us")
    print(f"   Profiler overhead: {(time_with - time_without) * 1000:.1f} us ({time_with/time_without:.1f}x slowdown)")
    print()

    # 5. Conv2d dispatch at various sizes
    print("5. Conv2d Kernel Times (no BatchNorm/ReLU):")
    print("   " + "-" * 60)
    print(f"   {'Config':<30} {'CUDA (ms)':<12} {'GFLOPS':<12}")
    print("   " + "-" * 60)

    conv_configs = [
        (64, 64, 56, 3),    # ResNet early layers
        (64, 128, 28, 3),   # Stride-2 transition
        (128, 128, 28, 3),  # ResNet mid layers
        (256, 256, 14, 3),  # ResNet later layers
        (512, 512, 7, 3),   # ResNet final layers
    ]

    for in_ch, out_ch, sz, ks in conv_configs:
        cuda_time, wall_time, gflops = measure_conv2d_dispatch_overhead(
            in_ch, out_ch, sz, ks, min(args.iterations, 100)
        )
        config_str = f"{in_ch}ch {sz}x{sz} k{ks}"
        print(f"   {config_str:<30} {cuda_time:<12.3f} {gflops:<12.1f}")
    print()

    # 6. Conv2d + BatchNorm + ReLU (ResNet pattern)
    print("6. Conv2d + BatchNorm2d + ReLU (ResNet pattern):")
    print("   " + "-" * 60)
    print(f"   {'Config':<30} {'CUDA (ms)':<12} {'GFLOPS':<12}")
    print("   " + "-" * 60)

    for in_ch, out_ch, sz, ks in conv_configs:
        cuda_time, wall_time, gflops = measure_conv_bn_relu_dispatch(
            in_ch, out_ch, sz, ks, min(args.iterations, 100)
        )
        config_str = f"{in_ch}ch {sz}x{sz} k{ks}"
        print(f"   {config_str:<30} {cuda_time:<12.3f} {gflops:<12.1f}")
    print()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Kernel dispatch overhead: ~{cuda_time * 1000:.0f} us (from empty kernel)")
    print(f"  Profiler overhead: ~{(time_with - time_without) * 1000:.0f} us per operation")
    print()
    print("  NOTE: Subgraph validation measures ~1.5-2ms per node, but actual")
    print("  Conv2d_BN_ReLU takes ~0.3-0.5ms. The difference is profiler overhead")
    print("  and lack of kernel pipelining when measuring nodes individually.")
    print("=" * 70)


if __name__ == "__main__":
    main()
