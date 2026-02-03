"""
Conv2d + BatchNorm2d Fusion Microbenchmark for ResNet-18 Layers

Measures the exact layer configurations from ResNet-18 to calibrate
the estimation model. Uses CUDA events for accurate timing.

Usage:
    python -m graphs.benchmarks.microbench.conv_bn_resnet --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class ResNetLayerConfig:
    """Configuration for a ResNet conv layer."""
    name: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    input_size: int  # H=W
    flops: int  # Computed FLOPs
    batch_size: int = 1

    @property
    def output_size(self) -> int:
        return (self.input_size + 2 * self.padding - self.kernel_size) // self.stride + 1


def get_resnet18_layer_configs(batch_size: int = 1) -> List[ResNetLayerConfig]:
    """Get all unique layer configurations from ResNet-18."""
    configs = []

    # Initial 7x7 conv (conv1)
    configs.append(ResNetLayerConfig(
        name="conv1_7x7",
        in_channels=3, out_channels=64,
        kernel_size=7, stride=2, padding=3,
        input_size=224,
        flops=2 * 3 * 64 * 7 * 7 * 112 * 112,  # 236M FLOPs
        batch_size=batch_size,
    ))

    # Layer 1: 64 -> 64, 56x56, stride 1 (4 conv layers)
    configs.append(ResNetLayerConfig(
        name="layer1_3x3_64",
        in_channels=64, out_channels=64,
        kernel_size=3, stride=1, padding=1,
        input_size=56,
        flops=2 * 64 * 64 * 3 * 3 * 56 * 56,  # 231M FLOPs
        batch_size=batch_size,
    ))

    # Layer 2: 64 -> 128 (stride 2) and 128 -> 128 (stride 1)
    configs.append(ResNetLayerConfig(
        name="layer2_3x3_64to128_s2",
        in_channels=64, out_channels=128,
        kernel_size=3, stride=2, padding=1,
        input_size=56,
        flops=2 * 64 * 128 * 3 * 3 * 28 * 28,  # 115.6M FLOPs
        batch_size=batch_size,
    ))
    configs.append(ResNetLayerConfig(
        name="layer2_3x3_128",
        in_channels=128, out_channels=128,
        kernel_size=3, stride=1, padding=1,
        input_size=28,
        flops=2 * 128 * 128 * 3 * 3 * 28 * 28,  # 231M FLOPs
        batch_size=batch_size,
    ))
    # 1x1 downsample for residual
    configs.append(ResNetLayerConfig(
        name="layer2_1x1_64to128_s2",
        in_channels=64, out_channels=128,
        kernel_size=1, stride=2, padding=0,
        input_size=56,
        flops=2 * 64 * 128 * 1 * 1 * 28 * 28,  # 12.8M FLOPs
        batch_size=batch_size,
    ))

    # Layer 3: 128 -> 256 (stride 2) and 256 -> 256 (stride 1)
    configs.append(ResNetLayerConfig(
        name="layer3_3x3_128to256_s2",
        in_channels=128, out_channels=256,
        kernel_size=3, stride=2, padding=1,
        input_size=28,
        flops=2 * 128 * 256 * 3 * 3 * 14 * 14,  # 115.6M FLOPs
        batch_size=batch_size,
    ))
    configs.append(ResNetLayerConfig(
        name="layer3_3x3_256",
        in_channels=256, out_channels=256,
        kernel_size=3, stride=1, padding=1,
        input_size=14,
        flops=2 * 256 * 256 * 3 * 3 * 14 * 14,  # 231M FLOPs
        batch_size=batch_size,
    ))
    configs.append(ResNetLayerConfig(
        name="layer3_1x1_128to256_s2",
        in_channels=128, out_channels=256,
        kernel_size=1, stride=2, padding=0,
        input_size=28,
        flops=2 * 128 * 256 * 1 * 1 * 14 * 14,  # 12.8M FLOPs
        batch_size=batch_size,
    ))

    # Layer 4: 256 -> 512 (stride 2) and 512 -> 512 (stride 1)
    configs.append(ResNetLayerConfig(
        name="layer4_3x3_256to512_s2",
        in_channels=256, out_channels=512,
        kernel_size=3, stride=2, padding=1,
        input_size=14,
        flops=2 * 256 * 512 * 3 * 3 * 7 * 7,  # 115.6M FLOPs
        batch_size=batch_size,
    ))
    configs.append(ResNetLayerConfig(
        name="layer4_3x3_512",
        in_channels=512, out_channels=512,
        kernel_size=3, stride=1, padding=1,
        input_size=7,
        flops=2 * 512 * 512 * 3 * 3 * 7 * 7,  # 231M FLOPs
        batch_size=batch_size,
    ))
    configs.append(ResNetLayerConfig(
        name="layer4_1x1_256to512_s2",
        in_channels=256, out_channels=512,
        kernel_size=1, stride=2, padding=0,
        input_size=14,
        flops=2 * 256 * 512 * 1 * 1 * 7 * 7,  # 12.8M FLOPs
        batch_size=batch_size,
    ))

    return configs


def benchmark_conv_bn(
    config: ResNetLayerConfig,
    device: str = 'cuda',
    with_relu: bool = True,
    warmup: int = 50,
    iterations: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark Conv2d + BatchNorm2d (+ ReLU) for a specific configuration.

    Returns:
        Tuple of (time_ms, gflops)
    """
    # Create input
    x = torch.randn(
        config.batch_size, config.in_channels,
        config.input_size, config.input_size,
        device=device,
    )

    # Create model
    layers = [
        nn.Conv2d(
            config.in_channels, config.out_channels,
            config.kernel_size, stride=config.stride,
            padding=config.padding, bias=False,
        ),
        nn.BatchNorm2d(config.out_channels),
    ]
    if with_relu:
        layers.append(nn.ReLU(inplace=True))

    model = nn.Sequential(*layers).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    gflops = (config.flops / 1e9) / (avg_time / 1000)

    return avg_time, gflops


def benchmark_conv_only(
    config: ResNetLayerConfig,
    device: str = 'cuda',
    warmup: int = 50,
    iterations: int = 100,
) -> Tuple[float, float]:
    """Benchmark Conv2d alone (no BN, no ReLU)."""
    x = torch.randn(
        config.batch_size, config.in_channels,
        config.input_size, config.input_size,
        device=device,
    )

    conv = nn.Conv2d(
        config.in_channels, config.out_channels,
        config.kernel_size, stride=config.stride,
        padding=config.padding, bias=False,
    ).to(device)
    conv.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = conv(x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = conv(x)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    gflops = (config.flops / 1e9) / (avg_time / 1000)

    return avg_time, gflops


def format_flops(flops: int) -> str:
    if flops >= 1e9:
        return f"{flops/1e9:.1f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.1f}M"
    else:
        return f"{flops/1e3:.1f}K"


def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 Conv2d+BatchNorm2d Calibration Benchmark"
    )
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Measurement iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available")
        return

    configs = get_resnet18_layer_configs(args.batch_size)

    print("=" * 90)
    print("  RESNET-18 CONV2D + BATCHNORM2D CALIBRATION BENCHMARK")
    print("=" * 90)
    print(f"  Device: {torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 90)
    print()

    # Table 1: Conv2d + BN + ReLU (fused estimation target)
    print("1. Conv2d + BatchNorm2d + ReLU (fused, used for estimation):")
    print("-" * 90)
    print(f"  {'Layer':<28} {'FLOPs':<10} {'Size':<12} {'Time (ms)':<12} {'GFLOPS':<10}")
    print("-" * 90)

    total_flops = 0
    total_time = 0
    for cfg in configs:
        time_ms, gflops = benchmark_conv_bn(
            cfg, args.device, with_relu=True,
            warmup=args.warmup, iterations=args.iterations
        )
        size_str = f"{cfg.in_channels}->{cfg.out_channels} {cfg.input_size}x{cfg.input_size}"
        print(f"  {cfg.name:<28} {format_flops(cfg.flops):<10} {size_str:<12} "
              f"{time_ms:<12.3f} {gflops:<10.1f}")
        total_flops += cfg.flops
        total_time += time_ms

    print("-" * 90)
    avg_gflops = (total_flops / 1e9) / (total_time / 1000)
    print(f"  {'TOTAL':<28} {format_flops(total_flops):<10} {'':<12} "
          f"{total_time:<12.3f} {avg_gflops:<10.1f}")
    print()

    # Table 2: Conv2d only (for comparison)
    print("2. Conv2d only (no BatchNorm, no ReLU):")
    print("-" * 90)
    print(f"  {'Layer':<28} {'FLOPs':<10} {'Size':<12} {'Time (ms)':<12} {'GFLOPS':<10}")
    print("-" * 90)

    total_flops = 0
    total_time = 0
    for cfg in configs:
        time_ms, gflops = benchmark_conv_only(
            cfg, args.device,
            warmup=args.warmup, iterations=args.iterations
        )
        size_str = f"{cfg.in_channels}->{cfg.out_channels} {cfg.input_size}x{cfg.input_size}"
        print(f"  {cfg.name:<28} {format_flops(cfg.flops):<10} {size_str:<12} "
              f"{time_ms:<12.3f} {gflops:<10.1f}")
        total_flops += cfg.flops
        total_time += time_ms

    print("-" * 90)
    avg_gflops = (total_flops / 1e9) / (total_time / 1000)
    print(f"  {'TOTAL':<28} {format_flops(total_flops):<10} {'':<12} "
          f"{total_time:<12.3f} {avg_gflops:<10.1f}")
    print()

    # Summary
    print("=" * 90)
    print("  CALIBRATION SUMMARY")
    print("=" * 90)
    print()
    print("  Use these GFLOPS values to calibrate the roofline model for ResNet-18.")
    print("  The Conv2d+BN+ReLU values represent realistic fused execution performance.")
    print()
    print("  Expected estimation approach:")
    print("  - Sum of individual layer times from Table 1")
    print("  - Full model will be faster due to pipelining/overlap (~0.7x)")
    print()


if __name__ == "__main__":
    main()
