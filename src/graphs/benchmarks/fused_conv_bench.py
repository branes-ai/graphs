"""
Fused Convolution Benchmarks

Benchmarks common convolution fusion patterns:
- Conv2d + Bias
- Conv2d + BatchNorm
- Conv2d + BatchNorm + ReLU (ResNet block)
- Conv2d + ReLU (simple fusion)

These are the most common patterns in CNNs (ResNet, VGG, MobileNet).
"""

import time
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def benchmark_conv2d_bn(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    image_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 50
) -> Dict:
    """
    Benchmark Conv2d + BatchNorm fusion.

    This is the core ResNet pattern (without ReLU).
    """
    # Create inputs
    X = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    # ========================================================================
    # UNFUSED VERSION (2 separate modules)
    # ========================================================================
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    bn = nn.BatchNorm2d(out_channels, device=device)
    conv.eval()
    bn.eval()

    def unfused_forward(X):
        Y = conv(X)
        Y = bn(Y)
        return Y

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = unfused_forward(X)

    # Benchmark unfused
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            Y_unfused = unfused_forward(X)
        if device == 'cuda':
            torch.cuda.synchronize()
        unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION (fold BatchNorm into Conv weights)
    # ========================================================================
    # During inference, BatchNorm can be folded into Conv2d weights/bias
    # This is standard optimization in deployment
    conv_fused = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, device=device)
    conv_fused.eval()

    # Fold BatchNorm parameters into Conv
    with torch.no_grad():
        # BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
        # Can be folded into conv: y = conv(x) * scale + bias
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_eps = bn.eps

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        bias = bn_bias - bn_mean * scale

        # Fold into conv weights
        conv_fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        conv_fused.bias.copy_(bias)

    def fused_forward(X):
        return conv_fused(X)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = fused_forward(X)

    # Benchmark fused
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            Y_fused = fused_forward(X)
        if device == 'cuda':
            torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    # Output spatial size
    out_size = (image_size + 2 * padding - kernel_size) // stride + 1

    # FLOPs for Conv2d: 2 * kernel_size^2 * in_channels * out_channels * out_size^2 * batch_size
    conv_flops = 2 * kernel_size * kernel_size * in_channels * out_channels * out_size * out_size * batch_size

    # FLOPs for BatchNorm: 2 ops per element (subtract mean, divide by std)
    bn_flops = 2 * out_channels * out_size * out_size * batch_size

    total_flops = conv_flops + bn_flops

    # Memory (unfused):
    #   Conv: read input, weights, write output
    #   BN: read output, write output
    unfused_bytes = (batch_size * in_channels * image_size * image_size +  # input
                    out_channels * in_channels * kernel_size * kernel_size +  # conv weights
                    batch_size * out_channels * out_size * out_size +  # conv output (write)
                    batch_size * out_channels * out_size * out_size +  # bn read
                    batch_size * out_channels * out_size * out_size) * 4  # bn write

    # Memory (fused): read input, weights (with bias), write output
    fused_bytes = (batch_size * in_channels * image_size * image_size +
                  out_channels * in_channels * kernel_size * kernel_size +
                  out_channels +  # bias
                  batch_size * out_channels * out_size * out_size) * 4

    # GFLOPS
    unfused_gflops = total_flops / unfused_time / 1e9
    fused_gflops = total_flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Conv2d_BN',
        'input_shape': (batch_size, in_channels, image_size, image_size),
        'output_shape': (batch_size, out_channels, out_size, out_size),
        'kernel_size': kernel_size,
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': total_flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }


def benchmark_conv2d_bn_relu(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    image_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 50
) -> Dict:
    """
    Benchmark Conv2d + BatchNorm + ReLU fusion.

    This is THE most common pattern in CNNs (ResNet, VGG, MobileNet).
    """
    # Create inputs
    X = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    # ========================================================================
    # UNFUSED VERSION (3 separate operations)
    # ========================================================================
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    bn = nn.BatchNorm2d(out_channels, device=device)
    conv.eval()
    bn.eval()

    def unfused_forward(X):
        Y = conv(X)
        Y = bn(Y)
        Y = F.relu(Y)
        return Y

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = unfused_forward(X)

    # Benchmark unfused
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            Y_unfused = unfused_forward(X)
        if device == 'cuda':
            torch.cuda.synchronize()
        unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION (fold BN + inline ReLU)
    # ========================================================================
    # PyTorch can fuse Conv+BN+ReLU on some backends (cudnn, mkldnn)
    # For benchmarking, we'll use explicit folding + JIT
    conv_fused = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, device=device)
    conv_fused.eval()

    # Fold BatchNorm
    with torch.no_grad():
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_eps = bn.eps

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        bias = bn_bias - bn_mean * scale

        conv_fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        conv_fused.bias.copy_(bias)

    # Simple fused forward (PyTorch may optimize this automatically)
    def fused_forward(X):
        return F.relu(conv_fused(X))

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = fused_forward(X)

    # Benchmark fused
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            Y_fused = fused_forward(X)
        if device == 'cuda':
            torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    out_size = (image_size + 2 * padding - kernel_size) // stride + 1

    conv_flops = 2 * kernel_size * kernel_size * in_channels * out_channels * out_size * out_size * batch_size
    bn_flops = 2 * out_channels * out_size * out_size * batch_size
    relu_flops = 0  # ReLU is essentially free (comparison operation)

    total_flops = conv_flops + bn_flops + relu_flops

    # Memory (unfused): conv output written 3 times (after conv, after bn, after relu)
    unfused_bytes = (batch_size * in_channels * image_size * image_size +
                    out_channels * in_channels * kernel_size * kernel_size +
                    batch_size * out_channels * out_size * out_size +  # conv write
                    batch_size * out_channels * out_size * out_size +  # bn read
                    batch_size * out_channels * out_size * out_size +  # bn write
                    batch_size * out_channels * out_size * out_size +  # relu read
                    batch_size * out_channels * out_size * out_size) * 4  # relu write

    # Memory (fused): output written once
    fused_bytes = (batch_size * in_channels * image_size * image_size +
                  out_channels * in_channels * kernel_size * kernel_size +
                  out_channels +
                  batch_size * out_channels * out_size * out_size) * 4

    unfused_gflops = total_flops / unfused_time / 1e9
    fused_gflops = total_flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Conv2d_BN_ReLU',
        'input_shape': (batch_size, in_channels, image_size, image_size),
        'output_shape': (batch_size, out_channels, out_size, out_size),
        'kernel_size': kernel_size,
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': total_flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }


def benchmark_conv2d_relu(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    image_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 50
) -> Dict:
    """
    Benchmark Conv2d + ReLU fusion (simple pattern).
    """
    X = torch.randn(batch_size, in_channels, image_size, image_size, device=device)

    # ========================================================================
    # UNFUSED
    # ========================================================================
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)
    conv.eval()

    def unfused_forward(X):
        Y = conv(X)
        Y = F.relu(Y)
        return Y

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = unfused_forward(X)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            Y_unfused = unfused_forward(X)
        if device == 'cuda':
            torch.cuda.synchronize()
        unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED (Conv2d with ReLU activation built-in)
    # ========================================================================
    conv_fused = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)
    conv_fused.eval()
    conv_fused.weight.data.copy_(conv.weight.data)
    conv_fused.bias.data.copy_(conv.bias.data)

    def fused_forward(X):
        return F.relu(conv_fused(X))

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = fused_forward(X)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            Y_fused = fused_forward(X)
        if device == 'cuda':
            torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # METRICS
    # ========================================================================

    out_size = (image_size + 2 * padding - kernel_size) // stride + 1
    conv_flops = 2 * kernel_size * kernel_size * in_channels * out_channels * out_size * out_size * batch_size

    unfused_bytes = (batch_size * in_channels * image_size * image_size +
                    out_channels * in_channels * kernel_size * kernel_size +
                    out_channels +
                    batch_size * out_channels * out_size * out_size +  # conv write
                    batch_size * out_channels * out_size * out_size +  # relu read
                    batch_size * out_channels * out_size * out_size) * 4  # relu write

    fused_bytes = (batch_size * in_channels * image_size * image_size +
                  out_channels * in_channels * kernel_size * kernel_size +
                  out_channels +
                  batch_size * out_channels * out_size * out_size) * 4

    unfused_gflops = conv_flops / unfused_time / 1e9
    fused_gflops = conv_flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Conv2d_ReLU',
        'input_shape': (batch_size, in_channels, image_size, image_size),
        'output_shape': (batch_size, out_channels, out_size, out_size),
        'kernel_size': kernel_size,
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': conv_flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }


def calibrate_conv_fusion_patterns(
    configs: List[Dict] = None,
    device: str = 'cpu',
    quick: bool = False
) -> List[Dict]:
    """
    Run calibration for convolution fusion patterns.

    Args:
        configs: List of conv configurations (batch, in_ch, out_ch, size, kernel)
        device: 'cpu' or 'cuda'
        quick: If True, use fewer trials

    Returns:
        List of calibration results
    """
    if configs is None:
        if quick:
            # Quick mode: small ResNet-style configs
            configs = [
                {'batch': 1, 'in_ch': 64, 'out_ch': 64, 'size': 56, 'kernel': 3},   # ResNet early layer
                {'batch': 4, 'in_ch': 128, 'out_ch': 128, 'size': 28, 'kernel': 3}, # ResNet middle layer
            ]
        else:
            # Full mode
            configs = [
                {'batch': 1, 'in_ch': 64, 'out_ch': 64, 'size': 56, 'kernel': 3},
                {'batch': 4, 'in_ch': 128, 'out_ch': 128, 'size': 28, 'kernel': 3},
                {'batch': 8, 'in_ch': 256, 'out_ch': 256, 'size': 14, 'kernel': 3},
            ]

    num_trials = 30 if quick else 50

    results = []

    print("  Conv Fusion Patterns")
    print("  " + "-" * 78)

    for cfg in configs:
        b, ic, oc, sz, k = cfg['batch'], cfg['in_ch'], cfg['out_ch'], cfg['size'], cfg['kernel']

        # Conv2d + ReLU
        print(f"  Conv+ReLU (B={b}, {ic}→{oc}, {sz}×{sz}, k={k})...", end=" ", flush=True)
        result = benchmark_conv2d_relu(b, ic, oc, sz, k, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

        # Conv2d + BN
        print(f"  Conv+BN (B={b}, {ic}→{oc}, {sz}×{sz}, k={k})...", end=" ", flush=True)
        result = benchmark_conv2d_bn(b, ic, oc, sz, k, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

        # Conv2d + BN + ReLU (ResNet block)
        print(f"  Conv+BN+ReLU (B={b}, {ic}→{oc}, {sz}×{sz}, k={k})...", end=" ", flush=True)
        result = benchmark_conv2d_bn_relu(b, ic, oc, sz, k, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Conv Fusion Patterns")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    args = parser.parse_args()

    print("=" * 80)
    print("Convolution Fusion Pattern Benchmarks")
    print("=" * 80)
    print()

    results = calibrate_conv_fusion_patterns(device=args.device, quick=args.quick)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    for result in results:
        print(f"\n{result['fusion_pattern']} {result['input_shape']} → {result['output_shape']}:")
        print(f"  Unfused: {result['unfused_latency_ms']:.2f} ms ({result['unfused_gflops']:.1f} GFLOPS)")
        print(f"  Fused:   {result['fused_latency_ms']:.2f} ms ({result['fused_gflops']:.1f} GFLOPS)")
        print(f"  Speedup: {result['speedup_factor']:.2f}× faster")
        print(f"  Memory:  {result['memory_reduction']*100:.1f}% reduction")
