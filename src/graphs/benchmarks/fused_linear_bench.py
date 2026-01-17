"""
Fused Linear Layer Benchmarks

Benchmarks common linear layer fusion patterns:
- Linear + Bias
- Linear + Bias + ReLU
- Linear + Bias + GELU (transformer FFN)
- Linear + Bias + ReLU + Dropout

Measures unfused vs fused performance to quantify fusion benefits.
"""

import time
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F


def benchmark_linear_bias(
    M: int, K: int, N: int,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 100
) -> Dict:
    """
    Benchmark Linear + Bias fusion.

    Args:
        M: Batch size
        K: Input features
        N: Output features
        device: 'cpu' or 'cuda'
        num_warmup: Warmup iterations
        num_trials: Measurement iterations

    Returns:
        Dict with benchmark results
    """
    # Create inputs
    X = torch.randn(M, K, device=device)
    W = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)

    # ========================================================================
    # UNFUSED VERSION (2 separate operations)
    # ========================================================================
    def unfused_forward(X, W, bias):
        Y = X @ W.T           # Linear (matmul)
        Y = Y + bias          # Add bias
        return Y

    # Warmup
    for _ in range(num_warmup):
        _ = unfused_forward(X, W, bias)

    # Benchmark unfused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_unfused = unfused_forward(X, W, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION (torch.addmm - fused matmul+add)
    # ========================================================================
    def fused_forward(X, W, bias):
        # torch.addmm(bias, X, W.T) computes bias + X @ W.T in one kernel
        return torch.addmm(bias.unsqueeze(0), X, W.T).squeeze(0)

    # Warmup
    for _ in range(num_warmup):
        _ = fused_forward(X, W, bias)

    # Benchmark fused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_fused = fused_forward(X, W, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    # FLOPs: matmul (2*M*K*N) + bias (M*N)
    flops = 2 * M * K * N + M * N

    # Memory (unfused): read X (M*K*4), read W (N*K*4), write Y (M*N*4),
    #                   read Y (M*N*4), read bias (N*4), write Y (M*N*4)
    unfused_bytes = (M*K + N*K + M*N + M*N + N + M*N) * 4

    # Memory (fused): read X, read W, read bias, write Y (single kernel)
    fused_bytes = (M*K + N*K + N + M*N) * 4

    # GFLOPS
    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    # Speedup and reduction
    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Linear_Bias',
        'input_shape': (M, K, N),
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }


def benchmark_linear_bias_relu(
    M: int, K: int, N: int,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 100
) -> Dict:
    """
    Benchmark Linear + Bias + ReLU fusion.

    This is the most common fusion pattern in MLPs and early FC layers.
    """
    # Create inputs
    X = torch.randn(M, K, device=device)
    W = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)

    # ========================================================================
    # UNFUSED VERSION (3 separate operations)
    # ========================================================================
    def unfused_forward(X, W, bias):
        Y = X @ W.T           # Linear
        Y = Y + bias          # Add bias
        Y = torch.relu(Y)     # ReLU
        return Y

    # Warmup
    for _ in range(num_warmup):
        _ = unfused_forward(X, W, bias)

    # Benchmark unfused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_unfused = unfused_forward(X, W, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION (using torch.jit.script for guaranteed fusion)
    # ========================================================================
    @torch.jit.script
    def fused_forward_jit(X, W, bias):
        # JIT will fuse this into a single kernel
        return torch.relu(torch.addmm(bias.unsqueeze(0), X, W.T).squeeze(0))

    # Warmup (JIT compilation)
    for _ in range(num_warmup):
        _ = fused_forward_jit(X, W, bias)

    # Benchmark fused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_fused = fused_forward_jit(X, W, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    # FLOPs: matmul (2*M*K*N) + bias (M*N) + relu (0)
    flops = 2 * M * K * N + M * N

    # Memory (unfused):
    #   matmul: read X, W, write Y
    #   bias: read Y, bias, write Y
    #   relu: read Y, write Y
    unfused_bytes = (M*K + N*K + M*N +  # matmul
                    M*N + N + M*N +      # bias
                    M*N + M*N) * 4       # relu

    # Memory (fused): read X, W, bias, write Y (single kernel)
    fused_bytes = (M*K + N*K + N + M*N) * 4

    # GFLOPS
    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    # Speedup
    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Linear_Bias_ReLU',
        'input_shape': (M, K, N),
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }


def benchmark_linear_bias_gelu(
    M: int, K: int, N: int,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 100
) -> Dict:
    """
    Benchmark Linear + Bias + GELU fusion.

    GELU is used in transformer FFN (Feed-Forward Networks).
    """
    # Create inputs
    X = torch.randn(M, K, device=device)
    W = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)

    # ========================================================================
    # UNFUSED VERSION
    # ========================================================================
    def unfused_forward(X, W, bias):
        Y = X @ W.T
        Y = Y + bias
        Y = F.gelu(Y)
        return Y

    # Warmup
    for _ in range(num_warmup):
        _ = unfused_forward(X, W, bias)

    # Benchmark unfused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_unfused = unfused_forward(X, W, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION
    # ========================================================================
    @torch.jit.script
    def fused_forward_jit(X, W, bias):
        return F.gelu(torch.addmm(bias.unsqueeze(0), X, W.T).squeeze(0))

    # Warmup
    for _ in range(num_warmup):
        _ = fused_forward_jit(X, W, bias)

    # Benchmark fused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_fused = fused_forward_jit(X, W, bias)
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    # FLOPs: matmul + bias + GELU (approximate: 8 ops per element)
    flops = 2 * M * K * N + M * N + 8 * M * N

    unfused_bytes = (M*K + N*K + M*N + M*N + N + M*N + M*N + M*N) * 4
    fused_bytes = (M*K + N*K + N + M*N) * 4

    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Linear_Bias_GELU',
        'input_shape': (M, K, N),
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }


def calibrate_linear_fusion_patterns(
    sizes: List[Tuple[int, int, int]] = None,
    device: str = 'cpu',
    quick: bool = False
) -> List[Dict]:
    """
    Run calibration for all linear fusion patterns.

    Args:
        sizes: List of (M, K, N) tuples to benchmark
        device: 'cpu' or 'cuda'
        quick: If True, use fewer trials

    Returns:
        List of calibration results
    """
    if sizes is None:
        if quick:
            # Quick mode: small and medium only
            sizes = [
                (128, 512, 512),   # Small
                (512, 1024, 1024), # Medium
            ]
        else:
            # Full mode: small, medium, large
            sizes = [
                (128, 512, 512),     # Small
                (512, 1024, 1024),   # Medium
                (1024, 2048, 2048),  # Large
            ]

    num_trials = 50 if quick else 100

    results = []

    print("  Linear Fusion Patterns")
    print("  " + "-" * 78)

    for M, K, N in sizes:
        # Linear + Bias
        print(f"  Linear+Bias ({M}×{K}×{N})...", end=" ", flush=True)
        result = benchmark_linear_bias(M, K, N, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

        # Linear + Bias + ReLU
        print(f"  Linear+Bias+ReLU ({M}×{K}×{N})...", end=" ", flush=True)
        result = benchmark_linear_bias_relu(M, K, N, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

        # Linear + Bias + GELU
        print(f"  Linear+Bias+GELU ({M}×{K}×{N})...", end=" ", flush=True)
        result = benchmark_linear_bias_gelu(M, K, N, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Linear Fusion Patterns")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer sizes)")
    args = parser.parse_args()

    print("=" * 80)
    print("Linear Fusion Pattern Benchmarks")
    print("=" * 80)
    print()

    results = calibrate_linear_fusion_patterns(device=args.device, quick=args.quick)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    for result in results:
        print(f"\n{result['fusion_pattern']} {result['input_shape']}:")
        print(f"  Unfused: {result['unfused_latency_ms']:.2f} ms ({result['unfused_gflops']:.1f} GFLOPS)")
        print(f"  Fused:   {result['fused_latency_ms']:.2f} ms ({result['fused_gflops']:.1f} GFLOPS)")
        print(f"  Speedup: {result['speedup_factor']:.2f}× faster")
        print(f"  Memory:  {result['memory_reduction']*100:.1f}% reduction")
