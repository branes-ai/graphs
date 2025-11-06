"""
Fused Attention Benchmarks

Benchmarks transformer attention fusion patterns:
- Q @ K.T (attention scores)
- Softmax(Q @ K.T) (attention weights)
- Softmax(Q @ K.T) @ V (attention output)
- Full attention head (Q, K, V → output)

These are critical for transformer models (BERT, GPT, ViT).
"""

import time
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
import math


def benchmark_qk_attention_scores(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 50
) -> Dict:
    """
    Benchmark Q @ K.T (attention scores computation).

    This is the first step of attention: computing similarity between queries and keys.
    """
    # Create Q and K
    Q = torch.randn(batch_size, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, seq_len, head_dim, device=device)

    scale = 1.0 / math.sqrt(head_dim)

    # ========================================================================
    # UNFUSED VERSION (matmul + scale separately)
    # ========================================================================
    def unfused_forward(Q, K, scale):
        scores = Q @ K.transpose(-2, -1)  # [B, seq_len, seq_len]
        scores = scores * scale            # Scale
        return scores

    # Warmup
    for _ in range(num_warmup):
        _ = unfused_forward(Q, K, scale)

    # Benchmark unfused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        scores_unfused = unfused_forward(Q, K, scale)
    if device == 'cuda':
        torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION (scaled dot product in one kernel)
    # ========================================================================
    # PyTorch 2.0+ has F.scaled_dot_product_attention
    # For older versions, use torch.bmm with scaling
    @torch.jit.script
    def fused_forward_jit(Q, K, scale: float):
        # Fuse matmul + scale
        scores = torch.bmm(Q, K.transpose(-2, -1)) * scale
        return scores

    # Warmup
    for _ in range(num_warmup):
        _ = fused_forward_jit(Q, K, scale)

    # Benchmark fused
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        scores_fused = fused_forward_jit(Q, K, scale)
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    # FLOPs: matmul (2 * seq_len * head_dim * seq_len) + scale (seq_len^2)
    flops = batch_size * (2 * seq_len * head_dim * seq_len + seq_len * seq_len)

    # Memory (unfused): read Q, K, write scores, read scores, write scores
    unfused_bytes = (batch_size * seq_len * head_dim +      # Q
                    batch_size * seq_len * head_dim +       # K
                    batch_size * seq_len * seq_len +        # scores write
                    batch_size * seq_len * seq_len +        # scores read
                    batch_size * seq_len * seq_len) * 4     # scores write

    # Memory (fused): read Q, K, write scores
    fused_bytes = (batch_size * seq_len * head_dim +
                  batch_size * seq_len * head_dim +
                  batch_size * seq_len * seq_len) * 4

    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'QK_AttentionScores',
        'input_shape': (batch_size, seq_len, head_dim),
        'output_shape': (batch_size, seq_len, seq_len),
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


def benchmark_attention_softmax(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 50
) -> Dict:
    """
    Benchmark Q @ K.T + Softmax (attention weights computation).

    This is the second step: computing attention weights from scores.
    """
    Q = torch.randn(batch_size, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, seq_len, head_dim, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    # ========================================================================
    # UNFUSED
    # ========================================================================
    def unfused_forward(Q, K, scale):
        scores = Q @ K.transpose(-2, -1)
        scores = scores * scale
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights

    for _ in range(num_warmup):
        _ = unfused_forward(Q, K, scale)

    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        weights_unfused = unfused_forward(Q, K, scale)
    if device == 'cuda':
        torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED
    # ========================================================================
    @torch.jit.script
    def fused_forward_jit(Q, K, scale: float):
        scores = torch.bmm(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights

    for _ in range(num_warmup):
        _ = fused_forward_jit(Q, K, scale)

    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        weights_fused = fused_forward_jit(Q, K, scale)
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # METRICS
    # ========================================================================

    # FLOPs: matmul + scale + softmax (exp + sum + div: ~5 ops per element)
    flops = batch_size * (2 * seq_len * head_dim * seq_len +
                         seq_len * seq_len +
                         5 * seq_len * seq_len)

    unfused_bytes = (batch_size * seq_len * head_dim +
                    batch_size * seq_len * head_dim +
                    batch_size * seq_len * seq_len +  # scores write
                    batch_size * seq_len * seq_len +  # scale write
                    batch_size * seq_len * seq_len +  # softmax read
                    batch_size * seq_len * seq_len) * 4  # softmax write

    fused_bytes = (batch_size * seq_len * head_dim +
                  batch_size * seq_len * head_dim +
                  batch_size * seq_len * seq_len) * 4

    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'QK_Softmax',
        'input_shape': (batch_size, seq_len, head_dim),
        'output_shape': (batch_size, seq_len, seq_len),
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


def benchmark_full_attention(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_trials: int = 50
) -> Dict:
    """
    Benchmark full attention: Softmax(Q @ K.T / sqrt(d)) @ V.

    This is the complete attention mechanism in one operation.
    """
    Q = torch.randn(batch_size, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, seq_len, head_dim, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    # ========================================================================
    # UNFUSED (4 separate operations)
    # ========================================================================
    def unfused_forward(Q, K, V, scale):
        scores = Q @ K.transpose(-2, -1)       # QK^T
        scores = scores * scale                 # Scale
        attn_weights = F.softmax(scores, dim=-1)  # Softmax
        output = attn_weights @ V               # Attention @ V
        return output

    for _ in range(num_warmup):
        _ = unfused_forward(Q, K, V, scale)

    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        out_unfused = unfused_forward(Q, K, V, scale)
    if device == 'cuda':
        torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED (PyTorch 2.0+ scaled_dot_product_attention or JIT)
    # ========================================================================
    # Try to use PyTorch's built-in fused attention if available
    has_sdpa = hasattr(F, 'scaled_dot_product_attention')

    if has_sdpa:
        def fused_forward(Q, K, V):
            # PyTorch 2.0+ has highly optimized fused attention
            return F.scaled_dot_product_attention(Q, K, V)

        for _ in range(num_warmup):
            _ = fused_forward(Q, K, V)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            out_fused = fused_forward(Q, K, V)
        if device == 'cuda':
            torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_trials
    else:
        # Fallback: JIT fusion
        @torch.jit.script
        def fused_forward_jit(Q, K, V, scale: float):
            scores = torch.bmm(Q, K.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.bmm(attn_weights, V)
            return output

        for _ in range(num_warmup):
            _ = fused_forward_jit(Q, K, V, scale)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            out_fused = fused_forward_jit(Q, K, V, scale)
        if device == 'cuda':
            torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # METRICS
    # ========================================================================

    # FLOPs: QK^T (2*S*D*S) + scale (S^2) + softmax (5*S^2) + attn@V (2*S*S*D)
    flops = batch_size * (2 * seq_len * head_dim * seq_len +  # QK^T
                         seq_len * seq_len +                   # scale
                         5 * seq_len * seq_len +               # softmax
                         2 * seq_len * seq_len * head_dim)     # attn @ V

    # Memory (unfused): many intermediate writes
    unfused_bytes = (batch_size * seq_len * head_dim +  # Q
                    batch_size * seq_len * head_dim +   # K
                    batch_size * seq_len * head_dim +   # V
                    batch_size * seq_len * seq_len +    # scores write
                    batch_size * seq_len * seq_len +    # scale write
                    batch_size * seq_len * seq_len +    # softmax write
                    batch_size * seq_len * head_dim) * 4  # output write

    # Memory (fused): only read inputs and write output
    fused_bytes = (batch_size * seq_len * head_dim +
                  batch_size * seq_len * head_dim +
                  batch_size * seq_len * head_dim +
                  batch_size * seq_len * head_dim) * 4

    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    fusion_type = "SDPA" if has_sdpa else "JIT"

    return {
        'fusion_pattern': f'FullAttention_{fusion_type}',
        'input_shape': (batch_size, seq_len, head_dim),
        'output_shape': (batch_size, seq_len, head_dim),
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


def calibrate_attention_fusion_patterns(
    configs: List[Dict] = None,
    device: str = 'cpu',
    quick: bool = False
) -> List[Dict]:
    """
    Run calibration for attention fusion patterns.

    Args:
        configs: List of attention configurations (batch, seq_len, head_dim)
        device: 'cpu' or 'cuda'
        quick: If True, use fewer trials

    Returns:
        List of calibration results
    """
    if configs is None:
        if quick:
            # Quick mode: typical transformer sizes
            configs = [
                {'batch': 1, 'seq_len': 128, 'head_dim': 64},   # Small
                {'batch': 4, 'seq_len': 256, 'head_dim': 64},   # Medium
            ]
        else:
            # Full mode
            configs = [
                {'batch': 1, 'seq_len': 128, 'head_dim': 64},
                {'batch': 4, 'seq_len': 256, 'head_dim': 64},
                {'batch': 8, 'seq_len': 512, 'head_dim': 64},
            ]

    num_trials = 30 if quick else 50

    results = []

    print("  Attention Fusion Patterns")
    print("  " + "-" * 78)

    for cfg in configs:
        b, s, d = cfg['batch'], cfg['seq_len'], cfg['head_dim']

        # Q @ K.T (attention scores)
        print(f"  QK^T (B={b}, S={s}, D={d})...", end=" ", flush=True)
        result = benchmark_qk_attention_scores(b, s, d, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

        # Q @ K.T + Softmax
        print(f"  QK^T+Softmax (B={b}, S={s}, D={d})...", end=" ", flush=True)
        result = benchmark_attention_softmax(b, s, d, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

        # Full attention
        print(f"  FullAttention (B={b}, S={s}, D={d})...", end=" ", flush=True)
        result = benchmark_full_attention(b, s, d, device=device, num_trials=num_trials)
        results.append(result)
        print(f"{result['speedup_factor']:.2f}× speedup, {result['memory_reduction']*100:.1f}% mem reduction")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Attention Fusion Patterns")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    args = parser.parse_args()

    print("=" * 80)
    print("Attention Fusion Pattern Benchmarks")
    print("=" * 80)
    print()

    results = calibrate_attention_fusion_patterns(device=args.device, quick=args.quick)

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
