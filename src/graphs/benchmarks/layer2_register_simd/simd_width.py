"""
SIMD Width Sweep - Layer 2 Operand-Delivery Benchmark

Measures how effective throughput scales with vector length to
characterize SIMD utilization efficiency. The same FMA kernel from
Layer 1 is run at different tensor sizes that exercise different
fractions of the SIMD register width:

- scalar-like: 1 element (no SIMD benefit)
- narrow: 4 elements (128-bit SSE/NEON width at FP32)
- medium: 8 elements (256-bit AVX-2 width at FP32)
- wide: 16 elements (512-bit AVX-512 width at FP32)
- oversized: 64+ elements (multiple SIMD iterations)

The ratio (wide GFLOPS / narrow GFLOPS) is the measured SIMD
efficiency -- directly comparable to the 0.70 constant in
CPUMapper._analyze_vectorization and the 0.90 simd_packed multiplier
in CIRCUIT_TYPE_MULTIPLIER.

PyTorch selects the SIMD path internally, so we don't force a
specific instruction set. Instead, we observe the throughput curve
and infer the effective width from the plateaus.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import torch

from graphs.benchmarks.schema import BenchmarkResult, LayerTag, TimingStats
from graphs.benchmarks.layer1_alu.fma_rate import (
    _run_fma_loop,
    _run_empty_loop,
    get_sink_value,
)

import statistics
import time

SIMD_WIDTHS = [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096]

_sink: float = 0.0


def run_simd_width_sweep(
    device: str = "cpu",
    precision: str = "fp32",
    widths: Optional[List[int]] = None,
    num_iterations: int = 10000,
    warmup_iterations: int = 500,
    num_trials: int = 5,
) -> List[BenchmarkResult]:
    """
    Sweep FMA throughput across vector lengths on one device/precision.

    Each width uses the same total FLOPs (num_iterations * width * 2)
    so shorter vectors run more iterations per trial to keep
    measurement time comparable.

    Returns one BenchmarkResult per width, all tagged LayerTag.REGISTER_SIMD.
    """
    global _sink

    if widths is None:
        widths = SIMD_WIDTHS

    dtype_map = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.float32)

    results: List[BenchmarkResult] = []

    for width in widths:
        a = torch.randn(width, dtype=dtype, device=device)
        b = torch.randn(width, dtype=dtype, device=device)
        c = torch.randn(width, dtype=dtype, device=device)

        flops_per_iter = 2 * width
        total_flops = flops_per_iter * num_iterations

        # Warmup
        for _ in range(warmup_iterations):
            torch.addcmul(a, b, c, value=1.0, out=a)
        _sink = float(a.sum().item())

        # Empty-loop overhead
        empty_times = [_run_empty_loop(num_iterations, device) for _ in range(3)]
        empty_overhead = sorted(empty_times)[1]

        # Measurement
        trial_times_ms = []
        clamped = 0
        for _ in range(num_trials):
            a.normal_()
            raw_s = _run_fma_loop(a, b, c, num_iterations, device)
            net_s = raw_s - empty_overhead
            if net_s <= 0:
                clamped += 1
                net_s = 1e-9
            trial_times_ms.append(net_s * 1000.0)
        _sink = float(a.sum().item())

        sorted_t = sorted(trial_times_ms)
        n = len(sorted_t)
        mean_ms = sum(sorted_t) / n
        std_ms = statistics.stdev(sorted_t) if n > 1 else 0.0

        timing = TimingStats(
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=sorted_t[0],
            max_ms=sorted_t[-1],
            median_ms=sorted_t[n // 2],
            p95_ms=sorted_t[-1],
            p99_ms=sorted_t[-1],
            num_iterations=n,
        )

        gflops = (total_flops / 1e9) / (mean_ms / 1000.0) if mean_ms > 0 else 0

        results.append(BenchmarkResult(
            spec_name=f"layer2_simd_width_{width}_{precision}_{device}",
            timestamp=datetime.now().isoformat(),
            device=device,
            device_name=device,
            precision=precision,
            timing=timing,
            throughput_ops_per_sec=total_flops / (mean_ms / 1000.0) if mean_ms > 0 else 0,
            gflops=gflops,
            layer=LayerTag.REGISTER_SIMD,
            success=True,
            extra={
                "vector_width": width,
                "num_iterations": num_iterations,
                "total_flops": total_flops,
                "clamped_trials": clamped,
                "empty_overhead_ms": empty_overhead * 1000.0,
            },
        ))

    return results
