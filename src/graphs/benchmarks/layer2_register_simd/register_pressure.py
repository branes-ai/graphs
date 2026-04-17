"""
Register Pressure Benchmark - Layer 2

Measures the throughput difference between independent FMA chains
(high ILP, fits in registers) and dependent FMA chains (serial
dependency, forces pipeline stalls or register spills).

The throughput ratio (independent / dependent) approximates the
register-delivery overhead: when the CPU can keep all operands in
registers and exploit ILP, throughput is high; when dependencies
force stalls or spills to L1, throughput drops.

This benchmark does NOT directly measure register-file energy in
picojoules (that would require on-die power instrumentation).
Instead it measures the throughput delta, which the fitter uses
together with the TechnologyProfile's analytical register energy
to validate or adjust the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch

from graphs.benchmarks.schema import BenchmarkResult, LayerTag, TimingStats
import statistics


@dataclass
class RegisterPressureResult:
    """Summary of register-pressure benchmark."""
    independent_gflops: float
    dependent_gflops: float
    ilp_ratio: float


def _run_independent_fma(
    size: int,
    dtype: torch.dtype,
    device: str,
    num_iterations: int,
) -> float:
    """Multiple independent accumulations -- high ILP, low register pressure."""
    a1 = torch.randn(size, dtype=dtype, device=device)
    a2 = torch.randn(size, dtype=dtype, device=device)
    a3 = torch.randn(size, dtype=dtype, device=device)
    a4 = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        torch.addcmul(a1, b, c, value=1.0, out=a1)
        torch.addcmul(a2, b, c, value=1.0, out=a2)
        torch.addcmul(a3, b, c, value=1.0, out=a3)
        torch.addcmul(a4, b, c, value=1.0, out=a4)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    # Touch all accumulators to prevent DCE
    _ = float((a1.sum() + a2.sum() + a3.sum() + a4.sum()).item())
    return elapsed


def _run_dependent_fma(
    size: int,
    dtype: torch.dtype,
    device: str,
    num_iterations: int,
) -> float:
    """Serial dependency chain -- low ILP, high register pressure."""
    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        # Each depends on the previous: a = a + b*c, then a = a + b*c, ...
        torch.addcmul(a, b, c, value=1.0, out=a)
        torch.addcmul(a, b, c, value=1.0, out=a)
        torch.addcmul(a, b, c, value=1.0, out=a)
        torch.addcmul(a, b, c, value=1.0, out=a)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    _ = float(a.sum().item())
    return elapsed


def run_register_pressure_benchmark(
    device: str = "cpu",
    precision: str = "fp32",
    num_elements: int = 4096,
    num_iterations: int = 2000,
    warmup_iterations: int = 200,
    num_trials: int = 5,
) -> BenchmarkResult:
    """
    Measure ILP/register-pressure effect on FMA throughput.

    Runs 4 independent FMA streams vs 4 dependent FMA streams
    (same total FLOPs) and reports the ILP ratio.

    Returns a BenchmarkResult tagged LayerTag.REGISTER_SIMD with
    the ILP ratio in extra["ilp_ratio"].
    """
    dtype_map = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.float32)

    # 4 FMA ops per iteration, 2 FLOPs each = 8 FLOPs/iter/element
    flops_per_iter = 8 * num_elements
    total_flops = flops_per_iter * num_iterations

    # Warmup both paths
    for _ in range(warmup_iterations):
        _run_independent_fma(num_elements, dtype, device, 1)
        _run_dependent_fma(num_elements, dtype, device, 1)

    # Measure independent
    ind_trials = []
    for _ in range(num_trials):
        t = _run_independent_fma(num_elements, dtype, device, num_iterations)
        ind_trials.append(t * 1000.0)

    # Measure dependent
    dep_trials = []
    for _ in range(num_trials):
        t = _run_dependent_fma(num_elements, dtype, device, num_iterations)
        dep_trials.append(t * 1000.0)

    ind_mean = sum(ind_trials) / len(ind_trials)
    dep_mean = sum(dep_trials) / len(dep_trials)

    ind_gflops = (total_flops / 1e9) / (ind_mean / 1000.0) if ind_mean > 0 else 0
    dep_gflops = (total_flops / 1e9) / (dep_mean / 1000.0) if dep_mean > 0 else 0
    ilp_ratio = ind_gflops / dep_gflops if dep_gflops > 0 else 1.0

    sorted_dep = sorted(dep_trials)
    n = len(sorted_dep)
    timing = TimingStats(
        mean_ms=dep_mean,
        std_ms=statistics.stdev(dep_trials) if n > 1 else 0.0,
        min_ms=sorted_dep[0],
        max_ms=sorted_dep[-1],
        median_ms=sorted_dep[n // 2],
        p95_ms=sorted_dep[-1],
        p99_ms=sorted_dep[-1],
        num_iterations=n,
    )

    return BenchmarkResult(
        spec_name=f"layer2_register_pressure_{precision}_{device}",
        timestamp=datetime.now().isoformat(),
        device=device,
        device_name=device,
        precision=precision,
        timing=timing,
        throughput_ops_per_sec=total_flops / (dep_mean / 1000.0) if dep_mean > 0 else 0,
        gflops=dep_gflops,
        layer=LayerTag.REGISTER_SIMD,
        success=True,
        extra={
            "num_elements": num_elements,
            "num_iterations": num_iterations,
            "independent_gflops": ind_gflops,
            "dependent_gflops": dep_gflops,
            "ilp_ratio": ilp_ratio,
            "independent_mean_ms": ind_mean,
            "dependent_mean_ms": dep_mean,
        },
    )
