"""
FMA Rate Microbenchmark — Layer 1 ALU Isolation

Measures per-precision throughput (ops/sec) and optionally energy
(pJ/op) in a tight FMA loop. Empty-loop subtraction removes timer
and loop-control overhead so the residual is pure ALU time.

Defense against dead-code elimination (DCE):
  The accumulator is written to a module-level ``_sink`` after each
  trial. Because ``_sink`` is read by ``get_sink_value()`` (called
  in tests), the compiler cannot prove the loop is side-effect-free.

Usage:
    from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
    result = run_fma_rate_benchmark(
        device="cpu",
        precision="fp32",
        num_elements=4096,
        num_iterations=1000,
    )
    print(f"{result.gflops:.1f} GFLOPS, {result.energy_joules} J")
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import torch

from graphs.benchmarks.schema import BenchmarkResult, LayerTag, TimingStats

_sink: float = 0.0


def get_sink_value() -> float:
    """Read the accumulator sink (prevents DCE of the benchmark loop)."""
    return _sink


def _get_torch_dtype(precision: str) -> torch.dtype:
    _MAP = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return _MAP.get(precision, torch.float32)


def _run_fma_loop(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iterations: int,
    device: str,
) -> float:
    """Run the FMA loop and return wall-clock seconds."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iterations):
            torch.addcmul(a, b, c, value=1.0, out=a)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0
    else:
        start = time.perf_counter()
        for _ in range(num_iterations):
            torch.addcmul(a, b, c, value=1.0, out=a)
        return time.perf_counter() - start


def _run_empty_loop(num_iterations: int, device: str) -> float:
    """Measure overhead of the loop + timer without any FMA work."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iterations):
            pass
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0
    else:
        start = time.perf_counter()
        for _ in range(num_iterations):
            pass
        return time.perf_counter() - start


def run_fma_rate_benchmark(
    device: str = "cpu",
    precision: str = "fp32",
    num_elements: int = 4096,
    num_iterations: int = 5000,
    warmup_iterations: int = 200,
    num_trials: int = 5,
    power_collector: Optional[object] = None,
) -> BenchmarkResult:
    """
    Measure FMA throughput for a single precision on a single device.

    Each trial runs ``num_iterations`` of element-wise FMA
    (``a = a + b * c``) on vectors of ``num_elements``. The vector
    size is chosen small enough to fit in L1 / registers so memory
    is not the bottleneck, but large enough to fill the SIMD lanes
    (4096 FP32 = 16 KB, well within any L1).

    Empty-loop subtraction removes per-iteration overhead.

    Args:
        device: "cpu" or "cuda" / "cuda:0"
        precision: "fp64", "fp32", "fp16", "bf16"
        num_elements: vector length (should fit in L1)
        num_iterations: FMA ops per trial
        warmup_iterations: warmup before measurement
        num_trials: how many measurement trials (for statistics)
        power_collector: optional MeasurementCollector (from power_meter)

    Returns:
        BenchmarkResult tagged with LayerTag.ALU
    """
    global _sink

    dtype = _get_torch_dtype(precision)
    a = torch.randn(num_elements, dtype=dtype, device=device)
    b = torch.randn(num_elements, dtype=dtype, device=device)
    c = torch.randn(num_elements, dtype=dtype, device=device)

    # 2 FLOPs per element per iteration (multiply + add)
    flops_per_iter = 2 * num_elements
    total_flops = flops_per_iter * num_iterations

    # Warmup
    for _ in range(warmup_iterations):
        torch.addcmul(a, b, c, value=1.0, out=a)
    _sink = float(a.sum().item())

    # Measure empty-loop overhead (median of 3)
    empty_times = [_run_empty_loop(num_iterations, device) for _ in range(3)]
    empty_overhead = sorted(empty_times)[1]

    # Start power collection if available
    if power_collector is not None:
        try:
            power_collector.start()
        except Exception:
            # Runtime start failure (e.g., RAPL perms); proceed without power.
            power_collector = None

    # Measurement trials
    trial_times_ms = []
    for _ in range(num_trials):
        a.normal_()
        raw_seconds = _run_fma_loop(a, b, c, num_iterations, device)
        net_seconds = max(raw_seconds - empty_overhead, 1e-9)
        trial_times_ms.append(net_seconds * 1000.0)
    _sink = float(a.sum().item())

    # Stop power collection
    energy_joules = None
    avg_power_watts = None
    peak_power_watts = None
    if power_collector is not None:
        try:
            power_collector.stop()
            from graphs.benchmarks.collectors import PowerMeasurement
            measurement = power_collector.get_measurement()
            if isinstance(measurement, PowerMeasurement) and measurement.success:
                energy_joules = measurement.energy_joules
                avg_power_watts = measurement.avg_power_watts
                peak_power_watts = measurement.peak_power_watts
        except Exception:
            # Power collection is best-effort; don't abort the benchmark.
            pass

    # Statistics
    sorted_t = sorted(trial_times_ms)
    n = len(sorted_t)
    mean_ms = sum(sorted_t) / n
    median_ms = sorted_t[n // 2]
    import statistics
    std_ms = statistics.stdev(sorted_t) if n > 1 else 0.0

    timing = TimingStats(
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=sorted_t[0],
        max_ms=sorted_t[-1],
        median_ms=median_ms,
        p95_ms=sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1],
        p99_ms=sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1],
        num_iterations=n,
    )

    gflops = (total_flops / 1e9) / (mean_ms / 1000.0)

    pj_per_op = None
    if energy_joules is not None and energy_joules > 0:
        total_ops_all_trials = total_flops * num_trials
        pj_per_op = (energy_joules / total_ops_all_trials) * 1e12

    return BenchmarkResult(
        spec_name=f"layer1_fma_rate_{precision}_{device}",
        timestamp=datetime.now().isoformat(),
        device=device,
        device_name=device,
        precision=precision,
        timing=timing,
        throughput_ops_per_sec=total_flops / (mean_ms / 1000.0),
        gflops=gflops,
        energy_joules=energy_joules,
        avg_power_watts=avg_power_watts,
        peak_power_watts=peak_power_watts,
        layer=LayerTag.ALU,
        success=True,
        extra={
            "num_elements": num_elements,
            "num_iterations": num_iterations,
            "num_trials": num_trials,
            "flops_per_iteration": flops_per_iter,
            "total_flops_per_trial": total_flops,
            "empty_loop_overhead_ms": empty_overhead * 1000.0,
            "pj_per_op": pj_per_op,
            "sink_value": _sink,
        },
    )
