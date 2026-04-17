"""
Precision Sweep - Layer 1 ALU Energy Characterization

Runs the FMA-rate benchmark across all supported precisions on a
single device, collecting throughput (GFLOPS) and energy (pJ/op via
PowerMeter) for each. The result set feeds ``layer1_alu_fitter``
to fit per-precision ``ComputeFabric.energy_scaling`` and
``ops_per_unit_per_clock``.

Usage:
    from graphs.benchmarks.layer1_alu import run_precision_sweep
    results = run_precision_sweep(device="cpu")
    for r in results:
        pj = r.extra.get("pj_per_op")
        print(f"{r.precision}: {r.gflops:.1f} GFLOPS, {pj} pJ/op")
"""

from __future__ import annotations

from typing import List, Optional

from graphs.benchmarks.schema import BenchmarkResult
from .fma_rate import run_fma_rate_benchmark

DEFAULT_PRECISIONS_CPU = ["fp64", "fp32", "fp16", "bf16"]
DEFAULT_PRECISIONS_CUDA = ["fp64", "fp32", "fp16", "bf16"]


def run_precision_sweep(
    device: str = "cpu",
    precisions: Optional[List[str]] = None,
    num_elements: int = 4096,
    num_iterations: int = 5000,
    warmup_iterations: int = 200,
    num_trials: int = 5,
    enable_power: bool = True,
) -> List[BenchmarkResult]:
    """
    Sweep FMA rate across precisions on one device.

    For each precision, optionally attaches a PowerMeter so pJ/op
    is recorded in ``result.extra["pj_per_op"]``.

    Args:
        device: target device
        precisions: list of precision strings; None = auto-detect
        num_elements: vector length per FMA call
        num_iterations: FMA iterations per trial
        warmup_iterations: warmup count
        num_trials: measurement trials per precision
        enable_power: attach PowerMeter (RAPL / NVML / tegrastats)

    Returns:
        One BenchmarkResult per precision, all tagged LayerTag.ALU
    """
    if precisions is None:
        precisions = (
            DEFAULT_PRECISIONS_CUDA
            if device.startswith("cuda")
            else DEFAULT_PRECISIONS_CPU
        )

    results: List[BenchmarkResult] = []
    for prec in precisions:
        collector = None
        if enable_power:
            try:
                from graphs.benchmarks.power_meter import auto_select_power_collector
                collector = auto_select_power_collector(device)
            except Exception:
                # PowerMeter unavailable; proceed without energy measurement.
                pass

        try:
            result = run_fma_rate_benchmark(
                device=device,
                precision=prec,
                num_elements=num_elements,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                num_trials=num_trials,
                power_collector=collector,
            )
        except Exception as exc:
            from datetime import datetime
            from graphs.benchmarks.schema import LayerTag
            result = BenchmarkResult(
                spec_name=f"layer1_fma_rate_{prec}_{device}",
                timestamp=datetime.now().isoformat(),
                device=device,
                precision=prec,
                layer=LayerTag.ALU,
                success=False,
                error_message=str(exc),
            )
        results.append(result)

    return results
