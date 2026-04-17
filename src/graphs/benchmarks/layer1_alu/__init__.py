"""
Layer 1 - ALU / MAC / Tensor-Core Microbenchmarks

Isolates per-ALU throughput and energy in tight FMA loops with
empty-loop subtraction, for every supported precision. Results
feed ``layer1_alu_fitter`` to calibrate ``ComputeFabric``
coefficients from measurement rather than datasheet values.

See ``docs/plans/bottom-up-microbenchmark-plan.md`` Phase 1.
"""

from .fma_rate import run_fma_rate_benchmark
from .precision_sweep import run_precision_sweep

__all__ = [
    "run_fma_rate_benchmark",
    "run_precision_sweep",
]
