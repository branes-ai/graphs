"""
Layer 2 - Register File / SIMD Width / Warp / Systolic Fill Benchmarks

Measures the cost of feeding operands into the ALU array beyond the
pure ALU rate captured in Layer 1. The delta between Layer 1 (raw FMA)
and Layer 2 (same FMA with SIMD-width and register-pressure variation)
isolates operand-delivery overhead.

See ``docs/plans/bottom-up-microbenchmark-plan.md`` Phase 2.
"""

from .simd_width import run_simd_width_sweep
from .register_pressure import run_register_pressure_benchmark

__all__ = [
    "run_simd_width_sweep",
    "run_register_pressure_benchmark",
]
