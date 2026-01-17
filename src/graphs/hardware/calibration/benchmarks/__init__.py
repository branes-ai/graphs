"""
Calibration Benchmarks

DEPRECATED: This module has been moved to graphs.benchmarks.
Please update your imports:
    from graphs.hardware.calibration.benchmarks import ...  ->  from graphs.benchmarks import ...

This shim will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.hardware.calibration.benchmarks is deprecated. Use graphs.benchmarks instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from graphs.benchmarks import (
    calibrate_matmul,
    calibrate_memory_bandwidth,
    calibrate_matmul_all_precisions,
)

__all__ = [
    'calibrate_matmul',
    'calibrate_memory_bandwidth',
    'calibrate_matmul_all_precisions',
]
