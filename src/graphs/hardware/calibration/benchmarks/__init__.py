"""
Calibration benchmarks for various operation types.

Each benchmark module provides a calibrate() function that returns
OperationCalibration objects for specific hardware.
"""

from .matmul_bench import calibrate_matmul
from .memory_bench import calibrate_memory_bandwidth
from .matmul_bench_multi import calibrate_matmul_all_precisions

__all__ = [
    'calibrate_matmul',
    'calibrate_memory_bandwidth',
    'calibrate_matmul_all_precisions',
]
