"""
Calibration Benchmarks

Individual operation benchmarks for hardware calibration.
Each benchmark module provides a calibrate() function that returns
OperationCalibration objects for specific hardware.

Migrated from graphs.hardware.calibration.benchmarks (deprecated).
"""

from .matmul_bench import calibrate_matmul
from .memory_bench import calibrate_memory_bandwidth
from .matmul_bench_multi import calibrate_matmul_all_precisions
from .power_meter import (
    RAPLPowerCollector,
    TegrastatsPowerCollector,
    NoOpPowerCollector,
    auto_select_power_collector,
)

__all__ = [
    'calibrate_matmul',
    'calibrate_memory_bandwidth',
    'calibrate_matmul_all_precisions',
    # Power measurement backends (Phase 0.5)
    'RAPLPowerCollector',
    'TegrastatsPowerCollector',
    'NoOpPowerCollector',
    'auto_select_power_collector',
]
