"""
Systolic Array Utilization Module

Analyze systolic array utilization across multiple array sizes.

Classes:
    SystolicArrayConfig: Configuration for a systolic array
    UtilizationResult: Utilization analysis for a single shape
    SystolicUtilizationCalculator: Calculate utilization for matrix operations
    ArraySizeSweeper: Sweep array sizes and analyze utilization
    SweepResult: Results from sweeping array sizes
"""

from graphs.research.systolic.utilization import (
    SystolicArrayConfig,
    UtilizationResult,
    SystolicUtilizationCalculator,
)
from graphs.research.systolic.sweep import (
    ArraySizeSweeper,
    SweepResult,
    ARRAY_SIZES,
    PRECISIONS,
    sweep_by_class,
    analyze_size_sensitivity,
)

__all__ = [
    'SystolicArrayConfig',
    'UtilizationResult',
    'SystolicUtilizationCalculator',
    'ArraySizeSweeper',
    'SweepResult',
    'ARRAY_SIZES',
    'PRECISIONS',
    'sweep_by_class',
    'analyze_size_sensitivity',
]
