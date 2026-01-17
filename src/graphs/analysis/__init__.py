"""
Performance Analysis

DEPRECATED: This module has been renamed to graphs.estimation.
Please update your imports:
    from graphs.analysis import ...  ->  from graphs.estimation import ...

This shim will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis is deprecated. Use graphs.estimation instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from graphs.estimation import (
    ConcurrencyAnalyzer,
    MemoryTimelineEntry,
    MemoryDescriptor,
    MemoryReport,
    MemoryEstimator,
    LatencyDescriptor,
    RooflinePoint,
    RooflineReport,
    RooflineAnalyzer,
    create_calibrated_analyzer,
    get_roofline_params_for_hardware,
    EnergyDescriptor,
    EnergyReport,
    EnergyAnalyzer,
)

__all__ = [
    'ConcurrencyAnalyzer',
    'MemoryTimelineEntry',
    'MemoryDescriptor',
    'MemoryReport',
    'MemoryEstimator',
    'LatencyDescriptor',
    'RooflinePoint',
    'RooflineReport',
    'RooflineAnalyzer',
    'create_calibrated_analyzer',
    'get_roofline_params_for_hardware',
    'EnergyDescriptor',
    'EnergyReport',
    'EnergyAnalyzer',
]
