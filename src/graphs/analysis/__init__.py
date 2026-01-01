"""
Performance Analysis

Analyzes computational graphs for performance characteristics without modifying them.
"""

from .concurrency import ConcurrencyAnalyzer
from .memory import (
    MemoryTimelineEntry,
    MemoryDescriptor,
    MemoryReport,
    MemoryEstimator,
)
from .roofline import (
    LatencyDescriptor,
    RooflinePoint,
    RooflineReport,
    RooflineAnalyzer,
    create_calibrated_analyzer,
    get_roofline_params_for_hardware,
)
from .energy import (
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
