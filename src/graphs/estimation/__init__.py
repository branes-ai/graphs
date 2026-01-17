"""
Performance Estimation

Analyzes computational graphs for performance characteristics.
Provides latency, energy, and memory estimation based on hardware models.

Migrated from graphs.analysis (deprecated).
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
