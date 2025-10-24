"""
Hardware Modeling and Mapping

Models hardware resources and provides algorithms for mapping computational
graphs to specific hardware architectures.
"""

from .resource_model import (
    HardwareType,
    Precision,
    ClockDomain,
    ComputeResource,
    PrecisionProfile,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    HardwareMapper,
)

__all__ = [
    'HardwareType',
    'Precision',
    'ClockDomain',
    'ComputeResource',
    'PrecisionProfile',
    'HardwareResourceModel',
    'HardwareAllocation',
    'GraphHardwareAllocation',
    'HardwareMapper',
]
