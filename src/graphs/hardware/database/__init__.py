"""
Hardware Database Module

Provides schema, manager, and detection for hardware specifications.
"""

from .schema import (
    HardwareSpec,
    HardwareDetectionResult,
    CoreInfo,
    CoreCluster,
    SystemInfo,
    MapperInfo,
    CalibrationSummary,
    CalibrationProfiles,
    # Precision taxonomy
    REQUIRED_PRECISIONS,
    IEEE_FLOAT_PRECISIONS,
    VENDOR_FLOAT_PRECISIONS,
    INTEGER_PRECISIONS,
    ALL_FLOAT_PRECISIONS,
    PRECISION_BITS,
    PRECISION_STORAGE_BITS,
)
from .manager import HardwareDatabase, get_database
from .detector import (
    HardwareDetector,
    DetectedCPU,
    DetectedGPU,
    DetectedBoard,
    MatchResult,
    BoardMatchResult,
)

__all__ = [
    # Core types
    'HardwareSpec',
    'HardwareDetectionResult',
    'CoreInfo',
    'CoreCluster',
    'SystemInfo',
    'MapperInfo',
    'CalibrationSummary',
    'CalibrationProfiles',
    # Precision taxonomy
    'REQUIRED_PRECISIONS',
    'IEEE_FLOAT_PRECISIONS',
    'VENDOR_FLOAT_PRECISIONS',
    'INTEGER_PRECISIONS',
    'ALL_FLOAT_PRECISIONS',
    'PRECISION_BITS',
    'PRECISION_STORAGE_BITS',
    # Database management
    'HardwareDatabase',
    'get_database',
    # Detection
    'HardwareDetector',
    'DetectedCPU',
    'DetectedGPU',
    'DetectedBoard',
    'MatchResult',
    'BoardMatchResult',
]
