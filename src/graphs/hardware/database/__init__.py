"""
Hardware Database Module

Provides schema, manager, and detection for hardware specifications.
"""

from .schema import HardwareSpec, HardwareDetectionResult, REQUIRED_PRECISIONS, CoreInfo, CoreCluster, SystemInfo, MapperInfo
from .manager import HardwareDatabase, get_database
from .detector import HardwareDetector, DetectedCPU, DetectedGPU, MatchResult

__all__ = [
    'HardwareSpec',
    'HardwareDetectionResult',
    'REQUIRED_PRECISIONS',
    'CoreInfo',
    'CoreCluster',
    'SystemInfo',
    'MapperInfo',
    'HardwareDatabase',
    'get_database',
    'HardwareDetector',
    'DetectedCPU',
    'DetectedGPU',
    'MatchResult',
]
