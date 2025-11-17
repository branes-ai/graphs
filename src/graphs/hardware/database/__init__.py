"""
Hardware Database Module

Provides schema, manager, and detection for hardware specifications.
"""

from .schema import HardwareSpec, HardwareDetectionResult
from .manager import HardwareDatabase, get_database
from .detector import HardwareDetector, DetectedCPU, DetectedGPU, MatchResult

__all__ = [
    'HardwareSpec',
    'HardwareDetectionResult',
    'HardwareDatabase',
    'get_database',
    'HardwareDetector',
    'DetectedCPU',
    'DetectedGPU',
    'MatchResult',
]
