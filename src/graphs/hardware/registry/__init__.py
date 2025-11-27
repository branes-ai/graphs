"""
Unified Hardware Registry

Provides a single entry point for hardware specifications and calibration data.
Consolidates what was previously two separate systems (hardware_database and
calibration profiles) into one coherent registry.

Usage:
    from graphs.hardware.registry import get_registry, HardwareProfile

    # Get the global registry instance
    registry = get_registry()

    # Get a hardware profile (spec + calibration if available)
    profile = registry.get("i7_12700k")

    # Calibrate and save in one step
    profile = registry.calibrate("i7_12700k")

    # Auto-detect current hardware
    profile = registry.detect_and_calibrate()
"""

from .registry import HardwareRegistry, get_registry
from .profile import HardwareProfile
from .config import RegistryConfig, get_config

__all__ = [
    'HardwareRegistry',
    'HardwareProfile',
    'RegistryConfig',
    'get_registry',
    'get_config',
]
