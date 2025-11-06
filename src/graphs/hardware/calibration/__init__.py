"""
Hardware Calibration Framework

This module provides tools to calibrate hardware performance models by running
actual benchmarks and measuring real-world performance, rather than relying
on theoretical peak specifications.

Key components:
- schema.py: Data structures for calibration results
- calibrator.py: Orchestrator for running calibration benchmarks
- benchmarks/: Individual operation benchmarks (matmul, conv2d, etc.)
- profiles/: Pre-calibrated hardware profiles (JSON files)

Usage:
    from graphs.hardware.calibration import calibrate_hardware, load_calibration

    # Run full calibration
    calibration = calibrate_hardware('i7-12700k')

    # Load existing profile
    calibration = load_calibration('profiles/i7_12700k.json')

    # Use in mapper
    mapper = CPUMapper(resource_model, calibration=calibration)
"""

from .schema import (
    OperationCalibration,
    HardwareCalibration,
    CalibrationMetadata,
)
from .calibrator import load_calibration

__all__ = [
    'OperationCalibration',
    'HardwareCalibration',
    'CalibrationMetadata',
    'load_calibration',
]
