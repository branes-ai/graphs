"""
Hardware Calibration Framework

DEPRECATED: This module has been moved to graphs.calibration.
Please update your imports:
    from graphs.hardware.calibration import ...  ->  from graphs.calibration import ...

This shim will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.hardware.calibration is deprecated. Use graphs.calibration instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from graphs.calibration import (
    OperationCalibration,
    HardwareCalibration,
    CalibrationMetadata,
    PrecisionTestResult,
    PrecisionCapabilityMatrix,
    GPUClockData,
    GPUClockInfo,
    load_calibration,
    calibrate_hardware,
    get_precision_capabilities,
    CalibrationLogger,
    get_logger,
    set_logger,
    LogAdapter,
    get_gpu_clock_info,
    get_gpu_clock_under_load,
    get_jetson_power_mode,
    estimate_theoretical_peak,
    print_gpu_clock_info,
)

__all__ = [
    'OperationCalibration',
    'HardwareCalibration',
    'CalibrationMetadata',
    'PrecisionTestResult',
    'PrecisionCapabilityMatrix',
    'GPUClockData',
    'GPUClockInfo',
    'load_calibration',
    'calibrate_hardware',
    'get_precision_capabilities',
    'CalibrationLogger',
    'get_logger',
    'set_logger',
    'LogAdapter',
    'get_gpu_clock_info',
    'get_gpu_clock_under_load',
    'get_jetson_power_mode',
    'estimate_theoretical_peak',
    'print_gpu_clock_info',
]
