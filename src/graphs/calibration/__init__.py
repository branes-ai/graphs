"""
Hardware Calibration Framework

This module provides tools to calibrate hardware performance models by running
actual benchmarks and measuring real-world performance, rather than relying
on theoretical peak specifications.

Key components:
- schema.py: Data structures for calibration results
- calibrator.py: Orchestrator for running calibration benchmarks
- profiles/: Pre-calibrated hardware profiles (JSON files)

Migrated from graphs.hardware.calibration (deprecated).

Usage:
    from graphs.calibration import calibrate_hardware, load_calibration

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
    PrecisionTestResult,
    PrecisionCapabilityMatrix,
    GPUClockData,
)
from .calibrator import load_calibration, calibrate_hardware
from .precision_detector import get_precision_capabilities
from .logging import CalibrationLogger, get_logger, set_logger, LogAdapter
from .gpu_clock import (
    GPUClockInfo,
    get_gpu_clock_info,
    get_gpu_clock_under_load,
    get_jetson_power_mode,
    estimate_theoretical_peak,
    print_gpu_clock_info,
)
from .roofline_fitter import (
    RooflineFitter,
    RooflineParameters,
    FitMetrics,
    FitQuality,
    fit_roofline,
)
from .efficiency_curves import (
    EfficiencyCurve,
    AsymptoticCurve,
    PiecewiseLinearCurve,
    PolynomialCurve,
    ConstantCurve,
    EfficiencyProfile,
    CurveType,
    CurveFitResult,
    fit_efficiency_curve,
    auto_fit_efficiency_curve,
)
from .energy_fitter import (
    EnergyFitter,
    EnergyCoefficients,
    EnergyFitMetrics,
    EnergyFitQuality,
    fit_energy_model,
)
from .power_model import (
    CalibratedPowerModel,
    PowerSource,
    PowerBreakdown,
    EnergyBreakdown,
)
from .utilization_fitter import (
    UtilizationFitter,
    UtilizationFitQuality,
    UtilizationFitMetrics,
    UtilizationCurveResult,
    UtilizationProfile,
    fit_utilization,
)
from .utilization_curves import (
    UtilizationCurve,
    AsymptoticUtilizationCurve,
    PiecewiseLinearUtilizationCurve,
    PolynomialUtilizationCurve,
    ConstantUtilizationCurve,
    fit_utilization_curve,
    auto_fit_utilization_curve,
    create_typical_compute_curve,
    create_typical_memory_curve,
    interpolate_utilization,
)

__all__ = [
    # Schema
    'OperationCalibration',
    'HardwareCalibration',
    'CalibrationMetadata',
    'PrecisionTestResult',
    'PrecisionCapabilityMatrix',
    'GPUClockData',
    'GPUClockInfo',
    # Calibration functions
    'load_calibration',
    'calibrate_hardware',
    'get_precision_capabilities',
    # Logging
    'CalibrationLogger',
    'get_logger',
    'set_logger',
    'LogAdapter',
    # GPU clock
    'get_gpu_clock_info',
    'get_gpu_clock_under_load',
    'get_jetson_power_mode',
    'estimate_theoretical_peak',
    'print_gpu_clock_info',
    # Roofline fitting (TASK-2026-006)
    'RooflineFitter',
    'RooflineParameters',
    'FitMetrics',
    'FitQuality',
    'fit_roofline',
    # Efficiency curves (TASK-2026-006)
    'EfficiencyCurve',
    'AsymptoticCurve',
    'PiecewiseLinearCurve',
    'PolynomialCurve',
    'ConstantCurve',
    'EfficiencyProfile',
    'CurveType',
    'CurveFitResult',
    'fit_efficiency_curve',
    'auto_fit_efficiency_curve',
    # Energy fitting (TASK-2026-008)
    'EnergyFitter',
    'EnergyCoefficients',
    'EnergyFitMetrics',
    'EnergyFitQuality',
    'fit_energy_model',
    # Power model (TASK-2026-008)
    'CalibratedPowerModel',
    'PowerSource',
    'PowerBreakdown',
    'EnergyBreakdown',
    # Utilization fitting (TASK-2026-009)
    'UtilizationFitter',
    'UtilizationFitQuality',
    'UtilizationFitMetrics',
    'UtilizationCurveResult',
    'UtilizationProfile',
    'fit_utilization',
    # Utilization curves (TASK-2026-009)
    'UtilizationCurve',
    'AsymptoticUtilizationCurve',
    'PiecewiseLinearUtilizationCurve',
    'PolynomialUtilizationCurve',
    'ConstantUtilizationCurve',
    'fit_utilization_curve',
    'auto_fit_utilization_curve',
    'create_typical_compute_curve',
    'create_typical_memory_curve',
    'interpolate_utilization',
]
