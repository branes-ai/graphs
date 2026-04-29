"""
Measurement-based validation for the M1-M7 micro-arch model.

Path A of the calibration plan: load existing per-model measurements
from ``calibration_data/`` and compare against the M1-M7 analytical
chain. Surfaces MAPE per (SKU, model, precision) tuple and promotes
the SKU's aggregate confidence from THEORETICAL -> INTERPOLATED when
end-to-end agreement is within tolerance.

This package does not modify the analytical model; it only validates
its outputs against measured ground truth.
"""
from .measurement_comparison import (
    MeasurementSummary,
    MeasurementRecord,  # deprecated alias for back-compat
    PredictionRecord,
    ValidationResult,
    SKUValidationSummary,
    load_measurement,
    list_available_measurements,
    predict_via_unified_analyzer,
    validate_sku,
    compute_mape,
)
from .sku_id_resolution import (
    sku_id_to_calibration_dir,
    sku_id_to_mapper_name,
    sku_id_to_thermal_profile,
)

__all__ = [
    "MeasurementSummary",
    "MeasurementRecord",  # deprecated alias for back-compat
    "PredictionRecord",
    "ValidationResult",
    "SKUValidationSummary",
    "load_measurement",
    "list_available_measurements",
    "predict_via_unified_analyzer",
    "validate_sku",
    "compute_mape",
    "sku_id_to_calibration_dir",
    "sku_id_to_mapper_name",
    "sku_id_to_thermal_profile",
]
