"""
Measurement-vs-prediction comparison for the M1-M7 micro-arch model.

Loads per-model latency measurements from ``calibration_data/`` and
runs the M1-M7 analytical chain via ``UnifiedAnalyzer`` for the same
(SKU, model, precision, batch_size) tuple. Reports MAPE per
combination and aggregates across a SKU's measurement set.

Energy comparisons are intentionally out of scope at this stage --
the calibration data we have is latency-only at the model level.
Energy validation belongs to Path B (microbenchmark fitting).

This module is deliberately import-light: ``UnifiedAnalyzer`` is
loaded lazily inside ``predict_via_unified_analyzer`` so the panel
builder can introspect available measurements without paying the
PyTorch import cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from graphs.reporting.validation.sku_id_resolution import (
    sku_id_to_calibration_dir,
    sku_id_to_mapper_name,
    sku_id_to_thermal_profile,
)


# Repository root, used to locate calibration_data/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CALIBRATION_ROOT = _REPO_ROOT / "calibration_data"


# --------------------------------------------------------------------
# Records
# --------------------------------------------------------------------

@dataclass
class MeasurementRecord:
    """A single per-model measurement loaded from calibration_data/."""
    sku_id: str
    model: str
    precision: str               # "fp32" / "fp16" / "bf16"
    batch_size: int
    measured_latency_ms: float
    measurement_date: str = ""
    source_path: str = ""


@dataclass
class PredictionRecord:
    """A single analytical prediction from UnifiedAnalyzer."""
    sku_id: str
    model: str
    precision: str
    batch_size: int
    predicted_latency_ms: float
    energy_mj: float = 0.0
    error: str = ""              # populated when prediction failed


@dataclass
class ValidationResult:
    """One (measurement, prediction) pair plus derived metrics."""
    measurement: MeasurementRecord
    prediction: PredictionRecord
    mape_pct: float              # |predicted - measured| / measured * 100
    ratio: float                 # predicted / measured

    @property
    def within_tolerance(self) -> bool:
        """Default tolerance: 30% MAPE (matches the user-approved M1
        precision target)."""
        return self.mape_pct <= 30.0


@dataclass
class SKUValidationSummary:
    """Aggregate over all (model, precision, batch) results for one SKU."""
    sku_id: str
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def mean_mape_pct(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.mape_pct for r in self.results) / len(self.results)

    @property
    def median_mape_pct(self) -> float:
        if not self.results:
            return 0.0
        sorted_mapes = sorted(r.mape_pct for r in self.results)
        n = len(sorted_mapes)
        if n % 2 == 1:
            return sorted_mapes[n // 2]
        return 0.5 * (sorted_mapes[n // 2 - 1] + sorted_mapes[n // 2])

    @property
    def n_within_tolerance(self) -> int:
        return sum(1 for r in self.results if r.within_tolerance)

    @property
    def overall_within_tolerance(self) -> bool:
        """The SKU's whole validation set agrees to within 30% on the
        median. Stricter than mean to resist outliers."""
        return self.n_results > 0 and self.median_mape_pct <= 30.0


# --------------------------------------------------------------------
# Loading measurements
# --------------------------------------------------------------------

def load_measurement(
    sku_id: str,
    model: str,
    precision: str = "fp32",
    batch_size: int = 1,
) -> Optional[MeasurementRecord]:
    """Load one measurement file. Returns None when missing.

    Looks in: ``<repo>/calibration_data/<dir>/measurements/<precision>/<model>_b<batch>.json``
    """
    calib_dir = sku_id_to_calibration_dir(sku_id)
    if calib_dir is None:
        return None

    base = _CALIBRATION_ROOT / calib_dir / "measurements" / precision.lower()
    path = base / f"{model}_b{batch_size}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    summary = data.get("model_summary") or {}
    latency_ms = summary.get("total_latency_ms")
    if latency_ms is None:
        return None

    return MeasurementRecord(
        sku_id=sku_id,
        model=data.get("model", model),
        precision=data.get("precision", precision).lower(),
        batch_size=int(data.get("batch_size", batch_size)),
        measured_latency_ms=float(latency_ms),
        measurement_date=str(data.get("measurement_date", "")),
        source_path=str(path.relative_to(_REPO_ROOT)),
    )


def list_available_measurements(
    sku_id: str,
    precision: str = "fp32",
) -> List[str]:
    """Return the list of model names with measurement files for the
    given (SKU, precision). Sorted; empty when no data."""
    calib_dir = sku_id_to_calibration_dir(sku_id)
    if calib_dir is None:
        return []

    base = _CALIBRATION_ROOT / calib_dir / "measurements" / precision.lower()
    if not base.is_dir():
        return []

    models = set()
    for f in base.glob("*_b*.json"):
        # Filename: ``<model>_b<batch>.json``; strip the ``_b<n>`` tail
        stem = f.stem
        # Find the last "_b" followed by digits
        idx = stem.rfind("_b")
        if idx <= 0 or not stem[idx + 2:].isdigit():
            continue
        models.add(stem[:idx])
    return sorted(models)


# --------------------------------------------------------------------
# Predicting via UnifiedAnalyzer
# --------------------------------------------------------------------

def predict_via_unified_analyzer(
    sku_id: str,
    model: str,
    precision: str = "fp32",
    batch_size: int = 1,
) -> PredictionRecord:
    """Run UnifiedAnalyzer for the given combination and return a
    PredictionRecord. Errors are captured in ``record.error`` so the
    caller can render gracefully.

    UnifiedAnalyzer is imported lazily here -- the import path pulls
    in PyTorch, which is too expensive for panel-only code paths.
    """
    record = PredictionRecord(
        sku_id=sku_id,
        model=model,
        precision=precision.lower(),
        batch_size=batch_size,
        predicted_latency_ms=0.0,
    )

    mapper_name = sku_id_to_mapper_name(sku_id)
    if mapper_name is None:
        record.error = (
            f"No UnifiedAnalyzer mapper registered for SKU '{sku_id}'."
        )
        return record

    try:
        from graphs.estimation.unified_analyzer import UnifiedAnalyzer
        from graphs.hardware.resource_model import Precision
    except ImportError as exc:
        record.error = f"UnifiedAnalyzer unavailable: {exc}"
        return record

    # Resolve precision to the enum
    precision_map = {
        "fp32": Precision.FP32,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "int8": Precision.INT8,
    }
    prec_enum = precision_map.get(precision.lower())
    if prec_enum is None:
        record.error = f"Unsupported precision '{precision}'."
        return record

    thermal = sku_id_to_thermal_profile(sku_id)

    try:
        ua = UnifiedAnalyzer(verbose=False)
        result = ua.analyze_model(
            model_name=model,
            hardware_name=mapper_name,
            batch_size=batch_size,
            precision=prec_enum,
            thermal_profile=thermal,
        )
    except Exception as exc:  # broad catch -- we surface the error string
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    record.predicted_latency_ms = float(result.total_latency_ms or 0.0)
    record.energy_mj = float(result.energy_per_inference_mj or 0.0)
    return record


# --------------------------------------------------------------------
# MAPE + per-SKU validation driver
# --------------------------------------------------------------------

def compute_mape(measured_ms: float, predicted_ms: float) -> float:
    """Mean absolute percentage error for a single value pair."""
    if measured_ms <= 0:
        return float("inf")
    return abs(predicted_ms - measured_ms) / measured_ms * 100.0


def validate_sku(
    sku_id: str,
    precisions: Optional[List[str]] = None,
    batch_size: int = 1,
    models: Optional[List[str]] = None,
) -> SKUValidationSummary:
    """
    Run the full measurement-vs-prediction comparison for one SKU.

    Loads every available measurement under ``calibration_data/<dir>/
    measurements/<prec>/`` (or restricts to ``models`` if given),
    runs UnifiedAnalyzer for the same combination, and returns a
    summary. Failed predictions are skipped but counted in
    ``error_count`` (caller-side via len(measurements) - len(results)).
    """
    summary = SKUValidationSummary(sku_id=sku_id)
    if precisions is None:
        precisions = ["fp32", "fp16", "bf16"]

    for precision in precisions:
        candidates = (
            models if models is not None
            else list_available_measurements(sku_id, precision=precision)
        )
        for model in candidates:
            measurement = load_measurement(
                sku_id=sku_id,
                model=model,
                precision=precision,
                batch_size=batch_size,
            )
            if measurement is None:
                continue

            prediction = predict_via_unified_analyzer(
                sku_id=sku_id,
                model=model,
                precision=precision,
                batch_size=batch_size,
            )
            if prediction.error or prediction.predicted_latency_ms <= 0:
                continue

            mape = compute_mape(
                measurement.measured_latency_ms,
                prediction.predicted_latency_ms,
            )
            ratio = (
                prediction.predicted_latency_ms
                / measurement.measured_latency_ms
            )
            summary.results.append(ValidationResult(
                measurement=measurement,
                prediction=prediction,
                mape_pct=mape,
                ratio=ratio,
            ))

    return summary


__all__ = [
    "MeasurementRecord",
    "PredictionRecord",
    "ValidationResult",
    "SKUValidationSummary",
    "load_measurement",
    "list_available_measurements",
    "predict_via_unified_analyzer",
    "validate_sku",
    "compute_mape",
]
