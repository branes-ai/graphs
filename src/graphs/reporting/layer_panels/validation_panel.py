"""
Path A: end-to-end measurement validation panel.

Builds a ``LayerTag.COMPOSITE``-tagged panel that surfaces
predicted-vs-measured MAPE per (model, precision) tuple for each
SKU that has measurement data. SKUs without measurements render
"No measurement data; SKU stays at THEORETICAL aggregate" -- not a
crash, not an empty section.

Promotion semantics:
- ``median_mape_pct <= 30%`` -> aggregate confidence promotes from
  THEORETICAL -> INTERPOLATED (panel badge becomes "interpolated")
- ``median_mape_pct >  30%`` -> SKU stays THEORETICAL with the
  numeric MAPE surfaced so a reader can judge the drift
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from graphs.benchmarks.schema import LayerTag
from graphs.reporting.microarch_schema import LayerPanel
from graphs.reporting.layer_panels.layer1_alu import resolve_sku_resource_model
from graphs.reporting.validation import (
    SKUValidationSummary,
    sku_id_to_calibration_dir,
    validate_sku,
)


@dataclass
class _CachedValidation:
    """Module-level cache: validation results are expensive to
    compute (each prediction calls UnifiedAnalyzer / FX-traces a
    PyTorch model). Hold a single in-process cache keyed by SKU id."""
    cache: Dict[str, SKUValidationSummary] = field(default_factory=dict)


_CACHE = _CachedValidation()


def _validate_or_cached(sku_id: str) -> Optional[SKUValidationSummary]:
    """Run ``validate_sku`` once per SKU id within a process; cache
    the result. Returns None for SKUs without measurement data so
    the caller can render the empty state without paying for an
    UnifiedAnalyzer import."""
    if sku_id_to_calibration_dir(sku_id) is None:
        return None
    if sku_id in _CACHE.cache:
        return _CACHE.cache[sku_id]
    summary = validate_sku(sku_id)
    _CACHE.cache[sku_id] = summary
    return summary


def clear_validation_cache() -> None:
    """Test helper: drop the cache between scenarios."""
    _CACHE.cache.clear()


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_validation_panel(
    sku_id: str,
) -> LayerPanel:
    """Build the Path A validation panel for one SKU.

    Tagged ``LayerTag.COMPOSITE`` so it renders alongside the seven
    layer panels without expanding the LayerTag enum.
    """
    title = "Validation: predicted vs measured"
    model = resolve_sku_resource_model(sku_id)
    if model is None:
        # Use ``theoretical`` here rather than ``not_populated``: the
        # generic renderer collapses every ``not_populated`` panel to
        # a "NOT YET POPULATED" placeholder and drops the custom
        # summary / notes. A theoretical-tagged panel preserves the
        # explanatory text we want the reader to see.
        return LayerPanel(
            layer=LayerTag.COMPOSITE,
            title=title,
            status="theoretical",
            summary=f"Resource model unavailable for {sku_id}.",
        )

    summary_obj = _validate_or_cached(sku_id)
    if summary_obj is None:
        return LayerPanel(
            layer=LayerTag.COMPOSITE,
            title=title,
            status="theoretical",
            summary=(
                f"No measurement data registered for {sku_id}; the "
                "SKU's aggregate confidence stays at THEORETICAL "
                "until a calibration campaign lands."
            ),
            notes=[
                "Path A validates the M1-M7 analytical chain end-to-end "
                "against per-model latency measurements in "
                "calibration_data/. Add an entry to "
                "graphs.reporting.validation.sku_id_resolution to "
                "register measurement data for additional SKUs."
            ],
        )

    if summary_obj.n_results == 0:
        return LayerPanel(
            layer=LayerTag.COMPOSITE,
            title=title,
            status="theoretical",
            summary=(
                f"Calibration data registered for {sku_id} but no "
                "measurement files matched the available models / "
                "precisions; check calibration_data/ contents."
            ),
        )

    # The per-result MAPE values are surfaced through the panel's
    # own metrics dict (below) which is what flows into the JSON
    # export. We deliberately avoid mutating ``model`` here -- since
    # ``resolve_sku_resource_model`` returns a fresh instance each
    # call, mutations on it would not survive a second resolution.
    # Downstream consumers should read from ``report.layers[i]
    # .metrics`` instead.

    metrics: Dict[str, Dict] = {
        "Models validated": {
            "value": str(summary_obj.n_results),
            "unit":  "",
            "provenance": "n/a",
        },
        "Median MAPE": {
            "value": f"{summary_obj.median_mape_pct:.1f}",
            "unit":  "%",
            "provenance": "CALIBRATED",
        },
        "Mean MAPE": {
            "value": f"{summary_obj.mean_mape_pct:.1f}",
            "unit":  "%",
            "provenance": "CALIBRATED",
        },
        "Within tolerance": {
            "value": f"{summary_obj.n_within_tolerance}/{summary_obj.n_results}",
            "unit":  "",
            "provenance": "n/a",
        },
    }

    # Top-3 best and top-3 worst per-model entries for quick scanning
    sorted_results = sorted(summary_obj.results, key=lambda r: r.mape_pct)
    for r in sorted_results[:3]:
        key = f"Best ({r.measurement.model}:{r.measurement.precision})"
        metrics[key] = {
            "value": f"{r.mape_pct:.1f}",
            "unit":  "% MAPE",
            "provenance": "CALIBRATED",
        }
    if len(sorted_results) >= 4:
        for r in sorted_results[-3:]:
            key = f"Worst ({r.measurement.model}:{r.measurement.precision})"
            metrics[key] = {
                "value": f"{r.mape_pct:.1f}",
                "unit":  "% MAPE",
                "provenance": "CALIBRATED",
            }

    if summary_obj.overall_within_tolerance:
        status = "interpolated"
        narrative = (
            f"End-to-end validation: median MAPE "
            f"{summary_obj.median_mape_pct:.1f}% across "
            f"{summary_obj.n_results} models -- within the +/-30% "
            "M1 tolerance. SKU aggregate confidence promotes from "
            "THEORETICAL to INTERPOLATED."
        )
    else:
        status = "theoretical"
        narrative = (
            f"End-to-end validation: median MAPE "
            f"{summary_obj.median_mape_pct:.1f}% across "
            f"{summary_obj.n_results} models -- exceeds the +/-30% "
            "tolerance. The drift is reported per-model so a "
            "downstream Path B microbenchmark campaign can target "
            "the layers most responsible."
        )

    notes = [
        narrative,
        "Predicted latency is from the M1-M7 analytical chain via "
        "UnifiedAnalyzer. Measured latency is from the per-model "
        "files under calibration_data/.",
    ]

    return LayerPanel(
        layer=LayerTag.COMPOSITE,
        title=title,
        status=status,
        summary=(
            f"Validated against {summary_obj.n_results} measured "
            f"models for {sku_id}; "
            f"median MAPE {summary_obj.median_mape_pct:.1f}%."
        ),
        metrics=metrics,
        notes=notes,
    )


# --------------------------------------------------------------------
# Cross-SKU summary
# --------------------------------------------------------------------

@dataclass
class CrossSKUValidationChart:
    skus: List[str]
    n_results: Dict[str, int]
    median_mape_pct: Dict[str, float]
    within_tolerance: Dict[str, bool]


def cross_sku_validation_chart(
    sku_ids: List[str],
) -> CrossSKUValidationChart:
    chart = CrossSKUValidationChart(
        skus=list(sku_ids),
        n_results={},
        median_mape_pct={},
        within_tolerance={},
    )
    for sku in sku_ids:
        summary_obj = _validate_or_cached(sku)
        if summary_obj is None or summary_obj.n_results == 0:
            continue
        chart.n_results[sku] = summary_obj.n_results
        chart.median_mape_pct[sku] = summary_obj.median_mape_pct
        chart.within_tolerance[sku] = summary_obj.overall_within_tolerance
    return chart


__all__ = [
    "build_validation_panel",
    "cross_sku_validation_chart",
    "CrossSKUValidationChart",
    "clear_validation_cache",
]
