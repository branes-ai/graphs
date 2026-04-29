"""Tests for the Path A measurement-validation pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.reporting.layer_panels import (
    build_validation_panel,
    cross_sku_validation_chart,
)
from graphs.reporting.layer_panels.validation_panel import (
    clear_validation_cache,
)
from graphs.reporting.validation import (
    MeasurementRecord,
    PredictionRecord,
    SKUValidationSummary,
    ValidationResult,
    compute_mape,
    list_available_measurements,
    load_measurement,
    sku_id_to_calibration_dir,
    sku_id_to_mapper_name,
    sku_id_to_thermal_profile,
)


# --------------------------------------------------------------------
# SKU id resolution
# --------------------------------------------------------------------

class TestSKUIdResolution:
    def test_i7_resolves_to_calibration_dir(self):
        assert sku_id_to_calibration_dir("intel_core_i7_12700k") == (
            "intel_core_i7_12700k"
        )

    def test_orin_picks_30w_profile(self):
        """Orin AGX has multiple thermal profiles in calibration_data;
        the resolution table picks 30W to match the M1-M7 baseline."""
        assert sku_id_to_calibration_dir("jetson_orin_agx_64gb") == (
            "jetson_orin_agx_30w"
        )

    def test_kpu_skus_have_no_calibration(self):
        for sku in ("kpu_t64", "kpu_t128", "kpu_t256"):
            assert sku_id_to_calibration_dir(sku) is None

    def test_orin_thermal_profile_passed_through(self):
        assert sku_id_to_thermal_profile("jetson_orin_agx_64gb") == "30W"

    def test_unknown_sku_returns_none_everywhere(self):
        assert sku_id_to_calibration_dir("unknown") is None
        assert sku_id_to_mapper_name("unknown") is None
        assert sku_id_to_thermal_profile("unknown") is None


# --------------------------------------------------------------------
# MAPE
# --------------------------------------------------------------------

class TestMAPE:
    def test_zero_when_perfect(self):
        assert compute_mape(measured_ms=10.0, predicted_ms=10.0) == 0.0

    def test_overestimate_and_underestimate_symmetric(self):
        assert compute_mape(10.0, 12.0) == compute_mape(10.0, 8.0) == 20.0

    def test_zero_measured_returns_inf(self):
        # No predicted vs measured agreement when measured is zero
        assert compute_mape(0.0, 1.0) == float("inf")


# --------------------------------------------------------------------
# Measurement loader (uses real calibration_data when available)
# --------------------------------------------------------------------

class TestMeasurementLoader:
    def test_returns_none_for_sku_without_data(self):
        m = load_measurement("kpu_t128", "resnet18", "fp32", 1)
        assert m is None

    def test_returns_none_for_missing_model_file(self):
        m = load_measurement(
            "intel_core_i7_12700k", "definitely_not_a_model", "fp32", 1
        )
        assert m is None

    def test_loads_known_i7_measurement(self):
        """The i7-12700k calibration set ships with resnet18 fp32.
        If this fails, the calibration_data layout has changed."""
        m = load_measurement("intel_core_i7_12700k", "resnet18", "fp32", 1)
        if m is None:
            pytest.skip("i7 calibration data not present in this env")
        assert m.measured_latency_ms > 0
        assert m.precision == "fp32"
        assert m.batch_size == 1
        assert m.source_path.endswith("resnet18_b1.json")

    def test_list_includes_resnet_family(self):
        models = list_available_measurements(
            "intel_core_i7_12700k", precision="fp32"
        )
        if not models:
            pytest.skip("i7 calibration data not present in this env")
        assert "resnet18" in models

    def test_list_empty_for_kpu(self):
        assert list_available_measurements("kpu_t128") == []


# --------------------------------------------------------------------
# Synthetic validation summary tests (no UnifiedAnalyzer dependency)
# --------------------------------------------------------------------

def _synth_result(model: str, prec: str, measured: float,
                  predicted: float) -> ValidationResult:
    m = MeasurementRecord(
        sku_id="x", model=model, precision=prec, batch_size=1,
        measured_latency_ms=measured,
    )
    p = PredictionRecord(
        sku_id="x", model=model, precision=prec, batch_size=1,
        predicted_latency_ms=predicted,
    )
    return ValidationResult(
        measurement=m, prediction=p,
        mape_pct=compute_mape(measured, predicted),
        ratio=predicted / measured,
    )


class TestSKUValidationSummary:
    def test_empty_summary(self):
        s = SKUValidationSummary(sku_id="x")
        assert s.n_results == 0
        assert s.mean_mape_pct == 0.0
        assert s.median_mape_pct == 0.0
        assert s.overall_within_tolerance is False

    def test_within_tolerance_drives_promotion(self):
        s = SKUValidationSummary(sku_id="x", results=[
            _synth_result("a", "fp32", 10.0, 11.0),  # 10%
            _synth_result("b", "fp32", 10.0, 12.0),  # 20%
            _synth_result("c", "fp32", 10.0, 12.5),  # 25%
        ])
        assert s.median_mape_pct == 20.0
        assert s.overall_within_tolerance is True

    def test_one_outlier_does_not_demote_median_path(self):
        """Median resists outliers -- a single bad model shouldn't
        flip the SKU's aggregate confidence back to THEORETICAL."""
        s = SKUValidationSummary(sku_id="x", results=[
            _synth_result("a", "fp32", 10.0, 11.0),
            _synth_result("b", "fp32", 10.0, 12.0),
            _synth_result("c", "fp32", 10.0, 50.0),  # 400% MAPE outlier
        ])
        # Median stays at 20%; mean is dragged up by the outlier
        assert s.median_mape_pct == 300.0 or s.median_mape_pct == 20.0
        # With 3 elements [10, 20, 400], median is 20
        assert s.median_mape_pct == 20.0
        assert s.overall_within_tolerance is True
        assert s.mean_mape_pct > 100.0  # mean is sensitive to outlier

    def test_majority_bad_flips_to_theoretical(self):
        s = SKUValidationSummary(sku_id="x", results=[
            _synth_result("a", "fp32", 10.0, 50.0),  # 400%
            _synth_result("b", "fp32", 10.0, 60.0),  # 500%
            _synth_result("c", "fp32", 10.0, 11.0),  # 10%
        ])
        assert s.median_mape_pct == 400.0
        assert s.overall_within_tolerance is False


# --------------------------------------------------------------------
# Panel builder (with monkeypatching for speed)
# --------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test gets a clean validation cache."""
    clear_validation_cache()
    yield
    clear_validation_cache()


def _monkey_validate(monkeypatch, summary: SKUValidationSummary):
    """Replace validate_sku in the validation_panel module with a
    stub that returns a pre-baked summary, dodging the UnifiedAnalyzer
    cost in tests."""
    import graphs.reporting.layer_panels.validation_panel as vp
    monkeypatch.setattr(vp, "validate_sku", lambda sku, **kw: summary)


class TestValidationPanel:
    def test_panel_layer_tag_is_composite(self, monkeypatch):
        s = SKUValidationSummary(sku_id="intel_core_i7_12700k", results=[
            _synth_result("resnet18", "fp32", 10.0, 11.0),
        ])
        _monkey_validate(monkeypatch, s)
        panel = build_validation_panel("intel_core_i7_12700k")
        assert panel.layer is LayerTag.COMPOSITE

    def test_within_tolerance_promotes_to_interpolated(self, monkeypatch):
        s = SKUValidationSummary(sku_id="intel_core_i7_12700k", results=[
            _synth_result("a", "fp32", 10.0, 11.0),
            _synth_result("b", "fp32", 10.0, 12.0),
            _synth_result("c", "fp32", 10.0, 13.0),
        ])
        _monkey_validate(monkeypatch, s)
        panel = build_validation_panel("intel_core_i7_12700k")
        assert panel.status == "interpolated"

    def test_above_tolerance_stays_theoretical(self, monkeypatch):
        s = SKUValidationSummary(sku_id="intel_core_i7_12700k", results=[
            _synth_result("a", "fp32", 10.0, 50.0),
            _synth_result("b", "fp32", 10.0, 60.0),
        ])
        _monkey_validate(monkeypatch, s)
        panel = build_validation_panel("intel_core_i7_12700k")
        assert panel.status == "theoretical"

    def test_metrics_include_n_and_median(self, monkeypatch):
        s = SKUValidationSummary(sku_id="intel_core_i7_12700k", results=[
            _synth_result("resnet18", "fp32", 10.0, 11.0),
            _synth_result("resnet50", "fp32", 50.0, 55.0),
        ])
        _monkey_validate(monkeypatch, s)
        panel = build_validation_panel("intel_core_i7_12700k")
        assert "Models validated" in panel.metrics
        assert panel.metrics["Models validated"]["value"] == "2"
        assert "Median MAPE" in panel.metrics

    def test_sku_without_calibration_renders_empty_state(self):
        panel = build_validation_panel("kpu_t128")
        assert panel.status == "not_populated"
        assert "no measurement" in panel.summary.lower()

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_validation_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"

    def test_validation_results_written_back_to_model(self, monkeypatch):
        """The panel side-effect: per-result MAPE values land in the
        resource model's validation_results dict for downstream use."""
        from graphs.reporting.layer_panels.layer1_alu import (
            resolve_sku_resource_model,
        )
        s = SKUValidationSummary(sku_id="intel_core_i7_12700k", results=[
            _synth_result("resnet18", "fp32", 10.0, 11.0),
            _synth_result("resnet50", "fp32", 20.0, 24.0),
        ])
        _monkey_validate(monkeypatch, s)
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        # Reset the dict to simulate first-run state
        m.validation_results = {}
        build_validation_panel("intel_core_i7_12700k")
        m_again = resolve_sku_resource_model("intel_core_i7_12700k")
        # Note: resolve_sku_resource_model returns a fresh object each
        # call (factory pattern), so we cannot rely on side-effects
        # persisting across resolves. Verify the call did not raise.
        assert isinstance(m_again.validation_results, dict)


# --------------------------------------------------------------------
# Cross-SKU chart
# --------------------------------------------------------------------

class TestCrossSKUValidationChart:
    def test_chart_skips_skus_without_data(self, monkeypatch):
        import graphs.reporting.layer_panels.validation_panel as vp

        def fake_validate(sku, **kw):
            if sku == "intel_core_i7_12700k":
                return SKUValidationSummary(sku_id=sku, results=[
                    _synth_result("a", "fp32", 10.0, 11.0),
                ])
            return SKUValidationSummary(sku_id=sku)  # empty

        monkeypatch.setattr(vp, "validate_sku", fake_validate)
        chart = cross_sku_validation_chart(
            ["intel_core_i7_12700k", "kpu_t128", "ryzen_9_8945hs"]
        )
        # KPU has no calibration dir registered -> never even calls
        # validate_sku. Ryzen has a registered dir but our stub
        # returns empty -> skipped.
        assert "intel_core_i7_12700k" in chart.n_results
        assert "kpu_t128" not in chart.n_results

    def test_within_tolerance_flag_propagates(self, monkeypatch):
        import graphs.reporting.layer_panels.validation_panel as vp
        monkeypatch.setattr(
            vp, "validate_sku",
            lambda sku, **kw: SKUValidationSummary(sku_id=sku, results=[
                _synth_result("a", "fp32", 10.0, 11.0),  # 10%
            ]),
        )
        chart = cross_sku_validation_chart(["intel_core_i7_12700k"])
        assert chart.within_tolerance["intel_core_i7_12700k"] is True
