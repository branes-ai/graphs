"""
Tests for HardwareResourceModel.field_provenance and its helper API.

The provenance map lets callers ask "what is the confidence behind this
resource-model field?" directly, rather than scraping source comments.
Bottom-up benchmark fitters write into it as each layer graduates from
THEORETICAL to INTERPOLATED or CALIBRATED.
"""

from __future__ import annotations

import pytest

from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
)


def _make_minimal_model() -> HardwareResourceModel:
    """Construct a HardwareResourceModel with just the required fields."""
    return HardwareResourceModel(
        name="test-sku",
        hardware_type=HardwareType.CPU,
        compute_units=4,
        threads_per_unit=1,
        warps_per_unit=1,
        peak_bandwidth=75e9,
        l1_cache_per_unit=32 * 1024,
        l2_cache_total=8 * 1024 * 1024,
        main_memory=16 * 1024 ** 3,
        energy_per_flop_fp32=1.5e-12,
        energy_per_byte=25e-12,
    )


class TestFieldProvenanceDefaults:
    """Zero-provenance behavior."""

    def test_fresh_model_has_empty_provenance(self):
        model = _make_minimal_model()
        assert model.field_provenance == {}

    def test_unrecorded_field_returns_unknown(self):
        model = _make_minimal_model()
        conf = model.get_provenance("peak_bandwidth")
        assert conf.level is ConfidenceLevel.UNKNOWN
        # get_provenance must return an EstimationConfidence, never None,
        # so callers can safely chain: model.get_provenance(x).level.
        assert isinstance(conf, EstimationConfidence)

    def test_empty_map_aggregates_to_unknown(self):
        model = _make_minimal_model()
        assert model.aggregate_confidence().level is ConfidenceLevel.UNKNOWN


class TestFieldProvenanceSetGet:
    """set/get round-trip and value preservation."""

    def test_set_then_get_preserves_confidence(self):
        model = _make_minimal_model()
        conf = EstimationConfidence.calibrated(
            score=0.92,
            source="layer1_alu_fitter/i7_12700k_avx2_fp32",
            calibration_id="i7_12700k_layer1_fp32_20260416",
        )
        model.set_provenance("energy_per_flop_fp32", conf)

        out = model.get_provenance("energy_per_flop_fp32")
        assert out is conf
        assert out.level is ConfidenceLevel.CALIBRATED
        assert out.score == pytest.approx(0.92)
        assert out.calibration_id == "i7_12700k_layer1_fp32_20260416"

    def test_independent_fields_do_not_interfere(self):
        model = _make_minimal_model()
        model.set_provenance("peak_bandwidth", EstimationConfidence.calibrated())
        model.set_provenance(
            "energy_per_byte", EstimationConfidence.theoretical()
        )

        assert model.get_provenance("peak_bandwidth").level is ConfidenceLevel.CALIBRATED
        assert model.get_provenance("energy_per_byte").level is ConfidenceLevel.THEORETICAL
        assert model.get_provenance("some_other_field").level is ConfidenceLevel.UNKNOWN

    def test_set_overrides_previous_entry(self):
        model = _make_minimal_model()
        model.set_provenance(
            "peak_bandwidth", EstimationConfidence.theoretical(source="datasheet")
        )
        model.set_provenance(
            "peak_bandwidth",
            EstimationConfidence.calibrated(source="STREAM triad @ 75 GB/s"),
        )
        assert model.get_provenance("peak_bandwidth").level is ConfidenceLevel.CALIBRATED

    def test_nested_field_name_is_supported(self):
        model = _make_minimal_model()
        model.set_provenance(
            "thermal.maxn.efficiency_factor",
            EstimationConfidence.interpolated(),
        )
        assert (
            model.get_provenance("thermal.maxn.efficiency_factor").level
            is ConfidenceLevel.INTERPOLATED
        )


class TestAggregateConfidence:
    """aggregate_confidence returns the weakest across all fields."""

    def test_all_calibrated_aggregates_calibrated(self):
        model = _make_minimal_model()
        model.set_provenance("a", EstimationConfidence.calibrated())
        model.set_provenance("b", EstimationConfidence.calibrated())
        assert model.aggregate_confidence().level is ConfidenceLevel.CALIBRATED

    def test_one_theoretical_demotes_aggregate(self):
        model = _make_minimal_model()
        model.set_provenance("good1", EstimationConfidence.calibrated())
        model.set_provenance("good2", EstimationConfidence.calibrated())
        model.set_provenance("weak", EstimationConfidence.theoretical())
        agg = model.aggregate_confidence()
        assert agg.level is ConfidenceLevel.THEORETICAL

    def test_unknown_dominates(self):
        model = _make_minimal_model()
        model.set_provenance("x", EstimationConfidence.calibrated())
        model.set_provenance("y", EstimationConfidence.unknown())
        assert model.aggregate_confidence().level is ConfidenceLevel.UNKNOWN

    def test_mixed_interpolated_theoretical_picks_theoretical(self):
        model = _make_minimal_model()
        model.set_provenance("x", EstimationConfidence.interpolated())
        model.set_provenance("y", EstimationConfidence.theoretical())
        assert model.aggregate_confidence().level is ConfidenceLevel.THEORETICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
