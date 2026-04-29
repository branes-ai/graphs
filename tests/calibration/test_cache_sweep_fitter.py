"""Tests for CacheSweepFitter (Path B PR-1)."""
from __future__ import annotations

from graphs.benchmarks.cache_sweep import WorkingSetPoint
from graphs.calibration.fitters.cache_sweep_fitter import (
    CacheFitResult,
    CacheSweepFitter,
)
from graphs.core.confidence import ConfidenceLevel


def _point(bytes_resident: int, bandwidth: float,
           energy_pj: float = None) -> WorkingSetPoint:
    return WorkingSetPoint(
        bytes_resident=bytes_resident,
        iterations=1,
        elapsed_seconds=0.1,
        bandwidth_gbps=bandwidth,
        energy_per_byte_pj=energy_pj,
    )


_CANONICAL = [
    _point(8 * 1024,    200.0, 0.20),
    _point(32 * 1024,   200.0, 0.20),
    _point(256 * 1024,   80.0, 0.50),
    _point(1024 * 1024,  80.0, 0.50),
    _point(4 * 1024**2,  35.0, 1.20),
    _point(16 * 1024**2, 35.0, 1.20),
    _point(64 * 1024**2,  8.0, 15.0),
]


class _StubModel:
    """Minimal HardwareResourceModel stand-in for testing
    apply_to_model without pulling in the full dataclass."""

    def __init__(self):
        self.field_provenance = {}

    def set_provenance(self, name, conf):
        self.field_provenance[name] = conf


class TestCacheSweepFitter:
    def test_fit_returns_per_level_energies(self):
        result = CacheSweepFitter().fit(_CANONICAL, sku_name="i7-12700k")
        assert isinstance(result, CacheFitResult)
        assert result.l1_energy_per_byte_pj == 0.20
        assert result.l2_energy_per_byte_pj == 0.50
        assert result.l3_energy_per_byte_pj == 1.20
        assert result.l1_bandwidth_gbps == 200.0
        assert result.l2_bandwidth_gbps == 80.0
        assert result.l3_bandwidth_gbps == 35.0
        assert result.num_points == len(_CANONICAL)

    def test_fit_records_sku_name(self):
        result = CacheSweepFitter().fit(_CANONICAL, sku_name="i7-12700k")
        assert result.sku_name == "i7-12700k"

    def test_apply_to_model_writes_calibrated(self):
        model = _StubModel()
        result = CacheSweepFitter().fit(_CANONICAL, sku_name="i7-12700k")
        CacheSweepFitter.apply_to_model(model, result)

        for field in ("l1_cache_energy_per_byte",
                      "l2_cache_energy_per_byte",
                      "l3_cache_energy_per_byte"):
            assert field in model.field_provenance
            conf = model.field_provenance[field]
            assert conf.level is ConfidenceLevel.CALIBRATED
            assert "cache_sweep_fitter" in conf.source

    def test_apply_to_model_skips_levels_without_energy(self):
        """When RAPL is unavailable for some levels, only the levels
        with measured energy get tagged CALIBRATED; the rest stay
        at whatever the model already had (THEORETICAL by default)."""
        no_energy = [
            _point(8 * 1024,   200.0, None),   # L1: no energy
            _point(32 * 1024,  200.0, None),
            _point(256 * 1024,  80.0, 0.50),   # L2: has energy
            _point(4 * 1024**2, 35.0, 1.20),   # L3: has energy
        ]
        model = _StubModel()
        result = CacheSweepFitter().fit(no_energy, sku_name="x")
        CacheSweepFitter.apply_to_model(model, result)

        assert "l1_cache_energy_per_byte" not in model.field_provenance
        assert "l2_cache_energy_per_byte" in model.field_provenance
        assert "l3_cache_energy_per_byte" in model.field_provenance

    def test_provenance_source_includes_bandwidth_and_point_count(self):
        model = _StubModel()
        result = CacheSweepFitter().fit(_CANONICAL, sku_name="i7-12700k")
        CacheSweepFitter.apply_to_model(model, result)
        src = model.field_provenance["l1_cache_energy_per_byte"].source
        assert "i7-12700k" in src
        assert "200" in src or "0.2" in src or "GB/s" in src
        assert "sweep points" in src

    def test_apply_handles_model_without_set_provenance(self):
        """Models that don't expose set_provenance (legacy paths)
        should not crash apply_to_model."""

        class Legacy:
            pass

        result = CacheSweepFitter().fit(_CANONICAL, sku_name="x")
        CacheSweepFitter.apply_to_model(Legacy(), result)  # no exception


class TestNoCalibration:
    def test_empty_sweep_returns_empty_result(self):
        result = CacheSweepFitter().fit([], sku_name="x")
        assert result.l1_energy_per_byte_pj is None
        assert result.l2_energy_per_byte_pj is None
        assert result.l3_energy_per_byte_pj is None
        assert result.num_points == 0

    def test_empty_sweep_apply_writes_nothing(self):
        model = _StubModel()
        CacheSweepFitter.apply_to_model(
            model, CacheSweepFitter().fit([], sku_name="x")
        )
        assert model.field_provenance == {}
