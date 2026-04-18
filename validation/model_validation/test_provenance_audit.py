"""
Provenance Audit Tests

Verifies that every mapper's resource model has essential fields
documented with non-zero values, and that the field_provenance
API works correctly across all mappers.

This test does NOT require field_provenance to be populated with
CALIBRATED entries (that's the job of the layer fitters). It
verifies the infrastructure works and the core fields are present.
"""

from __future__ import annotations

import pytest

from graphs.hardware.resource_model import Precision
from graphs.core.confidence import ConfidenceLevel


class TestResourceModelCompleteness:
    """Every mapper must have the essential modeling fields populated."""

    def _check_mappers(self, mappers, require_main_memory: bool = True):
        for name, mapper in mappers:
            rm = mapper.resource_model
            assert rm.compute_units > 0, f"{name}: zero compute_units"
            assert rm.peak_bandwidth > 0, f"{name}: zero peak_bandwidth"
            assert rm.energy_per_flop_fp32 > 0, f"{name}: zero energy_per_flop_fp32"
            assert rm.energy_per_byte > 0, f"{name}: zero energy_per_byte"
            assert rm.l1_cache_per_unit > 0, f"{name}: zero l1_cache_per_unit"
            if require_main_memory:
                assert rm.main_memory > 0, f"{name}: zero main_memory"

    def test_cpu(self, cpu_mappers):
        self._check_mappers(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check_mappers(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        # Edge TPUs (Coral) may have no external DRAM
        self._check_mappers(tpu_mappers, require_main_memory=False)

    def test_kpu(self, kpu_mappers):
        self._check_mappers(kpu_mappers)

    def test_dsp(self, dsp_mappers):
        self._check_mappers(dsp_mappers)

    def test_hailo(self, hailo_mappers):
        # On-chip accelerators may have no external DRAM
        self._check_mappers(hailo_mappers, require_main_memory=False)

    def test_dpu_cgra(self, dpu_cgra_mappers):
        self._check_mappers(dpu_cgra_mappers)


class TestProvenanceAPI:
    """field_provenance API works across all mappers."""

    def _check(self, mappers):
        for name, mapper in mappers:
            rm = mapper.resource_model

            # Empty provenance returns UNKNOWN
            prov = rm.get_provenance("nonexistent_field")
            assert prov.level is ConfidenceLevel.UNKNOWN

            # Aggregate of empty provenance is UNKNOWN
            if not rm.field_provenance:
                assert rm.aggregate_confidence().level is ConfidenceLevel.UNKNOWN

            # set_provenance round-trips
            from graphs.core.confidence import EstimationConfidence
            rm.set_provenance(
                "_test_field",
                EstimationConfidence.theoretical(source="provenance audit test"),
            )
            prov = rm.get_provenance("_test_field")
            assert prov.level is ConfidenceLevel.THEORETICAL

            # Clean up
            del rm.field_provenance["_test_field"]

    def test_cpu(self, cpu_mappers):
        self._check(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        self._check(tpu_mappers)

    def test_kpu(self, kpu_mappers):
        self._check(kpu_mappers)


class TestPrecisionProfileCompleteness:
    """Every mapper must have at least FP32 or its default precision in profiles."""

    def _check(self, mappers):
        for name, mapper in mappers:
            rm = mapper.resource_model
            has_default = rm.default_precision in rm.precision_profiles
            has_fp32 = Precision.FP32 in rm.precision_profiles
            assert has_default or has_fp32, (
                f"{name}: no precision profile for default ({rm.default_precision}) or FP32"
            )

    def test_cpu(self, cpu_mappers):
        self._check(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        self._check(tpu_mappers)

    def test_kpu(self, kpu_mappers):
        self._check(kpu_mappers)

    def test_dsp(self, dsp_mappers):
        self._check(dsp_mappers)

    def test_dpu_cgra(self, dpu_cgra_mappers):
        self._check(dpu_cgra_mappers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
