"""
Coefficient Sensitivity Tests

Systematically perturbs key resource-model coefficients by +/-10%
and verifies the model response is proportional: a 10% input change
should produce approximately a 10% output change in the affected
metric, not a 0% change (coefficient is dead) or a 50%+ change
(coefficient is coupled to something unexpected).

Covers all architecture classes via conftest mapper fixtures.
"""

from __future__ import annotations

import pytest

from graphs.hardware.resource_model import Precision


PERTURBATION = 0.10
LOWER_BOUND = 0.05
UPPER_BOUND = 0.25


def _perturb_and_measure_energy(mapper, ops, bytes_t, precision, field, factor):
    """Perturb a resource model field and measure energy delta."""
    original = getattr(mapper.resource_model, field)
    base_c, base_m = mapper._calculate_energy(ops, bytes_t, precision)

    setattr(mapper.resource_model, field, original * factor)
    pert_c, pert_m = mapper._calculate_energy(ops, bytes_t, precision)
    setattr(mapper.resource_model, field, original)

    return base_c, base_m, pert_c, pert_m


def _perturb_and_measure_latency(mapper, ops, bytes_t, precision, field, factor):
    """Perturb a resource model field and measure latency delta."""
    original = getattr(mapper.resource_model, field)
    units = mapper.resource_model.compute_units

    base_ct, base_mt, _ = mapper._calculate_latency(ops, bytes_t, units, 1.0, precision)
    setattr(mapper.resource_model, field, original * factor)
    pert_ct, pert_mt, _ = mapper._calculate_latency(ops, bytes_t, units, 1.0, precision)
    setattr(mapper.resource_model, field, original)

    return base_ct, base_mt, pert_ct, pert_mt


MEDIUM_OPS = 2 * 1024 * 1024 * 1024
MEDIUM_BYTES = 3 * 1024 * 1024 * 4


class TestEnergyPerFlopSensitivity:
    """energy_per_flop_fp32 perturbation across all architecture classes."""

    def _check(self, mappers, precision=Precision.FP32):
        for name, mapper in mappers:
            tile_model = getattr(mapper.resource_model, 'tile_energy_model', None)
            if tile_model is not None:
                continue
            base_c, _, pert_c, _ = _perturb_and_measure_energy(
                mapper, MEDIUM_OPS, MEDIUM_BYTES, precision,
                'energy_per_flop_fp32', 1.0 + PERTURBATION,
            )
            if base_c > 0:
                delta = (pert_c - base_c) / base_c
                assert LOWER_BOUND < delta < UPPER_BOUND, (
                    f"{name}: energy_per_flop +10% -> {delta*100:.1f}% compute energy change"
                )

    def test_cpu(self, cpu_mappers):
        self._check(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        self._check(tpu_mappers)

    def test_kpu(self, kpu_mappers):
        self._check(kpu_mappers, Precision.INT8)

    def test_dsp(self, dsp_mappers):
        self._check(dsp_mappers)

    def test_hailo(self, hailo_mappers):
        self._check(hailo_mappers, Precision.INT8)

    def test_dpu_cgra(self, dpu_cgra_mappers):
        self._check(dpu_cgra_mappers)


class TestEnergyPerByteSensitivity:
    """energy_per_byte perturbation across all architecture classes."""

    def _check(self, mappers, precision=Precision.FP32):
        for name, mapper in mappers:
            tile_model = getattr(mapper.resource_model, 'tile_energy_model', None)
            if tile_model is not None:
                continue
            _, base_m, _, pert_m = _perturb_and_measure_energy(
                mapper, MEDIUM_OPS, MEDIUM_BYTES, precision,
                'energy_per_byte', 1.0 + PERTURBATION,
            )
            if base_m > 0:
                delta = (pert_m - base_m) / base_m
                assert LOWER_BOUND < delta < UPPER_BOUND, (
                    f"{name}: energy_per_byte +10% -> {delta*100:.1f}% memory energy change"
                )

    def test_cpu(self, cpu_mappers):
        self._check(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        self._check(tpu_mappers)

    def test_kpu(self, kpu_mappers):
        self._check(kpu_mappers, Precision.INT8)

    def test_dsp(self, dsp_mappers):
        self._check(dsp_mappers)

    def test_dpu_cgra(self, dpu_cgra_mappers):
        self._check(dpu_cgra_mappers)


class TestBandwidthSensitivity:
    """peak_bandwidth perturbation should affect memory latency proportionally."""

    def _check(self, mappers, precision=Precision.FP32):
        for name, mapper in mappers:
            _, base_mt, _, pert_mt = _perturb_and_measure_latency(
                mapper, MEDIUM_OPS, MEDIUM_BYTES, precision,
                'peak_bandwidth', 0.5,
            )
            if base_mt > 0:
                ratio = pert_mt / base_mt
                assert 1.5 < ratio < 2.5, (
                    f"{name}: halving bandwidth -> {ratio:.2f}x memory latency (expected ~2x)"
                )

    def test_cpu(self, cpu_mappers):
        self._check(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        self._check(tpu_mappers)

    def test_kpu(self, kpu_mappers):
        self._check(kpu_mappers, Precision.INT8)

    def test_dsp(self, dsp_mappers):
        self._check(dsp_mappers)


class TestCrossIsolation:
    """Perturbing one coefficient must not affect the other metric."""

    def _check(self, mappers, precision=Precision.FP32):
        for name, mapper in mappers:
            tile_model = getattr(mapper.resource_model, 'tile_energy_model', None)
            if tile_model is not None:
                continue
            # Perturb energy_per_flop -> memory energy unchanged
            _, base_m, _, pert_m = _perturb_and_measure_energy(
                mapper, MEDIUM_OPS, MEDIUM_BYTES, precision,
                'energy_per_flop_fp32', 1.10,
            )
            if base_m > 0:
                assert abs(pert_m - base_m) / base_m < 0.001, (
                    f"{name}: energy_per_flop perturbation leaked into memory energy"
                )

            # Perturb energy_per_byte -> compute energy unchanged
            base_c, _, pert_c, _ = _perturb_and_measure_energy(
                mapper, MEDIUM_OPS, MEDIUM_BYTES, precision,
                'energy_per_byte', 1.10,
            )
            if base_c > 0:
                assert abs(pert_c - base_c) / base_c < 0.001, (
                    f"{name}: energy_per_byte perturbation leaked into compute energy"
                )

    def test_cpu(self, cpu_mappers):
        self._check(cpu_mappers)

    def test_gpu(self, gpu_mappers):
        self._check(gpu_mappers)

    def test_tpu(self, tpu_mappers):
        self._check(tpu_mappers)

    def test_kpu(self, kpu_mappers):
        self._check(kpu_mappers, Precision.INT8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
