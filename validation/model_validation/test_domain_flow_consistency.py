"""
DomainFlow (KPU) Model Consistency Tests

5-category consistency checks for Stillwater KPU mappers (T64, T256, T768).
"""

from __future__ import annotations

import pytest
from graphs.hardware.resource_model import Precision


class TestLayerDecomposition:

    def test_baseline_energy_is_positive(self, kpu_mappers, medium_matmul):
        for name, mapper in kpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.INT8)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"

    def test_architectural_overhead_adds_to_baseline(self, kpu_mappers, medium_matmul):
        for name, mapper in kpu_mappers:
            if mapper.resource_model.architecture_energy_model is None:
                continue
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            base_c, base_m = mapper._calculate_energy(ops, bytes_t, Precision.INT8)
            arch_c, arch_m, breakdown = mapper._calculate_energy_with_architecture(
                ops, bytes_t, Precision.INT8
            )
            assert breakdown is not None, f"{name}: no breakdown"
            assert arch_c >= base_c * 0.99, f"{name}: arch compute < base"


class TestMonotonicity:

    def test_more_ops_means_more_energy(self, kpu_mappers):
        for name, mapper in kpu_mappers:
            e_small = sum(mapper._calculate_energy(1_000_000, 4_000_000, Precision.INT8))
            e_large = sum(mapper._calculate_energy(100_000_000, 4_000_000, Precision.INT8))
            assert e_large > e_small, f"{name}: more ops did not increase energy"

    def test_more_bytes_means_more_energy(self, kpu_mappers):
        for name, mapper in kpu_mappers:
            e_small = sum(mapper._calculate_energy(10_000_000, 1_000_000, Precision.INT8))
            e_large = sum(mapper._calculate_energy(10_000_000, 100_000_000, Precision.INT8))
            assert e_large > e_small, f"{name}: more bytes did not increase energy"


class TestPrecisionScaling:

    def test_int8_cheaper_than_fp32(self, kpu_mappers):
        for name, mapper in kpu_mappers:
            e_fp32 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.FP32)
            e_int8 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT8)
            assert e_int8[0] < e_fp32[0], f"{name}: INT8 compute >= FP32"

    def test_int4_cheaper_than_int8(self, kpu_mappers):
        for name, mapper in kpu_mappers:
            e_int8 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT8)
            e_int4 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT4)
            assert e_int4[0] < e_int8[0], f"{name}: INT4 compute >= INT8"

    def test_energy_scaling_in_range(self, kpu_mappers):
        for name, mapper in kpu_mappers:
            for prec, scale in mapper.resource_model.energy_scaling.items():
                assert 0 < scale <= 4.0, f"{name}: energy_scaling[{prec}] = {scale}"


class TestCoefficientSensitivity:

    def test_energy_per_flop_perturbation(self, kpu_mappers, medium_matmul):
        for name, mapper in kpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            base_c, _ = mapper._calculate_energy(ops, bytes_t, Precision.INT8)
            original = mapper.resource_model.energy_per_flop_fp32
            mapper.resource_model.energy_per_flop_fp32 = original * 1.10
            pert_c, _ = mapper._calculate_energy(ops, bytes_t, Precision.INT8)
            mapper.resource_model.energy_per_flop_fp32 = original
            if base_c > 0:
                delta = (pert_c - base_c) / base_c
                assert 0.05 < delta < 0.20, f"{name}: {delta*100:.1f}% (expected ~10%)"


class TestCrossMapperOrdering:

    def test_t768_has_more_tiles_than_t64(self):
        from graphs.hardware.mappers import get_mapper_by_name
        try:
            t768 = get_mapper_by_name("Stillwater-KPU-T768")
            t64 = get_mapper_by_name("Stillwater-KPU-T64")
        except (KeyError, ValueError, TypeError, AttributeError):
            pytest.skip("Required mappers not found")
        assert t768.resource_model.compute_units > t64.resource_model.compute_units

    def test_t256_between_t64_and_t768(self):
        from graphs.hardware.mappers import get_mapper_by_name
        try:
            t64 = get_mapper_by_name("Stillwater-KPU-T64")
            t256 = get_mapper_by_name("Stillwater-KPU-T256")
            t768 = get_mapper_by_name("Stillwater-KPU-T768")
        except (KeyError, ValueError, TypeError, AttributeError):
            pytest.skip("Required mappers not found")
        assert t64.resource_model.compute_units < t256.resource_model.compute_units < t768.resource_model.compute_units

    def test_no_kpu_has_negative_latency(self, kpu_mappers, medium_matmul):
        for name, mapper in kpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            ct, mt, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0, precision=Precision.INT8,
            )
            assert ct >= 0, f"{name}: negative compute_time"
            assert mt >= 0, f"{name}: negative memory_time"

    def test_no_kpu_has_zero_bandwidth(self, kpu_mappers):
        for name, mapper in kpu_mappers:
            assert mapper.resource_model.peak_bandwidth > 0, f"{name}: zero bandwidth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
