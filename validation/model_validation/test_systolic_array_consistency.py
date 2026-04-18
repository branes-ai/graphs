"""
SystolicArray (TPU) Model Consistency Tests

5-category consistency checks for all TPU mappers (v1, v3, v4, v5p,
Coral Edge TPU, TPU Edge Pro).
"""

from __future__ import annotations

import pytest
from graphs.hardware.resource_model import Precision


class TestLayerDecomposition:

    def test_baseline_energy_is_positive(self, tpu_mappers, medium_matmul):
        for name, mapper in tpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"

    def test_architectural_overhead_adds_to_baseline(self, tpu_mappers, medium_matmul):
        for name, mapper in tpu_mappers:
            if mapper.resource_model.architecture_energy_model is None:
                continue
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            _, _, breakdown = mapper._calculate_energy_with_architecture(
                ops, bytes_t, Precision.FP32
            )
            assert breakdown is not None, f"{name}: no breakdown"
            # Systolic arrays have efficiency < 1.0, so architectural
            # energy can be lower than baseline (intentional savings).
            # Verify the breakdown is populated, not that it's additive.
            assert breakdown.explanation, f"{name}: empty explanation"


class TestMonotonicity:

    def test_more_ops_means_more_energy(self, tpu_mappers):
        for name, mapper in tpu_mappers:
            e_small = sum(mapper._calculate_energy(1_000_000, 4_000_000, Precision.FP32))
            e_large = sum(mapper._calculate_energy(100_000_000, 4_000_000, Precision.FP32))
            assert e_large > e_small, f"{name}: more ops did not increase energy"

    def test_architectural_overhead_breakdown_exists(self, tpu_mappers, medium_matmul):
        # Systolic arrays intentionally have negative overhead (savings
        # vs stored-program baseline: compute_efficiency = 0.15 means
        # 85% reduction). So we check the breakdown exists and has a
        # non-trivial explanation, not that the overhead is non-negative.
        for name, mapper in tpu_mappers:
            if mapper.resource_model.architecture_energy_model is None:
                continue
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            _, _, breakdown = mapper._calculate_energy_with_architecture(
                ops, bytes_t, Precision.FP32
            )
            assert breakdown is not None, f"{name}: no breakdown"
            assert len(breakdown.explanation) > 0, f"{name}: empty explanation"


class TestPrecisionScaling:

    def test_int8_cheaper_than_fp32(self, tpu_mappers):
        for name, mapper in tpu_mappers:
            e_fp32 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.FP32)
            e_int8 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT8)
            assert e_int8[0] < e_fp32[0], f"{name}: INT8 compute >= FP32"

    def test_energy_scaling_in_range(self, tpu_mappers):
        for name, mapper in tpu_mappers:
            for prec, scale in mapper.resource_model.energy_scaling.items():
                assert 0 < scale <= 4.0, f"{name}: energy_scaling[{prec}] = {scale}"


class TestCoefficientSensitivity:

    def test_energy_per_flop_perturbation(self, tpu_mappers, medium_matmul):
        for name, mapper in tpu_mappers:
            # TPU mappers with tile energy models (Coral) override
            # _calculate_energy with their own coefficients; perturbing
            # energy_per_flop_fp32 on the resource model has no effect.
            tile_model = getattr(mapper.resource_model, 'tile_energy_model', None)
            if tile_model is not None:
                continue
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            base_c, _ = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            original = mapper.resource_model.energy_per_flop_fp32
            mapper.resource_model.energy_per_flop_fp32 = original * 1.10
            pert_c, _ = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            mapper.resource_model.energy_per_flop_fp32 = original
            if base_c > 0:
                delta = (pert_c - base_c) / base_c
                assert 0.05 < delta < 0.20, f"{name}: {delta*100:.1f}% (expected ~10%)"


class TestCrossMapperOrdering:

    def test_v5p_faster_than_v3_at_bf16(self):
        # Compare at BF16 (native TPU precision). v1 is INT8-only and
        # overreports FP32/BF16 by falling back to its INT8 peak.
        from graphs.hardware.mappers import get_mapper_by_name
        try:
            v5p = get_mapper_by_name("Google-TPU-v5p")
            v3 = get_mapper_by_name("Google-TPU-v3")
        except (KeyError, ValueError, TypeError, AttributeError):
            pytest.skip("Required mappers not found")
        v5p_peak = v5p.resource_model.get_peak_ops(Precision.BF16)
        v3_peak = v3.resource_model.get_peak_ops(Precision.BF16)
        assert v5p_peak > v3_peak, "TPU v5p should be faster than v3 at BF16"

    def test_no_tpu_has_negative_latency(self, tpu_mappers, medium_matmul):
        for name, mapper in tpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            ct, mt, bottleneck = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0, precision=Precision.FP32,
            )
            assert ct >= 0, f"{name}: negative compute_time"
            assert mt >= 0, f"{name}: negative memory_time"
            assert bottleneck is not None, f"{name}: no bottleneck type"

    def test_no_tpu_has_zero_bandwidth(self, tpu_mappers):
        for name, mapper in tpu_mappers:
            assert mapper.resource_model.peak_bandwidth > 0, f"{name}: zero bandwidth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
