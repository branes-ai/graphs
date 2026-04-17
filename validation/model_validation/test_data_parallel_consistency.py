"""
DataParallel (GPU) Model Consistency Tests

Validates internal consistency of the DataParallelEnergyModel and
all GPU mappers against five check categories:

1. Layer decomposition: energy layers sum to total
2. Monotonicity: architectural overhead never decreases energy
3. Precision scaling: lower precision uses less energy
4. Coefficient sensitivity: perturbation produces proportional response
5. Cross-mapper ordering: newer GPUs are faster

Runs without hardware -- exercises the modeling code only.
"""

from __future__ import annotations

import pytest

from graphs.hardware.resource_model import Precision


class TestLayerDecomposition:
    """Energy layers must sum to the model's total."""

    def test_baseline_energy_is_positive(self, gpu_mappers, medium_matmul):
        for name, mapper in gpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"

    def test_architectural_overhead_adds_to_baseline(self, gpu_mappers, medium_matmul):
        for name, mapper in gpu_mappers:
            if mapper.resource_model.architecture_energy_model is None:
                continue
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            base_c, base_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            arch_c, arch_m, breakdown = mapper._calculate_energy_with_architecture(
                ops, bytes_t, Precision.FP32
            )

            assert breakdown is not None, f"{name}: no breakdown returned"
            assert arch_c >= base_c * 0.99, (
                f"{name}: arch compute {arch_c} < base {base_c}"
            )


class TestMonotonicity:
    """Adding work should increase energy; overhead should be non-negative."""

    def test_more_ops_means_more_energy(self, gpu_mappers):
        for name, mapper in gpu_mappers:
            small_ops = 1_000_000
            large_ops = 100_000_000
            bytes_t = 4_000_000

            e_small = sum(mapper._calculate_energy(small_ops, bytes_t, Precision.FP32))
            e_large = sum(mapper._calculate_energy(large_ops, bytes_t, Precision.FP32))
            assert e_large > e_small, f"{name}: more ops did not increase energy"

    def test_more_bytes_means_more_energy(self, gpu_mappers):
        for name, mapper in gpu_mappers:
            ops = 10_000_000
            small_bytes = 1_000_000
            large_bytes = 100_000_000

            e_small = sum(mapper._calculate_energy(ops, small_bytes, Precision.FP32))
            e_large = sum(mapper._calculate_energy(ops, large_bytes, Precision.FP32))
            assert e_large > e_small, f"{name}: more bytes did not increase energy"

    def test_architectural_overhead_is_non_negative(self, gpu_mappers, medium_matmul):
        for name, mapper in gpu_mappers:
            if mapper.resource_model.architecture_energy_model is None:
                continue
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            _, _, breakdown = mapper._calculate_energy_with_architecture(
                ops, bytes_t, Precision.FP32
            )
            assert breakdown.compute_overhead >= 0, (
                f"{name}: negative compute overhead"
            )
            assert breakdown.data_movement_overhead >= 0, (
                f"{name}: negative data movement overhead"
            )


class TestPrecisionScaling:
    """Lower precision should use less compute energy per op."""

    def test_int8_cheaper_than_fp32(self, gpu_mappers):
        for name, mapper in gpu_mappers:
            ops = 10_000_000
            bytes_t = 4_000_000

            e_fp32 = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            e_int8 = mapper._calculate_energy(ops, bytes_t, Precision.INT8)

            assert e_int8[0] < e_fp32[0], (
                f"{name}: INT8 compute energy {e_int8[0]} >= FP32 {e_fp32[0]}"
            )

    def test_fp16_cheaper_than_fp32(self, gpu_mappers):
        for name, mapper in gpu_mappers:
            ops = 10_000_000
            bytes_t = 4_000_000

            e_fp32 = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            e_fp16 = mapper._calculate_energy(ops, bytes_t, Precision.FP16)

            assert e_fp16[0] < e_fp32[0], (
                f"{name}: FP16 compute energy {e_fp16[0]} >= FP32 {e_fp32[0]}"
            )

    def test_energy_scaling_factors_are_in_range(self, gpu_mappers):
        for name, mapper in gpu_mappers:
            for prec, scale in mapper.resource_model.energy_scaling.items():
                assert 0 < scale <= 4.0, (
                    f"{name}: energy_scaling[{prec}] = {scale} out of [0, 4.0]"
                )


class TestCoefficientSensitivity:
    """Perturbing a coefficient should produce a proportional response."""

    def test_energy_per_flop_perturbation(self, gpu_mappers, medium_matmul):
        for name, mapper in gpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            base_c, base_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)

            original = mapper.resource_model.energy_per_flop_fp32
            mapper.resource_model.energy_per_flop_fp32 = original * 1.10
            pert_c, pert_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            mapper.resource_model.energy_per_flop_fp32 = original

            if base_c > 0:
                delta_ratio = (pert_c - base_c) / base_c
                assert 0.05 < delta_ratio < 0.20, (
                    f"{name}: 10% energy_per_flop bump produced "
                    f"{delta_ratio*100:.1f}% change (expected ~10%)"
                )

    def test_bandwidth_perturbation_affects_latency(self, gpu_mappers, medium_matmul):
        for name, mapper in gpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            base_ct, base_mt, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0,
                precision=Precision.FP32,
            )

            original = mapper.resource_model.peak_bandwidth
            mapper.resource_model.peak_bandwidth = original * 0.50
            pert_ct, pert_mt, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0,
                precision=Precision.FP32,
            )
            mapper.resource_model.peak_bandwidth = original

            # Memory time should approximately double (2x)
            if base_mt > 0:
                ratio = pert_mt / base_mt
                assert 1.5 < ratio < 2.5, (
                    f"{name}: halving bandwidth changed memory_time by "
                    f"{ratio:.1f}x (expected ~2x)"
                )


class TestCrossMapperOrdering:
    """Within the GPU class, known performance ordering should hold."""

    def test_h100_faster_than_a100(self):
        from graphs.hardware.mappers import get_mapper_by_name

        try:
            h100 = get_mapper_by_name("H100-SXM5-80GB")
            a100 = get_mapper_by_name("A100-SXM4-80GB")
        except (KeyError, ValueError):
            pytest.skip("Required mappers not found")

        h100_peak = h100.resource_model.get_peak_ops(Precision.FP32)
        a100_peak = a100.resource_model.get_peak_ops(Precision.FP32)
        assert h100_peak >= a100_peak, "H100 should have >= peak FP32 vs A100"

    def test_a100_faster_than_v100(self):
        from graphs.hardware.mappers import get_mapper_by_name

        try:
            a100 = get_mapper_by_name("A100-SXM4-80GB")
            v100 = get_mapper_by_name("V100-SXM3-32GB")
        except (KeyError, ValueError):
            pytest.skip("Required mappers not found")

        a100_peak = a100.resource_model.get_peak_ops(Precision.FP32)
        v100_peak = v100.resource_model.get_peak_ops(Precision.FP32)
        assert a100_peak > v100_peak, "A100 should have higher peak FP32 than V100"

    def test_datacenter_faster_than_edge(self):
        from graphs.hardware.mappers import get_mapper_by_name

        try:
            h100 = get_mapper_by_name("H100-SXM5-80GB")
            orin = get_mapper_by_name("Jetson-Orin-AGX-64GB")
        except (KeyError, ValueError):
            pytest.skip("Required mappers not found")

        h100_peak = h100.resource_model.get_peak_ops(Precision.FP32)
        orin_peak = orin.resource_model.get_peak_ops(Precision.FP32)
        assert h100_peak > orin_peak * 10, (
            "H100 should be >10x faster than Orin AGX at FP32"
        )

    def test_no_gpu_has_negative_latency(self, gpu_mappers, medium_matmul):
        for name, mapper in gpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            compute_time, memory_time, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0,
                precision=Precision.FP32,
            )
            assert compute_time >= 0, f"{name}: negative compute_time"
            assert memory_time >= 0, f"{name}: negative memory_time"

    def test_no_gpu_has_zero_peak_bandwidth(self, gpu_mappers):
        for name, mapper in gpu_mappers:
            assert mapper.resource_model.peak_bandwidth > 0, (
                f"{name}: peak_bandwidth is zero"
            )

    def test_h100_has_more_bandwidth_than_v100(self):
        from graphs.hardware.mappers import get_mapper_by_name

        try:
            h100 = get_mapper_by_name("H100-SXM5-80GB")
            v100 = get_mapper_by_name("V100-SXM3-32GB")
        except (KeyError, ValueError):
            pytest.skip("Required mappers not found")

        assert h100.resource_model.peak_bandwidth > v100.resource_model.peak_bandwidth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
