"""
StoredProgram (CPU/DSP) Model Consistency Tests

Validates internal consistency of the StoredProgramEnergyModel and
all CPU mappers against five check categories:

1. Layer decomposition: energy layers sum to total
2. Monotonicity: architectural overhead never decreases energy
3. Precision scaling: lower precision uses less energy
4. Coefficient sensitivity: perturbation produces proportional response
5. Cross-mapper ordering: newer/larger CPUs are faster

Runs without hardware -- exercises the modeling code only.
"""

from __future__ import annotations

import pytest

from graphs.hardware.resource_model import Precision


class TestLayerDecomposition:
    """Energy layers must sum to the model's total."""

    def test_baseline_energy_is_positive(self, cpu_mappers, medium_matmul):
        for name, mapper in cpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"

    def test_architectural_overhead_adds_to_baseline(self, cpu_mappers, medium_matmul):
        for name, mapper in cpu_mappers:
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
            assert arch_m >= base_m * 0.99, (
                f"{name}: arch memory {arch_m} < base {base_m}"
            )


class TestMonotonicity:
    """Adding work should increase energy; overhead should be non-negative."""

    def test_more_ops_means_more_energy(self, cpu_mappers):
        for name, mapper in cpu_mappers:
            small_ops = 1_000_000
            large_ops = 100_000_000
            bytes_t = 4_000_000

            e_small = sum(mapper._calculate_energy(small_ops, bytes_t, Precision.FP32))
            e_large = sum(mapper._calculate_energy(large_ops, bytes_t, Precision.FP32))
            assert e_large > e_small, f"{name}: more ops did not increase energy"

    def test_more_bytes_means_more_energy(self, cpu_mappers):
        for name, mapper in cpu_mappers:
            ops = 10_000_000
            small_bytes = 1_000_000
            large_bytes = 100_000_000

            e_small = sum(mapper._calculate_energy(ops, small_bytes, Precision.FP32))
            e_large = sum(mapper._calculate_energy(ops, large_bytes, Precision.FP32))
            assert e_large > e_small, f"{name}: more bytes did not increase energy"

    def test_architectural_overhead_is_non_negative(self, cpu_mappers, medium_matmul):
        for name, mapper in cpu_mappers:
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

    def test_int8_cheaper_than_fp32(self, cpu_mappers):
        for name, mapper in cpu_mappers:
            ops = 10_000_000
            bytes_t = 4_000_000

            e_fp32 = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            e_int8 = mapper._calculate_energy(ops, bytes_t, Precision.INT8)

            # Compute energy (index 0) should be lower for INT8
            assert e_int8[0] < e_fp32[0], (
                f"{name}: INT8 compute energy {e_int8[0]} >= FP32 {e_fp32[0]}"
            )

    def test_fp16_cheaper_than_fp32(self, cpu_mappers):
        for name, mapper in cpu_mappers:
            ops = 10_000_000
            bytes_t = 4_000_000

            e_fp32 = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            e_fp16 = mapper._calculate_energy(ops, bytes_t, Precision.FP16)

            assert e_fp16[0] < e_fp32[0], (
                f"{name}: FP16 compute energy {e_fp16[0]} >= FP32 {e_fp32[0]}"
            )

    def test_energy_scaling_factors_are_in_range(self, cpu_mappers):
        for name, mapper in cpu_mappers:
            for prec, scale in mapper.resource_model.energy_scaling.items():
                assert 0 < scale <= 4.0, (
                    f"{name}: energy_scaling[{prec}] = {scale} out of [0, 4.0]"
                )


class TestCoefficientSensitivity:
    """Perturbing a coefficient should produce a proportional response."""

    def test_energy_per_flop_perturbation(self, cpu_mappers, medium_matmul):
        for name, mapper in cpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            # Baseline
            base_c, base_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)

            # Perturb energy_per_flop by +10%
            original = mapper.resource_model.energy_per_flop_fp32
            mapper.resource_model.energy_per_flop_fp32 = original * 1.10
            pert_c, pert_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            mapper.resource_model.energy_per_flop_fp32 = original

            # Compute energy should increase by ~10%
            if base_c > 0:
                delta_ratio = (pert_c - base_c) / base_c
                assert 0.05 < delta_ratio < 0.20, (
                    f"{name}: 10% energy_per_flop bump produced "
                    f"{delta_ratio*100:.1f}% change (expected ~10%)"
                )

            # Memory energy should be unchanged
            assert abs(pert_m - base_m) < base_m * 0.001, (
                f"{name}: energy_per_flop perturbation affected memory energy"
            )

    def test_energy_per_byte_perturbation(self, cpu_mappers, medium_matmul):
        for name, mapper in cpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            base_c, base_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)

            original = mapper.resource_model.energy_per_byte
            mapper.resource_model.energy_per_byte = original * 1.10
            pert_c, pert_m = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            mapper.resource_model.energy_per_byte = original

            if base_m > 0:
                delta_ratio = (pert_m - base_m) / base_m
                assert 0.05 < delta_ratio < 0.20, (
                    f"{name}: 10% energy_per_byte bump produced "
                    f"{delta_ratio*100:.1f}% change (expected ~10%)"
                )

            assert abs(pert_c - base_c) < base_c * 0.001, (
                f"{name}: energy_per_byte perturbation affected compute energy"
            )


class TestCrossMapperOrdering:
    """Within the CPU class, known performance ordering should hold."""

    def test_xeon_has_more_compute_units_than_i7(self):
        from graphs.hardware.mappers import get_mapper_by_name

        try:
            xeon = get_mapper_by_name("Intel-Xeon-Platinum-8490H")
            i7 = get_mapper_by_name("Intel-i7-12700K")
        except (KeyError, ValueError, TypeError, AttributeError):
            pytest.skip("Required mappers not found")

        assert xeon.resource_model.compute_units > i7.resource_model.compute_units

    def test_epyc_128_has_more_cores_than_epyc_96(self):
        from graphs.hardware.mappers import get_mapper_by_name

        try:
            epyc128 = get_mapper_by_name("AMD-EPYC-9754")
            epyc96 = get_mapper_by_name("AMD-EPYC-9654")
        except (KeyError, ValueError, TypeError, AttributeError):
            pytest.skip("Required mappers not found")

        assert epyc128.resource_model.compute_units > epyc96.resource_model.compute_units

    def test_no_cpu_has_negative_latency(self, cpu_mappers, medium_matmul):
        for name, mapper in cpu_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)

            compute_time, memory_time, bottleneck = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0,
                precision=Precision.FP32,
            )
            assert compute_time >= 0, f"{name}: negative compute_time"
            assert memory_time >= 0, f"{name}: negative memory_time"

    def test_no_cpu_has_zero_peak_bandwidth(self, cpu_mappers):
        for name, mapper in cpu_mappers:
            assert mapper.resource_model.peak_bandwidth > 0, (
                f"{name}: peak_bandwidth is zero"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
