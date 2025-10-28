#!/usr/bin/env python3
"""
Hardware Power Modeling Tests

Tests for idle power modeling (50% TDP idle power consumption) across
all hardware mappers: GPU, TPU, CPU, DSP, DPU, KPU.

This validates that nanoscale leakage current modeling is correctly
implemented in all mappers.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from graphs.hardware.mappers.gpu import (
    create_h100_mapper,
    create_jetson_thor_mapper,
)
from graphs.hardware.mappers.cpu import (
    create_intel_xeon_platinum_8490h_mapper,
    create_amd_epyc_9654_mapper,
)
from graphs.hardware.mappers.accelerators.tpu import (
    create_tpu_v4_mapper,
    create_coral_edge_tpu_mapper,
)
from graphs.hardware.mappers.dsp import (
    create_qrb5165_mapper,
    create_ti_tda4vm_mapper,
)
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
    create_kpu_t256_mapper,
)


class TestIdlePowerConstant:
    """Test that all mappers have the correct IDLE_POWER_FRACTION constant."""

    def test_gpu_mapper_has_idle_power_constant(self):
        """GPU mappers should have IDLE_POWER_FRACTION = 0.5"""
        mapper = create_h100_mapper()
        assert hasattr(mapper, 'IDLE_POWER_FRACTION'), \
            "GPUMapper missing IDLE_POWER_FRACTION constant"
        assert mapper.IDLE_POWER_FRACTION == 0.5, \
            f"Expected 0.5, got {mapper.IDLE_POWER_FRACTION}"

    def test_tpu_mapper_has_idle_power_constant(self):
        """TPU mappers should have IDLE_POWER_FRACTION = 0.5"""
        mapper = create_tpu_v4_mapper()
        assert hasattr(mapper, 'IDLE_POWER_FRACTION'), \
            "TPUMapper missing IDLE_POWER_FRACTION constant"
        assert mapper.IDLE_POWER_FRACTION == 0.5, \
            f"Expected 0.5, got {mapper.IDLE_POWER_FRACTION}"

    def test_cpu_mapper_has_idle_power_constant(self):
        """CPU mappers should have IDLE_POWER_FRACTION = 0.5"""
        mapper = create_intel_xeon_platinum_8490h_mapper()
        assert hasattr(mapper, 'IDLE_POWER_FRACTION'), \
            "CPUMapper missing IDLE_POWER_FRACTION constant"
        assert mapper.IDLE_POWER_FRACTION == 0.5, \
            f"Expected 0.5, got {mapper.IDLE_POWER_FRACTION}"

    def test_dsp_mapper_has_idle_power_constant(self):
        """DSP mappers should have IDLE_POWER_FRACTION = 0.5"""
        mapper = create_qrb5165_mapper()
        assert hasattr(mapper, 'IDLE_POWER_FRACTION'), \
            "DSPMapper missing IDLE_POWER_FRACTION constant"
        assert mapper.IDLE_POWER_FRACTION == 0.5, \
            f"Expected 0.5, got {mapper.IDLE_POWER_FRACTION}"

    def test_dpu_mapper_has_idle_power_constant(self):
        """DPU mappers should have IDLE_POWER_FRACTION = 0.5"""
        mapper = create_dpu_vitis_ai_mapper()
        assert hasattr(mapper, 'IDLE_POWER_FRACTION'), \
            "DPUMapper missing IDLE_POWER_FRACTION constant"
        assert mapper.IDLE_POWER_FRACTION == 0.5, \
            f"Expected 0.5, got {mapper.IDLE_POWER_FRACTION}"

    def test_kpu_mapper_has_idle_power_constant(self):
        """KPU mappers should have IDLE_POWER_FRACTION = 0.5"""
        mapper = create_kpu_t64_mapper()
        assert hasattr(mapper, 'IDLE_POWER_FRACTION'), \
            "KPUMapper missing IDLE_POWER_FRACTION constant"
        assert mapper.IDLE_POWER_FRACTION == 0.5, \
            f"Expected 0.5, got {mapper.IDLE_POWER_FRACTION}"


class TestIdlePowerMethod:
    """Test that all mappers have the compute_energy_with_idle_power() method."""

    def test_gpu_mapper_has_idle_power_method(self):
        """GPU mappers should have compute_energy_with_idle_power() method"""
        mapper = create_h100_mapper()
        assert hasattr(mapper, 'compute_energy_with_idle_power'), \
            "GPUMapper missing compute_energy_with_idle_power() method"
        assert callable(mapper.compute_energy_with_idle_power), \
            "compute_energy_with_idle_power is not callable"

    def test_tpu_mapper_has_idle_power_method(self):
        """TPU mappers should have compute_energy_with_idle_power() method"""
        mapper = create_tpu_v4_mapper()
        assert hasattr(mapper, 'compute_energy_with_idle_power'), \
            "TPUMapper missing compute_energy_with_idle_power() method"
        assert callable(mapper.compute_energy_with_idle_power), \
            "compute_energy_with_idle_power is not callable"

    def test_cpu_mapper_has_idle_power_method(self):
        """CPU mappers should have compute_energy_with_idle_power() method"""
        mapper = create_intel_xeon_platinum_8490h_mapper()
        assert hasattr(mapper, 'compute_energy_with_idle_power'), \
            "CPUMapper missing compute_energy_with_idle_power() method"
        assert callable(mapper.compute_energy_with_idle_power), \
            "compute_energy_with_idle_power is not callable"

    def test_dsp_mapper_has_idle_power_method(self):
        """DSP mappers should have compute_energy_with_idle_power() method"""
        mapper = create_qrb5165_mapper()
        assert hasattr(mapper, 'compute_energy_with_idle_power'), \
            "DSPMapper missing compute_energy_with_idle_power() method"
        assert callable(mapper.compute_energy_with_idle_power), \
            "compute_energy_with_idle_power is not callable"

    def test_dpu_mapper_has_idle_power_method(self):
        """DPU mappers should have compute_energy_with_idle_power() method"""
        mapper = create_dpu_vitis_ai_mapper()
        assert hasattr(mapper, 'compute_energy_with_idle_power'), \
            "DPUMapper missing compute_energy_with_idle_power() method"
        assert callable(mapper.compute_energy_with_idle_power), \
            "compute_energy_with_idle_power is not callable"

    def test_kpu_mapper_has_idle_power_method(self):
        """KPU mappers should have compute_energy_with_idle_power() method"""
        mapper = create_kpu_t64_mapper()
        assert hasattr(mapper, 'compute_energy_with_idle_power'), \
            "KPUMapper missing compute_energy_with_idle_power() method"
        assert callable(mapper.compute_energy_with_idle_power), \
            "compute_energy_with_idle_power is not callable"


class TestIdlePowerCalculation:
    """Test that idle power calculations produce correct results."""

    def test_idle_power_equals_half_tdp_datacenter(self):
        """At very low dynamic power, total power should approach 50% TDP (datacenter)"""
        # Test with TPU v4 (350W TDP)
        mapper = create_tpu_v4_mapper()
        tdp = 350.0  # TPU v4 TDP
        latency = 1.0  # 1 second
        dynamic_energy = 0.001  # 1 mJ (negligible)

        total_energy, avg_power = mapper.compute_energy_with_idle_power(latency, dynamic_energy)

        # Expected: idle_energy = (350W × 0.5) × 1s = 175 J
        # Total: 175 + 0.001 = 175.001 J
        # Avg power: 175.001W
        expected_idle_energy = tdp * 0.5 * latency
        expected_total = expected_idle_energy + dynamic_energy

        assert abs(total_energy - expected_total) < 0.01, \
            f"Expected {expected_total}J, got {total_energy}J"
        assert abs(avg_power - tdp * 0.5) < 0.01, \
            f"Expected ~{tdp*0.5}W, got {avg_power}W"

    def test_idle_power_equals_half_tdp_edge(self):
        """At very low dynamic power, total power should approach 50% TDP (edge)"""
        # Test with KPU-T64 (6W TDP)
        mapper = create_kpu_t64_mapper()
        tdp = 6.0  # KPU-T64 default TDP
        latency = 0.01  # 10ms
        dynamic_energy = 0.0001  # 0.1 mJ (negligible)

        total_energy, avg_power = mapper.compute_energy_with_idle_power(latency, dynamic_energy)

        # Expected: idle_energy = (6W × 0.5) × 0.01s = 0.03 J = 30 mJ
        # Total: 30 + 0.1 = 30.1 mJ
        # Avg power: 3.01W
        expected_idle_energy = tdp * 0.5 * latency
        expected_total = expected_idle_energy + dynamic_energy

        assert abs(total_energy - expected_total) < 0.0001, \
            f"Expected {expected_total}J, got {total_energy}J"
        assert abs(avg_power - 3.0) < 0.1, \
            f"Expected ~3W, got {avg_power}W"

    def test_idle_power_dominates_low_utilization(self):
        """For low-utilization workloads, idle power should dominate"""
        # Test with Intel Xeon (350W TDP)
        mapper = create_intel_xeon_platinum_8490h_mapper()
        latency = 0.010  # 10ms
        dynamic_energy = 0.1  # 100 mJ

        total_energy, avg_power = mapper.compute_energy_with_idle_power(latency, dynamic_energy)

        # Idle component: (350W × 0.5) × 0.01s = 1.75 J = 1750 mJ
        idle_energy = 350.0 * 0.5 * latency

        # Idle should be much larger than dynamic
        idle_fraction = idle_energy / total_energy
        assert idle_fraction > 0.9, \
            f"Idle power should dominate (>90%), got {idle_fraction*100:.1f}%"

    def test_dynamic_power_dominates_high_utilization(self):
        """For high-utilization, long-running workloads, dynamic power should dominate"""
        # Test with QRB5165 DSP (7W TDP)
        mapper = create_qrb5165_mapper()
        latency = 1.0  # 1 second (long execution)
        dynamic_energy = 5.0  # 5 J (high dynamic)

        total_energy, avg_power = mapper.compute_energy_with_idle_power(latency, dynamic_energy)

        # Idle component: (7W × 0.5) × 1s = 3.5 J
        idle_energy = 7.0 * 0.5 * latency

        # Dynamic should be larger than idle
        dynamic_fraction = dynamic_energy / total_energy
        assert dynamic_fraction > 0.5, \
            f"Dynamic power should dominate (>50%), got {dynamic_fraction*100:.1f}%"

    def test_power_scales_with_latency(self):
        """Total energy should scale linearly with latency for fixed dynamic energy"""
        mapper = create_coral_edge_tpu_mapper()
        dynamic_energy = 0.01  # 10 mJ (constant)

        # Test at different latencies
        latency1 = 0.01  # 10ms
        latency2 = 0.02  # 20ms (2× longer)

        energy1, _ = mapper.compute_energy_with_idle_power(latency1, dynamic_energy)
        energy2, _ = mapper.compute_energy_with_idle_power(latency2, dynamic_energy)

        # Energy should roughly double (idle component doubles)
        # Allow some tolerance due to constant dynamic energy
        ratio = energy2 / energy1
        assert 1.5 <= ratio < 2.5, \
            f"Expected ~2× energy for 2× latency, got {ratio:.2f}×"


class TestThermalProfileIntegration:
    """Test that thermal profiles are correctly used in idle power calculations."""

    def test_kpu_uses_correct_thermal_profile(self):
        """KPU should use the specified thermal profile's TDP"""
        # Create KPU-T256 with different thermal profiles
        mapper_15w = create_kpu_t256_mapper()  # Defaults to some profile

        # Mock a calculation
        latency = 0.01
        dynamic_energy = 0.01

        total_energy, avg_power = mapper_15w.compute_energy_with_idle_power(
            latency, dynamic_energy
        )

        # Should use a valid TDP (not crash)
        assert total_energy > dynamic_energy, \
            "Total energy should exceed dynamic energy (includes idle)"
        assert avg_power > 0, \
            "Average power should be positive"

    def test_fallback_to_default_thermal_profile(self):
        """If thermal_profile not found, should fallback to 'default' profile"""
        mapper = create_dpu_vitis_ai_mapper()

        # DPU has 'default' thermal profile at 20W
        latency = 0.01
        dynamic_energy = 0.0

        total_energy, avg_power = mapper.compute_energy_with_idle_power(
            latency, dynamic_energy
        )

        # With zero dynamic energy, should get exactly idle power
        # Expected: (20W × 0.5) × 0.01s = 0.1 J
        expected = 20.0 * 0.5 * latency
        assert abs(total_energy - expected) < 0.001, \
            f"Expected {expected}J with 20W TDP, got {total_energy}J"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
