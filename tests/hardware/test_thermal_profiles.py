#!/usr/bin/env python3
"""
Hardware Thermal Profile Tests

Tests for thermal operating points across all hardware models.
Validates that all models have appropriate TDP values and thermal profiles.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from graphs.hardware.mappers.gpu import (
    create_h100_mapper,
    create_jetson_orin_agx_mapper,
    create_jetson_orin_nano_mapper,
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
    create_kpu_t768_mapper,
)


class TestThermalOperatingPoints:
    """Test that all hardware models have thermal_operating_points."""

    def test_gpu_models_have_thermal_points(self):
        """All GPU models should have thermal_operating_points"""
        mappers = [
            ("H100", create_h100_mapper()),
            ("Jetson Orin AGX", create_jetson_orin_agx_mapper()),
            ("Jetson Orin Nano", create_jetson_orin_nano_mapper()),
            ("Jetson Thor", create_jetson_thor_mapper()),
        ]

        for name, mapper in mappers:
            assert mapper.resource_model.thermal_operating_points is not None, \
                f"{name} missing thermal_operating_points"
            assert len(mapper.resource_model.thermal_operating_points) > 0, \
                f"{name} has empty thermal_operating_points"

    def test_tpu_models_have_thermal_points(self):
        """All TPU models should have thermal_operating_points"""
        mappers = [
            ("TPU v4", create_tpu_v4_mapper()),
            ("Coral Edge TPU", create_coral_edge_tpu_mapper()),
        ]

        for name, mapper in mappers:
            assert mapper.resource_model.thermal_operating_points is not None, \
                f"{name} missing thermal_operating_points"
            assert len(mapper.resource_model.thermal_operating_points) > 0, \
                f"{name} has empty thermal_operating_points"

    def test_cpu_models_have_thermal_points(self):
        """All CPU models should have thermal_operating_points"""
        mappers = [
            ("Intel Xeon 8490H", create_intel_xeon_platinum_8490h_mapper()),
            ("AMD EPYC 9654", create_amd_epyc_9654_mapper()),
        ]

        for name, mapper in mappers:
            assert mapper.resource_model.thermal_operating_points is not None, \
                f"{name} missing thermal_operating_points"
            assert len(mapper.resource_model.thermal_operating_points) > 0, \
                f"{name} has empty thermal_operating_points"

    def test_dsp_models_have_thermal_points(self):
        """All DSP models should have thermal_operating_points"""
        mappers = [
            ("QRB5165", create_qrb5165_mapper()),
            ("TI TDA4VM", create_ti_tda4vm_mapper()),
        ]

        for name, mapper in mappers:
            assert mapper.resource_model.thermal_operating_points is not None, \
                f"{name} missing thermal_operating_points"
            assert len(mapper.resource_model.thermal_operating_points) > 0, \
                f"{name} has empty thermal_operating_points"

    def test_dpu_models_have_thermal_points(self):
        """DPU model should have thermal_operating_points"""
        mapper = create_dpu_vitis_ai_mapper()
        assert mapper.resource_model.thermal_operating_points is not None, \
            "DPU missing thermal_operating_points"
        assert len(mapper.resource_model.thermal_operating_points) > 0, \
            "DPU has empty thermal_operating_points"

    def test_kpu_models_have_thermal_points(self):
        """All KPU models should have thermal_operating_points"""
        mappers = [
            ("KPU-T64", create_kpu_t64_mapper()),
            ("KPU-T256", create_kpu_t256_mapper()),
            ("KPU-T768", create_kpu_t768_mapper()),
        ]

        for name, mapper in mappers:
            assert mapper.resource_model.thermal_operating_points is not None, \
                f"{name} missing thermal_operating_points"
            assert len(mapper.resource_model.thermal_operating_points) > 0, \
                f"{name} has empty thermal_operating_points"


class TestTDPValues:
    """Test that TDP values are reasonable for each hardware category."""

    def test_datacenter_gpu_tdp_range(self):
        """Datacenter GPUs should have TDP in reasonable range (300-700W)"""
        mapper = create_h100_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points

        # Get first (or default) thermal profile
        profile = thermal_points.get("default") or next(iter(thermal_points.values()))
        tdp = profile.tdp_watts

        assert 300 <= tdp <= 700, \
            f"Datacenter GPU TDP should be 300-700W, got {tdp}W"

    def test_edge_gpu_tdp_range(self):
        """Edge GPUs should have TDP in reasonable range (15-100W)"""
        mappers = [
            create_jetson_orin_nano_mapper(),
            create_jetson_orin_agx_mapper(),
            create_jetson_thor_mapper(),
        ]

        for mapper in mappers:
            thermal_points = mapper.resource_model.thermal_operating_points
            profile = thermal_points.get("default") or thermal_points.get("15W") or next(iter(thermal_points.values()))
            tdp = profile.tdp_watts

            assert 5 <= tdp <= 150, \
                f"Edge GPU TDP should be 5-150W, got {tdp}W for {mapper.resource_model.name}"

    def test_datacenter_cpu_tdp_range(self):
        """Datacenter CPUs should have TDP in reasonable range (200-500W)"""
        mappers = [
            create_intel_xeon_platinum_8490h_mapper(),
            create_amd_epyc_9654_mapper(),
        ]

        for mapper in mappers:
            thermal_points = mapper.resource_model.thermal_operating_points
            profile = thermal_points.get("default") or next(iter(thermal_points.values()))
            tdp = profile.tdp_watts

            assert 200 <= tdp <= 600, \
                f"Datacenter CPU TDP should be 200-600W, got {tdp}W for {mapper.resource_model.name}"

    def test_dsp_tdp_range(self):
        """DSPs should have TDP in reasonable range (3-30W)"""
        mappers = [
            create_qrb5165_mapper(),
            create_ti_tda4vm_mapper(),
        ]

        for mapper in mappers:
            thermal_points = mapper.resource_model.thermal_operating_points
            profile = thermal_points.get("default") or next(iter(thermal_points.values()))
            tdp = profile.tdp_watts

            assert 3 <= tdp <= 30, \
                f"DSP TDP should be 3-30W, got {tdp}W for {mapper.resource_model.name}"

    def test_dpu_tdp_range(self):
        """DPU should have TDP in reasonable range (15-50W for edge FPGA)"""
        mapper = create_dpu_vitis_ai_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points
        profile = thermal_points.get("default")
        tdp = profile.tdp_watts

        assert 15 <= tdp <= 50, \
            f"Edge DPU TDP should be 15-50W, got {tdp}W"

    def test_kpu_tdp_range(self):
        """KPUs should have TDP in reasonable range (3-100W)"""
        mappers = [
            ("T64", create_kpu_t64_mapper(), 3, 10),
            ("T256", create_kpu_t256_mapper(), 15, 50),
            ("T768", create_kpu_t768_mapper(), 30, 100),
        ]

        for name, mapper, min_tdp, max_tdp in mappers:
            thermal_points = mapper.resource_model.thermal_operating_points
            profile = thermal_points.get("default") or next(iter(thermal_points.values()))
            tdp = profile.tdp_watts

            assert min_tdp <= tdp <= max_tdp, \
                f"KPU-{name} TDP should be {min_tdp}-{max_tdp}W, got {tdp}W"


class TestMultiPowerProfiles:
    """Test models with multiple power profiles."""

    def test_jetson_orin_agx_has_multiple_profiles(self):
        """Jetson Orin AGX should have 3 power profiles (15W, 30W, 60W)"""
        mapper = create_jetson_orin_agx_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points

        assert len(thermal_points) >= 3, \
            f"Expected at least 3 profiles, got {len(thermal_points)}"

        # Should have low, medium, and high power modes
        tdps = [p.tdp_watts for p in thermal_points.values()]
        assert min(tdps) <= 20, "Should have low-power profile (<= 20W)"
        assert max(tdps) >= 50, "Should have high-power profile (>= 50W)"

    def test_jetson_thor_has_multiple_profiles(self):
        """Jetson Thor should have multiple power profiles"""
        mapper = create_jetson_thor_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points

        assert len(thermal_points) >= 2, \
            f"Expected at least 2 profiles, got {len(thermal_points)}"

    def test_kpu_t64_has_multiple_profiles(self):
        """KPU-T64 should have multiple power profiles (3W, 6W, 10W)"""
        mapper = create_kpu_t64_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points

        assert len(thermal_points) >= 2, \
            f"Expected at least 2 profiles, got {len(thermal_points)}"

        # Should have profiles in the 3-10W range
        tdps = [p.tdp_watts for p in thermal_points.values()]
        assert min(tdps) >= 2, "Minimum TDP should be >= 2W"
        assert max(tdps) <= 15, "Maximum TDP should be <= 15W"

    def test_kpu_t256_has_multiple_profiles(self):
        """KPU-T256 should have multiple power profiles (15W, 30W, 50W)"""
        mapper = create_kpu_t256_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points

        assert len(thermal_points) >= 2, \
            f"Expected at least 2 profiles, got {len(thermal_points)}"

        # Should have profiles in the 15-50W range
        tdps = [p.tdp_watts for p in thermal_points.values()]
        assert min(tdps) >= 10, "Minimum TDP should be >= 10W"
        assert max(tdps) <= 60, "Maximum TDP should be <= 60W"


class TestThermalProfileStructure:
    """Test the structure and required fields of thermal operating points."""

    def test_thermal_point_has_required_fields(self):
        """Thermal operating points should have all required fields"""
        mapper = create_tpu_v4_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points
        profile = thermal_points.get("default")

        assert profile is not None, "Should have 'default' thermal profile"
        assert hasattr(profile, 'name'), "Missing 'name' field"
        assert hasattr(profile, 'tdp_watts'), "Missing 'tdp_watts' field"
        assert hasattr(profile, 'cooling_solution'), "Missing 'cooling_solution' field"
        assert hasattr(profile, 'performance_specs'), "Missing 'performance_specs' field"

    def test_tdp_is_positive(self):
        """TDP should always be a positive number"""
        mappers = [
            create_h100_mapper(),
            create_tpu_v4_mapper(),
            create_intel_xeon_platinum_8490h_mapper(),
            create_qrb5165_mapper(),
            create_dpu_vitis_ai_mapper(),
            create_kpu_t64_mapper(),
        ]

        for mapper in mappers:
            thermal_points = mapper.resource_model.thermal_operating_points
            for profile_name, profile in thermal_points.items():
                assert profile.tdp_watts > 0, \
                    f"{mapper.resource_model.name} profile '{profile_name}' has non-positive TDP: {profile.tdp_watts}"

    def test_cooling_solution_is_specified(self):
        """All thermal profiles should specify a cooling solution"""
        mapper = create_h100_mapper()
        thermal_points = mapper.resource_model.thermal_operating_points

        for profile_name, profile in thermal_points.items():
            assert profile.cooling_solution is not None, \
                f"Profile '{profile_name}' missing cooling_solution"
            assert len(profile.cooling_solution) > 0, \
                f"Profile '{profile_name}' has empty cooling_solution"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
