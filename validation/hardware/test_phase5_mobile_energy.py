#!/usr/bin/env python3
"""
Phase 5 Mobile Energy Model Validation

Validates that Phase 5 mobile GPU model has correct:
1. Process node matching actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configuration

This test ensures energy comparisons for mobile GPUs are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.mobile.arm_mali_g78_mp20 import arm_mali_g78_mp20_resource_model

# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, fabric_count, architecture_type)
MOBILE_MODELS = [
    ('ARM-Mali-G78-MP20', arm_mali_g78_mp20_resource_model, 7, 1.80, 1, 'Mobile-GPU'),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_count,arch_type", MOBILE_MODELS)
def test_mobile_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_count, arch_type):
    """Test mobile GPU model has correct process node and energy value."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) == fabric_count, \
        f"{name}: Expected {fabric_count} fabric(s), got {len(model.compute_fabrics)}"

    # Check fabric has correct process node
    fabric = model.compute_fabrics[0]
    assert fabric.process_node_nm == expected_process_nm, \
        f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"

    # Check energy value
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


def test_phase5_completeness():
    """Test that Phase 5 mobile model is accounted for."""
    assert len(MOBILE_MODELS) == 1, "Expected 1 Phase 5 mobile model"


def test_mali_g78_mp20_shader_fabric():
    """Test that ARM Mali-G78 MP20 has unified shader core fabric."""
    model = arm_mali_g78_mp20_resource_model()
    assert len(model.compute_fabrics) == 1, "Expected 1 fabric (unified shader cores)"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "mali_shader_core", "Expected mali_shader_core fabric"
    assert fabric.circuit_type == "standard_cell", "Mali shader cores should use standard_cell"
    assert fabric.num_units == 20, "MP20 should have 20 shader cores"


def test_mali_g78_mp20_precision_support():
    """Test that ARM Mali-G78 MP20 supports FP32/FP16/INT8 with correct ops per clock."""
    model = arm_mali_g78_mp20_resource_model()
    fabric = model.compute_fabrics[0]

    # Check ops per clock for each precision
    from graphs.hardware.resource_model import Precision
    assert Precision.FP32 in fabric.ops_per_unit_per_clock, "Missing FP32 support"
    assert Precision.FP16 in fabric.ops_per_unit_per_clock, "Missing FP16 support"
    assert Precision.INT8 in fabric.ops_per_unit_per_clock, "Missing INT8 support"

    # Check FP16 is 2× FP32 (graphics GPU characteristic)
    fp32_ops = fabric.ops_per_unit_per_clock[Precision.FP32]
    fp16_ops = fabric.ops_per_unit_per_clock[Precision.FP16]
    assert fp16_ops == fp32_ops * 2, f"FP16 should be 2× FP32, got {fp16_ops} vs {fp32_ops}"


def test_mali_g78_mp20_peak_performance():
    """Test that ARM Mali-G78 MP20 achieves expected peak performance."""
    model = arm_mali_g78_mp20_resource_model()
    fabric = model.compute_fabrics[0]

    from graphs.hardware.resource_model import Precision

    # Calculate peak FP32: 20 cores × 114 ops/cycle × 848 MHz = 1.93 TFLOPS
    num_cores = fabric.num_units
    ops_per_core = fabric.ops_per_unit_per_clock[Precision.FP32]
    clock_hz = fabric.core_frequency_hz

    peak_fp32_ops = num_cores * ops_per_core * clock_hz
    expected_fp32_tflops = 1.94e12  # 1.94 TFLOPS from model docs

    assert abs(peak_fp32_ops - expected_fp32_tflops) / expected_fp32_tflops < 0.01, \
        f"FP32 peak should be ~1.94 TFLOPS, got {peak_fp32_ops / 1e12:.2f} TFLOPS"

    # Calculate peak FP16: 20 cores × 228 ops/cycle × 848 MHz = 3.87 TFLOPS
    peak_fp16_ops = num_cores * fabric.ops_per_unit_per_clock[Precision.FP16] * clock_hz
    expected_fp16_tflops = 3.88e12  # 3.88 TFLOPS from model docs

    assert abs(peak_fp16_ops - expected_fp16_tflops) / expected_fp16_tflops < 0.01, \
        f"FP16 peak should be ~3.88 TFLOPS, got {peak_fp16_ops / 1e12:.2f} TFLOPS"


def test_process_node_7nm():
    """Test that 7nm process node has correct energy."""
    model = arm_mali_g78_mp20_resource_model()
    energy_pj = model.energy_per_flop_fp32 * 1e12

    # 7nm standard_cell should be 1.8 pJ
    assert abs(energy_pj - 1.8) < 0.01, f"7nm should be ~1.8 pJ, got {energy_pj:.2f} pJ"


def test_mali_g78_mp20_no_tensor_cores():
    """Test that Mali-G78 MP20 does not use tensor cores (graphics GPU)."""
    model = arm_mali_g78_mp20_resource_model()
    fabric = model.compute_fabrics[0]

    # Graphics GPUs use standard_cell, not tensor_core
    assert fabric.circuit_type == "standard_cell", \
        "Mali-G78 is graphics-focused, should use standard_cell"

    # Precision profiles should not claim tensor core support for INT8
    from graphs.hardware.resource_model import Precision
    int8_profile = model.precision_profiles.get(Precision.INT8)
    if int8_profile:
        assert not int8_profile.tensor_core_supported, \
            "Mali-G78 has no tensor cores (graphics GPU)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
