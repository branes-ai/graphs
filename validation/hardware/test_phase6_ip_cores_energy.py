#!/usr/bin/env python3
"""
Phase 6 IP Cores Energy Model Validation

Validates that all 3 Phase 6 IP core models have correct:
1. Process nodes matching their actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configurations

This test ensures energy comparisons across IP core models are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.ip_cores.synopsys_arc_ev7x import synopsys_arc_ev7x_resource_model
from graphs.hardware.models.ip_cores.ceva_neupro_npm11 import ceva_neupro_npm11_resource_model
from graphs.hardware.models.ip_cores.cadence_vision_q8 import cadence_vision_q8_resource_model

# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, fabric_count, architecture_type)
IP_CORE_MODELS = [
    ('Synopsys ARC EV7x', synopsys_arc_ev7x_resource_model, 16, 2.295, 2, 'Vision-IP'),
    ('CEVA NeuPro NPM11', ceva_neupro_npm11_resource_model, 16, 2.295, 2, 'Neural-IP'),
    ('Cadence Vision Q8', cadence_vision_q8_resource_model, 16, 2.43, 1, 'Vision-DSP'),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_count,arch_type", IP_CORE_MODELS)
def test_ip_core_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_count, arch_type):
    """Test IP core models have correct process nodes and energy values."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) == fabric_count, \
        f"{name}: Expected {fabric_count} fabric(s), got {len(model.compute_fabrics)}"

    # Check all fabrics have correct process node
    for fabric in model.compute_fabrics:
        assert fabric.process_node_nm == expected_process_nm, \
            f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"

    # Check energy value (using first fabric as baseline, or dominant fabric)
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


def test_phase6_completeness():
    """Test that all 3 Phase 6 IP core models are accounted for."""
    assert len(IP_CORE_MODELS) == 3, "Expected 3 Phase 6 IP core models"


def test_synopsys_arc_ev7x_heterogeneous():
    """Test that Synopsys ARC EV7x has VPU + DNN accelerator fabrics."""
    model = synopsys_arc_ev7x_resource_model()
    assert len(model.compute_fabrics) == 2, "Expected 2 fabrics (VPU + DNN accelerator)"

    fabric_types = [f.fabric_type for f in model.compute_fabrics]
    assert "ev7x_vpu" in fabric_types, "Missing ev7x_vpu fabric"
    assert "ev7x_dnn_accelerator" in fabric_types, "Missing ev7x_dnn_accelerator fabric"

    # VPU should use simd_packed
    vpu_fabric = next(f for f in model.compute_fabrics if f.fabric_type == "ev7x_vpu")
    assert vpu_fabric.circuit_type == "simd_packed", "VPU should use simd_packed"

    # DNN accelerator should use tensor_core
    dnn_fabric = next(f for f in model.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")
    assert dnn_fabric.circuit_type == "tensor_core", "DNN accelerator should use tensor_core"


def test_ceva_neupro_npm11_heterogeneous():
    """Test that CEVA NeuPro NPM11 has Tensor + Vector fabrics."""
    model = ceva_neupro_npm11_resource_model()
    assert len(model.compute_fabrics) == 2, "Expected 2 fabrics (Tensor + Vector)"

    fabric_types = [f.fabric_type for f in model.compute_fabrics]
    assert "neupro_tensor" in fabric_types, "Missing neupro_tensor fabric"
    assert "neupro_vector" in fabric_types, "Missing neupro_vector fabric"

    # Tensor should use tensor_core
    tensor_fabric = next(f for f in model.compute_fabrics if f.fabric_type == "neupro_tensor")
    assert tensor_fabric.circuit_type == "tensor_core", "Tensor fabric should use tensor_core"

    # Vector should use simd_packed
    vector_fabric = next(f for f in model.compute_fabrics if f.fabric_type == "neupro_vector")
    assert vector_fabric.circuit_type == "simd_packed", "Vector fabric should use simd_packed"


def test_cadence_vision_q8_simd():
    """Test that Cadence Vision Q8 has SIMD fabric only."""
    model = cadence_vision_q8_resource_model()
    assert len(model.compute_fabrics) == 1, "Expected 1 fabric (SIMD only)"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "vision_q8_simd", "Expected vision_q8_simd fabric"
    assert fabric.circuit_type == "simd_packed", "Vision Q8 should use simd_packed"


def test_process_node_16nm_consistency():
    """Test that all IP cores use 16nm process node."""
    models = [(name, model_fn()) for name, model_fn, _, _, _, _ in IP_CORE_MODELS]

    for name, model in models:
        if model.compute_fabrics:
            for fabric in model.compute_fabrics:
                assert fabric.process_node_nm == 16, \
                    f"{name}: Expected 16nm, got {fabric.process_node_nm}nm"


def test_synopsys_arc_ev7x_performance():
    """Test that Synopsys ARC EV7x achieves 35 TOPS INT8."""
    model = synopsys_arc_ev7x_resource_model()
    dnn_fabric = next(f for f in model.compute_fabrics if f.fabric_type == "ev7x_dnn_accelerator")

    from graphs.hardware.resource_model import Precision
    num_units = dnn_fabric.num_units
    ops_per_unit = dnn_fabric.ops_per_unit_per_clock[Precision.INT8]
    clock_hz = dnn_fabric.core_frequency_hz

    peak_tops = (num_units * ops_per_unit * clock_hz) / 1e12
    expected_tops = 35.0

    assert abs(peak_tops - expected_tops) / expected_tops < 0.01, \
        f"Expected ~35 TOPS INT8, got {peak_tops:.2f} TOPS"


def test_ceva_neupro_npm11_performance():
    """Test that CEVA NeuPro NPM11 achieves 20 TOPS INT8."""
    model = ceva_neupro_npm11_resource_model()
    tensor_fabric = next(f for f in model.compute_fabrics if f.fabric_type == "neupro_tensor")

    from graphs.hardware.resource_model import Precision
    num_units = tensor_fabric.num_units
    ops_per_unit = tensor_fabric.ops_per_unit_per_clock[Precision.INT8]
    clock_hz = tensor_fabric.core_frequency_hz

    peak_tops = (num_units * ops_per_unit * clock_hz) / 1e12
    expected_tops = 20.0

    assert abs(peak_tops - expected_tops) / expected_tops < 0.01, \
        f"Expected ~20 TOPS INT8, got {peak_tops:.2f} TOPS"


def test_cadence_vision_q8_performance():
    """Test that Cadence Vision Q8 achieves 3.8 TOPS INT8 and 129 GFLOPS FP32."""
    model = cadence_vision_q8_resource_model()
    simd_fabric = model.compute_fabrics[0]

    from graphs.hardware.resource_model import Precision
    num_units = simd_fabric.num_units
    clock_hz = simd_fabric.core_frequency_hz

    # INT8 performance
    int8_ops = simd_fabric.ops_per_unit_per_clock[Precision.INT8]
    peak_int8_tops = (num_units * int8_ops * clock_hz) / 1e12
    expected_int8_tops = 3.8

    assert abs(peak_int8_tops - expected_int8_tops) / expected_int8_tops < 0.01, \
        f"Expected ~3.8 TOPS INT8, got {peak_int8_tops:.2f} TOPS"

    # FP32 performance
    fp32_ops = simd_fabric.ops_per_unit_per_clock[Precision.FP32]
    peak_fp32_gflops = (num_units * fp32_ops * clock_hz) / 1e9
    expected_fp32_gflops = 129.0

    assert abs(peak_fp32_gflops - expected_fp32_gflops) / expected_fp32_gflops < 0.01, \
        f"Expected ~129 GFLOPS FP32, got {peak_fp32_gflops:.2f} GFLOPS"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
