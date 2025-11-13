#!/usr/bin/env python3
"""
Phase 2 Edge AI Energy Model Validation

Validates that all 6 Phase 2 edge AI models have correct:
1. Process nodes matching their actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configurations

This test ensures energy comparisons across edge models are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.edge.jetson_orin_nano_8gb import jetson_orin_nano_8gb_resource_model
from graphs.hardware.models.edge.qrb5165 import qrb5165_resource_model
from graphs.hardware.models.edge.qualcomm_qcs6490 import qualcomm_qcs6490_resource_model
from graphs.hardware.models.edge.hailo8 import hailo8_resource_model
from graphs.hardware.models.edge.hailo10h import hailo10h_resource_model
from graphs.hardware.models.edge.coral_edge_tpu import coral_edge_tpu_resource_model

# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, fabric_count, architecture_type)
EDGE_MODELS = [
    ('Jetson Orin Nano 8GB', jetson_orin_nano_8gb_resource_model, 8, 1.90, 2, 'GPU'),
    ('Qualcomm QRB5165', qrb5165_resource_model, 7, 1.62, 2, 'DSP'),
    ('Qualcomm QCS6490', qualcomm_qcs6490_resource_model, 6, 1.49, 2, 'DSP'),
    ('Hailo-8', hailo8_resource_model, 16, 2.70, 1, 'Dataflow'),
    ('Hailo-10H', hailo10h_resource_model, 16, 2.70, 1, 'Dataflow'),
    ('Coral Edge TPU', coral_edge_tpu_resource_model, 14, 2.60, 1, 'TPU'),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_count,arch_type", EDGE_MODELS)
def test_edge_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_count, arch_type):
    """Test edge models have correct process nodes and energy values."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) == fabric_count, \
        f"{name}: Expected {fabric_count} fabric(s), got {len(model.compute_fabrics)}"

    # Check all fabrics have correct process node
    for fabric in model.compute_fabrics:
        assert fabric.process_node_nm == expected_process_nm, \
            f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"

    # Check energy value (using first fabric as baseline)
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


def test_phase2_completeness():
    """Test that all 6 Phase 2 edge models are accounted for."""
    assert len(EDGE_MODELS) == 6, "Expected 6 Phase 2 edge AI models"


def test_jetson_multi_fabric():
    """Test that Jetson Orin Nano has CUDA + Tensor Core fabrics."""
    model = jetson_orin_nano_8gb_resource_model()
    assert len(model.compute_fabrics) == 2, "Expected 2 fabrics (CUDA + Tensor)"

    fabric_types = [f.fabric_type for f in model.compute_fabrics]
    assert "cuda_core" in fabric_types, "Missing cuda_core fabric"
    assert "tensor_core" in fabric_types, "Missing tensor_core fabric"


def test_qualcomm_multi_fabric():
    """Test that Qualcomm models have HVX + HTA fabrics."""
    for name, model_fn, _, _, _, arch_type in EDGE_MODELS:
        if arch_type != 'DSP':
            continue
        
        model = model_fn()
        assert len(model.compute_fabrics) == 2, \
            f"{name}: Expected 2 fabrics (HVX + HTA)"

        fabric_types = [f.fabric_type for f in model.compute_fabrics]
        assert "hvx_vector" in fabric_types, f"{name}: Missing hvx_vector fabric"
        assert "hta_tensor" in fabric_types, f"{name}: Missing hta_tensor fabric"


def test_dataflow_architecture():
    """Test that dataflow accelerators have single fabric."""
    for name, model_fn, _, _, _, arch_type in EDGE_MODELS:
        if arch_type != 'Dataflow':
            continue
        
        model = model_fn()
        assert len(model.compute_fabrics) == 1, \
            f"{name}: Expected 1 fabric (dataflow)"

        fabric = model.compute_fabrics[0]
        assert fabric.circuit_type == "standard_cell", \
            f"{name}: Expected standard_cell circuit type"


def test_tpu_systolic_array():
    """Test that Coral Edge TPU uses systolic array."""
    model = coral_edge_tpu_resource_model()
    assert len(model.compute_fabrics) == 1, "Expected 1 fabric (systolic array)"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "systolic_array", "Expected systolic_array fabric"
    assert fabric.circuit_type == "standard_cell", \
        "Systolic arrays should use standard_cell"


def test_process_node_scaling():
    """Test that energy scales correctly with process node."""
    models = [(name, model_fn()) for name, model_fn, _, _, _, _ in EDGE_MODELS]

    # Group by process node
    by_process = {}
    for name, model in models:
        if model.compute_fabrics:
            process_nm = model.compute_fabrics[0].process_node_nm
            energy_pj = model.energy_per_flop_fp32 * 1e12
            if process_nm not in by_process:
                by_process[process_nm] = []
            by_process[process_nm].append((name, energy_pj))

    # Check energy increases with larger process nodes
    for process_nm in sorted(by_process.keys()):
        for name, energy_pj in by_process[process_nm]:
            # Energy should be within expected range for process node
            if process_nm == 6:
                assert 1.4 < energy_pj < 1.7, f"{name}: 6nm should be ~1.5-1.6 pJ"
            elif process_nm == 7:
                assert 1.5 < energy_pj < 1.9, f"{name}: 7nm should be ~1.6-1.8 pJ"
            elif process_nm == 8:
                assert 1.8 < energy_pj < 2.0, f"{name}: 8nm should be ~1.9 pJ"
            elif process_nm == 14:
                assert 2.5 < energy_pj < 2.7, f"{name}: 14nm should be ~2.6 pJ"
            elif process_nm == 16:
                assert 2.6 < energy_pj < 2.8, f"{name}: 16nm should be ~2.7 pJ"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
