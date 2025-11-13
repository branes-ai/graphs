#!/usr/bin/env python3
"""
Phase 3 Automotive AI Energy Model Validation

Validates that all 7 Phase 3 automotive AI models have correct:
1. Process nodes matching their actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configurations

This test ensures energy comparisons across automotive models are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.automotive.jetson_thor_128gb import jetson_thor_128gb_resource_model
from graphs.hardware.models.automotive.qualcomm_sa8775p import qualcomm_sa8775p_resource_model
from graphs.hardware.models.automotive.qualcomm_snapdragon_ride import qualcomm_snapdragon_ride_resource_model
from graphs.hardware.models.automotive.ti_tda4al import ti_tda4al_resource_model
from graphs.hardware.models.automotive.ti_tda4vh import ti_tda4vh_resource_model
from graphs.hardware.models.automotive.ti_tda4vl import ti_tda4vl_resource_model
from graphs.hardware.models.automotive.ti_tda4vm import ti_tda4vm_resource_model

# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, fabric_count, architecture_type)
AUTOMOTIVE_MODELS = [
    ('Jetson Thor 128GB', jetson_thor_128gb_resource_model, 4, 1.30, 2, 'GPU'),
    ('Qualcomm SA8775P', qualcomm_sa8775p_resource_model, 5, 1.35, 2, 'DSP'),
    ('Qualcomm Snapdragon Ride', qualcomm_snapdragon_ride_resource_model, 4, 1.17, 2, 'DSP'),
    ('TI TDA4AL', ti_tda4al_resource_model, 28, 3.60, 2, 'DSP'),
    ('TI TDA4VH', ti_tda4vh_resource_model, 28, 3.60, 2, 'DSP'),
    ('TI TDA4VL', ti_tda4vl_resource_model, 28, 3.60, 2, 'DSP'),
    ('TI TDA4VM', ti_tda4vm_resource_model, 28, 3.60, 2, 'DSP'),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_count,arch_type", AUTOMOTIVE_MODELS)
def test_automotive_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_count, arch_type):
    """Test automotive models have correct process nodes and energy values."""
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


def test_phase3_completeness():
    """Test that all 7 Phase 3 automotive models are accounted for."""
    assert len(AUTOMOTIVE_MODELS) == 7, "Expected 7 Phase 3 automotive AI models"


def test_jetson_thor_multi_fabric():
    """Test that Jetson Thor has CUDA + Tensor Core fabrics."""
    model = jetson_thor_128gb_resource_model()
    assert len(model.compute_fabrics) == 2, "Expected 2 fabrics (CUDA + Tensor)"

    fabric_types = [f.fabric_type for f in model.compute_fabrics]
    assert "cuda_core" in fabric_types, "Missing cuda_core fabric"
    assert "tensor_core" in fabric_types, "Missing tensor_core fabric"


def test_qualcomm_automotive_multi_fabric():
    """Test that Qualcomm automotive models have HVX + tensor fabrics."""
    for name, model_fn, _, _, _, arch_type in AUTOMOTIVE_MODELS:
        if 'Qualcomm' not in name:
            continue

        model = model_fn()
        assert len(model.compute_fabrics) == 2, \
            f"{name}: Expected 2 fabrics (HVX + tensor)"

        fabric_types = [f.fabric_type for f in model.compute_fabrics]
        assert "hvx_vector" in fabric_types, f"{name}: Missing hvx_vector fabric"
        # SA8775P has HMX, Snapdragon Ride has AI tensor accelerator
        has_tensor = any(t in fabric_types for t in ["hmx_tensor", "ai_tensor_accelerator"])
        assert has_tensor, f"{name}: Missing tensor fabric (HMX or AI tensor)"


def test_ti_automotive_multi_fabric():
    """Test that TI TDA4 models have C7x + MMA fabrics."""
    for name, model_fn, _, _, _, arch_type in AUTOMOTIVE_MODELS:
        if 'TI' not in name:
            continue

        model = model_fn()
        assert len(model.compute_fabrics) == 2, \
            f"{name}: Expected 2 fabrics (C7x + MMA)"

        fabric_types = [f.fabric_type for f in model.compute_fabrics]
        assert "c7x_dsp" in fabric_types, f"{name}: Missing c7x_dsp fabric"
        # TDA4VM has MMAv1, others have MMAv2
        has_mma = any(t in fabric_types for t in ["mma_v1", "mma_v2"])
        assert has_mma, f"{name}: Missing MMA fabric (v1 or v2)"


def test_process_node_scaling():
    """Test that energy scales correctly with process node."""
    models = [(name, model_fn()) for name, model_fn, _, _, _, _ in AUTOMOTIVE_MODELS]

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
            if process_nm == 4:
                assert 1.1 < energy_pj < 1.4, f"{name}: 4nm should be ~1.2-1.3 pJ"
            elif process_nm == 5:
                assert 1.2 < energy_pj < 1.6, f"{name}: 5nm should be ~1.3-1.5 pJ"
            elif process_nm == 28:
                assert 3.3 < energy_pj < 3.8, f"{name}: 28nm should be ~3.4-3.6 pJ"


def test_ti_tda4_family_consistency():
    """Test that TI TDA4 family has consistent energy (same process node)."""
    ti_models = [(name, model_fn()) for name, model_fn, _, _, _, _ in AUTOMOTIVE_MODELS if 'TI' in name]

    energies = [model.energy_per_flop_fp32 * 1e12 for _, model in ti_models]

    # All TI TDA4 models should have same energy (3.6 pJ @ 28nm)
    for (name, _), energy in zip(ti_models, energies):
        assert abs(energy - 3.6) < 0.01, \
            f"{name}: Expected 3.6 pJ, got {energy:.2f} pJ"


def test_qualcomm_snapdragon_ride_tops():
    """Test that Qualcomm Snapdragon Ride achieves 700 TOPS INT8."""
    model = qualcomm_snapdragon_ride_resource_model()

    # Check fabrics
    assert len(model.compute_fabrics) == 2, "Expected HVX + AI tensor fabrics"

    hvx_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "hvx_vector"), None)
    ai_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "ai_tensor_accelerator"), None)

    assert hvx_fabric is not None, "Missing HVX fabric"
    assert ai_fabric is not None, "Missing AI tensor fabric"

    # Calculate total TOPS INT8
    from graphs.hardware.resource_model import Precision
    hvx_tops = hvx_fabric.num_units * hvx_fabric.ops_per_unit_per_clock[Precision.INT8] * hvx_fabric.core_frequency_hz / 1e12
    ai_tops = ai_fabric.num_units * ai_fabric.ops_per_unit_per_clock[Precision.INT8] * ai_fabric.core_frequency_hz / 1e12
    total_tops = hvx_tops + ai_tops

    assert 690 < total_tops < 710, \
        f"Expected ~700 TOPS, got {total_tops:.1f} TOPS (HVX: {hvx_tops:.1f}, AI: {ai_tops:.1f})"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
