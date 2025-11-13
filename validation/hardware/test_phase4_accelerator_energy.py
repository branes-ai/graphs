#!/usr/bin/env python3
"""
Phase 4 Accelerator Energy Model Validation

Validates that all 5 Phase 4 accelerator models have correct:
1. Process nodes matching their actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configurations

This test ensures energy comparisons across accelerator models are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.accelerators.kpu_t768 import kpu_t768_resource_model
from graphs.hardware.models.accelerators.xilinx_vitis_ai_dpu import xilinx_vitis_ai_dpu_resource_model
from graphs.hardware.models.accelerators.stanford_plasticine_cgra import stanford_plasticine_cgra_resource_model

# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, fabric_count, architecture_type)
ACCELERATOR_MODELS = [
    ('KPU-T64', kpu_t64_resource_model, 16, 2.70, 3, 'KPU'),
    ('KPU-T256', kpu_t256_resource_model, 16, 2.70, 3, 'KPU'),
    ('KPU-T768', kpu_t768_resource_model, 12, 2.50, 3, 'KPU'),
    ('Xilinx Vitis AI DPU', xilinx_vitis_ai_dpu_resource_model, 16, 2.70, 1, 'DPU'),
    ('Stanford Plasticine CGRA', stanford_plasticine_cgra_resource_model, 28, 4.00, 1, 'CGRA'),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_count,arch_type", ACCELERATOR_MODELS)
def test_accelerator_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_count, arch_type):
    """Test accelerator models have correct process nodes and energy values."""
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


def test_phase4_completeness():
    """Test that all 5 Phase 4 accelerator models are accounted for."""
    assert len(ACCELERATOR_MODELS) == 5, "Expected 5 Phase 4 accelerator models"


def test_kpu_multi_fabric():
    """Test that KPU models have INT8 + BF16 + Matrix tile fabrics."""
    for name, model_fn, _, _, _, arch_type in ACCELERATOR_MODELS:
        if arch_type != 'KPU':
            continue

        model = model_fn()
        assert len(model.compute_fabrics) == 3, \
            f"{name}: Expected 3 fabrics (INT8 + BF16 + Matrix tiles)"

        fabric_types = [f.fabric_type for f in model.compute_fabrics]
        assert "kpu_int8_tile" in fabric_types, f"{name}: Missing kpu_int8_tile fabric"
        assert "kpu_bf16_tile" in fabric_types, f"{name}: Missing kpu_bf16_tile fabric"
        assert "kpu_matrix_tile" in fabric_types, f"{name}: Missing kpu_matrix_tile fabric"


def test_xilinx_dpu_fabric():
    """Test that Xilinx Vitis AI DPU has AIE-ML fabric."""
    model = xilinx_vitis_ai_dpu_resource_model()
    assert len(model.compute_fabrics) == 1, "Expected 1 fabric (AIE-ML)"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "aie_ml_tile", "Expected aie_ml_tile fabric"
    assert fabric.circuit_type == "standard_cell", "AIE-ML should use standard_cell"


def test_stanford_plasticine_fabric():
    """Test that Stanford Plasticine CGRA has PCU spatial dataflow fabric."""
    model = stanford_plasticine_cgra_resource_model()
    assert len(model.compute_fabrics) == 1, "Expected 1 fabric (PCU spatial dataflow)"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "pcu_spatial_dataflow", "Expected pcu_spatial_dataflow fabric"
    assert fabric.circuit_type == "standard_cell", "PCU should use standard_cell"


def test_process_node_scaling():
    """Test that energy scales correctly with process node."""
    models = [(name, model_fn()) for name, model_fn, _, _, _, _ in ACCELERATOR_MODELS]

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
            if process_nm == 12:
                assert 2.4 < energy_pj < 2.6, f"{name}: 12nm should be ~2.5 pJ"
            elif process_nm == 16:
                assert 2.6 < energy_pj < 2.8, f"{name}: 16nm should be ~2.7 pJ"
            elif process_nm == 28:
                assert 3.8 < energy_pj < 4.2, f"{name}: 28nm should be ~4.0 pJ"


def test_kpu_t64_vs_t256_vs_t768():
    """Test that KPU family has correct tile counts and scaling."""
    t64 = kpu_t64_resource_model()
    t256 = kpu_t256_resource_model()
    t768 = kpu_t768_resource_model()

    # Check compute units (tile counts)
    assert t64.compute_units == 64, "KPU-T64 should have 64 tiles"
    assert t256.compute_units == 256, "KPU-T256 should have 256 tiles"
    assert t768.compute_units == 768, "KPU-T768 should have 768 tiles"

    # Check fabric types are consistent
    t64_types = {f.fabric_type for f in t64.compute_fabrics}
    t256_types = {f.fabric_type for f in t256.compute_fabrics}
    t768_types = {f.fabric_type for f in t768.compute_fabrics}

    assert t64_types == t256_types == t768_types, \
        "All KPU models should have the same fabric types"


def test_kpu_matrix_tile_tensor_core():
    """Test that KPU matrix tiles use tensor_core circuit type."""
    for name, model_fn, _, _, _, arch_type in ACCELERATOR_MODELS:
        if arch_type != 'KPU':
            continue

        model = model_fn()
        matrix_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "kpu_matrix_tile"), None)
        assert matrix_fabric is not None, f"{name}: Missing kpu_matrix_tile fabric"
        assert matrix_fabric.circuit_type == "tensor_core", \
            f"{name}: Matrix tiles should use tensor_core (15% more efficient)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
