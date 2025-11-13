#!/usr/bin/env python3
"""
Cross-Architecture Physics-Based Energy Model Validation

Validates that the physics-based energy model is consistent across different
hardware architectures (GPU, CPU, TPU, KPU). This ensures energy comparisons
across architectures are trustworthy.

Tests:
1. Base ALU energy depends only on process node (not architecture)
2. Tensor Cores/Matrix tiles show expected 15% efficiency gain
3. Process node scaling follows physics (16nm → 8nm → 7nm → 5nm)
4. Multi-fabric architectures have consistent energy per fabric type
"""
import sys
import pytest

sys.path.insert(0, 'src')

from graphs.hardware.models.datacenter.h100_sxm5_80gb import h100_sxm5_80gb_resource_model
from graphs.hardware.models.edge.jetson_orin_agx_64gb import jetson_orin_agx_64gb_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.edge.tpu_edge_pro import tpu_edge_pro_resource_model
from graphs.hardware.resource_model import Precision


# Expected base ALU energy by process node (pJ)
PROCESS_NODE_ENERGY = {
    4: 1.30,   # 4nm (TSMC 4N)
    5: 1.50,   # 5nm
    7: 1.80,   # 7nm
    8: 1.90,   # 8nm (Samsung)
    16: 2.70,  # 16nm
}

# Models to test: (name, factory_fn, expected_process_nm)
TEST_MODELS = [
    ('H100 SXM5', h100_sxm5_80gb_resource_model, 4),  # Updated: 4nm not 5nm
    ('Jetson Orin AGX GPU', jetson_orin_agx_64gb_resource_model, 8),
    ('KPU-T256', kpu_t256_resource_model, 16),
    ('TPU Edge Pro', tpu_edge_pro_resource_model, 7),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm", TEST_MODELS)
def test_base_alu_energy_by_process_node(name, model_fn, expected_process_nm):
    """
    Test that base ALU energy matches expected value for the process node.

    Physics principle: Energy per operation depends primarily on transistor
    switching energy, which scales with process node geometry.
    """
    model = model_fn()
    expected_energy_pj = PROCESS_NODE_ENERGY[expected_process_nm]
    actual_energy_pj = model.energy_per_flop_fp32 * 1e12

    # Allow 0.1 pJ tolerance for rounding and circuit variations
    assert abs(actual_energy_pj - expected_energy_pj) < 0.1, \
        f"{name} @ {expected_process_nm}nm: expected {expected_energy_pj:.2f} pJ, got {actual_energy_pj:.2f} pJ"


@pytest.mark.parametrize("name,model_fn,expected_process_nm", TEST_MODELS)
def test_compute_fabrics_exist(name, model_fn, expected_process_nm):
    """Test that models have multi-fabric architecture (not legacy single energy)."""
    model = model_fn()

    assert model.compute_fabrics is not None, \
        f"{name}: Missing compute_fabrics (still using legacy energy model)"

    assert len(model.compute_fabrics) >= 1, \
        f"{name}: Expected at least 1 compute fabric"


@pytest.mark.parametrize("name,model_fn,expected_process_nm", TEST_MODELS)
def test_fabric_process_nodes_consistent(name, model_fn, expected_process_nm):
    """Test that all fabrics in a model have the same process node."""
    model = model_fn()

    if not model.compute_fabrics:
        pytest.skip(f"{name}: No compute_fabrics to validate")

    for fabric in model.compute_fabrics:
        assert fabric.process_node_nm == expected_process_nm, \
            f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"


def test_tensor_core_efficiency_h100():
    """
    Test H100 Tensor Core efficiency vs CUDA cores.

    Tensor Cores use tensor_core circuit type (0.85× multiplier) vs
    CUDA cores using standard_cell (1.0× multiplier).
    Expected: 15% efficiency gain.
    """
    model = h100_sxm5_80gb_resource_model()

    assert len(model.compute_fabrics) >= 2, "H100 should have CUDA + Tensor fabrics"

    cuda_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "cuda_core"), None)
    tensor_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "tensor_core"), None)

    assert cuda_fabric is not None, "H100 missing cuda_core fabric"
    assert tensor_fabric is not None, "H100 missing tensor_core fabric"

    cuda_energy_pj = cuda_fabric.energy_per_flop_fp32 * 1e12
    tensor_energy_pj = tensor_fabric.energy_per_flop_fp32 * 1e12

    efficiency_gain_percent = (cuda_energy_pj - tensor_energy_pj) / cuda_energy_pj * 100

    # Expect ~15% efficiency gain (0.85× multiplier)
    assert efficiency_gain_percent == pytest.approx(15.0, abs=1.0), \
        f"H100 Tensor Core efficiency: expected ~15% gain, got {efficiency_gain_percent:.1f}%"


def test_tensor_core_efficiency_jetson():
    """Test Jetson Orin AGX Tensor Core efficiency vs CUDA cores."""
    model = jetson_orin_agx_64gb_resource_model()

    assert len(model.compute_fabrics) >= 2, "Jetson should have CUDA + Tensor fabrics"

    cuda_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "cuda_core"), None)
    tensor_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "tensor_core"), None)

    assert cuda_fabric is not None, "Jetson missing cuda_core fabric"
    assert tensor_fabric is not None, "Jetson missing tensor_core fabric"

    cuda_energy_pj = cuda_fabric.energy_per_flop_fp32 * 1e12
    tensor_energy_pj = tensor_fabric.energy_per_flop_fp32 * 1e12

    efficiency_gain_percent = (cuda_energy_pj - tensor_energy_pj) / cuda_energy_pj * 100

    assert efficiency_gain_percent == pytest.approx(15.0, abs=1.0), \
        f"Jetson Tensor Core efficiency: expected ~15% gain, got {efficiency_gain_percent:.1f}%"


def test_matrix_tile_efficiency_kpu():
    """
    Test KPU-T256 Matrix tile efficiency vs standard tiles.

    Matrix tiles use tensor_core circuit type (15% more efficient).
    """
    model = kpu_t256_resource_model()

    assert len(model.compute_fabrics) >= 3, "KPU should have INT8 + BF16 + Matrix fabrics"

    # Compare BF16 tiles (standard) vs Matrix tiles (tensor_core)
    bf16_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "bf16_tile"), None)
    matrix_fabric = next((f for f in model.compute_fabrics if f.fabric_type == "matrix_tile"), None)

    assert bf16_fabric is not None, "KPU missing bf16_tile fabric"
    assert matrix_fabric is not None, "KPU missing matrix_tile fabric"

    bf16_energy_pj = bf16_fabric.energy_per_flop_fp32 * 1e12
    matrix_energy_pj = matrix_fabric.energy_per_flop_fp32 * 1e12

    efficiency_gain_percent = (bf16_energy_pj - matrix_energy_pj) / bf16_energy_pj * 100

    assert efficiency_gain_percent == pytest.approx(15.0, abs=1.0), \
        f"KPU Matrix tile efficiency: expected ~15% gain, got {efficiency_gain_percent:.1f}%"


def test_tpu_systolic_array_uses_standard_cell():
    """
    Test TPU Edge Pro uses standard_cell circuit type.

    Systolic arrays don't have the same efficiency gain as Tensor Cores
    because they already have amortized control overhead.
    """
    model = tpu_edge_pro_resource_model()

    assert len(model.compute_fabrics) == 1, "TPU should have single systolic array fabric"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "systolic_array", "TPU should use systolic_array fabric"
    assert fabric.circuit_type == "standard_cell", \
        "Systolic arrays should use standard_cell (no additional tensor_core efficiency)"


def test_process_node_scaling():
    """
    Test that energy scales correctly across process nodes.

    Physics: Energy should decrease as process node shrinks.
    16nm → 8nm → 7nm → 5nm → 4nm should show monotonic decrease.
    """
    # Get actual energies from models
    h100 = h100_sxm5_80gb_resource_model()
    tpu = tpu_edge_pro_resource_model()
    jetson = jetson_orin_agx_64gb_resource_model()
    kpu = kpu_t256_resource_model()

    process_energy_pairs = [
        (16, kpu.energy_per_flop_fp32 * 1e12),
        (8, jetson.energy_per_flop_fp32 * 1e12),
        (7, tpu.energy_per_flop_fp32 * 1e12),
        (4, h100.energy_per_flop_fp32 * 1e12),
    ]

    # Sort by process node (descending)
    process_energy_pairs.sort(reverse=True, key=lambda x: x[0])

    # Check monotonic decrease in energy as process shrinks
    for i in range(len(process_energy_pairs) - 1):
        larger_nm, larger_energy = process_energy_pairs[i]
        smaller_nm, smaller_energy = process_energy_pairs[i + 1]

        assert larger_energy > smaller_energy, \
            f"Energy should decrease as process shrinks: {larger_nm}nm ({larger_energy:.2f} pJ) should be > {smaller_nm}nm ({smaller_energy:.2f} pJ)"


def test_circuit_type_multipliers():
    """
    Test that circuit type multipliers are applied correctly.

    For same process node, different circuit types should show expected energy ratios:
    - standard_cell: 1.0×
    - tensor_core: 0.85× (15% more efficient)
    """
    h100 = h100_sxm5_80gb_resource_model()

    cuda_fabric = next((f for f in h100.compute_fabrics if f.fabric_type == "cuda_core"), None)
    tensor_fabric = next((f for f in h100.compute_fabrics if f.fabric_type == "tensor_core"), None)

    assert cuda_fabric.circuit_type == "standard_cell", "CUDA cores should use standard_cell"
    assert tensor_fabric.circuit_type == "tensor_core", "Tensor Cores should use tensor_core"

    # Both should be same process node
    assert cuda_fabric.process_node_nm == tensor_fabric.process_node_nm

    # Ratio should be 0.85 (tensor_core multiplier)
    energy_ratio = tensor_fabric.energy_per_flop_fp32 / cuda_fabric.energy_per_flop_fp32

    assert energy_ratio == pytest.approx(0.85, abs=0.01), \
        f"Tensor Core energy should be 0.85× CUDA core energy, got {energy_ratio:.3f}×"


def test_cross_architecture_consistency():
    """
    Test that different architectures at same process node have same base energy.

    All 8nm chips should have ~1.9 pJ base energy regardless of architecture.
    """
    jetson = jetson_orin_agx_64gb_resource_model()

    # Jetson is 8nm Samsung
    expected_8nm_energy = PROCESS_NODE_ENERGY[8]
    actual_energy_pj = jetson.energy_per_flop_fp32 * 1e12

    assert abs(actual_energy_pj - expected_8nm_energy) < 0.1, \
        f"8nm chips should have {expected_8nm_energy:.2f} pJ energy, got {actual_energy_pj:.2f} pJ"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
