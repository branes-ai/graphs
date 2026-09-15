#!/usr/bin/env python3
"""
Phase 4 Accelerator Energy Model Validation

Validates that the Phase 4 accelerator resource models (KPU T64 / T256 / T768,
Xilinx Vitis AI DPU, Stanford Plasticine CGRA) are consistent with their
source of truth:

1. Every compute fabric's process node matches the node of the catalog
   ComputeProduct it is loaded from (embodied-schemas).
2. KPU fabrics mirror the catalog tile classes: one fabric per class, named
   ``kpu_<tile_type>``, with per-op FP32 energy taken from the ProcessNode
   for that class's standard-cell library (so the HP-logic Matrix tiles cost
   more per op than the balanced-logic INT8 / BF16 tiles).
3. KPU model-level FP32 energy is per FLOP: the ProcessNode per-MAC
   balanced-logic figure halved (graphs#81).
4. Across all five accelerators, energy per FP32 FLOP falls as the process
   node shrinks (7 nm < 16 nm < 28 nm).

Expected values are derived from the catalog and ProcessNode data, not
hard-coded, so they track catalog updates. The previous version pinned
fabric names, a 12 nm T768 and ``tensor_core`` Matrix tiles that the
YAML-loader models no longer have (refreshed in graphs#268 A3). DPU and CGRA
energy derivations are pinned in their own loader parity tests
(tests/hardware/test_dpu_yaml_loader_vitis_ai_parity.py,
tests/hardware/test_cgra_yaml_loader_plasticine_parity.py).

This test ensures energy comparisons across accelerator models are trustworthy.
"""
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from embodied_schemas import load_process_nodes  # noqa: E402
from embodied_schemas.process_node import CircuitClass  # noqa: E402

from graphs.hardware.compute_product_loader import load_compute_products_unified  # noqa: E402
from graphs.hardware.kpu_access import kpu_block_of, kpu_die_of  # noqa: E402
from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model  # noqa: E402
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model  # noqa: E402
from graphs.hardware.models.accelerators.kpu_t768 import kpu_t768_resource_model  # noqa: E402
from graphs.hardware.models.accelerators.stanford_plasticine_cgra import (  # noqa: E402
    stanford_plasticine_cgra_resource_model,
)
from graphs.hardware.models.accelerators.xilinx_vitis_ai_dpu import (  # noqa: E402
    xilinx_vitis_ai_dpu_resource_model,
)

# (name, factory, catalog ComputeProduct id, architecture type)
ACCELERATOR_MODELS = [
    ('KPU-T64', kpu_t64_resource_model, 'kpu_t64_32x32_lp5x4_16nm_tsmc_ffp', 'KPU'),
    ('KPU-T256', kpu_t256_resource_model, 'kpu_t256_32x32_lp5x16_16nm_tsmc_ffp', 'KPU'),
    ('KPU-T768', kpu_t768_resource_model, 'kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc', 'KPU'),
    ('Xilinx Vitis AI DPU', xilinx_vitis_ai_dpu_resource_model, 'xilinx_vitis_ai_b4096', 'DPU'),
    ('Stanford Plasticine CGRA', stanford_plasticine_cgra_resource_model, 'stanford_plasticine_v2', 'CGRA'),
]
KPU_MODELS = [m for m in ACCELERATOR_MODELS if m[3] == 'KPU']

_PRODUCTS = load_compute_products_unified()
_NODES = load_process_nodes()


def _catalog_node(product_id: str):
    """ProcessNodeEntry of the catalog product's (first compute) die."""
    cp = _PRODUCTS[product_id]
    die = kpu_die_of(cp) if product_id.startswith('kpu_') else cp.dies[0]
    return _NODES[die.process_node_id]


def _kpu_fabric_type(tile_type: str) -> str:
    return "kpu_" + tile_type.lower().replace("-", "_")


@pytest.mark.parametrize(
    "name,model_fn,product_id,arch_type", ACCELERATOR_MODELS, ids=[m[0] for m in ACCELERATOR_MODELS]
)
def test_fabric_process_node_matches_catalog(name, model_fn, product_id, arch_type):
    """Every compute fabric is modeled at the catalog product's process node."""
    model = model_fn()
    assert model.compute_fabrics, f"{name}: missing compute_fabrics"
    expected_nm = int(_catalog_node(product_id).node_nm)
    for fabric in model.compute_fabrics:
        assert fabric.process_node_nm == expected_nm, (
            f"{name}: fabric {fabric.fabric_type} at {fabric.process_node_nm} nm, "
            f"catalog {product_id} is {expected_nm} nm"
        )


def test_phase4_completeness():
    """Test that all 5 Phase 4 accelerator models are accounted for."""
    assert len(ACCELERATOR_MODELS) == 5, "Expected 5 Phase 4 accelerator models"
    assert all(pid in _PRODUCTS for _, _, pid, _ in ACCELERATOR_MODELS)


@pytest.mark.parametrize("name,model_fn,product_id,_arch", KPU_MODELS, ids=[m[0] for m in KPU_MODELS])
def test_kpu_fabrics_mirror_catalog_tile_classes(name, model_fn, product_id, _arch):
    """One fabric per catalog tile class (INT8-primary / BF16-primary / Matrix),
    each with the per-op FP32 energy of its tile class's standard-cell library."""
    model = model_fn()
    node = _catalog_node(product_id)
    tiles = kpu_block_of(_PRODUCTS[product_id]).tiles
    fabrics = {f.fabric_type: f for f in model.compute_fabrics}

    assert set(fabrics) == {_kpu_fabric_type(t.tile_type) for t in tiles}, name
    for tile in tiles:
        fabric = fabrics[_kpu_fabric_type(tile.tile_type)]
        assert fabric.num_units == tile.num_tiles, (name, tile.tile_type)
        expected_pj = node.energy_per_op_pj[f"{tile.pe_circuit_class.value}:fp32"]
        assert fabric.energy_per_flop_fp32 * 1e12 == pytest.approx(expected_pj), (
            f"{name}: {fabric.fabric_type} FP32 energy "
            f"{fabric.energy_per_flop_fp32 * 1e12:.3f} pJ, ProcessNode "
            f"{node.id} {tile.pe_circuit_class.value}:fp32 = {expected_pj} pJ"
        )


@pytest.mark.parametrize("name,model_fn,product_id,_arch", KPU_MODELS, ids=[m[0] for m in KPU_MODELS])
def test_kpu_model_energy_is_per_flop(name, model_fn, product_id, _arch):
    """Model-level FP32 energy = ProcessNode per-MAC balanced-logic energy / 2
    (2 FLOPs per MAC; graphs#81)."""
    model = model_fn()
    node = _catalog_node(product_id)
    per_mac_pj = node.energy_per_op_pj[f"{CircuitClass.BALANCED_LOGIC.value}:fp32"]
    assert model.energy_per_flop_fp32 * 1e12 == pytest.approx(per_mac_pj / 2.0), name


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


def test_matrix_tiles_use_the_hp_library():
    """Matrix tiles are built on HP logic in the catalog, so they cost more per
    op than the balanced-logic INT8 tiles on the same node."""
    for name, model_fn, product_id, _ in KPU_MODELS:
        tiles = {t.tile_type: t for t in kpu_block_of(_PRODUCTS[product_id]).tiles}
        assert tiles["Matrix"].pe_circuit_class == CircuitClass.HP_LOGIC, name
        fabrics = {f.fabric_type: f for f in model_fn().compute_fabrics}
        assert (
            fabrics["kpu_matrix"].energy_per_flop_fp32
            > fabrics["kpu_int8_primary"].energy_per_flop_fp32
        ), name


def test_energy_falls_with_process_node():
    """Energy per FP32 FLOP decreases strictly as the node shrinks: every model
    at a smaller node beats every model at a larger node."""
    by_node: dict[int, list[tuple[str, float]]] = {}
    for name, model_fn, product_id, _ in ACCELERATOR_MODELS:
        model = model_fn()
        nm = int(_catalog_node(product_id).node_nm)
        by_node.setdefault(nm, []).append((name, model.energy_per_flop_fp32 * 1e12))
    nodes = sorted(by_node)
    assert len(nodes) >= 3, f"expected several process nodes, got {nodes}"
    for smaller, larger in zip(nodes, nodes[1:]):
        worst_small = max(by_node[smaller], key=lambda x: x[1])
        best_large = min(by_node[larger], key=lambda x: x[1])
        assert worst_small[1] < best_large[1], (
            f"{worst_small[0]} at {smaller} nm ({worst_small[1]:.2f} pJ) is not below "
            f"{best_large[0]} at {larger} nm ({best_large[1]:.2f} pJ)"
        )


def test_kpu_t64_vs_t256_vs_t768():
    """KPU family: tile counts match the catalog, and the family shares one set
    of tile classes."""
    fabric_sets = []
    for name, model_fn, product_id, _ in KPU_MODELS:
        model = model_fn()
        assert model.compute_units == kpu_block_of(_PRODUCTS[product_id]).total_tiles, name
        fabric_sets.append({f.fabric_type for f in model.compute_fabrics})
    assert [m.compute_units for m in (f() for _, f, _, _ in KPU_MODELS)] == [64, 256, 768]
    assert all(s == fabric_sets[0] for s in fabric_sets), (
        "All KPU models should have the same fabric types"
    )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
