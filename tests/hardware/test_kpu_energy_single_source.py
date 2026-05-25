"""KPU dynamic energy is single-sourced from the process node (issue #81).

The EnergyAnalyzer bills compute energy as ``sg.flops * energy_per_flop_fp32 *
energy_scaling[precision]``. Both coefficients must derive from the embodied-
schemas ProcessNode ``energy_per_op_pj`` table -- not the parallel graphs-local
``PROCESS_NODE_ENERGY`` table, and not the hardcoded ``energy_scaling`` defaults.

Two correctness properties pinned here:
  1. UNITS: ``energy_per_op_pj`` is per-MAC; ``sg.flops`` is 2 FLOPs/MAC, so the
     per-FLOP base is the node FP32 figure / 2 (the #81 2x double-count fix).
  2. SINGLE SOURCE: per-precision ``energy_scaling`` equals the node's
     ``energy_per_op_pj`` ratios (prec / fp32), not the dataclass defaults.
"""

from __future__ import annotations

import pytest

from embodied_schemas import load_process_nodes

from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.accelerators.kpu_t128 import kpu_t128_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.accelerators.kpu_t768 import kpu_t768_resource_model
from graphs.hardware.resource_model import Precision


# (factory, embodied-schemas process_node_id) for each SKU.
_KPU_SKUS = [
    (kpu_t64_resource_model, "tsmc_n16"),
    (kpu_t128_resource_model, "tsmc_n16"),
    (kpu_t256_resource_model, "tsmc_n16"),
    (kpu_t768_resource_model, "tsmc_n7"),
]


@pytest.fixture(scope="module")
def nodes():
    return load_process_nodes()


@pytest.mark.parametrize("factory,node_id", _KPU_SKUS,
                         ids=[f.__name__ for f, _ in _KPU_SKUS])
def test_energy_per_flop_is_node_fp32_halved(factory, node_id, nodes):
    """FP32 per-FLOP base == node balanced_logic:fp32 per-MAC / 2."""
    model = factory()
    node_fp32_pj = nodes[node_id].energy_per_op_pj["balanced_logic:fp32"]
    expected_j_per_flop = node_fp32_pj * 1e-12 / 2.0
    assert model.energy_per_flop_fp32 == pytest.approx(expected_j_per_flop, rel=1e-9)


@pytest.mark.parametrize("factory,node_id", _KPU_SKUS,
                         ids=[f.__name__ for f, _ in _KPU_SKUS])
def test_energy_scaling_matches_node_ratios(factory, node_id, nodes):
    """Per-precision energy_scaling == node energy_per_op_pj ratios (prec/fp32),
    not the hardcoded dataclass defaults (which had INT8=0.125)."""
    model = factory()
    epj = nodes[node_id].energy_per_op_pj
    fp32 = epj["balanced_logic:fp32"]
    for prec, key in [(Precision.INT8, "balanced_logic:int8"),
                      (Precision.BF16, "balanced_logic:bf16")]:
        if key not in epj:
            continue
        expected = epj[key] / fp32
        assert model.energy_scaling[prec] == pytest.approx(expected, rel=1e-9), (
            f"{factory.__name__} {prec.value}: scaling {model.energy_scaling[prec]} "
            f"!= node ratio {expected}"
        )
    # INT8 must have moved off the 0.125 dataclass default.
    assert model.energy_scaling[Precision.INT8] != pytest.approx(0.125)


def test_effective_int8_mac_energy_equals_node(nodes):
    """End-to-end: the per-MAC INT8 compute energy the EnergyAnalyzer bills
    (energy_per_flop_fp32 * scaling[int8] * 2 FLOPs/MAC) equals the node's
    balanced_logic:int8 per-MAC figure for T64."""
    model = kpu_t64_resource_model()
    node_int8_pj = nodes["tsmc_n16"].energy_per_op_pj["balanced_logic:int8"]
    eff_j_per_mac = model.energy_per_flop_fp32 * model.energy_scaling[Precision.INT8] * 2.0
    assert eff_j_per_mac == pytest.approx(node_int8_pj * 1e-12, rel=1e-9)
