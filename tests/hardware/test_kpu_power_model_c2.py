"""Phase C2 of the KPU heterogeneous-tile refactor (graphs#268): kpu_power_model.

- Legacy-shaped profiles keep the original TDP formula (golden-pinned), and
  the heterogeneous engine reproduces it exactly for every catalog SKU.
- Per-kind compute energy: datapath EnergyRef (relative / absolute),
  systolic cells, fixed-function units/s x pj_per_unit scaled by node.
- Per power domain V/f and gating; the uncore Vdd for memory / NoC.
- ``tdp_scenario``: per-class activity, each class in its worst mode.
"""

from __future__ import annotations

import pytest
from embodied_schemas import load_compute_products, load_kpu_tile_classes, load_process_nodes
from embodied_schemas.process_node import CircuitClass

from graphs.hardware.kpu_access import has_kpu_block
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_power_model import (
    ANCHOR_OPS_PER_INVOCATION,
    DEFAULT_WORKLOAD,
    HeterogeneousTDPBreakdown,
    TDPBreakdown,
    compute_heterogeneous_tdp_breakdown,
    compute_thermal_profile_tdp_breakdown,
    compute_thermal_profile_tdp_w,
    is_legacy_shaped,
)
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.kpu_sku_input import KPUSKUInputSpec

NODES = load_process_nodes()
N16 = NODES["tsmc_n16"]
CATALOG = {k: v for k, v in load_compute_products().items() if has_kpu_block(v)}
HETERO_SPEC = input_spec_from_compute_product(build_heterogeneous_kpu())
TERMS = ("pe_compute_w", "l2_sram_w", "l3_sram_w", "noc_w", "dram_phy_w", "leakage_w")


def _profile(spec: KPUSKUInputSpec, **over):
    base = spec.thermal_profiles[0]
    return base.model_validate({**base.model_dump(mode="json"), **over})


def _spec_with(spec: KPUSKUInputSpec, mutate) -> KPUSKUInputSpec:
    data = spec.model_dump(mode="json")
    mutate(data)
    return KPUSKUInputSpec.model_validate(data)


def _scenario(**active) -> dict:
    """A tdp_scenario over every fixture class: listed ones active, others 0."""
    ids = [t.tile_class_id for t in HETERO_SPEC.kpu_architecture.tiles]
    return {cid: active.get(cid, 0.0) for cid in ids}


# ---------------------------------------------------------------------------
# Legacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(CATALOG))
def test_engine_reproduces_the_legacy_formula(sku):
    spec = input_spec_from_compute_product(CATALOG[sku])
    node = NODES[spec.process_node_id]
    for profile in spec.thermal_profiles:
        assert is_legacy_shaped(spec, profile)
        legacy = compute_thermal_profile_tdp_breakdown(spec, profile, node)
        assert type(legacy) is TDPBreakdown  # the golden records exactly these fields
        het = compute_heterogeneous_tdp_breakdown(spec, profile, node)
        assert het.worst_precision == legacy.worst_precision
        for term in TERMS:
            assert getattr(het, term) == pytest.approx(getattr(legacy, term), rel=1e-12), term
        assert het.fixed_function_w == 0.0
        assert het.total_tdp_w == pytest.approx(legacy.total_tdp_w, rel=1e-12)


def test_legacy_shape_detection():
    spec = input_spec_from_compute_product(CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"])
    profile = spec.thermal_profiles[0]
    assert is_legacy_shaped(spec, profile)
    assert not is_legacy_shaped(spec, profile.model_copy(update={"tdp_scenario": {"x": 1.0}}))
    assert not is_legacy_shaped(HETERO_SPEC, HETERO_SPEC.thermal_profiles[0])


def test_ratio_one_datapath_costs_exactly_the_legacy_anchor():
    """The library's legacy-equivalent INT8 class, run through the datapath
    path, gives the legacy numbers of the catalog INT8-primary class."""
    t64 = input_spec_from_compute_product(CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"])
    lib = load_kpu_tile_classes()["pe_int8_mac_i32"]

    def swap(data):
        tiles = data["kpu_architecture"]["tiles"]
        legacy = tiles[0]
        tiles[0] = lib.instantiate(
            legacy["num_tiles"], tile_type=legacy["tile_type"], tile_class_id="int8_primary"
        ).model_dump(mode="json")
    spec = _spec_with(t64, swap)
    for profile in t64.thermal_profiles:
        legacy = compute_thermal_profile_tdp_breakdown(t64, profile, N16)
        het = compute_heterogeneous_tdp_breakdown(spec, profile, N16)
        assert het.total_tdp_w == pytest.approx(legacy.total_tdp_w, rel=1e-12)


# ---------------------------------------------------------------------------
# The heterogeneous fixture
# ---------------------------------------------------------------------------


def test_fixture_flows_through_the_power_model():
    for profile in HETERO_SPEC.thermal_profiles:
        bd = compute_thermal_profile_tdp_breakdown(HETERO_SPEC, profile, N16)
        assert isinstance(bd, HeterogeneousTDPBreakdown)
        assert set(bd.compute_w_by_tile_class) == {
            t.tile_class_id for t in HETERO_SPEC.kpu_architecture.tiles
        }
        assert bd.fixed_function_w > 0 and bd.pe_compute_w > 0
        assert bd.dynamic_w == pytest.approx(
            bd.pe_compute_w + bd.l2_sram_w + bd.l3_sram_w + bd.noc_w + bd.dram_phy_w
            + bd.fixed_function_w
        )
        assert sum(bd.compute_w_by_tile_class.values()) == pytest.approx(
            bd.pe_compute_w + bd.fixed_function_w
        )
        assert compute_thermal_profile_tdp_w(HETERO_SPEC, profile, N16) == round(bd.total_tdp_w, 1)
        assert bd.notes == []


def test_relative_datapath_energy_lns():
    profile = _profile(HETERO_SPEC, tdp_scenario=_scenario(pe_lns16_mac=1.0))
    bd = compute_heterogeneous_tdp_breakdown(HETERO_SPEC, profile, N16)
    assert bd.worst_precision == "scenario"
    bf16 = N16.energy_per_op_pj["balanced_logic:bf16"]
    pes = 8 * 32 * 32
    # Modes: LNS16 1 lane at 0.45 x (2-op anchor); LNS8 2 lanes at 0.22. The
    # class runs its worst-power mode: LNS16 (0.90 vs 0.88 anchor-ops per clock).
    per_clock_pj = 0.45 * ANCHOR_OPS_PER_INVOCATION * bf16 * pes
    vscale = (profile.vdd_v / N16.nominal_vdd_v) ** 2
    expected = per_clock_pj * profile.clock_mhz * 1e6 * 1e-12 * vscale
    assert bd.compute_w_by_tile_class["pe_lns16_mac"] == pytest.approx(expected)
    assert bd.pe_compute_w == pytest.approx(expected) and bd.fixed_function_w == 0.0


def test_fixed_function_energy_is_scaled_from_its_reference_node():
    profile = _profile(HETERO_SPEC, tdp_scenario=_scenario(ff_vio_stereo_inertial=1.0))
    bd = compute_heterogeneous_tdp_breakdown(HETERO_SPEC, profile, N16)
    n65 = NODES["tsmc_n65"]
    r_logic = N16.energy_per_op_pj["balanced_logic:int8"] / n65.energy_per_op_pj["balanced_logic:int8"]
    r_sram = (
        N16.sram_access_pj_per_byte[CircuitClass.SRAM_HD]
        / n65.sram_access_pj_per_byte[CircuitClass.SRAM_HD]
    )
    scale = 0.79 * r_logic + 0.21 * r_sram
    units_per_s = profile.clock_mhz * 1e6 / 8.8e5
    vscale = (profile.vdd_v / N16.nominal_vdd_v) ** 2
    expected = units_per_s * 3.38e8 * scale * 1e-12 * vscale
    assert bd.fixed_function_w == pytest.approx(expected)
    assert bd.pe_compute_w == 0.0
    # Fixed-function IO rides the NoC; there is no cache-hierarchy traffic.
    assert bd.l2_sram_w == bd.l3_sram_w == bd.dram_phy_w == 0.0 and bd.noc_w > 0


def test_scenario_scales_each_class_by_its_activity():
    full = compute_heterogeneous_tdp_breakdown(
        HETERO_SPEC, _profile(HETERO_SPEC, tdp_scenario=_scenario(
            pe_int8_mac_i32=1.0, systolic_int8_ws=1.0, ff_stereo_sgm=1.0)), N16)
    half = compute_heterogeneous_tdp_breakdown(
        HETERO_SPEC, _profile(HETERO_SPEC, tdp_scenario=_scenario(
            pe_int8_mac_i32=0.5, systolic_int8_ws=1.0, ff_stereo_sgm=0.25)), N16)
    f, h = full.compute_w_by_tile_class, half.compute_w_by_tile_class
    assert h["pe_int8_mac_i32"] == pytest.approx(0.5 * f["pe_int8_mac_i32"])
    assert h["systolic_int8_ws"] == pytest.approx(f["systolic_int8_ws"])
    assert f["systolic_int8_ws"] > 0
    assert h["ff_stereo_sgm"] == pytest.approx(0.25 * f["ff_stereo_sgm"])
    assert f["pe_lns16_mac"] == 0.0  # activity 0
    # The uniform sweep leaves int8-only classes idle at the fp16 worst case;
    # a scenario runs every class concurrently.
    uniform = compute_heterogeneous_tdp_breakdown(HETERO_SPEC, HETERO_SPEC.thermal_profiles[0], N16)
    assert uniform.compute_w_by_tile_class["systolic_int8_ws"] == 0.0


# ---------------------------------------------------------------------------
# Power domains: V/f, gating, uncore
# ---------------------------------------------------------------------------


def _with_domains(data):
    arch = data["kpu_architecture"]
    arch["power_domains"] = [
        {"domain_id": "pe_int8", "kind": "tile_class", "members": ["pe_int8_mac_i32"]},
        {"domain_id": "ff_vio", "kind": "tile_class", "members": ["ff_vio_stereo_inertial"],
         "gateable": True},
        {"domain_id": "uncore", "kind": "uncore"},
    ]


DOMAIN_SPEC = _spec_with(HETERO_SPEC, _with_domains)


def test_gated_domain_draws_no_power_or_leakage():
    scenario = _scenario(pe_int8_mac_i32=1.0, ff_vio_stereo_inertial=1.0)
    on = compute_heterogeneous_tdp_breakdown(
        DOMAIN_SPEC, _profile(DOMAIN_SPEC, tdp_scenario=scenario), N16)
    off = compute_heterogeneous_tdp_breakdown(
        DOMAIN_SPEC, _profile(DOMAIN_SPEC, tdp_scenario=scenario,
                              domain_operating_points={"ff_vio": {"gated": True}}), N16)
    assert off.gated_tile_classes == ["ff_vio_stereo_inertial"]
    assert "ff_vio_stereo_inertial" not in off.compute_w_by_tile_class
    assert off.fixed_function_w == 0.0 < on.fixed_function_w
    assert off.leakage_w < on.leakage_w  # the VIO core's silicon is power-gated
    assert off.pe_compute_w == pytest.approx(on.pe_compute_w)


def test_domain_clock_and_vdd_scale_their_classes():
    scenario = _scenario(pe_int8_mac_i32=1.0, pe_minplus_i16=1.0)
    base_profile = _profile(DOMAIN_SPEC, tdp_scenario=scenario)
    base = compute_heterogeneous_tdp_breakdown(DOMAIN_SPEC, base_profile, N16)
    slow = compute_heterogeneous_tdp_breakdown(
        DOMAIN_SPEC,
        _profile(DOMAIN_SPEC, tdp_scenario=scenario, domain_operating_points={
            "pe_int8": {"clock_mhz": base_profile.clock_mhz / 2, "vdd_v": 0.6}}),
        N16,
    )
    ratio = 0.5 * (0.6 / base_profile.vdd_v) ** 2
    b, s = base.compute_w_by_tile_class, slow.compute_w_by_tile_class
    assert s["pe_int8_mac_i32"] == pytest.approx(ratio * b["pe_int8_mac_i32"])
    assert s["pe_minplus_i16"] == pytest.approx(b["pe_minplus_i16"])  # no domain: unchanged
    assert slow.leakage_w < base.leakage_w  # INT8 PE silicon at the lower Vdd


def test_uncore_vdd_scales_memory_and_noc():
    scenario = _scenario(pe_int8_mac_i32=1.0)
    base_profile = _profile(DOMAIN_SPEC, tdp_scenario=scenario)
    base = compute_heterogeneous_tdp_breakdown(DOMAIN_SPEC, base_profile, N16)
    low = compute_heterogeneous_tdp_breakdown(
        DOMAIN_SPEC,
        _profile(DOMAIN_SPEC, tdp_scenario=scenario,
                 domain_operating_points={"uncore": {"vdd_v": 0.6}}),
        N16,
    )
    k = (0.6 / base_profile.vdd_v) ** 2
    for term in ("l2_sram_w", "l3_sram_w", "noc_w", "dram_phy_w"):
        assert getattr(low, term) == pytest.approx(k * getattr(base, term)), term
    assert low.pe_compute_w == pytest.approx(base.pe_compute_w)


def test_workload_duty_applies_without_a_scenario():
    profile = HETERO_SPEC.thermal_profiles[0]
    bd = compute_heterogeneous_tdp_breakdown(HETERO_SPEC, profile, N16)
    doubled = compute_heterogeneous_tdp_breakdown(
        HETERO_SPEC, profile, N16,
        workload=DEFAULT_WORKLOAD.__class__(compute_duty_cycle=2 * DEFAULT_WORKLOAD.compute_duty_cycle),
    )
    assert doubled.fixed_function_w == pytest.approx(2 * bd.fixed_function_w)
    assert doubled.pe_compute_w == pytest.approx(2 * bd.pe_compute_w)


@pytest.mark.parametrize("missing", ["hp_logic:bf16", "balanced_logic:bf16"])
def test_missing_energy_key_keeps_traffic_like_the_legacy_formula(missing):
    """A precision the node has no energy for still drives memory / NoC /
    DRAM traffic (CodeRabbit on #280). The engine must match the legacy
    formula term by term on a node without that key. Without
    ``hp_logic:bf16`` the Matrix class's bf16 ops are the case that used to
    lose their traffic."""
    spec = input_spec_from_compute_product(CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"])
    energies = {k: v for k, v in N16.energy_per_op_pj.items() if k != missing}
    node = N16.model_copy(update={"energy_per_op_pj": energies})
    for profile in spec.thermal_profiles:
        legacy = compute_thermal_profile_tdp_breakdown(spec, profile, node)
        het = compute_heterogeneous_tdp_breakdown(spec, profile, node)
        assert het.worst_precision == legacy.worst_precision
        for term in TERMS:
            assert getattr(het, term) == pytest.approx(getattr(legacy, term), rel=1e-12), term
        assert any(missing in n for n in het.notes)


def test_fixed_function_power_does_not_make_a_zero_compute_precision_eligible():
    """With only an INT8 PE class whose energies the node lacks, plus an ISP,
    no precision has programmable compute: the TDP is the precision-free
    evaluation (ISP power and its NoC IO), not a precision whose zero-energy
    ops would add cache / DRAM traffic (CodeRabbit on #280)."""
    def int8_and_isp(data):
        arch = data["kpu_architecture"]
        keep = {"pe_int8_mac_i32", "ff_isp_raw2yuv"}
        arch["tiles"] = [t for t in arch["tiles"] if t["tile_class_id"] in keep]
        arch["total_tiles"] = sum(t["num_tiles"] for t in arch["tiles"])
        arch["checkerboard"]["spare_sites"] = 64 - arch["total_tiles"]
        arch["noc"]["overlays"] = None
        data["silicon_bin"]["blocks"] = [
            b for b in data["silicon_bin"]["blocks"] if b["name"] != "pe_minplus"
        ]
    spec = _spec_with(HETERO_SPEC, int8_and_isp)
    missing = {"balanced_logic:int8", "balanced_logic:int4", "balanced_logic:bf16"}
    node = N16.model_copy(update={
        "energy_per_op_pj": {k: v for k, v in N16.energy_per_op_pj.items() if k not in missing}
    })
    bd = compute_heterogeneous_tdp_breakdown(spec, spec.thermal_profiles[0], node)
    assert bd.worst_precision == "(none)"
    assert bd.pe_compute_w == 0.0 and bd.fixed_function_w > 0
    assert bd.l2_sram_w == bd.l3_sram_w == bd.dram_phy_w == 0.0
    assert bd.noc_w > 0  # the ISP's IO
