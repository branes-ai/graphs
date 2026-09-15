"""Phase C4 of the KPU heterogeneous-tile refactor (graphs#268): validators.

- The existing validators handle heterogeneous KPUs: area roll-ups and the
  library check include tile-carried silicon, the floorplan no longer fails
  on systolic / fixed-function tiles, and the TOPS/W envelope compares
  programmable TOPS with programmable power.
- New validators (``validators/heterogeneous.py``) are silent on every
  legacy SKU and fire on targeted variants of the synthetic fixture.
"""

from __future__ import annotations

import pytest
from embodied_schemas import (
    ComputeProduct,
    load_compute_products,
    load_cooling_solutions,
    load_process_nodes,
)

from graphs.hardware.kpu_access import has_kpu_block
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_sku_generator import generate_kpu_sku, input_spec_from_compute_product
from graphs.hardware.kpu_sku_input import KPUSKUInputSpec
from graphs.hardware.sku_validators import (
    Severity,
    ValidatorContext,
    build_context_for_kpu,
    default_registry,
    load_validators,
)
from graphs.hardware.sku_validators.validators import energy

load_validators()
NODES = load_process_nodes()
N16 = NODES["tsmc_n16"]
COOLING = load_cooling_solutions()
CATALOG = sorted(k for k, v in load_compute_products().items() if has_kpu_block(v))
HETERO_SPEC = input_spec_from_compute_product(build_heterogeneous_kpu())
HETERO = generate_kpu_sku(HETERO_SPEC, process_nodes=NODES)
NEW = [
    "fixed_function_energy_plausibility", "datapath_energy_resolution",
    "silicon_no_double_count", "tile_footprint_pitch_fit", "checkerboard_site_accounting",
    "stream_link_adjacency", "cluster_geometry_consistent", "overlay_consistency",
    "stream_link_bandwidth", "power_domain_coverage", "cluster_rail_and_clock",
]


def _ctx(sku: ComputeProduct, node=N16) -> ValidatorContext:
    return ValidatorContext(sku=sku, process_node=node, cooling_solutions=COOLING,
                            extras={"process_nodes": NODES})


def _run(name: str, sku: ComputeProduct, node=N16) -> list:
    return default_registry._validators[name].check(_ctx(sku, node))


def _spec_variant(mutate) -> ComputeProduct:
    """Edit the fixture's input spec and regenerate (roll-ups follow)."""
    data = HETERO_SPEC.model_dump(mode="json")
    mutate(data)
    return generate_kpu_sku(KPUSKUInputSpec.model_validate(data), process_nodes=NODES)


def _sku_variant(mutate) -> ComputeProduct:
    """Edit the generated SKU directly (for states the generator refuses)."""
    data = HETERO.model_dump(mode="json")
    mutate(data)
    return ComputeProduct.model_validate(data)


def _tile(data, cid):
    arch = data.get("kpu_architecture") or data["dies"][0]["blocks"][0]
    return next(t for t in arch["tiles"] if t["tile_class_id"] == cid)


def _sev(findings, sev):
    return [f for f in findings if f.severity == sev]


# ---------------------------------------------------------------------------
# Legacy and fixture-wide behavior
# ---------------------------------------------------------------------------


def test_all_new_validators_are_registered():
    assert set(NEW) <= set(default_registry.names())


@pytest.mark.parametrize("sku", CATALOG)
def test_new_validators_are_silent_on_legacy_skus(sku):
    ctx = build_context_for_kpu(sku)
    for name in NEW:
        assert default_registry._validators[name].check(ctx) == [], name


def test_every_validator_runs_on_the_fixture():
    ctx = _ctx(HETERO)
    for name, v in default_registry._validators.items():
        v.check(ctx)  # no validator raises
    pitch = _run("floorplan_pitch_match", HETERO)
    assert not any("could not derive floorplan" in f.message for f in pitch)
    assert _run("area_self_consistency", HETERO) == []  # carried silicon counted


def test_area_self_consistency_reports_a_carried_silicon_failure():
    """If the tile-carried silicon cannot be resolved (here: a reference
    node missing from the catalog), say so instead of silently comparing
    the silicon_bin alone (CodeRabbit on #282)."""
    ctx = ValidatorContext(sku=HETERO, process_node=N16, cooling_solutions=COOLING,
                           extras={"process_nodes": {"tsmc_n16": N16}})
    findings = default_registry._validators["area_self_consistency"].check(ctx)
    warned = [f for f in findings if "tile-carried silicon could not be resolved" in f.message]
    assert warned and warned[0].severity == Severity.WARNING
    assert "tsmc_n40" in warned[0].message


def test_block_library_validity_covers_carried_silicon():
    findings = _run("block_library_validity", HETERO)
    assert [f.block for f in findings] == ["systolic_int8_ws.memory.accumulator"]
    assert findings[0].severity == Severity.ERROR and "sram_hp" in findings[0].message
    # N7 offers dual-port SRAM (embodied-schemas#96 is an N16 / GF12 gap).
    assert _run("block_library_validity", HETERO, NODES["tsmc_n7"]) == []


def test_tops_envelope_uses_programmable_power(monkeypatch):
    ff = energy._fixed_function_w(_ctx(HETERO))
    assert ff > 0
    assert energy._fixed_function_w(build_context_for_kpu(CATALOG[0])) == 0.0
    tops = HETERO.performance.int8_tops
    # A TDP whose programmable share breaches the 16 nm ceiling (30 TOPS/W)
    # while the whole TDP would not.
    monkeypatch.setattr(energy, "_fixed_function_w", lambda ctx: 1.0)
    sku = _sku_variant(lambda d: d["power"].update(tdp_watts=tops / 31 + 1.0))
    findings = _run("tops_per_watt_envelope", sku)
    assert [f.severity for f in findings] == [Severity.ERROR]
    assert tops / sku.power.tdp_watts < 30  # the unsplit ratio passes
    # The message names the denominator it used (CodeRabbit on #282).
    assert findings[0].message.startswith(
        "int8_tops / programmable_watts (tdp_watts - 1.00 W fixed-function) = "
    )


# ---------------------------------------------------------------------------
# ENERGY
# ---------------------------------------------------------------------------


def test_fixed_function_energy_plausibility():
    assert _run("fixed_function_energy_plausibility", HETERO) == []  # = library anchors

    def inflate_isp(data):
        _tile(data, "ff_isp_raw2yuv")["core"]["energy"]["pj_per_unit"] *= 10
    findings = _run("fixed_function_energy_plausibility", _spec_variant(inflate_isp))
    assert [(f.severity, f.block) for f in findings] == [(Severity.WARNING, "ff_isp_raw2yuv")]
    assert "10.00x the library anchor 'ff_isp_raw2yuv'" in findings[0].message

    def unknown_function(data):
        t = _tile(data, "ff_isp_raw2yuv")
        t["core"]["function_id"] = "isp.hdr_merge"
        t["tile_class_ref"] = None
    findings = _run("fixed_function_energy_plausibility", _spec_variant(unknown_function))
    assert [f.severity for f in findings] == [Severity.INFO]
    assert "no tile-class library anchor for function 'isp.hdr_merge'" in findings[0].message


def test_datapath_energy_resolution():
    assert _run("datapath_energy_resolution", HETERO) == []
    no_bf16 = N16.model_copy(update={
        "energy_per_op_pj": {k: v for k, v in N16.energy_per_op_pj.items()
                             if k != "balanced_logic:bf16"}
    })
    errors = _sev(_run("datapath_energy_resolution", HETERO, no_bf16), Severity.ERROR)
    assert errors and all("'balanced_logic:bf16'" in f.message for f in errors)
    assert {f.block for f in errors} == {"pe_int8_mac_i32", "pe_lns16_mac"}

    def absolute_on_unknown_node(data):
        mode = _tile(data, "systolic_int8_ws")["mac"]["modes"][0]
        mode["energy"] = {"kind": "absolute", "pj": 0.2, "ref_node_id": "tsmc_n90"}
    findings = _run("datapath_energy_resolution", _spec_variant(absolute_on_unknown_node))
    assert [f.severity for f in findings] == [Severity.WARNING]
    assert "'tsmc_n90' is not in the catalog" in findings[0].message


# ---------------------------------------------------------------------------
# AREA
# ---------------------------------------------------------------------------


def test_silicon_no_double_count():
    assert _run("silicon_no_double_count", HETERO) == []

    def count_lns_twice(data):
        data["dies"][0]["silicon_bin"]["blocks"].append({
            "name": "pe_lns", "circuit_class": "balanced_logic",
            "transistor_source": {"kind": "per_pe", "per_unit_mtx": 0.01,
                                  "count_ref": "tile.LNS16-MAC"},
        })
    findings = _run("silicon_no_double_count", _sku_variant(count_lns_twice))
    assert [(f.severity, f.block) for f in findings] == [(Severity.ERROR, "pe_lns16_mac")]


def test_tile_footprint_pitch_fit():
    findings = _run("tile_footprint_pitch_fit", HETERO)
    # The 64x64 systolic array and the SGM core outgrow one 32x32-PE site;
    # the 2x2 VIO footprint with its absorbed memory cells fits.
    assert sorted(f.block for f in findings) == ["ff_stereo_sgm", "systolic_int8_ws"]
    assert all(f.severity == Severity.WARNING for f in findings)

    def shrink_vio(data):
        _tile(data, "ff_vio_stereo_inertial")["footprint"] = None
        data["kpu_architecture"]["checkerboard"]["spare_sites"] += 3
    blocks = {f.block for f in _run("tile_footprint_pitch_fit", _spec_variant(shrink_vio))}
    assert "ff_vio_stereo_inertial" in blocks


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------


def test_checkerboard_site_accounting():
    findings = _run("checkerboard_site_accounting", HETERO)
    assert [f.severity for f in findings] == [Severity.INFO]
    assert "16 spare (25%)" in findings[0].message

    def no_checkerboard(mesh_cols):
        def mutate(data):
            block = data["dies"][0]["blocks"][0]
            block["checkerboard"] = None
            block["noc"]["mesh_cols"] = mesh_cols
        return _sku_variant(mutate)
    wide = _run("checkerboard_site_accounting", no_checkerboard(8))  # 64 routers, 48 sites
    assert [f.severity for f in wide] == [Severity.WARNING] and "16 site(s) are implicit" in wide[0].message
    narrow = _run("checkerboard_site_accounting", no_checkerboard(5))  # 40 routers
    assert [f.severity for f in narrow] == [Severity.ERROR] and "drop 8 tile site(s)" in narrow[0].message


def test_stream_link_adjacency():
    findings = _run("stream_link_adjacency", HETERO)
    assert [f.severity for f in findings] == [Severity.INFO, Severity.INFO]

    def hint(data):
        _tile(data, "ff_stereo_sgm")["placement"] = {
            "adjacent_to": ["ff_isp_raw2yuv", "ff_vio_stereo_inertial"]}
    assert _run("stream_link_adjacency", _spec_variant(hint)) == []

    def explicit(sgm_at):
        def mutate(data):
            arch = data["kpu_architecture"]
            grid = [["." for _ in range(8)] for _ in range(8)]
            for r in range(2):
                for c in range(2):
                    grid[r][c] = "ff_vio_stereo_inertial"
            grid[0][2] = "ff_isp_raw2yuv"
            grid[sgm_at[0]][sgm_at[1]] = "ff_stereo_sgm"
            cells = [t["tile_class_id"] for t in arch["tiles"] if t["footprint"] is None
                     and t["tile_class_id"] not in ("ff_isp_raw2yuv", "ff_stereo_sgm")
                     for _ in range(t["num_tiles"])]
            free = [(r, c) for r in range(8) for c in range(8) if grid[r][c] == "."]
            for cid, (r, c) in zip(cells, free):
                grid[r][c] = cid
            arch["checkerboard"].update(placement="explicit", placement_map=grid)
        return _spec_variant(mutate)
    # SGM at (0,3): next to the ISP at (0,2), but not to the VIO block (rows 0-1, cols 0-1).
    apart = _run("stream_link_adjacency", explicit((0, 3)))
    assert [f.severity for f in apart] == [Severity.WARNING]
    assert "'ff_stereo_sgm' and 'ff_vio_stereo_inertial' are not adjacent" in apart[0].message


# ---------------------------------------------------------------------------
# INTERNAL / ELECTRICAL
# ---------------------------------------------------------------------------


def test_overlay_consistency():
    assert _run("overlay_consistency", HETERO) == []

    def uncosted(data):
        _tile(data, "pe_int8_mac_i32")["interconnect"]["overlays"][0].pop("mtx_per_instance")
    findings = _run("overlay_consistency", _spec_variant(uncosted))
    assert [f.severity for f in findings] == [Severity.INFO]
    assert "pe_int8_mac_i32.row_bcast" in findings[0].message


def test_stream_link_bandwidth():
    assert _run("stream_link_bandwidth", HETERO) == []

    def narrow_link(isp_out):
        def mutate(data):
            data["kpu_architecture"]["noc"]["overlays"][0]["width_bytes"] = 1
            _tile(data, "ff_isp_raw2yuv")["core"]["io"]["output_bytes_per_unit"] = isp_out
        return _spec_variant(mutate)
    # On a 1 B/clk link the VIO's input (a 752x480 stereo pair per 8.8e5
    # clocks, ~0.82 B/clk) is at 82%: a WARNING in both variants.
    vio = (Severity.WARNING, "ff_vio_stereo_inertial")
    over = _run("stream_link_bandwidth", narrow_link(1.5))  # ISP: 1.5 B/clk
    assert {(f.severity, f.block) for f in over} == {(Severity.ERROR, "ff_isp_raw2yuv"), vio}
    near = _run("stream_link_bandwidth", narrow_link(0.9))  # ISP: 90%
    assert {(f.severity, f.block) for f in near} == {(Severity.WARNING, "ff_isp_raw2yuv"), vio}


def _with_domains(domains, **tile_domain) -> ComputeProduct:
    def mutate(data):
        data["kpu_architecture"]["power_domains"] = domains
        for cid, did in tile_domain.items():
            _tile(data, cid)["power_domain_id"] = did
    return _spec_variant(mutate)


def _cluster(did, r0, r1, c0, c1, **extra):
    return {"domain_id": did, "kind": "cluster",
            "site_ranges": [{"row_min": r0, "row_max": r1, "col_min": c0, "col_max": c1}],
            **extra}


def test_power_domain_coverage():
    ff = [{"domain_id": f"d_{c}", "kind": "tile_class", "members": [c], "gateable": True}
          for c in ("ff_isp_raw2yuv", "ff_stereo_sgm", "ff_vio_stereo_inertial")]
    # Only the fixed-function classes are in domains: the PE / systolic classes are not.
    findings = _run("power_domain_coverage", _with_domains(ff))
    warned = sorted(f.block for f in _sev(findings, Severity.WARNING))
    assert warned == ["pe_int8_mac_i32", "pe_lns16_mac", "pe_minplus_i16", "systolic_int8_ws"]
    assert any("no uncore power domain" in f.message for f in _sev(findings, Severity.INFO))

    # Clusters cover the classes with PEs; an uncore domain silences the INFO.
    full = ff + [_cluster("c0", 0, 7, 0, 3, rail_id="v0", clock_domain_id="k0"),
                 _cluster("c1", 0, 7, 4, 7, rail_id="v1", clock_domain_id="k1"),
                 {"domain_id": "uncore", "kind": "uncore"}]
    assert _run("power_domain_coverage", _with_domains(full)) == []


def test_cluster_validators():
    ok = [_cluster(f"c{i}", 2 * (i // 4), 2 * (i // 4) + 1, 2 * (i % 4), 2 * (i % 4) + 1,
                   rail_id=f"v{i}", clock_domain_id=f"k{i}") for i in range(16)]
    sku = _with_domains(ok)
    assert _run("cluster_rail_and_clock", sku) == []
    assert _run("cluster_geometry_consistent", sku) == []  # 16 equal 2x2 clusters

    bad = [_cluster("big", 0, 3, 0, 7), _cluster("small", 4, 5, 0, 1, rail_id="v")]
    sku = _with_domains(bad)
    rails = _run("cluster_rail_and_clock", sku)
    assert {f.message.split("'")[1] for f in rails} == {"big", "small"}
    geometry = _run("cluster_geometry_consistent", sku)
    assert [f.severity for f in geometry] == [Severity.WARNING, Severity.INFO]
    assert "different shapes ['2x2', '4x8']" in geometry[0].message
    assert "2 cluster power domain(s)" in geometry[1].message

    # Same site count, different shape (CodeRabbit on #282): 2x8 vs 4x4.
    same_count = [_cluster("wide", 0, 1, 0, 7, rail_id="v0", clock_domain_id="k0"),
                  _cluster("square", 2, 5, 0, 3, rail_id="v1", clock_domain_id="k1")]
    geometry = _run("cluster_geometry_consistent", _with_domains(same_count))
    assert _sev(geometry, Severity.WARNING)
    assert "different shapes ['2x8', '4x4']" in _sev(geometry, Severity.WARNING)[0].message
