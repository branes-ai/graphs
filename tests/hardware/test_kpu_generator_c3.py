"""Phase C3 of the KPU heterogeneous-tile refactor (graphs#268): generator and
input spec.

- Input-spec tiles may reference the tile-class library (``use:``).
- The generator adds tile-carried silicon to the die roll-up, emits the
  performance roll-up by tile kind for heterogeneous architectures, and
  derives TDP through the heterogeneous power model.
- ``apply_pe_array_override`` is scoped to pe_fabric classes (optionally one
  class); ``apply_tile_mix`` sets tile counts.
- Uniform legacy SKUs regenerate unchanged (the KPU golden gate pins that).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from embodied_schemas import (
    derive_kpu_performance,
    load_compute_products,
    load_kpu_tile_classes,
    load_process_nodes,
)

from graphs.hardware.kpu_access import has_kpu_block, kpu_die_of
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_power_model import compute_thermal_profile_tdp_w
from graphs.hardware.kpu_sku_generator import (
    GeneratorError,
    apply_pe_array_override,
    apply_tile_mix,
    generate_kpu_sku,
    input_spec_from_compute_product,
)
from graphs.hardware.kpu_sku_input import KPUSKUInputSpec, resolve_tile_uses
from graphs.hardware.sku_validators import silicon_math as sm

NODES = load_process_nodes()
CATALOG = {k: v for k, v in load_compute_products().items() if has_kpu_block(v)}
T64 = CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
HETERO = build_heterogeneous_kpu()
HETERO_SPEC = input_spec_from_compute_product(HETERO)
LIBRARY = load_kpu_tile_classes()
_CLI = Path(__file__).resolve().parents[2] / "cli" / "generate_kpu_sku.py"


def _use_spec(**tile_over) -> dict:
    """The T64 spec as a dict, its tiles replaced by library references."""
    data = input_spec_from_compute_product(T64).model_dump(mode="json")
    arch = data["kpu_architecture"]
    arch["tiles"] = [
        {"use": "pe_int8_mac_i32", "num_tiles": 44, "overrides": {"tile_type": "INT8-primary"}},
        {"use": "pe_bf16_fma", "num_tiles": 13},
        {"use": "pe_minplus_i16", "num_tiles": 7, **tile_over},
    ]
    del arch["total_tiles"]  # derived from the tile counts
    return data


# ---------------------------------------------------------------------------
# Input spec: library references
# ---------------------------------------------------------------------------


def test_use_entries_resolve_through_the_library():
    spec = KPUSKUInputSpec.model_validate(_use_spec())
    tiles = spec.kpu_architecture.tiles
    assert [t.tile_class_ref for t in tiles] == ["pe_int8_mac_i32", "pe_bf16_fma", "pe_minplus_i16"]
    assert tiles[0].tile_type == "INT8-primary" and tiles[0].num_tiles == 44
    assert tiles[0].datapath == LIBRARY["pe_int8_mac_i32"].tile.datapath
    assert spec.kpu_architecture.total_tiles == 64


@pytest.mark.parametrize(
    "over, match",
    [
        ({"use": "pe_nope"}, "'use: pe_nope' is not in the tile-class library"),
        ({"pe_array_rows": 16}, r"unknown keys \['pe_array_rows'\]; put tile fields under"),
    ],
)
def test_use_entry_errors(over, match):
    with pytest.raises(ValueError, match=match):
        KPUSKUInputSpec.model_validate(_use_spec(**over))
    with pytest.raises(ValueError, match="needs num_tiles"):
        resolve_tile_uses([{"use": "pe_bf16_fma"}], LIBRARY)


def test_plain_tiles_pass_through_untouched():
    tiles = [{"tile_type": "x"}]
    assert resolve_tile_uses(tiles, {}) is tiles


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(CATALOG))
def test_legacy_regeneration_emits_no_rollup(sku):
    spec = input_spec_from_compute_product(CATALOG[sku])
    cp = generate_kpu_sku(spec, process_nodes=NODES)
    assert not cp.performance.declares_rollup
    forced = generate_kpu_sku(spec, process_nodes=NODES, performance_rollup=True)
    assert forced.performance.declares_rollup
    for field in ("int8_tops", "bf16_tflops", "fp32_tflops", "int4_tops"):
        assert getattr(forced.performance, field) == getattr(cp.performance, field)


def test_heterogeneous_generation():
    cp = generate_kpu_sku(HETERO_SPEC, process_nodes=NODES)
    node = NODES[HETERO_SPEC.process_node_id]
    placeholder = HETERO  # same architecture and silicon_bin
    bin_areas = sm.resolve_all_block_areas(placeholder, node)
    carried = sm.resolve_carried_areas(placeholder, node, NODES)
    assert carried  # the fixture carries silicon
    die = kpu_die_of(cp)
    assert die.die_size_mm2 == round(sum(b.area_mm2 for b in bin_areas + carried), 1)
    assert die.transistors_billion == round(
        sum(b.transistors_mtx for b in bin_areas + carried) / 1000, 3
    )

    default = next(p for p in HETERO_SPEC.thermal_profiles
                   if p.name == HETERO_SPEC.default_thermal_profile)
    rollup = derive_kpu_performance(HETERO_SPEC.kpu_architecture.tiles, default.clock_mhz)
    assert cp.performance.by_tile_kind == rollup.by_tile_kind
    assert cp.performance.fixed_function_throughput == rollup.fixed_function_throughput
    assert cp.performance.int8_tops == rollup.int8_tops

    for p in cp.power.thermal_profiles:
        src = next(s for s in HETERO_SPEC.thermal_profiles if s.name == p.name)
        assert p.tdp_watts == compute_thermal_profile_tdp_w(HETERO_SPEC, src, node, nodes=NODES)
    assert cp.power.idle_power_watts == round(sm.total_chip_leakage_w(placeholder, node, NODES), 2)
    assert {t.tile_class_ref for t in cp.dies[0].blocks[0].tiles} == set(LIBRARY) - {
        "pe_bf16_fma", "pe_fp16_lerp",
    }


def test_double_counted_tile_logic_is_a_generator_error():
    """A chip-level PER_PE block for a class that carries its own datapath
    silicon would inflate the die roll-up (CodeRabbit on #281)."""
    data = HETERO_SPEC.model_dump(mode="json")
    data["silicon_bin"]["blocks"].append({
        "name": "pe_lns",
        "circuit_class": "balanced_logic",
        "transistor_source": {"kind": "per_pe", "per_unit_mtx": 0.01, "count_ref": "tile.LNS16-MAC"},
    })
    spec = KPUSKUInputSpec.model_validate(data)
    with pytest.raises(GeneratorError, match=r"\['pe_lns16_mac'\] carry their own logic silicon"):
        generate_kpu_sku(spec, process_nodes=NODES)


def test_heterogeneous_generation_round_trips():
    first = generate_kpu_sku(HETERO_SPEC, process_nodes=NODES)
    again = generate_kpu_sku(input_spec_from_compute_product(first), process_nodes=NODES)
    assert again == first


# ---------------------------------------------------------------------------
# --pe-array and --tile-mix
# ---------------------------------------------------------------------------


def test_bare_pe_array_resizes_pe_fabric_classes_only():
    spec = apply_pe_array_override(HETERO_SPEC, 16, 16)
    for before, after in zip(HETERO_SPEC.kpu_architecture.tiles, spec.kpu_architecture.tiles):
        if before.tile_kind.value == "pe_fabric":
            assert (after.pe_array_rows, after.pe_array_cols) == (16, 16)
            assert after.ops_per_tile_per_clock == {
                k: v / 4 for k, v in before.ops_per_tile_per_clock.items()
            }
            assert after.pipeline_fill_cycles == 16
        else:
            assert after == before
    # Every legacy class is pe_fabric: the bare form still resizes them all.
    legacy = apply_pe_array_override(input_spec_from_compute_product(T64), 16, 16)
    assert {(t.pe_array_rows, t.pe_array_cols) for t in legacy.kpu_architecture.tiles} == {(16, 16)}


def test_scoped_pe_array():
    by_id = apply_pe_array_override(HETERO_SPEC, 16, 8, tile_class="pe_int8_mac_i32")
    by_label = apply_pe_array_override(HETERO_SPEC, 16, 8, tile_class="INT8-MAC")
    assert by_id == by_label
    rows = {t.tile_class_id: getattr(t, "pe_array_rows", None) for t in by_id.kpu_architecture.tiles}
    assert rows["pe_int8_mac_i32"] == 16 and rows["pe_lns16_mac"] == 32
    with pytest.raises(ValueError, match="is systolic; --pe-array resizes pe_fabric classes only"):
        apply_pe_array_override(HETERO_SPEC, 16, 16, tile_class="systolic_int8_ws")
    with pytest.raises(ValueError, match="'nope' is unknown"):
        apply_pe_array_override(HETERO_SPEC, 16, 16, tile_class="nope")


def test_pe_array_result_is_revalidated():
    """An overlay span that no longer fits the smaller array is rejected."""
    def add_express(data):
        tile = data["kpu_architecture"]["tiles"][0]
        tile["interconnect"]["overlays"].append({
            "overlay_id": "express16", "kind": "express", "instances_per": "row",
            "width_bits": 32, "span": 16,
        })
    data = HETERO_SPEC.model_dump(mode="json")
    add_express(data)
    spec = KPUSKUInputSpec.model_validate(data)
    with pytest.raises(ValueError, match="span 16 does not fit a 8x8 PE array"):
        apply_pe_array_override(spec, 8, 8, tile_class="pe_int8_mac_i32")


def test_tile_mix():
    spec = apply_tile_mix(HETERO_SPEC, {"pe_int8_mac_i32": 30, "SGM": 2})
    counts = {t.tile_class_id: t.num_tiles for t in spec.kpu_architecture.tiles}
    assert counts["pe_int8_mac_i32"] == 30 and counts["ff_stereo_sgm"] == 2
    assert spec.kpu_architecture.total_tiles == 45 + 6 + 1
    assert spec.kpu_architecture.checkerboard.spare_sites == 64 - (48 + 7)
    with pytest.raises(ValueError, match="needs 65 compute sites but the checkerboard has 8x8 = 64"):
        apply_tile_mix(HETERO_SPEC, {"pe_int8_mac_i32": 41})
    with pytest.raises(ValueError, match="counts must be positive"):
        apply_tile_mix(HETERO_SPEC, {"pe_int8_mac_i32": 0})
    with pytest.raises(ValueError, match="'nope' is unknown"):
        apply_tile_mix(HETERO_SPEC, {"nope": 1})


def test_tile_mix_refuses_an_explicit_placement_map():
    data = HETERO_SPEC.model_dump(mode="json")
    cb = data["kpu_architecture"]["checkerboard"]
    grid = [["." for _ in range(8)] for _ in range(8)]
    tiles = data["kpu_architecture"]["tiles"]
    cells = [t["tile_class_id"] for t in tiles if t["tile_class_id"] != "ff_vio_stereo_inertial"
             for _ in range(t["num_tiles"])]
    flat = [(r, c) for r in range(8) for c in range(8) if not (r < 2 and c < 2)]
    for cid, (r, c) in zip(cells, flat):
        grid[r][c] = cid
    for r in range(2):
        for c in range(2):
            grid[r][c] = "ff_vio_stereo_inertial"
    cb.update(placement="explicit", placement_map=grid)
    spec = KPUSKUInputSpec.model_validate(data)
    with pytest.raises(ValueError, match="explicit placement_map"):
        apply_tile_mix(spec, {"pe_int8_mac_i32": 20})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_generates_from_a_library_spec(cli_runner, tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_use_spec(), sort_keys=False))
    out = tmp_path / "sku.yaml"
    rc, _, err = cli_runner(_CLI, [
        "--input", str(spec_path), "--output", str(out),
        "--tile-mix", "pe_int8_mac_i32=40,pe_minplus_i16=11",
        "--pe-array", "pe_bf16_fma=16x16",
    ])
    assert rc == 0, err
    tiles = yaml.safe_load(out.read_text())["dies"][0]["blocks"][0]["tiles"]
    assert [t["num_tiles"] for t in tiles] == [40, 13, 11]
    assert [t["tile_class_ref"] for t in tiles] == ["pe_int8_mac_i32", "pe_bf16_fma", "pe_minplus_i16"]
    assert tiles[1]["pe_array_rows"] == 16

    rc, _, err = cli_runner(_CLI, ["--input", str(spec_path), "--tile-mix", "pe_bf16_fma=0"])
    assert rc == 2 and "--tile-mix: tile mix: pe_bf16_fma=0; counts must be positive" in err
    for bad in ("16by16", "=16x16"):  # an empty class must not mean "every class"
        rc, _, err = cli_runner(_CLI, ["--input", str(spec_path), "--pe-array", bad])
        assert rc == 2 and "--pe-array must be ROWSxCOLS or CLASS=ROWSxCOLS" in err, bad
