"""Phase C6 of the KPU heterogeneous-tile refactor (graphs#268): the
inspection CLIs describe a tile by its kind.

``show_kpu`` and ``list_kpus`` were written when every tile was a PE
fabric, so they read ``pe_array_rows``, ``pe_circuit_class`` and
``total_pes`` straight off each tile and raised ``AttributeError`` on a
heterogeneous SKU. They now go through ``kpu_tile_display``.

No heterogeneous SKU is in the catalog yet (Phase E1), so these drive the
CLIs with ``--from-file`` against the generated fixture -- which is also
the workflow the flag exists for: inspect and validate a design while it
is still being iterated on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from embodied_schemas import load_process_nodes

from graphs.hardware import kpu_tile_display as display
from graphs.hardware.compute_product_loader import (
    ComputeProductFileError,
    load_compute_product_file,
    load_compute_products_unified,
)
from graphs.hardware.kpu_access import has_kpu_block, kpu_block_of
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_sku_generator import (
    generate_kpu_sku,
    input_spec_from_compute_product,
)
from hardware.test_kpu_catalog_ids import LEGACY_KPU_SKU_IDS

_CLI = Path(__file__).resolve().parents[2] / "cli"
_LEGACY = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"

_NON_KPU = "amd_epyc_9654_sp5"

HETERO = generate_kpu_sku(
    input_spec_from_compute_product(build_heterogeneous_kpu()),
    process_nodes=load_process_nodes(),
)


@pytest.fixture(scope="module")
def hetero_yaml(tmp_path_factory) -> Path:
    """The generated heterogeneous fixture, as a file on disk."""
    path = tmp_path_factory.mktemp("c6") / "hetero.yaml"
    path.write_text(
        yaml.safe_dump(HETERO.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# kpu_tile_display
# ---------------------------------------------------------------------------


def test_every_tile_kind_describes_itself():
    block = kpu_block_of(HETERO)
    by_type = {t.tile_type: t for t in block.tiles}

    fabric = by_type["INT8-MAC"]
    assert display.tile_kind(fabric) == "pe_fabric"
    assert display.tile_geometry(fabric) == (32, 32)
    assert display.tile_circuit_class(fabric) == "balanced_logic"
    assert display.tile_pe_count(fabric) == 1024
    assert "int8=2048" in display.tile_ops_str(fabric)
    assert "datapath=int8_mac_i32" in display.tile_detail(fabric)

    systolic = by_type["Systolic-INT8-WS"]
    assert display.tile_kind(systolic) == "systolic"
    # Its array is array_rows/array_cols, not pe_array_rows.
    assert display.tile_geometry(systolic) == (64, 64)
    assert display.tile_pe_count(systolic) == 64 * 64
    assert "dataflow=weight_stationary" in display.tile_detail(systolic)
    assert "gemm" in display.tile_detail(systolic)

    vio = by_type["VIO"]
    assert display.tile_kind(vio) == "fixed_function"
    # No array and no PEs -- reported as absent, not as zero-sized.
    assert display.tile_geometry(vio) is None
    assert display.tile_circuit_class(vio) is None
    assert display.tile_pe_count(vio) == 0
    assert display.tile_geometry_str(vio) == "-"
    assert "vio.stereo_inertial" in display.tile_detail(vio)


def test_a_slow_core_reports_cycles_per_unit():
    """The VIO core is specified as cycles-per-frame, not frames-per-clock;
    printing 1.1e-06 frames/clock would be useless."""
    vio = {t.tile_type: t for t in kpu_block_of(HETERO).tiles}["VIO"]
    assert display.fixed_function_throughput_str(vio) == "1 frame / 880000 clocks"
    isp = {t.tile_type: t for t in kpu_block_of(HETERO).tiles}["ISP"]
    assert display.fixed_function_throughput_str(isp) == "1 pixel/clock"


def test_min_plus_class_has_no_precision_keyed_ops():
    """Its ops are not keyed by a numeric format, so the operand formats
    stand in rather than an empty column."""
    minplus = {t.tile_type: t for t in kpu_block_of(HETERO).tiles}["MINPLUS-I16"]
    assert minplus.ops_per_tile_per_clock == {}
    text = display.tile_ops_str(minplus)
    assert "not precision-keyed" in text and "int16" in text


def test_footprints_and_site_accounting():
    block = kpu_block_of(HETERO)
    by_type = {t.tile_type: t for t in block.tiles}
    assert display.tile_footprint_str(by_type["INT8-MAC"]) == "1x1"
    # The VIO core is 2x2 and swallows the memory cells it covers.
    assert display.tile_footprint_str(by_type["VIO"]) == "2x2*"
    assert display.absorbs_memory_cells(by_type["VIO"]) is True
    assert display.tile_sites(by_type["VIO"]) == 4
    # 45 tiles, but the 2x2 VIO costs four sites: 44 + 4 = 48.
    assert block.total_tiles == 45
    assert display.occupied_sites(block) == 48


def test_census_and_pe_total_exclude_fixed_function():
    block = kpu_block_of(HETERO)
    assert display.kind_counts(block) == {
        "pe_fabric": 38, "systolic": 4, "fixed_function": 3
    }
    assert display.kind_summary(block) == "pe:38 systolic:4 fixed-fn:3"
    assert display.is_heterogeneous(block) is True
    # 38 fabric tiles x 1024 + 4 systolic x 4096; the 3 fixed-function
    # tiles contribute none.
    assert display.total_pe_count(block) == 38 * 1024 + 4 * 4096


@pytest.mark.parametrize("sku", LEGACY_KPU_SKU_IDS)
def test_legacy_skus_are_uniform_and_keep_their_pe_total(sku):
    """Every catalog SKU is one kind, and the helper reproduces the PE
    total the CLIs printed before C6 (sum of tile.total_pes)."""
    block = kpu_block_of(load_compute_products_unified()[sku])
    assert display.is_heterogeneous(block) is False
    assert list(display.kind_counts(block)) == ["pe_fabric"]
    assert display.total_pe_count(block) == sum(t.total_pes for t in block.tiles)
    assert display.occupied_sites(block) == block.total_tiles


# ---------------------------------------------------------------------------
# load_compute_product_file
# ---------------------------------------------------------------------------


def test_load_compute_product_file_round_trips(hetero_yaml):
    assert load_compute_product_file(hetero_yaml).id == HETERO.id


def test_load_compute_product_file_rejects_junk(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ComputeProductFileError, match="cannot read"):
        load_compute_product_file(missing)

    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ComputeProductFileError, match="ComputeProduct mapping"):
        load_compute_product_file(not_a_mapping)

    wrong_shape = tmp_path / "wrong.yaml"
    wrong_shape.write_text("id: x\n", encoding="utf-8")
    with pytest.raises(ComputeProductFileError, match="does not validate"):
        load_compute_product_file(wrong_shape)


# ---------------------------------------------------------------------------
# show_kpu
# ---------------------------------------------------------------------------


def test_show_kpu_renders_the_heterogeneous_fixture(cli_runner, hetero_yaml):
    rc, out, err = cli_runner(_CLI / "show_kpu.py", ["--from-file", str(hetero_yaml)])
    assert rc == 0, err
    assert "Traceback" not in err
    assert "Tile census:     pe:38 systolic:4 fixed-fn:3" in out
    # Each kind shows what it has, and a dash where it has nothing.
    assert "Systolic-INT8-WS    systolic    4   64x64     4096" in out
    assert "dataflow=weight_stationary" in out
    assert "function=vio.stereo_inertial, 1 frame / 880000 clocks" in out
    assert "Total PEs:       55296   (fixed-function tiles contribute none)" in out
    # Heterogeneous-only sections.
    assert "Checkerboard: 8x8 compute sites (64), 48 occupied by 45 tiles" in out
    assert "By tile kind (ops/sec):" in out
    assert "systolic  int8=15.6T" in out


def test_show_kpu_legacy_view_has_no_heterogeneous_sections(cli_runner):
    rc, out, err = cli_runner(_CLI / "show_kpu.py", [_LEGACY])
    assert rc == 0, err
    assert "Total PEs:       65536" in out
    assert "(fixed-function tiles contribute none)" not in out
    assert "Checkerboard:" not in out
    assert "By tile kind" not in out
    assert "Power domains:" not in out
    # The kind column is still there, saying every tile is a PE fabric.
    assert "Tile census:     pe:64" in out


def test_show_kpu_from_file_rejects_a_non_kpu_product(cli_runner, tmp_path):
    path = tmp_path / "cpu.yaml"
    path.write_text(
        yaml.safe_dump(
            load_compute_products_unified()[_NON_KPU].model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rc, _, err = cli_runner(_CLI / "show_kpu.py", ["--from-file", str(path)])
    assert rc == 1
    assert "has no KPU block" in err and "Traceback" not in err


def test_show_kpu_from_file_survives_an_unavailable_process_node_catalog(
    cli_runner, hetero_yaml, monkeypatch
):
    """A local file is readable on its own; a process-node catalog problem
    must not block inspecting it, only leave the node unresolved."""
    import sys

    # cli_runner imports each script under a synthesized module name and
    # caches it; run once to populate the cache, then break the loader the
    # script bound at import time.
    cli_runner(_CLI / "show_kpu.py", ["--from-file", str(hetero_yaml)])
    module = next(
        m for name, m in sys.modules.items() if name.startswith("_clitest_show_kpu_")
    )

    def _offline(*args, **kwargs):
        raise RuntimeError("catalog offline")

    monkeypatch.setattr(module, "load_process_nodes", _offline)
    rc, out, err = cli_runner(_CLI / "show_kpu.py", ["--from-file", str(hetero_yaml)])
    assert rc == 0, err
    assert "process node not in catalog" in out
    assert "warning: process-node catalog unavailable" in err
    assert "Tile census:     pe:38 systolic:4 fixed-fn:3" in out


def test_show_kpu_needs_exactly_one_source(cli_runner, hetero_yaml):
    rc, _, err = cli_runner(_CLI / "show_kpu.py", [])
    assert rc != 0 and "--from-file" in err
    rc, _, err = cli_runner(
        _CLI / "show_kpu.py", [_LEGACY, "--from-file", str(hetero_yaml)]
    )
    assert rc != 0 and "not both" in err


# ---------------------------------------------------------------------------
# list_kpus
# ---------------------------------------------------------------------------


def test_list_kpus_census_column_appears_now_that_a_heterogeneous_sku_exists(
    cli_runner
):
    """C6 said the column would earn its place once a heterogeneous SKU
    landed. embodied-schemas 0.10.0 landed two, so it does."""
    rc, out, err = cli_runner(_CLI / "list_kpus.py", [])
    assert rc == 0, err
    assert "tile kinds" in out
    assert "pe:38 systolic:4 fixed-fn:3" in out   # kpu_h64_auto1
    assert "pe:64" in out                          # a uniform T64, unchanged


def test_list_kpus_markdown_matches_the_text_rule(cli_runner, tmp_path):
    """The census column earns its place in every format on the same
    condition, so Markdown shows it exactly when the text view does."""
    out_path = tmp_path / "kpus.md"
    rc, _, err = cli_runner(_CLI / "list_kpus.py", ["--output", str(out_path)])
    assert rc == 0, err
    header, separator = out_path.read_text(encoding="utf-8").splitlines()[:2]
    assert "tile kinds" in header
    # One separator cell per header cell, or the table renders broken.
    assert header.count("|") == separator.count("|")

    # Filtered to the uniform SKUs, the column stands down again -- the
    # rule is about the rows being shown, not about the catalog.
    uniform = tmp_path / "uniform.md"
    rc, _, err = cli_runner(
        _CLI / "list_kpus.py", ["--library", "ffp", "--output", str(uniform)]
    )
    assert rc == 0, err
    header, separator = uniform.read_text(encoding="utf-8").splitlines()[:2]
    assert "tile kinds" in header  # kpu_h64_auto1's n16 variant is an ffp part
    assert header.count("|") == separator.count("|")


def test_list_kpus_kind_filter(cli_runner):
    # Every KPU has a PE fabric, heterogeneous ones included.
    rc, out, err = cli_runner(_CLI / "list_kpus.py", ["--kind", "pe_fabric"])
    assert rc == 0, err
    assert "14 KPU SKU(s)" in out

    # Only kpu_h64_auto1 carries the other kinds.
    rc, out, err = cli_runner(_CLI / "list_kpus.py", ["--kind", "systolic"])
    assert rc == 0, err
    assert "2 KPU SKU(s)" in out and "kpu_h64_auto1" in out

    rc, out, err = cli_runner(_CLI / "list_kpus.py", ["--kind", "heterogeneous"])
    assert rc == 0, err
    assert "2 KPU SKU(s)" in out
    assert "kpu_h64_auto1" in out and "kpu_t64" not in out


def test_list_kpus_csv_carries_the_kinds(cli_runner, tmp_path):
    out_path = tmp_path / "kpus.csv"
    rc, _, err = cli_runner(_CLI / "list_kpus.py", ["--output", str(out_path)])
    assert rc == 0, err
    text = out_path.read_text(encoding="utf-8")
    assert "tile_census" in text.splitlines()[0]
    assert "kinds" in text.splitlines()[0]
    assert "pe:64,pe_fabric,False" in text


# ---------------------------------------------------------------------------
# validate_sku
# ---------------------------------------------------------------------------


def test_validate_sku_from_file(cli_runner, hetero_yaml):
    rc, out, err = cli_runner(
        _CLI / "validate_sku.py", ["--from-file", str(hetero_yaml)]
    )
    # The fixture carries one known ERROR: its systolic accumulator asks for
    # sram_hp, which tsmc_n16 does not offer (embodied-schemas#96).
    assert rc == 1, err
    assert f"SKU validation: {HETERO.id}" in out
    assert "block_library_validity" in out
    assert "Traceback" not in err


def test_validate_sku_from_file_reports_a_bad_path(cli_runner, tmp_path):
    rc, _, err = cli_runner(
        _CLI / "validate_sku.py", ["--from-file", str(tmp_path / "nope.yaml")]
    )
    assert rc == 2  # framework / context error, not a finding
    assert "cannot read" in err and "Traceback" not in err


def test_validate_sku_catalog_mode_still_works(cli_runner):
    rc, _, err = cli_runner(_CLI / "validate_sku.py", [_LEGACY])
    assert rc == 0, err


# ---------------------------------------------------------------------------
# show_floorplan (graphs#268 D2)
# ---------------------------------------------------------------------------


def test_show_floorplan_renders_the_heterogeneous_fixture(cli_runner, hetero_yaml):
    rc, out, err = cli_runner(
        _CLI / "show_floorplan.py", ["--from-file", str(hetero_yaml), "--width", "64"]
    )
    assert rc == 0, err
    assert "Traceback" not in err
    # The new roles get their own glyphs, and the legend names them.
    assert "S=systolic" in out and "F=fixed_function" in out
    assert "S" in out and "F" in out
    # 64 sites less the 4 the 2x2 VIO core sits on.
    assert "N=  60" in out


def test_show_floorplan_site_overlay_by_tile_class(cli_runner, hetero_yaml):
    rc, out, err = cli_runner(
        _CLI / "show_floorplan.py",
        ["--from-file", str(hetero_yaml), "--overlay", "tile-class"],
    )
    assert rc == 0, err
    assert "Compute sites by tile class (8x8 sites, auto placement)" in out
    assert "systolic_int8_ws" in out
    assert ".=spare" in out


def test_show_floorplan_overlay_says_when_there_is_no_checkerboard(cli_runner):
    """A uniform SKU has no explicit site grid; the CLI says so rather than
    rendering an empty or invented one."""
    rc, out, err = cli_runner(
        _CLI / "show_floorplan.py", [_LEGACY, "--overlay", "power-domain"]
    )
    assert rc == 0, err
    assert "no explicit checkerboard" in out


def test_show_floorplan_rejects_two_sources(cli_runner, hetero_yaml):
    rc, _, err = cli_runner(
        _CLI / "show_floorplan.py", [_LEGACY, "--from-file", str(hetero_yaml)]
    )
    assert rc != 0 and "not both" in err


@pytest.mark.parametrize("extra, expected", [
    (["--view", "circuit"], "--view architectural"),
    (["--json"], "only rendered in text output"),
    (["--output", "fp.json"], "only rendered in text output"),
])
def test_show_floorplan_rejects_an_overlay_it_cannot_render(
    cli_runner, hetero_yaml, extra, expected
):
    """--overlay is a text panel under the architectural view. Asking for
    it anywhere else used to succeed and silently drop it."""
    rc, _, err = cli_runner(
        _CLI / "show_floorplan.py",
        ["--from-file", str(hetero_yaml), "--overlay", "tile-class", *extra],
    )
    assert rc != 0
    assert expected in err


# ---------------------------------------------------------------------------
# analyze_kpu_tile_ladder (graphs#268 E2)
# ---------------------------------------------------------------------------


def test_tile_ladder_lists_its_functions(cli_runner):
    rc, out, err = cli_runner(_CLI / "analyze_kpu_tile_ladder.py", ["--list-functions"])
    assert rc == 0, err
    assert "stereo.sgm" in out and "gemm.int8" in out
    assert "INTERPOLATED" in out  # the SGM ops model is derived, not assumed


def test_tile_ladder_reports_every_rung_with_provenance(cli_runner, hetero_yaml):
    rc, out, err = cli_runner(
        _CLI / "analyze_kpu_tile_ladder.py",
        ["--from-file", str(hetero_yaml), "--function", "gemm.int8", "--show-ops"],
    )
    assert rc == 0, err
    assert "Traceback" not in err
    assert "systolic" in out and "pe_fabric" in out
    assert "gpu" in out and "cpu" in out
    assert "from:" in out          # per-rung provenance
    assert "derivation:" in out    # the ops model shows its arithmetic
    assert "Specialization order" in out


def test_tile_ladder_refuses_a_function_it_cannot_price(cli_runner, hetero_yaml):
    rc, _, err = cli_runner(
        _CLI / "analyze_kpu_tile_ladder.py",
        ["--from-file", str(hetero_yaml), "--function", "not.a.function"],
    )
    assert rc == 2
    assert "no ops model" in err and "Traceback" not in err


def test_tile_ladder_json_carries_the_model_and_the_caveats(cli_runner, hetero_yaml, tmp_path):
    out_path = tmp_path / "ladder.json"
    rc, _, err = cli_runner(
        _CLI / "analyze_kpu_tile_ladder.py",
        ["--from-file", str(hetero_yaml), "--output", str(out_path)],
    )
    assert rc == 0, err
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    by_function = {entry["function_id"]: entry for entry in payload}
    assert "stereo.sgm" in by_function
    vio = by_function["vio.stereo_inertial"]
    # The caveats travel with the numbers, not only in the text view.
    assert vio["arithmetic_bound"] is False
    assert vio["back_check"]
    assert by_function["stereo.sgm"]["ops_model"]["confidence"] == "INTERPOLATED"
    assert all(r["provenance"] for r in by_function["gemm.int8"]["rungs"])


# ---------------------------------------------------------------------------
# list_kpu_engines (graphs#268 E3)
# ---------------------------------------------------------------------------


def test_engine_registry_lists_engines_and_segments(cli_runner, hetero_yaml):
    rc, out, err = cli_runner(
        _CLI / "list_kpu_engines.py", ["--from-file", str(hetero_yaml)]
    )
    assert rc == 0, err
    assert "Traceback" not in err
    assert "=== Engines (7) ===" in out
    assert "=== Segments (1) ===" in out
    assert "isp_sgm_vio" in out and "stays on chip" in out
    assert "kernels: gemm, conv2d" in out          # the systolic engine
    assert "not precision-keyed" in out            # the min-plus engine


def test_engine_registry_exits_nonzero_on_an_unservable_requirement(
    cli_runner, hetero_yaml
):
    """A precision the die cannot carry is a hard finding, so the CLI is
    usable as a gate."""
    rc, out, err = cli_runner(
        _CLI / "list_kpu_engines.py",
        ["--from-file", str(hetero_yaml), "--require", "planning_qp=fp32"],
    )
    assert rc == 1
    assert "ERROR" in out and "no programmable engine" in out


def test_engine_registry_exits_zero_when_every_requirement_is_served(
    cli_runner, hetero_yaml
):
    rc, out, err = cli_runner(
        _CLI / "list_kpu_engines.py",
        ["--from-file", str(hetero_yaml), "--require", "trunk=int8"],
    )
    assert rc == 0, err
    assert "OK" in out and "served by" in out


def test_engine_registry_rejects_a_malformed_requirement(cli_runner, hetero_yaml):
    rc, _, err = cli_runner(
        _CLI / "list_kpu_engines.py",
        ["--from-file", str(hetero_yaml), "--require", "no_equals_sign"],
    )
    assert rc != 0
    assert "NAME=FORMAT" in err and "Traceback" not in err


def test_engine_registry_json_is_a_registry(cli_runner, hetero_yaml, tmp_path):
    out_path = tmp_path / "engines.json"
    rc, _, err = cli_runner(
        _CLI / "list_kpu_engines.py",
        ["--from-file", str(hetero_yaml), "--output", str(out_path),
         "--require", "planning_qp=fp32"],
    )
    assert rc == 1, err  # the finding still sets the exit code
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert {"engines", "segments", "precision_findings"} <= set(payload)
    by_id = {e["engine_id"]: e for e in payload["engines"]}
    assert len(by_id) == 7
    assert by_id["systolic_int8_ws"]["supported_kernels"] == ["gemm", "conv2d"]
    assert by_id["ff_stereo_sgm"]["function_id"] == "stereo.sgm"
    assert by_id["pe_int8_mac_i32"]["precision_floor"] == "int4"
    assert payload["segments"][0]["absorbed_bytes_by_unit"] == {"pixel": 7.0}
    assert payload["precision_findings"][0]["severity"] == "ERROR"


def test_engine_registry_works_on_a_uniform_sku(cli_runner):
    """The SoC study composes a uniform KPU as engines too."""
    rc, out, err = cli_runner(_CLI / "list_kpu_engines.py", [_LEGACY])
    assert rc == 0, err
    assert "=== Engines (3) ===" in out
    assert "no stream-linked chains" in out


def test_engine_registry_csv_carries_all_three_sections(cli_runner, hetero_yaml, tmp_path):
    """main() writes the file before returning a non-zero exit code, so a
    CSV that held only the engines could not explain the failure
    (CodeRabbit on #290)."""
    import csv as _csv

    out_path = tmp_path / "engines.csv"
    rc, _, err = cli_runner(
        _CLI / "list_kpu_engines.py",
        ["--from-file", str(hetero_yaml), "--output", str(out_path),
         "--require", "planning_qp=fp32"],
    )
    assert rc == 1, err
    rows = list(_csv.DictReader(out_path.read_text(encoding="utf-8").splitlines()))
    by_type = {}
    for row in rows:
        by_type.setdefault(row["record_type"], []).append(row)
    assert set(by_type) == {"engine", "segment", "precision_finding"}
    assert by_type["segment"][0]["segment_id"] == "isp_sgm_vio"
    assert "pixel=7" in by_type["segment"][0]["absorbed_bytes_by_unit"]
    finding = by_type["precision_finding"][0]
    assert finding["severity"] == "ERROR"
    assert finding["requirement"] == "planning_qp"


def test_engine_registry_survives_a_core_with_no_retargetable_energy(
    cli_runner, tmp_path, hetero_yaml
):
    """fixed_function_pj_per_unit returns None when the core's reference
    node is not in the catalog. Formatting that as a float crashed the
    default text view (CodeRabbit on #290)."""
    data = yaml.safe_load(hetero_yaml.read_text(encoding="utf-8"))
    for tile in data["dies"][0]["blocks"][0]["tiles"]:
        if tile.get("core"):
            tile["core"]["energy"]["ref_node_id"] = "not_a_real_node"
    path = tmp_path / "orphan_ref.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    rc, out, err = cli_runner(_CLI / "list_kpu_engines.py", ["--from-file", str(path)])
    assert rc == 0, err
    assert "Traceback" not in err
    assert "energy not retargetable" in out
