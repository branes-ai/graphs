"""cli/list_compute_products.py lists every ComputeProduct in the catalog,
of every block kind, with filters, sorting and the standard output formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from graphs.hardware.compute_product_loader import load_compute_products_unified
from graphs.hardware.kpu_access import has_kpu_block

_CLI = Path(__file__).resolve().parents[2] / "cli" / "list_compute_products.py"
_PRODUCTS = load_compute_products_unified()


def _json(cli_runner, *args) -> list[dict]:
    rc, out, err = cli_runner(_CLI, ["--format", "json", *args])
    assert rc == 0, err
    return json.loads(out)


def test_lists_every_product(cli_runner):
    rc, out, err = cli_runner(_CLI, [])
    assert rc == 0, err
    assert f"{len(_PRODUCTS)} compute product(s)" in out
    assert all(pid in out for pid in _PRODUCTS)
    assert "MISS" not in out  # every die's process node resolves

    rows = _json(cli_runner)
    assert sorted(r["id"] for r in rows) == sorted(_PRODUCTS)
    assert all(r["nodes_resolve"] for r in rows)


def test_json_fields_follow_the_catalog(cli_runner):
    rows = {r["id"]: r for r in _json(cli_runner)}
    epyc = rows["amd_epyc_9654_sp5"]
    cp = _PRODUCTS["amd_epyc_9654_sp5"]
    assert epyc["dies"] == len(cp.dies) == 2
    assert epyc["block_kinds"] == "cpu+io"
    assert epyc["process_nodes"] == "tsmc_n5,tsmc_n6"
    assert epyc["die_area_mm2"] == sum(d.die_size_mm2 for d in cp.dies)
    t64 = rows["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    kpu = _PRODUCTS["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    assert t64["int8_tops"] == kpu.performance.int8_tops
    assert t64["int8_tops_per_watt"] == kpu.performance.int8_tops / kpu.power.tdp_watts


def test_kind_filter(cli_runner):
    kpu = {r["id"] for r in _json(cli_runner, "--kind", "kpu")}
    assert kpu == {k for k, v in _PRODUCTS.items() if has_kpu_block(v)}
    cpu = {r["id"] for r in _json(cli_runner, "--kind", "cpu")}
    assert "amd_epyc_9654_sp5" in cpu  # a cpu+io product matches --kind cpu
    io_only = {r["id"] for r in _json(cli_runner, "--kind", "io")}
    assert io_only < cpu
    rc, _, err = cli_runner(_CLI, ["--kind", "fpga"])
    assert rc != 0 and "invalid choice" in err


def test_vendor_filter_and_sort(cli_runner):
    google = _json(cli_runner, "--vendor", "GOOGLE", "--sort", "tops")
    assert google and all(r["vendor"] == "google" for r in google)
    tops = [r["int8_tops"] for r in google]
    assert tops == sorted(tops, reverse=True)
    eff = [r["int8_tops_per_watt"] for r in _json(cli_runner, "--sort", "tops_per_watt")]
    assert eff == sorted(eff, reverse=True)
    rc, out, _ = cli_runner(_CLI, ["--vendor", "nobody"])
    assert rc == 0 and "(no products match)" in out


def test_output_formats_by_extension(cli_runner, tmp_path):
    for name in ("p.csv", "p.md", "p.json", "p.txt"):
        rc, _, err = cli_runner(_CLI, ["--kind", "kpu", "--output", str(tmp_path / name)])
        assert rc == 0, err
    rows = list(csv.DictReader(io.StringIO((tmp_path / "p.csv").read_text())))
    assert len(rows) == 12 and rows[0]["block_kinds"] == "kpu"
    assert (tmp_path / "p.md").read_text().startswith("| id | vendor |")
    assert len(json.loads((tmp_path / "p.json").read_text())) == 12
    assert "12 compute product(s): kpu 12" in (tmp_path / "p.txt").read_text()
    # --format wins over the extension.
    rc, _, _ = cli_runner(
        _CLI, ["--kind", "kpu", "--format", "json", "--output", str(tmp_path / "x.md")]
    )
    assert rc == 0 and json.loads((tmp_path / "x.md").read_text())


def test_text_and_markdown_show_lifecycle_and_node_check(cli_runner, tmp_path):
    rc, out, err = cli_runner(_CLI, ["--kind", "kpu"])
    assert rc == 0, err
    header, _, first = out.splitlines()[:3]
    assert "lifecycle" in header and "refs" in header
    assert " production " in first and first.rstrip().endswith("ok")

    md = tmp_path / "p.md"
    rc, _, err = cli_runner(_CLI, ["--kind", "kpu", "--output", str(md)])
    assert rc == 0, err
    lines = md.read_text().splitlines()
    assert lines[0].endswith("| market | lifecycle | confidence | nodes resolve |")
    assert lines[2].endswith("| production | theoretical | yes |")


def test_unwritable_output_is_a_clean_error(cli_runner, tmp_path):
    rc, out, err = cli_runner(_CLI, ["--output", str(tmp_path)])  # a directory
    assert rc == 1
    assert "error: cannot write" in err and "Traceback" not in err
