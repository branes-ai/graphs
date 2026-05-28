"""Tests for cli/compare_operator_roofline.py.

Pins the demonstration's core claims: dot/matvec are memory-bound and matmul is
compute-bound on every default architecture, the process node is populated from
the unified physical_spec, and latency/energy/efficiency are produced.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "cli" / "analyze_operator_roofline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_operator_roofline", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


@pytest.mark.parametrize(
    "label,name,precision", mod.DEFAULT_ARCHES,
    ids=[a[0] for a in mod.DEFAULT_ARCHES],
)
def test_arch_evaluation_structure_and_bottlenecks(label, name, precision):
    a = mod.evaluate_arch(label, name, precision)
    # Speed of light is real.
    assert a["peak_compute_tops"] > 0
    assert a["peak_bw_gbps"] > 0
    # Process node comes from the unified ComputeProduct physical_spec.
    assert a["node"] != "(unknown)"
    by = {r["op"]: r for r in a["rows"]}
    assert set(by) == {"dot", "matvec", "matmul"}
    # The headline story: dot/matvec memory-bound, matmul compute-bound.
    assert by["dot"]["bottleneck"] == "BANDWIDTH"
    assert by["matvec"]["bottleneck"] == "BANDWIDTH"
    assert by["matmul"]["bottleneck"] == "COMPUTE"
    for r in a["rows"]:
        assert r["latency_s"] > 0
        assert r["energy_j"] > 0
        assert r["mem_eff"] >= 0 and r["compute_eff"] >= 0


def test_memory_bound_ops_lean_on_bandwidth_compute_on_flops():
    """On the CPU default, the binding resource's efficiency dominates: dot uses
    bandwidth (mem_eff > compute_eff); matmul uses compute (compute_eff > mem_eff)."""
    a = mod.evaluate_arch(*mod.DEFAULT_ARCHES[0])  # CPU / i7
    by = {r["op"]: r for r in a["rows"]}
    assert by["dot"]["mem_eff"] > by["dot"]["compute_eff"]
    assert by["matvec"]["mem_eff"] > by["matvec"]["compute_eff"]
    assert by["matmul"]["compute_eff"] > by["matmul"]["mem_eff"]


def test_cli_main_runs_and_prints_all_three(capsys):
    rc = mod.main([])
    assert rc == 0
    out = capsys.readouterr().out
    for token in ("CPU:", "GPU:", "KPU:", "process node:", "dot", "matvec", "matmul"):
        assert token in out


def test_output_format_autodetected_by_extension(tmp_path):
    """--output emits JSON / CSV / Markdown / text by file extension (the
    CLI-tool --output contract)."""
    import csv as _csv
    import json as _json

    # JSON: structured, parseable, 3 arches each with 3 operator rows.
    jp = tmp_path / "r.json"
    mod.main(["--output", str(jp)])
    data = _json.loads(jp.read_text())
    assert len(data) == 3
    assert all(len(a["rows"]) == 3 for a in data)
    assert {a["label"] for a in data} == {"CPU", "GPU", "KPU"}

    # CSV: header + one row per (arch, operator) = 9 data rows.
    cp = tmp_path / "r.csv"
    mod.main(["--output", str(cp)])
    rows = list(_csv.reader(cp.read_text().splitlines()))
    assert rows[0] == mod._FLAT_COLS
    assert len(rows) == 1 + 9

    # Markdown: per-arch tables.
    mp = tmp_path / "r.md"
    mod.main(["--output", str(mp)])
    md = mp.read_text()
    assert "| operator | bottleneck |" in md and "### CPU" in md

    # Text fallback: the plain ASCII tables.
    tp = tmp_path / "r.txt"
    mod.main(["--output", str(tp)])
    assert "speed of light" in tp.read_text()
