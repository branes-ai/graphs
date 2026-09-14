"""Tests for cli/kpu_golden_snapshot.py: exit codes and --output formats."""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

from graphs.hardware import kpu_golden as kg

_SCRIPT = Path(__file__).resolve().parents[2] / "cli" / "kpu_golden_snapshot.py"
_SKU = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"


@pytest.fixture
def perturbed_golden_dir(tmp_path):
    """A copy of the goldens with one value changed for _SKU."""
    shutil.copytree(kg.DEFAULT_GOLDEN_DIR, tmp_path / "golden")
    golden_dir = tmp_path / "golden"
    snap = kg.load_snapshot(_SKU, golden_dir)
    snap["generator"]["die_size_mm2"] *= 1.05
    kg.write_snapshot(snap, golden_dir)
    return golden_dir


def test_check_passes_on_real_goldens(cli_runner):
    rc, out, _ = cli_runner(_SCRIPT, ["--sku", _SKU])
    assert rc == 0
    assert "PASS" in out


def test_check_fails_on_diff(cli_runner, perturbed_golden_dir):
    rc, out, _ = cli_runner(_SCRIPT, ["--sku", _SKU, "--golden-dir", str(perturbed_golden_dir)])
    assert rc == 1
    assert ".generator.die_size_mm2" in out


def test_unknown_sku_is_argument_error(cli_runner):
    rc, _, err = cli_runner(_SCRIPT, ["--sku", "not_a_sku"])
    assert rc == 2
    assert "unknown SKU" in err


@pytest.mark.parametrize("ext", ["json", "csv", "md", "txt", "unknown"])
def test_output_format_detection(cli_runner, perturbed_golden_dir, tmp_path, ext):
    out_path = tmp_path / f"report.{ext}"
    rc, _, _ = cli_runner(
        _SCRIPT,
        ["--sku", _SKU, "--golden-dir", str(perturbed_golden_dir), "--output", str(out_path)],
    )
    assert rc == 1
    text = out_path.read_text()
    if ext == "json":
        payload = json.loads(text)
        assert payload["stale"] == []
        assert any(".generator.die_size_mm2" in d for d in payload["results"][_SKU])
    elif ext == "csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows and set(rows[0]) == {"sku_id", "status", "difference"}
        assert all(r["sku_id"] == _SKU and r["status"] == "DIFF" for r in rows)
        assert any(".generator.die_size_mm2" in r["difference"] for r in rows)
    elif ext == "md":
        assert text.startswith("# KPU golden snapshot check")
        assert f"| `{_SKU}` | DIFF |" in text
    else:  # .txt and unrecognized extensions fall back to text
        assert text.startswith("=== KPU golden snapshot check")
        assert "FAIL" in text
