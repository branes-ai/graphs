"""Phase 7 PDK ingestion tests.

Covers:
  * ``cli/import_pdk.py`` happy paths + error paths
  * ``PROCESS_NODE_DATA_DIR`` overlay end-to-end through the graphs
    consumers (validator framework, KPU YAML loader)

The overlay path is implemented in ``embodied_schemas.load_process_nodes``
(Phase 7 PR in embodied-schemas); these tests live in graphs because
that's where the downstream consumers (sku_validators, kpu_yaml_loader)
exercise it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from embodied_schemas import load_process_nodes


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORT_PDK = REPO_ROOT / "cli" / "import_pdk.py"


# ---------------------------------------------------------------------------
# import_pdk.py -- happy paths
# ---------------------------------------------------------------------------

def _minimal_pdk_summary() -> dict:
    """A valid ProcessNodeEntry payload for the import tool to validate."""
    return {
        "id": "test_node_pdk",
        "foundry": "tsmc",
        "node_name": "N16-test",
        "node_nm": 16,
        "transistor_topology": "finfet",
        "nominal_vdd_v": 0.8,
        "densities": {
            "balanced_logic": {
                "circuit_class": "balanced_logic",
                "mtx_per_mm2": 30.5,
                "library_name": "test-lib",
                "confidence": "calibrated",
                "source": "test PDK",
            }
        },
        "last_updated": "2026-05-10",
    }


def _run_cli(args: list[str], cwd: Path = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(IMPORT_PDK), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True, text=True, env={**os.environ},
    )


def test_import_pdk_yaml_input_writes_calibrated_output(tmp_path: Path):
    """Default flow: YAML in, YAML out, confidence forced to calibrated,
    source set to the --source argument."""
    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump(_minimal_pdk_summary()))
    out = tmp_path / "out.yaml"

    result = _run_cli([
        "--input", str(summary),
        "--source", "TSMC N16FF+ PDK rev 2024Q1",
        "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert out.exists()
    written = yaml.safe_load(out.read_text())
    assert written["confidence"] == "calibrated"
    assert written["source"] == "TSMC N16FF+ PDK rev 2024Q1"
    assert written["id"] == "test_node_pdk"


def test_import_pdk_json_input_supported(tmp_path: Path):
    """JSON inputs auto-detected by extension."""
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_minimal_pdk_summary()))
    out = tmp_path / "out.yaml"

    result = _run_cli([
        "--input", str(summary),
        "--source", "JSON-source",
        "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(out.read_text())["source"] == "JSON-source"


def test_import_pdk_overrides_confidence_to_calibrated(tmp_path: Path):
    """Even if the input says confidence=theoretical, the tool stamps
    calibrated by default. PDK-derived data shouldn't masquerade as
    theoretical after ingestion."""
    payload = _minimal_pdk_summary()
    payload["confidence"] = "theoretical"  # try to slip through
    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump(payload))
    out = tmp_path / "out.yaml"

    result = _run_cli([
        "--input", str(summary),
        "--source", "PDK rev X",
        "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(out.read_text())["confidence"] == "calibrated"


def test_import_pdk_explicit_interpolated_confidence(tmp_path: Path):
    """Allow explicit override to interpolated (PDK-extrapolated values)."""
    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump(_minimal_pdk_summary()))
    out = tmp_path / "out.yaml"

    result = _run_cli([
        "--input", str(summary),
        "--source", "extrapolated between PDK points",
        "--confidence", "interpolated",
        "--output", str(out),
    ])
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(out.read_text())["confidence"] == "interpolated"


def test_import_pdk_uses_process_node_data_dir_when_no_output(tmp_path: Path):
    """If --output is omitted and PROCESS_NODE_DATA_DIR is set, write to
    that dir using <id>.yaml. Lets a CI / workflow drop PDK YAMLs into
    the overlay path without computing output paths by hand."""
    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump(_minimal_pdk_summary()))
    overlay = tmp_path / "overlay"
    overlay.mkdir()

    env = {**os.environ, "PROCESS_NODE_DATA_DIR": str(overlay)}
    result = subprocess.run(
        [sys.executable, str(IMPORT_PDK),
         "--input", str(summary), "--source", "auto-place test"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    expected = overlay / "test_node_pdk.yaml"
    assert expected.exists()


# ---------------------------------------------------------------------------
# import_pdk.py -- error paths
# ---------------------------------------------------------------------------

def test_import_pdk_invalid_input_fails_validation(tmp_path: Path):
    """Input missing required ProcessNodeEntry fields must fail with
    exit=1 (validation error), not silently produce a half-baked YAML."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"id": "no_required_fields"}))
    out = tmp_path / "out.yaml"

    result = _run_cli([
        "--input", str(bad), "--source", "x", "--output", str(out),
    ])
    assert result.returncode == 1
    assert "does not validate" in result.stderr
    assert not out.exists()


def test_import_pdk_existing_output_requires_force(tmp_path: Path):
    """Refuse to overwrite without --force."""
    summary = tmp_path / "summary.yaml"
    summary.write_text(yaml.safe_dump(_minimal_pdk_summary()))
    out = tmp_path / "out.yaml"
    out.write_text("existing")

    result = _run_cli([
        "--input", str(summary), "--source", "x", "--output", str(out),
    ])
    assert result.returncode == 2
    assert "already exists" in result.stderr
    # Original content preserved
    assert out.read_text() == "existing"

    # With --force, overwrite is allowed
    result2 = _run_cli([
        "--input", str(summary), "--source", "x",
        "--output", str(out), "--force",
    ])
    assert result2.returncode == 0
    assert out.read_text() != "existing"


def test_import_pdk_missing_input_fails_cleanly(tmp_path: Path):
    """Non-existent input path returns exit=2 with a clear message."""
    result = _run_cli([
        "--input", str(tmp_path / "nope.yaml"),
        "--source", "x",
        "--output", str(tmp_path / "out.yaml"),
    ])
    assert result.returncode == 2
    assert "input not found" in result.stderr


# ---------------------------------------------------------------------------
# PROCESS_NODE_DATA_DIR overlay end-to-end
# ---------------------------------------------------------------------------

def test_overlay_calibrated_supersedes_theoretical_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A CALIBRATED overlay entry for an id that already exists in the
    public catalog as THEORETICAL must win. The downstream graphs
    consumers (validator framework, KPU YAML loader) then see the
    sharper PDK-derived values.
    """
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    cal_yaml = overlay / "tsmc_n16_calibrated.yaml"
    cal_yaml.write_text(yaml.safe_dump({
        "id": "tsmc_n16",
        "foundry": "tsmc",
        "node_name": "N16",
        "node_nm": 16,
        "transistor_topology": "finfet",
        "nominal_vdd_v": 0.8,
        "densities": {
            "balanced_logic": {
                "circuit_class": "balanced_logic",
                "mtx_per_mm2": 32.0,  # PDK-measured (vs 28.5 baseline)
                "confidence": "calibrated",
                "source": "PDK rev 2024Q1",
            }
        },
        "source": "PDK rev 2024Q1 (private overlay)",
        "confidence": "calibrated",
        "last_updated": "2026-05-10",
    }))

    monkeypatch.setenv("PROCESS_NODE_DATA_DIR", str(overlay))
    nodes = load_process_nodes()
    n16 = nodes["tsmc_n16"]
    assert n16.confidence.value == "calibrated"
    assert n16.densities[
        next(c for c in n16.densities if c.value == "balanced_logic")
    ].mtx_per_mm2 == 32.0


def test_overlay_theoretical_does_not_downgrade_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A THEORETICAL overlay entry must NOT supersede an existing
    THEORETICAL baseline (or higher). Same-rank collisions keep the
    existing entry; lower-rank overlays are ignored.
    """
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "stale.yaml").write_text(yaml.safe_dump({
        "id": "tsmc_n16",
        "foundry": "tsmc",
        "node_name": "N16",
        "node_nm": 16,
        "transistor_topology": "finfet",
        "nominal_vdd_v": 0.8,
        "densities": {
            "balanced_logic": {
                "circuit_class": "balanced_logic",
                "mtx_per_mm2": 999.0,  # nonsense override that must be ignored
                "confidence": "theoretical",
                "source": "stale overlay",
            }
        },
        "source": "stale theoretical overlay",
        "confidence": "theoretical",
        "last_updated": "2026-05-10",
    }))

    monkeypatch.setenv("PROCESS_NODE_DATA_DIR", str(overlay))
    nodes = load_process_nodes()
    n16 = nodes["tsmc_n16"]
    # Density is the public catalog's value, NOT the overlay's 999
    bl = n16.densities[
        next(c for c in n16.densities if c.value == "balanced_logic")
    ]
    assert bl.mtx_per_mm2 < 999.0
    assert n16.source != "stale theoretical overlay"


def test_overlay_adds_new_node_without_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A new id (not in public catalog) should be added to the result."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "future.yaml").write_text(yaml.safe_dump({
        "id": "tsmc_n2_future",
        "foundry": "tsmc",
        "node_name": "N2-future",
        "node_nm": 2,
        "transistor_topology": "gaa",
        "nominal_vdd_v": 0.6,
        "densities": {
            "balanced_logic": {
                "circuit_class": "balanced_logic",
                "mtx_per_mm2": 320.0,
                "confidence": "calibrated",
                "source": "private",
            }
        },
        "source": "private",
        "confidence": "calibrated",
        "last_updated": "2026-05-10",
    }))

    monkeypatch.setenv("PROCESS_NODE_DATA_DIR", str(overlay))
    nodes = load_process_nodes()
    assert "tsmc_n2_future" in nodes
    assert nodes["tsmc_n2_future"].confidence.value == "calibrated"


def test_overlay_unset_uses_public_catalog_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without the env var set, the loader behaves exactly as before."""
    monkeypatch.delenv("PROCESS_NODE_DATA_DIR", raising=False)
    nodes = load_process_nodes()
    assert "tsmc_n16" in nodes
    assert nodes["tsmc_n16"].confidence.value == "theoretical"


def test_overlay_pointing_at_nonexistent_dir_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    """A bogus PROCESS_NODE_DATA_DIR must warn and continue, not crash."""
    monkeypatch.setenv("PROCESS_NODE_DATA_DIR", "/nonexistent/path/xyz")
    # Should still return the public catalog
    nodes = load_process_nodes()
    assert "tsmc_n16" in nodes
