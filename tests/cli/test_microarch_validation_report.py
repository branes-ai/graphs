"""Smoke tests for cli/microarch_validation_report.py.

M0 shipped empty panels; M1 populates Layer 1 (ALU); M2 populates
Layer 2 (Register File); M3 populates Layer 3 (L1 cache / scratchpad).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli_main():
    """Import the CLI main() by file path (cli/ is not a package)."""
    import importlib.util
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "cli" / "microarch_validation_report.py"
    spec = importlib.util.spec_from_file_location("microarch_cli", path)
    module = importlib.util.module_from_spec(spec)
    # Ensure repo root on sys.path for module's own imports.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec.loader.exec_module(module)
    return module


def test_json_bundle_emits_one_file_per_sku(tmp_path: Path, cli_main):
    rc = cli_main.main([
        "--hardware", "jetson_orin_agx_64gb", "kpu_t128",
        "--format", "json",
        "--output", str(tmp_path),
    ])
    assert rc == 0
    jetson = tmp_path / "data" / "jetson_orin_agx_64gb.json"
    kpu = tmp_path / "data" / "kpu_t128.json"
    assert jetson.exists()
    assert kpu.exists()
    # Verify JSON shape
    payload = json.loads(jetson.read_text())
    assert payload["sku"] == "jetson_orin_agx_64gb"
    assert payload["archetype"] == "simt"
    assert len(payload["layers"]) == 7
    layer_tags = [p["layer"] for p in payload["layers"]]
    expected = [
        "alu", "register", "l1_cache", "l2_cache", "l3_cache",
        "soc_data_movement", "external_memory",
    ]
    assert layer_tags == expected
    # M3: layers 1+2+3 populated; layers 4-7 remain not_populated.
    layer_status = {p["layer"]: p["status"] for p in payload["layers"]}
    assert layer_status["alu"] != "not_populated", (
        f"Layer 1 should be populated at M1, got {layer_status['alu']}"
    )
    assert layer_status["register"] != "not_populated", (
        f"Layer 2 should be populated at M2, got {layer_status['register']}"
    )
    assert layer_status["l1_cache"] != "not_populated", (
        f"Layer 3 should be populated at M3, got {layer_status['l1_cache']}"
    )
    for tag in ("l2_cache", "l3_cache",
                "soc_data_movement", "external_memory"):
        assert layer_status[tag] == "not_populated"


def test_html_bundle_writes_index_hardware_compare(tmp_path: Path, cli_main):
    rc = cli_main.main([
        "--hardware", "jetson_orin_agx_64gb",
        "--format", "html",
        "--output", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "hardware" / "jetson_orin_agx_64gb.html").exists()
    assert (tmp_path / "compare.html").exists()


def test_html_has_branes_branding_and_placeholder(tmp_path: Path, cli_main):
    cli_main.main([
        "--hardware", "kpu_t128",
        "--format", "html",
        "--output", str(tmp_path),
    ])
    sku_html = (tmp_path / "hardware" / "kpu_t128.html").read_text()
    # Branes logo embedded as data URI (or text fallback)
    assert "Branes" in sku_html or "branes" in sku_html.lower()
    # Confidence-ladder legend present
    assert "CALIBRATED" in sku_html
    assert "INTERPOLATED" in sku_html
    assert "THEORETICAL" in sku_html
    # M0 placeholder text
    assert "NOT YET POPULATED" in sku_html
    # All seven layer titles present
    for expected_title in [
        "Layer 1: ALU",
        "Layer 2: Register File",
        "Layer 3: L1 Cache",
        "Layer 4: L2 Cache",
        "Layer 5: L3",
        "Layer 6: SoC Data Movement",
        "Layer 7: External Memory",
    ]:
        assert expected_title in sku_html


def test_default_hardware_list_covers_locked_skus(cli_main):
    """Sanity: the CLI default list matches the locked SKU set."""
    expected = {
        "jetson_orin_agx_64gb",
        "intel_core_i7_12700k",
        "ryzen_9_8945hs",
        "kpu_t64", "kpu_t128", "kpu_t256",
        "coral_edge_tpu",
        "hailo8", "hailo10h",
    }
    assert set(cli_main.DEFAULT_SKU_LIST) == expected


def test_pptx_export_produces_deck(tmp_path: Path, cli_main):
    pptx = pytest.importorskip("pptx")  # skip if python-pptx missing
    rc = cli_main.main([
        "--hardware", "jetson_orin_agx_64gb",
        "--format", "pptx",
        "--output", str(tmp_path),
    ])
    assert rc == 0
    deck = tmp_path / "microarch_deck.pptx"
    assert deck.exists()
    # Validate it opens as a proper pptx
    prs = pptx.Presentation(str(deck))
    assert len(prs.slides) >= 2  # title + at least one SKU slide
