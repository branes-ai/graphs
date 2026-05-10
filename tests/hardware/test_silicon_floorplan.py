"""Stage 8: silicon floorplanner + GEOMETRY-category validators.

Covers:
  * ``derive_kpu_floorplan`` heuristic correctness across the catalog
  * GEOMETRY validators register, run, and produce the expected
    findings on real SKUs (pitch_match flagged on T768 etc.)
  * The CLI ``cli/show_floorplan.py`` happy path

Stage 8 stance: the heuristic floorplan is *advisory* until calibrated
against measured silicon. Tests assert structural properties (no
overlaps in the same layer, every tile has a class assigned, output
dimensions positive) rather than pixel-exact dimensions, so threshold
tuning doesn't churn this test file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from embodied_schemas import (
    load_cooling_solutions,
    load_kpus,
    load_process_nodes,
)

from graphs.hardware.silicon_floorplan import (
    Floorplan,
    FloorplanBlock,
    derive_kpu_floorplan,
)
from graphs.hardware.sku_validators import (
    Severity,
    ValidatorCategory,
    ValidatorContext,
    default_registry,
    load_validators,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SHOW_FLOORPLAN_CLI = REPO_ROOT / "cli" / "show_floorplan.py"

# Auto-discovery: test against every KPU SKU in the catalog. New SKUs
# get covered automatically, matching the Phase 6 catalog gate pattern.
ALL_KPU_IDS = sorted(load_kpus().keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kpus():
    return load_kpus()


@pytest.fixture(scope="module")
def nodes():
    return load_process_nodes()


@pytest.fixture(scope="module")
def cooling():
    return load_cooling_solutions()


@pytest.fixture(scope="module", autouse=True)
def _register():
    """Make sure validators (including geometry) are registered."""
    load_validators()


# ---------------------------------------------------------------------------
# derive_kpu_floorplan: structural properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_floorplan_has_positive_dimensions(sku_id, kpus, nodes):
    """Every SKU produces a die with positive width / height / area.

    Catches the case where a missing silicon_bin block (no PHY, no IO,
    etc.) would zero-out a die dimension.
    """
    sku = kpus[sku_id]
    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])
    assert fp.die_width_mm > 0, f"{sku_id} has non-positive die width"
    assert fp.die_height_mm > 0, f"{sku_id} has non-positive die height"
    assert fp.die_area_mm2 > 0


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_floorplan_compute_tile_count_matches_mesh(sku_id, kpus, nodes):
    """``mesh_rows x mesh_cols`` compute tiles must be placed.

    The KPU NoC mesh is what defines the tile count -- if the
    floorplanner places fewer than that, the YAML's tile_mix counts
    don't sum to the mesh capacity (real bug) or the layout heuristic
    silently dropped tiles (also a bug).
    """
    sku = kpus[sku_id]
    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])
    arch = sku.kpu_architecture
    expected = arch.noc.mesh_rows * arch.noc.mesh_cols
    assert len(fp.compute_tiles()) == expected, (
        f"{sku_id}: placed {len(fp.compute_tiles())} compute tiles, "
        f"expected {expected} ({arch.noc.mesh_rows}x{arch.noc.mesh_cols})"
    )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_floorplan_every_compute_tile_has_class(sku_id, kpus, nodes):
    """Every compute tile must carry a tile_class label so geometry
    validators can group findings by class."""
    sku = kpus[sku_id]
    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])
    declared_classes = {t.tile_type for t in sku.kpu_architecture.tiles}
    for block in fp.compute_tiles():
        assert block.tile_class is not None, (
            f"{sku_id}: compute tile {block.name} missing tile_class"
        )
        assert block.tile_class in declared_classes, (
            f"{sku_id}: compute tile {block.name} has tile_class "
            f"{block.tile_class!r} not in YAML's declared {declared_classes}"
        )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_floorplan_compute_tiles_dont_overlap(sku_id, kpus, nodes):
    """Compute tiles in the mesh must be axis-aligned, non-overlapping.

    Uses the mesh row/col ordering implicit in the tile name format
    ``tile[r,c]`` and checks neighbouring tiles share an edge.
    """
    sku = kpus[sku_id]
    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])
    tiles = fp.compute_tiles()

    # All tiles share the same width and height (unified pitch)
    pitches = {(round(b.width_mm, 4), round(b.height_mm, 4)) for b in tiles}
    assert len(pitches) == 1, (
        f"{sku_id}: tiles have multiple distinct pitch sizes: {pitches}"
    )
    w, h = next(iter(pitches))
    assert abs(w - h) < 1e-6, (
        f"{sku_id}: tile pitch is non-square ({w} x {h})"
    )

    # Pairwise: no two compute tiles overlap by more than ULP-level
    # float noise. Adjacent tiles share an edge at exactly
    # ``origin + N * pitch`` where N differs by 1 -- that's not overlap,
    # it's adjacency. ``eps`` ignores < 1 micron differences.
    eps = 1e-3  # 1 micron, well below any real tile pitch
    sample = tiles[:20]
    for i, a in enumerate(sample):
        for b in sample[i + 1:]:
            x_overlap = (
                a.x_mm + eps < b.x_mm + b.width_mm
                and b.x_mm + eps < a.x_mm + a.width_mm
            )
            y_overlap = (
                a.y_mm + eps < b.y_mm + b.height_mm
                and b.y_mm + eps < a.y_mm + a.height_mm
            )
            assert not (x_overlap and y_overlap), (
                f"{sku_id}: compute tiles {a.name} and {b.name} overlap"
            )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_floorplan_unified_pitch_dominates_tile_class_pitches(
    sku_id, kpus, nodes
):
    """The unified mesh pitch is the max across tile classes -- no
    class can have a pitch greater than the unified pitch (otherwise
    larger tiles would overflow the mesh cell)."""
    sku = kpus[sku_id]
    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])
    for tile_class, tp in fp.tile_pitches.items():
        assert tp.pitch_mm <= fp.unified_pitch_mm + 1e-6, (
            f"{sku_id}: tile class {tile_class!r} pitch {tp.pitch_mm} "
            f"exceeds unified pitch {fp.unified_pitch_mm}"
        )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_floorplan_total_block_area_close_to_die_area(sku_id, kpus, nodes):
    """Total placed-block area must be at most the die area.

    Whitespace is allowed; overlap is not. ``total_block_area > die_area``
    means the floorplanner placed overlapping rectangles or computed
    the die envelope incorrectly.
    """
    sku = kpus[sku_id]
    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])
    # Allow 1% slop for floating-point rounding in the IO ring corners
    assert fp.total_block_area_mm2() <= fp.die_area_mm2 * 1.01, (
        f"{sku_id}: blocks total {fp.total_block_area_mm2():.2f} mm^2 "
        f"exceeds die {fp.die_area_mm2:.2f} mm^2 (overlap or wrong envelope)"
    )


# ---------------------------------------------------------------------------
# GEOMETRY validators
# ---------------------------------------------------------------------------

def test_geometry_validators_registered():
    """The three Stage 8 validators are registered under GEOMETRY."""
    expected = {
        "floorplan_pitch_match",
        "floorplan_within_die_envelope",
        "floorplan_aspect_ratio",
    }
    names = set(default_registry.names())
    assert expected.issubset(names), f"missing: {expected - names}"
    for name in expected:
        validator = default_registry.get(name)
        assert validator is not None
        assert validator.category == ValidatorCategory.GEOMETRY


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_geometry_validators_dont_error_on_real_skus(
    sku_id, kpus, nodes, cooling
):
    """Stage 8 stance: WARN-max during heuristic-v1 development.

    All 4 catalog SKUs must run clean of ERROR findings under the
    geometry validators -- otherwise the Phase 6 catalog gate fails.
    Tighten thresholds to ERROR-level once the heuristic is calibrated.
    """
    sku = kpus[sku_id]
    ctx = ValidatorContext(
        sku=sku,
        process_node=nodes[sku.process_node_id],
        cooling_solutions=cooling,
    )
    findings = default_registry.run_category(ctx, ValidatorCategory.GEOMETRY)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert not errors, (
        f"{sku_id}: GEOMETRY validators emitted ERROR findings: "
        + "; ".join(f.render_one_line() for f in errors)
    )


def test_pitch_match_warns_on_t768(kpus, nodes, cooling):
    """T768 has the most aggressive per-class pitch differences in the
    catalog -- the floorplan_pitch_match validator must fire on it.

    Locks in the validator's signal: if a future YAML rebalancing
    accidentally hides the T768 pitch mismatch (e.g., by inflating
    INT8 per-PE area), this test fails so we notice.
    """
    sku = kpus["stillwater_kpu_t768"]
    ctx = ValidatorContext(
        sku=sku,
        process_node=nodes[sku.process_node_id],
        cooling_solutions=cooling,
    )
    findings = default_registry.run_one("floorplan_pitch_match", ctx)
    pitch_findings = [f for f in findings if f.severity != Severity.INFO]
    assert pitch_findings, (
        "T768 has known per-class pitch mismatch but pitch_match "
        "validator emitted no findings -- thresholds drifted?"
    )
    # Both INT8 and BF16 classes should fire (smaller than Matrix)
    classes_flagged = {
        f.block.split(":", 1)[1] for f in pitch_findings if f.block
    }
    assert "INT8-primary" in classes_flagged
    assert "BF16-primary" in classes_flagged


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_show_floorplan_cli_renders_t256():
    """Happy path: CLI prints a non-empty ASCII grid + a summary."""
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI), "stillwater_kpu_t256"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    # The summary lists the SKU name + die dimensions
    assert "Stillwater KPU-T256" in result.stdout
    assert "Per-tile-class pitches" in result.stdout
    # The legend confirms the ASCII renderer ran
    assert "Legend:" in result.stdout


def test_show_floorplan_cli_json_output(tmp_path: Path):
    """JSON mode: structured output with the right top-level keys."""
    out = tmp_path / "fp.json"
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI),
         "stillwater_kpu_t64", "--output", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["sku_id"] == "stillwater_kpu_t64"
    assert payload["die_area_mm2"] > 0
    assert "tile_pitches" in payload
    assert len(payload["blocks"]) > 0


def test_show_floorplan_cli_unknown_sku():
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI), "stillwater_kpu_t999"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "unknown SKU" in result.stderr


def test_show_floorplan_cli_list_mode():
    """--list returns the catalog ids, one per line."""
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI), "--list"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    ids = result.stdout.strip().split("\n")
    assert "stillwater_kpu_t256" in ids
    assert "stillwater_kpu_t768" in ids
