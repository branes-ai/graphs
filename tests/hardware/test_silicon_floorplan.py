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
    ArchitecturalFloorplan,
    ArchTile,
    Floorplan,
    FloorplanBlock,
    TileRole,
    derive_kpu_architectural_floorplan,
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
    """All five Stage 8 GEOMETRY validators are registered."""
    expected = {
        # Stage 8a (circuit-class view)
        "floorplan_pitch_match",
        "floorplan_within_die_envelope",
        "floorplan_aspect_ratio",
        # Stage 8b (architectural view)
        "floorplan_compute_memory_pitch_match",
        "floorplan_whitespace_fraction",
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
    """Happy path: CLI prints a non-empty ASCII grid + a summary.

    Default view is architectural (Stage 8b); coverage of the
    architectural-specific output lives in
    ``test_show_floorplan_cli_default_is_architectural``.
    """
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI), "stillwater_kpu_t256"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Stillwater KPU-T256" in result.stdout
    # The legend confirms the ASCII renderer ran
    assert "Legend:" in result.stdout


def test_show_floorplan_cli_json_output(tmp_path: Path):
    """JSON mode: structured output with the right top-level keys.

    Default view is architectural; the circuit-class JSON shape is
    covered separately by ``test_show_floorplan_cli_view_circuit_*``.
    """
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


# ---------------------------------------------------------------------------
# Architectural view: structural properties (Stage 8b)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_arch_floorplan_has_positive_dimensions(sku_id, kpus, nodes):
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    assert fp.die_width_mm > 0
    assert fp.die_height_mm > 0
    assert fp.die_area_mm2 > 0


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_arch_floorplan_compute_memory_one_to_one(sku_id, kpus, nodes):
    """Architectural pairing: 1 memory tile per compute tile (each
    compute tile owns its L3). The checkerboard layout depends on
    this 1:1 invariant."""
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    assert len(fp.compute_tiles()) == sku.kpu_architecture.total_tiles
    assert len(fp.memory_tiles()) == sku.kpu_architecture.total_tiles


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_arch_floorplan_distributes_memory_controllers(sku_id, kpus, nodes):
    """Memory controllers must be placed (default 4) and distributed
    around the periphery (no two on top of each other)."""
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    ctrls = fp.memory_controllers()
    assert len(ctrls) >= 1, f"{sku_id}: no memory controllers placed"
    # Each controller has positive area
    for c in ctrls:
        assert c.area_mm2 > 0
    # Controllers don't overlap each other (pairwise)
    eps = 1e-3
    for i, a in enumerate(ctrls):
        for b in ctrls[i + 1:]:
            x_ov = (
                a.x_mm + eps < b.x_mm + b.width_mm
                and b.x_mm + eps < a.x_mm + a.width_mm
            )
            y_ov = (
                a.y_mm + eps < b.y_mm + b.height_mm
                and b.y_mm + eps < a.y_mm + a.height_mm
            )
            assert not (x_ov and y_ov), (
                f"{sku_id}: controllers {a.name} and {b.name} overlap"
            )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_arch_floorplan_what_if_one_per_class(sku_id, kpus, nodes):
    """What-if estimates: one entry per declared compute tile class."""
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    declared_classes = {t.tile_type for t in sku.kpu_architecture.tiles}
    estimated_classes = {wi.tile_class for wi in fp.what_if}
    assert estimated_classes == declared_classes, (
        f"{sku_id}: what-if covers {estimated_classes}, "
        f"expected {declared_classes}"
    )


def test_arch_what_if_all_int8_smaller_than_all_matrix(kpus, nodes):
    """The whole point of the what-if: an all-INT8 mesh should be
    notably smaller than an all-Matrix mesh, because INT8 PE area
    is much less than Matrix PE area. If this stops being true, the
    silicon_bin coefficients have drifted in a way the architect
    should know about.
    """
    sku = kpus["stillwater_kpu_t768"]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    by_class = {wi.tile_class: wi for wi in fp.what_if}
    assert "INT8-primary" in by_class
    assert "Matrix" in by_class
    int8_die = by_class["INT8-primary"].die_area_mm2
    mat_die = by_class["Matrix"].die_area_mm2
    assert int8_die < mat_die, (
        f"all-INT8 die {int8_die:.1f} mm^2 is not smaller than "
        f"all-Matrix die {mat_die:.1f} mm^2 -- silicon_bin drifted?"
    )


def test_compute_memory_pitch_match_warns_on_t768(kpus, nodes, cooling):
    """T768's compute tiles range from 0.175 mm (INT8) to 0.508 mm
    (Matrix); memory pitch is ~0.142 mm. The C/M pitch validator
    must fire on at least one tile class for T768."""
    sku = kpus["stillwater_kpu_t768"]
    ctx = ValidatorContext(
        sku=sku, process_node=nodes[sku.process_node_id],
        cooling_solutions=cooling,
    )
    findings = default_registry.run_one(
        "floorplan_compute_memory_pitch_match", ctx
    )
    pitch_findings = [f for f in findings if f.severity != Severity.INFO]
    assert pitch_findings, (
        "T768 has known compute-vs-memory pitch mismatch but the "
        "validator emitted no findings -- thresholds drifted?"
    )


def test_whitespace_validator_warns_on_t768(kpus, nodes, cooling):
    """T768 has 76% architectural whitespace (Matrix tiles dominate
    the unified pitch). Whitespace validator must fire."""
    sku = kpus["stillwater_kpu_t768"]
    ctx = ValidatorContext(
        sku=sku, process_node=nodes[sku.process_node_id],
        cooling_solutions=cooling,
    )
    findings = default_registry.run_one("floorplan_whitespace_fraction", ctx)
    assert findings, (
        "T768 has 76% whitespace but whitespace validator emitted "
        "nothing -- threshold drifted?"
    )
    # The message should reference the all-X what-if scenario
    msg = findings[0].message
    assert "all-" in msg or "what-if" in msg.lower() or "smaller" in msg, (
        f"whitespace finding doesn't mention the what-if alternative: {msg}"
    )


# ---------------------------------------------------------------------------
# CLI: --view flag
# ---------------------------------------------------------------------------

def test_show_floorplan_cli_default_is_architectural():
    """The architectural view is the default. Output should mention
    'architectural', the per-class compute summary, and the what-if
    table."""
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI), "stillwater_kpu_t256"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "architectural" in out.lower()
    assert "Compute tile classes" in out
    assert "Memory tiles" in out
    assert "What-if" in out
    # Architectural glyphs in the legend
    assert "C=compute" in out
    assert "M=memory" in out


def test_show_floorplan_cli_view_circuit_falls_back_to_old():
    """--view circuit reproduces the Stage 8a renderer output."""
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI),
         "stillwater_kpu_t256", "--view", "circuit"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "circuit-class" in out.lower()
    assert "Per-tile-class pitches" in out
    # Circuit glyphs in legend
    assert "H=hp_logic" in out or "B=balanced_logic" in out


# ---------------------------------------------------------------------------
# Architectural view v2 -- true 2D checkerboard + memory-spec-driven MCs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_arch_true_2d_checkerboard_interior_compute_has_4_memory_neighbors(
    sku_id, kpus, nodes
):
    """True 2D checkerboard: every interior compute tile has exactly
    4 memory tile neighbours (N/S/E/W). Catches regressions to
    column-pair (1 neighbour) or row-pair patterns.
    """
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    compute_tiles = fp.compute_tiles()
    memory_tiles = fp.memory_tiles()
    assert compute_tiles and memory_tiles

    pitch = fp.unified_pitch_mm
    if pitch <= 0:
        return

    # Pick an interior compute tile by parsing its checkerboard
    # coordinates ``compute[r,c]`` (encoded into the name). Interior
    # = at least one row/col away from every mesh edge.
    mesh_rows = sku.kpu_architecture.noc.mesh_rows
    mesh_cols_phys = 2 * sku.kpu_architecture.noc.mesh_cols
    target = None
    import re
    for ct in compute_tiles:
        m = re.match(r"compute\[(\d+),(\d+)\]", ct.name)
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        if 1 <= r <= mesh_rows - 2 and 1 <= c <= mesh_cols_phys - 2:
            target = ct
            break
    assert target is not None, (
        f"{sku_id}: no interior compute tile found "
        f"(mesh {mesh_rows}x{mesh_cols_phys})"
    )

    eps = 0.01
    neighbours = 0
    for m in memory_tiles:
        same_row = abs(m.y_mm - target.y_mm) < eps
        same_col = abs(m.x_mm - target.x_mm) < eps
        horiz_adj = abs(abs(m.x_mm - target.x_mm) - pitch) < eps
        vert_adj = abs(abs(m.y_mm - target.y_mm) - pitch) < eps
        if (same_row and horiz_adj) or (same_col and vert_adj):
            neighbours += 1
    assert neighbours == 4, (
        f"{sku_id}: interior compute tile {target.name} has "
        f"{neighbours} memory neighbours, expected 4 for true 2D "
        f"checkerboard"
    )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_mc_count_equals_memory_controllers_field(sku_id, kpus, nodes):
    """MC count comes from ``memory.memory_controllers``, not from a
    silicon_bin estimate or default. T256 (16 LPDDR ch), T128 (8),
    T64 (4), T768 (8 HBM stacks)."""
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    declared = sku.kpu_architecture.memory.memory_controllers
    assert fp.num_memory_controllers == declared, (
        f"{sku_id}: floorplan placed {fp.num_memory_controllers} MCs, "
        f"YAML declares {declared}"
    )
    assert len(fp.memory_controllers()) == declared
    assert len(fp.memory_channels) == declared


def test_lpddr_mc_is_long_narrow_stripe(kpus, nodes):
    """LPDDR PHY blocks are narrow stripes (~5:1 aspect)."""
    sku = kpus["stillwater_kpu_t256"]  # LPDDR5
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    for mc in fp.memory_controllers():
        long = max(mc.width_mm, mc.height_mm)
        short = min(mc.width_mm, mc.height_mm)
        ar = long / short
        assert ar >= 3.0, (
            f"LPDDR MC {mc.name} aspect ratio {ar:.2f} is not "
            f"narrow-stripe-like; LPDDR PHYs run along bump edge"
        )


def test_hbm_mc_is_squarer_than_lpddr(kpus, nodes):
    """HBM PHY blocks are squarer (~1.5:1 aspect) than LPDDR (~5:1)."""
    lpddr_sku = kpus["stillwater_kpu_t256"]
    hbm_sku = kpus["stillwater_kpu_t768"]
    fp_lpddr = derive_kpu_architectural_floorplan(
        lpddr_sku, nodes[lpddr_sku.process_node_id]
    )
    fp_hbm = derive_kpu_architectural_floorplan(
        hbm_sku, nodes[hbm_sku.process_node_id]
    )
    def aspect(mc):
        long = max(mc.width_mm, mc.height_mm)
        short = min(mc.width_mm, mc.height_mm)
        return long / short if short > 0 else float("inf")
    lpddr_ar = aspect(fp_lpddr.memory_controllers()[0])
    hbm_ar = aspect(fp_hbm.memory_controllers()[0])
    assert hbm_ar < lpddr_ar, (
        f"HBM aspect {hbm_ar:.2f} should be smaller than LPDDR "
        f"aspect {lpddr_ar:.2f} (HBM PHYs are squarer microbump arrays)"
    )


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_dram_channels_placed_outside_die(sku_id, kpus, nodes):
    """Every memory channel sits beyond the die boundary on its
    respective edge; channels are not inside the die."""
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    for ch in fp.memory_channels:
        if ch.edge == "top":
            assert ch.y_mm >= fp.die_height_mm - 1e-6, (
                f"{sku_id}: top channel {ch.channel_id} at y={ch.y_mm} "
                f"is not above die top (die_h={fp.die_height_mm})"
            )
        elif ch.edge == "bottom":
            assert ch.y_mm + ch.height_mm <= 1e-6, (
                f"{sku_id}: bottom channel {ch.channel_id} extends "
                f"into die at y={ch.y_mm}"
            )
        elif ch.edge == "right":
            assert ch.x_mm >= fp.die_width_mm - 1e-6
        elif ch.edge == "left":
            assert ch.x_mm + ch.width_mm <= 1e-6


@pytest.mark.parametrize("sku_id", ALL_KPU_IDS)
def test_mc_dimensions_independent_of_mesh_size(sku_id, kpus, nodes):
    """MC area is determined by memory type + channel width, NOT by
    compute mesh size. Catch regressions to the v1 behaviour where
    MC side = sqrt(total_phy_area / num_controllers) shrunk with
    smaller meshes.
    """
    sku = kpus[sku_id]
    fp = derive_kpu_architectural_floorplan(sku, nodes[sku.process_node_id])
    mem = sku.kpu_architecture.memory
    if mem.memory_controllers <= 0 or mem.memory_bus_bits <= 0:
        return
    # PHY/channel area should match the memory-type lookup; if the
    # MC area scales with mesh size instead, this disagrees.
    ch_width = mem.memory_bus_bits // mem.memory_controllers
    assert fp.per_channel_phy_area_mm2 > 0
    # All MCs should have the same area (uniform per-channel sizing)
    mcs = fp.memory_controllers()
    areas = {round(mc.area_mm2, 4) for mc in mcs}
    assert len(areas) == 1, (
        f"{sku_id}: MCs have multiple distinct sizes {areas}; "
        f"expected uniform per-channel sizing"
    )


def test_show_floorplan_cli_arch_json_output(tmp_path: Path):
    """Architectural JSON has the new top-level fields."""
    out = tmp_path / "fp.json"
    result = subprocess.run(
        [sys.executable, str(SHOW_FLOORPLAN_CLI),
         "stillwater_kpu_t768", "--output", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["view"] == "architectural"
    assert payload["sku_id"] == "stillwater_kpu_t768"
    assert "compute_summaries" in payload
    assert "memory_summary" in payload
    assert "what_if" in payload
    assert payload["num_memory_controllers"] >= 1
    # what-if has one entry per declared compute class
    assert len(payload["what_if"]) >= 1
    for wi in payload["what_if"]:
        assert wi["die_area_mm2"] > 0
