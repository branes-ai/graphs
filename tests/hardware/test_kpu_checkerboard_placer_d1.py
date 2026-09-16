"""Phase D1 of the KPU heterogeneous-tile refactor (graphs#268): assigning
tiles to checkerboard compute sites.

A uniform SKU has no checkerboard and keeps the legacy row-major layout,
which the golden floorplan snapshot pins. The heterogeneous fixture gets a
deterministic placement that honours multi-site footprints, placement
affinities and stream-link adjacency, and an explicit ``placement_map`` is
read back exactly as the author drew it.
"""

from __future__ import annotations

import pytest
from embodied_schemas import ComputeProduct, load_compute_products, load_process_nodes
from embodied_schemas.kpu import SPARE_SITE


from graphs.hardware import kpu_tile_display as display
from graphs.hardware.kpu_access import has_kpu_block, kpu_block_of
from graphs.hardware.kpu_checkerboard_placer import (
    PlacementError,
    SitePlan,
    place_tiles,
    stream_link_partners,
)
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_sku_generator import (
    generate_kpu_sku,
    input_spec_from_compute_product,
)
from hardware.test_kpu_catalog_ids import LEGACY_KPU_SKU_IDS

NODES = load_process_nodes()
HETERO = generate_kpu_sku(
    input_spec_from_compute_product(build_heterogeneous_kpu()), process_nodes=NODES
)
BLOCK = kpu_block_of(HETERO)
PLAN = place_tiles(BLOCK)

ISP, SGM, VIO = "ff_isp_raw2yuv", "ff_stereo_sgm", "ff_vio_stereo_inertial"
VIO_TYPE = "VIO"

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _adjacent(plan: SitePlan, a: str, b: str) -> bool:
    sites_b = set(plan.sites_of(b))
    return any(
        (r + dr, c + dc) in sites_b
        for r, c in plan.sites_of(a)
        for dr, dc in _NEIGHBORS
    )


def _with_block(**checkerboard) -> ComputeProduct:
    """The fixture with its checkerboard fields overridden, re-validated."""
    data = HETERO.model_dump(mode="json")
    data["dies"][0]["blocks"][0]["checkerboard"].update(checkerboard)
    return ComputeProduct.model_validate(data)


def _unchecked_block(**checkerboard):
    """The fixture's block with its checkerboard overridden *without*
    re-validation.

    The schema already rejects most malformed checkerboards (a spare count
    that disagrees with the map, an undeclared class id, a wrong shape), so
    these cases cannot be built through ``model_validate``. ``model_copy``
    skips validation, which is how the placer's own guards -- the ones that
    matter for a block built by code rather than parsed from YAML -- get
    exercised.
    """
    block = kpu_block_of(HETERO)
    return block.model_copy(
        update={"checkerboard": block.checkerboard.model_copy(update=checkerboard)}
    )


def _as_map(plan: SitePlan) -> list:
    owner = plan.site_owner()
    return [
        [owner.get((r, c), SPARE_SITE) for c in range(plan.cols)]
        for r in range(plan.rows)
    ]


# ---------------------------------------------------------------------------
# Uniform SKUs have no checkerboard and no placer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", LEGACY_KPU_SKU_IDS)
def test_legacy_catalog_skus_do_not_go_through_the_placer(sku):
    """Each *legacy* catalog SKU is a uniform mesh with no checkerboard
    spec, so it keeps the legacy row-major floorplan the golden snapshot
    pins. The catalog also holds kpu_h64_auto1, which does have a
    checkerboard and is covered by the fixture tests above."""
    block = kpu_block_of(load_compute_products()[sku])
    assert block.checkerboard is None
    assert place_tiles(block) is None


# ---------------------------------------------------------------------------
# Auto placement
# ---------------------------------------------------------------------------


def test_plan_covers_every_tile_exactly_once():
    assert PLAN.mode == "auto"
    assert (PLAN.rows, PLAN.cols) == (8, 8)
    # 45 tiles, but the 2x2 VIO costs four sites: 44 + 4 = 48 of 64.
    assert len(PLAN.placements) == BLOCK.total_tiles == 45
    assert PLAN.occupied_sites == 48
    assert len(PLAN.spare_sites) == 16
    assert PLAN.occupied_sites + len(PLAN.spare_sites) == PLAN.total_sites

    # No site is claimed twice, and spares are exactly the unclaimed ones.
    owned = [site for p in PLAN.placements for site in p.sites]
    assert len(owned) == len(set(owned)) == 48
    assert set(owned).isdisjoint(PLAN.spare_sites)

    # Every class gets the tile count it declared.
    counts = {}
    for p in PLAN.placements:
        counts[p.tile_class_id] = counts.get(p.tile_class_id, 0) + 1
    assert counts == {t.tile_class_id: t.num_tiles for t in BLOCK.tiles}


def test_placement_is_deterministic():
    """The golden floorplan snapshot is worthless if the placer wanders."""
    first = place_tiles(BLOCK)
    for _ in range(3):
        assert place_tiles(BLOCK) == first


def test_multi_site_footprint_is_placed_as_a_rectangle():
    vio = [p for p in PLAN.placements if p.tile_class_id == VIO]
    assert len(vio) == 1
    assert (vio[0].rows, vio[0].cols) == (2, 2)
    assert vio[0].num_sites == 4
    # Its four sites are a contiguous square anchored at its top-left.
    r, c = vio[0].row, vio[0].col
    assert set(vio[0].sites) == {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def test_a_multi_site_tile_covers_and_absorbs_its_paired_cells():
    """Each site is paired 1:1 with a memory cell, and a multi-site tile's
    rectangle spans the memory halves between its compute halves -- so it
    sits on all four of its cells, its own included."""
    vio = next(p for p in PLAN.placements if p.tile_class_id == VIO)
    assert set(PLAN.covered_memory_cells) == set(vio.sites)
    assert len(PLAN.covered_memory_cells) == 4

    # This class declares absorbs_memory_cells, so those four cells are its
    # private state rather than shared L3.
    assert PLAN.absorbed_memory_cells == PLAN.covered_memory_cells
    # A 1x1 tile covers nothing; its cell stays on the mesh.
    assert set(PLAN.covered_memory_cells).isdisjoint(PLAN.sites_of(ISP))


def test_covering_is_geometry_and_absorbing_is_accounting():
    """Without absorbs_memory_cells the tile still sits on the cells --
    geometry does not change -- but they stay shared L3."""
    data = HETERO.model_dump(mode="json")
    for tile in data["dies"][0]["blocks"][0]["tiles"]:
        if tile["tile_class_id"] == VIO:
            tile["footprint"]["absorbs_memory_cells"] = False
    plan = place_tiles(kpu_block_of(ComputeProduct.model_validate(data)))
    assert len(plan.covered_memory_cells) == 4
    assert plan.absorbed_memory_cells == ()


def test_io_edge_affinity_puts_the_isp_on_the_edge():
    (row, col), = PLAN.sites_of(ISP)
    assert row == 0  # the IO pads edge
    assert min(row, col, PLAN.rows - 1 - row, PLAN.cols - 1 - col) == 0


def test_stream_linked_classes_end_up_adjacent():
    """ISP -> SGM -> VIO is a NoC stream-link overlay; if the placer
    separates them the link crosses the mesh."""
    assert stream_link_partners(BLOCK) == {
        ISP: {SGM}, SGM: {ISP, VIO}, VIO: {SGM},
    }
    assert _adjacent(PLAN, ISP, SGM)
    assert _adjacent(PLAN, SGM, VIO)


def test_affinity_outranks_adjacency_for_a_class_that_declared_one():
    """The ISP is both IO-edge-pinned and a stream partner. Its partner
    must not drag it off the edge."""
    isp = next(t for t in BLOCK.tiles if t.tile_class_id == ISP)
    assert isp.placement.affinity.value == "io_edge"
    (row, _), = PLAN.sites_of(ISP)
    assert row == 0
    assert _adjacent(PLAN, ISP, SGM)  # and it still got its partner


def test_render_is_a_grid_with_spares_marked():
    lines = PLAN.render().splitlines()
    assert len(lines) == PLAN.rows
    assert all(len(line.split()) == PLAN.cols for line in lines)
    # The last two rows are the 16 spare sites.
    assert set(lines[-1].split()) == {SPARE_SITE}
    assert SPARE_SITE not in lines[0].split()


def test_placer_reports_a_grid_that_is_too_small():
    """A 4x4 grid cannot hold 48 sites' worth of tiles; that is an error
    with a number in it, not a silent truncation."""
    from embodied_schemas.kpu import SiteGrid

    with pytest.raises(PlacementError, match="does not fit"):
        place_tiles(_unchecked_block(compute_sites=SiteGrid(rows=4, cols=4)))


# ---------------------------------------------------------------------------
# Explicit placement
# ---------------------------------------------------------------------------


def test_explicit_map_is_read_back_exactly():
    """Feeding the auto plan back as an explicit map reproduces it, so the
    two modes agree on what a placement means."""
    explicit = place_tiles(kpu_block_of(
        _with_block(placement="explicit", placement_map=_as_map(PLAN))
    ))
    assert explicit.mode == "explicit"
    assert explicit.notes == ()
    assert explicit.site_owner() == PLAN.site_owner()
    assert explicit.absorbed_memory_cells == PLAN.absorbed_memory_cells
    assert sorted(explicit.spare_sites) == sorted(PLAN.spare_sites)
    assert sorted((p.tile_class_id, p.row, p.col) for p in explicit.placements) == \
        sorted((p.tile_class_id, p.row, p.col) for p in PLAN.placements)


def test_explicit_map_recovers_a_multi_site_footprint_as_one_tile():
    explicit = place_tiles(kpu_block_of(
        _with_block(placement="explicit", placement_map=_as_map(PLAN))
    ))
    vio = [p for p in explicit.placements if p.tile_class_id == VIO]
    assert len(vio) == 1  # one tile, not four 1x1 tiles
    assert (vio[0].rows, vio[0].cols) == (2, 2)


def test_placer_rejects_a_map_whose_shape_is_not_the_grid():
    with pytest.raises(PlacementError, match="but compute_sites is 8x8"):
        place_tiles(_unchecked_block(placement_map=_as_map(PLAN)[:4]))


def test_explicit_map_rejects_an_undeclared_class():
    pmap = _as_map(PLAN)
    pmap[7][7] = "pe_not_a_real_class"
    with pytest.raises(PlacementError, match="does not declare"):
        place_tiles(_unchecked_block(placement_map=pmap))


def test_explicit_map_rejects_a_broken_footprint_rectangle():
    """A 2x2 class needs a 2x2 block of its own sites; a torn corner is an
    error, not a silently reshaped tile."""
    pmap = _as_map(PLAN)
    vio = next(p for p in PLAN.placements if p.tile_class_id == VIO)
    pmap[vio.row + 1][vio.col + 1] = SPARE_SITE
    with pytest.raises(PlacementError, match="block of its own sites"):
        place_tiles(_unchecked_block(placement_map=pmap))


def test_explicit_map_notes_a_tile_count_disagreement():
    """The map is the authority, so a count mismatch is reported rather
    than corrected; the C4 site-accounting validator turns it into a
    finding."""
    pmap = _as_map(PLAN)
    replaced = 0
    for r, row in enumerate(pmap):
        for c, cid in enumerate(row):
            if cid == "pe_int8_mac_i32" and replaced < 2:
                pmap[r][c] = SPARE_SITE
                replaced += 1
    plan = place_tiles(_unchecked_block(placement_map=pmap))
    assert len(plan.notes) == 1
    assert "holds 22 'pe_int8_mac_i32' tile(s)" in plan.notes[0]
    assert "declares 24" in plan.notes[0]


# ---------------------------------------------------------------------------
# The architectural floorplan consumes the plan
# ---------------------------------------------------------------------------


def _hetero_floorplan():
    from graphs.hardware.silicon_floorplan import derive_kpu_architectural_floorplan

    return derive_kpu_architectural_floorplan(HETERO, NODES["tsmc_n16"])


def test_floorplan_gives_each_kind_its_own_role():
    from graphs.hardware.silicon_floorplan import TileRole

    blocks = _hetero_floorplan().blocks
    roles = {}
    for b in blocks:
        roles[b.role] = roles.get(b.role, 0) + 1
    assert roles[TileRole.COMPUTE] == 38  # the PE-fabric classes
    assert roles[TileRole.SYSTOLIC] == 4
    assert roles[TileRole.FIXED_FUNCTION] == 3
    # 64 sites, 4 of them covered by the 2x2 VIO core.
    assert roles[TileRole.MEMORY] == 60


def test_floorplan_tiles_do_not_overlap():
    """A multi-site footprint spans the memory halves it covers, so the
    cells under it must not also be emitted."""
    from graphs.hardware.silicon_floorplan import TileRole

    boxes = [
        (b.x_mm, b.y_mm, b.width_mm, b.height_mm, b.name)
        for b in _hetero_floorplan().blocks
        if b.role != TileRole.IO_PAD  # the ring is a frame, not a cell
    ]

    def overlaps(a, b) -> bool:
        eps = 1e-9
        return (
            a[0] < b[0] + b[2] - eps and b[0] < a[0] + a[2] - eps
            and a[1] < b[1] + b[3] - eps and b[1] < a[1] + a[3] - eps
        )

    collisions = [
        (a[4], b[4])
        for i, a in enumerate(boxes)
        for b in boxes[i + 1:]
        if overlaps(a, b)
    ]
    assert collisions == []


def test_floorplan_multi_site_tile_spans_its_site_rectangle():
    vio = next(b for b in _hetero_floorplan().blocks if b.name.startswith(VIO))
    pitch = _hetero_floorplan().unified_pitch_mm
    # 2 sites wide = 4 physical cells, 1 site... 2 sites tall = 2 cells.
    assert vio.width_mm == pytest.approx(4 * pitch)
    assert vio.height_mm == pytest.approx(2 * pitch)
    # It carries the L3 of all four cells it sits on, so that SRAM stays in
    # the area roll-up instead of reading as whitespace.
    assert vio.l3_area_mm2 is not None and vio.l3_area_mm2 > 0
    assert vio.used_area_mm2 >= vio.l3_area_mm2


def test_every_class_resolves_to_real_compute_area():
    """Silicon carried on a tile class is not a silicon_bin per_pe block.
    Before D2 those classes sized to their L2 term alone; they now draw on
    ``carried_silicon`` as well, so every class on the fixture has area."""
    fp = _hetero_floorplan()
    assert "no resolved compute area" not in fp.notes
    for tile_type, summary in fp.compute_summaries.items():
        assert summary.pe_area_mm2 > 0, tile_type
    # The systolic and fixed-function classes are the ones that used to be
    # zero; they are now the largest, which is why they are there.
    assert fp.compute_summaries["Systolic-INT8-WS"].pe_area_mm2 > \
        fp.compute_summaries["INT8-MAC"].pe_area_mm2


def test_a_count_ref_by_class_id_resolves_like_one_by_label():
    """The fixture writes one per_pe count_ref as ``tile.<tile_class_id>``
    and another as ``tile.<tile_type>``; both are legal, and the floorplan
    keys on the class each points at rather than on the spelling."""
    block = kpu_block_of(HETERO)
    refs = {
        b.transistor_source.count_ref
        for b in HETERO.dies[0].silicon_bin.blocks
        if b.transistor_source.count_ref
        and b.transistor_source.count_ref.startswith("tile.")
    }
    assert "tile.pe_int8_mac_i32" in refs  # by tile_class_id
    assert "tile.MINPLUS-I16" in refs      # by tile_type label
    summaries = _hetero_floorplan().compute_summaries
    # Both land on their class, not on a key nothing matches.
    assert summaries["INT8-MAC"].pe_area_mm2 > 0
    assert summaries["MINPLUS-I16"].pe_area_mm2 > 0
    assert {t.tile_type for t in block.tiles} == set(summaries)


def test_a_library_the_node_lacks_is_reported_not_silently_dropped(caplog):
    """The fixture's systolic accumulator asks for sram_hp, which tsmc_n16
    does not offer (embodied-schemas#96). Its area cannot be resolved, so
    the floorplan says so."""
    import logging

    with caplog.at_level(logging.WARNING, logger="graphs.hardware.silicon_floorplan"):
        _hetero_floorplan()
    assert any("sram_hp" in r.getMessage() for r in caplog.records)


def test_legacy_floorplans_carry_no_such_note():
    from embodied_schemas import load_compute_products
    from graphs.hardware.silicon_floorplan import derive_kpu_architectural_floorplan

    cp = load_compute_products()["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    fp = derive_kpu_architectural_floorplan(cp, NODES["tsmc_n16"])
    assert "no resolved compute area" not in fp.notes
    assert fp.notes.startswith("Architectural v2:")


# ---------------------------------------------------------------------------
# D2: site overlays
# ---------------------------------------------------------------------------


def _with_domains():
    from embodied_schemas import ComputeProduct

    data = HETERO.model_dump(mode="json")
    data["dies"][0]["blocks"][0]["power_domains"] = [
        {"domain_id": "int8_pe", "kind": "tile_class",
         "members": ["pe_int8_mac_i32"], "gateable": True},
        {"domain_id": "vision", "kind": "tile_class",
         "members": [ISP, SGM, VIO], "gateable": True},
        {"domain_id": "quad0", "kind": "cluster",
         "site_ranges": [{"row_min": 4, "row_max": 5,
                          "col_min": 0, "col_max": 7}]},
    ]
    return kpu_block_of(ComputeProduct.model_validate(data))


def test_tile_class_domain_covers_the_sites_its_classes_landed_on():
    """A tile_class domain names classes, not sites, so which sites it
    gates is only knowable after placement."""
    from graphs.hardware.kpu_checkerboard_placer import site_power_domains

    block = _with_domains()
    plan = place_tiles(block)
    domains = site_power_domains(block, plan)
    assert set(plan.sites_of("pe_int8_mac_i32")) == {
        s for s, d in domains.items() if d == "int8_pe"
    }
    # The vision domain gathers all three fixed-function classes.
    vision = {s for s, d in domains.items() if d == "vision"}
    assert vision == set(plan.sites_of(ISP)) | set(plan.sites_of(SGM)) \
        | set(plan.sites_of(VIO))


def test_cluster_domain_covers_its_declared_site_range():
    from graphs.hardware.kpu_checkerboard_placer import site_power_domains

    block = _with_domains()
    domains = site_power_domains(block, place_tiles(block))
    quad = {s for s, d in domains.items() if d == "quad0"}
    assert quad == {(r, c) for r in (4, 5) for c in range(8)}


def test_sites_in_no_domain_are_left_unlabelled():
    from graphs.hardware.kpu_checkerboard_placer import site_power_domains

    block = _with_domains()
    plan = place_tiles(block)
    domains = site_power_domains(block, plan)
    occupied = set(plan.site_owner())
    uncovered = occupied - set(domains)
    assert uncovered, "the fixture should leave some sites ungated"

    # Every uncovered site belongs to a class no tile_class domain names,
    # and lies outside the cluster domain's rows.
    named = {"pe_int8_mac_i32", ISP, SGM, VIO}
    owner = plan.site_owner()
    for row, col in uncovered:
        assert owner[(row, col)] not in named
        assert row not in (4, 5)


def test_render_overlay_marks_spares_and_unlabelled_sites():
    from graphs.hardware.kpu_checkerboard_placer import (
        render_overlay,
        site_power_domains,
    )

    block = _with_domains()
    plan = place_tiles(block)
    grid, legend = render_overlay(plan, site_power_domains(block, plan))
    lines = grid.splitlines()
    assert len(lines) == plan.rows
    assert all(len(line.split()) == plan.cols for line in lines)
    assert set(legend.values()) == {"int8_pe", "vision", "quad0"}
    assert set(lines[-1].split()) == {SPARE_SITE}   # the spare rows
    assert "-" in grid                              # LNS / min-plus sites


def test_a_class_with_unresolvable_carried_sram_gets_no_shared_l2():
    """The fixture's systolic accumulator asks for sram_hp, absent on
    tsmc_n16, so its carried SRAM resolves to nothing. It must not silently
    fall back to a share of the chip-wide L2 pool it does not draw on
    (CodeRabbit on #287)."""
    fp = _hetero_floorplan()
    shared_l2 = fp.compute_summaries["INT8-MAC"].l2_area_mm2
    # A PE fabric carries no memory of its own, so it does take a share.
    assert shared_l2 > 0
    assert fp.compute_summaries["LNS16-MAC"].l2_area_mm2 == shared_l2

    # The systolic class declares two buffers: a weight buffer in sram_hd,
    # which resolves, and an accumulator in sram_hp, which tsmc_n16 does
    # not offer. It gets the part that resolved -- never the shared share.
    systolic = fp.compute_summaries["Systolic-INT8-WS"].l2_area_mm2
    assert 0 < systolic < shared_l2

    # A fixed-function core's silicon covers its SRAM, so it draws nothing
    # from the shared pool either.
    for ff in ("ISP", "SGM", "VIO"):
        assert fp.compute_summaries[ff].l2_area_mm2 == 0.0


def test_placed_tiles_report_their_own_memory_term():
    """``compute_mem_areas`` drives pitch and the class summary, so the
    placed block must use it too or ``used_area_mm2`` disagrees with the
    summary (CodeRabbit on #287).

    It is a per-*tile* figure, and a multi-site placement is still one
    tile, so it is not scaled by the site count -- scaling it was the bug
    this test originally enshrined (CodeRabbit on #288).
    """
    fp = _hetero_floorplan()
    by_class = {}
    for b in fp.blocks:
        if b.tile_class is not None:
            by_class.setdefault(b.tile_class, b)
    for tile_type, summary in fp.compute_summaries.items():
        block = by_class[tile_type]
        assert block.l2_area_mm2 == pytest.approx(summary.l2_area_mm2), tile_type
    # At least one class must have a non-zero term, or this proves nothing.
    assert any(b.l2_area_mm2 for b in by_class.values())


def test_the_circuit_view_gives_a_multi_site_tile_its_whole_footprint():
    """The architectural view sizes blocks from the placement; the circuit
    view used to emit one pitch-sized square per tile, so an 8-site core
    showed as one site and its area vanished from the roll-up
    (CodeRabbit on #288)."""
    from graphs.hardware.silicon_floorplan import derive_kpu_floorplan

    fp = derive_kpu_floorplan(HETERO, NODES["tsmc_n16"])
    tiles = fp.compute_tiles()
    assert len(tiles) == BLOCK.total_tiles  # one block per tile, not per site

    pitch = fp.unified_pitch_mm
    sites_by_type = {t.tile_type: display.tile_sites(t) for t in BLOCK.tiles}
    for block in tiles:
        sites = sites_by_type[block.tile_class]
        assert block.area_mm2 == pytest.approx(sites * pitch * pitch), block.name

    vio = next(b for b in tiles if b.tile_class == VIO_TYPE)
    assert vio.width_mm == pytest.approx(2 * pitch)
    assert vio.height_mm == pytest.approx(2 * pitch)
