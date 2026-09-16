"""Phase E1 of the KPU heterogeneous-tile refactor (graphs#268): the
reference heterogeneous SKU.

``data/sku_specs/kpu_h64_auto1_*.yaml`` are hand-authored input specs that
exercise the whole refactor end to end -- tile-class library ``use:``
references (C3), tile-carried silicon (C1), the by-kind performance
roll-up (B5), multi-site footprints with absorbed memory cells, placement
affinities and a stream link (D1), and the per-class area model (D2).

These tests are the contract: both specs generate, validate without an
ERROR, and place deterministically with the autonomy chain intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from embodied_schemas import load_cooling_solutions, load_process_nodes

from graphs.hardware.kpu_access import kpu_block_of
from graphs.hardware.kpu_checkerboard_placer import place_tiles
from graphs.hardware.kpu_sku_generator import KPUSKUInputSpec, generate_kpu_sku
from graphs.hardware.silicon_floorplan import (
    TileRole,
    derive_kpu_architectural_floorplan,
)
from graphs.hardware.sku_validators import (
    Severity,
    build_context_for_kpu,
    default_registry,
    load_validators,
)

SPEC_DIR = Path(__file__).resolve().parents[2] / "data" / "sku_specs"
SPECS = {
    "tsmc_n16": SPEC_DIR / "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp.yaml",
    "tsmc_n7": SPEC_DIR / "kpu_h64_auto1_lp5x4_7nm_tsmc_hpc.yaml",
}
NODES = load_process_nodes()

ISP, SGM, VIO = "ff_isp_raw2yuv", "ff_stereo_sgm", "ff_vio_stereo_inertial"
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _generate(node_id: str):
    spec = KPUSKUInputSpec.model_validate(
        yaml.safe_load(SPECS[node_id].read_text(encoding="utf-8"))
    )
    return generate_kpu_sku(spec, process_nodes=NODES)


SKUS = {node: _generate(node) for node in SPECS}


@pytest.fixture(scope="module", autouse=True)
def _validators():
    load_validators()


def _findings(sku):
    ctx = build_context_for_kpu(
        sku.id,
        kpus={sku.id: sku},
        process_nodes=NODES,
        cooling_solutions=load_cooling_solutions(),
    )
    return default_registry.run_all(ctx)


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_spec_generates(node_id):
    sku = SKUS[node_id]
    assert sku.id.startswith("kpu_h64_auto1")
    assert sku.dies[0].process_node_id == node_id


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_no_error_findings(node_id):
    """The acceptance bar for a reference SKU: nothing a validator calls an
    ERROR. Warnings are design observations and are allowed -- the uniform
    catalog SKUs carry them too."""
    errors = [f for f in _findings(SKUS[node_id]) if f.severity == Severity.ERROR]
    assert errors == [], "\n".join(f.message for f in errors)


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_every_tile_class_comes_from_the_library(node_id):
    """The SKU says how many of each class and where they sit; what a class
    *is* lives in the embodied-schemas tile-class library."""
    for tile in kpu_block_of(SKUS[node_id]).tiles:
        assert tile.tile_class_ref, tile.tile_class_id


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_site_accounting_balances(node_id):
    block = kpu_block_of(SKUS[node_id])
    cb = block.checkerboard
    occupied = sum(t.num_tiles * t.sites_per_tile for t in block.tiles)
    assert occupied + cb.spare_sites == cb.compute_sites.rows * cb.compute_sites.cols
    assert occupied == 59 and cb.spare_sites == 5


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_all_three_tile_kinds_are_present(node_id):
    kinds = {t.tile_kind.value for t in kpu_block_of(SKUS[node_id]).tiles}
    assert kinds == {"pe_fabric", "systolic", "fixed_function"}


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_the_autonomy_chain_is_placed_adjacent(node_id):
    """ISP -> SGM -> VIO must be 4-adjacent or the stream link crosses the
    mesh, which is the whole reason the chain exists."""
    plan = place_tiles(kpu_block_of(SKUS[node_id]))

    def adjacent(a: str, b: str) -> bool:
        sites_b = set(plan.sites_of(b))
        return any(
            (r + dr, c + dc) in sites_b
            for r, c in plan.sites_of(a)
            for dr, dc in _NEIGHBORS
        )

    assert adjacent(ISP, SGM)
    assert adjacent(SGM, VIO)


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_the_isp_is_on_an_edge(node_id):
    """It faces the camera PHYs. It is also a stream partner, so this is
    the case where an affinity and an adjacency both have to hold."""
    plan = place_tiles(kpu_block_of(SKUS[node_id]))
    (row, col), = plan.sites_of(ISP)
    assert min(row, col, plan.rows - 1 - row, plan.cols - 1 - col) == 0


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_the_multi_site_cores_absorb_the_cells_they_sit_on(node_id):
    """Every multi-site core here is opaque, so the L3 cells under it are
    its own storage, not scratchpad the mesh can still reach. Covered and
    absorbed therefore coincide: 8 (SGM) + 4 (VIO) + 4x2 (systolic)."""
    plan = place_tiles(kpu_block_of(SKUS[node_id]))
    vio = next(p for p in plan.placements if p.tile_class_id == VIO)
    assert (vio.rows, vio.cols) == (2, 2)
    assert set(vio.sites) <= set(plan.absorbed_memory_cells)
    assert plan.absorbed_memory_cells == plan.covered_memory_cells
    assert len(plan.absorbed_memory_cells) == 20


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_placement_is_deterministic(node_id):
    block = kpu_block_of(SKUS[node_id])
    first = place_tiles(block)
    assert place_tiles(block) == first


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_every_class_has_real_area_and_a_plausible_pitch(node_id):
    """The D2 area model in anger: no class may fall back to its memory
    term, and no class may need more silicon than its sites offer."""
    fp = derive_kpu_architectural_floorplan(SKUS[node_id], NODES[node_id])
    assert "no resolved compute area" not in fp.notes
    for tile_type, summary in fp.compute_summaries.items():
        assert summary.pe_area_mm2 > 0, tile_type
        # Per-site pitch, so a multi-site class is not judged as a 1x1 one.
        assert summary.pitch_mm <= fp.unified_pitch_mm + 1e-9, tile_type


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_the_floorplan_shows_each_kind_separately(node_id):
    fp = derive_kpu_architectural_floorplan(SKUS[node_id], NODES[node_id])
    roles = {}
    for b in fp.blocks:
        roles[b.role] = roles.get(b.role, 0) + 1
    assert roles[TileRole.SYSTOLIC] == 4
    assert roles[TileRole.FIXED_FUNCTION] == 3
    assert roles[TileRole.COMPUTE] == 38
    # 64 sites less the 20 that a multi-site core sits on: the 2x4 SGM,
    # the 2x2 VIO and the four 1x2 systolic arrays.
    assert roles[TileRole.MEMORY] == 44


@pytest.mark.parametrize("node_id", sorted(SPECS))
def test_performance_rolls_up_by_kind(node_id):
    """The systolic array and the PE fabric both run INT8, and the roll-up
    says how much each contributes (B5)."""
    by_kind = SKUS[node_id].performance.by_tile_kind
    assert set(by_kind) == {"pe_fabric", "systolic"}  # fixed-function has no ops
    assert by_kind["systolic"]["int8"] > 0
    assert by_kind["pe_fabric"]["int8"] > 0
    total = by_kind["pe_fabric"]["int8"] + by_kind["systolic"]["int8"]
    assert SKUS[node_id].performance.int8_tops == pytest.approx(
        round(total / 1e12, 1), abs=0.11
    )


def test_the_two_nodes_share_one_design():
    """The point of expressing a design in sites: moving nodes changes the
    process, the envelope and one SRAM library -- not the floorplan."""
    n16, n7 = kpu_block_of(SKUS["tsmc_n16"]), kpu_block_of(SKUS["tsmc_n7"])
    assert {t.tile_class_id: t.num_tiles for t in n16.tiles} == \
        {t.tile_class_id: t.num_tiles for t in n7.tiles}
    assert {t.tile_class_id: t.sites_per_tile for t in n16.tiles} == \
        {t.tile_class_id: t.sites_per_tile for t in n7.tiles}
    assert place_tiles(n16).site_owner() == place_tiles(n7).site_owner()


def test_n7_is_the_smaller_die():
    n16 = derive_kpu_architectural_floorplan(SKUS["tsmc_n16"], NODES["tsmc_n16"])
    n7 = derive_kpu_architectural_floorplan(SKUS["tsmc_n7"], NODES["tsmc_n7"])
    assert n7.die_area_mm2 < n16.die_area_mm2


def test_only_n16_overrides_the_accumulator_library():
    """tsmc_n7 offers sram_hp, so the tile-class library's own choice
    stands; n16 does not (branes-ai/embodied-schemas#96), so that SKU picks
    the fastest SRAM its node does offer."""
    def accumulator(sku):
        systolic = next(
            t for t in kpu_block_of(sku).tiles
            if t.tile_class_id == "systolic_int8_ws"
        )
        return next(
            m.circuit_class.value for m in systolic.local_memory
            if m.level.value == "accumulator"
        )

    from embodied_schemas.process_node import CircuitClass

    assert accumulator(SKUS["tsmc_n16"]) == "sram_hc"
    assert accumulator(SKUS["tsmc_n7"]) == "sram_hp"
    # And the reason: only one of the two nodes offers it.
    assert not NODES["tsmc_n16"].supports(CircuitClass.SRAM_HP)
    assert NODES["tsmc_n7"].supports(CircuitClass.SRAM_HP)
