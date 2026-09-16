"""Phase C1 of the KPU heterogeneous-tile refactor (graphs#268): silicon_math.

- Kind-aware rollups: PEs cover pe_fabric PEs + systolic cells; chip L1 / L2
  cover only tile classes that inherit the chip memory figures; L3 counts
  checkerboard memory cells minus absorbed ones.
- ``count_ref`` resolves by tile_class_id or tile_type label.
- Dynamic power dispatches on the tile a PER_PE block resolves to.
- Tile-carried silicon (``carried_silicon``), its areas and leakage, and
  double-count detection.

Legacy SKUs are unchanged (the KPU golden snapshot pins every value); the
tests below pin the new behavior on the synthetic heterogeneous fixture.
"""

from __future__ import annotations

import pytest
from embodied_schemas import ComputeProduct, load_compute_products, load_process_nodes

from graphs.hardware.kpu_access import has_kpu_block, kpu_block_of
from graphs.hardware.kpu_hetero_fixture import (
    LNS_DATAPATH_MTX_PER_PE,
    ROW_BROADCAST_MTX_PER_INSTANCE,
    STREAM_LINK_MTX,
    SYSTOLIC_CELL_MTX,
    build_heterogeneous_kpu,
)
from graphs.hardware.sku_validators import silicon_math as sm
from hardware.test_kpu_catalog_ids import LEGACY_KPU_SKU_IDS

NODES = load_process_nodes()
N16, N7 = NODES["tsmc_n16"], NODES["tsmc_n7"]
# The legacy contract is about the uniform SKUs that predate the
# heterogeneous work; see tests/hardware/test_kpu_catalog_ids.py.
CATALOG = {
    k: v for k, v in load_compute_products().items()
    if k in LEGACY_KPU_SKU_IDS
}
HETERO = build_heterogeneous_kpu()


def _with_block(cp: ComputeProduct, mutate) -> ComputeProduct:
    data = cp.model_dump(mode="json")
    mutate(data["dies"][0])
    return ComputeProduct.model_validate(data)


def _bin_block(cp: ComputeProduct, name: str):
    return next(b for b in cp.dies[0].silicon_bin.blocks if b.name == name)


# ---------------------------------------------------------------------------
# Legacy identities (the golden snapshot pins the exact values)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(CATALOG))
def test_legacy_skus_keep_the_uniform_identities(sku):
    cp = CATALOG[sku]
    block = kpu_block_of(cp)
    assert sm.carried_silicon(cp) == []
    assert sm.double_counted_tile_classes(cp) == []
    assert sm.l3_memory_cells(cp) == block.total_tiles
    assert sm.total_l1_kib(cp) == block.memory.l1_kib_per_pe * sm.total_pe_count(cp)
    assert sm.total_l2_kib(cp) == block.memory.l2_kib_per_tile * block.total_tiles
    node = NODES[cp.dies[0].process_node_id]
    assert sm.total_chip_leakage_w(cp, node) == sum(
        sm.estimate_block_leakage_w(b, cp, node) for b in cp.dies[0].silicon_bin.blocks
    )


# ---------------------------------------------------------------------------
# Kind-aware rollups on the heterogeneous fixture
# ---------------------------------------------------------------------------


def test_rollups_by_kind():
    block = kpu_block_of(HETERO)
    mem = block.memory
    # 24 + 8 + 6 pe_fabric tiles of 32x32, 4 systolic tiles of 64x64; FF has no PEs.
    assert sm.total_pe_count(HETERO) == 38 * 1024 + 4 * 4096
    assert set(sm.total_pes_by_tile_type(HETERO)) == {
        "INT8-MAC", "LNS16-MAC", "MINPLUS-I16", "Systolic-INT8-WS",
    }
    assert sm.num_tiles_by_type(HETERO)["VIO"] == 1
    # Chip L1 / L2 cover only the pe_fabric classes without local_memory.
    assert sm.total_l1_kib(HETERO) == mem.l1_kib_per_pe * 38 * 1024
    assert sm.total_l2_kib(HETERO) == mem.l2_kib_per_tile * 38
    # 64 checkerboard cells, 4 absorbed by the 2x2 VIO footprint (spares keep theirs).
    assert sm.l3_memory_cells(HETERO) == 60
    assert sm.total_l3_kib(HETERO) == mem.l3_kib_per_tile * 60


def test_l3_cells_without_a_checkerboard_follow_the_occupied_sites():
    def drop_checkerboard(die):
        die["blocks"][0]["checkerboard"] = None
    cp = _with_block(HETERO, drop_checkerboard)
    assert sm.l3_memory_cells(cp) == 48 - 4  # sites the tiles occupy, minus absorbed


# ---------------------------------------------------------------------------
# count_ref resolution and dynamic-power dispatch
# ---------------------------------------------------------------------------


def test_tile_refs_resolve_by_id_or_label():
    assert sm.resolve_tile_ref(HETERO, "pe_int8_mac_i32").tile_type == "INT8-MAC"
    assert sm.resolve_tile_ref(HETERO, "MINPLUS-I16").tile_class_id == "pe_minplus_i16"
    with pytest.raises(sm.SiliconMathError, match="unknown tile ref 'nope'"):
        sm.resolve_tile_ref(HETERO, "nope")
    # PER_PE blocks by id (pe_int8) and by label (pe_minplus) both resolve.
    assert sm.resolve_block_transistors(_bin_block(HETERO, "pe_int8"), HETERO) == pytest.approx(
        0.006 * 24 * 1024
    )
    assert sm.resolve_block_transistors(
        _bin_block(HETERO, "pe_minplus"), HETERO
    ) == pytest.approx(0.004 * 6 * 1024)


def test_ambiguous_label_and_pe_blocks_on_fixed_function_tiles_are_errors():
    def relabel(die):
        tiles = die["blocks"][0]["tiles"]
        tiles[1]["tile_type"] = tiles[0]["tile_type"]  # two classes labeled INT8-MAC
    cp = _with_block(HETERO, relabel)
    with pytest.raises(sm.SiliconMathError, match="matches 2 tile classes by label"):
        sm.resolve_tile_ref(cp, "INT8-MAC")

    def pe_block_on_isp(die):
        die["silicon_bin"]["blocks"][0]["transistor_source"]["count_ref"] = "tile.ff_isp_raw2yuv"
    cp = _with_block(HETERO, pe_block_on_isp)
    with pytest.raises(sm.SiliconMathError, match="is fixed_function and has no PEs"):
        sm.resolve_block_transistors(_bin_block(cp, "pe_int8"), cp)
    assert sm.estimate_block_peak_dynamic_w(
        _bin_block(cp, "pe_int8"), cp, N16, clock_mhz=500, precision="int8"
    ) == 0.0


def test_dynamic_power_dispatches_on_the_tile_not_the_name():
    int8 = _bin_block(HETERO, "pe_int8")
    dyn = sm.estimate_block_peak_dynamic_w(int8, HETERO, N16, clock_mhz=500, precision="int8")
    expected = 24 * 1024 * 2 * 500e6 * N16.energy_per_op_pj["balanced_logic:int8"] * 1e-12
    assert dyn == pytest.approx(expected)
    # Min-plus tiles have no int8 ops: no PE dynamic power at int8.
    minplus = _bin_block(HETERO, "pe_minplus")
    assert sm.estimate_block_peak_dynamic_w(minplus, HETERO, N16, clock_mhz=500) == 0.0

    # A PER_PE block without the pe_ prefix still gets PE dynamic power.
    def rename(die):
        die["silicon_bin"]["blocks"][0]["name"] = "int8_array"
    cp = _with_block(HETERO, rename)
    renamed = _bin_block(cp, "int8_array")
    assert sm.estimate_block_peak_dynamic_w(
        renamed, cp, N16, clock_mhz=500, precision="int8"
    ) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tile-carried silicon
# ---------------------------------------------------------------------------


def test_carried_silicon_pieces():
    got = {cs.name: cs for cs in sm.carried_silicon(HETERO)}
    n40, n65 = NODES["tsmc_n40"], NODES["tsmc_n65"]
    dens = lambda node, cc: node.density_for(sm.CircuitClass(cc)).mtx_per_mm2  # noqa: E731
    expected = {
        # 1 row-broadcast link per row x 32 rows x 24 tiles
        "pe_int8_mac_i32.overlay.row_bcast": ROW_BROADCAST_MTX_PER_INSTANCE * 32 * 24,
        "pe_lns16_mac.datapath": LNS_DATAPATH_MTX_PER_PE * 8 * 1024,
        "systolic_int8_ws.cells": SYSTOLIC_CELL_MTX * 64 * 64 * 4,
        "systolic_int8_ws.memory.weight_buffer": 16 * 0.052 * 4,
        "systolic_int8_ws.memory.accumulator": 64 * 0.052 * 8 / 6 * 4,
        "ff_isp_raw2yuv.core.isp_logic": 0.17 * dens(n40, "balanced_logic"),
        "ff_isp_raw2yuv.core.isp_line_buffers": 0.19 * dens(n40, "sram_hd"),
        "ff_stereo_sgm.core.sgm_core": 10.8 * dens(n40, "balanced_logic"),
        "ff_vio_stereo_inertial.core.vio_logic": 3.70 * dens(n65, "balanced_logic"),
        "ff_vio_stereo_inertial.core.vio_sram": 12.37 * dens(n65, "sram_hd"),
        "noc_overlay.isp_sgm_vio": STREAM_LINK_MTX,
    }
    assert set(got) == set(expected)
    for name, mtx in expected.items():
        assert got[name].transistors_mtx == pytest.approx(mtx), name
    # The VIO core's silicon already includes its memory: its local_memory
    # (854 KiB of state) is not counted again.
    assert not any(n.startswith("ff_vio_stereo_inertial.memory") for n in got)
    assert got["noc_overlay.isp_sgm_vio"].tile_class_id is None


def test_core_without_silicon_contributes_its_local_memory():
    def strip_isp_silicon(die):
        isp = next(t for t in die["blocks"][0]["tiles"] if t["tile_class_id"] == "ff_isp_raw2yuv")
        isp["core"]["silicon"] = None
        isp["core"]["local_memory"] = [{"level": "line_buffer", "kib": 96}]
    cp = _with_block(HETERO, strip_isp_silicon)
    isp = [cs for cs in sm.carried_silicon(cp) if cs.tile_class_id == "ff_isp_raw2yuv"]
    assert [(cs.name, cs.source) for cs in isp] == [
        ("ff_isp_raw2yuv.memory.line_buffer", "local_memory")
    ]
    assert isp[0].transistors_mtx == pytest.approx(96 * 0.052)


def test_carried_areas_and_leakage():
    areas = {ba.name: ba for ba in sm.resolve_carried_areas(HETERO, N7)}
    assert len(areas) == len(sm.carried_silicon(HETERO))  # N7 offers every library
    for ba in areas.values():
        assert ba.area_mm2 == pytest.approx(ba.transistors_mtx / ba.density_mtx_per_mm2)
    bin_leak = sum(
        sm.estimate_block_leakage_w(b, HETERO, N7) for b in HETERO.dies[0].silicon_bin.blocks
    )
    carried_leak = sum(N7.leakage_w_per_mm2.get(ba.circuit_class, 0.0) * ba.area_mm2
                       for ba in areas.values())
    assert carried_leak > 0
    assert sm.total_chip_leakage_w(HETERO, N7) == pytest.approx(bin_leak + carried_leak)


def test_unsupported_carried_libraries_are_reported_and_skipped():
    # tsmc_n16 has no sram_hp (embodied-schemas#96): the systolic accumulator.
    unsupported = sm.unsupported_carried_silicon(HETERO, N16)
    assert [cs.name for cs in unsupported] == ["systolic_int8_ws.memory.accumulator"]
    names = {ba.name for ba in sm.resolve_carried_areas(HETERO, N16)}
    assert "systolic_int8_ws.memory.accumulator" not in names
    assert len(names) == len(sm.carried_silicon(HETERO)) - 1


def test_missing_reference_node_is_an_error():
    with pytest.raises(sm.SiliconMathError, match="reference process node 'tsmc_n40'"):
        sm.carried_silicon(HETERO, nodes={"tsmc_n65": NODES["tsmc_n65"]})


def test_double_counted_tile_classes():
    assert sm.double_counted_tile_classes(HETERO) == []

    # A chip-level PER_PE block for the LNS class, which carries its datapath.
    def count_lns_twice(die):
        die["silicon_bin"]["blocks"].append({
            "name": "pe_lns",
            "circuit_class": "balanced_logic",
            "transistor_source": {
                "kind": "per_pe", "per_unit_mtx": 0.01, "count_ref": "tile.LNS16-MAC",
            },
        })
    cp = _with_block(HETERO, count_lns_twice)
    assert sm.double_counted_tile_classes(cp) == ["pe_lns16_mac"]

