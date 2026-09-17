"""The T768 Matrix class as a systolic tile (graphs#268 D8).

embodied-schemas 0.14.0 rewrites the T768's ``Matrix`` class from a PE-fabric
tile with a systolic throughput figure to a ``systolic`` tile that states the
mechanism: 8x8 cells of 64 MAC lanes (plus its own L1 / L2, which a systolic
tile does not inherit). Throughput and energy basis are unchanged by choice.

The migration moves the SKU onto the heterogeneous code paths -- the per-kind
power engine, capability-aware tile pools, tile-carried silicon -- so these
tests check each path against the legacy one on the *same chip written the
old way* (``_as_pe_fabric``), rather than against numbers copied from a
golden. What may differ, and by how much, is stated per test.

It also pins the floorplan fix the migration exposed: the shared L2 pool is
divided among the tiles that inherit it and L3 among the shared memory
cells, where both used to divide by ``total_tiles``.
"""

from __future__ import annotations

import math

import pytest
from embodied_schemas import load_compute_products, load_process_nodes

from graphs.hardware import silicon_floorplan as sf
from graphs.hardware.kpu_engines import describe_engines
from graphs.hardware.kpu_power_model import (
    compute_heterogeneous_tdp_breakdown,
    compute_thermal_profile_tdp_breakdown,
    is_legacy_shaped,
)
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.kpu_sku_input import KPUSKUInputSpec
from graphs.hardware.mappers.accelerators.kpu_tile_pools import build_tile_pool
from graphs.hardware.models.accelerators.kpu_yaml_loader import (
    load_kpu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import Precision
from graphs.hardware.sku_validators import (
    build_context_for_kpu,
    default_registry,
    load_validators,
)
from graphs.hardware.sku_validators import silicon_math as sm
from hardware.test_kpu_catalog_ids import (
    ALL_KPU_SKU_IDS,
    HETEROGENEOUS_KPU_SKU_IDS,
    LEGACY_KPU_SKU_IDS,
    MIGRATED_KPU_SKU_IDS,
)

T768 = "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc"
CATALOG = load_compute_products()
NODES = load_process_nodes()
NODE = NODES["tsmc_n7"]
load_validators()
DYNAMIC_TERMS = ("pe_compute_w", "l2_sram_w", "l3_sram_w", "noc_w", "dram_phy_w")


def test_the_t768_is_the_migrated_sku():
    assert MIGRATED_KPU_SKU_IDS == (T768,)


def _as_pe_fabric(spec: KPUSKUInputSpec) -> KPUSKUInputSpec:
    """The same chip with Matrix written as the PE-fabric tile it shipped
    as: the systolic tile's throughput as ``ops_per_tile_per_clock``, and no
    local memory (so it inherits the chip's L1 / L2 again)."""
    data = spec.model_dump(mode="json")
    tiles = data["kpu_architecture"]["tiles"]
    (i,) = [k for k, t in enumerate(tiles) if t["tile_type"] == "Matrix"]
    matrix = next(t for t in spec.kpu_architecture.tiles if t.tile_type == "Matrix")
    tiles[i] = {
        "tile_type": "Matrix",
        "num_tiles": matrix.num_tiles,
        "pe_array_rows": matrix.array_rows,
        "pe_array_cols": matrix.array_cols,
        "pe_circuit_class": matrix.circuit_class.value,
        "ops_per_tile_per_clock": dict(matrix.ops_per_tile_per_clock),
        "schedule_class": matrix.dataflow.value,
        "pipeline_fill_cycles": matrix.pipeline_fill_cycles,
        "pipeline_drain_cycles": matrix.pipeline_drain_cycles,
    }
    return KPUSKUInputSpec.model_validate(data)


@pytest.fixture(scope="module")
def spec() -> KPUSKUInputSpec:
    return input_spec_from_compute_product(CATALOG[T768])


@pytest.fixture(scope="module")
def legacy_spec(spec) -> KPUSKUInputSpec:
    return _as_pe_fabric(spec)


# ---------------------------------------------------------------------------
# Power: same energy basis, so the same dynamic power
# ---------------------------------------------------------------------------


def test_the_equivalent_is_legacy_shaped_and_the_migrated_sku_is_not(spec, legacy_spec):
    for profile in spec.thermal_profiles:
        assert not is_legacy_shaped(spec, profile)
        assert is_legacy_shaped(legacy_spec, profile)


def test_dynamic_power_is_the_legacy_formula_exactly(spec, legacy_spec):
    """Node-anchor energy (no systolic discount) means every dynamic term
    matches the legacy formula on the same chip, at every profile."""
    for profile in spec.thermal_profiles:
        het = compute_thermal_profile_tdp_breakdown(spec, profile, NODE, nodes=NODES)
        legacy = compute_thermal_profile_tdp_breakdown(legacy_spec, profile, NODE, nodes=NODES)
        assert het.worst_precision == legacy.worst_precision, profile.name
        for term in DYNAMIC_TERMS:
            assert getattr(het, term) == pytest.approx(getattr(legacy, term), rel=1e-12), (
                profile.name, term,
            )
        assert het.fixed_function_w == 0.0


def test_leakage_moves_only_by_the_matrix_memory_pricing(spec, legacy_spec):
    """The one power change: Matrix's own 4 + 32 KiB L1 / L2 is tile-carried
    silicon, priced at SRAM_MTX_PER_KIB (0.052 Mtx/KiB), where the chip's L2
    block prices this SKU's L2 at 0.022. At nominal Vdd the leakage delta is
    exactly that transistor difference in sram_hd."""
    cp = CATALOG[T768]
    l2_block = next(b for b in cp.dies[0].silicon_bin.blocks if b.name == "l2_sram")
    matrix = next(t for t in spec.kpu_architecture.tiles if t.tile_type == "Matrix")
    kib = spec.kpu_architecture.memory.l2_kib_per_tile
    extra_mtx = matrix.num_tiles * kib * (
        sm.SRAM_MTX_PER_KIB[l2_block.circuit_class] - l2_block.transistor_source.per_unit_mtx
    )
    density = NODE.density_for(l2_block.circuit_class).mtx_per_mm2
    expected_w = extra_mtx / density * NODE.leakage_w_per_mm2[l2_block.circuit_class]
    nominal = next(p for p in spec.thermal_profiles).model_copy(
        update={"vdd_v": NODE.nominal_vdd_v}
    )
    het = compute_heterogeneous_tdp_breakdown(spec, nominal, NODE, nodes=NODES)
    legacy = compute_thermal_profile_tdp_breakdown(legacy_spec, nominal, NODE, nodes=NODES)
    assert expected_w > 0
    assert het.leakage_w - legacy.leakage_w == pytest.approx(expected_w, rel=1e-9)
    # Small enough that no profile's declared TDP needs re-tuning.
    for profile in spec.thermal_profiles:
        bd = compute_thermal_profile_tdp_breakdown(spec, profile, NODE, nodes=NODES)
        assert abs(bd.total_tdp_w - profile.tdp_watts) < 0.05, profile.name


# ---------------------------------------------------------------------------
# Silicon and floorplan
# ---------------------------------------------------------------------------


def test_silicon_changes_only_by_the_matrix_memory(spec, legacy_spec):
    """Logic is the same pe_matrix block; the L1 / L2 moves from the chip's
    blocks onto the tile."""
    cp, legacy_cp = CATALOG[T768], _product(legacy_spec)
    matrix = next(t for t in spec.kpu_architecture.tiles if t.tile_type == "Matrix")
    mem = spec.kpu_architecture.memory
    assert sm.total_l1_kib(legacy_cp) - sm.total_l1_kib(cp) == matrix.num_tiles * mem.l1_kib_per_tile
    assert sm.total_l2_kib(legacy_cp) - sm.total_l2_kib(cp) == matrix.num_tiles * mem.l2_kib_per_tile
    carried = {c.name: c for c in sm.carried_silicon(cp)}
    assert set(carried) == {"matrix.memory.l1", "matrix.memory.l2"}
    assert all(c.source == "local_memory" for c in carried.values())
    assert sm.carried_silicon(legacy_cp) == []


def _product(spec: KPUSKUInputSpec):
    """A ComputeProduct for ``spec`` whose tiles are ``spec``'s -- the T768
    catalog entry with its KPU block swapped."""
    data = CATALOG[T768].model_dump(mode="json")
    block = data["dies"][0]["blocks"][0]
    block["tiles"] = spec.model_dump(mode="json")["kpu_architecture"]["tiles"]
    data["performance"] = {
        k: v for k, v in data["performance"].items()
        if k in ("int8_tops", "bf16_tflops", "fp32_tflops", "int4_tops")
    }
    return type(CATALOG[T768]).model_validate(data)


def test_pe_fabric_tiles_keep_their_floorplan_memory(spec, legacy_spec):
    """The shared L2 pool shrinks with the tiles that draw on it, so an
    INT8- or BF16-primary tile's L2 area is what it was before Matrix left
    the pool. Dividing by total_tiles understated it by 77/768."""
    new = sf.derive_kpu_architectural_floorplan(CATALOG[T768], NODE)
    old = sf.derive_kpu_architectural_floorplan(_product(legacy_spec), NODE)
    for cls in ("INT8-primary", "BF16-primary"):
        assert new.compute_summaries[cls].l2_area_mm2 == pytest.approx(
            old.compute_summaries[cls].l2_area_mm2, rel=1e-12
        )
        assert new.compute_summaries[cls].pe_area_mm2 == pytest.approx(
            old.compute_summaries[cls].pe_area_mm2, rel=1e-12
        )
    assert new.memory_summary.l3_area_mm2 == pytest.approx(
        old.memory_summary.l3_area_mm2, rel=1e-12
    )


def test_matrix_keeps_its_l3_cell_in_the_circuit_view():
    """The circuit view folds each tile's L3 cell into the tile. A class that
    carries its own memory used to lose that L3 share entirely."""
    cp = CATALOG[T768]
    fp = sf.derive_kpu_floorplan(cp, NODE)
    pe_by_type, l2, l3, _ = sf._classify_silicon_bin_blocks(cp, NODE)
    l3_per_cell = l3 / sm.l3_memory_cells(cp)
    carried = sum(
        c.transistors_mtx / NODE.density_for(c.circuit_class).mtx_per_mm2
        for c in sm.carried_silicon(cp) if c.tile_class_id == "matrix"
    ) / 77
    assert fp.tile_pitches["Matrix"].sram_area_mm2 == pytest.approx(carried + l3_per_cell)
    int8 = fp.tile_pitches["INT8-primary"].sram_area_mm2
    assert int8 == pytest.approx(l2 / (537 + 154) + l3_per_cell)


@pytest.mark.parametrize("sku", HETEROGENEOUS_KPU_SKU_IDS)
def test_shared_memory_divides_by_who_draws_on_it(sku):
    """On the H64 the three divisors all differ: 45 tiles, 38 of them
    inheriting chip L2, and 44 shared L3 cells (64 sites less the 20 the
    systolic, SGM and VIO footprints absorb)."""
    cp = CATALOG[sku]
    block = cp.dies[0].blocks[0]
    inheriting = sum(t.num_tiles for t in block.tiles if sm.inherits_chip_memory(t))
    cells = sm.l3_memory_cells(cp)
    assert len({block.total_tiles, inheriting, cells}) == 3
    _, l2, l3, _ = sf._classify_silicon_bin_blocks(cp, NODES[cp.dies[0].process_node_id])
    shares = sf._shared_memory_shares(cp, l2, l3)
    assert shares.l2_per_tile == pytest.approx(l2 / inheriting)
    assert shares.l3_per_cell == pytest.approx(l3 / cells)
    by_type = {t.tile_type: t for t in block.tiles}
    # A 1x1 fixed-function core that does not absorb its cell still sits
    # beside one; an absorbing footprint has none; no L2 for either.
    assert shares.l3_for(by_type["ISP"]) == pytest.approx(l3 / cells)
    assert shares.l3_for(by_type["VIO"]) == 0.0
    assert shares.l2_for(by_type["ISP"]) == 0.0


@pytest.mark.parametrize("sku", ALL_KPU_SKU_IDS)
def test_architectural_blocks_charge_the_shared_l3_pool_exactly_once(sku):
    """Every mm^2 of the chip's shared L3 lands on exactly one block: a
    memory cell, or the multi-site tile covering it. An absorbing footprint's
    cells are tile-local memory, not shared L3, and used to be charged anyway
    -- about 4.4 mm^2 too much on the 16 nm H64 (CodeRabbit on #295)."""
    cp = CATALOG[sku]
    node = NODES[cp.dies[0].process_node_id]
    fp = sf.derive_kpu_architectural_floorplan(cp, node)
    _, _, l3, _ = sf._classify_silicon_bin_blocks(cp, node)
    charged = sum(b.l3_area_mm2 or 0.0 for b in fp.blocks)
    assert charged == pytest.approx(l3, rel=1e-12)


@pytest.mark.parametrize("sku", LEGACY_KPU_SKU_IDS)
def test_shared_memory_is_the_old_division_on_uniform_skus(sku):
    """Tiles, inheriting tiles and L3 cells are one number on a uniform
    SKU, which is why the golden never showed the divisor."""
    cp = CATALOG[sku]
    block = cp.dies[0].blocks[0]
    _, l2, l3, _ = sf._classify_silicon_bin_blocks(cp, NODES[cp.dies[0].process_node_id])
    shares = sf._shared_memory_shares(cp, l2, l3)
    assert shares.l2_per_tile == pytest.approx(l2 / block.total_tiles, rel=1e-15)
    assert shares.l3_per_cell == pytest.approx(l3 / block.total_tiles, rel=1e-15)


# ---------------------------------------------------------------------------
# Mapper, engines, validators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resource_model():
    return load_kpu_resource_model_from_yaml(T768)


def _pool(rm, precision, prefer_systolic=False):
    tp = rm.thermal_operating_points[rm.default_thermal_profile]
    return build_tile_pool(
        rm, tp.performance_specs[precision].compute_resource, precision,
        rm.default_thermal_profile, prefer_systolic=prefer_systolic,
    )


def test_a_dense_product_takes_the_matrix_cells_first(resource_model):
    assert resource_model.is_heterogeneous_kpu
    pool = _pool(resource_model, Precision.INT8, prefer_systolic=True)
    assert pool.capable and pool.preferred_kind == "systolic"
    first = pool.specializations[0]
    assert (first.tile_type, first.pe_count, first.num_tiles) == ("Matrix", 64, 77)
    # 128 PEs of work fills exactly two Matrix tiles, not one PE-fabric tile.
    alloc = pool.allocate(128)
    assert alloc.classes == ("Matrix",) and alloc.num_tiles == 2 and alloc.threads == 128


@pytest.mark.parametrize("precision, classes", [
    (Precision.FP32, {"BF16-primary"}),
    (Precision.INT4, {"INT8-primary"}),
])
def test_formats_matrix_cannot_run_never_reach_it(resource_model, precision, classes):
    """The flat mapper handed FP32 and INT4 work to all 768 tiles."""
    pool = _pool(resource_model, precision, prefer_systolic=True)
    assert pool.capable
    assert {s.tile_type for s in pool.specializations} == classes


def test_matrix_is_a_systolic_engine():
    cp = CATALOG[T768]
    engines = {e.tile_type: e for e in describe_engines(cp, NODE, NODES)}
    matrix = engines["Matrix"]
    assert matrix.engine_kind == "systolic"
    assert set(matrix.supported_kernels) == {"gemm", "conv2d", "attention"}
    rates = {p.operand_format: p.ops_per_tile_per_clock for p in matrix.precisions}
    assert rates == {"int8": 8192.0, "bf16": 4096.0, "fp16": 4096.0}
    assert {engines[t].engine_kind for t in ("INT8-primary", "BF16-primary")} == {"pe_fabric"}


def test_the_matrix_pitch_is_now_named_by_the_footprint_validator():
    """The shipped SKU already paid for Matrix's size as die whitespace (the
    whitespace and pitch-match findings). As a systolic class it is also
    checked against the PE-fabric site pitch -- which it exceeds ~5.9x --
    and the footprint remedy needs an explicit checkerboard the T768 does
    not declare. Pinned so the finding is a known one, not a regression."""
    ctx = build_context_for_kpu(T768)
    fit = default_registry._validators["tile_footprint_pitch_fit"].check(ctx)
    assert [f.block for f in fit] == ["matrix"]
    need = float(fit[0].message.split("about ")[1].split(" sites")[0])
    assert 5.0 < need < 7.0
    drift = default_registry._validators["declared_tdp_matches_model"].check(ctx)
    assert drift == []


def test_silicon_dynamic_power_covers_every_supported_format():
    from graphs.hardware import kpu_golden as kg

    snap = kg.load_snapshot(T768)
    formats = {"int8", "int4", "bf16", "fp16", "fp32"}
    for block in snap["silicon"]["blocks"]:
        for profile, by_prec in block["peak_dynamic_w_by_profile"].items():
            assert set(by_prec) == formats, (block["name"], profile)
    assert not math.isnan(snap["generator"]["transistors_billion"])
