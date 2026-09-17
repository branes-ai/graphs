"""Which catalog KPU SKUs the legacy contracts apply to.

embodied-schemas 0.10.0 added ``kpu_h64_auto1`` at two nodes: the first
heterogeneous KPU products in the catalog (graphs#268 E1). A family of
tests here asserts things that are true of the twelve uniform SKUs and
deliberately false of the new ones -- every tile is a PE fabric, the mesh
places exactly ``mesh_rows * mesh_cols`` compute tiles at one pitch, the
block declares no checkerboard, the resource model attaches no
fixed-function units.

Those tests were parametrized over "every KPU SKU in the catalog", which
was the same set only while the catalog held nothing else. They now scope
to ``LEGACY_KPU_SKU_IDS``.

The list is explicit rather than derived, for the same reason the mirror
list in embodied-schemas is: deriving it (say, "every SKU with one tile
kind") would make each contract vacuous, because a legacy SKU that
accidentally grew a systolic tile would drop out of its own regression
test instead of failing it.

Named ``test_*`` so pytest collects the guard below -- ``pytest.ini`` sets
``python_files = test_*.py``, so a plain ``kpu_catalog_ids.py`` would be
importable but never run.
"""

from __future__ import annotations

from embodied_schemas import load_compute_products

from graphs.hardware.kpu_access import has_kpu_block, kpu_block_of

#: The uniform KPU SKUs that predate the heterogeneous work and keep their
#: original shape.
LEGACY_KPU_SKU_IDS = (
    "kpu_t64_32x32_lp5x4_12nm_gf_fdx",
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t64_32x32_lp5x4_7nm_tsmc_hpc",
    "kpu_t128_32x32_lp5x8_12nm_gf_fdx",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_7nm_tsmc_hpc",
    "kpu_t256_32x32_lp5x16_12nm_gf_fdx",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_7nm_tsmc_hpc",
    "kpu_t512_32x32_lp5x32_12nm_gf_fdx",
    "kpu_t512_32x32_lp5x32_7nm_tsmc_hpc",
)

#: SKUs that predate the heterogeneous work and were then deliberately moved
#: onto a new tile kind: the T768, whose Matrix class became ``systolic``
#: (graphs#268 D8). Mirrors the list in embodied-schemas. They left the
#: uniform contracts on purpose, so they get their own list rather than
#: silently dropping out of the legacy one.
MIGRATED_KPU_SKU_IDS = (
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
)

#: The heterogeneous reference design (graphs#268 E1), at both its nodes.
HETEROGENEOUS_KPU_SKU_IDS = (
    "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp",
    "kpu_h64_auto1_lp5x4_7nm_tsmc_hpc",
)

#: Every SKU that shipped before the heterogeneous work, migrated or not: the
#: scope of the contracts that are about the implicit mesh rather than about
#: every tile being a PE fabric.
SHIPPED_KPU_SKU_IDS = LEGACY_KPU_SKU_IDS + MIGRATED_KPU_SKU_IDS

#: Every KPU SKU in the catalog, for tests that really do mean all of them.
ALL_KPU_SKU_IDS = tuple(
    sorted(LEGACY_KPU_SKU_IDS + MIGRATED_KPU_SKU_IDS + HETEROGENEOUS_KPU_SKU_IDS)
)


def catalog_kpu_ids() -> tuple:
    return tuple(
        sorted(k for k, v in load_compute_products().items() if has_kpu_block(v))
    )


def test_the_id_lists_cover_the_catalog_and_do_not_overlap():
    """A new KPU SKU must be classified deliberately, not fall through.

    If this fails, someone added a SKU without deciding whether the legacy
    contracts apply to it.
    """
    catalog = set(catalog_kpu_ids())
    lists = (
        set(LEGACY_KPU_SKU_IDS), set(MIGRATED_KPU_SKU_IDS), set(HETEROGENEOUS_KPU_SKU_IDS)
    )
    assert sum(len(ids) for ids in lists) == len(set().union(*lists)), "a SKU is in two lists"
    classified = set().union(*lists)
    assert classified == catalog, (
        f"unclassified KPU SKUs: {sorted(catalog - classified)}; "
        f"missing from the catalog: {sorted(classified - catalog)}"
    )


def test_the_lists_say_what_they_claim_about_tile_kinds():
    """The classification is not just a label: a legacy SKU really is one
    kind, and a heterogeneous one really is more than one. Without this the
    lists could drift from reality and every contract scoped to them would
    quietly stop testing what it says.
    """
    products = load_compute_products()
    for sku in LEGACY_KPU_SKU_IDS:
        kinds = {t.tile_kind.value for t in kpu_block_of(products[sku]).tiles}
        assert kinds == {"pe_fabric"}, f"{sku} is no longer uniform: {kinds}"
    for sku in MIGRATED_KPU_SKU_IDS:
        block = kpu_block_of(products[sku])
        kinds = {t.tile_kind.value for t in block.tiles}
        assert len(kinds) > 1, f"{sku} is not migrated: {kinds}"
        assert block.checkerboard is None, f"{sku} left its implicit mesh"
    for sku in HETEROGENEOUS_KPU_SKU_IDS:
        kinds = {t.tile_kind.value for t in kpu_block_of(products[sku]).tiles}
        assert len(kinds) > 1, f"{sku} is no longer heterogeneous: {kinds}"
