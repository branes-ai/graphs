"""Tests for the unified ComputeProduct loader.

Originally (PR #156) this test file verified the dual-catalog adapter
behavior during the parallel-migration phase. After embodied-schemas
PR #18 retired the legacy ``data/kpus/`` catalog, the loader is a thin
wrapper around ``load_compute_products()`` and there's no adapter pass
to test. This file is correspondingly slim.

End-to-end coverage of the loader's downstream consumption (validator
context, mapper, generator, ...) lives in those modules' own test
suites.
"""

from __future__ import annotations

from embodied_schemas import ComputeProduct

from graphs.hardware.compute_product_loader import (
    get_compute_product,
    load_compute_products_unified,
)


T256_ID = "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"


def test_unified_loader_returns_full_catalog():
    """The catalog contains the 12 KPU SKUs we expect."""
    catalog = load_compute_products_unified()
    assert len(catalog) >= 12, (
        f"expected at least 12 SKUs in the compute_products catalog, "
        f"got {len(catalog)}"
    )
    assert T256_ID in catalog, (
        f"{T256_ID} not in catalog. Available: {sorted(catalog)}"
    )


def test_unified_loader_returns_compute_products_only():
    """Every value in the unified dict is a ComputeProduct."""
    catalog = load_compute_products_unified()
    for sku_id, cp in catalog.items():
        assert isinstance(cp, ComputeProduct), (
            f"unified[{sku_id!r}] is {type(cp).__name__}, not ComputeProduct"
        )


def test_get_compute_product_helper():
    """get_compute_product(id) returns the right SKU and None for unknown."""
    cp = get_compute_product(T256_ID)
    assert cp is not None
    assert cp.id == T256_ID

    missing = get_compute_product("kpu_t99999_does_not_exist")
    assert missing is None


def test_t256_has_expected_shape():
    """Spot-check that t256 (the canonical migration target) has the
    expected ComputeProduct shape: one Die with one KPUBlock."""
    cp = get_compute_product(T256_ID)
    assert cp is not None
    assert cp.vendor == "stillwater"
    assert len(cp.dies) == 1
    die = cp.dies[0]
    assert die.process_node_id == "tsmc_n16"
    assert len(die.blocks) == 1
    block = die.blocks[0]
    assert block.total_tiles == 256
