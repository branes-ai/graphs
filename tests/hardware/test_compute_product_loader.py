"""Tests for the unified ComputeProduct loader + KPUEntry adapter.

PR #3 of the v1 ComputeProduct migration POC. Verifies:

  1. The adapter mechanically converts every legacy KPUEntry in the
     catalog to a content-equivalent ComputeProduct
  2. The unified loader combines both sources and returns the full
     12-SKU catalog as ComputeProduct instances
  3. For SKUs in both catalogs (t256 today), prefer_native=True returns
     the native ComputeProduct and prefer_native=False returns the
     adapted KPUEntry
  4. The native and adapted t256 are content-equivalent
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    BlockKind,
    ComputeProduct,
    DieRole,
    KPUEntry,
    LifecycleStatus,
    PackagingKind,
    ProductKind,
    load_compute_products,
    load_kpus,
)

from graphs.hardware.compute_product_loader import (
    get_compute_product,
    kpu_entry_to_compute_product,
    load_compute_products_unified,
)


T256_ID = "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"


@pytest.fixture(scope="module")
def kpus():
    return load_kpus()


@pytest.fixture(scope="module")
def native_cps():
    return load_compute_products()


# ---------------------------------------------------------------------------
# 1. Adapter -- every legacy KPUEntry mechanically converts
# ---------------------------------------------------------------------------

def test_adapter_handles_every_legacy_kpu(kpus):
    """Every SKU in the legacy KPU catalog converts to ComputeProduct
    without raising. Proves the adapter covers the full catalog."""
    for sku_id, entry in kpus.items():
        cp = kpu_entry_to_compute_product(entry)
        assert isinstance(cp, ComputeProduct), (
            f"adapter returned non-ComputeProduct for {sku_id}"
        )
        assert cp.id == entry.id


@pytest.mark.parametrize("sku_id", [
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
])
def test_adapter_preserves_content(sku_id, kpus):
    """The adapter does not lose information. Verifies field-by-field
    that the resulting ComputeProduct carries the same data as the
    KPUEntry it was converted from."""
    entry = kpus[sku_id]
    cp = kpu_entry_to_compute_product(entry)

    # Identity
    assert cp.id == entry.id
    assert cp.name == entry.name
    assert cp.vendor == entry.vendor
    assert cp.kind == ProductKind.CHIP
    assert cp.packaging.kind == PackagingKind.MONOLITHIC
    assert cp.packaging.num_dies == entry.die.num_dies

    # Per-die structure
    assert len(cp.dies) == 1
    die = cp.dies[0]
    assert die.die_id == "kpu_compute"
    assert die.die_role == DieRole.COMPUTE
    assert die.process_node_id == entry.process_node_id
    assert die.die_size_mm2 == entry.die.die_size_mm2
    assert die.transistors_billion == entry.die.transistors_billion
    assert die.silicon_bin == entry.silicon_bin
    assert die.clocks == entry.clocks
    assert die.interconnects == []  # monolithic: no inter-die links

    # KPUBlock
    assert len(die.blocks) == 1
    block = die.blocks[0]
    assert block.kind == BlockKind.KPU
    assert block.total_tiles == entry.kpu_architecture.total_tiles
    assert block.tiles == entry.kpu_architecture.tiles
    assert block.noc == entry.kpu_architecture.noc
    assert block.memory == entry.kpu_architecture.memory

    # Roll-ups
    assert cp.performance == entry.performance
    assert cp.power.tdp_watts == entry.power.tdp_watts
    assert cp.power.thermal_profiles == entry.power.thermal_profiles

    # Lifecycle derived from is_discontinued
    expected_lifecycle = (
        LifecycleStatus.EOL if entry.market.is_discontinued
        else LifecycleStatus.PRODUCTION
    )
    assert cp.lifecycle == expected_lifecycle


# ---------------------------------------------------------------------------
# 2. Unified loader -- full catalog as ComputeProduct
# ---------------------------------------------------------------------------

def test_unified_loader_returns_full_catalog(kpus):
    """Unified loader returns every SKU from the legacy catalog plus
    any extras only in the native catalog. Today the legacy catalog is
    the superset (12 SKUs); the native catalog has 1 (t256). Unified
    must contain all 12."""
    unified = load_compute_products_unified()
    legacy_ids = set(kpus.keys())
    unified_ids = set(unified.keys())

    assert legacy_ids.issubset(unified_ids), (
        f"unified missing SKUs from legacy: {legacy_ids - unified_ids}"
    )


def test_unified_loader_returns_compute_products_only(kpus):
    """Every value in the unified dict is a ComputeProduct, regardless
    of whether it came from the native catalog or the adapter."""
    unified = load_compute_products_unified()
    for sku_id, cp in unified.items():
        assert isinstance(cp, ComputeProduct), (
            f"unified[{sku_id!r}] is {type(cp).__name__}, not ComputeProduct"
        )


def test_unified_loader_no_silent_loss(kpus):
    """Every legacy SKU id appears in the unified catalog -- no SKU
    silently dropped during the adapter pass."""
    unified = load_compute_products_unified()
    missing = set(kpus.keys()) - set(unified.keys())
    assert not missing, f"unified loader dropped SKUs: {missing}"


# ---------------------------------------------------------------------------
# 3. Precedence: native vs adapted for SKUs in both catalogs
# ---------------------------------------------------------------------------

def test_unified_prefer_native_uses_native_cp(native_cps):
    """When a SKU lives in both catalogs (t256 today), prefer_native=True
    returns the native ComputeProduct (content equal to the native
    loader's output, not just the adapter's). Today the native and
    adapted versions are content-equivalent by construction (PR #16's
    YAML was generated mechanically from the legacy KPUEntry), so the
    assertion is weak but correct -- it verifies the precedence path
    runs at all."""
    if T256_ID not in native_cps:
        pytest.skip(
            f"{T256_ID} not in native compute_products catalog -- "
            "PR #16 not present in pinned embodied-schemas?"
        )
    unified_native = load_compute_products_unified(prefer_native=True)
    # Content equality (Pydantic ==), not identity (Python is)
    assert unified_native[T256_ID] == native_cps[T256_ID], (
        "prefer_native=True should return content equal to the native "
        "loader's output for SKUs that exist in both catalogs"
    )


def test_unified_prefer_adapted_uses_kpu_entry_adapter(native_cps, kpus):
    """When a SKU lives in both catalogs, prefer_native=False returns
    the adapted KPUEntry (useful for testing equivalence of the two)."""
    if T256_ID not in native_cps:
        pytest.skip(f"{T256_ID} not in native compute_products catalog")

    unified_adapted = load_compute_products_unified(prefer_native=False)
    adapted = unified_adapted[T256_ID]

    # The adapted version should NOT be the native instance
    assert adapted is not native_cps[T256_ID]

    # But it should be a ComputeProduct equivalent in content
    assert adapted.id == kpus[T256_ID].id


def test_native_and_adapted_t256_are_content_equivalent(native_cps, kpus):
    """The native ComputeProduct and the adapted KPUEntry for t256 carry
    identical content. This is the key proof that the adapter and the
    PR #16 YAML are consistent -- if they ever diverge, this test
    catches it."""
    if T256_ID not in native_cps:
        pytest.skip(f"{T256_ID} not in native compute_products catalog")

    native = native_cps[T256_ID]
    adapted = kpu_entry_to_compute_product(kpus[T256_ID])

    # Compare key fields field-by-field
    assert native.id == adapted.id
    assert native.name == adapted.name
    assert native.vendor == adapted.vendor
    assert native.kind == adapted.kind
    assert native.packaging.num_dies == adapted.packaging.num_dies
    assert native.lifecycle == adapted.lifecycle

    # Per-die
    assert len(native.dies) == len(adapted.dies) == 1
    nd, ad = native.dies[0], adapted.dies[0]
    assert nd.process_node_id == ad.process_node_id
    assert nd.die_size_mm2 == ad.die_size_mm2
    assert nd.transistors_billion == ad.transistors_billion
    assert nd.silicon_bin == ad.silicon_bin
    assert nd.clocks == ad.clocks
    assert len(nd.blocks) == len(ad.blocks) == 1
    assert nd.blocks[0].kind == ad.blocks[0].kind
    assert nd.blocks[0].tiles == ad.blocks[0].tiles
    assert nd.blocks[0].noc == ad.blocks[0].noc
    assert nd.blocks[0].memory == ad.blocks[0].memory

    # Roll-ups
    assert native.performance == adapted.performance
    assert native.power.tdp_watts == adapted.power.tdp_watts
    assert native.power.thermal_profiles == adapted.power.thermal_profiles
    assert native.market.target_market == adapted.market.target_market


def test_get_compute_product_helper(kpus):
    """get_compute_product(id) returns the same instance as the unified
    loader for a known SKU, and None for an unknown one."""
    cp = get_compute_product(T256_ID)
    assert cp is not None
    assert cp.id == T256_ID

    missing = get_compute_product("kpu_t99999_does_not_exist")
    assert missing is None
