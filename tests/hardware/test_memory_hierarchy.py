"""Regression tests for the V5-1 memory-hierarchy scaffolding.

The MemoryTier dataclass + ``hw.memory_hierarchy`` derived property are
pure scaffolding -- nothing in the analyzer reads them yet. These tests
pin the contract:

1. The hierarchy is built from existing fields (l1_cache_per_unit +
   l1_bandwidth_per_unit_bps, l2_bandwidth_bps, l3_cache_total +
   l3_bandwidth_bps, main_memory + peak_bandwidth) plus per-tier
   access_latency_ns overrides.

2. Tiers are ordered innermost (smallest, fastest) to outermost.

3. Mappers without on-chip BW peaks (most of the 45+ existing mappers)
   return a hierarchy with only the DRAM tier -- backward compat.

4. Per-unit tiers store aggregate peak_bandwidth_bps (per-unit value
   * compute_units), so callers don't have to special-case per-unit
   vs shared tiers when reading the BW field.

5. access_latency_ns falls back to typical published values per tier
   class when the mapper doesn't override.
"""

from __future__ import annotations

import pytest

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import (
    create_h100_sxm5_80gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
)
from graphs.hardware.resource_model import MemoryTier


# ---------------------------------------------------------------------------
# i7-12700K: L1 (per-core) + L3 (LLC) + DRAM
# ---------------------------------------------------------------------------


def test_i7_12700k_hierarchy_is_l1_l3_dram():
    """i7-12700K has populated L1 BW (#61) and L3 BW (the LLC on x86,
    stored in l3_bandwidth_bps + l2_cache_total per the M1 schema). No
    distinct L2 hop."""
    hw = create_i7_12700k_mapper().resource_model
    tiers = hw.memory_hierarchy
    names = [t.name for t in tiers]
    assert names == ["L1", "L3", "DRAM"], (
        f"i7 hierarchy {names} != ['L1', 'L3', 'DRAM'] -- the LLC should "
        f"emit as L3 even though it's stored in l2_cache_total per the "
        f"x86 M1 schema convention."
    )


def test_i7_12700k_l1_aggregates_per_unit_bw():
    """L1 BW field stores the AGGREGATE (per-unit value * compute_units),
    so callers don't have to know whether a tier is per-unit when reading
    the BW field. Per-unit value is recoverable via
    peak_bandwidth_bps / num_units."""
    hw = create_i7_12700k_mapper().resource_model
    l1 = next(t for t in hw.memory_hierarchy if t.name == "L1")
    assert l1.is_per_unit is True
    assert l1.num_units == hw.compute_units
    expected_aggregate = hw.l1_bandwidth_per_unit_bps * hw.compute_units
    assert l1.peak_bandwidth_bps == expected_aggregate
    # Per-unit recovery
    assert l1.peak_bandwidth_bps / l1.num_units == hw.l1_bandwidth_per_unit_bps


def test_i7_12700k_total_capacity_per_unit_tier():
    """L1 total_capacity_bytes = per-core capacity * compute_units."""
    hw = create_i7_12700k_mapper().resource_model
    l1 = next(t for t in hw.memory_hierarchy if t.name == "L1")
    assert l1.total_capacity_bytes == hw.l1_cache_per_unit * hw.compute_units


def test_i7_12700k_l3_uses_l2_cache_total_capacity():
    """On x86 the LLC IS L3 and the M1 convention puts its capacity in
    ``l2_cache_total`` -- that 25 MB number must surface as the L3
    tier's capacity."""
    hw = create_i7_12700k_mapper().resource_model
    l3 = next(t for t in hw.memory_hierarchy if t.name == "L3")
    assert l3.capacity_bytes == hw.l2_cache_total
    assert l3.peak_bandwidth_bps == hw.l3_bandwidth_bps


# ---------------------------------------------------------------------------
# H100 SXM5: L1 (per-SM) + L2 (shared) + DRAM
# ---------------------------------------------------------------------------


def test_h100_sxm5_hierarchy_is_l1_l2_dram():
    """H100 has both per-SM L1 (`l1_bandwidth_per_unit_bps`) and shared
    L2 (`l2_bandwidth_bps`) populated per #61. Distinct L2 -- no L3."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    names = [t.name for t in hw.memory_hierarchy]
    assert names == ["L1", "L2", "DRAM"]


def test_h100_l2_is_shared_not_per_unit():
    """H100's 50 MB L2 is a single shared structure across all SMs."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    l2 = next(t for t in hw.memory_hierarchy if t.name == "L2")
    assert l2.is_per_unit is False
    assert l2.num_units == 1
    assert l2.total_capacity_bytes == l2.capacity_bytes


# ---------------------------------------------------------------------------
# Jetson Orin Nano: L1 (per-SM) + L2 (shared) + DRAM (Super, 102 GB/s)
# ---------------------------------------------------------------------------


def test_jetson_orin_nano_hierarchy_is_three_tier():
    """V5 follow-up (this PR): Orin Nano now populates
    ``l1_bandwidth_per_unit_bps`` (Ampere SM 8.6 spec, 83 GB/s/SM at
    650 MHz sustained) and ``l2_bandwidth_bps`` (204 GB/s, conservative
    2x DRAM Ampere estimate). The hierarchy is now L1 -> L2 -> DRAM,
    which lets the V5-3b eligibility predicate engage the tier-aware
    path on Orin Nano. Pre-followup the hierarchy was DRAM-only and
    the predicate declined the flag-on path."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    names = [t.name for t in hw.memory_hierarchy]
    assert names == ["L1", "L2", "DRAM"], (
        f"Orin Nano hierarchy {names} != ['L1', 'L2', 'DRAM']. If on-chip "
        f"BW peaks were unset, update both this test AND the V5 follow-up "
        f"calibration tests in test_tier_achievable_fractions.py."
    )


def test_jetson_orin_nano_dram_is_super_102gbps():
    """Verify post-#94 Super calibration is reflected in the DRAM tier."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    dram = next(t for t in hw.memory_hierarchy if t.name == "DRAM")
    assert dram.peak_bandwidth_bps == 102e9


# ---------------------------------------------------------------------------
# Backward compat: mapper without on-chip BW peaks returns DRAM-only
# ---------------------------------------------------------------------------


def test_mapper_without_onchip_bw_peaks_returns_dram_only():
    """The vast majority of 45+ existing mappers don't populate the
    on-chip BW peaks (#61's fields are Optional). Their hierarchy must
    still be valid -- just DRAM-only."""
    hw = create_i7_12700k_mapper().resource_model
    # Mutate a copy to clear the on-chip BW peaks, simulating an
    # un-augmented mapper:
    import copy

    hw2 = copy.copy(hw)
    hw2.l1_bandwidth_per_unit_bps = None
    hw2.l2_bandwidth_bps = None
    hw2.l3_bandwidth_bps = None

    tiers = hw2.memory_hierarchy
    assert len(tiers) == 1
    assert tiers[0].name == "DRAM"
    assert tiers[0].peak_bandwidth_bps == hw2.peak_bandwidth


# ---------------------------------------------------------------------------
# Ordering: innermost (smallest) to outermost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mapper_factory",
    [
        create_i7_12700k_mapper,
        create_h100_sxm5_80gb_mapper,
        create_jetson_orin_nano_8gb_mapper,
    ],
)
def test_hierarchy_ordered_innermost_first(mapper_factory):
    """The contract: tiers[0] is innermost (smallest total capacity);
    tiers[-1] is DRAM. The tier_picker (V5-3) walks innermost-out."""
    hw = mapper_factory().resource_model
    tiers = hw.memory_hierarchy
    assert tiers[-1].name == "DRAM"
    # Total capacity strictly increases (innermost is smallest)
    capacities = [t.total_capacity_bytes for t in tiers]
    assert capacities == sorted(capacities), (
        f"hierarchy {[t.name for t in tiers]} not ordered by capacity: " f"{capacities}"
    )


# ---------------------------------------------------------------------------
# access_latency_ns: defaults + mapper override
# ---------------------------------------------------------------------------


def test_access_latency_falls_back_to_typical_defaults():
    """When a mapper doesn't set ``l1_access_latency_ns`` etc., the
    property falls back to typical-tier-class defaults (1.5 / 10 / 30 /
    100 ns for L1 / L2 / L3 / DRAM)."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    # H100 mapper doesn't override latencies as of V5-1
    assert hw.l1_access_latency_ns is None
    tiers = {t.name: t for t in hw.memory_hierarchy}
    assert tiers["L1"].access_latency_ns == 1.5
    assert tiers["L2"].access_latency_ns == 10.0
    assert tiers["DRAM"].access_latency_ns == 100.0


def test_access_latency_respects_mapper_override():
    """A mapper that sets ``l1_access_latency_ns`` etc. wins over the
    typical defaults."""
    hw = create_i7_12700k_mapper().resource_model
    import copy

    hw2 = copy.copy(hw)
    hw2.l1_access_latency_ns = 0.8  # custom override
    hw2.dram_access_latency_ns = 75.0

    tiers = {t.name: t for t in hw2.memory_hierarchy}
    assert tiers["L1"].access_latency_ns == 0.8
    assert tiers["DRAM"].access_latency_ns == 75.0
    # L3 wasn't overridden -> falls back to default
    assert tiers["L3"].access_latency_ns == 30.0


# ---------------------------------------------------------------------------
# Effective BW = peak * achievable_fraction
# ---------------------------------------------------------------------------


def test_effective_bandwidth_defaults_to_peak():
    """V5-1 ships with achievable_fraction=1.0 (V5-5 will calibrate
    these per (hw, tier)). effective_bandwidth_bps = peak * 1.0 = peak."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    for tier in hw.memory_hierarchy:
        assert tier.achievable_fraction == 1.0
        assert tier.effective_bandwidth_bps == tier.peak_bandwidth_bps


def test_effective_bandwidth_scales_with_achievable_fraction():
    """Property contract: effective = peak * fraction."""
    t = MemoryTier(
        name="L2",
        capacity_bytes=1 << 22,
        is_per_unit=False,
        num_units=1,
        peak_bandwidth_bps=1e12,
        access_latency_ns=10.0,
        achievable_fraction=0.6,
    )
    assert t.effective_bandwidth_bps == 6e11


# ---------------------------------------------------------------------------
# __post_init__: physical invariant validation
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides):
    """Baseline-valid MemoryTier kwargs; tests override one field at a
    time to exercise each invariant check."""
    base = dict(
        name="L1",
        capacity_bytes=32 * 1024,
        is_per_unit=True,
        num_units=8,
        peak_bandwidth_bps=500e9,
        access_latency_ns=1.5,
        achievable_fraction=0.85,
    )
    base.update(overrides)
    return base


def test_post_init_rejects_empty_name():
    with pytest.raises(ValueError, match="name must be non-empty"):
        MemoryTier(**_valid_kwargs(name=""))


def test_post_init_rejects_negative_capacity():
    with pytest.raises(ValueError, match="capacity_bytes must be >= 0"):
        MemoryTier(**_valid_kwargs(capacity_bytes=-1))


def test_post_init_rejects_zero_or_negative_num_units():
    with pytest.raises(ValueError, match="num_units must be >= 1"):
        MemoryTier(**_valid_kwargs(num_units=0))
    with pytest.raises(ValueError, match="num_units must be >= 1"):
        MemoryTier(**_valid_kwargs(num_units=-3))


def test_post_init_rejects_negative_bandwidth():
    with pytest.raises(ValueError, match="peak_bandwidth_bps must be >= 0"):
        MemoryTier(**_valid_kwargs(peak_bandwidth_bps=-1.0))


def test_post_init_rejects_negative_latency():
    with pytest.raises(ValueError, match="access_latency_ns must be >= 0"):
        MemoryTier(**_valid_kwargs(access_latency_ns=-0.5))


def test_post_init_rejects_achievable_fraction_above_one():
    """A tier achieving > 100% of its peak BW would be physically
    impossible -- cache hits reflect a *different* tier's BW, not this
    one's peak * a >1 factor."""
    with pytest.raises(
        ValueError, match=r"achievable_fraction must be in \[0\.0, 1\.0\]"
    ):
        MemoryTier(**_valid_kwargs(achievable_fraction=1.5))


def test_post_init_rejects_negative_achievable_fraction():
    with pytest.raises(
        ValueError, match=r"achievable_fraction must be in \[0\.0, 1\.0\]"
    ):
        MemoryTier(**_valid_kwargs(achievable_fraction=-0.1))


def test_post_init_accepts_zero_bandwidth_and_zero_latency():
    """Zero is a valid sentinel (e.g., a tier in test fixtures); only
    negatives are rejected."""
    t = MemoryTier(
        **_valid_kwargs(
            peak_bandwidth_bps=0.0, access_latency_ns=0.0, achievable_fraction=0.0
        )
    )
    assert t.effective_bandwidth_bps == 0.0


def test_post_init_error_messages_name_the_field_and_tier():
    """Error messages include the tier name + offending field so a
    miscalibrated mapper traces back to the source quickly."""
    with pytest.raises(ValueError) as exc_info:
        MemoryTier(**_valid_kwargs(name="DRAM", peak_bandwidth_bps=-100e9))
    msg = str(exc_info.value)
    assert "MemoryTier('DRAM')" in msg
    assert "peak_bandwidth_bps" in msg
    assert "-100" in msg


def test_post_init_does_not_break_real_mapper_construction():
    """Sanity: the existing reference mappers still construct cleanly
    after adding the invariant checks. (i7 / H100 / Orin all parse the
    hierarchy without raising.)"""
    for factory in (
        create_i7_12700k_mapper,
        create_h100_sxm5_80gb_mapper,
        create_jetson_orin_nano_8gb_mapper,
    ):
        hw = factory().resource_model
        # Just access the property; any tier with bad values would raise
        tiers = hw.memory_hierarchy
        assert len(tiers) >= 1
