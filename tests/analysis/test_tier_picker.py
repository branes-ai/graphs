"""Unit tests for the V5-3b tier-picking algorithm.

Locks in:
- normalize_dtype maps PyTorch / FX strings to the reuse_models short form
- pick_binding_tier walks innermost-out and returns the next-larger tier
  as the binding tier; correctly handles the edge cases (empty hierarchy,
  single-tier DRAM-only, residency overflows even DRAM)
- For matmul, the tile chosen at the residency tier matches the optimal-square
  formula (already covered by V5-3a, sanity-check the integration here)
- Bytes loaded come back as ints and equal the per-op model output
"""

import pytest

from graphs.estimation.reuse_models import (
    MatmulReuseModel,
    VectorAddReuseModel,
)
from graphs.estimation.tier_picker import (
    BindingTierResult,
    normalize_dtype,
    pick_binding_tier,
)
from graphs.hardware.resource_model import MemoryTier


# ---------------------------------------------------------------------------
# normalize_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("float32", "fp32"),
        ("float16", "fp16"),
        ("bfloat16", "bf16"),
        ("float64", "fp64"),
        ("torch.float32", "fp32"),
        ("torch.bfloat16", "bf16"),
        # Pass-through for already-short forms
        ("fp32", "fp32"),
        ("bf16", "bf16"),
        # Pass-through for unknown -- the reuse_models bytes_per_element
        # call is what raises later in the pipeline
        ("complex64", "complex64"),
    ],
)
def test_normalize_dtype(raw, expected):
    assert normalize_dtype(raw) == expected


# ---------------------------------------------------------------------------
# Hierarchy fixtures
# ---------------------------------------------------------------------------


def _i7_like_hierarchy():
    """A 3-tier hierarchy resembling the i7-12700K: small per-core L1,
    aggregate LLC, then DRAM."""
    return [
        MemoryTier(
            name="L1",
            capacity_bytes=32 * 1024,
            is_per_unit=True,
            num_units=16,
            peak_bandwidth_bps=16 * 200e9,  # 16 cores * 200 GB/s per core
            access_latency_ns=1.5,
        ),
        MemoryTier(
            name="L3",
            capacity_bytes=25 * 1024 * 1024,  # 25 MB LLC
            is_per_unit=False,
            num_units=1,
            peak_bandwidth_bps=200e9,
            access_latency_ns=30.0,
        ),
        MemoryTier(
            name="DRAM",
            capacity_bytes=64 * 1024**3,
            is_per_unit=False,
            num_units=1,
            peak_bandwidth_bps=75e9,
            access_latency_ns=100.0,
        ),
    ]


def _dram_only_hierarchy():
    return [
        MemoryTier(
            name="DRAM",
            capacity_bytes=64 * 1024**3,
            is_per_unit=False,
            num_units=1,
            peak_bandwidth_bps=75e9,
            access_latency_ns=100.0,
        ),
    ]


# ---------------------------------------------------------------------------
# pick_binding_tier: empty hierarchy
# ---------------------------------------------------------------------------


def test_pick_binding_tier_returns_none_for_empty_hierarchy():
    assert pick_binding_tier(MatmulReuseModel(), (1024, 1024, 1024), "fp32", []) is None


# ---------------------------------------------------------------------------
# pick_binding_tier: matmul tier walk
# ---------------------------------------------------------------------------


def test_matmul_residency_in_l1_binds_at_l3():
    """Small matmul whose optimal tile fits in L1 should bind at L3
    (the next-larger tier). For C=512 KB (L1 aggregate=512KB), the
    matmul model picks tile t = floor(sqrt(512K/(3*4))) = floor(sqrt(43690))
    = 209. Residency = 3*209^2*4 = 524,108 bytes ~= 512 KB."""
    hierarchy = _i7_like_hierarchy()
    result = pick_binding_tier(
        MatmulReuseModel(), (1024, 1024, 1024), "fp32", hierarchy
    )
    assert result is not None
    assert result.residency_tier.name == "L1"
    assert result.binding_tier.name == "L3"
    # Tile should be at most L1's aggregate capacity
    assert result.tile.residency_bytes <= hierarchy[0].total_capacity_bytes


def test_matmul_4096_cube_binds_at_dram_due_to_operand_overflow():
    """Big matmul (4096, 4096, 4096) fp32: operand footprint =
    3 * 4096^2 * 4 = 192 MB, well past i7's 25 MB L3. Even though
    the L1-sized C-tile fits, the V5-5 operand-aware binding walk
    correctly escalates to DRAM."""
    hierarchy = _i7_like_hierarchy()
    result = pick_binding_tier(
        MatmulReuseModel(), (4096, 4096, 4096), "fp32", hierarchy
    )
    assert result is not None
    assert result.binding_tier.name == "DRAM"


def test_matmul_skinny_shape_escalates_to_dram_when_operands_overflow_l3():
    """V5-5 follow-up regression: matmul (64, 1024, 8192) fp32 has
    a tiny clamped C-tile that fits in L1, but operand footprint
    = (64*1024 + 1024*8192 + 64*8192) * 4 = ~35.9 MB, doesn't fit
    in i7's 25 MB L3. Operand-aware binding correctly returns
    DRAM here (whereas the V5-3b "next-outward-of-residency" rule
    that this PR replaces would have returned L3 -- the
    motivation for the V5-5 follow-up)."""
    hierarchy = _i7_like_hierarchy()
    result = pick_binding_tier(MatmulReuseModel(), (64, 1024, 8192), "fp32", hierarchy)
    assert result is not None
    assert result.binding_tier.name == "DRAM"
    assert result.residency_tier.name == "L1"


def test_matmul_small_shape_binds_at_l3_when_operands_fit():
    """Negative case: matmul (1024, 1024, 1024) fp32 has operand
    footprint = 3 * 1024^2 * 4 = 12 MB which fits L3 (25 MB). The
    binding tier should stay at L3, NOT escalate to DRAM."""
    hierarchy = _i7_like_hierarchy()
    result = pick_binding_tier(
        MatmulReuseModel(), (1024, 1024, 1024), "fp32", hierarchy
    )
    assert result is not None
    assert result.binding_tier.name == "L3"
    assert result.residency_tier.name == "L1"


# ---------------------------------------------------------------------------
# pick_binding_tier: vector_add (zero reuse, no useful tile)
# ---------------------------------------------------------------------------


def test_vector_add_small_n_binds_at_l1():
    """vector_add at N=1024 fp32 -> operand footprint = 12 KB. Fits in
    L1 (aggregate 512 KB), so V5-5 operand-aware binding correctly
    returns L1 (the lowest tier holding the data; warm-cache hits at
    L1 BW). Pre-V5-5-followup behavior was L3 (next outward of
    residency); the new behavior is more honest about where the data
    actually streams from."""
    hierarchy = _i7_like_hierarchy()
    result = pick_binding_tier(VectorAddReuseModel(), (1024,), "fp32", hierarchy)
    assert result is not None
    assert result.residency_tier.name == "L1"
    assert result.binding_tier.name == "L1"
    assert result.bytes_loaded == 3 * 1024 * 4


def test_vector_add_dram_sized_n_binds_at_dram():
    """vector_add at N=16M fp32 -> WS = 192 MB. Doesn't fit L1 (512 KB)
    or L3 (25 MB). Binds at DRAM (the outermost tier; the algorithm
    returns the outermost as binding when even it doesn't strictly
    'fit' the residency window)."""
    hierarchy = _i7_like_hierarchy()
    N = 16 * 1024 * 1024
    result = pick_binding_tier(VectorAddReuseModel(), (N,), "fp32", hierarchy)
    assert result is not None
    assert result.binding_tier.name == "DRAM"
    assert result.bytes_loaded == 3 * N * 4  # 192 MB


def test_vector_add_in_dram_only_hierarchy_binds_at_dram():
    """Single-tier hierarchy (only DRAM) -- residency tier == binding
    tier == DRAM (no tier larger to fall back to)."""
    hierarchy = _dram_only_hierarchy()
    result = pick_binding_tier(VectorAddReuseModel(), (1024,), "fp32", hierarchy)
    assert result is not None
    assert result.residency_tier.name == "DRAM"
    assert result.binding_tier.name == "DRAM"


# ---------------------------------------------------------------------------
# pick_binding_tier: out-of-order hierarchies
# ---------------------------------------------------------------------------


def test_pick_binding_tier_sorts_misordered_hierarchy():
    """A defensive guarantee: if a mapper happens to emit tiers
    out-of-capacity-order, the algorithm still walks innermost-out."""
    h = _i7_like_hierarchy()
    misordered = [h[2], h[0], h[1]]  # DRAM, L1, L3
    result = pick_binding_tier(
        MatmulReuseModel(), (1024, 1024, 1024), "fp32", misordered
    )
    assert result is not None
    assert result.residency_tier.name == "L1"
    assert result.binding_tier.name == "L3"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


def test_binding_tier_result_is_frozen():
    """BindingTierResult is intentionally frozen so callers can't
    mutate the binding tier or bytes_loaded after the picker returns."""
    hierarchy = _i7_like_hierarchy()
    result = pick_binding_tier(VectorAddReuseModel(), (1024,), "fp32", hierarchy)
    assert isinstance(result, BindingTierResult)
    with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
        result.bytes_loaded = 0  # type: ignore[misc]
