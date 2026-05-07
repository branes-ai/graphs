"""Unit tests for the V5-3a per-op reuse models.

Locks in:
- bytes_per_element rejects unknowns and matches the V4 classifier table
- VectorAddReuseModel: residency = full WS (no useful tile choice)
- MatmulReuseModel: optimal-square C-tile, clamped to [1, min(M,N)],
  bytes_loaded matches the per-plan formula and is monotone non-increasing
  in tile size.
- LinearReuseModel delegates to MatmulReuseModel cleanly.
- get_reuse_model raises on unknown op kinds.
"""

from math import floor, sqrt

import pytest

from graphs.estimation.reuse_models import (
    LinearReuseModel,
    MatmulReuseModel,
    REUSE_MODELS,
    ReuseModel,
    TileChoice,
    VectorAddReuseModel,
    bytes_per_element,
    get_reuse_model,
)


# ---------------------------------------------------------------------------
# bytes_per_element
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("fp64", 8),
        ("fp32", 4),
        ("tf32", 4),
        ("fp16", 2),
        ("bf16", 2),
        ("int8", 1),
        ("fp8", 1),
        ("fp8_e4m3", 1),
        ("fp8_e5m2", 1),
        ("int4", 0.5),
        ("fp4", 0.5),
    ],
)
def test_bytes_per_element(dtype, expected):
    assert bytes_per_element(dtype) == expected


def test_bytes_per_element_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown dtype"):
        bytes_per_element("complex64")


# ---------------------------------------------------------------------------
# TileChoice invariants
# ---------------------------------------------------------------------------


def test_tile_choice_rejects_empty_dims():
    with pytest.raises(ValueError, match="non-empty"):
        TileChoice(tile_dims=(), residency_bytes=0)


def test_tile_choice_rejects_zero_or_negative_dims():
    with pytest.raises(ValueError, match="positive"):
        TileChoice(tile_dims=(0,), residency_bytes=0)
    with pytest.raises(ValueError, match="positive"):
        TileChoice(tile_dims=(64, -1), residency_bytes=0)


def test_tile_choice_rejects_negative_residency():
    with pytest.raises(ValueError, match="non-negative"):
        TileChoice(tile_dims=(64,), residency_bytes=-1)


# ---------------------------------------------------------------------------
# VectorAddReuseModel
# ---------------------------------------------------------------------------


def test_vector_add_residency_is_full_working_set():
    """Vector add cannot benefit from tiling: residency must be the
    full working set 3*N*bpe regardless of the tier capacity hint."""
    model = VectorAddReuseModel()
    # N=1024 fp32 -> WS = 3 * 1024 * 4 = 12288 bytes
    tile = model.residency_window((1024,), "fp32", tier_capacity_bytes=512)
    assert tile.tile_dims == (1024,)
    assert tile.residency_bytes == 12288


def test_vector_add_residency_ignores_tier_capacity_hint():
    model = VectorAddReuseModel()
    tile_small = model.residency_window((4096,), "fp32", tier_capacity_bytes=64)
    tile_huge = model.residency_window((4096,), "fp32", tier_capacity_bytes=10**12)
    assert tile_small == tile_huge


@pytest.mark.parametrize(
    "N,dtype,expected_bytes",
    [
        (1024, "fp32", 3 * 1024 * 4),
        (1024, "fp16", 3 * 1024 * 2),
        (1, "fp64", 3 * 1 * 8),
        (256, "int8", 3 * 256 * 1),
    ],
)
def test_vector_add_bytes_loaded_equals_working_set(N, dtype, expected_bytes):
    model = VectorAddReuseModel()
    tile = model.residency_window((N,), dtype, tier_capacity_bytes=10**12)
    assert model.bytes_loaded_from_binding((N,), dtype, tile) == expected_bytes


def test_vector_add_rejects_non_1d_shape():
    model = VectorAddReuseModel()
    with pytest.raises(ValueError, match="1-D shape"):
        model.residency_window((1024, 1024), "fp32", tier_capacity_bytes=10**6)


def test_vector_add_rejects_zero_n():
    model = VectorAddReuseModel()
    with pytest.raises(ValueError, match="N >= 1"):
        model.residency_window((0,), "fp32", tier_capacity_bytes=10**6)


# ---------------------------------------------------------------------------
# MatmulReuseModel: tile sizing
# ---------------------------------------------------------------------------


def test_matmul_optimal_square_tile_size_for_small_capacity():
    """For C=12 KB and fp32, t = floor(sqrt(12288 / 12)) = floor(sqrt(1024)) = 32.
    Residency = 3 * 32^2 * 4 = 12288 bytes (fully fills the budget)."""
    model = MatmulReuseModel()
    tile = model.residency_window((1024, 1024, 1024), "fp32", tier_capacity_bytes=12288)
    assert tile.tile_dims == (32, 32)
    assert tile.residency_bytes == 12288


def test_matmul_optimal_square_for_l2_sized_capacity():
    """L2 = 256 KB on a per-core slice; fp16 -> t = floor(sqrt(256K/6)) = 209.
    Residency = 3 * 209^2 * 2 = 262_086 bytes."""
    model = MatmulReuseModel()
    tier = 256 * 1024
    tile = model.residency_window((4096, 4096, 4096), "fp16", tier_capacity_bytes=tier)
    expected_t = int(floor(sqrt(tier / (3 * 2))))
    assert tile.tile_dims == (expected_t, expected_t)
    assert tile.residency_bytes <= tier


def test_matmul_tile_clamped_to_min_dim():
    """If sqrt-formula gives a tile > min(M, N), clamp to min(M, N).
    Shape (4, 65536, 4) fp32, capacity huge: t_raw is enormous, clamped to 4."""
    model = MatmulReuseModel()
    tile = model.residency_window((4, 65536, 4), "fp32", tier_capacity_bytes=10**10)
    assert tile.tile_dims == (4, 4)
    assert tile.residency_bytes == 3 * 4 * 4 * 4


def test_matmul_tile_clamped_to_at_least_one():
    """If capacity is absurdly small (< one element), clamp t to 1."""
    model = MatmulReuseModel()
    tile = model.residency_window((1024, 1024, 1024), "fp32", tier_capacity_bytes=1)
    assert tile.tile_dims == (1, 1)
    assert tile.residency_bytes == 3 * 1 * 1 * 4


def test_matmul_rejects_non_3d_shape():
    model = MatmulReuseModel()
    with pytest.raises(ValueError, match="3-D shape"):
        model.residency_window((1024, 1024), "fp32", tier_capacity_bytes=10**6)


def test_matmul_rejects_non_positive_dims():
    model = MatmulReuseModel()
    with pytest.raises(ValueError, match="positive"):
        model.residency_window((0, 1024, 1024), "fp32", tier_capacity_bytes=10**6)


def test_matmul_rejects_non_positive_capacity():
    model = MatmulReuseModel()
    with pytest.raises(ValueError, match="tier_capacity_bytes"):
        model.residency_window((1024, 1024, 1024), "fp32", tier_capacity_bytes=0)


# ---------------------------------------------------------------------------
# MatmulReuseModel: bytes loaded from binding tier
# ---------------------------------------------------------------------------


def test_matmul_bytes_loaded_full_tile_equals_one_pass():
    """If the tile equals the full output (Mt=M, Nt=N), each input loaded
    exactly once: A_bytes = M*K*bpe, B_bytes = K*N*bpe, C_bytes = 2*M*N*bpe."""
    model = MatmulReuseModel()
    M, K, N = 1024, 1024, 1024
    bpe = 4
    tile = TileChoice(tile_dims=(M, N), residency_bytes=3 * M * N * bpe)
    actual = model.bytes_loaded_from_binding((M, K, N), "fp32", tile)
    expected = M * K * bpe + K * N * bpe + 2 * M * N * bpe
    assert actual == expected


def test_matmul_bytes_loaded_singleton_tile_equals_no_reuse():
    """With Mt=Nt=1, each output element forces a full A-row + B-col load
    once per output, with C read+written once per element. This is the
    pessimistic no-reuse bound."""
    model = MatmulReuseModel()
    M, K, N = 64, 64, 64  # small enough to keep arithmetic tractable
    bpe = 4
    tile = TileChoice(tile_dims=(1, 1), residency_bytes=3 * bpe)
    actual = model.bytes_loaded_from_binding((M, K, N), "fp32", tile)
    # A: M*K * (N/1) = M*K*N reloads, B: K*N * (M/1) = K*N*M, C: 2*M*N
    expected = M * K * N * bpe + K * N * M * bpe + 2 * M * N * bpe
    assert actual == expected


def test_matmul_bytes_loaded_monotone_decreasing_in_tile_size():
    """Larger C-tile -> more reuse -> fewer bytes streamed from the
    binding tier. Sweep tile sizes for a fixed problem and assert
    bytes_loaded is monotone non-increasing."""
    model = MatmulReuseModel()
    M, K, N = 512, 512, 512
    last = float("inf")
    for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        tile = TileChoice(tile_dims=(t, t), residency_bytes=3 * t * t * 4)
        b = model.bytes_loaded_from_binding((M, K, N), "fp32", tile)
        assert b <= last, f"non-monotone at t={t}: {b} > {last}"
        last = b


def test_matmul_bytes_loaded_rejects_non_2d_tile():
    model = MatmulReuseModel()
    bad_tile = TileChoice(tile_dims=(64,), residency_bytes=64 * 4)
    with pytest.raises(ValueError, match="2-D"):
        model.bytes_loaded_from_binding((512, 512, 512), "fp32", bad_tile)


# ---------------------------------------------------------------------------
# LinearReuseModel: delegates to MatmulReuseModel
# ---------------------------------------------------------------------------


def test_linear_residency_matches_matmul():
    """Linear (B, IN, OUT) must produce the same TileChoice as matmul
    (B, IN, OUT) for the same dtype and capacity."""
    lin = LinearReuseModel()
    mm = MatmulReuseModel()
    B, IN, OUT = 32, 512, 256
    cap = 64 * 1024
    assert lin.residency_window((B, IN, OUT), "fp32", cap) == mm.residency_window(
        (B, IN, OUT), "fp32", cap
    )


def test_linear_bytes_loaded_matches_matmul():
    lin = LinearReuseModel()
    mm = MatmulReuseModel()
    B, IN, OUT = 32, 512, 256
    cap = 64 * 1024
    tile = lin.residency_window((B, IN, OUT), "fp32", cap)
    assert lin.bytes_loaded_from_binding(
        (B, IN, OUT), "fp32", tile
    ) == mm.bytes_loaded_from_binding((B, IN, OUT), "fp32", tile)


def test_linear_rejects_non_3d_shape():
    lin = LinearReuseModel()
    with pytest.raises(ValueError, match="3-D shape"):
        lin.residency_window((512, 256), "fp32", tier_capacity_bytes=10**6)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_kind", ["vector_add", "matmul", "linear"])
def test_get_reuse_model_returns_correct_op(op_kind):
    model = get_reuse_model(op_kind)
    assert model.op_kind == op_kind


def test_get_reuse_model_unknown_raises():
    with pytest.raises(ValueError, match="No reuse model"):
        get_reuse_model("conv2d")


def test_registry_models_satisfy_protocol():
    for model in REUSE_MODELS.values():
        assert isinstance(model, ReuseModel)
