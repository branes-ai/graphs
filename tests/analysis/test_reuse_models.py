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


def test_matmul_skinny_dim_uses_skinny_branch_not_clamp():
    """Shape (4, 65536, 4) has min(M, N) = 4 < SKINNY_THRESHOLD,
    so the skinny branch fires (full-output tile = (4, 4)) BEFORE
    the clamp logic would. The result is the same tile_dims (4, 4)
    but via a different code path. Pinned separately so a future
    change that disables the skinny branch flips the right test
    rather than silently passing this one for the wrong reason."""
    model = MatmulReuseModel()
    tile = model.residency_window((4, 65536, 4), "fp32", tier_capacity_bytes=10**10)
    assert tile.tile_dims == (4, 4)
    # Skinny branch residency = 3 * M * N * bpe (full output tile)
    assert tile.residency_bytes == 3 * 4 * 4 * 4


def test_matmul_non_skinny_clamped_to_min_dim():
    """Coverage for the OPTIMAL-SQUARE CLAMP path (non-skinny):
    if sqrt-formula gives a tile > min(M, N), clamp to min(M, N).
    Use min(M, N) = 64 which is well above the 16 threshold so the
    skinny branch doesn't fire, then huge capacity to force the
    raw t to exceed 64.

    Pre per-op-calibration PR: this test was test_matmul_tile_clamped_to_min_dim
    using shape (4, 65536, 4), which inadvertently took the skinny
    branch when that landed. CodeRabbit flagged the silent coverage
    gap; this test restores explicit coverage for the clamp logic."""
    model = MatmulReuseModel()
    tile = model.residency_window(
        (64, 65536, 64), "fp32", tier_capacity_bytes=10**10
    )
    # min(M, N) = 64 >= SKINNY_THRESHOLD (16) -> skinny branch skipped
    # t_raw from sqrt(1e10 / 12) >> 64 -> clamped to 64
    assert tile.tile_dims == (64, 64)
    assert tile.residency_bytes == 3 * 64 * 64 * 4


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


def test_matmul_skinny_shape_uses_full_output_tile():
    """V5-3b flag-flip prerequisite: when min(M, N) < skinny threshold,
    the optimal-square model collapses to the small dim and inflates
    bytes_loaded by ceil(N/Mt) reloads. Real BLAS uses a different
    blocking strategy for skinny shapes; the model now picks the full
    output as the tile so bytes_loaded ~= operand_footprint with no
    reload inflation."""
    model = MatmulReuseModel()
    # Skinny B (linear with B=2 batch): min(M, N) = 2 < 16 threshold
    M, K, N = 2, 12288, 640
    tile = model.residency_window((M, K, N), "fp32", tier_capacity_bytes=10**8)
    assert tile.tile_dims == (
        M,
        N,
    ), f"Skinny shape should use full-output tile (M, N), got {tile.tile_dims}"
    # bytes_loaded with full-output tile collapses to one-pass:
    # A loaded once, B loaded once, C read + written once. That's
    # operand_footprint + one extra M*N for the C write (operand
    # counts each tensor once; bytes_loaded counts C twice for
    # read+write).
    bytes_ld = model.bytes_loaded_from_binding((M, K, N), "fp32", tile)
    operand = model.operand_footprint_bytes((M, K, N), "fp32")
    expected = operand + M * N * 4  # one extra C write at fp32
    assert bytes_ld == expected, (
        f"Skinny matmul bytes_loaded ({bytes_ld}) should equal "
        f"operand + C-write ({expected}); the full-output tile zeros "
        f"out reload counts but C is still read + written."
    )


def test_matmul_non_skinny_still_uses_optimal_square_tile():
    """Negative case: shapes with min(M, N) >= threshold use the
    optimal-square tile heuristic as before (V5-3a behavior). The
    skinny branch must not perturb the well-shaped majority."""
    model = MatmulReuseModel()
    # Non-skinny: min(M, N) = 1024, well above 16 threshold
    tile = model.residency_window((1024, 1024, 1024), "fp32", tier_capacity_bytes=12288)
    assert tile.tile_dims == (
        32,
        32,
    ), f"Non-skinny shape should use optimal-square tile, got {tile.tile_dims}"


def test_matmul_skinny_thresholds_pinned():
    """Pin the empirical thresholds. _SKINNY_THRESHOLD is the legacy
    alias kept for backward-compat (= _SKINNY_MIN_DIM). The aspect
    ratio threshold is independent and was added by the V5 follow-up
    aspect-ratio PR. Moving either changes which shapes hit the
    no-reuse model."""
    assert MatmulReuseModel._SKINNY_THRESHOLD == 16
    assert MatmulReuseModel._SKINNY_MIN_DIM == 16
    assert MatmulReuseModel._SKINNY_ASPECT_RATIO == 32


def test_matmul_high_aspect_ratio_uses_full_output_tile_even_with_large_min_dim():
    """V5 follow-up (Jetson Orin Nano): the optimal-square tile
    inflates reload counts on shapes with high aspect ratios even
    when min(M, N) is well above the absolute floor. Example:
    (64, 16384, 8192) has min=64 (above the 16 floor) but aspect
    ratio = 8192/64 = 128, so the (64, 64) optimal-square tile
    reloads A 128 times. The aspect-ratio check fires and switches
    to the full-output tile. Hardware-independent: this is a shape-
    intrinsic property."""
    model = MatmulReuseModel()
    M, K, N = 64, 16384, 8192
    tile = model.residency_window((M, K, N), "fp16", tier_capacity_bytes=10**8)
    # aspect ratio = max(64, 8192) / min(64, 8192) = 128 >= 32 -> skinny
    assert tile.tile_dims == (M, N), (
        f"High aspect ratio shape should use full-output tile, got {tile.tile_dims}"
    )
    assert tile.residency_bytes == 3 * M * N * 2  # fp16


def test_matmul_aspect_ratio_just_below_threshold_uses_optimal_square():
    """Negative case: aspect ratio < 32 with min(M, N) above the
    absolute floor must still use the optimal-square tile. Shape
    (256, 4096, 7936) has aspect = 7936/256 = 31, just below the
    threshold; min = 256, well above 16. Neither check fires.

    Why this region is excluded: empirically the optimal-square
    reload model is more accurate than full-output for AR<=32, and
    aggressively switching to full-output causes a +2.4% MAE
    regression on the Jetson Orin Nano matmul 16<AR<=32 bucket."""
    model = MatmulReuseModel()
    M, K, N = 256, 4096, 7936
    # aspect ratio = 7936 / 256 = 31.0 < 32
    # min(M, N) = 256, well above _SKINNY_MIN_DIM = 16
    tile = model.residency_window((M, K, N), "fp32", tier_capacity_bytes=12288)
    # Optimal-square clamp: t_max = min(M, N) = 256, t_raw = sqrt(12288/12) ~= 32
    # -> t = 32
    assert tile.tile_dims == (32, 32), (
        f"Aspect ratio just below threshold should use optimal-square, "
        f"got {tile.tile_dims}"
    )


def test_matmul_aspect_ratio_at_threshold_fires_skinny_branch():
    """Boundary: aspect ratio exactly equal to 32 should fire the
    skinny branch (>=, not >). Confirms the inclusive boundary."""
    model = MatmulReuseModel()
    # (256, 4096, 8192): aspect = 8192 / 256 = 32 exactly
    M, K, N = 256, 4096, 8192
    tile = model.residency_window((M, K, N), "fp32", tier_capacity_bytes=12288)
    assert tile.tile_dims == (M, N), (
        f"Aspect ratio at threshold should fire skinny branch (full-output), "
        f"got {tile.tile_dims}"
    )


def test_matmul_aspect_ratio_in_16_to_32_band_uses_optimal_square():
    """Coverage for the 16<AR<32 zone where keeping the optimal-
    square model is empirically better. (192, 6144, 6144) has
    aspect = 6144/192 = 32, exactly on the boundary -> fires.
    Use (192, 6144, 5760) for aspect 30 to test the band."""
    model = MatmulReuseModel()
    M, K, N = 192, 6144, 5760  # aspect = 5760 / 192 = 30 < 32
    tile = model.residency_window((M, K, N), "fp16", tier_capacity_bytes=10**6)
    # Should NOT take the skinny branch; should clamp to t_max = min(M, N) = 192
    # or sqrt-formula tile, whichever is smaller.
    assert tile.tile_dims != (M, N), (
        f"Aspect 30 should stay on optimal-square path, but got full-output tile {tile.tile_dims}"
    )


def test_matmul_bytes_loaded_uses_ceil_for_non_dividing_tile():
    """When Mt or Nt doesn't divide M or N, the partial last column
    of C-tiles still triggers a full A-row reload (and partial last
    row a full B-col reload). Float division underestimates that;
    ceil is the right count.

    Shape (1024, 1024, 1024) with tile (300, 300):
      ceil(1024/300) = 4 reloads of A (not 3.41)
      ceil(1024/300) = 4 reloads of B
    """
    model = MatmulReuseModel()
    M = K = N = 1024
    Mt = Nt = 300
    bpe = 4
    tile = TileChoice(tile_dims=(Mt, Nt), residency_bytes=3 * Mt * Nt * bpe)
    actual = model.bytes_loaded_from_binding((M, K, N), "fp32", tile)
    expected = (
        M * K * bpe * 4  # A reloaded ceil(N/Nt) = 4 times
        + K * N * bpe * 4  # B reloaded ceil(M/Mt) = 4 times
        + 2 * M * N * bpe  # C read + written
    )
    assert actual == expected
    # And the float-division (under)estimate would have been:
    underestimate = int(
        round(M * K * bpe * (N / Nt) + K * N * bpe * (M / Mt) + 2 * M * N * bpe)
    )
    assert actual > underestimate


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


# ---------------------------------------------------------------------------
# operand_footprint_bytes (V5-5 follow-up)
# ---------------------------------------------------------------------------


def test_vector_add_operand_footprint_equals_three_times_n_bytes():
    """vector_add operand footprint is 3*N*bpe -- same as residency
    (degenerate case for zero-reuse op). The V5-5 tier picker uses
    this to decide where the data lives."""
    model = VectorAddReuseModel()
    assert model.operand_footprint_bytes((1024,), "fp32") == 3 * 1024 * 4
    assert model.operand_footprint_bytes((4096,), "fp16") == 3 * 4096 * 2
    assert model.operand_footprint_bytes((1,), "fp64") == 24


def test_matmul_operand_footprint_is_a_plus_b_plus_c():
    """matmul (M, K, N) fp32: A=M*K, B=K*N, C=M*N elements; *bpe each."""
    model = MatmulReuseModel()
    M, K, N = 1024, 1024, 1024
    expected = (M * K + K * N + M * N) * 4
    assert model.operand_footprint_bytes((M, K, N), "fp32") == expected

    # Skinny shape: small C-tile, but operands are big
    M, K, N = 64, 1024, 8192
    expected = (M * K + K * N + M * N) * 4
    assert model.operand_footprint_bytes((M, K, N), "fp32") == expected


def test_matmul_operand_footprint_rejects_non_3d_shape():
    model = MatmulReuseModel()
    with pytest.raises(ValueError, match="3-D shape"):
        model.operand_footprint_bytes((1024, 1024), "fp32")


def test_linear_operand_footprint_matches_matmul_delegate():
    lin = LinearReuseModel()
    mm = MatmulReuseModel()
    B, IN, OUT = 32, 512, 256
    assert lin.operand_footprint_bytes(
        (B, IN, OUT), "fp32"
    ) == mm.operand_footprint_bytes((B, IN, OUT), "fp32")


def test_linear_operand_footprint_rejects_non_3d():
    lin = LinearReuseModel()
    with pytest.raises(ValueError, match="3-D shape"):
        lin.operand_footprint_bytes((512, 256), "fp32")


def test_get_reuse_model_unknown_raises():
    with pytest.raises(ValueError, match="No reuse model"):
        get_reuse_model("conv2d")


def test_registry_models_satisfy_protocol():
    for model in REUSE_MODELS.values():
        assert isinstance(model, ReuseModel)
