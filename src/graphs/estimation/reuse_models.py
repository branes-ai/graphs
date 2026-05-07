"""Per-operator reuse models for tier-aware roofline analysis.

V5-3a: pure additions, no analyzer behavior change. The V5-3b
``tier_picker`` will call the two methods on each model:

  1. ``residency_window(shape, dtype, tier_capacity_bytes) -> TileChoice``
       Given a hypothetical residency tier with capacity C bytes, what
       tile size would the op pick and what working set does that imply?

  2. ``bytes_loaded_from_binding(shape, dtype, tile) -> int``
       Given the chosen tile, how many bytes stream from the *binding*
       tier (the tier just outside the residency tier) across the kernel?

The two questions together let the analyzer walk the memory hierarchy
innermost-out, find the smallest tier whose capacity holds the residency
window, and price the kernel's memory time at the next-larger tier's
achievable bandwidth.

Reuse model derivations live in ``docs/plans/v5-memory-hierarchy-rewrite-plan.md``
sections "Per-operator reuse models" and "Tier-picking algorithm".
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, sqrt
from typing import Dict, Protocol, Sequence, Tuple, runtime_checkable


_BYTES_PER_ELEMENT: Dict[str, float] = {
    "fp64": 8,
    "fp32": 4,
    "tf32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int4": 0.5,
    "fp4": 0.5,
}


def bytes_per_element(dtype: str) -> float:
    if dtype not in _BYTES_PER_ELEMENT:
        raise ValueError(
            f"Unknown dtype {dtype!r}; known: {sorted(_BYTES_PER_ELEMENT)}"
        )
    return _BYTES_PER_ELEMENT[dtype]


@dataclass(frozen=True)
class TileChoice:
    """The tile dimensions an op would pick to fit a given tier capacity,
    plus the bytes that tile occupies in the residency tier.

    ``tile_dims`` is op-specific:
      vector_add: ``(N,)`` -- the full length (no useful tiling exists)
      matmul:     ``(Mt, Nt)`` -- the C-tile dimensions
      linear:     ``(Mt, Nt)`` -- delegated to matmul (M=batch, N=out)
    """

    tile_dims: Tuple[int, ...]
    residency_bytes: int

    def __post_init__(self) -> None:
        if not self.tile_dims:
            raise ValueError("tile_dims must be non-empty")
        if any(d < 1 for d in self.tile_dims):
            raise ValueError(f"tile_dims must be positive: {self.tile_dims}")
        if self.residency_bytes < 0:
            raise ValueError(
                f"residency_bytes must be non-negative: {self.residency_bytes}"
            )


@runtime_checkable
class ReuseModel(Protocol):
    """Per-op interface the V5-3b tier picker calls."""

    op_kind: str

    def residency_window(
        self,
        shape: Sequence[int],
        dtype: str,
        tier_capacity_bytes: int,
    ) -> TileChoice: ...

    def bytes_loaded_from_binding(
        self,
        shape: Sequence[int],
        dtype: str,
        tile: TileChoice,
    ) -> int: ...

    def operand_footprint_bytes(
        self,
        shape: Sequence[int],
        dtype: str,
    ) -> int:
        """Total unique-data footprint the op needs to keep alive
        somewhere in the memory hierarchy across one execution.

        For ops with reuse (matmul, linear): A + B + C bytes -- this
        is the working set that must live in some tier. The V5-5
        operand-aware tier picker uses this to escalate the binding
        tier when the C-tile fits a small inner cache but the
        operands overflow it (skinny matmul case).

        For ops without reuse (vector_add): same as the residency
        window (3*N*bpe). The model is consistent across op kinds
        even when the answer is degenerate.
        """


class VectorAddReuseModel:
    """``c[i] = a[i] + b[i]`` on N elements -- the zero-reuse op.

    No tile choice helps: each input byte is read exactly once, each
    output byte is written exactly once. The residency window is the
    full working set ``3 * N * bpe``; ``tier_capacity_bytes`` is
    ignored because no smaller tile reduces traffic.
    """

    op_kind = "vector_add"

    def residency_window(
        self,
        shape: Sequence[int],
        dtype: str,
        tier_capacity_bytes: int,  # noqa: ARG002 -- intentionally ignored
    ) -> TileChoice:
        if len(shape) != 1:
            raise ValueError(f"vector_add expects 1-D shape, got {tuple(shape)}")
        (N,) = shape
        if N < 1:
            raise ValueError(f"vector_add requires N >= 1, got {N}")
        bpe = bytes_per_element(dtype)
        return TileChoice(
            tile_dims=(int(N),),
            residency_bytes=int(round(3 * N * bpe)),
        )

    def bytes_loaded_from_binding(
        self,
        shape: Sequence[int],
        dtype: str,
        tile: TileChoice,  # noqa: ARG002 -- not parameterized by tile choice
    ) -> int:
        (N,) = shape
        bpe = bytes_per_element(dtype)
        return int(round(3 * N * bpe))

    def operand_footprint_bytes(
        self,
        shape: Sequence[int],
        dtype: str,
    ) -> int:
        (N,) = shape
        bpe = bytes_per_element(dtype)
        return int(round(3 * N * bpe))


class MatmulReuseModel:
    """``C = A @ B`` with shape ``(M, K, N)``.

    Tile heuristic: optimal-square C-tile of side
    ``t = floor(sqrt(tier_capacity / (3 * bpe)))``, clamped to
    ``[1, min(M, N)]``. Residency = ``3 * t * t * bpe`` (the C
    accumulator plus same-area A and B slice scratch).

    Bytes loaded from the binding tier across the kernel
    (per the plan, section "Per-operator reuse models / Matmul"):
      * A is reloaded ``ceil(N / Nt)`` times: ``M*K*bpe * ceil(N/Nt)``
      * B is reloaded ``ceil(M / Mt)`` times: ``K*N*bpe * ceil(M/Mt)``
      * C is read + written once: ``2 * M * N * bpe``

    The ``ceil`` is intentional: when ``Nt`` doesn't divide ``N`` (or
    ``Mt`` doesn't divide ``M``), the partial last column of C-tiles
    still triggers a full A-row reload (and the partial last row a
    full B-col reload). Float division underestimates that.
    """

    op_kind = "matmul"

    # Skinny matmul detection. Two independent checks; if EITHER fires,
    # the optimal-square tile model produces excessive reload counts
    # and we switch to full-output tile (K-blocking semantics).
    #
    # 1. Absolute small-dim check: ``min(M, N) < _SKINNY_MIN_DIM``.
    #    Catches tiny dims (e.g. B=2 linear) where the optimal-square
    #    tile collapses to a 2x2 tile and reloads A by ceil(N/2)
    #    times. Hardware-independent.
    #
    # 2. Aspect-ratio check: ``max(M, N) / min(M, N) >= _SKINNY_ASPECT_RATIO``.
    #    Catches shapes where min(M, N) is reasonable in absolute terms
    #    (e.g. 64-256) but the larger dim is much larger, producing high
    #    reload counts. The Jetson Orin Nano matmul failures cluster
    #    here: (64, 16384, 8192) has min=64, aspect=256, and the
    #    optimal-square tile (64, 64) reloads A 128 times.
    #    Hardware-independent: the underlying issue is the shape's
    #    aspect ratio, not the cache hierarchy.
    #
    # The aspect threshold of 32 is empirical (Jetson Orin Nano matmul
    # baseline). At AR<=16 the optimal-square reload model is
    # accurate enough; at AR>=32 the K-blocking full-output tile is
    # accurate enough; the 16-32 transition zone slightly favors
    # the optimal-square model in practice. Per-bucket MAE on Jetson
    # matmul: 32<AR<=64 -5.1%, 64<AR<=128 -12.9%, AR>128 -62.2%
    # (see PR description for the full table).
    #
    # V4 PASS counts on Jetson matmul don't move at this threshold
    # because the dominant failure mode for those shapes is the
    # regime classifier (predicted memory-bound, measured compute-
    # bound -- compute model under-derate, tracked separately). The
    # change still ships because (a) it's a memory-physics correction
    # that makes the model more honest and (b) when the compute
    # model is fixed, the latency precision we add now will translate
    # to PASS gains.
    _SKINNY_MIN_DIM = 16
    _SKINNY_ASPECT_RATIO = 32
    # Backward compat alias (some external code may import this; the
    # name is a misnomer post-aspect-ratio-extension but not breaking).
    _SKINNY_THRESHOLD = _SKINNY_MIN_DIM

    def residency_window(
        self,
        shape: Sequence[int],
        dtype: str,
        tier_capacity_bytes: int,
    ) -> TileChoice:
        if len(shape) != 3:
            raise ValueError(f"matmul expects 3-D shape (M, K, N), got {tuple(shape)}")
        M, K, N = shape  # noqa: F841 -- K unused here, used in bytes_loaded
        if M < 1 or K < 1 or N < 1:
            raise ValueError(f"matmul requires positive dims, got {(M, K, N)}")
        if tier_capacity_bytes < 1:
            raise ValueError(
                f"tier_capacity_bytes must be positive: {tier_capacity_bytes}"
            )
        bpe = bytes_per_element(dtype)

        # Skinny shape branch: switch to full-output tile (Mt = M,
        # Nt = N) when the optimal-square model would produce
        # inflated reload counts. Two independent triggers (see
        # class-level _SKINNY_MIN_DIM / _SKINNY_ASPECT_RATIO docs):
        #   1. min(M, N) below absolute floor (tiny-dim case)
        #   2. aspect ratio at or above _SKINNY_ASPECT_RATIO
        is_skinny = (
            min(M, N) < self._SKINNY_MIN_DIM
            or max(M, N) >= self._SKINNY_ASPECT_RATIO * min(M, N)
        )
        if is_skinny:
            return TileChoice(
                tile_dims=(int(M), int(N)),
                residency_bytes=int(round(3 * M * N * bpe)),
            )

        t_max = min(M, N)
        t_raw = int(floor(sqrt(tier_capacity_bytes / (3 * bpe))))
        t = max(1, min(t_max, t_raw))
        return TileChoice(
            tile_dims=(t, t),
            residency_bytes=int(round(3 * t * t * bpe)),
        )

    def bytes_loaded_from_binding(
        self,
        shape: Sequence[int],
        dtype: str,
        tile: TileChoice,
    ) -> int:
        M, K, N = shape
        bpe = bytes_per_element(dtype)
        if len(tile.tile_dims) != 2:
            raise ValueError(f"matmul tile must be 2-D (Mt, Nt), got {tile.tile_dims}")
        Mt, Nt = tile.tile_dims
        a_bytes = M * K * bpe * ceil(N / Nt)
        b_bytes = K * N * bpe * ceil(M / Mt)
        c_bytes = 2 * M * N * bpe
        return int(round(a_bytes + b_bytes + c_bytes))

    def operand_footprint_bytes(
        self,
        shape: Sequence[int],
        dtype: str,
    ) -> int:
        if len(shape) != 3:
            raise ValueError(f"matmul expects 3-D shape (M, K, N), got {tuple(shape)}")
        M, K, N = shape
        bpe = bytes_per_element(dtype)
        return int(round((M * K + K * N + M * N) * bpe))


class LinearReuseModel:
    """``y = x @ W^T + b`` with shape ``(B, IN, OUT)``.

    Algebraically a matmul with ``M=B``, ``K=IN``, ``N=OUT``; the bias
    add is a separate vector_add and not modeled here. Delegates to
    :class:`MatmulReuseModel`.
    """

    op_kind = "linear"

    def __init__(self) -> None:
        self._delegate = MatmulReuseModel()

    def residency_window(
        self,
        shape: Sequence[int],
        dtype: str,
        tier_capacity_bytes: int,
    ) -> TileChoice:
        if len(shape) != 3:
            raise ValueError(
                f"linear expects 3-D shape (B, IN, OUT), got {tuple(shape)}"
            )
        B, IN, OUT = shape
        return self._delegate.residency_window((B, IN, OUT), dtype, tier_capacity_bytes)

    def bytes_loaded_from_binding(
        self,
        shape: Sequence[int],
        dtype: str,
        tile: TileChoice,
    ) -> int:
        B, IN, OUT = shape
        return self._delegate.bytes_loaded_from_binding((B, IN, OUT), dtype, tile)

    def operand_footprint_bytes(
        self,
        shape: Sequence[int],
        dtype: str,
    ) -> int:
        if len(shape) != 3:
            raise ValueError(
                f"linear expects 3-D shape (B, IN, OUT), got {tuple(shape)}"
            )
        B, IN, OUT = shape
        return self._delegate.operand_footprint_bytes((B, IN, OUT), dtype)


REUSE_MODELS: Dict[str, ReuseModel] = {
    "vector_add": VectorAddReuseModel(),
    "matmul": MatmulReuseModel(),
    "linear": LinearReuseModel(),
}


def get_reuse_model(op_kind: str) -> ReuseModel:
    """Look up the reuse model for an op kind. Raises if unknown."""
    if op_kind not in REUSE_MODELS:
        raise ValueError(
            f"No reuse model for op_kind={op_kind!r}; " f"known: {sorted(REUSE_MODELS)}"
        )
    return REUSE_MODELS[op_kind]


__all__ = [
    "TileChoice",
    "ReuseModel",
    "VectorAddReuseModel",
    "MatmulReuseModel",
    "LinearReuseModel",
    "REUSE_MODELS",
    "bytes_per_element",
    "get_reuse_model",
]
