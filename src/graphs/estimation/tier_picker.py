"""Tier-picking algorithm for V5-3b roofline integration.

Given an op kind, shape, dtype, and the hardware's memory hierarchy,
walks the hierarchy innermost-out to find:
  * the *residency tier* -- the smallest tier whose aggregate capacity
    holds the op's residency window when tiled for that tier;
  * the *binding tier* -- the next-larger tier (the streaming source
    whose bandwidth gates kernel throughput);
  * the *tile choice* picked at the residency tier;
  * the *bytes loaded* from the binding tier across one kernel exec.

The plan algorithm lives in ``docs/plans/v5-memory-hierarchy-rewrite-plan.md``
section "Tier-picking algorithm". This module is consumed by the
roofline analyzer's V5-3b ``_get_effective_memory_time`` path (opt-in
via ``use_tier_aware_memory=True`` until V5-5 calibrates the per-tier
``achievable_fraction`` values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from graphs.estimation.reuse_models import ReuseModel, TileChoice
from graphs.hardware.resource_model import MemoryTier


# PyTorch / FX commonly uses 'float32', 'bfloat16', etc. The reuse_models
# bytes_per_element table uses the V4-classifier short forms ('fp32', 'bf16').
# Map both ways so callers can pass whatever they have.
_DTYPE_ALIASES = {
    "float64": "fp64",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
    "torch.float64": "fp64",
    "torch.float32": "fp32",
    "torch.float16": "fp16",
    "torch.bfloat16": "bf16",
    "torch.int8": "int8",
    "torch.uint8": "int8",
}


def normalize_dtype(dtype: str) -> str:
    """Map common dtype string variants to the reuse_models short form.

    Pass-through for already-short forms ('fp32', 'fp16', etc.). Raises
    KeyError if the dtype is unrecognized -- callers should catch and
    fall back to the scalar path rather than guess.
    """
    if dtype in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[dtype]
    return dtype  # already short form, or unknown -- let bytes_per_element decide


@dataclass(frozen=True)
class BindingTierResult:
    """The output of ``pick_binding_tier``: which tier gates throughput,
    what tile size was chosen, and how many bytes stream from the binding
    tier per kernel execution."""

    binding_tier: MemoryTier
    residency_tier: MemoryTier  # may equal binding_tier when at outermost
    tile: TileChoice
    bytes_loaded: int


def pick_binding_tier(
    reuse_model: ReuseModel,
    shape: Sequence[int],
    dtype: str,
    hierarchy: Sequence[MemoryTier],
) -> Optional[BindingTierResult]:
    """Walk the memory hierarchy to pick (residency tier, binding tier).

    Algorithm (V5-5 follow-up; supersedes the simpler "binding = next
    outward of residency" rule from V5-3b):

      1. Sort tiers innermost-out by aggregate capacity.
      2. Residency tier = smallest tier whose capacity holds the per-op
         tile (sized for that tier). For matmul this is the C-tile; for
         vector_add the full working set.
      3. Binding tier = smallest tier whose capacity holds the
         **operand footprint** (A + B + C for matmul; full WS for
         vector_add). This is independent of where residency landed.
         If the operand footprint exceeds even the outermost tier, the
         outermost tier (typically DRAM) is the binding tier and the
         analyzer will surface the right "memory-bound" verdict from
         the resulting bytes_loaded / effective_bw math.

    Why operand-aware? Skinny matmul shapes like (M=64, K=1024, N=8192)
    fp32 have an L1-resident C-tile but A+B+C = 35.9 MB which doesn't
    fit i7's 25 MB L3. The simple "binding = residency + 1" rule
    pinned binding=L3 -> apparent throughput at L3 BW (200 GB/s peak),
    but the operands actually have to come from DRAM (75 GB/s peak,
    35 GB/s calibrated). Operand-aware binding correctly escalates to
    DRAM for these shapes, which is exactly the case where the V5-5
    DRAM calibration bites.

    For vector_add at small N (e.g. 12 KB working set), this also
    changes the answer: instead of binding=L3 (next outward of L1
    residency), binding=L1 (operands fit L1). For warm-cache
    benchmarks that's the realistic answer.

    Returns ``None`` if the hierarchy is empty (the caller falls back
    to the scalar bw_efficiency_scale path).
    """
    if not hierarchy:
        return None

    tiers = sorted(hierarchy, key=lambda t: t.total_capacity_bytes)

    # 1) Residency tier: smallest tier whose capacity holds the op's tile.
    residency_tier: Optional[MemoryTier] = None
    tile: Optional[TileChoice] = None
    for tier in tiers:
        candidate_tile = reuse_model.residency_window(
            shape, dtype, tier_capacity_bytes=tier.total_capacity_bytes
        )
        if candidate_tile.residency_bytes <= tier.total_capacity_bytes:
            residency_tier = tier
            tile = candidate_tile
            break

    if residency_tier is None:
        # No tier holds the tile (working set overflows even outermost).
        # Fall through: residency = outermost; tile sized for it.
        residency_tier = tiers[-1]
        tile = reuse_model.residency_window(
            shape, dtype, tier_capacity_bytes=residency_tier.total_capacity_bytes
        )

    # 2) Binding tier: smallest tier whose capacity holds the operand
    #    footprint. Walks innermost-out independent of residency tier.
    operand_bytes = reuse_model.operand_footprint_bytes(shape, dtype)
    binding_tier: MemoryTier = tiers[-1]  # default = outermost
    for tier in tiers:
        if operand_bytes <= tier.total_capacity_bytes:
            binding_tier = tier
            break

    bytes_loaded = reuse_model.bytes_loaded_from_binding(shape, dtype, tile)
    return BindingTierResult(
        binding_tier=binding_tier,
        residency_tier=residency_tier,
        tile=tile,
        bytes_loaded=int(bytes_loaded),
    )


__all__ = [
    "BindingTierResult",
    "normalize_dtype",
    "pick_binding_tier",
]
