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
    """Walk the memory hierarchy innermost-out to find the binding tier.

    Algorithm (per the V5 plan):
      1. Sort tiers by aggregate capacity (already innermost-out for
         standard mapper-built hierarchies; explicit sort guards against
         mappers that emit out-of-order).
      2. For each tier, ask the reuse model what tile and residency
         window it would pick if this tier were the residency tier.
      3. The first tier whose total capacity holds that residency window
         is the residency tier. The binding tier is the next-larger one
         (or itself if we're already at the outermost).
      4. Bytes loaded from the binding tier come from the reuse model
         applied at the chosen tile.

    Returns ``None`` if the hierarchy is empty (the caller should fall
    back to the scalar bw_efficiency_scale path). All other inputs are
    handled by the reuse model and ``MemoryTier`` invariants -- no
    silent NaNs / negatives slip through.
    """
    if not hierarchy:
        return None

    tiers = sorted(hierarchy, key=lambda t: t.total_capacity_bytes)

    for idx, tier in enumerate(tiers):
        tile = reuse_model.residency_window(
            shape, dtype, tier_capacity_bytes=tier.total_capacity_bytes
        )
        if tile.residency_bytes <= tier.total_capacity_bytes:
            binding_tier = tiers[idx + 1] if idx + 1 < len(tiers) else tier
            bytes_loaded = reuse_model.bytes_loaded_from_binding(shape, dtype, tile)
            return BindingTierResult(
                binding_tier=binding_tier,
                residency_tier=tier,
                tile=tile,
                bytes_loaded=int(bytes_loaded),
            )

    # No tier holds the residency window -- the working set overflows even
    # the outermost tier (typically only happens for vector ops at sizes
    # that exceed DRAM, which is an OOM scenario, not a roofline one).
    # Fall through: bind at the outermost tier with whatever tile the
    # reuse model picks for that capacity. The analyzer will see a very
    # large bytes_loaded and surface the right "memory-bound" verdict.
    outer = tiers[-1]
    tile = reuse_model.residency_window(
        shape, dtype, tier_capacity_bytes=outer.total_capacity_bytes
    )
    bytes_loaded = reuse_model.bytes_loaded_from_binding(shape, dtype, tile)
    return BindingTierResult(
        binding_tier=outer,
        residency_tier=outer,
        tile=tile,
        bytes_loaded=int(bytes_loaded),
    )


__all__ = [
    "BindingTierResult",
    "normalize_dtype",
    "pick_binding_tier",
]
