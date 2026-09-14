"""
Locate the KPU block (and the die that carries it) inside a ComputeProduct.

KPU code used to assume the KPU block is ``cp.dies[0].blocks[0]``. That holds
for today's 12 monolithic catalog SKUs, but not for an SoC die that carries a
KPU next to CPU / GPU / ISP blocks, nor for a chiplet package whose KPU die is
not the first die (docs/plans/soc-microarchitecture-study-plan.md,
docs/plans/kpu-heterogeneous-tile-refactor-plan.md, graphs#268 A2).

These helpers find the KPU block by kind instead of by position:

- ``kpu_block_of(cp)`` -- the single ``KPUBlock`` in the product
- ``kpu_die_of(cp)``   -- the ``Die`` that carries it (silicon_bin, clocks,
  process_node_id, die_size_mm2 all live on the die)
- ``has_kpu_block(cp)`` -- whether the product has any KPU block

A product with no KPU block, or with more than one, raises
``KPUBlockLookupError``: there is no well-defined "the KPU" to return, and a
silent first-match would reintroduce the positional assumption. Callers that
own a domain-specific error type (``SiliconMathError``, ``GeneratorError``,
...) convert it at their boundary.

Leaf module: imports only embodied-schemas, so any graphs module can use it
without import cycles.
"""

from __future__ import annotations

from embodied_schemas.compute_product import ComputeProduct, Die, KPUBlock


class KPUBlockLookupError(LookupError):
    """The ComputeProduct has no KPU block, or more than one."""


def _kpu_locations(cp: ComputeProduct) -> list[tuple[Die, KPUBlock]]:
    # ``or []`` tolerates instances built with ``model_construct()``, which
    # bypasses the schema's min_length=1 on dies / blocks.
    return [
        (die, block)
        for die in (cp.dies or [])
        for block in (die.blocks or [])
        if isinstance(block, KPUBlock)
    ]


def _describe(cp: ComputeProduct) -> str:
    dies = cp.dies or []
    if not dies:
        return "no dies"
    parts = []
    for die in dies:
        kinds = [getattr(b, "kind", type(b).__name__) for b in (die.blocks or [])]
        kinds = [getattr(k, "value", k) for k in kinds]
        parts.append(f"{die.die_id}: [{', '.join(map(str, kinds)) or 'no blocks'}]")
    return "; ".join(parts)


def _locate(cp: ComputeProduct) -> tuple[Die, KPUBlock]:
    found = _kpu_locations(cp)
    if len(found) == 1:
        return found[0]
    if not found:
        raise KPUBlockLookupError(
            f"compute product {cp.id!r} has no KPUBlock ({_describe(cp)})"
        )
    raise KPUBlockLookupError(
        f"compute product {cp.id!r} has {len(found)} KPUBlocks "
        f"(on dies {[d.die_id for d, _ in found]}); expected exactly one"
    )


def kpu_block_of(cp: ComputeProduct) -> KPUBlock:
    """The single KPUBlock in ``cp``, wherever it sits.

    Raises:
        KPUBlockLookupError: if ``cp`` has no KPUBlock or more than one.
    """
    return _locate(cp)[1]


def kpu_die_of(cp: ComputeProduct) -> Die:
    """The Die that carries ``cp``'s single KPUBlock.

    Raises:
        KPUBlockLookupError: if ``cp`` has no KPUBlock or more than one.
    """
    return _locate(cp)[0]


def has_kpu_block(cp: ComputeProduct) -> bool:
    """True if any die of ``cp`` carries at least one KPUBlock."""
    return bool(_kpu_locations(cp))
