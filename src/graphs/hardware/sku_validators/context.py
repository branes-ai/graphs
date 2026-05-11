"""ValidatorContext builder.

Loads a SKU plus its referenced ProcessNode plus the full cooling-solution
catalog and assembles a ValidatorContext. Most CLI / test paths use
``build_context_for_kpu(sku_id)`` and never touch the raw loaders.

Cross-reference errors (missing process node, missing cooling solution)
raise ``ContextError`` here so the CLI can surface them as a single clear
message rather than letting individual validators each report the same
underlying problem.
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas import (
    load_cooling_solutions,
    load_kpus,
    load_process_nodes,
)
from embodied_schemas.cooling_solution import CoolingSolutionEntry
from embodied_schemas.kpu import KPUEntry
from embodied_schemas.process_node import ProcessNodeEntry

from .framework import ValidatorContext


class ContextError(Exception):
    """Raised when a ValidatorContext cannot be assembled."""


def build_context_for_kpu(
    sku_id: str,
    *,
    kpus: Optional[dict[str, KPUEntry]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    cooling_solutions: Optional[dict[str, CoolingSolutionEntry]] = None,
) -> ValidatorContext:
    """Build a ValidatorContext for KPU SKU ``sku_id``.

    Args:
        sku_id: The KPU SKU id, e.g., ``"kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"``.
        kpus / process_nodes / cooling_solutions: Optional pre-loaded
            catalogs. Tests pass small in-memory dicts here to avoid
            disk IO; the CLI lets them default to disk loaders.

    Raises:
        ContextError: if the SKU id isn't found, or its process_node_id
            doesn't resolve in the process-node catalog. Missing
            cooling_solution_id refs are NOT raised here -- they're
            reported by an INTERNAL validator so the SKU author sees
            them in the standard finding flow.
    """
    if kpus is None:
        kpus = load_kpus()
    if process_nodes is None:
        process_nodes = load_process_nodes()
    if cooling_solutions is None:
        cooling_solutions = load_cooling_solutions()

    sku = kpus.get(sku_id)
    if sku is None:
        available = ", ".join(sorted(kpus)) or "(none)"
        raise ContextError(
            f"no KPU SKU with id={sku_id!r}. Available: {available}"
        )

    pn = process_nodes.get(sku.process_node_id)
    if pn is None:
        available = ", ".join(sorted(process_nodes)) or "(none)"
        raise ContextError(
            f"SKU {sku_id!r} references process_node_id="
            f"{sku.process_node_id!r} but it does not resolve. "
            f"Available process nodes: {available}"
        )

    return ValidatorContext(
        sku=sku,
        process_node=pn,
        cooling_solutions=cooling_solutions,
    )
