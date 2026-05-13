"""Unified ComputeProduct loader.

Thin wrapper around ``embodied_schemas.load_compute_products()`` that
the graphs codebase consumes from. Originally (PR #156) this also
adapted KPUEntry instances from the legacy ``data/kpus/`` catalog,
but that catalog is retired (embodied-schemas PR #18). What remains
is a single canonical loader returning ``dict[str, ComputeProduct]``.

Kept as a separate module from ``embodied_schemas.load_compute_products``
itself so the graphs codebase has one consistent entry point, and so
the existing call sites across the consumer codebase don't need to
change as the underlying loader implementation evolves.
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas import (
    ComputeProduct,
    load_compute_products,
)


def load_compute_products_unified() -> dict[str, ComputeProduct]:
    """Return every KPU SKU as a ``ComputeProduct``.

    Reads ``data/compute_products/<vendor>/<id>.yaml`` from the
    embodied-schemas catalog. Returns a dict keyed by SKU id.

    Returns an empty dict if the catalog directory is missing
    (graceful behavior matching the underlying loader).
    """
    return load_compute_products()


def get_compute_product(sku_id: str) -> Optional[ComputeProduct]:
    """Convenience: return one ``ComputeProduct`` by id, or ``None`` if
    the SKU is not in the catalog."""
    return load_compute_products_unified().get(sku_id)
