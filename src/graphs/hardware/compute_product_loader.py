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

from pathlib import Path
from typing import Optional, Union

import yaml
from embodied_schemas import (
    ComputeProduct,
    load_compute_products,
)


class ComputeProductFileError(ValueError):
    """The file is not a readable ComputeProduct YAML / JSON document."""


def load_compute_products_unified() -> dict[str, ComputeProduct]:
    """Return every catalog ``ComputeProduct`` (all block kinds: KPU, GPU,
    CPU, NPU, ...). Filter with ``kpu_access.has_kpu_block`` for KPU SKUs.

    Reads ``data/compute_products/<vendor>/<id>.yaml`` from the
    embodied-schemas catalog. Returns a dict keyed by product id.

    Returns an empty dict if the catalog directory is missing
    (graceful behavior matching the underlying loader).
    """
    return load_compute_products()


def get_compute_product(sku_id: str) -> Optional[ComputeProduct]:
    """Convenience: return one ``ComputeProduct`` by id, or ``None`` if
    the SKU is not in the catalog."""
    return load_compute_products_unified().get(sku_id)


def load_compute_product_file(path: Union[str, Path]) -> ComputeProduct:
    """Read one ``ComputeProduct`` from a YAML or JSON file.

    A generated SKU (``cli/generate_kpu_sku.py``) is a file long before it
    is a catalog entry, and a heterogeneous design is inspected and
    validated while it is still being iterated on. The inspection CLIs take
    ``--from-file`` so that loop does not require a round trip through the
    embodied-schemas catalog (graphs#268 C6).

    Raises:
        ComputeProductFileError: the file is missing, unparseable, or does
            not validate as a ``ComputeProduct``.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComputeProductFileError(f"cannot read {path}: {exc}") from exc
    try:
        # safe_load parses JSON too: JSON is a subset of YAML.
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ComputeProductFileError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ComputeProductFileError(
            f"{path} does not contain a ComputeProduct mapping"
        )
    try:
        return ComputeProduct.model_validate(data)
    except Exception as exc:
        raise ComputeProductFileError(
            f"{path} does not validate as a ComputeProduct: {exc}"
        ) from exc
