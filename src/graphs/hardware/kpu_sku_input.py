"""KPU SKU input spec -- the architect-facing authoring shape.

A ``KPUSKUInputSpec`` is what an architect writes to define a new KPU
SKU: architectural choices (tile mix, NoC, memory subsystem), silicon
bin (per-block transistor coefficients), thermal profiles (with cooling
solution refs), market info. The generator
(``graphs.hardware.kpu_sku_generator.generate_kpu_sku``) reads this and
produces a fully-populated ``embodied_schemas.KPUEntry`` with the
roll-up fields (die size, transistor count, performance numbers,
rolled-up power) computed from the spec.

Design: the input spec is intentionally smaller than KPUEntry so that
the architect doesn't have to keep multiple roll-up numbers in sync by
hand -- the generator owns those derivations. The spec is also a
Pydantic model so YAML inputs get validated with the same rigor as
catalog entries.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field, model_validator

from embodied_schemas.kpu import (
    KPUArchitecture,
    KPUClocks,
    KPUMarket,
    KPUSiliconBin,
    KPUThermalProfile,
)


_USE_KEYS = {"use", "num_tiles", "overrides"}


def resolve_tile_uses(tiles: list, library: Optional[Mapping[str, Any]] = None) -> list:
    """Expand tile-class library references in a tile list (graphs#268 C3).

    A tile written as ``{use: <library id>, num_tiles: N, overrides: {...}}``
    becomes the library entry instantiated with ``N`` tiles and the
    overrides applied (``KPUTileClassEntry.instantiate``), so the resolved
    tile carries ``tile_class_ref``. Other tiles pass through unchanged.

    ``library`` defaults to ``embodied_schemas.load_kpu_tile_classes()``
    (which honors the ``KPU_TILE_DATA_DIR`` private overlay).
    """
    if not any(isinstance(t, dict) and "use" in t for t in tiles):
        return tiles
    if library is None:
        from embodied_schemas import load_kpu_tile_classes

        library = load_kpu_tile_classes()
    out = []
    for t in tiles:
        if not (isinstance(t, dict) and "use" in t):
            out.append(t)
            continue
        extra = sorted(set(t) - _USE_KEYS)
        if extra:
            raise ValueError(
                f"tile 'use: {t['use']}' has unknown keys {extra}; put tile fields "
                f"under 'overrides'"
            )
        entry = library.get(t["use"])
        if entry is None:
            raise ValueError(
                f"tile 'use: {t['use']}' is not in the tile-class library "
                f"(available: {sorted(library)})"
            )
        if "num_tiles" not in t:
            raise ValueError(f"tile 'use: {t['use']}' needs num_tiles")
        out.append(
            entry.instantiate(t["num_tiles"], **(t.get("overrides") or {})).model_dump(
                mode="json"
            )
        )
    return out


class KPUSKUInputSpec(BaseModel):
    """Architect-facing input spec for a new KPU SKU.

    Shape: every field that the architect *chooses*. Generator-derived
    fields (die.transistors_billion, die.die_size_mm2, performance.*,
    power.tdp_watts roll-up, power.idle_power_watts) are absent here --
    the generator computes them from the architecture + silicon_bin +
    referenced ProcessNode.

    The thermal profiles are still authored explicitly (architect picks
    the TDP / clock / cooling pairings); the generator validates those
    are achievable but doesn't guess them.
    """

    # Identity
    id: str = Field(..., description="Unique id, e.g., 'kpu_t1024_32x32_lp5x4_5nm_tsmc_hpc'")
    name: str = Field(..., description="Human-readable name")
    vendor: str = Field(..., description="Vendor, e.g., 'stillwater'")

    # Process node reference -- generator looks up densities, energies,
    # leakage from this entry.
    process_node_id: str = Field(
        ..., description="References data/process-nodes/<foundry>/<node>.yaml"
    )

    # Architecture + silicon decomposition
    kpu_architecture: KPUArchitecture = Field(
        ..., description="Tile mix, NoC, memory subsystem"
    )
    silicon_bin: KPUSiliconBin = Field(
        ..., description="Per-block transistor decomposition"
    )

    # Clocks (chip-level)
    clocks: KPUClocks = Field(...)

    # Per-profile (clock + TDP + cooling). Generator validates that
    # profile.tdp_watts fits cooling envelope and accommodates leakage,
    # but does NOT recompute TDP -- it's the architect's design choice.
    thermal_profiles: list[KPUThermalProfile] = Field(...)
    default_thermal_profile: str = Field(
        ..., description="Name of the default profile in thermal_profiles"
    )

    # Market metadata passes through unchanged
    market: KPUMarket = Field(...)

    # Optional metadata
    notes: str = Field("")
    datasheet_url: str | None = Field(None)
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _expand_library_tiles(cls, data: Any) -> Any:
        """Resolve ``use:`` tile references (``resolve_tile_uses``) and,
        when omitted, fill in ``total_tiles`` from the tile counts."""
        if not isinstance(data, dict) or not isinstance(data.get("kpu_architecture"), dict):
            return data
        arch = dict(data["kpu_architecture"])
        tiles = arch.get("tiles")
        if isinstance(tiles, list):
            tiles = resolve_tile_uses(tiles)
            arch["tiles"] = tiles
            if "total_tiles" not in arch and all(isinstance(t, dict) for t in tiles):
                arch["total_tiles"] = sum(t.get("num_tiles", 0) for t in tiles)
        return {**data, "kpu_architecture": arch}
