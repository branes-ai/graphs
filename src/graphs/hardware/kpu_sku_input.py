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

from pydantic import BaseModel, Field

from embodied_schemas.kpu import (
    KPUArchitecture,
    KPUClocks,
    KPUMarket,
    KPUSiliconBin,
    KPUThermalProfile,
)


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
