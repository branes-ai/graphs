"""Unified ComputeProduct loader + KPUEntry adapter (PR #3 of the v1 POC).

Provides the bridge between the legacy KPUEntry catalog
(``data/kpus/<vendor>/``) and the new ComputeProduct catalog
(``data/compute_products/<vendor>/``) during the parallel-migration
phase. Consumers that want a single uniform view of all KPU SKUs --
regardless of which catalog directory they live in -- call
``load_compute_products_unified()`` and get ``ComputeProduct`` instances
back.

For SKUs that exist in both catalogs (e.g., t256 lives in both today
since PR #16 added the ComputeProduct YAML alongside the still-present
legacy YAML), the native ComputeProduct takes precedence. For SKUs only
in the legacy catalog, ``kpu_entry_to_compute_product()`` performs the
mechanical KPUEntry -> ComputeProduct conversion.

Today (PR #3): consumers continue to use ``load_kpus()`` directly. This
module is additive -- it provides the unified interface but no consumer
is migrated yet. Subsequent PRs migrate the mapper, generator, power
model, validators, floorplanner, and CLI tools one by one.

The adapter mapping follows the v1 ComputeProduct schema (per the
assessment doc's chiplet caveat -- per-die ``process_node_id`` /
``silicon_bin`` / ``die_size_mm2`` / ``transistors_billion`` / ``clocks``;
discriminated ``blocks`` union with ``KPUBlock`` carrying the KPU
architecture content).
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas import (
    ComputeProduct,
    Die,
    DieRole,
    KPUBlock,
    KPUEntry,
    LifecycleStatus,
    Market,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
    load_compute_products,
    load_kpus,
)
from embodied_schemas.kpu import (
    KPUArchitecture,
    KPUDieSpec,
    KPUMarket,
    KPUPowerSpec,
)
from embodied_schemas.process_node import ProcessNodeEntry


def kpu_entry_to_compute_product(entry: KPUEntry) -> ComputeProduct:
    """Convert a legacy ``KPUEntry`` to the unified ``ComputeProduct``
    schema. Mechanical field-by-field mapping; preserves all data.

    For monolithic KPU products (the only KPU shape today):
      - ``ComputeProduct.dies`` has exactly one ``Die``
      - ``Die.die_id`` is "kpu_compute" by convention
      - ``Die.die_role`` is ``COMPUTE``
      - ``Die.interconnects`` is empty (intra-die NoC stays in
        ``KPUBlock.noc`` per the v1 schema)
      - ``ComputeProduct.lifecycle`` derives from ``KPUEntry.market.is_discontinued``
        (True -> EOL, False -> PRODUCTION)

    Future chiplet KPU products (none today) will populate
    ``ComputeProduct.dies`` with multiple dies, potentially on different
    process nodes. The adapter would need a corresponding extension.
    """
    return ComputeProduct(
        id=entry.id,
        name=entry.name,
        vendor=entry.vendor,
        kind=ProductKind.CHIP,
        packaging=Packaging(
            kind=PackagingKind.MONOLITHIC,
            num_dies=entry.die.num_dies,
            package_type="monolithic",
        ),
        lifecycle=(
            LifecycleStatus.EOL
            if entry.market.is_discontinued
            else LifecycleStatus.PRODUCTION
        ),
        dies=[
            Die(
                die_id="kpu_compute",
                die_role=DieRole.COMPUTE,
                process_node_id=entry.process_node_id,
                die_size_mm2=entry.die.die_size_mm2,
                transistors_billion=entry.die.transistors_billion,
                silicon_bin=entry.silicon_bin,
                clocks=entry.clocks,
                blocks=[
                    KPUBlock(
                        total_tiles=entry.kpu_architecture.total_tiles,
                        multi_precision_alu=entry.kpu_architecture.multi_precision_alu,
                        tiles=entry.kpu_architecture.tiles,
                        noc=entry.kpu_architecture.noc,
                        memory=entry.kpu_architecture.memory,
                    )
                ],
                interconnects=[],
            )
        ],
        performance=entry.performance,
        power=Power(
            tdp_watts=entry.power.tdp_watts,
            max_power_watts=entry.power.max_power_watts,
            min_power_watts=entry.power.min_power_watts,
            idle_power_watts=entry.power.idle_power_watts,
            default_thermal_profile=entry.power.default_thermal_profile,
            thermal_profiles=entry.power.thermal_profiles,
        ),
        market=Market(
            launch_date=entry.market.launch_date,
            launch_msrp_usd=entry.market.launch_msrp_usd,
            target_market=entry.market.target_market,
            product_family=entry.market.product_family,
            model_tier=entry.market.model_tier,
            is_available=entry.market.is_available,
        ),
        notes=entry.notes,
        datasheet_url=entry.datasheet_url,
        last_updated=entry.last_updated,
    )


def compute_product_to_kpu_entry(
    cp: ComputeProduct,
    process_node: ProcessNodeEntry,
) -> KPUEntry:
    """Reverse adapter: convert a v1-monolithic-KPU ``ComputeProduct``
    back to a legacy ``KPUEntry``.

    Bridge for consumers that haven't been migrated to ComputeProduct
    yet (silicon_floorplan, kpu_power_model, kpu_yaml_loader). Each of
    those owns a follow-up PR; once they accept ComputeProduct directly
    this adapter is unused and can be deleted.

    Requires the ``ProcessNodeEntry`` because the legacy ``KPUDieSpec``
    has redundant copies of foundry / process_name / process_nm that
    the unified schema offloaded to the process-node catalog. The
    caller passes the resolved node alongside the ComputeProduct.

    Raises ValueError if the ComputeProduct is not a v1 monolithic-KPU
    shape (multi-die or non-KPU block kinds aren't representable as
    KPUEntry).
    """
    if len(cp.dies) != 1:
        raise ValueError(
            f"compute_product_to_kpu_entry: KPUEntry can only represent "
            f"one die, got {len(cp.dies)} for {cp.id!r}"
        )
    die = cp.dies[0]
    if len(die.blocks) != 1:
        raise ValueError(
            f"compute_product_to_kpu_entry: KPUEntry can only represent "
            f"one KPUBlock per die, got {len(die.blocks)} for {cp.id!r}"
        )
    block = die.blocks[0]
    if not isinstance(block, KPUBlock):
        raise ValueError(
            f"compute_product_to_kpu_entry: only KPUBlock is supported, "
            f"got {type(block).__name__} for {cp.id!r}"
        )

    return KPUEntry(
        id=cp.id,
        name=cp.name,
        vendor=cp.vendor,
        process_node_id=die.process_node_id,
        die=KPUDieSpec(
            architecture="KPU Tile",
            foundry=process_node.foundry,
            process_nm=process_node.node_nm,
            process_name=process_node.node_name,
            transistors_billion=die.transistors_billion,
            die_size_mm2=die.die_size_mm2,
            is_chiplet=cp.packaging.kind != PackagingKind.MONOLITHIC,
            num_dies=cp.packaging.num_dies,
        ),
        kpu_architecture=KPUArchitecture(
            total_tiles=block.total_tiles,
            multi_precision_alu=block.multi_precision_alu,
            tiles=block.tiles,
            noc=block.noc,
            memory=block.memory,
        ),
        silicon_bin=die.silicon_bin,
        clocks=die.clocks,
        performance=cp.performance,
        power=KPUPowerSpec(
            tdp_watts=cp.power.tdp_watts,
            max_power_watts=cp.power.max_power_watts,
            min_power_watts=cp.power.min_power_watts,
            idle_power_watts=cp.power.idle_power_watts,
            default_thermal_profile=cp.power.default_thermal_profile,
            thermal_profiles=cp.power.thermal_profiles,
        ),
        market=KPUMarket(
            launch_date=cp.market.launch_date,
            launch_msrp_usd=cp.market.launch_msrp_usd,
            target_market=cp.market.target_market,
            product_family=cp.market.product_family,
            model_tier=cp.market.model_tier,
            is_available=cp.market.is_available,
            is_discontinued=(cp.lifecycle == LifecycleStatus.EOL),
        ),
        notes=cp.notes,
        datasheet_url=cp.datasheet_url,
        last_updated=cp.last_updated,
    )


def load_compute_products_unified(
    *,
    prefer_native: bool = True,
) -> dict[str, ComputeProduct]:
    """Return all KPU SKUs as ``ComputeProduct`` instances, drawing from
    both the legacy ``data/kpus/`` catalog (via the adapter) and the new
    ``data/compute_products/`` catalog (native).

    During the parallel-migration phase a SKU may live in either or both
    catalogs. When it lives in both, ``prefer_native=True`` (default)
    uses the native ``ComputeProduct`` YAML; ``prefer_native=False``
    uses the adapted ``KPUEntry`` (useful for testing that the two are
    content-equivalent).

    Returns a dict keyed by SKU id. The combined catalog is the union of
    both sources -- once all SKUs are migrated to the new path, the
    legacy directory can be deleted and this function becomes a thin
    wrapper around ``load_compute_products()``.
    """
    legacy_kpus = load_kpus()
    native_cps = load_compute_products()

    unified: dict[str, ComputeProduct] = {}

    # Adapter pass: every legacy KPU enters as adapted CP
    for sku_id, entry in legacy_kpus.items():
        unified[sku_id] = kpu_entry_to_compute_product(entry)

    # Native pass: overwrite (if prefer_native) or skip (if not)
    for sku_id, cp in native_cps.items():
        if prefer_native or sku_id not in unified:
            unified[sku_id] = cp

    return unified


def get_compute_product(
    sku_id: str,
    *,
    prefer_native: bool = True,
) -> Optional[ComputeProduct]:
    """Convenience: return one ``ComputeProduct`` by id, or ``None`` if
    the SKU is not in either catalog. Same precedence rule as
    ``load_compute_products_unified()``."""
    return load_compute_products_unified(prefer_native=prefer_native).get(sku_id)
