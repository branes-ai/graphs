"""KPU SKU generator.

Turns a ``KPUSKUInputSpec`` into a fully-populated ``ComputeProduct`` by
computing the roll-up fields from the architectural / silicon_bin / NoC
inputs and the referenced ``ProcessNodeEntry``:

* ``dies[0].transistors_billion`` -- sum of silicon_bin block transistors / 1000,
  plus the tile-carried silicon (``silicon_math.carried_silicon``)
* ``dies[0].die_size_mm2`` -- sum of silicon_bin block area (= transistors / density),
  plus the tile-carried silicon's area
* ``performance.{int8_tops, bf16_tflops, fp32_tflops, int4_tops}`` --
  sum(tile.num_tiles * ops_per_tile_per_clock) * default-profile clock. A
  heterogeneous architecture also gets the roll-up by tile kind and the
  fixed-function throughput (``derive_kpu_performance``, graphs#268 C3)
* ``power.{tdp_watts, max_power_watts, min_power_watts,
  thermal_profiles[].tdp_watts}`` -- DERIVED via the kpu_power_model
  from clock + architecture + ProcessNode energies under a
  ``WorkloadAssumption``. The architect's hand-authored tdp_watts in
  the input spec is ignored. The architect chooses clocks and cooling;
  TDP is the consequence.
* ``power.idle_power_watts`` -- chip-wide leakage from ProcessNode
  ``leakage_w_per_mm2`` * per-block area
* ``lifecycle`` -- derived from ``KPUMarket.is_discontinued`` (legacy
  market shape on the input spec): EOL if discontinued else PRODUCTION

The generator never modifies architect-provided fields. If
``input_spec.silicon_bin`` is incomplete (missing blocks) or
``kpu_architecture`` is inconsistent with claimed performance, the
generator still produces output -- the validator framework catches the
issue when ``--validate`` runs against the generated SKU.

The ``generate_kpu_sku`` function is pure: it never reads disk. Pass
in pre-loaded ``process_nodes`` / ``cooling_solutions`` dicts (the CLI
loads them from the catalog). This keeps the generator deterministic
for tests.
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas import (
    ComputeProduct,
    Die,
    derive_kpu_performance,
    DieRole,
    KPUBlock,
    LifecycleStatus,
    Market,
    Packaging,
    PackagingKind,
    Power,
    ProductKind,
    load_process_nodes,
)
from embodied_schemas.kpu import (
    KPUMarket,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
    KPUTileSpec,
)
from embodied_schemas.process_node import ProcessNodeEntry

from .kpu_power_model import (
    DEFAULT_WORKLOAD,
    WorkloadAssumption,
    compute_thermal_profile_tdp_w,
)
from .kpu_access import KPUBlockLookupError, kpu_block_of, kpu_die_of
from .kpu_sku_input import KPUSKUInputSpec
from .sku_validators.silicon_math import (
    SiliconMathError,
    resolve_block_area,
    resolve_carried_areas,
    total_chip_leakage_w,
)


class GeneratorError(Exception):
    """Raised when a SKU cannot be generated (missing process node,
    silicon_bin contradicts itself, default profile not in profiles, ...).

    The CLI surfaces this as a single clear message instead of letting
    the failure surface as a cascade of validator findings."""


def _placeholder_for_resolution(spec: KPUSKUInputSpec) -> ComputeProduct:
    """Build a ComputeProduct with placeholder die_size / transistor
    fields so silicon_math helpers can walk the silicon_bin during
    transistor-count resolution. Die size and transistor count get
    filled in with real numbers in the second pass."""
    return ComputeProduct(
        id=spec.id,
        name=spec.name,
        vendor=spec.vendor,
        kind=ProductKind.CHIP,
        packaging=Packaging(
            kind=PackagingKind.MONOLITHIC,
            num_dies=1,
            package_type="monolithic",
        ),
        dies=[
            Die(
                die_id="kpu_compute",
                die_role=DieRole.COMPUTE,
                process_node_id=spec.process_node_id,
                die_size_mm2=1.0,           # placeholder; recomputed below
                transistors_billion=1.0,    # placeholder
                silicon_bin=spec.silicon_bin,
                clocks=spec.clocks,
                blocks=[KPUBlock.from_architecture(spec.kpu_architecture)],
                interconnects=[],
            )
        ],
        performance=KPUTheoreticalPerformance(
            int8_tops=0.0, bf16_tflops=0.0, fp32_tflops=0.0,
        ),
        power=Power(
            tdp_watts=1.0,
            max_power_watts=1.0,
            min_power_watts=1.0,
            default_thermal_profile=spec.default_thermal_profile,
            thermal_profiles=spec.thermal_profiles,
        ),
        market=Market(
            launch_date=spec.market.launch_date,
            launch_msrp_usd=spec.market.launch_msrp_usd,
            target_market=spec.market.target_market,
            product_family=spec.market.product_family,
            model_tier=spec.market.model_tier,
            is_available=spec.market.is_available,
        ),
        notes=spec.notes,
        datasheet_url=spec.datasheet_url,
        last_updated=spec.last_updated,
    )


def generate_kpu_sku(
    spec: KPUSKUInputSpec,
    *,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    workload: Optional[WorkloadAssumption] = None,
    performance_rollup: Optional[bool] = None,
) -> ComputeProduct:
    """Produce a fully-populated ComputeProduct from an input spec.

    Args:
        spec: The architect-authored input.
        process_nodes: Optional pre-loaded process-node catalog. Falls
            back to ``embodied_schemas.load_process_nodes()`` if absent.
        performance_rollup: Emit the performance roll-up by tile kind
            (``peak_ops_per_sec_by_precision``, ``by_tile_kind``,
            ``fixed_function_throughput``). None (default) = only for a
            heterogeneous architecture (any tile that is not pe_fabric),
            so uniform legacy SKUs regenerate byte-identically.

    Raises:
        GeneratorError: if the spec's process_node_id doesn't resolve,
        if the default thermal profile name isn't in the profile list,
        or if no silicon_bin block resolves cleanly.

    Note:
        Cooling-solution refs on ``spec.thermal_profiles`` are NOT
        resolved here -- the validator framework's
        ``cross_ref_consistency`` check surfaces unresolvable ids when
        the caller runs the full registry against the generated SKU.
    """
    if process_nodes is None:
        process_nodes = load_process_nodes()

    # ---- Resolve the process node ----
    node = process_nodes.get(spec.process_node_id)
    if node is None:
        available = ", ".join(sorted(process_nodes)) or "(none)"
        raise GeneratorError(
            f"spec references process_node_id={spec.process_node_id!r} "
            f"but it does not resolve. Available: {available}"
        )

    # ---- Resolve the default thermal profile ----
    default_profile = next(
        (p for p in spec.thermal_profiles if p.name == spec.default_thermal_profile),
        None,
    )
    if default_profile is None:
        names = [p.name for p in spec.thermal_profiles]
        raise GeneratorError(
            f"default_thermal_profile={spec.default_thermal_profile!r} "
            f"is not in thermal_profiles ({names})"
        )

    # ---- Roll up silicon_bin -> die ----
    # Build a placeholder ComputeProduct so silicon_math helpers can walk
    # kpu_architecture for PER_PE / PER_KIB / PER_ROUTER / PER_CONTROLLER
    # expansions. The placeholder has bogus die_size/transistors, which
    # we replace with the rolled-up values from the silicon_bin pass.
    placeholder_cp = _placeholder_for_resolution(spec)

    total_area = 0.0
    total_mtx = 0.0
    unresolved: list[str] = []
    for block in spec.silicon_bin.blocks:
        try:
            ba = resolve_block_area(block, placeholder_cp, node)
        except SiliconMathError as exc:
            unresolved.append(f"{block.name}: {exc}")
            continue
        total_area += ba.area_mm2
        total_mtx += ba.transistors_mtx

    # Tile-carried silicon (datapaths, tile-local SRAM, overlays, fixed-
    # function cores). Empty for uniform legacy SKUs. Pieces in a library
    # the node lacks are skipped, like unresolved silicon_bin blocks.
    try:
        carried = resolve_carried_areas(placeholder_cp, node, process_nodes)
    except SiliconMathError as exc:
        raise GeneratorError(f"tile-carried silicon: {exc}") from exc
    for ba in carried:
        total_area += ba.area_mm2
        total_mtx += ba.transistors_mtx

    if total_area <= 0 or total_mtx <= 0:
        raise GeneratorError(
            "no silicon_bin block could be resolved against the process "
            "node. Unresolved: " + "; ".join(unresolved)
            if unresolved
            else "silicon_bin is empty or every block has 0 area."
        )

    derived_die_size_mm2 = round(total_area, 1)
    derived_transistors_billion = round(total_mtx / 1000.0, 3)

    # ---- Performance roll-up at default profile clock ----
    clock_hz = default_profile.clock_mhz * 1e6

    def _peak_for_precision(precision: str) -> float:
        ops_per_clock = sum(
            t.num_tiles * t.ops_per_tile_per_clock.get(precision, 0)
            for t in spec.kpu_architecture.tiles
        )
        return ops_per_clock * clock_hz / 1e12  # T-ops/s

    int8_tops = round(_peak_for_precision("int8"), 1)
    bf16_tflops = round(_peak_for_precision("bf16"), 1)
    fp32_tflops = round(_peak_for_precision("fp32"), 2)
    int4_peak = _peak_for_precision("int4")
    int4_tops = round(int4_peak, 1) if int4_peak > 0 else None

    performance = KPUTheoreticalPerformance(
        int8_tops=int8_tops,
        bf16_tflops=bf16_tflops,
        fp32_tflops=fp32_tflops,
        int4_tops=int4_tops,
    )
    if performance_rollup is None:
        performance_rollup = not all(
            isinstance(t, KPUTileSpec) for t in spec.kpu_architecture.tiles
        )
    if performance_rollup:
        rollup = derive_kpu_performance(spec.kpu_architecture.tiles, default_profile.clock_mhz)
        performance = performance.model_copy(update={
            "peak_ops_per_sec_by_precision": rollup.peak_ops_per_sec_by_precision,
            "by_tile_kind": rollup.by_tile_kind,
            "fixed_function_throughput": rollup.fixed_function_throughput,
        })
        performance = KPUTheoreticalPerformance.model_validate(performance.model_dump())

    # ---- Power roll-up ----
    # TDP per profile is DERIVED from the chip configuration (clock,
    # architecture, ProcessNode energies) under the workload assumption
    # in graphs.hardware.kpu_power_model.WorkloadAssumption. The
    # architect's hand-authored tdp_watts in the input spec is ignored
    # -- it's the consequence of the design, not an input.
    workload = workload or DEFAULT_WORKLOAD
    derived_profiles: list[KPUThermalProfile] = []
    for p in spec.thermal_profiles:
        derived_tdp = compute_thermal_profile_tdp_w(spec, p, node, workload, process_nodes)
        derived_profiles.append(p.model_copy(update={"tdp_watts": derived_tdp}))
    derived_default = next(
        dp for dp in derived_profiles if dp.name == spec.default_thermal_profile
    )
    max_w = max(p.tdp_watts for p in derived_profiles)
    min_w = min(p.tdp_watts for p in derived_profiles)
    leakage_w = total_chip_leakage_w(placeholder_cp, node, process_nodes)
    idle_w = round(leakage_w, 2) if leakage_w > 0 else None

    power = Power(
        tdp_watts=derived_default.tdp_watts,
        max_power_watts=max_w,
        min_power_watts=min_w,
        idle_power_watts=idle_w,
        default_thermal_profile=spec.default_thermal_profile,
        thermal_profiles=derived_profiles,
    )

    # ---- Construct the final ComputeProduct ----
    return ComputeProduct(
        id=spec.id,
        name=spec.name,
        vendor=spec.vendor,
        kind=ProductKind.CHIP,
        packaging=Packaging(
            kind=PackagingKind.MONOLITHIC,
            num_dies=1,
            package_type="monolithic",
        ),
        lifecycle=(
            LifecycleStatus.EOL
            if spec.market.is_discontinued
            else LifecycleStatus.PRODUCTION
        ),
        dies=[
            Die(
                die_id="kpu_compute",
                die_role=DieRole.COMPUTE,
                process_node_id=spec.process_node_id,
                die_size_mm2=derived_die_size_mm2,
                transistors_billion=derived_transistors_billion,
                silicon_bin=spec.silicon_bin,
                clocks=spec.clocks,
                blocks=[KPUBlock.from_architecture(spec.kpu_architecture)],
                interconnects=[],
            )
        ],
        performance=performance,
        power=power,
        market=Market(
            launch_date=spec.market.launch_date,
            launch_msrp_usd=spec.market.launch_msrp_usd,
            target_market=spec.market.target_market,
            product_family=spec.market.product_family,
            model_tier=spec.market.model_tier,
            is_available=spec.market.is_available,
        ),
        notes=spec.notes,
        datasheet_url=spec.datasheet_url,
        last_updated=spec.last_updated,
    )


def _find_tile(spec: KPUSKUInputSpec, ref: str):
    tiles = spec.kpu_architecture.tiles
    match = [t for t in tiles if t.tile_class_id == ref] or [t for t in tiles if t.tile_type == ref]
    if len(match) != 1:
        raise ValueError(
            f"tile class {ref!r} {'is ambiguous' if match else 'is unknown'} "
            f"(tile_class_id: {sorted(t.tile_class_id for t in tiles)})"
        )
    return match[0]


def _revalidated(spec: KPUSKUInputSpec, tiles: list, **arch_update) -> KPUSKUInputSpec:
    """A copy of ``spec`` with new tiles, re-validated so the schema checks
    (datapath / overlay consistency, site accounting, ...) run."""
    data = spec.model_dump(mode="json")
    data["kpu_architecture"].update(tiles=[t.model_dump(mode="json") for t in tiles], **arch_update)
    return KPUSKUInputSpec.model_validate(data)


def apply_pe_array_override(
    spec: KPUSKUInputSpec,
    pe_array_rows: int,
    pe_array_cols: int,
    tile_class: Optional[str] = None,
) -> KPUSKUInputSpec:
    """Resize the PE array of pe_fabric tile classes in a spec.

    Returns a new ``KPUSKUInputSpec`` with ``pe_array_rows`` /
    ``pe_array_cols`` set to the given dimensions and
    ``ops_per_tile_per_clock`` rescaled by ``new_pes / old_pes`` so the
    per-PE op throughput (e.g., int8=2 ops/PE/clock) is preserved.
    Pipeline fill / drain cycles are also rescaled to track the longer
    PE-array dimension, matching the family convention (T64/T128 use 32
    fill/drain at 32x32; T768 uses 16 at 16x8).

    * ``tile_class`` None: every pe_fabric tile class (every class of a
      uniform legacy SKU, so the result is unchanged from before C3).
      Systolic and fixed-function classes are left alone.
    * ``tile_class`` a ``tile_class_id`` or ``tile_type``: only that class,
      which must be pe_fabric (graphs#268 C3, ``--pe-array CLASS=RxC``).

    The result is re-validated, so a datapath or overlay that no longer
    fits raises ``ValueError``.

    Designed for roadmap sweeps -- run the generator across PE-array
    sizes without hand-editing each tile class.

    Note: silicon_bin coefficients are *not* touched -- per-PE blocks
    use ``kind=per_pe`` so total area auto-scales with the new PE
    count. Per-tile and fixed blocks are insensitive to PE size.
    """
    if pe_array_rows <= 0 or pe_array_cols <= 0:
        raise ValueError(
            f"pe_array dimensions must be positive; got "
            f"rows={pe_array_rows}, cols={pe_array_cols}"
        )
    if tile_class is not None:
        target = _find_tile(spec, tile_class)
        if not isinstance(target, KPUTileSpec):
            raise ValueError(
                f"tile class {tile_class!r} is {target.tile_kind.value}; --pe-array "
                f"resizes pe_fabric classes only"
            )
        targets = {target.tile_class_id}
    else:
        targets = {t.tile_class_id for t in spec.kpu_architecture.tiles if isinstance(t, KPUTileSpec)}
    new_pes = pe_array_rows * pe_array_cols
    new_pipeline_depth = max(pe_array_rows, pe_array_cols)
    new_tiles = []
    for t in spec.kpu_architecture.tiles:
        if t.tile_class_id not in targets:
            new_tiles.append(t)
            continue
        old_pes = t.pe_array_rows * t.pe_array_cols
        scale = new_pes / old_pes
        new_ops = {
            precision: ops * scale
            for precision, ops in t.ops_per_tile_per_clock.items()
        }
        new_tiles.append(
            t.model_copy(
                update={
                    "pe_array_rows": pe_array_rows,
                    "pe_array_cols": pe_array_cols,
                    "ops_per_tile_per_clock": new_ops,
                    "pipeline_fill_cycles": new_pipeline_depth,
                    "pipeline_drain_cycles": new_pipeline_depth,
                }
            )
        )
    return _revalidated(spec, new_tiles)


def apply_tile_mix(spec: KPUSKUInputSpec, mix: dict[str, int]) -> KPUSKUInputSpec:
    """Set ``num_tiles`` for the named tile classes (graphs#268 C3,
    ``--tile-mix CLASS=N,...``). Classes are named by ``tile_class_id`` or
    ``tile_type``; counts must be positive.

    ``total_tiles`` is recomputed. With a checkerboard in ``auto``
    placement, ``spare_sites`` is recomputed; a mix that needs more sites
    than the grid has is an error, as is a checkerboard with an explicit
    ``placement_map`` (edit the map instead).
    """
    counts: dict[str, int] = {}
    for ref, n in mix.items():
        if n <= 0:
            raise ValueError(f"tile mix: {ref}={n}; counts must be positive")
        counts[_find_tile(spec, ref).tile_class_id] = n
    new_tiles = [
        t.model_copy(update={"num_tiles": counts[t.tile_class_id]}) if t.tile_class_id in counts else t
        for t in spec.kpu_architecture.tiles
    ]
    update: dict = {"total_tiles": sum(t.num_tiles for t in new_tiles)}
    cb = spec.kpu_architecture.checkerboard
    if cb is not None:
        if cb.placement_map is not None:
            raise ValueError(
                "tile mix: the checkerboard has an explicit placement_map; edit the map "
                "(and spare_sites) instead"
            )
        used = sum(t.total_sites for t in new_tiles)
        spare = cb.compute_sites.sites - used
        if spare < 0:
            raise ValueError(
                f"tile mix needs {used} compute sites but the checkerboard has "
                f"{cb.compute_sites.rows}x{cb.compute_sites.cols} = {cb.compute_sites.sites}"
            )
        update["checkerboard"] = {**cb.model_dump(mode="json"), "spare_sites": spare}
    return _revalidated(spec, new_tiles, **update)


def input_spec_from_compute_product(cp: ComputeProduct) -> KPUSKUInputSpec:
    """Extract a KPUSKUInputSpec from an existing ComputeProduct.

    Used by tests for round-trip verification (load existing YAML ->
    extract spec -> regenerate -> diff against original) and by the CLI
    when an architect wants to start from an existing SKU as a template.

    Translates the ComputeProduct's per-die structure back to the
    spec's flat shape. The KPU block and the die carrying it are located
    by kind (``kpu_die_of`` / ``kpu_block_of``), so the KPU need not be
    the first block of the first die; the spec captures that one KPU die
    (the input spec doesn't model multi-die packages).

    Raises ``GeneratorError`` if the product has no KPU block or more than
    one (including instances built via ``model_construct()`` that bypass
    the schema's min_length checks).
    """
    try:
        die = kpu_die_of(cp)
        block = kpu_block_of(cp)
    except KPUBlockLookupError as exc:
        raise GeneratorError(f"input_spec_from_compute_product: {exc}") from exc
    return KPUSKUInputSpec(
        id=cp.id,
        name=cp.name,
        vendor=cp.vendor,
        process_node_id=die.process_node_id,
        kpu_architecture=block.to_architecture(),
        silicon_bin=die.silicon_bin,
        clocks=die.clocks,
        thermal_profiles=cp.power.thermal_profiles,
        default_thermal_profile=cp.power.default_thermal_profile,
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
