"""KPU SKU generator.

Turns a ``KPUSKUInputSpec`` into a fully-populated ``ComputeProduct`` by
computing the roll-up fields from the architectural / silicon_bin / NoC
inputs and the referenced ``ProcessNodeEntry``:

* ``dies[0].transistors_billion`` -- Σ silicon_bin block transistors / 1000
* ``dies[0].die_size_mm2`` -- Σ silicon_bin block area (= transistors / density)
* ``performance.{int8_tops, bf16_tflops, fp32_tflops, int4_tops}`` --
  Σ(tile.num_tiles × ops_per_tile_per_clock) × default-profile clock
* ``power.{tdp_watts, max_power_watts, min_power_watts,
  thermal_profiles[].tdp_watts}`` -- DERIVED via the kpu_power_model
  from clock + architecture + ProcessNode energies under a
  ``WorkloadAssumption``. The architect's hand-authored tdp_watts in
  the input spec is ignored. The architect chooses clocks and cooling;
  TDP is the consequence.
* ``power.idle_power_watts`` -- chip-wide leakage from ProcessNode
  ``leakage_w_per_mm2`` × per-block area
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
    KPUArchitecture,
    KPUMarket,
    KPUTheoreticalPerformance,
    KPUThermalProfile,
)
from embodied_schemas.process_node import ProcessNodeEntry

from .kpu_power_model import (
    DEFAULT_WORKLOAD,
    WorkloadAssumption,
    compute_thermal_profile_tdp_w,
)
from .kpu_sku_input import KPUSKUInputSpec
from .sku_validators.silicon_math import (
    SiliconMathError,
    resolve_block_area,
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
                blocks=[
                    KPUBlock(
                        total_tiles=spec.kpu_architecture.total_tiles,
                        multi_precision_alu=spec.kpu_architecture.multi_precision_alu,
                        tiles=spec.kpu_architecture.tiles,
                        noc=spec.kpu_architecture.noc,
                        memory=spec.kpu_architecture.memory,
                    )
                ],
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
) -> ComputeProduct:
    """Produce a fully-populated ComputeProduct from an input spec.

    Args:
        spec: The architect-authored input.
        process_nodes: Optional pre-loaded process-node catalog. Falls
            back to ``embodied_schemas.load_process_nodes()`` if absent.

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

    # ---- Power roll-up ----
    # TDP per profile is DERIVED from the chip configuration (clock,
    # architecture, ProcessNode energies) under the workload assumption
    # in graphs.hardware.kpu_power_model.WorkloadAssumption. The
    # architect's hand-authored tdp_watts in the input spec is ignored
    # -- it's the consequence of the design, not an input.
    workload = workload or DEFAULT_WORKLOAD
    derived_profiles: list[KPUThermalProfile] = []
    for p in spec.thermal_profiles:
        derived_tdp = compute_thermal_profile_tdp_w(spec, p, node, workload)
        derived_profiles.append(p.model_copy(update={"tdp_watts": derived_tdp}))
    derived_default = next(
        dp for dp in derived_profiles if dp.name == spec.default_thermal_profile
    )
    max_w = max(p.tdp_watts for p in derived_profiles)
    min_w = min(p.tdp_watts for p in derived_profiles)
    leakage_w = total_chip_leakage_w(placeholder_cp, node)
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
                blocks=[
                    KPUBlock(
                        total_tiles=spec.kpu_architecture.total_tiles,
                        multi_precision_alu=spec.kpu_architecture.multi_precision_alu,
                        tiles=spec.kpu_architecture.tiles,
                        noc=spec.kpu_architecture.noc,
                        memory=spec.kpu_architecture.memory,
                    )
                ],
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


def apply_pe_array_override(
    spec: KPUSKUInputSpec,
    pe_array_rows: int,
    pe_array_cols: int,
) -> KPUSKUInputSpec:
    """Resize the PE array on every tile class in a spec.

    Returns a new ``KPUSKUInputSpec`` with ``pe_array_rows`` /
    ``pe_array_cols`` set to the given dimensions on every tile class,
    and ``ops_per_tile_per_clock`` rescaled by ``new_pes / old_pes`` so
    the per-PE op throughput (e.g., int8=2 ops/PE/clock) is preserved.
    Pipeline fill / drain cycles are also rescaled to track the longer
    PE-array dimension, matching the family convention (T64/T128 use 32
    fill/drain at 32x32; T768 uses 16 at 16x8).

    Designed for roadmap sweeps -- run the generator across PE-array
    sizes without hand-editing each tile class. Caller is responsible
    for choosing dimensions that make architectural sense (e.g., a
    32x32 array is dense in NoC routers per PE; a 16x16 leaves more
    NoC headroom).

    Note: silicon_bin coefficients are *not* touched -- per-PE blocks
    use ``kind=per_pe`` so total area auto-scales with the new PE
    count. Per-tile and fixed blocks are insensitive to PE size.
    """
    if pe_array_rows <= 0 or pe_array_cols <= 0:
        raise ValueError(
            f"pe_array dimensions must be positive; got "
            f"rows={pe_array_rows}, cols={pe_array_cols}"
        )
    new_pes = pe_array_rows * pe_array_cols
    new_pipeline_depth = max(pe_array_rows, pe_array_cols)
    new_tiles = []
    for t in spec.kpu_architecture.tiles:
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
    new_arch = spec.kpu_architecture.model_copy(update={"tiles": new_tiles})
    return spec.model_copy(update={"kpu_architecture": new_arch})


def input_spec_from_compute_product(cp: ComputeProduct) -> KPUSKUInputSpec:
    """Extract a KPUSKUInputSpec from an existing ComputeProduct.

    Used by tests for round-trip verification (load existing YAML ->
    extract spec -> regenerate -> diff against original) and by the CLI
    when an architect wants to start from an existing SKU as a template.

    Translates the ComputeProduct's per-die structure back to the
    spec's flat shape. v1 KPU monolithic products are assumed (one Die,
    one KPUBlock); the input spec doesn't model multi-die yet.

    Raises ``GeneratorError`` for shape mismatches (empty dies, empty
    blocks, non-KPUBlock first block). The Pydantic schema enforces
    dies/blocks min_length=1 at construction, but instances built via
    ``model_construct()`` bypass validation -- the explicit checks
    surface the failure with a clear message rather than IndexError /
    AttributeError.
    """
    if not cp.dies:
        raise GeneratorError(
            f"input_spec_from_compute_product: ComputeProduct {cp.id!r} "
            f"has no dies; expected one Die for v1 KPU monolithic"
        )
    die = cp.dies[0]
    if not die.blocks:
        raise GeneratorError(
            f"input_spec_from_compute_product: ComputeProduct {cp.id!r} "
            f"die {die.die_id!r} has no blocks; expected one KPUBlock"
        )
    block = die.blocks[0]
    if not isinstance(block, KPUBlock):
        raise GeneratorError(
            f"input_spec_from_compute_product: ComputeProduct {cp.id!r} "
            f"die {die.die_id!r} first block is "
            f"{type(block).__name__}, expected KPUBlock"
        )
    return KPUSKUInputSpec(
        id=cp.id,
        name=cp.name,
        vendor=cp.vendor,
        process_node_id=die.process_node_id,
        kpu_architecture=KPUArchitecture(
            total_tiles=block.total_tiles,
            multi_precision_alu=block.multi_precision_alu,
            tiles=block.tiles,
            noc=block.noc,
            memory=block.memory,
        ),
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
