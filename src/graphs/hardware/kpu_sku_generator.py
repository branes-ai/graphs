"""KPU SKU generator.

Turns a ``KPUSKUInputSpec`` into a fully-populated ``KPUEntry`` by
computing the roll-up fields from the architectural / silicon_bin / NoC
inputs and the referenced ``ProcessNodeEntry``:

* ``die.transistors_billion`` -- Σ silicon_bin block transistors / 1000
* ``die.die_size_mm2`` -- Σ silicon_bin block area (= transistors / density)
* ``die.{foundry, process_nm, process_name}`` -- copied from ProcessNode
* ``performance.{int8_tops, bf16_tflops, fp32_tflops, int4_tops}`` --
  Σ(tile.num_tiles × ops_per_tile_per_clock) × default-profile clock
* ``power.{tdp_watts, max_power_watts, min_power_watts,
  default_thermal_profile, thermal_profiles}`` -- from the input
  thermal profiles (architect-declared)
* ``power.idle_power_watts`` -- chip-wide leakage from ProcessNode
  ``leakage_w_per_mm2`` × per-block area

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
    KPUDieSpec,
    KPUEntry,
    KPUPowerSpec,
    KPUTheoreticalPerformance,
    load_process_nodes,
)
from embodied_schemas.process_node import ProcessNodeEntry

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


def generate_kpu_sku(
    spec: KPUSKUInputSpec,
    *,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
) -> KPUEntry:
    """Produce a fully-populated KPUEntry from an input spec.

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
    # We pass a *temporary* KPUEntry-shaped object to silicon_math
    # because resolve_block_area() walks kpu_architecture for
    # PER_PE / PER_KIB / PER_ROUTER / PER_CONTROLLER expansions. We
    # build a placeholder die then fill in the real numbers; the spec's
    # KPUEntry construction at the end re-uses these.
    placeholder_die = KPUDieSpec(
        architecture="KPU Tile",
        foundry=node.foundry,
        process_nm=node.node_nm,
        process_name=node.node_name,
        transistors_billion=1.0,    # placeholder; overwritten below
        die_size_mm2=1.0,           # placeholder
        is_chiplet=False,
        num_dies=1,
    )
    placeholder_perf = KPUTheoreticalPerformance(
        int8_tops=0.0, bf16_tflops=0.0, fp32_tflops=0.0,
    )
    placeholder_power = KPUPowerSpec(
        tdp_watts=default_profile.tdp_watts,
        max_power_watts=default_profile.tdp_watts,
        min_power_watts=default_profile.tdp_watts,
        default_thermal_profile=spec.default_thermal_profile,
        thermal_profiles=spec.thermal_profiles,
    )
    placeholder_sku = KPUEntry(
        id=spec.id,
        name=spec.name,
        vendor=spec.vendor,
        process_node_id=spec.process_node_id,
        die=placeholder_die,
        kpu_architecture=spec.kpu_architecture,
        silicon_bin=spec.silicon_bin,
        clocks=spec.clocks,
        performance=placeholder_perf,
        power=placeholder_power,
        market=spec.market,
        notes=spec.notes,
        datasheet_url=spec.datasheet_url,
        last_updated=spec.last_updated,
    )

    total_area = 0.0
    total_mtx = 0.0
    unresolved: list[str] = []
    for block in spec.silicon_bin.blocks:
        try:
            ba = resolve_block_area(block, placeholder_sku, node)
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

    die = KPUDieSpec(
        architecture="KPU Tile",
        foundry=node.foundry,
        process_nm=node.node_nm,
        process_name=node.node_name,
        transistors_billion=round(total_mtx / 1000.0, 3),
        die_size_mm2=round(total_area, 1),
        is_chiplet=False,
        num_dies=1,
    )

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
    # tdp_watts comes from the default profile (architect-declared).
    # max/min from the highest/lowest profile TDP.
    max_w = max(p.tdp_watts for p in spec.thermal_profiles)
    min_w = min(p.tdp_watts for p in spec.thermal_profiles)
    leakage_w = total_chip_leakage_w(placeholder_sku, node)
    idle_w = round(leakage_w, 2) if leakage_w > 0 else None

    power = KPUPowerSpec(
        tdp_watts=default_profile.tdp_watts,
        max_power_watts=max_w,
        min_power_watts=min_w,
        idle_power_watts=idle_w,
        default_thermal_profile=spec.default_thermal_profile,
        thermal_profiles=spec.thermal_profiles,
    )

    # ---- Construct the final KPUEntry ----
    return KPUEntry(
        id=spec.id,
        name=spec.name,
        vendor=spec.vendor,
        process_node_id=spec.process_node_id,
        die=die,
        kpu_architecture=spec.kpu_architecture,
        silicon_bin=spec.silicon_bin,
        clocks=spec.clocks,
        performance=performance,
        power=power,
        market=spec.market,
        notes=spec.notes,
        datasheet_url=spec.datasheet_url,
        last_updated=spec.last_updated,
    )


def input_spec_from_kpu_entry(entry: KPUEntry) -> KPUSKUInputSpec:
    """Extract a KPUSKUInputSpec from an existing KPUEntry.

    Used by tests for round-trip verification (load existing YAML ->
    extract spec -> regenerate -> diff against original) and by the CLI
    when an architect wants to start from an existing SKU as a template.
    """
    return KPUSKUInputSpec(
        id=entry.id,
        name=entry.name,
        vendor=entry.vendor,
        process_node_id=entry.process_node_id,
        kpu_architecture=entry.kpu_architecture,
        silicon_bin=entry.silicon_bin,
        clocks=entry.clocks,
        thermal_profiles=entry.power.thermal_profiles,
        default_thermal_profile=entry.power.default_thermal_profile,
        market=entry.market,
        notes=entry.notes,
        datasheet_url=entry.datasheet_url,
        last_updated=entry.last_updated,
    )
