"""RELIABILITY category validators -- electromigration and related.

EM (electromigration) is governed by Black's equation: a metal line's
mean-time-to-failure depends on current density and temperature. PDKs
ship per-(metal-layer, temperature) J_max ceilings. ProcessNodeEntry
holds public-estimate or PDK-derived ``em_j_max_by_temp_c`` values.

The validator is a coarse plausibility check at SKU-block granularity
-- it can't replace per-line current analysis from a real router /
parasitic-extraction flow, but it catches designs whose block-level
current densities are obviously incompatible with the metal stack at
the operating Tj.

Margin model (per block):

  block_peak_current   = (block_dynamic_w + block_leakage_w) / nominal_vdd
  metal_capacity_per_um_chip_width =
      num_active_layers * (1 / metal_pitch_um) * line_width_um
      * line_thickness_um * J_max_at_Tj
  block_capacity_a     = sqrt(block_area_mm2 * 1e6) * metal_capacity_per_um
  margin               = block_peak_current / block_capacity_a

  margin <  50%  -> INFO   (safe)
  margin 50-100% -> WARNING (approaching EM limit; check at next PDK)
  margin >100%   -> ERROR   (peak operation would force EM violation)

Tj is taken from the hottest cooling solution attached to any thermal
profile -- worst-case for reliability accounting.
"""

from __future__ import annotations

import math
from typing import List

from embodied_schemas.process_node import ProcessNodeEntry

from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity
from ..silicon_math import (
    resolve_block_area,
    SiliconMathError,
)


# Defaults if process_node doesn't carry the geometry. These represent
# the *full power-delivery stack*, not just M0/M1 -- real chips deliver
# block current via M0..Mtop with thicker upper metals. v1 placeholder
# (12 layers, 0.20 um effective thickness as weighted average across
# the stack) until PDK data lands and we model layers individually.
_DEFAULT_METAL_THICKNESS_UM: float = 0.20
_DEFAULT_NUM_ACTIVE_METAL_LAYERS: int = 12

# Margin thresholds (block_sustained_current / block_em_capacity_a).
# Uses SUSTAINED current at TDP (not peak): the chip throttles to TDP
# in steady state, so EM lifetime is governed by sustained ops, not
# burst peaks. Peak EM stress is bounded by the thermal validator's
# DVFS-throttle finding.
_EM_MARGIN_WARN = 0.50
_EM_MARGIN_ERROR = 1.00


def _metal_capacity_a_per_um_chip_width(
    node: ProcessNodeEntry, metal_width_um: float, j_max_a_cm2: float
) -> float:
    """Current capacity through M0/M1 stack, per um of chip width.

    Heuristic: assume ``_DEFAULT_NUM_ACTIVE_METAL_LAYERS`` layers carry
    the block's switching current, each line at ``metal_width_um`` wide
    and ``_DEFAULT_METAL_THICKNESS_UM`` thick, with metals on a pitch of
    ``2 * metal_width_um`` (50 % wire / 50 % space).

    Lines per um chip width = layers / (2 * metal_width_um).
    Per-line cross-section = metal_width_um * metal_thickness_um (um^2).
    Per-line max current  = J_max * cross-section.

    Capacity per um chip width = lines_per_um * per_line_max_current.
    Convert J from A/cm^2 to A/um^2: 1 A/cm^2 = 1e-8 A/um^2.
    """
    if metal_width_um <= 0 or j_max_a_cm2 <= 0:
        return 0.0
    lines_per_um = _DEFAULT_NUM_ACTIVE_METAL_LAYERS / (2.0 * metal_width_um)
    cross_section_um2 = metal_width_um * _DEFAULT_METAL_THICKNESS_UM
    j_max_a_per_um2 = j_max_a_cm2 * 1e-8
    per_line_max_a = j_max_a_per_um2 * cross_section_um2
    return lines_per_um * per_line_max_a


def _hottest_tj_for_sku(ctx: ValidatorContext) -> float | None:
    """Highest Tj_max among any cooling solution any thermal_profile uses.

    Returns None if no profile / cooling resolves -- the EM check
    cannot pick a J_max in that case and skips silently.
    """
    tjs: list[float] = []
    for profile in ctx.sku.power.thermal_profiles:
        cs = ctx.cooling_solutions.get(profile.cooling_solution_id)
        if cs is not None:
            tjs.append(cs.junction_c_max)
    return max(tjs) if tjs else None


def _j_max_at_tj(node: ProcessNodeEntry, tj_c: float) -> float | None:
    """Pick the J_max from ``em_j_max_by_temp_c`` whose temperature is
    closest to (and >= preferred to) Tj."""
    if not node.em_j_max_by_temp_c:
        return None
    # Prefer temperatures >= Tj; if none, fall back to highest available.
    geq = sorted(t for t in node.em_j_max_by_temp_c if t >= tj_c)
    if geq:
        return node.em_j_max_by_temp_c[geq[0]]
    # All entries cooler than Tj -- use the hottest (most pessimistic J_max).
    hottest = max(node.em_j_max_by_temp_c)
    return node.em_j_max_by_temp_c[hottest]


@default_registry.register_class
class Electromigration:
    """Per-block peak current density vs metal-stack EM capacity.

    Uses the SKU's hottest Tj as the operating point for J_max lookup
    (worst-case reliability). Skips silently when the process node has
    no EM data or routing-metal-width entries -- the validator never
    invents PDK numbers.
    """

    name = "electromigration"
    category = ValidatorCategory.RELIABILITY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        sku = ctx.sku
        node = ctx.process_node

        tj_c = _hottest_tj_for_sku(ctx)
        if tj_c is None:
            # No resolvable cooling -> can't pick Tj. cross_ref_consistency
            # reports the underlying issue.
            return findings

        j_max = _j_max_at_tj(node, tj_c)
        if j_max is None or j_max <= 0:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.INFO,
                    message=(
                        f"process node {node.id!r} has no "
                        f"em_j_max_by_temp_c data; skipping per-block "
                        f"electromigration check. Populate "
                        f"em_j_max_by_temp_c on the ProcessNodeEntry "
                        f"to enable this validator."
                    ),
                    citation=f"process_node:{node.id}",
                )
            )
            return findings

        if node.nominal_vdd_v <= 0:
            return findings

        # Use the highest-clock thermal profile (= highest-TDP) for the
        # sustained-current accounting. The chip throttles to stay at
        # TDP in steady state.
        peak_profile = max(
            sku.power.thermal_profiles, key=lambda p: p.tdp_watts, default=None
        )
        if peak_profile is None:
            return findings

        # Compute total area of resolvable blocks for the proportional
        # current-distribution model.
        total_area_mm2 = 0.0
        for block in sku.dies[0].silicon_bin.blocks:
            try:
                total_area_mm2 += resolve_block_area(block, sku, node).area_mm2
            except SiliconMathError:
                continue
        if total_area_mm2 <= 0:
            return findings

        for block in sku.dies[0].silicon_bin.blocks:
            try:
                ba = resolve_block_area(block, sku, node)
            except SiliconMathError:
                continue
            if ba.area_mm2 <= 0:
                continue

            metal_width = node.routing_metal_width_um.get(block.circuit_class)
            if not metal_width or metal_width <= 0:
                # Node hasn't characterized routing for this library;
                # skip silently rather than guess.
                continue

            cap_per_um = _metal_capacity_a_per_um_chip_width(
                node, metal_width, j_max
            )
            if cap_per_um <= 0:
                continue

            # Block "width" approximated as sqrt(area) -- assumes square
            # blocks. Validators are blind to floorplan geometry until
            # the Stage-8 floorplanner lands, so this is the cleanest
            # geometry-free heuristic.
            block_width_um = math.sqrt(ba.area_mm2 * 1e6)
            block_capacity_a = cap_per_um * block_width_um
            if block_capacity_a <= 0:
                continue

            # Sustained current: distribute the SKU's TDP across blocks
            # proportional to area. This is a v1 uniform-density
            # approximation; future refinements will use per-block
            # activity weights derived from the workload mix.
            block_area_fraction = ba.area_mm2 / total_area_mm2
            block_sustained_w = peak_profile.tdp_watts * block_area_fraction
            block_current_a = block_sustained_w / node.nominal_vdd_v
            margin = block_current_a / block_capacity_a

            if margin >= _EM_MARGIN_ERROR:
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.ERROR,
                        block=block.name,
                        message=(
                            f"sustained current {block_current_a:.2f} A "
                            f"(at TDP {peak_profile.tdp_watts:.0f} W) "
                            f"would require {margin:.1f}x the EM capacity "
                            f"{block_capacity_a:.2f} A at Tj={tj_c:.0f} C "
                            f"(J_max={j_max:.1e} A/cm^2, "
                            f"metal_width={metal_width*1000:.0f} nm, "
                            f"{_DEFAULT_NUM_ACTIVE_METAL_LAYERS}-layer "
                            f"power stack). Sustained operation at TDP "
                            f"would violate EM lifetime targets."
                        ),
                        citation=(
                            f"process_node:{node.id} em_j_max_by_temp_c, "
                            f"routing_metal_width_um"
                        ),
                    )
                )
            elif margin >= _EM_MARGIN_WARN:
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.WARNING,
                        block=block.name,
                        message=(
                            f"sustained current {block_current_a:.2f} A "
                            f"is {margin:.0%} of EM capacity "
                            f"{block_capacity_a:.2f} A at Tj={tj_c:.0f} C "
                            f"(J_max={j_max:.1e} A/cm^2). Approaching EM "
                            f"limit -- verify with PDK current-density "
                            f"analysis when available."
                        ),
                        citation=(
                            f"process_node:{node.id} em_j_max_by_temp_c"
                        ),
                    )
                )
            # margin <50%: silent (no INFO clutter for the common case)

        return findings
