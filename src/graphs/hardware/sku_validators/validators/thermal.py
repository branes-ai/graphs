"""THERMAL category validators -- power-density / hotspot / DVFS feasibility.

The user explicitly described this bug class:

  > "If we have a very high density compute block that is slated to run
  > at very high clock frequencies, that design would have a thermal
  > hotspot that will require aggressive DVFS to stay below the
  > electro-migration reliability metrics."

The thermal_hotspot validator catches it three ways:

1. **TDP vs cooling envelope**: profile.tdp_watts must fit within
   cooling.max_total_w.
2. **Leakage vs TDP**: total chip leakage must be well below TDP --
   otherwise the chip can't even idle within its thermal envelope.
3. **Per-block peak power density vs cooling ceiling**: for every block
   (PE blocks especially), peak W/mm^2 vs cooling.max_power_density.
   Above 5x ceiling -> ERROR (throttle factor so aggressive the SKU
   claim is misleading; check coefficients or cooling choice).
   1-5x  -> WARNING with explicit "DVFS throttle to N%" required.
   0.8-1x -> INFO (operating near thermal limits).
   Thresholds are the ``_DENSITY_*_MULT`` constants below; see their
   docstring for the rationale.
"""

from __future__ import annotations

from typing import List

from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity
from ..silicon_math import (
    estimate_block_peak_dynamic_w,
    estimate_block_leakage_w,
    resolve_block_area,
    SiliconMathError,
    total_chip_leakage_w,
)


# Power-density tier thresholds (multiples of cooling.max_power_density)
# applied to PEAK block power density. Real chips throttle to stay within
# cooling, so "peak > ceiling" is normal and informative -- the question
# is HOW MUCH throttling is required.
#   <80%  ceiling: no finding
#   80-100%:       INFO (operating near limits)
#   1-5x:          WARNING (DVFS throttle required; advertised peak is
#                  burst, not sustained)
#   >5x:           ERROR (throttle factor so aggressive the SKU's claim
#                  is misleading; check coefficients or cooling choice)
_DENSITY_INFO_FRAC = 0.8
_DENSITY_WARN_MULT = 1.0
_DENSITY_ERROR_MULT = 5.0


@default_registry.register_class
class ThermalHotspot:
    """Per-block peak power density vs cooling-solution ceiling.

    Iterates every (thermal_profile, cooling_solution) pair. For each:
    runs three checks, emits Findings tied to the profile so a SKU author
    sees which operating mode the issue applies to.

    The "DVFS throttle to N%" message in WARNING findings is the actionable
    output the user asked for: it tells the SKU author the design is buildable
    but the advertised peak isn't sustainable -- quantifies how much of the
    headline number is burst vs steady-state.
    """

    name = "thermal_hotspot"
    category = ValidatorCategory.THERMAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        sku = ctx.sku
        node = ctx.process_node

        chip_leakage = total_chip_leakage_w(sku, node)

        for profile in sku.power.thermal_profiles:
            cooling = ctx.cooling_solutions.get(profile.cooling_solution_id)
            if cooling is None:
                # cross_ref_consistency already reports the dangling id;
                # don't duplicate. Skip this profile here.
                continue

            # ---- Check 1: TDP must fit cooling envelope ----
            if profile.tdp_watts > cooling.max_total_w:
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.ERROR,
                        profile=profile.name,
                        message=(
                            f"profile {profile.name!r} TDP "
                            f"{profile.tdp_watts:.0f} W exceeds cooling "
                            f"solution {cooling.id!r} envelope "
                            f"{cooling.max_total_w:.0f} W. The cooling "
                            f"cannot remove the SKU's claimed TDP -- "
                            f"either the cooling is undersized or the "
                            f"profile's TDP claim is wrong."
                        ),
                        citation=f"cooling:{cooling.id} max_total_w",
                    )
                )

            # ---- Check 2: leakage must be well below TDP ----
            if chip_leakage > 0 and chip_leakage > 0.6 * profile.tdp_watts:
                sev = (
                    Severity.ERROR
                    if chip_leakage > profile.tdp_watts
                    else Severity.WARNING
                )
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=sev,
                        profile=profile.name,
                        message=(
                            f"chip-wide leakage {chip_leakage:.2f} W is "
                            f"{chip_leakage / profile.tdp_watts:.0%} of "
                            f"profile {profile.name!r} TDP "
                            f"{profile.tdp_watts:.0f} W. Almost no "
                            f"thermal budget remains for dynamic switching "
                            f"-- check process_node leakage_w_per_mm2 "
                            f"figures, library choices, or whether the "
                            f"TDP is set too low."
                        ),
                        citation=f"process_node:{node.id} leakage_w_per_mm2",
                    )
                )

            # ---- Check 3: per-block power density at peak vs cooling ceiling ----
            ceiling = cooling.max_power_density_w_per_mm2
            for block in sku.silicon_bin.blocks:
                try:
                    ba = resolve_block_area(block, sku, node)
                except SiliconMathError:
                    continue
                if ba.area_mm2 <= 0:
                    continue

                dyn = estimate_block_peak_dynamic_w(
                    block, sku, node, clock_mhz=profile.clock_mhz
                )
                leak = estimate_block_leakage_w(block, sku, node)
                peak_w = dyn + leak
                if peak_w <= 0:
                    continue
                density = peak_w / ba.area_mm2
                ratio = density / ceiling

                if ratio >= _DENSITY_ERROR_MULT:
                    findings.append(
                        Finding(
                            validator=self.name,
                            category=self.category,
                            severity=Severity.ERROR,
                            block=block.name,
                            profile=profile.name,
                            message=(
                                f"block peak power density "
                                f"{density:.2f} W/mm^2 is {ratio:.1f}x the "
                                f"{cooling.id!r} ceiling "
                                f"{ceiling:.2f} W/mm^2. Even with maximum "
                                f"DVFS throttling, the block cannot sustain "
                                f"useful operation under this cooling -- "
                                f"the design is fundamentally hot for the "
                                f"chosen cooling solution."
                            ),
                            citation=f"cooling:{cooling.id} max_power_density",
                        )
                    )
                elif ratio >= _DENSITY_WARN_MULT:
                    throttle = ceiling / density
                    findings.append(
                        Finding(
                            validator=self.name,
                            category=self.category,
                            severity=Severity.WARNING,
                            block=block.name,
                            profile=profile.name,
                            message=(
                                f"block peak power density "
                                f"{density:.2f} W/mm^2 exceeds the "
                                f"{cooling.id!r} ceiling "
                                f"{ceiling:.2f} W/mm^2 by "
                                f"{(ratio - 1):.0%}. Sustained operation "
                                f"requires DVFS throttling to ~{throttle:.0%} "
                                f"of peak clock; advertised peak throughput "
                                f"for this block is burst-only."
                            ),
                            citation=f"cooling:{cooling.id} max_power_density",
                        )
                    )
                elif ratio >= _DENSITY_INFO_FRAC:
                    findings.append(
                        Finding(
                            validator=self.name,
                            category=self.category,
                            severity=Severity.INFO,
                            block=block.name,
                            profile=profile.name,
                            message=(
                                f"block peak power density "
                                f"{density:.2f} W/mm^2 is {ratio:.0%} of the "
                                f"{cooling.id!r} ceiling "
                                f"{ceiling:.2f} W/mm^2. Operating near "
                                f"thermal limits; modest DVFS headroom."
                            ),
                            citation=f"cooling:{cooling.id} max_power_density",
                        )
                    )

        return findings
