"""ELECTRICAL category validators."""

from __future__ import annotations

from typing import List

from ...kpu_access import kpu_die_of
from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity


@default_registry.register_class
class PowerProfileMonotonicity:
    """Thermal profiles, sorted by tdp_watts ascending, must have strictly
    monotonically increasing clock_mhz.

    A higher-TDP profile that runs slower than a lower-TDP profile is
    almost certainly a typo. Conversely, a lower-TDP profile with a
    higher clock is likely a swap.
    """

    name = "power_profile_monotonicity"
    category = ValidatorCategory.ELECTRICAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        profiles = list(ctx.sku.power.thermal_profiles)
        if len(profiles) < 2:
            return findings

        # Sort by TDP ascending; clocks must then be strictly ascending.
        sorted_profiles = sorted(profiles, key=lambda p: p.tdp_watts)
        for prev, curr in zip(sorted_profiles, sorted_profiles[1:]):
            if curr.clock_mhz <= prev.clock_mhz:
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.WARNING,
                        profile=curr.name,
                        message=(
                            f"profile {curr.name!r} has higher TDP "
                            f"({curr.tdp_watts:.0f} W vs "
                            f"{prev.tdp_watts:.0f} W) but does not run "
                            f"faster: clock {curr.clock_mhz:.0f} MHz <= "
                            f"{prev.name!r} clock {prev.clock_mhz:.0f} MHz. "
                            f"Higher-TDP profiles should always be at "
                            f"least as fast as lower-TDP ones."
                        ),
                    )
                )

        # Boost clock should be at least as fast as the highest profile clock.
        clock_top = max(p.clock_mhz for p in profiles)
        if kpu_die_of(ctx.sku).clocks.boost_clock_mhz < clock_top:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.WARNING,
                    message=(
                        f"clocks.boost_clock_mhz = "
                        f"{kpu_die_of(ctx.sku).clocks.boost_clock_mhz:.0f} MHz is below "
                        f"the highest thermal-profile clock "
                        f"{clock_top:.0f} MHz. Boost clock should be the "
                        f"chip's maximum advertised frequency."
                    ),
                )
            )

        # Base clock should be at least as fast as the lowest profile clock.
        clock_bot = min(p.clock_mhz for p in profiles)
        if kpu_die_of(ctx.sku).clocks.base_clock_mhz > clock_bot:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.INFO,
                    message=(
                        f"clocks.base_clock_mhz = "
                        f"{kpu_die_of(ctx.sku).clocks.base_clock_mhz:.0f} MHz exceeds "
                        f"the lowest thermal-profile clock "
                        f"{clock_bot:.0f} MHz. Base clock is normally the "
                        f"sustained guaranteed minimum."
                    ),
                )
            )

        return findings


#: How far a profile's declared TDP may sit from the power model's own
#: figure before it is a finding. The declared values are stated to 0.1 W,
#: so half a step is the tightest band that does not fire on rounding.
_TDP_DRIFT_W = 0.05

#: Beyond this the two numbers are not describing the same part.
_TDP_DRIFT_FRACTION = 0.10


@default_registry.register_class
class DeclaredTdpMatchesModel:
    """A profile's declared ``tdp_watts`` must be what the power model
    computes for it.

    Nothing checked this, and it drifted (graphs#268 F4). Every
    ``7nm_tsmc_hpc`` SKU's ``lp`` and ``default`` profiles declared more
    than the model computed -- the T512's ``lp`` claimed 10.3 W against a
    computed 8.9 W -- because their Vdd values were never re-tuned after
    leakage gained its Vdd scaling. Their ``boost`` profiles sat exactly at
    nominal Vdd, where that scaling is a no-op, so the one profile that
    could not reveal the problem was the one that looked healthy.

    The declared value is the target: Vdd is the knob tuned to hit it. So a
    mismatch means either the Vdd wants re-tuning or the envelope claim is
    stale, and the message says which direction the model went.
    """

    name = "declared_tdp_matches_model"
    category = ValidatorCategory.ELECTRICAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        # Imported here: the power model pulls in the generator, and the
        # validator package is imported from it.
        from ...kpu_power_model import compute_thermal_profile_tdp_w
        from ...kpu_sku_generator import (
            GeneratorError,
            input_spec_from_compute_product,
        )

        try:
            spec = input_spec_from_compute_product(ctx.sku)
        except GeneratorError as exc:
            # The only failure the generator documents: no KPU block, or
            # more than one. Anything else is a bug in this validator or
            # the model, and must reach the registry, which turns an
            # uncaught exception into an ERROR. Swallowing it here would
            # downgrade a real defect to a non-fatal INFO.
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.INFO,
                    message=(
                        f"cannot check declared TDP: this SKU does not round-trip "
                        f"through the generator ({exc})"
                    ),
                )
            ]

        findings: List[Finding] = []
        for profile in ctx.sku.power.thermal_profiles:
            computed = compute_thermal_profile_tdp_w(spec, profile, ctx.process_node)
            delta = computed - profile.tdp_watts
            if abs(delta) < _TDP_DRIFT_W:
                continue
            fraction = abs(delta) / profile.tdp_watts if profile.tdp_watts else 1.0
            severity = (
                Severity.ERROR if fraction >= _TDP_DRIFT_FRACTION else Severity.WARNING
            )
            direction = "below" if delta < 0 else "above"
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=severity,
                    profile=profile.name,
                    message=(
                        f"profile {profile.name!r} declares {profile.tdp_watts:.1f} W "
                        f"but the power model computes {computed:.1f} W at "
                        f"vdd={profile.vdd_v:.3f} V, {abs(delta):.2f} W "
                        f"({fraction * 100:.0f}%) {direction}. Either re-tune vdd_v "
                        f"to hit the declared envelope, or correct the envelope."
                    ),
                    citation=(
                        f"kpu_power_model.compute_thermal_profile_tdp_w at "
                        f"{ctx.process_node.node_name}; drift band "
                        f"WARN>={_TDP_DRIFT_W} W ERR>="
                        f"{_TDP_DRIFT_FRACTION * 100:.0f}%"
                    ),
                )
            )
        return findings
