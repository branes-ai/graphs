"""ELECTRICAL category validators."""

from __future__ import annotations

from typing import List

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
        if ctx.sku.dies[0].clocks.boost_clock_mhz < clock_top:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.WARNING,
                    message=(
                        f"clocks.boost_clock_mhz = "
                        f"{ctx.sku.dies[0].clocks.boost_clock_mhz:.0f} MHz is below "
                        f"the highest thermal-profile clock "
                        f"{clock_top:.0f} MHz. Boost clock should be the "
                        f"chip's maximum advertised frequency."
                    ),
                )
            )

        # Base clock should be at least as fast as the lowest profile clock.
        clock_bot = min(p.clock_mhz for p in profiles)
        if ctx.sku.dies[0].clocks.base_clock_mhz > clock_bot:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.INFO,
                    message=(
                        f"clocks.base_clock_mhz = "
                        f"{ctx.sku.dies[0].clocks.base_clock_mhz:.0f} MHz exceeds "
                        f"the lowest thermal-profile clock "
                        f"{clock_bot:.0f} MHz. Base clock is normally the "
                        f"sustained guaranteed minimum."
                    ),
                )
            )

        return findings
