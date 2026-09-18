"""Checks on a composed SoC (graphs#269 PR 2.4).

These run on an ``SoCInstance`` rather than a ``ComputeProduct`` -- Phase 2
does not emit one (decision P2-D1) -- but produce the same ``Finding`` type,
so ``validate_sku.py --soc`` renders them and sets its exit code exactly as it
does for a KPU SKU.

Each check separates *cannot tell* from *fine*. A composition with unanchored
silicon has no complete area, so a shoreline or reference comparison against
it is reported as not performed, never as passed.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from ..sku_validators.framework import Finding, Severity, ValidatorCategory
from .compose import SoCInstance

#: A composition this far off its reference is a WARNING (the parent plan's
#: acceptance tolerance for the Orin reconstruction).
REFERENCE_TOLERANCE = 0.15

#: PHYs and pads using more than this share of the perimeter leave little
#: room for anything else on the edge.
SHORELINE_WARN_FRACTION = 0.8


def _finding(name, category, severity, message, block=None) -> Finding:
    return Finding(
        validator=name, category=category, severity=severity, message=message, block=block,
    )


def area_completeness(soc: SoCInstance) -> List[Finding]:
    """Unanchored silicon makes every area total a lower bound."""
    name = "soc_area_completeness"
    if soc.complete:
        return []
    by_block: dict = {}
    for block, line in soc.gaps:
        by_block.setdefault(block, []).append(line.name)
    listed = "; ".join(f"{b}: {', '.join(lines)}" for b, lines in by_block.items())
    return [_finding(
        name, ValidatorCategory.AREA, Severity.WARNING,
        f"{len(soc.gaps)} silicon line(s) have no public area anchor, so the die area "
        f"({soc.die_area_mm2:.1f} mm^2) and transistor count ({soc.transistors_billion:.2f} B) "
        f"are lower bounds. Unanchored -- {listed}.",
    )]


def clock_retargeting(soc: SoCInstance) -> List[Finding]:
    """A clock that could not be carried to this node makes its peak
    provisional."""
    name = "soc_clock_retargeting"
    findings = []
    for block in soc.blocks:
        if block.clock_is_reference:
            continue
        ref = block.template.clock.reference_node if block.template.clock else "?"
        findings.append(_finding(
            name, ValidatorCategory.ELECTRICAL, Severity.WARNING,
            f"{block.name}: its {block.clock_ghz:g} GHz clock was characterized on {ref} and "
            f"no foundry-stated speed relation connects {ref} to {soc.node.id}, so its peak at "
            f"{soc.node.id} is provisional.",
            block=block.name,
        ))
    return findings


def phy_shoreline(soc: SoCInstance) -> List[Finding]:
    """PHYs and pads barely shrink with the node, so a small-node die can run
    out of edge before it runs out of area."""
    name = "soc_phy_shoreline"
    stating = [b for b in soc.blocks if b.template.shoreline is not None]
    if not stating:
        return [_finding(
            name, ValidatorCategory.GEOMETRY, Severity.INFO,
            "Shoreline not checked: no IP block in this design states the die edge it needs "
            "(PHY edge lengths are in vendor datasheets that were not retrievable).",
        )]
    required = soc.shoreline_required_mm
    perimeter = soc.die_perimeter_mm
    share = required / perimeter if perimeter else float("inf")
    qualifier = (
        " The die area is a lower bound, so the real perimeter may be larger."
        if not soc.complete else ""
    )
    if share > 1.0:
        severity, verdict = Severity.ERROR, "exceeds"
    elif share > SHORELINE_WARN_FRACTION:
        severity, verdict = Severity.WARNING, "uses most of"
    else:
        return []
    return [_finding(
        name, ValidatorCategory.GEOMETRY, severity,
        f"PHYs and pads need {required:.1f} mm of edge, which {verdict} the "
        f"{perimeter:.1f} mm perimeter ({share:.0%}): the die is pad-limited.{qualifier}",
    )]


def reference_comparison(soc: SoCInstance) -> List[Finding]:
    """Compare against a published die only when the comparison means
    something: an incomplete composition is a lower bound, not a
    reconstruction."""
    name = "soc_reference_comparison"
    ref = soc.design.reference
    if ref is None or ref.die_area_mm2 is None:
        return []
    if ref.process_node and ref.process_node != soc.node.id:
        return []
    if not soc.complete:
        return [_finding(
            name, ValidatorCategory.INTERNAL, Severity.INFO,
            f"Not compared with the reference {ref.die_area_mm2:g} mm^2: the composition has "
            f"{len(soc.gaps)} unanchored line(s), so its {soc.die_area_mm2:.1f} mm^2 is a lower "
            f"bound, not a reconstruction.",
        )]
    error = soc.die_area_mm2 / ref.die_area_mm2 - 1.0
    if abs(error) <= REFERENCE_TOLERANCE:
        return []
    return [_finding(
        name, ValidatorCategory.INTERNAL, Severity.WARNING,
        f"Composed die {soc.die_area_mm2:.1f} mm^2 is {error:+.0%} from the reference "
        f"{ref.die_area_mm2:g} mm^2, outside +/-{REFERENCE_TOLERANCE:.0%}.",
    )]


SOC_VALIDATORS: Tuple[Tuple[str, Callable[[SoCInstance], List[Finding]]], ...] = (
    ("soc_area_completeness", area_completeness),
    ("soc_clock_retargeting", clock_retargeting),
    ("soc_phy_shoreline", phy_shoreline),
    ("soc_reference_comparison", reference_comparison),
)


def validate_soc(soc: SoCInstance) -> List[Finding]:
    """Every SoC check, in a stable order."""
    findings: List[Finding] = []
    for _, check in SOC_VALIDATORS:
        findings.extend(check(soc))
    return findings
