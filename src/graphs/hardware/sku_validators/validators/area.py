"""AREA category validators -- silicon-area / density / library checks."""

from __future__ import annotations

from typing import List

from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity
from ..silicon_math import (
    SiliconMathError,
    resolve_block_area,
)


# Area-self-consistency thresholds:
#   |computed - claimed| / claimed
#     <  5%  -> pass silently
#     5-25%  -> WARNING (numbers are off; either silicon_bin or roll-up
#               needs adjustment)
#     >=25%  -> ERROR (one of them is structurally wrong)
_AREA_WARN_REL = 0.05
_AREA_ERROR_REL = 0.25

# Composite-density envelope: composite must lie within
#   [min_library_density * (1 - SLOP), max_library_density * (1 + SLOP)]
# The SLOP accounts for layout / floorplan whitespace + analog/IO
# density that's hard to roll into the library catalog precisely.
_DENSITY_ENVELOPE_SLOP = 0.10


@default_registry.register_class
class BlockLibraryValidity:
    """Every silicon_bin block's circuit_class must be a library the
    process node actually offers.

    Catches: tagging a block SRAM_HP on a node without a dual-port SRAM
    library; tagging a block ULL_LOGIC on a FinFET node that doesn't
    offer it; copy-paste errors in circuit_class.
    """

    name = "block_library_validity"
    category = ValidatorCategory.AREA

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        node = ctx.process_node
        for block in ctx.sku.dies[0].silicon_bin.blocks:
            if node.supports(block.circuit_class):
                continue
            available = ", ".join(sorted(c.value for c in node.densities))
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.ERROR,
                    block=block.name,
                    message=(
                        f"block {block.name!r} declares "
                        f"circuit_class={block.circuit_class.value!r} "
                        f"but node {node.id!r} does not offer that "
                        f"library. Available libraries on this node: "
                        f"{available}."
                    ),
                    citation=f"process_node:{node.id} densities catalog",
                )
            )
        return findings


@default_registry.register_class
class AreaSelfConsistency:
    """Σ(block area) computed from silicon_bin must approximate
    ``die.die_size_mm2``. Σ(block transistors) must approximate
    ``die.transistors_billion``.

    Block area = transistors_block / density(circuit_class). The validator
    rolls the silicon_bin up and compares to the SKU's claimed totals.
    Catches missing silicon_bin entries (chip claims to be larger than
    its decomposition explains) and over-aggressive entries (decomposition
    sums above the claimed die).
    """

    name = "area_self_consistency"
    category = ValidatorCategory.AREA

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        sku = ctx.sku
        die = sku.dies[0]
        node = ctx.process_node

        total_area = 0.0
        total_mtx = 0.0
        unresolved: List[str] = []
        for block in die.silicon_bin.blocks:
            try:
                ba = resolve_block_area(block, sku, node)
            except SiliconMathError as exc:
                unresolved.append(f"{block.name}: {exc}")
                continue
            total_area += ba.area_mm2
            total_mtx += ba.transistors_mtx

        if not total_area:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.ERROR,
                    message=(
                        "silicon_bin resolves to 0 area -- no blocks could "
                        "be resolved against the process node. "
                        "Unresolved blocks: " + "; ".join(unresolved)
                        if unresolved
                        else "silicon_bin resolves to 0 area."
                    ),
                )
            )
            return findings

        # 1. Area roll-up vs claimed die_size_mm2.
        claimed_area = die.die_size_mm2
        rel_area = abs(total_area - claimed_area) / claimed_area
        if rel_area >= _AREA_ERROR_REL:
            sev = Severity.ERROR
        elif rel_area >= _AREA_WARN_REL:
            sev = Severity.WARNING
        else:
            sev = None
        if sev:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=sev,
                    message=(
                        f"silicon_bin sums to {total_area:.1f} mm^2 but "
                        f"die.die_size_mm2 = {claimed_area:.1f} mm^2 "
                        f"({rel_area:+.1%} relative error). The "
                        f"decomposition is missing blocks (computed < "
                        f"claimed) or has over-counted ones (computed > "
                        f"claimed). Tolerance: WARNING above "
                        f"{_AREA_WARN_REL:.0%}, ERROR above "
                        f"{_AREA_ERROR_REL:.0%}."
                    ),
                    citation=f"process_node:{node.id}",
                )
            )

        # 2. Transistor roll-up vs claimed transistors_billion.
        claimed_mtx = die.transistors_billion * 1000.0  # B -> M
        rel_mtx = abs(total_mtx - claimed_mtx) / claimed_mtx
        if rel_mtx >= _AREA_ERROR_REL:
            sev = Severity.ERROR
        elif rel_mtx >= _AREA_WARN_REL:
            sev = Severity.WARNING
        else:
            sev = None
        if sev:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=sev,
                    message=(
                        f"silicon_bin sums to {total_mtx/1000:.2f} B "
                        f"transistors but die.transistors_billion = "
                        f"{die.transistors_billion:.2f} B "
                        f"({rel_mtx:+.1%} relative error). Tolerance: "
                        f"WARNING above {_AREA_WARN_REL:.0%}, ERROR "
                        f"above {_AREA_ERROR_REL:.0%}."
                    ),
                )
            )

        return findings


@default_registry.register_class
class CompositeDensityEnvelope:
    """Chip composite transistor density (transistors_billion * 1000 /
    die_size_mm2) must lie within the process node's library-density
    envelope, i.e., between the minimum and maximum library density
    offered by the node.

    A composite density above the highest-density library (typically
    SRAM-HD) is impossible -- it would require packing more transistors
    per mm^2 than any library on the node achieves. A composite below
    the lowest is suspicious -- the chip is mostly empty.

    A SLOP factor accounts for floorplan whitespace, IO ring overhead,
    and the analog/mixed-signal blocks the library catalog approximates
    coarsely.
    """

    name = "composite_density_envelope"
    category = ValidatorCategory.AREA

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        die = ctx.sku.dies[0]
        node = ctx.process_node
        if die.die_size_mm2 <= 0:
            return []
        composite = (
            die.transistors_billion * 1000.0 / die.die_size_mm2
        )
        lo, hi = node.density_envelope()
        if lo <= 0 or hi <= 0:
            return []
        upper = hi * (1 + _DENSITY_ENVELOPE_SLOP)
        lower = lo * (1 - _DENSITY_ENVELOPE_SLOP)
        if composite > upper:
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.ERROR,
                    message=(
                        f"composite density "
                        f"{composite:.1f} Mtx/mm^2 exceeds the highest "
                        f"library density on node {node.id!r} "
                        f"({hi:.1f} Mtx/mm^2 + {_DENSITY_ENVELOPE_SLOP:.0%} "
                        f"slop = {upper:.1f}). The chip cannot be denser "
                        f"than its densest library; either "
                        f"transistors_billion or die_size_mm2 is wrong."
                    ),
                    citation=f"process_node:{node.id} density envelope",
                )
            ]
        if composite < lower:
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.WARNING,
                    message=(
                        f"composite density "
                        f"{composite:.1f} Mtx/mm^2 is below the lowest "
                        f"library density on node {node.id!r} "
                        f"({lo:.1f} Mtx/mm^2 - {_DENSITY_ENVELOPE_SLOP:.0%} "
                        f"slop = {lower:.1f}). Either the chip has a lot "
                        f"of unused floorplan area, or transistors / die "
                        f"size is wrong."
                    ),
                    citation=f"process_node:{node.id} density envelope",
                )
            ]
        return []
