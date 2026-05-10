"""INTERNAL category validators -- SKU self-consistency.

Catches arithmetic / cross-reference bugs that don't depend on
process-node or cooling-solution data. These run first in the registry
sort order so authors see them before slower / fancier checks.
"""

from __future__ import annotations

from typing import List

from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity


# Tolerance for the performance-vs-tile-fabric arithmetic. 1 % accounts
# for rounding in hand-authored YAMLs (e.g., int8_tops: 287.0 vs computed
# 286.72 from 204800 ops/clock x 1.4 GHz). Anything beyond this is an
# arithmetic / data error.
_PERF_REL_TOLERANCE = 0.01


@default_registry.register_class
class TileMixConsistency:
    """Σ(tile.num_tiles) == total_tiles, and the SKU's claimed
    performance numbers match what the tile fabric + default-profile
    clock can deliver.

    This is the first place obviously-wrong YAMLs get caught: a copy-paste
    error in tile_mix or in the rolled-up performance number lands here.
    """

    name = "tile_mix_consistency"
    category = ValidatorCategory.INTERNAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        sku = ctx.sku
        arch = sku.kpu_architecture

        # 1. Σ(num_tiles) == total_tiles
        sum_tiles = sum(t.num_tiles for t in arch.tiles)
        if sum_tiles != arch.total_tiles:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.ERROR,
                    message=(
                        f"sum(tile.num_tiles) = {sum_tiles} != "
                        f"total_tiles = {arch.total_tiles}. Tile mix and "
                        f"total don't agree -- one of them is wrong."
                    ),
                )
            )

        # 2. Look up the default thermal profile's clock.
        default_name = sku.power.default_thermal_profile
        default_profile = next(
            (p for p in sku.power.thermal_profiles if p.name == default_name),
            None,
        )
        if default_profile is None:
            findings.append(
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.ERROR,
                    message=(
                        f"power.default_thermal_profile = {default_name!r} "
                        f"is not present in power.thermal_profiles "
                        f"({[p.name for p in sku.power.thermal_profiles]})."
                    ),
                )
            )
            return findings

        clock_hz = default_profile.clock_mhz * 1e6

        # 3. Compare claimed peak performance to fabric * clock for each
        #    precision the SKU advertises.
        for precision_attr, precision_key, scale, unit in [
            ("int8_tops", "int8", 1e12, "TOPS"),
            ("bf16_tflops", "bf16", 1e12, "TFLOPS"),
            ("fp32_tflops", "fp32", 1e12, "TFLOPS"),
            ("int4_tops", "int4", 1e12, "TOPS"),
        ]:
            claimed = getattr(sku.performance, precision_attr)
            if claimed is None or claimed <= 0:
                # Optional precision (int4 may be None) or genuinely zero.
                continue
            fabric_ops_per_clock = sum(
                t.num_tiles * t.ops_per_tile_per_clock.get(precision_key, 0)
                for t in arch.tiles
            )
            if fabric_ops_per_clock <= 0:
                # Claimed >0 but no tile reports this precision -- the
                # claim has no fabric support.
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.ERROR,
                        message=(
                            f"performance.{precision_attr} = {claimed} "
                            f"{unit} but no tile class declares "
                            f"ops_per_tile_per_clock[{precision_key!r}]. "
                            f"The advertised throughput has no fabric "
                            f"support."
                        ),
                    )
                )
                continue
            computed = fabric_ops_per_clock * clock_hz / scale
            if computed <= 0:
                continue
            rel_err = abs(claimed - computed) / computed
            if rel_err > _PERF_REL_TOLERANCE:
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.ERROR,
                        profile=default_profile.name,
                        message=(
                            f"performance.{precision_attr} = {claimed:.2f} "
                            f"{unit} disagrees with fabric x clock = "
                            f"{computed:.2f} {unit} "
                            f"({fabric_ops_per_clock} ops/clock x "
                            f"{default_profile.clock_mhz:.0f} MHz). "
                            f"Relative error {rel_err:.1%} exceeds "
                            f"{_PERF_REL_TOLERANCE:.0%} tolerance. The "
                            f"rolled-up performance number is wrong, OR "
                            f"the tile ops_per_tile_per_clock is wrong, "
                            f"OR the default profile clock is wrong."
                        ),
                    )
                )

        return findings


@default_registry.register_class
class CrossRefConsistency:
    """Every cross-reference in the SKU resolves: thermal-profile
    cooling_solution_id is in the cooling-solution catalog; every
    silicon_bin block's circuit_class is offered by the process node.

    These are 'are the foreign keys valid' checks. ContextError already
    catches the unresolved process_node_id at context-build time, so
    that one isn't repeated here.
    """

    name = "cross_ref_consistency"
    category = ValidatorCategory.INTERNAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        sku = ctx.sku

        # 1. Every thermal profile points at a cooling solution that
        #    actually exists in the catalog.
        for tp in sku.power.thermal_profiles:
            if tp.cooling_solution_id not in ctx.cooling_solutions:
                available = ", ".join(sorted(ctx.cooling_solutions)) or "(none)"
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.ERROR,
                        profile=tp.name,
                        message=(
                            f"thermal profile {tp.name!r} references "
                            f"cooling_solution_id={tp.cooling_solution_id!r} "
                            f"which does not exist in the cooling-solution "
                            f"catalog. Available: {available}"
                        ),
                    )
                )

        # 2. Every silicon_bin block's circuit_class must be supported
        #    by the SKU's process node. (Validator block_library_validity
        #    in area.py reports the same thing from the AREA category;
        #    we surface it here too because it's a foreign-key issue
        #    that AREA-suppressed runs would miss.)
        node = ctx.process_node
        for block in sku.silicon_bin.blocks:
            if not node.supports(block.circuit_class):
                available = ", ".join(
                    sorted(c.value for c in node.densities)
                )
                findings.append(
                    Finding(
                        validator=self.name,
                        category=self.category,
                        severity=Severity.ERROR,
                        block=block.name,
                        message=(
                            f"silicon_bin block {block.name!r} declares "
                            f"circuit_class={block.circuit_class.value!r} "
                            f"but process node {node.id!r} does not offer "
                            f"that library. Available: {available}"
                        ),
                    )
                )

        return findings
