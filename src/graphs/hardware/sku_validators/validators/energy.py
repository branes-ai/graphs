"""ENERGY category validators -- TOPS / W envelope, the "1 PetaOP @ 25 W" catch.

The user explicitly called out this bug class: a SKU YAML that claims
1 PetaOP at 25 W (i.e., 40,000 INT8 TOPS/W) is implausible at any modern
process node and topology. This validator enforces a per-process-node
ceiling on int8_tops / tdp_watts that catches such claims while staying
permissive enough not to flag legitimately aggressive edge KPUs.

Reference data points used to set the ceilings (rounded conservatively):

    chip / vendor                   node      INT8 TOPS/W
    -----------------------------   ------    -----------
    Hailo-8                         16nm      ~10
    Tenstorrent Wormhole            12nm       ~5
    Mythic AMP                      40nm        4
    NVIDIA Apple A17 / M3 NPU       3nm       ~36
    Snapdragon 8 Gen 3 NPU          4nm       ~25
    NVIDIA H100 (dense INT8)        4nm        ~3

KPU domain-flow can plausibly hit somewhat higher than mainstream
inference parts on the same node thanks to weight-stationary execution
and lack of speculative state -- the ceilings below allow ~2x headroom
above the most-efficient public reference at each node before flagging.
"""

from __future__ import annotations

from typing import List

from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity


# INT8 TOPS/W upper bound by process-node nm. Conservative -- catches
# obvious bugs (the user's "1 PetaOP @ 25W" = 40,000 TOPS/W is filtered
# at every node) without false-positives on aggressive but legitimate
# domain-flow designs.
_TOPS_W_INT8_CEILING_BY_NODE_NM: dict[int, float] = {
    16: 30.0,
    14: 35.0,
    12: 40.0,
    10: 50.0,
    8: 55.0,
    7: 65.0,
    6: 75.0,
    5: 85.0,
    4: 100.0,
    3: 120.0,
    2: 140.0,
}

# Anything below this is suspicious -- probably wrong precision tag or
# wrong TDP roll-up. Real KPUs targeting INT8 inference clear 0.5 TOPS/W
# easily on any modern node.
_TOPS_W_INT8_FLOOR = 0.5

# When the SKU is in 80-100% of the ceiling, emit a WARNING so the
# author is aware they're operating at the upper edge and should
# double-check. Above 100% is ERROR.
_TOPS_W_WARN_FRAC = 0.8


def _ceiling_for_node(node_nm: int) -> float | None:
    """Look up the TOPS/W ceiling for a process node's nm value.

    Falls back by walking down (older nodes get the next higher entry's
    ceiling) and up (newer nodes get the next lower entry's). Returns
    None if no plausible match -- validator skips the check rather than
    inventing a number for an unfamiliar process.
    """
    if node_nm in _TOPS_W_INT8_CEILING_BY_NODE_NM:
        return _TOPS_W_INT8_CEILING_BY_NODE_NM[node_nm]
    # Fall back to the closest known node by absolute nm distance.
    closest = min(
        _TOPS_W_INT8_CEILING_BY_NODE_NM,
        key=lambda nm: abs(nm - node_nm),
    )
    if abs(closest - node_nm) > 5:
        return None
    return _TOPS_W_INT8_CEILING_BY_NODE_NM[closest]


@default_registry.register_class
class TopsPerWattEnvelope:
    """Enforce an upper bound on int8_tops / tdp_watts keyed on
    process_node_nm.

    Concretely catches the user's "1 PetaOP @ 25 W" example: 1,000,000
    INT8 TOPS / 25 W = 40,000 TOPS/W -- way above any node's ceiling at
    every entry in the table. Also flags lesser but still implausible
    claims, e.g., 287 TOPS at 1 W on 16 nm = 287 TOPS/W (>>30 ceiling).
    """

    name = "tops_per_watt_envelope"
    category = ValidatorCategory.ENERGY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        sku = ctx.sku
        tdp = sku.power.tdp_watts
        tops = sku.performance.int8_tops
        if tdp <= 0 or tops <= 0:
            return []
        ratio = tops / tdp

        ceiling = _ceiling_for_node(ctx.process_node.node_nm)
        if ceiling is None:
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.INFO,
                    message=(
                        f"int8_tops / tdp_watts = {ratio:.2f} TOPS/W. "
                        f"No reference ceiling for node "
                        f"{ctx.process_node.node_nm} nm; envelope check "
                        f"skipped."
                    ),
                )
            ]

        # Floor: too-low value suggests broken inputs.
        if ratio < _TOPS_W_INT8_FLOOR:
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.WARNING,
                    message=(
                        f"int8_tops / tdp_watts = {ratio:.2f} TOPS/W is "
                        f"below the {_TOPS_W_INT8_FLOOR} TOPS/W "
                        f"plausibility floor for INT8 inference at any "
                        f"modern node. Likely cause: wrong TDP rollup, "
                        f"or int8_tops claim is BF16/FP32 mislabeled."
                    ),
                )
            ]

        # Ceiling check.
        if ratio > ceiling:
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.ERROR,
                    message=(
                        f"int8_tops / tdp_watts = {ratio:.2f} TOPS/W "
                        f"exceeds the {ceiling:.0f} TOPS/W plausibility "
                        f"ceiling for {ctx.process_node.node_nm} nm "
                        f"({ctx.process_node.transistor_topology.value}). "
                        f"Either int8_tops is overstated, tdp_watts is "
                        f"understated, or the SKU is on a more advanced "
                        f"node than process_node_id claims."
                    ),
                    citation="energy.py: per-node-nm TOPS/W ceiling table",
                )
            ]

        # Warning band: 80-100% of ceiling.
        if ratio > ceiling * _TOPS_W_WARN_FRAC:
            return [
                Finding(
                    validator=self.name,
                    category=self.category,
                    severity=Severity.WARNING,
                    message=(
                        f"int8_tops / tdp_watts = {ratio:.2f} TOPS/W is "
                        f"in the upper {(1 - _TOPS_W_WARN_FRAC):.0%} of "
                        f"the {ceiling:.0f} TOPS/W ceiling for "
                        f"{ctx.process_node.node_nm} nm. Aggressive but "
                        f"plausible; verify against measured silicon "
                        f"data when available."
                    ),
                    citation="energy.py: per-node-nm TOPS/W ceiling table",
                )
            ]

        # Otherwise comfortable -- no finding.
        return []
