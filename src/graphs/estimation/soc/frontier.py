"""Where a configuration sits on the energy-performance plane, per mission
(graphs#269 Phase 7.3).

The state-space matrix answers "is this configuration ruled out?" one row
at a time. It does not show the *shape* of the trade: how much performance
a joule buys, which configurations are on the efficient edge of what the
catalog can build, and which are inside it. That is what this module
computes, and what the report draws.

**Two axes, both bounds, both in the same direction.**

``energy_per_op_pj`` is the mission's own arithmetic priced at the process
node's energy per op, weighted over the formats each stage runs in and the
libraries each engine issues them from:

    energy per op = sum over stages and classes of (ops x pJ per op)
                    / sum over stages and classes of ops

No efficiency enters, so nothing can come in under it: it is a **floor**.
Where a stage's ops cannot be priced -- a node that states no figure for a
format -- the stage is left out and the floor is lower still, which the
point records.

``real_time_factor`` is how much of the mission's required rate the
configuration could carry: ``1 / utilization``, where a stage's time uses
the best efficiency anything states for it -- a measurement where one
exists, otherwise the domain-flow ceiling, which is an upper bound. So the
factor is a **ceiling on performance**: 1.0 is real time, and a point below
1.0 is *proven* short, because even the optimistic figure does not reach
it. Stages nothing prices are left out, which only flatters the point
further.

Both axes therefore flatter every configuration, and in the same direction:
the true point lies to the right of, and below, the one drawn. That makes
the **envelope** -- the upper-left staircase over the points -- a bound on
the whole catalog, not a promise about any member of it: nothing that can
be built from these parts sits above or left of it.

Dominance follows the bound-aware rule of ``pareto.py`` (P4-D2): a point is
*on* the envelope only when nothing provably beats it, and a figure that is
itself a bound proves nothing about the point being good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from graphs.core.pipeline_workload import CLASS_NAMES, MissionProfile, PipelineWorkload
from graphs.hardware.soc import SoCInstance

from .breakeven import RequiredEfficiency, StageRequirement
from .efficiency import EfficiencyTable, KernelClassMap
from .power import op_energy_pj

#: How a stage's efficiency was known, weakest last.
PROVENANCE = ("measured", "ceiling", "unpriced")


@dataclass(frozen=True)
class StagePlacement:
    """One stage of the mission on the engine that runs it."""

    stage: str
    engine: str
    kernel_class: str
    ops_per_s: float
    #: Seconds of one server per second of mission at the best known
    #: efficiency, or None when nothing prices the stage.
    server_seconds_per_s: Optional[float]
    efficiency: Optional[float]
    provenance: str
    energy_pj_per_op: Optional[float]
    energy_gap: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage, "engine": self.engine, "kernel_class": self.kernel_class,
            "ops_per_s": self.ops_per_s, "efficiency": self.efficiency,
            "provenance": self.provenance,
            "server_seconds_per_s": self.server_seconds_per_s,
            "energy_pj_per_op": self.energy_pj_per_op, "energy_gap": self.energy_gap,
        }


@dataclass(frozen=True)
class MissionPoint:
    """One configuration against one mission, on the energy-performance plane."""

    mission: str
    design: str
    node: str
    cpu_cores: int
    memory: str
    kpu_sku: Optional[str]
    #: pJ per op over the mission's own op mix: a floor.
    energy_per_op_pj: float
    #: Ops the floor could not price, over the mission's ops.
    unpriced_energy_fraction: float
    #: Busiest engine's utilization at the best known efficiencies.
    utilization: float
    #: Engines and their utilization, for the tooltip.
    utilization_by_engine: Dict[str, float] = field(default_factory=dict)
    #: Stages nothing prices, which only flatter the figures above.
    unpriced_stages: Tuple[str, ...] = ()
    #: Weakest provenance among the stages that are priced.
    provenance: str = "measured"
    placements: Tuple[StagePlacement, ...] = ()

    @property
    def real_time_factor(self) -> Optional[float]:
        """Mission rate the configuration could carry, as a multiple of what
        the mission needs. A ceiling: the true factor is no higher."""
        return None if self.utilization <= 0 else 1.0 / self.utilization

    @property
    def attainable(self) -> Optional[bool]:
        """False when even the optimistic factor is short of real time;
        None when it is not, because a ceiling proves nothing either way."""
        factor = self.real_time_factor
        if factor is None:
            return None
        return None if factor >= 1.0 else False

    @property
    def energy_is_floor(self) -> bool:
        return True  # always: no efficiency, no overhead, no leakage

    @property
    def figures_are_partial(self) -> bool:
        return bool(self.unpriced_stages) or self.unpriced_energy_fraction > 0

    def to_dict(self) -> dict:
        return {
            "mission": self.mission, "design": self.design, "node": self.node,
            "cpu_cores": self.cpu_cores, "memory": self.memory, "kpu_sku": self.kpu_sku,
            "energy_per_op_pj": self.energy_per_op_pj,
            "unpriced_energy_fraction": self.unpriced_energy_fraction,
            "utilization": self.utilization,
            "utilization_by_engine": dict(self.utilization_by_engine),
            "real_time_factor": self.real_time_factor,
            "attainable": self.attainable,
            "provenance": self.provenance,
            "unpriced_stages": list(self.unpriced_stages),
            "figures_are_partial": self.figures_are_partial,
        }


def _efficiency(stage: StageRequirement, kernel, kind: str, table: Optional[EfficiencyTable],
                ceilings) -> Tuple[Optional[float], str]:
    """The best efficiency anything states for a stage, and where it came
    from. A measurement beats a ceiling; a ceiling beats nothing."""
    measured = []
    for fmt in stage.formats.values():
        entry = table.lookup(kernel, kind, fmt) if table is not None else None
        if entry is None or not entry.known:
            measured = None
            break
        measured.append(entry.compute_eff)
    if measured:
        return min(measured), "measured"
    if ceilings is not None:
        values = []
        for fmt in stage.formats.values():
            ceiling = ceilings.best(kernel, fmt)
            if ceiling is None or not ceiling.value:
                values = None
                break
            values.append(ceiling.value)
        if values:
            return min(values), "ceiling"
    return None, "unpriced"


def mission_point(requirement: RequiredEfficiency, soc: SoCInstance, workload: PipelineWorkload,
                  profile: MissionProfile, kernels: KernelClassMap,
                  table: Optional[EfficiencyTable] = None, ceilings=None,
                  memory: str = "", kpu_sku: Optional[str] = None) -> MissionPoint:
    """Place one configuration on the plane for one mission."""
    blocks = {b.name: b for b in soc.blocks}
    engines = {e.engine: e for e in requirement.engines}
    stages = {d.stage.key: d for d in workload.demands(profile)}

    placements: List[StagePlacement] = []
    utilization: Dict[str, float] = {}
    ops_total = ops_priced = 0.0
    energy_total = 0.0
    unpriced: List[str] = []
    worst = 0

    for stage in requirement.stages:
        demand = stages[stage.stage]
        ops_per_s = demand.stage.ops_per_call * demand.rate_hz
        ops_total += ops_per_s
        if not stage.engine or stage.dense_seconds is None:
            unpriced.append(stage.stage)
            placements.append(StagePlacement(stage.stage, stage.engine or "", "", ops_per_s,
                                             None, None, "unpriced", None, stage.gap))
            continue
        kernel = kernels.of(stage.stage)
        kind = engines[stage.engine].kind if stage.engine in engines else ""
        fabric = ceilings if kind == "kpu" else None
        efficiency, provenance = _efficiency(stage, kernel, kind, table, fabric)
        worst = max(worst, PROVENANCE.index(provenance))
        if efficiency:
            seconds = (stage.dense_seconds / efficiency) * stage.rate_hz
            servers = engines[stage.engine].servers or 1
            utilization[stage.engine] = utilization.get(stage.engine, 0.0) + seconds / servers
        else:
            unpriced.append(stage.stage)

        # The energy floor: every class at its own format's price.
        pj_ops = 0.0
        for cls, share in zip(CLASS_NAMES, demand.stage.class_split):
            if share <= 0 or cls not in stage.formats:
                continue
            pj, why = op_energy_pj(blocks[stage.engine], soc.node, stage.formats[cls])
            class_ops = ops_per_s * share
            if pj is None:
                placements.append(StagePlacement(
                    stage.stage, stage.engine, kernel.value, class_ops, None, efficiency,
                    provenance, None, why))
                continue
            pj_ops += class_ops * pj
            ops_priced += class_ops
        energy_total += pj_ops
        placements.append(StagePlacement(
            stage.stage, stage.engine, kernel.value, ops_per_s,
            None if not efficiency else stage.dense_seconds / efficiency * stage.rate_hz,
            efficiency, provenance, None if not ops_per_s else pj_ops / ops_per_s))

    return MissionPoint(
        mission=profile.id, design=requirement.design, node=requirement.node,
        cpu_cores=next((e.servers for e in requirement.engines if e.kind == "cpu"), 0),
        memory=memory, kpu_sku=kpu_sku,
        energy_per_op_pj=0.0 if ops_priced <= 0 else energy_total / ops_priced,
        unpriced_energy_fraction=0.0 if ops_total <= 0 else 1.0 - ops_priced / ops_total,
        utilization=max(utilization.values(), default=0.0),
        utilization_by_engine=utilization,
        unpriced_stages=tuple(sorted(set(unpriced))),
        provenance=PROVENANCE[worst] if placements else "unpriced",
        placements=tuple(placements))


def envelope(points: Sequence[MissionPoint]) -> Tuple[MissionPoint, ...]:
    """The upper-left staircase: points no other point beats on both axes.

    Beating means less energy per op *and* a higher real-time factor. Both
    figures flatter their point, so this is an envelope over what the
    catalog could do, not a claim that any point on it performs.
    """
    usable = [p for p in points if p.real_time_factor is not None and p.energy_per_op_pj > 0]
    out = []
    for point in usable:
        beaten = any(
            other is not point
            and other.energy_per_op_pj <= point.energy_per_op_pj
            and other.real_time_factor >= point.real_time_factor
            and (other.energy_per_op_pj < point.energy_per_op_pj
                 or other.real_time_factor > point.real_time_factor)
            for other in usable)
        if not beaten:
            out.append(point)
    return tuple(sorted(out, key=lambda p: p.energy_per_op_pj))


def attainable_envelope(points: Sequence[MissionPoint]) -> Tuple[MissionPoint, ...]:
    """The envelope over points that are not already proven short of real
    time -- the configurations still worth arguing about."""
    return envelope([p for p in points if p.attainable is not False])


__all__ = ["MissionPoint", "PROVENANCE", "StagePlacement", "attainable_envelope", "envelope",
           "mission_point"]
