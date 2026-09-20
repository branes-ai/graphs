"""What a design would have to achieve, when nothing prices its engines
(graphs#269 Phase 6).

The analyzer prices a stage as ``ops / (peak x efficiency)``. For a KPU no
efficiency exists: no silicon has run these kernels, and the SKU's
hand-authored efficiency factors are not a source. Under the strict rule
that leaves every KPU stage a gap, and a CPU+KPU design cannot be compared
with anything.

This module asks the question the other way round, which needs no figure
nobody has: **what efficiency would each engine have to sustain for the
profile to fit?**

For a stage on one server of an engine, the seconds one call needs at a
perfect 100% of dense peak are

    dense_seconds = sum_c share_c x ops_per_call / peak(fmt_c)

and its *occupancy at peak* is ``dense_seconds x rate``: the share of one
server a flawless engine would still need. Efficiency divides throughput,
so an engine running every stage mapped to it at one efficiency ``e`` has

    utilization = (1 / e) x sum_s occupancy_at_peak_s / servers

and utilization <= 1 gives

    required efficiency = sum_s occupancy_at_peak_s / servers.

That number is the answer: the fraction of dense peak the engine must hold,
over its stages, for the profile to be carried. It is exact arithmetic on
the workload's ops and the design's peaks -- no efficiency is assumed, and
none is invented. Above 1 it says the engine cannot carry its stages at any
efficiency: the silicon is too small, not too slow.

Two things it does not fold in:

* **Memory.** ``memory_occupancy`` is the same ratio for the stage's DRAM
  time at the design's sustained bandwidth. No efficiency changes it, so a
  stage at or above 1 there is out of reach whatever the datapath does. It
  is a lower bound: it has the stage own the bandwidth (contention is the
  response analysis's business).
* **Scheduling.** Utilization at or under 1 is necessary, not sufficient;
  the response analysis is what proves a schedule.

``known_efficiency`` puts a table's measured figure beside the requirement,
so "needs 0.34, measures 0.08" reads directly, and
``utilization_at_known`` prices the stages the table does know at its own
figures -- a lower bound on utilization whenever it knows fewer than all of
them, because the rest add to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from graphs.core.confidence import EstimationConfidence
from graphs.core.pipeline_workload import MissionProfile, PipelineWorkload, StageDemand
from graphs.hardware.soc import Confidence, SoCInstance

from .efficiency import CLASS_NAMES, EfficiencyTable, KernelClass, KernelClassMap, execution_format
from .mapping import (
    SUSTAINED_DRAM_FRACTION,
    WORKLOAD_CONFIDENCE,
    Engine,
    engines_of,
    estimation_confidence,
    weakest,
)

def _dense_seconds(stage, engine: Engine) -> Optional[float]:
    """Seconds one call of ``stage`` takes on one server of ``engine`` at
    100% of dense peak, or None when the engine cannot run one of its
    classes."""
    total = 0.0
    for cls, share in zip(CLASS_NAMES, stage.class_split):
        if share <= 0:
            continue
        fmt = execution_format(cls, engine.formats)
        if fmt is None:
            return None
        total += stage.ops_per_call * share / engine.server_peak_ops_per_s(fmt)
    return total


@dataclass(frozen=True)
class StageRequirement:
    """One stage on one engine: what a perfect datapath would still need."""

    stage: str
    engine: str
    rate_hz: float
    formats: Dict[str, str] = field(default_factory=dict)
    #: Seconds one call needs at 100% of dense peak, on one server.
    dense_seconds: Optional[float] = None
    #: That time split by precision class, since each runs in its own
    #: format: a consumer with a per-format figure must not apply one of
    #: them to the whole stage.
    class_seconds: Dict[str, float] = field(default_factory=dict)
    #: Seconds of DRAM per call at the sustained bandwidth, owning it.
    memory_seconds: Optional[float] = None
    #: Seconds one call takes at the comparison table's own efficiencies,
    #: or None when it does not price every format the stage uses.
    known_seconds: Optional[float] = None
    #: Why no engine can run it, if none can.
    gap: Optional[str] = None

    @property
    def occupancy_at_peak(self) -> Optional[float]:
        """Share of one server the stage needs at 100% of dense peak."""
        return None if self.dense_seconds is None else self.dense_seconds * self.rate_hz

    @property
    def memory_occupancy(self) -> Optional[float]:
        return None if self.memory_seconds is None else self.memory_seconds * self.rate_hz

    @property
    def known_efficiency(self) -> Optional[float]:
        """The efficiency the table's figures amount to over this stage: the
        dense time over the time they give it."""
        if self.known_seconds is None or not self.known_seconds or self.dense_seconds is None:
            return None
        return self.dense_seconds / self.known_seconds

    @property
    def out_of_reach(self) -> bool:
        """True when no efficiency helps: the DRAM time alone fills the period."""
        return (self.memory_occupancy or 0.0) >= 1.0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "engine": self.engine,
            "rate_hz": self.rate_hz,
            "formats": dict(self.formats),
            "occupancy_at_peak": self.occupancy_at_peak,
            "class_occupancy_at_peak": {c: t * self.rate_hz
                                        for c, t in self.class_seconds.items()},
            "memory_occupancy": self.memory_occupancy,
            "known_efficiency": self.known_efficiency,
            "known_seconds": self.known_seconds,
            "out_of_reach": self.out_of_reach,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class EngineRequirement:
    """The efficiency one engine must sustain over the stages mapped to it."""

    engine: str
    kind: str
    servers: int
    stages: Tuple[str, ...]
    #: sum(occupancy_at_peak) / servers, or None when a stage has no time.
    required_efficiency: Optional[float]
    #: Stages whose DRAM time alone fills their period.
    out_of_reach: Tuple[str, ...] = ()
    #: Utilization with each stage priced at the table's own efficiency,
    #: over the stages the table prices. None when it prices none.
    utilization_at_known: Optional[float] = None
    #: Stages the comparison table states no efficiency for.
    unpriced: Tuple[str, ...] = ()

    @property
    def known_is_lower_bound(self) -> bool:
        """``utilization_at_known`` omits the stages nothing prices."""
        return bool(self.unpriced)

    @property
    def reachable(self) -> Optional[bool]:
        """False when even a perfect datapath is too small; None when open."""
        if self.required_efficiency is None:
            return None
        return self.required_efficiency <= 1.0 and not self.out_of_reach

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "kind": self.kind,
            "servers": self.servers,
            "stages": list(self.stages),
            "required_efficiency": self.required_efficiency,
            "reachable": self.reachable,
            "out_of_reach": list(self.out_of_reach),
            "utilization_at_known": self.utilization_at_known,
            "utilization_at_known_is_lower_bound": self.known_is_lower_bound,
            "unpriced": list(self.unpriced),
        }


@dataclass(frozen=True)
class RequiredEfficiency:
    """What a design would have to achieve to carry one profile."""

    design: str
    node: str
    profile: MissionProfile
    engines: Tuple[EngineRequirement, ...]
    stages: Tuple[StageRequirement, ...]
    dram_demand_gb_per_s: float
    dram_supply_gb_per_s: Optional[float]
    #: The weakest input: the workload's unit costs, and UNKNOWN when an
    #: engine's clock is provisional at this node (its peak does not apply
    #: here, so neither does a requirement measured against it).
    estimation_confidence: EstimationConfidence
    comparison_table: Optional[str] = None

    @property
    def unrunnable(self) -> Tuple[str, ...]:
        """Stages no engine in the design can run, whatever the efficiency."""
        return tuple(s.stage for s in self.stages if s.gap and not s.engine)

    @property
    def misassigned(self) -> Tuple[str, ...]:
        """Stages the mapping put on an engine that cannot run them. Another
        engine in the design may well be able to."""
        return tuple(s.stage for s in self.stages if s.gap and s.engine)

    def to_dict(self) -> dict:
        return {
            "design": self.design,
            "node": self.node,
            "profile": self.profile.id,
            "regime": self.profile.regime,
            "comparison_table": self.comparison_table,
            "confidence": None if self.estimation_confidence is None
            else self.estimation_confidence.level.value,
            "confidence_source": None if self.estimation_confidence is None
            else self.estimation_confidence.source,
            "engines": [e.to_dict() for e in self.engines],
            "stages": [s.to_dict() for s in self.stages],
            "unrunnable": list(self.unrunnable),
            "misassigned": list(self.misassigned),
            "memory": {
                "dram_demand_gb_per_s": self.dram_demand_gb_per_s,
                "dram_supply_gb_per_s": self.dram_supply_gb_per_s,
                "utilization": (None if not self.dram_supply_gb_per_s
                                else self.dram_demand_gb_per_s / self.dram_supply_gb_per_s),
            },
        }


def capability_mapping(demands: List[StageDemand],
                       engines: Mapping[str, Engine]) -> Dict[str, Optional[str]]:
    """Each stage on the engine that *can* run every class it has and would
    take the fewest seconds at dense peak, ties broken by name.

    A rule, not an optimum: it reads only the formats an engine has and its
    peak, never an efficiency, so it works on a design nothing prices. The
    engine kind does not enter -- an accelerator wins because its peak is
    higher, not because of what it is called.
    """
    out: Dict[str, Optional[str]] = {}
    for demand in demands:
        capable = [(seconds, name) for name, engine in engines.items()
                   if (seconds := _dense_seconds(demand.stage, engine)) is not None]
        out[demand.stage.key] = min(capable)[1] if capable else None
    return out


def _known_seconds(table: Optional[EfficiencyTable], kernels: Optional[KernelClassMap],
                   stage, engine: Engine, formats: Mapping[str, str]) -> Optional[float]:
    """Seconds one call takes at the table's own efficiencies, the same
    per-format sum the mapper uses. None unless the table prices *every*
    format the stage runs in: one priced class does not price the stage."""
    if table is None or kernels is None or table.kind == "pooled":
        return None
    kernel: KernelClass = kernels.of(stage.key)
    total = 0.0
    for cls, fmt in formats.items():
        entry = table.lookup(kernel, engine.kind, fmt)
        if entry is None or not entry.known:
            return None
        share = stage.class_split[CLASS_NAMES.index(cls)]
        total += stage.ops_per_call * share / (engine.server_peak_ops_per_s(fmt) * entry.compute_eff)
    return total


def required_efficiency(
    workload: PipelineWorkload,
    profile: MissionProfile,
    soc: SoCInstance,
    mapping: Optional[Mapping[str, Optional[str]]] = None,
    table: Optional[EfficiencyTable] = None,
    kernels: Optional[KernelClassMap] = None,
    sustained_fraction: float = SUSTAINED_DRAM_FRACTION,
) -> RequiredEfficiency:
    """What each engine of ``soc`` must sustain to carry ``profile``.

    ``mapping`` names the engine for the stages it lists; a stage it leaves
    out falls back to :func:`capability_mapping`, and a stage it maps to
    ``None`` is reported as one nothing runs. ``table`` and ``kernels``,
    when given, put the table's own efficiency beside each stage's
    requirement.
    """
    if not 0 < sustained_fraction <= 1:
        raise ValueError(f"sustained_fraction must be in (0, 1], got {sustained_fraction}")
    engines = engines_of(soc)
    demands = list(workload.demands(profile))
    assigned = capability_mapping(demands, engines)
    assigned.update(mapping or {})
    supply = (soc.dram_peak_gb_per_s * sustained_fraction) if soc.dram_peak_gb_per_s > 0 else None

    stages: List[StageRequirement] = []
    for demand in demands:
        stage = demand.stage
        name = assigned.get(stage.key)
        memory = stage.bytes_per_call / (supply * 1e9) if supply else None
        if name is None:
            classes = [c for c, share in zip(CLASS_NAMES, stage.class_split) if share > 0]
            stages.append(StageRequirement(
                stage.key, "", demand.rate_hz, memory_seconds=memory,
                gap=f"no engine runs Class {', '.join(classes)}"))
            continue
        engine = engines[name]
        formats: Dict[str, str] = {}
        per_class: Dict[str, float] = {}
        dense = 0.0
        for cls, share in zip(CLASS_NAMES, stage.class_split):
            if share <= 0:
                continue
            fmt = execution_format(cls, engine.formats)
            if fmt is None:
                dense = None  # type: ignore[assignment]
                break
            formats[cls] = fmt
            per_class[cls] = stage.ops_per_call * share / engine.server_peak_ops_per_s(fmt)
            dense += per_class[cls]
        if dense is None:
            stages.append(StageRequirement(
                stage.key, name, demand.rate_hz, formats=formats, memory_seconds=memory,
                gap=f"{engine.kind.value} cannot run every class of {stage.key}"))
            continue
        stages.append(StageRequirement(
            stage.key, name, demand.rate_hz, formats=formats, dense_seconds=dense,
            class_seconds=per_class, memory_seconds=memory,
            known_seconds=_known_seconds(table, kernels, stage, engine, formats)))

    per_engine: List[EngineRequirement] = []
    for name, engine in engines.items():
        mine = [s for s in stages if s.engine == name]
        if not mine:
            continue
        occupancies = [s.occupancy_at_peak for s in mine]
        total = (None if any(o is None for o in occupancies)
                 else sum(occupancies) / engine.servers)
        priced = [s for s in mine if s.known_seconds is not None]
        at_known = (sum(s.known_seconds * s.rate_hz for s in priced) / engine.servers
                    if priced else None)
        per_engine.append(EngineRequirement(
            name, engine.kind.value, engine.servers, tuple(s.stage for s in mine),
            required_efficiency=total,
            out_of_reach=tuple(s.stage for s in mine if s.out_of_reach),
            utilization_at_known=at_known,
            unpriced=tuple(s.stage for s in mine if s.known_seconds is None)))

    used = [engines[e.engine] for e in per_engine]
    provisional = [e.name for e in used if not e.block.clock_is_reference]
    level = weakest(WORKLOAD_CONFIDENCE, *([Confidence.UNKNOWN] if provisional else []))
    source = "workload unit costs and the design's dense peaks"
    if provisional:
        source = f"{', '.join(provisional)} clock is provisional at {soc.node.id}; " + source
    return RequiredEfficiency(
        design=soc.design.id, node=soc.node.id, profile=profile,
        engines=tuple(per_engine), stages=tuple(stages),
        dram_demand_gb_per_s=sum(d.bytes_per_s for d in demands) / 1e9,
        dram_supply_gb_per_s=supply,
        comparison_table=None if table is None else table.id,
        estimation_confidence=estimation_confidence(level, source))


__all__ = ["EngineRequirement", "RequiredEfficiency", "StageRequirement", "capability_mapping",
           "required_efficiency"]
