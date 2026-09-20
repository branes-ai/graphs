"""Sizing one SoC to one mission, stage by stage (graphs#269 Phase 7.6).

The catalogue pages answer "is this configuration ruled out?". This module
answers the question that comes after it: **given this mission, how big does
each engine have to be?**

The arithmetic is deliberately small, because the honesty is in the inputs
rather than in the algebra. For one stage on one engine:

    servers needed = sum over precision classes of
                     (ops per second in that class) / (peak per server x E)

where the format a class runs in is the narrowest the engine offers at or
above the class's floor (``execution_format``), and ``E`` is the efficiency
that kernel class attains in that format: a measurement where one exists,
otherwise the domain-flow ceiling, which bounds rather than predicts. If any
class the stage needs has no figure, the stage has **no fit** on that engine
and the gap is named -- nothing is interpolated, and a wider format is never
silently substituted for a narrower one.

Three quantities describe an engine once it is sized, and the report carries
all three because any two of them mislead:

    X  delivered throughput  = peak per server x E       (ops/s a server does)
    U  utilization           = demand / (X x servers)    (share of wall clock)
    E  efficiency            = X / peak                  (share of the datapath)

A fabric with a high E and a low U is oversized; a high U and a low E is
badly served by its kernel. Only X is what the mission actually receives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from graphs.core.confidence import EstimationConfidence
from graphs.core.pipeline_workload import (
    CLASS_NAMES,
    REACTIVE_CHAIN,
    MissionProfile,
    PipelineWorkload,
)
from graphs.hardware.soc import Confidence, SoCInstance

from .efficiency import EfficiencyTable, KernelClassMap, execution_format
from .mapping import estimation_confidence
from .power import op_energy_pj


@dataclass(frozen=True)
class ClassFit:
    """One precision class of one stage on one engine."""

    precision_class: str
    share: float
    fmt: Optional[str]
    ops_per_s: float
    peak_per_server: Optional[float]
    efficiency: Optional[float]
    provenance: str
    #: Delivered throughput of one server on this class, ops/s.
    throughput_per_server: Optional[float]
    servers_needed: Optional[float]
    gap: Optional[str] = None


@dataclass(frozen=True)
class EngineFit:
    """What one engine would have to be to carry one stage on its own."""

    engine: str
    kind: str
    classes: Tuple[ClassFit, ...]
    servers_needed: Optional[float]
    provenance: str
    gap: Optional[str] = None

    @property
    def fits(self) -> bool:
        return self.servers_needed is not None

    @property
    def seconds_per_call_per_server(self) -> Optional[float]:
        """Latency of one call on a single server, which is what a pipeline
        node is labelled with."""
        if not self.fits:
            return None
        total = 0.0
        for c in self.classes:
            if c.throughput_per_server is None or not c.throughput_per_server:
                return None
            total += c.ops_per_s / c.throughput_per_server
        return total


@dataclass(frozen=True)
class StageDemand:
    """One stage of the mission, and what each engine would need to carry it."""

    key: str
    name: str
    tier: str
    kernel_class: str
    unit: str
    rate_hz: float
    calls_per_s: float
    ops_per_call: float
    bytes_per_call: float
    ops_per_s: float
    bytes_per_s: float
    class_split: Dict[str, float]
    on_reactive_chain: bool
    basis: str
    fits: Dict[str, EngineFit] = field(default_factory=dict)

    @property
    def only_engine(self) -> Optional[str]:
        """The engine that can carry it, when exactly one can. That is the
        interesting case: the mapping is forced, not chosen."""
        able = [name for name, fit in self.fits.items() if fit.fits]
        return able[0] if len(able) == 1 else None


@dataclass(frozen=True)
class Provision:
    """One engine as sized: how many servers, and the three numbers."""

    engine: str
    kind: str
    servers_needed: float
    servers_provisioned: int
    unit: str
    #: X: ops/s one server delivers while it is running this work. Mixed
    #: formats are folded in by time, so X x servers x U is the demand
    #: exactly -- that identity is the point of carrying all three.
    throughput_ops_per_s: float
    #: U: share of wall clock the sized engine is busy.
    utilization: float
    #: E: ops-weighted mean of the per-class efficiencies.
    efficiency: float
    provenance: str
    stages: Tuple[str, ...]
    #: Dense peak of one server in ``peak_format``, for scale against X.
    peak_ops_per_s: float
    peak_format: str = ""

    @property
    def aggregate_throughput_ops_per_s(self) -> float:
        return self.throughput_ops_per_s * self.servers_provisioned

    @property
    def demand_ops_per_s(self) -> float:
        """X x servers x U -- the demand the sizing was built from."""
        return self.aggregate_throughput_ops_per_s * self.utilization

    @property
    def headroom(self) -> float:
        return 0.0 if self.utilization <= 0 else 1.0 / self.utilization


@dataclass(frozen=True)
class Placement:
    """One stage as placed and sized: where it runs and what that costs."""

    stage: str
    engine: str
    servers: int
    #: Calls of this stage per frame of the pipeline's output cadence.
    calls_per_frame: float
    #: Wall-clock seconds this stage occupies per frame on the sized engine.
    seconds_per_frame: float
    #: Share of the sized engine this stage alone consumes.
    utilization: float
    provenance: str


@dataclass(frozen=True)
class Dossier:
    """Everything the page draws for one mission on one SoC."""

    mission: str
    title: str
    power_budget_w: float
    deadline_ms: float
    note: str
    sensors: Dict[str, object]
    design: str
    node: str
    #: The pipeline's output cadence: the slowest stage's rate, which is the
    #: unit a frame of latency is measured in.
    frame_hz: float
    stages: Tuple[StageDemand, ...]
    provisions: Tuple[Provision, ...]
    placements: Tuple[Placement, ...] = ()
    dram_demand_gb_per_s: float = 0.0
    dram_supply_gb_per_s: Optional[float] = None
    dram_sustained_fraction: float = 1.0
    datapath_watts: Dict[str, float] = field(default_factory=dict)
    energy_gaps: Tuple[str, ...] = ()
    area_mm2: Dict[str, Optional[float]] = field(default_factory=dict)
    area_gaps: Tuple[str, ...] = ()
    unplaced: Tuple[str, ...] = ()
    #: Engines the design cannot supply enough servers for. The provision
    #: is still reported, capped, with a utilization above 1.
    oversubscribed: Tuple[str, ...] = ()

    @property
    def datapath_total_w(self) -> float:
        return sum(self.datapath_watts.values())

    @property
    def chain_seconds(self) -> Optional[float]:
        """Sense-to-act latency for one frame: every placed stage in
        pipeline order, each on the engine it was sized for. A serial sum,
        which is the honest reading for a pipeline whose stages feed one
        another within a frame."""
        if self.unplaced:
            return None
        return sum(p.seconds_per_frame for p in self.placements)

    @property
    def deadline_headroom(self) -> Optional[float]:
        chain = self.chain_seconds
        if chain is None or chain <= 0:
            return None
        return (self.deadline_ms / 1000.0) / chain

    @property
    def power_budget_fraction_used(self) -> Optional[float]:
        if self.power_budget_w <= 0:
            return None
        return self.datapath_total_w / self.power_budget_w

    @property
    def fits(self) -> bool:
        """False when the demand does not fit the design at all."""
        return not self.unplaced and not self.oversubscribed

    @property
    def estimation_confidence(self) -> EstimationConfidence:
        if self.unplaced:
            return estimation_confidence(
                Confidence.UNKNOWN,
                f"{len(self.unplaced)} stage(s) no engine can be sized for: "
                + ", ".join(self.unplaced))
        if self.oversubscribed:
            return estimation_confidence(
                Confidence.UNKNOWN,
                "the design cannot supply what the mission needs: "
                + "; ".join(self.oversubscribed))
        if any(p.provenance == "ceiling" for p in self.provisions):
            return estimation_confidence(
                Confidence.THEORETICAL,
                "at least one engine is sized on a domain-flow ceiling, which bounds "
                "rather than predicts: the engine needed is no smaller than stated")
        return estimation_confidence(
            Confidence.CALIBRATED, "every engine is sized on a measured efficiency")


def _efficiency_for(kernel, kind: str, fmt: str, table: Optional[EfficiencyTable],
                    ceilings) -> Tuple[Optional[float], str, Optional[str]]:
    """The efficiency one class attains, and where it came from."""
    entry = table.lookup(kernel, kind, fmt) if table is not None else None
    if entry is not None and entry.known:
        return entry.compute_eff, "measured", None
    if ceilings is not None:
        ceiling = ceilings.best(kernel, fmt)
        if ceiling is not None and ceiling.value:
            return ceiling.value, "ceiling", None
        return None, "unpriced", f"no {fmt} figure and no domain-flow schedule"
    return None, "unpriced", f"no {fmt} figure"


#: Weakest last, as in ``frontier``.
PROVENANCE = ("measured", "ceiling", "unpriced")


def fit_stage(stage, rate_hz: float, kernel, engine, table: Optional[EfficiencyTable],
              ceilings) -> EngineFit:
    """What one engine would have to be to carry one stage on its own."""
    classes: List[ClassFit] = []
    total: Optional[float] = 0.0
    worst = 0
    gap: Optional[str] = None
    ops_per_s = stage.ops_per_call * rate_hz
    for name, share in zip(CLASS_NAMES, stage.class_split):
        if share <= 0:
            continue
        fmt = execution_format(name, engine.formats)
        class_ops = ops_per_s * share
        if fmt is None:
            classes.append(ClassFit(name, share, None, class_ops, None, None, "unpriced",
                                    None, None, "the engine has no format for this class"))
            total, gap = None, gap or f"class {name}: the engine has no format for it"
            worst = max(worst, PROVENANCE.index("unpriced"))
            continue
        peak = engine.server_peak_ops_per_s(fmt)
        eff, provenance, why = _efficiency_for(kernel, engine.kind, fmt, table, ceilings)
        worst = max(worst, PROVENANCE.index(provenance))
        if not eff or not peak:
            classes.append(ClassFit(name, share, fmt, class_ops, peak, None, "unpriced",
                                    None, None, why or "no peak stated"))
            total, gap = None, gap or f"class {name} in {fmt}: {why or 'no peak stated'}"
            continue
        throughput = peak * eff
        needed = class_ops / throughput
        classes.append(ClassFit(name, share, fmt, class_ops, peak, eff, provenance,
                                throughput, needed))
        if total is not None:
            total += needed
    return EngineFit(engine=engine.name if hasattr(engine, "name") else "",
                     kind=_kind(engine), classes=tuple(classes), servers_needed=total,
                     provenance=PROVENANCE[worst], gap=gap)


def _kind(engine) -> str:
    """The engine kind as a plain string. ``EngineKind`` compares equal to
    its value but formats as ``EngineKind.KPU``, which lands in generated
    markup as a CSS variable nobody defined."""
    return getattr(engine.kind, "value", engine.kind)


def provision(needed: float, target_utilization: float, minimum: int = 1) -> int:
    """Servers to fit, at or under a target utilization. Whole servers: half
    a CPU core is not a thing you can buy."""
    if target_utilization <= 0 or target_utilization > 1:
        raise ValueError(f"target_utilization must be in (0, 1], got {target_utilization}")
    return max(minimum, int(math.ceil(needed / target_utilization)))


def dimension(workload: PipelineWorkload, profile: MissionProfile, soc: SoCInstance,
              kernels: KernelClassMap, table: Optional[EfficiencyTable] = None,
              ceilings=None, target_utilization: float = 0.85,
              tiles_per_server: int = 1) -> Dossier:
    """Size every engine of ``soc`` to ``profile``.

    Each stage is fitted to every engine; a stage only one engine can carry
    is placed there, with no choice to make. ``tiles_per_server`` divides a
    KPU engine's peak into the unit the fabric is actually built from, so
    the answer comes back in tiles rather than in whole catalogued cores.
    """
    from .mapping import engines_of  # noqa: PLC0415 -- avoids an import cycle

    engines = engines_of(soc)
    blocks = {b.name: b for b in soc.blocks}
    demands = workload.demands(profile)
    frame_hz = min((d.rate_hz for d in demands), default=1.0)

    stages: List[StageDemand] = []
    for demand in demands:
        stage = demand.stage
        kernel = kernels.of(stage.key)
        fits: Dict[str, EngineFit] = {}
        for name, engine in engines.items():
            fabric = ceilings if engine.kind == "kpu" else None
            unit = tiles_per_server if engine.kind == "kpu" else 1
            fit = fit_stage(stage, demand.rate_hz, kernel, _Scaled(engine, name, unit),
                            table, fabric)
            fits[name] = fit
        stages.append(StageDemand(
            key=stage.key, name=stage.name, tier=stage.pipeline_tier,
            kernel_class=kernel.value, unit=stage.unit, rate_hz=demand.rate_hz,
            calls_per_s=demand.rate_hz, ops_per_call=stage.ops_per_call,
            bytes_per_call=stage.bytes_per_call,
            ops_per_s=stage.ops_per_call * demand.rate_hz,
            bytes_per_s=stage.bytes_per_call * demand.rate_hz,
            class_split=dict(zip(CLASS_NAMES, stage.class_split)),
            on_reactive_chain=stage.key in REACTIVE_CHAIN,
            basis=stage.basis or "", fits=fits))

    # Place each stage on the engine that can carry it. Where more than one
    # can, the fewest servers wins; where none can, it is an explicit gap.
    placement: Dict[str, str] = {}
    unplaced: List[str] = []
    for stage in stages:
        able = {n: f for n, f in stage.fits.items() if f.fits}
        if not able:
            unplaced.append(stage.key)
            continue
        placement[stage.key] = min(able, key=lambda n: able[n].servers_needed)

    provisions: List[Provision] = []
    placements: List[Placement] = []
    oversubscribed: List[str] = []
    for name, engine in engines.items():
        mine = [s for s in stages if placement.get(s.key) == name]
        if not mine:
            continue
        needed = sum(s.fits[name].servers_needed for s in mine)
        unit = tiles_per_server if engine.kind == "kpu" else 1
        maximum = (engine.servers * unit) if engine.kind == "kpu" else engine.servers
        # The design caps how many servers exist. Asking for more is not a
        # sizing, it is a gap: capping it silently would report U > 1 as
        # though the configuration had been sized.
        unit_name = "tile" if engine.kind == "kpu" else "core"
        servers = min(provision(needed, target_utilization), maximum)
        if needed > maximum:
            oversubscribed.append(
                f"{name}: needs {needed:.3f} {unit_name}s but the design has {maximum}")
        worst = max(PROVENANCE.index(s.fits[name].provenance) for s in mine)
        # X, U and E for the sized engine, over the ops it actually carries.
        ops = sum(s.ops_per_s for s in mine)
        priced = [c for s in mine for c in s.fits[name].classes if c.efficiency]
        # E: ops-weighted mean of the per-class efficiencies. A single E
        # across two formats is only meaningful this way, because the peaks
        # underneath it differ.
        weighted_eff = (sum(c.ops_per_s * c.efficiency for c in priced) / ops) if ops else 0.0
        heaviest = max(priced, key=lambda c: c.ops_per_s, default=None)
        provisions.append(Provision(
            engine=name, kind=_kind(engine), servers_needed=needed,
            servers_provisioned=servers,
            unit="tile" if engine.kind == "kpu" else "core",
            throughput_ops_per_s=(ops / needed) if needed else 0.0,
            utilization=needed / servers if servers else 0.0,
            efficiency=weighted_eff, provenance=PROVENANCE[worst],
            stages=tuple(s.key for s in mine),
            peak_ops_per_s=(heaviest.peak_per_server if heaviest else 0.0),
            peak_format=(heaviest.fmt if heaviest else "")))
        for stage in mine:
            fit = stage.fits[name]
            calls_per_frame = stage.calls_per_s / frame_hz if frame_hz else 1.0
            seconds = sum(c.ops_per_s / c.throughput_per_server
                          for c in fit.classes if c.throughput_per_server)
            placements.append(Placement(
                stage=stage.key, engine=name, servers=servers,
                calls_per_frame=calls_per_frame,
                seconds_per_frame=seconds / servers / frame_hz if frame_hz else 0.0,
                utilization=fit.servers_needed / servers if servers else 0.0,
                provenance=fit.provenance))

    watts: Dict[str, float] = {}
    energy_gaps: List[str] = []
    for stage in stages:
        engine = placement.get(stage.key)
        if engine is None:
            continue
        for cls in stage.fits[engine].classes:
            if cls.fmt is None:
                continue
            pj, why = op_energy_pj(blocks[engine], soc.node, cls.fmt)
            if pj is None:
                energy_gaps.append(f"{stage.key} class {cls.precision_class}: {why}")
                continue
            watts[engine] = watts.get(engine, 0.0) + cls.ops_per_s * pj * 1e-12

    order = {s.key: i for i, s in enumerate(stages)}
    placements.sort(key=lambda p: order[p.stage])
    return Dossier(
        mission=profile.id, title=f"{profile.form_factor}: {profile.name}",
        power_budget_w=profile.power_budget_w, deadline_ms=profile.deadline_ms,
        note=profile.note or "", sensors=dict(profile.sensors or {}),
        design=soc.design.id if hasattr(soc, "design") else "", node=soc.node.id,
        frame_hz=frame_hz, stages=tuple(stages), provisions=tuple(provisions),
        placements=tuple(placements),
        dram_demand_gb_per_s=sum(s.bytes_per_s for s in stages) / 1e9,
        dram_supply_gb_per_s=soc.dram_peak_gb_per_s or None,
        datapath_watts=watts, energy_gaps=tuple(energy_gaps),
        unplaced=tuple(unplaced), oversubscribed=tuple(oversubscribed))


class _Scaled:
    """An engine seen one tile at a time rather than one core at a time."""

    def __init__(self, engine, name: str, divisor: int):
        self._engine, self.name, self._divisor = engine, name, divisor
        self.kind, self.formats = engine.kind, engine.formats
        self.servers = engine.servers * divisor

    def server_peak_ops_per_s(self, fmt: str) -> float:
        return self._engine.server_peak_ops_per_s(fmt) / self._divisor


__all__ = ["ClassFit", "Dossier", "EngineFit", "PROVENANCE", "Placement", "Provision",
           "StageDemand", "dimension", "fit_stage", "provision"]
