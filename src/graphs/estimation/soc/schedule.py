"""The per-profile schedule: utilization, DRAM, constraint ratios, critical path
(graphs#269 PR 3.2).

A ``Schedule`` is one mission profile run on one composed SoC with one
efficiency table and one mapping. It reports, per parent plan 5.6:

* **Stage ratios** -- service time over the stage's period (the annex's
  metric, so "stages over" compares directly with it);
* **Engine utilization** -- the sum of a server pool's stage occupancies over
  its server count; above 1 the engine cannot carry what is mapped to it;
* **Shared DRAM** -- the profile's bytes/s against peak x sustained fraction;
* **The reactive chain** -- one call of each sense-to-act stage in sequence,
  against the profile's deadline.

Every total is over the stages that could be priced. When any stage is a gap
-- infeasible on its engine, unmapped, or with no known efficiency -- the
schedule is **incomplete** and its totals are lower bounds; ``gaps`` says
which stages and why. Nothing is filled in to make a total look whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from graphs.core.pipeline_workload import REACTIVE_CHAIN, MissionProfile, PipelineWorkload
from graphs.core.confidence import EstimationConfidence
from graphs.hardware.soc import Confidence, SoCInstance

from .efficiency import EfficiencyTable, KernelClassMap
from .mapping import (
    POOLED,
    SUSTAINED_DRAM_FRACTION,
    Engine,
    StageService,
    check_explicit,
    engine_service,
    engines_of,
    estimation_confidence,
    greedy_mapping,
    pooled_service,
    split_service,
    weakest,
)


_LEVEL_ORDER = [Confidence.CALIBRATED, Confidence.INTERPOLATED, Confidence.THEORETICAL,
                Confidence.UNKNOWN]


@dataclass(frozen=True)
class Schedule:
    profile: MissionProfile
    table: str
    mapping: str  # "pooled", "explicit", "greedy" or "ilp"
    services: Tuple[StageService, ...]
    servers: Dict[str, int]
    dram_demand_gb_per_s: float
    dram_supply_gb_per_s: Optional[float]  # sustained; None when the design states none

    # ---- completeness ----------------------------------------------------

    @property
    def gaps(self) -> Tuple[Tuple[str, str], ...]:
        return tuple((s.stage, s.gap) for s in self.services if not s.served)

    @property
    def lower_bounds(self) -> Tuple[str, ...]:
        """Priced stages whose time is only a floor (a split with no stated
        transfer size)."""
        return tuple(s.stage for s in self.services if s.served and s.lower_bound)

    @property
    def complete(self) -> bool:
        return not self.gaps and not self.lower_bounds

    @property
    def served(self) -> Tuple[StageService, ...]:
        return tuple(s for s in self.services if s.served)

    # ---- the annex's metrics ---------------------------------------------

    @property
    def oversubscription(self) -> float:
        """Seconds of engine time demanded per second of mission, over priced
        stages. On a pooled table this is the annex's figure."""
        return sum(s.occupancy for s in self.served)

    def stages_over(self) -> Tuple[str, ...]:
        return tuple(s.stage for s in self.served if s.occupancy >= 1.0)

    # ---- engines and memory ----------------------------------------------

    def engine_utilization(self) -> Dict[str, float]:
        out = {name: 0.0 for name in self.servers}
        for s in self.served:
            for engine, seconds in s.engine_seconds.items():
                out[engine] = out.get(engine, 0.0) + seconds * s.rate_hz / self.servers.get(engine, 1)
        return out

    @property
    def bottleneck(self) -> Optional[Tuple[str, float]]:
        util = self.engine_utilization()
        if not util:
            return None
        name = max(util, key=util.get)
        return name, util[name]

    @property
    def dram_utilization(self) -> Optional[float]:
        if not self.dram_supply_gb_per_s:
            return None
        return self.dram_demand_gb_per_s / self.dram_supply_gb_per_s

    # ---- latency ---------------------------------------------------------

    @property
    def reactive_chain_ms(self) -> float:
        """One call of each chain stage the profile runs, in sequence. A lower
        bound twice over: no contention, and chain stages that are gaps add
        nothing (``reactive_chain_complete`` says whether any are)."""
        by_key = {s.stage: s for s in self.served}
        return 1e3 * sum(by_key[k].t_service_s for k in REACTIVE_CHAIN if k in by_key)

    @property
    def reactive_chain_complete(self) -> bool:
        gap_stages = {stage for stage, _ in self.gaps}
        return not gap_stages.intersection(REACTIVE_CHAIN)

    def meets_deadline(self) -> Optional[bool]:
        """``None`` when a gap on the chain leaves the answer open, unless the
        priced part alone already misses."""
        if self.reactive_chain_ms > self.profile.deadline_ms:
            return False
        return True if self.reactive_chain_complete else None

    # ---- feasibility and confidence -------------------------------------

    def feasible(self) -> Optional[bool]:
        """False on any proven violation; None when gaps leave it open."""
        violated = (
            bool(self.stages_over())
            or any(u > 1.0 for u in self.engine_utilization().values())
            or (self.dram_utilization or 0.0) > 1.0
            or self.meets_deadline() is False
        )
        if violated:
            return False
        if self.dram_demand_gb_per_s > 0 and self.dram_supply_gb_per_s is None:
            return None  # unknown supply is not zero utilization
        return True if self.complete else None

    @property
    def confidence(self) -> Confidence:
        levels = [s.confidence for s in self.served] or [Confidence.UNKNOWN]
        if not self.complete:
            levels.append(Confidence.UNKNOWN)
        return weakest(*levels)

    @property
    def estimation_confidence(self) -> EstimationConfidence:
        """The weakest stage's confidence, and which stage set it."""
        if not self.complete:
            source = (f"{len(self.gaps)} stage(s) unpriced, "
                      f"{len(self.lower_bounds)} priced only as a lower bound")
        else:
            weakest_stage = max(self.served, key=lambda s: _LEVEL_ORDER.index(s.confidence))
            source = f"{weakest_stage.stage}: {weakest_stage.confidence_source}"
        return estimation_confidence(self.confidence, source)

    def to_dict(self) -> dict:
        util = self.engine_utilization()
        return {
            "profile": self.profile.id,
            "regime": self.profile.regime,
            "efficiency": self.table,
            "mapping": self.mapping,
            "complete": self.complete,
            "gaps": [{"stage": s, "reason": r} for s, r in self.gaps],
            "lower_bounds": list(self.lower_bounds),
            "stages": [s.to_dict() for s in self.services],
            "engines": [
                {"engine": n, "servers": self.servers[n], "utilization": util.get(n, 0.0)}
                for n in self.servers
            ],
            "memory": {
                "dram_demand_gb_per_s": self.dram_demand_gb_per_s,
                "dram_supply_gb_per_s": self.dram_supply_gb_per_s,
                "utilization": self.dram_utilization,
            },
            "summary": {
                "oversubscription": self.oversubscription,
                "stages_over": list(self.stages_over()),
                "reactive_chain_ms": self.reactive_chain_ms,
                "reactive_chain_complete": self.reactive_chain_complete,
                "deadline_ms": self.profile.deadline_ms,
                "meets_deadline": self.meets_deadline(),
                "feasible": self.feasible(),
                "confidence": self.confidence.value,
                "confidence_source": self.estimation_confidence.source,
            },
        }


def schedule(
    workload: PipelineWorkload,
    profile: MissionProfile,
    soc: Optional[SoCInstance],
    table: EfficiencyTable,
    kernels: Optional[KernelClassMap] = None,
    mapping: Union[str, Dict[str, Any]] = "greedy",
    sustained_fraction: float = SUSTAINED_DRAM_FRACTION,
    transfers: Optional[Dict[str, float]] = None,
) -> Schedule:
    """Run one profile on one SoC.

    ``mapping`` is ``"greedy"``, ``"ilp"`` (optimal, needs scipy) or an
    explicit dict: ``{stage: engine}``, or ``{stage: {class: engine}}`` to
    split a stage's precision classes across engines, with ``transfers``
    giving each split's intermediate bytes per call; a
    pooled table ignores it. A stage an explicit mapping leaves out is a gap.
    ``soc`` may be ``None`` for a pooled table, in which case DRAM supply is
    unknown.
    """
    if not 0 < sustained_fraction <= 1:
        raise ValueError(f"sustained_fraction must be in (0, 1], got {sustained_fraction}")
    demands = workload.demands(profile)
    dram_demand = sum(d.bytes_per_s for d in demands) / 1e9
    supply = None
    if soc is not None and soc.dram_peak_gb_per_s > 0:
        supply = soc.dram_peak_gb_per_s * sustained_fraction

    if table.kind == "pooled":
        services = tuple(pooled_service(d, table) for d in demands)
        return Schedule(profile, table.id, POOLED, services, {POOLED: 1}, dram_demand, supply)

    if soc is None or kernels is None:
        raise ValueError(f"table {table.id!r} is per-engine: it needs a composed SoC and kernel classes")
    kernels.check_against(workload.stages)
    engines: Dict[str, Engine] = engines_of(soc)
    servers = {name: e.servers for name, e in engines.items()}

    if mapping in ("greedy", "ilp"):
        if mapping == "ilp":
            from .ilp import ilp_mapping  # noqa: PLC0415 -- scipy is optional

            by_stage = ilp_mapping(demands, engines, kernels, table, supply or 0.0)
        else:
            by_stage = greedy_mapping(demands, engines, kernels, table, supply or 0.0)
        services = tuple(by_stage[d.stage.key] for d in demands)
        return Schedule(profile, table.id, mapping, services, servers, dram_demand, supply)

    if isinstance(mapping, str):
        raise ValueError(f"mapping must be 'greedy', 'ilp' or a {{stage: engine}} dict, got {mapping!r}")
    check_explicit(mapping, engines)
    transfers = transfers or {}
    out = []
    for d in demands:
        target = mapping.get(d.stage.key)
        if target is None:
            out.append(StageService(stage=d.stage.key, engine="", rate_hz=d.rate_hz,
                                    gap="not in the explicit mapping"))
            continue
        kernel = kernels.of(d.stage.key)
        if isinstance(target, dict):
            out.append(split_service(d, kernel, {c: engines[e] for c, e in target.items()},
                                     table, supply or 0.0, transfers.get(d.stage.key)))
        else:
            out.append(engine_service(d, kernel, engines[target], table, supply or 0.0))
    # A split's intermediate crosses DRAM: its traffic is demand like any other.
    dram_demand += sum(svc.transfer_bytes * svc.rate_hz for svc in out if svc.served) / 1e9
    return Schedule(profile, table.id, "explicit", tuple(out), servers, dram_demand, supply)


__all__ = ["Schedule", "schedule"]
