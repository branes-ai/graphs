"""SoCAnalyzer: one design, one node, one profile, end to end (graphs#269 PR 3.4).

Composes the design at the node (Phase 2), schedules the profile on it with an
efficiency table and a mapping (3.1, 3.2), rolls up power (3.3), runs the SoC
checks, and returns a ``SoCAnalysisResult`` -- JSON-first, for the
Embodied-AI-Architect orchestrator (parent plan 5.8).

**Confidence propagates as the weakest input, and the result says which input
set it** (parent plan section 8). An incomplete input -- unanchored silicon,
an unpriced stage, a power term with gaps -- makes the result UNKNOWN in that
dimension; ``confidence_summary.limited_by`` names each one, so a consumer
can tell "estimated from theory" from "not estimated at all".

This bypasses ``UnifiedAnalyzer`` deliberately: there is no PyTorch model,
only a costed pipeline and silicon to price.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from embodied_schemas import load_process_nodes

from graphs.core.confidence import EstimationConfidence
from graphs.core.pipeline_workload import (
    MissionProfile,
    PipelineWorkload,
    Stage,
    load_autonomy_workload,
)
from graphs.hardware.soc import (
    Confidence,
    SoCDesign,
    SoCInstance,
    compose_soc,
    load_designs,
    load_ip_library,
)
from graphs.hardware.soc.validators import validate_soc

from .efficiency import EfficiencyTable, KernelClassMap, load_efficiency_tables, load_kernel_classes
from .mapping import (
    SUSTAINED_DRAM_FRACTION,
    engines_of,
    estimation_confidence,
    find_mapping,
    load_mapping,
    weakest,
)
from .power import PowerReport, roll_up
from .schedule import Schedule, schedule

MappingChoice = Union[str, Dict[str, str], Path]


@dataclass(frozen=True)
class SoCAnalysisResult:
    """Everything the L0 analyzer knows about one design on one profile."""

    soc: SoCInstance
    schedule: Schedule
    power: PowerReport
    findings: Tuple
    mapping_source: str
    #: The profile's own stage costs (a profile may override a stage's).
    stages: Mapping[str, Stage]

    @property
    def profile(self) -> MissionProfile:
        return self.schedule.profile

    def limited_by(self) -> List[str]:
        """Every input that keeps the result from being complete."""
        out = []
        if not self.soc.complete:
            out.append(f"die: {len(self.soc.gaps)} unanchored silicon line(s); area and "
                       "leakage are lower bounds")
        if not self.schedule.complete:
            out.append(f"schedule: {len(self.schedule.gaps)} stage(s) unpriced "
                       f"({', '.join(s for s, _ in self.schedule.gaps)})")
        for term in self.power.terms:
            if term.gaps:
                out.append(f"power.{term.name}: {'; '.join(term.gaps[:3])}"
                           + (f" (+{len(term.gaps) - 3} more)" if len(term.gaps) > 3 else ""))
            elif term.floor_only:
                out.append(f"power.{term.name}: ALU-only floor, architectural overhead unpriced")
        if self.soc.off_reference_clocks:
            out.append(f"clocks: {', '.join(self.soc.off_reference_clocks)} are off their "
                       "reference node with no stated speed relation; peaks are provisional")
        return out

    @property
    def confidence(self) -> Confidence:
        levels = [self.schedule.confidence, self.soc.confidence]
        if not self.power.complete:
            levels.append(Confidence.UNKNOWN)
        return weakest(*levels)

    @property
    def estimation_confidence(self) -> EstimationConfidence:
        """The repo-wide estimate descriptor; its source is the first input
        that limits the result, or the schedule's own when none does."""
        reasons = self.limited_by()
        source = reasons[0] if reasons else self.schedule.estimation_confidence.source
        return estimation_confidence(self.confidence, source)

    @property
    def complete(self) -> bool:
        return self.soc.complete and self.schedule.complete and self.power.complete

    def feasible(self) -> Optional[bool]:
        """False on any proven violation (a stage over, an engine or DRAM over
        1, the reactive chain past its deadline, a power lower bound past the
        budget); True only when complete and nothing fails; else None."""
        verdicts = (self.schedule.feasible(), self.power.within_budget())
        if False in verdicts:
            return False
        return True if all(v is True for v in verdicts) else None

    def to_dict(self) -> dict:
        soc, sched, power = self.soc, self.schedule, self.power
        util = sched.engine_utilization()
        block_power = {b.name: b for b in power.blocks}
        engines = engines_of(soc)
        stage_rows = []
        for svc in sched.services:
            stage = self.stages[svc.stage]
            row = svc.to_dict()
            row["pipeline_tier"] = stage.pipeline_tier
            row["dram_gb_per_s"] = stage.bytes_per_call * svc.rate_hz / 1e9
            stage_rows.append(row)
        return {
            "design": soc.design.id,
            "node": soc.node.id,
            "profile": self.profile.id,
            "regime": self.profile.regime,
            "efficiency": sched.table,
            "mapping": sched.mapping,
            "mapping_source": self.mapping_source,
            "complete": self.complete,
            "confidence_summary": {
                "level": self.confidence.value,
                "score": self.estimation_confidence.score,
                "limited_by": self.limited_by(),
            },
            "die": {
                "area_mm2": soc.die_area_mm2,
                "area_is_lower_bound": not soc.complete,
                "transistors_b": soc.transistors_billion,
                "by_block": [
                    {"block": b.name, "ip": b.template.id, "count": b.count,
                     # None, not 0, when no line of the block is anchored.
                     "area_mm2": b.area_mm2 if any(l.anchored for l in b.lines) else None,
                     "area_complete": b.complete, "clock_ghz": b.clock_ghz}
                    for b in soc.blocks
                ],
                "by_circuit_class": soc.area_by_class(),
                "findings": [
                    {"validator": f.validator, "severity": f.severity.value,
                     "category": f.category.value, "block": f.block, "message": f.message}
                    for f in self.findings
                ],
            },
            "peak": {
                name: {fmt: e.block.peak_ops_per_s(fmt) for fmt in e.block.template.compute.ops_per_clock}
                for name, e in engines.items()
            },
            "stages": stage_rows,
            "engines": [
                {"engine": name, "servers": sched.servers.get(name, 1),
                 "utilization": util.get(name, 0.0),
                 "dynamic_w": block_power[name].dynamic_w if name in block_power else None,
                 "leakage_w": block_power[name].leakage_w if name in block_power else None,
                 "gated": block_power[name].gated if name in block_power else False}
                for name in sched.servers
            ],
            "memory": {
                "dram_demand_gb_per_s": sched.dram_demand_gb_per_s,
                "dram_supply_gb_per_s": sched.dram_supply_gb_per_s,
                "sustained_fraction": (sched.dram_supply_gb_per_s / soc.dram_peak_gb_per_s
                                       if sched.dram_supply_gb_per_s and soc.dram_peak_gb_per_s else None),
                "headroom_gb_per_s": (sched.dram_supply_gb_per_s - sched.dram_demand_gb_per_s
                                 if sched.dram_supply_gb_per_s is not None else None),
            },
            "power": power.to_dict(),
            "summary": {
                "feasible": self.feasible(),
                "oversubscription": sched.oversubscription,
                "stages_over": list(sched.stages_over()),
                "e2e_latency_ms": sched.reactive_chain_ms,
                "e2e_latency_complete": sched.reactive_chain_complete,
                "deadline_ms": self.profile.deadline_ms,
                "useful_tops_per_w": power.useful_tops_per_w,
            },
        }


class SoCAnalyzer:
    """Loads the catalogs once and analyzes designs against profiles.

    Every catalog can be injected, so a study or a test can analyze a design
    that is not in the shipped library.
    """

    def __init__(
        self,
        workload: Optional[PipelineWorkload] = None,
        designs: Optional[Mapping[str, SoCDesign]] = None,
        library=None,
        nodes=None,
        tables: Optional[Mapping[str, EfficiencyTable]] = None,
        kernels: Optional[KernelClassMap] = None,
    ):
        self.workload = workload or load_autonomy_workload()
        self.designs = designs if designs is not None else load_designs()
        self.library = library if library is not None else load_ip_library()
        self.nodes = nodes if nodes is not None else load_process_nodes()
        self.tables = tables if tables is not None else load_efficiency_tables()
        self.kernels = kernels if kernels is not None else load_kernel_classes()

    def _mapping(self, design: SoCDesign, choice: MappingChoice) -> Tuple[Union[str, Dict[str, str]], str]:
        """Resolve a mapping choice to what ``schedule`` takes, and say where it came from."""
        if isinstance(choice, dict):
            return choice, "caller"
        if isinstance(choice, Path) or (isinstance(choice, str) and choice.endswith(".yaml")):
            mapping = load_mapping(Path(choice))
            if mapping.design != design.id or mapping.workload != self.workload.version:
                raise ValueError(
                    f"mapping {choice} is for {mapping.design}/{mapping.workload}, "
                    f"not {design.id}/{self.workload.version}")
            return mapping.assignments(), str(choice)
        shipped = find_mapping(design.id, self.workload.version)
        if choice == "explicit":
            if shipped is None:
                raise ValueError(f"no explicit mapping ships for {design.id} on {self.workload.version}")
            return shipped.assignments(), f"soc_designs/mappings/{design.id}__{self.workload.version}.yaml"
        if choice == "auto":
            if shipped is not None:
                return shipped.assignments(), f"soc_designs/mappings/{design.id}__{self.workload.version}.yaml"
            return "greedy", "greedy"
        if choice == "greedy":
            return "greedy", "greedy"
        raise ValueError(f"mapping must be auto, explicit, greedy, a .yaml file or a dict; got {choice!r}")

    def analyze(
        self,
        design: Union[str, SoCDesign],
        profile: Union[str, MissionProfile],
        node: Optional[str] = None,
        efficiency: str = "annex_v1",
        mapping: MappingChoice = "auto",
        gate_idle: bool = False,
        sustained_fraction: float = SUSTAINED_DRAM_FRACTION,
    ) -> SoCAnalysisResult:
        if isinstance(design, SoCDesign):
            spec = design  # e.g. a sweep variant, not in the catalog
        elif design in self.designs:
            spec = self.designs[design]
        else:
            raise KeyError(f"no SoC design {design!r}; have {sorted(self.designs)}")
        if efficiency not in self.tables:
            raise KeyError(f"no efficiency table {efficiency!r}; have {sorted(self.tables)}")
        prof = profile if isinstance(profile, MissionProfile) else self.workload.profile(profile)
        soc = compose_soc(spec, self.library, self.nodes, node)
        table = self.tables[efficiency]
        resolved, source = self._mapping(spec, mapping)
        if table.kind == "pooled":
            source = "pooled (no mapping)"
        sched = schedule(self.workload, prof, soc, table, self.kernels, resolved, sustained_fraction)
        return SoCAnalysisResult(
            soc=soc,
            schedule=sched,
            power=roll_up(sched, soc, self.workload, gate_idle),
            findings=tuple(validate_soc(soc)),
            mapping_source=source,
            stages={d.stage.key: d.stage for d in self.workload.demands(prof)},
        )

    def analyze_profiles(
        self, design: str, profiles: Optional[List[str]] = None, **kwargs
    ) -> List[SoCAnalysisResult]:
        """One result per profile; all profiles when none are named."""
        chosen = [self.workload.profile(p) for p in profiles] if profiles else list(self.workload.profiles)
        return [self.analyze(design, p, **kwargs) for p in chosen]


__all__ = ["SoCAnalysisResult", "SoCAnalyzer"]
