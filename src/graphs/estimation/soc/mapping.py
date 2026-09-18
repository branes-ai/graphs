"""Engines, stage service times and stage-to-engine mapping (graphs#269 PR 3.2).

**Engines.** Every composed block with compute and a clock is an engine. A
stage runs on one *server* of it and owns that server while it runs -- the
annex's assumption, and a lower bound until Phase 5 adds contention:

* a GPU or a KPU is one server: a stage launches across all its SMs, or
  runs as a wavefront across the whole tile fabric;
* anything else is ``count x units`` servers -- a CPU core, one DLA -- and a
  stage runs on one of them (the parent plan's "CPU clusters are N parallel
  servers"). A single-threaded view of CPU stages; multithreading is Phase 5.

**Service time** of one call on one server, per engine (``per_engine``
tables)::

    t_compute = sum over classes k of  ops * share_k / (peak(fmt_k) * eff(kernel, kind, fmt_k))
    t_memory  = bytes_per_call / (DRAM peak * sustained fraction)
    t_service = max(t_compute, t_memory)

``fmt_k`` is the lowest format the engine has at or above class ``k``'s floor
(``execution_format``). A class the engine cannot run makes the stage
infeasible there; an efficiency the table does not know makes it a gap.
Neither gets a number.

With a ``pooled`` table (``annex_v1``) there is one virtual engine, the whole
SoC, and the service time is the annex's own: ops over per-class effective
throughput, no memory term, so the annex reproduces exactly.

**Mapping** is one engine per stage (split mappings are Phase 5), either
explicit -- a file naming an engine per stage -- or greedy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, model_validator

from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.core.pipeline_workload import CLASS_NAMES, Stage, StageDemand
from graphs.hardware.soc import BlockInstance, Confidence, EngineKind, SoCInstance

from .efficiency import (
    PRECISION_ORDER,
    EfficiencyTable,
    KernelClass,
    KernelClassMap,
    execution_format,
)

#: Fraction of peak DRAM bandwidth sustained under the pipeline's scattered
#: access (parent plan 5.6, "per the PDF").
SUSTAINED_DRAM_FRACTION = 0.65

#: The pooled table's single virtual engine.
POOLED = "pooled"

#: Workload unit costs are structural estimates (parent plan section 8).
WORKLOAD_CONFIDENCE = Confidence.THEORETICAL

_ORDER = [Confidence.CALIBRATED, Confidence.INTERPOLATED, Confidence.THEORETICAL, Confidence.UNKNOWN]


def weakest(*levels: Confidence) -> Confidence:
    return max(levels, key=_ORDER.index)


def estimation_confidence(level: Confidence, source: str) -> EstimationConfidence:
    """The repo-wide estimate descriptor (``graphs.core``) for a level carried
    as the shared ``Confidence`` enum; the two enums have the same values."""
    return {
        Confidence.CALIBRATED: EstimationConfidence.calibrated,
        Confidence.INTERPOLATED: EstimationConfidence.interpolated,
        Confidence.THEORETICAL: EstimationConfidence.theoretical,
    }.get(level, lambda source: EstimationConfidence(level=ConfidenceLevel.UNKNOWN,
                                                      source=source))(source=source)


@dataclass(frozen=True)
class Engine:
    """A schedulable view of one composed block."""

    block: BlockInstance
    servers: int

    @property
    def name(self) -> str:
        return self.block.name

    @property
    def kind(self) -> EngineKind:
        return self.block.engine_kind

    def server_peak_ops_per_s(self, fmt: str) -> float:
        return self.block.peak_ops_per_s(fmt) / self.servers

    @property
    def formats(self) -> Dict[str, float]:
        """Per-server dense peak ops/s by format, positive entries only."""
        out = {f: self.server_peak_ops_per_s(f) for f in PRECISION_ORDER}
        return {f: v for f, v in out.items() if v > 0}


def engines_of(soc: SoCInstance) -> Dict[str, Engine]:
    """The blocks a stage can run on, keyed by instance name."""
    out: Dict[str, Engine] = {}
    for block in soc.blocks:
        compute = block.template.compute
        if compute is None or block.clock_ghz is None or not compute.ops_per_clock:
            continue
        servers = (1 if block.engine_kind in (EngineKind.GPU, EngineKind.KPU)
                   else block.count * compute.units)
        out[block.name] = Engine(block, servers)
    return out


@dataclass(frozen=True)
class StageService:
    """One call of one stage on one engine: a time, or the reason there is none."""

    stage: str
    engine: str
    rate_hz: float
    t_compute_s: Optional[float] = None
    t_memory_s: Optional[float] = None
    formats: Dict[str, str] = field(default_factory=dict)
    gap: Optional[str] = None
    confidence: Confidence = Confidence.UNKNOWN
    #: What set ``confidence``: the weakest input.
    confidence_source: str = ""
    #: A split stage: the engine each precision class runs on. Empty for a
    #: stage on one engine (``engine`` holds it).
    class_engines: Dict[str, str] = field(default_factory=dict)
    #: Compute seconds per engine, for utilization. Empty means all of
    #: ``t_compute_s`` on ``engine``.
    parts: Dict[str, float] = field(default_factory=dict)
    #: Seconds moving the intermediate between a split's engines via DRAM.
    t_transfer_s: float = 0.0
    #: Extra DRAM bytes per call the split's transfer adds.
    transfer_bytes: float = 0.0
    #: True when a figure is missing and the time is only a floor -- a split
    #: whose transfer size nothing states.
    lower_bound: bool = False

    @property
    def served(self) -> bool:
        return self.gap is None

    @property
    def t_service_s(self) -> Optional[float]:
        if not self.served:
            return None
        return max(self.t_compute_s + self.t_transfer_s, self.t_memory_s or 0.0)

    @property
    def estimation_confidence(self) -> EstimationConfidence:
        return estimation_confidence(self.confidence, self.confidence_source or (self.gap or ""))

    @property
    def bound(self) -> Optional[str]:
        if not self.served:
            return None
        return "memory" if (self.t_memory_s or 0.0) > self.t_compute_s + self.t_transfer_s else "compute"

    @property
    def engine_seconds(self) -> Dict[str, float]:
        """Seconds of each engine one call occupies (compute parts; the
        stage-level memory stall is not attributed, so utilization is a
        floor for memory-bound splits, as for any stage)."""
        if not self.served:
            return {}
        return dict(self.parts) if self.parts else {self.engine: self.t_service_s}

    @property
    def occupancy(self) -> Optional[float]:
        """Service time over the stage's period: the annex's constraint ratio.
        At or above 1 the stage cannot keep up with its own rate."""
        t = self.t_service_s
        return None if t is None else t * self.rate_hz

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "engine": self.engine,
            "rate_hz": self.rate_hz,
            "t_service_ms": None if self.t_service_s is None else 1e3 * self.t_service_s,
            "budget_ms": 1e3 / self.rate_hz,
            "ratio": self.occupancy,
            "bound": self.bound,
            "formats": dict(self.formats),
            "gap": self.gap,
            "confidence": self.confidence.value,
            "confidence_source": self.confidence_source or self.gap or "",
            "class_engines": dict(self.class_engines),
            "t_transfer_ms": 1e3 * self.t_transfer_s,
            "lower_bound": self.lower_bound,
        }


def pooled_service(demand: StageDemand, table: EfficiencyTable) -> StageService:
    """The annex's service time: the whole SoC as one machine."""
    throughput = table.pooled.as_effective_throughput()
    return StageService(
        stage=demand.stage.key,
        engine=POOLED,
        rate_hz=demand.rate_hz,
        t_compute_s=demand.stage.service_time_s(throughput),
        confidence=weakest(table.pooled.confidence, WORKLOAD_CONFIDENCE),
        confidence_source=f"{table.id} pooled class throughput; workload unit costs",
    )


def engine_service(
    demand: StageDemand,
    kernel: KernelClass,
    engine: Engine,
    table: EfficiencyTable,
    dram_sustained_gb_per_s: float,
) -> StageService:
    """One call of a stage on one server of ``engine``, or why it cannot be priced."""
    stage: Stage = demand.stage
    base = dict(stage=stage.key, engine=engine.name, rate_hz=demand.rate_hz)
    formats: Dict[str, str] = {}
    t_compute = 0.0
    confidence = WORKLOAD_CONFIDENCE
    for cls, share in zip(CLASS_NAMES, stage.class_split):
        if share <= 0:
            continue
        fmt = execution_format(cls, engine.formats)
        if fmt is None:
            return StageService(**base, gap=f"{engine.kind.value} cannot run Class {cls}")
        formats[cls] = fmt
        entry = table.lookup(kernel, engine.kind, fmt)
        if entry is None or not entry.known:
            return StageService(
                **base, formats=formats,
                gap=f"no {table.id} efficiency for {kernel.value}/{engine.kind.value}/{fmt}",
            )
        t_compute += stage.ops_per_call * share / (
            engine.server_peak_ops_per_s(fmt) * entry.compute_eff
        )
        confidence = weakest(confidence, entry.confidence)
    sources = [f"{table.id} efficiency for {kernel.value}/{engine.kind.value}", "workload unit costs"]
    if not engine.block.clock_is_reference:
        # The peak uses a clock no stated relation carries to this node, so
        # the service time rests on a figure that does not apply here. (A
        # block's silicon-line confidence is about its area, not its speed,
        # and does not enter.)
        confidence = Confidence.UNKNOWN
        sources.insert(0, f"{engine.name} clock is provisional at {engine.block.clock_basis}")
    t_memory = None
    if dram_sustained_gb_per_s > 0:
        t_memory = stage.bytes_per_call / (dram_sustained_gb_per_s * 1e9)
    return StageService(
        **base, t_compute_s=t_compute, t_memory_s=t_memory,
        formats=formats, confidence=confidence, confidence_source="; ".join(sources),
    )


def split_service(
    demand: StageDemand,
    kernel: KernelClass,
    class_engines: Mapping[str, Engine],
    table: EfficiencyTable,
    dram_sustained_gb_per_s: float,
    transfer_bytes: Optional[float] = None,
) -> StageService:
    """One call of a stage whose precision classes run on different engines.

    The parts run in sequence (the trunk feeds the head), each at its own
    engine's peak and efficiency, and the intermediate crosses DRAM between
    them -- written once, read once: ``2 x transfer_bytes`` of traffic. The
    workload does not state intermediate sizes, so the mapping must; without
    one the service is a **lower bound** (transfer unpriced), flagged, and the
    schedule it is in is incomplete.
    """
    stage: Stage = demand.stage
    label = "+".join(dict.fromkeys(e.name for e in class_engines.values()))
    base = dict(stage=stage.key, engine=label, rate_hz=demand.rate_hz,
                class_engines={c: e.name for c, e in class_engines.items()})
    needed = [c for c, share in zip(CLASS_NAMES, stage.class_split) if share > 0]
    if sorted(class_engines) != sorted(needed):
        return StageService(**base, gap=f"split must place exactly classes {needed}, "
                                        f"got {sorted(class_engines)}")
    parts: Dict[str, float] = {}
    formats: Dict[str, str] = {}
    confidence = WORKLOAD_CONFIDENCE
    for cls, share in zip(CLASS_NAMES, stage.class_split):
        if share <= 0:
            continue
        engine = class_engines[cls]
        fmt = execution_format(cls, engine.formats)
        if fmt is None:
            return StageService(**base, gap=f"{engine.kind.value} cannot run Class {cls}")
        formats[cls] = fmt
        entry = table.lookup(kernel, engine.kind, fmt)
        if entry is None or not entry.known:
            return StageService(**base, formats=formats,
                                gap=f"no {table.id} efficiency for {kernel.value}/{engine.kind.value}/{fmt}")
        seconds = stage.ops_per_call * share / (engine.server_peak_ops_per_s(fmt) * entry.compute_eff)
        parts[engine.name] = parts.get(engine.name, 0.0) + seconds
        confidence = weakest(confidence, entry.confidence)
    sources = [f"{table.id} efficiencies per class", "workload unit costs"]
    for engine in {e.name: e for e in class_engines.values()}.values():
        if not engine.block.clock_is_reference:
            confidence = Confidence.UNKNOWN
            sources.insert(0, f"{engine.name} clock is provisional at {engine.block.clock_basis}")
    crosses = len(parts) > 1
    lower_bound = crosses and transfer_bytes is None
    t_transfer, moved = 0.0, 0.0
    if crosses and transfer_bytes is not None:
        moved = 2.0 * transfer_bytes
        if dram_sustained_gb_per_s > 0:
            t_transfer = moved / (dram_sustained_gb_per_s * 1e9)
        else:
            lower_bound = True  # bytes known, bandwidth not: the time is still a floor
    if lower_bound:
        confidence = Confidence.UNKNOWN
        sources.insert(0, "split transfer unpriced: the mapping states no intermediate size"
                       if transfer_bytes is None else "split transfer unpriced: no DRAM supply")
    t_memory = None
    if dram_sustained_gb_per_s > 0:
        t_memory = (stage.bytes_per_call + moved) / (dram_sustained_gb_per_s * 1e9)
    return StageService(
        **base, t_compute_s=sum(parts.values()), t_memory_s=t_memory, formats=formats,
        confidence=confidence, confidence_source="; ".join(sources), parts=parts,
        t_transfer_s=t_transfer, transfer_bytes=moved, lower_bound=lower_bound,
    )


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class StageAssignment(BaseModel):
    """A stage on one engine, or split by precision class across several."""

    engine: Optional[str] = None
    #: A split: precision class (A / B / C) -> engine.
    engines: Optional[Dict[str, str]] = None
    #: Bytes of the intermediate a split hands between its engines per call,
    #: with where the figure comes from. Without it the split is a lower bound.
    transfer_bytes: Optional[float] = Field(None, gt=0)
    transfer_source: str = ""
    reason: str = Field(..., min_length=10)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _one_form(self) -> "StageAssignment":
        if (self.engine is None) == (self.engines is None):
            raise ValueError("give exactly one of engine or engines")
        if self.engines is not None and not set(self.engines) <= {"A", "B", "C"}:
            raise ValueError(f"split classes must be A, B or C, got {sorted(self.engines)}")
        if self.transfer_bytes is not None and self.engines is None:
            raise ValueError("transfer_bytes only qualifies a split (engines)")
        if self.transfer_bytes is not None and len(self.transfer_source) < 10:
            raise ValueError("transfer_bytes needs a transfer_source saying where it comes from")
        return self

    @property
    def target(self):
        return self.engine if self.engine is not None else dict(self.engines)


class MappingFile(BaseModel):
    """An explicit stage-to-engine mapping for one design and workload."""

    design: str
    workload: str
    stages: Dict[str, StageAssignment]
    notes: str = ""

    model_config = {"extra": "forbid"}

    def assignments(self) -> Dict[str, Any]:
        """stage -> engine name, or {class: engine} for a split."""
        return {k: v.target for k, v in self.stages.items()}

    def transfers(self) -> Dict[str, float]:
        return {k: v.transfer_bytes for k, v in self.stages.items() if v.transfer_bytes is not None}


DEFAULT_MAPPING_DIR = Path(__file__).resolve().parents[4] / "soc_designs" / "mappings"


def load_mapping(path: Path) -> MappingFile:
    return MappingFile.model_validate(yaml.safe_load(Path(path).read_text()))


def find_mapping(design: str, workload: str, root: Optional[Path] = None) -> Optional[MappingFile]:
    """The shipped explicit mapping for this design and workload, if any."""
    file = (Path(root) if root else DEFAULT_MAPPING_DIR) / f"{design}__{workload}.yaml"
    return load_mapping(file) if file.exists() else None


def check_explicit(assignments: Mapping[str, Any], engines: Mapping[str, Engine]) -> None:
    """Every named engine must exist in the composed design."""
    named = set()
    for target in assignments.values():
        named |= set(target.values()) if isinstance(target, dict) else {target}
    bad = sorted(e for e in named if e not in engines)
    if bad:
        raise KeyError(f"mapping names engines the design does not have: {bad}; have {sorted(engines)}")


def greedy_mapping(
    demands: Tuple[StageDemand, ...],
    engines: Mapping[str, Engine],
    kernels: KernelClassMap,
    table: EfficiencyTable,
    dram_sustained_gb_per_s: float,
) -> Dict[str, StageService]:
    """Heaviest stage first, each to the priced engine it loads least.

    The cost is the engine's utilization after adding the stage (occupancy
    over servers), ties broken by the shorter service time -- load balancing,
    not energy; PR 3.3's power model can replace the cost. A stage no engine
    can price stays unmapped, with every engine's reason.
    """
    load = {name: 0.0 for name in engines}
    out: Dict[str, StageService] = {}
    for demand in sorted(demands, key=lambda d: -d.ops_per_s):
        kernel = kernels.of(demand.stage.key)
        options: List[Tuple[float, float, StageService]] = []
        reasons: List[str] = []
        for name, engine in engines.items():
            svc = engine_service(demand, kernel, engine, table, dram_sustained_gb_per_s)
            if not svc.served:
                reasons.append(svc.gap)
                continue
            options.append((load[name] + svc.occupancy / engine.servers, svc.t_service_s, svc))
        if not options:
            out[demand.stage.key] = StageService(
                stage=demand.stage.key, engine="", rate_hz=demand.rate_hz,
                gap="unmapped: " + "; ".join(reasons or ["no engines"]),
            )
            continue
        _, _, best = min(options, key=lambda o: (o[0], o[1]))
        load[best.engine] += best.occupancy / engines[best.engine].servers
        out[demand.stage.key] = best
    return out


__all__ = [
    "POOLED",
    "SUSTAINED_DRAM_FRACTION",
    "Engine",
    "MappingFile",
    "StageAssignment",
    "StageService",
    "check_explicit",
    "engine_service",
    "split_service",
    "engines_of",
    "find_mapping",
    "greedy_mapping",
    "load_mapping",
    "pooled_service",
    "estimation_confidence",
    "weakest",
]
