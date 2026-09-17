"""The autonomy pipeline as data: costed stages, mission profiles, demand.

A port of the Branes.ai Autonomy Workload model (`docs/workload-model/`,
2026-09-17) into a form the estimators can consume. The model stays the
reference implementation: ``tests/core/test_pipeline_workload.py`` runs it and
requires this module to reproduce it, so when the two disagree the model is
right.

The shape, from the Data Annex (2026-09-17):

* A **stage** is one costed algorithm step, priced *per call* at a stated
  sensor configuration -- ``ops_per_call`` and ``bytes_per_call`` -- and split
  across three precision classes. Class A is INT8-eligible dense convolution
  and GEMM, Class B has an FP16 floor (softmax, normalization, attention
  scores), Class C an FP32/FP64 floor (factor graphs, signed-distance fields,
  MPC and barrier-function QPs). A mixed stage is bound by its hardest
  component, not its dominant one.
* A **mission profile** is an explicit sensor suite plus the update rate it
  assigns each stage. Stage demand is unit cost times rate; nothing is
  back-solved from a published total.
* **Service time** is seconds to execute one call with the whole machine to
  itself: the call's ops apportioned across the three classes and divided by
  each class's effective throughput. **Occupancy** is service time times rate
  -- the fraction of a second of machine time the stage consumes, so 1.0 means
  the stage cannot sustain its own rate even alone. **Oversubscription** is
  the sum over stages: seconds of compute demanded per second of mission.

``EffectiveThroughput`` carries the annex's stated Class A / B / C rates
(2,000 / 300 / 15 GOP/s), which are assumptions from the companion
measurement corpus rather than measurements of this workload. Everything here
is arithmetic on declared data: no hardware model, no mapping. What a
particular die does with this demand belongs in the comparison, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

#: Precision-class split of a stage's ops: (Class A, Class B, Class C).
ClassSplit = Tuple[float, float, float]

CLASS_NAMES = ("A", "B", "C")


@dataclass(frozen=True)
class EffectiveThroughput:
    """Per-class effective throughput in GOP/s (Data Annex section 2).

    Stated assumptions from the companion measurement annex, anchored to the
    fastest batch-1 result in its 360-configuration corpus and generous on B
    and C. They set every latency and occupancy figure in the model, which is
    why the annex's sensitivity sheet triples B and C: doing so moves
    oversubscription from about 17x to about 6x and removes it from no
    profile that has it.
    """

    a: float = 2000.0
    b: float = 300.0
    c: float = 15.0

    def of(self, split: ClassSplit) -> float:
        """Seconds per GOP for a call with this class split."""
        return split[0] / self.a + split[1] / self.b + split[2] / self.c


DEFAULT_THROUGHPUT = EffectiveThroughput()


@dataclass(frozen=True)
class Provenance:
    """Where a figure came from, and when it arrived."""

    document: str
    section: str = ""
    dated: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(**{k: data.get(k, "") for k in ("document", "section", "dated", "note")})


@dataclass(frozen=True)
class Stage:
    """One costed pipeline stage, priced per call at a stated configuration."""

    key: str
    name: str
    pipeline_tier: str  # T1..T7; not "tier", which means power class elsewhere
    unit: str  # what one call is: per pixel, per solve, per inference...
    ops_per_call: float
    bytes_per_call: float
    class_split: ClassSplit
    config: Dict[str, str] = field(default_factory=dict)
    basis: str = ""

    def __post_init__(self) -> None:
        total = sum(self.class_split)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"stage {self.key!r}: class split {self.class_split} sums to {total}, not 1"
            )
        if self.ops_per_call <= 0:
            raise ValueError(f"stage {self.key!r}: ops_per_call must be positive")

    @property
    def op_per_byte(self) -> float:
        """Arithmetic intensity. The annex's central observation is that most
        of the pipeline sits below 15, where the memory system sets the rate
        however wide the multiplier array is."""
        return self.ops_per_call / self.bytes_per_call if self.bytes_per_call else float("inf")

    @property
    def precision_floor(self) -> str:
        """The hardest class the stage needs: C, else B, else A."""
        a, b, c = self.class_split
        if c > 0:
            return "C"
        return "B" if b > 0 else "A"

    def service_time_s(self, throughput: EffectiveThroughput = DEFAULT_THROUGHPUT) -> float:
        """Seconds for one call with the whole machine to itself."""
        return (self.ops_per_call / 1e9) * throughput.of(self.class_split)

    def to_dict(self) -> dict:
        out = {
            "key": self.key,
            "name": self.name,
            "pipeline_tier": self.pipeline_tier,
            "unit": self.unit,
            "ops_per_call": self.ops_per_call,
            "bytes_per_call": self.bytes_per_call,
            "class_split": {n: v for n, v in zip(CLASS_NAMES, self.class_split) if v},
        }
        if self.config:
            out["config"] = dict(self.config)
        if self.basis:
            out["basis"] = self.basis
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stage":
        split = data["class_split"]
        return cls(
            key=data["key"],
            name=data["name"],
            pipeline_tier=data["pipeline_tier"],
            unit=data.get("unit", ""),
            ops_per_call=float(data["ops_per_call"]),
            bytes_per_call=float(data["bytes_per_call"]),
            class_split=tuple(float(split.get(n, 0.0)) for n in CLASS_NAMES),  # type: ignore[arg-type]
            config=dict(data.get("config") or {}),
            basis=data.get("basis", ""),
        )


@dataclass(frozen=True)
class StageDemand:
    """What one stage demands in one mission: rate times unit cost."""

    stage: Stage
    rate_hz: float
    throughput: EffectiveThroughput = DEFAULT_THROUGHPUT

    @property
    def ops_per_s(self) -> float:
        return self.stage.ops_per_call * self.rate_hz

    @property
    def bytes_per_s(self) -> float:
        return self.stage.bytes_per_call * self.rate_hz

    @property
    def service_time_s(self) -> float:
        return self.stage.service_time_s(self.throughput)

    @property
    def occupancy(self) -> float:
        """Fraction of one second of machine time this stage consumes. At or
        above 1.0 the stage cannot sustain its own rate alone on the machine,
        which is an infeasibility rather than a degradation."""
        return self.service_time_s * self.rate_hz

    def ops_per_s_by_class(self) -> Dict[str, float]:
        return {n: self.ops_per_s * f for n, f in zip(CLASS_NAMES, self.stage.class_split)}


#: The sense-to-act critical path: one call of each stage on it (Data Annex
#: section 3, reactive chain latency).
REACTIVE_CHAIN = (
    "sgm", "mono", "lidar", "lio", "vio", "esdf", "sdfenc", "policy", "mpc", "cbf", "ctrl",
)


@dataclass
class MissionProfile:
    """One mission: a sensor configuration and the rate it runs each stage."""

    form_factor: str
    name: str
    power_budget_w: float
    deadline_ms: float
    rates_hz: Dict[str, float]
    #: Unit costs that differ from the stage's reference configuration, because
    #: the stage is parametric in the sensor suite: SGM at 1280x720 with 96
    #: disparities is not SGM at 1440x1080 with 128, and an MPC over 60 states
    #: is not one over 12. ``{stage_key: {"ops_per_call": x, "bytes_per_call": y}}``.
    stage_costs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: The mission's sensor suite and configuration knobs, as the model states
    #: them: ``stereo: [cameras, width, height, fps, disparities]``,
    #: ``mono``/``vio``: ``[cameras, width, height, fps]``, ``radar``:
    #: ``[units, hz, rx, samples, chirps]``, ``mpc_dims``: ``[nx, nu, horizon]``,
    #: plus scalar rates. A fixed-function core has contract limits -- an SGM
    #: core rated to 1920x1080 at 30 fps, a VIO core to 752x480 -- and they can
    #: only be checked against a mission whose configuration is stated.
    sensors: Dict[str, Any] = field(default_factory=dict)
    note: str = ""
    #: The operating regime this profile is, when it is one of the five in
    #: the companion argument document.
    regime: Optional[str] = None
    #: That document's published figures for the regime, for cross-checking.
    published: Dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        slug = f"{self.form_factor}_{self.name}".lower()
        return "".join(ch if ch.isalnum() else "_" for ch in slug).strip("_").replace("__", "_")

    def to_dict(self) -> dict:
        out = {
            "form_factor": self.form_factor,
            "name": self.name,
            "power_budget_w": self.power_budget_w,
            "deadline_ms": self.deadline_ms,
            "rates_hz": dict(self.rates_hz),
        }
        if self.stage_costs:
            out["stage_costs"] = {k: dict(v) for k, v in self.stage_costs.items()}
        if self.sensors:
            out["sensors"] = dict(self.sensors)
        if self.note:
            out["note"] = self.note
        if self.regime:
            out["regime"] = self.regime
        if self.published:
            out["published"] = dict(self.published)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MissionProfile":
        return cls(
            form_factor=data["form_factor"],
            name=data["name"],
            power_budget_w=float(data["power_budget_w"]),
            deadline_ms=float(data["deadline_ms"]),
            rates_hz={k: float(v) for k, v in data["rates_hz"].items()},
            stage_costs={
                k: {ck: float(cv) for ck, cv in v.items()}
                for k, v in (data.get("stage_costs") or {}).items()
            },
            sensors=dict(data.get("sensors") or {}),
            note=data.get("note", ""),
            regime=data.get("regime"),
            published=dict(data.get("published") or {}),
        )


@dataclass
class MissionSummary:
    """Aggregate demand of one mission profile. Pure arithmetic on the stage
    demands: no hardware, no mapping, no efficiency beyond the annex's stated
    per-class throughputs."""

    profile: MissionProfile
    demands: Tuple[StageDemand, ...]

    @property
    def ops_per_s(self) -> float:
        return sum(d.ops_per_s for d in self.demands)

    @property
    def tops(self) -> float:
        return self.ops_per_s / 1e12

    @property
    def bytes_per_s(self) -> float:
        return sum(d.bytes_per_s for d in self.demands)

    @property
    def gb_per_s(self) -> float:
        return self.bytes_per_s / 1e9

    def class_shares(self) -> Dict[str, float]:
        total = self.ops_per_s
        if total <= 0:
            return {n: 0.0 for n in CLASS_NAMES}
        out = {n: 0.0 for n in CLASS_NAMES}
        for d in self.demands:
            for n, v in d.ops_per_s_by_class().items():
                out[n] += v
        return {n: v / total for n, v in out.items()}

    def tier_shares(self) -> Dict[str, float]:
        total = self.ops_per_s
        out: Dict[str, float] = {}
        for d in self.demands:
            out[d.stage.pipeline_tier] = out.get(d.stage.pipeline_tier, 0.0) + d.ops_per_s
        return {t: v / total for t, v in sorted(out.items())} if total > 0 else {}

    @property
    def oversubscription(self) -> float:
        """Seconds of compute demanded per second of mission."""
        return sum(d.occupancy for d in self.demands)

    def stages_over(self) -> Tuple[str, ...]:
        """Stages that cannot sustain their own rate alone on the machine."""
        return tuple(d.stage.key for d in self.demands if d.occupancy >= 1.0)

    @property
    def reactive_chain_ms(self) -> float:
        """Sense-to-act critical path with the whole machine available: one
        call of each chain stage the profile runs. A lower bound -- it assumes
        no contention, no launch overhead and a warm cache."""
        by_key = {d.stage.key: d for d in self.demands}
        return 1e3 * sum(by_key[k].service_time_s for k in REACTIVE_CHAIN if k in by_key)

    def meets_deadline(self) -> bool:
        return self.reactive_chain_ms <= self.profile.deadline_ms

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.id,
            "form_factor": self.profile.form_factor,
            "name": self.profile.name,
            "regime": self.profile.regime,
            "tops": self.tops,
            "gb_per_s": self.gb_per_s,
            "class_shares": self.class_shares(),
            "oversubscription": self.oversubscription,
            "stages_over": list(self.stages_over()),
            "reactive_chain_ms": self.reactive_chain_ms,
            "deadline_ms": self.profile.deadline_ms,
            "power_budget_w": self.profile.power_budget_w,
        }


@dataclass
class PipelineWorkload:
    """The costed stages and the missions that instantiate them."""

    stages: Dict[str, Stage]
    profiles: Tuple[MissionProfile, ...]
    throughput: EffectiveThroughput = DEFAULT_THROUGHPUT
    provenance: Optional[Provenance] = None
    version: str = ""

    def demands(self, profile: MissionProfile) -> Tuple[StageDemand, ...]:
        """Stage demands of one profile, in stage declaration order. A rate of
        zero means the profile omits the stage, and the annex is explicit that
        an omission is stated rather than buried in a coefficient."""
        out = []
        for key, stage in self.stages.items():
            rate = profile.rates_hz.get(key, 0.0)
            if rate <= 0:
                continue
            override = profile.stage_costs.get(key)
            if override:
                stage = replace(
                    stage,
                    ops_per_call=override.get("ops_per_call", stage.ops_per_call),
                    bytes_per_call=override.get("bytes_per_call", stage.bytes_per_call),
                    config={**stage.config, "configured_by": profile.name},
                )
            out.append(StageDemand(stage, rate, self.throughput))
        return tuple(out)

    def summary(self, profile: MissionProfile) -> MissionSummary:
        return MissionSummary(profile, self.demands(profile))

    def summaries(self) -> Tuple[MissionSummary, ...]:
        return tuple(self.summary(p) for p in self.profiles)

    def profile(self, name: str) -> MissionProfile:
        """Look a profile up by id, by regime name, or by mission name."""
        wanted = name.lower()
        for p in self.profiles:
            if wanted in (p.id.lower(), (p.regime or "").lower(), p.name.lower()):
                return p
        raise KeyError(f"no profile {name!r}; have {[p.id for p in self.profiles]}")

    def regimes(self) -> Tuple[MissionProfile, ...]:
        return tuple(p for p in self.profiles if p.regime)

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {"version": self.version} if self.version else {}
        if self.provenance:
            out["provenance"] = self.provenance.to_dict()
        out["effective_throughput_gops"] = {
            "A": self.throughput.a, "B": self.throughput.b, "C": self.throughput.c,
        }
        out["stages"] = [s.to_dict() for s in self.stages.values()]
        out["profiles"] = [p.to_dict() for p in self.profiles]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineWorkload":
        eff = data.get("effective_throughput_gops") or {}
        stages = [Stage.from_dict(s) for s in data["stages"]]
        unknown = {
            k for p in data["profiles"] for k in p["rates_hz"]
        } - {s.key for s in stages}
        if unknown:
            raise ValueError(f"profiles name stages that are not costed: {sorted(unknown)}")
        return cls(
            stages={s.key: s for s in stages},
            profiles=tuple(MissionProfile.from_dict(p) for p in data["profiles"]),
            throughput=EffectiveThroughput(
                a=float(eff.get("A", 2000.0)),
                b=float(eff.get("B", 300.0)),
                c=float(eff.get("C", 15.0)),
            ),
            provenance=Provenance.from_dict(data["provenance"]) if data.get("provenance") else None,
            version=data.get("version", ""),
        )

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, width=100))

    @classmethod
    def load(cls, path: Path) -> "PipelineWorkload":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


#: The catalog copy, generated from the reference model.
DEFAULT_WORKLOAD_PATH = (
    Path(__file__).resolve().parents[3] / "workloads" / "pipelines" / "autonomy"
    / "branes_7tier_v1.yaml"
)


def load_autonomy_workload(path: Optional[Path] = None) -> PipelineWorkload:
    """The Branes.ai 7-tier autonomy workload."""
    return PipelineWorkload.load(path or DEFAULT_WORKLOAD_PATH)
