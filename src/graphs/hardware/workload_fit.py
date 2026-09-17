"""Fitting an autonomy mission onto a KPU die (graphs#268, #269 Phase 3 L0).

Takes a mission profile's stage demands (``graphs.core.pipeline_workload``)
and a die's engines (``graphs.hardware.kpu_engines``) and asks, stage by
stage: can this die run this work, and how much of a second does it take?

Three ways a stage can land, and they are not interchangeable:

* **Absorbed by a function core.** The die has a fixed-function tile whose
  ``function_id`` implements the stage, and the mission's configuration is
  inside the core's contract limits. Occupancy is work units demanded over
  work units the core delivers -- pixels per second for an ISP or an SGM
  core, frames per second for a Navion-class VIO core -- not ops per second,
  because a function core has no meaningful op rate.
* **Programmable.** The stage runs on pe_fabric or systolic tiles at the
  precision class its numerics demand. Class A is INT8-eligible, Class B has
  an FP16 floor, Class C an FP32/FP64 floor.
* **Infeasible.** No engine offers the class the stage needs. This is the
  hard case the annex insists on: a stage with an FP32 floor does not run
  slowly on an INT8 fabric, it does not run.

**Achieved-to-peak.** Peak silicon rates flatter every die. The companion
measurement annex measures 0.008% to 12% of peak with a 0.31% median, and
the argument document sizes on 2% (realistic mixed pipeline) and 5%
(optimistic). Both are reported; peak is available and labelled as the upper
bound it is.

A contract-limit violation is reported, never silently ignored: an SGM core
rated to 1920x1080 at 30 fps does not become a 40 fps core because a mission
wants one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from embodied_schemas import ComputeProduct, ProcessNodeEntry

from graphs.core.pipeline_workload import (
    CLASS_NAMES,
    MissionProfile,
    PipelineWorkload,
    StageDemand,
)

from .kpu_access import kpu_block_of
from .kpu_engines import EngineDescriptor, describe_engines

#: Which function core implements which pipeline stage. Each is a claim about
#: what the core computes, not a name match:
#:
#: * ``sgm`` -> ``stereo.sgm``: both are semi-global matching over a rectified
#:   stereo pair producing a disparity map.
#: * ``mono`` -> ``isp.raw_to_yuv``: the mono camera front-end is the raw
#:   Bayer to YUV conversion an ISP pipeline performs.
#: * ``vio`` -> ``vio.stereo_inertial``: a Navion-class core is a whole stereo
#:   visual-inertial pipeline, frontend and factor-graph backend, so it covers
#:   the visual front-end stage. It does *not* claim the windowed bundle
#:   adjustment stage (``ba``), which is a larger periodic solve.
STAGE_FUNCTION = {
    "sgm": "stereo.sgm",
    "mono": "isp.raw_to_yuv",
    "vio": "vio.stereo_inertial",
}

#: Achieved-to-peak ratios (argument document section 7).
REALISTIC_ACHIEVED_TO_PEAK = 0.02
OPTIMISTIC_ACHIEVED_TO_PEAK = 0.05

#: Number formats that serve each precision class. Class A is INT8-eligible
#: dense work; Class B needs an FP16 floor, so bf16 and fp16 both serve it;
#: Class C needs FP32 or FP64. LNS formats rank with the float they replace
#: (kpu_engines.supports_at_least), but no catalog stage is priced in LNS, so
#: they do not appear here.
CLASS_FORMATS = {
    "A": ("int8",),
    "B": ("fp16", "bf16"),
    "C": ("fp32", "fp64"),
}


@dataclass(frozen=True)
class ContractCheck:
    """Whether a mission's configuration fits a core's stated limits."""

    limit: str
    required: float
    allowed: float

    @property
    def ok(self) -> bool:
        return self.required <= self.allowed

    def __str__(self) -> str:
        return f"{self.limit} {self.required:g} vs {self.allowed:g} allowed"


@dataclass
class StageFit:
    """How one stage lands on one die."""

    demand: StageDemand
    execution: str  # "absorbed" | "programmable" | "infeasible"
    engines: Tuple[str, ...] = ()
    occupancy: float = 0.0
    #: Work units per second the stage needs and the core delivers (absorbed).
    units_required_per_s: Optional[float] = None
    units_available_per_s: Optional[float] = None
    work_unit: str = ""
    #: Classes the die cannot run at all.
    missing_classes: Tuple[str, ...] = ()
    violations: Tuple[ContractCheck, ...] = ()
    watts: Optional[float] = None
    note: str = ""

    @property
    def key(self) -> str:
        return self.demand.stage.key

    @property
    def runnable(self) -> bool:
        """The die has an engine for this stage's numerics."""
        return self.execution != "infeasible"

    @property
    def within_contract(self) -> bool:
        """The mission's configuration is inside the core's stated limits."""
        return not self.violations

    @property
    def feasible(self) -> bool:
        return self.runnable and self.within_contract


@dataclass
class DieFit:
    """A whole mission on a whole die."""

    die_id: str
    profile: MissionProfile
    achieved_to_peak: float
    stages: Tuple[StageFit, ...]
    dram_demand_gb_per_s: float
    dram_available_gb_per_s: float
    class_capability_gops: Dict[str, float]

    @property
    def oversubscription(self) -> float:
        """Seconds of die time demanded per second of mission, counting only
        the stages the die can actually run."""
        return sum(s.occupancy for s in self.stages if s.feasible)

    @property
    def unrunnable_stages(self) -> Tuple[StageFit, ...]:
        """Stages whose precision class this die has no engine for. Not slow:
        absent."""
        return tuple(s for s in self.stages if not s.runnable)

    @property
    def out_of_contract_stages(self) -> Tuple[StageFit, ...]:
        """Stages a function core would absorb, at a configuration past what
        the core is rated for. A different failure from the one above: the
        silicon exists, the mission asks more of it than it claims."""
        return tuple(s for s in self.stages if s.runnable and not s.within_contract)

    @property
    def absorbed_stages(self) -> Tuple[StageFit, ...]:
        return tuple(s for s in self.stages if s.execution == "absorbed")

    @property
    def dram_ratio(self) -> float:
        return (
            self.dram_demand_gb_per_s / self.dram_available_gb_per_s
            if self.dram_available_gb_per_s else float("inf")
        )

    @property
    def binding_constraint(self) -> str:
        """What stops this die first. Precision is checked before rate: a
        stage the die cannot run is not a throughput problem."""
        if self.unrunnable_stages:
            return "precision"
        if self.out_of_contract_stages:
            return "contract"
        if self.dram_ratio > 1.0 and self.dram_ratio > self.oversubscription:
            return "bandwidth"
        if self.oversubscription > 1.0:
            return "compute"
        return "none"

    @property
    def feasible(self) -> bool:
        return self.binding_constraint == "none"

    def worst_stages(self, n: int = 3) -> Tuple[StageFit, ...]:
        return tuple(sorted(self.stages, key=lambda s: -s.occupancy)[:n])

    def to_dict(self) -> dict:
        return {
            "die": self.die_id,
            "profile": self.profile.id,
            "regime": self.profile.regime,
            "achieved_to_peak": self.achieved_to_peak,
            "oversubscription": self.oversubscription,
            "dram_demand_gb_per_s": self.dram_demand_gb_per_s,
            "dram_available_gb_per_s": self.dram_available_gb_per_s,
            "binding_constraint": self.binding_constraint,
            "feasible": self.feasible,
            "unrunnable_stages": [s.key for s in self.unrunnable_stages],
            "out_of_contract_stages": [s.key for s in self.out_of_contract_stages],
            "absorbed_stages": [s.key for s in self.absorbed_stages],
            "class_capability_gops": dict(self.class_capability_gops),
        }


def class_capability_gops(engines: Sequence[EngineDescriptor]) -> Dict[str, float]:
    """Peak GOP/s the die offers each precision class.

    A class is served by the engines that run one of its formats; the fastest
    format wins per engine, because a tile runs one mode at a time. Fixed
    function tiles contribute nothing here -- their capability is their own
    work unit, not a general op rate.
    """
    out = {name: 0.0 for name in CLASS_NAMES}
    for engine in engines:
        if engine.engine_kind == "fixed_function":
            continue
        rates = {p.operand_format: p.ops_per_second_per_tile * engine.num_tiles
                 for p in engine.precisions}
        for cls, formats in CLASS_FORMATS.items():
            best = max((rates.get(f, 0.0) for f in formats), default=0.0)
            out[cls] += best / 1e9
    return out


def _pixels_per_s(sensor: Optional[Sequence[float]]) -> float:
    """Pixels per second from a ``[cameras, width, height, fps]`` entry."""
    if not sensor or len(sensor) < 4:
        return 0.0
    n, w, h, fps = sensor[0], sensor[1], sensor[2], sensor[3]
    return float(n) * float(w) * float(h) * float(fps)


def _core_contract(engine: EngineDescriptor, profile: MissionProfile,
                   stage_key: str) -> Tuple[float, str, List[ContractCheck]]:
    """(work units per second the mission needs, unit name, limit violations)."""
    limits = dict(engine.config_limits or {})
    checks: List[ContractCheck] = []
    sensors = profile.sensors

    if stage_key == "sgm":
        stereo = sensors.get("stereo") or []
        units = _pixels_per_s(stereo)
        if len(stereo) >= 5:
            n, w, h, fps, disparities = stereo[:5]
            for key, required in (("max_width", w), ("max_height", h),
                                  ("max_fps", fps), ("disparities", disparities)):
                allowed = limits.get(key)
                if allowed is not None and required > allowed:
                    checks.append(ContractCheck(key, float(required), float(allowed)))
        return units, "pixel", checks

    if stage_key == "mono":
        return _pixels_per_s(sensors.get("mono")), "pixel", checks

    if stage_key == "vio":
        vio = sensors.get("vio") or []
        if len(vio) >= 4:
            n, w, h, fps = vio[:4]
            units = float(n) * float(fps) / 2.0  # a stereo pair per pose
            for key, required in (("max_width", w), ("max_height", h)):
                allowed = limits.get(key)
                if allowed is not None and required > allowed:
                    checks.append(ContractCheck(key, float(required), float(allowed)))
            return units, "frame", checks
        return 0.0, "frame", checks

    return 0.0, engine.work_unit or "", checks


def fit_profile(
    workload: PipelineWorkload,
    profile: MissionProfile,
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
    achieved_to_peak: float = REALISTIC_ACHIEVED_TO_PEAK,
) -> DieFit:
    """Fit one mission profile onto one die."""
    engines = describe_engines(cp, node, nodes)
    by_function = {e.function_id: e for e in engines if e.function_id}
    capability = class_capability_gops(engines)
    effective = {c: gops * achieved_to_peak for c, gops in capability.items()}

    fits: List[StageFit] = []
    absorbed_bytes = 0.0
    for demand in workload.demands(profile):
        stage = demand.stage
        function_id = STAGE_FUNCTION.get(stage.key)
        engine = by_function.get(function_id) if function_id else None

        if engine is not None:
            units_needed, unit, violations = _core_contract(engine, profile, stage.key)
            available = engine.units_per_second or 0.0
            occupancy = units_needed / available if available else float("inf")
            watts = (
                (engine.pj_per_unit or 0.0) * min(units_needed, available) * 1e-12
                if available else None
            )
            if not violations:
                absorbed_bytes += demand.bytes_per_s
            fits.append(StageFit(
                demand=demand,
                execution="absorbed",
                engines=(engine.tile_type,),
                occupancy=occupancy,
                units_required_per_s=units_needed,
                units_available_per_s=available,
                work_unit=unit,
                violations=tuple(violations),
                watts=watts,
                note=(
                    f"{engine.tile_type} core, {units_needed:,.0f} of "
                    f"{available:,.0f} {unit}/s"
                    + ("; outside its contract limits" if violations else "")
                ),
            ))
            continue

        ops_by_class = demand.ops_per_s_by_class()
        missing = tuple(
            c for c in CLASS_NAMES if ops_by_class[c] > 0 and effective[c] <= 0
        )
        if missing:
            fits.append(StageFit(
                demand=demand, execution="infeasible", missing_classes=missing,
                note=(
                    f"needs class {'/'.join(missing)} "
                    f"({stage.precision_floor} floor); this die has no engine for it"
                ),
            ))
            continue

        seconds = sum(
            (ops_by_class[c] / 1e9) / effective[c] for c in CLASS_NAMES if ops_by_class[c] > 0
        )
        used = tuple(sorted({
            e.tile_type for e in engines
            if e.engine_kind != "fixed_function"
            and any(p.operand_format in sum(
                (CLASS_FORMATS[c] for c in CLASS_NAMES if ops_by_class[c] > 0), ()
            ) for p in e.precisions)
        }))
        fits.append(StageFit(
            demand=demand, execution="programmable", engines=used, occupancy=seconds,
        ))

    block = kpu_block_of(cp)
    total_bytes = sum(d.bytes_per_s for d in workload.demands(profile))
    return DieFit(
        die_id=cp.id,
        profile=profile,
        achieved_to_peak=achieved_to_peak,
        stages=tuple(fits),
        # A stream link keeps an absorbed stage's traffic on chip (graphs#268
        # E3 segments), so it does not cross the DRAM bus.
        dram_demand_gb_per_s=(total_bytes - absorbed_bytes) / 1e9,
        dram_available_gb_per_s=float(block.memory.memory_bandwidth_gbps or 0.0),
        class_capability_gops=capability,
    )


def fit_all(
    workload: PipelineWorkload,
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
    achieved_to_peak: float = REALISTIC_ACHIEVED_TO_PEAK,
) -> Tuple[DieFit, ...]:
    return tuple(
        fit_profile(workload, p, cp, node, nodes, achieved_to_peak)
        for p in workload.profiles
    )
