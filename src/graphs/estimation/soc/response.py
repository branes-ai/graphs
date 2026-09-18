"""Response-time schedulability of a scheduled profile (graphs#269 Phase 5).

The schedule's service time assumes a stage owns its engine -- the annex's
**lower bound** on response time. This computes an **upper bound** with a
real scheduling policy, so feasibility can be proven in both directions:

* a stage whose lower bound already misses its deadline, or an engine over
  100% utilization, is **proven infeasible** (the schedule says so);
* a task set whose upper-bound response times all meet their deadlines is
  **proven schedulable** under the policy;
* anything between is **open** -- never reported as either.

Model. Each stage is a periodic task on its engine: period and deadline
``1 / rate`` (the annex's constraint ratio), execution time its per-engine
seconds (a split stage is one task per engine it uses, and must fit its
parts plus the transfer within one period). A multi-server engine (CPU
cores, DLA instances) is partitioned first-fit-decreasing by utilization;
a failed partition is a failed heuristic, not proof of infeasibility, so it
leaves the verdict open.

Policies, per engine kind:

* **CPU servers are preemptive.** ``rm``: fixed priorities by rate, exact
  response-time analysis (Joseph & Pandya 1986). ``edf``: schedulable iff
  utilization <= 1 (implicit deadlines, Liu & Layland 1973); the response
  bound is then the deadline.
* **GPU, NPU and KPU servers run a stage non-preemptively** -- conservative
  at stage granularity. ``rm`` uses the sufficient non-preemptive test of
  Davis, Burns, Bril & Lukkien (Real-Time Systems 35, 2007, eq. 16):
  ``w = max(B, C) + sum_hp (floor(w / T_j) + 1) C_j``, ``R = w + C``, with
  ``B`` the longest lower-priority execution. (The earlier analysis that
  omits the ``max`` and the ``+1`` was shown to be optimistic there.)
  ``edf`` on a non-preemptive server has no test here and is open unless
  utilization exceeds 1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from graphs.core.pipeline_workload import REACTIVE_CHAIN, MissionProfile, Stage
from graphs.hardware.soc import EngineKind

from .mapping import POOLED, Engine
from .schedule import Schedule

POLICIES = ("rm", "edf")
PREEMPTIVE_KINDS = (EngineKind.CPU, EngineKind.DSP)
_MAX_ITERATIONS = 100_000


def job_rate(stage: Stage, profile: MissionProfile, call_rate_hz: float) -> Tuple[Optional[float], str]:
    """Jobs per second of a stage, and how that was decided.

    The workload prices some stages per pixel, per point or per DoF: a call
    is then a fraction of a real job, and its "deadline" (tens of
    nanoseconds per pixel) is an accounting artifact that any blocking
    misses. A job is what a sensor or loop delivers at once:

    * per pixel: a camera frame -- ``sensors[stage] = [cameras, W, H, fps]``;
    * per update per DoF: one control update -- ``sensors['ctrl_hz']``;
    * per point: a scan, but no profile states a scan rate, so the job size
      is unknown (None) and the stage gets no verdict of its own;
    * anything else: the call is the job.
    """
    unit = stage.unit.lower()
    if unit == "per pixel":
        spec = profile.sensors.get(stage.key)
        if isinstance(spec, (list, tuple)) and len(spec) >= 4:
            cams, w, h, fps = spec[:4]
            if math.isclose(cams * w * h * fps, call_rate_hz, rel_tol=1e-9):
                return cams * fps, f"{cams} camera(s) x {fps} fps from the profile's sensors"
        return None, "per-pixel stage with no matching camera spec in the profile"
    if unit == "per update per dof":
        hz = profile.sensors.get("ctrl_hz")
        dof = profile.sensors.get("dof")
        if hz and dof and math.isclose(hz * dof, call_rate_hz, rel_tol=1e-9):
            return float(hz), f"ctrl_hz {hz} over {dof} DoF from the profile's sensors"
        return None, "per-DoF stage with no ctrl_hz x dof matching its rate"
    if re.search(r"\bpoints?\b", unit):  # "per input point", not "per candidate viewpoint"
        return None, f"{stage.unit}: the profile states points per second but no scan rate"
    return call_rate_hz, "one call per job"


@dataclass(frozen=True)
class Task:
    stage: str
    engine: str
    c_s: float        # execution time on this engine, per job
    period_s: float   # = deadline
    #: False when the job size is unknown: the task still interferes (as
    #: its calls), but earns no verdict of its own.
    judged: bool = True

    @property
    def utilization(self) -> float:
        return self.c_s / self.period_s


@dataclass(frozen=True)
class TaskResponse:
    task: Task
    server: int
    blocking_s: float
    response_s: Optional[float]   # upper bound; None when the test gives none
    meets: Optional[bool]         # None: open

    def to_dict(self) -> dict:
        return {"stage": self.task.stage, "engine": self.task.engine, "server": self.server,
                "c_ms": 1e3 * self.task.c_s, "deadline_ms": 1e3 * self.task.period_s,
                "blocking_ms": 1e3 * self.blocking_s,
                "response_ms": None if self.response_s is None else 1e3 * self.response_s,
                "meets": self.meets}


def _rta_preemptive(task: Task, higher: List[Task]) -> Optional[float]:
    r = task.c_s
    for _ in range(_MAX_ITERATIONS):
        nxt = task.c_s + sum(math.ceil(r / h.period_s) * h.c_s for h in higher)
        if nxt > task.period_s:
            return nxt  # past the deadline: a miss, the bound is enough
        if math.isclose(nxt, r, rel_tol=1e-12, abs_tol=0.0):
            return nxt
        r = nxt
    return None


def _rta_nonpreemptive(task: Task, higher: List[Task], blocking: float) -> Optional[float]:
    w = max(blocking, task.c_s)
    for _ in range(_MAX_ITERATIONS):
        nxt = max(blocking, task.c_s) + sum((math.floor(w / h.period_s) + 1) * h.c_s for h in higher)
        if nxt + task.c_s > task.period_s:
            return nxt + task.c_s
        if math.isclose(nxt, w, rel_tol=1e-12, abs_tol=0.0):
            return nxt + task.c_s
        w = nxt
    return None


def _partition(tasks: List[Task], servers: int) -> Tuple[List[List[Task]], bool]:
    """First-fit decreasing by utilization. The flag says whether every task
    fit under utilization 1; if not, the rest go to the least-loaded server."""
    bins: List[List[Task]] = [[] for _ in range(servers)]
    load = [0.0] * servers
    fitted = True
    for task in sorted(tasks, key=lambda t: -t.utilization):
        for i in range(servers):
            if load[i] + task.utilization <= 1.0:
                break
        else:
            fitted = False
            i = min(range(servers), key=load.__getitem__)
        bins[i].append(task)
        load[i] += task.utilization
    return bins, fitted


def _analyze_server(tasks: List[Task], server: int, preemptive: bool, policy: str) -> List[TaskResponse]:
    out: List[TaskResponse] = []
    utilization = sum(t.utilization for t in tasks)
    if policy == "edf":
        for t in tasks:
            if not t.judged:
                out.append(TaskResponse(t, server, 0.0, None, None))
            elif utilization > 1.0:
                out.append(TaskResponse(t, server, 0.0, None, False))
            elif preemptive:
                out.append(TaskResponse(t, server, 0.0, t.period_s, True))
            else:
                out.append(TaskResponse(t, server, 0.0, None, None))
        return out
    ordered = sorted(tasks, key=lambda t: (t.period_s, t.stage))  # rate monotonic
    for i, t in enumerate(ordered):
        higher, lower = ordered[:i], ordered[i + 1:]
        if preemptive:
            blocking, r = 0.0, _rta_preemptive(t, higher)
        else:
            blocking = max((lt.c_s for lt in lower), default=0.0)
            r = _rta_nonpreemptive(t, higher, blocking)
        if not t.judged:
            out.append(TaskResponse(t, server, blocking, None, None))
            continue
        out.append(TaskResponse(t, server, blocking, r, None if r is None else r <= t.period_s))
    return out


@dataclass(frozen=True)
class ResponseAnalysis:
    policy: str
    tasks: Tuple[TaskResponse, ...]
    stage_response_s: Dict[str, Optional[float]]  # upper bound per job, parts + transfer
    stage_deadline_s: Dict[str, Optional[float]]  # per job; None when the job is unknown
    partition_failed: Tuple[str, ...]  # engines FFD could not fit under utilization 1
    analyzed: bool                     # False for pooled or incomplete schedules
    #: How each stage's job was defined, or why it could not be.
    job_basis: Dict[str, str] = None  # type: ignore[assignment]

    def stage_meets(self, stage: str) -> Optional[bool]:
        """Whether the stage's upper bound meets its deadline; None when the
        policy gives no bound for one of its parts, or its job is unknown."""
        r, d = self.stage_response_s.get(stage), self.stage_deadline_s.get(stage)
        return None if r is None or d is None else r <= d

    @property
    def schedulable(self) -> Optional[bool]:
        """True only when every stage's upper bound meets its deadline. A
        miss here is *not* proof of infeasibility -- the bound is an upper
        bound, the partition a heuristic -- so it is None, not False."""
        if not self.analyzed:
            return None
        verdicts = [self.stage_meets(s) for s in self.stage_deadline_s]
        return True if verdicts and all(v is True for v in verdicts) else None

    @property
    def missing(self) -> Tuple[str, ...]:
        """Stages whose upper bound exceeds the deadline (open, not failed)."""
        return tuple(s for s in self.stage_deadline_s if self.stage_meets(s) is False)

    def reactive_chain_upper_ms(self) -> Optional[float]:
        """Upper bound on the sense-to-act chain: its stages' upper responses."""
        chain = [s for s in REACTIVE_CHAIN if s in self.stage_deadline_s]
        values = [self.stage_response_s.get(s) for s in chain]
        if not chain or any(v is None for v in values):
            return None
        return 1e3 * sum(values)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "analyzed": self.analyzed,
            "schedulable": self.schedulable,
            "stages_over_upper_bound": list(self.missing),
            "partition_failed": list(self.partition_failed),
            "reactive_chain_upper_ms": self.reactive_chain_upper_ms(),
            "stages": [
                {"stage": s, "job": (self.job_basis or {}).get(s, ""),
                 "deadline_ms": None if d is None else 1e3 * d,
                 "response_upper_ms": None if self.stage_response_s.get(s) is None
                 else 1e3 * self.stage_response_s[s],
                 "meets": self.stage_meets(s)}
                for s, d in self.stage_deadline_s.items()
            ],
            "tasks": [t.to_dict() for t in self.tasks],
        }


def response_analysis(schedule: Schedule, engines: Mapping[str, Engine],
                      stages: Mapping[str, Stage], policy: str = "rm") -> ResponseAnalysis:
    """Upper-bound response times, per job, for every priced stage of
    ``schedule``. ``stages`` are the profile's own (their units define jobs)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if schedule.mapping == POOLED or not schedule.complete:
        # A pooled machine has no engines to schedule on; an incomplete
        # schedule has stages with no execution time to schedule.
        return ResponseAnalysis(policy, (), {}, {}, (), analyzed=False)
    by_engine: Dict[str, List[Task]] = {}
    jobs: Dict[str, Optional[float]] = {}
    basis: Dict[str, str] = {}
    for svc in schedule.served:
        jobs[svc.stage], basis[svc.stage] = job_rate(stages[svc.stage], schedule.profile, svc.rate_hz)
        per_job = svc.rate_hz / jobs[svc.stage] if jobs[svc.stage] else 1.0
        rate = jobs[svc.stage] or svc.rate_hz  # an unknown job interferes as its calls
        for engine, seconds in svc.engine_seconds.items():
            by_engine.setdefault(engine, []).append(
                Task(svc.stage, engine, seconds * per_job, 1.0 / rate, judged=jobs[svc.stage] is not None))
    responses: List[TaskResponse] = []
    failed: List[str] = []
    for engine, tasks in by_engine.items():
        eng = engines[engine]
        bins, fitted = _partition(tasks, eng.servers)
        if not fitted:
            failed.append(engine)
        preemptive = eng.kind in PREEMPTIVE_KINDS
        for server, server_tasks in enumerate(bins):
            responses += _analyze_server(server_tasks, server, preemptive, policy)
    stage_response: Dict[str, Optional[float]] = {}
    deadlines: Dict[str, float] = {}
    for svc in schedule.served:
        parts = [r for r in responses if r.task.stage == svc.stage]
        job = jobs[svc.stage]
        deadlines[svc.stage] = None if job is None else 1.0 / job
        if job is None or any(r.response_s is None for r in parts):
            stage_response[svc.stage] = None
        else:
            # A split's parts carry compute only; any memory stall beyond
            # compute + transfer still delays the stage, so it is added back.
            per_job = svc.rate_hz / job
            stall = max(0.0, svc.t_service_s - svc.t_compute_s - svc.t_transfer_s) if svc.parts else 0.0
            stage_response[svc.stage] = sum(r.response_s for r in parts) + (svc.t_transfer_s + stall) * per_job
    return ResponseAnalysis(policy, tuple(responses), stage_response, deadlines,
                            tuple(failed), analyzed=True, job_basis=basis)


__all__ = ["POLICIES", "ResponseAnalysis", "Task", "TaskResponse", "job_rate", "response_analysis"]
