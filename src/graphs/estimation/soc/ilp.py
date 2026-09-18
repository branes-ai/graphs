"""Optimal stage-to-engine mapping as a mixed-integer program (graphs#269 Phase 5).

The greedy mapper (``mapping.greedy_mapping``) places the heaviest stage first
on the engine it loads least -- longest-processing-time scheduling, which can
miss the optimum by up to a third. This solves the assignment exactly with
``scipy.optimize.milp``:

    variables   x[s, e] in {0, 1}  for every (stage, engine) pair that can be priced
                U >= 0             the bottleneck utilization
    subject to  sum_e x[s, e] = 1                       each priced stage on one engine
                sum_s x[s, e] * occ[s, e] / servers[e] <= U   every engine
    minimize    U + eps * sum_{s,e} x[s, e] * occ[s, e] / servers[e]

So the objective is the bottleneck engine's utilization, and among mappings
with the same bottleneck, the one using the least engine time in total. As
with greedy, a stage runs on one engine (split mappings are separate work),
owns a server while it runs (the annex's lower bound), and a stage no engine
can price stays an unmapped gap with every engine's reason -- the ILP never
places a stage where the table cannot price it.

``pinned`` fixes chosen stages to chosen engines (a partial explicit
mapping) and optimizes the rest.

scipy is an optional dependency: ``pip install graphs[soc]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from graphs.core.pipeline_workload import StageDemand

from .efficiency import EfficiencyTable, KernelClassMap
from .mapping import Engine, StageService, engine_service

#: Weight of total engine time against the bottleneck: small enough never to
#: trade a worse bottleneck for it (utilizations here are O(1)-O(100)).
TIE_BREAK = 1e-6


class ILPUnavailable(ImportError):
    """scipy's milp is not installed."""


def _milp():
    try:
        import numpy as np  # noqa: PLC0415
        from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover -- exercised only without scipy
        raise ILPUnavailable(
            "the ILP mapper needs scipy >= 1.9 (pip install 'graphs[soc]')") from exc
    return np, Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Assignment:
    """The solver's answer on a plain occupancy matrix."""

    choice: Dict[int, int]      # stage index -> engine index
    bottleneck: float           # the optimal max utilization
    load: Tuple[float, ...]     # utilization per engine
    status: str


def solve_assignment(
    occupancy: Sequence[Sequence[Optional[float]]],
    servers: Sequence[int],
    pinned: Optional[Mapping[int, int]] = None,
) -> Assignment:
    """Assign every row (stage) to one column (engine) minimizing the largest
    column load, ``sum occupancy / servers``.

    ``occupancy[s][e]`` is ``None`` where stage ``s`` cannot run on engine
    ``e``; every row must have at least one number. ``pinned`` fixes rows to
    columns.
    """
    np, Bounds, LinearConstraint, milp = _milp()
    n_s, n_e = len(occupancy), len(servers)
    pinned = dict(pinned or {})
    pairs: List[Tuple[int, int]] = []
    for s, row in enumerate(occupancy):
        options = [e for e, v in enumerate(row) if v is not None]
        if not options:
            raise ValueError(f"stage row {s} has no engine it can run on")
        if s in pinned:
            if occupancy[s][pinned[s]] is None:
                raise ValueError(f"stage row {s} is pinned to engine {pinned[s]}, which cannot run it")
            options = [pinned[s]]
        pairs += [(s, e) for e in options]

    n_x = len(pairs)
    u = n_x  # index of the bottleneck variable
    weight = [occupancy[s][e] / servers[e] for s, e in pairs]
    cost = np.array(weight + [0.0]) * TIE_BREAK
    cost[u] = 1.0

    rows, lo, hi = [], [], []
    for s in range(n_s):                       # one engine per stage
        row = np.zeros(n_x + 1)
        for k, (ps, _) in enumerate(pairs):
            if ps == s:
                row[k] = 1.0
        rows.append(row); lo.append(1.0); hi.append(1.0)
    for e in range(n_e):                       # load(e) - U <= 0
        row = np.zeros(n_x + 1)
        for k, (_, pe) in enumerate(pairs):
            if pe == e:
                row[k] = weight[k]
        row[u] = -1.0
        rows.append(row); lo.append(-np.inf); hi.append(0.0)

    integrality = np.array([1] * n_x + [0])
    result = milp(
        cost, constraints=LinearConstraint(np.array(rows), lo, hi),
        integrality=integrality,
        bounds=Bounds([0.0] * n_x + [0.0], [1.0] * n_x + [np.inf]),
    )
    if not result.success:
        raise RuntimeError(f"milp failed: {result.message}")
    choice = {s: e for k, (s, e) in enumerate(pairs) if result.x[k] > 0.5}
    load = [0.0] * n_e
    for s, e in choice.items():
        load[e] += occupancy[s][e] / servers[e]
    return Assignment(choice, max(load) if load else 0.0, tuple(load), result.message)


def ilp_mapping(
    demands: Tuple[StageDemand, ...],
    engines: Mapping[str, Engine],
    kernels: KernelClassMap,
    table: EfficiencyTable,
    dram_sustained_gb_per_s: float,
    pinned: Optional[Mapping[str, str]] = None,
) -> Dict[str, StageService]:
    """The optimal one-engine-per-stage mapping, as ``greedy_mapping`` returns it."""
    names = list(engines)
    pinned = dict(pinned or {})
    unknown = sorted({e for e in pinned.values() if e not in engines})
    if unknown:
        raise KeyError(f"pinned engines the design does not have: {unknown}")
    out: Dict[str, StageService] = {}
    priced: List[StageDemand] = []
    options: List[Dict[str, StageService]] = []
    for demand in demands:
        kernel = kernels.of(demand.stage.key)
        services = {name: engine_service(demand, kernel, engines[name], table, dram_sustained_gb_per_s)
                    for name in names}
        served = {n: svc for n, svc in services.items() if svc.served}
        if not served:
            out[demand.stage.key] = StageService(
                stage=demand.stage.key, engine="", rate_hz=demand.rate_hz,
                gap="unmapped: " + "; ".join(svc.gap for svc in services.values()) if services
                else "unmapped: no engines")
            continue
        if demand.stage.key in pinned and pinned[demand.stage.key] not in served:
            out[demand.stage.key] = services[pinned[demand.stage.key]]  # the pinned engine's gap
            continue
        priced.append(demand)
        options.append(served)
    if priced:
        matrix = [[opt[n].occupancy if n in opt else None for n in names] for opt in options]
        pins = {i: names.index(pinned[d.stage.key]) for i, d in enumerate(priced)
                if d.stage.key in pinned}
        answer = solve_assignment(matrix, [engines[n].servers for n in names], pins)
        for i, demand in enumerate(priced):
            out[demand.stage.key] = options[i][names[answer.choice[i]]]
    return {d.stage.key: out[d.stage.key] for d in demands}


__all__ = ["Assignment", "ILPUnavailable", "TIE_BREAK", "ilp_mapping", "solve_assignment"]
