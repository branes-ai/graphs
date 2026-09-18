"""Bound-aware Pareto fronts and the "union of regimes" minimum design
(graphs#269 PR 4.2).

Sweep figures come in two kinds: exact, or a lower bound (an input had
gaps). Treating a lower bound as a value would put a design on a front it
may not belong to, so dominance here is *proven* (decision P4-D2):

    A dominates B on minimized metrics when, for every metric, A's value is
    exact and B's value -- exact, or a lower bound on B's true value -- is
    no smaller; and for at least one metric strictly larger.

A lower bound on A proves nothing about A being small, so A dominates
nothing on a metric where it is only bounded. Each point is then:

* ``front``     -- exact on every metric, and no point dominates it or,
  through a lower bound, could;
* ``dominated`` -- some point provably dominates it;
* ``undecided`` -- neither can be shown: it has a bound somewhere and
  nothing beats it provably.

The **union of regimes** asks which design variant meets every profile in
the study, and which of those is smallest. A variant is proven feasible for
the union only if it is proven feasible (``True``) in every profile; one
proven ``False`` rules it out; anything else is undecided. The minimum is
the smallest proven-feasible variant *by exact area*, and the report lists
undecided variants whose area lower bound is below it -- they might yet be
smaller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .study import SweepRow

_LEVELS = ["calibrated", "interpolated", "theoretical", "unknown"]

#: Sweep metrics that can be minimized, with the flag that marks a bound.
METRICS: Dict[str, Tuple[str, Optional[str]]] = {
    "area": ("die_area_mm2", "die_area_is_lower_bound"),
    "power": ("power_w", "power_is_lower_bound"),
    "oversubscription": ("oversubscription", "oversubscription_is_lower_bound"),
}


@dataclass(frozen=True)
class Bounded:
    value: float
    exact: bool


def _metric(row: dict, name: str) -> Bounded:
    key, flag = METRICS[name]
    return Bounded(row[key], not (row[flag] if flag else False))


def dominates(a: dict, b: dict, metrics: Sequence[str]) -> bool:
    """A provably no worse than B on every metric, and provably better on one."""
    strictly = False
    for m in metrics:
        va, vb = _metric(a, m), _metric(b, m)
        if not va.exact or vb.value < va.value:
            return False
        if vb.value > va.value:
            strictly = True
    return strictly


def could_dominate(a: dict, b: dict, metrics: Sequence[str]) -> bool:
    """Whether A's true values *might* beat B's: every metric no larger and
    one smaller, reading A's lower bounds at their floor. An exact point
    that some bounded peer could dominate is not provably on the front."""
    va = [_metric(a, m).value for m in metrics]
    vb = [_metric(b, m).value for m in metrics]
    return all(x <= y for x, y in zip(va, vb)) and any(x < y for x, y in zip(va, vb))


def _variant_key(row: dict) -> Tuple:
    return (row["design"], row["variant"], row["node"], row["efficiency"], row["gate_idle"])


def classify_front(rows: Sequence[SweepRow], metrics: Sequence[str]) -> List[Tuple[SweepRow, str]]:
    """Each row's Pareto status, compared within its own profile."""
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise KeyError(f"unknown Pareto metrics {unknown}; have {sorted(METRICS)}")
    flat = [r.to_row() for r in rows]
    out = []
    for i, row in enumerate(flat):
        peers = [p for j, p in enumerate(flat) if j != i and p["profile"] == row["profile"]]
        if any(dominates(p, row, metrics) for p in peers):
            status = "dominated"
        elif all(_metric(row, m).exact for m in metrics) and not any(
                could_dominate(p, row, metrics) for p in peers):
            status = "front"
        else:
            status = "undecided"
        out.append((rows[i], status))
    return out


@dataclass(frozen=True)
class UnionVerdict:
    variant: Tuple  # design, variant, node, efficiency, gate_idle
    feasible: Optional[bool]
    area: Bounded
    failing_profiles: Tuple[str, ...]
    open_profiles: Tuple[str, ...]
    #: Study profiles this variant has no point for: it cannot be proven there.
    missing_profiles: Tuple[str, ...] = ()
    #: The weakest estimation confidence across the variant's points.
    confidence: Optional[dict] = None

    def to_dict(self) -> dict:
        design, variant, node, eff, gate = self.variant
        return {"design": design, "variant": variant, "node": node, "efficiency": eff,
                "gate_idle": gate, "feasible_for_all": self.feasible,
                "die_area_mm2": self.area.value, "die_area_is_lower_bound": not self.area.exact,
                "estimation_confidence": self.confidence,
                "failing_profiles": list(self.failing_profiles),
                "open_profiles": list(self.open_profiles),
                "missing_profiles": list(self.missing_profiles)}


@dataclass(frozen=True)
class UnionReport:
    profiles: Tuple[str, ...]
    verdicts: Tuple[UnionVerdict, ...]

    @property
    def minimum(self) -> Optional[UnionVerdict]:
        """Smallest variant proven feasible in every profile, by exact area."""
        proven = [v for v in self.verdicts if v.feasible is True and v.area.exact]
        return min(proven, key=lambda v: v.area.value) if proven else None

    @property
    def could_be_smaller(self) -> Tuple[UnionVerdict, ...]:
        """Undecided variants whose area lower bound is below the minimum (or
        all undecided ones when nothing is proven)."""
        best = self.minimum
        return tuple(v for v in self.verdicts if v.feasible is None
                     and (best is None or v.area.value < best.area.value))

    def to_dict(self) -> dict:
        best = self.minimum
        return {"profiles": list(self.profiles),
                "minimum": best.to_dict() if best else None,
                "could_be_smaller": [v.to_dict() for v in self.could_be_smaller],
                "variants": [v.to_dict() for v in self.verdicts]}


def union_of_regimes(rows: Sequence[SweepRow]) -> UnionReport:
    flat = [r.to_row() for r in rows]
    profiles = tuple(dict.fromkeys(r["profile"] for r in flat))
    groups: Dict[Tuple, List[dict]] = {}
    for row in flat:
        groups.setdefault(_variant_key(row), []).append(row)
    verdicts = []
    for key, group in groups.items():
        failing = tuple(r["profile"] for r in group if r["feasible"] is False)
        open_ = tuple(r["profile"] for r in group if r["feasible"] is None)
        seen = {r["profile"] for r in group}
        missing = tuple(p for p in profiles if p not in seen)
        feasible = False if failing else (True if not missing and not open_ else None)
        area = Bounded(group[0]["die_area_mm2"], not group[0]["die_area_is_lower_bound"])
        confidences = [r["estimation_confidence"] for r in group if r.get("estimation_confidence")]
        weakest = max(confidences, key=lambda c: _LEVELS.index(c["level"])) if confidences else None
        verdicts.append(UnionVerdict(key, feasible, area, failing, open_, missing, weakest))
    return UnionReport(profiles, tuple(verdicts))


__all__ = ["METRICS", "Bounded", "UnionReport", "UnionVerdict", "classify_front",
           "could_dominate", "dominates", "union_of_regimes"]
