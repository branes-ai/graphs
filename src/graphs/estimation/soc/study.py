"""Studies: sweeps of SoC designs across nodes, profiles and allocations
(graphs#269 PR 4.1).

A study names base designs and what to vary. The points are the cross
product of its axes:

* ``designs`` x ``nodes`` x ``profiles`` x ``efficiency`` x ``gate_idle``;
* ``overrides``: each is one block field (``count``, ``clock_ghz``) or one
  layout field (``whitespace_fraction``, ``io_ring_mm``) with a list of
  values. A design stays a single design; the study says what varies
  (decision P4-D1).

Every point is one ``SoCAnalyzer`` run, flattened into a ``SweepRow``. Each
row carries the bound flags of its figures, so a table of rows never
presents a lower bound as a value: ``die_area_mm2`` with
``die_area_is_lower_bound``, ``power_w`` with ``power_is_lower_bound``,
``oversubscription`` with ``oversubscription_is_lower_bound``.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, model_validator

from graphs.hardware.soc import SoCDesign

from .analyzer import SoCAnalysisResult, SoCAnalyzer

_TARGET = re.compile(r"^(?:block:(?P<instance>[a-z0-9_]+)\.(?P<field>count|clock_ghz)"
                     r"|layout\.(?P<layout>whitespace_fraction|io_ring_mm))$")


class Override(BaseModel):
    """One design field and the values a study sweeps it over."""

    target: str = Field(..., description="block:<instance>.count|clock_ghz or layout.<field>")
    values: List[float] = Field(..., min_length=1)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _valid(self) -> "Override":
        m = _TARGET.match(self.target)
        if m is None:
            raise ValueError(
                f"override target {self.target!r}: expected block:<instance>.count, "
                "block:<instance>.clock_ghz, layout.whitespace_fraction or layout.io_ring_mm")
        if m["field"] == "count" and any(v != int(v) or v < 1 for v in self.values):
            raise ValueError(f"{self.target}: counts must be positive integers, got {self.values}")
        return self

    def apply(self, design: SoCDesign, value: float) -> SoCDesign:
        """A copy of ``design`` with this field set to ``value``."""
        m = _TARGET.match(self.target)
        if m["layout"]:
            layout = design.layout.model_copy(update={m["layout"]: value})
            return design.model_copy(update={"layout": layout})
        instance, fld = m["instance"], m["field"]
        if instance not in {b.instance for b in design.blocks}:
            raise KeyError(f"override {self.target}: design {design.id!r} has no block {instance!r}")
        value = int(value) if fld == "count" else value
        blocks = [b.model_copy(update={fld: value}) if b.instance == instance else b
                  for b in design.blocks]
        # Re-validate: model_copy skips validation, and a count of 0 must not pass.
        return SoCDesign.model_validate({**design.model_dump(), "blocks": [b.model_dump() for b in blocks]})


class Study(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    name: str
    workload: str
    designs: List[str] = Field(..., min_length=1)
    #: ``null`` in the list means each design's own node.
    nodes: List[Optional[str]] = Field(default_factory=lambda: [None], min_length=1)
    #: ``all``, ``regimes``, or profile ids / regime names.
    profiles: Union[Literal["all", "regimes"], List[str]] = "regimes"
    efficiency: List[str] = Field(default_factory=lambda: ["annex_v1"], min_length=1)
    mapping: str = "auto"
    gate_idle: List[bool] = Field(default_factory=lambda: [False], min_length=1)
    overrides: List[Override] = Field(default_factory=list)
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _one_override_per_target(self) -> "Study":
        targets = [o.target for o in self.overrides]
        dup = sorted({t for t in targets if targets.count(t) > 1})
        if dup:
            raise ValueError(f"study {self.id!r}: targets overridden twice: {dup}")
        return self


DEFAULT_STUDY_DIR = Path(__file__).resolve().parents[4] / "soc_designs" / "studies"


def load_study(path: Union[str, Path]) -> Study:
    """A study by file path, or by id from ``soc_designs/studies/``."""
    file = Path(path)
    if not file.suffix:
        file = DEFAULT_STUDY_DIR / f"{path}.yaml"
    study = Study.model_validate(yaml.safe_load(file.read_text()))
    if study.id != file.stem:
        raise ValueError(f"{file.name}: id {study.id!r} must match the file name")
    return study


@dataclass(frozen=True)
class SweepPoint:
    design: str
    variant: Tuple[Tuple[str, float], ...]  # (target, value), in override order
    node: Optional[str]
    profile: str
    efficiency: str
    gate_idle: bool

    @property
    def variant_label(self) -> str:
        return ",".join(f"{t.split(':')[-1]}={v:g}" for t, v in self.variant) or "base"


@dataclass(frozen=True)
class SweepRow:
    point: SweepPoint
    result: SoCAnalysisResult

    def to_row(self) -> dict:
        r, s, p = self.result, self.result.schedule, self.result.power
        return {
            "design": self.point.design,
            "variant": self.point.variant_label,
            "node": r.soc.node.id,
            "profile": r.profile.id,
            "regime": r.profile.regime or "",
            "efficiency": self.point.efficiency,
            "gate_idle": self.point.gate_idle,
            "complete": r.complete,
            "confidence": r.confidence.value,
            "feasible": r.feasible(),
            "die_area_mm2": r.soc.die_area_mm2,
            "die_area_is_lower_bound": not r.soc.complete,
            "transistors_b": r.soc.transistors_billion,
            "oversubscription": s.oversubscription,
            "oversubscription_is_lower_bound": not s.complete,
            "stages_over": len(s.stages_over()),
            "unpriced_stages": len(s.gaps),
            "dram_utilization": s.dram_utilization,
            "power_w": p.total_w,
            "power_is_lower_bound": not p.complete,
            "useful_tops_per_w": p.useful_tops_per_w,
        }


def _profile_ids(study: Study, analyzer: SoCAnalyzer) -> List[str]:
    wl = analyzer.workload
    if study.profiles == "all":
        return [p.id for p in wl.profiles]
    if study.profiles == "regimes":
        return [p.id for p in wl.regimes()]
    return [wl.profile(name).id for name in study.profiles]


def expand(study: Study, analyzer: SoCAnalyzer) -> List[Tuple[SweepPoint, SoCDesign]]:
    """Every point of the study with the design variant it runs."""
    if study.workload != analyzer.workload.version:
        raise ValueError(f"study {study.id!r} is for workload {study.workload!r}, "
                         f"the analyzer has {analyzer.workload.version!r}")
    missing = [d for d in study.designs if d not in analyzer.designs]
    if missing:
        raise KeyError(f"study {study.id!r}: unknown designs {missing}")
    profiles = _profile_ids(study, analyzer)
    override_grid = list(itertools.product(*[o.values for o in study.overrides]))
    out = []
    for design_id in study.designs:
        base = analyzer.designs[design_id]
        for values in override_grid:
            variant = base
            for override, value in zip(study.overrides, values):
                variant = override.apply(variant, value)
            assignment = tuple((o.target, v) for o, v in zip(study.overrides, values))
            for node, profile, eff, gate in itertools.product(
                    study.nodes, profiles, study.efficiency, study.gate_idle):
                out.append((SweepPoint(design_id, assignment, node, profile, eff, gate), variant))
    return out


def run_study(study: Study, analyzer: Optional[SoCAnalyzer] = None) -> List[SweepRow]:
    analyzer = analyzer or SoCAnalyzer()
    rows = []
    for point, design in expand(study, analyzer):
        result = analyzer.analyze(design, point.profile, node=point.node,
                                  efficiency=point.efficiency, mapping=study.mapping,
                                  gate_idle=point.gate_idle)
        rows.append(SweepRow(point, result))
    return rows


__all__ = ["Override", "Study", "SweepPoint", "SweepRow", "expand", "load_study", "run_study"]
