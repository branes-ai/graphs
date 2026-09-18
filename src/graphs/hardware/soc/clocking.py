"""Retargeting a block's rated clock across process nodes (graphs#269 PR 2.3).

An IP template's ``fmax_ghz_ref`` is characterized on one node. Pricing the
design on another node needs a speed relation between the two, and the only
defensible source is the foundry's own statement of speed gain *at iso-power*:
the same block, the same power budget, a faster clock. Relations live in
``soc_designs/node_speed.yaml``, each with its source and the power condition
that source states. TSMC states the condition for N16 -> N7 (~35% at the same
power). Its 15% for N7 -> N5 and 11% for N5 -> N4P come with no condition, so
they are recorded as ``unstated`` and do not retarget a clock: a gain that
may be bought with more power is not the clock the block reaches in its budget.

``retarget_fmax`` walks those relations (either direction, inverting as it
goes) and multiplies the ratios along the path, carrying low and high ends
through so a disagreement between two foundry documents stays visible as a
range rather than being averaged away.

**What is deliberately not modeled.** The parent plan proposed an
alpha-power law, fmax(Vdd) ~ (Vdd - Vth)^alpha / Vdd, to scale clocks with
supply voltage. Research (2026-09-18) found no public threshold voltages for
these FinFET nodes and no sourced alpha exponent for sub-20 nm devices, so the
law would have two free parameters and no anchor. Clocks are retargeted at
iso-power only, and Vdd scaling waits for data.

**Where no relation exists, retargeting refuses.** Samsung publishes no speed
figure from 10LPP to 8LPP, and no foundry states a Samsung-versus-TSMC
comparison, so an 8LPP clock cannot be carried to N7. ``retarget_fmax``
returns None, and composition keeps the block flagged as off its reference
node instead of inventing a number.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, model_validator

from .ip_block import Confidence


class SpeedRange(BaseModel):
    low: float = Field(..., gt=0)
    high: float = Field(..., gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _ordered(self) -> "SpeedRange":
        if self.low > self.high:
            raise ValueError(f"speed range low {self.low} exceeds high {self.high}")
        return self


class SpeedRelation(BaseModel):
    """A foundry-stated speed gain from one node to another, and the power
    condition the source attaches to it."""

    from_node: str
    to_node: str
    speed_ratio: SpeedRange
    #: ``iso_power`` only when the source says so ("at the same power");
    #: ``unstated`` otherwise. Only iso-power relations retarget a clock.
    power_condition: Literal["iso_power", "unstated"]
    source: str = Field(..., min_length=1)
    confidence: Confidence = Confidence.THEORETICAL

    model_config = {"extra": "forbid"}


class NodeSpeedTable(BaseModel):
    notes: str = ""
    relations: List[SpeedRelation] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _one_relation_per_pair(self) -> "NodeSpeedTable":
        pairs = [frozenset((r.from_node, r.to_node)) for r in self.relations]
        dup = {tuple(sorted(p)) for p in pairs if pairs.count(p) > 1}
        if dup:
            raise ValueError(f"node pairs stated more than once: {sorted(dup)}")
        return self


DEFAULT_SPEED_TABLE = Path(__file__).resolve().parents[4] / "soc_designs" / "node_speed.yaml"


def load_node_speed(path: Optional[Path] = None) -> NodeSpeedTable:
    return NodeSpeedTable.model_validate(
        yaml.safe_load(Path(path or DEFAULT_SPEED_TABLE).read_text())
    )


@dataclass(frozen=True)
class RetargetedClock:
    """A rated clock carried to another node along stated speed relations."""

    reference_ghz: float
    reference_node: str
    target_node: str
    low_ghz: float
    high_ghz: float
    path: Tuple[str, ...]      # nodes visited, reference first
    sources: Tuple[str, ...]   # one per relation used

    @property
    def ghz(self) -> float:
        """The conservative end: where two sources disagree, peak performance
        is quoted at the lower one."""
        return self.low_ghz


def retarget_fmax(
    fmax_ghz: float,
    reference_node: str,
    target_node: str,
    table: Optional[NodeSpeedTable] = None,
) -> Optional[RetargetedClock]:
    """Carry ``fmax_ghz`` from ``reference_node`` to ``target_node``.

    Returns None when no chain of iso-power relations connects the two
    nodes -- the caller must then treat the clock as unknown at the target,
    not as ``fmax_ghz``. Relations whose power condition is unstated are
    never used.
    """
    if reference_node == target_node:
        return RetargetedClock(fmax_ghz, reference_node, target_node, fmax_ghz, fmax_ghz,
                               (reference_node,), ())
    table = table or load_node_speed()

    # Undirected graph; going against a stated relation divides by it, and
    # dividing by a range swaps its ends.
    edges: Dict[str, List[Tuple[str, float, float, str]]] = {}
    for r in table.relations:
        if r.power_condition != "iso_power":
            continue
        lo, hi = r.speed_ratio.low, r.speed_ratio.high
        edges.setdefault(r.from_node, []).append((r.to_node, lo, hi, r.source))
        edges.setdefault(r.to_node, []).append((r.from_node, 1.0 / hi, 1.0 / lo, r.source))

    # Breadth first: the shortest chain uses the fewest stated relations,
    # so it compounds the fewest uncertainties.
    queue = deque([(reference_node, 1.0, 1.0, (reference_node,), ())])
    seen = {reference_node}
    while queue:
        node, lo, hi, path, sources = queue.popleft()
        if node == target_node:
            return RetargetedClock(fmax_ghz, reference_node, target_node,
                                   fmax_ghz * lo, fmax_ghz * hi, path, sources)
        for nxt, elo, ehi, src in edges.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, lo * elo, hi * ehi, path + (nxt,), sources + (src,)))
    return None
