"""Power roll-up for a scheduled profile (graphs#269 PR 3.3).

Parent plan 5.7 splits SoC power into dynamic, memory, leakage and
fixed-function terms. Each term here is either priced from a stated input or
reported as a gap, and a total with any gap is a **lower bound** -- the same
discipline as the die area (P2-D3) and the efficiency tables (P3-D1).

* **Dynamic** is the ALU energy of the ops each engine runs: the node's
  ``energy_per_op_pj`` for the engine's datapath library and the format each
  class runs in. This is a *floor*. Real engines pay operand fetch,
  instruction and control energy on top -- 10-50x for a CPU (parent plan,
  risk 1) -- and that architectural overhead is not priced: the repo's
  ``architectural_energy`` models report their terms with class-dependent
  meanings (a CPU's "compute overhead" includes the ALU, a systolic array's is
  a signed delta from a baseline), so they do not compose into one
  multiplier without inventing a convention (decision P3-D8). Overhead is
  always at least 1x, so the ALU energy bounds dynamic power from below.
* **Memory** is DRAM traffic times the node's ``dram_io_pj_per_byte`` (the
  on-die PHY). DRAM *device* energy has no source in the catalog and is
  always a gap.
* **Leakage** is anchored silicon area times the node's leakage density per
  library, at nominal Vdd. Unanchored silicon adds nothing and makes it a
  lower bound. ``gate_idle`` removes the leakage of engines the schedule
  provably leaves idle -- no stage mapped to them, priced or not -- which
  quantifies what idle silicon costs a regime.
* **Fixed-function** power is the ISP / codec work; the autonomy workload has
  no fixed-function stages (the parent plan's D5 extension stages are not
  costed), so it is zero, stated rather than assumed.

A pooled schedule (``annex_v1``) attributes no op to an engine, so its
dynamic power is a gap: there is no datapath to price the ops on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from embodied_schemas.process_node import CircuitClass

from graphs.core.pipeline_workload import CLASS_NAMES, PipelineWorkload
from graphs.hardware.soc import BlockInstance, SoCInstance

from .mapping import POOLED
from .schedule import Schedule

#: Libraries a datapath can be built in; SRAM, analog and IO lines are not the
#: ALU.
LOGIC_CLASSES = (CircuitClass.HP_LOGIC, CircuitClass.BALANCED_LOGIC, CircuitClass.LP_LOGIC)


@dataclass(frozen=True)
class Term:
    """One power term: watts over what could be priced, and what could not."""

    name: str
    watts: float = 0.0
    gaps: Tuple[str, ...] = ()
    floor_only: bool = False  # priced, but by construction below the true value

    @property
    def complete(self) -> bool:
        return not self.gaps and not self.floor_only

    def to_dict(self) -> dict:
        return {"watts": self.watts, "complete": self.complete,
                "floor_only": self.floor_only, "gaps": list(self.gaps)}


@dataclass(frozen=True)
class BlockPower:
    name: str
    area_mm2: float
    area_complete: bool
    dynamic_w: float
    leakage_w: float
    gated: bool

    @property
    def watts(self) -> float:
        return self.dynamic_w + self.leakage_w

    @property
    def w_per_mm2(self) -> Optional[float]:
        """Power density over anchored area. Both are lower bounds, so this is
        a ratio of bounds and says nothing on its own unless the area is
        complete; ``None`` when there is no anchored area to divide by."""
        return self.watts / self.area_mm2 if self.area_mm2 > 0 else None

    def to_dict(self) -> dict:
        return {"block": self.name, "area_mm2": self.area_mm2,
                "area_complete": self.area_complete, "dynamic_w": self.dynamic_w,
                "leakage_w": self.leakage_w, "gated": self.gated,
                "w_per_mm2": self.w_per_mm2}


@dataclass(frozen=True)
class PowerReport:
    schedule: Schedule
    dynamic: Term
    memory: Term
    dram_device: Term
    leakage: Term
    fixed: Term
    blocks: Tuple[BlockPower, ...]
    ops_per_s: float
    gate_idle: bool

    @property
    def terms(self) -> Tuple[Term, ...]:
        return (self.dynamic, self.memory, self.dram_device, self.leakage, self.fixed)

    @property
    def total_w(self) -> float:
        """Sum of what could be priced: a lower bound unless ``complete``."""
        return sum(t.watts for t in self.terms)

    @property
    def complete(self) -> bool:
        return self.schedule.complete and all(t.complete for t in self.terms)

    @property
    def budget_w(self) -> float:
        return self.schedule.profile.power_budget_w

    def within_budget(self) -> Optional[bool]:
        """False once even the lower bound exceeds the budget; None while
        gaps leave it open."""
        if self.total_w > self.budget_w:
            return False
        return True if self.complete else None

    @property
    def useful_tops_per_w(self) -> Optional[float]:
        """The workload's ops over the power -- an upper bound when the power
        is a lower bound.

        ``None`` when a stage is unscheduled or the dynamic term has gaps:
        their ops would count while their power did not, and dividing by
        leakage alone yields a "bound" in the hundreds of TOPS/W that is true,
        says nothing, and would be read as a result."""
        if not self.schedule.complete or self.dynamic.gaps or self.total_w <= 0:
            return None
        return self.ops_per_s / 1e12 / self.total_w

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "gate_idle": self.gate_idle,
            "dynamic_w": self.dynamic.to_dict(),
            "memory_w": self.memory.to_dict(),
            "dram_device_w": self.dram_device.to_dict(),
            "leakage_w": self.leakage.to_dict(),
            "fixed_w": self.fixed.to_dict(),
            "total_w": self.total_w,
            "total_is_lower_bound": not self.complete,
            "budget_w": self.budget_w,
            "within_budget": self.within_budget(),
            "useful_tops_per_w": self.useful_tops_per_w,
            "useful_tops_per_w_is_upper_bound": not self.complete,
            "blocks": [b.to_dict() for b in self.blocks],
        }


def datapath_library(block: BlockInstance) -> Optional[CircuitClass]:
    """The one logic library a block's datapath is built in, or ``None`` when
    it has none or several (then which one the ALU sits in is not stated)."""
    libs = {l.circuit_class for l in block.template.silicon if l.circuit_class in LOGIC_CLASSES}
    return next(iter(libs)) if len(libs) == 1 else None


def _dynamic(schedule: Schedule, soc: SoCInstance, workload: PipelineWorkload):
    """ALU-floor dynamic watts per engine, and the gaps."""
    per_engine: Dict[str, float] = {}
    gaps: List[str] = []
    if schedule.mapping == POOLED:
        return per_engine, ("pooled schedule: no op is attributed to an engine",)
    blocks = {b.name: b for b in soc.blocks}
    stages = {d.stage.key: d.stage for d in workload.demands(schedule.profile)}
    table = soc.node.energy_per_op_pj
    for svc in schedule.served:
        block = blocks[svc.engine]
        lib = datapath_library(block)
        if lib is None:
            gaps.append(f"{svc.stage}: {svc.engine} states no single datapath library")
            continue
        stage = stages[svc.stage]
        watts = 0.0
        for cls, share in zip(CLASS_NAMES, stage.class_split):
            if share <= 0:
                continue
            key = f"{lib.value}:{svc.formats[cls]}"
            pj = table.get(key)
            if pj is None:
                gaps.append(f"{svc.stage}: {soc.node.id} has no energy_per_op_pj[{key}]")
                watts = None
                break
            watts += stage.ops_per_call * share * svc.rate_hz * pj * 1e-12
        if watts is not None:
            per_engine[svc.engine] = per_engine.get(svc.engine, 0.0) + watts
    return per_engine, tuple(gaps)


def _leakage(block: BlockInstance, leakage: Dict[CircuitClass, float]) -> Tuple[float, List[str]]:
    watts, gaps = 0.0, []
    for line in block.lines:
        if not line.anchored:
            continue
        density = leakage.get(line.circuit_class)
        if density is None:
            gaps.append(f"{block.name}.{line.name}: no leakage density for {line.circuit_class.value}")
            continue
        watts += line.area_mm2 * density
    return watts, gaps


def roll_up(
    schedule: Schedule,
    soc: SoCInstance,
    workload: PipelineWorkload,
    gate_idle: bool = False,
) -> PowerReport:
    """Price one scheduled profile's power on its composed SoC."""
    node = soc.node
    dyn_by_engine, dyn_gaps = _dynamic(schedule, soc, workload)
    util = schedule.engine_utilization() if schedule.mapping != POOLED else {}
    # Idle means provably idle: no stage names the engine, priced or not. A
    # stage that is a gap on an engine leaves its utilization unknown, and a
    # stage no engine could take might have run anywhere, so then nothing is
    # provably idle.
    named = {s.engine for s in schedule.services}
    idle = set() if "" in named else {e for e in util if e not in named}

    blocks: List[BlockPower] = []
    leak_total, leak_gaps = 0.0, []
    for block in soc.blocks:
        leak, gaps = _leakage(block, node.leakage_w_per_mm2)
        leak_gaps += gaps
        if block.unanchored:
            leak_gaps.append(f"{block.name}: {len(block.unanchored)} unanchored line(s) add no leakage")
        gated = gate_idle and block.name in idle
        if gated:
            leak = 0.0
        leak_total += leak
        blocks.append(BlockPower(block.name, block.area_mm2, block.complete,
                                 dyn_by_engine.get(block.name, 0.0), leak, gated))
    if gate_idle and schedule.mapping == POOLED:
        leak_gaps.append("gate_idle: a pooled schedule has no per-engine idleness to gate")

    dram_bytes_per_s = schedule.dram_demand_gb_per_s * 1e9
    if node.dram_io_pj_per_byte is None:
        memory = Term("memory", gaps=(f"{node.id} states no dram_io_pj_per_byte",))
    else:
        memory = Term("memory", dram_bytes_per_s * node.dram_io_pj_per_byte * 1e-12)

    return PowerReport(
        schedule=schedule,
        dynamic=Term("dynamic", sum(dyn_by_engine.values()), dyn_gaps,
                     floor_only=schedule.mapping != POOLED),
        memory=memory,
        dram_device=Term("dram_device", gaps=("no DRAM device energy per byte in the catalog",)),
        leakage=Term("leakage", leak_total, tuple(leak_gaps)),
        fixed=Term("fixed", 0.0),
        blocks=tuple(blocks),
        ops_per_s=sum(d.ops_per_s for d in workload.demands(schedule.profile)),
        gate_idle=gate_idle,
    )


__all__ = ["BlockPower", "PowerReport", "Term", "datapath_library", "roll_up"]
