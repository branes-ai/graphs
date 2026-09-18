"""Process-node-independent IP block templates (graphs#269 PR 2.1).

An IP template describes one reusable SoC block -- a CPU cluster, a GPU SM, a
DLA, an LPDDR5 PHY channel -- in terms that do not depend on the process node,
so any node can be applied later by ``compose_soc``.

**Silicon.** Each block is a list of ``IPSilicon`` lines, each in one
standard-cell library (``CircuitClass``), stated one of two ways:

* ``mtx``: transistors in millions, when a count is known or estimated.
* ``reference_area_mm2`` at ``reference_node``: an area measured or disclosed
  on a real node -- which is how third-party IP is usually known, from die
  shots. It is converted to transistors with that node's density for the
  line's library, then priced at the target node with the target's density.

That second form is what makes a design retarget honestly: logic density
roughly triples from Samsung 8LPP to TSMC N5, while analog and IO barely
move, so a PHY measured at 8 nm stays nearly the same size at 5 nm and the
design becomes more pad- and PHY-dominated. Nothing special-cases it; it
falls out of the per-library density table.

**Provenance is required.** Third-party IP budgets are not public. Every
silicon line names its source and confidence, so ``show_soc`` can expose each
one to challenge, and so the Orin reconstruction -- the acceptance check --
is a test of independently sourced numbers rather than a fit.

**Compute** is optional (an LPDDR5 PHY computes nothing): unit count, peak
ops per clock per format, and the engine kind the Phase 3 mapper will key on.
**Clock** is ``fmax_ghz_ref`` at a reference node; retargeting it is
``clocking.py``'s job (PR 2.3).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import yaml
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry
from pydantic import BaseModel, Field, PositiveFloat, model_validator


class Confidence(str, Enum):
    """How much to trust a figure; mirrors ``graphs.core.ConfidenceLevel``."""

    CALIBRATED = "calibrated"
    INTERPOLATED = "interpolated"
    THEORETICAL = "theoretical"
    UNKNOWN = "unknown"


class EngineKind(str, Enum):
    """What the Phase 3 mapper will treat this block as.

    ``none`` is for blocks that carry silicon but no schedulable compute:
    memory PHYs, the system cache, the NoC, IO, the control island.
    """

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    DSP = "dsp"
    ISP = "isp"
    VIDEO_CODEC = "video_codec"
    KPU = "kpu"
    NONE = "none"


class IPSilicon(BaseModel):
    """One silicon line of an IP block, in one library."""

    name: str
    circuit_class: CircuitClass
    mtx: Optional[float] = Field(
        None, gt=0, description="Transistors, millions, per IP instance"
    )
    reference_area_mm2: Optional[float] = Field(
        None, gt=0, description="Area per IP instance, measured or disclosed"
    )
    reference_node: Optional[str] = Field(
        None, description="Process node id the reference area was measured on"
    )
    #: No credible public figure exists for this line. It still names its
    #: library and says, in ``source``, what is missing and where it could
    #: come from. Composition prices everything else and reports the die as a
    #: lower bound with this line listed as a gap -- never a number that
    #: looks complete.
    unanchored: bool = False
    source: str = Field(..., min_length=1, description="Where the figure comes from")
    confidence: Confidence = Confidence.THEORETICAL
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _exactly_one_form(self) -> "IPSilicon":
        has_mtx = self.mtx is not None
        has_area = self.reference_area_mm2 is not None
        if self.unanchored:
            if has_mtx or has_area or self.reference_node:
                raise ValueError(
                    f"silicon line {self.name!r}: an unanchored line carries no "
                    f"figure; state the gap in source instead"
                )
            if self.confidence != Confidence.UNKNOWN:
                raise ValueError(
                    f"silicon line {self.name!r}: an unanchored line's confidence "
                    f"is unknown, not {self.confidence.value}"
                )
            return self
        if has_mtx == has_area:
            raise ValueError(
                f"silicon line {self.name!r}: state exactly one of mtx or "
                f"reference_area_mm2 (got {'both' if has_mtx else 'neither'})"
            )
        if has_area and not self.reference_node:
            raise ValueError(
                f"silicon line {self.name!r}: reference_area_mm2 needs the "
                f"reference_node it was measured on"
            )
        if has_mtx and self.reference_node:
            raise ValueError(
                f"silicon line {self.name!r}: reference_node only qualifies "
                f"reference_area_mm2"
            )
        return self

    def transistors_mtx(self, nodes: Mapping[str, ProcessNodeEntry]) -> Optional[float]:
        """Transistors per instance, or None for an unanchored line. An area
        stated at a reference node is converted with that node's density for
        this library."""
        if self.unanchored:
            return None
        if self.mtx is not None:
            return self.mtx
        ref = nodes.get(self.reference_node or "")
        if ref is None:
            raise KeyError(
                f"silicon line {self.name!r}: reference node "
                f"{self.reference_node!r} is not in the process-node catalog"
            )
        if not ref.supports(self.circuit_class):
            raise ValueError(
                f"silicon line {self.name!r}: {self.reference_node} has no "
                f"{self.circuit_class.value} library to convert its area with"
            )
        return self.reference_area_mm2 * ref.density_for(self.circuit_class).mtx_per_mm2


class IPCompute(BaseModel):
    """Peak schedulable compute of one IP instance."""

    engine_kind: EngineKind
    units: int = Field(..., gt=0, description="Cores, SMs, MAC arrays per instance")
    unit_name: str = Field(..., description="What a unit is: core, SM, MAC array")
    #: Dense peak ops per clock per instance, by format. Sparsity-doubled
    #: vendor figures are not peaks and do not belong here.
    ops_per_clock: Dict[str, PositiveFloat] = Field(default_factory=dict)
    #: The ``architectural_energy.ArchitectureClass`` the Phase 3 power model
    #: will charge overhead by. Without it a CPU looks 10-50x too efficient.
    architecture_class: Optional[str] = None
    dispatch_overhead_us: Optional[float] = Field(
        None, ge=0, description="Per-kernel launch cost; dominates MPC/CBF stages"
    )
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _known_architecture_class(self) -> "IPCompute":
        if self.architecture_class is None:
            return self
        from graphs.hardware.architectural_energy import ArchitectureClass

        known = {e.value for e in ArchitectureClass}
        if self.architecture_class not in known:
            raise ValueError(
                f"architecture_class {self.architecture_class!r} is not an "
                f"ArchitectureClass; expected one of {sorted(known)}"
            )
        return self


class IPClock(BaseModel):
    """The block's rated clock where it was characterized."""

    fmax_ghz_ref: float = Field(..., gt=0)
    reference_node: str
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class IPShoreline(BaseModel):
    """Die-edge length one instance needs. PHYs and pads barely shrink with
    the node, so a small-node die can be pad-limited; PR 2.4's validator
    checks the sum against the die perimeter."""

    edge_mm: float = Field(..., gt=0)
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class IPMemoryInterface(BaseModel):
    """Off-chip bandwidth one instance provides: a DRAM controller + PHY.

    Peak, as the vendor states it; the Phase 3 schedule applies the sustained
    fraction, which depends on the access pattern rather than the block.
    """

    peak_gbps: PositiveFloat
    source: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class IPBlockTemplate(BaseModel):
    """One reusable, node-independent SoC block."""

    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    name: str
    vendor: str = ""
    engine_kind: EngineKind = EngineKind.NONE
    silicon: List[IPSilicon] = Field(..., min_length=1)
    compute: Optional[IPCompute] = None
    clock: Optional[IPClock] = None
    shoreline: Optional[IPShoreline] = None
    memory_interface: Optional[IPMemoryInterface] = None
    notes: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _compute_matches_kind(self) -> "IPBlockTemplate":
        if self.compute and self.compute.engine_kind != self.engine_kind:
            raise ValueError(
                f"IP {self.id!r}: compute.engine_kind {self.compute.engine_kind.value} "
                f"disagrees with engine_kind {self.engine_kind.value}"
            )
        names = [s.name for s in self.silicon]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"IP {self.id!r}: duplicate silicon line names {dup}")
        return self

    def transistors_mtx(self, nodes: Mapping[str, ProcessNodeEntry]) -> float:
        """Transistors of the anchored lines; see ``unanchored_lines``."""
        return sum(t for t in (s.transistors_mtx(nodes) for s in self.silicon) if t is not None)

    @property
    def unanchored_lines(self) -> List[IPSilicon]:
        return [s for s in self.silicon if s.unanchored]

    @property
    def weakest_confidence(self) -> Confidence:
        order = [Confidence.UNKNOWN, Confidence.THEORETICAL,
                 Confidence.INTERPOLATED, Confidence.CALIBRATED]
        return min((s.confidence for s in self.silicon), key=order.index)


#: The IP library that ships with the repo.
DEFAULT_IP_DIR = Path(__file__).resolve().parents[4] / "soc_designs" / "ip"


def load_ip_library(path: Optional[Path] = None) -> Dict[str, IPBlockTemplate]:
    """Every ``*.yaml`` IP template under ``path``, keyed by id."""
    root = Path(path) if path else DEFAULT_IP_DIR
    library: Dict[str, IPBlockTemplate] = {}
    for file in sorted(root.glob("*.yaml")):
        template = IPBlockTemplate.model_validate(yaml.safe_load(file.read_text()))
        if template.id in library:
            raise ValueError(f"IP id {template.id!r} is defined twice (again in {file.name})")
        if template.id != file.stem:
            raise ValueError(f"{file.name}: id {template.id!r} must match the file name")
        library[template.id] = template
    return library
