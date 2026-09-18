"""Compose an SoC design at a process node (graphs#269 PR 2.2).

``compose_soc(design, node)`` prices every IP block's silicon at ``node``
through ``silicon_math.area_for`` -- the same formula the KPU path uses -- and
rolls the blocks up into a die:

    core area = sum of block areas x (1 + whitespace_fraction)
    die side  = sqrt(core area) + 2 x io_ring_mm        (a square die)
    die area  = die side ^ 2

An IP line stated as an area at a reference node is first converted to
transistors with that node's density, then priced here with the target's, so
retargeting scales each library by its own density ratio. Logic density
roughly triples from 8 nm to 5 nm while analog and IO barely move, which is
what makes a small-node die PHY- and pad-dominated.

Clocks, in precedence order (``BlockInstance.clock_basis``):

* ``explicit`` -- the design's own ``clock_ghz``;
* ``reference`` -- the template's ``fmax_ghz_ref``, at the node it was
  characterized on;
* ``retargeted`` -- carried to the target node along foundry-stated speed
  relations (``clocking.retarget_fmax``), quoted at the conservative end of
  any range;
* ``unretargetable`` -- no stated relation connects the two nodes. The
  template's figure is kept so a peak can still be shown, but the instance
  lists the block in ``off_reference_clocks`` and that peak is provisional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from ..sku_validators.silicon_math import area_for
from .clocking import NodeSpeedTable, RetargetedClock, load_node_speed, retarget_fmax
from .design import DesignBlock, SoCDesign
from .ip_block import Confidence, EngineKind, IPBlockTemplate, IPSilicon

_CONFIDENCE_ORDER = [
    Confidence.UNKNOWN, Confidence.THEORETICAL, Confidence.INTERPOLATED, Confidence.CALIBRATED,
]


@dataclass(frozen=True)
class LineArea:
    """One IP silicon line, priced at the target node, for all instances.

    An unanchored line has no figure: its transistors and area are None, and
    every roll-up above it becomes a lower bound that names it.
    """

    name: str
    circuit_class: CircuitClass
    transistors_mtx_each: Optional[float]
    count: int
    area_mm2: Optional[float]
    source: str
    confidence: Confidence

    @property
    def anchored(self) -> bool:
        return self.area_mm2 is not None

    @property
    def transistors_mtx(self) -> float:
        return (self.transistors_mtx_each or 0.0) * self.count


@dataclass(frozen=True)
class BlockInstance:
    """One design block, priced at the target node."""

    block: DesignBlock
    template: IPBlockTemplate
    lines: Tuple[LineArea, ...]
    clock_ghz: Optional[float]
    clock_basis: str  # explicit | reference | retargeted | unretargetable | none
    clock_retarget: Optional[RetargetedClock] = None

    @property
    def clock_is_reference(self) -> bool:
        """Whether the clock is valid at the target node. False only when
        the template's figure had to be used where no speed relation could
        carry it."""
        return self.clock_basis != "unretargetable"

    @property
    def name(self) -> str:
        return self.block.instance

    @property
    def count(self) -> int:
        return self.block.count

    @property
    def area_mm2(self) -> float:
        """Area of the anchored lines: a lower bound when any line is not."""
        return sum(line.area_mm2 for line in self.lines if line.anchored)

    @property
    def transistors_mtx(self) -> float:
        return sum(line.transistors_mtx for line in self.lines if line.anchored)

    @property
    def unanchored(self) -> Tuple[LineArea, ...]:
        return tuple(line for line in self.lines if not line.anchored)

    @property
    def complete(self) -> bool:
        return not self.unanchored

    @property
    def engine_kind(self) -> EngineKind:
        return self.template.engine_kind

    def peak_ops_per_s(self, fmt: str) -> float:
        """Dense peak ops/s in ``fmt`` for every instance of this block."""
        compute = self.template.compute
        if compute is None or self.clock_ghz is None:
            return 0.0
        return self.count * compute.ops_per_clock.get(fmt, 0.0) * self.clock_ghz * 1e9

    @property
    def confidence(self) -> Confidence:
        return min((l.confidence for l in self.lines), key=_CONFIDENCE_ORDER.index)


@dataclass(frozen=True)
class SoCInstance:
    """A design priced at one node: the Phase 2 artifact.

    (Emitting a ``ComputeProduct`` from this is Phase 3; see
    ``docs/plans/soc-phase2-execution-plan.md``, decision P2-D1.)
    """

    design: SoCDesign
    node: ProcessNodeEntry
    blocks: Tuple[BlockInstance, ...]

    @property
    def block_area_mm2(self) -> float:
        return sum(b.area_mm2 for b in self.blocks)

    @property
    def core_area_mm2(self) -> float:
        """Blocks plus placement and routing whitespace, inside the pad ring."""
        return self.block_area_mm2 * (1.0 + self.design.layout.whitespace_fraction)

    @property
    def die_side_mm(self) -> float:
        """A square die: the core plus the IO ring on both sides."""
        return math.sqrt(self.core_area_mm2) + 2.0 * self.design.layout.io_ring_mm

    @property
    def die_area_mm2(self) -> float:
        return self.die_side_mm ** 2

    @property
    def die_perimeter_mm(self) -> float:
        return 4.0 * self.die_side_mm

    @property
    def io_ring_area_mm2(self) -> float:
        return self.die_area_mm2 - self.core_area_mm2

    @property
    def transistors_billion(self) -> float:
        return sum(b.transistors_mtx for b in self.blocks) / 1e3

    def area_by_class(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for b in self.blocks:
            for line in b.lines:
                if line.anchored:
                    out[line.circuit_class.value] = (
                        out.get(line.circuit_class.value, 0.0) + line.area_mm2
                    )
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def area_by_engine(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for b in self.blocks:
            out[b.engine_kind.value] = out.get(b.engine_kind.value, 0.0) + b.area_mm2
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def peak_ops_per_s(self, fmt: str, engine_kind: Optional[EngineKind] = None) -> float:
        return sum(
            b.peak_ops_per_s(fmt) for b in self.blocks
            if engine_kind is None or b.engine_kind == engine_kind
        )

    def peak_tops(self, fmt: str, engine_kind: Optional[EngineKind] = None) -> float:
        return self.peak_ops_per_s(fmt, engine_kind) / 1e12

    @property
    def shoreline_required_mm(self) -> float:
        """Die edge the PHYs and pads need, from the templates that state it."""
        return sum(
            b.count * b.template.shoreline.edge_mm
            for b in self.blocks if b.template.shoreline is not None
        )

    @property
    def dram_peak_gbps(self) -> float:
        """Off-chip bandwidth from the blocks that state it; 0 when none do."""
        return sum(
            b.count * b.template.memory_interface.peak_gbps
            for b in self.blocks if b.template.memory_interface is not None
        )

    @property
    def off_reference_clocks(self) -> Tuple[str, ...]:
        """Blocks whose clock could not be carried to this node: no stated
        speed relation connects it to the node the clock was measured on.
        Their peaks use the reference figure and are provisional."""
        return tuple(b.name for b in self.blocks if not b.clock_is_reference)

    @property
    def confidence(self) -> Confidence:
        return min((b.confidence for b in self.blocks), key=_CONFIDENCE_ORDER.index)

    # -- completeness ---------------------------------------------------------

    @property
    def gaps(self) -> Tuple[Tuple[str, LineArea], ...]:
        """Every unanchored line, as ``(block instance, line)``."""
        return tuple((b.name, line) for b in self.blocks for line in b.unanchored)

    @property
    def complete(self) -> bool:
        """Whether every silicon line has a figure. When it does not, every
        area and transistor total here is a **lower bound**, and a comparison
        against a published die is not a validation."""
        return not self.gaps


def _price_line(
    line: IPSilicon, count: int, node: ProcessNodeEntry, nodes: Mapping[str, ProcessNodeEntry]
) -> LineArea:
    each = line.transistors_mtx(nodes)
    if each is None:
        # Still check the library exists at the target: a gap in a library
        # the node cannot offer is two problems, and the second is fixable.
        area_for(line.name, 0.0, line.circuit_class, node)
        area_mm2 = None
    else:
        area_mm2 = area_for(line.name, each * count, line.circuit_class, node).area_mm2
    return LineArea(
        name=line.name,
        circuit_class=line.circuit_class,
        transistors_mtx_each=each,
        count=count,
        area_mm2=area_mm2,
        source=line.source,
        confidence=line.confidence,
    )


def compose_soc(
    design: SoCDesign,
    library: Mapping[str, IPBlockTemplate],
    nodes: Mapping[str, ProcessNodeEntry],
    node_id: Optional[str] = None,
    speed_table: Optional[NodeSpeedTable] = None,
) -> SoCInstance:
    """Price ``design`` at ``node_id`` (default: the design's own node)."""
    design.check_against(library)
    target_id = node_id or design.process_node
    node = nodes.get(target_id)
    if node is None:
        raise KeyError(f"process node {target_id!r} is not in the catalog")

    blocks: List[BlockInstance] = []
    for block in design.blocks:
        template = library[block.ip]
        lines = tuple(_price_line(line, block.count, node, nodes) for line in template.silicon)
        retarget: Optional[RetargetedClock] = None
        if block.clock_ghz is not None:
            clock, basis = block.clock_ghz, "explicit"
        elif template.clock is None:
            clock, basis = None, "none"
        elif template.clock.reference_node == node.id:
            clock, basis = template.clock.fmax_ghz_ref, "reference"
        else:
            if speed_table is None:
                speed_table = load_node_speed()
            retarget = retarget_fmax(
                template.clock.fmax_ghz_ref, template.clock.reference_node, node.id, speed_table
            )
            if retarget is not None:
                clock, basis = retarget.ghz, "retargeted"
            else:
                clock, basis = template.clock.fmax_ghz_ref, "unretargetable"
        blocks.append(BlockInstance(block, template, lines, clock, basis, retarget))
    return SoCInstance(design=design, node=node, blocks=tuple(blocks))
