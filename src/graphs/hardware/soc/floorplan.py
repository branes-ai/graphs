"""What a composed die is made of, and how its area scales (graphs#269).

``compose_soc`` already rolls IP blocks into a die. This module takes that
instance apart again into the quantities a sizing and costing argument
needs: transistors and area per line, the SRAM capacity behind an SRAM
line, gate-equivalents behind a logic one, and the same totals grouped by
circuit class so the density each was divided by is visible.

Two things here are conventions rather than catalogue data, and both are
labelled so a page can say so:

* ``NAND2_TRANSISTORS`` -- a gate-equivalent is a 2-input NAND, which is
  four transistors in a standard CMOS cell. Nothing in this repository
  carries a gate count; a gate figure is arithmetic on the transistor
  figure, not a synthesis result.
* the die roll-up itself -- a square die, whitespace as a fraction of
  block area, a pad ring of a stated depth. Those are the design's
  ``layout`` record, and ``compose_soc`` documents them.

Everything else is the catalogue's: transistors from the IP templates,
densities (with a library name and a source) from the process node, SRAM
transistors per KiB from ``silicon_math.SRAM_MTX_PER_KIB``.

Unanchored lines -- a line no figure exists for -- stay unanchored. They
contribute nothing and are listed, so every total here is a **lower
bound** that names what is missing from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from ..sku_validators.silicon_math import SRAM_MTX_PER_KIB
from .compose import BlockInstance, LineArea, SoCInstance
from .design import Layout

#: Transistors in a 2-input NAND, the gate-equivalent unit. A convention,
#: not a repository figure: see the module docstring.
NAND2_TRANSISTORS = 4

LOGIC_CLASSES = (
    CircuitClass.HP_LOGIC,
    CircuitClass.BALANCED_LOGIC,
    CircuitClass.LP_LOGIC,
    CircuitClass.ULL_LOGIC,
)


def sram_kib(line: LineArea) -> Optional[float]:
    """Bytes behind an SRAM line, from the same transistors-per-KiB figure
    the catalogue's PER_KIB blocks were built with. ``None`` for a line
    that is not SRAM, or that has no transistor figure."""
    per_kib = SRAM_MTX_PER_KIB.get(line.circuit_class)
    if per_kib is None or line.transistors_mtx_each is None:
        return None
    return line.transistors_mtx / per_kib


def gates_m(line: LineArea) -> Optional[float]:
    """Gate-equivalents (millions) behind a logic line, at the NAND2
    convention. ``None`` for SRAM, analog, IO, and for a line with no
    transistor figure."""
    if line.circuit_class not in LOGIC_CLASSES or line.transistors_mtx_each is None:
        return None
    return line.transistors_mtx / NAND2_TRANSISTORS


@dataclass(frozen=True)
class LineResource:
    """One silicon line with the arithmetic that produced its area."""

    block: str
    name: str
    circuit_class: CircuitClass
    library: str
    mtx_per_mm2: Optional[float]
    transistors_mtx: Optional[float]
    area_mm2: Optional[float]
    sram_kib: Optional[float]
    gates_m: Optional[float]
    confidence: str
    source: str

    @property
    def anchored(self) -> bool:
        return self.area_mm2 is not None


@dataclass(frozen=True)
class BlockResource:
    """One design block: its lines, its totals, and what it is missing."""

    name: str
    ip: str
    count: int
    engine_kind: str
    lines: Tuple[LineResource, ...]

    @property
    def anchored_lines(self) -> Tuple[LineResource, ...]:
        return tuple(ln for ln in self.lines if ln.anchored)

    @property
    def gaps(self) -> Tuple[LineResource, ...]:
        return tuple(ln for ln in self.lines if not ln.anchored)

    @property
    def area_mm2(self) -> float:
        return sum(ln.area_mm2 for ln in self.anchored_lines)

    @property
    def transistors_mtx(self) -> float:
        return sum(ln.transistors_mtx for ln in self.anchored_lines)

    @property
    def sram_kib(self) -> float:
        return sum(ln.sram_kib or 0.0 for ln in self.lines)

    @property
    def gates_m(self) -> float:
        return sum(ln.gates_m or 0.0 for ln in self.lines)

    @property
    def complete(self) -> bool:
        return not self.gaps


@dataclass(frozen=True)
class ClassResource:
    """Everything on the die drawn from one standard-cell or bitcell
    library, with the density it was divided by."""

    circuit_class: CircuitClass
    library: str
    mtx_per_mm2: Optional[float]
    density_source: str
    density_confidence: str
    transistors_mtx: float
    area_mm2: float
    sram_kib: Optional[float]
    gates_m: Optional[float]
    lines: int


@dataclass(frozen=True)
class Composition:
    """A composed die taken apart: the Phase 7 sizing and costing view."""

    node_id: str
    blocks: Tuple[BlockResource, ...]
    classes: Tuple[ClassResource, ...]
    whitespace_fraction: float
    io_ring_mm: float
    layout_source: str
    block_area_mm2: float
    core_area_mm2: float
    die_side_mm: float
    die_area_mm2: float
    io_ring_area_mm2: float
    die_perimeter_mm: float
    shoreline_required_mm: float

    @property
    def whitespace_mm2(self) -> float:
        return self.core_area_mm2 - self.block_area_mm2

    @property
    def transistors_mtx(self) -> float:
        return sum(b.transistors_mtx for b in self.blocks)

    @property
    def sram_kib(self) -> float:
        return sum(b.sram_kib for b in self.blocks)

    @property
    def gates_m(self) -> float:
        return sum(b.gates_m for b in self.blocks)

    @property
    def gaps(self) -> Tuple[LineResource, ...]:
        return tuple(ln for b in self.blocks for ln in b.gaps)

    @property
    def complete(self) -> bool:
        return not self.gaps

    @property
    def gap_blocks(self) -> Tuple[BlockResource, ...]:
        """Blocks with no anchored line at all: they occupy no area on the
        floorplan because nothing states how much they take."""
        return tuple(b for b in self.blocks if b.area_mm2 == 0.0 and b.gaps)

    def area_by_engine(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for block in self.blocks:
            out[block.engine_kind] = out.get(block.engine_kind, 0.0) + block.area_mm2
        return out


def _library(
    node: ProcessNodeEntry, circuit_class: CircuitClass
) -> Tuple[str, Optional[float], str, str]:
    """The library a class is built in: name, density, source, confidence."""
    entry = node.densities.get(circuit_class)
    if entry is None:
        return ("", None, "", "unknown")
    confidence = getattr(entry.confidence, "value", entry.confidence)
    return (entry.library_name or circuit_class.value, entry.mtx_per_mm2,
            entry.source or "", str(confidence))


def _line_resource(block: BlockInstance, line: LineArea, node: ProcessNodeEntry) -> LineResource:
    library, density, _source, _conf = _library(node, line.circuit_class)
    return LineResource(
        block=block.name,
        name=line.name,
        circuit_class=line.circuit_class,
        library=library,
        mtx_per_mm2=density,
        transistors_mtx=None if line.transistors_mtx_each is None else line.transistors_mtx,
        area_mm2=line.area_mm2,
        sram_kib=sram_kib(line),
        gates_m=gates_m(line),
        confidence=line.confidence.value,
        source=line.source,
    )


def compose_resources(soc: SoCInstance) -> Composition:
    """Take a composed die apart into lines, blocks and libraries."""
    blocks: List[BlockResource] = []
    for block in soc.blocks:
        blocks.append(BlockResource(
            name=block.name,
            ip=block.block.ip,
            count=block.count,
            engine_kind=block.engine_kind.value,
            lines=tuple(_line_resource(block, line, soc.node) for line in block.lines),
        ))

    by_class: Dict[CircuitClass, List[LineResource]] = {}
    for block in blocks:
        for line in block.anchored_lines:
            by_class.setdefault(line.circuit_class, []).append(line)
    classes: List[ClassResource] = []
    for circuit_class, lines in by_class.items():
        library, density, source, confidence = _library(soc.node, circuit_class)
        kib = sum(ln.sram_kib for ln in lines if ln.sram_kib is not None)
        gate = sum(ln.gates_m for ln in lines if ln.gates_m is not None)
        classes.append(ClassResource(
            circuit_class=circuit_class,
            library=library,
            mtx_per_mm2=density,
            density_source=source,
            density_confidence=confidence,
            transistors_mtx=sum(ln.transistors_mtx for ln in lines),
            area_mm2=sum(ln.area_mm2 for ln in lines),
            sram_kib=kib or None,
            gates_m=gate or None,
            lines=len(lines),
        ))
    classes.sort(key=lambda c: -c.area_mm2)

    layout = soc.design.layout
    return Composition(
        node_id=soc.node.id,
        blocks=tuple(blocks),
        classes=tuple(classes),
        whitespace_fraction=layout.whitespace_fraction,
        io_ring_mm=layout.io_ring_mm,
        layout_source=layout.source,
        block_area_mm2=soc.block_area_mm2,
        core_area_mm2=soc.core_area_mm2,
        die_side_mm=soc.die_side_mm,
        die_area_mm2=soc.die_area_mm2,
        io_ring_area_mm2=soc.io_ring_area_mm2,
        die_perimeter_mm=soc.die_perimeter_mm,
        shoreline_required_mm=soc.shoreline_required_mm,
    )


# ---------------------------------------------------------------------------
# How the three areas scale
# ---------------------------------------------------------------------------


def die_area_for(block_area_mm2: float, layout: Layout) -> float:
    """The same square-die roll-up ``compose_soc`` uses, applied to a block
    area this design does not have: core = blocks x (1 + whitespace), then
    a pad ring on every side."""
    core = block_area_mm2 * (1.0 + layout.whitespace_fraction)
    return (math.sqrt(core) + 2.0 * layout.io_ring_mm) ** 2


@dataclass(frozen=True)
class ScalingLaw:
    """Die area as a function of tiles and CPU cores, fitted over a family
    of catalogued designs that differ only in fabric size.

    An extrapolation in both directions: below the smallest catalogued
    fabric and above the largest one, this is a line through points that
    do not cover the count being asked for.
    """

    per_tile_mm2: float
    fabric_fixed_mm2: float
    per_core_mm2: float
    cores_per_cluster: int
    other_fixed_mm2: float
    whitespace_fraction: float
    io_ring_mm: float
    #: (tiles, kpu block area) for every design the fit used.
    points: Tuple[Tuple[int, float], ...]
    designs: Tuple[str, ...]
    #: Blocks whose area is zero because nothing states it.
    unpriced_blocks: Tuple[str, ...]

    @property
    def tiles_fitted(self) -> Tuple[int, int]:
        return (self.points[0][0], self.points[-1][0])

    def fabric_mm2(self, tiles: float) -> float:
        return self.fabric_fixed_mm2 + self.per_tile_mm2 * tiles

    def cpu_mm2(self, cores: float) -> float:
        return self.per_core_mm2 * cores

    def block_area_mm2(self, tiles: float, cores: float) -> float:
        return self.fabric_mm2(tiles) + self.cpu_mm2(cores) + self.other_fixed_mm2

    def die_area_mm2(self, tiles: float, cores: float) -> float:
        layout = Layout(whitespace_fraction=self.whitespace_fraction,
                        io_ring_mm=self.io_ring_mm)
        return die_area_for(self.block_area_mm2(tiles, cores), layout)

    def extrapolates(self, tiles: float) -> str:
        """Whether a tile count is inside the fitted range, and which way
        out it falls when it is not."""
        low, high = self.tiles_fitted
        if tiles < low:
            return "below"
        if tiles > high:
            return "above"
        return "within"


def fit_scaling_law(
    composed: Sequence[Tuple[str, SoCInstance, int]],
    fabric_block: str = "kpu",
    cpu_block: str = "cpu",
    cores_per_cluster: int = 0,
) -> Optional[ScalingLaw]:
    """Least squares over ``(design id, composed SoC, tile count)``.

    The fabric block's area is regressed on tile count; the CPU and every
    other block are taken from the first design, which is only honest when
    the family holds them fixed -- a family that does not is rejected.
    """
    points: List[Tuple[int, float]] = []
    cpu_area = other_area = None
    cores = cores_per_cluster
    layout = None
    unpriced: List[str] = []
    used: List[str] = []
    for name, soc, tiles in composed:
        fabric = next((b for b in soc.blocks if b.name == fabric_block), None)
        cpu = next((b for b in soc.blocks if b.name == cpu_block), None)
        if fabric is None or cpu is None:
            continue
        rest = sum(b.area_mm2 for b in soc.blocks if b.name not in (fabric_block, cpu_block))
        if cpu_area is None:
            cpu_area, other_area, layout = cpu.area_mm2, rest, soc.design.layout
            if not cores:
                compute = cpu.template.compute
                cores = cpu.count * (compute.units if compute else 1)
            unpriced = [b.name for b in soc.blocks
                        if b.area_mm2 == 0.0 and not b.complete]
        elif (abs(cpu.area_mm2 - cpu_area) > 1e-9 or abs(rest - other_area) > 1e-9):
            # The family varies something other than the fabric, so a
            # per-tile slope would absorb that difference too.
            return None
        points.append((tiles, fabric.area_mm2))
        used.append(name)
    if len(points) < 2 or cpu_area is None or layout is None or not cores:
        return None
    # Points and names travel together, so a page listing the family
    # lists it smallest fabric first rather than alphabetically.
    order = sorted(range(len(points)), key=lambda i: points[i][0])
    points = [points[i] for i in order]
    used = [used[i] for i in order]
    n = len(points)
    mean_t = sum(t for t, _ in points) / n
    mean_a = sum(a for _, a in points) / n
    variance = sum((t - mean_t) ** 2 for t, _ in points)
    if variance == 0:
        return None
    slope = sum((t - mean_t) * (a - mean_a) for t, a in points) / variance
    return ScalingLaw(
        per_tile_mm2=slope,
        fabric_fixed_mm2=mean_a - slope * mean_t,
        per_core_mm2=cpu_area / cores,
        cores_per_cluster=cores,
        other_fixed_mm2=other_area or 0.0,
        whitespace_fraction=layout.whitespace_fraction,
        io_ring_mm=layout.io_ring_mm,
        points=tuple(points),
        designs=tuple(used),
        unpriced_blocks=tuple(unpriced),
    )


__all__ = [
    "LOGIC_CLASSES",
    "NAND2_TRANSISTORS",
    "BlockResource",
    "ClassResource",
    "Composition",
    "LineResource",
    "ScalingLaw",
    "compose_resources",
    "die_area_for",
    "fit_scaling_law",
    "gates_m",
    "sram_kib",
]
