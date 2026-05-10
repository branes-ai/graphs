"""KPU silicon floorplan derivation (Stage 8).

Block-level transistor decomposition gives every silicon_bin entry an
*area*. Areas live in 2D, and architecture imposes geometric constraints
that area-only math misses:

- KPU checkerboard: alternating compute and memory tiles must share a
  pitch. If the compute-tile geometric pitch doesn't equal the
  memory-tile pitch, the die either has whitespace (lower utilization)
  or routes detour through dead silicon (longer wires, more energy).
- Aspect ratios, IO ring placement, NoC routability, and chiplet-bridge
  alignment are similar geometric constraints.

This module turns a SKU's silicon_bin into a 2D layout estimate. The
output feeds:

- ``cli/show_floorplan.py`` -- ASCII / SVG visualization
- ``sku_validators/validators/geometry.py`` -- the GEOMETRY category
  validators (pitch-match, aspect-ratio bounds, die-envelope sanity)

The KPU heuristic v1 (this module): assume every tile class shares the
same physical pitch (the max across classes), derive per-tile area
from PE-block area / num_tiles + per-tile L2/L3 SRAM area, place the
mesh in the centre, memory PHYs along the right edge, and IO ring
around the perimeter. Tile-class pitch *differences* surface as
whitespace -- the floorplan reports them; the geometric validators
flag them.

Future work (per ``docs/designs/kpu-sku-and-process-node-plan.md``
Stage 8): SVG visualization, NoC-routability validator, IO-ring
validator, multi-architecture floorplanners (CPU, GPU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from embodied_schemas.kpu import KPUEntry, KPUTileSpec
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from .models.accelerators.kpu_yaml_loader import _DRAM_READ_PJ_PER_BYTE  # noqa: F401  (re-export not needed but keeps import group tidy)
from .sku_validators.silicon_math import (
    SiliconMathError,
    resolve_block_area,
)


# ---------------------------------------------------------------------------
# Floorplan dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FloorplanBlock:
    """One block placed in the 2D floorplan.

    Coordinates use bottom-left-origin; (0, 0) is the lower-left corner
    of the die. All units are millimetres.
    """

    name: str
    """The silicon_bin block this rectangle represents, OR a synthetic
    name like ``compute_tile[0,0]`` for individual mesh tiles."""

    circuit_class: CircuitClass
    """Library tag from the originating silicon_bin block (or the
    dominant class for synthetic compute-tile blocks)."""

    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float

    is_compute_tile: bool = False
    """True for the per-tile rectangles in the compute mesh; False for
    aggregate blocks (memory PHY, IO ring, control logic)."""

    tile_class: Optional[str] = None
    """For compute tiles, the tile class name (e.g., 'INT8-primary')."""

    notes: str = ""

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    @property
    def aspect_ratio(self) -> float:
        """``max(w, h) / min(w, h)``. 1.0 = perfectly square; >2 is
        commonly considered hard to route at modern nodes."""
        wide = max(self.width_mm, self.height_mm)
        tall = min(self.width_mm, self.height_mm)
        return wide / tall if tall > 0 else float("inf")


@dataclass(frozen=True)
class TilePitch:
    """Per-tile-class pitch summary for the geometric validators."""

    tile_class: str
    pe_area_mm2: float
    sram_area_mm2: float
    total_area_mm2: float
    pitch_mm: float
    """``sqrt(total_area_mm2)`` -- side length if the tile is square."""


@dataclass(frozen=True)
class Floorplan:
    """A 2D placement of every silicon_bin block in a SKU."""

    sku_id: str
    sku_name: str
    die_width_mm: float
    die_height_mm: float
    blocks: list[FloorplanBlock]
    tile_pitches: dict[str, TilePitch] = field(default_factory=dict)
    """Per-tile-class pitch summary (one entry per tile class)."""

    unified_pitch_mm: float = 0.0
    """The uniform pitch the floorplan uses for the mesh -- typically
    the ``max(tile_pitches.values())`` so every tile class fits."""

    notes: str = ""

    @property
    def die_area_mm2(self) -> float:
        return self.die_width_mm * self.die_height_mm

    def total_block_area_mm2(self) -> float:
        return sum(b.area_mm2 for b in self.blocks)

    def whitespace_mm2(self) -> float:
        """Area inside the die not covered by any block. Sum of
        ``unified_pitch - tile_class_pitch`` rectangles across tile
        classes that don't fill the unified-pitch envelope."""
        return max(0.0, self.die_area_mm2 - self.total_block_area_mm2())

    def whitespace_fraction(self) -> float:
        """Whitespace as a fraction of die area; 0 = perfectly tiled."""
        if self.die_area_mm2 <= 0:
            return 0.0
        return self.whitespace_mm2() / self.die_area_mm2

    def compute_tiles(self) -> list[FloorplanBlock]:
        return [b for b in self.blocks if b.is_compute_tile]


# ---------------------------------------------------------------------------
# Per-tile area derivation
# ---------------------------------------------------------------------------

def _per_tile_pe_area_mm2(
    sku: KPUEntry,
    node: ProcessNodeEntry,
    tile: KPUTileSpec,
    pe_blocks_by_tile_type: dict[str, float],
) -> float:
    """PE-block area divided by num_tiles for this tile class.

    The silicon_bin's pe_* blocks aggregate across all tiles in a class;
    dividing by num_tiles gives the per-tile contribution.
    """
    aggregate_pe_area = pe_blocks_by_tile_type.get(tile.tile_type, 0.0)
    if tile.num_tiles <= 0:
        return 0.0
    return aggregate_pe_area / tile.num_tiles


def _per_tile_sram_area_mm2(
    sku: KPUEntry,
    node: ProcessNodeEntry,
    chip_l2_area_mm2: float,
    chip_l3_area_mm2: float,
) -> float:
    """Per-tile L2 + L3 SRAM area (chip-wide totals divided by num_tiles).

    The KPU's M0.5 architecture has per-tile L2 (32 KiB typical) and
    per-tile L3 (256 KiB scratchpad). silicon_bin reports them as
    chip-wide aggregates; per-tile is aggregate / num_tiles.
    """
    total_tiles = sku.kpu_architecture.total_tiles
    if total_tiles <= 0:
        return 0.0
    return (chip_l2_area_mm2 + chip_l3_area_mm2) / total_tiles


def _classify_silicon_bin_blocks(
    sku: KPUEntry, node: ProcessNodeEntry
) -> tuple[
    dict[str, float],         # pe_area by tile_type
    float,                    # chip-wide L2 area
    float,                    # chip-wide L3 area
    list[tuple[str, float, CircuitClass]],  # other blocks: (name, area, class)
]:
    """Walk the silicon_bin once and bucket each block.

    Buckets:
      - PE blocks: collected per tile class (count_ref = "tile.<type>")
      - L2 / L3 SRAM blocks: by count_ref (l2_total_kib / l3_total_kib)
      - Everything else: passed through to the floorplan as a
        non-mesh placed block (memory PHY, IO pads, control logic, NoC).
    """
    pe_area_by_tile_type: dict[str, float] = {}
    chip_l2_area = 0.0
    chip_l3_area = 0.0
    other_blocks: list[tuple[str, float, CircuitClass]] = []

    for block in sku.silicon_bin.blocks:
        try:
            ba = resolve_block_area(block, sku, node)
        except SiliconMathError:
            continue
        ts = block.transistor_source
        if ts.kind.value == "per_pe" and ts.count_ref and ts.count_ref.startswith("tile."):
            tile_type = ts.count_ref.split(".", 1)[1]
            pe_area_by_tile_type[tile_type] = (
                pe_area_by_tile_type.get(tile_type, 0.0) + ba.area_mm2
            )
        elif ts.kind.value == "per_kib" and ts.count_ref == "l2_total_kib":
            chip_l2_area += ba.area_mm2
        elif ts.kind.value == "per_kib" and ts.count_ref == "l3_total_kib":
            chip_l3_area += ba.area_mm2
        else:
            other_blocks.append((block.name, ba.area_mm2, block.circuit_class))

    return pe_area_by_tile_type, chip_l2_area, chip_l3_area, other_blocks


# ---------------------------------------------------------------------------
# Floorplan derivation -- KPU M0.5
# ---------------------------------------------------------------------------

# Edge thicknesses for the IO ring + corner control region. Heuristic
# placeholders; real silicon-floorplan tools derive these from pad pitch
# and a chip-wide area budget. Reasonable defaults that produce sensible
# floorplans across T64..T768.
_IO_RING_THICKNESS_MM = 0.30
_CONTROL_CORNER_FRACTION = 0.05  # Of die_height for the control square


def derive_kpu_floorplan(
    sku: KPUEntry, node: ProcessNodeEntry
) -> Floorplan:
    """Heuristic 2D floorplan for a KPU SKU.

    Layout:

    1. Compute the per-tile-class pitch:
       ``pitch[c] = sqrt(per_tile_pe_area[c] + per_tile_sram_area)``.
    2. Pick the unified mesh pitch as ``max(pitch[c])`` so every tile
       class fits without overflow. Smaller-pitch classes leave
       whitespace within their tile envelope -- the floorplan reports
       that and the geometry validators flag it.
    3. Place ``mesh_rows x mesh_cols`` compute tiles in a grid using
       the unified pitch, in row-major order. Tile class assignment
       follows the YAML's ``tile_mix`` proportions: INT8-primary fills
       the bulk, then BF16-primary, then Matrix.
    4. Place memory PHYs in a vertical strip on the RIGHT edge of the
       mesh (typical KPU layout; controllers drive an LPDDR/HBM PHY
       block adjacent to the mesh).
    5. Place IO pads as a thin ring around the perimeter
       (``_IO_RING_THICKNESS_MM`` thick).
    6. Place control logic in the bottom-left corner (a small square
       sized at ``_CONTROL_CORNER_FRACTION * die_height``).

    The die rectangle is sized to fit all of the above with no
    overlaps. NoC routers are *not* placed as separate blocks --
    in real silicon they're embedded inside the tile envelopes.
    """
    arch = sku.kpu_architecture

    # 1. Bucket silicon_bin blocks
    pe_by_tile_type, chip_l2_area, chip_l3_area, other_blocks = (
        _classify_silicon_bin_blocks(sku, node)
    )

    # 2. Per-tile-class pitch
    per_tile_sram = _per_tile_sram_area_mm2(
        sku, node, chip_l2_area, chip_l3_area
    )
    tile_pitches: dict[str, TilePitch] = {}
    for tile in arch.tiles:
        pe_area = _per_tile_pe_area_mm2(sku, node, tile, pe_by_tile_type)
        total = pe_area + per_tile_sram
        pitch = math.sqrt(total) if total > 0 else 0.0
        tile_pitches[tile.tile_type] = TilePitch(
            tile_class=tile.tile_type,
            pe_area_mm2=pe_area,
            sram_area_mm2=per_tile_sram,
            total_area_mm2=total,
            pitch_mm=pitch,
        )

    # 3. Unified mesh pitch
    unified_pitch = max(
        (tp.pitch_mm for tp in tile_pitches.values()),
        default=0.0,
    )

    # 4. Place compute tiles in a row-major grid using tile_mix proportions.
    #    Order: INT8-primary first, then BF16-primary, then Matrix
    #    (matches the YAML's tile order convention).
    mesh_rows = arch.noc.mesh_rows
    mesh_cols = arch.noc.mesh_cols
    mesh_width = mesh_cols * unified_pitch
    mesh_height = mesh_rows * unified_pitch

    # 5. Memory PHY strip on the right
    total_phy_area = sum(
        area for name, area, _ in other_blocks
        if "phy" in name.lower() or "memory_phys" in name.lower()
    )
    phy_strip_width = (
        total_phy_area / mesh_height if mesh_height > 0 else 0.0
    )

    # 6. Die dimensions: mesh + PHY strip + IO ring on all sides
    die_width = mesh_width + phy_strip_width + 2 * _IO_RING_THICKNESS_MM
    die_height = mesh_height + 2 * _IO_RING_THICKNESS_MM

    blocks: list[FloorplanBlock] = []

    # Place IO ring (4 edge rectangles)
    blocks.extend(_place_io_ring(die_width, die_height, other_blocks))

    # Place compute mesh tiles (origin offset by ring + nothing else; control
    # corner overlays the bottom-left mesh-adjacent space)
    mesh_origin_x = _IO_RING_THICKNESS_MM
    mesh_origin_y = _IO_RING_THICKNESS_MM
    blocks.extend(
        _place_compute_mesh(
            sku, mesh_origin_x, mesh_origin_y, unified_pitch
        )
    )

    # Place memory PHY strip on the right
    if phy_strip_width > 0:
        blocks.extend(_place_memory_phys(
            other_blocks,
            x_mm=mesh_origin_x + mesh_width,
            y_mm=mesh_origin_y,
            width_mm=phy_strip_width,
            height_mm=mesh_height,
        ))

    # Place control logic in the bottom-left corner (over the mesh,
    # symbolizing the control bus that fans out to all tiles)
    blocks.extend(_place_control(
        other_blocks,
        x_mm=mesh_origin_x,
        y_mm=mesh_origin_y,
        die_height=die_height,
    ))

    return Floorplan(
        sku_id=sku.id,
        sku_name=sku.name,
        die_width_mm=die_width,
        die_height_mm=die_height,
        blocks=blocks,
        tile_pitches=tile_pitches,
        unified_pitch_mm=unified_pitch,
        notes=(
            f"Heuristic v1 floorplan: {mesh_rows}x{mesh_cols} mesh @ "
            f"{unified_pitch:.3f} mm pitch, PHY strip + IO ring."
        ),
    )


def _place_io_ring(
    die_width_mm: float,
    die_height_mm: float,
    other_blocks: list[tuple[str, float, CircuitClass]],
) -> list[FloorplanBlock]:
    """Four edge rectangles forming the IO ring.

    The ``io_pads`` silicon_bin block contributes the total IO area;
    this function spreads it across the four edges proportional to
    edge length. Each edge is ``_IO_RING_THICKNESS_MM`` thick.
    """
    io_area = sum(
        area for name, area, _ in other_blocks
        if "io" in name.lower() and "phy" not in name.lower()
    )
    if io_area <= 0:
        return []
    perimeter = 2 * (die_width_mm + die_height_mm)
    if perimeter <= 0:
        return []
    # If the IO area exceeds what fits in the ring, the ring is "thick"
    # and may be larger than _IO_RING_THICKNESS_MM. Solve for the actual
    # thickness given total IO area + perimeter.
    # area = perimeter * t - 4 * t^2  (correcting for corner overlap)
    # For small t relative to die dimensions this is approximately
    # area ~ perimeter * t. Use the simpler estimate; corner overlap
    # is at most a small percentage.
    thickness = min(io_area / perimeter, _IO_RING_THICKNESS_MM)

    return [
        # Bottom
        FloorplanBlock(
            name="io_ring_bottom",
            circuit_class=CircuitClass.IO,
            x_mm=0.0, y_mm=0.0,
            width_mm=die_width_mm, height_mm=thickness,
        ),
        # Top
        FloorplanBlock(
            name="io_ring_top",
            circuit_class=CircuitClass.IO,
            x_mm=0.0, y_mm=die_height_mm - thickness,
            width_mm=die_width_mm, height_mm=thickness,
        ),
        # Left
        FloorplanBlock(
            name="io_ring_left",
            circuit_class=CircuitClass.IO,
            x_mm=0.0, y_mm=thickness,
            width_mm=thickness, height_mm=die_height_mm - 2 * thickness,
        ),
        # Right
        FloorplanBlock(
            name="io_ring_right",
            circuit_class=CircuitClass.IO,
            x_mm=die_width_mm - thickness, y_mm=thickness,
            width_mm=thickness, height_mm=die_height_mm - 2 * thickness,
        ),
    ]


def _place_compute_mesh(
    sku: KPUEntry,
    origin_x_mm: float,
    origin_y_mm: float,
    pitch_mm: float,
) -> list[FloorplanBlock]:
    """Place ``mesh_rows x mesh_cols`` compute tiles in row-major order.

    Tile-class assignment cycles through ``arch.tiles`` in declaration
    order (INT8-primary first, then BF16-primary, then Matrix), each
    class taking ``num_tiles`` consecutive grid positions.
    """
    arch = sku.kpu_architecture
    mesh_rows = arch.noc.mesh_rows
    mesh_cols = arch.noc.mesh_cols
    expected_total = mesh_rows * mesh_cols

    # Build the tile-class assignment list
    assignment: list[KPUTileSpec] = []
    for tile in arch.tiles:
        assignment.extend([tile] * tile.num_tiles)
    # Pad with the last class if mesh has more positions than tiles declared
    while len(assignment) < expected_total:
        assignment.append(arch.tiles[-1])
    # Truncate if tile_mix sums above mesh capacity
    assignment = assignment[:expected_total]

    blocks: list[FloorplanBlock] = []
    for idx in range(expected_total):
        row = idx // mesh_cols
        col = idx % mesh_cols
        tile = assignment[idx]
        # Pick a representative circuit class for the tile (PE class)
        cc = tile.pe_circuit_class
        blocks.append(FloorplanBlock(
            name=f"tile[{row},{col}]",
            circuit_class=cc,
            x_mm=origin_x_mm + col * pitch_mm,
            y_mm=origin_y_mm + row * pitch_mm,
            width_mm=pitch_mm,
            height_mm=pitch_mm,
            is_compute_tile=True,
            tile_class=tile.tile_type,
        ))
    return blocks


def _place_memory_phys(
    other_blocks: list[tuple[str, float, CircuitClass]],
    *,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
) -> list[FloorplanBlock]:
    """Single PHY strip on the right side of the mesh.

    Aggregates all memory_phys / *_phy blocks into one rectangle.
    Future work: split into per-controller blocks for finer-grained
    visualization.
    """
    phy_blocks = [
        (name, area, cc) for name, area, cc in other_blocks
        if "phy" in name.lower()
    ]
    if not phy_blocks:
        return []
    return [FloorplanBlock(
        name="memory_phy_strip",
        circuit_class=phy_blocks[0][2],  # Typically CircuitClass.ANALOG
        x_mm=x_mm,
        y_mm=y_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        notes=f"Aggregates {len(phy_blocks)} PHY block(s)",
    )]


def _place_control(
    other_blocks: list[tuple[str, float, CircuitClass]],
    *,
    x_mm: float,
    y_mm: float,
    die_height: float,
) -> list[FloorplanBlock]:
    """Control logic in the bottom-left corner."""
    ctrl_blocks = [
        (name, area, cc) for name, area, cc in other_blocks
        if "control" in name.lower()
    ]
    if not ctrl_blocks:
        return []
    total_area = sum(area for _, area, _ in ctrl_blocks)
    side = math.sqrt(total_area) if total_area > 0 else 0.0
    # Cap at a fraction of the die height to avoid the control block
    # eating the mesh
    side = min(side, die_height * _CONTROL_CORNER_FRACTION)
    if side <= 0:
        return []
    return [FloorplanBlock(
        name="control_logic",
        circuit_class=ctrl_blocks[0][2],
        x_mm=x_mm,
        y_mm=y_mm,
        width_mm=side,
        height_mm=side,
        notes=f"Aggregates {len(ctrl_blocks)} control block(s); "
              f"placed in bottom-left corner",
    )]


# ===========================================================================
# Architectural-view floorplan
# ===========================================================================
#
# The circuit-class view above keys on silicon library (HP_LOGIC, BALANCED,
# SRAM_HD, ANALOG, IO). That tells you what was *built*, not what it
# *does*. The architectural view re-bins the same areas by what the
# architect calls them:
#
#   * COMPUTE tile = PE fabric + L1 + L2 (the unit that does work)
#   * MEMORY tile  = L3 (the local scratchpad that alternates with
#                    compute in the KPU checkerboard)
#   * MEMORY_CONTROLLER = DRAM PHY block, distributed around the
#                    periphery at the arity of the memory configuration
#   * IO_PAD       = pad ring (perimeter)
#   * CONTROL      = scheduler/dispatch (corner)
#
# The primary geometric concern shifts from "do all compute tile classes
# share a pitch" to "do compute tiles share a pitch with memory tiles".
# When a checkerboard pairs a 0.335mm compute tile with a 0.471mm
# memory tile, the layout either leaves whitespace inside the smaller
# cell or routes detour around uneven cells -- both bad.
#
# This view also exposes a "what-if" lens: if every compute tile in the
# mesh were of class X, what would the die area + whitespace be? That
# tells the architect how much die-area cost the heterogeneous mix
# (Matrix tiles in particular) is paying vs an all-INT8 or all-BF16
# alternative.


class TileRole(str, Enum):
    """Architectural role of a placed block.

    The role is what the architect calls the block, independent of which
    silicon library implements it. A compute tile can be HP_LOGIC,
    BALANCED_LOGIC, or both at once (PEs + L1+L2 SRAM); the role stays
    COMPUTE.
    """

    COMPUTE = "compute"
    MEMORY = "memory"
    MEMORY_CONTROLLER = "memory_controller"
    IO_PAD = "io_pad"
    CONTROL = "control"


@dataclass(frozen=True)
class ArchTile:
    """One architectural block (a tile, controller, ring segment, etc.).

    Coordinates use bottom-left-origin; (0, 0) is the lower-left corner
    of the die. All units mm.
    """

    name: str
    role: TileRole
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float

    tile_class: Optional[str] = None
    """For COMPUTE tiles, the architectural tile class
    (e.g., 'INT8-primary'). None for non-compute roles."""

    pe_area_mm2: Optional[float] = None
    """For COMPUTE tiles: the PE-fabric area within this tile."""

    l2_area_mm2: Optional[float] = None
    """For COMPUTE tiles: the per-tile L2 area."""

    l3_area_mm2: Optional[float] = None
    """For MEMORY tiles: the per-tile L3 area."""

    notes: str = ""

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    @property
    def used_area_mm2(self) -> float:
        """Area actually consumed by silicon inside this tile's envelope.

        For COMPUTE: pe_area + l2_area. For MEMORY: l3_area. For
        non-tile roles (IO ring, controllers, control): the full
        ``area_mm2``. The difference (envelope - used) is whitespace
        inside the tile cell.
        """
        if self.role == TileRole.COMPUTE:
            return (self.pe_area_mm2 or 0.0) + (self.l2_area_mm2 or 0.0)
        if self.role == TileRole.MEMORY:
            return self.l3_area_mm2 or 0.0
        return self.area_mm2

    @property
    def whitespace_mm2(self) -> float:
        return max(0.0, self.area_mm2 - self.used_area_mm2)


@dataclass(frozen=True)
class ComputeClassSummary:
    """Per-class compute-tile area + pitch + whitespace breakdown.

    A SKU with 3 tile classes (INT8, BF16, Matrix) produces 3 of these.
    Comparing them shows which classes are pitch-bound (smaller than
    the unified pitch) and how much whitespace they collectively
    contribute to the die.
    """

    tile_class: str
    num_tiles: int
    pe_area_mm2: float
    l2_area_mm2: float
    total_area_mm2: float
    pitch_mm: float
    """``sqrt(total_area_mm2)`` -- side length if this class were the
    only thing in the mesh."""

    whitespace_per_tile_mm2: float
    """``unified_pitch^2 - total_area_mm2`` -- what each tile of this
    class wastes when placed in the unified-pitch envelope."""

    class_whitespace_mm2: float
    """``num_tiles * whitespace_per_tile_mm2`` -- total die area this
    class costs in whitespace alone."""


@dataclass(frozen=True)
class MemoryClassSummary:
    """Memory-tile area + pitch + whitespace.

    All memory tiles are uniform (one L3 per compute tile, same size
    everywhere) so there's just one of these per SKU.
    """

    num_tiles: int
    l3_area_mm2: float
    total_area_mm2: float
    pitch_mm: float
    whitespace_per_tile_mm2: float
    class_whitespace_mm2: float


@dataclass(frozen=True)
class WhatIfDieEstimate:
    """What the die would look like if every compute tile were one class.

    Computed independently for each compute tile class. The architect
    can compare these against the actual mixed-class die to see the
    cost of the heterogeneous mix.
    """

    tile_class: str
    """The compute class assumed for every tile in this estimate."""

    unified_pitch_mm: float
    """``max(class_pitch, memory_pitch)`` for this scenario."""

    mesh_area_mm2: float
    """Total mesh rectangle (compute + memory tiles, no periphery)."""

    die_area_mm2: float
    """Including IO ring + memory controllers + control corner."""

    whitespace_mm2: float
    """Die area not covered by silicon, in this scenario."""

    whitespace_fraction: float


@dataclass(frozen=True)
class ArchitecturalFloorplan:
    """Architectural-role floorplan: compute tiles, memory tiles,
    controllers, IO ring, control. Companion to the circuit-class
    ``Floorplan`` -- both are derived from the same silicon_bin.
    """

    sku_id: str
    sku_name: str
    die_width_mm: float
    die_height_mm: float
    blocks: list[ArchTile]

    compute_summaries: dict[str, ComputeClassSummary]
    memory_summary: MemoryClassSummary
    num_memory_controllers: int
    unified_pitch_mm: float
    """``max(max_compute_pitch, memory_pitch)`` -- the single mesh
    cell side length used for both compute and memory tiles."""

    what_if: list[WhatIfDieEstimate] = field(default_factory=list)
    """Per-class 'if every compute tile were class X' die estimates."""

    notes: str = ""

    @property
    def die_area_mm2(self) -> float:
        return self.die_width_mm * self.die_height_mm

    @property
    def compute_memory_pitch_ratio(self) -> float:
        """``max(max_compute_pitch, memory_pitch) /
        min(max_compute_pitch, memory_pitch)``. 1.0 = perfect
        checkerboard match; >1.2 starts wasting silicon."""
        cps = [s.pitch_mm for s in self.compute_summaries.values()]
        if not cps or self.memory_summary.pitch_mm <= 0:
            return float("inf")
        max_c = max(cps)
        if max_c <= 0:
            return float("inf")
        return max(max_c, self.memory_summary.pitch_mm) / min(
            max_c, self.memory_summary.pitch_mm
        )

    def total_used_area_mm2(self) -> float:
        return sum(b.used_area_mm2 for b in self.blocks)

    def whitespace_mm2(self) -> float:
        return max(0.0, self.die_area_mm2 - self.total_used_area_mm2())

    def whitespace_fraction(self) -> float:
        if self.die_area_mm2 <= 0:
            return 0.0
        return self.whitespace_mm2() / self.die_area_mm2

    def compute_tiles(self) -> list[ArchTile]:
        return [b for b in self.blocks if b.role == TileRole.COMPUTE]

    def memory_tiles(self) -> list[ArchTile]:
        return [b for b in self.blocks if b.role == TileRole.MEMORY]

    def memory_controllers(self) -> list[ArchTile]:
        return [b for b in self.blocks if b.role == TileRole.MEMORY_CONTROLLER]


# ---------------------------------------------------------------------------
# Architectural derivation
# ---------------------------------------------------------------------------

# Default memory-controller arity when silicon_bin doesn't expose a
# distinct count. 4 corresponds to one controller centered on each of
# the four die edges -- typical of LPDDR-class accelerators. HBM-class
# parts often want 8 (two per edge); the heuristic below upgrades to
# match if silicon_bin contains multiple distinct PHY blocks.
_DEFAULT_NUM_MEMORY_CONTROLLERS = 4


def derive_kpu_architectural_floorplan(
    sku: KPUEntry, node: ProcessNodeEntry
) -> ArchitecturalFloorplan:
    """Produce the architectural-role floorplan for a KPU SKU.

    Layout (heuristic v1):

    1. Per compute tile class: pitch = sqrt(per_tile_PE_area +
       per_tile_L2_area).
    2. Memory tile pitch = sqrt(per_tile_L3_area). One memory tile per
       compute tile (1:1 pairing -- each compute tile owns its L3).
    3. Unified pitch = max(max_compute_pitch, memory_pitch). Smaller
       cells leave whitespace.
    4. Physical mesh = ``mesh_rows x (2 * mesh_cols)``: every compute
       tile is paired with a memory tile to its right (column-pair
       checkerboard). True 2D checkerboards are also reasonable; this
       layout was picked because it renders cleanly in ASCII and
       preserves N/S compute-compute neighbours within the NoC.
    5. Memory controllers: ``N`` distinct PHY blocks (or default 4) of
       equal size, distributed around the perimeter clockwise from
       top-center.
    6. IO ring: thin pad ring around the perimeter.
    7. Control: corner block.

    NoC routers are not drawn -- they're embedded inside the tile
    envelopes physically. A future SVG/HTML renderer can overlay the
    NoC topology (mesh / ring / CLOS) on top of the placed blocks.
    """
    arch = sku.kpu_architecture
    pe_by_tile_type, chip_l2_area, chip_l3_area, other_blocks = (
        _classify_silicon_bin_blocks(sku, node)
    )
    total_compute_tiles = arch.total_tiles

    # Per-tile L2 / L3 (shared by every compute / memory tile)
    per_tile_l2 = (
        chip_l2_area / total_compute_tiles if total_compute_tiles > 0 else 0.0
    )
    per_tile_l3 = (
        chip_l3_area / total_compute_tiles if total_compute_tiles > 0 else 0.0
    )
    memory_pitch = math.sqrt(per_tile_l3) if per_tile_l3 > 0 else 0.0

    # Per-class compute pitches
    compute_pitches: dict[str, float] = {}
    compute_pe_areas: dict[str, float] = {}
    for tile in arch.tiles:
        pe_area = (
            pe_by_tile_type.get(tile.tile_type, 0.0) / tile.num_tiles
            if tile.num_tiles > 0 else 0.0
        )
        total = pe_area + per_tile_l2
        pitch = math.sqrt(total) if total > 0 else 0.0
        compute_pitches[tile.tile_type] = pitch
        compute_pe_areas[tile.tile_type] = pe_area

    # Unified pitch dominates compute and memory
    max_compute_pitch = max(compute_pitches.values(), default=0.0)
    unified_pitch = max(max_compute_pitch, memory_pitch)
    cell_area = unified_pitch * unified_pitch

    # Per-class summaries with whitespace
    compute_summaries: dict[str, ComputeClassSummary] = {}
    for tile in arch.tiles:
        pe_area = compute_pe_areas[tile.tile_type]
        total = pe_area + per_tile_l2
        ws_per_tile = max(0.0, cell_area - total)
        compute_summaries[tile.tile_type] = ComputeClassSummary(
            tile_class=tile.tile_type,
            num_tiles=tile.num_tiles,
            pe_area_mm2=pe_area,
            l2_area_mm2=per_tile_l2,
            total_area_mm2=total,
            pitch_mm=compute_pitches[tile.tile_type],
            whitespace_per_tile_mm2=ws_per_tile,
            class_whitespace_mm2=ws_per_tile * tile.num_tiles,
        )

    memory_ws_per_tile = max(0.0, cell_area - per_tile_l3)
    memory_summary = MemoryClassSummary(
        num_tiles=total_compute_tiles,
        l3_area_mm2=per_tile_l3,
        total_area_mm2=per_tile_l3,
        pitch_mm=memory_pitch,
        whitespace_per_tile_mm2=memory_ws_per_tile,
        class_whitespace_mm2=memory_ws_per_tile * total_compute_tiles,
    )

    # Mesh rectangle
    mesh_rows = arch.noc.mesh_rows
    mesh_cols = arch.noc.mesh_cols
    physical_cols = 2 * mesh_cols  # compute + memory pairs per row
    mesh_width = physical_cols * unified_pitch
    mesh_height = mesh_rows * unified_pitch

    # Memory controllers around the periphery
    phy_blocks = [
        (n, a, c) for (n, a, c) in other_blocks if "phy" in n.lower()
    ]
    phy_total_area = sum(a for _, a, _ in phy_blocks)
    distinct_phys = len(phy_blocks)
    num_controllers = (
        distinct_phys if distinct_phys > 1
        else _DEFAULT_NUM_MEMORY_CONTROLLERS
    )
    controller_side = (
        math.sqrt(phy_total_area / num_controllers)
        if num_controllers > 0 and phy_total_area > 0 else 0.0
    )

    # Die envelope: mesh + controller margin on all 4 edges + IO ring
    edge_pad = _IO_RING_THICKNESS_MM + controller_side
    die_width = mesh_width + 2 * edge_pad
    die_height = mesh_height + 2 * edge_pad
    mesh_origin_x = edge_pad
    mesh_origin_y = edge_pad

    blocks: list[ArchTile] = []

    # IO ring (4 thin edge segments at the very perimeter)
    blocks.extend(_arch_place_io_ring(die_width, die_height, other_blocks))

    # Compute + memory tiles (column-pair checkerboard)
    blocks.extend(_arch_place_checkerboard(
        arch, mesh_origin_x, mesh_origin_y, unified_pitch,
        per_tile_l2, per_tile_l3, compute_pe_areas,
    ))

    # Memory controllers around the periphery
    blocks.extend(_arch_place_memory_controllers(
        num_controllers, controller_side, phy_blocks,
        die_width, die_height,
        mesh_origin_x, mesh_origin_y, mesh_width, mesh_height,
    ))

    # Control logic in bottom-left corner of the mesh
    blocks.extend(_arch_place_control(
        other_blocks, mesh_origin_x, mesh_origin_y, die_height
    ))

    # What-if estimates: re-compute die area assuming every compute
    # tile is class X. Periphery scales the same; mesh resizes around
    # the new unified pitch.
    what_if = _arch_what_if_estimates(
        compute_pitches, memory_pitch, total_compute_tiles,
        mesh_rows, mesh_cols, edge_pad,
    )

    return ArchitecturalFloorplan(
        sku_id=sku.id,
        sku_name=sku.name,
        die_width_mm=die_width,
        die_height_mm=die_height,
        blocks=blocks,
        compute_summaries=compute_summaries,
        memory_summary=memory_summary,
        num_memory_controllers=num_controllers,
        unified_pitch_mm=unified_pitch,
        what_if=what_if,
        notes=(
            f"Architectural v1: {mesh_rows}x{mesh_cols} compute + "
            f"{mesh_rows}x{mesh_cols} memory in column-pair checkerboard "
            f"@ {unified_pitch:.3f} mm pitch; {num_controllers} memory "
            f"controllers around periphery."
        ),
    )


def _arch_place_io_ring(
    die_width_mm: float,
    die_height_mm: float,
    other_blocks: list[tuple[str, float, CircuitClass]],
) -> list[ArchTile]:
    io_area = sum(
        a for n, a, _ in other_blocks
        if "io" in n.lower() and "phy" not in n.lower()
    )
    if io_area <= 0:
        return []
    perimeter = 2 * (die_width_mm + die_height_mm)
    if perimeter <= 0:
        return []
    thickness = min(io_area / perimeter, _IO_RING_THICKNESS_MM)
    return [
        ArchTile(
            name="io_ring_bottom", role=TileRole.IO_PAD,
            x_mm=0.0, y_mm=0.0,
            width_mm=die_width_mm, height_mm=thickness,
        ),
        ArchTile(
            name="io_ring_top", role=TileRole.IO_PAD,
            x_mm=0.0, y_mm=die_height_mm - thickness,
            width_mm=die_width_mm, height_mm=thickness,
        ),
        ArchTile(
            name="io_ring_left", role=TileRole.IO_PAD,
            x_mm=0.0, y_mm=thickness,
            width_mm=thickness, height_mm=die_height_mm - 2 * thickness,
        ),
        ArchTile(
            name="io_ring_right", role=TileRole.IO_PAD,
            x_mm=die_width_mm - thickness, y_mm=thickness,
            width_mm=thickness, height_mm=die_height_mm - 2 * thickness,
        ),
    ]


def _arch_place_checkerboard(
    arch,
    origin_x: float,
    origin_y: float,
    pitch: float,
    per_tile_l2: float,
    per_tile_l3: float,
    compute_pe_areas: dict[str, float],
) -> list[ArchTile]:
    """Lay out compute + memory tiles in a column-pair checkerboard.

    Pattern (every row identical):
        [C M] [C M] [C M] [C M] ...
    Each row has ``mesh_cols`` compute + ``mesh_cols`` memory tiles.
    Compute classes assigned in row-major order over compute positions.
    """
    mesh_rows = arch.noc.mesh_rows
    mesh_cols = arch.noc.mesh_cols

    # Tile-class assignment for compute positions (row-major)
    compute_assignment: list[KPUTileSpec] = []
    for tile in arch.tiles:
        compute_assignment.extend([tile] * tile.num_tiles)
    while len(compute_assignment) < mesh_rows * mesh_cols:
        compute_assignment.append(arch.tiles[-1])
    compute_assignment = compute_assignment[: mesh_rows * mesh_cols]

    blocks: list[ArchTile] = []
    compute_idx = 0
    for r in range(mesh_rows):
        for c_pair in range(mesh_cols):
            # Compute tile (left half of the pair)
            tile = compute_assignment[compute_idx]
            compute_idx += 1
            blocks.append(ArchTile(
                name=f"compute[{r},{c_pair}]",
                role=TileRole.COMPUTE,
                x_mm=origin_x + (2 * c_pair) * pitch,
                y_mm=origin_y + r * pitch,
                width_mm=pitch,
                height_mm=pitch,
                tile_class=tile.tile_type,
                pe_area_mm2=compute_pe_areas.get(tile.tile_type, 0.0),
                l2_area_mm2=per_tile_l2,
            ))
            # Memory tile (right half of the pair)
            blocks.append(ArchTile(
                name=f"memory[{r},{c_pair}]",
                role=TileRole.MEMORY,
                x_mm=origin_x + (2 * c_pair + 1) * pitch,
                y_mm=origin_y + r * pitch,
                width_mm=pitch,
                height_mm=pitch,
                l3_area_mm2=per_tile_l3,
            ))
    return blocks


def _arch_place_memory_controllers(
    num: int,
    side: float,
    phy_blocks: list[tuple[str, float, CircuitClass]],
    die_w: float, die_h: float,
    mesh_x: float, mesh_y: float,
    mesh_w: float, mesh_h: float,
) -> list[ArchTile]:
    """Distribute ``num`` controllers around the perimeter.

    Walk: top edge first, then right, bottom, left, in order. Each edge
    gets ``num // 4`` plus a remainder. Controllers sit just inside
    the IO ring, hugging the mesh on the outer side.
    """
    if num <= 0 or side <= 0:
        return []
    # Even split with remainder absorbed by earlier edges
    base = num // 4
    rem = num % 4
    counts = [base + (1 if i < rem else 0) for i in range(4)]  # T, R, B, L
    blocks: list[ArchTile] = []
    idx = 0

    def make(name: str, x: float, y: float) -> ArchTile:
        return ArchTile(
            name=name, role=TileRole.MEMORY_CONTROLLER,
            x_mm=x, y_mm=y, width_mm=side, height_mm=side,
            notes=(
                f"PHY block #{idx} of {num} "
                f"(area {side * side:.2f} mm^2)"
            ),
        )

    # Top edge: hug above the mesh
    top_y = mesh_y + mesh_h
    for i in range(counts[0]):
        x = mesh_x + (i + 0.5) / counts[0] * mesh_w - side / 2
        blocks.append(make(f"mc_top_{i}", x, top_y))
        idx += 1
    # Right edge: hug right of the mesh
    right_x = mesh_x + mesh_w
    for i in range(counts[1]):
        y = mesh_y + (i + 0.5) / counts[1] * mesh_h - side / 2
        blocks.append(make(f"mc_right_{i}", right_x, y))
        idx += 1
    # Bottom edge: hug below the mesh
    bottom_y = mesh_y - side
    for i in range(counts[2]):
        x = mesh_x + (i + 0.5) / counts[2] * mesh_w - side / 2
        blocks.append(make(f"mc_bottom_{i}", x, bottom_y))
        idx += 1
    # Left edge: hug left of the mesh
    left_x = mesh_x - side
    for i in range(counts[3]):
        y = mesh_y + (i + 0.5) / counts[3] * mesh_h - side / 2
        blocks.append(make(f"mc_left_{i}", left_x, y))
        idx += 1
    return blocks


def _arch_place_control(
    other_blocks: list[tuple[str, float, CircuitClass]],
    mesh_origin_x: float, mesh_origin_y: float, die_height: float,
) -> list[ArchTile]:
    ctrl_blocks = [
        (n, a, c) for n, a, c in other_blocks if "control" in n.lower()
    ]
    if not ctrl_blocks:
        return []
    total_area = sum(a for _, a, _ in ctrl_blocks)
    side = math.sqrt(total_area) if total_area > 0 else 0.0
    side = min(side, die_height * _CONTROL_CORNER_FRACTION)
    if side <= 0:
        return []
    return [ArchTile(
        name="control_logic", role=TileRole.CONTROL,
        x_mm=mesh_origin_x, y_mm=mesh_origin_y,
        width_mm=side, height_mm=side,
        notes=f"Aggregates {len(ctrl_blocks)} control block(s)",
    )]


def _arch_what_if_estimates(
    compute_pitches: dict[str, float],
    memory_pitch: float,
    total_compute_tiles: int,
    mesh_rows: int,
    mesh_cols: int,
    edge_pad: float,
) -> list[WhatIfDieEstimate]:
    """One 'all-tiles-of-class-X' die estimate per compute class.

    Periphery (IO ring + controller margin) is held constant -- the
    interesting axis is how mesh size + whitespace move with the
    chosen unified pitch.
    """
    out: list[WhatIfDieEstimate] = []
    for class_name, class_pitch in compute_pitches.items():
        unified = max(class_pitch, memory_pitch)
        cell_area = unified * unified
        physical_cols = 2 * mesh_cols
        mesh_w = physical_cols * unified
        mesh_h = mesh_rows * unified
        mesh_area = mesh_w * mesh_h
        # Whitespace inside compute cells + memory cells
        compute_used = (class_pitch * class_pitch) * total_compute_tiles
        memory_used = (memory_pitch * memory_pitch) * total_compute_tiles
        whitespace = max(0.0, mesh_area - compute_used - memory_used)
        die_w = mesh_w + 2 * edge_pad
        die_h = mesh_h + 2 * edge_pad
        die_area = die_w * die_h
        out.append(WhatIfDieEstimate(
            tile_class=class_name,
            unified_pitch_mm=unified,
            mesh_area_mm2=mesh_area,
            die_area_mm2=die_area,
            whitespace_mm2=whitespace,
            whitespace_fraction=(
                whitespace / mesh_area if mesh_area > 0 else 0.0
            ),
        ))
    return out
