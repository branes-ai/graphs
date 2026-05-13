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

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


_logger = logging.getLogger(__name__)

from embodied_schemas.kpu import KPUEntry, KPUTileSpec
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from .compute_product_loader import kpu_entry_to_compute_product
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

    # Forward-adapt to ComputeProduct: silicon_math has migrated to the
    # unified schema, but silicon_floorplan still takes KPUEntry from its
    # callers. Bridge until silicon_floorplan migrates in a follow-up PR.
    cp = kpu_entry_to_compute_product(sku)
    for block in sku.silicon_bin.blocks:
        try:
            ba = resolve_block_area(block, cp, node)
        except SiliconMathError as exc:
            # Don't silently drop -- log so the missing area surfaces in
            # the floorplan output, but keep going so a single bad
            # silicon_bin block doesn't kill the whole visualizer for
            # an otherwise-valid SKU. The GEOMETRY validators (which
            # care about correctness) call resolve_block_area
            # independently and will surface the same failure as a
            # finding.
            _logger.warning(
                "silicon_floorplan: skipping block %r in sku %r: %s",
                block.name, sku.id, exc,
            )
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

    Uses ``_IO_RING_THICKNESS_MM`` (typical pad-pitch depth) regardless
    of silicon_bin io_pads area -- pad count and pad-ring depth are
    different concerns. ``derive_kpu_floorplan`` reserves
    ``2 * _IO_RING_THICKNESS_MM`` of die margin on each axis assuming
    this same thickness; using the area-derived value here would
    create a moat between the ring and the rest of the die.
    """
    io_area = sum(
        area for name, area, _ in other_blocks
        if "io" in name.lower() and "phy" not in name.lower()
    )
    if io_area <= 0:
        return []
    thickness = _IO_RING_THICKNESS_MM

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
    if not arch.tiles:
        raise ValueError(
            f"sku {sku.id!r}: kpu_architecture.tiles is empty; cannot "
            f"place a compute mesh"
        )
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
class MemoryChannel:
    """Off-die DRAM channel + connectivity to its driving memory
    controller.

    Drawn just OUTSIDE the die boundary by the visualizer so the
    architect can see (a) which controller drives which channel and
    (b) how the channel-count + I/O width affect the overall die
    geometry. Dimensions match the controller PHY block they pair
    with; placement is exterior-aligned.
    """

    channel_id: int
    """0..N-1 over all channels, walking the perimeter clockwise from
    top-center."""

    memory_type: str
    """e.g., 'lpddr5', 'hbm3' -- from the KPU memory spec."""

    width_bits: int
    """Per-channel I/O width (``memory_bus_bits / memory_controllers``).
    LPDDR5 typically 16, HBM3 stack 1024 (or 512 with two halves)."""

    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    """Position + dimensions of the channel rectangle. Sits adjacent
    to its controller on the outside of the die."""

    controller_name: str
    """Name of the corresponding ``ArchTile`` of role
    ``MEMORY_CONTROLLER`` inside the die."""

    edge: str
    """'top', 'right', 'bottom', or 'left' -- which die edge this
    channel sits beyond."""


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

    memory_channels: list[MemoryChannel] = field(default_factory=list)
    """Off-die DRAM channels paired 1:1 with the memory controllers
    inside the die. Empty if the SKU's memory spec doesn't expose
    channel info."""

    memory_type: str = ""
    """e.g., 'lpddr5', 'hbm3' from the SKU memory spec."""

    per_channel_width_bits: int = 0
    """Per-channel I/O width."""

    per_channel_phy_area_mm2: float = 0.0
    """Heuristic PHY area per channel; total MC area = num x this."""

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

# Default memory-controller arity when the SKU memory spec doesn't
# expose a distinct count. The architectural derivation prefers
# ``sku.kpu_architecture.memory.memory_controllers`` when present.
_DEFAULT_NUM_MEMORY_CONTROLLERS = 4


# ---------------------------------------------------------------------------
# Per-channel PHY heuristics (rectangular MCs)
# ---------------------------------------------------------------------------
#
# Memory-controller PHY blocks are NOT proportional to compute-mesh
# size -- they're proportional to the number + width of memory channels
# they drive, and shaped by the I/O bump pattern of the package
# interface:
#
#   * LPDDR{4,5,5X}: long narrow stripes parallel to the bump line
#     (ball pitch sets the long edge; thin perpendicular dimension).
#     Aspect ratio ~5:1.
#   * DDR{4,5}: similar to LPDDR but slightly squarer (4:1).
#   * GDDR6/6X: 3:1, sits next to GDDR packages on PCB.
#   * HBM{2,2e,3,3e}: nearly square (1.5:1) microbump arrays sitting
#     under the HBM stack via interposer.
#
# Per-channel area scales with channel width. Coefficients land in the
# published-PHY-area ballpark for current-gen ~16nm logic; refine in
# Stage 8c after PDK calibration.

_PER_CHANNEL_PHY_AREA_MM2: dict = {
    "lpddr4":  lambda w: 0.5 + 0.06 * w,    # 16b -> 1.46 mm^2
    "lpddr4x": lambda w: 0.5 + 0.06 * w,
    "lpddr5":  lambda w: 0.6 + 0.07 * w,    # 16b -> 1.72, 32b -> 2.84
    "lpddr5x": lambda w: 0.6 + 0.07 * w,
    "ddr4":    lambda w: 1.0 + 0.05 * w,    # 64b -> 4.20 mm^2
    "ddr5":    lambda w: 1.2 + 0.06 * w,    # 64b -> 5.04
    "gddr6":   lambda w: 0.8 + 0.06 * w,    # 32b -> 2.72
    "gddr6x":  lambda w: 0.8 + 0.06 * w,
    "hbm2":    lambda w: 1.5 + 0.012 * w,   # 1024b -> 13.79 (per stack)
    "hbm2e":   lambda w: 1.5 + 0.012 * w,
    "hbm3":    lambda w: 1.5 + 0.014 * w,   # 1024b -> 15.85, 512b -> 8.67
    "hbm3e":   lambda w: 1.5 + 0.014 * w,
}

_PHY_ASPECT_RATIO: dict = {
    "lpddr4":  5.0,
    "lpddr4x": 5.0,
    "lpddr5":  5.0,
    "lpddr5x": 5.0,
    "ddr4":    4.0,
    "ddr5":    4.0,
    "gddr6":   3.0,
    "gddr6x":  3.0,
    "hbm2":    1.5,
    "hbm2e":   1.5,
    "hbm3":    1.5,
    "hbm3e":   1.5,
}


def _per_channel_phy_dims(memory_type: str, width_bits: int) -> tuple[float, float, float]:
    """Per-channel PHY block dimensions (long_dim, short_dim, area_mm2).

    long_dim is the dimension parallel to the I/O bump edge (sits along
    the die edge); short_dim is perpendicular. For LPDDR
    long >> short; for HBM long ~ short.
    """
    type_key = memory_type.lower()
    area_fn = _PER_CHANNEL_PHY_AREA_MM2.get(
        type_key, lambda w: 1.0 + 0.05 * w
    )
    aspect = _PHY_ASPECT_RATIO.get(type_key, 3.0)
    area = area_fn(width_bits) if width_bits > 0 else 0.0
    if area <= 0 or aspect <= 0:
        return (0.0, 0.0, 0.0)
    long_dim = math.sqrt(area * aspect)
    short_dim = math.sqrt(area / aspect)
    return (long_dim, short_dim, area)


def derive_kpu_architectural_floorplan(
    sku: KPUEntry, node: ProcessNodeEntry
) -> ArchitecturalFloorplan:
    """Produce the architectural-role floorplan for a KPU SKU.

    Layout (heuristic v2 -- corrected from v1):

    1. Per compute tile class: pitch = sqrt(per_tile_PE_area +
       per_tile_L2_area). Memory tile pitch = sqrt(per_tile_L3_area).
    2. Unified pitch = max(max_compute_pitch, memory_pitch).
    3. **True 2D checkerboard**: physical mesh is
       ``mesh_rows x (2 * mesh_cols)`` cells, with compute placed
       where ``(row + col) % 2 == 0`` and memory where odd. Every
       interior compute tile has 4 memory neighbours (N/S/E/W) and
       vice versa.
    4. **Memory controllers** = exactly ``memory.memory_controllers``
       PHY blocks, sized from ``memory_type`` + ``per_channel_width``
       (LPDDR is long narrow stripes, HBM is squarer microbump arrays;
       see ``_per_channel_phy_dims``). Distributed as an edge ring;
       MC area is independent of mesh size, scales only with channel
       count + width.
    5. **DRAM channels** placed *outside* the die boundary, each
       paired 1:1 with its driving controller -- gives the architect
       a visual of MC <-> channel connectivity and channel-count
       impact on die geometry.
    6. IO ring + control corner unchanged from v1.
    """
    arch = sku.kpu_architecture
    mem = arch.memory
    pe_by_tile_type, chip_l2_area, chip_l3_area, other_blocks = (
        _classify_silicon_bin_blocks(sku, node)
    )
    total_compute_tiles = arch.total_tiles

    # Per-tile L2 / L3
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

    max_compute_pitch = max(compute_pitches.values(), default=0.0)
    unified_pitch = max(max_compute_pitch, memory_pitch)
    cell_area = unified_pitch * unified_pitch

    # Per-class summaries
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

    # Physical mesh rectangle: 2 * mesh_cols wide so total cells = 2N
    # and a true 2D checkerboard places exactly N compute + N memory.
    mesh_rows = arch.noc.mesh_rows
    mesh_cols = arch.noc.mesh_cols
    physical_cols = 2 * mesh_cols
    mesh_width = physical_cols * unified_pitch
    mesh_height = mesh_rows * unified_pitch

    # Memory configuration from the SKU schema (preferred) -- falls
    # back to the silicon_bin PHY blocks + a default 4 if unset.
    memory_type = mem.memory_type.value if mem.memory_type else "lpddr5"
    num_controllers = (
        mem.memory_controllers if mem.memory_controllers > 0
        else _DEFAULT_NUM_MEMORY_CONTROLLERS
    )
    bus_bits = mem.memory_bus_bits if mem.memory_bus_bits > 0 else 0
    per_channel_width_bits = (
        bus_bits // num_controllers if num_controllers > 0 else 0
    )
    long_dim, short_dim, per_channel_phy_area = _per_channel_phy_dims(
        memory_type, per_channel_width_bits
    )

    # Distribute channels across edges (equal split; remainder
    # absorbed top -> right -> bottom -> left). Used both to size the
    # die (so MCs fit their edge) and to lay them out below.
    base_per_edge = num_controllers // 4
    rem = num_controllers % 4
    edge_counts_for_size = [
        base_per_edge + (1 if i < rem else 0) for i in range(4)
    ]
    # Required edge length (x for top/bottom, y for left/right) =
    # MCs-on-that-edge x long_dim. If the mesh edge is shorter than
    # required, expand the die so the I/O perimeter actually fits --
    # MC area is bump-pitch-driven, not mesh-driven, so the die
    # follows the larger constraint.
    required_horizontal_edge = (
        max(edge_counts_for_size[0], edge_counts_for_size[2]) * long_dim
    )
    required_vertical_edge = (
        max(edge_counts_for_size[1], edge_counts_for_size[3]) * long_dim
    )
    # Corner reservation: vertical MCs claim full die_h (corners
    # belong to them); horizontal MCs are inset by short_dim on each
    # x-side so they don't visually overlap vertical MCs in the corners.
    # That means horizontal MCs need extra ``2 * short_dim`` of inner-x
    # to fit; vertical MCs use the full inner-y.
    extra_x = max(0.0, required_horizontal_edge + 2 * short_dim - mesh_width)
    extra_y = max(0.0, required_vertical_edge - mesh_height)

    # Die envelope: max(mesh, I/O perimeter) + IO ring + MC inset
    mc_inset = short_dim
    edge_pad = _IO_RING_THICKNESS_MM + mc_inset
    die_width = mesh_width + extra_x + 2 * edge_pad
    die_height = mesh_height + extra_y + 2 * edge_pad
    # Mesh centered inside the (possibly-larger) inner area
    mesh_origin_x = edge_pad + extra_x / 2
    mesh_origin_y = edge_pad + extra_y / 2

    blocks: list[ArchTile] = []

    # IO ring
    blocks.extend(_arch_place_io_ring(die_width, die_height, other_blocks))

    # Memory controllers (rectangular, sized by channel I/O width).
    # Placed BEFORE the mesh so that the rasterizer's later-block-wins
    # rule keeps mesh tiles on top at the mesh/MC boundary -- otherwise
    # pixel quantization at shared edges paints the top mesh row with
    # MC glyphs.
    mc_blocks, channel_blocks = _arch_place_memory_subsystem(
        num_channels=num_controllers,
        memory_type=memory_type,
        channel_width_bits=per_channel_width_bits,
        long_dim=long_dim, short_dim=short_dim,
        mesh_origin_x=mesh_origin_x, mesh_origin_y=mesh_origin_y,
        mesh_width=mesh_width, mesh_height=mesh_height,
        die_width=die_width, die_height=die_height,
    )
    blocks.extend(mc_blocks)

    # Compute + memory tiles -- TRUE 2D checkerboard (drawn last so
    # mesh wins shared pixels at the mesh/MC boundary).
    blocks.extend(_arch_place_checkerboard(
        arch, mesh_origin_x, mesh_origin_y, unified_pitch,
        per_tile_l2, per_tile_l3, compute_pe_areas,
    ))

    # Control logic (bottom-left corner gap, outside the mesh)
    blocks.extend(_arch_place_control(
        other_blocks,
        mesh_origin_x=mesh_origin_x, mesh_origin_y=mesh_origin_y,
        short_dim=short_dim, die_height=die_height,
    ))

    # What-if estimates (periphery-aware: edge_pad held constant)
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
        memory_channels=channel_blocks,
        memory_type=memory_type,
        per_channel_width_bits=per_channel_width_bits,
        per_channel_phy_area_mm2=per_channel_phy_area,
        notes=(
            f"Architectural v2: {mesh_rows}x{mesh_cols} mesh as true 2D "
            f"checkerboard ({physical_cols}x{mesh_rows} cells) @ "
            f"{unified_pitch:.3f} mm pitch; {num_controllers} {memory_type} "
            f"channels @ {per_channel_width_bits}b each "
            f"({per_channel_phy_area:.2f} mm^2 PHY/channel, "
            f"{long_dim:.2f}x{short_dim:.2f} mm) distributed as edge ring."
        ),
    )


def _arch_place_io_ring(
    die_width_mm: float,
    die_height_mm: float,
    other_blocks: list[tuple[str, float, CircuitClass]],
) -> list[ArchTile]:
    """Pad ring around the perimeter.

    Ring thickness is fixed at ``_IO_RING_THICKNESS_MM`` (typical pad-
    pitch depth for current packages) regardless of silicon_bin
    io_pads area -- the pad count drives how many pads fit, not how
    deep the ring sits. Earlier code scaled thickness by
    ``io_area/perimeter``, which collapsed the ring below visual
    resolution when io_pads was small (1 character cell renders only
    when the rect is at least mm_per_col wide).
    """
    io_area = sum(
        a for n, a, _ in other_blocks
        if "io" in n.lower() and "phy" not in n.lower()
    )
    if io_area <= 0:
        return []
    thickness = _IO_RING_THICKNESS_MM
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
    """Lay out compute + memory tiles in a TRUE 2D checkerboard.

    Physical mesh: ``mesh_rows`` rows x ``2 * mesh_cols`` cols. Cell
    is COMPUTE iff ``(row + col) % 2 == 0``, MEMORY otherwise. Every
    interior compute tile has 4 memory neighbours (N/S/E/W) and vice
    versa -- the L3 scratchpad is reachable from every direction.

    Pattern (rows alternate offset by one):
        C M C M C M C M C M C M ...
        M C M C M C M C M C M C ...
        C M C M C M C M C M C M ...
        M C M C M C M C M C M C ...

    Compute tile-class assignment: row-major walk over the compute
    positions only, drawing from ``arch.tiles`` in declaration order.
    """
    if not arch.tiles:
        raise ValueError(
            "kpu_architecture.tiles is empty; architectural floorplan "
            "needs at least one tile class"
        )
    mesh_rows = arch.noc.mesh_rows
    mesh_cols = arch.noc.mesh_cols
    physical_cols = 2 * mesh_cols
    expected_compute = mesh_rows * mesh_cols  # half the physical cells

    # Compute-class assignment (in declaration order)
    compute_assignment: list[KPUTileSpec] = []
    for tile in arch.tiles:
        compute_assignment.extend([tile] * tile.num_tiles)
    while len(compute_assignment) < expected_compute:
        compute_assignment.append(arch.tiles[-1])
    compute_assignment = compute_assignment[:expected_compute]

    blocks: list[ArchTile] = []
    compute_idx = 0
    for r in range(mesh_rows):
        for c in range(physical_cols):
            x = origin_x + c * pitch
            y = origin_y + r * pitch
            if (r + c) % 2 == 0:
                tile = compute_assignment[compute_idx]
                compute_idx += 1
                blocks.append(ArchTile(
                    name=f"compute[{r},{c}]",
                    role=TileRole.COMPUTE,
                    x_mm=x, y_mm=y, width_mm=pitch, height_mm=pitch,
                    tile_class=tile.tile_type,
                    pe_area_mm2=compute_pe_areas.get(tile.tile_type, 0.0),
                    l2_area_mm2=per_tile_l2,
                ))
            else:
                blocks.append(ArchTile(
                    name=f"memory[{r},{c}]",
                    role=TileRole.MEMORY,
                    x_mm=x, y_mm=y, width_mm=pitch, height_mm=pitch,
                    l3_area_mm2=per_tile_l3,
                ))
    return blocks


def _arch_place_memory_subsystem(
    *,
    num_channels: int,
    memory_type: str,
    channel_width_bits: int,
    long_dim: float,
    short_dim: float,
    mesh_origin_x: float, mesh_origin_y: float,
    mesh_width: float, mesh_height: float,
    die_width: float, die_height: float,
) -> tuple[list[ArchTile], list[MemoryChannel]]:
    """Place rectangular memory controllers + their off-die channels.

    Walks the perimeter clockwise from top-center, distributing
    ``num_channels`` controllers as an edge ring. Each MC has the
    long dimension parallel to its edge (matching the I/O bump
    pattern), short dimension perpendicular. The corresponding DRAM
    channel sits just *outside* the die boundary, mirroring the MC's
    shape so MC <-> channel association is visually obvious.

    Distribution: equal split across 4 edges; remainder absorbed top
    -> right -> bottom -> left so high channel counts (16 LPDDR ch
    on T256) get a balanced ring. Future Stage 8c work: bias HBM to
    interposer-stack edges only.
    """
    if num_channels <= 0 or long_dim <= 0 or short_dim <= 0:
        return ([], [])

    base = num_channels // 4
    rem = num_channels % 4
    edge_counts = [base + (1 if i < rem else 0) for i in range(4)]

    mc_blocks: list[ArchTile] = []
    channel_blocks: list[MemoryChannel] = []
    chan_id = 0

    # Channel rectangle sits short_dim outside the die, mirroring the
    # MC's cross-section so MC<->channel alignment is visually obvious.
    channel_offset = short_dim

    # Inner-edge ranges (MC distribution domain). The die may have been
    # expanded to fit the I/O perimeter; MCs distribute over the FULL
    # inner extent (mesh + any I/O-driven extras), not just the mesh.
    #
    # Corner reservation: vertical MCs (left/right) use the full inner
    # y-range and claim the corner regions. Horizontal MCs (top/bottom)
    # inset by short_dim on each x-side so they don't visually overlap
    # vertical MCs in the corners.
    edge_pad_inner = _IO_RING_THICKNESS_MM + short_dim    # = edge_pad
    inner_x_lo_horiz = edge_pad_inner + short_dim   # inset for corner
    inner_x_hi_horiz = die_width - edge_pad_inner - short_dim
    inner_w_horiz = inner_x_hi_horiz - inner_x_lo_horiz

    inner_y_lo_vert = edge_pad_inner   # vertical MCs span full inner
    inner_y_hi_vert = die_height - edge_pad_inner
    inner_h_vert = inner_y_hi_vert - inner_y_lo_vert

    def _ch_note(ch_id: int) -> str:
        return f"channel {ch_id} ({memory_type} {channel_width_bits}b)"

    # Top edge: MC anchored to die top (just inside top IO ring), NOT
    # to mesh top -- otherwise when extra_y > 0 the MC floats inside
    # the die instead of hugging the edge. Same for the other 3 edges
    # below.
    mc_top_y = die_height - _IO_RING_THICKNESS_MM - short_dim
    mc_bottom_y = _IO_RING_THICKNESS_MM
    mc_right_x = die_width - _IO_RING_THICKNESS_MM - short_dim
    mc_left_x = _IO_RING_THICKNESS_MM

    # Top edge: MC sits just inside the top IO ring; width = long_dim,
    # height = short_dim. Inset from corners so vertical MCs claim them.
    if edge_counts[0] > 0 and inner_w_horiz > 0:
        n = edge_counts[0]
        for i in range(n):
            x = inner_x_lo_horiz + (i + 0.5) / n * inner_w_horiz - long_dim / 2
            mc_y = mc_top_y
            mc_name = f"mc_top_ch{chan_id}"
            mc_blocks.append(ArchTile(
                name=mc_name, role=TileRole.MEMORY_CONTROLLER,
                x_mm=x, y_mm=mc_y,
                width_mm=long_dim, height_mm=short_dim,
                notes=_ch_note(chan_id),
            ))
            channel_blocks.append(MemoryChannel(
                channel_id=chan_id, memory_type=memory_type,
                width_bits=channel_width_bits,
                x_mm=x, y_mm=die_height,
                width_mm=long_dim, height_mm=channel_offset,
                controller_name=mc_name, edge="top",
            ))
            chan_id += 1

    # Right edge: MC anchored to die right (just inside right IO ring);
    # width = short_dim, height = long_dim. Spans full inner y
    # including corners.
    if edge_counts[1] > 0 and inner_h_vert > 0:
        n = edge_counts[1]
        for i in range(n):
            y = inner_y_lo_vert + (i + 0.5) / n * inner_h_vert - long_dim / 2
            mc_x = mc_right_x
            mc_name = f"mc_right_ch{chan_id}"
            mc_blocks.append(ArchTile(
                name=mc_name, role=TileRole.MEMORY_CONTROLLER,
                x_mm=mc_x, y_mm=y,
                width_mm=short_dim, height_mm=long_dim,
                notes=_ch_note(chan_id),
            ))
            channel_blocks.append(MemoryChannel(
                channel_id=chan_id, memory_type=memory_type,
                width_bits=channel_width_bits,
                x_mm=die_width, y_mm=y,
                width_mm=channel_offset, height_mm=long_dim,
                controller_name=mc_name, edge="right",
            ))
            chan_id += 1

    # Bottom edge: MC anchored to die bottom (just above bottom IO
    # ring). Inset from corners.
    if edge_counts[2] > 0 and inner_w_horiz > 0:
        n = edge_counts[2]
        for i in range(n):
            x = inner_x_lo_horiz + (i + 0.5) / n * inner_w_horiz - long_dim / 2
            mc_y = mc_bottom_y
            mc_name = f"mc_bottom_ch{chan_id}"
            mc_blocks.append(ArchTile(
                name=mc_name, role=TileRole.MEMORY_CONTROLLER,
                x_mm=x, y_mm=mc_y,
                width_mm=long_dim, height_mm=short_dim,
                notes=_ch_note(chan_id),
            ))
            channel_blocks.append(MemoryChannel(
                channel_id=chan_id, memory_type=memory_type,
                width_bits=channel_width_bits,
                x_mm=x, y_mm=-channel_offset,
                width_mm=long_dim, height_mm=channel_offset,
                controller_name=mc_name, edge="bottom",
            ))
            chan_id += 1

    # Left edge: MC anchored to die left (just inside left IO ring).
    # Spans full inner y including corners.
    if edge_counts[3] > 0 and inner_h_vert > 0:
        n = edge_counts[3]
        for i in range(n):
            y = inner_y_lo_vert + (i + 0.5) / n * inner_h_vert - long_dim / 2
            mc_x = mc_left_x
            mc_name = f"mc_left_ch{chan_id}"
            mc_blocks.append(ArchTile(
                name=mc_name, role=TileRole.MEMORY_CONTROLLER,
                x_mm=mc_x, y_mm=y,
                width_mm=short_dim, height_mm=long_dim,
                notes=_ch_note(chan_id),
            ))
            channel_blocks.append(MemoryChannel(
                channel_id=chan_id, memory_type=memory_type,
                width_bits=channel_width_bits,
                x_mm=-channel_offset, y_mm=y,
                width_mm=channel_offset, height_mm=long_dim,
                controller_name=mc_name, edge="left",
            ))
            chan_id += 1

    return (mc_blocks, channel_blocks)


def _arch_place_control(
    other_blocks: list[tuple[str, float, CircuitClass]],
    *,
    mesh_origin_x: float,
    mesh_origin_y: float,
    short_dim: float,
    die_height: float,
) -> list[ArchTile]:
    """Place control logic in the bottom-left corner GAP.

    With the v2 layout the bottom-left corner has a small unclaimed
    region: the bottom IO ring is at y < IO, the bottom horizontal MC
    starts at x = IO + 2*short_dim, and the left vertical MC starts at
    y = IO + short_dim. That leaves a rectangle of size
    ``2*short_dim x short_dim`` at ``(IO, IO)`` available for control,
    OUTSIDE the mesh. The previous version sat inside the mesh and
    overlapped the bottom-left compute tile.
    """
    ctrl_blocks = [
        (n, a, c) for n, a, c in other_blocks if "control" in n.lower()
    ]
    if not ctrl_blocks:
        return []
    total_area = sum(a for _, a, _ in ctrl_blocks)
    if total_area <= 0:
        return []
    # Available corner gap: max width = 2*short_dim, max height = short_dim.
    # Scale the silicon_bin's control block to fit inside this envelope
    # with the same aspect ratio.
    gap_w = 2 * short_dim
    gap_h = short_dim
    if gap_w <= 0 or gap_h <= 0:
        return []
    # Square-ish control block (side derived from area, capped by gap)
    side = math.sqrt(total_area)
    width = min(side, gap_w)
    height = min(side, gap_h)
    if width <= 0 or height <= 0:
        return []
    return [ArchTile(
        name="control_logic", role=TileRole.CONTROL,
        x_mm=_IO_RING_THICKNESS_MM,
        y_mm=_IO_RING_THICKNESS_MM,
        width_mm=width, height_mm=height,
        notes=(
            f"Aggregates {len(ctrl_blocks)} control block(s); "
            f"placed in bottom-left corner gap (outside mesh)"
        ),
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
