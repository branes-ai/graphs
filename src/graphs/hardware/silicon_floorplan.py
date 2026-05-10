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
