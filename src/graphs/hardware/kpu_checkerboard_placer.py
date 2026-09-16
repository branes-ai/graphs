"""Assign KPU tiles to checkerboard compute sites (graphs#268 D1).

A uniform KPU needs no placer: every tile is the same size and the same
kind, so laying them out row-major is both correct and the only sensible
answer. A heterogeneous checkerboard does need one. Its classes differ in
footprint (the fixture's VIO core is 2x2 and swallows the memory cells it
covers), some of them want a particular part of the die (an ISP belongs at
the IO edge, next to the MIPI PHYs), and stream-linked classes must end up
next to each other or the link crosses the mesh.

This module answers only the *combinatorial* question -- which class owns
which site -- in site coordinates. Turning sites into millimetre boxes
stays in ``silicon_floorplan``, which knows pitches and die geometry.
Keeping the two apart means the placement can be tested exhaustively on a
small grid without constructing a die.

Two modes, matching ``CheckerboardSpec.placement``:

- **explicit**: the SKU author wrote a ``placement_map``, a grid of
  ``tile_class_id`` strings with ``'.'`` (``SPARE_SITE``) for a spare. The
  placer reads it back, recovering multi-site footprints as rectangles,
  and reports any disagreement with the declared tile counts rather than
  silently re-placing.
- **auto**: a greedy placement in four passes, in the order the plan
  fixes: multi-site footprints first (they are the hardest to fit), then
  classes with a placement affinity, then stream-link and
  ``adjacent_to`` partners, then everything else fills row-major.

Determinism is a requirement, not an accident: the same block must always
produce the same plan, or the golden floorplan snapshot is worthless. Every
candidate-site ordering below therefore ends in a ``(row, col)`` tiebreak,
and every class iteration follows the block's declaration order.

Leaf module: imports embodied-schemas and ``kpu_tile_display`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from embodied_schemas.compute_product import KPUBlock
from embodied_schemas.kpu import SPARE_SITE, TilePlacementAffinity
from embodied_schemas.overlay import NoCOverlayKind

from graphs.hardware import kpu_tile_display as display

Site = Tuple[int, int]

#: 4-adjacency, the mesh's own neighbour relation.
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class PlacementError(ValueError):
    """The block cannot be placed on its declared checkerboard."""


@dataclass(frozen=True)
class TilePlacementResult:
    """One placed tile instance, in site coordinates.

    ``row`` / ``col`` are the top-left site; a 1x1 tile occupies exactly
    that site. ``instance`` distinguishes the tiles of a class and is what
    makes the plan reproducible when a class has many tiles.
    """

    tile_class_id: str
    tile_type: str
    row: int
    col: int
    rows: int = 1
    cols: int = 1
    instance: int = 0

    @property
    def sites(self) -> Tuple[Site, ...]:
        return tuple(
            (self.row + dr, self.col + dc)
            for dr in range(self.rows)
            for dc in range(self.cols)
        )

    @property
    def num_sites(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class SitePlan:
    """Which class owns which compute site."""

    rows: int
    cols: int
    placements: Tuple[TilePlacementResult, ...]
    spare_sites: Tuple[Site, ...]
    #: Cells a multi-site tile's rectangle sits on (geometry).
    covered_memory_cells: Tuple[Site, ...]
    #: The subset of those the tile claims as private state (accounting).
    absorbed_memory_cells: Tuple[Site, ...]
    mode: str  # "explicit" or "auto"
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_sites(self) -> int:
        return self.rows * self.cols

    @property
    def occupied_sites(self) -> int:
        return sum(p.num_sites for p in self.placements)

    def site_owner(self) -> Dict[Site, str]:
        """``{(row, col): tile_class_id}`` for every occupied site."""
        owner: Dict[Site, str] = {}
        for placement in self.placements:
            for site in placement.sites:
                owner[site] = placement.tile_class_id
        return owner

    def sites_of(self, tile_class_id: str) -> Tuple[Site, ...]:
        return tuple(
            site
            for placement in self.placements
            if placement.tile_class_id == tile_class_id
            for site in placement.sites
        )

    def render(self, width: int = 0) -> str:
        """The plan as a grid of short labels, for a CLI or a test.

        Labels are the first characters of each class id, so the grid stays
        readable; ``.`` marks a spare site.
        """
        owner = self.site_owner()
        labels = _short_labels(sorted({p.tile_class_id for p in self.placements}))
        cell_w = width or max([len(v) for v in labels.values()] + [1])
        lines = []
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                cid = owner.get((r, c))
                cells.append((labels[cid] if cid else SPARE_SITE).rjust(cell_w))
            lines.append(" ".join(cells))
        return "\n".join(lines)


def _short_labels(class_ids: Sequence[str]) -> Dict[str, str]:
    """A short, unique, deterministic label per class id."""
    labels: Dict[str, str] = {}
    used: set = set()
    for cid in class_ids:
        # Initials of the underscore-separated parts, e.g. pe_int8_mac_i32
        # -> "pimi"; lengthened until unique.
        parts = [p for p in cid.split("_") if p]
        base = "".join(p[0] for p in parts) or cid[:1]
        label, n = base, 1
        while label in used:
            n += 1
            label = f"{base}{n}"
        used.add(label)
        labels[cid] = label
    return labels


# ---------------------------------------------------------------------------
# Placement inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClassToPlace:
    tile_class_id: str
    tile_type: str
    num_tiles: int
    rows: int
    cols: int
    absorbs_memory_cells: bool
    affinity: TilePlacementAffinity
    adjacent_to: Tuple[str, ...]
    order: int  # declaration order, the final tiebreak

    @property
    def num_sites(self) -> int:
        return self.rows * self.cols


def _classes_to_place(block: KPUBlock) -> List[_ClassToPlace]:
    out = []
    for order, tile in enumerate(block.tiles):
        footprint = getattr(tile, "footprint", None)
        placement = getattr(tile, "placement", None)
        out.append(
            _ClassToPlace(
                tile_class_id=tile.tile_class_id,
                tile_type=tile.tile_type,
                num_tiles=tile.num_tiles,
                rows=footprint.rows if footprint else 1,
                cols=footprint.cols if footprint else 1,
                absorbs_memory_cells=display.absorbs_memory_cells(tile),
                affinity=(
                    placement.affinity if placement else TilePlacementAffinity.ANY
                ),
                adjacent_to=tuple(placement.adjacent_to) if placement else (),
                order=order,
            )
        )
    return out


def stream_link_partners(block: KPUBlock) -> Dict[str, set]:
    """Classes each class is stream-linked to.

    A stream link is a NoC overlay listing its endpoints in pipeline order
    (the fixture's ISP -> SGM -> VIO), so consecutive endpoints are the
    pairs that must end up adjacent. ``placement.adjacent_to`` is the
    explicit hint for the same thing; both feed the placer, so a SKU that
    declares the overlay gets adjacency even without repeating itself.
    """
    partners: Dict[str, set] = {}
    for overlay in block.noc.overlays or []:
        if overlay.kind != NoCOverlayKind.STREAM_LINK:
            continue
        for a, b in zip(overlay.endpoints, overlay.endpoints[1:]):
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)
    for tile in block.tiles:
        placement = getattr(tile, "placement", None)
        for other in (placement.adjacent_to if placement else ()):
            partners.setdefault(tile.tile_class_id, set()).add(other)
            partners.setdefault(other, set()).add(tile.tile_class_id)
    return partners


# ---------------------------------------------------------------------------
# Auto placement
# ---------------------------------------------------------------------------


def _affinity_key(affinity: TilePlacementAffinity, rows: int, cols: int):
    """Order candidate sites by how well they satisfy an affinity.

    The die's edges are not interchangeable, so the two edge affinities
    pick opposite ones: an IO block belongs at the top edge, where the
    floorplan puts the IO pads, and a memory-edge block at the bottom,
    alongside the memory controllers. Both fall back to any border before
    giving up. Every key ends in ``(row, col)`` so the result is stable.
    """
    last_row, last_col = rows - 1, cols - 1

    def border_distance(r: int, c: int) -> int:
        return min(r, c, last_row - r, last_col - c)

    if affinity == TilePlacementAffinity.IO_EDGE:
        return lambda rc: (rc[0], border_distance(*rc), rc[0], rc[1])
    if affinity == TilePlacementAffinity.MEMORY_EDGE:
        return lambda rc: (last_row - rc[0], border_distance(*rc), rc[0], rc[1])
    if affinity == TilePlacementAffinity.CENTER:
        # Chebyshev distance from the grid centre; ties by row then col.
        cr, cc = (rows - 1) / 2, (cols - 1) / 2
        return lambda rc: (max(abs(rc[0] - cr), abs(rc[1] - cc)), rc[0], rc[1])
    return lambda rc: (rc[0], rc[1])


def _fits(free: set, row: int, col: int, height: int, width: int,
          rows: int, cols: int) -> bool:
    if row + height > rows or col + width > cols:
        return False
    return all(
        (row + dr, col + dc) in free
        for dr in range(height)
        for dc in range(width)
    )


def _adjacency_bonus(row: int, col: int, height: int, width: int,
                     wanted: set, owner: Dict[Site, str]) -> int:
    """How many of a candidate footprint's neighbours are already one of
    the classes this one wants to sit next to. Higher is better."""
    if not wanted:
        return 0
    own = {(row + dr, col + dc) for dr in range(height) for dc in range(width)}
    touching = set()
    for r, c in own:
        for dr, dc in _NEIGHBORS:
            neighbour = (r + dr, c + dc)
            if neighbour in own:
                continue
            cid = owner.get(neighbour)
            if cid in wanted:
                touching.add(cid)
    return len(touching)


def _choose_site(
    cls: _ClassToPlace,
    free: set,
    owner: Dict[Site, str],
    partners: Dict[str, set],
    rows: int,
    cols: int,
) -> Optional[Site]:
    """The best free top-left site for one instance of ``cls``.

    Which constraint wins follows the pass the class is in. A class that
    declared an affinity asked to be somewhere specific, so the affinity
    ranks first and adjacency breaks its ties -- an ISP pinned to the IO
    edge must not be dragged inboard by its stream partner. A class with
    no affinity ranks adjacency first, since a link crossing the mesh is
    the expensive mistake. Both end in row-major order.
    """
    wanted = partners.get(cls.tile_class_id, set())
    affinity_key = _affinity_key(cls.affinity, rows, cols)
    has_affinity = cls.affinity != TilePlacementAffinity.ANY
    best = None
    best_key = None
    for row, col in sorted(free):
        if not _fits(free, row, col, cls.rows, cls.cols, rows, cols):
            continue
        adjacency = -_adjacency_bonus(row, col, cls.rows, cls.cols, wanted, owner)
        affinity = affinity_key((row, col))
        key = (affinity, adjacency) if has_affinity else (adjacency, affinity)
        if best_key is None or key < best_key:
            best, best_key = (row, col), key
    return best


def _placement_order(classes: Sequence[_ClassToPlace],
                     partners: Dict[str, set]) -> List[_ClassToPlace]:
    """The four passes, in the order the plan fixes.

    Multi-site footprints go first because they are the hardest to fit and
    a fragmented grid can leave no rectangle for them. Affinities come
    next, so an ISP claims the IO edge before the fabric fills it.
    Stream-link partners follow, placing themselves against whatever is
    already down. Everything else fills row-major.
    """

    def pass_of(cls: _ClassToPlace) -> int:
        if cls.num_sites > 1:
            return 0
        if cls.affinity != TilePlacementAffinity.ANY:
            return 1
        if partners.get(cls.tile_class_id):
            return 2
        return 3

    return sorted(
        classes,
        # Within the footprint pass, the largest first; then declaration
        # order, which is the author's own priority.
        key=lambda c: (pass_of(c), -c.num_sites, c.order),
    )


def _covered_cells(placements: Iterable[TilePlacementResult],
                   absorbing: set) -> Tuple[Tuple[Site, ...], Tuple[Site, ...]]:
    """``(covered, absorbed)`` memory cells.

    The checkerboard pairs each compute site 1:1 with a memory cell, so a
    tile spanning several sites sits on all of their cells -- its own
    included, since its rectangle spans the memory halves between its
    compute halves. Geometry is the same either way, so every site of a
    multi-site tile is *covered*.

    ``absorbs_memory_cells`` is an accounting switch, not a geometric one:
    it says that SRAM is the tile's private state (the fixture's VIO core
    holds its pose graph there) rather than shared L3 the mesh can still
    reach. Only those cells are *absorbed*.
    """
    covered: List[Site] = []
    absorbed: List[Site] = []
    for placement in placements:
        if placement.num_sites <= 1:
            continue
        covered.extend(placement.sites)
        if placement.tile_class_id in absorbing:
            absorbed.extend(placement.sites)
    return tuple(sorted(covered)), tuple(sorted(absorbed))


def _auto_plan(block: KPUBlock, rows: int, cols: int,
               classes: Sequence[_ClassToPlace]) -> SitePlan:
    partners = stream_link_partners(block)
    free = {(r, c) for r in range(rows) for c in range(cols)}
    owner: Dict[Site, str] = {}
    placements: List[TilePlacementResult] = []

    for cls in _placement_order(classes, partners):
        for instance in range(cls.num_tiles):
            site = _choose_site(cls, free, owner, partners, rows, cols)
            if site is None:
                raise PlacementError(
                    f"tile class {cls.tile_class_id!r} instance {instance} "
                    f"({cls.rows}x{cls.cols} sites) does not fit: "
                    f"{len(free)} of {rows * cols} sites free on the "
                    f"{rows}x{cols} checkerboard"
                )
            row, col = site
            placement = TilePlacementResult(
                tile_class_id=cls.tile_class_id,
                tile_type=cls.tile_type,
                row=row, col=col, rows=cls.rows, cols=cls.cols,
                instance=instance,
            )
            placements.append(placement)
            for occupied in placement.sites:
                free.discard(occupied)
                owner[occupied] = cls.tile_class_id

    absorbing = {c.tile_class_id for c in classes if c.absorbs_memory_cells}
    covered, absorbed = _covered_cells(placements, absorbing)
    # Sorted so the plan reads in grid order regardless of the pass that
    # produced each tile; the placement itself is already deterministic.
    return SitePlan(
        rows=rows, cols=cols,
        placements=tuple(sorted(
            placements, key=lambda p: (p.row, p.col, p.tile_class_id)
        )),
        spare_sites=tuple(sorted(free)),
        covered_memory_cells=covered,
        absorbed_memory_cells=absorbed,
        mode="auto",
    )


def _explicit_plan(block: KPUBlock, rows: int, cols: int,
                   classes: Sequence[_ClassToPlace],
                   placement_map: Sequence[Sequence[str]]) -> SitePlan:
    """Read back an author-written placement map.

    The map is the authority: the placer recovers each class's rectangles
    rather than re-placing, so what the author drew is what gets built. A
    disagreement with the declared tile counts is reported as a note, not
    silently corrected -- the ``checkerboard_site_accounting`` validator
    (C4) is where it becomes a finding.
    """
    if len(placement_map) != rows or any(len(r) != cols for r in placement_map):
        shape = f"{len(placement_map)}x{len(placement_map[0]) if placement_map else 0}"
        raise PlacementError(
            f"placement_map is {shape} but compute_sites is {rows}x{cols}"
        )
    by_id = {c.tile_class_id: c for c in classes}
    unknown = sorted(
        {cid for row in placement_map for cid in row
         if cid != SPARE_SITE and cid not in by_id}
    )
    if unknown:
        raise PlacementError(
            f"placement_map names tile classes the block does not declare: "
            f"{', '.join(unknown)}"
        )

    spare = [
        (r, c)
        for r, row in enumerate(placement_map)
        for c, cid in enumerate(row)
        if cid == SPARE_SITE
    ]
    claimed: set = set()
    placements: List[TilePlacementResult] = []
    counters: Dict[str, int] = {}
    # Row-major scan: the first unclaimed site of a class is a rectangle's
    # top-left corner, and the class's own footprint says how big it is.
    for r in range(rows):
        for c in range(cols):
            cid = placement_map[r][c]
            if cid == SPARE_SITE or (r, c) in claimed:
                continue
            cls = by_id[cid]
            instance = counters.get(cid, 0)
            counters[cid] = instance + 1
            placement = TilePlacementResult(
                tile_class_id=cid, tile_type=cls.tile_type,
                row=r, col=c, rows=cls.rows, cols=cls.cols, instance=instance,
            )
            for site in placement.sites:
                sr, sc = site
                if not (0 <= sr < rows and 0 <= sc < cols):
                    raise PlacementError(
                        f"tile class {cid!r} at ({r}, {c}) has a "
                        f"{cls.rows}x{cls.cols} footprint that runs off the "
                        f"{rows}x{cols} checkerboard"
                    )
                if placement_map[sr][sc] != cid:
                    raise PlacementError(
                        f"tile class {cid!r} at ({r}, {c}) needs a "
                        f"{cls.rows}x{cls.cols} block of its own sites, but "
                        f"({sr}, {sc}) holds {placement_map[sr][sc]!r}"
                    )
                claimed.add(site)
            placements.append(placement)

    notes = []
    for cls in classes:
        placed = counters.get(cls.tile_class_id, 0)
        if placed != cls.num_tiles:
            notes.append(
                f"placement_map holds {placed} {cls.tile_class_id!r} tile(s) "
                f"but the block declares {cls.num_tiles}"
            )

    absorbing = {c.tile_class_id for c in classes if c.absorbs_memory_cells}
    covered, absorbed = _covered_cells(placements, absorbing)
    return SitePlan(
        rows=rows, cols=cols,
        placements=tuple(placements),
        spare_sites=tuple(spare),
        covered_memory_cells=covered,
        absorbed_memory_cells=absorbed,
        mode="explicit",
        notes=tuple(notes),
    )


def place_tiles(block: KPUBlock) -> Optional[SitePlan]:
    """Assign ``block``'s tiles to its checkerboard compute sites.

    Returns None for a block with no ``checkerboard``: every uniform
    catalog SKU, whose floorplan stays on the legacy row-major path and is
    pinned by the golden snapshot.

    Raises:
        PlacementError: the tiles do not fit, or an explicit placement_map
            disagrees with the grid or names an undeclared class.
    """
    checkerboard = block.checkerboard
    if checkerboard is None:
        return None
    rows = checkerboard.compute_sites.rows
    cols = checkerboard.compute_sites.cols
    classes = _classes_to_place(block)

    if checkerboard.placement_map is not None:
        return _explicit_plan(block, rows, cols, classes, checkerboard.placement_map)
    return _auto_plan(block, rows, cols, classes)
