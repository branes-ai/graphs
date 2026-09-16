"""Validators for heterogeneous KPUs (graphs#268 Phase C4).

Each check targets a feature a uniform catalog SKU does not use --
tile-carried silicon, systolic and fixed-function tiles, checkerboards,
NoC stream links, power domains -- so every one of them is silent on the
12 legacy SKUs (the KPU golden snapshot pins their findings).

- fixed_function_energy_plausibility (ENERGY): a core's energy per work
  unit, retargeted to the SKU node, against the tile-class library anchor
  for the same function.
- datapath_energy_resolution (ENERGY): datapath / systolic energy anchors
  resolve on the SKU node.
- silicon_no_double_count (AREA): a tile class counted both on the tile and
  by a chip-level PER_PE block.
- tile_footprint_pitch_fit (AREA): a tile's silicon fits the compute sites
  (and absorbed memory cells) its footprint gives it.
- checkerboard_site_accounting (GEOMETRY): without a checkerboard, the NoC
  mesh must match the sites the tiles occupy.
- stream_link_adjacency (GEOMETRY): stream-linked tile classes are placed
  (or hinted) next to each other.
- cluster_geometry_consistent (GEOMETRY): cluster power domains share one
  shape; cluster count in the DVFS design's recommended range.
- overlay_consistency (INTERNAL): overlays declare their silicon.
- stream_link_bandwidth (ELECTRICAL): a stream link carries its producer's
  output.
- power_domain_coverage (ELECTRICAL): with power domains declared, every
  tile class and every placed site belongs to one.
- cluster_rail_and_clock (ELECTRICAL): a cluster power domain names its own
  rail and clock (the DVFS unit of
  docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md).

The DVFS design's harvesting and process-variation-bin validators need
schema fields that do not exist yet (per-cluster disable flags, variation
bins, a die voltage-rail registry); they are deferred.
"""

from __future__ import annotations

from typing import List, Optional

from embodied_schemas.datapath import AbsoluteEnergy, RelativeEnergy
from embodied_schemas.kpu import FixedFunctionTile, KPUTileSpec, SystolicTile
from embodied_schemas.overlay import NoCOverlayKind
from embodied_schemas.power_domain import PowerDomainKind

from ...kpu_access import kpu_block_of, kpu_die_of
from ...kpu_checkerboard_placer import PlacementError, place_tiles
from .. import ValidatorCategory, ValidatorContext, default_registry
from ..framework import Finding, Severity
from .. import silicon_math as sm

# fixed_function_energy_plausibility: flag energy per unit outside this
# factor of the library anchor (both retargeted to the SKU node).
_FF_ENERGY_BAND = 4.0
# tile_footprint_pitch_fit: slack before a tile is reported as not fitting.
_FOOTPRINT_SLACK = 0.05
# stream_link_bandwidth: warn above this fraction of the link capacity.
_LINK_WARN_FRAC = 0.8
# cluster_geometry_consistent: the DVFS design's recommended cluster count.
_CLUSTER_COUNT_RANGE = (8, 64)


def _nodes(ctx: ValidatorContext):
    """The process-node catalog (``ctx.extras['process_nodes']``, else loaded
    once and cached on the context)."""
    nodes = ctx.extras.get("process_nodes")
    if nodes is None:
        from embodied_schemas import load_process_nodes

        nodes = load_process_nodes()
        ctx.extras["process_nodes"] = nodes
    return nodes


def _library(ctx: ValidatorContext):
    lib = ctx.extras.get("tile_class_library")
    if lib is None:
        from embodied_schemas import load_kpu_tile_classes

        lib = load_kpu_tile_classes()
        ctx.extras["tile_class_library"] = lib
    return lib


def _default_clock_hz(ctx: ValidatorContext) -> float:
    power = ctx.sku.power
    profile = next(p for p in power.thermal_profiles if p.name == power.default_thermal_profile)
    return profile.clock_mhz * 1e6


def _finding(v, severity: Severity, message: str, *, block: Optional[str] = None,
             citation: Optional[str] = None) -> Finding:
    return Finding(validator=v.name, category=v.category, severity=severity,
                   message=message, block=block, citation=citation)


# ---------------------------------------------------------------------------
# ENERGY
# ---------------------------------------------------------------------------


@default_registry.register_class
class FixedFunctionEnergyPlausibility:
    """A fixed-function core's energy per work unit must be plausible.

    Both the core and its tile-class library anchor (the entry named by
    ``tile_class_ref``, else a library entry with the same ``function_id``)
    are retargeted to the SKU's process node with the power model's
    logic / SRAM scaling; a ratio outside ``_FF_ENERGY_BAND`` either way is
    a WARNING. The library entries carry the citations (Navion, SGM,
    Darkroom, ...), so this is the "cited anchor band" check of the plan.
    """

    name = "fixed_function_energy_plausibility"
    category = ValidatorCategory.ENERGY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        from ...kpu_power_model import fixed_function_pj_per_unit

        findings: List[Finding] = []
        tiles = [t for t in kpu_block_of(ctx.sku).tiles if isinstance(t, FixedFunctionTile)]
        if not tiles:
            return findings
        lib = _library(ctx)
        nodes = _nodes(ctx)
        node = ctx.process_node
        for t in tiles:
            anchor = lib.get(t.tile_class_ref) if t.tile_class_ref else None
            if anchor is None or not isinstance(anchor.tile, FixedFunctionTile):
                anchor = next(
                    (e for e in lib.values()
                     if isinstance(e.tile, FixedFunctionTile)
                     and e.tile.core.function_id == t.core.function_id),
                    None,
                )
            if anchor is None:
                findings.append(_finding(
                    self, Severity.INFO,
                    f"fixed-function tile {t.tile_class_id!r}: no tile-class library "
                    f"anchor for function {t.core.function_id!r}; energy per "
                    f"{t.core.throughput.unit.value} not cross-checked.",
                    block=t.tile_class_id,
                ))
                continue
            mine = fixed_function_pj_per_unit(t.core, node, nodes)
            ref = fixed_function_pj_per_unit(anchor.tile.core, node, nodes)
            if mine is None or ref is None or ref <= 0:
                continue  # an unresolvable reference node: reported by the power model
            ratio = mine / ref
            if ratio > _FF_ENERGY_BAND or ratio < 1.0 / _FF_ENERGY_BAND:
                unit = t.core.throughput.unit.value
                findings.append(_finding(
                    self, Severity.WARNING,
                    f"fixed-function tile {t.tile_class_id!r}: {mine:.4g} pJ per {unit} at "
                    f"{node.id} is {ratio:.2f}x the library anchor {anchor.id!r} "
                    f"({ref:.4g} pJ per {unit}), outside the {_FF_ENERGY_BAND:g}x "
                    f"plausibility band.",
                    block=t.tile_class_id,
                    citation=f"kpu-tile-classes:{anchor.id}; " + "; ".join(anchor.sources[:1]),
                ))
        return findings


@default_registry.register_class
class DatapathEnergyResolution:
    """Datapath and systolic-cell energies must resolve on the SKU node.

    A RelativeEnergy anchor the node lacks is an ERROR: the power model
    then charges that mode no compute energy. An AbsoluteEnergy whose
    reference node is not in the catalog is a WARNING (the figure is used
    unscaled). A mode without energy is charged the node anchor (no
    finding).
    """

    name = "datapath_energy_resolution"
    category = ValidatorCategory.ENERGY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        node = ctx.process_node
        for t in kpu_block_of(ctx.sku).tiles:
            if isinstance(t, KPUTileSpec) and t.datapath is not None:
                units = t.datapath.functional_units
            elif isinstance(t, SystolicTile):
                units = [t.mac]
            else:
                continue
            for u in units:
                for mode in u.modes:
                    e, where = mode.energy, f"{t.tile_class_id}.{u.unit_id}:{mode.operand_format}"
                    if isinstance(e, RelativeEnergy) and e.anchor not in node.energy_per_op_pj:
                        findings.append(_finding(
                            self, Severity.ERROR,
                            f"{where}: energy anchor {e.anchor!r} is not in "
                            f"{node.id}.energy_per_op_pj; the power model charges this "
                            f"mode no compute energy.",
                            block=t.tile_class_id, citation=f"process_node:{node.id}",
                        ))
                    elif isinstance(e, AbsoluteEnergy) and e.ref_node_id not in _nodes(ctx):
                        findings.append(_finding(
                            self, Severity.WARNING,
                            f"{where}: reference node {e.ref_node_id!r} is not in the "
                            f"catalog; its {e.pj:g} pJ is used unscaled.",
                            block=t.tile_class_id,
                        ))
        return findings


# ---------------------------------------------------------------------------
# AREA
# ---------------------------------------------------------------------------


@default_registry.register_class
class SiliconNoDoubleCount:
    """A tile class's logic may be counted on the tile (datapath, systolic
    cells, function core) or by a chip-level PER_PE silicon_bin block, never
    both. The generator refuses such a spec; this catches hand-authored or
    edited SKUs."""

    name = "silicon_no_double_count"
    category = ValidatorCategory.AREA

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        try:
            doubled = sm.double_counted_tile_classes(ctx.sku, _nodes(ctx))
        except sm.SiliconMathError:
            return []
        return [
            _finding(
                self, Severity.ERROR,
                f"tile class {cid!r} carries its own logic silicon and is also counted "
                f"by a chip-level PER_PE silicon_bin block; the die roll-up counts it twice.",
                block=cid,
            )
            for cid in doubled
        ]


def _per_tile_areas(ctx: ValidatorContext) -> dict[str, float]:
    """Per-tile area (mm^2) by tile_class_id: chip-level PER_PE share +
    tile-carried silicon + chip L2 share (classes that inherit it)."""
    sku, node = ctx.sku, ctx.process_node
    block = kpu_block_of(sku)
    area = {t.tile_class_id: 0.0 for t in block.tiles}
    l2_area = 0.0
    for b in kpu_die_of(sku).silicon_bin.blocks:
        ts = b.transistor_source
        try:
            ba = sm.resolve_block_area(b, sku, node)
        except sm.SiliconMathError:
            continue
        if ts.kind.value == "per_pe" and ts.count_ref:
            try:
                tile = sm.resolve_tile_ref(sku, ts.count_ref.removeprefix("tile."))
            except sm.SiliconMathError:
                continue
            area[tile.tile_class_id] += ba.area_mm2
        elif ts.count_ref == "l2_total_kib":
            l2_area += ba.area_mm2
    for ba in sm.resolve_carried_areas(sku, node, _nodes(ctx)):
        cid = ba.name.split(".", 1)[0]
        if cid in area:
            area[cid] += ba.area_mm2
    inheriting = sum(t.num_tiles for t in block.tiles if sm.inherits_chip_memory(t))
    out = {}
    for t in block.tiles:
        per_tile = area[t.tile_class_id] / t.num_tiles
        if sm.inherits_chip_memory(t) and inheriting:
            per_tile += l2_area / inheriting
        out[t.tile_class_id] = per_tile
    return out


def _memory_cell_area(ctx: ValidatorContext) -> float:
    sku, node = ctx.sku, ctx.process_node
    cells = sm.l3_memory_cells(sku)
    for b in kpu_die_of(sku).silicon_bin.blocks:
        if b.transistor_source.count_ref == "l3_total_kib" and cells:
            try:
                return sm.resolve_block_area(b, sku, node).area_mm2 / cells
            except sm.SiliconMathError:
                return 0.0
    return 0.0


@default_registry.register_class
class TileFootprintPitchFit:
    """A tile must fit the compute sites its footprint gives it.

    The KPU checkerboard has one site pitch (plan D4): the compute-site
    area is the largest pe_fabric tile (its PE share + carried silicon +
    L2). A systolic or fixed-function tile, or any tile with a footprint,
    has ``rows x cols`` sites, plus the memory cells when the footprint
    absorbs them. A tile that needs more than that (5% slack) is a
    WARNING: enlarge its footprint.
    """

    name = "tile_footprint_pitch_fit"
    category = ValidatorCategory.AREA

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        block = kpu_block_of(ctx.sku)
        tiles = block.tiles
        if all(isinstance(t, KPUTileSpec) and t.footprint is None for t in tiles):
            return []
        areas = _per_tile_areas(ctx)
        site = max((areas[t.tile_class_id] for t in tiles if isinstance(t, KPUTileSpec)),
                   default=0.0)
        if site <= 0:
            return []
        cell = _memory_cell_area(ctx)
        findings: List[Finding] = []
        for t in tiles:
            if isinstance(t, KPUTileSpec) and t.footprint is None:
                continue
            fp = t.footprint
            rows, cols = (fp.rows, fp.cols) if fp else (1, 1)
            absorbs = bool(fp and fp.absorbs_memory_cells)
            capacity = rows * cols * (site + (cell if absorbs else 0.0))
            need = areas[t.tile_class_id]
            if need > capacity * (1 + _FOOTPRINT_SLACK):
                findings.append(_finding(
                    self, Severity.WARNING,
                    f"tile class {t.tile_class_id!r} needs {need:.3f} mm^2 per tile but its "
                    f"{rows}x{cols} footprint offers {capacity:.3f} mm^2 "
                    f"({site:.3f} mm^2 per compute site"
                    f"{f' + {cell:.3f} mm^2 per absorbed memory cell' if absorbs else ''}); "
                    f"enlarge the footprint to about {need / (site + (cell if absorbs else 0.0)):.1f} sites.",
                    block=t.tile_class_id,
                ))
        return findings


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------


@default_registry.register_class
class CheckerboardSiteAccounting:
    """The compute sites must account for the tiles.

    With a checkerboard the schema enforces the site-accounting invariant;
    this reports its spare sites (INFO). Without one, the NoC mesh is the
    implicit grid: fewer routers than occupied sites is an ERROR (the
    floorplan would truncate tiles), more is a WARNING (implicit spare
    sites; declare a checkerboard). Replaces the silent pad / truncate
    (plan C11).
    """

    name = "checkerboard_site_accounting"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        block = kpu_block_of(ctx.sku)
        used = sum(t.total_sites for t in block.tiles)
        cb = block.checkerboard
        if cb is not None:
            if cb.spare_sites:
                return [_finding(
                    self, Severity.INFO,
                    f"checkerboard {cb.compute_sites.rows}x{cb.compute_sites.cols}: tiles "
                    f"occupy {used} sites, {cb.spare_sites} spare "
                    f"({cb.spare_sites / cb.compute_sites.sites:.0%}).",
                )]
            return []
        mesh = block.noc.mesh_rows * block.noc.mesh_cols
        if mesh == used:
            return []
        if mesh < used:
            return [_finding(
                self, Severity.ERROR,
                f"tiles occupy {used} compute sites but the "
                f"{block.noc.mesh_rows}x{block.noc.mesh_cols} NoC mesh has {mesh}; the "
                f"floorplan would drop {used - mesh} tile site(s). Declare a checkerboard.",
            )]
        return [_finding(
            self, Severity.WARNING,
            f"tiles occupy {used} compute sites of the {block.noc.mesh_rows}x"
            f"{block.noc.mesh_cols} NoC mesh; {mesh - used} site(s) are implicit spares. "
            f"Declare a checkerboard with spare_sites.",
        )]


def _stream_pairs(block):
    for ov in block.noc.overlays or []:
        if ov.kind == NoCOverlayKind.STREAM_LINK:
            for a, b in zip(ov.endpoints, ov.endpoints[1:]):
                yield ov, a, b


def _sites_by_class(pmap) -> dict[str, set]:
    out: dict[str, set] = {}
    for r, row in enumerate(pmap):
        for c, cid in enumerate(row):
            out.setdefault(cid, set()).add((r, c))
    return out


@default_registry.register_class
class StreamLinkAdjacency:
    """Stream-linked tile classes should sit next to each other.

    Consecutive stream-link endpoints must have 4-adjacent sites, or the
    link crosses the mesh and the segment encapsulation the chain exists
    for is lost.

    Since D1 there is a placer, so this checks where the tiles actually
    land -- under an explicit placement map *and* under auto placement,
    which used to be taken on trust (the check was only that a
    ``placement.adjacent_to`` hint existed, which says what the author
    wanted, not what the placer did). A block with no checkerboard has no
    site grid to check, and falls back to the hint.
    """

    name = "stream_link_adjacency"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        block = kpu_block_of(ctx.sku)
        by_id = {t.tile_class_id: t for t in block.tiles}
        cb = block.checkerboard
        sites = None
        where = ""
        if cb is not None and cb.placement_map:
            sites = _sites_by_class(cb.placement_map)
            where = "in the placement map"
        elif cb is not None:
            try:
                plan = place_tiles(block)
            except PlacementError:
                plan = None  # checkerboard_site_accounting reports this
            if plan is not None:
                sites = {
                    cid: set(plan.sites_of(cid))
                    for cid in {p.tile_class_id for p in plan.placements}
                }
                where = "as placed"
        findings: List[Finding] = []
        for ov, a, b in _stream_pairs(block):
            if sites is not None:
                sa, sb = sites.get(a, set()), sites.get(b, set())
                if not any((r + dr, c + dc) in sb for r, c in sa
                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    findings.append(_finding(
                        self, Severity.WARNING,
                        f"stream link {ov.overlay_id!r}: {a!r} and {b!r} are not adjacent "
                        f"{where}; the link crosses the mesh.",
                    ))
                continue
            hints = set()
            for x, y in ((a, b), (b, a)):
                p = by_id[x].placement
                if p is not None and y in p.adjacent_to:
                    hints.add(x)
            if not hints:
                findings.append(_finding(
                    self, Severity.INFO,
                    f"stream link {ov.overlay_id!r}: neither {a!r} nor {b!r} lists the "
                    f"other in placement.adjacent_to; the auto placer may separate them.",
                ))
        return findings


def _cluster_domains(block):
    return [d for d in block.power_domains or [] if d.kind == PowerDomainKind.CLUSTER]


@default_registry.register_class
class ClusterGeometryConsistent:
    """Cluster power domains should be one repeated shape, and their count
    within the DVFS design's 8-64 range (INFO otherwise: too coarse for
    meaningful DVFS / harvest, or too many rails and PLLs)."""

    name = "cluster_geometry_consistent"
    category = ValidatorCategory.GEOMETRY

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        clusters = _cluster_domains(kpu_block_of(ctx.sku))
        if not clusters:
            return []
        findings: List[Finding] = []
        # A cluster's shape: the (rows, cols) of each of its site ranges.
        shapes = {
            tuple(sorted((r.rows, r.cols) for r in d.site_ranges)) for d in clusters
        }
        if len(shapes) > 1:
            shown = sorted(" + ".join(f"{r}x{c}" for r, c in shape) for shape in shapes)
            findings.append(_finding(
                self, Severity.WARNING,
                f"cluster power domains have different shapes {shown}; the DVFS design "
                f"uses one repeated k x k cluster shape.",
                citation="docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md",
            ))
        lo, hi = _CLUSTER_COUNT_RANGE
        if not lo <= len(clusters) <= hi:
            findings.append(_finding(
                self, Severity.INFO,
                f"{len(clusters)} cluster power domain(s); the DVFS design recommends "
                f"{lo}-{hi} clusters per chip.",
                citation="docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md",
            ))
        return findings


# ---------------------------------------------------------------------------
# INTERNAL
# ---------------------------------------------------------------------------


@default_registry.register_class
class OverlayConsistency:
    """Overlays should declare their silicon (``mtx_per_instance``);
    otherwise they are free in the die roll-up (INFO)."""

    name = "overlay_consistency"
    category = ValidatorCategory.INTERNAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        block = kpu_block_of(ctx.sku)
        missing = [
            f"{t.tile_class_id}.{ov.overlay_id}"
            for t in block.tiles
            if isinstance(t, KPUTileSpec) and t.interconnect is not None
            for ov in t.interconnect.overlays
            if ov.mtx_per_instance is None
        ] + [f"noc.{ov.overlay_id}" for ov in block.noc.overlays or [] if ov.mtx_per_instance is None]
        if not missing:
            return []
        return [_finding(
            self, Severity.INFO,
            f"overlays without mtx_per_instance are not costed in the die roll-up: {missing}.",
        )]


# ---------------------------------------------------------------------------
# ELECTRICAL
# ---------------------------------------------------------------------------


@default_registry.register_class
class StreamLinkBandwidth:
    """A stream link must carry its producer's output and its consumer's
    input at the default profile clock: ``units_per_clock x num_tiles x
    bytes_per_unit`` against ``width_bytes x instances`` per clock. Above
    capacity is an ERROR, above 80% a WARNING. Endpoints without an IO
    model (programmable tiles) are not checked."""

    name = "stream_link_bandwidth"
    category = ValidatorCategory.ELECTRICAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        block = kpu_block_of(ctx.sku)
        by_id = {t.tile_class_id: t for t in block.tiles}
        hz = _default_clock_hz(ctx)
        findings: List[Finding] = []
        for ov, a, b in _stream_pairs(block):
            capacity = ov.width_bytes * ov.instances * hz
            for cid, side in ((a, "output"), (b, "input")):
                t = by_id[cid]
                if not isinstance(t, FixedFunctionTile) or t.core.io is None:
                    continue
                per_unit = (t.core.io.output_bytes_per_unit if side == "output"
                            else t.core.io.input_bytes_per_unit)
                demand = t.core.units_per_clock * t.num_tiles * per_unit * hz
                frac = demand / capacity if capacity else float("inf")
                if frac > 1.0:
                    sev = Severity.ERROR
                elif frac > _LINK_WARN_FRAC:
                    sev = Severity.WARNING
                else:
                    continue
                findings.append(_finding(
                    self, sev,
                    f"stream link {ov.overlay_id!r} ({a} -> {b}): {cid!r} {side} needs "
                    f"{demand / 1e9:.2f} GB/s, {frac:.0%} of the link's "
                    f"{capacity / 1e9:.2f} GB/s ({ov.width_bytes} B x {ov.instances} at "
                    f"{hz / 1e6:.0f} MHz).",
                    block=cid,
                ))
        return findings


@default_registry.register_class
class PowerDomainCoverage:
    """With power domains declared, every tile class must belong to one.

    A class is covered by its ``power_domain_id``, by a tile_class domain
    listing it, or -- for classes with PEs -- by the cluster domains. With
    an explicit placement map, every site of a cluster-covered class must
    lie in a cluster's site ranges. An uncovered class runs at the chip
    clock / Vdd and cannot be gated (WARNING). No uncore domain is INFO.
    """

    name = "power_domain_coverage"
    category = ValidatorCategory.ELECTRICAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        block = kpu_block_of(ctx.sku)
        domains = block.power_domains
        if not domains:
            return []
        findings: List[Finding] = []
        listed = {m for d in domains if d.kind == PowerDomainKind.TILE_CLASS for m in d.members}
        clusters = _cluster_domains(block)
        cluster_sites = set().union(*(d.sites() for d in clusters)) if clusters else set()
        by_cluster = []
        for t in block.tiles:
            if t.power_domain_id is not None or t.tile_class_id in listed:
                continue
            if clusters and sm.has_pes(t):
                by_cluster.append(t.tile_class_id)
                continue
            findings.append(_finding(
                self, Severity.WARNING,
                f"tile class {t.tile_class_id!r} is in no power domain: it runs at the "
                f"chip clock / Vdd and cannot be gated.",
                block=t.tile_class_id,
            ))
        cb = block.checkerboard
        if cb is not None and cb.placement_map is not None and by_cluster:
            sites = _sites_by_class(cb.placement_map)
            for cid in by_cluster:
                outside = sorted(sites.get(cid, set()) - cluster_sites)
                if outside:
                    findings.append(_finding(
                        self, Severity.WARNING,
                        f"tile class {cid!r} has {len(outside)} site(s) outside every "
                        f"cluster power domain, e.g. {outside[0]}.",
                        block=cid,
                    ))
        if not any(d.kind == PowerDomainKind.UNCORE for d in domains):
            findings.append(_finding(
                self, Severity.INFO,
                "no uncore power domain: NoC, L3 cells, PHYs and IO run on the chip rail.",
            ))
        return findings


@default_registry.register_class
class ClusterRailAndClock:
    """A cluster power domain is the DVFS unit: it should name its own
    supply rail and clock (``rail_id``, ``clock_domain_id``). Missing either
    is a WARNING."""

    name = "cluster_rail_and_clock"
    category = ValidatorCategory.ELECTRICAL

    def check(self, ctx: ValidatorContext) -> List[Finding]:
        findings: List[Finding] = []
        for d in _cluster_domains(kpu_block_of(ctx.sku)):
            missing = [f for f in ("rail_id", "clock_domain_id") if getattr(d, f) is None]
            if missing:
                findings.append(_finding(
                    self, Severity.WARNING,
                    f"cluster power domain {d.domain_id!r} has no {' / '.join(missing)}; a "
                    f"DVFS cluster needs its own rail and PLL.",
                    citation="docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md",
                ))
        return findings
