"""Silicon-math helpers for area / power / EM validators.

The KPU silicon_bin decomposes the chip into blocks, each tagged with a
``circuit_class`` and a ``transistor_source`` describing how its
transistor count is computed (fixed / per-PE / per-KiB-of-SRAM /
per-NoC-router / per-memory-controller).

This module turns those declarative blocks into absolute numbers --
transistor counts, areas, leakage powers -- that validators then
compare against process-node ceilings and roll-up claims.

The resolution of each ``count_ref`` form is documented inline so future
validators / generator code can reuse the same vocabulary.

Heterogeneous tiles (graphs#268 Phase C1):

* **Rollups are kind-aware.** PE counts cover the tile kinds that have PEs
  (pe_fabric PEs and systolic cells). Chip-level L1 / L2 cover only the tile
  classes that inherit the chip ``KPUMemorySubsystem`` figures (pe_fabric
  tiles without ``local_memory``). L3 counts checkerboard memory cells,
  minus the cells a footprint absorbs.
* **Tiles may carry their own silicon** (``carried_silicon``): datapath or
  systolic-cell transistors, tile-local SRAM, PE-fabric and NoC overlays,
  and fixed-function core silicon. The chip ``silicon_bin`` keeps the
  uncore. ``double_counted_tile_classes`` finds a tile class counted both
  ways.
* **Dynamic power dispatches on the tile a PER_PE block resolves to**, not
  on the block-name prefix.

For every catalog SKU (uniform pe_fabric tiles, no tile-carried silicon)
every value is unchanged; the KPU golden snapshot pins that.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Mapping, Optional

from embodied_schemas import ComputeProduct
from embodied_schemas.compute_product import Die, KPUBlock
from embodied_schemas.kpu import (
    FixedFunctionTile,
    KPUTileSpec,
    SiliconBinBlock,
    SystolicTile,
    TransistorSourceKind,
)
from embodied_schemas.local_memory import LocalMemory, LocalMemoryScope
from embodied_schemas.overlay import OverlayScope
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry

from graphs.hardware.kpu_access import KPUBlockLookupError, kpu_block_of, kpu_die_of


class SiliconMathError(Exception):
    """Raised when a silicon_bin block can't be resolved (bad count_ref,
    missing per_unit_mtx, etc.). Validators catch this and convert into
    Findings rather than crashing."""


def _kpu_block(cp: ComputeProduct) -> KPUBlock:
    """The product's single KPUBlock, located by kind (``kpu_block_of``).

    Raises ``SiliconMathError`` (not KPUBlockLookupError / IndexError) for
    a product with no KPU block or more than one, so the validator
    framework converts the failure into a Finding instead of crashing the
    validator.
    """
    try:
        return kpu_block_of(cp)
    except KPUBlockLookupError as exc:
        raise SiliconMathError(str(exc)) from exc


def _kpu_die(cp: ComputeProduct) -> Die:
    """The die carrying the product's KPUBlock (its silicon_bin, clocks,
    process node). Raises ``SiliconMathError`` like ``_kpu_block``."""
    try:
        return kpu_die_of(cp)
    except KPUBlockLookupError as exc:
        raise SiliconMathError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Architecture-level rollups
# ---------------------------------------------------------------------------

def has_pes(tile: Any) -> bool:
    """Whether a tile kind is an array of PEs / cells (pe_fabric, systolic).
    Fixed-function tiles have no PEs."""
    return isinstance(tile, (KPUTileSpec, SystolicTile))


def inherits_chip_memory(tile: Any) -> bool:
    """Whether a tile class takes its L1 / L2 from the chip-level
    ``KPUMemorySubsystem``: a pe_fabric tile that declares no
    ``local_memory``. Systolic and fixed-function tiles declare their own."""
    return isinstance(tile, KPUTileSpec) and tile.local_memory is None


def total_pe_count(cp: ComputeProduct) -> int:
    """Total PEs (pe_fabric PEs + systolic cells) across every tile class."""
    return sum(t.total_pes for t in _kpu_block(cp).tiles if has_pes(t))


def total_l1_kib(cp: ComputeProduct) -> int:
    """Chip-level per-PE L1 SRAM in KiB: ``l1_kib_per_pe`` times the PEs of
    the tile classes that inherit the chip memory figures. Tile-declared
    memory is tile-carried silicon (``carried_silicon``)."""
    block = _kpu_block(cp)
    pes = sum(t.total_pes for t in block.tiles if inherits_chip_memory(t))
    return block.memory.l1_kib_per_pe * pes


def total_l2_kib(cp: ComputeProduct) -> int:
    """Chip-level per-tile L2 SRAM in KiB: ``l2_kib_per_tile`` times the
    tiles of the classes that inherit the chip memory figures."""
    block = _kpu_block(cp)
    tiles = sum(t.num_tiles for t in block.tiles if inherits_chip_memory(t))
    return block.memory.l2_kib_per_tile * tiles


def l3_memory_cells(cp: ComputeProduct) -> int:
    """Shared L3 memory cells: one per compute site (the checkerboard's
    ``compute_sites``, or the sites the tiles occupy when there is no
    checkerboard), minus the cells a footprint with
    ``absorbs_memory_cells`` turns into tile-local memory."""
    block = _kpu_block(cp)
    if block.checkerboard is not None:
        sites = block.checkerboard.compute_sites.sites
    else:
        sites = sum(t.total_sites for t in block.tiles)
    absorbed = sum(
        t.total_sites
        for t in block.tiles
        if t.footprint is not None and t.footprint.absorbs_memory_cells
    )
    return sites - absorbed


def total_l3_kib(cp: ComputeProduct) -> int:
    """Distributed L3 SRAM in KiB: ``l3_kib_per_tile`` per memory cell."""
    return _kpu_block(cp).memory.l3_kib_per_tile * l3_memory_cells(cp)


def num_tiles_by_type(cp: ComputeProduct) -> dict[str, int]:
    """Map of tile_type -> num_tiles. Useful for ``count_ref="tile.<type>"``."""
    return {t.tile_type: t.num_tiles for t in _kpu_block(cp).tiles}


def total_pes_by_tile_type(cp: ComputeProduct) -> dict[str, int]:
    """Map of tile_type -> total PE count, for the tile classes with PEs."""
    return {t.tile_type: t.total_pes for t in _kpu_block(cp).tiles if has_pes(t)}


def resolve_tile_ref(cp: ComputeProduct, ref: str) -> Any:
    """The tile class named by a ``count_ref`` suffix (``tile.<ref>``).

    ``ref`` is a ``tile_class_id`` or, the legacy form, a ``tile_type``
    label (catalog SKUs use labels such as ``INT8-primary``). An id match
    wins; a label shared by several classes is ambiguous.
    """
    tiles = _kpu_block(cp).tiles
    by_id = [t for t in tiles if t.tile_class_id == ref]
    if by_id:
        return by_id[0]
    by_label = [t for t in tiles if t.tile_type == ref]
    if len(by_label) == 1:
        return by_label[0]
    if len(by_label) > 1:
        raise SiliconMathError(
            f"tile ref {ref!r} matches {len(by_label)} tile classes by label; "
            f"use a tile_class_id"
        )
    raise SiliconMathError(
        f"unknown tile ref {ref!r}. Available: tile_class_id "
        f"{sorted(t.tile_class_id for t in tiles)}, tile_type "
        f"{sorted(t.tile_type for t in tiles)}"
    )


# ---------------------------------------------------------------------------
# Transistor count resolution
# ---------------------------------------------------------------------------

def resolve_block_transistors(block: SiliconBinBlock, cp: ComputeProduct) -> float:
    """Compute the absolute transistor count (in millions) for a silicon_bin
    block, given the SKU's architecture context.

    Resolution by ``transistor_source.kind``:

    * **FIXED**: returns ``mtx`` directly. ``count_ref`` is ignored.
    * **PER_PE**: ``count_ref`` is ``"tile.<ref>"``, ``<ref>`` a
      ``tile_class_id`` or ``tile_type`` label (``resolve_tile_ref``).
      Returns ``per_unit_mtx * total_pes`` of that class, which must have
      PEs (pe_fabric or systolic).
    * **PER_KIB**: ``count_ref`` is one of ``l1_total_kib``,
      ``l2_total_kib``, ``l3_total_kib``. Returns
      ``per_unit_mtx * total_<level>_kib(cp)``.
    * **PER_ROUTER**: ``count_ref`` is ``"noc"``. Returns
      ``per_unit_mtx * noc.num_routers``.
    * **PER_CONTROLLER**: ``count_ref`` is ``"memory"``. Returns
      ``per_unit_mtx * memory.memory_controllers``.

    Raises SiliconMathError on malformed source (missing field,
    unrecognized count_ref).
    """
    ts = block.transistor_source

    if ts.kind == TransistorSourceKind.FIXED:
        if ts.mtx is None:
            raise SiliconMathError(
                f"block {block.name!r}: kind=FIXED requires 'mtx'"
            )
        return float(ts.mtx)

    # All non-FIXED kinds need per_unit_mtx and a count_ref.
    if ts.per_unit_mtx is None:
        raise SiliconMathError(
            f"block {block.name!r}: kind={ts.kind.value} requires 'per_unit_mtx'"
        )
    if not ts.count_ref:
        raise SiliconMathError(
            f"block {block.name!r}: kind={ts.kind.value} requires 'count_ref'"
        )

    if ts.kind == TransistorSourceKind.PER_PE:
        if not ts.count_ref.startswith("tile."):
            raise SiliconMathError(
                f"block {block.name!r}: PER_PE count_ref must be 'tile.<ref>', "
                f"got {ts.count_ref!r}"
            )
        try:
            tile = resolve_tile_ref(cp, ts.count_ref.split(".", 1)[1])
        except SiliconMathError as exc:
            raise SiliconMathError(f"block {block.name!r}: {exc}") from exc
        if not has_pes(tile):
            raise SiliconMathError(
                f"block {block.name!r}: tile class {tile.tile_class_id!r} is "
                f"{tile.tile_kind.value} and has no PEs; declare its silicon on the tile"
            )
        return float(ts.per_unit_mtx) * tile.total_pes

    if ts.kind == TransistorSourceKind.PER_KIB:
        kib_resolvers = {
            "l1_total_kib": total_l1_kib,
            "l2_total_kib": total_l2_kib,
            "l3_total_kib": total_l3_kib,
        }
        if ts.count_ref not in kib_resolvers:
            raise SiliconMathError(
                f"block {block.name!r}: PER_KIB count_ref must be one of "
                f"{sorted(kib_resolvers)}, got {ts.count_ref!r}"
            )
        kib = kib_resolvers[ts.count_ref](cp)
        return float(ts.per_unit_mtx) * kib

    if ts.kind == TransistorSourceKind.PER_ROUTER:
        if ts.count_ref != "noc":
            raise SiliconMathError(
                f"block {block.name!r}: PER_ROUTER count_ref must be 'noc', "
                f"got {ts.count_ref!r}"
            )
        return float(ts.per_unit_mtx) * _kpu_block(cp).noc.num_routers

    if ts.kind == TransistorSourceKind.PER_CONTROLLER:
        if ts.count_ref != "memory":
            raise SiliconMathError(
                f"block {block.name!r}: PER_CONTROLLER count_ref must be 'memory', "
                f"got {ts.count_ref!r}"
            )
        return (
            float(ts.per_unit_mtx)
            * _kpu_block(cp).memory.memory_controllers
        )

    raise SiliconMathError(
        f"block {block.name!r}: unhandled TransistorSourceKind {ts.kind!r}"
    )


# ---------------------------------------------------------------------------
# Area resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockArea:
    """Per-block area + transistor budget after resolution."""

    name: str
    circuit_class: CircuitClass
    transistors_mtx: float       # millions of transistors
    density_mtx_per_mm2: float   # density looked up from process node
    area_mm2: float              # transistors / density


def resolve_block_area(
    block: SiliconBinBlock, cp: ComputeProduct, node: ProcessNodeEntry
) -> BlockArea:
    """Compute area for one silicon_bin block at the SKU's process node.

    Raises SiliconMathError if the node doesn't offer the block's
    circuit_class -- callers (validators) catch this and emit a
    block_library_validity Finding.
    """
    if not node.supports(block.circuit_class):
        raise SiliconMathError(
            f"block {block.name!r}: process node {node.id!r} does not "
            f"offer library {block.circuit_class.value!r}. Available: "
            f"{sorted(c.value for c in node.densities)}"
        )
    transistors = resolve_block_transistors(block, cp)
    density = node.density_for(block.circuit_class).mtx_per_mm2
    return BlockArea(
        name=block.name,
        circuit_class=block.circuit_class,
        transistors_mtx=transistors,
        density_mtx_per_mm2=density,
        area_mm2=transistors / density,
    )


def resolve_all_block_areas(
    cp: ComputeProduct, node: ProcessNodeEntry
) -> list[BlockArea]:
    """Resolve every silicon_bin block. Skips blocks whose circuit_class
    is unsupported by the node (the block_library_validity validator
    reports those separately) so other validators can still see partial
    coverage.
    """
    out: list[BlockArea] = []
    for block in _kpu_die(cp).silicon_bin.blocks:
        try:
            out.append(resolve_block_area(block, cp, node))
        except SiliconMathError:
            continue
    return out


# ---------------------------------------------------------------------------
# Power estimates (used by thermal + EM validators)
# ---------------------------------------------------------------------------

# Default per-byte energy for off-chip memory PHYs (pJ/byte). Process-node
# value would be more precise; this is a node-agnostic v1 estimate.
_MEM_PHY_PJ_PER_BYTE_BY_TYPE: dict[str, float] = {
    "lpddr4": 12.0,
    "lpddr4x": 11.0,
    "lpddr5": 10.0,
    "lpddr5x": 9.5,
    "ddr5": 13.0,
    "gddr6": 8.0,
    "gddr6x": 7.5,
    "hbm2": 7.0,
    "hbm2e": 6.5,
    "hbm3": 6.0,
    "hbm3e": 5.5,
    "unified": 10.0,
}


def estimate_block_leakage_w(
    block: SiliconBinBlock, cp: ComputeProduct, node: ProcessNodeEntry
) -> float:
    """Per-block static (leakage) power in watts.

    Computed as ``leakage_w_per_mm2[circuit_class] * area_mm2``. Returns
    0.0 if the node has no leakage entry for the block's class -- the
    process-node author left the data sparse and we don't invent a value.
    """
    try:
        ba = resolve_block_area(block, cp, node)
    except SiliconMathError:
        return 0.0
    leakage_density = node.leakage_w_per_mm2.get(block.circuit_class, 0.0)
    return leakage_density * ba.area_mm2


def estimate_block_peak_dynamic_w(
    block: SiliconBinBlock,
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    *,
    clock_mhz: float,
    precision: str = "int8",
) -> float:
    """Per-block dynamic power at peak activity, in watts.

    Modeled per block (heuristic v1):

    * **PE blocks** (``PER_PE`` with a ``tile.<ref>`` count_ref): dispatches
      on the tile class the ref resolves to, not on the block name. A class
      with PEs (pe_fabric, systolic) gives
      ``total_pes * ops_per_pe_per_clock * clock * energy_per_op_pj``
      using ``process_node.energy_per_op_pj["{circuit_class}:{precision}"]``.
      Returns 0 if the ref does not resolve, the class has no PEs or no ops
      at this precision, or the node has no energy entry.
    * **memory_phys** (or ``mem_phy*``, ``hbm_phy*``, etc.): treats the
      block as the off-chip memory PHY and estimates dynamic from
      ``memory_bandwidth_gbps * pJ/byte`` using the per-memory-type
      table above.
    * **noc_routers**: estimates 5 % of TDP allocated to NoC switching
      activity at peak. Coarse but better than zero.
    * Other blocks: 0 (treated as leakage-only at v1; refine with
      activity-factor models in follow-up).

    The returned values are *peak* (sustained max-activity) estimates;
    the chip will throttle below this in practice. The thermal validator
    uses peak power to detect hotspots that would force throttling.
    """
    name = block.name.lower()
    kpu = _kpu_block(cp)
    ts = block.transistor_source

    # ---- PE blocks: dispatch on the tile the block counts ----
    if ts.kind == TransistorSourceKind.PER_PE and ts.count_ref:
        try:
            tile = resolve_tile_ref(cp, ts.count_ref.removeprefix("tile."))
        except SiliconMathError:
            return 0.0
        if not has_pes(tile):
            return 0.0
        # Ops per PE for this precision = ops_per_tile / total_pes_per_tile.
        ops_per_tile = tile.ops_per_tile_per_clock.get(precision, 0)
        if ops_per_tile <= 0:
            return 0.0
        ops_per_pe = ops_per_tile / max(1, tile.pes_per_tile)
        total_ops_per_sec = (
            tile.total_pes * ops_per_pe * (clock_mhz * 1e6)
        )
        energy_key = f"{block.circuit_class.value}:{precision}"
        energy_pj = node.energy_per_op_pj.get(energy_key, 0.0)
        if energy_pj <= 0:
            return 0.0
        return total_ops_per_sec * energy_pj * 1e-12

    # ---- Memory PHY ----
    if "phy" in name or "mem_phy" in name or name == "memory_phys":
        mem = kpu.memory
        pj_per_byte = _MEM_PHY_PJ_PER_BYTE_BY_TYPE.get(
            mem.memory_type.value, 10.0
        )
        return mem.memory_bandwidth_gbps * 1e9 * pj_per_byte * 1e-12

    # ---- NoC routers (coarse: 5 % of default TDP) ----
    if name.startswith("noc"):
        # Use the SKU's default-profile TDP as the budget.
        tdp = cp.power.tdp_watts
        return tdp * 0.05

    # ---- Other (control logic, IO pads): leakage-only at v1 ----
    return 0.0


def estimate_block_total_peak_w(
    block: SiliconBinBlock,
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    *,
    clock_mhz: float,
    precision: str = "int8",
) -> float:
    """Per-block total power at peak = leakage + dynamic."""
    return (
        estimate_block_leakage_w(block, cp, node)
        + estimate_block_peak_dynamic_w(
            block, cp, node, clock_mhz=clock_mhz, precision=precision
        )
    )


def total_chip_leakage_w(
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> float:
    """Sum of every silicon_bin block's leakage plus the leakage of the
    tile-carried silicon (zero for catalog SKUs). ``nodes`` resolves the
    reference nodes of fixed-function core areas (see ``carried_silicon``)."""
    return sum(
        estimate_block_leakage_w(b, cp, node)
        for b in _kpu_die(cp).silicon_bin.blocks
    ) + sum(
        node.leakage_w_per_mm2.get(ba.circuit_class, 0.0) * ba.area_mm2
        for ba in resolve_carried_areas(cp, node, nodes)
    )


# ---------------------------------------------------------------------------
# Tile-carried silicon (graphs#268 Phase C1)
# ---------------------------------------------------------------------------

# Transistors per KiB of tile-local SRAM, by library. 6T cells: 8192 bits x 6
# = 0.049 Mtx plus ~6% periphery = 0.052, the figure the catalog's
# l2_sram / l3_sram PER_KIB blocks use. Dual-port (sram_hp) cells are 8T.
SRAM_MTX_PER_KIB: dict[CircuitClass, float] = {
    CircuitClass.SRAM_HD: 0.052,
    CircuitClass.SRAM_HC: 0.052,
    CircuitClass.SRAM_HP: 0.052 * 8 / 6,
}


@dataclass(frozen=True)
class CarriedSilicon:
    """Silicon a tile class (or a NoC overlay) declares on itself.

    ``source`` is one of ``datapath``, ``systolic_cells``, ``local_memory``,
    ``fabric_overlay``, ``function_core`` or ``noc_overlay``.
    """

    name: str
    tile_class_id: Optional[str]  # None for chip-level NoC overlays
    circuit_class: CircuitClass
    transistors_mtx: float
    source: str


def _memory_mtx(mem: LocalMemory, per_pe_count: int, owner: str) -> float:
    per_kib = SRAM_MTX_PER_KIB.get(mem.circuit_class)
    if per_kib is None:
        raise SiliconMathError(
            f"{owner}: local_memory {mem.level.value!r} uses non-SRAM library "
            f"{mem.circuit_class.value!r}"
        )
    count = per_pe_count if mem.per == LocalMemoryScope.PE else 1
    return mem.kib * count * per_kib


def _datapath_mtx_per_pe(tile: KPUTileSpec) -> Optional[float]:
    dp = tile.datapath
    if dp is None:
        return None
    if dp.mtx_per_pe is not None:
        return dp.mtx_per_pe
    unit_mtx = [u.mtx for u in dp.functional_units if u.mtx is not None]
    return sum(unit_mtx) if unit_mtx else None


def _node_for(ref_node_id: str, nodes: Optional[Mapping[str, ProcessNodeEntry]]):
    if nodes is None:
        from embodied_schemas import load_process_nodes

        nodes = load_process_nodes()
    node = nodes.get(ref_node_id)
    if node is None:
        raise SiliconMathError(f"reference process node {ref_node_id!r} is not in the catalog")
    return node


def carried_silicon(
    cp: ComputeProduct, nodes: Optional[Mapping[str, ProcessNodeEntry]] = None
) -> list[CarriedSilicon]:
    """Every piece of silicon the product's tiles (and NoC overlays) declare
    on themselves, as transistor counts. Empty for catalog SKUs.

    * pe_fabric: the datapath (``mtx_per_pe``, else the sum of unit
      ``mtx``) x PEs; each ``local_memory``; fabric overlays
      (``mtx_per_instance`` x instances x rows / cols / 1 per tile).
    * systolic: the cell unit's ``mtx`` x cells; each ``local_memory``.
    * fixed_function: the core's ``silicon`` blocks (``mtx``, or
      ``area_mm2`` at ``ref_node_id`` converted with that node's density),
      which describe the whole core, memory included. Only a core without
      ``silicon`` contributes its ``local_memory``.
    * NoC overlays: ``mtx_per_instance`` x instances.

    Everything is multiplied by ``num_tiles``. Pieces without a declared
    figure contribute nothing (no silicon is invented). ``nodes`` maps
    process-node ids for the area conversions; the catalog is loaded when
    it is None and a conversion needs it.
    """
    out: list[CarriedSilicon] = []
    for t in _kpu_block(cp).tiles:
        cid, n = t.tile_class_id, t.num_tiles
        owner = f"tile class {cid!r}"

        def add(label: str, cc: CircuitClass, mtx: float, source: str) -> None:
            if mtx > 0:
                out.append(CarriedSilicon(f"{cid}.{label}", cid, cc, mtx, source))

        if isinstance(t, KPUTileSpec):
            per_pe = _datapath_mtx_per_pe(t)
            if per_pe is not None:
                cc = t.datapath.circuit_class or t.pe_circuit_class
                add("datapath", cc, per_pe * t.total_pes, "datapath")
            for ov in (t.interconnect.overlays if t.interconnect else []):
                if ov.mtx_per_instance is None:
                    continue
                per = {OverlayScope.ROW: t.pe_array_rows, OverlayScope.COL: t.pe_array_cols,
                       OverlayScope.TILE: 1}[ov.instances_per]
                add(f"overlay.{ov.overlay_id}", ov.circuit_class,
                    ov.mtx_per_instance * ov.instances * per * n, "fabric_overlay")
            memories = t.local_memory or []
        elif isinstance(t, SystolicTile):
            if t.mac.mtx is not None:
                add("cells", t.circuit_class, t.mac.mtx * t.total_pes, "systolic_cells")
            memories = t.local_memory or []
        elif isinstance(t, FixedFunctionTile):
            core = t.core
            for blk in core.silicon or []:
                if blk.mtx is not None:
                    mtx = blk.mtx
                else:
                    ref = _node_for(blk.ref_node_id, nodes)
                    if not ref.supports(blk.circuit_class):
                        raise SiliconMathError(
                            f"{owner}: core block {blk.name!r} library "
                            f"{blk.circuit_class.value!r} is not offered by {ref.id!r}"
                        )
                    mtx = blk.area_mm2 * ref.density_for(blk.circuit_class).mtx_per_mm2
                add(f"core.{blk.name}", blk.circuit_class, mtx * n, "function_core")
            memories = [] if core.silicon else (core.local_memory or [])
        else:  # pragma: no cover - AnyKPUTile has exactly these kinds
            raise SiliconMathError(f"{owner}: unhandled tile kind {t.tile_kind!r}")

        pes = t.pes_per_tile if has_pes(t) else 1
        for mem in memories:
            add(f"memory.{mem.level.value}", mem.circuit_class,
                _memory_mtx(mem, pes, owner) * n, "local_memory")

    for ov in _kpu_block(cp).noc.overlays or []:
        if ov.mtx_per_instance is not None and ov.mtx_per_instance > 0:
            out.append(CarriedSilicon(
                f"noc_overlay.{ov.overlay_id}", None, ov.circuit_class,
                ov.mtx_per_instance * ov.instances, "noc_overlay",
            ))
    return out


def unsupported_carried_silicon(
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> list[CarriedSilicon]:
    """Tile-carried pieces whose library the SKU's process node does not
    offer (e.g. an ``sram_hp`` accumulator on a node without dual-port
    SRAM). ``resolve_carried_areas`` skips them; the library-validity
    validator reports them, as it does for silicon_bin blocks."""
    return [cs for cs in carried_silicon(cp, nodes) if not node.supports(cs.circuit_class)]


def resolve_carried_areas(
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> list[BlockArea]:
    """Areas of ``carried_silicon`` at the SKU's process node. Like
    ``resolve_all_block_areas``, skips pieces whose library the node does
    not offer (``unsupported_carried_silicon`` lists them)."""
    out: list[BlockArea] = []
    for cs in carried_silicon(cp, nodes):
        if not node.supports(cs.circuit_class):
            continue
        density = node.density_for(cs.circuit_class).mtx_per_mm2
        out.append(BlockArea(cs.name, cs.circuit_class, cs.transistors_mtx, density,
                             cs.transistors_mtx / density))
    return out


def double_counted_tile_classes(
    cp: ComputeProduct, nodes: Optional[Mapping[str, ProcessNodeEntry]] = None
) -> list[str]:
    """Tile classes whose logic is counted twice: once as tile-carried
    silicon (datapath, systolic cells or core) and once by a chip-level
    PER_PE silicon_bin block. The ``silicon_no_double_count`` validator
    (Phase C4) reports these as errors.

    Tile memory cannot be double counted: chip-level L1 / L2 cover only
    classes without ``local_memory`` (``inherits_chip_memory``).
    """
    carried = {
        cs.tile_class_id
        for cs in carried_silicon(cp, nodes)
        if cs.source in ("datapath", "systolic_cells", "function_core")
    }
    counted = set()
    for b in _kpu_die(cp).silicon_bin.blocks:
        ts = b.transistor_source
        if ts.kind == TransistorSourceKind.PER_PE and ts.count_ref:
            try:
                counted.add(resolve_tile_ref(cp, ts.count_ref.removeprefix("tile.")).tile_class_id)
            except SiliconMathError:
                continue
    return sorted(carried & counted)
