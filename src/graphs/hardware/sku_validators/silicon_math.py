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
"""

from __future__ import annotations

from dataclasses import dataclass

from embodied_schemas import ComputeProduct
from embodied_schemas.compute_product import KPUBlock
from embodied_schemas.kpu import (
    SiliconBinBlock,
    TransistorSourceKind,
)
from embodied_schemas.process_node import CircuitClass, ProcessNodeEntry


class SiliconMathError(Exception):
    """Raised when a silicon_bin block can't be resolved (bad count_ref,
    missing per_unit_mtx, etc.). Validators catch this and convert into
    Findings rather than crashing."""


def _kpu_block(cp: ComputeProduct) -> KPUBlock:
    """Return the KPUBlock from a v1 monolithic-KPU ``ComputeProduct``.

    Today every KPU ComputeProduct has exactly one Die with one
    KPUBlock; chiplet KPU products will need iteration when they land.
    """
    return cp.dies[0].blocks[0]


# ---------------------------------------------------------------------------
# Architecture-level rollups
# ---------------------------------------------------------------------------

def total_pe_count(cp: ComputeProduct) -> int:
    """Total PE count summed across every tile class."""
    return sum(t.total_pes for t in _kpu_block(cp).tiles)


def total_l1_kib(cp: ComputeProduct) -> int:
    """Total per-PE L1 SRAM in KiB across the chip."""
    return _kpu_block(cp).memory.l1_kib_per_pe * total_pe_count(cp)


def total_l2_kib(cp: ComputeProduct) -> int:
    """Total per-tile L2 SRAM in KiB across the chip."""
    block = _kpu_block(cp)
    return block.memory.l2_kib_per_tile * block.total_tiles


def total_l3_kib(cp: ComputeProduct) -> int:
    """Total per-tile L3 SRAM in KiB across the chip."""
    block = _kpu_block(cp)
    return block.memory.l3_kib_per_tile * block.total_tiles


def num_tiles_by_type(cp: ComputeProduct) -> dict[str, int]:
    """Map of tile_type -> num_tiles. Useful for ``count_ref="tile.<type>"``."""
    return {t.tile_type: t.num_tiles for t in _kpu_block(cp).tiles}


def total_pes_by_tile_type(cp: ComputeProduct) -> dict[str, int]:
    """Map of tile_type -> total PE count for that tile class."""
    return {t.tile_type: t.total_pes for t in _kpu_block(cp).tiles}


# ---------------------------------------------------------------------------
# Transistor count resolution
# ---------------------------------------------------------------------------

def resolve_block_transistors(block: SiliconBinBlock, cp: ComputeProduct) -> float:
    """Compute the absolute transistor count (in millions) for a silicon_bin
    block, given the SKU's architecture context.

    Resolution by ``transistor_source.kind``:

    * **FIXED**: returns ``mtx`` directly. ``count_ref`` is ignored.
    * **PER_PE**: ``count_ref`` is ``"tile.<tile_type>"``. Returns
      ``per_unit_mtx * total_pes(<tile_type>)``.
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
                f"block {block.name!r}: PER_PE count_ref must be 'tile.<type>', "
                f"got {ts.count_ref!r}"
            )
        tile_type = ts.count_ref.split(".", 1)[1]
        pes_by_type = total_pes_by_tile_type(cp)
        if tile_type not in pes_by_type:
            raise SiliconMathError(
                f"block {block.name!r}: unknown tile_type {tile_type!r}. "
                f"Available: {sorted(pes_by_type)}"
            )
        return float(ts.per_unit_mtx) * pes_by_type[tile_type]

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
    for block in cp.dies[0].silicon_bin.blocks:
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

    Modeled per block name (heuristic v1):

    * **PE blocks** (name starts with ``pe_``): identifies the matching
      tile class from the block's ``transistor_source.count_ref`` (must
      be ``"tile.<type>"``), then computes
      ``total_pes * ops_per_pe_per_clock * clock * energy_per_op_pj``
      using ``process_node.energy_per_op_pj["{circuit_class}:{precision}"]``.
      Returns 0 if the node has no energy entry for this class+precision.
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

    # ---- PE blocks ----
    if name.startswith("pe_"):
        ts = block.transistor_source
        if ts.kind != TransistorSourceKind.PER_PE or not ts.count_ref:
            return 0.0
        tile_type = ts.count_ref.removeprefix("tile.")
        tile = next(
            (t for t in kpu.tiles if t.tile_type == tile_type),
            None,
        )
        if tile is None:
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


def total_chip_leakage_w(cp: ComputeProduct, node: ProcessNodeEntry) -> float:
    """Sum of every block's leakage."""
    return sum(
        estimate_block_leakage_w(b, cp, node)
        for b in cp.dies[0].silicon_bin.blocks
    )
