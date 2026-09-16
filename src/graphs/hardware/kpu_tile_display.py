"""Kind-aware descriptions of KPU tiles, for the inspection CLIs
(graphs#268 C6).

``show_kpu`` and ``list_kpus`` were written when every tile was a PE
fabric, so they read ``tile.pe_array_rows``, ``tile.pe_circuit_class`` and
``tile.total_pes`` straight off each tile. Those exist only on
``KPUTileSpec``: a ``SystolicTile`` names its array ``array_rows`` /
``array_cols`` and its library ``circuit_class``, and a
``FixedFunctionTile`` has neither an array nor PEs at all, so both CLIs
raise ``AttributeError`` on a heterogeneous SKU.

This module is the one place that knows what each kind looks like. It
reports what a tile *is* rather than what a PE fabric would be:

- ``tile_pe_count`` is 0 for a fixed-function tile, not an error. Its
  silicon is a function core, and summing PEs across a checkerboard should
  not pretend otherwise.
- ``tile_geometry`` and ``tile_circuit_class`` return None where the kind
  has no such thing, so a renderer prints a dash instead of a wrong number.
- ``tile_detail`` is the one-line "what is this tile" string: the datapath
  and schedule for a PE fabric, the dataflow and MAC for a systolic array,
  the function and its throughput for a fixed-function core.

Leaf module: imports only embodied-schemas, like ``kpu_access``.
"""

from __future__ import annotations

from typing import Optional, Tuple

from embodied_schemas.compute_product import KPUBlock
from embodied_schemas.kpu import FixedFunctionTile, KPUTileSpec, SystolicTile

#: Column header and width for the kind column shared by the CLIs.
KIND_LABELS = {
    "pe_fabric": "pe",
    "systolic": "systolic",
    "fixed_function": "fixed-fn",
}


def tile_kind(tile) -> str:
    """The tile's kind as a plain string.

    ``tile_kind`` has a default on every tile class, so this is really just
    ``.value``; it exists so callers never touch the enum directly.
    """
    return tile.tile_kind.value


def is_programmable(tile) -> bool:
    """True for kinds with PEs / cells running precision-typed ops."""
    return isinstance(tile, (KPUTileSpec, SystolicTile))


def tile_pe_count(tile) -> int:
    """PEs in one tile of this class, or 0 when the kind has none.

    A fixed-function tile has no PE array; counting it as zero is what
    lets a caller sum across a mixed checkerboard.
    """
    return tile.pes_per_tile if is_programmable(tile) else 0


def total_pe_count(block: KPUBlock) -> int:
    """PEs across the whole block, fixed-function tiles contributing none."""
    return sum(tile_pe_count(t) * t.num_tiles for t in block.tiles)


def tile_geometry(tile) -> Optional[Tuple[int, int]]:
    """The tile's array shape, or None for a kind that has no array."""
    if isinstance(tile, SystolicTile):
        return (tile.array_rows, tile.array_cols)
    if isinstance(tile, KPUTileSpec):
        return (tile.pe_array_rows, tile.pe_array_cols)
    return None


def tile_circuit_class(tile) -> Optional[str]:
    """The standard-cell library the tile's datapath is built from, or None
    for a fixed-function tile, whose core carries its own libraries."""
    if isinstance(tile, SystolicTile):
        return tile.circuit_class.value
    if isinstance(tile, KPUTileSpec):
        return tile.pe_circuit_class.value
    return None


def tile_geometry_str(tile) -> str:
    geometry = tile_geometry(tile)
    return "-" if geometry is None else f"{geometry[0]}x{geometry[1]}"


#: Suffix marking a footprint that swallows the memory cells it covers.
ABSORBS_MARK = "*"


def absorbs_memory_cells(tile) -> bool:
    footprint = getattr(tile, "footprint", None)
    return bool(footprint is not None and footprint.absorbs_memory_cells)


def tile_footprint_str(tile) -> str:
    """The tile's checkerboard footprint in sites, e.g. ``2x2``.

    A trailing ``*`` marks a footprint that absorbs the memory cells it
    covers; kept to one character so the column stays aligned, with the
    renderer printing a legend.
    """
    footprint = getattr(tile, "footprint", None)
    if footprint is None:
        return "1x1"
    mark = ABSORBS_MARK if footprint.absorbs_memory_cells else ""
    return f"{footprint.rows}x{footprint.cols}{mark}"


def tile_sites(tile) -> int:
    """Checkerboard sites one tile of this class occupies.

    A tile is one site unless it declares a larger footprint: the fixture's
    VIO core is 2x2, so its single tile costs four sites.
    """
    footprint = getattr(tile, "footprint", None)
    if footprint is None:
        return 1
    return footprint.rows * footprint.cols


def occupied_sites(block: KPUBlock) -> int:
    """Compute sites the block's tiles occupy, footprints included."""
    return sum(tile_sites(t) * t.num_tiles for t in block.tiles)


def fixed_function_throughput_str(tile) -> str:
    """A fixed-function core's rate, in whichever direction it was
    specified: ``1 pixel/clock``, or ``1 frame / 880000 clocks`` for a core
    slow enough that cycles-per-unit is the natural figure."""
    throughput = tile.core.throughput
    if throughput.units_per_clock is None and throughput.cycles_per_unit:
        return f"1 {throughput.unit.value} / {throughput.cycles_per_unit:g} clocks"
    return f"{tile.units_per_clock:g} {throughput.unit.value}/clock"


def _operand_formats(tile) -> str:
    """The operand formats a PE-fabric datapath declares, for a class whose
    ops are not precision-keyed (min-plus, abs-diff) and so has an empty
    ``ops_per_tile_per_clock``."""
    if tile.datapath is None:
        return ""
    formats = []
    for unit in tile.datapath.functional_units:
        for mode in unit.modes:
            if mode.operand_format not in formats:
                formats.append(mode.operand_format)
    return ", ".join(formats)


def tile_ops_str(tile) -> str:
    """Precision-typed ops per tile per clock.

    A fixed-function tile has none -- its work is counted in pixels or
    frames, so its throughput stands in. A min-plus or abs-diff fabric has
    none either, because its ops are not keyed by a numeric format; its
    operand formats stand in.
    """
    ops = tile.ops_per_tile_per_clock
    if ops:
        return ", ".join(f"{p}={int(v)}" for p, v in ops.items())
    if isinstance(tile, FixedFunctionTile):
        return fixed_function_throughput_str(tile)
    if isinstance(tile, KPUTileSpec):
        formats = _operand_formats(tile)
        if formats:
            return f"(not precision-keyed; operands: {formats})"
    return "-"


def tile_detail(tile) -> str:
    """One line saying what this tile actually is, per kind."""
    if isinstance(tile, SystolicTile):
        kernels = ", ".join(tile.supported_kernels) if tile.supported_kernels else "-"
        formats = ", ".join(
            dict.fromkeys(mode.operand_format for mode in tile.mac.modes)
        )
        return (
            f"dataflow={tile.dataflow.value}, cell={tile.mac.op.value} ({formats}), "
            f"kernels={kernels}"
        )
    if isinstance(tile, FixedFunctionTile):
        return (
            f"function={tile.core.function_id}, "
            f"{fixed_function_throughput_str(tile)}"
        )
    # A catalog PE fabric predating the tile-class library has no explicit
    # datapath: its ops come straight from ops_per_tile_per_clock.
    datapath = tile.datapath
    if datapath is None:
        return f"schedule={tile.schedule_class.value}"
    ops = ", ".join(sorted({u.op.value for u in datapath.functional_units}))
    return (
        f"datapath={datapath.datapath_id} ({ops}), "
        f"schedule={tile.schedule_class.value}"
    )


def kind_counts(block: KPUBlock) -> dict:
    """Tiles per kind, in the canonical kind order."""
    counts = {}
    for t in block.tiles:
        counts[tile_kind(t)] = counts.get(tile_kind(t), 0) + t.num_tiles
    return {k: counts[k] for k in KIND_LABELS if k in counts}


def kind_summary(block: KPUBlock) -> str:
    """A compact per-kind tile census, e.g. ``pe:38 systolic:4 fixed-fn:3``."""
    counts = kind_counts(block)
    if not counts:
        return "-"
    return " ".join(f"{KIND_LABELS[k]}:{n}" for k, n in counts.items())


def is_heterogeneous(block: KPUBlock) -> bool:
    """True when the block carries more than one tile kind.

    The same test the mapper uses for its capability-aware pools
    (``HardwareResourceModel.is_heterogeneous_kpu``), phrased on the
    catalog product rather than the resource model.
    """
    return len(kind_counts(block)) > 1
