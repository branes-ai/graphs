"""KPU tile classes as engines (graphs#268 E3).

The SoC micro-architecture study (graphs#269) models a die as a set of
*engines* and maps workload stages onto them. Its plan says each KPU tile
class is a separate engine, and this module is that export: it turns a
heterogeneous KPU's tile classes into descriptors an engine-efficiency
table and a stage mapper can consume, without either of those existing
yet.

An engine descriptor answers the three questions a mapper asks:

- **What can it run?** The precisions it declares ops for, the kernels a
  systolic class supports, or the function a fixed-function core
  implements. A class that declares none of these can run no stage, and
  saying so is more useful than a zero.
- **What does it cost?** Ops per second and energy per op at the SKU's
  node and default clock, plus silicon per tile. Fixed-function cores are
  quoted in their own work unit as well, because pixels are what they are
  published in.
- **What is its precision floor?** The narrowest format it supports. A
  stage needing more range than that cannot run on it, however fast it
  looks -- which is the finding ``precision_floor_findings`` produces.

**Segments** are the other half. A stream-linked chain of fixed-function
cores absorbs several pipeline stages, and their intermediate results
never reach DRAM. The SoC plan calls these workload segments; here they
are read back off the NoC overlays that declare them, so a mapper can
treat the chain as one placement rather than three.

This module describes; it does not evaluate. ``kpu_tile_ladder`` is the
analysis that prices one function across these engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from embodied_schemas import ComputeProduct
from embodied_schemas.datapath import AbsoluteEnergy, RelativeEnergy
from embodied_schemas.kpu import FixedFunctionTile, KPUTileSpec, SystolicTile
from embodied_schemas.overlay import NoCOverlayKind
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.hardware import kpu_tile_display as display
from graphs.hardware.kpu_access import kpu_block_of
from graphs.hardware.kpu_power_model import fixed_function_pj_per_unit
from graphs.hardware.sku_validators.silicon_math import carried_silicon

#: Operand formats ordered by the range they carry, narrowest first. The
#: *precision floor* of an engine is the narrowest format it supports, and
#: a stage needing a wider one cannot run there. LNS formats sit beside
#: their float equivalents: lns16 carries roughly bf16's dynamic range.
_FORMAT_RANK: Dict[str, int] = {
    "int4": 0, "uint4": 0,
    "int8": 1, "uint8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "lns8": 1,
    "int16": 2, "uint16": 2, "fp16": 2, "bf16": 2, "lns16": 2,
    "int32": 3, "fp32": 3,
    "int64": 4, "fp64": 4,
}


def format_rank(operand_format: str) -> Optional[int]:
    """Where a format sits on the range ladder, or None if unknown."""
    return _FORMAT_RANK.get(operand_format)


def class_area_mm2(
    cp: ComputeProduct, node: ProcessNodeEntry, tile_class_id: str
) -> Optional[float]:
    """Silicon one tile of a class occupies, from its carried silicon.

    None when the class declares none -- a legacy PE fabric whose area
    lives in a silicon_bin ``per_pe`` block, say. Pieces whose library the
    node does not offer are skipped, as ``resolve_carried_areas`` does.
    """
    total = 0.0
    found = False
    for cs in carried_silicon(cp):
        if cs.tile_class_id != tile_class_id or not node.supports(cs.circuit_class):
            continue
        total += cs.transistors_mtx / node.density_for(cs.circuit_class).mtx_per_mm2
        found = True
    if not found:
        return None
    tile = next(
        (t for t in kpu_block_of(cp).tiles if t.tile_class_id == tile_class_id), None
    )
    if tile is None or not tile.num_tiles:
        return None
    return total / tile.num_tiles


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnginePrecision:
    """One format an engine runs, with its throughput and energy."""

    operand_format: str
    ops_per_tile_per_clock: float
    ops_per_second_per_tile: float
    pj_per_op: Optional[float]

    @property
    def tops_per_watt(self) -> Optional[float]:
        """Ops per joule, in TOPS/W -- the figure an engine table compares."""
        if not self.pj_per_op:
            return None
        return 1.0 / (self.pj_per_op * 1e-12) / 1e12


@dataclass(frozen=True)
class EngineDescriptor:
    """One KPU tile class, as the SoC analyzer's mapper sees it."""

    engine_id: str  # the tile_class_id
    engine_kind: str  # pe_fabric | systolic | fixed_function
    tile_type: str
    num_tiles: int
    sites_per_tile: int
    clock_hz: float
    area_mm2_per_tile: Optional[float]
    precisions: Tuple[EnginePrecision, ...]
    supported_kernels: Tuple[str, ...]
    #: Fixed-function only: what it computes and at what rate and cost.
    function_id: Optional[str] = None
    work_unit: Optional[str] = None
    units_per_second: Optional[float] = None
    pj_per_unit: Optional[float] = None
    #: The core's stated configuration limits, e.g. ``{"max_width": 1920,
    #: "max_fps": 30, "disparities": 128}``. A core rated for one
    #: configuration does not silently serve a larger one, so a consumer can
    #: check a workload against these rather than against throughput alone.
    config_limits: Mapping[str, Any] = field(default_factory=dict)
    confidence: str = "THEORETICAL"
    provenance: str = ""
    notes: str = ""

    @property
    def precision_floor(self) -> Optional[str]:
        """The narrowest format this engine supports.

        None for a fixed-function core, which runs one function at whatever
        internal precision it was built for, and for a class whose ops are
        not format-keyed at all.
        """
        ranked = [
            (format_rank(p.operand_format), p.operand_format)
            for p in self.precisions
            if format_rank(p.operand_format) is not None
        ]
        return min(ranked)[1] if ranked else None

    @property
    def formats(self) -> Tuple[str, ...]:
        return tuple(p.operand_format for p in self.precisions)

    def supports(self, operand_format: str) -> bool:
        return operand_format in self.formats

    def supports_at_least(self, operand_format: str) -> bool:
        """Whether this engine carries at least ``operand_format``'s range.

        A stage needing fp32 does not become runnable because an engine is
        fast at int8; it needs a format at least as wide.
        """
        needed = format_rank(operand_format)
        if needed is None:
            return self.supports(operand_format)
        return any(
            (rank := format_rank(fmt)) is not None and rank >= needed
            for fmt in self.formats
        )

    @property
    def is_programmable(self) -> bool:
        return self.engine_kind in ("pe_fabric", "systolic")

    @property
    def total_area_mm2(self) -> Optional[float]:
        if self.area_mm2_per_tile is None:
            return None
        return self.area_mm2_per_tile * self.num_tiles


def _op_energy_pj_for_format(
    tile, node: ProcessNodeEntry, operand_format: str
) -> Optional[float]:
    """Cheapest declared energy per op on ``tile`` for one format.

    An anchor in ``energy_per_op_pj`` is one invocation, which the D3
    convention counts as ``ops_per_invocation`` ops.
    """
    if isinstance(tile, SystolicTile):
        units = [tile.mac]
    elif isinstance(tile, KPUTileSpec) and tile.datapath is not None:
        units = list(tile.datapath.functional_units)
    else:
        return None
    best: Optional[float] = None
    for unit in units:
        for mode in unit.modes:
            if mode.operand_format != operand_format:
                continue
            energy = mode.energy
            per_invocation: Optional[float] = None
            if isinstance(energy, RelativeEnergy):
                anchor = node.energy_per_op_pj.get(energy.anchor)
                if anchor is not None:
                    per_invocation = energy.ratio * anchor * 2.0
            elif isinstance(energy, AbsoluteEnergy):
                per_invocation = energy.pj
            if per_invocation is None:
                continue
            per_op = per_invocation / max(1.0, float(unit.ops_per_invocation or 2.0))
            if best is None or per_op < best:
                best = per_op
    return best


def _default_clock_hz(cp: ComputeProduct) -> float:
    for profile in cp.power.thermal_profiles:
        if profile.name == cp.power.default_thermal_profile:
            return profile.clock_mhz * 1e6
    return cp.dies[0].clocks.base_clock_mhz * 1e6


def describe_engines(
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
) -> Tuple[EngineDescriptor, ...]:
    """Every tile class of ``cp``, as an engine descriptor.

    In declaration order, which is the SKU author's own priority and keeps
    the export reproducible.
    """
    block = kpu_block_of(cp)
    clock_hz = _default_clock_hz(cp)
    out: List[EngineDescriptor] = []
    for tile in block.tiles:
        precisions = tuple(
            EnginePrecision(
                operand_format=fmt,
                ops_per_tile_per_clock=ops,
                ops_per_second_per_tile=ops * clock_hz,
                pj_per_op=_op_energy_pj_for_format(tile, node, fmt),
            )
            for fmt, ops in tile.ops_per_tile_per_clock.items()
        )
        kind = display.tile_kind(tile)
        common = dict(
            engine_id=tile.tile_class_id,
            engine_kind=kind,
            tile_type=tile.tile_type,
            num_tiles=tile.num_tiles,
            sites_per_tile=display.tile_sites(tile),
            clock_hz=clock_hz,
            area_mm2_per_tile=class_area_mm2(cp, node, tile.tile_class_id),
            precisions=precisions,
            supported_kernels=tuple(
                k.value for k in (getattr(tile, "supported_kernels", None) or [])
            ),
        )
        if isinstance(tile, FixedFunctionTile):
            core = tile.core
            pj = fixed_function_pj_per_unit(core, node, nodes)
            out.append(EngineDescriptor(
                **common,
                function_id=core.function_id,
                work_unit=core.throughput.unit.value,
                units_per_second=(tile.units_per_clock or 0.0) * clock_hz,
                pj_per_unit=pj,
                config_limits=dict(core.contract.config_limits or {}),
                confidence=core.confidence.value.upper(),
                provenance=core.energy.source or core.source,
                notes=(
                    "runs one function; it has no operand-format choice, so "
                    "no precision floor"
                ),
            ))
            continue
        notes = ""
        if not precisions:
            # A min-plus or abs-diff fabric: real compute, but its ops are
            # not keyed by a numeric format, so a precision-indexed mapper
            # sees nothing. Better said than silently zero.
            notes = (
                "ops are not precision-keyed (tropical / comparison "
                "arithmetic), so this engine carries no per-format throughput"
            )
        out.append(EngineDescriptor(
            **common,
            provenance=(
                f"tile class {tile.tile_class_id!r} at {node.node_name}, "
                f"{clock_hz / 1e6:.0f} MHz"
            ),
            notes=notes,
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A stream-linked chain of engines that absorbs several stages.

    The SoC plan's workload ``segments``: because the chain's intermediate
    results never reach DRAM, a mapper should place the whole chain as one
    unit rather than three stages that happen to be adjacent.
    """

    segment_id: str
    engine_ids: Tuple[str, ...]
    function_ids: Tuple[str, ...]
    #: Bytes that stay on-chip because of the links, per producer work
    #: unit, as ``((unit, bytes), ...)``. Keyed by unit rather than summed:
    #: a chain can mix per-pixel and per-frame stages, and adding those
    #: together would be meaningless.
    absorbed_bytes_by_unit: Tuple[Tuple[str, float], ...]

    @property
    def num_stages(self) -> int:
        return len(self.engine_ids)

    @property
    def work_units(self) -> Tuple[str, ...]:
        """The work units the absorbed traffic is quoted in."""
        return tuple(unit for unit, _ in self.absorbed_bytes_by_unit)

    def absorbed_bytes(self, work_unit: str) -> float:
        return dict(self.absorbed_bytes_by_unit).get(work_unit, 0.0)


def describe_segments(cp: ComputeProduct) -> Tuple[Segment, ...]:
    """The stream-linked chains ``cp`` declares, in overlay order."""
    block = kpu_block_of(cp)
    by_id = {t.tile_class_id: t for t in block.tiles}
    out: List[Segment] = []
    for overlay in block.noc.overlays or []:
        if overlay.kind != NoCOverlayKind.STREAM_LINK:
            continue
        engine_ids = tuple(overlay.endpoints)
        tiles = [by_id.get(eid) for eid in engine_ids]
        # Each link absorbs its *producer's* output, written and read back,
        # in that producer's own work unit. The last engine in the chain
        # produces nothing that stays on-chip, so it is not a producer.
        absorbed: Dict[str, float] = {}
        for producer in tiles[:-1]:
            core = getattr(producer, "core", None)
            io = getattr(core, "io", None)
            if io is None:
                continue
            unit = core.throughput.unit.value
            absorbed[unit] = absorbed.get(unit, 0.0) + io.output_bytes_per_unit * 2
        out.append(Segment(
            segment_id=overlay.overlay_id,
            engine_ids=engine_ids,
            function_ids=tuple(
                t.core.function_id for t in tiles
                if isinstance(t, FixedFunctionTile)
            ),
            absorbed_bytes_by_unit=tuple(sorted(absorbed.items())),
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Precision-floor findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionFinding:
    """A workload requirement measured against what the engines offer."""

    requirement: str  # the stage or kernel that needs it
    operand_format: str
    served_by: Tuple[str, ...]
    severity: str  # OK | WARNING | ERROR
    message: str


def precision_floor_findings(
    engines: Sequence[EngineDescriptor],
    requirements: Mapping[str, str],
) -> Tuple[PrecisionFinding, ...]:
    """Check each requirement against the engines' precision floors.

    ``requirements`` maps a stage or kernel name to the narrowest format it
    can tolerate. An engine serves it only by carrying at least that much
    range -- being fast at a narrower format does not help, which is the
    trap this exists to catch. A requirement nothing serves is an ERROR;
    one served only by a fixed-function core is a WARNING, because that
    core runs its own function and not arbitrary work.
    """
    out: List[PrecisionFinding] = []
    for name, operand_format in requirements.items():
        served = tuple(
            e.engine_id for e in engines
            if e.is_programmable and e.supports_at_least(operand_format)
        )
        if served:
            out.append(PrecisionFinding(
                requirement=name, operand_format=operand_format,
                served_by=served, severity="OK",
                message=(
                    f"{name!r} needs {operand_format}: served by "
                    f"{', '.join(served)}"
                ),
            ))
            continue
        floors = sorted(
            {e.precision_floor for e in engines if e.precision_floor} or {"none"}
        )
        out.append(PrecisionFinding(
            requirement=name, operand_format=operand_format, served_by=(),
            severity="ERROR",
            message=(
                f"{name!r} needs {operand_format}, which no programmable "
                f"engine on this die carries. Engine precision floors: "
                f"{', '.join(floors)}. A narrower format cannot stand in for "
                f"it, however fast the engine is."
            ),
        ))
    return tuple(out)
