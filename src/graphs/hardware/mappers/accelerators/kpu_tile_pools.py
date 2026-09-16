"""Capability-aware tile pools for the KPU mapper (graphs#268 C5b).

A uniform KPU is one pool: every tile runs every precision the chip claims,
so the mapper can treat ``compute_units`` as a flat count. A heterogeneous
KPU cannot. Its checkerboard mixes PE-fabric classes with different
datapaths (INT8 MAC, LNS16, min-plus), systolic classes, and
fixed-function tiles that run no precision-typed ops at all. Allocating a
GEMM across "all 64 tiles" there would hand work to tiles that physically
cannot execute it.

This module answers two questions for one (subgraph, precision):

1. *Which* tiles may run it -- the classes whose ``ops_per_tile_per_clock``
   lists the precision. Fixed-function tiles never appear, because the
   loader does not turn them into ``TileSpecialization`` at all.
2. *In what order* -- a GEMM or convolution prefers the systolic classes
   (a weight-stationary array is what they are for) and spills to the PE
   fabric; everything else prefers the PE fabric, whose output-stationary
   wavefront amortizes fill/drain.

The pool then reports the throughput of exactly the tiles it handed out,
rather than a fraction of the chip-wide figure: on a heterogeneous chip
those differ by more than the tile ratio, because the classes have
different array sizes and clocks.

Only heterogeneous models use this path (see
``HardwareResourceModel.is_heterogeneous_kpu``); legacy SKU mapper results
are pinned by the KPU golden snapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from graphs.core.structures import OperationType
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    KPUComputeResource,
    Precision,
    TileSpecialization,
)

# Operations a weight-stationary systolic array is built for: a dense
# matrix product, or a convolution lowered to one. The depthwise variant is
# deliberately absent -- its per-channel reduction leaves a systolic array
# almost entirely idle, so it stays on the PE fabric.
SYSTOLIC_PREFERRED_OPS = frozenset(
    {
        OperationType.CONV2D,
        OperationType.CONV2D_POINTWISE,
        OperationType.LINEAR,
        OperationType.MATMUL,
        OperationType.MULTIHEAD_ATTENTION,
    }
)

SYSTOLIC_KIND = "systolic"
PE_FABRIC_KIND = "pe_fabric"


def prefers_systolic(operation_types: Sequence[OperationType]) -> bool:
    """True when a fused subgraph's dominant work is a dense matrix product.

    A fusion group is named for its anchor op (conv + bias + relu), so one
    matching op in the group is enough.
    """
    return any(op in SYSTOLIC_PREFERRED_OPS for op in operation_types)


@dataclass(frozen=True)
class TileAllocation:
    """Tiles handed to one subgraph, broken out by class."""

    per_class: Tuple[Tuple[TileSpecialization, int], ...]
    ops_per_sec: float  # sustained throughput of exactly these tiles

    @property
    def num_tiles(self) -> int:
        return sum(n for _, n in self.per_class)

    @property
    def threads(self) -> int:
        return sum(spec.pe_count * n for spec, n in self.per_class)

    @property
    def classes(self) -> Tuple[str, ...]:
        return tuple(spec.tile_type for spec, n in self.per_class if n > 0)


@dataclass(frozen=True)
class TilePool:
    """The tiles that may run one (subgraph, precision), in preference order."""

    precision: Precision
    specializations: Tuple[TileSpecialization, ...]
    # Utilization denominator: every tile with precision-typed ops at this
    # thermal point. Smaller than ``compute_units``, which counts the
    # fixed-function tiles, and than the programmable tile count, which
    # includes classes whose ops are not in the ``Precision`` enum (LNS,
    # min-plus) and so have no specialization.
    fabric_tiles: int
    derate: float  # effective / sustained at this thermal point
    preferred_kind: str
    capable: bool  # False when no class runs this precision

    @property
    def num_tiles(self) -> int:
        return sum(spec.num_tiles for spec in self.specializations)

    @property
    def threads_per_tile(self) -> int:
        """PEs in the preferred class -- what the mapper divides a thread
        count by to size an allocation."""
        return self.specializations[0].pe_count if self.specializations else 1

    def allocate(self, threads_required: int, min_tiles: int = 1) -> TileAllocation:
        """Take tiles from the pool in preference order.

        Each class is filled before spilling to the next, so a GEMM that
        fits the systolic tiles never touches the PE fabric. ``min_tiles``
        is the memory-tiling floor (how many data tiles must be resident at
        once); the allocation satisfies whichever demand is larger.
        """
        per_class: List[Tuple[TileSpecialization, int]] = []
        threads_left = max(0, threads_required)
        tiles_taken = 0
        for spec in self.specializations:
            if threads_left <= 0 and tiles_taken >= min_tiles:
                break
            by_threads = math.ceil(threads_left / spec.pe_count) if spec.pe_count else 0
            by_tiling = max(0, min_tiles - tiles_taken)
            take = min(spec.num_tiles, max(by_threads, by_tiling))
            if take <= 0:
                continue
            per_class.append((spec, take))
            tiles_taken += take
            threads_left = max(0, threads_left - take * spec.pe_count)
        if not per_class and self.specializations:
            # Always hand out at least one tile, as the flat mapper does.
            per_class = [(self.specializations[0], 1)]
        return TileAllocation(
            per_class=tuple(per_class),
            ops_per_sec=self._ops_per_sec(per_class),
        )

    def _ops_per_sec(self, per_class: Sequence[Tuple[TileSpecialization, int]]) -> float:
        """Sustained ops/sec of the allocated tiles, derated by the thermal
        point's measured efficiency.

        This mirrors ``KPUComputeResource.calc_sustained_ops`` restricted to
        the tiles actually allocated, so a 4-tile systolic allocation is
        charged the systolic array's throughput, not 4/64 of the chip's.
        """
        total = 0.0
        for spec, n in per_class:
            ops_per_clock = spec.ops_per_tile_per_clock.get(self.precision)
            if not ops_per_clock:
                continue
            total += (
                n
                * ops_per_clock
                * spec.clock_domain.sustained_clock_hz
                * spec.optimization_level.get(self.precision, 1.0)
            )
        return total * self.derate


def _derate(resource_model: HardwareResourceModel, thermal_profile: Optional[str],
            precision: Precision) -> float:
    """``effective / sustained`` at this thermal point: the measured
    efficiency factor and tile utilization, with the clock divided out."""
    points = resource_model.thermal_operating_points or {}
    point = points.get(thermal_profile) if thermal_profile else None
    spec = point.performance_specs.get(precision) if point else None
    if spec is None:
        return 1.0
    sustained = spec.sustained_ops_per_sec
    if sustained <= 0:
        return 1.0
    return spec.effective_ops_per_sec / sustained


def build_tile_pool(
    resource_model: HardwareResourceModel,
    compute_resource: KPUComputeResource,
    precision: Precision,
    thermal_profile: Optional[str] = None,
    prefer_systolic: bool = False,
) -> TilePool:
    """The pool of tiles that can run ``precision``, systolic classes first
    when the work is a dense matrix product.

    When no class runs the precision the pool is marked ``capable=False``
    and falls back to every class, leaving the caller's emulation penalty
    to express the cost -- the same shape as the flat mapper's
    unsupported-precision path.
    """
    kinds: Dict[str, str] = resource_model.tile_kind_by_tile_type or {}
    capable_specs = compute_resource.get_tiles_for_precision(precision)
    capable = bool(capable_specs)
    specs = capable_specs or list(compute_resource.tile_specializations)

    def kind_of(spec: TileSpecialization) -> str:
        return kinds.get(spec.tile_type, PE_FABRIC_KIND)

    preferred = SYSTOLIC_KIND if prefer_systolic else PE_FABRIC_KIND
    # Stable: the preferred kind keeps declaration order, then the rest.
    ordered = [s for s in specs if kind_of(s) == preferred]
    ordered += [s for s in specs if kind_of(s) != preferred]
    if not ordered:
        preferred = PE_FABRIC_KIND
    elif kind_of(ordered[0]) != preferred:
        preferred = kind_of(ordered[0])

    return TilePool(
        precision=precision,
        specializations=tuple(ordered),
        fabric_tiles=sum(s.num_tiles for s in compute_resource.tile_specializations),
        derate=_derate(resource_model, thermal_profile, precision),
        preferred_kind=preferred,
        capable=capable,
    )
