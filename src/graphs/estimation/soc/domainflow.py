"""What a domain-flow KPU fabric can do at best, per kernel class
(graphs#269 Phase 6.4).

No silicon has run a KPU kernel, so no efficiency exists for one and the
analyzer prices every KPU stage as a gap (P6-D1). PR 6.1 answered the
comparison question the other way round: what each configuration would have
to *sustain*. This answers the half of the remaining question that can be
answered without measuring anything: **what the fabric could sustain at
best.**

Everything here is an upper bound, and each bound is sound on its own:

* **Wavefront.** A PE-fabric tile is an ``R x C`` mesh of FMAs running an
  output-stationary schedule with a stated fill and drain. A GEMM is tiled
  over the array, so it takes ``ceil(M/R) x ceil(N/C)`` passes of ``K``
  cycles, plus fill and drain per pass, spread over the tiles the format
  runs on. Quantization and pipeline overhead are the only losses counted:
  nothing else can make it *faster*.
* **Compulsory traffic.** Every operand and result crosses DRAM at least
  once. At the SKU's stated DRAM bandwidth that time bounds the call, and
  the analyzer's own model overlaps compute with memory (``max``), so the
  bound is ``compute time at peak / DRAM time``. Re-streaming a working set
  that does not fit on chip only adds traffic, so this stays an upper bound.
* **Operand delivery.** Where a tile class states its fabric interconnect,
  the bits per clock reaching the array bound the MACs per clock it can
  feed. Most catalog tiles state none, and then this bound is a gap, not
  an assumption.

The ceiling is the least of the bounds that have data. It is THEORETICAL
and it is **not an efficiency**: it says what cannot be exceeded, never
what will be achieved. It is deliberately not written as an
``EfficiencyTable``, so no analysis can price a stage with it.

**Shapes** are the benchmark suite's (``graphs.benchmarks.soc_kernels``),
so a ceiling and a measurement on another engine describe the same work.
A kernel class with no domain-flow schedule stated here -- FFT, Cholesky,
KKT, scatter-add and the classes with no kernel at all -- gets no ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from graphs.core.confidence import EstimationConfidence

from .efficiency import KernelClass
from .mapping import estimation_confidence
from graphs.hardware.soc import Confidence

#: Bytes per element of each format the analyzer models.
BYTES: Dict[str, int] = {"int8": 1, "fp16": 2, "fp32": 4, "fp64": 8}

#: Ops per MAC: the workload counts a multiply-accumulate as two.
OPS_PER_MAC = 2.0


@dataclass(frozen=True)
class Shape:
    """One kernel's work, sized as the benchmark suite sizes it."""

    kernel_class: KernelClass
    name: str
    #: ``(M, N, K)`` GEMMs the schedule runs, in order.
    gemms: Tuple[Tuple[int, int, int], ...]
    #: Elements that must cross DRAM at least once (operands plus results).
    compulsory_elements: int
    #: Ops the workload counts for one call (2 per MAC, or the stated rule).
    ops_per_call: float
    #: MACs the array actually issues; equal to ops/2 for a pure GEMM.
    source: str
    #: Ops per PE per clock this kernel can use, as a fraction of an FMA's
    #: two. An elementwise pass uses the adder only.
    ops_per_pe_fraction: float = 1.0

    @property
    def macs(self) -> float:
        return sum(m * n * k for m, n, k in self.gemms)


#: The benchmark suite's shapes (``soc_kernels.kernel_suite``), structured.
#: ``tests/estimation/test_soc_domainflow.py`` holds them to it.
SHAPES: Tuple[Shape, ...] = (
    Shape(KernelClass.DENSE_CONV_GEMM, "gemm", ((2048, 2048, 2048),),
          3 * 2048 * 2048, 2.0 * 2048 ** 3,
          "soc_kernels gemm 2048x2048x2048; operands A, B and C cross DRAM once"),
    Shape(KernelClass.DENSE_CONV_GEMM, "conv3x3", ((80 * 80, 256, 256 * 9),),
          256 * 80 * 80 + 256 * 256 * 9 + 256 * 80 * 80,
          2.0 * 256 * 256 * 9 * 80 * 80,
          "soc_kernels conv3x3 256->256 at 80x80, lowered to a GEMM: M = H x W, "
          "N = Cout, K = Cin x 9; input, weights and output cross DRAM once"),
    Shape(KernelClass.ATTENTION_PREFILL, "sdpa",
          ((16 * 1024, 1024, 64), (16 * 1024, 64, 1024)),
          4 * 16 * 1024 * 64, 2 * 2.0 * 16 * 1024 * 1024 * 64,
          "soc_kernels sdpa 16 heads x 1024 x 64: scores then context, two GEMMs; "
          "Q, K, V and the output cross DRAM once and the scores stay on chip"),
    Shape(KernelClass.WEIGHT_STREAM_DECODE, "gemv", ((1, 4096, 4096),),
          4096 * 4096 + 2 * 4096, 2.0 * 4096 * 4096,
          "soc_kernels gemv 1x4096 . 4096x4096: the weight matrix crosses DRAM once"),
    Shape(KernelClass.ELEMENTWISE_NORM, "layernorm", ((4096, 4096, 1),),
          2 * 4096 * 4096, 5.0 * 4096 * 4096,
          "soc_kernels layernorm 4096x4096: read and write once; the array's "
          "multiplier is idle, so a PE contributes one op per clock, not two",
          ops_per_pe_fraction=0.5),
)

#: Kernel classes this model states no schedule for, and why.
NO_SCHEDULE: Dict[KernelClass, str] = {
    KernelClass.FFT: "a radix-2 FFT is not a polyhedral GEMM schedule on this fabric",
    KernelClass.SMALL_DENSE_LINALG: "a Cholesky factorization's dependences are triangular, "
                                    "not a rectangular wavefront",
    KernelClass.SMALL_QP: "an LU on a KKT system is triangular, as above",
    KernelClass.SPARSE_HASH_SCATTER: "scatter-add is addressing, not a wavefront",
    KernelClass.COST_VOLUME_DP: "no representative kernel exists yet",
    KernelClass.FEATURE_TRACK: "no representative kernel exists yet",
    KernelClass.KNN_TREE: "no representative kernel exists yet",
    KernelClass.RAYCAST: "no representative kernel exists yet",
    KernelClass.WAVEFRONT: "no representative kernel exists yet",
    KernelClass.GRAPH_SEARCH: "no representative kernel exists yet",
    KernelClass.PIXEL_FIXED_FUNCTION: "a fixed-function tile, not the PE fabric",
}


@dataclass(frozen=True)
class Fabric:
    """The PE fabric that runs one format, as the SKU states it."""

    sku: str
    precision: str
    rows: int
    cols: int
    tiles: int
    fill_cycles: int
    drain_cycles: int
    clock_hz: float
    ops_per_clock: float          # over the tiles that run this format
    dram_bytes_per_s: float
    #: Bits per clock the tile interconnect delivers to one array, when the
    #: tile class states it; None when it does not.
    operand_bits_per_clock: Optional[int] = None

    @property
    def macs_per_clock(self) -> float:
        return self.ops_per_clock / OPS_PER_MAC

    @property
    def peak_ops_per_s(self) -> float:
        return self.ops_per_clock * self.clock_hz


@dataclass(frozen=True)
class Ceiling:
    """The most a fabric could do on one kernel, and what holds it there."""

    kernel_class: KernelClass
    kernel: str
    precision: str
    sku: str
    value: Optional[float]
    binding: str                  # "wavefront", "dram", "operand delivery" or "gap"
    bounds: Dict[str, Optional[float]]
    source: str
    gaps: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kernel_class": self.kernel_class.value,
            "kernel": self.kernel,
            "precision": self.precision,
            "sku": self.sku,
            "ceiling": self.value,
            "binding": self.binding,
            "bounds": dict(self.bounds),
            "source": self.source,
            "gaps": list(self.gaps),
        }


def _wavefront_ceiling(shape: Shape, fabric: Fabric) -> float:
    """Passes over the array, with fill and drain, spread over the tiles."""
    cycles = 0.0
    for m, n, k in shape.gemms:
        passes = math.ceil(m / fabric.rows) * math.ceil(n / fabric.cols)
        per_tile = math.ceil(passes / fabric.tiles)
        cycles += per_tile * (k + fabric.fill_cycles + fabric.drain_cycles)
    if cycles <= 0:
        return 0.0
    issued = fabric.tiles * fabric.rows * fabric.cols * shape.ops_per_pe_fraction
    return shape.macs / (cycles * issued)


def _dram_ceiling(shape: Shape, fabric: Fabric) -> float:
    """Compute time at peak over the time the compulsory traffic takes."""
    bytes_moved = shape.compulsory_elements * BYTES[fabric.precision]
    if bytes_moved <= 0 or fabric.dram_bytes_per_s <= 0:
        return 1.0
    t_dram = bytes_moved / fabric.dram_bytes_per_s
    t_compute = shape.ops_per_call / fabric.peak_ops_per_s
    return min(1.0, t_compute / t_dram)


def _operand_ceiling(shape: Shape, fabric: Fabric) -> Optional[float]:
    """MACs per clock the stated interconnect can feed, over the array's.

    An output-stationary pass consumes one operand per row and one per
    column per clock; the interconnect delivers ``operand_bits_per_clock``
    to each of them.
    """
    if fabric.operand_bits_per_clock is None:
        return None
    per_operand_bits = 8 * BYTES[fabric.precision]
    per_clock = fabric.operand_bits_per_clock / per_operand_bits  # per row, and per column
    needed = 1.0  # one A down each row, one B across each column, per clock
    return min(1.0, per_clock / needed)


def ceilings(fabric: Fabric, shapes: Tuple[Shape, ...] = SHAPES) -> Tuple[Ceiling, ...]:
    """Every ceiling this fabric's format has a shape for."""
    out: List[Ceiling] = []
    for shape in shapes:
        wavefront = _wavefront_ceiling(shape, fabric)
        dram = _dram_ceiling(shape, fabric)
        operand = _operand_ceiling(shape, fabric)
        bounds: Dict[str, Optional[float]] = {
            "wavefront": wavefront, "dram_compulsory": dram, "operand_delivery": operand}
        known = {k: v for k, v in bounds.items() if v is not None}
        binding = min(known, key=known.__getitem__)
        gaps = () if operand is not None else (
            f"{fabric.sku}: the tile class states no fabric interconnect, so operand "
            "delivery is not bounded here",)
        out.append(Ceiling(
            kernel_class=shape.kernel_class, kernel=shape.name, precision=fabric.precision,
            sku=fabric.sku, value=known[binding], binding=binding.replace("_", " "),
            bounds=bounds, gaps=gaps,
            source=(f"{shape.source}; {fabric.rows}x{fabric.cols} PEs x {fabric.tiles} tile(s), "
                    f"fill {fabric.fill_cycles} drain {fabric.drain_cycles} cycles at "
                    f"{fabric.clock_hz / 1e6:.0f} MHz, DRAM "
                    f"{fabric.dram_bytes_per_s / 1e9:.0f} GB/s")))
    return tuple(out)


@dataclass(frozen=True)
class FabricCeilings:
    """Every ceiling of one SKU, with the classes it states no schedule for."""

    sku: str
    fabrics: Tuple[Fabric, ...]
    entries: Tuple[Ceiling, ...]
    no_schedule: Dict[str, str]
    estimation_confidence: EstimationConfidence

    def best(self, kernel_class: KernelClass, precision: str) -> Optional[Ceiling]:
        """The highest ceiling stated for a pair: the bound that must hold."""
        matches = [c for c in self.entries
                   if c.kernel_class == kernel_class and c.precision == precision
                   and c.value is not None]
        return max(matches, key=lambda c: c.value) if matches else None

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "confidence": self.estimation_confidence.level.value,
            "confidence_source": self.estimation_confidence.source,
            "fabrics": [{"precision": f.precision, "rows": f.rows, "cols": f.cols,
                         "tiles": f.tiles, "clock_mhz": f.clock_hz / 1e6,
                         "peak_tops": f.peak_ops_per_s / 1e12,
                         "operand_bits_per_clock": f.operand_bits_per_clock}
                        for f in self.fabrics],
            "ceilings": [c.to_dict() for c in self.entries],
            "no_schedule": dict(self.no_schedule),
        }


def fabrics_of(spec, precisions: Tuple[str, ...] = ("int8", "fp16", "fp32")) -> Tuple[Fabric, ...]:
    """One fabric per format, over the PE-fabric tiles that run it.

    Reads the SKU: tile geometry, fill and drain, boost clock and DRAM
    bandwidth. Tiles of different geometry at one format are not merged --
    the schedule is per array -- so the largest homogeneous group wins and
    the rest are left out of the count, which keeps the bound sound.
    """
    out: List[Fabric] = []
    dram = getattr(spec.kpu_architecture, "memory", None)
    dram_bytes = (dram.memory_bandwidth_gbps * 1e9) if dram is not None else 0.0
    clock_hz = spec.clocks.boost_clock_mhz * 1e6
    for precision in precisions:
        groups: Dict[Tuple[int, int, int, int], List] = {}
        for tile in spec.kpu_architecture.tiles:
            per_clock = (tile.ops_per_tile_per_clock or {}).get(precision)
            rows = getattr(tile, "pe_array_rows", None)
            cols = getattr(tile, "pe_array_cols", None)
            if not per_clock or not rows or not cols:
                continue
            key = (rows, cols, tile.pipeline_fill_cycles or 0, tile.pipeline_drain_cycles or 0)
            groups.setdefault(key, []).append(tile)
        if not groups:
            continue
        key = max(groups, key=lambda k: sum(t.num_tiles for t in groups[k]))
        rows, cols, fill, drain = key
        tiles = groups[key]
        interconnect = next((t.interconnect for t in tiles if getattr(t, "interconnect", None)),
                            None)
        out.append(Fabric(
            sku=spec.id, precision=precision, rows=rows, cols=cols,
            tiles=sum(t.num_tiles for t in tiles), fill_cycles=fill, drain_cycles=drain,
            clock_hz=clock_hz,
            ops_per_clock=sum(t.num_tiles * t.ops_per_tile_per_clock[precision] for t in tiles),
            dram_bytes_per_s=dram_bytes,
            operand_bits_per_clock=None if interconnect is None else interconnect.link_bits))
    return tuple(out)


def fabric_ceilings(spec, shapes: Tuple[Shape, ...] = SHAPES) -> FabricCeilings:
    """Every ceiling of one KPU SKU, over the formats its fabric runs."""
    fabrics = fabrics_of(spec)
    entries: List[Ceiling] = []
    for fabric in fabrics:
        entries += list(ceilings(fabric, shapes))
    source = ("domain-flow wavefront and compulsory-traffic bounds over the SKU's stated "
              "geometry, clock and DRAM bandwidth; shapes from graphs.benchmarks.soc_kernels")
    return FabricCeilings(
        sku=spec.id, fabrics=fabrics, entries=tuple(entries),
        no_schedule={k.value: v for k, v in NO_SCHEDULE.items()},
        estimation_confidence=estimation_confidence(Confidence.THEORETICAL, source))


def against_requirement(ceiling: Optional[Ceiling], required: Optional[float]) -> Optional[bool]:
    """Whether the requirement is *reachable*: True when the ceiling leaves
    room, False when the ceiling is below it -- a decided no -- and None
    when either side has no figure."""
    if ceiling is None or ceiling.value is None or required is None:
        return None
    return ceiling.value >= required


__all__ = ["BYTES", "Ceiling", "Fabric", "FabricCeilings", "NO_SCHEDULE", "OPS_PER_MAC",
           "SHAPES", "Shape", "against_requirement", "ceilings", "fabric_ceilings", "fabrics_of"]
