"""The KPU tile-kind energy ladder (graphs#268 E2).

One function, priced on every implementation that could run it: a
fixed-function core, a systolic array, a PE fabric, and a GPU or CPU as
reference rungs. The question it answers is the one the heterogeneous
refactor exists to ask -- *what does specialization actually buy?* -- and
the answer is a ratio, not an absolute, so each rung carries its
provenance and confidence.

Two things make the comparison honest:

**A stated ops model.** A fixed-function core's energy is published per
pixel or per frame; a programmable tile's is published per op. Comparing
them needs a number for how much arithmetic one work unit costs, and that
number is a *workload* claim, not a silicon one. Each ``OpsModel`` below
shows its arithmetic and says where it came from -- derived from the same
paper that gives the core's energy where possible, a structural estimate
otherwise.

**A back-check.** Dividing a core's published energy by that ops model
gives its implied energy per op. When that lands far above what the
process node charges for an arithmetic op, the core is not
arithmetic-bound and the ops framing is weak for it -- which is worth
seeing rather than hiding. ``Rung.implied_pj_per_op`` and
``Ladder.back_check`` exist for that.

Data movement is the other half of the story. A fixed-function core in a
stream-linked chain never writes its output to DRAM, and for image
pipelines that traffic dominates: Darkroom measures 224 pJ/pixel of
compute against 1360 pJ/pixel of DRAM for the same pipeline.
``dram_pj_per_unit`` prices what the encapsulation avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from embodied_schemas import ComputeProduct
from embodied_schemas.datapath import AbsoluteEnergy, RelativeEnergy
from embodied_schemas.kpu import FixedFunctionTile, KPUTileSpec, SystolicTile
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.hardware import kpu_tile_display as display
from graphs.hardware.kpu_access import kpu_block_of
from graphs.hardware.kpu_power_model import fixed_function_pj_per_unit
from graphs.hardware.sku_validators.silicon_math import carried_silicon

#: Workload operand formats, mapped to the name of the ``Precision`` the
#: GPU and CPU resource models scale their energy by. A format with no
#: entry, or one the device does not model, is priced at fp32 and the rung
#: says so.
_PRECISION_NAME_BY_FORMAT = {
    "int4": "INT4", "int8": "INT8", "uint8": "INT8", "int16": "INT16",
    "fp16": "FP16", "bf16": "BF16", "fp32": "FP32", "fp64": "FP64",
}

#: LPDDR5 DRAM energy, matching the ``energy_per_byte`` the KPU resource
#: models carry (1e-11 J). Quoted per byte so the saving is comparable
#: with the compute column. For contrast, Darkroom measures 1360 pJ/pixel
#: of DRAM against 224 pJ/pixel of compute for an ISP pipeline -- an
#: older node and a full round trip per stage, but the same point: for an
#: image pipeline, moving the data can cost more than computing it.
DRAM_PJ_PER_BYTE = 10.0

#: An op is one arithmetic operation in the D3 convention: a MAC or FMA is
#: two, a min-plus is two, an abs-diff is two, a lerp is three.
OPS_PER_MAC = 2.0

#: Ops per invocation by op, for a functional unit that does not state it
#: (the D3 convention). A lerp is three; everything else here is two.
_DEFAULT_OPS_PER_INVOCATION = {
    "mac": 2.0, "fma": 2.0, "min_plus": 2.0, "abs_diff": 2.0, "lerp": 3.0,
    "cmp_select": 1.0, "add": 1.0, "mul": 1.0,
}


#: A fixed-function core implying more than this many node arithmetic ops
#: per op is not arithmetic-bound (see ``_back_check``).
_BACK_CHECK_RATIO = 4.0


class LadderError(ValueError):
    """The ladder cannot be built for this function or SKU."""


# ---------------------------------------------------------------------------
# Ops models: how much arithmetic one work unit costs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpsModel:
    """Arithmetic per work unit of a function, and where the figure is from.

    ``confidence`` follows the repo's ladder: INTERPOLATED when derived
    from published figures for this exact function, THEORETICAL when it is
    a structural op count.
    """

    ops_per_unit: float
    work_unit: str
    #: Operand formats this work can be done in, best first. A class is
    #: priced on the first one it declares.
    operand_formats: Tuple[str, ...]
    #: Ops a datapath must declare to be a candidate at all. Matching on
    #: format alone would price stereo path aggregation on a MAC array and
    #: a colour matrix on a min-plus fabric, neither of which can run it.
    required_ops: Tuple[str, ...]
    #: The kernel a systolic class must list in ``supported_kernels`` to be
    #: a candidate. None means no systolic rung: a weight-stationary array
    #: runs the kernels it was built for, and an image pipeline is not one
    #: of them however well its MACs price.
    systolic_kernel: Optional[str]
    derivation: str
    confidence: str
    citation: str


#: The functions the ladder knows how to price, keyed by ``function_id``
#: (the id a ``FunctionCore`` declares) or by a synthetic name for the
#: functions no fixed-function core in the library implements.
OPS_MODELS: Dict[str, OpsModel] = {
    "stereo.sgm": OpsModel(
        # The paper gives both an energy per pixel and an efficiency, and
        # their product is its own ops count -- no outside assumption.
        #   2.3 TOPS/W x 13440 pJ/pixel = 3.09e4 ops/pixel
        ops_per_unit=2.3e12 * 13440e-12,
        work_unit="pixel",
        operand_formats=("int16", "uint8"),
        required_ops=("min_plus", "abs_diff"),
        systolic_kernel=None,
        derivation=(
            "2.3 TOPS/W x 13440 pJ/pixel = 3.09e4 ops/pixel, both figures "
            "from the same paper, so the ops count is the paper's own"
        ),
        confidence="INTERPOLATED",
        citation=(
            "Z. Li et al., 'A 1920x1080 30fps 2.3TOPS/W Stereo-Depth "
            "Processor', ISSCC 2017, paper 3.7"
        ),
    ),
    "isp.raw_to_yuv": OpsModel(
        # Structural: the stages a raw-to-YUV pipeline runs per pixel.
        ops_per_unit=24 + 4 + 18 + 1 + 18 + 18,
        work_unit="pixel",
        operand_formats=("bf16", "fp16", "int8"),
        required_ops=("mac", "fma", "lerp"),
        systolic_kernel=None,
        derivation=(
            "bilinear demosaic 24 + black level / white balance 4 + 3x3 "
            "colour matrix 18 + gamma LUT 1 + 3x3 RGB-to-YUV 18 + 3x3 "
            "denoise 18 = 83 ops/pixel (a 3x3 matrix is 9 MACs = 18 ops)"
        ),
        confidence="THEORETICAL",
        citation=(
            "Structural, from the pipeline stages in J. Hegarty et al., "
            "'Darkroom', ACM TOG 33(4) 2014, Figs. 11-12"
        ),
    ),
    "vio.stereo_inertial": OpsModel(
        # Structural, and the weakest of the four -- see back_check().
        ops_per_unit=752 * 480 * 20 + 200 * 64 * 2 + 200 * 50 * 20 + 10e6,
        work_unit="frame",
        operand_formats=("fp32", "bf16", "fp16"),
        required_ops=("mac", "fma"),
        systolic_kernel=None,
        derivation=(
            "752x480 frontend feature detect / track at 20 ops/pixel "
            "= 7.2 Mops, + 200 features x 64 disparities x 2 = 25.6 kops "
            "stereo, + 200 x 50 RANSAC iterations x 20 = 200 kops, + ~10 "
            "Mops sparse factor-graph backend = 1.7e7 ops/frame"
        ),
        confidence="THEORETICAL",
        citation=(
            "Structural, from the pipeline in A. Suleiman et al., 'Navion', "
            "IEEE JSSC 54(4) 2019, Fig. 10"
        ),
    ),
    "gemm.int8": OpsModel(
        ops_per_unit=OPS_PER_MAC,
        work_unit="MAC",
        operand_formats=("int8",),
        required_ops=("mac", "fma"),
        systolic_kernel="gemm",
        derivation="one INT8 multiply-accumulate = 2 ops (the D3 convention)",
        confidence="THEORETICAL",
        citation="graphs#268 D3 ops-counting convention",
    ),
}


# ---------------------------------------------------------------------------
# Pricing one implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One implementation of the function, priced."""

    label: str
    kind: str  # fixed_function | systolic | pe_fabric | gpu | cpu
    pj_per_unit: float
    pj_per_op: Optional[float]  # None for a core priced per unit only
    area_mm2: Optional[float]
    confidence: str
    provenance: str
    notes: str = ""

    @property
    def implied_pj_per_op(self) -> Optional[float]:
        """What this rung's per-unit energy implies per op, under the
        ladder's ops model. For a programmable rung this is just
        ``pj_per_op``; for a fixed-function core it is the back-check."""
        return self.pj_per_op


def _op_energy_pj(
    tile, node: ProcessNodeEntry, ops: OpsModel
) -> Optional[Tuple[float, str, str]]:
    """``(pj_per_op, unit_id, operand_format)`` for the cheapest way this
    tile can do ``ops``'s work, or None when it cannot.

    A class must declare one of the required ops *and* one of the accepted
    formats. Matching on format alone would price stereo path aggregation
    on a MAC array, or a colour matrix on a min-plus fabric -- neither can
    run the work, and the resulting rung would be fiction.

    Formats are tried in the order the ops model prefers them, and within
    the first format the tile supports, the cheapest unit and mode wins.
    Letting the unit loop run outermost instead would pick a less-preferred
    format, or the first mode rather than the best one, purely from
    declaration order (CodeRabbit on #289).

    Generalizes the loader's MAC-only lookup, which is the point: a
    min-plus class declares no MAC, and pricing stereo on it is why the
    class exists.
    """
    if isinstance(tile, SystolicTile):
        if ops.systolic_kernel not in (tile.supported_kernels or []):
            return None
        units = [tile.mac]
    elif isinstance(tile, KPUTileSpec) and tile.datapath is not None:
        units = list(tile.datapath.functional_units)
    else:
        return None

    for operand_format in ops.operand_formats:
        best: Optional[Tuple[float, str, str]] = None
        for unit in units:
            if unit.op.value not in ops.required_ops:
                continue
            for mode in unit.modes:
                if mode.operand_format != operand_format:
                    continue
                energy = mode.energy
                per_invocation: Optional[float] = None
                if isinstance(energy, RelativeEnergy):
                    anchor = node.energy_per_op_pj.get(energy.anchor)
                    if anchor is not None:
                        per_invocation = energy.ratio * anchor * OPS_PER_MAC
                elif isinstance(energy, AbsoluteEnergy):
                    per_invocation = energy.pj
                if per_invocation is None:
                    continue
                # ops_per_invocation is optional; a unit that does not
                # state it follows the D3 default for its op.
                per_op = unit.ops_per_invocation or _DEFAULT_OPS_PER_INVOCATION.get(
                    unit.op.value, OPS_PER_MAC
                )
                priced = (
                    per_invocation / max(1.0, float(per_op)),
                    unit.op.value,
                    operand_format,
                )
                if best is None or priced[0] < best[0]:
                    best = priced
        if best is not None:
            return best
    return None


def _class_area_mm2(
    cp: ComputeProduct, node: ProcessNodeEntry, tile_class_id: str
) -> Optional[float]:
    """Silicon one tile of a class occupies, from its carried silicon."""
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
        t for t in kpu_block_of(cp).tiles if t.tile_class_id == tile_class_id
    )
    return total / tile.num_tiles if tile.num_tiles else None


def _fixed_function_rung(
    cp: ComputeProduct, node: ProcessNodeEntry, tile, ops: OpsModel, nodes
) -> Optional[Rung]:
    core = tile.core
    pj = fixed_function_pj_per_unit(core, node, nodes)
    if pj is None:
        return None
    area = _class_area_mm2(cp, node, tile.tile_class_id)
    return Rung(
        label=f"{tile.tile_class_id} (fixed-function core)",
        kind="fixed_function",
        pj_per_unit=pj,
        pj_per_op=pj / ops.ops_per_unit if ops.ops_per_unit else None,
        area_mm2=area,
        confidence=core.confidence.value.upper(),
        provenance=core.energy.source or core.source,
        notes=(
            f"published {core.energy.pj_per_unit:g} pJ/{ops.work_unit} at "
            f"{core.energy.ref_node_id}, retargeted to {node.node_name}"
        ),
    )


def _programmable_rung(
    cp: ComputeProduct, node: ProcessNodeEntry, tile, ops: OpsModel
) -> Optional[Rung]:
    priced = _op_energy_pj(tile, node, ops)
    if priced is None:
        return None
    pj_per_op, op_name, operand_format = priced
    return Rung(
        label=f"{tile.tile_class_id} ({display.tile_kind(tile).replace('_', ' ')})",
        kind=display.tile_kind(tile),
        pj_per_unit=pj_per_op * ops.ops_per_unit,
        pj_per_op=pj_per_op,
        area_mm2=_class_area_mm2(cp, node, tile.tile_class_id),
        confidence="THEORETICAL",
        provenance=(
            f"tile-class datapath energy for {op_name} on {operand_format} "
            f"at {node.node_name}, x the ladder's ops model"
        ),
        notes="arithmetic only: no control, staging or data movement",
    )


def _reference_rung(mapper_name: str, kind: str, ops: OpsModel) -> Optional[Rung]:
    """A GPU or CPU rung, priced from its resource model's energy per op."""
    from graphs.hardware.mappers import get_mapper_by_name
    from graphs.hardware.resource_model import Precision

    mapper = get_mapper_by_name(mapper_name)
    if mapper is None:
        return None
    rm = mapper.resource_model
    # Price at the format the workload actually prefers. Defaulting
    # everything non-int8 to FP16 mispriced the fp32 and int16 workloads by
    # a factor of two (CodeRabbit on #289).
    precision = None
    for operand_format in ops.operand_formats:
        name = _PRECISION_NAME_BY_FORMAT.get(operand_format)
        candidate = getattr(Precision, name, None) if name else None
        if candidate is not None and candidate in rm.energy_scaling:
            precision = candidate
            break
    fallback = ""
    if precision is None:
        # A format this mapper does not model (LNS, say). FP32 is the
        # unscaled reference, and the rung says it is standing in.
        precision = Precision.FP32
        fallback = (
            f"; {'/'.join(ops.operand_formats)} is not in this device's "
            f"energy model, priced at fp32"
        )
    scale = rm.energy_scaling.get(precision, 1.0)
    # energy_per_flop_fp32 is per op already (a FLOP is one op).
    pj_per_op = rm.energy_per_flop_fp32 * scale * 1e12
    return Rung(
        label=f"{rm.name} ({kind})",
        kind=kind,
        pj_per_unit=pj_per_op * ops.ops_per_unit,
        pj_per_op=pj_per_op,
        area_mm2=None,
        confidence="CALIBRATED",
        provenance=(
            f"{rm.name} resource model: energy_per_flop_fp32 x "
            f"energy_scaling[{precision.value}]"
        ),
        notes="whole-device figure; not a per-tile comparison" + fallback,
    )


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ladder:
    function_id: str
    sku_id: str
    node_id: str
    ops: OpsModel
    rungs: Tuple[Rung, ...]
    dram_bytes_per_unit: Optional[float]
    dram_pj_per_unit: Optional[float]
    dram_note: str = ""
    #: Where the ops model is carrying more weight than it can bear; see
    #: ``arithmetic_bound``. Computed at build time by ``_back_check``.
    back_check_notes: Tuple[str, ...] = ()

    @property
    def kind_order(self) -> Tuple[str, ...]:
        """The tile kinds, cheapest first."""
        return tuple(r.kind for r in self.rungs)

    @property
    def follows_specialization_order(self) -> bool:
        """Whether cheaper-per-unit really does mean more specialized.

        The claim the ladder is built to test: on this die, fixed function
        beats systolic beats PE fabric. It is a *finding*, not an invariant
        -- the rungs are sorted by energy, so asking whether that sort is
        ascending would prove nothing. This compares the resulting order
        against the expected one, and a False here is a result worth
        reading, not a bug.

        The GPU and CPU rungs are excluded: they are whole devices on
        another die, here as reference points, and where they land among
        the KPU's kinds is a separate question from whether specialization
        pays off within one chip.
        """
        rank = {"fixed_function": 0, "systolic": 1, "pe_fabric": 2}
        ranks = [rank[k] for k in self.kind_order if k in rank]
        return all(a <= b for a, b in zip(ranks, ranks[1:]))

    @property
    def arithmetic_bound(self) -> bool:
        """Whether the fixed-function rung's energy is mostly arithmetic.

        When it is not, the programmable rungs -- which price arithmetic
        and nothing else -- are a lower bound rather than a like-for-like
        comparison, and the ladder should be read as energy per work unit
        only. ``back_check`` says so in words.
        """
        return not self.back_check_notes

    def ratio_to_best(self, rung: Rung) -> float:
        best = min(r.pj_per_unit for r in self.rungs)
        return rung.pj_per_unit / best if best > 0 else float("inf")


def _back_check(rungs: Sequence[Rung], ops: OpsModel,
                node: ProcessNodeEntry) -> Tuple[str, ...]:
    """Where the ops model is carrying more weight than it can bear.

    A fixed-function core whose implied energy per op sits far above what
    the node charges for an arithmetic op is not arithmetic-bound: its
    energy is memory and control. Reading its ladder position as "N times
    better arithmetic" would then be wrong, and the programmable rungs --
    which price arithmetic and nothing else -- are a lower bound rather
    than a like-for-like comparison.
    """
    anchor = node.energy_per_op_pj.get("balanced_logic:int8")
    if anchor is None:
        return ()
    out = []
    for rung in rungs:
        if rung.kind != "fixed_function" or rung.pj_per_op is None:
            continue
        ratio = rung.pj_per_op / anchor
        if ratio > _BACK_CHECK_RATIO:
            out.append(
                f"{rung.label}: implied {rung.pj_per_op:.3g} pJ/op is {ratio:.0f}x "
                f"the node's {anchor:g} pJ arithmetic op, so this core is bound by "
                f"memory and control, not arithmetic. Read the ladder as energy "
                f"per {ops.work_unit}; the programmable rungs price arithmetic "
                f"only and are a lower bound, not a like-for-like comparison."
            )
    return tuple(out)


def _cores_for(block, function_id: str) -> List:
    """Every fixed-function tile class implementing ``function_id``.

    A die may carry two cores for one function -- a low-power one and a
    high-throughput one, say -- and the ladder promises to price every
    implementation, so returning the first would quietly drop the rest
    (CodeRabbit on #289).
    """
    return [
        tile for tile in block.tiles
        if isinstance(tile, FixedFunctionTile)
        and tile.core.function_id == function_id
    ]


def build_ladder(
    cp: ComputeProduct,
    node: ProcessNodeEntry,
    function_id: str,
    nodes: Optional[Mapping[str, ProcessNodeEntry]] = None,
    references: Sequence[Tuple[str, str]] = (
        ("Jetson-Orin-AGX-64GB", "gpu"),
        ("Jetson-Orin-AGX-CPU", "cpu"),
    ),
) -> Ladder:
    """Price ``function_id`` on every implementation ``cp`` offers, plus the
    reference GPU and CPU rungs, cheapest first."""
    ops = OPS_MODELS.get(function_id)
    if ops is None:
        raise LadderError(
            f"no ops model for {function_id!r}. Available: {sorted(OPS_MODELS)}"
        )
    block = kpu_block_of(cp)
    rungs: List[Rung] = []

    core_tiles = _cores_for(block, function_id)
    for core_tile in core_tiles:
        rung = _fixed_function_rung(cp, node, core_tile, ops, nodes)
        if rung is not None:
            rungs.append(rung)

    for tile in block.tiles:
        if not display.is_programmable(tile):
            continue
        rung = _programmable_rung(cp, node, tile, ops)
        if rung is not None:
            rungs.append(rung)

    # The reference rungs are context, not evidence that this SKU runs the
    # function. Without this check a KPU with no implementation at all
    # would still report the function as supported, on the strength of a
    # GPU number (CodeRabbit on #289).
    if not rungs:
        raise LadderError(
            f"no implementation of {function_id!r} could be priced on "
            f"{cp.id!r} at {node.node_name}"
        )

    for name, kind in references:
        rung = _reference_rung(name, kind, ops)
        if rung is not None:
            rungs.append(rung)

    dram_bytes, dram_note = _dram_traffic(
        core_tiles[0] if core_tiles else None, block, ops
    )
    ordered = tuple(sorted(rungs, key=lambda r: r.pj_per_unit))
    return Ladder(
        function_id=function_id,
        sku_id=cp.id,
        node_id=node.node_name,
        ops=ops,
        rungs=ordered,
        dram_bytes_per_unit=dram_bytes,
        dram_pj_per_unit=(
            dram_bytes * DRAM_PJ_PER_BYTE if dram_bytes is not None else None
        ),
        dram_note=dram_note,
        back_check_notes=_back_check(ordered, ops, node),
    )


def _dram_traffic(core_tile, block, ops: OpsModel) -> Tuple[Optional[float], str]:
    """DRAM energy a stream link avoids for one work unit.

    A core whose output feeds the next stage over a stream link never
    writes that result to DRAM, and the next stage never reads it back --
    two crossings saved per unit. Priced at the KPU's own DRAM energy.
    """
    if core_tile is None or core_tile.core.io is None:
        return None, ""
    from embodied_schemas.overlay import NoCOverlayKind

    linked = set()
    for overlay in block.noc.overlays or []:
        if overlay.kind == NoCOverlayKind.STREAM_LINK:
            linked.update(overlay.endpoints)
    if core_tile.tile_class_id not in linked:
        return None, "not stream-linked; its output goes to DRAM"
    out_bytes = core_tile.core.io.output_bytes_per_unit
    avoided = out_bytes * 2  # the write, and the next stage's read back
    return avoided, (
        f"{out_bytes:g} B/{ops.work_unit} not written and not read back = "
        f"{avoided:g} B = {avoided * DRAM_PJ_PER_BYTE:g} pJ of DRAM traffic "
        f"avoided per {ops.work_unit}"
    )
