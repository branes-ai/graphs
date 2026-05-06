"""Regime classifier for the model validation harness (v4).

For each (op, shape, dtype, hardware) tuple, classify which architectural
layer's spec a benchmark of that shape is *supposed* to validate. This is
Principle 1 of docs/plans/validation-harness-v4-plan.md: every sweep shape
lands in exactly one known roofline regime, so a per-shape failure points
at one architectural assumption.

Two-tier dispatch
-----------------
The classifier tries the bandwidth-aware path first; when the hardware
mapper exposes ``l1_bandwidth_per_unit_bps`` and ``l2_bandwidth_bps``
(populated for the V4 reference targets per issue #61), low-OI shapes
that fit in L1 land in ``L1_BOUND`` instead of being collapsed into
``L2_BOUND``. When either field is None, the classifier falls back to
the v4-1 capacity-only behavior unchanged.

Regimes:

* ``alu_bound``     - working set fits in L1 AND OI > peak_FLOPS / peak_DRAM_BW
                      (an ALU-bound op with everything in L1 is the cleanest
                      validation of peak FLOPS)
* ``l1_bound``      - working set fits in L1 AND OI < peak_FLOPS / peak_L1_BW
                      (validates per-unit L1 bandwidth). Only emitted when
                      l1_bandwidth_per_unit_bps is populated.
* ``l2_bound``      - working set in (L1, L2_or_LLC]; validates shared L2 BW
                      when l2_bandwidth_bps is populated, otherwise validates
                      L2 capacity only.
* ``dram_bound``    - working set > on-chip total
* ``launch_bound``  - predicted compute time < 5x kernel-launch overhead
* ``ambiguous``     - within +/-AMBIGUOUS_BAND of any boundary; rejected
                      from validation sweeps so failures attribute to one
                      layer
* ``unsupported``   - (hardware, dtype) combo has no peak FLOPS entry
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from graphs.hardware.resource_model import HardwareResourceModel, Precision


class Regime(Enum):
    """Roofline regime a shape is supposed to validate."""
    ALU_BOUND = "alu_bound"
    L1_BOUND = "l1_bound"         # On-chip per-unit L1 BW bound. Only emitted
                                  # when l1_bandwidth_per_unit_bps is populated;
                                  # see issue #61 for the bandwidth-aware
                                  # dispatch path.
    L2_BOUND = "l2_bound"
    DRAM_BOUND = "dram_bound"
    LAUNCH_BOUND = "launch_bound"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"   # (hardware, dtype) combo not modeled --
                                  # e.g., fp16 on Alder Lake P-cores. Sweep
                                  # entries returning UNSUPPORTED are dropped
                                  # from validation but reported separately.


# Reject shapes whose working-set or compute-time falls within +/-20% of a
# regime boundary. Tighter bands shrink the validation sweep but produce
# more attributable failures; wider bands cover more shapes but blur which
# layer drifted.
AMBIGUOUS_BAND = 0.20

# Default kernel-launch overhead in seconds. Should be calibrated per
# hardware later (e.g., GPUs ~5us, CPUs ~1us, accelerators vary). Used to
# detect ``launch_bound`` shapes: if predicted compute_time is within 5x of
# the launch overhead, the shape is launch-dominated and unsuitable for
# validating ALU/cache/DRAM peaks.
DEFAULT_LAUNCH_OVERHEAD_S = 5e-6
LAUNCH_BOUND_MULTIPLIER = 5.0


_PRECISION_BYTES: dict[str, float] = {
    "fp64": 8, "fp32": 4, "tf32": 4,
    "fp16": 2, "bf16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1,
    "int4": 0.5, "fp4": 0.5,
}


def bytes_per_element(dtype: str) -> float:
    """Return bytes per element for a dtype string, mirroring
    ``graphs.hardware.resource_model.precision_bytes_per_element`` but
    accepting plain strings so JSON sweep files can specify dtypes
    without importing the Precision enum."""
    key = dtype.lower()
    if key not in _PRECISION_BYTES:
        raise ValueError(
            f"Unknown dtype {dtype!r}; supported: {sorted(_PRECISION_BYTES)}"
        )
    return _PRECISION_BYTES[key]


# ---------------------------------------------------------------------------
# Per-op working set and FLOP counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpFootprint:
    """Working set (bytes), FLOPs, and operational intensity for an op."""
    working_set_bytes: int
    flops: int
    operational_intensity: float  # flops / working_set_bytes


def _matmul_footprint(shape: Tuple[int, int, int], dtype: str) -> OpFootprint:
    M, K, N = shape
    bpe = bytes_per_element(dtype)
    working_set = int(round((M * K + K * N + M * N) * bpe))
    flops = 2 * M * K * N
    oi = flops / working_set if working_set > 0 else 0.0
    return OpFootprint(working_set_bytes=working_set, flops=flops,
                       operational_intensity=oi)


def _linear_footprint(shape: Tuple[int, int, int], dtype: str) -> OpFootprint:
    # nn.Linear(IN, OUT) on a [B, IN] input -> [B, OUT] output.
    B, IN, OUT = shape
    bpe = bytes_per_element(dtype)
    working_set = int(round((B * IN + IN * OUT + B * OUT) * bpe))
    flops = 2 * B * IN * OUT  # bias is negligible
    oi = flops / working_set if working_set > 0 else 0.0
    return OpFootprint(working_set_bytes=working_set, flops=flops,
                       operational_intensity=oi)


def _vector_add_footprint(shape: Tuple[int, ...], dtype: str) -> OpFootprint:
    # Vector add: c[i] = a[i] + b[i]. Two N-element inputs + one
    # N-element output, no operand reuse. The zero-reuse ground truth
    # for tier-bandwidth measurement (V5 plan).
    (N,) = shape
    bpe = bytes_per_element(dtype)
    working_set = int(round(3 * N * bpe))  # a + b + c
    flops = N                              # one add per element
    oi = flops / working_set if working_set > 0 else 0.0
    return OpFootprint(working_set_bytes=working_set, flops=flops,
                       operational_intensity=oi)


_OP_FOOTPRINT = {
    "matmul": _matmul_footprint,
    "linear": _linear_footprint,
    "vector_add": _vector_add_footprint,
}


def op_footprint(op: str, shape: Tuple[int, ...], dtype: str) -> OpFootprint:
    """Compute working set, FLOPs, and OI for a supported op."""
    if op not in _OP_FOOTPRINT:
        raise ValueError(f"Unsupported op {op!r}; supported: {sorted(_OP_FOOTPRINT)}")
    return _OP_FOOTPRINT[op](shape, dtype)


# ---------------------------------------------------------------------------
# Hardware-derived capacity and roofline numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareCapacities:
    """Per-target memory and roofline constants used by the classifier.

    The on-chip BW peaks (#61) are Optional so existing 45+ mappers
    that don't populate them keep working through the capacity-only
    fallback path."""
    name: str
    l1_total_bytes: int          # sum of per-unit L1 across compute units
    l2_total_bytes: int          # shared L2 / LLC
    on_chip_total_bytes: int     # l1_total + l2_total
    peak_dram_bandwidth_bps: float
    peak_flops: float
    ai_breakpoint: float         # peak_flops / peak_dram_bw -- the OI above which
                                 # an op is compute-bound on the DRAM roofline

    # On-chip BW peaks (#61). None when the mapper hasn't populated the
    # corresponding HardwareResourceModel field; the classifier falls
    # back to capacity-only behavior.
    peak_l1_bandwidth_bps: float | None = None    # AGGREGATE across all units
                                                   # (= per_unit * compute_units)
    peak_l2_bandwidth_bps: float | None = None    # shared L2 / LLC bandwidth

    @property
    def l1_ai_breakpoint(self) -> float | None:
        """OI above which an L1-resident op becomes compute-bound on the
        L1 roofline. None when peak_l1_bandwidth_bps isn't populated."""
        if self.peak_l1_bandwidth_bps is None or self.peak_l1_bandwidth_bps <= 0:
            return None
        return self.peak_flops / self.peak_l1_bandwidth_bps


def hardware_capacities(
    hw: HardwareResourceModel, precision: Precision
) -> HardwareCapacities:
    """Pull the capacity and roofline numbers the classifier needs.

    The classifier asks "is this shape compute-bound or memory-bound on
    this hardware?" The honest answer requires the **effective** peak
    (what real workloads can attain after thermal / efficiency derate),
    not the raw theoretical spec. Otherwise a hardware whose theoretical
    peak doubles (e.g., #68 corrected i7-12700K from 720 -> 1440 GFLOPS)
    would silently shift sweep regime labels even though achievable
    throughput is unchanged.

    Effective peak comes from RooflineAnalyzer._get_effective_peak_ops,
    which honors thermal_operating_points -> default_thermal_profile ->
    legacy precision_profiles in priority order. Same code path the
    analyzer's roofline math uses.

    Only fields that already exist on HardwareResourceModel are used; if
    we add explicit L1/L2 bandwidth peaks later (#61), this is the one
    place that needs to grow.
    """
    # Late import: classify.py is the bottom of the v4 dependency stack;
    # graphs.estimation.roofline imports lots of things. Late binding
    # keeps the classifier importable in lightweight contexts (sweep
    # generation, unit tests with mocked analyzers).
    from graphs.estimation.roofline import RooflineAnalyzer

    # Two distinct peak queries:
    # 1. ``hw.get_peak_ops(precision)`` -- STRICT supported-precision check.
    #    Raises if the precision is not in ``precision_profiles``. This
    #    drives the UNSUPPORTED path: i7 has no native fp16, so trying to
    #    classify fp16 work on i7 raises here and the caller returns
    #    Regime.UNSUPPORTED.
    # 2. ``RooflineAnalyzer._get_effective_peak_ops`` -- effective peak after
    #    thermal / efficiency derate. This drives the AI breakpoint and the
    #    launch-bound threshold. Using effective (not theoretical) peak
    #    means a spec correction that changes only the theoretical headline
    #    (#68: 720 -> 1440 GFLOPS, with compensating efficiency_factor halve)
    #    does not silently shift sweep regime labels.
    hw.get_peak_ops(precision)  # raises -> caller -> Regime.UNSUPPORTED
    l1_total = hw.compute_units * hw.l1_cache_per_unit
    l2_total = hw.l2_cache_total or 0
    on_chip = l1_total + l2_total
    peak_flops = RooflineAnalyzer._get_effective_peak_ops(hw, precision)
    peak_bw = hw.peak_bandwidth
    ai_breakpoint = peak_flops / peak_bw if peak_bw > 0 else math.inf

    # On-chip BW peaks (#61). Aggregate L1 BW = per-unit * compute_units;
    # L2 BW is already aggregate by spec. Both Optional so unpopulated
    # mappers keep using the capacity-only path.
    peak_l1_bw: float | None = None
    if hw.l1_bandwidth_per_unit_bps is not None:
        peak_l1_bw = hw.l1_bandwidth_per_unit_bps * hw.compute_units
    peak_l2_bw = hw.l2_bandwidth_bps   # already aggregate or None

    return HardwareCapacities(
        name=hw.name,
        l1_total_bytes=l1_total,
        l2_total_bytes=l2_total,
        on_chip_total_bytes=on_chip,
        peak_dram_bandwidth_bps=peak_bw,
        peak_flops=peak_flops,
        ai_breakpoint=ai_breakpoint,
        peak_l1_bandwidth_bps=peak_l1_bw,
        peak_l2_bandwidth_bps=peak_l2_bw,
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _within_band(value: float, boundary: float, band: float = AMBIGUOUS_BAND) -> bool:
    """True if ``value`` is within +/-band of ``boundary``."""
    if boundary <= 0:
        return False
    return abs(value - boundary) / boundary < band


def classify_regime(
    op: str,
    shape: Tuple[int, ...],
    dtype: str,
    hardware: HardwareResourceModel,
    *,
    launch_overhead_s: float = DEFAULT_LAUNCH_OVERHEAD_S,
) -> Regime:
    """Classify which architectural layer this shape validates.

    Returns one of the Regime enum values. AMBIGUOUS shapes are rejected
    from validation sweeps; calibration sweeps may include them but the
    harness treats their results as advisory only.

    Pure function: no I/O, no measurement, no analyzer call. The
    classifier only reads HardwareResourceModel fields and the op
    footprint formula. This matters for unit-testing the regime math
    independently of the rest of the analyzer.
    """
    fp = op_footprint(op, shape, dtype)
    try:
        cap = hardware_capacities(hardware, _resolve_precision(dtype))
    except ValueError:
        # The hardware doesn't list this precision (post-#57: get_peak_ops
        # raises instead of silently falling back). The combo isn't
        # validatable; surface it so sweep generation can prune it.
        return Regime.UNSUPPORTED

    # Launch-bound check first: if predicted ideal compute time is small
    # relative to the launch overhead, the shape can only validate the
    # launch-overhead constant, not any roofline layer. Use peak FLOPS as
    # the optimistic compute-time floor.
    ideal_compute_time_s = fp.flops / cap.peak_flops if cap.peak_flops > 0 else math.inf
    if ideal_compute_time_s < launch_overhead_s * LAUNCH_BOUND_MULTIPLIER:
        return Regime.LAUNCH_BOUND

    # Capacity-based bucketing.
    ws = fp.working_set_bytes

    # Reject ambiguity at the L1 boundary.
    if _within_band(ws, cap.l1_total_bytes):
        return Regime.AMBIGUOUS
    # Reject ambiguity at the L2/LLC boundary.
    if _within_band(ws, cap.l2_total_bytes):
        return Regime.AMBIGUOUS

    if ws <= cap.l1_total_bytes:
        # Working set fits in L1. To be ALU_BOUND it also has to be on the
        # compute side of the DRAM roofline breakpoint.
        if _within_band(fp.operational_intensity, cap.ai_breakpoint):
            return Regime.AMBIGUOUS
        if fp.operational_intensity > cap.ai_breakpoint:
            return Regime.ALU_BOUND
        # OI < DRAM AI breakpoint and fits in L1: this is the L1-bandwidth-
        # bound regime if the hardware exposes a per-unit L1 BW peak (#61).
        # Without that peak, we can't validate L1 BW, so the shape stays
        # ambiguous (the v4-1 default).
        l1_breakpoint = cap.l1_ai_breakpoint
        if l1_breakpoint is not None:
            if _within_band(fp.operational_intensity, l1_breakpoint):
                return Regime.AMBIGUOUS
            if fp.operational_intensity < l1_breakpoint:
                return Regime.L1_BOUND
            # OI in (l1_breakpoint, ai_breakpoint): L1 BW isn't the
            # bottleneck (compute is fast enough to keep up with L1)
            # but DRAM AI breakpoint isn't reached either. Still
            # ambiguous -- this would need the L2 BW roofline to
            # resolve cleanly.
        return Regime.AMBIGUOUS

    if ws <= cap.l2_total_bytes:
        return Regime.L2_BOUND

    return Regime.DRAM_BOUND


def _resolve_precision(dtype: str) -> Precision:
    """Map a dtype string to the Precision enum used by the resource model."""
    key = dtype.lower()
    table = {
        "fp64": Precision.FP64, "fp32": Precision.FP32, "tf32": Precision.TF32,
        "fp16": Precision.FP16, "bf16": Precision.BF16,
        "fp8": Precision.FP8, "fp8_e4m3": Precision.FP8_E4M3, "fp8_e5m2": Precision.FP8_E5M2,
        "int64": Precision.INT64, "int32": Precision.INT32,
        "int16": Precision.INT16, "int8": Precision.INT8, "int4": Precision.INT4,
        "fp4": Precision.FP4,
    }
    if key not in table:
        raise ValueError(f"Unknown dtype {dtype!r} (no Precision enum mapping)")
    return table[key]
