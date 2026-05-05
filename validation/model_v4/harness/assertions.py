"""ValidationRecord + per-regime pass/fail logic for the v4 harness.

Implements Principle 3 of docs/plans/validation-harness-v4-plan.md:
per-shape failures attribute to one architectural-model assumption.
A `ValidationRecord` carries the predicted regime + latency + energy,
the measured regime + latency + energy, and a per-assertion pass flag.
The aggregator (report.py) then turns hundreds of these into a
hardware x regime heatmap so a `FAIL` cell points at one drifted
assumption.

The five regimes (per Principle 1):

* ``alu_bound``     - validates ALU peak FLOPS
* ``l2_bound``      - validates L2/LLC bandwidth (capacity-only in v4-1)
* ``dram_bound``    - validates DRAM peak bandwidth
* ``launch_bound``  - validates kernel-launch overhead constant
* ``ambiguous``     - rejected from validation; harness skips
* ``unsupported``   - (hw, dtype) combo has no peak FLOPS entry

Tolerance bands are per-regime: tighter where the underlying spec is
firm (peak FLOPS) and wider where real silicon is variable (DRAM).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

from validation.model_v4.sweeps.classify import Regime


# ---------------------------------------------------------------------------
# Per-regime tolerance bands (Principle 3 in the plan)
# ---------------------------------------------------------------------------

# Latency tolerance (relative error). Tightest for ALU_BOUND because peak
# FLOPS is a clean spec; widest for LAUNCH_BOUND because launch is
# inherently noisy.
TOL_LATENCY: dict[Regime, float] = {
    Regime.ALU_BOUND:    0.10,
    Regime.L2_BOUND:     0.20,
    Regime.DRAM_BOUND:   0.25,
    Regime.LAUNCH_BOUND: 0.30,
}

# Energy tolerance (relative error). Wider than latency because energy
# also varies with op mix and DVFS state.
TOL_ENERGY: dict[Regime, float] = {
    Regime.ALU_BOUND:    0.15,
    Regime.L2_BOUND:     0.25,
    Regime.DRAM_BOUND:   0.30,
    Regime.LAUNCH_BOUND: 0.40,
}


# Threshold used to classify a measurement as compute- vs bandwidth-bound.
# A run that achieves >= 70% of peak FLOPS is "really" compute-bound;
# ditto for DRAM bandwidth.
UTILIZATION_THRESHOLD = 0.70


# RAPL energy counter resolution on Intel client CPUs is 61 uJ; on top of
# that, the counter integrates across the *whole package* including
# unrelated background activity. Below ~1 ms of measurement, the relative
# noise on a single-trial energy reading easily reaches 50-100%, so the
# energy assertion is suppressed below this threshold (issue #71).
# NVML on NVIDIA GPUs has similar lower-bound resolution (~1 ms power
# samples), so the same threshold applies for GPU ground truth.
ENERGY_RELIABLE_LATENCY_S = 1e-3


# ---------------------------------------------------------------------------
# ValidationRecord -- one per (shape, hardware) measurement
# ---------------------------------------------------------------------------


@dataclass
class ValidationRecord:
    """A single (shape, hardware) validation outcome.

    All three pass flags must hold for the record to be a clean pass.
    A FAIL flag attributes to one specific architectural assumption:

    - ``pass_regime`` False -> the analyzer's bottleneck classification
      is wrong (it predicted compute-bound when the run was really
      memory-bound, or vice versa)
    - ``pass_latency`` False -> the analyzer's latency math is wrong
      (peak FLOPS, peak BW, or efficiency factor for this regime)
    - ``pass_energy`` False -> the analyzer's energy math is wrong
      (energy_per_flop, energy_per_byte, or static-power model)
    """

    # Identity
    hardware: str
    op: str                                # "matmul" or "linear"
    shape: tuple                           # (M, K, N) or (B, IN, OUT)
    dtype: str

    # Predicted (from the analyzer)
    regime_predicted: str                  # one of Regime.value
    latency_predicted_ms: float
    energy_predicted_j: Optional[float]    # None when energy not modelled

    # Measured (from the ground-truth measurer)
    regime_measured: str
    latency_measured_ms: float
    energy_measured_j: Optional[float]     # None when RAPL/NVML unreadable

    # Per-assertion outcomes
    pass_regime: bool
    pass_latency: bool
    pass_energy: Optional[bool]            # None when energy not measured

    # Tolerances applied (per-regime, see TOL_LATENCY / TOL_ENERGY)
    tolerance_latency: float
    tolerance_energy: float

    # Diagnostics
    bottleneck_layer: str                  # human-readable: "ALU peak FLOPS", "DRAM BW", ...
    notes: str = ""                        # free-form, e.g., "L2_BOUND inference soft (v4-1)"

    def all_pass(self) -> bool:
        """True iff every applicable assertion passed.

        Energy assertion is optional: a record where energy was not
        measured (e.g., RAPL unreadable) still counts as a pass overall
        if regime + latency pass.
        """
        if not (self.pass_regime and self.pass_latency):
            return False
        if self.pass_energy is False:
            return False
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict turns the shape tuple into a list; keep it as a list for
        # JSON-friendliness.
        return d


# ---------------------------------------------------------------------------
# regime_measured inference from raw measurements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementContext:
    """Per-(hw, dtype) constants the inference needs.

    Pulled out as a struct so the harness can build it once per target
    and reuse across many measurements.
    """
    peak_flops: float                      # ops/sec at the measurement dtype
    peak_dram_bandwidth_bps: float
    launch_overhead_s: float               # default 5e-6


def infer_regime_measured(
    flops: int,
    working_set_bytes: int,
    measured_latency_s: float,
    ctx: MeasurementContext,
) -> Regime:
    """Classify an actual measurement into a regime.

    v4-1 limitation: without explicit on-chip BW peaks (issue #61) we
    can only positively identify ALU_BOUND, DRAM_BOUND, and LAUNCH_BOUND
    from a measurement. Anything in-between gets AMBIGUOUS, which the
    asserter treats as "compatible with anything except ALU/DRAM/LAUNCH".
    """
    if measured_latency_s <= 0:
        return Regime.AMBIGUOUS
    if measured_latency_s < ctx.launch_overhead_s * 5:
        return Regime.LAUNCH_BOUND

    achieved_flops = flops / measured_latency_s
    achieved_bw = working_set_bytes / measured_latency_s
    flops_util = achieved_flops / ctx.peak_flops if ctx.peak_flops > 0 else 0.0
    bw_util = (achieved_bw / ctx.peak_dram_bandwidth_bps
               if ctx.peak_dram_bandwidth_bps > 0 else 0.0)

    if flops_util >= UTILIZATION_THRESHOLD:
        return Regime.ALU_BOUND
    if bw_util >= UTILIZATION_THRESHOLD:
        return Regime.DRAM_BOUND
    # Below both thresholds = somewhere in the middle. Without on-chip
    # BW peaks we can't disambiguate L2_BOUND from "just inefficient".
    return Regime.AMBIGUOUS


# ---------------------------------------------------------------------------
# Per-assertion checks
# ---------------------------------------------------------------------------


_BOTTLENECK_LAYER: dict[Regime, str] = {
    Regime.ALU_BOUND:    "ALU peak FLOPS",
    Regime.L2_BOUND:     "L2 / LLC capacity",  # capacity-only in v4-1
    Regime.DRAM_BOUND:   "DRAM peak bandwidth",
    Regime.LAUNCH_BOUND: "kernel-launch overhead",
}


def check_regime(predicted: Regime, measured: Regime) -> tuple[bool, str]:
    """Return (pass, notes). Implements the v4-1 L2_BOUND soft-match.

    Predicted L2_BOUND passes if the measurement is anything except
    ALU_BOUND or DRAM_BOUND -- because v4-1 cannot positively infer
    L2_BOUND from a measurement (no on-chip BW peak; issue #61).
    """
    if predicted == measured:
        return True, ""
    if predicted == Regime.L2_BOUND:
        if measured not in (Regime.ALU_BOUND, Regime.DRAM_BOUND):
            return True, ("predicted L2_BOUND, measured "
                          f"{measured.value} (v4-1 soft match)")
        return False, (f"predicted L2_BOUND, measured "
                       f"{measured.value} (boundary drift)")
    return False, (f"predicted {predicted.value}, "
                   f"measured {measured.value}")


def _within_band(predicted: float, measured: float, tol: float) -> bool:
    if measured <= 0 or not math.isfinite(measured):
        return False
    if not math.isfinite(predicted):
        return False
    return abs(predicted - measured) / measured <= tol


def assert_record(
    *,
    hardware: str,
    op: str,
    shape: tuple,
    dtype: str,
    regime_predicted: Regime,
    latency_predicted_s: float,
    energy_predicted_j: Optional[float],
    flops: int,
    working_set_bytes: int,
    measured_latency_s: float,
    measured_energy_j: Optional[float],
    ctx: MeasurementContext,
) -> ValidationRecord:
    """Build a fully-populated ValidationRecord from a prediction +
    measurement pair.

    All three assertions are evaluated independently so a record can
    fail one without short-circuiting the others -- the heatmap then
    distinguishes "regime drift" from "latency math drift" from "energy
    math drift" in a single run.
    """
    regime_measured = infer_regime_measured(flops, working_set_bytes,
                                            measured_latency_s, ctx)
    pass_regime, regime_notes = check_regime(regime_predicted, regime_measured)

    tol_lat = TOL_LATENCY.get(regime_predicted, 0.30)
    tol_egy = TOL_ENERGY.get(regime_predicted, 0.40)

    pass_lat = _within_band(latency_predicted_s, measured_latency_s, tol_lat)

    if energy_predicted_j is None or measured_energy_j is None:
        pass_egy: Optional[bool] = None
    elif measured_latency_s < ENERGY_RELIABLE_LATENCY_S:
        # RAPL/NVML below ~1 ms is too noisy to score an energy band;
        # treat as "not measured" rather than fail. See #71.
        pass_egy = None
    else:
        pass_egy = _within_band(energy_predicted_j, measured_energy_j, tol_egy)

    return ValidationRecord(
        hardware=hardware,
        op=op,
        shape=tuple(shape),
        dtype=dtype,
        regime_predicted=regime_predicted.value,
        latency_predicted_ms=latency_predicted_s * 1000,
        energy_predicted_j=energy_predicted_j,
        regime_measured=regime_measured.value,
        latency_measured_ms=measured_latency_s * 1000,
        energy_measured_j=measured_energy_j,
        pass_regime=pass_regime,
        pass_latency=pass_lat,
        pass_energy=pass_egy,
        tolerance_latency=tol_lat,
        tolerance_energy=tol_egy,
        bottleneck_layer=_BOTTLENECK_LAYER.get(regime_predicted, "(unknown)"),
        notes=regime_notes,
    )
