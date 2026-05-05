"""Tests for validation/model_v4/harness/assertions.py.

Covers the three assertion paths that turn a (prediction, measurement)
pair into a ValidationRecord:

1. ``infer_regime_measured`` -- classifies a measurement as ALU_BOUND,
   DRAM_BOUND, LAUNCH_BOUND, or AMBIGUOUS using achieved utilization.
2. ``check_regime`` -- compares predicted vs measured, with the v4-1
   L2_BOUND soft-match.
3. ``assert_record`` -- end-to-end, including per-regime tolerance
   bands and the optional-energy path.
"""

import math

import pytest

from validation.model_v4.harness.assertions import (
    MeasurementContext,
    TOL_ENERGY,
    TOL_LATENCY,
    assert_record,
    check_regime,
    infer_regime_measured,
)
from validation.model_v4.sweeps.classify import Regime


# ---------------------------------------------------------------------------
# infer_regime_measured
# ---------------------------------------------------------------------------


def _ctx(peak_flops: float = 1e12, peak_dram_bw_bps: float = 100e9,
         launch_overhead_s: float = 5e-6) -> MeasurementContext:
    return MeasurementContext(peak_flops=peak_flops,
                              peak_dram_bandwidth_bps=peak_dram_bw_bps,
                              launch_overhead_s=launch_overhead_s)


def test_infer_alu_bound_when_flops_util_above_threshold():
    """1 TFLOP / 1 ms = 1 TFLOPS achieved on a 1 TFLOPS peak -> 100% utilization."""
    ctx = _ctx(peak_flops=1e12, peak_dram_bw_bps=100e9)
    flops = int(1e12)
    bytes_ = int(1e6)  # tiny working set so BW utilization is irrelevant
    latency_s = 1e-3   # 1 ms
    assert infer_regime_measured(flops, bytes_, latency_s, ctx) == Regime.ALU_BOUND


def test_infer_dram_bound_when_bw_util_above_threshold():
    """100 GB / 1 ms = 100 TB/s ... too much. Use realistic numbers."""
    ctx = _ctx(peak_flops=1e15, peak_dram_bw_bps=100e9)
    # 1 GB working set / 12 ms ~= 83 GB/s -> 83% of 100 GB/s peak -> dram_bound
    flops = int(1e6)        # tiny FLOPs so flops_util is irrelevant
    bytes_ = int(1e9)
    latency_s = 12e-3
    assert infer_regime_measured(flops, bytes_, latency_s, ctx) == Regime.DRAM_BOUND


def test_infer_launch_bound_when_latency_below_5x_overhead():
    """Latency below 5*launch_overhead -> launch_bound regardless of flops/bytes."""
    ctx = _ctx(launch_overhead_s=5e-6)
    # 20us < 5 * 5us = 25us
    assert infer_regime_measured(1, 1, 2e-5, ctx) == Regime.LAUNCH_BOUND


def test_infer_ambiguous_when_neither_threshold_met():
    """Below both utilization thresholds and above launch_bound = AMBIGUOUS.
    v4-1 limitation: cannot positively identify L2_BOUND from a measurement
    without on-chip BW peaks."""
    ctx = _ctx(peak_flops=1e12, peak_dram_bw_bps=100e9, launch_overhead_s=5e-6)
    # 1 ms latency with low FLOPS and low byte traffic
    flops = int(1e8)        # 1e8 / 1e-3 = 1e11 = 10% of peak FLOPS
    bytes_ = int(1e6)       # 1e6 / 1e-3 = 1 GB/s = 1% of peak BW
    latency_s = 1e-3
    assert infer_regime_measured(flops, bytes_, latency_s, ctx) == Regime.AMBIGUOUS


def test_infer_handles_zero_latency():
    """Zero or negative latency is invalid; surface as AMBIGUOUS rather
    than divide-by-zero."""
    ctx = _ctx()
    assert infer_regime_measured(1, 1, 0.0, ctx) == Regime.AMBIGUOUS
    assert infer_regime_measured(1, 1, -1e-3, ctx) == Regime.AMBIGUOUS


# ---------------------------------------------------------------------------
# check_regime -- L2_BOUND soft-match is the interesting path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", [Regime.ALU_BOUND, Regime.DRAM_BOUND,
                                    Regime.LAUNCH_BOUND])
def test_check_regime_exact_match(regime):
    ok, _ = check_regime(regime, regime)
    assert ok


def test_check_regime_predicted_l2_passes_when_measured_ambiguous():
    """The v4-1 soft-match: a shape predicted L2_BOUND that measures as
    AMBIGUOUS (in-between) still passes -- we just couldn't pinpoint
    which on-chip BW it stressed."""
    ok, notes = check_regime(Regime.L2_BOUND, Regime.AMBIGUOUS)
    assert ok
    assert "soft match" in notes


def test_check_regime_predicted_l2_passes_when_measured_launch():
    """L2_BOUND predicted, LAUNCH_BOUND measured: tiny shape, soft match."""
    ok, _ = check_regime(Regime.L2_BOUND, Regime.LAUNCH_BOUND)
    assert ok


def test_check_regime_predicted_l2_fails_when_measured_alu_or_dram():
    """L2_BOUND predicted but measured as one of the firm regimes
    indicates a real boundary drift -- the shape isn't where the
    classifier said it was."""
    ok_alu, _ = check_regime(Regime.L2_BOUND, Regime.ALU_BOUND)
    ok_dram, _ = check_regime(Regime.L2_BOUND, Regime.DRAM_BOUND)
    assert not ok_alu
    assert not ok_dram


def test_check_regime_alu_predicted_dram_measured_fails():
    """The bottleneck-misclassification case the v4 plan is designed
    to catch."""
    ok, notes = check_regime(Regime.ALU_BOUND, Regime.DRAM_BOUND)
    assert not ok
    assert "alu_bound" in notes and "dram_bound" in notes


# ---------------------------------------------------------------------------
# Tolerance tables
# ---------------------------------------------------------------------------


def test_tolerance_table_is_monotonic():
    """Tighter on the firm-spec end (ALU), wider on the noisy end (LAUNCH).
    This ordering is a contract -- if we ever regress it, validation
    starts asserting more loosely on what should be the strongest signal."""
    order = [Regime.ALU_BOUND, Regime.L2_BOUND, Regime.DRAM_BOUND, Regime.LAUNCH_BOUND]
    lats = [TOL_LATENCY[r] for r in order]
    egys = [TOL_ENERGY[r] for r in order]
    assert lats == sorted(lats)
    assert egys == sorted(egys)
    # Energy should never be tighter than latency for the same regime
    for r in order:
        assert TOL_ENERGY[r] >= TOL_LATENCY[r]


# ---------------------------------------------------------------------------
# assert_record end-to-end
# ---------------------------------------------------------------------------


def _good_kwargs(**overrides) -> dict:
    """Synthetic fixture chosen so the measurement classifies as DRAM_BOUND.

    To land DRAM_BOUND we need bw_util >= 0.7 AND flops_util < 0.7. Use a
    deliberately memory-heavy / compute-light synthetic workload so the
    inference is unambiguous:

      bw_util  = 100MB / (1ms * 100 GB/s) = 100% -> DRAM_BOUND
      flops_util = 1e6  / (1ms * 1 TFLOPS) = 1e-3 -> not ALU_BOUND
      latency  = 1 ms >> 25 us launch threshold -> not LAUNCH_BOUND

    Predicted latency 0.9 ms vs measured 1.0 ms = 10% over, within the
    25% DRAM tolerance band. Energy similarly 0.09 vs 0.10 = 11% over,
    within the 30% DRAM energy band.
    """
    base = dict(
        hardware="i7_12700k",
        op="matmul",
        shape=(1024, 1024, 1024),
        dtype="fp32",
        regime_predicted=Regime.DRAM_BOUND,
        latency_predicted_s=0.9e-3,
        energy_predicted_j=0.09,
        flops=int(1e6),
        working_set_bytes=100 * 1024 * 1024,
        measured_latency_s=1.0e-3,
        measured_energy_j=0.10,
        ctx=_ctx(peak_flops=1e12, peak_dram_bw_bps=100e9),
    )
    base.update(overrides)
    return base


def test_assert_record_clean_pass_dram_regime():
    rec = assert_record(**_good_kwargs())
    assert rec.pass_regime is True
    assert rec.pass_latency is True
    assert rec.pass_energy is True
    assert rec.all_pass()
    assert rec.regime_predicted == "dram_bound"
    # Fields populated
    assert rec.tolerance_latency == TOL_LATENCY[Regime.DRAM_BOUND]
    assert rec.tolerance_energy == TOL_ENERGY[Regime.DRAM_BOUND]


def test_assert_record_latency_outside_band():
    """Predicted 0.9ms, measured 20ms -> way over -> fails DRAM 25% band.

    A measurement this far off also no longer classifies as DRAM_BOUND
    (utilization drops below the 70% threshold), so pass_regime fails
    too -- which is the correct propagation: if the model's latency
    prediction is wildly off, the bottleneck classification was probably
    wrong as well. The point of this test is just that pass_latency
    flips to False; whether pass_regime survives is shape-dependent.
    """
    rec = assert_record(**_good_kwargs(measured_latency_s=20e-3))
    assert rec.pass_latency is False
    assert not rec.all_pass()


def test_assert_record_energy_missing_does_not_fail_overall():
    """RAPL unreadable -> energy_measured_j=None -> pass_energy=None.
    Record still counts as pass overall if regime + latency pass."""
    rec = assert_record(**_good_kwargs(measured_energy_j=None))
    assert rec.pass_energy is None
    assert rec.all_pass() is True


def test_assert_record_predicted_energy_missing_skips_check():
    """Analyzer didn't model energy -> energy_predicted_j=None -> skip
    energy assertion."""
    rec = assert_record(**_good_kwargs(energy_predicted_j=None))
    assert rec.pass_energy is None
    assert rec.all_pass() is True


def test_assert_record_l2_predicted_ambiguous_measured_passes():
    """v4-1 soft match: L2_BOUND predicted, AMBIGUOUS measured (low
    utilization on every layer) still counts as a pass for the regime
    assertion, with a note explaining the soft match."""
    kwargs = _good_kwargs(
        regime_predicted=Regime.L2_BOUND,
        # measurement that lands in AMBIGUOUS: low FLOPS util AND low BW util
        flops=int(1e8),
        working_set_bytes=int(1e6),
        measured_latency_s=1e-3,
    )
    rec = assert_record(**kwargs)
    assert rec.pass_regime is True
    assert "soft match" in rec.notes


def test_assert_record_bottleneck_layer_label():
    """Each regime gets a human-readable bottleneck-layer label so a
    failure in the report points at the architectural assumption."""
    rec = assert_record(**_good_kwargs(regime_predicted=Regime.ALU_BOUND))
    assert "ALU" in rec.bottleneck_layer
    rec = assert_record(**_good_kwargs(regime_predicted=Regime.DRAM_BOUND))
    assert "DRAM" in rec.bottleneck_layer


def test_assert_record_to_dict_is_json_serializable():
    import json
    rec = assert_record(**_good_kwargs())
    d = rec.to_dict()
    # Can roundtrip through JSON without losing fields
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["hardware"] == "i7_12700k"
    assert parsed["op"] == "matmul"
    assert parsed["pass_regime"] is True


def test_within_band_edge_cases():
    """Indirectly via assert_record: NaN / inf predictions never pass."""
    rec = assert_record(**_good_kwargs(latency_predicted_s=math.nan))
    assert rec.pass_latency is False
    rec = assert_record(**_good_kwargs(latency_predicted_s=math.inf))
    assert rec.pass_latency is False
    # Zero measured -> can't compute relative error -> fail
    rec = assert_record(**_good_kwargs(measured_latency_s=0.0))
    assert rec.pass_latency is False
