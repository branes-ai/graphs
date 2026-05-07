"""Pin the V5-5 follow-up L3 calibration analysis for i7-12700K.

Background: vector_add at N=262K fp32 (WS=3 MB) on i7 measures 11 us
but the V4 floor predicts 19 us with L3 = 0.84 (current calibration).
The +75% over-prediction is NOT a calibration value error -- it's a
structural model gap.

The full analysis lives at
``docs/calibration/i7-12700k-l3-calibration-analysis.md``. The
load-bearing claim is:

  No L3 ``achievable_fraction`` in [0.0, 1.0] (the dataclass invariant
  range) can produce a prediction matching the N=262K measurement.
  Even at 1.0 (perfect L3 streaming, non-physical), the prediction
  would still exceed measured.

This test is the regression gate for that claim. If it ever fails,
either the dispatch overhead changed materially OR the operand-aware
picker stopped routing N=262K to L3 (e.g. because per-core L2 was
added as a tier), and the L3 calibration story needs revisiting.
"""

import pytest

from graphs.estimation.reuse_models import REUSE_MODELS
from graphs.estimation.tier_picker import pick_binding_tier
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper


# i7 V5-2b vector_add baseline at N=262K fp32: 10.96 us measured.
# Subtract the dispatch overhead derived in
# docs/calibration/i7-12700k-l1-calibration-analysis.md (1.79 us) to
# get the at-the-data time.
N_262K = 262144
WS_BYTES_FP32 = 3 * N_262K * 4  # 3,145,728
MEASURED_LATENCY_S = 10.96e-6
DISPATCH_OVERHEAD_S = 1.79e-6
AT_DATA_LATENCY_S = MEASURED_LATENCY_S - DISPATCH_OVERHEAD_S


def test_n_262k_binds_at_l3_per_operand_aware_picker():
    """Sanity: with the V5-5 followup operand-aware picker, the
    N=262K vector_add (WS=3 MB) does bind at L3 on i7. If this stops
    being true (e.g. an L2 tier got added), the analysis below is
    moot and needs revisiting."""
    hw = create_i7_12700k_mapper().resource_model
    result = pick_binding_tier(
        REUSE_MODELS["vector_add"], (N_262K,), "fp32", hw.memory_hierarchy
    )
    assert result is not None
    assert result.binding_tier.name == "L3", (
        f"Expected N=262K to bind at L3, got {result.binding_tier.name}. "
        f"If a per-core L2 tier was added to the i7 mapper, update "
        f"docs/calibration/i7-12700k-l3-calibration-analysis.md and "
        f"re-derive the L3 calibration from the new L3-only baseline."
    )


def test_l3_calibration_alone_cannot_match_n_262k_measurement():
    """The structural claim: even at L3.achievable_fraction = 1.0
    (the dataclass max, non-physical perfect streaming), the
    predicted memory_time exceeds the measurement at N=262K.

    No L3 calibration value can fix this. The fix is a model
    refinement (per-core L2 tier, weighted-tier model, or
    cache-line-aware partial residency)."""
    hw = create_i7_12700k_mapper().resource_model
    l3_peak = next(t.peak_bandwidth_bps for t in hw.memory_hierarchy if t.name == "L3")
    # Perfect (non-physical) L3 streaming: BW = peak, fraction = 1.0
    perfect_memory_time_s = WS_BYTES_FP32 / l3_peak
    # Even at perfect L3 BW, the prediction exceeds the measurement
    # (inclusive of dispatch). Any [0, 1] fraction makes this gap
    # worse, so the calibration alone can't close it.
    assert perfect_memory_time_s > AT_DATA_LATENCY_S, (
        f"At L3 peak (200 GB/s, fraction=1.0), memory_time = "
        f"{perfect_memory_time_s * 1e6:.2f} us, measurement at-data = "
        f"{AT_DATA_LATENCY_S * 1e6:.2f} us. If perfect L3 streaming "
        f"now matches, an L2 tier must have been added or the L3 peak "
        f"changed -- revisit the calibration analysis doc."
    )
    # Quantify the structural floor: the gap as a fraction of measured
    gap = (perfect_memory_time_s - AT_DATA_LATENCY_S) / AT_DATA_LATENCY_S
    assert gap > 0.5, (
        f"Structural gap is unexpectedly small ({gap*100:.0f}%); the "
        f"L3 calibration story may now be tractable. Re-evaluate."
    )


def test_n_262k_required_bandwidth_exceeds_l3_peak():
    """Forward-looking check: derive the BW the N=262K shape would
    need to land within the LAUNCH 30% tolerance band (the regime
    the V4 sweep classifies it into). If the required BW exceeds
    L3 peak, the model gap is structural -- no calibration fixes it.

    Required BW = WS / (measurement * (1 - tolerance) - dispatch),
    where measurement-tolerance gives the upper-bound prediction
    that still passes."""
    hw = create_i7_12700k_mapper().resource_model
    l3_peak = next(t.peak_bandwidth_bps for t in hw.memory_hierarchy if t.name == "L3")
    LAUNCH_BAND = 0.30
    upper_pred_s = MEASURED_LATENCY_S * (1 + LAUNCH_BAND)
    upper_at_data_s = upper_pred_s - DISPATCH_OVERHEAD_S
    required_bw_bps = WS_BYTES_FP32 / upper_at_data_s
    assert required_bw_bps > l3_peak, (
        f"Required BW ({required_bw_bps/1e9:.1f} GB/s) is now <= L3 peak "
        f"({l3_peak/1e9:.1f} GB/s); the N=262K floor failure may now be "
        f"reachable via L3 calibration alone. Recompute the regression "
        f"in docs/calibration/i7-12700k-l3-calibration-analysis.md."
    )
