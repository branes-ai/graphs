"""Pin the V5 follow-up resolution of the N=262K floor failure.

Background: PR #109's analysis showed that no L3 ``achievable_fraction``
in [0.0, 1.0] could close the N=262K vector_add over-prediction --
the structural model gap was the missing per-core L2 tier on the i7
mapper.

This PR added that tier (``l2_cache_per_unit`` + ``l2_bandwidth_per_unit_bps``
on i7-12700K). The picker now correctly identifies N=262K as
L2-binding (3 MB working set fits the per-core L2 aggregate of
~10 MB), and with the calibrated L2 fraction the V4 floor passes
within the LAUNCH 30% tolerance band.

This test pins the resolution. The companion file
``test_i7_l3_calibration_cannot_fix_partial_l2_shapes.py`` (the
"structural failure" record) was deleted -- the failure was
structural with respect to the model state at the time, but the
model has since evolved.
"""

import pytest

from graphs.estimation.reuse_models import REUSE_MODELS
from graphs.estimation.tier_picker import pick_binding_tier
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper


N_262K = 262144
WS_BYTES_FP32 = 3 * N_262K * 4  # 3,145,728 bytes = 3 MB


def test_n_262k_binds_at_l2_after_per_core_tier_added():
    """Post per-core L2 tier addition: N=262K vector_add (WS=3 MB)
    binds at L2 (aggregate ~10 MB), not L3 (25 MB). The 3 MB
    operand fits the L2 tier so the picker stops at L2."""
    hw = create_i7_12700k_mapper().resource_model
    result = pick_binding_tier(
        REUSE_MODELS["vector_add"], (N_262K,), "fp32", hw.memory_hierarchy
    )
    assert result is not None
    assert result.binding_tier.name == "L2", (
        f"N=262K should bind at L2 post per-core L2 tier addition, got "
        f"{result.binding_tier.name}. If you removed the per-core L2 "
        f"fields, the V4 floor on this shape will regress to the "
        f"pre-resolution structural failure."
    )


def test_n_262k_l2_calibration_yields_passing_v4_prediction():
    """Forward-looking check: the L2 calibration (0.22) produces a
    memory_time prediction within the V4 LAUNCH 30% band of the
    measured 11 us. If this fails, either the calibration drifted
    or the L2 BW peak / capacity changed materially."""
    hw = create_i7_12700k_mapper().resource_model
    result = pick_binding_tier(
        REUSE_MODELS["vector_add"], (N_262K,), "fp32", hw.memory_hierarchy
    )
    assert result is not None
    eff_bw = result.binding_tier.effective_bandwidth_bps
    memory_time_s = WS_BYTES_FP32 / eff_bw
    # Plus dispatch overhead (~2 us vector_add op-aware floor)
    pred_with_dispatch_s = memory_time_s + 2e-6
    measured_s = 11.0e-6
    rel_err = abs(pred_with_dispatch_s - measured_s) / measured_s
    LAUNCH_BAND = 0.30
    assert rel_err <= LAUNCH_BAND, (
        f"N=262K prediction ({pred_with_dispatch_s*1e6:.2f} us) is "
        f"{rel_err*100:.0f}% off measured ({measured_s*1e6:.2f} us); "
        f"exceeds LAUNCH band ({LAUNCH_BAND*100:.0f}%). Re-check L2 "
        f"calibration in i7 mapper."
    )
