"""Tests for the bandwidth-aware classifier dispatch (issue #61).

The classifier has two paths:

1. **Capacity-only** (v4-1 default): when the hardware mapper does NOT
   populate ``l1_bandwidth_per_unit_bps`` and ``l2_bandwidth_bps``,
   shapes that fit in L1 with low OI fall into ``AMBIGUOUS``.
   Validated by ``test_classify.py``.

2. **Bandwidth-aware** (#61): when the mapper DOES populate the
   on-chip BW peaks, the classifier emits ``L1_BOUND`` for shapes that
   fit in L1 with OI below the L1 roofline breakpoint.

This module exercises path 2 on the three V4 reference targets that
ship with BW peaks populated (i7-12700K, H100-SXM5-80GB, KPU-T64).
"""

from __future__ import annotations

import math

import pytest

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper
from graphs.hardware.resource_model import Precision

from validation.model_v4.sweeps.classify import (
    Regime,
    classify_regime,
    hardware_capacities,
    op_footprint,
)


# ---------------------------------------------------------------------------
# Schema: BW peak fields are populated on the three reference mappers
# ---------------------------------------------------------------------------


def test_i7_has_l1_and_l3_bandwidth_peaks_populated():
    """i7 has L3 (the LLC on x86) but no distinct L2 in the M1 schema
    convention; ``l2_bandwidth_bps`` stays None and ``l3_bandwidth_bps``
    carries the LLC bandwidth."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.l1_bandwidth_per_unit_bps is not None
    assert hw.l1_bandwidth_per_unit_bps > 0
    assert hw.l3_bandwidth_bps is not None
    assert hw.l3_bandwidth_bps > 0


def test_h100_has_l1_and_l2_bandwidth_peaks_populated():
    """H100 has both per-SM L1 BW and shared L2 BW from the Hopper
    whitepaper / microbench studies."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    assert hw.l1_bandwidth_per_unit_bps is not None
    assert hw.l1_bandwidth_per_unit_bps > 0
    assert hw.l2_bandwidth_bps is not None
    assert hw.l2_bandwidth_bps > 0


def test_kpu_t64_has_l1_and_l2_bandwidth_peaks_populated():
    """KPU-T64 vendor spec exposes both per-tile scratchpad BW and
    shared L2 BW."""
    hw = create_kpu_t64_mapper().resource_model
    assert hw.l1_bandwidth_per_unit_bps is not None
    assert hw.l1_bandwidth_per_unit_bps > 0
    assert hw.l2_bandwidth_bps is not None
    assert hw.l2_bandwidth_bps > 0


# ---------------------------------------------------------------------------
# Backward compat: mappers without BW peaks still classify cleanly
# ---------------------------------------------------------------------------


def test_legacy_mapper_without_bw_peaks_keeps_capacity_only_path():
    """A mapper that doesn't populate the new fields must still classify
    via the v4-1 capacity-only path. Verified by zero-ing the new fields
    on a copy of i7 and confirming hardware_capacities returns
    ``peak_l1_bandwidth_bps=None``."""
    hw = create_i7_12700k_mapper().resource_model
    # Mutate a fresh instance to simulate an un-augmented mapper
    import copy
    hw2 = copy.copy(hw)
    hw2.l1_bandwidth_per_unit_bps = None
    hw2.l2_bandwidth_bps = None
    hw2.l3_bandwidth_bps = None
    cap = hardware_capacities(hw2, Precision.FP32)
    assert cap.peak_l1_bandwidth_bps is None
    assert cap.peak_l2_bandwidth_bps is None
    assert cap.l1_ai_breakpoint is None


# ---------------------------------------------------------------------------
# HardwareCapacities: aggregate BW math is correct
# ---------------------------------------------------------------------------


def test_hardware_capacities_aggregates_l1_bw_across_units():
    """Aggregate L1 BW = per-unit * compute_units. Sanity-check on H100
    (132 SMs) that the value comes out in the published H100 ballpark."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    cap = hardware_capacities(hw, Precision.FP16)
    expected = hw.l1_bandwidth_per_unit_bps * hw.compute_units
    assert cap.peak_l1_bandwidth_bps == expected
    # Ballpark: 132 SMs * ~500 GB/s = ~66 TB/s. Allow a wide band so this
    # test doesn't break on minor per-SM-spec adjustments.
    assert 30e12 < cap.peak_l1_bandwidth_bps < 100e12


def test_hardware_capacities_l2_bw_passes_through_aggregate():
    """L2 BW is already aggregate by spec; HardwareCapacities passes it
    through unchanged."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    cap = hardware_capacities(hw, Precision.FP16)
    assert cap.peak_l2_bandwidth_bps == hw.l2_bandwidth_bps


def test_l1_ai_breakpoint_property_uses_aggregate_l1_bw():
    """The L1 AI breakpoint is peak_FLOPS / aggregate_L1_BW. For H100 fp16
    (~989 TFLOPS / ~67 TB/s ≈ 15 FLOPS/B) this is much lower than the
    DRAM AI breakpoint (~295) -- which is exactly the point: shapes that
    fit in L1 with OI in (L1_break, DRAM_break) become L1_BOUND, which
    the v4-1 capacity-only classifier was forced to call AMBIGUOUS."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    cap = hardware_capacities(hw, Precision.FP16)
    l1_bp = cap.l1_ai_breakpoint
    assert l1_bp is not None
    assert 5 < l1_bp < 50, (
        f"H100 fp16 L1 breakpoint {l1_bp:.2f} is outside the expected "
        f"5-50 FLOPS/byte band -- spec may have shifted, recheck the "
        f"aggregate L1 BW computation"
    )
    assert l1_bp < cap.ai_breakpoint, (
        f"L1 breakpoint {l1_bp:.2f} should be below DRAM breakpoint "
        f"{cap.ai_breakpoint:.2f}; otherwise L1_BOUND can never fire"
    )


# ---------------------------------------------------------------------------
# Dispatch: a hand-curated low-OI L1-resident shape lands in L1_BOUND
# ---------------------------------------------------------------------------


def _synthetic_low_l1_bw_mapper():
    """Build a copy of H100 with a deliberately *narrow* L1 BW so the
    L1 roofline breakpoint pushes UP into a band that real workload
    shapes can occupy.

    On the actual H100 mapper, L1 BW is so wide (~67 TB/s aggregate)
    that the L1 breakpoint (~15 FLOPS/B) sits below where any shape
    that clears LAUNCH_BOUND can land. That is true to the silicon
    -- L1_BOUND is structurally rare on flagship accelerators -- but
    means we can't unit-test the dispatch path on real H100 alone.
    Lowering peak_L1_BW makes L1_break large enough to be reachable.

    Returns the mutated resource_model. The mapper's resource_model is
    a dataclass instance; copy.copy gives a shallow copy, sufficient
    here because we only twiddle one float field.
    """
    import copy
    hw = create_h100_sxm5_80gb_mapper().resource_model
    hw2 = copy.copy(hw)
    # Shrink L1 BW by 100x: aggregate becomes ~670 GB/s -> L1 breakpoint
    # ~989/0.67 = ~1480 FLOPS/B, which is well above realistic OI for
    # most shapes that fit in L1 (so they classify as L1_BOUND).
    hw2.l1_bandwidth_per_unit_bps = (hw.l1_bandwidth_per_unit_bps or 1.0) / 100.0
    return hw2


def test_l1_bound_emitted_for_low_oi_l1_resident_shape():
    """Synthetic-narrow-L1 H100 with a tall-skinny linear: WS fits in L1
    AND OI is below the (now-elevated) L1 breakpoint. Pass a tiny
    ``launch_overhead_s`` to the classifier so LAUNCH_BOUND doesn't
    eat the test shape (real-H100 L1-resident shapes are always
    launch-bound; see ``test_l1_bound_unreachable_on_real_h100_is_documented``).
    This test isolates the L1-vs-ALU-vs-AMBIGUOUS dispatch logic."""
    hw = _synthetic_low_l1_bw_mapper()
    cap = hardware_capacities(hw, Precision.FP16)

    # B=4096, IN=2048, OUT=4 fp16:
    #   flops = 2*4096*2048*4 = 67M
    #   ws = (4096*2048 + 2048*4 + 4096*4) * 2 = 16.8 MB (< L1=33 MB)
    #   OI = 67M / 16.8M = ~4 FLOPS/B (well below the elevated L1 break)
    shape = (4096, 2048, 4)
    fp = op_footprint("linear", shape, "fp16")
    assert fp.working_set_bytes < cap.l1_total_bytes, (
        f"shape no longer fits in L1: ws={fp.working_set_bytes}, "
        f"L1={cap.l1_total_bytes}")
    l1_bp = cap.l1_ai_breakpoint
    assert l1_bp is not None
    assert fp.operational_intensity < l1_bp, (
        f"OI={fp.operational_intensity:.2f} should be below L1 "
        f"breakpoint {l1_bp:.2f} for this test setup")

    # Tiny launch overhead lets the L1_BOUND check fire instead of
    # LAUNCH_BOUND swallowing every L1-resident shape.
    regime = classify_regime("linear", shape, "fp16", hw,
                             launch_overhead_s=1e-12)
    assert regime == Regime.L1_BOUND, (
        f"expected L1_BOUND for low-OI L1-resident shape, got {regime.value}; "
        f"OI={fp.operational_intensity:.2f}, L1_break={l1_bp:.2f}, "
        f"DRAM_break={cap.ai_breakpoint:.2f}")


def test_l1_bound_falls_back_to_ambiguous_when_l1_bw_peak_missing():
    """The same shape on a hardware that doesn't populate
    ``l1_bandwidth_per_unit_bps`` falls back to AMBIGUOUS (the v4-1
    capacity-only behavior)."""
    hw = _synthetic_low_l1_bw_mapper()
    import copy
    hw2 = copy.copy(hw)
    hw2.l1_bandwidth_per_unit_bps = None   # disable the new path

    regime = classify_regime("linear", (4096, 2048, 4), "fp16", hw2,
                             launch_overhead_s=1e-12)
    assert regime == Regime.AMBIGUOUS, (
        f"with l1_bandwidth_per_unit_bps=None, low-OI L1-resident "
        f"shape must fall back to AMBIGUOUS; got {regime.value}")


def test_l1_bound_unreachable_on_real_h100_is_documented():
    """On the actual H100 mapper, L1 BW is so wide that no shape which
    fits in L1 AND clears LAUNCH_BOUND can have OI below the L1 break.
    Pin this property so a future change to the H100 BW spec doesn't
    silently shift the regime distribution -- if this test fails, the
    spec change has made L1_BOUND structurally reachable on H100, which
    is reportable in the PR description but not silently."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    cap = hardware_capacities(hw, Precision.FP16)
    # Maximum flops you can have at OI < L1_break with WS = L1_total:
    max_flops = cap.l1_ai_breakpoint * cap.l1_total_bytes
    # Minimum flops to clear LAUNCH_BOUND (5 * default 5us launch overhead):
    min_flops_for_compute_time = cap.peak_flops * 5e-6 * 5
    assert max_flops < min_flops_for_compute_time, (
        f"L1_BOUND has become reachable on real H100: max L1-resident "
        f"low-OI flops {max_flops:.2e} exceeds the LAUNCH_BOUND floor "
        f"{min_flops_for_compute_time:.2e}. Update test_classify_with_bw_peaks "
        f"to add a real-H100 L1_BOUND assertion."
    )


def test_alu_bound_still_fires_when_oi_above_dram_breakpoint():
    """Negative control: a shape with OI well above the DRAM breakpoint
    must still classify as ALU_BOUND, not L1_BOUND. Verifies the
    dispatch order is correct (ALU check first, L1 check second)."""
    # H100 fp16 alu_bound shape from the existing test_classify suite
    hw = create_h100_sxm5_80gb_mapper().resource_model
    regime = classify_regime("matmul", (2048, 2048, 2048), "fp16", hw)
    assert regime == Regime.ALU_BOUND


# ---------------------------------------------------------------------------
# i7-12700K: only L1 BW is populated; L2 BW lives in l3_bandwidth_bps
# ---------------------------------------------------------------------------


def test_i7_uses_l3_bandwidth_field_not_l2_bandwidth_field():
    """Per the M1 schema convention, on x86 the LLC IS L3, and
    ``l2_cache_total`` carries the L3 capacity. Matching, the L3 BW
    field carries the LLC bandwidth and ``l2_bandwidth_bps`` is None
    (no distinct L2 layer at the M1-schema level)."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.l2_bandwidth_bps is None, (
        "i7-12700K must keep l2_bandwidth_bps=None; the LLC value lives "
        "in l3_bandwidth_bps per the M1 schema convention")
    assert hw.l3_bandwidth_bps is not None
