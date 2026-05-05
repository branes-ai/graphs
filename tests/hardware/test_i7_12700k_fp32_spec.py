"""Regression tests for the i7-12700K FP32 spec correction (issue #68).

The pre-#68 spec used 16 ops/cycle/core for FP32, assuming a single
FMA unit. Real Alder Lake P-cores have two FMA units that MKL routinely
saturates -- V4 baseline measured 909 GFLOPS achieved on a 2048^3 cube
matmul, exceeding the spec's 720 GFLOPS theoretical peak (physically
impossible relative to that spec).

This PR doubled the theoretical peak (16 -> 32 ops/cycle, 720 -> 1440
GFLOPS) AND halved efficiency_factor (0.50 -> 0.25 in the consumer-
continuous thermal profile, 0.60 -> 0.30 in the consumer-continuous-
large profile) so the EFFECTIVE peak is unchanged at 360 GFLOPS. The
spec is now architecturally honest; effective sustained throughput is
preserved.

These tests lock in:
1. theoretical peak is now 1440 GFLOPS (matches AVX2 + dual-FMA math)
2. effective peak (post-thermal-derate) is unchanged at 360 GFLOPS
3. precision_profiles.peak_ops_per_sec matches the theoretical headline
4. ops_per_unit_per_clock for FP32 is 32 (not 16)
"""

import pytest

from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import (
    create_i7_12700k_mapper,
    create_i7_12700k_large_mapper,
)
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Tiny variant (default)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def i7_tiny():
    return create_i7_12700k_mapper().resource_model


def test_i7_tiny_theoretical_peak_fp32_is_dual_fma(i7_tiny):
    """Theoretical headline must reflect dual FMA (#68): 1440 GFLOPS."""
    assert i7_tiny.get_peak_ops(Precision.FP32) == 1440e9


def test_i7_tiny_effective_peak_fp32_unchanged(i7_tiny):
    """The whole point of the compensating efficiency_factor halve: the
    EFFECTIVE peak (what RooflineAnalyzer uses) stays at 360 GFLOPS.
    No V4 pass-rate disruption from a pure spec correction."""
    effective = RooflineAnalyzer._get_effective_peak_ops(i7_tiny, Precision.FP32)
    assert effective == 360e9


def test_i7_tiny_precision_profile_matches_theoretical(i7_tiny):
    """precision_profiles.peak_ops_per_sec carries the theoretical
    headline so the legacy fallback path (when thermal_operating_points
    is missing) returns the right value."""
    assert i7_tiny.precision_profiles[Precision.FP32].peak_ops_per_sec == 1440e9


# ---------------------------------------------------------------------------
# Large variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def i7_large():
    return create_i7_12700k_large_mapper().resource_model


def test_i7_large_theoretical_peak_fp32_matches_tiny(i7_large):
    """Same physical hardware -> same theoretical peak. Only the thermal
    derate values differ between the tiny and large variants."""
    assert i7_large.get_peak_ops(Precision.FP32) == 1440e9


def test_i7_large_effective_peak_fp32_unchanged(i7_large):
    """large variant: effective = sustained_ops * tile_utilization * efficiency_factor.
    Pre-#68:  720 * 0.80 * 0.60 = 345.6 GFLOPS
    Post-#68: 1440 * 0.80 * 0.30 = 345.6 GFLOPS (theoretical doubled,
    efficiency_factor halved -> net unchanged)."""
    effective = RooflineAnalyzer._get_effective_peak_ops(i7_large, Precision.FP32)
    assert effective == pytest.approx(345.6e9, rel=1e-6)


# ---------------------------------------------------------------------------
# Behavioral check: "above peak" no longer happens
# ---------------------------------------------------------------------------


def test_i7_max_measured_is_below_theoretical_peak(i7_tiny):
    """V4 baseline measured 909 GFLOPS as the max single-kernel matmul
    on i7-12700K. After #68 the theoretical spec accommodates this --
    no measurement exceeds the spec."""
    measured_max_gflops = 909.0  # from V4 baseline (2048^3 fp32)
    spec_theoretical_gflops = i7_tiny.get_peak_ops(Precision.FP32) / 1e9
    assert measured_max_gflops < spec_theoretical_gflops, (
        f"Measured {measured_max_gflops} GFLOPS still exceeds spec "
        f"{spec_theoretical_gflops} -- spec correction insufficient."
    )
    # Also: measured max should be a reasonable fraction of theoretical.
    # MKL on Alder Lake achieves ~63% of dual-FMA theoretical for cubes.
    util = measured_max_gflops / spec_theoretical_gflops
    assert 0.40 < util < 0.80, (
        f"Implausible utilization {util*100:.0f}% -- check spec or measurement."
    )
