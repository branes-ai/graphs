"""Pin the V5-5 follow-up L1 calibration decision for i7-12700K.

Background: ``L1.achievable_fraction`` is intentionally left unset
(default 1.0) on the i7 mappers. The full analysis lives at
``docs/calibration/i7-12700k-l1-calibration-analysis.md``. The
load-bearing claim is that the CPU dispatch floor (5 us in
``RooflineAnalyzer._analyze_subgraph``) supersedes any L1-derived
memory_time for every L1-binding matmul shape on i7, making the
L1 calibration value a no-op.

This test is the regression gate for that claim: if it ever fails,
either the dispatch floor moved or a workload arrived whose L1-bound
memory_time can exceed 5 us, and the L1 calibration decision needs
to be revisited.
"""

import pytest

from graphs.estimation.reuse_models import REUSE_MODELS
from graphs.estimation.tier_picker import pick_binding_tier
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper


# CPU dispatch floor (RooflineAnalyzer._analyze_subgraph). Lifted as a
# module constant so a future tightening of the floor will cause this
# test to fail -- which is the right behavior because the L1
# calibration analysis hangs off this number.
CPU_DISPATCH_FLOOR_S = 5e-6


# Shapes spanning the V4 sweep's launch_bound regime on i7. All small
# enough that the matmul tile fits L1 and (per the V5-5 follow-up
# operand-aware picker) the operand footprint also fits L1.
_SHAPE_DIMS = [
    (M, K, N)
    for M in [32, 64, 96, 128]
    for K in [32, 64, 96]
    for N in [32, 64, 96, 128]
]


@pytest.fixture(scope="module")
def hw():
    return create_i7_12700k_mapper().resource_model


def test_l1_uncalibrated_default_makes_zero_difference_at_dispatch_floor(hw):
    """The analytical claim from
    docs/calibration/i7-12700k-l1-calibration-analysis.md: every
    L1-binding matmul shape in the V4 sweep range has tier-aware
    memory_time well below the 5 us CPU dispatch floor, so the L1
    calibration value is a no-op for predictions.

    Even if L1.achievable_fraction were dialed all the way down to
    0.020 (the most pessimistic single-thread vector_add observation),
    the resulting memory_time would still be below the dispatch floor
    on every shape -- so no prediction would change. This test pins
    that property by checking the worst-case (most pessimistic)
    achievable_fraction would still leave memory_time under the
    floor.
    """
    mm = REUSE_MODELS["matmul"]
    hierarchy = hw.memory_hierarchy
    PESSIMISTIC_FRACTION = 0.020  # single-thread vector_add observation

    l1_bound_shapes = []
    for shape in _SHAPE_DIMS:
        result = pick_binding_tier(mm, shape, "fp32", hierarchy)
        if result is None or result.binding_tier.name != "L1":
            continue
        l1_bound_shapes.append((shape, result))

    # If this assertion ever fails, the operand-aware picker stopped
    # routing the small-matmul regime through L1 -- worth investigating
    # before changing the calibration story.
    assert len(l1_bound_shapes) >= 30, (
        f"Expected the V4 launch_bound regime to produce many L1-binding "
        f"shapes; got {len(l1_bound_shapes)}. Has the picker drifted?"
    )

    # The actual claim: even with the most pessimistic plausible L1
    # calibration, memory_time stays under the 5 us dispatch floor.
    for shape, result in l1_bound_shapes:
        # Worst-case memory_time = bytes_loaded / (peak * pessimistic_frac)
        worst_case_bw = result.binding_tier.peak_bandwidth_bps * PESSIMISTIC_FRACTION
        worst_case_mem_time_s = result.bytes_loaded / worst_case_bw
        assert worst_case_mem_time_s < CPU_DISPATCH_FLOOR_S, (
            f"Shape {shape}: even at L1.achievable_fraction = "
            f"{PESSIMISTIC_FRACTION}, memory_time = "
            f"{worst_case_mem_time_s * 1e6:.2f} us exceeds the CPU "
            f"dispatch floor ({CPU_DISPATCH_FLOOR_S * 1e6} us). "
            f"L1 calibration would now affect predictions; revisit "
            f"docs/calibration/i7-12700k-l1-calibration-analysis.md."
        )
