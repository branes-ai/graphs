"""Regression tests for the CPU branch of RooflineAnalyzer._get_compute_efficiency_scale.

Locks in the V4-calibrated efficiency curve (issue #67). The curve was
re-anchored against ``validation/model_v4/results/baselines/i7_12700k_matmul.csv``
so single-kernel matmul predictions stop being 2-3x over-pessimistic.

These tests are pure unit tests on the public-but-private
``_get_compute_efficiency_scale`` helper -- no model build, no
PyTorch dependency. They protect against an accidental re-tightening
of the curve back to the pre-#67 CNN-aggregate values.
"""

import pytest

from graphs.core.structures import (
    OperationType,
    SubgraphDescriptor,
)
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.resource_model import Precision


@pytest.fixture(scope="module")
def i7_analyzer():
    hw = create_i7_12700k_mapper().resource_model
    return RooflineAnalyzer(hw, precision=Precision.FP32)


def _matmul_sg(M: int, K: int, N: int) -> SubgraphDescriptor:
    """Hand-build a single-op matmul SubgraphDescriptor for the curve test."""
    flops = 2 * M * K * N
    bpe = 4  # fp32
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["matmul"],
        node_names=["matmul"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="matmul",
        total_flops=flops,
        total_macs=flops // 2,
        total_input_bytes=(M * K + K * N) * bpe,
        total_output_bytes=M * N * bpe,
        total_weight_bytes=0,
    )


# Each anchor: (flops, expected_scale, label).
# Values come from the docstring of the new curve in roofline.py:CPU branch
# and from the V4 baseline empirical medians per flops decade.
@pytest.mark.parametrize("flops,expected_scale,label", [
    # < 1M -> flat 0.30 (V4 baseline 10^5..10^6 median 0.34, conservative)
    (100_000, 0.30, "tiny op flat"),
    (999_999, 0.30, "just under 1M"),
    # 1M -> 10M: linear ramp 0.30 -> 0.95 in log space
    (1_000_000, 0.30, "1M boundary lower"),
    (10_000_000, 0.95, "10M boundary upper"),
    # 10M -> 100M: linear ramp 0.95 -> 1.20
    (100_000_000, 1.20, "100M boundary upper"),
    # > 100M plateau at 1.20
    (1_000_000_000, 1.20, "1G plateau"),
    (10_000_000_000, 1.20, "10G plateau"),
])
def test_cpu_efficiency_curve_anchors(i7_analyzer, flops, expected_scale, label):
    """Anchor points of the V4-calibrated CPU efficiency curve. If
    these change, the change MUST come with new V4 baseline data --
    not from a guess."""
    # Use a small symbolic shape; only flops matters for the curve.
    sg = _matmul_sg(M=64, K=64, N=64)
    object.__setattr__(sg, "total_flops", flops)
    sg.total_macs = flops // 2
    actual = i7_analyzer._get_compute_efficiency_scale(sg)
    assert actual == pytest.approx(expected_scale, rel=1e-3), (
        f"{label} (flops={flops:,}): expected {expected_scale}, got {actual}"
    )


def test_cpu_efficiency_is_monotonic_in_log_flops(i7_analyzer):
    """The curve must be non-decreasing in flops -- a smaller op should
    never be predicted MORE efficient than a larger one (the post-#67
    curve has a steep ramp and a plateau, never declines)."""
    flops_grid = [1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 1e9, 1e10]
    sg = _matmul_sg(64, 64, 64)
    scales = []
    for f in flops_grid:
        object.__setattr__(sg, "total_flops", int(f))
        sg.total_macs = int(f) // 2
        scales.append(i7_analyzer._get_compute_efficiency_scale(sg))
    for i in range(1, len(scales)):
        assert scales[i] >= scales[i - 1] - 1e-9, (
            f"non-monotonic at flops={flops_grid[i]:.0e}: "
            f"scale {scales[i]} < {scales[i-1]} (prev)"
        )


def test_cpu_efficiency_does_not_regress_to_pre_67_values(i7_analyzer):
    """Belt-and-braces: assert the curve produces values at least 1.2x
    higher than the pre-#67 calibration at small/medium flops where the
    fix matters most. Catches an accidental revert.

    The 1.2x floor is intentionally loose -- a future PR may legitimately
    tighten the curve based on richer V4 data (e.g., per-shape-aspect
    correction). 1.2x catches the obvious "someone reverted the constants
    back to the pre-#67 CNN-aggregate calibration" regression without
    blocking principled refinements."""
    # Pre-#67 values at these flops (for reference only):
    #   1M:   0.15
    #   10M:  0.25
    #   100M: 0.70
    #   1G:   1.00
    pre_67 = {1e6: 0.15, 1e7: 0.25, 1e8: 0.70, 1e9: 1.00}
    sg = _matmul_sg(64, 64, 64)
    for f, pre in pre_67.items():
        object.__setattr__(sg, "total_flops", int(f))
        sg.total_macs = int(f) // 2
        post = i7_analyzer._get_compute_efficiency_scale(sg)
        assert post >= pre * 1.2, (
            f"flops={f:.0e}: scale {post} not enough higher than pre-#67 {pre}"
        )
