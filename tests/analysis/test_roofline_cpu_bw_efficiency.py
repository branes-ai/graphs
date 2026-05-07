"""Regression tests for the CPU branch of RooflineAnalyzer._get_bandwidth_efficiency_scale.

Note: ``_get_bandwidth_efficiency_scale`` was deprecated in V5-6 Phase A
(see ``docs/v5/bw-efficiency-scale-retirement-status.md``). These tests
intentionally exercise the deprecated fallback path to lock in its
behavior until Phase B (final removal) ships. Each call emits a
``DeprecationWarning``, which pytest captures but does not fail on.

Locks in the V4-calibrated bandwidth efficiency curve (issue #74). The
pre-#74 implementation returned a constant 0.5 for CPU regardless of
working-set size, which was 1.8x pessimistic for large GEMM-style
streaming workloads (B=1 with large IN/OUT linear shapes). After
calibration against V4 baseline medians:

  total_bytes < 1M    -> 0.50 (legacy; matches old behavior; small ops were already passing)
  1M - 10M            -> 0.50 -> 0.75 (cache-resident streaming kicks in)
  > 10M               -> 0.85 plateau (sequential GEMM saturates BW)

These tests are pure unit tests on the public-but-private helper. They
guard against a future revert to the pre-#74 constant.
"""

import pytest

from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from graphs.hardware.resource_model import Precision


@pytest.fixture(scope="module")
def i7_analyzer():
    hw = create_i7_12700k_mapper().resource_model
    return RooflineAnalyzer(hw, precision=Precision.FP32)


@pytest.fixture(scope="module")
def h100_analyzer():
    hw = create_h100_sxm5_80gb_mapper().resource_model
    return RooflineAnalyzer(hw, precision=Precision.FP16)


def _sg_with_bytes(total_bytes: int) -> SubgraphDescriptor:
    """Hand-build a SubgraphDescriptor with a specific working-set size.

    The bw_efficiency function only reads input + output + weight bytes;
    the precise distribution doesn't matter."""
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["op"],
        node_names=["op"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="linear",
        total_flops=1,
        total_macs=0,
        total_input_bytes=total_bytes // 4,
        total_output_bytes=total_bytes // 4,
        total_weight_bytes=total_bytes // 2,
    )


# ---------------------------------------------------------------------------
# CPU curve anchor points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_bytes,expected,label",
    [
        # < 1M: legacy 0.5 (preserves pre-#74 behavior for small WS)
        (1_000, 0.50, "1KB"),
        (100_000, 0.50, "100KB"),
        (999_999, 0.50, "just under 1M"),
        # 1M -> 10M: ramp from 0.5 to 0.75 in log space (exclusive at 10M)
        (1_000_000, 0.50, "1M boundary lower"),
        (9_999_999, 0.75, "just under 10M (top of ramp)"),
        # >= 10M: plateau at 0.85 (the post-#74 fix for the over-prediction cohort)
        (10_000_000, 0.85, "10M plateau start"),
        (100_000_000, 0.85, "100M plateau"),
        (1_000_000_000, 0.85, "1G plateau"),
    ],
)
def test_cpu_bw_efficiency_curve_anchors(i7_analyzer, total_bytes, expected, label):
    """Anchor points of the V4-calibrated CPU bw_efficiency curve."""
    sg = _sg_with_bytes(total_bytes)
    actual = i7_analyzer._get_bandwidth_efficiency_scale(sg)
    assert actual == pytest.approx(
        expected, rel=1e-3
    ), f"{label} (bytes={total_bytes:,}): expected {expected}, got {actual}"


def test_cpu_bw_efficiency_is_non_decreasing(i7_analyzer):
    """The curve must be non-decreasing in working-set size -- larger
    transfers should never get a LOWER bandwidth efficiency than smaller
    ones (the post-#74 curve has a flat-then-ramp-then-plateau shape)."""
    bytes_grid = [1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 1e9]
    scales = []
    for b in bytes_grid:
        sg = _sg_with_bytes(int(b))
        scales.append(i7_analyzer._get_bandwidth_efficiency_scale(sg))
    for i in range(1, len(scales)):
        assert scales[i] >= scales[i - 1] - 1e-9, (
            f"non-monotonic at bytes={bytes_grid[i]:.0e}: "
            f"scale {scales[i]} < {scales[i-1]} (prev)"
        )


def test_cpu_bw_efficiency_does_not_regress_below_pre_74(i7_analyzer):
    """The pre-#74 constant was 0.5. The post-#74 curve must never drop
    BELOW 0.5 (that would regress the small-WS shapes that were already
    passing). Larger WS may go above; that's the whole point of the fix."""
    bytes_grid = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    for b in bytes_grid:
        sg = _sg_with_bytes(int(b))
        scale = i7_analyzer._get_bandwidth_efficiency_scale(sg)
        assert (
            scale >= 0.50 - 1e-9
        ), f"bytes={b:.0e}: scale {scale} dropped below pre-#74 floor of 0.50"


def test_cpu_bw_efficiency_at_large_ws_is_significantly_above_pre_74(i7_analyzer):
    """The whole point of #74: large WS shapes (the over-prediction
    cohort) should now use a significantly higher bw_efficiency than
    the pre-#74 constant 0.5. At least 1.5x higher at 100M bytes."""
    sg = _sg_with_bytes(100_000_000)
    scale = i7_analyzer._get_bandwidth_efficiency_scale(sg)
    assert scale >= 0.50 * 1.5, (
        f"100M bytes scale {scale} not enough higher than pre-#74 0.50; "
        f"the #74 fix isn't actually improving large-WS predictions."
    )


# ---------------------------------------------------------------------------
# GPU branch (untouched by #74) -- negative control
# ---------------------------------------------------------------------------


def test_gpu_bw_efficiency_branch_is_unchanged(h100_analyzer):
    """The #74 fix is CPU-specific. GPU branch must still use its own
    size-based curve (0.3 -> 0.7 across the size range)."""
    # GPU branch returns 0.3 for tiny ops, ramping to 0.7 for large
    sg_tiny = _sg_with_bytes(1_000)
    sg_large = _sg_with_bytes(100_000_000)
    tiny_scale = h100_analyzer._get_bandwidth_efficiency_scale(sg_tiny)
    large_scale = h100_analyzer._get_bandwidth_efficiency_scale(sg_large)
    # Sanity: GPU still has its own curve
    assert 0.0 < tiny_scale <= 0.5
    assert 0.4 <= large_scale <= 0.9
