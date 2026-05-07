"""Regression tests for the op-aware CPU dispatch-overhead floor.

History:
* #69 introduced a single 5 us CPU dispatch floor based on the
  smallest V4 linear measurement (1, 128, 128) fp32 = 5.20 us.
* This file's V5 follow-up split the floor per op kind based on
  the first-principles dispatch decomposition validated against
  i7-12700K V4 baselines.

What "dispatch" means on a CPU
------------------------------

Unlike GPU kernel launch (a discrete cudaLaunchKernel that the
device queues), CPU dispatch is the chain of software work
PyTorch performs before any FLOPs execute: Python frame entry,
ATen dispatcher key resolution, output tensor allocation, kernel
invocation. None of these are physical constraints -- a CPU can
context-switch in nanoseconds, take an ISR, return from an RPC --
but they ARE the unavoidable runtime cost of routing a Python-level
tensor op through the framework. The roofline math ignores all
of this, so we floor the prediction at the empirical
smallest-shape latency.

Per-op decomposition (first-principles, validated on i7 fp32):

  vector_add: ~2 us  (base ATen dispatch only)
  matmul:     ~6 us  (+ BLAS thread launch, stride/layout norm)
  linear:     ~9 us  (+ nn.Module parameter access, bias epilogue)

See the ``_get_cpu_dispatch_floor`` docstring in
``src/graphs/estimation/roofline.py`` for the full component
breakdown.

These tests lock in:
1. The single-floor-everywhere behavior is gone -- each op kind
   gets its own floor.
2. Smallest CPU op of each kind gets its own empirical floor
   (2 / 6 / 9 us).
3. GPU is unaffected (GPU has its own kernel-launch overhead model).
4. Floor only kicks in when the kernel is sub-floor; doesn't add
   to medium ops.
5. Unknown / fused / unset op types fall back to the legacy 5 us
   value so this PR is conservative for code paths it doesn't
   directly cover.
"""

import pytest

from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from graphs.hardware.resource_model import Precision


def _tiny_linear_subgraph() -> SubgraphDescriptor:
    """A 1-op SubgraphDescriptor matching what V4 builds for (1,128,128)
    linear fp32 -- 33K flops, ~66KB working set."""
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["linear"],
        node_names=["linear"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="linear",
        total_flops=2 * 1 * 128 * 128,  # 32,768
        total_macs=1 * 128 * 128,
        total_input_bytes=1 * 128 * 4,  # 512
        total_output_bytes=1 * 128 * 4,  # 512
        total_weight_bytes=(128 * 128 + 128) * 4,  # 66,048
    )


def _medium_linear_subgraph() -> SubgraphDescriptor:
    """(32, 384, 768) linear fp32 -- 18.87M flops, well above the floor."""
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["linear"],
        node_names=["linear"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="linear",
        total_flops=2 * 32 * 384 * 768,
        total_macs=32 * 384 * 768,
        total_input_bytes=32 * 384 * 4,
        total_output_bytes=32 * 768 * 4,
        total_weight_bytes=(384 * 768 + 768) * 4,
    )


# ---------------------------------------------------------------------------
# Floor IS applied for CPU
# ---------------------------------------------------------------------------


def test_cpu_floor_applies_to_tiny_op():
    """A predicted-sub-floor CPU op must be raised to the 5us floor."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _tiny_linear_subgraph()
    lat = analyzer._analyze_subgraph(sg)
    # The floor is 5us; any prediction below it gets raised.
    assert (
        lat.actual_latency >= 5e-6
    ), f"CPU floor not applied: actual_latency={lat.actual_latency*1e6:.2f}us"


def test_cpu_floor_does_not_inflate_medium_op():
    """A predicted-above-floor op must be unchanged by the floor."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _medium_linear_subgraph()
    lat = analyzer._analyze_subgraph(sg)
    # Medium op predicts well above 5us; floor must not move it.
    assert lat.actual_latency > 5e-6  # sanity
    # The floor would only matter if compute_time + memory_time + overhead
    # were < 5us; for this 18.87M flop op they're easily larger.
    no_floor_estimate = max(lat.compute_time, lat.memory_time) + lat.overhead
    assert lat.actual_latency == pytest.approx(no_floor_estimate, rel=1e-9), (
        f"Floor incorrectly inflated medium op: "
        f"actual={lat.actual_latency*1e6:.2f}us, no_floor={no_floor_estimate*1e6:.2f}us"
    )


# ---------------------------------------------------------------------------
# Floor is NOT applied for non-CPU hardware
# ---------------------------------------------------------------------------


def test_gpu_does_not_get_cpu_floor():
    """GPU has its own kernel-launch overhead model; the CPU dispatch
    floor must not bleed into GPU predictions."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    # Build a tiny op
    sg = SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["x"],
        node_names=["x"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="linear",
        total_flops=2 * 1 * 64 * 64,  # tiny
        total_macs=1 * 64 * 64,
        total_input_bytes=128,
        total_output_bytes=128,
        total_weight_bytes=(64 * 64 + 64) * 2,
    )
    lat = analyzer._analyze_subgraph(sg)
    # GPU has its own 5us launch overhead; the result should equal
    # bottleneck + overhead with NO additional CPU floor enforcement.
    no_floor_estimate = max(lat.compute_time, lat.memory_time) + lat.overhead
    assert lat.actual_latency == pytest.approx(no_floor_estimate, rel=1e-9)


# ---------------------------------------------------------------------------
# V4 baseline: (1,128,128) linear is the canonical anchor
# ---------------------------------------------------------------------------


def test_v4_typical_small_linear_predicts_within_band_of_baseline():
    """The V4 i7 linear baseline clusters at ~8.7-9.0 us for 7 of the
    8 smallest shapes (e.g. 1x384x128 = 8.76 us, 1x640x128 = 8.78 us).
    Only the very-smallest 1x128x128 (32K flops) is an outlier at 5.20
    us -- so tiny that BLAS thread-launch overhead is fully amortized.

    With the op-aware linear floor (9 us, the typical-cluster value),
    the prediction lands within the LAUNCH 30% band of any
    typical-cluster shape. The outlier 1x128x128 is over-predicted
    (~73%) but is structurally below the model's resolution; it's a
    documented limitation of the single-floor-per-op approach."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    # Use 1x384x128 -- a typical-cluster shape, not the 1x128x128 outlier
    sg = SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["linear"],
        node_names=["linear"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="linear",
        total_flops=2 * 1 * 384 * 128,
        total_macs=1 * 384 * 128,
        total_input_bytes=1 * 384 * 4,
        total_output_bytes=1 * 128 * 4,
        total_weight_bytes=(384 * 128 + 128) * 4,
    )
    lat = analyzer._analyze_subgraph(sg)
    measured_s = 8.76e-6  # i7 baseline: 1x384x128 fp32
    rel_err = abs(lat.actual_latency - measured_s) / measured_s
    assert rel_err <= 0.30, (
        f"Predicted {lat.actual_latency*1e6:.2f}us is {rel_err*100:.0f}% "
        f"off from typical-cluster baseline 8.76us (LAUNCH band: 30%)"
    )


# ---------------------------------------------------------------------------
# Op-aware floor values
# ---------------------------------------------------------------------------


def _vector_add_subgraph(N: int) -> SubgraphDescriptor:
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["va"],
        node_names=["va"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="vector_add",
        total_flops=N,
        total_macs=0,
        total_input_bytes=2 * N * 4,
        total_output_bytes=N * 4,
        total_weight_bytes=0,
    )


def _matmul_subgraph(M: int, K: int, N: int) -> SubgraphDescriptor:
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["mm"],
        node_names=["mm"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="matmul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=(M * K + K * N) * 4,
        total_output_bytes=M * N * 4,
        total_weight_bytes=0,
    )


def test_vector_add_floor_is_2us():
    """Empirical i7 floor for vector_add at smallest N (256, 1024) is
    1.84-1.97 us. Op-aware floor sets vector_add to 2 us, the
    base-ATen-dispatch component only (no BLAS, no parameter access,
    no bias)."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _vector_add_subgraph(1024)  # 12 KB working set; tiny
    lat = analyzer._analyze_subgraph(sg)
    # Floor binds for this tiny op; the math gives ~ns, floor wins.
    assert lat.actual_latency == pytest.approx(2e-6, rel=1e-9), (
        f"vector_add tiny op should hit 2us floor, got "
        f"{lat.actual_latency*1e6:.2f}us"
    )


def test_matmul_floor_is_6us():
    """Empirical i7 floor for matmul at smallest shapes (e.g. 64x64x96)
    is 6.2-6.8 us. Op-aware floor sets matmul to 6 us, capturing
    base ATen + BLAS thread launch + stride/layout normalization."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _matmul_subgraph(8, 8, 8)  # 512 flops, way below floor
    lat = analyzer._analyze_subgraph(sg)
    assert lat.actual_latency == pytest.approx(6e-6, rel=1e-9), (
        f"matmul tiny op should hit 6us floor, got " f"{lat.actual_latency*1e6:.2f}us"
    )


def test_linear_floor_is_9us():
    """Empirical i7 floor for linear at typical-small shapes (cluster
    of 7 of the 8 smallest) is 8.7-9.0 us. Op-aware floor sets linear
    to 9 us, capturing the matmul stack + parameter access + bias
    epilogue."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _tiny_linear_subgraph()
    lat = analyzer._analyze_subgraph(sg)
    assert lat.actual_latency == pytest.approx(9e-6, rel=1e-9), (
        f"linear tiny op should hit 9us floor, got " f"{lat.actual_latency*1e6:.2f}us"
    )


def test_unknown_op_falls_back_to_default_floor():
    """For op kinds not in the op-aware table (CONV2D, RELU, etc.),
    fall back to the legacy 5 us default. Ensures this PR can't
    silently break floor predictions for op kinds it didn't
    explicitly model."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["c"],
        node_names=["c"],
        operation_types=[OperationType.CONV2D],
        fusion_pattern="conv2d",
        total_flops=100,  # tiny
        total_macs=50,
        total_input_bytes=400,
        total_output_bytes=400,
        total_weight_bytes=0,
    )
    lat = analyzer._analyze_subgraph(sg)
    assert lat.actual_latency == pytest.approx(5e-6, rel=1e-9), (
        f"CONV2D tiny op should fall back to 5us default floor, got "
        f"{lat.actual_latency*1e6:.2f}us"
    )
