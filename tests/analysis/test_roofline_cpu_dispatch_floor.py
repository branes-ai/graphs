"""Regression tests for the CPU dispatch-overhead floor (issue #69).

Real PyTorch / nn.Module forward calls have a ~5us floor on CPU due to
Python dispatch + parameter access + bias-add overhead. The roofline
math doesn't model this, so very small ops (B=1 linear, tiny matmul)
under-predict latency by 50-80%.

The fix in #69 applies a 5us floor in RooflineAnalyzer._analyze_subgraph
*after* the roofline math: ``actual_latency = max(predicted, 5us)`` for
CPU. Implemented as a floor (not additive overhead) so it only kicks in
when the kernel is too small to dominate, avoiding over-correction of
medium ops where the overhead is amortized by real kernel time.

Empirical anchor: V4 baseline measurement of (1,128,128) linear fp32
on i7-12700K is 5.20us -- the wall-clock floor for any forward call.

These tests lock in:
1. CPU-typed hardware: floor is applied; predicted >= 5us
2. GPU-typed hardware: NOT affected (GPU has its own kernel-launch overhead model)
3. Floor only kicks in when the kernel is sub-floor; doesn't add to medium ops
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
        total_flops=2 * 1 * 128 * 128,        # 32,768
        total_macs=1 * 128 * 128,
        total_input_bytes=1 * 128 * 4,         # 512
        total_output_bytes=1 * 128 * 4,        # 512
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
    assert lat.actual_latency >= 5e-6, (
        f"CPU floor not applied: actual_latency={lat.actual_latency*1e6:.2f}us"
    )


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
        node_ids=["x"], node_names=["x"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="linear",
        total_flops=2 * 1 * 64 * 64,   # tiny
        total_macs=1 * 64 * 64,
        total_input_bytes=128, total_output_bytes=128,
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


def test_v4_anchor_shape_predicts_within_band_of_baseline():
    """The V4 baseline shows (1,128,128) linear fp32 on i7 measured at
    5.20us (the wall-clock floor). With the dispatch floor the prediction
    should land within the LAUNCH 30% tolerance band of that measurement."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _tiny_linear_subgraph()
    lat = analyzer._analyze_subgraph(sg)
    measured_s = 5.20e-6
    rel_err = abs(lat.actual_latency - measured_s) / measured_s
    assert rel_err <= 0.30, (
        f"Predicted {lat.actual_latency*1e6:.2f}us is {rel_err*100:.0f}% "
        f"off from baseline 5.20us (LAUNCH band: 30%)"
    )
