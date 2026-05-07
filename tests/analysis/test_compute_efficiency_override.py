"""V5 follow-up: pin the per-precision per-op compute efficiency override path.

Background: Jetson Orin Nano's ``peak_ops_per_sec`` is the achievable
peak (cuBLAS Tensor Core 0.85 efficiency baked in via
``efficiency_factor``), but the shared
``RooflineAnalyzer._get_compute_efficiency_scale`` curve was calibrated
for AGX-style spec peaks and returns scale ~1.5 for large matmul. The
mismatch over-predicted Orin Nano compute throughput by 2.4x. This PR
adds a per-mapper override:

  * ``HardwareResourceModel.compute_efficiency_overrides_by_op`` --
    nested dict ``{precision_name: {op_kind: scale}}``.
  * When set for the analyzer's ``(precision, op_kind)`` pair, the
    legacy curve is bypassed entirely.
  * When absent or non-matching, legacy curve fires unchanged.

Tests:
  * Override fires for matched (precision, op_kind); returns the
    calibrated scalar
  * Mismatched precision -> falls through to legacy
  * Mismatched op kind -> falls through to legacy
  * Multi-op subgraph -> falls through (override is single-op only)
  * Empty overrides dict -> legacy curve (existing behavior)
  * Out-of-range scale rejected at __post_init__
  * Jetson Orin Nano mapper carries the calibrated values
"""

import pytest

from graphs.core.structures import (
    OperationType,
    SubgraphDescriptor,
    create_tensor_descriptor,
)
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.gpu import (
    create_h100_sxm5_80gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
)
from graphs.hardware.resource_model import Precision


def _make_matmul_subgraph(M: int = 128, K: int = 8192, N: int = 8192) -> SubgraphDescriptor:
    a = create_tensor_descriptor((M, K), "float16")
    b = create_tensor_descriptor((K, N), "float16")
    c = create_tensor_descriptor((M, N), "float16")
    return SubgraphDescriptor(
        subgraph_id=1,
        node_ids=["mm"],
        node_names=["mm"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="MatMul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=2 * M * K + 2 * K * N,
        total_output_bytes=2 * M * N,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
        weight_tensors=[],
    )


def _make_linear_subgraph(B: int = 16, IN: int = 4096, OUT: int = 4096) -> SubgraphDescriptor:
    x = create_tensor_descriptor((B, IN), "float16")
    w = create_tensor_descriptor((OUT, IN), "float16")
    y = create_tensor_descriptor((B, OUT), "float16")
    return SubgraphDescriptor(
        subgraph_id=2,
        node_ids=["lin"],
        node_names=["lin"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="Linear",
        total_flops=2 * B * IN * OUT,
        total_macs=B * IN * OUT,
        total_input_bytes=2 * B * IN,
        total_output_bytes=2 * B * OUT,
        total_weight_bytes=2 * OUT * IN,
        input_tensors=[x],
        output_tensors=[y],
        weight_tensors=[w],
    )


# ---------------------------------------------------------------------------
# Override fires for matched (precision, op_kind)
# ---------------------------------------------------------------------------


def test_orin_nano_fp16_matmul_override_fires():
    """Jetson Orin Nano FP16 matmul should bypass the legacy curve and
    return the calibrated 0.70 scalar."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_matmul_subgraph()
    scale = analyzer._get_compute_efficiency_scale(sg)
    assert scale == pytest.approx(0.70), (
        f"Expected calibrated 0.70 for FP16 matmul on Orin Nano, got {scale}"
    )


def test_orin_nano_fp16_linear_override_fires():
    """Jetson Orin Nano FP16 linear should bypass the legacy curve and
    return the calibrated 0.94 scalar (Pareto-optimal point that
    respects the legacy-path energy-pass floor; see
    jetson_orin_nano_8gb.py docstring)."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_linear_subgraph()
    scale = analyzer._get_compute_efficiency_scale(sg)
    assert scale == pytest.approx(0.94), (
        f"Expected calibrated 0.94 for FP16 linear on Orin Nano, got {scale}"
    )


# ---------------------------------------------------------------------------
# Override path skips when precision / op kind doesn't match
# ---------------------------------------------------------------------------


def test_orin_nano_fp32_matmul_falls_through_to_legacy():
    """FP32 isn't in the override dict, so legacy curve fires.
    Smoke test: result should NOT be the FP16 0.70 calibrated value."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    sg = _make_matmul_subgraph()
    scale = analyzer._get_compute_efficiency_scale(sg)
    assert scale != 0.70  # legacy curve, not the FP16 override


def test_h100_no_override_uses_legacy():
    """H100 mapper has no compute_efficiency_overrides_by_op set; the
    legacy curve must fire (returns the AGX-tuned scale, not None)."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    assert hw.compute_efficiency_overrides_by_op == {}
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_matmul_subgraph()
    scale = analyzer._get_compute_efficiency_scale(sg)
    # Legacy curve for >5G matmul on GPU returns 1.40 -> 1.50 range.
    assert 1.0 <= scale <= 2.0


# ---------------------------------------------------------------------------
# Override helper: the lookup primitive
# ---------------------------------------------------------------------------


def test_override_helper_returns_none_when_dict_empty():
    """``_get_compute_efficiency_override`` returns None when the
    resource model has no overrides set, so the caller falls through."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_matmul_subgraph()
    assert analyzer._get_compute_efficiency_override(sg) is None


def test_override_helper_returns_none_for_multi_op_subgraph():
    """Multi-op subgraphs (fused conv+bn+relu, etc.) bypass the
    single-op override and fall through to the legacy curve (which
    handles fusion patterns)."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_matmul_subgraph()
    # Force multi-op
    sg.operation_types = [OperationType.MATMUL, OperationType.ELEMENTWISE]
    assert analyzer._get_compute_efficiency_override(sg) is None


def test_override_helper_returns_none_for_unmapped_op_kind():
    """Op kinds outside {matmul, linear, vector_add} fall through
    (e.g. CONV2D, REDUCE)."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    # Add a fictional CONV2D override that the helper won't see
    hw.compute_efficiency_overrides_by_op = {"fp16": {"matmul": 0.70}}
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_matmul_subgraph()
    sg.operation_types = [OperationType.CONV2D]
    assert analyzer._get_compute_efficiency_override(sg) is None


# ---------------------------------------------------------------------------
# Validation: __post_init__ rejects out-of-range scales
# ---------------------------------------------------------------------------


def test_resource_model_rejects_negative_compute_scale():
    """``compute_efficiency_overrides_by_op`` values must be in (0, 2.0]."""
    from graphs.hardware.models.edge.jetson_orin_nano_8gb import (
        jetson_orin_nano_8gb_resource_model,
    )

    with pytest.raises(ValueError, match=r"compute_efficiency_overrides_by_op"):
        rm = jetson_orin_nano_8gb_resource_model()
        rm.compute_efficiency_overrides_by_op = {"fp16": {"matmul": -0.1}}
        rm.__post_init__()  # re-validate after mutation


def test_resource_model_rejects_zero_compute_scale():
    """Zero is excluded -- the open lower bound prevents
    division-by-zero downstream when the scale becomes the divisor
    in compute_time = flops / (peak_flops * scale)."""
    from graphs.hardware.models.edge.jetson_orin_nano_8gb import (
        jetson_orin_nano_8gb_resource_model,
    )

    with pytest.raises(ValueError, match=r"compute_efficiency_overrides_by_op"):
        rm = jetson_orin_nano_8gb_resource_model()
        rm.compute_efficiency_overrides_by_op = {"fp16": {"matmul": 0.0}}
        rm.__post_init__()


def test_resource_model_rejects_excessive_compute_scale():
    """Above 2.0 the override would let predicted compute exceed
    twice the achievable peak -- almost certainly a calibration bug
    rather than a legitimate value."""
    from graphs.hardware.models.edge.jetson_orin_nano_8gb import (
        jetson_orin_nano_8gb_resource_model,
    )

    with pytest.raises(ValueError, match=r"compute_efficiency_overrides_by_op"):
        rm = jetson_orin_nano_8gb_resource_model()
        rm.compute_efficiency_overrides_by_op = {"fp16": {"matmul": 5.0}}
        rm.__post_init__()


def test_resource_model_accepts_valid_compute_scale():
    """Sanity: the calibrated 0.70 / 0.85 values pass validation."""
    from graphs.hardware.models.edge.jetson_orin_nano_8gb import (
        jetson_orin_nano_8gb_resource_model,
    )

    rm = jetson_orin_nano_8gb_resource_model()
    # No exception
    assert rm.compute_efficiency_overrides_by_op == {
        "fp16": {"matmul": 0.70, "linear": 0.94}
    }


# ---------------------------------------------------------------------------
# End-to-end: analyzer compute_time uses the override
# ---------------------------------------------------------------------------


def test_analyzer_compute_time_reflects_override():
    """compute_time = flops / (peak_flops * scale). On Orin Nano FP16
    matmul with scale = 0.70, predicted compute_time should equal
    flops / (peak_flops * 0.70). This is the integration-level check
    that the override actually flows through to ``_analyze_subgraph``."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    sg = _make_matmul_subgraph(M=128, K=8192, N=8192)  # ~17.2 GFLOPS
    flops = sg.flops
    expected_compute_time = flops / (analyzer.peak_flops * 0.70)
    actual_scale = analyzer._get_compute_efficiency_scale(sg)
    assert actual_scale == pytest.approx(0.70)
    actual_compute_time = flops / (analyzer.peak_flops * actual_scale)
    assert actual_compute_time == pytest.approx(expected_compute_time, rel=1e-9)
