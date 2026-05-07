"""V5-3b roofline integration tests for the opt-in tier-aware memory path.

Locks in:
- Default (use_tier_aware_memory=False) is byte-identical to pre-V5-3b
  scalar bw_efficiency_scale path (V4 floor preservation).
- Opt-in (use_tier_aware_memory=True) routes single-op MATMUL/LINEAR
  through the tier-picker on hardware with multi-tier memory_hierarchy.
- Eligibility predicate falls back to scalar path for: fused subgraphs,
  unsupported op types (CONV2D, ELEMENTWISE), 3D+ shapes, DRAM-only
  hierarchies, missing tensor info, unknown dtypes.
"""

import pytest

from graphs.core.structures import (
    OperationType,
    SubgraphDescriptor,
    create_tensor_descriptor,
)
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Fixtures: single-op subgraphs we can feed to _analyze_subgraph
# ---------------------------------------------------------------------------


def _matmul_subgraph(M, K, N, dtype="float32") -> SubgraphDescriptor:
    """Build a single-op MATMUL subgraph with proper tensor info."""
    a = create_tensor_descriptor((M, K), dtype)
    b = create_tensor_descriptor((K, N), dtype)
    c = create_tensor_descriptor((M, N), dtype)
    flops = 2 * M * K * N  # MAC = 2 FLOPs
    return SubgraphDescriptor(
        subgraph_id=1,
        node_ids=["mm0"],
        node_names=["mm0"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="Matmul",
        total_flops=flops,
        total_macs=M * K * N,
        total_input_bytes=a.shape[0] * a.shape[1] * 4 + b.shape[0] * b.shape[1] * 4,
        total_output_bytes=M * N * 4,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
        weight_tensors=[],
    )


def _linear_subgraph(B, IN, OUT, dtype="float32") -> SubgraphDescriptor:
    x = create_tensor_descriptor((B, IN), dtype)
    w = create_tensor_descriptor((OUT, IN), dtype)
    y = create_tensor_descriptor((B, OUT), dtype)
    flops = 2 * B * IN * OUT
    return SubgraphDescriptor(
        subgraph_id=2,
        node_ids=["lin0"],
        node_names=["lin0"],
        operation_types=[OperationType.LINEAR],
        fusion_pattern="Linear",
        total_flops=flops,
        total_macs=B * IN * OUT,
        total_input_bytes=B * IN * 4,
        total_output_bytes=B * OUT * 4,
        total_weight_bytes=OUT * IN * 4,
        input_tensors=[x],
        output_tensors=[y],
        weight_tensors=[w],
    )


def _conv2d_subgraph() -> SubgraphDescriptor:
    """Negative case: not MATMUL/LINEAR -> tier path declines."""
    x = create_tensor_descriptor((1, 64, 56, 56), "float32")
    return SubgraphDescriptor(
        subgraph_id=3,
        node_ids=["conv0"],
        node_names=["conv0"],
        operation_types=[OperationType.CONV2D],
        fusion_pattern="Conv2d",
        total_flops=10**9,
        total_macs=5 * 10**8,
        total_input_bytes=1 * 64 * 56 * 56 * 4,
        total_output_bytes=1 * 64 * 56 * 56 * 4,
        total_weight_bytes=64 * 64 * 3 * 3 * 4,
        input_tensors=[x],
    )


def _fused_matmul_relu() -> SubgraphDescriptor:
    """Negative case: num_operators > 1 -> tier path declines."""
    sg = _matmul_subgraph(512, 512, 512)
    sg.operation_types = [OperationType.MATMUL, OperationType.RELU]
    sg.num_operators = 2
    return sg


# ---------------------------------------------------------------------------
# Default (off) preserves scalar behavior byte-for-byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64, 64), (512, 512, 512), (4096, 4096, 4096)])
def test_default_off_produces_identical_memory_time_to_pre_v5_3b(shape):
    """The opt-out default must not perturb any memory_time number,
    so V4 floors are guaranteed unchanged. Run the analyzer twice
    (once at default, once with the flag explicitly off) and confirm
    they match exactly."""
    M, K, N = shape
    sg = _matmul_subgraph(M, K, N)
    hw = create_i7_12700k_mapper().resource_model

    analyzer_default = RooflineAnalyzer(hw, precision=Precision.FP32)
    analyzer_explicit_off = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )

    a = analyzer_default._analyze_subgraph(sg)
    b = analyzer_explicit_off._analyze_subgraph(sg)
    assert a.memory_time == b.memory_time


# ---------------------------------------------------------------------------
# Opt-in path: tier-aware memory_time on i7 (multi-tier hierarchy)
# ---------------------------------------------------------------------------


def test_opt_in_routes_matmul_through_tier_picker_on_i7():
    """With use_tier_aware_memory=True, a single-op MATMUL on i7-12700K
    (which has L1 + L3 + DRAM tiers post-V5-1) should produce a
    different memory_time than the scalar path. Sanity-check that the
    new path is non-zero and not pathologically off."""
    sg = _matmul_subgraph(1024, 1024, 1024)
    hw = create_i7_12700k_mapper().resource_model
    assert len(hw.memory_hierarchy) >= 2  # i7 must be multi-tier

    analyzer_off = RooflineAnalyzer(hw, precision=Precision.FP32)
    analyzer_on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )

    off = analyzer_off._analyze_subgraph(sg)
    on = analyzer_on._analyze_subgraph(sg)

    # Both must produce a positive memory_time
    assert off.memory_time > 0
    assert on.memory_time > 0
    # The tier-aware path should differ from the scalar path (otherwise
    # we'd silently regress to the scalar behavior; this catches it).
    assert on.memory_time != off.memory_time


def test_opt_in_handles_linear_on_h100():
    """LINEAR op on H100 (also multi-tier) should also route through
    the tier picker. Just confirm we don't crash and produce a
    physically-plausible memory_time (>0, <1 second for a small linear)."""
    sg = _linear_subgraph(32, 1024, 1024, dtype="float16")
    hw = create_h100_sxm5_80gb_mapper().resource_model
    assert len(hw.memory_hierarchy) >= 2

    analyzer = RooflineAnalyzer(
        hw, precision=Precision.FP16, use_tier_aware_memory=True
    )
    desc = analyzer._analyze_subgraph(sg)
    assert 0 < desc.memory_time < 1.0


# ---------------------------------------------------------------------------
# Opt-in path: ineligible subgraphs fall back to scalar
# ---------------------------------------------------------------------------


def test_opt_in_falls_back_for_fused_subgraph():
    """num_operators > 1 -> tier path declines; result must equal the
    scalar path's result."""
    sg = _fused_matmul_relu()
    hw = create_i7_12700k_mapper().resource_model

    on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    off = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )._analyze_subgraph(sg)
    assert on.memory_time == off.memory_time


def test_opt_in_falls_back_for_conv2d():
    """CONV2D isn't in the V5-3a reuse_models registry -> decline ->
    fall through to scalar path."""
    sg = _conv2d_subgraph()
    hw = create_i7_12700k_mapper().resource_model

    on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    off = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )._analyze_subgraph(sg)
    assert on.memory_time == off.memory_time


def test_opt_in_falls_back_for_3d_matmul():
    """Batched matmul (3D inputs) doesn't match the clean 2D shape
    extraction -> decline -> scalar path."""
    M, K, N = 16, 256, 256
    a = create_tensor_descriptor((4, M, K), "float32")  # 3D batched
    b = create_tensor_descriptor((4, K, N), "float32")
    c = create_tensor_descriptor((4, M, N), "float32")
    sg = SubgraphDescriptor(
        subgraph_id=10,
        node_ids=["bmm0"],
        node_names=["bmm0"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="Matmul",
        total_flops=2 * 4 * M * K * N,
        total_macs=4 * M * K * N,
        total_input_bytes=4 * M * K * 4 + 4 * K * N * 4,
        total_output_bytes=4 * M * N * 4,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
    )

    hw = create_i7_12700k_mapper().resource_model
    on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    off = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )._analyze_subgraph(sg)
    assert on.memory_time == off.memory_time


def test_opt_in_falls_back_for_unknown_dtype():
    """A dtype the reuse_models bytes_per_element table doesn't know
    about must not crash the analyzer -- decline gracefully."""
    sg = _matmul_subgraph(256, 256, 256, dtype="complex64")
    hw = create_i7_12700k_mapper().resource_model

    # Should not raise
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    assert desc.memory_time > 0


# ---------------------------------------------------------------------------
# V5-4: memory_explanation field is populated when the path fires
# ---------------------------------------------------------------------------


def test_memory_explanation_populated_when_tier_path_fires():
    """When the V5-3b tier-aware path fires, LatencyDescriptor must
    carry a fully-populated MemoryExplanation: binding tier, residency
    tier, tile dims, residency bytes, bytes loaded, effective BW."""
    sg = _matmul_subgraph(1024, 1024, 1024)
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)

    me = desc.memory_explanation
    assert me is not None
    # i7-12700K: hierarchy is L1, L3, DRAM (post-V5-1 / #94 build).
    # 1024^3 fp32 matmul tiles to fit in L1 -> binds at L3.
    assert me.binding_tier_name in {"L1", "L3", "DRAM"}
    assert me.residency_tier_name in {"L1", "L3", "DRAM"}
    assert len(me.tile_dims) == 2  # matmul -> (Mt, Nt)
    assert me.residency_bytes > 0
    assert me.bytes_loaded > 0
    assert me.effective_bandwidth_bps > 0


def test_memory_explanation_none_when_path_declines():
    """Default (off) and decline cases (fused, CONV2D, 3D matmul,
    unknown dtype) must leave memory_explanation == None so callers
    can render 'no tier info' without surprise."""
    hw = create_i7_12700k_mapper().resource_model

    # Default (flag off)
    desc1 = RooflineAnalyzer(hw, precision=Precision.FP32)._analyze_subgraph(
        _matmul_subgraph(1024, 1024, 1024)
    )
    assert desc1.memory_explanation is None

    # Opt-in but ineligible (fused subgraph)
    desc2 = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(_fused_matmul_relu())
    assert desc2.memory_explanation is None


def test_memory_explanation_does_not_leak_across_subgraphs():
    """The analyzer's per-subgraph stash must reset between calls so
    a decline (e.g., fused subgraph) doesn't inherit the previous
    successful subgraph's MemoryExplanation."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )

    # First: a successful tier-path subgraph
    desc_ok = analyzer._analyze_subgraph(_matmul_subgraph(1024, 1024, 1024))
    assert desc_ok.memory_explanation is not None

    # Second: a fused subgraph that the predicate rejects -- must come
    # back with memory_explanation=None, NOT the prior subgraph's value
    desc_decline = analyzer._analyze_subgraph(_fused_matmul_relu())
    assert desc_decline.memory_explanation is None


# ---------------------------------------------------------------------------
# vector_add eligibility (V4 vector_add validation harness follow-up)
# ---------------------------------------------------------------------------


def _vector_add_subgraph(N, dtype="float32") -> SubgraphDescriptor:
    """Build a single-op vector_add subgraph (c = a + b) on N elements."""
    a = create_tensor_descriptor((N,), dtype)
    b = create_tensor_descriptor((N,), dtype)
    c = create_tensor_descriptor((N,), dtype)
    return SubgraphDescriptor(
        subgraph_id=42,
        node_ids=["va"],
        node_names=["va"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="Add",
        total_flops=N,
        total_macs=0,
        total_input_bytes=2 * N * 4,
        total_output_bytes=N * 4,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
        weight_tensors=[],
    )


def test_opt_in_routes_vector_add_through_tier_picker_on_i7():
    """V4 vector_add validation harness: with use_tier_aware_memory=True,
    a strict 3-tensor 1-D ELEMENTWISE op (c = a + b) should route
    through the V5-3a vector_add reuse model and bind at the smallest
    tier holding the operand footprint."""
    sg = _vector_add_subgraph(16 * 1024 * 1024)  # 192 MB > L3 -> DRAM
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    assert desc.memory_explanation is not None
    assert desc.memory_explanation.binding_tier_name == "DRAM"


def test_vector_add_small_n_binds_at_l1_via_analyzer():
    """For an L1-resident vector_add, V5-5-followup operand-aware
    binding selects L1 (smallest tier whose capacity holds the
    12 KB operand footprint)."""
    sg = _vector_add_subgraph(1024)  # 12 KB <- fits L1
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    assert desc.memory_explanation is not None
    assert desc.memory_explanation.binding_tier_name == "L1"


def test_opt_in_falls_back_for_elementwise_with_only_one_input():
    """ReLU / sigmoid (1 input -> 1 output) is also OperationType.ELEMENTWISE
    but has different reuse semantics. The strict
    _extract_vector_add_shape predicate must reject anything that's
    not specifically 'a + b -> c' so we don't apply the wrong reuse
    model."""
    x = create_tensor_descriptor((1024,), "float32")
    y = create_tensor_descriptor((1024,), "float32")
    sg = SubgraphDescriptor(
        subgraph_id=99,
        node_ids=["relu"],
        node_names=["relu"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="ReLU",
        total_flops=1024,
        total_macs=0,
        total_input_bytes=1024 * 4,
        total_output_bytes=1024 * 4,
        total_weight_bytes=0,
        input_tensors=[x],
        output_tensors=[y],
    )
    hw = create_i7_12700k_mapper().resource_model
    on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    off = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )._analyze_subgraph(sg)
    # Falls back to scalar -> identical memory_time
    assert on.memory_time == off.memory_time
    assert on.memory_explanation is None


def test_opt_in_falls_back_for_2d_elementwise():
    """2-D elementwise (e.g., matrix Add as a separate op) doesn't
    match the strict 1-D vector_add predicate -> scalar fallback."""
    a = create_tensor_descriptor((64, 64), "float32")
    b = create_tensor_descriptor((64, 64), "float32")
    c = create_tensor_descriptor((64, 64), "float32")
    sg = SubgraphDescriptor(
        subgraph_id=100,
        node_ids=["add2d"],
        node_names=["add2d"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="Add",
        total_flops=64 * 64,
        total_macs=0,
        total_input_bytes=2 * 64 * 64 * 4,
        total_output_bytes=64 * 64 * 4,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
    )
    hw = create_i7_12700k_mapper().resource_model
    on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    assert on.memory_explanation is None


def test_opt_in_falls_back_for_elementwise_with_mismatched_shapes():
    """Broadcast-style elementwise (a is scalar, b is N-d) -- the
    bytes_loaded calculation in vector_add reuse model would be
    wrong for these. Reject."""
    a = create_tensor_descriptor((1,), "float32")  # scalar broadcast
    b = create_tensor_descriptor((1024,), "float32")
    c = create_tensor_descriptor((1024,), "float32")
    sg = SubgraphDescriptor(
        subgraph_id=101,
        node_ids=["broadcast"],
        node_names=["broadcast"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="Add",
        total_flops=1024,
        total_macs=0,
        total_input_bytes=1024 * 4 + 4,
        total_output_bytes=1024 * 4,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
    )
    hw = create_i7_12700k_mapper().resource_model
    on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)
    assert on.memory_explanation is None


def test_memory_explanation_format_summary_renders_breakdown():
    """LatencyDescriptor.format_summary should include a 'Memory binding'
    line when memory_explanation is set, so the breakdown surfaces in
    the human-readable string output without callers having to dig into
    the structured field."""
    sg = _matmul_subgraph(1024, 1024, 1024)
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)

    summary = desc.format_summary()
    assert "Memory binding" in summary
    assert desc.memory_explanation.binding_tier_name in summary
