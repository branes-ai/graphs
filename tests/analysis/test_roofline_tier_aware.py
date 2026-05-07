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
def test_default_uses_tier_aware_memory_path(shape):
    """V5-3b flag-flip: ``use_tier_aware_memory`` defaults to True now
    that the V5 plan's exit criterion is met (matmul + linear V4 floors
    equal-or-improve on i7). Constructing without the kwarg should
    produce the SAME predictions as constructing with True, and a
    DIFFERENT prediction than constructing with explicit False (the
    opt-out scalar path).

    Pre-flip this test asserted default == explicit-False. The flip
    inverts the relationship -- default now matches explicit-True."""
    M, K, N = shape
    sg = _matmul_subgraph(M, K, N)
    hw = create_i7_12700k_mapper().resource_model

    analyzer_default = RooflineAnalyzer(hw, precision=Precision.FP32)
    analyzer_explicit_on = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )
    analyzer_explicit_off = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )

    a = analyzer_default._analyze_subgraph(sg)
    b = analyzer_explicit_on._analyze_subgraph(sg)
    c = analyzer_explicit_off._analyze_subgraph(sg)
    # Default == explicit ON: same prediction
    assert a.memory_time == b.memory_time
    # Default != explicit OFF: tier-aware path produces different value
    # than the scalar bw_efficiency_scale path (proved on these matmul
    # shapes which all bind a non-DRAM tier under tier-aware).
    assert a.memory_time != c.memory_time


# ---------------------------------------------------------------------------
# Opt-in path: tier-aware memory_time on i7 (multi-tier hierarchy)
# ---------------------------------------------------------------------------


def test_tier_aware_routes_matmul_through_tier_picker_on_i7():
    """A single-op MATMUL on i7-12700K (L1 + L2 + L3 + DRAM
    post-V5 follow-ups) should produce a different memory_time
    under tier-aware vs the explicit-off scalar path. Sanity-check
    that the tier-aware path is non-zero and not pathologically off.

    Pre-flip the test compared default (off) vs explicit-on; post-flip
    the comparison is between explicit-on (= default) and
    explicit-off (the scalar opt-out)."""
    sg = _matmul_subgraph(1024, 1024, 1024)
    hw = create_i7_12700k_mapper().resource_model
    assert len(hw.memory_hierarchy) >= 2  # i7 must be multi-tier

    analyzer_scalar = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )
    analyzer_tier = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )

    scalar = analyzer_scalar._analyze_subgraph(sg)
    tier = analyzer_tier._analyze_subgraph(sg)

    # Both must produce a positive memory_time
    assert scalar.memory_time > 0
    assert tier.memory_time > 0
    # The tier-aware path differs from the scalar path (otherwise
    # we'd silently collapse to scalar behavior).
    assert tier.memory_time != scalar.memory_time


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
    """Decline cases must leave memory_explanation == None so callers
    can render 'no tier info' without surprise. Two ways to decline:
    (a) explicitly opt OUT (use_tier_aware_memory=False), or (b) opt
    in but the subgraph is ineligible (fused, CONV2D, 3D matmul,
    unknown dtype)."""
    hw = create_i7_12700k_mapper().resource_model

    # Explicit opt-out (the new way to skip the tier path)
    desc1 = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )._analyze_subgraph(_matmul_subgraph(1024, 1024, 1024))
    assert desc1.memory_explanation is None

    # Opt-in but ineligible (fused subgraph) -- default is now ON,
    # so omitting the kwarg gives the same behavior as explicit True
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


# ---------------------------------------------------------------------------
# Boundary-cliff OVERLAP model (V5 follow-up: resolves N=4M floor failure)
# ---------------------------------------------------------------------------


def test_vector_add_dram_binding_uses_overlap_model_at_boundary():
    """V5 follow-up: when vector_add binds DRAM with operand size
    comparable to the outermost cache (boundary regime), use the
    OVERLAP physics:
      memory_time = max(cache_fill_time, dram_stream_time)
    instead of pure DRAM streaming. The boundary case (N=4M on
    i7, WS=50 MB just past the 25 MB LLC) showed +140% over-
    prediction with binary tier picking; OVERLAP brings it to
    +15% (passes 25% DRAM band)."""
    N = 4 * 1024 * 1024
    sg = _vector_add_subgraph(N)
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)

    bytes_loaded = 3 * N * 4  # 50 MB
    # Pure DRAM (binary, what we'd get without OVERLAP):
    dram_eff_bw = next(
        t.effective_bandwidth_bps for t in hw.memory_hierarchy if t.name == "DRAM"
    )
    pure_dram_time = bytes_loaded / dram_eff_bw

    # OVERLAP-corrected memory_time must be SHORTER than pure DRAM
    # (cache hits help) but LONGER than zero. The exact value depends
    # on L3 capacity + BW; we assert the corridor.
    assert desc.memory_time < pure_dram_time, (
        f"OVERLAP didn't kick in: memory_time={desc.memory_time*1e6:.1f}us "
        f">= pure_dram={pure_dram_time*1e6:.1f}us. The L3 cache fill "
        f"component should be reducing the prediction."
    )
    # Sanity floor: prediction can't be smaller than the DRAM stream
    # for the overflow bytes alone (50 MB - 25 MB = 25 MB at DRAM BW).
    overflow_dram_time = (bytes_loaded - 25 * 1024 * 1024) / dram_eff_bw
    assert desc.memory_time >= overflow_dram_time * 0.95, (
        f"OVERLAP under-counted DRAM stream time: got "
        f"{desc.memory_time*1e6:.1f}us vs floor "
        f"{overflow_dram_time*1e6:.1f}us"
    )


def test_vector_add_deeply_dram_bound_falls_back_to_pure_dram():
    """For shapes vastly past LLC (e.g., N=67M, WS=768 MB on i7's
    25 MB LLC), the cache_fill_time component is negligible and
    OVERLAP collapses to pure DRAM streaming. The prediction
    should be dominated by dram_stream_time."""
    N = 67 * 1024 * 1024
    sg = _vector_add_subgraph(N)
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)

    bytes_loaded = 3 * N * 4
    dram_eff_bw = next(
        t.effective_bandwidth_bps for t in hw.memory_hierarchy if t.name == "DRAM"
    )
    pure_dram_time = bytes_loaded / dram_eff_bw

    # OVERLAP and pure-DRAM should match within 5% for deeply
    # DRAM-bound shapes (the L3 fill time of ~149us is negligible
    # vs the multi-millisecond DRAM stream).
    rel = abs(desc.memory_time - pure_dram_time) / pure_dram_time
    assert rel < 0.05, (
        f"OVERLAP perturbed deeply-DRAM-bound prediction by "
        f"{rel*100:.0f}% (threshold 5%); something's wrong with "
        f"the cache_fill_time vs dram_stream_time max() collapse."
    )


def test_overlap_does_not_apply_to_matmul_dram_binding():
    """The OVERLAP scope is vector_add only. Matmul / linear use
    the per-op reuse model's bytes_loaded which already encodes
    tile-streaming reload counts; OVERLAP physics (cache fill +
    concurrent DRAM stream) doesn't apply there. This test pins
    that scope so matmul predictions don't inadvertently get
    OVERLAP-discounted."""
    sg = _matmul_subgraph(4096, 4096, 4096)  # 192 MB operand, DRAM-bound
    hw = create_i7_12700k_mapper().resource_model
    desc = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )._analyze_subgraph(sg)

    # bytes_loaded for matmul includes tile-streaming reloads; the
    # raw division by binding-tier BW should give the same answer
    # whether OVERLAP fires or not (because OVERLAP doesn't fire
    # for matmul).
    me = desc.memory_explanation
    assert me is not None
    assert me.binding_tier_name == "DRAM"
    pure_dram_time = me.bytes_loaded / me.effective_bandwidth_bps
    # memory_time must equal pure DRAM for matmul (no OVERLAP)
    assert desc.memory_time == pytest.approx(pure_dram_time, rel=1e-9), (
        f"matmul DRAM-binding got OVERLAP-modified memory_time "
        f"({desc.memory_time*1e6:.1f}us vs pure-DRAM "
        f"{pure_dram_time*1e6:.1f}us); the scope predicate is wrong."
    )


def test_per_op_calibration_lookup_falls_back_correctly():
    """V5-3b flag-flip prerequisite: the per-op effective BW lookup
    falls back per-op -> per-tier -> 1.0. With i7's calibration
    (matmul/linear @ DRAM = 0.85, vector_add @ DRAM defaults to
    per-tier 0.47), matmul + vector_add at the same DRAM-bound shape
    should produce different memory_time predictions."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer_mm = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=True
    )
    # Matmul DRAM -> uses per-op 0.85 -> peak 75 * 0.85 = 63.75 GB/s
    matmul_dram = analyzer_mm._per_op_effective_bw(
        next(t for t in hw.memory_hierarchy if t.name == "DRAM"),
        "matmul",
    )
    assert matmul_dram == pytest.approx(75e9 * 0.85)

    # Vector_add DRAM -> falls back to per-tier 0.47 -> 35.25 GB/s
    vadd_dram = analyzer_mm._per_op_effective_bw(
        next(t for t in hw.memory_hierarchy if t.name == "DRAM"),
        "vector_add",
    )
    assert vadd_dram == pytest.approx(75e9 * 0.47)

    # Unknown op -> falls back to per-tier
    unknown_dram = analyzer_mm._per_op_effective_bw(
        next(t for t in hw.memory_hierarchy if t.name == "DRAM"),
        "conv2d",
    )
    assert unknown_dram == pytest.approx(75e9 * 0.47)


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
