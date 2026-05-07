"""V5-6 Phase A: pin the deprecation behavior of
``RooflineAnalyzer._get_bandwidth_efficiency_scale``.

The function is now a fallback-only path -- the V5-3b tier-aware
roofline (default since #113) supersedes it. This file pins:

1. A ``DeprecationWarning`` fires on first invocation per analyzer
   instance.
2. The warning fires only ONCE per instance (not per call).
3. The warning fires when callers OPT OUT via
   ``use_tier_aware_memory=False``.
4. The warning DOESN'T fire when the tier-aware path handles a
   subgraph cleanly.
5. The warning fires when the tier-aware path declines (e.g.
   multi-op subgraph, unsupported op type).

See ``docs/v5/bw-efficiency-scale-retirement-status.md`` for the
per-mapper coverage matrix gating Phase B (final removal).
"""

import warnings

import pytest

from graphs.core.structures import (
    OperationType,
    SubgraphDescriptor,
    create_tensor_descriptor,
)
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.resource_model import Precision


def _matmul_subgraph(M, K, N, dtype="float32") -> SubgraphDescriptor:
    """Tier-aware-eligible single-op MATMUL subgraph."""
    a = create_tensor_descriptor((M, K), dtype)
    b = create_tensor_descriptor((K, N), dtype)
    c = create_tensor_descriptor((M, N), dtype)
    return SubgraphDescriptor(
        subgraph_id=1,
        node_ids=["mm0"],
        node_names=["mm0"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="Matmul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=(M * K + K * N) * 4,
        total_output_bytes=M * N * 4,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
        weight_tensors=[],
    )


def _fused_subgraph(M, K, N) -> SubgraphDescriptor:
    """Multi-op fused subgraph -- tier-aware path declines, falls
    back to the deprecated bw_efficiency_scale path."""
    sg = _matmul_subgraph(M, K, N)
    sg.operation_types = [OperationType.MATMUL, OperationType.RELU]
    sg.num_operators = 2
    return sg


def test_deprecation_warning_fires_when_caller_opts_out():
    """``use_tier_aware_memory=False`` is now an explicit opt-out;
    the legacy bw_efficiency_scale path runs and emits the
    DeprecationWarning."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )
    with pytest.warns(DeprecationWarning, match="V5-6 Phase A"):
        analyzer._analyze_subgraph(_matmul_subgraph(64, 64, 64))


def test_deprecation_warning_fires_when_tier_aware_declines():
    """Even with the tier-aware path enabled (default), if the
    subgraph isn't eligible (multi-op, unsupported op, etc.), the
    fallback fires and the warning surfaces."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    with pytest.warns(DeprecationWarning, match="V5-6 Phase A"):
        analyzer._analyze_subgraph(_fused_subgraph(64, 64, 64))


def test_no_deprecation_warning_when_tier_aware_handles_subgraph():
    """The tier-aware path handles a clean single-op MATMUL on
    calibrated hardware; the legacy fallback never runs and no
    warning fires."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Would raise if any DeprecationWarning fired
        desc = analyzer._analyze_subgraph(_matmul_subgraph(1024, 1024, 1024))
    # Sanity: tier-aware actually handled it
    assert desc.memory_explanation is not None


def test_deprecation_warning_fires_only_once_per_instance():
    """The warning is gated on a per-instance flag so subsequent
    invocations of the legacy path don't re-emit. This keeps test /
    CI output readable when many subgraphs hit the fallback."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(
        hw, precision=Precision.FP32, use_tier_aware_memory=False
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        # Multiple calls -> only one warning total
        analyzer._analyze_subgraph(_matmul_subgraph(64, 64, 64))
        analyzer._analyze_subgraph(_matmul_subgraph(128, 128, 128))
        analyzer._analyze_subgraph(_matmul_subgraph(256, 256, 256))

    deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
    assert len(deprecations) == 1, (
        f"Expected exactly 1 DeprecationWarning per analyzer instance, "
        f"got {len(deprecations)}"
    )


def test_each_new_analyzer_instance_warns_fresh():
    """The per-instance flag means a new RooflineAnalyzer gets a
    fresh chance to surface the warning. Useful when a workflow
    constructs many analyzers, some of which trip the fallback."""
    hw = create_i7_12700k_mapper().resource_model

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        for _ in range(3):
            analyzer = RooflineAnalyzer(
                hw, precision=Precision.FP32, use_tier_aware_memory=False
            )
            analyzer._analyze_subgraph(_matmul_subgraph(64, 64, 64))

    deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
    assert len(deprecations) == 3, (
        f"Expected 3 DeprecationWarnings (one per fresh instance), "
        f"got {len(deprecations)}"
    )
