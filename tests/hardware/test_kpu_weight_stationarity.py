"""Regression tests for issue #51.

The KPU mapper used to model weights as re-fetched from DRAM at every
output tile, which made every transformer-class workload look ~4x more
bandwidth-bound than physics allows. The KPU's architectural reason for
existing is weight stationarity: weights live in the aggregate on-chip
tile fabric (L1 across all tiles + shared L2), and prefetch overlaps
compute. These tests lock in:

1. ViT-B/16 INT8 on KPU-T64 reports ~1.8 ms (issue's expected number),
   not the previous ~4 ms.
2. RooflineAnalyzer._dram_traffic_bytes excludes per-subgraph weight
   bytes when they fit in the on-chip budget.
3. RooflineAnalyzer._dram_traffic_bytes amortizes weights over outer
   tiles when they exceed the on-chip budget.
4. KPUMapper aggregate on-chip capacity matches physical specs.
5. Other accelerators are not affected by the KPU-specific stationarity
   model.
"""

import math

import pytest
import torch

from graphs.estimation.roofline import RooflineAnalyzer
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper
from graphs.hardware.resource_model import Precision
from graphs.core.structures import (
    SubgraphDescriptor,
    OperationType,
    ParallelismDescriptor,
    BottleneckType,
)


# ---------------------------------------------------------------------------
# End-to-end: ViT-B/16 INT8 on KPU-T64 hits the issue's target latency
# ---------------------------------------------------------------------------


def test_vit_b16_int8_on_kpu_t64_meets_stationarity_target():
    analyzer = UnifiedAnalyzer()
    result = analyzer.analyze_model(
        model_name='vit_b_16',
        hardware_name='kpu-t64',
        batch_size=1,
        precision=Precision.INT8,
    )

    # Issue #51 target: ~1.8 ms (weight-stationary baseline). Allow a wide
    # 30% band so refinements to other parts of the model don't break this.
    assert 1.0 <= result.total_latency_ms <= 2.5, (
        f"ViT-B/16 INT8 on KPU-T64 reported {result.total_latency_ms:.2f} ms; "
        f"expected ~1.8 ms (issue #51 stationarity target)"
    )


def test_vit_b16_int8_on_kpu_t64_has_compute_bound_ops():
    """Issue #51: 'attention QKV / MLP projections look bandwidth-bound when
    they should be compute-bound'. After stationarity, at least some of these
    must classify as compute-bound."""
    analyzer = UnifiedAnalyzer()
    result = analyzer.analyze_model(
        model_name='vit_b_16',
        hardware_name='kpu-t64',
        batch_size=1,
        precision=Precision.INT8,
    )

    compute_bound = sum(
        1 for lr in result.roofline_report.latencies
        if lr.bottleneck == BottleneckType.COMPUTE_BOUND
    )
    assert compute_bound >= 5, (
        f"Expected >=5 compute-bound ops on ViT-B/16/KPU-T64 after stationarity, "
        f"got {compute_bound}"
    )


# ---------------------------------------------------------------------------
# RooflineAnalyzer._dram_traffic_bytes unit tests
# ---------------------------------------------------------------------------


def _make_subgraph(input_b: int, output_b: int, weight_b: int) -> SubgraphDescriptor:
    return SubgraphDescriptor(
        subgraph_id=1,
        node_ids=['n1'],
        node_names=['n1'],
        operation_types=[OperationType.LINEAR],
        fusion_pattern='Linear',
        total_flops=1_000_000,
        total_macs=500_000,
        total_input_bytes=input_b,
        total_output_bytes=output_b,
        total_weight_bytes=weight_b,
        internal_bytes=0,
        num_operators=1,
        parallelism=ParallelismDescriptor(batch=1, channels=1, spatial=1, total_threads=1),
        depends_on=[],
    )


@pytest.fixture(scope="module")
def kpu_t64_resource_model():
    return create_kpu_t64_mapper().resource_model


def test_kpu_aggregate_on_chip_capacity(kpu_t64_resource_model):
    rm = kpu_t64_resource_model
    expected = rm.compute_units * rm.l1_cache_per_unit + rm.l2_cache_total
    # KPU-T64: 64 * 256KB + 4MB = 20 MB
    assert expected == 64 * 256 * 1024 + 4 * 1024 * 1024


def test_kpu_dram_traffic_excludes_weights_when_they_fit(kpu_t64_resource_model):
    """Per-subgraph weights that fit in the on-chip budget are pre-fetched
    in parallel with compute and don't contribute to per-layer memory floor."""
    analyzer = RooflineAnalyzer(kpu_t64_resource_model, precision=Precision.INT8)
    # 4 MB weights fit in 17 MB budget (= 80% of 20 MB on-chip).
    sg = _make_subgraph(input_b=1_000_000, output_b=1_000_000, weight_b=4 * 1024 * 1024)
    bytes_ = analyzer._dram_traffic_bytes(sg)
    assert bytes_ == 2_000_000  # input + output, no weight contribution


def test_kpu_dram_traffic_includes_weights_when_they_exceed_budget(kpu_t64_resource_model):
    """Per-subgraph weights that exceed the on-chip budget cannot fully
    overlap; must be amortized over outer tiles."""
    analyzer = RooflineAnalyzer(kpu_t64_resource_model, precision=Precision.INT8)
    # 50 MB weights vs ~17 MB budget -> ceil(50/17) = 3 outer loads
    weight_bytes = 50 * 1024 * 1024
    sg = _make_subgraph(input_b=1_000_000, output_b=1_000_000, weight_b=weight_bytes)
    on_chip = 64 * 256 * 1024 + 4 * 1024 * 1024
    weight_budget = int(on_chip * 0.8)
    expected_outer = math.ceil(weight_bytes / weight_budget)
    expected = 2_000_000 + weight_bytes * expected_outer
    assert analyzer._dram_traffic_bytes(sg) == expected
    assert expected_outer >= 2  # sanity


def test_kpu_dram_traffic_handles_zero_weights(kpu_t64_resource_model):
    """Activation-only subgraphs (e.g., elementwise ops) get pure activation
    traffic with no special-case behavior."""
    analyzer = RooflineAnalyzer(kpu_t64_resource_model, precision=Precision.INT8)
    sg = _make_subgraph(input_b=500_000, output_b=500_000, weight_b=0)
    assert analyzer._dram_traffic_bytes(sg) == 1_000_000


# ---------------------------------------------------------------------------
# Non-KPU hardware is unaffected
# ---------------------------------------------------------------------------


def test_gpu_dram_traffic_keeps_naive_accounting():
    """GPU/CPU/etc. retain the conservative input+output+weight accounting.
    Issue #51 stationarity is a KPU-specific architectural model."""
    from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
    rm = create_h100_sxm5_80gb_mapper().resource_model
    analyzer = RooflineAnalyzer(rm, precision=Precision.FP16)
    sg = _make_subgraph(input_b=1_000_000, output_b=1_000_000, weight_b=4 * 1024 * 1024)
    expected_naive = 1_000_000 + 1_000_000 + 4 * 1024 * 1024
    assert analyzer._dram_traffic_bytes(sg) == expected_naive
