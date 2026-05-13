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


def test_kpu_dram_traffic_outer_tiled_amortizes_correctly(kpu_t64_resource_model):
    """When weights exceed the on-chip budget, outer-tiled execution loads
    *different* weight slabs each pass (each weight byte loaded exactly
    once total), and the *activation* stream is what cycles through every
    slab. The total DRAM traffic is therefore
    ``activation_bytes * outer_loads + weight_bytes``, NOT
    ``weight_bytes * outer_loads + activation_bytes``.
    """
    analyzer = RooflineAnalyzer(kpu_t64_resource_model, precision=Precision.INT8)
    # 50 MB weights vs ~17 MB budget -> ceil(50/17) = 3 outer slabs
    weight_bytes = 50 * 1024 * 1024
    activation_bytes = 2_000_000
    sg = _make_subgraph(input_b=1_000_000, output_b=1_000_000, weight_b=weight_bytes)
    on_chip = 64 * 256 * 1024 + 4 * 1024 * 1024
    weight_budget = int(on_chip * 0.8)
    expected_outer = math.ceil(weight_bytes / weight_budget)
    expected = activation_bytes * expected_outer + weight_bytes
    assert analyzer._dram_traffic_bytes(sg) == expected
    assert expected_outer >= 2  # sanity


def test_kpu_dram_traffic_handles_zero_weights(kpu_t64_resource_model):
    """Activation-only subgraphs (e.g., elementwise ops) get pure activation
    traffic with no special-case behavior."""
    analyzer = RooflineAnalyzer(kpu_t64_resource_model, precision=Precision.INT8)
    sg = _make_subgraph(input_b=500_000, output_b=500_000, weight_b=0)
    assert analyzer._dram_traffic_bytes(sg) == 1_000_000


# ---------------------------------------------------------------------------
# GPU keeps streaming model (no aggregate weight pool across SMs)
# ---------------------------------------------------------------------------


def test_gpu_dram_traffic_keeps_naive_accounting():
    """GPU retains the conservative input+output+weight accounting.
    Per-SM shared memory doesn't aggregate into a coherent weight pool,
    and the per-launch L2 set-aside on Hopper+ isn't modeled here yet --
    streaming is the right default until there's a per-mapper opt-in."""
    from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
    rm = create_h100_sxm5_80gb_mapper().resource_model
    analyzer = RooflineAnalyzer(rm, precision=Precision.FP16)
    sg = _make_subgraph(input_b=1_000_000, output_b=1_000_000, weight_b=4 * 1024 * 1024)
    expected_naive = 1_000_000 + 1_000_000 + 4 * 1024 * 1024
    assert analyzer._dram_traffic_bytes(sg) == expected_naive


# ---------------------------------------------------------------------------
# CPU: weight residency in LLC (steady-state inference). Same architectural
# story as KPU stationarity but via L3 cache instead of explicit dataflow.
# ---------------------------------------------------------------------------


def test_cpu_dram_traffic_excludes_weights_when_they_fit_in_llc():
    """CPUs with sufficient L3 capture weight reuse in steady state.

    For a workload whose weights fit in LLC, the cold-start DRAM read
    amortizes away over many inferences and only activation traffic
    crosses DRAM steady-state. ``l2_cache_total`` carries the LLC value
    by codebase convention (see CPU mapper comments).
    """
    from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
    rm = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(rm, precision=Precision.FP16)
    # i7-12700K has 25 MB L3; an 8 MB matrix fits with room to spare.
    weight_bytes = 8 * 1024 * 1024
    sg = _make_subgraph(input_b=4096, output_b=4096, weight_b=weight_bytes)
    bytes_ = analyzer._dram_traffic_bytes(sg)
    # input + output only; weights stay resident in LLC.
    assert bytes_ == 4096 + 4096


def test_cpu_dram_traffic_streams_when_weights_exceed_llc():
    """When weights exceed LLC, the outer-tile model applies the same
    way as for KPU: ceil(weight/budget) passes, each loading a fresh
    slab; total weight bytes loaded == nominal weight size."""
    from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
    rm = create_i7_12700k_mapper().resource_model
    analyzer = RooflineAnalyzer(rm, precision=Precision.FP16)
    # 100 MB weights vs 25 MB L3 -> ceil(100/(25*0.8)) = 5 outer slabs.
    weight_bytes = 100 * 1024 * 1024
    activation_bytes = 4096 + 4096
    sg = _make_subgraph(input_b=4096, output_b=4096, weight_b=weight_bytes)
    weight_budget = int(rm.l2_cache_total * 0.8)
    expected_outer = math.ceil(weight_bytes / weight_budget)
    expected = activation_bytes * expected_outer + weight_bytes
    assert analyzer._dram_traffic_bytes(sg) == expected
    assert expected_outer >= 2  # sanity
