"""V5 follow-up: pin the ``kernel_launch_overhead_ns`` plumbing.

Background: the GPUMapper subclass tracks
``KERNEL_LAUNCH_OVERHEAD_EDGE = 80e-6`` for Jetson and
``KERNEL_LAUNCH_OVERHEAD_DATACENTER = 10e-6`` for desktop GPUs, but
``RooflineAnalyzer._estimate_overhead`` was using a hardcoded 5 us
base regardless of the device class. This caused L1-binding shapes
on Jetson to under-predict by 100+ us (predict 5-13 us, measure
113-173 us per arXiv:2508.08430 + V4 baselines).

This PR surfaces ``kernel_launch_overhead_ns`` on
``HardwareResourceModel`` and has GPUMapper write it during
construction, so RooflineAnalyzer can read the same value the
mapper already exposes.

Tests:
* GPUMapper construction sets the field correctly for edge / datacenter
* Explicit override on the resource_model wins over the mapper default
* RooflineAnalyzer's _estimate_overhead uses the new field for GPU
* Field stays None / unused for CPU mappers
"""

import pytest

from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import (
    GPUMapper,
    create_h100_sxm5_80gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
)
from graphs.hardware.resource_model import Precision


def test_jetson_mapper_writes_edge_overhead_to_resource_model():
    """Jetson Orin Nano (edge device) gets the 80 us EDGE constant
    written to ``resource_model.kernel_launch_overhead_ns``."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    assert hw.kernel_launch_overhead_ns == pytest.approx(
        GPUMapper.KERNEL_LAUNCH_OVERHEAD_EDGE * 1e9
    )
    assert hw.kernel_launch_overhead_ns == 80_000.0


def test_h100_mapper_writes_datacenter_overhead_to_resource_model():
    """H100 (datacenter, NOT edge) gets the 10 us DATACENTER
    constant written to ``resource_model.kernel_launch_overhead_ns``."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    assert hw.kernel_launch_overhead_ns == pytest.approx(
        GPUMapper.KERNEL_LAUNCH_OVERHEAD_DATACENTER * 1e9
    )
    assert hw.kernel_launch_overhead_ns == 10_000.0


def test_cpu_mapper_does_not_touch_kernel_launch_overhead_ns():
    """CPU mappers don't go through GPUMapper, so the field stays
    None. (CPU has its own op-aware dispatch floor in
    RooflineAnalyzer._get_cpu_dispatch_floor.)"""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.kernel_launch_overhead_ns is None


def test_explicit_override_on_resource_model_wins():
    """If the mapper / caller has already set
    ``kernel_launch_overhead_ns`` (e.g. for a future per-Jetson-
    power-mode override), the GPUMapper construction respects it."""
    from graphs.hardware.models.edge.jetson_orin_nano_8gb import (
        jetson_orin_nano_8gb_resource_model,
    )

    rm = jetson_orin_nano_8gb_resource_model()
    rm.kernel_launch_overhead_ns = 60_000.0  # 60 us, hypothetical custom value
    GPUMapper(rm)  # construction; should NOT overwrite the explicit value
    assert rm.kernel_launch_overhead_ns == 60_000.0


def test_analyzer_picks_up_jetson_edge_overhead():
    """RooflineAnalyzer._estimate_overhead reads the new field for
    GPU. For a Jetson resource_model with 80 us
    kernel_launch_overhead_ns, the base overhead in the GPU branch
    is 80 us (vs the legacy 5 us)."""
    from graphs.core.structures import (
        OperationType,
        SubgraphDescriptor,
        create_tensor_descriptor,
    )

    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    # Build a vector_add subgraph (no special op-type bumps in
    # _estimate_overhead, so the base passes through cleanly).
    a = create_tensor_descriptor((1024,), "float16")
    b = create_tensor_descriptor((1024,), "float16")
    c = create_tensor_descriptor((1024,), "float16")
    sg = SubgraphDescriptor(
        subgraph_id=1,
        node_ids=["va"],
        node_names=["va"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="Add",
        total_flops=1024,
        total_macs=0,
        total_input_bytes=2 * 1024 * 2,
        total_output_bytes=1024 * 2,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
        weight_tensors=[],
    )
    overhead = analyzer._estimate_overhead(sg)
    # Should be at least the Jetson 80 us base (might be higher if
    # an op-type bump fires; vanilla ELEMENTWISE shouldn't trigger
    # any of the bumps in the function body).
    assert overhead >= 80e-6, (
        f"Expected at least 80 us base for Jetson, got " f"{overhead*1e6:.2f} us"
    )


def test_analyzer_picks_up_datacenter_overhead():
    """H100 mapper has 10 us datacenter overhead; the analyzer's
    GPU branch picks it up the same way as Jetson."""
    from graphs.core.structures import (
        OperationType,
        SubgraphDescriptor,
        create_tensor_descriptor,
    )

    hw = create_h100_sxm5_80gb_mapper().resource_model
    analyzer = RooflineAnalyzer(hw, precision=Precision.FP16)
    a = create_tensor_descriptor((1024,), "float16")
    b = create_tensor_descriptor((1024,), "float16")
    c = create_tensor_descriptor((1024,), "float16")
    sg = SubgraphDescriptor(
        subgraph_id=2,
        node_ids=["va"],
        node_names=["va"],
        operation_types=[OperationType.ELEMENTWISE],
        fusion_pattern="Add",
        total_flops=1024,
        total_macs=0,
        total_input_bytes=2 * 1024 * 2,
        total_output_bytes=1024 * 2,
        total_weight_bytes=0,
        input_tensors=[a, b],
        output_tensors=[c],
        weight_tensors=[],
    )
    overhead = analyzer._estimate_overhead(sg)
    assert overhead >= 10e-6, (
        f"Expected at least 10 us base for H100, got " f"{overhead*1e6:.2f} us"
    )
    # And less than the Jetson 80 us (sanity)
    assert overhead < 80e-6
