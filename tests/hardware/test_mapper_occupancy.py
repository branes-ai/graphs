"""Occupancy is the fill of the allocated units, not the chip fraction.

``HardwareMapper._calculate_latency`` charges ``ops/sec x (allocated /
compute_units) x occupancy``. Seven flat mappers (CPU, DSP, Hailo, DPU, CGRA,
the KPU flat path and the TPU vector path) passed ``allocated /
compute_units`` as ``occupancy`` too, so a partial allocation ran at the
*square* of its chip fraction: 33 of 768 KPU tiles at ~1/541 of the chip
instead of 33/768. Those mappers size their allocation to the work in whole
units, so the units are full: occupancy 1.0, with the chip fraction reported
as ``utilization``. The GPU mapper, whose occupancy is a real warp fill, and
the TPU systolic path, whose occupancy is the array fill, are unchanged.

The strongest check is on the KPU, which has two independent latency paths:
the flat one through ``_calculate_latency`` and the capability-pool path,
which charges the allocated tiles' own throughput. On a uniform SKU they must
now agree exactly, partial allocations included.
"""

from __future__ import annotations

import pytest

from graphs.core.structures import OperationType, ParallelismDescriptor, SubgraphDescriptor
from graphs.hardware import kpu_golden as kg
from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.mappers.accelerators.kpu import KPUMapper
from graphs.hardware.models.accelerators.kpu_yaml_loader import load_kpu_resource_model_from_yaml
from graphs.hardware.resource_model import Precision
from hardware.test_kpu_catalog_ids import LEGACY_KPU_SKU_IDS


def _subgraph(op: OperationType, macs: int, threads: int = 64, channels: int = 32):
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["sg"],
        node_names=["sg"],
        operation_types=[op],
        fusion_pattern="sg",
        total_flops=2 * macs,
        total_macs=macs,
        total_input_bytes=1024,
        total_output_bytes=1024,
        total_weight_bytes=1024,
        parallelism=ParallelismDescriptor(
            batch=1, channels=channels, spatial=1, total_threads=threads
        ),
    )


def _spy(mapper) -> dict:
    """Record the keyword arguments of the mapper's _calculate_latency."""
    seen: dict = {}
    original = mapper._calculate_latency

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    mapper._calculate_latency = spy
    return seen


def test_compute_time_is_inverse_in_allocated_units_not_its_square():
    mapper = get_mapper_by_name("Stillwater-KPU-T256")
    ops, nbytes = 10**9, 1024

    def compute_time(units: int, occupancy: float = 1.0) -> float:
        return mapper._calculate_latency(
            ops=ops, bytes_transferred=nbytes, allocated_units=units,
            occupancy=occupancy, precision=Precision.INT8,
        )[0]

    assert compute_time(1) / compute_time(4) == pytest.approx(4.0, rel=1e-12)
    # Occupancy is its own factor: half-full allocated units, twice the time.
    assert compute_time(4, occupancy=0.5) / compute_time(4) == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize(
    "name, precision, op, macs",
    [
        ("Intel-i7-12700K", Precision.FP32, OperationType.CONV2D, 200_000),
        ("AMD-EPYC-9654", Precision.FP32, OperationType.CONV2D, 200_000),
        ("TI-TDA4VM", Precision.INT8, OperationType.CONV2D, 200_000),
        ("Hailo-8", Precision.INT8, OperationType.CONV2D, 200_000),
        ("Xilinx-Vitis-AI-DPU", Precision.INT8, OperationType.CONV2D, 200_000),
        ("Stanford-Plasticine-v2", Precision.INT8, OperationType.CONV2D, 1_000),
        ("Google-TPU-v4", Precision.INT8, OperationType.RELU, 200_000),  # vector path
        ("Stillwater-KPU-T256", Precision.INT8, OperationType.MATMUL, 200_000),
    ],
)
def test_flat_mappers_pass_full_occupancy_on_a_partial_allocation(name, precision, op, macs):
    mapper = get_mapper_by_name(name)
    seen = _spy(mapper)
    alloc = mapper.map_subgraph(_subgraph(op, macs), 0, 1, precision)
    units = mapper.resource_model.compute_units
    # The case is only meaningful if the allocation is partial.
    assert 0 < alloc.compute_units_allocated < units
    assert seen["allocated_units"] == alloc.compute_units_allocated
    assert seen["occupancy"] == 1.0
    assert alloc.occupancy == 1.0
    chip_fraction = alloc.compute_units_allocated / units
    if name == "Stanford-Plasticine-v2":
        assert alloc.utilization <= chip_fraction  # scaled by spatial efficiency
    else:
        assert alloc.utilization == pytest.approx(chip_fraction)


def test_gpu_occupancy_is_still_the_warp_fill():
    """Untouched: a small kernel on wave-quantized SMs genuinely leaves the
    allocated SMs mostly empty, and that is what slows it."""
    mapper = get_mapper_by_name("H100-SXM5-80GB")
    seen = _spy(mapper)
    alloc = mapper.map_subgraph(
        _subgraph(OperationType.CONV2D, 10_000, threads=100), 0, 1, Precision.FP32
    )
    assert 0 < alloc.occupancy < 1
    assert seen["occupancy"] == alloc.occupancy


@pytest.mark.parametrize("sku", LEGACY_KPU_SKU_IDS)
def test_kpu_flat_path_agrees_with_the_pool_path_on_a_uniform_sku(sku):
    """Two independent derivations of the same allocation's compute time.
    Before the fix they disagreed on every partial allocation by the chip
    fraction; the T256 / T512 synthetic set has six such cases each."""
    rm = load_kpu_resource_model_from_yaml(sku)
    flat, pooled = KPUMapper(rm), KPUMapper(rm)
    pooled.heterogeneous = True  # force the capability-pool path
    compared = partial = 0
    for precision in rm.precision_profiles:
        resource = flat._compute_resource(precision)
        if resource is None:
            continue
        # Only where every class runs the precision: elsewhere the pool path
        # also narrows the allocation, which is a different (capability) fix.
        if len(resource.get_tiles_for_precision(precision)) != len(resource.tile_specializations):
            continue
        bpe = float(rm.precision_profiles[precision].bytes_per_element or 1)
        for sg in kg._synthetic_subgraphs(bpe):
            a = flat.map_subgraph(sg, 0, 1, precision)
            b = pooled.map_subgraph(sg, 0, 1, precision)
            assert a.compute_units_allocated == b.compute_units_allocated
            assert a.compute_time == pytest.approx(b.compute_time, rel=1e-12), (
                precision.value, sg.node_names[0],
            )
            compared += 1
            partial += a.compute_units_allocated < rm.compute_units
    assert compared >= 8
    if rm.compute_units >= 128:
        assert partial >= 1  # the property is exercised where it matters
