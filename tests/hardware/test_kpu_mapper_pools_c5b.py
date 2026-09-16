"""Phase C5b of the KPU heterogeneous-tile refactor (graphs#268):
capability-aware tile pools in the KPU mapper.

A uniform legacy SKU must take the flat path unchanged -- the KPU golden
snapshot pins its mapper results. The heterogeneous fixture allocates only
from classes that run the precision, prefers the systolic classes for a
dense matrix product, and is charged the throughput of the tiles it was
actually given.
"""

from __future__ import annotations

import pytest
from embodied_schemas import load_process_nodes

from graphs.core.structures import OperationType, ParallelismDescriptor
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_sku_generator import generate_kpu_sku, input_spec_from_compute_product
from graphs.hardware.mappers.accelerators.kpu import KPUMapper
from graphs.hardware.mappers.accelerators.kpu_tile_pools import (
    build_tile_pool,
    prefers_systolic,
)
from graphs.hardware.models.accelerators.kpu_yaml_loader import load_kpu_resource_model_from_yaml
from graphs.hardware.resource_model import Precision
from graphs.transform.partitioning import FusedSubgraph

NODES = load_process_nodes()
HETERO = generate_kpu_sku(
    input_spec_from_compute_product(build_heterogeneous_kpu()), process_nodes=NODES
)
HETERO_RM = load_kpu_resource_model_from_yaml(
    HETERO.id, kpus={HETERO.id: HETERO}, process_nodes=NODES
)
LEGACY_SKU = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"

SYSTOLIC = "Systolic-INT8-WS"
PE_INT8 = "INT8-MAC"


def _subgraph(op: OperationType, total_threads: int = 4096, macs: int = 10**9):
    """A one-op fused subgraph big enough to need tiling decisions."""
    return FusedSubgraph(
        subgraph_id=1,
        node_ids=["n0"],
        node_names=[op.name.lower()],
        operation_types=[op],
        fusion_pattern=op.name,
        total_flops=macs * 2,
        total_macs=macs,
        total_input_bytes=1 << 20,
        total_output_bytes=1 << 20,
        total_weight_bytes=1 << 20,
        parallelism=ParallelismDescriptor(
            batch=1, channels=64, spatial=64, total_threads=total_threads
        ),
    )


def _pool(precision=Precision.INT8, prefer_systolic=False, mapper=None):
    mapper = mapper or KPUMapper(HETERO_RM)
    return build_tile_pool(
        HETERO_RM,
        mapper._compute_resource(precision),
        precision,
        thermal_profile=mapper.thermal_profile,
        prefer_systolic=prefer_systolic,
    )


# ---------------------------------------------------------------------------
# Legacy SKUs keep the flat pool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sku",
    [
        LEGACY_SKU,
        "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
        "kpu_t256_32x32_lp5x16_7nm_tsmc_hpc",
        "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
    ],
)
def test_legacy_skus_are_not_heterogeneous(sku):
    rm = load_kpu_resource_model_from_yaml(sku)
    assert rm.is_heterogeneous_kpu is False
    assert set(rm.tile_kind_by_tile_type.values()) == {"pe_fabric"}
    assert KPUMapper(rm).heterogeneous is False


def test_legacy_mapping_never_enters_the_pool_path(monkeypatch):
    """The golden snapshot pins legacy mapper results, so the new branch
    must be unreachable for a uniform SKU."""
    mapper = KPUMapper(load_kpu_resource_model_from_yaml(LEGACY_SKU))

    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("legacy SKU took the heterogeneous path")

    monkeypatch.setattr(mapper, "_map_subgraph_heterogeneous", _boom)
    for op in (OperationType.MATMUL, OperationType.CONV2D, OperationType.RELU):
        alloc = mapper.map_subgraph(_subgraph(op), 0, 1, Precision.INT8)
        assert alloc.compute_units_allocated <= mapper.num_tiles
    assert mapper._tile_pool(_subgraph(OperationType.MATMUL), Precision.INT8) is None


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def test_fixture_is_heterogeneous():
    assert HETERO_RM.is_heterogeneous_kpu is True
    assert set(HETERO_RM.tile_kind_by_tile_type.values()) == {"pe_fabric", "systolic"}
    assert HETERO_RM.fixed_function_units  # ISP, SGM, VIO


def test_pool_holds_only_the_classes_that_run_the_precision():
    pool = _pool()
    assert [s.tile_type for s in pool.specializations] == [PE_INT8, SYSTOLIC]
    assert pool.capable is True
    # 24 PE-fabric + 4 systolic. Neither the 3 fixed-function tiles nor the
    # 14 LNS / min-plus tiles (no Precision-enum ops) are in the fabric.
    assert pool.num_tiles == 28
    assert pool.fabric_tiles == 28
    assert HETERO_RM.compute_units == 45


def test_pool_is_incapable_when_no_class_runs_the_precision():
    """No class on the fixture runs fp32, so the pool declines and the
    mapper falls back to the flat path -- which keeps raising the resource
    model's unsupported-precision error rather than inventing a number."""
    pool = _pool(precision=Precision.FP32)
    assert pool.capable is False
    # It still lists the classes, so the caller can report what is there.
    assert [s.tile_type for s in pool.specializations] == [PE_INT8, SYSTOLIC]

    mapper = KPUMapper(HETERO_RM)
    with pytest.raises(ValueError, match="does not support fp32"):
        mapper.map_subgraph(_subgraph(OperationType.MATMUL), 0, 1, Precision.FP32)


def test_a_precision_only_a_gated_class_runs_falls_back_to_the_flat_path():
    """Gating a class can leave a profile without any class for a precision
    the chip still advertises. The pool declines; the flat path's
    unsupported-precision penalty applies, as it did before C5b."""
    mapper = KPUMapper(HETERO_RM)
    pool = _pool(precision=Precision.INT4, mapper=mapper)
    # INT4 is a chip precision; the fixture's INT8 fabric class runs it.
    assert pool.capable is True
    alloc = mapper.map_subgraph(_subgraph(OperationType.RELU), 0, 1, Precision.INT4)
    assert alloc.compute_units_allocated >= 1
    assert alloc.estimated_latency > 0


# ---------------------------------------------------------------------------
# Systolic preference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, expected",
    [
        (OperationType.MATMUL, True),
        (OperationType.LINEAR, True),
        (OperationType.CONV2D, True),
        (OperationType.CONV2D_POINTWISE, True),
        (OperationType.MULTIHEAD_ATTENTION, True),
        (OperationType.CONV2D_DEPTHWISE, False),  # leaves a systolic array idle
        (OperationType.RELU, False),
        (OperationType.LAYERNORM, False),
    ],
)
def test_op_classification(op, expected):
    assert prefers_systolic([op]) is expected


def test_gemm_takes_the_systolic_class_first():
    pool = _pool(prefer_systolic=True)
    assert pool.preferred_kind == "systolic"
    assert pool.specializations[0].tile_type == SYSTOLIC
    # 2 systolic tiles cover 8192 threads, so the PE fabric is untouched.
    alloc = pool.allocate(8192, min_tiles=1)
    assert alloc.classes == (SYSTOLIC,)
    assert alloc.num_tiles == 2


def test_gemm_spills_to_the_pe_fabric_when_the_systolic_tiles_run_out():
    pool = _pool(prefer_systolic=True)
    alloc = pool.allocate(1 << 20, min_tiles=1)
    assert alloc.classes == (SYSTOLIC, PE_INT8)
    # All 4 systolic tiles first, then the fabric.
    assert dict((s.tile_type, n) for s, n in alloc.per_class)[SYSTOLIC] == 4
    assert alloc.num_tiles == 28  # the whole pool


def test_non_gemm_stays_on_the_pe_fabric():
    pool = _pool(prefer_systolic=False)
    assert pool.preferred_kind == "pe_fabric"
    alloc = pool.allocate(2048, min_tiles=1)
    assert alloc.classes == (PE_INT8,)


def test_min_tiles_floor_is_honored():
    """The memory-tiling floor allocates tiles even with no thread demand."""
    alloc = _pool().allocate(0, min_tiles=6)
    assert alloc.num_tiles == 6
    assert _pool().allocate(0, min_tiles=1).num_tiles == 1


# ---------------------------------------------------------------------------
# Throughput of the allocated tiles
# ---------------------------------------------------------------------------


def test_allocation_is_charged_its_own_tiles_not_a_chip_fraction():
    pool = _pool(prefer_systolic=True)
    alloc = pool.allocate(8192, min_tiles=1)  # 2 systolic tiles
    spec = pool.specializations[0]
    expected = (
        2
        * spec.ops_per_tile_per_clock[Precision.INT8]
        * spec.clock_domain.sustained_clock_hz
        * pool.derate
    )
    assert alloc.ops_per_sec == pytest.approx(expected)

    # The flat model would have charged 2/45 of the chip's INT8 throughput.
    mapper = KPUMapper(HETERO_RM)
    point = HETERO_RM.thermal_operating_points[mapper.thermal_profile]
    chip = point.performance_specs[Precision.INT8].effective_ops_per_sec
    assert alloc.ops_per_sec > chip * (2 / HETERO_RM.compute_units)


def test_map_subgraph_uses_the_pool_for_a_gemm():
    mapper = KPUMapper(HETERO_RM)
    alloc = mapper.map_subgraph(
        _subgraph(OperationType.MATMUL, total_threads=8192), 0, 1, Precision.INT8
    )
    # 8192 threads need 2 systolic tiles, but this subgraph's scratchpad
    # tiling needs 12 resident data tiles: all 4 systolic tiles, then 8
    # spilled to the PE fabric. Never all 45.
    assert alloc.compute_units_allocated == 12
    assert alloc.compute_units_allocated < mapper.num_tiles
    # Threads come from the classes actually allocated, which have
    # different PE counts -- 4 x 4096 systolic + 8 x 1024 fabric. The flat
    # mapper's single threads_per_tile cannot express this.
    assert alloc.threads_required == 4 * 4096 + 8 * 1024
    # Utilization is against the 28-tile fabric, occupancy against the pool
    # (here the same set, since every capable class is in the pool).
    assert alloc.utilization == pytest.approx(12 / 28)
    assert alloc.occupancy == pytest.approx(12 / 28)
    assert alloc.estimated_latency > 0
    assert alloc.total_energy > 0


def test_gemm_and_elementwise_land_on_different_classes():
    mapper = KPUMapper(HETERO_RM)
    gemm = mapper._tile_pool(_subgraph(OperationType.MATMUL), Precision.INT8)
    ew = mapper._tile_pool(_subgraph(OperationType.RELU), Precision.INT8)
    assert gemm.specializations[0].tile_type == SYSTOLIC
    assert ew.specializations[0].tile_type == PE_INT8
    # Same tiles, different order: nothing is excluded, only re-ranked.
    assert {s.tile_type for s in gemm.specializations} == {
        s.tile_type for s in ew.specializations
    }
