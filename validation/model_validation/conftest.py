"""
Shared fixtures for model-based validation tests.

Provides standardized synthetic workloads and mapper collections so
each architecture-class test module can focus on its consistency
checks rather than boilerplate construction.
"""

from __future__ import annotations

from typing import List

import pytest

from graphs.core.structures import SubgraphDescriptor, OperationType
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Synthetic workload fixtures
# ---------------------------------------------------------------------------

def _make_subgraph(
    name: str,
    op_type: OperationType,
    flops: int,
    macs: int,
    input_bytes: int,
    output_bytes: int,
    weight_bytes: int,
) -> SubgraphDescriptor:
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["0"],
        node_names=[name],
        operation_types=[op_type],
        fusion_pattern=name,
        total_flops=flops,
        total_macs=macs,
        total_input_bytes=input_bytes,
        total_output_bytes=output_bytes,
        total_weight_bytes=weight_bytes,
    )


@pytest.fixture
def tiny_matmul() -> SubgraphDescriptor:
    """64x64x64 matmul: 524K FLOPs, tests small-op overhead."""
    M, N, K = 64, 64, 64
    flops = 2 * M * N * K
    macs = M * N * K
    return _make_subgraph(
        "tiny_matmul", OperationType.MATMUL,
        flops=flops, macs=macs,
        input_bytes=M * K * 4, output_bytes=M * N * 4, weight_bytes=K * N * 4,
    )


@pytest.fixture
def medium_matmul() -> SubgraphDescriptor:
    """1024x1024x1024 matmul: 2.1G FLOPs, typical layer."""
    M, N, K = 1024, 1024, 1024
    flops = 2 * M * N * K
    macs = M * N * K
    return _make_subgraph(
        "medium_matmul", OperationType.MATMUL,
        flops=flops, macs=macs,
        input_bytes=M * K * 4, output_bytes=M * N * 4, weight_bytes=K * N * 4,
    )


@pytest.fixture
def large_matmul() -> SubgraphDescriptor:
    """4096x4096x4096 matmul: 137G FLOPs, compute-bound."""
    M, N, K = 4096, 4096, 4096
    flops = 2 * M * N * K
    macs = M * N * K
    return _make_subgraph(
        "large_matmul", OperationType.MATMUL,
        flops=flops, macs=macs,
        input_bytes=M * K * 4, output_bytes=M * N * 4, weight_bytes=K * N * 4,
    )


@pytest.fixture
def elementwise_relu() -> SubgraphDescriptor:
    """1M-element ReLU: near-zero compute, pure bandwidth."""
    n = 1_000_000
    return _make_subgraph(
        "elementwise_relu", OperationType.RELU,
        flops=0, macs=0,
        input_bytes=n * 4, output_bytes=n * 4, weight_bytes=0,
    )


@pytest.fixture
def depthwise_conv() -> SubgraphDescriptor:
    """3x3 depthwise conv on 56x56x64: memory-bound, low arithmetic intensity."""
    H, W, C = 56, 56, 64
    K = 3
    flops = 2 * K * K * C * H * W
    macs = K * K * C * H * W
    return _make_subgraph(
        "depthwise_conv", OperationType.CONV2D_DEPTHWISE,
        flops=flops, macs=macs,
        input_bytes=C * H * W * 4, output_bytes=C * H * W * 4,
        weight_bytes=C * K * K * 4,
    )


# ---------------------------------------------------------------------------
# Mapper collection fixtures
# ---------------------------------------------------------------------------

RESEARCH_MAPPERS = {"Stillwater-DFM-128"}


def _get_mappers_by_category(category: str) -> List[tuple]:
    """Return (name, mapper) pairs for a mapper category.

    Excludes research/reference architectures (e.g., DFM) that
    intentionally model energy savings rather than additive overhead.
    """
    from graphs.hardware.mappers import list_all_mappers, get_mapper_by_name

    all_names = list_all_mappers()
    results = []
    for name in sorted(all_names):
        if name in RESEARCH_MAPPERS:
            continue
        try:
            mapper = get_mapper_by_name(name)
            hw_type = mapper.resource_model.hardware_type.value
            if hw_type == category:
                results.append((name, mapper))
        except Exception:
            # Skip mappers that fail to instantiate (missing deps, etc.)
            pass
    return results


@pytest.fixture(scope="module")
def cpu_mappers():
    """All CPU mappers as (name, mapper) pairs."""
    mappers = _get_mappers_by_category("cpu")
    assert len(mappers) > 0, "No CPU mappers found -- mapper registry is broken"
    return mappers


@pytest.fixture(scope="module")
def gpu_mappers():
    """All GPU mappers as (name, mapper) pairs."""
    mappers = _get_mappers_by_category("gpu")
    assert len(mappers) > 0, "No GPU mappers found -- mapper registry is broken"
    return mappers


@pytest.fixture(scope="module")
def tpu_mappers():
    """All TPU mappers as (name, mapper) pairs."""
    mappers = _get_mappers_by_category("tpu")
    assert len(mappers) > 0, "No TPU mappers found -- mapper registry is broken"
    return mappers


@pytest.fixture(scope="module")
def kpu_mappers():
    """KPU mappers (excludes Hailo which shares the kpu hardware type)."""
    mappers = [
        (name, m) for name, m in _get_mappers_by_category("kpu")
        if "KPU" in name or "Stillwater" in name
    ]
    assert len(mappers) > 0, "No KPU mappers found"
    return mappers


@pytest.fixture(scope="module")
def hailo_mappers():
    """Hailo mappers (registered under kpu hardware type)."""
    mappers = [
        (name, m) for name, m in _get_mappers_by_category("kpu")
        if "Hailo" in name
    ]
    assert len(mappers) > 0, "No Hailo mappers found"
    return mappers


@pytest.fixture(scope="module")
def dsp_mappers():
    """All DSP mappers as (name, mapper) pairs."""
    mappers = _get_mappers_by_category("dsp")
    assert len(mappers) > 0, "No DSP mappers found"
    return mappers


@pytest.fixture(scope="module")
def dpu_cgra_mappers():
    """DPU + CGRA mappers as (name, mapper) pairs."""
    dpu = _get_mappers_by_category("dpu")
    cgra = _get_mappers_by_category("cgra")
    mappers = dpu + cgra
    assert len(mappers) > 0, "No DPU/CGRA mappers found"
    return mappers


# ---------------------------------------------------------------------------
# Precision fixtures
# ---------------------------------------------------------------------------

COMMON_PRECISIONS = [Precision.FP32, Precision.FP16, Precision.INT8]


@pytest.fixture
def common_precisions():
    return COMMON_PRECISIONS
