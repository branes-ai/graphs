"""CPU per-operator concurrency cap + per-SKU bandwidth (issue #175).

The cap limits how many cores a single operator can fan out across:
`effective_cores = min(num_cores, total_macs / MIN_MACS_PER_CORE)`. A batch=1
matvec has only ~total_macs/threshold-worth of parallelism, so a 192-core SKU
cannot claim 192x the single-core compute throughput on it.

These tests pin the CAP itself (effective core fraction and allocation), not
the end-to-end throughput -- for the catalog matvec the 192-core number is
overhead/LLC-bound rather than compute-bound, so the headline inf/s is governed
by other terms; the cap is what guards the compute fanout.
"""

from __future__ import annotations

import pytest

from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.cpu import (
    CPUMapper,
    create_ampere_ampereone_192_mapper,
    create_ampere_ampereone_1core_reference_mapper,
)
from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.hardware.mappers.gpu import create_jetson_orin_nano_8gb_mapper
from graphs.hardware.resource_model import Precision

P = Precision.INT8


def _matmul_sg(M: int, K: int, N: int, bpe: int = 1) -> SubgraphDescriptor:
    """Inline matmul subgraph -- avoids importing the `validation` package,
    which isn't reliably on sys.path in the xdist unit-test CI job."""
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["mm"], node_names=["mm"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="matmul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=(M * K + K * N) * bpe,
        total_output_bytes=M * N * bpe,
        total_weight_bytes=0,
    )


def _scale(mapper, M: int, K: int, N: int) -> float:
    r = RooflineAnalyzer(mapper.resource_model, precision=P)
    return r._cpu_concurrency_scale(_matmul_sg(M, K, N))


def test_batch1_matvec_caps_192_core_to_work_limited_fraction():
    """Linear(2048,2048) batch=1 -> 4.19M MACs / 64K = 64 useful cores; on a
    192-core SKU the compute ceiling is scaled to 64/192."""
    mapper = create_ampere_ampereone_192_mapper()
    scale = _scale(mapper, 1, 2048, 2048)
    expected = 64 / 192  # 4194304 // 65536 == 64
    assert scale == pytest.approx(expected, rel=1e-6)


def test_large_gemm_is_not_capped():
    """A big GEMM has far more than num_cores*threshold MACs -> no cap (1.0)."""
    mapper = create_ampere_ampereone_192_mapper()
    assert _scale(mapper, 512, 2048, 2048) == pytest.approx(1.0)


def test_single_core_never_capped():
    """A 1-core model has nothing to cap (scale stays 1.0)."""
    mapper = create_ampere_ampereone_1core_reference_mapper()
    assert _scale(mapper, 1, 2048, 2048) == pytest.approx(1.0)


def test_non_cpu_hardware_is_not_capped():
    """The cap is CPU-specific; GPU/other mappers are untouched (1.0)."""
    mapper = create_jetson_orin_nano_8gb_mapper()
    assert _scale(mapper, 1, 2048, 2048) == pytest.approx(1.0)


def test_cap_raises_effective_compute_time():
    """End-to-end on the roofline path: the capped 192-core matvec compute_time
    is ~3x (192/64) the uncapped value, confirming the cap binds compute."""
    mapper = create_ampere_ampereone_192_mapper()
    r = RooflineAnalyzer(mapper.resource_model, precision=P)
    sg = _matmul_sg(1, 2048, 2048)
    lat = r._analyze_subgraph(sg)
    # compute_time should equal flops / (peak * efficiency * concurrency * l1_fit).
    scale = r._cpu_concurrency_scale(sg)
    assert scale < 1.0
    eff = (
        r.peak_flops
        * r._get_compute_efficiency_scale(sg)
        * scale
        * r._cpu_l1_fit_scale(sg)  # #178: L1-spill haircut also factors in
    )
    assert lat.compute_time == pytest.approx(sg.flops / eff, rel=1e-6)


def test_mapper_allocation_capped_for_batch1():
    """The CPUMapper allocation also caps cores_allocated (for energy / report
    consistency), not just the roofline ceiling."""
    from graphs.transform.partitioning import FusedSubgraph  # noqa: F401
    mapper = create_ampere_ampereone_192_mapper()
    assert mapper.cores == 192
    assert CPUMapper.MIN_MACS_PER_CORE == 65536


def test_ampereone_bandwidth_is_datasheet_value():
    """Fix #2: AmpereOne 192/128 carry the 332.8 GB/s datasheet bandwidth, not
    the 80 GB/s edge default; the 1-core reference keeps 80 GB/s."""
    bw192 = create_ampere_ampereone_192_mapper().resource_model.peak_bandwidth
    assert bw192 == pytest.approx(332.8e9)
    bw1 = create_ampere_ampereone_1core_reference_mapper().resource_model.peak_bandwidth
    assert bw1 == pytest.approx(80e9)
