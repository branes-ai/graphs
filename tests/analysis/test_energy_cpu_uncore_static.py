"""CPU static power is package-level, not per-core (issue #78).

Uncore, IMC, ring bus, and shared L3 draw power regardless of how many cores
are active. The per-unit allocation branch in EnergyAnalyzer used to divide the
calibrated package power across allocated cores (and power-gate the rest),
underestimating CPU static energy -- worse now that the #175 concurrency cap
allocates fewer cores at batch=1. The fix charges the full #71-calibrated
idle_power_watts for CPU regardless of allocation, while non-CPU accelerators
keep their per-unit power-gating model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.energy import EnergyAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
from graphs.hardware.resource_model import Precision

P = Precision.INT8
LAT = 0.001


def _matmul_sg(M: int, K: int, N: int, bpe: int = 1) -> SubgraphDescriptor:
    """Build a matmul subgraph inline (no dependency on the `validation`
    package, which isn't importable in the unit-test CI job)."""
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


SG = _matmul_sg(1, 2048, 2048)


def _alloc(units, total):
    return SimpleNamespace(compute_units_allocated=units, utilization=units / total)


def _static_power(analyzer, alloc):
    d = analyzer._analyze_subgraph(SG, LAT, alloc)
    return d.static_energy_j / LAT


def test_cpu_charges_full_package_power_under_partial_allocation():
    """i7 with power gating on + 4 cores allocated must still charge the full
    calibrated package power (uncore can't be gated)."""
    hw = create_i7_12700k_mapper().resource_model
    e = EnergyAnalyzer(hw, precision=P, power_gating_enabled=True)
    sp = _static_power(e, _alloc(4, e.total_compute_units))
    assert sp == pytest.approx(e.idle_power_watts, rel=1e-9)
    # Specifically NOT the per-core fraction that was the bug.
    assert sp > e.idle_power_watts * 4 / e.total_compute_units * 1.5


def test_cpu_static_power_independent_of_allocated_cores():
    """Charging is package-level: 2 vs 8 allocated cores -> same static power."""
    hw = create_i7_12700k_mapper().resource_model
    e = EnergyAnalyzer(hw, precision=P, power_gating_enabled=True)
    tot = e.total_compute_units
    assert _static_power(e, _alloc(2, tot)) == pytest.approx(_static_power(e, _alloc(8, tot)))


def test_cpu_allocation_matches_no_allocation_fallback():
    """The per-unit path now matches the #71-calibrated no-allocation path."""
    hw = create_i7_12700k_mapper().resource_model
    e = EnergyAnalyzer(hw, precision=P, power_gating_enabled=True)
    with_alloc = _static_power(e, _alloc(4, e.total_compute_units))
    no_alloc = e._analyze_subgraph(SG, LAT, None).static_energy_j / LAT
    assert with_alloc == pytest.approx(no_alloc)


def test_gpu_power_gating_still_gates():
    """Regression: non-CPU accelerators keep per-unit power gating (the CPU
    special-case must not leak into GPU)."""
    hw = create_h100_pcie_80gb_mapper().resource_model
    e = EnergyAnalyzer(hw, precision=P, power_gating_enabled=True)
    units = max(1, e.total_compute_units // 4)
    sp = _static_power(e, _alloc(units, e.total_compute_units))
    assert sp < e.idle_power_watts  # gated below full package
