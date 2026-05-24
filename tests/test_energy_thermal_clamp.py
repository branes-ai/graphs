"""Regression tests for the EnergyAnalyzer thermal-envelope clamp (#177 / #121).

TDP is a *sustained* thermal limit, not an instantaneous power cap. Before the
clamp, the KPU energy model divided an (uncalibrated) per-inference energy by a
sub-TDP burst latency and reported average power 18x-37x above TDP for tiny
ALU-dominated ops -- physically impossible for the thermal envelope.

The clamp keeps energy physical (truthful joules) and floors the *effective*
(sustained) latency at ``energy / TDP``, so ``average_power_w`` can never exceed
TDP. These tests pin that behavior on the regime that exposed the bug and assert
the clamp is inert when the workload already fits under TDP.
"""

from __future__ import annotations

import pytest

from graphs.estimation.energy import EnergyAnalyzer
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper
from graphs.hardware.resource_model import Precision

from validation.model_v4.invariants.kpu import _MatmulShape, _bpe


PRECISION = Precision.INT8


def _analyze(hw, shape: _MatmulShape):
    """Roofline-latency + energy for one matmul shape on ``hw``."""
    roof = RooflineAnalyzer(hw, precision=PRECISION)
    energy = EnergyAnalyzer(hw, precision=PRECISION)
    sg = shape.to_subgraph(_bpe(PRECISION))
    lat = roof._analyze_subgraph(sg)
    report = energy.analyze(subgraphs=[sg], latencies=[lat.actual_latency])
    return energy, report, lat.actual_latency


def test_compute_bound_matmul_clamps_avg_power_to_tdp():
    """A compute-bound matmul whose uncalibrated burst power exceeds TDP (the
    #81 regime) must report avg power <= TDP after the clamp."""
    hw = create_kpu_t64_mapper().resource_model
    energy, report, burst_latency = _analyze(hw, _MatmulShape(1024, 1024, 1024))

    # Burst latency alone would imply avg power above TDP -> throttle fires.
    assert report.thermal_throttle_active is True
    assert report.average_power_w <= energy.tdp_watts * 1.001
    # Effective latency floored upward; burst latency preserved separately.
    assert report.thermally_bound_latency_s > report.total_latency_s
    assert report.total_latency_s == pytest.approx(burst_latency)


def test_energy_is_not_faked_down_by_clamp():
    """The clamp moves latency, never energy -- joules stay physical."""
    hw = create_kpu_t64_mapper().resource_model
    _, report, _ = _analyze(hw, _MatmulShape(1024, 1024, 1024))

    # average_power_w == energy / thermally_bound_latency (the clamped quantity),
    # NOT energy / burst_latency. Energy equals its three components untouched.
    assert report.average_power_w == pytest.approx(
        report.total_energy_j / report.thermally_bound_latency_s
    )
    assert report.total_energy_j == pytest.approx(
        report.compute_energy_j + report.memory_energy_j + report.static_energy_j
    )


def test_clamp_is_inert_when_under_tdp():
    """A memory-bound matvec whose burst power already fits under TDP must be
    left untouched -- the clamp is a backstop, not an unconditional cap."""
    hw = create_kpu_t64_mapper().resource_model
    energy, report, _ = _analyze(hw, _MatmulShape(2048, 2048, 1))

    assert report.average_power_w < energy.tdp_watts  # genuinely below TDP
    assert report.thermal_throttle_active is False
    assert report.thermally_bound_latency_s == pytest.approx(report.total_latency_s)
