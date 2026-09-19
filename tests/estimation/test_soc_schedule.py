"""Engines, mapping and the per-profile schedule (graphs#269 PR 3.2)."""

from __future__ import annotations

import pytest
from embodied_schemas import load_process_nodes

from graphs.core.pipeline_workload import load_autonomy_workload
from graphs.estimation.soc import (
    EfficiencyTable,
    KernelClass,
    load_efficiency_tables,
    load_kernel_classes,
)
from graphs.estimation.soc.mapping import (
    SUSTAINED_DRAM_FRACTION,
    engine_service,
    engines_of,
    find_mapping,
)
from graphs.estimation.soc.schedule import schedule
from graphs.hardware.soc import Confidence, compose_soc, load_designs, load_ip_library

WORKLOAD = load_autonomy_workload()
TABLES = load_efficiency_tables()
KERNELS = load_kernel_classes()
DESIGN = load_designs()["orin_class_reference"]
FAR, AIR = WORKLOAD.regimes()


@pytest.fixture(scope="module")
def orin():
    return compose_soc(DESIGN, load_ip_library(), load_process_nodes(), "samsung_8lpp")


@pytest.fixture(scope="module")
def explicit():
    return find_mapping(DESIGN.id, WORKLOAD.version).assignments()


def _everything_known(eff: float = 0.5) -> EfficiencyTable:
    """A synthetic table pricing every pair, to exercise the arithmetic."""
    entries = [
        dict(kernel_class=k.value, engine_kind=e, precision=p, compute_eff=eff,
             confidence="theoretical", source="test")
        for k in KernelClass for e in ("cpu", "gpu", "npu")
        for p in ("int8", "fp16", "fp32", "fp64")
    ]
    return EfficiencyTable.model_validate(
        dict(id="all_known", name="test", kind="per_engine", entries=entries))


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def test_orin_engines_and_their_servers(orin):
    engines = engines_of(orin)
    assert {n: e.servers for n, e in engines.items()} == {"gpu_sm": 1, "dla": 2, "cpu": 12}


def test_a_server_is_one_core_one_dla_or_the_whole_gpu(orin):
    engines = engines_of(orin)
    assert engines["cpu"].server_peak_ops_per_s("fp32") == pytest.approx(16 * 2.2e9)
    assert engines["dla"].server_peak_ops_per_s("int8") == pytest.approx(16384 * 1.6e9)
    assert engines["gpu_sm"].server_peak_ops_per_s("int8") == pytest.approx(16 * 4096 * 1.3e9)


def test_blocks_without_compute_are_not_engines(orin):
    assert not {"pva", "isp", "codec", "memory"} & set(engines_of(orin))


def test_orin_dram_is_nvidias_figure(orin):
    assert orin.dram_peak_gb_per_s == pytest.approx(204.8)


# ---------------------------------------------------------------------------
# annex_v1: the analyzer reproduces the annex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", WORKLOAD.profiles, ids=lambda p: p.id)
def test_annex_v1_reproduces_every_profile(orin, profile):
    ref = WORKLOAD.summary(profile)
    got = schedule(WORKLOAD, profile, orin, TABLES["annex_v1"])
    assert got.complete
    assert got.oversubscription == pytest.approx(ref.oversubscription, rel=1e-12)
    assert got.stages_over() == ref.stages_over()
    assert got.reactive_chain_ms == pytest.approx(ref.reactive_chain_ms, rel=1e-12)
    assert got.dram_demand_gb_per_s == pytest.approx(ref.gb_per_s, rel=1e-12)


@pytest.mark.parametrize("profile, tolerance", [(FAR, 0.01), (AIR, 0.04)], ids=["far", "air"])
def test_annex_v1_matches_the_argument_documents_regimes(orin, profile, tolerance):
    """16.2 and 17.3 s/s from the argument document; air superiority sits at
    the annex's own 4% cross-check (Phase 1)."""
    got = schedule(WORKLOAD, profile, orin, TABLES["annex_v1"]).oversubscription
    assert got == pytest.approx(profile.published["oversubscription"], rel=tolerance)


def test_pooled_schedule_needs_no_soc_but_then_knows_no_dram_supply():
    got = schedule(WORKLOAD, FAR, None, TABLES["annex_v1"])
    assert got.dram_supply_gb_per_s is None and got.dram_utilization is None


def test_far_flight_outruns_orins_dram(orin):
    """The annex's own DRAM demand for far flight (306 GB/s) exceeds Orin's
    204.8 GB/s peak, let alone the 65% sustained: infeasible on memory before
    compute is considered."""
    got = schedule(WORKLOAD, FAR, orin, TABLES["annex_v1"])
    assert got.dram_supply_gb_per_s == pytest.approx(204.8 * SUSTAINED_DRAM_FRACTION)
    assert got.dram_demand_gb_per_s > orin.dram_peak_gb_per_s
    assert got.dram_utilization > 2
    assert got.feasible() is False


# ---------------------------------------------------------------------------
# default_v1: gaps, never numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mapping", ["greedy", "explicit"])
def test_default_v1_on_orin_is_all_gaps(orin, explicit, mapping):
    """Only two pairs are measured, and neither is a format these stages run
    in, so every stage is a stated gap: no service time, no oversubscription
    presented as if it were whole."""
    got = schedule(WORKLOAD, AIR, orin, TABLES["default_v1"], KERNELS,
                   explicit if mapping == "explicit" else "greedy")
    assert not got.complete
    assert len(got.gaps) == len(WORKLOAD.demands(AIR))
    assert got.oversubscription == 0.0
    assert got.feasible() is None
    assert got.confidence == Confidence.UNKNOWN
    for _, reason in got.gaps:
        assert "default_v1 efficiency" in reason


def test_the_explicit_mapping_covers_every_stage_with_a_reason():
    mapping = find_mapping(DESIGN.id, WORKLOAD.version)
    assert set(mapping.stages) == set(WORKLOAD.stages)
    assert mapping.design == DESIGN.id and mapping.workload == WORKLOAD.version


def test_a_dla_cannot_run_a_class_b_share(orin):
    """det is 15% Class B; the DLA template is INT8 only. Fail closed."""
    dla = engines_of(orin)["dla"]
    demand = next(d for d in WORKLOAD.demands(AIR) if d.stage.key == "det")
    svc = engine_service(demand, KernelClass.DENSE_CONV_GEMM, dla, _everything_known(), 0.0)
    assert not svc.served and svc.gap == "npu cannot run Class B"


def test_an_unknown_engine_name_is_an_error(orin):
    with pytest.raises(KeyError, match="does not have"):
        schedule(WORKLOAD, AIR, orin, TABLES["default_v1"], KERNELS, {"det": "tpu"})


def test_a_stage_left_out_of_an_explicit_mapping_is_a_gap(orin):
    got = schedule(WORKLOAD, AIR, orin, _everything_known(), KERNELS, {"det": "gpu_sm"})
    reasons = dict(got.gaps)
    assert "det" not in reasons and reasons["mpc"] == "not in the explicit mapping"


# ---------------------------------------------------------------------------
# Arithmetic, on a synthetic table that prices everything
# ---------------------------------------------------------------------------


def test_service_time_is_ops_over_peak_times_efficiency_per_class(orin):
    gpu = engines_of(orin)["gpu_sm"]
    demand = next(d for d in WORKLOAD.demands(AIR) if d.stage.key == "det")
    svc = engine_service(demand, KernelClass.DENSE_CONV_GEMM, gpu, _everything_known(0.5), 0.0)
    a, b, _ = demand.stage.class_split
    ops = demand.stage.ops_per_call
    # Class A in INT8; Class B in FP16, the SM's sourced tensor rate.
    assert svc.formats == {"A": "int8", "B": "fp16"}
    expected = ops * a / (gpu.server_peak_ops_per_s("int8") * 0.5) + ops * b / (
        gpu.server_peak_ops_per_s("fp16") * 0.5)
    assert svc.t_compute_s == pytest.approx(expected)
    assert svc.t_memory_s is None and svc.bound == "compute"


def test_a_low_intensity_stage_is_memory_bound(orin):
    gpu = engines_of(orin)["gpu_sm"]
    demand = next(d for d in WORKLOAD.demands(FAR) if d.stage.key == "gain")  # ~2 op/B
    supply = orin.dram_peak_gb_per_s * SUSTAINED_DRAM_FRACTION
    svc = engine_service(demand, KernelClass.RAYCAST, gpu, _everything_known(0.5), supply)
    assert svc.bound == "memory"
    assert svc.t_service_s == pytest.approx(demand.stage.bytes_per_call / (supply * 1e9))


def test_greedy_prices_every_stage_when_everything_is_known(orin):
    got = schedule(WORKLOAD, AIR, orin, _everything_known(), KERNELS)
    assert got.complete and got.mapping == "greedy"
    assert got.confidence == Confidence.THEORETICAL
    # Class C stages can only go where FP32 exists: never the DLA.
    for svc in got.services:
        stage = WORKLOAD.stages[svc.stage]
        if stage.class_split[2] > 0:
            assert svc.engine != "dla"


def test_utilization_is_occupancy_over_servers(orin, explicit):
    got = schedule(WORKLOAD, AIR, orin, _everything_known(), KERNELS, explicit)
    util = got.engine_utilization()
    cpu = sum(s.occupancy for s in got.services if s.engine == "cpu")
    assert util["cpu"] == pytest.approx(cpu / 12)
    assert util["dla"] == 0.0
    assert got.bottleneck[0] in util


def test_a_proven_violation_is_infeasible_even_with_gaps(orin):
    """Gaps leave feasibility open only when nothing priced already fails."""
    got = schedule(WORKLOAD, FAR, orin, _everything_known(1e-6), KERNELS, {"det": "gpu_sm"})
    assert not got.complete
    assert got.stages_over() == ("det",)
    assert got.feasible() is False


def test_the_result_serializes(orin, explicit):
    out = schedule(WORKLOAD, AIR, orin, TABLES["default_v1"], KERNELS, explicit).to_dict()
    assert out["summary"]["feasible"] is None
    assert out["stages"][0]["t_service_ms"] is None
    assert {e["engine"] for e in out["engines"]} == {"gpu_sm", "dla", "cpu"}


def test_a_per_engine_table_needs_the_soc_and_kernel_classes():
    with pytest.raises(ValueError, match="per-engine"):
        schedule(WORKLOAD, AIR, None, TABLES["default_v1"])


# ---------------------------------------------------------------------------
# Review findings on #305
# ---------------------------------------------------------------------------


def test_unknown_dram_supply_leaves_feasibility_open():
    """Without a composed SoC nothing states the DRAM supply; that is not zero
    utilization, so a light profile that violates nothing is open, not
    feasible."""
    light = min(WORKLOAD.profiles, key=lambda p: WORKLOAD.summary(p).oversubscription)
    got = schedule(WORKLOAD, light, None, TABLES["annex_v1"])
    assert got.complete and not got.stages_over()
    assert got.dram_demand_gb_per_s > 0 and got.dram_supply_gb_per_s is None
    assert got.feasible() is None


def test_a_provisional_clock_makes_its_services_unknown(explicit):
    """At N7, Orin's clocks have no stated relation from 8LPP: the peaks the
    service times divide by are provisional, so those estimates are UNKNOWN
    and say why."""
    at_n7 = compose_soc(DESIGN, load_ip_library(), load_process_nodes(), "tsmc_n7")
    got = schedule(WORKLOAD, AIR, at_n7, _everything_known(), KERNELS, explicit)
    assert got.complete
    for svc in got.services:
        assert svc.confidence == Confidence.UNKNOWN
        assert "clock is provisional" in svc.confidence_source
    assert got.confidence == Confidence.UNKNOWN


def test_estimates_carry_the_repo_estimation_confidence(orin, explicit):
    from graphs.core import ConfidenceLevel, EstimationConfidence

    got = schedule(WORKLOAD, AIR, orin, _everything_known(), KERNELS, explicit)
    svc = got.services[0]
    assert isinstance(svc.estimation_confidence, EstimationConfidence)
    assert svc.estimation_confidence.level == ConfidenceLevel.THEORETICAL
    assert "efficiency" in svc.estimation_confidence.source
    assert got.estimation_confidence.level == ConfidenceLevel.THEORETICAL
    gaps = schedule(WORKLOAD, AIR, orin, TABLES["default_v1"], KERNELS, explicit)
    assert gaps.estimation_confidence.level == ConfidenceLevel.UNKNOWN
    assert "unpriced" in gaps.estimation_confidence.source
