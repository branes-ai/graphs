"""Fitting autonomy missions onto the two 64-site KPU dies (graphs#268).

The deferred item on #268: compare the uniform T64 against the heterogeneous
kpu_h64_auto1. These tests pin the comparison's machinery and the findings it
produces, so a change to either die -- or to the workload -- shows up as a
changed verdict rather than a changed narrative.

The findings, in the order the model reaches them:

1. Precision beats throughput. 14 of 19 stages carry an FP32/FP64 floor, and
   the heterogeneous die has no FP32 at all, so about ten stages per mission
   do not run on it. That is the trade the tile refactor made.
2. A function core is rated for a configuration. The SGM core is a 30 fps
   part; air superiority runs stereo at 40. The Navion-class VIO core is a
   752x480 part; the drones run 1440x1080.
3. Absorbed traffic does not cross the DRAM bus, which is the heterogeneous
   die's real advantage here and is worth about 30 GB/s on far flight.
"""

from __future__ import annotations

import pytest
from embodied_schemas import load_compute_products, load_process_nodes

from graphs.core.pipeline_workload import load_autonomy_workload
from graphs.hardware.kpu_engines import describe_engines
from graphs.hardware.workload_fit import (
    CLASS_FORMATS,
    OPTIMISTIC_ACHIEVED_TO_PEAK,
    REALISTIC_ACHIEVED_TO_PEAK,
    class_capability_gops,
    fit_profile,
)

T64 = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"
H64 = "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp"

NODES = load_process_nodes()
PRODUCTS = load_compute_products()


@pytest.fixture(scope="module")
def workload():
    return load_autonomy_workload()


def _fit(workload, die_id, profile_name, achieved_to_peak=REALISTIC_ACHIEVED_TO_PEAK):
    cp = PRODUCTS[die_id]
    node = NODES[cp.dies[0].process_node_id]
    return fit_profile(
        workload, workload.profile(profile_name), cp, node, NODES, achieved_to_peak
    )


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


def test_class_capability_comes_from_the_engines():
    """Per-class peak, from the tile classes that can run each class's
    formats. Fixed-function tiles contribute nothing: their capability is
    their own work unit, not an op rate."""
    for die_id, expect_c in ((T64, True), (H64, False)):
        cp = PRODUCTS[die_id]
        engines = describe_engines(cp, NODES[cp.dies[0].process_node_id], NODES)
        caps = class_capability_gops(engines)
        assert caps["A"] > 0 and caps["B"] > 0
        assert (caps["C"] > 0) is expect_c, die_id
    t64 = class_capability_gops(
        describe_engines(PRODUCTS[T64], NODES["tsmc_n16"], NODES)
    )
    # The T64's headline: 62.3 INT8 TOPS, and FP32 only on its BF16 class.
    assert t64["A"] == pytest.approx(62259.2, rel=1e-6)
    assert t64["C"] == pytest.approx(3161.6, rel=1e-6)


def test_the_heterogeneous_die_has_no_fp32_at_all():
    """The finding #268 E3 reached for one stage, now over the whole
    pipeline: no engine on kpu_h64_auto1 offers an FP32 or FP64 format."""
    cp = PRODUCTS[H64]
    engines = describe_engines(cp, NODES["tsmc_n16"], NODES)
    formats = {p.operand_format for e in engines for p in e.precisions}
    assert not formats & set(CLASS_FORMATS["C"])
    assert class_capability_gops(engines)["C"] == 0.0


# ---------------------------------------------------------------------------
# Finding 1: precision, not throughput
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["far flight", "air superiority"])
def test_class_c_stages_do_not_run_on_the_heterogeneous_die(workload, regime):
    fit = _fit(workload, H64, regime)
    unrunnable = {s.key for s in fit.unrunnable_stages}
    assert {"mpc", "cbf", "ba", "lio", "tsdf", "esdf"} <= unrunnable
    assert len(unrunnable) >= 10
    for stage_fit in fit.unrunnable_stages:
        assert stage_fit.missing_classes == ("C",)
        assert stage_fit.demand.stage.precision_floor == "C"
    assert fit.binding_constraint == "precision"


@pytest.mark.parametrize("regime", ["far flight", "air superiority"])
def test_the_uniform_die_runs_every_stage(workload, regime):
    """The T64 keeps FP32 on its BF16-primary class, so nothing is out of
    reach; its problem is rate, not reach."""
    fit = _fit(workload, T64, regime)
    assert fit.unrunnable_stages == ()
    assert fit.binding_constraint == "compute"
    assert fit.oversubscription > 1.0


def test_precision_outranks_every_other_constraint(workload):
    """A stage the die cannot run is not a throughput problem, so the verdict
    says precision even though this mission also exceeds the bus and the
    clock."""
    fit = _fit(workload, H64, "air superiority")
    assert fit.unrunnable_stages and fit.out_of_contract_stages
    assert fit.dram_ratio > 1.0 and fit.oversubscription > 1.0
    assert fit.binding_constraint == "precision"


# ---------------------------------------------------------------------------
# Finding 2: a function core is rated for a configuration
# ---------------------------------------------------------------------------


def test_the_sgm_core_is_a_thirty_fps_part(workload):
    """Air superiority runs stereo at 40 fps; the core is rated to 30. Far
    flight runs 20 and is inside the contract."""
    fast = _fit(workload, H64, "air superiority")
    sgm_fast = next(s for s in fast.stages if s.key == "sgm")
    assert [v.limit for v in sgm_fast.violations] == ["max_fps"]
    assert (sgm_fast.violations[0].required, sgm_fast.violations[0].allowed) == (40.0, 30.0)

    slow = _fit(workload, H64, "far flight")
    sgm_slow = next(s for s in slow.stages if s.key == "sgm")
    assert sgm_slow.violations == ()
    assert sgm_slow.execution == "absorbed"
    # 1440x1080 at 20 fps is 31.1 Mpx/s against the core's 173.9 Mpx/s.
    assert sgm_slow.occupancy == pytest.approx(31104000 / 173850000, rel=1e-6)


@pytest.mark.parametrize("regime", ["far flight", "air superiority"])
def test_the_navion_core_is_a_752x480_part(workload, regime):
    """Every drone mission runs VIO at 1440x1080, which is 4.3x the pixels the
    Navion-class core is rated for. Throughput alone would have said yes: 20
    frames of 540."""
    fit = _fit(workload, H64, regime)
    vio = next(s for s in fit.stages if s.key == "vio")
    assert {v.limit for v in vio.violations} == {"max_width", "max_height"}
    assert vio.units_required_per_s < vio.units_available_per_s
    assert vio in fit.out_of_contract_stages


def test_a_core_within_contract_absorbs_its_stage(workload):
    """The ISP core states no configuration limits and is far from saturated:
    62.2 Mpx/s of mono front-end against 475 Mpx/s."""
    fit = _fit(workload, H64, "air superiority")
    mono = next(s for s in fit.stages if s.key == "mono")
    assert mono.execution == "absorbed" and mono.violations == ()
    assert mono.engines == ("ISP",)
    assert mono.work_unit == "pixel"
    assert mono.occupancy == pytest.approx(62208000 / 475000000, rel=1e-6)
    assert mono.watts and mono.watts > 0


# ---------------------------------------------------------------------------
# Finding 3: absorbed traffic stays off the bus
# ---------------------------------------------------------------------------


def test_absorbed_stages_keep_their_traffic_off_the_dram_bus(workload):
    """The stream links between ISP, SGM and VIO keep an absorbed stage's
    bytes on chip, which is the heterogeneous die's real advantage on the
    bandwidth-bound regime."""
    hetero = _fit(workload, H64, "far flight")
    uniform = _fit(workload, T64, "far flight")
    assert hetero.dram_demand_gb_per_s < uniform.dram_demand_gb_per_s
    saved = uniform.dram_demand_gb_per_s - hetero.dram_demand_gb_per_s
    assert saved == pytest.approx(32.8, abs=0.5)
    # Both still exceed the same 64 GB/s LPDDR5 bus by a wide margin, so this
    # is an improvement, not a fix.
    assert hetero.dram_available_gb_per_s == uniform.dram_available_gb_per_s == 64.0
    assert hetero.dram_ratio > 4.0


# ---------------------------------------------------------------------------
# The machinery
# ---------------------------------------------------------------------------


def test_achieved_to_peak_scales_occupancy(workload):
    """The assumption that dominates every compute verdict, so it is a
    parameter: 5% of peak is 2.5x the throughput of 2%."""
    realistic = _fit(workload, T64, "air superiority", REALISTIC_ACHIEVED_TO_PEAK)
    optimistic = _fit(workload, T64, "air superiority", OPTIMISTIC_ACHIEVED_TO_PEAK)
    ratio = OPTIMISTIC_ACHIEVED_TO_PEAK / REALISTIC_ACHIEVED_TO_PEAK
    assert realistic.oversubscription == pytest.approx(
        optimistic.oversubscription * ratio, rel=1e-9
    )
    # Even at 5% the sizing regime does not close on the T64.
    assert optimistic.oversubscription > 1.0


def test_a_small_mission_is_feasible(workload):
    """A sanity check that the model can say yes: the smallest edge profile
    fits the T64 with room, on compute and on bandwidth."""
    fit = _fit(workload, T64, "Event detection & classification")
    assert fit.feasible
    assert fit.binding_constraint == "none"
    assert fit.oversubscription < 1.0
    assert fit.dram_ratio < 1.0


def test_every_stage_is_accounted_for(workload):
    """No stage is silently dropped: each is absorbed, programmable, or named
    as unrunnable."""
    for die_id in (T64, H64):
        for profile in workload.profiles:
            fit = _fit(workload, die_id, profile.name)
            assert len(fit.stages) == len(workload.demands(profile))
            for stage_fit in fit.stages:
                assert stage_fit.execution in {"absorbed", "programmable", "infeasible"}
                if stage_fit.execution == "programmable":
                    assert stage_fit.engines
                if stage_fit.execution == "absorbed":
                    assert stage_fit.units_available_per_s is not None
