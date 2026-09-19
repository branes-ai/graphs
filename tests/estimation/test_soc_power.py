"""Power roll-up (graphs#269 PR 3.3): priced terms, stated gaps, lower bounds."""

from __future__ import annotations

import pytest
from embodied_schemas import load_process_nodes
from embodied_schemas.process_node import CircuitClass

from graphs.core.pipeline_workload import load_autonomy_workload
from graphs.estimation.soc import (
    EfficiencyTable,
    KernelClass,
    find_mapping,
    load_efficiency_tables,
    load_kernel_classes,
    schedule,
)
from graphs.estimation.soc.power import datapath_library, roll_up
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library

WORKLOAD = load_autonomy_workload()
NODES = load_process_nodes()
TABLES = load_efficiency_tables()
KERNELS = load_kernel_classes()
DESIGN = load_designs()["orin_class_reference"]
LIBRARY = load_ip_library()
FAR, AIR = WORKLOAD.regimes()


def _soc(node="samsung_8lpp"):
    return compose_soc(DESIGN, LIBRARY, NODES, node)


def _everything_known(eff: float = 0.3) -> EfficiencyTable:
    return EfficiencyTable.model_validate(dict(
        id="all_known", name="test", kind="per_engine",
        entries=[dict(kernel_class=k.value, engine_kind=e, precision=p, compute_eff=eff,
                      confidence="theoretical", source="test")
                 for k in KernelClass for e in ("cpu", "gpu", "npu")
                 for p in ("int8", "fp16", "fp32", "fp64")]))


def _explicit():
    return find_mapping(DESIGN.id, WORKLOAD.version).assignments()


# ---------------------------------------------------------------------------
# The Orin reference: what can and cannot be priced
# ---------------------------------------------------------------------------


def test_pooled_orin_power_is_a_lower_bound_with_no_dynamic_term():
    soc = _soc()
    report = roll_up(schedule(WORKLOAD, AIR, soc, TABLES["annex_v1"]), soc, WORKLOAD)
    assert not report.complete
    assert report.dynamic.watts == 0 and report.dynamic.gaps
    # 8LPP states no DRAM I/O energy; no source prices DRAM devices.
    assert report.memory.gaps == ("samsung_8lpp states no dram_io_pj_per_byte",)
    assert report.dram_device.gaps
    # Only the priced SRAM leaks: a lower bound far under any real figure.
    assert 0 < report.leakage.watts < 0.1
    assert report.within_budget() is None
    assert report.useful_tops_per_w is None


def test_orins_gops_per_w_criterion_cannot_be_evaluated_yet():
    """Phase 3 criterion 3 compares Orin at 25 W with the measured 101.8
    GOPS/W. With the annex's pooled model no op lands on a datapath and the
    catalog has no 8LPP DRAM energy, so there is no figure to compare -- the
    report withholds one rather than dividing by leakage alone."""
    soc = _soc()
    for profile in WORKLOAD.profiles:
        report = roll_up(schedule(WORKLOAD, profile, soc, TABLES["annex_v1"]), soc, WORKLOAD)
        assert report.useful_tops_per_w is None, profile.id


def test_dram_io_energy_is_priced_where_the_node_states_it():
    soc = _soc("tsmc_n7")
    s = schedule(WORKLOAD, AIR, soc, TABLES["annex_v1"])
    report = roll_up(s, soc, WORKLOAD)
    assert report.memory.complete
    assert report.memory.watts == pytest.approx(
        s.dram_demand_gb_per_s * 1e9 * NODES["tsmc_n7"].dram_io_pj_per_byte * 1e-12)


def test_leakage_is_anchored_area_times_the_library_density():
    soc = _soc()
    report = roll_up(schedule(WORKLOAD, AIR, soc, TABLES["annex_v1"]), soc, WORKLOAD)
    leak = NODES["samsung_8lpp"].leakage_w_per_mm2
    expected = sum(line.area_mm2 * leak[line.circuit_class]
                   for b in soc.blocks for line in b.lines if line.anchored)
    assert report.leakage.watts == pytest.approx(expected)
    assert any("unanchored" in g for g in report.leakage.gaps)


# ---------------------------------------------------------------------------
# Dynamic power: an ALU floor, per engine
# ---------------------------------------------------------------------------


def test_datapath_library_is_the_one_logic_line():
    by_name = {b.name: b for b in _soc().blocks}
    assert datapath_library(by_name["gpu_sm"]) == CircuitClass.HP_LOGIC
    assert datapath_library(by_name["dla"]) == CircuitClass.BALANCED_LOGIC
    assert datapath_library(by_name["cpu"]) == CircuitClass.HP_LOGIC


def test_dynamic_power_is_the_alu_floor_of_the_mapped_ops():
    soc = _soc()
    s = schedule(WORKLOAD, AIR, soc, _everything_known(), KERNELS, _explicit())
    report = roll_up(s, soc, WORKLOAD)
    pj = NODES["samsung_8lpp"].energy_per_op_pj
    # The profile's own stage costs: air superiority overrides det and sgm.
    stages = {d.stage.key: d.stage for d in WORKLOAD.demands(AIR)}
    assert stages["det"].ops_per_call != WORKLOAD.stages["det"].ops_per_call
    expected = 0.0
    for svc in s.served:
        stage = stages[svc.stage]
        lib = datapath_library(next(b for b in soc.blocks if b.name == svc.engine)).value
        for cls, share in zip("ABC", stage.class_split):
            if share > 0:
                expected += stage.ops_per_call * share * svc.rate_hz * pj[f"{lib}:{svc.formats[cls]}"] * 1e-12
    assert report.dynamic.watts == pytest.approx(expected)
    assert report.dynamic.floor_only and not report.dynamic.complete
    by_block = {b.name: b for b in report.blocks}
    assert sum(by_block[n].dynamic_w for n in ("gpu_sm", "cpu", "dla")) == pytest.approx(expected)


def test_a_floor_makes_tops_per_w_an_upper_bound_not_a_result():
    soc = _soc("tsmc_n7")
    report = roll_up(schedule(WORKLOAD, AIR, soc, _everything_known(), KERNELS, _explicit()),
                     soc, WORKLOAD)
    assert report.useful_tops_per_w is not None
    assert report.to_dict()["useful_tops_per_w_is_upper_bound"] is True
    assert report.to_dict()["total_is_lower_bound"] is True


def test_a_format_the_node_has_no_energy_for_is_a_gap_not_a_proxy():
    """No node in the catalog states FP16 (BF16 is a different format), so a
    class that runs in FP16 cannot be priced -- it is not billed as BF16."""
    soc = _soc()
    assert all(not k.endswith(":fp16") for k in NODES["samsung_8lpp"].energy_per_op_pj)
    # The CPU template has FP16, so Class B on the CPU runs in FP16.
    s = schedule(WORKLOAD, AIR, soc, _everything_known(), KERNELS, {"det": "cpu"})
    report = roll_up(s, soc, WORKLOAD)
    assert any("hp_logic:fp16" in g for g in report.dynamic.gaps)
    assert report.useful_tops_per_w is None


# ---------------------------------------------------------------------------
# Budget and idle gating
# ---------------------------------------------------------------------------


def test_a_lower_bound_over_budget_is_a_proven_violation():
    soc = _soc()
    s = schedule(WORKLOAD, FAR, soc, _everything_known(), KERNELS, _explicit())
    report = roll_up(s, soc, WORKLOAD)
    if report.total_w > report.budget_w:
        assert report.within_budget() is False
    else:
        assert report.within_budget() is None  # incomplete, under budget: open


def test_gate_idle_zeroes_idle_engines_leakage_only():
    soc = _soc("tsmc_n7")
    everything_on_cpu = {d.stage.key: "cpu" for d in WORKLOAD.demands(AIR)}
    s = schedule(WORKLOAD, AIR, soc, _everything_known(), KERNELS, everything_on_cpu)
    on, gated = roll_up(s, soc, WORKLOAD), roll_up(s, soc, WORKLOAD, gate_idle=True)
    by_on = {b.name: b for b in on.blocks}
    by_gated = {b.name: b for b in gated.blocks}
    assert by_gated["gpu_sm"].gated and by_gated["gpu_sm"].leakage_w == 0
    assert by_on["gpu_sm"].leakage_w > 0
    assert not by_gated["cpu"].gated
    assert by_gated["system_cache"].leakage_w == by_on["system_cache"].leakage_w  # not an engine
    assert gated.leakage.watts < on.leakage.watts
    assert gated.dynamic.watts == on.dynamic.watts


def test_gate_idle_on_a_pooled_schedule_says_it_cannot():
    soc = _soc()
    report = roll_up(schedule(WORKLOAD, AIR, soc, TABLES["annex_v1"]), soc, WORKLOAD, gate_idle=True)
    assert any("pooled" in g for g in report.leakage.gaps)
    assert not any(b.gated for b in report.blocks)


def test_block_power_density_is_over_anchored_area():
    soc = _soc("tsmc_n7")
    report = roll_up(schedule(WORKLOAD, AIR, soc, _everything_known(), KERNELS, _explicit()),
                     soc, WORKLOAD)
    by_block = {b.name: b for b in report.blocks}
    gpu = by_block["gpu_sm"]
    assert gpu.w_per_mm2 == pytest.approx(gpu.watts / gpu.area_mm2)
    assert not gpu.area_complete
    assert by_block["dla"].w_per_mm2 is None  # no anchored area at all


def test_the_report_serializes():
    soc = _soc()
    out = roll_up(schedule(WORKLOAD, AIR, soc, TABLES["annex_v1"]), soc, WORKLOAD).to_dict()
    assert out["total_is_lower_bound"] is True
    assert out["useful_tops_per_w"] is None
    assert out["dram_device_w"]["gaps"]


def test_an_engine_whose_stages_are_gaps_is_not_idle():
    """Under default_v1 every stage mapped to the GPU is a gap: its
    utilization is unknown, not zero, so gating must leave it on."""
    soc = _soc("tsmc_n7")
    s = schedule(WORKLOAD, AIR, soc, TABLES["default_v1"], KERNELS, _explicit())
    assert not s.served and s.engine_utilization()["gpu_sm"] == 0.0
    gated = roll_up(s, soc, WORKLOAD, gate_idle=True)
    by = {b.name: b for b in gated.blocks}
    assert not by["gpu_sm"].gated and not by["cpu"].gated
    assert by["dla"].gated  # nothing is mapped to the DLAs at all
    assert gated.leakage.watts == pytest.approx(
        roll_up(s, soc, WORKLOAD).leakage.watts - roll_up(s, soc, WORKLOAD).blocks[
            [b.name for b in gated.blocks].index("dla")].leakage_w)


def test_an_unmapped_stage_means_no_engine_is_provably_idle():
    soc = _soc("tsmc_n7")
    s = schedule(WORKLOAD, AIR, soc, TABLES["default_v1"], KERNELS, "greedy")
    assert any(svc.engine == "" for svc in s.services)
    assert not any(b.gated for b in roll_up(s, soc, WORKLOAD, gate_idle=True).blocks)


def test_power_carries_its_own_estimation_confidence():
    """UNKNOWN whenever a term has gaps or is a floor, with that term as the
    source -- not the schedule's confidence (#306 review)."""
    soc = _soc()
    pooled = roll_up(schedule(WORKLOAD, AIR, soc, TABLES["annex_v1"]), soc, WORKLOAD)
    assert pooled.schedule.confidence.value == "theoretical"
    assert pooled.estimation_confidence.level.value == "unknown"
    assert pooled.estimation_confidence.source.startswith("power.dynamic")
    floor = roll_up(schedule(WORKLOAD, AIR, _soc("tsmc_n7"), _everything_known(), KERNELS,
                             _explicit()), _soc("tsmc_n7"), WORKLOAD)
    assert not floor.dynamic.gaps
    assert floor.estimation_confidence.level.value == "unknown"
    assert floor.estimation_confidence.source == (
        "power.dynamic: ALU-only floor, architectural overhead unpriced")
    assert pooled.to_dict()["confidence"] == "unknown"


def test_a_stages_priced_classes_count_when_another_class_is_a_gap():
    """det on the CPU: Class A in INT8 prices, Class B in FP16 does not (no
    node states FP16). The INT8 part still counts toward the floor; only
    the FP16 part is the gap (#311 review)."""
    soc = _soc()
    s = schedule(WORKLOAD, AIR, soc, _everything_known(), KERNELS, {"det": "cpu"})
    report = roll_up(s, soc, WORKLOAD)
    assert any("hp_logic:fp16" in g for g in report.dynamic.gaps)
    det = next(d.stage for d in WORKLOAD.demands(AIR) if d.stage.key == "det")
    rate = next(svc.rate_hz for svc in s.served if svc.stage == "det")
    pj = NODES["samsung_8lpp"].energy_per_op_pj["hp_logic:int8"]
    expected = det.ops_per_call * det.class_split[0] * rate * pj * 1e-12
    assert report.dynamic.watts == pytest.approx(expected)
