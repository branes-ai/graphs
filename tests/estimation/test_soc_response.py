"""Response-time schedulability (graphs#269 Phase 5)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from graphs.estimation.soc import EfficiencyTable, KernelClass, SoCAnalyzer
from graphs.estimation.soc.response import Task, _analyze_server, _partition, job_rate

REPO = Path(__file__).resolve().parents[2]


def _tasks(*ct):
    return [Task(f"t{i}", "e", c, t) for i, (c, t) in enumerate(ct)]


# ---------------------------------------------------------------------------
# The analyses, on textbook task sets
# ---------------------------------------------------------------------------


def test_preemptive_rm_matches_the_textbook_iteration():
    """(C, T) = (1, 4), (2, 6), (3, 12): R3 iterates 3, 6, 7, 9, 10, 10."""
    out = {r.task.stage: r for r in _analyze_server(_tasks((1, 4), (2, 6), (3, 12)), 0, True, "rm")}
    assert [out[k].response_s for k in ("t0", "t1", "t2")] == pytest.approx([1, 3, 10])
    assert all(r.meets for r in out.values())


def test_nonpreemptive_rm_charges_lower_priority_blocking():
    """(1, 4) high, (3, 12) low. Non-preemptive: the high task can wait out
    the low one's 3 -- Davis et al. 2007: w = max(B, C) = 3, R = 3 + 1 = 4."""
    out = {r.task.stage: r for r in _analyze_server(_tasks((1, 4), (3, 12)), 0, False, "rm")}
    assert out["t0"].blocking_s == 3 and out["t0"].response_s == pytest.approx(4)
    assert out["t0"].meets is True
    preemptive = {r.task.stage: r for r in _analyze_server(_tasks((1, 4), (3, 12)), 0, True, "rm")}
    assert preemptive["t0"].response_s == pytest.approx(1)  # no blocking when preemptive


def test_a_miss_is_reported_against_the_deadline():
    # (3, 12) would converge to exactly 12 and meet; (4, 12) iterates
    # 4, 7, 10, 13 and misses.
    out = _analyze_server(_tasks((3, 4), (4, 12)), 0, True, "rm")
    assert {r.task.stage: r.meets for r in out} == {"t0": True, "t1": False}


def test_preemptive_edf_is_exact_at_utilization_one():
    ok = _analyze_server(_tasks((2, 4), (3, 6)), 0, True, "edf")  # U = 1.0
    assert all(r.meets for r in ok) and [r.response_s for r in ok] == [4, 6]
    over = _analyze_server(_tasks((3, 4), (3, 6)), 0, True, "edf")  # U = 1.25
    assert all(r.meets is False for r in over)


def test_nonpreemptive_edf_is_open_not_guessed():
    out = _analyze_server(_tasks((1, 4), (1, 6)), 0, False, "edf")
    assert all(r.meets is None and r.response_s is None for r in out)


def test_partition_is_first_fit_decreasing_and_flags_failure():
    bins, fitted = _partition(_tasks((6, 10), (5, 10), (4, 10)), 2)
    # 0.6 first; 0.5 does not fit beside it; 0.4 does: loads 1.0 and 0.5.
    assert fitted and sorted(sum(t.utilization for t in b) for b in bins) == pytest.approx([0.5, 1.0])
    _, fitted = _partition(_tasks((6, 10), (6, 10), (6, 10)), 2)
    assert not fitted


# ---------------------------------------------------------------------------
# Jobs: a pixel is not a job
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workload():
    return SoCAnalyzer().workload


def test_a_per_pixel_stage_is_scheduled_per_frame(workload):
    air = workload.profile("air superiority")
    stage = workload.stages["mono"]
    rate = air.rates_hz["mono"]
    jobs, basis = job_rate(stage, air, rate)
    cams, w, h, fps = air.sensors["mono"][:4]
    assert jobs == cams * fps and "fps" in basis
    assert rate / jobs == w * h  # pixels per job


def test_control_is_per_update_across_dofs(workload):
    air = workload.profile("air superiority")
    jobs, _ = job_rate(workload.stages["ctrl"], air, air.rates_hz["ctrl"])
    assert jobs == air.sensors["ctrl_hz"]


def test_a_point_stream_with_no_scan_rate_has_no_job(workload):
    air = workload.profile("air superiority")
    jobs, basis = job_rate(workload.stages["lidar"], air, air.rates_hz["lidar"])
    assert jobs is None and "no scan rate" in basis


def test_a_viewpoint_is_not_a_point(workload):
    far = workload.profile("far flight")
    jobs, basis = job_rate(workload.stages["gain"], far, far.rates_hz["gain"])
    assert jobs == far.rates_hz["gain"] and basis == "one call per job"


def test_a_camera_spec_that_disagrees_with_the_rate_gives_no_job(workload):
    air = workload.profile("air superiority")
    jobs, _ = job_rate(workload.stages["mono"], air, air.rates_hz["mono"] * 2)
    assert jobs is None


# ---------------------------------------------------------------------------
# In the analyzer
# ---------------------------------------------------------------------------


def _all_known() -> EfficiencyTable:
    return EfficiencyTable.model_validate(dict(
        id="all_known", name="t", kind="per_engine",
        entries=[dict(kernel_class=k.value, engine_kind=e, precision=p, compute_eff=0.3,
                      confidence="theoretical", source="t")
                 for k in KernelClass for e in ("cpu", "gpu", "npu", "kpu")
                 for p in ("int8", "fp16", "fp32", "fp64")]))


@pytest.fixture(scope="module")
def analyzer():
    return SoCAnalyzer(tables={**SoCAnalyzer().tables, "all_known": _all_known()})


def test_a_light_profile_is_proven_schedulable(analyzer):
    result = analyzer.analyze("orin_class_reference", "edge_ai_device_event_detection__classification",
                              node="tsmc_n7", efficiency="all_known", mapping="greedy")
    assert result.response.schedulable is True
    # Power is still a floor, so the overall verdict stays open -- never True
    # on a partial proof.
    assert result.power.within_budget() is None and result.feasible() is None


def test_point_stages_get_no_verdict_and_leave_schedulability_open(analyzer):
    result = analyzer.analyze("orin_class_reference", "far flight", node="tsmc_n7",
                              efficiency="all_known", mapping="greedy")
    rs = result.response
    assert {s for s, d in rs.stage_deadline_s.items() if d is None} >= {"lidar", "lio", "tsdf"}
    assert rs.schedulable is None
    assert "no scan rate" in rs.to_dict()["stages"][[s["stage"] for s in rs.to_dict()["stages"]].index(
        "lidar")]["job"]


def test_the_upper_bound_is_never_below_the_lower_bound(analyzer):
    for profile in analyzer.workload.profiles:
        result = analyzer.analyze("orin_class_reference", profile, node="tsmc_n7",
                                  efficiency="all_known", mapping="greedy")
        rs = result.response
        for svc in result.schedule.served:
            upper, deadline = rs.stage_response_s.get(svc.stage), rs.stage_deadline_s.get(svc.stage)
            if upper is None or deadline is None:
                continue
            per_job = svc.rate_hz * deadline  # calls per job
            assert upper >= svc.t_service_s * per_job * (1 - 1e-9), (profile.id, svc.stage)


def test_a_pooled_or_incomplete_schedule_is_not_analyzed(analyzer):
    pooled = analyzer.analyze("orin_class_reference", "far flight", efficiency="annex_v1")
    assert pooled.response.analyzed is False and pooled.response.schedulable is None
    gaps = analyzer.analyze("orin_class_reference", "far flight", efficiency="default_v1")
    assert gaps.response.analyzed is False


def test_a_split_stage_sums_its_parts_and_transfer(analyzer):
    from graphs.estimation.soc import find_mapping

    mapping = find_mapping("orin_class_reference", analyzer.workload.version).assignments()
    mapping["det"] = {"A": "dla", "B": "gpu_sm"}
    result = analyzer.analyze("orin_class_reference", "air superiority", node="tsmc_n7",
                              efficiency="all_known", mapping=mapping, transfers={"det": 6.4e6})
    rs = result.response
    det = next(s for s in result.schedule.served if s.stage == "det")
    parts = [t for t in rs.tasks if t.task.stage == "det"]
    assert {t.task.engine for t in parts} == {"dla", "gpu_sm"}
    if rs.stage_response_s["det"] is not None:
        assert rs.stage_response_s["det"] >= sum(t.response_s for t in parts) + det.t_transfer_s - 1e-12


def test_bad_policy_is_an_error(analyzer):
    with pytest.raises(ValueError, match="policy"):
        analyzer.analyze("orin_class_reference", "far flight", efficiency="all_known",
                         node="tsmc_n7", mapping="greedy", policy="fifo")


def test_cli_reports_schedulability():
    result = subprocess.run(
        [sys.executable, "cli/analyze_soc.py", "--design", "orin_class_reference",
         "--regime", "far flight", "--policy", "edf"],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "schedulable (edf)" in result.stdout



def test_the_analysis_carries_its_estimation_confidence(analyzer):
    """UNKNOWN when not analyzed, a partition failed or a job is unknown;
    otherwise the schedule's own level (#312 review)."""
    pooled = analyzer.analyze("orin_class_reference", "far flight", efficiency="annex_v1").response
    assert pooled.confidence.level.value == "unknown" and "not analyzed" in pooled.confidence.source
    points = analyzer.analyze("orin_class_reference", "far flight", node="tsmc_n7",
                              efficiency="all_known", mapping="greedy").response
    assert points.confidence.level.value == "unknown"
    assert "partition failed" in points.confidence.source or "job size unknown" in points.confidence.source
    # At Orin's own node its clocks are the reference ones; at N7 they would be
    # provisional, the schedule UNKNOWN, and the analysis would inherit that.
    light = analyzer.analyze("orin_class_reference", "edge_ai_device_event_detection__classification",
                             efficiency="all_known", mapping="greedy")
    assert light.response.analyzed and not light.response.partition_failed
    assert light.response.confidence.level.value == light.schedule.confidence.value == "theoretical"
    at_n7 = analyzer.analyze("orin_class_reference", "edge_ai_device_event_detection__classification",
                             node="tsmc_n7", efficiency="all_known", mapping="greedy")
    assert at_n7.response.confidence.level.value == "unknown"
    assert light.response.confidence.source.startswith("rm upper bounds")
    assert light.to_dict()["schedulability"]["confidence"] == "theoretical"
