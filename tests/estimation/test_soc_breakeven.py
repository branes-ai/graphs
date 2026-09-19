"""Required efficiency: what a design would have to achieve (graphs#269 Phase 6)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from embodied_schemas import load_process_nodes

from graphs.core.pipeline_workload import CLASS_NAMES, load_autonomy_workload
from graphs.estimation.soc import (
    EfficiencyTable,
    KernelClass,
    capability_mapping,
    engines_of,
    execution_format,
    load_efficiency_tables,
    load_kernel_classes,
    required_efficiency,
    schedule,
)
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = load_autonomy_workload()
KERNELS = load_kernel_classes()
TABLES = load_efficiency_tables()
FAR, AIR = WORKLOAD.regimes()


@pytest.fixture(scope="module")
def socs():
    designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
    return {d: compose_soc(designs[d], library, nodes, "tsmc_n7")
            for d in ("orin_class_reference", "kpu_heterogeneous_h64")}


@pytest.fixture(scope="module")
def kpu(socs):
    return socs["kpu_heterogeneous_h64"]


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_the_requirement_is_the_stages_occupancy_at_dense_peak(kpu):
    """Hand-computed from the workload's ops and the design's peaks."""
    result = required_efficiency(WORKLOAD, AIR, kpu)
    engines = engines_of(kpu)
    by_stage = {s.stage: s for s in result.stages}
    for engine in result.engines:
        expected = 0.0
        for name in engine.stages:
            demand = next(d for d in WORKLOAD.demands(AIR) if d.stage.key == name)
            eng = engines[engine.engine]
            dense = sum(demand.stage.ops_per_call * share / eng.server_peak_ops_per_s(
                execution_format(cls, eng.formats))
                for cls, share in zip(CLASS_NAMES, demand.stage.class_split) if share > 0)
            assert by_stage[name].occupancy_at_peak == pytest.approx(dense * demand.rate_hz)
            expected += dense * demand.rate_hz
        assert engine.required_efficiency == pytest.approx(expected / engine.servers)


def test_at_the_required_efficiency_the_engine_is_exactly_full(kpu):
    """The definition, checked against the analyzer: give every engine the
    efficiency the requirement names and its utilization lands on 1."""
    result = required_efficiency(WORKLOAD, AIR, kpu)
    mapping = {s.stage: s.engine for s in result.stages if s.engine}
    for engine in result.engines:
        table = EfficiencyTable.model_validate(dict(
            id="at_requirement", name="t", kind="per_engine",
            entries=[dict(kernel_class=k.value, engine_kind=e, precision=p,
                          compute_eff=engine.required_efficiency, confidence="theoretical",
                          source="the requirement itself")
                     for k in KernelClass for e in ("cpu", "gpu", "npu", "kpu")
                     for p in ("int8", "fp16", "fp32", "fp64")]))
        sched = schedule(WORKLOAD, AIR, kpu, table, KERNELS, mapping)
        # Memory-bound stages are the exception: their service time is the
        # DRAM time, which no efficiency changes.
        if any(s.bound == "memory" for s in sched.served if s.engine == engine.engine):
            continue
        assert sched.engine_utilization()[engine.engine] == pytest.approx(1.0, rel=1e-9)


def test_a_requirement_above_one_is_unreachable_at_any_efficiency(kpu):
    """Every stage on the CPU complement: it would have to run at 859% of
    its dense peak, which is not an efficiency question."""
    cpu_only = {d.stage.key: "cpu" for d in WORKLOAD.demands(AIR)}
    result = required_efficiency(WORKLOAD, AIR, kpu, mapping=cpu_only)
    cpu = next(e for e in result.engines if e.engine == "cpu")
    assert cpu.required_efficiency > 1.0 and cpu.reachable is False
    # The KPU carries the difference in the design's own mapping.
    shared = required_efficiency(WORKLOAD, AIR, kpu)
    assert next(e for e in shared.engines if e.engine == "cpu").required_efficiency \
        < cpu.required_efficiency


def test_memory_alone_can_put_a_stage_out_of_reach(kpu):
    """vlm in far flight moves more bytes per call than the sustained
    bandwidth delivers in its period: no efficiency helps."""
    result = required_efficiency(WORKLOAD, FAR, kpu)
    vlm = next(s for s in result.stages if s.stage == "vlm")
    assert vlm.memory_occupancy > 1.0 and vlm.out_of_reach
    engine = next(e for e in result.engines if "vlm" in e.stages)
    assert engine.out_of_reach == ("vlm",) and engine.reachable is False


# ---------------------------------------------------------------------------
# The capability rule
# ---------------------------------------------------------------------------


def test_capability_keeps_class_c_off_an_engine_without_fp32(kpu):
    demands = list(WORKLOAD.demands(AIR))
    mapping = capability_mapping(demands, engines_of(kpu))
    for demand in demands:
        c_share = demand.stage.class_split[CLASS_NAMES.index("C")]
        assert mapping[demand.stage.key] == ("cpu" if c_share > 0 else "kpu"), demand.stage.key


def test_capability_prefers_the_accelerator_over_the_cpu(socs):
    demands = list(WORKLOAD.demands(AIR))
    mapping = capability_mapping(demands, engines_of(socs["orin_class_reference"]))
    # Orin's GPU has every format the classes floor at, so it takes them all.
    assert set(mapping.values()) == {"gpu_sm"}


def test_a_class_the_mapped_engine_cannot_run_is_a_stated_gap(kpu):
    c_stage = next(d.stage.key for d in WORKLOAD.demands(AIR)
                   if d.stage.class_split[CLASS_NAMES.index("C")] > 0)
    result = required_efficiency(WORKLOAD, AIR, kpu, mapping={c_stage: "kpu"})
    stage = next(s for s in result.stages if s.stage == c_stage)
    assert stage.gap and "cannot run every class" in stage.gap
    assert stage.occupancy_at_peak is None
    assert next(e for e in result.engines if e.engine == "kpu").required_efficiency is None


def test_a_stage_no_engine_runs_is_named(kpu):
    c_stage = next(d.stage.key for d in WORKLOAD.demands(AIR)
                   if d.stage.class_split[CLASS_NAMES.index("C")] > 0)
    result = required_efficiency(WORKLOAD, AIR, kpu, mapping={c_stage: None})
    assert c_stage in result.unrunnable


# ---------------------------------------------------------------------------
# Beside a table's own figures
# ---------------------------------------------------------------------------


def test_measured_efficiencies_sit_beside_the_requirement(kpu):
    result = required_efficiency(WORKLOAD, AIR, kpu, table=TABLES["orin_nano_measured_v1"],
                                 kernels=KERNELS)
    assert result.comparison_table == "orin_nano_measured_v1"
    kpu_engine = next(e for e in result.engines if e.engine == "kpu")
    # Nothing has measured a KPU kernel, so its stages are all unpriced.
    assert kpu_engine.utilization_at_known is None
    assert set(kpu_engine.unpriced) == set(kpu_engine.stages)
    cpu = next(e for e in result.engines if e.engine == "cpu")
    priced = [s for s in result.stages if s.engine == "cpu" and s.known_efficiency]
    assert priced and cpu.known_is_lower_bound
    assert cpu.utilization_at_known == pytest.approx(
        sum(s.occupancy_at_peak / s.known_efficiency for s in priced) / cpu.servers)


def test_without_a_table_no_stage_claims_a_known_efficiency(kpu):
    result = required_efficiency(WORKLOAD, AIR, kpu)
    assert result.comparison_table is None
    assert all(s.known_efficiency is None for s in result.stages)
    assert all(e.utilization_at_known is None for e in result.engines)


def test_confidence_names_a_provisional_clock(kpu, socs):
    at_n7 = required_efficiency(WORKLOAD, AIR, kpu)
    assert at_n7.estimation_confidence.level.value == "unknown"
    assert "clock is provisional" in at_n7.estimation_confidence.source
    designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
    own = compose_soc(designs["orin_class_reference"], library, nodes, None)
    at_own = required_efficiency(WORKLOAD, AIR, own)
    assert at_own.estimation_confidence.level.value == "theoretical"
    assert "workload unit costs" in at_own.estimation_confidence.source


def test_a_bad_sustained_fraction_is_an_error(kpu):
    with pytest.raises(ValueError, match="sustained_fraction"):
        required_efficiency(WORKLOAD, AIR, kpu, sustained_fraction=0.0)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _cli(*args, expect=0):
    result = subprocess.run([sys.executable, "cli/analyze_required_efficiency.py", *args],
                            capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == expect, result.stderr
    return result


def test_cli_reports_the_requirement_and_the_measured_comparison():
    out = _cli("--design", "kpu_heterogeneous_h64", "--regime", "air superiority",
               "--node", "tsmc_n7", "--efficiency", "orin_nano_measured_v1").stdout
    assert "needs" in out and "of dense peak" in out
    assert "orin_nano_measured_v1 prices none of its stages" in out  # the KPU
    assert "LOWER BOUND" in out  # the CPU, priced on some stages only


def test_cli_json_carries_the_engine_requirements(tmp_path):
    path = tmp_path / "need.json"
    _cli("--design", "kpu_heterogeneous_h64", "--regime", "far flight",
         "--mapping", "capability", "--output", str(path))
    payload = json.loads(path.read_text())
    assert len(payload) == 1
    engines = {e["engine"]: e for e in payload[0]["engines"]}
    assert set(engines) == {"kpu", "cpu"}
    assert engines["kpu"]["required_efficiency"] > 0
    assert payload[0]["confidence"] in ("theoretical", "unknown")


def test_cli_rejects_an_unknown_design():
    assert "unknown design" in _cli("--design", "nope", "--regime", "far flight",
                                    expect=2).stderr


# ---------------------------------------------------------------------------
# What the #321 review changed
# ---------------------------------------------------------------------------


def test_a_mapping_that_leaves_a_stage_out_falls_back_to_capability(kpu):
    """A split assignment names several engines, so the CLI leaves it out of
    the single-engine mapping. The stage must still get the capability
    engine, not become a stage nothing runs."""
    demands = list(WORKLOAD.demands(AIR))
    partial = {d.stage.key: "cpu" for d in demands[1:]}
    left_out = demands[0].stage.key
    result = required_efficiency(WORKLOAD, AIR, kpu, mapping=partial)
    stage = next(s for s in result.stages if s.stage == left_out)
    assert stage.engine == capability_mapping(demands, engines_of(kpu))[left_out]
    assert not result.unrunnable
    # An explicit None is different: that stage is one nothing runs.
    assert required_efficiency(WORKLOAD, AIR, kpu,
                               mapping={left_out: None}).unrunnable == (left_out,)


def test_a_misassigned_stage_is_not_called_unrunnable(kpu):
    """Mapped to an engine that cannot run its classes: another engine in
    the design still can, so it is misassigned, not unrunnable."""
    c_stage = next(d.stage.key for d in WORKLOAD.demands(AIR)
                   if d.stage.class_split[CLASS_NAMES.index("C")] > 0)
    result = required_efficiency(WORKLOAD, AIR, kpu, mapping={c_stage: "kpu"})
    assert result.misassigned == (c_stage,) and result.unrunnable == ()
    assert c_stage in result.to_dict()["misassigned"]


def test_capability_ranks_engines_by_the_seconds_not_the_kind(socs):
    """The rule has no kind preference: the engine that would take the
    fewest seconds at dense peak wins."""
    from graphs.estimation.soc.breakeven import _dense_seconds

    for design, soc in socs.items():
        engines = engines_of(soc)
        demands = list(WORKLOAD.demands(AIR))
        mapping = capability_mapping(demands, engines)
        for demand in demands:
            chosen = mapping[demand.stage.key]
            times = {name: _dense_seconds(demand.stage, e) for name, e in engines.items()}
            best = min(t for t in times.values() if t is not None)
            assert times[chosen] == pytest.approx(best), (design, demand.stage.key)


def test_one_priced_format_does_not_price_a_mixed_stage(kpu):
    """A stage running INT8 and FP32 is unpriced unless the table prices
    both; the old minimum-of-known rule called it priced."""
    mixed = [d.stage for d in WORKLOAD.demands(AIR)
             if sum(1 for share in d.stage.class_split if share > 0) > 1]
    assert mixed, "the workload has a stage with more than one class"
    table = TABLES["orin_nano_measured_v1"]
    result = required_efficiency(WORKLOAD, AIR, kpu, table=table, kernels=KERNELS)
    for stage in result.stages:
        if stage.known_seconds is None or not stage.formats:
            continue
        kernel = KERNELS.of(stage.stage)
        kind = engines_of(kpu)[stage.engine].kind
        assert all(table.lookup(kernel, kind, fmt) is not None
                   and table.lookup(kernel, kind, fmt).known
                   for fmt in stage.formats.values()), stage.stage


def test_known_efficiency_is_the_dense_time_over_the_priced_time(kpu):
    result = required_efficiency(WORKLOAD, AIR, kpu, table=TABLES["orin_nano_measured_v1"],
                                 kernels=KERNELS)
    priced = [s for s in result.stages if s.known_seconds]
    assert priced
    for s in priced:
        assert s.known_efficiency == pytest.approx(s.dense_seconds / s.known_seconds)
        assert 0 < s.known_efficiency <= 1
