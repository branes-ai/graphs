"""The ILP stage-to-engine mapper (graphs#269 Phase 5)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy.optimize", reason="the ILP mapper needs scipy (graphs[soc])")

from graphs.estimation.soc import EfficiencyTable, KernelClass, SoCAnalyzer  # noqa: E402
from graphs.estimation.soc.ilp import solve_assignment  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The solver core
# ---------------------------------------------------------------------------


def test_the_ilp_beats_longest_processing_time_where_it_fails():
    """Two identical engines, jobs 3,3,2,2,2. Heaviest-first (the greedy
    mapper's rule) puts 3+2+2 on one engine: max 7. The optimum splits
    3+3 | 2+2+2: max 6."""
    jobs = [3, 3, 2, 2, 2]
    answer = solve_assignment([[j, j] for j in jobs], [1, 1])
    assert answer.bottleneck == pytest.approx(6)
    assert sorted(answer.load) == pytest.approx([6, 6])


def test_servers_divide_the_load():
    answer = solve_assignment([[4.0, 4.0]], [1, 4])
    assert answer.choice == {0: 1} and answer.bottleneck == pytest.approx(1.0)


def test_an_engine_that_cannot_run_a_stage_is_never_chosen():
    answer = solve_assignment([[None, 5.0], [1.0, None]], [1, 1])
    assert answer.choice == {0: 1, 1: 0}


def test_pinning_fixes_a_stage_and_optimizes_the_rest():
    answer = solve_assignment([[1.0, 1.0], [1.0, 1.0]], [1, 1], pinned={0: 0})
    assert answer.choice == {0: 0, 1: 1}


@pytest.mark.parametrize("matrix, pinned, match", [
    ([[None, None]], None, "no engine"),
    ([[1.0, None]], {0: 1}, "cannot run it"),
])
def test_impossible_assignments_are_errors(matrix, pinned, match):
    with pytest.raises(ValueError, match=match):
        solve_assignment(matrix, [1, 1], pinned)


def test_ties_break_toward_less_total_engine_time():
    """Same bottleneck either way; the ILP picks the engine that uses less
    engine time for the stage that does not set it."""
    # Stage 0 sets the bottleneck (5) on engine 0 whatever stage 1 does;
    # stage 1 can take 1.0 on engine 1 or 3.0 on engine 2.
    answer = solve_assignment([[5.0, None, None], [None, 1.0, 3.0]], [1, 1, 1])
    assert answer.bottleneck == pytest.approx(5.0)
    assert answer.choice[1] == 1


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
    tables = SoCAnalyzer().tables
    return SoCAnalyzer(tables={**tables, "all_known": _all_known()})


@pytest.mark.parametrize("design", ["orin_class_reference", "kpu_heterogeneous_h64"])
def test_ilp_is_never_worse_than_greedy(analyzer, design):
    for profile in analyzer.workload.profiles:
        g = analyzer.analyze(design, profile, node="tsmc_n7", efficiency="all_known", mapping="greedy")
        i = analyzer.analyze(design, profile, node="tsmc_n7", efficiency="all_known", mapping="ilp")
        assert i.schedule.complete and i.schedule.mapping == "ilp"
        assert i.schedule.bottleneck[1] <= g.schedule.bottleneck[1] + 1e-9, profile.id


def test_every_priced_stage_is_on_one_engine_that_can_price_it(analyzer):
    result = analyzer.analyze("kpu_heterogeneous_h64", "far flight", efficiency="all_known", mapping="ilp")
    for svc in result.schedule.services:
        stage = analyzer.workload.stages[svc.stage]
        assert svc.served
        if stage.class_split[2] > 0:
            assert svc.engine == "cpu"  # the KPU has no FP32


def test_unpriced_stages_stay_gaps_with_every_engines_reason(analyzer):
    """Under default_v1 nothing is priced on Orin: the ILP places nothing,
    and reports the same reasons greedy does."""
    ilp = analyzer.analyze("orin_class_reference", "air superiority", efficiency="default_v1", mapping="ilp")
    greedy = analyzer.analyze("orin_class_reference", "air superiority", efficiency="default_v1", mapping="greedy")
    assert not ilp.schedule.served
    assert dict(ilp.schedule.gaps) == dict(greedy.schedule.gaps)


def test_pinned_stages_in_the_mapping_helper(analyzer):
    from graphs.estimation.soc import engines_of
    from graphs.estimation.soc.ilp import ilp_mapping
    from graphs.hardware.soc import compose_soc

    a = analyzer
    soc = compose_soc(a.designs["orin_class_reference"], a.library, a.nodes, "tsmc_n7")
    demands = a.workload.demands(a.workload.profile("air superiority"))
    out = ilp_mapping(demands, engines_of(soc), a.kernels, a.tables["all_known"], 133.0, pinned={"det": "cpu"})
    assert out["det"].engine == "cpu"
    with pytest.raises(KeyError, match="does not have"):
        ilp_mapping(demands, engines_of(soc), a.kernels, a.tables["all_known"], 133.0, pinned={"det": "tpu"})


def test_cli_accepts_ilp():
    result = subprocess.run(
        [sys.executable, "cli/analyze_soc.py", "--design", "orin_class_reference",
         "--regime", "far flight", "--efficiency", "default_v1", "--mapping", "ilp"],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "mapping ilp" in result.stdout
