"""SoCAnalyzer, SoCAnalysisResult and cli/analyze_soc.py (graphs#269 PR 3.4)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from graphs.estimation.soc import EfficiencyTable, KernelClass, SoCAnalyzer

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def analyzer():
    return SoCAnalyzer()


def _everything_known() -> EfficiencyTable:
    return EfficiencyTable.model_validate(dict(
        id="all_known", name="test", kind="per_engine",
        entries=[dict(kernel_class=k.value, engine_kind=e, precision=p, compute_eff=0.3,
                      confidence="theoretical", source="test")
                 for k in KernelClass for e in ("cpu", "gpu", "npu")
                 for p in ("int8", "fp16", "fp32", "fp64")]))


# ---------------------------------------------------------------------------
# Phase 3 acceptance, as far as the data reaches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime, published, tolerance", [
    ("far flight", 16.2, 0.05), ("air superiority", 17.3, 0.05),
])
def test_annex_v1_reproduces_the_regimes_within_5_percent(analyzer, regime, published, tolerance):
    """Criterion 1: the parent plan's 5% on the two regimes that have
    per-stage data (Phase 1's axis decision)."""
    result = analyzer.analyze("orin_class_reference", regime, efficiency="annex_v1")
    assert result.schedule.oversubscription == pytest.approx(published, rel=tolerance)


@pytest.mark.parametrize("regime", ["far flight", "air superiority"])
def test_annex_v1_reproduces_the_annex_stages_over(analyzer, regime):
    """Criterion 2, against the annex model rather than the argument
    document's counts (those assume regimes without per-stage data)."""
    result = analyzer.analyze("orin_class_reference", regime)
    ref = analyzer.workload.summary(analyzer.workload.profile(regime))
    assert result.schedule.stages_over() == ref.stages_over()


def test_orins_gops_per_w_criterion_is_reported_open(analyzer):
    """Criterion 3 cannot be evaluated: TOPS/W is withheld, not guessed."""
    result = analyzer.analyze("orin_class_reference", "air superiority")
    assert result.power.useful_tops_per_w is None
    assert any("power.dynamic" in x for x in result.limited_by())


# ---------------------------------------------------------------------------
# Result semantics
# ---------------------------------------------------------------------------


def test_an_incomplete_result_is_unknown_and_says_why(analyzer):
    result = analyzer.analyze("orin_class_reference", "air superiority", efficiency="default_v1")
    assert not result.complete
    assert result.confidence.value == "unknown"
    reasons = result.limited_by()
    assert any(x.startswith("die:") for x in reasons)
    assert any(x.startswith("schedule:") for x in reasons)
    assert result.feasible() is None
    assert result.estimation_confidence.level.value == "unknown"
    assert result.estimation_confidence.source.startswith("die:")


def test_a_proven_violation_is_infeasible_despite_gaps(analyzer):
    """Far flight's DRAM demand exceeds Orin's supply whatever the gaps."""
    result = analyzer.analyze("orin_class_reference", "far flight")
    assert result.schedule.dram_utilization > 1
    assert result.feasible() is False


def test_the_result_follows_the_parent_plan_schema(analyzer):
    d = analyzer.analyze("orin_class_reference", "far flight").to_dict()
    assert {"design", "node", "profile", "regime", "confidence_summary", "die", "peak",
            "stages", "engines", "memory", "power", "summary"} <= set(d)
    assert {"feasible", "stages_over", "e2e_latency_ms", "useful_tops_per_w"} <= set(d["summary"])
    assert d["die"]["area_is_lower_bound"] is True
    dla = next(b for b in d["die"]["by_block"] if b["block"] == "dla")
    assert dla["area_mm2"] is None  # no anchored line: unknown, never zero
    assert d["peak"]["gpu_sm"]["int8"] == pytest.approx(16 * 4096 * 1.3e9)
    assert all("pipeline_tier" in s and "dram_gb_per_s" in s for s in d["stages"])
    json.dumps(d)  # serializable as is


def test_stage_rows_use_the_profiles_own_costs(analyzer):
    """Air superiority overrides det's cost; the reported DRAM rate follows."""
    result = analyzer.analyze("orin_class_reference", "air superiority")
    det = next(s for s in result.to_dict()["stages"] if s["stage"] == "det")
    stage = result.stages["det"]
    assert stage.bytes_per_call != analyzer.workload.stages["det"].bytes_per_call
    assert det["dram_gb_per_s"] == pytest.approx(stage.bytes_per_call * det["rate_hz"] / 1e9)


# ---------------------------------------------------------------------------
# Mapping choices
# ---------------------------------------------------------------------------


def test_auto_mapping_uses_the_shipped_file(analyzer):
    result = analyzer.analyze("orin_class_reference", "air superiority", efficiency="default_v1")
    assert result.schedule.mapping == "explicit"
    assert result.mapping_source.endswith("orin_class_reference__branes_7tier_v1.yaml")


def test_greedy_and_caller_mappings(analyzer):
    greedy = analyzer.analyze("orin_class_reference", "air superiority",
                              efficiency="default_v1", mapping="greedy")
    assert greedy.schedule.mapping == "greedy"
    caller = analyzer.analyze("orin_class_reference", "air superiority",
                              efficiency="default_v1", mapping={"det": "gpu_sm"})
    assert caller.mapping_source == "caller"


def test_a_mapping_file_for_another_design_is_refused(analyzer, tmp_path):
    other = tmp_path / "m.yaml"
    other.write_text("design: other\nworkload: branes_7tier_v1\nstages: {}\n")
    with pytest.raises(ValueError, match="not orin_class_reference"):
        analyzer.analyze("orin_class_reference", "air superiority",
                         efficiency="default_v1", mapping=other)


def test_an_injected_table_prices_the_whole_pipeline():
    tables = {"all_known": _everything_known()}
    result = SoCAnalyzer(tables=tables).analyze(
        "orin_class_reference", "air superiority", node="tsmc_n7", efficiency="all_known")
    assert result.schedule.complete
    assert result.power.dynamic.floor_only
    # Every stage has a service time and every format an energy (Class B's
    # FP16 on the Orin SM included), so dynamic power has no gaps. It is
    # still an ALU-only floor, so TOPS/W is an upper bound.
    assert not result.power.dynamic.gaps
    assert result.power.to_dict()["useful_tops_per_w_is_upper_bound"] is True
    assert not result.complete  # the die and the power floor still are not


def test_unknown_inputs_are_errors(analyzer):
    with pytest.raises(KeyError, match="design"):
        analyzer.analyze("nope", "air superiority")
    with pytest.raises(KeyError, match="efficiency"):
        analyzer.analyze("orin_class_reference", "air superiority", efficiency="nope")
    with pytest.raises(ValueError, match="mapping"):
        analyzer.analyze("orin_class_reference", "air superiority", mapping="bogus")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(*args):
    return subprocess.run([sys.executable, "cli/analyze_soc.py", *args],
                          capture_output=True, text=True, cwd=REPO, timeout=300)


def test_cli_text_report_leads_with_completeness():
    result = _run("--design", "orin_class_reference", "--regime", "far flight", "-v")
    assert result.returncode == 0, result.stderr
    assert "complete: NO" in result.stdout
    assert "feasible: NO" in result.stdout
    assert "LOWER BOUND" in result.stdout
    assert "withheld" in result.stdout
    assert "limited by:" in result.stdout
    assert "sgm" in result.stdout  # the -v stage table


def test_cli_writes_json_and_csv(tmp_path):
    out = tmp_path / "far.json"
    result = _run("--design", "orin_class_reference", "--regime", "far flight",
                  "-o", str(out))
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["regime"] == "far flight"
    table = tmp_path / "all.csv"
    result = _run("--design", "orin_class_reference", "--all", "-o", str(table))
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(table.open(newline="")))
    assert len(rows) == 18
    assert all(r["power_is_lower_bound"] == "True" for r in rows)


def test_cli_markdown(tmp_path):
    out = tmp_path / "air.md"
    result = _run("--design", "orin_class_reference", "-o", str(out), "-v")
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert text.startswith("## drone_interceptor_terminal_engagement")
    assert "| stage |" in text


def test_cli_bad_input_exits_2():
    result = _run("--design", "nope")
    assert result.returncode == 2, result.stderr
    result = _run("--design", "orin_class_reference", "--efficiency", "nope")
    assert result.returncode == 2, result.stderr
    result = _run("--design", "orin_class_reference", "--profile", "nope")
    assert result.returncode == 2, result.stderr


def test_cli_lists_profiles():
    result = _run("--list-profiles")
    assert result.returncode == 0
    assert "far flight" in result.stdout and "air superiority" in result.stdout


def test_cli_lists_profiles_in_the_requested_format(tmp_path):
    out = tmp_path / "profiles.json"
    result = _run("--list-profiles", "-o", str(out))
    assert result.returncode == 0, result.stderr
    rows = json.loads(out.read_text())
    assert len(rows) == 18 and {"profile", "regime", "budget_w", "deadline_ms"} <= set(rows[0])
    table = tmp_path / "profiles.csv"
    result = _run("--list-profiles", "-o", str(table))
    assert result.returncode == 0, result.stderr
    with table.open(newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 18
    md = tmp_path / "profiles.md"
    result = _run("--list-profiles", "-o", str(md))
    assert result.returncode == 0, result.stderr
    text = md.read_text()
    assert text.startswith("## Profiles (18)")
