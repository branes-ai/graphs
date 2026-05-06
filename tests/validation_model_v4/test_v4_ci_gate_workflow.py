"""Meta-tests for the V4 model-validation CI gate (V4-6).

The gate is a separate workflow file (`.github/workflows/v4-validation.yml`)
intentionally disjoint from the catch-all `CI` workflow so that:
  * doc-only PRs don't trigger it (path filter)
  * it surfaces as a distinct check in the PR status rollup

These tests pin the contract:
  1. The workflow file exists and parses as YAML.
  2. It triggers on pull_request to main with a path filter.
  3. The path filter covers every code area whose drift the V4 plan
     says the gate must catch (estimation, hardware, transform, core,
     plus the validation harness and its tests).
  4. The job actually runs the V4 test directories the gate is supposed
     to enforce.

If a future contributor narrows the path filter or removes one of the
test invocations, these tests fire with a clear message pointing at
which V4 contract regressed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/validation_model_v4/test_v4_ci_gate_workflow.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "v4-validation.yml"


@pytest.fixture(scope="module")
def workflow():
    """Parse the V4 gate workflow once per module."""
    yaml = pytest.importorskip("yaml")
    assert WORKFLOW_PATH.exists(), (
        f"V4 CI gate workflow missing at {WORKFLOW_PATH.relative_to(REPO_ROOT)}; "
        f"V4-6 requires this file. Re-add it from git history if accidentally deleted."
    )
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists()


def test_workflow_triggers_on_pull_request_to_main(workflow):
    """The gate must run on every PR to main (with the path filter)."""
    # PyYAML parses the unquoted YAML key `on:` as the boolean True,
    # so check both keys for compatibility.
    on = workflow.get("on") or workflow.get(True)
    assert on is not None, "workflow has no `on:` block"
    assert "pull_request" in on
    pr = on["pull_request"]
    assert pr.get("branches") == ["main"], (
        f"V4 gate must target PRs to main; got branches={pr.get('branches')}"
    )


def test_workflow_path_filter_covers_required_areas(workflow):
    """Per the v4 plan, the gate must trigger on PRs touching
    estimation/, hardware/, transform/. Plus it must rerun when the
    harness or its tests change."""
    on = workflow.get("on") or workflow.get(True)
    paths = set(on["pull_request"].get("paths", []))
    must_have = {
        "src/graphs/estimation/**",
        "src/graphs/hardware/**",
        "src/graphs/transform/**",
        "validation/model_v4/**",
        "tests/validation_model_v4/**",
    }
    missing = must_have - paths
    assert not missing, (
        f"V4 gate path filter is missing required areas: {sorted(missing)}. "
        f"Per docs/plans/validation-harness-v4-plan.md V4-6, the gate must "
        f"trigger on PRs touching every area where the analytical model "
        f"or its tests can drift."
    )


def test_workflow_runs_v4_harness_tests(workflow):
    """The job must actually invoke pytest on tests/validation_model_v4/."""
    jobs = workflow.get("jobs", {})
    assert "v4-validation" in jobs, "expected job `v4-validation`"
    steps = jobs["v4-validation"].get("steps", [])
    run_blocks = [s.get("run", "") for s in steps if "run" in s]
    combined = "\n".join(run_blocks)
    assert "tests/validation_model_v4/" in combined, (
        "V4 gate must run pytest against tests/validation_model_v4/; "
        "found run steps did not reference that path"
    )


def test_workflow_runs_v4_anchored_cpu_regression_tests(workflow):
    """The four CPU regression test files (locked in by #67/#69/#71/#74)
    must be in the gate. They encode the V4 calibration floors -- a
    regression there is the canonical 'real drift' the gate must catch."""
    jobs = workflow.get("jobs", {})
    steps = jobs["v4-validation"].get("steps", [])
    combined = "\n".join(s.get("run", "") for s in steps if "run" in s)
    must_run = [
        "tests/analysis/test_roofline_cpu_efficiency.py",       # #67
        "tests/analysis/test_roofline_cpu_dispatch_floor.py",   # #69
        "tests/analysis/test_roofline_cpu_bw_efficiency.py",    # #74
        "tests/analysis/test_energy_cpu_active_power.py",       # #71
    ]
    missing = [p for p in must_run if p not in combined]
    assert not missing, (
        "V4 gate must invoke the V4-anchored CPU regression tests; "
        f"missing: {missing}"
    )


def test_workflow_job_name_is_distinct_in_pr_status(workflow):
    """The gate's purpose is visibility -- the job's display name must
    be 'V4 Model Validation' so it's easily found in the PR check
    rollup. If a future change renames the job, this fails so the
    contributor updates branch-protection settings to match."""
    jobs = workflow.get("jobs", {})
    job = jobs.get("v4-validation", {})
    assert job.get("name") == "V4 Model Validation", (
        f"job display name drifted from 'V4 Model Validation' to "
        f"{job.get('name')!r}; if intentional, also update the branch "
        f"protection required-checks list to match."
    )
