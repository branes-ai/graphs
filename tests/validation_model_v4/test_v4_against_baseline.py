"""Integration tests: V4 harness against the committed i7-12700K baseline.

Locks in the pass-rate floors after issue #67's fix. If a future change
to the analyzer (compute efficiency curve, peak FLOPS spec, energy
model, etc.) regresses single-kernel matmul predictions on i7, these
tests fail loudly with a clear pointer at the analyzer.

Pass-rate floors are deliberately conservative -- they protect against
regressions, not against being too low. Improvements to the analyzer
should bump these numbers (and that's the whole point of the V4 dev
loop). If a change makes the harness pass MORE than the floor, the
test still passes; only a regression below the floor fails.

Note: ``pass_energy`` is not asserted here. Issue #71 tracks the
systematic energy under-prediction for short ops; until that's
addressed, energy assertions cannot pass at any meaningful rate.
"""

from pathlib import Path

import pytest

from validation.model_v4.harness.runner import RunnerConfig, run_sweep


# tests/validation_model_v4/test_v4_against_baseline.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "validation" / "model_v4" / "sweeps"
BASELINE_DIR = REPO_ROOT / "validation" / "model_v4" / "results" / "baselines"


def _validation_pass_rate(op: str) -> tuple[int, int, int, int]:
    """Run the i7-12700K validation sweep, return (total, passes_regime,
    passes_latency, passes_energy)."""
    cfg = RunnerConfig(
        sweep_path=SWEEP_DIR / f"{op}_validation.json",
        hardware_key="i7_12700k",
    )
    result = run_sweep(cfg)
    total = len(result.records)
    passes_regime = sum(1 for r in result.records if r.pass_regime)
    passes_latency = sum(1 for r in result.records if r.pass_latency)
    passes_energy = sum(1 for r in result.records if r.pass_energy)
    return total, passes_regime, passes_latency, passes_energy


def test_baseline_csvs_exist():
    """Without the committed baseline CSVs there is no validation; this
    test fails loud if the baseline ever gets removed by accident."""
    assert (BASELINE_DIR / "i7_12700k_matmul.csv").exists()
    assert (BASELINE_DIR / "i7_12700k_linear.csv").exists()


# ---------------------------------------------------------------------------
# Floor: matmul on i7-12700K
# ---------------------------------------------------------------------------


def test_i7_matmul_validation_records_loaded():
    """Sanity: the harness produces 60 validation records (matches the
    sweep's i7-tagged entries)."""
    total, _, _, _ = _validation_pass_rate("matmul")
    assert total == 60, f"expected 60 i7 matmul validation records, got {total}"


def test_i7_matmul_pass_regime_floor():
    """Issue #67 fix: at least 30 of 60 matmul records must classify
    correctly (the launch_bound + l2_bound regimes work; 30 dram_bound
    shapes still drift due to #68 which is a separate issue)."""
    total, passes_regime, _, _ = _validation_pass_rate("matmul")
    assert passes_regime >= 30, (
        f"matmul pass_regime regressed below floor: {passes_regime}/{total} "
        f"(floor: 30). Likely root cause: #67 fix reverted, #68 worsened, "
        f"or the regime classifier in classify.py drifted."
    )


def test_i7_matmul_pass_latency_floor():
    """Issue #67 fix: at least 30 of 60 matmul records must be within
    the per-regime latency tolerance band (was 0/60 before #67 fix).
    Improvement to >= 50 would require fixing #68 (peak FLOPS spec)."""
    total, _, passes_latency, _ = _validation_pass_rate("matmul")
    assert passes_latency >= 30, (
        f"matmul pass_latency regressed below floor: {passes_latency}/{total} "
        f"(floor: 30, was 0/60 before #67 fix). Compute efficiency curve in "
        f"RooflineAnalyzer._get_compute_efficiency_scale (CPU branch) likely "
        f"reverted toward pre-#67 values."
    )


# ---------------------------------------------------------------------------
# Floor: linear on i7-12700K (#67 didn't target this; #69 is the linear bug)
# ---------------------------------------------------------------------------


def test_i7_linear_validation_records_loaded():
    total, _, _, _ = _validation_pass_rate("linear")
    assert total == 60


def test_i7_linear_pass_regime_floor():
    """Most linear shapes classify correctly even with #69 outstanding."""
    total, passes_regime, _, _ = _validation_pass_rate("linear")
    assert passes_regime >= 40, (
        f"linear pass_regime regressed below floor: {passes_regime}/{total}"
    )


def test_i7_linear_pass_latency_floor():
    """Linear has its own under-prediction bug (#69) so the floor is
    lower than matmul -- this test guards against further regression
    from current state."""
    total, _, passes_latency, _ = _validation_pass_rate("linear")
    assert passes_latency >= 8, (
        f"linear pass_latency regressed below floor: {passes_latency}/{total}. "
        f"Even if #69 isn't fixed, this number should not get worse."
    )
