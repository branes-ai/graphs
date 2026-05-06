"""Integration tests: V4 harness against committed per-hardware baselines.

Locks in the pass-rate floors per (hardware, op). Floors protect
against regressions, not against being too low. Improvements to the
analyzer / mapper / classifier should bump these numbers; only a
regression below the floor fails the test.

Each hardware target gets its own set of floor tests because:
* The shape population is hardware-dependent (the augmenter classifies
  shapes per-target; some shapes are LAUNCH_BOUND on flagship silicon
  but DRAM_BOUND on edge silicon).
* The analyzer constants are calibrated per-target (CPU got #67, #68,
  #69, #71, #74; Jetson Orin Nano is uncalibrated as of Phase B
  capture, tracked separately).

Targets covered today:
  * i7-12700K: matmul + linear (CPU path, #67/#68/#69/#71/#74)
  * Jetson Orin Nano 8GB: matmul + linear (Phase B baseline, uncalibrated)

Note on ``pass_energy``:
  i7: #71 calibrated CPU_IDLE_POWER_FRACTION (0.1 -> 0.7) and added
      a sub-ms RAPL-noise skip. matmul reaches 12/60 pass_energy with
      30/60 skipped; linear stays at 0/60 pass with 60/60 skipped
      because every linear shape is sub-ms.
  Jetson: NVML/INA3221 has the same sub-ms noise floor as RAPL; the
      assert_record skip applies uniformly. Post-#88 fix (workload
      moved to GPU before measuring), matmul reports 13 pass_energy
      with 10 skipped on the validation sweep.
"""

from pathlib import Path

from validation.model_v4.harness.runner import RunnerConfig, run_sweep


# tests/validation_model_v4/test_v4_against_baseline.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "validation" / "model_v4" / "sweeps"
BASELINE_DIR = REPO_ROOT / "validation" / "model_v4" / "results" / "baselines"


def _validation_pass_rate(op: str, hardware_key: str = "i7_12700k") -> tuple[int, int, int, int]:
    """Run the validation sweep on ``hardware_key``, return
    (total, passes_regime, passes_latency, passes_energy)."""
    cfg = RunnerConfig(
        sweep_path=SWEEP_DIR / f"{op}_validation.json",
        hardware_key=hardware_key,
    )
    result = run_sweep(cfg)
    total = len(result.records)
    passes_regime = sum(1 for r in result.records if r.pass_regime)
    passes_latency = sum(1 for r in result.records if r.pass_latency)
    # pass_energy is tri-state (True / False / None); only True is counted as
    # a pass. None (skipped, e.g. sub-ms RAPL per #71) and False (band miss)
    # are both excluded. Mirrors the explicit `is False` pattern used in
    # validation/model_v4/harness/report.py for failure counts.
    passes_energy = sum(1 for r in result.records if r.pass_energy)
    return total, passes_regime, passes_latency, passes_energy


def test_baseline_csvs_exist():
    """Without the committed baseline CSVs there is no validation; this
    test fails loud if the baseline ever gets removed by accident."""
    assert (BASELINE_DIR / "i7_12700k_matmul.csv").exists()
    assert (BASELINE_DIR / "i7_12700k_linear.csv").exists()
    assert (BASELINE_DIR / "jetson_orin_nano_8gb_matmul.csv").exists()
    assert (BASELINE_DIR / "jetson_orin_nano_8gb_linear.csv").exists()


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


def test_i7_matmul_pass_energy_floor():
    """Issue #71 fix: at least 10 of 60 matmul records must pass the
    per-regime energy tolerance band (was 0/60 before #71). The fix has
    two parts:
      1. CPU_IDLE_POWER_FRACTION 0.1 -> 0.7 (calibrated to V4 RAPL baseline)
      2. Skip energy assertion when measured_latency < 1 ms (RAPL noise)

    Sub-ms shapes (~30 of 60) get pass_energy=None (skipped), not False.
    Of the remaining ~30 above-1ms shapes, ~12 pass and ~18 fail -- the
    failing cohort all have under-predicted latency (separate issue);
    static_energy = active_power * latency, so when latency is wrong,
    energy is wrong by the same factor."""
    total, _, _, passes_energy = _validation_pass_rate("matmul")
    assert passes_energy >= 10, (
        f"matmul pass_energy regressed below floor: {passes_energy}/{total} "
        f"(floor: 10, was 0/60 before #71 fix). Likely root cause: "
        f"CPU_IDLE_POWER_FRACTION in EnergyAnalyzer reverted toward pre-#71 "
        f"0.1 (the literal-leakage interpretation), or the sub-ms RAPL skip "
        f"was removed from assert_record."
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
    """Linear pass_latency floor history:
    - Pre-#67/#69/#74: 10/60 (constant CPU efficiency curve was over-pessimistic)
    - Post-#74 (bw_efficiency_scale calibrated to V4 baseline): 21/60

    The fix in #74 anchored CPU bandwidth efficiency to the V4 baseline
    medians for large WS shapes, which is where #74's over-prediction
    cohort lived. The remaining ~40 failures are split between:
    - Cache-effect shapes (real workloads achieve > peak DRAM BW via
      cache hits; analyzer can't model without cache-hierarchy support)
    - Small B=1 shapes where #69's dispatch floor isn't enough"""
    total, _, passes_latency, _ = _validation_pass_rate("linear")
    assert passes_latency >= 18, (
        f"linear pass_latency regressed below floor: {passes_latency}/{total} "
        f"(floor: 18, was 21 after #74). Likely root cause: bw_efficiency_scale "
        f"CPU branch in RooflineAnalyzer reverted toward the pre-#74 constant 0.5."
    )


# ---------------------------------------------------------------------------
# Floor: Jetson Orin Nano 8GB (Phase B baseline, uncalibrated mapper)
# ---------------------------------------------------------------------------
#
# These floors reflect the analyzer's *uncalibrated* state for the
# Orin Nano mapper. The numbers are deliberately low because the mapper
# hasn't had the equivalent of i7's #67/#68/#69/#71/#74 calibration
# rounds yet -- the Phase B baseline (PR #90) just surfaced the gap.
# Calibration is tracked separately; once any of those constants land
# the floors should be bumped.
#
# Diagnostic from the baseline:
#   matmul: 4/48 regime, 3/48 latency, 13/48 energy (10 skipped sub-ms)
#   linear: 1/46 regime, 1/46 latency,  8/46 energy (10 skipped sub-ms)
#   median latency_pred / latency_meas ratio: ~4x over-prediction
#   100% of dram_bound predictions fail pass_regime -- the analyzer
#   thinks Orin Nano is DRAM-bound where the iGPU is actually doing
#   better than its conservatively-spec'd peak_bandwidth suggests.


_JETSON = "jetson_orin_nano_8gb"


def test_jetson_orin_nano_matmul_validation_records_loaded():
    """Sanity: harness produces 48 validation records for Orin Nano
    matmul (the sweep entries the augmenter classified for this key,
    excluding AMBIGUOUS / UNSUPPORTED)."""
    total, _, _, _ = _validation_pass_rate("matmul", _JETSON)
    assert total == 48, (
        f"expected 48 jetson_orin_nano_8gb matmul validation records, "
        f"got {total}; sweep augmentation may have drifted"
    )


def test_jetson_orin_nano_matmul_pass_regime_floor():
    """Phase B floor. Currently 4/48 -- mostly the launch_bound shapes.
    All 38 dram_bound predictions fail because the analyzer over-
    predicts Orin DRAM latency by ~4x (see calibration tracker)."""
    total, passes_regime, _, _ = _validation_pass_rate("matmul", _JETSON)
    assert passes_regime >= 4, (
        f"jetson matmul pass_regime regressed below floor: "
        f"{passes_regime}/{total} (floor: 4). Likely root cause: a "
        f"change to the GPU branch of _get_compute_efficiency_scale "
        f"or _get_bandwidth_efficiency_scale in RooflineAnalyzer that "
        f"made Orin predictions worse, or a peak_bandwidth change in "
        f"jetson_orin_nano_8gb mapper that shifted regime boundaries."
    )


def test_jetson_orin_nano_matmul_pass_latency_floor():
    """Phase B floor. Currently 3/48. Median latency_pred/meas is ~4x
    over-predict; the 3 passing entries are within the LAUNCH_BOUND
    30% tolerance band where the launch-overhead constant happens to
    hit. Calibrating Orin Nano's GPU efficiency curve should bump
    this substantially."""
    total, _, passes_latency, _ = _validation_pass_rate("matmul", _JETSON)
    assert passes_latency >= 3, (
        f"jetson matmul pass_latency regressed below floor: "
        f"{passes_latency}/{total} (floor: 3). Likely root cause: GPU "
        f"compute / bw efficiency curve in RooflineAnalyzer drifted, "
        f"or the analyzer's Orin Nano peak_bandwidth / peak_FLOPS "
        f"shifted in the wrong direction."
    )


def test_jetson_orin_nano_matmul_pass_energy_floor():
    """Phase B floor. Currently 13/48 with 10 sub-ms skipped, 25 fail.
    The fails track latency over-prediction since static_energy =
    avg_power * latency. Improvements to latency calibration will
    bump this."""
    total, _, _, passes_energy = _validation_pass_rate("matmul", _JETSON)
    assert passes_energy >= 10, (
        f"jetson matmul pass_energy regressed below floor: "
        f"{passes_energy}/{total} (floor: 10). Likely root cause: GPU "
        f"IDLE_POWER_FRACTION drifted (currently 0.3, V4-#71-style "
        f"calibration pending), or Orin's TDP / energy_per_flop "
        f"constants regressed."
    )


def test_jetson_orin_nano_linear_validation_records_loaded():
    total, _, _, _ = _validation_pass_rate("linear", _JETSON)
    assert total == 46, (
        f"expected 46 jetson_orin_nano_8gb linear validation records, "
        f"got {total}"
    )


def test_jetson_orin_nano_linear_pass_regime_floor():
    """Phase B floor. Currently 1/46 -- linear is even worse than
    matmul on Orin because of the tall-skinny B=1 shapes that the
    classifier puts into dram_bound but the analyzer over-predicts."""
    total, passes_regime, _, _ = _validation_pass_rate("linear", _JETSON)
    assert passes_regime >= 1, (
        f"jetson linear pass_regime regressed below floor: "
        f"{passes_regime}/{total} (floor: 1)."
    )


def test_jetson_orin_nano_linear_pass_latency_floor():
    """Phase B floor. Currently 1/46. Same root cause as matmul
    pass_latency floor."""
    total, _, passes_latency, _ = _validation_pass_rate("linear", _JETSON)
    assert passes_latency >= 1, (
        f"jetson linear pass_latency regressed below floor: "
        f"{passes_latency}/{total} (floor: 1)."
    )


def test_jetson_orin_nano_linear_pass_energy_floor():
    """Phase B floor. Currently 8/46 with 10 sub-ms skipped, 28 fail."""
    total, _, _, passes_energy = _validation_pass_rate("linear", _JETSON)
    assert passes_energy >= 6, (
        f"jetson linear pass_energy regressed below floor: "
        f"{passes_energy}/{total} (floor: 6)."
    )
