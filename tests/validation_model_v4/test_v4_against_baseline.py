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
# Floor: Jetson Orin Nano 8GB (Super; Phase B baseline + #94 first-pass calib)
# ---------------------------------------------------------------------------
#
# These floors reflect the post-#94 calibration: per-SM Tensor Core
# ops/clock corrected from 64 -> 128 MACs/TC (Ampere SM 8.6 spec),
# peak_bandwidth from 68 -> 102 GB/s (Super memory), and a new MAXN
# thermal profile reflecting the user's full-power deployment.
#
# Pass-rate evolution (Phase B baseline captured at 15W on Super):
#   matmul: 4/48 regime, 3/48 latency, 13/48 energy  (PR #90 baseline)
#        -> 4/48 regime, 5/48 latency, 21/48 energy  (post-#94 fix)
#   linear: 1/46 regime, 1/46 latency,  8/46 energy  (PR #90 baseline)
#        -> 1/46 regime, 1/46 latency, 29/46 energy  (post-#94 fix)
#
# Energy bumped substantially because static_energy = avg_power *
# latency, and latency is now closer to measured. Regime/latency
# barely moved because the regime classifier still calls every
# non-launch-bound shape DRAM_BOUND; the bw_efficiency_scale GPU
# branch needs further calibration to fix that (next round of #91).


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
    """Post measurement-priority sweep augmentation floor: 42/48.
    Earlier (analytical-classifier era) was 4/48. The augmenter
    rewrites sweep regime labels from baseline measurements when the
    measurement is concrete (ALU/DRAM/LAUNCH); 38 matmul shapes
    moved from analytical DRAM_BOUND to measurement-truth ALU_BOUND.
    See validation/model_v4/sweeps/_augment_from_baseline.py."""
    total, passes_regime, _, _ = _validation_pass_rate("matmul", _JETSON)
    assert passes_regime >= 40, (
        f"jetson matmul pass_regime regressed below floor: "
        f"{passes_regime}/{total} (floor: 40, was 42 after sweep "
        f"measurement-priority augmentation). Likely root cause: the "
        f"sweep regime labels drifted from baseline measurements -- "
        f"re-run "
        f"`python -m validation.model_v4.sweeps._augment_from_baseline`"
        f" to refresh, OR the GPU compute_efficiency_overrides_by_op "
        f"on jetson_orin_nano_8gb.py was reverted."
    )


def test_jetson_orin_nano_matmul_pass_latency_floor():
    """Post sweep-augmentation floor: 12/48 (legacy memory path).
    Compute model derate (PR #118) fits matmul to scale 0.70, which
    lifts pass_latency to 26 on the legacy memory path -- but the
    sweep-augmentation reclassifies many shapes from DRAM_BOUND
    (25% latency tolerance) to ALU_BOUND (10% tolerance), tightening
    the band. The 12/48 floor is the post-tightening number; further
    gains require the per-shape compute calibration (analog of CPU
    #67) to land predictions within the 10% ALU band."""
    total, _, passes_latency, _ = _validation_pass_rate("matmul", _JETSON)
    assert passes_latency >= 10, (
        f"jetson matmul pass_latency regressed below floor: "
        f"{passes_latency}/{total} (floor: 10, was 12 after sweep "
        f"augmentation). Likely root cause: GPU compute_efficiency_"
        f"overrides_by_op for matmul drifted, or the MAXN thermal "
        f"profile sustained_clock_hz changed."
    )


def test_jetson_orin_nano_matmul_pass_energy_floor():
    """Post sweep-augmentation floor: 14/48 (legacy memory path).
    Earlier was 24/48. The drop is a tolerance artifact, not a
    prediction regression: shapes that used to validate against the
    DRAM_BOUND energy band (30% tolerance) now validate against the
    ALU_BOUND band (15%), and many fall in the 15-30% bucket."""
    total, _, _, passes_energy = _validation_pass_rate("matmul", _JETSON)
    assert passes_energy >= 12, (
        f"jetson matmul pass_energy regressed below floor: "
        f"{passes_energy}/{total} (floor: 12, was 14 after sweep "
        f"measurement-priority augmentation, was 24 pre-augmentation "
        f"with the wider DRAM_BOUND tolerance)."
    )


def test_jetson_orin_nano_linear_validation_records_loaded():
    total, _, _, _ = _validation_pass_rate("linear", _JETSON)
    assert total == 46, (
        f"expected 46 jetson_orin_nano_8gb linear validation records, "
        f"got {total}"
    )


def test_jetson_orin_nano_linear_pass_regime_floor():
    """Post sweep-augmentation floor: 37/46. Earlier was 1/46. The
    augmenter rewrites sweep regime labels from baseline measurements
    when the measurement is concrete; 36 linear shapes moved from
    analytical DRAM_BOUND to measurement-truth ALU_BOUND. See
    validation/model_v4/sweeps/_augment_from_baseline.py."""
    total, passes_regime, _, _ = _validation_pass_rate("linear", _JETSON)
    assert passes_regime >= 35, (
        f"jetson linear pass_regime regressed below floor: "
        f"{passes_regime}/{total} (floor: 35, was 37 after sweep "
        f"measurement-priority augmentation). Likely root cause: the "
        f"sweep regime labels drifted from baseline measurements -- "
        f"re-run "
        f"`python -m validation.model_v4.sweeps._augment_from_baseline`."
    )


def test_jetson_orin_nano_linear_pass_latency_floor():
    """Post sweep-augmentation floor: 9/46 (legacy memory path).
    Compute model derate (PR #118) fits linear to scale 0.94, which
    raises pass_latency to 25 on the legacy memory path before the
    augmentation. After augmentation, many shapes move from
    DRAM_BOUND (25% tolerance) to ALU_BOUND (10%); 9 records still
    pass the tighter band."""
    total, _, passes_latency, _ = _validation_pass_rate("linear", _JETSON)
    assert passes_latency >= 7, (
        f"jetson linear pass_latency regressed below floor: "
        f"{passes_latency}/{total} (floor: 7, was 9 after sweep "
        f"augmentation)."
    )


def test_jetson_orin_nano_linear_pass_energy_floor():
    """Post sweep-augmentation floor: 18/46 (legacy memory path).
    Earlier was 28/46. The drop is a tolerance artifact, not a
    prediction regression: shapes that used to validate against the
    DRAM_BOUND energy band (30% tolerance) now validate against the
    ALU_BOUND band (15%), and many fall in the 15-30% bucket."""
    total, _, _, passes_energy = _validation_pass_rate("linear", _JETSON)
    assert passes_energy >= 16, (
        f"jetson linear pass_energy regressed below floor: "
        f"{passes_energy}/{total} (floor: 16, was 18 after sweep "
        f"measurement-priority augmentation, was 28 pre-augmentation "
        f"with the wider DRAM_BOUND tolerance)."
    )
