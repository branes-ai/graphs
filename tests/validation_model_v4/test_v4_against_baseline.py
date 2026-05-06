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
    """Post-#94 floor: still 4/48. The Tensor Core peak fix bumped the
    AI breakpoint from 11.7 to 69.6 FLOPS/byte, so most non-launch-
    bound shapes still classify as DRAM_BOUND. Resolving this needs
    further calibration of the GPU bw_efficiency_scale curve (#91)."""
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
    """Post-#94 floor: 5/48 (was 3/48 pre-fix). Median latency_pred/meas
    flipped from 4x over-prediction to 0.47x under-prediction -- the
    Tensor Core peak fix overshot for small shapes that don't saturate
    cuBLAS. Further calibration via a per-shape compute-efficiency curve
    (analog of CPU #67) is the next round of #91."""
    total, _, passes_latency, _ = _validation_pass_rate("matmul", _JETSON)
    assert passes_latency >= 4, (
        f"jetson matmul pass_latency regressed below floor: "
        f"{passes_latency}/{total} (floor: 4, was 5 after #94). Likely "
        f"root cause: GPU compute / bw efficiency curve drifted, or "
        f"the MAXN thermal profile sustained_clock_hz changed."
    )


def test_jetson_orin_nano_matmul_pass_energy_floor():
    """Post-#94 floor: 25/48 (was 13/48 pre-fix; +12 / +92%). The
    Tensor Core peak fix made predicted latency closer to measured,
    which pulls predicted static_energy = avg_power * latency closer
    to measured. The remaining 13 fails are mostly the small under-
    predicted-latency shapes."""
    total, _, _, passes_energy = _validation_pass_rate("matmul", _JETSON)
    assert passes_energy >= 20, (
        f"jetson matmul pass_energy regressed below floor: "
        f"{passes_energy}/{total} (floor: 20, was 21 after #94, was "
        f"13 pre-fix). Likely root cause: the Tensor Core peak fix in "
        f"jetson_orin_nano_8gb.py reverted (fp16_ops_per_sm_per_clock "
        f"back to 512 from 1024), or peak_bandwidth back to 68 GB/s, "
        f"or default_thermal_profile back to '7W' from '15W'."
    )


def test_jetson_orin_nano_linear_validation_records_loaded():
    total, _, _, _ = _validation_pass_rate("linear", _JETSON)
    assert total == 46, (
        f"expected 46 jetson_orin_nano_8gb linear validation records, "
        f"got {total}"
    )


def test_jetson_orin_nano_linear_pass_regime_floor():
    """Post-#94 floor: still 1/46. Linear has tall-skinny B=1 shapes
    that the classifier puts into dram_bound; the Tensor Core peak fix
    didn't move the regime classification because AI breakpoint went
    UP (more shapes still classify as DRAM-bound). Resolving needs the
    GPU bw_efficiency_scale curve calibration (next round of #91)."""
    total, passes_regime, _, _ = _validation_pass_rate("linear", _JETSON)
    assert passes_regime >= 1, (
        f"jetson linear pass_regime regressed below floor: "
        f"{passes_regime}/{total} (floor: 1)."
    )


def test_jetson_orin_nano_linear_pass_latency_floor():
    """Post-#94 floor: still 1/46. Same root cause as matmul
    pass_latency floor; small linear shapes have the same compute-
    efficiency overshoot as small matmul."""
    total, _, passes_latency, _ = _validation_pass_rate("linear", _JETSON)
    assert passes_latency >= 1, (
        f"jetson linear pass_latency regressed below floor: "
        f"{passes_latency}/{total} (floor: 1)."
    )


def test_jetson_orin_nano_linear_pass_energy_floor():
    """Post-#94 floor: 29/46 (was 8/46 pre-fix; +21 / +263%). Same root
    cause as matmul energy floor: predicted latency now closer to
    measured -> static_energy = avg_power * latency comes out closer."""
    total, _, _, passes_energy = _validation_pass_rate("linear", _JETSON)
    assert passes_energy >= 27, (
        f"jetson linear pass_energy regressed below floor: "
        f"{passes_energy}/{total} (floor: 27, was 29 after #94)."
    )
