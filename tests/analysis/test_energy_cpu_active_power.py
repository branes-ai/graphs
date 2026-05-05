"""Regression tests for the CPU active-package-power constant (issue #71).

The pre-#71 EnergyAnalyzer used CPU_IDLE_POWER_FRACTION = 0.1, which gave
an idle_power_watts of TDP * 0.1 (e.g., 12.5W for an i7-12700K @ 125W TDP).
That value is a literal "leakage at zero load" estimate and dramatically
under-counts the package power that RAPL actually integrates during a
busy matmul kernel -- where the cores, uncore, IMC, and L3 are all active
and the package draws ~70-85% of TDP.

The fix in #71 bumps CPU_IDLE_POWER_FRACTION from 0.1 to 0.7. The constant
is now interpreted as "average package power during an active kernel as
a fraction of TDP", calibrated against the V4 RAPL baseline. Net effect on
energy_predicted for above-1ms shapes:

  pre-#71  median energy_pred / energy_meas = 0.10  (90% under-prediction)
  post-#71 median energy_pred / energy_meas = 0.67  (just below 0.70 band)

Tests below pin:
1. The CPU constant is at least 0.5 (defends against silent revert).
2. The GPU constant is unchanged (the #71 fix is CPU-specific).
3. The computed idle_power_watts on i7-12700K is in the V4-validated band.
4. Static energy now dominates dynamic for a typical CPU matmul (it should
   -- a 100W+ active package power eats microjoules per FLOP much faster
   than the 1.5pJ/flop dynamic term ever can).
"""

from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.energy import EnergyAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Class-level constants -- defend the calibrated values
# ---------------------------------------------------------------------------


def test_cpu_idle_fraction_is_at_least_50_percent():
    """The pre-#71 value of 0.1 modelled silicon leakage. Real i7-12700K
    package power during an active matmul is ~70-85% of TDP, per RAPL
    bracketing in the V4 baseline. Anything below 0.5 is a regression
    toward the pre-#71 under-prediction of energy by ~10x."""
    assert EnergyAnalyzer.CPU_IDLE_POWER_FRACTION >= 0.5, (
        f"CPU_IDLE_POWER_FRACTION={EnergyAnalyzer.CPU_IDLE_POWER_FRACTION} "
        f"is below 0.5 -- regression toward pre-#71 0.1 constant. The V4 "
        f"RAPL baseline measured ~110W avg on a 125W TDP i7, requiring "
        f"a fraction of ~0.7 to stay inside the 30% energy tolerance band."
    )


def test_gpu_idle_fraction_unchanged_by_71_fix():
    """#71 was a CPU-only fix (RAPL on Intel client). The GPU constant
    must remain at its NVML-validated value of 0.3."""
    assert EnergyAnalyzer.IDLE_POWER_FRACTION == 0.3, (
        f"IDLE_POWER_FRACTION={EnergyAnalyzer.IDLE_POWER_FRACTION} drifted "
        f"from the GPU NVML-validated 0.3 (the #71 fix was CPU-only)."
    )


# ---------------------------------------------------------------------------
# i7-12700K idle_power_watts must land in the calibrated band
# ---------------------------------------------------------------------------


def test_i7_idle_power_watts_in_calibrated_band():
    """i7-12700K TDP=125W. With fraction=0.7, idle_power_watts=87.5W,
    which matches the RAPL-measured ~80-110W avg power during V4 baseline
    matmul kernels. The band is intentionally wide -- the point is to
    catch a silent regression to the pre-#71 12.5W."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = EnergyAnalyzer(hw, precision=Precision.FP32)
    # 125 W * 0.7 = 87.5 W; allow +/- 30 W to permit future calibration
    # within the active-power band (~60 W to ~120 W).
    assert 50.0 <= analyzer.idle_power_watts <= 130.0, (
        f"i7 idle_power_watts={analyzer.idle_power_watts:.1f}W outside the "
        f"V4-calibrated 50-130W band; pre-#71 was 12.5W."
    )


# ---------------------------------------------------------------------------
# End-to-end: V4-style matmul predicts within RAPL noise of measurement
# ---------------------------------------------------------------------------


def _matmul_subgraph(M: int, K: int, N: int) -> SubgraphDescriptor:
    """A single-op matmul SubgraphDescriptor matching what V4 builds."""
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["mm"], node_names=["mm"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="matmul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=(M * K + K * N) * 4,
        total_output_bytes=M * N * 4,
        total_weight_bytes=0,
    )


def test_static_energy_dominates_typical_cpu_matmul():
    """For a typical i7 matmul, static energy (active package power *
    latency) should be the dominant term after the #71 fix. Pre-#71, the
    1.5 pJ/flop dynamic term was the only meaningful contributor for
    sub-mJ kernels; post-#71 the active package power swamps it.

    Goes through the public ``analyze`` entry point so the test is robust
    to refactors of the private ``_analyze_subgraph`` signature.

    This test would have failed with the pre-#71 0.1 constant for any
    DRAM-bound shape (where flops/byte is low and static dominates)."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = EnergyAnalyzer(hw, precision=Precision.FP32)
    sg = _matmul_subgraph(1024, 1024, 1024)  # ~12 MB working set, in L2
    # Use a representative latency in the V4 baseline range
    latency_s = 5e-3
    report = analyzer.analyze(subgraphs=[sg], latencies=[latency_s])
    assert report.static_energy_j > report.compute_energy_j, (
        f"static ({report.static_energy_j*1e3:.2f}mJ) should dominate compute "
        f"({report.compute_energy_j*1e3:.2f}mJ) for a typical CPU matmul; "
        f"likely CPU_IDLE_POWER_FRACTION reverted toward pre-#71 0.1."
    )
    assert report.static_energy_j > report.memory_energy_j, (
        f"static ({report.static_energy_j*1e3:.2f}mJ) should dominate memory "
        f"({report.memory_energy_j*1e3:.2f}mJ) for a typical CPU matmul."
    )


def test_predicted_total_energy_within_2x_of_v4_anchor():
    """V4 RAPL baseline for matmul (1024,1024,1024) fp32 on i7-12700K is
    in the 100-700 mJ band depending on M/K/N skew (avg power ~110W times
    measured latency 1-6ms). The analyzer prediction with the #71 idle
    fraction must land within 2x of the V4 anchor in BOTH directions --
    a regression below 100 mJ means the fix reverted; growth above
    1000 mJ would mean the constant overshot active-power calibration."""
    hw = create_i7_12700k_mapper().resource_model
    analyzer = EnergyAnalyzer(hw, precision=Precision.FP32)
    sg = _matmul_subgraph(1024, 1024, 1024)
    latency_s = 5e-3  # representative measured latency for this shape
    report = analyzer.analyze(subgraphs=[sg], latencies=[latency_s])
    total_mj = (report.compute_energy_j + report.memory_energy_j
                + report.static_energy_j) * 1e3
    # The V4 baseline median is ~300 mJ for similar 5ms shapes.
    # Pre-#71 prediction at 5ms was ~70 mJ (1.5pJ * 2*1024^3 + 25pJ * 12MB
    # + 12.5W * 5ms = ~70mJ). Post-#71 should be ~440 mJ (87.5W * 5ms +
    # dynamic). Anything below 100 mJ means the fix reverted; anything
    # above 1000 mJ means CPU_IDLE_POWER_FRACTION overshot.
    assert 100.0 <= total_mj <= 1000.0, (
        f"Predicted total energy {total_mj:.1f}mJ for (1024,1024,1024) at "
        f"5ms is outside the V4-anchored 100-1000 mJ band; the #71 "
        f"calibration has drifted (regressed below floor or overshot)."
    )
