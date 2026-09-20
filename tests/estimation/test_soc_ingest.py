"""Benchmark ingest and efficiency-table layering (graphs#269 PR 5.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphs.estimation.soc import (
    EfficiencyTable,
    KernelClass,
    SoCAnalyzer,
    load_efficiency_tables,
)
from graphs.estimation.soc.efficiency import resolve_layers
from graphs.hardware.soc import EngineKind

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import ingest_soc_kernel_benchmarks as ingest  # noqa: E402

GPU_FP32_PEAK_PER_CLOCK = 16 * 256  # the Orin SM template, whole GPU
CPU_FP32_PER_CORE = 16              # A78AE: 2 FMLA x 4 lanes x 2


def _result(kernel_class="dense_conv_gemm", engine="gpu", prec="fp32", attained=1e12,
            clock_hz=1.3e9, verified=True, name="gemm", status="ok", parallelism=1.0):
    cpu = engine == "cpu"
    return {"kernel_class": kernel_class, "name": name, "precision": prec, "engine_kind": engine,
            "device": "cpu" if cpu else "cuda", "shape": "2048^3",
            "ops_per_call": 1.0, "ops_rule": "-", "calls": 10, "seconds_per_call": 1.0,
            "attained_ops_per_s": attained, "status": status, "message": "",
            "clock": None if clock_hz is None else {"median_hz": clock_hz, "verified": verified},
            "cpu_parallelism": parallelism if cpu else None,
            "single_thread": (parallelism is not None and parallelism <= 1.25) if cpu else None}


def _run(*results, quick=False, hardware="jetson_orin_agx_64gb", tf32_disabled=True):
    return ("run.json", {"schema": ingest.SCHEMA, "hardware": hardware, "power_mode": "MAXN",
                         "quick": quick, "tf32_disabled": tf32_disabled, "results": list(results)})


def _build(*runs, counts=None):
    return ingest.build_table(list(runs), "orin_class_reference", "samsung_8lpp", counts or {}, "t_measured")


# ---------------------------------------------------------------------------
# Efficiency at the measured clock
# ---------------------------------------------------------------------------


def test_efficiency_is_attained_over_the_peak_at_the_measured_clock():
    attained = 0.5 * GPU_FP32_PEAK_PER_CLOCK * 1.0e9
    table, skipped = _build(_run(_result(attained=attained, clock_hz=1.0e9)))
    (entry,) = table["entries"]
    assert entry["compute_eff"] == pytest.approx(0.5)
    assert entry["confidence"] == "calibrated"
    assert table["base"] == "default_v1" and not skipped


def test_the_measured_clock_not_the_design_clock_sets_the_peak():
    """Same throughput at a lower measured clock is a higher efficiency --
    dividing by the 1.3 GHz design clock is the 2026-02 mistake."""
    attained = 0.3 * GPU_FP32_PEAK_PER_CLOCK * 0.6e9
    table, _ = _build(_run(_result(attained=attained, clock_hz=0.6e9)))
    assert table["entries"][0]["compute_eff"] == pytest.approx(0.3)


def test_an_unverified_clock_is_interpolated():
    table, _ = _build(_run(_result(attained=1e12, verified=False)))
    assert table["entries"][0]["confidence"] == "interpolated"
    assert "clock not verified" in table["entries"][0]["source"]


def test_a_cpu_result_is_per_core():
    attained = 0.4 * CPU_FP32_PER_CORE * 2.2e9
    table, _ = _build(_run(_result(kernel_class="small_qp", engine="cpu", attained=attained,
                                   clock_hz=2.2e9, name="kkt_solve")))
    assert table["entries"][0]["compute_eff"] == pytest.approx(0.4)


def test_a_count_override_matches_the_measured_hardware():
    """An Orin Nano has 8 SMs: the same throughput is twice the efficiency
    of the 16-SM design's."""
    attained = 0.25 * GPU_FP32_PEAK_PER_CLOCK * 1e9
    full, _ = _build(_run(_result(attained=attained, clock_hz=1e9)))
    nano, _ = _build(_run(_result(attained=attained, clock_hz=1e9)), counts={"gpu_sm": 8})
    assert nano["entries"][0]["compute_eff"] == pytest.approx(2 * full["entries"][0]["compute_eff"])


def test_several_kernels_give_a_median_and_a_range():
    peak = GPU_FP32_PEAK_PER_CLOCK * 1e9
    table, _ = _build(_run(_result(attained=0.2 * peak, clock_hz=1e9, name="gemm"),
                           _result(attained=0.4 * peak, clock_hz=1e9, name="conv3x3", verified=False)))
    (entry,) = table["entries"]
    assert entry["compute_eff"] == pytest.approx(0.3)
    assert entry["eff_range"] == [0.2, 0.4]
    assert entry["confidence"] == "interpolated"  # the weakest


# ---------------------------------------------------------------------------
# What is refused, and why
# ---------------------------------------------------------------------------


def test_a_format_with_no_template_peak_is_reported_not_ingested():
    _, skipped = _build(_run(_result(prec="fp64", attained=5e9)), _run(_result()))
    assert any("states no fp64 peak" in s and "5.0 GOP/s" in s for s in skipped)


def test_an_efficiency_above_one_is_refused():
    table, skipped = _build(_run(_result(attained=10 * GPU_FP32_PEAK_PER_CLOCK * 1.3e9), _result(
        kernel_class="attention_prefill", name="sdpa", attained=1e11)))
    assert any("> 1" in s for s in skipped)
    assert {e["kernel_class"] for e in table["entries"]} == {"attention_prefill"}


def test_no_clock_samples_means_no_entry():
    _, skipped = _build(_run(_result(clock_hz=None), _result(kernel_class="fft", name="fft2")))
    assert any("no clock samples" in s for s in skipped)


def test_quick_runs_are_refused():
    with pytest.raises(ValueError, match="quick"):
        _build(_run(_result(), quick=True))


def test_unsupported_and_error_results_are_reported():
    _, skipped = _build(_run(_result(status="unsupported"), _result(kernel_class="fft", name="fft2")))
    assert any("unsupported" in s for s in skipped)


def test_nothing_ingestible_is_an_error():
    with pytest.raises(ValueError, match="no result"):
        _build(_run(_result(status="error")))


# ---------------------------------------------------------------------------
# Layering, and what it buys the analyzer
# ---------------------------------------------------------------------------


def _layered(entries):
    tables = {k: v for k, v in load_efficiency_tables().items() if k in ("default_v1", "annex_v1")}
    tables["t_measured"] = EfficiencyTable.model_validate(
        {"id": "t_measured", "name": "t", "kind": "per_engine", "base": "default_v1", "entries": entries})
    return resolve_layers(tables)


def test_a_layered_table_overrides_and_inherits():
    measured = {"kernel_class": "dense_conv_gemm", "engine_kind": "gpu", "precision": "int8",
                "compute_eff": 0.4, "confidence": "calibrated", "source": "test run"}
    table = _layered([measured])["t_measured"]
    got = table.lookup(KernelClass.DENSE_CONV_GEMM, EngineKind.GPU, "int8")
    assert got.compute_eff == 0.4 and got.confidence.value == "calibrated"
    inherited = table.lookup(KernelClass.DENSE_CONV_GEMM, EngineKind.NPU, "int8")
    assert inherited.compute_eff == pytest.approx(0.0226)  # default_v1's DLA entry
    assert table.lookup(KernelClass.SMALL_QP, EngineKind.CPU, "fp32").known is False


def test_measured_gpu_int8_prices_the_detector_on_orin():
    """The point of the measurement path: measured GPU INT8 and FP16 dense
    conv -- the formats det's Class A and B run in on the Orin SM -- price
    `det` on Orin, which default_v1 alone left a gap."""
    measured = [{"kernel_class": "dense_conv_gemm", "engine_kind": "gpu", "precision": p,
                 "compute_eff": 0.4, "confidence": "calibrated", "source": "test run"}
                for p in ("int8", "fp16")]
    analyzer = SoCAnalyzer(tables=_layered(measured))
    result = analyzer.analyze("orin_class_reference", "air superiority", efficiency="t_measured")
    det = next(s for s in result.schedule.services if s.stage == "det")
    assert det.served and det.t_service_s > 0
    baseline = analyzer.analyze("orin_class_reference", "air superiority", efficiency="default_v1")
    assert not next(s for s in baseline.schedule.services if s.stage == "det").served


@pytest.mark.parametrize("tables, match", [
    ({"a": {"id": "a", "name": "a", "kind": "per_engine", "base": "b"},
      "b": {"id": "b", "name": "b", "kind": "per_engine", "base": "a"}}, "cycle"),
    ({"a": {"id": "a", "name": "a", "kind": "per_engine", "base": "missing"}}, "not a table"),
])
def test_bad_layers_are_errors(tables, match):
    entry = {"kernel_class": "fft", "engine_kind": "gpu", "precision": "fp32",
             "confidence": "unknown", "source": "No test"}
    built = {k: EfficiencyTable.model_validate({**v, "entries": [entry]}) for k, v in tables.items()}
    with pytest.raises((ValueError, KeyError), match=match):
        resolve_layers(built)


def test_a_pooled_table_cannot_layer():
    with pytest.raises(ValidationError, match="pooled table cannot layer"):
        EfficiencyTable.model_validate({
            "id": "p", "name": "p", "kind": "pooled", "base": "default_v1",
            "pooled": {"a": 1, "b": 1, "c": 1, "confidence": "theoretical", "source": "x"}})


def test_a_cpu_result_that_ran_wide_is_refused():
    """The first Orin Nano run: FP32 GEMM 'on one core' at 5x a core's peak,
    because the BLAS pool ran on all six. Parallelism is now measured; a
    wide run is refused even when its efficiency would look plausible."""
    wide = _result(kernel_class="small_dense_linalg", engine="cpu", name="cholesky_solve",
                   attained=0.3 * CPU_FP32_PER_CORE * 1.7e9, clock_hz=1.7e9, parallelism=5.8)
    _, skipped = _build(_run(wide, _result()))
    assert any("not verified single-threaded" in s and "5.8 cores" in s for s in skipped)


def test_a_run_from_before_the_thread_check_gives_no_cpu_entries():
    old = _result(kernel_class="small_qp", engine="cpu", name="kkt_solve", clock_hz=1.7e9,
                  attained=0.3 * CPU_FP32_PER_CORE * 1.7e9)
    old["single_thread"] = None
    old["cpu_parallelism"] = None
    _, skipped = _build(_run(old, _result()))
    assert any("predates the single-thread check" in s for s in skipped)


def test_gpu_fp32_needs_tf32_locked_off():
    """cuDNN runs FP32 convolutions as TF32 by default: the first Orin Nano
    'FP32' conv measured twice the FP32 peak."""
    _, skipped = _build(_run(_result(), tf32_disabled=None),
                        _run(_result(prec="int8", attained=1e12), tf32_disabled=None))
    assert any("TF32" in s for s in skipped)
    table, _ = _build(_run(_result(prec="int8", attained=1e12), tf32_disabled=None))
    assert table["entries"][0]["precision"] == "int8"  # other precisions are unaffected


def test_a_newly_covered_class_ingests_like_any_other():
    """The six classes that gained a kernel in 6.5 are ordinary rows: the
    ingest neither special-cases nor refuses them."""
    attained = 0.25 * CPU_FP32_PER_CORE * 2.0e9
    rows = [_result(kernel_class=cls, engine="cpu", prec="fp32", attained=attained,
                    clock_hz=2.0e9, name=name)
            for cls, name in (("cost_volume_dp", "sgm"), ("feature_track", "klt"),
                              ("raycast", "tsdf_raycast"), ("wavefront", "esdf_propagate"),
                              ("graph_search", "frontier_relax"),
                              ("pixel_fixed_function", "isp_pipeline"))]
    table, skipped = _build(_run(*rows))
    entries = {e["kernel_class"]: e for e in table["entries"]}
    assert set(entries) == {r["kernel_class"] for r in rows}, skipped
    for entry in entries.values():
        assert entry["compute_eff"] == pytest.approx(0.25)
        assert entry["confidence"] == "calibrated" and entry["engine_kind"] == "cpu"
