"""Stage-kernel benchmark harness (graphs#269 PR 5.1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from graphs.benchmarks.soc_kernels import (
    CLOCK_TOLERANCE,
    NOT_COVERED,
    SCHEMA,
    SINGLE_THREAD_LIMIT,
    ClockSampler,
    ClockSamples,
    kernel_suite,
    run_kernel,
    run_suite,
)

REPO = Path(__file__).resolve().parents[2]

#: The CPUs this process may use, captured before any test pins it to one.
_INITIAL_AFFINITY = frozenset(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


# ---------------------------------------------------------------------------
# Clock verification: the thing the 2026-02 calibrations got wrong
# ---------------------------------------------------------------------------


def test_a_steady_clock_is_verified():
    assert ClockSamples("gpu", [1.30e9] * 10).verified


def test_a_wandering_clock_is_not():
    """A DVFS clock that moves more than 5% during the run cannot anchor an
    efficiency: the peak it divides by is not one number."""
    samples = [1.3e9, 1.3e9, 0.9e9, 1.3e9, 1.3e9, 1.3e9]
    assert not ClockSamples("gpu", samples).verified
    assert ClockSamples("gpu", samples).spread > CLOCK_TOLERANCE


def test_too_few_samples_are_not_verified():
    assert not ClockSamples("gpu", [1.3e9, 1.3e9]).verified
    assert not ClockSamples("unavailable", []).verified


def test_the_sampler_polls_while_the_block_runs():
    ticks = iter(range(1, 10_000))
    with ClockSampler(lambda: 1e9 + next(ticks), "fake", period_s=0.001) as sampler:
        time.sleep(0.05)
    assert len(sampler.result.samples_hz) >= 5


def test_no_reader_means_unavailable_not_verified():
    with ClockSampler(None, "gpu") as sampler:
        pass
    assert sampler.result.source == "unavailable" and not sampler.result.verified


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def test_ops_counts_follow_the_stated_rules():
    specs = {(k.name, k.precision): k for k in kernel_suite()}
    assert specs[("gemm", "fp32")].ops_per_call == 2 * 2048 ** 3
    assert specs[("conv3x3", "fp16")].ops_per_call == 2 * 256 * 256 * 9 * 80 * 80
    assert specs[("gemv", "fp16")].ops_per_call == 2 * 4096 * 4096
    for spec in specs.values():
        assert spec.ops_per_call > 0 and spec.ops_rule


def test_uncovered_classes_are_named_not_faked():
    """Every kernel class the analyzer knows is either measured here or
    listed as not covered -- a class is never silently missing."""
    from graphs.estimation.soc import KernelClass

    measured = {k.kernel_class for k in kernel_suite()}
    assert measured | set(NOT_COVERED) == {k.value for k in KernelClass}
    assert not measured & set(NOT_COVERED)


def test_an_unsupported_device_is_reported_not_run():
    int8 = next(k for k in kernel_suite(quick=True) if k.name == "gemm" and k.precision == "int8")
    result = run_kernel(int8, "cpu")
    assert result.status == "unsupported" and result.attained_ops_per_s is None


def test_a_kernel_error_is_recorded_and_the_run_continues():
    from graphs.benchmarks.soc_kernels import KernelSpec

    def broken(device, dtype):
        raise RuntimeError("no such op")

    spec = KernelSpec("fft", "broken", "fp32", "-", 1.0, "-", broken)
    result = run_kernel(spec, "cpu")
    assert result.status == "error" and "no such op" in result.message


def test_a_cpu_run_produces_the_document():
    doc = run_suite(["cpu"], "test_box", quick=True, min_seconds=0.02,
                    kernels=["small_qp", "elementwise_norm"])
    assert doc["schema"] == SCHEMA and doc["hardware"] == "test_box"
    assert "1 thread" in doc["cpu_threading"]
    ok = [r for r in doc["results"] if r["status"] == "ok"]
    assert ok and all(r["attained_ops_per_s"] > 0 and r["engine_kind"] == "cpu" for r in ok)
    assert {r["kernel_class"] for r in doc["results"]} == {"small_qp", "elementwise_norm"}
    json.dumps(doc)


def test_cli_writes_the_json(tmp_path):
    out = tmp_path / "bench.json"
    result = subprocess.run(
        [sys.executable, "cli/benchmark_soc_kernels.py", "--hardware", "test_box", "--quick",
         "--min-seconds", "0.02", "--kernels", "small_dense_linalg", "--devices", "cpu",
         "-o", str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 0, result.stderr
    doc = json.loads(out.read_text())
    assert doc["quick"] is True and doc["results"]
    assert "small_dense_linalg" in result.stdout


@pytest.mark.parametrize("args", [["--hardware", "Bad Name"], ["--hardware", "x", "--devices", "tpu"]])
def test_cli_rejects_bad_arguments(args, tmp_path):
    result = subprocess.run([sys.executable, "cli/benchmark_soc_kernels.py", *args,
                             "-o", str(tmp_path / "x.json")],
                            capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 2


def test_a_cpu_run_records_its_parallelism():
    doc = run_suite(["cpu"], "test_box", quick=True, min_seconds=0.02, kernels=["small_qp"])
    ok = [r for r in doc["results"] if r["status"] == "ok"]
    assert ok and all(r["cpu_parallelism"] is not None for r in ok)
    assert doc["tf32_disabled"] is None  # no GPU in this run
    assert doc["environment"]["torch_threads"] >= 1


def test_a_wide_cpu_run_is_flagged():
    """Force a multi-threaded GEMM on several cores: the parallelism check
    must catch it. Earlier tests pin this process to one core, so the test
    runs on the CPUs the process was *allowed* at import -- not cpu_count(),
    which a cgroup can exceed -- and restores the pinning afterwards."""
    import torch

    if _INITIAL_AFFINITY is None or len(_INITIAL_AFFINITY) < 2:
        pytest.skip("needs Linux affinity and more than one allowed CPU")
    affinity, threads = os.sched_getaffinity(0), torch.get_num_threads()
    try:
        os.sched_setaffinity(0, _INITIAL_AFFINITY)
        torch.set_num_threads(min(4, len(_INITIAL_AFFINITY)))
        spec = next(k for k in kernel_suite() if k.name == "gemm" and k.precision == "fp32")
        result = run_kernel(spec, "cpu", min_seconds=0.3)
    finally:
        torch.set_num_threads(threads)
        os.sched_setaffinity(0, affinity)
    if result.cpu_parallelism <= SINGLE_THREAD_LIMIT:
        # The threads were asked for but the machine did not give them: on a
        # loaded runner (four test shards at once) the BLAS pool contends and
        # the run stays near one core's worth. Nothing is wrong with the
        # check, and test_a_run_is_flagged_by_its_parallelism covers the rule
        # itself without racing the scheduler.
        pytest.skip(f"the runner gave {result.cpu_parallelism:.2f} cores' worth, "
                    f"not more than {SINGLE_THREAD_LIMIT}")
    assert result.single_thread is False


@pytest.mark.parametrize("parallelism, single", [(0.5, True), (1.0, True), (2.0, False),
                                                (5.1, False)])
def test_a_run_is_flagged_by_its_parallelism(parallelism, single, monkeypatch):
    """The rule itself, without racing the scheduler: a run is
    single-threaded exactly while its CPU time per wall second is within
    the limit. The clock is faked, so the flag is checked against what the
    harness measured rather than against what was asked for."""
    spec = next(k for k in kernel_suite(quick=True) if k.name == "kkt_solve")
    monkeypatch.setattr("graphs.benchmarks.soc_kernels.time.process_time",
                        _clock(step=parallelism), raising=True)
    result = run_kernel(spec, "cpu", min_seconds=0.02, warmup=1)
    assert result.cpu_parallelism == pytest.approx(parallelism, rel=0.2)
    assert result.single_thread is (result.cpu_parallelism <= SINGLE_THREAD_LIMIT)
    assert result.single_thread is single


def _clock(step: float):
    """A process_time that advances ``step`` seconds per wall second."""
    start = time.perf_counter()

    def now() -> float:
        return (time.perf_counter() - start) * step
    return now


# ---------------------------------------------------------------------------
# The classes that had no kernel before (graphs#269 6.5)
# ---------------------------------------------------------------------------

NEW_CLASSES = ("cost_volume_dp", "feature_track", "raycast", "wavefront", "graph_search",
               "pixel_fixed_function")


def test_every_class_a_stage_uses_now_has_a_kernel():
    """knn_tree is the only class left without one, and no stage of the
    workload uses it."""
    from graphs.estimation.soc import load_kernel_classes

    used = {k.value for k in load_kernel_classes().stages.values()
            for k in [k.kernel_class]}
    measured = {k.kernel_class for k in kernel_suite()}
    assert used <= measured
    assert set(NOT_COVERED) == {"knn_tree"} and "knn_tree" not in used


@pytest.mark.parametrize("kernel_class", NEW_CLASSES)
def test_the_new_kernels_state_their_shape_and_ops_rule(kernel_class):
    specs = [k for k in kernel_suite() if k.kernel_class == kernel_class]
    assert specs, kernel_class
    for spec in specs:
        assert spec.ops_per_call > 0
        assert len(spec.ops_rule) > 20, spec.ops_rule  # a rule, not a word
        assert spec.shape and any(ch.isdigit() for ch in spec.shape)


def test_the_new_ops_counts_follow_their_rules():
    specs = {(k.name, k.precision): k for k in kernel_suite()}
    # SGM: the cost volume, then 7 per element per path per column.
    assert specs[("sgm", "fp32")].ops_per_call == 64 * 720 * 1280 + 4 * 64 * 720 * 1279 * 7
    # The raycast's 24 ops per trilinear sample.
    assert specs[("tsdf_raycast", "fp32")].ops_per_call == 65536 * 64 * 24
    # The wavefront's 6 neighbours, add and min, per sweep.
    assert specs[("esdf_propagate", "fp32")].ops_per_call == 8 * 6 * 160 ** 3 * 2
    # One add and one min per edge per relaxation.
    assert specs[("frontier_relax", "fp32")].ops_per_call == 30 * 32768 * 8 * 2
    # 84 ops per pixel through the ISP chain.
    assert specs[("isp_pipeline", "fp16")].ops_per_call == 1920 * 1080 * 84


@pytest.mark.parametrize("kernel_class", NEW_CLASSES)
def test_each_new_kernel_runs_on_the_cpu(kernel_class):
    """The quick shapes, so the test is cheap; a real run uses the full
    ones. Every kernel must produce a throughput, not an error."""
    spec = next(k for k in kernel_suite(quick=True) if k.kernel_class == kernel_class)
    result = run_kernel(spec, "cpu", min_seconds=0.02, warmup=1)
    assert result.status == "ok", result.message
    assert result.attained_ops_per_s > 0
    assert result.calls >= 1 and result.seconds_per_call > 0
