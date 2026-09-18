"""Stage-kernel benchmark harness (graphs#269 PR 5.1)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from graphs.benchmarks.soc_kernels import (
    CLOCK_TOLERANCE,
    NOT_COVERED,
    SCHEMA,
    ClockSampler,
    ClockSamples,
    kernel_suite,
    run_kernel,
    run_suite,
)

REPO = Path(__file__).resolve().parents[2]


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
