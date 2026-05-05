"""Tests for validation/model_v4/ground_truth/pytorch_cpu.py.

The measurer has two responsibilities:

1. Run a model many times and return a clean median latency.
2. Bracket the run with RAPL energy reads when available, returning
   ``energy_j=None`` (with explanatory notes) when not.

Unit tests use stub models (no PyTorch dependency) and stub RAPL paths
(via tmp_path). One integration test exercises the real path on a real
PyTorch workload to confirm the wiring end-to-end -- it is light and
runs in well under a second on the dev machine.
"""

from pathlib import Path

import pytest
import torch

from validation.model_v4.ground_truth.pytorch_cpu import (
    DEFAULT_TIMED_TRIALS,
    PyTorchCPUMeasurer,
    _detect_rapl,
    _RAPLProbe,
)
from validation.model_v4.workloads.matmul import build_matmul


# ---------------------------------------------------------------------------
# RAPL probe detection
# ---------------------------------------------------------------------------


def test_detect_rapl_returns_none_when_path_absent(tmp_path):
    """No file -> None, no exception."""
    nonexistent = tmp_path / "no-such-rapl"
    assert _detect_rapl(nonexistent) is None


def test_detect_rapl_returns_none_when_files_missing(tmp_path):
    """Directory exists but the energy_uj / max_energy_range_uj files
    don't -- still safe."""
    (tmp_path / "intel-rapl:0").mkdir()
    assert _detect_rapl(tmp_path / "intel-rapl:0") is None


def test_detect_rapl_returns_none_on_malformed_contents(tmp_path):
    """Garbage in energy_uj -> None, no crash."""
    pkg = tmp_path / "intel-rapl:0"
    pkg.mkdir()
    (pkg / "energy_uj").write_text("not-a-number")
    (pkg / "max_energy_range_uj").write_text("262143328850")
    assert _detect_rapl(pkg) is None


def test_detect_rapl_returns_probe_on_valid_path(tmp_path):
    pkg = tmp_path / "intel-rapl:0"
    pkg.mkdir()
    (pkg / "energy_uj").write_text("12345")
    (pkg / "max_energy_range_uj").write_text("262143328850")
    (pkg / "name").write_text("package-0")
    probe = _detect_rapl(pkg)
    assert probe is not None
    assert probe.name == "package-0"
    assert probe.read_energy_uj() == 12345


# ---------------------------------------------------------------------------
# Counter wrap handling
# ---------------------------------------------------------------------------


def test_rapl_delta_normal_case():
    probe = _RAPLProbe(energy_uj_path=Path("/dev/null"),
                      max_energy_range_uj=1_000_000, name="test")
    assert probe.delta_uj(100, 250) == 150


def test_rapl_delta_handles_counter_wrap():
    """When after < before, the counter wrapped -- compute the delta
    as max + after - before + 1."""
    max_val = 1_000_000
    probe = _RAPLProbe(energy_uj_path=Path("/dev/null"),
                      max_energy_range_uj=max_val, name="test")
    # before close to max, after small -> wrapped
    delta = probe.delta_uj(before_uj=999_990, after_uj=50)
    assert delta == (max_val - 999_990) + 50 + 1


# ---------------------------------------------------------------------------
# Measurer with a stub model (no PyTorch)
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal callable that records how many times it was invoked."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1


def test_measure_runs_warmup_plus_timed_trials():
    stub = _StubModel()
    m = PyTorchCPUMeasurer(hardware="test", warmup_trials=2, timed_trials=5)
    # Force probe off so the test doesn't depend on the host
    m._probe = None
    meas = m.measure(stub, ())
    assert stub.calls == 2 + 5
    assert meas.trial_count == 5
    assert meas.latency_s > 0
    assert meas.energy_j is None
    assert "not available" in meas.notes


def test_measure_returns_energy_when_rapl_available(tmp_path):
    """Wire a fake RAPL counter that ticks up by a known amount per
    measure() call; assert the reported energy matches."""
    pkg = tmp_path / "intel-rapl:0"
    pkg.mkdir()
    (pkg / "max_energy_range_uj").write_text("262143328850")
    (pkg / "name").write_text("package-0")
    energy_path = pkg / "energy_uj"
    energy_path.write_text("1000000")  # 1 J initial
    probe = _detect_rapl(pkg)

    # Build a stub model that bumps the RAPL counter on each call so
    # measure() sees a deterministic energy delta.
    class _BumpModel:
        def __init__(self, path, delta_uj):
            self.path = path
            self.delta_uj = delta_uj

        def __call__(self, *_args):
            cur = int(self.path.read_text())
            self.path.write_text(str(cur + self.delta_uj))

    m = PyTorchCPUMeasurer(hardware="test", warmup_trials=0,
                          timed_trials=10, _probe=probe)
    bump = _BumpModel(energy_path, delta_uj=500_000)  # 0.5 J per call
    meas = m.measure(bump, ())
    # 10 calls * 0.5 J = 5 J total; mean per trial = 0.5 J
    assert meas.energy_j is not None
    assert meas.energy_j == pytest.approx(0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# End-to-end on a real PyTorch workload (lightweight)
# ---------------------------------------------------------------------------


def test_measure_on_real_matmul_returns_sensible_numbers():
    """A 1024^3 fp32 matmul takes ~ms on a modern desktop -- long
    enough that RAPL (~1ms update interval) sees a real delta, and
    short enough to keep the test fast.

    Energy may still be ``None`` if RAPL is unreadable on the test
    host (e.g., CI runners without /sys/class/powercap access). Both
    outcomes are OK."""
    w = build_matmul(1024, 1024, 1024, "fp32")
    m = PyTorchCPUMeasurer(hardware="i7_12700k",
                          warmup_trials=2, timed_trials=11)
    meas = m.measure(w.model, w.inputs)
    assert 0 < meas.latency_s < 1.0
    assert meas.trial_count == 11
    if meas.energy_j is not None:
        # RAPL was readable. 1024^3 fp32 matmul is ~2 GFLOPs which on
        # a desktop CPU is single-digit to tens of mJ; allow zero too
        # because RAPL has ~1ms / ~1uJ resolution and very fast
        # back-to-back kernels can land below it.
        assert 0 <= meas.energy_j < 1.0


def test_default_trial_counts_are_sensible():
    """Sanity-check the module-level defaults so a future tweak to
    the constants is intentional, not accidental."""
    assert DEFAULT_TIMED_TRIALS >= 5
    assert DEFAULT_TIMED_TRIALS % 2 == 1, "odd so median is well-defined"
