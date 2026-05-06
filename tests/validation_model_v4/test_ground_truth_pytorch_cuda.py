"""Tests for validation/model_v4/ground_truth/pytorch_cuda.py.

The actual measurement path requires real CUDA hardware and a working
NVML driver, so these tests focus on what we CAN verify on the i7
development box:

* The probe detector returns None cleanly when pynvml isn't installed
  or NVML init fails (so the measurer doesn't crash on CPU-only hosts).
* The measurer returns a Measurement with energy_j=None and an
  explanatory note when the probe is unavailable.
* The measurer returns nan latency + a CUDA note when torch.cuda is
  unavailable, rather than crashing inside torch.cuda.Event.
* The probe correctly chooses total-energy vs power-fallback paths
  based on what NVML reports as supported.

These tests use a mock pynvml to avoid requiring real hardware. The
end-to-end capture path is exercised separately on the target box (see
docs/plans/v4-capture-on-target.md).
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch

import pytest

from validation.model_v4.ground_truth.base import Measurement
from validation.model_v4.ground_truth.pytorch_cuda import (
    PyTorchCUDAMeasurer,
    _detect_nvml,
    _NVMLProbe,
)


# ---------------------------------------------------------------------------
# _detect_nvml: graceful degradation
# ---------------------------------------------------------------------------


def test_detect_nvml_returns_none_when_pynvml_missing(monkeypatch):
    """Force ``import pynvml`` to fail at the import statement inside
    ``_detect_nvml`` and confirm the detector returns None instead of
    raising. This is deterministic regardless of whether pynvml is
    actually installed on the dev box."""
    # Hide any previously-imported pynvml so the in-function import path
    # has to re-resolve; install a meta-finder that raises ImportError.
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)

    class _BlockPynvml:
        def find_module(self, name, path=None):    # py<3.4 API, harmless
            return self if name == "pynvml" else None

        def find_spec(self, name, path=None, target=None):
            if name == "pynvml":
                raise ImportError("forced absent for test")
            return None

    monkeypatch.setattr(sys, "meta_path",
                        [_BlockPynvml(), *sys.meta_path])
    probe = _detect_nvml(device_index=0)
    assert probe is None


def test_detect_nvml_returns_none_when_init_fails():
    """If pynvml is importable but nvmlInit raises (driver mismatch,
    no GPU), the detector returns None instead of letting the exception
    bubble up."""
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.side_effect = RuntimeError("driver mismatch")

    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        # Force re-import inside _detect_nvml
        probe = _detect_nvml(device_index=0)

    assert probe is None


def test_detect_nvml_returns_none_when_no_query_supported():
    """If neither total-energy nor power queries work (very old GPU),
    _detect_nvml refuses to return a probe -- there's nothing to read."""
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.return_value = None
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "fake-handle"
    fake_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = RuntimeError("unsupported")
    fake_pynvml.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("unsupported")

    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        probe = _detect_nvml(device_index=0)

    assert probe is None
    # Cleanup must have been attempted
    fake_pynvml.nvmlShutdown.assert_called()


def test_detect_nvml_falls_back_to_power_when_total_energy_missing():
    """Older Tesla cards expose nvmlDeviceGetPowerUsage but not the
    cumulative energy counter. The probe must accept this and mark
    supports_total_energy=False (the measurer then uses the
    power-fallback path)."""
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.return_value = None
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "fake-handle"
    fake_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = RuntimeError("unsupported")
    fake_pynvml.nvmlDeviceGetPowerUsage.return_value = 250_000   # 250W in mW
    fake_pynvml.nvmlDeviceGetName.return_value = b"Tesla V100"

    with patch.dict("sys.modules", {"pynvml": fake_pynvml}):
        probe = _detect_nvml(device_index=0)

    assert probe is not None
    assert probe.supports_total_energy is False
    assert probe.read_power_mw() == 250_000


# ---------------------------------------------------------------------------
# Measurer: graceful degradation when CUDA / NVML unavailable
# ---------------------------------------------------------------------------


def test_measurer_returns_nan_latency_when_cuda_unavailable():
    """On a CPU-only host, calling .measure() returns a Measurement
    with nan latency and a CUDA note rather than crashing inside
    torch.cuda.Event."""
    measurer = PyTorchCUDAMeasurer(hardware="h100_sxm5_80gb", _probe=None)

    # Stub torch with cuda.is_available()=False
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    assert isinstance(meas, Measurement)
    assert math.isnan(meas.latency_s)
    assert meas.energy_j is None
    assert "CUDA" in meas.notes


def test_measurer_skips_energy_when_no_probe():
    """No NVML probe but CUDA available: latency still measured via
    cudaEvent, energy comes back as None with a note."""
    measurer = PyTorchCUDAMeasurer(
        hardware="h100_sxm5_80gb",
        warmup_trials=1, timed_trials=3,
        _probe=None,
    )

    fake_torch = _make_fake_cuda_torch(per_trial_ms=2.0)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    assert meas.energy_j is None
    assert "NVML: probe not available" in meas.notes
    # Median of three 2ms readings = 2ms = 2e-3 s
    assert meas.latency_s == pytest.approx(2e-3, rel=1e-6)
    assert meas.trial_count == 3


# ---------------------------------------------------------------------------
# Measurer: total-energy path using a mocked probe
# ---------------------------------------------------------------------------


def test_measurer_uses_total_energy_when_supported():
    """With a probe that reports supports_total_energy=True, the
    measurer should use the cumulative-counter delta and report the
    per-trial energy in joules."""
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [
        100_000,     # before: 100000 mJ
        100_330,     # after:  100330 mJ -> delta 330 mJ across 11 trials
    ]
    probe = _NVMLProbe(handle="h", supports_total_energy=True,
                       pynvml=fake_pynvml, name="H100")

    measurer = PyTorchCUDAMeasurer(
        hardware="h100_sxm5_80gb",
        warmup_trials=1, timed_trials=11,
        _probe=probe,
    )

    fake_torch = _make_fake_cuda_torch(per_trial_ms=2.0)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    # 330 mJ / 11 trials = 30 mJ per trial = 30e-3 J
    assert meas.energy_j == pytest.approx(30e-3, rel=1e-6)
    # No fallback note expected
    assert "power-fallback" not in meas.notes


def test_measurer_falls_back_to_power_when_no_total_energy():
    """With a power-only probe, the measurer should compute
    energy = power * latency."""
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetPowerUsage.return_value = 250_000  # 250 W
    probe = _NVMLProbe(handle="h", supports_total_energy=False,
                       pynvml=fake_pynvml, name="V100")

    measurer = PyTorchCUDAMeasurer(
        hardware="h100_sxm5_80gb",
        warmup_trials=1, timed_trials=5,
        _probe=probe,
    )

    fake_torch = _make_fake_cuda_torch(per_trial_ms=4.0)  # 4ms per trial

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    # 250 W * 4ms = 1.0 J per trial
    assert meas.energy_j == pytest.approx(1.0, rel=1e-6)
    assert "power-fallback" in meas.notes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_cuda_torch(per_trial_ms: float):
    """Build a torch stub whose cuda.Event pair returns ``per_trial_ms``
    from start.elapsed_time(stop). The stop event's elapsed_time method
    needs to be set on the start event since cuda.Event.elapsed_time is
    actually called as start.elapsed_time(stop)."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    def _make_event(*args, **kwargs):
        ev = MagicMock()
        ev.elapsed_time.return_value = per_trial_ms
        return ev

    fake_torch.cuda.Event.side_effect = _make_event
    fake_torch.cuda.synchronize.return_value = None
    return fake_torch
