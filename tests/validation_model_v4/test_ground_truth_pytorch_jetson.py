"""Tests for validation/model_v4/ground_truth/pytorch_jetson.py.

Like the CUDA tests, this only verifies the parts we can exercise on
the i7 dev box (which doesn't have INA3221 sysfs). The end-to-end
capture path is exercised on the actual Jetson hardware separately.

We test:
* The probe detector returns None when no INA3221 sysfs is present.
* The probe detector parses both JetPack-5+ (hwmon) and JetPack-4
  (iio:device) layouts using a temporary fake sysfs tree.
* The rail-pattern filter correctly selects/rejects rails by name.
* The measurer returns nan latency + a CUDA note when torch.cuda is
  unavailable (CPU-only test host).
* The measurer integrates rail power x window time correctly when
  given a fake probe.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from validation.model_v4.ground_truth.base import Measurement
from validation.model_v4.ground_truth.pytorch_jetson import (
    DEFAULT_RAIL_PATTERN,
    PyTorchJetsonMeasurer,
    _detect_ina3221,
    _INAProbe,
    _INARail,
)


# ---------------------------------------------------------------------------
# _detect_ina3221: graceful degradation
# ---------------------------------------------------------------------------


def test_detect_ina3221_returns_none_on_non_jetson_host():
    """The i7 dev box has no INA3221 sysfs entries; the detector must
    return None instead of raising."""
    probe = _detect_ina3221()
    assert probe is None


def test_detect_ina3221_finds_jp5_hwmon_layout(tmp_path, monkeypatch):
    """Synthesize a JetPack-5+ hwmon layout under tmp_path and confirm
    the detector finds the rails."""
    fake_root = tmp_path / "ina3221" / "1-0040" / "hwmon" / "hwmon3"
    fake_root.mkdir(parents=True)
    # Two rails: GPU_SOC (matches default pattern) and a filtered DDR rail.
    (fake_root / "in1_label").write_text("VDD_GPU_SOC\n")
    (fake_root / "power1_input").write_text("3500000\n")  # 3500 mW in uW
    (fake_root / "in2_label").write_text("VDD_DDR\n")
    (fake_root / "power2_input").write_text("1200000\n")

    monkeypatch.setattr(
        "validation.model_v4.ground_truth.pytorch_jetson._INA_SEARCH_ROOTS",
        [tmp_path / "ina3221", tmp_path / "ina3221x_nope"],
    )
    probe = _detect_ina3221()
    assert probe is not None
    assert probe.backend == "ina3221"
    assert len(probe.rails) == 1   # DDR filtered out by default pattern
    assert probe.rails[0].name == "VDD_GPU_SOC"
    # Rail reads back correctly: 3500000 uW / 1000 = 3500 mW
    assert probe.rails[0].read_power_mw() == 3500


def test_detect_ina3221_finds_jp4_iio_layout(tmp_path, monkeypatch):
    """JetPack-4 used iio:device with rail_name_<idx> + in_power<idx>_input
    in milliwatts."""
    fake_root = tmp_path / "ina3221x" / "1-0040" / "iio:device0"
    fake_root.mkdir(parents=True)
    (fake_root / "rail_name_0").write_text("GPU\n")
    (fake_root / "in_power0_input").write_text("4500\n")    # mW direct
    (fake_root / "rail_name_1").write_text("VIN_SYS_5V0\n")
    (fake_root / "in_power1_input").write_text("2000\n")

    monkeypatch.setattr(
        "validation.model_v4.ground_truth.pytorch_jetson._INA_SEARCH_ROOTS",
        [tmp_path / "ina3221_nope", tmp_path / "ina3221x"],
    )
    probe = _detect_ina3221()
    assert probe is not None
    assert probe.backend == "ina3221x"
    assert len(probe.rails) == 2
    assert probe.read_total_power_mw() == 6500  # 4500 + 2000


def test_rail_pattern_filters_to_caller_choice(tmp_path, monkeypatch):
    """Caller-supplied pattern selects only matching rails."""
    fake_root = tmp_path / "ina3221x" / "1-0040" / "iio:device0"
    fake_root.mkdir(parents=True)
    for i, (name, mw) in enumerate([
            ("GPU", 4000),
            ("CPU", 2000),
            ("MEM", 1000),
    ]):
        (fake_root / f"rail_name_{i}").write_text(f"{name}\n")
        (fake_root / f"in_power{i}_input").write_text(f"{mw}\n")

    monkeypatch.setattr(
        "validation.model_v4.ground_truth.pytorch_jetson._INA_SEARCH_ROOTS",
        [tmp_path / "ina3221_nope", tmp_path / "ina3221x"],
    )
    probe = _detect_ina3221(rail_pattern=r"^GPU$")
    assert probe is not None
    assert [r.name for r in probe.rails] == ["GPU"]
    assert probe.read_total_power_mw() == 4000


def test_default_rail_pattern_is_a_valid_regex():
    """Sanity: the DEFAULT_RAIL_PATTERN constant compiles."""
    import re
    re.compile(DEFAULT_RAIL_PATTERN)


# ---------------------------------------------------------------------------
# Measurer: graceful degradation when CUDA / INA unavailable
# ---------------------------------------------------------------------------


def test_measurer_returns_nan_latency_when_cuda_unavailable():
    measurer = PyTorchJetsonMeasurer(hardware="jetson_orin_nano_8gb", _probe=None)

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    assert isinstance(meas, Measurement)
    assert math.isnan(meas.latency_s)
    assert meas.energy_j is None
    assert "CUDA" in meas.notes


def test_measurer_skips_energy_when_no_probe():
    """No INA3221 probe but CUDA available: latency measured, energy
    None with 'probe not available' note."""
    measurer = PyTorchJetsonMeasurer(
        hardware="jetson_orin_nano_8gb",
        warmup_trials=1, timed_trials=3,
        _probe=None,
    )
    fake_torch = _make_fake_cuda_torch(per_trial_ms=2.0)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    assert meas.energy_j is None
    assert "INA3221: probe not available" in meas.notes


# ---------------------------------------------------------------------------
# Measurer: power integration with a fake probe
# ---------------------------------------------------------------------------


def test_measurer_integrates_rail_power_into_energy():
    """A fake probe reports 5000 mW each side; with 3 trials at 4 ms each
    the per-trial energy should be (5W * 12ms) / 3 = 20 mJ = 0.02 J."""
    rail = _INARail(name="GPU", read_power_mw=lambda: 5000)
    probe = _INAProbe(rails=[rail], backend="ina3221x")

    measurer = PyTorchJetsonMeasurer(
        hardware="jetson_orin_nano_8gb",
        warmup_trials=1, timed_trials=3,
        _probe=probe,
    )
    fake_torch = _make_fake_cuda_torch(per_trial_ms=4.0)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    # window = 3 * 4ms = 12 ms; avg_power = 5W; total = 60 mJ
    # per-trial = 60 / 3 = 20 mJ = 0.02 J
    assert meas.energy_j == pytest.approx(0.02, rel=1e-6)
    assert "rails=['GPU']" in meas.notes


def test_measurer_warns_on_sub_8ms_window():
    """INA3221 has 8 ms internal averaging; windows smaller than that
    get a noise warning in the notes."""
    rail = _INARail(name="GPU", read_power_mw=lambda: 5000)
    probe = _INAProbe(rails=[rail], backend="ina3221x")

    measurer = PyTorchJetsonMeasurer(
        hardware="jetson_orin_nano_8gb",
        warmup_trials=1, timed_trials=3,
        _probe=probe,
    )
    # 3 trials * 1 ms = 3 ms total, well below 8 ms.
    fake_torch = _make_fake_cuda_torch(per_trial_ms=1.0)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        meas = measurer.measure(model=MagicMock(), inputs=[])

    assert meas.energy_j is not None
    assert "below 8ms averaging period" in meas.notes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_cuda_torch(per_trial_ms: float):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    def _make_event(*args, **kwargs):
        ev = MagicMock()
        ev.elapsed_time.return_value = per_trial_ms
        return ev

    fake_torch.cuda.Event.side_effect = _make_event
    fake_torch.cuda.synchronize.return_value = None
    return fake_torch
