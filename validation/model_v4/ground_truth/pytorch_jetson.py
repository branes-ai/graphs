"""PyTorch CUDA + INA3221 ground-truth measurer for NVIDIA Jetson devices.

The Jetson family (Orin Nano/NX/AGX, Thor) exposes per-rail power
telemetry via the INA3221 sysfs interface. This is *more accurate* than
NVML on Jetson because it reads the dedicated current-sense hardware on
the SoC, capturing the integrated GPU + CPU + memory + IO power as
separate rails -- whereas NVML on Tegra reports an aggregate that
sometimes excludes the memory-controller draw (which is a meaningful
fraction of inference power).

Latency:
    Same as ``pytorch_cuda.py``: ``torch.cuda.Event(enable_timing=True)``
    bracketing each timed iteration.

Energy:
    Sum of selected INA3221 rail power readings, integrated across the
    timed window. Sysfs paths vary by JetPack version:

    * JetPack 5+ (Orin, Thor):
      ``/sys/bus/i2c/drivers/ina3221/<bus>/hwmon/<hwmonN>/in<idx>_input``
      voltage in mV
      ``/sys/bus/i2c/drivers/ina3221/<bus>/hwmon/<hwmonN>/curr<idx>_input``
      current in mA
      Power_mW = (V_mV * I_mA) / 1000

    * JetPack 4 (Xavier):
      ``/sys/bus/i2c/drivers/ina3221x/<bus>/iio:device<n>/in_power<idx>_input``
      power in mW directly

    Both paths are detected at __init__ time. Rail labels (e.g.,
    ``"VDD_GPU_SOC"``, ``"VDD_CPU_CV"``, ``"VDD_SOC"``) are read from
    the matching ``rail_name_<idx>`` or ``in_name`` files so the caller
    can pick which rails to integrate.

    Default rail set: GPU + SOC + CPU. The user can override via the
    ``rail_pattern`` argument (a regex matched against rail names).

If INA3221 isn't accessible (path missing, permission denied, no rails
match the pattern), the measurer returns
``Measurement(energy_j=None, notes="INA3221: <reason>")``. Like the
CPU/CUDA cases, the assertions module treats this as "not failed".
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple


DEFAULT_WARMUP_TRIALS = 3
DEFAULT_TIMED_TRIALS = 11

# Default rails to integrate. Conservative pattern that catches the
# main inference draws on Orin Nano / AGX / NX without picking up DDR
# (which is power-managed independently and may double-count).
DEFAULT_RAIL_PATTERN = r"^(VDD_GPU_SOC|VDD_CPU_CV|VDD_SOC|VDD_IN|GPU|CPU|SOC|VIN_SYS_5V0)$"

# Sysfs roots to probe in order of preference.
_INA_SEARCH_ROOTS = [
    Path("/sys/bus/i2c/drivers/ina3221"),     # JetPack 5+
    Path("/sys/bus/i2c/drivers/ina3221x"),    # JetPack 4
]


# ---------------------------------------------------------------------------
# INA3221 probe
# ---------------------------------------------------------------------------


@dataclass
class _INARail:
    """One INA3221 rail with a power reader (mW)."""
    name: str
    read_power_mw: Any   # callable: () -> Optional[int]


@dataclass
class _INAProbe:
    """Selected set of INA3221 rails to integrate."""
    rails: list[_INARail] = field(default_factory=list)
    backend: str = "unknown"   # "ina3221" (jp5+) or "ina3221x" (jp4)

    def read_total_power_mw(self) -> Optional[int]:
        """Sum the rail power readings. None if any rail fails."""
        total = 0
        for r in self.rails:
            v = r.read_power_mw()
            if v is None:
                return None
            total += v
        return total


def _detect_ina3221(rail_pattern: str = DEFAULT_RAIL_PATTERN) -> Optional[_INAProbe]:
    """Walk the sysfs roots, find rails whose name matches ``rail_pattern``,
    and return an _INAProbe ready to read total power. None on any failure."""
    pat = re.compile(rail_pattern)

    # JetPack 5+ path: hwmon-style power = V*I / 1000
    for hwmon_root in _INA_SEARCH_ROOTS[:1]:
        if not hwmon_root.exists():
            continue
        rails: list[_INARail] = []
        for dev in hwmon_root.glob("*/hwmon/hwmon*"):
            for name_path in sorted(dev.glob("in*_label")):
                try:
                    rail_name = name_path.read_text().strip()
                except OSError:
                    continue
                if not pat.match(rail_name):
                    continue
                # name_path looks like "in1_label"; the matching power rail
                # is "in1_input" + "curr1_input" -- but on hwmon the simpler
                # path is "power1_input" in microwatts.
                idx_match = re.match(r"in(\d+)_label", name_path.name)
                if not idx_match:
                    continue
                idx = idx_match.group(1)
                power_path = dev / f"power{idx}_input"
                if power_path.exists():
                    rails.append(_INARail(
                        name=rail_name,
                        read_power_mw=_make_uw_reader(power_path),
                    ))
                else:
                    # Fall back to V*I (both in milli-units)
                    volt_path = dev / f"in{idx}_input"
                    curr_path = dev / f"curr{idx}_input"
                    if volt_path.exists() and curr_path.exists():
                        rails.append(_INARail(
                            name=rail_name,
                            read_power_mw=_make_vi_reader(volt_path, curr_path),
                        ))
        if rails:
            return _INAProbe(rails=rails, backend="ina3221")

    # JetPack 4 path: iio-style, direct power in mW
    for iio_root in _INA_SEARCH_ROOTS[1:]:
        if not iio_root.exists():
            continue
        rails = []
        for dev in iio_root.glob("*/iio:device*"):
            # rail names live in in_name<N>; power in in_power<N>_input (mW)
            for name_path in sorted(dev.glob("rail_name_*")):
                try:
                    rail_name = name_path.read_text().strip()
                except OSError:
                    continue
                if not pat.match(rail_name):
                    continue
                idx_match = re.match(r"rail_name_(\d+)", name_path.name)
                if not idx_match:
                    continue
                idx = idx_match.group(1)
                power_path = dev / f"in_power{idx}_input"
                if power_path.exists():
                    rails.append(_INARail(
                        name=rail_name,
                        read_power_mw=_make_mw_reader(power_path),
                    ))
        if rails:
            return _INAProbe(rails=rails, backend="ina3221x")

    return None


def _make_mw_reader(path: Path):
    def _read() -> Optional[int]:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            return None
    return _read


def _make_uw_reader(path: Path):
    def _read() -> Optional[int]:
        try:
            return int(path.read_text().strip()) // 1000  # uW -> mW
        except (OSError, ValueError):
            return None
    return _read


def _make_vi_reader(volt_path: Path, curr_path: Path):
    def _read() -> Optional[int]:
        try:
            v_mv = int(volt_path.read_text().strip())
            i_ma = int(curr_path.read_text().strip())
            return (v_mv * i_ma) // 1000  # mV*mA = uW; uW/1000 = mW
        except (OSError, ValueError):
            return None
    return _read


# ---------------------------------------------------------------------------
# Measurer
# ---------------------------------------------------------------------------


@dataclass
class PyTorchJetsonMeasurer:
    """cudaEvent + INA3221 measurer for PyTorch on a Jetson device.

    Use this for Orin Nano/NX/AGX and Thor. For desktop/server NVIDIA
    use ``PyTorchCUDAMeasurer`` (NVML) instead.

    Energy integration strategy: sample INA3221 power once at the start
    and once at the end of the timed window, average, multiply by total
    window time. INA3221 has ~8ms internal averaging built in (per the
    datasheet), so for our typical millisecond-scale windows the read-
    once-per-side approach is roughly equivalent to a tight polling loop
    while costing far less CPU overhead. The notes field surfaces if the
    window was below the INA3221 averaging period (results unreliable).
    """
    hardware: str
    device_index: int = 0
    warmup_trials: int = DEFAULT_WARMUP_TRIALS
    timed_trials: int = DEFAULT_TIMED_TRIALS
    rail_pattern: str = DEFAULT_RAIL_PATTERN
    name: str = "pytorch_jetson_ina3221"
    _probe: Optional[_INAProbe] = field(default=None)

    def __post_init__(self) -> None:
        if self._probe is None:
            self._probe = _detect_ina3221(self.rail_pattern)

    # -- public API ----------------------------------------------------------

    def measure(self, model, inputs):
        """Run ``model(*inputs)`` warmup + N times, return Measurement."""
        import torch
        from validation.model_v4.ground_truth.base import Measurement

        if not torch.cuda.is_available():
            return Measurement(
                latency_s=float("nan"),
                energy_j=None,
                trial_count=0,
                notes="CUDA: torch.cuda.is_available() returned False",
            )

        for _ in range(self.warmup_trials):
            model(*inputs)
        torch.cuda.synchronize(self.device_index)

        latencies_s, energy_j, notes = self._timed_window(model, inputs, torch)
        median_latency_s = statistics.median(latencies_s) if latencies_s else float("nan")

        return Measurement(
            latency_s=median_latency_s,
            energy_j=energy_j,
            trial_count=self.timed_trials,
            notes=notes,
        )

    # -- internals -----------------------------------------------------------

    def _timed_window(self, model, inputs, torch) -> Tuple[list[float], Optional[float], str]:
        notes_parts: list[str] = []

        power_before_mw: Optional[int] = None
        if self._probe is not None:
            power_before_mw = self._probe.read_total_power_mw()
            if power_before_mw is None:
                notes_parts.append("INA3221: pre-window power read failed")

        latencies_s: list[float] = []
        for _ in range(self.timed_trials):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            stop.record()
            torch.cuda.synchronize(self.device_index)
            latencies_s.append(start.elapsed_time(stop) * 1e-3)

        energy_j: Optional[float] = None
        if self._probe is None:
            notes_parts.append("INA3221: probe not available on this host")
        elif power_before_mw is not None:
            power_after_mw = self._probe.read_total_power_mw()
            if power_after_mw is None:
                notes_parts.append("INA3221: post-window power read failed")
            else:
                # Average power across the window, multiplied by window time.
                # This double-counts if the workload is bursty (e.g., short
                # kernels with idle gaps) but for back-to-back inference
                # trials this is close enough.
                median_latency_s = statistics.median(latencies_s)
                window_s = median_latency_s * self.timed_trials
                avg_power_w = (power_before_mw + power_after_mw) / 2.0 * 1e-3
                # Per-trial mean energy
                energy_j = (avg_power_w * window_s) / self.timed_trials
                # Warn if the window is below INA3221's 8ms averaging period
                if window_s < 8e-3:
                    notes_parts.append(f"INA3221: window {window_s*1e3:.1f}ms "
                                       "below 8ms averaging period; energy noisy")
                notes_parts.append(f"rails={[r.name for r in self._probe.rails]}")

        return latencies_s, energy_j, "; ".join(notes_parts)
