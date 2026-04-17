"""
Power Measurement Backends for Bottom-Up Benchmarks

Extends the existing MeasurementCollector ecosystem (see collectors.py)
with platform-specific energy measurement backends:

- RAPLPowerCollector: Intel RAPL energy counters (x86 Linux)
- TegrastatsPowerCollector: Jetson tegrastats daemon (Orin Nano / AGX)
- NoOpPowerCollector: fallback when no power domain is available
- auto_select_power_collector: probes available backends, returns best

The existing PowerCollector (NVML/nvidia-smi for discrete GPUs) is
preferred when a CUDA device is the target. These new collectors cover
the CPU-side and edge-device measurement gaps.

Usage:
    collector = auto_select_power_collector("cpu")
    collector.start()
    # ... run benchmark ...
    collector.stop()
    measurement = collector.get_measurement()
    print(f"Energy: {measurement.energy_joules:.6f} J")
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .collectors import (
    MeasurementCollector,
    PowerCollector,
    PowerMeasurement,
)

MIN_MEASUREMENT_DURATION_MS = 100.0


def _rapl_available() -> bool:
    """Check if Intel RAPL sysfs interface is readable."""
    energy_path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    try:
        energy_path.read_text()
        return True
    except (PermissionError, FileNotFoundError, OSError):
        return False


def _tegrastats_available() -> bool:
    """Check if tegrastats binary exists (Jetson platforms)."""
    try:
        result = subprocess.run(
            ["which", "tegrastats"],
            capture_output=True, timeout=2.0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _nvml_available() -> bool:
    """Check if pynvml can initialize (discrete NVIDIA GPU present)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


class RAPLPowerCollector(MeasurementCollector):
    """
    Intel RAPL energy counter collector.

    Reads /sys/class/powercap/intel-rapl:N/energy_uj before and after
    the measurement scope. The delta divided by wall-clock time gives
    average power. No background thread needed -- just two reads.

    Requires read access to /sys/class/powercap/intel-rapl:*/energy_uj.
    On most distros this is readable by default; if not, run:
        sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj

    RAPL domains (typical desktop/server):
        intel-rapl:0        package-0 (entire socket)
        intel-rapl:0:0      core (CPU cores only)
        intel-rapl:0:1      uncore (memory controller, L3, etc.)
        intel-rapl:0:2      dram

    By default this collector reads package-0 (the whole socket) which
    is the most useful single number for bottom-up validation.
    """

    def __init__(self, domain: str = "intel-rapl:0"):
        super().__init__()
        self._domain = domain
        self._energy_path = Path(f"/sys/class/powercap/{domain}/energy_uj")
        self._energy_start_uj: int = 0
        self._energy_end_uj: int = 0
        self._max_energy_uj: Optional[int] = None

        max_path = Path(f"/sys/class/powercap/{domain}/max_energy_range_uj")
        try:
            self._max_energy_uj = int(max_path.read_text().strip())
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            pass

    def _read_energy_uj(self) -> int:
        return int(self._energy_path.read_text().strip())

    def start(self) -> None:
        self._energy_start_uj = self._read_energy_uj()
        self._start_time = time.perf_counter()
        self._running = True

    def stop(self) -> None:
        self._end_time = time.perf_counter()
        self._energy_end_uj = self._read_energy_uj()
        self._running = False

    def get_measurement(self) -> PowerMeasurement:
        duration_ms = self.duration_ms
        delta_uj = self._energy_end_uj - self._energy_start_uj

        # Handle counter wrap-around
        if delta_uj < 0 and self._max_energy_uj is not None:
            delta_uj += self._max_energy_uj

        energy_j = delta_uj * 1e-6
        duration_s = duration_ms / 1000.0
        avg_power = energy_j / duration_s if duration_s > 0 else 0.0

        return PowerMeasurement(
            duration_ms=duration_ms,
            avg_power_watts=avg_power,
            peak_power_watts=avg_power,
            energy_joules=energy_j,
            sampling_interval_ms=0.0,
            success=True,
        )

    def reset(self) -> None:
        super().reset()
        self._energy_start_uj = 0
        self._energy_end_uj = 0


class TegrastatsPowerCollector(MeasurementCollector):
    """
    Jetson tegrastats power collector.

    Launches ``tegrastats --interval <ms>`` in a background thread and
    parses the VDD_CPU, VDD_GPU, and VDD_SOC power rails. Summing
    these gives total SoC power.

    tegrastats output format (Orin):
        ... VDD_GPU_SOC 2596mW/2596mW VDD_CPU_CV 1534mW/1534mW ...

    Requires tegrastats to be in PATH (standard on JetPack).
    """

    _POWER_RAIL_RE = re.compile(
        r"(VDD_\w+)\s+(\d+)mW/(\d+)mW"
    )

    def __init__(self, interval_ms: int = 100):
        super().__init__()
        self._interval_ms = interval_ms
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._samples: List[float] = []
        self._lock = threading.Lock()

    def _reader_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for raw_line in self._process.stdout:
            if not self._running:
                break
            line = raw_line.decode("utf-8", errors="replace")
            total_mw = 0.0
            for match in self._POWER_RAIL_RE.finditer(line):
                total_mw += float(match.group(2))
            if total_mw > 0:
                with self._lock:
                    self._samples.append(total_mw / 1000.0)

    def start(self) -> None:
        self._samples = []
        self._running = True
        self._start_time = time.perf_counter()
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True,
            )
            self._reader_thread.start()
        except FileNotFoundError:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self._end_time = time.perf_counter()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def get_measurement(self) -> PowerMeasurement:
        with self._lock:
            samples = self._samples.copy()

        if not samples:
            return PowerMeasurement(
                duration_ms=self.duration_ms,
                success=False,
                error_message="No tegrastats power samples collected",
                sampling_interval_ms=float(self._interval_ms),
            )

        return PowerMeasurement(
            duration_ms=self.duration_ms,
            samples=samples,
            sampling_interval_ms=float(self._interval_ms),
            success=True,
        )

    def reset(self) -> None:
        super().reset()
        self._samples = []


class NoOpPowerCollector(MeasurementCollector):
    """
    Fallback collector when no power measurement backend is available.

    Returns a PowerMeasurement with success=False and a descriptive
    message so callers can distinguish "no power domain" from "power
    measurement failed."
    """

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._running = True

    def stop(self) -> None:
        self._end_time = time.perf_counter()
        self._running = False

    def get_measurement(self) -> PowerMeasurement:
        return PowerMeasurement(
            duration_ms=self.duration_ms,
            success=False,
            error_message="No power measurement backend available on this host",
        )


def auto_select_power_collector(
    device: str = "cpu",
    sampling_interval_ms: float = 100.0,
) -> MeasurementCollector:
    """
    Probe available power-measurement backends and return the best one.

    Selection priority:
    1. CUDA device requested + NVML available -> existing PowerCollector
    2. x86 Linux + RAPL readable -> RAPLPowerCollector
    3. Jetson + tegrastats in PATH -> TegrastatsPowerCollector
    4. Nothing available -> NoOpPowerCollector

    Args:
        device: Target device string ("cpu", "cuda", "cuda:0", etc.)
        sampling_interval_ms: Sampling interval for thread-based collectors

    Returns:
        A MeasurementCollector subclass ready to start()
    """
    if device.startswith("cuda") and _nvml_available():
        device_index = 0
        if ":" in device:
            try:
                device_index = int(device.split(":")[1])
            except (IndexError, ValueError):
                pass
        return PowerCollector(
            device_index=device_index,
            sampling_interval_ms=sampling_interval_ms,
        )

    if _rapl_available():
        return RAPLPowerCollector()

    if _tegrastats_available():
        return TegrastatsPowerCollector(interval_ms=int(sampling_interval_ms))

    return NoOpPowerCollector()
