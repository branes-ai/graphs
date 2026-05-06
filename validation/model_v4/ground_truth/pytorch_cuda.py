"""PyTorch CUDA ground-truth measurer with optional NVML energy.

Used to capture per-shape latency (and energy when available) on real
NVIDIA GPU hardware so the v4 harness can validate the analyzer's
predictions on accelerators (H100, A100, RTX, etc.).

Latency:
    GPU-accurate via ``torch.cuda.Event(enable_timing=True)``. CUDA event
    timing brackets only the kernel launch + execution window on the
    device, NOT host-side dispatch overhead -- which is the right thing
    to measure when we want to validate the analyzer's *kernel* roofline
    prediction. (Host overhead shows up in our launch_bound regime via a
    different mechanism: the analyzer's launch_overhead_s constant.)

    A small warm-up run primes any kernel autotune / cache before the
    timed window; ``torch.cuda.synchronize`` is called before the start
    event and after the stop event so the host-side measurement isn't
    polluted by uncompleted launches.

Energy:
    NVML's ``nvmlDeviceGetTotalEnergyConsumption`` returns the cumulative
    energy in millijoules since driver load (monotonic, wraps only at
    2^64 mJ which is effectively never). We bracket the timed window
    with two reads and take the delta.

    If ``nvmlDeviceGetTotalEnergyConsumption`` isn't supported on the
    device (older GPUs, integrated tegra), we fall back to
    ``nvmlDeviceGetPowerUsage`` (instantaneous milliwatts) sampled
    once and multiplied by latency -- a coarse approximation, marked
    as such in the Measurement notes.

    NVML is **device-level** (one reading per GPU). Like RAPL on CPU,
    a noisy host that's running other CUDA workloads will inflate the
    energy figure -- the harness assumes the test box is otherwise idle.

If NVML is unavailable (pynvml not installed, driver mismatch, etc.)
the measurer returns ``Measurement(energy_j=None,
notes="NVML: <reason>")``. Per the assertions module that lets the
record still pass overall on regime + latency.

Use this measurer for desktop/server NVIDIA GPUs. For Jetson devices
(Orin Nano/AGX/NX, Thor) use ``pytorch_jetson.py`` instead -- those
read INA3221 power rails via sysfs, which gives a more accurate
package-level number than the SoC-side NVML telemetry.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


# Default warm-up + trial counts. CUDA kernels are typically less noisy
# than CPU wall-clock so we can use a smaller trial count if desired,
# but the defaults match pytorch_cpu.py for cross-target comparability.
DEFAULT_WARMUP_TRIALS = 3
DEFAULT_TIMED_TRIALS = 11    # odd so median is well-defined


# ---------------------------------------------------------------------------
# NVML probe
# ---------------------------------------------------------------------------


@dataclass
class _NVMLProbe:
    """Minimal NVML energy reader. None when unavailable.

    Holds a device handle (opaque ``int`` in pynvml) and remembers
    whether total-energy is supported -- some Tesla cards and most
    consumer GPUs only expose power, not the cumulative energy counter.
    """
    handle: Any                    # pynvml device handle (opaque)
    supports_total_energy: bool
    pynvml: Any                    # the imported pynvml module
    name: str = "gpu-0"

    def read_energy_mj(self) -> Optional[int]:
        """Return cumulative energy since driver load in millijoules,
        or None if the device doesn't support the counter."""
        if not self.supports_total_energy:
            return None
        try:
            return int(self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except Exception:
            return None

    def read_power_mw(self) -> Optional[int]:
        """Instantaneous power draw in milliwatts, or None on error."""
        try:
            return int(self.pynvml.nvmlDeviceGetPowerUsage(self.handle))
        except Exception:
            return None


def _detect_nvml(device_index: int = 0) -> Optional[_NVMLProbe]:
    """Initialize NVML on the given device or return None with no surprise.

    Returns None for any of: pynvml not installed, NVML init failed,
    invalid device index, both energy and power queries unsupported.
    Callers should treat None as "energy not measured" rather than crash.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            # Best-effort cleanup -- if shutdown also fails (driver wedge),
            # there's nothing we can do; the caller still gets None.
            pass
        return None

    # Check support: try one read each; if both fail we can't measure energy.
    supports_total_energy = True
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception:
        supports_total_energy = False

    supports_power = True
    try:
        pynvml.nvmlDeviceGetPowerUsage(handle)
    except Exception:
        supports_power = False

    if not supports_total_energy and not supports_power:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            # Best-effort cleanup on the unsupported-device path.
            pass
        return None

    try:
        name_bytes = pynvml.nvmlDeviceGetName(handle)
        name = (name_bytes.decode() if isinstance(name_bytes, bytes) else str(name_bytes))
    except Exception:
        name = f"gpu-{device_index}"

    return _NVMLProbe(handle=handle, supports_total_energy=supports_total_energy,
                      pynvml=pynvml, name=name)


# ---------------------------------------------------------------------------
# Measurer
# ---------------------------------------------------------------------------


@dataclass
class PyTorchCUDAMeasurer:
    """cudaEvent + NVML measurer for PyTorch on a desktop/server NVIDIA GPU.

    Stateful only insofar as it caches the NVML device handle. Inputs
    are assumed to already be on the target device; the measurer does
    not move tensors -- the workload builder owns placement.
    """
    hardware: str                 # caller supplies the hardware key
    device_index: int = 0
    warmup_trials: int = DEFAULT_WARMUP_TRIALS
    timed_trials: int = DEFAULT_TIMED_TRIALS
    name: str = "pytorch_cuda_nvml"
    _probe: Optional[_NVMLProbe] = field(default=None)

    def __post_init__(self) -> None:
        if self._probe is None:
            self._probe = _detect_nvml(self.device_index)

    # -- public API ----------------------------------------------------------

    def measure(self, model, inputs):
        """Run ``model(*inputs)`` warmup + N times, return Measurement.

        Imports torch lazily so the test harness can import this module
        on a CPU-only machine without torch.cuda available.
        """
        import torch
        from validation.model_v4.ground_truth.base import Measurement

        if not torch.cuda.is_available():
            # Without CUDA we can't measure on a GPU; surface as a clean
            # error rather than crashing inside torch.cuda.Event below.
            return Measurement(
                latency_s=float("nan"),
                energy_j=None,
                trial_count=0,
                notes="CUDA: torch.cuda.is_available() returned False",
            )

        # Warm-up (results discarded; primes kernel autotune + caches)
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
        """Run the timed trials. Returns (latencies, energy, notes)."""
        notes_parts: list[str] = []

        # Energy bracketing: prefer cumulative-energy delta; fall back to
        # power * latency if the device doesn't support the counter.
        energy_before_mj: Optional[int] = None
        if self._probe is not None and self._probe.supports_total_energy:
            energy_before_mj = self._probe.read_energy_mj()
            if energy_before_mj is None:
                notes_parts.append("NVML: total-energy read failed pre-window")

        # cudaEvent timing per trial. Each event lives on the same device;
        # we synchronize after the stop event before reading elapsed time.
        latencies_s: list[float] = []
        for _ in range(self.timed_trials):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            stop.record()
            torch.cuda.synchronize(self.device_index)
            # elapsed_time returns milliseconds
            latencies_s.append(start.elapsed_time(stop) * 1e-3)

        # Median per-trial latency for the power-fallback energy estimate
        # (avoids inflating energy from a single tail-latency outlier).
        median_latency_s = statistics.median(latencies_s)

        energy_j: Optional[float] = None
        if self._probe is None:
            notes_parts.append("NVML: probe not available on this host")
        elif self._probe.supports_total_energy and energy_before_mj is not None:
            energy_after_mj = self._probe.read_energy_mj()
            if energy_after_mj is None:
                notes_parts.append("NVML: total-energy read failed post-window")
            else:
                # NVML monotonic counter -- straightforward subtraction.
                # Wraparound at 2^64 mJ is ~580 million years of full TDP,
                # so we don't bother handling it.
                delta_mj = energy_after_mj - energy_before_mj
                if delta_mj < 0:
                    notes_parts.append("NVML: energy counter went backwards "
                                       "(driver restart?); energy=None")
                else:
                    mean_energy_mj = delta_mj / self.timed_trials
                    energy_j = mean_energy_mj * 1e-3
        else:
            # Power-fallback path: one read of instantaneous power, multiply
            # by per-trial latency. Less accurate but better than nothing for
            # devices without the cumulative counter.
            power_mw = self._probe.read_power_mw()
            if power_mw is None:
                notes_parts.append("NVML: power read failed (no fallback)")
            else:
                energy_j = (power_mw * 1e-3) * median_latency_s
                notes_parts.append("NVML: power-fallback (no total-energy "
                                   f"counter); P={power_mw}mW * lat")

        return latencies_s, energy_j, "; ".join(notes_parts)

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """Optional: shut down NVML to release the handle. Safe to call
        multiple times. Not strictly required -- NVML cleans up on
        process exit."""
        if self._probe is not None:
            try:
                self._probe.pynvml.nvmlShutdown()
            except Exception:
                # Teardown is best-effort; NVML cleans up on process exit
                # anyway, so a shutdown failure here is not actionable.
                pass
            self._probe = None
