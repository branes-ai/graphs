"""PyTorch CPU ground-truth measurer with optional Intel RAPL energy.

Used to capture per-shape latency (and energy when available) on real
CPU hardware so the v4 harness can validate the analyzer's predictions.

Latency:
    Wall-clock via ``time.perf_counter_ns`` over N trials, taking the
    median to reject outliers from OS noise / GC / context switches.
    A small warm-up run primes caches and JIT compilers before the
    timed window.

Energy:
    Intel RAPL counter at ``/sys/class/powercap/intel-rapl:0/energy_uj``
    (package-0 in microjoules, wraps at ``max_energy_range_uj``). We
    measure the RAPL delta over the same window as the latency
    measurement and report the mean per-trial energy.

    RAPL is **package-level**, not per-process. The measurer assumes
    the test machine is otherwise idle; a noisy system inflates the
    energy figure. The harness should warn (not fail) when system load
    exceeds a threshold.

If RAPL is unreadable (file missing, permission denied, counter wrap
detected, ...) the measurer returns ``Measurement(energy_j=None,
notes="RAPL: <reason>")``. Per the assertions module that lets the
record still pass overall on regime + latency.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from validation.model_v4.ground_truth.base import Measurement


# Default warm-up + trial counts. Tunable per-call.
DEFAULT_WARMUP_TRIALS = 3
DEFAULT_TIMED_TRIALS = 11    # odd so median is well-defined

# RAPL package-0 lives here on most Intel systems. AMD CPUs expose
# similar counters under powercap/amd_energy_*, not implemented yet.
_RAPL_PACKAGE_PATH = Path("/sys/class/powercap/intel-rapl:0")


# ---------------------------------------------------------------------------
# RAPL probe
# ---------------------------------------------------------------------------


@dataclass
class _RAPLProbe:
    """Minimal RAPL energy reader. None when unavailable."""
    energy_uj_path: Path
    max_energy_range_uj: int
    name: str             # "package-0", "package-1", ...

    def read_energy_uj(self) -> int:
        return int(self.energy_uj_path.read_text().strip())

    def delta_uj(self, before_uj: int, after_uj: int) -> int:
        """Handle counter wrap: if ``after < before`` we wrapped once."""
        if after_uj >= before_uj:
            return after_uj - before_uj
        return (self.max_energy_range_uj - before_uj) + after_uj + 1


def _detect_rapl(path: Path = _RAPL_PACKAGE_PATH) -> Optional[_RAPLProbe]:
    """Open the RAPL package-0 counter or return None with no surprise.

    Returns None for any of: file missing, permission denied, malformed
    contents, or zero max_energy_range. Callers should treat None as
    "energy not measured" rather than crash.
    """
    energy_path = path / "energy_uj"
    range_path = path / "max_energy_range_uj"
    name_path = path / "name"
    try:
        if not energy_path.exists() or not range_path.exists():
            return None
        # Permission probe -- read once before claiming the probe is ready
        _ = int(energy_path.read_text().strip())
        max_uj = int(range_path.read_text().strip())
        if max_uj <= 0:
            return None
        name = name_path.read_text().strip() if name_path.exists() else "package-0"
        return _RAPLProbe(energy_uj_path=energy_path,
                          max_energy_range_uj=max_uj, name=name)
    except (OSError, PermissionError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Measurer
# ---------------------------------------------------------------------------


@dataclass
class PyTorchCPUMeasurer:
    """Wall-clock + RAPL measurer for PyTorch on CPU.

    Stateful only insofar as it caches the RAPL probe handle (one
    file-read per ``__init__`` then reused).
    """
    hardware: str                 # caller supplies the hardware key
    warmup_trials: int = DEFAULT_WARMUP_TRIALS
    timed_trials: int = DEFAULT_TIMED_TRIALS
    name: str = "pytorch_cpu_rapl"
    _probe: Optional[_RAPLProbe] = None

    def __post_init__(self) -> None:
        # Use a private setter style so re-detection is possible if the
        # caller wants to skip RAPL entirely (set ._probe = None).
        if self._probe is None:
            self._probe = _detect_rapl()

    # -- public API ----------------------------------------------------------

    def measure(self, model, inputs) -> Measurement:
        """Run ``model(*inputs)`` warmup + N times, return Measurement."""
        # Warm-up (results discarded; primes caches and any one-time JIT)
        for _ in range(self.warmup_trials):
            model(*inputs)

        # Timed window
        latencies_s, energy_j, notes = self._timed_window(model, inputs)
        median_latency_s = statistics.median(latencies_s)

        return Measurement(
            latency_s=median_latency_s,
            energy_j=energy_j,
            trial_count=self.timed_trials,
            notes=notes,
        )

    # -- internals -----------------------------------------------------------

    def _timed_window(self, model, inputs) -> Tuple[list[float], Optional[float], str]:
        """Run the timed trials. Returns (latencies, mean_energy, notes)."""
        notes_parts: list[str] = []
        rapl_before: Optional[int] = None
        rapl_at: Optional[int] = None

        # Capture RAPL bracketing the entire timed window (one read each
        # side, not per-trial -- per-trial reads add measurable noise to
        # very small benchmarks).
        if self._probe is not None:
            try:
                rapl_before = self._probe.read_energy_uj()
            except (OSError, ValueError) as e:
                rapl_before = None
                notes_parts.append(f"RAPL pre-read failed: {e}")

        latencies_s: list[float] = []
        for _ in range(self.timed_trials):
            t0 = time.perf_counter_ns()
            model(*inputs)
            t1 = time.perf_counter_ns()
            latencies_s.append((t1 - t0) * 1e-9)

        if self._probe is not None and rapl_before is not None:
            try:
                rapl_at = self._probe.read_energy_uj()
            except (OSError, ValueError) as e:
                rapl_at = None
                notes_parts.append(f"RAPL post-read failed: {e}")

        energy_j: Optional[float] = None
        if (self._probe is not None
                and rapl_before is not None and rapl_at is not None):
            delta_uj = self._probe.delta_uj(rapl_before, rapl_at)
            mean_energy_uj = delta_uj / self.timed_trials
            energy_j = mean_energy_uj * 1e-6
        elif self._probe is None:
            notes_parts.append("RAPL: probe not available on this host")

        return latencies_s, energy_j, "; ".join(notes_parts)
