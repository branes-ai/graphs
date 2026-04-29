"""
Working-set-size sweep for cache-bandwidth measurement.

Logarithmic sweep over buffer sizes from L1-fitting (8 KiB) to
deep-DRAM (64 MiB by default). For each size, runs a stream-style
kernel for a target wallclock duration and emits effective
bandwidth + measured energy when available.

The resulting curve has plateaus at L1, L2, L3, and DRAM with
transition steps at the cache capacity boundaries. The transition
points are interpreted by ``cache_sweep.analysis.detect_levels``.

Energy capture is best-effort:
- On Linux x86, RAPL is read via ``/sys/class/powercap/intel-rapl``.
- On other platforms (or when RAPL is unavailable), the sweep
  reports ``energy_joules=None`` and downstream analysis falls back
  to the analytical TechnologyProfile estimate. The CLI emits a
  clear warning so the operator knows the result is time-only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAVE_NUMPY = False


# Default sweep range: 8 KiB up to 64 MiB. Covers L1 (~32-48 KiB),
# L2 (~256 KiB - 2 MiB), L3 (~16 - 64 MiB), and a few points into
# DRAM. Log-spaced so the bandwidth curve has comparable resolution
# at each level.
_DEFAULT_MIN_BYTES = 8 * 1024
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_POINTS = 24


@dataclass
class SweepConfig:
    """Parameters for a working-set-size sweep."""
    min_bytes: int = _DEFAULT_MIN_BYTES
    max_bytes: int = _DEFAULT_MAX_BYTES
    num_points: int = _DEFAULT_POINTS
    target_seconds_per_point: float = 0.5  # wallclock per size point
    repeats_per_point: int = 3              # take min latency over N runs
    capture_energy: bool = True             # try RAPL on Linux x86
    rapl_zone: str = "intel-rapl:0"         # package energy zone

    def working_set_sizes(self) -> List[int]:
        """Log-spaced byte counts spanning the configured range."""
        if self.num_points < 2:
            return [self.min_bytes]
        log_min = math.log2(self.min_bytes)
        log_max = math.log2(self.max_bytes)
        return [
            int(round(2 ** (log_min + i * (log_max - log_min)
                            / (self.num_points - 1))))
            for i in range(self.num_points)
        ]


@dataclass
class WorkingSetPoint:
    """One (working_set_size, bandwidth, energy) measurement."""
    bytes_resident: int            # working-set size in bytes
    iterations: int                # times the buffer was streamed
    elapsed_seconds: float         # min wallclock across repeats
    bandwidth_gbps: float          # bytes_resident * iterations / elapsed
    energy_joules: Optional[float] = None
    energy_per_byte_pj: Optional[float] = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------
# RAPL energy probe
# --------------------------------------------------------------------

class _RAPLProbe:
    """Read package energy from /sys/class/powercap/intel-rapl.

    Returns ``None`` from ``sample()`` when RAPL is unavailable or the
    counter wraps. Best-effort -- the caller treats ``None`` as
    "no energy capture available."
    """
    def __init__(self, zone: str = "intel-rapl:0"):
        self._path = Path(
            f"/sys/class/powercap/{zone}/energy_uj"
        )
        self._available = self._path.exists() and self._path.is_file()

    @property
    def available(self) -> bool:
        return self._available

    def sample(self) -> Optional[float]:
        if not self._available:
            return None
        try:
            uj = int(self._path.read_text().strip())
        except (OSError, ValueError):
            return None
        return uj / 1e6  # joules


# --------------------------------------------------------------------
# Stream kernel
# --------------------------------------------------------------------

def _stream_once(buf, iterations: int) -> None:
    """Touch every byte of ``buf`` ``iterations`` times.

    Uses an in-place bitwise XOR (``np.bitwise_xor(buf, 1, out=buf)``)
    rather than a sum-reduce: in-place ops do a full read + write pass
    over every element with no side-allocation, and an XOR with 1
    cannot be optimised away because the result is observable
    (the buffer mutates between iterations). On modern CPUs this
    saturates the cache / memory bandwidth at every working-set size,
    which is what makes the L1 / L2 / L3 / DRAM plateaus visible in
    the bandwidth-vs-size curve.

    A pure ``np.add.reduce`` was tried first but is dominated by
    NumPy call overhead on sub-megabyte buffers, masking the cache
    transitions.
    """
    for _ in range(iterations):
        np.bitwise_xor(buf, 1, out=buf)


def _calibrate_iterations(buf, target_seconds: float) -> int:
    """Find an iteration count that hits ~target_seconds for ``buf``.

    Doubles the iteration count until the kernel takes at least
    ``target_seconds / 8``, then projects the iteration count so the
    measurement loop targets ``target_seconds``. Single-shot
    measurement at the projected count goes back to the caller.
    """
    iters = 1
    elapsed = 0.0
    while elapsed < target_seconds / 8 and iters < 1 << 20:
        t0 = time.perf_counter()
        _stream_once(buf, iters)
        elapsed = time.perf_counter() - t0
        if elapsed < target_seconds / 8:
            iters *= 2
    if elapsed <= 0:
        return iters
    return max(1, int(iters * target_seconds / elapsed))


# --------------------------------------------------------------------
# Main sweep entry point
# --------------------------------------------------------------------

def run_sweep(
    config: Optional[SweepConfig] = None,
) -> Iterator[WorkingSetPoint]:
    """Run the working-set-size sweep and yield per-point results.

    Streaming generator so the CLI can checkpoint each point as it
    completes -- a 24-point sweep at 0.5 s / point is ~24 s of useful
    work, but a few points may take longer on slow systems.

    Raises ``ImportError`` if NumPy is unavailable; the kernel is
    NumPy-vectorised by construction.
    """
    if not _HAVE_NUMPY:
        raise ImportError(
            "cache_sweep requires NumPy. Install it with `pip install numpy`."
        )
    if config is None:
        config = SweepConfig()

    rapl = _RAPLProbe(zone=config.rapl_zone) if config.capture_energy else None
    rapl_ok = rapl is not None and rapl.available

    for size in config.working_set_sizes():
        # int8 array sized exactly to the working-set target. NumPy
        # arrays are 64-byte aligned by default -> cache lines align.
        buf = np.zeros(size, dtype=np.int8)
        iters = _calibrate_iterations(buf, config.target_seconds_per_point)

        # Repeats: take the min wallclock + average energy
        best_elapsed = float("inf")
        energy_samples: List[float] = []
        for _ in range(config.repeats_per_point):
            energy_start = rapl.sample() if rapl_ok else None
            t0 = time.perf_counter()
            _stream_once(buf, iters)
            elapsed = time.perf_counter() - t0
            energy_end = rapl.sample() if rapl_ok else None

            if elapsed < best_elapsed:
                best_elapsed = elapsed
            if (energy_start is not None
                    and energy_end is not None
                    and energy_end >= energy_start):
                energy_samples.append(energy_end - energy_start)

        bytes_streamed = float(size) * iters
        bandwidth_gbps = (
            bytes_streamed / best_elapsed / 1e9
            if best_elapsed > 0 else 0.0
        )
        energy_j = (
            sum(energy_samples) / len(energy_samples)
            if energy_samples else None
        )
        energy_per_byte_pj = (
            (energy_j / bytes_streamed) * 1e12
            if energy_j is not None and bytes_streamed > 0 else None
        )

        yield WorkingSetPoint(
            bytes_resident=size,
            iterations=iters,
            elapsed_seconds=best_elapsed,
            bandwidth_gbps=bandwidth_gbps,
            energy_joules=energy_j,
            energy_per_byte_pj=energy_per_byte_pj,
            extra={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rapl_available": rapl_ok,
                "repeats": config.repeats_per_point,
            },
        )


__all__ = [
    "SweepConfig",
    "WorkingSetPoint",
    "run_sweep",
]
