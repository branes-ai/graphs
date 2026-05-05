"""Disk-backed cache of (hardware, op, shape, dtype) -> Measurement.

The dev loop is prediction-only after the first ground-truth pass per
hardware: edit the analyzer, re-run validation, get a heatmap in
seconds. That loop relies on this cache.

Cache layout (one file per hardware):

    validation/model_v4/results/baselines/<hardware>_<op>.csv

CSV (not JSON) so the baselines are human-readable in `git diff`. Each
row is one measurement; the (op, shape, dtype) tuple is the lookup
key.

Cache invalidation is **explicit**: the user runs
``capture_ground_truth`` to refresh. The harness never auto-refreshes
on cache miss -- that would silently move the validation reference.
A miss returns None and the harness reports "no baseline" for that
shape, leaving the user to decide whether to capture or skip.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from validation.model_v4.ground_truth.base import Measurement


# Cache files live under validation/model_v4/results/baselines/
# (next to the harness, not under tests/).
DEFAULT_BASELINE_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "baselines"
)


@dataclass(frozen=True)
class CacheKey:
    """Lookup key. All fields part of the JSON / CSV row."""
    hardware: str
    op: str               # "matmul" or "linear"
    shape: Tuple[int, ...]
    dtype: str

    def to_csv_row(self) -> dict:
        return {
            "hardware": self.hardware,
            "op": self.op,
            "shape": "x".join(str(d) for d in self.shape),
            "dtype": self.dtype,
        }


# CSV columns. Order is stable so `git diff` on baselines is meaningful.
_FIELDNAMES = [
    "hardware", "op", "shape", "dtype",
    "latency_s", "energy_j", "trial_count", "notes",
]


def _baseline_path(hardware: str, op: str,
                   baseline_dir: Path = DEFAULT_BASELINE_DIR) -> Path:
    """One file per (hardware, op) pair so each PR's diff is small."""
    return baseline_dir / f"{hardware}_{op}.csv"


def _parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split("x"))


def _parse_optional_float(s: str) -> Optional[float]:
    if s == "" or s.lower() == "none":
        return None
    return float(s)


def _format_optional_float(v: Optional[float]) -> str:
    return "" if v is None else repr(v)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def load_baseline(
    hardware: str,
    op: str,
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
) -> dict[CacheKey, Measurement]:
    """Read the per-(hardware, op) baseline CSV into a {key: Measurement}
    dict. Returns {} if the file does not exist (no auto-creation)."""
    path = _baseline_path(hardware, op, baseline_dir)
    if not path.exists():
        return {}
    out: dict[CacheKey, Measurement] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = CacheKey(
                hardware=row["hardware"],
                op=row["op"],
                shape=_parse_shape(row["shape"]),
                dtype=row["dtype"],
            )
            out[key] = Measurement(
                latency_s=float(row["latency_s"]),
                energy_j=_parse_optional_float(row["energy_j"]),
                trial_count=int(row["trial_count"]),
                notes=row.get("notes", ""),
            )
    return out


def lookup(
    key: CacheKey,
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
) -> Optional[Measurement]:
    """Return the Measurement for ``key`` or None.

    Repeated calls re-open the CSV; harness runners that need to look
    up many keys should call ``load_baseline`` once and keep the dict.
    """
    return load_baseline(key.hardware, key.op, baseline_dir=baseline_dir).get(key)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def store_many(
    entries: Iterable[Tuple[CacheKey, Measurement]],
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    merge: bool = True,
) -> dict[Tuple[str, str], int]:
    """Persist a batch of measurements.

    With ``merge=True`` (default) the existing baseline file is
    re-read, the new measurements overwrite same-key rows, and the
    union is rewritten. With ``merge=False`` the file is replaced.

    Returns a {(hardware, op): count_written} report.

    The CSV is sorted by (op, dtype, shape) for stable diffs.
    """
    grouped: dict[Tuple[str, str], dict[CacheKey, Measurement]] = {}
    for key, m in entries:
        grouped.setdefault((key.hardware, key.op), {})[key] = m

    written: dict[Tuple[str, str], int] = {}
    for (hardware, op), new_rows in grouped.items():
        existing = (
            load_baseline(hardware, op, baseline_dir=baseline_dir) if merge else {}
        )
        existing.update(new_rows)
        path = _baseline_path(hardware, op, baseline_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        sorted_keys = sorted(existing.keys(),
                             key=lambda k: (k.op, k.dtype, k.shape))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            w.writeheader()
            for k in sorted_keys:
                m = existing[k]
                row = k.to_csv_row()
                row.update({
                    "latency_s": repr(m.latency_s),
                    "energy_j": _format_optional_float(m.energy_j),
                    "trial_count": str(m.trial_count),
                    "notes": m.notes,
                })
                w.writerow(row)
        written[(hardware, op)] = len(existing)
    return written


def store(
    key: CacheKey, measurement: Measurement,
    *, baseline_dir: Path = DEFAULT_BASELINE_DIR,
) -> None:
    """Convenience wrapper for storing a single (key, measurement) pair."""
    store_many([(key, measurement)], baseline_dir=baseline_dir)
