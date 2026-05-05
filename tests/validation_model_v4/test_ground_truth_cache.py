"""Tests for validation/model_v4/ground_truth/cache.py.

The cache is the spine of the v4 dev loop: validation runs against
cached ground truth so iterating on the analyzer takes seconds, not
minutes. These tests lock in:

- write -> read round-trip preserves all fields including Optional energy
- merge=True overlays without dropping unrelated rows
- one file per (hardware, op) to keep PR diffs small and reviewable
- baseline rows are sorted for stable git diffs
- lookup of a missing key returns None (not raises) -- harness can
  then report "no baseline" rather than crashing
"""

import csv

from validation.model_v4.ground_truth.base import Measurement
from validation.model_v4.ground_truth.cache import (
    CacheKey,
    load_baseline,
    lookup,
    store,
    store_many,
    _baseline_path,
)


# ---------------------------------------------------------------------------
# CacheKey
# ---------------------------------------------------------------------------


def test_cache_key_is_hashable_and_equal():
    a = CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32")
    b = CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32")
    assert a == b
    assert hash(a) == hash(b)
    # different shape -> different key
    c = CacheKey("i7_12700k", "matmul", (1024, 1024, 2048), "fp32")
    assert a != c


def test_cache_key_csv_row_uses_x_separator():
    """Shape encoded as e.g. '1024x1024x1024' so it's atomic in CSV."""
    k = CacheKey("i7_12700k", "matmul", (1024, 768, 256), "fp16")
    row = k.to_csv_row()
    assert row["shape"] == "1024x768x256"
    assert row["dtype"] == "fp16"


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def _make_records():
    return [
        (CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32"),
         Measurement(latency_s=0.0008, energy_j=0.05, trial_count=100, notes="warm")),
        (CacheKey("i7_12700k", "matmul", (2048, 2048, 2048), "fp32"),
         Measurement(latency_s=0.012, energy_j=0.6, trial_count=100)),
        (CacheKey("i7_12700k", "matmul", (4096, 4096, 4096), "fp32"),
         Measurement(latency_s=0.3, energy_j=None, trial_count=10,
                     notes="RAPL unreadable")),
    ]


def test_store_and_load_round_trip(tmp_path):
    store_many(_make_records(), baseline_dir=tmp_path, merge=False)
    loaded = load_baseline("i7_12700k", "matmul", baseline_dir=tmp_path)
    assert len(loaded) == 3
    for key, m in _make_records():
        assert key in loaded
        got = loaded[key]
        assert got.latency_s == m.latency_s
        assert got.energy_j == m.energy_j
        assert got.trial_count == m.trial_count
        assert got.notes == m.notes


def test_optional_energy_is_preserved(tmp_path):
    """Energy=None must round-trip as None, not become 0.0 or empty string."""
    key = CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32")
    store(key, Measurement(latency_s=1e-3, energy_j=None, trial_count=10),
          baseline_dir=tmp_path)
    loaded = load_baseline("i7_12700k", "matmul", baseline_dir=tmp_path)
    assert loaded[key].energy_j is None


def test_lookup_missing_key_returns_none(tmp_path):
    store_many(_make_records(), baseline_dir=tmp_path, merge=False)
    missing = CacheKey("i7_12700k", "matmul", (9999, 9999, 9999), "fp32")
    assert lookup(missing, baseline_dir=tmp_path) is None


def test_lookup_when_baseline_file_absent_returns_none(tmp_path):
    """No file = no baseline. Harness should report 'no baseline' rather
    than crash."""
    key = CacheKey("h100_sxm5_80gb", "matmul", (1024, 1024, 1024), "fp16")
    assert lookup(key, baseline_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# merge semantics
# ---------------------------------------------------------------------------


def test_merge_overlays_without_dropping_other_rows(tmp_path):
    """Storing one new measurement must not delete unrelated existing rows."""
    store_many(_make_records(), baseline_dir=tmp_path, merge=False)

    # Now store a new record and an overwrite of an existing one
    new = [
        (CacheKey("i7_12700k", "matmul", (512, 512, 512), "fp32"),
         Measurement(latency_s=0.0001, energy_j=0.005, trial_count=200)),
        # Overwrite of (1024, 1024, 1024)
        (CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32"),
         Measurement(latency_s=0.0007, energy_j=0.04, trial_count=200,
                     notes="re-measured")),
    ]
    store_many(new, baseline_dir=tmp_path, merge=True)

    loaded = load_baseline("i7_12700k", "matmul", baseline_dir=tmp_path)
    # Original 3 + 1 new + 0 (overwrite of 1024-cube) = 4 rows
    assert len(loaded) == 4
    # The overwritten row reflects the new measurement
    overwritten = loaded[CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32")]
    assert overwritten.latency_s == 0.0007
    assert overwritten.notes == "re-measured"
    # Original 4096 row is untouched
    assert (CacheKey("i7_12700k", "matmul", (4096, 4096, 4096), "fp32")
            in loaded)


def test_merge_false_replaces_file(tmp_path):
    store_many(_make_records(), baseline_dir=tmp_path, merge=False)
    new = [(CacheKey("i7_12700k", "matmul", (512, 512, 512), "fp32"),
            Measurement(latency_s=0.0001, energy_j=0.005, trial_count=200))]
    store_many(new, baseline_dir=tmp_path, merge=False)
    loaded = load_baseline("i7_12700k", "matmul", baseline_dir=tmp_path)
    assert len(loaded) == 1


def test_one_file_per_hardware_op_pair(tmp_path):
    """Storing a (i7_12700k, matmul) record must not affect the
    (i7_12700k, linear) baseline file."""
    store(CacheKey("i7_12700k", "matmul", (256, 256, 256), "fp32"),
          Measurement(latency_s=1e-4, energy_j=0.001, trial_count=100),
          baseline_dir=tmp_path)
    store(CacheKey("i7_12700k", "linear", (8, 64, 32), "fp32"),
          Measurement(latency_s=1e-5, energy_j=0.0001, trial_count=100),
          baseline_dir=tmp_path)
    matmul_path = _baseline_path("i7_12700k", "matmul", tmp_path)
    linear_path = _baseline_path("i7_12700k", "linear", tmp_path)
    assert matmul_path.exists()
    assert linear_path.exists()
    assert matmul_path != linear_path


# ---------------------------------------------------------------------------
# stable git diffs
# ---------------------------------------------------------------------------


def test_baseline_csv_rows_are_sorted_for_stable_diffs(tmp_path):
    """Re-storing the same data in different order must produce
    identical bytes on disk -- so `git diff` only highlights real
    changes."""
    records_a = _make_records()
    records_b = list(reversed(records_a))

    store_many(records_a, baseline_dir=tmp_path, merge=False)
    bytes_a = _baseline_path("i7_12700k", "matmul", tmp_path).read_bytes()

    store_many(records_b, baseline_dir=tmp_path, merge=False)
    bytes_b = _baseline_path("i7_12700k", "matmul", tmp_path).read_bytes()

    assert bytes_a == bytes_b


def test_baseline_csv_header_is_stable(tmp_path):
    """The CSV header order is the contract; if it ever changes, every
    historical baseline becomes unreadable."""
    expected = ["hardware", "op", "shape", "dtype",
                "latency_s", "energy_j", "trial_count", "notes"]
    store(CacheKey("i7_12700k", "matmul", (256, 256, 256), "fp32"),
          Measurement(latency_s=1e-4, energy_j=0.001, trial_count=100),
          baseline_dir=tmp_path)
    with _baseline_path("i7_12700k", "matmul", tmp_path).open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == expected
