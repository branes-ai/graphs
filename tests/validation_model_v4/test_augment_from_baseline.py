"""Unit tests for the measurement-priority sweep augmenter.

Pins:
  * ``augment_sweep_from_baseline`` rewrites concrete-measurement
    regime labels onto the sweep entry's ``regime_per_hw[hw]``
  * AMBIGUOUS measurements leave the analytical label intact
    (the runner skips AMBIGUOUS records, so we don't want to shrink
    the validation pool just because a shape sits between bounds)
  * Idempotent: re-running with the same baselines makes no further
    label changes
  * No-baseline (hw, op) pair: tolerated as a clean no-op (returns
    zero counts)
"""

import csv
import json
from pathlib import Path
from typing import Optional

import pytest

from validation.model_v4.sweeps._augment_from_baseline import (
    augment_sweep_from_baseline,
)
from validation.model_v4.sweeps.classify import Regime


# ---------------------------------------------------------------------------
# Fixtures: minimal sweep + baseline trees in tmp_path
# ---------------------------------------------------------------------------


def _write_sweep(path: Path, op: str, shapes: list[dict]) -> None:
    payload = {
        "op": op,
        "purpose": "validation",
        "generator_seed": 0,
        "generated_against_hardware": ["i7_12700k"],
        "shapes": shapes,
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_baseline(
    baseline_dir: Path,
    hw_key: str,
    op: str,
    rows: list[dict],
) -> None:
    """Write a CSV in the cache.py-expected format."""
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / f"{hw_key}_{op}.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hardware", "op", "shape", "dtype",
                "latency_s", "energy_j", "trial_count", "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row.setdefault("notes", "")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Concrete measurement -> sweep label updated
# ---------------------------------------------------------------------------


def test_augmenter_rewrites_concrete_alu_measurement(tmp_path):
    """A matmul shape with measured flops_util >= 0.70 (so
    infer_regime_measured returns ALU_BOUND) should have its sweep
    label rewritten to ``alu_bound``, even if the analytical
    classifier said ``dram_bound``."""
    sweep_path = tmp_path / "matmul_validation.json"

    # Shape (128, 8192, 8192) fp16. flops = 17.18G; pick a latency
    # such that flops_util > 0.70 on Jetson Orin Nano (peak 7.10 TFLOPS).
    # latency = 17.18e9 / (0.70 * 7.10e12) = 3.46 ms -> just above 70%.
    # Use 3.0 ms to be comfortably ALU_BOUND.
    target_shape = [128, 8192, 8192]
    _write_sweep(sweep_path, "matmul", [
        {
            "shape": target_shape,
            "dtype": "fp16",
            "regime_per_hw": {"jetson_orin_nano_8gb": "dram_bound"},
        },
    ])
    baseline_dir = tmp_path / "baselines"
    _write_baseline(baseline_dir, "jetson_orin_nano_8gb", "matmul", [
        {
            "hardware": "jetson_orin_nano_8gb",
            "op": "matmul",
            "shape": "x".join(str(d) for d in target_shape),
            "dtype": "fp16",
            "latency_s": "3.0e-3",
            "energy_j": "0.10",
            "trial_count": "10",
        },
    ])

    rewritten, kept_amb, no_base = augment_sweep_from_baseline(
        sweep_path, "jetson_orin_nano_8gb", baseline_dir=baseline_dir,
    )

    assert rewritten == 1
    assert kept_amb == 0
    assert no_base == 0

    sweep = json.loads(sweep_path.read_text())
    new_label = sweep["shapes"][0]["regime_per_hw"]["jetson_orin_nano_8gb"]
    assert new_label == Regime.ALU_BOUND.value


# ---------------------------------------------------------------------------
# AMBIGUOUS measurement -> label preserved
# ---------------------------------------------------------------------------


def test_augmenter_keeps_ambiguous_measurement_intact(tmp_path):
    """A shape whose measured util falls below 70% on BOTH compute and
    bandwidth produces an AMBIGUOUS measured regime. The augmenter
    must leave the analytical label intact (rewriting to AMBIGUOUS
    would cause the runner to skip the record entirely)."""
    sweep_path = tmp_path / "matmul_validation.json"

    # Pick a slow latency so flops_util < 0.70 AND bw_util < 0.70.
    # (128, 8192, 8192) fp16: flops 17.18G, ws ~ 268 MB. Latency 50 ms ->
    # flops_util = 17.18e9 / (50e-3 * 7.10e12) = 0.048 (well below 0.70)
    # bw_util = 268e6 / (50e-3 * 102e9) = 0.052 (well below 0.70)
    target_shape = [128, 8192, 8192]
    _write_sweep(sweep_path, "matmul", [
        {
            "shape": target_shape,
            "dtype": "fp16",
            "regime_per_hw": {"jetson_orin_nano_8gb": "dram_bound"},
        },
    ])
    baseline_dir = tmp_path / "baselines"
    _write_baseline(baseline_dir, "jetson_orin_nano_8gb", "matmul", [
        {
            "hardware": "jetson_orin_nano_8gb",
            "op": "matmul",
            "shape": "x".join(str(d) for d in target_shape),
            "dtype": "fp16",
            "latency_s": "50.0e-3",
            "energy_j": "0.50",
            "trial_count": "10",
        },
    ])

    rewritten, kept_amb, no_base = augment_sweep_from_baseline(
        sweep_path, "jetson_orin_nano_8gb", baseline_dir=baseline_dir,
    )

    assert rewritten == 0
    assert kept_amb == 1
    assert no_base == 0

    # Label preserved as the original "dram_bound", NOT overwritten to "ambiguous"
    sweep = json.loads(sweep_path.read_text())
    new_label = sweep["shapes"][0]["regime_per_hw"]["jetson_orin_nano_8gb"]
    assert new_label == "dram_bound"


# ---------------------------------------------------------------------------
# Idempotent
# ---------------------------------------------------------------------------


def test_augmenter_is_idempotent(tmp_path):
    """Re-running on the same sweep + baseline produces the same JSON."""
    sweep_path = tmp_path / "matmul_validation.json"
    target_shape = [128, 8192, 8192]
    _write_sweep(sweep_path, "matmul", [
        {
            "shape": target_shape,
            "dtype": "fp16",
            "regime_per_hw": {"jetson_orin_nano_8gb": "dram_bound"},
        },
    ])
    baseline_dir = tmp_path / "baselines"
    _write_baseline(baseline_dir, "jetson_orin_nano_8gb", "matmul", [
        {
            "hardware": "jetson_orin_nano_8gb",
            "op": "matmul",
            "shape": "x".join(str(d) for d in target_shape),
            "dtype": "fp16",
            "latency_s": "3.0e-3",
            "energy_j": "0.10",
            "trial_count": "10",
        },
    ])

    augment_sweep_from_baseline(
        sweep_path, "jetson_orin_nano_8gb", baseline_dir=baseline_dir,
    )
    snapshot = sweep_path.read_text()

    # Run again -- no further changes
    rewritten, _, _ = augment_sweep_from_baseline(
        sweep_path, "jetson_orin_nano_8gb", baseline_dir=baseline_dir,
    )
    assert rewritten == 0
    assert sweep_path.read_text() == snapshot


# ---------------------------------------------------------------------------
# No baseline -> clean no-op
# ---------------------------------------------------------------------------


def test_augmenter_handles_missing_baseline(tmp_path):
    """When no baseline CSV exists for the (hw, op) pair, the augmenter
    returns zero counts and leaves the sweep untouched."""
    sweep_path = tmp_path / "matmul_validation.json"
    _write_sweep(sweep_path, "matmul", [
        {
            "shape": [256, 256, 256],
            "dtype": "fp16",
            "regime_per_hw": {"jetson_orin_nano_8gb": "dram_bound"},
        },
    ])
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()  # empty dir, no CSV

    rewritten, kept_amb, no_base = augment_sweep_from_baseline(
        sweep_path, "jetson_orin_nano_8gb", baseline_dir=baseline_dir,
    )

    assert (rewritten, kept_amb, no_base) == (0, 0, 0)


# ---------------------------------------------------------------------------
# dtype mismatch -> entry skipped
# ---------------------------------------------------------------------------


def test_augmenter_skips_dtype_mismatch(tmp_path):
    """The augmenter only touches entries whose dtype matches the
    target's calibration dtype. Entries with other dtypes are left
    alone."""
    sweep_path = tmp_path / "matmul_validation.json"
    _write_sweep(sweep_path, "matmul", [
        {
            "shape": [256, 256, 256],
            "dtype": "fp32",  # i7's dtype, not Jetson's fp16
            "regime_per_hw": {"jetson_orin_nano_8gb": "dram_bound"},
        },
    ])
    baseline_dir = tmp_path / "baselines"
    _write_baseline(baseline_dir, "jetson_orin_nano_8gb", "matmul", [
        # Baseline row exists but has fp16 dtype, won't match the
        # fp32 sweep entry above
        {
            "hardware": "jetson_orin_nano_8gb",
            "op": "matmul",
            "shape": "256x256x256",
            "dtype": "fp16",
            "latency_s": "3.0e-3",
            "energy_j": "0.10",
            "trial_count": "10",
        },
    ])

    rewritten, kept_amb, no_base = augment_sweep_from_baseline(
        sweep_path, "jetson_orin_nano_8gb", baseline_dir=baseline_dir,
    )

    # The fp32 entry is filtered out (dtype != target.dtype="fp16")
    # before the baseline-lookup; counts are all zero.
    assert (rewritten, kept_amb, no_base) == (0, 0, 0)
    sweep = json.loads(sweep_path.read_text())
    assert sweep["shapes"][0]["regime_per_hw"]["jetson_orin_nano_8gb"] == "dram_bound"
