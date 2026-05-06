"""Tests for validation/model_v4/sweeps/_augment.py.

The augmenter must be:
* Additive: existing entries (shape, dtype, original target regimes)
  are never modified.
* Idempotent: re-running with the same target overwrites the same
  value, no duplication.
* Dtype-aware: only entries with matching dtype get classified for a
  given target.
* Schema-stable: the augmented JSON still satisfies the sweep schema
  expected by the runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.model_v4.sweeps._augment import (
    KNOWN_TARGETS,
    augment_sweep,
)
from validation.model_v4.sweeps.classify import Regime


@pytest.fixture
def tmp_sweep(tmp_path: Path) -> Path:
    """Write a tiny synthetic sweep with 3 fp32 + 2 fp16 entries."""
    payload = {
        "op": "matmul",
        "purpose": "calibration",
        "generator_seed": 42,
        "generated_against_hardware": ["i7_12700k"],
        "shapes": [
            {"shape": [256, 256, 256], "dtype": "fp32",
             "regime_per_hw": {"i7_12700k": "l2_bound"}},
            {"shape": [1024, 1024, 1024], "dtype": "fp32",
             "regime_per_hw": {"i7_12700k": "l2_bound"}},
            {"shape": [4096, 4096, 4096], "dtype": "fp32",
             "regime_per_hw": {"i7_12700k": "dram_bound"}},
            {"shape": [1024, 1024, 1024], "dtype": "fp16",
             "regime_per_hw": {"h100_sxm5_80gb": "launch_bound"}},
            {"shape": [4096, 4096, 4096], "dtype": "fp16",
             "regime_per_hw": {"h100_sxm5_80gb": "dram_bound"}},
        ],
    }
    sweep_path = tmp_path / "matmul_calibration.json"
    sweep_path.write_text(json.dumps(payload, indent=2))
    return sweep_path


def test_augmenter_classifies_only_matching_dtype_entries(tmp_sweep):
    """Augmenting with an fp16 target should only classify the 2 fp16
    entries; the 3 fp32 entries get untouched."""
    target = KNOWN_TARGETS["jetson_orin_nano_8gb"]   # dtype=fp16
    classified, skipped = augment_sweep(tmp_sweep, target)
    assert classified == 2
    assert skipped == 3


def test_augmenter_preserves_existing_target_regimes(tmp_sweep):
    """The original i7_12700k regime labels must stay byte-identical
    after augmentation -- the augmenter never overwrites *other*
    targets' regimes."""
    before = json.loads(tmp_sweep.read_text())
    augment_sweep(tmp_sweep, KNOWN_TARGETS["jetson_orin_nano_8gb"])
    after = json.loads(tmp_sweep.read_text())

    for b, a in zip(before["shapes"], after["shapes"]):
        # All original keys preserved with original values
        for k, v in b["regime_per_hw"].items():
            assert a["regime_per_hw"][k] == v


def test_augmenter_is_idempotent(tmp_sweep):
    """Running the same augmentation twice must produce identical output."""
    target = KNOWN_TARGETS["jetson_orin_agx_64gb"]
    augment_sweep(tmp_sweep, target)
    snapshot = json.loads(tmp_sweep.read_text())
    augment_sweep(tmp_sweep, target)
    after = json.loads(tmp_sweep.read_text())
    assert snapshot == after


def test_augmenter_records_target_in_metadata(tmp_sweep):
    """``augmented_with_hardware`` should list every target ever applied,
    sorted, no duplicates."""
    augment_sweep(tmp_sweep, KNOWN_TARGETS["jetson_orin_nano_8gb"])
    augment_sweep(tmp_sweep, KNOWN_TARGETS["h100_sxm5_80gb"])
    augment_sweep(tmp_sweep, KNOWN_TARGETS["jetson_orin_nano_8gb"])  # dup

    payload = json.loads(tmp_sweep.read_text())
    assert payload["augmented_with_hardware"] == [
        "h100_sxm5_80gb", "jetson_orin_nano_8gb",
    ]


def test_augmenter_emits_only_valid_regime_values(tmp_sweep):
    """All regime values written by the augmenter must be valid Regime
    enum strings (the sweep schema test in test_sweeps.py enforces this
    on the committed files; here we double-check on synthetic data)."""
    valid = {r.value for r in Regime}
    augment_sweep(tmp_sweep, KNOWN_TARGETS["jetson_orin_nano_8gb"])
    payload = json.loads(tmp_sweep.read_text())
    for entry in payload["shapes"]:
        for r in entry["regime_per_hw"].values():
            assert r in valid


def test_known_targets_includes_jetson_keys():
    """The augmenter registry must list at least the i7 + jetson +
    h100 entries that the runner's SWEEP_HW_TO_MAPPER expects."""
    expected = {"i7_12700k", "h100_sxm5_80gb",
                "jetson_orin_nano_8gb", "jetson_orin_agx_64gb",
                "jetson_orin_nx_16gb"}
    assert expected.issubset(set(KNOWN_TARGETS))


def test_each_known_target_has_a_mapper_factory():
    """Every entry must produce a usable HardwareResourceModel without
    raising. Catches signature drift in the gpu/cpu factories."""
    for key, target in KNOWN_TARGETS.items():
        hw = target.hw
        assert hw.peak_bandwidth > 0, f"{key} has no peak_bandwidth"
        assert hw.compute_units > 0, f"{key} has no compute_units"
