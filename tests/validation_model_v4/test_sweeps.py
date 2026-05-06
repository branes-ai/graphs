"""Schema and consistency tests for the v4 sweep JSON files.

Locks in:
- Each JSON file is loadable, has the expected top-level keys.
- Calibration and validation sets are disjoint (Principle 4).
- Every shape's recorded regime label still matches what the classifier
  produces today (drift in classify.py would silently break sweeps
  otherwise).
- Every recorded entry actually classifies in some regime (no AMBIGUOUS,
  no UNSUPPORTED -- the generator already filters these, this is the
  belt-and-braces check).
- Coverage sanity floors: at minimum N entries per (hardware, regime)
  for the regimes the v4-2 plan calls out as load-bearing. Thin cells
  are tolerated with explicit minimums; if a future change drops below
  even the minimum, the test fails loudly.
"""

import json
from pathlib import Path

import pytest

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import (
    create_h100_sxm5_80gb_mapper,
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
    create_jetson_orin_nx_16gb_mapper,
)
from validation.model_v4.sweeps.classify import Regime, classify_regime


SWEEP_DIR = Path(__file__).resolve().parents[2] / "validation" / "model_v4" / "sweeps"

SWEEP_FILES = [
    "matmul_calibration.json",
    "matmul_validation.json",
    "linear_calibration.json",
    "linear_validation.json",
]


@pytest.fixture(scope="module")
def hw():
    """All hardware keys the augmenter knows about. Must stay in sync with
    KNOWN_TARGETS in validation/model_v4/sweeps/_augment.py -- if a new
    target is added there, add the matching mapper here."""
    return {
        "i7_12700k": create_i7_12700k_mapper().resource_model,
        "h100_sxm5_80gb": create_h100_sxm5_80gb_mapper().resource_model,
        "jetson_orin_nano_8gb": create_jetson_orin_nano_8gb_mapper().resource_model,
        "jetson_orin_agx_64gb": create_jetson_orin_agx_64gb_mapper().resource_model,
        "jetson_orin_nx_16gb": create_jetson_orin_nx_16gb_mapper().resource_model,
    }


@pytest.mark.parametrize("filename", SWEEP_FILES)
def test_sweep_file_loads_with_expected_schema(filename):
    payload = json.loads((SWEEP_DIR / filename).read_text())
    assert payload["op"] in {"matmul", "linear"}
    assert payload["purpose"] in {"calibration", "validation"}
    assert isinstance(payload["generator_seed"], int)
    assert isinstance(payload["generated_against_hardware"], list)
    assert payload["generated_against_hardware"], "must list at least one HW"
    assert isinstance(payload["shapes"], list)
    assert payload["shapes"], "must contain at least one shape"
    for entry in payload["shapes"]:
        assert "shape" in entry and isinstance(entry["shape"], list)
        assert "dtype" in entry and isinstance(entry["dtype"], str)
        assert "regime_per_hw" in entry and isinstance(entry["regime_per_hw"], dict)
        # regime labels must be valid Regime enum values
        for r in entry["regime_per_hw"].values():
            assert r in {x.value for x in Regime}


@pytest.mark.parametrize("op", ["matmul", "linear"])
def test_calibration_and_validation_are_disjoint(op):
    """Principle 4: same shapes never appear in both sets."""
    cal = json.loads((SWEEP_DIR / f"{op}_calibration.json").read_text())["shapes"]
    val = json.loads((SWEEP_DIR / f"{op}_validation.json").read_text())["shapes"]
    cal_keys = {(tuple(e["shape"]), e["dtype"]) for e in cal}
    val_keys = {(tuple(e["shape"]), e["dtype"]) for e in val}
    overlap = cal_keys & val_keys
    assert not overlap, f"calibration / validation overlap for {op}: {overlap}"


@pytest.mark.parametrize("filename", SWEEP_FILES)
def test_recorded_regime_labels_still_match_classifier(filename, hw):
    """The JSON's recorded regime labels were correct at generation time;
    if classify.py changes regime math, the labels go stale silently
    without this check. CI catches the drift before sweeps mislead the
    harness."""
    payload = json.loads((SWEEP_DIR / filename).read_text())
    op = payload["op"]
    for entry in payload["shapes"]:
        for hw_key, recorded in entry["regime_per_hw"].items():
            actual = classify_regime(op, tuple(entry["shape"]), entry["dtype"], hw[hw_key])
            assert actual.value == recorded, (
                f"{filename}: shape={entry['shape']} dtype={entry['dtype']} "
                f"hw={hw_key} recorded={recorded} actual={actual.value}"
            )


@pytest.mark.parametrize("filename", SWEEP_FILES)
def test_no_ambiguous_or_unsupported_in_committed_sweeps(filename):
    """Generator filters AMBIGUOUS / UNSUPPORTED out (Principle 1 + the
    UNSUPPORTED design decision); confirm none slipped through."""
    payload = json.loads((SWEEP_DIR / filename).read_text())
    for entry in payload["shapes"]:
        for r in entry["regime_per_hw"].values():
            assert r not in {"ambiguous", "unsupported"}, (
                f"{filename}: shape={entry['shape']} has bad regime {r}"
            )


# ---------------------------------------------------------------------------
# Coverage sanity floors per (hardware, regime).
# These are MINIMUMS, not targets. Some cells are structurally thin
# (e.g., i7-12700K ALU_BOUND = 0 because per-core L1 is too small for any
# matrix that fits to clear the 25us launch threshold). The floors below
# document what we actually have and protect against silent regressions.
# Re-run _generate.py and update these numbers when the classifier or
# candidate pools change deliberately.
# ---------------------------------------------------------------------------


# (op, purpose, hardware, regime) -> minimum entries expected
COVERAGE_FLOORS = {
    # Matmul calibration
    ("matmul", "calibration", "i7_12700k", "dram_bound"):       8,
    ("matmul", "calibration", "i7_12700k", "l2_bound"):         6,
    ("matmul", "calibration", "i7_12700k", "launch_bound"):     4,
    ("matmul", "calibration", "h100_sxm5_80gb", "alu_bound"):   2,
    ("matmul", "calibration", "h100_sxm5_80gb", "dram_bound"):  8,
    ("matmul", "calibration", "h100_sxm5_80gb", "l2_bound"):    6,
    ("matmul", "calibration", "h100_sxm5_80gb", "launch_bound"):4,
    # Matmul validation
    ("matmul", "validation",  "i7_12700k", "dram_bound"):      30,
    ("matmul", "validation",  "i7_12700k", "l2_bound"):        20,
    ("matmul", "validation",  "i7_12700k", "launch_bound"):    10,
    ("matmul", "validation",  "h100_sxm5_80gb", "alu_bound"):   2,
    ("matmul", "validation",  "h100_sxm5_80gb", "dram_bound"): 30,
    ("matmul", "validation",  "h100_sxm5_80gb", "l2_bound"):    6,
    ("matmul", "validation",  "h100_sxm5_80gb", "launch_bound"):10,
    # Linear calibration
    ("linear", "calibration", "i7_12700k", "dram_bound"):       8,
    ("linear", "calibration", "i7_12700k", "l2_bound"):         6,
    ("linear", "calibration", "i7_12700k", "launch_bound"):     4,
    ("linear", "calibration", "h100_sxm5_80gb", "alu_bound"):   3,
    ("linear", "calibration", "h100_sxm5_80gb", "dram_bound"):  8,
    ("linear", "calibration", "h100_sxm5_80gb", "l2_bound"):    3,
    ("linear", "calibration", "h100_sxm5_80gb", "launch_bound"):4,
    # Linear validation
    ("linear", "validation",  "i7_12700k", "dram_bound"):      30,
    ("linear", "validation",  "i7_12700k", "l2_bound"):        20,
    ("linear", "validation",  "i7_12700k", "launch_bound"):    10,
    ("linear", "validation",  "h100_sxm5_80gb", "alu_bound"):   3,
    ("linear", "validation",  "h100_sxm5_80gb", "dram_bound"): 30,
    ("linear", "validation",  "h100_sxm5_80gb", "l2_bound"):    3,
    ("linear", "validation",  "h100_sxm5_80gb", "launch_bound"):10,
}


@pytest.mark.parametrize("op,purpose,hw_key,regime,floor",
                         [(*k, v) for k, v in COVERAGE_FLOORS.items()])
def test_coverage_floor(op, purpose, hw_key, regime, floor):
    payload = json.loads((SWEEP_DIR / f"{op}_{purpose}.json").read_text())
    actual = sum(
        1 for e in payload["shapes"]
        if e["regime_per_hw"].get(hw_key) == regime
    )
    assert actual >= floor, (
        f"{op}/{purpose}: {hw_key} {regime} has {actual} entries; "
        f"floor is {floor}. Re-run _generate.py if this is intentional."
    )
