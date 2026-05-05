"""Regression tests for issue #53.

KPU-T64's precision_profiles previously listed only INT8/BF16/INT4. Asking
the analyzer for FP16 or FP32 silently fell back to the default (INT8),
which made `mcp analyze -p fp16` and `-p int8` produce byte-identical
output. This file locks in:

- FP16 and FP32 entries exist in the profile dict.
- get_peak_ops returns distinct values for each supported precision.
- bytes_per_element is correct (4 for fp32, 2 for fp16).
"""

import pytest

from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.resource_model import Precision


@pytest.fixture(scope="module")
def kpu_t64():
    return kpu_t64_resource_model()


@pytest.mark.parametrize(
    "precision",
    [Precision.INT4, Precision.INT8, Precision.BF16, Precision.FP16, Precision.FP32],
)
def test_kpu_t64_lists_precision(kpu_t64, precision):
    assert precision in kpu_t64.precision_profiles


@pytest.mark.parametrize(
    "precision,expected_tops",
    [
        (Precision.INT4, 132.8),
        (Precision.INT8, 66.4),
        (Precision.BF16, 33.2),
        (Precision.FP16, 33.2),
        (Precision.FP32, 1.5),
    ],
)
def test_kpu_t64_peak_ops(kpu_t64, precision, expected_tops):
    """get_peak_ops must return the per-precision value, not the INT8 fallback."""
    actual_tops = kpu_t64.get_peak_ops(precision) / 1e12
    assert actual_tops == pytest.approx(expected_tops, rel=1e-6)


@pytest.mark.parametrize(
    "precision,expected_bpe",
    [
        (Precision.INT4, 0.5),
        (Precision.INT8, 1),
        (Precision.BF16, 2),
        (Precision.FP16, 2),
        (Precision.FP32, 4),
    ],
)
def test_kpu_t64_bytes_per_element(kpu_t64, precision, expected_bpe):
    profile = kpu_t64.get_precision_profile(precision)
    assert profile.bytes_per_element == expected_bpe


def test_kpu_t64_fp16_and_int8_are_distinct(kpu_t64):
    """The original bug: -p fp16 and -p int8 produced identical output."""
    fp16_ops = kpu_t64.get_peak_ops(Precision.FP16)
    int8_ops = kpu_t64.get_peak_ops(Precision.INT8)
    assert fp16_ops != int8_ops, "FP16 lookup must not silently fall back to INT8"


def test_get_peak_ops_raises_on_unsupported_precision(kpu_t64):
    """Companion fix: missing precision must raise, not silently fall back.

    Without this, every other accelerator with sparse precision_profiles
    would hide the same class of bug as #53.
    """
    # KPU-T64 has no FP64 entry; this must raise rather than return INT8's value.
    with pytest.raises(ValueError, match="does not support fp64"):
        kpu_t64.get_peak_ops(Precision.FP64)


def test_get_precision_profile_raises_on_unsupported_precision(kpu_t64):
    with pytest.raises(ValueError, match="supported precisions are"):
        kpu_t64.get_precision_profile(Precision.FP64)
