"""V4-5 KPU consistency-only invariant tests.

The KPU is a research/in-development spatial-dataflow accelerator with
no commercially-available silicon to measure against. Per the v4 plan
(Principle 2), KPU validation is consistency-only -- we don't assert
per-shape predicted-vs-measured bands, but we DO assert the analyzer's
KPU predictions satisfy basic math/physical invariants.

Test coverage matrix:

  Invariant                              T64  T128  T256  T768
  -------------------------------------  ---  ----  ----  ----
  roofline_self_consistency              PASS PASS  PASS  PASS
  latency_non_decreasing_in_size         PASS PASS  PASS  PASS
  memory_time_scales_with_bytes          PASS PASS  PASS  PASS
  achieved_compute_below_peak            PASS PASS  PASS  PASS
  achieved_bw_below_peak                 PASS PASS  PASS  PASS
  avg_power_below_tdp                    XFAIL XFAIL XFAIL XFAIL  (#81)
  family_latency_non_increasing                 (cross-mapper test)

The avg_power_below_tdp invariant is marked xfail because the KPU
energy model uses the GPU IDLE_POWER_FRACTION (0.3) and uncalibrated
dynamic energy_per_flop coefficients -- the same pre-#71 problem that
afflicted CPU. Issue #81 tracks the KPU energy calibration; once that
lands the xfail comes off and this becomes a regular pass.
"""

from __future__ import annotations

import pytest

from graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
    create_kpu_t128_mapper,
    create_kpu_t256_mapper,
    create_kpu_t768_mapper,
)
from graphs.hardware.resource_model import Precision

from validation.model_v4.invariants.kpu import (
    DEFAULT_SHAPE_GRID,
    check_achieved_bw_below_peak,
    check_achieved_compute_below_peak,
    check_avg_power_below_tdp,
    check_family_latency_non_increasing,
    check_latency_non_decreasing_in_size,
    check_memory_time_scales_with_bytes,
    check_roofline_self_consistency,
    run_kpu_invariants,
)


# Order matters here -- many tests rely on this being increasing in
# capability (T64 -> T768).
_KPU_FAMILY = [
    ("T64",  create_kpu_t64_mapper),
    ("T128", create_kpu_t128_mapper),
    ("T256", create_kpu_t256_mapper),
    ("T768", create_kpu_t768_mapper),
]


# BF16 is the only precision supported by every KPU in the family
# (T128 and T768 don't list FP16/FP32 in their precision_profiles).
# Using BF16 universally lets all 4 mappers run the same test set.
PRECISION = Precision.BF16


# ---------------------------------------------------------------------------
# Per-mapper hard invariants (must always hold)
# ---------------------------------------------------------------------------


@pytest.fixture(params=_KPU_FAMILY, ids=[name for name, _ in _KPU_FAMILY])
def kpu_mapper(request):
    name, factory = request.param
    return name, factory().resource_model


def test_roofline_self_consistency(kpu_mapper):
    """``actual_latency = max(compute_time, memory_time) + overhead`` --
    if this fails the analyzer's own definition has drifted."""
    name, hw = kpu_mapper
    failures = check_roofline_self_consistency(hw, PRECISION)
    assert not failures, (
        f"KPU-{name} roofline self-consistency violations:\n  "
        + "\n  ".join(failures)
    )


def test_latency_non_decreasing_in_size(kpu_mapper):
    """Sweeping square matmul size: bigger workload cannot run faster."""
    name, hw = kpu_mapper
    failures = check_latency_non_decreasing_in_size(hw, PRECISION)
    assert not failures, (
        f"KPU-{name} latency non-monotonic in shape size:\n  "
        + "\n  ".join(failures)
    )


def test_memory_time_scales_with_bytes(kpu_mapper):
    """Memory time non-decreasing in working_set_bytes (cache effects
    mean it doesn't have to be linear, but it must be monotonic)."""
    name, hw = kpu_mapper
    failures = check_memory_time_scales_with_bytes(hw, PRECISION)
    assert not failures, (
        f"KPU-{name} memory_time non-monotonic in bytes:\n  "
        + "\n  ".join(failures)
    )


def test_achieved_compute_below_peak(kpu_mapper):
    """Hard physical bound: flops / compute_time <= peak FLOPS * slack."""
    name, hw = kpu_mapper
    failures = check_achieved_compute_below_peak(hw, PRECISION)
    assert not failures, (
        f"KPU-{name} achieved compute exceeds theoretical peak:\n  "
        + "\n  ".join(failures)
    )


def test_achieved_bw_below_peak(kpu_mapper):
    """Hard physical bound: bytes / memory_time <= peak BW * slack."""
    name, hw = kpu_mapper
    failures = check_achieved_bw_below_peak(hw, PRECISION)
    assert not failures, (
        f"KPU-{name} achieved memory BW exceeds theoretical peak:\n  "
        + "\n  ".join(failures)
    )


# ---------------------------------------------------------------------------
# Diagnostic: power-below-TDP is currently violated (xfail with reason)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=(
    "KPU energy model uses the GPU IDLE_POWER_FRACTION (0.3) and "
    "uncalibrated dynamic energy_per_flop coefficients. Predicted "
    "average power exceeds TDP by ~1.5x-3x for typical matmul shapes. "
    "Tracked in #81; mirrors the pre-#71 CPU situation. xfail "
    "strict=True so if the energy model gets fixed, the test flips "
    "to a regular pass and CI flags the xfail for removal."
))
def test_avg_power_below_tdp_known_violated(kpu_mapper):
    """Diagnostic: predicted avg power (energy/latency) <= TDP * 1.10.

    Currently KNOWN VIOLATED on every KPU mapper -- see xfail reason.
    The check itself is implemented and ready to flip to a hard test
    once the energy model is calibrated."""
    name, hw = kpu_mapper
    failures = check_avg_power_below_tdp(hw, PRECISION)
    assert not failures, (
        f"KPU-{name} predicted avg power exceeds TDP:\n  "
        + "\n  ".join(failures)
    )


# ---------------------------------------------------------------------------
# Cross-mapper invariant (single test, not parametrized per-mapper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_idx", range(len(DEFAULT_SHAPE_GRID)),
                         ids=[f"({s.M},{s.K},{s.N})" for s in DEFAULT_SHAPE_GRID])
def test_family_latency_non_increasing(shape_idx):
    """Across the KPU family ordered by capability (T64 -> T128 -> T256
    -> T768), predicted latency for a fixed shape must be non-increasing.
    More tiles + more BW cannot make a fixed workload slower."""
    family = [(f"KPU-{name}", factory().resource_model)
              for name, factory in _KPU_FAMILY]
    shape = DEFAULT_SHAPE_GRID[shape_idx]
    failures = check_family_latency_non_increasing(family, PRECISION, shape)
    assert not failures, (
        f"family latency monotonicity violated at shape "
        f"({shape.M},{shape.K},{shape.N}):\n  "
        + "\n  ".join(failures)
    )


# ---------------------------------------------------------------------------
# Battery: convenience pass-rate check via run_kpu_invariants
# ---------------------------------------------------------------------------


def test_run_kpu_invariants_returns_zero_hard_failures(kpu_mapper):
    """Sanity: ``run_kpu_invariants`` produces 0 *hard* failures for every
    KPU mapper. (Soft failures = avg_power_below_tdp; excluded via
    ``hard_failures()``.) This guards against a future invariant being
    added that the existing mappers can't satisfy."""
    name, hw = kpu_mapper
    report = run_kpu_invariants(hw, PRECISION, hardware_name=f"KPU-{name}")
    hard = report.hard_failures()
    assert not hard, (
        f"KPU-{name} battery has hard failures:\n  "
        + "\n  ".join(f"[{k}] {v}" for k, vs in hard.items() for v in vs)
    )
