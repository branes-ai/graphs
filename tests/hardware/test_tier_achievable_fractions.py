"""V5-5 calibration tests for HardwareResourceModel.tier_achievable_fractions.

Locks in:
- The new field defaults to an empty dict; absence of an override means
  the tier's effective_bandwidth_bps == peak_bandwidth_bps (i.e., the
  V5-1 / V5-3b-shipped behavior).
- Per-tier overrides flow through memory_hierarchy: setting
  ``{"DRAM": 0.47}`` on the resource model produces a DRAM tier whose
  effective_bandwidth_bps is 47% of peak.
- The calibrated mappers (i7-12700K, Jetson Orin Nano 8GB) have the
  V5-5 DRAM achievable_fraction values from the V5-2b vector_add
  baselines (i7 = 0.47, Jetson = 0.55).
"""

import pytest

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_jetson_orin_nano_8gb_mapper
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
)


# ---------------------------------------------------------------------------
# Field default + plumbing
# ---------------------------------------------------------------------------


def _bare_resource_model(**overrides) -> HardwareResourceModel:
    """Minimal HardwareResourceModel for testing the field plumbing.

    Only the required fields + DRAM-bandwidth + main_memory are set;
    omitting on-chip BW peaks gives a DRAM-only memory_hierarchy
    (matches the V5-1 docstring on un-calibrated mappers)."""
    base = dict(
        name="test",
        hardware_type=HardwareType.CPU,
        compute_units=1,
        threads_per_unit=1,
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1e12,
                tensor_core_supported=False,
            )
        },
        default_precision=Precision.FP32,
        peak_bandwidth=100e9,  # 100 GB/s
        l1_cache_per_unit=32 * 1024,
        l2_cache_total=1 * 1024 * 1024,
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=1e-12,
        energy_per_byte=1e-11,
    )
    base.update(overrides)
    return HardwareResourceModel(**base)


def test_default_tier_achievable_fractions_is_empty_dict():
    """Field must default to an empty dict so existing un-calibrated
    mappers keep their pre-V5-5 behavior (every tier at peak)."""
    hw = _bare_resource_model()
    assert hw.tier_achievable_fractions == {}


def test_dram_tier_uses_peak_bandwidth_when_no_override():
    """Without an override, MemoryTier.achievable_fraction defaults to
    1.0 (V5-1 contract) and effective_bandwidth_bps == peak."""
    hw = _bare_resource_model()
    dram = next(t for t in hw.memory_hierarchy if t.name == "DRAM")
    assert dram.achievable_fraction == 1.0
    assert dram.effective_bandwidth_bps == dram.peak_bandwidth_bps == 100e9


def test_dram_override_flows_through_to_effective_bandwidth():
    """Setting {"DRAM": 0.47} on the resource model produces a DRAM
    MemoryTier with effective_bandwidth_bps == 47% of peak."""
    hw = _bare_resource_model(tier_achievable_fractions={"DRAM": 0.47})
    dram = next(t for t in hw.memory_hierarchy if t.name == "DRAM")
    assert dram.achievable_fraction == 0.47
    assert dram.effective_bandwidth_bps == pytest.approx(100e9 * 0.47)


def test_unset_tiers_default_to_one_when_other_tier_is_overridden():
    """Setting only DRAM must not perturb other tiers (they default to
    1.0 even though the overrides dict is non-empty)."""
    hw = _bare_resource_model(
        l1_bandwidth_per_unit_bps=200e9,
        tier_achievable_fractions={"DRAM": 0.47},
    )
    tiers = {t.name: t for t in hw.memory_hierarchy}
    assert tiers["L1"].achievable_fraction == 1.0
    assert tiers["DRAM"].achievable_fraction == 0.47


def test_invalid_fraction_raises_at_construction():
    """MemoryTier validates achievable_fraction is in [0, 1] in
    __post_init__. An override out of range must surface as a
    construction-time ValueError, not a silent garbage value at
    analysis time."""
    hw = _bare_resource_model(tier_achievable_fractions={"DRAM": 1.5})
    with pytest.raises(ValueError, match="achievable_fraction"):
        _ = hw.memory_hierarchy


# ---------------------------------------------------------------------------
# Calibrated mappers: i7-12700K + Jetson Orin Nano 8GB
# ---------------------------------------------------------------------------


def test_i7_12700k_dram_calibration():
    """i7-12700K DRAM achievable_fraction = 0.47 derived from the
    V5-2b vector_add baseline (median of 35.2 / 34.7 / 36.7 GB/s on
    16M / 67M / 268M elements; peak 75 GB/s)."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.tier_achievable_fractions.get("DRAM") == 0.47
    dram = next(t for t in hw.memory_hierarchy if t.name == "DRAM")
    assert dram.achievable_fraction == 0.47
    # 75 GB/s * 0.47 = 35.25 GB/s -- matches the measured plateau
    assert dram.effective_bandwidth_bps == pytest.approx(75e9 * 0.47)


def test_i7_12700k_non_dram_tiers_uncalibrated():
    """V5-5 only ships DRAM calibration; L1 and L3 stay at 1.0
    pending the matmul-anchored follow-up."""
    hw = create_i7_12700k_mapper().resource_model
    for t in hw.memory_hierarchy:
        if t.name != "DRAM":
            assert t.achievable_fraction == 1.0


def test_i7_12700k_large_mapper_inherits_dram_calibration():
    """create_i7_12700k_large_mapper() represents the same DDR5
    subsystem as the tiny-model variant. The DRAM achievable_fraction
    must match (0.47), or once tier-aware memory is enabled the large
    variant would over-predict throughput by 2x relative to its
    sibling on the same hardware."""
    from graphs.hardware.mappers.cpu import create_i7_12700k_large_mapper

    hw = create_i7_12700k_large_mapper().resource_model
    assert hw.tier_achievable_fractions.get("DRAM") == 0.47


def test_jetson_orin_nano_dram_calibration():
    """Jetson Orin Nano (Super) DRAM achievable_fraction = 0.55
    derived from the V5-2b vector_add baseline at the 15W thermal
    profile (median of 61.6 / 56.4 / 56.6 GB/s on 16M / 67M / 268M
    fp16 elements; peak 102 GB/s LPDDR5-6400)."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    assert hw.tier_achievable_fractions.get("DRAM") == 0.55
    dram = next(t for t in hw.memory_hierarchy if t.name == "DRAM")
    assert dram.achievable_fraction == 0.55
    assert dram.effective_bandwidth_bps == pytest.approx(102e9 * 0.55)
