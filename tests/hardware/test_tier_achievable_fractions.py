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


def test_i7_12700k_l2_calibration():
    """Per-core L2 tier follow-up: i7 ``L2.achievable_fraction =
    0.22`` derived from V5-2b vector_add at N=262K (the canonical
    L2-binding shape). Calibration accounts for the gap between
    the aggregate L2 BW peak (1.5 TB/s) and the
    actually-realized BW under PyTorch's multi-threaded kernel
    dispatch (326 GB/s observed, after dispatch correction).
    See docs/calibration/i7-12700k-l2-calibration-analysis.md."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.tier_achievable_fractions.get("L2") == 0.22
    l2 = next(t for t in hw.memory_hierarchy if t.name == "L2")
    assert l2.achievable_fraction == 0.22


def test_i7_12700k_l2_per_unit_bandwidth_matches_alder_lake_spec():
    """Per-core L2 BW is conservatively set to 150 GB/s/effective-unit
    (Alder Lake P-core peak ~288 GB/s at 4.5 GHz, weighted across
    P/E hybrid). Aggregate = 150 * 10 = 1.5 TB/s."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.l2_bandwidth_per_unit_bps == 150e9
    l2 = next(t for t in hw.memory_hierarchy if t.name == "L2")
    assert l2.peak_bandwidth_bps == 150e9 * hw.compute_units


def test_i7_12700k_l3_calibration():
    """V5-5 follow-up: i7-12700K L3 achievable_fraction = 0.84
    derived by 2-point regression on dispatch-corrected L3-bound
    rows of i7_12700k_vector_add.csv (N=65K and N=1M; the N=262K
    point is rejected as showing non-physical 1.71x L3 peak from
    per-core L2 hits the i7 mapper doesn't currently model). Mean
    fraction across the two clean points = 0.836, rounded to 0.84.
    See docs/calibration/i7-12700k-l3-calibration-analysis.md."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.tier_achievable_fractions.get("L3") == 0.84
    l3 = next(t for t in hw.memory_hierarchy if t.name == "L3")
    assert l3.achievable_fraction == 0.84
    assert l3.effective_bandwidth_bps == pytest.approx(200e9 * 0.84)


def test_i7_12700k_per_op_calibration():
    """V5-3b flag-flip prerequisite: i7 has per-op DRAM and L2
    overrides for matmul / linear. The single-fraction-per-tier
    model couldn't capture that vector_add (zero-reuse) and matmul
    (structured-reuse) achieve different effective BW at the same
    tier; per-op overrides resolve this."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.tier_achievable_fractions_by_op == {
        "matmul": {"L2": 0.10, "DRAM": 0.85},
        "linear": {"L2": 0.10, "DRAM": 0.85},
    }


def test_per_op_calibration_falls_back_to_per_tier():
    """vector_add isn't in the per-op overrides; its tier fractions
    must come from the per-tier ``tier_achievable_fractions``
    (V5-5 baseline values)."""
    hw = create_i7_12700k_mapper().resource_model
    # No vector_add entry in tier_achievable_fractions_by_op
    assert "vector_add" not in hw.tier_achievable_fractions_by_op
    # So vector_add picks up the per-tier values
    assert hw.tier_achievable_fractions["L2"] == 0.22
    assert hw.tier_achievable_fractions["DRAM"] == 0.47


def test_i7_12700k_l1_calibration():
    """V5 follow-up: i7 ``L1.achievable_fraction = 0.020`` from the
    2-point regression on N=256 and N=1024 vector_add baseline rows
    (BW = 69 GB/s after dispatch correction; aggregate L1 peak is
    3500 GB/s, so 69/3500 = 0.020). Originally judged a no-op (PR
    #105) because dispatch floor dominates for matmul L1-binding
    shapes -- which is true -- but vector_add at medium N (4K-16K)
    has WS in the 48-192 KB range where the math part exceeds the
    2 us op-aware dispatch floor and L1 fraction drives the
    prediction. Setting 0.020 lands V4 vector_add N=16K within the
    30% LAUNCH band."""
    hw = create_i7_12700k_mapper().resource_model
    assert hw.tier_achievable_fractions.get("L1") == 0.02
    l1 = next(t for t in hw.memory_hierarchy if t.name == "L1")
    assert l1.achievable_fraction == 0.02
    # Verify the effective BW lands at the regression value (69 GB/s)
    assert l1.effective_bandwidth_bps == pytest.approx(3500e9 * 0.02)


def test_i7_12700k_large_mapper_inherits_dram_calibration():
    """create_i7_12700k_large_mapper() represents the same DDR5
    subsystem as the tiny-model variant. The DRAM achievable_fraction
    must match (0.47), or once tier-aware memory is enabled the large
    variant would over-predict throughput by 2x relative to its
    sibling on the same hardware."""
    from graphs.hardware.mappers.cpu import create_i7_12700k_large_mapper

    hw = create_i7_12700k_large_mapper().resource_model
    assert hw.tier_achievable_fractions.get("DRAM") == 0.47
    # V5-5 follow-up also propagates the L3 calibration so the two
    # sibling mappers stay in lock-step on the same hardware.
    assert hw.tier_achievable_fractions.get("L3") == 0.84


def test_jetson_orin_nano_emits_three_tier_hierarchy():
    """V5 follow-up: Orin Nano now populates l1_bandwidth_per_unit_bps
    and l2_bandwidth_bps so memory_hierarchy emits L1 + L2 + DRAM.
    Without these, the V5-3b eligibility predicate's >= 2 tier gate
    declines the tier-aware path even with the flag on."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    tier_names = [t.name for t in hw.memory_hierarchy]
    assert tier_names == ["L1", "L2", "DRAM"], (
        f"Orin Nano memory_hierarchy must be L1 -> L2 -> DRAM (3 tiers); "
        f"got {tier_names}. Did the on-chip BW peaks get unset?"
    )


def test_jetson_orin_nano_l1_bandwidth_matches_ampere_spec():
    """Ampere SM 8.6 L1 cache + shared memory throughput is 128 B/cycle.
    At Orin Nano's 650 MHz sustained baseline (the same clock used by
    the compute fabrics in the mapper), per-SM L1 BW = 83.2 GB/s.
    The mapper rounds this to 83 GB/s.

    Per the V5-1 contract, ``l1_bandwidth_per_unit_bps`` is per-SM and
    the memory_hierarchy aggregates by multiplying by ``compute_units``
    (= 8 SMs), giving 664 GB/s aggregate L1."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    assert hw.l1_bandwidth_per_unit_bps == 83e9
    l1 = next(t for t in hw.memory_hierarchy if t.name == "L1")
    assert l1.peak_bandwidth_bps == 83e9 * hw.compute_units
    assert l1.peak_bandwidth_bps == pytest.approx(664e9)


def test_jetson_orin_nano_l2_bandwidth_preserves_tier_ordering():
    """L2 BW = 204 GB/s (2x DRAM peak, conservative Ampere estimate;
    not directly published by NVIDIA for Orin Nano). The exact value
    will tighten with calibration; the test that matters is that the
    tier ordering holds: L1 > L2 > DRAM. Without this ordering the
    V5-3b tier picker would walk in the wrong direction."""
    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    assert hw.l2_bandwidth_bps == 204e9
    tiers = {t.name: t for t in hw.memory_hierarchy}
    assert (
        tiers["L1"].peak_bandwidth_bps
        > tiers["L2"].peak_bandwidth_bps
        > tiers["DRAM"].peak_bandwidth_bps
    )


def test_jetson_orin_nano_v5_3b_path_now_activates():
    """The V5-3b eligibility predicate gates on memory_hierarchy
    having >= 2 tiers. Pre-followup Orin Nano emitted DRAM-only and
    the flag was a no-op; post-followup the path activates and the
    DRAM achievable_fraction = 0.55 from V5-5 (#103) reaches
    DRAM-bound shapes through the analyzer."""
    from graphs.estimation.reuse_models import REUSE_MODELS
    from graphs.estimation.tier_picker import pick_binding_tier

    hw = create_jetson_orin_nano_8gb_mapper().resource_model
    assert len(hw.memory_hierarchy) >= 2

    # Large vector_add binds at DRAM and uses the calibrated
    # effective_bandwidth_bps (102 * 0.55 = 56.1 GB/s).
    result = pick_binding_tier(
        REUSE_MODELS["vector_add"], (16777216,), "fp16", hw.memory_hierarchy
    )
    assert result is not None
    assert result.binding_tier.name == "DRAM"
    assert result.binding_tier.effective_bandwidth_bps == pytest.approx(102e9 * 0.55)


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
