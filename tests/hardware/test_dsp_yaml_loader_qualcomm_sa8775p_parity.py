"""Parity test: YAML-loaded Qualcomm SA8775P matches the hand-coded factory.

Cleanup PR of DSP batch #223 SKU 7 (first 3-profile DVFS DSP; first
5nm DSP). Retires the 369-LOC hand-coded body of
qualcomm_sa8775p_resource_model() in favor of a thin loader wrapper
with a **BOM overlay** (SA8775P is the first DSP SKU with commercial
BOM; the YAML loader doesn't produce BOM so the wrapper attaches it
post-load).

**6 documented drifts pre-cleanup** that collapse to identity
post-cleanup:

  - ``compute_units``: hand=32 (abstracted via HVX+HMX combined),
    yaml=2 (HMX's num_units; loader picks fabric[0])
  - ``threads_per_unit``: hand=128 (HMX accelerator threads),
    yaml=4 (wave_quantization, loader's standard)
  - ``warp_size``: hand=32 (sensible SIMD width estimate),
    yaml=2 (=compute_units, loader's standard)
  - ``energy_per_flop_fp32``: hand=1.35 pJ (HVX baseline, 5nm
    simd_packed), yaml=1.27 pJ (HMX baseline, 5nm tensor_core,
    fabric[0])
  - ``precision_profiles``: hand has INT8/INT4 only (sparse),
    yaml has full INT8/INT16/FP16/INT4
  - ``bom_cost_profile``: hand-coded factory sets it; YAML loader
    doesn't. **The wrapper attaches BOM post-load** to preserve
    parity. Both fixtures have BOM post-cleanup.

Same shape as the other DSP parity tests; BOM overlay is the new
pattern.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.automotive.qualcomm_sa8775p import (
    qualcomm_sa8775p_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "qualcomm_sa8775p"
LEGACY_NAME = "SA8775P-Snapdragon-Ride"


@pytest.fixture(scope="module")
def hand_coded():
    return qualcomm_sa8775p_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    """Bare YAML load -- does NOT include BOM overlay."""
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={
            DSPFabricKind.TENSOR_MATRIX: "hmx_tensor",
            DSPFabricKind.VECTOR_SIMD: "hvx_vector",
        },
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == LEGACY_NAME


def test_hardware_type_is_dsp(hand_coded, yaml_loaded):
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.DSP


# ---------------------------------------------------------------------------
# Compute fabrics: HMX tensor (primary) + HVX vector
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_hmx_fabric_first(hand_coded, yaml_loaded):
    """HMX is fabric[0] -- primary INT8 path (30 of 32 TOPS)."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "hmx_tensor"
    assert hand_coded.compute_fabrics[0].fabric_type == "hmx_tensor"
    # 2 HMX units; 7500 INT8 ops/cycle/unit
    yl, hc = yaml_loaded.compute_fabrics[0], hand_coded.compute_fabrics[0]
    assert yl.num_units == hc.num_units == 2
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 7500


def test_hvx_fabric_second(hand_coded, yaml_loaded):
    """HVX is fabric[1] -- 1024-bit SIMD for activations."""
    assert yaml_loaded.compute_fabrics[1].fabric_type == "hvx_vector"
    assert hand_coded.compute_fabrics[1].fabric_type == "hvx_vector"
    # 4 HVX units; 256 INT8 ops/cycle/unit
    yl, hc = yaml_loaded.compute_fabrics[1], hand_coded.compute_fabrics[1]
    assert yl.num_units == hc.num_units == 4
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 256


# ---------------------------------------------------------------------------
# 6 documented drifts collapse to identity post-cleanup
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=32 (abstracted HVX+HMX), yaml=2 (HMX num_units).
    Post: both = 2 (HMX is fabric[0], loader's pick-first convention)."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 2


def test_threads_per_unit_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=128 (HMX accelerator threads), yaml=4 (wave_quantization).
    Post: both = 4."""
    assert hand_coded.threads_per_unit == yaml_loaded.threads_per_unit == 4


def test_warp_size_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=32 (SIMD width estimate), yaml=2 (=compute_units).
    Post: both = 2."""
    assert hand_coded.warp_size == yaml_loaded.warp_size == 2


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=1.35 pJ (HVX baseline), yaml=1.27 pJ (HMX tensor_core baseline).
    Post: both = 1.27 pJ (HMX is fabric[0])."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32 == \
        pytest.approx(1.27e-12, rel=0.01)


def test_precision_profiles_collapse_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand has INT8/INT4 only. Post: both have full
    INT8/INT16/FP16/INT4."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    for p in (Precision.INT4, Precision.INT8, Precision.INT16, Precision.FP16):
        assert p in yaml_loaded.precision_profiles


def test_bom_cost_profile_preserved_via_wrapper_overlay(hand_coded):
    """**SA8775P is the first DSP SKU with commercial BOM.** The YAML
    loader doesn't produce BOM; the wrapper attaches it post-load.
    Both fixtures have BOM post-cleanup."""
    assert hand_coded.bom_cost_profile is not None
    assert hand_coded.bom_cost_profile.silicon_die_cost == 180.0
    assert hand_coded.bom_cost_profile.process_node == "5nm"


# ---------------------------------------------------------------------------
# Precision peaks: 32 TOPS INT8 / 64 TOPS INT4 (Qualcomm-marketed)
# ---------------------------------------------------------------------------

def test_int8_peak_is_32_tops(hand_coded, yaml_loaded):
    """32 TOPS INT8 (Qualcomm-marketed; HMX dominates)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(32e12, rel=0.01)


def test_int4_peak_is_64_tops(hand_coded, yaml_loaded):
    """64 TOPS INT4 (2x INT8 throughput)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT4].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT4].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(64e12, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory: 5nm LPDDR5 automotive
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """90 GB/s LPDDR5 automotive."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(90e9)


def test_l1_l2_main_memory_match(hand_coded, yaml_loaded):
    """128 KB L1/unit + 8 MB L2 + 16 GB LPDDR5."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 128 * 1024
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 8 * 1024 * 1024
    assert yaml_loaded.main_memory == hand_coded.main_memory == 16 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.5


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    """16 (automotive multi-task ADAS + cockpit)."""
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 16


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Thermal: 3 profiles (first 3-profile DVFS DSP)
# ---------------------------------------------------------------------------

def test_three_thermal_profiles(hand_coded, yaml_loaded):
    """**First 3-profile DVFS DSP**: 20W + 30W + 45W."""
    assert set(yaml_loaded.thermal_operating_points) == \
        set(hand_coded.thermal_operating_points) == {"20W", "30W", "45W"}


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    """30W is most common automotive deployment."""
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "30W"


def test_all_three_profiles_have_correct_tdp(hand_coded, yaml_loaded):
    for name, expected_tdp in (("20W", 20.0), ("30W", 30.0), ("45W", 45.0)):
        yl = yaml_loaded.thermal_operating_points[name]
        hc = hand_coded.thermal_operating_points[name]
        assert yl.tdp_watts == hc.tdp_watts == expected_tdp
