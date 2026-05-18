"""Parity test: YAML-loaded Qualcomm QCS6490 matches the hand-coded factory.

**Closes DSP batch #223 sprint** (10/10 SKUs YAML-backed once cleanup
lands). Retires the 365-LOC hand-coded body in favor of a thin loader
wrapper with a BOM overlay (same pattern as SA8775P #230 and
QRB5165 #231).

**6 documented drifts** collapse to identity post-cleanup (same shape
as SA8775P):

  - ``compute_units``: hand=16 (HVX num_units), yaml=1 (HTA num_units;
    loader picks fabric[0])
  - ``threads_per_unit``: hand=64 (DSP vector threads), yaml=2
    (wave_quantization)
  - ``warp_size``: hand=32 (SIMD estimate), yaml=1 (=compute_units)
  - ``energy_per_flop_fp32``: hand=1.485 pJ (HVX 6nm simd_packed),
    yaml=1.40 pJ (HTA 6nm tensor_core; HTA is fabric[0])
  - ``precision_profiles``: hand has INT8/INT4 only, yaml has full
    INT8/INT16/FP16/INT4
  - ``bom_cost_profile``: hand has it; YAML loader doesn't; wrapper
    attaches BOM overlay post-load
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.edge.qualcomm_qcs6490 import (
    qualcomm_qcs6490_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "qualcomm_qcs6490"
LEGACY_NAME = "Qualcomm-QCS6490-Hexagon-V79"


@pytest.fixture(scope="module")
def hand_coded():
    return qualcomm_qcs6490_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    """Bare YAML load (no BOM overlay)."""
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={
            DSPFabricKind.TENSOR_MATRIX: "hta_tensor",
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
# Compute fabrics: HTA tensor + HVX vector (16 HVX units, more than QRB5165)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_hta_fabric_first(hand_coded, yaml_loaded):
    """HTA is fabric[0] -- primary INT8 path (~7.5 of 12 TOPS)."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "hta_tensor"
    assert hand_coded.compute_fabrics[0].fabric_type == "hta_tensor"
    yl, hc = yaml_loaded.compute_fabrics[0], hand_coded.compute_fabrics[0]
    assert yl.num_units == hc.num_units == 1
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 5000


def test_hvx_has_16_units(hand_coded, yaml_loaded):
    """16 HVX units -- more vector parallelism than QRB5165's 4."""
    assert yaml_loaded.compute_fabrics[1].fabric_type == "hvx_vector"
    assert hand_coded.compute_fabrics[1].fabric_type == "hvx_vector"
    yl, hc = yaml_loaded.compute_fabrics[1], hand_coded.compute_fabrics[1]
    assert yl.num_units == hc.num_units == 16
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 200


# ---------------------------------------------------------------------------
# 6 documented drifts collapse to identity post-cleanup
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=16 (HVX num_units), yaml=1 (HTA num_units; loader picks
    fabric[0]). Post: both = 1."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 1


def test_threads_per_unit_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=64 (DSP vector threads), yaml=2 (wave_quantization).
    Post: both = 2."""
    assert hand_coded.threads_per_unit == yaml_loaded.threads_per_unit == 2


def test_warp_size_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=32 (SIMD estimate), yaml=1 (=compute_units). Post: both = 1."""
    assert hand_coded.warp_size == yaml_loaded.warp_size == 1


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=1.485 pJ (HVX baseline), yaml=1.40 pJ (HTA tensor_core).
    Post: both = 1.40 pJ."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32 == \
        pytest.approx(1.40e-12, rel=0.01)


def test_precision_profiles_collapse_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand has INT8/INT4 only. Post: both have INT8/INT16/FP16/INT4."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    for p in (Precision.INT4, Precision.INT8, Precision.INT16, Precision.FP16):
        assert p in yaml_loaded.precision_profiles


def test_bom_cost_profile_preserved_via_wrapper_overlay(hand_coded):
    """Wrapper attaches BOM overlay post-load (same pattern as
    SA8775P/QRB5165). QCS6490 is entry-level so BOM is smallest of
    the Qualcomm trio: $45 silicon (vs $55 QRB5165, $180 SA8775P)."""
    assert hand_coded.bom_cost_profile is not None
    assert hand_coded.bom_cost_profile.silicon_die_cost == 45.0
    assert hand_coded.bom_cost_profile.process_node == "6nm"


# ---------------------------------------------------------------------------
# Precision peaks: 12 TOPS INT8 / 24 TOPS INT4
# ---------------------------------------------------------------------------

def test_int8_peak_is_12_tops(hand_coded, yaml_loaded):
    """12 TOPS INT8 (Qualcomm-marketed)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(12e12, rel=0.01)


def test_int4_peak_is_24_tops(hand_coded, yaml_loaded):
    """24 TOPS INT4 (2x INT8)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT4].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT4].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(24e12, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory: 6nm LPDDR4X
# ---------------------------------------------------------------------------

def test_peak_bandwidth_is_40_gbps(hand_coded, yaml_loaded):
    """LPDDR4X 40 GB/s (cheaper than QRB5165's LPDDR5 44 GB/s)."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(40e9)


def test_l1_l2_main_memory_match(hand_coded, yaml_loaded):
    """64 KB L1/unit + 3 MB L2 + 8 GB LPDDR4X."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 64 * 1024
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 3 * 1024 * 1024
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.5


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 8


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    """wave_quantization=2 (smaller than SA8775P/QRB5165's 4)."""
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 2


# ---------------------------------------------------------------------------
# Thermal: 3 profiles (5W/10W/15W)
# ---------------------------------------------------------------------------

def test_three_thermal_profiles(hand_coded, yaml_loaded):
    """3-profile DVFS: 5W battery + 10W standard + 15W max."""
    assert set(yaml_loaded.thermal_operating_points) == \
        set(hand_coded.thermal_operating_points) == {"5W", "10W", "15W"}


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    """10W is the standard edge AI deployment."""
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "10W"


def test_all_three_profiles_have_correct_tdp(hand_coded, yaml_loaded):
    for name, expected_tdp in (("5W", 5.0), ("10W", 10.0), ("15W", 15.0)):
        yl = yaml_loaded.thermal_operating_points[name]
        hc = hand_coded.thermal_operating_points[name]
        assert yl.tdp_watts == hc.tdp_watts == expected_tdp
