"""Parity test: YAML-loaded Qualcomm QRB5165 matches the hand-coded factory.

Cleanup PR of DSP batch #223 SKU 8 (robotics platform, Hexagon 698 DSP).
Retires the 328-LOC hand-coded body in favor of a thin loader wrapper
with a BOM overlay (same pattern as SA8775P #230).

**4 documented drifts** collapse to identity post-cleanup (fewer than
SA8775P's 6 because the hand-coded QRB5165 precision_profiles set
already matches the YAML's):

  - ``compute_units``: hand=32 (abstracted HVX+HTA), yaml=1 (HTA's
    single unit; loader picks fabric[0])
  - ``warp_size``: hand=32 (SIMD estimate), yaml=1 (=compute_units)
  - ``energy_per_flop_fp32``: hand=1.62 pJ (HVX 7nm simd_packed),
    yaml=1.53 pJ (HTA 7nm tensor_core; HTA is fabric[0])
  - ``bom_cost_profile``: hand-coded has it; YAML loader doesn't;
    **wrapper attaches BOM overlay post-load**

(``threads_per_unit`` and ``precision_profiles`` both MATCH pre-cleanup
-- no drift on those.)
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.edge.qrb5165 import (
    qrb5165_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "qualcomm_qrb5165"
LEGACY_NAME = "Qualcomm-QRB5165-Hexagon698"


@pytest.fixture(scope="module")
def hand_coded():
    return qrb5165_resource_model()


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
# Compute fabrics: HTA tensor + HVX vector
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_hta_fabric_first(hand_coded, yaml_loaded):
    """HTA is fabric[0] -- primary INT8 path (~12 of 15 TOPS)."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "hta_tensor"
    assert hand_coded.compute_fabrics[0].fabric_type == "hta_tensor"
    yl, hc = yaml_loaded.compute_fabrics[0], hand_coded.compute_fabrics[0]
    assert yl.num_units == hc.num_units == 1
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 9024


def test_hvx_fabric_second(hand_coded, yaml_loaded):
    """HVX is fabric[1] -- 4x 1024-bit SIMD units."""
    assert yaml_loaded.compute_fabrics[1].fabric_type == "hvx_vector"
    assert hand_coded.compute_fabrics[1].fabric_type == "hvx_vector"
    yl, hc = yaml_loaded.compute_fabrics[1], hand_coded.compute_fabrics[1]
    assert yl.num_units == hc.num_units == 4
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 256


# ---------------------------------------------------------------------------
# 4 documented drifts collapse to identity post-cleanup
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=32 (abstracted), yaml=1 (HTA's single num_units).
    Post: both = 1."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 1


def test_warp_size_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=32 (SIMD estimate), yaml=1 (=compute_units). Post: both = 1."""
    assert hand_coded.warp_size == yaml_loaded.warp_size == 1


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=1.62 pJ (HVX baseline), yaml=1.53 pJ (HTA tensor_core baseline).
    Post: both = 1.53 pJ (HTA is fabric[0])."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32 == \
        pytest.approx(1.53e-12, rel=0.01)


def test_bom_cost_profile_preserved_via_wrapper_overlay(hand_coded):
    """**Wrapper attaches BOM overlay post-load** (same pattern as SA8775P).
    QRB5165 is consumer-grade so BOM is smaller ($55 silicon vs $180)."""
    assert hand_coded.bom_cost_profile is not None
    assert hand_coded.bom_cost_profile.silicon_die_cost == 55.0
    assert hand_coded.bom_cost_profile.process_node == "7nm"


# ---------------------------------------------------------------------------
# Fields that MATCH pre-cleanup (no drift; called out for clarity)
# ---------------------------------------------------------------------------

def test_threads_per_unit_matches_no_drift(hand_coded, yaml_loaded):
    """No drift: hand=4 already matches yaml=4 (wave_quantization)."""
    assert hand_coded.threads_per_unit == yaml_loaded.threads_per_unit == 4


def test_precision_profiles_match_no_drift(hand_coded, yaml_loaded):
    """No drift: both have INT8/INT16/FP16 (yaml doesn't add INT4 because
    QRB5165 YAML's theoretical_performance also omits int4)."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)


# ---------------------------------------------------------------------------
# Precision peaks: 15 TOPS INT8 marketed
# ---------------------------------------------------------------------------

def test_int8_peak_is_15_tops(hand_coded, yaml_loaded):
    """15 TOPS INT8 (Qualcomm-marketed; HTA dominates)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(15e12, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_peak_bandwidth_is_44_gbps(hand_coded, yaml_loaded):
    """LPDDR5 quad-channel: 4ch * 16-bit * 2750 MHz * 2 = 44 GB/s."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(44e9)


def test_l1_l2_main_memory_match(hand_coded, yaml_loaded):
    """128 KB L1/unit + 4 MB L2 + 16 GB LPDDR5."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 128 * 1024
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 4 * 1024 * 1024
    assert yaml_loaded.main_memory == hand_coded.main_memory == 16 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.60


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 8


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Thermal: single 7W passive
# ---------------------------------------------------------------------------

def test_single_thermal_profile(hand_coded, yaml_loaded):
    assert set(yaml_loaded.thermal_operating_points) == \
        set(hand_coded.thermal_operating_points) == {"7W"}


def test_7w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["7W"]
    hc = hand_coded.thermal_operating_points["7W"]
    assert yl.tdp_watts == hc.tdp_watts == 7.0


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "7W"
