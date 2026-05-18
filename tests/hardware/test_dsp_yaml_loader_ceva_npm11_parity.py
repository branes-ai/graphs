"""Parity test: YAML-loaded CEVA NeuPro-M NPM11 matches the hand-coded factory.

Closes batch 1 of the DSP follow-up sprint (graphs#223). Both IP-core
SKUs (Synopsys EV7x + CEVA NPM11) now have YAML-backed thin loader
wrappers; the 254-LOC hand-coded body of NPM11 is retired here.

**Tensor-first fabric ordering**: unlike Synopsys EV7x which had 3
documented drifts from VPU-first ordering, NPM11's YAML places the
tensor fabric first so the loader's pick-first convention maps it
to chip-level surfaces. Result: only ONE documented drift (INT4 in
precision_profiles -- YAML is more complete than the hand-coded
factory which omitted INT4 from precision_profiles but kept it on
the tensor fabric).

After cleanup both fixtures resolve through the loader; the drift
collapses to identity (precision_profiles include INT4 on both sides).

Same shape as ``tests/hardware/test_dsp_yaml_loader_synopsys_ev7x_parity.py``.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.ip_cores.ceva_neupro_npm11 import (
    ceva_neupro_npm11_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "ceva_neupro_m_npm11"
LEGACY_NAME = "CEVA-NeuPro-M-NPM11"


@pytest.fixture(scope="module")
def hand_coded():
    return ceva_neupro_npm11_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={
            DSPFabricKind.TENSOR_MATRIX: "neupro_tensor",
            DSPFabricKind.VECTOR_SIMD: "neupro_vector",
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
# Compute fabrics (tensor first; vector second)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    """NPM11 ships 2 fabrics: tensor (INT) + vector (FP16)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_tensor_fabric_is_fabric_zero(hand_coded, yaml_loaded):
    """**Tensor-first ordering**: tensor is fabric[0] in both, so
    the loader's pick-first convention maps it to chip-level surfaces."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "neupro_tensor"
    assert hand_coded.compute_fabrics[0].fabric_type == "neupro_tensor"


def test_vector_fabric_is_fabric_one(hand_coded, yaml_loaded):
    """Vector is fabric[1] in both."""
    assert yaml_loaded.compute_fabrics[1].fabric_type == "neupro_vector"
    assert hand_coded.compute_fabrics[1].fabric_type == "neupro_vector"


def test_tensor_num_units_matches(hand_coded, yaml_loaded):
    """64 tensor units."""
    assert yaml_loaded.compute_fabrics[0].num_units == \
        hand_coded.compute_fabrics[0].num_units == 64


def test_vector_num_units_matches(hand_coded, yaml_loaded):
    """64 vector units."""
    assert yaml_loaded.compute_fabrics[1].num_units == \
        hand_coded.compute_fabrics[1].num_units == 64


def test_tensor_int8_ops_match(hand_coded, yaml_loaded):
    """312 INT8 MACs/cycle/unit -> 20 TOPS @ 1 GHz."""
    assert yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock[Precision.INT8] == \
        hand_coded.compute_fabrics[0].ops_per_unit_per_clock[Precision.INT8] == 312


def test_tensor_int4_ops_match(hand_coded, yaml_loaded):
    """624 INT4 MACs/cycle/unit -> 40 TOPS (2x INT8)."""
    assert yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock[Precision.INT4] == \
        hand_coded.compute_fabrics[0].ops_per_unit_per_clock[Precision.INT4] == 624


def test_vector_fp16_ops_match(hand_coded, yaml_loaded):
    """156 FP16 MACs/cycle/unit -> 10 TFLOPS @ 1 GHz."""
    assert yaml_loaded.compute_fabrics[1].ops_per_unit_per_clock[Precision.FP16] == \
        hand_coded.compute_fabrics[1].ops_per_unit_per_clock[Precision.FP16] == 156


# ---------------------------------------------------------------------------
# Chip-level surfaces (NO DRIFT thanks to tensor-first ordering)
# ---------------------------------------------------------------------------

def test_compute_units_matches(hand_coded, yaml_loaded):
    """64 -- both pick tensor's num_units. NO DRIFT (vs EV7x where
    VPU-first ordering caused hand=128 / yaml=4 drift)."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 64


def test_energy_per_flop_fp32_matches(hand_coded, yaml_loaded):
    """2.295 pJ -- both pick tensor_core baseline. NO DRIFT."""
    assert yaml_loaded.energy_per_flop_fp32 == hand_coded.energy_per_flop_fp32 == \
        pytest.approx(2.295e-12, rel=0.01)


# ---------------------------------------------------------------------------
# Precision profiles (post-cleanup: both include INT4)
# ---------------------------------------------------------------------------

def test_precision_profile_keys_match_post_cleanup(hand_coded, yaml_loaded):
    """Post-cleanup both resolve through the loader, so the precision
    set is identical (both include INT4 from the YAML's theoretical_performance).
    Pre-cleanup the hand-coded factory omitted INT4 from precision_profiles
    despite the tensor fabric supporting it."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    assert Precision.INT4 in yaml_loaded.precision_profiles


def test_int8_peak_matches_marketed(hand_coded, yaml_loaded):
    """20 TOPS INT8 (CEVA-marketed)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(20e12, rel=0.01)


def test_int4_peak_matches_2x_int8(hand_coded, yaml_loaded):
    """40 TOPS INT4 (NeuPro-M's INT4 capability)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT4].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT4].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(40e12, rel=0.01)


def test_fp16_peak_matches_marketed(hand_coded, yaml_loaded):
    """10 TFLOPS FP16 (CEVA-marketed)."""
    hc_peak = hand_coded.precision_profiles[Precision.FP16].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.FP16].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(10e12, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory subsystem
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """50 GB/s typical-integration LPDDR5 bandwidth (mobile)."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(50e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    """64 KiB per unit."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 64 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """2 MiB shared L2."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 2 * 1024 * 1024


def test_main_memory_matches(hand_coded, yaml_loaded):
    """8 GiB typical mobile pairing."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.70


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 8


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Thermal operating points
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    assert set(yaml_loaded.thermal_operating_points) == set(hand_coded.thermal_operating_points)


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "2W"


def test_thermal_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["2W"]
    hc = hand_coded.thermal_operating_points["2W"]
    assert yl.tdp_watts == hc.tdp_watts == 2.0
