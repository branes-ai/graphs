"""Parity test: YAML-loaded TI TDA4AL matches the hand-coded factory.

Cleanup PR of DSP batch #223 SKU 4. Retires the 188-LOC hand-coded
body of ti_tda4al_resource_model() in favor of a thin loader wrapper.

**5 documented drifts pre-cleanup** that collapse to identity
post-cleanup (more than TDA4VM's 3 because the TDA4AL hand-coded
factory had additional bugs):

  - ``compute_units``: hand=32 (abstracted via num_dsp_units),
    yaml=8 (real C7x num_units; loader picks fabric[0])
  - ``threads_per_unit``: hand=250 (BUG: an ops_per_unit_per_clock
    value got pasted into threads_per_unit; should be 4 like TDA4VM),
    yaml=4 (wave_quantization, loader's standard convention)
  - ``warp_size``: hand=1 (BUG: probably forgotten; TDA4VM has 16),
    yaml=8 (=compute_units, loader's standard convention)
  - ``energy_per_flop_fp32``: hand=3.6 pJ (28nm bug per TI's "16nm
    FinFET" official docs), yaml=2.43 pJ (16nm correct)
  - ``precision_profiles``: hand has only INT8/FP32 (omits INT16
    and FP16 despite both fabrics supporting them), yaml has full
    INT8/INT16/FP16/FP32

Same pattern as TDA4VM (#226); the additional bugs are TDA4AL-
specific. All drifts collapse to YAML-corrected values post-cleanup.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.automotive.ti_tda4al import (
    ti_tda4al_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "ti_tda4al"
LEGACY_NAME = "TI-TDA4AL-C7x-MMAv2"


@pytest.fixture(scope="module")
def hand_coded():
    return ti_tda4al_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={
            DSPFabricKind.VLIW_SCALAR: "c7x_dsp",
            DSPFabricKind.TENSOR_MATRIX: "mma_v2",
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
# Compute fabrics (C7x VLIW + MMAv2 tensor)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_c7x_fabric_first(hand_coded, yaml_loaded):
    assert yaml_loaded.compute_fabrics[0].fabric_type == "c7x_dsp"
    assert hand_coded.compute_fabrics[0].fabric_type == "c7x_dsp"
    assert yaml_loaded.compute_fabrics[0].num_units == \
        hand_coded.compute_fabrics[0].num_units == 8


def test_mma_v2_fabric_second(hand_coded, yaml_loaded):
    """**TDA4AL delta vs TDA4VM**: fabric_type is 'mma_v2' (vs 'mma_v1')."""
    assert yaml_loaded.compute_fabrics[1].fabric_type == "mma_v2"
    assert hand_coded.compute_fabrics[1].fabric_type == "mma_v2"
    assert yaml_loaded.compute_fabrics[1].num_units == \
        hand_coded.compute_fabrics[1].num_units == 1


# ---------------------------------------------------------------------------
# Chip-level surfaces: 5 documented drifts collapse to identity
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand=32 (abstracted), yaml=8 (real C7x num_units).
    Post-cleanup: both = 8."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 8


def test_threads_per_unit_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand=250 (BUG: ops/unit/clock value pasted into
    threads_per_unit), yaml=4 (wave_quantization). Post-cleanup: both = 4."""
    assert hand_coded.threads_per_unit == yaml_loaded.threads_per_unit == 4


def test_warp_size_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand=1 (BUG: probably forgotten), yaml=8 (=compute_units).
    Post-cleanup: both = 8."""
    assert hand_coded.warp_size == yaml_loaded.warp_size == 8


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand=3.6 pJ (28nm bug; same as TDA4VM), yaml=2.43 pJ.
    Post-cleanup: both = 2.43 pJ (16nm per TI's official docs)."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32 == \
        pytest.approx(2.43e-12, rel=0.01)


def test_precision_profiles_collapse_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand has only INT8/FP32 (omits INT16 and FP16
    despite both fabrics supporting them). Post-cleanup: both have full
    INT8/INT16/FP16/FP32."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    for p in (Precision.INT8, Precision.INT16, Precision.FP16, Precision.FP32):
        assert p in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Precision peaks
# ---------------------------------------------------------------------------

def test_int8_peak_matches_marketed(hand_coded, yaml_loaded):
    """8 TOPS INT8 (TI-marketed; MMAv2 path)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(8e12, rel=0.01)


def test_fp32_peak_matches_marketed(hand_coded, yaml_loaded):
    """80 GFLOPS FP32 (TI-marketed; C7x path)."""
    hc_peak = hand_coded.precision_profiles[Precision.FP32].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.FP32].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(80e9, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory subsystem
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """60 GB/s LPDDR4x (same as TDA4VM; measured datasheet)."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(60e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 48 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 8 * 1024 * 1024


def test_main_memory_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.70


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 4


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Thermal operating points (TDA4AL: 10W + 18W, vs TDA4VM's 10W + 20W)
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    """**TDA4AL delta**: 18W max profile (vs TDA4VM's 20W)."""
    assert set(yaml_loaded.thermal_operating_points) == \
        set(hand_coded.thermal_operating_points) == {"10W", "18W"}


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "10W"


def test_10w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["10W"]
    hc = hand_coded.thermal_operating_points["10W"]
    assert yl.tdp_watts == hc.tdp_watts == 10.0


def test_18w_profile_tdp_matches(hand_coded, yaml_loaded):
    """TDA4AL max profile is 18W (vs TDA4VM's 20W)."""
    yl = yaml_loaded.thermal_operating_points["18W"]
    hc = hand_coded.thermal_operating_points["18W"]
    assert yl.tdp_watts == hc.tdp_watts == 18.0
