"""Parity test: YAML-loaded TI TDA4VH matches the hand-coded factory.

Cleanup PR of DSP batch #223 SKU 5 (third TI TDA4 family member;
4x TDA4VM scale; first LPDDR5 DSP). Retires the 193-LOC hand-coded
body of ti_tda4vh_resource_model() in favor of a thin loader wrapper.

**5 documented drifts pre-cleanup** that collapse to identity
post-cleanup (same pattern as TDA4AL #227):

  - ``compute_units``: hand=128 (abstracted via num_dsp_units=
    4xMMA*8000/250), yaml=32 (real C7x num_units; loader picks
    fabric[0])
  - ``threads_per_unit``: hand=250 (BUG: ops/clock value pasted),
    yaml=8 (=wave_quantization, loader's standard)
  - ``warp_size``: hand=1 (BUG: forgotten), yaml=32 (=compute_units)
  - ``energy_per_flop_fp32``: hand=3.6 pJ (28nm bug), yaml=2.43 pJ
    (16nm per TI's official docs)
  - ``precision_profiles``: hand has only INT8/FP32, yaml has full
    INT8/INT16/FP16/FP32

Same shape as TDA4AL parity test.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.automotive.ti_tda4vh import (
    ti_tda4vh_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "ti_tda4vh"
LEGACY_NAME = "TI-TDA4VH-4xC7x-4xMMAv2"


@pytest.fixture(scope="module")
def hand_coded():
    return ti_tda4vh_resource_model()


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
# Compute fabrics: 4x scale (32 C7x cores + 4 MMAv2 units)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_c7x_fabric_has_32_cores(hand_coded, yaml_loaded):
    """4x TDA4VM's 8 cores."""
    yl, hc = yaml_loaded.compute_fabrics[0], hand_coded.compute_fabrics[0]
    assert yl.fabric_type == hc.fabric_type == "c7x_dsp"
    assert yl.num_units == hc.num_units == 32


def test_mma_fabric_has_4_units(hand_coded, yaml_loaded):
    """4x MMAv2 accelerators (vs TDA4VM/AL's 1)."""
    yl, hc = yaml_loaded.compute_fabrics[1], hand_coded.compute_fabrics[1]
    assert yl.fabric_type == hc.fabric_type == "mma_v2"
    assert yl.num_units == hc.num_units == 4


# ---------------------------------------------------------------------------
# 5 documented drifts collapse to identity post-cleanup
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=128 (abstracted), yaml=32. Post: both = 32."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 32


def test_threads_per_unit_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=250 (BUG), yaml=8. Post: both = 8."""
    assert hand_coded.threads_per_unit == yaml_loaded.threads_per_unit == 8


def test_warp_size_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=1 (BUG), yaml=32 (=compute_units). Post: both = 32."""
    assert hand_coded.warp_size == yaml_loaded.warp_size == 32


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=3.6 pJ (28nm bug), yaml=2.43 pJ. Post: both = 2.43 pJ."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32 == \
        pytest.approx(2.43e-12, rel=0.01)


def test_precision_profiles_collapse_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand has INT8/FP32 only. Post: both have INT8/INT16/FP16/FP32."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    for p in (Precision.INT8, Precision.INT16, Precision.FP16, Precision.FP32):
        assert p in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Precision peaks: 4x TDA4VM
# ---------------------------------------------------------------------------

def test_int8_peak_is_32_tops(hand_coded, yaml_loaded):
    """4x TDA4VM (4x MMAv2 units): 32 TOPS INT8."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(32e12, rel=0.01)


def test_fp32_peak_is_320_gflops(hand_coded, yaml_loaded):
    """4x TDA4VM (4x C7x cluster): 320 GFLOPS FP32."""
    hc_peak = hand_coded.precision_profiles[Precision.FP32].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.FP32].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(320e9, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory subsystem: 2x TDA4VM (LPDDR5)
# ---------------------------------------------------------------------------

def test_peak_bandwidth_is_100_gbps(hand_coded, yaml_loaded):
    """LPDDR5 @ 6400 MT/s; first DSP SKU with LPDDR5."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(100e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    """48 KiB L1D per C7x core (same as TDA4VM/AL)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 48 * 1024


def test_l2_cache_is_16_mib(hand_coded, yaml_loaded):
    """16 MiB MSMC (2x TDA4VM's 8 MiB)."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 16 * 1024 * 1024


def test_main_memory_is_16_gb(hand_coded, yaml_loaded):
    """Up to 16 GiB LPDDR5 (2x TDA4VM)."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 16 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes: multi-accelerator deltas (already match in hand-coded)
# ---------------------------------------------------------------------------

def test_min_occupancy_is_0_60(hand_coded, yaml_loaded):
    """Lower than TDA4VM/AL's 0.70 -- multi-accelerator coordination cost."""
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.60


def test_max_concurrent_kernels_is_8(hand_coded, yaml_loaded):
    """4x accelerators allow 8 concurrent kernels."""
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 8


def test_wave_quantization_is_8(hand_coded, yaml_loaded):
    """4-MMAv2 grouping (vs TDA4VM/AL's 4)."""
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 8


# ---------------------------------------------------------------------------
# Thermal operating points: 20W + 35W (higher than TDA4VM/AL)
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    """20W multi-camera L2+ + 35W full L3-4 autonomy."""
    assert set(yaml_loaded.thermal_operating_points) == \
        set(hand_coded.thermal_operating_points) == {"20W", "35W"}


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    """20W is the default for multi-camera L2+ ADAS."""
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "20W"


def test_20w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["20W"]
    hc = hand_coded.thermal_operating_points["20W"]
    assert yl.tdp_watts == hc.tdp_watts == 20.0


def test_35w_profile_tdp_matches(hand_coded, yaml_loaded):
    """35W for full L3-4 autonomy."""
    yl = yaml_loaded.thermal_operating_points["35W"]
    hc = hand_coded.thermal_operating_points["35W"]
    assert yl.tdp_watts == hc.tdp_watts == 35.0
