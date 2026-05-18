"""Parity test: YAML-loaded TI TDA4VL matches the hand-coded factory.

**Closes TI TDA4 family** (DSP batch #223 SKU 6 / batch 2 complete).
Retires the 225-LOC hand-coded body of ti_tda4vl_resource_model() in
favor of a thin loader wrapper.

**5 documented drifts** collapse to identity post-cleanup -- identical
shape to TDA4AL #227 and TDA4VH #228:

  - ``compute_units``: hand=16 (abstracted) -> 4 (real C7x num_units)
  - ``threads_per_unit``: hand=250 (BUG) -> 4 (wave_quantization)
  - ``warp_size``: hand=1 (BUG) -> 4 (=compute_units)
  - ``energy_per_flop_fp32``: hand=3.6 pJ (28nm bug) -> 2.43 pJ
    (16nm per TI's official docs)
  - ``precision_profiles``: hand has INT8/FP32 only -> full
    INT8/INT16/FP16/FP32

Same shape as the other TI TDA4 parity tests.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.automotive.ti_tda4vl import (
    ti_tda4vl_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "ti_tda4vl"
LEGACY_NAME = "TI-TDA4VL-C7x-MMAv2"


@pytest.fixture(scope="module")
def hand_coded():
    return ti_tda4vl_resource_model()


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
# Compute fabrics: half scale (4 C7x cores + 1 MMAv2 at half capacity)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_c7x_fabric_has_4_cores(hand_coded, yaml_loaded):
    """Half of TDA4VM's 8 cores."""
    yl, hc = yaml_loaded.compute_fabrics[0], hand_coded.compute_fabrics[0]
    assert yl.fabric_type == hc.fabric_type == "c7x_dsp"
    assert yl.num_units == hc.num_units == 4


def test_mma_fabric_at_half_capacity(hand_coded, yaml_loaded):
    """1 MMAv2 unit at 4000 ops/cycle (half of TDA4AL's 8000)."""
    yl, hc = yaml_loaded.compute_fabrics[1], hand_coded.compute_fabrics[1]
    assert yl.fabric_type == hc.fabric_type == "mma_v2"
    assert yl.num_units == hc.num_units == 1
    assert yl.ops_per_unit_per_clock[Precision.INT8] == \
        hc.ops_per_unit_per_clock[Precision.INT8] == 4000


# ---------------------------------------------------------------------------
# 5 documented drifts collapse to identity post-cleanup
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=16 (abstracted), yaml=4. Post: both = 4."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 4


def test_threads_per_unit_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=250 (BUG), yaml=4. Post: both = 4."""
    assert hand_coded.threads_per_unit == yaml_loaded.threads_per_unit == 4


def test_warp_size_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre: hand=1 (BUG), yaml=4. Post: both = 4."""
    assert hand_coded.warp_size == yaml_loaded.warp_size == 4


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
# Precision peaks: half of TDA4VM
# ---------------------------------------------------------------------------

def test_int8_peak_is_4_tops(hand_coded, yaml_loaded):
    """Half of TDA4VM's 8 TOPS."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(4e12, rel=0.01)


def test_fp32_peak_is_40_gflops(hand_coded, yaml_loaded):
    """Half of TDA4VM's 80 GFLOPS."""
    hc_peak = hand_coded.precision_profiles[Precision.FP32].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.FP32].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(40e9, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory: cost-optimized (4 GiB vs TDA4VM's 8 GiB)
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """60 GB/s LPDDR4x (same as TDA4VM; cost optimization preserves BW)."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(60e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 48 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """8 MiB MSMC (same as TDA4VM)."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 8 * 1024 * 1024


def test_main_memory_is_4_gb(hand_coded, yaml_loaded):
    """**Cost optimization**: 4 GiB (half of TDA4VM's 8 GiB)."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 4 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes (match TDA4VM/AL: 0.70 / 4 / 4)
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.70


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 4


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Thermal: 7W + 12W (lowest envelope in TDA4 family; both passive)
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    assert set(yaml_loaded.thermal_operating_points) == \
        set(hand_coded.thermal_operating_points) == {"7W", "12W"}


def test_default_thermal_profile_is_7w(hand_coded, yaml_loaded):
    """7W for entry-level single-camera ADAS."""
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "7W"


def test_7w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["7W"]
    hc = hand_coded.thermal_operating_points["7W"]
    assert yl.tdp_watts == hc.tdp_watts == 7.0


def test_12w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["12W"]
    hc = hand_coded.thermal_operating_points["12W"]
    assert yl.tdp_watts == hc.tdp_watts == 12.0
