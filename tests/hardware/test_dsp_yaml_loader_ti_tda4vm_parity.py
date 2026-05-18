"""Parity test: YAML-loaded TI TDA4VM matches the hand-coded factory.

Cleanup PR of DSP batch #223 SKU 3 (first SoC-integrated DSP).
The 368-LOC hand-coded body of ti_tda4vm_resource_model() is retired
here in favor of a thin loader wrapper.

**3 documented drifts pre-cleanup** that collapse to identity
post-cleanup (both fixtures now resolve through the loader):

  - ``compute_units``: hand=32 (abstracted via num_dsp_units=
    MMA's 8000 ops/cycle / 250 ops/cycle/unit), yaml=8 (C7x's real
    num_units from fabric[0]). Loader convention picks fabric[0]'s
    actual count; cleaner than the abstraction.
  - ``energy_per_flop_fp32``: hand=3.6 pJ (28nm simd_packed,
    pre-existing factory bug), yaml=2.43 pJ (16nm simd_packed,
    YAML-corrected). TI's official docs say TDA4VM is 16nm FinFET;
    the hand-coded factory's process_node_nm=28 was wrong (the same
    factory's docstring contradicts itself: "Process: 16nm FinFET").
  - ``precision_profiles``: hand-coded omits FP16 from chip-level
    profiles despite the C7x fabric supporting it; yaml includes it.

Same shape as ``test_dsp_yaml_loader_synopsys_ev7x_parity.py`` and
``test_dsp_yaml_loader_ceva_npm11_parity.py``.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.automotive.ti_tda4vm import (
    ti_tda4vm_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "ti_tda4vm"
LEGACY_NAME = "TI-TDA4VM-C7x-DSP"


@pytest.fixture(scope="module")
def hand_coded():
    return ti_tda4vm_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={
            DSPFabricKind.VLIW_SCALAR: "c7x_dsp",
            DSPFabricKind.TENSOR_MATRIX: "mma_v1",
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
# Compute fabrics (C7x VLIW + MMA tensor, in that order)
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(hand_coded, yaml_loaded):
    """TDA4VM ships 2 fabrics: C7x DSP + MMAv1 accelerator."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_c7x_fabric_is_fabric_zero(hand_coded, yaml_loaded):
    assert yaml_loaded.compute_fabrics[0].fabric_type == "c7x_dsp"
    assert hand_coded.compute_fabrics[0].fabric_type == "c7x_dsp"


def test_mma_fabric_is_fabric_one(hand_coded, yaml_loaded):
    assert yaml_loaded.compute_fabrics[1].fabric_type == "mma_v1"
    assert hand_coded.compute_fabrics[1].fabric_type == "mma_v1"


def test_c7x_num_units_matches(hand_coded, yaml_loaded):
    """8 C7x DSP cores."""
    assert yaml_loaded.compute_fabrics[0].num_units == \
        hand_coded.compute_fabrics[0].num_units == 8


def test_mma_num_units_matches(hand_coded, yaml_loaded):
    """1 MMA unit (single accelerator)."""
    assert yaml_loaded.compute_fabrics[1].num_units == \
        hand_coded.compute_fabrics[1].num_units == 1


def test_c7x_fp32_ops_match(hand_coded, yaml_loaded):
    """10 FP32 ops/cycle/unit -> 80 GFLOPS @ 1 GHz across 8 cores."""
    assert yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock[Precision.FP32] == \
        hand_coded.compute_fabrics[0].ops_per_unit_per_clock[Precision.FP32] == 10


def test_mma_int8_ops_match(hand_coded, yaml_loaded):
    """8000 INT8 MACs/cycle/unit -> 8 TOPS @ 1 GHz."""
    assert yaml_loaded.compute_fabrics[1].ops_per_unit_per_clock[Precision.INT8] == \
        hand_coded.compute_fabrics[1].ops_per_unit_per_clock[Precision.INT8] == 8000


# ---------------------------------------------------------------------------
# Chip-level surfaces -- the 3 documented drifts collapse post-cleanup
# ---------------------------------------------------------------------------

def test_compute_units_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand=32 (abstracted), yaml=8 (C7x's num_units).
    Post-cleanup: both = 8 (cleaner than the abstraction)."""
    assert hand_coded.compute_units == yaml_loaded.compute_units == 8


def test_energy_per_flop_fp32_collapses_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand=3.6 pJ (28nm bug), yaml=2.43 pJ (16nm correct).
    Post-cleanup: both = 2.43 pJ (16nm correction per TI's official docs)."""
    assert hand_coded.energy_per_flop_fp32 == yaml_loaded.energy_per_flop_fp32 == \
        pytest.approx(2.43e-12, rel=0.01)


def test_precision_profiles_collapse_post_cleanup(hand_coded, yaml_loaded):
    """Pre-cleanup: hand omits FP16 from chip-level profiles despite C7x
    supporting it. Post-cleanup: both include FP16 (YAML is more complete)."""
    assert set(hand_coded.precision_profiles) == set(yaml_loaded.precision_profiles)
    assert Precision.FP16 in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Precision peaks
# ---------------------------------------------------------------------------

def test_int8_peak_matches_marketed(hand_coded, yaml_loaded):
    """8 TOPS INT8 (TI-marketed; MMA path)."""
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
    """60 GB/s LPDDR4x dual-channel @ 3733 MT/s (measured datasheet)."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(60e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    """48 KiB L1D per C7x core (32 KB cache + 16 KB SRAM)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 48 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """8 MiB MSMC SRAM."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 8 * 1024 * 1024


def test_main_memory_matches(hand_coded, yaml_loaded):
    """Up to 8 GiB LPDDR4x."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


# ---------------------------------------------------------------------------
# Scheduler attributes
# ---------------------------------------------------------------------------

def test_min_occupancy_matches(hand_coded, yaml_loaded):
    """Automotive deterministic scheduling: 0.70."""
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy == 0.70


def test_max_concurrent_kernels_matches(hand_coded, yaml_loaded):
    """Limited to 4 for automotive determinism."""
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 4


def test_wave_quantization_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 4


# ---------------------------------------------------------------------------
# Thermal operating points (first multi-profile DSP)
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    """**First DSP SKU with 2 thermal profiles**: 10W front-camera +
    20W full-ADAS-system."""
    assert set(yaml_loaded.thermal_operating_points) == set(hand_coded.thermal_operating_points)
    assert set(yaml_loaded.thermal_operating_points) == {"10W", "20W"}


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    """10W front-camera is the most common automotive deployment."""
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "10W"


def test_10w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["10W"]
    hc = hand_coded.thermal_operating_points["10W"]
    assert yl.tdp_watts == hc.tdp_watts == 10.0


def test_20w_profile_tdp_matches(hand_coded, yaml_loaded):
    yl = yaml_loaded.thermal_operating_points["20W"]
    hc = hand_coded.thermal_operating_points["20W"]
    assert yl.tdp_watts == hc.tdp_watts == 20.0
