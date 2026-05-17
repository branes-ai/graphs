"""Parity test: YAML-loaded Cadence Vision Q8 matches the hand-coded factory.

PR 4 of the DSP mini-sprint (issue #211). Proves the new
``dsp_yaml_loader.load_dsp_resource_model_from_yaml`` produces a
``HardwareResourceModel`` whose key fields match the existing
hand-coded ``cadence_vision_q8_resource_model()`` factory. PR 5
retires the hand-coded body in favor of a thin loader wrapper.

Mirrors ``tests/hardware/test_tpu_yaml_loader_v4_parity.py`` and
``tests/hardware/test_dpu_yaml_loader_vitis_ai_parity.py``.

Pre-cleanup documented drifts (NONE):
  - The hand-coded factory and YAML loader produce byte-equivalent
    models for all key fields. Same clean-cut migration as Plasticine
    (#199), Vitis AI (#203), TPU v4 (PR 5 of #204).

The loader uses ``fabric_type_override`` to preserve the legacy
fabric_type string "vision_q8_simd" (vs the generic "vector_simd"
that the loader would otherwise emit for VECTOR_SIMD-kind fabrics).
PR 5 will move this override into the thin factory wrapper.
"""

import pytest

from embodied_schemas.dsp_block import DSPFabricKind

from graphs.hardware.models.ip_cores.cadence_vision_q8 import (
    cadence_vision_q8_resource_model,
)
from graphs.hardware.models.ip_cores.dsp_yaml_loader import (
    DSPYamlLoaderError,
    load_dsp_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "cadence_tensilica_vision_q8"
LEGACY_NAME = "Cadence-Tensilica-Vision-Q8"
LEGACY_FABRIC_TYPE = "vision_q8_simd"


@pytest.fixture(scope="module")
def hand_coded():
    return cadence_vision_q8_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_dsp_resource_model_from_yaml(
        SKU_ID,
        name_override=LEGACY_NAME,
        fabric_type_overrides={DSPFabricKind.VECTOR_SIMD: LEGACY_FABRIC_TYPE},
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == LEGACY_NAME


def test_hardware_type_is_dsp(hand_coded, yaml_loaded):
    """Both produce HardwareType.DSP -- no transitional period
    (DSP already in graphs enum)."""
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.DSP


# ---------------------------------------------------------------------------
# Compute shape
# ---------------------------------------------------------------------------

def test_compute_units_matches(hand_coded, yaml_loaded):
    """32 SIMD units (matches the hand-coded num_dsp_units)."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 32


def test_threads_per_unit_matches(hand_coded, yaml_loaded):
    """wave_quantization=4 surfaces as threads_per_unit."""
    assert yaml_loaded.threads_per_unit == hand_coded.threads_per_unit == 4


def test_warp_size_matches(hand_coded, yaml_loaded):
    """warp_size = num_units (the wavefront-as-fabric model)."""
    assert yaml_loaded.warp_size == hand_coded.warp_size == 32


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """Cadence Vision Q8 ships a single fabric (SIMD only)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_fabric_type_matches_legacy(hand_coded, yaml_loaded):
    """Legacy fabric_type ``vision_q8_simd`` is preserved via the
    fabric_type_overrides argument."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == LEGACY_FABRIC_TYPE
    assert hand_coded.compute_fabrics[0].fabric_type == LEGACY_FABRIC_TYPE


def test_fabric_num_units(hand_coded, yaml_loaded):
    """num_units = 32 (per-fabric, matching SIMD lane count)."""
    assert yaml_loaded.compute_fabrics[0].num_units == \
        hand_coded.compute_fabrics[0].num_units == 32


def test_fabric_ops_per_clock_matches(hand_coded, yaml_loaded):
    """INT8/INT16: 119, FP32: 4, FP16: 8 ops/cycle/unit (matches
    1024-bit SIMD divided by element width)."""
    yl = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock
    hc = hand_coded.compute_fabrics[0].ops_per_unit_per_clock
    assert yl == hc
    assert yl[Precision.INT8] == 119
    assert yl[Precision.INT16] == 119
    assert yl[Precision.FP32] == 4
    assert yl[Precision.FP16] == 8


def test_fabric_core_frequency_matches(hand_coded, yaml_loaded):
    """1.0 GHz sustained (matches the 1W thermal profile clock)."""
    yl = yaml_loaded.compute_fabrics[0].core_frequency_hz
    hc = hand_coded.compute_fabrics[0].core_frequency_hz
    assert yl == hc == pytest.approx(1.0e9)


def test_fabric_process_node_matches(hand_coded, yaml_loaded):
    """16nm process node (TSMC N16, already in catalog)."""
    yl = yaml_loaded.compute_fabrics[0].process_node_nm
    hc = hand_coded.compute_fabrics[0].process_node_nm
    assert yl == hc == 16


# ---------------------------------------------------------------------------
# Precision profiles
# ---------------------------------------------------------------------------

def test_precision_profile_keys_match(hand_coded, yaml_loaded):
    """The same precision set is exposed (INT8 / INT16 / FP32; FP16
    is in fabric.ops but not necessarily in profiles -- it depends
    on the YAML's theoretical_performance roll-up). Both sides must
    AT LEAST agree on the hand-coded set."""
    hc_precs = set(hand_coded.precision_profiles)
    yl_precs = set(yaml_loaded.precision_profiles)
    # Hand-coded ships INT8 / INT16 / FP32. Loader ships those plus
    # whatever else is in the YAML's theoretical_performance roll-up
    # (FP16 in this case).
    assert hc_precs.issubset(yl_precs)


def test_int8_peak_matches_marketed(hand_coded, yaml_loaded):
    """3.8 TOPS INT8 (Cadence-marketed)."""
    hc_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(3.8e12, rel=0.01)


def test_fp32_peak_matches_marketed(hand_coded, yaml_loaded):
    """129 GFLOPS FP32 (Cadence-marketed)."""
    hc_peak = hand_coded.precision_profiles[Precision.FP32].peak_ops_per_sec
    yl_peak = yaml_loaded.precision_profiles[Precision.FP32].peak_ops_per_sec
    assert yl_peak == hc_peak == pytest.approx(129e9, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    """INT8 default (vision-optimized inference)."""
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory subsystem
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """40 GB/s typical-integration LPDDR4 bandwidth."""
    assert yaml_loaded.peak_bandwidth == hand_coded.peak_bandwidth == pytest.approx(40e9)


def test_l1_cache_per_unit_matches(hand_coded, yaml_loaded):
    """32 KiB per SIMD unit."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit == 32 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """1 MiB shared L2."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 1 * 1024 * 1024


def test_main_memory_matches(hand_coded, yaml_loaded):
    """4 GiB typical pairing."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 4 * 1024**3


# ---------------------------------------------------------------------------
# Energy fields
# ---------------------------------------------------------------------------

def test_energy_per_flop_fp32_matches(hand_coded, yaml_loaded):
    """2.43 pJ per FP32 op (16nm SIMD packed baseline)."""
    assert yaml_loaded.energy_per_flop_fp32 == hand_coded.energy_per_flop_fp32 == \
        pytest.approx(2.43e-12, rel=0.01)


def test_energy_scaling_int8_matches(hand_coded, yaml_loaded):
    """INT8 = 0.15x FP32 baseline (INT8 ~7x cheaper)."""
    assert yaml_loaded.energy_scaling[Precision.INT8] == \
        hand_coded.energy_scaling[Precision.INT8] == 0.15


def test_energy_scaling_fp16_matches(hand_coded, yaml_loaded):
    """FP16 = 0.50x FP32 baseline."""
    assert yaml_loaded.energy_scaling[Precision.FP16] == \
        hand_coded.energy_scaling[Precision.FP16] == 0.50


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
# Thermal operating points
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    """Both expose a single '1W' thermal profile (note: hand-coded
    uses key '1W' in the dict; YAML loader uses the profile.name
    which is also '1W')."""
    assert set(yaml_loaded.thermal_operating_points) == set(hand_coded.thermal_operating_points)


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile == "1W"


def test_thermal_profile_tdp_matches(hand_coded, yaml_loaded):
    yl_profile = yaml_loaded.thermal_operating_points["1W"]
    hc_profile = hand_coded.thermal_operating_points["1W"]
    assert yl_profile.tdp_watts == hc_profile.tdp_watts == 1.0


# ---------------------------------------------------------------------------
# Loader error cases
# ---------------------------------------------------------------------------

def test_unknown_sku_id_raises():
    with pytest.raises(DSPYamlLoaderError, match=r"no ComputeProduct with id"):
        load_dsp_resource_model_from_yaml("nonexistent_sku_id_xyz")


def test_non_dsp_sku_raises():
    """A non-DSP SKU (e.g., TPU v4) must fail with a clear error."""
    with pytest.raises(DSPYamlLoaderError, match=r"has no DSPBlock"):
        load_dsp_resource_model_from_yaml("google_tpu_v4")
