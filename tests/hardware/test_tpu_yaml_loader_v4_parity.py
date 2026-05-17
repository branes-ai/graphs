"""Contract test: TPU v4 YAML loader produces the expected model.

PR 4 of the TPU mini-sprint scoped at issue #204. Mirror of
``tests/hardware/test_dpu_yaml_loader_vitis_ai_parity.py``. Compares
the YAML-loaded model against the hand-coded factory at
``src/graphs/hardware/models/datacenter/tpu_v4.py``.

PR 5 will retire the hand-coded body and make ``tpu_v4_resource_model()``
a thin wrapper over the loader; at that point both ``hand_coded`` and
``yaml_loaded`` fixtures will resolve through the same loader and
these structural assertions become contract tests.

Documented drifts:

  - ``energy_per_flop_fp32``: ~4x drift (0.45 pJ YAML vs 1.8 pJ
    hand-coded). YAML computes from BF16 baseline (0.225 pJ) *
    FP32 scaling (2.0) = 0.45 pJ. Hand-coded uses
    ``get_base_alu_energy(7, 'standard_cell')`` = 1.8 pJ static
    7nm baseline. The schema's BF16-relative FP32 scaling of 2.0
    is optimistic for emulated FP32; v8 reconciliation may add a
    direct ``energy_per_flop_fp32_override`` field.
  - ``memory_technology``: "HBM2E" (yaml) vs None (hand-coded).
    Legacy doesn't populate this field; YAML loader does.
  - **TPU v4 modeled as 2 MXUs (not 8)** -- both YAML and legacy
    model TPU v4 as 2 MXUs * 128x128 = 32,768 MACs. Google's actual
    TPU v4 has 2 TensorCores * 4 MXUs each = 8 MXUs * 128x128 =
    131,072 MACs (= 275 TFLOPS BF16 chip-level peak). The schema's
    `chip.performance.bf16_tflops = 275.0` matches the marketed
    number, but the `compute_fabric`-derived precision_profiles[BF16]
    gives 68.8 TOPS (= 1/4 of marketed). This is a documented
    legacy modeling choice that the parity test honors; v8
    reconciliation can model 2 TC * 4 MXUs properly.

Where the hand-coded and YAML-loaded agree (actual parity assertions):

  - compute_units, threads_per_unit, warps_per_unit, warp_size
  - peak_bandwidth (1.2 TB/s -- HBM2e)
  - l1_cache_per_unit (16 MiB per MXU = UB / 2)
  - l2_cache_total (32 MiB = full UB)
  - main_memory (32 GiB HBM2e)
  - default_precision (BF16 -- training-first)
  - hardware_type (TPU)
  - thermal profile shape (single profile, 350W liquid)
  - scheduler attrs (min_occupancy=0.5, max_concurrent_kernels=1,
    wave_quantization=1)
  - precision_profiles[BF16/INT8] peak ops (68.8 TOPS for 2-MXU model)
  - tile_energy_model (TPUTileEnergyModel attached on both sides)
"""

import pytest

from graphs.hardware.architectural_energy import TPUTileEnergyModel
from graphs.hardware.models.datacenter.tpu_v4 import tpu_v4_resource_model
from graphs.hardware.models.datacenter.tpu_yaml_loader import (
    TPUYamlLoaderError,
    load_tpu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "google_tpu_v4"
LEGACY_NAME = "TPU-v4"


@pytest.fixture(scope="module")
def hand_coded():
    return tpu_v4_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_tpu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == "TPU-v4"


def test_hardware_type_is_tpu(hand_coded, yaml_loaded):
    """Both produce HardwareType.TPU -- no transitional period
    (TPU already in graphs enum)."""
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.TPU


def test_compute_units_matches(hand_coded, yaml_loaded):
    """2 MXUs (legacy modeling; see drift note re: actual 8-MXU TPU v4)."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 2


def test_threads_per_unit_matches(hand_coded, yaml_loaded):
    """128 * 128 = 16,384 MACs per MXU."""
    assert yaml_loaded.threads_per_unit == hand_coded.threads_per_unit == 16384


def test_warps_per_unit_matches(hand_coded, yaml_loaded):
    """128 rows per systolic array."""
    assert yaml_loaded.warps_per_unit == hand_coded.warps_per_unit == 128


def test_warp_size_matches(hand_coded, yaml_loaded):
    """128 cols per systolic array."""
    assert yaml_loaded.warp_size == hand_coded.warp_size == 128


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """TPU v4 has one fabric (systolic array)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_compute_fabric_is_systolic_array(hand_coded, yaml_loaded):
    """First SKU to exercise TPUFabricKind.TPU_V2_PLUS -> fabric_type
    'systolic_array' via the loader's mapping table."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "systolic_array"
    assert hand_coded.compute_fabrics[0].fabric_type == "systolic_array"


def test_compute_fabric_num_units(hand_coded, yaml_loaded):
    """num_units = 2 MXUs * 128 * 128 = 32,768 total MACs."""
    assert yaml_loaded.compute_fabrics[0].num_units == hand_coded.compute_fabrics[0].num_units == 2 * 128 * 128


def test_compute_fabric_bf16_ops_per_clock(hand_coded, yaml_loaded):
    """BF16: 2 ops/MAC (multiply + accumulate)."""
    hand_bf16 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.BF16)
    yaml_bf16 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.BF16)
    assert yaml_bf16 == hand_bf16 == 2


def test_compute_fabric_int8_ops_per_clock(hand_coded, yaml_loaded):
    """INT8: 2 ops/MAC at the same per-MAC rate as BF16."""
    hand_int8 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    yaml_int8 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    assert yaml_int8 == hand_int8 == 2


def test_compute_fabric_energy_documented_drift(hand_coded, yaml_loaded):
    """DOCUMENTED DRIFT: ~4x drift in FP32 fabric energy.

    YAML: 0.225 pJ BF16 baseline * 2.0 FP32 scaling = 0.45 pJ
    Hand-coded: get_base_alu_energy(7, 'standard_cell') = 1.8 pJ

    Both are valid 7nm estimates with different methodologies. The
    schema's 2.0x BF16-relative FP32 scaling is optimistic for
    emulated FP32; v8 reconciliation may add a direct
    energy_per_flop_fp32_override field on TPUComputeFabric."""
    # Range sanity: 7nm FP32 should be ~0.2-3 pJ
    for fab in (hand_coded.compute_fabrics[0], yaml_loaded.compute_fabrics[0]):
        assert 1e-13 < fab.energy_per_flop_fp32 < 5e-12, (
            f"fabric energy_per_flop_fp32={fab.energy_per_flop_fp32} "
            f"outside [0.1, 5] pJ range for 7nm"
        )


# ---------------------------------------------------------------------------
# Precision profiles (the BF16/INT8 peaks match exactly)
# ---------------------------------------------------------------------------

def test_bf16_peak_matches(hand_coded, yaml_loaded):
    """68.8 TOPS BF16 (= 32,768 MACs * 2 ops * 1.05 GHz).

    Note: this is 1/4 of the marketed 275 TFLOPS BF16 because both
    legacy and YAML model TPU v4 as 2 MXUs (not the actual 8 MXUs =
    2 TensorCores * 4 MXUs). Documented drift; v8 reconciliation."""
    hand_peak = hand_coded.precision_profiles[Precision.BF16].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[Precision.BF16].peak_ops_per_sec
    assert yaml_peak == pytest.approx(hand_peak, rel=0.01)
    assert yaml_peak == pytest.approx(68.8e12, rel=0.01)


def test_int8_peak_matches(hand_coded, yaml_loaded):
    """68.8 TOPS INT8 (same per-MAC rate as BF16 in this fabric model)."""
    hand_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yaml_peak == pytest.approx(hand_peak, rel=0.01)


def test_default_precision_matches(hand_coded, yaml_loaded):
    """Both prefer BF16 as default (TPU training-first design)."""
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.BF16


def test_int4_not_in_precision_profiles(hand_coded, yaml_loaded):
    """TPU v4 doesn't support INT4 (added in v5p)."""
    assert Precision.INT4 not in hand_coded.precision_profiles
    assert Precision.INT4 not in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Memory hierarchy (UB + HBM2e)
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches_hbm2e(hand_coded, yaml_loaded):
    """1.2 TB/s HBM2e -- chip-attached external DRAM."""
    assert yaml_loaded.peak_bandwidth == pytest.approx(1.2e12, rel=0.01)
    assert hand_coded.peak_bandwidth == pytest.approx(1.2e12, rel=0.01)


def test_main_memory_matches(hand_coded, yaml_loaded):
    """32 GiB HBM2e."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 32 * 1024**3


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """16 MiB per MXU (= 32 MiB UB / 2 MXUs)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit
    assert yaml_loaded.l1_cache_per_unit == 16 * 1024 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """32 MiB shared (full UB)."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total
    assert yaml_loaded.l2_cache_total == 32 * 1024 * 1024


def test_l3_absent(yaml_loaded):
    """TPUs don't have L3."""
    assert yaml_loaded.l3_present is False
    assert yaml_loaded.l3_cache_total == 0


def test_l1_storage_kind_is_scratchpad(yaml_loaded):
    """UB is software-managed."""
    assert yaml_loaded.l1_storage_kind == "scratchpad"


def test_coherence_protocol_is_none(yaml_loaded):
    """XLA-routed systolic dataflow has no coherence."""
    assert yaml_loaded.coherence_protocol == "none"


def test_yaml_populates_memory_technology(yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded does NOT set memory_technology
    (returns None); YAML loader populates from external_dram_type.
    YAML's "HBM2E" is structurally correct."""
    assert yaml_loaded.memory_technology == "HBM2E"


def test_hand_coded_memory_technology_is_none(hand_coded):
    """Pinned for visibility: PR 5 cleanup makes both sides populate."""
    assert hand_coded.memory_technology is None


# ---------------------------------------------------------------------------
# SoC fabric (UB-to-MXU crossbar)
# ---------------------------------------------------------------------------

def test_yaml_populates_soc_fabric(yaml_loaded):
    """YAML loader populates soc_fabric from the noc block. Hand-coded
    didn't model the SoC fabric (returns None); PR 5 cleanup makes both
    sides match."""
    from graphs.hardware.fabric_model import Topology
    assert yaml_loaded.soc_fabric is not None
    assert yaml_loaded.soc_fabric.topology == Topology.CROSSBAR
    assert yaml_loaded.soc_fabric.controller_count == 2
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(2000.0)


def test_hand_coded_soc_fabric_is_none(hand_coded):
    """Pinned for visibility: PR 5 cleanup makes the YAML loader the
    only source."""
    assert hand_coded.soc_fabric is None


# ---------------------------------------------------------------------------
# Thermal profiles (single 350W liquid-cooled profile)
# ---------------------------------------------------------------------------

def test_thermal_profile_count_matches(hand_coded, yaml_loaded):
    assert len(yaml_loaded.thermal_operating_points) == 1
    assert len(hand_coded.thermal_operating_points) == 1


def test_default_thermal_profile_tdp(hand_coded, yaml_loaded):
    """Both default profiles report 350W."""
    hand_default = hand_coded.thermal_operating_points[hand_coded.default_thermal_profile]
    yaml_default = yaml_loaded.thermal_operating_points[yaml_loaded.default_thermal_profile]
    assert yaml_default.tdp_watts == hand_default.tdp_watts == 350.0


# ---------------------------------------------------------------------------
# Scheduler attrs
# ---------------------------------------------------------------------------

def test_scheduler_attrs_match(hand_coded, yaml_loaded):
    """0.5 occupancy (TPU systolic needs high utilization), 1 concurrent
    model (datacenter training pattern), wave_q=1."""
    assert yaml_loaded.min_occupancy == pytest.approx(0.5, rel=0.05)
    assert hand_coded.min_occupancy == pytest.approx(0.5, rel=0.05)
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 1
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 1


# ---------------------------------------------------------------------------
# Tile energy model -- TPU-specific architectural decomposition
# ---------------------------------------------------------------------------

def test_tile_energy_model_attached_on_both_sides(hand_coded, yaml_loaded):
    """Both factories attach a TPUTileEnergyModel. The YAML loader
    reconstructs it from the schema's TPUTileEnergyCoefficients
    sub-type + the per-MXU fields on TPUBlock."""
    assert isinstance(hand_coded.tile_energy_model, TPUTileEnergyModel)
    assert isinstance(yaml_loaded.tile_energy_model, TPUTileEnergyModel)


def test_tile_energy_model_dimensions_match(hand_coded, yaml_loaded):
    """128x128 array, 2 MXUs."""
    for tem in (hand_coded.tile_energy_model, yaml_loaded.tile_energy_model):
        assert tem.array_width == 128
        assert tem.array_height == 128
        assert tem.num_arrays == 2
        assert tem.unified_buffer_size == 32 * 1024 * 1024


def test_tile_energy_model_mac_energy_matches(hand_coded, yaml_loaded):
    """0.25 pJ per BF16 MAC (the TPU v4 reference number)."""
    assert hand_coded.tile_energy_model.mac_energy == pytest.approx(0.25e-12)
    assert yaml_loaded.tile_energy_model.mac_energy == pytest.approx(0.25e-12)


def test_tile_energy_model_weight_memory_energy_matches(hand_coded, yaml_loaded):
    """10 pJ/byte for HBM2e access."""
    assert hand_coded.tile_energy_model.weight_memory_energy_per_byte == pytest.approx(10e-12)
    assert yaml_loaded.tile_energy_model.weight_memory_energy_per_byte == pytest.approx(10e-12)


# ---------------------------------------------------------------------------
# TPU-specific: ICI surface lives on the schema, not yet on
# HardwareResourceModel
# ---------------------------------------------------------------------------

def test_ici_surface_on_underlying_compute_product():
    """The TPU v4 YAML carries ici_port_count=6, ici_bandwidth_per_port_gbps=400.0,
    and ici_topology_hint='3d_torus_2x2x2' on the TPUBlock. The legacy
    HardwareResourceModel has no equivalent fields, so the loader
    doesn't surface them; downstream consumers that need ICI surface
    read directly from the ComputeProduct.

    Pins values via direct schema read so a future v8 HardwareResourceModel
    extension that surfaces ICI has known-good references."""
    from embodied_schemas.loaders import load_compute_products
    from embodied_schemas import TPUBlock
    cp = load_compute_products().get("google_tpu_v4")
    assert cp is not None
    block = next(b for b in cp.dies[0].blocks if isinstance(b, TPUBlock))
    assert block.ici_port_count == 6
    assert block.ici_bandwidth_per_port_gbps == pytest.approx(400.0)
    assert block.ici_topology_hint == "3d_torus_2x2x2"


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(TPUYamlLoaderError, match=r"no ComputeProduct with id"):
        load_tpu_resource_model_from_yaml("definitely_not_a_real_tpu")


def test_loader_raises_on_kpu_sku():
    """KPU ComputeProducts have no TPUBlock; loader should reject."""
    with pytest.raises(TPUYamlLoaderError, match=r"no TPUBlock"):
        load_tpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")


def test_loader_raises_on_npu_sku():
    """NPU ComputeProducts have no TPUBlock; loader should reject."""
    with pytest.raises(TPUYamlLoaderError, match=r"no TPUBlock"):
        load_tpu_resource_model_from_yaml("hailo_hailo_8")


def test_loader_raises_on_cgra_sku():
    """CGRA ComputeProducts have no TPUBlock; loader should reject."""
    with pytest.raises(TPUYamlLoaderError, match=r"no TPUBlock"):
        load_tpu_resource_model_from_yaml("stanford_plasticine_v2")


def test_loader_raises_on_dpu_sku():
    """DPU ComputeProducts have no TPUBlock; loader should reject."""
    with pytest.raises(TPUYamlLoaderError, match=r"no TPUBlock"):
        load_tpu_resource_model_from_yaml("xilinx_vitis_ai_b4096")


def test_loader_raises_on_gpu_sku():
    with pytest.raises(TPUYamlLoaderError, match=r"no TPUBlock"):
        load_tpu_resource_model_from_yaml("nvidia_jetson_agx_orin_64gb")


def test_loader_raises_on_cpu_sku():
    with pytest.raises(TPUYamlLoaderError, match=r"no TPUBlock"):
        load_tpu_resource_model_from_yaml("intel_core_i7_12700k")
