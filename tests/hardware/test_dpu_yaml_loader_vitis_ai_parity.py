"""Contract test: Vitis AI B4096 YAML loader produces the expected model.

PR 4 of the DPU mini-sprint scoped at issue #200. Mirror of
``tests/hardware/test_cgra_yaml_loader_plasticine_parity.py``.
Compares the YAML-loaded model against the hand-coded factory at
``src/graphs/hardware/models/accelerators/xilinx_vitis_ai_dpu.py``.

PR 5 will retire the hand-coded body and make ``xilinx_vitis_ai_dpu_resource_model()``
a thin wrapper over the loader; at that point both ``hand_coded`` and
``yaml_loaded`` fixtures will resolve through the same loader and
these structural assertions become contract tests.

Documented drifts (the loader follows a different convention; parity
test pins the YAML values as the new contract):

  - ``INT8 peak: 10.24 TOPS`` (yaml THEORETICAL) vs ``7.68 TOPS``
    (hand-coded REALISTIC at 75% efficiency). The YAML reports
    theoretical chip-level peak in precision_profiles; efficiency
    is captured separately in thermal_operating_points'
    efficiency_factor_by_precision. The Vitis AI marketed number
    is 7.68 TOPS (realistic) but the schema convention is theoretical
    peak; downstream consumers apply efficiency from the thermal
    profile.
  - ``FP16 peak: 2.56 TOPS`` (yaml THEORETICAL) vs ``1.92 TOPS``
    (hand-coded REALISTIC). Same root cause -- 75% efficiency factor.
  - ``FP32 peak``: not in YAML precision_profiles vs 0.96 TFLOPS
    (hand-coded). YAML's compute_fabrics declare INT8 + FP16 in
    ops_per_unit_per_clock; FP32 is captured only as energy_scaling
    (for cost modeling) since it's emulated. Downstream consumers
    can synthesize FP32 ops as INT8 / fp32_scaling if needed.
  - ``memory_technology``: ``"DDR4"`` (yaml) vs ``None`` (hand-coded).
    The legacy doesn't populate this field; the YAML loader does.
  - ``soc_fabric``: populated (yaml) vs ``None`` (hand-coded). Same.
  - ``energy_per_flop_fp32``: ~8% drift (2.5 pJ yaml vs 2.7 pJ hand-coded).
    YAML uses INT8 * FPGA-overhead * FP32-scaling = 0.4 * 1.25 * 5.0
    = 2.5 pJ. Hand-coded uses ``get_base_alu_energy(16, 'standard_cell')``
    = 2.7 pJ. Both are valid 16nm estimates.

Where the hand-coded and YAML-loaded agree (the actual parity
assertions):

  - compute_units, threads_per_unit, warps_per_unit
  - peak_bandwidth (50 GB/s -- DDR4, chip-attached)
  - l1_cache_per_unit, l2_cache_total, main_memory
  - default_precision (INT8)
  - hardware_type (DPU)
  - thermal profile shape (single profile, 20W)
  - scheduler attrs (min_occupancy, max_concurrent_kernels=4,
    wave_quantization=2)
"""

import pytest

from graphs.hardware.models.accelerators.dpu_yaml_loader import (
    DPUYamlLoaderError,
    load_dpu_resource_model_from_yaml,
)
from graphs.hardware.models.accelerators.xilinx_vitis_ai_dpu import (
    xilinx_vitis_ai_dpu_resource_model,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "xilinx_vitis_ai_b4096"
LEGACY_NAME = "DPU-Vitis-AI-B4096"


@pytest.fixture(scope="module")
def hand_coded():
    return xilinx_vitis_ai_dpu_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_dpu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == "DPU-Vitis-AI-B4096"


def test_hardware_type_is_dpu(hand_coded, yaml_loaded):
    """Both produce HardwareType.DPU -- no transitional period needed
    (DPU already in graphs enum)."""
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.DPU


def test_compute_units_matches(hand_coded, yaml_loaded):
    """64 AIE-ML v1 tiles."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 64


def test_threads_per_unit_matches(hand_coded, yaml_loaded):
    """64 MACs per AIE tile."""
    assert yaml_loaded.threads_per_unit == hand_coded.threads_per_unit == 64


def test_warps_per_unit_matches(hand_coded, yaml_loaded):
    """8 SIMD vector lanes per AIE-ML v1 tile."""
    assert yaml_loaded.warps_per_unit == hand_coded.warps_per_unit == 8


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """Vitis AI B4096 has one fabric (AIE-ML v1 tile array)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_compute_fabric_is_aie_ml_tile(hand_coded, yaml_loaded):
    """First SKU to exercise DPUFabricKind.AIE_ML_V1 -> fabric_type
    'aie_ml_tile' via the loader's mapping table."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "aie_ml_tile"
    assert hand_coded.compute_fabrics[0].fabric_type == "aie_ml_tile"


def test_compute_fabric_num_units(hand_coded, yaml_loaded):
    """num_units = 64 AIE tiles."""
    assert yaml_loaded.compute_fabrics[0].num_units == hand_coded.compute_fabrics[0].num_units == 64


def test_compute_fabric_int8_ops_per_clock(hand_coded, yaml_loaded):
    """128 INT8 ops/AIE tile/clock = 64 MACs * 2 ops/MAC."""
    hand_int8 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    yaml_int8 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    assert yaml_int8 == hand_int8 == 128


def test_compute_fabric_fp16_ops_per_clock(hand_coded, yaml_loaded):
    """32 FP16 ops/AIE tile/clock = 1/4 INT8 (native AIE-ML)."""
    hand_fp16 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.FP16)
    yaml_fp16 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.FP16)
    assert yaml_fp16 == hand_fp16 == 32


def test_compute_fabric_energy_documented_drift(hand_coded, yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded uses
    ``get_base_alu_energy(16, 'standard_cell')`` ~= 2.7 pJ; loader
    uses INT8 * FPGA-overhead * FP32-scaling = 0.4 * 1.25 * 5.0 =
    2.5 pJ. ~8% drift; both are valid 16nm estimates."""
    for fab in (hand_coded.compute_fabrics[0], yaml_loaded.compute_fabrics[0]):
        # Range sanity check: 16nm FP32 should be ~2-5 pJ
        assert 1e-12 < fab.energy_per_flop_fp32 < 10e-12, (
            f"fabric energy_per_flop_fp32={fab.energy_per_flop_fp32} "
            f"outside [1, 10] pJ range for 16nm"
        )


# ---------------------------------------------------------------------------
# Precision profiles -- different conventions (theoretical vs realistic)
# ---------------------------------------------------------------------------

def test_int8_peak_yaml_is_theoretical_chip_level(yaml_loaded):
    """YAML-loaded INT8 peak = 10.24 TOPS = 64 tiles * 128 ops/clk *
    1.25 GHz. Theoretical chip-level peak (schema convention).
    Efficiency factor (75%) lives in thermal_operating_points'
    efficiency_factor_by_precision, applied by downstream consumers."""
    assert yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec == pytest.approx(
        10.24e12, rel=0.01
    )


def test_int8_peak_hand_coded_is_realistic(hand_coded):
    """Pinned for visibility: legacy hand-coded reports 7.68 TOPS
    (= 10.24 * 0.75 efficiency). Pre-multiplied efficiency factor.
    PR 5 cleanup will retire the hand-coded path; downstream consumers
    will read theoretical from precision_profiles and apply efficiency
    from the thermal profile."""
    legacy_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert legacy_peak == pytest.approx(7.68e12, rel=0.01), (
        f"unexpected legacy peak {legacy_peak/1e12:.3f} TOPS; if this "
        f"changed the legacy was updated -- review this drift annotation."
    )


def test_fp16_peak_yaml_is_theoretical(yaml_loaded):
    """YAML FP16 peak = 2.56 TFLOPS theoretical. Native AIE-ML FP16."""
    assert yaml_loaded.precision_profiles[Precision.FP16].peak_ops_per_sec == pytest.approx(
        2.56e12, rel=0.01
    )


def test_yaml_omits_emulated_fp32_from_precision_profiles(yaml_loaded, hand_coded):
    """DOCUMENTED DRIFT: YAML's compute_fabrics declare INT8 + FP16 in
    ops_per_unit_per_clock; FP32 is captured only as energy_scaling
    (for cost modeling) since it's emulated. Hand-coded includes FP32
    in precision_profiles at 0.96 TFLOPS (emulated, 1/8 INT8 realistic).
    Downstream consumers needing FP32 ops/sec can compute as
    INT8_realistic / 8."""
    assert Precision.FP32 not in yaml_loaded.precision_profiles
    # Hand-coded does include it (for reference)
    assert Precision.FP32 in hand_coded.precision_profiles


def test_default_precision_matches(hand_coded, yaml_loaded):
    """Both prefer INT8 as default."""
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


def test_int4_not_in_precision_profiles(hand_coded, yaml_loaded):
    """AIE-ML v1 doesn't support INT4 (added in AIE-ML v2)."""
    assert Precision.INT4 not in hand_coded.precision_profiles
    assert Precision.INT4 not in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Memory hierarchy (chip-attached DDR4)
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches_ddr4(hand_coded, yaml_loaded):
    """50 GB/s -- chip-attached DDR4-3200 dual-channel. The loader
    picks DDR4 (DPU loader convention: external DRAM is the
    bottleneck tier for DPU workloads, unlike CGRA where on-chip
    mesh dominates)."""
    assert yaml_loaded.peak_bandwidth == pytest.approx(50e9, rel=0.01)
    assert hand_coded.peak_bandwidth == pytest.approx(50e9, rel=0.01)


def test_main_memory_matches(hand_coded, yaml_loaded):
    """8 GiB chip-attached DDR4."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """64 KiB AIE tile scratchpad."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit
    assert yaml_loaded.l1_cache_per_unit == 64 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """4 MiB shared L2."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total
    assert yaml_loaded.l2_cache_total == 4 * 1024 * 1024


def test_l2_per_unit_calculated_correctly(yaml_loaded):
    """4 MiB / 64 tiles = 64 KiB per-tile share."""
    assert yaml_loaded.l2_cache_per_unit == (4 * 1024 * 1024) // 64


def test_l3_absent(yaml_loaded):
    """DPUs don't have L3 by construction."""
    assert yaml_loaded.l3_present is False
    assert yaml_loaded.l3_cache_total == 0


def test_l1_storage_kind_is_scratchpad(yaml_loaded):
    """AIE tile scratchpad is software-managed."""
    assert yaml_loaded.l1_storage_kind == "scratchpad"


def test_coherence_protocol_is_none(yaml_loaded):
    """AIE streaming dataflow has no coherence (compiler-routed)."""
    assert yaml_loaded.coherence_protocol == "none"


def test_yaml_populates_memory_technology(yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded does NOT set memory_technology
    (returns None); YAML loader populates it from external_dram_type.
    The YAML's "DDR4" is structurally correct."""
    assert yaml_loaded.memory_technology == "DDR4"


def test_hand_coded_memory_technology_is_none(hand_coded):
    """Pinned for visibility: PR 5 cleanup makes the YAML loader the
    only source; both sides will then report "DDR4"."""
    assert hand_coded.memory_technology is None


# ---------------------------------------------------------------------------
# SoC fabric (8x8 AIE mesh -- DRIFT: hand-coded doesn't populate)
# ---------------------------------------------------------------------------

def test_yaml_populates_soc_fabric(yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded does NOT set soc_fabric (returns
    None); YAML loader populates it from NoC schema. The YAML's
    8x8 mesh structure is correct and unblocks downstream Layer 6
    reporting that the hand-coded couldn't support."""
    from graphs.hardware.fabric_model import Topology
    assert yaml_loaded.soc_fabric is not None
    assert yaml_loaded.soc_fabric.topology == Topology.MESH_2D
    assert yaml_loaded.soc_fabric.mesh_dimensions == (8, 8)
    assert yaml_loaded.soc_fabric.controller_count == 64
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(80.0)
    assert yaml_loaded.soc_fabric.low_confidence is True


def test_hand_coded_soc_fabric_is_none(hand_coded):
    """Pinned for visibility: PR 5 cleanup makes the YAML loader the
    only source."""
    assert hand_coded.soc_fabric is None


# ---------------------------------------------------------------------------
# Thermal profiles (single 20W operating point)
# ---------------------------------------------------------------------------

def test_thermal_profile_count_matches(hand_coded, yaml_loaded):
    assert len(yaml_loaded.thermal_operating_points) == 1
    assert len(hand_coded.thermal_operating_points) == 1


def test_default_thermal_profile_tdp(hand_coded, yaml_loaded):
    """Both default profiles report 20W."""
    hand_default = hand_coded.thermal_operating_points[hand_coded.default_thermal_profile]
    yaml_default = yaml_loaded.thermal_operating_points[yaml_loaded.default_thermal_profile]
    assert yaml_default.tdp_watts == hand_default.tdp_watts == 20.0


# ---------------------------------------------------------------------------
# Scheduler attrs (DPU-specific: multi-model + pair-quantization)
# ---------------------------------------------------------------------------

def test_scheduler_attrs_match(hand_coded, yaml_loaded):
    """0.3 occupancy (FPGA fabric overhead), 4 concurrent models
    (DPUs partition AIE tiles), wave_q=2 (pair-quantization)."""
    assert yaml_loaded.min_occupancy == pytest.approx(0.3, rel=0.05)
    assert hand_coded.min_occupancy == pytest.approx(0.3, rel=0.05)
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 4
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 2


# ---------------------------------------------------------------------------
# DPU-specific: reconfig + FPGA overhead live on the schema, not on
# HardwareResourceModel
# ---------------------------------------------------------------------------

def test_reconfig_and_fpga_overhead_on_underlying_compute_product():
    """The Vitis AI YAML carries is_statically_reconfigurable=True,
    bitstream_load_time_ms=2000, and fpga_fabric_overhead_factor=1.25
    on the DPUBlock. The legacy HardwareResourceModel has no equivalent
    fields, so the loader doesn't surface them; downstream consumers
    that need them read directly from the ComputeProduct.

    This test pins the values via direct schema read so a future v7
    HardwareResourceModel extension that surfaces these has known-good
    references."""
    from embodied_schemas.loaders import load_compute_products
    from embodied_schemas import DPUBlock
    cp = load_compute_products().get("xilinx_vitis_ai_b4096")
    assert cp is not None
    block = next(b for b in cp.dies[0].blocks if isinstance(b, DPUBlock))
    assert block.is_statically_reconfigurable is True
    assert block.bitstream_load_time_ms == pytest.approx(2000.0)
    assert block.compute_fabrics[0].fpga_fabric_overhead_factor == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(DPUYamlLoaderError, match=r"no ComputeProduct with id"):
        load_dpu_resource_model_from_yaml("definitely_not_a_real_dpu")


def test_loader_raises_on_kpu_sku():
    """KPU ComputeProducts have no DPUBlock; loader should reject."""
    with pytest.raises(DPUYamlLoaderError, match=r"no DPUBlock"):
        load_dpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")


def test_loader_raises_on_npu_sku():
    """NPU ComputeProducts have no DPUBlock; loader should reject."""
    with pytest.raises(DPUYamlLoaderError, match=r"no DPUBlock"):
        load_dpu_resource_model_from_yaml("hailo_hailo_8")


def test_loader_raises_on_cgra_sku():
    """CGRA ComputeProducts have no DPUBlock; loader should reject."""
    with pytest.raises(DPUYamlLoaderError, match=r"no DPUBlock"):
        load_dpu_resource_model_from_yaml("stanford_plasticine_v2")


def test_loader_raises_on_gpu_sku():
    with pytest.raises(DPUYamlLoaderError, match=r"no DPUBlock"):
        load_dpu_resource_model_from_yaml("nvidia_jetson_agx_orin_64gb")


def test_loader_raises_on_cpu_sku():
    with pytest.raises(DPUYamlLoaderError, match=r"no DPUBlock"):
        load_dpu_resource_model_from_yaml("intel_core_i7_12700k")
