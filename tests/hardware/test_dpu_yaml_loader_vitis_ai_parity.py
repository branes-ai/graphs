"""Contract test: Vitis AI B4096 factory produces the expected model.

Originally PR 4's parity test, comparing YAML-loaded against the
hand-coded factory. PR 5 retired the 160-LOC hand-coded body of
``xilinx_vitis_ai_dpu_resource_model()`` and made it a thin wrapper
over ``load_dpu_resource_model_from_yaml``, so today both
``hand_coded`` and ``yaml_loaded`` fixtures resolve through the same
loader.

The structural assertions are now **contract tests on the YAML
loader's output** rather than parity checks. They still catch loader
regressions and YAML drift; they also fail loudly if anyone changes
the factory's name override or substitutes a different SKU id.

Zero factory overlays exist for Vitis AI (like Plasticine; the
simplest cleanups of any sprint -- the YAML conventions match the
chip's natural representation).

One v7 reconciliation item:
  - ``is_statically_reconfigurable`` / ``bitstream_load_time_ms`` /
    ``fpga_fabric_overhead_factor`` (the defining DPU characteristics)
    live on ``DPUBlock`` in the v6 schema but have no equivalent
    fields on ``HardwareResourceModel``. Tested via direct
    ComputeProduct read below.

(Before PR 5 / cleanup, the legacy hand-coded had SIX documented
drifts -- buggy INT8/FP16/FP32 realistic-at-efficiency formulas,
missing memory_technology, missing soc_fabric, slightly different
FP32 energy. Those are all resolved now because the hand-coded
body is gone; both sides resolve through the loader.)
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

def test_int8_peak_is_theoretical_chip_level(hand_coded, yaml_loaded):
    """INT8 peak = 10.24 TOPS = 64 tiles * 128 ops/clk * 1.25 GHz.
    Theoretical chip-level peak (schema convention). Both factories
    now resolve through the loader; before PR 5 cleanup the legacy
    reported 7.68 TOPS (= 10.24 * 0.75 pre-applied efficiency).
    Downstream consumers that want realistic peak multiply by
    ``thermal_operating_points[default].efficiency_factor_by_precision[int8]``."""
    assert yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec == pytest.approx(
        10.24e12, rel=0.01
    )
    assert hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec == pytest.approx(
        10.24e12, rel=0.01
    )


def test_fp16_peak_is_theoretical(hand_coded, yaml_loaded):
    """FP16 peak = 2.56 TFLOPS theoretical. Native AIE-ML FP16."""
    assert yaml_loaded.precision_profiles[Precision.FP16].peak_ops_per_sec == pytest.approx(
        2.56e12, rel=0.01
    )
    assert hand_coded.precision_profiles[Precision.FP16].peak_ops_per_sec == pytest.approx(
        2.56e12, rel=0.01
    )


def test_fp32_not_in_precision_profiles(hand_coded, yaml_loaded):
    """Both factories now resolve through the loader, which declares
    INT8 + FP16 in compute_fabrics; FP32 is captured only as
    ``energy_scaling`` (for cost modeling) since it's emulated.
    Downstream consumers needing FP32 ops/sec compute as
    INT8 / fp32_scaling. Before PR 5 cleanup the legacy included
    FP32 at 0.96 TFLOPS in precision_profiles (emulated, 1/8 INT8
    realistic)."""
    assert Precision.FP32 not in yaml_loaded.precision_profiles
    assert Precision.FP32 not in hand_coded.precision_profiles


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


def test_memory_technology_populated(hand_coded, yaml_loaded):
    """Both factories now resolve through the loader, which populates
    ``memory_technology`` from the YAML's ``external_dram_type=ddr4``.
    Before PR 5 cleanup the legacy returned None here."""
    assert yaml_loaded.memory_technology == hand_coded.memory_technology == "DDR4"


# ---------------------------------------------------------------------------
# SoC fabric (8x8 AIE mesh -- DRIFT: hand-coded doesn't populate)
# ---------------------------------------------------------------------------

def test_soc_fabric_populated(hand_coded, yaml_loaded):
    """Both factories now resolve through the loader, which populates
    ``soc_fabric`` from the YAML's ``noc`` block (8x8 AIE mesh of 64
    tiles). Before PR 5 cleanup the legacy returned None, which
    blocked downstream Layer 6 reporting for DPU."""
    from graphs.hardware.fabric_model import Topology
    for label, m in (("yaml_loaded", yaml_loaded), ("hand_coded", hand_coded)):
        assert m.soc_fabric is not None, f"{label}.soc_fabric is None"
        assert m.soc_fabric.topology == Topology.MESH_2D
        assert m.soc_fabric.mesh_dimensions == (8, 8)
        assert m.soc_fabric.controller_count == 64
        assert m.soc_fabric.bisection_bandwidth_gbps == pytest.approx(80.0)
        assert m.soc_fabric.low_confidence is True


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
