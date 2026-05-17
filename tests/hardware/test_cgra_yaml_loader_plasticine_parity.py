"""Contract test: Plasticine v2 YAML loader produces the expected model.

PR 4 of the CGRA mini-sprint scoped at issue #196. Mirror of
``tests/hardware/test_npu_yaml_loader_hailo8_parity.py``. Compares the
YAML-loaded model against the hand-coded factory at
``src/graphs/hardware/models/accelerators/stanford_plasticine_cgra.py``.

PR 5 will retire the hand-coded body and make ``stanford_plasticine_cgra_resource_model()``
a thin wrapper over the loader; at that point both ``hand_coded`` and
``yaml_loaded`` fixtures will resolve through the same code path and
these structural assertions become contract tests on the loader.

Documented drifts (the loader CORRECTS legacy bugs; parity test pins
the YAML values as the new contract):

  - ``INT8 peak: 10.24 TOPS`` (yaml) vs ``0.307 TOPS`` (hand-coded).
    The legacy factory's ``theoretical_tops`` formula uses
    ``num_pcus * macs_per_pcu * ops_per_mac * clock`` = 32*8*2*1e9 =
    0.512 TOPS, then * 0.60 efficiency = 0.307 TOPS. This ignores the
    PCU's dual-issue / pipeline multiplier. The YAML's ComputeFabric
    correctly reports 320 ops/PCU/clock = 10.24 TOPS chip-level
    (which matches the Plasticine paper's marketed number).
  - ``FP16 peak: 2.56 TOPS`` (yaml) vs ``0.077 TOPS`` (hand-coded).
    Same root cause.
  - ``memory_technology``: ``"DDR4"`` (yaml) vs ``None`` (hand-coded).
    The legacy doesn't populate this field; the YAML loader does.
  - ``soc_fabric``: populated (yaml) vs ``None`` (hand-coded). Same.
  - ``energy_per_flop_fp32``: ~0.5% drift (4.02 pJ vs 4.0 pJ). The
    YAML loader uses the CGRA fabric's FP32 scaling (6.7x INT8 =
    4.02 pJ); the hand-coded uses ``get_base_alu_energy(28,
    'standard_cell')`` = 4.0 pJ. Both are correct interpretations.

Where the hand-coded and YAML-loaded agree (these are the actual
parity assertions):

  - compute_units, threads_per_unit
  - peak_bandwidth (40 GB/s -- PCU mesh, on-chip)
  - l1_cache_per_unit, l2_cache_total, main_memory
  - default_precision (INT8)
  - hardware_type (CGRA)
  - thermal profile shape (single profile, 15W)
  - scheduler attrs (min_occupancy, max_concurrent_kernels)
"""

import pytest

from graphs.hardware.models.accelerators.cgra_yaml_loader import (
    CGRAYamlLoaderError,
    load_cgra_resource_model_from_yaml,
)
from graphs.hardware.models.accelerators.stanford_plasticine_cgra import (
    stanford_plasticine_cgra_resource_model,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "stanford_plasticine_v2"
LEGACY_NAME = "CGRA-Plasticine-v2"


@pytest.fixture(scope="module")
def hand_coded():
    return stanford_plasticine_cgra_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_cgra_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == "CGRA-Plasticine-v2"


def test_hardware_type_is_cgra(hand_coded, yaml_loaded):
    """Both produce HardwareType.CGRA -- no transitional period needed
    (unlike NPU's #191 enum gap)."""
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.CGRA


def test_compute_units_matches(hand_coded, yaml_loaded):
    """32 PCUs (Pattern Compute Units)."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 32


def test_threads_per_unit_matches(hand_coded, yaml_loaded):
    """8 MACs per PCU."""
    assert yaml_loaded.threads_per_unit == hand_coded.threads_per_unit == 8


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """Plasticine has one fabric (PCU spatial dataflow)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_compute_fabric_is_pcu_spatial_dataflow(hand_coded, yaml_loaded):
    """First SKU to exercise CGRAFabricKind.PCU_SPATIAL_DATAFLOW ->
    fabric_type 'pcu_spatial_dataflow' via the loader's mapping table."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "pcu_spatial_dataflow"
    assert hand_coded.compute_fabrics[0].fabric_type == "pcu_spatial_dataflow"


def test_compute_fabric_num_units(hand_coded, yaml_loaded):
    """num_units = 32 PCUs."""
    assert yaml_loaded.compute_fabrics[0].num_units == hand_coded.compute_fabrics[0].num_units == 32


def test_compute_fabric_int8_ops_per_clock(hand_coded, yaml_loaded):
    """320 INT8 ops/PCU/clock (the actual chip-level number; the
    legacy hand-coded's PrecisionProfile uses a buggy formula but the
    ComputeFabric is correct)."""
    hand_int8 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    yaml_int8 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    assert yaml_int8 == hand_int8 == 320


def test_compute_fabric_fp16_ops_per_clock(hand_coded, yaml_loaded):
    """80 FP16 ops/PCU/clock (1/4 INT8 rate -- Plasticine emulates FP16)."""
    hand_fp16 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.FP16)
    yaml_fp16 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.FP16)
    assert yaml_fp16 == hand_fp16 == 80


def test_compute_fabric_energy_documented_drift(hand_coded, yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded uses ``get_base_alu_energy(28,
    'standard_cell')`` ~= 4.0 pJ; loader uses CGRA fabric's
    energy_per_op_int8 * energy_scaling[FP32] = 0.6 * 6.7 = 4.02 pJ.
    ~0.5% drift; both are valid interpretations of 28nm SLP FP32 energy."""
    for fab in (hand_coded.compute_fabrics[0], yaml_loaded.compute_fabrics[0]):
        # Range sanity check: 28nm planar FP32 should be ~3-8 pJ
        assert 1e-12 < fab.energy_per_flop_fp32 < 10e-12, (
            f"fabric energy_per_flop_fp32={fab.energy_per_flop_fp32} "
            f"outside [1, 10] pJ range for 28nm planar"
        )


# ---------------------------------------------------------------------------
# Precision profiles -- the YAML CORRECTS a legacy bug
# ---------------------------------------------------------------------------

def test_int8_peak_is_yaml_corrected_chip_level(yaml_loaded):
    """YAML-loaded INT8 peak = 10.24 TOPS (= 32 PCUs * 320 ops/clk *
    1 GHz). This is the actual Plasticine v2 marketed number. The
    legacy hand-coded reports 0.307 TOPS (a buggy formula that
    ignores the per-PCU ops/clock multiplier); the YAML loader
    CORRECTS this. Parity test pins the YAML value as the new contract."""
    assert yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec == pytest.approx(
        10.24e12, rel=0.01
    )


def test_legacy_int8_peak_undershoots_chip_level(hand_coded):
    """Pinned for visibility: legacy hand-coded reports 0.307 TOPS
    (about 33x smaller than the actual chip-level peak). PR 5 cleanup
    will retire the hand-coded path and this test becomes redundant
    (loader becomes the only source). Until then, document the bug."""
    legacy_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert legacy_peak == pytest.approx(0.307e12, rel=0.05), (
        f"unexpected legacy peak {legacy_peak/1e12:.3f} TOPS; if this "
        f"changed the legacy was fixed -- update this drift annotation."
    )


def test_default_precision_matches(hand_coded, yaml_loaded):
    """Both prefer INT8 as default."""
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


def test_int4_not_in_precision_profiles(hand_coded, yaml_loaded):
    """Plasticine targets INT8 minimum -- no INT4 (unlike Hailo SKUs)."""
    assert Precision.INT4 not in hand_coded.precision_profiles
    assert Precision.INT4 not in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Memory hierarchy (PMU + L2 + host DDR4)
# ---------------------------------------------------------------------------

def test_peak_bandwidth_matches_on_chip_mesh(hand_coded, yaml_loaded):
    """40 GB/s -- PCU mesh bisection. The loader picks on_chip not
    host_dram (CGRA workloads stay on-chip; host bus is only for
    bitstream load + spills). Matches Plasticine's hand-coded 40 GB/s."""
    assert yaml_loaded.peak_bandwidth == pytest.approx(40e9, rel=0.01)
    assert hand_coded.peak_bandwidth == pytest.approx(40e9, rel=0.01)


def test_main_memory_matches(hand_coded, yaml_loaded):
    """4 GiB host DDR4."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 4 * 1024**3


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """64 KiB PMU per PCU."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit
    assert yaml_loaded.l1_cache_per_unit == 64 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """2 MiB shared L2."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total
    assert yaml_loaded.l2_cache_total == 2 * 1024 * 1024


def test_l2_per_unit_calculated_correctly(yaml_loaded):
    """2 MiB / 32 PCUs = 64 KiB per-PCU share. Hand-coded doesn't
    set l2_cache_per_unit; pin the YAML's value."""
    assert yaml_loaded.l2_cache_per_unit == (2 * 1024 * 1024) // 32


def test_l3_absent(yaml_loaded):
    """CGRAs don't have L3 by construction."""
    assert yaml_loaded.l3_present is False
    assert yaml_loaded.l3_cache_total == 0


def test_l1_storage_kind_is_scratchpad(yaml_loaded):
    """PMU is software-managed scratchpad."""
    assert yaml_loaded.l1_storage_kind == "scratchpad"


def test_coherence_protocol_is_none(yaml_loaded):
    """Spatial dataflow has no coherence (compiler-routed)."""
    assert yaml_loaded.coherence_protocol == "none"


def test_yaml_populates_memory_technology(yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded does NOT set memory_technology
    (returns None); YAML loader populates it from host_dram_type.
    The YAML's "DDR4" is structurally correct."""
    assert yaml_loaded.memory_technology == "DDR4"


# ---------------------------------------------------------------------------
# SoC fabric (4x8 mesh -- DRIFT: hand-coded doesn't populate)
# ---------------------------------------------------------------------------

def test_yaml_populates_soc_fabric(yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded does NOT set soc_fabric (returns
    None); YAML loader populates it from NoC schema. The YAML's
    4x8 mesh structure is correct and unblocks downstream Layer 6
    reporting that the hand-coded couldn't support."""
    from graphs.hardware.fabric_model import Topology
    assert yaml_loaded.soc_fabric is not None
    assert yaml_loaded.soc_fabric.topology == Topology.MESH_2D
    assert yaml_loaded.soc_fabric.mesh_dimensions == (4, 8)
    assert yaml_loaded.soc_fabric.controller_count == 32
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(40.0)
    assert yaml_loaded.soc_fabric.low_confidence is True


def test_hand_coded_soc_fabric_is_none(hand_coded):
    """Pinned for visibility: hand-coded didn't model the SoC fabric.
    PR 5 cleanup makes the YAML loader the only source."""
    assert hand_coded.soc_fabric is None


# ---------------------------------------------------------------------------
# Thermal profiles (single 15W operating point)
# ---------------------------------------------------------------------------

def test_thermal_profile_count_matches(hand_coded, yaml_loaded):
    assert len(yaml_loaded.thermal_operating_points) == 1
    assert len(hand_coded.thermal_operating_points) == 1


def test_default_thermal_profile_tdp(hand_coded, yaml_loaded):
    """Both default profiles report 15W."""
    hand_default = hand_coded.thermal_operating_points[hand_coded.default_thermal_profile]
    yaml_default = yaml_loaded.thermal_operating_points[yaml_loaded.default_thermal_profile]
    assert yaml_default.tdp_watts == hand_default.tdp_watts == 15.0


# ---------------------------------------------------------------------------
# Scheduler attrs
# ---------------------------------------------------------------------------

def test_scheduler_attrs_match(hand_coded, yaml_loaded):
    """0.3 occupancy (CGRA fabric overhead; lower than fixed-function
    NPU), single concurrent program, wave_q=1."""
    assert yaml_loaded.min_occupancy == pytest.approx(0.3, rel=0.05)
    assert hand_coded.min_occupancy == pytest.approx(0.3, rel=0.05)
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 1
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 1


# ---------------------------------------------------------------------------
# CGRA-specific: reconfig overhead lives on the schema, not yet on
# HardwareResourceModel
# ---------------------------------------------------------------------------

def test_reconfig_overhead_on_underlying_compute_product():
    """The Plasticine v2 YAML carries reconfig_overhead_cycles=1000 on
    the CGRABlock. The legacy HardwareResourceModel has no equivalent
    field, so the loader doesn't surface it; downstream consumers
    that need it read directly from the ComputeProduct.

    This test pins the value via direct schema read so a future v6
    HardwareResourceModel extension that surfaces reconfig overhead
    has a known-good reference."""
    from embodied_schemas.loaders import load_compute_products
    from embodied_schemas import CGRABlock
    cp = load_compute_products().get("stanford_plasticine_v2")
    assert cp is not None
    block = next(b for b in cp.dies[0].blocks if isinstance(b, CGRABlock))
    assert block.reconfig_overhead_cycles == 1000
    assert block.supports_partial_reconfig is False


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(CGRAYamlLoaderError, match="no ComputeProduct with id"):
        load_cgra_resource_model_from_yaml("definitely_not_a_real_cgra")


def test_loader_raises_on_kpu_sku():
    """KPU ComputeProducts have no CGRABlock; loader should reject."""
    with pytest.raises(CGRAYamlLoaderError, match="no CGRABlock"):
        load_cgra_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")


def test_loader_raises_on_npu_sku():
    """NPU ComputeProducts have no CGRABlock; loader should reject."""
    with pytest.raises(CGRAYamlLoaderError, match="no CGRABlock"):
        load_cgra_resource_model_from_yaml("hailo_hailo_8")


def test_loader_raises_on_gpu_sku():
    with pytest.raises(CGRAYamlLoaderError, match="no CGRABlock"):
        load_cgra_resource_model_from_yaml("nvidia_jetson_agx_orin_64gb")


def test_loader_raises_on_cpu_sku():
    with pytest.raises(CGRAYamlLoaderError, match="no CGRABlock"):
        load_cgra_resource_model_from_yaml("intel_core_i7_12700k")
