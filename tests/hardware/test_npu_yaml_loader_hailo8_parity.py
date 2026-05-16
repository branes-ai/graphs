"""Contract test: Hailo-8 factory produces the expected model.

Originally PR 4's parity test, comparing YAML-loaded against the
hand-coded factory. PR 5 retired the 360-LOC hand-coded body of
``hailo8_resource_model()`` and made it a thin wrapper over
``load_npu_resource_model_from_yaml``, so today both ``hand_coded``
and ``yaml_loaded`` fixtures resolve through the same loader.

The structural assertions are now **contract tests on the YAML
loader's output** rather than parity checks. They still catch loader
regressions and YAML drift; they also fail loudly if anyone changes
the factory's name override or substitutes a different SKU id.

One factory overlay exists for Hailo-8 (the BoM cost; the v4 schema
doesn't carry BoM data yet). No memory_clock overlay needed -- NPUs
don't gate memory clock on thermal profile (single fixed clock,
SRAM-only memory in the common case). The BoM-overlay tests below
are specific to PR 5 and pin both that the factory adds it AND that
the loader alone does NOT (so when v5 absorbs Market.bom the
overlay-presence test fires as a deliberate-cleanup signal).

Two documented shape quirks remain (both v5 reconciliation items):
  - ``HardwareType.KPU`` (sic!) -- the graphs ``HardwareType`` enum
    has no NPU value; loader preserves the hand-coded choice for
    parity. Adding ``HardwareType.NPU`` is a separate followup.
  - ``energy_per_flop_fp32`` synthesis -- NPUs don't ship FP32; the
    loader synthesizes it as ``energy_per_op_int8 * 8`` per the
    standard-cell rule of thumb.
"""

import pytest

from graphs.hardware.models.edge.hailo8 import hailo8_resource_model
from graphs.hardware.models.edge.npu_yaml_loader import (
    NPUYamlLoaderError,
    load_npu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "hailo_hailo_8"
LEGACY_NAME = "Hailo-8"


@pytest.fixture(scope="module")
def hand_coded():
    return hailo8_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_npu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == "Hailo-8"


def test_hardware_type_matches(hand_coded, yaml_loaded):
    """Both produce HardwareType.KPU (the graphs enum has no NPU value).
    Adding HardwareType.NPU is a separate followup -- this test fires
    when that lands so the substitution can happen deliberately."""
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.KPU


def test_compute_units_matches(hand_coded, yaml_loaded):
    """32 dataflow units."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 32


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """Hailo-8 has one fabric (structure-driven dataflow)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_compute_fabric_num_units(hand_coded, yaml_loaded):
    """num_units = 32 dataflow units on both."""
    assert yaml_loaded.compute_fabrics[0].num_units == hand_coded.compute_fabrics[0].num_units == 32


def test_compute_fabric_int8_ops_per_clock(hand_coded, yaml_loaded):
    """500 INT8 ops/unit/clock."""
    hand_int8 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    yaml_int8 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    assert yaml_int8 == hand_int8 == 500


def test_compute_fabric_int4_ops_per_clock(hand_coded, yaml_loaded):
    """1000 INT4 ops/unit/clock (2x INT8)."""
    hand_int4 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT4)
    yaml_int4 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT4)
    assert yaml_int4 == hand_int4 == 1000


def test_no_fp_precisions_in_fabric(hand_coded, yaml_loaded):
    """Hailo-8 doesn't ship FP -- neither fabric should expose FP precisions."""
    fp_precs = {Precision.FP64, Precision.FP32, Precision.FP16, Precision.BF16}
    hand_precs = set(hand_coded.compute_fabrics[0].ops_per_unit_per_clock)
    yaml_precs = set(yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock)
    assert not (hand_precs & fp_precs)
    assert not (yaml_precs & fp_precs)


def test_compute_fabric_energy_in_sane_range(hand_coded, yaml_loaded):
    """FP32 energy at 16nm should be in the right order of magnitude.
    Hand-coded uses ``get_base_alu_energy(16, 'standard_cell')`` ~= 2.7 pJ;
    loader synthesizes from ``energy_per_op_int8 * 8`` ~= 2.72 pJ.
    Same ballpark; documented in module docstring."""
    for fab in (hand_coded.compute_fabrics[0], yaml_loaded.compute_fabrics[0]):
        assert 1e-12 < fab.energy_per_flop_fp32 < 10e-12, (
            f"fabric energy_per_flop_fp32={fab.energy_per_flop_fp32} "
            f"outside [1, 10] pJ range for 16nm"
        )


# ---------------------------------------------------------------------------
# Precision profiles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", [Precision.INT8, Precision.INT4])
def test_precision_peak_ops_match(hand_coded, yaml_loaded, precision):
    """26 TOPS INT8, 52 TOPS INT4 (the marketed numbers). Tight 5%
    tolerance because both come from the same arithmetic (32 units *
    ops/clk * 1.6 GHz)."""
    if precision not in hand_coded.precision_profiles:
        pytest.skip(f"hand-coded doesn't expose {precision.name}")
    hand_peak = hand_coded.precision_profiles[precision].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[precision].peak_ops_per_sec
    assert yaml_peak == pytest.approx(hand_peak, rel=0.05)


def test_default_precision_matches(hand_coded, yaml_loaded):
    """Both prefer INT8 as default."""
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory hierarchy (SRAM-only on Hailo-8)
# ---------------------------------------------------------------------------

def test_main_memory_is_zero(hand_coded, yaml_loaded):
    """No external DRAM on Hailo-8."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 0


def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    """200 GB/s on-chip SRAM bandwidth."""
    assert yaml_loaded.peak_bandwidth == pytest.approx(
        hand_coded.peak_bandwidth, rel=0.01
    )


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """512 KiB per dataflow unit (sram_kib_per_unit * 1024)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit
    assert yaml_loaded.l1_cache_per_unit == 512 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """8 MiB shared SRAM (the "LLC" of NPU-land)."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total
    assert yaml_loaded.l2_cache_total == 8 * 1024 * 1024


def test_l2_per_unit_matches(hand_coded, yaml_loaded):
    """8 MiB / 32 dataflow units = 256 KiB per-unit share."""
    assert yaml_loaded.l2_cache_per_unit == hand_coded.l2_cache_per_unit


def test_l3_absent(hand_coded, yaml_loaded):
    """NPUs don't have L3 by construction."""
    assert yaml_loaded.l3_present is False
    assert hand_coded.l3_present is False
    assert yaml_loaded.l3_cache_total == hand_coded.l3_cache_total == 0


def test_l1_storage_kind_is_scratchpad(hand_coded, yaml_loaded):
    """NPUs use software-managed scratchpad, not hardware cache."""
    assert yaml_loaded.l1_storage_kind == hand_coded.l1_storage_kind == "scratchpad"


def test_coherence_protocol_is_none(hand_coded, yaml_loaded):
    """Dataflow architectures have no coherence (compiler-routed)."""
    assert yaml_loaded.coherence_protocol == hand_coded.coherence_protocol == "none"


# ---------------------------------------------------------------------------
# SoC fabric
# ---------------------------------------------------------------------------

def test_soc_fabric_topology_is_mesh_2d(hand_coded, yaml_loaded):
    """Both report MESH_2D for the dataflow mesh."""
    from graphs.hardware.fabric_model import Topology
    assert yaml_loaded.soc_fabric.topology == hand_coded.soc_fabric.topology
    assert yaml_loaded.soc_fabric.topology == Topology.MESH_2D


def test_soc_fabric_controller_count(hand_coded, yaml_loaded):
    """32 endpoints (one per dataflow unit)."""
    assert yaml_loaded.soc_fabric.controller_count == hand_coded.soc_fabric.controller_count == 32


def test_soc_fabric_bandwidth_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(
        hand_coded.soc_fabric.bisection_bandwidth_gbps, rel=0.01
    )


def test_soc_fabric_low_confidence_flag(hand_coded, yaml_loaded):
    """Hailo doesn't publish NoC details -- both flag low confidence."""
    assert yaml_loaded.soc_fabric.low_confidence is True
    assert hand_coded.soc_fabric.low_confidence is True


def test_soc_fabric_mesh_dimensions_match(hand_coded, yaml_loaded):
    """8x4 mesh of 32 dataflow units."""
    assert yaml_loaded.soc_fabric.mesh_dimensions == hand_coded.soc_fabric.mesh_dimensions
    assert yaml_loaded.soc_fabric.mesh_dimensions == (8, 4)


# ---------------------------------------------------------------------------
# Thermal profiles (single profile, no DVFS)
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    """Single profile. Hand-coded uses '2.5W-passive'; YAML uses '2.5W'.
    Allow either, but assert exactly one profile on each."""
    assert len(yaml_loaded.thermal_operating_points) == 1
    assert len(hand_coded.thermal_operating_points) == 1


def test_default_thermal_profile_tdp(hand_coded, yaml_loaded):
    """Both default profiles must report 2.5W."""
    hand_default = hand_coded.thermal_operating_points[hand_coded.default_thermal_profile]
    yaml_default = yaml_loaded.thermal_operating_points[yaml_loaded.default_thermal_profile]
    assert yaml_default.tdp_watts == hand_default.tdp_watts == 2.5


# ---------------------------------------------------------------------------
# Scheduler attrs
# ---------------------------------------------------------------------------

def test_scheduler_attrs_match(hand_coded, yaml_loaded):
    """High occupancy (0.8+), single concurrent model, wave_q=1."""
    # NB: hand-coded uses min_occupancy=0.8; YAML uses 0.85.
    # Both >= 0.8 -- pin the floor, not the exact value.
    assert yaml_loaded.min_occupancy >= 0.8
    assert hand_coded.min_occupancy >= 0.8
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 1
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 1


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(NPUYamlLoaderError, match="no ComputeProduct with id"):
        load_npu_resource_model_from_yaml("definitely_not_a_real_npu")


def test_loader_raises_on_kpu_sku():
    """KPU ComputeProducts have no NPUBlock; loader should reject."""
    with pytest.raises(NPUYamlLoaderError, match="no NPUBlock"):
        load_npu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")


def test_loader_raises_on_gpu_sku():
    with pytest.raises(NPUYamlLoaderError, match="no NPUBlock"):
        load_npu_resource_model_from_yaml("nvidia_jetson_agx_orin_64gb")


def test_loader_raises_on_cpu_sku():
    with pytest.raises(NPUYamlLoaderError, match="no NPUBlock"):
        load_npu_resource_model_from_yaml("intel_core_i7_12700k")


# ---------------------------------------------------------------------------
# PR 5 specifics: factory's BoM overlay (the only thing the factory adds
# on top of the YAML loader's output)
# ---------------------------------------------------------------------------

def test_factory_attaches_bom_cost_profile(hand_coded):
    """The thin factory layers a BOMCostProfile onto the YAML-loaded
    model because the v4 ComputeProduct schema doesn't carry BoM data.
    Validation scripts that read ``hardware.bom_cost_profile`` would
    fall through to None without this overlay -- gracefully degrading
    but losing the cost data.

    Numbers should match the pre-PR-5 hand-coded BoM:
    $25 die + $8 package + $0 memory + $4 PCB + $1 thermal + $2 other
    = $40 BOM, $160 retail (Hailo-8 M.2 module market pricing)."""
    bom = hand_coded.bom_cost_profile
    assert bom is not None, "factory must attach BOMCostProfile (PR 5 overlay)"
    assert bom.retail_price == pytest.approx(160.0)
    assert bom.process_node == "16nm"
    # Hailo-8 is all on-chip SRAM -- no external memory cost
    assert bom.memory_cost == 0.0


def test_loader_alone_does_not_attach_bom_cost(yaml_loaded):
    """Confirms BoM is the FACTORY's contribution, not the loader's.
    When v5 schema adds Market.bom and the loader starts populating
    it, this test will fail and force a deliberate update -- delete
    the factory's overlay and this guard."""
    assert yaml_loaded.bom_cost_profile is None, (
        "loader is not expected to attach BOMCostProfile; the v4 "
        "ComputeProduct schema doesn't carry BoM data. If this fails, "
        "the schema gained a Market.bom field; update the factory to "
        "drop its BoM overlay accordingly."
    )
