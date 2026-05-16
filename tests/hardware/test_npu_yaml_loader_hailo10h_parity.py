"""Contract test: Hailo-10H factory produces the expected model.

Mirror of ``test_npu_yaml_loader_hailo8_parity.py`` for the second NPU
SKU. The graphs cleanup PR (issue #192) retired the ~360-LOC hand-
coded body of ``hailo10h_resource_model()`` and made it a thin
wrapper over ``load_npu_resource_model_from_yaml``, so today both
``hand_coded`` and ``yaml_loaded`` fixtures resolve through the same
loader.

The structural assertions are contract tests on the YAML loader's
output. They catch loader regressions and YAML drift; they also fail
loudly if anyone changes the factory's name override or substitutes
a different SKU id.

One factory overlay exists for Hailo-10H (BoM cost; the v4 schema
doesn't carry BoM data yet). Pinned both that the factory adds it
AND that the loader alone does NOT.

Two documented shape drifts remain (v5 reconciliation items):
  - ``energy_per_flop_fp32`` synthesis -- NPUs don't ship FP32; the
    loader synthesizes it as ``energy_per_op_int8 * 8`` per the
    standard-cell rule of thumb (~1% drift from the hand-coded value).
  - ``default_precision`` -- the loader picks INT8 (NPU default); the
    hand-coded Hailo-10H used INT4 (primary GenAI use case). The thin
    factory does NOT override; v5 will add a per-SKU hint to the
    schema. Tested below as a documented-drift assertion.
"""

import pytest

from graphs.hardware.models.edge.hailo10h import hailo10h_resource_model
from graphs.hardware.models.edge.npu_yaml_loader import (
    NPUYamlLoaderError,
    load_npu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "hailo_hailo_10h"
LEGACY_NAME = "Hailo-10H"


@pytest.fixture(scope="module")
def hand_coded():
    return hailo10h_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_npu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == "Hailo-10H"


def test_hardware_type_is_npu(hand_coded, yaml_loaded):
    """Both produce HardwareType.NPU after the YAML migration."""
    assert yaml_loaded.hardware_type == hand_coded.hardware_type == HardwareType.NPU


def test_compute_units_matches(hand_coded, yaml_loaded):
    """40 dataflow units (vs Hailo-8's 32)."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 40


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """Hailo-10H has one fabric (structure-driven dataflow)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_compute_fabric_num_units(hand_coded, yaml_loaded):
    """num_units = 40 dataflow units on both."""
    assert yaml_loaded.compute_fabrics[0].num_units == hand_coded.compute_fabrics[0].num_units == 40


def test_compute_fabric_int8_ops_per_clock(hand_coded, yaml_loaded):
    """500 INT8 ops/unit/clock (same per-unit as Hailo-8; difference is unit count + clock)."""
    hand_int8 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    yaml_int8 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    assert yaml_int8 == hand_int8 == 500


def test_compute_fabric_int4_ops_per_clock(hand_coded, yaml_loaded):
    """1000 INT4 ops/unit/clock (2x INT8)."""
    hand_int4 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT4)
    yaml_int4 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT4)
    assert yaml_int4 == hand_int4 == 1000


def test_no_fp_precisions_in_fabric(hand_coded, yaml_loaded):
    """Hailo-10H doesn't ship FP -- neither fabric should expose FP precisions."""
    fp_precs = {Precision.FP64, Precision.FP32, Precision.FP16, Precision.BF16}
    hand_precs = set(hand_coded.compute_fabrics[0].ops_per_unit_per_clock)
    yaml_precs = set(yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock)
    assert not (hand_precs & fp_precs)
    assert not (yaml_precs & fp_precs)


def test_compute_fabric_energy_in_sane_range(hand_coded, yaml_loaded):
    """FP32 energy at 16nm. Hand-coded uses
    ``get_base_alu_energy(16, 'standard_cell')`` ~= 2.7 pJ; loader
    synthesizes from ``energy_per_op_int8 * 8`` ~= 2.72 pJ. Same
    ballpark; documented in module docstring."""
    for fab in (hand_coded.compute_fabrics[0], yaml_loaded.compute_fabrics[0]):
        assert 1e-12 < fab.energy_per_flop_fp32 < 10e-12, (
            f"fabric energy_per_flop_fp32={fab.energy_per_flop_fp32} "
            f"outside [1, 10] pJ range for 16nm"
        )


# ---------------------------------------------------------------------------
# Precision profiles -- the marketed TOPS numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", [Precision.INT8, Precision.INT4])
def test_precision_peak_ops_match(hand_coded, yaml_loaded, precision):
    """20 TOPS INT8, 40 TOPS INT4 (the marketed numbers). Tight 5%
    tolerance because both come from the same arithmetic (40 units *
    ops/clk * 1.0 GHz)."""
    if precision not in hand_coded.precision_profiles:
        pytest.skip(f"hand-coded doesn't expose {precision.name}")
    hand_peak = hand_coded.precision_profiles[precision].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[precision].peak_ops_per_sec
    assert yaml_peak == pytest.approx(hand_peak, rel=0.05)


def test_default_precision_documented_drift(hand_coded, yaml_loaded):
    """Documented drift: hand-coded picks INT4 (primary GenAI use
    case); loader picks INT8 (NPU default convention). v5 cleanup
    will add ``default_precision`` to ``NPUBlock`` so the YAML can
    pin this per-SKU; until then the thin factory does not override.

    This test pins the drift so a future v5 update fires it as a
    deliberate-cleanup signal."""
    assert hand_coded.default_precision == Precision.INT8, (
        "hand-coded factory now uses the YAML loader's default (INT8); "
        "if this assertion fails the v5 default_precision hint is "
        "presumably in -- update the factory to drop the workaround."
    )
    assert yaml_loaded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Memory hierarchy (this is the first NPU SKU with external DRAM)
# ---------------------------------------------------------------------------

def test_main_memory_matches(hand_coded, yaml_loaded):
    """8 GiB LPDDR4X. First NPU SKU with external DRAM (Hailo-8 is SRAM-only)."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 8 * 1024**3


def test_peak_bandwidth_picks_dram_when_external_dram_present(hand_coded, yaml_loaded):
    """Hailo-10H has LPDDR4X external DRAM, so peak_bandwidth picks
    the DRAM tier (40 GB/s), not the on-chip SRAM bandwidth. The
    loader update in this PR aligned with the hand-coded convention:
    DRAM dominates the roofline bottleneck when weights + KV cache
    live there."""
    # Hand-coded uses 40 GB/s (LPDDR4X); loader picks the same since
    # external_dram_bandwidth_gbps=40 in the YAML.
    assert yaml_loaded.peak_bandwidth == pytest.approx(40e9, rel=0.01)
    assert hand_coded.peak_bandwidth == pytest.approx(40e9, rel=0.01)


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """512 KiB per dataflow unit (sram_kib_per_unit * 1024)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit
    assert yaml_loaded.l1_cache_per_unit == 512 * 1024


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """12 MiB shared SRAM (KV-cache-sized; vs Hailo-8's 8 MiB)."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total
    assert yaml_loaded.l2_cache_total == 12 * 1024 * 1024


def test_l2_per_unit_matches(hand_coded, yaml_loaded):
    """12 MiB / 40 dataflow units = 307 KiB per-unit share."""
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
    """40 endpoints (one per dataflow unit)."""
    assert yaml_loaded.soc_fabric.controller_count == hand_coded.soc_fabric.controller_count == 40


def test_soc_fabric_bandwidth_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(
        hand_coded.soc_fabric.bisection_bandwidth_gbps, rel=0.01
    )


def test_soc_fabric_low_confidence_flag(hand_coded, yaml_loaded):
    """Hailo doesn't publish NoC details -- both flag low confidence."""
    assert yaml_loaded.soc_fabric.low_confidence is True
    assert hand_coded.soc_fabric.low_confidence is True


def test_soc_fabric_mesh_dimensions_match(hand_coded, yaml_loaded):
    """8x5 mesh of 40 dataflow units (vs Hailo-8's 8x4 / 32 units)."""
    assert yaml_loaded.soc_fabric.mesh_dimensions == hand_coded.soc_fabric.mesh_dimensions
    assert yaml_loaded.soc_fabric.mesh_dimensions == (8, 5)


# ---------------------------------------------------------------------------
# Thermal profiles (single 2.5W profile, no DVFS)
# ---------------------------------------------------------------------------

def test_thermal_profile_count_matches(hand_coded, yaml_loaded):
    """Single profile. Hand-coded used '2.5W-passive'; YAML uses '2.5W'.
    Allow either name; assert exactly one profile on each."""
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
    """0.75 occupancy (lower than Hailo-8's 0.85 -- transformers vary
    more), single concurrent model, wave_q=1."""
    assert yaml_loaded.min_occupancy == pytest.approx(0.75, rel=0.05)
    assert hand_coded.min_occupancy == pytest.approx(0.75, rel=0.05)
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 1
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 1


# ---------------------------------------------------------------------------
# PR specifics: factory's BoM overlay (the only thing the factory adds
# on top of the YAML loader's output)
# ---------------------------------------------------------------------------

def test_factory_attaches_bom_cost_profile(hand_coded):
    """The thin factory layers a BOMCostProfile onto the YAML-loaded
    model because the v4 ComputeProduct schema doesn't carry BoM data.

    Numbers should match the pre-cleanup hand-coded BoM:
    $30 die + $10 package + $20 LPDDR4X + $5 PCB + $1 thermal + $4 other
    = $70 BOM, $240 retail (Hailo-10H M.2 module estimated pricing)."""
    bom = hand_coded.bom_cost_profile
    assert bom is not None, "factory must attach BOMCostProfile (overlay)"
    assert bom.retail_price == pytest.approx(240.0)
    assert bom.process_node == "16nm"
    # Hailo-10H has LPDDR4X -- $20 memory cost (vs Hailo-8's $0)
    assert bom.memory_cost == pytest.approx(20.0)


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


# ---------------------------------------------------------------------------
# Hailo-10H-specific: KVCacheSpec round-trip from YAML
# ---------------------------------------------------------------------------

def test_kv_cache_present_on_underlying_compute_product():
    """The Hailo-10H YAML is the first SKU to populate kv_cache. The
    HardwareResourceModel doesn't carry KVCacheSpec yet (legacy field
    set predates the schema extension), but the loader's source
    ComputeProduct must have it. Pin via direct schema read."""
    from embodied_schemas.loaders import load_compute_products
    from embodied_schemas import KVCacheStreamingKind, NPUBlock
    cp = load_compute_products().get("hailo_hailo_10h")
    assert cp is not None
    block = next(b for b in cp.dies[0].blocks if isinstance(b, NPUBlock))
    assert block.kv_cache is not None
    assert block.kv_cache.has_offload_to_dram is True
    assert block.kv_cache.streaming_strategy == KVCacheStreamingKind.RING_BUFFER
    assert block.kv_cache.max_context_length == 8192


# ---------------------------------------------------------------------------
# Loader error paths (re-pinned here too; same as Hailo-8 parity)
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(NPUYamlLoaderError, match="no ComputeProduct with id"):
        load_npu_resource_model_from_yaml("definitely_not_a_real_npu")
