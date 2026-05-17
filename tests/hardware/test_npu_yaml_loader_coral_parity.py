"""Contract test: Coral Edge TPU factory produces the expected model.

Mirror of ``test_npu_yaml_loader_hailo{8,10h}_parity.py`` for the third
NPU SKU migration. The graphs cleanup PR (final piece of issue #192)
retired the ~360-LOC hand-coded body of ``coral_edge_tpu_resource_model()``
and made it a thin wrapper over ``load_npu_resource_model_from_yaml``
plus SIX factory overlays.

The structural assertions are contract tests on the YAML loader's
output + the factory's overlays. They catch loader regressions, YAML
drift, and overlay drift; they also pin the v5 reconciliation items
(taxonomy mismatch, host-bus bandwidth, TPU-specific fields) so the
deliberate-cleanup signals fire when v5 lands.

The SIX factory overlays:
  1. hardware_type=TPU (taxonomy mismatch; v5 unify NPU/TPU split)
  2. peak_bandwidth=4 GB/s (host bus bottleneck; v5 host_memory_bw concept)
  3. memory_technology="LPDDR4 (host)" (consistent with peak_bandwidth)
  4. pipeline_fill_overhead=0.15 (TPU-specific; v5 add to NPUBlock)
  5. tile_energy_model=TPUTileEnergyModel (TPU-specific architectural model)
  6. BOMCostProfile (v5 Market.bom)

Two documented shape drifts:
  - energy_per_flop_fp32 -- ~38% drift from INT8*8 synthesis. Larger
    than Hailo's 1% because Coral's fabric INT8 energy (0.45 pJ)
    reflects systolic reuse already, so 8x-ing it overshoots the
    raw 28nm FP32 estimate of 2.6 pJ.
  - threads_per_unit -- YAML reports 4096 (= lanes_per_unit for the
    64x64 systolic array); legacy hand-coded was 256 (an estimate).
    YAML is structurally correct; parity test pins YAML as new contract.
"""

import pytest

from graphs.hardware.architectural_energy import TPUTileEnergyModel
from graphs.hardware.models.edge.coral_edge_tpu import coral_edge_tpu_resource_model
from graphs.hardware.models.edge.npu_yaml_loader import (
    NPUYamlLoaderError,
    load_npu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "google_coral_edge_tpu"
LEGACY_NAME = "Coral-Edge-TPU"


@pytest.fixture(scope="module")
def hand_coded():
    return coral_edge_tpu_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_npu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name == "Coral-Edge-TPU"


def test_factory_overlays_hardware_type_to_tpu(hand_coded, yaml_loaded):
    """OVERLAY 1: the v4 schema classifies Coral as NPU (NPUBlock-bearing);
    the graphs HardwareType taxonomy uses TPU for systolic accelerators
    so TPUMapper's guard accepts it. v5 reconciliation: unify the split."""
    # Hand-coded (= factory output) reports TPU after overlay
    assert hand_coded.hardware_type == HardwareType.TPU
    # YAML loader alone reports NPU (the schema's classification)
    assert yaml_loaded.hardware_type == HardwareType.NPU


def test_compute_units_matches(hand_coded, yaml_loaded):
    """1 dataflow unit (the 64x64 systolic array)."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 1


def test_threads_per_unit_yaml_corrects_legacy_estimate(hand_coded, yaml_loaded):
    """DOCUMENTED DRIFT: legacy hand-coded reported 256 (an estimate);
    the YAML reports 4096 (the actual 64x64 systolic array lane count).
    The YAML value is structurally correct. This parity test pins the
    YAML value as the new contract; the factory does NOT override."""
    assert yaml_loaded.threads_per_unit == hand_coded.threads_per_unit == 4096


# ---------------------------------------------------------------------------
# Compute fabric (systolic)
# ---------------------------------------------------------------------------

def test_single_compute_fabric(hand_coded, yaml_loaded):
    """Coral has one fabric (systolic array)."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 1


def test_compute_fabric_is_systolic_array(hand_coded, yaml_loaded):
    """First SKU to exercise ``NPUDataflowKind.SYSTOLIC`` -> fabric_type
    'systolic_array' via the loader's mapping table."""
    assert yaml_loaded.compute_fabrics[0].fabric_type == "systolic_array"
    assert hand_coded.compute_fabrics[0].fabric_type == "systolic_array"


def test_compute_fabric_int8_ops_per_clock(hand_coded, yaml_loaded):
    """8000 INT8 ops per unit per clock (4096 MACs * 2 ops each)."""
    yaml_int8 = yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    hand_int8 = hand_coded.compute_fabrics[0].ops_per_unit_per_clock.get(Precision.INT8)
    assert yaml_int8 == hand_int8 == 8000


def test_no_fp_or_int4_precisions_in_fabric(hand_coded, yaml_loaded):
    """Coral is INT8-only -- no FP, no INT4. Distinguishes from
    Hailo SKUs which support INT4."""
    extra_precs = {Precision.FP64, Precision.FP32, Precision.FP16,
                   Precision.BF16, Precision.INT4}
    yaml_precs = set(yaml_loaded.compute_fabrics[0].ops_per_unit_per_clock)
    hand_precs = set(hand_coded.compute_fabrics[0].ops_per_unit_per_clock)
    assert not (yaml_precs & extra_precs)
    assert not (hand_precs & extra_precs)


def test_compute_fabric_energy_documented_drift(hand_coded, yaml_loaded):
    """DOCUMENTED DRIFT: hand-coded uses
    ``get_base_alu_energy(14, 'standard_cell')`` ~= 2.6 pJ (Coral was
    14nm in legacy data); loader synthesizes ~3.6 pJ from
    ``energy_per_op_int8 * 8`` (0.45 * 8). 38% drift because Coral's
    fabric INT8 energy reflects systolic reuse already. v5 reconciliation."""
    # Both fall in the sane range; the drift is documented not tested as exact
    for fab in (hand_coded.compute_fabrics[0], yaml_loaded.compute_fabrics[0]):
        assert 1e-12 < fab.energy_per_flop_fp32 < 10e-12, (
            f"fabric energy_per_flop_fp32={fab.energy_per_flop_fp32} "
            f"outside [1, 10] pJ range"
        )


# ---------------------------------------------------------------------------
# Precision profiles
# ---------------------------------------------------------------------------

def test_int8_peak_matches_marketing(hand_coded, yaml_loaded):
    """4 TOPS INT8 (the marketed number). 1 unit * 8000 ops/clk * 500 MHz."""
    hand_peak = hand_coded.precision_profiles[Precision.INT8].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[Precision.INT8].peak_ops_per_sec
    assert yaml_peak == hand_peak == pytest.approx(4e12, rel=0.05)


def test_default_precision_is_int8(hand_coded, yaml_loaded):
    """Both default to INT8 -- the only precision Coral supports."""
    assert yaml_loaded.default_precision == hand_coded.default_precision == Precision.INT8


def test_int4_not_in_precision_profiles(hand_coded, yaml_loaded):
    """Distinguishes Coral from the Hailo SKUs (which expose INT4)."""
    assert Precision.INT4 not in hand_coded.precision_profiles
    assert Precision.INT4 not in yaml_loaded.precision_profiles


# ---------------------------------------------------------------------------
# Memory hierarchy
# ---------------------------------------------------------------------------

def test_factory_overlays_peak_bandwidth_to_host_bus(hand_coded, yaml_loaded):
    """OVERLAY 2: factory overrides loader's 128 GB/s (UB-to-systolic
    direct connect) with 4 GB/s (host bus bottleneck via USB 3.0 /
    PCIe Gen2). Coral has no external DRAM, so memory traffic crosses
    the host bus."""
    assert hand_coded.peak_bandwidth == pytest.approx(4e9, rel=0.01)
    # Loader alone picks the on-chip SRAM bandwidth
    assert yaml_loaded.peak_bandwidth == pytest.approx(128e9, rel=0.01)


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """512 KiB unified buffer (the UB acts as L1 in NPU-land)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit
    assert yaml_loaded.l1_cache_per_unit == 512 * 1024


def test_l2_collapses_into_unified_buffer(hand_coded, yaml_loaded):
    """TPU collapses L2 into the UB by design; both report 0."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total == 0


def test_main_memory_is_zero(hand_coded, yaml_loaded):
    """No external DRAM (host memory accessed via USB/PCIe; not modeled
    as main_memory in the legacy layer)."""
    assert yaml_loaded.main_memory == hand_coded.main_memory == 0


def test_l3_absent(hand_coded, yaml_loaded):
    """TPU has no L3 layer."""
    assert yaml_loaded.l3_present is False
    assert hand_coded.l3_present is False
    assert yaml_loaded.l3_cache_total == hand_coded.l3_cache_total == 0


def test_l1_storage_kind_is_scratchpad(hand_coded, yaml_loaded):
    """Software-managed unified buffer."""
    assert yaml_loaded.l1_storage_kind == hand_coded.l1_storage_kind == "scratchpad"


def test_coherence_protocol_is_none(hand_coded, yaml_loaded):
    """Systolic array has no inter-tile coherence."""
    assert yaml_loaded.coherence_protocol == hand_coded.coherence_protocol == "none"


def test_factory_overlays_memory_technology_to_host_lpddr(hand_coded, yaml_loaded):
    """OVERLAY 3: consistent with the peak_bandwidth overlay --
    Coral's effective memory tier is host LPDDR4 reached via the
    host bus, not on-chip SRAM."""
    assert hand_coded.memory_technology == "LPDDR4 (host)"
    # Loader alone sets "on-chip SRAM (no external DRAM)" by default
    assert "on-chip SRAM" in yaml_loaded.memory_technology


# ---------------------------------------------------------------------------
# SoC fabric (crossbar -- single dataflow unit)
# ---------------------------------------------------------------------------

def test_soc_fabric_topology_is_crossbar(hand_coded, yaml_loaded):
    """First SKU in catalog with CROSSBAR NoC -- single dataflow unit
    + UB direct connect."""
    from graphs.hardware.fabric_model import Topology
    assert yaml_loaded.soc_fabric.topology == hand_coded.soc_fabric.topology
    assert yaml_loaded.soc_fabric.topology == Topology.CROSSBAR


def test_soc_fabric_controller_count(hand_coded, yaml_loaded):
    """1 endpoint (the single systolic tile)."""
    assert yaml_loaded.soc_fabric.controller_count == hand_coded.soc_fabric.controller_count == 1


def test_soc_fabric_bandwidth_matches(hand_coded, yaml_loaded):
    """128 GB/s UB-to-systolic direct connect."""
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(
        hand_coded.soc_fabric.bisection_bandwidth_gbps, rel=0.01
    )
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(128.0)


# ---------------------------------------------------------------------------
# Thermal profile (single 2W operating point)
# ---------------------------------------------------------------------------

def test_thermal_profile_count_matches(hand_coded, yaml_loaded):
    """Single profile."""
    assert len(yaml_loaded.thermal_operating_points) == 1
    assert len(hand_coded.thermal_operating_points) == 1


def test_default_thermal_profile_tdp(hand_coded, yaml_loaded):
    """Both default profiles report 2.0W."""
    hand_default = hand_coded.thermal_operating_points[hand_coded.default_thermal_profile]
    yaml_default = yaml_loaded.thermal_operating_points[yaml_loaded.default_thermal_profile]
    assert yaml_default.tdp_watts == hand_default.tdp_watts == 2.0


# ---------------------------------------------------------------------------
# Scheduler attrs
# ---------------------------------------------------------------------------

def test_scheduler_attrs_match(hand_coded, yaml_loaded):
    """0.85 occupancy (systolic arrays sustain high occupancy on regular
    workloads), single concurrent model, wave_q=1."""
    assert yaml_loaded.min_occupancy == pytest.approx(0.85, rel=0.05)
    assert hand_coded.min_occupancy == pytest.approx(0.85, rel=0.05)
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels == 1
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization == 1


# ---------------------------------------------------------------------------
# Overlay 4: TPU-systolic pipeline_fill_overhead
# ---------------------------------------------------------------------------

def test_factory_overlays_pipeline_fill_overhead(hand_coded, yaml_loaded):
    """OVERLAY 4: derived from TPUMapper._analyze_systolic_utilization
    averaged over typical edge-AI matrix shapes. Not in the v4 schema.
    The deliberate-cleanup signal fires when v5 adds this to NPUBlock."""
    assert hand_coded.pipeline_fill_overhead == pytest.approx(0.15)
    # Loader alone returns the resource_model default (0.0 or unset)
    assert yaml_loaded.pipeline_fill_overhead in (0.0, None) or yaml_loaded.pipeline_fill_overhead == 0


# ---------------------------------------------------------------------------
# Overlay 5: TPU tile energy model
# ---------------------------------------------------------------------------

def test_factory_overlays_tile_energy_model(hand_coded, yaml_loaded):
    """OVERLAY 5: the architectural tile-energy decomposition (FIFO,
    accumulator, UB, MAC) is TPU-specific. Not in the v4 schema."""
    assert isinstance(hand_coded.tile_energy_model, TPUTileEnergyModel)
    assert hand_coded.tile_energy_model.array_width == 64
    assert hand_coded.tile_energy_model.array_height == 64
    # Loader alone doesn't attach
    assert getattr(yaml_loaded, "tile_energy_model", None) is None


# ---------------------------------------------------------------------------
# Overlay 6: BoM
# ---------------------------------------------------------------------------

def test_factory_attaches_bom_cost_profile(hand_coded):
    """OVERLAY 6: same pattern as Hailo-8/10H factories. Coral M.2
    module retails at $75; $25 BoM ($12 die + $5 package + $0 memory
    + $3 PCB + $1 thermal + $4 other)."""
    bom = hand_coded.bom_cost_profile
    assert bom is not None, "factory must attach BOMCostProfile (overlay)"
    assert bom.retail_price == pytest.approx(75.0)
    assert bom.process_node == "28nm"
    # No memory cost -- uses host memory
    assert bom.memory_cost == 0.0


def test_loader_alone_does_not_attach_bom_cost(yaml_loaded):
    """Confirms BoM is the FACTORY's contribution, not the loader's."""
    assert yaml_loaded.bom_cost_profile is None


# ---------------------------------------------------------------------------
# Loader error paths (re-pinned here for the systolic SKU)
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(NPUYamlLoaderError, match="no ComputeProduct with id"):
        load_npu_resource_model_from_yaml("definitely_not_a_real_npu")
