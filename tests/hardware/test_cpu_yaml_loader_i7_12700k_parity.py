"""Parity test: YAML-loaded i7-12700K matches hand-coded factory.

CPU sprint #182 PR 4. The hand-coded factory at
``src/graphs/hardware/models/edge/intel_core_i7_12700k.py`` is the
ground truth that the YAML-loaded model needs to reproduce. This
test compares the two field-by-field across every axis the
downstream consumers (CPUMapper, roofline, energy estimator) read.

Once this test passes, PR 5 (cleanup) can replace the hand-coded
factory with a thin ``load_cpu_resource_model_from_yaml`` call
without changing chip-level numerical output.

**Known parity gaps** -- documented here, accepted as v3 trade-offs:
  - Per-precision peak ops differ by ~5%. The hand-coded model uses
    cluster-specific clocks (P=4.7 GHz, E=3.6 GHz); the loader uses
    a single chip-level clock (the default profile's 4.7 GHz) for
    both clusters because the chip-level Power.thermal_profiles
    carries only a scalar clock_mhz. Per-cluster fabric frequency
    is v4 scope (CPUBlock already has CPUThermalProfile.per_cluster_clock_domain
    but it lives at the block level, not the chip-level Power).
  - Cooling solution string: loader produces "active_fan" (catalog
    YAML); hand-coded had "tower-cooler". This is a deliberate
    improvement (the YAML references the canonical cooling-solution
    catalog) and not asserted by the parity test.
  - fabric_type string: loader produces "avx_vnni" (ISA-named);
    hand-coded had "alder_lake_p_core_avx2" / "gracemont_e_core_avx2"
    (architecture-named). Both are valid; the loader's choice is
    consistent across all CPU vendors.
"""

import pytest

from graphs.hardware.models.datacenter.cpu_yaml_loader import (
    CPUYamlLoaderError,
    load_cpu_resource_model_from_yaml,
)
from graphs.hardware.models.edge.intel_core_i7_12700k import (
    intel_core_i7_12700k_resource_model,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "intel_core_i7_12700k"
LEGACY_NAME = "Intel-Core-i7-12700K"


@pytest.fixture(scope="module")
def hand_coded():
    return intel_core_i7_12700k_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_cpu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name


def test_hardware_type_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.hardware_type == HardwareType.CPU == hand_coded.hardware_type


def test_compute_units_matches(hand_coded, yaml_loaded):
    """Effective cores: 8 P + int(4 E * 0.6) = 10."""
    assert yaml_loaded.compute_units == hand_coded.compute_units == 10


def test_simd_lanes_matches(hand_coded, yaml_loaded):
    """AVX2 -> 8 FP32 lanes."""
    assert yaml_loaded.warp_size == hand_coded.warp_size == 8


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_compute_fabric_count_matches(hand_coded, yaml_loaded):
    """Two fabrics: P-cluster + E-cluster."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_compute_fabric_num_units_match(hand_coded, yaml_loaded):
    """P-cluster has 8 cores, E-cluster has 4 cores."""
    yaml_units = sorted(f.num_units for f in yaml_loaded.compute_fabrics)
    hand_units = sorted(f.num_units for f in hand_coded.compute_fabrics)
    assert yaml_units == hand_units == [4, 8]


def test_compute_fabric_energies_documented_gap(hand_coded, yaml_loaded):
    """The two models source FP32 energy differently:
      - YAML (loader): Intel 7 process node's hp_logic:fp32 entry
        (0.85 pJ), authored from Intel 7 process brief.
      - Hand-coded: get_base_alu_energy(10nm, 'simd_packed') from the
        Horowitz process-node-energy table (1.89 pJ, ~2x higher).
    Neither is wrong; they're from different sources. Real Intel 7
    AVX2 FP32 sits in between (~1.3 pJ per analytical estimate of
    HP_logic + SIMD packing overhead). v4 cleanup can reconcile.

    This test PINS the gap rather than asserting parity, so a future
    YAML edit that bridges the gap won't silently change the test
    (it'll narrow the assertion and fire as a deliberate-update
    reminder)."""
    yaml_p = next(f for f in yaml_loaded.compute_fabrics if f.num_units == 8)
    hand_p = next(f for f in hand_coded.compute_fabrics if f.num_units == 8)
    # Both must be in the right order of magnitude (~1 pJ FP32 at 10nm
    # is the floor; 5 pJ would be wrong).
    assert 0.5e-12 < yaml_p.energy_per_flop_fp32 < 5e-12
    assert 0.5e-12 < hand_p.energy_per_flop_fp32 < 5e-12
    # And document the ratio so PR review surfaces the gap.
    ratio = hand_p.energy_per_flop_fp32 / yaml_p.energy_per_flop_fp32
    assert 1.5 < ratio < 3.0, (
        f"FP32 energy ratio hand/yaml = {ratio:.2f}; expected ~2.2 "
        f"(hand=1.89 pJ Horowitz 10nm simd_packed, yaml=0.85 pJ Intel 7 hp_logic). "
        f"If this ratio has changed, the YAML or process-node entry was edited; "
        f"update this assertion deliberately."
    )


# ---------------------------------------------------------------------------
# Precision profiles -- WIDE tolerance for the per-cluster clock gap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", [
    Precision.FP64, Precision.FP32, Precision.FP16, Precision.INT8, Precision.INT4,
])
def test_precision_present_in_both(hand_coded, yaml_loaded, precision):
    """Both models should expose the same precision set."""
    if precision not in hand_coded.precision_profiles:
        pytest.skip(f"hand-coded doesn't expose {precision.name}")
    assert precision in yaml_loaded.precision_profiles


@pytest.mark.parametrize("precision", [
    Precision.FP64, Precision.FP32, Precision.FP16, Precision.INT8,
])
def test_precision_peak_ops_close(hand_coded, yaml_loaded, precision):
    """Peak ops must be in the same ballpark. WIDE 20% tolerance
    because the loader uses the chip-level default clock (4.7 GHz)
    for both clusters while the hand-coded model uses cluster-specific
    clocks (P=4.7, E=3.6). The E-cluster contribution differs by the
    clock ratio (4.7/3.6 = 1.31). Loader is ~5% higher overall;
    20% is comfortable headroom for the v4 per-cluster-clock fix to
    land within this assertion."""
    if precision not in hand_coded.precision_profiles:
        pytest.skip(f"hand-coded doesn't expose {precision.name}")
    hand_peak = hand_coded.precision_profiles[precision].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[precision].peak_ops_per_sec
    assert yaml_peak == pytest.approx(hand_peak, rel=0.20)


# ---------------------------------------------------------------------------
# Memory hierarchy
# ---------------------------------------------------------------------------

def test_main_memory_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.main_memory == hand_coded.main_memory


def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.peak_bandwidth == pytest.approx(
        hand_coded.peak_bandwidth, rel=0.001
    )


def test_l1_per_unit_matches(hand_coded, yaml_loaded):
    """P-core L1 = 48 KB (the hand-coded model reports the P-core value
    as the per-unit L1; loader picks the PERFORMANCE cluster too)."""
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit


def test_l2_cache_total_matches(hand_coded, yaml_loaded):
    """Convention: l2_cache_total IS the LLC. For Alder Lake that's
    the 25 MB L3 (no distinct chip-shared L2)."""
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total
    assert yaml_loaded.l2_cache_total == 25 * 1024 * 1024


def test_l2_per_unit_matches(hand_coded, yaml_loaded):
    """Physical L2 / effective_cores. Hand-coded: 12 MB / 10. Loader
    computes physical L2 from per-cluster (8 * 1.25 MB P + 2 MB E) =
    12 MB, divided by 10."""
    assert yaml_loaded.l2_cache_per_unit == hand_coded.l2_cache_per_unit


def test_l3_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l3_present is True
    assert hand_coded.l3_present is True
    assert yaml_loaded.l3_cache_total == hand_coded.l3_cache_total
    assert yaml_loaded.l3_cache_total == 25 * 1024 * 1024


def test_coherence_protocol_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.coherence_protocol == hand_coded.coherence_protocol
    assert yaml_loaded.coherence_protocol == "snoopy_mesi"


def test_memory_technology_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.memory_technology == hand_coded.memory_technology == "DDR5"


def test_memory_byte_energies_match(hand_coded, yaml_loaded):
    assert yaml_loaded.memory_read_energy_per_byte_pj == pytest.approx(
        hand_coded.memory_read_energy_per_byte_pj, rel=0.01
    )
    assert yaml_loaded.memory_write_energy_per_byte_pj == pytest.approx(
        hand_coded.memory_write_energy_per_byte_pj, rel=0.01
    )


# ---------------------------------------------------------------------------
# SoC fabric -- CPU specifics
# ---------------------------------------------------------------------------

def test_soc_fabric_topology_matches(hand_coded, yaml_loaded):
    """Both report RING. Loader maps DOUBLE_RING -> RING because the
    graphs Topology enum has no DOUBLE_RING value."""
    from graphs.hardware.fabric_model import Topology
    assert yaml_loaded.soc_fabric.topology == hand_coded.soc_fabric.topology
    assert yaml_loaded.soc_fabric.topology == Topology.RING


def test_soc_fabric_stops_match(hand_coded, yaml_loaded):
    """12 ring stops = 8 P + 4 E."""
    assert yaml_loaded.soc_fabric.controller_count == hand_coded.soc_fabric.controller_count == 12


def test_soc_fabric_bandwidth_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(
        hand_coded.soc_fabric.bisection_bandwidth_gbps, rel=0.001
    )


def test_soc_fabric_latency_and_energy_match(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.hop_latency_ns == pytest.approx(
        hand_coded.soc_fabric.hop_latency_ns, rel=0.001
    )
    assert yaml_loaded.soc_fabric.pj_per_flit_per_hop == pytest.approx(
        hand_coded.soc_fabric.pj_per_flit_per_hop, rel=0.001
    )


# ---------------------------------------------------------------------------
# CPU-specific: SIMD efficiency
# ---------------------------------------------------------------------------

def test_simd_efficiency_matches(hand_coded, yaml_loaded):
    """CPU-specific concept (graphs CPUMapper reads it for
    vectorization friendliness)."""
    assert yaml_loaded.simd_efficiency == hand_coded.simd_efficiency
    # And the exact reference values from the design doc
    assert yaml_loaded.simd_efficiency == {
        "elementwise": 0.95, "matrix": 0.80, "default": 0.70,
    }


# ---------------------------------------------------------------------------
# Thermal profiles
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    """Both expose the 125W-PL1 profile."""
    assert set(yaml_loaded.thermal_operating_points) == \
           set(hand_coded.thermal_operating_points)


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile


def test_thermal_profile_tdp_matches(hand_coded, yaml_loaded):
    for name in hand_coded.thermal_operating_points:
        h = hand_coded.thermal_operating_points[name]
        y = yaml_loaded.thermal_operating_points[name]
        assert y.tdp_watts == h.tdp_watts


# ---------------------------------------------------------------------------
# Scheduler attrs
# ---------------------------------------------------------------------------

def test_scheduler_attrs_match(hand_coded, yaml_loaded):
    assert yaml_loaded.min_occupancy == hand_coded.min_occupancy
    assert yaml_loaded.max_concurrent_kernels == hand_coded.max_concurrent_kernels
    assert yaml_loaded.wave_quantization == hand_coded.wave_quantization


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    with pytest.raises(CPUYamlLoaderError, match="no ComputeProduct with id"):
        load_cpu_resource_model_from_yaml("definitely_not_a_real_cpu")


def test_loader_raises_on_kpu_sku():
    """KPU ComputeProducts have no CPUBlock; loader should reject."""
    with pytest.raises(CPUYamlLoaderError, match="no CPUBlock"):
        load_cpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")


def test_loader_raises_on_gpu_sku():
    """GPU ComputeProducts have no CPUBlock either."""
    with pytest.raises(CPUYamlLoaderError, match="no CPUBlock"):
        load_cpu_resource_model_from_yaml("nvidia_jetson_agx_orin_64gb")
