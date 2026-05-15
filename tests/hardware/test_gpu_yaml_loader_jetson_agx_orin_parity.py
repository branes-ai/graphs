"""Contract test: Jetson AGX Orin 64GB factory produces the expected model.

Originally PR 4's parity test, comparing the YAML-loaded model
against the hand-coded factory. PR 5 retired the hand-coded body
and made the factory a thin wrapper over
``load_gpu_resource_model_from_yaml``, so today both ``hand_coded``
and ``yaml_loaded`` fixtures resolve through the same loader.

The structural assertions (SM hierarchy, compute fabrics, memory
hierarchy, SoC fabric, thermal profiles, scheduler attrs) are now
**contract tests on the YAML loader's output** rather than parity
checks. They still catch loader regressions and YAML drift; they
also fail loudly if anyone changes the factory's name override or
substitutes a different SKU id.

The BOM cost preservation test below is specific to PR 5: the BoM
overlay is the only piece of data the factory adds on top of the
loader output (the v2 ComputeProduct schema doesn't carry BoM cost
yet). Removing the overlay would break validation scripts that read
``hardware.bom_cost_profile``.
"""

import pytest

from graphs.hardware.models.edge.gpu_yaml_loader import (
    GPUYamlLoaderError,
    load_gpu_resource_model_from_yaml,
)
from graphs.hardware.models.edge.jetson_orin_agx_64gb import (
    jetson_orin_agx_64gb_resource_model,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "nvidia_jetson_agx_orin_64gb"
LEGACY_NAME = "Jetson-Orin-AGX-64GB"


@pytest.fixture(scope="module")
def hand_coded():
    return jetson_orin_agx_64gb_resource_model()


@pytest.fixture(scope="module")
def yaml_loaded():
    return load_gpu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / shape
# ---------------------------------------------------------------------------

def test_name_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.name == hand_coded.name


def test_hardware_type_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.hardware_type == HardwareType.GPU == hand_coded.hardware_type


def test_sm_hierarchy_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.compute_units == hand_coded.compute_units
    assert yaml_loaded.cuda_cores_per_sm == hand_coded.cuda_cores_per_sm
    assert yaml_loaded.tensor_cores_per_sm == hand_coded.tensor_cores_per_sm
    assert yaml_loaded.warp_size == hand_coded.warp_size


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_compute_fabric_count_matches(hand_coded, yaml_loaded):
    """Both should expose two fabrics: CUDA core + Tensor core."""
    assert len(yaml_loaded.compute_fabrics) == len(hand_coded.compute_fabrics) == 2


def test_compute_fabric_kinds_match(hand_coded, yaml_loaded):
    yaml_kinds = sorted(f.fabric_type for f in yaml_loaded.compute_fabrics)
    hand_kinds = sorted(f.fabric_type for f in hand_coded.compute_fabrics)
    assert yaml_kinds == hand_kinds


def test_compute_fabric_chip_total_units_match(hand_coded, yaml_loaded):
    """2048 CUDA cores + 64 Tensor cores chip-wide."""
    yaml_by_kind = {f.fabric_type: f.num_units for f in yaml_loaded.compute_fabrics}
    hand_by_kind = {f.fabric_type: f.num_units for f in hand_coded.compute_fabrics}
    assert yaml_by_kind == hand_by_kind


def test_compute_fabric_energy_matches(hand_coded, yaml_loaded):
    """CUDA: 1.9 pJ; Tensor: 1.62 pJ."""
    yaml_by_kind = {
        f.fabric_type: f.energy_per_flop_fp32 for f in yaml_loaded.compute_fabrics
    }
    hand_by_kind = {
        f.fabric_type: f.energy_per_flop_fp32 for f in hand_coded.compute_fabrics
    }
    for kind, energy in hand_by_kind.items():
        assert yaml_by_kind[kind] == pytest.approx(energy, rel=0.01)


# ---------------------------------------------------------------------------
# Precision profiles -- the headline performance numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", [
    Precision.FP64, Precision.FP32, Precision.FP16, Precision.INT8,
])
def test_precision_peak_ops_match(hand_coded, yaml_loaded, precision):
    """Peak ops/sec per precision must agree to within 0.5%. The
    hand-coded model rounds to the published TOPS / TFLOPS; the
    loader computes from raw fabric arithmetic. Allow a small
    tolerance for that quantization."""
    if precision not in hand_coded.precision_profiles:
        pytest.skip(f"hand-coded model doesn't expose {precision.name}")
    if precision not in yaml_loaded.precision_profiles:
        pytest.fail(f"YAML-loaded model missing {precision.name}")
    hand_peak = hand_coded.precision_profiles[precision].peak_ops_per_sec
    yaml_peak = yaml_loaded.precision_profiles[precision].peak_ops_per_sec
    assert yaml_peak == pytest.approx(hand_peak, rel=0.005)


@pytest.mark.parametrize("precision", [
    Precision.FP64, Precision.FP32, Precision.FP16, Precision.INT8,
])
def test_precision_metadata_matches(hand_coded, yaml_loaded, precision):
    """Beyond peak ops, the per-precision PrecisionProfile carries
    tensor_core_supported, accumulator_precision, and relative_speedup
    -- downstream consumers (mappers, energy estimator) read all three.
    Pin them so a loader regression that mis-classifies tensor support
    or drops the accumulator hint shows up here, not in some downstream
    estimator's silent miscount."""
    if precision not in hand_coded.precision_profiles:
        pytest.skip(f"hand-coded model doesn't expose {precision.name}")
    hand = hand_coded.precision_profiles[precision]
    yaml = yaml_loaded.precision_profiles[precision]
    assert yaml.tensor_core_supported == hand.tensor_core_supported, (
        f"{precision.name}: tensor_core_supported mismatch "
        f"(yaml={yaml.tensor_core_supported}, hand={hand.tensor_core_supported})"
    )
    assert yaml.accumulator_precision == hand.accumulator_precision, (
        f"{precision.name}: accumulator_precision mismatch"
    )
    assert yaml.relative_speedup == pytest.approx(hand.relative_speedup, rel=0.05)


def test_model_energy_scaling_matches(hand_coded, yaml_loaded):
    """Model-level energy_scaling dict drives the energy estimator's
    per-precision joule-per-op math. Loader now derives it from the
    YAML's per-fabric energy_scaling rather than a hardcoded dict; this
    test pins agreement with the hand-coded chip-wide values."""
    for precision, hand_factor in hand_coded.energy_scaling.items():
        assert precision in yaml_loaded.energy_scaling, (
            f"YAML-loaded model missing energy_scaling[{precision.name}]"
        )
        yaml_factor = yaml_loaded.energy_scaling[precision]
        assert yaml_factor == pytest.approx(hand_factor, rel=0.05), (
            f"energy_scaling[{precision.name}] mismatch: "
            f"yaml={yaml_factor}, hand={hand_factor}"
        )


def test_default_precision_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_precision == hand_coded.default_precision


# ---------------------------------------------------------------------------
# Memory hierarchy
# ---------------------------------------------------------------------------

def test_main_memory_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.main_memory == hand_coded.main_memory


def test_peak_bandwidth_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.peak_bandwidth == pytest.approx(
        hand_coded.peak_bandwidth, rel=0.001
    )


def test_l1_per_sm_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l1_cache_per_unit == hand_coded.l1_cache_per_unit


def test_l2_total_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l2_cache_total == hand_coded.l2_cache_total


def test_l2_per_sm_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.l2_cache_per_unit == hand_coded.l2_cache_per_unit


def test_l3_absent_matches(hand_coded, yaml_loaded):
    """Ampere SoCs don't have L3."""
    assert yaml_loaded.l3_present is False
    assert hand_coded.l3_present is False
    assert yaml_loaded.l3_cache_total == 0
    assert hand_coded.l3_cache_total == 0


def test_storage_kinds_match(hand_coded, yaml_loaded):
    assert yaml_loaded.l1_storage_kind == hand_coded.l1_storage_kind
    assert yaml_loaded.l2_topology == hand_coded.l2_topology


def test_coherence_protocol_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.coherence_protocol == hand_coded.coherence_protocol


def test_memory_technology_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.memory_technology == hand_coded.memory_technology


def test_memory_byte_energies_match(hand_coded, yaml_loaded):
    assert yaml_loaded.memory_read_energy_per_byte_pj == pytest.approx(
        hand_coded.memory_read_energy_per_byte_pj, rel=0.01
    )
    assert yaml_loaded.memory_write_energy_per_byte_pj == pytest.approx(
        hand_coded.memory_write_energy_per_byte_pj, rel=0.01
    )


# ---------------------------------------------------------------------------
# SoC fabric
# ---------------------------------------------------------------------------

def test_soc_fabric_topology_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.topology == hand_coded.soc_fabric.topology


def test_soc_fabric_bandwidth_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.bisection_bandwidth_gbps == pytest.approx(
        hand_coded.soc_fabric.bisection_bandwidth_gbps, rel=0.001
    )


def test_soc_fabric_geometry_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.controller_count == hand_coded.soc_fabric.controller_count
    assert yaml_loaded.soc_fabric.flit_size_bytes == hand_coded.soc_fabric.flit_size_bytes


def test_soc_fabric_latency_and_energy_match(hand_coded, yaml_loaded):
    assert yaml_loaded.soc_fabric.hop_latency_ns == pytest.approx(
        hand_coded.soc_fabric.hop_latency_ns, rel=0.001
    )
    assert yaml_loaded.soc_fabric.pj_per_flit_per_hop == pytest.approx(
        hand_coded.soc_fabric.pj_per_flit_per_hop, rel=0.001
    )


# ---------------------------------------------------------------------------
# Thermal profiles
# ---------------------------------------------------------------------------

def test_thermal_profile_names_match(hand_coded, yaml_loaded):
    assert set(yaml_loaded.thermal_operating_points) == \
           set(hand_coded.thermal_operating_points)


def test_default_thermal_profile_matches(hand_coded, yaml_loaded):
    assert yaml_loaded.default_thermal_profile == hand_coded.default_thermal_profile


@pytest.mark.parametrize("profile_name", ["15W", "30W", "50W", "MAXN"])
def test_thermal_profile_tdp_matches(hand_coded, yaml_loaded, profile_name):
    """The hand-coded model uses suffixed names (e.g., '15W-passive');
    the YAML uses bare names ('15W'). The hand-coded factory stores the
    bare name as the thermal-points dict key, so this comparison works.
    TDP values must agree exactly."""
    yaml_profile = yaml_loaded.thermal_operating_points[profile_name]
    hand_profile = hand_coded.thermal_operating_points[profile_name]
    assert yaml_profile.tdp_watts == hand_profile.tdp_watts


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
    with pytest.raises(GPUYamlLoaderError, match="no ComputeProduct with id"):
        load_gpu_resource_model_from_yaml("definitely_not_a_real_sku")


def test_loader_raises_on_kpu_sku():
    """A KPU ComputeProduct has no GPUBlock; loader should reject."""
    with pytest.raises(GPUYamlLoaderError, match="no GPUBlock"):
        load_gpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")


# ---------------------------------------------------------------------------
# PR 5 specifics: factory's BoM overlay (the only thing the factory adds
# on top of the YAML loader's output)
# ---------------------------------------------------------------------------

def test_factory_attaches_bom_cost_profile(hand_coded):
    """The thin factory layers a BOMCostProfile onto the YAML-loaded
    model because the v2 ComputeProduct schema doesn't carry BoM data.
    Removing this overlay would break validation scripts that read
    ``hardware.bom_cost_profile``."""
    bom = hand_coded.bom_cost_profile
    assert bom is not None, "factory must attach BOMCostProfile (PR 5 overlay)"
    # Spot-check the headline numbers; full BOMCostProfile content is
    # AGX Orin specific and cited in the factory module.
    assert bom.retail_price == pytest.approx(899.0)
    assert bom.process_node == "8nm"


def test_factory_attaches_memory_clock_to_thermal_profiles(hand_coded):
    """Orin AGX runs LPDDR5-6400 at 3200 MHz internal across all four
    nvpmodel profiles. The chip-level KPUThermalProfile shape doesn't
    carry memory_clock_mhz; the factory layers it on after loading.
    Without this overlay the cli/list_hardware_resources Phase 4
    tests fail."""
    for name, profile in hand_coded.thermal_operating_points.items():
        assert profile.memory_clock_mhz == 3200.0, (
            f"profile {name!r} missing memory_clock_mhz overlay "
            f"(got {profile.memory_clock_mhz})"
        )


def test_loader_alone_does_not_attach_memory_clock(yaml_loaded):
    """Confirms memory_clock_mhz is the FACTORY's contribution. The
    v3 schema generalization (ThermalProfile w/ optional memory clock)
    will move it into the YAML; this assertion will fire then and force
    a deliberate update -- delete the overlay and this guard."""
    for name, profile in yaml_loaded.thermal_operating_points.items():
        assert profile.memory_clock_mhz is None, (
            f"loader populated memory_clock_mhz on profile {name!r} "
            f"(expected None until v3 schema). The factory's overlay "
            f"can now be deleted."
        )


def test_loader_alone_does_not_attach_bom_cost(yaml_loaded):
    """Confirms the BoM is the FACTORY's contribution, not the loader's.
    If the v3 schema PR adds Market.bom and the loader starts populating
    it, this test will fail and force a deliberate update -- at which
    point the factory's overlay can be deleted."""
    assert yaml_loaded.bom_cost_profile is None, (
        "loader is not expected to attach BOMCostProfile; the v2 "
        "ComputeProduct schema doesn't carry BoM data. If this assertion "
        "fails, the YAML schema gained a Market.bom field; update the "
        "factory to drop its BoM overlay accordingly."
    )
