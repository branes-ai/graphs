"""Parity test: YAML-loaded Jetson AGX Orin 64GB matches hand-coded factory.

PR 4 of the GPU sprint scoped at #171. The hand-coded factory at
``src/graphs/hardware/models/edge/jetson_orin_agx_64gb.py`` is the
ground truth that the YAML-loaded model needs to reproduce. This
test compares the two field-by-field across every axis the
downstream consumers (mappers, roofline, energy estimator) read.

Once this test passes, PR 5 (cleanup) can replace the hand-coded
factory with a thin ``load_gpu_resource_model_from_yaml`` call without
changing any chip-level numerical output.
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
