"""Contract test: Jetson Thor 128GB factory produces the expected model.

Mirrors ``test_gpu_yaml_loader_jetson_agx_orin_parity.py``. The Thor
factory now wraps ``load_gpu_resource_model_from_yaml`` (analogous to
graphs#181 for AGX Orin), so this file pins the loader's output for
the Thor SKU plus the two factory overlays (BoM cost and per-profile
memory_clock_mhz).

The structural assertions catch loader regressions, YAML drift, and
any future change to the factory's name override or SKU id binding.
The overlay tests pin the data the v2 ComputeProduct schema doesn't
yet carry (BoM, memory_clock_mhz); when the v3 schema absorbs those
fields the overlay-presence tests will fail and force a deliberate
cleanup of the factory.
"""

import pytest

from graphs.hardware.models.automotive.jetson_thor_128gb import (
    jetson_thor_128gb_resource_model,
)
from graphs.hardware.models.edge.gpu_yaml_loader import (
    load_gpu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import HardwareType, Precision


SKU_ID = "nvidia_jetson_agx_thor_128gb"
LEGACY_NAME = "Jetson-Thor-128GB"


@pytest.fixture(scope="module")
def factory_loaded():
    return jetson_thor_128gb_resource_model()


@pytest.fixture(scope="module")
def yaml_only():
    """Loader output WITHOUT the factory's BoM / memory_clock overlays.
    Used to confirm those overlays are factory contributions, not
    loader contributions."""
    return load_gpu_resource_model_from_yaml(
        SKU_ID, name_override=LEGACY_NAME,
    )


# ---------------------------------------------------------------------------
# Identity / SM hierarchy
# ---------------------------------------------------------------------------

def test_name_matches(factory_loaded):
    assert factory_loaded.name == LEGACY_NAME


def test_hardware_type_is_gpu(factory_loaded):
    assert factory_loaded.hardware_type == HardwareType.GPU


def test_sm_hierarchy(factory_loaded):
    """64 Blackwell SMs * 128 CUDA cores = 8192 cores;
    4 Tensor cores/SM * 64 = 256 Tensor cores."""
    rm = factory_loaded
    assert rm.compute_units == 64
    assert rm.cuda_cores_per_sm == 128
    assert rm.tensor_cores_per_sm == 4
    assert rm.compute_units * rm.cuda_cores_per_sm == 8192
    assert rm.compute_units * rm.tensor_cores_per_sm == 256


# ---------------------------------------------------------------------------
# Compute fabrics
# ---------------------------------------------------------------------------

def test_two_compute_fabrics(factory_loaded):
    """CUDA core + Tensor core."""
    fabrics = factory_loaded.compute_fabrics
    assert len(fabrics) == 2
    kinds = sorted(f.fabric_type for f in fabrics)
    assert kinds == ["cuda_core", "tensor_core"]


def test_compute_fabric_chip_total_units(factory_loaded):
    by_kind = {f.fabric_type: f.num_units for f in factory_loaded.compute_fabrics}
    assert by_kind["cuda_core"] == 8192
    assert by_kind["tensor_core"] == 256


# ---------------------------------------------------------------------------
# Memory hierarchy
# ---------------------------------------------------------------------------

def test_memory_subsystem(factory_loaded):
    """LPDDR5X / 128 GB / 273 GB/s; 256 KiB L1 per SM (doubled vs Orin);
    8 MiB shared L2; no L3."""
    rm = factory_loaded
    assert rm.main_memory == 128 * 1024**3
    assert rm.peak_bandwidth == pytest.approx(273e9, rel=0.001)
    assert rm.l1_cache_per_unit == 256 * 1024
    assert rm.l2_cache_total == 8 * 1024**2
    assert rm.l3_present is False
    assert rm.l3_cache_total == 0
    assert rm.memory_technology == "LPDDR5X"


# ---------------------------------------------------------------------------
# SoC fabric: Thor uses MESH_2D (vs AGX Orin's CROSSBAR -- 64 SMs vs 16)
# ---------------------------------------------------------------------------

def test_soc_fabric_is_mesh_2d(factory_loaded):
    from graphs.hardware.fabric_model import Topology
    sf = factory_loaded.soc_fabric
    assert sf.topology == Topology.MESH_2D
    assert sf.controller_count == 64
    assert sf.bisection_bandwidth_gbps == pytest.approx(4096.0, rel=0.001)


# ---------------------------------------------------------------------------
# Thermal profiles: 30W / 60W / 100W (no MAXN unlike AGX Orin)
# ---------------------------------------------------------------------------

def test_thermal_profiles_are_thor_specific(factory_loaded):
    """Thor ships three nvpmodel profiles. Default is 60W."""
    rm = factory_loaded
    assert set(rm.thermal_operating_points) == {"30W", "60W", "100W"}
    assert rm.default_thermal_profile == "60W"


@pytest.mark.parametrize("profile_name,tdp", [
    ("30W", 30.0), ("60W", 60.0), ("100W", 100.0),
])
def test_thermal_profile_tdp(factory_loaded, profile_name, tdp):
    assert factory_loaded.thermal_operating_points[profile_name].tdp_watts == tdp


# ---------------------------------------------------------------------------
# Precision profiles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", [
    Precision.FP32, Precision.FP16, Precision.INT8,
])
def test_precision_present(factory_loaded, precision):
    """Thor's YAML declares fp32/fp16/int8 in multi_precision_alu;
    the loader produces a PrecisionProfile for each."""
    assert precision in factory_loaded.precision_profiles
    pp = factory_loaded.precision_profiles[precision]
    assert pp.peak_ops_per_sec > 0


def test_default_precision(factory_loaded):
    """INT8 is preferred default (gpu_yaml_loader rule)."""
    assert factory_loaded.default_precision == Precision.INT8


# ---------------------------------------------------------------------------
# Scheduler attrs
# ---------------------------------------------------------------------------

def test_scheduler_attrs(factory_loaded):
    rm = factory_loaded
    assert rm.min_occupancy == pytest.approx(0.3)
    assert rm.max_concurrent_kernels == 16
    assert rm.wave_quantization == 4


# ---------------------------------------------------------------------------
# Factory overlays (the only thing the factory adds beyond the loader)
# ---------------------------------------------------------------------------

def test_factory_attaches_bom_cost_profile(factory_loaded):
    """Factory layers a BOMCostProfile after loading because the v2
    schema doesn't carry BoM data. Validation scripts that read
    ``hardware.bom_cost_profile`` would break without this overlay."""
    bom = factory_loaded.bom_cost_profile
    assert bom is not None, "factory must attach BOMCostProfile"
    assert bom.retail_price == pytest.approx(2500.0)
    assert bom.process_node == "4nm"
    assert bom.year == 2025


def test_loader_alone_does_not_attach_bom_cost(yaml_only):
    """Confirms BoM is the FACTORY's contribution, not the loader's.
    When v3 schema gains Market.bom this assertion fires -- delete the
    factory's BoM overlay then."""
    assert yaml_only.bom_cost_profile is None


def test_factory_attaches_memory_clock_to_thermal_profiles(factory_loaded):
    """LPDDR5X-8533 internal clock is 4267 MHz across all three Thor
    nvpmodel profiles (NVIDIA datasheet). KPUThermalProfile shape
    doesn't carry memory_clock_mhz; factory overlays it."""
    for name, profile in factory_loaded.thermal_operating_points.items():
        assert profile.memory_clock_mhz == 4267.0, (
            f"profile {name!r} missing memory_clock_mhz overlay "
            f"(got {profile.memory_clock_mhz})"
        )


def test_loader_alone_does_not_attach_memory_clock(yaml_only):
    """When v3 ThermalProfile generalization absorbs memory_clock_mhz
    this fires -- delete the factory's overlay then."""
    for name, profile in yaml_only.thermal_operating_points.items():
        assert profile.memory_clock_mhz is None


# ---------------------------------------------------------------------------
# Loader error paths (Thor doesn't trigger these but the tests pin
# loader contracts that the AGX Orin parity test also covers; replicated
# here so a Thor-only run still catches loader-level regressions).
# ---------------------------------------------------------------------------

def test_loader_raises_on_unknown_sku():
    from graphs.hardware.models.edge.gpu_yaml_loader import GPUYamlLoaderError
    with pytest.raises(GPUYamlLoaderError, match="no ComputeProduct with id"):
        load_gpu_resource_model_from_yaml("definitely_not_a_real_thor")
