"""Phase 4a tests: KPU YAML loader.

Verifies the loader produces a HardwareResourceModel that:
  - Has the right structural shape (compute_units, fabrics, profiles).
  - Matches the YAML's authoritative numbers (tile mix, performance).
  - Is consumable by the existing KPUMapper without modification.

These tests are *additive* -- they don't replace the existing
``test_kpu_t64_precision_profiles.py`` etc. that exercise the
hand-coded ``kpu_t64_resource_model()`` factories. Phase 4b's job is
to swap those factories over once we're confident the loader
reproduces the contractually-required values.
"""

from __future__ import annotations

import pytest

from embodied_schemas import load_kpus, load_process_nodes

from graphs.hardware.mappers.accelerators.kpu import KPUMapper
from graphs.hardware.models.accelerators.kpu_yaml_loader import (
    KPUYamlLoaderError,
    load_kpu_resource_model_from_yaml,
)
from graphs.hardware.resource_model import (
    HardwareType,
    Precision,
    ThermalOperatingPoint,
)


@pytest.fixture(scope="module")
def catalogs():
    return {
        "kpus": load_kpus(),
        "process_nodes": load_process_nodes(),
    }


SKU_IDS = [
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
]


# ---------------------------------------------------------------------------
# Structural shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_returns_kpu_hardware_type(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    assert m.hardware_type == HardwareType.KPU


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_compute_units_matches_total_tiles(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    assert m.compute_units == sku.kpu_architecture.total_tiles


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_one_fabric_per_tile_class(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    assert len(m.compute_fabrics) == len(sku.kpu_architecture.tiles)


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_thermal_profiles_round_trip(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    assert m.default_thermal_profile == sku.power.default_thermal_profile
    assert set(m.thermal_operating_points) == \
        {p.name for p in sku.power.thermal_profiles}
    for name, top in m.thermal_operating_points.items():
        assert isinstance(top, ThermalOperatingPoint)
        ya_profile = next(
            p for p in sku.power.thermal_profiles if p.name == name
        )
        assert top.tdp_watts == ya_profile.tdp_watts


# ---------------------------------------------------------------------------
# Performance equivalence with YAML
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_int8_peak_matches_yaml(sku_id, catalogs):
    """Loader's INT8 peak must match the YAML's performance.int8_tops
    within rounding (the YAML was authored to match fabric x clock; the
    loader recomputes from the same fabric x clock at the default
    profile clock)."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    int8_profile = m.precision_profiles.get(Precision.INT8)
    assert int8_profile is not None
    loader_int8_tops = int8_profile.peak_ops_per_sec / 1e12
    yaml_int8_tops = sku.performance.int8_tops
    assert abs(loader_int8_tops - yaml_int8_tops) / yaml_int8_tops <= 0.02


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_bf16_peak_matches_yaml(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    bf16_profile = m.precision_profiles.get(Precision.BF16)
    if sku.performance.bf16_tflops <= 0:
        return
    assert bf16_profile is not None
    loader_bf16_tflops = bf16_profile.peak_ops_per_sec / 1e12
    yaml_bf16_tflops = sku.performance.bf16_tflops
    assert abs(loader_bf16_tflops - yaml_bf16_tflops) / yaml_bf16_tflops <= 0.02


# ---------------------------------------------------------------------------
# Memory + cache mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_memory_bandwidth_matches_yaml(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    expected = sku.kpu_architecture.memory.memory_bandwidth_gbps * 1e9
    assert m.peak_bandwidth == expected


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_main_memory_matches_yaml(sku_id, catalogs):
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    expected = int(sku.kpu_architecture.memory.memory_size_gb * 1024**3)
    assert m.main_memory == expected


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_l1_per_tile_is_l3_kib_per_tile(sku_id, catalogs):
    """Convention: HardwareResourceModel.l1_cache_per_unit on KPUs maps
    to YAML's per-tile L3 scratchpad."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    expected = sku.kpu_architecture.memory.l3_kib_per_tile * 1024
    assert m.l1_cache_per_unit == expected


# ---------------------------------------------------------------------------
# KPUMapper compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_output_is_consumable_by_kpu_mapper(sku_id, catalogs):
    """KPUMapper must accept the loader-produced HardwareResourceModel
    and expose the expected derived fields (num_tiles, scratchpad,
    on-chip totals)."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    mapper = KPUMapper(m)
    assert mapper.num_tiles == m.compute_units
    assert mapper.scratchpad_per_tile == m.l1_cache_per_unit
    assert mapper.l2_cache_total == m.l2_cache_total
    assert mapper.total_on_chip_bytes > 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_loader_unknown_base_id_raises(catalogs):
    with pytest.raises(KPUYamlLoaderError, match="no KPU SKU"):
        load_kpu_resource_model_from_yaml("not_a_real_sku", **catalogs)


def test_loader_unresolved_process_node_raises(catalogs):
    """Hand-craft a SKU pointing at a non-existent process node."""
    real = catalogs["kpus"]["stillwater_kpu_t256"]
    bad_sku = real.model_copy(update={"process_node_id": "nonexistent"})
    with pytest.raises(KPUYamlLoaderError, match="does not resolve"):
        load_kpu_resource_model_from_yaml(
            bad_sku.id,
            kpus={bad_sku.id: bad_sku},
            process_nodes=catalogs["process_nodes"],
        )
