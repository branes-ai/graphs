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
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
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
    real = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    bad_sku = real.model_copy(update={"process_node_id": "nonexistent"})
    with pytest.raises(KPUYamlLoaderError, match="does not resolve"):
        load_kpu_resource_model_from_yaml(
            bad_sku.id,
            kpus={bad_sku.id: bad_sku},
            process_nodes=catalogs["process_nodes"],
        )


# ---------------------------------------------------------------------------
# Phase 4b PR 2: tile_energy_model + soc_fabric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_populates_tile_energy_model(sku_id, catalogs):
    """The loader must attach a KPUTileEnergyModel; the existing factory
    factories also do this, so analyzers expect it to be present."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    assert m.tile_energy_model is not None


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_tile_energy_model_architectural_shape_matches_yaml(sku_id, catalogs):
    """num_tiles, tile_mesh_dimensions, l1/l2/l3 sizes, dram bandwidth,
    clock are all direct reads from the SKU YAML."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    arch = sku.kpu_architecture
    tem = m.tile_energy_model
    assert tem.num_tiles == arch.total_tiles
    assert tem.tile_mesh_dimensions == (arch.noc.mesh_rows, arch.noc.mesh_cols)
    assert tem.dram_bandwidth_gb_s == arch.memory.memory_bandwidth_gbps
    assert tem.l3_size_per_tile == arch.memory.l3_kib_per_tile * 1024
    assert tem.l2_size_per_tile == arch.memory.l2_kib_per_tile * 1024
    assert tem.l1_size_per_pe == arch.memory.l1_kib_per_pe * 1024
    # Default profile clock
    default = next(
        p for p in sku.power.thermal_profiles
        if p.name == sku.power.default_thermal_profile
    )
    assert tem.clock_frequency_hz == default.clock_mhz * 1e6


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_tile_energy_model_pes_per_tile_uses_dominant_class(sku_id, catalogs):
    """pes_per_tile reads from the tile class with the largest num_tiles
    (the dominant compute class). For all 4 Stillwater SKUs this is the
    INT8-primary class."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    dominant = max(sku.kpu_architecture.tiles, key=lambda t: t.num_tiles)
    expected = dominant.pe_array_rows * dominant.pe_array_cols
    assert m.tile_energy_model.pes_per_tile == expected


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_tile_energy_model_mac_energies_in_plausible_range(sku_id, catalogs):
    """MAC energies must land in the same range the existing
    test_kpu_*_gemm tests expect for energy_per_mac_j (0.3-1.5 pJ for
    16 nm, scaling down at advanced nodes). The loader pulls from the
    process node's energy_per_op_pj table."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    tem = m.tile_energy_model
    # All four current SKUs have these; relaxed plausibility ranges.
    assert 0.05e-12 <= tem.mac_energy_int8 <= 5e-12, \
        f"INT8 MAC energy {tem.mac_energy_int8 * 1e12:.3f} pJ implausible"
    assert 0.1e-12 <= tem.mac_energy_bf16 <= 10e-12, \
        f"BF16 MAC energy {tem.mac_energy_bf16 * 1e12:.3f} pJ implausible"
    assert 0.1e-12 <= tem.mac_energy_fp32 <= 20e-12, \
        f"FP32 MAC energy {tem.mac_energy_fp32 * 1e12:.3f} pJ implausible"
    # FP32 should be at least as expensive as BF16 which is at least as
    # expensive as INT8 (datapath width ratios).
    assert tem.mac_energy_int8 <= tem.mac_energy_bf16 <= tem.mac_energy_fp32


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_tile_energy_model_dram_phy_energy_per_byte(sku_id, catalogs):
    """DRAM read energy comes from the per-memory-type table (LPDDR5
    = 10 pJ/byte, HBM3 = 6 pJ/byte, etc.); write is 1.2x read."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    tem = m.tile_energy_model
    # Read >= write is wrong; write is heavier.
    assert tem.dram_write_energy_per_byte > tem.dram_read_energy_per_byte
    # Both in plausible 4-15 pJ/byte range.
    assert 4e-12 <= tem.dram_read_energy_per_byte <= 15e-12
    assert 4e-12 <= tem.dram_write_energy_per_byte <= 18e-12


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_tile_energy_model_sram_hierarchy_inversely_ordered(sku_id, catalogs):
    """L1 <= L2 <= L3 <= DRAM in per-byte energy (closer to PE = cheaper)."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    tem = m.tile_energy_model
    assert tem.l1_read_energy_per_byte < tem.l2_read_energy_per_byte
    assert tem.l2_read_energy_per_byte < tem.l3_read_energy_per_byte
    assert tem.l3_read_energy_per_byte < tem.dram_read_energy_per_byte


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_populates_soc_fabric(sku_id, catalogs):
    """The loader must attach a SoCFabricModel matching the YAML's
    kpu_architecture.noc declaration."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    assert m.soc_fabric is not None


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_soc_fabric_topology_and_dimensions_match_yaml(sku_id, catalogs):
    """topology, mesh_dimensions, flit_size_bytes, controller_count,
    bisection_bandwidth_gbps -- all direct reads from
    kpu_architecture.noc + total_tiles."""
    from graphs.hardware.fabric_model import Topology
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    noc = sku.kpu_architecture.noc
    fab = m.soc_fabric
    # All 4 Stillwater SKUs use mesh_2d
    assert fab.topology == Topology.MESH_2D
    assert fab.low_confidence is False
    assert fab.mesh_dimensions == (noc.mesh_rows, noc.mesh_cols)
    assert fab.flit_size_bytes == noc.flit_bytes
    assert fab.controller_count == sku.kpu_architecture.total_tiles
    if noc.bisection_bandwidth_gbps is not None:
        assert fab.bisection_bandwidth_gbps == noc.bisection_bandwidth_gbps


def test_soc_fabric_unknown_topology_marks_low_confidence(catalogs):
    """Unknown topology strings should produce Topology.UNKNOWN with
    low_confidence=True instead of crashing -- defensive behavior so
    new YAML topology values land cleanly even before the enum map
    is updated."""
    from graphs.hardware.fabric_model import Topology
    real = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    bad_arch = real.kpu_architecture.model_copy(
        update={"noc": real.kpu_architecture.noc.model_copy(
            update={"topology": "future_topology_3d_torus"}
        )}
    )
    bad_sku = real.model_copy(update={"kpu_architecture": bad_arch})
    m = load_kpu_resource_model_from_yaml(
        bad_sku.id,
        kpus={bad_sku.id: bad_sku},
        process_nodes=catalogs["process_nodes"],
    )
    assert m.soc_fabric.topology == Topology.UNKNOWN
    assert m.soc_fabric.low_confidence is True


def test_soc_fabric_lossy_torus_mapping_marks_low_confidence(catalogs):
    """``torus_2d`` is approximated as ``Topology.MESH_2D`` because the
    Topology enum has no TORUS_2D value. Mark this lossy mapping with
    ``low_confidence=True`` so consumers (hop-count, bisection-bandwidth
    formulas) see the approximation."""
    from graphs.hardware.fabric_model import Topology
    real = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    arch = real.kpu_architecture.model_copy(
        update={"noc": real.kpu_architecture.noc.model_copy(
            update={"topology": "torus_2d"}
        )}
    )
    bad_sku = real.model_copy(update={"kpu_architecture": arch})
    m = load_kpu_resource_model_from_yaml(
        bad_sku.id,
        kpus={bad_sku.id: bad_sku},
        process_nodes=catalogs["process_nodes"],
    )
    # Maps to MESH_2D (the closest available enum value)...
    assert m.soc_fabric.topology == Topology.MESH_2D
    # ...but flagged as approximation.
    assert m.soc_fabric.low_confidence is True


# ---------------------------------------------------------------------------
# Phase 4b PR 3: per-(profile, precision) efficiency / tile_utilization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_reads_per_precision_efficiency_from_yaml(sku_id, catalogs):
    """When KPUThermalProfile.efficiency_factor_by_precision is set in
    the YAML, the loader must propagate it to the matching
    PerformanceCharacteristics. Phase 4b PR 3 backfilled the four
    SKU YAMLs with values from the corresponding factories."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    for profile in sku.power.thermal_profiles:
        if not profile.efficiency_factor_by_precision:
            continue
        top = m.thermal_operating_points[profile.name]
        for precision_name, expected_eff in profile.efficiency_factor_by_precision.items():
            precision = next(
                (p for p in Precision if p.value == precision_name), None
            )
            if precision is None or precision not in top.performance_specs:
                continue
            actual = top.performance_specs[precision].efficiency_factor
            assert actual == expected_eff, (
                f"{sku_id} {profile.name} {precision_name}: "
                f"loader produced efficiency_factor={actual} but YAML says {expected_eff}"
            )


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_reads_per_precision_tile_utilization_from_yaml(sku_id, catalogs):
    """Same as efficiency but for tile_utilization."""
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    sku = catalogs["kpus"][sku_id]
    for profile in sku.power.thermal_profiles:
        if not profile.tile_utilization_by_precision:
            continue
        top = m.thermal_operating_points[profile.name]
        for precision_name, expected_util in profile.tile_utilization_by_precision.items():
            precision = next(
                (p for p in Precision if p.value == precision_name), None
            )
            if precision is None or precision not in top.performance_specs:
                continue
            actual = top.performance_specs[precision].tile_utilization
            assert actual == expected_util, (
                f"{sku_id} {profile.name} {precision_name}: "
                f"loader produced tile_utilization={actual} but YAML says {expected_util}"
            )


def test_loader_falls_back_to_placeholder_when_yaml_omits_efficiency(catalogs):
    """A YAML thermal_profile without
    ``efficiency_factor_by_precision`` should produce the historical
    flat placeholder (0.70 / 0.95). This is the backward-compat path
    for external user YAMLs that haven't backfilled."""
    real = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    profiles = [
        p.model_copy(
            update={
                "efficiency_factor_by_precision": None,
                "tile_utilization_by_precision": None,
            }
        )
        for p in real.power.thermal_profiles
    ]
    bare_power = real.power.model_copy(update={"thermal_profiles": profiles})
    bare_sku = real.model_copy(update={"power": bare_power})
    m = load_kpu_resource_model_from_yaml(
        bare_sku.id,
        kpus={bare_sku.id: bare_sku},
        process_nodes=catalogs["process_nodes"],
    )
    # All profiles should now use the placeholder.
    for top in m.thermal_operating_points.values():
        for ps in top.performance_specs.values():
            assert ps.efficiency_factor == 0.70
            assert ps.tile_utilization == 0.95


def test_loader_efficiency_increases_with_tdp_per_yaml(catalogs):
    """Sanity check on the catalog values: across the catalog, INT8
    efficiency_factor should be non-decreasing as TDP rises within
    a SKU (more thermal headroom -> less DVFS throttling -> higher
    sustained efficiency)."""
    for sku_id in SKU_IDS:
        m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
        # Sort profiles by TDP and check INT8 efficiency is monotonic.
        sku = catalogs["kpus"][sku_id]
        sorted_profiles = sorted(
            sku.power.thermal_profiles, key=lambda p: p.tdp_watts
        )
        prev_eff = 0.0
        for profile in sorted_profiles:
            top = m.thermal_operating_points[profile.name]
            int8_eff = top.performance_specs[Precision.INT8].efficiency_factor
            assert int8_eff >= prev_eff, (
                f"{sku_id} INT8 efficiency at {profile.name} "
                f"({int8_eff}) is lower than at the previous TDP "
                f"({prev_eff}) -- expected monotonic per Phase 4b PR 3 backfill"
            )
            prev_eff = int8_eff


def test_tile_energy_model_mac_fallback_is_node_scaled(catalogs):
    """When a process node lacks ``balanced_logic:<precision>`` entries,
    the MAC fallback must scale with ``node.node_nm`` via
    ``get_base_alu_energy``, not collapse to a fixed constant. Lower
    nm -> lower fallback MAC energy."""
    from embodied_schemas.process_node import CircuitClass
    real = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    n16_node = catalogs["process_nodes"]["tsmc_n16"]
    # Strip every balanced_logic energy entry to force the fallback path.
    sparse_n16 = n16_node.model_copy(update={
        "energy_per_op_pj": {
            k: v for k, v in n16_node.energy_per_op_pj.items()
            if not k.startswith(f"{CircuitClass.BALANCED_LOGIC.value}:")
        }
    })
    m_n16 = load_kpu_resource_model_from_yaml(
        real.id,
        kpus={real.id: real},
        process_nodes={real.process_node_id: sparse_n16},
    )

    # Same SKU, but pretend it's on TSMC N5 (much smaller node).
    n5_node = catalogs["process_nodes"]["tsmc_n5"]
    sparse_n5 = n5_node.model_copy(update={
        "id": "tsmc_n16",  # rebrand so the SKU's process_node_id resolves
        "energy_per_op_pj": {
            k: v for k, v in n5_node.energy_per_op_pj.items()
            if not k.startswith(f"{CircuitClass.BALANCED_LOGIC.value}:")
        }
    })
    m_n5 = load_kpu_resource_model_from_yaml(
        real.id,
        kpus={real.id: real},
        process_nodes={real.process_node_id: sparse_n5},
    )

    # N5 is a smaller node than N16 -> fallback MAC energy must be lower.
    assert m_n5.tile_energy_model.mac_energy_int8 < m_n16.tile_energy_model.mac_energy_int8
    assert m_n5.tile_energy_model.mac_energy_bf16 < m_n16.tile_energy_model.mac_energy_bf16
    assert m_n5.tile_energy_model.mac_energy_fp32 < m_n16.tile_energy_model.mac_energy_fp32
    # Sanity: ordering INT8 <= BF16 <= FP32 still holds in fallback path.
    tem = m_n16.tile_energy_model
    assert tem.mac_energy_int8 <= tem.mac_energy_bf16 <= tem.mac_energy_fp32


@pytest.mark.parametrize("sku_id", SKU_IDS)
def test_loader_output_remains_kpu_mapper_compatible(sku_id, catalogs):
    """The KPUMapper energy paths use ``mapper.tile_energy_model``;
    confirm the loader-populated model passes through cleanly."""
    from graphs.hardware.mappers.accelerators.kpu import KPUMapper
    m = load_kpu_resource_model_from_yaml(sku_id, **catalogs)
    mapper = KPUMapper(m)
    # The mapper reads l1_cache_per_unit, l2_cache_total, num_tiles --
    # verify they're consistent with the loaded model.
    assert mapper.num_tiles == m.tile_energy_model.num_tiles
    assert mapper.scratchpad_per_tile == m.l1_cache_per_unit
