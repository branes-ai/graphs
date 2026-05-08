"""Unit tests for PhysicalSpec dataclass and mapper integration."""

from graphs.hardware.physical_spec import PhysicalSpec
from graphs.hardware.mappers.gpu import (
    create_h100_sxm5_80gb_mapper,
    create_a100_sxm4_80gb_mapper,
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
    create_jetson_orin_nx_16gb_mapper,
    create_jetson_thor_128gb_mapper,
)


class TestPhysicalSpec:
    def test_all_fields_optional(self):
        spec = PhysicalSpec()
        assert spec.die_size_mm2 is None
        assert spec.transistors_billion is None
        assert spec.process_node_nm is None
        assert spec.foundry is None
        assert spec.architecture is None
        assert spec.num_dies == 1
        assert spec.is_chiplet is False
        assert spec.extras == {}

    def test_density_computed_when_both_known(self):
        spec = PhysicalSpec(die_size_mm2=814.0, transistors_billion=80.0)
        # 80B / 814mm^2 = 98.28 Mtx/mm^2
        assert abs(spec.transistor_density_mtx_mm2 - 98.28) < 0.1

    def test_density_none_when_die_unknown(self):
        spec = PhysicalSpec(transistors_billion=80.0)
        assert spec.transistor_density_mtx_mm2 is None

    def test_density_none_when_transistors_unknown(self):
        spec = PhysicalSpec(die_size_mm2=814.0)
        assert spec.transistor_density_mtx_mm2 is None

    def test_density_handles_zero_die_size(self):
        spec = PhysicalSpec(die_size_mm2=0.0, transistors_billion=80.0)
        assert spec.transistor_density_mtx_mm2 is None

    def test_density_rejects_negative_die_size(self):
        spec = PhysicalSpec(die_size_mm2=-1.0, transistors_billion=80.0)
        assert spec.transistor_density_mtx_mm2 is None

    def test_density_rejects_negative_transistors(self):
        spec = PhysicalSpec(die_size_mm2=814.0, transistors_billion=-1.0)
        assert spec.transistor_density_mtx_mm2 is None

    def test_density_rejects_zero_transistors(self):
        spec = PhysicalSpec(die_size_mm2=814.0, transistors_billion=0.0)
        assert spec.transistor_density_mtx_mm2 is None

    def test_to_dict_serializes_cleanly(self):
        spec = PhysicalSpec(
            die_size_mm2=814.0,
            transistors_billion=80.0,
            architecture="Hopper",
        )
        d = spec.to_dict()
        assert d["die_size_mm2"] == 814.0
        assert d["transistors_billion"] == 80.0
        assert d["architecture"] == "Hopper"
        assert d["num_dies"] == 1
        assert d["extras"] == {}

    def test_extras_accepts_arbitrary_keys(self):
        spec = PhysicalSpec(extras={"chiplet_count": 4, "interposer_type": "CoWoS"})
        assert spec.extras["chiplet_count"] == 4
        assert spec.extras["interposer_type"] == "CoWoS"


class TestMapperPhysicalSpecIntegration:
    """Verify the PhysicalSpec attribute is wired onto every mapper, that
    populated factories surface real data, and that unpopulated factories
    return None without crashing."""

    def test_h100_sxm5_factory_populates_physical_spec(self):
        mapper = create_h100_sxm5_80gb_mapper()
        assert mapper.physical_spec is not None
        assert mapper.physical_spec.die_size_mm2 == 814.0
        assert mapper.physical_spec.transistors_billion == 80.0
        assert mapper.physical_spec.process_node_nm == 4
        assert mapper.physical_spec.foundry == "tsmc"
        assert mapper.physical_spec.architecture == "Hopper"
        assert mapper.physical_spec.source.startswith("embodied-schemas:")

    def test_unpopulated_factory_returns_none_physical_spec(self):
        # A100 factory hasn't been backfilled yet -- physical_spec should be
        # None, not raise. This guards graceful degradation for the long tail
        # of mappers that haven't had die specs populated.
        mapper = create_a100_sxm4_80gb_mapper()
        assert mapper.physical_spec is None

    def test_physical_spec_does_not_affect_mapping_path(self):
        # The mapping path reads resource_model, not physical_spec. Confirm
        # both populated and unpopulated mappers expose the same operational
        # surface area.
        h100 = create_h100_sxm5_80gb_mapper()
        a100 = create_a100_sxm4_80gb_mapper()
        assert h100.resource_model is not None
        assert a100.resource_model is not None
        assert h100.resource_model.compute_units == 132
        assert a100.resource_model.compute_units == 108


class TestJetsonPhysicalSpecs:
    """Verify the four Jetson SKUs are populated. Orin AGX/NX/Nano share the
    GA10B die so chip-level fields (die_size, transistors, process) match
    across all three; only module-level fields (launch info) differ. Thor is
    a separate generation -- die details not publicly disclosed yet."""

    def test_orin_agx_chip_fields(self):
        spec = create_jetson_orin_agx_64gb_mapper().physical_spec
        assert spec is not None
        assert spec.die_size_mm2 == 455.0
        assert spec.transistors_billion == 17.0
        assert spec.process_node_nm == 8
        assert spec.foundry == "samsung"
        assert spec.architecture == "Ampere"
        assert spec.launch_date == "2022-11-08"
        assert spec.launch_msrp_usd == 1999.0

    def test_orin_nx_chip_fields_match_agx(self):
        # NX uses the same GA10B die as AGX -- chip-level fields identical
        agx = create_jetson_orin_agx_64gb_mapper().physical_spec
        nx = create_jetson_orin_nx_16gb_mapper().physical_spec
        assert nx.die_size_mm2 == agx.die_size_mm2
        assert nx.transistors_billion == agx.transistors_billion
        assert nx.process_node_nm == agx.process_node_nm
        assert nx.foundry == agx.foundry
        # But module-level fields differ
        assert nx.launch_date != agx.launch_date
        assert nx.launch_msrp_usd != agx.launch_msrp_usd

    def test_orin_nano_chip_fields_match_agx(self):
        agx = create_jetson_orin_agx_64gb_mapper().physical_spec
        nano = create_jetson_orin_nano_8gb_mapper().physical_spec
        assert nano.die_size_mm2 == agx.die_size_mm2
        assert nano.transistors_billion == agx.transistors_billion
        assert nano.process_node_nm == agx.process_node_nm
        # Module-level
        assert nano.launch_date == "2023-03-22"
        assert nano.launch_msrp_usd == 499.0

    def test_thor_partial_population(self):
        # Thor's die size and transistor count are not publicly disclosed,
        # so they remain None. The rest of the chip-level fields plus
        # module info are populated.
        spec = create_jetson_thor_128gb_mapper().physical_spec
        assert spec is not None
        assert spec.die_size_mm2 is None
        assert spec.transistors_billion is None
        assert spec.transistor_density_mtx_mm2 is None  # depends on die + transistors
        assert spec.process_node_nm == 4
        assert spec.foundry == "tsmc"
        assert spec.architecture == "Blackwell"
        assert spec.launch_date == "2025-08-25"
        assert spec.launch_msrp_usd == 2999.0

    def test_orin_density_is_consistent_across_skus(self):
        # All three Orin variants are the same die -> identical density.
        agx = create_jetson_orin_agx_64gb_mapper().physical_spec
        nx = create_jetson_orin_nx_16gb_mapper().physical_spec
        nano = create_jetson_orin_nano_8gb_mapper().physical_spec
        d_agx = agx.transistor_density_mtx_mm2
        assert d_agx is not None
        # 17B / 455 mm^2 = ~37.4 Mtx/mm^2 (Samsung 8nm density floor)
        assert abs(d_agx - 37.4) < 0.5
        assert nx.transistor_density_mtx_mm2 == d_agx
        assert nano.transistor_density_mtx_mm2 == d_agx


class TestMemoryFields:
    """Phase 3 of #136: memory_type + memory_bus_width_bits on PhysicalSpec.

    Anchored to the embodied-schemas YAMLs (data/gpus/<vendor>/*.yaml,
    ``memory:`` block). These are silicon-bin / module SKU constants --
    they don't change per nvpmodel power profile. Per-profile memory
    CLOCK is a separate field on ThermalOperatingPoint (Phase 4).
    """

    def test_default_physical_spec_has_no_memory_data(self):
        # Backward compat: the new fields default to None so unpopulated
        # PhysicalSpec instances and existing call sites don't need updates.
        spec = PhysicalSpec()
        assert spec.memory_type is None
        assert spec.memory_bus_width_bits is None

    def test_h100_memory_populated(self):
        spec = create_h100_sxm5_80gb_mapper().physical_spec
        assert spec.memory_type == "hbm3"
        assert spec.memory_bus_width_bits == 5120

    def test_orin_agx_memory_populated(self):
        spec = create_jetson_orin_agx_64gb_mapper().physical_spec
        assert spec.memory_type == "lpddr5"
        assert spec.memory_bus_width_bits == 256

    def test_orin_nx_memory_populated(self):
        spec = create_jetson_orin_nx_16gb_mapper().physical_spec
        assert spec.memory_type == "lpddr5"
        assert spec.memory_bus_width_bits == 128

    def test_orin_nano_memory_populated(self):
        # Orin Nano is 128-bit LPDDR5. Verified via bandwidth math:
        #   68 GB/s / 4.267 GT/s (LPDDR5-4267) = 128 bits
        # The embodied-schemas YAML (orin_nano_gpu_8gb_lpddr5.yaml)
        # incorrectly lists ``memory_bus_bits: 64`` despite also listing
        # ``memory_bandwidth_gbps: 68.0`` (which would be impossible at
        # 64 bits and the 4.267 GT/s rate). The 128-bit value is correct
        # per NVIDIA's Orin Nano datasheet and matches the Orin NX
        # 16GB SKU's bus width.
        spec = create_jetson_orin_nano_8gb_mapper().physical_spec
        assert spec.memory_type == "lpddr5"
        assert spec.memory_bus_width_bits == 128

    def test_thor_memory_populated(self):
        # Thor is 256-bit LPDDR5X. Confirmed by NVIDIA's Jetson Thor
        # announcement blog ("256-bit LPDDR5X, 273 GB/s") and verified
        # via bandwidth math: 273 GB/s / 8.533 GT/s (LPDDR5X-8533) = 256
        # bits. The embodied-schemas YAML (thor_gpu_128gb_lpddr5x.yaml)
        # incorrectly lists ``memory_bus_bits: 512`` despite also listing
        # ``memory_bandwidth_gbps: 273.0`` (which would be impossible at
        # 512 bits and any reasonable LPDDR5X rate).
        spec = create_jetson_thor_128gb_mapper().physical_spec
        assert spec.memory_type == "lpddr5x"
        assert spec.memory_bus_width_bits == 256

    def test_orin_family_bus_widths_match_bandwidth_spec(self):
        # All three Orin variants are LPDDR5. AGX has the widest bus
        # (256-bit) for the highest sustained bandwidth; NX and Nano
        # share the same 128-bit width but different sustained DRAM
        # rates yield different headline bandwidths (NX 102 GB/s vs
        # Nano 68 GB/s on original 2023 silicon).
        agx = create_jetson_orin_agx_64gb_mapper().physical_spec
        nx = create_jetson_orin_nx_16gb_mapper().physical_spec
        nano = create_jetson_orin_nano_8gb_mapper().physical_spec
        # Same memory type across the family
        assert agx.memory_type == nx.memory_type == nano.memory_type == "lpddr5"
        # AGX is wider than the rest; NX and Nano share 128-bit
        assert agx.memory_bus_width_bits == 256
        assert nx.memory_bus_width_bits == 128
        assert nano.memory_bus_width_bits == 128
