"""Unit tests for PhysicalSpec dataclass and mapper integration."""

from graphs.hardware.physical_spec import PhysicalSpec
from graphs.hardware.mappers.gpu import (
    create_h100_sxm5_80gb_mapper,
    create_a100_sxm4_80gb_mapper,
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
