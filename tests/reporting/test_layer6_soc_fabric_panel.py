"""Self-consistency tests for the M6 Layer 6 (SoC fabric) panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.fabric_model import SoCFabricModel, Topology
from graphs.reporting.layer_panels import (
    build_layer6_soc_fabric_panel,
    cross_sku_layer6_chart,
    resolve_sku_resource_model,
)


REQUIRED_SKUS = [
    "jetson_orin_agx_64gb",
    "intel_core_i7_12700k",
    "ryzen_9_8945hs",
    "kpu_t64",
    "kpu_t128",
    "kpu_t256",
    "coral_edge_tpu",
]
OPTIONAL_SKUS = ["hailo8", "hailo10h"]
TARGET_SKUS = REQUIRED_SKUS + OPTIONAL_SKUS

# Architectural partitioning
RING_SKUS = ["intel_core_i7_12700k", "ryzen_9_8945hs"]
CROSSBAR_SKUS = ["jetson_orin_agx_64gb", "coral_edge_tpu"]
MESH_SKUS = ["kpu_t64", "kpu_t128", "kpu_t256", "hailo8", "hailo10h"]
LOW_CONFIDENCE_SKUS = ["hailo8", "hailo10h"]


def _skip_if_optional_unresolved(sku: str) -> None:
    if sku in OPTIONAL_SKUS and resolve_sku_resource_model(sku) is None:
        pytest.skip(f"{sku} model unavailable in this environment")


# --------------------------------------------------------------------
# Topology scaling tests (acceptance criterion from issue body)
# --------------------------------------------------------------------

class TestTopologyScaling:
    """The hop_count_avg curve must scale according to topology:
    crossbar = O(1), ring = O(N), 2D mesh = O(sqrt(N))."""

    def test_crossbar_constant(self):
        f = SoCFabricModel(topology=Topology.CROSSBAR, controller_count=4)
        h4 = f.hop_count_avg()
        f2 = SoCFabricModel(topology=Topology.CROSSBAR, controller_count=64)
        h64 = f2.hop_count_avg()
        assert h4 == h64 == 1.0

    def test_ring_linear_in_n(self):
        f4 = SoCFabricModel(topology=Topology.RING, controller_count=4)
        f16 = SoCFabricModel(topology=Topology.RING, controller_count=16)
        f64 = SoCFabricModel(topology=Topology.RING, controller_count=64)
        # Linear: doubling N doubles avg hop count (within float)
        ratio_16_4 = f16.hop_count_avg() / f4.hop_count_avg()
        ratio_64_16 = f64.hop_count_avg() / f16.hop_count_avg()
        assert abs(ratio_16_4 - 4.0) < 1e-9
        assert abs(ratio_64_16 - 4.0) < 1e-9

    def test_mesh_2d_sqrt_n_when_square(self):
        """For square w x w mesh: avg hops = (w + w)/3 = 2w/3 ~ sqrt(N)."""
        f4 = SoCFabricModel(topology=Topology.MESH_2D, mesh_dimensions=(4, 4))
        f16 = SoCFabricModel(topology=Topology.MESH_2D,
                             mesh_dimensions=(16, 16))
        # 4x4 -> 8/3; 16x16 -> 32/3; ratio is 4 = sqrt(256/16).
        ratio = f16.hop_count_avg() / f4.hop_count_avg()
        assert abs(ratio - 4.0) < 1e-9

    def test_routing_factor_scales_linearly(self):
        f1 = SoCFabricModel(topology=Topology.MESH_2D, mesh_dimensions=(8, 8),
                            routing_distance_factor=1.0)
        f2 = SoCFabricModel(topology=Topology.MESH_2D, mesh_dimensions=(8, 8),
                            routing_distance_factor=1.5)
        assert abs(f2.hop_count_avg() / f1.hop_count_avg() - 1.5) < 1e-9

    def test_unknown_topology_returns_zero(self):
        f = SoCFabricModel(topology=Topology.UNKNOWN, controller_count=10)
        assert f.hop_count_avg() == 0.0


# --------------------------------------------------------------------
# Per-SKU schema fields
# --------------------------------------------------------------------

class TestPerSKUSchemaFields:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_soc_fabric_populated(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric is not None
        assert m.soc_fabric.topology is not Topology.UNKNOWN

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_soc_fabric_provenance_theoretical(self, sku):
        m = resolve_sku_resource_model(sku)
        conf = m.get_provenance("soc_fabric")
        assert conf.level is ConfidenceLevel.THEORETICAL

    @pytest.mark.parametrize("sku", RING_SKUS)
    def test_cpu_skus_use_ring(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric.topology is Topology.RING

    @pytest.mark.parametrize("sku", CROSSBAR_SKUS)
    def test_crossbar_skus(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric.topology is Topology.CROSSBAR

    @pytest.mark.parametrize("sku",
                             ["kpu_t64", "kpu_t128", "kpu_t256"])
    def test_kpu_uses_mesh_2d(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric.topology is Topology.MESH_2D
        assert m.soc_fabric.mesh_dimensions is not None

    @pytest.mark.parametrize("sku", LOW_CONFIDENCE_SKUS)
    def test_low_confidence_skus_flagged(self, sku):
        _skip_if_optional_unresolved(sku)
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric.low_confidence is True

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_provenance_string_populated(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric.provenance, f"{sku} fabric missing provenance"


# --------------------------------------------------------------------
# KPU equivalence: M6 must not regress numeric output
# --------------------------------------------------------------------

class TestKPUNumericEquivalence:
    """Critical M6 invariant: the KPU energy formula's numeric output
    must be identical before and after routing through soc_fabric."""

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_routing_distance_unchanged(self, sku):
        m = resolve_sku_resource_model(sku)
        tem = m.tile_energy_model
        # With fabric attached (default after M6)
        with_fabric = tem._estimate_l3_routing_distance()

        # Detach fabric and verify the legacy formula returns the same
        original = tem.soc_fabric
        try:
            tem.soc_fabric = None
            without_fabric = tem._estimate_l3_routing_distance()
        finally:
            tem.soc_fabric = original

        assert with_fabric == without_fabric, (
            f"{sku}: fabric path returns {with_fabric}, "
            f"legacy path returns {without_fabric}"
        )

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_routing_distance_factor_matches_legacy_constant(self, sku):
        """The fabric's routing_distance_factor must match the
        KPUTileEnergyModel.l3_routing_distance_factor default (1.2)."""
        m = resolve_sku_resource_model(sku)
        assert m.soc_fabric.routing_distance_factor == 1.2
        assert m.tile_energy_model.l3_routing_distance_factor == 1.2

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_mesh_dims_match_tile_energy_model(self, sku):
        m = resolve_sku_resource_model(sku)
        assert (m.soc_fabric.mesh_dimensions
                == m.tile_energy_model.tile_mesh_dimensions)


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_soc_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer6_soc_fabric_panel(sku)
        assert panel.layer is LayerTag.SOC_DATA_MOVEMENT
        assert "SoC" in panel.title or "Data Movement" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_populated(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer6_soc_fabric_panel(sku)
        assert panel.status != "not_populated"
        assert panel.metrics

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_topology_and_hop_count(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer6_soc_fabric_panel(sku)
        assert "Topology" in panel.metrics
        assert "Avg hop count" in panel.metrics

    @pytest.mark.parametrize("sku", LOW_CONFIDENCE_SKUS)
    def test_low_confidence_panels_flag_in_summary(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer6_soc_fabric_panel(sku)
        joined = " ".join(panel.notes).lower()
        assert "low confidence" in joined or "low-confidence" in joined

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_layer6_soc_fabric_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"


# --------------------------------------------------------------------
# Cross-SKU chart
# --------------------------------------------------------------------

class TestCrossSKUChart:
    def test_chart_includes_required_skus(self):
        chart = cross_sku_layer6_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            assert sku in chart.topology
            assert sku in chart.avg_hop_count

    def test_kpu_t256_has_largest_hop_count(self):
        """KPU T256 (16x16 mesh) should have the largest average hop
        count in the catalog."""
        chart = cross_sku_layer6_chart(REQUIRED_SKUS)
        winner = max(chart.avg_hop_count, key=chart.avg_hop_count.get)
        assert winner == "kpu_t256"

    def test_crossbar_skus_have_minimal_hops(self):
        chart = cross_sku_layer6_chart(REQUIRED_SKUS)
        for sku in CROSSBAR_SKUS:
            # 1.0 for true crossbar; 0.0 if controller_count=1
            assert chart.avg_hop_count[sku] <= 1.0

    def test_low_confidence_flag_passed_through(self):
        chart = cross_sku_layer6_chart(REQUIRED_SKUS + LOW_CONFIDENCE_SKUS)
        for sku in LOW_CONFIDENCE_SKUS:
            if sku not in chart.low_confidence:
                continue  # optional unresolved
            assert chart.low_confidence[sku] is True

    def test_provenance_all_theoretical(self):
        chart = cross_sku_layer6_chart(REQUIRED_SKUS)
        expected = ConfidenceLevel.THEORETICAL.value.upper()
        assert all(p == expected for p in chart.provenance.values())


# --------------------------------------------------------------------
# Round-trip serialization of the new fabric fields
# --------------------------------------------------------------------

class TestSerialization:
    def test_round_trip_preserves_mesh_dimensions(self):
        f = SoCFabricModel(
            topology=Topology.MESH_2D,
            mesh_dimensions=(16, 8),
            routing_distance_factor=1.2,
            low_confidence=False,
        )
        d = f.to_dict()
        r = SoCFabricModel.from_dict(d)
        assert r.mesh_dimensions == (16, 8)
        assert r.routing_distance_factor == 1.2
        assert r.low_confidence is False

    def test_round_trip_preserves_low_confidence(self):
        f = SoCFabricModel(topology=Topology.RING, low_confidence=True)
        d = f.to_dict()
        r = SoCFabricModel.from_dict(d)
        assert r.low_confidence is True

    def test_from_dict_rejects_malformed_mesh_dimensions(self):
        """A malformed mesh_dimensions blob must fail at
        deserialization, not later in hop_count_avg() where the
        (w, h) unpack would crash."""
        bad_inputs = [
            {"topology": "mesh_2d", "mesh_dimensions": [16]},        # too short
            {"topology": "mesh_2d", "mesh_dimensions": [16, 8, 4]},  # too long
            {"topology": "mesh_2d", "mesh_dimensions": [0, 8]},      # zero
            {"topology": "mesh_2d", "mesh_dimensions": [-4, 8]},     # negative
            {"topology": "mesh_2d", "mesh_dimensions": ["a", "b"]},  # non-int
            {"topology": "mesh_2d", "mesh_dimensions": "16x8"},      # string
        ]
        for blob in bad_inputs:
            with pytest.raises(ValueError, match="mesh_dimensions"):
                SoCFabricModel.from_dict(blob)

    def test_from_dict_accepts_well_formed_mesh_dimensions(self):
        f = SoCFabricModel.from_dict({
            "topology": "mesh_2d",
            "mesh_dimensions": [16, 8],
        })
        assert f.mesh_dimensions == (16, 8)
        assert f.hop_count_avg() == (16 + 8) / 3.0  # rdf default = 1.0
