"""Self-consistency tests for the M4 Layer 4 (L2 cache) panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.architectural_energy import DataParallelEnergyModel
from graphs.reporting.layer_panels import (
    build_layer4_l2_cache_panel,
    cross_sku_layer4_chart,
    resolve_sku_resource_model,
)
from graphs.reporting.layer_panels.layer4_l2_cache import (
    VALID_L2_TOPOLOGIES,
    _has_l2_field,
    _normalize_topology,
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

# Topology partitioning that downstream tests rely on.
PER_UNIT_SKUS = [
    "intel_core_i7_12700k", "ryzen_9_8945hs",
    "kpu_t64", "kpu_t128", "kpu_t256",
]
SHARED_LLC_SKUS = [
    "jetson_orin_agx_64gb", "coral_edge_tpu", "hailo8", "hailo10h",
]


def _skip_if_optional_unresolved(sku: str) -> None:
    if sku in OPTIONAL_SKUS and resolve_sku_resource_model(sku) is None:
        pytest.skip(f"{sku} model unavailable in this environment")


class TestPerSKUSchemaFields:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l2_cache_per_unit_field_present(self, sku):
        m = resolve_sku_resource_model(sku)
        assert _has_l2_field(m), f"{sku} missing l2_cache_per_unit"

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l2_topology_set(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l2_topology in VALID_L2_TOPOLOGIES, (
            f"{sku} l2_topology={m.l2_topology!r}"
        )

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l2_provenance_theoretical(self, sku):
        m = resolve_sku_resource_model(sku)
        for key in ("l2_cache_per_unit", "l2_topology"):
            conf = m.get_provenance(key)
            assert conf.level is ConfidenceLevel.THEORETICAL, (
                f"{sku}/{key} provenance={conf.level}"
            )

    @pytest.mark.parametrize("sku", PER_UNIT_SKUS)
    def test_per_unit_skus_classified_correctly(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l2_topology == "per-unit"

    @pytest.mark.parametrize("sku", SHARED_LLC_SKUS)
    def test_shared_llc_skus_classified_correctly(self, sku):
        _skip_if_optional_unresolved(sku)
        m = resolve_sku_resource_model(sku)
        assert m.l2_topology == "shared-llc"


class TestLegacyFieldUntouched:
    """The schema's legacy ``l2_cache_total`` field (which holds the
    LLC per M1 convention) must not collide with the new
    ``l2_cache_per_unit``."""

    def test_legacy_l2_total_unchanged_on_cpu(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert m.l2_cache_total == 25 * 1024 * 1024  # 25 MB L3 LLC

    def test_legacy_l2_total_holds_llc_on_ryzen(self):
        m = resolve_sku_resource_model("ryzen_9_8945hs")
        assert m.l2_cache_total == 16 * 1024 * 1024  # 16 MB L3 LLC

    def test_per_unit_distinct_from_total_on_cpu(self):
        """Physical L2 per core (1.25 MB) should differ from the LLC."""
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert m.l2_cache_per_unit < m.l2_cache_total


class TestEnergyModelL2Lookup:
    def test_per_op_l2_table_exists(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert isinstance(m.l2_hit_rate_by_op, dict)
        for op_kind in ("matrix", "elementwise", "default"):
            assert op_kind in m.l2_hit_rate_by_op
            rate = m.l2_hit_rate_by_op[op_kind]
            assert 0.0 < rate <= 1.0

    def test_matrix_l2_hit_above_elementwise(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert m.l2_hit_rate_by_op["matrix"] > m.l2_hit_rate_by_op["elementwise"]

    def test_scalar_field_unchanged(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert m.l2_hit_rate == 0.90

    def test_prefetch_effectiveness_default(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert 0.0 < m.prefetch_effectiveness <= 1.0

    def test_per_op_lookup_drives_compute_energy(self):
        """L2 per-op rate must flow through compute_architectural_energy.
        Like the M3 L1 case, the explanation echoes the chosen rate."""
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        m.l2_hit_rate_by_op = {
            "matrix":      0.95,
            "elementwise": 0.40,
            "default":     0.90,
        }
        e_matrix = m.compute_architectural_energy(
            ops=10000, bytes_transferred=40000,
            execution_context={"op_kind": "matrix"},
        )
        e_elem = m.compute_architectural_energy(
            ops=10000, bytes_transferred=40000,
            execution_context={"op_kind": "elementwise"},
        )
        # The L2 explanation line carries the rate label
        assert "95% of Shared/L1 misses" in e_matrix.explanation
        assert "40% of Shared/L1 misses" in e_elem.explanation


class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_l2_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer4_l2_cache_panel(sku)
        assert panel.layer is LayerTag.L2_CACHE
        assert "L2" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_populated(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer4_l2_cache_panel(sku)
        assert panel.status != "not_populated"
        assert panel.metrics

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_topology(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer4_l2_cache_panel(sku)
        assert "Topology" in panel.metrics
        assert panel.metrics["Topology"]["value"] in VALID_L2_TOPOLOGIES

    def test_cpu_panel_includes_per_op_hit_rates(self):
        panel = build_layer4_l2_cache_panel("intel_core_i7_12700k")
        hit_metrics = [k for k in panel.metrics if "Hit rate (" in k]
        assert len(hit_metrics) >= 3

    def test_kpu_panel_marks_deterministic(self):
        panel = build_layer4_l2_cache_panel("kpu_t128")
        joined = " ".join(panel.notes).lower()
        assert "software-managed" in joined or "deterministic" in joined

    def test_coral_panel_marks_collapsed(self):
        """TPU has L2 collapsed into UB; panel must not pretend a
        capacity exists."""
        panel = build_layer4_l2_cache_panel("coral_edge_tpu")
        assert "L2 status" in panel.metrics
        assert panel.metrics["L2 status"]["value"] == "collapsed into LLC"
        # Must NOT have "L2 per unit" / "L2 total" fields when collapsed
        assert "L2 per unit" not in panel.metrics
        assert "L2 total" not in panel.metrics

    def test_shared_llc_panels_have_llc_note(self):
        """Every shared-llc SKU must mention the L2-is-LLC fact."""
        for sku in ("jetson_orin_agx_64gb", "coral_edge_tpu",
                    "hailo8", "hailo10h"):
            _skip_if_optional_unresolved(sku)
            panel = build_layer4_l2_cache_panel(sku)
            joined = " ".join(panel.notes).lower()
            assert ("last-level cache" in joined or "llc" in joined), (
                f"{sku} shared-llc panel missing LLC annotation"
            )

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_layer4_l2_cache_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"


class TestCrossSKUChart:
    def test_chart_includes_required_skus(self):
        chart = cross_sku_layer4_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            assert sku in chart.l2_per_unit_bytes
            assert sku in chart.topology
            assert sku in chart.energy_pj_per_byte

    def test_l2_total_equals_per_unit_times_units(self):
        chart = cross_sku_layer4_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            m = resolve_sku_resource_model(sku)
            expected = m.l2_cache_per_unit * (m.compute_units or 1)
            assert chart.l2_total_bytes[sku] == expected

    def test_cpu_l2_per_unit_above_kpu(self):
        """CPU per-core L2 (>= 1 MB) should be much larger than KPU
        per-tile L2 (~32 KiB)."""
        chart = cross_sku_layer4_chart(["intel_core_i7_12700k", "kpu_t128"])
        assert (chart.l2_per_unit_bytes["intel_core_i7_12700k"]
                > chart.l2_per_unit_bytes["kpu_t128"] * 10)

    def test_kpu_dominates_aggregate_at_t256(self):
        """T256 has 256 tiles * 32 KiB = 8 MiB total L2, larger than
        most cache-based SKUs in absolute terms."""
        chart = cross_sku_layer4_chart(REQUIRED_SKUS)
        kpu_t256_total = chart.l2_total_bytes["kpu_t256"]
        # Assert it's at least 4 MiB
        assert kpu_t256_total >= 4 * 1024 * 1024

    def test_provenance_all_theoretical(self):
        chart = cross_sku_layer4_chart(REQUIRED_SKUS)
        assert all(p == "THEORETICAL" for p in chart.provenance.values())


class TestTopologyValidation:
    def test_normalize_canonicalizes_case(self):
        assert _normalize_topology("Per-Unit", "x") == "per-unit"
        assert _normalize_topology(" SHARED ", "x") == "shared"
        assert _normalize_topology("SHARED-LLC", "x") == "shared-llc"

    def test_normalize_defaults_when_unset(self):
        assert _normalize_topology(None, "x") == "per-unit"
        assert _normalize_topology("", "x") == "per-unit"

    def test_normalize_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid l2_topology"):
            _normalize_topology("disjoint", "test_sku")

    def test_panel_raises_on_invalid_topology(self):
        m = resolve_sku_resource_model("kpu_t128")
        original = m.l2_topology
        try:
            m.l2_topology = "MUDDLED"
            with pytest.raises(ValueError):
                build_layer4_l2_cache_panel("kpu_t128", model=m)
        finally:
            m.l2_topology = original


class TestNoDoubleCounting:
    """KPU SKUs read l2_cache_per_unit from the resource model, which
    the SKU factory populated from the M0.5 KPUTileEnergyModel.
    Confirm the values match exactly."""

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_kpu_l2_matches_tile_energy_model(self, sku):
        m = resolve_sku_resource_model(sku)
        assert hasattr(m, "tile_energy_model")
        assert m.l2_cache_per_unit == m.tile_energy_model.l2_size_per_tile
