"""Self-consistency tests for the M3 Layer 3 (L1 cache / scratchpad) panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.architectural_energy import DataParallelEnergyModel
from graphs.reporting.layer_panels import (
    build_layer3_l1_cache_panel,
    cross_sku_layer3_chart,
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

CACHE_BASED_SKUS = [
    "jetson_orin_agx_64gb",
    "intel_core_i7_12700k",
    "ryzen_9_8945hs",
]
SCRATCHPAD_BASED_SKUS = [
    "kpu_t64",
    "kpu_t128",
    "kpu_t256",
    "coral_edge_tpu",
    "hailo8",
    "hailo10h",
]


def _skip_if_optional_unresolved(sku: str) -> None:
    if sku in OPTIONAL_SKUS and resolve_sku_resource_model(sku) is None:
        pytest.skip(f"{sku} model unavailable in this environment")


class TestPerSKUSchemaFields:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l1_cache_per_unit_populated(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l1_cache_per_unit > 0

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l1_storage_kind_set(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l1_storage_kind in ("cache", "scratchpad"), (
            f"{sku} l1_storage_kind={m.l1_storage_kind!r}"
        )

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l1_cache_per_unit_provenance(self, sku):
        m = resolve_sku_resource_model(sku)
        conf = m.get_provenance("l1_cache_per_unit")
        assert conf.level is ConfidenceLevel.THEORETICAL

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l1_storage_kind_provenance(self, sku):
        m = resolve_sku_resource_model(sku)
        conf = m.get_provenance("l1_storage_kind")
        assert conf.level is ConfidenceLevel.THEORETICAL

    @pytest.mark.parametrize("sku", CACHE_BASED_SKUS)
    def test_cache_skus_classified_cache(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l1_storage_kind == "cache"

    @pytest.mark.parametrize("sku", SCRATCHPAD_BASED_SKUS)
    def test_scratchpad_skus_classified_scratchpad(self, sku):
        _skip_if_optional_unresolved(sku)
        m = resolve_sku_resource_model(sku)
        assert m.l1_storage_kind == "scratchpad"


class TestEnergyModelLookup:
    def test_per_op_hit_rate_exists(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert isinstance(m.shared_mem_l1_hit_rate_by_op, dict)
        for op_kind in ("matrix", "elementwise", "default"):
            assert op_kind in m.shared_mem_l1_hit_rate_by_op
            rate = m.shared_mem_l1_hit_rate_by_op[op_kind]
            assert 0.0 < rate <= 1.0

    def test_matrix_hit_rate_above_elementwise(self):
        """Matrix workloads with weight reuse should hit higher than
        elementwise streams with poor locality."""
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        rates = m.shared_mem_l1_hit_rate_by_op
        assert rates["matrix"] > rates["elementwise"]

    def test_scalar_field_unchanged(self):
        """Backwards compat: legacy callers reading the scalar still
        get the matrix-equivalent default."""
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert m.shared_mem_l1_hit_rate == 0.95
        assert (m.shared_mem_l1_hit_rate
                == m.shared_mem_l1_hit_rate_by_op["matrix"])

    def test_per_op_lookup_drives_compute_energy(self):
        """The per-op-type lookup must actually flow through the
        compute_architectural_energy() path, otherwise the lookup is
        decorative. The explanation string must echo the chosen op_kind
        and percentage.
        """
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        m.shared_mem_l1_hit_rate_by_op = {
            "matrix":      0.95,
            "elementwise": 0.50,
            "default":     0.50,
        }
        e_matrix = m.compute_architectural_energy(
            ops=10000, bytes_transferred=40000,
            execution_context={"op_kind": "matrix"},
        )
        e_elem = m.compute_architectural_energy(
            ops=10000, bytes_transferred=40000,
            execution_context={"op_kind": "elementwise"},
        )
        assert "op_kind=matrix" in e_matrix.explanation
        assert "op_kind=elementwise" in e_elem.explanation
        assert "95% hit rate" in e_matrix.explanation
        assert "50% hit rate" in e_elem.explanation


class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_l1_cache_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer3_l1_cache_panel(sku)
        assert panel.layer is LayerTag.L1_CACHE
        assert "L1" in panel.title or "Scratchpad" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_populated(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer3_l1_cache_panel(sku)
        assert panel.status != "not_populated"
        assert panel.metrics

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_capacity_and_storage_kind(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer3_l1_cache_panel(sku)
        assert any("L1 per unit" == k for k in panel.metrics)
        assert any("Storage kind" == k for k in panel.metrics)

    @pytest.mark.parametrize("sku", CACHE_BASED_SKUS)
    def test_cache_panel_includes_per_op_hit_rates(self, sku):
        panel = build_layer3_l1_cache_panel(sku)
        hit_metrics = [k for k in panel.metrics if "Hit rate (" in k]
        assert len(hit_metrics) >= 3, (
            f"Expected per-op hit rates for cache SKU {sku}, got "
            f"{hit_metrics}"
        )

    @pytest.mark.parametrize("sku", SCRATCHPAD_BASED_SKUS)
    def test_scratchpad_panel_marks_deterministic(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer3_l1_cache_panel(sku)
        # Either has a "Hit rate (deterministic)" metric or notes
        # that explicitly mention deterministic / software-managed.
        has_metric = any("deterministic" in k.lower() for k in panel.metrics)
        joined_notes = " ".join(panel.notes).lower()
        has_note = ("software-managed" in joined_notes or
                    "deterministic" in joined_notes)
        assert has_metric or has_note

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_layer3_l1_cache_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"


class TestCrossSKUChart:
    def test_chart_includes_required_skus(self):
        chart = cross_sku_layer3_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            assert sku in chart.l1_per_unit_bytes
            assert sku in chart.l1_total_bytes
            assert sku in chart.storage_kind
            assert sku in chart.energy_pj_per_byte

    def test_l1_total_equals_per_unit_times_units(self):
        chart = cross_sku_layer3_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            m = resolve_sku_resource_model(sku)
            expected = m.l1_cache_per_unit * (m.compute_units or 1)
            assert chart.l1_total_bytes[sku] == expected

    def test_kpu_dominates_total_l1_among_scratchpad_skus(self):
        """KPU tile counts (64/128/256) put their aggregate L1 well
        above any single-die scratchpad accelerator."""
        chart = cross_sku_layer3_chart(SCRATCHPAD_BASED_SKUS)
        winner = max(chart.l1_total_bytes, key=chart.l1_total_bytes.get)
        assert winner.startswith("kpu_"), (
            f"Expected a KPU SKU to win total scratchpad capacity; "
            f"got {winner}"
        )

    def test_energy_in_plausible_range(self):
        """L1 read energy should sit between ~0.1 and ~4 pJ/byte for
        any modern SRAM technology, with the upper bound covering
        bulk-planar 28nm (Coral Edge TPU on GF 28nm SLP lands at
        ~3.2 pJ/B per the TechnologyProfile derivation -- legitimate
        for that node, vs ~1.6 at 14/16nm FinFET)."""
        chart = cross_sku_layer3_chart(REQUIRED_SKUS)
        for sku, e in chart.energy_pj_per_byte.items():
            assert 0.1 < e < 4.0, (
                f"{sku} L1 energy {e:.3f} pJ/byte outside plausible range"
            )

    def test_provenance_all_theoretical(self):
        chart = cross_sku_layer3_chart(REQUIRED_SKUS)
        assert all(p == "THEORETICAL" for p in chart.provenance.values())


class TestStorageKindValidation:
    """l1_storage_kind must be normalized + validated to avoid silent
    misclassification on typos."""

    def test_normalization_canonicalizes_case(self):
        from graphs.reporting.layer_panels.layer3_l1_cache import (
            _normalize_storage_kind,
        )
        assert _normalize_storage_kind("Cache", "x") == "cache"
        assert _normalize_storage_kind(" SCRATCHPAD ", "x") == "scratchpad"

    def test_normalization_defaults_when_unset(self):
        from graphs.reporting.layer_panels.layer3_l1_cache import (
            _normalize_storage_kind,
        )
        assert _normalize_storage_kind(None, "x") == "cache"
        assert _normalize_storage_kind("", "x") == "cache"

    def test_normalization_rejects_unknown_value(self):
        from graphs.reporting.layer_panels.layer3_l1_cache import (
            _normalize_storage_kind,
        )
        with pytest.raises(ValueError, match="Invalid l1_storage_kind"):
            _normalize_storage_kind("buffer", "test_sku")

    def test_panel_raises_on_invalid_kind(self):
        """Panel construction surfaces the invalid value loudly rather
        than silently misreporting hit-rate semantics."""
        m = resolve_sku_resource_model("kpu_t128")
        original = m.l1_storage_kind
        try:
            m.l1_storage_kind = "BUFFER_TYPO"
            with pytest.raises(ValueError):
                build_layer3_l1_cache_panel("kpu_t128", model=m)
        finally:
            m.l1_storage_kind = original


class TestPositiveCapacityGuard:
    """The capacity guard rejects None, zero, and negative values
    rather than relying on truthiness."""

    def test_zero_capacity_marks_unpopulated(self):
        """Mutate a real SKU's L1 to 0 and confirm the panel marks
        it not_populated rather than rendering a bogus 0 KiB cache."""
        m = resolve_sku_resource_model("kpu_t128")
        original = m.l1_cache_per_unit
        try:
            m.l1_cache_per_unit = 0
            panel = build_layer3_l1_cache_panel("kpu_t128", model=m)
            assert panel.status == "not_populated"
        finally:
            m.l1_cache_per_unit = original

    def test_negative_capacity_marks_unpopulated(self):
        m = resolve_sku_resource_model("kpu_t128")
        original = m.l1_cache_per_unit
        try:
            m.l1_cache_per_unit = -1
            panel = build_layer3_l1_cache_panel("kpu_t128", model=m)
            assert panel.status == "not_populated"
        finally:
            m.l1_cache_per_unit = original

    def test_helper_rejects_none(self):
        from graphs.reporting.layer_panels.layer3_l1_cache import (
            _has_positive_l1,
        )
        assert _has_positive_l1(None) is False


class TestNoDoubleCounting:
    """Critical M3 invariant: KPU panels read tile-local SRAM directly
    from the M0.5 abstraction without recomputing capacity."""

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_kpu_l1_matches_resource_model(self, sku):
        m = resolve_sku_resource_model(sku)
        panel = build_layer3_l1_cache_panel(sku)

        # The L1-per-unit metric value (in formatted KiB) must match
        # the model's l1_cache_per_unit value.
        expected_kib = m.l1_cache_per_unit / 1024
        per_unit_metric = panel.metrics["L1 per unit"]
        # value is "256" for 256 KiB, or e.g. "0.5" for sub-KiB MiB
        # but our SKUs all sit in pure KiB territory.
        assert per_unit_metric["unit"] in ("KiB", "MiB")
        if per_unit_metric["unit"] == "KiB":
            assert float(per_unit_metric["value"]) == expected_kib

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_kpu_marked_scratchpad_in_panel(self, sku):
        panel = build_layer3_l1_cache_panel(sku)
        assert panel.metrics["Storage kind"]["value"] == "scratchpad"
