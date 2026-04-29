"""Self-consistency tests for the M5 Layer 5 (L3 / LLC) panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.architectural_energy import (
    DataParallelEnergyModel,
    StoredProgramEnergyModel,
)
from graphs.reporting.layer_panels import (
    build_layer5_l3_cache_panel,
    cross_sku_layer5_chart,
    resolve_sku_resource_model,
)
from graphs.reporting.layer_panels.layer5_l3_cache import (
    VALID_COHERENCE_PROTOCOLS,
    _has_l3,
    _normalize_coherence,
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

# Architectural partitioning the panel relies on. Note: KPU SKUs
# carry a distributed per-tile L3 SRAM scratchpad via the M0.5
# KPUTileEnergyModel abstraction, so they classify as l3_present=True
# even though their coherence_protocol is 'none' (software-managed).
COHERENT_L3_SKUS = ["intel_core_i7_12700k", "ryzen_9_8945hs"]
SCRATCHPAD_L3_SKUS = ["kpu_t64", "kpu_t128", "kpu_t256"]
L3_PRESENT_SKUS = COHERENT_L3_SKUS + SCRATCHPAD_L3_SKUS
NO_L3_SKUS = [
    "jetson_orin_agx_64gb", "coral_edge_tpu", "hailo8", "hailo10h",
]


def _skip_if_optional_unresolved(sku: str) -> None:
    if sku in OPTIONAL_SKUS and resolve_sku_resource_model(sku) is None:
        pytest.skip(f"{sku} model unavailable in this environment")


class TestPerSKUSchemaFields:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l3_fields_set(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l3_present is not None
        assert m.l3_cache_total is not None
        assert m.coherence_protocol in VALID_COHERENCE_PROTOCOLS

    @pytest.mark.parametrize("sku", COHERENT_L3_SKUS)
    def test_cpu_skus_have_coherent_l3(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.l3_present is True
        assert m.l3_cache_total > 0
        assert m.coherence_protocol == "snoopy_mesi"

    @pytest.mark.parametrize("sku", SCRATCHPAD_L3_SKUS)
    def test_kpu_skus_have_scratchpad_l3(self, sku):
        """KPU SKUs carry a distributed per-tile L3 SRAM via M0.5,
        but coherence is 'none' (software-managed)."""
        m = resolve_sku_resource_model(sku)
        assert m.l3_present is True
        assert m.l3_cache_total > 0
        assert m.coherence_protocol == "none"

    @pytest.mark.parametrize("sku", NO_L3_SKUS)
    def test_no_l3_skus_classified(self, sku):
        _skip_if_optional_unresolved(sku)
        m = resolve_sku_resource_model(sku)
        assert m.l3_present is False
        assert (m.l3_cache_total or 0) == 0
        assert m.coherence_protocol == "none"

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_l5_provenance_theoretical(self, sku):
        m = resolve_sku_resource_model(sku)
        for key in ("l3_present", "coherence_protocol"):
            conf = m.get_provenance(key)
            assert conf.level is ConfidenceLevel.THEORETICAL, (
                f"{sku}/{key} provenance={conf.level}"
            )

    def test_legacy_l2_total_unchanged(self):
        """The legacy l2_cache_total field still holds the LLC value
        per M1 convention (= L3 on x86) -- M5 must not collide with it."""
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert m.l2_cache_total == 25 * 1024 * 1024
        # And the new M5 field carries the same number, explicitly:
        assert m.l3_cache_total == 25 * 1024 * 1024


class TestEnergyModelL3Lookup:
    def test_per_op_l3_table_exists(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert isinstance(m.l3_hit_rate_by_op, dict)
        for op_kind in ("matrix", "elementwise", "default"):
            assert op_kind in m.l3_hit_rate_by_op
            rate = m.l3_hit_rate_by_op[op_kind]
            assert 0.0 < rate <= 1.0

    def test_matrix_l3_above_elementwise(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert m.l3_hit_rate_by_op["matrix"] > m.l3_hit_rate_by_op["elementwise"]

    def test_l3_scalar_field_unchanged(self):
        """Backward compat: legacy callers still get l3_hit_rate."""
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert m.l3_hit_rate == 0.95

    def test_coherence_protocol_overhead_set(self):
        """Both energy models expose the panel-facing pJ/request value."""
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        sp = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        dp = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert sp.coherence_protocol_overhead_pj_per_request > 0
        assert dp.coherence_protocol_overhead_pj_per_request > 0
        # Same TechnologyProfile -> same value
        assert (sp.coherence_protocol_overhead_pj_per_request
                == dp.coherence_protocol_overhead_pj_per_request)


class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_l3_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer5_l3_cache_panel(sku)
        assert panel.layer is LayerTag.L3_CACHE
        assert "L3" in panel.title or "Last-Level" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_populated(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer5_l3_cache_panel(sku)
        assert panel.status != "not_populated"
        assert panel.metrics
        assert "Coherence protocol" in panel.metrics

    @pytest.mark.parametrize("sku", COHERENT_L3_SKUS)
    def test_coherent_l3_panel_includes_per_op_hit_rates(self, sku):
        panel = build_layer5_l3_cache_panel(sku)
        hit_metrics = [k for k in panel.metrics if "Hit rate (" in k
                       and "deterministic" not in k]
        assert len(hit_metrics) >= 3, (
            f"{sku} expected per-op L3 hit rates, got {hit_metrics}"
        )
        assert "L3 total" in panel.metrics

    @pytest.mark.parametrize("sku", SCRATCHPAD_L3_SKUS)
    def test_scratchpad_l3_panel_marks_deterministic(self, sku):
        panel = build_layer5_l3_cache_panel(sku)
        # KPU panels show capacity but no per-op cache hit rates.
        assert "L3 total" in panel.metrics
        assert "Hit rate (deterministic)" in panel.metrics
        assert not any(
            "Hit rate (" in k and "deterministic" not in k
            for k in panel.metrics
        )

    @pytest.mark.parametrize("sku", NO_L3_SKUS)
    def test_no_l3_panel_marks_status(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer5_l3_cache_panel(sku)
        assert "L3 status" in panel.metrics
        assert panel.metrics["L3 status"]["value"] == "no L3 by design"
        # Must NOT have the L3-present metrics
        assert "L3 total" not in panel.metrics
        assert not any("Hit rate (" in k for k in panel.metrics)

    def test_cpu_panel_mentions_protocol_vs_transport(self):
        """The split between PROTOCOL (Layer 5) and TRANSPORT (Layer 6)
        is the core M5 narrative; CPU panels must call it out."""
        panel = build_layer5_l3_cache_panel("intel_core_i7_12700k")
        joined = " ".join(panel.notes).lower()
        assert "protocol" in joined and "transport" in joined

    def test_no_l3_panel_does_not_emit_coherence_pj(self):
        """SKUs with coherence='none' should not advertise a
        coherence-PROTOCOL energy cost."""
        panel = build_layer5_l3_cache_panel("kpu_t128")
        assert "Coherence pJ / request" not in panel.metrics

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_layer5_l3_cache_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"


class TestCrossSKUChart:
    def test_chart_includes_required_skus(self):
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            assert sku in chart.l3_present
            assert sku in chart.coherence_protocol

    def test_l3_total_matches_resource_model(self):
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            m = resolve_sku_resource_model(sku)
            expected = m.l3_cache_total or 0
            assert chart.l3_total_bytes[sku] == expected

    def test_l3_presence_matches_partitioning(self):
        """CPUs and KPU (distributed scratchpad) carry l3_present=True;
        Orin / Coral / Hailo do not."""
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        for sku, present in chart.l3_present.items():
            if sku in L3_PRESENT_SKUS:
                assert present is True, f"{sku} should have L3"
            else:
                assert present is False, f"{sku} should not have L3"

    def test_no_phantom_energies_for_skus_without_l3(self):
        """When l3_present=False, the chart must not advertise a
        non-zero L3 energy or coherence cost (would contradict the
        per-SKU panel)."""
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        for sku in NO_L3_SKUS:
            if sku not in chart.l3_total_bytes:
                continue  # optional SKU unresolved
            assert chart.l3_total_bytes[sku] == 0
            assert chart.energy_pj_per_byte[sku] == 0.0

    def test_no_phantom_coherence_when_protocol_none(self):
        """SKUs with coherence='none' must show 0 coherence pJ/req."""
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        for sku, proto in chart.coherence_protocol.items():
            if proto == "none":
                assert chart.coherence_pj_per_request[sku] == 0.0

    def test_coherence_protocol_categorical(self):
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        for proto in chart.coherence_protocol.values():
            assert proto in VALID_COHERENCE_PROTOCOLS

    def test_l3_energy_in_plausible_range(self):
        """L3 read energy should be in [0.1, 10.0] pJ/byte across the
        catalog. CPUs at 4-10 nm land near 1 pJ/byte; KPU at 16 nm
        lands near 7 pJ/byte (older process, larger SRAM cells)."""
        chart = cross_sku_layer5_chart(L3_PRESENT_SKUS)
        for sku, e in chart.energy_pj_per_byte.items():
            assert 0.1 < e < 10.0, (
                f"{sku} L3 energy {e:.3f} pJ/byte out of plausible range"
            )

    def test_provenance_all_theoretical(self):
        chart = cross_sku_layer5_chart(REQUIRED_SKUS)
        expected = ConfidenceLevel.THEORETICAL.value.upper()
        assert all(p == expected for p in chart.provenance.values())


class TestCoherenceValidation:
    def test_normalize_canonicalizes_case(self):
        assert _normalize_coherence("Snoopy_MESI", "x") == "snoopy_mesi"
        assert _normalize_coherence(" DIRECTORY ", "x") == "directory"
        assert _normalize_coherence("None", "x") == "none"

    def test_normalize_defaults_when_unset(self):
        assert _normalize_coherence(None, "x") == "none"
        assert _normalize_coherence("", "x") == "none"

    def test_normalize_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid coherence_protocol"):
            _normalize_coherence("mesif_extended", "test_sku")

    def test_panel_raises_on_invalid_coherence(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        original = m.coherence_protocol
        try:
            m.coherence_protocol = "GARBAGE"
            with pytest.raises(ValueError):
                build_layer5_l3_cache_panel("intel_core_i7_12700k", model=m)
        finally:
            m.coherence_protocol = original


class TestL3PresenceGuard:
    def test_has_l3_requires_both_flag_and_capacity(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert _has_l3(m)
        # Mutate to break the invariants
        original_present = m.l3_present
        original_total = m.l3_cache_total
        try:
            m.l3_present = False
            assert not _has_l3(m)
            m.l3_present = True
            m.l3_cache_total = 0
            assert not _has_l3(m)
        finally:
            m.l3_present = original_present
            m.l3_cache_total = original_total

    def test_zero_l3_marks_no_l3_panel(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        original = (m.l3_present, m.l3_cache_total)
        try:
            m.l3_present = False
            m.l3_cache_total = 0
            panel = build_layer5_l3_cache_panel(
                "intel_core_i7_12700k", model=m
            )
            assert "L3 status" in panel.metrics
            assert "L3 total" not in panel.metrics
        finally:
            m.l3_present, m.l3_cache_total = original
