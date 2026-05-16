"""Self-consistency tests for the M7 Layer 7 (external memory) panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.architectural_energy import (
    DataParallelEnergyModel,
    StoredProgramEnergyModel,
)
from graphs.reporting.layer_panels import (
    build_layer7_external_memory_panel,
    cross_sku_layer7_chart,
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


def _skip_if_optional_unresolved(sku: str) -> None:
    if sku in OPTIONAL_SKUS and resolve_sku_resource_model(sku) is None:
        pytest.skip(f"{sku} model unavailable in this environment")


class TestPerSKUSchemaFields:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_memory_technology_set(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.memory_technology
        assert isinstance(m.memory_technology, str)

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_read_write_energy_set(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m.memory_read_energy_per_byte_pj is not None
        assert m.memory_write_energy_per_byte_pj is not None
        assert m.memory_read_energy_per_byte_pj > 0
        assert m.memory_write_energy_per_byte_pj > 0

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_write_at_or_above_read(self, sku):
        """DRAM writes always cost >= reads (precharge tail)."""
        m = resolve_sku_resource_model(sku)
        assert (m.memory_write_energy_per_byte_pj
                >= m.memory_read_energy_per_byte_pj)

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_asymmetry_in_plausible_range(self, sku):
        """W/R asymmetry should be in [1.0, 1.5] for any modern DRAM."""
        m = resolve_sku_resource_model(sku)
        ratio = (m.memory_write_energy_per_byte_pj
                 / m.memory_read_energy_per_byte_pj)
        assert 1.0 <= ratio <= 1.5, (
            f"{sku} W/R asymmetry {ratio:.2f}x outside [1.0, 1.5]"
        )

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_layer7_provenance_theoretical(self, sku):
        m = resolve_sku_resource_model(sku)
        for key in ("memory_technology",
                    "memory_read_energy_per_byte_pj",
                    "memory_write_energy_per_byte_pj"):
            conf = m.get_provenance(key)
            assert conf.level is ConfidenceLevel.THEORETICAL, (
                f"{sku}/{key} provenance={conf.level}"
            )


class TestEnergyModelAccessPattern:
    def test_access_pattern_table_on_both_models(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        sp = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        dp = DataParallelEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        for m in (sp, dp):
            assert isinstance(m.memory_access_pattern_multiplier, dict)
            for pattern in ("sequential", "strided", "random"):
                assert pattern in m.memory_access_pattern_multiplier

    def test_pattern_ordering(self):
        """Sequential is cheapest; strided is in the middle; random
        is most expensive."""
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        seq = m.memory_access_pattern_multiplier["sequential"]
        strd = m.memory_access_pattern_multiplier["strided"]
        rnd = m.memory_access_pattern_multiplier["random"]
        assert seq <= strd < rnd

    def test_sequential_is_baseline(self):
        from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
        m = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)
        assert m.memory_access_pattern_multiplier["sequential"] == 1.0


class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_external_memory_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer7_external_memory_panel(sku)
        assert panel.layer is LayerTag.EXTERNAL_MEMORY
        assert "External" in panel.title or "Memory" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_populated(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer7_external_memory_panel(sku)
        assert panel.status != "not_populated"
        assert panel.metrics

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_required_metrics(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer7_external_memory_panel(sku)
        assert "Memory technology" in panel.metrics
        assert "Peak bandwidth" in panel.metrics
        assert "Read energy" in panel.metrics
        assert "Write energy" in panel.metrics
        assert "W/R asymmetry" in panel.metrics

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_access_pattern_metrics(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer7_external_memory_panel(sku)
        for pattern in ("sequential", "strided", "random"):
            key = f"Access pattern ({pattern})"
            assert key in panel.metrics

    def test_hailo8_panel_calls_out_on_chip_split(self):
        """Hailo-8 deploys weights from on-chip SRAM in steady state
        (no external DRAM) -- the panel must surface this rather than
        papering over with phantom DRAM numbers.

        Hailo-10H is excluded: it ships with 4-8 GiB LPDDR4X external
        DRAM holding the model weights + KV cache spill, so the on-
        chip note correctly does NOT fire there. The split is gated on
        ``memory_technology`` -- see ``layer7_external_memory.py``."""
        _skip_if_optional_unresolved("hailo8")
        panel = build_layer7_external_memory_panel("hailo8")
        joined = " ".join(panel.notes).lower()
        assert "on-chip" in joined or "sram" in joined

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_layer7_external_memory_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"


class TestCrossSKUChart:
    def test_chart_includes_required_skus(self):
        chart = cross_sku_layer7_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            assert sku in chart.memory_technology
            assert sku in chart.peak_bandwidth_gbps
            assert sku in chart.read_energy_pj
            assert sku in chart.write_energy_pj

    def test_lpddr5_cheaper_than_ddr5(self):
        """The motivation chart: mobile / on-package LPDDR5 should
        cost less per byte than desktop DDR5 even though both are
        DDR-class JEDEC standards."""
        chart = cross_sku_layer7_chart(["intel_core_i7_12700k",
                                         "jetson_orin_agx_64gb"])
        assert (chart.read_energy_pj["jetson_orin_agx_64gb"]
                < chart.read_energy_pj["intel_core_i7_12700k"])

    def test_kpu_t256_highest_bandwidth(self):
        """T256 is the high-end KPU SKU; should top the BW chart
        in this catalog."""
        chart = cross_sku_layer7_chart(REQUIRED_SKUS)
        winner = max(chart.peak_bandwidth_gbps,
                     key=chart.peak_bandwidth_gbps.get)
        assert winner == "kpu_t256"

    def test_asymmetry_consistent_with_per_sku(self):
        """Chart's W/R asymmetry must equal write/read ratio."""
        chart = cross_sku_layer7_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            expected = chart.write_energy_pj[sku] / chart.read_energy_pj[sku]
            assert abs(chart.asymmetry[sku] - expected) < 1e-9

    def test_provenance_all_theoretical(self):
        chart = cross_sku_layer7_chart(REQUIRED_SKUS)
        expected = ConfidenceLevel.THEORETICAL.value.upper()
        assert all(p == expected for p in chart.provenance.values())


class TestPlausibility:
    """Energy ranges sanity check against the M7 issue's motivation."""

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_read_energy_in_plausible_range(self, sku):
        """Modern DRAM read energy: 5 - 35 pJ/B. HBM at the low end,
        legacy desktop DDR4/5 at the high end."""
        m = resolve_sku_resource_model(sku)
        assert 5.0 <= m.memory_read_energy_per_byte_pj <= 35.0

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_bandwidth_in_plausible_range(self, sku):
        """Bandwidth across the catalog: 1 - 1024 GB/s."""
        m = resolve_sku_resource_model(sku)
        bw_gbs = m.peak_bandwidth / 1e9
        assert 1.0 <= bw_gbs <= 1024.0
