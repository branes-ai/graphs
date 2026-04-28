"""Self-consistency tests for the M2 Layer 2 (register file) panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.reporting.layer_panels import (
    build_layer2_register_panel,
    cross_sku_layer2_chart,
    resolve_sku_resource_model,
)
from graphs.reporting.layer_panels.layer2_register import (
    _kpu_fill_drain_overhead,
    _process_node_nm,
    _representative_alu_energy_pj,
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
    def test_i7_12700k_carries_simd_efficiency(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert m.simd_efficiency is not None
        for op_kind in ("elementwise", "matrix", "default"):
            assert op_kind in m.simd_efficiency
            assert 0.0 < m.simd_efficiency[op_kind] <= 1.0

    def test_ryzen_9_carries_simd_efficiency(self):
        m = resolve_sku_resource_model("ryzen_9_8945hs")
        assert m.simd_efficiency is not None
        for op_kind in ("elementwise", "matrix", "default"):
            assert op_kind in m.simd_efficiency
            assert 0.0 < m.simd_efficiency[op_kind] <= 1.0

    def test_zen4_simd_efficiency_above_alder_lake(self):
        """Native AVX-512 + no hybrid scheduling -> Zen 4 wins on every
        op-kind by construction."""
        i7 = resolve_sku_resource_model("intel_core_i7_12700k")
        r9 = resolve_sku_resource_model("ryzen_9_8945hs")
        for op_kind in ("elementwise", "matrix", "default"):
            assert r9.simd_efficiency[op_kind] > i7.simd_efficiency[op_kind]

    def test_simd_efficiency_provenance_theoretical(self):
        for sku in ("intel_core_i7_12700k", "ryzen_9_8945hs"):
            m = resolve_sku_resource_model(sku)
            for op_kind in ("elementwise", "matrix", "default"):
                conf = m.get_provenance(f"simd_efficiency.{op_kind}")
                assert conf.level is ConfidenceLevel.THEORETICAL

    def test_coral_edge_tpu_carries_pipeline_fill_overhead(self):
        m = resolve_sku_resource_model("coral_edge_tpu")
        assert m.pipeline_fill_overhead is not None
        assert 0.0 <= m.pipeline_fill_overhead <= 1.0

    def test_pipeline_fill_overhead_provenance_theoretical(self):
        m = resolve_sku_resource_model("coral_edge_tpu")
        conf = m.get_provenance("pipeline_fill_overhead")
        assert conf.level is ConfidenceLevel.THEORETICAL

    def test_kpu_skus_have_no_simd_or_pfo_field(self):
        """KPU SKUs intentionally do not carry simd_efficiency or
        pipeline_fill_overhead -- they're not relevant to dataflow."""
        for sku in ("kpu_t64", "kpu_t128", "kpu_t256"):
            m = resolve_sku_resource_model(sku)
            assert m.simd_efficiency is None
            assert m.pipeline_fill_overhead is None


class TestProcessNodeAndALUHelpers:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_process_node_in_plausible_range(self, sku):
        m = resolve_sku_resource_model(sku)
        node = _process_node_nm(m)
        assert 3 <= node <= 28, f"{sku} process node {node} nm out of range"

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_representative_alu_energy_positive(self, sku):
        m = resolve_sku_resource_model(sku)
        pj = _representative_alu_energy_pj(m)
        assert pj > 0
        assert pj < 100, f"{sku} ALU energy {pj} pJ implausibly large"


class TestKPUFillDrain:
    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_kpu_fill_drain_present(self, sku):
        """M0.5 already populates fill / drain on every KPU tile spec.
        M2 reads them; this test guards against that breaking."""
        m = resolve_sku_resource_model(sku)
        fd = _kpu_fill_drain_overhead(m)
        assert fd is not None, f"{sku} should expose tile fill / drain"
        fill, drain = fd
        assert fill > 0
        assert drain > 0

    def test_non_kpu_sku_returns_none(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert _kpu_fill_drain_overhead(m) is None


class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_register_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer2_register_panel(sku)
        assert panel.layer is LayerTag.REGISTER
        assert "Register" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_populated_status(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer2_register_panel(sku)
        assert panel.status != "not_populated"
        assert panel.metrics

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_register_read_and_write(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer2_register_panel(sku)
        assert any("read" in k.lower() for k in panel.metrics)
        assert any("write" in k.lower() for k in panel.metrics)

    def test_cpu_panel_includes_simd_metrics(self):
        panel = build_layer2_register_panel("intel_core_i7_12700k")
        simd_keys = [k for k in panel.metrics if "SIMD" in k]
        assert len(simd_keys) == 3, (
            f"Expected 3 SIMD-eff metrics, got {simd_keys}"
        )

    def test_tpu_panel_includes_fill_overhead(self):
        panel = build_layer2_register_panel("coral_edge_tpu")
        assert any("fill" in k.lower() for k in panel.metrics)

    def test_kpu_panel_includes_tile_fill_drain(self):
        panel = build_layer2_register_panel("kpu_t128")
        keys_lower = [k.lower() for k in panel.metrics]
        assert any("fill" in k for k in keys_lower)
        assert any("drain" in k for k in keys_lower)

    def test_kpu_panel_notes_dataflow_dodge(self):
        """The KPU panel must surface the non-double-counting story."""
        panel = build_layer2_register_panel("kpu_t128")
        joined = " ".join(panel.notes).lower()
        assert "dataflow" in joined or "domain-flow" in joined

    def test_unknown_sku_returns_unpopulated(self):
        panel = build_layer2_register_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"


class TestCrossSKUChart:
    def test_chart_includes_required_skus(self):
        chart = cross_sku_layer2_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            assert sku in chart.register_read_pj
            assert sku in chart.register_write_pj
            assert sku in chart.read_alu_ratio

    def test_register_write_above_read(self):
        """Write energy is always >= read in a register file."""
        chart = cross_sku_layer2_chart(REQUIRED_SKUS)
        for sku in REQUIRED_SKUS:
            r = chart.register_read_pj[sku]
            w = chart.register_write_pj[sku]
            assert w >= r, f"{sku} write {w} should be >= read {r}"

    def test_ryzen_lower_register_energy_than_i7(self):
        """4 nm Phoenix beats 10 nm Alder Lake on every per-access
        energy by physics."""
        chart = cross_sku_layer2_chart(["intel_core_i7_12700k", "ryzen_9_8945hs"])
        assert (chart.register_read_pj["ryzen_9_8945hs"]
                < chart.register_read_pj["intel_core_i7_12700k"])

    def test_read_alu_ratio_in_plausible_range(self):
        """Register-as-fraction-of-ALU should be in [0.1, 5.0] on any
        modern silicon."""
        chart = cross_sku_layer2_chart(REQUIRED_SKUS)
        for sku, ratio in chart.read_alu_ratio.items():
            assert 0.1 < ratio < 5.0, (
                f"{sku} ratio {ratio:.3f} outside plausible range"
            )

    def test_provenance_all_theoretical_at_m2(self):
        chart = cross_sku_layer2_chart(REQUIRED_SKUS)
        assert all(p == "THEORETICAL" for p in chart.provenance.values())


class TestNoDoubleCounting:
    """Critical M2 invariant: the KPU panel must not introduce a new
    fill/drain energy beyond what M0.5 already accounts for."""

    @pytest.mark.parametrize("sku", ("kpu_t64", "kpu_t128", "kpu_t256"))
    def test_kpu_panel_reads_existing_fill_drain(self, sku):
        m = resolve_sku_resource_model(sku)
        panel = build_layer2_register_panel(sku)

        # Pull the spec values directly from the M0.5 tile abstraction
        fd = _kpu_fill_drain_overhead(m)
        assert fd is not None
        fill, drain = fd

        # The panel must surface exactly these numbers
        fill_metric = panel.metrics.get("Tile fill cycles")
        drain_metric = panel.metrics.get("Tile drain cycles")
        assert fill_metric is not None
        assert drain_metric is not None
        assert int(fill_metric["value"]) == fill
        assert int(drain_metric["value"]) == drain
