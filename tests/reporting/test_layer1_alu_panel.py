"""Self-consistency tests for the M1 Layer 1 ALU panel."""
from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.resource_model import Precision
from graphs.reporting.layer_panels import (
    build_layer1_panel,
    cross_sku_layer1_chart,
    resolve_sku_resource_model,
)
from graphs.reporting.layer_panels.layer1_alu import (
    LAYER1_PRECISIONS,
    _peak_ops_per_sec,
    _provenance_tag,
)


# SKUs that must always resolve (no optional dependency on Hailo SDK).
REQUIRED_SKUS = [
    "jetson_orin_agx_64gb",
    "intel_core_i7_12700k",
    "ryzen_9_8945hs",
    "kpu_t64",
    "kpu_t128",
    "kpu_t256",
    "coral_edge_tpu",
]

# Hailo SKUs are optional in the runtime contract; tests must mirror that.
OPTIONAL_SKUS = ["hailo8", "hailo10h"]

TARGET_SKUS = REQUIRED_SKUS + OPTIONAL_SKUS


class TestSKUResolution:
    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_required_sku_resolves(self, sku):
        """Required SKUs must always resolve."""
        m = resolve_sku_resource_model(sku)
        assert m is not None, f"{sku} did not resolve"

    @pytest.mark.parametrize("sku", OPTIONAL_SKUS)
    def test_optional_sku_resolves_or_skips(self, sku):
        """Hailo SKUs are optional: either resolve and have fabrics,
        or return None when their model module fails to import."""
        m = resolve_sku_resource_model(sku)
        if m is None:
            pytest.skip(f"{sku} model unavailable in this environment")
        assert m.compute_fabrics, f"{sku} resolved but has no compute_fabrics"

    @pytest.mark.parametrize("sku", REQUIRED_SKUS)
    def test_required_sku_has_compute_fabrics(self, sku):
        m = resolve_sku_resource_model(sku)
        assert m is not None
        assert m.compute_fabrics, f"{sku} has no compute_fabrics"

    def test_unknown_sku_returns_none(self):
        assert resolve_sku_resource_model("definitely_not_a_sku") is None


class TestNewCPUResourceModels:
    """The two new CPU resource models built for M1."""

    def test_i7_12700k_has_p_and_e_core_fabrics(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        assert m is not None
        types = {f.fabric_type for f in m.compute_fabrics}
        assert "alder_lake_p_core_avx2" in types
        assert "gracemont_e_core_avx2" in types

    def test_i7_12700k_p_core_count(self):
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        p = next(f for f in m.compute_fabrics
                 if f.fabric_type == "alder_lake_p_core_avx2")
        assert p.num_units == 8
        e = next(f for f in m.compute_fabrics
                 if f.fabric_type == "gracemont_e_core_avx2")
        assert e.num_units == 4

    def test_i7_12700k_int8_higher_than_fp32(self):
        """VNNI gives 4x INT8 vs FP32 throughput on a single fabric."""
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        peak_fp32 = _peak_ops_per_sec(m, Precision.FP32)
        peak_int8 = _peak_ops_per_sec(m, Precision.INT8)
        assert peak_int8 > 3.5 * peak_fp32  # ~4x VNNI advantage

    def test_i7_12700k_no_avx512_so_bf16_emulated(self):
        """Alder Lake retail has no native BF16 -- BF16 should be far
        below INT8 (which has VNNI) and below FP32."""
        m = resolve_sku_resource_model("intel_core_i7_12700k")
        peak_bf16 = _peak_ops_per_sec(m, Precision.BF16)
        peak_fp32 = _peak_ops_per_sec(m, Precision.FP32)
        assert peak_bf16 < peak_fp32  # emulated BF16

    def test_ryzen_9_8945hs_has_zen4_fabric(self):
        m = resolve_sku_resource_model("ryzen_9_8945hs")
        assert m is not None
        assert any(f.fabric_type == "zen4_avx512" for f in m.compute_fabrics)

    def test_ryzen_9_8945hs_native_bf16(self):
        """Zen 4 has native AVX-512_BF16 -- BF16 should be 2x FP32."""
        m = resolve_sku_resource_model("ryzen_9_8945hs")
        peak_fp32 = _peak_ops_per_sec(m, Precision.FP32)
        peak_bf16 = _peak_ops_per_sec(m, Precision.BF16)
        # 2x within +/-30% tolerance per M1 spec
        assert 1.4 * peak_fp32 <= peak_bf16 <= 2.6 * peak_fp32

    def test_ryzen_9_8945hs_native_int8_vnni(self):
        m = resolve_sku_resource_model("ryzen_9_8945hs")
        peak_fp32 = _peak_ops_per_sec(m, Precision.FP32)
        peak_int8 = _peak_ops_per_sec(m, Precision.INT8)
        assert peak_int8 >= 1.5 * peak_fp32

    def test_new_cpu_skus_have_provenance_tags(self):
        """Both new CPUs tag every populated precision with THEORETICAL."""
        for sku in ("intel_core_i7_12700k", "ryzen_9_8945hs"):
            m = resolve_sku_resource_model(sku)
            for prec in (Precision.FP64, Precision.FP32, Precision.BF16,
                         Precision.INT8, Precision.INT4):
                conf = m.get_provenance(
                    f"compute_fabric.ops_per_clock.{prec.value}"
                )
                assert conf.level is ConfidenceLevel.THEORETICAL, (
                    f"{sku}/{prec.value} provenance is {conf.level}"
                )


def _skip_if_optional_unresolved(sku: str) -> None:
    """Skip the test when an optional SKU's model module is missing."""
    if sku in OPTIONAL_SKUS and resolve_sku_resource_model(sku) is None:
        pytest.skip(f"{sku} model unavailable in this environment")


class TestPanelBuilder:
    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_is_alu_layer(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer1_panel(sku)
        assert panel.layer is LayerTag.ALU
        assert "ALU" in panel.title

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_status_and_metrics(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer1_panel(sku)
        assert panel.status != "not_populated", (
            f"{sku} produced an unpopulated panel"
        )
        assert panel.metrics, f"{sku} panel has no metrics"

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_panel_has_summary_text(self, sku):
        _skip_if_optional_unresolved(sku)
        panel = build_layer1_panel(sku)
        assert panel.summary
        assert "compute fabric" in panel.summary

    def test_unknown_sku_panel_is_not_populated(self):
        panel = build_layer1_panel("definitely_not_a_sku")
        assert panel.status == "not_populated"

    def test_metrics_carry_provenance(self):
        """Every per-precision peak metric carries a provenance tag."""
        panel = build_layer1_panel("kpu_t128")
        peak_metrics = [
            (k, v) for k, v in panel.metrics.items() if "peak" in k
        ]
        assert peak_metrics
        for name, entry in peak_metrics:
            assert "provenance" in entry, f"{name} has no provenance"
            assert entry["provenance"] in (
                "CALIBRATED", "INTERPOLATED", "THEORETICAL", "UNKNOWN"
            )

    def test_calibrated_cpu_metrics_tagged_theoretical(self):
        """The new CPUs were tagged THEORETICAL by construction."""
        panel = build_layer1_panel("intel_core_i7_12700k")
        peak_metrics = [v for k, v in panel.metrics.items() if "peak" in k]
        assert all(v["provenance"] == "THEORETICAL" for v in peak_metrics)


class TestCrossSKUChart:
    def test_chart_includes_every_target_sku(self):
        chart = cross_sku_layer1_chart(TARGET_SKUS)
        assert chart.skus == TARGET_SKUS

    def test_chart_has_at_least_int8_and_fp32(self):
        """Every architecture in the catalog supports at least INT8 or FP32."""
        chart = cross_sku_layer1_chart(TARGET_SKUS)
        assert "int8" in chart.precisions
        assert "fp32" in chart.precisions

    def test_kpu_dominates_int8(self):
        """KPU T256 should be the highest INT8 peak in the M1 catalog."""
        chart = cross_sku_layer1_chart(TARGET_SKUS)
        int8_peaks = {
            sku: chart.peak_ops[("int8", sku)]
            for sku in chart.skus
            if ("int8", sku) in chart.peak_ops
        }
        winner = max(int8_peaks, key=int8_peaks.get)
        assert winner == "kpu_t256", (
            f"Expected kpu_t256 to lead INT8; got {winner} with peaks "
            f"{int8_peaks}"
        )

    def test_no_negative_peaks(self):
        chart = cross_sku_layer1_chart(TARGET_SKUS)
        for v in chart.peak_ops.values():
            assert v > 0


class TestPhysicalPlausibility:
    """Plausibility ranges for the M1 +/-30% tolerance."""

    @pytest.mark.parametrize("sku", TARGET_SKUS)
    def test_int8_peak_in_plausible_range(self, sku):
        """INT8 peak ops/sec must be in [1 GOPS, 1 POPS] for any
        embodied-AI-class SKU."""
        _skip_if_optional_unresolved(sku)
        m = resolve_sku_resource_model(sku)
        peak = _peak_ops_per_sec(m, Precision.INT8)
        if peak == 0:
            pytest.skip(f"{sku} does not advertise INT8")
        assert 1e9 < peak < 1e15, (
            f"{sku} INT8 peak {peak/1e12:.2f} TOPS outside plausible range"
        )

    def test_aggregate_status_reflects_provenance(self):
        """If any precision is THEORETICAL, the panel cannot be CALIBRATED."""
        for sku in TARGET_SKUS:
            m = resolve_sku_resource_model(sku)
            if m is None:
                continue  # optional SKU unavailable
            tags = {
                _provenance_tag(m, p)
                for p in LAYER1_PRECISIONS
                if _peak_ops_per_sec(m, p) > 0
            }
            panel = build_layer1_panel(sku)
            if "THEORETICAL" in tags:
                assert panel.status in ("theoretical", "unknown",
                                         "not_populated")
