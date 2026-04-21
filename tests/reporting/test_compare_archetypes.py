"""Smoke tests for the M0.5 GPU/TPU/KPU comparison harness."""
from __future__ import annotations

from pathlib import Path

from graphs.hardware.resource_model import Precision, TileScheduleClass
from graphs.reporting.compare_archetypes import (
    ArchetypeComparisonReport,
    ArchetypeEntry,
    DEFAULT_TILE_COUNTS,
    REPRESENTATIVE_STEADY_CYCLES_PER_TILE,
    build_default_comparison,
    render_archetype_page,
    _synthesize_utilization_curve,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSynthesizeUtilizationCurve:
    def test_os_climbs_monotonically(self):
        curve = _synthesize_utilization_curve(
            TileScheduleClass.OUTPUT_STATIONARY, fill=24, drain=24,
        )
        ys = [u for _, u in curve]
        assert ys == sorted(ys)
        assert ys[0] < ys[-1]
        assert ys[-1] > 0.99

    def test_ws_is_flat(self):
        curve = _synthesize_utilization_curve(
            TileScheduleClass.WEIGHT_STATIONARY, fill=64, drain=64,
        )
        ys = [u for _, u in curve]
        assert max(ys) - min(ys) < 1e-9
        assert ys[0] < 1.0

    def test_unspecified_is_flat_065(self):
        curve = _synthesize_utilization_curve(
            TileScheduleClass.UNSPECIFIED, fill=0, drain=0,
        )
        ys = [u for _, u in curve]
        assert all(abs(y - 0.65) < 1e-9 for y in ys)


class TestBuildDefaultComparison:
    def test_three_archetypes_present(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        assert isinstance(rpt, ArchetypeComparisonReport)
        assert len(rpt.archetypes) == 3
        names = {a.archetype for a in rpt.archetypes}
        assert names == {"SIMT + Tensor Core", "Systolic (TPU)", "Domain Flow (KPU)"}

    def test_kpu_wins_energy_per_op(self):
        """The KPU entry must have the lowest energy-per-op."""
        rpt = build_default_comparison(precision=Precision.INT8)
        by_name = {a.archetype: a for a in rpt.archetypes}
        kpu = by_name["Domain Flow (KPU)"]
        tc = by_name["SIMT + Tensor Core"]
        assert kpu.energy_per_op_pj < tc.energy_per_op_pj

    def test_kpu_utilization_saturates(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        kpu = [a for a in rpt.archetypes if a.archetype == "Domain Flow (KPU)"][0]
        util_at_last = kpu.utilization_curve[-1][1]
        assert util_at_last > 0.98

    def test_tpu_utilization_is_flat(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        tpu = [a for a in rpt.archetypes if a.archetype == "Systolic (TPU)"][0]
        ys = [u for _, u in tpu.utilization_curve]
        assert max(ys) - min(ys) < 1e-6

    def test_array_scaling_populated_for_kpu_only(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        by_name = {a.archetype: a for a in rpt.archetypes}
        assert len(by_name["Domain Flow (KPU)"].array_scaling_curve) > 0
        assert len(by_name["SIMT + Tensor Core"].array_scaling_curve) == 0
        assert len(by_name["Systolic (TPU)"].array_scaling_curve) == 0


class TestRenderArchetypePage:
    def test_html_contains_five_chart_divs(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        for cid in ("chart1", "chart2", "chart3", "chart4", "chart5"):
            assert f'id="{cid}"' in html

    def test_html_loads_plotly_cdn(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        assert "cdn.plot.ly" in html
        assert "Plotly.newPlot" in html

    def test_html_is_branes_branded(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        assert "Branes" in html or "branes" in html.lower()

    def test_html_mentions_scheduling_story(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        assert "output_stationary" in html or "output-stationary" in html.lower()
        assert "weight_stationary" in html or "weight-stationary" in html.lower()

    def test_all_three_skus_named(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        assert "Jetson Orin AGX" in html
        assert "Coral Edge TPU" in html
        assert "KPU T128" in html


class TestToDict:
    """The data schema round-trips to a plain dict (for JSON emit)."""

    def test_entry_to_dict_has_all_fields(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        d = rpt.archetypes[0].to_dict()
        for key in ("archetype", "sku", "display_name", "energy_per_op_pj",
                    "peak_ops_per_sec", "ops_per_watt", "schedule_class",
                    "utilization_curve", "tdp_watts"):
            assert key in d, f"missing field {key}"

    def test_report_to_dict_roundtrip_stable(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        d = rpt.to_dict()
        assert "archetypes" in d
        assert len(d["archetypes"]) == 3
        assert d["precision"] == "int8"
