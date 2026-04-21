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

    def test_unspecified_is_flat_one(self):
        """UNSPECIFIED has no pipeline model; returns flat 1.0."""
        curve = _synthesize_utilization_curve(
            TileScheduleClass.UNSPECIFIED, fill=0, drain=0,
        )
        ys = [u for _, u in curve]
        assert all(abs(y - 1.0) < 1e-9 for y in ys)

    def test_simt_data_parallel_flat_and_capped(self):
        """SIMT scheduling is flat across tile counts and capped by
        warp divergence x occupancy x coherence."""
        curve = _synthesize_utilization_curve(
            TileScheduleClass.SIMT_DATA_PARALLEL, fill=0, drain=0,
            warp_divergence_rate=0.05,
            warp_occupancy=0.75,
            coherence_efficiency=0.90,
        )
        ys = [u for _, u in curve]
        # Flat across all tile counts
        assert max(ys) - min(ys) < 1e-9
        # Expected: (1 - 0.025) * 0.75 * 0.90 = 0.65812
        expected = (1 - 0.5 * 0.05) * 0.75 * 0.90
        assert abs(ys[0] - expected) < 1e-9

    def test_simt_data_parallel_parameters_cap_utilization(self):
        """Lower warp occupancy lowers utilization; does not amortize with M."""
        low = _synthesize_utilization_curve(
            TileScheduleClass.SIMT_DATA_PARALLEL, 0, 0,
            warp_occupancy=0.30,
        )
        high = _synthesize_utilization_curve(
            TileScheduleClass.SIMT_DATA_PARALLEL, 0, 0,
            warp_occupancy=0.90,
        )
        # Low-occupancy < high-occupancy at every tile count
        for (_, lu), (_, hu) in zip(low, high):
            assert lu < hu


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

    def test_efficiency_curve_follows_linear_schedule_formula(self):
        """Chart 5 must follow eff = 1 - 2/(3M) (user's linear-schedule formula).

        Efficiency must be scale-invariant in N (all PE array sizes give
        the same curve) and must plateau toward 1.0 as M grows.
        """
        rpt = build_default_comparison(precision=Precision.INT8)
        kpu = [a for a in rpt.archetypes if a.archetype == "Domain Flow (KPU)"][0]
        assert len(kpu.array_scaling_curve) >= 2

        # Scale-invariance: every array-size curve is identical
        reference = kpu.array_scaling_curve[0]["points"]
        for curve in kpu.array_scaling_curve[1:]:
            assert curve["points"] == reference, (
                "Efficiency must be scale-invariant in PE array size N; "
                f"curve for {curve['pe_array_dim']}x{curve['pe_array_dim']} "
                "differs from the reference curve."
            )

        # User's formula: eff(M) = 1 - 2/(3M). Check a few M values.
        by_m = {p["tile_count"]: p["efficiency"] for p in reference}
        for m, expected in [(1, 1/3), (2, 2/3), (12, 17/18), (64, 1 - 2/192)]:
            assert m in by_m
            assert abs(by_m[m] - expected) < 1e-9, (
                f"At M={m}, expected 1-2/(3M)={expected:.6f}, "
                f"got {by_m[m]:.6f}"
            )

        # Plateau: rises monotonically, asymptotes toward 1.0
        ms = sorted(by_m.keys())
        effs = [by_m[m] for m in ms]
        assert effs == sorted(effs), "Efficiency must be monotonically increasing in M"
        assert effs[-1] > 0.99, f"Large-M plateau must exceed 0.99; got {effs[-1]:.4f}"


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


class TestChart5Rendering:
    """The rendered HTML must show the scale-invariance + plateau story."""

    def test_chart5_title_describes_linear_schedule_formula(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        # Title / caption must mention the user's formula rather than
        # the old (incorrect) "ops/W scaling with array size" claim.
        assert "1 - 2/(3M)" in html or "eff = 1 - 2/(3M)" in html
        assert "scale-invariant" in html or "scale invariant" in html.lower()
        # Old (incorrect) caption must be gone
        assert "fill/drain becomes negligible" not in html


class TestProcessAndFullAdderReference:
    """Energy reports must carry process node + FA reference for calibration."""

    def test_every_archetype_has_process_node_and_fa_reference(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        for a in rpt.archetypes:
            assert a.process_node_nm > 0, (
                f"{a.display_name} missing process_node_nm")
            assert a.full_adder_energy_pj > 0, (
                f"{a.display_name} missing full_adder_energy_pj")

    def test_fa_reference_sanity_by_process(self):
        """FA energy decreases monotonically with newer process nodes."""
        rpt = build_default_comparison(precision=Precision.INT8)
        # Pair process nm -> FA energy across all archetypes
        points = [(a.process_node_nm, a.full_adder_energy_pj) for a in rpt.archetypes]
        # For any two distinct points, smaller nm => smaller (or equal) FA energy
        for (nm1, fa1), (nm2, fa2) in [(p, q) for p in points for q in points]:
            if nm1 < nm2:
                assert fa1 <= fa2, f"FA @ {nm1}nm ({fa1}) > FA @ {nm2}nm ({fa2})"

    def test_html_shows_process_and_fa_reference(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        html = render_archetype_page(rpt, REPO_ROOT)
        assert " nm" in html              # process node column
        assert "full-adder" in html.lower() or "full adder" in html.lower()
        # Calibration note language
        assert "calibration" in html.lower() or "reference" in html.lower()


class TestToDict:
    """The data schema round-trips to a plain dict (for JSON emit)."""

    def test_entry_to_dict_has_all_fields(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        d = rpt.archetypes[0].to_dict()
        for key in ("archetype", "sku", "display_name", "energy_per_op_pj",
                    "peak_ops_per_sec", "ops_per_watt", "schedule_class",
                    "utilization_curve", "tdp_watts",
                    "process_node_nm", "full_adder_energy_pj"):
            assert key in d, f"missing field {key}"

    def test_report_to_dict_roundtrip_stable(self):
        rpt = build_default_comparison(precision=Precision.INT8)
        d = rpt.to_dict()
        assert "archetypes" in d
        assert len(d["archetypes"]) == 3
        assert d["precision"] == "int8"
