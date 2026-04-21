"""Tests for the competitive-trajectory analysis."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.competitive_trajectory import (
    JETSON_HISTORY,
    HistoricalPoint,
    TrajectoryReport,
    build_default_report,
    render_trajectory_page,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestHistoricalData:
    def test_history_covers_xavier_and_orin(self):
        products = {p.product for p in JETSON_HISTORY}
        assert any("Xavier AGX" in p for p in products)
        assert any("Orin AGX" in p for p in products)

    def test_every_point_has_valid_values(self):
        for p in JETSON_HISTORY:
            assert p.year >= 2019
            assert p.tops_per_watt > 0
            assert p.tdp_w > 0
            assert p.peak_tops_int8 > 0
            computed = p.peak_tops_int8 / p.tdp_w
            assert abs(computed - p.tops_per_watt) < 0.02


class TestTrajectoryMath:
    def test_demonstrated_cagr_is_reasonable(self):
        rpt = build_default_report()
        cagr = rpt.demonstrated_cagr()
        assert 0.15 < cagr < 0.35, (
            f"CAGR {cagr*100:.1f}%/yr outside plausible 15-35% range.")

    def test_years_to_target_monotone_in_rate(self):
        rpt = build_default_report()
        prev = None
        for rate in sorted(rpt.growth_rates):
            yrs = rpt.years_to_target(rpt.kpu_sustained_tops_per_watt, rate)
            if prev is not None:
                assert yrs < prev
            prev = yrs

    def test_already_above_target_returns_zero(self):
        rpt = build_default_report()
        anchor = rpt.anchor_point.tops_per_watt
        assert rpt.years_to_target(anchor * 0.5, 0.20) == 0.0
        assert rpt.years_to_target(anchor, 0.20) == 0.0

    def test_peak_takes_longer_than_sustained(self):
        """Peak target (12.3) > sustained target (11.04), so peak
        requires more years to reach at any given growth rate."""
        rpt = build_default_report()
        peak_years = rpt.years_to_target(rpt.kpu_peak_tops_per_watt, 0.20)
        sust_years = rpt.years_to_target(rpt.kpu_sustained_tops_per_watt, 0.20)
        assert peak_years > sust_years


class TestHTMLRender:
    def test_html_has_chart_div(self):
        rpt = build_default_report()
        html = render_trajectory_page(rpt, REPO_ROOT)
        assert 'id="chart_trajectory"' in html

    def test_html_has_parity_table_and_history_table(self):
        rpt = build_default_report()
        html = render_trajectory_page(rpt, REPO_ROOT)
        assert 'class="trajectory"' in html
        for rate in rpt.growth_rates:
            assert f"{rate*100:.0f}%/yr" in html

    def test_html_mentions_methodology_assumptions(self):
        rpt = build_default_report()
        html = render_trajectory_page(rpt, REPO_ROOT)
        assert "dense INT8" in html
        assert "sparsity" in html.lower() or "sparse" in html.lower()
        assert "THEORETICAL" in html

    def test_html_back_link_to_index(self):
        rpt = build_default_report()
        html = render_trajectory_page(rpt, REPO_ROOT)
        assert 'href="index.html"' in html
