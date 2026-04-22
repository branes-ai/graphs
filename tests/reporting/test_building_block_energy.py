"""Tests for the per-clock engine-level (building-block) energy view."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.building_block_energy import (
    BuildingBlock,
    EngineComponent,
    SocComposition,
    build_default_report,
    build_kpu_tile_building_block,
    build_nvidia_sm_building_block,
    render_building_block_page,
)
from graphs.reporting.microarch_accounting import (
    StructureCategory,
    build_default_report as mar_default_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestEngineTotals:
    def test_sm_total_in_plausible_range(self):
        sm = build_nvidia_sm_building_block()
        # At 8 nm, 0.65 GHz sustained: per-SM should be 500-1500 pJ/clock
        assert 500 < sm.total_pj_per_clock < 1500

    def test_kpu_tile_total_in_plausible_range(self):
        tile = build_kpu_tile_building_block()
        # At 16 nm, 1 GHz: per-tile should be 100-300 pJ/clock
        assert 100 < tile.total_pj_per_clock < 300

    def test_power_matches_pj_times_ghz(self):
        sm = build_nvidia_sm_building_block()
        expected_mw = sm.total_pj_per_clock * sm.clock_ghz
        assert abs(sm.power_mw - expected_mw) < 1e-9

    def test_native_throughput_matches_architecture(self):
        sm = build_nvidia_sm_building_block()
        assert sm.native_macs_per_clock == 4096  # 4 TCs x 1024 MACs/clock
        tile = build_kpu_tile_building_block()
        assert tile.native_macs_per_clock == 576  # 24 x 24


class TestEngineComposition:
    def test_sm_has_all_major_components(self):
        sm = build_nvidia_sm_building_block()
        names = [c.name.lower() for c in sm.components]
        assert any("register file" in n for n in names)
        assert any("instruction pipeline" in n for n in names)
        assert any("warp scheduler" in n for n in names)
        assert any("operand collector" in n for n in names)
        assert any("cuda" in n for n in names)
        assert any("sfu" in n for n in names)
        assert any("tensor core" in n for n in names)

    def test_kpu_tile_has_l1_and_mesh(self):
        tile = build_kpu_tile_building_block()
        names = [c.name.lower() for c in tile.components]
        assert any("l1 scratchpad" in n for n in names)
        assert any("mesh" in n for n in names)

    def test_execute_fraction_higher_for_kpu(self):
        """KPU tile should spend a larger fraction on Execute than SM,
        because it lacks the instruction-pipeline + scheduler tax."""
        sm = build_nvidia_sm_building_block()
        tile = build_kpu_tile_building_block()
        assert tile.execute_fraction > sm.execute_fraction

    def test_all_components_have_citations(self):
        for b in build_default_report().blocks:
            for c in b.components:
                assert c.citation, f"{c.name} missing citation"
                assert c.activity_note, f"{c.name} missing activity note"


class TestCrossValidationWithPerMacView:
    """The derived pJ/MAC from this per-clock view must agree with the
    independent per-MAC accounting in microarch_accounting.py."""

    def test_sm_pj_per_mac_agrees_with_per_mac_view(self):
        sm = build_nvidia_sm_building_block()
        mar = mar_default_report()
        sm_permac = next(b for b in mar.blocks
                         if "Multiprocessor" in b.building_block)
        delta = (abs(sm.derived_pj_per_mac - sm_permac.total_pj_per_mac)
                 / sm_permac.total_pj_per_mac)
        assert delta < 0.30, (
            f"SM derived {sm.derived_pj_per_mac:.3f} vs per-MAC "
            f"{sm_permac.total_pj_per_mac:.3f}, delta {delta*100:.0f}%")

    def test_kpu_pj_per_mac_agrees_with_per_mac_view(self):
        tile = build_kpu_tile_building_block()
        mar = mar_default_report()
        kpu_permac = next(b for b in mar.blocks
                          if "KPU" in b.building_block)
        delta = (abs(tile.derived_pj_per_mac - kpu_permac.total_pj_per_mac)
                 / kpu_permac.total_pj_per_mac)
        assert delta < 0.30, (
            f"KPU derived {tile.derived_pj_per_mac:.3f} vs per-MAC "
            f"{kpu_permac.total_pj_per_mac:.3f}, delta {delta*100:.0f}%")


class TestSocComposition:
    def test_soc_power_grows_with_block_count(self):
        tile = build_kpu_tile_building_block()
        small = SocComposition(tile, 64, 0.55, 1500.0)
        large = SocComposition(tile, 256, 0.55, 4500.0)
        assert large.total_power_w > small.total_power_w

    def test_peak_tops_is_block_count_times_macs_per_clock(self):
        tile = build_kpu_tile_building_block()
        soc = SocComposition(tile, 128, 0.55, 2500.0)
        macs_per_sec = 128 * 576 * 1.0e9  # 128 tiles, 576 MACs/clk, 1 GHz
        expected = macs_per_sec * 2 / 1e12
        assert abs(soc.peak_tops_int8 - expected) < 1e-6

    def test_sustained_tops_per_watt_positive(self):
        for s in build_default_report().socs:
            assert s.sustained_tops_per_watt > 0

    def test_utilization_scales_sustained_but_not_peak(self):
        tile = build_kpu_tile_building_block()
        low = SocComposition(tile, 128, 0.3, 2500.0)
        high = SocComposition(tile, 128, 0.8, 2500.0)
        assert low.peak_tops_int8 == high.peak_tops_int8
        assert high.sustained_tops_int8 > low.sustained_tops_int8


class TestHTMLRender:
    def test_html_has_both_blocks(self):
        rpt = build_default_report()
        html = render_building_block_page(rpt, REPO_ROOT)
        assert "Streaming Multiprocessor" in html
        assert "KPU Compute Tile" in html

    def test_html_has_charts(self):
        rpt = build_default_report()
        html = render_building_block_page(rpt, REPO_ROOT)
        assert 'id="chart_per_clock"' in html
        assert 'id="chart_soc"' in html

    def test_html_has_soc_composition_table(self):
        rpt = build_default_report()
        html = render_building_block_page(rpt, REPO_ROOT)
        assert "SoC composition" in html
        assert "Sustained TOPS" in html

    def test_html_includes_cross_validation_table(self):
        rpt = build_default_report()
        html = render_building_block_page(rpt, REPO_ROOT)
        assert "Cross-validation" in html
        assert "microarch_accounting" in html

    def test_html_back_link_to_index(self):
        rpt = build_default_report()
        html = render_building_block_page(rpt, REPO_ROOT)
        assert 'href="index.html"' in html

    def test_html_links_to_per_mac_view(self):
        rpt = build_default_report()
        html = render_building_block_page(rpt, REPO_ROOT)
        assert 'href="microarch_accounting.html"' in html
