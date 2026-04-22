"""Tests for the per-clock engine-level (building-block) energy view."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.building_block_energy import (
    SocComposition,
    build_default_report,
    build_kpu_tile_building_block,
    build_nvidia_sm_building_block,
    build_nvidia_sm_cuda_path_building_block,
    render_building_block_page,
)
from graphs.reporting.microarch_accounting import (
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
        # At 8 nm, 1.5 GHz: per-tile should be 50-200 pJ/clock
        assert 50 < tile.total_pj_per_clock < 200

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

    def test_every_component_has_transistor_count(self):
        for b in build_default_report().blocks:
            for c in b.components:
                assert c.transistor_count_m > 0, (
                    f"{c.name} missing transistor count")

    def test_every_component_has_silicon_area(self):
        for b in build_default_report().blocks:
            for c in b.components:
                assert c.area_mm2 > 0, f"{c.name} missing area_mm2"

    def test_sm_total_area_in_ampere_range(self):
        """Published Ampere SM is ~2-5 mm^2 of compute silicon."""
        sm = build_nvidia_sm_building_block()
        assert 2.0 < sm.total_area_mm2 < 6.0

    def test_kpu_tile_area_much_smaller_than_sm(self):
        """KPU tile should be order-of-magnitude smaller than an SM."""
        sm = build_nvidia_sm_building_block()
        tile = build_kpu_tile_building_block()
        assert tile.total_area_mm2 < sm.total_area_mm2 / 10

    def test_sm_transistor_count_in_ampere_range(self):
        """Published Ampere SM lands near 240 M transistors."""
        sm = build_nvidia_sm_building_block()
        assert 200 < sm.total_transistor_count_m < 300

    def test_kpu_tile_transistor_count_is_tiny_vs_sm(self):
        """KPU tile is mesh + scratchpad + a little control; much
        smaller than an SM."""
        sm = build_nvidia_sm_building_block()
        tile = build_kpu_tile_building_block()
        assert tile.total_transistor_count_m < sm.total_transistor_count_m / 10

    def test_sm_cuda_path_has_same_silicon_as_tc_path(self):
        """Both views describe the same SM silicon; only activity
        (pJ/clock per component) differs."""
        tc = build_nvidia_sm_building_block()
        cuda = build_nvidia_sm_cuda_path_building_block()
        assert abs(tc.total_transistor_count_m
                   - cuda.total_transistor_count_m) < 1.0

    def test_sm_cuda_path_is_much_less_efficient_per_mac(self):
        """FP32 FMA per-MAC energy must be much higher than TC HMMA
        per-MAC - the quantitative reason Tensor Cores exist."""
        tc = build_nvidia_sm_building_block()
        cuda = build_nvidia_sm_cuda_path_building_block()
        assert cuda.derived_pj_per_mac > 10 * tc.derived_pj_per_mac


class TestCrossValidationWithPerMacView:
    """The derived pJ/MAC from this per-clock view must agree with the
    independent per-MAC accounting in microarch_accounting.py."""

    # The two views measure subtly different things:
    #   - microarch_accounting.py itemizes the HMMA instruction's
    #     dynamic energy + a 20% leakage adder. It excludes SM
    #     sub-units that are idle during HMMA (RT, texture, L1 D$).
    #   - building_block_energy.py reports the FULL SM silicon's
    #     per-clock energy at steady state, including the idle sub-
    #     units' clock-tree + leakage.
    # A 50% cross-check tolerance accommodates this structural
    # difference; the two still anchor each other.

    def test_sm_pj_per_mac_agrees_with_per_mac_view(self):
        sm = build_nvidia_sm_building_block()
        mar = mar_default_report()
        sm_permac = next(b for b in mar.blocks
                         if "Multiprocessor" in b.building_block)
        delta = (abs(sm.derived_pj_per_mac - sm_permac.total_pj_per_mac)
                 / sm_permac.total_pj_per_mac)
        assert delta < 0.50, (
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
        # 128 tiles, 576 MACs/clk at the tile's native clock (1.5 GHz).
        macs_per_sec = 128 * 576 * tile.clock_ghz * 1e9
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
