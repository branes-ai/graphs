"""Tests for the silicon speed-of-light analysis."""
from __future__ import annotations

import math

from graphs.reporting.silicon_speed_of_light import (
    BareALU,
    DEFAULT_DIE_AREA_MM2,
    DEFAULT_SILICON_CLOCK_GHZ,
    DEFAULT_TDP_TARGETS_W,
    ProductReference,
    SoLAnalysis,
    build_default_sol_report,
    default_alu_catalog,
    default_product_references,
    render_alu_catalog_table,
    render_gap_to_products_table,
    render_sol_summary_table,
    render_tdp_sweep_table,
)


class TestBareALUCeiling:
    def test_tops_per_watt_is_ops_per_pj(self):
        """The fundamental silicon ceiling is ops_per_mac / pJ_per_MAC.
        Clock-independent, count-independent."""
        kpu = BareALU(
            name="test", precision="INT8", process_nm=8,
            area_mm2=0.0001, transistor_count_k=10.0,
            pj_per_clock=0.050, ops_per_mac=2,
        )
        assert abs(kpu.tops_per_watt_ceiling - 40.0) < 1e-6

    def test_lower_energy_gives_higher_ceiling(self):
        a = BareALU(
            name="cheap", precision="INT8", process_nm=8,
            area_mm2=0.0001, transistor_count_k=10.0,
            pj_per_clock=0.05,
        )
        b = BareALU(
            name="dear", precision="INT8", process_nm=8,
            area_mm2=0.0001, transistor_count_k=10.0,
            pj_per_clock=0.10,
        )
        assert a.tops_per_watt_ceiling > b.tops_per_watt_ceiling


class TestSoLAnalysis:
    def _kpu_like(self) -> BareALU:
        return BareALU(
            name="KPU INT8", precision="INT8", process_nm=8,
            area_mm2=0.000185, transistor_count_k=12.0,
            pj_per_clock=0.050, ops_per_mac=2,
        )

    def test_num_alus_scales_with_die_area(self):
        a = SoLAnalysis(alu=self._kpu_like(), die_area_mm2=100.0)
        b = SoLAnalysis(alu=self._kpu_like(), die_area_mm2=200.0)
        # int() truncation can leave a 1-ALU rounding slack
        assert abs(b.num_alus - 2 * a.num_alus) <= 1

    def test_peak_tops_scales_linearly_with_clock(self):
        a = SoLAnalysis(alu=self._kpu_like())
        assert math.isclose(a.peak_tops(2.0), 2 * a.peak_tops(1.0),
                            rel_tol=1e-9)

    def test_die_power_scales_linearly_with_clock(self):
        a = SoLAnalysis(alu=self._kpu_like())
        assert math.isclose(a.die_power_w(2.0), 2 * a.die_power_w(1.0),
                            rel_tol=1e-9)

    def test_tops_per_watt_invariant_across_clocks(self):
        """Proof that TOPS/W is truly a silicon ceiling, not an
        operating-point artifact."""
        a = SoLAnalysis(alu=self._kpu_like())
        ratios = [a.peak_tops(f) / a.die_power_w(f)
                  for f in (0.1, 0.5, 1.0, 1.5)]
        for r in ratios[1:]:
            assert math.isclose(r, ratios[0], rel_tol=1e-9)
        assert math.isclose(ratios[0], a.alu.tops_per_watt_ceiling,
                            rel_tol=1e-9)

    def test_clock_for_tdp_matches_die_power(self):
        a = SoLAnalysis(alu=self._kpu_like())
        for tdp in (5.0, 30.0, 150.0):
            f = a.clock_for_tdp(tdp)
            assert math.isclose(a.die_power_w(f), tdp, rel_tol=1e-9)

    def test_tdp_sweep_flags_silicon_limited(self):
        """At high enough TDP, the die saturates at silicon_clock_ghz
        and the sweep marks the row silicon-limited."""
        a = SoLAnalysis(
            alu=self._kpu_like(),
            die_area_mm2=DEFAULT_DIE_AREA_MM2,
            silicon_clock_ghz=1.5,
        )
        swept = a.tdp_sweep()
        # The highest TDP in the default sweep (300 W) should be
        # silicon-limited for the KPU (100 W at 1.5 GHz).
        high = [r for r in swept if r["tdp_w"] == 300.0]
        assert high, "300 W row missing from sweep"
        assert high[0]["silicon_limited"] is True
        assert math.isclose(high[0]["effective_clock_ghz"], 1.5,
                            rel_tol=1e-9)


class TestDefaultCatalog:
    def test_catalog_includes_both_int8_paths(self):
        catalog = default_alu_catalog()
        names = [a.name for a in catalog]
        assert any("KPU" in n for n in names)
        assert any("Tensor" in n or "TC" in n for n in names)

    def test_kpu_ceiling_higher_than_tc_ceiling(self):
        """The KPU bare MAC is slightly more energy-efficient than
        the Ampere TC bare MAC at matched process, so its TOPS/W
        ceiling is higher."""
        catalog = default_alu_catalog()
        kpu = next(a for a in catalog if "KPU" in a.name)
        tc = next(a for a in catalog
                  if "TC" in a.name or "Tensor" in a.name)
        assert kpu.tops_per_watt_ceiling > tc.tops_per_watt_ceiling

    def test_fp32_ceiling_much_lower_than_int8(self):
        """FP32 FMA is ~50x more expensive per op than INT8 MAC at
        the same process, so its ceiling collapses accordingly."""
        catalog = default_alu_catalog()
        kpu = next(a for a in catalog if "KPU" in a.name)
        fp32 = next(a for a in catalog if a.precision == "FP32")
        assert kpu.tops_per_watt_ceiling > 20 * fp32.tops_per_watt_ceiling


class TestDefaultReport:
    def test_build_default_returns_populated_report(self):
        r = build_default_sol_report()
        assert len(r.alus) >= 3
        assert len(r.analyses) == len(r.alus)
        assert len(r.products) >= 2
        assert r.die_area_mm2 == DEFAULT_DIE_AREA_MM2
        assert r.silicon_clock_ghz == DEFAULT_SILICON_CLOCK_GHZ

    def test_orin_actual_is_well_below_sol_at_30w(self):
        """Product-gap invariant: Orin AGX at 30 W (dense INT8 ~68 TOPS)
        must be well below the silicon-capability SoL for a 30 W
        KPU-style die - this is the innovation headroom we are
        showing. Expect Orin to be under 15% of SoL."""
        r = build_default_sol_report()
        orin = next(p for p in r.products
                    if "30 W" in p.name and "dense" in p.name)
        kpu_analysis = next(a for a in r.analyses if "KPU" in a.alu.name)
        f_30w = kpu_analysis.clock_for_tdp(30.0)
        sol_30w_tops = kpu_analysis.peak_tops(
            min(f_30w, r.silicon_clock_ghz)
        )
        assert orin.peak_int8_tops < sol_30w_tops
        ratio = orin.peak_int8_tops / sol_30w_tops
        assert ratio < 0.15, (
            f"Orin 30 W actual/SoL is {ratio*100:.0f}%; the "
            "intended headline is ~6% (17x headroom)."
        )

    def test_orin_tops_per_watt_roughly_constant_across_modes(self):
        """Dense-INT8 TOPS/W should be roughly flat across Orin's
        configurable power modes (silicon property, not mode
        property)."""
        r = build_default_sol_report()
        orins = [p for p in r.products
                 if "Orin" in p.name and "dense" in p.name]
        assert len(orins) >= 2
        ratios = [p.tops_per_watt for p in orins]
        # Spread shouldn't exceed ~15% - the data actually shows
        # ~2.1 to ~2.3 TOPS/W across modes.
        assert (max(ratios) - min(ratios)) / max(ratios) < 0.15


class TestHTMLRendering:
    def test_alu_catalog_table_contains_every_alu(self):
        r = build_default_sol_report()
        html = render_alu_catalog_table(r.alus)
        for alu in r.alus:
            assert alu.name in html

    def test_sol_summary_table_has_die_area_header(self):
        r = build_default_sol_report()
        html = render_sol_summary_table(r.analyses, r.die_area_mm2)
        assert f"{r.die_area_mm2:.0f}" in html

    def test_tdp_sweep_table_lists_every_target(self):
        r = build_default_sol_report()
        html = render_tdp_sweep_table(r.analyses, DEFAULT_TDP_TARGETS_W)
        for tdp in DEFAULT_TDP_TARGETS_W:
            assert f"{tdp:.0f} W" in html

    def test_gap_to_products_table_shows_ratios(self):
        r = build_default_sol_report()
        html = render_gap_to_products_table(r.products, r.analyses)
        # Each product appears as a row
        for p in r.products:
            assert p.name in html
        # Percentage suffix indicates the actual/SoL ratio
        assert "%" in html
