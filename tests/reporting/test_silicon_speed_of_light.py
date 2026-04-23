"""Tests for the silicon speed-of-light analysis."""
from __future__ import annotations

import math

from graphs.reporting.silicon_speed_of_light import (
    AccumMode,
    BareALU,
    DEFAULT_DIE_AREA_MM2,
    DEFAULT_SILICON_CLOCK_GHZ,
    DEFAULT_TDP_TARGETS_W,
    PROCESS_DENSITY_MT_PER_MM2,
    DotProductALU,
    ProductReference,
    ReuseTopology,
    SoLAnalysis,
    build_default_sol_report,
    default_alu_catalog,
    default_product_references,
    generate_parametric_curve,
    parametric_dot_product_alu,
    process_density_mt_per_mm2,
    render_alu_instance_table,
    render_alu_per_mac_table,
    render_gap_to_products_table,
    render_parametric_curve_table,
    render_sol_summary_table,
    render_tdp_sweep_table,
    render_tradeoff_chart_js,
)


class TestDensityInvariant:
    """On a fixed die at a fixed process node, the invariant

        die_area / ALU_area  ==  die_transistors / ALU_transistors

    must hold. That requires every ALU at that node to share the
    same transistor density. This class verifies the invariant."""

    def test_every_catalog_alu_matches_process_density(self):
        for a in default_alu_catalog():
            expected_density = process_density_mt_per_mm2(a.process_nm)
            actual_density = a.transistor_count_k / 1000.0 / a.area_mm2
            rel = abs(actual_density - expected_density) / expected_density
            assert rel < 0.06, (
                f"{a.name} has density {actual_density:.1f} MT/mm² but "
                f"process node {a.process_nm} nm canonical density is "
                f"{expected_density:.1f} MT/mm²"
            )

    def test_same_process_archetypes_have_equal_die_transistors(self):
        """A 250 mm² die at 8 nm has the same total transistor count
        regardless of which 8 nm ALU archetype is tiled across it."""
        catalog = default_alu_catalog()
        eight_nm = [a for a in catalog if a.process_nm == 8]
        assert len(eight_nm) >= 2
        die = 250.0
        totals = []
        for a in eight_nm:
            n = int(die / a.area_mm2)
            totals.append(n * a.transistor_count_k / 1000.0)
        spread = (max(totals) - min(totals)) / max(totals)
        assert spread < 0.02, (
            f"8 nm die-transistor totals: {totals}, spread "
            f"{spread*100:.1f}% exceeds 2% tolerance"
        )

    def test_die_transistors_equals_die_area_times_density(self):
        for a in default_alu_catalog():
            analysis = SoLAnalysis(alu=a, die_area_mm2=250.0)
            density = process_density_mt_per_mm2(a.process_nm)
            expected = 250.0 * density
            assert abs(
                analysis.die_transistor_count_m - expected
            ) / expected < 0.02

    def test_n_alus_from_area_equals_n_alus_from_transistors(self):
        """The two equivalent formulas must agree for every
        archetype."""
        die_area = 250.0
        for a in default_alu_catalog():
            from_area = int(die_area / a.area_mm2)
            density = process_density_mt_per_mm2(a.process_nm)
            die_trans = die_area * density  # in M
            from_trans = int(die_trans * 1000 / a.transistor_count_k)
            # Allow 1-ALU slack from truncation
            assert abs(from_area - from_trans) <= 1, (
                f"{a.name}: N from area={from_area}, "
                f"N from transistors={from_trans}"
            )

    def test_post_init_derives_area_when_zero(self):
        a = DotProductALU(
            name="t", precision="INT8", process_nm=8,
            area_mm2=0.0, transistor_count_k=80.0,
            pj_per_clock=1.0,
        )
        # 80 K / 80 MT/mm² = 0.001 mm²
        assert abs(a.area_mm2 - 0.001) < 1e-9

    def test_post_init_overrides_inconsistent_area(self):
        """A caller-supplied area that disagrees with the canonical
        density by more than 5% gets replaced to preserve the
        invariant."""
        a = DotProductALU(
            name="t", precision="INT8", process_nm=8,
            area_mm2=0.005,       # grossly inconsistent
            transistor_count_k=80.0,
            pj_per_clock=1.0,
        )
        # Should be recomputed to 80 K / 80 MT/mm² = 0.001 mm²
        assert abs(a.area_mm2 - 0.001) < 1e-9


class TestBareALUCeiling:
    def test_tops_per_watt_is_ops_per_pj(self):
        """Silicon ceiling is ops_per_mac / pJ_per_MAC. Clock-
        independent, count-independent."""
        alu = BareALU(
            name="test", precision="INT8", process_nm=8,
            area_mm2=0.0001, transistor_count_k=10.0,
            pj_per_clock=0.050, ops_per_mac=2,
        )
        assert abs(alu.tops_per_watt_ceiling - 40.0) < 1e-6

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

    def test_default_fields_keep_bareaLU_backwards_compat(self):
        """Constructing BareALU with only the original fields still
        works: W defaults to 1, accum_mode to LOSSLESS, reuse to
        ISOLATED, both bandwidth fields to 2.0 B/MAC."""
        alu = BareALU(
            name="legacy", precision="INT8", process_nm=8,
            area_mm2=0.0001, transistor_count_k=10.0,
            pj_per_clock=0.050,
        )
        assert alu.W == 1
        assert alu.accum_mode is AccumMode.LOSSLESS
        assert alu.reuse is ReuseTopology.ISOLATED
        assert alu.bytes_per_mac_alu == 2.0
        assert alu.bytes_per_mac_die == 2.0
        # deprecated alias still works
        assert alu.bytes_per_mac == alu.bytes_per_mac_die


class TestDotProductALUW:
    """Per-instance and per-MAC semantics for W > 1."""

    def _tc_like(self) -> DotProductALU:
        """Synthetic W=16 INT8 dot-product ALU."""
        return DotProductALU(
            name="tc-like", precision="INT8", process_nm=8,
            area_mm2=0.00137,        # per instance
            transistor_count_k=110.0,
            pj_per_clock=1.28,       # per instance, all 16 MACs firing
            W=16,
            accum_mode=AccumMode.MIXED_PRECISION,
            reuse=ReuseTopology.INTRA_ALU_BROADCAST,
            bytes_per_mac_alu=2.0,    # 2 * 1 B (INT8) per MAC, regardless of W
            bytes_per_mac_die=0.125,  # 16x16 matmul reuses frags -> 2/16 B/MAC
        )

    def test_per_mac_is_per_instance_divided_by_W(self):
        alu = self._tc_like()
        assert math.isclose(alu.area_per_mac_mm2, alu.area_mm2 / 16.0)
        assert math.isclose(
            alu.transistor_count_per_mac_k,
            alu.transistor_count_k / 16.0,
        )
        assert math.isclose(alu.pj_per_mac, alu.pj_per_clock / 16.0)

    def test_tops_per_watt_uses_per_mac_energy(self):
        alu = self._tc_like()
        expected = 2.0 / alu.pj_per_mac
        assert math.isclose(alu.tops_per_watt_ceiling, expected)


class TestParametricModel:
    def test_curve_covers_all_widths(self):
        curve = generate_parametric_curve()
        assert [p.W for p in curve] == [1, 2, 4, 8, 16, 32, 64]

    def test_transistor_per_mac_grows_weakly_with_W(self):
        """Per-MAC transistor count should be nearly flat, with a
        mild rise as reduction-tree bit widths grow past W=8."""
        curve = generate_parametric_curve()
        first = curve[0].transistor_count_per_mac_k
        last = curve[-1].transistor_count_per_mac_k
        # First (W=1) is <= 6 K and last (W=64) is <= 8 K
        assert 4.0 <= first <= 7.0
        assert 6.0 <= last <= 9.0
        # Growth is positive but modest
        assert last > first
        assert last < 2.0 * first

    def test_bytes_per_mac_alu_is_constant_across_W(self):
        """A dot-product ALU of width W reads W A-operands + W B-
        operands per clock and produces W MACs - 2 bytes/MAC at the
        ALU level, regardless of W. There is NO reuse inside a
        single dot-product ALU; operand-reuse savings happen at the
        TOPOLOGY level (mesh, systolic, cross-ALU broadcast)."""
        curve = generate_parametric_curve(precision="INT8")
        for p in curve:
            # All INT8 parametric ALUs have bytes_per_mac_alu = 2.0
            assert math.isclose(p.bytes_per_mac_alu, 2.0, rel_tol=1e-6), (
                f"W={p.W} gave ALU-level {p.bytes_per_mac_alu} B/MAC"
            )

    def test_bytes_per_mac_alu_scales_with_precision(self):
        """FP32 parametric ALU reads 2 * 4 = 8 B/MAC at ALU level."""
        a = parametric_dot_product_alu(W=1, precision="FP32")
        assert math.isclose(a.bytes_per_mac_alu, 8.0, rel_tol=1e-6)

    def test_bytes_per_mac_die_defaults_to_alu_level(self):
        """Parametric model does not assume any topology, so die-
        level bandwidth equals ALU-level bandwidth."""
        for p in generate_parametric_curve():
            assert math.isclose(p.bytes_per_mac_alu, p.bytes_per_mac_die)

    def test_parametric_reports_is_parametric_true(self):
        p = parametric_dot_product_alu(W=4)
        assert p.is_parametric is True

    def test_catalog_entries_are_not_parametric(self):
        for a in default_alu_catalog():
            assert a.is_parametric is False


class TestSoLAnalysis:
    def _kpu_like(self) -> DotProductALU:
        return DotProductALU(
            name="KPU INT8", precision="INT8", process_nm=8,
            area_mm2=0.000185, transistor_count_k=12.0,
            pj_per_clock=0.050,
            W=1, accum_mode=AccumMode.LOSSLESS,
            reuse=ReuseTopology.MESH_STREAMING,
            bytes_per_mac_alu=2.0, bytes_per_mac_die=0.0625,
        )

    def test_num_alus_scales_with_die_area(self):
        a = SoLAnalysis(alu=self._kpu_like(), die_area_mm2=100.0)
        b = SoLAnalysis(alu=self._kpu_like(), die_area_mm2=200.0)
        assert abs(b.num_alus - 2 * a.num_alus) <= 1

    def test_num_macs_on_die_is_num_alus_times_W(self):
        """For a W>1 ALU, total MACs on die = instances * W."""
        tc = DotProductALU(
            name="tc", precision="INT8", process_nm=8,
            area_mm2=0.00137, transistor_count_k=110.0,
            pj_per_clock=1.28, W=16,
            accum_mode=AccumMode.MIXED_PRECISION,
            reuse=ReuseTopology.INTRA_ALU_BROADCAST,
            bytes_per_mac_alu=2.0, bytes_per_mac_die=0.125,
        )
        analysis = SoLAnalysis(alu=tc, die_area_mm2=250.0)
        assert analysis.num_macs_on_die == analysis.num_alus * 16

    def test_peak_tops_scales_linearly_with_clock(self):
        a = SoLAnalysis(alu=self._kpu_like())
        assert math.isclose(a.peak_tops(2.0), 2 * a.peak_tops(1.0),
                            rel_tol=1e-9)

    def test_die_power_scales_linearly_with_clock(self):
        a = SoLAnalysis(alu=self._kpu_like())
        assert math.isclose(a.die_power_w(2.0), 2 * a.die_power_w(1.0),
                            rel_tol=1e-9)

    def test_tops_per_watt_invariant_across_clocks(self):
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


class TestDefaultCatalog:
    def test_catalog_covers_design_space(self):
        """Catalog has at least one entry at W=1 (bare FMA), one at
        W>=4 (dot-product), and at least one of each accum_mode."""
        cat = default_alu_catalog()
        assert any(a.W == 1 for a in cat)
        assert any(a.W >= 4 for a in cat)
        modes = {a.accum_mode for a in cat}
        assert AccumMode.LOSSLESS in modes
        assert AccumMode.MIXED_PRECISION in modes

    def test_catalog_covers_reuse_topologies(self):
        cat = default_alu_catalog()
        reuses = {a.reuse for a in cat}
        assert ReuseTopology.MESH_STREAMING in reuses
        assert ReuseTopology.INTRA_ALU_BROADCAST in reuses
        assert ReuseTopology.ISOLATED in reuses
        assert ReuseTopology.SYSTOLIC_STATIONARY in reuses

    def test_fp32_ceiling_much_lower_than_int8(self):
        catalog = default_alu_catalog()
        kpu = next(a for a in catalog if "KPU PE (bare" in a.name)
        fp32 = next(a for a in catalog if a.precision == "FP32")
        assert kpu.tops_per_watt_ceiling > 20 * fp32.tops_per_watt_ceiling


class TestDefaultReport:
    def test_build_default_populates_curve_and_products(self):
        r = build_default_sol_report()
        assert len(r.alus) >= 5
        assert len(r.parametric_curve) >= 5
        assert len(r.products) >= 2
        assert r.die_area_mm2 == DEFAULT_DIE_AREA_MM2

    def test_orin_actual_is_well_below_sol_at_30w(self):
        r = build_default_sol_report()
        orin = next(p for p in r.products
                    if "30 W" in p.name and "dense" in p.name)
        kpu_analysis = next(
            a for a in r.analyses if "KPU PE (bare" in a.alu.name
        )
        f_30w = kpu_analysis.clock_for_tdp(30.0)
        sol_30w_tops = kpu_analysis.peak_tops(
            min(f_30w, r.silicon_clock_ghz)
        )
        assert orin.peak_int8_tops < sol_30w_tops
        ratio = orin.peak_int8_tops / sol_30w_tops
        assert ratio < 0.15, (
            f"Orin 30 W actual/SoL is {ratio*100:.0f}%; "
            "should be ~6% against bare-FMA SoL."
        )

    def test_orin_tops_per_watt_roughly_constant_across_modes(self):
        r = build_default_sol_report()
        orins = [p for p in r.products
                 if "Orin" in p.name and "dense" in p.name]
        assert len(orins) >= 2
        ratios = [p.tops_per_watt for p in orins]
        assert (max(ratios) - min(ratios)) / max(ratios) < 0.15

    def test_product_catalog_is_embedded_only(self):
        """Gap-to-product table scope is embedded AI at 8 nm /
        5-60 W TDP class. Datacenter parts (H100, MI300, TPU v4
        pod, etc.) should not regress into the default catalog -
        they belong in a separate datacenter-specific comparison."""
        for p in default_product_references():
            assert p.tdp_w <= 60.0, (
                f"{p.name} at {p.tdp_w} W exceeds embedded-AI "
                "TDP class"
            )
            assert p.process_nm == 8, (
                f"{p.name} at {p.process_nm} nm is not 8 nm - "
                "mixed-process references muddy the comparison"
            )
            assert p.die_area_mm2 <= 250.0, (
                f"{p.name} die is {p.die_area_mm2} mm² - too "
                "large for the embedded-AI class"
            )


class TestHTMLRendering:
    def test_alu_instance_table_shows_W_and_topology(self):
        r = build_default_sol_report()
        html = render_alu_instance_table(r.alus)
        for alu in r.alus:
            assert alu.name in html
        # Topology labels appear
        assert "mesh" in html.lower() or "broadcast" in html.lower()

    def test_alu_per_mac_table_reports_tops_per_watt(self):
        r = build_default_sol_report()
        html = render_alu_per_mac_table(r.alus)
        assert "TOPS / W" in html or "TOPS/W" in html

    def test_parametric_curve_table_has_rows_for_each_width(self):
        r = build_default_sol_report()
        html = render_parametric_curve_table(r.parametric_curve)
        for p in r.parametric_curve:
            # Each width appears at least once (as <strong>N</strong>
            # in the first cell)
            assert f"<strong>{p.W}</strong>" in html

    def test_tradeoff_chart_js_emits_three_panels(self):
        r = build_default_sol_report()
        js = render_tradeoff_chart_js(r.parametric_curve, r.alus)
        assert "chart_sol_trans_per_mac" in js
        assert "chart_sol_pj_per_mac" in js
        assert "chart_sol_bytes_per_mac" in js

    def test_sol_summary_shows_die_area(self):
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
        for p in r.products:
            assert p.name in html
        assert "%" in html
