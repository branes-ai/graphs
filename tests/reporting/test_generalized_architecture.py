"""Tests for the generalized-architecture comparison module."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.generalized_architecture import (
    CANONICAL_ARCHETYPES,
    DEFAULT_PROCESS_NODES,
    GeneralizedArchetype,
    build_default_report,
    compute_components_pj_per_mac,
    render_generalized_page,
    total_pj_per_mac,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestArchetypeDefinitions:
    def test_canonical_set_covers_expected_categories(self):
        cats = {a.category for a in CANONICAL_ARCHETYPES}
        for c in ("CPU", "GPU", "TPU", "KPU", "DSP", "DFM", "CGRA"):
            assert c in cats, f"missing category {c}"

    def test_every_archetype_has_positive_alu(self):
        for a in CANONICAL_ARCHETYPES:
            assert a.alu_fa_eq > 0, f"{a.name} has zero ALU FA-eq"

    def test_ordering_by_floor_at_16nm_is_stable(self):
        ranked = sorted(CANONICAL_ARCHETYPES,
                        key=lambda a: total_pj_per_mac(a, 16))
        names_in_order = [a.name for a in ranked]
        assert "TPU" in names_in_order[0] or "CGRA" in names_in_order[0]
        assert "CPU" in names_in_order[-1]
        tpu_e = next(total_pj_per_mac(a, 16) for a in CANONICAL_ARCHETYPES
                     if a.category == "TPU")
        kpu_e = next(total_pj_per_mac(a, 16) for a in CANONICAL_ARCHETYPES
                     if a.category == "KPU")
        assert tpu_e < kpu_e


class TestProcessScaling:
    def test_energy_decreases_with_process_shrink(self):
        for a in CANONICAL_ARCHETYPES:
            nodes = sorted(DEFAULT_PROCESS_NODES, reverse=True)
            energies = [total_pj_per_mac(a, n) for n in nodes]
            for prev, curr in zip(energies[:-1], energies[1:]):
                assert curr <= prev, (
                    f"{a.name}: {prev:.3f} -> {curr:.3f} at smaller node")

    def test_ratios_roughly_preserved_across_nodes(self):
        ordering_at_16 = sorted(
            CANONICAL_ARCHETYPES, key=lambda a: total_pj_per_mac(a, 16))
        ordering_at_5 = sorted(
            CANONICAL_ARCHETYPES, key=lambda a: total_pj_per_mac(a, 5))
        names_16 = [a.name for a in ordering_at_16]
        names_5 = [a.name for a in ordering_at_5]
        assert names_16 == names_5


class TestComponentBreakdown:
    def test_components_sum_to_total(self):
        for a in CANONICAL_ARCHETYPES:
            for nm in (28, 16, 5):
                comps = compute_components_pj_per_mac(a, nm)
                total = total_pj_per_mac(a, nm)
                assert abs(sum(comps.values()) - total) < 1e-9

    def test_tpu_has_no_fetch_or_schedule_overhead(self):
        tpu = next(a for a in CANONICAL_ARCHETYPES if a.category == "TPU")
        comps = compute_components_pj_per_mac(tpu, 16)
        assert comps["Instruction fetch"] == 0
        assert comps["Decode"] == 0
        assert comps["Scheduling / coherence"] == 0

    def test_kpu_has_no_fetch_decode_schedule_overhead(self):
        kpu = next(a for a in CANONICAL_ARCHETYPES if a.category == "KPU")
        comps = compute_components_pj_per_mac(kpu, 16)
        assert comps["Instruction fetch"] == 0
        assert comps["Decode"] == 0
        assert comps["Scheduling / coherence"] == 0

    def test_cpu_has_dominant_fetch_decode_overhead(self):
        cpu = next(a for a in CANONICAL_ARCHETYPES if a.category == "CPU")
        comps = compute_components_pj_per_mac(cpu, 16)
        overhead = (comps["Instruction fetch"] + comps["Decode"]
                    + comps["Scheduling / coherence"])
        assert overhead > comps["ALU"]


class TestReportAndHTML:
    def test_default_report_uses_reference_16nm(self):
        rpt = build_default_report()
        assert rpt.reference_process_nm == 16
        assert rpt.power_budget_w == 12.0
        assert len(rpt.archetypes) == len(CANONICAL_ARCHETYPES)

    def test_html_renders_three_charts_and_table(self):
        rpt = build_default_report()
        html = render_generalized_page(rpt, REPO_ROOT)
        assert 'id="chart_same_process"' in html
        assert 'id="chart_process_scaling"' in html
        assert 'id="chart_tdp_capability"' in html
        assert 'class="generalized"' in html

    def test_html_cross_links_to_native_op_page(self):
        rpt = build_default_report()
        html = render_generalized_page(rpt, REPO_ROOT)
        assert 'native_op_energy.html' in html

    def test_html_mentions_all_archetype_categories(self):
        rpt = build_default_report()
        html = render_generalized_page(rpt, REPO_ROOT)
        for cat in ("CPU", "GPU", "TPU", "KPU", "DSP", "DFM", "CGRA"):
            assert cat in html

    def test_html_back_link_to_index(self):
        rpt = build_default_report()
        html = render_generalized_page(rpt, REPO_ROOT)
        assert 'href="index.html"' in html
