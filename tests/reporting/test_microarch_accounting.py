"""Tests for per-structure micro-architectural energy accounting."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.microarch_accounting import (
    StructureCategory,
    build_default_report,
    build_kpu_tile_accounting,
    build_nvidia_sm_accounting,
    render_accounting_page,
)
from graphs.reporting.generalized_architecture import (
    CANONICAL_ARCHETYPES,
    total_pj_per_mac,
)
from graphs.reporting.native_op_energy import _fa_pj


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestStructureInvariants:
    def test_every_sm_structure_has_positive_energy(self):
        sm = build_nvidia_sm_accounting()
        for s in sm.structures:
            assert s.per_op_pj > 0, f"{s.name} has non-positive energy"
            assert s.amortization_factor >= 1
            assert s.per_mac_pj > 0

    def test_every_kpu_structure_has_positive_energy(self):
        kpu = build_kpu_tile_accounting()
        for s in kpu.structures:
            assert s.per_op_pj > 0, f"{s.name} has non-positive energy"
            assert s.amortization_factor >= 1
            assert s.per_mac_pj > 0

    def test_every_structure_has_citation(self):
        for b in build_default_report().blocks:
            for s in b.structures:
                assert s.citation, f"{s.name} missing citation"
                assert len(s.citation) > 20, (
                    f"{s.name} citation too short: {s.citation!r}")

    def test_every_structure_has_transistor_count(self):
        for b in build_default_report().blocks:
            for s in b.structures:
                assert s.transistor_count_m > 0, (
                    f"{s.name} missing transistor count")

    def test_sm_unique_silicon_in_hmma_path_range(self):
        """SM's per-MAC view itemizes the HMMA path; total unique
        silicon should be a large fraction of the ~240 M full SM
        but less than all of it (non-HMMA sub-units like RT cores
        and texture are not itemized in the per-MAC view)."""
        sm = build_default_report().blocks[0]
        assert 100 < sm.total_unique_transistor_count_m < 240

    def test_kpu_unique_silicon_matches_building_block_view(self):
        """Per-MAC and per-clock views should agree on KPU tile
        silicon footprint to within a few M transistors. At 32x32
        the per-MAC view lands around 22 M; the per-clock view is
        close (both scale per-PE counts the same way)."""
        kpu = build_default_report().blocks[1]
        assert 18 < kpu.total_unique_transistor_count_m < 30


class TestStructuralPresenceAbsence:
    def test_sm_has_fetch_decode_schedule(self):
        sm = build_nvidia_sm_accounting()
        cats = {s.category for s in sm.structures}
        assert StructureCategory.FETCH in cats
        assert StructureCategory.DECODE in cats
        assert StructureCategory.SCHEDULE in cats

    def test_kpu_tile_lacks_fetch_and_decode(self):
        """Domain-flow fabric has no instruction fetch or decode."""
        kpu = build_kpu_tile_accounting()
        cats = {s.category for s in kpu.structures}
        assert StructureCategory.FETCH not in cats
        assert StructureCategory.DECODE not in cats

    def test_both_blocks_have_execute(self):
        for b in build_default_report().blocks:
            cats = {s.category for s in b.structures}
            assert StructureCategory.EXECUTE in cats

    def test_kpu_tile_has_no_router_or_rom(self):
        """Guard against regressing to hallucinated structures.
        KPU tile is a 2D mesh of FMAs; no packet router, no schedule
        ROM, no micro-sequencer counter."""
        kpu = build_kpu_tile_accounting()
        for s in kpu.structures:
            name_lower = s.name.lower()
            assert "router" not in name_lower, (
                f"KPU tile should not contain a router: {s.name}")
            assert "rom" not in name_lower, (
                f"KPU tile should not contain a ROM: {s.name}")
            assert "schedule counter" not in name_lower, (
                f"KPU tile should not contain a schedule counter: {s.name}")

    def test_kpu_tile_throughput_is_pe_count(self):
        """Native-op throughput basis for a 2D mesh is PE count per
        clock, not some invented wavefront total."""
        kpu = build_kpu_tile_accounting()
        assert kpu.macs_per_native_op == 1024, (
            f"32x32 mesh does 1024 MACs per clock, got "
            f"{kpu.macs_per_native_op}")


class TestTotals:
    def test_dynamic_total_is_sum_of_structures(self):
        sm = build_nvidia_sm_accounting()
        expected = sum(s.per_mac_pj for s in sm.structures)
        assert abs(sm.dynamic_pj_per_mac - expected) < 1e-9

    def test_total_includes_leakage(self):
        sm = build_nvidia_sm_accounting()
        assert sm.total_pj_per_mac > sm.dynamic_pj_per_mac
        expected = sm.dynamic_pj_per_mac * (1.0 + sm.leakage_fraction)
        assert abs(sm.total_pj_per_mac - expected) < 1e-9

    def test_physical_plausibility(self):
        """Per-MAC energies must be in a sensible range for modern silicon."""
        for b in build_default_report().blocks:
            assert 0.05 < b.total_pj_per_mac < 2.0, (
                f"{b.building_block}: {b.total_pj_per_mac:.3f} pJ/MAC "
                "outside plausible range [0.05, 2.0]")


class TestCrossValidation:
    """Detailed view totals should agree with the simplified
    architectural-efficiency model within ~60%. The detailed view adds
    leakage and itemizes small structures (clock tree, token-match)
    the simplified view rolls up."""

    def test_sm_agrees_with_tensor_core_archetype(self):
        r = build_default_report()
        tc = next(a for a in CANONICAL_ARCHETYPES
                  if a.category == "GPU" and "Tensor Core" in a.name)
        sm = r.blocks[0]
        sm_norm = sm.total_pj_per_mac * _fa_pj(16) / _fa_pj(sm.process_nm)
        simp = total_pj_per_mac(tc, 16)
        delta = abs(sm_norm - simp) / simp
        assert delta < 0.60, (
            f"SM cross-validation delta {delta*100:.0f}% exceeds 60%: "
            f"detailed {sm_norm:.3f} vs simplified {simp:.3f}")

    def test_kpu_agrees_with_domain_flow_archetype(self):
        r = build_default_report()
        kpu_a = next(a for a in CANONICAL_ARCHETYPES if a.category == "KPU")
        kpu_det = r.blocks[1].total_pj_per_mac
        simp = total_pj_per_mac(kpu_a, 16)
        delta = abs(kpu_det - simp) / simp
        assert delta < 0.60, (
            f"KPU cross-validation delta {delta*100:.0f}% exceeds 60%: "
            f"detailed {kpu_det:.3f} vs simplified {simp:.3f}")

    def test_sm_higher_energy_than_kpu_at_matched_process(self):
        """The architectural advantage must survive matched-process
        comparison. Both blocks are already at 8 nm (see
        test_html_has_process_nodes), so no normalization is needed
        - compare raw totals."""
        r = build_default_report()
        sm = r.blocks[0]
        kpu = r.blocks[1]
        assert sm.process_nm == kpu.process_nm, (
            "Cross-validation requires matched process; got "
            f"SM={sm.process_nm} nm, KPU={kpu.process_nm} nm"
        )
        assert sm.total_pj_per_mac > kpu.total_pj_per_mac, (
            f"SM @ {sm.process_nm}nm ({sm.total_pj_per_mac:.3f}) not "
            f"higher than KPU ({kpu.total_pj_per_mac:.3f}) - "
            "architectural advantage lost"
        )


class TestHTMLRender:
    def test_html_has_both_blocks(self):
        r = build_default_report()
        html = render_accounting_page(r, REPO_ROOT)
        assert "NVIDIA Streaming Multiprocessor" in html
        assert "KPU Compute Tile" in html

    def test_html_has_two_charts(self):
        r = build_default_report()
        html = render_accounting_page(r, REPO_ROOT)
        assert 'id="chart_by_category"' in html
        assert 'id="chart_linear"' in html

    def test_html_has_citations(self):
        r = build_default_report()
        html = render_accounting_page(r, REPO_ROOT)
        assert "Horowitz" in html
        assert "domain-flow" in html.lower() or "domain flow" in html.lower()

    def test_html_back_link_to_index(self):
        r = build_default_report()
        html = render_accounting_page(r, REPO_ROOT)
        assert 'href="index.html"' in html

    def test_html_has_process_nodes(self):
        """Both building blocks are now reported at 8 nm (matched to
        the Jetson Ampere baseline so direct comparisons need no
        renormalization)."""
        r = build_default_report()
        html = render_accounting_page(r, REPO_ROOT)
        assert "8 nm" in html

    def test_html_includes_cross_validation_note(self):
        r = build_default_report()
        html = render_accounting_page(r, REPO_ROOT)
        assert "Cross-validation" in html
        assert "generalized_architecture.py" in html
