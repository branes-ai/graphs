"""Tests for mission-capability analysis."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.mission_capability import (
    ArchProfile,
    PhysicsThreshold,
    build_default_report,
    default_mission_catalog,
    evaluate,
    gpu_profile,
    hours_enabled_curve,
    kpu_profile,
    render_mission_capability_page,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestArchProfile:
    def test_kpu_is_10x_gpu_tops_per_watt(self):
        gpu = gpu_profile()
        kpu = kpu_profile()
        assert 8.0 < kpu.tops_per_watt / gpu.tops_per_watt < 12.0

    def test_compute_power_scales_linearly(self):
        gpu = gpu_profile()
        assert abs(gpu.compute_power_w(100.0) - 100.0 / gpu.tops_per_watt) < 1e-9

    def test_compute_power_clamped_to_module_floor(self):
        arch = ArchProfile(
            name="tiny", tops_per_watt=1000.0,
            min_module_mass_g=1.0, min_module_power_w=0.5,
            module_thermal_density_w_per_cm2=0.1,
        )
        # 1 TOPS at 1000 TOPS/W = 0.001 W, but clamped to 0.5
        assert arch.compute_power_w(1.0) == 0.5


class TestEvaluate:
    def test_auv_gpu_infeasible_kpu_feasible(self):
        """30-day AUV: GPU should fail energy budget, KPU should pass."""
        catalog = default_mission_catalog()
        auv = next(m for m in catalog if "abyssal AUV" in m.name)
        gpu_r = evaluate(auv, gpu_profile())
        kpu_r = evaluate(auv, kpu_profile())
        assert not gpu_r.feasible
        assert kpu_r.feasible
        assert gpu_r.binding_violation and "energy" in gpu_r.binding_violation

    def test_nano_swarm_mass_bound(self):
        """Nano-swarm: GPU fails mass budget even before power."""
        catalog = default_mission_catalog()
        swarm = next(m for m in catalog if "nano-swarm" in m.name)
        gpu_r = evaluate(swarm, gpu_profile())
        assert not gpu_r.feasible
        assert not gpu_r.mass_feasible

    def test_exoskeleton_thermal_bound(self):
        catalog = default_mission_catalog()
        exo = next(m for m in catalog if "exoskeleton" in m.name)
        gpu_r = evaluate(exo, gpu_profile())
        kpu_r = evaluate(exo, kpu_profile())
        # GPU should violate thermal envelope
        assert not gpu_r.feasible
        assert not gpu_r.thermal_feasible
        # KPU should fit the 3 W skin-safe envelope
        assert kpu_r.feasible

    def test_hours_enabled_matches_battery_over_power(self):
        catalog = default_mission_catalog()
        auv = next(m for m in catalog if "abyssal AUV" in m.name)
        kpu = kpu_profile()
        r = evaluate(auv, kpu)
        expected = auv.battery_wh / r.total_power_w
        assert abs(r.mission_hours_enabled - expected) < 1e-6


class TestMissionCatalog:
    def test_catalog_has_10_missions(self):
        assert len(default_mission_catalog()) == 10

    def test_every_mission_has_citation(self):
        for m in default_mission_catalog():
            assert m.citation and len(m.citation) > 20

    def test_every_binding_threshold_represented(self):
        """All four physics thresholds must appear in the catalog."""
        cats = {m.binding_threshold for m in default_mission_catalog()}
        assert PhysicsThreshold.PAYLOAD_MASS in cats
        assert PhysicsThreshold.ENERGY_BUDGET in cats
        assert PhysicsThreshold.THERMAL_ENVELOPE in cats
        assert PhysicsThreshold.CONCURRENT_DENSITY in cats


class TestDefaultReport:
    def test_kpu_passes_more_missions_than_gpu(self):
        report = build_default_report()
        gpu = next(a for a in report.archs if "GPU" in a.name)
        kpu = next(a for a in report.archs if "KPU" in a.name)
        gpu_f, _ = report.feasibility_ratio(gpu)
        kpu_f, _ = report.feasibility_ratio(kpu)
        assert kpu_f > gpu_f, (
            f"KPU feasible on {kpu_f}, GPU feasible on {gpu_f}; "
            "KPU should dominate"
        )

    def test_kpu_feasible_on_at_least_seven_missions(self):
        """Investor-narrative invariant: KPU handles at least 7/10."""
        report = build_default_report()
        kpu = next(a for a in report.archs if "KPU" in a.name)
        kpu_f, total = report.feasibility_ratio(kpu)
        assert kpu_f >= 7, (
            f"KPU only feasible on {kpu_f}/{total} missions"
        )


class TestHoursEnabledCurve:
    def test_curve_monotone_decreasing_in_tops(self):
        catalog = default_mission_catalog()
        auv = next(m for m in catalog if "abyssal AUV" in m.name)
        curve = hours_enabled_curve(auv, kpu_profile())
        assert len(curve) > 10
        # strictly non-increasing
        for a, b in zip(curve[:-1], curve[1:]):
            assert b[1] <= a[1] + 1e-9

    def test_kpu_curve_above_gpu_curve(self):
        catalog = default_mission_catalog()
        auv = next(m for m in catalog if "abyssal AUV" in m.name)
        gpu_curve = hours_enabled_curve(auv, gpu_profile())
        kpu_curve = hours_enabled_curve(auv, kpu_profile())
        # At every sampled TOPS the KPU enables more hours
        for (t_g, h_g), (t_k, h_k) in zip(gpu_curve, kpu_curve):
            assert abs(t_g - t_k) < 1e-6
            assert h_k >= h_g


class TestHTMLRender:
    def test_page_contains_every_mission(self):
        report = build_default_report()
        html = render_mission_capability_page(report, REPO_ROOT)
        for m in report.missions:
            assert m.name in html

    def test_page_shows_feasibility_tally(self):
        report = build_default_report()
        html = render_mission_capability_page(report, REPO_ROOT)
        assert "Feasibility tally" in html

    def test_page_has_wattage_chart(self):
        report = build_default_report()
        html = render_mission_capability_page(report, REPO_ROOT)
        assert 'id="chart_wattage_vs_budget"' in html

    def test_page_back_link(self):
        report = build_default_report()
        html = render_mission_capability_page(report, REPO_ROOT)
        assert 'href="index.html"' in html
