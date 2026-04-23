"""Tests for the silicon composition hierarchy view."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.silicon_composition import (
    ArchitectureCategory,
    ArchitectureHierarchy,
    BlockLevel,
    CompositionBlock,
    build_arm_a78_cpu_hierarchy,
    build_default_composition_report,
    build_kpu_t128_hierarchy,
    build_nvidia_ampere_orin_hierarchy,
    build_qualcomm_hexagon_hierarchy,
    default_hierarchies,
    render_composition_page,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCompositionBlock:
    def test_tops_per_watt_is_ops_per_pj(self):
        b = CompositionBlock(
            name="test", level=BlockLevel.ALU,
            architecture=ArchitectureCategory.NPU,
            process_nm=8, children_per_instance=1,
            transistor_count_m=0.007, macs_per_clock=1,
            pj_per_clock=0.050,
        )
        # 2 ops/MAC * 1 MAC/clk / 0.050 pJ/clk = 40 TOPS/W
        assert abs(b.tops_per_watt - 40.0) < 1e-6

    def test_area_derives_from_transistors_and_process(self):
        b = CompositionBlock(
            name="test", level=BlockLevel.ALU,
            architecture=ArchitectureCategory.NPU,
            process_nm=8, children_per_instance=1,
            transistor_count_m=8.0, macs_per_clock=1,
            pj_per_clock=1.0,
        )
        # 8 M trans at 80 MT/mm^2 at 8 nm = 0.1 mm^2
        assert abs(b.area_mm2 - 0.1) < 1e-9


class TestHierarchies:
    def test_every_hierarchy_has_alu_and_soc(self):
        for h in default_hierarchies():
            assert h.alu is not None, f"{h.name} missing ALU level"
            assert h.soc is not None, f"{h.name} missing SoC level"

    def test_efficiency_decays_from_alu_to_soc(self):
        """ALU TOPS/W is always the ceiling; SoC TOPS/W is always
        below it (scaffolding tax grows monotonically up the tree)."""
        for h in default_hierarchies():
            alu = h.alu
            soc = h.soc
            assert alu is not None and soc is not None
            assert alu.tops_per_watt > soc.tops_per_watt, (
                f"{h.name}: ALU {alu.tops_per_watt:.2f} TOPS/W "
                f"should exceed SoC {soc.tops_per_watt:.2f} TOPS/W"
            )

    def test_npu_has_lowest_efficiency_decay(self):
        """NPU (KPU) is the design point intentionally built to
        minimize scaffolding tax; should have the smallest ALU→SoC
        efficiency decay of the four architectures."""
        hierarchies = default_hierarchies()
        npu = next(h for h in hierarchies
                   if h.architecture is ArchitectureCategory.NPU)
        cpu = next(h for h in hierarchies
                   if h.architecture is ArchitectureCategory.CPU)
        dsp = next(h for h in hierarchies
                   if h.architecture is ArchitectureCategory.DSP)
        gpu = next(h for h in hierarchies
                   if h.architecture is ArchitectureCategory.GPU)
        assert npu.efficiency_decay < cpu.efficiency_decay
        assert npu.efficiency_decay < gpu.efficiency_decay
        # DSP should also be worse than NPU but not as bad as CPU
        assert npu.efficiency_decay < dsp.efficiency_decay

    def test_active_fraction_in_valid_range(self):
        for h in default_hierarchies():
            for b in h.blocks:
                assert 0.0 < b.active_fraction <= 1.0, (
                    f"{h.name}/{b.name} active_fraction "
                    f"{b.active_fraction} out of range"
                )

    def test_cpu_has_lowest_active_fraction_at_soc(self):
        """CPUs spend the most silicon on control / caches / OOO,
        so the SoC-level active fraction is the lowest."""
        hierarchies = default_hierarchies()
        soc_active = {
            h.architecture.value: h.soc.active_fraction
            for h in hierarchies if h.soc is not None
        }
        assert soc_active["CPU"] < soc_active["NPU"]


class TestArchitectures:
    def test_nvidia_orin_has_expected_levels(self):
        h = build_nvidia_ampere_orin_hierarchy()
        levels = [b.level for b in h.blocks]
        assert BlockLevel.ALU in levels
        assert BlockLevel.PE in levels
        assert BlockLevel.TILE in levels
        assert BlockLevel.SOC in levels

    def test_kpu_t128_tile_matches_building_block_energy(self):
        """KPU tile data in composition view must be consistent with
        building_block_energy.build_kpu_tile_building_block()."""
        h = build_kpu_t128_hierarchy()
        tile = h.by_level(BlockLevel.TILE)
        assert tile is not None
        # From building_block_energy: 22 M trans, 144 pJ/clk,
        # 1024 MAC/clk (32x32 mesh)
        assert abs(tile.transistor_count_m - 22.0) < 1.0
        assert tile.macs_per_clock == 1024
        assert abs(tile.pj_per_clock - 144.0) < 10.0

    def test_arm_core_active_fraction_is_tiny(self):
        """A78 core transistor budget is dominated by OOO, caches,
        and control. SDOT is a thin slice."""
        h = build_arm_a78_cpu_hierarchy()
        core = h.by_level(BlockLevel.TILE)
        assert core is not None
        assert core.active_fraction < 0.02

    def test_hexagon_dsp_mac_rate_matches_hvx(self):
        """Hexagon HVX = 1024-bit vector, 128 INT8 MACs per VMAC
        instruction per clock."""
        h = build_qualcomm_hexagon_hierarchy()
        pe = h.by_level(BlockLevel.PE)
        assert pe is not None
        assert pe.macs_per_clock == 128


class TestHTMLRendering:
    def test_page_has_every_architecture(self):
        report = build_default_composition_report()
        html = render_composition_page(report, REPO_ROOT)
        for h in report.hierarchies:
            assert h.name in html

    def test_page_has_level_badges(self):
        report = build_default_composition_report()
        html = render_composition_page(report, REPO_ROOT)
        for lvl in BlockLevel:
            assert lvl.value in html

    def test_page_has_efficiency_chart(self):
        report = build_default_composition_report()
        html = render_composition_page(report, REPO_ROOT)
        assert 'id="chart_composition_efficiency"' in html

    def test_page_has_back_link(self):
        report = build_default_composition_report()
        html = render_composition_page(report, REPO_ROOT)
        assert 'href="index.html"' in html
