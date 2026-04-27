"""Tests for the native-op energy breakdown module."""
from __future__ import annotations

from pathlib import Path

from graphs.reporting.native_op_energy import (
    REG_ACCESS_PJ_PER_BYTE_BY_PROCESS,
    build_default_comparison,
    build_kpu_native_op,
    build_tensor_core_native_op,
    build_tpu_native_op,
    register_pj_per_byte,
    render_native_op_page,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestRegisterScaling:
    def test_known_process_returns_exact(self):
        assert register_pj_per_byte(16) == 0.025
        assert register_pj_per_byte(8) == 0.014

    def test_monotone_in_process_node(self):
        nodes = sorted(REG_ACCESS_PJ_PER_BYTE_BY_PROCESS.keys())
        for prev, curr in zip(nodes[:-1], nodes[1:]):
            assert (REG_ACCESS_PJ_PER_BYTE_BY_PROCESS[prev]
                    <= REG_ACCESS_PJ_PER_BYTE_BY_PROCESS[curr])


class TestKPUBreakdown:
    def test_kpu_breakdown_has_three_layers(self):
        b = build_kpu_native_op()
        assert len(b.layers) == 3
        assert b.layers[0].name.startswith("ALU")
        assert "Register" in b.layers[1].name
        assert "L1" in b.layers[2].name

    def test_cumulative_energy_is_sum_of_incrementals(self):
        b = build_kpu_native_op()
        running = 0.0
        for layer in b.layers:
            running += layer.energy_pj_per_mac
            assert abs(running - layer.cumulative_pj_per_mac) < 1e-9

    def test_kpu_process_node_and_full_adder(self):
        b = build_kpu_native_op()
        assert b.process_node_nm == 16
        assert b.full_adder_energy_pj == 0.010

    def test_kpu_alu_matches_model(self):
        """ALU layer equals tile_energy_model.mac_energy_int8."""
        from graphs.hardware.mappers import get_mapper_by_name
        tem = get_mapper_by_name("Stillwater-KPU-T128").resource_model.tile_energy_model
        b = build_kpu_native_op()
        assert abs(b.layers[0].energy_pj_per_mac - tem.mac_energy_int8 * 1e12) < 1e-9


class TestTensorCoreBreakdown:
    def test_tc_breakdown_has_four_layers(self):
        b = build_tensor_core_native_op()
        assert len(b.layers) == 4
        # Last layer is the warp-scheduling / coherence overhead,
        # which is specific to SIMT and not present in KPU/TPU.
        assert "sched" in b.layers[-1].name.lower() or "coherence" in b.layers[-1].name.lower()

    def test_tc_native_op_is_hmma(self):
        b = build_tensor_core_native_op()
        assert "HMMA" in b.native_op_name
        assert b.macs_per_native_op == 4096  # 16x16x16

    def test_tc_is_higher_floor_than_kpu(self):
        """Tensor Core has a higher per-MAC energy floor than KPU due
        to warp-level machinery, even at a more advanced process node."""
        tc = build_tensor_core_native_op()
        kpu = build_kpu_native_op()
        assert tc.total_energy_per_mac_pj > kpu.total_energy_per_mac_pj


class TestTPUBreakdown:
    def test_tpu_weight_stationary_amortization(self):
        """TPU's L1 layer is small because weight-stationary amortizes
        the UB load over K reductions (reuse factor 64 for 64x64 array)."""
        b = build_tpu_native_op()
        assert b.layers[0].name.startswith("ALU")
        # L1 layer should be the smallest due to heavy amortization
        l1_layer = b.layers[-1]
        assert "L1" in l1_layer.name or "UB" in l1_layer.name
        assert l1_layer.energy_pj_per_mac < b.layers[0].energy_pj_per_mac

    def test_tpu_process_is_14nm(self):
        b = build_tpu_native_op()
        assert b.process_node_nm == 14


class TestComparison:
    def test_default_comparison_has_three_breakdowns(self):
        c = build_default_comparison()
        assert len(c.breakdowns) == 3
        archs = {b.archetype for b in c.breakdowns}
        assert archs == {
            "SIMT + Tensor Core", "Systolic (TPU)", "Domain Flow (KPU)"
        }

    def test_each_breakdown_carries_process_and_fa(self):
        c = build_default_comparison()
        for b in c.breakdowns:
            assert b.process_node_nm > 0
            assert b.full_adder_energy_pj > 0


class TestHTMLRender:
    def test_html_contains_chart_divs_and_table(self):
        c = build_default_comparison()
        html = render_native_op_page(c, REPO_ROOT)
        assert 'id="chart_progression"' in html
        assert 'id="chart_cumulative"' in html
        assert 'class="breakdown"' in html

    def test_html_calls_out_methodology(self):
        c = build_default_comparison()
        html = render_native_op_page(c, REPO_ROOT)
        assert "Methodology" in html
        assert "Horowitz" in html
        assert "reuse" in html.lower()

    def test_html_mentions_every_process_node(self):
        c = build_default_comparison()
        html = render_native_op_page(c, REPO_ROOT)
        for b in c.breakdowns:
            assert f"{b.process_node_nm} nm" in html

    def test_html_back_link_to_index(self):
        c = build_default_comparison()
        html = render_native_op_page(c, REPO_ROOT)
        assert 'href="index.html"' in html
