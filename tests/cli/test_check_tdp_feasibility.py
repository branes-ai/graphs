"""Smoke + correctness tests for cli/check_tdp_feasibility.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "cli" / "check_tdp_feasibility.py"
    module_name = "tdp_tool"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Register before exec so dataclass @dataclass decorator can find
    # the module when resolving class __module__.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestFullAdderReference:
    def test_known_process_returns_exact(self):
        tool = _load_tool()
        assert tool.full_adder_energy_pj(16) == 0.010
        assert tool.full_adder_energy_pj(8) == 0.005
        assert tool.full_adder_energy_pj(5) == 0.003

    def test_unknown_process_falls_back_to_nearest(self):
        tool = _load_tool()
        fa6 = tool.full_adder_energy_pj(6)
        assert fa6 in (tool.full_adder_energy_pj(5), tool.full_adder_energy_pj(7))

    def test_monotone_in_process_node(self):
        """Sorted ascending by nm, FA energy should be non-decreasing:
        smaller process nodes have lower (or equal) dynamic energy."""
        tool = _load_tool()
        nodes = sorted(tool.FULL_ADDER_ENERGY_PJ_BY_PROCESS.keys())
        for prev, curr in zip(nodes[:-1], nodes[1:]):
            assert (tool.FULL_ADDER_ENERGY_PJ_BY_PROCESS[prev]
                    <= tool.FULL_ADDER_ENERGY_PJ_BY_PROCESS[curr]), (
                f"{prev}nm FA ({tool.FULL_ADDER_ENERGY_PJ_BY_PROCESS[prev]}) "
                f"> {curr}nm FA ({tool.FULL_ADDER_ENERGY_PJ_BY_PROCESS[curr]})"
            )


class TestKPUFeasibility:
    def test_t256_feasible(self):
        """T256 (20x20 tile, 30 W) is still feasible. T64 and T128 now
        use the canonical 32x32 tile, which exceeds their original 6 W
        / 12 W envelopes - those targets need to be re-set before the
        feasibility assertion can apply to them. See the separate
        TODO test below."""
        tool = _load_tool()
        row = tool.check_sku("Stillwater-KPU-T256")
        assert row is not None, "T256 not found"
        assert row.feasible, (
            f"T256 is TDP-infeasible: ALU {row.alu_power_w:.2f} W vs "
            f"budget {row.alu_budget_w:.2f} W "
            f"(over by {row.overshoot:.2f}x)"
        )

    def test_t64_t128_feasible_at_32x32(self):
        """T64/T128 moved to canonical 32x32 tile; PR #153 then dropped
        catalog clocks and added per-profile Vdd so derived TDPs land on
        the 6 W / 12 W targets cleanly. Both SKUs are now feasible at
        their default operating points (xfail removed)."""
        tool = _load_tool()
        for sku in ("Stillwater-KPU-T64", "Stillwater-KPU-T128"):
            row = tool.check_sku(sku)
            assert row is not None, f"{sku} not found"
            assert row.feasible, (
                f"{sku} is TDP-infeasible: "
                f"ALU {row.alu_power_w:.2f} W vs "
                f"budget {row.alu_budget_w:.2f} W "
                f"(over by {row.overshoot:.2f}x)"
            )

    def test_kpu_entries_report_process_node(self):
        tool = _load_tool()
        row = tool.check_sku("Stillwater-KPU-T128")
        assert row.process_node_nm == 16
        assert row.full_adder_pj == 0.010

    def test_mac_energy_reasonable_vs_full_adder(self):
        tool = _load_tool()
        row = tool.check_sku("Stillwater-KPU-T128")
        ratio = row.mac_energy_pj / row.full_adder_pj
        assert 4.0 < ratio < 20.0, (
            f"MAC/FA ratio {ratio:.1f}x is outside the 4-20x plausible "
            f"range for optimized 16nm INT8 MAC."
        )


class TestCLI:
    def test_cli_runs_default(self):
        tool = _load_tool()
        # Tool returns 0 by default whether or not the SKU is feasible;
        # only --fail-on-infeasible elevates the exit code (T128 is in
        # fact feasible at its post-PR#153 12W operating point).
        rc = tool.main(["--hardware", "kpu_t128"])
        assert rc == 0

    def test_cli_fail_on_infeasible_flag(self):
        tool = _load_tool()
        # PR #153 dropped catalog clocks and added per-profile Vdd; T64,
        # T128, and T256 now derive cleanly to their 6 W / 12 W / 30 W
        # targets and are all feasible. --fail-on-infeasible should
        # return 0 since no infeasible SKUs are passed.
        rc = tool.main(["--hardware", "kpu_t64", "kpu_t128", "kpu_t256",
                        "--fail-on-infeasible"])
        assert rc == 0

    def test_cli_t256_alone_still_feasible(self):
        tool = _load_tool()
        rc = tool.main(["--hardware", "kpu_t256", "--fail-on-infeasible"])
        assert rc == 0
