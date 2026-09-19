"""Smoke + correctness tests for cli/check_tdp_feasibility.py."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path



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
    def test_uniform_kpus_over_the_65pct_alu_budget_are_warnings(self):
        """T64/T128/T256 declare the TDP the power model computes. Since the
        catalog's BF16 energy was re-derived (Horowitz, 0.23 x FP32), that
        TDP is set by FP16 and is about 30% lower, and the model puts PE
        compute at ~84% of it. The 65% ALU budget is a balanced-SoC rule of
        thumb, so for a KPU exceeding it (here by 1.11-1.16x) is a warning;
        the INT8 ALU peak still fits inside the whole TDP, so they are
        feasible."""
        tool = _load_tool()
        for sku in ("Stillwater-KPU-T64", "Stillwater-KPU-T128", "Stillwater-KPU-T256"):
            row = tool.check_sku(sku)
            assert row is not None, f"{sku} not found"
            assert row.feasible and row.warning, sku
            assert 1.0 < row.overshoot < 1.25, (sku, row.overshoot)
            assert row.alu_power_w < row.tdp_w, sku
            assert "warning for kpu" in row.notes, sku

    def test_a_kpu_within_budget_passes_without_a_warning(self):
        tool = _load_tool()
        row = tool.check_sku("Stillwater-KPU-T768")
        assert row.feasible and not row.warning and row.overshoot < 1.0

    def test_a_kpu_over_its_whole_tdp_is_infeasible(self, monkeypatch):
        tool = _load_tool()
        mapper = copy.deepcopy(tool.get_mapper_by_name("Stillwater-KPU-T64"))
        rm = mapper.resource_model
        rm.thermal_operating_points[rm.default_thermal_profile].tdp_watts = 1.0
        monkeypatch.setattr(tool, "get_mapper_by_name", lambda name: mapper)
        row = tool.check_sku("Stillwater-KPU-T64")
        assert row.alu_power_w > row.tdp_w == 1.0
        assert not row.feasible and not row.warning

    def test_the_budget_still_decides_for_other_categories(self):
        tool = _load_tool()
        row = tool.check_sku("Jetson-Orin-AGX-64GB", alu_fraction_of_tdp=0.1)
        assert row.category != "kpu"
        assert not row.feasible and not row.warning

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
        # Tool returns 0 by default whether or not the SKU is feasible
        # (T128 is a warning at the 65% budget); only --fail-on-infeasible
        # elevates the exit code, and only for an infeasible SKU.
        rc = tool.main(["--hardware", "kpu_t128"])
        assert rc == 0

    def test_cli_fail_on_infeasible_flag(self):
        tool = _load_tool()
        # T64/T128/T256 exceed the default 65% ALU budget, but for KPUs that
        # is a warning, so the flag does not fail the run.
        rc = tool.main(["--hardware", "kpu_t64", "kpu_t128", "kpu_t256",
                        "--fail-on-infeasible"])
        assert rc == 0

    def test_cli_fails_on_an_infeasible_non_kpu(self):
        tool = _load_tool()
        rc = tool.main(["--hardware", "Jetson-Orin-AGX-64GB", "--alu-fraction", "0.1",
                        "--fail-on-infeasible"])
        assert rc == 1

    def test_cli_marks_kpu_warnings(self, capsys):
        tool = _load_tool()
        tool.main(["--hardware", "kpu_t64", "kpu_t768"])
        lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()
                 if line.startswith("Stillwater")}
        assert lines["Stillwater-KPU-T64"].rstrip().endswith("WARN")
        assert lines["Stillwater-KPU-T768"].rstrip().endswith("PASS")
