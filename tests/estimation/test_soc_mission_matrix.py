"""The CPU + KPU design state space against the missions (graphs#269 Phase 7)."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from embodied_schemas import load_process_nodes

from graphs.core.pipeline_workload import CLASS_NAMES, load_autonomy_workload
from graphs.estimation.soc import engines_of, load_kernel_classes
from graphs.estimation.soc.domainflow import NO_SCHEDULE
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = load_autonomy_workload()
KERNELS = load_kernel_classes()
AIR = next(p for p in WORKLOAD.regimes() if p.regime == "air superiority")


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "mission_matrix_cli", REPO / "cli" / "analyze_mission_matrix.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the @dataclass decorator can resolve the
    # module while the class body runs (as tests/cli/test_check_tdp_feasibility.py does).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def soc():
    return compose_soc(load_designs()["kpu_t128_n7"], load_ip_library(), load_process_nodes(), None)


# ---------------------------------------------------------------------------
# The division of labour
# ---------------------------------------------------------------------------


def test_the_accelerator_takes_only_what_it_has_a_schedule_for(cli, soc):
    """The classes the domain-flow model states no schedule for go to the
    CPU, whatever the fabric's formats allow. That is the split a CPU
    partner has to size for."""
    mapping = cli.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
    for stage, engine in mapping.items():
        scheduled = KERNELS.of(stage).value not in NO_SCHEDULE
        assert engine == ("kpu" if scheduled else "cpu"), (stage, engine)
    assert set(mapping.values()) == {"kpu", "cpu"}


def test_capability_would_hide_the_cpu(cli, soc):
    """Why the default is not the capability rule: the T128 can run every
    class, so that rule leaves the CPU with nothing and says nothing about
    the silicon a partner would build."""
    from graphs.estimation.soc import capability_mapping

    capability = capability_mapping(list(WORKLOAD.demands(AIR)), engines_of(soc))
    assert set(capability.values()) == {"kpu"}
    aware = cli.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
    assert sum(1 for e in aware.values() if e == "cpu") >= 10


def test_a_stage_the_accelerator_cannot_run_falls_to_the_cpu(cli):
    """The H64 has no FP32, so a scheduled class with a Class C share still
    goes to the CPU."""
    h64 = compose_soc(load_designs()["kpu_h64_n7"], load_ip_library(), load_process_nodes(), None)
    mapping = cli.schedule_aware_mapping(WORKLOAD, AIR, h64, KERNELS)
    for demand in WORKLOAD.demands(AIR):
        if demand.stage.class_split[CLASS_NAMES.index("C")] > 0:
            assert mapping[demand.stage.key] == "cpu", demand.stage.key


# ---------------------------------------------------------------------------
# The energy floor
# ---------------------------------------------------------------------------


def test_the_energy_floor_is_the_ops_at_the_nodes_energy_per_op(cli, soc):
    """Hand-computed: every op charged once at the node's figure for its
    engine's library and format. No efficiency enters, which is what makes
    it a floor."""
    from graphs.estimation.soc import execution_format
    from graphs.estimation.soc.power import datapath_library

    mapping = cli.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
    watts, gaps = cli.energy_floor_w(WORKLOAD, AIR, soc, mapping)
    blocks = {b.name: b for b in soc.blocks}
    engines = engines_of(soc)
    expected = 0.0
    for demand in WORKLOAD.demands(AIR):
        engine = mapping[demand.stage.key]
        lib = datapath_library(blocks[engine])
        if lib is None:      # priced as a gap below, not as free ops
            continue
        for cls, share in zip(CLASS_NAMES, demand.stage.class_split):
            if share <= 0:
                continue
            fmt = execution_format(cls, engines[engine].formats)
            pj = soc.node.energy_per_op_pj[f"{lib.value}:{fmt}"]
            expected += demand.stage.ops_per_call * share * demand.rate_hz * pj * 1e-12
    assert watts == pytest.approx(expected)


def test_a_block_with_two_logic_libraries_is_a_stated_gap(cli, soc):
    """The T-series core's tiles span balanced and hp logic, so the
    template states no one datapath class and its ops cannot be charged.
    The floor says so instead of counting them as free."""
    from graphs.estimation.soc.power import datapath_library

    blocks = {b.name: b for b in soc.blocks}
    assert datapath_library(blocks["kpu"]) is None
    assert datapath_library(blocks["cpu"]) is not None
    mapping = cli.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
    _watts, gaps = cli.energy_floor_w(WORKLOAD, AIR, soc, mapping)
    on_kpu = {s for s, e in mapping.items() if e == "kpu"}
    assert on_kpu and all(any(g.startswith(s) for g in gaps) for s in on_kpu)


def test_a_node_without_an_energy_figure_leaves_the_floor_a_lower_bound(cli, soc):
    """A format the node prices nothing for is a stated gap, and the floor
    says so rather than counting the ops as free."""
    stripped = soc.node.model_copy(update={"energy_per_op_pj": {
        k: v for k, v in soc.node.energy_per_op_pj.items() if not k.endswith(":fp32")}})
    blind = soc.__class__(**{**soc.__dict__, "node": stripped})
    mapping = cli.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
    watts, gaps = cli.energy_floor_w(WORKLOAD, AIR, blind, mapping)
    full, full_gaps = cli.energy_floor_w(WORKLOAD, AIR, soc, mapping)
    assert watts < full and len(gaps) > len(full_gaps)
    assert any("hp_logic:fp32" in g for g in gaps)


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


def _rows(cli, tmp_path, *args) -> list:
    path = tmp_path / "m.csv"
    assert cli.main([*args, "--output", str(path)]) == 0
    return list(csv.DictReader(path.open()))


def test_a_requirement_is_also_given_as_throughput(cli, tmp_path):
    """The fraction of dense peak is this model's; GOP/s per core is not,
    so a partner can hold their own core against it."""
    rows = _rows(cli, tmp_path, "--design", "kpu_t128_n7", "--mission", "air superiority",
                 "--cpu-clusters", "3", "--memory", "lpddr5_phy_256b")
    row = rows[0]
    soc = compose_soc(load_designs()["kpu_t128_n7"], load_ip_library(), load_process_nodes(), None)
    cpu = engines_of(soc)["cpu"]
    peak_gops = cpu.server_peak_ops_per_s("fp32") / 1e9
    assert float(row["cpu_needs_gops_per_core"]) == pytest.approx(
        float(row["cpu_needs_fraction"]) * peak_gops, rel=1e-6)
    assert float(row["cpu_needs_gops_per_core"]) < peak_gops * 1.001


def test_the_memory_axis_moves_the_dram_verdict(cli, tmp_path):
    rows = _rows(cli, tmp_path, "--design", "kpu_t256_n7", "--mission", "far flight",
                 "--cpu-clusters", "3", "--memory", "lpddr5_phy_256b", "lpddr5x_phy_512b",
                 "hbm3_1stack")
    dram = {r["memory"]: float(r["dram_utilization"]) for r in rows}
    assert dram["lpddr5_phy_256b"] > 2.0
    assert dram["lpddr5x_phy_512b"] < 1.0 and dram["hbm3_1stack"] < dram["lpddr5x_phy_512b"]
    over = next(r for r in rows if r["memory"] == "lpddr5_phy_256b")
    assert over["verdict"] == "no" and "DRAM" in over["why"]


def test_more_cores_lower_the_requirement_proportionally(cli, tmp_path):
    rows = _rows(cli, tmp_path, "--design", "kpu_t128_n7", "--mission", "air superiority",
                 "--cpu-clusters", "1", "2", "3", "--memory", "lpddr5_phy_256b")
    need = {int(r["cpu_cores"]): float(r["cpu_needs_fraction"]) for r in rows}
    assert need[4] == pytest.approx(3 * need[12], rel=1e-6)
    assert need[8] == pytest.approx(1.5 * need[12], rel=1e-6)


def test_a_verdict_is_never_yes(cli, tmp_path):
    """The model proves violations, not sufficiency: a complete proof needs
    a schedule and a power figure nothing gap-free supports yet."""
    rows = _rows(cli, tmp_path, "--design", "kpu_t128_n7", "--all",
                 "--cpu-clusters", "3", "--memory", "lpddr5x_phy_512b")
    assert {r["verdict"] for r in rows} <= {"no", "open"}
    assert all(r["why"] for r in rows if r["verdict"] == "no")


def test_the_state_space_covers_every_generated_core(cli, tmp_path):
    from graphs.hardware.soc.kpu_cores import KPU_CORES

    rows = _rows(cli, tmp_path, "--state-space", "--mission", "air superiority",
                 "--cpu-clusters", "3", "--memory", "lpddr5_phy_256b")
    designs = load_designs()
    with_core = {d for d, design in designs.items()
                 if any(b.ip in KPU_CORES for b in design.blocks)}
    assert {r["design"] for r in rows} == with_core
    assert {r["kpu_sku"] for r in rows} == {KPU_CORES[b.ip][0] for d in with_core
                                            for b in designs[d].blocks if b.ip in KPU_CORES}


def test_json_and_markdown_render(cli, tmp_path):
    for name, check in (("m.json", lambda t: json.loads(t)[0]["mission"]),
                        ("m.md", lambda t: "| mission |" in t)):
        path = tmp_path / name
        assert cli.main(["--design", "kpu_t64_n16", "--mission", "air superiority",
                         "--cpu-clusters", "3", "--memory", "lpddr5_phy_256b",
                         "--output", str(path)]) == 0
        assert check(path.read_text())


def test_unknown_inputs_are_errors():
    for args, message in ((["--design", "nope", "--mission", "far flight"], "unknown design"),
                          (["--design", "kpu_t64_n7", "--mission", "nope"], "no mission matches"),
                          (["--design", "kpu_t64_n7", "--memory", "nope"], "unknown IP template")):
        result = subprocess.run([sys.executable, "cli/analyze_mission_matrix.py", *args],
                                capture_output=True, text=True, cwd=REPO, timeout=300)
        assert result.returncode == 2, result.stdout
        assert message in result.stderr
