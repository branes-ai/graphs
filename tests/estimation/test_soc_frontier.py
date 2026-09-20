"""The energy-performance plane and its envelope (graphs#269 Phase 7.3)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from embodied_schemas import load_compute_products, load_process_nodes

from graphs.core.pipeline_workload import CLASS_NAMES, load_autonomy_workload
from graphs.estimation.soc import (
    load_efficiency_tables,
    load_kernel_classes,
    required_efficiency,
)
from graphs.estimation.soc.domainflow import fabric_ceilings
from graphs.estimation.soc.frontier import MissionPoint, envelope, mission_point
from graphs.estimation.soc.power import op_energy_pj
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library
from graphs.hardware.soc.kpu_cores import sku_of

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = load_autonomy_workload()
KERNELS = load_kernel_classes()
TABLE = load_efficiency_tables()["orin_nano_measured_v1"]
AIR = next(p for p in WORKLOAD.regimes() if p.regime == "air superiority")


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location(
        "mm_cli", REPO / "cli" / "analyze_mission_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def soc():
    return compose_soc(load_designs()["kpu_t128_n7"], load_ip_library(),
                       load_process_nodes(), None)


@pytest.fixture(scope="module")
def ceilings():
    sku = sku_of("kpu_t128_core")
    return fabric_ceilings(input_spec_from_compute_product(load_compute_products()[sku]))


@pytest.fixture(scope="module")
def point(matrix, soc, ceilings):
    mapping = matrix.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
    req = required_efficiency(WORKLOAD, AIR, soc, mapping=mapping, table=TABLE, kernels=KERNELS)
    return mission_point(req, soc, WORKLOAD, AIR, KERNELS, TABLE, ceilings, "lpddr5_phy_256b",
                         sku_of("kpu_t128_core"))


# ---------------------------------------------------------------------------
# A block whose datapath spans two libraries
# ---------------------------------------------------------------------------


def test_a_split_datapath_is_priced_by_its_mix(soc):
    """A T-series fabric runs INT8 on balanced-logic PE tiles and an
    hp-logic systolic tile; the op costs the weighted sum, not either one."""
    kpu = next(b for b in soc.blocks if b.name == "kpu")
    mix = kpu.template.compute.datapath_mix["int8"]
    assert len(mix) == 2 and sum(mix.values()) == pytest.approx(1.0)
    expected = sum(share * soc.node.energy_per_op_pj[f"{library.value}:int8"]
                   for library, share in mix.items())
    pj, why = op_energy_pj(kpu, soc.node, "int8")
    assert why is None and pj == pytest.approx(expected)
    # Between the two libraries it mixes, and not equal to either.
    singles = [soc.node.energy_per_op_pj[f"{lib.value}:int8"] for lib in mix]
    assert min(singles) < pj < max(singles)


def test_a_format_the_node_cannot_price_is_a_gap(soc):
    kpu = next(b for b in soc.blocks if b.name == "kpu")
    node = soc.node.model_copy(update={"energy_per_op_pj": {
        k: v for k, v in soc.node.energy_per_op_pj.items() if not k.startswith("hp_logic:")}})
    pj, why = op_energy_pj(kpu, node, "int8")
    assert pj is None and "hp_logic:int8" in why


# ---------------------------------------------------------------------------
# Placing a configuration on the plane
# ---------------------------------------------------------------------------


def test_energy_per_op_is_the_missions_ops_at_the_nodes_price(point, soc):
    """Hand-computed over the mission's own op mix."""
    blocks = {b.name: b for b in soc.blocks}
    by_stage = {p.stage: p for p in point.placements}
    total_pj = total_ops = 0.0
    for demand in WORKLOAD.demands(AIR):
        placement = by_stage[demand.stage.key]
        if not placement.engine:
            continue
        ops = demand.stage.ops_per_call * demand.rate_hz
        requirement = next(s for s in [placement] if s.stage == demand.stage.key)
        assert requirement is not None
        for cls, share in zip(CLASS_NAMES, demand.stage.class_split):
            if share <= 0:
                continue
            fmt = {"A": "int8", "B": "fp16", "C": "fp32"}[cls]
            pj, why = op_energy_pj(blocks[placement.engine], soc.node, fmt)
            if pj is None:
                continue
            total_pj += ops * share * pj
            total_ops += ops * share
    assert point.energy_per_op_pj == pytest.approx(total_pj / total_ops, rel=1e-6)
    assert point.energy_is_floor


def test_the_real_time_factor_is_one_over_the_busiest_engine(point):
    assert point.utilization == pytest.approx(max(point.utilization_by_engine.values()))
    assert point.real_time_factor == pytest.approx(1.0 / point.utilization)


def test_a_factor_below_one_is_proven_short(point):
    """The factor is a ceiling, so below 1 is a proof; at or above 1 it
    proves nothing, because the true factor is no higher."""
    assert point.real_time_factor < 1.0
    assert point.attainable is False
    better = MissionPoint(mission="m", design="d", node="tsmc_n7", cpu_cores=12, memory="m",
                          kpu_sku=None, energy_per_op_pj=0.2, unpriced_energy_fraction=0.0,
                          utilization=0.5)
    assert better.real_time_factor == pytest.approx(2.0) and better.attainable is None


def test_each_stage_records_where_its_efficiency_came_from(point):
    """A measurement where one exists, the domain-flow ceiling otherwise,
    and 'unpriced' when neither -- never silently filled."""
    kinds = {p.provenance for p in point.placements}
    assert kinds <= {"measured", "ceiling", "unpriced"}
    assert {"measured", "ceiling"} <= kinds
    for placement in point.placements:
        if placement.provenance == "measured":
            assert placement.efficiency is not None
        if placement.provenance == "unpriced":
            assert placement.efficiency is None
    # The point reports the weakest of them, and its unpriced stages.
    assert point.provenance in ("ceiling", "unpriced")
    assert all(s in {p.stage for p in point.placements} for s in point.unpriced_stages)


def test_more_cores_move_the_point_up_and_not_sideways(matrix, ceilings):
    """The CPU complement buys rate; it does not change what an op costs."""
    from graphs.estimation.soc import Override

    design = load_designs()["kpu_t128_n7"]
    library, nodes = load_ip_library(), load_process_nodes()
    points = []
    for clusters in (1, 3):
        variant = Override(target="block:cpu.count", values=[clusters]).apply(design, clusters)
        soc = compose_soc(variant, library, nodes, None)
        mapping = matrix.schedule_aware_mapping(WORKLOAD, AIR, soc, KERNELS)
        req = required_efficiency(WORKLOAD, AIR, soc, mapping=mapping, table=TABLE,
                                  kernels=KERNELS)
        points.append(mission_point(req, soc, WORKLOAD, AIR, KERNELS, TABLE, ceilings,
                                    "lpddr5_phy_256b", None))
    one, three = points
    assert three.real_time_factor > one.real_time_factor
    assert three.energy_per_op_pj == pytest.approx(one.energy_per_op_pj)


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def _point(energy: float, utilization: float, design: str = "d") -> MissionPoint:
    return MissionPoint(mission="m", design=design, node="tsmc_n7", cpu_cores=12, memory="lp",
                        kpu_sku=None, energy_per_op_pj=energy, unpriced_energy_fraction=0.0,
                        utilization=utilization)


def test_the_envelope_keeps_what_nothing_beats_on_both_axes():
    cheap_slow = _point(0.2, 4.0, "cheap")       # 0.25x real time
    dear_fast = _point(0.5, 0.5, "fast")         # 2.0x
    beaten = _point(0.6, 4.0, "beaten")          # dearer and no faster than cheap_slow
    front = envelope([cheap_slow, dear_fast, beaten])
    assert {p.design for p in front} == {"cheap", "fast"}
    assert [p.energy_per_op_pj for p in front] == [0.2, 0.5]  # sorted by energy


def test_an_exact_tie_keeps_both():
    """Neither beats the other, so the envelope carries both: the page
    reports the tie rather than picking one."""
    front = envelope([_point(0.3, 2.0, "a"), _point(0.3, 2.0, "b")])
    assert {p.design for p in front} == {"a", "b"}


def test_a_point_that_is_better_on_both_axes_clears_the_rest():
    front = envelope([_point(0.4, 2.0, "worse"), _point(0.2, 1.0, "better")])
    assert [p.design for p in front] == ["better"]


def test_points_with_no_factor_are_left_off():
    assert envelope([_point(0.3, 0.0, "no-utilization")]) == ()


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def test_the_page_renders_every_mission_with_its_furniture(point):
    from graphs.reporting.mission_frontier import render

    page = render({"m1": [point, _point(0.4, 2.0, "other")]}, {"m1": "Mission one"},
                  {"m1": "2 W budget"}, {"kpu_t128_n7": 128}, "2026-09-20", "<h1>Intro</h1>")
    assert page.count("<figure class=\"chart\">") == 1
    assert "real time" in page and "energy per op (pJ)" in page
    assert "real-time factor" in page
    assert "prefers-color-scheme: dark" in page          # dark mode is selected, not flipped
    assert "<table>" in page and "<caption>" in page      # the table view
    assert "mouseenter" in page                           # the hover layer
    assert "Mission one" in page and "2 W budget" in page


def test_the_page_states_both_axes_are_bounds(point):
    from graphs.reporting.mission_frontier import render

    page = render({"m1": [point]}, {"m1": "m"}, {"m1": ""}, {}, "2026-09-20", "<h1>x</h1>")
    assert "floor" in page and "ceiling" in page


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _cli(*args, expect=0):
    result = subprocess.run([sys.executable, "cli/report_mission_frontier.py", *args],
                            capture_output=True, text=True, cwd=REPO, timeout=600)
    assert result.returncode == expect, result.stderr
    return result


def test_cli_writes_a_standalone_page(tmp_path):
    out = tmp_path / "frontier.html"
    _cli("--mission", "air superiority", "-o", str(out))
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "<script src" not in page and "<link" not in page   # nothing to fetch
    assert page.count("<figure class=\"chart\">") == 1


def test_cli_writes_the_points_as_json(tmp_path):
    out = tmp_path / "points.json"
    _cli("--mission", "air superiority", "--format", "json", "-o", str(out))
    payload = json.loads(out.read_text())
    points = next(iter(payload.values()))
    assert points and {"energy_per_op_pj", "real_time_factor", "attainable"} <= set(points[0])


def test_cli_rejects_an_unknown_mission():
    assert "no mission matches" in _cli("--mission", "nope", expect=2).stderr


def test_a_point_carries_its_confidence(point):
    """Unpriced work makes it UNKNOWN; otherwise THEORETICAL, because the
    energy side is derived whatever the rate side rests on."""
    assert point.estimation_confidence.level.value == "unknown"
    assert "unpriced" in point.estimation_confidence.source
    whole = MissionPoint(mission="m", design="d", node="tsmc_n7", cpu_cores=12, memory="m",
                         kpu_sku=None, energy_per_op_pj=0.2, unpriced_energy_fraction=0.0,
                         utilization=0.5, provenance="measured")
    assert whole.estimation_confidence.level.value == "theoretical"
    assert whole.to_dict()["confidence"] == "theoretical"


def test_one_placement_per_stage_even_when_a_class_is_unpriced(matrix, soc, ceilings):
    """A class the node cannot price is named on the stage's own record;
    a second record would let a consumer count its ops twice."""
    node = soc.node.model_copy(update={"energy_per_op_pj": {
        k: v for k, v in soc.node.energy_per_op_pj.items() if not k.endswith(":int8")}})
    blind = soc.__class__(**{**soc.__dict__, "node": node})
    mapping = matrix.schedule_aware_mapping(WORKLOAD, AIR, blind, KERNELS)
    req = required_efficiency(WORKLOAD, AIR, blind, mapping=mapping, table=TABLE, kernels=KERNELS)
    point = mission_point(req, blind, WORKLOAD, AIR, KERNELS, TABLE, ceilings, "m", None)
    stages = [p.stage for p in point.placements]
    assert len(stages) == len(set(stages))
    gapped = [p for p in point.placements if p.energy_gap]
    assert gapped and all("int8" in p.energy_gap for p in gapped)
    assert point.unpriced_energy_fraction > 0


def test_the_page_says_so_when_nothing_can_be_placed():
    from graphs.reporting.mission_frontier import render

    blank = MissionPoint(mission="m", design="d", node="tsmc_n7", cpu_cores=4, memory="m",
                         kpu_sku=None, energy_per_op_pj=0.0, unpriced_energy_fraction=1.0,
                         utilization=0.0)
    page = render({"m": [blank]}, {"m": "m"}, {"m": ""}, {}, "2026-09-20", "<h1>x</h1>")
    assert "No configuration could be placed" in page
    assert page.rstrip().endswith("</html>")


def test_cli_takes_the_format_from_the_output_name(tmp_path):
    out = tmp_path / "points.json"
    _cli("--mission", "air superiority", "-o", str(out))
    assert json.loads(out.read_text())
