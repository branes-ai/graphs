"""Sizing one SoC to one mission (graphs#269 Phase 7.6).

The catalogue pages ask whether a configuration is ruled out. This asks how
big each engine has to be, which is a different question with a different
failure mode: a sizing that silently substitutes a figure it does not have
produces a confident number that is wrong, where the catalogue pages would
produce an honest blank.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from embodied_schemas import load_compute_products, load_process_nodes

from graphs.core.pipeline_workload import load_autonomy_workload
from graphs.estimation.soc import load_efficiency_tables, load_kernel_classes
from graphs.estimation.soc.dimensioning import dimension, provision
from graphs.estimation.soc.domainflow import fabric_ceilings
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library
from graphs.hardware.soc.kpu_cores import sku_of
from graphs.reporting.mission_dossier import (
    block_diagram,
    pipeline_graph,
    render,
    sizing_diagram,
)

REPO = Path(__file__).resolve().parents[2]
MISSION = "edge_ai_device_multi_stream_tracking__anomaly"
WORKLOAD = load_autonomy_workload()
KERNELS = load_kernel_classes()
TABLE = load_efficiency_tables()["orin_nano_measured_v1"]


@pytest.fixture(scope="module")
def soc():
    return compose_soc(load_designs()["kpu_t128_n7"], load_ip_library(),
                       load_process_nodes(), None)


@pytest.fixture(scope="module")
def ceilings():
    return fabric_ceilings(input_spec_from_compute_product(
        load_compute_products()[sku_of("kpu_t64_core")]))


@pytest.fixture(scope="module")
def dossier(soc, ceilings):
    profile = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    return dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings,
                     target_utilization=0.85, tiles_per_server=128)


# ---------------------------------------------------------------------------
# The arithmetic the whole page rests on
# ---------------------------------------------------------------------------


def test_the_xue_identity_holds(dossier):
    """X x servers x U is the demand, exactly. If it is not, one of the
    three numbers on the block diagram is decorative."""
    assert dossier.provisions
    for prov in dossier.provisions:
        demand = sum(s.ops_per_s for s in dossier.stages if s.key in prov.stages)
        assert prov.demand_ops_per_s == pytest.approx(demand, rel=1e-9), prov.engine


def test_efficiency_is_a_share_of_peak_not_of_the_provisioned_engine(dossier):
    """E must not absorb U: a half-idle engine is not a half-efficient one.
    E is the ops-weighted mean of the per-class efficiencies, which is the
    only reading that survives a stage spanning two formats with different
    peaks underneath them."""
    for prov in dossier.provisions:
        assert 0 < prov.efficiency <= 1.0, prov.engine
        classes = [c for key in prov.stages
                   for s in dossier.stages if s.key == key
                   for c in s.fits[prov.engine].classes if c.efficiency]
        ops = sum(c.ops_per_s for c in classes)
        expected = sum(c.ops_per_s * c.efficiency for c in classes) / ops
        assert prov.efficiency == pytest.approx(expected), prov.engine
        # U is carried separately, so E stays well above it on an idle engine.
        assert prov.efficiency != pytest.approx(prov.utilization)


def test_the_cpu_is_the_badly_matched_engine_not_the_small_one(dossier):
    """The page's conclusion: the engine with no margin is the one with the
    low E, and it is carrying almost none of the arithmetic."""
    cpu = next(p for p in dossier.provisions if p.engine == "cpu")
    kpu = next(p for p in dossier.provisions if p.engine == "kpu")
    assert cpu.utilization > kpu.utilization
    assert cpu.efficiency < 0.1 < 0.9 < kpu.efficiency
    total = sum(s.ops_per_s for s in dossier.stages)
    mono = next(s for s in dossier.stages if s.key == "mono")
    assert mono.ops_per_s / total < 0.01


def test_a_stage_needs_every_class_priced_or_it_does_not_fit(dossier):
    """All-or-nothing: one unpriced class makes the whole stage unfittable
    on that engine. A partial fit would understate the size needed."""
    for stage in dossier.stages:
        for name, fit in stage.fits.items():
            if fit.fits:
                assert all(c.servers_needed is not None for c in fit.classes), (stage.key, name)
                assert fit.servers_needed == pytest.approx(
                    sum(c.servers_needed for c in fit.classes))
            else:
                assert fit.gap and any(c.servers_needed is None for c in fit.classes)


def test_each_stage_of_this_mission_has_exactly_one_engine(dossier):
    """The mapping is forced, not chosen -- that is the argument section 6
    makes, and it has to survive a rebuild."""
    for stage in dossier.stages:
        assert stage.only_engine is not None, stage.key
    placed = {p.stage: p.engine for p in dossier.placements}
    assert placed == {"mono": "cpu", "det": "kpu"}


def test_the_detector_has_no_cpu_fit_because_int8_is_unmeasured(dossier):
    det = next(s for s in dossier.stages if s.key == "det")
    assert "int8" in det.fits["cpu"].gap
    # ...and not because the CPU lacks an INT8 datapath.
    assert any(c.fmt == "int8" and c.peak_per_server for c in det.fits["cpu"].classes)


def test_the_light_path_has_no_kpu_fit_because_nothing_schedules_it(dossier):
    mono = next(s for s in dossier.stages if s.key == "mono")
    assert "schedule" in mono.fits["kpu"].gap


def test_provisioning_never_leaves_an_engine_over_its_target():
    assert provision(3.1558, 0.85) == 4
    assert provision(0.4757, 0.85) == 1
    assert provision(1.0, 1.0) == 1
    for needed, target in ((0.01, 0.85), (7.9, 0.5), (12.0, 0.9)):
        assert needed / provision(needed, target) <= target + 1e-12
    with pytest.raises(ValueError):
        provision(1.0, 0.0)


def test_latency_is_measured_in_frames_not_in_calls(dossier):
    """mono's call is one pixel; a per-call latency would be meaningless,
    so the pipeline node is labelled per frame of the output cadence."""
    assert dossier.frame_hz == 60.0
    mono = next(p for p in dossier.placements if p.stage == "mono")
    assert mono.calls_per_frame == pytest.approx(1920 * 1080)  # one 1080p frame
    assert dossier.chain_seconds == pytest.approx(
        sum(p.seconds_per_frame for p in dossier.placements))
    assert dossier.deadline_headroom > 1.0


def test_the_ceiling_makes_the_whole_dossier_theoretical(dossier):
    assert dossier.estimation_confidence.level.value == "theoretical"
    assert any(p.provenance == "ceiling" for p in dossier.provisions)


def test_energy_is_a_floor_over_the_placed_stages(dossier):
    assert not dossier.energy_gaps
    assert 0 < dossier.datapath_total_w < dossier.power_budget_w
    assert set(dossier.datapath_watts) == {"cpu", "kpu"}


def test_a_design_too_small_is_a_gap_not_a_sizing(soc, ceilings):
    """Capping servers at what the design has would report a utilization
    above 1 as though the configuration had been sized. The module's
    contract is that a figure it cannot support is named. The humanoid
    cobot needs far more CPU than this design has, and the state-space
    page independently rules it out."""
    profile = next(p for p in WORKLOAD.profiles
                   if p.id == "humanoid_cobot_human_adjacent_contact_rich")
    heavy = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings,
                      target_utilization=0.85, tiles_per_server=128)
    assert heavy.oversubscribed and not heavy.fits
    assert any(o.startswith("cpu:") for o in heavy.oversubscribed)
    assert heavy.estimation_confidence.level.value == "unknown"
    # The provision is still reported, capped, and says so through U > 1.
    cpu = next(p for p in heavy.provisions if p.engine == "cpu")
    assert cpu.servers_provisioned == 12 and cpu.utilization > 1.0


def test_a_design_that_fits_reports_no_oversubscription(dossier):
    assert not dossier.oversubscribed and dossier.fits
    assert all(p.utilization <= 1.0 for p in dossier.provisions)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _all_svgs(dossier) -> str:
    return (pipeline_graph(dossier) + block_diagram(dossier, [("ISP", "idle")])
            + sizing_diagram(dossier, [{"label": "alt", "unit": "core", "needed": 19.9,
                                        "provisioned": 24, "kind": "cpu", "rejected": True,
                                        "note": "n"}]))


def test_every_shape_closes_its_own_tag(dossier):
    """An unquoted attribute value swallows the closing slash; the element
    never self-closes and every later sibling nests inside it and vanishes."""
    markup = _all_svgs(dossier)
    for tag in re.findall(r"<(?:rect|line|circle|path)\b[^>]*>", markup):
        assert tag.endswith("/>"), tag
    for attribute in re.findall(r'\s[a-z-]+=(?!")[^\s>]+', markup):
        raise AssertionError(f"unquoted attribute value: {attribute}")


def test_no_enum_repr_reaches_the_markup(dossier):
    """EngineKind compares equal to its value but formats as
    'EngineKind.KPU', which lands in a stylesheet as a variable nobody
    defined and paints the mark black."""
    markup = _all_svgs(dossier)
    assert "EngineKind" not in markup and "CircuitClass" not in markup
    assert "var(--kpu)" in markup and "var(--cpu)" in markup


def test_the_dram_bar_cannot_overflow_its_track(soc, ceilings):
    """A mission can ask for more bandwidth than the interface has. An
    unclamped fill draws past the box and contradicts the figure printed
    beside it."""
    profile = next(p for p in WORKLOAD.profiles
                   if p.id == "autonomous_vehicle_sae_l4__l5_high__full_automation")
    heavy = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings,
                      target_utilization=0.85, tiles_per_server=128)
    assert heavy.dram_demand_gb_per_s > heavy.dram_supply_gb_per_s, "expected an overflow"
    svg = block_diagram(heavy)
    track = float(re.search(r'<rect class="track"[^>]*width="([\d.]+)"', svg).group(1))
    fill = float(re.search(r'<rect class="fill[^"]*"[^>]*width="([\d.]+)"', svg).group(1))
    assert fill <= track + 1e-6
    assert 'class="fill over"' in svg          # and it says it is over


def test_the_html_has_no_unmatched_paragraph_tags(dossier):
    """An unmatched </p> inside an <li> makes the parser insert an empty
    paragraph, which changes the DOM the reader gets."""
    page = render(dossier, {"analysis": "<ul><li>a</li></ul>"}, [], (), (), "today")
    body = page.split("<body>")[1]
    assert body.count("</p>") == len(re.findall(r"<p[ >]", body))
    assert "</p></li>" not in body


QUADRUPED = "quadruped_isr_dismounted_comms_denied"


@pytest.fixture(scope="module")
def unfittable(soc, ceilings):
    """A mission no configuration in the catalogue serves, which exercises
    every branch the fitting mission does not."""
    profile = next(p for p in WORKLOAD.profiles if p.id == QUADRUPED)
    return dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings,
                     target_utilization=0.85, tiles_per_server=128)


def test_a_requirement_is_never_reported_met_when_it_is_not(tmp_path):
    """The first cut said "met: 128 KPU tile at 1455%". A partner document
    that claims a requirement is met because a number exists is worse than
    no document."""
    spec = importlib.util.spec_from_file_location(
        "dossier_cli_rows", REPO / "cli" / "report_mission_dossier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for mission, expect_met in ((QUADRUPED, False), (MISSION, True)):
        dossier, _soc, _t = module.build(mission, "kpu_t128_n7",
                                         "orin_nano_measured_v1", 0.85)
        rows = module.requirements_rows(dossier, None)
        throughput = [r for r in rows if r[0] in ("Detection throughput", "Camera ingest")]
        assert throughput, mission
        for _name, _figure, _source, status in throughput:
            assert status.startswith("met:") is expect_met, (mission, status)


def test_a_pipeline_with_no_common_cadence_quotes_no_latency(unfittable, dossier):
    """mono runs at 110 MHz and the VLM at 1 Hz; a "per frame" latency
    across that is meaningless, so it is not drawn at all."""
    assert unfittable.chain_seconds is None
    svg = pipeline_graph(unfittable, latency_per_frame=False)
    assert "/frame" not in svg
    assert "needs 18.6 cores" in svg          # the demand carries the meaning instead
    # ...and the mission that does have a common cadence still shows it.
    assert "/frame" in pipeline_graph(dossier, latency_per_frame=True)


def test_the_pipeline_wraps_rather_than_shrinking(unfittable, dossier):
    """18 boxes on one line scale down to an illegible strip."""
    wide = pipeline_graph(unfittable, latency_per_frame=False)
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', wide)
    width, height = float(box.group(1)), float(box.group(2))
    assert width <= 1500, "the 18-stage pipeline should wrap, not stretch"
    assert height > 600, "wrapping should cost height"
    assert wide.count('class="node"') == len(unfittable.stages)
    # A short pipeline still fits on one row.
    short = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"',
                      pipeline_graph(dossier, latency_per_frame=True))
    assert float(short.group(2)) < height


def test_each_sizing_bar_is_a_share_of_its_own_engine(unfittable):
    """Tiles and cores are different units and must not share a scale. A
    full bar is that engine's capacity, whatever the unit."""
    svg = sizing_diagram(unfittable)
    widths = [float(w) for w in re.findall(
        r'<rect x="284"[^>]*width="([\d.]+)"', svg)]
    tracks = [float(w) for w in re.findall(
        r'<rect class="track"[^>]*width="([\d.]+)"', svg)]
    assert widths and len(widths) == len(tracks)
    for fill, track in zip(widths, tracks):
        assert fill <= track + 1e-6
    # Both engines are over capacity here, so both bars are full and red.
    assert svg.count("var(--warn)") >= 2
    assert "15x over" in svg and "3x over" in svg


def test_the_diagrams_survive_an_empty_dossier(dossier):
    empty = type(dossier)(
        mission="m", title="t", power_budget_w=1, deadline_ms=1, note="", sensors={},
        design="d", node="n", frame_hz=1.0, stages=(), provisions=())
    assert "no stage" in pipeline_graph(empty)
    assert "no engine" in block_diagram(empty)
    assert "nothing to size" in sizing_diagram(empty)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _cli(*args, expect=0):
    result = subprocess.run([sys.executable, "cli/report_mission_dossier.py", *args],
                            capture_output=True, text=True, cwd=REPO, timeout=900)
    assert result.returncode == expect, result.stderr
    return result


def test_cli_writes_a_standalone_page(tmp_path):
    out = tmp_path / "d.html"
    _cli("-o", str(out))
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "<script" not in page and "<link" not in page
    assert page.count('class="diagram"') == 3
    for heading in ("The use case", "Product requirements", "The workload",
                    "The configuration", "dimensioning the engines", "What this rests on"):
        assert heading in page


def test_the_page_states_its_gaps_rather_than_filling_them(tmp_path):
    out = tmp_path / "d.html"
    _cli("-o", str(out))
    page = out.read_text()
    assert page.count("NOT STATED") >= 4          # thermal, SWaP, full-SoC power, area
    assert "Thermal limit" in page and "Weight" in page


def test_cli_writes_json(tmp_path):
    out = tmp_path / "d.json"
    _cli("-o", str(out))
    data = json.loads(out.read_text())
    assert data["mission"] == MISSION
    assert {p["engine"] for p in data["provisions"]} == {"cpu", "kpu"}
    assert data["counterfactual"]["servers_needed"] > 10     # det in FP32 on CPU
    assert data["tile_area_fit"]["slope_mm2_per_tile"] > 0


def test_cli_rejects_an_impossible_target():
    assert "target-utilization" in _cli("--target-utilization", "0", expect=2).stderr


def test_the_projection_follows_the_configured_target_utilization(tmp_path):
    """The 10%-efficiency projection used to hardcode 0.85, so it
    contradicted the sizing whenever --target-utilization said otherwise."""
    counts = {}
    for target in ("0.85", "0.5"):
        out = tmp_path / f"d{target}.html"
        _cli("--target-utilization", target, "-o", str(out))
        found = re.search(r"(\d+) cores provisioned instead of (\d+)", out.read_text())
        assert found, f"projection missing at target {target}"
        counts[target] = tuple(int(g) for g in found.groups())
    # A lower target buys headroom, so both the projected and the actual
    # core count rise with it.
    assert counts["0.5"][0] > counts["0.85"][0]
    assert counts["0.5"][1] > counts["0.85"][1]


def test_cli_rejects_an_unknown_mission():
    assert "unknown mission" in _cli("--mission", "nope", expect=2).stderr


def test_cli_rejects_a_pooled_table_with_its_documented_exit_code():
    """A pooled table maps no stage to an engine, so there is nothing to
    size. Letting it through raised a TypeError from deeper in and exited
    1 instead of the documented 2."""
    result = _cli("--efficiency", "annex_v1", expect=2)
    assert "per-engine" in result.stderr and "Traceback" not in result.stderr


def test_the_area_fit_is_least_squares_over_every_catalogued_core():
    """The page claims a least-squares line through four cores; a two-point
    slope through the endpoints would make that claim false."""
    spec = importlib.util.spec_from_file_location(
        "dossier_cli", REPO / "cli" / "report_mission_dossier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fit = module.tile_area_fit()
    assert len(fit["points"]) == 4
    slope, intercept = fit["slope_mm2_per_tile"], fit["intercept_mm2"]
    # A least-squares line has residuals summing to zero; the two-point
    # slope through the endpoints does not.
    residuals = [area - (slope * tiles + intercept) for tiles, area in fit["points"]]
    assert sum(residuals) == pytest.approx(0.0, abs=1e-9)
    assert max(abs(r) for r in residuals) < 0.1


def test_the_page_names_the_table_it_was_built_from(tmp_path):
    out = tmp_path / "d.json"
    _cli("-o", str(out))
    data = json.loads(out.read_text())
    assert data["efficiency_table"] == "orin_nano_measured_v1"
    assert data["counterfactual_table"] == "orin_nano_measured_v1"
    page = tmp_path / "d.html"
    _cli("-o", str(page))
    text = page.read_text()
    # The reproduction command has to carry the table, or it reproduces
    # something else.
    assert "--efficiency orin_nano_measured_v1" in text
