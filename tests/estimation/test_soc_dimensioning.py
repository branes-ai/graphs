"""Sizing one SoC to one mission (graphs#269 Phase 7.6).

The catalogue pages ask whether a configuration is ruled out. This asks how
big each engine has to be, which is a different question with a different
failure mode: a sizing that silently substitutes a figure it does not have
produces a confident number that is wrong, where the catalogue pages would
produce an honest blank.
"""

from __future__ import annotations

import html
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
from dataclasses import replace

from graphs.estimation.soc.dimensioning import (
    dimension,
    overlap_concern,
    provision,
)
from graphs.estimation.soc.domainflow import fabric_ceilings
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library
from graphs.hardware.soc.kpu_cores import sku_of
from graphs.hardware.soc.floorplan import compose_resources
from graphs.reporting.mission_dossier import (
    block_diagram,
    density_table,
    floorplan_diagram,
    pipeline_graph,
    render,
    scaling_table,
    silicon_table,
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


@pytest.fixture(scope="module")
def composition(soc):
    return compose_resources(soc)


def _all_svgs(dossier, composition=None) -> str:
    return (pipeline_graph(dossier) + block_diagram(dossier, [("ISP", "idle")])
            + sizing_diagram(dossier, [{"label": "alt", "unit": "core", "needed": 19.9,
                                        "provisioned": 24, "kind": "cpu", "rejected": True,
                                        "note": "n"}])
            + ("" if composition is None else floorplan_diagram(composition, "Andes RISC-V")))


def test_every_shape_closes_its_own_tag(dossier, composition):
    """An unquoted attribute value swallows the closing slash; the element
    never self-closes and every later sibling nests inside it and vanishes."""
    markup = _all_svgs(dossier, composition)
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
def cli():
    spec = importlib.util.spec_from_file_location(
        "dossier_cli_mod", REPO / "cli" / "report_mission_dossier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        throughput = [r for r in rows if r[0].endswith("throughput")]
        assert throughput, mission
        for _name, _figure, _source, status in throughput:
            assert status.startswith("met:") is expect_met, (mission, status)
        # Every engine that carries stages gets a row -- the first cut
        # reported only two named stages and silently dropped the rest.
        assert len(throughput) == len(dossier.provisions), mission


def test_a_pipeline_with_no_common_cadence_quotes_no_latency(unfittable, dossier):
    """mono runs at 110 MHz and the VLM at 1 Hz; a "per frame" latency
    across that is meaningless, so it is not drawn at all."""
    assert unfittable.chain_seconds is None
    svg = pipeline_graph(unfittable, latency_per_frame=False)
    assert "/frame" not in svg
    assert "needs 18.6 cores" in svg          # the demand carries the meaning instead
    # ...and the mission that does have a common cadence still shows it.
    assert "/frame" in pipeline_graph(dossier, latency_per_frame=True)


def test_the_pipeline_draws_no_dependency_it_was_not_given(unfittable, dossier):
    """Connecting adjacent boxes and labelling each edge with the
    destination's own byte demand invented a dataflow. Some stage
    configurations do name what they read -- cbf names the ESDF -- so the
    page claims an incomplete dependency graph, not an absent one."""
    for d in (unfittable, dossier):
        svg = pipeline_graph(d, latency_per_frame=False)
        assert "url(#arrow)" not in svg
        assert 'class="edge-l"' not in svg
        assert "no complete dependency graph" in svg
        assert "no dependency between stages" not in svg
    # Each box still carries its own demand, so nothing real was lost.
    svg = pipeline_graph(dossier, latency_per_frame=False)
    assert svg.count('class="n-row"') >= 2 * len(dossier.stages)


def test_a_tier_band_only_labels_a_row_of_one_tier(unfittable, dossier):
    """A wrapped row can span tiers; naming it after its first stage
    would be wrong about the rest."""
    for d in (unfittable, dossier):
        svg = pipeline_graph(d, latency_per_frame=False)
        bands = re.findall(r'class="tier-band"[^>]*>([^<]*)', svg)
        tiers = {st.tier for st in d.stages}
        assert len(bands) <= len(tiers)
    # The two-stage mission spans T1 and T4 on one row, so it gets none.
    assert not re.findall(r'class="tier-band"',
                          pipeline_graph(dossier, latency_per_frame=False))


def test_the_pipeline_subtitle_fits_its_own_viewbox(dossier, unfittable):
    """SVG text does not wrap, so a one-line explanation overflowed the
    narrow layouts and was clipped."""
    for d in (dossier, unfittable):
        svg = pipeline_graph(d, latency_per_frame=False)
        width = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
        for line in re.findall(r'class="dsub"[^>]*>([^<]*)', svg):
            # 11px text, conservatively ~5.2 units per character.
            assert len(line) * 5.2 < width, (len(line), width)


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
    # The ratio keeps its precision: 14.5x, not a rounded 15x.
    assert re.search(r"14\.\dx over", svg) and re.search(r"3\.\d+x over", svg)


INTERCEPTOR = "drone_interceptor_terminal_engagement"


@pytest.fixture(scope="module")
def interceptor(soc, ceilings):
    """A mission where the accelerator fits and only the CPU is short --
    the case that catches a verdict which lumps every engine together."""
    profile = next(p for p in WORKLOAD.profiles if p.id == INTERCEPTOR)
    return dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings,
                     target_utilization=0.85, tiles_per_server=128)


def test_an_engine_that_fits_is_not_listed_as_a_shortfall(interceptor, cli):
    """9.31 of 11 tiles is not a shortfall, and the first cut said it was:
    it listed every provision as evidence the mission failed."""
    kpu = next(p for p in interceptor.provisions if p.kind == "kpu")
    cpu = next(p for p in interceptor.provisions if p.kind == "cpu")
    assert kpu.utilization <= 1.0 < cpu.utilization
    text = cli.sections(interceptor, None, None, None)["verdict"]
    short = text.split("enough elsewhere")[0]
    assert "cores against" in short and "tiles against" not in short
    assert "tiles against" in text                 # ...but it is still reported
    assert "one owner" in text                     # only the CPU has one


def test_the_published_cross_check_is_about_the_workload_not_the_soc(interceptor, cli):
    """The annex's oversubscription is against a pooled machine, so it must
    not be read as a verdict on this SoC."""
    derived = cli.pooled_figures(INTERCEPTOR)
    assert derived and interceptor.published
    block = " ".join(cli._crosscheck(interceptor, derived).split())
    assert "validates the demand model, not the sizing" in block
    assert "pooled machine" in block
    assert "not comparable to the per-engine utilizations" in block
    for key in ("tops", "class_a_share", "oversubscription"):
        ours, theirs = derived[key], interceptor.published[key]
        assert abs(ours - theirs) / theirs < 0.10, key


def test_memory_demand_counts_unplaced_stages_so_it_is_not_a_floor(unfittable):
    """Engine sizing and datapath power omit stages no engine can take;
    memory traffic does not, because it is summed from every stage's byte
    count. Calling all three floors understates the distinction."""
    assert unfittable.unplaced
    counted = sum(s.bytes_per_s for s in unfittable.stages) / 1e9
    assert unfittable.dram_demand_gb_per_s == pytest.approx(counted)
    unplaced_bytes = sum(s.bytes_per_s for s in unfittable.stages
                         if s.key in unfittable.unplaced) / 1e9
    assert unplaced_bytes > 0, "this mission should have unplaced stages carrying bytes"
    # ...and those bytes are already inside the reported figure.
    assert unfittable.dram_demand_gb_per_s > counted - unplaced_bytes


def test_the_verdict_does_not_deny_what_does_serve_the_mission(interceptor, cli):
    """One resource over capacity does not mean nothing serves the
    mission: here the KPU and the memory interface both do."""
    text = cli.sections(interceptor, None, None, None)["verdict"]
    assert "Nothing in this parts list" not in text
    assert "This design does not serve this mission" in text


COBOT = "humanoid_cobot_human_adjacent_contact_rich"


@pytest.fixture(scope="module")
def cobot(soc, ceilings):
    profile = next(p for p in WORKLOAD.profiles if p.id == COBOT)
    return dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings,
                     target_utilization=0.85, tiles_per_server=128)


def test_a_call_that_outlasts_its_period_is_detected(cobot):
    """Sizing adds servers to meet a rate. It says nothing about whether
    one call finishes before the next is due, and for a control loop that
    is the only question that matters."""
    by_stage = {p.stage: p for p in cobot.placements}
    mpc = by_stage["mpc"]
    assert mpc.period_s == pytest.approx(1 / 500)
    assert mpc.seconds_per_call > mpc.period_s
    assert mpc.calls_in_flight == pytest.approx(
        mpc.seconds_per_call / mpc.period_s)
    assert mpc.calls_overlap
    # The 1 kHz safety filter itself closes comfortably.
    cbf = by_stage["cbf"]
    assert not cbf.calls_overlap
    assert cbf.seconds_per_call < cbf.period_s / 5


def test_overlap_is_read_through_the_stage_unit(cobot):
    """A pixel overlapping is parallelism; a solve overlapping is a loop
    that cannot close. The unit the catalogue states is what separates
    them -- not a judgement about the algorithm."""
    assert overlap_concern("per pixel") == "independent"
    assert overlap_concern("per measured point") == "independent"
    assert overlap_concern("per solve") == "sequential"
    assert overlap_concern("per map update") == "sequential"
    assert overlap_concern("per inference") == "pipelined"
    by_key = {st.key: st for st in cobot.stages}
    sequential = [p.stage for p in cobot.overlapping
                  if overlap_concern(by_key[p.stage].unit) == "sequential"]
    assert "mpc" in sequential and "esdf" in sequential
    # ...and the point-wise stages are not counted as violations.
    assert "tsdf" not in sequential and "mono" not in sequential


def test_overlap_does_not_change_whether_the_design_fits(cobot):
    """Overlap is a latency statement, not a server-count one, and the two
    are reported separately."""
    assert cobot.overlapping
    assert cobot.oversubscribed          # this mission is short of CPU too
    trimmed = replace(cobot, oversubscribed=(), unplaced=())
    assert trimmed.fits and trimmed.overlapping


def test_the_safety_section_covers_the_whole_chain(cobot, soc, cli):
    block = cli._safety(cobot, soc)
    chain = [st.key for st in cobot.stages if st.on_reactive_chain]
    assert chain
    for key in chain:
        assert f"<b>{key}</b>" in block, key
    # The filter closes and its inputs do not: that is the finding.
    assert "The filter closes; its inputs do not" in block
    assert "11.2x its period" in block
    # Chain stages nothing prices are named rather than silently omitted.
    for key in cobot.unplaced:
        if key in chain:
            assert key in block


def test_the_safety_section_says_nothing_mission_specific(cli, soc, ceilings):
    """It renders for every mission with a reactive chain, so a sentence
    true of one must not be asserted of the others. Four pages shipped
    claiming to be "human-adjacent and contact-rich"."""
    for mission in (MISSION, INTERCEPTOR, AMR_HARD, COBOT):
        profile = next(p for p in WORKLOAD.profiles if p.id == mission)
        d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
        block = " ".join(cli._safety(d, soc).split())
        if not block:
            continue
        assert "human-adjacent" not in block, mission
        assert "contact-rich" not in block, mission
        by_key = {st.key: st for st in d.stages}
        from graphs.estimation.soc.dimensioning import overlap_concern as _oc
        broken = [p for p in d.overlapping
                  if p.stage in by_key and by_key[p.stage].on_reactive_chain
                  and _oc(by_key[p.stage].unit) == "sequential"]
        # Plurals follow the count rather than the mission I wrote it for.
        if len(broken) == 1:
            assert "Both are" not in block, mission
            assert "either one" not in block, mission
        # The ESDF claim only where the ESDF is actually broken.
        if not any(p.stage == "esdf" for p in broken):
            assert "reads its hazards out of the" not in block, mission


def test_the_esdf_claim_is_checked_against_the_filter_config(cobot, soc, cli):
    """The sentence says cbf reads its hazards out of the ESDF. That is a
    claim about the stage's configuration, so it is read from the
    configuration -- a comment promising the check is not the check."""
    cbf = next(st for st in cobot.stages if st.key == "cbf")
    assert any("esdf" in k.lower() for k in cbf.config), cbf.config
    block_on = cli._safety(cobot, soc)
    assert "reads its hazards out of the" in block_on
    # The ratio is service time over period, never asserted as the age of
    # the data: late updates may queue, so the age is unbounded.
    assert "service-time-to-period ratio, not the age of the data" in " ".join(block_on.split())
    assert "the age is unbounded here" in " ".join(block_on.split())
    # Strip the ESDF out of the filter's config and the claim goes with it.
    blind = replace(cobot, stages=tuple(
        replace(st, config={k: v for k, v in st.config.items()
                            if "esdf" not in k.lower()})
        if st.key == "cbf" else st for st in cobot.stages))
    block = cli._safety(blind, soc)
    assert "reads its hazards out of the" not in block
    # ...while the rest of the finding survives.
    assert "The filter closes" in block


def test_the_safety_section_names_no_standard_and_promises_nothing(cobot, soc, cli):
    """A throughput model must not read as a safety case."""
    block = " ".join(cli._safety(cobot, soc).split())
    assert "makes no claim about any standard" in block
    assert "Worst-case execution time" in block
    assert "necessary condition, not a sufficient one" in block
    for word in ("certified", "compliant", "SIL ", "PL d", "guarantee"):
        assert word not in block, word


def test_the_interference_note_names_this_mission_s_own_traffic(cobot, soc, cli):
    """Hardcoding "the detector and the VLA" would be wrong for a mission
    that has neither."""
    block = cli._safety(cobot, soc)
    off_chain = sorted((st for st in cobot.stages if not st.on_reactive_chain),
                       key=lambda st: -st.bytes_per_s)[:2]
    for st in off_chain:
        assert f"<code>{st.key}</code>" in block


def test_a_mission_with_no_reactive_chain_gets_no_safety_section(cli, soc, ceilings):
    from dataclasses import replace as _replace

    profile = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    base = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    flat = _replace(base, stages=tuple(
        _replace(st, on_reactive_chain=False) for st in base.stages))
    assert cli._safety(flat, soc) == ""


AV_L45 = "autonomous_vehicle_sae_l4__l5_high__full_automation"


def test_a_profile_covering_two_levels_says_so(cli, soc, ceilings):
    """The catalogue has one profile for SAE L4 and L5. They differ by
    whether an operational design domain bounds the problem at all, so a
    page that presents one figure for both must say which it describes."""
    profile = next(p for p in WORKLOAD.profiles if p.id == AV_L45)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    row = next((r for r in cli.requirements_rows(d, None)
                if r[0] == "Scope of this profile"), None)
    assert row is not None, "the conflation should be an explicit gap"
    assert row[3] == "gap"
    assert "lower bound for L5" in row[2]
    blurb = cli.sections(d, soc, None, None)["use_case"]
    assert "L4 and L5 together" in blurb
    assert "lower bound for L5" in blurb
    # Missions with no such caveat get no such row.
    other = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    plain = dimension(WORKLOAD, other, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    assert not any(r[0] == "Scope of this profile"
                   for r in cli.requirements_rows(plain, None))


def test_a_sensor_subset_over_the_interface_settles_it(cli, soc, ceilings):
    """A subset of the traffic that already exceeds the interface proves
    the interface is short. A subset under it proves nothing, because
    every other stage shares the same interface."""
    heavy = next(p for p in WORKLOAD.profiles if p.id == AV_L45)
    d = dimension(WORKLOAD, heavy, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    front = sum(st.bytes_per_s for st in d.stages if st.tier == "T1")
    assert front > d.dram_supply_gb_per_s * 1e9
    row = next(r for r in cli.requirements_rows(d, None)
               if r[0].startswith("Sensor front end"))
    assert row[3].startswith("NOT MET"), row[3]

    light = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    e = dimension(WORKLOAD, light, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    assert sum(st.bytes_per_s for st in e.stages if st.tier == "T1") < e.dram_supply_gb_per_s * 1e9
    row = next(r for r in cli.requirements_rows(e, None)
               if r[0].startswith("Sensor front end"))
    assert row[3].startswith("NOT CHECKED"), row[3]


def test_the_l5_caveat_claims_only_the_demand_figures(cli, soc, ceilings):
    """Capacities are design inputs and the deadline and budget are this
    profile's own targets. Neither is a lower bound for L5, and the
    caveat must not say they are."""
    profile = next(p for p in WORKLOAD.profiles if p.id == AV_L45)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    raw = cli.sections(d, soc, None, None)["use_case"]
    blurb = " ".join(re.sub(r"<[^>]+>", "", raw).split())
    assert "every figure on this page is a lower bound" not in blurb
    assert "are a lower bound for L5" in blurb
    for phrase in ("deadline and power budget", "are inputs from the selected design"):
        assert phrase in blurb, phrase
    assert "graphs#339" in blurb


def test_one_chain_stage_over_the_budget_settles_the_deadline(cli, soc, ceilings):
    """Unpriced stages make the chain total unknown, but a single chain
    stage that cannot finish inside the whole budget settles it anyway:
    no sum over the rest can rescue it."""
    profile = next(p for p in WORKLOAD.profiles if p.id == AV_L45)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    assert d.unplaced and d.chain_seconds is None
    busted = d.chain_stages_over_deadline(d.stages)
    assert busted, "esdf alone should exceed the 100 ms budget"
    assert all(p.seconds_per_call > d.deadline_ms / 1000.0 for p in busted)
    row = next(r for r in cli.requirements_rows(d, None)
               if r[0] == "Sense-to-act latency")
    assert row[3].startswith("NOT MET"), row[3]
    assert "at the modelled efficiencies" in row[3]


def test_servers_do_not_shorten_a_call_in_the_chain_total(dossier):
    """Two sequential chain stages at 60 ms per call, each on two
    servers, take 120 ms serially. Dividing each by its server count
    would report 60 ms and call a 100 ms deadline met."""
    placements = tuple(
        replace(p, servers=2, seconds_per_call=0.060, period_s=0.060,
                calls_per_frame=1.0, seconds_per_frame=0.030)
        for p in dossier.placements[:1]) + tuple(
        replace(p, servers=2, seconds_per_call=0.060, period_s=0.060,
                calls_per_frame=1.0, seconds_per_frame=0.030)
        for p in dossier.placements[1:2])
    stages = tuple(replace(st, unit="per solve", on_reactive_chain=True)
                   for st in dossier.stages[:2])
    serial = replace(dossier, placements=placements, stages=stages,
                     unplaced=(), deadline_ms=100.0)
    assert serial.chain_seconds == pytest.approx(0.120)
    assert serial.deadline_headroom < 1.0
    # The same two stages element-wise do split across servers.
    parallel = replace(serial, stages=tuple(
        replace(st, unit="per pixel") for st in stages))
    assert parallel.chain_seconds == pytest.approx(0.060)


def test_a_deadline_with_no_busted_stage_is_not_claimed_missed(cli, soc, ceilings):
    profile = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    assert not d.chain_stages_over_deadline(d.stages)
    row = next(r for r in cli.requirements_rows(d, None)
               if r[0] == "Sense-to-act latency")
    assert row[3].startswith("met:"), row[3]


def test_a_power_floor_over_budget_is_a_proven_miss(cli, soc, ceilings):
    """A floor under budget proves nothing, because the terms it omits
    only add. A floor over budget proves the budget is missed."""
    profile = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    under = next(r for r in cli.requirements_rows(d, None) if r[0] == "Power budget")
    assert under[3].startswith("NOT CHECKED"), under[3]
    starved = replace(d, power_budget_w=d.datapath_total_w / 2)
    over = next(r for r in cli.requirements_rows(starved, None) if r[0] == "Power budget")
    assert over[3].startswith("NOT MET"), over[3]
    assert "the floor alone is over" in over[3]


def test_the_power_pressure_is_labelled_as_the_datapath_floor(cli, soc, ceilings):
    """The ratio comes from datapath watts only, so it must not read as
    full-SoC headroom."""
    profile = next(p for p in WORKLOAD.profiles if p.id == AV_L45)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    names = [name for name, _ratio in cli._facts(d, None)["pressure"]]
    assert "datapath power (arithmetic only)" in names
    assert "power" not in names


#: The marker for the memory-alternatives paragraph, so its absence can
#: be asserted as directly as its presence.
MEMORY_OPTIONS_MARKER = "On bandwidth alone"


def _steps(cli, d):
    return cli._sizing_steps(d, cli._facts(d, None),
                             next(p for p in d.provisions if p.kind == "kpu"),
                             next(p for p in d.provisions if p.kind == "cpu"), None, 128)


def test_a_memory_shortfall_names_the_interfaces_that_carry_it(cli, soc, ceilings):
    """The only mission where memory itself is over. Unlike the engines,
    the parts list holds interfaces wide enough, and the page should name
    every one of them."""
    profile = next(p for p in WORKLOAD.profiles if p.id == AV_L45)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    assert d.dram_demand_gb_per_s > d.dram_supply_gb_per_s
    options = cli._memory_options(d.dram_demand_gb_per_s)
    names = {name for name, _peak in options}
    assert {"lpddr5x_phy_512b", "hbm3_1stack"} <= names, names
    assert all(peak >= d.dram_demand_gb_per_s for _name, peak in options)
    assert options == sorted(options, key=lambda kv: kv[1])
    block = _steps(cli, d)
    assert MEMORY_OPTIONS_MARKER in block
    for name, _peak in options:
        assert name in block, name
    # Offered on bandwidth only: HBM is not a part-number substitution.
    assert "compatible with this SoC is a question this model does not answer" in block


def test_no_memory_options_line_when_memory_is_not_over(cli, soc, ceilings):
    profile = next(p for p in WORKLOAD.profiles if p.id == MISSION)
    d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
    assert d.dram_demand_gb_per_s < d.dram_supply_gb_per_s
    block = _steps(cli, d)
    assert MEMORY_OPTIONS_MARKER not in block
    for name, _peak in cli._memory_options(d.dram_demand_gb_per_s):
        assert name not in block, name


def test_the_unpriced_bytes_callout_follows_its_own_threshold(cli, soc, ceilings):
    """The callout fires above 20% of traffic and says "majority" only
    above 50%. Both boundaries are asserted from missions on either side
    rather than from the one it was written for."""
    shares = {}
    for mission in (AV_L45, COBOT, MISSION):
        profile = next(p for p in WORKLOAD.profiles if p.id == mission)
        d = dimension(WORKLOAD, profile, soc, KERNELS, TABLE, ceilings, 0.85, 128)
        unpriced = sum(st.bytes_per_s for st in d.stages if st.key in d.unplaced)
        share = unpriced / (d.dram_demand_gb_per_s * 1e9) if d.dram_demand_gb_per_s else 0.0
        block = _steps(cli, d)
        shares[mission] = share
        assert ("no engine can price" in block) is (share > 0.2), (mission, share)
        # "majority" is a claim about more than half, not about 21%.
        assert ("majority of the memory demand" in block) is (share > 0.5), (mission, share)
    # The three missions straddle both thresholds, so neither is vacuous.
    assert shares[AV_L45] > 0.5 and 0.2 < shares[COBOT] <= 0.5
    assert shares[MISSION] <= 0.2


AMR_HARD = "amr_logistics_mixed__dynamic_yard"
AMR_EASY = "amr_warehousing_structured_aisles"


def test_two_missions_on_one_design_compare_as_a_ratio(cli):
    """The same vehicle and pipeline in a harder environment: the point of
    the section is the ratio, so both sides must be sized identically."""
    hard, _s1, _t1 = cli.build(AMR_HARD, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    easy, _s2, _t2 = cli.build(AMR_EASY, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    block = " ".join(cli._comparison(hard, easy).split())
    assert "Against AMR: Warehousing" in block
    for engine in ("CPU", "KPU"):
        assert f">{engine}</td>" in cli._comparison(hard, easy)
    # Both notes are quoted, so a reader sees what actually differs.
    assert html.escape(easy.note) in cli._comparison(hard, easy)
    assert html.escape(hard.note) in cli._comparison(hard, easy)
    # The harder mission needs strictly more of both engines.
    for kind in ("cpu", "kpu"):
        a = next(p for p in easy.provisions if p.kind == kind)
        b = next(p for p in hard.provisions if p.kind == kind)
        assert b.servers_needed > a.servers_needed, kind


def test_added_stages_render_even_when_nothing_moves(cli):
    """The added-stage line used to sit behind an `if movers` guard, so a
    comparison that adds a stage without moving any existing one showed
    nothing at all."""
    hard, _s, _t = cli.build(AMR_HARD, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    trimmed = replace(hard, stages=tuple(
        st for st in hard.stages if st.key in ("det", "gain", "radar")))
    other = replace(trimmed, stages=tuple(
        st for st in trimmed.stages if st.key == "det"), mission="other", title="Other")
    block = cli._comparison(trimmed, other)
    assert "present only in" in block
    assert "<code>gain</code>" in block and "<code>radar</code>" in block


def test_the_comparison_claims_no_shared_vehicle_class(cli):
    """--compare takes any two missions, so the page must not describe
    every pair as the same robot doing a harder job."""
    hard, _s1, _t1 = cli.build(AMR_HARD, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    easy, _s2, _t2 = cli.build(AMR_EASY, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    block = " ".join(cli._comparison(hard, easy).split())
    for claim in ("same robot", "same pipeline", "same vehicle", "harder job"):
        assert claim not in block, claim
    assert "same design" in block          # what is actually shared is stated


def test_the_comparison_is_also_data(cli):
    hard, _s1, _t1 = cli.build(AMR_HARD, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    easy, _s2, _t2 = cli.build(AMR_EASY, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    data = cli._comparison_data(hard, easy)
    assert set(data["engines"]) == {"cpu", "kpu"}
    assert all(v["ratio"] > 1 for v in data["engines"].values())
    assert set(data["added_stages"]) == {"gain", "radar"}
    assert data["note"] and data["other_note"]
    assert data["stages"]["det"]["ratio"] > 4
    assert cli._comparison_data(hard, None) is None


DRONE_ENDURANCE = "drone_isr_endurance__wide_area_search"
HOUSE_WORK = "humanoid_house_work_open_world_long_horizon"


def test_two_models_on_one_fabric_are_compared_directly(cli):
    """House work carries both a VLA and a VLM, so the weight-streaming
    cost can be shown within one mission instead of across two."""
    d, soc, tiles = cli.build(HOUSE_WORK, "kpu_t128_n7", "orin_nano_measured_v1", 0.85)
    raw = cli._the_ask(
        d, cli._facts(d, None),
        next(p for p in d.provisions if p.kind == "kpu"),
        next(p for p in d.provisions if p.kind == "cpu"), 0.85, None, tiles)
    block = " ".join(re.sub(r"<[^>]+>", "", raw).split())
    assert "comparison is on this page" in block
    # The comparand is the biggest other arithmetic load, not the best
    # ceiling: "the stage doing the most work needs the fewest tiles".
    assert "vla does" in block and "of vlm" in block
    assert "fewer tiles" in block
    # The caveat is derived, not a fixed superlative: it names the
    # dominant stage's own kernel class and its share of the requirement.
    assert "Neither ceiling has been measured" in block
    assert "weight_stream_decode" in block
    assert "widest gap in this study" not in block
    by = {st.key: st for st in d.stages}
    assert by["vla"].ops_per_s > by["vlm"].ops_per_s
    vla_fit = by["vla"].fits["kpu"]
    vlm_fit = by["vlm"].fits["kpu"]
    assert vlm_fit.servers_needed > 100 * vla_fit.servers_needed


def test_the_comparison_is_phrased_from_the_true_side(cli):
    """On the endurance drone the dominant stage does *more* arithmetic
    than its comparand, so "does 0.72x the arithmetic" would be a
    contortion."""
    d, _soc, tiles = cli.build(DRONE_ENDURANCE, "kpu_t128_n7",
                               "orin_nano_measured_v1", 0.85)
    raw = cli._the_ask(
        d, cli._facts(d, None),
        next(p for p in d.provisions if p.kind == "kpu"),
        next(p for p in d.provisions if p.kind == "cpu"), 0.85, None, tiles)
    block = " ".join(re.sub(r"<[^>]+>", "", raw).split())
    assert "vlm does only" in block and "more tiles" in block
    # Every pair that follows names its subjects in the lead's order.
    pair = re.search(r"intensity[^\d]*([\d.]+) operations per byte against[^\d]*([\d.]+)",
                     block)
    assert pair, block[:200]
    assert float(pair.group(1)) < float(pair.group(2))   # vlm is the less intense
    ceilings = re.search(r"ceiling of ([\d.]+)% against ([\d.]+)%", block)
    assert ceilings, block[:200]
    assert float(ceilings.group(1)) < float(ceilings.group(2))  # vlm's comes first
    # The figures follow the subject: the bigger tile count comes first.
    tail = block.split("more tiles:")[1]
    first = float(tail.split("against")[0].strip().replace(",", ""))
    second = float(tail.split("against")[1].split(".")[0].strip().replace(",", ""))
    assert first > second


def test_a_comparison_reports_falls_as_well_as_rises(cli):
    """The endurance drone needs less of almost everything than the
    interceptor and 399x the accelerator. Reporting only the increases
    would make that pair look as though nothing moved."""
    hard, _s1, _t1 = cli.build(DRONE_ENDURANCE, "kpu_t128_n7",
                               "orin_nano_measured_v1", 0.85)
    other, _s2, _t2 = cli.build(INTERCEPTOR, "kpu_t128_n7",
                                "orin_nano_measured_v1", 0.85)
    block = cli._comparison(hard, other)
    assert "at least halves" in block
    assert "<code>det</code>" in block.split("at least halves")[1]
    data = cli._comparison_data(hard, other)
    assert data["engines"]["kpu"]["ratio"] > 100
    assert data["engines"]["cpu"]["ratio"] < 1
    assert "vlm" in data["added_stages"]


def test_a_tiny_quantity_qualifies_its_own_disagreement(cli):
    """A 23.8% difference on a share that is 0.3% of the workload says
    less than the number suggests; the page says so rather than quoting
    it bare."""
    d, _soc, _t = cli.build(DRONE_ENDURANCE, "kpu_t128_n7",
                            "orin_nano_measured_v1", 0.85)
    block = " ".join(cli._crosscheck(d, cli.pooled_figures(DRONE_ENDURANCE)).split())
    assert "largest disagreement" in block
    assert "of the workload, where the published figure carries one significant figure" in block


def test_no_comparison_section_without_a_second_mission(dossier, cli):
    assert cli._comparison(dossier, None) == ""


def test_a_mission_with_no_published_figures_gets_no_cross_check(dossier, cli):
    assert not dossier.published
    assert cli._crosscheck(dossier, cli.pooled_figures(MISSION)) == ""


def test_every_use_case_blurb_renders(cli):
    """A blurb with a placeholder the context does not supply raises at
    render time, which is a broken page rather than a failed build."""
    for mission in cli.USE_CASES:
        dossier, soc, tiles = cli.build(mission, "kpu_t128_n7",
                                        "orin_nano_measured_v1", 0.85)
        text = cli.sections(dossier, soc, None, None, tiles)["use_case"]
        assert "{" not in text and "}" not in text, mission


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
    assert page.count('class="diagram"') == 4
    for heading in ("The use case", "Product requirements", "The workload",
                    "The configuration", "dimensioning the engines", "What this rests on"):
        assert heading in page


def test_the_page_states_its_gaps_rather_than_filling_them(tmp_path):
    out = tmp_path / "d.html"
    _cli("-o", str(out))
    page = out.read_text()
    assert page.count("NOT STATED") >= 4          # thermal, SWaP, full-SoC power, area
    assert "Thermal limit" in page and "Weight" in page


def test_cli_json_carries_the_cross_check(tmp_path):
    """A caller on --format json could not reach the published comparison
    at all: the HTML path had it and the JSON path dropped it."""
    out = tmp_path / "i.json"
    _cli("--mission", INTERCEPTOR, "-o", str(out))
    data = json.loads(out.read_text())
    assert data["published"]["tops"] > 0
    assert data["crosscheck"]["tops"] > 0
    assert abs(data["crosscheck"]["tops"] - data["published"]["tops"]) \
        / data["published"]["tops"] < 0.10


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


# ---------------------------------------------------------------------------
# Section 7: the floorplan, and what the silicon is made of
# ---------------------------------------------------------------------------


def test_the_floorplan_places_exactly_the_area_it_claims(composition):
    """A treemap that does not tile its rectangle is a picture of areas
    nobody has. Every cell must be inside the core square, and the cells
    together must be the core square."""
    from graphs.reporting.mission_dossier import DIE_PX, _squarify

    cells = [(b.name, b.area_mm2) for b in composition.blocks if b.area_mm2 > 0]
    cells.append(("whitespace", composition.whitespace_mm2))
    cells.sort(key=lambda r: -r[1])
    ring = DIE_PX * (composition.io_ring_mm / composition.die_side_mm)
    core = DIE_PX - 2 * ring
    placed = _squarify(cells, ring, ring, core, core)
    assert len(placed) == len(cells)
    assert sum(w * h for _k, _x, _y, w, h in placed) == pytest.approx(core * core)
    for key, x, y, w, h in placed:
        assert x >= ring - 1e-6 and y >= ring - 1e-6, key
        assert x + w <= ring + core + 1e-6, key
        assert y + h <= ring + core + 1e-6, key
        # ...and in proportion: each cell's share of the square is its
        # share of the core area.
        share = dict(cells)[key] / composition.core_area_mm2
        assert (w * h) / (core * core) == pytest.approx(share, rel=1e-6), key


def test_a_block_with_no_figure_is_not_drawn_as_though_it_had_one(composition):
    """Five blocks on this design price at nothing. Giving them a cell
    would be inventing an area; leaving them out silently would make the
    drawing read as a complete die."""
    svg = floorplan_diagram(composition, "Andes RISC-V")
    assert composition.gap_blocks
    for block in composition.gap_blocks:
        assert f">{block.name}<" in svg
    assert "nothing states a size" in svg
    assert "floor" in svg
    # The cell colours are engine hues and the gap outline; no block with
    # no area gets a filled cell.
    filled = re.findall(r'<rect[^>]*fill="var\(--(cpu|kpu|other)\)"', svg)
    assert len(filled) == 2 * (len([b for b in composition.blocks if b.area_mm2 > 0]))


def test_the_page_says_a_gate_count_is_a_convention_not_a_figure(cli, dossier,
                                                                 composition):
    """Nothing in this repository carries a gate count. A gates column on
    a partner document that does not say so is a fabricated figure."""
    assert "gates (M, NAND2)" in silicon_table(composition, "Andes RISC-V")
    kpu = next(p for p in dossier.provisions if p.kind == "kpu")
    cpu = next(p for p in dossier.provisions if p.kind == "cpu")
    law = cli.scaling_law_for("kpu_t128_n7")
    out = cli._floorplan(composition, law, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
    note = out["silicon_note"]
    assert "convention" in note
    assert "2-input NAND" in note
    assert "not as a synthesis result" in note


def test_the_page_claims_no_cost_model_it_does_not_have(cli, dossier, composition):
    """Area is the proxy. There is no wafer price, no defect density and
    no yield model anywhere in this repository, so no figure here may be
    a cost."""
    kpu = next(p for p in dossier.provisions if p.kind == "kpu")
    cpu = next(p for p in dossier.provisions if p.kind == "cpu")
    law = cli.scaling_law_for("kpu_t128_n7")
    out = cli._floorplan(composition, law, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
    costing = " ".join(out["costing"].split())
    assert "Nothing here carries a wafer price, a defect density or a yield model" in costing
    assert "is not a cost ratio" in costing
    for page in out.values():
        assert "$" not in page
        assert "cost per" not in page


def test_the_worked_example_names_the_block_its_line_belongs_to(cli, dossier,
                                                               composition):
    """The first cut called every line "the fabric's", including ones that
    were not. A worked example that misattributes its own figure is worse
    than none."""
    kpu = next(p for p in dossier.provisions if p.kind == "kpu")
    cpu = next(p for p in dossier.provisions if p.kind == "cpu")
    law = cli.scaling_law_for("kpu_t128_n7")
    out = cli._floorplan(composition, law, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
    intro = out["floorplan"]
    assert "The fabric's" not in intro
    logic = max((ln for b in composition.blocks for ln in b.lines if ln.gates_m),
                key=lambda ln: ln.transistors_mtx)
    assert f"{logic.block}'s <code>{logic.name}</code>" in intro
    # ...and the division it quotes is the one that produced the area.
    assert f"{logic.mtx_per_mm2:,.0f} Mtx/mm&sup2;" in intro
    assert f"{logic.area_mm2:.3f} mm&sup2;" in intro


def test_the_floorplan_says_nothing_about_the_mission_but_its_sizing(cli, composition):
    """The recurring defect on this page: prose written against one
    mission carried to the next with a claim that only held for the
    first. Section 7 describes a die, so only the sizing paragraph may
    differ between two missions on the same design."""
    law = cli.scaling_law_for("kpu_t128_n7")
    built = {}
    for mission in (MISSION, QUADRUPED):
        dossier, _soc, _t = cli.build(mission, "kpu_t128_n7",
                                      "orin_nano_measured_v1", 0.85)
        kpu = next(p for p in dossier.provisions if p.kind == "kpu")
        cpu = next(p for p in dossier.provisions if p.kind == "cpu")
        built[mission] = cli._floorplan(composition, law, dossier, kpu, cpu, 0.85,
                                        "Andes RISC-V", 128)
    one, two = built[MISSION], built[QUADRUPED]
    assert set(one) == set(two)
    for key in one:
        if key == "scaling":
            assert one[key] != two[key]
            continue
        assert one[key] == two[key], key
    # The two missions size very differently, and the sizing paragraph is
    # where that shows.
    sized = {m: " ".join(built[m]["scaling"].split()) for m in built}
    assert "1 tile and 4 cores provisioned" in sized[MISSION]
    assert "1 tile and 4 cores provisioned" not in sized[QUADRUPED]


def test_a_die_the_catalogue_does_not_contain_is_labelled_an_extrapolation(cli,
                                                                          composition,
                                                                          dossier):
    """The line was fitted between 64 and 512 tiles. Pricing a one-tile
    fabric with it is arithmetic off the end of the evidence."""
    kpu = next(p for p in dossier.provisions if p.kind == "kpu")
    cpu = next(p for p in dossier.provisions if p.kind == "cpu")
    law = cli.scaling_law_for("kpu_t128_n7")
    out = cli._floorplan(composition, law, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
    assert "fitted between 64 and 512 tiles" in out["scaling"]
    assert "extrapolation" in out["scaling"]
    table = scaling_table(law, 1, 4, "Andes RISC-V")
    assert "1 tile<" in table and "1 tiles" not in table
    assert f"{law.die_area_mm2(1, 4):.3f}" in table


def test_the_density_table_cites_the_library_every_area_was_divided_by(composition):
    """An area is a division, and the page owes the reader the divisor and
    where it came from."""
    table = density_table(composition)
    for entry in composition.classes:
        assert entry.library in table
        assert entry.density_source in table
        assert f"{entry.mtx_per_mm2:,.0f}" in table


def _trim(comp, keep, drop_gaps=()):
    """The same composition with a chosen set of blocks, so the page can
    be asked what it says about a die the catalogue does not contain."""
    from dataclasses import replace

    blocks = []
    for block in comp.blocks:
        if block.name not in keep:
            continue
        lines = tuple(ln for ln in block.lines
                      if ln.anchored or ln.name not in drop_gaps)
        blocks.append(replace(block, lines=lines))
    return replace(comp, blocks=tuple(blocks))


def test_the_gap_note_follows_its_own_counts(cli, dossier, composition):
    """The paragraph was written against one design's five-and-three. On
    a die with one missing line it read "1 silicon line have no figure",
    and on a complete one it claimed a floor it did not have."""
    kpu = next(p for p in dossier.provisions if p.kind == "kpu")
    cpu = next(p for p in dossier.provisions if p.kind == "cpu")
    law = cli.scaling_law_for("kpu_t128_n7")

    def note(comp):
        out = cli._floorplan(comp, law, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
        return " ".join(out["floorplan_note"].split())

    # One line missing, in a block that is otherwise anchored.
    one = _trim(composition, {"kpu", "cpu"})
    assert len(one.gaps) == 1
    text = note(one)
    assert "1 silicon line has no figure" in text
    assert "line have" not in text
    assert "Andes RISC-V is anchored in part" in text
    assert "close it." in text

    # Nothing missing: there is no floor to claim, so there is no note.
    whole = _trim(composition, {"kpu"})
    assert whole.complete
    assert note(whole) == ""
    svg = floorplan_diagram(whole, "Andes RISC-V")
    assert "Areas are a" not in svg
    assert "nothing states a size" not in svg
    assert "Every line of silicon on this design carries a figure." in svg
    table = silicon_table(whole, "Andes RISC-V")
    assert "a floor" not in table
    assert "lines</td>" not in table


def test_the_costing_paragraph_needs_blocks_to_point_at(cli, dossier, composition):
    """"The blocks nothing prices --  -- are exactly the ones..." is not
    a sentence. A law with nothing unpriced drops the clause."""
    from dataclasses import replace

    kpu = next(p for p in dossier.provisions if p.kind == "kpu")
    cpu = next(p for p in dossier.provisions if p.kind == "cpu")
    law = cli.scaling_law_for("kpu_t128_n7")
    full = cli._floorplan(composition, law, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
    assert "The floor also leans one way" in full["costing"]

    priced = replace(law, unpriced_blocks=())
    out = cli._floorplan(composition, priced, dossier, kpu, cpu, 0.85, "Andes RISC-V", 128)
    assert "The floor also leans one way" not in out["costing"]
    # ...and the part that is true whatever is priced still stands.
    assert "is not a cost ratio" in out["costing"]


def test_no_unplaced_block_is_drawn_off_the_edge_of_the_drawing(composition):
    """Chips advanced along one row. Past the sixth the browser clipped
    them, which omits a block as surely as not drawing it does."""
    from dataclasses import replace

    from graphs.reporting.mission_dossier import CHIP_W

    many = replace(composition, blocks=composition.blocks + tuple(
        replace(b, name=f"{b.name}_{i}") for i in range(1, 4)
        for b in composition.gap_blocks[:3]))
    assert len(many.gap_blocks) > 6
    svg = floorplan_diagram(many, "Andes RISC-V")
    viewbox = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    width, height = float(viewbox.group(1)), float(viewbox.group(2))
    chips = re.findall(r'<rect class="source chip" x="([\d.]+)" y="([\d.]+)"', svg)
    assert len(chips) == len(many.gap_blocks)
    for x, y in chips:
        assert float(x) + CHIP_W <= width, x
        assert float(y) + 40 <= height, y
    for block in many.gap_blocks:
        assert f">{block.name}<" in svg


def test_the_family_is_fitted_once_not_twice(cli):
    """``main`` fits the law and then asked ``tile_area_fit`` to fit it
    again, composing four designs and reloading the compute products for
    each a second time."""
    law = cli.scaling_law_for("kpu_t128_n7")
    calls = []
    original = cli.scaling_law_for
    cli.scaling_law_for = lambda design_id: (calls.append(design_id), original(design_id))[1]
    try:
        handed = cli.tile_area_fit("kpu_t128_n7", law)
        assert calls == []
        assert cli.tile_area_fit("kpu_t128_n7") == handed
        assert calls == ["kpu_t128_n7"]
    finally:
        cli.scaling_law_for = original
    assert handed["slope_mm2_per_tile"] == law.per_tile_mm2
