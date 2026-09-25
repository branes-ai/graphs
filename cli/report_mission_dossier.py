#!/usr/bin/env python
"""
One mission, sized onto a CPU + KPU SoC (graphs#269 Phase 7.6).

Writes a standalone HTML page: the use case, the product requirements, the
workload as a pipeline graph with demands and latencies per node, the
quantified demand per stage, the configuration as a block diagram with
bandwidths and X/U/E per compute block, and the analysis that dimensions
each engine until the demand fits.

    python cli/report_mission_dossier.py -o docs/assessments/dossier-edge-tracking.html
    python cli/report_mission_dossier.py --mission humanoid_cobot_human_adjacent_contact_rich
    python cli/report_mission_dossier.py --target-utilization 0.7 --format json -o d.json

Exit codes:
    0 = report written
    2 = unknown mission, design or efficiency table, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402
from embodied_schemas import load_compute_products, load_process_nodes  # noqa: E402

from graphs.core.pipeline_workload import load_autonomy_workload  # noqa: E402
from graphs.estimation.soc import load_efficiency_tables, load_kernel_classes  # noqa: E402
from graphs.estimation.soc.dimensioning import dimension, provision  # noqa: E402
from graphs.estimation.soc.domainflow import fabric_ceilings  # noqa: E402
from graphs.estimation.soc.power import op_energy_pj  # noqa: E402
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product  # noqa: E402
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402
from graphs.hardware.soc.floorplan import (  # noqa: E402
    NAND2_TRANSISTORS, compose_resources, fit_scaling_law)
from graphs.hardware.soc.kpu_cores import KPU_CORES, sku_of  # noqa: E402
from graphs.hardware.sku_validators.silicon_math import SRAM_MTX_PER_KIB  # noqa: E402
from graphs.reporting.mission_dossier import gb, kib, ms, render, scaling_table, si  # noqa: E402
from graphs.reporting.output_format import write_report  # noqa: E402

#: The smallest catalogued core with a stated domain-flow schedule. Sizing
#: against it keeps the ceiling from carrying a DRAM bound that belongs to a
#: much larger fabric.
CEILING_CORE = "kpu_t64_core"

#: The counterfactual in section 6 is a measured FP32 operating point, and
#: comes from this table whatever ``--efficiency`` selects for the sizing.
COUNTERFACTUAL_TABLE = "orin_nano_measured_v1"


def build(mission: str, design_id: str, efficiency: str, target_utilization: float):
    workload = load_autonomy_workload()
    kernels = load_kernel_classes()
    tables = load_efficiency_tables()
    if efficiency not in tables:
        raise KeyError(f"unknown efficiency table {efficiency!r}")
    # A pooled table states one efficiency for the whole machine and maps no
    # stage to an engine, so there is nothing to size per engine. Rejecting
    # it here keeps the documented exit code instead of a TypeError from
    # deeper in.
    if tables[efficiency].kind != "per_engine":
        raise ValueError(
            f"efficiency table {efficiency!r} is {tables[efficiency].kind}; sizing needs a "
            f"per-engine table, because a pooled one maps no stage to an engine")
    designs = load_designs()
    if design_id not in designs:
        raise KeyError(f"unknown design {design_id!r}")
    profile = next((p for p in workload.profiles if p.id == mission), None)
    if profile is None:
        raise KeyError(f"unknown mission {mission!r}")

    soc = compose_soc(designs[design_id], load_ip_library(), load_process_nodes(), None)
    core = next((b.ip for b in designs[design_id].blocks if b.ip in KPU_CORES), None)
    tiles = _tiles(core) if core else 1
    ceilings = fabric_ceilings(input_spec_from_compute_product(
        load_compute_products()[sku_of(CEILING_CORE)]))
    dossier = dimension(workload, profile, soc, kernels, tables[efficiency], ceilings,
                        target_utilization=target_utilization, tiles_per_server=tiles)
    return dossier, soc, tiles


def _tiles(core: str) -> int:
    spec = input_spec_from_compute_product(load_compute_products()[sku_of(core)])
    return sum(t.num_tiles for t in spec.kpu_architecture.tiles)


def counterfactual(dossier, soc, stage_key: str, engine: str, fmt: str):
    """What it would take to run one stage on an engine in a wider format
    than its class floor -- a different measured operating point, not a
    substitute for the missing figure."""
    from graphs.estimation.soc import engines_of, load_efficiency_tables as _tables

    table = _tables()[COUNTERFACTUAL_TABLE]
    stage = next((s for s in dossier.stages if s.key == stage_key), None)
    eng = engines_of(soc).get(engine)
    if stage is None or eng is None:
        return None
    entry = table.lookup(_kernel_of(stage.kernel_class), eng.kind, fmt)
    if entry is None or not entry.known:
        return None
    peak = eng.server_peak_ops_per_s(fmt)
    throughput = peak * entry.compute_eff
    blocks = {b.name: b for b in soc.blocks}
    pj, _why = op_energy_pj(blocks[engine], soc.node, fmt)
    return {
        "stage": stage_key, "engine": engine, "fmt": fmt,
        "efficiency": entry.compute_eff, "confidence": entry.confidence.value,
        "peak_per_server": peak, "throughput_per_server": throughput,
        "servers_needed": stage.ops_per_s / throughput,
        "watts": None if pj is None else stage.ops_per_s * pj * 1e-12,
    }


def _kernel_of(name: str):
    from graphs.estimation.soc.efficiency import KernelClass

    return KernelClass(name)


def pooled_figures(mission: str) -> Optional[dict]:
    """The workload's own pooled summary, in the same quantities the annex
    publishes, so the two can be compared without either being recomputed
    from the other."""
    workload = load_autonomy_workload()
    profile = next((p for p in workload.profiles if p.id == mission), None)
    if profile is None or not profile.published:
        return None
    summary = workload.summary(profile)
    shares = summary.class_shares()
    out = {"tops": summary.ops_per_s / 1e12,
           "class_a_share": shares.get("A", 0.0),
           "class_c_share": shares.get("C", 0.0),
           "oversubscription": summary.oversubscription}
    dram = getattr(summary, "dram_gb_per_s", None)
    if dram is not None:
        out["dram_gb_per_s"] = dram
    return out


#: A tile family: designs that differ only in how many tiles the fabric
#: has. Anything else in the id -- a heterogeneous core, another node --
#: is a different family, and a per-tile slope across two families would
#: absorb the difference between them.
FAMILY = re.compile(r"^kpu_t(\d+)_(?P<suffix>[a-z0-9]+)$")


def family_of(design_id: str):
    """Every catalogued design in the same tile family, composed, with its
    tile count. Empty when the design is not one of a family."""
    match = FAMILY.match(design_id)
    if match is None:
        return []
    designs, lib, nodes = load_designs(), load_ip_library(), load_process_nodes()
    node = designs[design_id].process_node
    out = []
    for name, design in sorted(designs.items()):
        other = FAMILY.match(name)
        if other is None or other.group("suffix") != match.group("suffix"):
            continue
        if design.process_node != node:
            continue
        core = next((b.ip for b in design.blocks if b.ip in KPU_CORES), None)
        if core is None:
            continue
        out.append((name, compose_soc(design, lib, nodes, None), _tiles(core)))
    return out


def scaling_law_for(design_id: str):
    """How this design's die area moves with tiles and with cores."""
    family = family_of(design_id)
    return fit_scaling_law(family) if len(family) >= 2 else None


def tile_area_fit(design_id: str = "kpu_t128_n7", law=None):
    """Marginal area per tile, fitted over the catalogued family. Used to
    price a fabric no SKU in the catalogue has, which is an extrapolation
    and is labelled as one.

    ``law`` lets a caller that has already fitted the family hand the fit
    over: composing four designs and reloading the compute products for
    each is not work worth doing twice.
    """
    if law is None:
        law = scaling_law_for(design_id)
    if law is None:
        return None
    return {"slope_mm2_per_tile": law.per_tile_mm2, "intercept_mm2": law.fabric_fixed_mm2,
            "points": [list(point) for point in law.points]}


#: Where a catalogued profile covers more than one thing, what it does
#: and does not describe. Rendered as an explicit gap in the requirements
#: table so the scope of the page is stated with its other limits.
#: ``mission -> (what the row shows, why it is a gap)``.
SCOPE_CAVEATS: Dict[str, Tuple[str, str]] = {
    "autonomous_vehicle_sae_l4__l5_high__full_automation": (
        "L4 and L5 together",
        "one profile for both levels; the stated suite is an L4-class build, so these "
        "figures are a lower bound for L5 (graphs#339)"),
    "quadruped_surveillance_persistent_patrol": (
        "re-identification and day/night not costed",
        "the profile note states person re-identification and day/night operation; the "
        "pipeline builds neither a re-ID stage nor an infrared sensor, so the detector is "
        "the whole appearance model here and these figures are a lower bound (graphs#343)"),
}

#: The human framing of a mission, which is domain knowledge and not in
#: the catalogue. Anything else on the page is derived from the data.
USE_CASES: Dict[str, str] = {
    "edge_ai_device_multi_stream_tracking__anomaly": """
<p>A fanless, mains- or PoE-powered appliance watching four camera streams: a loading bay, a
retail floor, a perimeter. It detects, tracks and flags anomalies on every frame, unattended,
for years.</p>""",
    "autonomous_vehicle_sae_l4__l5_high__full_automation": """
<p>A vehicle with no fallback driver. Not a driver-assist system with a human watching it &mdash;
the stack is the driver, and there is nobody to hand back to. The sensor suite says as much:
{cameras:g} cameras, {radars:g} radars, lidar, and a dual-redundant stack behind them.</p>
<p>That is the whole difference from the level below. L3 assumes a human who takes over when
the system gives up; this removes that human, and everything the human was implicitly covering
has to be computed instead.</p>
<p class="note"><b>This profile covers L4 and L5 together, and they are not the same
problem.</b> L4 is bounded by an operational design domain &mdash; a geofence, a weather
envelope, a road class &mdash; and everything outside it is a reason to stop rather than a case
to handle. L5 has no such boundary.</p>
<p class="note">The sensor suite and rates stated here describe an L4-class build, so read the
page in three parts. The <b>demand</b> figures &mdash; operations, bytes, cores and tiles needed
&mdash; are a <b>lower bound for L5</b>, by an amount the catalogue does not state. The
<b>deadline and power budget</b> are this profile's own targets and are not L5 requirements;
whether an L5 platform's envelope differs at all is not something we hold. The <b>engine
capacities</b> are inputs from the selected design and say nothing about either level.
Separating the two needs an L5 workload with sourced rates, which we do not have
(graphs#339).</p>""",
    "humanoid_house_work_open_world_long_horizon": """
<p>A {dof:g}-degree-of-freedom humanoid doing household work it was not programmed for: open
vocabulary tasking, a 3 B vision-language-action policy producing motion at {vla_hz:g} Hz, and a
2 B vision-language model planning what to do next. Long-horizon, because the task is "tidy the
kitchen" rather than a trajectory, and open-world, because the kitchen is not a fixture.</p>
<p>It is the only mission in the catalogue carrying <b>both</b> a VLA and a VLM, which makes it
the one place the two can be compared on the same fabric in the same second.</p>""",
    "humanoid_cobot_human_adjacent_contact_rich": """
<p>A {dof:g}-degree-of-freedom humanoid working alongside people and touching things: predicting
human pose and intent, scheduling contact, and running a {cbf_hz:g} Hz safety filter over
{cbf_haz:g} tracked hazards. There is no cage and no light curtain &mdash; the separation
between the robot and a person is maintained in software, on this SoC, every millisecond.</p>
<p>That makes it the one mission in the catalogue where latency is not a performance property.
A late frame on a warehouse AMR is a slower robot; a late frame here is a person struck by a
{dof:g}-joint arm. The {deadline:g} ms deadline on the sense-to-act chain is the whole
specification.</p>""",
    "amr_logistics_mixed__dynamic_yard": """
<p>A logistics AMR working a mixed yard: indoor-outdoor transitions, humans and vehicles moving
around it, no fiducials to lean on, {speed:g} m/s. The same vehicle class as a warehouse AMR and
very nearly the same pipeline &mdash; what differs is that nothing about the environment is known
in advance.</p>""",
    "drone_isr_endurance__wide_area_search": """
<p>A small drone searching wide areas for a long time, emissions-restricted: it cannot radio
home for help, so whatever interprets what it sees has to run on the airframe. That is why
there is a 2-billion-parameter vision-language model in the loop at {vlm_qps:g} queries a
second, and it is the whole reason this mission looks the way it does.</p>
<p>Everything else is modest. {cameras:g} cameras at 20 Hz, one stereo pair, one radar, a
{deadline:g} ms deadline that is generous for the speeds involved, and {budget:g} W to fly on.
The interesting comparison is the interceptor: the same airframe class and the same power
budget, flown at the opposite time constant, with no VLM aboard.</p>""",
    "drone_interceptor_terminal_engagement": """
<p>A small interceptor flying the terminal phase of an engagement: closing at 150-250 m/s on a
manoeuvring target, with the whole sense-decide-act loop onboard. No VLM, no operator, no second
look. At that closure the {deadline:g} ms deadline is {metres_lo:.0f}-{metres_hi:.0f} m of
travel, so a late frame is not a dropped frame, it is a miss.</p>
<p>Everything else follows from the closure rate: stereo and lidar at high rate because the
scene changes fast, detection and the whole reactive stack at {det_hz:g} Hz, and
{budget:g} W to do it in on an airframe that also has to fly.</p>""",
    "quadruped_surveillance_persistent_patrol": """
<p>A legged robot walking a perimeter, over and over, on {budget:g} W. A {cameras:g}-camera
ring, a stereo pair and a lidar; it looks for people at {det_hz:g} Hz and keeps itself upright
on {dof:g} joints while it does it. No operator, no end to the patrol &mdash; the job is
repetitive on purpose, and the machine has to be cheap enough to leave running.</p>
<p>It is worth costing beside the ISR quadruped because it is that robot with two stages taken
out: the 2-billion-parameter vision-language model, and the radar. The remaining sixteen stages
are the same ones at somewhat lower rates. So the pair answers a question the other dossiers can
only gesture at &mdash; how much of a robot's compute problem is the language model, and how
much is the robot.</p>
<p class="note"><b>The note promises more than the pipeline costs.</b> It says "person
detection and re-identification, day/night", and the catalogue builds one detector, no re-ID
stage and no infrared sensor. Re-identification is not the same work as detection &mdash; an
embedding over each detection's crop, and a gallery search whose cost grows over a patrol
&mdash; and nothing here carries either. No figure has been invented for it, so the demand on
this page is a <b>lower bound</b> by an amount the catalogue does not state
(graphs#343).</p>""",
    "quadruped_isr_dismounted_comms_denied": """
<p>A legged robot carrying ISR for a dismounted team, over unstructured terrain, with no
datalink to lean on. Everything runs on the animal: stereo and lidar for terrain it has never
seen, visual-inertial odometry because GPS is not assumed, a detector and a 2-billion-parameter
vision-language model for what it is looking at, and a full reactive control stack so it stays
upright while doing all of it.</p>
<p>Comms-denied is the whole constraint. There is no cloud to hand the VLM to, no operator in
the loop, and a {budget:g} W budget that has to cover the compute alongside the actuators.</p>""",
}


def _facts(dossier, alt):
    """The handful of derived quantities the prose is built from, so the
    narrative cannot drift from the numbers beside it."""
    total_ops = sum(s.ops_per_s for s in dossier.stages) or 1.0
    place = {p.stage: p.engine for p in dossier.placements}
    by_engine = {}
    for prov in dossier.provisions:
        rows = [(s, s.fits[prov.engine]) for s in dossier.stages
                if place.get(s.key) == prov.engine]
        rows.sort(key=lambda r: -(r[1].servers_needed or 0))
        by_engine[prov.engine] = rows
    bytes_sorted = sorted(dossier.stages, key=lambda s: -s.bytes_per_s)
    supply = (dossier.dram_supply_gb_per_s or 0) * 1e9
    pressure = []
    if supply:
        pressure.append(("memory bandwidth",
                         dossier.dram_demand_gb_per_s * 1e9 / supply))
    for prov in dossier.provisions:
        pressure.append((f"the {prov.engine.upper()}",
                         prov.servers_needed / max(prov.servers_provisioned, 1)))
    if dossier.power_budget_w:
        pressure.append(("datapath power (arithmetic only)",
                         dossier.datapath_total_w / dossier.power_budget_w))
    pressure.sort(key=lambda kv: -kv[1])
    return {
        "total_ops": total_ops, "by_engine": by_engine,
        "bytes_top": bytes_sorted[0] if bytes_sorted else None,
        "bytes_supply": supply, "pressure": pressure,
        "unpriced": [s for s in dossier.stages if s.key in dossier.unplaced],
    }


def _worst(rows):
    return rows[0] if rows else (None, None)


def sections(dossier, soc, alt, area_fit, catalogued_tiles: int = 0,
             efficiency: str = "", target_utilization: float = 0.85,
             output: str = "", crosscheck: Optional[dict] = None,
             compare=None, composition=None, law=None,
             cpu_label: str = "CPU") -> dict:
    """The argument. Every number is interpolated from the dossier, so the
    prose cannot drift from the analysis it describes."""
    f = _facts(dossier, alt)
    kpu = next((p for p in dossier.provisions if p.kind == "kpu"), None)
    cpu = next((p for p in dossier.provisions if p.kind == "cpu"), None)
    total_ops = f["total_ops"]
    fits = dossier.fits
    out = {}

    out["lede"] = f"""
<p class="lede">{html_escape(dossier.note)}. {len(dossier.stages)} stages,
{dossier.power_budget_w:g} W, {dossier.deadline_ms:g} ms. What has to be true of the CPU and the
accelerator for that to close.</p>"""

    worst_name, worst_ratio = f["pressure"][0] if f["pressure"] else ("", 0.0)
    if fits:
        out["verdict"] = f"""
<div class="verdict">
<b>{cpu.servers_provisioned} CPU cores and {kpu.servers_provisioned} KPU
tile{'s' if kpu and kpu.servers_provisioned != 1 else ''}.</b> Nothing binds hard: the tightest
of memory, compute and power is {html_escape(worst_name)} at {worst_ratio:.0%}. The part to
argue about is the CPU, at {cpu.efficiency:.1%} of its peak on the work it was given.
</div>"""
    else:
        # Say what is over and what is not. Listing an engine that fits
        # among the reasons a mission fails is simply wrong, and this page
        # had it wrong: 9.31 of 11 tiles is not a shortfall.
        # Three bands, because "enough" and "nearly out" are not the same
        # claim -- especially when unpriced stages make the figure a floor.
        TIGHT = 0.85
        over, tight, ok = [], [], []
        for prov in dossier.provisions:
            phrase = (f"{_amount(prov.servers_needed)} {prov.unit}"
                      f"{'s' if prov.servers_needed != 1 else ''} against "
                      f"{prov.servers_provisioned}")
            bucket = (over if prov.utilization > 1.0
                      else tight if prov.utilization > TIGHT else ok)
            bucket.append((prov, phrase))
        supply = dossier.dram_supply_gb_per_s
        dram_ratio = (dossier.dram_demand_gb_per_s / supply) if supply else None
        if dram_ratio is not None:
            memory = f"memory at {dram_ratio:.0%} of the interface"
            if dram_ratio > 1.0:
                over.append((None, f"{dossier.dram_demand_gb_per_s:.3g} GB/s of memory "
                                   f"bandwidth against {supply:g}"))
            elif dram_ratio > TIGHT:
                tight.append((None, memory))
            else:
                ok.append((None, memory))

        headline = ("<b>This design does not serve this mission.</b> "
                    if over else
                    "<b>This mission is not sized: stages remain that no engine can take.</b> ")
        shortfall = (f"It is short of {_join(p for _x, p in over)}. "
                     if over else "")
        fitting = (f"What it has is enough elsewhere: {_join(p for _x, p in ok)}. "
                   if ok else "")
        margins = (f"With no margin left: {_join(p for _x, p in tight)}. "
                   if tight else "")
        # Compute sizing and datapath power omit unplaced stages; memory
        # traffic does not, because it is summed from every stage's byte
        # count whether or not an engine can run it.
        unpriced_note = (f"{_count(len(dossier.unplaced)).capitalize()} stage"
                         f"{'s' if len(dossier.unplaced) != 1 else ''} cannot be priced at "
                         f"all, so the engine and power figures here are floors -- the "
                         f"memory figure is not, since it counts every stage's bytes. "
                         if dossier.unplaced else "")
        owners = []
        for prov, _phrase in over:
            if prov is None:
                continue
            rows = f["by_engine"].get(prov.engine, [])
            if not rows or not prov.servers_needed:
                continue
            top, top_fit = rows[0]
            owners.append(f"<b>{html_escape(top.key)}</b> is "
                          f"{(top_fit.servers_needed or 0) / prov.servers_needed:.1%} of the "
                          f"{prov.engine.upper()} demand")
        tail = (f"<br><br>The shortfall has {_count(len(owners))} owner"
                f"{'s' if len(owners) != 1 else ''}: {_join(owners)}."
                if owners else "")
        out["verdict"] = f"""
<div class="verdict warn">
{headline}{shortfall}{fitting}{margins}{unpriced_note}{tail}</div>"""

    blurb = USE_CASES.get(dossier.mission, f"<p>{html_escape(dossier.note)}</p>")
    det = next((s for s in dossier.stages if s.key == "det"), None)
    context = {
        "budget": dossier.power_budget_w,
        "deadline": dossier.deadline_ms,
        "det_hz": det.rate_hz if det else 0.0,
        # The note states a closure range, so both ends are carried rather
        # than a midpoint nobody published.
        "metres_lo": 150.0 * dossier.deadline_ms / 1000.0,
        "metres_hi": 250.0 * dossier.deadline_ms / 1000.0,
        # From the profile note, which states it; not inferred.
        "speed": _speed_from_note(dossier.note),
        "dof": float(dossier.sensors.get("dof") or 0),
        "cbf_hz": float(dossier.sensors.get("cbf_hz") or 0),
        "cbf_haz": float(dossier.sensors.get("cbf_haz") or 0),
        "cameras": float((dossier.sensors.get("mono") or [0])[0]),
        "radars": float((dossier.sensors.get("radar") or [0])[0]),
        "vlm_qps": float(dossier.sensors.get("vlm_qps") or 0),
        "vla_hz": float(dossier.sensors.get("vla_hz") or 0),
    }
    out["use_case"] = (blurb.format(**context) if "{" in blurb else blurb) + f"""
<p>{_cams(dossier)}. {len(dossier.stages)} stages over
{len({s.tier for s in dossier.stages})} tiers, {sum(1 for s in dossier.stages if s.on_reactive_chain)}
of them on the sense-to-act chain that the {dossier.deadline_ms:g} ms deadline applies to.</p>"""

    out["requirements_intro"] = """
<p>What the product must do, and where each figure comes from. Anything the catalogue does not
state is marked <b>NOT STATED</b> rather than assumed.</p>"""

    out["requirements_note"] = f"""
<p class="note"><b>Thermal and SWaP</b> are yours to set; they are not in our model.
<b>Full-SoC power</b> is a different kind of gap: {dossier.datapath_total_w * 1e3:.0f} mW is
arithmetic only &mdash; no memory traffic, clock tree, leakage or idle, and it omits any stage
no engine can take. On a mission moving {dossier.dram_demand_gb_per_s:.0f} GB/s the memory term
alone will dwarf it. <b>CPU core
area</b> is the one gap you can close, and the closing section says why it is open.</p>"""

    heaviest = max(dossier.stages, key=lambda s: s.ops_per_s)
    out["workload"] = f"""
<p>{len(dossier.stages)} stages, from the sensor front end through state estimation, mapping and
semantics to reactive control and actuation. They do not run at one rate: from
{si(min(s.rate_hz for s in dossier.stages), '/s')} to
{si(max(s.rate_hz for s in dossier.stages), '/s')} in their own units, which is why each box
carries what it demands of an engine rather than a share of some common frame.</p>
<p>The arithmetic is concentrated. <code>{html_escape(heaviest.key)}</code> alone is
{heaviest.ops_per_s / total_ops:.0%} of the operations. The bytes are concentrated somewhere
else: <code>{html_escape(f['bytes_top'].key)}</code> is
{f['bytes_top'].bytes_per_s / sum(s.bytes_per_s for s in dossier.stages):.0%} of the traffic.
Those two facts pull the design in different directions, and section 6 is about which one
wins.</p>"""

    out["workload_note"] = (f"""
<p class="note">Drawn as a serial sum, which is pessimistic &mdash; a real build overlaps the
stages. Even so, {ms(dossier.chain_seconds)} against {dossier.deadline_ms:g} ms leaves
{dossier.deadline_headroom:.1f}x in hand.</p>""" if dossier.chain_seconds else f"""
<p class="note">No end-to-end latency is quoted. {len(dossier.unplaced)} stage(s) on this
pipeline have no efficiency figure on either engine, so any chain total would silently omit
them. The deadline cannot be checked until those are measured.</p>""")

    out["demand"] = """
<p>Per stage: the kernel class its arithmetic belongs to, the precision it needs, its rate, and
the operations and bytes that follow. <b>Kernel class</b> is the level at which hardware
efficiency is a real number, so it is the level everything is measured and sized at.</p>"""

    unpriced = f["unpriced"]
    out["demand_note"] = (f"""
<p class="note"><b>{len(unpriced)} stages cannot be priced on either engine:</b>
{', '.join('<code>' + html_escape(s.key) + '</code>' for s in unpriced)}. Between them they are
{sum(s.ops_per_s for s in unpriced) / total_ops:.1%} of the operations but
{sum(s.bytes_per_s for s in unpriced) / sum(s.bytes_per_s for s in dossier.stages):.0%} of the
bytes. The compute and datapath-power figures below therefore understate the mission; the memory
figure does not, because it counts every stage's bytes whether or not an engine can run it.</p>"""
        if unpriced else "")

    out["configuration"] = f"""
<p>One KPU fabric, a cluster of Andes RISC-V cores, a shared fabric and one LPDDR5 interface at
{dossier.node}. The <i>capacities</i> below &mdash; how many cores and tiles exist &mdash; come
from the selected design and are an input. What section 6 produces is how many the mission
<i>needs</i>, which is the number to compare them against.</p>
<p class="note">The CPU block carries <b>baseline</b> figures: we have measured an Arm
Cortex-A78AE, not an Andes core. They stand in for the socket until you replace them, which is
what section 6 asks for. Nothing here is a claim about Andes silicon.</p>
<p>Three numbers per compute block, because any two mislead: <b>X</b>, the throughput one server
actually delivers; <b>U</b>, the share of wall clock it is busy; <b>E</b>, the ops-weighted mean of the
per-class efficiencies it runs at. X and U compose exactly &mdash; <b>demand = X &times; servers
&times; U</b> &mdash; while E says how well the kernels suit the machine. Where a stage needs
many servers at a low ceiling and another needs few at a high one, E and X/peak diverge, and E
is the one that describes the kernels rather than the sizing.</p>"""

    if kpu and cpu and not fits:
        out["configuration_note"] = """
<p class="note">A U above 100% is not a sizing. It is the design being asked for more than it
has, and the figure beside it says by how much.</p>"""
    elif kpu and cpu:
        out["configuration_note"] = f"""
<p class="note">The KPU delivers {si(kpu.throughput_ops_per_s, 'OP/s')} per tile at
<b>E = {kpu.efficiency:.1%}</b>; the CPU {si(cpu.throughput_ops_per_s, 'OP/s')} per core at
<b>E = {cpu.efficiency:.1%}</b>.</p>"""

    out["analysis"] = _analysis(dossier, f, kpu, cpu)
    out["analysis_2"] = _sizing_steps(dossier, f, kpu, cpu, alt, catalogued_tiles)
    out["analysis_3"] = _the_ask(dossier, f, kpu, cpu, target_utilization,
                                 alt, catalogued_tiles)

    fit_note = ""
    if area_fit:
        fit_note = f"""
<p><b>Area.</b> Section 7 takes the die apart: transistors from the IP templates, densities
from the node's libraries, and a least-squares line over the catalogued family for how the
fabric scales ({area_fit['slope_mm2_per_tile']:.4f} mm&sup2; a tile plus
{area_fit['intercept_mm2']:.3f} mm&sup2;). One column there is open. We anchor the CPU
cluster's caches but not its core logic, because no absolute core area is published for the
baseline. <b>An Andes core area would close it outright.</b></p>"""

    if composition is not None:
        out.update(_floorplan(composition, law, dossier, kpu, cpu, target_utilization,
                              cpu_label, catalogued_tiles))

    out["safety"] = _safety(dossier, soc)
    out["comparison"] = _comparison(dossier, compare)
    out["crosscheck"] = _crosscheck(dossier, crosscheck)
    out["provenance"] = f"""
<p>Four kinds of figure, and the page says which each is:</p>
<ul>
<li><b>Catalogue.</b> The mission, its rates, per-call operation and byte counts, the SoC
composition, the node's energy per operation.</li>
<li><b>Measured.</b> CPU efficiencies from <code>{efficiency}</code>, with the run behind each
figure cited in the table. <b>These are measured on Arm Cortex-A78AE cores in a Jetson Orin
Nano</b> &mdash; our reference baseline, not a claim about any other core. They are why the CPU
requirement is stated as a throughput target rather than a core count.</li>
<li><b>Ceiling.</b> KPU efficiencies are our domain-flow upper bound, taken from the smallest
core with a stated schedule ({CEILING_CORE}). A ceiling bounds, it does not predict, so every
tile count is a <i>lower</i> bound.</li>
<li><b>Gap.</b> Named, never filled.</li>
</ul>
{fit_note}
<p>Reproduce with:</p>
<p><code>python cli/report_mission_dossier.py --mission {dossier.mission} \\<br>
&nbsp;&nbsp;&nbsp;&nbsp;--design {dossier.design} --efficiency {efficiency} \\<br>
&nbsp;&nbsp;&nbsp;&nbsp;-o {output or "dossier.html"}</code></p>"""
    return out


def _floorplan(comp, law, dossier, kpu, cpu, target_utilization: float,
               cpu_label: str, catalogued_tiles: int) -> Dict[str, str]:
    """Section 7: how every area on the die was arrived at, what the
    silicon is made of, and what the sizing in section 6 would take.

    Everything here is a property of the design and the node. The mission
    enters once, as the tile and core counts section 6 produced.
    """
    out: Dict[str, str] = {}
    logic = max((ln for b in comp.blocks for ln in b.lines if ln.gates_m),
                key=lambda ln: ln.transistors_mtx, default=None)
    sram = max((ln for b in comp.blocks for ln in b.lines if ln.sram_kib),
               key=lambda ln: ln.transistors_mtx, default=None)
    per_kib = SRAM_MTX_PER_KIB.get(sram.circuit_class) if sram else None
    labels = {b.name: (cpu_label if b.name == "cpu" else b.name) for b in comp.blocks}
    worked = ""
    if logic is not None:
        worked += (
            f" The largest logic line on this die is "
            f"{html_escape(labels.get(logic.block, logic.block))}'s "
            f"<code>{html_escape(logic.name)}</code>: it is "
            f"{logic.transistors_mtx:,.0f} Mtx of "
            f"<code>{html_escape(logic.circuit_class.value)}</code>, and "
            f"{html_escape(comp.node_id)} puts {html_escape(logic.library)} at "
            f"{logic.mtx_per_mm2:,.0f} Mtx/mm&sup2;, so it takes "
            f"{logic.area_mm2:.3f} mm&sup2;.")
    if sram is not None and per_kib:
        worked += (
            f" The largest SRAM line, "
            f"{html_escape(labels.get(sram.block, sram.block))}'s "
            f"<code>{html_escape(sram.name)}</code>, is "
            f"{sram.transistors_mtx:,.0f} Mtx of "
            f"<code>{html_escape(sram.circuit_class.value)}</code> at "
            f"{sram.mtx_per_mm2:,.0f} Mtx/mm&sup2;, which is {sram.area_mm2:.3f} mm&sup2; "
            f"&mdash; and, at {per_kib:g} Mtx per KiB, {kib(sram.sram_kib)} of storage.")

    out["floorplan"] = f"""
<p>An area here is arithmetic, not a drawing. Every block declares its silicon one line at a
time: a transistor count and the circuit class it is built in. The node states a density for
that class &mdash; a named library, a figure in Mtx/mm&sup2;, and a source &mdash; and the
area is the division.{worked}</p>
<p>The blocks then roll up the same way on every design.
{comp.block_area_mm2:.2f} mm&sup2; of blocks, plus
{comp.whitespace_fraction:.0%} for placement and routing whitespace, is
{comp.core_area_mm2:.2f} mm&sup2; of core; a square core is
{comp.core_area_mm2 ** 0.5:.2f} mm a side, and a {comp.io_ring_mm:g} mm pad ring on each
side makes the die {comp.die_side_mm:.2f} mm a side and
{comp.die_area_mm2:.1f} mm&sup2;. The whitespace fraction and the ring depth are the
design's layout record rather than any block's silicon&nbsp;&mdash;
{html_escape(comp.layout_source)}</p>"""

    # Each clause is built only when it has something to say. A design
    # with every line anchored gets no note at all, because "every area
    # here is a floor" would then be false rather than merely clumsy.
    partial = [b for b in comp.blocks if b.gaps and b.area_mm2 > 0]
    empty = list(comp.gap_blocks)
    missing = len(comp.gaps)
    clauses = []
    if empty:
        clauses.append(
            f"{_count(len(empty)).capitalize()} block{'s' if len(empty) != 1 else ''} "
            f"&mdash; {_join(html_escape(labels.get(b.name, b.name)) for b in empty)} "
            f"&mdash; {'have' if len(empty) != 1 else 'has'} no anchored line at all, so "
            f"{'they occupy' if len(empty) != 1 else 'it occupies'} nothing in the drawing "
            f"above")
    if partial:
        clauses.append(
            f"{_join(html_escape(labels.get(b.name, b.name)) for b in partial)} "
            f"{'are' if len(partial) != 1 else 'is'} anchored in part, with "
            f"{_join(f'<code>{html_escape(ln.name)}</code>' for b in partial for ln in b.gaps)}"
            f" missing")
    out["floorplan_note"] = "" if not missing else f"""
<p class="note"><b>Every area on this page is a floor.</b>
{missing} silicon line{"s" if missing != 1 else ""} {"have" if missing != 1 else "has"} no
figure. {"; ".join(clauses)}. Nothing is filled in to close
{"them" if missing != 1 else "it"}.</p>"""

    out["silicon_note"] = f"""
<p class="note"><b>Two columns are conventions, and the page owes you which.</b> Nothing in
this catalogue carries a gate count. The gate column divides the transistor figure by
{NAND2_TRANSISTORS}, the transistors in a 2-input NAND in a standard CMOS cell, which is the
usual gate-equivalent; read it as the transistor figure restated in the unit an RTL team
works in, not as a synthesis result. The SRAM column is a division the catalogue itself
makes, run backwards: an SRAM line is built at {SRAM_MTX_PER_KIB[sram.circuit_class]:g} Mtx
per KiB &mdash; a 6T bitcell plus about 6% periphery &mdash; so the capacity comes back out
of the transistors.</p>""" if sram is not None else ""

    sram_area = sum(c.area_mm2 for c in comp.classes if c.sram_kib)
    logic_area = sum(c.area_mm2 for c in comp.classes if c.gates_m)
    confidences = sorted({c.density_confidence for c in comp.classes})
    out["density_note"] = f"""
<p>{sram_area / comp.block_area_mm2:.0%} of the block area on this die is SRAM: that is what
{kib(comp.sram_kib)} of on-chip storage costs at {comp.node_id}. The other
{logic_area / comp.block_area_mm2:.0%} is logic.</p>
<p class="note">Every density above is <b>{_join(c.upper() for c in confidences)}</b>: a
published library figure for the node, not a measurement of this design placed and routed. A
density is also a single number standing in for a whole block's mix of cells, utilization and
routing, so treat the third decimal of any area on this page as arithmetic rather than
signal.</p>"""

    if law is None or kpu is None or cpu is None:
        return out

    tiles = provision(kpu.servers_needed, target_utilization)
    cores = provision(cpu.servers_needed, target_utilization)
    low, high = law.tiles_fitted
    where = law.extrapolates(tiles)
    reach = {"below": f"below the {low} of the smallest core in it",
             "above": f"above the {high} of the largest core in it",
             "within": "inside it"}[where]
    out["scaling"] = f"""
<h3>What section 6's sizing would take</h3>
<p>The die above is <code>{html_escape(dossier.design)}</code>, and its size is an input: it
has {catalogued_tiles} tiles and {law.cores_per_cluster} cores whatever the mission asks for.
Section 6 asked for {_amount(kpu.servers_needed)} tiles and
{_amount(cpu.servers_needed)} cores &mdash; {tiles:,g} tile{"s" if tiles != 1 else ""} and
{cores:,g} core{"s" if cores != 1 else ""} provisioned at {target_utilization:.0%}. Pricing that means separating the part of the die that
moves with the fabric from the part that does not.</p>
<p>{_count(len(law.designs))} catalogued cores in this family
({_join(f"<code>{html_escape(d)}</code>" for d in law.designs)}) hold the CPU cluster
and every other block fixed and vary only the fabric, so a least-squares line through them
separates the two: <b>{law.per_tile_mm2:.4f} mm&sup2; a tile</b> plus
{law.fabric_fixed_mm2:.3f} mm&sup2; the fabric carries at any size. The CPU side is
{law.per_core_mm2:.4f} mm&sup2; a core, which is its caches and nothing else: no absolute core
area is published for the baseline, so that term is itself a floor. Everything
else&nbsp;&mdash; {law.other_fixed_mm2:.3f} mm&sup2;&nbsp;&mdash; moves with neither.</p>
<div class="panel">{scaling_table(law, tiles, cores, cpu_label)}</div>
<p class="note">The line was fitted between {low} and {high} tiles, and {tiles:,g} is
{reach}. {"A line is not evidence outside the points that made it, so this die is an extrapolation on top of a floor." if where != "within" else "The tile count is covered by the fit; the die is still a floor, for the lines nothing states."}</p>"""

    leans = "" if not law.unpriced_blocks else f"""
<p>The floor also leans one way. The blocks nothing prices &mdash;
{_join(html_escape(b) for b in law.unpriced_blocks)} &mdash; are exactly the ones that do not
shrink when the fabric does. They are fixed area, so the smaller the fabric, the larger the
share of the die this page is missing. Missing blocks only ever add area, so at these
densities every die here is smaller than the design it describes, and the fixed term is the
part that is understated.</p>"""
    out["costing"] = f"""
<h3>What this says about cost, and what it does not</h3>
<p>Area is the proxy, and on this page it is the only one. Nothing here carries a wafer price,
a defect density or a yield model, so no figure converts mm&sup2; into money &mdash; and a die
area ratio is not a cost ratio, because yield falls with area and a die twice the size costs
more than twice as much. What the area does support is the comparison a wafer price would not
change the sign of: two configurations at {html_escape(comp.node_id)}, priced the same way.</p>
{leans}"""
    return out


def _amount(value: float) -> str:
    """Engine counts a reader can say out loud: 1,862 rather than
    1.86e+03, and 0.476 rather than 0."""
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.3g}"


def _speed_from_note(note: str) -> float:
    """The speed a mission note states, in m/s. Zero when it states none,
    so a blurb that wants it can only be written for a mission that has
    it."""
    found = re.search(r"([\d.]+)\s*m/s", note or "")
    return float(found.group(1)) if found else 0.0


def _join(items) -> str:
    """Oxford-free list: a, b and c."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _count(n: int) -> str:
    """Small counts read better as words, and must match the list below."""
    return {1: "one", 2: "two", 3: "three"}.get(n, str(n))


def _times(rate_hz: float) -> str:
    """How often, in words that survive a rate of exactly one."""
    if rate_hz == 1:
        return "once a second"
    if rate_hz < 1:
        return f"every {1 / rate_hz:.3g} seconds"
    return f"on each of its {si(rate_hz, '')} calls a second"


def _safety(dossier, soc) -> str:
    """What the model can and cannot say about the safety path.

    Nothing here is written for a particular mission: the stages, the
    counts, the plurals and the named blocks are all read from the
    dossier, because this section renders for every mission with a
    reactive chain and a sentence true of one is false of the others.
    """
    from graphs.estimation.soc.dimensioning import overlap_concern

    by_key = {st.key: st for st in dossier.stages}
    chain = [st for st in dossier.stages if st.on_reactive_chain]
    if not chain:
        return ""
    placed = {pl.stage: pl for pl in dossier.placements}

    rows = []
    for st in chain:
        pl = placed.get(st.key)
        if pl is None:
            rows.append(f"<tr class=\"gap\"><td><b>{html_escape(st.key)}</b></td>"
                        f"<td>{html_escape(st.unit)}</td>"
                        f"<td class=\"num\">{st.rate_hz:g} Hz</td>"
                        f"<td class=\"num\">-</td><td class=\"num\">-</td>"
                        f"<td>no engine prices it</td></tr>")
            continue
        concern = overlap_concern(st.unit)
        verdict = ("closes" if not pl.calls_overlap else
                   f"<b>{pl.calls_in_flight:.3g}x over</b>" if concern == "sequential"
                   else f"{pl.calls_in_flight:.3g}x in flight")
        rows.append(
            f"<tr><td><b>{html_escape(st.key)}</b></td>"
            f"<td>{html_escape(st.unit)}</td>"
            f"<td class=\"num\">{st.rate_hz:g} Hz</td>"
            f"<td class=\"num\">{pl.period_s * 1e3:.3g} ms</td>"
            f"<td class=\"num\">{pl.seconds_per_call * 1e3:.3g} ms</td>"
            f"<td>{verdict} &#183; {html_escape(concern)}</td></tr>")

    broken = [pl for pl in dossier.overlapping
              if pl.stage in by_key and by_key[pl.stage].on_reactive_chain
              and overlap_concern(by_key[pl.stage].unit) == "sequential"]
    finding = ""
    if broken:
        units = sorted({by_key[b.stage].unit for b in broken})
        names = _join(f"<code>{html_escape(b.stage)}</code> at "
                      f"{b.calls_in_flight:.3g}x its period" for b in broken)
        # A filter sentence only where a placed filter exists, and only
        # where its own configuration names what it reads.
        filt = by_key.get("cbf")
        filt_pl = placed.get("cbf") if filt is not None else None
        if filt_pl is not None and not filt_pl.calls_overlap:
            fit = filt.fits[filt_pl.engine]
            finding += (
                f"<p><b>The filter closes; {'its inputs do' if len(broken) > 1 else 'an input it reads does'}"
                f" not.</b> The {filt.rate_hz:g} Hz <code>{html_escape(filt.key)}</code> filter "
                f"takes {filt_pl.seconds_per_call * 1e6:.0f} us of its "
                f"{filt_pl.period_s * 1e3:.3g} ms period and needs "
                f"{_amount(fit.servers_needed)} of a core. That part is comfortable. What is "
                f"not is {names}.</p>")
            # The ESDF link, only when the ESDF is one of the broken
            # stages *and* the filter's own configuration says it reads
            # one. Both halves are checked; neither is assumed.
            esdf = next((b for b in broken if b.stage == "esdf"), None)
            reads_esdf = any("esdf" in str(k).lower() or "esdf" in str(v).lower()
                             for k, v in (filt.config or {}).items())
            if esdf is not None and reads_esdf:
                finding += (
                    f"<p><code>{html_escape(filt.key)}</code> reads its hazards out of the "
                    f"ESDF, so the filter is correct arithmetic over a distance field the "
                    f"machine cannot refresh at the rate the mission asks for. One update "
                    f"takes {esdf.calls_in_flight:.3g}x its own period &mdash; that is a "
                    f"service-time-to-period ratio, not the age of the data. How old the "
                    f"hazards actually are depends on whether late updates queue or are "
                    f"dropped, which is a scheduling policy this model does not carry, so "
                    f"the age is unbounded here rather than {esdf.calls_in_flight:.3g}x.</p>")
        else:
            finding += (f"<p><b>{_count(len(broken)).capitalize()} stage"
                        f"{'s' if len(broken) != 1 else ''} on the sense-to-act chain cannot "
                        f"close {'their' if len(broken) != 1 else 'its'} loop:</b> "
                        f"{names}.</p>")
        finding += (
            f"<p>A server count does not fix "
            f"{'any of them' if len(broken) > 2 else 'either one' if len(broken) == 2 else 'it'}. "
            f"{'Their units are' if len(broken) != 1 else 'Its unit is'} "
            f"{_join(html_escape(u) for u in units)}: the call <i>is</i> the loop iteration, so "
            f"k servers carry k overlapping iterations rather than one faster one. That is a "
            f"latency result, not a throughput result.</p>")

    off_chain = sorted((st for st in dossier.stages if not st.on_reactive_chain),
                       key=lambda st: -st.bytes_per_s)[:2]
    if off_chain and dossier.dram_supply_gb_per_s and off_chain[0].bytes_per_s:
        share = sum(st.bytes_per_s for st in off_chain) / (
            dossier.dram_supply_gb_per_s * 1e9)
        interference = (
            "The chain shares one memory interface with everything else, and the largest other "
            + ("consumers are " if len(off_chain) > 1 else "consumer is ")
            + _join(f"<code>{html_escape(st.key)}</code> at {gb(st.bytes_per_s)}"
                    for st in off_chain)
            + f" &mdash; {share:.0%} of the interface"
            + (" between them" if len(off_chain) > 1 else "")
            + ". Nothing here bounds what that contention does to a period on the chain.")
    else:
        interference = ("The chain shares one memory interface with every other stage, and "
                        "nothing here bounds what that contention does to a period on it.")

    unpriced_chain = [st.key for st in chain if st.key not in placed]
    if unpriced_chain:
        finding += (
            f"<p><b>{_count(len(unpriced_chain)).capitalize()} stage"
            f"{'s' if len(unpriced_chain) != 1 else ''} on the chain "
            f"{'have' if len(unpriced_chain) != 1 else 'has'} no figure at all:</b> "
            + _join(f"<code>{html_escape(k)}</code>" for k in unpriced_chain)
            + ". Nothing prices them on either engine, so they contribute no time to the "
              "table above. The chain's true latency is longer than anything here by an "
              "unknown amount, which is why this page quotes no end-to-end number.</p>")

    island = next((b for b in soc.blocks if b.name == "safety_island"), None) if soc else None
    island_line = ""
    if island is not None:
        anchored = [ln.name for ln in island.lines if ln.anchored]
        unanchored = [ln.name for ln in island.lines if not ln.anchored]
        island_line = (
            "<li><b>The safety island cannot be sized.</b> The design carries one ("
            + _join(html_escape(a) for a in anchored) + " anchored"
            + (", " + _join(html_escape(u) for u in unanchored) + " not" if unanchored else "")
            + "), but the catalogue states no compute capability for it, so no stage can be "
              "placed on it and this page cannot tell you what could run there instead.</li>")

    return f"""
<h2>The safety path</h2>
<p>Sizing answers a rate: can the machine do this much work per second. A safety argument needs
a deadline: does each loop on the sense-to-act chain close inside its own period. Those are
different questions, and this page answers only the first. The table below is the closest it
gets to the second.</p>
<div class="panel"><table><thead><tr><th>stage</th><th>unit</th><th>rate</th><th>period</th>
<th>one call</th><th>verdict</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{finding}
<h3>What this model cannot tell you</h3>
<p>Everything below is required for a safety case and absent here. None of it is a gap we can
close by computing harder; each needs evidence this repository does not hold.</p>
<ul>
<li><b>Worst-case execution time.</b> Every figure on this page is a mean rate from measured
throughput. A deadline is met or missed in the tail, and we model no tail: no cache behaviour,
no contention, no interrupt latency, no scheduler.</li>
<li><b>Freedom from interference.</b> {interference}</li>
{island_line}
<li><b>Integrity level, fault coverage, diagnostic coverage, FIT rates and safe-state
transition time.</b> The catalogue carries none of these, so this page makes no claim about
any standard, and its numbers cannot be cited towards one.</li>
</ul>
<p class="note">Read the table as a necessary condition, not a sufficient one, and read it
against the efficiencies in it. A loop that does not close here does not close <i>at these
efficiencies</i> &mdash; a faster implementation of the same kernel could close it, which is
exactly what section 6 asks for. A loop that closes here has passed only the easiest of the
tests it has to pass.</p>"""


def _comparison_data(dossier, other) -> Optional[dict]:
    """The comparison as data, so a JSON consumer gets what the page has."""
    if other is None:
        return None
    ours = {p.engine: p for p in dossier.provisions}
    theirs = {p.engine: p for p in other.provisions}
    engines = {}
    for name in sorted(set(ours) | set(theirs)):
        a, b = theirs.get(name), ours.get(name)
        if a is None or b is None:
            continue
        engines[name] = {
            "unit": b.unit, "other_needed": a.servers_needed, "needed": b.servers_needed,
            "ratio": (b.servers_needed / a.servers_needed) if a.servers_needed else None,
        }
    theirs_stage = {st.key: st for st in other.stages}
    changed, added = {}, []
    for st in dossier.stages:
        was = theirs_stage.get(st.key)
        if was is None or not was.ops_per_s:
            added.append(st.key)
            continue
        changed[st.key] = {"other_ops_per_s": was.ops_per_s, "ops_per_s": st.ops_per_s,
                           "ratio": st.ops_per_s / was.ops_per_s,
                           "other_rate_hz": was.rate_hz, "rate_hz": st.rate_hz}
    return {
        "mission": dossier.mission, "other_mission": other.mission,
        "note": dossier.note, "other_note": other.note,
        "engines": engines, "stages": changed,
        "added_stages": added,
        "removed_stages": sorted(set(theirs_stage) - {st.key for st in dossier.stages}),
    }


def _factor(ratio: float) -> str:
    """A ratio a reader can hold. Past an order of magnitude either way,
    "0.00265x" is arithmetic nobody reads correctly."""
    if ratio <= 0:
        return "-"
    if ratio < 0.1:
        return f"{1 / ratio:,.0f}x less"
    if ratio > 10:
        return f"{ratio:,.0f}x more"
    return f"{_amount(ratio)}x"


def _comparison(dossier, other) -> str:
    """One mission against another on the same design.

    Two profiles of the same vehicle class differ only in what the world
    is allowed to do, so the difference between their silicon is the price
    of that freedom, stated in cores and tiles.
    """
    if other is None:
        return ""
    ours = {p.engine: p for p in dossier.provisions}
    theirs = {p.engine: p for p in other.provisions}
    rows = []
    for name in sorted(set(ours) | set(theirs)):
        a, b = theirs.get(name), ours.get(name)
        if a is None or b is None:
            continue
        ratio = (b.servers_needed / a.servers_needed) if a.servers_needed else 0.0
        rows.append(
            f"<tr><td>{html_escape(name.upper())}</td>"
            f"<td class=\"num\">{_amount(a.servers_needed)} {a.unit}s</td>"
            f"<td class=\"num\">{_amount(b.servers_needed)} {b.unit}s</td>"
            f"<td class=\"num\">{_factor(ratio)}</td></tr>")
    ours_stage = {st.key: st for st in dossier.stages}
    theirs_stage = {st.key: st for st in other.stages}
    movers, fallers, added = [], [], []
    for key, st in ours_stage.items():
        was = theirs_stage.get(key)
        if was is None or not was.ops_per_s:
            added.append(key)
            continue
        ratio = st.ops_per_s / was.ops_per_s
        if ratio >= 2.0:
            movers.append((ratio, key, was.rate_hz, st.rate_hz))
        elif ratio <= 0.5:
            fallers.append((ratio, key, was.rate_hz, st.rate_hz))
    movers.sort(reverse=True)
    fallers.sort()
    if not rows:
        return ""
    driver = ""
    if movers:
        top = ", ".join(f"<code>{html_escape(k)}</code> {r:.2g}x" for r, k, _a, _b in movers[:6])
        driver += f"<p>Stages whose arithmetic at least doubles: {top}.</p>"
    if fallers:
        # A comparison can run the easier way, and reporting only the
        # increases would make such a pair look as if nothing moved.
        low = ", ".join(f"<code>{html_escape(k)}</code> {r:.2g}x" for r, k, _a, _b in fallers[:6])
        driver += f"<p>Stages whose arithmetic at least halves: {low}.</p>"
    if added:
        # Rendered independently: a comparison can add stages without any
        # existing stage moving, and the guard used to hide that entirely.
        names = ", ".join(f"<code>{html_escape(a)}</code>" for a in sorted(added))
        driver += (f"<p>Stages present only in {html_escape(dossier.title)}: {names}.</p>")
    # ...and the same the other way. A stage the other mission has and
    # this one does not is invisible in a ratio, and it can be most of
    # the difference between the two columns above, so it is named with
    # what it cost over there.
    dropped = []
    their_place = {p.stage: p.engine for p in other.placements}
    for key, st in theirs_stage.items():
        if ours_stage.get(key) is not None and ours_stage[key].ops_per_s:
            continue
        engine = their_place.get(key)
        fit = st.fits.get(engine) if engine else None
        dropped.append((key, engine, fit.servers_needed if fit and fit.fits else None))
    if dropped:
        parts = []
        for key, engine, needed in sorted(dropped):
            prov = next((p for p in other.provisions if p.engine == engine), None)
            cost = ("no engine prices it there" if needed is None else
                    f"{_amount(needed)} of that mission's "
                    f"{_amount(prov.servers_needed)} {prov.unit}s" if prov else
                    f"{_amount(needed)} {engine}")
            parts.append(f"<code>{html_escape(key)}</code> ({cost})")
        driver += (f"<p>Stages present only in {html_escape(other.title)}: "
                   f"{_join(parts)}.</p>")
    # No claim about vehicle class or ordering: --compare takes any two
    # missions, so the notes carry the difference and the page does not
    # assert a cause.
    notes = (f"<p>What each states: <i>{html_escape(other.note)}</i> against "
             f"<i>{html_escape(dossier.note)}</i>.</p>")
    return f"""
<h2>Against {html_escape(other.title)}</h2>
<p>Both missions sized on the same design, the same efficiency table and the same target
utilization, so the engine demands differ only where the workloads do.</p>
{notes}
<div class="panel"><table><thead><tr><th>engine</th>
<th>{html_escape(other.title)}</th><th>{html_escape(dossier.title)}</th><th>factor</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{driver}
<p class="note">Neither figure is a budget: both are what the demand needs, and both omit the
stages no engine can price. The ratio is what a product decision turns on.</p>"""


def _memory_options(demand_gb_per_s: float):
    """Catalogued memory interfaces that carry a demand, widest last."""
    library = load_ip_library()
    out = []
    for name, template in library.items():
        interface = getattr(template, "memory_interface", None)
        peak = getattr(interface, "peak_gb_per_s", None) if interface else None
        if peak and peak >= demand_gb_per_s:
            out.append((name, peak))
    return sorted(out, key=lambda kv: kv[1])


def _crosscheck(dossier, derived) -> str:
    """Our demand model against an external document's published figures.

    This checks the *workload*, not the sizing: the published numbers are
    for a pooled reference machine with its own assumed throughputs, so
    they say nothing about this SoC. Agreement means our operation counts
    and precision mix are not invented.
    """
    published = dossier.published
    if not published or not derived:
        return ""
    derived = dict(derived)
    # The same quantity by the same definition: the sum of per-stage byte
    # counts at the mission's rates.
    derived.setdefault("dram_gb_per_s", dossier.dram_demand_gb_per_s)
    rows, deltas = [], []
    for label, key, fmt in (("Total arithmetic", "tops", "{:.2f} TOP/s"),
                            ("Class A (INT8) share", "class_a_share", "{:.3f}"),
                            ("Class C (FP32) share", "class_c_share", "{:.3f}"),
                            ("Compute oversubscription", "oversubscription", "{:.2f}x"),
                            ("Memory traffic", "dram_gb_per_s", "{:.1f} GB/s")):
        if key not in published or key not in derived:
            continue
        ours, theirs = derived[key], published[key]
        delta = abs(ours - theirs) / theirs if theirs else 0.0
        deltas.append((delta, label, key, theirs))
        rows.append(f"<tr><td>{html_escape(label)}</td>"
                    f"<td class=\"num\">{fmt.format(ours)}</td>"
                    f"<td class=\"num\">{fmt.format(theirs)}</td>"
                    f"<td class=\"num\">{delta:.1%}</td></tr>")
    if not rows:
        return ""
    worst_delta, worst_label, worst_key, worst_value = max(deltas)
    # A large relative difference on a quantity that is a fraction of a
    # percent says less than the number suggests, and saying so is more
    # honest than quoting it bare or quietly dropping it.
    tiny = worst_key.endswith("_share") and worst_value < 0.05
    qualifier = (f" &mdash; a quantity that is {worst_value:.1%} of the workload, where the "
                 f"published figure carries one significant figure" if tiny else "")
    return f"""
<h2>Cross-check against the published annex</h2>
<p>This mission is one of two the companion annex publishes figures for. Ours are derived
independently from per-stage operation and byte counts; theirs come from
<i>BranesAI-Autonomy-Compute-Requirements</i> section 4.</p>
<div class="panel"><table><thead><tr><th>quantity</th><th>ours</th><th>published</th>
<th>difference</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">{len(rows)} independent quantities, and the largest disagreement is
{worst_delta:.1%} ({html_escape(worst_label.lower())}){qualifier}. <b>This validates the demand
model, not the sizing.</b> The published oversubscription is against the annex's own pooled
machine &mdash; one throughput figure per precision class, no engines &mdash; so it is not
comparable to the per-engine utilizations in section 6, and we do not treat it as if it were.</p>"""


def _analysis(dossier, f, kpu, cpu) -> str:
    """Step 1: the mapping, and why it is forced rather than chosen."""
    lines = []
    for engine, rows in f["by_engine"].items():
        if not rows:
            continue
        other = "cpu" if engine == "kpu" else "kpu"
        example = rows[0][0]
        gap = example.fits[other].gap or "no figure"
        lines.append(
            f'<li><b>{len(rows)} stage(s) go to the {engine.upper()}</b>, headed by '
            f'<code>{html_escape(example.key)}</code>. On the {other.upper()} they have no fit: '
            f'<i>{html_escape(gap)}</i>.</li>')
    unpriced = f["unpriced"]
    if unpriced:
        lines.append(
            f'<li><b>{len(unpriced)} stage(s) go nowhere:</b> '
            + ", ".join(f'<code>{html_escape(s.key)}</code>' for s in unpriced)
            + '. Neither engine has a figure for every precision class they need, so they are '
            'left out of the sizing rather than guessed at.</li>')
    return f"""
<h3>1. The mapping is forced, not chosen</h3>
<p>A precision class runs in the narrowest format the engine offers at or above its floor, and
that (kernel class, engine, format) triple needs an efficiency from somewhere &mdash; a
measurement or the domain-flow ceiling. Where there is none, there is no number, and we do not
invent one.</p>
<ul>{''.join(lines)}</ul>"""


def _sizing_steps(dossier, f, kpu, cpu, alt, catalogued_tiles: int) -> str:
    """Steps 2 and 3: the sizing, and what actually binds."""
    parts = ["<h3>2. Size each engine</h3>"]
    for prov in dossier.provisions:
        rows = f["by_engine"].get(prov.engine, [])
        top, top_fit = (rows[0] if rows else (None, None))
        share = (top_fit.servers_needed / prov.servers_needed) if top_fit and prov.servers_needed else 0
        worst_e = min((c.efficiency for _s, fit in rows for c in fit.classes if c.efficiency),
                      default=None)
        parts.append(
            f"<p><b>{prov.engine.upper()}.</b> {len(rows)} stages need "
            f"<b>{_amount(prov.servers_needed)} {prov.unit}s</b> between them, against the "
            f"{prov.servers_provisioned} this design has. "
            + (f"<code>{html_escape(top.key)}</code> alone is {share:.1%} of that"
               + (f", at an efficiency of {min((c.efficiency for c in top_fit.classes if c.efficiency), default=0):.2%}"
                  if worst_e is not None else "") + ".</p>"
               if top is not None else "</p>"))
    supply = f["bytes_supply"]
    byte_top = f["bytes_top"]
    parts.append("<h3>3. What binds</h3><ul>")
    for name, ratio in f["pressure"]:
        verdict = "over" if ratio > 1 else "clear"
        parts.append(f'<li><b>{html_escape(name)}:</b> {ratio:.0%} '
                     f'{"&mdash; <b>over capacity</b>" if verdict == "over" else "of what is there"}'
                     f'.</li>')
    parts.append("</ul>")
    if supply and dossier.dram_demand_gb_per_s > (dossier.dram_supply_gb_per_s or 0):
        options = _memory_options(dossier.dram_demand_gb_per_s)
        if options:
            parts.append(
                "<p><b>Memory is the one constraint here the parts list can answer.</b> The "
                f"mission asks {dossier.dram_demand_gb_per_s:.0f} GB/s and this design supplies "
                f"{dossier.dram_supply_gb_per_s:g}. On bandwidth alone, "
                + _join(f"<code>{html_escape(name)}</code> at {peak:g} GB/s would carry it at "
                        f"{dossier.dram_demand_gb_per_s / peak:.0%}"
                        for name, peak in options)
                + ". These are bandwidth-qualified candidates and nothing more: a wider LPDDR "
                  "interface changes the PHY and the floorplan, and an HBM stack changes the "
                  "controller, the package and the thermal design as well. Whether either is "
                  "compatible with this SoC is a question this model does not answer. What it "
                  "does say is that the shortfall is within reach of parts we already "
                  "catalogue, which is not true of either engine.</p>")
        else:
            parts.append(
                "<p><b>No interface in the parts list carries this mission's memory "
                f"traffic.</b> It asks {dossier.dram_demand_gb_per_s:.0f} GB/s and the widest "
                "we catalogue is short of it.</p>")
    unpriced_bytes = sum(st.bytes_per_s for st in dossier.stages
                         if st.key in dossier.unplaced)
    if unpriced_bytes and dossier.dram_demand_gb_per_s:
        share = unpriced_bytes / (dossier.dram_demand_gb_per_s * 1e9)
        if share > 0.2:
            parts.append(
                f"<p>Note what that traffic is made of: <b>{share:.0%} of it comes from stages "
                f"no engine can price</b> &mdash; "
                + _join(f"<code>{html_escape(k)}</code>" for k in dossier.unplaced)
                + ". Bytes are counted whether or not an engine can run the work, so the memory "
                  "figure is complete where the engine figures are floors"
                + (", and on this mission the majority of the memory demand comes from the "
                   "stages the compute numbers leave out." if share > 0.5
                   else f", and {share:.0%} of the memory demand comes from stages the compute "
                        f"numbers leave out.")
                + "</p>")
    if supply and byte_top:
        over_memory = dossier.dram_demand_gb_per_s > (dossier.dram_supply_gb_per_s or 0)
        top_share = byte_top.bytes_per_s / (dossier.dram_demand_gb_per_s * 1e9)
        detail = (f"at {byte_top.rate_hz:g} Hz, a {si(byte_top.bytes_per_call, 'B')} working "
                  f"set pulled through DRAM on every call")
        if over_memory:
            # The paragraphs above already give demand against supply, so
            # this one says only what the traffic is made of.
            lead = ("Most of it is one stage: " if top_share >= 0.5
                    else "The largest single contributor is ")
            parts.append(
                f"<p>{lead}<code>{html_escape(byte_top.key)}</code> at {top_share:.0%} of the "
                f"traffic &mdash; {detail}.</p>")
        else:
            parts.append(
                f"<p>Memory is worth spelling out. The interface carries "
                f"{dossier.dram_demand_gb_per_s:.0f} GB/s of compulsory traffic against "
                f"{dossier.dram_supply_gb_per_s:g} GB/s of peak, and "
                f"<code>{html_escape(byte_top.key)}</code> is {top_share:.0%} of it on its own "
                f"&mdash; {detail}.</p>")
    return "".join(parts)


def _the_ask(dossier, f, kpu, cpu, target_utilization: float, alt=None,
             catalogued_tiles: int = 0) -> str:
    """Steps 4 and 5: what the accelerator is worth, and the CPU ask."""
    parts = []
    if cpu and cpu.servers_needed:
        rows = f["by_engine"].get("cpu", [])
        worst = [(s, fit, min((c.efficiency for c in fit.classes if c.efficiency), default=1.0))
                 for s, fit in rows]
        worst.sort(key=lambda r: r[1].servers_needed or 0, reverse=True)
        top = worst[:3]
        covered = sum(r[1].servers_needed for r in top) / cpu.servers_needed
        parts.append("<h3>4. The CPU ask</h3>")
        parts.append(
            f"<p>The CPU needs <b>{_amount(cpu.servers_needed)} cores</b> and the design has "
            f"{cpu.servers_provisioned}. "
            + (f"The gap is not spread evenly: {_count(len(top))} of the {len(worst)} stages "
               f"on it are {covered:.0%} of the demand.</p><ul>"
               if len(top) < len(worst) else "</p><ul>"))
        for stage, fit, eff in top:
            parts.append(
                f"<li><code>{html_escape(stage.key)}</code> "
                f"(<code>{html_escape(stage.kernel_class)}</code>): "
                f"<b>{_amount(fit.servers_needed)} cores</b> at <b>{eff:.2%}</b> of FP32 peak.</li>")
        parts.append("</ul>")
        by_class = {}
        for stage, fit, eff in worst:
            by_class.setdefault(stage.kernel_class, [0.0, eff])
            by_class[stage.kernel_class][0] += fit.servers_needed or 0.0
        ranked = sorted(by_class.items(), key=lambda kv: -kv[1][0])
        lead, (lead_cores, lead_eff) = ranked[0]
        parts.append(
            f"<p>Grouped by kernel class, the answer is blunter still: "
            f"<code>{html_escape(lead)}</code> is "
            f"<b>{_amount(lead_cores)} of the {_amount(cpu.servers_needed)} cores</b> "
            f"({lead_cores / cpu.servers_needed:.0%}) at {lead_eff:.2%} of peak. "
            f"So the question for an Andes core is not a general one:</p>")
        total_lead_ops = sum(s.ops_per_s for s, _fit, _e in worst
                             if s.kernel_class == lead)
        parts.append(
            f"<blockquote><b>What does it take to run "
            f"{si(total_lead_ops, 'OP/s')} of <code>{html_escape(lead)}</code> "
            f"in FP32 at materially better than {lead_eff:.2%} of peak?</b></blockquote>")
        ten = lead_cores * lead_eff / 0.10
        after = provision(cpu.servers_needed - lead_cores + ten, target_utilization)
        # Whether that lands inside the design is the finding, and "N
        # instead of N" is not a sentence. Three cases, and the data picks.
        if after < cpu.servers_provisioned:
            lands = (f"<b>{after:g} cores provisioned instead of "
                     f"{cpu.servers_provisioned}</b>")
        elif after == cpu.servers_provisioned:
            lands = (f"<b>{after:g} cores provisioned &mdash; exactly what this design "
                     f"already has</b>. One kernel class is the whole shortfall")
        else:
            lands = (f"{after:g} cores provisioned, still past the "
                     f"{cpu.servers_provisioned} this design has")
        parts.append(
            f"<p>The leverage is enormous because the baseline is so low. At 10% of peak "
            f"instead of {lead_eff:.2%}, that class falls from {_amount(lead_cores)} cores to "
            f"{_amount(ten)}, and the whole CPU requirement from {_amount(cpu.servers_needed)} to "
            f"{_amount(cpu.servers_needed - lead_cores + ten)} &mdash; {lands}.</p>")
    if kpu and kpu.servers_needed:
        rows = f["by_engine"].get("kpu", [])
        rows.sort(key=lambda r: -(r[1].servers_needed or 0))
        top, top_fit = rows[0]
        eff = min((c.efficiency for c in top_fit.classes if c.efficiency), default=0)
        share = (top_fit.servers_needed or 0) / kpu.servers_needed
        others = sum(r[1].servers_needed or 0 for r in rows[1:])
        # One stage dominating the fabric at a low ceiling is a bandwidth
        # story. Anything else is the ordinary "what is the accelerator
        # worth" story. They are different arguments, so the data picks.
        if share > 0.5 and eff < 0.10 and len(rows) > 1:
            rest = ", ".join(f"<code>{html_escape(r[0].key)}</code>" for r in rows[1:])
            parts.append("<h3>5. The accelerator ask is a memory ask</h3>")
            parts.append(
                f"<p><code>{html_escape(top.key)}</code> is {share:.1%} of the "
                f"{_amount(kpu.servers_needed)}-tile requirement, and not because it is the "
                f"biggest arithmetic: it is {top.ops_per_s / f['total_ops']:.0%} of the "
                f"operations. It is because its domain-flow ceiling is <b>{eff:.2%}</b>. The "
                f"fabric is not computing, it is waiting: {si(top.bytes_per_call, 'B')} of "
                f"weights crossing DRAM {_times(top.rate_hz)}.</p>")
            # Where another stage on the same engine runs at a wildly
            # better ceiling, the pair says more than either alone: it is
            # the same fabric, the same second, and the difference is
            # what the kernel does with its weights.
            priced = [(r[0], r[1], min((cl.efficiency for cl in r[1].classes
                                        if cl.efficiency), default=0.0))
                      for r in rows[1:]]
            # The biggest other arithmetic load, not the highest ceiling:
            # "the stage doing the most work needs the fewest tiles" is
            # the statement worth making, and it is the stronger one.
            best = max((r for r in priced if r[2] > 0),
                       key=lambda r: r[0].ops_per_s, default=None)
            if best is not None and best[2] > 10 * eff:
                other, other_fit, other_eff = best
                tiles_ratio = (top_fit.servers_needed or 0) / (other_fit.servers_needed or 1)
                # Say it from whichever side makes the sentence true: the
                # comparand does not always do more arithmetic.
                # Whichever stage the lead names first, the pairs that
                # follow name in the same order.
                other_first = other.ops_per_s >= top.ops_per_s
                if other_first:
                    lead = (f"<code>{html_escape(other.key)}</code> does "
                            f"{other.ops_per_s / top.ops_per_s:.2g}x the arithmetic of "
                            f"<code>{html_escape(top.key)}</code> &mdash; "
                            f"{si(other.ops_per_s, 'OP/s')} against "
                            f"{si(top.ops_per_s, 'OP/s')} &mdash; and needs "
                            f"{tiles_ratio:.0f}x <i>fewer</i> tiles: "
                            f"{_amount(other_fit.servers_needed or 0)} against "
                            f"{_amount(top_fit.servers_needed or 0)}")
                else:
                    lead = (f"<code>{html_escape(top.key)}</code> does only "
                            f"{top.ops_per_s / other.ops_per_s:.2g}x the arithmetic of "
                            f"<code>{html_escape(other.key)}</code> &mdash; "
                            f"{si(top.ops_per_s, 'OP/s')} against "
                            f"{si(other.ops_per_s, 'OP/s')} &mdash; and needs "
                            f"{tiles_ratio:.0f}x <i>more</i> tiles: "
                            f"{_amount(top_fit.servers_needed or 0)} against "
                            f"{_amount(other_fit.servers_needed or 0)}")
                # Arithmetic intensity only where both stages state bytes.
                ours_ai = (top.ops_per_call / top.bytes_per_call
                           if top.bytes_per_call else None)
                theirs_ai = (other.ops_per_call / other.bytes_per_call
                             if other.bytes_per_call else None)
                first_ai, second_ai = ((theirs_ai, ours_ai) if other_first
                                       else (ours_ai, theirs_ai))
                first_eff, second_eff = ((other_eff, eff) if other_first
                                         else (eff, other_eff))
                if ours_ai and theirs_ai:
                    intensity = (f" What separates them is arithmetic intensity &mdash; "
                                 f"{first_ai:.3g} operations per byte against "
                                 f"{second_ai:.3g} &mdash; and what the domain-flow model "
                                 f"makes of it: a ceiling of {first_eff:.2%} against "
                                 f"{second_eff:.2%}.")
                else:
                    intensity = (f" Their domain-flow ceilings are {first_eff:.2%} and "
                                 f"{second_eff:.2%}.")
                parts.append(
                    f"<p><b>The comparison is on this page.</b> {lead}. "
                    f"Same fabric, same second.{intensity}</p>")
                # The "worse than its bytes imply" reading only holds when
                # the dominant stage is the less intense of the two.
                caveat = (
                    f"<p class=\"note\">Neither ceiling has been measured. Both are upper "
                    f"bounds, so both tile counts are lower bounds &mdash; and the "
                    f"<code>{html_escape(top.key)}</code> figure carries most of the risk on "
                    f"this page, because it sets {share:.0%} of the requirement. One "
                    f"measurement of <code>{html_escape(top.kernel_class)}</code> on this "
                    f"fabric would settle it.")
                if ours_ai and theirs_ai and theirs_ai > ours_ai:
                    caveat += (
                        f" Note the shape of the claim: an intensity ratio of "
                        f"{theirs_ai / ours_ai:.2g}x produces an efficiency ratio of "
                        f"{other_eff / eff:.0f}x, so the model is saying a wavefront fabric "
                        f"handles <code>{html_escape(top.kernel_class)}</code> far worse than "
                        f"its byte count alone implies.")
                parts.append(caveat + "</p>")
            parts.append(
                f"<p>Take it out and the rest of the accelerator work &mdash; {rest} &mdash; "
                f"needs <b>{_amount(others)} tile{'s' if others != 1 else ''}</b>. That is the "
                f"shape of the decision: this is not a mission that needs a "
                f"{_amount(kpu.servers_needed)}-tile fabric, it is a mission with a "
                f"weight-streaming problem bolted to a {_amount(others)}-tile one. "
                f"Quantisation, weight caching in on-die SRAM, a smaller model or a wider "
                f"memory interface are all attacks on the same number; more tiles is not.</p>")
        elif alt and alt.get("servers_needed"):
            mine = next((r for r in rows if r[0].key == alt["stage"]), None)
            on_kpu = (mine[1].servers_needed or 0) if mine else kpu.servers_needed
            watts = dossier.datapath_watts.get("kpu", 0.0)
            parts.append("<h3>5. What the accelerator is worth</h3>")
            parts.append(
                f"<p><code>{html_escape(alt['stage'])}</code> has no measured INT8 figure on a "
                f"CPU, but its INT8 class may run in any wider format and FP32 <i>is</i> "
                f"measured. At E = {alt['efficiency']:.1%} one core delivers "
                f"{si(alt['throughput_per_server'], 'OP/s')}, so it would need "
                f"<b>{_amount(alt['servers_needed'])} CPU cores</b> and "
                f"{alt['watts'] * 1e3:.0f} mW &mdash; against {_amount(on_kpu)} of a tile and "
                f"{watts * 1e3:.0f} mW. That is "
                f"<b>{alt['servers_needed'] / max(on_kpu, 1e-9):.0f} cores of work per "
                f"tile</b>"
                + (f" at {alt['watts'] / watts:.1f}x less energy.</p>" if watts else ".</p>"))
            if catalogued_tiles and kpu.servers_needed:
                # How small a fabric, from the mission's own sizing. Saying
                # "one tile" on a mission that needs five is the same
                # defect as any other sentence written against one page.
                want = provision(kpu.servers_needed, target_utilization)
                parts.append(
                    f"<p>It is equally an argument for a <i>small</i> fabric: this mission "
                    f"sizes to {want:g} tile{'s' if want != 1 else ''}, and the catalogued "
                    f"part has {catalogued_tiles} &mdash; "
                    f"{catalogued_tiles / kpu.servers_needed:.0f}x more than it can use.</p>")
    return "".join(parts)


def html_escape(text: str) -> str:
    import html as _html

    return _html.escape(text or "")


def _cams(dossier) -> str:
    cams = dossier.sensors.get("mono")
    if isinstance(cams, (list, tuple)) and len(cams) == 4:
        return f"{cams[0]} x {cams[1]}x{cams[2]} streams at {cams[3]} Hz"
    return "the stated sensor configuration"


def _status(ok: Optional[bool], text: str) -> str:
    """A requirement is met, missed, or not checkable. Never say "met"
    because a number exists."""
    if ok is None:
        return f"NOT CHECKED: {text}"
    return ("met: " if ok else "NOT MET: ") + text


def requirements_rows(dossier, alt):
    """(requirement, figure, source, status).

    Product-level rather than per-stage: one row per engine covering every
    stage placed on it, so nothing a mission demands goes unreported. The
    per-stage detail is the fit table in section 6.
    """
    rows = []
    # Camera ingest is the camera stage's own traffic. Taking the first
    # pipeline stage instead reported a lidar byte rate under a camera
    # heading.
    camera = next((st for st in dossier.stages if st.key == "mono"), None)
    if camera is not None:
        rows.append(("Camera ingest", si(camera.bytes_per_s, "B/s"),
                     f"mission profile sensors.mono, stage {camera.key}",
                     _status(True, _cams(dossier))))
    sensing = [st for st in dossier.stages if st.tier == "T1"]
    if sensing:
        front_end = sum(st.bytes_per_s for st in sensing)
        supply = (dossier.dram_supply_gb_per_s or 0) * 1e9
        # A subset over the interface settles it on its own. A subset
        # under it settles nothing, because every other stage shares the
        # same interface.
        over = bool(supply) and front_end > supply
        rows.append(("Sensor front end, all paths", si(front_end, "B/s"),
                     f"{len(sensing)} tier-1 stages: "
                     + ", ".join(st.key for st in sensing),
                     _status(False if over else None,
                             f"the tier-1 paths alone exceed the "
                             f"{dossier.dram_supply_gb_per_s:g} GB/s interface" if over
                             else "sums the tier-1 paths; whether the interface carries "
                                  "them is the memory row below")))
    for prov in dossier.provisions:
        ops = sum(s.ops_per_s for s in dossier.stages if s.key in prov.stages)
        rows.append((f"{prov.engine.upper()} throughput",
                     f"{si(ops, 'OP/s')} over {len(prov.stages)} stage"
                     f"{'s' if len(prov.stages) != 1 else ''}",
                     "per-stage operation counts at the mission's rates",
                     _status(prov.utilization <= 1.0,
                             f"{_amount(prov.servers_needed)} of "
                             f"{prov.servers_provisioned} {prov.unit}"
                             f"{'s' if prov.servers_provisioned != 1 else ''} "
                             f"({prov.utilization:.0%})")))
    if dossier.unplaced:
        rows.append(("Stages with no engine", f"{len(dossier.unplaced)} of "
                     f"{len(dossier.stages)}",
                     "no efficiency figure for a precision class they need",
                     _status(False, ", ".join(dossier.unplaced))))
    # One chain stage that cannot finish inside the whole budget settles
    # the question, whatever the unpriced stages would have added.
    busted = dossier.chain_stages_over_deadline(dossier.stages)
    if busted:
        worst = max(busted, key=lambda p: p.seconds_per_call)
        rows.append(("Sense-to-act latency", f"{dossier.deadline_ms:g} ms budget",
                     "mission profile deadline_ms",
                     _status(False, f"{worst.stage} alone takes "
                                    f"{ms(worst.seconds_per_call)} per call, at the modelled "
                                    f"efficiencies")))
    elif dossier.chain_seconds is not None:
        rows.append(("Sense-to-act latency", f"{dossier.deadline_ms:g} ms budget",
                     "mission profile deadline_ms",
                     _status(dossier.deadline_headroom >= 1.0,
                             f"{ms(dossier.chain_seconds)}, "
                             f"{dossier.deadline_headroom:.1f}x headroom")))
    else:
        rows.append(("Sense-to-act latency", f"{dossier.deadline_ms:g} ms budget",
                     "mission profile deadline_ms",
                     _status(None, f"{len(dossier.unplaced)} stage(s) unpriced, so a chain "
                                   f"total would omit them")))
    # A datapath floor under budget does not mean the budget is met: the
    # model carries no memory, clock-tree, leakage or idle term.
    used = dossier.power_budget_fraction_used
    # A floor above budget already proves the budget is missed; a floor
    # under it proves nothing, because the terms it omits only add.
    rows.append(("Power budget", f"{dossier.power_budget_w:g} W",
                 "mission profile power_budget_w",
                 _status(False if (used is not None and used > 1.0) else None,
                         f"datapath floor {dossier.datapath_total_w * 1e3:.0f} mW = "
                         f"{(used or 0):.1%} of budget"
                         + (", and the floor alone is over" if used and used > 1.0 else
                            ", with no memory, clock-tree, leakage or idle term in it"))))
    rows.append(("Full-SoC power", "-",
                 "no memory, clock-tree, leakage or idle term in this model", "gap"))
    if dossier.dram_supply_gb_per_s:
        used = dossier.dram_demand_gb_per_s / dossier.dram_supply_gb_per_s
        # Every stage's bytes are counted, placed or not, so this one is
        # not understated the way the engine figures are.
        note = f"{used:.1%} of {dossier.dram_supply_gb_per_s:g} GB/s peak"
        rows.append(("Memory bandwidth", f"{dossier.dram_demand_gb_per_s:.2f} GB/s demand",
                     "sum of per-stage byte counts", _status(used <= 1.0, note)))
    else:
        rows.append(("Memory bandwidth", f"{dossier.dram_demand_gb_per_s:.2f} GB/s demand",
                     "sum of per-stage byte counts", "gap"))
    caveat = SCOPE_CAVEATS.get(dossier.mission)
    if caveat:
        figure, why = caveat
        rows.append(("Scope of this profile", figure, why, "gap"))
    rows.append(("Thermal limit", "-", "no cooling solution attached to this design", "gap"))
    rows.append(("Size / volume", "-", "not a field of the mission profile", "gap"))
    rows.append(("Weight", "-", "not a field of the mission profile", "gap"))
    rows.append(("Silicon area", "KPU sized, CPU open",
                 "KPU from a fit over 4 catalogued cores; no absolute core area is published "
                 "for the CPU baseline", "gap"))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--mission", default="edge_ai_device_multi_stream_tracking__anomaly",
                        help="Mission id (default: the edge tracking appliance)")
    parser.add_argument("--design", default="kpu_t128_n7", help="Design id to size from")
    parser.add_argument("--efficiency", default="orin_nano_measured_v1")
    parser.add_argument("--target-utilization", type=float, default=0.85,
                        help="Size each engine to at most this utilization (default 0.85)")
    parser.add_argument("--format", choices=["html", "json"], default=None,
                        help="Default: from the --output suffix, else html")
    parser.add_argument("--compare", help="Another mission id to size on the same design and "
                                          "show the difference against")
    parser.add_argument("--cpu-label", default="Andes RISC-V",
                        help="What to call the CPU complex in the diagrams")
    parser.add_argument("--cpu-baseline", default="X, U and E from the A78AE baseline",
                        help="Whose figures the CPU block is showing, said on the block")
    parser.add_argument("--output", "-o", help="Write here (default: stdout)")
    args = parser.parse_args(argv)
    if args.format is None:
        args.format = "json" if (args.output or "").lower().endswith(".json") else "html"

    try:
        if not 0 < args.target_utilization <= 1:
            raise ValueError("--target-utilization must be in (0, 1]")
        dossier, soc, _tiles_n = build(args.mission, args.design, args.efficiency,
                                       args.target_utilization)
        alt = counterfactual(dossier, soc, "det", "cpu", "fp32")
        composition = compose_resources(soc)
        law = scaling_law_for(args.design)
        fit = tile_area_fit(args.design, law)
        crosscheck = pooled_figures(args.mission)
        compare = None
        if args.compare:
            compare, _soc2, _t2 = build(args.compare, args.design, args.efficiency,
                                        args.target_utilization)
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        write_report(json.dumps({
            "mission": dossier.mission, "design": dossier.design, "node": dossier.node,
            "frame_hz": dossier.frame_hz,
            "stages": [{
                "key": s.key, "tier": s.tier, "kernel_class": s.kernel_class,
                "rate_hz": s.rate_hz, "ops_per_s": s.ops_per_s, "bytes_per_s": s.bytes_per_s,
                "class_split": s.class_split,
                "fits": {n: {"servers_needed": f.servers_needed, "provenance": f.provenance,
                             "gap": f.gap} for n, f in s.fits.items()},
            } for s in dossier.stages],
            "provisions": [{
                "engine": p.engine, "unit": p.unit, "servers_needed": p.servers_needed,
                "servers_provisioned": p.servers_provisioned,
                "throughput_ops_per_s": p.throughput_ops_per_s,
                "utilization": p.utilization, "efficiency": p.efficiency,
                "peak_ops_per_s": p.peak_ops_per_s, "peak_format": p.peak_format,
                "provenance": p.provenance,
            } for p in dossier.provisions],
            "chain_seconds": dossier.chain_seconds,
            "deadline_headroom": dossier.deadline_headroom,
            "dram_demand_gb_per_s": dossier.dram_demand_gb_per_s,
            "dram_supply_gb_per_s": dossier.dram_supply_gb_per_s,
            "datapath_watts": dossier.datapath_watts,
            "counterfactual": alt, "tile_area_fit": fit,
            "floorplan": {
                "die_area_mm2": composition.die_area_mm2,
                "die_side_mm": composition.die_side_mm,
                "core_area_mm2": composition.core_area_mm2,
                "block_area_mm2": composition.block_area_mm2,
                "io_ring_area_mm2": composition.io_ring_area_mm2,
                "transistors_mtx": composition.transistors_mtx,
                "gates_m_nand2": composition.gates_m,
                "sram_kib": composition.sram_kib,
                "complete": composition.complete,
                "unpriced_lines": [f"{ln.block}.{ln.name}" for ln in composition.gaps],
                "blocks": [{
                    "name": b.name, "ip": b.ip, "count": b.count,
                    "engine_kind": b.engine_kind, "area_mm2": b.area_mm2,
                    "transistors_mtx": b.transistors_mtx, "gates_m_nand2": b.gates_m,
                    "sram_kib": b.sram_kib, "complete": b.complete,
                } for b in composition.blocks],
                "by_circuit_class": [{
                    "circuit_class": c.circuit_class.value, "library": c.library,
                    "mtx_per_mm2": c.mtx_per_mm2, "transistors_mtx": c.transistors_mtx,
                    "area_mm2": c.area_mm2, "sram_kib": c.sram_kib,
                    "gates_m_nand2": c.gates_m, "confidence": c.density_confidence,
                } for c in composition.classes],
            },
            "scaling_law": None if law is None else {
                "per_tile_mm2": law.per_tile_mm2,
                "fabric_fixed_mm2": law.fabric_fixed_mm2,
                "per_core_mm2": law.per_core_mm2,
                "other_fixed_mm2": law.other_fixed_mm2,
                "cores_per_cluster": law.cores_per_cluster,
                "tiles_fitted": list(law.tiles_fitted),
                "designs": list(law.designs),
                "unpriced_blocks": list(law.unpriced_blocks),
            },
            "published": dict(dossier.published),
            "crosscheck": crosscheck,
            "comparison": _comparison_data(dossier, compare),
            "efficiency_table": args.efficiency,
            "counterfactual_table": COUNTERFACTUAL_TABLE,
            "oversubscribed": list(dossier.oversubscribed),
            "unplaced": list(dossier.unplaced),
            "confidence": dossier.estimation_confidence.level.value,
        }, indent=2), args.output)
        return 0

    alternatives = []
    cpu_prov = next((p for p in dossier.provisions if p.kind == "cpu"), None)
    if alt and cpu_prov:
        # Measured against the CPU the design actually has, not against a
        # hypothetical cluster sized to fit it -- the point of the row is
        # how far out of reach it is.
        alternatives.append({
            "label": "if det ran on the CPU instead", "unit": "core",
            "needed": alt["servers_needed"],
            "provisioned": cpu_prov.servers_provisioned,
            "kind": "cpu", "rejected": True,
            "note": f"FP32 at E {alt['efficiency']:.1%} - measured - "
                    f"{alt['watts'] * 1e3:.0f} mW of datapath",
        })
    idle = [("ISP", "no capability stated"), ("codec", "not used")]
    # Attach the sensor path to the stage that actually consumes it, not
    # to whichever stage happens to sort first.
    # Named for the stage it measures: a mission with a stereo pair has
    # camera traffic this path does not carry.
    camera = next((st for st in dossier.stages if st.key == "mono"), None)
    ingress = ("mono cameras", camera.bytes_per_s) if camera else None
    write_report(render(dossier, sections(dossier, soc, alt, fit, _tiles_n, args.efficiency,
                                 args.target_utilization, args.output or "",
                                 crosscheck, compare, composition, law, args.cpu_label),
                        requirements_rows(dossier, alt), alternatives, idle,
                        date.today().isoformat(), ingress, args.cpu_label,
                        {"cpu": args.cpu_baseline}, composition), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
