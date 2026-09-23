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
from typing import Dict, List, Optional

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
from graphs.hardware.soc.kpu_cores import KPU_CORES, sku_of  # noqa: E402
from graphs.reporting.mission_dossier import gb, ms, render, si  # noqa: E402
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


def tile_area_fit():
    """Marginal area per tile, fitted over the catalogued N7 cores. Used to
    price a fabric smaller than any SKU in the catalogue, which is an
    extrapolation and is labelled as one."""
    designs, lib, nodes = load_designs(), load_ip_library(), load_process_nodes()
    points = []
    for name in ("kpu_t64_n7", "kpu_t128_n7", "kpu_t256_n7", "kpu_t512_n7"):
        if name not in designs:
            continue
        soc = compose_soc(designs[name], lib, nodes, None)
        block = next((b for b in soc.blocks if b.name == "kpu"), None)
        core = next((b.ip for b in designs[name].blocks if b.ip in KPU_CORES), None)
        if block is None or block.area_mm2 is None or core is None:
            continue
        points.append((_tiles(core), block.area_mm2))
    if len(points) < 2:
        return None
    points.sort()
    count = len(points)
    mean_t = sum(t for t, _ in points) / count
    mean_a = sum(a for _, a in points) / count
    variance = sum((t - mean_t) ** 2 for t, _ in points)
    if variance == 0:
        return None
    slope = sum((t - mean_t) * (a - mean_a) for t, a in points) / variance
    return {"slope_mm2_per_tile": slope, "intercept_mm2": mean_a - slope * mean_t,
            "points": points}


#: The human framing of a mission, which is domain knowledge and not in
#: the catalogue. Anything else on the page is derived from the data.
USE_CASES: Dict[str, str] = {
    "edge_ai_device_multi_stream_tracking__anomaly": """
<p>A fanless, mains- or PoE-powered appliance watching four camera streams: a loading bay, a
retail floor, a perimeter. It detects, tracks and flags anomalies on every frame, unattended,
for years.</p>""",
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
    "drone_interceptor_terminal_engagement": """
<p>A small interceptor flying the terminal phase of an engagement: closing at 150-250 m/s on a
manoeuvring target, with the whole sense-decide-act loop onboard. No VLM, no operator, no second
look. At that closure the {deadline:g} ms deadline is {metres_lo:.0f}-{metres_hi:.0f} m of
travel, so a late frame is not a dropped frame, it is a miss.</p>
<p>Everything else follows from the closure rate: stereo and lidar at high rate because the
scene changes fast, detection and the whole reactive stack at {det_hz:g} Hz, and
{budget:g} W to do it in on an airframe that also has to fly.</p>""",
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
        pressure.append(("power", dossier.datapath_total_w / dossier.power_budget_w))
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
             compare=None) -> dict:
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
bytes. Every compute and bandwidth figure below therefore understates the mission.</p>"""
        if unpriced else "")

    out["configuration"] = f"""
<p>One KPU fabric, a cluster of Andes RISC-V cores, a shared fabric and one LPDDR5 interface at
{dossier.node}. Engine counts are the <i>output</i> of section 6, not an input.</p>
<p class="note">The CPU block carries <b>baseline</b> figures: we have measured an Arm
Cortex-A78AE, not an Andes core. They stand in for the socket until you replace them, which is
what section 6 asks for. Nothing here is a claim about Andes silicon.</p>
<p>Three numbers per compute block, because any two mislead: <b>X</b>, the throughput one server
actually delivers; <b>U</b>, the share of wall clock it is busy; <b>E</b>, the share of its dense
peak that X represents. They compose exactly &mdash; <b>demand = X &times; servers &times;
U</b>.</p>"""

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
        one = area_fit["slope_mm2_per_tile"] + area_fit["intercept_mm2"]
        fit_note = f"""
<p><b>Area.</b> A least-squares line through our four catalogued N7 cores gives
{area_fit['slope_mm2_per_tile']:.4f} mm&sup2; per tile plus
{area_fit['intercept_mm2']:.3f} mm&sup2; fixed, so a one-tile fabric is about
{one:.2f} mm&sup2;. The CPU side we cannot price at all: we anchor the cluster's caches but not
its core logic, because no absolute core area is published for the baseline. <b>An Andes core
area would close that column outright.</b></p>"""

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
                    f"ESDF, so at these rates it is correct arithmetic over a distance field "
                    f"that is {esdf.calls_in_flight:.3g}x out of date.</p>")
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
<p class="note">Read the table as a necessary condition, not a sufficient one: a loop that
does not close here will not close on real silicon, but a loop that closes here has only passed
the easiest of the tests it has to pass.</p>"""


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
            f"<td class=\"num\">{ratio:.2g}x</td></tr>")
    ours_stage = {st.key: st for st in dossier.stages}
    theirs_stage = {st.key: st for st in other.stages}
    movers, added = [], []
    for key, st in ours_stage.items():
        was = theirs_stage.get(key)
        if was is None or not was.ops_per_s:
            added.append(key)
            continue
        ratio = st.ops_per_s / was.ops_per_s
        if ratio >= 2.0:
            movers.append((ratio, key, was.rate_hz, st.rate_hz))
    movers.sort(reverse=True)
    if not rows:
        return ""
    driver = ""
    if movers:
        top = ", ".join(f"<code>{html_escape(k)}</code> {r:.2g}x" for r, k, _a, _b in movers[:6])
        driver += f"<p>Stages whose arithmetic at least doubles: {top}.</p>"
    if added:
        # Rendered independently: a comparison can add stages without any
        # existing stage moving, and the guard used to hide that entirely.
        names = ", ".join(f"<code>{html_escape(a)}</code>" for a in sorted(added))
        driver += (f"<p>Stages present only in {html_escape(dossier.title)}: {names}.</p>")
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
        deltas.append((delta, label))
        rows.append(f"<tr><td>{html_escape(label)}</td>"
                    f"<td class=\"num\">{fmt.format(ours)}</td>"
                    f"<td class=\"num\">{fmt.format(theirs)}</td>"
                    f"<td class=\"num\">{delta:.1%}</td></tr>")
    if not rows:
        return ""
    worst_delta, worst_label = max(deltas)
    return f"""
<h2>Cross-check against the published annex</h2>
<p>This mission is one of two the companion annex publishes figures for. Ours are derived
independently from per-stage operation and byte counts; theirs come from
<i>BranesAI-Autonomy-Compute-Requirements</i> section 4.</p>
<div class="panel"><table><thead><tr><th>quantity</th><th>ours</th><th>published</th>
<th>difference</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">{len(rows)} independent quantities, and the largest disagreement is
{worst_delta:.1%} ({html_escape(worst_label.lower())}). <b>This validates the demand model, not
the sizing.</b> The published oversubscription is against the annex's own pooled
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
    if supply and byte_top:
        parts.append(
            f"<p>Memory is worth spelling out. The interface carries "
            f"{dossier.dram_demand_gb_per_s:.0f} GB/s of compulsory traffic against "
            f"{dossier.dram_supply_gb_per_s:g} GB/s of peak, and "
            f"<code>{html_escape(byte_top.key)}</code> is "
            f"{byte_top.bytes_per_s / supply:.0%} of the interface on its own "
            f"&mdash; at {byte_top.rate_hz:g} Hz. That is a "
            f"{si(byte_top.bytes_per_call, 'B')} working set pulled through DRAM on every "
            f"call.</p>")
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
        parts.append(
            f"<p>The leverage is enormous because the baseline is so low. At 10% of peak "
            f"instead of {lead_eff:.2%}, that class falls from {_amount(lead_cores)} cores to "
            f"{_amount(ten)}, and the whole CPU requirement from {_amount(cpu.servers_needed)} to "
            f"{_amount(cpu.servers_needed - lead_cores + ten)} &mdash; "
            f"{provision(cpu.servers_needed - lead_cores + ten, target_utilization):g} cores "
            f"provisioned instead of {cpu.servers_provisioned}.</p>")
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
            if catalogued_tiles:
                parts.append(
                    f"<p>It is equally an argument for <i>one</i> tile: the catalogued part has "
                    f"{catalogued_tiles} tiles, "
                    f"{catalogued_tiles / kpu.servers_needed:.0f}x more fabric than this "
                    f"mission can use.</p>")
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
    first = dossier.stages[0] if dossier.stages else None
    if first is not None:
        rows.append(("Sensor ingest", si(first.bytes_per_s, "B/s"),
                     "mission profile sensors, first pipeline stage",
                     _status(True, f"{_cams(dossier)}")))
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
    if dossier.chain_seconds is not None:
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
    rows.append(("Power budget", f"{dossier.power_budget_w:g} W",
                 "mission profile power_budget_w",
                 _status(dossier.power_budget_fraction_used is not None
                         and dossier.power_budget_fraction_used <= 1.0,
                         f"datapath floor {dossier.datapath_total_w * 1e3:.0f} mW = "
                         f"{(dossier.power_budget_fraction_used or 0):.1%}")))
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
        fit = tile_area_fit()
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
    first = dossier.stages[0] if dossier.stages else None
    ingress = ("sensors", first.bytes_per_s) if first else None
    write_report(render(dossier, sections(dossier, soc, alt, fit, _tiles_n, args.efficiency,
                                 args.target_utilization, args.output or "",
                                 crosscheck, compare),
                        requirements_rows(dossier, alt), alternatives, idle,
                        date.today().isoformat(), ingress, args.cpu_label,
                        {"cpu": args.cpu_baseline}), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
