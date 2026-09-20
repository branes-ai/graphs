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
from graphs.reporting.mission_dossier import ms, render, si  # noqa: E402
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
             output: str = "") -> dict:
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
        cpu_rows = f["by_engine"].get("cpu", [])
        kpu_rows = f["by_engine"].get("kpu", [])
        cpu_top, cpu_fit = _worst(cpu_rows)
        kpu_top, kpu_fit = _worst(kpu_rows)
        out["verdict"] = f"""
<div class="verdict warn">
<b>Nothing in this parts list serves this mission.</b> It needs
{cpu.servers_needed:.0f} CPU cores against the {cpu.servers_provisioned} the design has, and
{kpu.servers_needed:.0f} KPU tiles against {kpu.servers_provisioned} &mdash; and it asks
{dossier.dram_demand_gb_per_s:.0f} GB/s of a {dossier.dram_supply_gb_per_s:g} GB/s memory
interface, {dossier.dram_demand_gb_per_s / dossier.dram_supply_gb_per_s:.0%} of it.
{len(dossier.unplaced)} stages cannot be priced at all, so every one of those figures is a
floor.<br><br>
That is not one problem but three, and they have different owners:
<b>{html_escape(kpu_top.key)}</b> is {kpu_fit.servers_needed / kpu.servers_needed:.1%} of the
accelerator demand, <b>{html_escape(cpu_top.key)}</b> is
{cpu_fit.servers_needed / cpu.servers_needed:.0%} of the CPU demand, and
<b>{html_escape(f['bytes_top'].key)}</b> alone is
{f['bytes_top'].bytes_per_s / f['bytes_supply']:.0%} of the memory interface.
</div>"""

    blurb = USE_CASES.get(dossier.mission, f"<p>{html_escape(dossier.note)}</p>")
    out["use_case"] = blurb.format(budget=dossier.power_budget_w) + f"""
<p>{_cams(dossier)}. {len(dossier.stages)} stages over
{len({s.tier for s in dossier.stages})} tiers, {sum(1 for s in dossier.stages if s.on_reactive_chain)}
of them on the sense-to-act chain that the {dossier.deadline_ms:g} ms deadline applies to.</p>"""

    out["requirements_intro"] = """
<p>What the product must do, and where each figure comes from. Anything the catalogue does not
state is marked <b>NOT STATED</b> rather than assumed.</p>"""

    out["requirements_note"] = f"""
<p class="note"><b>Thermal and SWaP</b> are yours to set; they are not in our model.
<b>Full-SoC power</b> is a different kind of gap: {dossier.datapath_total_w * 1e3:.0f} mW is
arithmetic only &mdash; no memory traffic, clock tree, leakage or idle. On a mission moving
{dossier.dram_demand_gb_per_s:.0f} GB/s the memory term alone will dwarf it. <b>CPU core
area</b> is the one gap you can close, and section 7 says why it is open.</p>"""

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
    out["analysis_3"] = _the_ask(dossier, f, kpu, cpu, target_utilization)

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


def _times(rate_hz: float) -> str:
    """How often, in words that survive a rate of exactly one."""
    if rate_hz == 1:
        return "once a second"
    if rate_hz < 1:
        return f"every {1 / rate_hz:.3g} seconds"
    return f"on each of its {si(rate_hz, '')} calls a second"


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
            f"<b>{prov.servers_needed:.3g} {prov.unit}s</b> between them, against the "
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


def _the_ask(dossier, f, kpu, cpu, target_utilization: float) -> str:
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
            f"<p>The CPU needs <b>{cpu.servers_needed:.3g} cores</b> and the design has "
            f"{cpu.servers_provisioned}. The gap is not spread evenly: three stages are "
            f"{covered:.0%} of it.</p><ul>")
        for stage, fit, eff in top:
            parts.append(
                f"<li><code>{html_escape(stage.key)}</code> "
                f"(<code>{html_escape(stage.kernel_class)}</code>): "
                f"<b>{fit.servers_needed:.3g} cores</b> at <b>{eff:.2%}</b> of FP32 peak.</li>")
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
            f"<b>{lead_cores:.3g} of the {cpu.servers_needed:.3g} cores</b> "
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
            f"instead of {lead_eff:.2%}, that class falls from {lead_cores:.3g} cores to "
            f"{ten:.2g}, and the whole CPU requirement from {cpu.servers_needed:.3g} to "
            f"{cpu.servers_needed - lead_cores + ten:.3g} &mdash; "
            f"{provision(cpu.servers_needed - lead_cores + ten, target_utilization):g} cores "
            f"provisioned instead of {cpu.servers_provisioned}.</p>")
    if kpu and kpu.servers_needed:
        rows = f["by_engine"].get("kpu", [])
        rows.sort(key=lambda r: -(r[1].servers_needed or 0))
        top, top_fit = rows[0]
        eff = min((c.efficiency for c in top_fit.classes if c.efficiency), default=0)
        share = top_fit.servers_needed / kpu.servers_needed
        parts.append("<h3>5. The accelerator ask is a memory ask</h3>")
        parts.append(
            f"<p><code>{html_escape(top.key)}</code> is {share:.1%} of the "
            f"{kpu.servers_needed:.0f}-tile requirement, and it is not because it is the "
            f"biggest arithmetic: it is {top.ops_per_s / f['total_ops']:.0%} of the operations. "
            f"It is because its domain-flow ceiling is <b>{eff:.2%}</b>. The fabric is not "
            f"computing, it is waiting: {si(top.bytes_per_call, 'B')} of weights crossing DRAM "
            f"{_times(top.rate_hz)}.</p>")
        others = sum(r[1].servers_needed for r in rows[1:])
        parts.append(
            f"<p>Take it out and the rest of the accelerator work &mdash; detection, the SDF "
            f"encoder, the policy &mdash; needs <b>{others:.2g} tiles</b>. That is the shape of "
            f"the decision: this is not a mission that needs a {kpu.servers_needed:.0f}-tile "
            f"fabric, it is a mission with a weight-streaming problem bolted to a "
            f"{others:.2g}-tile one. Quantisation, weight caching in on-die SRAM, a smaller "
            f"model or a wider memory interface are all attacks on the same number; more tiles "
            f"is not.</p>")
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
    """(requirement, figure, source, status). "gap" means the catalogue does
    not state it -- which is not the same as a requirement that is met."""
    rows = []
    by_kind = {p.kind: p for p in dossier.provisions}
    for stage in dossier.stages:
        place = next((p for p in dossier.placements if p.stage == stage.key), None)
        if place is None or stage.key not in ("det", "mono"):
            continue
        prov = by_kind.get("kpu" if place.engine == "kpu" else "cpu")
        if prov is None:
            continue
        ok = prov.utilization <= 1.0
        label = ("Detection throughput" if stage.key == "det" else "Camera ingest")
        figure = (f"{stage.rate_hz:g} inferences/s" if stage.key == "det"
                  else si(stage.rate_hz, "px/s"))
        rows.append((label, figure, f"mission profile rates_hz.{stage.key}",
                     _status(ok, f"{prov.servers_provisioned} {prov.unit}"
                                 f"{'s' if prov.servers_provisioned != 1 else ''} at "
                                 f"{prov.utilization:.0%}")))
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
                 _status(dossier.power_budget_fraction_used <= 1.0,
                         f"datapath floor {dossier.datapath_total_w * 1e3:.0f} mW = "
                         f"{dossier.power_budget_fraction_used:.1%}")))
    rows.append(("Full-SoC power", "-",
                 "no memory, clock-tree, leakage or idle term in this model", "gap"))
    if dossier.dram_supply_gb_per_s:
        used = dossier.dram_demand_gb_per_s / dossier.dram_supply_gb_per_s
        note = (f"{used:.1%} of {dossier.dram_supply_gb_per_s:g} GB/s peak"
                + (" -- and a floor, with unpriced stages left out"
                   if dossier.unplaced else ""))
        rows.append(("Memory bandwidth", f"{dossier.dram_demand_gb_per_s:.2f} GB/s demand",
                     "sum of per-stage byte counts", _status(used <= 1.0, note)))
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
                                 args.target_utilization, args.output or ""),
                        requirements_rows(dossier, alt), alternatives, idle,
                        date.today().isoformat(), ingress, args.cpu_label,
                        {"cpu": args.cpu_baseline}), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
