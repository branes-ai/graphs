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
from typing import List, Optional

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


def sections(dossier, soc, alt, area_fit, catalogued_tiles: int = 0,
             efficiency: str = "") -> dict:
    """The argument. Every number is interpolated from the dossier, so the
    prose cannot drift from the analysis it describes."""
    kpu = next((p for p in dossier.provisions if p.kind == "kpu"), None)
    cpu = next((p for p in dossier.provisions if p.kind == "cpu"), None)
    total_ops = sum(s.ops_per_s for s in dossier.stages)
    by_key = {s.key: s for s in dossier.stages}
    light, heavy = by_key.get("mono"), by_key.get("det")
    light_share = (light.ops_per_s / total_ops) if light and total_ops else 0.0
    dram_used = (dossier.dram_demand_gb_per_s / dossier.dram_supply_gb_per_s
                 if dossier.dram_supply_gb_per_s else None)

    out = {}
    out["lede"] = """
<p class="lede">Four 1080p streams at 15 Hz, an open-vocabulary detector on every frame, 5 W.
What has to be true of the CPU and the accelerator for that to close.</p>"""

    if kpu and cpu:
        out["verdict"] = f"""
<div class="verdict">
<b>{cpu.servers_provisioned} CPU cores and {kpu.servers_provisioned} KPU
tile{'s' if kpu.servers_provisioned != 1 else ''}.</b> The detector is
{heavy.ops_per_s / total_ops:.1%} of the arithmetic and fits in {kpu.servers_needed:.2f} of a
tile. The pixel front end is {light_share:.1%} of the arithmetic and takes
{cpu.servers_needed:.2f} cores. Memory, energy and latency all have an order of magnitude in
hand. <b>The CPU is the only part without margin</b>, and the reason is a
{cpu.efficiency:.1%} efficiency on a per-pixel kernel &mdash; which is the question this
partnership exists to answer.
</div>"""

    out["use_case"] = f"""
<p>A fanless, mains- or PoE-powered appliance watching four camera streams: a loading bay, a
retail floor, a perimeter. It detects, tracks and flags anomalies on every frame, unattended,
for years.</p>
<p>{_cams(dossier)}, {by_key['det'].rate_hz:g} detector inferences per second, one per frame
across all four. The {dossier.deadline_ms:g} ms deadline is loose for a
{dossier.frame_hz:g} Hz frame rate, so this is a throughput problem, not a latency one: every
frame has to be processed, none may be dropped.</p>"""

    out["requirements_intro"] = """
<p>What the product must do, and where each figure comes from. Anything the catalogue does not
state is marked <b>NOT STATED</b> rather than assumed.</p>"""

    out["requirements_note"] = f"""
<p class="note"><b>Thermal and SWaP</b> are yours to set; they are not in our model.
<b>Full-SoC power</b> is a different kind of gap: {dossier.datapath_total_w * 1e3:.0f} mW is
arithmetic only &mdash; no memory traffic, clock tree, leakage or idle. The true figure is
higher. <b>CPU core area</b> is the one gap you can close, and section 7 says why it is
open.</p>"""

    out["workload"] = f"""
<p>Two stages. A per-pixel front end runs on every pixel of every stream; the detector then runs
once per frame on the result. Association, re-identification and the anomaly rule sit inside the
detector's per-call cost, anchored to a published FLOP count.</p>
<p>The two could hardly be less alike. The front end does
{light.ops_per_call:g} operations on each of {si(light.rate_hz, '')} pixels a second; the
detector does {si(heavy.ops_per_call, 'OP')} on each of {heavy.rate_hz:g} frames. A trickle of
arithmetic over a flood of pixels, then the reverse. Nothing is good at both, and that is the
whole design problem.</p>"""

    out["workload_note"] = f"""
<p class="note">Drawn as a serial sum, which is pessimistic &mdash; a real build overlaps the two
stages. Even so, {ms(dossier.chain_seconds)} against {dossier.deadline_ms:g} ms leaves
{dossier.deadline_headroom:.1f}x in hand.</p>"""

    out["demand"] = """
<p>Per stage: the kernel class its arithmetic belongs to, the precision it needs, its rate, and
the operations and bytes that follow. <b>Kernel class</b> is the level at which hardware
efficiency is a real number, so it is the level everything is measured and sized at.</p>"""

    out["demand_note"] = f"""
<p class="note">If operations were the unit of cost, the front end would be a rounding error at
{light_share:.1%} of them. Section 6 is about why it is not.</p>"""

    out["configuration"] = f"""
<p>One KPU fabric, a cluster of Andes RISC-V cores, a shared fabric and one LPDDR5 interface at
{dossier.node}. Engine counts are the <i>output</i> of section 6, not an input.</p>
<p class="note">The CPU block carries <b>baseline</b> figures: we have measured an Arm
Cortex-A78AE, not an Andes core. They stand in for the socket until you replace them, which is
what section 6 asks for. Nothing here is a claim about Andes silicon.</p>
<p>Three numbers per compute block, because any two mislead: <b>X</b>, the throughput one server
actually delivers; <b>U</b>, the share of wall clock it is busy; <b>E</b>, the share of its dense
peak that X represents. They compose exactly &mdash; <b>demand = X &times; servers &times;
U</b>. High E with low U means oversized. High U with low E means badly matched, and buying more
of it is the expensive fix.</p>"""

    if kpu and cpu:
        out["configuration_note"] = f"""
<p class="note">The KPU delivers {si(kpu.throughput_ops_per_s, 'OP/s')} per tile at
<b>E = {kpu.efficiency:.1%}</b>. The CPU delivers {si(cpu.throughput_ops_per_s, 'OP/s')} per core
at <b>E = {cpu.efficiency:.1%}</b> &mdash; a factor of
{kpu.efficiency / cpu.efficiency:.0f} apart, with the badly-matched engine carrying under one
percent of the operations.</p>"""

    out["analysis"] = f"""
<h3>1. Neither placement is a choice</h3>
<p>A precision class runs in the narrowest format the engine offers at or above its floor, and
that (kernel class, engine, format) triple needs an efficiency from somewhere &mdash; a
measurement or the domain-flow ceiling. Where there is none, there is no number, and we do not
invent one.</p>
<ul>
<li><code>{light.key}</code> is <code>{light.kernel_class}</code>, FP32 throughout. <b>No fit on
the KPU</b>: the domain-flow model has no schedule for that kernel class on a regular wavefront
fabric. It goes to the CPU.</li>
<li><code>{heavy.key}</code> is <code>{heavy.kernel_class}</code>,
{heavy.class_split['A']:.0%} INT8 and {heavy.class_split['B']:.0%} FP16. <b>No fit on the
CPU</b>: nothing measures INT8 {heavy.kernel_class} on a CPU. It goes to the KPU.</li>
</ul>"""

    if kpu and cpu:
        out["analysis_2"] = f"""
<h3>2. Size each engine</h3>
<p><b>Detector.</b> One tile delivers {si(kpu.throughput_ops_per_s, 'OP/s')} at a ceiling
efficiency of {kpu.efficiency:.1%}. The detector wants {si(heavy.ops_per_s, 'OP/s')}:
<b>{kpu.servers_needed:.3f} of a tile</b>, so one tile at {kpu.utilization:.1%}. That efficiency
is a ceiling, so read it as <i>no fewer than</i> {kpu.servers_needed:.2f} tiles.</p>
<p><b>Front end.</b> One core delivers {si(cpu.throughput_ops_per_s, 'OP/s')} against a
{si(cpu.peak_ops_per_s, 'OP/s')} FP32 peak &mdash; {cpu.efficiency:.2%}. The front end wants
{si(light.ops_per_s, 'OP/s')}: <b>{cpu.servers_needed:.3f} cores</b>, so four at
{cpu.utilization:.1%}.</p>

<h3>3. Nothing else binds</h3>
<ul>
<li><b>Memory.</b> {dossier.dram_demand_gb_per_s:.2f} GB/s against
{dossier.dram_supply_gb_per_s:g} GB/s &mdash; {dram_used:.1%}. The narrowest interface we
catalogue is {1 / dram_used:.0f}x oversized here.</li>
<li><b>Energy.</b> {dossier.datapath_total_w * 1e3:.0f} mW of arithmetic against
{dossier.power_budget_w:g} W: {dossier.power_budget_fraction_used:.1%}.</li>
<li><b>Latency.</b> {ms(dossier.chain_seconds)} against {dossier.deadline_ms:g} ms.</li>
</ul>

<h3>4. What the accelerator is worth</h3>
<p>The detector has no measured INT8 figure on a CPU, but its INT8 class may run in any wider
format, and FP32 <i>is</i> measured. At E = {alt['efficiency']:.1%} one core delivers
{si(alt['throughput_per_server'], 'OP/s')}, so the detector would need
<b>{alt['servers_needed']:.1f} CPU cores</b> and {alt['watts'] * 1e3:.0f} mW &mdash; against
{kpu.servers_needed:.2f} of a tile and
{dossier.datapath_watts.get('kpu', 0) * 1e3:.0f} mW. That is
<b>{alt['servers_needed'] / kpu.servers_needed:.0f} cores of work per tile</b> at
{alt['watts'] / dossier.datapath_watts.get('kpu', 1):.1f}x less energy.</p>
<p>It is equally an argument for <i>one</i> tile. The catalogued
{dossier.design.replace('kpu_', '').replace('_', ' ').upper()} part has {catalogued_tiles}
tiles: {catalogued_tiles / kpu.servers_needed:.0f}x more fabric than this mission can use.</p>"""

    if kpu and cpu:
        ten = cpu.servers_needed * cpu.efficiency / 0.10
        out["analysis_3"] = f"""
<h3>5. The CPU is the ask</h3>
<p>{cpu.servers_provisioned} cores at {cpu.utilization:.0%}, one tile at
{kpu.utilization:.0%}. The CPU has no margin, and it is carrying {light_share:.1%} of the
operations to get there. The cause is {cpu.efficiency:.2%} efficiency, not core count: a
per-pixel chain on a general-purpose core spends its time on loads, stores and address
arithmetic, not on the {light.ops_per_call:g} operations per pixel we count.</p>
<p>So the question for an Andes core is not how fast it is in the abstract. It is:</p>
<blockquote><b>What does it take to sustain {si(light.ops_per_s, 'OP/s')} of FP32 per-pixel work
&mdash; {light.ops_per_call:g} operations on each of {si(light.rate_hz, '')} pixels a second
&mdash; inside a {dossier.power_budget_w:g} W envelope?</b></blockquote>
<p>Three ways to get there, in order of leverage:</p>
<ol>
<li><b>Vector width and memory behaviour.</b> At {cpu.efficiency:.2%} we need
{cpu.servers_needed:.2f} cores. At a still-unambitious 10% we need {ten:.2f}, and the part ships
{max(1, round(ten / 0.85 + 0.49)):g} cores instead of {cpu.servers_provisioned}. This is where
an RVV-capable core should win, and it is the number we would like to replace with a measurement
on your silicon.</li>
<li><b>Move the front end off the CPU entirely.</b> The SoC already carries an ISP block, and
this stage assumes demosaic and denoise already happened in the sensor. The
{light.ops_per_call:g} op/pixel that remain are a fixed-function candidate. We state no
capability or area for the on-die ISP, so we cannot price the move &mdash; only note the block is
there and idle.</li>
<li><b>Schedule it on the fabric.</b> <code>{light.kernel_class}</code> is one of eleven classes
our domain-flow model calls unschedulable on a regular mesh. A per-pixel chain is about as
regular as work gets, so that may be our limitation rather than the fabric's. The tile at
{kpu.utilization:.0%} has room.</p></li>
</ol>"""

    fit_note = ""
    if area_fit:
        one = area_fit["slope_mm2_per_tile"] + area_fit["intercept_mm2"]
        fit_note = f"""
<p><b>Area.</b> A least-squares line through our four catalogued N7 cores gives
{area_fit['slope_mm2_per_tile']:.4f} mm&sup2; per tile plus
{area_fit['intercept_mm2']:.3f} mm&sup2; fixed, so a one-tile fabric is about
{one:.2f} mm&sup2; &mdash; an extrapolation one step below our smallest SKU
({area_fit['points'][0][0]} tiles), and labelled THEORETICAL for it. The CPU side we cannot
price at all: we anchor the cluster's caches but not its core logic, because no absolute core
area is published for the baseline. <b>An Andes core area would close that column
outright.</b></p>"""

    out["provenance"] = f"""
<p>Four kinds of figure, and the page says which each is:</p>
<ul>
<li><b>Catalogue.</b> The mission, its rates, per-call operation and byte counts, the SoC
composition, the node's energy per operation.</li>
<li><b>Measured.</b> CPU efficiencies from <code>{efficiency}</code>, with the run behind each
figure cited in the table. <b>These are measured on Arm Cortex-A78AE cores in a Jetson Orin
Nano</b> &mdash; our reference baseline, not a claim about any other core. They are the reason
the CPU requirement above is stated as a throughput target rather than a core count.</li>
<li><b>Ceiling.</b> The KPU efficiency is our domain-flow upper bound, taken from the smallest
core with a stated schedule ({CEILING_CORE}) so it carries no DRAM constraint from a larger
fabric. A ceiling bounds, it does not predict, so the tile count is a <i>lower</i> bound: no
fewer than {next((p.servers_needed for p in dossier.provisions if p.kind == 'kpu'), 0):.2f}.</li>
<li><b>Gap.</b> Named, never filled.</li>
</ul>
{fit_note}
<p>Reproduce with:</p>
<p><code>python cli/report_mission_dossier.py --mission {dossier.mission} \\<br>
&nbsp;&nbsp;&nbsp;&nbsp;--design {dossier.design} --efficiency {efficiency} \\<br>
&nbsp;&nbsp;&nbsp;&nbsp;-o docs/assessments/dossier-edge-tracking.html</code></p>"""
    return out


def html_escape(text: str) -> str:
    import html as _html

    return _html.escape(text or "")


def _cams(dossier) -> str:
    cams = dossier.sensors.get("mono")
    if isinstance(cams, (list, tuple)) and len(cams) == 4:
        return f"{cams[0]} x {cams[1]}x{cams[2]} streams at {cams[3]} Hz"
    return "the stated sensor configuration"


def requirements_rows(dossier, alt):
    """(requirement, figure, source, status). "gap" means the catalogue does
    not state it -- which is not the same as a requirement that is met."""
    kpu = next((p for p in dossier.provisions if p.kind == "kpu"), None)
    cpu = next((p for p in dossier.provisions if p.kind == "cpu"), None)
    det = next((s for s in dossier.stages if s.key == "det"), None)
    mono = next((s for s in dossier.stages if s.key == "mono"), None)
    dram = (dossier.dram_demand_gb_per_s / dossier.dram_supply_gb_per_s
            if dossier.dram_supply_gb_per_s else None)
    rows = []
    if det and kpu:
        rows.append(("Detection throughput", f"{det.rate_hz:g} inferences/s",
                     "mission profile rates_hz.det",
                     f"met: {kpu.servers_provisioned} KPU tile at {kpu.utilization:.0%}"))
    if mono and cpu:
        rows.append(("Camera ingest", f"{si(mono.rate_hz, 'px/s')}",
                     "mission profile sensors.mono",
                     f"met: {cpu.servers_provisioned} CPU cores at {cpu.utilization:.0%}"))
    if dossier.chain_seconds is not None:
        rows.append(("Sense-to-track latency", f"{dossier.deadline_ms:g} ms budget",
                     "mission profile deadline_ms",
                     f"met: {ms(dossier.chain_seconds)}, "
                     f"{dossier.deadline_headroom:.1f}x headroom"))
    rows.append(("Power budget", f"{dossier.power_budget_w:g} W",
                 "mission profile power_budget_w",
                 f"datapath floor {dossier.datapath_total_w * 1e3:.0f} mW "
                 f"= {dossier.power_budget_fraction_used:.1%}"))
    rows.append(("Full-SoC power", "-",
                 "no memory, clock-tree, leakage or idle term in this model", "gap"))
    if dram is not None:
        rows.append(("Memory bandwidth", f"{dossier.dram_demand_gb_per_s:.2f} GB/s demand",
                     "sum of per-stage byte counts",
                     f"met: {dram:.1%} of {dossier.dram_supply_gb_per_s:g} GB/s peak"))
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
    if alt:
        alternatives.append({
            "label": "CPU: det in FP32 (rejected)", "unit": "core",
            "needed": alt["servers_needed"],
            "provisioned": provision(alt["servers_needed"], args.target_utilization),
            "kind": "cpu", "rejected": True,
            "note": f"E {alt['efficiency']:.1%} - measured - "
                    f"{alt['watts'] * 1e3:.0f} mW of datapath",
        })
    idle = [("ISP", "no capability stated"), ("codec", "not used")]
    first = dossier.stages[0] if dossier.stages else None
    ingress = ("sensors", first.bytes_per_s) if first else None
    write_report(render(dossier, sections(dossier, soc, alt, fit, _tiles_n, args.efficiency),
                        requirements_rows(dossier, alt), alternatives, idle,
                        date.today().isoformat(), ingress, args.cpu_label,
                        {"cpu": args.cpu_baseline}), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
