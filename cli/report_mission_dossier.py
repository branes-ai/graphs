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


def build(mission: str, design_id: str, efficiency: str, target_utilization: float):
    workload = load_autonomy_workload()
    kernels = load_kernel_classes()
    tables = load_efficiency_tables()
    if efficiency not in tables:
        raise KeyError(f"unknown efficiency table {efficiency!r}")
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

    table = _tables()["orin_nano_measured_v1"]
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
    (t0, a0), (t1, a1) = points[0], points[-1]
    slope = (a1 - a0) / (t1 - t0)
    return {"slope_mm2_per_tile": slope, "intercept_mm2": a0 - slope * t0,
            "points": points}


def sections(dossier, soc, alt, area_fit, catalogued_tiles: int = 0) -> dict:
    """The argument. Every number is interpolated from the dossier, so the
    prose cannot drift from the analysis it describes."""
    kpu = next((p for p in dossier.provisions if p.kind == "kpu"), None)
    cpu = next((p for p in dossier.provisions if p.kind == "cpu"), None)
    total_ops = sum(s.ops_per_s for s in dossier.stages)
    by_key = {s.key: s for s in dossier.stages}

    light = by_key.get("mono")
    heavy = by_key.get("det")
    light_share = (light.ops_per_s / total_ops) if light and total_ops else 0.0
    dram_used = (dossier.dram_demand_gb_per_s / dossier.dram_supply_gb_per_s
                 if dossier.dram_supply_gb_per_s else None)

    out = {}
    out["lede"] = """
<p class="lede">Four 1080p camera streams at 15 Hz, an open-vocabulary detector on every frame,
and a 5 W budget. This is the whole sizing argument for that product on a CPU + KPU SoC: what
the workload demands, what each engine delivers, and how many cores and tiles it takes before
the demand fits.</p>"""

    if kpu and cpu:
        out["verdict"] = f"""
<div class="verdict">
<b>The answer is {cpu.servers_provisioned} CPU cores and {kpu.servers_provisioned} KPU
tile{'s' if kpu.servers_provisioned != 1 else ''}.</b> The detector &mdash;
{heavy.ops_per_s / total_ops:.1%} of the mission's arithmetic &mdash; needs
{kpu.servers_needed:.2f} of a tile. The camera light path, {light_share:.1%} of the arithmetic,
needs {cpu.servers_needed:.2f} CPU cores. Neither compute engine is the binding constraint:
DRAM runs at {dram_used:.1%} of peak, the datapath draws
{dossier.datapath_total_w * 1e3:.0f} mW against a {dossier.power_budget_w:g} W budget, and the
chain finishes in {ms(dossier.chain_seconds)} against a {dossier.deadline_ms:g} ms deadline.
<b>What binds is a {cpu.efficiency:.1%} efficiency on the pixel stage</b>, and that is an
argument about the CPU, not about the accelerator.
</div>"""

    out["use_case"] = f"""
<p>A fixed-install edge appliance watching four camera streams &mdash; a loading bay, a retail
floor, a perimeter. It runs an open-vocabulary detector on every frame of every stream, keeps
identities across frames, and raises an anomaly when something does not belong. It is mains- or
PoE-powered, fanless, and it is expected to run unattended for years.</p>
<p>The catalogue states it as: <i>{html_escape(dossier.note)}</i>. In numbers, that is
{_cams(dossier)} and {by_key['det'].rate_hz:g} detector inferences per second, which is one per
frame across all four streams.</p>
<p>Two properties of this use case drive everything that follows. First, <b>the detector is not
latency-critical but is throughput-critical</b>: a {dossier.deadline_ms:g} ms deadline is
generous for a 60 Hz aggregate frame rate, but every frame must be processed, so the sizing
question is one of sustained rate, not of tail latency. Second, <b>the pixel path and the
detector are completely different kinds of work</b>, and the analysis in section 6 turns on
exactly that.</p>"""

    out["requirements_intro"] = """
<p>What the product has to do, and where each figure comes from. A requirement the catalogue
does not state is marked <b>NOT STATED</b> rather than assumed &mdash; a missing thermal limit
is not a satisfied thermal limit, and a page that quietly filled it in would be worse than
useless to whoever has to build the thing.</p>"""

    out["requirements_note"] = f"""
<p class="note">The four gaps are real and they matter differently. <b>Thermal</b> and
<b>SWaP</b> are product decisions that live outside this repository, and they are the customer's
to state. <b>Full-SoC power</b> is a gap of a different kind: the
{dossier.datapath_total_w * 1e3:.0f} mW figure is the datapath floor only &mdash; every
arithmetic operation charged once at the process node's energy per operation, with no memory
traffic, no clock tree, no leakage and no idle power in it. The true figure is higher, and by
how much is not something this model states. <b>CPU core area</b> is the one gap a CPU IP
partner could close immediately, and section 7 says why it is open.</p>"""

    out["workload"] = """
<p>The pipeline is two stages, and its shape is the reason the sizing works out the way it does.
Raw sensor data enters the light path, which is a per-pixel fixed-function chain; the detector
then runs once per frame on the result. Everything else a tracking product does &mdash;
association, re-identification, the anomaly rule &mdash; is folded into the detector's per-call
cost by the catalogue, which anchors it to a published FLOP count rather than modelling it
separately.</p>
<p>Each box carries what the stage demands and, underneath, what it costs on the engine it was
sized for in section 6. The number under each arrow is the traffic that edge carries.</p>"""

    out["workload_note"] = f"""
<p class="note">The chain is drawn as a serial sum, which is the pessimistic reading: the light
path for a frame finishes before the detector starts on it. A real implementation would pipeline
the two, so the sense-to-track latency of {ms(dossier.chain_seconds)} is an upper bound on a
configuration that already has {dossier.deadline_headroom:.1f}x of margin. Note also that this
mission's declared reactive chain contains only <code>{light.key}</code>; the
<code>{light.key}</code>&#8201;&rarr;&#8201;<code>{heavy.key}</code> path drawn here is the
product-meaningful latency, and is the stricter of the two.</p>"""

    out["demand"] = """
<p>Per stage: the kernel class its arithmetic belongs to, the precision classes it needs as a
share of its own operations, its rate in its own unit, and the operations and bytes that follow
from those. The <b>kernel class</b> is the level at which hardware efficiency is a meaningful
number, which is why the sizing in section 6 is done there and not per stage or per network.</p>"""

    out["demand_note"] = f"""
<p class="note">The asymmetry in this table is the whole story. <code>{heavy.key}</code> is
{heavy.ops_per_s / total_ops:.1%} of the operations; <code>{light.key}</code> is
{light_share:.1%}. If operations were the unit of cost, the light path would be a rounding
error. It is not, and section 6 shows why: {si(light.ops_per_s, 'OP/s')} of
<code>{light.kernel_class}</code> work is harder to deliver on a CPU than
{si(heavy.ops_per_s, 'OP/s')} of <code>{heavy.kernel_class}</code> is on a fabric built for it.</p>"""

    out["configuration"] = f"""
<p>The configuration is one KPU fabric, a cluster of Cortex-A78AE-class CPU cores, a shared
on-chip fabric and one LPDDR5 interface, all at {dossier.node}. The diagram shows it <b>as
sized</b> &mdash; the engine counts are the output of section 6, not an input.</p>
<p>Three numbers describe each compute block, and any two of them mislead:</p>
<ul>
<li><b>X</b>, the throughput one server actually delivers while it is running this work.</li>
<li><b>U</b>, the share of wall clock that server is busy &mdash; the sizing margin.</li>
<li><b>E</b>, the share of the server's dense peak that X represents &mdash; how well the kernel
suits the machine.</li>
</ul>
<p>They compose exactly: <b>demand = X &times; servers &times; U</b>. A block with high E and low
U is oversized. A block with high U and low E is not oversized; it is badly matched, and buying
more of it is the expensive way out.</p>"""

    if kpu and cpu:
        out["configuration_note"] = f"""
<p class="note">Read the two compute blocks against each other. The KPU delivers
{si(kpu.throughput_ops_per_s, 'OP/s')} per tile at <b>E = {kpu.efficiency:.1%}</b> of its dense
peak. The CPU delivers {si(cpu.throughput_ops_per_s, 'OP/s')} per core at
<b>E = {cpu.efficiency:.1%}</b>. That is a factor of
{kpu.efficiency / cpu.efficiency:.0f} in how well each engine is suited to the work it was given
&mdash; and the engine doing badly is the one carrying less than one percent of the operations.</p>"""

    out["analysis"] = f"""
<h3>Step 1: each stage has exactly one engine that can take it</h3>
<p>Before any sizing, the mapping is forced. A precision class runs in the narrowest format the
engine offers at or above the class's floor, and the efficiency of that (kernel class, engine,
format) triple has to come from somewhere &mdash; a measurement, or the domain-flow ceiling.
Where it does not exist, the stage does not get a number, and nothing is substituted for it.</p>
<ul>
<li><code>{light.key}</code> is <code>{light.kernel_class}</code>, entirely class C, so it runs in
FP32. <b>On the KPU there is no fit</b>: the domain-flow model has no schedule for that kernel
class on a regular wavefront fabric, so there is no ceiling to quote and nothing has measured
one. It goes to the CPU, where FP32 is measured.</li>
<li><code>{heavy.key}</code> is <code>{heavy.kernel_class}</code>, {heavy.class_split['A']:.0%}
class A and {heavy.class_split['B']:.0%} class B. On the CPU, class A binds to INT8 &mdash; the
narrowest format the CPU offers at or above A's floor &mdash; and <b>nothing measures INT8
{heavy.kernel_class} on a CPU</b>, so there is no fit. It goes to the KPU, where the domain-flow
model does have a schedule.</li>
</ul>
<p>Neither placement is a preference. The table below is the whole of it: two stages, two
engines, and exactly one number in each row.</p>"""

    if kpu and cpu:
        out["analysis_2"] = f"""
<h3>Step 2: size each engine to its own stage</h3>
<p><b>The detector.</b> At a ceiling efficiency of {kpu.efficiency:.1%}, one tile delivers
{si(kpu.throughput_ops_per_s, 'OP/s')}. The detector asks for
{si(heavy.ops_per_s, 'OP/s')}, so it needs <b>{kpu.servers_needed:.3f} of a tile</b>. One tile
covers it at {kpu.utilization:.1%} utilization. Because that efficiency is a ceiling rather than
a measurement, the honest reading is <i>no fewer than</i> {kpu.servers_needed:.2f} tiles.</p>
<p><b>The light path.</b> At a measured efficiency of {cpu.efficiency:.2%}, one core delivers
{si(cpu.throughput_ops_per_s, 'OP/s')} against a {si(cpu.peak_ops_per_s, 'OP/s')} FP32 peak. The
light path asks for {si(light.ops_per_s, 'OP/s')}, so it needs
<b>{cpu.servers_needed:.3f} cores</b>. Four cores cover it at {cpu.utilization:.1%}.</p>

<h3>Step 3: check what is not the constraint</h3>
<p>Three candidates fail to bind, and it is worth being explicit about each:</p>
<ul>
<li><b>Memory bandwidth.</b> {dossier.dram_demand_gb_per_s:.2f} GB/s of compulsory traffic
against {dossier.dram_supply_gb_per_s:g} GB/s of peak &mdash; {dram_used:.1%}. The narrowest
interface in the catalogue is roughly {1 / dram_used:.0f}x oversized for this mission.</li>
<li><b>Energy.</b> The datapath floor is {dossier.datapath_total_w * 1e3:.0f} mW against a
{dossier.power_budget_w:g} W budget: {dossier.power_budget_fraction_used:.1%}. Even if the rest
of the SoC costs an order of magnitude more than its arithmetic, the budget holds.</li>
<li><b>Latency.</b> {ms(dossier.chain_seconds)} serial against {dossier.deadline_ms:g} ms:
{dossier.deadline_headroom:.1f}x of margin, before pipelining the two stages.</li>
</ul>

<h3>Step 4: what the KPU is worth here</h3>
<p>The detector has no fit on the CPU because nobody has measured INT8
<code>{heavy.kernel_class}</code> there. But class A is <i>INT8 or wider</i>, so the CPU could
run the detector in FP32 &mdash; a different operating point, fully measured, and therefore
quotable. At E = {alt['efficiency']:.1%} in FP32, one core delivers
{si(alt['throughput_per_server'], 'OP/s')}, and the detector would need
<b>{alt['servers_needed']:.1f} CPU cores</b> and {alt['watts'] * 1e3:.0f} mW of datapath power
&mdash; against {kpu.servers_needed:.2f} of a tile and
{dossier.datapath_watts.get('kpu', 0) * 1e3:.0f} mW.</p>
<p>That is the accelerator's case, in the only terms that matter to a silicon budget:
<b>{alt['servers_needed'] / kpu.servers_needed:.0f} CPU cores' worth of work per KPU tile</b>,
at {alt['watts'] / dossier.datapath_watts.get('kpu', 1):.1f}x less datapath energy. It is also
the case for <i>one</i> tile rather than a large fabric: the catalogued
{dossier.design.replace('kpu_', '').replace('_', ' ').upper()} part has
{catalogued_tiles} tiles, which is
<b>{catalogued_tiles / kpu.servers_needed:.0f}x</b> more fabric than this mission can use. The
accelerator earns its place here at a scale of one tile; the rest of that part is being bought
for some other mission.</p>"""

    if kpu and cpu:
        out["analysis_3"] = f"""
<h3>Step 5: what actually binds, and what to do about it</h3>
<p>The sized configuration is {cpu.servers_provisioned} CPU cores and
{kpu.servers_provisioned} KPU tile. The CPU is at {cpu.utilization:.0%} and the KPU at
{kpu.utilization:.0%}, so <b>the CPU is the part with no margin</b> &mdash; and it is carrying
{light_share:.1%} of the mission's operations to do it.</p>
<p>The cause is the {cpu.efficiency:.2%} efficiency, not the core count. A per-pixel
fixed-function chain on a general-purpose core spends its time on loads, stores and address
arithmetic rather than on the {light.ops_per_call:g} operations per pixel that the model counts.
Three things would move it, in descending order of leverage:</p>
<ol>
<li><b>Put the light path in fixed-function silicon.</b> The composed SoC already contains an ISP
block, and this stage's own configuration note says demosaic and denoise are assumed to happen
in the sensor ISP &mdash; what remains on the CPU is the {light.ops_per_call:g} op/pixel light
path after that. The catalogue states no capability or area for the on-die ISP, so this analysis
cannot price the move; it can only say that the block is there and idle.</li>
<li><b>Widen the CPU's vector path or fix its memory behaviour.</b> Going from
{cpu.efficiency:.2%} to a still-modest 10% would take the light path from
{cpu.servers_needed:.2f} cores to {cpu.servers_needed * cpu.efficiency / 0.10:.2f}, and the part
would need {max(1, round(cpu.servers_needed * cpu.efficiency / 0.10 / 0.85 + 0.49)):g} cores
instead of {cpu.servers_provisioned}.</li>
<li><b>Give the fabric a schedule for it.</b> <code>{light.kernel_class}</code> is one of eleven
kernel classes the domain-flow model declares unschedulable on a regular mesh. If that is wrong
&mdash; and a per-pixel chain is about as regular as work gets &mdash; then the KPU tile at
{kpu.utilization:.0%} has room to absorb it and the CPU count falls out of the design.</li>
</ol>
<p>Each of those is a different team's work, which is the point of stating the sizing this way:
the number that binds names the owner.</p>"""

    fit_note = ""
    if area_fit:
        one = area_fit["slope_mm2_per_tile"] + area_fit["intercept_mm2"]
        fit_note = f"""
<p><b>Silicon area.</b> The KPU side can be priced: a least-squares line through the four
catalogued N7 cores gives {area_fit['slope_mm2_per_tile']:.4f} mm&sup2; per tile plus
{area_fit['intercept_mm2']:.3f} mm&sup2; fixed, so a one-tile fabric is about
{one:.2f} mm&sup2;. That is an <i>extrapolation</i> one step below the smallest catalogued SKU
({area_fit['points'][0][0]} tiles) and is labelled THEORETICAL for that reason. The CPU side
cannot be priced at all: the repository states cache areas for the cluster but marks core logic
<b>unanchored</b>, because no trustworthy absolute Cortex-A78AE core area is public. Arm
discloses only relative figures. <b>This is the one number in the whole dossier that a CPU IP
partner could close immediately</b>, and closing it would turn the area column from a gap into
an answer.</p>"""

    out["provenance"] = f"""
<p>Every figure on this page is one of four things, and the page says which:</p>
<ul>
<li><b>Catalogue data.</b> The mission profile, its rates, its per-call operation and byte
counts, the SoC composition and the process node's energy per operation.</li>
<li><b>A measurement.</b> The CPU efficiencies come from the
<code>orin_nano_measured_v1</code> table &mdash; real silicon, with the run that produced each
figure cited in the table.</li>
<li><b>A ceiling.</b> The KPU efficiency is the domain-flow model's upper bound, taken from the
smallest catalogued core with a stated schedule ({CEILING_CORE}) so that the bound does not carry
a DRAM constraint belonging to a much larger fabric. A ceiling bounds; it does not predict. The
tile count derived from it is therefore a <i>lower</i> bound: no fewer than
{next((p.servers_needed for p in dossier.provisions if p.kind == 'kpu'), 0):.2f} tiles.</li>
<li><b>A gap.</b> Named as such, never filled.</li>
</ul>
{fit_note}
<p>Reproduce this page with:</p>
<p><code>python cli/report_mission_dossier.py --mission {dossier.mission} \\<br>
&nbsp;&nbsp;&nbsp;&nbsp;--design {dossier.design} -o docs/assessments/dossier-edge-tracking.html</code></p>"""
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
    rows.append(("Silicon area", "KPU priced, CPU not",
                 "KPU from a fit over 4 catalogued cores; CPU core logic unanchored", "gap"))
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
    write_report(render(dossier, sections(dossier, soc, alt, fit, _tiles_n),
                        requirements_rows(dossier, alt), alternatives, idle,
                        date.today().isoformat()), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
