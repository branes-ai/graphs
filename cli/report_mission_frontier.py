#!/usr/bin/env python
"""
The energy-performance frontier of the CPU + KPU state space, as a page
(graphs#269 Phase 7.3).

Writes a standalone HTML page: one chart per mission capability, energy per
op across and real-time factor up, with the envelope over the catalog and
the region a configuration is already proven short of. Inline SVG, inline
CSS, a few lines of vanilla JavaScript for the hover layer -- nothing to
install to read it.

    python cli/report_mission_frontier.py -o docs/assessments/mission-frontier.html
    python cli/report_mission_frontier.py --mission humanoid_cobot_human_adjacent_contact_rich
    python cli/report_mission_frontier.py --format json -o points.json

Exit codes:
    0 = report written
    2 = unknown mission, IP or efficiency table, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import importlib.util
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
from graphs.estimation.soc import (  # noqa: E402
    Override,
    load_efficiency_tables,
    load_kernel_classes,
    required_efficiency,
)
from graphs.estimation.soc.domainflow import fabric_ceilings  # noqa: E402
from graphs.estimation.soc.frontier import MissionPoint, mission_point  # noqa: E402
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product  # noqa: E402
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402
from graphs.hardware.soc.kpu_cores import KPU_CORES, sku_of  # noqa: E402
from graphs.reporting.mission_frontier import render  # noqa: E402
from graphs.reporting.output_format import write_report  # noqa: E402

CPU_CLUSTERS = (1, 2, 3)
MEMORY = ("lpddr5_phy_256b", "lpddr5x_phy_512b", "hbm3_1stack")

INTRO = """
<h1>What the mission costs, and what the silicon could deliver</h1>
<p class="lede">Every robot mission in the catalogue, against every CPU + KPU configuration
the parts list can build. Two axes, both of them bounds, and one line that decides
feasibility.</p>

<h2>The model, in four steps</h2>
<p>A <b>mission</b> is a sensor configuration and a rate for each stage of a pipeline: how many
cameras at what frame rate, how often the controller runs, how long the robot has between
sensing and acting. It is the demand.</p>
<p>A <b>stage</b> is one pipeline step &mdash; stereo matching, TSDF integration, the policy
network. Each carries an operation count and a byte count per call, and a split across three
precision classes: A runs in INT8, B needs FP16, C needs FP32.</p>
<p>A <b>kernel class</b> is what a stage's arithmetic looks like to silicon: a dense GEMM, an
attention prefill, a min-plus scan, a raycast. Fifteen of them cover the pipeline. This is the
level at which hardware efficiency is a meaningful number, so it is the level everything is
measured and modelled at.</p>
<p>An <b>engine</b> runs a stage: a CPU core, a KPU fabric. The division of labour is not a
preference &mdash; the KPU takes the kernel classes our domain-flow model has a schedule for, and
the CPU carries the rest, because a regular wavefront fabric has no schedule for a factor graph
or a graph search.</p>

<h2>The two axes</h2>
<p><b>Across: energy per operation (pJ).</b> The mission's own arithmetic, priced at the process
node's energy per operation, weighted over the formats each stage runs in and the libraries each
engine issues them from. No efficiency enters and no overhead is added, so
<b>nothing can come in under it</b>: it is a floor, and the true cost lies to the right.</p>
<p><b>Up: real-time factor (log scale).</b> How much of the mission's required rate the
configuration could carry, using the best efficiency anything states for each stage &mdash; a
measurement where one exists, otherwise the domain-flow ceiling, which is itself an upper bound.
So <b>the true factor is no higher</b> than the one drawn: 1.0 is real time, and a point below
1.0 is proven short, because even the flattering figure does not reach it.</p>
<p>Both axes flatter every configuration and in the same direction, which is what makes the
<b>envelope</b> &mdash; the staircase over the points &mdash; a bound on the whole catalogue rather
than a promise about any member of it. Nothing built from these parts sits above or to the left
of that line.</p>

<h2>How to read a chart</h2>
<ul>
<li><b>Colour is the process node</b>, so the horizontal spread within a mission is mostly the
node: energy per operation is roughly 2.5&times; cheaper at N7 than at N16.</li>
<li><b>Dot size is the KPU's tile count.</b> Where dots of every size land on the same height,
the accelerator is not what limits that mission.</li>
<li><b>Vertical position is set by the busiest engine.</b> Hover any dot to see which one, and
whether its efficiency came from a measurement or from a ceiling.</li>
<li><b>The shaded band is proven short of real time.</b> A mission whose whole cloud sits in it
has no configuration in this space, whatever the accelerator.</li>
</ul>
"""


def _matrix_module():
    """The state-space CLI owns the schedule-aware mapping; reuse it rather
    than restating the division of labour in two places."""
    spec = importlib.util.spec_from_file_location(
        "mission_matrix_cli", REPO / "cli" / "analyze_mission_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect(missions: Optional[List[str]], efficiency: str) -> Dict[str, List[MissionPoint]]:
    matrix = _matrix_module()
    workload = load_autonomy_workload()
    kernels = load_kernel_classes()
    tables = load_efficiency_tables()
    if efficiency not in tables:
        raise KeyError(f"unknown efficiency table {efficiency!r}")
    table = tables[efficiency]
    designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
    products = load_compute_products()

    profiles = list(workload.profiles)
    if missions:
        wanted = [m.lower() for m in missions]
        profiles = [p for p in profiles if any(
            w in (p.id.lower(), (p.regime or "").lower(), p.name.lower()) for w in wanted)]
        if not profiles:
            raise KeyError(f"no mission matches {missions}")

    ceilings: Dict[str, object] = {}
    out: Dict[str, List[MissionPoint]] = {p.id: [] for p in profiles}
    for name in sorted(d for d in designs
                       if any(b.ip in KPU_CORES for b in designs[d].blocks)):
        design = designs[name]
        sku = sku_of(next(b.ip for b in design.blocks if b.ip in KPU_CORES))
        if sku not in ceilings:
            ceilings[sku] = fabric_ceilings(input_spec_from_compute_product(products[sku]))
        for clusters in CPU_CLUSTERS:
            with_cpu = Override(target="block:cpu.count",
                                values=[clusters]).apply(design, clusters)
            for memory in MEMORY:
                variant = Override(target="block:memory.ip",
                                   values=[memory]).apply(with_cpu, memory)
                soc = compose_soc(variant, library, nodes, None)
                for profile in profiles:
                    mapping = matrix.schedule_aware_mapping(workload, profile, soc, kernels)
                    requirement = required_efficiency(workload, profile, soc, mapping=mapping,
                                                      table=table, kernels=kernels)
                    out[profile.id].append(mission_point(
                        requirement, soc, workload, profile, kernels, table, ceilings[sku],
                        memory, sku))
    return out


def _tiles() -> Dict[str, int]:
    """Tiles per design, for the marker size."""
    designs, products = load_designs(), load_compute_products()
    out: Dict[str, int] = {}
    for name, design in designs.items():
        core = next((b.ip for b in design.blocks if b.ip in KPU_CORES), None)
        if core is None:
            continue
        spec = input_spec_from_compute_product(products[sku_of(core)])
        out[name] = sum(t.num_tiles for t in spec.kpu_architecture.tiles)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--mission", action="append",
                        help="Mission id, regime or name (repeatable; default all)")
    parser.add_argument("--efficiency", default="orin_nano_measured_v1",
                        help="Table the measured efficiencies come from")
    parser.add_argument("--format", choices=["html", "json"], default=None,
                        help="Default: from the --output suffix, else html")
    parser.add_argument("--output", "-o", help="Write here (default: stdout)")
    args = parser.parse_args(argv)
    if args.format is None:
        args.format = "json" if (args.output or "").lower().endswith(".json") else "html"

    try:
        points = collect(args.mission, args.efficiency)
        workload = load_autonomy_workload()
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        write_report(json.dumps({m: [p.to_dict() for p in ps] for m, ps in points.items()},
                                indent=2), args.output)
        return 0

    profiles = {p.id: p for p in workload.profiles}
    titles = {m: f"{profiles[m].form_factor}: {profiles[m].name}" for m in points}
    subtitles = {
        m: (f"{profiles[m].power_budget_w:g} W budget, {profiles[m].deadline_ms:g} ms deadline, "
            f"{len(points[m])} configurations")
        for m in points}
    write_report(render(points, titles, subtitles, _tiles(), date.today().isoformat(), INTRO),
                 args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
