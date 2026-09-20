#!/usr/bin/env python
"""
The CPU + KPU state space, brushable (graphs#269 Phase 7.5).

Writes a standalone HTML page: every mission capability down, every
configuration the parts list can build across, and a cell saying how close
that pair comes to real time. Drag across the columns to select a slab of
the space and read back what its members share and how many missions
survive it; raise the headroom control to ask for margin rather than bare
real time.

    python cli/report_state_space.py -o docs/assessments/state-space.html
    python cli/report_state_space.py --headroom 2 -o with-margin.html
    python cli/report_state_space.py --format json -o space.json

Exit codes:
    0 = report written
    2 = unknown efficiency table, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from graphs.core.pipeline_workload import load_autonomy_workload  # noqa: E402
from graphs.reporting.output_format import write_report  # noqa: E402
from graphs.reporting.state_space import (  # noqa: E402
    Configuration,
    MissionRow,
    render,
)

INTRO = """
<h1>Which configurations serve which missions, and what it is about them</h1>
<p class="lede">The whole CPU + KPU state space at once: every mission capability down, every
configuration the parts list can build across. One picture for the question a silicon architect
actually arrives with &mdash; not "how fast is this part" but "what has to be true of the part
for this mission to close".</p>

<h2>What a cell says, and what it does not</h2>
<p>A cell is one mission against one configuration: a KPU design, a CPU core count and a memory
interface. Its shade is the <b>real-time factor</b> &mdash; the share of the mission's required
rate that configuration could carry &mdash; computed with the best efficiency anything states for
each stage: a measurement where one exists, otherwise the domain-flow ceiling, which is itself an
upper bound.</p>
<p>That makes the reading one-sided, and it is the whole point. A cell short of the bar is
<b>proven short</b>: even the flattering figure does not reach it, so no amount of tuning inside
this model rescues it. A cell that clears the bar is <b>not ruled out</b> &mdash; it is not proven
to work, because the figure that got it there was a bound. The legend says "still standing" for
exactly that reason.</p>

<h2>How to read the grid</h2>
<p>Columns are grouped by <b>CPU core count</b> and ordered by <b>design</b> within each group,
with the three memory interfaces adjacent. A pattern that follows one dimension is therefore a
pattern you can see: if a mission's surviving cells form the right-hand third of the grid, the
CPU core count decided it and nothing else did. If they form three narrow stripes, the design
decided it, in every core count.</p>
<p>Two readings fall straight out of the picture. <b>Thirteen of the eighteen missions have no
cell that clears the bar at all</b> &mdash; nothing in this parts list serves them, whatever the
accelerator. Of the five that do, two are served by exactly the fifty-one twelve-core
configurations, every design and every node among them, and one is served only by the two T512
designs. The first pair is a CPU problem; the second is a fabric problem. They call for different
silicon.</p>

<h2>The brush</h2>
<p><b>Drag across the columns</b> to select a slab of the space. The readout under the grid names
what the selection has in common and how many missions still stand inside it. <b>The chips</b>
narrow by core count, node or memory. <b>The headroom control</b> raises the bar from bare real
time to a multiple of it &mdash; which is the honest way to ask for margin, since every figure
here flatters its configuration and a design that only just clears 1.0 has nothing in hand.</p>
"""


def _frontier_module():
    """The frontier CLI owns the sweep over the catalogue; reuse it rather
    than building the same product in two places."""
    spec = importlib.util.spec_from_file_location(
        "mission_frontier_cli", REPO / "cli" / "report_mission_frontier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(efficiency: str) -> Tuple[List[Configuration], List[MissionRow]]:
    """The space as columns and rows, with the column order shared by every
    mission so a column index means the same thing in every row."""
    points = _frontier_module().collect(None, efficiency)
    workload = load_autonomy_workload()
    profiles = {p.id: p for p in workload.profiles}

    order: Optional[List[str]] = None
    columns: List[Configuration] = []
    rows: List[MissionRow] = []
    for mission, mission_points in points.items():
        ordered = sorted(mission_points,
                         key=lambda p: (p.cpu_cores, p.design, p.memory))
        keys = [f"{p.design}|{p.cpu_cores}|{p.memory}" for p in ordered]
        if order is None:
            order = keys
            columns = [Configuration(design=p.design, node=p.node, cpu_cores=p.cpu_cores,
                                     memory=p.memory, kpu_sku=p.kpu_sku) for p in ordered]
        elif keys != order:
            raise ValueError(f"mission {mission} spans a different set of configurations")
        profile = profiles[mission]
        rows.append(MissionRow(
            mission=mission,
            title=f"{profile.form_factor}: {profile.name}",
            power_budget_w=profile.power_budget_w, deadline_ms=profile.deadline_ms,
            factors=tuple(p.real_time_factor for p in ordered),
            partial=tuple(p.figures_are_partial for p in ordered)))
    rows.sort(key=lambda r: (-r.standing(), r.title))
    return columns, rows


def _json(columns: Sequence[Configuration], rows: Sequence[MissionRow]) -> str:
    return json.dumps({
        "configurations": [c.to_dict() for c in columns],
        "missions": [{"mission": r.mission, "title": r.title,
                      "power_budget_w": r.power_budget_w, "deadline_ms": r.deadline_ms,
                      "real_time_factor": list(r.factors),
                      "figures_are_partial": list(r.partial),
                      "still_standing": r.standing()} for r in rows],
    }, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--efficiency", default="orin_nano_measured_v1",
                        help="Table the measured efficiencies come from")
    parser.add_argument("--headroom", type=float, default=1.0,
                        help="Real-time multiple the bar starts at (default 1.0)")
    parser.add_argument("--format", choices=["html", "json"], default=None,
                        help="Default: from the --output suffix, else html")
    parser.add_argument("--output", "-o", help="Write here (default: stdout)")
    args = parser.parse_args(argv)
    if args.format is None:
        args.format = "json" if (args.output or "").lower().endswith(".json") else "html"
    if args.headroom <= 0:
        print("error: --headroom must be positive", file=sys.stderr)
        return 2

    try:
        columns, rows = build(args.efficiency)
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        write_report(_json(columns, rows), args.output)
        return 0
    write_report(render(columns, rows, INTRO, date.today().isoformat(), args.headroom),
                 args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
