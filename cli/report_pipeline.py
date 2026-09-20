#!/usr/bin/env python
"""
What each pipeline stage demands, and what each engine gives it
(graphs#269 Phase 7.4).

Writes a standalone HTML page: pick a mission, and see its stages in
pipeline order with, for each, the operations and bytes it needs per second
of mission, the precision classes it runs in, and the share of each engine
one second of mission would consume there.

    python cli/report_pipeline.py -o docs/assessments/pipeline-demand.html
    python cli/report_pipeline.py --design kpu_t256_n7 --cpu-clusters 3
    python cli/report_pipeline.py --mission "air superiority" -o one.html

Exit codes:
    0 = report written
    2 = unknown design, mission or efficiency table, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402
from embodied_schemas import load_compute_products, load_process_nodes  # noqa: E402

from graphs.core.pipeline_workload import (  # noqa: E402
    CLASS_NAMES,
    REACTIVE_CHAIN,
    load_autonomy_workload,
)
from graphs.estimation.soc import (  # noqa: E402
    Override,
    engines_of,
    load_efficiency_tables,
    load_kernel_classes,
    required_efficiency,
)
from graphs.estimation.soc.domainflow import fabric_ceilings  # noqa: E402
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product  # noqa: E402
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402
from graphs.hardware.soc.kpu_cores import KPU_CORES, sku_of  # noqa: E402
from graphs.reporting.output_format import write_report  # noqa: E402
from graphs.reporting.pipeline_view import StageRow, render  # noqa: E402

#: What each tier of the workload is for. The ids come from the workload;
#: these are the labels a reader needs to see the shape of the pipeline.
TIERS: Dict[str, str] = {
    "T1": "sensor front end",
    "T2": "state estimation",
    "T3": "mapping",
    "T4": "semantics and policy",
    "T5": "exploration and planning",
    "T6": "reactive control",
    "T7": "actuation",
}

INTRO = """
<h1>What each stage demands, and what each engine gives it</h1>
<p class="lede">A mission is not one workload. It is a chain of stages over seven tiers, each
with its own arithmetic, its own byte traffic and its own precision floor &mdash; and each engine
answers them very differently. This is that, stage by stage.</p>

<h2>How to read a row</h2>
<p>Left to right: the <b>stage</b> and the <b>kernel class</b> its arithmetic belongs to; the
<b>precision</b> classes it needs, as a share of its own operations; what it <b>demands</b> per
second of mission, in operations and in bytes, on a log scale; and then one column per
<b>engine</b>, showing the share of that engine one second of mission would consume there.</p>
<p>A bar drawn to the full width, in red, is past one whole engine: that stage alone cannot keep
up there, before anything else is scheduled beside it. A cell with no bar says why there is no
figure &mdash; usually that the engine has no format for a class the stage needs, or that nothing
has measured that kernel class on that engine.</p>
<p>The <b>step of a bar</b> is where its efficiency came from: a measurement on real silicon, or
the domain-flow ceiling, which is an upper bound rather than a measurement. The ceiling step is
the lighter of the two, and a stage drawn in it is being given the benefit of the doubt. The
legend names both, and every figure on the page is repeated in the table under each chart.</p>

<h2>Why the rows are grouped</h2>
<p>The tiers are the pipeline's own decomposition: sensing feeds state estimation, which feeds
mapping, which feeds semantics and planning, which feed reactive control and actuation. The
<b>filled circle</b> marks the stages on the sense-to-act chain &mdash; the path whose total
latency the mission's deadline applies to. A stage off that chain can be late without the robot
being late; a stage on it cannot.</p>
"""


def _matrix_module():
    spec = importlib.util.spec_from_file_location(
        "mission_matrix_cli", REPO / "cli" / "analyze_mission_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(design_id: str, clusters: int, memory: str, missions: Optional[List[str]],
          efficiency: str):
    matrix = _matrix_module()
    workload = load_autonomy_workload()
    kernels = load_kernel_classes()
    tables = load_efficiency_tables()
    if efficiency not in tables:
        raise KeyError(f"unknown efficiency table {efficiency!r}")
    table = tables[efficiency]
    designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
    if design_id not in designs:
        raise KeyError(f"unknown design {design_id!r}")
    design = Override(target="block:cpu.count", values=[clusters]).apply(designs[design_id],
                                                                        clusters)
    design = Override(target="block:memory.ip", values=[memory]).apply(design, memory)
    soc = compose_soc(design, library, nodes, None)
    engines = engines_of(soc)
    core = next((b.ip for b in designs[design_id].blocks if b.ip in KPU_CORES), None)
    ceilings = (fabric_ceilings(input_spec_from_compute_product(
        load_compute_products()[sku_of(core)])) if core else None)

    profiles = list(workload.profiles)
    if missions:
        wanted = [m.lower() for m in missions]
        profiles = [p for p in profiles if any(
            w in (p.id.lower(), (p.regime or "").lower(), p.name.lower()) for w in wanted)]
        if not profiles:
            raise KeyError(f"no mission matches {missions}")

    out: Dict[str, List[StageRow]] = {}
    for profile in profiles:
        assigned = matrix.schedule_aware_mapping(workload, profile, soc, kernels)
        # One run per engine, so every stage is costed on every engine and
        # not only on the one the schedule picked.
        per_engine = {
            name: {s.stage: s for s in required_efficiency(
                workload, profile, soc,
                mapping={d.stage.key: name for d in workload.demands(profile)},
                table=table, kernels=kernels).stages}
            for name in engines}
        rows: List[StageRow] = []
        for demand in workload.demands(profile):
            stage = demand.stage
            cells = {}
            for name, engine in engines.items():
                requirement = per_engine[name][stage.key]
                if requirement.dense_seconds is None:
                    cells[name] = (None, "unpriced", None, "no format for a class it needs")
                    continue
                kernel = kernels.of(stage.key)
                efficiency_value, provenance = matrix_efficiency(
                    requirement, kernel, engine.kind.value, table,
                    ceilings if engine.kind.value == "kpu" else None)
                if not efficiency_value:
                    cells[name] = (None, "unpriced", None,
                                   _why_unpriced(requirement, kernel, table,
                                                 ceilings if engine.kind.value == "kpu" else None,
                                                 engine.kind.value))
                    continue
                share = (requirement.dense_seconds / efficiency_value
                         * requirement.rate_hz / (engine.servers or 1))
                cells[name] = (share, provenance, efficiency_value, None)
            rows.append(StageRow(
                key=stage.key, name=stage.name, tier=stage.pipeline_tier,
                kernel_class=kernels.of(stage.key).value, unit=stage.unit,
                rate_hz=demand.rate_hz, ops_per_s=stage.ops_per_call * demand.rate_hz,
                bytes_per_s=stage.bytes_per_call * demand.rate_hz,
                class_split=dict(zip(CLASS_NAMES, stage.class_split)),
                on_reactive_chain=stage.key in REACTIVE_CHAIN,
                assigned=assigned.get(stage.key), engines=cells))
        out[profile.id] = rows
    return out, list(engines), soc


def _why_unpriced(requirement, kernel, table, ceilings, kind: str) -> str:
    """Which format is missing, so the cell names the measurement to take
    rather than only saying there is none. On an engine with a domain-flow
    model, also say whether a ceiling exists, since a missing schedule and a
    missing measurement call for different work."""
    formats = sorted(set(requirement.formats.values()))
    missing = [fmt for fmt in formats
               if (entry := table.lookup(kernel, kind, fmt)) is None or not entry.known]
    known = [fmt for fmt in formats if fmt not in missing]
    if not missing:
        return "no figure"
    reason = f"no {'/'.join(missing)} figure"
    if known:
        reason += f" ({'/'.join(known)} measured)"
    elif ceilings is not None:
        unscheduled = [fmt for fmt in formats
                       if (c := ceilings.best(kernel, fmt)) is None or not c.value]
        reason += ", no schedule" if unscheduled else ", ceiling short"
    return reason


def matrix_efficiency(requirement, kernel, kind, table, ceilings):
    """The best efficiency anything states, as the frontier module picks it."""
    from graphs.estimation.soc.frontier import _efficiency

    return _efficiency(requirement, kernel, kind, table, ceilings)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--design", default="kpu_t128_n7", help="Design id (default kpu_t128_n7)")
    parser.add_argument("--cpu-clusters", type=int, default=3,
                        help="Clusters of 4 Cortex-A78AE cores (default 3)")
    parser.add_argument("--memory", default="lpddr5_phy_256b", help="Memory IP id")
    parser.add_argument("--mission", action="append", help="Mission id, regime or name")
    parser.add_argument("--efficiency", default="orin_nano_measured_v1")
    parser.add_argument("--output", "-o", help="Write here (default: stdout)")
    args = parser.parse_args(argv)

    try:
        missions, engines, soc = build(args.design, args.cpu_clusters, args.memory,
                                       args.mission, args.efficiency)
        workload = load_autonomy_workload()
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    profiles = {p.id: p for p in workload.profiles}
    titles = {m: f"{profiles[m].form_factor}: {profiles[m].name}" for m in missions}
    cores = next((e.servers for name, e in engines_of(soc).items() if e.kind.value == "cpu"), 0)
    subtitles = {
        m: (f"{profiles[m].power_budget_w:g} W budget, {profiles[m].deadline_ms:g} ms deadline"
            f" - {args.design} at {soc.node.id}, {cores} CPU cores, {args.memory}")
        for m in missions}
    write_report(render(missions, titles, subtitles, engines, TIERS, INTRO,
                        date.today().isoformat()), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
