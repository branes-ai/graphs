#!/usr/bin/env python
"""
The CPU + KPU design state space against the mission capabilities
(graphs#269 Phase 7).

One row per (design point, mission): what each engine would have to
sustain, what the measured efficiencies already say, whether the
domain-flow ceiling rules the point out, whether the memory system can
carry the traffic, and where the die area and power stand. It is the join
of the three analyses a partner needs to read together:

  analyze_required_efficiency.py   what the silicon must sustain
  analyze_kpu_ceiling.py           what the KPU fabric could sustain at best
  sweep_soc.py                     area, power and DRAM

**Every column is IP-neutral where it can be.** Besides the fraction of
dense peak, each requirement is also given as sustained throughput -- GOP/s
per CPU core, TOPS on the accelerator -- so a partner can hold their own
core against it without adopting this model's peaks.

**The power column is a floor.** `datapath_floor_w` charges every op once
at the node's energy per op, so no efficiency enters and nothing can come
in under it. A `+` marks a floor with gaps: a T-series KPU core's tiles
span two logic libraries, so its template states no one datapath class and
its ops cannot be charged at all -- there the floor is the CPU's alone.

**Verdicts.** `no` is proven: a requirement above 1, a memory system that
cannot carry the traffic, a measured utilization above 1, or a ceiling
below what the mission needs. `open` is everything else: this model never
says yes, because a complete proof needs a schedule and a power figure
that no gap-free input exists for yet.

Usage:
    python cli/analyze_mission_matrix.py --design kpu_t128_n7 --all
    python cli/analyze_mission_matrix.py --state-space --output matrix.csv
    python cli/analyze_mission_matrix.py --design kpu_t256_n16 kpu_t256_n7 \\
        --cpu-clusters 1 2 3 --memory lpddr5_phy_256b lpddr5x_phy_512b \\
        --mission humanoid_cobot_human_adjacent_contact_rich --verbose

Exit codes:
    0 = report produced
    2 = unknown design, mission, IP or efficiency table, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from embodied_schemas import load_compute_products, load_process_nodes  # noqa: E402

from graphs.core.pipeline_workload import load_autonomy_workload  # noqa: E402
from graphs.estimation.soc import (  # noqa: E402
    Override,
    RequiredEfficiency,
    engines_of,
    load_efficiency_tables,
    load_kernel_classes,
    required_efficiency,
)
from graphs.estimation.soc.domainflow import SHAPES, fabric_ceilings  # noqa: E402
from graphs.estimation.soc.power import datapath_library  # noqa: E402
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product  # noqa: E402
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402
from graphs.hardware.soc.kpu_cores import KPU_CORES, sku_of  # noqa: E402
from graphs.reporting.output_format import csv_writer, detect_format, write_report  # noqa: E402

#: The state space the Phase 7 document lays out: every generated KPU core
#: at the node its SKU states, the CPU complement, and the memory system.
STATE_SPACE_CPU_CLUSTERS = (1, 2, 3)
STATE_SPACE_MEMORY = ("lpddr5_phy_256b", "lpddr5x_phy_256b", "lpddr5x_phy_512b", "hbm3_1stack")

#: Kernel classes the domain-flow model states a schedule for. Under the
#: default mapping the accelerator takes only these, and the CPU carries
#: the rest -- which is the honest division of labour, and the one a CPU
#: partner has to size for.
SCHEDULED_CLASSES = frozenset(shape.kernel_class for shape in SHAPES)


def schedule_aware_mapping(workload, profile, soc, kernels) -> Dict[str, Optional[str]]:
    """The accelerator takes the classes it has a domain-flow schedule for
    and can run; every other stage goes to the CPU."""
    engines = engines_of(soc)
    accelerator = next((n for n, e in engines.items() if e.kind.value in ("kpu", "gpu", "npu")),
                       None)
    cpu = next((n for n, e in engines.items() if e.kind.value == "cpu"), None)
    out: Dict[str, Optional[str]] = {}
    for demand in workload.demands(profile):
        stage = demand.stage
        scheduled = kernels.of(stage.key) in SCHEDULED_CLASSES
        target = accelerator if (scheduled and accelerator) else cpu
        if target is not None and _dense_ok(stage, engines[target]):
            out[stage.key] = target
        elif cpu is not None and _dense_ok(stage, engines[cpu]):
            out[stage.key] = cpu
        else:
            out[stage.key] = None
    return out


def _dense_ok(stage, engine) -> bool:
    """Whether an engine has a format for every class the stage runs."""
    from graphs.estimation.soc import execution_format
    from graphs.core.pipeline_workload import CLASS_NAMES

    return all(execution_format(cls, engine.formats) is not None
               for cls, share in zip(CLASS_NAMES, stage.class_split) if share > 0)


def energy_floor_w(workload, profile, soc, mapping) -> tuple:
    """Watts the datapath alone costs at the node's energy per op, and the
    gaps. Independent of efficiency: every op is charged once, so this is a
    floor on dynamic power whatever the schedule achieves."""
    from graphs.core.pipeline_workload import CLASS_NAMES
    from graphs.estimation.soc import execution_format

    blocks = {b.name: b for b in soc.blocks}
    engines = engines_of(soc)
    table = soc.node.energy_per_op_pj
    watts, gaps = 0.0, []
    for demand in workload.demands(profile):
        stage = demand.stage
        target = mapping.get(stage.key)
        if target is None:
            gaps.append(f"{stage.key}: no engine")
            continue
        lib = datapath_library(blocks[target])
        for cls, share in zip(CLASS_NAMES, stage.class_split):
            if share <= 0:
                continue
            fmt = execution_format(cls, engines[target].formats)
            key = None if (lib is None or fmt is None) else f"{lib.value}:{fmt}"
            pj = table.get(key) if key else None
            if pj is None:
                gaps.append(f"{stage.key} class {cls}: no energy for {key or 'an unrunnable class'}")
                continue
            watts += stage.ops_per_call * share * demand.rate_hz * pj * 1e-12
    return watts, tuple(gaps)


def _fmt(value, spec: str = ".3g", none: str = "") -> str:
    return none if value is None else format(value, spec)


@dataclass(frozen=True)
class Point:
    design: str
    node: str
    kpu_sku: Optional[str]
    cpu_clusters: int
    memory: str


def _engine_row(req: RequiredEfficiency, engine, soc, ceilings) -> dict:
    """One engine's demand, in fractions of peak and in throughput."""
    e = engines_of(soc)[engine.engine]
    # The stage mix decides which format's peak to quote; take the one the
    # most stages use, which is what the requirement is dominated by.
    formats = [f for s in req.stages if s.engine == engine.engine for f in s.formats.values()]
    fmt = max(set(formats), key=formats.count) if formats else next(iter(e.formats), "")
    peak = e.server_peak_ops_per_s(fmt) if fmt else 0.0
    need = engine.required_efficiency
    return {
        "engine": engine.engine,
        "kind": engine.kind,
        "servers": engine.servers,
        "stages": len(engine.stages),
        "format": fmt,
        "peak_per_server_gops": peak / 1e9,
        "needs_fraction_of_peak": need,
        "needs_sustained_gops_per_server": None if need is None else need * peak / 1e9,
        "needs_sustained_tops_total": None if need is None
        else need * peak * engine.servers / 1e12,
        "measured_utilization": engine.utilization_at_known,
        "measured_is_lower_bound": engine.known_is_lower_bound,
        "unpriced_stages": len(engine.unpriced),
    }


def _ceiling_verdict(req: RequiredEfficiency, engine, ceilings, kernels) -> Optional[bool]:
    """False when the fabric's own ceiling cannot carry the stages: their
    time at the ceiling exceeds the period. None when a stage has none."""
    if ceilings is None:
        return None
    total = 0.0
    for stage in req.stages:
        if stage.engine != engine.engine or not stage.class_seconds:
            continue
        for cls, seconds in stage.class_seconds.items():
            ceiling = ceilings.best(kernels.of(stage.stage), stage.formats[cls])
            if ceiling is None or not ceiling.value:
                return None
            total += seconds * stage.rate_hz / ceiling.value
    return None if total == 0 else total <= 1.0


def _row(point: Point, req: RequiredEfficiency, soc, ceilings, kernels, profile,
         floor_w: float, floor_gaps) -> dict:
    engines = {e.engine: e for e in req.engines}
    kpu = next((e for e in req.engines if e.kind == "kpu"), None)
    cpu = next((e for e in req.engines if e.kind == "cpu"), None)
    accelerator = kpu or next((e for e in req.engines if e.kind in ("gpu", "npu")), None)
    dram = (req.dram_demand_gb_per_s / req.dram_supply_gb_per_s
            if req.dram_supply_gb_per_s else None)
    rows = {e.engine: _engine_row(req, e, soc, ceilings) for e in req.engines}
    acc = rows.get(accelerator.engine) if accelerator else {}
    cpu_row = rows.get(cpu.engine) if cpu else {}
    ceiling_ok = _ceiling_verdict(req, accelerator, ceilings, kernels) if accelerator else None

    budget = profile.power_budget_w
    over_budget = floor_w > budget
    reasons = []
    if over_budget:
        reasons.append(f"datapath energy alone is {floor_w:.1f} W of a {budget:g} W budget")
    if dram is not None and dram > 1.0:
        reasons.append(f"DRAM {dram:.2f}")
    for name, e in engines.items():
        if e.required_efficiency is not None and e.required_efficiency > 1.0:
            reasons.append(f"{name} needs {e.required_efficiency:.2f} of dense peak")
        if e.out_of_reach:
            reasons.append(f"{name}: {', '.join(e.out_of_reach)} out of reach on DRAM alone")
        if e.utilization_at_known is not None and e.utilization_at_known > 1.0:
            reasons.append(f"{name} measures {e.utilization_at_known:.2f}")
    if ceiling_ok is False:
        reasons.append(f"{accelerator.engine} over its domain-flow ceiling")
    if req.unrunnable:
        reasons.append(f"no engine runs {', '.join(req.unrunnable)}")

    return {
        "mission": profile.id,
        "form_factor": profile.form_factor,
        "regime": profile.regime or "",
        "power_budget_w": profile.power_budget_w,
        "deadline_ms": profile.deadline_ms,
        "design": point.design,
        "node": point.node,
        "kpu_sku": point.kpu_sku or "",
        "cpu_clusters": point.cpu_clusters,
        "cpu_cores": engines_of(soc)["cpu"].servers if "cpu" in engines_of(soc) else 0,
        "cpu_stages": cpu_row.get("stages", 0),
        "memory": point.memory,
        "dram_gb_per_s": req.dram_supply_gb_per_s,
        "dram_utilization": dram,
        "acc_needs_fraction": acc.get("needs_fraction_of_peak"),
        "acc_needs_tops": acc.get("needs_sustained_tops_total"),
        "acc_measured_utilization": acc.get("measured_utilization"),
        "acc_ceiling_allows": ceiling_ok,
        "cpu_needs_fraction": cpu_row.get("needs_fraction_of_peak"),
        "cpu_needs_gops_per_core": cpu_row.get("needs_sustained_gops_per_server"),
        "cpu_measured_utilization": cpu_row.get("measured_utilization"),
        "datapath_floor_w": floor_w,
        "datapath_floor_is_lower_bound": bool(floor_gaps),
        "power_budget_fraction_used": None if not budget else floor_w / budget,
        "verdict": "no" if reasons else "open",
        "why": "; ".join(reasons),
        "confidence": req.estimation_confidence.level.value,
    }


def _points(args, designs) -> List[Point]:
    if args.state_space:
        chosen = sorted(d for d in designs
                        if any(b.ip in KPU_CORES for b in designs[d].blocks))
    else:
        chosen = list(args.design or ())
    if not chosen:
        raise KeyError("no design chosen: pass --design or --state-space")
    out = []
    for name in chosen:
        if name not in designs:
            raise KeyError(f"unknown design {name!r}; have {', '.join(sorted(designs))}")
        core = next((b.ip for b in designs[name].blocks if b.ip in KPU_CORES), None)
        for clusters in args.cpu_clusters:
            for memory in args.memory:
                out.append(Point(name, designs[name].process_node,
                                 sku_of(core) if core else None, clusters, memory))
    return out


def _table(rows: List[dict], columns: List[str]) -> str:
    if not rows:
        return "(no point)\n"
    shown = [{c: r[c] for c in columns} for r in rows]
    widths = {c: max(len(c), *(len(str(r[c])) for r in shown)) for c in columns}
    out = ["  ".join(c.ljust(widths[c]) for c in columns),
           "  ".join("-" * widths[c] for c in columns)]
    out += ["  ".join(str(r[c]).ljust(widths[c]) for c in columns) for r in shown]
    return "\n".join(out) + "\n"


def _text(rows: List[dict], verbose: bool) -> str:
    display = [{
        "mission": r["mission"][:38],
        "design": r["design"],
        "cpu": f"{r['cpu_cores']}c/{r['cpu_stages']}st",
        "memory": r["memory"].replace("_phy", "").replace("lpddr5", "lp5"),
        "dram": _fmt(r["dram_utilization"], ".2f"),
        "acc_needs": _fmt(r["acc_needs_fraction"], ".1%"),
        "acc_tops": _fmt(r["acc_needs_tops"], ".1f"),
        "cpu_needs": _fmt(r["cpu_needs_fraction"], ".1%"),
        "cpu_gops": _fmt(r["cpu_needs_gops_per_core"], ".1f"),
        "measured": _fmt(r["cpu_measured_utilization"] or r["acc_measured_utilization"], ".2f"),
        "floor_w": _fmt(r["datapath_floor_w"], ".2f") + ("+" if r["datapath_floor_is_lower_bound"] else ""),
        "budget_w": _fmt(r["power_budget_w"], "g"),
        "verdict": r["verdict"],
    } for r in rows]
    columns = list(display[0]) if display else []
    text = _table(display, columns)
    if verbose:
        text += "\nwhy:\n" + "".join(
            f"  {r['mission'][:38]:38s} {r['design']:14s} {r['why']}\n" for r in rows if r["why"])
    return text


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="The CPU + KPU state space against the mission capabilities "
                    "(graphs#269 Phase 7).")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--design", nargs="+", help="Design id(s)")
    which.add_argument("--state-space", action="store_true",
                       help="Every design with a generated KPU core")
    missions = parser.add_mutually_exclusive_group()
    missions.add_argument("--mission", action="append", help="Mission id or name (repeatable)")
    missions.add_argument("--all", action="store_true", help="Every mission (the default)")
    parser.add_argument("--cpu-clusters", nargs="+", type=int, default=list(STATE_SPACE_CPU_CLUSTERS),
                        help="Cluster counts of 4 Cortex-A78AE cores (default 1 2 3)")
    parser.add_argument("--memory", nargs="+", default=list(STATE_SPACE_MEMORY),
                        help="Memory IP ids to sweep (default: the four the catalog states)")
    parser.add_argument("--efficiency", default="orin_nano_measured_v1",
                        help="Table whose measured figures sit beside the requirement")
    parser.add_argument("--mapping", choices=["schedule-aware", "capability"],
                        default="schedule-aware",
                        help="schedule-aware (default): the accelerator takes only the kernel "
                             "classes the domain-flow model has a schedule for, and the CPU "
                             "carries the rest. capability: the accelerator takes everything it "
                             "can run, which leaves the CPU idle and says nothing about it.")
    parser.add_argument("--no-ceilings", action="store_true",
                        help="Skip the domain-flow ceiling column")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print why each point failed")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    try:
        workload = load_autonomy_workload()
        designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
        kernels = load_kernel_classes()
        tables = load_efficiency_tables()
        if args.efficiency not in tables:
            raise KeyError(f"unknown efficiency table {args.efficiency!r}")
        table = tables[args.efficiency]
        for ip in args.memory:
            if ip not in library:
                raise KeyError(f"unknown IP template {ip!r}")
        profiles = list(workload.profiles)
        if args.mission:
            wanted = [m.lower() for m in args.mission]
            profiles = [p for p in profiles
                        if any(w in (p.id.lower(), (p.regime or "").lower(), p.name.lower())
                               for w in wanted)]
            if not profiles:
                raise KeyError(f"no mission matches {args.mission}")

        ceilings: Dict[str, object] = {}
        rows: List[dict] = []
        for point in _points(args, designs):
            design = designs[point.design]
            design = Override(target="block:cpu.count",
                              values=[point.cpu_clusters]).apply(design, point.cpu_clusters)
            design = Override(target="block:memory.ip",
                              values=[point.memory]).apply(design, point.memory)
            soc = compose_soc(design, library, nodes, None)
            fabric = None
            if point.kpu_sku and not args.no_ceilings:
                if point.kpu_sku not in ceilings:
                    ceilings[point.kpu_sku] = fabric_ceilings(input_spec_from_compute_product(
                        load_compute_products()[point.kpu_sku]))
                fabric = ceilings[point.kpu_sku]
            for profile in profiles:
                mapping = (schedule_aware_mapping(workload, profile, soc, kernels)
                           if args.mapping == "schedule-aware" else None)
                req = required_efficiency(workload, profile, soc, mapping=mapping,
                                          table=table, kernels=kernels)
                assigned = {s.stage: s.engine or None for s in req.stages}
                floor_w, floor_gaps = energy_floor_w(workload, profile, soc, assigned)
                rows.append(_row(point, req, soc, fabric, kernels, profile, floor_w, floor_gaps))
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fmt = detect_format(args.output)
    if fmt == "json":
        payload = json.dumps(rows, indent=2)
    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv_writer(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        payload = buf.getvalue()
    elif fmt == "md":
        columns = ["mission", "design", "cpu_cores", "memory", "dram_utilization",
                   "acc_needs_fraction", "cpu_needs_gops_per_core", "verdict"]
        header = "| " + " | ".join(columns) + " |"
        payload = "\n".join(
            [f"## {len(rows)} points", "", header, "|" + "|".join(["---"] * len(columns)) + "|"]
            + ["| " + " | ".join(_fmt(r[c], ".3g") if isinstance(r[c], float) else str(r[c])
                                 for c in columns) + " |" for r in rows]) + "\n"
    else:
        payload = _text(rows, args.verbose)
    write_report(payload, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
