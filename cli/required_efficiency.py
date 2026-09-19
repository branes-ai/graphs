#!/usr/bin/env python
"""
What a design would have to achieve (graphs#269 Phase 6).

`analyze_soc.py` prices a stage as ops / (peak x efficiency) and reports a
gap when no efficiency exists. For a KPU none does: no silicon has run these
kernels. This asks the question the other way round, which needs no figure
nobody has:

    required efficiency = sum over the engine's stages of
                          (ops per call / dense peak) x rate / servers

the fraction of dense peak the engine must sustain for the profile to fit.
It is exact arithmetic on the workload's ops and the design's peaks. Above
1, no efficiency suffices: the silicon is too small, not too slow.

With --efficiency, each stage's measured figure is printed beside its
requirement, and the engine's utilization at those figures. That figure is a
LOWER BOUND whenever the table prices fewer than all of the engine's stages,
because the rest would add to it.

Utilization at or under 1 is necessary, not sufficient: use
`analyze_soc.py` for the schedule, response times and power.

Usage:
    python cli/required_efficiency.py --design kpu_heterogeneous_h64 --regime "air superiority"
    python cli/required_efficiency.py --design orin_class_reference --all \\
        --efficiency orin_nano_measured_v1 --verbose
    python cli/required_efficiency.py --design kpu_heterogeneous_h64 --regime "far flight" \\
        --node tsmc_n5 --mapping capability --output need.json

Exit codes:
    0 = report produced
    2 = unknown design, profile, node, table or mapping, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from embodied_schemas import load_process_nodes  # noqa: E402

from graphs.core.pipeline_workload import load_autonomy_workload  # noqa: E402
from graphs.estimation.soc import (  # noqa: E402
    RequiredEfficiency,
    find_mapping,
    load_efficiency_tables,
    load_kernel_classes,
    load_mapping,
    required_efficiency,
)
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402
from graphs.reporting.output_format import csv_writer, detect_format, write_report  # noqa: E402


def _fmt(value, spec: str = ".3g", none: str = "n/a") -> str:
    return none if value is None else format(value, spec)


def _verdict(value: Optional[bool]) -> str:
    return "open" if value is None else ("yes" if value else "NO")


def _summary_lines(r: RequiredEfficiency) -> List[str]:
    mem = r.dram_demand_gb_per_s / r.dram_supply_gb_per_s if r.dram_supply_gb_per_s else None
    lines = [
        f"profile: {r.profile.id}  ({r.profile.regime or 'no regime'})",
        f"design:  {r.design} at {r.node};  confidence {r.estimation_confidence.level.value}"
        f" -- {r.estimation_confidence.source}",
    ]
    for e in r.engines:
        need = e.required_efficiency
        line = (f"{e.engine} ({e.kind}, {e.servers} server(s), {len(e.stages)} stage(s)): "
                f"needs {_fmt(need, '.1%')} of dense peak -- reachable: {_verdict(e.reachable)}")
        if e.out_of_reach:
            line += f"; DRAM alone fills the period for {', '.join(e.out_of_reach)}"
        if e.utilization_at_known is not None:
            line += (f"; at {r.comparison_table} efficiencies utilization is "
                     f"{e.utilization_at_known:.2f}"
                     + (f" (LOWER BOUND: {len(e.unpriced)} of {len(e.stages)} stages unpriced)"
                        if e.known_is_lower_bound else ""))
        elif r.comparison_table:
            line += f"; {r.comparison_table} prices none of its stages"
        lines.append(line)
    if r.unrunnable:
        lines.append(f"no engine runs: {', '.join(r.unrunnable)}")
    lines.append(f"DRAM: demand {r.dram_demand_gb_per_s:.1f} GB/s vs sustained supply "
                 f"{_fmt(r.dram_supply_gb_per_s, '.1f')} GB/s"
                 + (f" -- utilization {mem:.2f}" if mem else ""))
    return lines


def _stage_rows(r: RequiredEfficiency) -> List[dict]:
    return [{
        "stage": s.stage,
        "engine": s.engine or "-",
        "formats": ",".join(f"{c}:{f}" for c, f in s.formats.items()) or "-",
        "rate_hz": f"{s.rate_hz:.3g}",
        "at_peak": _fmt(s.occupancy_at_peak, ".3g"),
        "memory": _fmt(s.memory_occupancy, ".3g"),
        "known_eff": _fmt(s.known_efficiency, ".3g"),
        "note": s.gap or ("DRAM alone fills the period" if s.out_of_reach else ""),
    } for s in r.stages]


def _table(rows: List[dict]) -> str:
    if not rows:
        return "(none)\n"
    widths = {k: max(len(k), *(len(str(row[k])) for row in rows)) for k in rows[0]}
    out = ["  ".join(k.ljust(widths[k]) for k in rows[0]),
           "  ".join("-" * widths[k] for k in rows[0])]
    out += ["  ".join(str(row[k]).ljust(widths[k]) for k in rows[0]) for row in rows]
    return "\n".join(out) + "\n"


def _md_table(rows: List[dict]) -> List[str]:
    if not rows:
        return ["(none)"]
    keys = list(rows[0])
    return (["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
            + ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows])


def _render(results: List[RequiredEfficiency], fmt: str, verbose: bool) -> str:
    if fmt == "json":
        return json.dumps([r.to_dict() for r in results], indent=2)
    if fmt == "csv":
        rows = [{"profile": r.profile.id, "regime": r.profile.regime or "", "design": r.design,
                 "node": r.node, "engine": e.engine, "kind": e.kind, "servers": e.servers,
                 "stages": len(e.stages), "required_efficiency": e.required_efficiency,
                 "reachable": e.reachable, "utilization_at_known": e.utilization_at_known,
                 "utilization_at_known_is_lower_bound": e.known_is_lower_bound,
                 "confidence": r.estimation_confidence.level.value}
                for r in results for e in r.engines]
        buf = io.StringIO()
        writer = csv_writer(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()
    parts = []
    for r in results:
        if fmt == "md":
            lines = [f"## {r.profile.id}", ""] + [f"- {x}" for x in _summary_lines(r)]
            if verbose:
                lines += ["", *_md_table(_stage_rows(r))]
            parts.append("\n".join(lines) + "\n")
        else:
            text = "\n".join(_summary_lines(r)) + "\n"
            if verbose:
                text += "\n" + _table(_stage_rows(r))
            parts.append(text)
    sep = "\n" if fmt == "md" else "\n" + "=" * 72 + "\n"
    return sep.join(parts)


def _profiles(workload, args) -> List:
    if args.all:
        return list(workload.profiles)
    wanted = args.profile or args.regime or []
    out = []
    for name in wanted:
        match = [p for p in workload.profiles
                 if name in (p.id, p.regime, p.name) or name.lower() == (p.regime or "").lower()]
        if not match:
            raise KeyError(f"no profile, regime or mission named {name!r}")
        out += [p for p in match if p not in out]
    return out or list(workload.regimes())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="The efficiency each engine must sustain to carry a profile (graphs#269 Phase 6).")
    parser.add_argument("--design", required=True, help="Design id from soc_designs/designs/")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--profile", action="append", help="Profile id or mission name (repeatable)")
    which.add_argument("--regime", action="append", help="Named regime (repeatable)")
    which.add_argument("--all", action="store_true", help="Every profile in the workload")
    parser.add_argument("--node", help="Process node (default: the design's own)")
    parser.add_argument("--mapping", default="auto",
                        help="auto (the shipped mapping, else capability), capability "
                             "(each stage on the engine that can run it, accelerators first), "
                             "or a mapping .yaml file")
    parser.add_argument("--efficiency", help="Table to print beside the requirement, "
                                             "e.g. orin_nano_measured_v1")
    parser.add_argument("--sustained-fraction", type=float, default=None,
                        help="Sustained fraction of peak DRAM bandwidth (default 0.65)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Add the per-stage table")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    try:
        workload = load_autonomy_workload()
        designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
        if args.design not in designs:
            raise KeyError(f"unknown design {args.design!r}; have {', '.join(sorted(designs))}")
        soc = compose_soc(designs[args.design], library, nodes, args.node)
        kernels = load_kernel_classes()
        table = None
        if args.efficiency:
            tables = load_efficiency_tables()
            if args.efficiency not in tables:
                raise KeyError(f"unknown efficiency table {args.efficiency!r}; "
                               f"have {', '.join(sorted(tables))}")
            table = tables[args.efficiency]
        mapping = None
        if args.mapping != "capability":
            if args.mapping not in ("auto",):
                shipped = load_mapping(Path(args.mapping))
            else:
                shipped = find_mapping(args.design, workload.version)
            if shipped is not None:
                # A split assignment names several engines; this analysis
                # puts a stage on one engine, so a split is left to the
                # capability rule rather than charged to one of its parts.
                mapping = {stage: engine for stage, engine in shipped.assignments().items()
                           if isinstance(engine, str)}
        profiles = _profiles(workload, args)
        kwargs = {} if args.sustained_fraction is None else {
            "sustained_fraction": args.sustained_fraction}
        results = [required_efficiency(workload, p, soc, mapping=mapping, table=table,
                                       kernels=kernels, **kwargs)
                   for p in profiles]
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_report(_render(results, detect_format(args.output), args.verbose), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
