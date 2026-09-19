#!/usr/bin/env python
"""
SoC L0 Analyzer

Runs a pipeline workload's mission profiles on an SoC design composed at a
process node: per-stage service time and constraint ratio, engine and DRAM
utilization, the sense-to-act chain against its deadline, and power
(graphs#269 Phase 3).

This bypasses UnifiedAnalyzer deliberately: there is no PyTorch model, only a
costed pipeline and silicon to price -- the same exception
analyze_operator_roofline.py documents.

Read the completeness and confidence lines first. Every figure computed over
an input with gaps -- unanchored silicon, a stage no efficiency prices, a
power term the catalog has no energy for -- is a LOWER BOUND (power, area)
or an UPPER BOUND (TOPS/W), and "limited by" names each gap. Nothing is
estimated to fill them. Feasibility is "no" on any proven violation, "yes"
only when complete, and "open" otherwise.

Efficiency tables:
    annex_v1    the Data Annex's pooled class throughput (the annex's own
                model; reproduces its oversubscription on every profile)
    default_v1  per engine, measured pairs only; unmeasured pairs are gaps

Usage:
    python cli/analyze_soc.py --design orin_class_reference --regime "air superiority"
    python cli/analyze_soc.py --design orin_class_reference --all --output orin.csv
    python cli/analyze_soc.py --design orin_class_reference --regime "far flight" \\
        --efficiency default_v1 --mapping greedy --verbose
    python cli/analyze_soc.py --list-profiles

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

from graphs.estimation.soc import SoCAnalysisResult, SoCAnalyzer  # noqa: E402
from graphs.hardware.sku_validators.silicon_math import SiliconMathError  # noqa: E402
from graphs.reporting.output_format import csv_writer, detect_format, write_report  # noqa: E402


def _fmt(value, spec: str = ".3g", none: str = "n/a") -> str:
    return none if value is None else format(value, spec)


def _contention(c) -> str:
    """The upper bounds' DRAM share, when the response analysis ran."""
    if not c or not c.get("requesters"):
        return ""
    share = c.get("share_gb_per_s")
    return (f"; upper bounds share it among {c['requesters']} requesters"
            + (f", {share:.1f} GB/s each" if share else "") + " (fair arbiter assumed)")


def _verdict(value: Optional[bool]) -> str:
    return {True: "yes", False: "NO", None: "open"}[value]


def _summary_lines(result: SoCAnalysisResult) -> List[str]:
    d = result.to_dict()
    s, p, m = d["summary"], d["power"], d["memory"]
    lb = " (LOWER BOUND)"
    lines = [
        f"profile: {d['profile']}" + (f"  [{d['regime']}]" if d["regime"] else ""),
        f"design:  {d['design']} at {d['node']};  efficiency {d['efficiency']};  mapping {d['mapping_source']}",
        f"complete: {'yes' if d['complete'] else 'NO'};  confidence {d['confidence_summary']['level']}",
        f"feasible: {_verdict(s['feasible'])}",
        f"oversubscription: {s['oversubscription']:.2f} s/s"
        + ("" if result.schedule.complete else lb)
        + f";  stages over: {', '.join(s['stages_over']) or 'none'}",
        f"sense-to-act chain: {s['e2e_latency_ms']:.1f} ms"
        + ("" if s["e2e_latency_complete"] else lb)
        + (f", at most {s['reactive_chain_upper_ms']:.1f} ms" if s.get("reactive_chain_upper_ms") else "")
        + f" vs deadline {s['deadline_ms']:g} ms",
        f"schedulable ({(d.get('schedulability') or {}).get('policy', '-')}): "
        f"{_verdict(s.get('schedulable'))}"
        + (f"; upper bound over deadline: {', '.join(d['schedulability']['stages_over_upper_bound'])}"
           if (d.get("schedulability") or {}).get("stages_over_upper_bound") else ""),
        f"DRAM: demand {m['dram_demand_gb_per_s']:.1f} GB/s vs sustained supply "
        f"{_fmt(m['dram_supply_gb_per_s'], '.1f')} GB/s"
        + _contention((d.get("schedulability") or {}).get("dram_contention")),
        f"power: {p['total_w']:.3g} W" + (lb if p["total_is_lower_bound"] else "")
        + f" vs budget {p['budget_w']:g} W -- within: {_verdict(p['within_budget'])}",
        f"useful TOPS/W: {_fmt(p['useful_tops_per_w'], '.3g', 'withheld (dynamic power unpriced)')}"
        + (" (UPPER BOUND)" if p["useful_tops_per_w"] is not None and p["useful_tops_per_w_is_upper_bound"] else ""),
    ]
    if d["confidence_summary"]["limited_by"]:
        lines.append("limited by:")
        lines += [f"  - {x}" for x in d["confidence_summary"]["limited_by"]]
    return lines


def _stage_rows(result: SoCAnalysisResult) -> List[dict]:
    return [
        {"stage": r["stage"], "tier": r["pipeline_tier"], "engine": r["engine"] or "-",
         "rate_hz": _fmt(r["rate_hz"], "g"), "t_service_ms": _fmt(r["t_service_ms"], ".3g", "-"),
         "ratio": _fmt(r["ratio"], ".3g", "-"), "bound": r["bound"] or "-",
         "gap": r["gap"] or ""}
        for r in result.to_dict()["stages"]
    ]


def _table(rows: List[dict]) -> str:
    if not rows:
        return "(none)\n"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    out = ["  ".join(k.ljust(widths[k]) for k in keys),
           "  ".join("-" * widths[k] for k in keys)]
    out += ["  ".join(str(r[k]).ljust(widths[k]) for k in keys) for r in rows]
    return "\n".join(out) + "\n"


def _md_table(rows: List[dict]) -> List[str]:
    if not rows:
        return []
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    return lines + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]


def _csv_row(result: SoCAnalysisResult) -> dict:
    d = result.to_dict()
    s, p, m = d["summary"], d["power"], d["memory"]
    return {
        "design": d["design"], "node": d["node"], "profile": d["profile"],
        "regime": d["regime"] or "", "efficiency": d["efficiency"], "mapping": d["mapping"],
        "complete": d["complete"], "confidence": d["confidence_summary"]["level"],
        "feasible": _verdict(s["feasible"]), "oversubscription": s["oversubscription"],
        "stages_over": ";".join(s["stages_over"]), "unpriced_stages": len(result.schedule.gaps),
        "e2e_latency_ms": s["e2e_latency_ms"], "deadline_ms": s["deadline_ms"],
        "dram_demand_gb_per_s": m["dram_demand_gb_per_s"], "dram_supply_gb_per_s": m["dram_supply_gb_per_s"],
        "power_w": p["total_w"], "power_is_lower_bound": p["total_is_lower_bound"],
        "budget_w": p["budget_w"], "useful_tops_per_w": p["useful_tops_per_w"],
    }


def render(results: List[SoCAnalysisResult], fmt: str, verbose: bool) -> str:
    if fmt == "json":
        payload = [r.to_dict() for r in results]
        return json.dumps(payload[0] if len(payload) == 1 else payload, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        rows = [_csv_row(r) for r in results]
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pipeline mission profiles on an SoC design (graphs#269 Phase 3)."
    )
    parser.add_argument("--design", help="Design id from soc_designs/designs/")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--profile", action="append",
                       help="Profile id, regime or mission name (repeatable)")
    which.add_argument("--regime", action="append", help="Named regime, e.g. 'far flight' (repeatable)")
    which.add_argument("--all", action="store_true", help="Every profile in the workload")
    parser.add_argument("--list-profiles", action="store_true", help="List profiles and exit")
    parser.add_argument("--node", help="Process node (default: the design's own)")
    parser.add_argument("--efficiency", default="annex_v1",
                        help="Efficiency table: annex_v1 (default) or default_v1")
    parser.add_argument("--mapping", default="auto",
                        help="auto (shipped explicit mapping, else greedy), explicit, greedy, "
                             "ilp (optimal bottleneck, needs scipy), "
                             "or a mapping .yaml file")
    parser.add_argument("--policy", choices=["rm", "edf"], default="rm",
                        help="Scheduling policy for response-time analysis (default rm)")
    parser.add_argument("--gate-idle", action="store_true",
                        help="Power-gate engines the schedule leaves idle (zero their leakage)")
    parser.add_argument("--sustained-fraction", type=float, default=None,
                        help="Sustained fraction of peak DRAM bandwidth (default 0.65)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Add the per-stage table")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    try:
        analyzer = SoCAnalyzer()
    except (ValueError, yaml.YAMLError) as exc:  # pydantic's ValidationError is a ValueError
        print(f"error: a catalog failed to load: {exc}", file=sys.stderr)
        return 2

    if args.list_profiles:
        rows = [{"profile": p.id, "regime": p.regime or "", "budget_w": p.power_budget_w,
                 "deadline_ms": p.deadline_ms} for p in analyzer.workload.profiles]
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
            payload = "\n".join([f"## Profiles ({len(rows)})", ""] + _md_table(rows)) + "\n"
        else:
            payload = _table(rows)
        write_report(payload, args.output)
        return 0
    if not args.design:
        parser.error("--design is required unless --list-profiles")

    names = args.profile or args.regime or ([] if args.all else ["air superiority"])
    kwargs = dict(node=args.node, efficiency=args.efficiency, mapping=args.mapping,
                  gate_idle=args.gate_idle, policy=args.policy)
    if args.sustained_fraction is not None:
        kwargs["sustained_fraction"] = args.sustained_fraction
    try:
        results = analyzer.analyze_profiles(args.design, names or None, **kwargs)
    except (KeyError, ValueError, SiliconMathError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_report(render(results, detect_format(args.output), args.verbose), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
