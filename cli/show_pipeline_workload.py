#!/usr/bin/env python
"""
Autonomy Pipeline Workload CLI

Shows the Branes.ai 7-tier autonomy workload: what each costed stage demands,
what each mission profile asks of a machine, and where the demand exceeds one
second of compute per second of mission.

The data is generated from the reference model in ``docs/workload-model/``
(Autonomy Workload Data Annex, 2026-09-17) by
``tools/generate_autonomy_workload.py``. Two of the eighteen profiles are the
operating regimes of the companion argument document (2026-09-13) and are
shown with its published figures beside the derived ones.

Nothing here involves hardware. Occupancy is against the annex's stated
per-class effective throughputs (Class A 2,000 GOP/s, B 300, C 15), which are
assumptions from a measurement corpus, not properties of any particular die.

Usage:
    python cli/show_pipeline_workload.py                       # the 18 profiles
    python cli/show_pipeline_workload.py --regimes             # just the two regimes
    python cli/show_pipeline_workload.py --stages              # the costed stages
    python cli/show_pipeline_workload.py --profile "air superiority"
    python cli/show_pipeline_workload.py --throughput 2000,900,45   # sensitivity
    python cli/show_pipeline_workload.py --output profiles.csv

Exit codes:
    0 = report produced
    2 = argument error (unknown profile, bad --throughput)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphs.core.pipeline_workload import (  # noqa: E402
    EffectiveThroughput,
    MissionSummary,
    PipelineWorkload,
    load_autonomy_workload,
)
from graphs.reporting.output_format import csv_writer, detect_format, write_report# noqa: E402


def _profile_rows(summaries) -> list[dict]:
    rows = []
    for s in summaries:
        shares = s.class_shares()
        rows.append({
            "form_factor": s.profile.form_factor,
            "mission": s.profile.name,
            "regime": s.profile.regime or "",
            "tops": round(s.tops, 3),
            "gb_per_s": round(s.gb_per_s, 1),
            "class_a": round(shares["A"], 3),
            "class_b": round(shares["B"], 3),
            "class_c": round(shares["C"], 3),
            "oversubscription": round(s.oversubscription, 2),
            "stages_over": len(s.stages_over()),
            "chain_ms": round(s.reactive_chain_ms, 1),
            "deadline_ms": s.profile.deadline_ms,
            "budget_w": s.profile.power_budget_w,
        })
    return rows


def _stage_rows(workload: PipelineWorkload) -> list[dict]:
    return [{
        "key": s.key,
        "tier": s.pipeline_tier,
        "stage": s.name,
        "unit": s.unit,
        "ops_per_call": s.ops_per_call,
        "bytes_per_call": s.bytes_per_call,
        "op_per_byte": round(s.op_per_byte, 2),
        "class_a": s.class_split[0],
        "class_b": s.class_split[1],
        "class_c": s.class_split[2],
        "floor": s.precision_floor,
    } for s in workload.stages.values()]


def _demand_rows(workload: PipelineWorkload, summary: MissionSummary) -> list[dict]:
    return [{
        "key": d.stage.key,
        "tier": d.stage.pipeline_tier,
        "stage": d.stage.name,
        "rate_hz": round(d.rate_hz, 3),
        "gops": round(d.ops_per_s / 1e9, 3),
        "gb_per_s": round(d.bytes_per_s / 1e9, 3),
        "floor": d.stage.precision_floor,
        "service_ms": round(d.service_time_s * 1e3, 4),
        "occupancy": round(d.occupancy, 4),
    } for d in sorted(summary.demands, key=lambda d: -d.occupancy)]


def _render_text(title: str, rows: list[dict], notes: list[str]) -> str:
    if not rows:
        return f"{title}\n(nothing to show)\n"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    out = io.StringIO()
    out.write(f"=== {title} ===\n\n")
    out.write("  ".join(k.ljust(widths[k]) for k in keys) + "\n")
    out.write("  ".join("-" * widths[k] for k in keys) + "\n")
    for r in rows:
        out.write("  ".join(str(r[k]).ljust(widths[k]) for k in keys) + "\n")
    for note in notes:
        out.write(f"\n{note}\n")
    return out.getvalue()


def _render_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv_writer(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _render_md(title: str, rows: list[dict], notes: list[str]) -> str:
    if not rows:
        return "\n".join([f"## {title}", "", "(nothing to show)"] + notes) + "\n"
    keys = list(rows[0])
    lines = [f"## {title}", "", "| " + " | ".join(keys) + " |",
             "|" + "|".join("---" for _ in keys) + "|"]
    lines += ["| " + " | ".join(str(r[k]) for r in [r] for k in keys) + " |" for r in rows]
    lines += [""] + notes
    return "\n".join(lines) + "\n"


def _regime_notes(workload: PipelineWorkload) -> list[str]:
    notes = []
    for profile in workload.regimes():
        got = workload.summary(profile)
        pub = profile.published
        if not pub.get("tops"):
            # A labelled regime without published figures has nothing to be
            # checked against; say so rather than dividing by zero.
            notes.append(f"{profile.regime}: derived {got.tops:.2f} TOP/s; no published figures.")
            continue
        notes.append(
            f"{profile.regime}: derived {got.tops:.2f} TOP/s / {got.gb_per_s:.0f} GB/s / "
            f"{got.oversubscription:.1f} s per s; published "
            f"{pub['tops']:.2f} / {pub['dram_gb_per_s']:.0f} / {pub['oversubscription']:.1f} "
            f"({100 * (got.tops / pub['tops'] - 1):+.1f}% on arithmetic). "
            f"Independent derivations, so the agreement is a check."
        )
    return notes


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show the autonomy pipeline workload: stages, missions and demand."
    )
    parser.add_argument("--workload", type=Path, default=None,
                        help="Workload YAML (default: the generated catalog copy).")
    parser.add_argument("--profile", metavar="NAME",
                        help="Per-stage demand for one mission, regime name or id.")
    parser.add_argument("--stages", action="store_true",
                        help="The costed stage catalog instead of the missions.")
    parser.add_argument("--regimes", action="store_true",
                        help="Only the profiles that are named operating regimes.")
    parser.add_argument("--throughput", metavar="A,B,C",
                        help="Override the per-class effective throughput in GOP/s "
                             "(default 2000,300,15) to test the annex's sensitivity.")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    workload = load_autonomy_workload(args.workload)
    if args.throughput:
        try:
            a, b, c = (float(x) for x in args.throughput.split(","))
        except ValueError:
            parser.error("--throughput takes three comma-separated GOP/s figures, e.g. 2000,900,45")
        workload = PipelineWorkload(
            stages=workload.stages, profiles=workload.profiles,
            throughput=EffectiveThroughput(a=a, b=b, c=c),
            provenance=workload.provenance, version=workload.version,
        )

    notes: list[str] = []
    if args.stages:
        title = f"Costed stages ({len(workload.stages)})"
        rows = _stage_rows(workload)
        below = sum(1 for r in rows if r["op_per_byte"] < 15)
        notes.append(f"{below} of {len(rows)} stages sit below 15 op per byte, where the "
                     f"memory system sets the rate however wide the multiplier array is.")
    elif args.profile:
        try:
            profile = workload.profile(args.profile)
        except KeyError as exc:
            parser.error(str(exc))
        summary = workload.summary(profile)
        title = f"{profile.form_factor} / {profile.name}"
        rows = _demand_rows(workload, summary)
        notes.append(profile.note)
        notes.append(
            f"{summary.tops:.2f} TOP/s, {summary.gb_per_s:.0f} GB/s, "
            f"{summary.oversubscription:.1f} s of compute per s of mission, "
            f"{len(summary.stages_over())} stage(s) over their own rate, "
            f"reactive chain {summary.reactive_chain_ms:.0f} ms against a "
            f"{profile.deadline_ms:.0f} ms deadline."
        )
    else:
        summaries = workload.summaries()
        if args.regimes:
            summaries = tuple(s for s in summaries if s.profile.regime)
            title = f"Operating regimes ({len(summaries)} of {len(workload.profiles)} profiles)"
        else:
            title = f"Mission profiles ({len(summaries)})"
        rows = _profile_rows(summaries)
        notes.extend(_regime_notes(workload))

    if workload.provenance:
        notes.append(
            f"Source: {workload.provenance.document} ({workload.provenance.dated}); "
            f"effective throughput A/B/C = {workload.throughput.a:.0f}/"
            f"{workload.throughput.b:.0f}/{workload.throughput.c:.0f} GOP/s."
        )

    fmt = detect_format(args.output)
    if fmt == "json":
        payload = json.dumps({"title": title, "rows": rows, "notes": notes}, indent=2)
    elif fmt == "csv":
        payload = _render_csv(rows)
    elif fmt == "md":
        payload = _render_md(title, rows, notes)
    else:
        payload = _render_text(title, rows, [n for n in notes if n])

    write_report(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
