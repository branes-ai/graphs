#!/usr/bin/env python
"""
KPU Die vs Autonomy Workload CLI

Compares KPU dies on the Branes.ai autonomy workload: for each mission
profile, what each die can run, what it cannot, and how many seconds of die
time one second of mission demands.

This is graphs#268's deferred item -- the uniform T64 against the
heterogeneous kpu_h64_auto1 -- on the axis the workload model supports: all
eighteen mission profiles, with the two that are named operating regimes
(far flight, air superiority) labelled.

Three verdicts, in precedence order, because they are different failures:

    precision   a stage's numeric floor has no engine on this die. Absent,
                not slow: an FP32 factor graph does not run on an INT8 fabric
    contract    a function core would absorb the stage, but the mission's
                configuration is past what the core is rated for
    bandwidth   DRAM traffic exceeds the bus
    compute     more than one second of die time per second of mission

Occupancy is against an achieved-to-peak ratio, because peak silicon rates
flatter every die: the argument document sizes on 2% (realistic mixed
pipeline) and 5% (optimistic), against a measured median of 0.31%. The same
ratio is applied to both dies, which is the comparison's main assumption --
see the caveat printed with every report.

Usage:
    python cli/analyze_dies_on_workload.py
    python cli/analyze_dies_on_workload.py --regimes
    python cli/analyze_dies_on_workload.py --profile "air superiority"
    python cli/analyze_dies_on_workload.py --achieved-to-peak 0.05
    python cli/analyze_dies_on_workload.py --dies kpu_t64_32x32_lp5x4_7nm_tsmc_hpc
    python cli/analyze_dies_on_workload.py --output comparison.md

Exit codes:
    0 = report produced
    2 = argument error (unknown die or profile)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_schemas import load_compute_products, load_process_nodes  # noqa: E402

from graphs.core.pipeline_workload import load_autonomy_workload  # noqa: E402
from graphs.hardware.workload_fit import (  # noqa: E402
    REALISTIC_ACHIEVED_TO_PEAK,
    DieFit,
    fit_profile,
)
from graphs.reporting.output_format import csv_writer, detect_format, write_report# noqa: E402

#: The two 64-site dies graphs#268 set out to compare, at 16 nm.
DEFAULT_DIES = (
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp",
)

CAVEAT = (
    "Both dies are charged the same achieved-to-peak ratio, measured on "
    "general-purpose embedded silicon. A KPU's argument is that it achieves a "
    "higher fraction of peak on this work; that claim is not tested here, and "
    "it would move every compute verdict below."
)


def _short(die_id: str) -> str:
    return die_id.replace("kpu_", "").replace("_lp5x4", "").replace("_32x32", "")


def _summary_rows(workload, fits_by_die, profiles) -> List[dict]:
    rows = []
    for profile in profiles:
        baseline = workload.summary(profile)
        row = {
            "form_factor": profile.form_factor,
            "mission": profile.name,
            "regime": profile.regime or "",
            "demand_tops": round(baseline.tops, 2),
            "baseline_s_per_s": round(baseline.oversubscription, 1),
        }
        for die_id, fits in fits_by_die.items():
            fit = fits[profile.id]
            name = _short(die_id)
            row[f"{name} s/s"] = round(fit.oversubscription, 1)
            row[f"{name} binding"] = fit.binding_constraint
            row[f"{name} unrunnable"] = len(fit.unrunnable_stages)
        rows.append(row)
    return rows


def _stage_rows(fits: List[DieFit]) -> List[dict]:
    rows = []
    keys = [s.key for s in fits[0].stages]
    for key in keys:
        row = {"stage": key, "tier": "", "floor": "", "gops": 0.0}
        for fit in fits:
            stage_fit = next(s for s in fit.stages if s.key == key)
            demand = stage_fit.demand
            row["tier"] = demand.stage.pipeline_tier
            row["floor"] = demand.stage.precision_floor
            row["gops"] = round(demand.ops_per_s / 1e9, 2)
            name = _short(fit.die_id)
            if not stage_fit.runnable:
                verdict = f"no class {'/'.join(stage_fit.missing_classes)}"
            elif stage_fit.violations:
                verdict = "over contract: " + "; ".join(str(v) for v in stage_fit.violations)
            elif stage_fit.execution == "absorbed":
                verdict = f"{stage_fit.engines[0]} core {stage_fit.occupancy:.3f}"
            else:
                verdict = f"{stage_fit.occupancy:.3f}"
            row[name] = verdict
        rows.append(row)
    return rows


def _render_text(title: str, rows: List[dict], notes: List[str]) -> str:
    if not rows:
        return f"{title}\n(nothing to show)\n"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    out = io.StringIO()
    out.write(f"=== {title} ===\n\n")
    out.write("  ".join(k.ljust(widths[k]) for k in keys) + "\n")
    out.write("  ".join("-" * widths[k] for k in keys) + "\n")
    for r in rows:
        out.write("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys) + "\n")
    for note in notes:
        out.write(f"\n{note}\n")
    return out.getvalue()


def _render_md(title: str, rows: List[dict], notes: List[str]) -> str:
    if not rows:
        return "\n".join([f"## {title}", "", "(nothing to show)"] + notes) + "\n"
    keys = list(rows[0])
    lines = [f"## {title}", "", "| " + " | ".join(keys) + " |",
             "|" + "|".join("---" for _ in keys) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    lines += [""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def _render_csv(rows: List[dict]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv_writer(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare KPU dies on the autonomy workload, mission by mission."
    )
    parser.add_argument("--dies", help="Comma-separated SKU ids (default: the two 16 nm 64-site dies).")
    parser.add_argument("--profile", metavar="NAME", help="Per-stage detail for one mission or regime.")
    parser.add_argument("--regimes", action="store_true", help="Only the named operating regimes.")
    parser.add_argument("--achieved-to-peak", type=float, default=REALISTIC_ACHIEVED_TO_PEAK,
                        help="Fraction of peak the dies sustain (default 0.02; 1.0 for peak).")
    parser.add_argument("--workload", type=Path, default=None)
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    workload = load_autonomy_workload(args.workload)
    products, nodes = load_compute_products(), load_process_nodes()
    die_ids = [d.strip() for d in args.dies.split(",")] if args.dies else list(DEFAULT_DIES)
    for die_id in die_ids:
        if die_id not in products:
            parser.error(f"unknown SKU {die_id!r}")

    profiles = list(workload.profiles)
    if args.regimes:
        profiles = [p for p in profiles if p.regime]
    if args.profile:
        try:
            profiles = [workload.profile(args.profile)]
        except KeyError as exc:
            parser.error(str(exc))

    fits_by_die = {}
    for die_id in die_ids:
        cp = products[die_id]
        node = nodes[cp.dies[0].process_node_id]
        fits_by_die[die_id] = {
            p.id: fit_profile(workload, p, cp, node, nodes, args.achieved_to_peak)
            for p in profiles
        }

    notes = [CAVEAT]
    if args.profile:
        profile = profiles[0]
        fits = [fits_by_die[d][profile.id] for d in die_ids]
        title = (f"{profile.form_factor} / {profile.name}"
                 + (f" ({profile.regime})" if profile.regime else ""))
        rows = _stage_rows(fits)
        for fit in fits:
            notes.append(
                f"{_short(fit.die_id)}: {fit.oversubscription:.1f} s per s over the stages it "
                f"can run, {len(fit.unrunnable_stages)} stage(s) it cannot, DRAM "
                f"{fit.dram_demand_gb_per_s:.0f} of {fit.dram_available_gb_per_s:.0f} GB/s, "
                f"binding constraint: {fit.binding_constraint}."
            )
    else:
        title = (f"KPU dies on the autonomy workload "
                 f"({len(profiles)} profiles, achieved-to-peak {args.achieved_to_peak:.0%})")
        rows = _summary_rows(workload, fits_by_die, profiles)
        for die_id in die_ids:
            fits = fits_by_die[die_id]
            ok = sum(1 for f in fits.values() if f.feasible)
            notes.append(f"{_short(die_id)}: feasible on {ok} of {len(fits)} profiles.")
        notes.append(
            "baseline_s_per_s is the annex's own figure for general-purpose silicon "
            "at 2,000 / 300 / 15 GOP/s per class, for reference."
        )

    fmt = detect_format(args.output)
    if fmt == "json":
        payload = json.dumps({
            "title": title, "rows": rows, "notes": notes,
            "fits": [f.to_dict() for fits in fits_by_die.values() for f in fits.values()],
        }, indent=2)
    elif fmt == "csv":
        payload = _render_csv(rows)
    elif fmt == "md":
        payload = _render_md(title, rows, notes)
    else:
        payload = _render_text(title, rows, notes)

    write_report(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
