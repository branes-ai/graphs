#!/usr/bin/env python
"""
SoC Study Sweep

Runs a study -- base designs x overrides x nodes x profiles x efficiency
tables x idle gating -- through the SoC L0 analyzer and tabulates one row
per point (graphs#269 Phase 4).

This bypasses UnifiedAnalyzer deliberately: there is no PyTorch model, only a
costed pipeline and silicon to price -- the same exception
analyze_operator_roofline.py documents.

Every figure travels with its bound flag. A die area, power or
oversubscription computed over an input with gaps is a LOWER BOUND and is
marked as one; nothing is estimated to fill a gap, so a column can mix exact
values and bounds and says which is which.

Studies live in soc_designs/studies/<id>.yaml. An ad-hoc study can be given
on the command line instead.

Usage:
    python cli/sweep_soc.py --study orin_node_scaling
    python cli/sweep_soc.py --study orin_node_scaling --output sweep.csv
    python cli/sweep_soc.py --designs orin_class_reference \\
        --nodes samsung_8lpp,tsmc_n7,tsmc_n5 --profiles all --output sweep.json

Exit codes:
    0 = sweep produced
    2 = unknown study, design, node, profile, table, or a catalog that fails to load
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

from graphs.estimation.soc import SoCAnalyzer  # noqa: E402
from graphs.estimation.soc.study import Study, SweepRow, load_study, run_study  # noqa: E402
from graphs.hardware.sku_validators.silicon_math import SiliconMathError  # noqa: E402
from graphs.reporting.output_format import csv_writer, detect_format, write_report  # noqa: E402

LB = "(LB)"


def _cell(value, bound: bool = False, spec: str = ".3g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    text = format(value, spec) if isinstance(value, float) else str(value)
    return f"{text} {LB}" if bound else text


def _display(row: dict) -> dict:
    return {
        "design": row["design"], "variant": row["variant"], "node": row["node"],
        "profile": row["regime"] or row["profile"], "eff": row["efficiency"],
        "feasible": {True: "yes", False: "NO", None: "open"}[row["feasible"]],
        "die_mm2": _cell(row["die_area_mm2"], row["die_area_is_lower_bound"], ".1f"),
        "oversub": _cell(row["oversubscription"], row["oversubscription_is_lower_bound"], ".2f"),
        "over": row["stages_over"], "gaps": row["unpriced_stages"],
        "dram_util": _cell(row["dram_utilization"], spec=".2f"),
        "power_w": _cell(row["power_w"], row["power_is_lower_bound"]),
        "tops_w": _cell(row["useful_tops_per_w"]),
    }


def _table(rows: List[dict]) -> str:
    if not rows:
        return "(no points)\n"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    out = ["  ".join(k.ljust(widths[k]) for k in keys), "  ".join("-" * widths[k] for k in keys)]
    out += ["  ".join(str(r[k]).ljust(widths[k]) for k in keys) for r in rows]
    return "\n".join(out) + "\n"


def render(study: Study, rows: List[SweepRow], fmt: str) -> str:
    flat = [r.to_row() for r in rows]
    if fmt == "json":
        return json.dumps({"study": study.model_dump(), "points": flat}, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv_writer(buf, fieldnames=list(flat[0]) if flat else ["design"])
        writer.writeheader()
        writer.writerows(flat)
        return buf.getvalue()
    shown = [_display(r) for r in flat]
    head = (f"study: {study.id} -- {study.name}\n"
            f"points: {len(flat)};  {LB} = lower bound (an input has gaps)\n")
    if fmt == "md":
        keys = list(shown[0]) if shown else []
        lines = [f"## {study.id}", "", head.replace("\n", "  \n"),
                 "| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
        lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in shown]
        return "\n".join(lines) + "\n"
    return head + "\n" + _table(shown)


def _csv_list(text: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep SoC designs through the L0 analyzer (graphs#269 Phase 4).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--study", help="Study id in soc_designs/studies/, or a study .yaml path")
    src.add_argument("--designs", help="Ad-hoc study: comma-separated design ids")
    parser.add_argument("--nodes", help="Ad-hoc: comma-separated nodes (default: each design's own)")
    parser.add_argument("--profiles", default="regimes",
                        help="Ad-hoc: all, regimes (default), or comma-separated profiles")
    parser.add_argument("--efficiency", default="annex_v1", help="Ad-hoc: comma-separated tables")
    parser.add_argument("--gate-idle", choices=["off", "on", "both"], default="off",
                        help="Ad-hoc: idle-engine gating")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    try:
        analyzer = SoCAnalyzer()
        if args.study:
            study = load_study(args.study)
        else:
            profiles = args.profiles if args.profiles in ("all", "regimes") else _csv_list(args.profiles)
            study = Study(
                id="adhoc", name="ad-hoc sweep", workload=analyzer.workload.version,
                designs=_csv_list(args.designs), nodes=_csv_list(args.nodes) or [None],
                profiles=profiles, efficiency=_csv_list(args.efficiency),
                gate_idle={"off": [False], "on": [True], "both": [False, True]}[args.gate_idle],
            )
        rows = run_study(study, analyzer)
    except (KeyError, ValueError, FileNotFoundError, SiliconMathError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_report(render(study, rows, detect_format(args.output)), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
