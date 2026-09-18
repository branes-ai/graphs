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
    python cli/sweep_soc.py --study orin_node_scaling --pareto area,power --union
    python cli/sweep_soc.py --study orin_node_scaling --plot front.png   # needs matplotlib

Pareto (--pareto) and the union of regimes (--union) are bound-aware: a
lower bound never places a point on a front or makes a design the minimum.
Points that cannot be separated either way are reported as undecided.

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
from graphs.estimation.soc.pareto import classify_front, union_of_regimes  # noqa: E402
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


def _union_lines(report) -> List[str]:
    best = report.minimum
    lines = [f"union of regimes over {len(report.profiles)} profile(s): "
             + (f"minimum is {best.variant[0]} [{best.variant[1]}] at {best.variant[2]}, "
                f"{best.area.value:.1f} mm^2" if best else "no variant is proven feasible for all")]
    for v in report.verdicts:
        verdict = {True: "feasible for all", False: "infeasible", None: "undecided"}[v.feasible]
        why = (f" -- fails {', '.join(v.failing_profiles)}" if v.failing_profiles
               else f" -- open in {', '.join(v.open_profiles)}" if v.open_profiles else "")
        if v.missing_profiles:
            why += f" -- no point for {', '.join(v.missing_profiles)}"
        area = f"{v.area.value:.1f}{'' if v.area.exact else ' ' + LB}"
        lines.append(f"  {v.variant[0]} [{v.variant[1]}] {v.variant[2]} {v.variant[3]}"
                     f"{' gated' if v.variant[4] else ''}: {verdict}, {area} mm^2{why}")
    if best and report.could_be_smaller:
        lines.append(f"  {len(report.could_be_smaller)} undecided variant(s) might be smaller")
    return lines


def _plot(rows: List[SweepRow], statuses: List[str], metrics: List[str], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("--plot needs matplotlib (the 'viz' extra)") from exc
    from graphs.estimation.soc.pareto import METRICS

    (xk, _), (yk, _) = METRICS[metrics[0]], METRICS[metrics[1]]
    fig, ax = plt.subplots(figsize=(7, 5))
    style = {"front": ("o", "tab:green"), "dominated": ("x", "tab:gray"), "undecided": ("^", "tab:orange")}
    for status, (marker, color) in style.items():
        pts = [r.to_row() for r, s in zip(rows, statuses) if s == status]
        if pts:
            ax.scatter([p[xk] for p in pts], [p[yk] for p in pts], marker=marker, color=color,
                       label=status, facecolors="none" if marker == "^" else None)
    ax.set_xlabel(f"{metrics[0]} ({xk})")
    ax.set_ylabel(f"{metrics[1]} ({yk})")
    ax.set_title("Bound-aware front: triangles are lower bounds (true value up and right)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render(study: Study, rows: List[SweepRow], fmt: str,
           pareto: Optional[List[str]] = None, union: bool = False) -> str:
    flat = [r.to_row() for r in rows]
    if pareto:
        for row, (_, status) in zip(flat, classify_front(rows, pareto)):
            row["pareto"] = status
    report = union_of_regimes(rows) if union else None
    if fmt == "json":
        payload = {"study": study.model_dump(), "points": flat}
        if pareto:
            payload["pareto_metrics"] = pareto
        if report:
            payload["union_of_regimes"] = report.to_dict()
        return json.dumps(payload, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv_writer(buf, fieldnames=list(flat[0]) if flat else ["design"])
        writer.writeheader()
        writer.writerows(flat)
        return buf.getvalue()
    shown = [_display(r) | ({"pareto": r["pareto"]} if pareto else {}) for r in flat]
    head = (f"study: {study.id} -- {study.name}\n"
            f"points: {len(flat)};  {LB} = lower bound (an input has gaps)\n")
    if fmt == "md":
        keys = list(shown[0]) if shown else []
        lines = [f"## {study.id}", "", head.replace("\n", "  \n"),
                 "| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
        lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in shown]
        if report:
            lines += ["", "### Union of regimes", ""] + [f"- {x.strip()}" for x in _union_lines(report)]
        return "\n".join(lines) + "\n"
    text = head + "\n" + _table(shown)
    if report:
        text += "\n" + "\n".join(_union_lines(report)) + "\n"
    return text


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
    parser.add_argument("--pareto", help="Classify points on these minimized metrics, e.g. area,power "
                                         "(area, power, oversubscription)")
    parser.add_argument("--union", action="store_true",
                        help="Report the smallest variant proven feasible in every profile")
    parser.add_argument("--plot", help="Write a front plot (PNG) of the first two --pareto metrics")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)
    pareto = _csv_list(args.pareto)
    if args.plot and (not pareto or len(pareto) < 2):
        parser.error("--plot needs --pareto with two metrics")
    if args.union and detect_format(args.output) == "csv":
        parser.error("--union is a report, not a row: use a .json, .md or text output "
                     "(CSV carries the points only)")

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
        payload = render(study, rows, detect_format(args.output), pareto, args.union)
        if args.plot:
            _plot(rows, [s for _, s in classify_front(rows, pareto)], pareto, args.plot)
    except (KeyError, ValueError, FileNotFoundError, SiliconMathError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_report(payload, args.output)
    if args.plot:
        print(f"wrote {args.plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
