#!/usr/bin/env python
"""
Cooling-Solution Database Inspector

Lists every cooling-solution entry in the embodied-schemas catalog.
Cooling is a peer of ProcessNode, not a sub-field -- the same node can
ship in fanless edge modules and liquid-cooled datacenter cards.

Output columns:
    id, mechanism, max W/mm^2 (the thermal-hotspot validator's ceiling),
    max total W (whole-package envelope), Tj_max (feeds the EM validator),
    weight, cost, confidence.

Usage:
    python cli/list_cooling_solutions.py
    python cli/list_cooling_solutions.py --mechanism active_fan
    python cli/list_cooling_solutions.py --output cooling.csv
    python cli/list_cooling_solutions.py --output cooling.md
    python cli/list_cooling_solutions.py --output cooling.json
"""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

from embodied_schemas import load_cooling_solutions
from embodied_schemas.cooling_solution import CoolingSolutionEntry


@dataclass
class CoolingRow:
    id: str
    mechanism: str
    max_w_per_mm2: float
    max_total_w: float
    ambient_c_max: float
    junction_c_max: float
    weight_g: Optional[float]
    cost_usd: Optional[float]
    confidence: str
    source: str


def _build_row(e: CoolingSolutionEntry) -> CoolingRow:
    return CoolingRow(
        id=e.id,
        mechanism=e.cooling_mechanism.value,
        max_w_per_mm2=e.max_power_density_w_per_mm2,
        max_total_w=e.max_total_w,
        ambient_c_max=e.ambient_c_max,
        junction_c_max=e.junction_c_max,
        weight_g=e.weight_g,
        cost_usd=e.cost_usd,
        confidence=e.confidence.value,
        source=e.source,
    )


def _na(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:g}"


def _render_text(rows: List[CoolingRow]) -> str:
    if not rows:
        return "(no entries match)\n"
    header = (
        f"{'id':28s} {'mechanism':30s} {'W/mm^2':>8s} {'max W':>7s} "
        f"{'Tamb':>5s} {'Tj_max':>7s} {'wt(g)':>7s} {'$':>6s} {'conf':>11s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.id:28s} {r.mechanism:30s} {r.max_w_per_mm2:>8.2f} "
            f"{r.max_total_w:>7.0f} {r.ambient_c_max:>5.0f} "
            f"{r.junction_c_max:>7.0f} {_na(r.weight_g):>7s} "
            f"{_na(r.cost_usd):>6s} {r.confidence:>11s}"
        )
    lines.append("")
    lines.append(f"{len(rows)} cooling solution(s)")
    return "\n".join(lines) + "\n"


def _render_json(rows: List[CoolingRow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2) + "\n"


def _render_csv(rows: List[CoolingRow]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(asdict(r))
    return buf.getvalue()


def _render_md(rows: List[CoolingRow]) -> str:
    if not rows:
        return "_no entries match_\n"
    lines = [
        "| id | mechanism | max W/mm^2 | max W | Tamb | Tj_max | weight (g) | cost (USD) | confidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.id}` | {r.mechanism} | {r.max_w_per_mm2:.2f} | "
            f"{r.max_total_w:.0f} | {r.ambient_c_max:.0f} | "
            f"{r.junction_c_max:.0f} | {_na(r.weight_g)} | "
            f"{_na(r.cost_usd)} | {r.confidence} |"
        )
    lines.append("")
    lines.append(f"_{len(rows)} cooling solution(s)_")
    return "\n".join(lines) + "\n"


_RENDERERS = {
    "text": _render_text,
    "json": _render_json,
    "csv": _render_csv,
    "md": _render_md,
}


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "csv": "csv", "md": "md", "markdown": "md", "txt": "text"}.get(
        ext, "text"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List cooling-solution entries from the embodied-schemas catalog."
    )
    parser.add_argument("--mechanism", help="Filter by cooling mechanism")
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.csv/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        sols = load_cooling_solutions()
    except Exception as exc:
        print(f"error: failed to load cooling solutions: {exc}", file=sys.stderr)
        return 1

    rows = [_build_row(s) for s in sols.values()]
    if args.mechanism:
        rows = [r for r in rows if r.mechanism == args.mechanism.lower()]
    rows.sort(key=lambda r: r.max_w_per_mm2)

    fmt = _detect_format(args.output)
    rendered = _RENDERERS[fmt](rows)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
