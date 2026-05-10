#!/usr/bin/env python
"""
Process-Node Database Inspector

Lists every process-node entry in the embodied-schemas catalog. Pair with
``show_process_node.py <id>`` for the full per-library breakdown of a
single node.

Output columns:
    id, foundry, node, topology, body_bias, density envelope (min/max
    Mtx/mm^2 across libraries), library count, source confidence.

Each ProcessNodeEntry can come from public estimates (``confidence:
theoretical``) or a PDK ingest (``confidence: calibrated``). The table
shows both kinds side by side -- the confidence column is how a SKU
author tells which figures to trust.

Usage:
    python cli/list_process_nodes.py
    python cli/list_process_nodes.py --foundry tsmc
    python cli/list_process_nodes.py --topology fd_soi
    python cli/list_process_nodes.py --sort density
    python cli/list_process_nodes.py --output nodes.csv
    python cli/list_process_nodes.py --output nodes.md
    python cli/list_process_nodes.py --output nodes.json
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

from embodied_schemas import load_process_nodes
from embodied_schemas.process_node import ProcessNodeEntry


@dataclass
class NodeRow:
    id: str
    foundry: str
    node_name: str
    node_nm: int
    topology: str
    body_bias: bool
    nominal_vdd_v: float
    density_min: float
    density_max: float
    library_count: int
    confidence: str
    source: str


def _build_row(entry: ProcessNodeEntry) -> NodeRow:
    lo, hi = entry.density_envelope()
    return NodeRow(
        id=entry.id,
        foundry=entry.foundry.value,
        node_name=entry.node_name,
        node_nm=entry.node_nm,
        topology=entry.transistor_topology.value,
        body_bias=entry.body_bias_supported,
        nominal_vdd_v=entry.nominal_vdd_v,
        density_min=lo,
        density_max=hi,
        library_count=len(entry.densities),
        confidence=entry.confidence.value,
        source=entry.source,
    )


_SORT_KEYS = {
    "id": lambda r: r.id,
    "foundry": lambda r: (r.foundry, r.node_nm),
    "node": lambda r: r.node_nm,
    "density": lambda r: -r.density_max,  # highest density first
    "topology": lambda r: r.topology,
}


def _filter_rows(
    rows: List[NodeRow],
    foundry: Optional[str],
    topology: Optional[str],
) -> List[NodeRow]:
    out = rows
    if foundry:
        out = [r for r in out if r.foundry == foundry.lower()]
    if topology:
        out = [r for r in out if r.topology == topology.lower()]
    return out


def _render_text(rows: List[NodeRow]) -> str:
    if not rows:
        return "(no entries match)\n"
    header = (
        f"{'id':30s} {'foundry':16s} {'node':6s} {'nm':>3s} "
        f"{'topology':12s} {'BB':>3s} {'Vdd':>5s} "
        f"{'min Mtx/mm^2':>13s} {'max Mtx/mm^2':>13s} "
        f"{'libs':>5s} {'conf':>11s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.id:30s} {r.foundry:16s} {r.node_name:6s} {r.node_nm:>3d} "
            f"{r.topology:12s} {('Y' if r.body_bias else 'N'):>3s} "
            f"{r.nominal_vdd_v:>5.2f} "
            f"{r.density_min:>13.1f} {r.density_max:>13.1f} "
            f"{r.library_count:>5d} {r.confidence:>11s}"
        )
    lines.append("")
    lines.append(f"{len(rows)} process node(s)")
    return "\n".join(lines) + "\n"


def _render_json(rows: List[NodeRow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2) + "\n"


def _render_csv(rows: List[NodeRow]) -> str:
    if not rows:
        return ""
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(asdict(r))
    return buf.getvalue()


def _render_md(rows: List[NodeRow]) -> str:
    if not rows:
        return "_no entries match_\n"
    lines = [
        "| id | foundry | node | nm | topology | body bias | Vdd | min Mtx/mm^2 | max Mtx/mm^2 | libs | confidence |",
        "|---|---|---|---:|---|:---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        bb = "yes" if r.body_bias else "no"
        lines.append(
            f"| `{r.id}` | {r.foundry} | {r.node_name} | {r.node_nm} | "
            f"{r.topology} | {bb} | {r.nominal_vdd_v:.2f} | "
            f"{r.density_min:.1f} | {r.density_max:.1f} | "
            f"{r.library_count} | {r.confidence} |"
        )
    lines.append("")
    lines.append(f"_{len(rows)} process node(s)_")
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
        description="List process-node entries from the embodied-schemas catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--foundry", help="Filter by foundry (tsmc/samsung/globalfoundries/...)")
    parser.add_argument(
        "--topology",
        help="Filter by transistor topology (bulk_planar/finfet/fd_soi/gaa)",
    )
    parser.add_argument(
        "--sort",
        choices=sorted(_SORT_KEYS),
        default="foundry",
        help="Sort key (default: foundry)",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.csv/.md/.txt). "
        "Default: stdout as text.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose stderr logging")
    args = parser.parse_args()

    try:
        nodes = load_process_nodes()
    except Exception as exc:
        print(f"error: failed to load process nodes: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"info: loaded {len(nodes)} process node(s)", file=sys.stderr)

    rows = [_build_row(n) for n in nodes.values()]
    rows = _filter_rows(rows, args.foundry, args.topology)
    rows.sort(key=_SORT_KEYS[args.sort])

    fmt = _detect_format(args.output)
    rendered = _RENDERERS[fmt](rows)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        if args.verbose:
            print(f"info: wrote {fmt} output to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
