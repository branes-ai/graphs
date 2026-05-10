#!/usr/bin/env python
"""
Circuit-Class Database Inspector

Prints the CircuitClass enumeration with descriptions and a cross-tab
view of which process nodes support each class and at what density.

The output answers two questions a SKU author keeps asking:
  1. "What library types does the validator framework recognize?"
  2. "If I tag a silicon_bin block as SRAM_HD, which process nodes can
      I target, and what's the density range across the catalog?"

Usage:
    python cli/list_circuit_classes.py
    python cli/list_circuit_classes.py --class sram_hd
    python cli/list_circuit_classes.py --output classes.md
    python cli/list_circuit_classes.py --output classes.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from embodied_schemas import load_process_nodes
from embodied_schemas.process_node import CircuitClass


# Human-readable descriptions of each class. The validator framework keys
# on the enum value; this table is documentation for SKU authors.
_CLASS_DESCRIPTIONS = {
    CircuitClass.HP_LOGIC: (
        "High-performance logic. Larger cells, faster, higher leakage. "
        "Typical use: critical-path datapath, tensor cores, NoC routers."
    ),
    CircuitClass.BALANCED_LOGIC: (
        "Mainstream balanced logic. Default for compute datapaths."
    ),
    CircuitClass.LP_LOGIC: (
        "Low-power logic. Smaller cells, lower leakage, slower. Typical "
        "use: control plane, peripheral logic, mobile-class datapaths."
    ),
    CircuitClass.ULL_LOGIC: (
        "Ultra-low-leakage logic. For retention domains and always-on "
        "blocks. FD-SOI nodes use back-bias to reach this."
    ),
    CircuitClass.SRAM_HD: (
        "High-density single-port SRAM. Caches, scratchpads, "
        "weight-stationary tile buffers."
    ),
    CircuitClass.SRAM_HC: (
        "High-current single-port SRAM. Faster access, larger bitcell."
    ),
    CircuitClass.SRAM_HP: (
        "High-performance dual-port SRAM. Register files, NoC FIFOs."
    ),
    CircuitClass.ANALOG: (
        "Analog blocks: memory PHYs, PLLs, SerDes, RF/mixed-signal frontends."
    ),
    CircuitClass.IO: (
        "IO pad ring, level shifters, ESD."
    ),
    CircuitClass.MIXED: (
        "Weighted-average placeholder for blocks not decomposed further. "
        "Use sparingly -- decomposition is preferable."
    ),
}


def _build_cross_tab() -> Dict[str, Dict[str, Optional[float]]]:
    """For each class, map node_id -> Mtx/mm^2 (or None if unsupported)."""
    nodes = load_process_nodes()
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for cc in CircuitClass:
        per_node: Dict[str, Optional[float]] = {}
        for node_id, node in nodes.items():
            if cc in node.densities:
                per_node[node_id] = node.densities[cc].mtx_per_mm2
            else:
                per_node[node_id] = None
        result[cc.value] = per_node
    return result


def _render_text(filter_class: Optional[str]) -> str:
    out: List[str] = []
    cross = _build_cross_tab()
    classes = list(CircuitClass)
    if filter_class:
        classes = [c for c in classes if c.value == filter_class.lower()]

    for cc in classes:
        out.append(f"=== {cc.value} ===")
        out.append(_CLASS_DESCRIPTIONS[cc])
        out.append("")
        out.append("  Per-node density (Mtx/mm^2):")
        per_node = cross[cc.value]
        supported = [(n, v) for n, v in per_node.items() if v is not None]
        unsupported = [n for n, v in per_node.items() if v is None]
        if supported:
            supported.sort(key=lambda kv: -kv[1])
            for nid, v in supported:
                out.append(f"    {nid:30s} {v:>8.1f}")
            mn = min(v for _, v in supported)
            mx = max(v for _, v in supported)
            out.append(f"    {'(range)':30s} {mn:.1f} .. {mx:.1f}")
        else:
            out.append("    (not supported by any node in catalog)")
        if unsupported:
            out.append(f"  Not offered by: {', '.join(sorted(unsupported))}")
        out.append("")
    return "\n".join(out)


def _render_md(filter_class: Optional[str]) -> str:
    nodes = load_process_nodes()
    node_ids = sorted(nodes)
    classes = list(CircuitClass)
    if filter_class:
        classes = [c for c in classes if c.value == filter_class.lower()]

    lines = ["# Circuit-Class Database", ""]
    lines.append("## Definitions")
    lines.append("")
    for cc in classes:
        lines.append(f"### `{cc.value}`")
        lines.append("")
        lines.append(_CLASS_DESCRIPTIONS[cc])
        lines.append("")

    lines.append("## Density cross-tab (Mtx/mm^2)")
    lines.append("")
    header = "| CircuitClass | " + " | ".join(node_ids) + " |"
    sep = "|---|" + "---:|" * len(node_ids)
    lines.append(header)
    lines.append(sep)
    for cc in classes:
        row = [f"`{cc.value}`"]
        for nid in node_ids:
            d = nodes[nid].densities.get(cc)
            row.append(f"{d.mtx_per_mm2:.1f}" if d else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_json(filter_class: Optional[str]) -> str:
    cross = _build_cross_tab()
    payload: Dict[str, Any] = {}
    classes = list(CircuitClass)
    if filter_class:
        classes = [c for c in classes if c.value == filter_class.lower()]
    for cc in classes:
        payload[cc.value] = {
            "description": _CLASS_DESCRIPTIONS[cc],
            "density_mtx_per_mm2_by_node": {
                k: v for k, v in cross[cc.value].items() if v is not None
            },
            "unsupported_nodes": [
                k for k, v in cross[cc.value].items() if v is None
            ],
        }
    return json.dumps(payload, indent=2)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "md": "md", "markdown": "md", "txt": "text"}.get(ext, "text")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List CircuitClass enumeration with descriptions and per-node "
            "density cross-tab from the embodied-schemas process-node catalog."
        )
    )
    parser.add_argument(
        "--class",
        dest="cls",
        help="Show only this CircuitClass (e.g., sram_hd, balanced_logic)",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt).",
    )
    args = parser.parse_args()

    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = _render_json(args.cls) + "\n"
    elif fmt == "md":
        rendered = _render_md(args.cls) + "\n"
    else:
        rendered = _render_text(args.cls)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
