#!/usr/bin/env python
"""
Process-Node Detail Inspector

Prints the full per-library breakdown of one ProcessNodeEntry: every
CircuitClass with its density, library name, leakage, energy-per-op
table, EM J_max, routing-metal widths, and provenance.

Pair with ``list_process_nodes.py`` for the catalog-level view.

Usage:
    python cli/show_process_node.py tsmc_n16
    python cli/show_process_node.py gf_12fdx
    python cli/show_process_node.py tsmc_n7 --output n7.json
    python cli/show_process_node.py tsmc_n5 --output n5.md
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from embodied_schemas import load_process_nodes
from embodied_schemas.process_node import ProcessNodeEntry


def _node_to_dict(entry: ProcessNodeEntry) -> Dict[str, Any]:
    return entry.model_dump(mode="json")


def _render_text(entry: ProcessNodeEntry) -> str:
    out = []
    out.append(f"=== ProcessNode: {entry.id} ===")
    out.append(f"  Name:        {entry.node_name}  ({entry.node_nm} nm)")
    out.append(f"  Foundry:     {entry.foundry.value}")
    out.append(f"  Topology:    {entry.transistor_topology.value}")
    out.append(f"  Body bias:   {'yes' if entry.body_bias_supported else 'no'}")
    if entry.back_bias_range_mv:
        out.append(
            f"  BB range:    {entry.back_bias_range_mv[0]:+d} .. "
            f"{entry.back_bias_range_mv[1]:+d} mV"
        )
    if entry.vt_options:
        out.append(f"  Vt options:  {', '.join(entry.vt_options)}")
    out.append(f"  Nominal Vdd: {entry.nominal_vdd_v:.2f} V")
    if entry.m0_pitch_nm:
        out.append(f"  M0 pitch:    {entry.m0_pitch_nm} nm")
    if entry.m1_pitch_nm:
        out.append(f"  M1 pitch:    {entry.m1_pitch_nm} nm")
    out.append(f"  Confidence:  {entry.confidence.value}")
    out.append(f"  Source:      {entry.source}")
    out.append(f"  Updated:     {entry.last_updated}")
    out.append("")

    # Per-library densities
    out.append("--- Per-library densities ---")
    out.append(
        f"  {'CircuitClass':18s} {'Mtx/mm^2':>10s} {'Library':30s} "
        f"{'Confidence':>11s}  Source"
    )
    for cc, ld in sorted(entry.densities.items(), key=lambda kv: -kv[1].mtx_per_mm2):
        lib = ld.library_name or "-"
        out.append(
            f"  {cc.value:18s} {ld.mtx_per_mm2:>10.1f} {lib:30s} "
            f"{ld.confidence.value:>11s}  {ld.source}"
        )
    out.append("")

    # Leakage
    if entry.leakage_w_per_mm2:
        out.append("--- Leakage (W/mm^2 at nominal Vdd, Tj=85C) ---")
        for cc, w in sorted(entry.leakage_w_per_mm2.items(), key=lambda kv: -kv[1]):
            out.append(f"  {cc.value:18s} {w:.4g}")
        out.append("")

    # Energy per op
    if entry.energy_per_op_pj:
        out.append("--- Energy per op (pJ) ---")
        for key in sorted(entry.energy_per_op_pj):
            out.append(f"  {key:35s} {entry.energy_per_op_pj[key]:.3g}")
        out.append("")

    # EM
    if entry.em_j_max_by_temp_c:
        out.append("--- Electromigration J_max (A/cm^2) ---")
        for tc in sorted(entry.em_j_max_by_temp_c):
            out.append(f"  Tj={tc:>3d} C   {entry.em_j_max_by_temp_c[tc]:.2e}")
        out.append("")

    # Routing
    if entry.routing_metal_width_um:
        out.append("--- Local routing metal width (um) ---")
        for cc, w in entry.routing_metal_width_um.items():
            out.append(f"  {cc.value:18s} {w:.3f}")
        out.append("")

    if entry.cooling_compatible:
        out.append(f"Cooling compatible: {', '.join(entry.cooling_compatible)}")
        out.append("")

    if entry.notes:
        out.append("--- Notes ---")
        out.append(entry.notes.rstrip())
        out.append("")

    return "\n".join(out)


def _render_md(entry: ProcessNodeEntry) -> str:
    lines = [f"# ProcessNode: `{entry.id}`", ""]
    lines.append(f"- **Name**: {entry.node_name} ({entry.node_nm} nm)")
    lines.append(f"- **Foundry**: {entry.foundry.value}")
    lines.append(f"- **Topology**: {entry.transistor_topology.value}")
    lines.append(f"- **Body bias**: {'yes' if entry.body_bias_supported else 'no'}")
    lines.append(f"- **Nominal Vdd**: {entry.nominal_vdd_v:.2f} V")
    lines.append(f"- **Confidence**: {entry.confidence.value}")
    lines.append(f"- **Source**: {entry.source}")
    lines.append("")
    lines.append("## Per-library densities")
    lines.append("")
    lines.append("| CircuitClass | Mtx/mm^2 | Library | Confidence | Source |")
    lines.append("|---|---:|---|---|---|")
    for cc, ld in sorted(entry.densities.items(), key=lambda kv: -kv[1].mtx_per_mm2):
        lib = ld.library_name or "-"
        lines.append(
            f"| `{cc.value}` | {ld.mtx_per_mm2:.1f} | {lib} | "
            f"{ld.confidence.value} | {ld.source} |"
        )
    lines.append("")
    if entry.leakage_w_per_mm2:
        lines.append("## Leakage (W/mm^2)")
        lines.append("")
        lines.append("| CircuitClass | W/mm^2 |")
        lines.append("|---|---:|")
        for cc, w in sorted(entry.leakage_w_per_mm2.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{cc.value}` | {w:.4g} |")
        lines.append("")
    if entry.notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(entry.notes.rstrip())
        lines.append("")
    return "\n".join(lines)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "md": "md", "markdown": "md", "txt": "text"}.get(ext, "text")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show full per-library breakdown of one ProcessNodeEntry."
    )
    parser.add_argument(
        "node_id", help="Process-node id (e.g., tsmc_n16, gf_12fdx, samsung_8lpp)"
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        nodes = load_process_nodes()
    except Exception as exc:
        print(f"error: failed to load process nodes: {exc}", file=sys.stderr)
        return 1

    entry = nodes.get(args.node_id)
    if entry is None:
        print(
            f"error: no process node with id={args.node_id!r}. "
            f"Available: {', '.join(sorted(nodes))}",
            file=sys.stderr,
        )
        return 1

    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = json.dumps(_node_to_dict(entry), indent=2) + "\n"
    elif fmt == "md":
        rendered = _render_md(entry) + "\n"
    else:
        rendered = _render_text(entry) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
