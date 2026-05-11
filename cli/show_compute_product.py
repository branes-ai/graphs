#!/usr/bin/env python
"""
ComputeProduct Hierarchy Inspector

Prints the full assembled ComputeProduct view for one KPU SKU:
the SKU layer, the referenced ProcessNode, and every CoolingSolution
referenced by a thermal profile. Adds a thermal cross-check panel that
flags TDP-vs-cooling and power-density-vs-cooling violations.

ComputeProduct = ProcessNode + CoolingSolution(s) + SKU. The peer
inspectors print one layer each (show_kpu, show_process_node,
show_cooling_solution); this one composes them.

Today only KPU SKUs are supported -- extend when other product types
need the same composed view.

Usage:
    python cli/show_compute_product.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp
    python cli/show_compute_product.py kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc --output t768.json
    python cli/show_compute_product.py kpu_t64_32x32_lp5x4_16nm_tsmc_ffp  --output t64.md
"""

import argparse
import csv
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from embodied_schemas import load_cooling_solutions, load_kpus, load_process_nodes
from embodied_schemas.cooling_solution import CoolingSolutionEntry
from embodied_schemas.kpu import KPUEntry
from embodied_schemas.process_node import ProcessNodeEntry

# Reuse the layer renderers from sibling inspectors so the composed view
# stays formatting-identical to the per-layer tools.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from show_cooling_solution import _render_md as _render_cool_md  # noqa: E402
from show_cooling_solution import _render_text as _render_cool_text  # noqa: E402
from show_kpu import _render_text as _render_kpu_text  # noqa: E402
from show_process_node import _render_md as _render_node_md  # noqa: E402
from show_process_node import _render_text as _render_node_text  # noqa: E402


# --- Cross-checks -------------------------------------------------------

def _thermal_check_rows(
    e: KPUEntry,
    cools: Dict[str, CoolingSolutionEntry],
) -> List[Tuple[str, str, str, str, str, str, str]]:
    """One row per thermal profile, ready for any renderer:
    (profile, tdp_w, w_per_mm2, cooling_id, cool_max_w, cool_max_w_per_mm2, status)."""
    rows = []
    die_mm2 = e.die.die_size_mm2
    for tp in e.power.thermal_profiles:
        cool = cools.get(tp.cooling_solution_id)
        density = tp.tdp_watts / die_mm2 if die_mm2 > 0 else 0.0
        if cool is None:
            rows.append((
                tp.name, f"{tp.tdp_watts:.0f}", f"{density:.3f}",
                tp.cooling_solution_id, "?", "?", "MISSING_COOLING_REF",
            ))
            continue
        flags = []
        if tp.tdp_watts > cool.max_total_w:
            flags.append("TDP>cool.max_total_w")
        if density > cool.max_power_density_w_per_mm2:
            flags.append("density>cool.max_W/mm^2")
        status = "OK" if not flags else "; ".join(flags)
        rows.append((
            tp.name, f"{tp.tdp_watts:.0f}", f"{density:.3f}",
            cool.id, f"{cool.max_total_w:.0f}",
            f"{cool.max_power_density_w_per_mm2:.2f}", status,
        ))
    return rows


def _render_crosscheck_text(
    e: KPUEntry, cools: Dict[str, CoolingSolutionEntry]
) -> str:
    out = ["=== ComputeProduct cross-checks ===", ""]
    out.append("--- Thermal headroom (per profile) ---")
    out.append(
        f"  {'profile':10s} {'TDP':>5s} {'W/mm^2':>8s}  "
        f"{'cooling_id':28s} {'max W':>6s} {'max W/mm^2':>10s}  status"
    )
    for r in _thermal_check_rows(e, cools):
        out.append(
            f"  {r[0]:10s} {r[1]:>5s} {r[2]:>8s}  "
            f"{r[3]:28s} {r[4]:>6s} {r[5]:>10s}  {r[6]}"
        )
    out.append("")
    return "\n".join(out)


def _render_crosscheck_md(
    e: KPUEntry, cools: Dict[str, CoolingSolutionEntry]
) -> str:
    lines = [
        "## ComputeProduct cross-checks", "",
        "### Thermal headroom (per profile)", "",
        "| profile | TDP (W) | W/mm^2 | cooling_id | max W | max W/mm^2 | status |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for r in _thermal_check_rows(e, cools):
        lines.append(
            f"| {r[0]} | {r[1]} | {r[2]} | `{r[3]}` | {r[4]} | {r[5]} | {r[6]} |"
        )
    lines.append("")
    return "\n".join(lines)


# --- Composition --------------------------------------------------------

def _banner(sku: KPUEntry, node_resolved: bool, cool_unresolved: List[str]) -> List[str]:
    cooling_ids = sorted({tp.cooling_solution_id for tp in sku.power.thermal_profiles})
    bar = "#" * 60
    lines = [
        bar,
        f"# ComputeProduct: {sku.id}",
        f"#   process_node:   {sku.process_node_id} "
        f"({'resolved' if node_resolved else 'UNRESOLVED'})",
        f"#   cooling refs:   {', '.join(cooling_ids)}",
    ]
    if cool_unresolved:
        lines.append(f"#   UNRESOLVED:     {', '.join(cool_unresolved)}")
    lines.append(bar)
    lines.append("")
    return lines


def _compose_text(
    sku: KPUEntry, node: Optional[ProcessNodeEntry],
    cools: Dict[str, CoolingSolutionEntry],
    cool_unresolved: List[str],
) -> str:
    parts = _banner(sku, node is not None, cool_unresolved)
    parts.append(_render_kpu_text(sku))
    parts.append("")
    if node is not None:
        parts.append(_render_node_text(node))
        parts.append("")
    else:
        parts.append(f"!! ProcessNode {sku.process_node_id!r} not found in catalog")
        parts.append("")
    for cid in sorted({tp.cooling_solution_id for tp in sku.power.thermal_profiles}):
        cool = cools.get(cid)
        if cool is None:
            parts.append(f"!! CoolingSolution {cid!r} not found in catalog")
            parts.append("")
        else:
            parts.append(_render_cool_text(cool))
            parts.append("")
    parts.append(_render_crosscheck_text(sku, cools))
    return "\n".join(parts)


def _compose_md(
    sku: KPUEntry, node: Optional[ProcessNodeEntry],
    cools: Dict[str, CoolingSolutionEntry],
    cool_unresolved: List[str],
) -> str:
    cooling_ids = sorted({tp.cooling_solution_id for tp in sku.power.thermal_profiles})
    parts = [f"# ComputeProduct: `{sku.id}`", ""]
    parts.append(
        f"- process_node: `{sku.process_node_id}`"
        f"{'' if node is not None else ' (UNRESOLVED)'}"
    )
    parts.append(f"- cooling refs: {', '.join(f'`{c}`' for c in cooling_ids)}")
    if cool_unresolved:
        parts.append(f"- **UNRESOLVED**: {', '.join(cool_unresolved)}")
    parts.append("")
    # show_kpu has no MD renderer; embed text in a fenced block.
    parts.append("## KPU SKU")
    parts.append("")
    parts.append("```")
    parts.append(_render_kpu_text(sku))
    parts.append("```")
    parts.append("")
    if node is not None:
        parts.append(_render_node_md(node))
        parts.append("")
    else:
        parts.append(f"> ProcessNode `{sku.process_node_id}` not found in catalog")
        parts.append("")
    for cid in cooling_ids:
        cool = cools.get(cid)
        if cool is None:
            parts.append(f"> CoolingSolution `{cid}` not found in catalog")
            parts.append("")
        else:
            parts.append(_render_cool_md(cool))
            parts.append("")
    parts.append(_render_crosscheck_md(sku, cools))
    return "\n".join(parts)


def _compose_json(
    sku: KPUEntry, node: Optional[ProcessNodeEntry],
    cools: Dict[str, CoolingSolutionEntry],
    cool_unresolved: List[str],
) -> Dict[str, Any]:
    return {
        "compute_product_id": sku.id,
        "sku": sku.model_dump(mode="json"),
        "process_node": node.model_dump(mode="json") if node is not None else None,
        "cooling_solutions": {
            cid: cool.model_dump(mode="json") for cid, cool in cools.items()
        },
        "unresolved_cooling_refs": cool_unresolved,
        "cross_checks": {
            "thermal_headroom": [
                {
                    "profile": r[0],
                    "tdp_w": None if r[1] == "?" else float(r[1]),
                    "power_density_w_per_mm2": None if r[2] == "?" else float(r[2]),
                    "cooling_id": r[3],
                    "cool_max_total_w": None if r[4] == "?" else float(r[4]),
                    "cool_max_w_per_mm2": None if r[5] == "?" else float(r[5]),
                    "status": r[6],
                }
                for r in _thermal_check_rows(sku, cools)
            ],
        },
    }


def _compose_csv(
    sku: KPUEntry,
    node: Optional[ProcessNodeEntry],
    cools: Dict[str, CoolingSolutionEntry],
) -> str:
    """One CSV row per thermal profile; the assembled hierarchy doesn't
    flatten cleanly into a single table, so this emits the cross-check
    panel (the most spreadsheet-friendly slice) with the SKU id and node
    resolution status as context columns. Use --output FILE.json for the
    full nested view."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "compute_product_id",
            "process_node_id",
            "process_node_resolved",
            "profile",
            "tdp_w",
            "power_density_w_per_mm2",
            "cooling_id",
            "cool_max_total_w",
            "cool_max_w_per_mm2",
            "status",
        ],
    )
    writer.writeheader()
    for r in _thermal_check_rows(sku, cools):
        writer.writerow({
            "compute_product_id": sku.id,
            "process_node_id": sku.process_node_id,
            "process_node_resolved": node is not None,
            "profile": r[0],
            "tdp_w": None if r[1] == "?" else float(r[1]),
            "power_density_w_per_mm2": None if r[2] == "?" else float(r[2]),
            "cooling_id": r[3],
            "cool_max_total_w": None if r[4] == "?" else float(r[4]),
            "cool_max_w_per_mm2": None if r[5] == "?" else float(r[5]),
            "status": r[6],
        })
    return buf.getvalue()


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {
        "json": "json",
        "csv": "csv",
        "md": "md",
        "markdown": "md",
        "txt": "text",
    }.get(ext, "text")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the full assembled ComputeProduct hierarchy "
                    "(SKU + ProcessNode + CoolingSolutions) for one KPU SKU."
    )
    parser.add_argument("kpu_id", help="KPU SKU id, e.g., kpu_t256_32x32_lp5x16_16nm_tsmc_ffp")
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension "
        "(.json/.csv/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        kpus = load_kpus()
        nodes = load_process_nodes()
        sols = load_cooling_solutions()
    except Exception as exc:
        print(f"error: failed to load catalog: {exc}", file=sys.stderr)
        return 1

    sku = kpus.get(args.kpu_id)
    if sku is None:
        print(
            f"error: no KPU SKU with id={args.kpu_id!r}. "
            f"Available: {', '.join(sorted(kpus))}",
            file=sys.stderr,
        )
        return 1

    node = nodes.get(sku.process_node_id)
    cooling_ids = sorted({tp.cooling_solution_id for tp in sku.power.thermal_profiles})
    cools = {cid: sols[cid] for cid in cooling_ids if cid in sols}
    cool_unresolved = [cid for cid in cooling_ids if cid not in sols]

    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = json.dumps(
            _compose_json(sku, node, cools, cool_unresolved), indent=2
        ) + "\n"
    elif fmt == "csv":
        rendered = _compose_csv(sku, node, cools)
    elif fmt == "md":
        rendered = _compose_md(sku, node, cools, cool_unresolved) + "\n"
    else:
        rendered = _compose_text(sku, node, cools, cool_unresolved) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)

    if node is None or cool_unresolved:
        print(
            "warning: ComputeProduct has unresolved refs "
            f"(node_resolved={node is not None}, "
            f"cooling_unresolved={cool_unresolved})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
