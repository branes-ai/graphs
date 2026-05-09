#!/usr/bin/env python
"""
KPU SKU Database Inspector

Lists every KPU SKU in the embodied-schemas catalog. KPUs (Knowledge
Processing Units) are general parallel execution engines, peer of GPUs /
CPUs / NPUs. Each entry references a ProcessNode by id (silicon
fabrication) and a CoolingSolution per thermal profile (thermal removal).

Output columns:
    id, vendor, process_node, total_tiles, total_PEs, default TDP,
    INT8 TOPS, die_size_mm2, transistors_billion, target_market.

Cross-reference health: the loader also surfaces whether each SKU's
process_node_id resolves and whether all thermal-profile
cooling_solution_ids resolve in the cooling-solution catalog.

Usage:
    python cli/list_kpus.py
    python cli/list_kpus.py --vendor stillwater
    python cli/list_kpus.py --target-market datacenter
    python cli/list_kpus.py --sort tops
    python cli/list_kpus.py --output kpus.csv
    python cli/list_kpus.py --output kpus.md
"""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from embodied_schemas import load_kpus, load_process_nodes, load_cooling_solutions
from embodied_schemas.kpu import KPUEntry


@dataclass
class KPURow:
    id: str
    name: str
    vendor: str
    process_node_id: str
    process_node_resolved: bool
    total_tiles: int
    total_pes: int
    default_tdp_w: float
    int8_tops: float
    bf16_tflops: float
    die_size_mm2: float
    transistors_billion: float
    memory_gb: float
    memory_bandwidth_gbps: float
    target_market: str
    model_tier: str
    cooling_unresolved: List[str]


def _build_row(
    e: KPUEntry,
    nodes: Dict[str, Any],
    sols: Dict[str, Any],
) -> KPURow:
    total_pes = sum(t.total_pes for t in e.kpu_architecture.tiles)
    cooling_unresolved = [
        tp.cooling_solution_id
        for tp in e.power.thermal_profiles
        if tp.cooling_solution_id not in sols
    ]
    return KPURow(
        id=e.id,
        name=e.name,
        vendor=e.vendor,
        process_node_id=e.process_node_id,
        process_node_resolved=e.process_node_id in nodes,
        total_tiles=e.kpu_architecture.total_tiles,
        total_pes=total_pes,
        default_tdp_w=e.power.tdp_watts,
        int8_tops=e.performance.int8_tops,
        bf16_tflops=e.performance.bf16_tflops,
        die_size_mm2=e.die.die_size_mm2,
        transistors_billion=e.die.transistors_billion,
        memory_gb=e.kpu_architecture.memory.memory_size_gb,
        memory_bandwidth_gbps=e.kpu_architecture.memory.memory_bandwidth_gbps,
        target_market=e.market.target_market,
        model_tier=e.market.model_tier,
        cooling_unresolved=cooling_unresolved,
    )


_SORT_KEYS = {
    "id": lambda r: r.id,
    "tiles": lambda r: r.total_tiles,
    "pes": lambda r: r.total_pes,
    "tdp": lambda r: r.default_tdp_w,
    "tops": lambda r: -r.int8_tops,
    "die": lambda r: -r.die_size_mm2,
    "transistors": lambda r: -r.transistors_billion,
}


def _filter_rows(
    rows: List[KPURow],
    vendor: Optional[str],
    target_market: Optional[str],
) -> List[KPURow]:
    out = rows
    if vendor:
        out = [r for r in out if r.vendor == vendor.lower()]
    if target_market:
        out = [r for r in out if r.target_market == target_market.lower()]
    return out


def _render_text(rows: List[KPURow]) -> str:
    if not rows:
        return "(no entries match)\n"
    header = (
        f"{'id':30s} {'tiles':>5s} {'PEs':>7s} {'TDP':>5s} "
        f"{'INT8 TOPS':>10s} {'BF16 TFLOPS':>11s} "
        f"{'die mm^2':>9s} {'B trans':>8s} {'mem GB':>7s} {'GB/s':>7s} "
        f"{'node':>16s} {'tier':>11s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        node_str = r.process_node_id if r.process_node_resolved else f"{r.process_node_id}(!)"
        lines.append(
            f"{r.id:30s} {r.total_tiles:>5d} {r.total_pes:>7d} "
            f"{r.default_tdp_w:>5.0f} {r.int8_tops:>10.0f} "
            f"{r.bf16_tflops:>11.0f} {r.die_size_mm2:>9.0f} "
            f"{r.transistors_billion:>8.1f} {r.memory_gb:>7.0f} "
            f"{r.memory_bandwidth_gbps:>7.0f} {node_str:>16s} {r.model_tier:>11s}"
        )
        if r.cooling_unresolved:
            lines.append(
                f"  ! UNRESOLVED cooling refs: {', '.join(r.cooling_unresolved)}"
            )
    lines.append("")
    lines.append(f"{len(rows)} KPU SKU(s)")
    return "\n".join(lines) + "\n"


def _render_json(rows: List[KPURow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2) + "\n"


def _render_csv(rows: List[KPURow]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    for r in rows:
        d = asdict(r)
        d["cooling_unresolved"] = ";".join(d["cooling_unresolved"])
        writer.writerow(d)
    return buf.getvalue()


def _render_md(rows: List[KPURow]) -> str:
    if not rows:
        return "_no entries match_\n"
    lines = [
        "| id | tiles | PEs | TDP (W) | INT8 TOPS | BF16 TFLOPS | die mm^2 | B trans | mem GB | GB/s | node | tier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        node_str = r.process_node_id if r.process_node_resolved else f"{r.process_node_id} (!)"
        lines.append(
            f"| `{r.id}` | {r.total_tiles} | {r.total_pes} | "
            f"{r.default_tdp_w:.0f} | {r.int8_tops:.0f} | "
            f"{r.bf16_tflops:.0f} | {r.die_size_mm2:.0f} | "
            f"{r.transistors_billion:.1f} | {r.memory_gb:.0f} | "
            f"{r.memory_bandwidth_gbps:.0f} | `{node_str}` | {r.model_tier} |"
        )
    lines.append("")
    lines.append(f"_{len(rows)} KPU SKU(s)_")
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
        description="List KPU SKU entries from the embodied-schemas catalog."
    )
    parser.add_argument("--vendor", help="Filter by vendor")
    parser.add_argument(
        "--target-market", help="Filter by target market (edge/embodied/datacenter)"
    )
    parser.add_argument(
        "--sort",
        choices=sorted(_SORT_KEYS),
        default="tiles",
        help="Sort key (default: tiles)",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.csv/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        kpus = load_kpus()
        nodes = load_process_nodes()
        sols = load_cooling_solutions()
    except Exception as exc:
        print(f"error: failed to load catalog: {exc}", file=sys.stderr)
        return 1

    rows = [_build_row(k, nodes, sols) for k in kpus.values()]
    rows = _filter_rows(rows, args.vendor, args.target_market)
    rows.sort(key=_SORT_KEYS[args.sort])

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
