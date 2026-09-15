#!/usr/bin/env python
"""
Compute-Product Catalog Inspector

Lists every ComputeProduct in the embodied-schemas catalog, of every block
kind (KPU, GPU, CPU, NPU, CGRA, DPU, TPU, DSP; IO blocks appear alongside the
compute blocks of chiplet products). Pair with the per-kind inspectors for
detail: ``show_compute_product.py`` / ``show_kpu.py`` for a KPU SKU,
``list_hardware_resources.py`` for the mapper-registry view.

Output columns:
    id, vendor, block kinds, die count, process node(s), total die area
    (mm^2), total transistors (B), INT8 TOPS, BF16 TFLOPS, FP32 TFLOPS,
    default-profile TDP (W), INT8 TOPS/W, target market, lifecycle,
    data confidence, and whether every die's process node resolves in the
    process-node catalog.

Usage:
    python cli/list_compute_products.py
    python cli/list_compute_products.py --kind kpu
    python cli/list_compute_products.py --vendor google --sort tops
    python cli/list_compute_products.py --sort tops_per_watt
    python cli/list_compute_products.py --format json
    python cli/list_compute_products.py --output products.csv
    python cli/list_compute_products.py --output products.md
"""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from embodied_schemas import load_process_nodes
from embodied_schemas.compute_product import BlockKind, ComputeProduct

from graphs.hardware.compute_product_loader import load_compute_products_unified


def _val(x) -> str:
    """Enum or plain string -> string (block kinds load as either)."""
    return x.value if isinstance(x, Enum) else str(x)


@dataclass
class ProductRow:
    id: str
    name: str
    vendor: str
    block_kinds: str
    dies: int
    process_nodes: str
    die_area_mm2: float
    transistors_billion: float
    int8_tops: float
    bf16_tflops: float
    fp32_tflops: float
    tdp_watts: float
    int8_tops_per_watt: float
    target_market: str
    lifecycle: str
    confidence: str
    nodes_resolve: bool


def _build_row(cp: ComputeProduct, known_nodes: set) -> ProductRow:
    kinds = sorted({_val(b.kind) for d in cp.dies for b in d.blocks})
    nodes = []
    for d in cp.dies:
        if d.process_node_id not in nodes:
            nodes.append(d.process_node_id)
    tdp = cp.power.tdp_watts
    return ProductRow(
        id=cp.id,
        name=cp.name,
        vendor=cp.vendor,
        block_kinds="+".join(kinds),
        dies=len(cp.dies),
        process_nodes=",".join(nodes),
        die_area_mm2=sum(d.die_size_mm2 for d in cp.dies),
        transistors_billion=sum(d.transistors_billion for d in cp.dies),
        int8_tops=cp.performance.int8_tops,
        bf16_tflops=cp.performance.bf16_tflops,
        fp32_tflops=cp.performance.fp32_tflops,
        tdp_watts=tdp,
        int8_tops_per_watt=cp.performance.int8_tops / tdp if tdp > 0 else 0.0,
        target_market=cp.market.target_market,
        lifecycle=_val(cp.lifecycle),
        confidence=_val(cp.confidence),
        nodes_resolve=all(n in known_nodes for n in nodes),
    )


_SORT_KEYS: Dict[str, Callable[[ProductRow], object]] = {
    "id": lambda r: r.id,
    "vendor": lambda r: (r.vendor, r.id),
    "kind": lambda r: (r.block_kinds, r.id),
    "tops": lambda r: -r.int8_tops,  # highest first
    "tdp": lambda r: -r.tdp_watts,
    "tops_per_watt": lambda r: -r.int8_tops_per_watt,
    "die_size": lambda r: -r.die_area_mm2,
}

BLOCK_KINDS = sorted(k.value for k in BlockKind)


def _filter_rows(
    rows: List[ProductRow],
    kind: Optional[str],
    vendor: Optional[str],
    market: Optional[str],
) -> List[ProductRow]:
    out = rows
    if kind:
        out = [r for r in out if kind in r.block_kinds.split("+")]
    if vendor:
        out = [r for r in out if r.vendor == vendor.lower()]
    if market:
        out = [r for r in out if r.target_market == market.lower()]
    return out


def _render_text(rows: List[ProductRow]) -> str:
    if not rows:
        return "(no products match)\n"
    header = (
        f"{'id':40s} {'vendor':10s} {'blocks':8s} {'dies':>4s} {'process node(s)':22s} "
        f"{'mm^2':>7s} {'Btx':>7s} {'INT8 TOPS':>9s} {'BF16 TF':>8s} {'FP32 TF':>8s} "
        f"{'TDP W':>7s} {'TOPS/W':>7s} {'market':12s} {'lifecycle':11s} {'conf':>11s} "
        f"{'refs':>4s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.id:40s} {r.vendor:10s} {r.block_kinds:8s} {r.dies:>4d} "
            f"{r.process_nodes:22s} {r.die_area_mm2:>7.1f} "
            f"{r.transistors_billion:>7.2f} {r.int8_tops:>9.1f} {r.bf16_tflops:>8.1f} "
            f"{r.fp32_tflops:>8.2f} {r.tdp_watts:>7.1f} {r.int8_tops_per_watt:>7.2f} "
            f"{r.target_market:12s} {r.lifecycle:11s} {r.confidence:>11s} "
            f"{('ok' if r.nodes_resolve else 'MISS'):>4s}"
        )
    lines.append("")
    by_kind: Dict[str, int] = {}
    for r in rows:
        by_kind[r.block_kinds] = by_kind.get(r.block_kinds, 0) + 1
    summary = ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items()))
    lines.append(f"{len(rows)} compute product(s): {summary}")
    return "\n".join(lines) + "\n"


def _render_json(rows: List[ProductRow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2) + "\n"


def _render_csv(rows: List[ProductRow]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(asdict(r))
    return buf.getvalue()


def _render_md(rows: List[ProductRow]) -> str:
    if not rows:
        return "_no products match_\n"
    header = (
        "| id | vendor | blocks | dies | process node(s) | die mm^2 | B transistors | "
        "INT8 TOPS | BF16 TFLOPS | FP32 TFLOPS | TDP W | INT8 TOPS/W | market | "
        "lifecycle | confidence | nodes resolve |"
    )
    lines = [
        header,
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.id}` | {r.vendor} | {r.block_kinds} | {r.dies} | {r.process_nodes} | "
            f"{r.die_area_mm2:.1f} | {r.transistors_billion:.2f} | {r.int8_tops:.1f} | "
            f"{r.bf16_tflops:.1f} | {r.fp32_tflops:.2f} | {r.tdp_watts:.1f} | "
            f"{r.int8_tops_per_watt:.2f} | {r.target_market} | {r.lifecycle} | "
            f"{r.confidence} | {'yes' if r.nodes_resolve else 'NO'} |"
        )
    lines.append("")
    lines.append(f"_{len(rows)} compute product(s)_")
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
        description="List every ComputeProduct in the embodied-schemas catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kind", choices=BLOCK_KINDS, help="Only products with this block kind")
    parser.add_argument("--vendor", help="Filter by vendor (stillwater/nvidia/google/...)")
    parser.add_argument("--market", help="Filter by target market (edge/embodied/datacenter/...)")
    parser.add_argument(
        "--sort", choices=sorted(_SORT_KEYS), default="kind", help="Sort key (default: kind)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "md"],
        help="Output format; overrides the --output extension",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.csv/.md/.txt). "
        "Default: stdout as text.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose stderr logging")
    args = parser.parse_args()

    try:
        products = load_compute_products_unified()
        known_nodes = set(load_process_nodes())
    except Exception as exc:
        print(f"error: failed to load the catalog: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"info: loaded {len(products)} compute product(s)", file=sys.stderr)

    rows = [_build_row(cp, known_nodes) for cp in products.values()]
    rows = _filter_rows(rows, args.kind, args.vendor, args.market)
    rows.sort(key=_SORT_KEYS[args.sort])

    fmt = args.format or _detect_format(args.output)
    rendered = _RENDERERS[fmt](rows)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(rendered)
        except OSError as exc:
            print(f"error: cannot write {args.output!r}: {exc}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"info: wrote {fmt} output to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
