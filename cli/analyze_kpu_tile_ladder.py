#!/usr/bin/env python
"""
KPU Tile-Kind Energy Ladder (graphs#268 E2)

Prices one function on every implementation a heterogeneous KPU offers --
a fixed-function core, a systolic array, a PE fabric -- with a GPU and a
CPU as reference rungs, and reports what specialization actually buys.

Each rung carries its provenance and confidence, because the answer is a
ratio and a ratio is only as good as the two numbers in it. Two things
keep the comparison honest:

  * The ops model is stated. Comparing a core published in pJ/pixel
    against a tile published in pJ/op needs a figure for how much
    arithmetic a pixel costs, and that is a workload claim, not a silicon
    one. --show-ops prints the arithmetic and its source.

  * The back-check is printed. Dividing a core's published energy by that
    ops model gives its implied energy per op; when that is far above what
    the process node charges for an arithmetic op, the core is not
    arithmetic-bound and the programmable rungs are a lower bound rather
    than a like-for-like comparison. The report says so.

Usage:
    python cli/analyze_kpu_tile_ladder.py --from-file build/kpu_h64.yaml
    python cli/analyze_kpu_tile_ladder.py --from-file build/kpu_h64.yaml \
        --function stereo.sgm --show-ops
    python cli/analyze_kpu_tile_ladder.py kpu_h64_auto1_lp5x4_16nm_tsmc_ffp \
        --output ladder.md
    python cli/analyze_kpu_tile_ladder.py --list-functions

Exit codes:
    0 = report produced
    1 = no rung could be priced for the requested function
    2 = SKU / argument error
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import List, Optional

from embodied_schemas import load_process_nodes

from graphs.hardware.compute_product_loader import (
    ComputeProductFileError,
    load_compute_product_file,
    load_compute_products_unified,
)
from graphs.hardware.kpu_access import KPUBlockLookupError, has_kpu_block, kpu_die_of
from graphs.hardware.kpu_tile_ladder import (
    OPS_MODELS,
    Ladder,
    LadderError,
    build_ladder,
)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {
        "json": "json", "csv": "csv", "md": "md",
        "markdown": "md", "txt": "text",
    }.get(ext, "text")


def _render_text(ladders: List[Ladder], show_ops: bool) -> str:
    out: List[str] = []
    for ladder in ladders:
        ops = ladder.ops
        out.append(f"=== {ladder.function_id} on {ladder.sku_id} @ {ladder.node_id} ===")
        out.append(
            f"  work unit:   1 {ops.work_unit} = {ops.ops_per_unit:.4g} ops "
            f"({ops.confidence})"
        )
        if show_ops:
            out.append(f"    derivation: {ops.derivation}")
            out.append(f"    source:     {ops.citation}")
        out.append("")
        unit = ops.work_unit
        out.append(
            f"  {'implementation':44s} {'kind':15s} {'pJ/' + unit:>13s} "
            f"{'pJ/op':>9s} {'mm^2/tile':>10s} {'vs best':>8s}  confidence"
        )
        for rung in ladder.rungs:
            per_op = f"{rung.pj_per_op:.4g}" if rung.pj_per_op is not None else "-"
            area = f"{rung.area_mm2:.4f}" if rung.area_mm2 is not None else "-"
            out.append(
                f"  {rung.label:44s} {rung.kind:15s} {rung.pj_per_unit:13.4g} "
                f"{per_op:>9s} {area:>10s} {ladder.ratio_to_best(rung):7.1f}x  "
                f"{rung.confidence}"
            )
            if show_ops:
                out.append(f"      from: {rung.provenance}")
                if rung.notes:
                    out.append(f"      note: {rung.notes}")
        out.append("")

        verdict = (
            "holds" if ladder.follows_specialization_order
            else "DOES NOT hold -- see the order above"
        )
        out.append(f"  Specialization order (fixed-function < systolic < PE fabric): {verdict}")
        for note in ladder.back_check_notes:
            out.append(f"  ! {note}")
        if ladder.dram_note:
            out.append(f"  Segment encapsulation: {ladder.dram_note}")
        out.append("")
    return "\n".join(out)


def _rows(ladders: List[Ladder]) -> List[dict]:
    return [
        {
            "function_id": ladder.function_id,
            "sku_id": ladder.sku_id,
            "node": ladder.node_id,
            "work_unit": ladder.ops.work_unit,
            "ops_per_unit": ladder.ops.ops_per_unit,
            "ops_confidence": ladder.ops.confidence,
            "implementation": rung.label,
            "kind": rung.kind,
            "pj_per_unit": rung.pj_per_unit,
            "pj_per_op": rung.pj_per_op,
            "area_mm2_per_tile": rung.area_mm2,
            "ratio_to_best": ladder.ratio_to_best(rung),
            "confidence": rung.confidence,
            "provenance": rung.provenance,
            "notes": rung.notes,
        }
        for ladder in ladders
        for rung in ladder.rungs
    ]


def _render_json(ladders: List[Ladder]) -> str:
    payload = [
        {
            "function_id": ladder.function_id,
            "sku_id": ladder.sku_id,
            "node": ladder.node_id,
            "ops_model": {
                "ops_per_unit": ladder.ops.ops_per_unit,
                "work_unit": ladder.ops.work_unit,
                "derivation": ladder.ops.derivation,
                "confidence": ladder.ops.confidence,
                "citation": ladder.ops.citation,
            },
            "follows_specialization_order": ladder.follows_specialization_order,
            "arithmetic_bound": ladder.arithmetic_bound,
            "back_check": list(ladder.back_check_notes),
            "dram_bytes_avoided_per_unit": ladder.dram_bytes_per_unit,
            "dram_pj_avoided_per_unit": ladder.dram_pj_per_unit,
            "dram_note": ladder.dram_note,
            "rungs": [
                {
                    "implementation": r.label, "kind": r.kind,
                    "pj_per_unit": r.pj_per_unit, "pj_per_op": r.pj_per_op,
                    "area_mm2_per_tile": r.area_mm2,
                    "ratio_to_best": ladder.ratio_to_best(r),
                    "confidence": r.confidence, "provenance": r.provenance,
                    "notes": r.notes,
                }
                for r in ladder.rungs
            ],
        }
        for ladder in ladders
    ]
    return json.dumps(payload, indent=2) + "\n"


def _render_csv(ladders: List[Ladder]) -> str:
    rows = _rows(ladders)
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _render_md(ladders: List[Ladder]) -> str:
    out: List[str] = []
    for ladder in ladders:
        ops = ladder.ops
        out.append(f"## `{ladder.function_id}` on `{ladder.sku_id}` @ {ladder.node_id}")
        out.append("")
        out.append(
            f"1 {ops.work_unit} = {ops.ops_per_unit:.4g} ops "
            f"({ops.confidence}). {ops.derivation}. Source: {ops.citation}"
        )
        out.append("")
        out.append(
            f"| implementation | kind | pJ/{ops.work_unit} | pJ/op | mm^2/tile "
            "| vs best | confidence |"
        )
        out.append("|---|---|---:|---:|---:|---:|---|")
        for r in ladder.rungs:
            per_op = f"{r.pj_per_op:.4g}" if r.pj_per_op is not None else "-"
            area = f"{r.area_mm2:.4f}" if r.area_mm2 is not None else "-"
            out.append(
                f"| {r.label} | {r.kind} | {r.pj_per_unit:.4g} | {per_op} "
                f"| {area} | {ladder.ratio_to_best(r):.1f}x | {r.confidence} |"
            )
        out.append("")
        out.append(
            "Specialization order (fixed-function < systolic < PE fabric): "
            + ("**holds**" if ladder.follows_specialization_order
               else "**does not hold**")
        )
        for note in ladder.back_check_notes:
            out.append("")
            out.append(f"> {note}")
        if ladder.dram_note:
            out.append("")
            out.append(f"Segment encapsulation: {ladder.dram_note}")
        out.append("")
    return "\n".join(out)


_RENDERERS = {"json": _render_json, "csv": _render_csv, "md": _render_md}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one function across fixed-function, systolic and "
            "PE-fabric implementations, with GPU and CPU references."
        )
    )
    parser.add_argument(
        "sku_id", nargs="?",
        help="KPU SKU id from the catalog. Omit with --from-file.",
    )
    parser.add_argument(
        "--from-file", metavar="PATH",
        help="Read the SKU from a ComputeProduct YAML / JSON file instead.",
    )
    parser.add_argument(
        "--function", action="append", metavar="ID",
        help=(
            "Function to price (repeatable). Default: every function this "
            "SKU can run. See --list-functions."
        ),
    )
    parser.add_argument(
        "--list-functions", action="store_true",
        help="List the functions the ladder knows how to price, and exit.",
    )
    parser.add_argument(
        "--show-ops", action="store_true",
        help="Show each ops model's derivation and each rung's provenance.",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension "
             "(.json/.csv/.md/.txt).",
    )
    args = parser.parse_args()

    if args.list_functions:
        for fid, ops in sorted(OPS_MODELS.items()):
            print(
                f"{fid:24s} {ops.ops_per_unit:>12.4g} ops/{ops.work_unit:<6s} "
                f"{ops.confidence}"
            )
        return 0

    if args.sku_id and args.from_file:
        parser.error("give a SKU id or --from-file, not both")
    if not args.sku_id and not args.from_file:
        parser.error("give a KPU SKU id, or --from-file PATH, or --list-functions")

    if args.from_file:
        try:
            cp = load_compute_product_file(args.from_file)
        except ComputeProductFileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not has_kpu_block(cp):
            print(
                f"error: {args.from_file} holds compute product {cp.id!r}, "
                "which has no KPU block",
                file=sys.stderr,
            )
            return 2
    else:
        products = load_compute_products_unified()
        cp = products.get(args.sku_id)
        if cp is None or not has_kpu_block(cp):
            kpu_ids = sorted(k for k, v in products.items() if has_kpu_block(v))
            print(
                f"error: no KPU SKU with id={args.sku_id!r}. "
                f"Available: {', '.join(kpu_ids)}",
                file=sys.stderr,
            )
            return 2

    nodes = load_process_nodes()
    try:
        node_id = kpu_die_of(cp).process_node_id
    except KPUBlockLookupError as exc:
        print(f"error: invalid KPU SKU {cp.id!r}: {exc}", file=sys.stderr)
        return 2
    node = nodes.get(node_id)
    if node is None:
        print(
            f"error: SKU references unknown process_node_id {node_id!r}",
            file=sys.stderr,
        )
        return 2

    requested = args.function or sorted(OPS_MODELS)
    unknown = [f for f in requested if f not in OPS_MODELS]
    if unknown:
        print(
            f"error: no ops model for {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(OPS_MODELS))}",
            file=sys.stderr,
        )
        return 2

    ladders: List[Ladder] = []
    for function_id in requested:
        try:
            ladders.append(build_ladder(cp, node, function_id, nodes=nodes))
        except LadderError as exc:
            # Only a hard error when the user asked for this function by
            # name; a default sweep just skips what this SKU cannot run.
            if args.function:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"info: skipping {function_id}: {exc}", file=sys.stderr)

    if not ladders:
        print(
            f"error: none of the requested functions could be priced on "
            f"{cp.id!r}",
            file=sys.stderr,
        )
        return 1

    fmt = _detect_format(args.output)
    renderer = _RENDERERS.get(fmt)
    rendered = renderer(ladders) if renderer else _render_text(ladders, args.show_ops)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(rendered)
        except OSError as exc:
            print(f"error: cannot write {args.output}: {exc}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
