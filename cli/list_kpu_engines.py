#!/usr/bin/env python
"""
KPU Engine Registry (graphs#268 E3)

Exports a heterogeneous KPU's tile classes as *engines*: the form the SoC
micro-architecture study (graphs#269) maps workload stages onto. One
descriptor per tile class, with what it can run, what it costs, and its
precision floor; the stream-linked chains it declares as workload
segments; and, given a set of stage precision requirements, which of them
this die can actually serve.

This is a description, not an analysis. For what one function costs across
the kinds, use ``cli/analyze_kpu_tile_ladder.py``.

Usage:
    python cli/list_kpu_engines.py --from-file build/kpu_h64.yaml
    python cli/list_kpu_engines.py kpu_t64_32x32_lp5x4_16nm_tsmc_ffp
    python cli/list_kpu_engines.py --from-file build/kpu_h64.yaml \
        --require planning_qp=fp32 --require perception_trunk=int8
    python cli/list_kpu_engines.py --from-file build/kpu_h64.yaml \
        --output engines.json

Exit codes:
    0 = report produced, and every --require was served
    1 = a --require could not be served by any programmable engine
    2 = SKU / argument error
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from embodied_schemas import load_process_nodes

from graphs.hardware.compute_product_loader import (
    ComputeProductFileError,
    load_compute_product_file,
    load_compute_products_unified,
)
from graphs.hardware.kpu_access import KPUBlockLookupError, has_kpu_block, kpu_die_of
from graphs.hardware.kpu_engines import (
    EngineDescriptor,
    PrecisionFinding,
    Segment,
    describe_engines,
    describe_segments,
    precision_floor_findings,
)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {
        "json": "json", "csv": "csv", "md": "md",
        "markdown": "md", "txt": "text",
    }.get(ext, "text")


def _render_text(
    engines: Sequence[EngineDescriptor],
    segments: Sequence[Segment],
    findings: Sequence[PrecisionFinding],
) -> str:
    out: List[str] = []
    out.append(f"=== Engines ({len(engines)}) ===")
    out.append(
        f"  {'engine_id':24s} {'kind':15s} {'n':>3s} {'floor':>7s} "
        f"{'mm^2/tile':>10s}  formats / function"
    )
    for engine in engines:
        area = (
            f"{engine.area_mm2_per_tile:.4f}"
            if engine.area_mm2_per_tile is not None else "-"
        )
        what = ", ".join(engine.formats) or (engine.function_id or "-")
        out.append(
            f"  {engine.engine_id:24s} {engine.engine_kind:15s} "
            f"{engine.num_tiles:>3d} {(engine.precision_floor or '-'):>7s} "
            f"{area:>10s}  {what}"
        )
        for precision in engine.precisions:
            tops_w = (
                f"{precision.tops_per_watt:.1f} TOPS/W"
                if precision.tops_per_watt else "energy not declared"
            )
            out.append(
                f"      {precision.operand_format:8s} "
                f"{precision.ops_per_second_per_tile / 1e12:7.3f} TOPS/tile  "
                f"{tops_w}"
            )
        if engine.function_id:
            rate = (
                f"{engine.units_per_second:.4g} {engine.work_unit}/s"
                if engine.units_per_second else "rate not declared"
            )
            # fixed_function_pj_per_unit returns None when the core's
            # reference node is not in the catalog, so it cannot be
            # retargeted. Say that rather than crash on the format.
            energy = (
                f"{engine.pj_per_unit:.4g} pJ/{engine.work_unit}"
                if engine.pj_per_unit is not None
                else "energy not retargetable (reference node unavailable)"
            )
            out.append(f"      {rate}  {energy}")
        if engine.supported_kernels:
            out.append(f"      kernels: {', '.join(engine.supported_kernels)}")
        if engine.notes:
            out.append(f"      note: {engine.notes}")
    out.append("")

    out.append(f"=== Segments ({len(segments)}) ===")
    if not segments:
        out.append("  (no stream-linked chains: every stage reaches DRAM)")
    for segment in segments:
        out.append(
            f"  {segment.segment_id}: {' -> '.join(segment.engine_ids)} "
            f"({segment.num_stages} stages)"
        )
        for unit, absorbed in segment.absorbed_bytes_by_unit:
            out.append(
                f"      {absorbed:g} B/{unit} stays on chip "
                f"(not written, not read back)"
            )
    out.append("")

    if findings:
        out.append(f"=== Precision floors ({len(findings)}) ===")
        for finding in findings:
            out.append(f"  {finding.severity:8s} {finding.message}")
        out.append("")
    return "\n".join(out)


def _engine_rows(engines: Sequence[EngineDescriptor]) -> List[dict]:
    rows: List[dict] = []
    for engine in engines:
        base = {
            "engine_id": engine.engine_id,
            "engine_kind": engine.engine_kind,
            "tile_type": engine.tile_type,
            "num_tiles": engine.num_tiles,
            "sites_per_tile": engine.sites_per_tile,
            "clock_hz": engine.clock_hz,
            "area_mm2_per_tile": engine.area_mm2_per_tile,
            "precision_floor": engine.precision_floor,
            "supported_kernels": ";".join(engine.supported_kernels),
            "function_id": engine.function_id,
            "work_unit": engine.work_unit,
            "units_per_second": engine.units_per_second,
            "pj_per_unit": engine.pj_per_unit,
            "confidence": engine.confidence,
            "provenance": engine.provenance,
            "notes": engine.notes,
        }
        if not engine.precisions:
            rows.append({**base, "operand_format": None,
                         "ops_per_second_per_tile": None, "pj_per_op": None})
            continue
        for precision in engine.precisions:
            rows.append({
                **base,
                "operand_format": precision.operand_format,
                "ops_per_second_per_tile": precision.ops_per_second_per_tile,
                "pj_per_op": precision.pj_per_op,
            })
    return rows


def _render_csv(engines, segments, findings) -> str:
    """One table for all three sections, keyed by ``record_type``.

    Writing only the engines would mean a CSV that cannot explain the
    requirement the run failed on, or name the segments -- and ``main``
    writes this file *before* returning a non-zero exit code, so that is
    exactly when the reader needs them (CodeRabbit on #290).
    """
    rows: List[dict] = [{"record_type": "engine", **row} for row in _engine_rows(engines)]
    rows += [
        {
            "record_type": "segment",
            "segment_id": s.segment_id,
            "engine_ids": ";".join(s.engine_ids),
            "function_ids": ";".join(s.function_ids),
            "num_stages": s.num_stages,
            "absorbed_bytes_by_unit": ";".join(
                f"{unit}={value:g}" for unit, value in s.absorbed_bytes_by_unit
            ),
        }
        for s in segments
    ]
    rows += [
        {
            "record_type": "precision_finding",
            "requirement": f.requirement,
            "operand_format": f.operand_format,
            "served_by": ";".join(f.served_by),
            "severity": f.severity,
            "message": f.message,
        }
        for f in findings
    ]
    if not rows:
        return ""
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, restval="")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _render_json(engines, segments, findings) -> str:
    payload = {
        "engines": [
            {
                "engine_id": e.engine_id, "engine_kind": e.engine_kind,
                "tile_type": e.tile_type, "num_tiles": e.num_tiles,
                "sites_per_tile": e.sites_per_tile, "clock_hz": e.clock_hz,
                "area_mm2_per_tile": e.area_mm2_per_tile,
                "total_area_mm2": e.total_area_mm2,
                "precision_floor": e.precision_floor,
                "supported_kernels": list(e.supported_kernels),
                "function_id": e.function_id, "work_unit": e.work_unit,
                "units_per_second": e.units_per_second,
                "pj_per_unit": e.pj_per_unit,
                "confidence": e.confidence, "provenance": e.provenance,
                "notes": e.notes,
                "precisions": [
                    {
                        "operand_format": p.operand_format,
                        "ops_per_tile_per_clock": p.ops_per_tile_per_clock,
                        "ops_per_second_per_tile": p.ops_per_second_per_tile,
                        "pj_per_op": p.pj_per_op,
                        "tops_per_watt": p.tops_per_watt,
                    }
                    for p in e.precisions
                ],
            }
            for e in engines
        ],
        "segments": [
            {
                "segment_id": s.segment_id,
                "engine_ids": list(s.engine_ids),
                "function_ids": list(s.function_ids),
                "num_stages": s.num_stages,
                "absorbed_bytes_by_unit": dict(s.absorbed_bytes_by_unit),
            }
            for s in segments
        ],
        "precision_findings": [
            {
                "requirement": f.requirement,
                "operand_format": f.operand_format,
                "served_by": list(f.served_by),
                "severity": f.severity,
                "message": f.message,
            }
            for f in findings
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def _render_md(engines, segments, findings) -> str:
    out: List[str] = ["## Engines", ""]
    out.append(
        "| engine_id | kind | tiles | precision floor | mm^2/tile "
        "| formats / function |"
    )
    out.append("|---|---|---:|---|---:|---|")
    for e in engines:
        area = f"{e.area_mm2_per_tile:.4f}" if e.area_mm2_per_tile is not None else "-"
        what = ", ".join(e.formats) or (e.function_id or "-")
        out.append(
            f"| `{e.engine_id}` | {e.engine_kind} | {e.num_tiles} "
            f"| {e.precision_floor or '-'} | {area} | {what} |"
        )
    out.append("")
    out.append("## Segments")
    out.append("")
    if not segments:
        out.append("_No stream-linked chains: every stage reaches DRAM._")
    for s in segments:
        absorbed = ", ".join(
            f"{b:g} B/{u}" for u, b in s.absorbed_bytes_by_unit
        ) or "nothing"
        out.append(
            f"- `{s.segment_id}`: {' -> '.join(s.engine_ids)} "
            f"-- {absorbed} stays on chip"
        )
    if findings:
        out.append("")
        out.append("## Precision floors")
        out.append("")
        for f in findings:
            out.append(f"- **{f.severity}** {f.message}")
    out.append("")
    return "\n".join(out)


_RENDERERS = {"json": _render_json, "csv": _render_csv, "md": _render_md}


def _parse_requirements(pairs: Sequence[str], parser) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in pairs or ():
        if "=" not in pair:
            parser.error(f"--require expects NAME=FORMAT, got {pair!r}")
        name, _, operand_format = pair.partition("=")
        if not name or not operand_format:
            parser.error(f"--require expects NAME=FORMAT, got {pair!r}")
        out[name] = operand_format
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a KPU's tile classes as engines for the SoC study: what "
            "each can run, what it costs, and its precision floor."
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
        "--require", action="append", metavar="NAME=FORMAT",
        help=(
            "A stage and the narrowest operand format it tolerates "
            "(repeatable). Reports which engines can serve it, and exits 1 "
            "if none can."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension "
             "(.json/.csv/.md/.txt).",
    )
    args = parser.parse_args()

    if args.sku_id and args.from_file:
        parser.error("give a SKU id or --from-file, not both")
    if not args.sku_id and not args.from_file:
        parser.error("give a KPU SKU id, or --from-file PATH")
    requirements = _parse_requirements(args.require, parser)

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

    engines = describe_engines(cp, node, nodes)
    segments = describe_segments(cp)
    findings = precision_floor_findings(engines, requirements)

    fmt = _detect_format(args.output)
    renderer = _RENDERERS.get(fmt, _render_text)
    rendered = renderer(engines, segments, findings)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(rendered)
        except OSError as exc:
            print(f"error: cannot write {args.output}: {exc}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)

    return 1 if any(f.severity == "ERROR" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
