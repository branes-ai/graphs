#!/usr/bin/env python
"""
SoC Design Inspector

Shows an SoC design from soc_designs/ composed at a process node: die area,
transistors, per-block area and peak throughput, clocks and how each was
obtained, and every silicon line that has no public area anchor.

This is graphs#269's Phase 2 inspector. It bypasses UnifiedAnalyzer
deliberately: there is no model to analyze, only silicon to price, the same
exception analyze_operator_roofline.py documents.

Read the completeness line first. When a design has unanchored silicon, every
area and transistor figure here is a LOWER BOUND, and the gap list says which
blocks are missing. Nothing is estimated to fill them.

Usage:
    python cli/show_soc.py --design orin_class_reference
    python cli/show_soc.py --design orin_class_reference --node tsmc_n5
    python cli/show_soc.py --design orin_class_reference --sources   # cite every line
    python cli/show_soc.py --list-ip
    python cli/show_soc.py --design orin_class_reference --output orin.md

Exit codes:
    0 = report produced
    2 = unknown design or node, or a library the node does not offer
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
from embodied_schemas import load_process_nodes  # noqa: E402

from graphs.hardware.soc import (  # noqa: E402
    SoCInstance,
    compose_soc,
    load_designs,
    load_ip_library,
)
from graphs.hardware.sku_validators.silicon_math import SiliconMathError  # noqa: E402
from graphs.reporting.output_format import csv_writer, detect_format, write_report  # noqa: E402


def _block_rows(soc: SoCInstance) -> List[dict]:
    rows = []
    for b in soc.blocks:
        int8 = b.peak_ops_per_s("int8") / 1e12
        rows.append({
            "instance": b.name,
            "ip": b.template.id,
            "count": b.count,
            "engine": b.engine_kind.value,
            # A block with no anchored line has no area at all, which must not
            # read as zero: it is unknown.
            "area_mm2": round(b.area_mm2, 3) if len(b.unanchored) < len(b.lines) else "unknown",
            "transistors_mtx": (
                round(b.transistors_mtx, 1) if len(b.unanchored) < len(b.lines) else "unknown"
            ),
            "anchored": (
                "yes" if b.complete
                else "none" if len(b.unanchored) == len(b.lines)
                else f"partial ({len(b.unanchored)} of {len(b.lines)} unanchored)"
            ),
            "clock_ghz": "" if b.clock_ghz is None else round(b.clock_ghz, 3),
            "clock_basis": b.clock_basis,
            "int8_tops_dense": round(int8, 2) if int8 else "",
        })
    return rows


def _ip_rows(library) -> List[dict]:
    return [{
        "ip": t.id,
        "name": t.name,
        "engine": t.engine_kind.value,
        "lines": len(t.silicon),
        "unanchored": len(t.unanchored_lines),
        "weakest_confidence": t.weakest_confidence.value,
        "clock": f"{t.clock.fmax_ghz_ref:g} GHz @ {t.clock.reference_node}" if t.clock else "",
    } for t in library.values()]


def _summary(soc: SoCInstance) -> List[str]:
    bound = "" if soc.complete else " (LOWER BOUND)"
    lines = [
        f"design: {soc.design.id} -- {soc.design.name}",
        f"node:   {soc.node.id}",
        f"complete: {'yes' if soc.complete else f'NO -- {len(soc.gaps)} unanchored silicon line(s)'}",
        f"die:    {soc.die_area_mm2:.1f} mm^2{bound}  ({soc.die_side_mm:.2f} mm square; "
        f"blocks {soc.block_area_mm2:.1f} + whitespace "
        f"{soc.core_area_mm2 - soc.block_area_mm2:.1f} + IO ring {soc.io_ring_area_mm2:.1f})",
        f"transistors: {soc.transistors_billion:.2f} B{bound}",
    ]
    ref = soc.design.reference
    if ref and ref.die_area_mm2 and (ref.process_node in (None, soc.node.id)):
        verdict = ("not comparable: this is a lower bound" if not soc.complete
                   else f"{soc.die_area_mm2 / ref.die_area_mm2 - 1:+.0%}")
        lines.append(f"reference: {ref.die_area_mm2:g} mm^2 / "
                     f"{ref.transistors_billion or '?'} B -- {verdict}")
    for fmt in ("int8", "fp32"):
        tops = soc.peak_tops(fmt)
        if tops:
            lines.append(f"peak {fmt} (dense): {tops:.1f} TOPS")
    if soc.off_reference_clocks:
        lines.append(f"provisional clocks (no speed relation to {soc.node.id}): "
                     f"{', '.join(soc.off_reference_clocks)}")
    return lines


def _table(rows: List[dict]) -> str:
    if not rows:
        return "(none)\n"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    out = io.StringIO()
    out.write("  ".join(k.ljust(widths[k]) for k in keys) + "\n")
    out.write("  ".join("-" * widths[k] for k in keys) + "\n")
    for r in rows:
        out.write("  ".join(str(r[k]).ljust(widths[k]) for k in keys) + "\n")
    return out.getvalue()


def _render_text(soc: SoCInstance, sources: bool) -> str:
    out = io.StringIO()
    out.write("=== SoC design ===\n\n" + "\n".join(_summary(soc)) + "\n\n")
    out.write("--- blocks ---\n" + _table(_block_rows(soc)) + "\n")
    classes = soc.area_by_class()
    if classes:
        out.write("--- priced area by library ---\n")
        for cc, area in classes.items():
            out.write(f"  {cc:16} {area:9.2f} mm^2\n")
        out.write("\n")
    if soc.gaps:
        out.write("--- unanchored silicon (no public figure; not estimated) ---\n")
        for block, line in soc.gaps:
            out.write(f"  {block}.{line.name} [{line.circuit_class.value}]\n")
            if sources:
                out.write(f"      {line.source}\n")
        out.write("\n")
    if sources:
        out.write("--- sources of priced lines ---\n")
        for b in soc.blocks:
            for line in b.lines:
                if line.anchored:
                    out.write(f"  {b.name}.{line.name}: {line.source}\n")
    return out.getvalue()


def _md_table(rows: List[dict]) -> List[str]:
    """A Markdown table, or nothing for no rows."""
    if not rows:
        return []
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    return lines + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]


def _render_md(soc: SoCInstance, sources: bool) -> str:
    lines = [f"## {soc.design.name} at {soc.node.id}", ""]
    lines += [f"- {s}" for s in _summary(soc)] + [""]
    lines += _md_table(_block_rows(soc))
    if soc.gaps:
        lines += ["", "### Unanchored silicon", ""]
        lines += [f"- `{b}.{l.name}` ({l.circuit_class.value})"
                  + (f": {l.source}" if sources else "") for b, l in soc.gaps]
    return "\n".join(lines) + "\n"


def _render_json(soc: SoCInstance) -> str:
    return json.dumps({
        "design": soc.design.id,
        "node": soc.node.id,
        "complete": soc.complete,
        "die_area_mm2": soc.die_area_mm2,
        "die_area_is_lower_bound": not soc.complete,
        "block_area_mm2": soc.block_area_mm2,
        "transistors_billion": soc.transistors_billion,
        "area_by_class": soc.area_by_class(),
        "peak_tops_dense": {f: soc.peak_tops(f) for f in ("int8", "fp32") if soc.peak_tops(f)},
        "blocks": _block_rows(soc),
        "gaps": [{"block": b, "line": l.name, "circuit_class": l.circuit_class.value,
                  "source": l.source} for b, l in soc.gaps],
        "provisional_clocks": list(soc.off_reference_clocks),
    }, indent=2)


def _render_csv(rows: List[dict]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv_writer(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show an SoC design composed at a process node (graphs#269)."
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--design", help="Design id from soc_designs/designs/")
    what.add_argument("--list-ip", action="store_true", help="List the IP library instead")
    parser.add_argument("--node", help="Process node (default: the design's own)")
    parser.add_argument("--sources", action="store_true",
                        help="Cite the source of every silicon line, priced and unanchored")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    try:
        library = load_ip_library()
        designs = load_designs()
    except (ValueError, yaml.YAMLError) as exc:  # pydantic's ValidationError is a ValueError
        print(f"error: SoC catalog failed to load: {exc}", file=sys.stderr)
        return 2
    fmt = detect_format(args.output)

    if args.list_ip:
        rows = _ip_rows(library)
        if fmt == "json":
            payload = json.dumps(rows, indent=2)
        elif fmt == "csv":
            payload = _render_csv(rows)
        elif fmt == "md":
            payload = "\n".join([f"## IP library ({len(rows)})", ""] + _md_table(rows)) + "\n"
        else:
            payload = f"=== IP library ({len(rows)}) ===\n\n" + _table(rows)
        write_report(payload, args.output)
        return 0

    if args.design not in designs:
        print(f"error: no SoC design {args.design!r}. Available: {', '.join(sorted(designs))}",
              file=sys.stderr)
        return 2
    try:
        soc = compose_soc(designs[args.design], library, load_process_nodes(), args.node)
    except (KeyError, ValueError, SiliconMathError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if fmt == "json":
        payload = _render_json(soc)
    elif fmt == "csv":
        payload = _render_csv(_block_rows(soc))
    elif fmt == "md":
        payload = _render_md(soc, args.sources)
    else:
        payload = _render_text(soc, args.sources)
    write_report(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
