#!/usr/bin/env python
"""
KPU Floorplan Visualization (Stage 8)

Renders the heuristic 2D floorplan of a KPU SKU as ASCII art so the
silicon decomposition can be eyeballed. Useful for:

  * Spot-checking tile pitches across classes (the geometry validators
    flag mismatches; this tool shows them).
  * Sanity-checking PHY / IO / control placement vs the mesh size.
  * Generating reproducible artifacts for design reviews.

The ASCII renderer is deliberately simple (no matplotlib / SVG dep).
The same Floorplan dataclass can be fed to a future SVG renderer
without touching this module.

Usage:
    python cli/show_floorplan.py stillwater_kpu_t256
    python cli/show_floorplan.py stillwater_kpu_t768 --width 100
    python cli/show_floorplan.py stillwater_kpu_t64 --output floorplan.txt
    python cli/show_floorplan.py stillwater_kpu_t128 --json --output fp.json

Exit codes:
    0 = rendered successfully
    2 = SKU not found / argument error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from embodied_schemas import load_kpus, load_process_nodes
from embodied_schemas.process_node import CircuitClass

from graphs.hardware.silicon_floorplan import (
    Floorplan,
    FloorplanBlock,
    derive_kpu_floorplan,
)


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

# Per-circuit-class glyphs. Choice rationale:
#   - I/O ring: colon (rare in identifiers, pads well at edges)
#   - SRAM (any flavor): hash (dense visual weight matches dense storage)
#   - Logic (HP/BAL/LP/ULL): letter giving the speed tier at a glance
#   - Analog: tilde (suggests sinusoid / continuous signal)
#   - Mixed: question mark (ambiguous / heterogeneous)
_GLYPH_BY_CLASS = {
    CircuitClass.HP_LOGIC: "H",
    CircuitClass.BALANCED_LOGIC: "B",
    CircuitClass.LP_LOGIC: "L",
    CircuitClass.ULL_LOGIC: "U",
    CircuitClass.SRAM_HD: "#",
    CircuitClass.SRAM_HC: "#",
    CircuitClass.SRAM_HP: "#",
    CircuitClass.ANALOG: "~",
    CircuitClass.IO: ":",
    CircuitClass.MIXED: "?",
}


def _glyph_for(block: FloorplanBlock) -> str:
    return _GLYPH_BY_CLASS.get(block.circuit_class, "*")


def render_ascii(fp: Floorplan, *, char_width: int = 80) -> str:
    """Render the floorplan as an ASCII grid.

    The grid is ``char_width`` characters wide. Height scales to keep
    the die's true aspect ratio (one character cell ~= 0.5 mm wide *
    1.0 mm tall, since terminal characters are ~2x tall).
    """
    if fp.die_width_mm <= 0 or fp.die_height_mm <= 0:
        return "(empty floorplan)"

    # Char cell aspect: most terminal fonts are about 2:1 tall:wide. So
    # vertical scaling per row covers 2x the horizontal scaling per column.
    mm_per_col = fp.die_width_mm / char_width
    mm_per_row = mm_per_col * 2.0
    rows = max(1, int(fp.die_height_mm / mm_per_row + 0.5))

    # Render top-down: row 0 = top of the die. Floorplan uses
    # bottom-left origin; flip when projecting to grid coordinates.
    grid = [[" "] * char_width for _ in range(rows)]
    for block in fp.blocks:
        col_lo = max(0, int(block.x_mm / mm_per_col))
        col_hi = min(char_width, int((block.x_mm + block.width_mm) / mm_per_col + 0.5))
        # Flip y: top of grid (row 0) corresponds to top of die (high y)
        y_top = block.y_mm + block.height_mm
        row_lo = max(0, int((fp.die_height_mm - y_top) / mm_per_row))
        row_hi = min(rows, int((fp.die_height_mm - block.y_mm) / mm_per_row + 0.5))
        glyph = _glyph_for(block)
        for r in range(row_lo, row_hi):
            for c in range(col_lo, col_hi):
                grid[r][c] = glyph
    return "\n".join("".join(row) for row in grid)


def _glyph_legend(fp: Floorplan) -> str:
    """One-line legend mapping glyphs back to circuit classes used in the floorplan."""
    used = {block.circuit_class for block in fp.blocks}
    parts = []
    for cc, glyph in _GLYPH_BY_CLASS.items():
        if cc in used:
            parts.append(f"{glyph}={cc.value}")
    return "  Legend: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Text summary (printed alongside the ASCII grid)
# ---------------------------------------------------------------------------

def _render_summary(fp: Floorplan) -> str:
    lines = [
        f"Floorplan: {fp.sku_name} ({fp.sku_id})",
        f"  die:               {fp.die_width_mm:.2f} x {fp.die_height_mm:.2f} mm "
        f"= {fp.die_area_mm2:.1f} mm^2",
        f"  unified pitch:     {fp.unified_pitch_mm:.3f} mm",
        f"  whitespace:        {fp.whitespace_fraction()*100:.1f}% "
        f"({fp.whitespace_mm2():.1f} mm^2)",
        f"  blocks:            {len(fp.blocks)} "
        f"(compute tiles: {len(fp.compute_tiles())})",
        "",
        "Per-tile-class pitches:",
    ]
    if fp.tile_pitches:
        max_pitch = max(tp.pitch_mm for tp in fp.tile_pitches.values())
        for tile_class, tp in sorted(fp.tile_pitches.items()):
            ratio = max_pitch / tp.pitch_mm if tp.pitch_mm > 0 else float("inf")
            mark = " <- max" if abs(tp.pitch_mm - max_pitch) < 1e-6 else (
                f"  ({ratio:.2f}x smaller)" if ratio > 1.05 else ""
            )
            lines.append(
                f"  {tile_class:18s}  PE={tp.pe_area_mm2:.4f}  "
                f"SRAM={tp.sram_area_mm2:.4f}  total={tp.total_area_mm2:.4f} mm^2  "
                f"pitch={tp.pitch_mm:.3f} mm{mark}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON serialization (machine-readable output)
# ---------------------------------------------------------------------------

def _floorplan_to_dict(fp: Floorplan) -> dict:
    return {
        "sku_id": fp.sku_id,
        "sku_name": fp.sku_name,
        "die_width_mm": fp.die_width_mm,
        "die_height_mm": fp.die_height_mm,
        "die_area_mm2": fp.die_area_mm2,
        "unified_pitch_mm": fp.unified_pitch_mm,
        "whitespace_mm2": fp.whitespace_mm2(),
        "whitespace_fraction": fp.whitespace_fraction(),
        "tile_pitches": {
            tc: {
                "pe_area_mm2": tp.pe_area_mm2,
                "sram_area_mm2": tp.sram_area_mm2,
                "total_area_mm2": tp.total_area_mm2,
                "pitch_mm": tp.pitch_mm,
            } for tc, tp in fp.tile_pitches.items()
        },
        "blocks": [
            {
                "name": b.name,
                "circuit_class": b.circuit_class.value,
                "x_mm": b.x_mm,
                "y_mm": b.y_mm,
                "width_mm": b.width_mm,
                "height_mm": b.height_mm,
                "is_compute_tile": b.is_compute_tile,
                "tile_class": b.tile_class,
                "notes": b.notes,
            } for b in fp.blocks
        ],
        "notes": fp.notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a KPU SKU's heuristic floorplan as ASCII art (or JSON)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sku_id",
        nargs="?",
        help="KPU SKU id (e.g., stillwater_kpu_t256). Omit with --list to list ids.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available KPU SKU ids and exit.",
    )
    parser.add_argument(
        "--width", type=int, default=80,
        help="ASCII grid width in characters (default 80).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON (machine-readable) instead of ASCII art.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write to a file (extension .json/.txt selects format if --json not set).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print extra diagnostic information.",
    )
    args = parser.parse_args()

    kpus = load_kpus()
    if args.list:
        for kid in sorted(kpus):
            print(kid)
        return 0
    if not args.sku_id:
        parser.error("sku_id is required (or pass --list)")
        return 2
    if args.sku_id not in kpus:
        print(f"error: unknown SKU id {args.sku_id!r}", file=sys.stderr)
        print(f"hint: try one of {sorted(kpus)}", file=sys.stderr)
        return 2

    sku = kpus[args.sku_id]
    nodes = load_process_nodes()
    if sku.process_node_id not in nodes:
        print(
            f"error: SKU references unknown process_node_id "
            f"{sku.process_node_id!r}",
            file=sys.stderr,
        )
        return 2

    fp = derive_kpu_floorplan(sku, nodes[sku.process_node_id])

    # Choose output format
    use_json = args.json or (
        args.output and args.output.lower().endswith(".json")
    )

    if use_json:
        payload = json.dumps(_floorplan_to_dict(fp), indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"info: wrote {args.output}", file=sys.stderr)
        else:
            print(payload)
    else:
        text = _render_summary(fp) + "\n\n" + render_ascii(
            fp, char_width=args.width
        ) + "\n" + _glyph_legend(fp)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"info: wrote {args.output}", file=sys.stderr)
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
