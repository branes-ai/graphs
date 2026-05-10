#!/usr/bin/env python
"""
KPU Floorplan Visualization (Stage 8)

Renders a KPU SKU's heuristic 2D floorplan as ASCII art.

Two views are available:

  --view architectural (default)
      Bins blocks by *role*: COMPUTE tile (PE+L1+L2), MEMORY tile (L3),
      MEMORY_CONTROLLER (DRAM PHY), IO_PAD, CONTROL. Renders the
      checkerboard layout that pairs each compute tile with a memory
      tile. Includes a what-if table showing what the die would look
      like if every compute tile were one class (cost of mixing
      Matrix/BF16/INT8).

  --view circuit
      Bins blocks by *silicon library*: HP_LOGIC, BALANCED_LOGIC,
      SRAM_HD, ANALOG, IO. Useful for seeing where dense storage vs
      fast logic vs analog macros end up on the die. This is the
      original Stage 8a renderer.

Future work: SVG / HTML renderer that overlays the NoC topology
(2D mesh / ring / CLOS) on top of the placed blocks.

Usage:
    python cli/show_floorplan.py stillwater_kpu_t256
    python cli/show_floorplan.py stillwater_kpu_t768 --width 100
    python cli/show_floorplan.py stillwater_kpu_t256 --view circuit
    python cli/show_floorplan.py stillwater_kpu_t64 --output fp.txt
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

from embodied_schemas import load_kpus, load_process_nodes
from embodied_schemas.process_node import CircuitClass

from graphs.hardware.silicon_floorplan import (
    ArchitecturalFloorplan,
    ArchTile,
    Floorplan,
    FloorplanBlock,
    TileRole,
    derive_kpu_architectural_floorplan,
    derive_kpu_floorplan,
)


# ---------------------------------------------------------------------------
# Architectural view: glyphs by tile role
# ---------------------------------------------------------------------------

# One glyph per architectural role. The whole point of this view is to
# show the checkerboard structure, so compute / memory get distinct
# easily-distinguishable glyphs.
_ARCH_GLYPH_BY_ROLE = {
    TileRole.COMPUTE: "C",
    TileRole.MEMORY: "M",
    TileRole.MEMORY_CONTROLLER: "D",  # D for DRAM controller / PHY
    TileRole.IO_PAD: ":",
    TileRole.CONTROL: "*",
}


def _arch_glyph_for(block: ArchTile) -> str:
    return _ARCH_GLYPH_BY_ROLE.get(block.role, "?")


# ---------------------------------------------------------------------------
# Circuit-class view: glyphs by silicon library (Stage 8a)
# ---------------------------------------------------------------------------

_CIRCUIT_GLYPH_BY_CLASS = {
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


def _circuit_glyph_for(block: FloorplanBlock) -> str:
    return _CIRCUIT_GLYPH_BY_CLASS.get(block.circuit_class, "*")


# ---------------------------------------------------------------------------
# Generic ASCII grid rasterizer
# ---------------------------------------------------------------------------

def _render_blocks_to_ascii(
    blocks_with_glyphs,  # list[(rect, glyph)] where rect = (x, y, w, h)
    die_width_mm: float,
    die_height_mm: float,
    *,
    char_width: int = 80,
) -> str:
    """Rasterize a list of (rectangle, glyph) pairs into an ASCII grid.

    Terminal characters are roughly 2:1 tall:wide; the height scales
    accordingly so the printed die preserves its true aspect ratio.
    """
    if die_width_mm <= 0 or die_height_mm <= 0:
        return "(empty floorplan)"
    mm_per_col = die_width_mm / char_width
    mm_per_row = mm_per_col * 2.0
    rows = max(1, int(die_height_mm / mm_per_row + 0.5))
    grid = [[" "] * char_width for _ in range(rows)]
    for (x, y, w, h), glyph in blocks_with_glyphs:
        col_lo = max(0, int(x / mm_per_col))
        col_hi = min(char_width, int((x + w) / mm_per_col + 0.5))
        # Flip y: grid row 0 = top of die (high y)
        y_top = y + h
        row_lo = max(0, int((die_height_mm - y_top) / mm_per_row))
        row_hi = min(rows, int((die_height_mm - y) / mm_per_row + 0.5))
        for r in range(row_lo, row_hi):
            for c in range(col_lo, col_hi):
                grid[r][c] = glyph
    return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# Architectural view: text summary + ASCII
# ---------------------------------------------------------------------------

def _render_architectural_summary(fp: ArchitecturalFloorplan) -> str:
    lines = [
        f"Floorplan (architectural): {fp.sku_name} ({fp.sku_id})",
        f"  die:               {fp.die_width_mm:.2f} x {fp.die_height_mm:.2f} mm "
        f"= {fp.die_area_mm2:.1f} mm^2",
        f"  unified pitch:     {fp.unified_pitch_mm:.3f} mm",
        f"  C/M pitch ratio:   {fp.compute_memory_pitch_ratio:.2f}x"
        f"  (compute vs memory tile pitch)",
        f"  total whitespace:  {fp.whitespace_fraction()*100:.1f}% "
        f"({fp.whitespace_mm2():.1f} mm^2)",
        f"  blocks:            compute={len(fp.compute_tiles())}, "
        f"memory={len(fp.memory_tiles())}, "
        f"controllers={fp.num_memory_controllers}",
        "",
        "Compute tile classes (PE + L2 per tile):",
    ]
    if fp.compute_summaries:
        max_cp = max(s.pitch_mm for s in fp.compute_summaries.values())
        for tc, s in sorted(fp.compute_summaries.items()):
            mark = " <- max" if abs(s.pitch_mm - max_cp) < 1e-6 else ""
            lines.append(
                f"  {tc:14s}  N={s.num_tiles:4d}  "
                f"PE={s.pe_area_mm2:.4f}  L2={s.l2_area_mm2:.4f}  "
                f"total={s.total_area_mm2:.4f} mm^2  "
                f"pitch={s.pitch_mm:.3f} mm  "
                f"whitespace={s.class_whitespace_mm2:.1f} mm^2{mark}"
            )
    lines.append("")
    lines.append("Memory tiles (L3 per tile):")
    s = fp.memory_summary
    lines.append(
        f"  {'memory':14s}  N={s.num_tiles:4d}  "
        f"L3={s.l3_area_mm2:.4f} mm^2  "
        f"pitch={s.pitch_mm:.3f} mm  "
        f"whitespace={s.class_whitespace_mm2:.1f} mm^2"
    )
    lines.append("")
    lines.append(
        "What-if -- die area if every compute tile were one class:"
    )
    if fp.what_if:
        # Anchor: the actual mixed-class die area
        anchor = fp.die_area_mm2
        for wi in sorted(fp.what_if, key=lambda w: w.die_area_mm2):
            delta = wi.die_area_mm2 - anchor
            pct = (delta / anchor * 100.0) if anchor > 0 else 0.0
            sign = "+" if delta >= 0 else "-"
            lines.append(
                f"  if all {wi.tile_class:14s}  pitch={wi.unified_pitch_mm:.3f}  "
                f"die={wi.die_area_mm2:.1f} mm^2  "
                f"({sign}{abs(pct):.0f}% vs actual)  "
                f"whitespace={wi.whitespace_fraction*100:.1f}%"
            )
    return "\n".join(lines)


def _arch_glyph_legend(fp: ArchitecturalFloorplan) -> str:
    used = {b.role for b in fp.blocks}
    parts = []
    for role, glyph in _ARCH_GLYPH_BY_ROLE.items():
        if role in used:
            parts.append(f"{glyph}={role.value}")
    return "  Legend: " + "  ".join(parts)


def render_architectural_ascii(
    fp: ArchitecturalFloorplan, *, char_width: int = 80
) -> str:
    pairs = [
        ((b.x_mm, b.y_mm, b.width_mm, b.height_mm), _arch_glyph_for(b))
        for b in fp.blocks
    ]
    return _render_blocks_to_ascii(
        pairs, fp.die_width_mm, fp.die_height_mm, char_width=char_width
    )


def _arch_to_dict(fp: ArchitecturalFloorplan) -> dict:
    return {
        "view": "architectural",
        "sku_id": fp.sku_id,
        "sku_name": fp.sku_name,
        "die_width_mm": fp.die_width_mm,
        "die_height_mm": fp.die_height_mm,
        "die_area_mm2": fp.die_area_mm2,
        "unified_pitch_mm": fp.unified_pitch_mm,
        "compute_memory_pitch_ratio": fp.compute_memory_pitch_ratio,
        "whitespace_mm2": fp.whitespace_mm2(),
        "whitespace_fraction": fp.whitespace_fraction(),
        "num_memory_controllers": fp.num_memory_controllers,
        "compute_summaries": {
            tc: {
                "num_tiles": s.num_tiles,
                "pe_area_mm2": s.pe_area_mm2,
                "l2_area_mm2": s.l2_area_mm2,
                "total_area_mm2": s.total_area_mm2,
                "pitch_mm": s.pitch_mm,
                "whitespace_per_tile_mm2": s.whitespace_per_tile_mm2,
                "class_whitespace_mm2": s.class_whitespace_mm2,
            } for tc, s in fp.compute_summaries.items()
        },
        "memory_summary": {
            "num_tiles": fp.memory_summary.num_tiles,
            "l3_area_mm2": fp.memory_summary.l3_area_mm2,
            "pitch_mm": fp.memory_summary.pitch_mm,
            "whitespace_per_tile_mm2":
                fp.memory_summary.whitespace_per_tile_mm2,
            "class_whitespace_mm2": fp.memory_summary.class_whitespace_mm2,
        },
        "what_if": [
            {
                "tile_class": wi.tile_class,
                "unified_pitch_mm": wi.unified_pitch_mm,
                "mesh_area_mm2": wi.mesh_area_mm2,
                "die_area_mm2": wi.die_area_mm2,
                "whitespace_mm2": wi.whitespace_mm2,
                "whitespace_fraction": wi.whitespace_fraction,
            } for wi in fp.what_if
        ],
        "blocks": [
            {
                "name": b.name,
                "role": b.role.value,
                "x_mm": b.x_mm, "y_mm": b.y_mm,
                "width_mm": b.width_mm, "height_mm": b.height_mm,
                "tile_class": b.tile_class,
                "pe_area_mm2": b.pe_area_mm2,
                "l2_area_mm2": b.l2_area_mm2,
                "l3_area_mm2": b.l3_area_mm2,
                "notes": b.notes,
            } for b in fp.blocks
        ],
        "notes": fp.notes,
    }


# ---------------------------------------------------------------------------
# Circuit-class view: text summary + ASCII (Stage 8a; kept as --view circuit)
# ---------------------------------------------------------------------------

def _render_circuit_summary(fp: Floorplan) -> str:
    lines = [
        f"Floorplan (circuit-class): {fp.sku_name} ({fp.sku_id})",
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


def render_circuit_ascii(fp: Floorplan, *, char_width: int = 80) -> str:
    pairs = [
        ((b.x_mm, b.y_mm, b.width_mm, b.height_mm), _circuit_glyph_for(b))
        for b in fp.blocks
    ]
    return _render_blocks_to_ascii(
        pairs, fp.die_width_mm, fp.die_height_mm, char_width=char_width
    )


def _circuit_glyph_legend(fp: Floorplan) -> str:
    used = {b.circuit_class for b in fp.blocks}
    parts = []
    for cc, glyph in _CIRCUIT_GLYPH_BY_CLASS.items():
        if cc in used:
            parts.append(f"{glyph}={cc.value}")
    return "  Legend: " + "  ".join(parts)


def _circuit_to_dict(fp: Floorplan) -> dict:
    return {
        "view": "circuit",
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
                "x_mm": b.x_mm, "y_mm": b.y_mm,
                "width_mm": b.width_mm, "height_mm": b.height_mm,
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
        help=(
            "KPU SKU id (e.g., stillwater_kpu_t256). Omit with --list to "
            "list available ids."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available KPU SKU ids and exit.",
    )
    parser.add_argument(
        "--view", choices=["architectural", "circuit"],
        default="architectural",
        help=(
            "Floorplan view. 'architectural' (default) groups blocks by "
            "role (compute/memory/controller/io/control) and shows the "
            "checkerboard layout + what-if die estimates. 'circuit' "
            "groups by silicon library (HP_LOGIC/BAL/SRAM/ANALOG/IO)."
        ),
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
        help=(
            "Write to a file (extension .json selects JSON unless --json "
            "is also passed)."
        ),
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
    node = nodes[sku.process_node_id]

    use_json = args.json or (
        args.output and args.output.lower().endswith(".json")
    )

    if args.view == "architectural":
        fp_arch = derive_kpu_architectural_floorplan(sku, node)
        if use_json:
            payload = json.dumps(_arch_to_dict(fp_arch), indent=2)
        else:
            payload = (
                _render_architectural_summary(fp_arch)
                + "\n\n"
                + render_architectural_ascii(fp_arch, char_width=args.width)
                + "\n" + _arch_glyph_legend(fp_arch)
            )
    else:  # circuit
        fp_circ = derive_kpu_floorplan(sku, node)
        if use_json:
            payload = json.dumps(_circuit_to_dict(fp_circ), indent=2)
        else:
            payload = (
                _render_circuit_summary(fp_circ)
                + "\n\n"
                + render_circuit_ascii(fp_circ, char_width=args.width)
                + "\n" + _circuit_glyph_legend(fp_circ)
            )

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"info: wrote {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
