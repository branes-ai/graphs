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
    python cli/show_floorplan.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp
    python cli/show_floorplan.py kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc --width 100
    python cli/show_floorplan.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp --view circuit
    python cli/show_floorplan.py kpu_t64_32x32_lp5x4_16nm_tsmc_ffp --output fp.txt
    python cli/show_floorplan.py kpu_t128_32x32_lp5x8_16nm_tsmc_ffp --json --output fp.json

Exit codes:
    0 = rendered successfully
    2 = SKU not found / argument error
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

from embodied_schemas import load_process_nodes
from embodied_schemas.process_node import CircuitClass

from graphs.hardware.compute_product_loader import load_compute_products_unified
from graphs.hardware.silicon_floorplan import (
    ArchitecturalFloorplan,
    ArchTile,
    Floorplan,
    FloorplanBlock,
    MemoryChannel,
    TileRole,
    derive_kpu_architectural_floorplan,
    derive_kpu_floorplan,
)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    """argparse type-checker: reject zero/negative widths early.

    ``_render_blocks_to_ascii`` divides by ``char_width``; a non-positive
    value crashes deep in the rasterizer with no actionable message.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"--width must be a positive integer (got {parsed})"
        )
    return parsed


def _detect_format(output: Optional[str], json_flag: bool) -> str:
    """Map ``--output`` extension to format. Mirrors validate_sku.py.

    Per the project's CLI rules, every ``--output`` accepts JSON, CSV,
    MD, and text via extension auto-detection. ``--json`` overrides
    extension when set.
    """
    if json_flag:
        return "json"
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {
        "json": "json", "csv": "csv", "md": "md",
        "markdown": "md", "txt": "text",
    }.get(ext, "text")


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

# Off-die channel glyphs by edge -- each channel is a strip of these
# characters running parallel to its edge.
_CHANNEL_GLYPH_BY_EDGE = {
    "top": "=",
    "bottom": "=",
    "left": "|",
    "right": "|",
}


def _arch_glyph_for(block: ArchTile) -> str:
    return _ARCH_GLYPH_BY_ROLE.get(block.role, "?")


def _channel_glyph_for(ch: MemoryChannel) -> str:
    return _CHANNEL_GLYPH_BY_EDGE.get(ch.edge, "#")


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
    """Rasterize (rectangle, glyph) pairs into an ASCII grid.

    Terminal characters are roughly 2:1 tall:wide; the height scales
    accordingly so the printed die preserves its true aspect ratio.

    Both edge endpoints use ``floor`` (``int()``) so adjacent
    rectangles share a boundary cell (no gaps, no overlaps) AND every
    tile gets the same visual size. ``round`` was tried earlier but
    produced periodic +1-cell tiles (e.g., 7 tiles of 2 rows + 1 of
    3) because the rounding direction flipped at half-pitch boundaries.
    With floor, tile_size = floor(N*pitch/r) - floor((N-1)*pitch/r) is
    uniform across the mesh.
    """
    if die_width_mm <= 0 or die_height_mm <= 0:
        return "(empty floorplan)"
    mm_per_col = die_width_mm / char_width
    mm_per_row = mm_per_col * 2.0
    rows = max(1, int(die_height_mm / mm_per_row))
    grid = [[" "] * char_width for _ in range(rows)]
    for (x, y, w, h), glyph in blocks_with_glyphs:
        col_lo = max(0, int(x / mm_per_col))
        col_hi = min(char_width, int((x + w) / mm_per_col))
        # Flip y: grid row 0 = top of die (high y)
        y_top = y + h
        row_lo = max(0, int((die_height_mm - y_top) / mm_per_row))
        row_hi = min(rows, int((die_height_mm - y) / mm_per_row))
        # Ensure non-empty rectangles render at least 1 cell -- otherwise
        # thin blocks (IO ring at 0.3mm with mm_per_col 0.23) collapse
        # to col_lo == col_hi and disappear.
        if col_hi == col_lo and w > 0:
            col_hi = min(char_width, col_lo + 1)
        if row_hi == row_lo and h > 0:
            row_hi = min(rows, row_lo + 1)
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
        f"  memory subsystem:  {fp.memory_type}  "
        f"channels={fp.num_memory_controllers}  "
        f"width/ch={fp.per_channel_width_bits}b  "
        f"PHY/ch={fp.per_channel_phy_area_mm2:.2f} mm^2",
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
    if fp.memory_channels:
        # Show the off-die channel glyphs in the legend too
        edges_used = {ch.edge for ch in fp.memory_channels}
        ch_glyphs = sorted({_CHANNEL_GLYPH_BY_EDGE[e] for e in edges_used})
        parts.append(
            f"{'/'.join(ch_glyphs)}=dram_channel ({fp.memory_type} "
            f"{fp.per_channel_width_bits}b)"
        )
    return "  Legend: " + "  ".join(parts)


def render_architectural_ascii(
    fp: ArchitecturalFloorplan, *, char_width: int = 80
) -> str:
    """Render the die + off-die DRAM channels in a single canvas.

    Channels live just outside the die boundary at negative or beyond-
    die-edge coordinates. The renderer expands the canvas to include
    them and translates everything into canvas coordinates so the
    visual MC <-> channel association is preserved.
    """
    # Compute outer margins from channel placements
    top_margin = 0.0
    right_margin = 0.0
    bottom_margin = 0.0
    left_margin = 0.0
    for ch in fp.memory_channels:
        if ch.edge == "top":
            top_margin = max(top_margin, ch.height_mm)
        elif ch.edge == "bottom":
            bottom_margin = max(bottom_margin, ch.height_mm)
        elif ch.edge == "left":
            left_margin = max(left_margin, ch.width_mm)
        elif ch.edge == "right":
            right_margin = max(right_margin, ch.width_mm)

    canvas_w = fp.die_width_mm + left_margin + right_margin
    canvas_h = fp.die_height_mm + bottom_margin + top_margin
    if canvas_w <= 0 or canvas_h <= 0:
        return "(empty floorplan)"

    pairs = []
    # Die blocks: translate by (left_margin, bottom_margin)
    for b in fp.blocks:
        pairs.append((
            (b.x_mm + left_margin, b.y_mm + bottom_margin,
             b.width_mm, b.height_mm),
            _arch_glyph_for(b),
        ))
    # Off-die channels: translate. Channels for 'top' have y_mm >=
    # die_height_mm; 'bottom' have y_mm < 0; etc. Translation aligns
    # them in canvas space.
    for ch in fp.memory_channels:
        pairs.append((
            (ch.x_mm + left_margin, ch.y_mm + bottom_margin,
             ch.width_mm, ch.height_mm),
            _channel_glyph_for(ch),
        ))
    return _render_blocks_to_ascii(
        pairs, canvas_w, canvas_h, char_width=char_width
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
        "memory_type": fp.memory_type,
        "per_channel_width_bits": fp.per_channel_width_bits,
        "per_channel_phy_area_mm2": fp.per_channel_phy_area_mm2,
        "memory_channels": [
            {
                "channel_id": ch.channel_id,
                "memory_type": ch.memory_type,
                "width_bits": ch.width_bits,
                "x_mm": ch.x_mm, "y_mm": ch.y_mm,
                "width_mm": ch.width_mm, "height_mm": ch.height_mm,
                "controller_name": ch.controller_name,
                "edge": ch.edge,
            } for ch in fp.memory_channels
        ],
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
# CSV / Markdown renderers (architectural + circuit views)
# ---------------------------------------------------------------------------

def _arch_to_csv(fp: ArchitecturalFloorplan) -> str:
    """One row per placed block + per memory channel. Header columns:
    sku_id, kind, role/edge, name, tile_class, x_mm, y_mm, width_mm,
    height_mm, area_mm2.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "sku_id", "kind", "role_or_edge", "name", "tile_class",
            "x_mm", "y_mm", "width_mm", "height_mm", "area_mm2",
        ],
    )
    writer.writeheader()
    for b in fp.blocks:
        writer.writerow({
            "sku_id": fp.sku_id, "kind": "block",
            "role_or_edge": b.role.value, "name": b.name,
            "tile_class": b.tile_class or "",
            "x_mm": f"{b.x_mm:.4f}", "y_mm": f"{b.y_mm:.4f}",
            "width_mm": f"{b.width_mm:.4f}", "height_mm": f"{b.height_mm:.4f}",
            "area_mm2": f"{b.area_mm2:.4f}",
        })
    for ch in fp.memory_channels:
        writer.writerow({
            "sku_id": fp.sku_id, "kind": "dram_channel",
            "role_or_edge": ch.edge,
            "name": f"channel_{ch.channel_id}",
            "tile_class": "",
            "x_mm": f"{ch.x_mm:.4f}", "y_mm": f"{ch.y_mm:.4f}",
            "width_mm": f"{ch.width_mm:.4f}", "height_mm": f"{ch.height_mm:.4f}",
            "area_mm2": f"{ch.width_mm * ch.height_mm:.4f}",
        })
    return buf.getvalue()


def _arch_to_md(fp: ArchitecturalFloorplan) -> str:
    lines = [
        f"# Architectural floorplan: `{fp.sku_id}`", "",
        f"- **die**: {fp.die_width_mm:.2f} x {fp.die_height_mm:.2f} mm "
        f"= {fp.die_area_mm2:.1f} mm^2",
        f"- **unified pitch**: {fp.unified_pitch_mm:.3f} mm",
        f"- **C/M pitch ratio**: {fp.compute_memory_pitch_ratio:.2f}x",
        f"- **whitespace**: {fp.whitespace_fraction()*100:.1f}%",
        f"- **memory subsystem**: {fp.memory_type}, "
        f"{fp.num_memory_controllers} channels @ "
        f"{fp.per_channel_width_bits}b "
        f"({fp.per_channel_phy_area_mm2:.2f} mm^2/channel)",
        "",
        "## Compute tile classes (PE + L2)",
        "",
        "| class | N | PE mm^2 | L2 mm^2 | total mm^2 | pitch mm | whitespace mm^2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tc, s in sorted(fp.compute_summaries.items()):
        lines.append(
            f"| `{tc}` | {s.num_tiles} | {s.pe_area_mm2:.4f} | "
            f"{s.l2_area_mm2:.4f} | {s.total_area_mm2:.4f} | "
            f"{s.pitch_mm:.3f} | {s.class_whitespace_mm2:.1f} |"
        )
    s = fp.memory_summary
    lines.extend([
        "",
        "## Memory tile (L3)",
        "",
        f"| N | L3 mm^2 | pitch mm | whitespace mm^2 |",
        "|---:|---:|---:|---:|",
        f"| {s.num_tiles} | {s.l3_area_mm2:.4f} | {s.pitch_mm:.3f} | "
        f"{s.class_whitespace_mm2:.1f} |",
        "",
        "## What-if (all-class-X die area)",
        "",
        "| class | unified pitch mm | die mm^2 | whitespace |",
        "|---|---:|---:|---:|",
    ])
    for wi in sorted(fp.what_if, key=lambda w: w.die_area_mm2):
        lines.append(
            f"| `{wi.tile_class}` | {wi.unified_pitch_mm:.3f} | "
            f"{wi.die_area_mm2:.1f} | {wi.whitespace_fraction*100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _circuit_to_csv(fp: Floorplan) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "sku_id", "name", "circuit_class", "is_compute_tile",
            "tile_class", "x_mm", "y_mm", "width_mm", "height_mm",
            "area_mm2",
        ],
    )
    writer.writeheader()
    for b in fp.blocks:
        writer.writerow({
            "sku_id": fp.sku_id, "name": b.name,
            "circuit_class": b.circuit_class.value,
            "is_compute_tile": b.is_compute_tile,
            "tile_class": b.tile_class or "",
            "x_mm": f"{b.x_mm:.4f}", "y_mm": f"{b.y_mm:.4f}",
            "width_mm": f"{b.width_mm:.4f}", "height_mm": f"{b.height_mm:.4f}",
            "area_mm2": f"{b.area_mm2:.4f}",
        })
    return buf.getvalue()


def _circuit_to_md(fp: Floorplan) -> str:
    lines = [
        f"# Circuit-class floorplan: `{fp.sku_id}`", "",
        f"- **die**: {fp.die_width_mm:.2f} x {fp.die_height_mm:.2f} mm "
        f"= {fp.die_area_mm2:.1f} mm^2",
        f"- **unified pitch**: {fp.unified_pitch_mm:.3f} mm",
        f"- **whitespace**: {fp.whitespace_fraction()*100:.1f}%",
        "",
        "## Per-tile-class pitches",
        "",
        "| tile class | PE mm^2 | SRAM mm^2 | total mm^2 | pitch mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for tc, tp in sorted(fp.tile_pitches.items()):
        lines.append(
            f"| `{tc}` | {tp.pe_area_mm2:.4f} | {tp.sram_area_mm2:.4f} | "
            f"{tp.total_area_mm2:.4f} | {tp.pitch_mm:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


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
            "KPU SKU id (e.g., kpu_t256_32x32_lp5x16_16nm_tsmc_ffp). Omit with --list to "
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
        "--width", type=_positive_int, default=80,
        help="ASCII grid width in characters (default 80, must be > 0).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON (overrides extension auto-detect on --output).",
    )
    parser.add_argument(
        "--output", "-o",
        help=(
            "Write to a file. Format auto-detected from extension: "
            ".json/.csv/.md/.markdown/.txt; default text."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print extra diagnostic information.",
    )
    args = parser.parse_args()

    kpus = load_compute_products_unified()
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
    process_node_id = sku.dies[0].process_node_id
    if process_node_id not in nodes:
        print(
            f"error: SKU references unknown process_node_id "
            f"{process_node_id!r}",
            file=sys.stderr,
        )
        return 2
    node = nodes[process_node_id]

    fmt = _detect_format(args.output, args.json)

    if args.view == "architectural":
        fp_arch = derive_kpu_architectural_floorplan(sku, node)
        if fmt == "json":
            payload = json.dumps(_arch_to_dict(fp_arch), indent=2)
        elif fmt == "csv":
            payload = _arch_to_csv(fp_arch)
        elif fmt == "md":
            payload = _arch_to_md(fp_arch)
        else:
            payload = (
                _render_architectural_summary(fp_arch)
                + "\n\n"
                + render_architectural_ascii(fp_arch, char_width=args.width)
                + "\n" + _arch_glyph_legend(fp_arch)
            )
    else:  # circuit
        fp_circ = derive_kpu_floorplan(sku, node)
        if fmt == "json":
            payload = json.dumps(_circuit_to_dict(fp_circ), indent=2)
        elif fmt == "csv":
            payload = _circuit_to_csv(fp_circ)
        elif fmt == "md":
            payload = _circuit_to_md(fp_circ)
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
