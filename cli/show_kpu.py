#!/usr/bin/env python
"""
KPU SKU Detail Inspector

Prints the full breakdown of one KPU ComputeProduct: identity, packaging,
per-die structure (process node, die size, transistors, silicon_bin,
clocks), the KPU block (tiles + NoC + memory), performance roll-up,
power profiles with cooling refs, market.

Pair with ``list_kpus.py`` for the catalog-level view.

Migrated from the legacy ``KPUEntry`` view to the unified
``ComputeProduct`` view. Output for KPU monolithic SKUs is content-
identical to the legacy view (same numbers, same structure); the
underlying types and field paths differ. JSON output is now the
``ComputeProduct`` schema rather than the legacy ``KPUEntry`` schema.

Usage:
    python cli/show_kpu.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp
    python cli/show_kpu.py kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc --output t768.json
    python cli/show_kpu.py kpu_t64_32x32_lp5x4_16nm_tsmc_ffp --output t64.md
"""

import argparse
import csv
import io
import json
import os
import sys
from typing import Optional

from embodied_schemas import ComputeProduct, PackagingKind, load_process_nodes
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.hardware.compute_product_loader import load_compute_products_unified


def _kpu_block(cp: ComputeProduct):
    """Return the KPUBlock from a monolithic-KPU ComputeProduct.

    v1 KPU products always have one Die with one KPUBlock; this helper
    pulls it out so callers don't have to walk ``cp.dies[0].blocks[0]``
    each time. Future chiplet KPU products will need an iteration."""
    return cp.dies[0].blocks[0]


def _render_csv(cp: ComputeProduct) -> str:
    """Single-row CSV with the headline-numbers a SKU author tracks.

    Detail-view CSV is awkward (ComputeProduct has nested structures), so
    this renders a flat row of the most-comparable scalars. JSON is the
    right format for the full nested view; CSV is here for spreadsheet
    interop on the headline metrics."""
    block = _kpu_block(cp)
    die = cp.dies[0]
    total_pes = sum(t.total_pes for t in block.tiles)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "name", "vendor", "process_node_id",
        "total_tiles", "total_pes",
        "die_size_mm2", "transistors_billion",
        "default_tdp_w", "default_clock_mhz",
        "int8_tops", "bf16_tflops", "fp32_tflops", "int4_tops",
        "memory_type", "memory_size_gb", "memory_bandwidth_gbps",
    ])
    default_clock = next(
        (p.clock_mhz for p in cp.power.thermal_profiles
         if p.name == cp.power.default_thermal_profile),
        0.0,
    )
    writer.writerow([
        cp.id, cp.name, cp.vendor, die.process_node_id,
        block.total_tiles, total_pes,
        die.die_size_mm2, die.transistors_billion,
        cp.power.tdp_watts, default_clock,
        cp.performance.int8_tops, cp.performance.bf16_tflops,
        cp.performance.fp32_tflops, cp.performance.int4_tops or "",
        block.memory.memory_type.value,
        block.memory.memory_size_gb,
        block.memory.memory_bandwidth_gbps,
    ])
    return buf.getvalue()


def _render_text(cp: ComputeProduct, node: Optional[ProcessNodeEntry]) -> str:
    out = []
    out.append(f"=== KPU SKU: {cp.id} ===")
    out.append(f"  Name:            {cp.name}")
    out.append(f"  Vendor:          {cp.vendor}")
    die = cp.dies[0]
    out.append(f"  Process node:    {die.process_node_id}")
    out.append(f"  Last updated:    {cp.last_updated}")
    out.append("")

    # Die roll-up
    block = _kpu_block(cp)
    out.append("--- Die (roll-up) ---")
    out.append(f"  Architecture:    {block.kind.value.upper()}")
    if node is not None:
        out.append(
            f"  Foundry / node:  {node.foundry.value} {node.node_name} "
            f"({node.node_nm} nm)"
        )
    else:
        out.append(f"  Foundry / node:  {die.process_node_id} (process node not in catalog)")
    out.append(f"  Transistors:     {die.transistors_billion:.2f} B")
    out.append(f"  Die size:        {die.die_size_mm2:.1f} mm^2")
    if cp.packaging.kind != PackagingKind.MONOLITHIC:
        out.append(f"  Chiplet:         yes ({cp.packaging.num_dies} dies)")
    out.append("")

    # Architecture (KPU block)
    out.append("--- Architecture ---")
    out.append(f"  Total tiles:     {block.total_tiles}")
    out.append(f"  Multi-precision: {', '.join(block.multi_precision_alu)}")
    out.append("")
    out.append("  Tile classes:")
    out.append(
        f"    {'tile_type':18s} {'num':>5s} {'PE array':>10s} {'PEs/tile':>9s} "
        f"{'lib':>16s}  ops/tile/clock"
    )
    total_pes = 0
    for t in block.tiles:
        ops_str = ", ".join(f"{p}={int(v)}" for p, v in t.ops_per_tile_per_clock.items())
        total_pes += t.total_pes
        out.append(
            f"    {t.tile_type:18s} {t.num_tiles:>5d} "
            f"{t.pe_array_rows:>4d}x{t.pe_array_cols:<5d} "
            f"{t.pes_per_tile:>9d} {t.pe_circuit_class.value:>16s}  {ops_str}"
        )
    out.append(f"  Total PEs:       {total_pes}")
    out.append("")
    out.append(
        f"  NoC: {block.noc.topology} {block.noc.mesh_rows}x{block.noc.mesh_cols}, "
        f"{block.noc.flit_bytes}-byte flits, "
        f"router_lib={block.noc.router_circuit_class.value}"
    )
    out.append(
        f"  Memory: {block.memory.memory_type.value} "
        f"{block.memory.memory_size_gb:.0f} GB, "
        f"{block.memory.memory_bandwidth_gbps:.0f} GB/s, "
        f"{block.memory.memory_bus_bits}-bit, "
        f"{block.memory.memory_controllers} controllers"
    )
    out.append(
        f"  L1: {block.memory.l1_kib_per_pe} KiB/PE  "
        f"L2: {block.memory.l2_kib_per_tile} KiB/tile  "
        f"L3: {block.memory.l3_kib_per_tile} KiB/tile  "
        f"(L3 total: {block.memory.l3_kib_per_tile * block.total_tiles / 1024:.1f} MiB)"
    )
    out.append("")

    # Silicon bin (under die now)
    out.append("--- Silicon bin (per-block transistor decomposition) ---")
    out.append(f"  {'block':20s} {'circuit_class':18s} {'kind':>16s}  source")
    for b in die.silicon_bin.blocks:
        ts = b.transistor_source
        if ts.kind.value == "fixed":
            src = f"{ts.mtx} Mtx fixed"
        else:
            src = f"{ts.per_unit_mtx} Mtx/unit, ref={ts.count_ref}"
        out.append(
            f"  {b.name:20s} {b.circuit_class.value:18s} {ts.kind.value:>16s}  {src}"
        )
    out.append("")

    # Performance (top-level roll-up)
    out.append("--- Performance (roll-up) ---")
    out.append(f"  INT8:  {cp.performance.int8_tops:>8.1f} TOPS")
    out.append(f"  BF16:  {cp.performance.bf16_tflops:>8.1f} TFLOPS")
    out.append(f"  FP32:  {cp.performance.fp32_tflops:>8.1f} TFLOPS")
    if cp.performance.int4_tops is not None:
        out.append(f"  INT4:  {cp.performance.int4_tops:>8.1f} TOPS")
    out.append("")

    # Clocks (under die now)
    out.append("--- Clocks ---")
    out.append(f"  Base:  {die.clocks.base_clock_mhz} MHz")
    out.append(f"  Boost: {die.clocks.boost_clock_mhz} MHz")
    out.append("")

    # Power + thermal profiles
    out.append("--- Power ---")
    out.append(f"  Default profile: {cp.power.default_thermal_profile}")
    out.append(f"  TDP (default):   {cp.power.tdp_watts} W")
    out.append(f"  Max:             {cp.power.max_power_watts} W")
    out.append(f"  Min:             {cp.power.min_power_watts} W")
    if cp.power.idle_power_watts is not None:
        out.append(f"  Idle:            {cp.power.idle_power_watts} W")
    out.append("")
    out.append("  Thermal profiles:")
    out.append(f"    {'profile':10s} {'TDP':>5s} {'clock':>9s}  cooling solution")
    for tp in cp.power.thermal_profiles:
        out.append(
            f"    {tp.name:10s} {tp.tdp_watts:>5.0f} {tp.clock_mhz:>7.0f} MHz  "
            f"-> {tp.cooling_solution_id}"
        )
    out.append("")

    # Market
    out.append("--- Market ---")
    out.append(f"  Target:          {cp.market.target_market}")
    out.append(f"  Tier:            {cp.market.model_tier}")
    out.append(f"  Family:          {cp.market.product_family}")
    if cp.market.launch_date:
        out.append(f"  Launch:          {cp.market.launch_date}")
    if cp.market.launch_msrp_usd is not None:
        out.append(f"  MSRP:            ${cp.market.launch_msrp_usd:,.0f}")
    out.append(f"  Available:       {'yes' if cp.market.is_available else 'no'}")
    out.append("")

    if cp.notes:
        out.append("--- Notes ---")
        out.append(cp.notes.rstrip())
        out.append("")
    return "\n".join(out)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {
        "json": "json", "csv": "csv", "md": "md",
        "markdown": "md", "txt": "text",
    }.get(ext, "text")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show full spec of one KPU ComputeProduct."
    )
    parser.add_argument("kpu_id", help="KPU SKU id, e.g., kpu_t256_32x32_lp5x16_16nm_tsmc_ffp")
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        cps = load_compute_products_unified()
        nodes = load_process_nodes()
    except Exception as exc:
        print(f"error: failed to load catalog: {exc}", file=sys.stderr)
        return 1

    cp = cps.get(args.kpu_id)
    if cp is None:
        print(
            f"error: no KPU SKU with id={args.kpu_id!r}. "
            f"Available: {', '.join(sorted(cps))}",
            file=sys.stderr,
        )
        return 1

    node = nodes.get(cp.dies[0].process_node_id)

    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = json.dumps(cp.model_dump(mode="json"), indent=2) + "\n"
    elif fmt == "csv":
        rendered = _render_csv(cp)
    elif fmt == "md":
        # Reuse text but wrap in code block; keeps the inspector simple
        rendered = "```\n" + _render_text(cp, node) + "```\n"
    else:
        rendered = _render_text(cp, node) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
