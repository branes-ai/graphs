#!/usr/bin/env python
"""
KPU SKU Detail Inspector

Prints the full breakdown of one KPUEntry: identity, process-node ref,
die rolls, kpu_architecture (tiles + NoC + memory), silicon_bin
decomposition, performance, power profiles with cooling refs, market.

Pair with ``list_kpus.py`` for the catalog-level view.

Usage:
    python cli/show_kpu.py stillwater_kpu_t256
    python cli/show_kpu.py stillwater_kpu_t768 --output t768.json
    python cli/show_kpu.py stillwater_kpu_t64 --output t64.md
"""

import argparse
import json
import os
import sys
from typing import Optional

from embodied_schemas import load_kpus
from embodied_schemas.kpu import KPUEntry


def _render_text(e: KPUEntry) -> str:
    out = []
    out.append(f"=== KPU SKU: {e.id} ===")
    out.append(f"  Name:            {e.name}")
    out.append(f"  Vendor:          {e.vendor}")
    out.append(f"  Process node:    {e.process_node_id}")
    out.append(f"  Last updated:    {e.last_updated}")
    out.append("")

    # Die
    out.append("--- Die (roll-up) ---")
    out.append(f"  Architecture:    {e.die.architecture}")
    out.append(f"  Foundry / node:  {e.die.foundry.value} {e.die.process_name} ({e.die.process_nm} nm)")
    out.append(f"  Transistors:     {e.die.transistors_billion:.2f} B")
    out.append(f"  Die size:        {e.die.die_size_mm2:.1f} mm^2")
    if e.die.is_chiplet:
        out.append(f"  Chiplet:         yes ({e.die.num_dies} dies)")
    out.append("")

    # Architecture
    arch = e.kpu_architecture
    out.append("--- Architecture ---")
    out.append(f"  Total tiles:     {arch.total_tiles}")
    out.append(f"  Multi-precision: {', '.join(arch.multi_precision_alu)}")
    out.append("")
    out.append("  Tile classes:")
    out.append(
        f"    {'tile_type':18s} {'num':>5s} {'PE array':>10s} {'PEs/tile':>9s} "
        f"{'lib':>16s}  ops/tile/clock"
    )
    total_pes = 0
    for t in arch.tiles:
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
        f"  NoC: {arch.noc.topology} {arch.noc.mesh_rows}x{arch.noc.mesh_cols}, "
        f"{arch.noc.flit_bytes}-byte flits, "
        f"router_lib={arch.noc.router_circuit_class.value}"
    )
    out.append(
        f"  Memory: {arch.memory.memory_type.value} "
        f"{arch.memory.memory_size_gb:.0f} GB, "
        f"{arch.memory.memory_bandwidth_gbps:.0f} GB/s, "
        f"{arch.memory.memory_bus_bits}-bit, "
        f"{arch.memory.memory_controllers} controllers"
    )
    out.append(
        f"  L1: {arch.memory.l1_kib_per_pe} KiB/PE  "
        f"L2: {arch.memory.l2_kib_per_tile} KiB/tile  "
        f"L3: {arch.memory.l3_kib_per_tile} KiB/tile  "
        f"(L3 total: {arch.memory.l3_kib_per_tile * arch.total_tiles / 1024:.1f} MiB)"
    )
    out.append("")

    # Silicon bin
    out.append("--- Silicon bin (per-block transistor decomposition) ---")
    out.append(f"  {'block':20s} {'circuit_class':18s} {'kind':>16s}  source")
    for b in e.silicon_bin.blocks:
        ts = b.transistor_source
        if ts.kind.value == "fixed":
            src = f"{ts.mtx} Mtx fixed"
        else:
            src = f"{ts.per_unit_mtx} Mtx/unit, ref={ts.count_ref}"
        out.append(
            f"  {b.name:20s} {b.circuit_class.value:18s} {ts.kind.value:>16s}  {src}"
        )
    out.append("")

    # Performance
    out.append("--- Performance (roll-up) ---")
    out.append(f"  INT8:  {e.performance.int8_tops:>8.1f} TOPS")
    out.append(f"  BF16:  {e.performance.bf16_tflops:>8.1f} TFLOPS")
    out.append(f"  FP32:  {e.performance.fp32_tflops:>8.1f} TFLOPS")
    if e.performance.int4_tops is not None:
        out.append(f"  INT4:  {e.performance.int4_tops:>8.1f} TOPS")
    out.append("")

    # Clocks
    out.append("--- Clocks ---")
    out.append(f"  Base:  {e.clocks.base_clock_mhz} MHz")
    out.append(f"  Boost: {e.clocks.boost_clock_mhz} MHz")
    out.append("")

    # Power + thermal profiles
    out.append("--- Power ---")
    out.append(f"  Default profile: {e.power.default_thermal_profile}")
    out.append(f"  TDP (default):   {e.power.tdp_watts} W")
    out.append(f"  Max:             {e.power.max_power_watts} W")
    out.append(f"  Min:             {e.power.min_power_watts} W")
    if e.power.idle_power_watts is not None:
        out.append(f"  Idle:            {e.power.idle_power_watts} W")
    out.append("")
    out.append("  Thermal profiles:")
    out.append(f"    {'profile':10s} {'TDP':>5s} {'clock':>9s}  cooling solution")
    for tp in e.power.thermal_profiles:
        out.append(
            f"    {tp.name:10s} {tp.tdp_watts:>5.0f} {tp.clock_mhz:>7.0f} MHz  "
            f"-> {tp.cooling_solution_id}"
        )
    out.append("")

    # Market
    out.append("--- Market ---")
    out.append(f"  Target:          {e.market.target_market}")
    out.append(f"  Tier:            {e.market.model_tier}")
    out.append(f"  Family:          {e.market.product_family}")
    if e.market.launch_date:
        out.append(f"  Launch:          {e.market.launch_date}")
    if e.market.launch_msrp_usd is not None:
        out.append(f"  MSRP:            ${e.market.launch_msrp_usd:,.0f}")
    out.append(f"  Available:       {'yes' if e.market.is_available else 'no'}")
    out.append("")

    if e.notes:
        out.append("--- Notes ---")
        out.append(e.notes.rstrip())
        out.append("")
    return "\n".join(out)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "md": "md", "markdown": "md", "txt": "text"}.get(ext, "text")


def main() -> int:
    parser = argparse.ArgumentParser(description="Show full spec of one KPUEntry.")
    parser.add_argument("kpu_id", help="KPU SKU id, e.g., stillwater_kpu_t256")
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        kpus = load_kpus()
    except Exception as exc:
        print(f"error: failed to load KPUs: {exc}", file=sys.stderr)
        return 1

    e = kpus.get(args.kpu_id)
    if e is None:
        print(
            f"error: no KPU SKU with id={args.kpu_id!r}. "
            f"Available: {', '.join(sorted(kpus))}",
            file=sys.stderr,
        )
        return 1

    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = json.dumps(e.model_dump(mode="json"), indent=2) + "\n"
    elif fmt == "md":
        # Reuse text but wrap in code block; keeps the inspector simple
        rendered = "```\n" + _render_text(e) + "```\n"
    else:
        rendered = _render_text(e) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
