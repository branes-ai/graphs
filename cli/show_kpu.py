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
    python cli/show_kpu.py --from-file build/kpu_h64.yaml
"""

import argparse
import csv
import io
import json
import sys
from typing import Optional

from embodied_schemas import ComputeProduct, PackagingKind, load_process_nodes
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.hardware import kpu_tile_display as display
from graphs.hardware.compute_product_loader import (
    ComputeProductFileError,
    load_compute_product_file,
    load_compute_products_unified,
)
from graphs.hardware.kpu_access import (
    KPUBlockLookupError,
    has_kpu_block,
    kpu_block_of,
    kpu_die_of,
)
from graphs.reporting.output_format import detect_format  # noqa: E402


def _kpu_block(cp: ComputeProduct):
    """The product's single KPUBlock, located by kind (see
    ``graphs.hardware.kpu_access``)."""
    return kpu_block_of(cp)


def _render_csv(cp: ComputeProduct) -> str:
    """Single-row CSV with the headline-numbers a SKU author tracks.

    Detail-view CSV is awkward (ComputeProduct has nested structures), so
    this renders a flat row of the most-comparable scalars. JSON is the
    right format for the full nested view; CSV is here for spreadsheet
    interop on the headline metrics."""
    block = _kpu_block(cp)
    die = kpu_die_of(cp)
    total_pes = display.total_pe_count(block)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "name", "vendor", "process_node_id",
        "total_tiles", "tile_census", "total_pes",
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
        block.total_tiles, display.kind_summary(block), total_pes,
        die.die_size_mm2, die.transistors_billion,
        cp.power.tdp_watts, default_clock,
        cp.performance.int8_tops, cp.performance.bf16_tflops,
        cp.performance.fp32_tflops, cp.performance.int4_tops or "",
        block.memory.memory_type.value,
        block.memory.memory_size_gb,
        block.memory.memory_bandwidth_gbps,
    ])
    return buf.getvalue()


def _render_power_domains(block) -> list:
    """The power-domain lines of the architecture section.

    A regular DVFS partition -- every cluster the same shape, each on its own
    rail and PLL, the default since graphs#268 F2 -- collapses to one summary
    line: listing 48 identical T768 clusters would bury the uncore and any
    tile_class domain under noise, while saying nothing about the sites. An
    irregular partition is listed domain by domain with its site ranges,
    because then the individual detail is the point. ``show_floorplan
    --overlay power-domain`` draws the site map either way.
    """
    lines = []
    clusters = [d for d in block.power_domains if d.kind.value == "cluster"]
    others = [d for d in block.power_domains if d.kind.value != "cluster"]
    shapes = {
        tuple(sorted((r.rows, r.cols) for r in d.site_ranges)) for d in clusters
    }
    regular = (
        clusters
        and len(shapes) == 1
        and len(next(iter(shapes))) == 1
        and len({d.rail_id for d in clusters}) == len(clusters)
        and len({d.clock_domain_id for d in clusters}) == len(clusters)
        and all(d.rail_id and d.clock_domain_id for d in clusters)
        and len({d.gateable for d in clusters}) == 1
    )
    if regular:
        (rows, cols), = next(iter(shapes))
        gateable = "yes" if clusters[0].gateable else "no"
        lines.append(
            f"    {len(clusters)} clusters of {rows}x{cols} sites, each its own rail "
            f"and PLL, gateable={gateable}  "
            f"({clusters[0].domain_id} .. {clusters[-1].domain_id})"
        )
    else:
        for d in clusters:
            ranges = ", ".join(
                f"r{r.row_min}-{r.row_max} c{r.col_min}-{r.col_max}" for r in d.site_ranges
            )
            lines.append(
                f"    {d.domain_id:18s} cluster      "
                f"gateable={'yes' if d.gateable else 'no':3s}  sites: {ranges}  "
                f"rail={d.rail_id or '-'} clock={d.clock_domain_id or '-'}"
            )
    for d in others:
        members = ", ".join(d.members) if d.members else "-"
        lines.append(
            f"    {d.domain_id:18s} {d.kind.value:12s} "
            f"gateable={'yes' if d.gateable else 'no':3s}  members: {members}"
        )
    return lines


def _render_text(cp: ComputeProduct, node: Optional[ProcessNodeEntry]) -> str:
    out = []
    out.append(f"=== KPU SKU: {cp.id} ===")
    out.append(f"  Name:            {cp.name}")
    out.append(f"  Vendor:          {cp.vendor}")
    die = kpu_die_of(cp)
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
    # Tile classes, rendered per kind (graphs#268 C6). A systolic tile
    # names its array and library differently from a PE fabric, and a
    # fixed-function tile has neither an array nor PEs, so each row shows
    # what its kind actually has and a detail line says what it is.
    out.append(f"  Tile census:     {display.kind_summary(block)}")
    out.append("")
    out.append("  Tile classes:")
    out.append(
        f"    {'tile_type':18s} {'kind':>9s} {'num':>4s} {'array':>7s} "
        f"{'PEs/tile':>8s} {'footprint':>10s} {'lib':>15s}"
    )
    for t in block.tiles:
        pes = display.tile_pe_count(t)
        out.append(
            f"    {t.tile_type:18s} {display.KIND_LABELS[display.tile_kind(t)]:>9s} "
            f"{t.num_tiles:>4d} {display.tile_geometry_str(t):>7s} "
            f"{(str(pes) if pes else '-'):>8s} {display.tile_footprint_str(t):>10s} "
            f"{(display.tile_circuit_class(t) or '-'):>15s}"
        )
        out.append(f"      class={t.tile_class_id}  {display.tile_detail(t)}")
        out.append(f"      ops/tile/clock: {display.tile_ops_str(t)}")
    if any(display.absorbs_memory_cells(t) for t in block.tiles):
        out.append(
            f"    {display.ABSORBS_MARK} footprint absorbs the memory cells it covers"
        )
    pe_note = (
        "   (fixed-function tiles contribute none)"
        if any(not display.is_programmable(t) for t in block.tiles)
        else ""
    )
    out.append(f"  Total PEs:       {display.total_pe_count(block)}{pe_note}")
    out.append("")
    # Checkerboard and power domains exist only on heterogeneous SKUs
    # (graphs#268 B4); a uniform SKU leaves both unset and prints neither.
    cb = block.checkerboard
    if cb is not None:
        sites = cb.compute_sites.rows * cb.compute_sites.cols
        out.append(
            f"  Checkerboard: {cb.compute_sites.rows}x{cb.compute_sites.cols} "
            f"compute sites ({sites}), {display.occupied_sites(block)} occupied "
            f"by {block.total_tiles} tiles, {cb.spare_sites} spare, "
            f"placement={cb.placement.value}"
        )
        if cb.memory_cell is not None:
            out.append(f"    memory cell: {cb.memory_cell.kib_per_cell} KiB/cell")
    if block.power_domains:
        out.append("  Power domains:")
        out.extend(_render_power_domains(block))
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
        f"  L1: {block.memory.l1_kib_per_tile} KiB/tile  "
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
    # The by-kind split (graphs#268 B5) says how much of the roll-up each
    # kind contributes; a uniform SKU omits it, since the answer is "all".
    by_kind = getattr(cp.performance, "by_tile_kind", None)
    if by_kind:
        out.append("")
        out.append("  By tile kind (ops/sec):")
        for kind, ops in by_kind.items():
            label = display.KIND_LABELS.get(kind, kind)
            rates = ", ".join(f"{p}={v / 1e12:.1f}T" for p, v in ops.items())
            out.append(f"    {label:>9s}  {rates}")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show full spec of one KPU ComputeProduct."
    )
    parser.add_argument(
        "kpu_id",
        nargs="?",
        help="KPU SKU id, e.g., kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="Inspect a ComputeProduct YAML / JSON file instead of a catalog "
             "SKU, e.g. the output of cli/generate_kpu_sku.py.",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt).",
    )
    args = parser.parse_args()

    if not args.kpu_id and not args.from_file:
        parser.error("give a KPU SKU id, or --from-file PATH")
    if args.kpu_id and args.from_file:
        parser.error("give a KPU SKU id or --from-file, not both")

    if args.from_file:
        try:
            cp = load_compute_product_file(args.from_file)
        except ComputeProductFileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not has_kpu_block(cp):
            print(
                f"error: {args.from_file} holds compute product {cp.id!r}, "
                "which has no KPU block",
                file=sys.stderr,
            )
            return 1
        label = args.from_file
    else:
        try:
            cps = load_compute_products_unified()
        except Exception as exc:
            print(f"error: failed to load catalog: {exc}", file=sys.stderr)
            return 1
        cp = cps.get(args.kpu_id)
        if cp is None or not has_kpu_block(cp):
            kpu_ids = sorted(k for k, v in cps.items() if has_kpu_block(v))
            print(
                f"error: no KPU SKU with id={args.kpu_id!r}. "
                f"Available: {', '.join(kpu_ids)}",
                file=sys.stderr,
            )
            return 1
        label = args.kpu_id

    try:
        die = kpu_die_of(cp)
    except KPUBlockLookupError as exc:
        print(f"error: invalid KPU SKU {label!r}: {exc}", file=sys.stderr)
        return 1

    # Process nodes are looked up after the source is settled: a --from-file
    # product is readable on its own, and every renderer handles an
    # unresolved node, so a catalog problem must not block inspecting a
    # local file. A catalog SKU still needs the catalog.
    try:
        node = load_process_nodes().get(die.process_node_id)
    except Exception as exc:
        if not args.from_file:
            print(f"error: failed to load catalog: {exc}", file=sys.stderr)
            return 1
        print(
            f"warning: process-node catalog unavailable ({exc}); "
            f"showing {die.process_node_id} unresolved",
            file=sys.stderr,
        )
        node = None

    fmt = detect_format(args.output)
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
