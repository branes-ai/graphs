#!/usr/bin/env python
"""Generate SoC IP templates from KPU ComputeProducts (graphs#269 PR 4.3).

A KPU SKU in embodied-schemas is a standalone chip: its ``silicon_bin`` lists
transistor budgets per PE, per KiB of SRAM, per NoC router, per memory
controller, plus fixed pads and control. To place a KPU on an SoC die, this
script turns one into an ``IPBlockTemplate`` in ``soc_designs/ip/``:

* **Silicon.** Each silicon_bin block becomes a line with its resolved
  transistor count (``silicon_math.resolve_block_transistors``), in its own
  library, THEORETICAL (the generator's budgets). A *core* template -- the
  KPU as an on-die block of an SoC -- drops the chip's own memory PHYs and
  IO pads, because the SoC's memory system and pad ring are the design's.
* **Tile classes nothing prices.** A tile class that no ``PER_PE`` line
  references has no silicon in the SKU; it becomes an **unanchored** line
  naming the class, so the composition reports it as a gap instead of
  presenting the KPU as complete (the strict rule, P2-D3).
* **Compute.** Peak ops per clock per format is exact from the SKU: the sum
  over tile classes of ``num_tiles x ops_per_tile_per_clock``, for the
  formats the analyzer models (INT8, FP16, FP32, FP64). BF16, INT4 and LNS
  are not FP16, INT8 or anything the precision classes floor at, so they are
  left out rather than counted as a neighbour. Fixed-function tiles state no
  ops per clock and add nothing.
* **Clock.** The SKU's boost clock at its own node.

Usage:
    python tools/generate_kpu_ip.py          # write every template in GENERATED
    python tools/generate_kpu_ip.py --check  # fail if any file on disk differs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_schemas import __version__ as SCHEMAS_VERSION  # noqa: E402
from embodied_schemas import load_compute_products  # noqa: E402

from graphs.hardware.sku_validators.silicon_math import (  # noqa: E402
    resolve_block_transistors,
    silicon_die,
)
from graphs.hardware.soc.kpu_cores import KPU_CORES  # noqa: E402

IP_DIR = REPO / "soc_designs" / "ip"

#: template id -> (ComputeProduct id, core_only), from the package so that
#: analyses can resolve the SKU behind a design's KPU block too.
GENERATED: Dict[str, Tuple[str, bool]] = dict(KPU_CORES)

#: silicon_bin blocks that belong to a standalone chip, not to an on-die core.
CHIP_ONLY = {"memory_phys", "io_pads"}

#: Formats the SoC analyzer's precision classes run in (efficiency.PRECISION_ORDER).
MODELED_FORMATS = ("int8", "fp16", "fp32", "fp64")


def _kpu_block(cp):
    return next(b for b in silicon_die(cp).blocks if b.kind.value == "kpu")


def build(template_id: str, product_id: str, core_only: bool) -> dict:
    cp = load_compute_products()[product_id]
    die = silicon_die(cp)
    kpu = _kpu_block(cp)
    origin = f"embodied-schemas {SCHEMAS_VERSION} ComputeProduct {product_id}"

    silicon: List[dict] = []
    priced_tiles = set()
    for block in die.silicon_bin.blocks:
        if core_only and block.name in CHIP_ONLY:
            continue
        ref = block.transistor_source.count_ref or ""
        if ref.startswith("tile."):
            priced_tiles.add(ref.split(".", 1)[1])
        silicon.append({
            "name": block.name,
            "circuit_class": block.circuit_class.value,
            "mtx": round(resolve_block_transistors(block, cp), 6),
            "source": f"{origin}, silicon_bin block {block.name!r} "
                      f"({block.transistor_source.kind.value}); the KPU generator's budget",
            "confidence": "theoretical",
        })
    for tile in kpu.tiles:
        if tile.tile_class_id in priced_tiles or tile.tile_type in priced_tiles:
            continue
        library = _library(tile)
        silicon.append({
            "name": f"tile_{tile.tile_class_id}",
            "circuit_class": (library or "balanced_logic"),
            "unanchored": True,
            "source": f"{origin}: {tile.num_tiles} {tile.tile_kind.value} tile(s) of class "
                      f"{tile.tile_class_id!r} have no silicon_bin line, so the SKU states no "
                      "transistors for them. Needs a per-PE or fixed budget in the SKU."
                      + ("" if library else " The SKU states no library for this tile either; "
                         "balanced_logic is recorded only because a line needs one, and no "
                         "figure depends on it."),
            "confidence": "unknown",
        })

    compute_libraries = {_library(t) for t in kpu.tiles
                         if t.ops_per_tile_per_clock and _library(t)}
    ops: Dict[str, float] = {}
    for tile in kpu.tiles:
        for fmt, per_tile in (tile.ops_per_tile_per_clock or {}).items():
            if fmt in MODELED_FORMATS:
                ops[fmt] = ops.get(fmt, 0.0) + tile.num_tiles * per_tile
    clock_ghz = die.clocks.boost_clock_mhz / 1e3
    template = {
        "id": template_id,
        "name": f"{cp.name} {'core' if core_only else 'chip'} (generated)",
        "vendor": cp.vendor,
        "engine_kind": "kpu",
        "silicon": silicon,
        "compute": {
            "engine_kind": "kpu",
            "units": kpu.total_tiles,
            "unit_name": "tile",
            "ops_per_clock": {f: ops[f] for f in MODELED_FORMATS if ops.get(f)},
            "architecture_class": "domain_flow",
            **({"datapath_class": next(iter(compute_libraries))}
               if len(compute_libraries) == 1 else {}),
            "source": f"{origin}: sum over tile classes of num_tiles x ops_per_tile_per_clock; "
                      "BF16, INT4 and LNS rates are not counted as any modeled format",
        },
        "clock": {
            "fmax_ghz_ref": clock_ghz,
            "reference_node": die.process_node_id,
            "source": f"{origin}: boost clock {die.clocks.boost_clock_mhz:g} MHz",
        },
        "notes": ("Generated by tools/generate_kpu_ip.py; do not edit by hand. "
                  + ("Core only: the SKU's memory PHYs and IO pads are the SoC's, not the KPU's."
                     if core_only else "")).strip(),
    }
    if not core_only and kpu.memory is not None:
        template["memory_interface"] = {
            "peak_gb_per_s": kpu.memory.memory_bandwidth_gbps,
            "source": f"{origin}: memory_bandwidth_gbps",
        }
    return template


def _library(tile):
    """The library a tile's datapath is built in, as the SKU states it."""
    cc = getattr(tile, "pe_circuit_class", None) or getattr(tile, "circuit_class", None)
    return cc.value if cc is not None else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Compare against the files on disk instead of writing them.")
    args = parser.parse_args(argv)
    failed = False
    for template_id, (product_id, core_only) in GENERATED.items():
        path = IP_DIR / f"{template_id}.yaml"
        data = build(template_id, product_id, core_only)
        if args.check:
            if not path.exists() or yaml.safe_load(path.read_text()) != data:
                print(f"FAIL: {path} differs from {product_id}; regenerate it")
                failed = True
            else:
                print(f"OK: {path}")
            continue
        path.write_text(yaml.safe_dump(data, sort_keys=False, width=100))
        print(f"wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
