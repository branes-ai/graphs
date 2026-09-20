#!/usr/bin/env python
"""Write one SoC design per generated KPU core (graphs#269 Phase 7).

The design state space a silicon partner asks about is *which KPU, with how
much CPU, in which process*. The KPU axis and the process axis are not
independent: a core's clock is stated by its SKU at one node, so composing
it elsewhere makes the clock provisional and every service time with it.

This writes one design per core in ``KPU_CORES``, at the node that core's
SKU is stated at, so each point of the state space is sourced. Everything
but the accelerator is the Orin-class platform's IP, so differences across
the space are the accelerator's, the CPU complement's and the memory
system's.

The CPU complement and the memory system stay sweep axes rather than
designs: ``block:cpu.count`` and ``block:memory.ip``.

Usage:
    python tools/generate_kpu_soc_designs.py          # write
    python tools/generate_kpu_soc_designs.py --check  # fail if one differs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_schemas import load_compute_products  # noqa: E402

from graphs.hardware.soc.kpu_cores import CORE_NODES, KPU_CORES  # noqa: E402

DESIGN_DIR = REPO / "soc_designs" / "designs"

#: The Phase 6 ladder's designs, which name their rungs by hand and are not
#: regenerated here.
HAND_WRITTEN = {"kpu_heterogeneous_h64", "kpu_uniform_t64", "kpu_uniform_t128",
                "kpu_uniform_t256"}

#: Short names for the nodes, for design ids.
NODE_SUFFIX = {"tsmc_n7": "n7", "tsmc_n16": "n16", "gf_12fdx": "12fdx"}


def design_id(core: str) -> str:
    """``kpu_t128_core_n16`` -> ``kpu_t128_n16``."""
    fabric = core.replace("_core", "").replace("kpu_", "")
    fabric = fabric.split("_")[0]
    return f"kpu_{fabric}_{NODE_SUFFIX[CORE_NODES[core]]}"


def document(core: str) -> dict:
    sku = KPU_CORES[core][0]
    if sku not in load_compute_products():
        raise KeyError(f"{core}: ComputeProduct {sku!r} is not in the catalog")
    node = CORE_NODES[core]
    fabric = core.replace("kpu_", "").replace("_core", "").split("_")[0].upper()
    return {
        "id": design_id(core),
        "name": f"KPU-{fabric} at {node} (Orin-class platform)",
        "process_node": node,
        "blocks": [
            {"instance": "kpu", "ip": core, "count": 1,
             "notes": f"The {fabric} KPU core, generated from ComputeProduct {sku}, which states "
                      f"its clock at {node}. It replaces the GPU SMs, GPU L2, both DLAs and the "
                      f"PVA."},
            {"instance": "cpu", "ip": "arm_cortex_a78ae_x4", "count": 3,
             "notes": "The Orin-class CPU complement. A study sweeps block:cpu.count, so the "
                      "cluster count is an axis rather than a design."},
            {"instance": "isp", "ip": "nvidia_orin_hdr_isp"},
            {"instance": "codec", "ip": "nvidia_orin_video_codec"},
            {"instance": "system_cache", "ip": "orin_system_cache"},
            {"instance": "safety_island", "ip": "orin_safety_island"},
            {"instance": "memory", "ip": "lpddr5_phy_256b",
             "notes": "The reference platform's 256-bit LPDDR5. A study sweeps block:memory.ip "
                      "over the wider options, because far flight needs more bandwidth than this "
                      "delivers whatever the accelerator is."},
            {"instance": "io", "ip": "orin_io_complex"},
            {"instance": "fabric", "ip": "orin_fabric"},
        ],
        "layout": {"whitespace_fraction": 0.15, "io_ring_mm": 0.3,
                   "source": "The reference design's layout overheads, kept equal across the state "
                             "space so area differences come from the blocks."},
        "notes": f"One point of the CPU + KPU design state space (graphs#269 Phase 7), generated "
                 f"by tools/generate_kpu_soc_designs.py; do not edit by hand. The node is the one "
                 f"{sku} states its clock at, so nothing here is provisional.",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Compare with the files on disk instead of writing")
    args = parser.parse_args(argv)

    status = 0
    for core in sorted(KPU_CORES):
        payload = document(core)
        if payload["id"] in HAND_WRITTEN:
            continue
        out = DESIGN_DIR / f"{payload['id']}.yaml"
        if args.check:
            if not out.exists() or yaml.safe_load(out.read_text()) != payload:
                print(f"FAIL: {out} differs from its core; regenerate it")
                status = 1
            else:
                print(f"OK: {out}")
            continue
        out.write_text(yaml.safe_dump(payload, sort_keys=False, width=100))
        print(f"wrote {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
