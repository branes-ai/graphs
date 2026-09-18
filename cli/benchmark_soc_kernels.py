#!/usr/bin/env python
"""
SoC Stage-Kernel Benchmark

Measures the stage kernels the SoC analyzer's efficiency tables key on
(dense conv/GEMM, attention, GEMV decode, normalization, FFT, small dense
linear algebra, KKT solves, scatter) on this machine, sampling the device
clock throughout every timed loop (graphs#269 PR 5.1).

Run it on the target -- a Jetson Orin Nano or AGX -- and commit the JSON to
soc_designs/efficiency/measurements/. tools/ingest_soc_kernel_benchmarks.py
(PR 5.2) turns it into efficiency-table entries; an entry is CALIBRATED only
when its clock was verified (enough samples, within 5% of each other).

On a Jetson, lock the clocks first so the run measures one operating point:
    sudo nvpmodel -m 0 && sudo jetson_clocks

CPU kernels run pinned to one core, single-threaded: the analyzer treats a
CPU core as one server. GPU kernels use the whole GPU.

Usage:
    python cli/benchmark_soc_kernels.py --hardware jetson_orin_nano_8gb --power-mode MAXN
    python cli/benchmark_soc_kernels.py --hardware jetson_orin_agx_64gb --devices cuda
    python cli/benchmark_soc_kernels.py --hardware dev_box --quick --output /tmp/smoke.json

Exit codes:
    0 = measurements written (individual kernels may be unsupported or errors; see the table)
    2 = bad arguments
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from graphs.benchmarks.soc_kernels import run_suite  # noqa: E402

MEASUREMENT_DIR = REPO / "soc_designs" / "efficiency" / "measurements"


def _table(doc: dict) -> str:
    rows = []
    for r in doc["results"]:
        clock = r["clock"] or {}
        rows.append((
            r["engine_kind"], r["kernel_class"], r["name"], r["precision"], r["status"],
            f"{r['attained_ops_per_s'] / 1e9:.2f}" if r["attained_ops_per_s"] else "-",
            f"{clock['median_hz'] / 1e9:.3f}" if clock.get("median_hz") else "-",
            {True: "yes", False: "NO", None: "-"}[clock.get("verified")],
            r["message"][:48],
        ))
    head = ("engine", "kernel_class", "kernel", "prec", "status", "GOP/s", "clock_GHz", "clock_ok", "note")
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    return "\n".join([line(head), line(["-" * w for w in widths]), *map(line, rows)]) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure SoC stage-kernel throughput (graphs#269 Phase 5).")
    parser.add_argument("--hardware", required=True,
                        help="What this machine is, e.g. jetson_orin_nano_8gb (goes in the file name)")
    parser.add_argument("--power-mode", default="", help="e.g. MAXN, 15W (recorded, not set)")
    parser.add_argument("--devices", help="cpu, cuda or cpu,cuda (default: every available)")
    parser.add_argument("--kernels", help="Comma-separated kernel classes (default: all)")
    parser.add_argument("--quick", action="store_true", help="Small shapes, for a smoke test")
    parser.add_argument("--min-seconds", type=float, default=0.5, help="Timed loop length per kernel")
    parser.add_argument("--cpu-core", type=int, default=0, help="Core CPU kernels are pinned to")
    parser.add_argument("--verbose", "-v", action="store_true", help="Also print the JSON")
    parser.add_argument("--output", "-o",
                        help="JSON path (default: soc_designs/efficiency/measurements/<hardware>_<date>.json)")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[a-z0-9_]+", args.hardware):
        parser.error("--hardware must be lower-case letters, digits and underscores")
    import torch  # noqa: PLC0415

    devices = ([d.strip() for d in args.devices.split(",")] if args.devices
               else ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    if "cuda" in devices and not torch.cuda.is_available():
        print("error: --devices cuda but no CUDA device is available", file=sys.stderr)
        return 2
    if any(d not in ("cpu", "cuda") for d in devices):
        parser.error(f"unknown device in {devices}")

    doc = run_suite(devices, args.hardware, args.power_mode, quick=args.quick,
                    min_seconds=args.min_seconds, cpu_core=args.cpu_core,
                    kernels=[k.strip() for k in args.kernels.split(",")] if args.kernels else None)
    doc["quick"] = args.quick
    out = Path(args.output) if args.output else (
        MEASUREMENT_DIR / f"{args.hardware}_{args.power_mode.lower() or 'default'}_{date.today():%Y%m%d}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(_table(doc))
    if args.verbose:
        print(json.dumps(doc, indent=2))
    unverified = sum(1 for r in doc["results"] if r["status"] == "ok" and not (r["clock"] or {}).get("verified"))
    if unverified:
        print(f"note: {unverified} kernel(s) ran with an unverified clock; lock clocks and re-run "
              "for CALIBRATED entries", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
