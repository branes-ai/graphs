#!/usr/bin/env python
"""
Run the Path B cache-sweep microbenchmark and persist results.

Sweeps a streaming kernel over a logarithmic range of working-set
sizes and writes per-point bandwidth (and per-byte energy when RAPL
is available) to JSON. The result file feeds
``graphs.calibration.fitters.cache_sweep_fitter`` which writes
CALIBRATED provenance into a HardwareResourceModel.

Usage:

    python cli/run_cache_sweep.py --hardware intel_core_i7_12700k \\
        --output sweeps/i7_12700k_cache_sweep.json

The ``--hardware`` argument is a free-form SKU id used only to tag
the output file -- the sweep itself runs whatever CPU is hosting
the Python process.

Energy capture is best-effort:

  * On Linux x86 with intel-rapl, the RAPL package energy counter
    is read before / after each measurement.
  * On other platforms, or when RAPL access is denied, energy
    columns come back ``None`` and downstream analysis falls back
    to the analytical TechnologyProfile estimate. The CLI emits a
    clear warning so the operator knows the result is time-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root so `graphs.*` imports work when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from graphs.benchmarks.cache_sweep import (  # noqa: E402
    SweepConfig,
    run_sweep,
)
from graphs.benchmarks.cache_sweep.working_set_sweep import (  # noqa: E402
    _RAPLProbe,
)


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="run_cache_sweep",
        description=("Run the Path B cache-sweep microbenchmark on the "
                     "host CPU and persist per-point results to JSON."),
    )
    p.add_argument(
        "--hardware",
        required=True,
        help="SKU id used to tag the output (e.g., intel_core_i7_12700k).",
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON path.",
    )
    p.add_argument(
        "--min-bytes",
        type=int,
        default=8 * 1024,
        help="Smallest working-set size in bytes (default 8 KiB).",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Largest working-set size in bytes (default 64 MiB).",
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=24,
        help="Number of log-spaced size points (default 24).",
    )
    p.add_argument(
        "--seconds-per-point",
        type=float,
        default=0.5,
        help="Wallclock target per size point (default 0.5 s).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per size point; min wallclock kept "
             "(default 3).",
    )
    p.add_argument(
        "--no-energy",
        action="store_true",
        help="Skip RAPL energy capture even when available.",
    )
    p.add_argument(
        "--rapl-zone",
        default="intel-rapl:0",
        help="RAPL zone path under /sys/class/powercap/ (default intel-rapl:0).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-point progress output.",
    )
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = SweepConfig(
        min_bytes=args.min_bytes,
        max_bytes=args.max_bytes,
        num_points=args.num_points,
        target_seconds_per_point=args.seconds_per_point,
        repeats_per_point=args.repeats,
        capture_energy=not args.no_energy,
        rapl_zone=args.rapl_zone,
    )

    rapl = _RAPLProbe(zone=args.rapl_zone)
    rapl_ok = (not args.no_energy) and rapl.available
    if not rapl_ok:
        print(
            "warning: RAPL energy capture unavailable -- result will be "
            "time-only. Energy/byte columns will be null.",
            file=sys.stderr,
        )

    if not args.quiet:
        print(
            f"running cache_sweep on {args.hardware}: "
            f"{config.num_points} points "
            f"in [{config.min_bytes // 1024} KiB, "
            f"{config.max_bytes // (1024 * 1024)} MiB], "
            f"~{config.target_seconds_per_point:.1f}s/point, "
            f"{config.repeats_per_point} repeat(s)"
        )

    points = []
    for i, point in enumerate(run_sweep(config)):
        if not args.quiet:
            energy_str = (
                f"  energy={point.energy_per_byte_pj:.2f} pJ/B"
                if point.energy_per_byte_pj is not None
                else ""
            )
            print(
                f"  [{i+1:>2}/{config.num_points}] "
                f"{point.bytes_resident:>10} B  "
                f"{point.bandwidth_gbps:>6.1f} GB/s"
                f"{energy_str}"
            )
        points.append({
            "bytes_resident": point.bytes_resident,
            "iterations": point.iterations,
            "elapsed_seconds": point.elapsed_seconds,
            "bandwidth_gbps": point.bandwidth_gbps,
            "energy_joules": point.energy_joules,
            "energy_per_byte_pj": point.energy_per_byte_pj,
            "extra": point.extra,
        })

    payload = {
        "schema_version": "1.0",
        "tool": "cli/run_cache_sweep.py",
        "hardware_id": args.hardware,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_bytes": config.min_bytes,
            "max_bytes": config.max_bytes,
            "num_points": config.num_points,
            "target_seconds_per_point": config.target_seconds_per_point,
            "repeats_per_point": config.repeats_per_point,
            "capture_energy": config.capture_energy,
            "rapl_available": rapl_ok,
        },
        "points": points,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(points)} points to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
