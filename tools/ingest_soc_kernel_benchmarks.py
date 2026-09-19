#!/usr/bin/env python
"""Turn stage-kernel benchmark runs into an SoC efficiency table
(graphs#269 PR 5.2).

Reads ``soc_kernel_bench/1`` documents written by
``cli/benchmark_soc_kernels.py`` and writes a per-engine efficiency table
that layers over ``default_v1`` (``base: default_v1``): measured pairs
replace unknowns, everything unmeasured stays as ``default_v1`` has it.

**Efficiency is attained over the composed peak at the measured clock**::

    eff = attained_ops_per_s / (per_server_ops_per_clock(fmt) x measured_clock_hz)

where the per-server peak comes from the design that describes the measured
hardware -- a GPU is one server (all SMs), a CPU core one server -- so the
efficiency means what the analyzer means by it. Using the measured clock,
not the design's, is the point: the 2026-02 calibrations divided by a clock
the GPU was not running at.

Rules, each strict:

* **Confidence** is CALIBRATED when the run's clock was verified (enough
  samples, within 5%), INTERPOLATED when sampled but not verified. A run with
  no clock samples gives no entry.
* **A format the IP template states no peak for** gives no entry (Orin's SM
  template has no FP16 rate): the measurement is reported, and the pair
  stays a gap until the template is sourced.
* **An efficiency above 1** means the template's peak or the clock is wrong;
  it is refused, not clipped.
* Several kernels for one (kernel class, engine, precision) give the median
  as the value and the spread as ``eff_range``, at the weakest confidence.
* Quick runs (``--quick`` shapes) are refused: they measure launch overhead,
  not the stage.
* A CPU result counts only when the run verified it single-threaded (the
  analyzer's CPU server is one core); a GPU FP32 result only when the run
  locked TF32 off. Runs from before those checks are refused on both.

Usage:
    python tools/ingest_soc_kernel_benchmarks.py \\
        --design orin_class_reference --node samsung_8lpp --count gpu_sm=8 \\
        --table orin_nano_measured_v1 soc_designs/efficiency/measurements/jetson_orin_nano_*.json
    python tools/ingest_soc_kernel_benchmarks.py ... --check   # fail if the table differs
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from embodied_schemas import load_process_nodes  # noqa: E402

from graphs.estimation.soc.mapping import engines_of  # noqa: E402
from graphs.estimation.soc.study import Override  # noqa: E402
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402

SCHEMA = "soc_kernel_bench/1"
EFFICIENCY_DIR = REPO / "soc_designs" / "efficiency"


def _per_server_ops_per_clock(engine, fmt: str) -> Optional[float]:
    """Ops per clock of one server of ``engine`` in ``fmt``, or None."""
    compute = engine.block.template.compute
    per_instance = compute.ops_per_clock.get(fmt)
    if not per_instance:
        return None
    return per_instance * engine.block.count / engine.servers


def build_table(runs: List[Tuple[str, dict]], design_id: str, node: Optional[str],
                counts: Dict[str, int], table_id: str) -> Tuple[dict, List[str]]:
    """The table document, and a report line per result not ingested."""
    design = load_designs()[design_id]
    for instance, count in counts.items():
        design = Override(target=f"block:{instance}.count", values=[count]).apply(design, count)
    soc = compose_soc(design, load_ip_library(), load_process_nodes(), node)
    engines = engines_of(soc)
    by_kind = {}
    for name, engine in engines.items():
        by_kind.setdefault(engine.kind.value, []).append(engine)

    skipped: List[str] = []
    measured: Dict[Tuple[str, str, str], List[dict]] = {}
    for label, doc in runs:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"{label}: schema {doc.get('schema')!r}, expected {SCHEMA}")
        if doc.get("quick"):
            raise ValueError(f"{label}: a --quick run measures launch overhead, not the stage; re-run without --quick")
        for r in doc["results"]:
            tag = f"{label}: {r['kernel_class']}/{r['engine_kind']}/{r['precision']} ({r['name']})"
            if r["status"] != "ok":
                skipped.append(f"{tag}: {r['status']} -- {r['message']}")
                continue
            if r["engine_kind"] == "cpu" and r.get("single_thread") is not True:
                why = ("the run predates the single-thread check" if r.get("single_thread") is None
                       else f"it ran on {r.get('cpu_parallelism', 0):.1f} cores' worth of CPU time")
                skipped.append(f"{tag}: not verified single-threaded ({why}); a CPU server is one core")
                continue
            if r["engine_kind"] == "gpu" and r["precision"] == "fp32" and doc.get("tf32_disabled") is not True:
                skipped.append(f"{tag}: FP32 may have run as TF32 (the run did not lock TF32 off)")
                continue
            candidates = by_kind.get(r["engine_kind"], [])
            if len(candidates) != 1:
                skipped.append(f"{tag}: design {design_id} has {len(candidates)} {r['engine_kind']} engines")
                continue
            engine = candidates[0]
            per_clock = _per_server_ops_per_clock(engine, r["precision"])
            if per_clock is None:
                skipped.append(f"{tag}: {engine.block.template.id} states no {r['precision']} peak; "
                               f"measured {r['attained_ops_per_s'] / 1e9:.1f} GOP/s kept here only")
                continue
            clock = r.get("clock") or {}
            if not clock.get("median_hz"):
                skipped.append(f"{tag}: no clock samples, so no peak to divide by")
                continue
            eff = r["attained_ops_per_s"] / (per_clock * clock["median_hz"])
            if eff > 1.0:
                skipped.append(f"{tag}: efficiency {eff:.3f} > 1 -- the template peak or the clock "
                               "is wrong; refused")
                continue
            measured.setdefault((r["kernel_class"], r["engine_kind"], r["precision"]), []).append(
                {"eff": eff, "verified": bool(clock.get("verified")), "run": label, "kernel": r["name"],
                 "shape": r["shape"], "clock_ghz": clock["median_hz"] / 1e9,
                 "hardware": doc["hardware"], "power_mode": doc.get("power_mode", "")})

    entries = []
    for (kc, kind, prec), items in sorted(measured.items()):
        effs = [i["eff"] for i in items]
        value = statistics.median(effs)
        verified = all(i["verified"] for i in items)
        entry = {
            "kernel_class": kc, "engine_kind": kind, "precision": prec,
            "compute_eff": round(value, 4),
            "confidence": "calibrated" if verified else "interpolated",
            "source": "; ".join(
                f"{i['hardware']} {i['power_mode'] or ''} {i['kernel']} {i['shape']} at "
                f"{i['clock_ghz']:.3f} GHz measured{'' if i['verified'] else ' (clock not verified)'}"
                f" = {i['eff']:.4f} ({i['run']})" for i in items).replace("  ", " ")
            + f"; over the {design_id} per-server peak at the measured clock",
        }
        if len(items) > 1:
            entry["eff_range"] = [round(min(effs), 4), round(max(effs), 4)]
        entries.append(entry)
    if not entries:
        raise ValueError("no result could be ingested:\n  " + "\n  ".join(skipped))
    table = {
        "id": table_id,
        "name": f"Measured stage-kernel efficiencies ({', '.join(sorted({l for l, _ in runs}))})",
        "kind": "per_engine",
        "base": "default_v1",
        "notes": ("Generated by tools/ingest_soc_kernel_benchmarks.py from stage-kernel benchmark "
                  f"runs, against design {design_id}"
                  + (f" with counts {counts}" if counts else "") + "; do not edit by hand. "
                  "Unmeasured pairs inherit default_v1. Efficiency assumes clock-invariance, "
                  "which holds for compute-bound kernels; each source states the clock."),
        "entries": entries,
    }
    return table, skipped


def _parse_counts(items: List[str]) -> Dict[str, int]:
    out = {}
    for item in items:
        instance, _, value = item.partition("=")
        if not value.isdigit():
            raise ValueError(f"--count expects INSTANCE=N, got {item!r}")
        out[instance] = int(value)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="soc_kernel_bench/1 JSON files")
    parser.add_argument("--design", required=True, help="Design that describes the measured hardware")
    parser.add_argument("--node", help="Node to compose it at (default: its own)")
    parser.add_argument("--count", action="append", default=[],
                        help="Override a block count to match the hardware, e.g. gpu_sm=8 for an Orin Nano")
    parser.add_argument("--table", required=True, help="Table id; written to soc_designs/efficiency/<id>.yaml")
    parser.add_argument("--check", action="store_true", help="Compare with the file on disk instead of writing")
    args = parser.parse_args(argv)

    try:
        runs = [(p.name, json.loads(p.read_text())) for p in args.runs]
        table, skipped = build_table(runs, args.design, args.node, _parse_counts(args.count), args.table)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for line in skipped:
        print(f"not ingested: {line}", file=sys.stderr)
    out = EFFICIENCY_DIR / f"{args.table}.yaml"
    if args.check:
        if not out.exists() or yaml.safe_load(out.read_text()) != table:
            print(f"FAIL: {out} differs from its runs; regenerate it")
            return 1
        print(f"OK: {out}")
        return 0
    out.write_text(yaml.safe_dump(table, sort_keys=False, width=100))
    print(f"wrote {out} ({len(table['entries'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
