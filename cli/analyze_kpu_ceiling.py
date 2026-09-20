#!/usr/bin/env python
"""
What a KPU fabric could do at best, against what the workload needs
(graphs#269 Phase 6.4).

`analyze_required_efficiency.py` says what each engine would have to sustain
for a profile to fit. This says what the fabric could sustain at best, from
its own geometry and bandwidth, and puts the two together:

    time at the ceiling = occupancy at dense peak / ceiling

Summed over an engine's stages, that is its utilization at the ceiling.
**Above 1 the configuration is decided: not even a flawless schedule on this
silicon carries the profile.** At or under 1 it stays open -- a ceiling is
what cannot be exceeded, never what will be achieved.

The ceilings are upper bounds, each sound on its own (see
`graphs.estimation.soc.domainflow`):

  wavefront          the output-stationary schedule's passes, fill and drain
  dram compulsory    every operand and result crossing DRAM once
  operand delivery   the tile interconnect's bits per clock, where stated

They are THEORETICAL and are not efficiencies: no analysis prices a stage
with them.

Usage:
    python cli/analyze_kpu_ceiling.py --design kpu_uniform_t128 --regime "air superiority"
    python cli/analyze_kpu_ceiling.py --design kpu_uniform_t64 kpu_uniform_t128 \\
        kpu_uniform_t256 --all --node tsmc_n7 --output ceilings.csv
    python cli/analyze_kpu_ceiling.py --sku kpu_t256_32x32_lp5x16_7nm_tsmc_hpc --ceilings-only

Exit codes:
    0 = report produced
    2 = unknown design, SKU, profile or node, or a catalog that fails to load
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from embodied_schemas import load_compute_products, load_process_nodes  # noqa: E402

from graphs.core.pipeline_workload import load_autonomy_workload  # noqa: E402
from graphs.estimation.soc import (  # noqa: E402
    find_mapping,
    load_kernel_classes,
    required_efficiency,
)
from graphs.estimation.soc.domainflow import FabricCeilings, fabric_ceilings  # noqa: E402
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product  # noqa: E402
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library  # noqa: E402
from graphs.hardware.soc.kpu_cores import KPU_CORES, sku_of  # noqa: E402
from graphs.reporting.output_format import csv_writer, detect_format, write_report  # noqa: E402


def _fmt(value, spec: str = ".3g", none: str = "n/a") -> str:
    return none if value is None else format(value, spec)


def _ceilings_for(sku: str) -> FabricCeilings:
    products = load_compute_products()
    if sku not in products:
        raise KeyError(f"unknown ComputeProduct {sku!r}")
    return fabric_ceilings(input_spec_from_compute_product(products[sku]))


def _kpu_sku(soc) -> tuple:
    """The (block name, SKU id) of the design's KPU core."""
    for block in soc.blocks:
        if block.template.id in KPU_CORES:
            return block.name, sku_of(block.template.id)
    raise KeyError(f"design {soc.design.id!r} has no generated KPU core block; "
                   f"have {', '.join(sorted(KPU_CORES))}")


def _ceiling_rows(fc: FabricCeilings) -> List[dict]:
    return [{
        "kernel_class": c.kernel_class.value,
        "kernel": c.kernel,
        "precision": c.precision,
        "ceiling": f"{c.value:.4f}",
        "binding": c.binding,
        "wavefront": _fmt(c.bounds["wavefront"], ".4f"),
        "dram": _fmt(c.bounds["dram_compulsory"], ".4f"),
        "operand": _fmt(c.bounds["operand_delivery"], ".4f"),
    } for c in fc.entries]


def _stage_rows(result, fc: FabricCeilings, kernels, engine: str) -> List[dict]:
    """Each KPU stage: what it needs, the ceiling, and the time at it."""
    rows = []
    for stage in result.stages:
        if stage.engine != engine or stage.occupancy_at_peak is None:
            continue
        kernel = kernels.of(stage.stage)
        worst = None
        for fmt in stage.formats.values():
            ceiling = fc.best(kernel, fmt)
            if ceiling is None or ceiling.value is None:
                worst = None
                break
            worst = ceiling if worst is None or ceiling.value < worst.value else worst
        at_ceiling = None if worst is None else stage.occupancy_at_peak / worst.value
        rows.append({
            "stage": stage.stage,
            "kernel_class": kernel.value,
            "formats": ",".join(sorted(set(stage.formats.values()))),
            "at_peak": f"{stage.occupancy_at_peak:.4g}",
            "ceiling": "gap" if worst is None else f"{worst.value:.4f}",
            "binding": "-" if worst is None else worst.binding,
            "at_ceiling": "gap" if at_ceiling is None else f"{at_ceiling:.3g}",
            "fits": "-" if at_ceiling is None else ("yes" if at_ceiling <= 1 else "NO"),
        })
    return rows


def _summary(design: str, node: str, sku: str, result, fc: FabricCeilings,
             engine: str, rows: List[dict]) -> List[str]:
    priced = [r for r in rows if r["at_ceiling"] != "gap"]
    total = sum(float(r["at_ceiling"]) for r in priced)
    gaps = [r["stage"] for r in rows if r["at_ceiling"] == "gap"]
    need = next((e.required_efficiency for e in result.engines if e.engine == engine), None)
    lines = [
        f"profile: {result.profile.id}  ({result.profile.regime or 'no regime'})",
        f"design:  {design} at {node};  KPU core from {sku}",
        f"confidence: {fc.estimation_confidence.level.value} -- {fc.estimation_confidence.source}",
        f"{engine}: {len(rows)} stage(s); needs {_fmt(need, '.1%')} of dense peak",
        f"utilization at the ceiling: {total:.3g}"
        + (f" (LOWER BOUND: {len(gaps)} stage(s) with no ceiling: {', '.join(gaps)})" if gaps else "")
        + f" -- {'DECIDED: over 1, no schedule on this silicon carries it' if total > 1 else 'open'}",
    ]
    over = [r["stage"] for r in rows if r["fits"] == "NO"]
    if over:
        lines.append(f"stages that alone exceed their period at the ceiling: {', '.join(over)}")
    return lines


def _table(rows: List[dict]) -> str:
    if not rows:
        return "(none)\n"
    widths = {k: max(len(k), *(len(str(row[k])) for row in rows)) for k in rows[0]}
    out = ["  ".join(k.ljust(widths[k]) for k in rows[0]),
           "  ".join("-" * widths[k] for k in rows[0])]
    out += ["  ".join(str(row[k]).ljust(widths[k]) for k in rows[0]) for row in rows]
    return "\n".join(out) + "\n"


def _profiles(workload, args) -> List:
    if args.all:
        return list(workload.profiles)
    wanted = args.profile or args.regime or []
    out = []
    for name in wanted:
        match = [p for p in workload.profiles
                 if name in (p.id, p.regime, p.name) or name.lower() == (p.regime or "").lower()]
        if not match:
            raise KeyError(f"no profile, regime or mission named {name!r}")
        out += [p for p in match if p not in out]
    return out or list(workload.regimes())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="KPU domain-flow ceilings against the workload's requirement "
                    "(graphs#269 Phase 6.4).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--design", nargs="+", help="Design id(s) with a generated KPU core")
    source.add_argument("--sku", help="A KPU ComputeProduct id, for --ceilings-only")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--profile", action="append", help="Profile id or mission name (repeatable)")
    which.add_argument("--regime", action="append", help="Named regime (repeatable)")
    which.add_argument("--all", action="store_true", help="Every profile in the workload")
    parser.add_argument("--node", help="Process node (default: the design's own)")
    parser.add_argument("--mapping", default="capability",
                        help="capability (default), auto (the shipped mapping) or a .yaml file")
    parser.add_argument("--ceilings-only", action="store_true",
                        help="Just the ceilings, with no workload")
    parser.add_argument("--verbose", "-v", action="store_true", help="Add the per-stage table")
    parser.add_argument("--output", "-o", help="Write to a file; format from extension.")
    args = parser.parse_args(argv)

    try:
        if args.sku or args.ceilings_only:
            skus = [args.sku] if args.sku else [
                sku_of(_kpu_sku_template(name)) for name in (args.design or [])]
            payloads = [_ceilings_for(sku) for sku in skus]
            fmt = detect_format(args.output)
            if fmt == "json":
                text = json.dumps([p.to_dict() for p in payloads], indent=2)
            elif fmt == "csv":
                rows = [{"sku": p.sku, **row} for p in payloads for row in _ceiling_rows(p)]
                buf = io.StringIO()
                writer = csv_writer(buf, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                text = buf.getvalue()
            else:
                text = "\n".join(
                    f"{p.sku}: {p.estimation_confidence.level.value}\n" + _table(_ceiling_rows(p))
                    for p in payloads)
            write_report(text, args.output)
            return 0

        workload = load_autonomy_workload()
        designs, library, nodes = load_designs(), load_ip_library(), load_process_nodes()
        kernels = load_kernel_classes()
        profiles = _profiles(workload, args)
        blocks, texts, csv_rows, payload = [], [], [], []
        for name in args.design:
            if name not in designs:
                raise KeyError(f"unknown design {name!r}; have {', '.join(sorted(designs))}")
            soc = compose_soc(designs[name], library, nodes, args.node)
            engine, sku = _kpu_sku(soc)
            fc = _ceilings_for(sku)
            mapping = None
            if args.mapping != "capability":
                shipped = (find_mapping(name, workload.version) if args.mapping == "auto"
                           else None)
                if shipped is not None:
                    mapping = {s: e for s, e in shipped.assignments().items()
                               if isinstance(e, str)}
            for profile in profiles:
                result = required_efficiency(workload, profile, soc, mapping=mapping,
                                             kernels=kernels)
                rows = _stage_rows(result, fc, kernels, engine)
                if not rows:
                    continue
                blocks.append((name, soc.node.id, sku, result, fc, engine, rows))
                texts.append(_summary(name, soc.node.id, sku, result, fc, engine, rows)
                             + ([""] + _table(rows).splitlines() if args.verbose else []))
                csv_rows += [{"design": name, "node": soc.node.id, "sku": sku,
                              "profile": profile.id, "regime": profile.regime or "",
                              "engine": engine, **row} for row in rows]
                payload.append({"design": name, "node": soc.node.id, "sku": sku,
                                "profile": profile.id, "regime": profile.regime or "",
                                "engine": engine, "stages": rows,
                                "ceilings": fc.to_dict()})
        if not blocks:
            raise KeyError("no design put a stage on its KPU; nothing to compare")
        fmt = detect_format(args.output)
        if fmt == "json":
            text = json.dumps(payload, indent=2)
        elif fmt == "csv":
            buf = io.StringIO()
            writer = csv_writer(buf, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
            text = buf.getvalue()
        else:
            text = ("\n" + "=" * 72 + "\n").join("\n".join(t) + "\n" for t in texts)
        write_report(text, args.output)
        return 0
    except (KeyError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _kpu_sku_template(design_id: str) -> str:
    """The KPU core template a design uses, for --ceilings-only."""
    designs = load_designs()
    if design_id not in designs:
        raise KeyError(f"unknown design {design_id!r}")
    for block in designs[design_id].blocks:
        if block.ip in KPU_CORES:
            return block.ip
    raise KeyError(f"design {design_id!r} has no generated KPU core block")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
