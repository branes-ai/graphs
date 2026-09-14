#!/usr/bin/env python
"""
KPU Golden Snapshot CLI

Checks (default) or regenerates (``--update``) the per-SKU golden snapshots
that pin every KPU catalog SKU's modeled outputs: generator round-trip,
5-term TDP breakdown, silicon areas / leakage / dynamic power, both
floorplans, PhysicalSpec, resource model, mapper results on synthetic
subgraphs, and validator findings. See ``graphs.hardware.kpu_golden``.

This is the zero-diff gate for the KPU heterogeneous-tile refactor
(docs/plans/kpu-heterogeneous-tile-refactor-plan.md, Phase A0). The same
contract is enforced in CI by ``tests/hardware/test_kpu_golden.py``.

When a diff is EXPECTED (a declared model change, or an embodied-schemas
catalog data change), review the diff, then regenerate with ``--update``
and commit the updated JSON alongside the change that caused it.

Not a UnifiedAnalyzer tool: it snapshots the SKU-modeling layers directly,
which is the point of a regression gate below the analyzer.

Usage:
    python cli/kpu_golden_snapshot.py                      # check all catalog SKUs
    python cli/kpu_golden_snapshot.py --sku kpu_t64_32x32_lp5x4_16nm_tsmc_ffp
    python cli/kpu_golden_snapshot.py --max-diffs 50       # show more diff lines per SKU
    python cli/kpu_golden_snapshot.py --output diff.json   # machine-readable report
    python cli/kpu_golden_snapshot.py --update             # regenerate every golden
    python cli/kpu_golden_snapshot.py --update --sku kpu_t256_32x32_lp5x16_16nm_tsmc_ffp

Exit codes:
    0 = every checked SKU matches its golden (or --update succeeded)
    1 = at least one SKU differs, lacks a golden, or a stale golden exists
    2 = argument / catalog error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from graphs.hardware import kpu_golden as kg


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "md": "md", "markdown": "md", "txt": "text"}.get(ext, "text")


def _render_text(results: dict[str, list[str]], stale: list[str], max_diffs: int) -> str:
    lines = [f"=== KPU golden snapshot check ({len(results)} SKUs) ==="]
    for sku_id, diffs in results.items():
        if not diffs:
            lines.append(f"  OK    {sku_id}")
            continue
        lines.append(f"  DIFF  {sku_id}  ({len(diffs)} differences)")
        for d in diffs[:max_diffs]:
            lines.append(f"          {d}")
        if len(diffs) > max_diffs:
            lines.append(f"          ... {len(diffs) - max_diffs} more (use --max-diffs)")
    for sku_id in stale:
        lines.append(f"  STALE {sku_id}  (golden exists but SKU is not in the catalog)")
    failed = sum(1 for d in results.values() if d) + len(stale)
    lines.append("")
    lines.append("PASS" if not failed else f"FAIL: {failed} SKU(s) differ or are stale")
    return "\n".join(lines) + "\n"


def _render_md(results: dict[str, list[str]], stale: list[str], max_diffs: int) -> str:
    lines = ["# KPU golden snapshot check", "", "| SKU | Status | Differences |", "|---|---|---|"]
    for sku_id, diffs in results.items():
        lines.append(f"| `{sku_id}` | {'OK' if not diffs else 'DIFF'} | {len(diffs)} |")
    for sku_id in stale:
        lines.append(f"| `{sku_id}` | STALE | - |")
    for sku_id, diffs in results.items():
        if diffs:
            lines += ["", f"## {sku_id}", "", "```"]
            lines += diffs[:max_diffs]
            if len(diffs) > max_diffs:
                lines.append(f"... {len(diffs) - max_diffs} more")
            lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check or regenerate the KPU golden snapshots (zero-diff "
        "regression gate for the KPU modeling layers)."
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Regenerate golden files instead of checking against them.",
    )
    parser.add_argument(
        "--sku", action="append", default=None, metavar="SKU_ID",
        help="Restrict to this SKU id (repeatable). Default: every KPU SKU "
        "in the embodied-schemas catalog.",
    )
    parser.add_argument(
        "--golden-dir", type=Path, default=kg.DEFAULT_GOLDEN_DIR,
        help=f"Golden snapshot directory (default: {kg.DEFAULT_GOLDEN_DIR}).",
    )
    parser.add_argument(
        "--max-diffs", type=int, default=20,
        help="Maximum diff lines printed per SKU (default: 20).",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write the check report to a file; format from extension "
        "(.json, .md, .txt).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress.")
    args = parser.parse_args(argv)

    try:
        catalogs = kg.load_catalogs()
        catalog_ids = kg.catalog_kpu_sku_ids()
    except Exception as exc:  # catalog load failure
        print(f"error: failed to load catalogs: {exc}", file=sys.stderr)
        return 2

    sku_ids = args.sku or catalog_ids
    unknown = [s for s in sku_ids if s not in catalogs["kpus"]]
    if unknown:
        print(f"error: unknown SKU id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.update:
        for sku_id in sku_ids:
            if args.verbose:
                print(f"building {sku_id} ...", file=sys.stderr)
            path = kg.write_snapshot(kg.build_snapshot(sku_id, catalogs), args.golden_dir)
            print(f"wrote {path}")
        stale = sorted(set(kg.golden_sku_ids(args.golden_dir)) - set(catalog_ids))
        for sku_id in stale:
            print(
                f"warning: stale golden {kg.golden_path(sku_id, args.golden_dir)} "
                f"(SKU not in catalog); delete it if the SKU was retired",
                file=sys.stderr,
            )
        return 0

    results = {}
    for sku_id in sku_ids:
        if args.verbose:
            print(f"checking {sku_id} ...", file=sys.stderr)
        results.update(kg.check_skus([sku_id], golden_dir=args.golden_dir, catalogs=catalogs))
    stale = (
        sorted(set(kg.golden_sku_ids(args.golden_dir)) - set(catalog_ids))
        if not args.sku else []
    )

    fmt = _detect_format(args.output)
    if fmt == "json":
        payload = json.dumps({"results": results, "stale": stale}, indent=2) + "\n"
    elif fmt == "md":
        payload = _render_md(results, stale, args.max_diffs)
    else:
        payload = _render_text(results, stale, args.max_diffs)

    if args.output:
        Path(args.output).write_text(payload)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload)

    failed = any(results.values()) or bool(stale)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
