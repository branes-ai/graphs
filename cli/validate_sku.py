#!/usr/bin/env python
"""
SKU Validator CLI

Runs every registered validator against a SKU and prints findings, sorted
by severity. Returns non-zero exit code if any ERROR finding is produced
-- suitable as a CI gate.

The validator package self-registers a category-by-category set of
checks at import time (consistency, electrical, area, energy, thermal,
reliability). Use ``--category`` to focus on one risk class.

Phase 6 catalog gate: pass ``--all`` (in place of an SKU id) to validate
every KPU SKU in the embodied-schemas catalog and exit non-zero on any
ERROR. Identical contract to ``tests/hardware/test_sku_catalog_validation.py``;
useful for human-driven sweeps.

Usage:
    python cli/validate_sku.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp
    python cli/validate_sku.py kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc --category thermal
    python cli/validate_sku.py kpu_t64_32x32_lp5x4_16nm_tsmc_ffp --severity warning
    python cli/validate_sku.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp --output findings.json
    python cli/validate_sku.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp --output findings.csv
    python cli/validate_sku.py kpu_t256_32x32_lp5x16_16nm_tsmc_ffp --output findings.md
    python cli/validate_sku.py --all                  # gate every catalog SKU
    python cli/validate_sku.py --all --strict         # also fail on warnings

Exit codes:
    0 = no ERROR findings (or filtered out by --severity)
    1 = at least one ERROR finding
    2 = framework / context error (sku not found, broken cross-ref, etc.)
"""

import argparse
import csv
import io
import json
import os
import sys
from typing import List, Optional

from graphs.hardware.sku_validators import (
    ContextError,
    Finding,
    Severity,
    ValidatorCategory,
    build_context_for_kpu,
    default_registry,
    filter_findings,
    has_errors,
    load_validators,
)


def _render_text(
    findings: List[Finding], sku_id: str, validator_count: int
) -> str:
    out = []
    out.append(f"=== SKU validation: {sku_id} ===")
    out.append(f"  validators registered: {validator_count}")
    out.append(f"  findings:              {len(findings)}")
    if validator_count == 0:
        out.append("")
        out.append("  No validators are registered. Ensure")
        out.append("  graphs.hardware.sku_validators.validators is importable.")
        out.append("")
        return "\n".join(out)
    if not findings:
        out.append("")
        out.append("  All validators passed.")
        out.append("")
        return "\n".join(out)
    # Tally by severity / category.
    sev_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    for f in findings:
        sev_counts[f.severity.value] = sev_counts.get(f.severity.value, 0) + 1
        cat_counts[f.category.value] = cat_counts.get(f.category.value, 0) + 1
    out.append("  by severity: " + ", ".join(
        f"{k}={v}" for k, v in sorted(sev_counts.items())
    ))
    out.append("  by category: " + ", ".join(
        f"{k}={v}" for k, v in sorted(cat_counts.items())
    ))
    out.append("")
    for f in findings:
        out.append("  " + f.render_one_line())
        if f.citation:
            out.append(f"      citation: {f.citation}")
    out.append("")
    return "\n".join(out)


def _render_json(findings: List[Finding], sku_id: str, validator_count: int) -> str:
    payload = {
        "sku_id": sku_id,
        "validator_count": validator_count,
        "finding_count": len(findings),
        "findings": [
            {
                "validator": f.validator,
                "category": f.category.value,
                "severity": f.severity.value,
                "message": f.message,
                "block": f.block,
                "profile": f.profile,
                "citation": f.citation,
            }
            for f in findings
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def _render_md(findings: List[Finding], sku_id: str, validator_count: int) -> str:
    lines = [f"# SKU validation: `{sku_id}`", ""]
    lines.append(f"- **validators registered**: {validator_count}")
    lines.append(f"- **findings**: {len(findings)}")
    lines.append("")
    if validator_count == 0:
        lines.append(
            "_No validators are registered. Ensure "
            "`graphs.hardware.sku_validators.validators` is importable._"
        )
        lines.append("")
        return "\n".join(lines)
    if not findings:
        lines.append("All validators passed.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| severity | category | validator | block | profile | message |")
    lines.append("|---|---|---|---|---|---|")
    for f in findings:
        lines.append(
            f"| {f.severity.value} | {f.category.value} | `{f.validator}` | "
            f"{f.block or '-'} | {f.profile or '-'} | {f.message} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_csv(findings: List[Finding], sku_id: str, validator_count: int) -> str:
    """One row per finding. Header columns: sku_id, severity, category,
    validator, block, profile, message, citation."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "sku_id", "severity", "category", "validator",
            "block", "profile", "message", "citation",
        ],
    )
    writer.writeheader()
    for f in findings:
        writer.writerow({
            "sku_id": sku_id,
            "severity": f.severity.value,
            "category": f.category.value,
            "validator": f.validator,
            "block": f.block or "",
            "profile": f.profile or "",
            "message": f.message,
            "citation": f.citation or "",
        })
    return buf.getvalue()


def _run_catalog_sweep(args: argparse.Namespace, validator_count: int) -> int:
    """Phase 6 catalog gate -- iterate over every KPU SKU and report
    a per-SKU table to stdout. Returns non-zero on any ERROR finding
    (or any WARNING if ``--strict``).

    Scoped to KPU SKUs because every registered validator today is
    KPU-shaped (assumes ``KPUBlock`` with ``total_tiles`` / ``tiles``).
    GPU SKUs joined the catalog in v2 (e.g.
    ``nvidia_jetson_agx_orin_64gb`` from embodied-schemas#20) and would
    crash these validators. GPU-shaped validators are a separate
    follow-up; until they exist this gate skips non-KPU products.
    """
    from graphs.hardware.compute_product_loader import load_compute_products_unified
    try:
        all_products = load_compute_products_unified()
    except Exception as exc:
        print(f"error: failed to load KPU catalog: {exc}", file=sys.stderr)
        return 2

    def _kind_str(block) -> str:
        kind = block.kind
        raw = kind.value if hasattr(kind, "value") else kind
        return str(raw).strip().lower()

    kpus = {
        sku: cp for sku, cp in all_products.items()
        if cp.dies and cp.dies[0].blocks
        and _kind_str(cp.dies[0].blocks[0]) == "kpu"
    }
    skipped_non_kpu = sorted(set(all_products) - set(kpus))

    if not kpus:
        print("error: KPU catalog is empty", file=sys.stderr)
        return 2

    if skipped_non_kpu:
        print(
            f"info: skipping {len(skipped_non_kpu)} non-KPU SKU(s) "
            f"(KPU-shaped validators only): {', '.join(skipped_non_kpu)}",
            file=sys.stderr,
        )

    print(f"=== SKU catalog gate ({len(kpus)} SKUs, "
          f"{validator_count} validators) ===")
    print(f"  {'sku_id':32s} {'errors':>6s} {'warnings':>9s} {'info':>5s}  status")

    overall_errors = 0
    overall_warnings = 0
    failed_skus: List[str] = []

    for sku_id in sorted(kpus):
        try:
            ctx = build_context_for_kpu(sku_id)
        except ContextError as exc:
            print(f"  {sku_id:32s} {'-':>6s} {'-':>9s} {'-':>5s}  context-error: {exc}")
            failed_skus.append(sku_id)
            continue

        if args.category:
            cat = ValidatorCategory(args.category)
            findings = default_registry.run_category(ctx, cat)
        else:
            findings = default_registry.run_all(ctx)
        if args.severity:
            findings = filter_findings(
                findings, min_severity=Severity(args.severity)
            )

        n_err = sum(1 for f in findings if f.severity == Severity.ERROR)
        n_warn = sum(1 for f in findings if f.severity == Severity.WARNING)
        n_info = sum(1 for f in findings if f.severity == Severity.INFO)
        overall_errors += n_err
        overall_warnings += n_warn

        status = "ok"
        if n_err > 0:
            status = "FAIL"
            failed_skus.append(sku_id)
        elif args.strict and n_warn > 0:
            status = "FAIL (--strict)"
            failed_skus.append(sku_id)

        print(f"  {sku_id:32s} {n_err:>6d} {n_warn:>9d} {n_info:>5d}  {status}")

    print()
    print(f"Catalog summary: {overall_errors} error(s), "
          f"{overall_warnings} warning(s) across {len(kpus)} SKU(s)")

    if failed_skus:
        print(f"\nFailing SKUs ({len(failed_skus)}):")
        for sku in failed_skus:
            print(f"  - {sku}")
        return 1
    return 0


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
        description="Run SKU validators against a KPU SKU and report findings."
    )
    sku_group = parser.add_mutually_exclusive_group(required=True)
    sku_group.add_argument(
        "sku_id", nargs="?", help="SKU id, e.g., kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"
    )
    sku_group.add_argument(
        "--all",
        action="store_true",
        help="Phase 6 catalog gate: validate every KPU SKU in the "
        "embodied-schemas catalog. Exit non-zero on any ERROR finding "
        "across the whole catalog.",
    )
    parser.add_argument(
        "--category",
        choices=sorted(c.value for c in ValidatorCategory),
        help="Run only validators in this category",
    )
    parser.add_argument(
        "--severity",
        choices=sorted(s.value for s in Severity),
        help="Filter findings to this severity floor (info/warning/error)",
    )
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt). "
        "Default: stdout text. Ignored in --all mode.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARNING findings as ERROR for exit-code purposes",
    )
    args = parser.parse_args()

    # Trigger validator self-registration.
    validator_count = load_validators()

    # ---- --all mode: catalog sweep ----
    if args.all:
        return _run_catalog_sweep(args, validator_count)

    # Build context.
    try:
        ctx = build_context_for_kpu(args.sku_id)
    except ContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to build context: {exc}", file=sys.stderr)
        return 2

    # Run validators.
    if args.category:
        cat = ValidatorCategory(args.category)
        findings = default_registry.run_category(ctx, cat)
    else:
        findings = default_registry.run_all(ctx)

    # Apply severity filter.
    if args.severity:
        findings = filter_findings(
            findings, min_severity=Severity(args.severity)
        )

    # Render.
    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = _render_json(findings, args.sku_id, validator_count)
    elif fmt == "csv":
        rendered = _render_csv(findings, args.sku_id, validator_count)
    elif fmt == "md":
        rendered = _render_md(findings, args.sku_id, validator_count) + "\n"
    else:
        rendered = _render_text(findings, args.sku_id, validator_count) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)

    # Exit code.
    if has_errors(findings):
        return 1
    if args.strict and any(f.severity == Severity.WARNING for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
