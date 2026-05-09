#!/usr/bin/env python
"""
SKU Validator CLI

Runs every registered validator against a SKU and prints findings, sorted
by severity. Returns non-zero exit code if any ERROR finding is produced
-- suitable as a CI gate.

Phase 2a: framework only -- no validators registered yet. The CLI runs
cleanly and reports 0 findings until Phase 2b populates
``graphs.hardware.sku_validators.validators``.

Usage:
    python cli/validate_sku.py stillwater_kpu_t256
    python cli/validate_sku.py stillwater_kpu_t768 --category thermal
    python cli/validate_sku.py stillwater_kpu_t64 --severity warning
    python cli/validate_sku.py stillwater_kpu_t256 --output findings.json
    python cli/validate_sku.py stillwater_kpu_t256 --output findings.md

Exit codes:
    0 = no ERROR findings (or filtered out by --severity)
    1 = at least one ERROR finding
    2 = framework / context error (sku not found, broken cross-ref, etc.)
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
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
        out.append("  No validators are registered. Phase 2a ships the")
        out.append("  validator framework only; Phase 2b adds the validators.")
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
            "_No validators are registered. Phase 2a ships the framework only._"
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


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "md": "md", "markdown": "md", "txt": "text"}.get(
        ext, "text"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SKU validators against a KPU SKU and report findings."
    )
    parser.add_argument("sku_id", help="SKU id, e.g., stillwater_kpu_t256")
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
        "Default: stdout text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARNING findings as ERROR for exit-code purposes",
    )
    args = parser.parse_args()

    # Trigger validator self-registration.
    validator_count = load_validators()

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
