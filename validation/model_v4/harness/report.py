"""Aggregate ValidationRecords into the v4 drift-attribution report.

Implements Principle 5: drift attribution is the output, not just
pass/fail. The standard output is a (hardware, op, regime) heatmap
plus a per-failure detail list. A FAIL cell points at one
architectural-model assumption that drifted.

Three output formats:

- ``format_text(...)``       human-readable terminal output
- ``format_markdown(...)``   for pasting into PR descriptions / issues
- ``format_json(...)``       machine-readable, for trend tracking

The text/markdown reports also surface the v4-1 known-thin cells
(e.g. i7-12700K ALU_BOUND = 0 entries) as a header note so a reader
isn't surprised by missing rows.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

from validation.model_v4.harness.assertions import ValidationRecord
from validation.model_v4.harness.runner import RunnerResult


# Display order matches the regime severity ordering in the plan
# (firm-spec layers first).
_REGIME_DISPLAY_ORDER = ["alu_bound", "l2_bound", "dram_bound", "launch_bound"]


@dataclass
class CellSummary:
    """One (hardware, regime) cell of the heatmap."""
    hardware: str
    op: str
    regime: str
    n_total: int
    n_pass: int
    n_fail_regime: int
    n_fail_latency: int
    n_fail_energy: int

    @property
    def all_pass(self) -> bool:
        return self.n_total > 0 and self.n_pass == self.n_total


def _aggregate(records: List[ValidationRecord]) -> dict[Tuple[str, str, str], CellSummary]:
    """Group records by (hardware, op, regime_predicted) and tally
    pass/fail counts per assertion."""
    groups: dict[Tuple[str, str, str], List[ValidationRecord]] = defaultdict(list)
    for r in records:
        key = (r.hardware, r.op, r.regime_predicted)
        groups[key].append(r)

    out: dict[Tuple[str, str, str], CellSummary] = {}
    for (hw, op, regime), bucket in groups.items():
        n_total = len(bucket)
        n_pass = sum(1 for r in bucket if r.all_pass())
        n_fail_regime = sum(1 for r in bucket if not r.pass_regime)
        n_fail_lat = sum(1 for r in bucket if not r.pass_latency)
        n_fail_egy = sum(1 for r in bucket if r.pass_energy is False)
        out[(hw, op, regime)] = CellSummary(
            hardware=hw, op=op, regime=regime,
            n_total=n_total, n_pass=n_pass,
            n_fail_regime=n_fail_regime,
            n_fail_latency=n_fail_lat,
            n_fail_energy=n_fail_egy,
        )
    return out


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


def _cell_glyph(cell: CellSummary) -> str:
    """One short token per cell so the heatmap is scannable."""
    if cell.n_total == 0:
        return "  -"
    if cell.all_pass:
        return f" {cell.n_pass:>2d}OK"
    # Failed cell: report the dominant failure type
    leads = []
    if cell.n_fail_regime:
        leads.append(f"R{cell.n_fail_regime}")
    if cell.n_fail_latency:
        leads.append(f"L{cell.n_fail_latency}")
    if cell.n_fail_energy:
        leads.append(f"E{cell.n_fail_energy}")
    return "/".join(leads)[:5].rjust(5)


def format_text(result: RunnerResult, *, op: str = "(any)") -> str:
    """Render a heatmap + per-shape failure list as plain text.

    Glyph legend:
      ``NN OK``  -- all NN entries in the cell passed
      ``RN``     -- N regime-classification failures
      ``LN``     -- N latency-band failures
      ``EN``     -- N energy-band failures
      `` -``     -- no entries in the cell (often expected; see header)
    """
    lines: List[str] = []
    cells = _aggregate(result.records)

    # Header
    lines.append(f"V4 validation -- op={op}")
    lines.append(f"  records: {len(result.records)}  "
                 f"skipped(no baseline): {len(result.skipped_no_baseline)}  "
                 f"skipped(unsupported): {len(result.skipped_unsupported)}")
    lines.append("")

    # Heatmap: rows = (hw, op), cols = regime
    hardware_op_pairs = sorted({(c.hardware, c.op) for c in cells.values()})
    if not hardware_op_pairs:
        lines.append("  (no records produced; heatmap is empty)")
    else:
        col_w = max(len(r) for r in _REGIME_DISPLAY_ORDER)
        # Header row
        header = "  " + " " * 28 + "  " + "  ".join(r.rjust(col_w) for r in _REGIME_DISPLAY_ORDER)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for hw, op_name in hardware_op_pairs:
            row_label = f"{hw} {op_name}".ljust(28)
            tokens = []
            for regime in _REGIME_DISPLAY_ORDER:
                cell = cells.get((hw, op_name, regime))
                if cell is None:
                    tokens.append("  -".rjust(col_w))
                else:
                    tokens.append(_cell_glyph(cell).rjust(col_w))
            lines.append(f"  {row_label}  " + "  ".join(tokens))

    # Per-failure detail (top N)
    failures = [r for r in result.records if not r.all_pass()]
    if failures:
        lines.append("")
        lines.append(f"  Failures ({len(failures)} of {len(result.records)} records):")
        lines.append("    " + "-" * 100)
        for r in failures[:30]:  # cap so a flood doesn't drown the report
            tag = []
            if not r.pass_regime: tag.append("R")
            if not r.pass_latency: tag.append("L")
            if r.pass_energy is False: tag.append("E")
            lines.append(
                f"    [{''.join(tag):3s}] {r.op:7s} {str(r.shape):20s} {r.dtype:5s} "
                f"pred={r.regime_predicted:11s} meas={r.regime_measured:11s} "
                f"lat={r.latency_predicted_ms:>8.3f}ms vs {r.latency_measured_ms:>8.3f}ms "
                f"({_relative(r.latency_predicted_ms, r.latency_measured_ms):>+5.0%}) "
                f"-- {r.bottleneck_layer}"
            )
        if len(failures) > 30:
            lines.append(f"    ... and {len(failures) - 30} more failures")

    return "\n".join(lines)


def _relative(predicted_ms: float, measured_ms: float) -> float:
    if measured_ms <= 0:
        return float("nan")
    return (predicted_ms - measured_ms) / measured_ms


# ---------------------------------------------------------------------------
# format_markdown
# ---------------------------------------------------------------------------


def format_markdown(result: RunnerResult, *, op: str = "(any)") -> str:
    """A markdown variant of format_text -- for PR descriptions, issues."""
    lines: List[str] = []
    cells = _aggregate(result.records)

    lines.append(f"## V4 validation -- op `{op}`")
    lines.append("")
    lines.append(f"- records: **{len(result.records)}**")
    lines.append(f"- skipped (no baseline): {len(result.skipped_no_baseline)}")
    lines.append(f"- skipped (unsupported): {len(result.skipped_unsupported)}")
    lines.append("")

    hardware_op_pairs = sorted({(c.hardware, c.op) for c in cells.values()})
    if hardware_op_pairs:
        lines.append("### Heatmap")
        lines.append("")
        header_cols = " | ".join(["target"] + _REGIME_DISPLAY_ORDER)
        sep_cols = " | ".join(["---"] * (len(_REGIME_DISPLAY_ORDER) + 1))
        lines.append(f"| {header_cols} |")
        lines.append(f"| {sep_cols} |")
        for hw, op_name in hardware_op_pairs:
            row = [f"{hw} {op_name}"]
            for regime in _REGIME_DISPLAY_ORDER:
                cell = cells.get((hw, op_name, regime))
                row.append(_cell_glyph(cell).strip() if cell else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    failures = [r for r in result.records if not r.all_pass()]
    if failures:
        lines.append(f"### Failures ({len(failures)} of {len(result.records)})")
        lines.append("")
        lines.append("| op | shape | dtype | regime pred -> meas | latency pred vs meas | bottleneck |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in failures[:50]:
            lines.append(
                f"| {r.op} | {r.shape} | {r.dtype} | "
                f"{r.regime_predicted} -> {r.regime_measured} | "
                f"{r.latency_predicted_ms:.3f}ms vs {r.latency_measured_ms:.3f}ms "
                f"({_relative(r.latency_predicted_ms, r.latency_measured_ms):+.0%}) | "
                f"{r.bottleneck_layer} |"
            )
        if len(failures) > 50:
            lines.append(f"")
            lines.append(f"... and {len(failures) - 50} more failures")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


def format_json(result: RunnerResult) -> str:
    """Machine-readable dump for trend tracking / CI artifacts.

    Schema:
      {
        "summary": {"records": N, "passed": N, "failed": N, ...},
        "cells": [ {hardware, op, regime, n_total, n_pass, n_fail_*}, ...],
        "records": [ ValidationRecord.to_dict(), ...]
      }
    """
    cells = _aggregate(result.records)
    n_pass = sum(1 for r in result.records if r.all_pass())
    payload = {
        "summary": {
            "records": len(result.records),
            "passed": n_pass,
            "failed": len(result.records) - n_pass,
            "skipped_no_baseline": len(result.skipped_no_baseline),
            "skipped_unsupported": len(result.skipped_unsupported),
        },
        "cells": [
            {
                "hardware": c.hardware, "op": c.op, "regime": c.regime,
                "n_total": c.n_total, "n_pass": c.n_pass,
                "n_fail_regime": c.n_fail_regime,
                "n_fail_latency": c.n_fail_latency,
                "n_fail_energy": c.n_fail_energy,
            }
            for c in sorted(cells.values(),
                            key=lambda x: (x.hardware, x.op, x.regime))
        ],
        "records": [r.to_dict() for r in result.records],
    }
    return json.dumps(payload, indent=2, default=str)
