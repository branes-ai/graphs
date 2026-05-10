#!/usr/bin/env python
"""
Cooling-Solution Detail Inspector

Prints the full spec of one CoolingSolutionEntry: thermal envelope,
form-factor constraints, mechanical / cost data, and provenance.

Usage:
    python cli/show_cooling_solution.py active_fan
    python cli/show_cooling_solution.py datacenter_dtc --output dtc.json
"""

import argparse
import json
import os
import sys
from typing import Optional

from embodied_schemas import load_cooling_solutions
from embodied_schemas.cooling_solution import CoolingSolutionEntry


def _na(v: Optional[float], unit: str = "") -> str:
    return "N/A" if v is None else f"{v:g}{unit}"


def _render_text(e: CoolingSolutionEntry) -> str:
    out = []
    out.append(f"=== CoolingSolution: {e.id} ===")
    out.append(f"  Name:           {e.name}")
    out.append(f"  Mechanism:      {e.cooling_mechanism.value}")
    out.append(f"  Max W/mm^2:     {e.max_power_density_w_per_mm2:.2f}")
    out.append(f"  Max total W:    {e.max_total_w:.0f}")
    out.append(f"  Ambient C max:  {e.ambient_c_max:.0f}")
    out.append(f"  Junction C max: {e.junction_c_max:.0f}")
    out.append(f"  Weight:         {_na(e.weight_g, ' g')}")
    out.append(f"  Cost:           {_na(e.cost_usd, ' USD')}")
    out.append(f"  Confidence:     {e.confidence.value}")
    out.append(f"  Source:         {e.source}")
    out.append(f"  Updated:        {e.last_updated}")
    out.append("")

    if e.form_factor_constraints:
        out.append("--- Form-factor constraints ---")
        for c in e.form_factor_constraints:
            out.append(f"  - {c}")
        out.append("")

    if e.notes:
        out.append("--- Notes ---")
        out.append(e.notes.rstrip())
        out.append("")

    return "\n".join(out)


def _render_md(e: CoolingSolutionEntry) -> str:
    lines = [f"# CoolingSolution: `{e.id}`", ""]
    lines.append(f"- **Name**: {e.name}")
    lines.append(f"- **Mechanism**: {e.cooling_mechanism.value}")
    lines.append(f"- **Max W/mm^2**: {e.max_power_density_w_per_mm2:.2f}")
    lines.append(f"- **Max total W**: {e.max_total_w:.0f}")
    lines.append(f"- **Ambient C max**: {e.ambient_c_max:.0f}")
    lines.append(f"- **Junction C max**: {e.junction_c_max:.0f}")
    lines.append(f"- **Weight**: {_na(e.weight_g, ' g')}")
    lines.append(f"- **Cost**: {_na(e.cost_usd, ' USD')}")
    lines.append(f"- **Confidence**: {e.confidence.value}")
    lines.append(f"- **Source**: {e.source}")
    lines.append("")
    if e.form_factor_constraints:
        lines.append("## Form-factor constraints")
        lines.append("")
        for c in e.form_factor_constraints:
            lines.append(f"- {c}")
        lines.append("")
    if e.notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(e.notes.rstrip())
        lines.append("")
    return "\n".join(lines)


def _detect_format(output: Optional[str]) -> str:
    if not output:
        return "text"
    ext = os.path.splitext(output)[1].lower().lstrip(".")
    return {"json": "json", "md": "md", "markdown": "md", "txt": "text"}.get(ext, "text")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show full spec of one CoolingSolutionEntry."
    )
    parser.add_argument("cooling_id", help="Cooling-solution id")
    parser.add_argument(
        "--output",
        help="Output file. Format auto-detected from extension (.json/.md/.txt).",
    )
    args = parser.parse_args()

    try:
        sols = load_cooling_solutions()
    except Exception as exc:
        print(f"error: failed to load cooling solutions: {exc}", file=sys.stderr)
        return 1

    e = sols.get(args.cooling_id)
    if e is None:
        print(
            f"error: no cooling solution with id={args.cooling_id!r}. "
            f"Available: {', '.join(sorted(sols))}",
            file=sys.stderr,
        )
        return 1

    fmt = _detect_format(args.output)
    if fmt == "json":
        rendered = json.dumps(e.model_dump(mode="json"), indent=2) + "\n"
    elif fmt == "md":
        rendered = _render_md(e) + "\n"
    else:
        rendered = _render_text(e) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
