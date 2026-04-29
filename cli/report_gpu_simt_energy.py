#!/usr/bin/env python
"""
Generate the GPU SIMT instruction-energy report.

Produces a markdown document with:

- The technology-profile primitives being used (the show-your-work
  reference table).
- 9 baseline-ALU tables (3 op kinds x 3 precisions) covering the
  irreducible-compute floor.
- 9 SIMT-pipeline tables (3 op kinds x 3 precisions) covering one
  Ampere SM-cycle of 128-lane execution.
- A cross-comparison summary normalizing per-op and per-FLOP.
- An architectural commentary section.

Standalone -- not wired into microarch_validation_report.py.

Usage:

    python cli/report_gpu_simt_energy.py \\
        --output reports/gpu_simt_energy.md
    python cli/report_gpu_simt_energy.py \\
        --profile edge-8nm-lpddr5 --output report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add repo root so `graphs.*` imports work when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from graphs.hardware.technology_profile import (  # noqa: E402
    TECHNOLOGY_PROFILES,
    TechnologyProfile,
)
from graphs.reporting.baseline_alu_energy import (  # noqa: E402
    BaselineALUEnergy,
    OpKind,
    Precision,
    baseline_alu_energy,
)
from graphs.reporting.gpu_simt_energy import (  # noqa: E402
    DEFAULT_LANES_PER_SUBPART,
    DEFAULT_SUBPARTITIONS,
    SIMTInstructionEnergy,
    STAGE_DESCRIPTIONS,
    STAGE_LABELS,
    simt_instruction_energy,
)


# --------------------------------------------------------------------
# Markdown rendering helpers
# --------------------------------------------------------------------

def _fmt_pj(v: float) -> str:
    """Format pJ with adaptive precision."""
    if v == 0:
        return "--"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def _render_primitive_table(profile: TechnologyProfile) -> str:
    """The reference table -- exactly what we pulled from
    technology_profile.py."""
    rows = [
        ("Process node",                    f"{profile.process_node_nm} nm"),
        ("Memory technology",               profile.memory_type.value.upper()),
        ("Target market",                   profile.target_market),
        ("Typical frequency",               f"{profile.typical_frequency_ghz:.2f} GHz"),
        ("Typical TDP",                     f"{profile.typical_tdp_w:.0f} W"),
        ("",                                ""),
        ("base_alu_energy_pj (FP32 FMA)",   f"{profile.base_alu_energy_pj:.3f} pJ"),
        ("simd_mac_energy_pj",              f"{profile.simd_mac_energy_pj:.3f} pJ"),
        ("tensor_core_mac_energy_pj",       f"{profile.tensor_core_mac_energy_pj:.3f} pJ"),
        ("",                                ""),
        ("instruction_fetch_energy_pj",     f"{profile.instruction_fetch_energy_pj:.3f} pJ"),
        ("instruction_decode_energy_pj",    f"{profile.instruction_decode_energy_pj:.3f} pJ"),
        ("instruction_dispatch_energy_pj",  f"{profile.instruction_dispatch_energy_pj:.3f} pJ"),
        ("",                                ""),
        ("register_read_energy_pj",         f"{profile.register_read_energy_pj:.3f} pJ"),
        ("register_write_energy_pj",        f"{profile.register_write_energy_pj:.3f} pJ"),
    ]
    out = ["| Field | Value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k or '&nbsp;'} | {v} |")
    return "\n".join(out)


# Baseline-table column order
_BASELINE_COLS = ["FF_read", "MUL", "ADD", "FF_write", "Total"]


def _render_baseline_table(b: BaselineALUEnergy) -> str:
    row = b.as_row()
    header = "| Stage | " + " | ".join(_BASELINE_COLS) + " |"
    sep    = "|---|" + "---|" * len(_BASELINE_COLS)
    cells = ["pJ"] + [_fmt_pj(row[c]) for c in _BASELINE_COLS]
    body = "| " + " | ".join(cells) + " |"
    return "\n".join([header, sep, body])


def _render_simt_table(s: SIMTInstructionEnergy) -> str:
    """Render the gantt-style table: rows = operations, cols = stages.

    Energy at each cell, totals at the right and bottom.
    """
    rows = list(s.rows.keys())
    cols = STAGE_LABELS
    header = "| Operation | " + " | ".join(cols) + " | Row total |"
    sep    = "|---|" + "---|" * (len(cols) + 1)

    lines = [header, sep]
    for r in rows:
        cells = [r]
        row_total = 0.0
        for c in cols:
            v = s.rows[r].get(c, 0.0)
            cells.append(_fmt_pj(v))
            row_total += v
        cells.append(_fmt_pj(row_total))
        lines.append("| " + " | ".join(cells) + " |")
    # Stage-total row
    stage_totals = [stg.total_pj for stg in s.stages]
    grand = sum(stage_totals)
    cells = ["**Stage total**"] + [_fmt_pj(v) for v in stage_totals] + [f"**{_fmt_pj(grand)}**"]
    lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------
# Report builder
# --------------------------------------------------------------------

def _build_report(profile_key: str) -> str:
    profile = TECHNOLOGY_PROFILES[profile_key]
    sub = DEFAULT_SUBPARTITIONS
    lanes = DEFAULT_LANES_PER_SUBPART

    parts: List[str] = []
    parts.append(f"# GPU SIMT instruction energy -- {profile.name}\n")
    parts.append(
        "Per-stage energy accounting for one Ampere SM-cycle of "
        f"{sub} subpartitions x {lanes} lanes = {sub * lanes} lane-ops, "
        "with all data resident in the per-subpartition register "
        "file (no L1, no shared memory, no off-chip traffic).\n"
    )

    # ---------- Reference primitives ----------
    parts.append("## 1. Technology-profile reference (source of truth)\n")
    parts.append(
        "All energy primitives in this report are derived from the "
        f"selected `TechnologyProfile` (`{profile_key}`). The full "
        "set of values flowing into the model:\n"
    )
    parts.append(_render_primitive_table(profile))
    parts.append("")
    parts.append(
        "Derived-from-base ratios used in this report:\n\n"
        "| Derived field | Formula |\n"
        "|---|---|\n"
        "| `ff_read_each_pj` | `register_read * 5%` (a directly-clocked flip-flop is far smaller than a banked RF cell) |\n"
        "| `oc_per_op_pj`    | `register_read * 5% * 2` (1 flop write + 1 flop read per operand) |\n"
        "| `alu_disp_pj`     | `register_read * 25%` (operand-collector to lane wire drive) |\n"
        "| `sched_pj`        | `instruction_decode * 50%` (scoreboard read + arbitration) |\n"
        "| Op-kind ALU ratio | FADD : FMUL : FMA = 0.22 : 0.88 : 1.00 (Horowitz 45nm tutorial) |\n"
        "| Precision scaling | FP32 : FP16 : INT8 = 1.00 : 0.50 : 0.20 (multiplier area scales with bits-squared) |\n"
        "| Packing factor    | FP32 : FP16 : INT8 = 1 : 2 : 4 (HFMA2 / DP4A semantics) |\n"
    )

    # ---------- Pipeline-stage glossary ----------
    parts.append("## 2. SIMT pipeline stages\n")
    parts.append(
        "Stages 1-4 fire ONCE PER SUBPARTITION (each Ampere "
        "subpartition has its own L0 I-cache, decoder, warp "
        "scheduler, and dispatch -- so they are counted x4). "
        "Stages 5-9 fire at the full SM lane count.\n"
    )
    parts.append("| # | Label | Description |\n|---|---|---|")
    for i, lbl in enumerate(STAGE_LABELS, start=1):
        parts.append(f"| {i} | {lbl} | {STAGE_DESCRIPTIONS[lbl]} |")
    parts.append("")

    # ---------- Baseline ALU ----------
    parts.append("## 3. Baseline ALU energy (irreducible compute floor)\n")
    parts.append(
        "Single ALU stripped of all SIMT overhead: input flip-flops "
        "wired directly to the ALU inputs, output flip-flop on the "
        "result. No register file, no operand collector, no "
        "instruction fetch / decode / scheduler / dispatch.\n"
    )
    for op in (OpKind.FADD, OpKind.FMUL, OpKind.FMA):
        parts.append(f"### 3.{op.value}\n")
        for prec in (Precision.FP32, Precision.FP16, Precision.INT8):
            b = baseline_alu_energy(profile, op, prec)
            parts.append(f"**{op.value} {prec.value}**  ({b.components.sources} source operands, {b.components.flops_per_op} FLOP/op)\n")
            parts.append(_render_baseline_table(b))
            parts.append(
                f"\nPer-op total: **{_fmt_pj(b.total_pj)} pJ**, "
                f"per-FLOP: **{_fmt_pj(b.pj_per_flop)} pJ/FLOP**.\n"
            )

    # ---------- SIMT pipeline ----------
    parts.append("## 4. SIMT pipeline energy (one SM-cycle, 128 lanes)\n")
    parts.append(
        "Rows trace each operand / control flow through the 9 "
        "pipeline stages; each cell is the energy attributable to "
        "that operation at that stage. The Stage total row at the "
        "bottom sums vertically; the Row total column on the right "
        "sums horizontally; the bold corner cell is the per-instruction "
        "total.\n"
    )
    for op in (OpKind.FADD, OpKind.FMUL, OpKind.FMA):
        parts.append(f"### 4.{op.value}\n")
        for prec in (Precision.FP32, Precision.FP16, Precision.INT8):
            s = simt_instruction_energy(profile, op, prec,
                                        sm_subpartitions=sub,
                                        lanes_per_subpartition=lanes)
            parts.append(
                f"**{op.value} {prec.value}** -- {s.lanes} lanes x "
                f"{s.packing_factor} packed = "
                f"{s.ops_executed} ops "
                f"({s.flops_executed} {'FLOPS' if prec is not Precision.INT8 else 'IntOPS'})\n"
            )
            parts.append(_render_simt_table(s))
            parts.append(
                f"\nPer-instruction: **{_fmt_pj(s.total_pj)} pJ**, "
                f"per-op: **{_fmt_pj(s.pj_per_op)} pJ/op**, "
                f"per-{'FLOP' if prec is not Precision.INT8 else 'IntOP'}: "
                f"**{_fmt_pj(s.pj_per_flop)} pJ**.\n"
            )

    # ---------- Cross-comparison ----------
    parts.append("## 5. Cross-comparison summary\n")
    parts.append(
        "Headline: per-op energy in the full SIMT pipeline vs the "
        "irreducible baseline ALU. Ratio = SIMT-overhead tax.\n"
    )
    header = (
        "| Op | Precision | Baseline pJ/op | SIMT pJ/op | "
        "SIMT pJ/FLOP | SIMT/Baseline | Compute % | RF+OC+wire % |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    parts.append(header)
    parts.append(sep)
    for op in (OpKind.FADD, OpKind.FMUL, OpKind.FMA):
        for prec in (Precision.FP32, Precision.FP16, Precision.INT8):
            b = baseline_alu_energy(profile, op, prec)
            s = simt_instruction_energy(profile, op, prec,
                                        sm_subpartitions=sub,
                                        lanes_per_subpartition=lanes)
            stage_by_label = {st.label: st.total_pj for st in s.stages}
            compute = stage_by_label["Exe"]
            rf_oc_wire = (stage_by_label["Rd"] + stage_by_label["OC"]
                          + stage_by_label["Disp"] + stage_by_label["WB"])
            grand = s.total_pj
            parts.append(
                f"| {op.value} | {prec.value} | "
                f"{_fmt_pj(b.total_pj)} | "
                f"{_fmt_pj(s.pj_per_op)} | "
                f"{_fmt_pj(s.pj_per_flop)} | "
                f"{s.pj_per_op / b.total_pj:.1f}x | "
                f"{100 * compute / grand:.1f}% | "
                f"{100 * rf_oc_wire / grand:.1f}% |"
            )
    parts.append("")

    # ---------- Commentary ----------
    parts.append("## 6. Architectural reading\n")
    parts.append(
        "1. **Cheap compute = high overhead share.** FADD has the "
        "smallest ALU energy per op, so a larger fraction of the "
        "instruction energy goes into RF reads, the operand "
        "collector, and the writeback. The SIMT/baseline overhead "
        "ratio is highest for FADD and lowest for FMA.\n\n"
        "2. **Packing recovers narrow-precision energy efficiency.** "
        "Per-instruction energy is approximately constant across "
        "fp32 / fp16-packed / int8-packed, because the same 32-bit "
        "datapath does 1, 2, or 4 useful ops respectively. Per-op "
        "energy halves (fp16) or quarters (int8) -- not from "
        "shrinking the RF, but from amortizing it over more useful "
        "work.\n\n"
        "3. **Per-subpartition control is x4 at SM level.** Each "
        "Ampere subpartition runs its own fetch / decode / scheduler "
        "/ dispatch. Counting any of these once at SM level (the "
        "naive accounting) under-reports the control-overhead share "
        "by 4x.\n\n"
        "4. **The TechnologyProfile RF energies are derived from a "
        "CPU-style scaling.** Published GPU RF reads at 8nm are "
        "typically 3-5x higher than the values used here, because "
        "GPU register files are larger, more banked, and have more "
        "ports. The model's relative energy decomposition is correct; "
        "absolute SIMT/baseline ratios scale with the assumed RF "
        "energy.\n\n"
        "5. **Operand collector is poorly characterized in public "
        "literature.** This report models it as a per-operand flop "
        "write + read. Real GPU operand collectors are more complex "
        "(multi-entry, cross-bar to RF banks). Treat the OC column "
        "as INTERPOLATED.\n"
    )

    # ---------- Validation gates ----------
    parts.append("## 7. Self-validation\n")
    parts.append(
        "These are the sanity checks the test suite runs against the "
        "model output. They are stable across TechnologyProfile "
        "changes within +/-30% RF energy.\n\n"
        "- fp16-packed / fp32 per-op ratio in [0.45, 0.55] for every op.\n"
        "- int8-packed / fp32 per-op ratio in [0.20, 0.30] for every op.\n"
        "- FMA ALU energy / FMUL ALU energy in [1.10, 1.18] (FMA = MUL + small ADD).\n"
        "- SIMT FADD overhead ratio > FMUL > FMA at every precision.\n"
        "- Stages 1-4 each fire `sm_subpartitions` times, not once.\n"
    )
    return "\n".join(parts)


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main(argv):
    p = argparse.ArgumentParser(
        prog="report_gpu_simt_energy",
        description=("Generate the GPU SIMT instruction-energy "
                     "report (markdown)."),
    )
    p.add_argument(
        "--profile", default="edge-8nm-lpddr5",
        choices=sorted(TECHNOLOGY_PROFILES.keys()),
        help="Technology profile key (default: edge-8nm-lpddr5 = Jetson Orin).",
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="Markdown output path.",
    )
    args = p.parse_args(argv)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_report(args.profile))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
