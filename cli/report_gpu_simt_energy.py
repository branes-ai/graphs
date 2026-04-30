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
from graphs.reporting.gpu_register_file import (  # noqa: E402
    GPURegisterFileBankModel,
    default_ampere_subpartition_rf,
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
    """Single-row energy table (cells in pJ)."""
    row = b.as_row()
    header = "| Stage (pJ) | " + " | ".join(_BASELINE_COLS) + " |"
    sep    = "|---|" + "---|" * len(_BASELINE_COLS)
    cells = ["Energy"] + [_fmt_pj(row[c]) for c in _BASELINE_COLS]
    body = "| " + " | ".join(cells) + " |"
    return "\n".join([header, sep, body])


def _render_simt_table(s: SIMTInstructionEnergy) -> str:
    """Render the gantt-style energy table: rows = operations, cols =
    stages. Every cell is in **pJ** -- the product of activity-count
    and per-event energy. The corresponding activity counts are
    rendered separately by ``_render_activity_table`` so the reader
    can see how each cell decomposes.

    All values are SM-level totals (i.e. the ``Fch`` cell already
    includes the x4 subpartition fanout).
    """
    rows = list(s.rows.keys())
    cols = STAGE_LABELS
    header = "| Operation | " + " | ".join(cols) + " | Row total |"
    sep    = "|---|" + "---|" * (len(cols) + 1)

    lines = [
        "*All values in pJ; SM-level totals.*",
        "",
        header,
        sep,
    ]
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


def _render_activity_table(s: SIMTInstructionEnergy) -> str:
    """Render the activity-count breakdown for each pipeline stage.

    Shows how each ``Stage total`` in the energy table decomposes
    into ``activity_count x pJ_each``. This is what tells the reader
    "176 pJ = 4 wide-bank reads x 43.89 pJ" rather than mistakenly
    reading the energy cell as an access count.
    """
    lines = [
        "<details><summary>Stage activity (count x pJ/event)</summary>",
        "",
        "| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |",
        "|---|---|---|---|",
    ]
    for stg in s.stages:
        lines.append(
            f"| {stg.label} | {stg.activity_count} | "
            f"{_fmt_pj(stg.pj_each)} | {_fmt_pj(stg.total_pj)} |"
        )
    lines.append("")
    lines.append("</details>")
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

    # ---------- RF bank model ----------
    rf = default_ampere_subpartition_rf(profile)
    parts.append("## 1b. Banked SRAM register file (the SIMT energy story)\n")
    parts.append(
        "A SIMT pipeline only works when many warps are in flight "
        "concurrently; that requires hundreds-to-thousands of "
        "registers per subpartition, which can only come from a "
        "banked SRAM. The banked-SRAM cost IS the GPU's "
        "architectural overhead vs accelerators (TPU / KPU / CGRA) "
        "that either eliminate the general-purpose register file "
        "(systolic data flows bank-to-bank) or replace it with "
        "FIFOs (dataflow streams). Quantifying this cost is the "
        "purpose of this report.\n"
    )
    parts.append(
        "| Bank-model field | Value |\n|---|---|\n"
        f"| Bytes per subpartition | {rf.bytes_per_subpartition // 1024} KiB |\n"
        f"| Number of banks | {rf.num_banks} |\n"
        f"| Bank size | {rf.bank_size_bytes // 1024} KiB |\n"
        f"| Bank access width | {rf.bank_width_bits} bits ({rf.bytes_per_bank_access} bytes) |\n"
        f"| Per-byte SRAM dynamic energy | {rf.sram_energy_per_byte_pj:.3f} pJ "
        f"(`get_sram_energy_per_byte_pj({profile.process_node_nm}, 'register_file')`) |\n"
        f"| Wide-bank read energy | **{rf.bank_read_energy_pj():.2f} pJ** (per access) |\n"
        f"| Wide-bank write energy | **{rf.bank_write_energy_pj():.2f} pJ** (per access) |\n"
        f"| Reads per warp source operand | {rf.reads_per_warp_source()} (1024-bit bank "
        f"matches 32 threads x 32 bits exactly) |\n"
    )
    parts.append(
        "At the SM level, one SIMT instruction issues across "
        f"{DEFAULT_SUBPARTITIONS} subpartitions in parallel. RF "
        "activity per cycle:\n\n"
        "| Op kind | RF reads (SM-cycle) | RF writes (SM-cycle) |\n"
        "|---|---|---|\n"
        f"| FADD / FMUL (2 sources) | {rf.sm_bank_reads_per_instruction(DEFAULT_SUBPARTITIONS, 2)} | "
        f"{rf.sm_bank_writes_per_instruction(DEFAULT_SUBPARTITIONS)} |\n"
        f"| FMA (3 sources) | {rf.sm_bank_reads_per_instruction(DEFAULT_SUBPARTITIONS, 3)} | "
        f"{rf.sm_bank_writes_per_instruction(DEFAULT_SUBPARTITIONS)} |\n"
    )
    parts.append(
        "**Reading the Rd column in the SIMT tables.** \"8 wide-bank "
        "reads\" or \"12 wide-bank reads\" looks small until you "
        "realise one wide-bank read returns **the full warp's worth "
        "of one source operand at once** -- 32 threads x 32 bits = "
        "1024 bits = 128 bytes, delivered in a single banked-SRAM "
        "access. The reader ISN'T pulling 32 threads' worth of "
        "operand one byte at a time; the bank is wide enough to "
        "deliver the whole warp's source operand in one cycle. "
        f"Across {DEFAULT_SUBPARTITIONS} subpartitions "
        f"issuing the same instruction, the number of wide-bank "
        f"accesses is `subpartitions x sources_per_op x "
        f"reads_per_warp_source` -- which evaluates to 8 (FADD/FMUL) "
        "or 12 (FMA). Each one of those costs ~44 pJ at 8 nm "
        "(128 bytes x 0.343 pJ/byte SRAM dynamic energy). That's "
        "what makes the Rd column dominate the SIMT energy budget "
        "even though the access count looks tiny.\n"
    )

    # ---------- Primitives at a glance ----------
    rd_pj = rf.bank_read_energy_pj()
    wr_pj = rf.bank_write_energy_pj()
    base = profile.base_alu_energy_pj
    parts.append("## 1c. Primitives at a glance (where each is counted)\n")
    parts.append(
        "Quick reference for every per-event energy used in the "
        "report, what it represents, and how many times it fires "
        "per cycle.\n"
    )
    parts.append(
        "| Primitive | pJ at 8 nm | Counted in baseline | Counted per SIMT instr (SM-cycle) |\n"
        "|---|---|---|---|\n"
        f"| FF read 32b           | {0.05 * profile.register_read_energy_pj * 20:.3f} | 2x FADD/FMUL, 3x FMA | -- |\n"
        f"| FF write 32b          | {0.05 * profile.register_read_energy_pj * 20:.3f} | 1x | -- |\n"
        f"| ALU FADD 32b          | {base * 0.22:.3f} | 1x FADD or FMA | 128 lanes |\n"
        f"| ALU FMUL 32b          | {base * 0.88:.3f} | 1x FMUL or FMA | 128 lanes |\n"
        f"| **Wide-bank RF read** | **{rd_pj:.2f}** | -- | **8** (FADD/FMUL), **12** (FMA) |\n"
        f"| **Wide-bank RF write**| **{wr_pj:.2f}** | -- | **4** |\n"
        f"| OC flop write+read    | {rd_pj * 0.10:.2f} | -- | 8 / 12 (matches Rd count) |\n"
        f"| ALU dispatch wire     | {rd_pj * 0.25 / DEFAULT_LANES_PER_SUBPART:.3f} | -- | 256 (FADD/FMUL), 384 (FMA) |\n"
        f"| Instr fetch           | {profile.instruction_fetch_energy_pj:.3f} | -- | 4 (one per subpart) |\n"
        f"| Instr decode          | {profile.instruction_decode_energy_pj:.3f} | -- | 4 |\n"
        f"| Warp schedule         | {profile.instruction_decode_energy_pj * 0.5:.3f} | -- | 4 |\n"
        f"| Instr dispatch        | {profile.instruction_dispatch_energy_pj:.3f} | -- | 4 |\n"
    )
    parts.append(
        "The bolded rows are the SIMT energy story. Per-instruction "
        "RF traffic is `8x44 + 4x55 = 572 pJ` for a 2-source op or "
        "`12x44 + 4x55 = 747 pJ` for FMA -- which is more than the "
        "full ALU compute energy on every op in this set.\n"
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
        "instruction fetch / decode / scheduler / dispatch. **All "
        "values in pJ**, for one ALU performing one op.\n"
    )
    parts.append("### 3.0 The three baseline ALU shapes\n")
    parts.append(
        "Each baseline ALU is the same op embedded in the smallest "
        "possible circuit: input flip-flops on every source, ALU "
        "logic in the middle, one output flip-flop. The number of "
        "input flops varies with op kind (2 for FADD/FMUL, 3 for "
        "FMA); the ALU shape varies (single adder, single "
        "multiplier, or fused multiplier+adder).\n"
    )
    parts.append(
        "```text\n"
        "FADD (2-source):                     FMUL (2-source):\n"
        "                                                          \n"
        "   FF_A ----+                          FF_A ----+         \n"
        "            +--> [ ADD ] --> FF_OUT            +--> [ MUL ] --> FF_OUT\n"
        "   FF_B ----+                          FF_B ----+         \n"
        "                                                          \n"
        "                                                          \n"
        "FMA (3-source):                                           \n"
        "                                                          \n"
        "   FF_A ----+                                             \n"
        "            +--> [ MUL ] ----+                            \n"
        "   FF_B ----+                +--> [ ADD ] --> FF_OUT      \n"
        "                             |                            \n"
        "   FF_C ---------------------+                            \n"
        "```\n"
    )
    parts.append(
        "Energy decomposition per op (fp32, Samsung 8nm, "
        "primitives from the table in section 1c):\n\n"
        f"- **FADD**: 2 x FF_read + ALU FADD + FF_write = "
        f"`2 x {0.05 * profile.register_read_energy_pj * 20:.3f} + "
        f"{base * 0.22:.3f} + {0.05 * profile.register_read_energy_pj * 20:.3f}` "
        f"= ~{(2 * 0.05 * profile.register_read_energy_pj * 20) + base * 0.22 + (0.05 * profile.register_read_energy_pj * 20):.3f} pJ\n"
        f"- **FMUL**: 2 x FF_read + ALU FMUL + FF_write = "
        f"`2 x {0.05 * profile.register_read_energy_pj * 20:.3f} + "
        f"{base * 0.88:.3f} + {0.05 * profile.register_read_energy_pj * 20:.3f}` "
        f"= ~{(2 * 0.05 * profile.register_read_energy_pj * 20) + base * 0.88 + (0.05 * profile.register_read_energy_pj * 20):.3f} pJ\n"
        f"- **FMA**:  3 x FF_read + ALU MUL + ALU ADD + FF_write = "
        f"`3 x {0.05 * profile.register_read_energy_pj * 20:.3f} + "
        f"{base * 0.88:.3f} + {base * 0.12:.3f} + {0.05 * profile.register_read_energy_pj * 20:.3f}` "
        f"= ~{(3 * 0.05 * profile.register_read_energy_pj * 20) + base + (0.05 * profile.register_read_energy_pj * 20):.3f} pJ "
        f"(~{((3 * 0.05 * profile.register_read_energy_pj * 20) + base + (0.05 * profile.register_read_energy_pj * 20)) / 2:.3f} pJ/FLOP)\n"
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
        "pipeline stages; each cell is the **energy in pJ** "
        "attributable to that operation at that stage. All values "
        "are SM-level totals (the x4 subpartition fanout is already "
        "applied). The Stage total row at the bottom sums vertically; "
        "the Row total column on the right sums horizontally; the "
        "bold corner cell is the per-instruction total.\n"
    )
    parts.append(
        "Each precision section also has a collapsible \"Stage "
        "activity\" table showing how each Stage total decomposes "
        "into `activity_count x pJ_each` (e.g. an FMA fp32 Rd column "
        "of 527 pJ resolves to `12 wide-bank reads x 43.89 pJ each`). "
        "Treat the cells in the main energy table as **pJ totals**, "
        "not access counts.\n"
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
            parts.append("")
            parts.append(_render_activity_table(s))
            parts.append(
                f"\nPer-instruction: **{_fmt_pj(s.total_pj)} pJ**, "
                f"per-op: **{_fmt_pj(s.pj_per_op)} pJ/op**, "
                f"per-{'FLOP' if prec is not Precision.INT8 else 'IntOP'}: "
                f"**{_fmt_pj(s.pj_per_flop)} pJ**.\n"
            )

    # ---------- Cross-comparison ----------
    parts.append("## 5. The SIMT inefficiency tax\n")
    parts.append(
        "This is the architectural punchline of the whole report. "
        "We compare the energy of one op in the full SIMT pipeline "
        "(banked-SRAM RF, operand collector, per-subpartition "
        "scheduler, lane ALUs) against the energy of the *same op* "
        "in the irreducible baseline ALU (a few flip-flops + the "
        "ALU itself, no SIMT overhead at all).\n"
    )
    parts.append(
        "**Headline: GPUs pay a 6-10x energy tax for SIMT control "
        "and banked-RF traffic, even before you count anything off-"
        "chip.** The cheaper the op, the bigger the tax fraction:\n"
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
                f"**{s.pj_per_op / b.total_pj:.1f}x** | "
                f"{100 * compute / grand:.1f}% | "
                f"{100 * rf_oc_wire / grand:.1f}% |"
            )
    parts.append("")
    parts.append("### Reading this table\n")
    # Compute headline ratios for the narrative
    fadd_ratio = (
        simt_instruction_energy(profile, OpKind.FADD, Precision.FP32).pj_per_op
        / baseline_alu_energy(profile, OpKind.FADD, Precision.FP32).total_pj
    )
    fma_ratio = (
        simt_instruction_energy(profile, OpKind.FMA, Precision.FP32).pj_per_op
        / baseline_alu_energy(profile, OpKind.FMA, Precision.FP32).total_pj
    )
    parts.append(
        f"- **fp32 FADD**: SIMT pays **{fadd_ratio:.1f}x** the "
        "baseline-ALU energy. The adder itself is so cheap that "
        "the fixed RF + OC + control overhead is the entire show. "
        "FADD is the worst-case for SIMT efficiency in this set.\n\n"
        f"- **fp32 FMA**:  SIMT pays **{fma_ratio:.1f}x** the "
        "baseline. The multiplier is large enough that the SIMT "
        "overhead is a smaller share -- but it's still 4x. This is "
        "the *best* case for SIMT, on the *most* compute-dense op "
        "in the linear-algebra set.\n\n"
        "- **Narrow precision shrinks per-op energy proportionally** "
        "(fp16 packed -> 0.5x fp32, int8 packed -> 0.2-0.25x). The "
        "RF traffic is unchanged; the useful work scales with "
        "packing. So narrow precision wins not by reducing the SIMT "
        "tax but by amortizing it over more ops.\n\n"
        "- **The compute % column is the diagnostic.** When it's "
        "single-digit (FADD), the SIMT tax dominates and the "
        "architecture is severely mismatched to the workload. When "
        "it climbs into the 20-30% range (FMUL / FMA), the GPU is "
        "doing what it's built for. A TPU or KPU executing the same "
        "FADD would push compute % past 70% by eliminating the RF "
        "and the per-op control entirely -- which is the topline "
        "energy comparison we're trying to make.\n"
    )

    # ---------- Commentary ----------
    parts.append("## 6. Architectural reading\n")
    parts.append(
        "1. **Banked SRAM RF traffic dominates SIMT energy.** Stage "
        "5 (Rd) + stage 9 (WB) together exceed the ALU compute (stage "
        "8) for every (op, precision) combo. This is the GPU's "
        "architectural \"tax\" for keeping hundreds of warps in "
        "flight: a 64 KiB banked RF per subpartition that costs "
        "tens of pJ per wide-bank access. Accelerators that "
        "eliminate the RF (TPU systolic, KPU dataflow, CGRA spatial) "
        "skip this tax entirely -- which is exactly the energy gap "
        "this report quantifies.\n\n"
        "2. **Cheap compute = high overhead share.** FADD has the "
        "smallest ALU energy per op, so a larger fraction of the "
        "instruction energy is fixed RF + control overhead. The "
        "SIMT/baseline ratio is highest for FADD and lowest for "
        "FMA.\n\n"
        "3. **Packing recovers narrow-precision energy efficiency.** "
        "Per-instruction energy is approximately constant across "
        "fp32 / fp16-packed / int8-packed, because the same 32-bit "
        "datapath does 1, 2, or 4 useful ops respectively, AND the "
        "same wide-bank RF reads supply data regardless of packing. "
        "Per-op energy halves (fp16) or quarters (int8) -- not from "
        "shrinking the RF, but from amortizing it over more useful "
        "work.\n\n"
        "4. **Per-subpartition control is x4 at SM level.** Each "
        "Ampere subpartition runs its own fetch / decode / scheduler "
        "/ dispatch. Counting any of these once at SM level (the "
        "naive accounting) under-reports the control-overhead share "
        "by 4x. Same applies to the RF: each subpartition's banks "
        "fire independently in parallel.\n\n"
        "5. **The wide-bank read assumption.** This model assumes the "
        "1024-bit bank width matches a warp's source operand exactly "
        "(32 threads x 32 bits) so 1 wide-bank read suffices per "
        "source. Real GPUs have register-bank-conflict cycles when "
        "two source operands map to the same bank; the operand "
        "collector hides these by buffering operands across cycles. "
        "Modeling bank conflicts is future work; the current model "
        "captures the no-conflict case (the optimistic floor for "
        "RF energy).\n\n"
        "6. **Operand collector is poorly characterized in public "
        "literature.** This report models it as a per-operand wide-"
        "buffer flop write + read at 5% of bank-read energy. Real "
        "GPU operand collectors are multi-entry with crossbars to "
        "RF banks. Treat the OC column as INTERPOLATED.\n"
    )

    # ---------- Validation gates ----------
    parts.append("## 7. Self-validation\n")
    parts.append(
        "These are the sanity checks the test suite runs against the "
        "model output.\n\n"
        "- fp16-packed / fp32 per-op ratio in [0.45, 0.55] for every op.\n"
        "- int8-packed / fp32 per-op ratio in [0.20, 0.30] for every op.\n"
        "- FMA ALU energy / FMUL ALU energy in [1.10, 1.18] (FMA = MUL + small ADD).\n"
        "- SIMT FADD overhead ratio > FMUL > FMA at every precision.\n"
        "- Stages 1-4 each fire `sm_subpartitions` times, not once.\n"
        "- **RF traffic (Rd + WB) > ALU compute (Exe).** This is the "
        "  architectural punchline; if it ever flips, the bank model "
        "  is mis-parameterised.\n"
        "- Default Ampere bank model: 4 banks of 16 KiB / 1024-bit wide "
        "  -> 1 wide-bank read per warp source operand.\n"
        "- Per-bank read energy at 8 nm in [30, 80] pJ.\n"
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
