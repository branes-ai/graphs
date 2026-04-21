"""
Generalized-architecture comparison: architectural-efficiency floors
for CPU, GPU (SIMT), GPU (Tensor Core), TPU, KPU, DSP, DFM, and CGRA
at the *same* manufacturing process, plus process-scaling curves.

This complements ``native_op_energy.py`` which compares specific
shipping products (Coral 14nm vs. KPU 16nm vs. Jetson 8nm). That
view mixes process advantages with architectural advantages. This
module strips out the process confound and reports:

  (1) Per-MAC energy floor for each generic archetype at a user-
      selectable process node (default 16 nm).
  (2) Process-scaling curves for each archetype across 45/28/16/
      14/12/10/8/7/5/4/3/2 nm nodes.
  (3) Peak throughput at a fixed TDP budget as a function of
      process node for each archetype.

Archetypes are characterized by full-adder-equivalent (FA-eq)
multipliers for logic components and byte-access equivalents for
memory components. Both scale naturally with process node via the
tables in ``native_op_energy.py`` (FA energies from Horowitz 2014
and subsequent process scaling).

Values are analytical estimates calibrated against published figures
(Jouppi ISCA 2017 for TPU; Ampere whitepapers for Tensor Core;
Horowitz ISSCC 2014 for CPU overhead). They are not silicon
measurements - the bottom-up validation plan grades them from
THEORETICAL to CALIBRATED.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphs.reporting.microarch_html_template import (
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
)
from graphs.reporting.native_op_energy import (
    FULL_ADDER_PJ_BY_PROCESS,
    REG_ACCESS_PJ_PER_BYTE_BY_PROCESS,
    register_pj_per_byte,
    _fa_pj,
)


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class GeneralizedArchetype:
    """
    A generic architectural pattern expressed in process-node-agnostic
    units (full-adder equivalents for logic, byte-access equivalents
    for memory).

    Total per-MAC energy at a given process node:
        E(proc) = logic_fa_eq * FA(proc) + memory_byte_eq * REG(proc)
                 + l1_byte_eq * L1_BYTE(proc)

    where FA(proc) is the 1-bit full-adder dynamic energy,
    REG(proc) is per-byte register-file access energy, and
    L1_BYTE(proc) is per-byte L1 SRAM access energy (~3x REG).

    The components are named so the stacked-bar view can show which
    architecture pays its budget where.
    """
    name: str                       # "CPU (OOO superscalar)"
    category: str                   # "CPU", "GPU", "TPU", "KPU", "DSP", "DFM", "CGRA"
    color: str
    description: str

    # Logic components, in FA-equivalents per INT8 MAC (amortized through
    # whatever native parallelism the architecture provides - SIMD width,
    # warp size, systolic-array population, etc.).
    alu_fa_eq: float                # bare MAC (8 for ideal INT8 MAC)
    instruction_fetch_fa_eq: float  # I-cache + TLB + branch pred, amortized
    decode_fa_eq: float             # instruction decode, amortized
    scheduling_fa_eq: float         # warp/token/coherence scheduling
    interconnect_fa_eq: float       # NoC / mesh / crossbar traversal

    # Memory components, in byte-access equivalents per MAC
    register_byte_eq: float         # register-file reads + writes per MAC
    l1_byte_eq_amortized: float     # L1/scratchpad bytes per MAC (after reuse)

    # Realistic effective utilization on representative workloads.
    # This is the structural ceiling set by the architecture's scheduling
    # class + memory subsystem, NOT the marketing peak. Used for Chart 3
    # (sustained TOPS at fixed TDP). Citation per archetype in notes.
    realistic_utilization: float = 1.0

    # Notes / source
    notes: str = ""


def compute_components_pj_per_mac(
    arch: GeneralizedArchetype, process_nm: int,
) -> Dict[str, float]:
    """Return each component's per-MAC energy in pJ at the given process node."""
    fa = _fa_pj(process_nm)
    reg = register_pj_per_byte(process_nm)
    # L1 SRAM access energy per byte is ~3x register access at the same node
    # (larger array, longer wires). Simple multiplicative factor; scales with
    # process like the register line.
    l1_byte = reg * 3.0

    return {
        "ALU": arch.alu_fa_eq * fa,
        "Instruction fetch": arch.instruction_fetch_fa_eq * fa,
        "Decode": arch.decode_fa_eq * fa,
        "Scheduling / coherence": arch.scheduling_fa_eq * fa,
        "Interconnect / routing": arch.interconnect_fa_eq * fa,
        "Register file": arch.register_byte_eq * reg,
        "L1 / scratchpad": arch.l1_byte_eq_amortized * l1_byte,
    }


def total_pj_per_mac(arch: GeneralizedArchetype, process_nm: int) -> float:
    return sum(compute_components_pj_per_mac(arch, process_nm).values())


# ----------------------------------------------------------------------
# Canonical archetype definitions
# ----------------------------------------------------------------------
#
# Calibrated to give plausible per-MAC energies at 16 nm INT8:
#   CPU (OOO):            ~1.2 pJ/MAC
#   GPU SIMT (CUDA core): ~0.55 pJ/MAC
#   GPU Tensor Core:      ~0.25 pJ/MAC
#   TPU (WS systolic):    ~0.13 pJ/MAC
#   KPU (domain flow):    ~0.19 pJ/MAC
#   DSP (VLIW):           ~0.25 pJ/MAC
#   DFM (token flow):     ~0.55 pJ/MAC
#   CGRA (spatial):       ~0.17 pJ/MAC
#
# Logic FA-eq: ALU=8 is the theoretical floor for INT8 MAC; overhead
# components represent amortized costs per MAC (fetch/decode divided
# by SIMD width or warp size or systolic array population).


CANONICAL_ARCHETYPES: List[GeneralizedArchetype] = [
    GeneralizedArchetype(
        name="CPU (OOO superscalar, SIMD)",
        category="CPU",
        color="#7f3b8d",
        description=(
            "Out-of-order superscalar with SIMD / AVX-style vector "
            "extensions. Instruction fetch / decode / issue amortized "
            "across a 16-lane SIMD MAC. Representative of x86 or ARM "
            "Neoverse-class cores."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=30.0,
        decode_fa_eq=15.0,
        scheduling_fa_eq=8.0,
        interconnect_fa_eq=0.0,
        register_byte_eq=3.0,
        l1_byte_eq_amortized=0.5,
        realistic_utilization=0.15,
        notes=(
            "Largest per-MAC overhead of any mainstream architecture. "
            "Realistic utilization on ML-like workloads ~0.15 "
            "(memory-bound, branch-heavy). Dynamic scheduling + branch "
            "prediction + register renaming are structurally un-"
            "amortizable beyond SIMD width."
        ),
    ),
    GeneralizedArchetype(
        name="GPU (SIMT, CUDA core)",
        category="GPU",
        color="#5b8ff9",
        description=(
            "Classical GPU compute without Tensor Cores. Warp-level "
            "lockstep execution across 32 threads."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=2.0,
        decode_fa_eq=1.0,
        scheduling_fa_eq=10.0,
        interconnect_fa_eq=0.0,
        register_byte_eq=6.0,
        l1_byte_eq_amortized=0.3,
        realistic_utilization=0.35,
        notes=(
            "Realistic utilization ~0.35 on mixed workloads - warp "
            "divergence + memory subsystem contention + occupancy "
            "limits. Better on pure GEMM (~0.50) but worse on "
            "irregular kernels."
        ),
    ),
    GeneralizedArchetype(
        name="GPU Tensor Core",
        category="GPU",
        color="#8bbafc",
        description=(
            "Warp-level matrix-multiply-accumulate instruction (HMMA). "
            "Amortizes scheduling over 4096 MACs per instruction."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.5,
        decode_fa_eq=0.25,
        scheduling_fa_eq=3.0,
        interconnect_fa_eq=0.0,
        register_byte_eq=5.0,
        l1_byte_eq_amortized=0.5,
        realistic_utilization=0.50,
        notes=(
            "Realistic utilization ~0.50 on well-tuned HMMA kernels. "
            "Warp divergence is less of a problem inside the HMMA "
            "primitive itself, but still present in surrounding code."
        ),
    ),
    GeneralizedArchetype(
        name="TPU (Weight-Stationary Systolic)",
        category="TPU",
        color="#e98c3f",
        description=(
            "Systolic array with stationary weights. Inputs stream in "
            "row-by-row; partial sums accumulate in PE registers."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,
        decode_fa_eq=0.0,
        scheduling_fa_eq=0.0,
        interconnect_fa_eq=0.0,
        register_byte_eq=2.0,
        l1_byte_eq_amortized=0.05,
        realistic_utilization=0.40,
        notes=(
            "Realistic utilization ~0.40 on well-tuned dense GEMM "
            "(Jouppi ISCA 2017 Table 3 cites 10-25% on TPU v1 "
            "production workloads, higher for tuned kernels). The "
            "structural loss is shape/tile mismatch against fixed PE "
            "dimensions, NOT per-MAC overhead."
        ),
    ),
    GeneralizedArchetype(
        name="KPU (Domain Flow)",
        category="KPU",
        color="#3fc98a",
        description=(
            "Distributed domain-flow fabric executing statically "
            "scheduled systems of affine recurrence equations. "
            "Output-stationary schedule; fill/drain overlaps across "
            "adjacent tiles on the fabric."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,
        decode_fa_eq=0.0,
        scheduling_fa_eq=0.0,
        interconnect_fa_eq=0.2,
        register_byte_eq=3.0,
        l1_byte_eq_amortized=0.08,
        realistic_utilization=0.90,
        notes=(
            "Realistic utilization ~0.90 on real-workload tile counts "
            "M >= 12. Output-stationary scheduling amortizes fill/drain "
            "across the workload's tile decomposition. This is THE "
            "structural advantage - not a lower MAC floor than TPU."
        ),
    ),
    GeneralizedArchetype(
        name="DSP (VLIW + SIMD)",
        category="DSP",
        color="#d4860b",
        description=(
            "VLIW DSP with explicit SIMD lanes. Bundled instructions "
            "issue per cycle; no dynamic scheduling."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=4.0,
        decode_fa_eq=1.5,
        scheduling_fa_eq=0.0,
        interconnect_fa_eq=0.0,
        register_byte_eq=2.5,
        l1_byte_eq_amortized=0.2,
        realistic_utilization=0.60,
        notes=(
            "Realistic utilization ~0.60 on well-tuned kernels. VLIW "
            "bundles are statically scheduled by the compiler so the "
            "utilization story depends on instruction-level parallelism "
            "in the workload."
        ),
    ),
    GeneralizedArchetype(
        name="DFM (Data-Flow Machine, token-based)",
        category="DFM",
        color="#586374",
        description=(
            "Classical token-based dataflow with CAM-stored "
            "instruction tokens. Stillwater DFM-128 is an instance."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,
        decode_fa_eq=0.0,
        scheduling_fa_eq=25.0,
        interconnect_fa_eq=10.0,
        register_byte_eq=2.0,
        l1_byte_eq_amortized=0.5,
        realistic_utilization=0.50,
        notes=(
            "Realistic utilization ~0.50. Token-matching latency is "
            "predictable but adds structural per-MAC overhead via the "
            "CAM and token-routing network. Distinct from KPU's "
            "statically-scheduled domain-flow."
        ),
    ),
    GeneralizedArchetype(
        name="CGRA (Coarse-Grained Reconfigurable Array)",
        category="CGRA",
        color="#a04bdd",
        description=(
            "Spatial dataflow fabric with PE-level reconfiguration "
            "(Plasticine, SambaNova, Xilinx Versal AI Engine). "
            "Graph mapped to fabric; no per-MAC fetch, but the "
            "reconfigurable switch network and configuration memory "
            "are structural costs paid on every MAC."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,
        decode_fa_eq=0.0,
        # Config-memory readout + switch state maintenance per MAC.
        # Not per-reconfig - the reconfigurable switches and config
        # SRAM sit in the datapath and cost energy every cycle.
        scheduling_fa_eq=5.0,
        # Reconfigurable switch network is structurally heavier than a
        # fixed NoC because every switch must read its config bit each
        # cycle. Representative: 6x a fixed-fabric interconnect.
        interconnect_fa_eq=3.0,
        register_byte_eq=2.0,
        l1_byte_eq_amortized=0.1,
        # Reconfiguration events add macroscopic dead time on top of
        # the per-MAC cost. FPGAs and CGRAs are never energy-
        # competitive with fixed-function dataflow fabrics because of
        # the combined per-cycle config overhead + reconfig dead time.
        realistic_utilization=0.25,
        notes=(
            "Realistic utilization ~0.25 after reconfig dead time is "
            "factored in. Per-MAC overhead is elevated by the reconfig "
            "switch network and config-SRAM readout which the fabric "
            "pays EVERY cycle - not just at reconfig events. Combined "
            "with reconfig dead time this is why FPGAs/CGRAs are not "
            "energy-competitive for sustained AI workloads."
        ),
    ),
]


# ----------------------------------------------------------------------
# Representative process nodes (omit 45nm to keep y-axes readable)
# ----------------------------------------------------------------------

DEFAULT_PROCESS_NODES = [28, 22, 16, 14, 12, 10, 8, 7, 5, 4, 3, 2]


# ----------------------------------------------------------------------
# Report schema
# ----------------------------------------------------------------------

@dataclass
class GeneralizedReport:
    archetypes: List[GeneralizedArchetype]
    reference_process_nm: int = 16
    process_nodes: List[int] = field(default_factory=lambda: list(DEFAULT_PROCESS_NODES))
    power_budget_w: float = 12.0
    generated_at: str = ""


def build_default_report(
    reference_process_nm: int = 16,
    power_budget_w: float = 12.0,
) -> GeneralizedReport:
    from datetime import datetime, timezone
    return GeneralizedReport(
        archetypes=CANONICAL_ARCHETYPES,
        reference_process_nm=reference_process_nm,
        process_nodes=list(DEFAULT_PROCESS_NODES),
        power_budget_w=power_budget_w,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Component ordering matches the stacked-bar trace stack
_COMPONENT_ORDER = [
    "ALU",
    "Instruction fetch",
    "Decode",
    "Scheduling / coherence",
    "Interconnect / routing",
    "Register file",
    "L1 / scratchpad",
]

_COMPONENT_COLORS = {
    "ALU": "#3fc98a",
    "Instruction fetch": "#7f3b8d",
    "Decode": "#a04bdd",
    "Scheduling / coherence": "#5b8ff9",
    "Interconnect / routing": "#8bbafc",
    "Register file": "#d4860b",
    "L1 / scratchpad": "#e98c3f",
}


def _render_chart_js(report: GeneralizedReport) -> str:
    """Emit Plotly chart definitions as a single JS block."""
    ref_nm = report.reference_process_nm

    # Chart 1: stacked bar at reference process - component breakdown
    component_traces = []
    arch_names = [a.name for a in report.archetypes]
    for comp_name in _COMPONENT_ORDER:
        ys = []
        for a in report.archetypes:
            comps = compute_components_pj_per_mac(a, ref_nm)
            ys.append(comps.get(comp_name, 0.0))
        component_traces.append({
            "type": "bar",
            "orientation": "h",
            "name": comp_name,
            "y": arch_names,
            "x": ys,
            "marker": {"color": _COMPONENT_COLORS.get(comp_name, "#888")},
            "text": [f"{v:.3f}" if v > 0.005 else "" for v in ys],
            "textposition": "inside",
        })
    chart_same_process = {
        "data": component_traces,
        "layout": {
            "title": (f"Per-MAC energy at {ref_nm} nm: generic architectures "
                      "(component breakdown)"),
            "xaxis": {"title": "pJ / MAC"},
            "yaxis": {"title": "", "automargin": True},
            "barmode": "stack",
            "margin": {"t": 50, "b": 50, "l": 240, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    # Chart 2: process-scaling - per-MAC energy vs. node per archetype
    scale_traces = []
    for a in report.archetypes:
        xs = report.process_nodes
        ys = [total_pj_per_mac(a, nm) for nm in xs]
        scale_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": a.name,
            "x": xs,
            "y": ys,
            "line": {"color": a.color, "width": 2},
            "marker": {"size": 8},
        })
    chart_process_scaling = {
        "data": scale_traces,
        "layout": {
            "title": "Per-MAC energy vs. process node, by architecture",
            "xaxis": {
                "title": "Process node (nm)",
                "autorange": "reversed",
                "type": "log",
                "tickmode": "array",
                "tickvals": report.process_nodes,
                "ticktext": [str(n) for n in report.process_nodes],
            },
            "yaxis": {"title": "pJ / MAC", "type": "log"},
            "margin": {"t": 50, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.3},
        },
    }

    # Chart 3: SUSTAINED throughput at fixed power budget.
    # Peak TOPS * realistic_utilization. The utilization factor captures
    # each architecture's structural ceiling - TPU's shape-mismatch +
    # bandwidth cap, KPU's near-1.0 output-stationary saturation, GPU's
    # warp divergence, CGRA's reconfig dead time. No 100% duty-cycle
    # illusion.
    tdp_traces = []
    for a in report.archetypes:
        xs = report.process_nodes
        # peak_TOPS = 2 * TDP_W / pJ_per_MAC (unit cancellation: pJ = J*1e-12,
        # TOPS = ops/s*1e-12, so (W/pJ)*2 = TOPS directly)
        # sustained_TOPS = peak_TOPS * realistic_utilization
        ys = [
            2.0 * report.power_budget_w / total_pj_per_mac(a, nm)
            * a.realistic_utilization
            for nm in xs
        ]
        tdp_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": f"{a.name} (util {a.realistic_utilization:.2f})",
            "x": xs,
            "y": ys,
            "line": {"color": a.color, "width": 2},
            "marker": {"size": 8},
        })
    chart_tdp_capability = {
        "data": tdp_traces,
        "layout": {
            "title": (f"Sustained INT8 TOPS at {report.power_budget_w:.0f} W TDP "
                      "vs. process node (realistic utilization)"),
            "xaxis": {
                "title": "Process node (nm)",
                "autorange": "reversed",
                "type": "log",
                "tickmode": "array",
                "tickvals": report.process_nodes,
                "ticktext": [str(n) for n in report.process_nodes],
            },
            "yaxis": {"title": "Sustained INT8 TOPS", "type": "log"},
            "margin": {"t": 50, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.3},
        },
    }

    payload = {
        "chart_same_process": chart_same_process,
        "chart_process_scaling": chart_process_scaling,
        "chart_tdp_capability": chart_tdp_capability,
    }
    return (
        f"const CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_archetype_table(report: GeneralizedReport) -> str:
    ref = report.reference_process_nm
    tdp = report.power_budget_w
    rows = []
    for a in report.archetypes:
        comps = compute_components_pj_per_mac(a, ref)
        total = sum(comps.values())
        # peak_TOPS = 2 * TDP / pJ_per_MAC (pJ and 1e-12 cancel out);
        # sustained_TOPS = peak * realistic_utilization
        peak_tops = 2.0 * tdp / total
        sustained_tops = peak_tops * a.realistic_utilization
        rows.append(
            f'<tr>'
            f'<td style="border-left:4px solid {a.color};padding-left:10px;">'
            f'<strong>{html.escape(a.name)}</strong></td>'
            f'<td>{html.escape(a.category)}</td>'
            f'<td class="num">{total:.3f}</td>'
            f'<td class="num">{a.realistic_utilization:.2f}</td>'
            f'<td class="num">{peak_tops:.1f}</td>'
            f'<td class="num"><strong>{sustained_tops:.1f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="generalized">'
        f'<thead><tr><th>Architecture</th><th>Category</th>'
        f'<th>Total pJ/MAC @ {ref}nm</th><th>Realistic util</th>'
        f'<th>Peak TOPS @ {tdp:.0f}W</th>'
        f'<th>Sustained TOPS @ {tdp:.0f}W</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_generalized_page(
    report: GeneralizedReport,
    repo_root: Path,
) -> str:
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Generalized architecture comparison",
        (f"Apples-to-apples at {report.reference_process_nm} nm "
         f"| Power budget {report.power_budget_w:.0f} W "
         f"| Generated {report.generated_at}"),
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.generalized { width: 100%; border-collapse: collapse; background: #fff;
                    margin-bottom: 18px; }
table.generalized th, table.generalized td { padding: 8px 10px;
                                              border-bottom: 1px solid #e3e6eb;
                                              vertical-align: top; font-size: 13px; }
table.generalized th { font-size: 12px; text-transform: uppercase;
                       color: #586374; background: #f3f5f8; text-align: left; }
table.generalized td.num { text-align: right; font-variant-numeric: tabular-nums; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin-bottom: 18px; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
  margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px; margin: 0 0 12px; }
.plot { width: 100%; min-height: 420px; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    method_note = (
        '<section class="method-note">'
        '<strong>Methodology.</strong> Each generic archetype is '
        'specified by full-adder-equivalent multipliers for its logic '
        'components (bare MAC, fetch/decode, scheduling/coherence, '
        'interconnect) and byte-access-equivalents for its memory '
        'components (register file, L1/scratchpad). The same multipliers '
        'apply at every process node; only the base energies '
        '(FA per process, register-access per process) vary. This strips '
        'out process-technology advantages so pure architectural cost '
        'can be compared. Logic energies scale via Horowitz ISSCC 2014 '
        'FA table; memory energies scale similarly. Values are analytical '
        'estimates calibrated against Jouppi ISCA 2017 (TPU), Ampere '
        'whitepapers (Tensor Core), and published CPU/GPU benchmarks. '
        'They ship as THEORETICAL confidence and graduate to CALIBRATED '
        'as silicon measurements come in.'
        '</section>'
    )

    complement_note = (
        '<section class="method-note" style="border-left-color:#3fc98a;">'
        '<strong>Companion view:</strong> this page compares <em>generic '
        'architectures at matched process</em>. For comparison of '
        'specific <em>shipping products</em> with their native process '
        'nodes (Coral 14nm vs. KPU 16nm vs. Jetson 8nm), see '
        '<a href="native_op_energy.html">native_op_energy.html</a>.'
        '</section>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Generalized architecture comparison</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}
{extra_css}
  </style>
</head>
<body>
{header}
<main>
  <p><a class="nav-back" href="index.html">&larr; Back to index</a></p>
  <section class="page-header">
    <h2>Apples-to-apples architectural comparison</h2>
    <div class="meta">CPU vs. GPU vs. TPU vs. KPU vs. DSP vs. DFM vs. CGRA
      at the same process - pure architectural pros/cons.</div>
  </section>
  {method_note}
  {complement_note}
  <section class="chart-section">
    <h3>Chart 1: Per-MAC energy at {report.reference_process_nm} nm, component breakdown</h3>
    <p class="chart-desc">Each stacked bar shows the generic architecture's
      per-MAC energy decomposed by contributing component. Process-
      technology advantage is held constant so bar length measures
      pure architectural cost.</p>
    <div id="chart_same_process" class="plot"></div>
  </section>
  {_render_archetype_table(report)}
  <section class="chart-section">
    <h3>Chart 2: Process-scaling - pJ/MAC vs. node per architecture</h3>
    <p class="chart-desc">How each architecture scales from 28 nm down
      to 2 nm. Log-log axes: ratios between architectures stay
      roughly constant; absolute energies drop as process shrinks.</p>
    <div id="chart_process_scaling" class="plot"></div>
  </section>
  <section class="chart-section">
    <h3>Chart 3: Sustained INT8 TOPS at {report.power_budget_w:.0f} W TDP vs. process node</h3>
    <p class="chart-desc">The chart that decides which architecture wins
      for your application: sustained throughput = peak * realistic
      utilization. Marketing peak-TOPS numbers assume 100% ALU duty
      cycle, which is structurally impossible for WS systolic (shape
      mismatch caps util at ~0.40), CGRA (reconfig dead time caps at
      ~0.25), SIMT GPU (warp divergence), and CPU (memory-bound
      ~0.15). KPU output-stationary scheduling saturates toward 0.90,
      so its sustained TOPS is the highest even though its theoretical
      MAC floor is higher than TPU's.</p>
    <div id="chart_tdp_capability" class="plot"></div>
  </section>
</main>
{footer}
<script>
{_render_chart_js(report)}
</script>
</body>
</html>
"""
