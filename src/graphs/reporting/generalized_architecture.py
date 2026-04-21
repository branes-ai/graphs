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
        instruction_fetch_fa_eq=30.0,   # amortized across 16-lane SIMD
        decode_fa_eq=15.0,              # x86-style decode
        scheduling_fa_eq=8.0,            # OOO issue + register rename + retire
        interconnect_fa_eq=0.0,
        register_byte_eq=3.0,            # 3 byte-access per MAC
        l1_byte_eq_amortized=0.5,        # 0.5 byte / MAC amortized
        notes=(
            "Largest per-MAC overhead of any mainstream architecture. "
            "Dynamic scheduling + branch prediction + register renaming "
            "are structurally un-amortizable beyond SIMD width."
        ),
    ),
    GeneralizedArchetype(
        name="GPU (SIMT, CUDA core)",
        category="GPU",
        color="#5b8ff9",
        description=(
            "Classical GPU compute without Tensor Cores. Warp-level "
            "lockstep execution across 32 threads. One MAC per thread "
            "per issue cycle. Represents baseline GPU efficiency."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=2.0,    # amortized across warp + ILP
        decode_fa_eq=1.0,
        scheduling_fa_eq=10.0,           # warp scheduling + divergence
        interconnect_fa_eq=0.0,
        register_byte_eq=6.0,            # warp register file, larger per access
        l1_byte_eq_amortized=0.3,
        notes=(
            "Warp-level amortization eliminates most fetch/decode cost. "
            "The main overhead is warp scheduling + divergence handling "
            "and the larger register file access energy."
        ),
    ),
    GeneralizedArchetype(
        name="GPU Tensor Core",
        category="GPU",
        color="#8bbafc",
        description=(
            "Warp-level matrix-multiply-accumulate instruction (HMMA). "
            "Amortizes scheduling over 4096 MACs per instruction. "
            "Lower floor than a SIMT CUDA core kernel for GEMM."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.5,
        decode_fa_eq=0.25,
        scheduling_fa_eq=3.0,
        interconnect_fa_eq=0.0,
        register_byte_eq=5.0,
        l1_byte_eq_amortized=0.5,
        notes=(
            "Matrix-native instruction buys a 4096x amortization of "
            "scheduling overhead. Register-file access is still the "
            "dominant memory-side cost."
        ),
    ),
    GeneralizedArchetype(
        name="TPU (Weight-Stationary Systolic)",
        category="TPU",
        color="#e98c3f",
        description=(
            "Systolic array with stationary weights. Inputs stream "
            "through row-by-row; partial sums accumulate in PE "
            "registers. Weight reload is double-buffered and amortized "
            "over the reduction dimension K."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,    # no per-MAC fetch
        decode_fa_eq=0.0,
        scheduling_fa_eq=0.0,            # statically scheduled
        interconnect_fa_eq=0.0,
        register_byte_eq=2.0,            # input read + accumulator update
        l1_byte_eq_amortized=0.05,       # UB load amortized by array row dim
        notes=(
            "Lowest theoretical floor among mainstream archetypes. "
            "Loss is not per-MAC - it is shape/tile mismatch against "
            "the fixed PE dimensions at workload level."
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
        interconnect_fa_eq=0.2,          # tile-to-tile NoC per MAC (amortized)
        register_byte_eq=3.0,            # PE-local: 2 reads + acc write
        l1_byte_eq_amortized=0.08,
        notes=(
            "Like TPU in having no per-MAC fetch/decode, but with an "
            "output-stationary schedule that gives near-1.0 utilization "
            "at real-workload tile counts. Slightly higher register "
            "cost than TPU (no weight stationarity) but real-workload "
            "efficiency wins."
        ),
    ),
    GeneralizedArchetype(
        name="DSP (VLIW + SIMD)",
        category="DSP",
        color="#d4860b",
        description=(
            "Very Long Instruction Word DSP with explicit SIMD lanes. "
            "Bundled instructions issue per cycle; no dynamic "
            "scheduling. Representative of TI C7x, Qualcomm Hexagon, "
            "or Hailo-style cores."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=4.0,     # VLIW bundle fetch amortized over lanes
        decode_fa_eq=1.5,
        scheduling_fa_eq=0.0,            # static, no OOO
        interconnect_fa_eq=0.0,
        register_byte_eq=2.5,
        l1_byte_eq_amortized=0.2,
        notes=(
            "Much simpler than OOO CPU but retains instruction fetch + "
            "decode cost. Absent dynamic scheduling keeps overhead low "
            "compared to SIMT GPU."
        ),
    ),
    GeneralizedArchetype(
        name="DFM (Data-Flow Machine, token-based)",
        category="DFM",
        color="#586374",
        description=(
            "Classical token-based dataflow. Instruction tokens "
            "stored in a CAM; ready tokens dispatch to PEs when all "
            "operands arrive. Stillwater DFM-128 is an instance."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,
        decode_fa_eq=0.0,
        scheduling_fa_eq=25.0,           # CAM-based token matching is expensive
        interconnect_fa_eq=10.0,         # token routing on 2D mesh
        register_byte_eq=2.0,
        l1_byte_eq_amortized=0.5,
        notes=(
            "Elegant in concept but the CAM-based token match is a "
            "structural cost per MAC. Distinct from KPU's domain-flow "
            "approach which is statically scheduled."
        ),
    ),
    GeneralizedArchetype(
        name="CGRA (Coarse-Grained Reconfigurable Array)",
        category="CGRA",
        color="#a04bdd",
        description=(
            "Spatial dataflow fabric with PE-level reconfiguration. "
            "Graph mapped to fabric, no per-MAC fetch. Representative "
            "of Plasticine, Samba Nova, or Xilinx Versal AI Engine."
        ),
        alu_fa_eq=8.0,
        instruction_fetch_fa_eq=0.0,
        decode_fa_eq=0.0,
        scheduling_fa_eq=0.0,
        interconnect_fa_eq=0.5,          # reconfigurable fabric routing
        register_byte_eq=2.0,
        l1_byte_eq_amortized=0.1,
        notes=(
            "Low per-MAC energy similar to TPU/KPU. Main cost is fabric "
            "reconfiguration overhead (paid per graph, not per MAC)."
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

    # Chart 3: peak throughput at fixed power budget
    # For each archetype at each process, compute peak MACs/sec if the
    # entire power budget were spent on MACs: peak_macs = P / E_per_MAC.
    # Peak TOPS = 2 * peak_macs / 1e12. This assumes 100% ALU duty which
    # is an upper bound; real workloads hit <100%.
    tdp_traces = []
    for a in report.archetypes:
        xs = report.process_nodes
        ys = [(report.power_budget_w / total_pj_per_mac(a, nm) / 1e12) * 2.0
              for nm in xs]
        tdp_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": a.name,
            "x": xs,
            "y": ys,
            "line": {"color": a.color, "width": 2},
            "marker": {"size": 8},
        })
    chart_tdp_capability = {
        "data": tdp_traces,
        "layout": {
            "title": (f"Peak INT8 TOPS at {report.power_budget_w:.0f} W TDP vs. "
                      "process node"),
            "xaxis": {
                "title": "Process node (nm)",
                "autorange": "reversed",
                "type": "log",
                "tickmode": "array",
                "tickvals": report.process_nodes,
                "ticktext": [str(n) for n in report.process_nodes],
            },
            "yaxis": {"title": "Peak INT8 TOPS (upper bound)", "type": "log"},
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
    rows = []
    for a in report.archetypes:
        comps = compute_components_pj_per_mac(a, ref)
        total = sum(comps.values())
        rows.append(
            f'<tr>'
            f'<td style="border-left:4px solid {a.color};padding-left:10px;">'
            f'<strong>{html.escape(a.name)}</strong></td>'
            f'<td>{html.escape(a.category)}</td>'
            f'<td class="num">{comps["ALU"]:.3f}</td>'
            f'<td class="num">{comps["Instruction fetch"] + comps["Decode"]:.3f}</td>'
            f'<td class="num">{comps["Scheduling / coherence"] + comps["Interconnect / routing"]:.3f}</td>'
            f'<td class="num">{comps["Register file"] + comps["L1 / scratchpad"]:.3f}</td>'
            f'<td class="num"><strong>{total:.3f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="generalized">'
        f'<thead><tr><th>Architecture</th><th>Category</th>'
        f'<th>ALU</th><th>Fetch+Decode</th><th>Schedule+Routing</th>'
        f'<th>Reg+L1</th><th>Total pJ/MAC @ {ref}nm</th></tr></thead>'
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
    <h3>Chart 3: Peak INT8 TOPS at {report.power_budget_w:.0f} W TDP vs. process node</h3>
    <p class="chart-desc">Capability envelope: given a fixed
      {report.power_budget_w:.0f} W power budget, how much peak INT8
      throughput can each architecture deliver at each process node?
      Upper bound - assumes 100% ALU duty cycle. Real workloads hit
      less (see compare_archetypes.html chart 4 for the utilization
      story).</p>
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
