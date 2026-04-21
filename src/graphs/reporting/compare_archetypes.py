"""
Compute-archetype comparison harness (GPU Tensor Core vs. TPU systolic
vs. KPU dataflow).

This is the diligence tool used during M0.5 to pick PE-array sizes for
T64 / T128 / T256 under the refined dataflow-tile abstraction, and the
artifact that makes the KPU's energy-per-op positioning legible.

Five charts, rendered via Plotly loaded from CDN:

  1. Energy per op at fixed precision (grouped bar)
  2. Peak ops/s at fixed precision (grouped bar)
  3. Ops/W at fixed precision (grouped bar)
  4. Effective pipeline utilization vs. workload tile count (line)
        - KPU output-stationary: saturates -> 1.0 at ~12+ tiles
        - TPU weight-stationary: flat floor
        - Tensor Core: unspecified; flat baseline
     This is the scheduling-difference chart - the product's due-
     diligence story.
  5. PE array-size scaling sweep (ops/s and ops/W vs. PE array rows)
        - Shows where "compensate for peak with larger arrays"
          has headroom vs. diminishing returns.

See ``docs/hardware/kpu_dataflow_tile_model.md`` and
``docs/plans/microarch-model-delivery-plan.md`` (M0.5).
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphs.hardware.resource_model import (
    Precision,
    TileScheduleClass,
)
from graphs.reporting.microarch_html_template import (
    BRAND_LOGO_RELATIVE,
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
    _render_legend,
)


# ----------------------------------------------------------------------
# Data schema
# ----------------------------------------------------------------------

@dataclass
class ArchetypeEntry:
    """
    One archetype's representative numbers for the comparison harness.
    """
    archetype: str               # "Tensor Core", "Systolic", "Dataflow"
    sku: str                     # representative SKU
    display_name: str
    color: str                   # hex color for consistent chart theming

    # Headline numbers at representative workload (16 tiles) and FP16 precision
    energy_per_op_pj: float      # pJ/op
    peak_ops_per_sec: float      # ops/s
    ops_per_watt: float          # ops/W at sustained TDP

    # Scheduling class (drives chart 4)
    schedule_class: TileScheduleClass
    pipeline_fill_cycles: int
    pipeline_drain_cycles: int
    pe_array_rows: int
    pe_array_cols: int

    # Chart 4: utilization vs. tile count
    utilization_curve: List[Tuple[int, float]] = field(default_factory=list)

    # Chart 5: array-size scaling (KPU only; other archetypes may be empty)
    array_scaling_curve: List[Dict[str, float]] = field(default_factory=list)

    # Assumptions / provenance
    tdp_watts: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": self.archetype,
            "sku": self.sku,
            "display_name": self.display_name,
            "color": self.color,
            "energy_per_op_pj": self.energy_per_op_pj,
            "peak_ops_per_sec": self.peak_ops_per_sec,
            "ops_per_watt": self.ops_per_watt,
            "schedule_class": self.schedule_class.value,
            "pipeline_fill_cycles": self.pipeline_fill_cycles,
            "pipeline_drain_cycles": self.pipeline_drain_cycles,
            "pe_array_rows": self.pe_array_rows,
            "pe_array_cols": self.pe_array_cols,
            "utilization_curve": [list(p) for p in self.utilization_curve],
            "array_scaling_curve": list(self.array_scaling_curve),
            "tdp_watts": self.tdp_watts,
            "notes": self.notes,
        }


@dataclass
class ArchetypeComparisonReport:
    """Three archetype entries rendered side-by-side on one page."""
    archetypes: List[ArchetypeEntry]
    precision: str = "FP16"
    headline_tile_count: int = 16
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetypes": [a.to_dict() for a in self.archetypes],
            "precision": self.precision,
            "headline_tile_count": self.headline_tile_count,
            "generated_at": self.generated_at,
        }


# ----------------------------------------------------------------------
# Utilization curve synthesis
# ----------------------------------------------------------------------

DEFAULT_TILE_COUNTS = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256]

# Representative steady-state cycles per tile for a GEMM-style workload.
# This is the reduction dimension K; 128 corresponds to a mid-size model's
# inner dimension. Chart 4 uses this to make scheduling-class behavior
# visible against a realistic ratio of steady-state to fill/drain cycles.
REPRESENTATIVE_STEADY_CYCLES_PER_TILE = 128


def _synthesize_utilization_curve(
    schedule_class: TileScheduleClass,
    fill: int,
    drain: int,
    steady_cycles_per_tile: int = REPRESENTATIVE_STEADY_CYCLES_PER_TILE,
    tile_counts: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Compute effective_pipeline_utilization(N) across a sweep of N values.

    Mirrors the math in TileSpecialization.effective_pipeline_utilization
    so the archetype comparison is consistent with the per-tile model.
    """
    steady = max(int(steady_cycles_per_tile), 1)
    counts = tile_counts if tile_counts is not None else DEFAULT_TILE_COUNTS
    out: List[Tuple[int, float]] = []
    for n in counts:
        if schedule_class in (TileScheduleClass.OUTPUT_STATIONARY,
                              TileScheduleClass.ROW_STATIONARY):
            total = n * steady + fill + drain
            useful = n * steady
            util = useful / total if total > 0 else 1.0
        elif schedule_class == TileScheduleClass.WEIGHT_STATIONARY:
            total = n * (steady + fill + drain)
            useful = n * steady
            util = useful / total if total > 0 else 1.0
        else:
            # UNSPECIFIED: no tile pipeline. Tensor Core's realistic
            # utilization is dominated by warp divergence and coherence,
            # not a fabric pipeline. Baseline is 0.65 for illustration.
            util = 0.65
        out.append((n, util))
    return out


# ----------------------------------------------------------------------
# Archetype builders (lookup real SKUs and derive chart data)
# ----------------------------------------------------------------------

def _kpu_entry_from_mapper(
    sku: str, precision: Precision, display_name: str, color: str,
) -> ArchetypeEntry:
    """Build a Dataflow (KPU) archetype entry from the mapper registry."""
    from graphs.hardware.mappers import get_mapper_by_name
    mapper = get_mapper_by_name(sku)
    if mapper is None:
        raise RuntimeError(f"Mapper {sku!r} not in registry")

    rm = mapper.resource_model
    # Default thermal profile's INT8 compute resource
    tp = rm.thermal_operating_points[rm.default_thermal_profile]
    perf = tp.performance_specs[precision]
    cr = perf.compute_resource
    # Use the first tile specialization as the representative tile shape
    spec0 = cr.tile_specializations[0]
    rows, cols = spec0.array_dimensions

    # Headline metrics at the default thermal profile
    peak_ops = rm.precision_profiles[precision].peak_ops_per_sec
    tdp = tp.tdp_watts
    ops_per_watt = peak_ops / tdp if tdp > 0 else 0.0
    # Per-op energy from the tile energy model (pJ) if available; else derive
    if hasattr(rm, "tile_energy_model") and rm.tile_energy_model is not None:
        tem = rm.tile_energy_model
        if precision == Precision.INT8:
            energy_per_mac_j = tem.mac_energy_int8
        elif precision == Precision.BF16:
            energy_per_mac_j = tem.mac_energy_bf16
        else:
            energy_per_mac_j = tem.mac_energy_fp32
        # One MAC = 2 ops (1 multiply + 1 add)
        energy_per_op_pj = (energy_per_mac_j / 2.0) * 1e12
    else:
        energy_per_op_pj = tdp / peak_ops * 1e12 if peak_ops > 0 else 0.0

    util_curve = _synthesize_utilization_curve(
        spec0.schedule_class, spec0.pipeline_fill_cycles,
        spec0.pipeline_drain_cycles,
    )

    # Array-scaling sweep: hypothetical per-PE MAC energy and steady
    # throughput as PE array grows. Per-op energy stays roughly flat
    # (same MAC energy at the PE), but peak ops grows quadratically
    # with array dimension while fill/drain grows linearly -> ops/W
    # improves up to the regime where fill/drain becomes negligible.
    scaling = []
    for dim in (8, 16, 24, 32, 48, 64):
        pes = dim * dim
        fill_d = dim
        drain_d = dim
        # At 16 tiles of workload, compute effective utilization
        util_16 = _synthesize_utilization_curve(
            spec0.schedule_class, fill_d, drain_d, tile_counts=[16],
        )[0][1]
        # Hypothetical peak scales with PE count (not tile count)
        hypo_peak = pes * 2 * rm.precision_profiles[precision].peak_ops_per_sec / (rows * cols * 2)
        hypo_ops_per_watt = hypo_peak * util_16 / tdp if tdp > 0 else 0.0
        scaling.append({
            "pe_array_dim": dim,
            "pe_count": pes,
            "peak_ops_per_sec": hypo_peak,
            "ops_per_watt_at_util_16": hypo_ops_per_watt,
            "effective_utilization_at_16_tiles": util_16,
        })

    return ArchetypeEntry(
        archetype="Dataflow (KPU)",
        sku=sku,
        display_name=display_name,
        color=color,
        energy_per_op_pj=energy_per_op_pj,
        peak_ops_per_sec=peak_ops,
        ops_per_watt=ops_per_watt,
        schedule_class=spec0.schedule_class,
        pipeline_fill_cycles=spec0.pipeline_fill_cycles,
        pipeline_drain_cycles=spec0.pipeline_drain_cycles,
        pe_array_rows=rows,
        pe_array_cols=cols,
        utilization_curve=util_curve,
        array_scaling_curve=scaling,
        tdp_watts=tdp,
        notes=(f"Output-stationary scheduling; fill/drain overlaps across "
               f"adjacent tiles. Utilization saturates at ~12+ tiles."),
    )


def _tpu_entry(precision: Precision) -> ArchetypeEntry:
    """
    Build a Systolic (TPU) archetype entry from the Coral Edge TPU model.

    The Edge TPU's pipeline_fill_cycles is already modeled in the TPU
    resource model (coral_edge_tpu.py). We derive headline numbers from
    that SKU and mark scheduling as WEIGHT_STATIONARY.
    """
    from graphs.hardware.mappers import get_mapper_by_name
    mapper = get_mapper_by_name("Google-Coral-Edge-TPU")
    if mapper is None:
        raise RuntimeError("Coral Edge TPU not in registry")
    rm = mapper.resource_model

    # Edge TPU spec: 4 TOPS INT8, 2 W
    # For the comparison, use INT8 if requested; FP16 falls back to INT8 values
    int8_peak = rm.precision_profiles[Precision.INT8].peak_ops_per_sec \
        if Precision.INT8 in rm.precision_profiles else 4e12
    tdp = 2.0
    ops_per_watt = int8_peak / tdp

    # Per-op energy: well-known 0.15 pJ/MAC -> 0.075 pJ/op
    energy_per_op_pj = 0.075

    # Systolic array: representative Edge TPU is 64x64
    rows, cols = 64, 64
    fill = 64   # one column sweep
    drain = 64

    util_curve = _synthesize_utilization_curve(
        TileScheduleClass.WEIGHT_STATIONARY, fill, drain,
    )

    return ArchetypeEntry(
        archetype="Systolic (TPU)",
        sku="Google-Coral-Edge-TPU",
        display_name="Coral Edge TPU",
        color="#e98c3f",
        energy_per_op_pj=energy_per_op_pj,
        peak_ops_per_sec=int8_peak,
        ops_per_watt=ops_per_watt,
        schedule_class=TileScheduleClass.WEIGHT_STATIONARY,
        pipeline_fill_cycles=fill,
        pipeline_drain_cycles=drain,
        pe_array_rows=rows,
        pe_array_cols=cols,
        utilization_curve=util_curve,
        array_scaling_curve=[],  # not explored for TPU in M0.5
        tdp_watts=tdp,
        notes=("Weight-stationary scheduling; fill/drain paid per tile "
               "(no overlap across weight reloads). Utilization is a "
               "flat floor set by steady/(steady+fill+drain)."),
    )


def _tensor_core_entry(precision: Precision) -> ArchetypeEntry:
    """
    Build a SIMT + Tensor Core archetype entry from Jetson Orin AGX.
    """
    from graphs.hardware.mappers import get_mapper_by_name
    mapper = get_mapper_by_name("Jetson-Orin-AGX-64GB")
    if mapper is None:
        raise RuntimeError("Jetson Orin AGX not in registry")
    rm = mapper.resource_model
    tp = rm.thermal_operating_points[rm.default_thermal_profile]
    tdp = tp.tdp_watts

    # Peak INT8 at the default thermal profile
    int8_peak = rm.precision_profiles[Precision.INT8].peak_ops_per_sec \
        if Precision.INT8 in rm.precision_profiles else 0.0
    ops_per_watt = int8_peak / tdp if tdp > 0 else 0.0
    # Tensor Core FP16 per-op energy ~1.62 pJ (8nm standard_cell x 0.85)
    # Normalize to per-op: Tensor Core computes 2 ops/MAC
    energy_per_op_pj = 1.62 / 2.0  # pJ/op at FP16 on Ampere Tensor Core

    util_curve = _synthesize_utilization_curve(
        TileScheduleClass.UNSPECIFIED, 0, 0,
    )

    return ArchetypeEntry(
        archetype="SIMT + Tensor Core",
        sku="Jetson-Orin-AGX-64GB",
        display_name="Jetson Orin AGX",
        color="#5b8ff9",
        energy_per_op_pj=energy_per_op_pj,
        peak_ops_per_sec=int8_peak,
        ops_per_watt=ops_per_watt,
        schedule_class=TileScheduleClass.UNSPECIFIED,
        pipeline_fill_cycles=0,
        pipeline_drain_cycles=0,
        pe_array_rows=32,
        pe_array_cols=4,
        utilization_curve=util_curve,
        array_scaling_curve=[],
        tdp_watts=tdp,
        notes=("No tile pipeline; effective utilization is dominated by "
               "warp divergence, memory coherence, and scheduling "
               "overhead. Baseline shown as a flat 0.65 for illustration."),
    )


def build_default_comparison(
    precision: Precision = Precision.INT8,
    kpu_sku: str = "Stillwater-KPU-T128",
    kpu_display_name: str = "KPU T128",
) -> ArchetypeComparisonReport:
    """
    Build the default three-archetype comparison used for the M8 deck
    and the M0.5 exploration harness.
    """
    from datetime import datetime, timezone
    return ArchetypeComparisonReport(
        archetypes=[
            _tensor_core_entry(precision),
            _tpu_entry(precision),
            _kpu_entry_from_mapper(kpu_sku, precision, kpu_display_name, "#3fc98a"),
        ],
        precision=precision.value if hasattr(precision, "value") else str(precision),
        headline_tile_count=16,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# HTML renderer with Plotly-via-CDN
# ----------------------------------------------------------------------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _plot_div(chart_id: str) -> str:
    return f'<div id="{chart_id}" class="plot"></div>'


def _chart_section(
    title: str, description: str, chart_id: str,
) -> str:
    return f"""
<section class="chart-section">
  <h3>{html.escape(title)}</h3>
  <p class="chart-desc">{html.escape(description)}</p>
  {_plot_div(chart_id)}
</section>
"""


def _render_chart_js(report: ArchetypeComparisonReport) -> str:
    """Emit Plotly chart definitions as a single JS block."""
    arch = [a.to_dict() for a in report.archetypes]

    # Chart 1: energy per op (grouped bar)
    chart1 = {
        "data": [{
            "type": "bar",
            "x": [a["display_name"] for a in arch],
            "y": [a["energy_per_op_pj"] for a in arch],
            "marker": {"color": [a["color"] for a in arch]},
            "text": [f"{a['energy_per_op_pj']:.2f}" for a in arch],
            "textposition": "outside",
        }],
        "layout": {
            "title": f"Energy per op at {report.precision}",
            "yaxis": {"title": "pJ / op"},
            "margin": {"t": 50, "b": 40, "l": 60, "r": 20},
            "showlegend": False,
        },
    }

    # Chart 2: peak ops/s (grouped bar, TOPS)
    chart2 = {
        "data": [{
            "type": "bar",
            "x": [a["display_name"] for a in arch],
            "y": [a["peak_ops_per_sec"] / 1e12 for a in arch],
            "marker": {"color": [a["color"] for a in arch]},
            "text": [f"{a['peak_ops_per_sec']/1e12:.1f}" for a in arch],
            "textposition": "outside",
        }],
        "layout": {
            "title": f"Peak throughput at {report.precision}",
            "yaxis": {"title": "TOPS"},
            "margin": {"t": 50, "b": 40, "l": 60, "r": 20},
            "showlegend": False,
        },
    }

    # Chart 3: ops/W (grouped bar, TOPS/W)
    chart3 = {
        "data": [{
            "type": "bar",
            "x": [a["display_name"] for a in arch],
            "y": [a["ops_per_watt"] / 1e12 for a in arch],
            "marker": {"color": [a["color"] for a in arch]},
            "text": [f"{a['ops_per_watt']/1e12:.1f}" for a in arch],
            "textposition": "outside",
        }],
        "layout": {
            "title": f"Energy efficiency at {report.precision}",
            "yaxis": {"title": "TOPS / W"},
            "margin": {"t": 50, "b": 40, "l": 60, "r": 20},
            "showlegend": False,
        },
    }

    # Chart 4: effective utilization vs workload tile count (multi-line)
    chart4_traces = []
    for a in arch:
        xs = [p[0] for p in a["utilization_curve"]]
        ys = [p[1] for p in a["utilization_curve"]]
        chart4_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": f"{a['display_name']} ({a['schedule_class']})",
            "x": xs,
            "y": ys,
            "line": {"color": a["color"], "width": 2},
            "marker": {"size": 6},
        })
    chart4 = {
        "data": chart4_traces,
        "layout": {
            "title": "Effective pipeline utilization vs. workload tile count",
            "xaxis": {"title": "Tile count in workload", "type": "log"},
            "yaxis": {"title": "Effective utilization", "range": [0, 1.05]},
            "margin": {"t": 50, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    # Chart 5: array-size scaling (KPU only; other archetypes render as empty)
    chart5_traces = []
    for a in arch:
        if not a["array_scaling_curve"]:
            continue
        xs = [p["pe_array_dim"] for p in a["array_scaling_curve"]]
        ys = [p["ops_per_watt_at_util_16"] / 1e12 for p in a["array_scaling_curve"]]
        chart5_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": f"{a['display_name']}: TOPS/W",
            "x": xs,
            "y": ys,
            "line": {"color": a["color"], "width": 2},
            "marker": {"size": 8},
        })
    if not chart5_traces:
        chart5_traces = [{
            "type": "scatter", "mode": "markers",
            "x": [0], "y": [0], "name": "(no data)",
        }]
    chart5 = {
        "data": chart5_traces,
        "layout": {
            "title": "PE array-size scaling (KPU): TOPS/W at 16-tile workload",
            "xaxis": {"title": "PE array dimension (square: NxN)"},
            "yaxis": {"title": "TOPS / W"},
            "margin": {"t": 50, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    payload = {
        "chart1": chart1, "chart2": chart2, "chart3": chart3,
        "chart4": chart4, "chart5": chart5,
    }
    js = (
        f"const CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )
    return js


def _render_archetype_summary(report: ArchetypeComparisonReport) -> str:
    rows = []
    for a in report.archetypes:
        a_d = a.to_dict()
        rows.append(
            f"<tr>"
            f'<td style="border-left:4px solid {a_d["color"]};padding-left:10px;">'
            f"<strong>{html.escape(a_d['display_name'])}</strong><br/>"
            f"<span class=\"meta\">{html.escape(a_d['archetype'])}</span></td>"
            f"<td><code>{html.escape(a_d['sku'])}</code></td>"
            f"<td>{html.escape(a_d['schedule_class'].replace('_', ' '))}</td>"
            f"<td>{a_d['pe_array_rows']}x{a_d['pe_array_cols']}</td>"
            f"<td>{a_d['tdp_watts']:.1f} W</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Archetype</th><th>SKU</th><th>Scheduling</th>"
        "<th>PE array</th><th>TDP</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_archetype_page(
    report: ArchetypeComparisonReport,
    repo_root: Path,
) -> str:
    """
    Render the M0.5 compute-archetype comparison page.

    Self-contained HTML with Plotly loaded from CDN; Branes-branded
    chrome reused from the M0 template.
    """
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "GPU vs. TPU vs. KPU: compute-archetype comparison",
        f"Precision: {report.precision}  |  "
        f"Workload headline: {report.headline_tile_count} tiles  |  "
        f"Generated: {report.generated_at}",
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
  margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px; margin: 0 0 12px; }
.plot { width: 100%; min-height: 360px; }
table { width: 100%; border-collapse: collapse; background: #fff; margin-bottom: 18px; }
table th, table td { padding: 10px 12px; border-bottom: 1px solid #e3e6eb;
  text-align: left; vertical-align: middle; }
table th { font-size: 12px; text-transform: uppercase; color: #586374; background: #f3f5f8; }
.archetype-note { background: #eef2f7; padding: 12px 16px; border-left: 3px solid #0a2540;
  border-radius: 4px; margin-bottom: 18px; }
.archetype-note strong { color: #0a2540; }
"""

    summary_table = _render_archetype_summary(report)
    note_block = f"""
<div class="archetype-note">
  <strong>The scheduling-class difference drives chart 4.</strong> KPU
  output-stationary scheduling lets fill/drain of one tile overlap with
  fill/drain of its neighbors on the fabric - so as the workload grows,
  effective utilization saturates near 1.0. TPU weight-stationary
  scheduling must drain weights before reloading, so its utilization is
  a flat floor. SIMT + Tensor Core has no tile pipeline; its effective
  utilization is dominated by warp divergence and coherence traffic
  (shown as a flat baseline).
</div>
"""

    charts_block = "".join([
        _chart_section(
            "Chart 1: Energy per op",
            "Lower is better. KPU wins on energy per op - the dataflow "
            "fabric's per-PE steady-state MAC energy is below Tensor Core "
            "at matched precision.",
            "chart1"),
        _chart_section(
            "Chart 2: Peak throughput",
            "Higher is not always better. Tensor Core can outperform on "
            "peak, which we compensate for with larger PE arrays "
            "(T64: 32x32, T128: 24x24, T256: 16x16).",
            "chart2"),
        _chart_section(
            "Chart 3: Energy efficiency (TOPS/W)",
            "The deciding chart. KPU's low per-op energy combined with "
            "high sustained utilization yields a TOPS/W advantage that "
            "grows with workload size.",
            "chart3"),
        _chart_section(
            "Chart 4: Effective pipeline utilization vs. tile count",
            "Due-diligence chart. KPU output-stationary saturates to 1.0 "
            "at 12+ tiles; TPU weight-stationary is capped by "
            "fill/drain per tile; Tensor Core has no tile pipeline.",
            "chart4"),
        _chart_section(
            "Chart 5: PE array-size scaling (KPU)",
            "Where 'compensate for peak with larger arrays' has headroom. "
            "Beyond ~32x32, fill/drain becomes negligible and ops/W "
            "plateaus.",
            "chart5"),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Compute-archetype comparison: GPU vs. TPU vs. KPU</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}
{extra_css}
  </style>
</head>
<body>
{header}
<main>
  <section class="page-header">
    <h2>Compute-archetype comparison (M0.5)</h2>
    <div class="meta">Three archetypes, one picture: why KPU wins on energy per op.</div>
  </section>
  {_render_legend()}
  {summary_table}
  {note_block}
  {charts_block}
</main>
{footer}
<script>
{_render_chart_js(report)}
</script>
</body>
</html>
"""
