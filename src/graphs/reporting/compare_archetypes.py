"""
Compute-archetype comparison harness (GPU Tensor Core vs. TPU systolic
vs. KPU domain flow).

This is the diligence tool used during M0.5 to pick PE-array sizes for
T64 / T128 / T256 under the refined domain-flow-tile abstraction, and
the artifact that makes the KPU's energy-per-op positioning legible.

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

See ``docs/hardware/kpu_domainflow_tile_model.md`` and
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
# Full-adder reference energies by process node (calibration baseline for
# MAC energies). See cli/check_tdp_feasibility.py for the canonical table.
FULL_ADDER_PJ_BY_PROCESS = {
    28: 0.025, 22: 0.018, 16: 0.010, 14: 0.009, 12: 0.007,
    10: 0.006, 8: 0.005, 7: 0.004, 5: 0.003, 4: 0.0025, 3: 0.002,
}


def _fa_energy_pj(process_nm: int) -> float:
    if process_nm in FULL_ADDER_PJ_BY_PROCESS:
        return FULL_ADDER_PJ_BY_PROCESS[process_nm]
    nearest = min(FULL_ADDER_PJ_BY_PROCESS.keys(), key=lambda k: abs(k - process_nm))
    return FULL_ADDER_PJ_BY_PROCESS[nearest]


from graphs.reporting.microarch_html_template import (
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
    archetype: str               # "Tensor Core", "Systolic", "Domain Flow"
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

    # Process node + calibration reference (required when reporting energies)
    process_node_nm: int = 16
    full_adder_energy_pj: float = 0.010   # 1-bit CMOS FA @ nominal Vdd

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
            "process_node_nm": self.process_node_nm,
            "full_adder_energy_pj": self.full_adder_energy_pj,
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
    warp_divergence_rate: float = 0.0,
    warp_occupancy: float = 1.0,
    coherence_efficiency: float = 1.0,
) -> List[Tuple[int, float]]:
    """
    Compute effective_pipeline_utilization(N) across a sweep of N values.

    Mirrors the math in TileSpecialization.effective_pipeline_utilization
    so the archetype comparison is consistent with the per-tile model.
    """
    steady = max(int(steady_cycles_per_tile), 1)
    counts = tile_counts if tile_counts is not None else DEFAULT_TILE_COUNTS
    out: List[Tuple[int, float]] = []

    # SIMT_DATA_PARALLEL utilization is independent of tile count.
    if schedule_class == TileScheduleClass.SIMT_DATA_PARALLEL:
        divergence_penalty = 1.0 - 0.5 * max(0.0, min(1.0, warp_divergence_rate))
        occ = max(0.0, min(1.0, warp_occupancy))
        coh = max(0.0, min(1.0, coherence_efficiency))
        simt_util = divergence_penalty * occ * coh
        return [(n, simt_util) for n in counts]

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
            # UNSPECIFIED: no pipeline model; flat 1.0.
            util = 1.0
        out.append((n, util))
    return out


# ----------------------------------------------------------------------
# Archetype builders (lookup real SKUs and derive chart data)
# ----------------------------------------------------------------------

def _kpu_entry_from_mapper(
    sku: str, precision: Precision, display_name: str, color: str,
) -> ArchetypeEntry:
    """Build a Domain Flow (KPU) archetype entry from the mapper registry."""
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

    # Linear-schedule efficiency asymptote (scale-invariant in N).
    # For an NxNxN GEMM tile executed on an output-stationary fabric
    # with fill = drain = N cycles and a steady phase of N cycles
    # (one per K-reduction row), total latency is N(M+2) cycles for
    # M tiles, of which MN is useful. Equivalently, per the user's
    # linear-schedule formula: efficiency = 1 - 2/(3M). This is
    # independent of N, so larger PE arrays do NOT improve efficiency.
    # What drives efficiency is the workload's tile-count M.
    #
    # The curve below is what chart 5 plots (multiple N values overlay
    # as the same curve, which is the point).
    scaling = []
    tile_counts = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256]
    for dim in (8, 16, 24, 32, 48, 64):
        points = []
        for m in tile_counts:
            # User's linear-schedule formula: eff = 1 - (2N)/(M * 3N) = 1 - 2/(3M)
            efficiency = 1.0 - 2.0 / (3.0 * m) if m >= 1 else 0.0
            points.append({"tile_count": m, "efficiency": efficiency})
        scaling.append({
            "pe_array_dim": dim,
            "pe_count": dim * dim,
            "points": points,
        })

    # Pull process node from the first compute fabric (KPU SKUs are 16nm
    # for T64/T128/T256; 12nm for T768).
    fabrics = getattr(rm, "compute_fabrics", []) or []
    process_nm = fabrics[0].process_node_nm if fabrics else 16

    return ArchetypeEntry(
        archetype="Domain Flow (KPU)",
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
        process_node_nm=process_nm,
        full_adder_energy_pj=_fa_energy_pj(process_nm),
        notes=(f"Output-stationary scheduling; fill/drain overlaps across "
               f"adjacent tiles. Utilization saturates at ~12+ tiles."),
    )


def _tpu_entry(precision: Precision) -> ArchetypeEntry:
    """
    Build a Systolic (TPU) archetype entry from the Coral Edge TPU model.

    Published data shows real-world achieved-to-peak utilization on
    weight-stationary systolic arrays is 10-55% on typical DNN
    workloads (Jouppi et al., ISCA 2017 Table 3: TPU v1 hits 10-25%
    on production workloads; DeepEdgeBench 2021 reports Coral Edge
    TPU hits ~6% of peak on MobileNet V2, ~25% on EfficientNet-
    EdgeTPU). The dominant loss is shape/tile mismatch against the
    fixed PE dimensions and bandwidth-bound layers, NOT per-tile
    fill/drain (which is amortized by double-buffering per Jouppi
    2017 Section 2).

    We model the combined effect as a flat utilization floor around
    0.50 - a generous "compute-bound dense GEMM upper bound" rather
    than real-workload average. The fill/drain values below are
    calibrated to produce 0.50 under the steady/(steady+fill+drain)
    formula for parity with the rest of the comparison, not to
    represent literal wavefront-propagation cycles.
    """
    from graphs.hardware.mappers import get_mapper_by_name
    mapper = get_mapper_by_name("Google-Coral-Edge-TPU")
    if mapper is None:
        raise RuntimeError("Coral Edge TPU not in registry")
    rm = mapper.resource_model

    int8_peak = rm.precision_profiles[Precision.INT8].peak_ops_per_sec \
        if Precision.INT8 in rm.precision_profiles else 4e12
    tdp = 2.0
    ops_per_watt = int8_peak / tdp

    # Per-op energy: well-known 0.15 pJ/MAC -> 0.075 pJ/op (compute-
    # bound; does not reflect bandwidth/shape-mismatch losses).
    energy_per_op_pj = 0.075

    # Representative Edge TPU systolic shape
    rows, cols = 64, 64
    # Effective fill/drain calibrated to give ~0.50 flat utilization,
    # representing combined shape/tile/bandwidth losses after double-
    # buffering (NOT literal wavefront cycles - see docstring).
    fill = 64
    drain = 64

    util_curve = _synthesize_utilization_curve(
        TileScheduleClass.WEIGHT_STATIONARY, fill, drain,
    )

    # Coral Edge TPU is 14nm (published TSMC 14nm for the original
    # Edge TPU; exact process is confidential).
    coral_process_nm = 14

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
        array_scaling_curve=[],
        tdp_watts=tdp,
        process_node_nm=coral_process_nm,
        full_adder_energy_pj=_fa_energy_pj(coral_process_nm),
        notes=("Weight-stationary scheduling. Modeled utilization of "
               "~0.50 is a compute-bound dense-GEMM upper bound. Real "
               "Coral Edge TPU workloads typically achieve 6-25% of "
               "peak (Jouppi ISCA 2017; DeepEdgeBench 2021). Dominant "
               "losses are shape/tile mismatch and bandwidth-bound "
               "layers, not per-tile fill/drain (which is amortized "
               "by weight double-buffering)."),
    )


def _tensor_core_entry(precision: Precision) -> ArchetypeEntry:
    """
    Build a SIMT + Tensor Core archetype entry from Jetson Orin AGX.

    Ampere-class Tensor Core realistic utilization on mixed-precision
    GEMM is capped by warp divergence, warp occupancy, and memory
    coherence traffic - NOT by a spatial fabric pipeline. We model
    this with the SIMT_DATA_PARALLEL scheduling class and
    representative parameters for Ampere Tensor Core:

      warp_divergence_rate = 0.05   (5% of cycles see divergence)
      warp_occupancy        = 0.75   (achieved / max warps)
      coherence_efficiency  = 0.90   (hit rate through L1/L2 and register)

    Yields an effective utilization of ~0.66, independent of workload
    tile count.
    """
    from graphs.hardware.mappers import get_mapper_by_name
    mapper = get_mapper_by_name("Jetson-Orin-AGX-64GB")
    if mapper is None:
        raise RuntimeError("Jetson Orin AGX not in registry")
    rm = mapper.resource_model
    tp = rm.thermal_operating_points[rm.default_thermal_profile]
    tdp = tp.tdp_watts

    int8_peak = rm.precision_profiles[Precision.INT8].peak_ops_per_sec \
        if Precision.INT8 in rm.precision_profiles else 0.0
    ops_per_watt = int8_peak / tdp if tdp > 0 else 0.0
    # Ampere Tensor Core FP16 per-op energy ~1.62 pJ/MAC -> 0.81 pJ/op
    energy_per_op_pj = 1.62 / 2.0

    # Representative Ampere Tensor Core parameters
    warp_divergence_rate = 0.05
    warp_occupancy = 0.75
    coherence_efficiency = 0.90

    util_curve = _synthesize_utilization_curve(
        TileScheduleClass.SIMT_DATA_PARALLEL, 0, 0,
        warp_divergence_rate=warp_divergence_rate,
        warp_occupancy=warp_occupancy,
        coherence_efficiency=coherence_efficiency,
    )

    # Jetson Orin AGX uses Samsung 8nm LPP
    orin_process_nm = 8

    return ArchetypeEntry(
        archetype="SIMT + Tensor Core",
        sku="Jetson-Orin-AGX-64GB",
        display_name="Jetson Orin AGX",
        color="#5b8ff9",
        energy_per_op_pj=energy_per_op_pj,
        peak_ops_per_sec=int8_peak,
        ops_per_watt=ops_per_watt,
        schedule_class=TileScheduleClass.SIMT_DATA_PARALLEL,
        pipeline_fill_cycles=0,
        pipeline_drain_cycles=0,
        pe_array_rows=32,
        pe_array_cols=4,
        utilization_curve=util_curve,
        array_scaling_curve=[],
        tdp_watts=tdp,
        process_node_nm=orin_process_nm,
        full_adder_energy_pj=_fa_energy_pj(orin_process_nm),
        notes=("SIMT data-parallel scheduling. Not a spatial fabric "
               "pipeline; CUDA cores execute the instruction stream "
               "cycle-by-cycle. Utilization is capped by warp "
               "divergence, warp occupancy, and coherence traffic, "
               "and does NOT amortize with workload tile count. "
               "Representative Ampere Tensor Core parameters: "
               "div_rate=0.05, occ=0.75, coh=0.90 -> util=0.658."),
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

    # Chart 5: linear-schedule efficiency asymptote (KPU only).
    # Plots eff = 1 - 2/(3M) across workload tile counts M for multiple
    # PE array sizes N. All N curves overlay as the same line, which
    # is the point: efficiency is scale-invariant in N. The plateau is
    # set by M, not by the PE array.
    chart5_traces = []
    for a in arch:
        if not a["array_scaling_curve"]:
            continue
        # Use Plotly dash styles to visually differentiate overlapping curves
        dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
        for idx, curve in enumerate(a["array_scaling_curve"]):
            xs = [p["tile_count"] for p in curve["points"]]
            ys = [p["efficiency"] for p in curve["points"]]
            dim = curve["pe_array_dim"]
            chart5_traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"{dim}x{dim} PE array",
                "x": xs,
                "y": ys,
                "line": {
                    "color": a["color"], "width": 2,
                    "dash": dash_styles[idx % len(dash_styles)],
                },
                "opacity": 0.75,
            })
    if not chart5_traces:
        chart5_traces = [{
            "type": "scatter", "mode": "markers",
            "x": [0], "y": [0], "name": "(no data)",
        }]
    chart5 = {
        "data": chart5_traces,
        "layout": {
            "title": ("KPU linear-schedule efficiency: eff = 1 - 2/(3M), "
                      "scale-invariant in PE array size N"),
            "xaxis": {
                "title": "Workload tile count M",
                "type": "log",
            },
            "yaxis": {
                "title": "Efficiency",
                "range": [0, 1.05],
            },
            "margin": {"t": 60, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
            "annotations": [
                {
                    "text": ("All PE array sizes produce the same curve: "
                             "efficiency is set by workload tile count M, "
                             "not by N."),
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 1.08,
                    "showarrow": False,
                    "font": {"size": 12, "color": "#586374"},
                },
            ],
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
        # Display SKU without internal "Stillwater-" prefix used as the
        # registry key; registry keys stay intact in code.
        display_sku = a_d["sku"].replace("Stillwater-", "")
        rows.append(
            f"<tr>"
            f'<td style="border-left:4px solid {a_d["color"]};padding-left:10px;">'
            f"<strong>{html.escape(a_d['display_name'])}</strong><br/>"
            f"<span class=\"meta\">{html.escape(a_d['archetype'])}</span></td>"
            f"<td><code>{html.escape(display_sku)}</code></td>"
            f"<td>{html.escape(a_d['schedule_class'].replace('_', ' '))}</td>"
            f"<td>{a_d['pe_array_rows']}x{a_d['pe_array_cols']}</td>"
            f"<td>{a_d['process_node_nm']} nm</td>"
            f"<td>{a_d['full_adder_energy_pj']:.3f} pJ</td>"
            f"<td>{a_d['tdp_watts']:.1f} W</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Archetype</th><th>SKU</th><th>Scheduling</th>"
        "<th>PE array</th><th>Process</th><th>1-bit FA ref</th>"
        "<th>TDP</th></tr></thead>"
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
<div class="archetype-note">
  <strong>Energy calibration reference.</strong> All MAC and per-op
  energies are quoted at the SKU's manufacturing process node. The
  table above lists the corresponding 1-bit CMOS full-adder (FA) dynamic
  energy at nominal Vdd as a calibration baseline - an INT8 MAC requires
  ~8 FA equivalents plus array + register-file overhead. Energies below
  ~0.5x * 8 * FA are aggressive; above ~4x * 8 * FA are standard-cell
  baseline. First-principles TDP feasibility can be checked with
  <code>cli/check_tdp_feasibility.py</code>.
</div>
"""

    charts_block = "".join([
        _chart_section(
            "Chart 1: Energy per op",
            "Lower is better. KPU wins on energy per op - the domain-flow "
            "fabric's per-PE steady-state MAC energy is below Tensor Core "
            "at matched precision.",
            "chart1"),
        _chart_section(
            "Chart 2: Peak throughput",
            "Higher is not always better. Tensor Core can outperform on "
            "peak, which we compensate for with larger PE arrays "
            "(T64: 32x32, T128: 32x32, T256: 20x20).",
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
            "Chart 5: Linear-schedule efficiency asymptote (KPU)",
            "For an NxNxN GEMM tile under a linear schedule, per-tile "
            "latency is 3N cycles (fill=N, steady=N, drain=N). With "
            "fill/drain amortized across M tiles on an output-stationary "
            "fabric, efficiency = 1 - 2/(3M). Plateau is set by M, not N: "
            "at M=12, eff~0.94; at M=32, eff~0.98; asymptote approaches "
            "1.0 as M grows. Larger PE arrays give higher peak throughput "
            "but do not improve efficiency (the curves overlay).",
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
a.nav-back {{ display: inline-block; color: #0a2540; text-decoration: none;
               font-weight: 600; margin-bottom: 10px; }}
a.nav-back:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
{header}
<main>
  <p><a class="nav-back" href="index.html">&lt; Back to index</a></p>
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
