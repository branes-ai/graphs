"""
Competitive-trajectory analysis: how long would it take NVIDIA's
Jetson product line to reach the KPU T128 TOPS/W target?

Overlays:
  - Historical Jetson dense INT8 TOPS/W at each launch year
  - Exponential extrapolations from the most recent shipping product
    (Orin AGX, 2022) at 20% / 25% / 30% annual growth rates
  - Horizontal target lines at the KPU T128 peak and sustained
    TOPS/W (from the generalized_architecture / check_tdp_feasibility
    models)
  - Annotations showing the crossing year (parity) for each rate

Data sources:
  - NVIDIA Jetson datasheets and Anandtech/Tom's Hardware teardowns
    for each generation (values below).
  - KPU targets from this repo's M0.5 parametrization (12.3 TOPS/W
    peak, 11.04 TOPS/W sustained at 12 W TDP, 16nm optimized domain
    flow - verified by cli/check_tdp_feasibility.py).

Modelling choice: dense INT8 TOPS/W, not "AI TOPS" with sparsity
multipliers. Sparsity-inflated marketing figures compare unfairly
against a dense-INT8 baseline and are excluded from the trend fit.
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphs.reporting.microarch_html_template import (
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
)


# ----------------------------------------------------------------------
# Historical Jetson data
# ----------------------------------------------------------------------

@dataclass
class HistoricalPoint:
    year: float                     # launch year (fractional allowed)
    tops_per_watt: float            # dense INT8 peak
    product: str
    tdp_w: float
    peak_tops_int8: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "tops_per_watt": self.tops_per_watt,
            "product": self.product,
            "tdp_w": self.tdp_w,
            "peak_tops_int8": self.peak_tops_int8,
            "notes": self.notes,
        }


# Dense INT8 peak, sourced from NVIDIA datasheets + independent
# confirmations. Pre-Xavier Jetsons had no INT8 Tensor Cores, so the
# trend fit starts at Xavier (2019).
JETSON_HISTORY: List[HistoricalPoint] = [
    HistoricalPoint(
        year=2019.0, tops_per_watt=1.07,
        product="Jetson Xavier AGX",
        tdp_w=30.0, peak_tops_int8=32.0,
        notes="First Jetson with INT8 Tensor Cores (Volta GV10B).",
    ),
    HistoricalPoint(
        year=2020.0, tops_per_watt=1.40,
        product="Jetson Xavier NX",
        tdp_w=15.0, peak_tops_int8=21.0,
        notes="Cut-down Xavier for embedded.",
    ),
    HistoricalPoint(
        year=2022.0, tops_per_watt=2.67,
        product="Jetson Orin Nano",
        tdp_w=15.0, peak_tops_int8=40.0,
        notes="Ampere SMs + Tensor Cores, LPDDR5.",
    ),
    HistoricalPoint(
        year=2022.25, tops_per_watt=4.00,
        product="Jetson Orin NX",
        tdp_w=25.0, peak_tops_int8=100.0,
        notes="Mid-range Ampere.",
    ),
    HistoricalPoint(
        year=2022.5, tops_per_watt=2.28,
        product="Jetson Orin AGX (60W)",
        tdp_w=60.0, peak_tops_int8=137.0,
        notes=("Full Orin at 60W. Published 'AI TOPS' uses FP16 "
               "sparse = 275; dense INT8 is 137."),
    ),
]


# ----------------------------------------------------------------------
# Report schema
# ----------------------------------------------------------------------

@dataclass
class TrajectoryReport:
    history: List[HistoricalPoint]
    kpu_peak_tops_per_watt: float         # KPU T128 peak
    kpu_sustained_tops_per_watt: float    # KPU T128 sustained
    kpu_label: str
    growth_rates: List[float]             # e.g., [0.15, 0.20, 0.25, 0.30]
    extrapolate_from_year: int
    extrapolate_to_year: int
    generated_at: str = ""

    @property
    def anchor_point(self) -> HistoricalPoint:
        """Most recent shipping generation used as the extrapolation anchor."""
        # Pick the Orin AGX (60W) entry - it's the published flagship
        for p in self.history:
            if "Orin AGX" in p.product:
                return p
        return self.history[-1]

    def demonstrated_cagr(self) -> float:
        """Geometric mean growth from first Xavier to Orin AGX."""
        first = self.history[0]
        last = self.anchor_point
        years = last.year - first.year
        if years <= 0:
            return 0.0
        return (last.tops_per_watt / first.tops_per_watt) ** (1.0 / years) - 1.0

    def years_to_target(self, target_tops_per_watt: float,
                         rate: float) -> Optional[float]:
        """Years from anchor to reach a target TOPS/W at the given annual rate."""
        anchor = self.anchor_point.tops_per_watt
        if target_tops_per_watt <= anchor:
            return 0.0
        if rate <= 0:
            return None
        return math.log(target_tops_per_watt / anchor) / math.log(1.0 + rate)


def build_default_report(
    kpu_peak_tops_per_watt: float = 12.3,
    kpu_sustained_tops_per_watt: float = 11.04,
    kpu_label: str = "KPU T128 (Stillwater)",
) -> TrajectoryReport:
    from datetime import datetime, timezone
    return TrajectoryReport(
        history=JETSON_HISTORY,
        kpu_peak_tops_per_watt=kpu_peak_tops_per_watt,
        kpu_sustained_tops_per_watt=kpu_sustained_tops_per_watt,
        kpu_label=kpu_label,
        growth_rates=[0.15, 0.20, 0.25, 0.30],
        extrapolate_from_year=2022,
        extrapolate_to_year=2040,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _render_chart_js(report: TrajectoryReport) -> str:
    anchor = report.anchor_point
    start_year = anchor.year
    end_year = report.extrapolate_to_year

    traces = []

    # 1. Historical data points
    traces.append({
        "type": "scatter",
        "mode": "markers+text",
        "name": "Jetson (shipping)",
        "x": [p.year for p in report.history],
        "y": [p.tops_per_watt for p in report.history],
        "marker": {"size": 12, "color": "#5b8ff9",
                   "line": {"color": "#0a2540", "width": 1.5}},
        "text": [p.product.replace("Jetson ", "") for p in report.history],
        "textposition": "top center",
        "textfont": {"size": 10, "color": "#3a4452"},
        "hovertemplate": (
            "<b>%{text}</b><br>"
            "Year: %{x}<br>"
            "TOPS/W: %{y:.2f}<extra></extra>"
        ),
    })

    # 2. Extrapolation lines per growth rate
    years = [start_year + 0.5 * i for i in range(int((end_year - start_year) * 2) + 1)]
    rate_palette = ["#d4860b", "#e98c3f", "#7f3b8d", "#a04bdd"]
    for i, rate in enumerate(report.growth_rates):
        ys = [anchor.tops_per_watt * (1.0 + rate) ** (y - start_year)
              for y in years]
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": f"{rate*100:.0f}%/yr extrapolation",
            "x": years,
            "y": ys,
            "line": {"color": rate_palette[i % len(rate_palette)],
                     "width": 2, "dash": "dot"},
            "hovertemplate": (
                f"<b>{rate*100:.0f}%/yr</b><br>"
                "Year: %{x:.1f}<br>"
                "TOPS/W: %{y:.2f}<extra></extra>"
            ),
        })

    # 3. KPU target lines (horizontal)
    target_years = [report.history[0].year, end_year]
    traces.append({
        "type": "scatter",
        "mode": "lines",
        "name": f"{report.kpu_label} peak ({report.kpu_peak_tops_per_watt:.1f} TOPS/W)",
        "x": target_years,
        "y": [report.kpu_peak_tops_per_watt, report.kpu_peak_tops_per_watt],
        "line": {"color": "#3fc98a", "width": 3},
    })
    traces.append({
        "type": "scatter",
        "mode": "lines",
        "name": f"{report.kpu_label} sustained ({report.kpu_sustained_tops_per_watt:.1f} TOPS/W)",
        "x": target_years,
        "y": [report.kpu_sustained_tops_per_watt, report.kpu_sustained_tops_per_watt],
        "line": {"color": "#3fc98a", "width": 2, "dash": "dash"},
    })

    # Annotations: crossing years for each rate against the sustained target
    annotations = []
    for i, rate in enumerate(report.growth_rates):
        y_to_target = report.years_to_target(
            report.kpu_sustained_tops_per_watt, rate
        )
        if y_to_target is not None:
            cross_year = start_year + y_to_target
            if cross_year <= end_year:
                annotations.append({
                    "x": cross_year,
                    "y": report.kpu_sustained_tops_per_watt,
                    "text": f"{rate*100:.0f}%: {cross_year:.0f}",
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": 0, "ay": 30 + i * 20,
                    "font": {"size": 10,
                             "color": rate_palette[i % len(rate_palette)]},
                })

    chart = {
        "data": traces,
        "layout": {
            "title": ("Jetson dense INT8 TOPS/W trajectory vs. "
                      f"{report.kpu_label} target"),
            "xaxis": {
                "title": "Year",
                "range": [report.history[0].year - 0.5, end_year + 0.5],
            },
            "yaxis": {
                "title": "Dense INT8 TOPS / W (log scale)",
                "type": "log",
            },
            "margin": {"t": 60, "b": 50, "l": 60, "r": 20},
            "legend": {"orientation": "h", "y": -0.15},
            "annotations": annotations,
        },
    }

    return (
        f"const CHARTS = {{\"chart_trajectory\": {json.dumps(chart)}}};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_parity_table(report: TrajectoryReport) -> str:
    anchor = report.anchor_point
    rows = []
    for rate in report.growth_rates:
        peak_years = report.years_to_target(report.kpu_peak_tops_per_watt, rate)
        sust_years = report.years_to_target(report.kpu_sustained_tops_per_watt, rate)
        peak_year_str = (f"{anchor.year + peak_years:.0f}"
                         if peak_years is not None else "n/a")
        sust_year_str = (f"{anchor.year + sust_years:.0f}"
                         if sust_years is not None else "n/a")
        peak_delta = (f"+{peak_years:.1f}"
                      if peak_years is not None else "n/a")
        sust_delta = (f"+{sust_years:.1f}"
                      if sust_years is not None else "n/a")
        rows.append(
            f'<tr>'
            f'<td><strong>{rate*100:.0f}%/yr</strong></td>'
            f'<td class="num">{peak_delta} yr</td>'
            f'<td class="num">{peak_year_str}</td>'
            f'<td class="num">{sust_delta} yr</td>'
            f'<td class="num">{sust_year_str}</td>'
            f'</tr>'
        )
    return (
        '<table class="trajectory">'
        '<thead><tr><th>Growth rate</th>'
        f'<th>Years to peak ({report.kpu_peak_tops_per_watt:.1f} TOPS/W)</th>'
        f'<th>Parity year (peak)</th>'
        f'<th>Years to sustained ({report.kpu_sustained_tops_per_watt:.1f} TOPS/W)</th>'
        f'<th>Parity year (sustained)</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_history_table(report: TrajectoryReport) -> str:
    rows = []
    for p in report.history:
        rows.append(
            f'<tr>'
            f'<td>{p.year:.2f}</td>'
            f'<td><strong>{html.escape(p.product)}</strong></td>'
            f'<td class="num">{p.peak_tops_int8:.0f}</td>'
            f'<td class="num">{p.tdp_w:.0f}</td>'
            f'<td class="num"><strong>{p.tops_per_watt:.2f}</strong></td>'
            f'<td class="src">{html.escape(p.notes)}</td>'
            f'</tr>'
        )
    return (
        '<table class="trajectory">'
        '<thead><tr><th>Year</th><th>Product</th>'
        '<th>Peak INT8 TOPS (dense)</th><th>TDP (W)</th>'
        '<th>TOPS/W</th><th>Notes</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_trajectory_page(
    report: TrajectoryReport,
    repo_root: Path,
) -> str:
    assets = _load_logo(repo_root)
    anchor = report.anchor_point
    demonstrated = report.demonstrated_cagr()

    header = _render_brand_header(
        assets,
        "Competitive trajectory: Jetson vs. KPU target",
        (f"Dense INT8 TOPS/W | anchor: {anchor.product} ({anchor.year:.1f}, "
         f"{anchor.tops_per_watt:.2f} TOPS/W) | "
         f"demonstrated Xavier->Orin CAGR: {demonstrated*100:.0f}%/yr | "
         f"generated {report.generated_at}"),
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.trajectory { width: 100%; border-collapse: collapse; background: #fff;
                   margin-bottom: 18px; }
table.trajectory th, table.trajectory td { padding: 8px 10px;
                                            border-bottom: 1px solid #e3e6eb;
                                            vertical-align: top; font-size: 13px; }
table.trajectory th { font-size: 12px; text-transform: uppercase;
                      color: #586374; background: #f3f5f8; text-align: left; }
table.trajectory td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.trajectory td.src { color: #586374; font-size: 12px; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin-bottom: 18px; }
section.method-note.target { border-left-color: #3fc98a; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
  margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px; margin: 0 0 12px; }
.plot { width: 100%; min-height: 460px; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    method_note = f"""
<section class="method-note">
  <strong>Methodology.</strong> Historical data points are each shipping
  Jetson product's dense INT8 peak TOPS / rated TDP. Pre-Xavier
  generations had no INT8 Tensor Cores and are excluded. Published
  "AI TOPS" numbers using FP16 sparse or FP4 multipliers are NOT used -
  we compare dense INT8 peak apples-to-apples. The extrapolations
  anchor on the Orin AGX 60W operating point (2.28 TOPS/W, 2022) which
  represents the flagship dense INT8 ceiling at that generation.
  Demonstrated Xavier-to-Orin CAGR is <strong>{demonstrated*100:.1f}%/yr</strong>.
</section>
"""

    target_note = f"""
<section class="method-note target">
  <strong>Targets.</strong> KPU T128 peak ({report.kpu_peak_tops_per_watt:.2f}
  TOPS/W) and sustained ({report.kpu_sustained_tops_per_watt:.2f} TOPS/W)
  values are the M0.5 parametrization at 16 nm domain-flow silicon,
  verified by <code>cli/check_tdp_feasibility.py</code>. Peak = 12.3
  TOPS/W from 12 W TDP / 147 TOPS peak; sustained = 11.04 TOPS/W with
  0.90 realistic utilization (output-stationary schedule saturates).
  These are THEORETICAL model numbers until silicon measurement
  graduates them to CALIBRATED.
</section>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Competitive trajectory: Jetson vs. KPU</title>
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
    <h2>How long to close the gap?</h2>
    <div class="meta">Jetson TOPS/W trajectory with extrapolations to
      KPU T128 target, at a range of growth rates.</div>
  </section>
  {method_note}
  {target_note}
  <section class="chart-section">
    <h3>TOPS/W trajectory (log scale)</h3>
    <p class="chart-desc">Blue markers: shipping Jetson products (dense
      INT8 peak). Dashed lines: extrapolations from Orin AGX (2022) at
      15/20/25/30 percent annual growth. Solid green: KPU T128 peak
      target; dashed green: KPU T128 sustained target. Annotations show
      the year each extrapolation crosses the sustained target.</p>
    <div id="chart_trajectory" class="plot"></div>
  </section>
  <h3>Years to parity</h3>
  {_render_parity_table(report)}
  <h3>Jetson historical data (dense INT8)</h3>
  {_render_history_table(report)}
  <section class="method-note">
    <strong>Load-bearing assumptions.</strong>
    (1) NVIDIA maintains a Tensor Core-based SIMT architecture (CUDA
    compatibility moat). An architectural pivot to dataflow/systolic
    could change these numbers.
    (2) Process scaling continues at historical pace; post-3 nm cost
    per transistor is flat.
    (3) Comparison uses dense INT8, not sparsity-inflated FP4
    marketing figures. Switching to FP4-sparse accounting would close
    the gap by 3-4x nominally but would not reflect actual per-op
    energy improvement.
    (4) KPU utilization = 0.90 holds. If real-workload utilization
    comes in at 0.70, the sustained target drops from 11.04 to 8.6
    TOPS/W and parity years shrink by ~1.5 years per growth rate.
  </section>
</main>
{footer}
<script>
{_render_chart_js(report)}
</script>
</body>
</html>
"""
