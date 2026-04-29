"""
Branes-branded HTML renderer for the micro-architectural validation report.

Produces self-contained HTML pages (embedded CSS, no external JS frameworks)
so the output is diffable, shareable by file, and works offline. Plotly is
loaded from CDN when charts are present; during M0 (no populated content)
the HTML contains only the page chrome and NOT-YET-POPULATED placeholders.

See ``docs/plans/microarch-model-delivery-plan.md`` M0 deliverables.
"""
from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from graphs.reporting.microarch_schema import (
    LayerPanel,
    MicroarchReport,
)


# Whitelist of status values whose CSS class is safe to emit unescaped.
# Anything outside this set is coerced to "unknown" to prevent HTML
# attribute injection through a crafted panel.status.
_SAFE_STATUS_CLASSES = {
    "calibrated",
    "interpolated",
    "theoretical",
    "not_populated",
    "unknown",
}


def _safe_status_class(status: str) -> str:
    """Return an HTML-safe class token for a panel status."""
    token = (status or "").strip().lower()
    if token in _SAFE_STATUS_CLASSES:
        return token
    return "unknown"


# Brand asset path (repo-relative).
BRAND_LOGO_RELATIVE = "docs/img/Branes_Logo.jpg"


@dataclass
class BrandAssets:
    """
    Resolved brand asset paths and encoded content for HTML embedding.
    """
    logo_data_uri: str = ""
    logo_alt: str = "Branes"


def _load_logo(repo_root: Path) -> BrandAssets:
    """Embed the Branes logo as a data URI so HTML is self-contained."""
    logo_path = repo_root / BRAND_LOGO_RELATIVE
    if not logo_path.exists():
        return BrandAssets(logo_data_uri="", logo_alt="Branes")
    content = logo_path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return BrandAssets(
        logo_data_uri=f"data:image/jpeg;base64,{encoded}",
        logo_alt="Branes",
    )


_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0;
  color: #1a1a1a;
  background: #f6f7f9;
  line-height: 1.5;
}
header.brand {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 32px;
  background: #ffffff;
  border-bottom: 2px solid #0a2540;
}
header.brand .logo { height: 44px; }
header.brand .title-block { text-align: right; }
header.brand .title-block h1 {
  margin: 0;
  font-size: 18px;
  color: #0a2540;
}
header.brand .title-block .subtitle {
  font-size: 13px;
  color: #586374;
}
main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 32px 80px;
}
section.page-header {
  background: #ffffff;
  padding: 20px 24px;
  border-radius: 6px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
section.page-header h2 {
  margin: 0 0 6px;
  color: #0a2540;
}
section.page-header .meta {
  font-size: 13px;
  color: #586374;
}
section.legend {
  background: #eef2f7;
  border-left: 3px solid #0a2540;
  padding: 14px 18px;
  margin-bottom: 24px;
  font-size: 13px;
}
section.legend .ladder span {
  margin-right: 18px;
  font-weight: 600;
}
section.legend .ladder .calibrated { color: #1e8449; }
section.legend .ladder .interpolated { color: #d4860b; }
section.legend .ladder .theoretical { color: #7f3b8d; }
section.legend .ladder .unknown { color: #7b8490; }
.layer-panel {
  background: #ffffff;
  margin-bottom: 18px;
  padding: 20px 24px;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.layer-panel header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.layer-panel header h3 {
  margin: 0;
  color: #0a2540;
  font-size: 17px;
}
.layer-panel .badge {
  font-size: 11px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.badge.calibrated { background: #d5f0df; color: #1e8449; }
.badge.interpolated { background: #fbefd8; color: #d4860b; }
.badge.theoretical { background: #eeddf3; color: #7f3b8d; }
.badge.not_populated { background: #e3e6eb; color: #586374; }
.badge.unknown { background: #e3e6eb; color: #586374; }
.layer-panel .summary {
  color: #3a4452;
  margin: 6px 0 10px;
}
.layer-panel .placeholder {
  color: #7b8490;
  font-style: italic;
  padding: 14px 0;
}
.layer-panel .metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.layer-panel .metric {
  background: #f3f5f8;
  padding: 10px 12px;
  border-radius: 4px;
}
.layer-panel .metric .name {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #586374;
}
.layer-panel .metric .value {
  font-size: 18px;
  font-weight: 600;
  color: #0a2540;
}
.layer-panel .metric .unit {
  font-size: 12px;
  color: #586374;
  margin-left: 4px;
}
footer.brand {
  text-align: center;
  padding: 18px;
  font-size: 12px;
  color: #7b8490;
  background: #ffffff;
  border-top: 1px solid #e3e6eb;
}
footer.brand .plan-ref {
  margin-top: 4px;
  color: #586374;
}
"""


def _render_brand_header(assets: BrandAssets, page_title: str, subtitle: str) -> str:
    logo_html = (
        f'<img class="logo" src="{assets.logo_data_uri}" alt="{assets.logo_alt}"/>'
        if assets.logo_data_uri
        else f'<span class="logo-text">{assets.logo_alt}</span>'
    )
    return f"""
<header class="brand">
  {logo_html}
  <div class="title-block">
    <h1>{html.escape(page_title)}</h1>
    <div class="subtitle">{html.escape(subtitle)}</div>
  </div>
</header>
"""


def _render_brand_footer(plan_version: str) -> str:
    return f"""
<footer class="brand">
  <div>Branes AI -- micro-architectural model delivery</div>
  <div class="plan-ref">Plan: {html.escape(plan_version)}</div>
</footer>
"""


def _render_legend() -> str:
    return """
<section class="legend">
  <div><strong>Confidence ladder:</strong> every metric ships with one of the
  tags below. A higher tag only lands when measurement backs it up.</div>
  <div class="ladder">
    <span class="calibrated">CALIBRATED</span>
    <span class="interpolated">INTERPOLATED</span>
    <span class="theoretical">THEORETICAL</span>
    <span class="unknown">UNKNOWN</span>
  </div>
</section>
"""


def _render_metric(name: str, entry: Dict) -> str:
    value = entry.get("value", "")
    unit = entry.get("unit", "")
    return f"""
<div class="metric">
  <div class="name">{html.escape(name)}</div>
  <div class="value">{html.escape(str(value))}<span class="unit">{html.escape(unit)}</span></div>
</div>
"""


def _render_panel(panel: LayerPanel) -> str:
    raw_status = panel.status or "not_populated"
    badge_class = _safe_status_class(raw_status)
    summary_block = (
        f'<p class="summary">{html.escape(panel.summary)}</p>'
        if panel.summary and badge_class != "not_populated"
        else ""
    )
    if badge_class == "not_populated":
        body = '<div class="placeholder">NOT YET POPULATED</div>'
    else:
        metrics_html = "".join(
            _render_metric(k, v) for k, v in panel.metrics.items()
        )
        body = (
            f'{summary_block}'
            f'<div class="metrics-grid">{metrics_html}</div>'
        )
    # Display text is fully HTML-escaped; class attribute uses the whitelisted token.
    display_status = html.escape(raw_status.replace("_", " "))
    return f"""
<div class="layer-panel">
  <header>
    <h3>{html.escape(panel.title)}</h3>
    <span class="badge {badge_class}">{display_status}</span>
  </header>
  {body}
</div>
"""


def render_sku_page(
    report: MicroarchReport,
    repo_root: Path,
) -> str:
    """
    Render a single-SKU HTML page from a MicroarchReport.

    The returned HTML is self-contained (embedded CSS + data-URI logo).
    """
    assets = _load_logo(repo_root)
    panels_html = "".join(_render_panel(p) for p in report.layers)
    page_title = f"Micro-arch report: {report.display_name or report.sku}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(page_title)}</title>
  <style>{_CSS}</style>
</head>
<body>
{_render_brand_header(assets, page_title, report.generated_at)}
<main>
  <section class="page-header">
    <h2>{html.escape(report.display_name or report.sku)}</h2>
    <div class="meta">
      SKU: <code>{html.escape(report.sku)}</code> &middot;
      Archetype: {html.escape(report.archetype or "unspecified")} &middot;
      Overall confidence: <strong>{html.escape(report.overall_confidence)}</strong>
    </div>
  </section>
  {_render_legend()}
  {panels_html}
</main>
{_render_brand_footer(report.plan_version)}
</body>
</html>
"""


def _render_layer1_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render a peak-TOPS table per precision across all SKUs.

    Pulls data via the layer-panel builder so the table reflects the
    same ComputeFabric numbers as the per-SKU panels.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer1_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer1_chart(sku_ids)
    if not chart.precisions:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    body_rows = []
    for prec in chart.precisions:
        cells = [f"<th>{html.escape(prec.upper())}</th>"]
        for sku in chart.skus:
            peak = chart.peak_ops.get((prec, sku))
            if peak is None:
                cells.append("<td>--</td>")
                continue
            tops = peak / 1e12
            prov = chart.provenance.get((prec, sku), "UNKNOWN")
            badge = _safe_status_class(prov.lower())
            cells.append(
                f'<td>{tops:.2f}<br/>'
                f'<span class="badge {badge}">{html.escape(prov)}</span>'
                f'</td>'
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_rows)

    return f"""
<section>
  <h3>Layer 1: peak ALU throughput across SKUs (TOPS)</h3>
  <p class="meta">Sum across populated ComputeFabrics per SKU. Provenance
  badge tracks how each per-precision rate was sourced.</p>
  <table class="layer1-cross">
    <thead><tr><th>Precision</th>{header_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</section>
"""


def _render_layer2_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render a register-energy table across all SKUs.

    Surfaces register read / write energy and the read-as-fraction-
    of-ALU ratio. Pulls data from the same builder the per-SKU panels
    use so values stay consistent.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer2_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer2_chart(sku_ids)
    if not chart.register_read_pj:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    def _row(label: str, getter, fmt: str) -> str:
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in chart.skus:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
                continue
            cells.append(f"<td>{fmt.format(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    rows = (
        _row("Register read (pJ)",
             lambda s: chart.register_read_pj.get(s), "{:.3f}")
        + _row("Register write (pJ)",
               lambda s: chart.register_write_pj.get(s), "{:.3f}")
        + _row("Read / ALU ratio",
               lambda s: chart.read_alu_ratio.get(s), "{:.2f}")
    )

    badge_cells = []
    for sku in chart.skus:
        prov = chart.provenance.get(sku, "UNKNOWN")
        badge = _safe_status_class(prov.lower())
        badge_cells.append(
            f'<td><span class="badge {badge}">{html.escape(prov)}</span></td>'
        )
    rows += "<tr><th>Provenance</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Layer 2: register-file energy across SKUs</h3>
  <p class="meta">Read / write energies sourced from
  TechnologyProfile keyed by each SKU's process node and deployment
  market. Read / ALU ratio quantifies operand-fetch overhead -- the
  KPU's domain-flow fabric is intentionally absent from this fetch.</p>
  <table class="layer2-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _render_layer3_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render an L1 capacity / energy / storage-kind table across SKUs.

    Pulls data from cross_sku_layer3_chart so the values stay in sync
    with each per-SKU panel.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer3_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer3_chart(sku_ids)
    if not chart.l1_per_unit_bytes:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    def _kib_str(b):
        if b is None:
            return "--"
        if b >= 1024 * 1024:
            return f"{b / (1024*1024):.1f} MiB"
        return f"{b / 1024:.0f} KiB"

    def _row(label, getter, fmt):
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in chart.skus:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    rows = (
        _row("L1 per unit",
             lambda s: chart.l1_per_unit_bytes.get(s), _kib_str)
        + _row("L1 total",
               lambda s: chart.l1_total_bytes.get(s), _kib_str)
        + _row("Storage kind",
               lambda s: chart.storage_kind.get(s),
               lambda v: html.escape(v))
        + _row("Energy (pJ/byte)",
               lambda s: chart.energy_pj_per_byte.get(s),
               lambda v: f"{v:.3f}")
    )

    badge_cells = []
    for sku in chart.skus:
        prov = chart.provenance.get(sku, "UNKNOWN")
        badge = _safe_status_class(prov.lower())
        badge_cells.append(
            f'<td><span class="badge {badge}">{html.escape(prov)}</span></td>'
        )
    rows += "<tr><th>Provenance</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Layer 3: L1 cache / scratchpad capacity across SKUs</h3>
  <p class="meta">Cache-managed SKUs (CPU, GPU L1) carry a per-op-type
  hit rate; scratchpad-managed SKUs (KPU tile-local SRAM, TPU unified
  buffer, Hailo on-chip SRAM) are deterministic 1.0 by design and
  rely on the host compiler for staging.</p>
  <table class="layer3-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _render_layer4_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render an L2 capacity / topology / energy table across SKUs.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer4_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer4_chart(sku_ids)
    if not chart.l2_per_unit_bytes:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    def _kib_str(b):
        if b is None:
            return "--"
        if b == 0:
            return "collapsed"
        if b >= 1024 * 1024:
            return f"{b / (1024*1024):.1f} MiB"
        return f"{b / 1024:.0f} KiB"

    def _row(label, getter, fmt):
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in chart.skus:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    rows = (
        _row("L2 per unit",
             lambda s: chart.l2_per_unit_bytes.get(s), _kib_str)
        + _row("L2 total",
               lambda s: chart.l2_total_bytes.get(s), _kib_str)
        + _row("Topology",
               lambda s: chart.topology.get(s),
               lambda v: html.escape(v))
        + _row("Energy (pJ/byte)",
               lambda s: chart.energy_pj_per_byte.get(s),
               lambda v: f"{v:.3f}")
    )

    badge_cells = []
    for sku in chart.skus:
        prov = chart.provenance.get(sku, "UNKNOWN")
        badge = _safe_status_class(prov.lower())
        badge_cells.append(
            f'<td><span class="badge {badge}">{html.escape(prov)}</span></td>'
        )
    rows += "<tr><th>Provenance</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Layer 4: L2 cache capacity / topology across SKUs</h3>
  <p class="meta">Physical L2 from
  <code>l2_cache_per_unit</code> (M4 field). Topology distinguishes
  private per-core L2 (CPUs, KPU tile-local) from shared L2 and
  shared-LLC (Ampere SoCs, TPU UB collapse, Hailo on-chip SRAM).
  &quot;collapsed&quot; means the SKU has no distinct L2 layer
  -- the unified buffer covers L1 + L2 staging by design.</p>
  <table class="layer4-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _render_layer5_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render an L3 / LLC presence + capacity + coherence table across SKUs.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer5_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer5_chart(sku_ids)
    if not chart.l3_present:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    def _row(label, getter, fmt):
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in chart.skus:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    def _cap(b):
        if b is None or b == 0:
            return "no L3"
        if b >= 1024 * 1024:
            return f"{b / (1024 * 1024):.1f} MiB"
        return f"{b / 1024:.0f} KiB"

    # Suppress numeric energy rows when the SKU explicitly has no
    # Layer 5 (no L3, no coherence). Otherwise the table renders a
    # phantom non-zero value next to a "no L3" cell, which contradicts
    # the per-SKU panel's classification.
    def _l3_energy_or_dash(s):
        if not chart.l3_present.get(s):
            return None
        return chart.energy_pj_per_byte.get(s)

    def _coherence_pj_or_dash(s):
        if chart.coherence_protocol.get(s) == "none":
            return None
        return chart.coherence_pj_per_request.get(s)

    rows = (
        _row("L3 present",
             lambda s: chart.l3_present.get(s),
             lambda v: "yes" if v else "no")
        + _row("L3 total",
               lambda s: chart.l3_total_bytes.get(s), _cap)
        + _row("Coherence",
               lambda s: chart.coherence_protocol.get(s),
               lambda v: html.escape(v))
        + _row("Coherence pJ/req",
               _coherence_pj_or_dash,
               lambda v: f"{v:.2f}")
        + _row("L3 energy (pJ/byte)",
               _l3_energy_or_dash,
               lambda v: f"{v:.3f}")
    )

    badge_cells = []
    for sku in chart.skus:
        prov = chart.provenance.get(sku, "UNKNOWN")
        badge = _safe_status_class(prov.lower())
        badge_cells.append(
            f'<td><span class="badge {badge}">{html.escape(prov)}</span></td>'
        )
    rows += "<tr><th>Provenance</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Layer 5: L3 / LLC presence + coherence across SKUs</h3>
  <p class="meta">CPUs carry a distinct L3 (LLC); GPU SoCs collapse
  L2 into the LLC; KPU / TPU / Hailo dataflow architectures have no
  L3-equivalent and route data through the mesh fabric.
  Layer 5 owns the coherence-PROTOCOL energy cost (snoop messages,
  state transitions); the TRANSPORT cost (NoC hops) is M6 (Layer 6).</p>
  <table class="layer5-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _render_layer6_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render a SoC fabric topology + hop / energy table across SKUs.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer6_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer6_chart(sku_ids)
    if not chart.topology:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    def _row(label, getter, fmt):
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in chart.skus:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    rows = (
        _row("Topology",
             lambda s: chart.topology.get(s),
             lambda v: html.escape(v))
        + _row("Avg hops",
               lambda s: chart.avg_hop_count.get(s),
               lambda v: f"{v:.2f}")
        + _row("Hop latency (ns)",
               lambda s: chart.hop_latency_ns.get(s),
               lambda v: f"{v:.2f}")
        + _row("pJ / flit / hop",
               lambda s: chart.pj_per_flit_per_hop.get(s),
               lambda v: f"{v:.2f}")
        + _row("Bisection BW (Gbps)",
               lambda s: chart.bisection_bandwidth_gbps.get(s),
               lambda v: f"{v:.0f}")
    )

    # Confidence row: blends provenance with the low_confidence flag
    badge_cells = []
    for sku in chart.skus:
        lc = chart.low_confidence.get(sku, False)
        prov = chart.provenance.get(sku, "UNKNOWN")
        if lc:
            label = "LOW-CONFIDENCE"
            badge = "theoretical"  # show as theoretical-style badge
        else:
            label = prov
            badge = _safe_status_class(prov.lower())
        badge_cells.append(
            f'<td><span class="badge {badge}">{html.escape(label)}</span></td>'
        )
    rows += "<tr><th>Confidence</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Layer 6: on-chip fabric / NoC across SKUs</h3>
  <p class="meta">Topology and per-hop coefficients for the on-chip
  interconnect that moves packets between cores, caches, and memory
  controllers. Layer 6 owns the TRANSPORT cost; the coherence
  PROTOCOL cost stays at Layer 5. SKUs whose vendors do not publish
  NoC details ship with a LOW-CONFIDENCE badge per the M6 issue
  constraint.</p>
  <table class="layer6-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _render_layer7_cross_sku_table(reports: List[MicroarchReport]) -> str:
    """
    Render the Layer 7 external-memory comparison: technology,
    bandwidth, R/W energy, asymmetry. The bandwidth-vs-pJ/byte
    column pair gives the visually striking comparison from the M7
    motivation -- DDR5 pays >2x the pJ/byte of LPDDR5 at lower BW.
    """
    try:
        from graphs.reporting.layer_panels import cross_sku_layer7_chart
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_layer7_chart(sku_ids)
    if not chart.memory_technology:
        return ""

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in chart.skus
    )

    def _row(label, getter, fmt):
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in chart.skus:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    rows = (
        _row("Memory technology",
             lambda s: chart.memory_technology.get(s),
             lambda v: html.escape(v))
        + _row("Peak BW (GB/s)",
               lambda s: chart.peak_bandwidth_gbps.get(s),
               lambda v: f"{v:.1f}")
        + _row("Read energy (pJ/B)",
               lambda s: chart.read_energy_pj.get(s),
               lambda v: f"{v:.2f}")
        + _row("Write energy (pJ/B)",
               lambda s: chart.write_energy_pj.get(s),
               lambda v: f"{v:.2f}")
        + _row("W/R asymmetry",
               lambda s: chart.asymmetry.get(s),
               lambda v: f"{v:.2f}x")
    )

    badge_cells = []
    for sku in chart.skus:
        prov = chart.provenance.get(sku, "UNKNOWN")
        badge = _safe_status_class(prov.lower())
        badge_cells.append(
            f'<td><span class="badge {badge}">{html.escape(prov)}</span></td>'
        )
    rows += "<tr><th>Provenance</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Layer 7: external memory across SKUs</h3>
  <p class="meta">External-memory characteristics. Bandwidth and
  pJ/byte trade off across technology generations: DDR5 desktop
  pays the highest energy per byte; LPDDR5 mobile / on-package
  is the sweet spot for edge inference; HBM3 (not in this catalog
  yet) drops below 10 pJ/B at the cost of die area. Hailo SKUs
  load weights from host DRAM at initialization and serve
  steady-state inference from on-chip SRAM, so their host-side
  numbers describe the cold-start path.</p>
  <table class="layer7-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _render_validation_cross_sku_table(
    reports: List[MicroarchReport],
) -> str:
    """
    Render the Path A measurement-validation cross-SKU summary.

    Only emits a table when at least one SKU in the report set has
    validation results attached. Reads from the in-process cache the
    validation panel populated, so this is a free read (no extra
    UnifiedAnalyzer calls).
    """
    try:
        from graphs.reporting.layer_panels import (
            cross_sku_validation_chart,
        )
    except Exception:
        return ""

    sku_ids = [r.sku for r in reports]
    chart = cross_sku_validation_chart(sku_ids)
    if not chart.n_results:
        return ""  # Nothing validated; suppress the section entirely

    name_lookup = {r.sku: (r.display_name or r.sku) for r in reports}
    only_validated = [s for s in chart.skus if s in chart.n_results]
    header_cells = "".join(
        f"<th>{html.escape(name_lookup.get(sku, sku))}</th>"
        for sku in only_validated
    )

    def _row(label, getter, fmt):
        cells = [f"<th>{html.escape(label)}</th>"]
        for sku in only_validated:
            v = getter(sku)
            if v is None:
                cells.append("<td>--</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    rows = (
        _row("Models validated",
             lambda s: chart.n_results.get(s),
             lambda v: str(v))
        + _row("Median MAPE",
               lambda s: chart.median_mape_pct.get(s),
               lambda v: f"{v:.1f}%")
    )
    badge_cells = []
    for sku in only_validated:
        ok = chart.within_tolerance.get(sku, False)
        cls = "interpolated" if ok else "theoretical"
        label = "INTERPOLATED" if ok else "THEORETICAL"
        badge_cells.append(
            f'<td><span class="badge {cls}">{label}</span></td>'
        )
    rows += "<tr><th>Aggregate confidence</th>" + "".join(badge_cells) + "</tr>"

    return f"""
<section>
  <h3>Path A: end-to-end measurement validation</h3>
  <p class="meta">SKUs with measurement data in
  <code>calibration_data/</code> have their M1-M7 analytical
  predictions compared against measured per-model latency. Median
  MAPE within +/-30% promotes the SKU's aggregate confidence from
  THEORETICAL to INTERPOLATED. SKUs without measurements are
  omitted from this table; their confidence stays at THEORETICAL
  until a calibration campaign lands.</p>
  <table class="validation-cross">
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def render_comparison_page(
    reports: List[MicroarchReport],
    repo_root: Path,
) -> str:
    """
    Render a cross-SKU comparison page.

    Currently surfaces:
      - SKU summary table (always)
      - Layer 1 peak-TOPS-per-precision table (M1)
      - Layer 2 register-energy table (M2)
      - Layer 3 L1-capacity / storage-kind table (M3)
      - Layer 4 L2-capacity / topology table (M4)
      - Layer 5 L3-presence / coherence table (M5)
      - Layer 6 SoC fabric topology + hop coefficients (M6)
      - Layer 7 external-memory technology + bandwidth + pJ/B (M7)
      - Path A measurement validation summary (when --with-validation)

    M8 adds the engineering-deck export.
    """
    assets = _load_logo(repo_root)
    rows = "".join(
        f"<tr><td>{html.escape(r.display_name or r.sku)}</td>"
        f"<td><code>{html.escape(r.sku)}</code></td>"
        f"<td>{html.escape(r.archetype or '')}</td>"
        f"<td>{html.escape(r.overall_confidence)}</td></tr>"
        for r in reports
    )
    layer1_section = _render_layer1_cross_sku_table(reports)
    layer2_section = _render_layer2_cross_sku_table(reports)
    layer3_section = _render_layer3_cross_sku_table(reports)
    layer4_section = _render_layer4_cross_sku_table(reports)
    layer5_section = _render_layer5_cross_sku_table(reports)
    layer6_section = _render_layer6_cross_sku_table(reports)
    layer7_section = _render_layer7_cross_sku_table(reports)
    validation_section = _render_validation_cross_sku_table(reports)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Cross-SKU comparison</title>
  <style>{_CSS}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
table th, table td {{ padding: 10px 12px; border-bottom: 1px solid #e3e6eb; text-align: left; }}
table th {{ font-size: 12px; text-transform: uppercase; color: #586374; background: #f3f5f8; }}
table.layer1-cross td, table.layer2-cross td, table.layer3-cross td, table.layer4-cross td, table.layer5-cross td, table.layer6-cross td, table.layer7-cross td {{ text-align: right; }}
table.layer1-cross th:first-child, table.layer2-cross th:first-child, table.layer3-cross th:first-child, table.layer4-cross th:first-child, table.layer5-cross th:first-child, table.layer6-cross th:first-child, table.layer7-cross th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
{_render_brand_header(assets, "Cross-SKU comparison", "Micro-architectural model delivery")}
<main>
  <section class="page-header">
    <h2>Compare across SKUs</h2>
    <div class="meta">
      All seven layers populated (ALU, Register File, L1 Cache /
      Scratchpad, L2 Cache, L3 / LLC, SoC Data Movement, External
      Memory). Engineering-deck export lands at M8.
    </div>
  </section>
  {_render_legend()}
  <section>
    <table>
      <thead><tr><th>SKU</th><th>ID</th><th>Archetype</th><th>Confidence</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  {layer1_section}
  {layer2_section}
  {layer3_section}
  {layer4_section}
  {layer5_section}
  {layer6_section}
  {layer7_section}
  {validation_section}
</main>
{_render_brand_footer("microarch-model-delivery-plan.md")}
</body>
</html>
"""


def render_index_page(
    reports: List[MicroarchReport],
    repo_root: Path,
) -> str:
    """
    Render the landing / index page linking to each per-SKU page and the
    comparison view.
    """
    assets = _load_logo(repo_root)
    entries = "".join(
        f'<li><a href="hardware/{html.escape(r.sku)}.html">'
        f'{html.escape(r.display_name or r.sku)}</a> '
        f'-- <span class="meta">{html.escape(r.archetype or "unspecified")}, '
        f'{html.escape(r.overall_confidence)}</span></li>'
        for r in reports
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Micro-architectural model delivery</title>
  <style>{_CSS}
ul.sku-list {{ list-style: none; padding: 0; }}
ul.sku-list li {{ padding: 10px 14px; background: #fff; margin-bottom: 6px;
                 border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
ul.sku-list a {{ color: #0a2540; text-decoration: none; font-weight: 600; }}
ul.sku-list a:hover {{ text-decoration: underline; }}
.meta {{ color: #586374; font-size: 13px; font-weight: 400; }}
section.highlighted-link {{ background: #fff; border-left: 4px solid #3fc98a;
                             padding: 14px 18px; border-radius: 4px;
                             margin-bottom: 18px;
                             box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
section.highlighted-link h3 {{ margin: 0 0 6px; color: #0a2540; }}
section.highlighted-link p {{ margin: 6px 0; color: #3a4452; }}
a.primary-link {{ display: inline-block; padding: 8px 16px; background: #0a2540;
                  color: #fff; border-radius: 4px; text-decoration: none;
                  font-weight: 600; }}
a.primary-link:hover {{ background: #15385c; }}
  </style>
</head>
<body>
{_render_brand_header(assets, "Micro-architectural model delivery", "Layers 1-7 across the target SKU set")}
<main>
  <section class="page-header">
    <h2>Report index</h2>
    <div class="meta">Per-SKU layer panels, cross-SKU comparison, and the compute-archetype exploration harness.</div>
  </section>
  {_render_legend()}
  <section class="highlighted-link">
    <h3>Compute-archetype comparison (GPU vs. TPU vs. KPU)</h3>
    <p>The M0.5 exploration harness: five Plotly charts comparing energy per op,
    peak throughput, ops/W, pipeline utilization vs. tile count, and PE array-size scaling.</p>
    <p><a class="primary-link" href="compare_archetypes.html">Open compute-archetype comparison &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Native-op energy breakdown (specific products)</h3>
    <p>Theoretical energy floor per MAC for each <em>shipping product</em>'s
    native operation, broken down by memory-hierarchy layer. Coral (14nm)
    vs. KPU T128 (16nm) vs. Jetson Orin AGX (8nm) - process advantages
    mixed in.</p>
    <p><a class="primary-link" href="native_op_energy.html">Open native-op breakdown &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Generalized architecture comparison (apples-to-apples)</h3>
    <p>Same data but with the process-technology advantage stripped out:
    CPU / GPU / TPU / KPU / DSP / DFM / CGRA at matched process, plus
    process-scaling curves and peak-TOPS-at-fixed-TDP envelopes.</p>
    <p><a class="primary-link" href="generalized_architecture.html">Open generalized comparison &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Competitive trajectory: Jetson vs. KPU target</h3>
    <p>How long would NVIDIA's Jetson line take to reach the KPU T128
    TOPS/W target at its demonstrated rate of improvement? Historical
    trajectory + extrapolations + parity-year analysis.</p>
    <p><a class="primary-link" href="competitive_trajectory.html">Open trajectory analysis &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Per-structure energy accounting (SM vs. KPU tile)</h3>
    <p>Model-validation view: itemizes every micro-architectural structure
    that fires on the native operation (HMMA instruction for the
    Streaming Multiprocessor, PE MAC in domain-flow wavefront for the
    KPU tile) with citations and amortization factors. Cross-validates
    the simplified architectural-efficiency model.</p>
    <p><a class="primary-link" href="microarch_accounting.html">Open per-structure accounting &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Building-block energy (per-clock view, for SoC composition)</h3>
    <p>Engine-level budget: total pJ/clock for every major component of
    the SM (register file, instruction pipeline, warp scheduler,
    operand collectors, CUDA cores, SFUs, Tensor Cores) and the KPU
    tile (L1 scratchpad, 2D FMA mesh, edge injectors). Used to
    compose SoCs and super-clusters: total power = count x utilization
    x power-per-block + overhead. Cross-validates the per-MAC view.</p>
    <p><a class="primary-link" href="building_block_energy.html">Open building-block energy &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Silicon composition hierarchy (ALU &rarr; PE &rarr; Tile &rarr; Cluster &rarr; SoC)</h3>
    <p>Tracks silicon efficiency as an ALU is wrapped in successively
    larger composition levels. Shows TOPS/W decay across canonical
    hierarchies: NVIDIA Ampere GPU (Orin AGX), KPU T128 NPU,
    ARM Cortex-A78 CPU cluster, and Qualcomm Hexagon DSP. The
    per-level "active fraction" exposes how much silicon serves
    coordination vs compute and is the primary input to workload-
    driven SoC specialization.</p>
    <p><a class="primary-link" href="silicon_composition.html">Open silicon-composition hierarchy &rarr;</a></p>
  </section>
  <section class="highlighted-link">
    <h3>Mission Capability per Watt (embodied-AI feasibility)</h3>
    <p>Applies the silicon-efficiency analysis to ten catalogued
    embodied-AI missions (nano-swarm, body-worn exoskeleton,
    30-day AUV, 6-month glider fleet, 72-hour UGV pack mule, 7-day
    USAR microrobot, HALE pseudosatellite, LEO sat onboard AI,
    autonomous ag tractor, 8-hour humanoid). For each mission,
    GPU-class vs KPU-class architectures are evaluated against the
    platform's binding physical threshold (battery, thermal
    envelope, payload mass). Outputs a binary CAN-DO / CANNOT-DO
    verdict plus mission-hours-enabled curves for energy-bound
    missions.</p>
    <p><a class="primary-link" href="mission_capability.html">Open mission-capability analysis &rarr;</a></p>
  </section>
  <section>
    <h3>Per-SKU layer panels</h3>
    <p class="meta">Layer 1-7 content populates in milestones M1-M7.
    Panels currently show <em>NOT YET POPULATED</em> placeholders by design.</p>
    <ul class="sku-list">{entries}</ul>
    <p><a href="compare.html">Cross-SKU layer comparison (shell, populated at M8) &rarr;</a></p>
  </section>
</main>
{_render_brand_footer("microarch-model-delivery-plan.md")}
</body>
</html>
"""
