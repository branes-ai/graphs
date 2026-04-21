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
  <div>Branes AI &mdash; micro-architectural model delivery</div>
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


def render_comparison_page(
    reports: List[MicroarchReport],
    repo_root: Path,
) -> str:
    """
    Render a cross-SKU comparison shell. M0 ships the shell only; M8
    fills in the interactive Plotly charts.
    """
    assets = _load_logo(repo_root)
    rows = "".join(
        f"<tr><td>{html.escape(r.display_name or r.sku)}</td>"
        f"<td><code>{html.escape(r.sku)}</code></td>"
        f"<td>{html.escape(r.archetype or '')}</td>"
        f"<td>{html.escape(r.overall_confidence)}</td></tr>"
        for r in reports
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Cross-SKU comparison</title>
  <style>{_CSS}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
table th, table td {{ padding: 10px 12px; border-bottom: 1px solid #e3e6eb; text-align: left; }}
table th {{ font-size: 12px; text-transform: uppercase; color: #586374; background: #f3f5f8; }}
  </style>
</head>
<body>
{_render_brand_header(assets, "Cross-SKU comparison", "Micro-architectural model delivery")}
<main>
  <section class="page-header">
    <h2>Compare across SKUs</h2>
    <div class="meta">
      M0 ships the shell; interactive Plotly comparison charts land at M8.
    </div>
  </section>
  {_render_legend()}
  <section>
    <table>
      <thead><tr><th>SKU</th><th>ID</th><th>Archetype</th><th>Confidence</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
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
        f'&mdash; <span class="meta">{html.escape(r.archetype or "unspecified")}, '
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
  </style>
</head>
<body>
{_render_brand_header(assets, "Micro-architectural model delivery", "Layers 1-7 across the target SKU set")}
<main>
  <section class="page-header">
    <h2>Report index</h2>
    <div class="meta">Per-SKU layer panels + cross-SKU comparison.</div>
  </section>
  {_render_legend()}
  <section>
    <h3>SKUs</h3>
    <ul class="sku-list">{entries}</ul>
    <p><a href="compare.html">Cross-SKU comparison &rarr;</a></p>
  </section>
</main>
{_render_brand_footer("microarch-model-delivery-plan.md")}
</body>
</html>
"""
