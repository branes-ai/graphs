"""The pipeline, stage by stage: what each one demands and what each engine
gives it (graphs#269 Phase 7.4).

The frontier page places whole configurations on a plane. It cannot show
*why* a configuration lands where it does, because that happens one stage
at a time: a mission is 19 stages over seven tiers, each with its own
arithmetic, its own byte traffic, its own precision floor, and a different
answer from every engine.

This draws that. One row per stage, in pipeline order:

* **what it demands** -- operations and bytes per second of mission, and
  the precision classes it needs;
* **what each engine gives it** -- the share of that engine one second of
  mission would consume, at the best efficiency anything states, or the
  reason there is no figure.

A bar past the full mark means one second of mission needs more than one
second of that engine: the stage cannot keep up there, on its own, before
anything else is scheduled beside it.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Ordinal steps for the precision classes: one neutral ramp, wider format
#: further from the surface. Drawn through CSS variables so dark mode gets
#: its own steps rather than an inversion of these: on a dark panel the
#: ramp runs light where this one runs dark, and the legend, which reads
#: the same variables, follows it.
CLASS_STEPS: Dict[str, str] = {"A": "var(--cls-a)", "B": "var(--cls-b)", "C": "var(--cls-c)"}
CLASS_FORMATS: Dict[str, str] = {"A": "INT8 or wider", "B": "FP16 or wider", "C": "FP32 or wider"}

#: Ordinal steps for where a figure came from. Not status colours: this is
#: an ordered quality, measured being the strongest, and the ceiling step
#: is the lighter of the two in both modes.
PROVENANCE_STEPS: Dict[str, str] = {"measured": "var(--prov-measured)",
                                    "ceiling": "var(--prov-ceiling)"}

#: The steps themselves, per mode, for the palette check and the CSS.
LIGHT_STEPS: Dict[str, str] = {"cls-a": "#b0aea6", "cls-b": "#7f7e76", "cls-c": "#4a4945",
                               "prov-measured": "#256abf", "prov-ceiling": "#86b6ef",
                               "over": "#c93434"}
DARK_STEPS: Dict[str, str] = {"cls-a": "#5b5a54", "cls-b": "#939187", "cls-c": "#d6d4cb",
                              "prov-measured": "#3987e5", "prov-ceiling": "#9cc3f2",
                              "over": "#e66767"}

ROW_H = 26
LABEL_W = 210
DEMAND_W = 150
ENGINE_W = 190


@dataclass(frozen=True)
class StageRow:
    """One stage of one mission, and what each engine does with it."""

    key: str
    name: str
    tier: str
    kernel_class: str
    unit: str
    rate_hz: float
    ops_per_s: float
    bytes_per_s: float
    class_split: Dict[str, float]
    on_reactive_chain: bool
    assigned: Optional[str] = None
    #: engine -> (share of that engine per second of mission, provenance,
    #: efficiency, why there is no figure)
    engines: Dict[str, Tuple[Optional[float], str, Optional[float], Optional[str]]] = \
        field(default_factory=dict)


def _log_width(value: float, lo: float, hi: float, width: float) -> float:
    if value <= 0:
        return 0.0
    span = math.log10(hi) - math.log10(lo) or 1.0
    return max(1.0, min(1.0, (math.log10(max(value, lo)) - math.log10(lo)) / span) * width)


def _si(value: float, unit: str) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= scale:
            return f"{value / scale:.3g} {suffix}{unit}"
    return f"{value:.3g} {unit}"


def _rate(rate_hz: float, unit: str) -> str:
    """The stage's own rate in its own unit. A stage is not always called at
    a frame rate: mono runs per pixel, ctrl per update per degree of
    freedom, and "111 M/s" alone would not say which."""
    what = unit[4:] if unit.startswith("per ") else (unit or "call")
    return f"{_si(rate_hz, '')} {html.escape(what)}/s".replace("  ", " ")


def _engine_cell(x: float, y: float, share: Optional[float], provenance: str,
                 efficiency: Optional[float], why: Optional[str], stage: str,
                 engine: str, assigned: bool) -> str:
    """One engine's answer for one stage: a bar against its own capacity."""
    parts = [f'<rect class="track" x="{x}" y="{y + 7}" width="{ENGINE_W - 56}" height="{10}"/>']
    if share is None:
        parts.append(f'<text class="none" x="{x + 4}" y="{y + 16}">'
                     f'{html.escape(why or "no figure")}</text>')
    else:
        full = ENGINE_W - 56
        over = share > 1.0
        width = full if over else max(1.5, share * full)
        colour = PROVENANCE_STEPS.get(provenance, "var(--prov-ceiling)")
        payload = html.escape(json.dumps({
            "stage": stage, "engine": engine, "share": round(share, 4),
            "eff": None if efficiency is None else round(efficiency, 5),
            "prov": provenance, "assigned": assigned,
        }), quote=True)
        parts.append(f'<rect class="bar{" over" if over else ""}" x="{x}" y="{y + 7}" '
                     f'width="{width:.1f}" height="10" style="--c:{colour}" '
                     f'data-cell="{payload}"/>')
        label = (f"{share:.2f}x" if share >= 0.995 else f"{share * 100:.2g}%")
        parts.append(f'<text class="share{" over" if over else ""}" '
                     f'x="{x + full + 5}" y="{y + 16}">{label}</text>')
    if assigned:
        parts.append(f'<text class="assigned" x="{x - 7}" y="{y + 16}">&#9656;</text>')
    return "".join(parts)


def _row(row: StageRow, y: float, ops: Tuple[float, float], byts: Tuple[float, float],
         engines: Sequence[str]) -> str:
    parts = [f'<text class="stage" x="26" y="{y + 16}">{html.escape(row.key)}</text>',
             f'<text class="kernel" x="82" y="{y + 16}">{html.escape(row.kernel_class)}</text>']
    if row.on_reactive_chain:
        parts.append(f'<text class="chain" x="16" y="{y + 16}" '
                     f'aria-label="on the sense-to-act chain">&#9679;</text>')

    # The precision classes the stage needs, as a share of its own ops.
    x = LABEL_W
    for cls in ("A", "B", "C"):
        share = row.class_split.get(cls, 0.0)
        if share <= 0:
            continue
        width = share * 46
        parts.append(f'<rect class="cls" x="{x:.1f}" y="{y + 7}" width="{width:.1f}" '
                     f'height="10" fill="{CLASS_STEPS[cls]}">'
                     f'<title>{share:.0%} {html.escape(CLASS_FORMATS[cls])}</title></rect>')
        x += width

    x = LABEL_W + 58
    track = DEMAND_W - 74
    for value, (lo, hi), unit in ((row.ops_per_s, ops, "OP/s"), (row.bytes_per_s, byts, "B/s")):
        parts.append(f'<rect class="track" x="{x}" y="{y + 9}" width="{track}" height="6"/>')
        parts.append(f'<rect class="demand" x="{x}" y="{y + 9}" '
                     f'width="{_log_width(value, lo, hi, track):.1f}" height="6"/>')
        parts.append(f'<text class="amount" x="{x + DEMAND_W - 10}" y="{y + 16}">'
                     f'{_si(value, unit)}</text>')
        x += DEMAND_W
    for engine in engines:
        share, provenance, efficiency, why = row.engines.get(engine, (None, "unpriced", None,
                                                                      "not offered"))
        parts.append(_engine_cell(x + 10, y, share, provenance, efficiency, why, row.key,
                                  engine, row.assigned == engine))
        x += ENGINE_W
    return "".join(parts)


def pipeline_svg(rows: Sequence[StageRow], engines: Sequence[str], tiers: Dict[str, str]) -> str:
    """The whole pipeline for one mission, one row per stage."""
    if not rows:
        return "<p>no stage to draw</p>"
    ops = (max(min(r.ops_per_s for r in rows if r.ops_per_s > 0), 1.0),
           max(r.ops_per_s for r in rows))
    byts = (max(min(r.bytes_per_s for r in rows if r.bytes_per_s > 0), 1.0),
            max(r.bytes_per_s for r in rows))
    width = LABEL_W + 58 + 2 * DEMAND_W + len(engines) * ENGINE_W + 20
    header = 46
    body: List[str] = []
    y = header
    for tier in sorted({r.tier for r in rows}):
        band = [r for r in rows if r.tier == tier]
        body.append(f'<text class="tier" x="8" y="{y + 10}">{html.escape(tier)} '
                    f'&middot; {html.escape(tiers.get(tier, ""))}</text>')
        y += 16
        for row in sorted(band, key=lambda r: -r.ops_per_s):
            body.append(f'<rect class="rowbg" x="8" y="{y}" width="{width - 16}" '
                        f'height="{ROW_H - 2}"/>')
            body.append(_row(row, y, ops, byts, engines))
            y += ROW_H
        y += 6
    head = [
        '<text class="col" x="26" y="20">stage</text>',
        '<text class="col" x="82" y="20">kernel class</text>',
        f'<text class="col" x="{LABEL_W}" y="20">precision</text>',
        f'<text class="col" x="{LABEL_W + 58}" y="20">demands: operations</text>',
        f'<text class="col" x="{LABEL_W + 58 + DEMAND_W}" y="20">and bytes</text>',
    ]
    for i, engine in enumerate(engines):
        head.append(f'<text class="col" x="{LABEL_W + 58 + 2 * DEMAND_W + i * ENGINE_W + 10}" '
                    f'y="20">gets from the {html.escape(engine)}</text>')
        head.append(f'<text class="col-sub" x="{LABEL_W + 58 + 2 * DEMAND_W + i * ENGINE_W + 10}" '
                    f'y="33">share of it per second of mission</text>')
    return (f'<svg viewBox="0 0 {width} {y}" class="pipeline" role="img" '
            f'aria-label="pipeline stages, their demand and what each engine gives them">'
            + "".join(head) + "".join(body) + "</svg>")


def _steps_css(steps: Dict[str, str]) -> str:
    return " ".join(f"--{name}:{value};" for name, value in steps.items())


def stage_table(rows: Sequence[StageRow], engines: Sequence[str],
                tiers: Dict[str, str]) -> str:
    """The same rows as text, so identity never rests on colour alone and
    the figures can be read, copied and checked without the chart."""
    head = ("<tr><th>tier</th><th>stage</th><th>kernel class</th><th>precision</th>"
            "<th>rate</th><th>operations/s</th><th>bytes/s</th>"
            + "".join(f"<th>on the {html.escape(e)}</th>" for e in engines)
            + "<th>scheduled on</th><th>sense-to-act</th></tr>")
    body = []
    for row in sorted(rows, key=lambda r: (r.tier, -r.ops_per_s)):
        cells = []
        for engine in engines:
            share, provenance, efficiency, why = row.engines.get(
                engine, (None, "unpriced", None, "not offered"))
            if share is None:
                cells.append(f'<td>{html.escape(why or "no figure")}</td>')
            else:
                cells.append(f'<td class="num">{share * 100:.3g}% of one'
                             f' ({html.escape(provenance)})</td>')
        precision = ", ".join(f"{share:.0%} {CLASS_FORMATS[cls]}"
                              for cls in ("A", "B", "C")
                              if (share := row.class_split.get(cls, 0.0)) > 0)
        body.append(
            f'<tr><td>{html.escape(row.tier)} {html.escape(tiers.get(row.tier, ""))}</td>'
            f'<td>{html.escape(row.key)}</td><td>{html.escape(row.kernel_class)}</td>'
            f'<td>{html.escape(precision)}</td>'
            f'<td class="num">{_rate(row.rate_hz, row.unit)}</td>'
            f'<td class="num">{_si(row.ops_per_s, "OP/s")}</td>'
            f'<td class="num">{_si(row.bytes_per_s, "B/s")}</td>'
            + "".join(cells)
            + f'<td>{html.escape(row.assigned or "-")}</td>'
            + f'<td>{"yes" if row.on_reactive_chain else "no"}</td></tr>')
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


STYLE = """
:root { color-scheme: light dark; --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b;
  --ink-2:#52514e; --ink-3:#78766f; --rule:#e2e1dc; --row:#f7f6f3; --track:#eceae5;
  __LIGHT__ }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface:#1a1a19; --panel:#232322; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8f8e85;
  --rule:#34342f; --row:#1f1f1e; --track:#2c2c29;
  __DARK__ } }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:1280px; margin:0 auto; padding:32px 16px 80px; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:19px; margin:36px 0 10px; }
p, li { color:var(--ink-2); max-width:76ch; }
.lede { font-size:17px; color:var(--ink); }
.panel { background:var(--panel); border:1px solid var(--rule); border-radius:10px;
  padding:14px 12px; margin-top:14px; overflow-x:auto; }
svg.pipeline { width:100%; height:auto; min-width:980px; display:block; }
text { font-size:11px; fill:var(--ink); }
text.col { font-size:10px; fill:var(--ink); font-weight:600; }
text.col-sub, text.tier-name { font-size:9px; fill:var(--ink-3); }
text.tier { font-size:10px; fill:var(--ink-2); font-weight:600; }
text.stage { font-size:11.5px; fill:var(--ink); font-weight:600; }
text.kernel, text.amount { font-size:10px; fill:var(--ink-2); }
text.amount { text-anchor:end; }
text.none { font-size:9.5px; fill:var(--ink-3); font-style:italic; }
text.share { font-size:10px; fill:var(--ink-2); }
text.share.over { fill:var(--over); font-weight:600; }
text.chain { font-size:9px; fill:var(--ink-3); }
text.assigned { font-size:11px; fill:var(--ink); }
rect.rowbg { fill:var(--row); }
rect.track { fill:var(--track); }
rect.demand { fill:var(--ink-3); }
rect.bar { fill:var(--c); }
rect.bar.over { fill:var(--over); }
rect.bar:hover { stroke:var(--ink); stroke-width:1.5; }
details.tableview { margin-top:12px; overflow-x:auto; }
details.tableview summary { cursor:pointer; font-size:13px; color:var(--ink-2); }
table { border-collapse:collapse; margin-top:10px; font-size:12px; width:100%; }
th, td { text-align:left; padding:4px 10px 4px 0; border-bottom:1px solid var(--rule);
  color:var(--ink-2); white-space:nowrap; }
th { color:var(--ink); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.legend { display:flex; flex-wrap:wrap; gap:16px; align-items:center; font-size:13px;
  color:var(--ink-2); margin-top:12px; }
.key { display:inline-flex; align-items:center; gap:6px; }
.chip { width:22px; height:10px; border-radius:2px; display:inline-block; }
select { font:inherit; padding:6px 8px; border-radius:8px; border:1px solid var(--rule);
  background:var(--panel); color:var(--ink); }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--panel); color:var(--ink); border:1px solid var(--rule); border-radius:8px;
  padding:8px 10px; font-size:12px; line-height:1.45; box-shadow:0 6px 20px rgba(0,0,0,.18);
  max-width:280px; z-index:10; }
section.mission[hidden] { display:none; }
"""

STYLE = (STYLE.replace("__LIGHT__", _steps_css(LIGHT_STEPS))
              .replace("__DARK__", _steps_css(DARK_STEPS)))

SCRIPT = """
const pick = document.getElementById('mission');
const show = id => document.querySelectorAll('section.mission').forEach(
  s => s.hidden = (s.dataset.mission !== id));
pick.addEventListener('change', () => show(pick.value));
show(pick.value);
const tip = document.getElementById('tip');
document.querySelectorAll('rect.bar').forEach(bar => {
  bar.addEventListener('mouseenter', () => {
    const d = JSON.parse(bar.dataset.cell);
    const pct = d.share >= 1 ? `${d.share.toFixed(2)} engines' worth`
      : `${(d.share * 100).toPrecision(3)}% of one`;
    tip.innerHTML = `<b>${d.stage}</b> on the ${d.engine}<br>${pct} per second of mission<br>`
      + (d.eff === null ? 'no efficiency stated' :
         `efficiency ${d.eff} &middot; ${d.prov}`)
      + (d.assigned ? '<br>this is where the schedule puts it' : '');
    tip.style.opacity = 1;
  });
  bar.addEventListener('mousemove', e => {
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 290) + 'px';
    tip.style.top = Math.max(e.clientY - 10, 8) + 'px';
  });
  bar.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
});
"""


def render(missions: Dict[str, Sequence[StageRow]], titles: Dict[str, str],
           subtitles: Dict[str, str], engines: Sequence[str], tiers: Dict[str, str],
           intro: str, generated: str) -> str:
    """The page: a mission picker, then that mission's pipeline."""
    options = "".join(f'<option value="{html.escape(m)}">{html.escape(titles.get(m, m))}</option>'
                      for m in missions)
    sections = "".join(
        f'<section class="mission" data-mission="{html.escape(m)}" hidden>'
        f'<h2>{html.escape(titles.get(m, m))}</h2>'
        f'<p class="sub">{html.escape(subtitles.get(m, ""))}</p>'
        f'<div class="panel">{pipeline_svg(rows, engines, tiers)}</div>'
        f'<details class="tableview"><summary>Table view: the same rows as numbers</summary>'
        f'{stage_table(rows, engines, tiers)}</details></section>'
        for m, rows in missions.items())
    legend = (
        '<div class="legend">'
        + "".join(f'<span class="key"><span class="chip" style="background:{c}"></span>'
                  f'{html.escape(CLASS_FORMATS[k])}</span>' for k, c in CLASS_STEPS.items())
        + "".join(f'<span class="key"><span class="chip" style="background:{c}"></span>'
                  f'{k} efficiency</span>' for k, c in PROVENANCE_STEPS.items())
        + '<span class="key"><span class="chip" style="background:var(--over)"></span>'
          'over one engine</span>'
        + '<span class="key">&#9679; on the sense-to-act chain</span>'
        + '<span class="key">&#9656; where the schedule puts it</span></div>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline demand and supply</title><style>{STYLE}</style></head>
<body><main>
{intro}
<h2>Pick a mission</h2>
<p><select id="mission">{options}</select></p>
{legend}
{sections}
<p class="sub">Generated {html.escape(generated)}.</p>
</main><div id="tip" role="status"></div><script>{SCRIPT}</script></body></html>
"""


__all__ = ["CLASS_FORMATS", "CLASS_STEPS", "DARK_STEPS", "LIGHT_STEPS",
           "PROVENANCE_STEPS", "StageRow", "pipeline_svg", "render",
           "stage_table"]
