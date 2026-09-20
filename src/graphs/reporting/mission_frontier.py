"""The energy-performance plane for every mission, as a standalone page
(graphs#269 Phase 7.3).

The state-space tables say which configurations are ruled out. They do not
show the *shape* of the trade, and a table of 2,754 rows is not a thing a
person reads. This renders the same data as one chart per mission: energy
per op across, real-time factor up, the envelope over the catalog, and the
region where a configuration is already proven short.

The page is self-contained -- inline SVG, inline CSS, a few lines of
vanilla JavaScript for the hover layer -- so it can be mailed to a partner
or opened from a file share with nothing installed.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from graphs.estimation.soc.frontier import MissionPoint, envelope

#: Node -> categorical slot, light and dark. Three slots, which is what a
#: scatter can carry: the dataviz method's all-pairs cap.
NODE_COLORS: Dict[str, Tuple[str, str]] = {
    "tsmc_n7": ("#2a78d6", "#3987e5"),
    "tsmc_n16": ("#eb6834", "#d95926"),
    "gf_12fdx": ("#1baf7a", "#199e70"),
}
NODE_LABELS: Dict[str, str] = {
    "tsmc_n7": "TSMC N7", "tsmc_n16": "TSMC N16", "gf_12fdx": "GF 12FDX",
}

CHART_W, CHART_H = 340, 268
PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 26, 56


@dataclass(frozen=True)
class Scale:
    lo: float
    hi: float
    px_lo: float
    px_hi: float
    log: bool = False

    def __call__(self, value: float) -> float:
        lo, hi, v = self.lo, self.hi, max(value, 1e-12)
        if self.log:
            lo, hi, v = math.log10(lo), math.log10(hi), math.log10(v)
        span = (hi - lo) or 1.0
        return self.px_lo + (v - lo) / span * (self.px_hi - self.px_lo)


def _radius(point: MissionPoint, tiles: Dict[str, int]) -> float:
    """Marker size carries the fabric's tile count, so a bigger KPU is a
    bigger dot -- identity never rests on colour alone."""
    count = tiles.get(point.design, 64)
    return 2.6 + 2.4 * math.log10(max(count, 16) / 16.0)


def _ticks(lo: float, hi: float, log: bool) -> List[float]:
    if log:
        out, decade = [], math.floor(math.log10(lo))
        while 10 ** decade <= hi * 1.001:
            for step in (1, 3):
                value = step * 10 ** decade
                if lo * 0.999 <= value <= hi * 1.001:
                    out.append(value)
            decade += 1
        return out
    step = (hi - lo) / 4.0
    return [lo + i * step for i in range(5)]


def _fmt(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.3g}"
    return f"{value:.3g}"


def _chart(mission: str, title: str, subtitle: str, points: Sequence[MissionPoint],
           x: Scale, y: Scale, tiles: Dict[str, int]) -> str:
    front = envelope(points)
    front_ids = {id(p) for p in front}
    real_time_y = y(1.0)

    parts: List[str] = [
        f'<svg viewBox="0 0 {CHART_W} {CHART_H}" role="img" '
        f'aria-label="{html.escape(title)}: real-time factor against energy per op">',
        f'<title>{html.escape(title)}</title>',
    ]
    # The region a point cannot be in and still work: below real time.
    if real_time_y < PAD_T + (CHART_H - PAD_T - PAD_B):
        top = max(real_time_y, PAD_T)
        parts.append(f'<rect class="short" x="{PAD_L}" y="{top:.1f}" '
                     f'width="{CHART_W - PAD_L - PAD_R}" '
                     f'height="{CHART_H - PAD_B - top:.1f}"/>')
    # Grid and axes.
    for tick in _ticks(y.lo, y.hi, y.log):
        py = y(tick)
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{py:.1f}" '
                     f'x2="{CHART_W - PAD_R}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{py + 3:.1f}" '
                     f'text-anchor="end">{_fmt(tick)}</text>')
    for tick in _ticks(x.lo, x.hi, x.log):
        px = x(tick)
        parts.append(f'<text class="tick" x="{px:.1f}" y="{CHART_H - PAD_B + 14:.1f}" '
                     f'text-anchor="middle">{_fmt(tick)}</text>')
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{CHART_H - PAD_B}" '
                 f'x2="{CHART_W - PAD_R}" y2="{CHART_H - PAD_B}"/>')
    if y.lo <= 1.0 <= y.hi:
        parts.append(f'<line class="realtime" x1="{PAD_L}" y1="{real_time_y:.1f}" '
                     f'x2="{CHART_W - PAD_R}" y2="{real_time_y:.1f}"/>')
        parts.append(f'<text class="realtime-label" x="{CHART_W - PAD_R}" '
                     f'y="{real_time_y - 4:.1f}" text-anchor="end">real time</text>')

    # The envelope staircase, over its *distinct* positions: configurations
    # that tie exactly are one place on the plane, not a zero-length line.
    corners: List[Tuple[float, float]] = []
    for p in front:
        corner = (round(x(p.energy_per_op_pj), 2), round(y(p.real_time_factor), 2))
        if corner not in corners:
            corners.append(corner)
    if len(corners) > 1:
        steps = []
        for i, (px, py) in enumerate(corners):
            steps.append(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}")
            if i + 1 < len(corners):
                steps.append(f"L{corners[i + 1][0]:.1f},{py:.1f}")
        parts.append(f'<path class="envelope" d="{" ".join(steps)}"/>')

    groups: Dict[Tuple[float, float], List[MissionPoint]] = {}
    for point in points:
        if point.real_time_factor is None or point.energy_per_op_pj <= 0:
            continue
        key = (round(x(point.energy_per_op_pj), 1), round(y(point.real_time_factor), 1))
        groups.setdefault(key, []).append(point)
    for (px, py), members in sorted(groups.items(), key=lambda kv: any(
            id(p) in front_ids for p in kv[1])):
        on_front = any(id(p) in front_ids for p in members)
        # The smallest fabric that reaches this spot is the one worth
        # sizing the mark by: the bigger ones buy nothing here.
        smallest = min(members, key=lambda p: tiles.get(p.design, 64))
        colour = NODE_COLORS.get(smallest.node, NODE_COLORS["tsmc_n7"])[0]
        payload = html.escape(json.dumps({
            "design": smallest.design, "node": NODE_LABELS.get(smallest.node, smallest.node),
            "cores": smallest.cpu_cores, "memory": smallest.memory,
            "rt": round(smallest.real_time_factor, 4),
            "pj": round(smallest.energy_per_op_pj, 4),
            "engine": max(smallest.utilization_by_engine,
                          key=smallest.utilization_by_engine.get, default=""),
            "prov": smallest.provenance, "unpriced": len(smallest.unpriced_stages),
            "front": on_front, "also": sorted({p.design for p in members})[:6],
            "n": len(members),
        }), quote=True)
        parts.append(
            f'<circle class="pt{" on-front" if on_front else ""}" '
            f'cx="{px:.2f}" cy="{py:.2f}" '
            f'r="{_radius(smallest, tiles) + (1.6 if on_front else 0):.2f}" '
            f'style="--c:{colour}" data-point="{payload}"/>')

    parts.append(f'<text class="axis-title" x="{PAD_L + (CHART_W - PAD_L - PAD_R) / 2:.0f}" '
                 f'y="{CHART_H - 8}" text-anchor="middle">energy per op (pJ) &#8594;</text>')
    parts.append(f'<text class="axis-title" transform="translate(12,'
                 f'{PAD_T + (CHART_H - PAD_T - PAD_B) / 2:.0f}) rotate(-90)" '
                 f'text-anchor="middle">real-time factor &#8594;</text>')
    parts.append("</svg>")

    reach = sum(1 for p in points if (p.real_time_factor or 0) >= 1.0)
    cheapest = min(front, key=lambda p: p.energy_per_op_pj, default=None)
    fastest = max(front, key=lambda p: p.real_time_factor, default=None)
    if cheapest is None:
        verdict = "nothing in the space can be placed on the plane"
    elif len(corners) == 1:
        ties = sum(1 for p in front if p is not cheapest) + 1
        verdict = (f"one point dominates: {cheapest.energy_per_op_pj:.3f} pJ/op at "
                   f"{cheapest.real_time_factor:.2f}x real time"
                   + (f", tied by {ties} configurations" if ties > 1 else "")
                   + f". {reach or 'none'} of {len(points)} reach real time")
    else:
        verdict = (f"cheapest {cheapest.energy_per_op_pj:.3f} pJ/op at "
                   f"{cheapest.real_time_factor:.2f}x; fastest {fastest.real_time_factor:.2f}x at "
                   f"{fastest.energy_per_op_pj:.3f} pJ/op. "
                   f"{reach or 'none'} of {len(points)} reach real time")

    return (f'<figure class="chart"><figcaption><h3>{html.escape(title)}</h3>'
            f'<p class="sub">{html.escape(subtitle)}</p></figcaption>'
            + "".join(parts)
            + f'<p class="verdict">{html.escape(verdict)}</p></figure>')


def _table(points_by_mission: Dict[str, List[MissionPoint]], titles: Dict[str, str]) -> str:
    rows = []
    for mission, points in points_by_mission.items():
        front = envelope(points)
        for p in front:
            rows.append(
                f"<tr><td>{html.escape(titles.get(mission, mission))}</td>"
                f"<td>{html.escape(p.design)}</td>"
                f"<td>{html.escape(NODE_LABELS.get(p.node, p.node))}</td>"
                f"<td class='n'>{p.cpu_cores}</td>"
                f"<td>{html.escape(p.memory)}</td>"
                f"<td class='n'>{p.energy_per_op_pj:.3f}</td>"
                f"<td class='n'>{p.real_time_factor:.3f}</td>"
                f"<td>{'proven short' if p.attainable is False else 'open'}</td></tr>")
    return (
        "<table><caption>The envelope of each mission: the configurations nothing in the "
        "catalog provably beats on both axes. Energy is a floor; the real-time factor is a "
        "ceiling.</caption><thead><tr><th>Mission</th><th>Design</th><th>Node</th>"
        "<th>CPU cores</th><th>Memory</th><th>pJ per op (floor)</th>"
        "<th>Real-time factor (ceiling)</th><th>Verdict</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


STYLE = """
:root { color-scheme: light dark; --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b;
  --ink-2: #52514e; --ink-3: #78766f; --rule: #e2e1dc; --short: #f2f1ee;
  --n7: #2a78d6; --n16: #eb6834; --n12: #1baf7a; --front: #0b0b0b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface: #1a1a19; --panel: #232322; --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8f8e85;
  --rule: #34342f; --short: #222221;
  --n7: #3987e5; --n16: #d95926; --n12: #199e70; --front: #ffffff; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 26px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 10px; letter-spacing: -0.01em; }
h3 { font-size: 14px; margin: 0; font-weight: 600; }
p, li { color: var(--ink-2); max-width: 74ch; }
.lede { font-size: 17px; color: var(--ink); }
.grid { display: grid; gap: 18px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  margin-top: 18px; }
.chart { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
  padding: 12px 10px 8px; }
.chart svg { width: 100%; height: auto; display: block; }
figcaption { padding: 0 4px 4px; }
.sub, .verdict { font-size: 12px; color: var(--ink-3); margin: 2px 4px 0; }
.verdict { margin-top: 4px; }
.axis { stroke: var(--ink-3); stroke-width: 1; }
.grid-line, .grid { stroke: var(--rule); stroke-width: 1; }
text.tick { font-size: 9.5px; fill: var(--ink-2); }
text.axis-title { font-size: 11px; fill: var(--ink); font-weight: 500; }
.realtime { stroke: var(--ink-2); stroke-width: 1.5; stroke-dasharray: 4 3; }
.realtime-label { font-size: 9px; fill: var(--ink-2); }
.short { fill: var(--short); }
.envelope { fill: none; stroke: var(--front); stroke-width: 2; stroke-opacity: 0.55;
  stroke-linejoin: round; }
circle.pt { fill: var(--c); fill-opacity: 0.72; stroke: var(--panel); stroke-width: 1; }
circle.pt.on-front { fill-opacity: 1; stroke-width: 2; }
circle.pt:hover { stroke: var(--ink); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin: 14px 0 0;
  font-size: 13px; color: var(--ink-2); }
.key { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
table { border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13px; }
caption { text-align: left; color: var(--ink-3); font-size: 12px; padding-bottom: 8px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule); }
th { color: var(--ink); font-weight: 600; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
#tip { position: fixed; pointer-events: none; opacity: 0; transition: opacity .08s;
  background: var(--panel); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.45;
  box-shadow: 0 6px 20px rgba(0,0,0,.18); max-width: 260px; z-index: 10; }
#tip b { color: var(--ink); }
details { margin-top: 10px; } summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
"""

SCRIPT = """
const tip = document.getElementById('tip');
document.querySelectorAll('circle.pt').forEach(node => {
  node.addEventListener('mouseenter', event => {
    const d = JSON.parse(node.dataset.point);
    tip.innerHTML = `<b>${d.design}</b> &middot; ${d.node}<br>`
      + `${d.cores} CPU cores &middot; ${d.memory}<br>`
      + (d.n > 1 ? `<span style="opacity:.75">${d.n} configurations land here: `
          + `${d.also.join(', ')}${d.also.length < d.n ? '&hellip;' : ''}</span><br>` : '')
      + `real-time factor <b>${d.rt}</b> (ceiling)<br>`
      + `energy <b>${d.pj} pJ/op</b> (floor)<br>`
      + `busiest engine: ${d.engine || 'n/a'} &middot; ${d.prov}`
      + (d.unpriced ? `<br>${d.unpriced} stage(s) unpriced` : '')
      + (d.front ? '<br>on the envelope' : '');
    tip.style.opacity = 1;
  });
  node.addEventListener('mousemove', event => {
    tip.style.left = Math.min(event.clientX + 14, window.innerWidth - 270) + 'px';
    tip.style.top = Math.max(event.clientY - 10, 8) + 'px';
  });
  node.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
});
"""


def render(points_by_mission: Dict[str, List[MissionPoint]], titles: Dict[str, str],
           subtitles: Dict[str, str], tiles: Dict[str, int], generated: str,
           intro: str) -> str:
    """The whole page: the model, the charts, the envelope table."""
    every = [p for points in points_by_mission.values() for p in points
             if p.real_time_factor and p.energy_per_op_pj > 0]
    if not every:
        # Nothing could be placed: a node that prices none of the mission's
        # formats, say. Say so rather than dividing by an empty range.
        return (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                f"<title>Mission energy-performance frontier</title><style>{STYLE}</style>"
                f"</head><body><main>{intro}<h2>No configuration could be placed</h2>"
                f"<p>Every configuration was missing either an energy figure or an efficiency "
                f"for every stage, so none has a position on the plane. The gaps are in the "
                f"process-node catalogue and the efficiency table, not in the designs.</p>"
                f"<p class=\"sub\">Generated {html.escape(generated)}.</p></main></body></html>\n")
    x = Scale(min(p.energy_per_op_pj for p in every) * 0.94,
              max(p.energy_per_op_pj for p in every) * 1.06,
              PAD_L, CHART_W - PAD_R)
    y = Scale(min(min(p.real_time_factor for p in every), 0.9) * 0.8,
              max(max(p.real_time_factor for p in every), 1.2) * 1.25,
              CHART_H - PAD_B, PAD_T, log=True)

    charts = "".join(
        _chart(mission, titles.get(mission, mission), subtitles.get(mission, ""), points, x, y,
               tiles)
        for mission, points in points_by_mission.items())
    legend = "".join(
        f'<span class="key"><span class="dot" style="background:var(--{slot})"></span>'
        f'{html.escape(NODE_LABELS[node])}</span>'
        for node, slot in (("tsmc_n7", "n7"), ("tsmc_n16", "n16"), ("gf_12fdx", "n12")))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mission energy-performance frontier</title>
<style>{STYLE}</style></head>
<body><main>
{intro}
<h2>One chart per mission</h2>
<p>Each dot is one configuration: a KPU fabric at the node its SKU states a clock for, a CPU
complement, and a memory system. Across: what the mission's own arithmetic costs on that
configuration, in picojoules per operation. Up, on a log scale: how much of the mission's
required rate it could carry. The dashed line is real time. <b>Everything below it is proven
short</b> &mdash; the figure plotted is already the optimistic one.</p>
<div class="legend">{legend}
<span class="key">dot size = KPU tiles</span>
<span class="key">outlined dots and the dark line = the envelope</span>
<span class="key">shaded band = proven short of real time</span></div>
<p class="sub">Where a mission's envelope is a single point, no configuration in the catalogue
trades energy for performance: one is cheaper <em>and</em> no slower than every other. That is
the usual case here, because the process node moves energy per operation while the busiest
engine &mdash; almost always the CPU &mdash; decides the rate.</p>
<div class="grid">{charts}</div>
<h2>The envelope, as a table</h2>
{_table(points_by_mission, titles)}
<p class="sub">Generated {html.escape(generated)}.</p>
</main><div id="tip" role="status"></div>
<script>{SCRIPT}</script></body></html>
"""


__all__ = ["NODE_COLORS", "NODE_LABELS", "Scale", "render"]
