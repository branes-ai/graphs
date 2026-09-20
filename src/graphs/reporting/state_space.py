"""The CPU + KPU state space, brushable (graphs#269 Phase 7.5).

The frontier page shows one mission's configurations on a plane, and the
pipeline page shows one configuration's stages. Neither answers the
question a silicon architect actually arrives with: *which* configurations
serve *which* missions, and what is it about them that does it?

This draws the whole space at once. One row per mission, one column per
configuration -- a design, a CPU core count and a memory interface -- and a
cell saying how close that pair comes to real time. The columns are grouped
by CPU core count and ordered by design within each group, so a pattern
that follows one dimension is a pattern you can see.

**A cell is a bound, so the reading is one-sided.** The real-time factor
behind it uses the best efficiency anything states -- a measurement where
one exists, otherwise the domain-flow ceiling -- so a cell short of the
threshold is *proven* short: even the flattering figure does not reach it.
A cell that clears the threshold is not proven to work; it is only *not
ruled out*, which is why the legend says "still standing" and never
"feasible". The colour ramp under the threshold is how far short, and is
read the same way: a bound on the gap, not a measurement of it.

**The brush.** Dragging across the columns selects a slab of the space and
reports what its members have in common and how many missions survive it.
The headroom control raises the bar from real time to a multiple of it,
which is the honest way to ask for margin: every figure on this page
flatters its configuration, so a design that only just clears 1.0 has
nothing in hand.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Steps for "how far short", palest furthest from the threshold, and one
#: status colour for a configuration the evidence does not rule out. Given
#: per mode, so dark mode is selected rather than flipped.
LIGHT_STEPS: Dict[str, str] = {
    "short-1": "#b0aea6", "short-2": "#96948c", "short-3": "#7f7e76", "short-4": "#4a4945",
    "standing": "#256abf", "brush": "#c93434",
}
DARK_STEPS: Dict[str, str] = {
    "short-1": "#5b5a54", "short-2": "#787770", "short-3": "#a3a199", "short-4": "#d6d4cb",
    "standing": "#3987e5", "brush": "#e66767",
}

#: What each step means, as a fraction of the required headroom.
BANDS: Tuple[Tuple[float, str, str], ...] = (
    (0.1, "short-1", "under a tenth of the bar"),
    (0.3, "short-2", "a tenth to a third"),
    (0.6, "short-3", "a third to two thirds"),
    (1.0, "short-4", "two thirds up, still short"),
)

CELL_W = 7
CELL_H = 17
LABEL_W = 306
GROUP_GAP = 14


@dataclass(frozen=True)
class Configuration:
    """One column: a design, a CPU core count and a memory interface."""

    design: str
    node: str
    cpu_cores: int
    memory: str
    kpu_sku: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.design}|{self.cpu_cores}|{self.memory}"

    def to_dict(self) -> dict:
        return {"design": self.design, "node": self.node, "cpu_cores": self.cpu_cores,
                "memory": self.memory, "kpu_sku": self.kpu_sku}


@dataclass(frozen=True)
class MissionRow:
    """One row: a mission, and its real-time factor against every column."""

    mission: str
    title: str
    power_budget_w: float
    deadline_ms: float
    #: Aligned with the column order; None where nothing prices the mission.
    factors: Tuple[Optional[float], ...]
    #: True where the figures omit an unpriced stage, which flatters them.
    partial: Tuple[bool, ...] = ()

    def standing(self, headroom: float = 1.0) -> int:
        return sum(1 for f in self.factors if f is not None and f >= headroom)


def _short(design: str) -> str:
    return design[4:] if design.startswith("kpu_") else design


def _band(factor: Optional[float], headroom: float) -> Optional[str]:
    """Which step a cell wears, or None when nothing prices it."""
    if factor is None:
        return None
    if factor >= headroom:
        return "standing"
    for fraction, step, _label in BANDS:
        if factor < fraction * headroom:
            return step
    return BANDS[-1][1]


def _columns_svg(columns: Sequence[Configuration], rows: Sequence[MissionRow],
                 headroom: float) -> str:
    """The matrix itself: missions down, configurations across."""
    if not columns or not rows:
        return "<p>no configuration to draw</p>"
    groups: List[Tuple[int, List[int]]] = []
    for index, column in enumerate(columns):
        if not groups or groups[-1][0] != column.cpu_cores:
            groups.append((column.cpu_cores, []))
        groups[-1][1].append(index)

    x_of: Dict[int, float] = {}
    x = LABEL_W
    heads: List[str] = []
    for cores, members in groups:
        start = x
        for index in members:
            x_of[index] = x
            x += CELL_W
        heads.append(f'<text class="group" x="{start}" y="20">{cores} CPU cores</text>')
        heads.append(f'<line class="group-rule" x1="{start}" y1="26" '
                     f'x2="{x - 2}" y2="26"/>')
        x += GROUP_GAP
    width = x + 70

    body: List[str] = []
    y = 34
    for row in rows:
        body.append(f'<text class="mission-label" x="{LABEL_W - 8}" y="{y + 12}">'
                    f'{html.escape(row.title)}</text>')
        for index, factor in enumerate(row.factors):
            step = _band(factor, headroom)
            fill = "url(#nothing)" if step is None else f"var(--{step})"
            # Quoted, always: an unquoted value swallows the closing slash,
            # the rect never self-closes, and every later cell nests inside it.
            partial = ' data-partial="1"' if (index < len(row.partial)
                                              and row.partial[index]) else ""
            amount = "" if factor is None else round(factor, 4)
            body.append(
                f'<rect class="cell" data-col="{index}" data-row="{html.escape(row.mission)}" '
                f'x="{x_of[index]:.0f}" y="{y}" width="{CELL_W - 1}" height="{CELL_H - 1}" '
                f'fill="{fill}" data-factor="{amount}"{partial}/>')
        body.append(f'<text class="count" x="{width - 62}" y="{y + 12}" '
                    f'data-count="{html.escape(row.mission)}">'
                    f'{row.standing(headroom)}/{len(row.factors)}</text>')
        y += CELL_H
    # The design axis, once under each group of columns.
    for cores, members in groups:
        seen = set()
        for index in members:
            design = columns[index].design
            if design in seen:
                continue
            seen.add(design)
            body.append(f'<text class="design-label" x="{x_of[index] + 3}" y="{y + 6}" '
                        f'transform="rotate(-60 {x_of[index] + 3} {y + 6})">'
                        f'{html.escape(_short(design))}</text>')
    height = y + 74
    return (f'<svg viewBox="0 0 {width} {height}" class="space" role="img" '
            f'aria-label="missions against configurations, shaded by how close each '
            f'comes to real time">'
            f'<defs><pattern id="nothing" width="4" height="4" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)">'
            f'<rect width="4" height="4" fill="var(--track)"/>'
            f'<line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-3)" stroke-width="1"/>'
            f'</pattern></defs>'
            f'<rect id="brush" x="0" y="0" width="0" height="0"/>'
            + "".join(heads) + "".join(body) + "</svg>")


def _shared(standing: Sequence[Configuration],
            everything: Sequence[Configuration]) -> str:
    """What the surviving configurations have in common, naming only the
    dimensions they do *not* span -- a dimension every survivor takes every
    value of did not decide anything, and saying so would be noise."""
    if not standing:
        return "-"
    parts = []
    for label, attribute in (("CPU cores", "cpu_cores"), ("node", "node"),
                             ("memory", "memory"), ("design", "design")):
        values = {getattr(c, attribute) for c in standing}
        if len(values) == len({getattr(c, attribute) for c in everything}):
            continue
        shown = sorted(_short(str(v)) for v in values)
        parts.append(f"{label}: {', '.join(shown[:4])}"
                     + (f" and {len(shown) - 4} more" if len(shown) > 4 else ""))
    return "; ".join(parts) or "every dimension: nothing narrows it"


def state_table(columns: Sequence[Configuration], rows: Sequence[MissionRow],
                headroom: float) -> str:
    """The same space as numbers: for each mission, how many configurations
    are still standing and what they have in common."""
    body = []
    for row in rows:
        standing = [columns[i] for i, f in enumerate(row.factors)
                    if f is not None and f >= headroom]
        best = max((f for f in row.factors if f is not None), default=None)
        body.append(
            f'<tr><td>{html.escape(row.title)}</td>'
            f'<td class="num">{row.power_budget_w:g} W</td>'
            f'<td class="num">{row.deadline_ms:g} ms</td>'
            f'<td class="num">{len(standing)}/{len(row.factors)}</td>'
            f'<td class="num">{"-" if best is None else f"{best:.3g}x"}</td>'
            f'<td>{html.escape(_shared(standing, columns))}</td></tr>')
    return ("<table><thead><tr><th>mission</th><th>power</th><th>deadline</th>"
            "<th>still standing</th><th>best factor</th>"
            "<th>what the survivors share</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def _steps_css(steps: Dict[str, str]) -> str:
    return " ".join(f"--{name}:{value};" for name, value in steps.items())


STYLE = """
:root { color-scheme: light dark; --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b;
  --ink-2:#52514e; --ink-3:#78766f; --rule:#e2e1dc; --track:#eceae5; __LIGHT__ }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface:#1a1a19; --panel:#232322; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8f8e85;
  --rule:#34342f; --track:#2c2c29; __DARK__ } }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:1360px; margin:0 auto; padding:32px 16px 80px; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:19px; margin:34px 0 10px; }
p, li { color:var(--ink-2); max-width:76ch; }
.lede { font-size:17px; color:var(--ink); }
.panel { background:var(--panel); border:1px solid var(--rule); border-radius:10px;
  padding:14px 12px; margin-top:12px; overflow-x:auto; }
svg.space { width:100%; height:auto; min-width:1080px; display:block; cursor:crosshair; }
text { font-size:11px; fill:var(--ink); }
text.group { font-size:10.5px; fill:var(--ink); font-weight:600; }
line.group-rule { stroke:var(--rule); stroke-width:1; }
text.mission-label { font-size:11px; fill:var(--ink-2); text-anchor:end; }
text.count { font-size:10px; fill:var(--ink-3); font-variant-numeric:tabular-nums; }
text.design-label { font-size:8.5px; fill:var(--ink-3); text-anchor:end; }
rect.cell { shape-rendering:crispEdges; }
rect.cell.dim { opacity:.18; }
rect.cell.hit { stroke:var(--ink); stroke-width:1.2; }
#brush { fill:var(--brush); fill-opacity:.12; stroke:var(--brush); stroke-width:1;
  pointer-events:none; }
.controls { display:flex; flex-wrap:wrap; gap:18px; align-items:flex-end; margin-top:14px; }
.control { display:flex; flex-direction:column; gap:5px; }
.control > span { font-size:11px; color:var(--ink-3); text-transform:uppercase;
  letter-spacing:.04em; }
#headroom-value { text-transform:none; letter-spacing:0; font-variant-numeric:tabular-nums; }
.chips { display:flex; gap:5px; flex-wrap:wrap; }
button.chip { font:inherit; font-size:12.5px; padding:4px 10px; border-radius:999px;
  border:1px solid var(--rule); background:var(--panel); color:var(--ink-2); cursor:pointer; }
button.chip[aria-pressed="true"] { background:var(--standing); border-color:var(--standing);
  color:#fff; }
input[type=range] { width:190px; accent-color:var(--standing); }
.readout { margin-top:14px; padding:12px 14px; border:1px solid var(--rule);
  border-radius:10px; background:var(--panel); font-size:13.5px; color:var(--ink-2); }
.readout b { color:var(--ink); }
.legend { display:flex; flex-wrap:wrap; gap:14px; align-items:center; font-size:12.5px;
  color:var(--ink-2); margin-top:12px; }
.key { display:inline-flex; align-items:center; gap:6px; }
.chip-swatch { width:20px; height:11px; border-radius:2px; display:inline-block; }
details.tableview { margin-top:14px; overflow-x:auto; }
details.tableview summary { cursor:pointer; font-size:13px; color:var(--ink-2); }
table { border-collapse:collapse; margin-top:10px; font-size:12px; width:100%; }
th, td { text-align:left; padding:4px 10px 4px 0; border-bottom:1px solid var(--rule);
  color:var(--ink-2); }
th { color:var(--ink); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--panel); color:var(--ink); border:1px solid var(--rule); border-radius:8px;
  padding:8px 10px; font-size:12px; line-height:1.45; box-shadow:0 6px 20px rgba(0,0,0,.18);
  max-width:300px; z-index:10; }
"""

STYLE = (STYLE.replace("__LIGHT__", _steps_css(LIGHT_STEPS))
              .replace("__DARK__", _steps_css(DARK_STEPS)))


SCRIPT = """
const DATA = JSON.parse(document.getElementById('data').textContent);
const svg = document.querySelector('svg.space');
const cells = Array.from(svg.querySelectorAll('rect.cell'));
const brush = document.getElementById('brush');
const readout = document.getElementById('readout');
const tip = document.getElementById('tip');
const counts = Array.from(svg.querySelectorAll('text.count'));
let headroom = 1.0;
let selection = null;          // [firstColumn, lastColumn] or null
const filters = { cpu_cores: new Set(), node: new Set(), memory: new Set() };

const passes = i => {
  const c = DATA.columns[i];
  return Object.entries(filters).every(([k, set]) => !set.size || set.has(String(c[k])));
};
const inBrush = i => !selection || (i >= selection[0] && i <= selection[1]);
const chosen = () => DATA.columns.map((_, i) => i).filter(i => passes(i) && inBrush(i));

const bandOf = f => {
  if (f === null) return null;
  if (f >= headroom) return 'standing';
  for (const [fraction, step] of DATA.bands) if (f < fraction * headroom) return step;
  return DATA.bands[DATA.bands.length - 1][1];
};

function repaint() {
  const live = new Set(chosen());
  for (const cell of cells) {
    const i = +cell.dataset.col;
    cell.classList.toggle('dim', !live.has(i));
    const raw = cell.dataset.factor;
    const step = bandOf(raw === '' ? null : +raw);
    if (step) cell.setAttribute('fill', `var(--${step})`);
  }
  for (const label of counts) {
    const row = DATA.rows[label.dataset.count];
    const n = row.filter((f, i) => live.has(i) && f !== null && f >= headroom).length;
    label.textContent = `${n}/${live.size}`;
  }
  describe(live);
}

function describe(live) {
  const picked = DATA.columns.filter((_, i) => live.has(i));
  const span = key => {
    const values = [...new Set(picked.map(c => String(c[key])))].sort();
    const all = [...new Set(DATA.columns.map(c => String(c[key])))];
    return values.length === all.length ? null : values.join(', ');
  };
  const narrow = [['CPU cores', 'cpu_cores'], ['node', 'node'], ['memory', 'memory'],
                  ['design', 'design']]
    .map(([label, key]) => [label, span(key)]).filter(([, v]) => v)
    .map(([label, v]) => `${label} ${v}`);
  const served = Object.entries(DATA.rows).filter(([, row]) =>
    row.some((f, i) => live.has(i) && f !== null && f >= headroom));
  const none = Object.keys(DATA.rows).length - served.length;
  readout.innerHTML =
    `<b>${picked.length}</b> of ${DATA.columns.length} configurations selected`
    + (narrow.length ? ` &mdash; ${narrow.join('; ')}` : ' &mdash; the whole space')
    + `.<br><b>${served.length}</b> of ${Object.keys(DATA.rows).length} missions have a`
    + ` configuration left standing at ${headroom.toFixed(2)}x real time`
    + (none ? `; <b>${none}</b> have none.` : '.')
    + (served.length ? `<br>Standing: ${served.map(([m]) => DATA.titles[m]).join(', ')}.` : '');
}

// --- the brush: drag across the columns -----------------------------------
const columnAt = clientX => {
  const box = svg.getBoundingClientRect();
  const x = (clientX - box.left) * (svg.viewBox.baseVal.width / box.width);
  let best = null, gap = Infinity;
  DATA.x.forEach((left, i) => {
    const d = Math.abs(left + DATA.cellW / 2 - x);
    if (d < gap) { gap = d; best = i; }
  });
  return best;
};
let anchor = null;
svg.addEventListener('mousedown', e => {
  anchor = columnAt(e.clientX);
  selection = [anchor, anchor];
  drawBrush();
  e.preventDefault();
});
window.addEventListener('mousemove', e => {
  if (anchor === null) return;
  const here = columnAt(e.clientX);
  selection = [Math.min(anchor, here), Math.max(anchor, here)];
  drawBrush();
});
window.addEventListener('mouseup', () => {
  if (anchor === null) return;
  anchor = null;
  if (selection && selection[0] === selection[1]) { selection = null; drawBrush(); }
  repaint();
});
function drawBrush() {
  if (!selection) { brush.setAttribute('width', 0); repaint(); return; }
  const left = DATA.x[selection[0]];
  const right = DATA.x[selection[1]] + DATA.cellW;
  brush.setAttribute('x', left - 1);
  brush.setAttribute('y', 28);
  brush.setAttribute('width', right - left + 1);
  brush.setAttribute('height', DATA.matrixH);
  repaint();
}
document.getElementById('clear').addEventListener('click', () => {
  selection = null; anchor = null;
  for (const set of Object.values(filters)) set.clear();
  document.querySelectorAll('button.chip[data-dim]').forEach(
    b => b.setAttribute('aria-pressed', 'false'));
  drawBrush();
});

// --- the chips and the headroom -------------------------------------------
document.querySelectorAll('button.chip[data-dim]').forEach(button => {
  button.addEventListener('click', () => {
    const on = button.getAttribute('aria-pressed') !== 'true';
    button.setAttribute('aria-pressed', String(on));
    const set = filters[button.dataset.dim];
    on ? set.add(button.dataset.value) : set.delete(button.dataset.value);
    repaint();
  });
});
const slider = document.getElementById('headroom');
const shown = document.getElementById('headroom-value');
slider.addEventListener('input', () => {
  headroom = Math.pow(10, +slider.value / 100);
  shown.textContent = headroom.toFixed(2) + 'x';
  repaint();
});

// --- the hover layer ------------------------------------------------------
for (const cell of cells) {
  cell.addEventListener('mouseenter', () => {
    const c = DATA.columns[+cell.dataset.col];
    const raw = cell.dataset.factor;
    const factor = raw === '' ? null : +raw;
    cell.classList.add('hit');
    tip.innerHTML = `<b>${DATA.titles[cell.dataset.row]}</b><br>`
      + `${c.design} &middot; ${c.cpu_cores} CPU cores &middot; ${c.memory}<br>`
      + `${c.node}<br>`
      + (factor === null ? 'nothing prices this mission here'
         : (factor >= headroom
            ? `<b>${factor.toFixed(2)}x real time</b> &mdash; not ruled out`
            : `<b>${factor.toFixed(2)}x real time</b> &mdash; proven short of `
              + `${headroom.toFixed(2)}x`))
      + (cell.dataset.partial ? '<br>figures omit an unpriced stage, so this flatters it' : '');
    tip.style.opacity = 1;
  });
  cell.addEventListener('mousemove', e => {
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 310) + 'px';
    tip.style.top = Math.max(e.clientY - 10, 8) + 'px';
  });
  cell.addEventListener('mouseleave', () => {
    cell.classList.remove('hit'); tip.style.opacity = 0;
  });
}
repaint();
"""


def render(columns: Sequence[Configuration], rows: Sequence[MissionRow],
           intro: str, generated: str, headroom: float = 1.0) -> str:
    """The page: controls, the brushable matrix, a readout and a table."""
    x_of: List[float] = []
    x = LABEL_W
    previous: Optional[int] = None
    for column in columns:
        if previous is not None and column.cpu_cores != previous:
            x += GROUP_GAP
        previous = column.cpu_cores
        x_of.append(x)
        x += CELL_W

    data = {
        "columns": [c.to_dict() for c in columns],
        "rows": {r.mission: list(r.factors) for r in rows},
        "titles": {r.mission: r.title for r in rows},
        "bands": [[fraction, step] for fraction, step, _ in BANDS],
        "x": x_of, "cellW": CELL_W, "matrixH": len(rows) * CELL_H + 4,
    }

    def chips(label: str, dimension: str, values: Sequence[str]) -> str:
        buttons = "".join(
            f'<button class="chip" type="button" aria-pressed="false" '
            f'data-dim="{dimension}" data-value="{html.escape(str(v))}">'
            f'{html.escape(_short(str(v)))}</button>' for v in values)
        return (f'<div class="control"><span>{html.escape(label)}</span>'
                f'<div class="chips">{buttons}</div></div>')

    controls = (
        '<div class="controls">'
        + chips("CPU cores", "cpu_cores", sorted({c.cpu_cores for c in columns}))
        + chips("process node", "node", sorted({c.node for c in columns}))
        + chips("memory", "memory", sorted({c.memory for c in columns}))
        + ('<div class="control"><span>required headroom</span>'
           '<input type="range" id="headroom" min="0" max="100" step="1" value="0" '
           'aria-label="required real-time headroom">'
           f'<span id="headroom-value">{headroom:.2f}x</span></div>')
        + '<div class="control"><span>&nbsp;</span>'
          '<button class="chip" type="button" id="clear">clear the brush</button></div>'
        + '</div>')

    legend = (
        '<div class="legend">'
        + "".join(f'<span class="key"><span class="chip-swatch" '
                  f'style="background:var(--{step})"></span>{html.escape(label)}</span>'
                  for _fraction, step, label in BANDS)
        + '<span class="key"><span class="chip-swatch" style="background:var(--standing)">'
          '</span>clears the bar: not ruled out</span>'
        + '<span class="key"><span class="chip-swatch" '
          'style="background:var(--track);border:1px solid var(--ink-3)"></span>'
          'nothing prices it</span></div>')

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The CPU + KPU state space</title><style>{STYLE}</style></head>
<body><main>
{intro}
<h2>Brush the space</h2>
{controls}
<div class="panel">{_columns_svg(columns, rows, headroom)}</div>
{legend}
<div class="readout" id="readout" role="status"></div>
<details class="tableview"><summary>Table view: the same space as numbers</summary>
{state_table(columns, rows, headroom)}</details>
<p class="sub">Generated {html.escape(generated)}.</p>
</main><div id="tip" role="status"></div>
<script type="application/json" id="data">{json.dumps(data)}</script>
<script>{SCRIPT}</script></body></html>
"""


__all__ = ["BANDS", "Configuration", "DARK_STEPS", "LIGHT_STEPS", "MissionRow",
           "render", "state_table"]
