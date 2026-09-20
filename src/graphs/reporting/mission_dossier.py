"""One mission, one SoC, sized (graphs#269 Phase 7.6).

A dossier rather than a catalogue: what the product has to do, what the
workload demands, what the configuration is, and how big each engine has to
be for the demand to fit. Three drawings carry it --- the workload as a
pipeline graph with demands and latencies on every node, the configuration
as a block diagram with bandwidths on the edges and X/U/E inside every
compute block, and the sizing as a bar per engine against what was
provisioned.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence, Tuple

#: One hue per engine, validated all-pairs in both modes.
LIGHT_STEPS: Dict[str, str] = {
    "cpu": "#256abf", "kpu": "#eb6834", "other": "#7f7e76",
    "fill": "#256abf", "warn": "#c93434", "gap": "#8a8880",
}
DARK_STEPS: Dict[str, str] = {
    "cpu": "#3987e5", "kpu": "#d95926", "other": "#a3a199",
    "fill": "#3987e5", "warn": "#e66767", "gap": "#9a9890",
}

CLASS_FORMATS: Dict[str, str] = {"A": "INT8 or wider", "B": "FP16 or wider",
                                 "C": "FP32 or wider"}


def si(value: Optional[float], unit: str, digits: int = 3) -> str:
    if value is None:
        return "-"
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.{digits}g} {suffix}{unit}"
    return f"{value:.{digits}g} {unit}"


def ms(seconds: Optional[float]) -> str:
    return "-" if seconds is None else f"{seconds * 1e3:.3g} ms"


def _engine_colour(kind) -> str:
    name = getattr(kind, "value", kind)
    return f"var(--{name})" if name in ("cpu", "kpu") else "var(--other)"


# ---------------------------------------------------------------------------
# 1. The workload, as a pipeline graph
# ---------------------------------------------------------------------------

NODE_W, NODE_H, NODE_GAP = 250, 132, 78


def pipeline_graph(dossier) -> str:
    """One box per stage, left to right, with what it demands and what it
    costs on the engine it was sized for."""
    stages = list(dossier.stages)
    if not stages:
        return "<p>no stage to draw</p>"
    placements = {p.stage: p for p in dossier.placements}
    source_w = 150
    first_x = source_w + NODE_GAP          # the arrow needs the whole gap to itself
    width = first_x + len(stages) * (NODE_W + NODE_GAP) + 130
    height = NODE_H + 150
    top = 62
    parts: List[str] = []

    parts.append(f'<text class="dtitle" x="8" y="20">The pipeline, per frame at '
                 f'{dossier.frame_hz:g} Hz</text>')
    parts.append('<text class="dsub" x="8" y="36">box = stage; the number under each arrow '
                 'is the traffic it carries</text>')

    x = first_x
    # The sensors, as the source. The edge label carries the traffic, so the
    # box itself only says what the sensors are.
    parts.append(f'<rect class="source" x="0" y="{top + 24}" width="{source_w}" '
                 f'height="62" rx="9"/>')
    mid = source_w / 2
    parts.append(f'<text class="src-t" x="{mid}" y="{top + 48}">sensors</text>')
    cams = dossier.sensors.get("mono")
    if isinstance(cams, (list, tuple)) and len(cams) == 4:
        parts.append(f'<text class="src-s" x="{mid}" y="{top + 68}">'
                     f'{cams[0]} x {cams[1]}x{cams[2]} @ {cams[3]} Hz</text>')

    for index, stage in enumerate(stages):
        place = placements.get(stage.key)
        kind = place.engine if place else "other"
        colour = _engine_colour(kind)
        # the arrow into this stage
        parts.append(f'<line class="edge" x1="{x - NODE_GAP + 6}" y1="{top + NODE_H / 2}" '
                     f'x2="{x - 7}" y2="{top + NODE_H / 2}" marker-end="url(#arrow)"/>')
        parts.append(f'<text class="edge-l" x="{x - NODE_GAP / 2}" '
                     f'y="{top + NODE_H / 2 - 10}">{si(stage.bytes_per_s, "B/s")}</text>')

        parts.append(f'<rect class="node" x="{x}" y="{top}" width="{NODE_W}" '
                     f'height="{NODE_H}" rx="10"/>')
        parts.append(f'<rect class="node-bar" x="{x}" y="{top}" width="5" '
                     f'height="{NODE_H}" style="fill:{colour}"/>')
        parts.append(f'<text class="n-key" x="{x + 16}" y="{top + 24}">'
                     f'{html.escape(stage.key)}</text>')
        parts.append(f'<text class="n-tier" x="{x + NODE_W - 12}" y="{top + 24}">'
                     f'{html.escape(stage.tier)}</text>')
        parts.append(f'<text class="n-name" x="{x + 16}" y="{top + 41}">'
                     f'{html.escape(stage.name)}</text>')
        parts.append(f'<text class="n-kc" x="{x + 16}" y="{top + 58}">'
                     f'{html.escape(stage.kernel_class)}</text>')
        split = ", ".join(f"{share:.0%} {cls}" for cls, share in stage.class_split.items()
                          if share > 0)
        parts.append(f'<text class="n-row" x="{x + 16}" y="{top + 77}">'
                     f'precision {html.escape(split)}</text>')
        parts.append(f'<text class="n-row" x="{x + 16}" y="{top + 93}">'
                     f'{si(stage.ops_per_s, "OP/s")} &#183; '
                     f'{si(stage.ops_per_call, "OP")}/call</text>')
        if place:
            parts.append(f'<text class="n-lat" x="{x + 16}" y="{top + 116}" '
                         f'style="fill:{colour}">{ms(place.seconds_per_frame)} on '
                         f'{place.servers} {html.escape(kind)} '
                         f'{"tile" if kind == "kpu" else "core"}'
                         f'{"s" if place.servers != 1 else ""}</text>')
        else:
            parts.append(f'<text class="n-gap" x="{x + 16}" y="{top + 116}">'
                         f'no engine can be sized for it</text>')
        x += NODE_W + NODE_GAP

    parts.append(f'<line class="edge" x1="{x - NODE_GAP + 4}" y1="{top + NODE_H / 2}" '
                 f'x2="{x - 8}" y2="{top + NODE_H / 2}" marker-end="url(#arrow)"/>')
    parts.append(f'<rect class="source" x="{x}" y="{top + 24}" width="114" height="62" rx="9"/>')
    parts.append(f'<text class="src-t" x="{x + 57}" y="{top + 48}">tracks</text>')
    parts.append(f'<text class="src-s" x="{x + 57}" y="{top + 66}">'
                 f'{dossier.frame_hz:g} Hz</text>')

    chain = dossier.chain_seconds
    if chain is not None:
        y = top + NODE_H + 40
        parts.append(f'<line class="chain" x1="{first_x}" y1="{y}" '
                     f'x2="{x - NODE_GAP}" y2="{y}"/>')
        parts.append(f'<text class="chain-l" x="{first_x}" y="{y + 20}">'
                     f'sense to track: <tspan class="strong">{ms(chain)}</tspan> '
                     f'against a {dossier.deadline_ms:g} ms deadline'
                     f' &#183; {dossier.deadline_headroom:.1f}x headroom</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="diagram" role="img" '
            f'aria-label="the mission pipeline, one box per stage">'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-3)"/></marker></defs>'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------
# 2. The configuration, as a block diagram
# ---------------------------------------------------------------------------

def block_diagram(dossier, idle_blocks: Sequence[Tuple[str, str]] = ()) -> str:
    """Compute blocks with X, U and E inside them; the interconnect and the
    memory interface with bandwidths on the edges."""
    provisions = list(dossier.provisions)
    if not provisions:
        return "<p>no engine to draw</p>"
    box_w, box_h, gap = 268, 150, 34
    width = max(940, 60 + len(provisions) * (box_w + gap) + len(idle_blocks) * 150 + 60)
    height = 470
    parts: List[str] = []
    parts.append('<text class="dtitle" x="8" y="20">The configuration, as sized</text>')
    parts.append('<text class="dsub" x="8" y="36">X = throughput one server delivers, '
                 'U = share of wall clock it is busy, E = share of its dense peak. '
                 'Demand = X x servers x U.</text>')

    top = 60
    x = 30
    centres: List[Tuple[float, str]] = []
    for prov in provisions:
        colour = _engine_colour(prov.kind)
        parts.append(f'<rect class="node" x="{x}" y="{top}" width="{box_w}" height="{box_h}" '
                     f'rx="11"/>')
        parts.append(f'<rect class="node-bar" x="{x}" y="{top}" width="5" height="{box_h}" '
                     f'style="fill:{colour}"/>')
        label = (f"{prov.servers_provisioned} x {prov.unit}"
                 f"{'s' if prov.servers_provisioned != 1 else ''}")
        parts.append(f'<text class="b-title" x="{x + 16}" y="{top + 26}" style="fill:{colour}">'
                     f'{html.escape(prov.engine.upper())}</text>')
        parts.append(f'<text class="b-sub" x="{x + box_w - 14}" y="{top + 26}">'
                     f'{html.escape(label)}</text>')
        parts.append(f'<text class="b-row" x="{x + 16}" y="{top + 50}">'
                     f'peak {si(prov.peak_ops_per_s, "OP/s")} / {prov.unit} '
                     f'({html.escape(prov.peak_format)})</text>')
        for i, (key, value) in enumerate((
                ("X", f'{si(prov.throughput_ops_per_s, "OP/s")} per {prov.unit}'),
                ("U", f"{prov.utilization:.1%}"),
                ("E", f"{prov.efficiency:.1%}"))):
            yy = top + 74 + i * 20
            parts.append(f'<text class="b-k" x="{x + 16}" y="{yy}">{key}</text>')
            parts.append(f'<text class="b-v" x="{x + 40}" y="{yy}">{html.escape(value)}</text>')
        parts.append(f'<text class="b-note" x="{x + box_w - 14}" y="{top + 134}">'
                     f'{html.escape(", ".join(prov.stages))} &#183; '
                     f'{html.escape(prov.provenance)}</text>')
        centres.append((x + box_w / 2, prov.kind))
        x += box_w + gap

    for name, note in idle_blocks:
        parts.append(f'<rect class="node idle" x="{x}" y="{top + 24}" width="134" '
                     f'height="{box_h - 48}" rx="11"/>')
        parts.append(f'<text class="b-idle" x="{x + 67}" y="{top + 58}">{html.escape(name)}</text>')
        parts.append(f'<text class="b-idle-s" x="{x + 67}" y="{top + 76}">'
                     f'{html.escape(note)}</text>')
        centres.append((x + 67, "other"))
        x += 150

    # The fabric, then the memory interface.
    bus_y = top + box_h + 54
    bus_x0, bus_x1 = 30, width - 60
    parts.append(f'<rect class="bus" x="{bus_x0}" y="{bus_y}" width="{bus_x1 - bus_x0}" '
                 f'height="34" rx="8"/>')
    parts.append(f'<text class="bus-l" x="{bus_x0 + 14}" y="{bus_y + 22}">'
                 f'on-chip fabric</text>')
    supply = dossier.dram_supply_gb_per_s
    parts.append(f'<text class="bus-r" x="{bus_x1 - 14}" y="{bus_y + 22}">'
                 f'carries {dossier.dram_demand_gb_per_s:.3g} GB/s of DRAM traffic</text>')
    for cx, kind in centres:
        parts.append(f'<line class="edge" x1="{cx}" y1="{top + box_h}" x2="{cx}" '
                     f'y2="{bus_y}" marker-end="url(#arrow2)"/>')

    mem_y = bus_y + 76
    parts.append(f'<rect class="node" x="{bus_x0}" y="{mem_y}" width="{bus_x1 - bus_x0}" '
                 f'height="74" rx="11"/>')
    parts.append(f'<text class="b-title" x="{bus_x0 + 16}" y="{mem_y + 26}">DRAM</text>')
    if supply:
        used = dossier.dram_demand_gb_per_s / supply
        parts.append(f'<text class="b-row" x="{bus_x0 + 16}" y="{mem_y + 48}">'
                     f'peak {supply:g} GB/s &#183; demand {dossier.dram_demand_gb_per_s:.3g} GB/s '
                     f'&#183; {used:.2%} used</text>')
        track_x = bus_x0 + 360
        track_w = bus_x1 - track_x - 20
        parts.append(f'<rect class="track" x="{track_x}" y="{mem_y + 34}" width="{track_w}" '
                     f'height="14" rx="3"/>')
        parts.append(f'<rect class="fill" x="{track_x}" y="{mem_y + 34}" '
                     f'width="{max(2.0, used * track_w):.1f}" height="14" rx="3"/>')
    else:
        parts.append(f'<text class="n-gap" x="{bus_x0 + 16}" y="{mem_y + 48}">'
                     f'no bandwidth stated</text>')
    parts.append(f'<line class="edge" x1="{width / 2}" y1="{bus_y + 34}" x2="{width / 2}" '
                 f'y2="{mem_y}" marker-end="url(#arrow2)"/>')
    return (f'<svg viewBox="0 0 {width} {height}" class="diagram" role="img" '
            f'aria-label="the sized configuration as a block diagram">'
            f'<defs><marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-3)"/></marker></defs>'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------
# 3. The sizing, as bars
# ---------------------------------------------------------------------------

def sizing_diagram(dossier, alternatives: Sequence[dict] = ()) -> str:
    """One bar per engine: what the mission needs against what was
    provisioned, with any rejected alternative drawn to the same scale."""
    rows: List[dict] = [
        {"label": f"{p.engine.upper()}: {p.stages[0] if len(p.stages) == 1 else 'stages'}",
         "unit": p.unit, "needed": p.servers_needed, "provisioned": p.servers_provisioned,
         "kind": p.kind, "note": f"U {p.utilization:.0%} - E {p.efficiency:.1%}"
                                 f" - {p.provenance}"}
        for p in dossier.provisions]
    rows += [dict(a) for a in alternatives]
    if not rows:
        return "<p>nothing to size</p>"
    biggest = max(max(r["provisioned"], r["needed"]) for r in rows)
    scale = 560 / max(biggest, 1e-9)
    width, row_h = 980, 52
    height = 66 + len(rows) * row_h + 18
    parts = ['<text class="dtitle" x="8" y="20">What the mission needs, and what it '
             'was given</text>',
             '<text class="dsub" x="8" y="36">the pale bar is what was provisioned; '
             'the solid bar is what the demand needs</text>']
    y = 62
    for row in rows:
        colour = _engine_colour(row["kind"])
        parts.append(f'<text class="s-label" x="268" y="{y + 20}">'
                     f'{html.escape(row["label"])}</text>')
        parts.append(f'<rect class="track" x="284" y="{y + 6}" '
                     f'width="{row["provisioned"] * scale:.1f}" height="20" rx="3"/>')
        # Quoted, always: an unquoted attribute value swallows the closing
        # slash and everything after it nests inside this rect.
        faded = ' opacity="0.55"' if row.get("rejected") else ""
        parts.append(f'<rect x="284" y="{y + 6}" width="{max(2.0, row["needed"] * scale):.1f}" '
                     f'height="20" rx="3" style="fill:{colour}"{faded}/>')
        amount = (f'{row["needed"]:.2f} of {row["provisioned"]:g} '
                  f'{row["unit"]}{"s" if row["provisioned"] != 1 else ""}')
        parts.append(f'<text class="s-amount" x="{284 + max(row["provisioned"], row["needed"]) * scale + 12}" '
                     f'y="{y + 20}">{html.escape(amount)}</text>')
        parts.append(f'<text class="s-note" x="268" y="{y + 38}">'
                     f'{html.escape(row.get("note", ""))}</text>')
        y += row_h
    return (f'<svg viewBox="0 0 {width} {height}" class="diagram" role="img" '
            f'aria-label="engine sizing, needed against provisioned">'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def requirements_table(rows: Sequence[Tuple[str, str, str, str]]) -> str:
    """(requirement, figure, source, status) -- status "gap" marks a
    requirement the catalogue does not state, which is not the same as a
    requirement that is met."""
    body = "".join(
        f'<tr class="{"gap" if status == "gap" else ""}">'
        f'<td>{html.escape(name)}</td><td class="num">{html.escape(figure)}</td>'
        f'<td class="src">{html.escape(source)}</td>'
        f'<td>{"NOT STATED" if status == "gap" else html.escape(status)}</td></tr>'
        for name, figure, source, status in rows)
    return ('<table><thead><tr><th>requirement</th><th>figure</th><th>where it comes from</th>'
            f'<th>status</th></tr></thead><tbody>{body}</tbody></table>')


def demand_table(dossier) -> str:
    placements = {p.stage: p for p in dossier.placements}
    body = []
    for stage in dossier.stages:
        place = placements.get(stage.key)
        split = ", ".join(f"{share:.0%} {cls}" for cls, share in stage.class_split.items()
                          if share > 0)
        body.append(
            f'<tr><td>{html.escape(stage.tier)}</td><td><b>{html.escape(stage.key)}</b></td>'
            f'<td>{html.escape(stage.kernel_class)}</td>'
            f'<td>{html.escape(split)}</td>'
            f'<td class="num">{si(stage.rate_hz, "/s")} {html.escape(stage.unit[4:] if stage.unit.startswith("per ") else stage.unit)}</td>'
            f'<td class="num">{si(stage.ops_per_call, "OP")}</td>'
            f'<td class="num">{si(stage.bytes_per_call, "B")}</td>'
            f'<td class="num">{si(stage.ops_per_s, "OP/s")}</td>'
            f'<td class="num">{si(stage.bytes_per_s, "B/s")}</td>'
            f'<td class="num">{ms(place.seconds_per_frame) if place else "-"}</td></tr>')
    return ('<table><thead><tr><th>tier</th><th>stage</th><th>kernel class</th>'
            '<th>precision</th><th>rate</th><th>OP/call</th><th>B/call</th>'
            '<th>OP/s</th><th>B/s</th><th>latency/frame</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def fit_table(dossier) -> str:
    """Every stage against every engine: the number, or why there is none."""
    engines = sorted({e for s in dossier.stages for e in s.fits})
    body = []
    for stage in dossier.stages:
        cells = []
        for name in engines:
            fit = stage.fits[name]
            if fit.fits:
                detail = "; ".join(
                    f"{c.fmt} {c.share:.0%}: E={c.efficiency:.3g} -> {c.servers_needed:.3f}"
                    for c in fit.classes if c.servers_needed is not None)
                cells.append(f'<td class="num"><b>{fit.servers_needed:.3f}</b> '
                             f'{"tiles" if fit.kind == "kpu" else "cores"}'
                             f'<br><span class="src">{html.escape(detail)}</span></td>')
            else:
                cells.append(f'<td class="gapcell">{html.escape(fit.gap or "no fit")}</td>')
        body.append(f'<tr><td><b>{html.escape(stage.key)}</b></td>'
                    f'<td>{html.escape(stage.kernel_class)}</td>{"".join(cells)}</tr>')
    heads = "".join(f"<th>on the {html.escape(e)}</th>" for e in engines)
    return ('<table><thead><tr><th>stage</th><th>kernel class</th>'
            f'{heads}</tr></thead><tbody>{"".join(body)}</tbody></table>')


def _steps_css(steps: Dict[str, str]) -> str:
    return " ".join(f"--{name}:{value};" for name, value in steps.items())


STYLE = """
:root { color-scheme: light dark; --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b;
  --ink-2:#52514e; --ink-3:#78766f; --rule:#e2e1dc; --track:#eceae5; --soft:#f7f6f3;
  __LIGHT__ }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface:#1a1a19; --panel:#232322; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8f8e85;
  --rule:#34342f; --track:#2c2c29; --soft:#1f1f1e; __DARK__ } }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font:15.5px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:34px 18px 90px; }
h1 { font-size:29px; margin:0 0 8px; letter-spacing:-0.015em; line-height:1.2; }
h2 { font-size:21px; margin:44px 0 6px; letter-spacing:-0.01em;
  padding-top:14px; border-top:1px solid var(--rule); }
h3 { font-size:16px; margin:26px 0 6px; }
p, li { color:var(--ink-2); max-width:76ch; }
.lede { font-size:17.5px; color:var(--ink); max-width:70ch; }
.eyebrow { font-size:12px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--ink-3); margin:0 0 6px; }
.verdict { margin:22px 0 6px; padding:16px 18px; border:1px solid var(--rule);
  border-left:4px solid var(--fill); border-radius:10px; background:var(--panel);
  font-size:16px; color:var(--ink); max-width:none; }
.verdict b { font-weight:650; }
.panel { background:var(--panel); border:1px solid var(--rule); border-radius:12px;
  padding:16px 14px; margin-top:14px; overflow-x:auto; }
svg.diagram { width:100%; height:auto; display:block; min-width:900px; }
text { font-size:12px; fill:var(--ink); }
text.dtitle { font-size:13px; font-weight:650; fill:var(--ink); }
text.dsub { font-size:11px; fill:var(--ink-3); }
rect.node { fill:var(--soft); stroke:var(--rule); stroke-width:1; }
rect.node.idle { fill:none; stroke-dasharray:4 3; }
rect.source { fill:none; stroke:var(--rule); stroke-width:1; stroke-dasharray:4 3; }
rect.bus { fill:var(--track); }
rect.track { fill:var(--track); }
rect.fill { fill:var(--fill); }
text.src-t { font-size:12px; font-weight:600; text-anchor:middle; fill:var(--ink-2); }
text.src-s { font-size:10.5px; text-anchor:middle; fill:var(--ink-3); }
text.n-key { font-size:15px; font-weight:670; fill:var(--ink); }
text.n-name { font-size:11.5px; fill:var(--ink-2); }
text.n-kc { font-size:11px; fill:var(--ink-3); font-family:ui-monospace,monospace; }
text.n-tier { font-size:10px; fill:var(--ink-3); text-anchor:end; }
text.n-row { font-size:11px; fill:var(--ink-2); }
text.n-lat { font-size:12.5px; font-weight:640; }
text.n-gap { font-size:11px; fill:var(--gap); font-style:italic; }
line.edge { stroke:var(--ink-3); stroke-width:1.4; }
text.edge-l { font-size:10px; fill:var(--ink-3); text-anchor:middle; }
line.chain { stroke:var(--ink-3); stroke-width:1; stroke-dasharray:3 3; }
text.chain-l { font-size:12px; fill:var(--ink-2); }
text.chain-l .strong, tspan.strong { font-weight:680; fill:var(--ink); }
text.b-title { font-size:14px; font-weight:680; }
text.b-sub { font-size:11.5px; fill:var(--ink-2); text-anchor:end; }
text.b-row { font-size:11px; fill:var(--ink-3); }
text.b-k { font-size:12px; font-weight:700; fill:var(--ink-3); }
text.b-v { font-size:12.5px; fill:var(--ink); }
text.b-note { font-size:10px; fill:var(--ink-3); text-anchor:end; }
text.b-idle { font-size:12px; font-weight:600; text-anchor:middle; fill:var(--ink-3); }
text.b-idle-s { font-size:10px; text-anchor:middle; fill:var(--ink-3); }
text.bus-l { font-size:11.5px; font-weight:600; fill:var(--ink-2); }
text.bus-r { font-size:11px; fill:var(--ink-3); text-anchor:end; }
text.s-label { font-size:12.5px; fill:var(--ink); text-anchor:end; font-weight:600; }
text.s-note { font-size:10.5px; fill:var(--ink-3); text-anchor:end; }
text.s-amount { font-size:11.5px; fill:var(--ink-2); }
table { border-collapse:collapse; margin-top:12px; font-size:13px; width:100%; }
th, td { text-align:left; padding:7px 12px 7px 0; border-bottom:1px solid var(--rule);
  color:var(--ink-2); vertical-align:top; }
th { color:var(--ink); font-weight:650; font-size:12px; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.src, span.src { font-size:11px; color:var(--ink-3); }
tr.gap td { color:var(--gap); }
td.gapcell { color:var(--gap); font-style:italic; font-size:12px; }
.legend { display:flex; flex-wrap:wrap; gap:16px; align-items:center; font-size:12.5px;
  color:var(--ink-2); margin-top:12px; }
.key { display:inline-flex; align-items:center; gap:6px; }
.chip { width:20px; height:11px; border-radius:2px; display:inline-block; }
.note { font-size:13px; color:var(--ink-3); border-left:2px solid var(--rule);
  padding-left:12px; margin:14px 0; max-width:72ch; }
"""

STYLE = (STYLE.replace("__LIGHT__", _steps_css(LIGHT_STEPS))
              .replace("__DARK__", _steps_css(DARK_STEPS)))


def render(dossier, sections: Dict[str, str], requirements: Sequence[Tuple[str, str, str, str]],
           alternatives: Sequence[dict] = (), idle_blocks: Sequence[Tuple[str, str]] = (),
           generated: str = "") -> str:
    """The dossier as one standalone page."""
    legend = (
        '<div class="legend">'
        + "".join(f'<span class="key"><span class="chip" style="background:var(--{k})">'
                  f'</span>{html.escape(k.upper())}</span>' for k in ("cpu", "kpu"))
        + '<span class="key"><span class="chip" style="background:var(--gap)"></span>'
          'a figure nothing states</span></div>')
    confidence = dossier.estimation_confidence
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(dossier.title)} - sizing a CPU + KPU SoC</title>
<style>{STYLE}</style></head>
<body><main>
<p class="eyebrow">Mission dossier &#183; {html.escape(dossier.design)} on
{html.escape(dossier.node)}</p>
<h1>{html.escape(dossier.title)}</h1>
{sections.get("lede", "")}
{sections.get("verdict", "")}

<h2>1. The use case</h2>
{sections.get("use_case", "")}

<h2>2. Product requirements</h2>
{sections.get("requirements_intro", "")}
<div class="panel">{requirements_table(requirements)}</div>
{sections.get("requirements_note", "")}

<h2>3. The workload</h2>
{sections.get("workload", "")}
<div class="panel">{pipeline_graph(dossier)}</div>
{legend}
{sections.get("workload_note", "")}

<h2>4. What it demands</h2>
{sections.get("demand", "")}
<div class="panel">{demand_table(dossier)}</div>
{sections.get("demand_note", "")}

<h2>5. The configuration</h2>
{sections.get("configuration", "")}
<div class="panel">{block_diagram(dossier, idle_blocks)}</div>
{sections.get("configuration_note", "")}

<h2>6. The analysis: dimensioning the engines</h2>
{sections.get("analysis", "")}
<div class="panel">{fit_table(dossier)}</div>
{sections.get("analysis_2", "")}
<div class="panel">{sizing_diagram(dossier, alternatives)}</div>
{sections.get("analysis_3", "")}

<h2>7. What this rests on</h2>
{sections.get("provenance", "")}
<p class="note"><b>Confidence: {html.escape(confidence.level.value.upper())}.</b>
{html.escape(confidence.source)}</p>
<p class="note">Generated {html.escape(generated)} from
<code>{html.escape(dossier.design)}</code> at <code>{html.escape(dossier.node)}</code>.
Every figure on this page is reproducible with the command in section 7.</p>
</main></body></html>
"""


__all__ = ["CLASS_FORMATS", "DARK_STEPS", "LIGHT_STEPS", "block_diagram", "demand_table",
           "fit_table", "ms", "pipeline_graph", "render", "requirements_table", "si",
           "sizing_diagram"]
