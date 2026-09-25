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
import math
from typing import Dict, List, Optional, Sequence, Tuple

#: One hue per engine, validated all-pairs in both modes.
#: Marks and text are different jobs. A bar clears at 3:1 as a graphical
#: object; a label on it has to clear 4.5:1, and the mark hues do not.
LIGHT_STEPS: Dict[str, str] = {
    "cpu": "#256abf", "kpu": "#eb6834", "other": "#7f7e76",
    "cpu-text": "#256abf", "kpu-text": "#b8401a", "other-text": "#6e6c66",
    "fill": "#256abf", "warn": "#c93434", "gap": "#726f68",
}
DARK_STEPS: Dict[str, str] = {
    "cpu": "#3987e5", "kpu": "#d95926", "other": "#a3a199",
    "cpu-text": "#5ba0ef", "kpu-text": "#f0713c", "other-text": "#a3a199",
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


def wrap(text: str, width: int, lines: int = 2) -> List[str]:
    """Break on word boundaries, never mid-word. The last line is elided
    with a single character rather than losing half a word."""
    words, out, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            out.append(current)
        current = word
        if len(out) == lines - 1 and len(words) > len(" ".join(out).split()) + 1:
            break
    if current and len(out) < lines:
        out.append(current)
    joined = " ".join(out)
    if len(joined.split()) < len(words):
        out[-1] = out[-1][:width - 1].rstrip() + "\u2026"
    return out[:lines]


def _engine_colour(kind) -> str:
    """The mark colour for an engine."""
    name = getattr(kind, "value", kind)
    return f"var(--{name})" if name in ("cpu", "kpu") else "var(--other)"


def _engine_ink(kind) -> str:
    """The text colour for an engine, which needs 4.5:1 where the mark
    only needs 3:1."""
    name = getattr(kind, "value", kind)
    return f"var(--{name}-text)" if name in ("cpu", "kpu") else "var(--other-text)"


# ---------------------------------------------------------------------------
# 1. The workload, as a pipeline graph
# ---------------------------------------------------------------------------

NODE_W, NODE_H, NODE_GAP = 250, 150, 78


#: Widest the pipeline may run before it wraps onto another row. A
#: 2-stage pipeline reads as one line; an 18-stage one has to fold, or it
#: scales down to an illegible strip.
MAX_ROW_W = 1480


def pipeline_graph(dossier, latency_per_frame: bool = True) -> str:
    """One box per stage, in pipeline order, with what it demands and what
    it costs on the engine it was sized for.

    ``latency_per_frame`` only makes sense when the stages share a cadence.
    A pipeline whose rates span 1 Hz to 110 MHz has no common frame, so the
    caller ties this to whether a chain total is computable at all: the
    per-node latency and the chain figure appear together or not at all.
    """
    stages = list(dossier.stages)
    if not stages:
        return "<p>no stage to draw</p>"
    placements = {p.stage: p for p in dossier.placements}
    row_gap = 40
    first_x = 34                           # room for the tier label
    per_row = max(1, int((MAX_ROW_W - first_x - 130) // (NODE_W + NODE_GAP)))
    rows = [stages[i:i + per_row] for i in range(0, len(stages), per_row)]
    width = min(MAX_ROW_W,
                first_x + max(len(r) for r in rows) * (NODE_W + NODE_GAP) + 130)
    top = 86
    row_h = NODE_H + row_gap
    height = top + len(rows) * row_h + 66
    parts: List[str] = [
        f'<text class="dtitle" x="8" y="20">The pipeline: {len(stages)} stages, '
        f'{sum(1 for s in stages if s.on_reactive_chain)} on the sense-to-act chain</text>',
        # Not "no dependencies": some stage configurations do name what
        # they read -- the safety section uses one. What is missing is a
        # complete graph, which is what drawing edges would need.
        ('<text class="dsub" x="8" y="34">'
         "in pipeline-tier order. The workload gives each stage's own demand but"
         "</text>"),
        ('<text class="dsub" x="8" y="48">'
         "no complete dependency graph, so no edges are drawn: a box's bytes are"
         "</text>"),
        ('<text class="dsub" x="8" y="62">'
         "its DRAM traffic, not a transfer to its neighbour.</text>")]

    index = 0
    for row_i, row in enumerate(rows):
        y = top + row_i * row_h
        x = first_x
        # Only when the row is one tier. A row that spans tiers has each
        # box's own tier in its corner, and a band naming the first would
        # be wrong about the rest.
        tiers = {st.tier for st in row}
        if len(tiers) == 1:
            parts.append(f'<text class="tier-band" x="0" y="{y + NODE_H / 2 + 4}">'
                         f'{html.escape(row[0].tier)}</text>')
        for stage in row:
            place = placements.get(stage.key)
            kind = place.engine if place else "other"
            colour = _engine_colour(kind)
            parts.append(f'<rect class="node" x="{x}" y="{y}" width="{NODE_W}" '
                         f'height="{NODE_H}" rx="10"/>')
            parts.append(f'<rect class="node-bar" x="{x}" y="{y}" width="5" '
                         f'height="{NODE_H}" style="fill:{colour}"/>')
            parts.append(f'<text class="n-key" x="{x + 16}" y="{y + 24}">'
                         f'{html.escape(stage.key)}</text>')
            parts.append(f'<text class="n-tier" x="{x + NODE_W - 12}" y="{y + 24}">'
                         f'{html.escape(stage.tier)}'
                         f'{" &#9679;" if stage.on_reactive_chain else ""}</text>')
            for i, line in enumerate(wrap(stage.name, 33, 2)):
                parts.append(f'<text class="n-name" x="{x + 16}" y="{y + 41 + i * 13}">'
                             f'{html.escape(line)}</text>')
            parts.append(f'<text class="n-kc" x="{x + 16}" y="{y + 70}">'
                         f'{html.escape(stage.kernel_class)}</text>')
            split = ", ".join(f"{share:.0%} {cls}"
                              for cls, share in stage.class_split.items() if share > 0)
            parts.append(f'<text class="n-row" x="{x + 16}" y="{y + 88}">'
                         f'precision {html.escape(split)}</text>')
            parts.append(f'<text class="n-row" x="{x + 16}" y="{y + 105}">'
                         f'{si(stage.ops_per_s, "OP/s")} &#183; '
                         f'{si(stage.bytes_per_s, "B/s")}</text>')
            if place:
                fit = stage.fits.get(kind)
                unit = "tile" if kind == "kpu" else "core"
                need = "" if fit is None or not fit.fits else (
                    f"needs {fit.servers_needed:.3g} {unit}"
                    f"{'s' if fit.servers_needed != 1 else ''}")
                if latency_per_frame:
                    need += f" &#183; {ms(place.seconds_per_frame)}/frame"
                parts.append(f'<text class="n-lat" x="{x + 16}" y="{y + 128}" '
                             f'style="fill:{_engine_ink(kind)}">{need}</text>')
            else:
                parts.append(f'<text class="n-gap" x="{x + 16}" y="{y + 128}">'
                             f'nothing prices it on either engine</text>')
            x += NODE_W + NODE_GAP
            index += 1

    chain = dossier.chain_seconds
    if chain is not None:
        y = top + len(rows) * row_h + 4
        parts.append(f'<text class="chain-l" x="{first_x}" y="{y + 16}">'
                     f'sense to act: <tspan class="strong">{ms(chain)}</tspan> '
                     f'against a {dossier.deadline_ms:g} ms deadline'
                     f' &#183; {dossier.deadline_headroom:.1f}x headroom</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="diagram" role="img" '
            f'aria-label="the mission pipeline, one box per stage">'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-3)"/></marker></defs>'
            + "".join(parts) + "</svg>")


def gb(value: Optional[float]) -> str:
    if not value:
        return "-"
    return f"{value / 1e9:.3g} GB/s" if value >= 1e9 else f"{value / 1e6:.3g} MB/s"


def block_diagram(dossier, idle_blocks: Sequence[Tuple[str, str]] = (),
                  ingress: Optional[Tuple[str, float]] = None,
                  cpu_label: str = "CPU", notes: Optional[Dict[str, str]] = None) -> str:
    """Blocks with their compute throughput, links with their data
    throughput. Every edge carries a number."""
    provisions = list(dossier.provisions)
    if not provisions:
        return "<p>no engine to draw</p>"
    # Left to right in dataflow order, so the sensors feed the block that
    # actually ingests rather than whichever engine sorted first.
    order = {s.key: i for i, s in enumerate(dossier.stages)}
    provisions.sort(key=lambda p: min((order.get(k, 99) for k in p.stages), default=99))
    box_w, box_h, gap = 250, 40, 40
    box_h = 138
    lane = 30
    src_w = 118 if ingress else 0
    width = max(1000, lane + src_w + len(provisions) * (box_w + gap)
                + len(idle_blocks) * 136 + 260)
    bus_y, mem_y = 286, 372
    height = 476
    parts: List[str] = [
        '<text class="dtitle" x="8" y="20">Block diagram, as sized</text>',
        ('<text class="dsub" x="8" y="36">'
         "X = throughput per server &#183; U = share of wall clock &#183; "
         "E = ops-weighted mean of the per-class efficiencies</text>"),
    ]

    top = 86
    x = lane
    if ingress:
        label, rate = ingress
        parts.append(f'<rect class="source" x="{lane}" y="{top + 34}" width="{src_w - 26}" '
                     f'height="70" rx="9"/>')
        mid = lane + (src_w - 26) / 2
        parts.append(f'<text class="src-t" x="{mid}" y="{top + 62}">{html.escape(label)}</text>')
        parts.append(f'<text class="src-s" x="{mid}" y="{top + 80}">{gb(rate)}</text>')
        parts.append(f'<line class="edge" x1="{lane + src_w - 26}" y1="{top + 69}" '
                     f'x2="{lane + src_w - 7}" y2="{top + 69}" marker-end="url(#arrow2)"/>')
        x = lane + src_w
    for prov in provisions:
        name = cpu_label if prov.kind == "cpu" else prov.engine.upper()
        colour = _engine_colour(prov.kind)
        parts.append(f'<rect class="node" x="{x}" y="{top}" width="{box_w}" '
                     f'height="{box_h}" rx="11"/>')
        parts.append(f'<rect class="node-bar" x="{x}" y="{top}" width="5" height="{box_h}" '
                     f'style="fill:{colour}"/>')
        unit = f"{prov.servers_provisioned} x {prov.unit}"
        parts.append(f'<text class="b-title" x="{x + 16}" y="{top + 25}" '
                     f'style="fill:{_engine_ink(prov.kind)}">{html.escape(name)}</text>')
        parts.append(f'<text class="b-sub" x="{x + box_w - 14}" y="{top + 25}">'
                     f'{html.escape(unit)}{"s" if prov.servers_provisioned != 1 else ""}</text>')
        parts.append(f'<text class="b-row" x="{x + 16}" y="{top + 45}">'
                     f'dense peak {si(prov.peak_ops_per_s, "OP/s")} / {prov.unit} '
                     f'({html.escape(prov.peak_format)})</text>')
        for i, (key, value) in enumerate((
                ("X", f'{si(prov.throughput_ops_per_s, "OP/s")} per {prov.unit}'),
                ("U", f"{prov.utilization:.1%}"),
                ("E", f"{prov.efficiency:.1%} weighted"))):
            yy = top + 70 + i * 21
            parts.append(f'<text class="b-k" x="{x + 16}" y="{yy}">{key}</text>')
            parts.append(f'<text class="b-v" x="{x + 42}" y="{yy}">{html.escape(value)}</text>')
        footnote = (notes or {}).get(prov.engine, "")
        parts.append(f'<text class="b-note" x="{x + box_w - 14}" y="{top + 130}">'
                     f'{html.escape(", ".join(prov.stages))}'
                     f'{" &#183; " + html.escape(footnote) if footnote else ""}</text>')
        # the link down to the fabric, labelled with what it carries
        cx = x + box_w / 2
        parts.append(f'<line class="edge" x1="{cx}" y1="{top + box_h}" x2="{cx}" '
                     f'y2="{bus_y}" marker-end="url(#arrow2)"/>')
        parts.append(f'<text class="link-l" x="{cx + 8}" y="{(top + box_h + bus_y) / 2 + 4}">'
                     f'{gb(prov.bytes_per_s)}</text>')
        x += box_w + gap

    for name, note in idle_blocks:
        parts.append(f'<rect class="node idle" x="{x}" y="{top + 22}" width="120" '
                     f'height="{box_h - 44}" rx="11"/>')
        parts.append(f'<text class="b-idle" x="{x + 60}" y="{top + 58}">'
                     f'{html.escape(name)}</text>')
        parts.append(f'<text class="b-idle-s" x="{x + 60}" y="{top + 76}">'
                     f'{html.escape(note)}</text>')
        cx = x + 60
        parts.append(f'<line class="edge idle-edge" x1="{cx}" y1="{top + box_h - 22}" '
                     f'x2="{cx}" y2="{bus_y}"/>')
        parts.append(f'<text class="link-l" x="{cx + 8}" '
                     f'y="{(top + box_h + bus_y) / 2 + 4}">0</text>')
        x += 136

    bus_x1 = width - 40
    parts.append(f'<rect class="bus" x="{lane}" y="{bus_y}" width="{bus_x1 - lane}" '
                 f'height="32" rx="8"/>')
    parts.append(f'<text class="bus-l" x="{lane + 14}" y="{bus_y + 21}">on-chip fabric</text>')
    # Traffic from stages no engine carries still crosses the fabric. If
    # only the engine links were drawn, the total would not add up and the
    # diagram would imply the engines supply it.
    placed_bytes = sum(p.bytes_per_s for p in provisions)
    unplaced_bytes = max(0.0, dossier.dram_demand_gb_per_s * 1e9 - placed_bytes)
    if unplaced_bytes > 0.005 * dossier.dram_demand_gb_per_s * 1e9:
        ux = bus_x1 - 190
        parts.append(f'<rect class="node idle" x="{ux}" y="{top + 22}" width="170" '
                     f'height="{box_h - 44}" rx="11"/>')
        parts.append(f'<text class="b-idle" x="{ux + 85}" y="{top + 54}">'
                     f'unplaced stages</text>')
        parts.append(f'<text class="b-idle-s" x="{ux + 85}" y="{top + 72}">'
                     f'no engine prices them</text>')
        parts.append(f'<text class="b-idle-s" x="{ux + 85}" y="{top + 88}">'
                     f'traffic still crosses</text>')
        parts.append(f'<line class="edge idle-edge" x1="{ux + 85}" y1="{top + box_h - 22}" '
                     f'x2="{ux + 85}" y2="{bus_y}"/>')
        parts.append(f'<text class="link-l" x="{ux + 93}" '
                     f'y="{(top + box_h + bus_y) / 2 + 4}">{gb(unplaced_bytes)}</text>')
    parts.append(f'<text class="bus-r" x="{bus_x1 - 14}" y="{bus_y + 21}">'
                 f'{gb(dossier.dram_demand_gb_per_s * 1e9)} total</text>')

    supply = dossier.dram_supply_gb_per_s
    cx = (lane + bus_x1) / 2
    parts.append(f'<line class="edge" x1="{cx}" y1="{bus_y + 32}" x2="{cx}" y2="{mem_y}" '
                 f'marker-end="url(#arrow2)"/>')
    parts.append(f'<text class="link-l" x="{cx + 8}" y="{(bus_y + 32 + mem_y) / 2 + 4}">'
                 f'{gb(dossier.dram_demand_gb_per_s * 1e9)}</text>')
    parts.append(f'<rect class="node" x="{lane}" y="{mem_y}" width="{bus_x1 - lane}" '
                 f'height="72" rx="11"/>')
    parts.append(f'<text class="b-title" x="{lane + 16}" y="{mem_y + 26}">DRAM</text>')
    if supply:
        used = dossier.dram_demand_gb_per_s / supply
        parts.append(f'<text class="b-row" x="{lane + 16}" y="{mem_y + 47}">'
                     f'{supply:g} GB/s available &#183; {dossier.dram_demand_gb_per_s:.2f} GB/s '
                     f'used &#183; {used:.1%}</text>')
        track_x, track_w = lane + 330, bus_x1 - lane - 360
        parts.append(f'<rect class="track" x="{track_x}" y="{mem_y + 33}" width="{track_w}" '
                     f'height="14" rx="3"/>')
        # A mission can ask for more bandwidth than the interface has, so the
        # fill is clamped to the track; past 100% the bar turns and the
        # figure beside it carries the real number.
        over = used > 1.0
        parts.append(f'<rect class="fill{" over" if over else ""}" x="{track_x}" '
                     f'y="{mem_y + 33}" '
                     f'width="{max(2.0, min(used, 1.0) * track_w):.1f}" height="14" rx="3"/>')
    else:
        parts.append(f'<text class="n-gap" x="{lane + 16}" y="{mem_y + 47}">'
                     f'no bandwidth stated</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="diagram" role="img" '
            f'aria-label="the sized configuration, with a throughput on every link">'
            f'<defs><marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-3)"/></marker></defs>'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------
# 3. The sizing, as bars
# ---------------------------------------------------------------------------

def sizing_diagram(dossier, alternatives: Sequence[dict] = ()) -> str:
    """One bar per engine, drawn as a share of what that engine has.

    Tiles and cores are different units and must not share a scale. Every
    row is therefore ``needed / provisioned``: a full bar is exactly the
    capacity, and past that the bar clamps and turns, with the true ratio
    in the label.
    """
    rows: List[dict] = [
        {"label": f"{p.engine.upper()}: {len(p.stages)} stage"
                  f"{'s' if len(p.stages) != 1 else ''}",
         "unit": p.unit, "needed": p.servers_needed, "provisioned": p.servers_provisioned,
         "kind": p.kind,
         "note": f"U {p.utilization:.0%} - E {p.efficiency:.1%} - {p.provenance}"}
        for p in dossier.provisions]
    rows += [dict(a) for a in alternatives]
    if not rows:
        return "<p>nothing to size</p>"
    full, width, row_h = 520, 1180, 52
    height = 66 + len(rows) * row_h + 18
    title = ('<text class="dtitle" x="8" y="20">'
             "What the mission needs, against what the design has</text>")
    subtitle = ('<text class="dsub" x="8" y="36">'
                "a full bar is exactly the capacity; past it the bar clamps and the "
                "number carries the overshoot</text>")
    parts = [title, subtitle]
    y = 62
    for row in rows:
        colour = _engine_colour(row["kind"])
        ratio = (row["needed"] / row["provisioned"]) if row["provisioned"] else 0.0
        over = ratio > 1.0
        parts.append(f'<text class="s-label" x="268" y="{y + 20}">'
                     f'{html.escape(row["label"])}</text>')
        parts.append(f'<rect class="track" x="284" y="{y + 6}" width="{full}" '
                     f'height="20" rx="3"/>')
        faded = ' opacity="0.55"' if row.get("rejected") else ""
        fill = "var(--warn)" if over else colour
        parts.append(f'<rect x="284" y="{y + 6}" '
                     f'width="{max(2.0, min(ratio, 1.0) * full):.1f}" '
                     f'height="20" rx="3" style="fill:{fill}"{faded}/>')
        amount = (f'{row["needed"]:.3g} of {row["provisioned"]:g} '
                  f'{row["unit"]}{"s" if row["provisioned"] != 1 else ""}'
                  f'{f" - {ratio:.3g}x over" if over else ""}')
        parts.append(f'<text class="s-amount{" over" if over else ""}" x="{284 + full + 12}" '
                     f'y="{y + 20}">{html.escape(amount)}</text>')
        parts.append(f'<text class="s-note" x="268" y="{y + 38}">'
                     f'{html.escape(row.get("note", ""))}</text>')
        y += row_h
    return (f'<svg viewBox="0 0 {width} {height}" class="diagram" role="img" '
            f'aria-label="engine sizing, needed against what the design has">'
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


def demand_table(dossier, latency_per_frame: bool = True) -> str:
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
            + (f'<td class="num">{ms(place.seconds_per_frame) if place else "-"}</td>'
               if latency_per_frame else "")
            + '</tr>')
    return ('<table><thead><tr><th>tier</th><th>stage</th><th>kernel class</th>'
            '<th>precision</th><th>rate</th><th>OP/call</th><th>B/call</th>'
            '<th>OP/s</th><th>B/s</th>'
            + ('<th>latency/frame</th>' if latency_per_frame else '')
            + f'</tr></thead><tbody>{"".join(body)}</tbody></table>')


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


# ---------------------------------------------------------------------------
# 4. The floorplan, to scale
# ---------------------------------------------------------------------------


def mm2(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}g} mm&sup2;"


def kib(value: Optional[float]) -> str:
    """SRAM capacity a reader can check against a datasheet."""
    if not value:
        return "-"
    if value >= 1024:
        return f"{value / 1024:,.3g} MiB"
    return f"{value:,.0f} KiB"


def _squarify(items: Sequence[Tuple[str, float]], x: float, y: float,
              width: float, height: float) -> List[Tuple[str, float, float, float, float]]:
    """Squarified treemap: areas in proportion, cells as square as they go.

    ``items`` are (key, value) with values already sorted descending. The
    cells returned tile the rectangle exactly, so the drawing cannot claim
    an area the totals do not have.
    """
    total = sum(v for _, v in items if v > 0)
    if total <= 0 or width <= 0 or height <= 0:
        return []
    out: List[Tuple[str, float, float, float, float]] = []
    remaining = [(k, v) for k, v in items if v > 0]
    scale = (width * height) / total

    def worst(row: List[float], side: float) -> float:
        area = sum(row)
        if area <= 0 or side <= 0:
            return float("inf")
        big, small = max(row), min(row)
        return max(side * side * big / (area * area), area * area / (side * side * small))

    while remaining:
        side = min(width, height)
        row: List[Tuple[str, float]] = []
        while remaining:
            candidate = row + [remaining[0]]
            if row and worst([v * scale for _, v in candidate], side) > \
                    worst([v * scale for _, v in row], side):
                break
            row = candidate
            remaining.pop(0)
        row_area = sum(v for _, v in row) * scale
        thickness = row_area / side if side else 0.0
        offset = 0.0
        for key, value in row:
            length = (value * scale) / thickness if thickness else 0.0
            if width >= height:
                out.append((key, x, y + offset, thickness, length))
            else:
                out.append((key, x + offset, y, length, thickness))
            offset += length
        if width >= height:
            x += thickness
            width -= thickness
        else:
            y += thickness
            height -= thickness
    return out


#: The die drawing's edge, in pixels. Everything inside is in proportion
#: to it, so one page's die cannot be compared with another's by eye
#: unless both state the same side length -- which the caption does.
DIE_PX = 520

#: A chip for a block that could not be placed, and the pitch it repeats
#: at across and down.
CHIP_W, CHIP_PITCH, CHIP_ROW_H = 150, 162, 54


def _cell(key: str, kind: str, x: float, y: float, w: float, h: float,
          label: bool = True) -> str:
    """One floorplan cell. Whitespace is not silicon and is not drawn as
    though it were: it carries the dashed outline this page already uses
    for something that is there but is not a block."""
    if kind == "gap":
        return (f'<rect class="source" x="{x:.1f}" y="{y:.1f}" '
                f'width="{max(w - 2, 0):.1f}" height="{max(h - 2, 0):.1f}" rx="2"/>')
    colour = _engine_colour(kind)
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w - 2, 0):.1f}" '
            f'height="{max(h - 2, 0):.1f}" rx="2" fill="{colour}" fill-opacity="0.22" '
            f'stroke="{colour}" stroke-width="1.4"/>')


def floorplan_diagram(comp, cpu_label: str = "CPU") -> str:
    """The composed die at scale: pad ring, whitespace, and one cell per
    block in proportion to its area.

    A block with no area figure cannot be placed. Those are drawn beneath
    the die as outlines with no size, because a floorplan that quietly
    omitted them would read as a complete one.
    """
    placed = sorted(((b.name, b.area_mm2, b.engine_kind) for b in comp.blocks
                     if b.area_mm2 > 0), key=lambda r: -r[1])
    if not placed:
        return "<p>no block on this die carries an area figure</p>"
    kinds = {name: kind for name, _, kind in placed}
    cells = list(placed)
    if comp.whitespace_mm2 > 0:
        cells.append(("whitespace", comp.whitespace_mm2, "gap"))
        kinds["whitespace"] = "gap"
    cells.sort(key=lambda r: -r[1])

    ring_px = DIE_PX * (comp.io_ring_mm / comp.die_side_mm)
    core_px = DIE_PX - 2 * ring_px
    left, top = 8.0, 96.0
    gaps = comp.gap_blocks
    width = DIE_PX + 470
    # A chip past the right edge is clipped, which omits the block as
    # surely as not drawing it. Wrap instead, and grow for the rows.
    per_chip_row = max(1, int((width - 2 * left) // CHIP_PITCH))
    chip_rows = -(-len(gaps) // per_chip_row) if gaps else 0
    gap_row_h = (42 + CHIP_ROW_H * chip_rows) if gaps else 0
    height = top + DIE_PX + gap_row_h + 46

    parts: List[str] = [
        f'<text class="dtitle" x="8" y="22">The die at {html.escape(comp.node_id)}: '
        f'{comp.die_side_mm:.2f} mm a side, {comp.die_area_mm2:.1f} mm&sup2;</text>',
        f'<text class="dsub" x="8" y="42">Every cell is in proportion to its area. '
        f'{comp.block_area_mm2:.1f} mm&sup2; of blocks, '
        f'{comp.whitespace_mm2:.1f} mm&sup2; of placement and routing whitespace '
        f'({comp.whitespace_fraction:.0%}), and a {comp.io_ring_mm:g} mm pad ring '
        f'({comp.io_ring_area_mm2:.1f} mm&sup2;) around all of it.</text>',
        # Not "across N blocks": the blocks with no figure at all are not
        # the only blocks missing a line, and the drawing below shows only
        # the former.
        (f'<text class="dsub" x="8" y="60">Areas are a <tspan class="strong">floor</tspan>: '
         f'{len(comp.gaps)} silicon line{"s" if len(comp.gaps) != 1 else ""} '
         f'ha{"ve" if len(comp.gaps) != 1 else "s"} no figure and take'
         f'{"" if len(comp.gaps) != 1 else "s"} no space here.</text>'
         if comp.gaps else
         '<text class="dsub" x="8" y="60">Every line of silicon on this design carries a '
         'figure.</text>'),
        f'<rect class="node" x="{left:.1f}" y="{top:.1f}" '
        f'width="{DIE_PX}" height="{DIE_PX}" rx="3"/>',
        f'<rect class="node" x="{left + ring_px:.1f}" y="{top + ring_px:.1f}" '
        f'width="{core_px:.1f}" height="{core_px:.1f}" rx="2"/>',
    ]
    for key, x, y, w, h in _squarify([(k, v) for k, v, _ in cells],
                                     left + ring_px, top + ring_px, core_px, core_px):
        kind = kinds[key]
        parts.append(_cell(key, kind, x, y, w, h))
        area = next(v for k, v, _ in cells if k == key)
        label = cpu_label if key == "cpu" else key
        # Two grey blocks side by side are not identified by grey. A cell
        # wide enough for a name gets one even when it has no room for the
        # figure as well; the key beside the die carries the rest.
        if w >= 76 and h >= 34:
            parts.append(f'<text class="n-row" x="{x + 7:.1f}" y="{y + 19:.1f}">'
                         f'{html.escape(label)}</text>')
            parts.append(f'<text class="n-row" x="{x + 7:.1f}" y="{y + 35:.1f}">'
                         f'{area:.2f} mm&sup2;</text>')
        elif w >= 46 and h >= 18:
            # wrap() never breaks a single long word, and two block names
            # running into each other is worse than two clipped ones.
            room = max(3, int((w - 12) / 5.6))
            for i, line in enumerate(wrap(label, room, 2)):
                clipped = html.escape(line[:room]) + ("&#8230;" if len(line) > room else "")
                parts.append(f'<text class="n-gap" x="{x + 5:.1f}" '
                             f'y="{y + 15 + i * 12:.1f}">{clipped}</text>')

    key_x = left + DIE_PX + 34
    parts.append(f'<text class="b-title" x="{key_x:.0f}" y="{top + 16:.0f}">'
                 f'What takes the area</text>')
    row_y = top + 42
    for key, area, kind in cells:
        label = key if key != "cpu" else cpu_label
        parts.append(_cell(key, kind, key_x, row_y - 9, 18, 13, label=False))
        parts.append(f'<text class="n-row" x="{key_x + 24:.0f}" y="{row_y:.0f}">'
                     f'{html.escape(label)}</text>')
        parts.append(f'<text class="link-l" x="{key_x + 300:.0f}" y="{row_y:.0f}" '
                     f'text-anchor="end">{area:.2f} mm&sup2; '
                     f'({area / comp.die_area_mm2:.0%})</text>')
        row_y += 22
    parts.append(f'<rect x="{key_x:.0f}" y="{row_y - 9:.0f}" width="16" height="11" rx="2" '
                 f'fill="none" stroke="var(--rule)" stroke-width="1.2"/>')
    parts.append(f'<text class="n-row" x="{key_x + 24:.0f}" y="{row_y:.0f}">pad ring</text>')
    parts.append(f'<text class="link-l" x="{key_x + 300:.0f}" y="{row_y:.0f}" '
                 f'text-anchor="end">{comp.io_ring_area_mm2:.2f} mm&sup2; '
                 f'({comp.io_ring_area_mm2 / comp.die_area_mm2:.0%})</text>')

    if gaps:
        gap_y = top + DIE_PX + 34
        parts.append(f'<text class="b-title" x="8" y="{gap_y:.0f}">'
                     f'Placed nowhere, because nothing states a size</text>')
        for i, block in enumerate(gaps):
            chip_x = left + (i % per_chip_row) * CHIP_PITCH
            chip_y = gap_y + 14 + (i // per_chip_row) * CHIP_ROW_H
            parts.append(f'<rect class="source chip" x="{chip_x:.0f}" y="{chip_y:.0f}" '
                         f'width="{CHIP_W:.0f}" height="40" rx="3"/>')
            parts.append(f'<text class="n-gap" x="{chip_x + 10:.0f}" y="{chip_y + 19:.0f}">'
                         f'{html.escape(block.name)}</text>')
            parts.append(f'<text class="n-gap" x="{chip_x + 10:.0f}" y="{chip_y + 34:.0f}">'
                         f'{len(block.gaps)} line'
                         f'{"s" if len(block.gaps) != 1 else ""}, no figure</text>')
    return (f'<svg class="diagram" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'role="img" aria-label="Die floorplan to scale">{"".join(parts)}</svg>')


def silicon_table(comp, cpu_label: str = "CPU") -> str:
    """Block by block: area, what it is built from, and what is missing."""
    body = []
    for block in sorted(comp.blocks, key=lambda b: -b.area_mm2):
        label = cpu_label if block.name == "cpu" else block.name
        missing = ", ".join(ln.name for ln in block.gaps)
        # A block nothing prices is unknown, not zero. Printing 0.000 mm2
        # in a column of real areas reads as a figure.
        blank = block.area_mm2 == 0.0
        body.append(
            f'<tr class="{"gap" if blank else ""}">'
            f'<td><b>{html.escape(label)}</b>'
            f'{f" &times;{block.count}" if block.count != 1 else ""}'
            f'<br><span class="src">{html.escape(block.ip)}</span></td>'
            f'<td class="num">{"-" if blank else f"{block.area_mm2:.3f}"}</td>'
            f'<td class="num">'
            f'{"-" if blank else f"{block.area_mm2 / comp.die_area_mm2:.1%}"}</td>'
            f'<td class="num">{"-" if blank else f"{block.transistors_mtx:,.1f}"}</td>'
            f'<td class="num">{f"{block.gates_m:,.1f}" if block.gates_m else "-"}</td>'
            f'<td class="num">{kib(block.sram_kib)}</td>'
            f'<td class="gapcell">{html.escape(missing) if missing else ""}</td></tr>')
    body.append(
        f'<tr><td><b>all blocks</b></td>'
        f'<td class="num"><b>{comp.block_area_mm2:.3f}</b></td>'
        f'<td class="num">{comp.block_area_mm2 / comp.die_area_mm2:.1%}</td>'
        f'<td class="num"><b>{comp.transistors_mtx:,.1f}</b></td>'
        f'<td class="num"><b>{comp.gates_m:,.1f}</b></td>'
        f'<td class="num"><b>{kib(comp.sram_kib)}</b></td>'
        f'<td class="gapcell">'
        f'{f"{len(comp.gaps)} lines" if comp.gaps else ""}</td></tr>')
    body.append(
        f'<tr><td>whitespace and pad ring</td>'
        f'<td class="num">{comp.whitespace_mm2 + comp.io_ring_area_mm2:.3f}</td>'
        f'<td class="num">'
        f'{(comp.whitespace_mm2 + comp.io_ring_area_mm2) / comp.die_area_mm2:.1%}</td>'
        f'<td class="num">-</td><td class="num">-</td><td class="num">-</td>'
        f'<td class="src">layout record, not silicon</td></tr>')
    body.append(
        f'<tr><td><b>die</b></td><td class="num"><b>{comp.die_area_mm2:.3f}</b></td>'
        f'<td class="num">100%</td><td class="num">-</td><td class="num">-</td>'
        f'<td class="num">-</td>'
        f'<td class="src">{"a floor" if comp.gaps else ""}</td></tr>')
    return ('<table><thead><tr><th>block</th><th>area (mm&sup2;)</th><th>of die</th>'
            '<th>transistors (Mtx)</th><th>gates (M, NAND2)</th><th>SRAM</th>'
            f'<th>no figure for</th></tr></thead><tbody>{"".join(body)}</tbody></table>')


def density_table(comp) -> str:
    """One row per library: the density every area on the die was divided
    by, and where that density comes from."""
    body = []
    for entry in comp.classes:
        holds = kib(entry.sram_kib) if entry.sram_kib else (
            f"{entry.gates_m:,.1f} M gates" if entry.gates_m else "-")
        body.append(
            f'<tr><td><b>{html.escape(entry.circuit_class.value)}</b></td>'
            f'<td>{html.escape(entry.library)}</td>'
            f'<td class="num">{entry.mtx_per_mm2:,.0f}</td>'
            f'<td class="num">{entry.transistors_mtx:,.1f}</td>'
            f'<td class="num">{entry.area_mm2:.3f}</td>'
            f'<td class="num">{entry.area_mm2 / comp.block_area_mm2:.1%}</td>'
            f'<td class="num">{holds}</td>'
            f'<td class="src">{html.escape(entry.density_source)}</td></tr>')
    return ('<table><thead><tr><th>circuit class</th><th>library</th>'
            '<th>Mtx/mm&sup2;</th><th>transistors (Mtx)</th><th>area (mm&sup2;)</th>'
            '<th>of blocks</th><th>what it holds</th><th>density from</th>'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table>')


def scaling_table(law, tiles: float, cores: float, cpu_label: str = "CPU") -> str:
    """The three terms of the die, and what the sized counts make of them.

    ``tiles`` and ``cores`` are what the mission was sized for, so the
    last column is a die this catalogue does not contain.
    """
    fabric = law.fabric_mm2(tiles)
    cpu = law.cpu_mm2(cores)
    rows = [
        ("KPU fabric", f"{law.fabric_fixed_mm2:.3f} mm&sup2; + "
                       f"{law.per_tile_mm2:.4f} mm&sup2; &times; tiles",
         f"{tiles:,.0f} tile{'s' if tiles != 1 else ''}", fabric),
        (f"{cpu_label} cores", f"{law.per_core_mm2:.4f} mm&sup2; &times; cores "
                               f"(caches only)",
         f"{cores:,.0f} core{'s' if cores != 1 else ''}", cpu),
        ("everything else", "fixed: it does not move with either",
         "-", law.other_fixed_mm2),
    ]
    body = "".join(
        f'<tr><td><b>{html.escape(name)}</b></td><td class="src">{how}</td>'
        f'<td class="num">{html.escape(count)}</td>'
        f'<td class="num">{area:.3f}</td></tr>'
        for name, how, count, area in rows)
    block = law.block_area_mm2(tiles, cores)
    die = law.die_area_mm2(tiles, cores)
    body += (f'<tr><td><b>blocks</b></td>'
             f'<td class="src">the three terms above</td><td class="num">-</td>'
             f'<td class="num"><b>{block:.3f}</b></td></tr>')
    body += (f'<tr><td><b>die</b></td>'
             f'<td class="src">+{law.whitespace_fraction:.0%} whitespace, '
             f'+{law.io_ring_mm:g} mm pad ring a side</td>'
             f'<td class="num">{math.sqrt(block * (1 + law.whitespace_fraction)):.2f} mm '
             f'core side</td>'
             f'<td class="num"><b>{die:.3f}</b></td></tr>')
    return ('<table><thead><tr><th>term</th><th>how it scales</th><th>sized for</th>'
            f'<th>area (mm&sup2;)</th></tr></thead><tbody>{body}</tbody></table>')


def _steps_css(steps: Dict[str, str]) -> str:
    return " ".join(f"--{name}:{value};" for name, value in steps.items())


STYLE = """
:root { color-scheme: light dark; --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b;
  --ink-2:#52514e; --ink-3:#6e6c66; --rule:#e2e1dc; --track:#eceae5; --soft:#f7f6f3;
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
rect.fill.over { fill:var(--warn); }
text.src-t { font-size:12px; font-weight:600; text-anchor:middle; fill:var(--ink-2); }
text.tier-band { font-size:11px; font-weight:650; fill:var(--ink-3); }
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
text.link-l { font-size:11px; fill:var(--ink-2); font-variant-numeric:tabular-nums; }
text.ingress { font-size:11px; fill:var(--ink-3); }
line.idle-edge { stroke-dasharray:3 3; opacity:.5; }
text.s-label { font-size:12.5px; fill:var(--ink); text-anchor:end; font-weight:600; }
text.s-note { font-size:10.5px; fill:var(--ink-3); text-anchor:end; }
text.s-amount { font-size:11.5px; fill:var(--ink-2); }
text.s-amount.over { fill:var(--warn); font-weight:650; }
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
           generated: str = "", ingress: Optional[Tuple[str, float]] = None,
           cpu_label: str = "CPU", notes: Optional[Dict[str, str]] = None,
           composition=None) -> str:
    """The dossier as one standalone page."""
    legend = (
        '<div class="legend">'
        + f'<span class="key"><span class="chip" style="background:var(--cpu)"></span>'
          f'{html.escape(cpu_label)}</span>'
        + '<span class="key"><span class="chip" style="background:var(--kpu)"></span>KPU</span>'
        + '<span class="key"><span class="chip" style="background:var(--gap)"></span>'
          'a figure nothing states</span></div>')
    confidence = dossier.estimation_confidence
    floorplan = "" if composition is None else f"""
<h2>7. The floorplan: where the area goes</h2>
{sections.get("floorplan", "")}
<div class="panel">{floorplan_diagram(composition, cpu_label)}</div>
{sections.get("floorplan_note", "")}
<h3>Block by block</h3>
<div class="panel">{silicon_table(composition, cpu_label)}</div>
{sections.get("silicon_note", "")}
<h3>What the transistors are</h3>
<div class="panel">{density_table(composition)}</div>
{sections.get("density_note", "")}
{sections.get("scaling", "")}
{sections.get("costing", "")}"""
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
<div class="panel">{pipeline_graph(dossier, dossier.chain_seconds is not None)}</div>
{legend}
{sections.get("workload_note", "")}

<h2>4. What it demands</h2>
{sections.get("demand", "")}
<div class="panel">{demand_table(dossier, dossier.chain_seconds is not None)}</div>
{sections.get("demand_note", "")}

<h2>5. The configuration</h2>
{sections.get("configuration", "")}
<div class="panel">{block_diagram(dossier, idle_blocks, ingress, cpu_label, notes)}</div>
{sections.get("configuration_note", "")}

<h2>6. The analysis: dimensioning the engines</h2>
{sections.get("analysis", "")}
<div class="panel">{fit_table(dossier)}</div>
{sections.get("analysis_2", "")}
<div class="panel">{sizing_diagram(dossier, alternatives)}</div>
{sections.get("analysis_3", "")}
{floorplan}

{sections.get("safety", "")}

{sections.get("comparison", "")}

{sections.get("crosscheck", "")}

<h2>What this rests on</h2>
{sections.get("provenance", "")}
<p class="note"><b>Confidence: {html.escape(confidence.level.value.upper())}.</b>
{html.escape(confidence.source)}</p>
<p class="note">Generated {html.escape(generated)} from
<code>{html.escape(dossier.design)}</code> at <code>{html.escape(dossier.node)}</code>.
Every figure on this page is reproducible with the command above.</p>
</main></body></html>
"""


__all__ = ["CLASS_FORMATS", "DARK_STEPS", "DIE_PX", "LIGHT_STEPS", "block_diagram",
           "demand_table", "density_table", "fit_table", "floorplan_diagram", "gb", "kib",
           "mm2", "ms", "pipeline_graph", "wrap", "render", "requirements_table",
           "scaling_table", "si", "silicon_table", "sizing_diagram"]
