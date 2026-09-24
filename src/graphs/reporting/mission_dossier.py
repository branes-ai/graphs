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
        ('<text class="dsub" x="8" y="34">'
         "in pipeline-tier order. The workload states each stage's own demand and no"
         "</text>"),
        ('<text class="dsub" x="8" y="48">'
         "dependency between stages, so no edges are drawn: a box's bytes are its"
         "</text>"),
        ('<text class="dsub" x="8" y="62">'
         "DRAM traffic, not a transfer to its neighbour.</text>")]

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
           cpu_label: str = "CPU", notes: Optional[Dict[str, str]] = None) -> str:
    """The dossier as one standalone page."""
    legend = (
        '<div class="legend">'
        + f'<span class="key"><span class="chip" style="background:var(--cpu)"></span>'
          f'{html.escape(cpu_label)}</span>'
        + '<span class="key"><span class="chip" style="background:var(--kpu)"></span>KPU</span>'
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

{sections.get("safety", "")}

{sections.get("comparison", "")}

{sections.get("crosscheck", "")}

<h2>What this rests on</h2>
{sections.get("provenance", "")}
<p class="note"><b>Confidence: {html.escape(confidence.level.value.upper())}.</b>
{html.escape(confidence.source)}</p>
<p class="note">Generated {html.escape(generated)} from
<code>{html.escape(dossier.design)}</code> at <code>{html.escape(dossier.node)}</code>.
Every figure on this page is reproducible with the command in section 7.</p>
</main></body></html>
"""


__all__ = ["CLASS_FORMATS", "DARK_STEPS", "LIGHT_STEPS", "block_diagram", "demand_table",
           "fit_table", "gb", "ms", "pipeline_graph", "wrap", "render", "requirements_table", "si",
           "sizing_diagram"]
