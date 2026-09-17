# -*- coding: utf-8 -*-
"""Render the compute graphs as PNG in the Branes.AI house style."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from graphmodel import (OPS, OPNAME, OPKIND, ARCS, BY_PROFILE, arcs_for,
                        active_ops, fmt_bytes, fmt_rate)

NAVY = '#0B2E4F'; RUST = '#B03A2E'; GREY = '#4A4A4A'; RULE = '#C6CED6'
BAND = '#F2F5F8'; WHITE = '#FFFFFF'; MUTE = '#9DB4CC'
plt.rcParams['font.family'] = ['Calibri', 'DejaVu Sans']

POS = {
 'CAM': (0.9, 8.5), 'LID': (0.9, 5.7), 'RAD': (0.9, 3.7), 'IMU': (0.9, 2.0),
 'RECT': (3.9, 8.5), 'DESK': (3.9, 5.7), 'FFT': (3.9, 3.7),
 'STER': (6.9, 8.5), 'TRK': (6.9, 6.5), 'CNN': (6.9, 1.1),
 'OPT': (9.6, 4.9), 'XFMR': (9.6, 1.1),
 'FUSE': (12.3, 7.9), 'DIST': (14.7, 7.9),
 'SRCH': (12.6, 2.9), 'CSOL': (16.3, 5.1), 'ACT': (18.9, 5.1),
}
NW, NH = 1.85, 0.86        # node box size

# arc routing curvature, per (src,dst)
RAD = {
 ('RECT', 'CNN'): -0.40, ('STER', 'FUSE'): -0.10, ('DESK', 'FUSE'): -0.30,
 ('IMU', 'OPT'): -0.12, ('OPT', 'CSOL'): 0.30, ('DIST', 'CSOL'): 0.14,
 ('CNN', 'SRCH'): -0.10, ('XFMR', 'SRCH'): -0.08, ('XFMR', 'CSOL'): -0.30,
 ('FFT', 'OPT'): 0.08, ('DESK', 'OPT'): -0.06, ('TRK', 'OPT'): 0.06,
 ('OPT', 'FUSE'): -0.10, ('DIST', 'SRCH'): 0.22, ('SRCH', 'CSOL'): 0.10,
 ('CNN', 'XFMR'): 0.0, ('RECT', 'TRK'): 0.0, ('RECT', 'STER'): 0.0,
 ('FUSE', 'DIST'): 0.0, ('CAM', 'RECT'): 0.0, ('LID', 'DESK'): 0.0,
 ('RAD', 'FFT'): 0.0, ('CSOL', 'ACT'): 0.0,
}
# label offset nudges, per (src,dst): (dx, dy)
NUDGE = {
 # (dx, dy, t)  t = fraction along the arc where the label sits
 ('CAM', 'RECT'): (0, 0.34, 0.5), ('LID', 'DESK'): (0, 0.34, 0.5),
 ('RAD', 'FFT'): (0, 0.34, 0.5), ('CSOL', 'ACT'): (0, 0.66, 0.5),
 ('RECT', 'STER'): (0, 0.36, 0.5), ('RECT', 'TRK'): (0.15, 0.36, 0.5),
 ('RECT', 'CNN'): (-1.55, 0.0, 0.45),
 ('STER', 'FUSE'): (0, 0.44, 0.5), ('DESK', 'OPT'): (0.10, 0.34, 0.42),
 ('DESK', 'FUSE'): (0.55, -0.46, 0.30),
 ('FFT', 'OPT'): (0.10, -0.40, 0.5), ('IMU', 'OPT'): (0.75, -0.34, 0.42),
 ('TRK', 'OPT'): (0.50, 0.24, 0.5),
 ('OPT', 'FUSE'): (-0.70, 0.10, 0.55), ('OPT', 'CSOL'): (0, -0.52, 0.5),
 ('FUSE', 'DIST'): (0, -0.62, 0.5),
 ('DIST', 'SRCH'): (1.02, 0.22, 0.42), ('DIST', 'CSOL'): (0.80, 0.40, 0.46),
 ('CNN', 'XFMR'): (0, 0.46, 0.5), ('CNN', 'SRCH'): (0.85, -0.98, 0.08),
 ('XFMR', 'SRCH'): (0.10, 0.40, 0.62), ('XFMR', 'CSOL'): (1.30, -0.30, 0.45),
 ('SRCH', 'CSOL'): (0.72, -0.14, 0.5),
}
def _edge_point(a, b, w=NW, h=NH):
    """Where the arrow leaves box a heading toward b."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return a
    sx = (w/2) / abs(dx) if dx else 1e9
    sy = (h/2) / abs(dy) if dy else 1e9
    s = min(sx, sy)
    return (ax + dx*s, ay + dy*s)


def draw_graph(path, arcs, title, subtitle, label_arcs=True, figsize=(14.6, 8.0),
               active=None, weight=False, node_fs=7.4, arc_fs=5.6):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 20.3); ax.set_ylim(-0.2, 10.9)
    ax.axis('off')

    act = active if active is not None else set(
        [a['src'] for a in arcs] + [a['dst'] for a in arcs])
    seen = set()
    ext = [a for a in arcs if not a['internal']]
    mx = max((a['bps'] for a in ext), default=1)

    # ---- arcs
    for a in ext:
        s, d = a['src'], a['dst']
        if s == d:
            continue
        p0 = _edge_point(POS[s], POS[d]); p1 = _edge_point(POS[d], POS[s])
        rad = RAD.get((s, d), 0.0)
        if weight:
            lw = 0.5 + 3.0 * (math.log10(max(a['bps'], 1e3)) - 3) / \
                 max(math.log10(mx) - 3, 1)
            lw = max(0.45, min(lw, 4.0))
        else:
            lw = 0.9
        col = RUST if (weight and a['bps'] > 0.05*mx) else NAVY
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                     arrowstyle='-|>', mutation_scale=9, lw=lw,
                     color=col, alpha=0.82, zorder=1))
        if label_arcs and (s, d) not in seen:
            seen.add((s, d))
            nx, ny, tf = NUDGE.get((s, d), (0, 0.30, 0.5))
            dxx, dyy = p1[0]-p0[0], p1[1]-p0[1]
            cx = (p0[0]+p1[0])/2 - dyy*rad; cy = (p0[1]+p1[1])/2 + dxx*rad
            u = 1 - tf
            mxp = u*u*p0[0] + 2*u*tf*cx + tf*tf*p1[0]
            myp = u*u*p0[1] + 2*u*tf*cy + tf*tf*p1[1]
            txt = f"{a['name']}\n{a['dtype']}  ·  {fmt_bytes(a['bytes_per'])}"
            ax.text(mxp+nx, myp+ny, txt, ha='center', va='center', fontsize=arc_fs,
                    color=GREY, linespacing=1.35, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='none',
                              alpha=0.86))

    # ---- nodes
    internal = {}
    for a in arcs:
        if a['internal']:
            internal[a['src']] = a
    for oid, label, kernel, tier, kind in OPS:
        if oid not in POS:
            continue
        on = oid in act
        x, y = POS[oid]
        if kind == 'source':
            fc, ec, tc = (WHITE if on else '#FBFBFC'), (MUTE if on else RULE), \
                         (NAVY if on else RULE)
            style = 'round,pad=0.02,rounding_size=0.36'
        elif kind == 'sink':
            fc, ec, tc = (BAND if on else '#FBFBFC'), (MUTE if on else RULE), \
                         (GREY if on else RULE)
            style = 'round,pad=0.02,rounding_size=0.36'
        else:
            fc, ec, tc = (NAVY if on else '#FFFFFF'), (NAVY if on else RULE), \
                         (WHITE if on else RULE)
            style = 'round,pad=0.02,rounding_size=0.10'
        ax.add_patch(FancyBboxPatch((x-NW/2, y-NH/2), NW, NH, boxstyle=style,
                     fc=fc, ec=ec, lw=1.0, zorder=4))
        ax.text(x, y+(0.10 if kind == 'op' else 0), label, ha='center', va='center',
                fontsize=node_fs, fontweight='bold', color=tc, zorder=5)
        if kind == 'op':
            ax.text(x, y-0.20, tier, ha='center', va='center', fontsize=node_fs-2.0,
                    color=(MUTE if on else RULE), zorder=5)
        if on and oid in internal and label_arcs:
            a = internal[oid]
            ax.text(x, y+NH/2+0.20, f"⟲ {a['name']}  {fmt_bytes(a['bytes_per'])}",
                    ha='center', va='center', fontsize=arc_fs-0.2, color=RUST,
                    style='italic', zorder=5)

    ax.text(-0.5, 10.80, title, fontsize=11.5, fontweight='bold', color=NAVY, va='top')
    if subtitle:
        ax.text(-0.5, 10.36, subtitle, fontsize=7.6, color=GREY, va='top', style='italic')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(path, dpi=210, facecolor='white')
    plt.close(fig)
    return path


if __name__ == '__main__':
    key = 'Drone | ISR (endurance / wide-area search)'
    draw_graph('g_canonical.png', ARCS[key],
               'The canonical compute graph',
               'Twelve operators. Arc labels give the data structure and the size of one '
               'transfer; the reference configuration is the endurance-ISR drone.')
    print('wrote g_canonical.png')
