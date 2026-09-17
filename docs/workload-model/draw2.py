# -*- coding: utf-8 -*-
"""Page-sized compute graphs: two canonical halves plus the mission variants."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from graphmodel import OPS, ARCS, fmt_bytes, fmt_rate

NAVY = '#0B2E4F'; RUST = '#B03A2E'; GREY = '#4A4A4A'; RULE = '#C6CED6'
BAND = '#F2F5F8'; WHITE = '#FFFFFF'; MUTE = '#9DB4CC'
plt.rcParams['font.family'] = ['Calibri', 'DejaVu Sans']
KIND = {o[0]: o[4] for o in OPS}
LABEL = {o[0]: o[1] for o in OPS}
TIER = {o[0]: o[3] for o in OPS}


def render(path, arcs, pos, rad, nudge, xlim, ylim, figsize, title, subtitle,
           label_arcs=True, weight=False, active=None, nw=1.95, nh=0.84,
           node_fs=8.2, arc_fs=6.1, ghost=()):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
    keep = set(pos)
    arcs = [a for a in arcs if a['src'] in keep and a['dst'] in keep]
    act = active if active is not None else set(
        [a['src'] for a in arcs] + [a['dst'] for a in arcs])
    ext = [a for a in arcs if not a['internal']]
    mx = max((a['bps'] for a in ext), default=1)

    def edge(a, b):
        (ax_, ay), (bx, by) = a, b
        dx, dy = bx-ax_, by-ay
        if dx == 0 and dy == 0:
            return a
        s = min((nw/2)/abs(dx) if dx else 1e9, (nh/2)/abs(dy) if dy else 1e9)
        return (ax_+dx*s, ay+dy*s)

    seen = set()
    for a in ext:
        s, d = a['src'], a['dst']
        if s == d:
            continue
        p0, p1 = edge(pos[s], pos[d]), edge(pos[d], pos[s])
        r = rad.get((s, d), 0.0)
        if weight:
            lw = 0.5 + 3.2*(math.log10(max(a['bps'], 1e3))-3)/max(math.log10(mx)-3, 1)
            lw = max(0.5, min(lw, 4.2))
        else:
            lw = 1.0
        col = RUST if (weight and a['bps'] > 0.05*mx) else NAVY
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={r}',
                     arrowstyle='-|>', mutation_scale=10, lw=lw, color=col,
                     alpha=0.85, zorder=1))
        if label_arcs and (s, d) not in seen:
            seen.add((s, d))
            nx, ny, tf = nudge.get((s, d), (0, 0.34, 0.5))
            dxx, dyy = p1[0]-p0[0], p1[1]-p0[1]
            cx = (p0[0]+p1[0])/2 - dyy*r; cy = (p0[1]+p1[1])/2 + dxx*r
            u = 1-tf
            mx_ = u*u*p0[0] + 2*u*tf*cx + tf*tf*p1[0]
            my_ = u*u*p0[1] + 2*u*tf*cy + tf*tf*p1[1]
            ax.text(mx_+nx, my_+ny,
                    f"{a['name']}\n{a['dtype']}  ·  {fmt_bytes(a['bytes_per'])}",
                    ha='center', va='center', fontsize=arc_fs, color=GREY,
                    linespacing=1.3, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.9))

    internal = {a['src']: a for a in arcs if a['internal']}
    for oid in pos:
        on = oid in act
        gh = oid in ghost
        x, y = pos[oid]
        k = KIND[oid]
        if gh:
            fc, ec, tc, style = WHITE, MUTE, MUTE, 'round,pad=0.02,rounding_size=0.10'
        elif k == 'source':
            fc, ec, tc = (WHITE if on else '#FCFCFD'), (MUTE if on else RULE), \
                         (NAVY if on else RULE)
            style = 'round,pad=0.02,rounding_size=0.34'
        elif k == 'sink':
            fc, ec, tc = (BAND if on else '#FCFCFD'), (MUTE if on else RULE), \
                         (GREY if on else RULE)
            style = 'round,pad=0.02,rounding_size=0.34'
        else:
            fc, ec, tc = (NAVY if on else WHITE), (NAVY if on else RULE), \
                         (WHITE if on else RULE)
            style = 'round,pad=0.02,rounding_size=0.10'
        ax.add_patch(FancyBboxPatch((x-nw/2, y-nh/2), nw, nh, boxstyle=style,
                     fc=fc, ec=ec, lw=1.0, ls=('--' if gh else '-'), zorder=4))
        ax.text(x, y+(0.10 if k == 'op' else 0), LABEL[oid], ha='center', va='center',
                fontsize=node_fs, fontweight='bold', color=tc, zorder=5)
        if k == 'op':
            ax.text(x, y-0.20, TIER[oid], ha='center', va='center', fontsize=node_fs-2.2,
                    color=(MUTE if (on and not gh) else RULE), zorder=5)
        if on and not gh and oid in internal and label_arcs:
            a = internal[oid]
            ax.text(x, y+nh/2+0.21, f"⟲ {a['name']}  {fmt_bytes(a['bytes_per'])}",
                    ha='center', va='center', fontsize=arc_fs-0.3, color=RUST,
                    style='italic', zorder=6)

    ax.text(xlim[0], ylim[1]-0.06, title, fontsize=10.6, fontweight='bold',
            color=NAVY, va='top')
    if subtitle:
        ax.text(xlim[0], ylim[1]-0.52, subtitle, fontsize=6.9, color=GREY, va='top',
                style='italic')
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


# ================================================================ figure A
A_POS = {
 'CAM': (1.15, 6.55), 'LID': (1.15, 4.15), 'RAD': (1.15, 2.15), 'IMU': (1.15, 0.60),
 'RECT': (4.30, 6.55), 'DESK': (4.30, 4.15), 'FFT': (4.30, 2.15),
 'STER': (7.45, 7.45), 'TRK': (7.45, 5.30), 'OPT': (7.85, 2.15),
 'FUSE': (10.75, 6.55), 'DIST': (10.75, 4.15),
}
A_RAD = {('STER', 'FUSE'): -0.10, ('DESK', 'FUSE'): -0.34, ('IMU', 'OPT'): -0.10,
         ('OPT', 'FUSE'): -0.16, ('DESK', 'OPT'): -0.06, ('TRK', 'OPT'): 0.06,
         ('FFT', 'OPT'): 0.06}
A_NUD = {
 ('CAM', 'RECT'): (0, 0.40, 0.5), ('LID', 'DESK'): (0, 0.40, 0.5),
 ('RAD', 'FFT'): (0, 0.40, 0.5), ('RECT', 'STER'): (0.05, 0.42, 0.5),
 ('RECT', 'TRK'): (0.15, 0.40, 0.5), ('STER', 'FUSE'): (0, 0.46, 0.5),
 ('DESK', 'OPT'): (0.05, 0.40, 0.42), ('DESK', 'FUSE'): (-0.15, -0.56, 0.24),
 ('FFT', 'OPT'): (0.05, -0.44, 0.5), ('IMU', 'OPT'): (0.30, 0.38, 0.55),
 ('TRK', 'OPT'): (0.68, 0.30, 0.55), ('OPT', 'FUSE'): (-0.88, -0.22, 0.42),
 ('FUSE', 'DIST'): (-1.32, 0.0, 0.5),
}

# ================================================================ figure B
B_POS = {
 'RECT': (1.30, 5.60), 'OPT': (1.30, 3.30), 'DIST': (1.30, 1.15),
 'CNN': (4.65, 5.60), 'XFMR': (7.95, 5.60),
 'SRCH': (7.95, 2.55), 'CSOL': (11.15, 3.30), 'ACT': (11.15, 0.95),
}
B_RAD = {('CNN', 'SRCH'): -0.10, ('XFMR', 'SRCH'): 0.0, ('OPT', 'CSOL'): 0.10,
         ('DIST', 'CSOL'): -0.16, ('XFMR', 'CSOL'): 0.14, ('CNN', 'CSOL'): 0.20}
B_NUD = {
 ('RECT', 'CNN'): (0, 0.42, 0.5), ('CNN', 'XFMR'): (0, 0.42, 0.5),
 ('CNN', 'SRCH'): (-0.55, -0.30, 0.5), ('XFMR', 'SRCH'): (0.86, 0.0, 0.5),
 ('XFMR', 'CSOL'): (0.30, 0.46, 0.5), ('SRCH', 'CSOL'): (0.10, -0.44, 0.5),
 ('OPT', 'CSOL'): (0, -0.46, 0.55), ('DIST', 'CSOL'): (0.30, -0.44, 0.55),
 ('CSOL', 'ACT'): (0.80, 0.10, 0.5), ('CNN', 'CSOL'): (0.4, -0.5, 0.5),
}

CANON = 'Drone | ISR (endurance / wide-area search)'

if __name__ == '__main__':
    a = ARCS[CANON]
    render('fig_a.png', a, A_POS, A_RAD, A_NUD, (0.0, 11.95), (-0.1, 8.55),
           (7.35, 5.28),
           'Compute graph, part 1 — sensing, state estimation and mapping',
           'Four sensors and eight operators. Arc labels give the data structure and the '
           'size of one transfer; red marks the dominant sweep internal to an operator.')
    render('fig_b.png', a, B_POS, B_RAD, B_NUD, (0.0, 12.45), (-0.1, 7.05),
           (7.35, 4.20),
           'Compute graph, part 2 — semantic perception, planning and control',
           'Five operators plus the actuator sink. The three dashed boxes on the left are '
           'upstream operators from part 1, repeated for continuity.',
           ghost=('RECT', 'OPT', 'DIST'))
    print('fig_a.png fig_b.png')
