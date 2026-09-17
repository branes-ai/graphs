# -*- coding: utf-8 -*-
"""Render every compute graph used by the dataflow document."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from draw import draw_graph
from graphmodel import ARCS, BY_PROFILE, fmt_rate

CANON = 'Drone | ISR (endurance / wide-area search)'

# one representative per form factor, plus the interceptor for the ISR contrast
VARIANTS = [
 ('v_edge',  'Edge AI device | Supervisory control & command grounding',
  'Edge AI device — supervisory control',
  'Four of twelve operators. No mapping, no state estimation, no closed loop; the graph terminates at perception and results leave over a datalink.'),
 ('v_isr',   'Drone | ISR (endurance / wide-area search)',
  'Drone — endurance ISR',
  'Full graph. The transformer weight sweep dominates every other arc by two orders of magnitude.'),
 ('v_intc',  'Drone | Interceptor (terminal engagement)',
  'Drone — terminal interceptor',
  'Same airframe, no transformer. Traffic shifts to the stereo front end and the control loop.'),
 ('v_quad',  'Quadruped | ISR (dismounted, comms-denied)',
  'Quadruped — dismounted ISR',
  'The drone ISR graph with a heavier control tier: 12-DoF whole-body solve at 250 Hz.'),
 ('v_amr',   'AMR | Loading / unloading (manipulation)',
  'AMR — loading and unloading',
  'The transformer is a vision-language-action policy: one weight sweep per action chunk, not per token.'),
 ('v_hum',   'Humanoid | House work (open-world, long-horizon)',
  'Humanoid — open-world house work',
  'Both transformer modes at once: a VLM task planner and a VLA policy at 30 Hz.'),
 ('v_av',    'Autonomous vehicle | SAE L4 / L5 (high / full automation)',
  'Autonomous vehicle — SAE L4 / L5',
  'All twelve operators. Twenty-plus cameras and four LiDAR — the widest sensor ingress in the matrix.'),
]

if __name__ == '__main__':
    draw_graph('g_canonical.png', ARCS[CANON],
               'The canonical compute graph',
               'Twelve operators. Arc labels give the data structure and the size of one '
               'transfer; the reference configuration is the endurance-ISR drone. Red marks '
               'the dominant sweep internal to an operator.')
    print('g_canonical.png')
    for slug, key, title, sub in VARIANTS:
        a = ARCS[key]
        tot = sum(x['bps'] for x in a)
        draw_graph(f'{slug}.png', a, title,
                   sub + f'   Total arc traffic {fmt_rate(tot)}.',
                   label_arcs=False, weight=True, figsize=(14.6, 7.2),
                   node_fs=8.6)
        print(f'{slug}.png  {len(a)} arcs  {fmt_rate(tot)}')
