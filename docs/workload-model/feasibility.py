# -*- coding: utf-8 -*-
"""
Five-architecture feasibility assessment.

For each architecture, an operator either executes on the accelerator or falls
back to the host CPU cluster. Power is the demanded operation rate divided by
the achieved efficiency of whatever executes it, plus memory energy.
"""
from profiles import PROFILES
from pipeline import CLS
from tiles import ROWS as TILE_ROWS, E_DRAM, E_DRAM_K, E_SRAM, TILES

CPU = dict(A=(50.0, 5.0), B=(30.0, 4.0), C=(15.0, 2.0))    # (GOP/s, GOPS/W)

ARCH = {
 'CPU only': dict(
    caps=dict(A=CPU['A'], B=CPU['B'], C=CPU['C']),
    note='Arm cluster with SIMD. No tensor path at all.'),
 'DLA': dict(
    caps=dict(A=(1000.0, 150.0), B=CPU['B'], C=CPU['C']),
    note='Fixed-function INT8 CNN engine — nvDLA, Andes anDLA, Hailo class. '
         'Everything outside the CNN graph falls to the host.'),
 'TPU': dict(
    caps=dict(A=(2500.0, 200.0), B=(200.0, 30.0), C=CPU['C']),
    note='Systolic INT8/bf16 array. Reaches part of the FP16 tier; no FP64 path.'),
 'GPU': dict(
    caps=dict(A=(2000.0, 101.8), B=(300.0, 20.0), C=CPU['C']),
    note='SIMT with tensor cores. Serves Class A and B; FP64 falls to the CPU cluster.'),
}


def assess(p):
    out = {}
    tot_bytes = sum(s['bytes_s'] for s in p.stages.values())
    for name, spec in ARCH.items():
        w = 0.0; t = 0.0
        for k, s in p.stages.items():
            a, b, c = CLS[k]
            ops = s['ops_s'] / 1e9
            for frac, cls in ((a, 'A'), (b, 'B'), (c, 'C')):
                if frac <= 0:
                    continue
                thr, eff = spec['caps'][cls]
                t += ops*frac/thr
                w += ops*frac/eff
        w += tot_bytes * E_DRAM
        out[name] = dict(occ=t, w=w)
    # KPU comes from the tile model
    tr = next(r for r in TILE_ROWS if r['ff'] == p.ff and r['name'] == p.name)
    out['KPU'] = dict(occ=tr['kpu_occ'], w=tr['kpu_w'])
    return out


ROWS = []
for p in PROFILES:
    a = assess(p)
    ROWS.append(dict(ff=p.ff, name=p.name, budget=p.budget_w, arch=a))

ORDER = ['CPU only', 'DLA', 'TPU', 'GPU', 'KPU']


def verdict(v, budget):
    """Feasible if it fits the platform's compute allocation."""
    return v['w'] <= budget


if __name__ == '__main__':
    print(f"{'mission':40}{'budget':>7}" + ''.join(f'{a:>13}' for a in ORDER))
    for r in ROWS:
        cells = ''.join(f"{r['arch'][a]['w']:10,.0f} W " for a in ORDER)
        print(f"{(r['ff'][:10]+' | '+r['name'][:25]):40}{r['budget']:6,.0f}W{cells}")
    print()
    for a in ORDER:
        n = sum(1 for r in ROWS if verdict(r['arch'][a], r['budget']))
        print(f'  {a:10} feasible on {n:2} of 18 profiles')
    print('\n--- L2 / L3 / L4 detail (watts)')
    for r in ROWS:
        if r['ff'] != 'Autonomous vehicle':
            continue
        print(f"  {r['name'][:34]:36} budget {r['budget']:4,.0f} W  " +
              '  '.join(f"{a} {r['arch'][a]['w']:,.0f}" for a in ORDER))
