# -*- coding: utf-8 -*-
"""
Mission pipeline model, built on forward-derived unit costs (derive.py).

Every mission is an explicit sensor configuration plus a set of stage update
rates. Stage cost = unit cost at that configuration x its rate. Nothing is
back-solved from a published total.
"""
from math import log2
from derive import BY_KEY, D as DERIVS

# effective throughput by precision class, GOP/s (companion annex, Section 8)
EFF = dict(A=2000.0, B=300.0, C=15.0)
CLS = {r['key']: r['cls'] for r in DERIVS}
TIER = {r['key']: r['tier'] for r in DERIVS}
NAME = {r['key']: r['name'] for r in DERIVS}
UOPS = {r['key']: r['ops'] for r in DERIVS}
UBYT = {r['key']: r['bytes'] for r in DERIVS}

# ---------------------------------------------------------------- parametric unit costs
def sgm_ops(w, h, d):
    px = w * h
    return px * 140 + px * d * 31

def sgm_bytes(w, h, d):
    px = w * h
    return px * 22 + px * d * 8

def radar_ops(rx, ns, nc):
    return (rx*ns*nc*8 + int(rx*nc*(ns/2*log2(ns)*6)) + int(rx*ns*(nc/2*log2(nc)*6))
            + ns*nc*rx*40 + ns*nc*rx*10)

def radar_bytes(rx, ns, nc):
    return rx*ns*nc*4 + 4*rx*ns*nc*8 + ns*nc*4 + 16384

def det_ops(gmac):   return 2*gmac*1e9 + 0.34e9
def det_bytes(mpar): return mpar*1e6 * 7 + 2.5e6

def vlm_ops(par_b, prefill, decode, vis_gop):
    return 2*par_b*1e9*(prefill+decode) + vis_gop*1e9

def vlm_bytes(par_b, decode, ctx):
    return par_b*1e9*(1+decode) + ctx*2*32*2048*2

def vla_ops(par_b, tokens, vis_gop):
    return 2*par_b*1e9*tokens + vis_gop*1e9

def vla_bytes(par_b, tokens):
    return par_b*1e9 + tokens*2*36*2048*2 + 2*448*448*3*2

def mpc_ops(nx, nu, N, iters=5):
    core = (N*(nx*nx*6 + nx*nu*6 + 200)
            + N*(nx**3 + nx**2*nu + nx*nu**2)*2
            + N*(nx**2*nu)*2)
    return iters * core

def mpc_bytes(nx, nu, N, iters=5):
    return N*(nx+nu)*8 + iters*N*nx*nx*8 + N*nu*8

def cbf_ops(nhaz, win):
    return win**3*6 + nhaz*120 + nhaz*90 + 40*3*nhaz*4 + 600

def cbf_bytes(nhaz, win):
    return int(win**3*4*0.1 + win*win*64*0.5) + 64 + nhaz*32

# ---------------------------------------------------------------- mission spec
class Mission:
    def __init__(self, ff, name, budget_w, note, deadline_ms, **k):
        self.ff, self.name, self.budget_w, self.note = ff, name, budget_w, note
        self.deadline_ms = deadline_ms
        self.k = k
        self.stages = {}     # key -> dict(rate, uops, ubytes, ops_s, bytes_s)
        self._build()

    def add(self, key, rate, uops, ubytes):
        if rate <= 0 or uops <= 0:
            return
        self.stages[key] = dict(rate=rate, uops=uops, ubytes=ubytes,
                                ops_s=uops*rate, bytes_s=ubytes*rate)

    def _build(self):
        k = self.k
        # --- T1
        sp = k.get('stereo')                      # (n, w, h, fps, disp)
        if sp:
            n, w, h, fps, d = sp
            self.add('sgm', n*fps, sgm_ops(w, h, d), sgm_bytes(w, h, d))
        mo = k.get('mono')                        # (n, w, h, fps)
        if mo:
            n, w, h, fps = mo
            self.add('mono', n*w*h*fps, UOPS['mono'], UBYT['mono'])
        rd = k.get('radar')                       # (n, hz, rx, ns, nc)
        if rd:
            n, hz, rx, ns, nc = rd
            self.add('radar', n*hz, radar_ops(rx, ns, nc), radar_bytes(rx, ns, nc))
        lp = k.get('lidar_pts_s', 0)
        self.add('lidar', lp, UOPS['lidar'], UBYT['lidar'])
        # --- T2
        self.add('lio', lp if k.get('lio', True) else 0, UOPS['lio'], UBYT['lio'])
        vc = k.get('vio')                         # (n, w, h, fps)
        if vc:
            n, w, h, fps = vc
            self.add('vio', n*w*h*fps, UOPS['vio'], UBYT['vio'])
        self.add('ba', k.get('ba_hz', 0), UOPS['ba'], UBYT['ba'])
        # --- T3
        self.add('tsdf', k.get('tsdf_pts_s', 0), UOPS['tsdf'], UBYT['tsdf'])
        self.add('esdf', k.get('esdf_hz', 0), UOPS['esdf'], UBYT['esdf'])
        # --- T4
        if k.get('det_hz'):
            g = k.get('det_gmac', 45); mp = k.get('det_mpar', 24)
            self.add('det', k['det_hz'], det_ops(g), det_bytes(mp))
        if k.get('vlm_qps'):
            p = k.get('vlm_par', 2); pre = k.get('vlm_prefill', 600)
            dec = k.get('vlm_decode', 64); vg = k.get('vlm_vis_gop', 160)
            self.add('vlm', k['vlm_qps'], vlm_ops(p, pre, dec, vg),
                     vlm_bytes(p, dec, pre+dec))
        if k.get('vla_hz'):
            p = k.get('vla_par', 3); tk = k.get('vla_tokens', 56)
            self.add('vla', k['vla_hz'], vla_ops(p, tk, 42), vla_bytes(p, tk))
        # --- T5
        self.add('gain', k.get('gain_evals_s', 0), UOPS['gain'], UBYT['gain'])
        self.add('graph', k.get('replan_hz', 0), UOPS['graph'], UBYT['graph'])
        # --- T6
        self.add('sdfenc', k.get('sdfenc_hz', 0), UOPS['sdfenc'], UBYT['sdfenc'])
        self.add('policy', k.get('policy_hz', 0), UOPS['policy'], UBYT['policy'])
        if k.get('mpc_hz'):
            nx, nu, N = k.get('mpc_dims', (12, 4, 30))
            self.add('mpc', k['mpc_hz'], mpc_ops(nx, nu, N), mpc_bytes(nx, nu, N))
        if k.get('cbf_hz'):
            nh = k.get('cbf_haz', 12); wn = k.get('cbf_win', 15)
            self.add('cbf', k['cbf_hz'], cbf_ops(nh, wn), cbf_bytes(nh, wn))
        # --- T7
        dof = k.get('dof', 4)
        self.add('ctrl', k.get('ctrl_hz', 0)*dof, UOPS['ctrl'], UBYT['ctrl'])

    # ---------------------------------------------------------- roll-ups
    def service_time(self, key):
        """Seconds to execute one call of this stage, alone on the machine."""
        a, b, c = CLS[key]
        o = self.stages[key]['uops'] / 1e9        # GOP
        return o*a/EFF['A'] + o*b/EFF['B'] + o*c/EFF['C']

    def summary(self):
        tot = sum(s['ops_s'] for s in self.stages.values())
        byt = sum(s['bytes_s'] for s in self.stages.values())
        A = sum(s['ops_s']*CLS[k][0] for k, s in self.stages.items())
        B = sum(s['ops_s']*CLS[k][1] for k, s in self.stages.items())
        C = sum(s['ops_s']*CLS[k][2] for k, s in self.stages.items())
        occ = {k: self.service_time(k)*s['rate'] for k, s in self.stages.items()}
        tiers = {}
        for k, s in self.stages.items():
            tiers[TIER[k]] = tiers.get(TIER[k], 0) + s['ops_s']
        # reactive critical path: one call each of the sense->act chain
        chain = ['sgm', 'mono', 'lidar', 'lio', 'vio', 'esdf', 'sdfenc',
                 'policy', 'mpc', 'cbf', 'ctrl']
        lat = sum(self.service_time(k) for k in chain if k in self.stages)
        return dict(
            ff=self.ff, name=self.name, note=self.note, budget=self.budget_w,
            tops=tot/1e12, gops=tot/1e9, bw=byt/1e9,
            A=A/tot, B=B/tot, C=C/tot,
            occupancy=sum(occ.values()), occ=occ,
            over=[k for k, v in occ.items() if v >= 1.0],
            tiers={t: v/tot for t, v in tiers.items()},
            chain_ms=lat*1e3, deadline_ms=self.deadline_ms,
            np5=tot/1e12/0.05, np2=tot/1e12/0.02,
            wsi=(tot/1e12/0.02)/2.3, wceil=(tot/1e9)/101.8)
