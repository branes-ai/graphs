# -*- coding: utf-8 -*-
"""
Tile specialization model: what each pipeline operator gains from a dedicated
KPU tile instead of a general-purpose SIMT datapath.

Two independent effects are modeled and kept separate:
  throughput gain  — effective GOP/s of the datapath on that operator's kernel
  energy gain      — GOPS/W of the datapath on that operator's kernel
  retention        — fraction of the operator's byte traffic that stays on-chip

The Navion measurement anchors the state-estimation tile; every other tile is
scaled from it by how far the operator is from a SIMT machine's native form.
"""
from profiles import PROFILES
from pipeline import CLS, TIER, NAME

# --------------------------------------------------------------- GP baseline
# achieved energy efficiency of general-purpose embedded silicon, by class
GP_EFF_W = dict(A=101.8, B=20.0, C=2.0)      # GOPS per watt
# effective throughput by class (companion annex, Section 8)
GP_THR = dict(A=2000.0, B=300.0, C=15.0)     # GOP/s
E_DRAM = 25e-12                              # J per byte, LPDDR5 incl. row activation
E_DRAM_K = 12.5e-12                          # J per byte, row-streamed under a static schedule
E_SRAM = 1.5e-12                             # J per byte, on-chip

# --------------------------------------------------------------- tile catalog
# stage key -> (tile name, throughput gain, energy gain, on-chip retention, basis)
TILES = {
 'rect': None,
 'mono': ('Geometric resample tile', 12, 60, 0.70,
   'Bilinear warp with a fixed sampling pattern. On a GPU this runs on shader cores '
   'with address arithmetic in the general datapath; as a tile it is a fixed-stride '
   'interpolator with the line buffer resident.'),
 'sgm': ('Stereo cost-volume tile', 8, 80, 0.92,
   'Census transform is bit-parallel comparison and Hamming popcount — 32-bit SIMT lanes '
   'waste most of their width on it. The 398 MB cost volume is tiled and consumed in '
   'place, so aggregation never round-trips to DRAM.'),
 'radar': ('Spectral tile (FFT / CFAR)', 15, 120, 0.85,
   'Radix-2 butterflies and a sliding-window constant-false-alarm reduction. Both are '
   'fixed dataflow with no divergence; a systolic FFT engine with the data cube resident '
   'is the textbook case for fixed-function silicon.'),
 'lidar': ('Point-transform and hash tile', 20, 100, 0.80,
   'SE(3) interpolation per point plus a scattered hash probe. On a GPU the probe '
   'diverges the warp and every 4-byte access moves a 64-byte line; the tile holds the '
   'hash in local SRAM.'),
 'lio': ('State-estimation tile (Navion class)', 25, 330, 1.00,
   'The measured anchor. MIT Navion runs a complete visual-inertial pipeline — frontend '
   'tracking and factor-graph backend — at sensor rate on 2 mW in 65 nm with 854 kB of '
   'on-chip SRAM. The whole working set is resident, so there is no DRAM term at all.'),
 'vio': ('Feature-tracking tile', 20, 150, 0.90,
   'Pyramidal Lucas-Kanade is a small dense solve per feature. On a SIMT machine this is '
   'hundreds of divergent 21x21 windows; as a tile it is a fixed pipeline with the pyramid '
   'in local memory.'),
 'ba': ('State-estimation tile (Navion class)', 25, 330, 1.00,
   'Schur complement and Cholesky on an ill-conditioned FP64 system. Tensor cores cannot '
   'be used at all; on a GPU this falls back to the CPU cluster.'),
 'tsdf': ('Volumetric integration tile', 15, 90, 0.85,
   'Scattered read-modify-write over a voxel hash at 1.25 op per byte — a pure memory '
   'problem on any general datapath. The tile keeps the active block set resident and '
   'streams whole rows.'),
 'esdf': ('Wavefront / distance-transform tile', 18, 100, 0.85,
   'Incremental brushfire relaxation. Irregular, serial-ish, and it defeats a warp '
   'scheduler; a dedicated queue-and-relax engine holds the frontier on-chip.'),
 'det': ('Dense convolution tile', 4.0, 10, 0.60,
   'The one operator a tensor array is already built for. The gain is scheduling and '
   'activation residency, not datapath — and it is correspondingly small.'),
 'vlm': ('Transformer tile', 2.0, 4, 0.05,
   'The exception. A 2 B-parameter weight set re-read once per generated token exceeds '
   'any plausible on-chip capacity, so retention is near zero and the operator stays '
   'memory-bound. Specialization buys scheduling and layout, not a class change.'),
 'vla': ('Transformer tile', 2.2, 4.5, 0.10,
   'Slightly better than the VLM case because a chunked policy reads its weights once per '
   'action chunk rather than once per token, so a larger share can be held across the '
   'forward pass.'),
 'gain': ('Raycast / gain tile', 20, 110, 0.90,
   'Frustum raycasting at 1.0 op per byte: tens of millions of scattered voxel queries. '
   'Entirely a locality problem, and the tile is a resident occupancy cache with a DDA '
   'stepper.'),
 'graph': ('Graph-search tile', 15, 80, 0.90,
   'Sampling, collision query and shortest path — pointer chasing that a SIMT machine '
   'serializes. The roadmap and the local distance field are both small enough to hold.'),
 'sdfenc': ('Dense convolution tile', 4.0, 10, 0.65,
   'Convolutional trunk on the tensor path; the implicit SDF head is FP16 and fuses into '
   'the same tile instead of spilling.'),
 'policy': ('Dense convolution tile', 4.0, 10, 0.65,
   'As above. Encoder is Class A, actor head is Class B, and fusing them removes the '
   'round trip between them.'),
 'mpc': ('Constraint-solver tile (FP64)', 30, 250, 1.00,
   'Riccati recursion and sequential QP in FP64 at hundreds of hertz. Tensor cores are '
   'structurally inapplicable — they have no FP64 path — so on every general accelerator '
   'this work runs on the host CPU. A small dense FP64 systolic array with the horizon '
   'resident is a different machine.'),
 'cbf': ('Constraint-solver tile (FP64)', 30, 250, 1.00,
   'A three-variable barrier-function QP at up to 1 kHz, gated by a distance-field sweep. '
   'Trivial arithmetic, FP64 floor, latency-critical: the worst possible fit for a wide '
   'SIMT machine and a natural fit for a small dedicated solver.'),
 'ctrl': ('Servo loop (off-die)', 1.0, 1.0, 1.00,
   'Runs on the autopilot or joint microcontroller already. No change.'),
}


def analyze(p):
    """Return GP and KPU latency and energy roll-ups for one mission profile."""
    gp_t = gp_e = kpu_t = kpu_e = 0.0
    gp_mem = kpu_mem = 0.0
    per = []
    for k, s in p.stages.items():
        a, b, c = CLS[k]
        ops = s['ops_s'] / 1e9                     # GOP/s
        byts = s['bytes_s']                        # B/s
        tile = TILES.get(k)
        tg, eg, ret = (1.0, 1.0, 0.0) if not tile else (tile[1], tile[2], tile[3])

        t_gp = ops*a/GP_THR['A'] + ops*b/GP_THR['B'] + ops*c/GP_THR['C']
        t_kp = t_gp / tg
        e_gp = ops*a/GP_EFF_W['A'] + ops*b/GP_EFF_W['B'] + ops*c/GP_EFF_W['C']
        e_kp = e_gp / eg
        m_gp = byts * E_DRAM
        m_kp = byts * ((1-ret)*E_DRAM_K + ret*E_SRAM)

        gp_t += t_gp; kpu_t += t_kp
        gp_e += e_gp; kpu_e += e_kp
        gp_mem += m_gp; kpu_mem += m_kp
        per.append(dict(key=k, name=NAME[k], tier=TIER[k],
                        tile=(tile[0] if tile else '—'),
                        t_gp=t_gp, t_kp=t_kp, e_gp=e_gp+m_gp, e_kp=e_kp+m_kp,
                        tg=tg, eg=eg, ret=ret))
    return dict(
        ff=p.ff, name=p.name, budget=p.budget_w,
        gp_occ=gp_t, kpu_occ=kpu_t, lat_gain=gp_t/kpu_t if kpu_t else 0,
        gp_w=gp_e+gp_mem, kpu_w=kpu_e+kpu_mem,
        gp_compute_w=gp_e, gp_mem_w=gp_mem,
        kpu_compute_w=kpu_e, kpu_mem_w=kpu_mem,
        e_gain=(gp_e+gp_mem)/(kpu_e+kpu_mem) if (kpu_e+kpu_mem) else 0,
        per=per)


ROWS = [analyze(p) for p in PROFILES]

if __name__ == '__main__':
    print(f"{'mission':44}{'GP s/s':>8}{'KPU s/s':>9}{'lat x':>7}"
          f"{'GP W':>9}{'KPU W':>8}{'E x':>7}{'bud':>5}{'fits':>6}")
    for r in ROWS:
        print(f"{(r['ff'][:11]+' | '+r['name'][:28]):44}{r['gp_occ']:8.1f}"
              f"{r['kpu_occ']:9.2f}{r['lat_gain']:7.1f}{r['gp_w']:9.0f}"
              f"{r['kpu_w']:8.1f}{r['e_gain']:7.0f}{r['budget']:5d}"
              f"{'yes' if r['kpu_w'] <= r['budget'] else 'no':>6}")
