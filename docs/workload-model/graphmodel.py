# -*- coding: utf-8 -*-
"""
The compute graph: twelve high-level operators, and the typed, sized data
structures that travel between them.

Arc sizes are computed from the same mission configurations that drive the
workload model, so the sum of arc traffic is the bandwidth figure in the
mission summary.
"""
from profiles import PROFILES

# ---------------------------------------------------------------- operators
# id, label, algorithmic kernel, tier, kind
OPS = [
 ('CAM',  'Cameras',              'Image sensors, global shutter',            '—',  'source'),
 ('LID',  'LiDAR',                'Spinning or solid-state range sensor',     '—',  'source'),
 ('RAD',  'Radar',                'FMCW MIMO transceiver',                    '—',  'source'),
 ('IMU',  'IMU',                  'Accelerometer + gyroscope, 800 Hz',        '—',  'source'),

 ('RECT', 'Image rectify',        'Bilinear geometric resample',              'T1', 'op'),
 ('DESK', 'Cloud deskew',         'SE(3) warp + spatial-hash voxelize',       'T1', 'op'),
 ('FFT',  'Radar FFT + CFAR',     'Range-Doppler FFT, CA-CFAR, angle FFT',    'T1', 'op'),
 ('STER', 'Stereo match',         'Census-Hamming cost, 4-path aggregation',  'T1', 'op'),
 ('TRK',  'Feature track',        'FAST corners + pyramidal KLT',             'T2', 'op'),
 ('OPT',  'State optimizer',      'Nonlinear least squares: IEKF, Schur/LM',  'T2', 'op'),
 ('FUSE', 'Volumetric fusion',    'Bundled raycast, weighted truncated SDF',  'T3', 'op'),
 ('DIST', 'Distance transform',   'Incremental ESDF wavefront',               'T3', 'op'),
 ('CNN',  'CNN inference',        'Open-vocabulary detector',                 'T4', 'op'),
 ('XFMR', 'Transformer inference','VLM / VLA, prefill + autoregressive decode','T4', 'op'),
 ('SRCH', 'Graph search',         'Sampling, collision query, shortest path', 'T5', 'op'),
 ('CSOL', 'Constraint solver',    'MPC sequential QP + barrier-function QP',  'T6', 'op'),

 ('ACT',  'Actuators',            'Servo loops on the autopilot / joint MCU', 'T7', 'sink'),
]
OPNAME = {o[0]: o[1] for o in OPS}
OPKIND = {o[0]: o[4] for o in OPS}
OPKERNEL = {o[0]: o[2] for o in OPS}
OPTIER = {o[0]: o[3] for o in OPS}


def arcs_for(m):
    """Typed, sized arcs for one mission. bytes_per is one transfer; bps is sustained."""
    k = m.k
    A = []

    def add(src, dst, name, dtype, shape, bytes_per, rate, internal=False):
        if bytes_per <= 0 or rate <= 0:
            return
        A.append(dict(src=src, dst=dst, name=name, dtype=dtype, shape=shape,
                      bytes_per=bytes_per, rate=rate, bps=bytes_per*rate,
                      internal=internal))

    st = k.get('stereo')            # (n, w, h, fps, disp)
    mo = k.get('mono')              # (n, w, h, fps)
    rd = k.get('radar')             # (n, hz, rx, ns, nc)
    lp = k.get('lidar_pts_s', 0)
    vio = k.get('vio')

    # ---- camera path
    ncam = (st[0]*2 if st else 0) + (mo[0] if mo else 0)
    if st:
        n, w, h, fps, disp = st
        add('CAM', 'RECT', 'raw stereo frames', 'uint8', f'2 x {w}x{h}', 2*w*h, n*fps)
        add('RECT', 'STER', 'rectified pair', 'uint8', f'2 x {w}x{h}', 2*w*h, n*fps)
        add('STER', 'STER', 'cost volume', 'uint16', f'{w}x{h}x{disp}',
            w*h*disp*2, n*fps, internal=True)
        add('STER', 'FUSE', 'disparity map', 'uint16', f'{w}x{h}', w*h*2, n*fps)
    if mo:
        n, w, h, fps = mo
        add('CAM', 'RECT', 'raw frames', 'uint8', f'{n} x {w}x{h}', n*w*h, fps)
    if vio:
        n, w, h, fps = vio
        add('RECT', 'TRK', 'pyramid level 0', 'uint8', f'{n} x {w}x{h}', n*w*h, fps)
        add('TRK', 'OPT', 'feature tracks', 'float32[8]', '300 tracks', 300*32, fps)
    if k.get('det_hz'):
        add('RECT', 'CNN', 'network input tensor', 'float16', '3 x 640 x 640',
            640*640*3*2, k['det_hz'])
        if k.get('replan_hz'):
            add('CNN', 'SRCH', 'detections', 'struct{box,cls,score}', '300 objects',
                300*64, k['det_hz'])
        if k.get('cbf_hz') and not k.get('esdf_hz'):
            # no volumetric map: the safety filter takes an object list directly,
            # which is how object-level ADAS (ACC, AEB) is actually built
            add('CNN', 'CSOL', 'tracked obstacle set', 'struct{box,vel,ttc}',
                '64 objects', 64*96, k['det_hz'])
        mp = k.get('det_mpar', 24)
        add('CNN', 'CNN', 'weights + activation spill', 'int8', f'{mp} M params',
            int(mp*1e6*7), k['det_hz'], internal=True)

    # ---- LiDAR path
    if lp:
        add('LID', 'DESK', 'raw point cloud', 'float32[4]', 'x, y, z, intensity', 16, lp)
        add('DESK', 'OPT', 'deskewed cloud', 'float32[4]', 'motion-compensated', 16, lp)
        add('DESK', 'FUSE', 'voxel centroids', 'float32[4]', '1 in 8 retained', 2, lp)
        add('DESK', 'DESK', 'voxel hash probes', 'hash buckets', '64 B cache lines',
            192, lp, internal=True)

    # ---- radar path
    if rd:
        n, hz, rx, ns, nc = rd
        add('RAD', 'FFT', 'ADC data cube', 'complex int16', f'{rx} x {ns} x {nc}',
            rx*ns*nc*4, n*hz)
        add('FFT', 'FFT', 'range-Doppler spectra', 'complex float32', f'{rx} x {ns} x {nc}',
            4*rx*ns*nc*8, n*hz, internal=True)
        add('FFT', 'OPT', 'radar detections', 'float32[8]', '~64 targets', 64*32, n*hz)

    # ---- inertial. State estimation exists only where there is range or visual
    # odometry to fuse it with; a fixed smart sensor carries no IMU at all.
    est = bool(lp) or bool(vio)
    if est:
        add('IMU', 'OPT', 'inertial samples', 'float32[7]', 'accel, gyro, timestamp', 28, 800)

    # ---- state estimate fan-out
    odo = (vio[3] if vio else 20)
    if est and k.get('tsdf_pts_s'):
        add('OPT', 'FUSE', 'pose + covariance', 'float64', 'SE(3) + 6x6', 48+288, odo)
    if est:
        add('OPT', 'CSOL', 'state estimate', 'float64[18]', 'pose, vel, bias', 144,
            k.get('mpc_hz', 0))
    if k.get('ba_hz'):
        add('OPT', 'OPT', 'normal equations', 'float64', '90x90 reduced system',
            90*90*8*5, k['ba_hz'], internal=True)

    # ---- mapping
    if k.get('tsdf_pts_s'):
        add('FUSE', 'FUSE', 'TSDF voxel read-modify-write', 'float32 + weight',
            '132 voxels per ray', 1193, k['tsdf_pts_s'], internal=True)
    if k.get('esdf_hz'):
        add('FUSE', 'DIST', 'changed TSDF blocks', 'int8 blocks', '400 k voxels',
            400_000*4, k['esdf_hz'])
        add('DIST', 'DIST', 'wavefront relaxation', 'float32', '26-connected sweep',
            15_840_000, k['esdf_hz'], internal=True)
        if k.get('replan_hz'):
            add('DIST', 'SRCH', 'ESDF region', 'float32 grid', '100 k voxels',
                100_000*4, k['replan_hz'])
    if k.get('cbf_hz') and k.get('esdf_hz'):
        win = k.get('cbf_win', 15)
        add('DIST', 'CSOL', 'local distance window', 'float32', f'{win}^3 voxels',
            win**3*4, k['cbf_hz'])

    # ---- language models
    if k.get('vlm_qps'):
        p = k.get('vlm_par', 2); dec = k.get('vlm_decode', 64)
        add('CNN', 'XFMR', 'region embeddings', 'float16', '300 x 512', 300*512*2,
            k['vlm_qps'])
        add('XFMR', 'XFMR', 'autoregressive weight sweep', 'int8', f'{p:g} B params',
            int(p*1e9*(1+dec)), k['vlm_qps'], internal=True)
        add('XFMR', 'SRCH', 'grounded task plan', 'tokens', '64 tokens', 64*4,
            k['vlm_qps'])
    if k.get('vla_hz'):
        p = k.get('vla_par', 3)
        add('CNN', 'XFMR', 'visual context', 'float16', '2 x 448 x 448 x 3',
            2*448*448*3*2, k['vla_hz'])
        add('XFMR', 'XFMR', 'weight sweep per chunk', 'int8', f'{p:g} B params',
            int(p*1e9), k['vla_hz'], internal=True)
        add('XFMR', 'CSOL', 'action chunk', 'float32[8]', '8 control steps', 8*8*4,
            k['vla_hz'])

    # ---- planning and control
    if k.get('replan_hz') and (k.get('mpc_hz') or k.get('cbf_hz')):
        add('SRCH', 'CSOL', 'waypoint path', 'float64[7]', '30 poses', 30*56,
            k['replan_hz'])
    dof = k.get('dof', 4)
    if k.get('ctrl_hz'):
        add('CSOL', 'ACT', 'control vector', 'float64', f'{dof} channels', dof*8,
            k['ctrl_hz'])
    return A


ARCS = {p.ff + ' | ' + p.name: arcs_for(p) for p in PROFILES}
BY_PROFILE = {p.ff + ' | ' + p.name: p for p in PROFILES}


def active_ops(m):
    a = arcs_for(m)
    s = set()
    for x in a:
        s.add(x['src']); s.add(x['dst'])
    return s


def fmt_bytes(b):
    for u, d in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if b >= d:
            return f'{b/d:,.1f} {u}'
    return f'{b:,.0f} B'


def fmt_rate(b):
    for u, d in (('GB/s', 1e9), ('MB/s', 1e6), ('kB/s', 1e3)):
        if b >= d:
            return f'{b/d:,.1f} {u}'
    return f'{b:,.0f} B/s'


if __name__ == '__main__':
    for key, a in ARCS.items():
        tot = sum(x['bps'] for x in a)
        ext = sum(x['bps'] for x in a if not x['internal'])
        print(f'{key[:50]:52} {len(a):3} arcs  total {fmt_rate(tot):>10}  '
              f'inter-operator {fmt_rate(ext):>10}')
