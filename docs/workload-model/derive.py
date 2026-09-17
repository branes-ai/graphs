# -*- coding: utf-8 -*-
"""
Forward derivation of every stage unit cost from algorithm structure.

Each stage returns an itemized op count and byte count at a STATED configuration.
Nothing here is back-solved from a published total; the comparison against the
companion annex's implied values is done separately, in compare().
"""
from math import log2

D = []   # derivation records

def stage(key, name, tier, unit, config, op_items, byte_items, cls, basis):
    ops = sum(v for _, _, v in op_items)
    byts = sum(v for _, _, v in byte_items)
    rec = dict(key=key, name=name, tier=tier, unit=unit, config=config,
               op_items=op_items, byte_items=byte_items, ops=ops, bytes=byts,
               opb=(ops / byts if byts else float('inf')), cls=cls, basis=basis)
    D.append(rec)
    return rec

# ---------------------------------------------------------------- T1
W, H, DISP = 1440, 1080, 128
PX = W * H                                    # 1.5552 M pixels

stage('lidar', 'LiDAR deskew + voxel downsample', 'T1', 'per input point',
  {'points per revolution': '~100 k', 'voxel edge': '0.10 m', 'hash load factor': '0.7'},
  [('Timestamp normalize + trajectory bracket lookup', '10', 10),
   ('Quaternion SLERP (dot, acos, 2 sin, 4 mul-add, normalize)', '60', 60),
   ('Translation LERP', '3 mul + 3 add', 6),
   ('Quaternion to rotation matrix', '30', 30),
   ('Rotate + translate point', '9 mul + 6 add + 3 add', 18),
   ('Voxel index (3 div + 3 floor)', '3*4', 12),
   ('Spatial hash (3 mul, 2 xor, 1 mod)', '10', 10),
   ('Bucket probe + key compare, 1.5 probes avg', '1.5*8', 12),
   ('Centroid accumulate (3 add + count)', '4', 4),
   ('Range / intensity / NaN validity filter', '8', 8)],
  [('Read point (x,y,z,intensity, fp32)', '4*4', 16),
   ('Hash bucket read, cache-line granular', '64', 64),
   ('Hash bucket write-back', '64', 64),
   ('Voxel accumulator read-modify-write', '64', 64),
   ('Trajectory sample read (amortized 1/64 points)', '256/64', 4),
   ('Surviving downsampled point write (1 in 8)', '16/8', 2)],
  (0, 0, 1.0),
  'SE(3) motion compensation per FAST-LIO2 / LOAM deskew; voxel-hash downsample. '
  'Byte count is cache-line granular because the hash probe is scattered.')

stage('sgm', 'Stereo rectify + SGM disparity', 'T1', 'per stereo frame-pair',
  {'resolution': f'{W}x{H} mono', 'disparities': DISP, 'aggregation paths': 4,
   'cost': 'Census 5x5 + Hamming'},
  [('Rectify both images (bilinear resample, 20 op/px)', f'2*{PX}*20', 2*PX*20),
   ('Census 5x5 transform both images (24 compare + pack)', f'2*{PX}*30', 2*PX*30),
   ('Matching cost: XOR + popcount per pixel-disparity', f'{PX}*{DISP}*4', PX*DISP*4),
   ('Path aggregation, 4 paths x (3 compare + 2 add + 1 sub)', f'{PX}*{DISP}*4*6', PX*DISP*24),
   ('Sum aggregated paths', f'{PX}*{DISP}*3', PX*DISP*3),
   ('Winner-take-all + sub-pixel + L-R consistency + median', f'{PX}*40', PX*40)],
  [('Read both source images (8-bit)', f'2*{PX}', 2*PX),
   ('Rectified images write + read', f'2*2*{PX}', 4*PX),
   ('Census descriptors write + read (32-bit)', f'2*2*{PX}*4', 16*PX),
   ('Cost volume write (uint16)', f'{PX}*{DISP}*2', PX*DISP*2),
   ('Cost volume read by aggregation', f'{PX}*{DISP}*2', PX*DISP*2),
   ('Aggregated volume write + read for path sum', f'2*{PX}*{DISP}*2', PX*DISP*4),
   ('Disparity map write (uint16)', f'{PX}*2', PX*2)],
  (0.85, 0, 0.15),
  'Semi-global matching per Hirschmuller, 4-path real-time variant with Census-Hamming '
  'cost. Cost-volume traffic assumes competent tiling; an untiled implementation moves '
  'an order of magnitude more.')

RX, NS, NC = 32, 1024, 256      # virtual channels, range samples, chirps
stage('radar', 'Radar FMCW post-processing', 'T1', 'per radar scan',
  {'virtual channels': RX, 'range samples': NS, 'chirps per frame': NC,
   'array': '4 Tx x 8 Rx MIMO'},
  [('Windowing + calibration + DC removal', f'{RX}*{NS}*{NC}*8/1000', RX*NS*NC*8),
   ('Range FFT (radix-2, 6 flop/butterfly)', f'{RX}*{NC}*({NS}/2*{int(log2(NS))}*6)',
    int(RX*NC*(NS/2*log2(NS)*6))),
   ('Doppler FFT', f'{RX}*{NS}*({NC}/2*{int(log2(NC))}*6)', int(RX*NS*(NC/2*log2(NC)*6))),
   ('2-D CA-CFAR, 32-cell training window', f'{NS}*{NC}*{RX}*40', NS*NC*RX*40),
   ('Angle-of-arrival FFT over virtual array + peak interpolation',
    f'{NS}*{NC}*{RX}*10', NS*NC*RX*10)],
  [('ADC cube in (complex int16)', f'{RX}*{NS}*{NC}*4', RX*NS*NC*4),
   ('Range-FFT output write + read (complex fp32)', f'2*{RX}*{NS}*{NC}*8', 2*RX*NS*NC*8),
   ('Doppler-FFT output write + read', f'2*{RX}*{NS}*{NC}*8', 2*RX*NS*NC*8),
   ('CFAR magnitude map read', f'{NS}*{NC}*4', NS*NC*4),
   ('Detection list out', '64*256', 16384)],
  (0.70, 0, 0.30),
  'Standard FMCW MIMO chain: range FFT, Doppler FFT, CFAR, angle estimation. '
  'Aperture is the dominant assumption — a 12-channel 512x128 radar costs ~12x less.')

stage('mono', 'Mono camera front-end', 'T1', 'per pixel',
  {'assumption': 'demosaic and denoise performed in the sensor ISP'},
  [('Rectify (bilinear resample)', '20', 20),
   ('Resize / pyramid level 0', '8', 8),
   ('Normalize + format convert for the detector', '5', 5),
   ('Photometric calibration (vignette, gain)', '7', 7)],
  [('Read source pixel (8-bit)', '1', 1),
   ('Write rectified pixel', '1', 1),
   ('Write resized tensor element (fp16)', '2', 2)],
  (0, 0, 1.0),
  'Light path only. A full software ISP (demosaic + 3DNR) would add roughly 35 op/px.')

# ---------------------------------------------------------------- T2
stage('lio', 'LiDAR-inertial odometry', 'T2', 'per input point',
  {'ikd-tree size': '~1e5 nodes', 'kNN k': 5, 'IEKF iterations': 3,
   'downsampling into optimizer': 'none — all deskewed points used'},
  [('ikd-tree kNN search with backtracking (~180 node visits x 20 op)', '180*20', 3600),
   ('Plane fit over 5 neighbors (3x3 scatter + eigen)', '300', 300),
   ('Point-to-plane residual + Huber weight', '60', 60),
   ('Jacobian row build (1 x 18)', '90', 90),
   ('IEKF normal-equation accumulation, 3 iterations (18x18 rank-1)',
    '3*18*18*2', 3*18*18*2),
   ('Incremental tree insert / rebalance (amortized)', '150', 150)],
  [('Read deskewed point', '16', 16),
   ('ikd-tree node traversal, cache-line granular (~12 lines)', '12*64', 768),
   ('Neighbor point fetch (5 x 16, 2 lines)', '128', 128),
   ('Jacobian row write', '18*4', 72),
   ('Normal-equation accumulator RMW (amortized)', '64', 64)],
  (0, 0, 1.0),
  'FAST-LIO2-class iterated-EKF formulation. The no-downsampling assumption is the '
  'single largest lever in this stage: 10x decimation into the optimizer divides it by ~10.')

FEAT = 300
stage('vio', 'Visual front-end (per camera)', 'T2', 'per pixel',
  {'tracked features': FEAT, 'KLT window': '21x21', 'pyramid levels': 4,
   'KLT iterations per level': 20},
  [('FAST-9 corner test, early-exit average', '20', 20),
   ('Shi-Tomasi score on 1% candidate pixels (7x7 second-moment)', '0.01*300', 3),
   ('Non-maximum suppression', '5', 5),
   ('Pyramidal KLT: 4 lvl x 20 iter x 441 px x 14 op, per feature, per pixel',
    f'4*20*441*14*{FEAT}/{PX}', 4*20*441*14*FEAT/PX),
   ('Fundamental-matrix RANSAC outlier rejection (amortized)',
    f'500*7*{FEAT}/{PX}', 500*7*FEAT/PX)],
  [('Read pixel', '1', 1),
   ('Pyramid build write + read', '3', 3),
   ('KLT window fetches per feature, amortized over frame',
    f'4*20*441*2*{FEAT}/{PX}', 4*20*441*2*FEAT/PX)],
  (0.60, 0, 0.40),
  'FAST/Shi-Tomasi extraction with pyramidal Lucas-Kanade tracking. Cost is dominated '
  'by the KLT term, which scales with feature count, not resolution.')

KF, LM = 15, 800
stage('ba', 'Windowed bundle adjustment', 'T2', 'per solve',
  {'keyframes in window': KF, 'landmarks': LM, 'cameras': 2, 'LM iterations': 5},
  [('Residual + Jacobian build (KF x LM x 2 obs x 540 op)',
    f'{KF}*{LM}*2*540', KF*LM*2*540),
   ('Schur complement: per landmark V^-1 and U -= W V^-1 W^T',
    f'{LM}*(50+{KF}*6*3*3+{KF}*6*3*{KF}*6)', LM*(50 + KF*6*3*3 + KF*6*3*KF*6)),
   ('Cholesky of reduced camera system (90x90)', f'{KF*6}**3/3', int((KF*6)**3/3)),
   ('Back-substitution for landmarks', f'{LM}*{KF}*6*3*2', LM*KF*6*3*2),
   ('Levenberg-Marquardt damping + cost re-evaluation, 5 iterations',
    'x5 on the above', 0)],
  [('Landmark + pose state read', f'({LM}*3+{KF}*6)*8', (LM*3+KF*6)*8),
   ('Observation read', f'{KF}*{LM}*2*8', KF*LM*2*8),
   ('Jacobian block write + read', f'2*{KF}*{LM}*2*9*8', 2*KF*LM*2*9*8),
   ('Reduced-system RMW', f'2*{KF*6}**2*8', 2*(KF*6)**2*8)],
  (0, 0, 1.0),
  'Schur-complement bundle adjustment with Levenberg-Marquardt. Cost scales as the cube '
  'of the keyframe window and linearly in landmarks; the window size is the lever.')
# apply the 5 LM iterations to the four preceding items
_ba = D[-1]
_core = sum(v for _, _, v in _ba['op_items'][:4])
_ba['op_items'][4] = ('Levenberg-Marquardt damping + re-evaluation (x5 on the above)',
                      '4*core', 4 * _core)
_ba['ops'] = sum(v for _, _, v in _ba['op_items'])
_ba['opb'] = _ba['ops'] / _ba['bytes']

# ---------------------------------------------------------------- T3
VOX, TRUNC, RANGE = 0.05, 0.30, 30.0
NB = int(2 * TRUNC / VOX)            # voxels in the truncation band
NF = int(RANGE / VOX / 5)            # free-space voxels carved, 1-in-5 density
stage('tsdf', 'TSDF integration', 'T3', 'per measured point',
  {'voxel edge': f'{VOX} m', 'truncation': f'+/-{TRUNC} m', 'sensor range': f'{RANGE} m',
   'free-space carving': '1 voxel in 5 along the ray'},
  [(f'Truncation-band update, {NB} voxels x (hash 10 + SDF 10 + weighted update 6 + write 2)',
    f'{NB}*28', NB*28),
   (f'Free-space carving, {NF} voxels x 28 op', f'{NF}*28', NF*28),
   ('Ray setup + DDA stepping state', '40', 40)],
  [(f'Truncation-band voxel RMW, cache-line granular ({NB} x 64 B, 60% line reuse)',
    f'{NB}*64*0.4', int(NB*64*0.4)),
   (f'Free-space voxel RMW ({NF} x 64 B, 90% line reuse)', f'{NF}*64*0.1', int(NF*64*0.1)),
   ('Block-hash lookups', '2*64', 128),
   ('Read measured point', '16', 16)],
  (0, 0, 1.0),
  'Bundled raycasting with weighted truncated SDF update, per Voxblox/nvblox. '
  'Free-space carving density is the main lever; band-only integration costs ~11x less.')

CHG = 400_000
stage('esdf', 'ESDF propagation', 'T3', 'per map update',
  {'voxels changed per update': f'{CHG:,}', 'neighborhood': '26-connected',
   'wavefront passes': 3},
  [(f'Wavefront relaxation: {CHG:,} voxels x 26 neighbors x (compare 8 + update 7)',
    f'{CHG}*26*15', CHG*26*15),
   ('Priority-queue push/pop bookkeeping', f'{CHG}*26*10', CHG*26*10),
   ('Convergence passes (x3 on the above)', 'x3', 0)],
  [(f'Voxel distance RMW ({CHG:,} x 26 x 4 B, 85% line reuse)',
    f'{CHG}*26*4*0.15', int(CHG*26*4*0.15)),
   ('Queue traffic', f'{CHG}*3*8', CHG*3*8)],
  (0, 0, 1.0),
  'Incremental brushfire/wavefront over the TSDF. The changed-voxel count scales with '
  'platform speed and is the dominant assumption.')
_e = D[-1]
_e['op_items'][2] = ('Convergence passes (x3 on the above)', '2*core',
                     2 * sum(v for _, _, v in _e['op_items'][:2]))
_e['ops'] = sum(v for _, _, v in _e['op_items']); _e['opb'] = _e['ops'] / _e['bytes']

# ---------------------------------------------------------------- T4
DET_P, DET_R = 24e6, 640
stage('det', 'Open-vocabulary detection', 'T4', 'per inference',
  {'backbone': 'YOLOe / YOLO-World class', 'parameters': '24 M', 'input': f'{DET_R}x{DET_R}',
   'text embeddings': 'precomputed and cached'},
  [('Backbone + neck + head: 45 GMAC published at 640^2, ops = 2 x MAC', '2*45e9', 90e9),
   ('Region-text similarity over cached embeddings (300 x 512 x 80 x 2)',
    '300*512*80*2', 300*512*80*2),
   ('NMS + decode', '3e8', 3e8),
   ('Letterbox + normalize', '640*640*3*10', 640*640*3*10)],
  [('Weights, INT8', '24e6', 24e6),
   ('Input tensor (fp16)', '640*640*3*2', 640*640*3*2),
   ('Activation spill to DRAM (measured ~6x weights on Orin-class cache)',
    '6*24e6', 6*24e6),
   ('Cached text embeddings', '80*512*2', 80*512*2)],
  (0.85, 0.15, 0),
  'Anchored to the published FLOP count of a YOLOe-class open-vocabulary detector at '
  '640x640. This is the best-supported unit cost in the model.')

VLM_P, PRE, DEC = 2e9, 600, 64
stage('vlm', 'Onboard vision-language reasoning', 'T4', 'per query',
  {'parameters': '2 B', 'quantization': 'INT8 weights', 'prefill tokens': PRE,
   'decode tokens': DEC},
  [(f'Prefill: 2 x {VLM_P:.0e} params x {PRE} tokens', f'2*2e9*{PRE}', 2*VLM_P*PRE),
   (f'Decode: 2 x params x {DEC} tokens', f'2*2e9*{DEC}', 2*VLM_P*DEC),
   ('Vision encoder (ViT-L/14 at 336px)', '160e9', 160e9)],
  [('Prefill weight sweep (INT8), once', '2e9', VLM_P),
   (f'Decode weight sweep, once per generated token ({DEC}x)', f'{DEC}*2e9', DEC*VLM_P),
   ('KV cache write + read', '664*2*32*2048*2', 664*2*32*2048*2)],
  (0.60, 0.40, 0),
  'Transformer arithmetic is 2 ops per parameter per token. Decode re-reads the entire '
  'weight set per token, which is why this stage is bandwidth-bound rather than '
  'compute-bound.')

VLA_P, VLA_T = 3e9, 56
stage('vla', 'Vision-language-action policy', 'T4', 'per action-chunk step',
  {'backbone': '3 B', 'tokens per forward pass': VLA_T,
   'action chunking': '8 control steps per pass'},
  [(f'Forward pass: 2 x {VLA_P:.0e} params x {VLA_T} tokens', f'2*3e9*{VLA_T}',
    2*VLA_P*VLA_T),
   ('Vision tokenizer for 2 views', '2*20e9', 2*20e9),
   ('Action decode head', '2e9', 2e9)],
  [('Weight sweep per forward pass (INT8)', '3e9', VLA_P),
   ('KV / activation traffic', '56*2*36*2048*2', 56*2*36*2048*2),
   ('Two camera views in (fp16)', '2*448*448*3*2', 2*448*448*3*2)],
  (0.60, 0.40, 0),
  'Chunked VLA in the pi-0 / OpenVLA family. Chunking amortizes one forward pass over '
  'several control steps, which is what makes 30 Hz action output affordable at all.')

# ---------------------------------------------------------------- T5
RAYS, RVOX = 200, 120
stage('gain', 'Volumetric information-gain evaluation', 'T5', 'per candidate viewpoint',
  {'rays per frustum': RAYS, 'voxels per ray': RVOX},
  [(f'Frustum raycast: {RAYS} rays x {RVOX} voxels x (DDA step 3 + hash 10 + classify 3)',
    f'{RAYS}*{RVOX}*16', RAYS*RVOX*16),
   ('Gain accumulation + entropy weighting', f'{RAYS}*{RVOX}*2', RAYS*RVOX*2),
   ('Pose sampling + frustum setup', '2000', 2000)],
  [(f'Scattered voxel occupancy reads, cache-line granular ({RAYS}*{RVOX} x 64 B, '
    '85% line reuse)', f'{RAYS}*{RVOX}*64*0.15', int(RAYS*RVOX*64*0.15)),
   ('Gain accumulator write', '8', 8)],
  (0, 0, 1.0),
  'Frustum raycasting per candidate vertex over unknown space. Roughly 1 op per byte: '
  'this is memory traffic wearing the label of planning.')

NSAMP = 2000
stage('graph', 'Sampling-based graph + collision check', 'T5', 'per replan',
  {'samples': NSAMP, 'neighbors per sample': 10, 'collision queries per edge': 50},
  [(f'Sample + nearest-neighbor ({NSAMP} x 200 op)', f'{NSAMP}*200', NSAMP*200),
   (f'Edge collision checking ({NSAMP} x 10 edges x 50 ESDF queries x 15 op)',
    f'{NSAMP}*10*50*15', NSAMP*10*50*15),
   ('Dijkstra / A* over the roadmap', f'{NSAMP}*10*20', NSAMP*10*20),
   ('Path smoothing + reparameterization', '200000', 200000)],
  [('ESDF queries, cache-line granular (85% reuse)',
    f'{NSAMP}*10*50*64*0.15', int(NSAMP*10*50*64*0.15)),
   ('Graph structure RMW', f'{NSAMP}*10*32', NSAMP*10*32)],
  (0, 0, 1.0),
  'PRM/RRT-class roadmap with edge collision queries against the ESDF. Sample count is '
  'the lever and varies by more than an order of magnitude between regimes.')

# ---------------------------------------------------------------- T6
stage('sdfenc', 'Learned SDF encoder (neural SDF-MPC)', 'T6', 'per call',
  {'input': '64 x 1024 range image', 'encoder': 'conv trunk + MLP SDF head, 6 M params'},
  [('Convolutional range-image encoder, 11 GMAC (ResNet-18 trunk on 64x1024), ops = 2 x MAC', '2*11e9', 22e9),
   ('MLP SDF head evaluated at 4096 query points', '4096*3*512*512*2/1000',
    4096*3*512*2*2),
   ('Range-image projection + normalize', '64*1024*40', 64*1024*40)],
  [('Weights, INT8', '6e6', 6e6),
   ('Range image in (fp16)', '64*1024*2', 64*1024*2),
   ('Activation traffic (~8x weights)', '8*6e6', 8*6e6),
   ('Query points + SDF out', '4096*4*4', 4096*16)],
  (0.85, 0.15, 0),
  'Convolutional encoder plus an implicit SDF head, sized to published neural-SDF '
  'planners. Class A in the trunk, Class B in the head.')

stage('policy', 'Learned navigation policy (DRL)', 'T6', 'per call',
  {'encoder': 'depth-collision CNN, 2.5 M params', 'head': '3-layer actor MLP'},
  [('Depth collision encoder, 4.5 GMAC, ops = 2 x MAC', '2*4.5e9', 9e9),
   ('Actor MLP (512-512-6)', '2*(512*512+512*6)', 2*(512*512+512*6)),
   ('Observation assembly + normalization', '5e7', 5e7)],
  [('Weights, INT8', '2.5e6', 2.5e6),
   ('Depth input (fp16)', '224*224*2', 224*224*2),
   ('Activation traffic (~10x weights)', '10*2.5e6', 25e6)],
  (0.85, 0.15, 0),
  'Actor network only; the critic is not evaluated at inference.')

def nmpc(nx, nu, N, iters=5):
    ric = N * (nx**3 + nx**2 * nu + nx * nu**2) * 2
    dyn = N * (nx * nx * 6 + nx * nu * 6 + 200)
    cond = N * (nx**2 * nu) * 2
    return iters * (ric + dyn + cond)

stage('mpc', 'Nonlinear MPC solve (quadrotor, 4 DoF)', 'T6', 'per solve',
  {'state dim': 12, 'control dim': 4, 'horizon': 30, 'SQP iterations': 5},
  [('Dynamics + Jacobian evaluation over the horizon', '30*(12*12*6+12*4*6+200)',
    30*(12*12*6+12*4*6+200)),
   ('Riccati recursion (nx^3 + nx^2 nu + nx nu^2) per step', '30*(1728+576+192)*2',
    30*(12**3+12**2*4+12*4**2)*2),
   ('Condensing / KKT assembly', '30*12^2*4*2', 30*(12**2*4)*2),
   ('SQP iterations (x5 on the above)', 'x5', 0)],
  [('State + reference trajectory', '30*16*8', 30*16*8),
   ('Hessian / KKT working set RMW', '5*30*12*12*8', 5*30*144*8),
   ('Solution write-back', '30*4*8', 960)],
  (0, 0, 1.0),
  'Sequential quadratic programming with a Riccati recursion over the horizon. '
  'Cost scales as horizon x nx^3, which is why the humanoid case is 100x the quadrotor.')
_m = D[-1]
_m['op_items'][3] = ('SQP iterations (x5 on the above)', '4*core',
                     4 * sum(v for _, _, v in _m['op_items'][:3]))
_m['ops'] = sum(v for _, _, v in _m['op_items']); _m['opb'] = _m['ops'] / _m['bytes']

stage('cbf', 'Composite CBF safety filter', 'T6', 'per solve',
  {'decision variables': 3, 'active hazard constraints': 12,
   'ESDF window': '15 x 15 x 15 voxels', 'QP solver': 'OSQP / active set'},
  [('Sweep local ESDF window for live hazards (3375 voxels x 6 op)', '3375*6', 3375*6),
   ('Composite barrier: smooth-min blend over 12 hazards + gradients', '12*120', 1440),
   ('Lie-derivative evaluation Lf h and Lg h', '12*90', 1080),
   ('QP solve, 3 vars x 12 constraints, ~40 iterations', '40*3*12*4', 40*3*12*4),
   ('Feasibility fallback + slack relaxation', '600', 600)],
  [('ESDF window read, cache-line granular (3375 x 4 B, 90% reuse)',
    '3375*4*0.1+15*15*64*0.5', int(3375*4*0.1 + 15*15*64*0.5)),
   ('Nominal control in, filtered control out', '2*4*8', 64),
   ('Constraint working set', '12*4*8', 384)],
  (0, 0, 1.0),
  'Weighted virtual-obstacle QP over a sliding map window. The solve itself is trivial; '
  'the cost is the map sweep that finds which hazards are live.')

stage('ctrl', 'Attitude / rate or joint servo control', 'T7', 'per update per DoF',
  {'controller': 'cascaded PID or geometric attitude controller'},
  [('Setpoint shaping + feed-forward', '120', 120),
   ('Cascaded PID (3 loops x 30 op)', '90', 90),
   ('Mixer / allocation row', '80', 80),
   ('Saturation, anti-windup, rate limiting', '60', 60),
   ('Sensor filtering (2nd-order IIR x 3 axes)', '90', 90)],
  [('IMU / encoder sample in', '24', 24),
   ('Controller state RMW', '64', 64),
   ('Actuator command out', '8', 8)],
  (0, 0, 1.0),
  'Runs on the flight-control microcontroller or joint controller, not the mission '
  'computer. Included for completeness; it is under 0.1% of every mission total.')

BY_KEY = {r['key']: r for r in D}

if __name__ == '__main__':
    print(f"{'stage':44}{'unit':28}{'ops':>14}{'bytes':>14}{'op/B':>9}")
    for r in D:
        print(f"{r['name'][:43]:44}{r['unit'][:27]:28}{r['ops']:14,.0f}"
              f"{r['bytes']:14,.0f}{r['opb']:9.2f}")
