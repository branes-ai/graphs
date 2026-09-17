# -*- coding: utf-8 -*-
"""Build the workload data annex workbook: every number, with its derivation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from derive import D as DERIVS, BY_KEY
from pipeline import EFF, CLS, TIER, NAME
from profiles import PROFILES, ROWS

NAVY = '0B2E4F'; RUST = 'B03A2E'; BAND = 'F2F5F8'; GREY = '4A4A4A'; RULE = 'C6CED6'
HDR = Font(name='Calibri', size=9, bold=True, color='FFFFFF')
TTL = Font(name='Calibri', size=14, bold=True, color=NAVY)
SUB = Font(name='Calibri', size=9, italic=True, color=GREY)
BOD = Font(name='Calibri', size=9)
BLD = Font(name='Calibri', size=9, bold=True, color=NAVY)
RED = Font(name='Calibri', size=9, bold=True, color=RUST)
FILLH = PatternFill('solid', fgColor=NAVY)
FILLB = PatternFill('solid', fgColor=BAND)
THIN = Border(bottom=Side('thin', color=RULE))
WRAP = Alignment(wrap_text=True, vertical='top')
TOP = Alignment(vertical='top')

wb = Workbook()

def sheet(title, widths, header, freeze='A3', note=None):
    ws = wb.create_sheet(title)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws['A1'] = title; ws['A1'].font = TTL
    if note:
        ws['A2'] = note; ws['A2'].font = SUB
    r = 4 if note else 3
    for i, h in enumerate(header, 1):
        c = ws.cell(row=r, column=i, value=h)
        c.font = HDR; c.fill = FILLH; c.alignment = WRAP
    ws.freeze_panes = f'A{r+1}'
    return ws, r + 1

def put(ws, r, vals, fills=None, fonts=None, fmts=None, wrap=()):
    for i, v in enumerate(vals, 1):
        c = ws.cell(row=r, column=i, value=v)
        c.font = (fonts or {}).get(i, BOD)
        c.alignment = WRAP if i in wrap else TOP
        c.border = THIN
        if fills: c.fill = fills
        if fmts and i in fmts: c.number_format = fmts[i]

# ---------------------------------------------------------------- 1 README
ws = wb.active; ws.title = 'README'
ws.column_dimensions['A'].width = 26; ws.column_dimensions['B'].width = 118
ws['A1'] = 'Autonomy Workload — Data Annex'; ws['A1'].font = Font(
    name='Calibri', size=16, bold=True, color=NAVY)
ws['A2'] = ('The complete computational model behind the Branes.ai autonomy-compute '
            'document series. Every figure in those documents is produced by the '
            'sheets in this workbook.'); ws['A2'].font = SUB
README = [
 ('What this is',
  'A forward derivation of the compute and bandwidth demand of eighteen robot mission '
  'profiles. "Forward" means every unit cost is built up from algorithm structure at a '
  'stated configuration. No unit cost is back-solved from a published total.'),
 ('How to audit a number',
  'Any figure in the summary traces back through exactly four steps: Mission summary -> '
  'Stage detail (that mission x that stage) -> Unit cost derivation (the itemized op and '
  'byte count) -> Precision classes (which fraction runs at which numeric floor). Every '
  'intermediate is on a sheet in this workbook.'),
 ('Sheet: Unit cost derivation',
  'One block per pipeline stage. Each line is a named term in the op or byte count with '
  'its arithmetic. The block total is the cost of ONE call of that stage at the stated '
  'configuration. This is the sheet to attack first.'),
 ('Sheet: Precision classes',
  'The Class A / B / C split assigned to each stage, and the justification. Class A is '
  'INT8-eligible, Class B has an FP16 floor, Class C has an FP32/FP64 floor.'),
 ('Sheet: Mission configuration',
  'The explicit sensor suite and update rates for each of the eighteen profiles. These '
  'are the inputs a reviewer is most likely to disagree with, and they are all in one '
  'place so they can be substituted.'),
 ('Sheet: Stage detail',
  'The cross product: 18 missions x up to 19 stages. Sustained ops/s and bytes/s, plus '
  'service time, occupancy and whether the stage misses its own deadline.'),
 ('Sheet: Mission summary',
  'The eighteen-row roll-up that appears as Exhibit B in the mission-compute matrix.'),
 ('Sheet: Latency & occupancy',
  'Service time per call, occupancy per stage, total oversubscription, and the '
  'reactive-chain latency against each mission deadline.'),
 ('Sheet: Validation',
  'Independent checks: the drone rows against the companion annex\'s published regimes, '
  'and the SAE rows against silicon actually fielded at each level.'),
 ('Sheet: Exposure register',
  'Every stage where this forward derivation disagrees with the value implied by the '
  'companion annex by more than 2x, with the share of each mission total it affects. '
  'This is the honest list of what a hostile reviewer should go after.'),
 ('Convention: ops',
  'One operation is one scalar arithmetic operation. A multiply-accumulate is TWO ops. '
  'Published network costs quoted as "GFLOPs" in the Ultralytics/timm convention are MAC '
  'counts and are doubled here. Transformer arithmetic is 2 ops per parameter per token.'),
 ('Convention: bytes',
  'Bytes are counted at DRAM, cache-line granular (64 B) wherever access is scattered, '
  'because a scattered 4-byte read moves a 64-byte line. Byte counts assume competent '
  'cache blocking and are therefore LOWER bounds on what a fielded system moves.'),
 ('Convention: effective throughput',
  f'Class A {EFF["A"]:.0f} GOP/s, Class B {EFF["B"]:.0f} GOP/s, Class C {EFF["C"]:.0f} '
  'GOP/s. These are the companion annex\'s stated assumptions, anchored to the fastest '
  'batch-1 result in the 360-configuration measurement corpus. They set service time and '
  'therefore all latency and occupancy figures.'),
 ('Convention: occupancy',
  'Occupancy of a stage = service time per call x calls per second. It is the fraction '
  'of one second of machine time the stage consumes with the whole machine to itself. '
  'Occupancy >= 1.00 means the stage cannot sustain its own rate. The sum over stages is '
  'the oversubscription factor: seconds of compute demanded per second of mission.'),
 ('What is NOT modeled',
  'Kernel launch and scheduling overhead (excluded, and it runs against the incumbent — '
  'MPC and CBF are serial chains of small dependent kernels). Inter-process communication '
  'and serialization. Sensor driver cost. Thermal throttling. Memory allocation. All of '
  'these add demand; none subtracts.'),
 ('Reproducing',
  'derive.py holds the unit-cost derivations, pipeline.py the model, profiles.py the '
  'eighteen mission specifications, book.py this workbook. Changing one number in '
  'profiles.py and re-running regenerates every figure in the document series.'),
]
r = 4
for k, v in README:
    ws.cell(row=r, column=1, value=k).font = BLD
    ws.cell(row=r, column=1).alignment = TOP
    c = ws.cell(row=r, column=2, value=v); c.font = BOD; c.alignment = WRAP
    ws.row_dimensions[r].height = 13 * (1 + len(v)//115)
    r += 1

# ---------------------------------------------------------------- 2 unit costs
ws, r = sheet('Unit cost derivation', [30, 46, 26, 18, 14, 14],
  ['Stage', 'Term', 'Arithmetic', 'Configuration', 'Ops', 'Bytes'],
  note='Each block totals the cost of ONE call of that stage. Attack this sheet first.')
for d in DERIVS:
    ws.cell(row=r, column=1, value=f"{d['tier']} · {d['name']}").font = BLD
    ws.cell(row=r, column=2, value=d['unit']).font = SUB
    cfg = '; '.join(f'{k}={v}' for k, v in d['config'].items())
    c = ws.cell(row=r, column=4, value=cfg); c.font = SUB; c.alignment = WRAP
    for i in range(1, 7): ws.cell(row=r, column=i).fill = FILLB
    r += 1
    for lbl, expr, val in d['op_items']:
        put(ws, r, [None, lbl, expr, None, val, None], fmts={5: '#,##0'}, wrap=(2,)); r += 1
    for lbl, expr, val in d['byte_items']:
        put(ws, r, [None, lbl, expr, None, None, val], fmts={6: '#,##0'}, wrap=(2,)); r += 1
    put(ws, r, [None, 'TOTAL per call', f"op/byte = {d['opb']:.2f}", None,
                d['ops'], d['bytes']],
        fonts={2: BLD, 3: BLD, 5: BLD, 6: BLD}, fmts={5: '#,##0', 6: '#,##0'}); r += 1
    c = ws.cell(row=r, column=2, value='Basis: ' + d['basis']); c.font = SUB; c.alignment = WRAP
    ws.row_dimensions[r].height = 13 * (1 + len(d['basis'])//60)
    r += 2

# ---------------------------------------------------------------- 3 precision
ws, r = sheet('Precision classes', [34, 10, 10, 10, 10, 74],
  ['Stage', 'Tier', 'Class A', 'Class B', 'Class C', 'Why'],
  note='Class A is INT8-eligible; Class B has an FP16 floor; Class C an FP32/FP64 floor.')
WHY = {
 'lidar': 'SE(3) interpolation and geometric transform; FP32 minimum for pose composition.',
 'sgm': 'Census-Hamming cost and path aggregation are integer (Class A); rectification '
        'resampling and sub-pixel refinement are FP32.',
 'radar': 'FFT butterflies quantize acceptably; CFAR thresholding and angle interpolation '
          'need floating-point dynamic range.',
 'mono': 'Geometric resampling and photometric calibration, FP32 throughout.',
 'lio': 'Factor-graph residuals, Jacobians and covariance propagation. Condition numbers '
        'reach 1e6-1e10; a single Hessian entry overflows FP16.',
 'vio': 'Corner detection and descriptor work quantizes; KLT gradient solve and RANSAC '
        'geometry are FP32.',
 'ba': 'Schur complement and Cholesky on an ill-conditioned normal-equation system. FP64 '
       'in every production implementation.',
 'tsdf': 'Signed distances need fine resolution near surfaces and wide range far from '
         'them simultaneously — the defining case for FP32.',
 'esdf': 'Same as TSDF; wavefront relaxation accumulates error across the propagation.',
 'det': 'Convolution trunk quantizes to INT8 (Class A); softmax, normalization and the '
        'regression heads hold an FP16 floor (Class B).',
 'vlm': 'GEMM quantizes; attention softmax, layer norm and the output distribution do not.',
 'vla': 'Same split as the VLM; the action head is a regression output and stays FP16.',
 'gain': 'Ray-voxel traversal and entropy accumulation in FP32.',
 'graph': 'Collision geometry and graph search; a quantization error here is a missed '
          'collision.',
 'sdfenc': 'Convolutional trunk INT8; implicit SDF head FP16 because it regresses a '
           'signed unbounded quantity.',
 'policy': 'Convolutional encoder INT8; actor MLP output head FP16.',
 'mpc': 'Constrained optimization. A quantization error does not degrade an estimate, it '
        'violates a constraint.',
 'cbf': 'Barrier-function QP. A violated safety constraint is a collision. FP64.',
 'ctrl': 'Attitude and joint servo loops; FP32 on the flight controller or joint MCU.',
}
for d in DERIVS:
    a, b, c = d['cls']
    put(ws, r, [d['name'], d['tier'], a, b, c, WHY[d['key']]],
        fonts={1: BLD}, fmts={3: '0%', 4: '0%', 5: '0%'}, wrap=(6,))
    ws.row_dimensions[r].height = 13 * (1 + len(WHY[d['key']])//95); r += 1

# ---------------------------------------------------------------- 4 mission config
ws, r = sheet('Mission configuration', [17, 30, 9, 9, 68],
  ['Form factor', 'Mission capability', 'Budget W', 'Deadline ms',
   'Sensor suite and update rates (the model inputs)'],
  note='The inputs a reviewer is most likely to dispute, gathered in one place.')
KEYNAME = {
 'stereo': 'stereo (n, w, h, fps, disparities)', 'mono': 'mono cameras (n, w, h, fps)',
 'radar': 'radar (n, Hz, virt ch, range samples, chirps)', 'lidar_pts_s': 'LiDAR points/s',
 'vio': 'visual front-end (n, w, h, fps)', 'ba_hz': 'bundle adjustment Hz',
 'tsdf_pts_s': 'TSDF points/s', 'esdf_hz': 'ESDF updates/s', 'det_hz': 'detector Hz',
 'det_gmac': 'detector GMAC', 'det_mpar': 'detector M params', 'vlm_qps': 'VLM queries/s',
 'vlm_par': 'VLM B params', 'vlm_prefill': 'VLM prefill tokens',
 'vlm_decode': 'VLM decode tokens', 'vlm_vis_gop': 'VLM vision encoder GOP',
 'vla_hz': 'VLA steps/s', 'vla_par': 'VLA B params', 'gain_evals_s': 'gain evals/s',
 'replan_hz': 'replan Hz', 'sdfenc_hz': 'SDF encoder Hz', 'policy_hz': 'DRL policy Hz',
 'mpc_hz': 'MPC Hz', 'mpc_dims': 'MPC (nx, nu, horizon)', 'cbf_hz': 'CBF Hz',
 'cbf_haz': 'CBF hazards', 'cbf_win': 'CBF window', 'ctrl_hz': 'servo Hz', 'dof': 'DoF',
}
for p in PROFILES:
    spec = '; '.join(f'{KEYNAME.get(k, k)} = {v}' for k, v in p.k.items())
    put(ws, r, [p.ff, p.name, p.budget_w, p.deadline_ms, spec],
        fonts={1: BLD}, wrap=(2, 5))
    ws.row_dimensions[r].height = 13 * (1 + len(spec)//92); r += 1
ws.cell(row=r+1, column=1, value='Note').font = BLD
c = ws.cell(row=r+1, column=2, value=p.note); c.font = SUB
for i, p in enumerate(PROFILES):
    ws.cell(row=r+1+i, column=2, value=f'{p.ff} · {p.name}: {p.note}').font = SUB

# ---------------------------------------------------------------- 5 stage detail
ws, r = sheet('Stage detail',
  [17, 28, 30, 7, 11, 12, 12, 12, 11, 10, 9],
  ['Form factor', 'Mission capability', 'Stage', 'Tier', 'Rate /s',
   'Unit ops/call', 'Sustained GOP/s', 'Sustained GB/s', 'Service ms',
   'Occupancy', 'Misses?'],
  note='The cross product. Sustained GOP/s = unit ops x rate. Occupancy = service time x rate.')
for p in PROFILES:
    for k, s in p.stages.items():
        st = p.service_time(k); occ = st * s['rate']
        put(ws, r, [p.ff, p.name, NAME[k], TIER[k], s['rate'], s['uops'],
                    s['ops_s']/1e9, s['bytes_s']/1e9, st*1e3, occ,
                    'MISS' if occ >= 1.0 else ''],
            fmts={5: '#,##0.0', 6: '#,##0', 7: '#,##0.000', 8: '#,##0.000',
                  9: '#,##0.00', 10: '#,##0.000'},
            fonts={11: RED} if occ >= 1.0 else None)
        r += 1

# ---------------------------------------------------------------- 6 summary
ws, r = sheet('Mission summary',
  [17, 30, 10, 10, 9, 9, 9, 11, 11, 10, 9, 9],
  ['Form factor', 'Mission capability', 'Sustained TOP/s', 'Sustained GB/s',
   'Class A', 'Class B', 'Class C', 'Nameplate @5% TOPS', 'Nameplate @2% TOPS',
   'Silicon W @2.3 TOPS/W', 'Budget W', 'Deficit'],
  note='Exhibit B of the mission-compute matrix, regenerated from this workbook.')
for x in ROWS:
    d = x['wsi']/x['budget']
    put(ws, r, [x['ff'], x['name'], x['tops'], x['bw'], x['A'], x['B'], x['C'],
                x['np5'], x['np2'], x['wsi'], x['budget'], d],
        fmts={3: '#,##0.00', 4: '#,##0.0', 5: '0.0%', 6: '0.0%', 7: '0.0%',
              8: '#,##0', 9: '#,##0', 10: '#,##0', 12: '#,##0.0"x"'},
        fonts={3: BLD, 12: (RED if d >= 1.05 else BOD)}); r += 1

# ---------------------------------------------------------------- 7 latency
ws, r = sheet('Latency & occupancy',
  [17, 30, 12, 12, 12, 12, 44],
  ['Form factor', 'Mission capability', 'Oversubscription (s/s)',
   'Reactive chain ms', 'Deadline ms', 'Chain / deadline', 'Stages that miss their own rate'],
  note='Occupancy assumes each stage owns the whole machine, so these are LOWER bounds.')
for x in ROWS:
    over = ', '.join(NAME[k].split('(')[0].strip() for k in x['over']) or '—'
    ratio = x['chain_ms']/x['deadline_ms'] if x['deadline_ms'] else 0
    put(ws, r, [x['ff'], x['name'], x['occupancy'], x['chain_ms'], x['deadline_ms'],
                ratio, over],
        fmts={3: '#,##0.0', 4: '#,##0', 5: '#,##0', 6: '#,##0.00"x"'},
        fonts={3: BLD, 6: (RED if ratio >= 1.0 else BOD)}, wrap=(7,)); r += 1


# ---------------------------------------------------------------- 8 validation
ws, r = sheet('Validation', [34, 30, 18, 18, 12, 44],
  ['Check', 'Quantity', 'This model', 'Independent value', 'Delta', 'Source of the independent value'],
  note='Two independent checks the model was not fitted to.')
S = {x['ff'] + '|' + x['name']: x for x in ROWS}
DR = lambda n: S['Drone|' + n]
AV = lambda n: S['Autonomous vehicle|' + n]
VAL = [
 ('Companion annex, far-flight regime', 'Sustained arithmetic, TOP/s',
  DR('ISR (endurance / wide-area search)')['tops'], 10.24,
  'The Compute Requirements of Embodied Autonomy, Exhibit A'),
 ('Companion annex, air-superiority regime', 'Sustained arithmetic, TOP/s',
  DR('Interceptor (terminal engagement)')['tops'], 12.81,
  'The Compute Requirements of Embodied Autonomy, Exhibit A'),
 ('Companion annex, far-flight regime', 'Oversubscription, s of compute per s of flight',
  DR('ISR (endurance / wide-area search)')['occupancy'], 16.2,
  'The Compute Requirements of Embodied Autonomy, Section 9'),
 ('Companion annex, air-superiority regime', 'Oversubscription, s of compute per s of flight',
  DR('Interceptor (terminal engagement)')['occupancy'], 17.3,
  'The Compute Requirements of Embodied Autonomy, Section 9'),
]
for chk, qty, mine, ref, src in VAL:
    put(ws, r, [chk, qty, mine, ref, mine/ref - 1, src],
        fonts={1: BLD, 3: BLD}, fmts={3: '#,##0.00', 4: '#,##0.00', 5: '+0.0%;-0.0%'},
        wrap=(2, 6)); r += 1

# SAE levels are checked as a band against a band: does the derived 5%-2% range
# overlap the range of silicon actually fielded at that level?
BANDS = [
 ('SAE L2 / L2+', AV('SAE L2 / L2+ (partial automation)'), 2.5, 144.0,
  'Mobileye EyeQ4 2.5 TOPS through Tesla HW3 144 TOPS, all fielded at L2/L2+'),
 ('SAE L3', AV('SAE L3 (conditional automation)'), 24.0, 254.0,
  'Mobileye EyeQ5 24 TOPS through NVIDIA Orin AGX 254 TOPS'),
 ('SAE L4 / L5', AV('SAE L4 / L5 (high / full automation)'), 254.0, 2000.0,
  'NVIDIA Orin AGX 254 TOPS through Thor ~2,000 TOPS; robotaxi stacks draw 1-2 kW'),
]
for lvl, x, lo, hi, src in BANDS:
    ok = not (x['np2'] < lo or x['np5'] > hi)
    put(ws, r, [lvl, 'Derived band vs fielded band, TOPS',
                f"{x['np5']:,.0f} - {x['np2']:,.0f}", f"{lo:,.0f} - {hi:,.0f}",
                'overlaps' if ok else 'NO OVERLAP', src],
        fonts={1: BLD, 3: BLD, 5: (BOD if ok else RED)}, wrap=(2, 6)); r += 1
c = ws.cell(row=r+1, column=1, value=
  'The drone rows are the only ones with an independent arithmetic reference, and the '
  'forward derivation reproduces both regimes within 3% without being fitted to them. '
  'The SAE rows bracket silicon fielded at each level. Neither check was used to set any '
  'unit cost.')
c.font = SUB; c.alignment = WRAP
ws.merge_cells(start_row=r+1, start_column=1, end_row=r+3, end_column=6)

# ---------------------------------------------------------------- 9 exposure
ws, r = sheet('Exposure register', [32, 15, 15, 10, 13, 62],
  ['Stage', 'Forward ops/call', 'Annex implies', 'Ratio',
   'Max share of any mission', 'What a reviewer should ask, and the answer'],
  note='Every stage where this derivation and the companion annex disagree by more than 2x.')
ANNEX = {'lidar': 210, 'sgm': 796262400, 'radar': 4e8, 'mono': 45, 'lio': 8000,
         'vio': 88, 'ba': 2.1e8, 'tsdf': 3900, 'esdf': 8e8, 'det': 9e10,
         'vlm': 2.66e12, 'vla': 3.4e11, 'gain': 7.7e7, 'graph': 2.5e8,
         'sdfenc': 2.5e10, 'policy': 1e10, 'mpc': 2.5e7, 'cbf': 5e6, 'ctrl': 1e6}
ASK = {
 'sgm': 'Q: why is the per-frame cost 8x what the annex implies? A: it is not a disagreement about '
        'total demand — the annex implies a cheaper frame at ~300 Hz, this model a fully '
        'itemized 128-disparity frame at 40 Hz. The products agree within 7%. The frame '
        'rate here is the defensible one.',
 'radar': 'Q: why 2.3x the annex per scan? A: aperture. This model states a 32-channel '
          '1024x256 MIMO radar at 20 Hz; the annex implies a smaller radar at a higher '
          'rate. Sustained demand differs by 6%. A 12-channel radar costs ~12x less and is '
          'used for the lighter profiles.',
 'gain': 'Q: gain evaluation is 177x cheaper here than in the annex. A: agreed, and this '
         'model uses the cheaper forward value. At 200 rays x 120 voxels a candidate '
         'evaluation is 0.43 MOP. It is under 0.2% of every mission total, so the '
         'difference does not move any conclusion.',
 'graph': 'Q: 16x cheaper. A: same answer. 2,000 samples with 50 collision queries per '
          'edge is 16 MOP. Under 0.05% of every mission total.',
 'mpc': 'Q: 22x cheaper than the annex. A: a 30-step Riccati recursion at nx=12 is 1.1 MOP; '
        'the annex implies a much larger problem. Under 0.02% of the drone totals. It '
        'matters only for the humanoid, where nx=60 raises it by 125x and it is modeled '
        'explicitly.',
 'cbf': 'Q: 172x cheaper. A: a 3-variable QP over 12 constraints is genuinely small; the '
        'cost is the ESDF window sweep, which is 20 kOP. Under 0.01% of every total. Its '
        'importance is its FP64 floor and 1 kHz rate, not its op count.',
 'ctrl': 'Q: 2,270x cheaper. A: a cascaded PID is a few hundred operations. The annex '
         'figure appears to include the whole flight-control task. It runs on the '
         'autopilot MCU, not the mission computer, and is under 0.01% of every total.',
}
shares = {}
for k in ANNEX:
    shares[k] = max((p.stages[k]['ops_s'] / (sum(s['ops_s'] for s in p.stages.values()))
                     if k in p.stages else 0) for p in PROFILES)
flag = 0
for d in DERIVS:
    k = d['key']; ratio = d['ops'] / ANNEX[k]
    if 0.5 <= ratio <= 2.0: continue
    flag += 1
    put(ws, r, [d['name'], d['ops'], ANNEX[k], ratio, shares[k], ASK.get(k, '')],
        fonts={1: BLD, 4: RED}, fmts={2: '#,##0', 3: '#,##0', 4: '#,##0.000"x"',
                                      5: '0.00%'}, wrap=(6,))
    ws.row_dimensions[r].height = 13 * (1 + len(ASK.get(k, ''))//78); r += 1
c = ws.cell(row=r+1, column=1, value=
  f'{flag} of {len(DERIVS)} stages disagree by more than 2x. Two of them (SGM, radar) are '
  'reparameterizations that leave sustained demand unchanged. The other five are all under '
  '0.2% of every mission total, and in every case this model is the CHEAPER of the two — '
  'so correcting them would raise the demand figures, not lower them. The stages that '
  'dominate every total (detection, VLM, VLA, state estimation, mapping) agree within 20%.')
c.font = SUB; c.alignment = WRAP
ws.merge_cells(start_row=r+1, start_column=1, end_row=r+4, end_column=6)


wb.save('BranesAI-Workload-Data-Annex.xlsx')
print('workbook written:', os.path.getsize('BranesAI-Workload-Data-Annex.xlsx'), 'bytes')
