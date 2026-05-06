#!/usr/bin/env bash
# ============================================================================
# Jetson Orin Nano (Super) -- V5-2b vector_add baseline capture
#
# Captures the zero-reuse op across the cache hierarchy on Jetson Orin
# silicon with INA3221 power telemetry. Run from inside the cloned
# graphs repo. Wall time: ~5-30 seconds for both files combined
# (Jetson at fp16 is fast; the largest N=256M is ~1.5 GB working set).
#
# This runbook does NOT include the clone / install prelude -- the
# operator is assumed to be in a working repo with the JetPack-matched
# torch wheel installed and `pip install -e .` already done.
#
# Output: validation/model_v4/results/baselines/jetson_orin_nano_8gb_vector_add.csv
# (~22 rows: ~10 calibration + ~12 validation; entries are fp16 per
# the augmenter's KNOWN_TARGETS dtype, but a few may be skipped if
# memory pressure surfaces on the largest shape.)
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# Step 1: Verify CUDA + INA3221 are ready
# ----------------------------------------------------------------------------
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
    print(f'CUDA: {torch.cuda.get_device_name()}')"
# CHECKPOINT 1a: prints the iGPU name; if not, install JetPack-matched
# torch (see https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/).

ls /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label 2>/dev/null \
  || sudo chmod -R a+r /sys/bus/i2c/drivers/ina3221/

PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_jetson import _detect_ina3221
probe = _detect_ina3221()
if probe is None:
    print('NO RAILS DETECTED -- check sysfs paths / permissions')
else:
    print(f'backend={probe.backend}, rails={[r.name for r in probe.rails]}')
"
# CHECKPOINT 1b: must see rail names + a plausible backend.

# ----------------------------------------------------------------------------
# Step 2: Lock GPU clocks for consistent measurements
# ----------------------------------------------------------------------------
# The 15W profile on Orin Nano Super sustains near-boost clocks under
# cuBLAS load (per the post-#94 calibration), but vector add is fast
# enough that DVFS settling can dominate the smaller shapes. Lock
# clocks so the measurement at each N is comparable.
sudo nvpmodel -m 1   # 15W profile (matches the Phase B baseline conditions)
sudo jetson_clocks   # lock all clocks at the profile's max

# ----------------------------------------------------------------------------
# Step 3: Smoke test (largest shape only) before the full run
# ----------------------------------------------------------------------------
# Verify the largest fp16 vector_add (N=256M, ~1.5 GB working set) fits
# in Orin Nano's 8 GB shared memory before committing to the full sweep:
PYTHONPATH=src python3 -c "
import torch
from validation.model_v4.workloads.vector_add import build_vector_add
w = build_vector_add(268435456, 'fp16')
# Move to GPU explicitly to mirror what PyTorchJetsonMeasurer does (#88):
device = 'cuda:0'
inputs = tuple(x.to(device) for x in w.inputs)
model = w.model.to(device)
out = model(*inputs)
torch.cuda.synchronize()
print(f'smoke OK: shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}')
"
# CHECKPOINT 3: prints 'smoke OK: ... device=cuda:0'. OOM here means
# the largest shape won't fit; you can --limit the capture to skip it.

# ----------------------------------------------------------------------------
# Step 4: Full capture (~5-30 s wall time)
# ----------------------------------------------------------------------------
# Both purposes share one CSV (cache key = (hardware, op)).
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw jetson_orin_nano_8gb --op vector_add --purpose calibration

PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw jetson_orin_nano_8gb --op vector_add --purpose validation

# OOM handling: if the largest shapes fail, rerun with --limit to skip
# them and note in the commit message which N values are missing. The
# Orin Nano sweep was generated at fp16, so the per-tensor footprint
# is 2 bytes/elem -- N=256M = 512 MB/tensor x 3 tensors = 1.5 GB. That
# should fit on Orin's 8 GB shared, but background processes can push
# it over the edge.

# ----------------------------------------------------------------------------
# Step 5: Sanity-check the CSV
# ----------------------------------------------------------------------------
CSV=validation/model_v4/results/baselines/jetson_orin_nano_8gb_vector_add.csv
ls -la "$CSV"
wc -l "$CSV"
# Expect: ~12-23 lines (1 header + 11+ data rows, depending on whether
# both purposes' fp16 entries land in the file).

# Spot-check tier transitions:
PYTHONPATH=src python3 -c "
import csv
rows = list(csv.DictReader(open('$CSV')))
print(f'{\"N\":>10s} {\"WS_KB\":>8s} {\"latency_us\":>12s} {\"GB/s\":>8s}')
for r in sorted(rows, key=lambda r: int(r['shape'])):
    N = int(r['shape']); ws = 3 * N * 2  # fp16
    lat = float(r['latency_s'])
    bw = ws / lat / 1e9 if lat > 0 else 0
    print(f'{N:>10d} {ws/1024:>8.1f} {lat*1e6:>12.2f} {bw:>8.1f}')
"
# Expect: BW rises with N as the launch / kernel-dispatch floor stops
# dominating, peaks in L2-resident shapes (Orin L2 = 2 MB, so peak
# residency is around N=300K), then plateaus at the iGPU's effective
# DRAM BW (Orin Nano Super peak = 102 GB/s; achievable typically
# 50-70 GB/s for streaming).

# ----------------------------------------------------------------------------
# Step 6: Restore default clocks (optional)
# ----------------------------------------------------------------------------
# nvpmodel resets on reboot; jetson_clocks doesn't. Restore if you'll
# use the box for other work afterwards:
# sudo jetson_clocks --restore /tmp/orin-l4t-clocks-default || true

# ----------------------------------------------------------------------------
# Step 7: Commit
# ----------------------------------------------------------------------------
git checkout -b feat/v5-2b-jetson-orin-nano-vector-add-baseline
git add validation/model_v4/results/baselines/jetson_orin_nano_8gb_vector_add.csv
git commit -m "feat(validation): V5-2b -- Jetson Orin Nano vector_add baseline

Captured via PyTorchJetsonMeasurer (cudaEvent + INA3221) at the 15W
thermal profile (matches the Phase B baseline conditions per #94).
"
git push -u origin feat/v5-2b-jetson-orin-nano-vector-add-baseline

# Then open the PR from your dev box:
#   gh pr create --draft --title "feat(validation): V5-2b -- Jetson Orin Nano vector_add baseline"
