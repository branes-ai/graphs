#!/usr/bin/env bash
# ============================================================================
# i7-12700K -- V5-2b vector_add baseline capture
#
# Captures the zero-reuse op across the cache hierarchy on a Linux x86
# host with Intel RAPL energy telemetry. Run from inside the cloned
# graphs repo. Wall time: ~5 seconds for both files combined.
#
# This runbook does NOT include the clone / install prelude -- the
# operator is assumed to be in a working repo with `pip install -e .`
# already done.
#
# Output: validation/model_v4/results/baselines/i7_12700k_vector_add.csv
# (11 rows: 6 calibration + 5 validation, sharing one CSV per the
# cache-key convention `(hardware, op)`.)
# ============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# Step 1: Verify RAPL is readable
# ----------------------------------------------------------------------------
# On most Intel CPUs ``/sys/class/powercap/intel-rapl:0/energy_uj`` is
# group-readable, but some hardened distros lock it to root.
ls /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null \
  || { echo "RAPL not readable; energy column will be 'n/a'. To enable:"; \
       echo "  sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj"; }

# Sanity-check the probe detector picks up RAPL:
PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_cpu import _detect_rapl
probe = _detect_rapl()
print(f'RAPL: {\"OK\" if probe else \"unavailable; latency-only capture\"}')
"
# CHECKPOINT 1: should print 'RAPL: OK'. If not, the energy column will
# come back zero/n/a but latency capture still works.

# ----------------------------------------------------------------------------
# Step 2: Suppress system load (recommended)
# ----------------------------------------------------------------------------
# Vector add at the DRAM plateau is sensitive to memory-controller
# contention. Close browsers / kill background CUDA / build jobs.
# Optional: pin to performance governor + disable boost for stability.
#
#   sudo cpupower frequency-set -g performance
#
# (Skip if you want post-boost steady-state numbers; the V4-3g i7 baseline
# was captured WITHOUT this, so vector_add measurements stay comparable.)

# ----------------------------------------------------------------------------
# Step 3: Smoke test (largest shape only) before the full run
# ----------------------------------------------------------------------------
# The largest shape (N=256M) allocates 3 GB of fp32 tensors. Verify
# memory-allocator paths work before committing to the whole sweep:
PYTHONPATH=src python3 -c "
from validation.model_v4.workloads.vector_add import build_vector_add
w = build_vector_add(268435456, 'fp32')
out = w.model(*w.inputs)
print(f'smoke OK: shape={tuple(out.shape)}, dtype={out.dtype}')
"
# CHECKPOINT 3: should print 'smoke OK: shape=(268435456,), dtype=torch.float32'.
# Failure here is usually OOM (need >3 GB free) or a torch import problem.

# ----------------------------------------------------------------------------
# Step 4: Full capture (~5 s wall time)
# ----------------------------------------------------------------------------
# Both purposes share one CSV (cache key = (hardware, op)).
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw i7_12700k --op vector_add --purpose calibration

PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw i7_12700k --op vector_add --purpose validation

# ----------------------------------------------------------------------------
# Step 5: Sanity-check the CSV
# ----------------------------------------------------------------------------
ls -la validation/model_v4/results/baselines/i7_12700k_vector_add.csv
wc -l validation/model_v4/results/baselines/i7_12700k_vector_add.csv
# Expect: 12 lines (1 header + 11 data rows: N from 256 to 268M).

# Spot-check tier transitions visually:
PYTHONPATH=src python3 -c "
import csv
rows = list(csv.DictReader(open('validation/model_v4/results/baselines/i7_12700k_vector_add.csv')))
print(f'{\"N\":>10s} {\"WS_KB\":>8s} {\"latency_us\":>12s} {\"GB/s\":>8s}')
for r in sorted(rows, key=lambda r: int(r['shape'])):
    N = int(r['shape']); ws = 3 * N * 4
    lat = float(r['latency_s'])
    bw = ws / lat / 1e9 if lat > 0 else 0
    print(f'{N:>10d} {ws/1024:>8.1f} {lat*1e6:>12.2f} {bw:>8.1f}')
"
# Expect: GB/s rises through L1/L2 (sub-µs latency floor dominates),
# peaks in LLC residency (often >100 GB/s due to cache hits exceeding
# peak DRAM 75 GB/s), then plateaus at ~35-40 GB/s on DRAM-bound shapes.

# ----------------------------------------------------------------------------
# Step 6: Commit
# ----------------------------------------------------------------------------
# Idempotent branch step: switch to the branch if it already exists (e.g. on
# a rerun after a partial capture), otherwise create it. We deliberately do
# NOT use `git checkout -B` -- that would reset any prior commits on the
# branch (e.g. an earlier partial capture you intend to amend or build on).
git checkout feat/v5-2b-i7-vector-add-baseline 2>/dev/null \
  || git checkout -b feat/v5-2b-i7-vector-add-baseline
git add validation/model_v4/results/baselines/i7_12700k_vector_add.csv
git commit -m "feat(validation): V5-2b -- i7-12700K vector_add baseline

Captured via PyTorchCPUMeasurer (wall-clock + Intel RAPL).
11 rows spanning N=256 to N=268M (1 KB to 3 GB working set).
"
git push -u origin feat/v5-2b-i7-vector-add-baseline
