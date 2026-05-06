#!/usr/bin/env bash
# ============================================================================
# Jetson Orin Nano 8GB -- V4 baseline capture runbook
#
# Run this ON the Jetson, not on the dev box. Wall time: ~40 minutes
# (~10 min per CSV; the validation sweeps have ~78 shapes each at 11
# trials each).
#
# Treat the steps as separate "checkpoints" -- run one at a time, verify
# the output before proceeding to the next. The script does NOT execute
# end-to-end with `bash` -- it's a runbook with copy-pasteable commands.
# ============================================================================

set -euo pipefail   # for any block you do choose to run as a script

# ----------------------------------------------------------------------------
# Step 1: Get the code and deps onto the Jetson
# ----------------------------------------------------------------------------
git clone https://github.com/branes-ai/graphs.git
cd graphs
# After PR #80 merges to main, drop this checkout. Until then:
git checkout feat/v4-4-gpu-jetson-ground-truth

# Verify torch sees CUDA. Use the JetPack-matched wheel -- DON'T `pip install
# torch` from PyPI (those wheels are CPU-only or x86-only).
python3 -c "import torch; print('torch:', torch.__version__,
  'cuda:', torch.cuda.is_available(),
  'device:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'none')"
# CHECKPOINT 1: cuda:True is required. If False:
#   https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/

pip install -e .

# ----------------------------------------------------------------------------
# Step 2: Verify INA3221 sysfs is readable
# ----------------------------------------------------------------------------
# Newer JetPack puts rails under /sys/bus/i2c/drivers/ina3221/.../hwmon/
ls /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label 2>/dev/null

# If that dir is missing OR you get permission denied:
sudo chmod -R a+r /sys/bus/i2c/drivers/ina3221/

# Sanity-check the probe detector picks up rails:
PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_jetson import _detect_ina3221
probe = _detect_ina3221()
if probe is None:
    print('NO RAILS DETECTED -- check sysfs paths / permissions')
else:
    print(f'backend={probe.backend}, rails={[r.name for r in probe.rails]}')
    print(f'total power now: {probe.read_total_power_mw()} mW')
"
# CHECKPOINT 2: must see rail names + a plausible mW reading (typically
# 2000-7000 mW for an idle Orin Nano). If you see a stale reading or
# only one rail, widen the rail_pattern in pytorch_jetson.py temporarily
# to inspect what's available, then update the default if needed.

# ----------------------------------------------------------------------------
# Step 3: Smoke test with --limit 3 (~30 seconds total)
# ----------------------------------------------------------------------------
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw jetson_orin_nano_8gb --op matmul --purpose validation --limit 3

# Inspect the smoke-test CSV:
cat validation/model_v4/results/baselines/jetson_orin_nano_8gb_matmul.csv
# CHECKPOINT 3: 3 rows; latency_ms sub-100ms for the small shapes;
# energy_mJ should be present (not 'n/a'). If you see 'INA3221: probe
# not available' in the notes column, go back to Step 2.

# Clear the smoke-test CSV before the full run (clean start is easier
# to reason about than per-key overwrites):
rm validation/model_v4/results/baselines/jetson_orin_nano_8gb_matmul.csv

# ----------------------------------------------------------------------------
# Step 4: Lock the chip at full clocks for the full capture
# ----------------------------------------------------------------------------
# Otherwise DVFS smears measurements across runs. Reverts on reboot.
sudo nvpmodel -m 0
sudo jetson_clocks

# ----------------------------------------------------------------------------
# Step 5: Full capture (~40 minutes total)
# ----------------------------------------------------------------------------
# Make the box otherwise idle. Close browsers, kill background CUDA work.
for op in matmul linear; do
  for purpose in calibration validation; do
    PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
        --hw jetson_orin_nano_8gb --op $op --purpose $purpose
  done
done

# OOM handling: the Orin Nano has only 8 GB shared CPU/GPU. The validation
# sweep includes shapes (e.g. 16384^3 fp16) that won't fit. The capture
# script prints which shape failed; note the list -- a follow-up will
# mark those UNSUPPORTED in _augment.py for orin_nano. Don't rerun in a
# way that hides the OOMs.

# ----------------------------------------------------------------------------
# Step 6: Sanity-check the CSVs
# ----------------------------------------------------------------------------
ls -la validation/model_v4/results/baselines/jetson_orin_nano_8gb_*.csv
wc -l validation/model_v4/results/baselines/jetson_orin_nano_8gb_*.csv
# Expect each CSV to be ~78 rows (18 cal + 60 val, minus skips/OOMs).

head -5 validation/model_v4/results/baselines/jetson_orin_nano_8gb_matmul.csv
# Spot-check: latency_ms grows with shape size; energy_mJ grows with
# latency (active power roughly constant ~5-15 W across the sweep).

# ----------------------------------------------------------------------------
# Step 7: Commit and push
# ----------------------------------------------------------------------------
JETPACK_VER=$(cat /etc/nv_tegra_release | head -1 | sed 's/.*R\([0-9]*\).*/L4T R\1/' || echo "unknown")
git checkout -b feat/v4-4-phase-b-jetson-orin-nano-baseline
git add validation/model_v4/results/baselines/jetson_orin_nano_8gb_*.csv
git commit -m "feat(validation): V4-4 Phase B -- Jetson Orin Nano 8GB baseline (matmul + linear)

Captured on ${JETPACK_VER} via PyTorchJetsonMeasurer (cudaEvent + INA3221).
Power mode: MAXN (nvpmodel -m 0), jetson_clocks locked.
"
git push -u origin feat/v4-4-phase-b-jetson-orin-nano-baseline

# Then open the PR from your dev box:
#   gh pr create --draft --title "feat(validation): V4-4 Phase B -- Jetson Orin Nano baseline"
# and request the floor-test follow-up.
