#!/usr/bin/env bash
# ============================================================================
# H100 SXM5 80GB -- V4 baseline capture runbook
#
# Run this ON the H100 host (rented cloud instance, on-prem server, ...),
# not on the dev box. Wall time: ~2 minutes total. The H100 is fast enough
# that most of the sweep is sub-millisecond -- a fundamentally different
# regime from the Jetson runbook.
#
# Treat the steps as separate "checkpoints" -- run one at a time, verify
# the output before proceeding. The script does NOT execute end-to-end
# with `bash` -- it's a runbook with copy-pasteable commands.
# ============================================================================

set -euo pipefail   # for any block you do choose to run as a script

# ----------------------------------------------------------------------------
# Step 1: Get the code and deps onto the H100 host
# ----------------------------------------------------------------------------
git clone https://github.com/branes-ai/graphs.git
cd graphs
# After PR #80 merges to main, drop this checkout. Until then:
git checkout feat/v4-4-gpu-jetson-ground-truth

# H100 hosts typically have a CUDA-enabled torch already (cloud images,
# nvidia/pytorch container). Verify:
python3 -c "import torch; print('torch:', torch.__version__,
  'cuda:', torch.cuda.is_available(),
  'device:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'none')"
# CHECKPOINT 1: cuda:True and device should report 'NVIDIA H100' (or similar).
# If you're on a multi-GPU host, set CUDA_VISIBLE_DEVICES=0 to pin.

pip install -e . pynvml

# ----------------------------------------------------------------------------
# Step 2: Verify NVML is queryable
# ----------------------------------------------------------------------------
nvidia-smi   # confirms the driver is loaded and reports the GPU(s)

# Sanity-check the probe detector picks the right backend (total-energy
# preferred, power-fallback acceptable):
PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_cuda import _detect_nvml
probe = _detect_nvml()
if probe is None:
    print('NVML probe FAILED -- pynvml missing, driver mismatch, or both queries unsupported')
else:
    print(f'name={probe.name!r}, total_energy={probe.supports_total_energy}')
    print(f'power now: {probe.read_power_mw()} mW')
    if probe.supports_total_energy:
        print(f'energy counter now: {probe.read_energy_mj()} mJ')
"
# CHECKPOINT 2: name should match your GPU; total_energy=True is preferred
# (H100 supports it). Power should be ~50-100 W idle.

# ----------------------------------------------------------------------------
# Step 3: Smoke test with --limit 3 (~5 seconds total)
# ----------------------------------------------------------------------------
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw h100_sxm5_80gb --op matmul --purpose validation --limit 3

# Inspect the smoke-test CSV:
cat validation/model_v4/results/baselines/h100_sxm5_80gb_matmul.csv
# CHECKPOINT 3: 3 rows. latency_ms will be VERY small (microseconds to a
# few ms), energy_mJ should still be present. If energy is 'n/a', go back
# to Step 2 -- the power-fallback path is acceptable but the total-energy
# path is preferred for H100.

# Clear the smoke-test CSV before the full run:
rm validation/model_v4/results/baselines/h100_sxm5_80gb_matmul.csv

# ----------------------------------------------------------------------------
# Step 4: Lock the GPU clocks for consistent measurements
# ----------------------------------------------------------------------------
# Disable autoboost / lock to a fixed clock pair. Skip this on shared cloud
# instances if you don't have permission -- the kernel-level cudaEvent
# timings are still valid, just noisier between trials.
sudo nvidia-smi -pm 1                                    # persistence mode on
sudo nvidia-smi -ac 2619,1980                            # H100 SXM5 default mem,gfx
# (If the above fails, fall back to: sudo nvidia-smi -lgc 1980)

# ----------------------------------------------------------------------------
# Step 5: Full capture (~2 minutes total)
# ----------------------------------------------------------------------------
# Make the GPU otherwise idle. Confirm with nvidia-smi -- "Volatile GPU-Util"
# should be 0% before you start.
for op in matmul linear; do
  for purpose in calibration validation; do
    PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
        --hw h100_sxm5_80gb --op $op --purpose $purpose
  done
done

# Unlike Orin Nano, H100 has 80 GB so OOMs are unlikely on the current
# sweep shapes. If anything fails, capture the error -- it might surface
# a real bug in the workload builder for very large shapes.

# ----------------------------------------------------------------------------
# Step 6: Sanity-check the CSVs
# ----------------------------------------------------------------------------
ls -la validation/model_v4/results/baselines/h100_sxm5_80gb_*.csv
wc -l validation/model_v4/results/baselines/h100_sxm5_80gb_*.csv
# Expect each CSV to be ~78 rows (18 cal + 60 val).

head -5 validation/model_v4/results/baselines/h100_sxm5_80gb_matmul.csv
# Spot-check: latency_ms is small (sub-ms for most shapes); energy_mJ
# correlates with latency at ~300-700 W average power (H100 SXM5 TDP).

# Note: many H100 shapes will be sub-1ms, which means the V4 harness will
# skip pass_energy assertion for them per the #71 RAPL/NVML noise floor.
# That's expected; the floor tests for H100 will need to be calibrated to
# whatever fraction of shapes ARE above the threshold.

# ----------------------------------------------------------------------------
# Step 7: Restore default clocks (optional cleanup)
# ----------------------------------------------------------------------------
sudo nvidia-smi -rac           # reset to default app clocks
# Skip if you'll keep using the box for V4 work.

# ----------------------------------------------------------------------------
# Step 8: Commit and push
# ----------------------------------------------------------------------------
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
git checkout -b feat/v4-4-phase-b-h100-baseline
git add validation/model_v4/results/baselines/h100_sxm5_80gb_*.csv
git commit -m "feat(validation): V4-4 Phase B -- H100 SXM5 80GB baseline (matmul + linear)

Captured via PyTorchCUDAMeasurer (cudaEvent + NVML total-energy).
Driver ${DRIVER_VER}, locked clocks (mem=2619, gfx=1980), persistence mode on.
"
git push -u origin feat/v4-4-phase-b-h100-baseline

# Then open the PR from your dev box:
#   gh pr create --draft --title "feat(validation): V4-4 Phase B -- H100 baseline"
# and request the floor-test follow-up.
