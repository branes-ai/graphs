#!/usr/bin/env bash
# ============================================================================
# Ryzen AI NUC -- V4 baseline capture runbook
#
# Run this ON the NUC, not on the dev box. Three execution backends share
# one physical package: Zen 4/5 cores (CPU), Radeon iGPU (RDNA 3.5/4),
# and XDNA NPU (NPU1 on Hawk Point, NPU2 on Strix Point/Halo).
#
# This runbook covers all three; pick the section(s) that match your
# capture goal. The Jetson runbook precedent is one runbook per physical
# box because CPU + GPU share package power and the operator typically
# wants to characterize the full SoC.
#
# Total wall time: ~25-45 min depending on which backends run.
#
# Treat the steps as separate "checkpoints" -- run one at a time, verify
# the output before proceeding. The script does NOT execute end-to-end
# with `bash` -- it's a runbook with copy-pasteable commands.
# ============================================================================
set -euo pipefail

# ============================================================================
# Phase A prerequisites (CODE THAT MUST LAND BEFORE THIS RUNBOOK CAN RUN)
# ============================================================================
#
# Tracking issue: #86 -- "V4-4 Phase A2: Ryzen AI NUC measurer + mapper
# support (CPU + Radeon iGPU + XDNA NPU)"
#
# Unlike the Jetson/H100 runbooks which shipped alongside their measurer
# code (PR #80), the Ryzen AI NUC support is *not yet* in the harness.
# The following must land in a separate PR (V4-4 Phase A2) before this
# runbook can do anything beyond Step 1. Full file list and design
# notes live in #86; abbreviated summary below:
#
# 1. Hardware mappers (registry):
#    src/graphs/hardware/mappers/cpu.py
#      - create_ryzen_ai_9_hx_370_mapper      (Strix Point, Zen 5, NPU 50 TOPS)
#      - create_ryzen_7_8840hs_mapper         (Hawk Point, Zen 4, NPU 16 TOPS)
#      - create_ryzen_ai_max_plus_395_mapper  (Strix Halo, Zen 5, NPU 50 TOPS)
#      Each captures: P-core/E-core counts, AVX2/AVX-512 SIMD, L1/L2/L3,
#      DDR5/LPDDR5X DRAM BW, integrated graphics block reference.
#
#    src/graphs/hardware/mappers/gpu.py
#      - create_radeon_780m_mapper            (RDNA 3, 12 CUs, Hawk Point)
#      - create_radeon_890m_mapper            (RDNA 3.5, 16 CUs, Strix Point)
#      - create_radeon_8060s_mapper           (RDNA 3.5, 40 CUs, Strix Halo)
#      Each captures: CU count, peak FP32/FP16, shared memory hierarchy,
#      ROCm precision profiles.
#
#    src/graphs/hardware/mappers/accelerators/xdna.py (new file)
#      - create_xdna1_npu_mapper              (Phoenix/Hawk Point, ~16 TOPS INT8)
#      - create_xdna2_npu_mapper              (Strix Point/Halo, ~50 TOPS INT8)
#      Heterogeneous tile architecture similar in shape to KPU but with
#      different scratchpad/dataflow semantics. Per the v4 plan, the NPU
#      may need to be CONSISTENCY-ONLY validated (Principle 2) until
#      simulator output is available.
#
# 2. Ground-truth measurers (validation/model_v4/ground_truth/):
#    pytorch_amd_cpu.py
#      - PyTorchAMDCPUMeasurer: wall-clock + amd_energy sysfs
#        (/sys/devices/platform/amd_energy.0/hwmon/hwmonN/energy*_input)
#        Some Zen 4+ kernels also expose Intel-style RAPL at
#        /sys/class/powercap/intel-rapl:0/ -- the probe should try both
#        paths and report which one succeeded in notes.
#
#    pytorch_rocm.py
#      - PyTorchROCmMeasurer: hipEvent timing + amdgpu sysfs power
#        (/sys/class/drm/cardN/device/hwmon/hwmonM/power1_average in uW)
#        Note: ROCm support on consumer Radeon (RDNA 3+) is fragile;
#        the measurer must surface "ROCm not available" cleanly when
#        torch.cuda.is_available() returns False on a non-NVIDIA system.
#
#    onnx_xdna.py
#      - OnnxXdnaMeasurer: ONNX Runtime with Vitis AI / Ryzen AI EP
#        + xrt-smi telemetry. Different shape from the PyTorch-based
#        measurers; takes an exported ONNX model rather than nn.Module.
#        May share a thin adapter so the V4 harness can drive it via
#        the same ground_truth.base.Measurer Protocol.
#
# 3. Plumbing:
#    validation/model_v4/sweeps/_augment.py KNOWN_TARGETS
#    validation/model_v4/harness/runner.py SWEEP_HW_TO_MAPPER
#    validation/model_v4/cli/capture_ground_truth.py _MEASURER_FACTORY
#    tests/validation_model_v4/test_sweeps.py hw fixture
#    Each must learn the new hardware keys (ryzen_ai_9_hx_370,
#    ryzen_7_8840hs, ryzen_ai_max_plus_395, radeon_780m, radeon_890m,
#    radeon_8060s, xdna1_npu, xdna2_npu).
#
# 4. Augment committed sweep JSONs with the new keys (run
#    `python -m validation.model_v4.sweeps._augment --all` after the
#    KNOWN_TARGETS update).
#
# Until Phase A2 (#86) lands, only Step 1 ("preflight: identify which
# NUC you have") can run -- the actual capture commands will fail with
# "No ground-truth measurer registered for ryzen_ai_*".
# ============================================================================


# ----------------------------------------------------------------------------
# Step 1: Identify the NUC (always works; no Phase A2 dependency)
# ----------------------------------------------------------------------------
# Determine which Ryzen AI variant you have so you know which hardware
# key + measurer backend to use:

# CPU model (look at "Model name:" line)
lscpu | grep -E "Model name|Vendor ID"

# iGPU model (look for "AMD/ATI" in the rendering line)
lspci -v | grep -A 2 -E "VGA|3D|Display"

# NPU presence (XDNA driver loaded?)
lsmod | grep -E "amdxdna|xdna" || echo "No XDNA driver loaded (NPU access requires driver install)"

# JetPack-style "where am I" detection isn't standard on Linux x86;
# parse the CPU model to derive the hardware key:
#   "Ryzen 7 8840HS" / "8845HS"        -> ryzen_7_8840hs
#   "Ryzen AI 9 HX 370"                -> ryzen_ai_9_hx_370
#   "Ryzen AI MAX+ 395"                -> ryzen_ai_max_plus_395
# Match the iGPU integer to:
#   780M -> radeon_780m
#   890M -> radeon_890m
#   8060S (Strix Halo only) -> radeon_8060s
# CHECKPOINT 1: Note the three keys you'll feed to capture_ground_truth.

# Capture one for use in later steps:
NUC_CPU_KEY=ryzen_7_8840hs       # adjust per Step 1 detection
NUC_GPU_KEY=radeon_780m          # adjust per Step 1 detection
NUC_NPU_KEY=xdna1_npu            # adjust per Step 1 detection

# ----------------------------------------------------------------------------
# Step 2: Get the code and deps onto the NUC
# ----------------------------------------------------------------------------
git clone https://github.com/branes-ai/graphs.git
cd graphs
# After Phase A2 merges to main, drop this checkout. Until then,
# check out the Phase A2 branch (TBD).
# git checkout feat/v4-4-phase-a2-amd-measurers

# Vanilla CPU torch is fine for the CPU + NPU paths. ROCm torch is
# needed for the iGPU path; install only if you plan to capture iGPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# Verify the harness imports the (forthcoming) AMD CPU measurer:
PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_amd_cpu import PyTorchAMDCPUMeasurer
print('AMD CPU measurer importable')
" || echo "PHASE A2 NOT LANDED: pytorch_amd_cpu.py missing -- skip CPU capture"

# ============================================================================
# Backend 1: Zen 4/5 CPU side via amd_energy sysfs
# ============================================================================

# ----------------------------------------------------------------------------
# Step 3 (CPU): Verify amd_energy sysfs is readable
# ----------------------------------------------------------------------------
# Newer kernels expose AMD package energy at:
ls /sys/devices/platform/amd_energy.0/hwmon/hwmon*/energy*_input 2>/dev/null \
  || echo "amd_energy not loaded; try: sudo modprobe amd_energy"

# Some Zen 4+ systems also expose Intel-style RAPL even on AMD silicon:
ls /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null \
  && echo "RAPL fallback available" \
  || echo "no RAPL fallback (this is expected on most AMD systems)"

# If energy* files are missing entirely, run as root once:
sudo modprobe amd_energy
ls /sys/devices/platform/amd_energy.0/hwmon/hwmon*/energy*_input

# Sanity-check the probe detector (Phase A2):
PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_amd_cpu import _detect_amd_energy
probe = _detect_amd_energy()
if probe is None:
    print('NO ENERGY PROBE -- amd_energy + RAPL fallback both unavailable')
else:
    print(f'backend={probe.backend}, current uJ={probe.read_energy_uj()}')
"
# CHECKPOINT 3: must see a backend tag and a plausible energy reading
# (the counter is monotonically increasing, in microjoules).

# ----------------------------------------------------------------------------
# Step 4 (CPU): Smoke test
# ----------------------------------------------------------------------------
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw $NUC_CPU_KEY --op matmul --purpose validation --limit 3

cat validation/model_v4/results/baselines/${NUC_CPU_KEY}_matmul.csv
# CHECKPOINT 4: 3 rows, latency_ms sub-100ms, energy_mJ present.

rm validation/model_v4/results/baselines/${NUC_CPU_KEY}_matmul.csv

# ----------------------------------------------------------------------------
# Step 5 (CPU): Lock the package power for consistent measurements
# ----------------------------------------------------------------------------
# Disable boost so the CPU runs at sustained base frequency:
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
# Set the performance governor:
sudo cpupower frequency-set -g performance

# OR, if you want to model the post-boost steady state, leave boost on
# but pin the workload to one core via taskset (and update the CPU
# mapper's `compute_units=1` for that capture).

# ----------------------------------------------------------------------------
# Step 6 (CPU): Full capture (~10-15 min depending on SKU)
# ----------------------------------------------------------------------------
for op in matmul linear; do
  for purpose in calibration validation; do
    PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
        --hw $NUC_CPU_KEY --op $op --purpose $purpose
  done
done

# ============================================================================
# Backend 2: Radeon iGPU via ROCm + amdgpu sysfs
# ============================================================================

# ----------------------------------------------------------------------------
# Step 7 (iGPU): Install ROCm torch
# ----------------------------------------------------------------------------
# Skip if you don't plan to capture iGPU. ROCm support on consumer
# Radeon (RDNA 3+) is fragile; expect setup pain.
#
# Per AMD's ROCm install guide for Ubuntu 24.04+:
sudo apt install -y rocm-dev rocm-libs rocm-utils
# Then a ROCm-specific torch wheel:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4

python3 -c "import torch; print('cuda(rocm):', torch.cuda.is_available(),
                                   'device:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'none')"
# CHECKPOINT 7: cuda is True and device names the Radeon iGPU.
# Note: PyTorch ROCm reports HIP devices as "cuda" by convention.

# ----------------------------------------------------------------------------
# Step 8 (iGPU): Verify amdgpu power sysfs
# ----------------------------------------------------------------------------
ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_average 2>/dev/null \
  || ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_input 2>/dev/null \
  || echo "amdgpu hwmon power node missing; check rocm install"

# Also useful for spot-checking:
rocm-smi --showpower

PYTHONPATH=src python3 -c "
from validation.model_v4.ground_truth.pytorch_rocm import _detect_amdgpu
probe = _detect_amdgpu()
if probe is None:
    print('NO AMDGPU PROBE')
else:
    print(f'card={probe.card_index}, current power mW={probe.read_power_mw()}')
"
# CHECKPOINT 8: must see plausible power reading (1-30 W idle for iGPU).

# ----------------------------------------------------------------------------
# Step 9 (iGPU): Smoke test + full capture (~5-10 min)
# ----------------------------------------------------------------------------
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw $NUC_GPU_KEY --op matmul --purpose validation --limit 3

cat validation/model_v4/results/baselines/${NUC_GPU_KEY}_matmul.csv
rm validation/model_v4/results/baselines/${NUC_GPU_KEY}_matmul.csv

for op in matmul linear; do
  for purpose in calibration validation; do
    PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
        --hw $NUC_GPU_KEY --op $op --purpose $purpose
  done
done

# ============================================================================
# Backend 3: XDNA NPU via ONNX Runtime + xrt-smi
# ============================================================================

# ----------------------------------------------------------------------------
# Step 10 (NPU): Install Ryzen AI Software stack
# ----------------------------------------------------------------------------
# Skip if you don't plan to capture NPU. The NPU path is fundamentally
# different from CPU/iGPU: it goes through ONNX Runtime with the
# Vitis AI / Ryzen AI execution provider, not torch.
#
# Per AMD's Ryzen AI install guide:
#   https://ryzenai.docs.amd.com/en/latest/inst.html
# (URL stable as of 2026-05; verify current path before running)
#
# 1. Install xrt + xrt-amdxdna driver (Linux: kernel >= 6.10 with
#    in-tree amdxdna; Windows: pre-built driver from AMD site).
# 2. Install onnxruntime + ryzen-ai EP wheel.

python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('available providers:', providers)
assert any('VitisAI' in p or 'RyzenAI' in p for p in providers), (
    'Ryzen AI EP not registered with onnxruntime')
"
# CHECKPOINT 10: 'VitisAIExecutionProvider' or similar must be present.

# ----------------------------------------------------------------------------
# Step 11 (NPU): Verify xrt-smi reports the NPU
# ----------------------------------------------------------------------------
xrt-smi examine
# Should list one device with type "NPU" and a partition for the
# specific XDNA generation (XDNA1 = Phoenix/Hawk Point, XDNA2 =
# Strix Point/Halo).

# Power telemetry on XDNA is driver-version-dependent; as of late 2025
# the rails appear under:
ls /sys/bus/pci/drivers/amdxdna/*/hwmon/hwmon*/ 2>/dev/null \
  || ls /sys/class/amdxdna/*/power 2>/dev/null \
  || echo "XDNA power telemetry path unknown; xrt-smi tools may have to suffice"

# ----------------------------------------------------------------------------
# Step 12 (NPU): Smoke test
# ----------------------------------------------------------------------------
# The NPU measurer (Phase A2) takes an ONNX model rather than a torch
# nn.Module, so the workload builder needs to export to ONNX first.
# The capture CLI handles this transparently when --hw is an XDNA key.
PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
    --hw $NUC_NPU_KEY --op matmul --purpose validation --limit 3 \
    --onnx-export-cache /tmp/v4-onnx-cache

# CHECKPOINT 12: 3 rows in the CSV; expect latency_ms in tens of
# microseconds for small INT8 matmul (XDNA's sweet spot).
# If energy_mJ is 'n/a' with a "no power telemetry" note, that's
# acceptable for this pass -- file a follow-up to track the kernel /
# xrt-smi path that exposes it.

rm validation/model_v4/results/baselines/${NUC_NPU_KEY}_matmul.csv

# ----------------------------------------------------------------------------
# Step 13 (NPU): Full capture (~15-20 min)
# ----------------------------------------------------------------------------
for op in matmul linear; do
  for purpose in calibration validation; do
    PYTHONPATH=src python3 -m validation.model_v4.cli.capture_ground_truth \
        --hw $NUC_NPU_KEY --op $op --purpose $purpose \
        --onnx-export-cache /tmp/v4-onnx-cache
  done
done

# ============================================================================
# Step 14: Sanity-check + commit
# ============================================================================
ls -la validation/model_v4/results/baselines/${NUC_CPU_KEY}_*.csv \
       validation/model_v4/results/baselines/${NUC_GPU_KEY}_*.csv \
       validation/model_v4/results/baselines/${NUC_NPU_KEY}_*.csv

wc -l validation/model_v4/results/baselines/${NUC_CPU_KEY}_*.csv \
      validation/model_v4/results/baselines/${NUC_GPU_KEY}_*.csv \
      validation/model_v4/results/baselines/${NUC_NPU_KEY}_*.csv

KERNEL_VER=$(uname -r)
ROCM_VER=$(rocm-smi --version 2>/dev/null | head -1 || echo "rocm not installed")
RYZEN_AI_VER=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "ort not installed")

git checkout -b feat/v4-baseline-ryzen-ai-nuc-${NUC_CPU_KEY}
git add validation/model_v4/results/baselines/${NUC_CPU_KEY}_*.csv \
        validation/model_v4/results/baselines/${NUC_GPU_KEY}_*.csv \
        validation/model_v4/results/baselines/${NUC_NPU_KEY}_*.csv
git commit -m "feat(validation): V4 baselines -- Ryzen AI NUC ($NUC_CPU_KEY + $NUC_GPU_KEY + $NUC_NPU_KEY)

CPU side captured via PyTorchAMDCPUMeasurer (wall-clock + amd_energy sysfs).
iGPU captured via PyTorchROCmMeasurer (hipEvent + amdgpu hwmon).
NPU captured via OnnxXdnaMeasurer (ONNX Runtime VitisAI EP + xrt-smi).

Kernel: ${KERNEL_VER}
ROCm:   ${ROCM_VER}
RyzenAI/ort: ${RYZEN_AI_VER}
Power: cpufreq-boost off, performance governor.
"
git push -u origin feat/v4-baseline-ryzen-ai-nuc-${NUC_CPU_KEY}

# Then open the PR from your dev box:
#   gh pr create --draft --title "feat(validation): V4 baselines -- Ryzen AI NUC"
# and request the floor-test follow-up.
