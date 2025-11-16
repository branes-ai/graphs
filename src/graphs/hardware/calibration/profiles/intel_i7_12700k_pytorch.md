Platform Validation:
  Current architecture: x86_64
  Expected architecture: x86_64
  ✓ Platform validation passed


================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CPU
  Actual device:    CPU
  Framework:        PYTORCH

================================================================================
Hardware Calibration: Intel-i7-12700K
================================================================================

System Information:
  CPU: x86_64
  Cores: 12 physical, 20 logical
  Memory: 31.1 GB
  Python: 3.11.14
  NumPy: 2.2.6
  PyTorch: 2.7.1+cu126

Execution Device:
  Running on: CPU
  Framework:  PYTORCH
              (PyTorch DL framework, GPU-accelerated)

Running calibration benchmarks...

1. Memory Bandwidth
--------------------------------------------------------------------------------
Framework: PyTorch 2.7.1+cu126
  Device: CPU

Calibrating memory copy 64 MB... 47.5 GB/s (63.4% efficiency)
Calibrating memory copy 128 MB... 45.7 GB/s (61.0% efficiency)
Calibrating memory copy 256 MB... 46.9 GB/s (62.5% efficiency)
Calibrating memory copy 512 MB... 46.0 GB/s (61.3% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.7.1+cu126
  Device: CPU

Calibrating matmul 256×256 across 5 precisions...
  fp64    ... ✓   234.1 GFLOPS (   0.1ms)  65.0% eff
  fp32    ... ✓   455.7 GFLOPS (   0.1ms)  63.3% eff
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'
Calibrating matmul 1024×1024 across 5 precisions...
  fp64    ... ✓   253.0 GFLOPS (   8.5ms)  70.3% eff
  fp32    ... ✓   563.5 GFLOPS (   3.8ms)  78.3% eff
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'
Calibrating matmul 2048×2048 across 5 precisions...
  fp64    ... ✓   441.9 GFLOPS (  38.9ms) 122.8% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, GPU boost clocks, or conservative theoretical peak
  fp32    ... ✓   870.3 GFLOPS (  19.7ms) 120.9% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, GPU boost clocks, or conservative theoretical peak
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'
Calibrating matmul 4096×4096 across 5 precisions...
  fp64    ... ✓   539.3 GFLOPS ( 254.9ms) 149.8% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, GPU boost clocks, or conservative theoretical peak
  fp32    ... ✓   921.0 GFLOPS ( 149.2ms) 127.9% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, GPU boost clocks, or conservative theoretical peak
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'

Building precision capability matrix...

Calibration saved to: /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k_pytorch.json

================================================================================
Hardware Calibration: Intel-i7-12700K
Date: 2025-11-16T09:49:27.540541
================================================================================

Framework: PYTORCH
Device:    CPU

Theoretical Specifications:
  Peak GFLOPS (FP32): 720.0
  Peak Bandwidth:     75.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=128       45.7 GB/s       61.0%
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=256       46.9 GB/s       62.5%
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=512       46.0 GB/s       61.3%
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=64       47.5 GB/s       63.4%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp32       256×256           0.1ms      455.7      455.7      455.7       63.3%
  fp32       1024×1024         3.8ms      563.5      563.5      563.5       78.3%
  fp32       2048×2048        19.7ms      870.3      870.3      870.3      120.9%  ⚠ ABOVE THEORETICAL
  fp32       4096×4096       149.2ms      921.0      921.0      921.0      127.9%  ⚠ ABOVE THEORETICAL
  fp64       256×256           0.1ms      234.1      234.1      234.1       65.0%
  fp64       1024×1024         8.5ms      253.0      253.0      253.0       70.3%
  fp64       2048×2048        38.9ms      441.9      441.9      441.9      122.8%  ⚠ ABOVE THEORETICAL
  fp64       4096×4096       254.9ms      539.3      539.3      539.3      149.8%  ⚠ ABOVE THEORETICAL
  int16      256×256               -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      256×256               -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       256×256               -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)

Precision Support Summary:
  Supported:   fp64, fp32, fp16, bf16, int32, int16, int8
  Unsupported: fp8_e4m3, fp8_e5m2

Note on >100% Efficiency:
  Efficiency above theoretical peak typically indicates:
    • Turbo Boost / GPU Boost clocks exceeding base frequency
    • Optimized BLAS libraries (MKL, cuBLAS) exceeding naive calculations
    • Conservative theoretical peaks (based on sustained, not peak clocks)
  This is normal and indicates good hardware utilization.


================================================================================
Calibration Complete!
================================================================================

Calibration file: /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k_pytorch.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware i7-12700k \
         --calibration /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k_pytorch.json

