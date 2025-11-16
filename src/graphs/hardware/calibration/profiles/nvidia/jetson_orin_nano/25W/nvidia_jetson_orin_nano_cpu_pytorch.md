Platform Validation:
  Current architecture: aarch64
  Expected architecture: aarch64
  ✓ Platform validation passed


================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CPU
  Actual device:    CPU
  Framework:        PYTORCH

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-CPU
================================================================================

System Information:
  CPU: aarch64
  Cores: 6 physical, 6 logical
  Memory: 7.4 GB
  Python: 3.10.12
  NumPy: 1.26.4
  PyTorch: 2.8.0

Execution Device:
  Running on: CPU
  Framework:  PYTORCH
              (PyTorch DL framework, GPU-accelerated)

Running calibration benchmarks...

1. Memory Bandwidth
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: CPU

Calibrating memory copy 64 MB... 27.3 GB/s (40.1% efficiency)
Calibrating memory copy 128 MB... 26.5 GB/s (39.0% efficiency)
Calibrating memory copy 256 MB... 28.1 GB/s (41.3% efficiency)
Calibrating memory copy 512 MB... 32.0 GB/s (47.1% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: CPU

Calibrating matmul 256×256 across 6 precisions...
  fp64    ... ✓    26.2 GFLOPS (   1.3ms)  28.8% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  fp32    ... ✓    82.1 GFLOPS (   0.4ms)  45.0% eff
  fp16    ... ✓     0.6 GFLOPS (  53.1ms)   0.2% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'
Calibrating matmul 1024×1024 across 6 precisions...
  fp64    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  fp32    ... ✓    97.2 GFLOPS (  22.1ms)  53.3% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'
Calibrating matmul 2048×2048 across 6 precisions...
  fp64    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  fp32    ... ✓   109.4 GFLOPS ( 157.1ms)  60.0% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'
Calibrating matmul 4096×4096 across 6 precisions...
  fp64    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  fp32    ... ✓   113.2 GFLOPS (  1.2s)  62.1% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int32   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Int'
  int16   ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Short'
  int8    ... ✗ FAIL: "normal_kernel_cpu" not implemented for 'Char'

Building precision capability matrix...

Calibration saved to: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_cpu_pytorch.json

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-CPU
Date: 2025-11-16T10:25:21.579224
================================================================================

Framework: PYTORCH
Device:    CPU

Theoretical Specifications:
  Peak GFLOPS (FP32): 182.4
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=128       26.5 GB/s       39.0%
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=256       28.1 GB/s       41.3%
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=512       32.0 GB/s       47.1%
  add_device=cpu_framework=pytorch_operation=memory_copy_size_mb=64       27.3 GB/s       40.1%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256          53.1ms        0.6        0.6        0.6        0.2%
  fp16       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp16       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp16       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp32       256×256           0.4ms       82.1       82.1       82.1       45.0%
  fp32       1024×1024        22.1ms       97.2       97.2       97.2       53.3%
  fp32       2048×2048       157.1ms      109.4      109.4      109.4       60.0%
  fp32       4096×4096         1.21s      113.2      113.2      113.2       62.1%
  fp64       256×256           1.3ms       26.2       26.2       26.2       28.8%
  fp64       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp64       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp64       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
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


================================================================================
Calibration Complete!
================================================================================

Calibration file: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_cpu_pytorch.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware jetson-orin-nano-cpu \
         --calibration /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_cpu_pytorch.json

