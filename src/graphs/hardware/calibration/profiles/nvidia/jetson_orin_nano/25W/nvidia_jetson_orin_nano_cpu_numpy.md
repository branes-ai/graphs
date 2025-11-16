Platform Validation:
  Current architecture: aarch64
  Expected architecture: aarch64
  ✓ Platform validation passed


================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CPU
  Actual device:    CPU
  Framework:        NUMPY

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
  Framework:  NUMPY
              (CPU-only, real-world signal processing performance)

Running calibration benchmarks...

1. Memory Bandwidth
--------------------------------------------------------------------------------
Framework: NumPy (CPU-only)

Calibrating memory copy 64 MB... 13.1 GB/s (19.3% efficiency)
Calibrating memory copy 128 MB... 13.2 GB/s (19.3% efficiency)
Calibrating memory copy 256 MB... 13.1 GB/s (19.3% efficiency)
Calibrating memory copy 512 MB... 13.2 GB/s (19.4% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: NumPy (CPU-only)

Calibrating matmul 256×256 across 6 precisions...
  fp64    ... ✓    44.2 GFLOPS (   0.8ms)  48.5% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  fp32    ... ✓    85.6 GFLOPS (   0.4ms)  46.9% eff
  fp16    ... ✓     0.3 GFLOPS ( 100.2ms)   0.1% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int32   ... ✓     0.6 GIOPS (  52.6ms)   0.7% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int16   ... ✓     0.7 GIOPS (  51.2ms)   0.4% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int8    ... ✓     0.9 GIOPS (  36.1ms)   0.3% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
Calibrating matmul 1024×1024 across 6 precisions...
  fp64    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  fp32    ... ✓   104.4 GFLOPS (  20.6ms)  57.3% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int32   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int16   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
Calibrating matmul 2048×2048 across 6 precisions...
  fp64    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  fp32    ... ✓   111.6 GFLOPS ( 153.9ms)  61.2% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int32   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int16   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
Calibrating matmul 4096×4096 across 6 precisions...
  fp64    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  fp32    ... ✓   114.2 GFLOPS (  1.2s)  62.6% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int32   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int16   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)

Building precision capability matrix...

Calibration saved to: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_cpu_numpy.json

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-CPU
Date: 2025-11-16T10:26:41.780372
================================================================================

Framework: NUMPY
Device:    CPU

Theoretical Specifications:
  Peak GFLOPS (FP32): 182.4
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_framework=numpy_operation=memory_copy_size_mb=128       13.2 GB/s       19.3%
  add_framework=numpy_operation=memory_copy_size_mb=256       13.1 GB/s       19.3%
  add_framework=numpy_operation=memory_copy_size_mb=512       13.2 GB/s       19.4%
  add_framework=numpy_operation=memory_copy_size_mb=64       13.1 GB/s       19.3%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256         100.2ms        0.3        0.3        0.3        0.1%
  fp16       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp16       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp16       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp32       256×256           0.4ms       85.6       85.6       85.6       46.9%
  fp32       1024×1024        20.6ms      104.4      104.4      104.4       57.3%
  fp32       2048×2048       153.9ms      111.6      111.6      111.6       61.2%
  fp32       4096×4096         1.20s      114.2      114.2      114.2       62.6%
  fp64       256×256           0.8ms       44.2       44.2       44.2       48.5%
  fp64       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp64       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp64       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      256×256          51.2ms        0.7        0.7        0.7        0.4%
  int16      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      256×256          52.6ms        0.6        0.6        0.6        0.7%
  int32      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       256×256          36.1ms        0.9        0.9        0.9        0.3%
  int8       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)

Precision Support Summary:
  Supported:   fp64, fp32, fp16, bf16, int32, int16, int8
  Unsupported: fp8_e4m3, fp8_e5m2


================================================================================
Calibration Complete!
================================================================================

Calibration file: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_cpu_numpy.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware jetson-orin-nano-cpu \
         --calibration /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_cpu_numpy.json

