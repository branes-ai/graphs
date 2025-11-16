Platform Validation:
  Current architecture: x86_64
  Expected architecture: x86_64
  ✓ Platform validation passed


================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CPU
  Actual device:    CPU
  Framework:        NUMPY

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
  Framework:  NUMPY
              (CPU-only, real-world signal processing performance)

Running calibration benchmarks...

1. Memory Bandwidth
--------------------------------------------------------------------------------
Framework: NumPy (CPU-only)

Calibrating memory copy 64 MB... 48.0 GB/s (64.0% efficiency)
Calibrating memory copy 128 MB... 46.5 GB/s (62.0% efficiency)
Calibrating memory copy 256 MB... 53.0 GB/s (70.6% efficiency)
Calibrating memory copy 512 MB... 51.4 GB/s (68.6% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: NumPy (CPU-only)

Calibrating matmul 256×256 across 5 precisions...
  fp64    ... ✓   201.9 GFLOPS (   0.2ms)  56.1% eff
  fp32    ... ✓   315.2 GFLOPS (   0.1ms)  43.8% eff
  int32   ... ✓     3.7 GIOPS (   9.0ms)   1.0% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int16   ... ✓     4.2 GIOPS (   7.9ms)   0.6% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int8    ... ✓     4.3 GIOPS (   7.9ms)   0.3% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
Calibrating matmul 1024×1024 across 5 precisions...
  fp64    ... ✓   383.1 GFLOPS (   5.6ms) 106.4% eff
  fp32    ... ✓   747.2 GFLOPS (   2.9ms) 103.8% eff
  int32   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int16   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
Calibrating matmul 2048×2048 across 5 precisions...
  fp64    ... ✓   361.6 GFLOPS (  47.5ms) 100.4% eff
  fp32    ... ✓   781.1 GFLOPS (  22.0ms) 108.5% eff
  int32   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int16   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
Calibrating matmul 4096×4096 across 5 precisions...
  fp64    ... ✓   410.3 GFLOPS ( 335.0ms) 114.0% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, optimized BLAS, or conservative theoretical peak
  fp32    ... ✓   824.3 GFLOPS ( 166.7ms) 114.5% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, optimized BLAS, or conservative theoretical peak
  int32   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int16   ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)

Building precision capability matrix...

Calibration saved to: /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k_numpy.json

================================================================================
Hardware Calibration: Intel-i7-12700K
Date: 2025-11-16T09:51:09.240603
================================================================================

Framework: NUMPY
Device:    CPU

Theoretical Specifications:
  Peak GFLOPS (FP32): 720.0
  Peak Bandwidth:     75.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_framework=numpy_operation=memory_copy_size_mb=128       46.5 GB/s       62.0%
  add_framework=numpy_operation=memory_copy_size_mb=256       53.0 GB/s       70.6%
  add_framework=numpy_operation=memory_copy_size_mb=512       51.4 GB/s       68.6%
  add_framework=numpy_operation=memory_copy_size_mb=64       48.0 GB/s       64.0%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp32       256×256           0.1ms      315.2      315.2      315.2       43.8%
  fp32       1024×1024         2.9ms      747.2      747.2      747.2      103.8%
  fp32       2048×2048        22.0ms      781.1      781.1      781.1      108.5%
  fp32       4096×4096       166.7ms      824.3      824.3      824.3      114.5%  ⚠ ABOVE THEORETICAL
  fp64       256×256           0.2ms      201.9      201.9      201.9       56.1%
  fp64       1024×1024         5.6ms      383.1      383.1      383.1      106.4%
  fp64       2048×2048        47.5ms      361.6      361.6      361.6      100.4%
  fp64       4096×4096       335.0ms      410.3      410.3      410.3      114.0%  ⚠ ABOVE THEORETICAL
  int16      256×256           7.9ms        4.2        4.2        4.2        0.6%
  int16      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      256×256           9.0ms        3.7        3.7        3.7        1.0%
  int32      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       256×256           7.9ms        4.3        4.3        4.3        0.3%
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

Calibration file: /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k_numpy.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware i7-12700k \
         --calibration /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k_numpy.json

