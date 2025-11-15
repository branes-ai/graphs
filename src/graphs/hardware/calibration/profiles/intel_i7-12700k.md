================================================================================
Hardware Calibration: Intel-i7-12700K
Date: 2025-11-15T12:35:52.940136
================================================================================

Theoretical Specifications:
  Peak GFLOPS (FP32): 720.0
  Peak Bandwidth:     75.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_operation=memory_copy_size_mb=128          46.4 GB/s       61.9%
  add_operation=memory_copy_size_mb=256          53.2 GB/s       70.9%
  add_operation=memory_copy_size_mb=512          52.2 GB/s       69.7%
  add_operation=memory_copy_size_mb=64           41.4 GB/s       55.2%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp32       256×256           0.1ms      309.0      309.0      309.0       42.9%
  fp32       1024×1024         2.9ms      748.4      748.4      748.4      103.9%
  fp32       2048×2048        22.1ms      778.9      778.9      778.9      108.2%
  fp32       4096×4096       166.8ms      824.2      824.2      824.2      114.5%
  fp64       256×256           0.2ms      200.9      200.9      200.9       55.8%
  fp64       1024×1024         5.5ms      389.7      389.7      389.7      108.3%
  fp64       2048×2048        47.9ms      358.9      358.9      358.9       99.7%
  fp64       4096×4096       337.9ms      406.8      406.8      406.8      113.0%
  int16      256×256           7.9ms        4.2        4.2        4.2        0.6%
  int16      1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int16      4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int32      256×256           8.9ms        3.8        3.8        3.8        1.0%
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


================================================================================
Calibration Complete!
================================================================================

Calibration file: /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware i7-12700k \
         --calibration /home/stillwater/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/intel_i7_12700k.json
