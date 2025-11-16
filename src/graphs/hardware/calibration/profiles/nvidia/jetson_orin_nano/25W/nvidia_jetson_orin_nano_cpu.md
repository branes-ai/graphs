================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-CPU
Date: 2025-11-15T12:36:40.295815
================================================================================

Theoretical Specifications:
  Peak GFLOPS (FP32): 182.4
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_operation=memory_copy_size_mb=128          13.2 GB/s       19.5%
  add_operation=memory_copy_size_mb=256          13.2 GB/s       19.4%
  add_operation=memory_copy_size_mb=512          13.2 GB/s       19.5%
  add_operation=memory_copy_size_mb=64           13.3 GB/s       19.5%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256          85.7ms        0.4        0.4        0.4        0.1%
  fp32       256×256           1.6ms       21.5       21.5       21.5       11.8%
  fp64       256×256           0.8ms       39.8       39.8       39.8       43.6%
  int16      256×256          51.4ms        0.7        0.7        0.7        0.4%
  int32      256×256          53.0ms        0.6        0.6        0.6        0.7%
  int8       256×256          36.0ms        0.9        0.9        0.9        0.3%

Precision Support Summary:
  Supported:   fp64, fp32, fp16, bf16, int32, int16, int8
  Unsupported: fp8_e4m3, fp8_e5m2


================================================================================
Calibration Complete!
================================================================================

