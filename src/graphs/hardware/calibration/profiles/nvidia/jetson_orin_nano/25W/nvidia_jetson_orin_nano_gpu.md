p310) branes@Branes-Jetson:~/dev/branes/clones/graphs$ python cli/calibrate_hardware.py --preset jetson-orin-nano-gpu
Platform Validation:
  Current architecture: aarch64
  Expected architecture: aarch64
  CUDA available: True
  CUDA device: Orin
  ✓ Platform validation passed

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-GPU
================================================================================

System Information:
  CPU: aarch64
  Cores: 6 physical, 6 logical
  Memory: 7.4 GB
  Python: 3.10.12
  NumPy: 1.26.4
  PyTorch: 2.8.0

Running calibration benchmarks...

1. Memory Bandwidth
--------------------------------------------------------------------------------
Calibrating memory copy 64 MB... 13.2 GB/s (19.4% efficiency)
Calibrating memory copy 128 MB... 13.2 GB/s (19.4% efficiency)
Calibrating memory copy 256 MB... 13.2 GB/s (19.4% efficiency)
Calibrating memory copy 512 MB... 13.2 GB/s (19.4% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------

Calibrating matmul 256×256 across 3 precisions...
  fp32    ... ✓    86.5 GFLOPS (   0.4ms)   6.8% eff
  fp16    ... ✓     0.3 GFLOPS ( 100.8ms)   0.0% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip this precision for larger sizes
  int8    ... ✗ FAIL: int8 not supported on cuda

Calibrating matmul 1024×1024 across 3 precisions...
  fp32    ... ✓   106.6 GFLOPS (  20.1ms)   8.3% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ✗ FAIL: int8 not supported on cuda

Calibrating matmul 2048×2048 across 3 precisions...
  fp32    ... ✓   113.4 GFLOPS ( 151.5ms)   8.9% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ✗ FAIL: int8 not supported on cuda

Calibrating matmul 4096×4096 across 3 precisions...
  fp32    ... ✓   114.5 GFLOPS (  1.2s)   8.9% eff
  fp16    ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
  int8    ... ✗ FAIL: int8 not supported on cuda

Building precision capability matrix...

Calibration saved to: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu.json

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-GPU
Date: 2025-11-16T08:54:27.815465
================================================================================

Theoretical Specifications:
  Peak GFLOPS (FP32): 1280.0
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_operation=memory_copy_size_mb=128          13.2 GB/s       19.4%
  add_operation=memory_copy_size_mb=256          13.2 GB/s       19.4%
  add_operation=memory_copy_size_mb=512          13.2 GB/s       19.4%
  add_operation=memory_copy_size_mb=64           13.2 GB/s       19.4%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256         100.8ms        0.3        0.3        0.3        0.0%
  fp16       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp16       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp16       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)
  fp32       256×256           0.4ms       86.5       86.5       86.5        6.8%
  fp32       1024×1024        20.1ms      106.6      106.6      106.6        8.3%
  fp32       2048×2048       151.5ms      113.4      113.4      113.4        8.9%
  fp32       4096×4096         1.20s      114.5      114.5      114.5        8.9%
  int8       256×256               -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       2048×2048             -          -          -          -          -  SKIPPED (< 50 GOPS)
  int8       4096×4096             -          -          -          -          -  SKIPPED (< 50 GOPS)

Precision Support Summary:
  Supported:   fp64, fp32, fp16, bf16
  Unsupported: int32, int16, int8, fp8_e4m3, fp8_e5m2


================================================================================
Calibration Complete!
================================================================================

Calibration file: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware jetson-orin-nano-gpu \
         --calibration /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu.json

