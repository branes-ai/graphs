Platform Validation:
  Current architecture: aarch64
  Expected architecture: aarch64
  CUDA available: True
  CUDA device: Orin
  ✓ Platform validation passed


================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CUDA
  Actual device:    GPU (Orin)
  Framework:        PYTORCH

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

Execution Device:
  Running on: GPU (Orin)
  Framework:  PYTORCH
              (PyTorch DL framework, GPU-accelerated)

Running calibration benchmarks...

1. Memory Bandwidth
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.6

Calibrating memory copy 64 MB... 35.1 GB/s (51.7% efficiency)
Calibrating memory copy 128 MB... 36.1 GB/s (53.2% efficiency)
Calibrating memory copy 256 MB... 50.1 GB/s (73.6% efficiency)
Calibrating memory copy 512 MB... 49.2 GB/s (72.4% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.6

Calibrating matmul 256×256 across 3 precisions...
  fp32    ... ✓   201.0 GFLOPS (   0.2ms)  15.7% eff
  fp16    ... ✓   306.0 GFLOPS (   0.1ms)   4.0% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 1024×1024 across 3 precisions...
  fp32    ... ✓   578.2 GFLOPS (   3.7ms)  45.2% eff
  fp16    ... ✓  3883.4 GFLOPS (   0.6ms)  51.1% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 2048×2048 across 3 precisions...
  fp32    ... ✓   798.7 GFLOPS (  21.5ms)  62.4% eff
  fp16    ... ✓  5591.8 GFLOPS (   3.1ms)  73.6% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 4096×4096 across 3 precisions...
  fp32    ... ✓   800.7 GFLOPS ( 171.6ms)  62.6% eff
  fp16    ... ✓  5908.0 GFLOPS (  23.3ms)  77.7% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'

Building precision capability matrix...

Calibration saved to: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu_pytorch.json

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-GPU
Date: 2025-11-16T10:55:50.389665
================================================================================

Framework: PYTORCH
Device:    CUDA

Theoretical Specifications:
  Peak GFLOPS (FP32): 1280.0
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=128       36.1 GB/s       53.2%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=256       50.1 GB/s       73.6%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=512       49.2 GB/s       72.4%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=64       35.1 GB/s       51.7%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256           0.1ms      306.0      306.0      306.0        4.0%
  fp16       1024×1024         0.6ms     3883.4     3883.4     3883.4       51.1%
  fp16       2048×2048         3.1ms     5591.8     5591.8     5591.8       73.6%
  fp16       4096×4096        23.3ms     5908.0     5908.0     5908.0       77.7%
  fp32       256×256           0.2ms      201.0      201.0      201.0       15.7%
  fp32       1024×1024         3.7ms      578.2      578.2      578.2       45.2%
  fp32       2048×2048        21.5ms      798.7      798.7      798.7       62.4%
  fp32       4096×4096       171.6ms      800.7      800.7      800.7       62.6%
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

Calibration file: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu_pytorch.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware jetson-orin-nano-gpu \
         --calibration /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu_pytorch.json

