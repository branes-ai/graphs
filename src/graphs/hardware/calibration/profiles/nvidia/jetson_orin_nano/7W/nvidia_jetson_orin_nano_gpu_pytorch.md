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
  Cores: 4 physical, 4 logical
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

Calibrating memory copy 64 MB... 23.7 GB/s (34.9% efficiency)
Calibrating memory copy 128 MB... 26.5 GB/s (39.0% efficiency)
Calibrating memory copy 256 MB... 25.2 GB/s (37.0% efficiency)
Calibrating memory copy 512 MB... 24.8 GB/s (36.4% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.6

Calibrating matmul 256×256 across 3 precisions...
  fp32    ... ✓    98.4 GFLOPS (   0.3ms)   7.7% eff
  fp16    ... ✓   205.1 GFLOPS (   0.2ms)   2.7% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 1024×1024 across 3 precisions...
  fp32    ... ✓   226.3 GFLOPS (   9.5ms)  17.7% eff
  fp16    ... ✓  2118.1 GFLOPS (   1.0ms)  27.9% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 2048×2048 across 3 precisions...
  fp32    ... ✓   267.1 GFLOPS (  64.3ms)  20.9% eff
  fp16    ... ✓  2912.4 GFLOPS (   5.9ms)  38.3% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 4096×4096 across 3 precisions...
  fp32    ... ✓   270.0 GFLOPS ( 509.1ms)  21.1% eff
  fp16    ... ✓  3152.8 GFLOPS (  43.6ms)  41.5% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'

Building precision capability matrix...

Calibration saved to: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu_pytorch.json

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-GPU
Date: 2025-11-16T11:04:44.917259
================================================================================

Framework: PYTORCH
Device:    CUDA

Theoretical Specifications:
  Peak GFLOPS (FP32): 1280.0
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=128       26.5 GB/s       39.0%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=256       25.2 GB/s       37.0%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=512       24.8 GB/s       36.4%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=64       23.7 GB/s       34.9%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256           0.2ms      205.1      205.1      205.1        2.7%
  fp16       1024×1024         1.0ms     2118.1     2118.1     2118.1       27.9%
  fp16       2048×2048         5.9ms     2912.4     2912.4     2912.4       38.3%
  fp16       4096×4096        43.6ms     3152.8     3152.8     3152.8       41.5%
  fp32       256×256           0.3ms       98.4       98.4       98.4        7.7%
  fp32       1024×1024         9.5ms      226.3      226.3      226.3       17.7%
  fp32       2048×2048        64.3ms      267.1      267.1      267.1       20.9%
  fp32       4096×4096       509.1ms      270.0      270.0      270.0       21.1%
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

