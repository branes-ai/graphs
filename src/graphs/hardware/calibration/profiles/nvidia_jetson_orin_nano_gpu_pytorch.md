(p310) branes@Branes-Jetson:~/dev/branes/clones/graphs$ python cli/calibrate_hardware.py --preset jetson-orin-nano-gpu --framework pytorch
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

Calibrating memory copy 64 MB... 38.6 GB/s (56.8% efficiency)
Calibrating memory copy 128 MB... 57.8 GB/s (85.0% efficiency)
Calibrating memory copy 256 MB... 56.6 GB/s (83.3% efficiency)
Calibrating memory copy 512 MB... 54.8 GB/s (80.6% efficiency)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.6

Calibrating matmul 256×256 across 3 precisions...
  fp32    ... ✓   171.9 GFLOPS (   0.2ms)  13.4% eff
  fp16    ... ✓   276.6 GFLOPS (   0.1ms)   3.6% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 1024×1024 across 3 precisions...
  fp32    ... ✓   425.3 GFLOPS (   5.0ms)  33.2% eff
  fp16    ... ✓  4929.3 GFLOPS (   0.4ms)  64.9% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 2048×2048 across 3 precisions...
  fp32    ... ✓  1188.5 GFLOPS (  14.5ms)  92.9% eff
  fp16    ... ✓  4484.8 GFLOPS (   3.8ms)  59.0% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'
Calibrating matmul 4096×4096 across 3 precisions...
  fp32    ... ✓  1199.1 GFLOPS ( 114.6ms)  93.7% eff
  fp16    ... ✓  7324.9 GFLOPS (  18.8ms)  96.4% eff
  int8    ... ✗ FAIL: "normal_kernel_cuda" not implemented for 'Char'

Building precision capability matrix...

Calibration saved to: /home/branes/dev/branes/clones/graphs/src/graphs/hardware/calibration/profiles/nvidia_jetson_orin_nano_gpu_pytorch.json

================================================================================
Hardware Calibration: NVIDIA-Jetson-Orin-Nano-GPU
Date: 2025-11-16T10:04:26.791492
================================================================================

Framework: PYTORCH
Device:    CUDA

Theoretical Specifications:
  Peak GFLOPS (FP32): 1280.0
  Peak Bandwidth:     68.0 GB/s

Memory Operations:
  Operation                                   Bandwidth   Efficiency
  ----------------------------------------------------------------------
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=128       57.8 GB/s       85.0%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=256       56.6 GB/s       83.3%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=512       54.8 GB/s       80.6%
  add_device=cuda_framework=pytorch_operation=memory_copy_size_mb=64       38.6 GB/s       56.8%

Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp16       256×256           0.1ms      276.6      276.6      276.6        3.6%
  fp16       1024×1024         0.4ms     4929.3     4929.3     4929.3       64.9%
  fp16       2048×2048         3.8ms     4484.8     4484.8     4484.8       59.0%
  fp16       4096×4096        18.8ms     7324.9     7324.9     7324.9       96.4%
  fp32       256×256           0.2ms      171.9      171.9      171.9       13.4%
  fp32       1024×1024         5.0ms      425.3      425.3      425.3       33.2%
  fp32       2048×2048        14.5ms     1188.5     1188.5     1188.5       92.9%
  fp32       4096×4096       114.6ms     1199.1     1199.1     1199.1       93.7%
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
