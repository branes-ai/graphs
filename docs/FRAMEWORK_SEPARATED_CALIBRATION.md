# Framework-Separated Calibration Architecture

**Date**: 2025-11-16
**Status**: ✅ Complete

## Problem Statement

The original calibration system had a critical flaw: it silently used NumPy (CPU-only) even when CUDA was requested, producing incorrect calibration data for GPUs. This happened because:

1. NumPy has no GPU support - it always runs on CPU
2. The fallback from GPU to CPU was silent and invisible to users
3. NumPy and PyTorch serve different purposes in Embodied AI applications

## Solution: Framework-Separated Architecture

We've redesigned the calibration system to clearly separate NumPy and PyTorch benchmarks, making the framework choice explicit and preventing silent fallbacks.

### Architecture Overview

```
src/graphs/hardware/calibration/benchmarks/
├── numpy/                      # NumPy benchmarks (CPU-only)
│   ├── __init__.py
│   ├── matmul_bench.py        # NumPy matmul (signal processing)
│   └── memory_bench.py        # NumPy memory bandwidth
├── pytorch/                    # PyTorch benchmarks (CPU or GPU)
│   ├── __init__.py
│   ├── matmul_bench.py        # PyTorch matmul (GPU-accelerated)
│   └── memory_bench.py        # PyTorch memory (GPU)
└── [legacy files...]
```

### Framework Selection Logic

**Automatic Selection:**
- **CPU presets** → NumPy (represents real-world signal processing)
- **GPU presets** → PyTorch (only framework that can use GPU)

**Explicit Override:**
```bash
# Force PyTorch on CPU (for comparison)
./cli/calibrate_hardware.py --preset i7-12700k --framework pytorch

# Force NumPy on CPU (default)
./cli/calibrate_hardware.py --preset i7-12700k --framework numpy

# GPU always requires PyTorch
./cli/calibrate_hardware.py --preset jetson-orin-nano-gpu  # Uses PyTorch automatically
```

### Key Design Decisions

#### 1. Why Keep NumPy?

NumPy benchmarks are essential for understanding CPU performance in real Embodied AI applications where:
- Signal processing uses NumPy operations
- Sensor fusion pipelines use NumPy
- Learning that NumPy operators might not be optimized is key to identifying performance issues

#### 2. Why Add PyTorch?

PyTorch benchmarks are required for:
- Actual GPU calibration (NumPy cannot use GPU)
- DL model inference performance
- CUDA-accelerated operations

#### 3. Clear Separation Benefits

- **No silent fallbacks**: User always knows which framework is running
- **Correct benchmarks**: GPU presets use GPU-capable framework
- **Real-world accuracy**: CPU presets use frameworks that match production code

## Output Files

Calibration files now include the framework name to prevent overwriting:

- **NumPy CPU**: `intel_i7_12700k_numpy.json`
- **PyTorch CPU**: `intel_i7_12700k_pytorch.json`
- **PyTorch GPU**: `nvidia_jetson_orin_nano_gpu_pytorch.json`

This allows you to:
- Compare NumPy vs PyTorch performance on the same CPU
- Keep calibrations from different frameworks separate
- Understand which framework was used for historical calibrations

The framework is also stored in the JSON metadata:
```json
{
  "metadata": {
    "hardware_name": "Intel-i7-12700K",
    "framework": "numpy",
    "device_type": "cpu",
    ...
  }
}
```

## Usage Examples

### GPU Calibration (Jetson Orin Nano)

```bash
# Automatic: Uses PyTorch for GPU
./cli/calibrate_hardware.py --preset jetson-orin-nano-gpu
# Output: nvidia_jetson_orin_nano_gpu_pytorch.json
```

Output shows:
```
================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CUDA
  Actual device:    GPU (Orin)

Execution Device:
  Running on: GPU (Orin)
  Framework:  PYTORCH
              (PyTorch DL framework, GPU-accelerated)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.1

Calibrating matmul 256×256 across 3 precisions...
  fp32    ... ✓  1143.2 GFLOPS (   0.3ms)  89.3% eff
  fp16    ... ✓  2234.7 GFLOPS (   0.1ms)  87.3% eff
  int8    ... ✓  4521.3 GIOPS (   0.1ms)  88.3% eff
```

### CPU Calibration (NumPy for Signal Processing)

```bash
# Automatic: Uses NumPy for CPU
./cli/calibrate_hardware.py --preset i7-12700k
# Output: intel_i7_12700k_numpy.json
```

Output shows:
```
Execution Device:
  Running on: CPU
  Framework:  NUMPY
              (CPU-only, real-world signal processing performance)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: NumPy (CPU-only)

Calibrating matmul 256×256 across 5 precisions...
  fp64    ... ✓   204.1 GFLOPS (   0.2ms)  56.7% eff
  fp32    ... ✓   323.1 GFLOPS (   0.1ms)  44.9% eff
```

### CPU Calibration with PyTorch (for comparison)

```bash
# Explicit override to use PyTorch on CPU
./cli/calibrate_hardware.py --preset i7-12700k --framework pytorch
# Output: intel_i7_12700k_pytorch.json (different from NumPy version!)
```

Output shows:
```
Execution Device:
  Running on: CPU
  Framework:  PYTORCH
              (PyTorch DL framework, GPU-accelerated)

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.7.1+cu126
  Device: CPU

Calibrating matmul 256×256 across 3 precisions...
  fp32    ... ✓   287.3 GFLOPS (   0.1ms)  39.9% eff
```

## Enhanced Device Reporting

### Before Calibration Starts

Users now see clear device information:

```
================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CUDA
  Actual device:    GPU (Orin)
```

Or if fallback occurs:

```
================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CUDA
  Actual device:    CPU (fallback)

  ⚠ WARNING: Device Fallback Occurred!
  Reason: CUDA requested but not available

  This will produce INCORRECT calibration data for the requested hardware!
  The calibration will reflect CPU performance, not GPU performance.

  Continue anyway? (yes/no):
```

### During Calibration

Every section shows which framework is running:

```
1. Memory Bandwidth
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.1

2. Matrix Multiplication (Multi-Precision)
--------------------------------------------------------------------------------
Framework: PyTorch 2.8.0
  Device: Orin
  CUDA:   12.1
```

## API Reference

### CLI Arguments

```bash
./cli/calibrate_hardware.py --preset <preset> [options]

Options:
  --framework {numpy,pytorch}   Override framework selection
                                Default: numpy for CPU, pytorch for GPU
  --quick                       Quick calibration (fewer sizes/trials)
  --operations OPERATIONS       Comma-separated operations (matmul,memory)
  --output OUTPUT               Output JSON file path
  --skip-platform-check         Skip platform validation (USE WITH CAUTION)
```

### Python API

```python
from graphs.hardware.calibration.calibrator import calibrate_hardware

# GPU calibration (automatic PyTorch)
calibration = calibrate_hardware(
    hardware_name='NVIDIA-Jetson-Orin-Nano-GPU',
    theoretical_peak_gflops=1280.0,
    theoretical_bandwidth_gbps=68.0,
    theoretical_peaks={'fp32': 1280.0, 'fp16': 2560.0, 'int8': 5120.0},
    device='cuda',  # Automatically uses PyTorch
    output_path='jetson_orin_nano_gpu.json'
)

# CPU calibration (automatic NumPy)
calibration = calibrate_hardware(
    hardware_name='Intel-i7-12700K',
    theoretical_peak_gflops=720.0,
    theoretical_bandwidth_gbps=75.0,
    theoretical_peaks={'fp64': 360.0, 'fp32': 720.0},
    device='cpu',  # Automatically uses NumPy
    output_path='i7_12700k.json'
)

# Force PyTorch on CPU
calibration = calibrate_hardware(
    hardware_name='Intel-i7-12700K',
    theoretical_peak_gflops=720.0,
    theoretical_bandwidth_gbps=75.0,
    theoretical_peaks={'fp64': 360.0, 'fp32': 720.0},
    device='cpu',
    framework='pytorch',  # Explicit override
    output_path='i7_12700k_pytorch.json'
)
```

## Implementation Details

### NumPy Benchmark Module

**Location**: `src/graphs/hardware/calibration/benchmarks/numpy/`

**Features**:
- Pure NumPy implementation
- CPU-only (no GPU support)
- Represents real-world signal processing performance
- Important for understanding NumPy bottlenecks in Embodied AI

**Supported Precisions**: FP64, FP32, FP16, INT32, INT16, INT8

### PyTorch Benchmark Module

**Location**: `src/graphs/hardware/calibration/benchmarks/pytorch/`

**Features**:
- PyTorch implementation with GPU support
- Can run on CPU or CUDA
- Uses optimized CUDA kernels when available
- Represents DL framework inference performance

**Supported Precisions**: FP64, FP32, FP16, BF16, INT32, INT16, INT8

**GPU-Specific Features**:
- CUDA synchronization for accurate timing
- Device memory allocation
- Tensor Core utilization (on supported GPUs)

### Framework Selection

**Location**: `src/graphs/hardware/calibration/calibrator.py`

**Function**: `select_framework(device, framework_override)`

**Logic**:
1. If `framework_override` specified → validate and use
2. If `device == 'cuda'` → require PyTorch (only GPU-capable framework)
3. If `device == 'cpu'` → prefer NumPy (represents real-world Embodied AI)
4. Fallback to PyTorch if NumPy not available

**Error Handling**:
- Raises `RuntimeError` if NumPy requested for CUDA
- Raises `RuntimeError` if PyTorch not installed but required for CUDA
- Raises `RuntimeError` if neither framework is available

## Testing

### Test NumPy Benchmarks

```bash
# Standalone matmul test
python src/graphs/hardware/calibration/benchmarks/numpy/matmul_bench.py

# Standalone memory test
python src/graphs/hardware/calibration/benchmarks/numpy/memory_bench.py

# Full calibration (CPU with NumPy)
./cli/calibrate_hardware.py --preset i7-12700k --quick --framework numpy
```

### Test PyTorch Benchmarks

```bash
# Standalone matmul test (auto-detects device)
python src/graphs/hardware/calibration/benchmarks/pytorch/matmul_bench.py

# Standalone memory test
python src/graphs/hardware/calibration/benchmarks/pytorch/memory_bench.py

# Full calibration (GPU with PyTorch)
./cli/calibrate_hardware.py --preset jetson-orin-nano-gpu --quick

# Full calibration (CPU with PyTorch, explicit)
./cli/calibrate_hardware.py --preset i7-12700k --quick --framework pytorch
```

## Migration Guide

### For Existing Users

**No breaking changes** - the CLI interface remains the same. The framework is selected automatically based on device type.

**Recommended Actions**:
1. Re-run GPU calibrations to get actual GPU performance (not CPU fallback)
2. Review CPU calibrations - they now use NumPy by default
3. Add `--framework pytorch` if you want PyTorch on CPU for comparison

### For Developers

**Old Code** (silently used NumPy for everything):
```python
from graphs.hardware.calibration.benchmarks.matmul_bench_multi import calibrate_matmul_all_precisions

# This always used NumPy, even if device='cuda'!
results = calibrate_matmul_all_precisions(
    sizes=[1024],
    precisions=[Precision.FP32],
    theoretical_peaks={'fp32': 1280.0},
    device='cuda',  # ❌ Ignored! Still used NumPy (CPU)
    num_trials=10
)
```

**New Code** (explicit framework selection):
```python
from graphs.hardware.calibration.benchmarks.pytorch import calibrate_matmul_pytorch

# This actually uses PyTorch with CUDA!
results = calibrate_matmul_pytorch(
    sizes=[1024],
    precisions=[Precision.FP32],
    theoretical_peaks={'fp32': 1280.0},
    device='cuda',  # ✅ Uses PyTorch with CUDA
    num_trials=10
)
```

## Future Enhancements

### Potential Additions

1. **JAX Backend**: Add JAX benchmarks for TPU support
2. **TensorRT Backend**: Add TensorRT benchmarks for optimized inference
3. **ONNX Runtime**: Add ONNX Runtime benchmarks for cross-platform
4. **MPS Backend**: Add Metal Performance Shaders for Apple Silicon

### Architecture Extensibility

The framework-separated architecture makes it easy to add new backends:

```
benchmarks/
├── numpy/      # NumPy (CPU)
├── pytorch/    # PyTorch (CPU/CUDA)
├── jax/        # JAX (CPU/TPU) - Future
├── tensorrt/   # TensorRT (CUDA) - Future
└── onnx/       # ONNX Runtime (CPU/CUDA/DirectML) - Future
```

## Efficiency Warnings

The calibration system now annotates anomalous efficiency values:

### Low Efficiency (<50 GOPS)
Operations with very low throughput are flagged and skipped for larger sizes:
```
  int8    ... ✓     4.3 GIOPS (   7.9ms)   0.3% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
  int8    1024×1024 ... ⊘ SKIPPED (< 50.0 GOPS on smaller size)
```

### High Efficiency (>110%)
Performance exceeding theoretical peak is flagged with explanation:
```
  fp32    ... ✓   850.0 GFLOPS (   2.5ms) 115.1% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, optimized BLAS, or conservative theoretical peak
```

Summary tables also show warnings:
```
Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp32       1024×1024         2.5ms      850.0      850.0      850.0      115.1%  ⚠ ABOVE THEORETICAL

Note on >100% Efficiency:
  Efficiency above theoretical peak typically indicates:
    • Turbo Boost / GPU Boost clocks exceeding base frequency
    • Optimized BLAS libraries (MKL, cuBLAS) exceeding naive calculations
    • Conservative theoretical peaks (based on sustained, not peak clocks)
  This is normal and indicates good hardware utilization.
```

See `docs/EFFICIENCY_THRESHOLDS.md` for detailed explanation of efficiency ranges and what they mean.

## Summary

The framework-separated calibration architecture provides:

✅ **Correct GPU benchmarks** using PyTorch/CUDA
✅ **Accurate CPU benchmarks** using NumPy (real-world Embodied AI)
✅ **No silent fallbacks** - always shows which framework is running
✅ **Clear device reporting** - users see exactly what's being tested
✅ **Framework flexibility** - can override automatic selection
✅ **Extensible design** - easy to add more backends
✅ **Efficiency warnings** - flags both low (<50 GOPS) and high (>110%) performance
✅ **Framework-aware filenames** - prevents overwriting different framework results

This ensures calibration data accurately represents the performance characteristics of the hardware being tested, whether CPU or GPU.
