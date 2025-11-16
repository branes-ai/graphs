# Efficiency Threshold Guide

**Date**: 2025-11-16

## Overview

Calibration efficiency compares measured performance to theoretical peak performance from hardware datasheets. Understanding what different efficiency levels mean is crucial for interpreting calibration results.

## Efficiency Formula

```
Efficiency = (Measured GOPS) / (Theoretical Peak GOPS) × 100%
```

Where GOPS = GFLOPS for floating-point, GIOPS for integer operations.

## Efficiency Thresholds

### 🟢 Normal Range: 70-110%

**What it means**: Hardware is performing as expected.

**Typical causes**:
- **70-90%**: Typical efficiency for real-world workloads
- **90-100%**: Excellent performance, approaching theoretical limits
- **100-110%**: Normal variance due to:
  - Turbo Boost / GPU Boost clocks (running faster than base frequency)
  - Optimized BLAS libraries (MKL, OpenBLAS, cuBLAS)
  - Conservative theoretical peaks (datasheets use sustained, not peak clocks)

**Examples from real calibrations**:
```
fp32  1024×1024  2.9ms  742.3 GFLOPS  103.1% eff  ✓ Normal (Turbo Boost)
fp64  1024×1024  5.5ms  387.4 GFLOPS  107.6% eff  ✓ Normal (Turbo Boost)
fp32  2048×2048 22.0ms  781.1 GFLOPS  108.5% eff  ✓ Normal (optimized BLAS)
```

### ⚠️ Above Theoretical: >110%

**What it means**: Performance significantly exceeds datasheet specifications.

**System flags with**: `⚠ ABOVE THEORETICAL`

**Typical causes**:
- **Aggressive Turbo Boost**: CPU running well above base clock (e.g., i7-12700K: 3.6 GHz base → 5.0 GHz boost)
- **Highly optimized libraries**: Intel MKL, cuBLAS using specialized instructions not in theoretical calculations
- **Very conservative theoretical peaks**: Datasheet values significantly underestimate real capability
- **FMA counting**: If theoretical uses 1 FLOP per FMA but benchmarks count 2 FLOPs
- **⭐ Tensor Cores (GPUs)**: FP16/INT8 use specialized matrix hardware much faster than CUDA cores (see below)

**Examples**:
```
Calibrating matmul 1024×1024 across 5 precisions...
  fp32    ... ✓   850.0 GFLOPS (   2.5ms) 118.1% eff ⚠ ABOVE THEORETICAL
    ℹ Likely caused by: Turbo Boost, optimized BLAS, or conservative theoretical peak
```

**Action required**:
- ✅ Normal if <120%: Indicates excellent hardware utilization
- ⚠️ Review if >120%: May indicate measurement error or incorrect theoretical peak

### 🔴 Low Efficiency: <50%

**What it means**: Hardware is significantly underperforming.

**System flags with**: `⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes`

**Typical causes**:
- **Unoptimized operations**: Generic implementations without SIMD/vectorization
- **Integer operations on CPU**: NumPy doesn't use VNNI instructions
- **Small matrices**: Not enough work to amortize overhead
- **Memory-bound**: Operation limited by bandwidth, not compute

**Examples**:
```
Calibrating matmul 256×256 across 5 precisions...
  int32   ... ✓     3.4 GIOPS (   9.7ms)   1.0% eff
    ⚠ Warning: Throughput <50.0 GOPS, will skip for larger sizes
```

**What happens**:
- Precision is skipped for larger matrix sizes (no point testing if unusable)
- Indicates operation would be a bottleneck in production

### 📊 Very Low Efficiency: <1%

**What it means**: Operation is essentially unusable for Embodied AI applications.

**Typical causes**:
- **Software emulation**: Hardware doesn't natively support the operation
- **No SIMD**: Scalar code on vector hardware
- **Severe memory bottleneck**: Accessing RAM randomly

**Example**:
```
int8    ... ✓     4.3 GIOPS (   7.9ms)   0.3% eff
```

**Action required**:
- Avoid using this operation/precision in production
- Consider alternative implementations or frameworks

## Hardware-Specific Considerations

### Intel CPUs (i7-12700K)

**Base vs Turbo frequencies**:
- Base: 3.6 GHz (P-cores), 2.7 GHz (E-cores)
- Turbo: up to 5.0 GHz (P-cores), 3.8 GHz (E-cores)

**Expected efficiency**:
- Sustained workloads: 80-100% (at base clock)
- Burst workloads: 100-140% (with Turbo Boost)

**Theoretical peaks in presets**:
```python
'fp64': 360.0,   # Based on ~3.0 GHz sustained
'fp32': 720.0,   # Based on ~3.0 GHz sustained
```

**Real performance with Turbo**:
- FP64: ~380-390 GFLOPS (105-108% eff)
- FP32: ~740-780 GFLOPS (103-108% eff)

### NVIDIA GPUs (Jetson Orin Nano)

**⭐ IMPORTANT: GPUs Have Two Compute Paths**

1. **CUDA Cores** (general purpose)
   - FP32 operations
   - Theoretical: 1280 GFLOPS
   - Expected efficiency: 85-95%

2. **Tensor Cores** (specialized matrix multiply)
   - FP16/INT8 matrix operations
   - Theoretical: 7600 GFLOPS FP16, 15200 GIOPS INT8
   - **6× faster than CUDA cores for matmul!**
   - Expected efficiency: 80-100%

**Why FP16 Shows 298% Efficiency with Old Peaks**:
- Old theoretical peak: 2560 GFLOPS (assumed CUDA cores only)
- Real FP16 uses Tensor Cores: 7630 GFLOPS measured
- Efficiency: 7630 / 2560 = 298% ❌ (wrong theoretical peak)
- **After correction**: 7630 / 7600 = 100.4% ✅ (correct!)

**Expected efficiency by precision**:
- FP32 (CUDA Cores): 85-95%
- FP16 (Tensor Cores): 80-100%
- INT8 (Tensor Cores): 85-100%

**See `docs/TENSOR_CORE_PERFORMANCE.md` for complete explanation.**

### CPU Integer Operations

**Why NumPy INT8/INT16 is <1% efficient**:

NumPy uses generic integer multiply-accumulate:
```python
# Generic implementation (slow)
result = 0
for i in range(n):
    result += a[i] * b[i]
```

Intel CPUs have VNNI (Vector Neural Network Instructions) for fast INT8:
```asm
; VNNI instruction (fast, but NumPy doesn't use this)
vpdpbusd zmm0, zmm1, zmm2  ; 64 INT8 MACs in one instruction
```

**Theoretical INT8 peak**: 1440 GIOPS (assuming VNNI)
**Actual NumPy INT8**: 4-5 GIOPS (0.3% eff - no VNNI!)

**Solution**: Use frameworks with VNNI support:
- PyTorch with MKL-DNN backend
- TensorFlow with oneDNN
- Direct MKL calls

## Calibration Summary Annotations

When viewing calibration summaries, efficiency warnings appear inline:

```
Matrix Multiplication Performance (by precision):
  Precision  Size            Latency   Min GOPS   Avg GOPS   Max GOPS  Efficiency
  ------------------------------------------------------------------------------------------
  fp32       1024×1024         2.9ms      742.3      742.3      742.3      103.1%
  fp32       2048×2048        22.0ms      781.1      781.1      781.1      108.5%
  fp64       1024×1024         5.5ms      387.4      387.4      387.4      107.6%
  fp64       2048×2048        47.6ms      360.7      360.7      360.7      100.2%
  int8       256×256           7.9ms        4.3        4.3        4.3        0.3%
  int8       1024×1024             -          -          -          -          -  SKIPPED (< 50 GOPS)

Precision Support Summary:
  Supported:   fp64, fp32, fp16, bf16, int32, int16, int8
  Unsupported: fp8_e4m3, fp8_e5m2

Note on >100% Efficiency:
  Efficiency above theoretical peak typically indicates:
    • Turbo Boost / GPU Boost clocks exceeding base frequency
    • Optimized BLAS libraries (MKL, cuBLAS) exceeding naive calculations
    • Conservative theoretical peaks (based on sustained, not peak clocks)
  This is normal and indicates good hardware utilization.
```

## Best Practices

### When Calibrating

1. **Understand your hardware specs**:
   - Check base vs boost frequencies
   - Note thermal throttling limits
   - Know which instructions are available (AVX2, AVX-512, VNNI, Tensor Cores)

2. **Interpret efficiency in context**:
   - 100-110%: Expected for CPUs with Turbo Boost
   - 80-95%: Expected for GPUs (perfect occupancy is rare)
   - <50%: May indicate unoptimized operation

3. **Compare frameworks**:
   ```bash
   # NumPy (may not use advanced instructions)
   ./cli/calibrate_hardware.py --preset i7-12700k --framework numpy

   # PyTorch (better optimized, uses MKL)
   ./cli/calibrate_hardware.py --preset i7-12700k --framework pytorch
   ```

4. **Check for consistency**:
   - Run calibration multiple times
   - Efficiency should be stable (±5%)
   - Large variance indicates thermal throttling or background processes

### When Using Calibration Data

1. **Choose appropriate framework**:
   - Signal processing → Use NumPy calibration
   - DL inference → Use PyTorch calibration
   - Mixed workload → Compare both

2. **Understand precision efficiency**:
   - High efficiency (>80%): Good choice for this hardware
   - Low efficiency (<10%): Avoid in production
   - Skipped: Hardware can't handle workload at this precision

3. **Plan for thermal throttling**:
   - Burst calibration: 110% eff (Turbo Boost)
   - Sustained workload: 90% eff (thermal throttling kicks in)
   - Use sustained efficiency for long-running inference

## Summary

| Efficiency Range | Interpretation | Action |
|-----------------|----------------|--------|
| **>120%** | Exceptional / Check for errors | Review theoretical peak and measurement |
| **110-120%** | Above theoretical (flagged) | Normal with Turbo Boost, verify if >120% |
| **100-110%** | Excellent, near-peak | Ideal performance |
| **80-100%** | Good performance | Normal for real workloads |
| **50-80%** | Moderate performance | Acceptable, room for optimization |
| **10-50%** | Low performance (flagged) | Consider alternatives |
| **<10%** | Very low performance | Avoid in production |
| **<1%** | Essentially unusable | Unsupported or software emulated |

Understanding these thresholds helps you:
- ✅ Identify when hardware is performing well (>100% is good!)
- ✅ Spot problematic operations (<50%)
- ✅ Make informed decisions about precision and framework choice
- ✅ Set realistic performance expectations for production workloads
