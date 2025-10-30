# Root Cause Analysis: TPU v4 Unrealistic Performance on ResNet18

## Problem Statement

TPU v4 showing **10,009 FPS** throughput for ResNet18 (batch=1), which translates to **0.0999 ms latency**. This is unrealistic for several reasons:

1. **Hardware Reality**: ResNet18 inference on real TPU v4 takes ~1-2ms at batch=1
2. **Comparison**: Even H100 GPU achieves only 2,317 FPS
3. **Physics**: The TPU result assumes >95% utilization of a 275 TFLOPS chip on tiny kernels

## Root Cause

### Issue: Peak SoC Performance Division

**Current flawed algorithm:**
```python
# TPU v4 has 2 MXUs, 275 TFLOPS total peak
total_ops = 1.8 GFLOPS  # ResNet18
peak_ops = 275 TFLOPS   # Peak SoC performance
latency = 1.8G / 275T = 0.0065 seconds = 6.5 ms

# But TPU mapper further divides by allocated_units...
latency = ops / (peak_ops_per_unit * allocated_units * occupancy)
```

**Problems:**
1. **Assumes both MXUs are fully utilized** - ResNet18 layers are too small
2. **No discrete resource allocation** - Can't use 0.3 MXUs!
3. **Ignores systolic array geometry** - 128×128 arrays need large matrices to be efficient
4. **No kernel launch overhead** - Each layer is a separate kernel with setup time

### TPU v4 Architecture Details

```
TPU v4 Chip (Google)
├── MXU 0
│   └── 128×128 Systolic Array (16,384 MACs)
│       └── 137.5 TFLOPS @ 2 GHz (per core)
└── MXU 1
    └── 128×128 Systolic Array (16,384 MACs)
        └── 137.5 TFLOPS @ 2 GHz (per core)

Total: 275 TFLOPS (both cores fully utilized)
```

### ResNet18 Reality Check

**Typical ResNet18 Conv Layer:**
- Input: 64 channels, 56×56 spatial
- Output: 64 channels, 56×56 spatial
- Kernel: 3×3
- Matrix dimensions: M=3136 (56×56), N=64, K=576 (64×3×3)

**Systolic Array Utilization:**
- Optimal: 128×128 matrices
- ResNet18: Much smaller matrices (M=3136, N=64, K=576)
- Row utilization: min(3136/128, 1.0) = 100%
- Col utilization: min(64/128, 1.0) = 50%
- **Effective utilization: ~50% on ONE array**

**Concurrency:**
- ResNet18 at batch=1: Limited parallelism
- Can't keep 2 MXUs busy simultaneously
- Most layers run on 1 MXU sequentially

## How GPU Mapper Was Fixed (for reference)

The GPU mapper had the same issue and was fixed with:

### 1. Sequential Execution Detection
```python
def should_use_sequential_execution(fusion_report, batch_size):
    avg_flops = fusion_report.total_flops / num_subgraphs
    return (batch_size < 8 and avg_flops < 200e6)
```

### 2. Discrete Resource Allocation
```python
def determine_sm_allocation(subgraph):
    if flops < 10e6:   return 2   # Very small
    if flops < 50e6:   return 8   # Small
    if flops < 200e6:  return 24  # Medium (ResNet range)
    else:              return 48  # Large
```

### 3. Per-Unit Throughput
```python
# Don't use peak SoC performance!
sm_flops = (cuda_cores_per_sm × ops_per_clock × frequency)
sm_bandwidth = peak_bandwidth / total_sms

# Calculate on allocated SMs only
compute_time = ops / (sm_flops * sms_allocated)
memory_time = bytes / (sm_bandwidth * sms_allocated)
```

### 4. Kernel Launch Overhead
```python
kernel_latency = max(compute_time, memory_time) + 10e-6  # +10µs
```

## Required Fix for TPU

Apply same pattern:

### 1. Detect Small Workloads
```python
def should_use_sequential_execution(fusion_report, batch_size):
    # TPU needs even larger batches to saturate (batch >= 16)
    # Average layer should be > 500M FLOPs to justify parallel
    avg_flops = fusion_report.total_flops / num_subgraphs
    return (batch_size < 16 and avg_flops < 500e6)
```

### 2. Discrete Array Allocation
```python
def determine_array_allocation(subgraph):
    """
    Allocate 1 or 2 MXUs based on workload size.

    Rationale:
    - < 100M FLOPs: 1 array (small kernels like ResNet18 layers)
    - >= 100M FLOPs: 2 arrays (large kernels, can saturate both)
    """
    flops = subgraph.total_flops
    if flops < 100e6:
        return 1  # Use 1 MXU
    else:
        return 2  # Use both MXUs
```

### 3. Per-Array Throughput
```python
# TPU v4: 2 MXUs, each with 128×128 systolic array
mxu_flops = 137.5e12  # 275 TFLOPS / 2 cores
mxu_bandwidth = 600e9  # 1.2 TB/s / 2 cores

# Calculate on allocated arrays only
compute_time = ops / (mxu_flops * arrays_allocated)
memory_time = bytes / (mxu_bandwidth * arrays_allocated)
```

### 4. Systolic Array Overhead
```python
# Systolic arrays have setup overhead (filling pipeline)
# For 128×128 array: ~128 cycle pipeline depth
array_setup_overhead = 64e-9  # ~64 ns (128 cycles @ 2 GHz)
kernel_latency = max(compute_time, memory_time) + array_setup_overhead
```

### 5. Matrix Dimension Underutilization
```python
# Account for matrices smaller than 128×128
matrix_utilization = (min(M, 128) / 128) * (min(N, 128) / 128)
effective_flops = flops / matrix_utilization  # Inflate to account for unused MACs
```

## Expected Results After Fix

**Before Fix:**
- Latency: 0.0999 ms (unrealistic)
- Throughput: 10,009 FPS
- Assumes: 95% utilization of 275 TFLOPS

**After Fix (estimated):**
- Latency: ~2-3 ms (realistic)
- Throughput: ~333-500 FPS
- Assumes: 1 MXU, 50% utilization, sequential kernels

**Validation:**
- Should be similar to KPU (2,169 FPS, 0.46 ms)
- Slower than GPU (2,317 FPS, 0.43 ms) due to array setup overhead
- Much faster than CPU (676 FPS, 1.48 ms)

## Implementation Plan

1. Add `should_use_sequential_execution()` method
2. Add `determine_array_allocation()` method
3. Add `compute_sequential_latency()` method with per-array throughput
4. Update `map_graph()` to detect and use sequential mode
5. Test on ResNet18 and validate results are realistic
