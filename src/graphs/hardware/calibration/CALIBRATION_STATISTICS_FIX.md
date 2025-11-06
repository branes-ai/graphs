# Calibration Statistics Fix - RCA and Implementation

## Issue

**Symptom**: Calibration summary displayed confusing statistics:
```
Measured Performance:
  Best GFLOPS:    790.7 (79.1% efficiency)
  Avg GFLOPS:     348.6 (66.8% efficiency)
  Worst GFLOPS:   0.0 (56.0% efficiency)  ← CONFUSING!
  Bandwidth:      51.6 GB/s (68.8% efficiency)
```

**Problem**: "Worst GFLOPS: 0.0 (56.0% efficiency)" is misleading because:
1. 0.0 GFLOPS should have 0% efficiency, not 56%
2. The 56% efficiency actually belongs to a different operation (512×512 matmul at 560.4 GFLOPS)

## Root Cause Analysis

### The Bug

**Location**: `src/graphs/hardware/calibration/schema.py:233-247`

**Original Code**:
```python
def _update_statistics(self):
    """Update aggregate statistics from operation profiles"""
    if not self.operation_profiles:
        return

    efficiencies = [p.efficiency for p in self.operation_profiles.values()]
    gflops = [p.measured_gflops for p in self.operation_profiles.values()]

    self.best_efficiency = max(efficiencies)
    self.avg_efficiency = sum(efficiencies) / len(efficiencies)
    self.worst_efficiency = min(efficiencies)  # ← min efficiency

    self.best_measured_gflops = max(gflops)
    self.avg_measured_gflops = sum(gflops) / len(gflops)
    self.worst_measured_gflops = min(gflops)  # ← min GFLOPS = 0.0!
```

**The Problem**: The code computes statistics across **all operations**, mixing:

1. **Compute operations** (matmul, conv):
   - Have non-zero GFLOPS
   - Efficiency = measured GFLOPS / theoretical peak GFLOPS

2. **Memory operations** (copy):
   - Have **0.0 GFLOPS** (no floating-point operations)
   - "Efficiency" is actually **bandwidth efficiency** = measured GB/s / theoretical GB/s

### Why This Happened

The calibration data contains both types of operations:

```
Memory operations (memory-bound, 0 GFLOPS):
  64 MB copy:   0.0 GFLOPS, 60.9% efficiency (bandwidth efficiency)
  128 MB copy:  0.0 GFLOPS, 59.1% efficiency
  256 MB copy:  0.0 GFLOPS, 68.8% efficiency
  512 MB copy:  0.0 GFLOPS, 67.0% efficiency

Matmul operations (compute-bound):
  512×512:   560.4 GFLOPS, 56.0% efficiency ← lowest compute efficiency
  1024×1024: 708.0 GFLOPS, 70.8% efficiency
  2048×2048: 729.7 GFLOPS, 73.0% efficiency
  4096×4096: 790.7 GFLOPS, 79.1% efficiency ← highest
```

When the statistics aggregation computed:
```python
worst_measured_gflops = min(gflops)  # = 0.0 from memory operations
worst_efficiency = min(efficiencies)  # = 0.560 from 512×512 matmul
```

These came from **different operations**, creating the confusing display:
- "Worst GFLOPS: 0.0" came from memory operations
- "(56% efficiency)" came from the smallest matmul

### Why Memory Ops Report 0.0 GFLOPS

Memory bandwidth benchmarks use `np.copyto()` which performs no floating-point operations:
```python
# Memory copy benchmark
A = np.random.randn(size_mb * 1024 * 1024 // 4)  # Allocate
B = np.empty_like(A)

start = time.perf_counter()
np.copyto(B, A)  # Pure memory copy, 0 FLOPs
end = time.perf_counter()

# Compute bandwidth, not GFLOPS
bytes_transferred = 2 * A.nbytes  # Read A, write B
bandwidth_gbps = bytes_transferred / (end - start) / 1e9
```

The "efficiency" reported for memory operations is:
```python
efficiency = measured_bandwidth / theoretical_bandwidth
# e.g., 52.6 GB/s / 75 GB/s = 0.701 (70.1%)
```

This is fundamentally different from compute efficiency.

## The Fix

**Implementation**: Separate compute and memory operations when computing statistics.

### Updated Code

**File**: `src/graphs/hardware/calibration/schema.py`

```python
def _update_statistics(self):
    """
    Update aggregate statistics from operation profiles.

    Separates compute operations (matmul, conv) from memory operations (copy)
    to avoid mixing GFLOPS (0.0 for memory ops) with bandwidth efficiency.
    """
    if not self.operation_profiles:
        return

    # Separate compute operations (those that actually do FLOPs)
    # from memory operations (bandwidth-bound, 0 GFLOPS)
    compute_profiles = [p for p in self.operation_profiles.values()
                       if p.measured_gflops > 0]
    memory_profiles = [p for p in self.operation_profiles.values()
                      if p.memory_bound and p.measured_gflops == 0]

    # Compute statistics (only from compute operations)
    # This avoids "worst GFLOPS = 0.0" from memory operations
    if compute_profiles:
        compute_gflops = [p.measured_gflops for p in compute_profiles]
        compute_effs = [p.efficiency for p in compute_profiles]

        self.best_measured_gflops = max(compute_gflops)
        self.avg_measured_gflops = sum(compute_gflops) / len(compute_gflops)
        self.worst_measured_gflops = min(compute_gflops)

        self.best_efficiency = max(compute_effs)
        self.avg_efficiency = sum(compute_effs) / len(compute_effs)
        self.worst_efficiency = min(compute_effs)
    else:
        # Fallback if no compute operations (shouldn't happen)
        self.best_measured_gflops = 0.0
        self.avg_measured_gflops = 0.0
        self.worst_measured_gflops = 0.0
        self.best_efficiency = 0.0
        self.avg_efficiency = 0.0
        self.worst_efficiency = 0.0

    # Memory bandwidth statistics (already handled separately)
    # measured_bandwidth_gbps and bandwidth_efficiency are set directly
    # by the calibrator, not computed here
```

### Key Changes

1. **Filter compute operations**: `measured_gflops > 0`
2. **Filter memory operations**: `memory_bound and measured_gflops == 0`
3. **Compute statistics only from compute operations**
4. **Memory bandwidth statistics remain separate** (already correct)

## Validation

### Before Fix

```
Measured Performance:
  Best GFLOPS:    790.7 (79.1% efficiency)
  Avg GFLOPS:     348.6 (66.8% efficiency)  ← average includes 0.0!
  Worst GFLOPS:   0.0 (56.0% efficiency)    ← WRONG: mixed metrics
  Bandwidth:      51.6 GB/s (68.8% efficiency)
```

**Analysis**:
- Average GFLOPS: `(0 + 0 + 0 + 0 + 560.4 + 708.0 + 729.7 + 790.7) / 8 = 348.6` ✗
- Worst GFLOPS: `min(0, 0, 0, 0, 560.4, ...) = 0.0` ✗
- Worst efficiency: `min(0.609, 0.591, 0.688, 0.670, 0.560, ...) = 0.560` (from matmul)

The display showed metrics from **different operations**.

### After Fix

```
Measured Performance:
  Best GFLOPS:    775.0 (77.5% efficiency)  ✓ 2048×2048 matmul
  Avg GFLOPS:     759.9 (76.0% efficiency)  ✓ average of matmuls only
  Worst GFLOPS:   744.7 (74.5% efficiency)  ✓ 1024×1024 matmul
  Bandwidth:      52.4 GB/s (69.9% efficiency)  ✓ memory operations
```

**Analysis**:
- Average GFLOPS: `(744.7 + 775.0) / 2 = 759.9` ✓ (matmuls only)
- Worst GFLOPS: `min(744.7, 775.0) = 744.7` ✓ (1024×1024 matmul)
- Worst efficiency: `min(0.745, 0.775) = 0.745` ✓ (matches worst GFLOPS)

All metrics are now **consistent** and come from the **same category** (compute operations).

### Test Results

```bash
$ python test_calibration_integration.py

Measured Performance:
  Best GFLOPS:    775.0 (77.5% efficiency)
  Avg GFLOPS:     759.9 (76.0% efficiency)
  Worst GFLOPS:   744.7 (74.5% efficiency)
  Bandwidth:      52.4 GB/s (69.9% efficiency)

Operation Profiles (4 total):
  Operation                     GFLOPS   Efficiency        Bound
  -----------------------------------------------------------------
  add_operation=memory_copy_size_mb=128        0.0        59.2%       Memory
  add_operation=memory_copy_size_mb=256        0.0        69.9%       Memory
  matmul_cpp_gflops=None_implementation=numpy_blas_matrix_size=1024_medium      744.7        74.5%      Compute
  matmul_cpp_gflops=None_implementation=numpy_blas_matrix_size=2048_medium      775.0        77.5%      Compute
```

✅ All statistics are now correct and meaningful.

## Impact

### Clarified Statistics

The summary now clearly shows:

1. **Compute Performance** (GFLOPS, efficiency):
   - Best: Largest matmul (highest GFLOPS)
   - Worst: Smallest matmul (lowest GFLOPS, but still >0)
   - Average: Mean across all compute operations

2. **Memory Performance** (GB/s, bandwidth efficiency):
   - Reported separately
   - Not mixed with GFLOPS statistics

### Improved Usability

Users can now:
- Quickly identify compute bottlenecks (low worst GFLOPS)
- Quickly identify memory bottlenecks (low bandwidth)
- Understand efficiency without confusion
- Compare different hardware profiles meaningfully

## Files Modified

1. **`src/graphs/hardware/calibration/schema.py`**
   - Updated `_update_statistics()` method (+41 lines, -9 lines)
   - Added documentation explaining the separation

2. **`src/graphs/hardware/calibration/profiles/intel_i7_12700k.json`**
   - Regenerated with correct statistics
   - Worst GFLOPS now 744.7 (not 0.0)
   - Average GFLOPS now 759.9 (not 348.6)

## Future Enhancements

While the fix solves the immediate problem, future improvements could include:

1. **Separate Display Sections**:
   ```
   Compute Performance:
     Best:  775.0 GFLOPS (77.5%)
     Avg:   759.9 GFLOPS (76.0%)
     Worst: 744.7 GFLOPS (74.5%)

   Memory Performance:
     Best:  52.4 GB/s (69.9%)
     Avg:   48.4 GB/s (64.5%)
     Worst: 44.4 GB/s (59.2%)
   ```

2. **Per-Category Statistics**:
   - Track matmul statistics separately from conv2d
   - Track different memory access patterns separately

3. **Weighted Averages**:
   - Weight by typical workload mix
   - E.g., ResNet has 90% conv, 10% matmul

## Conclusion

The fix successfully **separates compute and memory statistics**, eliminating the confusing "0.0 GFLOPS (56% efficiency)" display. The calibration summary now provides clear, actionable performance data for both compute-bound and memory-bound operations.

**Status**: ✅ COMPLETE
**Date**: 2025-11-05
**Impact**: Improved usability and clarity of calibration results
