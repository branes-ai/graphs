# TPU MXU Terminology Fix - Complete Summary

## Issue
TPU v4 hardware was incorrectly modeled using NVIDIA GPU terminology ("TensorCore") instead of Google's actual terminology ("MXU" - Matrix Multiplier Unit).

## Changes Made

### 1. TPU Resource Model (`src/graphs/hardware/models/datacenter/tpu_v4.py`)

**Updated documentation:**
```python
"""
Google TPU v4 resource model.

Architecture:
- 2 Matrix Multiplier Units (MXUs)
- Each MXU: 128×128 systolic array (16,384 MACs)
- Per MXU: 137.5 TFLOPS BF16, 275 TOPS INT8
"""
```

**Updated field comments:**
- `compute_units=2` → now says "2 MXUs (Matrix Multiplier Units)"
- `threads_per_unit=128 * 128` → now says "128×128 systolic array per MXU"
- `l1_cache_per_unit` → now says "16 MB per MXU"

### 2. TPU Mapper (`src/graphs/hardware/mappers/accelerators/tpu.py`)

**Module docstring:**
- Changed "2 TensorCores vs 132 SMs" → "2 MXUs vs 132 GPU SMs"
- Changed "Massive systolic arrays (128×128 = 16K MACs)" → "Each MXU: 128×128 systolic array"
- Updated examples to show "137.5 TFLOPS per MXU"

**Variable renamings:**
- `self.num_tensor_cores` → `self.num_mxus`
- `self.threads_per_core` → `self.threads_per_mxu`
- `tensor_core_ops` → `mxu_ops`
- `tensor_core_bandwidth` → `mxu_bandwidth`
- `cores_allocated` → `mxus_allocated`
- `cores_needed` → `mxus_needed`
- `threads_per_core` → `threads_per_mxu`

**Method documentation updates:**
- `should_use_sequential_execution()`: "saturate multiple TensorCores" → "saturate multiple MXUs"
- `determine_array_allocation()`: Returns "Number of MXUs to allocate (1-2)"
- `compute_sequential_latency()`: "Each subgraph uses 1-2 MXUs"
- `map_graph()`: "Each subgraph uses 1-2 MXUs sequentially"

**Comments:**
- "Per-TensorCore performance" → "Per-MXU performance"
- "TPU v4: 2 TensorCores, 275 TFLOPS total → 137.5 TFLOPS per core" → "TPU v4: 2 MXUs, 275 TFLOPS total → 137.5 TFLOPS per MXU"
- "Roofline model on allocated TensorCores" → "Roofline model on allocated MXUs"
- "Utilization: fraction of TensorCores used" → "Utilization: fraction of MXUs used"

### 3. Roofline Analyzer (`src/graphs/analysis/roofline.py`)

**Updated correction factor comments:**
```python
if hw_type == 'TPU':
    # TPU v4: 2 MXUs (Matrix Multiplier Units), each 128×128 systolic array
    # Small kernels suffer from:
    # 1. Can only use 1 MXU (2× penalty)
    # 2. Matrix dimensions < 128×128 (poor array utilization)
    # 3. Sequential execution overhead

    if sg.flops < 10e6:
        # Tiny kernels: 1 MXU, ~20% utilization → 10× penalty
        return 10.0
    elif sg.flops < 100e6:
        # Small kernels: 1 MXU, ~50% utilization → 4× penalty
        return 4.0
    elif sg.flops < 500e6:
        # Medium kernels: can start using both MXUs → 2× penalty
        return 2.0
    else:
        # Large kernels: both MXUs, good utilization
        return 1.0
```

### 4. RCA Documentation (`docs/bugs/tpu/`)

Updated all three RCA reports:
- `TPU_RCA_Report.md`
- `TPU_Fix_Implementation_Plan.md`
- `TPU_RCA_Final_Report.md`

**Replacements:**
- All "TensorCore" → "MXU"
- All "tensor_core" → "mxu"
- Updated architecture diagrams:
  ```
  TPU v4 Chip (Google)
  ├── MXU 0: 128×128 Systolic Array (16,384 MACs)
  │   └── 137.5 TFLOPS @ 2 GHz
  └── MXU 1: 128×128 Systolic Array (16,384 MACs)
      └── 137.5 TFLOPS @ 2 GHz
  ```

## Verification

### Test Results (ResNet18, batch=1, FP32)

```
Architecture    Energy       Latency      Throughput
CPU             408.59 mJ    1.48 ms      676 FPS
GPU             48.89 mJ     431.50 µs    2,317 FPS   ⭐ (speed)
KPU             5.76 mJ      461.04 µs    2,169 FPS   ⭐ (energy)
TPU             62.09 mJ     566.27 µs    1,766 FPS
DFM             99.02 mJ     58.77 ms     17 FPS
```

### Correctness Validation

✅ **TPU is slower than GPU at batch=1** (realistic!)
- TPU: 566 µs
- GPU: 431 µs
- TPU 31% slower (expected for small batches)

✅ **Terminology is correct**
- No NVIDIA-specific terms in TPU code
- Uses Google's official "MXU" terminology
- Architecture accurately described

✅ **No regressions**
- GPU still shows 431 µs (unchanged)
- KPU still shows 461 µs (unchanged)
- CPU still shows 1.48 ms (unchanged)

## Files Modified

### Source Code
1. `src/graphs/hardware/models/datacenter/tpu_v4.py` - Resource model
2. `src/graphs/hardware/mappers/accelerators/tpu.py` - Mapper implementation
3. `src/graphs/analysis/roofline.py` - Correction factors

### Documentation
4. `docs/bugs/tpu/TPU_RCA_Report.md` - Initial RCA
5. `docs/bugs/tpu/TPU_Fix_Implementation_Plan.md` - Implementation plan
6. `docs/bugs/tpu/TPU_RCA_Final_Report.md` - Final RCA with results

## Summary

All TPU v4 modeling now correctly uses Google's terminology:
- **MXU (Matrix Multiplier Unit)** instead of TensorCore
- Clear distinction from NVIDIA GPU hardware
- Accurate architectural representation

The fix maintains all performance improvements from the discrete resource allocation work while using the correct hardware terminology.

**Status:** ✅ Complete and verified
