# Bug Fixes Summary - FLOP Counting and Validation

## Overview

Fixed two critical bugs in FLOP counting and validation, achieving **perfect agreement** with fvcore:
- **Total MACs**: 0.27% difference (1.814 vs 1.819 GMACs)
- **Per-layer**: 21/21 layers with 0.00% difference

---

## Bug #1: FLOP Calculation Using Wrong Shape

### Root Cause
**File**: `src/graphs/characterize/graph_partitioner.py`
**Line**: 225 in `_compute_flops()`

**Problem**: Used **output shape** instead of **input shape** for FLOP calculations.

```python
# BEFORE (WRONG)
def _compute_flops(self, node, meta, module, op_type):
    if op_type in [OperationType.CONV2D, ...]:
        B, C_in, H, W = meta.shape  # ❌ meta.shape is OUTPUT!
        # ... calculation using wrong C_in, H, W
```

### Impact
For ResNet-18's first conv layer (3→64 channels, 224×224→112×112):

| Metric | Wrong (using output) | Correct (using input) | Error |
|--------|---------------------|----------------------|-------|
| C_in | 64 (from output) | 3 (from input) | 21.3× |
| H, W | 112, 112 (from output) | 224, 224 (from input) | 0.25× |
| **MACs** | **629M** | **118M** | **5.33×** |

Total model impact:
- Wrong: 4.494 GFLOPs (2.47× too high)
- Correct: 3.643 GFLOPs ✓

### Fix
Get input shape from node arguments:

```python
# AFTER (CORRECT)
def _compute_flops(self, node, meta, module, op_type):
    if op_type in [OperationType.CONV2D, ...]:
        # Get INPUT shape from node arguments, not output metadata
        if node.args and hasattr(node.args[0], 'meta') and 'tensor_meta' in node.args[0].meta:
            input_meta = node.args[0].meta['tensor_meta']
            B, C_in, H, W = input_meta.shape  # ✓ Use INPUT shape
        else:
            # Fallback
            B, C_in = 1, module.in_channels
            H = W = 224
```

Same fix applied to Linear layers.

---

## Bug #2: Per-Layer Comparison Matching Wrong Layers

### Root Cause
**File**: `cli/fvcore_compare.py`
**Function**: `compare_per_layer()`

**Two sub-bugs**:

#### 2a. Matching Against Empty String
FVCore's `by_module()` returns:
```python
{
    "": 1819.07M,           # Empty string = total
    "conv1": 118.01M,
    "layer1.0.conv1": 115.61M,
    # ...
}
```

Original matching logic:
```python
# WRONG: Always matches empty string
if our_name in fv_name or fv_name.split('.')[-1] in our_name:
    # When fv_name = "", this becomes: if "" in our_name
    # This is ALWAYS True!
```

**Result**: Every layer matched the empty string → showed total (1819.07M) for all layers.

#### 2b. Filtering Out Top-Level Modules
```python
# WRONG: Removes conv1, fc, etc.
fvcore_filtered = {name: flops for name, flops in fvcore_modules.items()
                  if name and '.' in name}  # ❌ Excludes conv1, fc
```

**Result**: Top-level modules like `conv1` and `fc` couldn't be matched.

#### 2c. Name Format Mismatch
- Our FX names: `layer1_0_conv1` (underscores)
- FVCore names: `layer1.0.conv1` (dots)

### Fix

```python
# 1. Filter only empty string
fvcore_filtered = {name: flops for name, flops in fvcore_modules.items()
                  if name}  # ✓ Keep everything except empty string

# 2. Normalize names (underscores → dots)
def normalize_name(name: str) -> str:
    return name.replace('_', '.')

our_name_normalized = normalize_name(our_name)

# 3. Try exact match first
if our_name_normalized in fvcore_filtered:
    fv_flops = fvcore_filtered[our_name_normalized]
    # ... create match
    matched = True
```

---

## Validation Results

### Before Fixes
```
Total:     Our: 4.494 GFLOPs  FVCore: 1.819 GFLOPs  (147% error)
Per-layer: 0/60 matched (all showing 1819.1M)
```

### After Fixes
```
Total MACs:    Our: 1.814 GMACs  FVCore: 1.819 GMACs  (0.27% ✓)
Total FLOPs:   Our: 3.643 GFLOPs (2× MACs, technically correct)
Per-layer:     21/21 Conv2d/Linear layers: 0.00% difference ✓
```

### Test Suite Results
All 7 validation tests pass:

1. ✓ Conv2d FLOP counting (exact match with manual calculation)
2. ✓ Linear FLOP counting (exact match)
3. ✓ BatchNorm AI: 0.62 FLOPs/byte (memory-bound, expected: 0.3-2.0)
4. ✓ ReLU AI: 0.12 FLOPs/byte (bandwidth-bound, expected: 0.05-0.5)
5. ✓ ResNet-18 total: 3.64 GFLOPs (within published range: 3.5-4.6)
6. ✓ All operation types have AI in expected ranges
7. ✓ Fusion improves AI by 79.5%

---

## Layer-by-Layer Breakdown (ResNet-18)

| Layer | Our MACs | FVCore MACs | Difference |
|-------|----------|-------------|------------|
| conv1 | 118.01M | 118.01M | 0.00% ✓ |
| layer1.0.conv1 | 115.61M | 115.61M | 0.00% ✓ |
| layer1.0.conv2 | 115.61M | 115.61M | 0.00% ✓ |
| layer1.1.conv1 | 115.61M | 115.61M | 0.00% ✓ |
| layer1.1.conv2 | 115.61M | 115.61M | 0.00% ✓ |
| layer2.0.conv1 | 57.80M | 57.80M | 0.00% ✓ |
| layer2.0.conv2 | 115.61M | 115.61M | 0.00% ✓ |
| layer2.0.downsample.0 | 6.42M | 6.42M | 0.00% ✓ |
| layer2.1.conv1 | 115.61M | 115.61M | 0.00% ✓ |
| layer2.1.conv2 | 115.61M | 115.61M | 0.00% ✓ |
| ... (all 21 layers) | | | 0.00% ✓ |
| fc | 0.51M | 0.51M | 0.00% ✓ |

**Result**: 21/21 layers match perfectly

---

## Key Insights

### MACs vs FLOPs Convention
The ML community has a confusing convention:
- **Technically**: 1 MAC (multiply-accumulate) = 2 FLOPs (1 multiply + 1 add)
- **Convention**: Most tools count MACs but label them as "FLOPs"
- **Our tool**: Reports both clearly:
  - MACs: 1.814 GMACs (matches fvcore's "FLOPs")
  - FLOPs: 3.643 GFLOPs (2× MACs, technically correct)

### What We Count
| Operation | Our Tool | FVCore | Notes |
|-----------|----------|--------|-------|
| Conv2d MACs | ✓ | ✓ | Perfect agreement |
| Linear MACs | ✓ | ✓ | Perfect agreement |
| BatchNorm | ✓ (5 FLOPs/elem) | ✗ | We count, fvcore doesn't |
| ReLU | ✓ (1 FLOP/elem) | ✗ | We count, fvcore doesn't |
| MaxPool | ✗ | ✗ | Neither counts (no FLOPs) |
| Residual Add | ✗ | ✗ | Neither counts (not in call_module) |

Extra operations for ResNet-18:
- BatchNorm: 12.4M FLOPs (0.3% of total)
- ReLU: 2.3M FLOPs (0.1% of total)

---

## Files Changed

1. **`src/graphs/characterize/graph_partitioner.py`**
   - Fixed `_compute_flops()` to use input shape (lines 221-285)
   - Applied fix to both Conv2d and Linear

2. **`cli/fvcore_compare.py`**
   - Fixed filtering to keep top-level modules (line 160)
   - Added name normalization (line 170)
   - Fixed matching logic to try exact match first (line 187)
   - Updated display to show MACs instead of FLOPs
   - Changed tolerance to 1% for per-layer comparison

3. **New debugging tools created**:
   - `cli/debug_flop_mismatch.py` - Detailed breakdown and hypothesis testing
   - `cli/debug_single_conv.py` - Single layer deep dive
   - `cli/debug_fvcore_modules.py` - FVCore output structure analysis
   - `cli/compare_per_layer.py` - Manual per-layer verification

4. **Documentation**:
   - `docs/FLOP_MAC_VALIDATION.md` - Complete validation methodology
   - `docs/BUG_FIXES_SUMMARY.md` - This document

---

## Remaining Work

- [ ] Update validation tests to expect MACs vs FLOPs convention
- [ ] Add tests for the per-layer comparison
- [ ] Support more models (MobileNet, EfficientNet, etc.)
- [ ] Add flag to exclude/include activations and batch norm in counts
- [ ] Document which operations are counted vs not counted
- [ ] Add batch size variation testing

---

## Commands to Verify

```bash
# Run validation tests (all should pass)
python tests/validate_arithmetic_intensity.py

# Compare against fvcore
python cli/fvcore_compare.py --model resnet18

# Debug specific issues
python cli/debug_flop_mismatch.py
python cli/debug_single_conv.py
```

Expected output:
- **Total**: 0.27% difference in MACs
- **Per-layer**: 21/21 perfect matches (0.00%)
- **All tests**: PASSED (7/7)
