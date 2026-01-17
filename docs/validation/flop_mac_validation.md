# FLOP/MAC Validation and Bug Fix

## Summary

Successfully identified and fixed a critical bug in FLOP counting, achieving 99.73% agreement with fvcore (Facebook's FLOP counting library).

## The Bug

**Location**: `src/graphs/characterize/graph_partitioner.py`, line 225 in `_compute_flops()`

**Problem**: The code was using the **output shape** instead of **input shape** for FLOP calculations.

```python
# BEFORE (WRONG)
B, C_in, H, W = meta.shape  # meta.shape is OUTPUT shape!

# AFTER (CORRECT)
if node.args and hasattr(node.args[0], 'meta') and 'tensor_meta' in node.args[0].meta:
    input_meta = node.args[0].meta['tensor_meta']
    B, C_in, H, W = input_meta.shape  # Get INPUT shape from arguments
```

**Impact**: For ResNet-18's first conv layer:
- Wrong calculation: Used output shape [1, 64, 112, 112] as input
- Resulted in 629M MACs (5.33x too high)
- Correct: Using input shape [1, 3, 224, 224] gives 118M MACs ✓

## MACs vs FLOPs Convention

### Technical Definitions
- **MAC (Multiply-Accumulate)**: A single fused operation: `y += a * b`
- **FLOP (Floating Point Operation)**: Individual operations (1 MAC = 1 multiply + 1 add = 2 FLOPs)

### Community Convention
The ML community convention is **confusing but standard**:
- Most tools (fvcore, torchinfo, etc.) count **MACs**
- But they **label them as "FLOPs"** or "MAdd" (multiply-adds)
- Published ResNet-18 values: ~1.8 "GFLOPs" (actually GMACs)

### Our Tool
- **Technically correct**: FLOPs = 2 × MACs
- **Reports both**:
  - FLOPs: 3.643 GFLOPs (2× MACs)
  - MACs: 1.814 GMACs
- **Comparison**: Our MACs vs fvcore shows 0.27% difference ✓

## Validation Results

### ResNet-18 (224×224 input)

| Tool | Count | Type | Notes |
|------|-------|------|-------|
| Our Tool | 3.643 GFLOPs | FLOPs | 2× MACs (technically correct) |
| Our Tool | 1.814 GMACs | MACs | Standard ML convention |
| FVCore | 1.819 GFLOPs | MACs* | *Labeled as FLOPs but counts MACs |
| **Difference** | **0.27%** | | **Excellent match!** |

### Published Values
- Various sources: 1.8-1.9 "GFLOPs" (MACs)
- Some sources: 3.6-4.5 GFLOPs (actual FLOPs = 2× MACs)
- Our values match both conventions ✓

## What We Count

### Operations Included
| Operation | Our Tool | FVCore |
|-----------|----------|--------|
| Conv2d MACs | ✓ | ✓ |
| Linear MACs | ✓ | ✓ |
| BatchNorm | ✓ (5 FLOPs/element) | ✗ |
| ReLU | ✓ (1 FLOP/element) | ✗ |
| MaxPool | ✗ (no FLOPs) | ✗ |
| Residual Add | ✗ (not in call_module) | ✗ |

### Extra Operations
For ResNet-18:
- BatchNorm: 12.4M FLOPs (20 layers)
- ReLU: 2.3M FLOPs (17 layers)
- **Total extra: 0.015 GFLOPs (0.4% of total)**

## Formulas Used

### Conv2d
```python
# Input shape: [B, C_in, H_in, W_in]
# Output shape: [B, C_out, H_out, W_out]

H_out = (H_in + 2*padding - kernel_h) // stride_h + 1
W_out = (W_in + 2*padding - kernel_w) // stride_w + 1

# Standard convolution
MACs = B × C_out × H_out × W_out × (C_in/groups) × K_h × K_w
FLOPs = 2 × MACs

# Depthwise convolution (groups == C_in == C_out)
MACs = B × C_out × H_out × W_out × K_h × K_w
FLOPs = 2 × MACs
```

### Linear
```python
# Input shape: [B, D_in]
# Output shape: [B, D_out]

MACs = B × D_in × D_out
FLOPs = 2 × MACs
```

## Testing Tools Created

1. **`tests/validate_arithmetic_intensity.py`**
   - Comprehensive validation suite with 7 tests
   - Tests Conv2d, Linear, BatchNorm, ReLU
   - Validates ResNet-18 total FLOPs
   - Checks AI ranges and fusion benefits

2. **`cli/fvcore_compare.py`**
   - Cross-validates against fvcore
   - Shows both MACs and FLOPs
   - Explains convention differences
   - Supports multiple models

3. **`cli/debug_flop_mismatch.py`**
   - Detailed breakdown by operation type
   - Hypothesis testing
   - Identifies what fvcore counts/doesn't count

4. **`cli/debug_single_conv.py`**
   - Deep dive into single layer calculation
   - Shows input vs output shape issue
   - Demonstrates bug and fix

## Usage Recommendations

### For ML Community Alignment
```python
# Report MACs (what most tools call "FLOPs")
report = partitioner.partition(fx_graph)
print(f"MACs: {report.total_macs / 1e9:.2f} GMACs")
```

### For Technical Accuracy
```python
# Report both, clearly labeled
print(f"MACs: {report.total_macs / 1e9:.2f} GMACs")
print(f"FLOPs (2×MACs): {report.total_flops / 1e9:.2f} GFLOPs")
```

### For Comparison with Published Values
- If source says "FLOPs" around 1.8G for ResNet-18 → they mean MACs
- If source says "FLOPs" around 3.6G for ResNet-18 → they mean actual FLOPs
- Always check if value is ~2× to determine convention

## Key Takeaways

1. ✅ **Bug Fixed**: Now correctly uses input shape for FLOP calculations
2. ✅ **Validated**: 0.27% agreement with fvcore on ResNet-18
3. ✅ **Clear Reporting**: Shows both MACs and FLOPs with explanations
4. ✅ **Convention Aware**: Understands ML community "FLOPs" = MACs
5. ✅ **Comprehensive**: Also counts activations and batch norm (optional)

## Future Work

- [ ] Add flag to exclude activations/batch norm from count
- [ ] Improve per-layer matching in fvcore_compare.py
- [ ] Add support for more model architectures
- [ ] Create visualization of compute vs memory-bound operations
- [ ] Add support for quantized models (INT8 operations)
