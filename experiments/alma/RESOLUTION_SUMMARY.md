# Resolution Summary: Missing Alma Conversions

## Issue Reported

User was only getting 2 of 4 requested conversions:
- ✓ EAGER - Running
- ✗ COMPILE_INDUCTOR - **Missing**
- ✓ ONNX_CPU - Running  
- ✗ OPENVINO - **Missing**

## Root Cause

**Incorrect conversion names** - Alma requires exact string matches from its `MODEL_CONVERSION_OPTIONS` dictionary.

The conversions were silently filtered out because:
1. `COMPILE_INDUCTOR` is not valid → Should be `COMPILE_INDUCTOR_DEFAULT`
2. `OPENVINO` is not valid → Should be `COMPILE_OPENVINO`

**Critical Issue**: Alma does NOT raise errors for invalid names - it silently skips them!

## Investigation Process

### 1. Verified the issue
```bash
python3 experiments/alma/cpu_minimal_example.py 2>&1 | grep "results:"
# Only showed EAGER and ONNX_CPU
```

### 2. Discovered Alma's conversion registry
```python
from alma.conversions.select import MODEL_CONVERSION_OPTIONS
# Contains 72 conversion options with specific naming
```

### 3. Found correct names
```python
# COMPILE_INDUCTOR → COMPILE_INDUCTOR_DEFAULT
# OPENVINO → COMPILE_OPENVINO
```

## Solution Implemented

### Code Fix

**File**: `experiments/alma/cpu_minimal_example.py` (Lines 207-212)

**Before:**
```python
cpu_conversions = [
    "EAGER",
    "COMPILE_INDUCTOR",      # ✗ Invalid - silently skipped
    "ONNX_CPU",
    "OPENVINO",              # ✗ Invalid - silently skipped
]
```

**After:**
```python
cpu_conversions = [
    "EAGER",                      # ✓ Valid
    "COMPILE_INDUCTOR_DEFAULT",   # ✓ Valid - FIXED
    "ONNX_CPU",                   # ✓ Valid
    "COMPILE_OPENVINO",           # ✓ Valid - FIXED
]
```

### Documentation Created

1. **RCA_CONVERSION_NAMES.md** - Full analysis with:
   - Root cause explanation
   - Complete list of valid conversion names
   - Common naming mistakes
   - Validation code snippet

2. **Updated QUICKREF.md** - Added warning about common mistakes

3. **Updated README.md** - Added note about correct names

## Verification

### Before Fix
```
Testing 4 CPU conversions:
  - EAGER
  - COMPILE_INDUCTOR
  - ONNX_CPU
  - OPENVINO

Running Alma benchmark...
Benchmarking EAGER on cpu: 100%
Benchmarking ONNX_CPU on cpu: 100%

Results: Only 2 conversions ran
```

### After Fix
```
Testing 4 CPU conversions:
  - EAGER
  - COMPILE_INDUCTOR_DEFAULT
  - ONNX_CPU
  - COMPILE_OPENVINO

Running Alma benchmark...
Benchmarking EAGER on cpu: 100%
Benchmarking ONNX_CPU on cpu: 100%
Benchmarking COMPILE_INDUCTOR_DEFAULT on cpu: 100%  ← NEW!
Benchmarking COMPILE_OPENVINO on cpu: 100%          ← NEW!

Results: All 4 conversions ran successfully ✓
```

## Performance Results (i7-12700K)

| Conversion | Latency | Throughput | Speedup |
|------------|---------|------------|---------|
| **EAGER** (baseline) | 2.15 ms | 465 inf/s | 1.0x |
| **ONNX_CPU** | 0.49 ms | 2039 inf/s | **4.38x** ⭐ |
| **COMPILE_INDUCTOR_DEFAULT** | 0.49 ms | 2024 inf/s | **4.35x** |
| **COMPILE_OPENVINO** | 0.50 ms | 2004 inf/s | **4.31x** |

**Key Finding**: All optimized backends provide ~4.3x speedup on CPU!

## Common Conversion Name Mistakes

| ❌ Wrong | ✅ Correct | Backend |
|---------|-----------|---------|
| `COMPILE_INDUCTOR` | `COMPILE_INDUCTOR_DEFAULT` | torch.compile |
| `OPENVINO` | `COMPILE_OPENVINO` | OpenVINO |
| `INDUCTOR` | `COMPILE_INDUCTOR_DEFAULT` | torch.compile |
| `TENSORRT` | `COMPILE_TENSORRT` | TensorRT |
| `ONNX` | `ONNX_CPU` or `ONNX_GPU` | ONNX Runtime |

## Valid CPU Conversion Names (Alma v0.3.7)

### Always Available
- `EAGER`
- `COMPILE_INDUCTOR_DEFAULT`
- `COMPILE_INDUCTOR_REDUCE_OVERHEAD`
- `COMPILE_INDUCTOR_MAX_AUTOTUNE`
- `JIT_TRACE`
- `TORCH_SCRIPT`

### Requires Packages
- `ONNX_CPU` - Requires: `pip install onnxruntime`
- `COMPILE_OPENVINO` - Requires: `pip install openvino`
- `COMPILE_TVM` - Requires TVM installation
- `COMPILE_ONNXRT` - Requires ONNX Runtime

### GPU Only (Don't use on CPU)
- `ONNX_GPU`
- `COMPILE_CUDAGRAPHS`
- `COMPILE_TENSORRT`
- `COMPILE_OPENXLA`

## Prevention: Validation Function

Add this to your scripts to catch invalid names:

```python
from alma.conversions.select import MODEL_CONVERSION_OPTIONS

def validate_conversions(conversions):
    """Validate conversion names before running."""
    valid_modes = set()
    for value in MODEL_CONVERSION_OPTIONS.values():
        mode = str(value).split(' ')[0].replace('mode=', '').replace("'", '')
        valid_modes.add(mode)
    
    invalid = [c for c in conversions if c not in valid_modes]
    
    if invalid:
        print(f"⚠️  Invalid conversions (will be skipped): {invalid}")
        for inv in invalid:
            suggestions = [m for m in valid_modes if inv.upper() in m.upper()]
            if suggestions:
                print(f"  Did you mean: {suggestions[:3]}")
        return False
    return True

# Usage
conversions = ["EAGER", "COMPILE_INDUCTOR", "ONNX_CPU"]
if not validate_conversions(conversions):
    print("Fix names before proceeding!")
```

## Impact

**Before:**
- ❌ Only 2/4 backends running
- ❌ Missing ~4x performance improvement
- ❌ Incomplete comparison
- ❌ No error/warning from Alma

**After:**
- ✅ All 4/4 backends running
- ✅ Complete performance data
- ✅ Found best backend (ONNX_CPU)
- ✅ Documented for future users

## Lessons Learned

1. **Alma silently filters invalid names** - No errors raised
2. **Always check `results.keys()`** - Verify what actually ran
3. **Use exact names** from MODEL_CONVERSION_OPTIONS
4. **Add validation** - Catch errors before long benchmarks
5. **Document correct names** - Prevent future mistakes

## Files Modified

1. `cpu_minimal_example.py` - Fixed conversion names (Lines 207-212)
2. `RCA_CONVERSION_NAMES.md` - Created (full analysis)
3. `QUICKREF.md` - Updated (added warnings)
4. `README.md` - Updated (added notes)
5. `RESOLUTION_SUMMARY.md` - Created (this file)

## Testing

```bash
# Verify all 4 conversions run
python3 experiments/alma/cpu_minimal_example.py 2>&1 | grep -c "Benchmarking"
# Should output: 4

# Check results
python3 experiments/alma/cpu_minimal_example.py 2>&1 | grep "results:"
# Should show all 4: EAGER, COMPILE_INDUCTOR_DEFAULT, ONNX_CPU, COMPILE_OPENVINO
```

## References

- Issue: Only 2/4 conversions running
- Root Cause: Invalid conversion names
- Solution: Use exact names from MODEL_CONVERSION_OPTIONS
- Status: ✅ Resolved and documented

---

**Date**: 2025-11-07
**Reporter**: User (i7-12700K CPU-only server)
**Resolver**: Claude Code
**Alma Version**: 0.3.7
**Status**: ✅ Resolved
