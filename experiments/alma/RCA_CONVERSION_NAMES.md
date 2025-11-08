# RCA: Missing Conversions in Alma Benchmarking

## Issue

Only 2 of 4 requested conversions were running:
- ✓ `EAGER` - Running
- ✗ `COMPILE_INDUCTOR` - **NOT running (silently filtered)**
- ✓ `ONNX_CPU` - Running
- ✗ `OPENVINO` - **NOT running (silently filtered)**

## Root Cause

**Incorrect conversion names** - Alma uses specific string names from `MODEL_CONVERSION_OPTIONS` dictionary.

The conversions were silently filtered out because:
1. `COMPILE_INDUCTOR` is not a valid name → Should be `COMPILE_INDUCTOR_DEFAULT`
2. `OPENVINO` is not a valid name → Should be `COMPILE_OPENVINO`

Alma does **NOT** raise an error for invalid conversion names - it simply skips them silently.

## Investigation Steps

### Step 1: Check what conversions ran
```python
results = benchmark_model(model, config, conversions, data_loader)
print(results.keys())  # Only ['EAGER', 'ONNX_CPU']
```

### Step 2: Check Alma's available conversions
```python
from alma.conversions.select import MODEL_CONVERSION_OPTIONS

# Print all available modes
for value in MODEL_CONVERSION_OPTIONS.values():
    print(value)
```

### Step 3: Find the correct names
```python
# Search for inductor variants
modes = set()
for value in MODEL_CONVERSION_OPTIONS.values():
    mode = str(value).split(' ')[0].replace('mode=', '').replace("'", '')
    if 'INDUCTOR' in mode:
        modes.add(mode)

# Results:
# COMPILE_INDUCTOR_DEFAULT
# COMPILE_INDUCTOR_REDUCE_OVERHEAD
# COMPILE_INDUCTOR_MAX_AUTOTUNE
# COMPILE_INDUCTOR_EAGER_FALLBACK
# etc.
```

## Solution

### Corrected Conversion Names

**Before (incorrect):**
```python
cpu_conversions = [
    "EAGER",
    "COMPILE_INDUCTOR",       # ✗ WRONG - silently skipped
    "ONNX_CPU",
    "OPENVINO",               # ✗ WRONG - silently skipped
]
```

**After (correct):**
```python
cpu_conversions = [
    "EAGER",
    "COMPILE_INDUCTOR_DEFAULT",   # ✓ CORRECT
    "ONNX_CPU",
    "COMPILE_OPENVINO",           # ✓ CORRECT
]
```

### Results After Fix

All 4 conversions now run successfully:

```
EAGER results:
  Latency: 2.149 ms
  Throughput: 465.26 samples/second

ONNX_CPU results:
  Latency: 0.491 ms  (4.38x vs EAGER)
  Throughput: 2038.61 samples/second

COMPILE_INDUCTOR_DEFAULT results:
  Latency: 0.494 ms  (4.35x vs EAGER)
  Throughput: 2023.62 samples/second

COMPILE_OPENVINO results:
  Latency: 0.499 ms  (4.31x vs EAGER)
  Throughput: 2003.71 samples/second
```

## Available Alma Conversion Names (v0.3.7)

### CPU-Compatible Conversions

| Category | Conversion Name | Notes |
|----------|-----------------|-------|
| **Baseline** | `EAGER` | Standard PyTorch |
| | `EXPORT+EAGER` | Exported then eager |
| **Inductor** | `COMPILE_INDUCTOR_DEFAULT` | Default torch.compile |
| | `COMPILE_INDUCTOR_REDUCE_OVERHEAD` | Less overhead |
| | `COMPILE_INDUCTOR_MAX_AUTOTUNE` | Aggressive tuning |
| | `COMPILE_INDUCTOR_EAGER_FALLBACK` | Fallback on error |
| **ONNX** | `ONNX_CPU` | ONNX Runtime CPU |
| | `ONNX+DYNAMO_EXPORT` | Dynamo export to ONNX |
| **OpenVINO** | `COMPILE_OPENVINO` | Intel OpenVINO |
| | `FP16+COMPILE_OPENVINO` | OpenVINO with FP16 |
| **TorchScript** | `JIT_TRACE` | JIT trace |
| | `TORCH_SCRIPT` | TorchScript |
| **Export+Compile** | `EXPORT+COMPILE_INDUCTOR_DEFAULT` | Export then compile |
| | `EXPORT+AOT_INDUCTOR` | AOT inductor |

### GPU-Only Conversions (Don't use on CPU)

- `ONNX_GPU` - ONNX Runtime GPU (requires CUDA)
- `COMPILE_CUDAGRAPHS` - CUDA Graphs (requires CUDA)
- `COMPILE_TENSORRT` - TensorRT (requires CUDA + TensorRT)
- `COMPILE_OPENXLA` - OpenXLA (requires XLA)

### Full List (72 conversions)

See `MODEL_CONVERSION_OPTIONS` dictionary for complete list:

```python
from alma.conversions.select import MODEL_CONVERSION_OPTIONS
for key, value in MODEL_CONVERSION_OPTIONS.items():
    print(f"{key}: {value}")
```

## Common Naming Mistakes

| Wrong Name | Correct Name | Notes |
|------------|--------------|-------|
| `COMPILE_INDUCTOR` | `COMPILE_INDUCTOR_DEFAULT` | Need `_DEFAULT` suffix |
| `OPENVINO` | `COMPILE_OPENVINO` | Need `COMPILE_` prefix |
| `ONNX` | `ONNX_CPU` or `ONNX_GPU` | Need device suffix |
| `INDUCTOR` | `COMPILE_INDUCTOR_DEFAULT` | Need full name |
| `TENSORRT` | `COMPILE_TENSORRT` | Need `COMPILE_` prefix |

## Debugging Silent Filtering

### Check if conversion will run

```python
from alma.conversions.select import MODEL_CONVERSION_OPTIONS

# Get all valid mode names
valid_modes = set()
for value in MODEL_CONVERSION_OPTIONS.values():
    mode = str(value).split(' ')[0].replace('mode=', '').replace("'", '')
    valid_modes.add(mode)

# Check your conversions
my_conversions = ["EAGER", "COMPILE_INDUCTOR", "ONNX_CPU", "OPENVINO"]
for conv in my_conversions:
    is_valid = conv in valid_modes
    print(f"{conv}: {'✓ VALID' if is_valid else '✗ INVALID (will be skipped!)'}")
```

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now Alma might print warnings about invalid conversions
results = benchmark_model(model, config, conversions, data_loader)
```

## Recommendations

1. **Always validate conversion names** before running long benchmarks
2. **Check results keys** to see what actually ran: `print(results.keys())`
3. **Use exact names** from `MODEL_CONVERSION_OPTIONS`
4. **Search for variants** if unsure: `grep -i "inductor" <(list all modes)`

## Prevention

Add this validation at the start of your script:

```python
from alma.conversions.select import MODEL_CONVERSION_OPTIONS

def validate_conversions(conversions):
    """Validate conversion names before benchmarking."""
    valid_modes = set()
    for value in MODEL_CONVERSION_OPTIONS.values():
        mode = str(value).split(' ')[0].replace('mode=', '').replace("'", '')
        valid_modes.add(mode)

    invalid = []
    for conv in conversions:
        if conv not in valid_modes:
            invalid.append(conv)

    if invalid:
        print(f"⚠️  WARNING: Invalid conversions (will be skipped):")
        for inv in invalid:
            print(f"  - {inv}")
            # Suggest alternatives
            suggestions = [m for m in valid_modes if inv.upper() in m.upper()]
            if suggestions:
                print(f"    Did you mean: {suggestions[:3]}")
        return False
    return True

# Use it
conversions = ["EAGER", "COMPILE_INDUCTOR", "ONNX_CPU", "OPENVINO"]
if not validate_conversions(conversions):
    print("Fix conversion names before proceeding!")
```

## Impact

**Before fix:**
- Only 2/4 conversions running
- Missing COMPILE_INDUCTOR performance data (4.35x speedup!)
- Missing COMPILE_OPENVINO performance data (4.31x speedup!)
- No error/warning from Alma

**After fix:**
- All 4/4 conversions running ✓
- Complete performance comparison
- Found best backend (ONNX_CPU: 4.38x speedup)

---

**Date**: 2025-11-07
**Alma Version**: 0.3.7
**Issue**: Silent filtering of invalid conversion names
**Status**: ✅ Fixed in `cpu_minimal_example.py`
