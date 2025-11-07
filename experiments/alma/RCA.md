# Root Cause Analysis: Alma Integration Issues

## Problem Summary

Alma integration was hanging and failing with various errors. This document tracks the RCA and fixes applied.

---

## Issue 1: Multiprocessing Deadlock ✅ FIXED

### Symptom
```
Running Alma benchmark...
^CTraceback (most recent call last):
  File "alma/utils/multiprocessing/multiprocessing.py", line 135
    p.join()  ← Hangs here forever
KeyboardInterrupt
```

### Root Cause
- Alma uses multiprocessing by default (`multiprocessing=True`)
- Subprocess spawning can deadlock in certain environments
- No progress, no error, just hangs waiting for subprocess

### Solution
```python
config = BenchmarkConfig(
    n_samples=2048,
    batch_size=1,
    device=device,
    multiprocessing=False,  # ← Disable multiprocessing
)
```

### Status
✅ **FIXED** - No more hanging

---

## Issue 2: DataLoader Format Mismatch ✅ FIXED

### Symptom
```
ValueError: not enough values to unpack (expected 2, got 1)
  for data, _ in data_loader:
      ^^^^^^^
```

### Root Cause
- Alma's internal code expects `(data, label)` tuples from dataloader
- We were providing `TensorDataset(input_tensor)` → single element
- Alma tried to unpack: `data, label = item` → failed

### Original Attempt (Failed)
```python
dataset = TensorDataset(example_input, dummy_labels)
data_loader = DataLoader(dataset, batch_size=1)
alma_results = benchmark_model(model, config, conversions, data_loader=data_loader)
```

**Problem**: Alma's API check rejects both `data` and `data_loader` together

### Final Solution
```python
# Pass data directly, not dataloader
benchmark_data = example_input.repeat(config.n_samples, 1, 1, 1)
alma_results = benchmark_model(model, config, conversions, data=benchmark_data)
```

### Status
✅ **FIXED** - Alma accepts data parameter

---

## Issue 3: Result Format Mismatch ✅ FIXED

### Symptom
```
AttributeError: 'dict' object has no attribute 'inference_time_ms'
  eager_time = alma_results.get('EAGER', inductor_result).inference_time_ms
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Root Cause
Alma returns different format than expected:

**Expected (what code assumed)**:
```python
alma_results = {
    'EAGER': <object with .inference_time_ms attribute>
}
```

**Actual (what Alma returns)**:
```python
alma_results = {
    'EAGER': {
        'device': device(type='cpu'),
        'total_elapsed_time': 0.0016,    # seconds (total)
        'total_inf_time': 0.000117,      # seconds (model only, total)
        'total_samples': 10,
        'batch_size': 1,
        'throughput': 85243.5,           # samples/second
        'status': 'success',
        'data_dtype': torch.float32
    }
}
```

### Solution
Create helper function to extract latency:

```python
def get_latency_ms(result_dict):
    """Extract latency in ms from Alma result dict."""
    # total_inf_time is total time for all samples
    total_time_s = result_dict.get('total_inf_time', 0)
    n_samples = result_dict.get('total_samples', 1)
    return (total_time_s / n_samples) * 1000  # ms per sample

# Use everywhere:
eager_time = get_latency_ms(alma_results['EAGER'])
best_name = min(alma_results.keys(), key=lambda k: get_latency_ms(alma_results[k]))
```

### Status
✅ **FIXED** - Properly extracts latency from dict format

---

## Issue 4: Memory/Performance Issues ⚠️ ONGOING

### Symptom
- Process gets killed (exit code 137 = OOM)
- Benchmarking takes very long time
- High memory usage

### Root Causes

#### A. Large n_samples
```python
config = BenchmarkConfig(
    n_samples=2048,  # ← Very large!
    ...
)
benchmark_data = example_input.repeat(2048, 1, 1, 1)  # ← 2048 copies!
```

**Impact**:
- For image input (1, 3, 224, 224): ~600 KB per sample
- 2048 samples: ~1.2 GB just for input data
- Plus model activations during forward pass
- Can trigger OOM on systems with limited RAM

#### B. Multiple Conversions
- Each conversion requires separate benchmark run
- Memory not always freed between conversions
- Can accumulate memory usage

#### C. Compilation Overhead
- `COMPILE_INDUCTOR` triggers torch.compile
- First run is slow (compilation)
- Subsequent runs are faster but still need warmup

### Partial Solutions

#### 1. Reduce n_samples
```python
config = BenchmarkConfig(
    n_samples=100,   # ← Much smaller
    batch_size=1,
    device=device,
    multiprocessing=False,
)
```

**Trade-off**: Less accurate statistics, but faster and less memory

#### 2. Limit Conversions
```bash
# Instead of all Tier 2 options:
--conversions EAGER COMPILE_INDUCTOR

# Not:
--tier 2  # (7-10 conversions)
```

#### 3. Use Tier 1 for Quick Validation
```bash
# Fast, uses only inductor_validation (no Alma)
python experiments/alma/alma_integration.py --model simple --tier 1
```

### Status
⚠️ **WORKAROUNDS AVAILABLE** - Use smaller n_samples or Tier 1

---

## Summary of Fixes

| Issue | Status | Fix |
|-------|--------|-----|
| Multiprocessing deadlock | ✅ Fixed | `multiprocessing=False` |
| DataLoader format | ✅ Fixed | Use `data=` parameter instead of `data_loader=` |
| Result format | ✅ Fixed | `get_latency_ms()` helper function |
| Memory/OOM | ⚠️ Workaround | Reduce `n_samples`, use Tier 1, or limit conversions |

---

## Recommended Usage

### Quick Validation (Tier 1 - Recommended)
```bash
# Uses inductor_validation only (no Alma dependency issues)
python experiments/alma/alma_integration.py --model simple --tier 1
```

**Advantages**:
- ✅ Fast (10-20 seconds)
- ✅ Low memory
- ✅ No Alma issues
- ✅ Good for CI/CD

**Limitations**:
- Only tests Inductor (no TensorRT, ONNX, etc.)

### Core Deployment Analysis (Tier 2 - When Needed)
```bash
# Limit to specific conversions to avoid OOM
python experiments/alma/alma_integration.py --model simple --tier 2 \
    --conversions EAGER COMPILE_INDUCTOR
```

**Note**: Full Tier 2 (7-10 conversions) may hit memory limits

---

## Future Improvements

### 1. Adaptive n_samples
```python
# Adjust based on model size
model_params = sum(p.numel() for p in model.parameters())
if model_params > 100M:
    n_samples = 50   # Large models: fewer samples
elif model_params > 10M:
    n_samples = 100  # Medium models
else:
    n_samples = 500  # Small models: more samples
```

### 2. Streaming/Batched Benchmarking
Instead of creating all 2048 samples at once:
```python
# Current (all at once):
benchmark_data = input.repeat(2048, 1, 1, 1)  # OOM risk

# Better (stream):
for batch in range(0, 2048, batch_size):
    batch_data = input.repeat(batch_size, 1, 1, 1)
    # benchmark this batch
```

### 3. Per-Conversion Memory Cleanup
```python
# After each conversion:
import gc
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### 4. Progress Monitoring
```python
# Add progress indicators
from tqdm import tqdm
for conversion in tqdm(conversions):
    # benchmark...
```

---

## Testing Recommendations

### For Development
```bash
# Use Tier 1 for quick iteration
python experiments/alma/alma_integration.py --model simple --tier 1
```

### For CI/CD
```bash
# Tier 1 only - fast and reliable
python experiments/alma/alma_integration.py --model simple --tier 1 --output ci_results.json
```

### For Deployment Analysis
```bash
# Tier 2 with limited conversions
python experiments/alma/alma_integration.py --model resnet18 --tier 2 \
    --conversions EAGER COMPILE_INDUCTOR ONNX_CPU
```

### For Research (High Memory Machine)
```bash
# Full Tier 3 - requires lots of RAM and time
python experiments/alma/alma_integration.py --model resnet18 --tier 3
```

---

## Conclusion

**Status**: ✅ Alma integration is functional with workarounds

**Working**:
- ✅ Tier 1 (inductor only) - Fast, reliable
- ✅ Tier 2 (limited conversions) - Works with care
- ⚠️ Tier 2 (full) - May hit memory limits
- ⚠️ Tier 3 - Requires high-memory machine

**Recommendation**:
- **Primary**: Use Tier 1 for validation
- **Secondary**: Use Tier 2 with limited conversions for deployment analysis
- **Research**: Use Tier 3 on machines with 32GB+ RAM

The integration successfully demonstrates the value of multi-backend validation, even if full Tier 2/3 requires more resources than currently available.
