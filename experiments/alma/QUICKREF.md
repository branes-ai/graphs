# Alma CPU Benchmarking - Quick Reference Card

## TL;DR - Run This First

```bash
# SimpleCNN (default, fast - ~10 seconds)
python3 experiments/alma/cpu_minimal_example.py

# ResNet50
python3 experiments/alma/cpu_minimal_example.py --model resnet50

# Vision Transformer
python3 experiments/alma/cpu_minimal_example.py --model vit-b-16

# Show all options
python3 experiments/alma/cpu_minimal_example.py --help
```

Expected: Completes in ~10-60 seconds depending on model, shows CPU config and benchmark results.

---

## Critical Configuration Lines

### 1. CPU Thread Count (Line 44)
```python
num_threads = min(12, cpu_count)  # Use physical cores
torch.set_num_threads(num_threads)
```
**Adjust**: Change `12` to your CPU's physical core count.

### 2. Alma CPU Config (Lines 228-233)
```python
config = BenchmarkConfig(
    device=torch.device('cpu'),     # ← MUST BE CPU
    allow_cuda=False,                # ← MUST BE FALSE
    allow_device_override=False      # ← MUST BE FALSE
)
```
**Don't change these!** They force CPU-only execution.

### 3. Sample Count (Line 223)
```python
n_samples = 128  # Small count to avoid memory issues
```
**Adjust**: Increase to 256/512 if you have more RAM, decrease to 64 if OOM.

---

## Where Platform is Configured

| What | Where | Critical? |
|------|-------|-----------|
| CPU threads | Line 44 | ✅ Yes - affects performance |
| Environment vars | Lines 46-50 | ✅ Yes - MKL/OpenBLAS config |
| Alma device | Line 228 | ✅ Yes - force CPU |
| CUDA disable | Line 229 | ✅ Yes - prevent GPU |
| Device override | Line 233 | ✅ Yes - prevent auto-select |
| Sample count | Line 223 | ⚠️ Optional - memory control |
| Multiprocessing | Line 231 | ⚠️ Optional - avoid hangs |

---

## Runtime Checks (Auto-Printed)

**CPU Environment Section:**
```
System CPU count: 20
PyTorch threads set to: 12        ← Should match physical cores
CUDA available: False             ← Should be False
Test tensor device: cpu           ← Should be cpu
```

**Alma Configuration Section:**
```
✓ Device: cpu                     ← Should be cpu
✓ CUDA disabled: allow_cuda=False ← Should be False
✓ Device override: False          ← Should be False
```

If any of these don't match, check your configuration!

---

## Tested Backends

**Default (No extra install):**
- `EAGER` - PyTorch eager mode ✅
- `COMPILE_INDUCTOR_DEFAULT` - torch.compile ✅ (NOTE: Use `_DEFAULT` suffix!)

**Optional (Require packages):**
- `ONNX_CPU` - `pip install onnxruntime` ✅
- `COMPILE_OPENVINO` - `pip install openvino` (Intel CPUs only) ✅ (NOTE: Use `COMPILE_` prefix!)

**⚠️ Common Mistake**: Using `COMPILE_INDUCTOR` or `OPENVINO` without proper suffixes/prefixes
- Wrong: `COMPILE_INDUCTOR` → Correct: `COMPILE_INDUCTOR_DEFAULT`
- Wrong: `OPENVINO` → Correct: `COMPILE_OPENVINO`

See `RCA_CONVERSION_NAMES.md` for full list of valid conversion names.

---

## Modify for Your Model

Find this section (around line 380):

```python
# ===== REPLACE THIS =====
model = SimpleCNN()
model.eval()
model = model.to(device)

input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
# ========================

# With your model:
from torchvision.models import resnet18
model = resnet18(weights=None)
model.eval()
model = model.to(device)

input_tensor = torch.randn(1, 3, 224, 224, device=device)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory** | Reduce `n_samples` to 64 (line 223) |
| **Process hangs** | Already fixed: `multiprocessing=False` |
| **"CUDA error"** | Check `allow_cuda=False` (line 229) |
| **Wrong thread count** | Check `torch.set_num_threads()` (line 44) |
| **Alma not found** | `pip install alma-torch` |

---

## Performance Expectations

**i7-12700K (12 cores, 20 threads):**

| Model | Parameters | Eager | COMPILE_INDUCTOR |
|-------|------------|-------|------------------|
| SimpleCNN | 1M | ~2.0 ms | ~1.3 ms (1.5x) |
| ResNet18 | 11M | ~15 ms | ~8 ms (2x) |
| ResNet50 | 25M | ~40 ms | ~20 ms (2x) |

**Your results may vary** based on:
- CPU model (more cores = better)
- Memory bandwidth (DDR5 > DDR4)
- Thermal throttling
- Background processes

---

## File Locations

| File | Purpose |
|------|---------|
| `cpu_minimal_example.py` | **Run this** - Minimal working example |
| `CPU_SETUP_SUMMARY.md` | Detailed explanation of configuration |
| `QUICKREF.md` | **This file** - Quick reference |
| `README.md` | Full documentation |
| `alma_integration.py` | Advanced (may fail on CPU-only) |

---

## Hardware-Specific Notes

### Intel CPUs (i7-12700K, i9, Xeon)
- Use physical core count for threads
- Install `openvino` for best performance
- Expected speedup: 2-3x with COMPILE_INDUCTOR

### AMD CPUs (Ryzen, EPYC)
- Use physical core count for threads
- COMPILE_INDUCTOR works well
- OpenVINO is Intel-only (skip it)

### ARM CPUs (Graviton, M1/M2 Mac)
- Adjust thread count based on core config
- Use `allow_mps=True` on Mac for Metal GPU
- COMPILE_INDUCTOR support varies

---

## Integration with graphs Package

```python
# Get prediction from graphs
from graphs.analysis.unified_analyzer import UnifiedAnalyzer
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'Intel-i7-12700k', batch_size=1)

# Run Alma validation (modify cpu_minimal_example.py)
# Compare result.latency_ms with Alma results
```

---

## Need More Help?

1. **Read**: `CPU_SETUP_SUMMARY.md` - Detailed configuration guide
2. **Read**: `README.md` - Full documentation
3. **Check**: Alma docs at https://github.com/saifhaq/alma
4. **Ask**: File an issue in the graphs repository

---

**Last Updated**: 2025-11-07
**Tested On**: Intel i7-12700K, PyTorch 2.7.1, Alma 0.3.7
