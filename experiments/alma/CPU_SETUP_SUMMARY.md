# Alma CPU Benchmarking - Setup Summary

## What Was Created

A minimal, working example of Alma benchmarking for CPU-only environments.

**File**: `experiments/alma/cpu_minimal_example.py`

## The Problem

The original `alma_integration.py` failed on CPU-only servers because:
1. High memory usage (2048 samples) → Out of memory
2. Multiprocessing enabled → Process hangs
3. Auto device detection → Tried to use GPU on CPU-only server
4. Incorrect data shape → Conv2D shape errors

## The Solution

A minimal example with explicit CPU configuration:

### 1. CPU Thread Configuration (Lines 25-65)

**Purpose**: Set optimal thread count for your CPU

```python
def configure_cpu_environment():
    # Detect CPU count
    cpu_count = multiprocessing.cpu_count()  # 20 on i7-12700K

    # Use physical cores only (not hyperthreads)
    num_threads = min(12, cpu_count)

    # Configure PyTorch
    torch.set_num_threads(num_threads)

    # Configure underlying math libraries
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
```

**Why This Matters**:
- Prevents thread oversubscription
- Uses physical cores (better performance than hyperthreads)
- Configures all math libraries consistently

**Verification**:
```bash
python3 -c "import torch; print(f'Threads: {torch.get_num_threads()}')"
# Expected: Threads: 12
```

### 2. Alma CPU Configuration (Lines 225-234)

**Purpose**: Force Alma to use CPU only

```python
config = BenchmarkConfig(
    n_samples=128,                       # Small count (avoid OOM)
    batch_size=1,                        # Minimal memory
    device=torch.device('cpu'),          # ← EXPLICIT CPU
    allow_cuda=False,                    # ← CRITICAL: Disable GPU
    allow_mps=False,                     # ← Disable macOS GPU
    multiprocessing=False,               # ← Avoid hangs
    fail_on_error=False,                 # ← Continue on error
    allow_device_override=False          # ← CRITICAL: Prevent auto-override
)
```

**Critical Parameters**:
- `allow_cuda=False` - Prevents Alma from trying to use CUDA
- `allow_device_override=False` - Prevents Alma from overriding our CPU choice
- `multiprocessing=False` - Avoids process hangs on some systems
- `n_samples=128` - Small count to avoid memory issues

**Verification**:
```python
print(f"Device: {config.device}")              # Should be: cpu
print(f"CUDA disabled: {not config.allow_cuda}") # Should be: True
```

### 3. DataLoader Configuration (Lines 244-253)

**Purpose**: Proper data format for Alma

```python
from torch.utils.data import TensorDataset, DataLoader

# Create dataset (n_samples of individual inputs)
dataset_inputs = input_tensor.repeat(n_samples, 1, 1, 1)  # (128, 3, 224, 224)
dataset_labels = torch.zeros(n_samples, dtype=torch.long)

dataset = TensorDataset(dataset_inputs, dataset_labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Use data_loader (not raw tensor)
results = benchmark_model(model, config, conversions, data_loader=data_loader)
```

**Why DataLoader**:
- Alma expects DataLoader format (not raw tensors)
- Prevents shape mismatch errors
- Allows batching control

## How to Use

### Quick Start

```bash
# Run the minimal example
python3 experiments/alma/cpu_minimal_example.py
```

**Expected Output**:
```
CPU threads: 12
Model parameters: 1,008,618
Baseline latency: 2.0 ms (~500 inf/sec)
EAGER: 2.1 ms
✓ Example completed successfully
```

### Testing Different Models

Edit the file and replace `SimpleCNN`:

```python
# Option 1: ResNet18
from torchvision.models import resnet18
model = resnet18(weights=None)
input_tensor = torch.randn(1, 3, 224, 224, device=device)

# Option 2: MobileNetV2
from torchvision.models import mobilenet_v2
model = mobilenet_v2(weights=None)
input_tensor = torch.randn(1, 3, 224, 224, device=device)

# Option 3: Custom model
model = YourCustomModel()
input_tensor = torch.randn(1, *your_input_shape, device=device)
```

### Adding More Backends

Uncomment optional conversions in the code:

```python
cpu_conversions = [
    "EAGER",
    "COMPILE_INDUCTOR",
    "ONNX_CPU",      # ← Uncomment after: pip install onnxruntime
    "OPENVINO",      # ← Uncomment after: pip install openvino
]
```

## Where Platform Configuration Happens

| Configuration | Line Numbers | What It Does |
|--------------|--------------|--------------|
| **CPU Thread Setup** | 25-65 | Sets optimal thread count, configures MKL/OpenBLAS |
| **Environment Validation** | 47-63 | Verifies CPU-only setup, tests tensor creation |
| **Alma Device Config** | 225-234 | Forces CPU device, disables GPU/MPS |
| **DataLoader Setup** | 244-253 | Creates proper data format for Alma |
| **Runtime Validation** | 236-242 | Prints config to verify settings |

## Configuration Checklist

Before running Alma on your CPU server:

- [x] **Set thread count** to physical cores (not hyperthreads)
- [x] **Set environment variables** (OMP_NUM_THREADS, MKL_NUM_THREADS)
- [x] **Force CPU device** (`device=torch.device('cpu')`)
- [x] **Disable GPU detection** (`allow_cuda=False`)
- [x] **Prevent auto-override** (`allow_device_override=False`)
- [x] **Use small sample count** (`n_samples=128`)
- [x] **Use DataLoader** (not raw tensors)
- [x] **Disable multiprocessing** (`multiprocessing=False`)

## Verification Commands

```bash
# 1. Check CPU configuration
python3 -c "import torch; print(f'Threads: {torch.get_num_threads()}, CUDA: {torch.cuda.is_available()}')"

# Expected: Threads: 12, CUDA: False

# 2. Check Alma installation
python3 -c "from alma import benchmark_model; print('Alma OK')"

# Expected: Alma OK

# 3. Run minimal example
python3 experiments/alma/cpu_minimal_example.py

# Expected: Completes in ~10 seconds with ✓ marks
```

## Performance Expectations

### SimpleCNN (1M parameters, batch_size=1)
- **Baseline (eager)**: ~2.0 ms/inference
- **COMPILE_INDUCTOR**: ~1.3 ms/inference (1.5x speedup)
- **OPENVINO** (if Intel CPU): ~1.0 ms/inference (2x speedup)

### ResNet18 (11M parameters, batch_size=1)
- **Baseline (eager)**: ~15-20 ms/inference
- **COMPILE_INDUCTOR**: ~8-12 ms/inference (1.5-2x speedup)
- **OPENVINO** (if Intel CPU): ~5-8 ms/inference (2-3x speedup)

**Note**: Actual performance depends on:
- CPU model (i7-12700K has 12 cores, 20 threads)
- Memory bandwidth (DDR4/DDR5)
- Thermal throttling
- Background processes

## Common Issues Fixed

| Issue | Original Code | Fixed Code |
|-------|---------------|------------|
| **OOM** | `n_samples=2048` | `n_samples=128` |
| **Process hang** | `multiprocessing=True` | `multiprocessing=False` |
| **GPU error** | Auto device detection | `allow_cuda=False` |
| **Shape error** | Raw tensor (`data=`) | DataLoader (`data_loader=`) |
| **Thread contention** | Default (20 threads) | Physical cores (12 threads) |

## Integration with graphs Package

The minimal example can be integrated with graphs analysis:

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer

# Get prediction
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'Intel-i7-12700k', batch_size=1)

# Run Alma validation
# ... (modify cpu_minimal_example.py to accept predicted_latency parameter)

# Compare
error_percent = abs(predicted - actual) / actual * 100
```

## Next Steps

1. **Verify setup**: Run `cpu_minimal_example.py` to ensure it works
2. **Test with your models**: Replace SimpleCNN with your models
3. **Add backends**: Install onnxruntime or openvino
4. **Batch size sweep**: Test different batch sizes (1, 4, 8, 16)
5. **Compare with predictions**: Use graphs.analysis predictions

## Files Created

- `experiments/alma/cpu_minimal_example.py` - Minimal working example (✅ Ready)
- `experiments/alma/README.md` - Updated with CPU-only quick start
- `experiments/alma/CPU_SETUP_SUMMARY.md` - This file

## References

- Alma GitHub: https://github.com/saifhaq/alma
- PyTorch Threading: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
- Intel MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

---

**Status**: ✅ Working on CPU-only i7-12700K server
**Last Updated**: 2025-11-07
**Tested On**: Intel i7-12700K (12 cores, 20 threads), PyTorch 2.7.1
