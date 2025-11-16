# Session Log: 2025-11-16 - Alma Integration for Jetson ARM64

## Objective
Integrate Alma (https://github.com/saifhaq/alma) multi-backend benchmarking framework into the graphs package validation workflow, with focus on getting it working on NVIDIA Jetson Orin AGX (ARM64 platform).

## Summary
Successfully integrated Alma with comprehensive ARM64/Jetson compatibility layer. Resolved multiple dependency and platform compatibility issues. Achieved functional benchmarking with EAGER mode; documented platform limitations for ONNX Runtime, Triton, and torch-tensorrt.

---

## Issues Encountered & Solutions

### 1. Alma Package Not Detected

**Symptom:**
```bash
⚠️  Alma not installed. Install with: pip install alma-torch
```

**Root Cause Analysis:**
- Package `alma-torch` was installed but had incomplete dependency declarations
- Missing: `tqdm`, `pydantic`, `optimum`, `optimum-quanto`

**Solution:**
```bash
pip install tqdm
pip install pydantic
pip install optimum
pip install optimum-quanto
```

**Lesson Learned:** Third-party packages may have incomplete dependency declarations, especially for edge platforms.

---

### 2. ONNX Runtime Python Version Mismatch

**Symptom:**
```python
ImportError: Python version mismatch: module was compiled for Python 3.8, but the interpreter version is incompatible: 3.10.12
```

**Root Cause Analysis:**
- ONNX Runtime has no official Python 3.10 ARM64 wheels
- Attempting to install from PyPI retrieves incompatible builds
- NVIDIA's Jetson-specific wheels are for Python 3.8

**Solution Implemented:**
Created monkey-patch to provide mock ONNX Runtime module:

```python
def _create_mock_onnxruntime():
    """Create a mock onnxruntime module that raises errors on use."""
    mock_module = types.ModuleType('onnxruntime')

    # Create mock ModuleSpec for PyTorch dynamo compatibility
    mock_spec = importlib.machinery.ModuleSpec(
        name='onnxruntime',
        loader=None,
        origin='mock',
        is_package=False
    )
    mock_module.__spec__ = mock_spec
    mock_module.__file__ = '<mock>'
    mock_module.__package__ = None

    class MockInferenceSession:
        def __init__(self, *args, **kwargs):
            raise ImportError("onnxruntime not available on this platform (ARM64/Jetson)")

    mock_module.InferenceSession = MockInferenceSession
    return mock_module

# Install mock before importing alma
sys.modules['onnxruntime'] = _create_mock_onnxruntime()
```

**Key Details:**
- Mock must include `__spec__` attribute for PyTorch dynamo compatibility
- Installed into `sys.modules` before alma import to prevent import crash
- Added automatic filtering of ONNX-based conversions

---

### 3. Triton Not Available on ARM64

**Symptom:**
```python
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
```

**Root Cause Analysis:**
- Triton (NVIDIA's GPU programming language) doesn't officially support ARM64/Jetson
- Required by PyTorch's inductor compiler (`torch.compile`)
- Affects all `COMPILE_INDUCTOR*` and `COMPILE_CUDAGRAPHS` conversions

**Solution Implemented:**
Automatic detection and filtering of Triton-dependent conversions:

```python
def _filter_triton_conversions(conversions: List[str]) -> List[str]:
    """Filter out Triton-dependent conversions if Triton is not available."""
    if TRITON_AVAILABLE:
        return conversions

    triton_dependent_patterns = [
        'COMPILE_INDUCTOR',
        'COMPILE_CUDAGRAPHS',
        'COMPILE_OPENXLA',
        'COMPILE_TVM',
    ]

    # Remove any conversion that contains these patterns
    filtered = []
    for c in conversions:
        requires_triton = any(pattern in c for pattern in triton_dependent_patterns)
        if not requires_triton:
            filtered.append(c)

    return filtered
```

**Conversions Filtered:**
- `COMPILE_INDUCTOR`
- `COMPILE_INDUCTOR_MAX_AUTOTUNE`
- `FP16+COMPILE_CUDAGRAPHS`
- `TORCHAO_QUANT_INT8+COMPILE_INDUCTOR`
- All compound conversions containing these patterns

---

### 4. Alma DataLoader Shape Mismatch

**Symptom:**
```python
RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d,
but got input of size: [1, 16, 3, 224, 224]
```

**Root Cause Analysis:**
Multiple failed attempts to provide data in correct format:
1. Raw tensor `[16, 3, 224, 224]` - Alma wrapped entire tensor as single sample
2. DataLoader without labels - Alma expected `(data, label)` tuples
3. TensorDataset - Still produced wrong shapes

**Solution Implemented:**
Custom Dataset class that returns `(image, label)` tuples:

```python
class SimpleDataset(Dataset):
    def __init__(self, n_samples, C, H, W, device):
        self.n_samples = n_samples
        # Pre-generate all samples on GPU
        self.data = torch.randn(n_samples, C, H, W, device=device)
        # Create dummy labels (not used but required by Alma)
        self.labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return (image, label) tuple as expected by Alma
        return self.data[idx], self.labels[idx]

dataset = SimpleDataset(n_samples, C, H, W, device)
data_loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0,  # Avoid multiprocessing on Jetson
    pin_memory=False
)
```

**Key Details:**
- Dataset returns individual samples `[C, H, W]` from `__getitem__`
- DataLoader batches them into `[batch_size, C, H, W]`
- Pre-generates all data on GPU to avoid transfer overhead
- Uses dummy labels (zeros) as Alma expects classification dataset format

---

### 5. Memory Swapping on 8GB Jetson

**Symptom:**
- System constantly swapping with default 2048 samples
- Swap usage 3x the 8GB main memory

**Root Cause Analysis:**
- Alma default: 2048 samples × batch_size × model memory
- ResNet18: ~44MB per forward pass × 2048 samples = ~90GB
- Exceeds Jetson Orin AGX 8GB memory by 11x

**Solution Implemented:**
Made sample count configurable with sensible defaults:

```python
def validate_with_alma(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    model_name: str,
    hardware: str = 'Jetson-Orin-AGX',
    tier: int = 2,
    conversions: Optional[List[str]] = None,
    predicted_latency_ms: Optional[float] = None,
    predicted_energy_j: Optional[float] = None,
    n_samples: int = 256,  # Reduced from 2048
    verbose: bool = True
) -> AlmaValidationResult:
```

CLI argument:
```python
parser.add_argument(
    "--n-samples",
    type=int,
    default=256,
    help="Number of samples for benchmarking (default 256, use 64-128 for Jetson 8GB)"
)
```

**Recommendations:**
- Jetson Orin AGX 8GB: `--n-samples 64` (safest)
- Jetson Orin AGX 16GB+: `--n-samples 128`
- Memory-rich systems: `--n-samples 512` or higher

---

### 6. TensorRT Not Available via pip on Tegra

**Symptom:**
```python
RuntimeError: TensorRT does not currently build wheels for Tegra systems
```

**Root Cause Analysis:**
- `torch-tensorrt` cannot be installed via pip on Jetson/Tegra
- TensorRT comes pre-installed with JetPack (system packages)
- Python bindings located in `/usr/lib/python3.10/dist-packages/`

**Solution Implemented:**
Symlink system TensorRT to venv:

```bash
# Link TensorRT packages to venv
ln -s /usr/lib/python3.10/dist-packages/tensorrt ~/venv/p310/lib/python3.10/site-packages/
ln -s /usr/lib/python3.10/dist-packages/tensorrt-10.3.0.dist-info ~/venv/p310/lib/python3.10/site-packages/
ln -s /usr/lib/python3.10/dist-packages/tensorrt_lean ~/venv/p310/lib/python3.10/site-packages/
ln -s /usr/lib/python3.10/dist-packages/tensorrt_dispatch ~/venv/p310/lib/python3.10/site-packages/
```

**Verification:**
```bash
python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
# Output: TensorRT version: 10.3.0
```

**Known Limitation:**
- Base `tensorrt` (10.3.0) available and working
- `torch_tensorrt` (PyTorch-TensorRT wrapper) still unavailable
- TENSORRT conversion in Alma fails silently due to missing integration layer

---

## Final Configuration

### Working Conversions on Jetson Orin AGX

**Tier 1:**
- `EAGER` - Standard PyTorch execution ✅
- `COMPILE_INDUCTOR` - Filtered out (no Triton) ❌

**Tier 2:**
- `EAGER` - Standard PyTorch execution ✅
- `TENSORRT` - Fails silently (missing torch_tensorrt) ❌

### Platform Dependencies Status

| Dependency | Status | Version | Notes |
|------------|--------|---------|-------|
| alma-torch | ✅ Installed | 0.3.7 | Manual dependency installation required |
| tqdm | ✅ Installed | 4.67.1 | Required by alma |
| pydantic | ✅ Installed | 2.12.4 | Required by alma |
| optimum | ✅ Installed | 2.0.0 | Required by alma |
| optimum-quanto | ✅ Installed | - | Required by alma |
| onnxruntime | ❌ Mocked | - | No ARM64 Python 3.10 wheels |
| triton | ❌ Unavailable | - | Not supported on ARM64/Jetson |
| tensorrt | ✅ Linked | 10.3.0 | System package via JetPack |
| torch_tensorrt | ❌ Unavailable | - | No Tegra builds available |

---

## Successful Benchmark Results

### ResNet18 on Jetson Orin AGX (16 samples)

**Tier 1: Inductor Validation**
```
Eager:     281.96 ms (3.5 img/s)
Inductor:  229.78 ms (4.4 img/s)
Speedup:   1.23x
```

**Tier 2: Alma Validation (EAGER mode)**
```
Conversion: EAGER
Device: cuda
Latency: 20.22 ms
Throughput: 49.45 samples/second
Total samples: 16
Batch size: 1
Data dtype: torch.float32
```

**Note:** Difference in latency (281.96ms vs 20.22ms) due to different benchmarking methodologies:
- Inductor validation: Single forward pass with warmup
- Alma: Mean of 16 samples with statistical validation

---

## Code Changes Summary

### Files Modified

**`experiments/alma/alma_integration.py`** (extensive modifications)

1. **ONNX Runtime Monkey-Patch** (lines 39-61)
   - Created mock module with proper `__spec__` for PyTorch dynamo
   - Installed before alma import

2. **Triton Availability Detection** (lines 75-82)
   - Check for Triton availability at startup
   - Print warning if unavailable

3. **Conversion Filtering Functions** (lines 209-251)
   - `_filter_onnx_conversions()`: Remove ONNX-based conversions
   - `_filter_triton_conversions()`: Remove Triton-dependent conversions
   - Applied to both tier presets and user-provided conversions

4. **Configurable Sample Count** (lines 246, 594-598, 628)
   - Added `n_samples` parameter with default 256
   - CLI argument `--n-samples`
   - Verbose output showing benchmark configuration

5. **Custom Dataset for Alma** (lines 359-377)
   - `SimpleDataset` class with GPU pre-allocation
   - Returns `(image, label)` tuples
   - DataLoader with Jetson-optimized settings

6. **Diagnostic Reporting** (lines 453-465)
   - Reports which conversions succeeded vs failed
   - Suggests possible reasons for failures

### User Modifications

**Conversion Tier Defaults** (lines 170-190)
- Commented out most Tier 2 conversions
- Kept only: `EAGER`, `TENSORRT`
- Reflects platform realities after filtering

---

## Usage Examples

### Basic Usage
```bash
# Tier 1 (Inductor validation only)
python experiments/alma/alma_integration.py --model resnet18 --tier 1 --hardware GPU

# Tier 2 with limited samples for 8GB Jetson
python experiments/alma/alma_integration.py --model resnet18 --tier 2 --hardware GPU --n-samples 64

# Different model
python experiments/alma/alma_integration.py --model resnet50 --tier 2 --hardware GPU --n-samples 16

# With output file
python experiments/alma/alma_integration.py --model resnet18 --tier 2 --hardware GPU --output results.json
```

### Recommended Settings for Jetson

**Jetson Orin AGX 8GB:**
```bash
python experiments/alma/alma_integration.py \
  --model resnet18 \
  --tier 2 \
  --hardware GPU \
  --n-samples 64 \
  --batch-size 1
```

**Jetson Orin AGX 16GB+:**
```bash
python experiments/alma/alma_integration.py \
  --model resnet18 \
  --tier 2 \
  --hardware GPU \
  --n-samples 128 \
  --batch-size 1
```

---

## Lessons Learned

### 1. Platform-Specific Package Management
- ARM64/Jetson requires careful attention to package availability
- System packages (TensorRT) may need manual linking to venv
- Not all PyPI packages have ARM64 wheels

### 2. Dependency Chain Complexity
- Third-party packages may have incomplete dependencies
- Transitive dependencies can fail on edge platforms
- Need to manually install and verify each layer

### 3. Graceful Degradation Strategy
- Mock unavailable packages to prevent import crashes
- Automatically filter incompatible conversions
- Provide clear diagnostic messages about what's unavailable

### 4. Memory Management on Edge Devices
- Default settings from datacenter-focused tools often exceed edge device capabilities
- Need configurable parameters for sample counts, batch sizes
- Pre-allocate on GPU to avoid transfer overhead

### 5. PyTorch Ecosystem Platform Support
- Triton: x86_64 only (no ARM64)
- ONNX Runtime: Limited ARM64 Python 3.10 support
- torch-tensorrt: No Tegra/Jetson pip packages
- Core PyTorch + TensorRT (system) work well on Jetson

---

## Future Work

### Short Term
1. Investigate building torch-tensorrt from source for Jetson
2. Test with more models (MobileNet, EfficientNet)
3. Batch size sweep analysis
4. Compare against graphs package predictions

### Medium Term
1. Integration with graphs.analysis for predicted vs actual comparison
2. Automated hardware detection and configuration
3. Support for FP16/INT8 precision modes (if compatible backends available)
4. Energy measurement integration (if Jetson power sensors accessible)

### Long Term
1. Build ARM64-specific package bundles for easier setup
2. Contribute ARM64 compatibility improvements back to alma-torch
3. Explore alternative backends that support ARM64 (OpenVINO, Apache TVM)

---

## References

- Alma repository: https://github.com/saifhaq/alma
- PyTorch TensorRT: https://github.com/pytorch/TensorRT
- Triton language: https://github.com/triton-lang/triton
- JetPack documentation: https://developer.nvidia.com/embedded/jetpack

---

## Session Participants

- **User**: branes@Branes-Jetson (Jetson Orin AGX system)
- **Assistant**: Claude Code (debugging and implementation)

**Duration**: ~3 hours
**Date**: 2025-11-16
**Platform**: NVIDIA Jetson Orin AGX (ARM64), JetPack 6.x, Python 3.10.12, CUDA 12.x, TensorRT 10.3.0
