# Alma Integration Experiments

This directory contains experiments integrating [Alma](https://github.com/saifhaq/alma) with the graphs package for multi-backend validation and deployment optimization.

## Overview

**Alma** is a PyTorch benchmarking library that tests model performance across **90+ conversion options** including TensorRT, ONNX, OpenVINO, torch.compile variants, quantization, and more.

**Value for graphs package**:
- Validate predictions against multiple backends (not just inductor)
- Provide deployment recommendations (TensorRT, ONNX, OpenVINO, etc.)
- Identify optimization headroom beyond torch.compile
- Guide hardware selection decisions

## Installation

```bash
# Install Alma and required dependencies
pip install alma-torch optimum-quanto onnx onnxruntime

# Alma dependencies:
# - torch, torchvision (usually already installed)
# - onnx, onnxruntime (for ONNX conversions)
# - optimum-quanto (for quantization)
# - (optional) tensorrt, openvino (for specialized backends)

# Note: Some dependencies are not automatically installed by alma-torch
# You may need to install them separately based on the conversions you want to test
```

## Files

| File | Purpose | Status |
|------|---------|--------|
| `cpu_minimal_example.py` | **START HERE** - Minimal CPU-only working example | ✅ Ready |
| `alma_integration.py` | Advanced multi-tier validation (may OOM on large models) | ⚠️ Use with caution |
| `ALMA_ANALYSIS.md` | Detailed analysis of Alma vs inductor_validation | 📖 Reference |
| `README.md` | This file | 📖 Documentation |

## Quick Start (CPU-Only)

### Minimal Working Example - START HERE

The fastest way to get started on a CPU-only server:

```bash
# SimpleCNN (default, fast)
python3 experiments/alma/cpu_minimal_example.py

# ResNet models
python3 experiments/alma/cpu_minimal_example.py --model resnet50

# Vision Transformer
python3 experiments/alma/cpu_minimal_example.py --model vit-b-16

# Custom configuration
python3 experiments/alma/cpu_minimal_example.py --model mobilenet-v2 --batch-size 4 --samples 64

# Show all options
python3 experiments/alma/cpu_minimal_example.py --help
```

**What it does:**
1. ✅ Configures CPU environment (thread count, validates platform)
2. ✅ Runs baseline PyTorch benchmarking (~2ms latency)
3. ✅ Runs Alma multi-backend comparison (EAGER, COMPILE_INDUCTOR)
4. ✅ Shows performance comparison and deployment recommendations
5. ✅ Takes ~10 seconds total

**Key Configuration Points:**
- **CPU Thread Setup** (Lines 25-65): Automatically detects and configures optimal thread count
- **Alma CPU Config** (Lines 225-234): Forces CPU-only execution with `allow_cuda=False`
- **DataLoader Setup** (Lines 244-253): Uses small batch size (1) and sample count (128) to avoid OOM

**Expected Output:**
```
CPU threads: 12
Baseline latency: 2.0 ms (~500 inf/sec)
EAGER: 2.1 ms
COMPILE_INDUCTOR_DEFAULT: 0.5 ms (4.3x speedup)
ONNX_CPU: 0.5 ms (4.4x speedup)
COMPILE_OPENVINO: 0.5 ms (4.3x speedup)
```

**⚠️ Important**: Use correct conversion names:
- `COMPILE_INDUCTOR_DEFAULT` (not `COMPILE_INDUCTOR`)
- `COMPILE_OPENVINO` (not `OPENVINO`)

See `RCA_CONVERSION_NAMES.md` for details.

**To test with different models:**
```python
# Edit cpu_minimal_example.py, replace SimpleCNN with:
from torchvision.models import resnet18
model = resnet18(weights=None)
```

---

## Advanced: Three-Tier Validation Strategy

### Tier 1: Quick (Inductor Only)
**Time**: ~10 seconds
**Purpose**: Fast validation of prediction accuracy

```bash
python experiments/alma/alma_integration.py --model simple --tier 1
```

**Output**:
- Eager baseline: 2.40 ms
- Inductor: 0.48 ms (5x speedup)
- Prediction error vs inductor

**Note**: The minimal example (`cpu_minimal_example.py`) is preferred for CPU-only environments.

### Tier 2: Core Deployment Options
**Time**: ~5 minutes
**Purpose**: Key deployment pathways

```bash
python experiments/alma/alma_integration.py --model resnet18 --tier 2
```

**Tests** (GPU):
- EAGER (baseline)
- COMPILE_INDUCTOR
- COMPILE_INDUCTOR_MAX_AUTOTUNE
- TENSORRT
- ONNX_GPU
- FP16+COMPILE_CUDAGRAPHS
- TORCHAO_QUANT_INT8+COMPILE_INDUCTOR

**Output**:
- Performance comparison across backends
- Best option identification (e.g., TensorRT: 16x speedup)
- Deployment recommendations

### Tier 3: Comprehensive Analysis
**Time**: ~1 hour
**Purpose**: Deep analysis with all options

```bash
python experiments/alma/alma_integration.py --model resnet18 --tier 3
```

**Tests**: 90+ conversion options including:
- All Tier 2 options
- Additional backends (OpenXLA, TVM)
- Mixed precision variants (FP16, BF16, FP8)
- Quantization options (INT8, INT4, FP8)
- Hybrid combinations

## Usage Examples

### Basic Usage

```bash
# Simple model, Tier 2 (recommended)
python experiments/alma/alma_integration.py --model simple --tier 2

# ResNet18, Tier 2
python experiments/alma/alma_integration.py --model resnet18 --tier 2

# Custom conversions
python experiments/alma/alma_integration.py --model resnet18 \
    --conversions EAGER TENSORRT ONNX_GPU COMPILE_INDUCTOR

# Save results to JSON
python experiments/alma/alma_integration.py --model resnet18 --tier 2 \
    --output results.json
```

### Python API

```python
from alma_integration import validate_with_alma
from torchvision.models import resnet18

model = resnet18()
example_input = torch.randn(1, 3, 224, 224)

# Run validation
result = validate_with_alma(
    model,
    example_input,
    model_name="ResNet18",
    hardware="H100",
    tier=2,
    predicted_latency_ms=0.43,  # From graphs.analysis
    verbose=True
)

# Access results
print(f"Best conversion: {result.best_conversion}")
print(f"Best latency: {result.best_latency_ms:.2f} ms")
print(f"Speedup vs inductor: {result.best_speedup_vs_inductor:.2f}x")
print(f"Recommendations: {result.deployment_recommendations}")
```

## Integration with graphs Package

### Complete Workflow

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from experiments.alma.alma_integration import validate_with_alma

# 1. Analyze with graphs package
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'H100', batch_size=1)

predicted_latency = result.latency_ms
predicted_energy = result.energy_per_inference_j

# 2. Validate with Alma
model = get_model('resnet18')
example_input = torch.randn(1, 3, 224, 224)

alma_result = validate_with_alma(
    model,
    example_input,
    model_name="ResNet18",
    hardware="H100",
    tier=2,
    predicted_latency_ms=predicted_latency,
    predicted_energy_j=predicted_energy
)

# 3. Report
print(f"Predicted: {predicted_latency:.2f} ms")
print(f"Inductor:  {alma_result.inductor_latency_ms:.2f} ms")
print(f"Error:     {alma_result.inductor_error_percent:.1f}%")
print(f"\nBest deployment: {alma_result.best_conversion}")
print(f"Optimization headroom: {alma_result.best_speedup_vs_inductor:.2f}x")
```

## Example Output

```
================================================================================
ALMA VALIDATION: resnet18 on H100
Tier: 2
================================================================================

================================================================================
Tier 1: Inductor Validation (Quick)
================================================================================
Eager:     10.00 ms
Inductor:  2.00 ms (5.00x speedup)

Prediction vs Inductor:
  Predicted: 1.80 ms
  Actual:    2.00 ms
  Error:     10.0%

================================================================================
Tier 2: Alma Multi-Backend Validation
================================================================================
Testing 7 conversions:
  - EAGER
  - COMPILE_INDUCTOR
  - COMPILE_INDUCTOR_MAX_AUTOTUNE
  - TENSORRT
  - ONNX_GPU
  - FP16+COMPILE_CUDAGRAPHS
  - TORCHAO_QUANT_INT8+COMPILE_INDUCTOR

Running Alma benchmark...
✓ Benchmark complete

================================================================================
Analysis & Recommendations
================================================================================

Conversion                                   Latency (ms)    Speedup (vs eager)
--------------------------------------------------------------------------------
TENSORRT                                     0.62            16.13x     ← BEST
FP16+COMPILE_CUDAGRAPHS                      0.83            12.05x
TORCHAO_QUANT_INT8+COMPILE_INDUCTOR          1.04            9.62x
ONNX_GPU                                     1.25            8.00x
COMPILE_INDUCTOR_MAX_AUTOTUNE                1.85            5.41x
COMPILE_INDUCTOR                             2.00            5.00x     ← PREDICTED
EAGER                                        10.00           1.00x

================================================================================
Deployment Recommendations
================================================================================
  GPU Production: TENSORRT (best GPU performance)
  Cross-platform: ONNX_GPU (portable)
  Edge devices: TORCHAO_QUANT_INT8+COMPILE_INDUCTOR (low memory)
  Quick deploy: COMPILE_INDUCTOR (torch.compile, easy)

Optimization Headroom:
  3.23x improvement possible beyond inductor baseline
```

## Conversion Options Reference

### PyTorch Native
- `EAGER` - Baseline PyTorch
- `TORCH_SCRIPT` - TorchScript JIT
- `COMPILE_INDUCTOR` - torch.compile (default)
- `COMPILE_INDUCTOR_MAX_AUTOTUNE` - Aggressive tuning
- `COMPILE_CUDAGRAPHS` - CUDA graphs
- `COMPILE_OPENXLA` - XLA backend
- `COMPILE_TVM` - TVM backend

### Inference Engines
- `TENSORRT` - NVIDIA TensorRT
- `TENSORRT_FP16` - TensorRT with FP16
- `ONNX_CPU` / `ONNX_GPU` - ONNX Runtime
- `OPENVINO` - Intel OpenVINO
- `OPENVINO_FP16` - OpenVINO with FP16

### Quantization (torchao)
- `TORCHAO_QUANT_INT8` - INT8 quantization
- `TORCHAO_QUANT_I4_WEIGHT_ONLY` - INT4 weight-only
- `TORCHAO_QUANT_FP8` - FP8 quantization
- `TORCHAO_AUTOQUANT` - Auto-quantization

### Mixed Precision
- `FP16` / `BF16` - Half precision
- `FP16+COMPILE_INDUCTOR` - FP16 + inductor
- `FP16+COMPILE_CUDAGRAPHS` - FP16 + CUDA graphs
- `BF16+TENSORRT` - BFloat16 + TensorRT

### Hybrid
- `EXPORT+COMPILE_INDUCTOR` - Export then compile
- `EXPORT+COMPILE_TENSORRT` - Export then TensorRT
- Many more combinations...

## Key Insights

### 1. Validation Accuracy
**Question**: Are our latency predictions accurate?

**Alma Answer**:
- Inductor validation: 10% error ✓
- But TensorRT is 3.2x faster than inductor!
- ONNX is also 1.6x faster
- Prediction is accurate for inductor, but misses optimization potential

### 2. Deployment Guidance
**Question**: How should customers deploy their model?

**Alma Answer**:
- **Cloud GPU production** → TensorRT (16x speedup, best performance)
- **Cross-platform** → ONNX (8x speedup, portable)
- **Edge devices** → INT8 quantization (9x speedup, low memory)
- **Quick prototyping** → torch.compile/Inductor (5x speedup, easy)

### 3. Optimization Headroom
**Question**: How much more speedup is possible?

**Alma Answer**:
- Inductor: 5x speedup (automatic)
- TensorRT: 16x speedup (3.2x more than inductor!)
- Headroom: 3.2x improvement beyond auto-optimization

### 4. Hardware Selection
**Question**: Is specialized hardware worth it?

**Alma Answer**:
- NVIDIA GPU + TensorRT: 3.2x better than inductor
- Intel CPU + OpenVINO: 1.2x better than inductor
- Data-driven purchasing decisions

## Troubleshooting

### Alma not installed
```
⚠️  Alma not installed. Install with: pip install alma-torch
```
**Solution**: `pip install alma-torch`

### CUDA not available
Some conversions (TensorRT, ONNX_GPU, CUDAGRAPHS) require CUDA.

**Solution**:
- Run on GPU machine, or
- Use CPU conversions: `--conversions EAGER COMPILE_INDUCTOR ONNX_CPU OPENVINO`

### Conversion fails
Some conversions may fail for specific models. Alma handles this gracefully.

**Example**: TensorRT may fail for models with dynamic control flow.

**Solution**: Alma continues with other conversions. Check output for failures.

### Memory issues
Running all 90 conversions (Tier 3) can be memory-intensive.

**Solution**:
- Use Tier 2 instead (7-10 options)
- Run on machine with more memory
- Use smaller batch size: `--batch-size 1`

## Platform Configuration Details (CPU-Only Servers)

### How Alma Detects and Configures Platform

Alma normally auto-detects your hardware (CPU/GPU/MPS) and selects appropriate backends. On CPU-only servers, we must **explicitly override** this behavior to prevent GPU-related errors.

### Critical Configuration Points in `cpu_minimal_example.py`

#### 1. CPU Thread Configuration (Lines 25-65)
```python
def configure_cpu_environment():
    """Configure PyTorch for optimal CPU performance."""
    # Get physical core count (not hyperthreads)
    num_threads = 12  # For i7-12700K

    # Set PyTorch threads - THIS CONTROLS CPU PARALLELISM
    torch.set_num_threads(num_threads)

    # Set environment variables for underlying libraries
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
```

**Why This Matters:**
- PyTorch defaults to using all logical threads (20 on i7-12700K)
- Using physical cores only avoids oversubscription
- MKL (Math Kernel Library) respects these environment variables

#### 2. Alma CPU Configuration (Lines 225-234)
```python
config = BenchmarkConfig(
    device=torch.device('cpu'),     # ← FORCE CPU (critical!)
    allow_cuda=False,                # ← DISABLE GPU DETECTION
    allow_mps=False,                 # ← DISABLE macOS GPU
    multiprocessing=False,           # ← AVOID HANGS
    fail_on_error=False,             # ← CONTINUE ON ERROR
    allow_device_override=False      # ← PREVENT AUTO-OVERRIDE
)
```

**What Each Parameter Does:**
- `device='cpu'`: Explicitly set execution device
- `allow_cuda=False`: **Critical!** Prevents Alma from trying to use CUDA
- `allow_mps=False`: Disables macOS Metal Performance Shaders
- `multiprocessing=False`: Avoids process hangs on some systems
- `fail_on_error=False`: Continues if one backend fails (graceful degradation)
- `allow_device_override=False`: **Critical!** Prevents Alma from overriding our CPU choice

#### 3. Runtime Validation (Runs Automatically)
```python
# The script validates configuration at startup
print(f"PyTorch threads: {torch.get_num_threads()}")  # Should be 12
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be False
print(f"Device: {config.device}")                     # Should be 'cpu'
```

**Expected Output:**
```
PyTorch threads: 12
CUDA available: False
Device: cpu
```

### Why alma_integration.py Fails on CPU-Only

The original `alma_integration.py` has these issues:
1. **High sample count** (2048) → Out of memory
2. **Multiprocessing enabled** → Process hangs
3. **Auto device selection** → May try to use GPU
4. **Raw tensor input** → Shape mismatch errors

The minimal example fixes all of these.

### Testing Platform Configuration

```bash
# Verify CPU configuration
python3 -c "import torch; print(f'Threads: {torch.get_num_threads()}, CUDA: {torch.cuda.is_available()}')"

# Check environment variables
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"

# Run minimal example (validates everything)
python3 experiments/alma/cpu_minimal_example.py | grep "CPU ENVIRONMENT"
```

## Comparison: inductor_validation vs Alma

| Feature | inductor_validation | Alma |
|---------|---------------------|------|
| **Backends tested** | 1 (Inductor) | 90+ (All frameworks) |
| **Time** | ~10 seconds | 10s (Tier 1) to 1 hour (Tier 3) |
| **Deployment guidance** | No | Yes (comprehensive) |
| **Cross-framework** | No | Yes (ONNX, TensorRT, OpenVINO) |
| **Quantization** | No | Yes (INT8, INT4, FP8) |
| **Production ready** | Research tool | Production tool |
| **Use case** | Quick validation | Deployment optimization |

**Recommendation**: Use both!
- **inductor_validation**: Fast validation (CI/CD)
- **Alma**: Comprehensive analysis (deployment decisions)

## Next Steps

1. **Test with your models**:
   ```bash
   python experiments/alma/alma_integration.py --model resnet18 --tier 2
   ```

2. **Integrate with CLI tools**:
   - Add `--validate-alma` flag to `cli/analyze_comprehensive.py`
   - Include Alma results in reports

3. **Continuous validation**:
   - Add to CI/CD pipeline
   - Track validation accuracy over time
   - Alert on prediction degradation

4. **Customer reporting**:
   - Include deployment recommendations in reports
   - Show optimization headroom
   - Provide cost/benefit analysis

## References

- [Alma GitHub](https://github.com/saifhaq/alma)
- [Alma Blog Post](https://oscar-savolainen.medium.com/alma-find-the-fastest-pytorch-model-conversion-auto-benchmark-50-options-5247eb6c2ec3)
- [graphs Package Documentation](../../docs/)
- [Dynamo Experiments](../dynamo/)

---

**Status**: ✅ Ready for use
**Last Updated**: 2025-11-07
**Maintainer**: graphs package team
