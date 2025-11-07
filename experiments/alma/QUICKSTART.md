# Alma Integration Quick Start

## Installation (Fixed)

Alma has several dependencies that aren't automatically installed. Here's the complete setup:

```bash
# 1. Install Alma
pip install alma-torch

# 2. Install required dependencies (not auto-installed by alma-torch)
pip install optimum-quanto onnx onnxruntime

# 3. (Optional) Install specialized backends
# pip install tensorrt  # NVIDIA GPUs only
# pip install openvino  # Intel hardware
```

### Why These Dependencies?

- **`optimum-quanto`**: Required for quantization conversions (INT8, INT4, FP8)
- **`onnx` + `onnxruntime`**: Required for ONNX conversions
- **`tensorrt`**: Optional, for NVIDIA TensorRT conversions (requires CUDA)
- **`openvino`**: Optional, for Intel OpenVINO conversions

## Quick Test

### Tier 1: Inductor Only (No Alma Required)

```bash
# Fast validation with just inductor (works without Alma dependencies)
python experiments/alma/alma_integration.py --model simple --tier 1
```

**Output:**
```
Eager:     2.48 ms
Inductor:  0.57 ms
Speedup:   4.37x
```

### Tier 2: Core Backends (Alma Required)

```bash
# Test with core deployment options
python experiments/alma/alma_integration.py --model simple --tier 2

# Or specify specific conversions
python experiments/alma/alma_integration.py --model simple --tier 2 \
    --conversions EAGER COMPILE_INDUCTOR ONNX_CPU
```

**Note**: Alma benchmarks can take several minutes depending on the number of conversions.

## Fixed Issues

### Issue 1: Alma Import Failure

**Problem**: `ModuleNotFoundError: No module named 'optimum'`

**Solution**: Install missing dependencies
```bash
pip install optimum-quanto onnx onnxruntime
```

### Issue 2: Model Copy Error

**Problem**: `TypeError: ResNet.__init__() missing 2 required positional arguments`

**Solution**: Fixed in `inductor_validation.py` - now uses model directly without copying
```python
# Before (broken):
model_copy = type(model)().eval()  # Fails for ResNet

# After (fixed):
# Use model directly - torch.compile handles versioning
compiled = torch.compile(model, backend="inductor")
```

## Tested Working Commands

```bash
# 1. Tier 1 (inductor only) - Fast, always works
python experiments/alma/alma_integration.py --model simple --tier 1

# 2. Tier 1 with ResNet18
python experiments/alma/alma_integration.py --model resnet18 --tier 1

# 3. Custom conversions (limited set for speed)
python experiments/alma/alma_integration.py --model simple \
    --conversions EAGER COMPILE_INDUCTOR --tier 2

# 4. Save results to JSON
python experiments/alma/alma_integration.py --model simple --tier 1 \
    --output results.json
```

## Performance Notes

### Benchmark Times

| Tier | Conversions | Approximate Time |
|------|-------------|------------------|
| 1 | 2 (Eager + Inductor) | 10-20 seconds |
| 2 | 7-10 (core options) | 5-10 minutes |
| 3 | 90+ (all options) | 30-60 minutes |

**Recommendation**: Start with Tier 1 for quick validation, use Tier 2 for deployment decisions.

### Why Is Alma Slow?

Alma runs **full benchmarks** for each conversion:
- 2048 samples per conversion (configurable)
- Warmup runs
- Multiple iterations for statistics
- Compilation overhead for each backend

This is intentional for accurate measurements but means comprehensive benchmarks take time.

## Troubleshooting

### "Alma not installed" error

**Symptom**: Script falls back to Tier 1 even after `pip install alma-torch`

**Cause**: Missing dependencies (optimum, onnx)

**Solution**:
```bash
pip install optimum-quanto onnx onnxruntime
python -c "from alma import benchmark_model; print('Success!')"
```

### "ModuleNotFoundError: No module named 'tensorrt'"

**Symptom**: Conversion fails when trying TensorRT

**Cause**: TensorRT not installed (requires NVIDIA GPU + CUDA)

**Solution**:
- Skip TensorRT: `--conversions EAGER COMPILE_INDUCTOR ONNX_GPU`
- Or install TensorRT (requires NVIDIA drivers): See NVIDIA docs

### Benchmark hangs or takes too long

**Symptom**: Script runs for many minutes

**Cause**: Alma is running full benchmarks (intentional)

**Solution**:
- Use Tier 1 for quick validation
- Limit conversions: `--conversions EAGER COMPILE_INDUCTOR`
- Or be patient - accurate benchmarking takes time!

### CUDA out of memory

**Symptom**: OOM error during benchmarking

**Solution**:
- Reduce batch size: `--batch-size 1`
- Use smaller model: `--model simple` instead of `--model resnet50`
- Skip GPU-intensive conversions

## Next Steps

1. **Verify installation**:
   ```bash
   python -c "from alma import benchmark_model; print('Alma ready!')"
   ```

2. **Quick validation**:
   ```bash
   python experiments/alma/alma_integration.py --model simple --tier 1
   ```

3. **Core deployment analysis**:
   ```bash
   python experiments/alma/alma_integration.py --model resnet18 --tier 2
   ```

4. **Read full documentation**:
   - `README.md` - Complete usage guide
   - `ALMA_ANALYSIS.md` - Why Alma vs inductor_validation

## Summary

**Installation fixed**: Install `alma-torch optimum-quanto onnx onnxruntime`

**Model copy fixed**: `inductor_validation.py` no longer tries to copy models

**Tier 1 works**: Fast validation without Alma dependencies

**Tier 2 ready**: Full multi-backend validation with Alma

**Start here**: `python experiments/alma/alma_integration.py --model simple --tier 1`
