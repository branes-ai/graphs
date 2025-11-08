# Alma CPU Benchmarking - CLI Usage Guide

## Quick Start

```bash
# Default (SimpleCNN)
python3 experiments/alma/cpu_minimal_example.py

# Specify model
python3 experiments/alma/cpu_minimal_example.py --model resnet50

# Full options
python3 experiments/alma/cpu_minimal_example.py \
    --model vit-b-16 \
    --batch-size 4 \
    --samples 64 \
    --baseline-runs 50
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `simple` | Model to benchmark (see supported models below) |
| `--batch-size` | int | `1` | Batch size for inference |
| `--samples` | int | `128` | Number of samples for Alma benchmark |
| `--baseline-runs` | int | `100` | Number of runs for baseline benchmark |

## Supported Models

### Lightweight (Fast, <10M params)
- `simple` - SimpleCNN (1M params, ~2ms on i7-12700K)
- `mobilenet_v2` - MobileNet V2 (3.5M params)
- `mobilenet_v3_small` - MobileNet V3 Small (2.5M params)
- `mobilenet_v3_large` - MobileNet V3 Large (5.5M params)
- `efficientnet_b0` - EfficientNet B0 (5.3M params)
- `efficientnet_b1` - EfficientNet B1 (7.8M params)

### Medium (10-30M params)
- `resnet18` - ResNet-18 (11.7M params, ~15ms)
- `resnet34` - ResNet-34 (21.8M params)
- `resnet50` - ResNet-50 (25.6M params, ~30ms)
- `convnext_tiny` - ConvNeXt Tiny (28.6M params)
- `convnext_small` - ConvNeXt Small (50.2M params)

### Large (>50M params, slow on CPU)
- `resnet101` - ResNet-101 (44.5M params)
- `resnet152` - ResNet-152 (60.2M params)
- `vit_b_16` - Vision Transformer Base/16 (86.6M params, ~100ms)
- `vit_b_32` - Vision Transformer Base/32 (88.2M params)
- `vit_l_16` - Vision Transformer Large/16 (304M params, very slow)
- `convnext_base` - ConvNeXt Base (88.6M params)

**Note**: Model names accept hyphens or underscores (e.g., `resnet-50` or `resnet_50`)

## Usage Examples

### Basic Model Benchmarking

```bash
# Simple CNN (fastest)
python3 experiments/alma/cpu_minimal_example.py --model simple

# ResNet-18 (recommended for testing)
python3 experiments/alma/cpu_minimal_example.py --model resnet18

# ResNet-50 (production baseline)
python3 experiments/alma/cpu_minimal_example.py --model resnet50
```

### Batch Size Exploration

```bash
# Batch size 1 (lowest latency)
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --batch-size 1

# Batch size 4 (better throughput)
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --batch-size 4

# Batch size 8 (max throughput, may OOM)
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --batch-size 8
```

### Sample Count Tuning

```bash
# Quick test (16 samples, ~5 seconds)
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --samples 16

# Standard (128 samples, ~20 seconds)
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --samples 128

# High accuracy (256 samples, ~40 seconds)
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --samples 256
```

**Recommendation**: Use 32-64 samples for large models to avoid long runtimes.

### Complete Configuration

```bash
# Full custom configuration
python3 experiments/alma/cpu_minimal_example.py \
    --model mobilenet_v2 \
    --batch-size 4 \
    --samples 64 \
    --baseline-runs 50
```

## Performance Expectations (i7-12700K, 12 cores)

### Latency (batch_size=1)

| Model | Parameters | EAGER | COMPILE_OPENVINO | Speedup |
|-------|------------|-------|------------------|---------|
| SimpleCNN | 1.0M | 2.1 ms | 0.5 ms | 4.2x |
| ResNet-18 | 11.7M | 15 ms | 5 ms | 3.0x |
| ResNet-50 | 25.6M | 32 ms | 10 ms | 3.2x |
| MobileNet-V2 | 3.5M | 8 ms | 3 ms | 2.7x |
| ViT-B/16 | 86.6M | 101 ms | 42 ms | 2.4x |

### Throughput (batch_size=8)

| Model | EAGER (inf/s) | COMPILE_OPENVINO (inf/s) | Speedup |
|-------|---------------|--------------------------|---------|
| ResNet-50 | ~40 | ~120 | 3.0x |
| MobileNet-V2 | ~150 | ~400 | 2.7x |

**Note**: Actual performance depends on CPU model, memory bandwidth, and system load.

## Output Sections

The tool outputs 5 main sections:

### 1. CPU Environment Configuration
```
CPU ENVIRONMENT CONFIGURATION
System CPU count: 20
PyTorch threads set to: 12
CUDA available: False
✓ CPU environment configured successfully
```

### 2. Model Setup
```
MODEL SETUP
Model: ResNet-50
Parameters: 25,557,032
Device: cpu
✓ Forward pass successful
```

### 3. Baseline Benchmarking
```
BASELINE BENCHMARKING (PyTorch Eager Mode)
Mean: 32.1 ms
Throughput: 31.1 inferences/sec
```

### 4. Alma Multi-Backend Comparison
```
ALMA BENCHMARKING (CPU-Optimized Conversions)
Testing 4 CPU conversions:
  - EAGER
  - COMPILE_INDUCTOR_DEFAULT
  - ONNX_CPU
  - COMPILE_OPENVINO
```

### 5. Performance Summary
```
PERFORMANCE COMPARISON
Conversion                     Latency (ms)    Throughput (inf/s)
COMPILE_OPENVINO               10.3            96.9               (3.1x vs EAGER)
ONNX_CPU                       12.2            81.7               (2.6x vs EAGER)

RECOMMENDATIONS FOR CPU DEPLOYMENT
✓ Best performing: COMPILE_OPENVINO (10.3 ms)
```

## Troubleshooting

### Out of Memory

**Symptom**: Process killed or OOM error

**Solutions**:
```bash
# Reduce samples
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --samples 32

# Reduce batch size
python3 experiments/alma/cpu_minimal_example.py --model resnet50 --batch-size 1

# Use smaller model
python3 experiments/alma/cpu_minimal_example.py --model resnet18
```

### Model Not Found

**Symptom**: `Unknown model: xyz`

**Solutions**:
- Check model name spelling
- Use underscores or hyphens consistently
- Run `--help` to see supported models
- Ensure torchvision is installed

### Slow Performance

**Symptom**: Benchmark takes too long

**Solutions**:
```bash
# Reduce samples (default: 128)
--samples 32

# Reduce baseline runs (default: 100)
--baseline-runs 50

# Use smaller model
--model mobilenet_v2  # instead of resnet101
```

## Integration with graphs Package

You can use this tool alongside the graphs package CLI tools:

```bash
# Step 1: Get prediction from graphs package
python3 cli/analyze_comprehensive.py --model resnet50 --hardware Intel-i7-12700k

# Step 2: Validate with Alma
python3 experiments/alma/cpu_minimal_example.py --model resnet50

# Compare predicted vs actual latency
```

## Advanced Usage

### Testing Multiple Models

```bash
# Bash loop
for model in resnet18 resnet34 resnet50; do
    echo "Testing $model..."
    python3 experiments/alma/cpu_minimal_example.py --model $model --samples 32
done
```

### Saving Results

```bash
# Redirect output
python3 experiments/alma/cpu_minimal_example.py --model resnet50 > resnet50_results.txt 2>&1

# Extract just the summary
python3 experiments/alma/cpu_minimal_example.py --model resnet50 2>&1 | grep -A 20 "SUMMARY"
```

## See Also

- **QUICKREF.md** - Quick reference card
- **CPU_SETUP_SUMMARY.md** - Detailed configuration guide
- **README.md** - Complete documentation
- **RCA_CONVERSION_NAMES.md** - Alma conversion names reference

---

**Last Updated**: 2025-11-07
**Tested On**: Intel i7-12700K (12 cores, 20 threads)
**Status**: ✅ Production ready
