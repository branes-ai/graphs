# Hardware Calibration Benchmarks

This directory contains microbenchmarks for measuring hardware performance characteristics, including both individual operations and fused kernel patterns.

## Overview

### Operation Benchmarks

**Individual operation benchmarks** measure the performance of single operations:

- **`matmul_bench.py`**: Matrix multiplication (BLAS)
- **`memory_bandwidth_bench.py`**: Memory copy bandwidth

These provide baseline performance metrics (GFLOPS, bandwidth, efficiency).

### Fusion Benchmarks (NEW)

**Fused kernel benchmarks** measure the performance benefit of fusing multiple operations together compared to running them separately:

- **`fused_linear_bench.py`**: Linear layer fusion patterns
- **`fused_conv_bench.py`**: Convolution fusion patterns
- **`fused_attention_bench.py`**: Attention mechanism fusion patterns

These quantify **fusion benefits** (speedup, memory reduction, GFLOPS improvement).

## Fusion Benchmarks

### Linear Fusion Patterns

**File**: `fused_linear_bench.py`

**Patterns**:
- `Linear + Bias`: Fused matmul + bias using `torch.addmm`
- `Linear + Bias + ReLU`: Add ReLU activation
- `Linear + Bias + GELU`: Add GELU activation (transformer FFN)

**Usage**:
```bash
# Run all linear fusion patterns
python fused_linear_bench.py

# Quick mode (fewer sizes/trials)
python fused_linear_bench.py --quick

# GPU mode
python fused_linear_bench.py --device cuda
```

**Programmatic**:
```python
from graphs.hardware.calibration.benchmarks.fused_linear_bench import \
    calibrate_linear_fusion_patterns

results = calibrate_linear_fusion_patterns(quick=True)

for result in results:
    pattern = result['fusion_pattern']
    speedup = result['speedup_factor']
    memory = result['memory_reduction']
    print(f"{pattern}: {speedup:.2f}× speedup, {memory*100:.1f}% mem reduction")
```

### Convolution Fusion Patterns

**File**: `fused_conv_bench.py`

**Patterns**:
- `Conv2d + ReLU`: Simple activation fusion
- `Conv2d + BatchNorm`: BN folding at inference time
- `Conv2d + BatchNorm + ReLU`: Full ResNet block fusion

**Key Technique**: **BatchNorm Folding**
- At inference time, BN can be folded into Conv weights/bias
- Eliminates BN computation entirely
- Standard optimization in production deployments

**Usage**:
```bash
# Run all conv fusion patterns
python fused_conv_bench.py

# Quick mode
python fused_conv_bench.py --quick

# GPU mode
python fused_conv_bench.py --device cuda
```

**Programmatic**:
```python
from graphs.hardware.calibration.benchmarks.fused_conv_bench import \
    calibrate_conv_fusion_patterns

results = calibrate_conv_fusion_patterns(quick=True)

for result in results:
    print(f"{result['fusion_pattern']}: {result['speedup_factor']:.2f}× speedup")
```

### Attention Fusion Patterns

**File**: `fused_attention_bench.py`

**Patterns**:
- `Q @ K.T`: Attention scores (matmul)
- `Q @ K.T + Softmax`: Attention weights
- `Full Attention`: `Softmax(Q @ K.T / sqrt(d)) @ V`

**Special Features**:
- Uses PyTorch 2.0+ `F.scaled_dot_product_attention` (SDPA) when available
- Fallback to manual fusion for older PyTorch versions
- Critical for transformer models (BERT, GPT, ViT)

**Usage**:
```bash
# Run all attention fusion patterns
python fused_attention_bench.py

# Quick mode
python fused_attention_bench.py --quick

# GPU mode
python fused_attention_bench.py --device cuda
```

**Programmatic**:
```python
from graphs.hardware.calibration.benchmarks.fused_attention_bench import \
    calibrate_attention_fusion_patterns

results = calibrate_attention_fusion_patterns(quick=True)

for result in results:
    print(f"{result['fusion_pattern']}: {result['speedup_factor']:.2f}× speedup")
```

## Benchmark Results (i7-12700K CPU)

### Linear Fusion

```
Linear+Bias (512×1024×1024):
  Speedup: 0.99× (no benefit)
  Memory reduction: 33.3%

Linear+Bias+ReLU (512×1024×1024):
  Speedup: 0.98× (no benefit)
  Memory reduction: 50.0%
```

**Finding**: Linear fusion shows minimal benefit on CPU (~1.0×).

### Conv Fusion

```
Conv+BN (B=4, 128→128, 28×28):
  Speedup: 1.03× faster
  Memory reduction: 45.8%

Conv+BN+ReLU (B=4, 128→128, 28×28):
  Speedup: 1.03× faster
  Memory reduction: 62.8%
```

**Finding**: Conv fusion shows moderate benefit (1.03-1.07×) due to BN folding.

### Attention Fusion

```
QK^T (B=4, S=256, D=64):
  Speedup: 1.07× faster
  Memory reduction: 57.1%

FullAttention (B=4, S=256, D=64):
  Speedup: 0.71× slower!
  Memory reduction: 75.0%
```

**Finding**: Simple QK^T fusion helps (1.07×), but full SDPA is slower (0.71×) on CPU.

## Key Insights

### CPU Fusion Benefits Are Modest

| Pattern | CPU Speedup | Expected GPU Speedup |
|---------|-------------|---------------------|
| Linear+Bias | 0.99× | 1.5-2× |
| Conv+BN | 1.03× | 2-3× |
| Conv+BN+ReLU | 1.03× | 2-4× |
| Attention | 0.71-1.07× | 3-5× |

**Why?**
- CPU has deep caches that hide memory latency
- Element-wise ops (ReLU) are already fast on CPU
- PyTorch CPU backend doesn't aggressively fuse

### GPU Fusion Benefits Are Much Stronger (Expected)

GPU fusion is expected to show **2-5× speedup** because:
- Kernel launch overhead reduction
- Memory bandwidth savings
- Aggressive cuDNN/cuBLAS fusion

### Hardware-Aware Fusion Strategy

**For CPU**:
- Fuse: Conv+BN (1.03-1.07× benefit)
- Conditionally fuse: QK^T (1.07× benefit)
- Avoid: Full Attention (0.71× slower!)

**For GPU** (future):
- Fuse everything (2-5× benefits expected)

## Integration with Calibrator

The fusion benchmarks are integrated into the main calibrator:

```bash
# Calibrate with fusion patterns
python -m graphs.hardware.calibration.calibrator \
    --hardware i7-12700K \
    --peak-gflops 1000 \
    --peak-bandwidth 50 \
    --fusion-patterns all \
    --output profiles/i7_12700k_fusion.json
```

This will:
1. Run standard calibration (matmul, memory bandwidth)
2. Run all fusion benchmarks (linear, conv, attention)
3. Store results in `fusion_profiles` in the calibration JSON
4. Print summary with fusion verdicts

## Benchmark Design

### Measurement Strategy

Each fusion benchmark follows this pattern:

1. **Unfused Baseline**: Run operations separately
   ```python
   Y = X @ W.T     # Linear
   Y = Y + bias    # Add bias
   Y = relu(Y)     # ReLU
   ```

2. **Fused Version**: Run as single operation
   ```python
   Y = relu(torch.addmm(bias, X, W.T))  # Fused
   ```

3. **Measure Both**:
   - Latency (ms)
   - GFLOPS (compute throughput)
   - Memory traffic (bytes)

4. **Compute Metrics**:
   - Speedup = unfused_time / fused_time
   - Memory reduction = 1 - (fused_bytes / unfused_bytes)
   - GFLOPS improvement = (fused_gflops - unfused_gflops) / unfused_gflops

### Result Dictionary Format

All fusion benchmarks return a dictionary with this structure:

```python
{
    'fusion_pattern': str,           # "Linear_Bias_ReLU"
    'input_shape': Tuple[int, ...],  # (512, 1024, 1024)

    # Unfused performance
    'unfused_latency_ms': float,
    'unfused_gflops': float,
    'unfused_bytes': int,

    # Fused performance
    'fused_latency_ms': float,
    'fused_gflops': float,
    'fused_bytes': int,

    # Fusion benefits
    'speedup_factor': float,         # unfused / fused
    'memory_reduction': float,       # 0.0 to 1.0

    # Additional info
    'extra_params': Dict[str, Any],  # kernel_size, channels, etc.
    'flops': int,                    # Total FLOPs
}
```

### Implementation Details

#### Linear Fusion

Uses `torch.addmm` for fused matrix multiply + add:
```python
# Unfused
Y = X @ W.T
Y = Y + bias

# Fused
Y = torch.addmm(bias.unsqueeze(0), X, W.T)
```

For activation fusion, wraps with `torch.jit.script`:
```python
@torch.jit.script
def fused_forward(X, W, bias):
    return torch.relu(torch.addmm(bias.unsqueeze(0), X, W.T))
```

#### Conv Fusion

BatchNorm folding technique:
```python
# Fold BN parameters into Conv weights/bias
scale = bn.weight / torch.sqrt(bn.var + bn.eps)
bias = bn.bias - bn.mean * scale

conv_fused.weight = conv.weight * scale.view(-1, 1, 1, 1)
conv_fused.bias = bias
```

This eliminates BN computation entirely at inference time.

#### Attention Fusion

Uses `F.scaled_dot_product_attention` (PyTorch 2.0+):
```python
# Unfused
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(d)
attn_weights = F.softmax(scores, dim=-1)
output = attn_weights @ V

# Fused (SDPA)
output = F.scaled_dot_product_attention(Q, K, V)
```

Note: SDPA has overhead on CPU but is highly optimized for GPU.

## Extending the Benchmarks

### Adding New Fusion Patterns

To add a new fusion pattern:

1. Create benchmark function in appropriate file:
   ```python
   def benchmark_new_pattern(
       # input parameters
       device: str = 'cpu',
       num_warmup: int = 10,
       num_trials: int = 100
   ) -> Dict:
       # 1. Create inputs
       # 2. Implement unfused version
       # 3. Implement fused version
       # 4. Benchmark both
       # 5. Compute metrics
       # 6. Return result dictionary
   ```

2. Add to `calibrate_*_fusion_patterns()`:
   ```python
   def calibrate_linear_fusion_patterns(...):
       results = []

       for M, K, N in sizes:
           result = benchmark_new_pattern(M, K, N, ...)
           results.append(result)

       return results
   ```

3. Update calibrator.py to handle new pattern category (if needed)

### Adding GPU Support

To extend benchmarks to GPU:

1. Update device argument handling
2. Add CUDA synchronization:
   ```python
   if device == 'cuda':
       torch.cuda.synchronize()
   ```
3. Use appropriate GPU-optimized operations (cuDNN, cuBLAS)
4. Test on actual GPU hardware

## Files

```
benchmarks/
├── README.md                      # This file
├── __init__.py                    # Package exports
├── matmul_bench.py               # Matrix multiplication benchmark
├── memory_bandwidth_bench.py     # Memory bandwidth benchmark
├── fused_linear_bench.py         # Linear fusion benchmarks (450 lines)
├── fused_conv_bench.py           # Conv fusion benchmarks (500 lines)
└── fused_attention_bench.py      # Attention fusion benchmarks (450 lines)
```

## Documentation

For more information, see:
- **`../FUSION_CALIBRATION_DESIGN.md`**: Design document
- **`../FUSION_BENCHMARKS_SUMMARY.md`**: Implementation summary and results
- **`../FUSION_INTEGRATION_EXAMPLE.md`**: Usage examples
- **`../FUSION_WORK_COMPLETE.md`**: Complete work summary

## Testing

```bash
# Test all fusion benchmarks
cd benchmarks
python fused_linear_bench.py --quick
python fused_conv_bench.py --quick
python fused_attention_bench.py --quick

# Test calibrator integration
cd ..
python -m graphs.hardware.calibration.calibrator \
    --hardware TestCPU \
    --peak-gflops 1000 \
    --peak-bandwidth 50 \
    --fusion-patterns all \
    --quick \
    --output profiles/test.json
```

## Contributing

When adding new benchmarks:
1. Follow the result dictionary format
2. Measure unfused baseline vs fused version
3. Compute speedup, memory reduction, GFLOPS improvement
4. Add comprehensive docstrings
5. Test on both CPU and GPU (if applicable)
6. Update this README

## References

- **BatchNorm Folding**: https://neondagan.github.io/batch-norm-folding/
- **PyTorch JIT**: https://pytorch.org/docs/stable/jit.html
- **SDPA**: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- **Fusion Benefits**: See `../FUSION_BENCHMARKS_SUMMARY.md`
