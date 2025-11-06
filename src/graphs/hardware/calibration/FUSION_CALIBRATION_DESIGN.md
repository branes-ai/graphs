# Fused Kernel Calibration - Design Document

## Overview

Extend the hardware calibration framework to benchmark **fused kernel patterns** that emerge from graph partitioning, measuring the actual speedup from fusion vs running operators separately.

## Motivation

### Why Measure Fusion?

1. **Validate Partitioning Decisions**
   - Graph partitioner fuses operations to reduce memory traffic
   - Need empirical data: does fusion actually improve performance?
   - How much speedup? 10%? 50%? 100%?

2. **Quantify Fusion Benefits**
   - Current assumption: fusion is always beneficial
   - Reality: fusion has costs (kernel complexity, register pressure, cache effects)
   - Measure actual speedup for different fusion patterns

3. **Improve Cost Models**
   - Current cost model assumes theoretical memory savings
   - Need real measurements: unfused latency vs fused latency
   - Feed back into partitioner's cost function

4. **Hardware-Specific Fusion Strategies**
   - Some fusions work better on CPUs (cache-friendly)
   - Others work better on GPUs (reduce kernel launches)
   - Calibration reveals which patterns benefit which hardware

### Example: Linear + Bias + ReLU

**Unfused** (3 separate kernels):
```python
# Kernel 1: Linear (matmul)
Y = X @ W.T         # Write Y to memory (M×N×4 bytes)

# Kernel 2: Add bias
Y = Y + bias        # Read Y, write Y (2×M×N×4 bytes)

# Kernel 3: ReLU
Y = max(Y, 0)       # Read Y, write Y (2×M×N×4 bytes)

# Total memory traffic: 5×M×N×4 bytes
# Total kernel launches: 3
```

**Fused** (1 kernel):
```python
# Single fused kernel
Y = max(X @ W.T + bias, 0)  # Write Y once (M×N×4 bytes)

# Total memory traffic: M×N×4 bytes (5× reduction!)
# Total kernel launches: 1
```

**Expected Speedup**: 2-3× for memory-bound operations

But does it actually happen? Let's measure!

## Design

### 1. Fusion Patterns to Benchmark

Based on common graph partitioner outputs:

#### **Linear Fusion Patterns**
1. `Linear + Bias`
2. `Linear + Bias + ReLU`
3. `Linear + Bias + ReLU + Dropout`
4. `Linear + Bias + GELU` (transformer FFN)
5. `Linear + Bias + ReLU + Linear + Bias` (2-layer MLP)

#### **Convolution Fusion Patterns**
1. `Conv2d + Bias`
2. `Conv2d + BatchNorm`
3. `Conv2d + BatchNorm + ReLU` (ResNet block)
4. `Conv2d + ReLU` (simple fusion)
5. `Conv2d + BatchNorm + ReLU + MaxPool` (VGG-style)

#### **Attention Fusion Patterns**
1. `Q @ K.T + Softmax` (attention scores)
2. `Softmax @ V` (attention output)
3. `Q @ K.T + Softmax + @ V` (full attention head)
4. `MultiHead(Q, K, V) + Linear` (transformer block)

#### **Element-wise Fusion Patterns**
1. `Add + ReLU` (residual connection)
2. `Add + LayerNorm` (transformer residual)
3. `Mul + Add + ReLU` (FusedMulAdd)
4. `Sigmoid + Mul` (SwiGLU activation)

### 2. Calibration Schema Extension

Add fusion-specific fields to `OperationCalibration`:

```python
@dataclass
class FusionCalibration:
    """Calibration for a fused kernel pattern"""

    # Fusion pattern identification
    fusion_pattern: str  # "Linear_Bias_ReLU"
    operators: List[str]  # ["linear", "bias", "relu"]
    num_operators: int    # 3

    # Unfused performance (baseline)
    unfused_latency_ms: float
    unfused_memory_bytes: int
    unfused_gflops: float

    # Fused performance (optimized)
    fused_latency_ms: float
    fused_memory_bytes: int
    fused_gflops: float

    # Fusion benefits
    speedup_factor: float  # fused_latency / unfused_latency
    memory_reduction: float  # 1 - (fused_bytes / unfused_bytes)
    efficiency_improvement: float  # (fused_eff - unfused_eff) / unfused_eff

    # Operation parameters
    input_shape: Tuple[int, ...]
    extra_params: Dict[str, Any]  # kernel_size, channels, etc.

    # Hardware characteristics
    bottleneck: BottleneckType  # Before and after fusion
    arithmetic_intensity_unfused: float
    arithmetic_intensity_fused: float
```

### 3. Benchmark Implementation Strategy

#### **PyTorch-Based Benchmarks**

Use PyTorch's built-in fusion capabilities:

1. **Unfused Baseline**: Run operators sequentially
2. **Fused Version**: Use `torch.jit.script` or manual fusion
3. **Measure Both**: Compare latency and memory

Example for Linear+Bias+ReLU:

```python
import torch
import time

def benchmark_linear_bias_relu(M, K, N, device='cpu', num_trials=100):
    """
    Benchmark Linear + Bias + ReLU fusion.

    Args:
        M: Batch size
        K: Input features
        N: Output features
    """
    # Create inputs
    X = torch.randn(M, K, device=device)
    W = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)

    # ========================================================================
    # UNFUSED VERSION (3 separate operations)
    # ========================================================================
    def unfused_forward(X, W, bias):
        Y = X @ W.T           # Linear
        Y = Y + bias          # Add bias
        Y = torch.relu(Y)     # ReLU
        return Y

    # Warmup
    for _ in range(10):
        _ = unfused_forward(X, W, bias)

    # Benchmark unfused
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_unfused = unfused_forward(X, W, bias)
    torch.cuda.synchronize() if device == 'cuda' else None
    unfused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # FUSED VERSION (torch.nn.functional or manual)
    # ========================================================================
    def fused_forward(X, W, bias):
        # PyTorch may fuse this automatically, or use explicit fusion:
        Y = torch.addmm(bias.unsqueeze(0), X, W.T)  # Fused Linear+Bias
        Y = torch.relu(Y)                            # ReLU
        return Y

    # Better: use torch.jit.script for guaranteed fusion
    @torch.jit.script
    def fused_forward_jit(X, W, bias):
        return torch.relu(torch.addmm(bias.unsqueeze(0), X, W.T))

    # Warmup
    for _ in range(10):
        _ = fused_forward_jit(X, W, bias)

    # Benchmark fused
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    for _ in range(num_trials):
        Y_fused = fused_forward_jit(X, W, bias)
    torch.cuda.synchronize() if device == 'cuda' else None
    fused_time = (time.perf_counter() - start) / num_trials

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    # FLOPs: matmul (2*M*K*N) + bias (M*N) + relu (0)
    flops = 2 * M * K * N + M * N

    # Memory (unfused): write matmul output, read+write bias, read+write relu
    unfused_bytes = M * N * 4 * 5  # 5 accesses

    # Memory (fused): write final output only
    fused_bytes = M * N * 4  # 1 write

    # GFLOPS
    unfused_gflops = flops / unfused_time / 1e9
    fused_gflops = flops / fused_time / 1e9

    # Speedup
    speedup = unfused_time / fused_time
    memory_reduction = 1 - (fused_bytes / unfused_bytes)

    return {
        'fusion_pattern': 'Linear_Bias_ReLU',
        'input_shape': (M, K, N),
        'unfused_latency_ms': unfused_time * 1000,
        'fused_latency_ms': fused_time * 1000,
        'unfused_gflops': unfused_gflops,
        'fused_gflops': fused_gflops,
        'speedup_factor': speedup,
        'memory_reduction': memory_reduction,
        'flops': flops,
        'unfused_bytes': unfused_bytes,
        'fused_bytes': fused_bytes,
    }
```

### 4. Integration with Calibrator

Add fusion calibration to the main calibrator:

```python
# calibrator.py

def calibrate_hardware(
    hardware_name: str,
    theoretical_peak_gflops: float,
    theoretical_bandwidth_gbps: float,
    output_path: Path,
    operations: Optional[List[str]] = None,
    fusion_patterns: Optional[List[str]] = None,  # NEW
    quick: bool = False
) -> HardwareCalibration:
    """
    Run hardware calibration including fusion patterns.

    Args:
        fusion_patterns: List of fusion patterns to benchmark
            e.g., ['linear_bias_relu', 'conv_bn_relu', 'attention']
    """

    # ... existing calibration ...

    # NEW: Fusion calibration
    if fusion_patterns and 'all' in fusion_patterns:
        fusion_patterns = [
            'linear_bias_relu',
            'conv_bn_relu',
            'attention_qkv',
            'add_relu',
        ]

    if fusion_patterns:
        print("\n3. Fused Kernel Patterns")
        print("-" * 80)

        for pattern in fusion_patterns:
            if pattern == 'linear_bias_relu':
                from .benchmarks.fused_linear_bench import calibrate_linear_bias_relu
                fusion_cals = calibrate_linear_bias_relu(
                    sizes=[(512, 512, 512), (1024, 1024, 1024)],
                    theoretical_peak_gflops=theoretical_peak_gflops
                )
                for cal in fusion_cals:
                    calibration.add_fusion_pattern(cal)

            elif pattern == 'conv_bn_relu':
                from .benchmarks.fused_conv_bench import calibrate_conv_bn_relu
                fusion_cals = calibrate_conv_bn_relu(...)
                for cal in fusion_cals:
                    calibration.add_fusion_pattern(cal)
```

### 5. Expected Results

Example output for i7-12700K:

```
================================================================================
Fusion Pattern Calibration: Intel-i7-12700K
================================================================================

Linear + Bias + ReLU (512×512):
  Unfused: 2.34 ms (219.2 GFLOPS)
  Fused:   0.89 ms (576.4 GFLOPS)
  Speedup: 2.63× faster
  Memory reduction: 80.0%

Linear + Bias + ReLU (1024×1024):
  Unfused: 8.12 ms (264.5 GFLOPS)
  Fused:   2.98 ms (720.1 GFLOPS)
  Speedup: 2.72× faster
  Memory reduction: 80.0%

Conv2d + BatchNorm + ReLU (256×256, 64 channels):
  Unfused: 12.4 ms (145.2 GFLOPS)
  Fused:   4.8 ms (375.0 GFLOPS)
  Speedup: 2.58× faster
  Memory reduction: 75.0%

Attention Head (512 seq len, 64 dim):
  Unfused: 5.67 ms (189.3 GFLOPS)
  Fused:   2.12 ms (506.1 GFLOPS)
  Speedup: 2.67× faster
  Memory reduction: 66.7%
```

### 6. Benefits for Graph Partitioner

The calibration data feeds back into partitioning decisions:

```python
# In graph partitioner cost model
class FusionCostModel:
    def __init__(self, calibration: HardwareCalibration):
        self.calibration = calibration

    def should_fuse(self, op1: str, op2: str) -> bool:
        """Decide whether to fuse based on calibrated speedup"""
        pattern = f"{op1}_{op2}"

        fusion_cal = self.calibration.get_fusion_pattern(pattern)
        if fusion_cal:
            # Use measured speedup
            return fusion_cal.speedup_factor > 1.2  # 20% threshold
        else:
            # Fallback to heuristic
            return True  # Assume fusion is beneficial

    def fusion_benefit(self, pattern: str) -> float:
        """Get calibrated fusion speedup"""
        fusion_cal = self.calibration.get_fusion_pattern(pattern)
        if fusion_cal:
            return fusion_cal.speedup_factor
        return 1.0  # No benefit if unknown
```

### 7. Implementation Plan

#### Phase 1: Linear Fusion Benchmarks
- `benchmarks/fused_linear_bench.py`
- Patterns: Linear+Bias, Linear+Bias+ReLU, Linear+Bias+GELU
- Target: CPU and GPU

#### Phase 2: Convolution Fusion Benchmarks
- `benchmarks/fused_conv_bench.py`
- Patterns: Conv+BN, Conv+BN+ReLU, Conv+ReLU
- Target: CPU and GPU

#### Phase 3: Attention Fusion Benchmarks
- `benchmarks/fused_attention_bench.py`
- Patterns: QK^T, Softmax, Attention head
- Target: GPU (CPU attention is rare)

#### Phase 4: Element-wise Fusion Benchmarks
- `benchmarks/fused_elementwise_bench.py`
- Patterns: Add+ReLU, Mul+Add, Sigmoid+Mul
- Target: CPU and GPU

#### Phase 5: Integration
- Extend `HardwareCalibration` schema
- Update calibrator orchestrator
- Generate fusion-aware profiles

### 8. Validation Strategy

1. **Sanity Check**: Fused should always be faster than unfused
2. **Expected Speedup**: 1.5-3× for memory-bound patterns
3. **Cross-Hardware**: Compare CPU vs GPU fusion benefits
4. **Real Workloads**: Validate against ResNet, BERT inference

### 9. Open Questions

1. **How to handle PyTorch's automatic fusion?**
   - PyTorch may fuse automatically in eager mode
   - Use `torch.jit.script` for controlled fusion
   - Or use lower-level APIs (cuDNN, MKL-DNN)

2. **What about different tensor sizes?**
   - Small tensors: fusion overhead may dominate
   - Large tensors: fusion benefits memory bandwidth
   - Need size-dependent calibration

3. **How to measure memory accurately?**
   - PyTorch memory profiler?
   - Theoretical calculation?
   - Hardware performance counters?

4. **Should we benchmark on GPU too?**
   - GPU fusion is even more critical (kernel launch overhead)
   - But requires CUDA/cuDNN expertise
   - Start with CPU, extend to GPU later

## Next Steps

1. Implement `fused_linear_bench.py` (Linear+Bias+ReLU)
2. Test on i7-12700K
3. Validate speedup matches expectations
4. Extend schema to store fusion calibration
5. Add to calibrator orchestrator
6. Generate fusion-aware profile

## Expected Impact

- **Quantify fusion benefits**: Real data instead of assumptions
- **Improve partitioner**: Use measured speedup in cost model
- **Hardware-aware fusion**: Different strategies for CPU vs GPU
- **Validate optimizations**: Prove fusion actually helps
- **Debugging tool**: Identify when fusion hurts performance

This extension makes the calibration framework **complete** for real-world DNN workloads.
