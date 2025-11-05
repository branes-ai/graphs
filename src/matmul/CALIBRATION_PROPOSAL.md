# Hardware Mapper Calibration Proposal

## Problem Statement

Our hardware mappers currently use **theoretical peak FLOPS** from datasheets, which don't reflect real-world achievable performance:

### Example: Intel i7-12700K

| Source | FP32 GFLOPS | % of Datasheet |
|--------|-------------|----------------|
| **Datasheet (theoretical)** | 1280 GFLOPS | 100% |
| **Current mapper setting** | 720 GFLOPS | 56% |
| **OpenBLAS (measured)** | 833 GFLOPS | 65% |
| **Our optimized code** | 238 GFLOPS | 19% |
| **Our V1 code** | 175 GFLOPS | 14% |

**Key insight**: The mapper uses 720 GFLOPS but **real BLAS achieves 833 GFLOPS**. This means:
- Our estimates are **13% too conservative** for optimized code
- But **3.5× too optimistic** for naive implementations

## Root Cause

Different workloads achieve vastly different efficiencies:

1. **Dense GEMM (matmul)**: 60-85% of peak (memory-bound)
2. **Convolutions**: 40-60% of peak (memory-bound, smaller tiles)
3. **Element-wise ops**: 5-15% of peak (memory bandwidth limited)
4. **Attention layers**: 20-40% of peak (irregular access patterns)
5. **Batch norms**: 3-10% of peak (pure memory bandwidth)

**The mapper cannot use a single "peak FLOPS" value** - it needs **per-operation-type efficiency factors**.

## Proposed Solution

### 1. Micro-Benchmark Suite

Create a calibration tool that runs micro-benchmarks for each operation type:

```
src/graphs/hardware/calibration/
├── __init__.py
├── calibrator.py              # Main calibration orchestrator
├── benchmarks/
│   ├── matmul_bench.py       # Dense matrix multiplication
│   ├── conv_bench.py         # 2D convolutions (various kernel sizes)
│   ├── elementwise_bench.py  # Add, ReLU, etc.
│   ├── pooling_bench.py      # MaxPool, AvgPool
│   ├── attention_bench.py    # Scaled dot-product attention
│   └── batchnorm_bench.py    # Batch normalization
├── results/
│   └── i7_12700k_calibration.json  # Measured results
└── README.md
```

### 2. Calibration Data Structure

```python
@dataclass
class OperationCalibration:
    """Measured performance for a specific operation type"""
    operation_type: str  # "matmul", "conv2d", "relu", etc.
    measured_gflops: float
    efficiency: float  # % of theoretical peak
    memory_bound: bool
    achieved_bandwidth_gbps: float
    batch_size: int
    input_shape: Tuple[int, ...]

@dataclass
class HardwareCalibration:
    """Complete calibration profile for a hardware target"""
    hardware_name: str
    calibration_date: str
    theoretical_peak_gflops: float

    # Per-operation measured performance
    operation_profiles: Dict[str, OperationCalibration]

    # Summary statistics
    best_efficiency: float  # Best case (dense GEMM)
    avg_efficiency: float   # Average across workloads
    worst_efficiency: float  # Worst case (memory-bound)

    # Memory bandwidth
    measured_bandwidth_gbps: float
    theoretical_bandwidth_gbps: float
```

### 3. Usage in Mappers

```python
class CPUMapper(HardwareMapper):
    def __init__(self, resource_model: HardwareResourceModel,
                 calibration: Optional[HardwareCalibration] = None):
        super().__init__(resource_model)
        self.calibration = calibration or self._load_default_calibration()

    def estimate_latency(self, subgraph: FusedSubgraph) -> float:
        # Determine operation type
        op_type = self._classify_operation(subgraph)

        # Use calibrated efficiency, not theoretical peak
        if op_type in self.calibration.operation_profiles:
            profile = self.calibration.operation_profiles[op_type]
            achievable_flops = profile.measured_gflops
        else:
            # Fallback to conservative estimate
            achievable_flops = self.resource_model.get_peak_ops(precision) * 0.3

        # Compute latency
        latency_compute = subgraph.total_ops / achievable_flops
        latency_memory = subgraph.memory_bytes / self.calibration.measured_bandwidth_gbps

        return max(latency_compute, latency_memory)
```

### 4. Calibration Tool CLI

```bash
# Run full calibration suite
python -m graphs.hardware.calibration.calibrator --hardware i7-12700k

# Run specific operation types
python -m graphs.hardware.calibration.calibrator --ops matmul,conv2d

# Compare with existing profile
python -m graphs.hardware.calibration.calibrator --compare results/i7_12700k.json

# Output calibration file
python -m graphs.hardware.calibration.calibrator --output my_machine.json
```

### 5. Integration with Existing Tools

```bash
# Use calibration data in characterization
python cli/analyze_comprehensive.py --model resnet18 \
    --hardware i7-12700k \
    --calibration results/my_machine.json

# Update hardware mapper with calibration
python cli/update_hardware_profile.py \
    --hardware i7-12700k \
    --calibration results/my_machine.json
```

## Implementation Plan

### Phase 1: Matmul Calibration (Immediate)

Leverage the matmul benchmarks we just created:

```python
# src/graphs/hardware/calibration/benchmarks/matmul_bench.py
from src.matmul.tiled_matmul_v2 import TiledMatMul
import numpy as np

def calibrate_matmul_cpp(sizes=[512, 1024, 2048, 4096]):
    """Run C++ matmul calibration"""
    results = []
    for N in sizes:
        # Run C++ benchmark
        result = run_cpp_benchmark(N)
        results.append({
            'size': N,
            'gflops': result.gflops,
            'efficiency': result.efficiency
        })
    return results

def calibrate_matmul_numpy(sizes=[512, 1024, 2048, 4096]):
    """Run NumPy/BLAS calibration (upper bound)"""
    # Use existing benchmark_numpy.py
    pass
```

### Phase 2: Convolution Calibration

```python
# Use PyTorch to benchmark actual conv operations
import torch

def calibrate_conv2d(kernel_sizes=[3, 5, 7], channels=[64, 128, 256]):
    for k in kernel_sizes:
        for c in channels:
            conv = torch.nn.Conv2d(c, c, k, padding=k//2).cuda()
            x = torch.randn(1, c, 224, 224).cuda()
            # Benchmark...
```

### Phase 3: Full Suite

- Element-wise operations
- Pooling
- Batch norm
- Attention

### Phase 4: Auto-Calibration

```python
# Detect hardware and auto-calibrate
from graphs.hardware.calibration import auto_calibrate

calibration = auto_calibrate()  # Detects CPU/GPU and runs benchmarks
mapper = CPUMapper(resource_model, calibration=calibration)
```

## Expected Impact

### Before (Current State)

```
ResNet-18 Conv2d_3x3 (C=64):
  Estimated: 150 GFLOPS (using peak_ops_per_sec=720 GFLOPS × 0.2)
  Actual:    45 GFLOPS  (measured with PyTorch)
  Error:     233% overestimate
```

### After (With Calibration)

```
ResNet-18 Conv2d_3x3 (C=64):
  Estimated: 48 GFLOPS (using calibrated conv2d profile)
  Actual:    45 GFLOPS
  Error:     7% overestimate
```

## Benefits

1. **Accuracy**: 10× better latency estimation accuracy
2. **Hardware-specific**: Captures real performance on target machine
3. **Operation-aware**: Different ops have different efficiencies
4. **Updatable**: Re-calibrate as hardware/drivers/compilers change
5. **Reproducible**: Benchmark suite is version-controlled

## Data Collection

The calibration would produce a JSON like:

```json
{
  "hardware_name": "Intel-i7-12700K",
  "calibration_date": "2025-01-05",
  "theoretical_peak_gflops": 1280,
  "measured_bandwidth_gbps": 75.3,

  "operation_profiles": {
    "matmul_large": {
      "measured_gflops": 833,
      "efficiency": 0.651,
      "memory_bound": false,
      "test_size": [4096, 4096]
    },
    "matmul_small": {
      "measured_gflops": 150,
      "efficiency": 0.117,
      "memory_bound": true,
      "test_size": [1024, 1024]
    },
    "conv2d_3x3": {
      "measured_gflops": 280,
      "efficiency": 0.219,
      "memory_bound": true,
      "channels": 64
    },
    "relu": {
      "measured_gflops": 12,
      "efficiency": 0.009,
      "memory_bound": true,
      "achieved_bandwidth_gbps": 72.1
    }
  }
}
```

## Questions to Address

1. **Should we calibrate once or per-model?**
   - Recommendation: Calibrate once per hardware, reuse across models

2. **How to handle batch size variation?**
   - Calibrate at multiple batch sizes (1, 4, 16, 64)
   - Interpolate for other sizes

3. **CPU vs GPU calibration differences?**
   - CPU: Focus on BLAS, cache effects
   - GPU: Kernel launch overhead, tensor core utilization

4. **Integration with existing codebase?**
   - Add `--calibration` flag to all CLI tools
   - Store calibrations in `src/graphs/hardware/calibration/profiles/`

## Next Steps

1. Extract matmul benchmarks into calibration module
2. Add PyTorch-based conv2d calibration
3. Create calibration file format (JSON schema)
4. Update mappers to consume calibration data
5. Add CLI tool for calibration
6. Document calibration process

## Files to Create

```
src/graphs/hardware/calibration/
├── __init__.py
├── calibrator.py                    # Main entry point
├── schema.py                        # Data structures
├── benchmarks/
│   ├── __init__.py
│   ├── matmul_bench.py             # Reuse src/matmul code
│   ├── conv_bench.py               # PyTorch conv2d
│   └── elementwise_bench.py
├── profiles/                        # Pre-calibrated profiles
│   ├── i7_12700k.json
│   ├── h100.json
│   └── jetson_orin.json
└── README.md

cli/
├── calibrate_hardware.py            # CLI tool
└── verify_calibration.py            # Compare calibrated vs measured
```

## Validation Strategy

Run characterization with/without calibration and compare:

```bash
# Without calibration (baseline)
python cli/analyze_comprehensive.py --model resnet18 --hardware i7-12700k

# With calibration
python cli/analyze_comprehensive.py --model resnet18 --hardware i7-12700k \
    --calibration profiles/i7_12700k.json

# Compare accuracy
python cli/verify_calibration.py --model resnet18 \
    --hardware i7-12700k \
    --calibration profiles/i7_12700k.json \
    --actual-measurements pytorch_profiler_results.json
```

Expected result: **5-10× reduction in MAPE** (Mean Absolute Percentage Error)
