# Empirical Benchmarking and Model Calibration

This directory contains empirical benchmark sweeps for calibrating analytical performance models against real hardware measurements.

## Purpose

**Empirical benchmarks answer**: "How accurate are our performance estimates?"

These benchmarks:
- Run **actual PyTorch models** on real hardware (CPU/GPU)
- Measure **real execution time, memory, energy**
- Compare against **analytical estimates** from FXGraphWalker
- Calibrate **efficiency_factor** coefficients in hardware mappers
- Identify **bottleneck transitions** (when compute → memory bound)

## Directory Structure

```
validation/empirical/
├── README.md                       # This file
├── sweep_mlp.py                    # MLP parameter sweep
├── sweep_conv.py                   # Conv2D parameter sweep
├── sweep_resnet.py                 # ResNet family sweep
├── calibration_analysis.py         # Calibration and error analysis
└── results/                        # Empirical data (CSV/JSON)
    ├── mlp_sweep_cpu.csv
    ├── mlp_sweep_gpu.csv
    └── calibration_report.md
```

## Empirical vs Validation vs Tests

| Aspect | Empirical | Validation | Tests |
|--------|-----------|------------|-------|
| **Purpose** | Calibrate models | Verify accuracy | Verify correctness |
| **Question** | "How close to reality?" | "Are estimates accurate?" | "Does code work?" |
| **Execution** | Real hardware | Analytical only | Analytical only |
| **Speed** | Slow (seconds-minutes) | Medium | Fast (<1s) |
| **Requires** | PyTorch + hardware | PyTorch | PyTorch |
| **Output** | Timing + estimates | Estimates only | Pass/fail |

## Sweep Parameters

### MLP Sweep Dimensions

```python
SWEEP_PARAMS = {
    # Model architecture
    'input_dim': [256, 512, 1024, 2048, 4096],
    'hidden_dims': [
        [128],                # 1-layer
        [512, 512],           # 2-layer
        [1024, 1024, 1024],   # 3-layer
        [2048] * 4,           # 4-layer
    ],
    'output_dim': [10, 64, 128, 256],

    # Execution parameters
    'batch_size': [1, 4, 16, 32, 64, 128],
    'precision': ['fp32', 'fp16', 'int8'],

    # Hardware targets
    'device': ['cpu', 'cuda'],

    # Memory scenarios (designed to trigger spills)
    'memory_scenario': [
        'L1_fit',      # Small enough for L1 cache
        'L2_fit',      # Fits in L2, spills from L1
        'L3_fit',      # Fits in L3, spills from L2
        'DRAM_spill',  # Too large for any cache
    ]
}
```

### Memory Scenario Calculations

```python
# For CPU (Intel Xeon example)
L1_DATA_CACHE = 32 * 1024      # 32 KB per core
L2_CACHE = 1 * 1024 * 1024     # 1 MB per core
L3_CACHE = 32 * 1024 * 1024    # 32 MB shared

# L1_fit scenario: Total working set < 32 KB
# Example: input=128, hidden=128, output=64, batch=1
# → ~200 KB parameters + 512 B activations < 32 KB? NO
# → Need smaller models or specific tile sizes

# L3_fit: Working set 1-32 MB
# Example: input=2048, hidden=2048, output=1024, batch=1
# → ~12 MB parameters → fits in L3, spills from L2
```

## Usage

### 1. Run MLP Sweep

```bash
# Full parameter sweep (slow - may take hours!)
python validation/empirical/sweep_mlp.py --full

# Quick smoke test (5 configs)
python validation/empirical/sweep_mlp.py --quick

# Specific device
python validation/empirical/sweep_mlp.py --device cuda --batch-sizes 1,16,64

# Target specific memory scenario
python validation/empirical/sweep_mlp.py --scenario L3_fit
```

### 2. Analyze Calibration

```bash
# Generate calibration report
python validation/empirical/calibration_analysis.py \
    --empirical results/mlp_sweep_cpu.csv \
    --analytical results/mlp_analytical_cpu.csv \
    --output results/calibration_report.md
```

### 3. Update Hardware Mappers

After calibration, update `efficiency_factor` coefficients:

```python
# Before calibration (in hardware_mapper.py)
efficiency_factor=0.70,  # Guessed

# After calibration (from calibration_report.md)
efficiency_factor=0.63,  # Measured: 63% of theoretical peak
```

## Output Format

### Empirical Sweep CSV

```csv
model,input_dim,hidden_dims,output_dim,batch_size,device,precision,
  empirical_time_ms,empirical_memory_mb,empirical_flops,
  analytical_time_ms,analytical_memory_mb,analytical_flops,
  time_error_pct,memory_error_pct,flops_error_pct

mlp_1layer,1024,[512],64,32,cpu,fp32,
  12.34,44.2,2.1e9,
  10.50,44.6,2.1e9,
  17.5,0.9,0.0
```

### Calibration Report

```markdown
# Calibration Report: MLP Sweep on Intel CPU

## Summary
- **Total configs tested**: 150
- **Mean Absolute Percentage Error (MAPE†)**:
  - Latency: 15.3%
  - Memory: 2.1%
  - FLOPs: 0.5% (excellent!)

† **MAPE** = Mean Absolute Percentage Error: Average of |empirical - analytical| / empirical × 100%

## Bottleneck Analysis
- **Compute-bound configs**: 45 (arithmetic intensity > 10)
  - MAPE latency: 8.2% (good!)
- **Memory-bound configs**: 105 (arithmetic intensity < 10)
  - MAPE latency: 18.7% (needs tuning)

## Recommended Updates
```python
# CPU mapper (src/graphs/characterize/cpu_mapper.py)
Precision.FP32: PerformanceCharacteristics(
    efficiency_factor=0.63,  # Was: 0.70 → Update to 0.63
    memory_bottleneck_factor=0.55,  # Was: 0.60 → Update
)
```

## Memory Scenario Transitions

| Scenario | Working Set | Observed Slowdown | AI Threshold |
|----------|-------------|-------------------|--------------|
| L1_fit   | < 32 KB    | 1.0× (baseline)   | > 50         |
| L2_fit   | 32 KB - 1 MB | 1.8×            | 20-50        |
| L3_fit   | 1-32 MB    | 3.2×              | 5-20         |
| DRAM_spill | > 32 MB  | 12.5×             | < 5          |
```

## Key Insights from Sweeps

### What We Learn

1. **Empirical Derate Coefficients**
   - FP32 on CPU: ~63% of peak (memory bottleneck)
   - FP32 on GPU: ~85% of peak (better memory bandwidth)
   - INT8 on GPU: ~40% of peak (instruction overhead)

2. **Batch Size Effects**
   - batch=1: Heavy kernel launch overhead
   - batch=16-32: Sweet spot for GPUs
   - batch>64: Diminishing returns, memory bound

3. **Memory Hierarchy Effects**
   - L1→L2 transition: 1.8× slowdown
   - L2→L3 transition: 1.8× additional slowdown
   - L3→DRAM transition: 3.9× additional slowdown

4. **Precision Effects**
   - FP16 on GPU: 1.8-2.0× faster than FP32 (Tensor Cores)
   - INT8 on GPU: 2.5-3.0× faster than FP32 (when quantized properly)
   - FP16 on CPU: Minimal speedup (emulated)

## Calibration Workflow

```
1. Define sweep parameters → validation/empirical/sweep_mlp.py
2. Run empirical benchmarks → Measure real hardware
3. Run analytical estimates → FXGraphWalker + HardwareMapper
4. Compare and analyze → calibration_analysis.py
5. Update coefficients → hardware_mapper.py (efficiency_factor)
6. Re-run validation → validation/estimators/ tests
7. Verify MAPE < 10% → Success!
```

## Adding New Sweeps

### Template: `sweep_<model_family>.py`

```python
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import time
import itertools

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner
from src.graphs.characterize.cpu_mapper import create_intel_cpu_mapper

# Define sweep parameters
SWEEP_PARAMS = {
    'param1': [value1, value2, ...],
    'param2': [value1, value2, ...],
}

# Model builder
def build_model(**params):
    # Construct model from parameters
    return model

# Empirical benchmark
def run_empirical(model, input_tensor, device='cpu'):
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    # Measure
    start = time.perf_counter()
    output = model(input_tensor)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.perf_counter()

    return {
        'time_ms': (end - start) * 1000,
        'memory_mb': get_memory_usage(device),
    }

# Analytical estimate
def run_analytical(model, input_tensor):
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    partitioner = FusionBasedPartitioner()
    report = partitioner.partition_graph(traced)

    mapper = create_intel_cpu_mapper()
    hw_report = mapper.map_to_hardware(report, precision='fp32')

    return {
        'time_ms': hw_report.total_latency * 1000,
        'memory_mb': hw_report.total_memory_traffic / 1e6,
        'flops': report.total_flops,
    }

# Sweep runner
def run_sweep():
    results = []

    for params in itertools.product(*SWEEP_PARAMS.values()):
        param_dict = dict(zip(SWEEP_PARAMS.keys(), params))

        model = build_model(**param_dict)
        input_tensor = create_input(**param_dict)

        empirical = run_empirical(model, input_tensor)
        analytical = run_analytical(model, input_tensor)

        results.append({**param_dict, **empirical, **analytical})

    return results
```

## References

- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- torch.cuda.memory_stats: https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
- Intel VTune Profiler: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
