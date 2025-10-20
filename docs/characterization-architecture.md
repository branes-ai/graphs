# Characterization Package Architecture

## Overview

The characterization package is designed to estimate performance metrics (FLOPs, memory, latency, energy) for DNN models across different hardware architectures. It uses PyTorch FX for graph tracing and a plugin-based architecture for extensibility.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Script                             │
│                  (run_characterization.py)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─ Models (MLP, Conv2D, ResNet)
                             ├─ Inputs (tensors)
                             ├─ ArchitectureProfiles (CPU/GPU/TPU/KPU)
                             └─ FusedOpRegistry
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SweepHarness                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  For each model:                                          │  │
│  │    1. FX Trace      → symbolic_trace(model)               │  │
│  │    2. Shape Prop    → ShapeProp.propagate(input)          │  │
│  │    3. For each architecture:                              │  │
│  │         FXGraphWalker.walk(fx_graph, arch) → metrics      │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FXGraphWalker                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Extract call_module nodes from FX graph               │  │
│  │  2. Match fused patterns (via FusedOpRegistry)            │  │
│  │  3. For each matched pattern:                             │  │
│  │       a. Estimate FLOPs & Memory (Estimator)              │  │
│  │       b. Compute tiles (TilingStrategy)                   │  │
│  │       c. Calculate latency (ArchProfile + scheduler)      │  │
│  │       d. Calculate energy (ArchProfile + tiles)           │  │
│  │  4. Aggregate all metrics                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├──────────────┬──────────────┬──────────────┐
                             ▼              ▼              ▼              ▼
                    ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐
                    │ FusedOp     │  │ Arch     │  │ Tiling       │  │ Models  │
                    │ Registry    │  │ Profile  │  │ Strategy     │  │         │
                    └─────────────┘  └──────────┘  └──────────────┘  └─────────┘
```

---

## Core Components

### 1. **SweepHarness** (`sweep.py`)
**Role**: Orchestration layer for batch characterization

**Responsibilities**:
- Manages collections of models, inputs, and architectures
- Performs FX tracing and shape propagation
- Coordinates walker execution across all model and architecture combinations
- Returns structured results

**Key Methods**:
```python
trace_and_propagate(model, input_tensor) → fx_graph
run() → List[{model, arch, metrics}]
```

**Design Pattern**: Facade pattern - provides simple interface to complex subsystem

---

### 2. **FXGraphWalker** (`walker.py`)
**Role**: Core characterization engine

**Responsibilities**:
- Traverses FX graph nodes
- Matches fused operation patterns
- Calls estimators to compute FLOPs/memory
- Applies architecture-specific tiling and scheduling
- Aggregates metrics

**Algorithm**:
```python
1. Extract all call_module nodes from FX graph
2. Use FusedOpRegistry to match multi-node patterns
3. For each matched pattern:
   a. estimator.estimate(nodes) → (flops, memory)
   b. tiling_strategy.compute_tile_count() → tiles
   c. latency = flops / peak_flops × scheduler_overhead × (1 + 5% × (tiles-1))
   d. energy = (flops × energy_per_flop + mem × energy_per_byte) × (1 + 5% × (tiles-1))
4. Sum all metrics
5. Return {FLOPs, Memory, Tiles, Latency, Energy}
```

**Key Insight**: The walker is **architecture-aware** - it takes an `ArchitectureProfile` and uses it to convert raw FLOPs into latency/energy estimates.

---

### 3. **ArchitectureProfile** (`arch_profiles.py`)
**Role**: Hardware specification container

**Attributes**:
```python
name              : str          # "CPU", "GPU", "TPU", "KPU"
peak_flops        : float        # Peak FLOPS (e.g., 10e12 for GPU)
mem_bandwidth     : float        # Memory bandwidth (bytes/sec)
energy_per_flop   : float        # Energy per FLOP (Joules)
energy_per_byte   : float        # Energy per byte transferred
scheduler_model   : Callable     # Function modeling scheduling overhead
tiling_strategy   : TilingStrategy  # How to tile operations
```

**Design Pattern**: Value Object + Strategy Pattern

**Purpose**:
- Encapsulates all hardware-specific parameters
- Makes it easy to add new architectures (just create a new profile)
- Separates hardware concerns from graph analysis

**Example**:
```python
gpu_profile = ArchitectureProfile(
    name="GPU",
    peak_flops=10e12,              # 10 TFLOPS
    mem_bandwidth=900e9,           # 900 GB/s
    energy_per_flop=0.5e-9,        # 0.5 nJ per FLOP
    energy_per_byte=30e-12,        # 30 pJ per byte
    scheduler_model=fused_scheduler,  # 0.6× multiplier (fusion benefit)
    tiling_strategy=GPUTilingStrategy()
)
```

---

### 4. **FusedOpRegistry** (`fused_ops.py`)
**Role**: Pattern matching engine for operation fusion

**Concept**: Real compilers fuse operations (e.g., Conv+BN+ReLU) into single kernels for efficiency. The registry identifies these patterns in the FX graph.

**Structure**:
```python
patterns = [
    {
        "name": "conv_relu",
        "sequence": [nn.Conv2d, nn.ReLU],
        "estimator": ConvReLUEstimator()
    },
    {
        "name": "linear_relu",
        "sequence": [nn.Linear, nn.ReLU],
        "estimator": LinearReLUEstimator()
    }
]
```

**Algorithm** (`match()` method):
```python
1. Iterate through FX graph nodes with sliding window
2. For each window, check if module types match a registered pattern
3. If match found:
   - Return (pattern_name, matched_nodes, estimator)
   - Skip ahead by pattern length
4. Continue until all nodes processed
```

**Key Insight**: Patterns are matched **greedily** in registration order. Longer patterns should be registered first (e.g., Conv+BN+ReLU before Conv+ReLU).

---

### 5. **Estimators** (`fused_ops.py`)
**Role**: Compute operation-specific FLOPs and memory

**Base Class**:
```python
class FusedEstimator:
    def estimate(self, fused_nodes) → (flops, memory)
```

**Example: ConvReLUEstimator**
```python
def estimate(self, fused_nodes):
    conv_node = fused_nodes[0]
    meta = conv_node.meta['tensor_meta']  # Shape from ShapeProp
    mod = conv_node.graph.owning_module.get_submodule(conv_node.target)

    # Extract dimensions
    B, C_in, H, W = meta.shape
    C_out, K_h, K_w = mod.out_channels, mod.kernel_size

    # Calculate output dimensions
    H_out = (H - K_h + 2*P) // S_h + 1
    W_out = (W - K_w + 2*P) // S_w + 1

    # FLOPs: 2 ops per MAC (multiply-accumulate) + ReLU
    flops = B × C_out × H_out × W_out × (2 × C_in × K_h × K_w + 1)

    # Memory: input + weights + output (4 bytes per float32)
    mem = B×C_in×H×W×4 + C_out×C_in×K_h×K_w×4 + B×C_out×H_out×W_out×4

    return flops, mem
```

**Key Insight**: Estimators use **tensor metadata** from ShapeProp to get actual tensor shapes, making estimates input-size-aware.

---

### 6. **TilingStrategy** (`tiling.py`)
**Role**: Compute how many tiles an operation requires

**Concept**: Operations are often split into "tiles" to fit in cache/scratchpad memory. More tiles = more overhead.

**Base Class**:
```python
class TilingStrategy:
    def compute_tile_count(self, op_type, op_metadata) → int
```

**Example: GPUTilingStrategy**
```python
def compute_tile_count(self, op_type, op_metadata):
    tile_mem = 48 * 1024  # 48 KB shared memory per block
    shape = op_metadata['input_shape']
    total_bytes = sum([dim * 4 for dim in shape])
    return max(1, total_bytes // tile_mem)
```

**Architecture-Specific Logic**:
- **CPU**: 256 KB L2 cache per core
- **GPU**: 48 KB shared memory per SM
- **TPU**: 24 MB unified buffer
- **KPU**: Wavefront-based (N×N must fit in 64 MB)

**Key Insight**: Tiling strategies capture memory hierarchy differences between architectures.

---

## Data Flow Example

Let's trace a Conv2D model through the pipeline:

### Step 1: FX Tracing
```python
# Input: PyTorch model
model = ParamConv2DStack(in_ch=3, out_ch=16, num_layers=3)

# FX trace creates graph IR
fx_graph = symbolic_trace(model)
# Graph: x → Conv2d → ReLU → Conv2d → ReLU → Conv2d → ReLU → out
```

### Step 2: Shape Propagation
```python
input_tensor = torch.randn(32, 3, 64, 64)
ShapeProp(fx_graph).propagate(input_tensor)

# Now each node has .meta['tensor_meta'].shape:
# Conv2d_0: [32, 16, 64, 64]
# ReLU_0:   [32, 16, 64, 64]
# Conv2d_1: [32, 16, 64, 64]
# ...
```

### Step 3: Pattern Matching
```python
nodes = [Conv2d_0, ReLU_0, Conv2d_1, ReLU_1, Conv2d_2, ReLU_2]
registry.match(nodes) → [
    ("conv_relu", [Conv2d_0, ReLU_0], ConvReLUEstimator),
    ("conv_relu", [Conv2d_1, ReLU_1], ConvReLUEstimator),
    ("conv_relu", [Conv2d_2, ReLU_2], ConvReLUEstimator)
]
```

### Step 4: Estimation (first Conv2d+ReLU)
```python
estimator = ConvReLUEstimator()
flops, mem = estimator.estimate([Conv2d_0, ReLU_0])
# flops = 32 × 16 × 64 × 64 × (2 × 3 × 3 × 3 + 1) = 600,866,816
# mem   = 32×3×64×64×4 + 16×3×3×3×4 + 32×16×64×64×4 = 1,572,864 + 1,728 + 2,097,152
```

### Step 5: Tiling
```python
gpu_tiling = GPUTilingStrategy()
tiles = gpu_tiling.compute_tile_count("conv_relu", {"input_shape": [32, 16, 64, 64]})
# tiles = 1 (fits in 48 KB)
```

### Step 6: Latency & Energy
```python
# GPU: 10 TFLOPS peak, 0.6× scheduler multiplier
latency = 600866816 / 10e12 × 0.6 × (1 + 0.05 × (1-1))
        = 3.605e-5 seconds

energy = (600866816 × 0.5e-9 + 1572864 × 30e-12) × (1 + 0.05 × (1-1))
       = 0.3004 + 0.00047 = 0.3005 J
```

### Step 7: Aggregation
```python
# Repeat for all 3 Conv2d+ReLU pairs
total_flops = 3 × 600866816 = 1,802,600,448
total_memory = 3 × 1,572,864 = ~4.7 MB
total_tiles = 3
total_latency = 3 × 3.605e-5 = ~0.0001 sec
total_energy = 3 × 0.3005 = ~0.9 J
```

---

## Design Principles

### 1. **Separation of Concerns**
- **Graph analysis** (walker, registry) ≠ **hardware modeling** (profiles, tiling)
- Can add new architectures without touching graph code
- Can add new fusion patterns without touching architecture code

### 2. **Extensibility via Plugins**
- New operations: Add estimator + register pattern
- New architectures: Create profile + tiling strategy
- New metrics: Extend walker's aggregation logic

### 3. **Composability**
- Components work independently: can test estimators without walker
- SweepHarness is optional: can use walker directly for single models
- FX tracing is decoupled: could swap for ONNX, TorchScript, etc.

### 4. **Data-Driven**
- All hardware params in profiles (not hardcoded)
- Estimators use actual tensor shapes (not assumptions)
- Registry is declarative (pattern matching, not if-else)

---

## Key Abstractions

| Component | Abstraction | Why It Matters |
|-----------|-------------|----------------|
| **ArchitectureProfile** | Hardware specs as data | Add new HW without code changes |
| **TilingStrategy** | Memory hierarchy as strategy | Different tiling logic per architecture |
| **FusedEstimator** | Operation cost model | Easy to add new ops (Conv3D, LSTM, etc.) |
| **FusedOpRegistry** | Pattern matching engine | Compiler-like fusion detection |
| **SweepHarness** | Batch orchestration | Run experiments at scale |

---

## Extending the System

### Adding a New Architecture (e.g., "NPU")
```python
# 1. Define tiling strategy
class NPUTilingStrategy(TilingStrategy):
    def compute_tile_count(self, op_type, op_metadata):
        # NPU-specific logic
        return tiles

# 2. Create profile
npu_profile = ArchitectureProfile(
    name="NPU",
    peak_flops=5e12,
    mem_bandwidth=200e9,
    energy_per_flop=0.3e-9,
    energy_per_byte=15e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=NPUTilingStrategy()
)

# 3. Add to sweep
arch_profiles = [cpu_profile, gpu_profile, tpu_profile, kpu_profile, npu_profile]
```

### Adding a New Fusion Pattern (e.g., "Conv+Add")
```python
# 1. Create estimator
class ConvAddEstimator(FusedEstimator):
    def estimate(self, fused_nodes):
        # Compute FLOPs for Conv + elementwise Add
        return flops, mem

# 2. Register pattern
registry.register("conv_add", [nn.Conv2d, Add], ConvAddEstimator())
```

---

## Limitations & Future Work

### Current Limitations
1. **Greedy pattern matching**: May miss optimal fusions
2. **Static shapes only**: Doesn't handle dynamic batch sizes
3. **No memory bandwidth modeling**: Assumes compute-bound
4. **Simple tiling**: Doesn't optimize tile sizes
5. **No operator support**: Only Conv2d, Linear, ReLU, BatchNorm

### Future Enhancements
1. **Graph optimization**: Find optimal fusion strategy (dynamic programming)
2. **Roofline modeling**: Memory vs compute bottleneck analysis
3. **Autotuning**: Search for best tile sizes per workload
4. **Wider operator coverage**: Attention, LSTM, LayerNorm, etc.
5. **Multi-device**: Model pipeline parallelism, tensor parallelism
6. **Calibration**: Fit energy models to real hardware measurements

---

## Summary

The characterization package uses a **layered architecture**:

1. **Orchestration Layer** (SweepHarness): Batch processing
2. **Analysis Layer** (FXGraphWalker): Graph traversal & aggregation
3. **Pattern Layer** (FusedOpRegistry + Estimators): Operation detection & costing
4. **Hardware Layer** (ArchitectureProfile + TilingStrategy): Architecture modeling

**Key Innovation**: Decoupling graph analysis from hardware modeling via profiles and strategies enables characterizing any model on any architecture with minimal code changes.

**Core Philosophy**: Make adding new models, operators, and architectures **configuration, not code**.
