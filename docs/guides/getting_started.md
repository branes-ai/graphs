# Getting Started with Graph Partitioner

This guide will help you start working with the graph partitioner and understand how your neural network models interact with hardware.

## What is the Graph Partitioner?

The graph partitioner analyzes PyTorch models to understand:
- **Computational complexity**: How many FLOPs/MACs?
- **Memory requirements**: How much data movement?
- **Parallelism potential**: Can operations run concurrently?
- **Hardware bottlenecks**: Is the model compute-bound or memory-bound?

This is the foundation for **realistic performance modeling** - understanding how models actually run on real hardware, not just theoretical peak performance.

## Quick Start (5 minutes)

### Step 1: Run the Quick Start Example

```bash
cd /home/stillwater/dev/branes/clones/graphs
source ~/venv/p311/bin/activate
python examples/quick_start_partitioner.py
```

This will show you:
- How ResNet-18 is partitioned into 60 subgraphs
- Concurrency analysis (9 stages, max 12 parallel operations)
- Top 5 most compute-intensive operations
- Bottleneck classification (compute-bound vs memory-bound)

### Step 2: Compare Different Models

```bash
python examples/compare_models.py
```

This compares ResNet-18, MobileNet-V2, and EfficientNet-B0 side-by-side:
- **ResNet-18**: 4.49 GFLOPs, compute-intensive (AI=31)
- **MobileNet-V2**: 1.91 GFLOPs, balanced (AI=14)
- **EfficientNet-B0**: 2.39 GFLOPs, balanced (AI=17), best graph parallelism (27×)

**Key insight**: EfficientNet-B0 has the most graph-level parallelism (27 parallel ops) but MobileNet-V2 has the lowest FLOPs!

### Step 3: Validate a Model

```bash
python tests/test_graph_partitioner_general.py resnet18
```

This runs comprehensive validation checks to ensure the partitioner is working correctly.

## Understanding the Output

### Partition Summary

```
Total subgraphs: 60               ← Number of operations detected
Total FLOPs: 4.49 G              ← Computational complexity
Average arithmetic intensity: 31  ← FLOPs per byte (higher = compute-bound)
```

**Arithmetic Intensity (AI)** is crucial:
- **AI > 40**: Compute-bound - benefits from high-FLOPS hardware
- **AI 10-40**: Balanced - typical for most edge accelerators
- **AI < 10**: Memory-bound - high bandwidth critical (common on edge devices)

### Operation Types

```
Operation types:
  conv2d: 17 (28%)                ← Standard convolutions (compute-intensive)
  batchnorm: 20 (33%)             ← Normalization (memory-bound)
  relu: 17 (28%)                  ← Activations (bandwidth-bound)
  conv2d_depthwise: 0             ← Efficient convs (only in MobileNet/EfficientNet)
```

### Bottleneck Distribution

```
Bottleneck distribution:
  bandwidth_bound: 38 (63%)       ← Most operations are limited by memory bandwidth
  compute_bound: 11 (18%)         ← Few operations actually compute-bound
```

**Key insight**: Even though ResNet-18 overall is compute-intensive (AI=31), **most operations** (63%) are bandwidth-bound! This is because activations and batch norms dominate the operation count.

### Concurrency Analysis

```
Graph-level Parallelism:
  Max parallel ops per stage: 12   ← At most 12 operations can run simultaneously

Parallelism Potential:
  - Graph-level: 12× (operations that can run in parallel)
  - Batch-level: 1× (independent samples in batch)
  - Thread-level: 110,267 avg threads per op
```

**Critical insight**: With batch=1, ResNet-18 can only achieve **12x parallelism** from the graph structure. To fully utilize hardware like Jetson Orin AGX (2048 CUDA cores), you need:
- **Batch size >= 4** to better saturate the hardware
- Or other forms of parallelism (multiple requests, pipelining)

This is why naive latency estimates based on peak FLOPS are often too optimistic!

## Key Concepts

### 1. Subgraph = Kernel/Operation

Each subgraph represents a single operation (convolution, activation, etc.) that will map to a hardware kernel.

```python
# After partitioning...
for sg in report.subgraphs:
    print(f"{sg.node_name}: {sg.flops / 1e9:.3f} GFLOPs, "
          f"AI={sg.arithmetic_intensity:.1f}, "
          f"{sg.parallelism.total_threads:,} threads")
```

### 2. Arithmetic Intensity = Compute/Memory Ratio

**Formula**: `AI = FLOPs / (input_bytes + output_bytes + weight_bytes)`

This tells you whether an operation is:
- **Limited by compute** (high AI) → Benefits from faster cores
- **Limited by memory bandwidth** (low AI) → Benefits from faster memory

### 3. Graph-Level vs Thread-Level Parallelism

**Graph-level parallelism**: How many operations can run **concurrently**
- ResNet-18: Max 12 ops in parallel (residual blocks can be independent)
- EfficientNet-B0: Max 27 ops in parallel (more branches)

**Thread-level parallelism**: How many threads within **each operation**
- Conv layers: 100K-800K threads (very parallel)
- Linear layers: 1K-10K threads (less parallel)

**Both are needed** to saturate hardware!

### 4. Critical Path

The **critical path** is the longest chain of sequential dependencies. It determines the **minimum latency** even with infinite parallelism.

```
Critical path length: 9 operations
Critical path FLOPs: 1.73 GFLOPs (38% of total)
```

Even if you had infinite parallel compute, you cannot go faster than executing these 9 operations sequentially.

## Common Use Cases

### Use Case 1: Choosing Hardware for a Model

```bash
python examples/compare_models.py --models resnet18 mobilenet_v2
```

Look at:
- **Arithmetic Intensity**: High AI → GPU, Low AI → edge device
- **Max Parallel Ops**: High parallelism → can utilize big GPUs
- **Memory footprint**: Total weights + activations

**Example**:
- ResNet-18 (AI=31, 12 parallel) - Jetson Orin AGX, batch>=4
- MobileNet-V2 (AI=14, 12 parallel) - Jetson Orin Nano or Coral Edge TPU

### Use Case 2: Understanding Why Model is Slow

```python
# After partitioning...

# Check bottleneck distribution
from collections import Counter
bottlenecks = Counter(sg.recommended_bottleneck.value for sg in report.subgraphs)
print(bottlenecks)
# → If mostly 'bandwidth_bound', you need faster memory, not faster compute!

# Check graph parallelism
analyzer = ConcurrencyAnalyzer()
concurrency = analyzer.analyze(report)
print(f"Max parallel: {concurrency.max_parallel_ops_per_stage}")
# → If low (e.g., <16), you need batching to saturate GPU
```

### Use Case 3: Validating FLOP Counts

```bash
python tests/test_graph_partitioner_general.py resnet18
```

This checks:
- ✓ FLOPs in expected range (3.5-5.0 G for ResNet-18)
- ✓ Parallelism detected
- ✓ Concurrency analysis valid

### Use Case 4: Finding Depthwise Convolutions

```python
# MobileNet and EfficientNet use depthwise separable convolutions
depthwise_ops = [sg for sg in report.subgraphs
                 if sg.parallelism and sg.parallelism.is_depthwise]

for sg in depthwise_ops:
    print(f"{sg.node_name}: {sg.arithmetic_intensity:.1f} AI")
    # → Depthwise convs have LOW arithmetic intensity (memory-bound!)
```

**Why this matters**: Depthwise convolutions have limited channel parallelism and are memory-bound. They may not fully utilize GPU tensor cores even though they're "efficient" in FLOPs.

## Learning Path

We recommend this progression:

### Week 1: Basics
1. ✅ Run `quick_start_partitioner.py` on ResNet-18
2. ✅ Run `compare_models.py` on 3+ models
3. ✅ Run validation tests to understand what "correct" looks like
4. **Exercise**: Try different batch sizes, see how parallelism changes

### Week 2: Understanding
1. Read Tutorial 1-2 in `docs/graph_partitioner_tutorial.md`
2. Write code to analyze a custom model
3. **Exercise**: Count bottleneck types, find depthwise convolutions
4. **Exercise**: Compare critical path across models

### Week 3: Deep Dive
1. Read Tutorial 3 (Concurrency analysis)
2. Visualize dependency graphs
3. **Exercise**: Calculate minimum batch size to saturate hardware
4. **Exercise**: Predict which model will be faster on specific hardware

### Week 4: Advanced
1. Read Tutorial 4-5 (Custom validation, debugging)
2. Create custom validation for your use case
3. Add new models to validation framework
4. **Exercise**: Validate memory budgets, hardware suitability

## Example Scripts

### Compare Multiple Models

```python
from examples.compare_models import analyze_model, AVAILABLE_MODELS

results = []
for name, model_fn in [("resnet18", AVAILABLE_MODELS['resnet18']),
                       ("mobilenet_v2", AVAILABLE_MODELS['mobilenet_v2'])]:
    result = analyze_model(name, model_fn, verbose=True)
    results.append(result)

# Which has better parallelism?
for r in results:
    print(f"{r['name']}: {r['max_parallel']} parallel ops")
```

### Analyze Custom Model

```python
import torch
import torch.nn as nn
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.reporting import ReportGenerator

# Your custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Analyze using UnifiedAnalyzer
model = MyModel()
input_tensor = torch.randn(1, 3, 224, 224)

analyzer = UnifiedAnalyzer()
result = analyzer.analyze_graph(
    model=model,
    input_tensor=input_tensor,
    model_name='MyCustomCNN',
    hardware_name='Jetson-Orin-Nano'
)

# Generate report
generator = ReportGenerator()
print(generator.generate_text_report(result))
```

### Check Memory Budget

```python
# Will my model fit in 256MB?
total_weights = sum(sg.total_weight_bytes for sg in report.subgraphs)
max_activation = max(sg.total_input_bytes + sg.total_output_bytes
                     for sg in report.subgraphs)
peak_memory = (total_weights + max_activation) / (1024**2)

print(f"Peak memory: {peak_memory:.1f} MB")
if peak_memory <= 256:
    print("✓ Fits in budget")
else:
    print(f"✗ Exceeds by {peak_memory - 256:.1f} MB")
```

## What's Next?

### Milestone 1 (Complete): Foundation Consolidation
- Package reorganization: `core/`, `estimation/`, `frontends/`, `calibration/`, `benchmarks/`
- Confidence tracking for all estimates
- UnifiedAnalyzer orchestrates all analysis components

### Current Capabilities
- **Graph Partitioning**: FusionBasedPartitioner decomposes models into fused subgraphs
- **Hardware Mapping**: Map subgraphs to hardware resources (SMs, cores, tiles, systolic arrays)
- **Roofline Analysis**: Realistic latency estimation based on compute/memory bounds
- **Energy Estimation**: Three-component model (compute, memory, static)
- **Memory Analysis**: Peak memory, activation timeline, hardware fit checks
- **Concurrency Analysis**: Graph-level parallelism and critical path

## Resources

- **Tutorial**: `docs/graph_partitioner_tutorial.md` (comprehensive hands-on guide)
- **Validation**: `docs/graph_partitioner_validation.md` (how validation works)
- **Implementation Plan**: `docs/realistic_performance_modeling_plan.md` (Phase 1-3 roadmap)
- **Examples**: `examples/` directory (quick start, comparison, custom analysis)
- **Tests**: `tests/test_graph_partitioner*.py` (validation tests)
- **Source Code**: `src/graphs/` (core/, estimation/, frontends/, hardware/)

## Troubleshooting

### FX Tracing Fails

Some models can't be traced:
```python
try:
    fx_graph = symbolic_trace(model)
except:
    print("FX tracing failed - model has dynamic control flow")
    # Try torch.jit.trace as fallback
```

### Zero FLOPs Detected

Check operation types:
```python
print(report.operation_type_counts)
# Look for 'unknown' operations that aren't being counted
```

### Unexpected Memory

Remember:
- Weights are counted per subgraph (may appear multiple times)
- Activations are intermediate tensors (not total memory footprint)
- Peak memory = weights + largest activation

## Questions?

Start with the tutorial and examples. If you're still stuck:
1. Check the validation tests to see expected behavior
2. Examine source code in `src/graphs/` (core/, estimation/, frontends/, hardware/)
3. Look at implementation plan for architecture overview

Happy analyzing!
