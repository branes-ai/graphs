# Graph Partitioner Tutorial: Hands-On Guide

## Getting Started

This tutorial will help you understand and work with the graph partitioner in detail.

### Prerequisites

```bash
# Activate virtual environment
source ~/venv/p311/bin/activate

# Ensure package is installed
cd /home/stillwater/dev/branes/clones/graphs
pip install -e .
```

---

## Tutorial 1: Your First Partition

Let's start with a simple model to understand the basics.

### Step 1: Create a Simple Model

```python
# simple_partition_example.py
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.concurrency_analyzer import ConcurrencyAnalyzer

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create and trace model
model = SimpleCNN()
model.eval()
input_tensor = torch.randn(1, 3, 32, 32)

fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)

# Partition the graph
partitioner = GraphPartitioner()
report = partitioner.partition(fx_graph)

# Analyze concurrency
analyzer = ConcurrencyAnalyzer()
concurrency = analyzer.analyze(report)

# Print results
print(report.summary_stats())
print("\nSubgraphs:")
for i, sg in enumerate(report.subgraphs, 1):
    print(f"\n{i}. {sg.node_name} ({sg.operation_type.value})")
    print(f"   FLOPs: {sg.flops:,}")
    print(f"   Memory: {(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes)/1024:.1f} KB")
    if sg.parallelism:
        print(f"   Threads: {sg.parallelism.total_threads:,}")
```

### Exercise 1.1: Run and Explore

1. Save the code above as `simple_partition_example.py`
2. Run it: `python simple_partition_example.py`
3. **Questions to answer**:
   - How many subgraphs were created?
   - Which operation has the most FLOPs?
   - Which operation has the most parallelism?
   - What's the total memory traffic?

### Exercise 1.2: Modify the Model

Try modifying the model:
```python
# Add BatchNorm
self.bn1 = nn.BatchNorm2d(16)
# In forward(): x = self.bn1(x) after relu1

# Add more layers
self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

**Questions**:
- How does adding BatchNorm change the subgraph count?
- How does the arithmetic intensity change?

---

## Tutorial 2: Understanding Subgraph Properties

Each subgraph has rich information. Let's explore it.

### Step 2.1: Inspect Individual Subgraphs

```python
# inspect_subgraph.py
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner

# Use a real model
model = models.resnet18(weights=None)
model.eval()

fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(torch.randn(1, 3, 224, 224))

partitioner = GraphPartitioner()
report = partitioner.partition(fx_graph)

# Find the most compute-intensive operation
most_flops = max(report.subgraphs, key=lambda sg: sg.flops)

print("Most Compute-Intensive Operation:")
print("=" * 60)
print(f"Name: {most_flops.node_name}")
print(f"Type: {most_flops.operation_type.value}")
print(f"FLOPs: {most_flops.flops / 1e9:.3f} GFLOPs ({most_flops.flops / report.total_flops * 100:.1f}%)")
print(f"\nMemory:")
print(f"  Input: {most_flops.total_input_bytes / 1e6:.2f} MB")
print(f"  Output: {most_flops.total_output_bytes / 1e6:.2f} MB")
print(f"  Weights: {most_flops.total_weight_bytes / 1e6:.2f} MB")
print(f"  Total: {(most_flops.total_input_bytes + most_flops.total_output_bytes + most_flops.total_weight_bytes) / 1e6:.2f} MB")
print(f"\nArithmetic Intensity: {most_flops.arithmetic_intensity:.2f} FLOPs/byte")
print(f"Bottleneck: {most_flops.recommended_bottleneck.value}")

if most_flops.parallelism:
    p = most_flops.parallelism
    print(f"\nParallelism:")
    print(f"  Batch: {p.batch}")
    print(f"  Channels: {p.channels}")
    print(f"  Spatial: {p.spatial}")
    print(f"  Total threads: {p.total_threads:,}")
    print(f"  Depthwise: {p.is_depthwise}")
    print(f"  Can split channels: {p.can_split_channels}")

print(f"\nDependencies: {len(most_flops.depends_on)} operations")
```

### Exercise 2.1: Bottleneck Classification

The partitioner classifies each operation as:
- **compute_bound**: High arithmetic intensity (>50 FLOPs/byte)
- **balanced**: Moderate (10-50 FLOPs/byte)
- **memory_bound**: Low (1-10 FLOPs/byte)
- **bandwidth_bound**: Very low (<1 FLOPs/byte)

**Task**: Write code to count how many operations fall into each category.

```python
# Solution starter:
bottleneck_counts = {}
for sg in report.subgraphs:
    bt = sg.recommended_bottleneck.value
    bottleneck_counts[bt] = bottleneck_counts.get(bt, 0) + 1

# Calculate what percentage of FLOPs are compute-bound vs memory-bound
```

### Exercise 2.2: Find Depthwise Convolutions

For MobileNet or EfficientNet:

```python
# Find all depthwise convolutions
depthwise_ops = [sg for sg in report.subgraphs
                 if sg.parallelism and sg.parallelism.is_depthwise]

print(f"Found {len(depthwise_ops)} depthwise convolutions")
for sg in depthwise_ops:
    print(f"  {sg.node_name}: {sg.parallelism.channels} channels")
```

**Questions**:
- How many depthwise convolutions are there?
- What's their average arithmetic intensity compared to standard convolutions?
- Why are they more memory-bound?

---

## Tutorial 3: Understanding Concurrency

Concurrency analysis tells you how parallelizable your model is.

### Step 3.1: Visualize Dependency Graph

```python
# visualize_dependencies.py
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.concurrency_analyzer import ConcurrencyAnalyzer

model = models.resnet18(weights=None)
model.eval()

fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(torch.randn(1, 3, 224, 224))

partitioner = GraphPartitioner()
report = partitioner.partition(fx_graph)

analyzer = ConcurrencyAnalyzer()
concurrency = analyzer.analyze(report)

# Print stages (operations that can run in parallel)
print("Execution Stages:")
print("=" * 60)
for i, stage in enumerate(concurrency.stages, 1):
    print(f"\nStage {i}: {len(stage)} operations can run in parallel")

    # Get subgraph names for this stage
    stage_ops = [sg.node_name for sg in report.subgraphs if sg.node_id in stage]
    if len(stage_ops) <= 5:
        for op in stage_ops:
            print(f"  - {op}")
    else:
        print(f"  - {stage_ops[0]}")
        print(f"  - {stage_ops[1]}")
        print(f"  ... ({len(stage_ops) - 4} more)")
        print(f"  - {stage_ops[-2]}")
        print(f"  - {stage_ops[-1]}")

print(f"\n\nCritical Path:")
print("=" * 60)
critical_path_ids = report.critical_path_subgraphs
critical_ops = [sg for sg in report.subgraphs if sg.node_id in critical_path_ids]
print(f"Length: {len(critical_ops)} operations")
print(f"Total FLOPs on critical path: {sum(sg.flops for sg in critical_ops) / 1e9:.2f} GFLOPs")
print(f"Percentage of total FLOPs: {sum(sg.flops for sg in critical_ops) / report.total_flops * 100:.1f}%")

# Optional: Visualize (requires matplotlib)
# Note: This may create a large graph for complex models
if len(report.subgraphs) < 50:
    try:
        # Build dependency graph
        G = nx.DiGraph()
        for sg in report.subgraphs:
            G.add_node(sg.node_id, label=sg.node_name[:10])
            for dep in sg.depends_on:
                if dep in [s.node_id for s in report.subgraphs]:
                    G.add_edge(dep, sg.node_id)

        # Draw
        pos = nx.spring_layout(G)
        plt.figure(figsize=(15, 10))
        nx.draw(G, pos, with_labels=False, node_size=100,
                node_color='lightblue', arrows=True)
        plt.title(f"Dependency Graph ({len(report.subgraphs)} nodes)")
        plt.savefig('dependency_graph.png', dpi=150, bbox_inches='tight')
        print("\nDependency graph saved to dependency_graph.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
```

### Exercise 3.1: Analyze Parallelism Potential

**Task**: Calculate the theoretical maximum speedup with different batch sizes.

```python
def analyze_batching_potential(concurrency, batch_sizes=[1, 4, 8, 16, 32]):
    """Analyze how batching affects potential speedup"""

    max_parallel_ops = concurrency.max_parallel_ops_per_stage

    print("Batching Analysis:")
    print("=" * 60)
    for batch in batch_sizes:
        # Theoretical speedup = graph parallelism × batch parallelism
        theoretical_speedup = max_parallel_ops * batch
        print(f"Batch size {batch:2d}: {theoretical_speedup:4d}× theoretical speedup")

    print(f"\nNote: Actual speedup limited by hardware (e.g., 132 SMs on H100)")

analyze_batching_potential(concurrency)
```

**Questions**:
- At what batch size do you saturate an H100 GPU (132 SMs)?
- What's the minimum batch size needed to utilize 50% of the GPU?

### Exercise 3.2: Compare Different Models

```python
# Compare concurrency across architectures
models_to_compare = [
    ("ResNet-18", models.resnet18),
    ("MobileNet-V2", models.mobilenet_v2),
    ("EfficientNet-B0", models.efficientnet_b0),
]

results = []
for name, model_fn in models_to_compare:
    model = model_fn(weights=None)
    model.eval()

    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(torch.randn(1, 3, 224, 224))

    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    analyzer = ConcurrencyAnalyzer()
    concurrency = analyzer.analyze(report)

    results.append({
        'name': name,
        'stages': concurrency.num_stages,
        'max_parallel': concurrency.max_parallel_ops_per_stage,
        'critical_path': concurrency.critical_path_length,
        'avg_threads': sum(sg.parallelism.total_threads for sg in report.subgraphs
                          if sg.parallelism) / len(report.subgraphs)
    })

# Print comparison
print("\nModel Concurrency Comparison:")
print("=" * 80)
print(f"{'Model':<20} {'Stages':<10} {'Max Parallel':<15} {'Critical Path':<15} {'Avg Threads':<15}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<20} {r['stages']:<10} {r['max_parallel']:<15} "
          f"{r['critical_path']:<15} {r['avg_threads']:<15,.0f}")
```

**Questions**:
- Which model has the most graph-level parallelism?
- Which has the longest critical path?
- What does this mean for single-sample vs batched inference?

---

## Tutorial 4: Custom Validation

Let's create custom validation checks for your specific use case.

### Step 4.1: Memory Budget Validation

```python
# memory_budget_validation.py
def validate_memory_budget(report, max_memory_mb=512):
    """Check if model fits in a specific memory budget"""

    # Calculate peak memory (weights + largest activation)
    total_weights = sum(sg.total_weight_bytes for sg in report.subgraphs)
    max_activation = max(sg.total_input_bytes + sg.total_output_bytes
                        for sg in report.subgraphs)

    peak_memory_bytes = total_weights + max_activation
    peak_memory_mb = peak_memory_bytes / (1024**2)

    fits = peak_memory_mb <= max_memory_mb

    print(f"Memory Budget Validation:")
    print(f"  Budget: {max_memory_mb} MB")
    print(f"  Weights: {total_weights / (1024**2):.2f} MB")
    print(f"  Max activation: {max_activation / (1024**2):.2f} MB")
    print(f"  Peak memory: {peak_memory_mb:.2f} MB")
    print(f"  Status: {'✓ FITS' if fits else '✗ EXCEEDS'}")

    if not fits:
        print(f"  Over budget by: {peak_memory_mb - max_memory_mb:.2f} MB")

    return fits

# Test on different models
for name, model_fn in [("ResNet-18", models.resnet18),
                       ("MobileNet-V2", models.mobilenet_v2)]:
    print(f"\n{name}:")
    model = model_fn(weights=None)
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(torch.randn(1, 3, 224, 224))

    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    validate_memory_budget(report, max_memory_mb=100)
```

### Exercise 4.1: Hardware Suitability Check

**Task**: Check if a model is suitable for a specific hardware target.

```python
def check_hardware_suitability(report, concurrency, hardware_profile):
    """
    Check if model is suitable for target hardware

    hardware_profile = {
        'name': 'Edge Device',
        'compute_units': 4,
        'memory_mb': 256,
        'min_arithmetic_intensity': 5.0  # memory-bound hardware
    }
    """

    checks = []

    # Check 1: Memory fits
    peak_memory_mb = (sum(sg.total_weight_bytes for sg in report.subgraphs) +
                     max(sg.total_input_bytes + sg.total_output_bytes
                         for sg in report.subgraphs)) / (1024**2)

    memory_fits = peak_memory_mb <= hardware_profile['memory_mb']
    checks.append(('Memory fits', memory_fits,
                   f"{peak_memory_mb:.1f} MB vs {hardware_profile['memory_mb']} MB"))

    # Check 2: Can utilize hardware
    can_utilize = concurrency.max_parallel_ops_per_stage >= hardware_profile['compute_units'] * 0.5
    checks.append(('Can utilize hardware', can_utilize,
                   f"{concurrency.max_parallel_ops_per_stage} parallel ops vs {hardware_profile['compute_units']} units"))

    # Check 3: Arithmetic intensity suitable
    suitable_intensity = report.average_arithmetic_intensity >= hardware_profile['min_arithmetic_intensity']
    checks.append(('Suitable arithmetic intensity', suitable_intensity,
                   f"{report.average_arithmetic_intensity:.1f} vs {hardware_profile['min_arithmetic_intensity']} min"))

    # Print results
    print(f"\nHardware Suitability Check: {hardware_profile['name']}")
    print("=" * 60)
    for name, passed, detail in checks:
        status = '✓' if passed else '✗'
        print(f"{status} {name}")
        print(f"    {detail}")

    overall = all(passed for _, passed, _ in checks)
    print(f"\nOverall: {'✓ SUITABLE' if overall else '✗ NOT SUITABLE'}")

    return overall

# Example usage
edge_device = {
    'name': 'Edge Device (Raspberry Pi 4)',
    'compute_units': 4,
    'memory_mb': 256,
    'min_arithmetic_intensity': 5.0
}

datacenter_gpu = {
    'name': 'Datacenter GPU (H100)',
    'compute_units': 132,
    'memory_mb': 80000,
    'min_arithmetic_intensity': 10.0
}

# Test your model against both
```

---

## Tutorial 5: Debugging and Troubleshooting

Common issues and how to diagnose them.

### Issue 1: Low FLOPs Detected

```python
def diagnose_low_flops(report, expected_flops_gflops):
    """Diagnose why FLOP count might be low"""

    actual_flops = report.total_flops / 1e9

    if actual_flops < expected_flops_gflops * 0.8:
        print("⚠ FLOPs lower than expected!")
        print(f"  Expected: {expected_flops_gflops:.2f} GFLOPs")
        print(f"  Actual: {actual_flops:.2f} GFLOPs")
        print(f"  Difference: {(1 - actual_flops/expected_flops_gflops)*100:.1f}%")
        print("\nPossible causes:")

        # Check for zero-FLOP operations
        zero_flop_ops = [sg for sg in report.subgraphs if sg.flops == 0]
        if zero_flop_ops:
            print(f"  1. {len(zero_flop_ops)} operations have zero FLOPs")
            print(f"     Types: {set(sg.operation_type.value for sg in zero_flop_ops)}")
            print(f"     → May need to add estimators for these operation types")

        # Check for 'unknown' operations
        unknown_ops = [sg for sg in report.subgraphs
                      if sg.operation_type.value == 'unknown']
        if unknown_ops:
            print(f"  2. {len(unknown_ops)} operations classified as 'unknown'")
            print(f"     → These operations not being counted")

        # Check if all conv layers detected
        conv_ops = [sg for sg in report.subgraphs
                   if 'conv' in sg.operation_type.value]
        print(f"  3. Found {len(conv_ops)} convolution operations")
        print(f"     → Verify this matches expected conv layer count")

# Usage
diagnose_low_flops(report, expected_flops_gflops=3.79)  # For ResNet-18
```

### Issue 2: FX Tracing Fails

```python
# Some models don't trace easily. Here's how to handle it:

def try_trace_with_fallbacks(model, input_shape):
    """Try multiple strategies to trace a model"""

    input_tensor = torch.randn(*input_shape)

    # Strategy 1: Standard symbolic_trace
    try:
        print("Trying standard symbolic_trace...")
        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        print("✓ Success with standard trace")
        return fx_graph
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Strategy 2: concrete_args (for dynamic models)
    try:
        print("\nTrying with concrete_args...")
        from torch.fx import symbolic_trace
        fx_graph = symbolic_trace(model, concrete_args={'x': input_tensor})
        ShapeProp(fx_graph).propagate(input_tensor)
        print("✓ Success with concrete_args")
        return fx_graph
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Strategy 3: torch.jit.trace (fallback)
    print("\n⚠ FX tracing not supported for this model")
    print("  Consider using torch.jit.trace or TorchScript instead")
    return None

# Test on problematic model
fx_graph = try_trace_with_fallbacks(model, (1, 3, 224, 224))
```

---

## Next Steps

### Recommended Learning Path

1. **Start Here**: Tutorial 1 (Simple partition)
2. **Understand Properties**: Tutorial 2 (Subgraph inspection)
3. **Learn Concurrency**: Tutorial 3 (Dependency analysis)
4. **Custom Validation**: Tutorial 4 (Your use case)
5. **Troubleshooting**: Tutorial 5 (When things go wrong)

### Advanced Topics

Once comfortable with basics:
- **Fusion pattern detection** (Phase 2)
- **Hardware mapping** (Phase 2)
- **Roofline modeling** (Phase 3)
- **Custom operation estimators** (add new op types)

### Questions to Explore

1. How does batch size affect memory usage? (Hint: modify input_tensor batch dimension)
2. What's the relationship between model depth and critical path length?
3. Can you predict which model will run faster on GPU by looking at partition stats?
4. How do skip connections (ResNet) affect graph-level parallelism?

---

## Getting Help

- **Code location**: `/home/stillwater/dev/branes/clones/graphs/src/graphs/characterize/`
- **Tests**: `/home/stillwater/dev/branes/clones/graphs/tests/test_graph_partitioner*.py`
- **Documentation**: `/home/stillwater/dev/branes/clones/graphs/docs/`

Try modifying the examples, break things, and see what happens. That's the best way to learn!
