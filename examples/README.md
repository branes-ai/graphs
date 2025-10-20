# Graph Partitioner Examples

This directory contains hands-on examples for working with the graph partitioner.

## Quick Start

Start here if you're new to the graph partitioner:

```bash
# navigate to the root directory of the Branes.ai graphs repo
cd <DEVROOT>/branes/clones/graphs
python examples/quick_start_partitioner.py
```

This will:
- Load ResNet-18 from the PyTorch torchvision.models package
- Trace that model with PyTorch FX
- Propagate all the tensor shapes throughout the model
- Partition the model into subgraphs
- Analyze concurrency
- Show top compute-intensive operations
- Provide next steps for exploration

## Available Examples

### 1. Quick Start (quick_start_partitioner.py)
**What it does**: Basic introduction to partitioning and concurrency analysis

**When to use**: First time using the partitioner

**Output**:
- Partition summary (FLOPs, memory, operation counts)
- Concurrency analysis (stages, parallelism)
- Top 5 most compute-intensive operations

### 2. Validation Tests (../tests/test_graph_partitioner_general.py)
**What it does**: Validates partitioner correctness across multiple models

**When to use**: Verify that partitioner works correctly on a model

**Usage**:
```bash
# Test single model
python tests/test_graph_partitioner_general.py resnet18

# Test multiple models
python tests/test_graph_partitioner_general.py resnet18 mobilenet_v2 efficientnet_b0

# Test all defined models
python tests/test_graph_partitioner_general.py
```

**Output**:
- Universal validation checks (applies to all models)
- Architecture-specific validation (expected ranges)
- Pass/fail status for each check

## Learning Path

We recommend this progression:

1. **Start**: Run `quick_start_partitioner.py` to see basic output
2. **Validate**: Run validation tests to understand what "correct" looks like
3. **Tutorial**: Work through `docs/graph_partitioner_tutorial.md` tutorials
4. **Experiment**: Modify examples and create your own

## Common Tasks

### Compare Two Models

```python
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.concurrency_analyzer import ConcurrencyAnalyzer

def analyze_model(model, name, input_shape=(1, 3, 224, 224)):
    model.eval()
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(torch.randn(*input_shape))

    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    analyzer = ConcurrencyAnalyzer()
    concurrency = analyzer.analyze(report)

    return {
        'name': name,
        'flops': report.total_flops / 1e9,
        'subgraphs': report.total_subgraphs,
        'arithmetic_intensity': report.average_arithmetic_intensity,
        'stages': concurrency.num_stages,
        'max_parallel': concurrency.max_parallel_ops_per_stage
    }

# Compare ResNet-18 vs MobileNet-V2
resnet = analyze_model(models.resnet18(weights=None), "ResNet-18")
mobilenet = analyze_model(models.mobilenet_v2(weights=None), "MobileNet-V2")

print(f"ResNet-18:    {resnet['flops']:.2f} GFLOPs, AI={resnet['arithmetic_intensity']:.1f}")
print(f"MobileNet-V2: {mobilenet['flops']:.2f} GFLOPs, AI={mobilenet['arithmetic_intensity']:.1f}")
```

### Find Specific Operations

```python
# After partitioning a model...

# Find all depthwise convolutions
depthwise = [sg for sg in report.subgraphs
             if sg.parallelism and sg.parallelism.is_depthwise]
print(f"Found {len(depthwise)} depthwise convolutions")

# Find compute-bound operations
compute_bound = [sg for sg in report.subgraphs
                 if sg.recommended_bottleneck.value == 'compute_bound']
print(f"{len(compute_bound)} compute-bound operations")

# Find operations with high parallelism
high_parallel = [sg for sg in report.subgraphs
                 if sg.parallelism and sg.parallelism.total_threads > 100000]
print(f"{len(high_parallel)} operations with >100K threads")
```

### Analyze Memory Footprint

```python
# After partitioning...

# Total weights
total_weights = sum(sg.total_weight_bytes for sg in report.subgraphs)
print(f"Total weights: {total_weights / 1e6:.2f} MB")

# Peak activation memory (largest single operation)
max_activation = max(sg.total_input_bytes + sg.total_output_bytes
                     for sg in report.subgraphs)
print(f"Peak activation: {max_activation / 1e6:.2f} MB")

# Total memory traffic
total_traffic = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                   for sg in report.subgraphs)
print(f"Total memory traffic: {total_traffic / 1e9:.2f} GB")
```

### Visualize Critical Path

```python
import networkx as nx

# After concurrency analysis...

# Build dependency graph
G = nx.DiGraph()
for sg in report.subgraphs:
    G.add_node(sg.node_id)
    for dep in sg.depends_on:
        if dep in [s.node_id for s in report.subgraphs]:
            G.add_edge(dep, sg.node_id)

# Find critical path
flop_map = {sg.node_id: sg.flops for sg in report.subgraphs}
critical_path = nx.dag_longest_path(G, weight=lambda u, v, d: flop_map.get(v, 0))

# Print critical path operations
print("Critical Path:")
for node_id in critical_path:
    sg = next(s for s in report.subgraphs if s.node_id == node_id)
    print(f"  {sg.node_name} ({sg.flops / 1e9:.3f} GFLOPs)")
```

## Troubleshooting

### FX Tracing Fails

Some models have dynamic control flow that FX cannot trace. Try:

1. Use concrete_args: `symbolic_trace(model, concrete_args={'x': input_tensor})`
2. Use torch.jit.trace as fallback
3. Check if model has dynamic shapes (e.g., YOLO)

### Zero FLOPs Detected

If total FLOPs is unexpectedly low:

1. Check operation type counts: `report.operation_type_counts`
2. Look for 'unknown' operations that aren't being counted
3. Verify all conv layers were detected

### Memory Estimates Seem Off

Remember:
- Weights are counted once per subgraph (may be reused)
- Activations are intermediate tensors (not final memory footprint)
- Peak memory ≠ total memory traffic

## Next Steps

- Read the full tutorial: `docs/graph_partitioner_tutorial.md`
- Explore validation framework: `docs/graph_partitioner_validation.md`
- See implementation plan: `docs/realistic_performance_modeling_plan.md`

## Questions?

Check the docs or examine the source code:
- Source: `src/graphs/characterize/`
- Tests: `tests/test_graph_partitioner*.py`
