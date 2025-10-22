# Examples - Capability Demonstrations

This directory contains demonstrations that showcase the capabilities of the graph characterization and partitioning system.

## Purpose

**Examples answer:** "How do I use this system?"

These demos show:
- Basic usage patterns
- End-to-end workflows
- Visualization techniques
- Model comparison methods

## Examples vs Tests/Validation

| Aspect | Examples (`./examples/`) | Tests (`./tests/`) | Validation (`./validation/`) |
|--------|--------------------------|-------------------|------------------------------|
| **Purpose** | Show how to use | Verify correctness | Check accuracy |
| **Question** | "How do I...?" | "Does it work?" | "Is it accurate?" |
| **User** | End users | Developers | Researchers |
| **Speed** | Interactive | Fast (<1s) | Slow (seconds) |

## Available Examples

### 1. Quick Start (`quick_start_partitioner.py`)
**30-second introduction** to graph partitioning and analysis.

```bash
python examples/quick_start_partitioner.py
```

**What it shows:**
- Load a model (ResNet-18)
- Trace with PyTorch FX
- Partition into subgraphs
- Analyze concurrency
- View top compute operations

**Best for:** First-time users

---

### 2. Fusion Comparison (`demo_fusion_comparison.py`)
Compare **unfused vs fused** partitioning to see memory reduction benefits.

```bash
python examples/demo_fusion_comparison.py

# Test on different models
python examples/demo_fusion_comparison.py --model mobilenet_v2
```

**What it shows:**
- Fusion pattern detection (Conv+BN+ReLU)
- Memory traffic reduction (20-42%)
- Kernel launch reduction (1.9-2.1×)
- Per-subgraph fusion benefits

**Best for:** Understanding fusion impact

---

### 3. New Performance Model (`demo_new_performance_model.py`)
Demonstrate the **Phase 2 hardware mapping** system.

```bash
python examples/demo_new_performance_model.py
```

**What it shows:**
- Realistic hardware utilization modeling
- GPU SM allocation
- Memory bandwidth constraints
- Latency estimation across precisions

**Best for:** Hardware-aware optimization

---

### 4. Model Comparison (`compare_models.py`)
Compare **multiple models** side-by-side.

```bash
python examples/compare_models.py --models resnet18 mobilenet_v2 efficientnet_b0
```

**What it shows:**
- FLOPs comparison
- Memory footprint
- Arithmetic intensity
- Parallelism characteristics
- Concurrency metrics

**Best for:** Architecture selection

---

### 5. Partitioning Visualization (`visualize_partitioning.py`)
Generate **visual diagrams** of partition structure.

```bash
python examples/visualize_partitioning.py --model resnet18 --output graph.pdf
```

**What it shows:**
- Subgraph dependency graph
- Critical path highlighting
- Bottleneck visualization
- Parallelism opportunities

**Best for:** Understanding model structure

---

## Quick Start

### First Time User
```bash
# 1. Quick intro (30 seconds)
python examples/quick_start_partitioner.py

# 2. See fusion benefits
python examples/demo_fusion_comparison.py

# 3. Compare models
python examples/compare_models.py --models resnet18 mobilenet_v2
```

### Understanding Your Model
```bash
# 1. Partition and analyze
python examples/quick_start_partitioner.py

# 2. Visualize structure
python examples/visualize_partitioning.py --model <your_model>

# 3. Compare to baselines
python examples/compare_models.py --models <your_model> resnet18
```

### Hardware Optimization
```bash
# 1. See hardware mapping
python examples/demo_new_performance_model.py

# 2. Compare across hardware
cd ../validation/hardware
python test_all_hardware.py
```

## Code Patterns

### Load and Partition Any Model
```python
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner

# Load model
model = ...  # Your PyTorch model
model.eval()

# Trace with FX
traced = symbolic_trace(model)
ShapeProp(traced).propagate(torch.randn(1, 3, 224, 224))

# Partition with fusion
partitioner = FusionBasedPartitioner()
report = partitioner.partition_graph(traced)

# Access results
print(f"Total FLOPs: {report.total_flops / 1e9:.2f} G")
print(f"Fused subgraphs: {len(report.fused_subgraphs)}")
print(f"Memory reduction: {report.memory_reduction_percent:.1f}%")
```

### Compare Two Models
```python
def analyze_model(model, name):
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(torch.randn(1, 3, 224, 224))

    partitioner = FusionBasedPartitioner()
    report = partitioner.partition_graph(traced)

    return {
        'name': name,
        'flops_g': report.total_flops / 1e9,
        'subgraphs': len(report.fused_subgraphs),
        'memory_mb': report.total_memory_traffic / 1e6,
    }

# Compare
resnet = analyze_model(models.resnet18(), "ResNet-18")
mobilenet = analyze_model(models.mobilenet_v2(), "MobileNet-V2")

print(f"{resnet['name']}: {resnet['flops_g']:.2f} GFLOPs")
print(f"{mobilenet['name']}: {mobilenet['flops_g']:.2f} GFLOPs")
```

### Find Bottlenecks
```python
# After partitioning...
for subgraph in report.fused_subgraphs:
    ai = subgraph.total_flops / subgraph.total_memory_traffic
    bottleneck = "compute" if ai > 10 else "memory"
    print(f"{subgraph.subgraph_id}: {bottleneck}-bound (AI={ai:.1f})")
```

## Requirements

All examples require:
- Python 3.8+
- PyTorch
- torchvision

Install:
```bash
pip install torch torchvision
```

Optional (for visualization):
```bash
pip install matplotlib networkx graphviz
```

## Common Issues

**Import errors:**
- Run from repo root: `python examples/quick_start_partitioner.py`
- Or set PYTHONPATH: `export PYTHONPATH=/path/to/repo`

**Model not found:**
- Use torchvision models: `models.resnet18()`
- For custom models, provide model object (not name string)

**FX tracing fails:**
- Some models have dynamic control flow
- Try: `symbolic_trace(model, concrete_args={...})`
- See PyTorch FX docs for workarounds

**Slow execution:**
- Examples should run in seconds
- If slower, check model size (VIT, large ResNets take longer)
- For batch processing, see `validation/` tests

## Next Steps

After exploring examples:

1. **Run validation tests** to see accuracy results:
   ```bash
   python validation/hardware/test_all_hardware.py
   python validation/estimators/test_resnet_family.py
   ```

2. **Read documentation** for deeper understanding:
   - `../docs/GETTING_STARTED.md` - Getting started guide
   - `../docs/graph_partitioner_tutorial.md` - Detailed tutorials
   - `../docs/realistic_performance_modeling_plan.md` - Architecture

3. **Use CLI tools** for production workflows:
   ```bash
   ./cli/partitioner.py --model resnet18 --output results.json
   ./cli/profile_graph.py --model mobilenet_v2
   ```

4. **Write your own examples** - use these as templates!

## Contributing Examples

To add a new example:

1. Create `demo_<feature>.py` or `<task>_example.py`
2. Include docstring explaining what it demonstrates
3. Add argparse for customization options
4. Keep runtime under 30 seconds
5. Add to this README with description

Example template:
```python
#!/usr/bin/env python
"""
Demonstration of <feature>

Shows how to <accomplish task> using <components>.
"""

import argparse
import torch
from torch.fx import symbolic_trace

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.graphs.characterize.<module> import <Component>


def main():
    parser = argparse.ArgumentParser(description="Demo of <feature>")
    parser.add_argument('--model', default='resnet18', help="Model to use")
    args = parser.parse_args()

    # Demo code here
    print("Demonstrating <feature>...")


if __name__ == '__main__':
    main()
```

## Documentation

See also:
- `../tests/README.md` - Unit tests
- `../validation/README.md` - Accuracy validation
- `../cli/README.md` - Command-line tools
- `../docs/` - Full documentation
