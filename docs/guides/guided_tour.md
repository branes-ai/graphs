# Graph Characterization Framework - Guided Tour

**Welcome!** This guide will take you from absolute beginner to advanced user in a structured, progressive way.

Too many docs? Start here. This is your roadmap through the framework.

---

## How to Use This Guide

This guide is organized into **5 progressive levels**, from complete beginner to expert. Each level:
- Has a clear time estimate
- Lists what you'll learn
- Provides hands-on exercises
- Points to the next level

**Start at Level 0** and progress at your own pace. Skip levels if you're already familiar with the basics.

---

## Level 0: First Contact (5 minutes)

**Goal**: Run your first analysis and understand what this project does.

### What You'll Learn
- What this framework analyzes
- How to run your first command
- What the output means

### Prerequisites
- Python 3.8+ installed
- PyTorch and torchvision installed
- Cloned this repository

### Quick Start

**Step 1: Discover available models**
```bash
cd /path/to/graphs
./cli/discover_models.py
```

You'll see a list of available models. We'll use ResNet-18.

**Step 2: Run your first analysis**
```bash
./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-Nano
```

**Step 3: Understand the output**

You'll see something like:
```
EXECUTIVE SUMMARY
─────────────────────────────────────────────────────
Model:                   ResNet-18
Hardware:                Jetson-Orin-Nano-8GB
Precision:               FP32
Batch Size:              1

Performance:             12.5 ms latency, 80 fps
Energy:                  125 mJ total (125 mJ/inference)
Memory:                  Peak 55.0 MB
```

**What this means:**
- **Latency**: How long one inference takes (12.5 milliseconds)
- **FPS**: Frames/inferences per second (80)
- **Energy**: Power consumption per inference (125 millijoules)
- **Memory**: Peak memory usage (55 MB)

**Congratulations!** You just analyzed a neural network's performance on Jetson Orin Nano edge hardware.

### What's Next?
- **Level 1** if you want to understand what just happened
- **Skip to Level 2** if you know the basics and want practical workflows

---

## Level 1: Understanding the Basics (30-60 minutes)

**Goal**: Understand core concepts and run basic analyses on your own.

### What You'll Learn
- What graph partitioning means
- How to read bottleneck classifications
- How to visualize models
- Key metrics (FLOPs, arithmetic intensity, memory)

### Step 1: Core Concepts (10 minutes)

**Read this section:** `docs/getting_started.md` (first 100 lines)

Key concepts to understand:
- **Subgraph**: A fused operation (e.g., Conv+BatchNorm+ReLU)
- **Arithmetic Intensity (AI)**: FLOPs per byte (tells you if compute or memory bound)
- **Bottleneck**: Is the operation limited by compute or memory?
- **Graph-level parallelism**: How many operations can run simultaneously

### Step 2: Explore a Model (10 minutes)

**List available hardware:**
```bash
./cli/list_hardware_mappers.py
```

**Profile a simple model:**
```bash
./cli/profile_graph.py --model resnet18
```

**Look at the output:**
- Total FLOPs (computational cost)
- Total memory (data movement)
- Bottleneck distribution (how many ops are compute vs memory bound)

### Step 3: Visualize with Mermaid (10 minutes)

**Generate a visual report:**
```bash
./cli/analyze_comprehensive.py \
    --model mobilenet_v2 \
    --hardware Jetson-Orin-Nano \
    --output mobilenet_analysis.md \
    --include-diagrams
```

**Open `mobilenet_analysis.md` in GitHub or VSCode** to see:
- Graph structure with color-coded bottlenecks
- Performance metrics
- Recommendations

**Colors mean:**
- Green: Compute-bound (good for GPUs)
- Red: Memory-bound (bandwidth limited)
- Orange: Balanced

### Step 4: Compare Two Models (10 minutes)

**Compare ResNet-18 on multiple hardware:**
```bash
./cli/compare_models.py resnet18
```

**Or compare multiple models using batch analysis:**
```bash
./cli/analyze_batch.py --models resnet18 mobilenet_v2 --hardware Jetson-Orin-Nano --batch-size 1
```

**Notice:**
- Different FLOP counts
- Different memory requirements
- Different bottleneck distributions

**Exercise**: Which model would run faster on an edge device with limited memory bandwidth?

### What's Next?
- **Level 2** for practical workflows (batch analysis, hardware selection)
- Review `docs/graph_partitioner_tutorial.md` for deeper understanding

---

## Level 2: Common Workflows (1-3 hours)

**Goal**: Execute real-world analysis workflows for model deployment decisions.

### What You'll Learn
- How to find optimal batch size
- How to choose hardware for a model
- How to optimize for energy efficiency
- How to analyze multiple configurations

### Workflow 1: Batch Size Optimization (20 minutes)

**Problem**: What batch size gives best throughput without excessive latency?

```bash
./cli/analyze_batch.py \
    --model resnet18 \
    --hardware Jetson-Orin-Nano \
    --batch-size 1 2 4 8 16 32 \
    --output batch_sweep.csv
```

**Read the insights section** in the output. It tells you:
- Best batch for throughput
- Best batch for energy efficiency
- Best batch for latency
- Memory growth pattern

**Exercise**: Run the same analysis for `mobilenet_v2`. How do the batch size recommendations differ?

### Workflow 2: Hardware Selection (20 minutes)

**Problem**: Which hardware is best for my model?

```bash
./cli/analyze_batch.py \
    --model resnet50 \
    --hardware Jetson-Orin-AGX Jetson-Orin-Nano KPU-T256 \
    --batch-size 1 8 16 \
    --output hardware_comparison.csv
```

**Look at the CSV output:**
- Compare latency across hardware
- Compare energy per inference
- Compare throughput

**Exercise**: For an edge deployment with <10W power budget, which hardware would you choose?

### Workflow 3: Energy Analysis (20 minutes)

**Problem**: How can I minimize energy consumption?

```bash
./cli/analyze_comprehensive.py \
    --model mobilenet_v2 \
    --hardware Jetson-Orin-Nano \
    --output energy_analysis.json \
    --precision fp16
```

**Look at the energy breakdown:**
- Compute energy
- Memory energy
- Static/leakage energy

**Key insight**: Static energy dominates at small batch sizes!

**Exercise**: Try different precisions (fp32, fp16) and batch sizes. What gives best energy efficiency?

### Workflow 4: Model Comparison (20 minutes)

**Problem**: Which model architecture is best for my use case?

```bash
./cli/analyze_batch.py \
    --models resnet18 mobilenet_v2 efficientnet_b0 \
    --hardware Jetson-Orin-AGX \
    --batch-size 1 8 \
    --output model_comparison.csv
```

**Compare:**
- Latency (for real-time applications)
- Energy (for battery-powered devices)
- Memory (for resource-constrained devices)

### Workflow 5: Specialized Comparisons (20 minutes)

**Automotive ADAS platforms:**
```bash
python cli/compare_automotive_adas.py
```

**Edge AI platforms:**
```bash
python cli/compare_edge_ai_platforms.py
```

**Datacenter CPUs:**
```bash
python cli/compare_datacenter_cpus.py
```

**What you'll learn:**
- Which hardware is optimized for which workload
- Power vs performance tradeoffs
- Cost vs capability analysis

### Hands-On Exercise: Complete Deployment Analysis (30 minutes)

**Scenario**: Deploy ResNet-50 on Jetson Orin AGX for real-time video (30 FPS requirement)

**Step 1**: Analyze baseline performance
```bash
./cli/analyze_comprehensive.py \
    --model resnet50 \
    --hardware Jetson-Orin-AGX \
    --output baseline.json
```

**Step 2**: Find batch size that meets 30 FPS
```bash
./cli/analyze_batch.py \
    --model resnet50 \
    --hardware Jetson-Orin-AGX \
    --batch-size 1 2 4 8 \
    --output batch_analysis.csv
```

**Step 3**: Try FP16 precision for speedup
```bash
./cli/analyze_comprehensive.py \
    --model resnet50 \
    --hardware Jetson-Orin-AGX \
    --precision fp16 \
    --batch-size 4 \
    --output fp16_analysis.json
```

**Questions to answer:**
- Can you meet 30 FPS requirement?
- What batch size maximizes throughput while meeting latency requirement?
- Does FP16 help? By how much?
- What's the energy consumption?

### What's Next?
- **Level 3** for custom models and Python API
- **Level 2 Deep Dive** (see "Further Reading" below)

### Further Reading
- `cli/README.md` - Complete tool reference
- `docs/MERMAID_QUICK_START.md` - Visualization guide
- `cli/docs/` - Individual tool guides

---

## Level 3: Advanced Usage (Half day)

**Goal**: Analyze custom models, use Python API, and understand advanced features.

### What You'll Learn
- How to analyze custom PyTorch models
- How to use the Python API programmatically
- How to create custom analysis workflows
- How to interpret detailed profiling data

### Workflow 1: Analyze Custom Models (30 minutes)

**Read**: `docs/graph_partitioner_tutorial.md` Tutorial 1-2

**Exercise**: Analyze your own model

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.reporting import ReportGenerator

# Define your custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Analyze it
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

**What to look for:**
- Which layers are compute-bound vs memory-bound?
- What's the peak memory usage?
- Where are the bottlenecks?

### Workflow 2: Python API Mastery (45 minutes)

**Read**: `docs/unified_framework_api.md` (Quick Start and UnifiedAnalyzer sections)

**Exercise 1: Batch size sweep**
```python
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.reporting import ReportGenerator

analyzer = UnifiedAnalyzer(verbose=False)
generator = ReportGenerator()

# Sweep batch sizes
results = []
for batch_size in [1, 2, 4, 8, 16, 32]:
    result = analyzer.analyze_model('resnet18', 'Jetson-Orin-Nano', batch_size=batch_size)
    results.append(result)

# Generate comparison report
comparison = generator.generate_comparison_report(results, format='csv')
with open('batch_comparison.csv', 'w') as f:
    f.write(comparison)

# Find best configurations
best_throughput = max(results, key=lambda r: r.throughput_fps)
best_energy = min(results, key=lambda r: r.energy_per_inference_mj)

print(f"Best throughput: batch {best_throughput.batch_size} "
      f"({best_throughput.throughput_fps:.1f} fps)")
print(f"Best energy: batch {best_energy.batch_size} "
      f"({best_energy.energy_per_inference_mj:.1f} mJ/inf)")
```

**Exercise 2: Automated hardware selection**
```python
def find_best_hardware(model_name, batch_size, metric='energy'):
    """Find best hardware for a model based on a metric."""

    hardware_targets = [
        'Jetson-Orin-AGX', 'Jetson-Orin-Nano', 'Coral-Edge-TPU', 'KPU-T256'
    ]

    analyzer = UnifiedAnalyzer(verbose=False)
    results = {}

    for hw in hardware_targets:
        try:
            result = analyzer.analyze_model(model_name, hw, batch_size=batch_size)
            results[hw] = result
        except Exception as e:
            print(f"Skipping {hw}: {e}")

    # Find best by metric
    if metric == 'energy':
        best = min(results.items(), key=lambda x: x[1].energy_per_inference_mj)
    elif metric == 'latency':
        best = min(results.items(), key=lambda x: x[1].total_latency_ms)
    elif metric == 'throughput':
        best = max(results.items(), key=lambda x: x[1].throughput_fps)

    return best[0], best[1]

# Use it
best_hw, result = find_best_hardware('mobilenet_v2', batch_size=8, metric='energy')
print(f"Best hardware: {best_hw}")
print(f"Energy: {result.energy_per_inference_mj:.1f} mJ")
print(f"Latency: {result.total_latency_ms:.2f} ms")
```

### Workflow 3: Detailed Profiling (30 minutes)

**Interactive graph exploration:**
```bash
# Explore graph structure
./cli/graph_explorer.py --model resnet18

# Visualize specific sections
./cli/graph_explorer.py --model resnet18 --start 10 --end 30

# Focus on critical operations
./cli/graph_explorer.py --model resnet18 --around 25 --context 10
```

**Partition analysis:**
```bash
# Compare partitioning strategies
./cli/partition_analyzer.py --model resnet18 --strategy all --compare

# Visualize fusion benefits
./cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize
```

### Workflow 4: Optimization Workflow (45 minutes)

**Complete optimization workflow for a model:**

**Step 1: Baseline analysis**
```bash
./cli/analyze_comprehensive.py \
    --model efficientnet_b0 \
    --hardware Jetson-Orin-Nano \
    --output baseline.md \
    --include-diagrams
```

**Step 2: Identify bottlenecks**
- Look at the diagram colors (red = memory bottleneck)
- Look at the performance breakdown
- Check the recommendations

**Step 3: Test optimizations**
```bash
# Try FP16 precision
./cli/analyze_comprehensive.py \
    --model efficientnet_b0 \
    --hardware Jetson-Orin-Nano \
    --precision fp16 \
    --output fp16_optimized.json

# Try different batch sizes
./cli/analyze_batch.py \
    --model efficientnet_b0 \
    --hardware Jetson-Orin-Nano \
    --batch-size 1 2 4 8 \
    --precision fp16 \
    --output batch_optimization.csv
```

**Step 4: Compare results**
```python
import json

with open('baseline.json') as f:
    baseline = json.load(f)
with open('fp16_optimized.json') as f:
    optimized = json.load(f)

baseline_lat = baseline['derived_metrics']['latency_ms']
opt_lat = optimized['derived_metrics']['latency_ms']
speedup = baseline_lat / opt_lat

print(f"Speedup: {speedup:.2f}×")
print(f"Energy reduction: {baseline['derived_metrics']['total_energy_mj'] / optimized['derived_metrics']['total_energy_mj']:.2f}×")
```

### Hands-On Project: Custom Deployment Analysis

**Build a script that:**
1. Takes a model name and deployment constraints (latency budget, power budget)
2. Tests multiple hardware options
3. Tests multiple batch sizes and precisions
4. Generates a recommendation report

**Template:**
```python
#!/usr/bin/env python3
"""
Deployment analyzer: Find best configuration for deployment constraints.
"""
import argparse
from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision

def analyze_deployment(model_name, max_latency_ms, max_power_w):
    """Find best hardware/config for deployment constraints."""

    # Define search space
    hardware_options = ['Jetson-Orin-AGX', 'Jetson-Orin-Nano', 'Coral-Edge-TPU', 'KPU-T256']
    batch_sizes = [1, 2, 4, 8]
    precisions = [Precision.FP32, Precision.FP16]

    analyzer = UnifiedAnalyzer(verbose=False)
    candidates = []

    # Search space
    for hw in hardware_options:
        for bs in batch_sizes:
            for prec in precisions:
                try:
                    result = analyzer.analyze_model(model_name, hw,
                                                   batch_size=bs,
                                                   precision=prec)

                    # Check constraints
                    latency_ok = result.total_latency_ms <= max_latency_ms
                    # Note: Power analysis would need TDP from hardware model

                    if latency_ok:
                        candidates.append((hw, bs, prec, result))

                except Exception as e:
                    continue

    # Find best by throughput
    if candidates:
        best = max(candidates, key=lambda x: x[3].throughput_fps)
        print(f"\nBest configuration:")
        print(f"  Hardware: {best[0]}")
        print(f"  Batch size: {best[1]}")
        print(f"  Precision: {best[2].name}")
        print(f"  Throughput: {best[3].throughput_fps:.1f} fps")
        print(f"  Latency: {best[3].total_latency_ms:.2f} ms")
        print(f"  Energy: {best[3].energy_per_inference_mj:.1f} mJ")
    else:
        print("No configurations meet constraints!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-latency-ms', type=float, required=True)
    parser.add_argument('--max-power-w', type=float, required=True)
    args = parser.parse_args()

    analyze_deployment(args.model, args.max_latency_ms, args.max_power_w)
```

### What's Next?
- **Level 4** for expert topics (contributing, internals, new hardware)
- Deep dive into specific subsystems (see "Further Reading")

### Further Reading
- `docs/unified_framework_api.md` - Complete Python API reference
- `docs/graph_partitioner_tutorial.md` - Comprehensive tutorial with exercises
- `docs/visualization_guide.md` - Advanced visualization techniques
- `tests/integration/test_unified_workflows.py` - Example usage patterns

---

## Level 4: Expert Topics (Ongoing)

**Goal**: Contribute to the project, add new hardware, understand internals.

### What You'll Learn
- Internal architecture and design
- How to add new hardware targets
- How to extend analysis capabilities
- How to contribute improvements

### Topic 1: Understanding the Architecture (1-2 hours)

**Read these in order:**
1. `docs/characterization-architecture.md` - Overall architecture
2. `CLAUDE.md` - Project structure and conventions
3. `docs/realistic_performance_modeling_plan.md` - Design philosophy

**Key subsystems:**
- **Core (Intermediate Representation)**: `src/graphs/core/structures.py` (also confidence tracking)
- **Frontends**: `src/graphs/frontends/` (PyTorch Dynamo/FX tracing)
- **Transform**: `src/graphs/transform/` (partitioning, fusion, tiling)
- **Estimation**: `src/graphs/estimation/` (roofline, energy, memory analyzers)
- **Hardware**: `src/graphs/hardware/` (resource models, mappers)
- **Calibration**: `src/graphs/calibration/` (hardware calibration profiles)
- **Benchmarks**: `src/graphs/benchmarks/` (calibration benchmarks)

**Exercise**: Draw a diagram of how data flows from a PyTorch model to final performance estimates.

### Topic 2: Adding New Hardware (2-3 hours)

**Read**: `docs/hardware_characterization_2025-10.md`

**Example: Add a custom accelerator**

```python
# src/graphs/hardware/resource_model.py

# Step 1: Define hardware characteristics
custom_accel = HardwareResourceModel(
    name='MyAccelerator-X1',
    arch_type='Accelerator',
    peak_tflops_fp32=50.0,
    memory_bandwidth_gb_s=300.0,
    l2_cache_mb=8.0,
    precision_profiles={
        Precision.FP32: 1.0,
        Precision.FP16: 2.0,
        Precision.INT8: 4.0
    },
    tdp_watts=25.0
)

# Step 2: Create a mapper
# src/graphs/hardware/mappers/custom_accel.py

from .base import HardwareMapper

class CustomAccelMapper(HardwareMapper):
    """Maps operations to custom accelerator resources."""

    def map_to_hardware(self, subgraph):
        """Map subgraph to accelerator units."""
        # Implement your mapping logic
        pass
```

**Validation:**
```bash
python validation/hardware/test_custom_hardware.py
```

### Topic 3: Advanced Analysis (2-3 hours)

**Extend the analysis framework:**

**Example: Add a new analyzer**
```python
# src/graphs/estimation/cache_analyzer.py

class CacheAnalyzer:
    """Analyzes cache hit rates and locality."""

    def __init__(self, l1_size_kb, l2_size_kb):
        self.l1_size_kb = l1_size_kb
        self.l2_size_kb = l2_size_kb

    def analyze(self, subgraphs, partition_report):
        """Estimate cache behavior."""
        # Your analysis logic
        return CacheReport(...)
```

**Integrate into UnifiedAnalyzer:**
```python
# Extend AnalysisConfig
config = AnalysisConfig(
    run_roofline=True,
    run_energy=True,
    run_memory=True,
    run_cache=True  # Your new analyzer
)
```

**Working with Confidence Tracking:**

The framework tracks confidence levels for all estimates, helping you understand
how reliable each result is:

```python
from graphs.core.confidence import ConfidenceLevel, EstimationConfidence

# Check confidence on analysis results
result = analyzer.analyze_model('resnet18', 'Jetson-Orin-Nano')

# Access confidence from descriptors
for lat_desc in result.roofline_report.latency_descriptors:
    conf = lat_desc.confidence
    print(f"Subgraph {lat_desc.subgraph_name}: "
          f"{conf.level.value} confidence ({conf.score*100:.0f}%)")
    if conf.calibration_id:
        print(f"  Calibration: {conf.calibration_id}")

# Confidence levels:
# - CALIBRATED: Based on real benchmark measurements (highest confidence)
# - INTERPOLATED: Derived from calibrated data points
# - THEORETICAL: Based on vendor specs/theoretical peaks
# - UNKNOWN: Confidence not tracked (default for backward compatibility)
```

### Topic 4: Contributing (Ongoing)

**Read**:
- `docs/CHANGELOG_MANAGEMENT.md` - How to document changes
- `docs/sessions/README.md` - Session documentation format

**Contribution workflow:**

1. **Find an issue or propose enhancement**
2. **Create a session doc**: `docs/sessions/YYYY-MM-DD_topic.md`
3. **Implement and test**
4. **Update documentation**
5. **Submit pull request**

**Testing your changes:**
```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Validation tests
python validation/hardware/test_all_hardware.py
```

### Expert Projects

**Project Ideas:**
1. **Add support for new model architecture** (Transformers, LLMs, etc.)
2. **Implement new fusion patterns** for better partitioning
3. **Add interactive visualization** (HTML/D3.js export)
4. **Extend energy model** with component-level breakdown
5. **Add cost analysis** ($ per inference, TCO calculations)
6. **Implement auto-tuning** that searches optimization space automatically

### What's Next?

**You're now an expert!** Consider:
- Contributing to the project
- Using this in your research
- Extending it for your specific domain
- Teaching others using this guide

### Further Reading (Deep Dives)
- `docs/sessions/` - Detailed implementation sessions
- `docs/phase*.md` - Phase-specific design documents
- `docs/bugs/` - Bug investigations and fixes
- Source code in `src/graphs/` - Implementation details

---

## Quick Reference: Where to Find What

### "I want to..."

| Goal | Start Here | Time |
|------|------------|------|
| Run my first analysis | Level 0 | 5 min |
| Understand basic concepts | Level 1 | 30 min |
| Find optimal batch size | Level 2, Workflow 1 | 20 min |
| Choose hardware for deployment | Level 2, Workflow 2 | 20 min |
| Analyze energy consumption | Level 2, Workflow 3 | 20 min |
| Analyze my custom model | Level 3, Workflow 1 | 30 min |
| Use Python API | Level 3, Workflow 2 | 45 min |
| Add new hardware | Level 4, Topic 2 | 2-3 hours |
| Contribute to the project | Level 4, Topic 4 | Ongoing |

### Common Commands Cheat Sheet

```bash
# Discover models
./cli/discover_models.py

# List hardware
./cli/list_hardware_mappers.py

# Quick analysis
./cli/analyze_comprehensive.py --model MODEL --hardware HW

# Batch size sweep
./cli/analyze_batch.py --model MODEL --hardware HW --batch-size 1 4 8 16

# Visual report with diagrams
./cli/analyze_comprehensive.py --model MODEL --hardware HW \
    --output report.md --include-diagrams

# Compare models
./cli/analyze_batch.py --models MODEL1 MODEL2 MODEL3 --hardware HW

# Compare hardware
./cli/analyze_batch.py --model MODEL --hardware HW1 HW2 HW3

# Explore graph structure
./cli/graph_explorer.py --model MODEL
```

### Key Documentation Files

| Topic | Document | Level |
|-------|----------|-------|
| **Getting Started** | `docs/getting_started.md` | Beginner |
| **Quick Visualization** | `docs/MERMAID_QUICK_START.md` | Beginner |
| **Tutorial** | `docs/graph_partitioner_tutorial.md` | Intermediate |
| **Python API** | `docs/unified_framework_api.md` | Advanced |
| **CLI Tools** | `cli/README.md` | All levels |
| **Migration Guide** | `docs/migration_guide_phase4_2.md` | Advanced |
| **Architecture** | `docs/characterization-architecture.md` | Expert |
| **Hardware** | `docs/hardware_characterization_2025-10.md` | Expert |

---

## Troubleshooting

### "I'm lost, where should I start?"
- Start at **Level 0** (5 minutes)
- If you already know PyTorch and ML, skip to **Level 1**
- If you need to make deployment decisions now, go to **Level 2**

### "The tools aren't working"
```bash
# Check Python version (need 3.8+)
python --version

# Check dependencies
pip install torch torchvision

# Install in development mode
cd /path/to/graphs
pip install -e .

# Try a simple command
./cli/discover_models.py
```

### "Too much output, can't find what I need"
Use the `--quiet` flag:
```bash
./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-Nano --quiet
```

### "I want to go deeper on a specific topic"
Check the "Further Reading" section at the end of each level for deep-dive documents.

---

## Learning Pathways

### Path A: ML Engineer (Deploy Models)
1. Level 0 (understand basics)
2. Level 2 (practical workflows)
3. Level 3, Workflows 1-2 (custom models, Python API)
4. Done! You can now make informed deployment decisions.

**Time investment**: 3-5 hours

### Path B: Research/Academia (Understand Performance)
1. Level 0 (basics)
2. Level 1 (concepts)
3. Level 2 (all workflows)
4. Level 3 (all workflows)
5. Level 4, Topics 1 & 3 (architecture, advanced analysis)

**Time investment**: 1-2 days

### Path C: Contributor/Developer
1. All levels in order
2. Focus on Level 4 (all topics)
3. Study implementation in `src/graphs/`
4. Review `docs/sessions/` for implementation patterns

**Time investment**: 1 week

### Path D: Hardware Architect
1. Level 0-1 (basics)
2. Level 2, Workflows 2 & 5 (hardware selection, specialized comparisons)
3. Level 4, Topic 2 (adding hardware)
4. Study `src/graphs/hardware/` implementation

**Time investment**: 1-2 days

---

## Tips for Success

1. **Don't skip levels** - Each builds on the previous
2. **Do the exercises** - Hands-on practice cements learning
3. **Experiment freely** - The tools won't break anything
4. **Use visualization** - `--include-diagrams` makes everything clearer
5. **Start small** - Use ResNet-18 or MobileNet-V2 for learning
6. **Ask questions** - Check existing docs before asking
7. **Document as you learn** - Take notes on what worked

---

## What This Framework Does

At its core, this framework:
1. **Analyzes** PyTorch models to understand computational characteristics
2. **Partitions** them into executable subgraphs (fused operations)
3. **Maps** those subgraphs to specific hardware architectures
4. **Estimates** performance (latency, throughput, energy, memory)
5. **Recommends** optimizations based on bottleneck analysis

**Why this matters:**
- Make informed hardware selection decisions
- Optimize models before deployment
- Understand energy costs (critical for edge/mobile)
- Predict real-world performance (not just theoretical peaks)
- Compare models fairly across different hardware

---

## Success Stories

**What users have done with this framework:**
- Selected optimal hardware for edge AI deployments
- Optimized batch sizes for 3× energy efficiency improvements
- Identified memory bottlenecks before deployment
- Compared model architectures for specific hardware constraints
- Made data-driven decisions on FP16 vs FP32 precision

---

## Final Notes

This is a **living guide**. As the framework evolves, so will this tour.

**Current version**: Milestone 1 (Foundation Consolidation)
**Last updated**: 2026-01-17

**Your feedback matters!** If something in this guide is:
- Unclear
- Missing
- Wrong
- Could be improved

Please let us know.

**Happy analyzing!**
