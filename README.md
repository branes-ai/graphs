# Graphs: Neural Network Performance Analysis and Characterization

Collection of compute graph definitions and tools for realistic performance modeling.

This repository provides tools to analyze PyTorch models and understand how they map to hardware resources, enabling realistic performance predictions instead of overly-optimistic peak theoretical estimates.

## Quick Start

```bash
cd /home/stillwater/dev/branes/clones/graphs
source ~/venv/p311/bin/activate

# Analyze a model
python examples/quick_start_partitioner.py

# Compare multiple models
python examples/compare_models.py

# Validate partitioner
python tests/test_graph_partitioner_general.py resnet18
```

**New to graph partitioning?** Start with the [Getting Started Guide](docs/GETTING_STARTED.md)

## Key Features

### Phase 1: Graph Partitioning and Concurrency Analysis ✅ (Complete)

- **GraphPartitioner**: Decomposes PyTorch FX graphs into analyzable subgraphs with:
  - FLOPs/MACs computation
  - Memory traffic analysis
  - Arithmetic intensity classification
  - Parallelism detection (batch, channel, spatial dimensions)
  - Depthwise convolution support

- **ConcurrencyAnalyzer**: Analyzes available parallelism at multiple levels:
  - Graph-level: Which operations can run in parallel (stages)
  - Subgraph-level: Thread-level parallelism within each operation
  - Critical path analysis: Minimum latency bounds

- **Validation Framework**: Generalized validation across architectures:
  - Universal checks (apply to all models)
  - Architecture-specific validation (expected ranges)
  - Pre-configured profiles for ResNet, MobileNet, EfficientNet, VGG

### Phase 2: Hardware Mapping (In Progress)

- CPU/GPU/TPU/KPU resource models
- Realistic utilization estimates
- SM/core/tile allocation

### Phase 3: Performance/Energy/Memory Estimation (Future)

- Roofline-based latency modeling
- Component-level energy breakdown
- Peak memory estimation

## Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Start here! 5-minute intro to graph partitioning
- **[Tutorial](docs/graph_partitioner_tutorial.md)** - Hands-on tutorials from basic to advanced
- **[Validation Framework](docs/graph_partitioner_validation.md)** - How validation works
- **[Implementation Plan](docs/realistic_performance_modeling_plan.md)** - Phase 1-3 roadmap
- **[Examples](examples/)** - Quick start scripts and model comparison tools

## Example Output

```bash
$ python examples/quick_start_partitioner.py

Graph Partition Summary
=======================
Total subgraphs: 60
Total FLOPs: 4.49 G
Average arithmetic intensity: 31.06 FLOPs/byte

Concurrency Analysis
====================
Graph-level Parallelism:
  Max parallel ops per stage: 12
  Critical path length: 9 sequential operations

Parallelism Potential:
  - Graph-level: 12× (max ops that can run simultaneously)
  - Thread-level: 110,267 threads avg

Key Insight: With batch=1, ResNet-18 speedup limited to 12× by graph structure
→ Need batch≥10 to saturate H100 GPU (132 SMs)
```

## Repository Structure

Repo structure for synthetic workloads and FX graph-based workload characterization and estimation

```txt
graphs/
├── src/
│   └── graphs/
│       ├── models/                        # Synthetic DNN definitions
│       │   ├── mlp.py                    # Parameterized MLP
│       │   ├── resnet_block.py           # Synthetic ResNet block
│       │   └── conv2d_stack.py           # Stacked Conv2D layers
│       │
│       └── characterize/                  # NEW: Phase 1 implementation ✅
│           ├── graph_structures.py       # Data structures (SubgraphDescriptor, etc.)
│           ├── graph_partitioner.py      # Graph partitioning engine
│           ├── concurrency_analyzer.py   # Concurrency analysis
│           ├── arch_profiles.py          # Hardware architecture profiles
│           ├── tiling.py                 # Tiling strategies (CPU/GPU/TPU/KPU)
│           ├── fused_ops.py              # Fused operation registry
│           ├── introspect.py             # Operator introspection
│           ├── walker.py                 # FX graph walker
│           ├── sweep.py                  # Sweep harness
│           └── visualize.py              # Visualization utilities
│
├── examples/                              # NEW: Hands-on examples ✅
│   ├── README.md                         # Examples overview
│   ├── quick_start_partitioner.py        # Quick start example
│   └── compare_models.py                 # Model comparison tool
│
├── tests/                                 # NEW: Validation framework ✅
│   ├── test_graph_partitioner.py         # ResNet-18 validation
│   └── test_graph_partitioner_general.py # Generalized validation
│
├── docs/                                  # NEW: Comprehensive documentation ✅
│   ├── GETTING_STARTED.md                # 5-minute quick start guide
│   ├── graph_partitioner_tutorial.md     # Hands-on tutorials
│   ├── graph_partitioner_validation.md   # Validation framework docs
│   └── realistic_performance_modeling_plan.md  # Phase 1-3 roadmap
│
├── scripts/
│   └── run_characterization.py           # CLI entry point
│
├── README.md
└── requirements.txt
```
