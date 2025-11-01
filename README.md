# Graphs: Neural Network Performance Analysis and Characterization

[![CI](https://github.com/branes-ai/graphs/workflows/CI/badge.svg)](https://github.com/branes-ai/graphs/actions/workflows/ci.yml)

Collection of compute graph analysis tools for realistic performance modeling across many different hardware platforms.

This repository provides tools to analyze PyTorch models and understand how they map to hardware resources, enabling realistic performance predictions instead of overly-optimistic peak theoretical estimates.

## Quick Start

```bash
# Discover available hardware platforms
python cli/list_hardware_mappers.py

# Analyze a model on specific hardware
python cli/analyze_graph_mapping.py resnet18 --hardware H100

# Compare multiple hardware targets
python cli/analyze_graph_mapping.py resnet18 --compare "H100,A100,Jetson-Orin-AGX"

# Compare models across hardware
python cli/compare_models.py resnet50 --deployment datacenter

# Profile graph partitioning
python cli/profile_graph.py resnet18

# Test partitioner with examples
python examples/quick_start_partitioner.py
```

**New to the project?** Start with the [CLI Documentation](cli/README.md) or [Getting Started Guide](docs/getting_started.md)

## Key Features

### Phase 1: Graph Partitioning and Concurrency Analysis ✅ (Complete)

- **Graph Partitioning**: Decomposes PyTorch FX graphs into analyzable subgraphs
  - FLOPs/MACs computation with depthwise convolution support
  - Memory traffic analysis and arithmetic intensity classification
  - Fusion-based partitioning for realistic operator grouping
  - Parallelism detection (batch, channel, spatial dimensions)

- **Concurrency Analysis**: Multi-level parallelism analysis
  - Graph-level: Parallel execution stages
  - Subgraph-level: Thread-level parallelism within operations
  - Critical path analysis for minimum latency bounds

### Phase 2: Hardware Mapping ✅ (Complete)

- **32+ Hardware Models** across 5 deployment categories:
  - **Datacenter**: H100, A100, V100, T4, TPU v4, Intel Xeon, AMD EPYC, Ampere
  - **Edge**: Jetson Orin (AGX/Nano), Coral Edge TPU, QRB5165
  - **Automotive**: Jetson Thor, TI TDA4x series (VM/VL/AL/VH)
  - **Accelerators**: Stillwater KPU (T64/T256/T768), Xilinx Vitis AI DPU, CGRA
  - **IP Cores**: CEVA NeuPro, Cadence Vision Q8, Synopsys ARC EV7x

- **Architecture-Specific Mappers**:
  - CPU: Multi-core + SIMD (AVX-512, AMX) with NUMA awareness
  - GPU: SM allocation with sequential/parallel execution modes
  - TPU: Systolic array mapping with tensor core utilization
  - KPU: Heterogeneous tile allocation (INT8/BF16/Matrix tiles)
  - DSP: Vector/tensor unit mapping (Hexagon HVX)
  - DPU/CGRA: Reconfigurable fabric mapping

- **Realistic Performance Modeling**:
  - Microarchitecture-aware (CUDA cores/SM, Tensor Cores, AMX tiles)
  - Leakage-based power modeling (50% idle power for nanoscale chips)
  - Memory bandwidth bottleneck analysis
  - Thermal operating points with multi-power profiles

### Phase 3: Advanced Analysis (In Progress)

- Roofline-based bottleneck identification
- Component-level energy breakdown (compute vs memory)
- Multi-batch performance scaling
- Fusion optimization opportunities

## Documentation

### CLI Tools Documentation
- **[CLI README](cli/README.md)** - Overview of all command-line tools
- **[Analyze Graph Mapping](cli/docs/analyze_graph_mapping.md)** - Hardware mapping analysis
- **[Compare Models](cli/docs/compare_models.md)** - Multi-model hardware comparison
- **[List Hardware](cli/docs/list_hardware_mappers.md)** - Hardware catalog and discovery
- **[Discover Models](cli/docs/discover_models.md)** - FX-traceable model discovery
- **[Profile Graph](cli/docs/profile_graph.md)** - Hardware-independent profiling
- **[Partitioner](cli/docs/partitioner.md)** - Graph partitioning and fusion
- **[Comparison Tools](cli/docs/comparison_tools.md)** - Specialized comparison utilities

### Core Documentation
- **[Getting Started Guide](docs/getting_started.md)** - Introduction to the framework
- **[Graph Partitioner Tutorial](docs/graph_partitioner_tutorial.md)** - Hands-on tutorials
- **[CI Workflow](docs/ci_workflow.md)** - Continuous integration and testing
- **[Package Reorganization](docs/sessions/2025-10-24_package_reorganization.md)** - Architecture overview
- **[Hardware Characterization](docs/hardware_characterization_2025-10.md)** - Hardware modeling details
- **[Examples](examples/)** - Quick start scripts and demonstrations

### Hardware Architecture Documentation
- **[Hardware Architecture Taxonomy](docs/hardware/architecture_taxonomy.md)** 🌟 **Essential Reference**
  - Comprehensive guide to CPU, GPU, TPU, KPU, DSP, DPU, and CGRA execution models
  - Flynn's taxonomy classification and programming paradigms
  - Execution model comparison (temporal vs spatial, SIMD vs SIMT vs MIMD)
  - Architecture selection guide and mapper implementation patterns
- **[Hardware Documentation Index](docs/hardware/)** - All hardware specifications and references

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

```txt
graphs/
├── src/graphs/                            # Core library packages
│   ├── ir/                                   # Intermediate Representation
│   │   └── structures.py                        # Graph data structures (SubgraphDescriptor, etc.)
│   │
│   ├── transform/                            # Graph Transformations
│   │   ├── partitioning/                        # Graph partitioning strategies
│   │   │   ├── graph_partitioner.py                # Basic partitioning
│   │   │   └── fusion_partitioner.py               # Fusion-based partitioning
│   │   ├── fusion/                              # Fusion transformations (future)
│   │   ├── tiling/                              # Tiling transformations (future)
│   │   └── visualization.py                     # Graph visualization
│   │
│   ├── analysis/                             # Performance Analysis
│   │   ├── concurrency.py                       # Multi-level parallelism analysis
│   │   └── allocation.py                        # Resource allocation analysis
│   │
│   ├── hardware/                             # Hardware Modeling & Mapping
│   │   ├── resource_model.py                    # Base hardware resource models
│   │   ├── table_formatter.py                   # Hardware comparison tables
│   │   ├── models/                              # Hardware specifications
│   │   │   ├── datacenter/                         # H100, A100, V100, T4, TPU v4, Xeon, EPYC
│   │   │   ├── edge/                               # Jetson Orin, Coral Edge TPU
│   │   │   ├── automotive/                         # Jetson Thor, TI TDA4x
│   │   │   ├── mobile/                             # ARM Mali, Qualcomm Adreno
│   │   │   └── accelerators/                       # KPU, DPU, CGRA, NPU IP cores
│   │   └── mappers/                             # Architecture-specific mappers
│   │       ├── cpu.py                              # CPU multi-core + SIMD
│   │       ├── gpu.py                              # GPU SM allocation
│   │       ├── dsp.py                              # DSP vector/tensor units
│   │       └── accelerators/                       # TPU, KPU, DPU, CGRA, Hailo
│   │
│   ├── subgraphs/                            # Synthetic DNN definitions
│   │   ├── mlp.py                               # Parameterized MLP
│   │   ├── resnet_block.py                      # Synthetic ResNet block
│   │   └── conv2d_stack.py                      # Stacked Conv2D layers
│   │
│   ├── execute/                              # Execution utilities
│   │   ├── walker.py                            # FX graph walker
│   │   └── fused_ops.py                         # Fused operation registry
│   │
│   ├── compile/                              # Compilation utilities
│   │   └── tiling.py                            # Tiling strategies
│   │
│   ├── experiment/                           # Experimental tools
│   │   ├── complexity.py                        # Complexity analysis
│   │   ├── sweep.py                             # Parameter sweeps
│   │   └── estimateEDP.py                       # Energy-delay product
│   │
│   └── validation/                           # Validation utilities
│
├── cli/                                   # Command-line tools
│   ├── analyze_graph_mapping.py              # Hardware mapping analysis
│   ├── compare_models.py                     # Multi-model comparison
│   ├── compare_datacenter_cpus.py            # CPU comparison (datacenter)
│   ├── compare_edge_ai_platforms.py          # Edge AI comparison
│   ├── compare_automotive_adas.py            # Automotive ADAS comparison
│   ├── compare_ip_cores.py                   # IP core comparison
│   ├── list_hardware_mappers.py              # Hardware catalog
│   ├── discover_models.py                    # Model discovery
│   ├── profile_graph.py                      # Graph profiling
│   ├── partitioner.py                        # Graph partitioning
│   └── docs/                                 # CLI documentation (7 guides)
│
├── validation/                            # Validation scripts
│   ├── estimators/                           # Estimator validation
│   ├── hardware/                             # Hardware model validation
│   └── empirical/                            # Empirical calibration
│
├── tests/                                 # Test suite
│   ├── transform/partitioning/               # Partitioning tests
│   ├── hardware/                             # Hardware mapper tests
│   ├── analysis/                             # Analysis tests
│   └── ir/                                   # IR structure tests
│
├── examples/                              # Example scripts
│   ├── quick_start_partitioner.py            # Quick start
│   ├── demo_fusion_comparison.py             # Fusion demo
│   └── visualize_partitioning.py             # Visualization demo
│
├── docs/                                  # Documentation
│   ├── sessions/                             # Development session notes
│   ├── hardware/                             # Hardware specifications
│   ├── validation/                           # Validation reports
│   └── bugs/                                 # Bug tracking and fixes
│
├── data/profiles/                         # Profiling data and analysis
├── workloads/                             # Reference workloads (PyTorch, JAX, TF)
├── experiments/                           # Research experiments
├── archive/                               # Archived/deprecated code
└── tools/                                 # Utility tools
```
