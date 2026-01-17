# Target Software Architecture

> This document defines the target architecture for the graphs repository to become a high-quality performance and energy estimator for DNNs and Embodied AI applications.

## Vision

**Goal**: A rich, capable modeling platform that provides latency, memory, performance, and energy estimates with 10-15% accuracy vs real measurements.

**Core Value Proposition**: Map computational graphs to hardware resources and produce trustworthy estimates backed by calibration data.

## Core Insight

The difference between "toy estimates" and "10-15% accurate estimates" is calibration. The architecture treats calibration and benchmarking as first-class concerns, not afterthoughts.

## Architectural Layers

```
+------------------------------------------------------------------+
|                        CLI / API Layer                           |
|  (analyze, benchmark, calibrate, validate, compare)              |
+------------------------------------------------------------------+
                              |
+------------------------------------------------------------------+
|                     Unified Analyzer                             |
|  (orchestrates the pipeline, returns estimates with confidence)  |
+------------------------------------------------------------------+
          |                   |                    |
+-----------------+  +------------------+  +------------------+
|   Graph Layer   |  |  Hardware Layer  |  |  Quality Layer   |
+-----------------+  +------------------+  +------------------+
| - Frontends     |  | - Specs          |  | - Benchmarks     |
| - IR            |  | - Mappers        |  | - Calibration    |
| - Transforms    |  | - Resource Models|  | - Validation     |
| - Analysis      |  | - Registry       |  | - Registries     |
+-----------------+  +------------------+  +------------------+
```

## Core CLI Operations

The four fundamental operations form a characterization pipeline:

| Operation | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **analyze** | Produce estimates | Model + Hardware | Latency, memory, energy estimates |
| **benchmark** | Define & run measurements | Benchmark suite + Hardware | Raw measurements |
| **calibrate** | Fit models to data | Measurements | Calibration coefficients |
| **validate** | Check accuracy | Estimates + Measurements | Accuracy report |

**compare** sits above these as a higher-level operation that uses analyze across multiple models/hardware.

## Package Structure

```
src/graphs/
|
+-- core/                      # Framework-independent abstractions
|   +-- ir/                    # Internal Representation
|   |   +-- graph.py           # ComputeGraph, Node, Edge
|   |   +-- operator.py        # Operator catalog with compute profiles
|   |   +-- tensor.py          # TensorDescriptor
|   |   +-- subgraph.py        # SubgraphDescriptor, PartitionReport
|   |
|   +-- types.py               # Enums, precision types, common types
|   +-- metrics.py             # Metric definitions (FLOPS, bytes, etc.)
|
+-- frontends/                 # Graph import from frameworks
|   +-- base.py                # Frontend interface
|   +-- pytorch_fx.py          # PyTorch FX tracing + shape propagation
|   +-- onnx.py                # ONNX import (future)
|   +-- tflite.py              # TFLite import (future)
|
+-- transform/                 # Graph transformations
|   +-- partitioning/          # Graph partitioning strategies
|   +-- fusion/                # Operator fusion
|   +-- tiling/                # Tiling strategies
|
+-- hardware/                  # Hardware modeling
|   +-- specs/                 # Hardware specifications (data)
|   |   +-- schema.py          # Pydantic models for hardware specs
|   |   +-- compute.py         # Compute unit specs (cores, SMs, systolic)
|   |   +-- memory.py          # Memory hierarchy specs
|   |   +-- power.py           # TDP, power states, leakage
|   |
|   +-- mappers/               # Architecture-specific mapping (logic)
|   |   +-- base.py            # HardwareMapper interface
|   |   +-- cpu/
|   |   +-- gpu/
|   |   +-- accelerators/
|   |
|   +-- registry/              # Hardware registry
|       +-- entries/           # YAML/JSON per-hardware specs
|       +-- loader.py          # Registry loader with validation
|
+-- estimation/                # Estimation engines
|   +-- base.py                # EstimationResult with confidence
|   +-- latency.py             # Roofline-based latency
|   +-- memory.py              # Peak memory, memory timeline
|   +-- energy.py              # Three-component energy model
|   +-- unified.py             # UnifiedAnalyzer orchestrator
|
+-- benchmarks/                # Benchmark subsystem (CORE COMPONENT)
|   +-- definitions/           # Benchmark specifications
|   |   +-- microbenchmarks/   # Operator-level (GEMM, Conv2d, etc.)
|   |   +-- workloads/         # Full model benchmarks
|   |   +-- schema.py          # Benchmark definition schema
|   |
|   +-- runners/               # Platform-specific execution
|   |   +-- base.py            # Runner interface
|   |   +-- pytorch.py         # PyTorch benchmark runner
|   |   +-- cuda.py            # CUDA-specific runner
|   |   +-- cpu.py             # CPU runner
|   |
|   +-- collectors/            # Metric collection
|   |   +-- latency.py         # Timing collection
|   |   +-- power.py           # Power measurement (nvidia-smi, etc.)
|   |   +-- memory.py          # Memory profiling
|   |
|   +-- registry/              # Benchmark results storage
|       +-- results/           # Raw measurement data
|       +-- loader.py          # Results loader
|
+-- calibration/               # Calibration subsystem (CORE COMPONENT)
|   +-- fitting/               # Model fitting algorithms
|   |   +-- roofline.py        # Fit achieved bandwidth/FLOPS
|   |   +-- energy.py          # Fit pJ/op, pJ/byte coefficients
|   |   +-- efficiency.py      # Fit utilization factors
|   |
|   +-- registry/              # Calibration data storage
|   |   +-- entries/           # Per-hardware calibration profiles
|   |   +-- loader.py          # Calibration registry loader
|   |
|   +-- analysis.py            # Calibration quality analysis
|
+-- validation/                # Validation subsystem (CORE COMPONENT)
|   +-- comparators/           # Estimate vs measurement comparison
|   +-- reporters/             # Accuracy report generation
|   +-- regression/            # Regression test infrastructure
|
+-- reporting/                 # Output generation
|   +-- formats/               # Text, JSON, CSV, Markdown
|   +-- visualization/         # Plots, diagrams
|
+-- cli/                       # Command-line tools
    +-- analyze.py             # Main analysis
    +-- benchmark.py           # Run benchmarks
    +-- calibrate.py           # Fit models from measurements
    +-- validate.py            # Compare estimates vs measurements
    +-- compare.py             # Compare models/hardware (higher-level)
```

## Key Design Principles

### 1. Estimates Always Include Confidence

```python
@dataclass
class EstimationResult:
    value: float                    # The estimate
    unit: str                       # ms, mJ, MB, etc.
    confidence: ConfidenceLevel     # HIGH, MEDIUM, LOW, THEORETICAL
    calibration_source: str | None  # Which calibration data was used
    error_bound: float | None       # +/- percentage if calibrated
```

Confidence levels:
- **CALIBRATED**: Based on measurements on this exact hardware
- **INTERPOLATED**: Derived from similar hardware measurements
- **THEORETICAL**: Based on vendor specs (often 2-3x optimistic)

### 2. Separation of Specs vs Calibration

| Hardware Specs | Calibration Data |
|----------------|------------------|
| Vendor-provided | Measured |
| Theoretical peaks | Achieved performance |
| TDP | Actual power draw |
| Memory bandwidth (spec) | Memory bandwidth (measured) |
| Static, rarely changes | Updated with new measurements |

### 3. Characterization Pipeline

```
Benchmark Definitions --> Benchmark Runs --> Raw Measurements
                                                    |
                                                    v
                                            Calibration Fitting
                                                    |
                                                    v
Hardware Specs + Calibration Data --> Estimation Models --> Estimates
                                                                |
                                                                v
                                            Validation (compare to measurements)
```

### 4. Operator-Centric Modeling

The fundamental unit of estimation is the **operator** (GEMM, Conv2d, Attention, etc.). Each operator has:
- A compute profile (FLOPs as function of input shapes)
- A memory profile (bytes read/written)
- Hardware-specific calibration (achieved efficiency on each platform)

### 5. Three Registries

```
hardware_registry/           # What hardware exists and its specs
+-- cpu/
|   +-- intel_xeon_w9-3595x.yaml
|   +-- amd_epyc_9754.yaml
+-- gpu/
|   +-- nvidia_h100_sxm5_80gb.yaml
|   +-- nvidia_jetson_orin_agx.yaml
+-- accelerators/
    +-- google_tpu_v4.yaml

calibration_registry/        # Fitted model parameters
+-- nvidia_h100_sxm5_80gb/
|   +-- roofline.yaml        # Achieved bandwidth, peak FLOPS
|   +-- energy.yaml          # pJ/op, pJ/byte coefficients
|   +-- efficiency.yaml      # Utilization factors by op type
+-- intel_xeon_w9-3595x/
    +-- ...

benchmark_registry/          # Raw measurement data
+-- nvidia_h100_sxm5_80gb/
|   +-- gemm_fp16.csv        # GEMM measurements by size
|   +-- conv2d_fp16.csv      # Conv2d measurements
|   +-- resnet50_batch.csv   # Full model measurements
+-- intel_xeon_w9-3595x/
    +-- ...
```

## Data Flow for Analysis

```
1. IMPORT: PyTorch model --> FX trace --> ComputeGraph IR
2. TRANSFORM: Partition --> Fuse --> Schedule
3. MAP: Subgraph + HardwareSpec --> MappedSubgraph (resource allocation)
4. ESTIMATE: MappedSubgraph + CalibrationData --> EstimationResult
5. REPORT: EstimationResults --> Report (with confidence indicators)
```

## Documentation Structure

Aligned with the architecture:

```
docs/
+-- architecture/            # How the system works
|   +-- overview.md          # High-level architecture
|   +-- target_architecture.md  # This document
|   +-- ir.md                # Internal representation
|   +-- estimation.md        # Estimation theory (roofline, energy)
|   +-- calibration.md       # Calibration methodology
|
+-- guides/                  # How to use the system
|   +-- getting_started.md
|   +-- analyzing_models.md
|   +-- adding_hardware.md
|   +-- benchmarking.md
|   +-- calibrating.md
|
+-- reference/               # API and CLI reference
|   +-- cli.md
|   +-- api.md
|
+-- hardware/                # Hardware-specific details
|   +-- taxonomy.md          # Architecture taxonomy
|   +-- cpu.md
|   +-- gpu.md
|   +-- accelerators.md
|
+-- validation/              # Accuracy reports
|   +-- methodology.md
|   +-- results/
|
+-- archive/                 # Historical docs
    +-- sessions/            # Development session logs
    +-- legacy/              # Superseded designs
```

## Summary: Key Architectural Decisions

1. **Four core operations**: analyze, benchmark, calibrate, validate
2. **Benchmarks elevated to core component** - Peer to calibration, not child of it
3. **Estimates include confidence** - Every estimate indicates its reliability
4. **Three registries** - Hardware specs, calibration data, benchmark results
5. **Operator-centric modeling** - Operators are the fundamental unit of estimation
6. **Clean layer separation** - Graph, Hardware, Quality (benchmark/calibrate/validate)
7. **Validation pipeline** - Systematic accuracy tracking to achieve 10-15% target

---

*Document created: 2025-01-16*
*Status: Target architecture for repository evolution*
