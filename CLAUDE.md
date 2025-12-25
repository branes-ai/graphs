# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains synthetic DNN graph models and characterization tools for Deep Learning workload analysis. It focuses on FX graph-based workload characterization, estimation, and benchmarking across different target architectures (CPU, DSP, GPU, TPU, KPU, DPU, CGRA).

**Primary Focus**: Graph characterization pipeline that estimates FLOPs, memory usage, tiling requirements, latency, and energy consumption for neural networks across different hardware architectures.

## Project Structure

```
graphs/
├── src/graphs/                 # Main Python package
│   ├── models/                 # Synthetic DNN models (MLP, Conv2D, ResNet blocks)
│   ├── ir/                     # Intermediate Representation
│   │   ├── structures.py          # Core graph data structures
│   │   └── __init__.py
│   ├── transform/              # Graph Transformations
│   │   ├── partitioning/          # Graph partitioning strategies
│   │   │   ├── graph_partitioner.py   # Basic graph partitioning
│   │   │   ├── fusion_partitioner.py  # Fusion-based partitioning
│   │   │   └── __init__.py
│   │   ├── fusion/                # Fusion transformations
│   │   ├── tiling/                # Tiling transformations
│   │   ├── visualization.py       # Graph visualization utilities
│   │   └── __init__.py
│   ├── analysis/               # Performance Analysis
│   │   ├── unified_analyzer.py    # Unified analysis orchestrator
│   │   ├── roofline_analyzer.py   # Roofline model + latency estimation
│   │   ├── energy_analyzer.py     # Three-component energy model
│   │   ├── memory_estimator.py    # Peak memory + memory timeline
│   │   ├── concurrency.py         # Multi-level parallelism analysis
│   │   └── __init__.py
│   ├── reporting/              # Report Generation
│   │   ├── report_generator.py    # Multi-format report generation
│   │   └── __init__.py
│   ├── hardware/               # Hardware Modeling & Mapping
│   │   ├── resource_model.py      # Hardware resource models & base mapper
│   │   ├── table_formatter.py     # Hardware comparison tables
│   │   ├── mappers/               # Architecture-specific mappers
│   │   │   ├── base.py               # Base mapper interface
│   │   │   ├── cpu.py                # CPU multi-core + SIMD (Intel, AMD, Ampere)
│   │   │   ├── gpu.py                # GPU SM allocation (H100, Jetson)
│   │   │   ├── dsp.py                # DSP vector/tensor (Qualcomm Hexagon)
│   │   │   ├── accelerators/         # Fixed-function & reconfigurable accelerators
│   │   │   │   ├── tpu.py               # TPU systolic arrays (Google TPU v4, Coral)
│   │   │   │   ├── kpu.py               # KPU tile-based (Kendryte K210)
│   │   │   │   ├── dpu.py               # DPU FPGA (Xilinx Vitis AI)
│   │   │   │   ├── cgra.py              # CGRA spatial dataflow (Plasticine-style)
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── scripts/                # Entry point scripts
│       └── run_characterization.py
├── cli/                        # Command-line tools
│   ├── analyze_comprehensive.py     # Comprehensive analysis (Phase 4.2 unified framework)
│   ├── analyze_batch.py             # Batch size analysis (Phase 4.2 unified framework)
│   ├── analyze_comprehensive_v1.py  # Legacy comprehensive analysis
│   ├── analyze_batch_v1.py          # Legacy batch analysis
│   ├── analyze_graph_mapping.py     # Hardware mapping analysis
│   ├── partitioner.py               # Graph partitioning CLI
│   ├── profile_graph.py             # Model profiling tool
│   ├── discover_models.py           # Model discovery utility
│   ├── show_fvcore_table.py         # FLOP comparison tool
│   └── README.md
├── examples/                   # Capability demonstrations (REORGANIZED)
│   ├── quick_start_partitioner.py  # 30-second intro
│   ├── demo_fusion_comparison.py   # Fusion benefits demo
│   ├── demo_new_performance_model.py # Phase 2 demo
│   ├── compare_models.py          # Model comparison
│   ├── visualize_partitioning.py  # Visualization
│   └── README.md
├── tests/                      # Unit tests
│   ├── characterize/          # Legacy tests (to be reorganized)
│   │   ├── test_graph_partitioner.py
│   │   ├── test_graph_partitioner_general.py
│   │   ├── test_fusion_partitioner.py
│   │   └── test_arithmetic_intensity.py
│   └── README.md
├── validation/                 # Functional validation (NEW)
│   ├── hardware/              # Hardware mapper validation
│   │   ├── test_all_hardware.py  # 10-way comparison
│   │   ├── test_cgra_mapper.py
│   │   ├── test_dpu_mapper.py
│   │   ├── test_cpu_vs_gpu_mapping.py
│   │   ├── test_gpu_cpu_kpu_comparison.py
│   │   ├── test_hardware_mapping.py
│   │   ├── test_kpu_simple.py
│   │   └── README.md
│   ├── estimators/            # Estimator accuracy tests
│   │   ├── test_conv2d.py
│   │   ├── test_resnet18.py
│   │   ├── test_resnet_family.py
│   │   ├── test_mobilenet.py
│   │   ├── test_efficientnet.py
│   │   └── README.md
│   └── README.md
├── archive/                    # Debug/development artifacts
│   └── (debug scripts from development)
├── experiments/                # Research experiments
│   ├── fx/                    # PyTorch FX tracing experiments
│   └── CNN/                   # CNN building blocks
├── workloads/                  # Reference workloads
│   ├── pytorch/, jax/, tensorflow/, litert/, onnx/
├── tools/                      # Utility tools
│   ├── pt/                    # PyTorch model utilities
│   └── chrome-tracer/         # Chrome trace format tools
├── mlir/                       # MLIR generation and tools
└── docs/                       # Documentation

```

**Note**:
- **2025-10-24**: Package structure reorganized. The `characterize/` package has been split into focused packages: `ir/` (intermediate representation), `transform/` (graph transformations), `analysis/` (performance analysis), and `hardware/` (hardware modeling and mapping). See `docs/sessions/2025-10-24_package_reorganization.md` for details.
- **2025-10-28**: Phase 4.2 unified framework added. New `UnifiedAnalyzer` and `ReportGenerator` provide simplified API with 61.5% code reduction. Refactored CLI tools (`*_v2.py`) are production-ready. See `docs/UNIFIED_FRAMEWORK_API.md` and `docs/MIGRATION_GUIDE_PHASE4_2.md` for details.

## Common Development Commands

### Python Environment Setup
```bash
# Install all dependencies (recommended)
pip install -e .

# Or install manually
pip install torch torchvision pandas psutil py-cpuinfo

# Or from requirements.txt
pip install -r requirements.txt
```

**Core dependencies:**
- `torch`, `torchvision`: DNN models and graph tracing
- `pandas`: Results tabulation
- `psutil`: Cross-platform CPU/system info (hardware detection)
- `py-cpuinfo`: Cross-platform CPU details (hardware detection)

### Running Characterization
```bash
# Run the main characterization script
python src/graphs/scripts/run_characterization.py

# Output: sweep_results.csv containing FLOPs, memory, tiles, latency, energy per model/architecture
```

### Validation Tests
```bash
# Hardware validation (10-way comparison)
python validation/hardware/test_all_hardware.py

# Estimator accuracy validation
python validation/estimators/test_conv2d.py
python validation/estimators/test_resnet18.py
python validation/estimators/test_resnet_family.py

# See validation/README.md for details
```

### Examples (Demonstrations)
```bash
# Quick start (30-second intro)
python examples/quick_start_partitioner.py

# Fusion benefits demo
python examples/demo_fusion_comparison.py

# Model comparison
python examples/compare_models.py --models resnet18 mobilenet_v2

# See examples/README.md for all demos
```

### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/

# Run specific test suite
python tests/characterize/test_graph_partitioner.py

# See tests/README.md for details
```

### CLI Tools

**Phase 4.2 Unified Framework (Recommended):**
```bash
# Comprehensive analysis with all Phase 3 components
./cli/analyze_comprehensive.py --model resnet18 --hardware H100

# Batch size impact analysis
./cli/analyze_batch.py --model resnet18 --hardware H100 --batch-size 1 4 8 16

# JSON/CSV/Markdown output
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output report.json
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output report.csv
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output report.md

# Power management analysis (NEW 2025-11-03)
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --power-gating
./cli/analyze_batch.py --model resnet50 --hardware H100 --batch-size 1 4 16 64 --power-gating

# See cli/README.md for full documentation (including Power Management Analysis section)
```

**Other Tools:**
```bash
# Graph partitioning
./cli/partitioner.py --model resnet18 --output results.json

# Model profiling
./cli/profile_graph.py --model mobilenet_v2

# See cli/README.md for all tools
```

**Python API (Unified Framework):**
```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator

# Single call for complete analysis
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'H100')

# Generate reports in multiple formats
generator = ReportGenerator()
text_report = generator.generate_text_report(result)
json_report = generator.generate_json_report(result)
csv_report = generator.generate_csv_report(result)

# Power management analysis (NEW 2025-11-03)
config = AnalysisConfig(
    run_hardware_mapping=True,
    power_gating_enabled=True
)
result_pg = analyzer.analyze_model('resnet18', 'H100', batch_size=1, config=config)
print(f"Power Gating Savings: {result_pg.energy_report.total_power_gating_savings_j * 1000:.1f} mJ")

# See docs/UNIFIED_FRAMEWORK_API.md for full API documentation
```

### CMake Build (for C++ components)
```bash
# Configure with tests enabled (default)
cmake -B build -DGRAPHS_ENABLE_TESTS=ON

# Build
cmake --build build

# Run tests
cd build && ctest
```

## Architecture Overview

### Characterization Pipeline

The characterization system works through these stages:

1. **Model Definition** (`src/graphs/models/`): Synthetic DNNs with configurable parameters
2. **FX Tracing**: PyTorch FX symbolic tracing with shape propagation
3. **Graph Partitioning** (`transform/partitioning/`): Split graphs into fused subgraphs
4. **Concurrency Analysis** (`analysis/concurrency.py`): Identify parallelism opportunities
5. **Hardware Mapping** (`hardware/mappers/`): Map subgraphs to specific hardware architectures
6. **Estimation**: Calculate FLOPs, memory, latency, and energy using roofline models

### Key Components

**Phase 4.2: Unified Analysis Framework (Recommended)**

**UnifiedAnalyzer** (`analysis/unified_analyzer.py`):
- Single orchestrator for all Phase 3 analyzers
- Handles FX tracing, shape propagation, partitioning automatically
- Configurable analysis (roofline, energy, memory, concurrency)
- Returns UnifiedAnalysisResult with derived metrics
- 6-10× code reduction compared to manual orchestration

**ReportGenerator** (`reporting/report_generator.py`):
- Multi-format output (text, JSON, CSV, markdown)
- Consistent formatting across all tools
- Comparison reports for batch sweeps, model/hardware comparison
- Auto-detection from file extension
- Selective sections and customizable styles

**UnifiedAnalysisResult**:
- Metadata (model, hardware, batch size, precision)
- Phase 3 reports (roofline, energy, memory, concurrency)
- Derived metrics (latency, throughput, energy/inference, memory)
- Graph structure (partition report, subgraphs)

**Phase 3 Analyzers:**

**Roofline Analyzer** (`analysis/roofline_analyzer.py`):
- Latency estimation using roofline model
- Bottleneck analysis (compute vs memory-bound)
- Hardware utilization calculation
- Per-subgraph latency breakdown

**Energy Analyzer** (`analysis/energy_analyzer.py`):
- Three-component energy model
  - Compute energy (from FLOPs)
  - Memory energy (from data transfers)
  - Static/leakage energy (from latency)
- Per-subgraph energy breakdown
- Total energy and energy per inference

**Memory Estimator** (`analysis/memory_estimator.py`):
- Peak memory usage (activations + weights)
- Memory timeline (live tensors over time)
- Activation vs weight memory breakdown
- Hardware fit analysis (L2 cache, total memory)

**Core Infrastructure:**

**Graph Data Structures** (`ir/structures.py`):
- `TensorDescriptor`: Shape, dtype, memory footprint
- `ParallelismDescriptor`: Thread/warp/block parallelism
- `SubgraphDescriptor`: Fused operation metadata
- `PartitionReport`: Complete graph partitioning results

**Fusion Partitioner** (`transform/partitioning/fusion_partitioner.py`):
- Fuses operations to minimize data movement
- Creates subgraphs with compute/memory characteristics
- Determines partition boundaries based on bottleneck changes

**Hardware Resource Models** (`hardware/resource_model.py`):
- `HardwareResourceModel`: Peak FLOPS, memory bandwidth, precision profiles
- `HardwareMapper`: Base class for architecture-specific mapping
- Roofline model for latency estimation
- Energy models based on ops and bytes transferred

**Architecture-Specific Mappers** (`hardware/mappers/`):
- **CPU**: Multi-core + SIMD mapping with core allocation
- **GPU**: SM allocation with warp scheduling and wave quantization
- **DSP**: Vector/tensor unit mapping for signal processors
- **Accelerators**: TPU systolic arrays, KPU tile engines, DPU/FPGA, CGRA

## Workload Organization

### PyTorch Workloads (`workloads/pytorch/`)
- MLPs: `mlp/oneLayerMLP.py`, `mlp/twoLayerMLP.py`, `mlp/threeLayerMLP.py`
- CNNs: Various sizes for serialization testing
- MobileNet V2: Full model with MLIR conversion scripts
- DETR: Object detection transformer

### JAX Workloads (`workloads/jax/`)
- MLPs at various scales
- Conv1D/Conv2D examples
- JAX JIT and MLIR export experiments

### LiteRT/IREE Workflow (`workloads/litert/`)
Typical workflow:
1. Generate TFLite model: `python oneLayerMLP.py`
2. Import to MLIR: `iree-import-tflite oneLayerMLP.tflite -o oneLayerMLP_tosa.mlir`
3. Compile to VMFB: `iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu oneLayerMLP_tosa.mlir -o oneLayerMLP.vmfb`
4. Run: `iree-run-module --module=oneLayerMLP.vmfb --function=main --input="1x4xf32=[1.0 2.0 3.0 4.0]"`

## FX Experiments

### Custom Tracers (`experiments/fx/tutorial/`)
- `custom_tracer.py`: Basic custom tracer
- `profiling_tracer.py`: Tracer with torch.profiler integration
- `module_tracer.py`: Module-level tracing
- `conv-bn-fusion-optimization.py`: Conv+BatchNorm fusion example

### Usage Pattern for Profiling
```python
# Define model
class MyModel(torch.nn.Module):
    def forward(self, x):
        with torch.profiler.record_function('my_block'):
            return torch.relu(x) * 2

# Trace with profiler
pt = ProfilerTracer()
graph = pt.trace(model)
traced = torch.fx.GraphModule(pt.root, graph)

# Profile execution
with torch.autograd.profiler.profile() as prof:
    traced(input_tensor)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

## Model Factories

All synthetic models expose factory functions for easy instantiation:

```python
from graphs.models.mlp import make_mlp
from graphs.models.conv2d_stack import make_conv2d
from graphs.models.resnet_block import make_resnet_block

mlp = make_mlp(in_dim=128, hidden_dim=256, out_dim=64)
conv = make_conv2d(in_channels=3, out_channels=16, kernel_size=3)
resnet = make_resnet_block(in_channels=64, out_channels=128)
```

## Testing Philosophy

- **Regression Levels**: Configurable via CMake options
  - `GRAPHS_BUILD_REGRESSION_SANITY` (level 1): Quick sanity tests
  - `GRAPHS_BUILD_REGRESSION_STRESS` (level 4): Exhaustive tests
- C++ tests located in subdirectories (e.g., `tools/pt/`)
- Python characterization tests should validate estimator accuracy

## MLIR Conversion

For PyTorch models:
```bash
# Create .pt binary
python workloads/pytorch/create_pt_bin_file.py

# Convert to MLIR
python workloads/pytorch/convert_pt_bin_to_mlir.py
```

For TFLite/TOSA:
```bash
# Generate TFLite model, then:
iree-import-tflite model.tflite -o model_tosa.mlir
```

## Related Repositories

This project is part of a multi-repo architecture:

```
embodied-schemas (shared dependency)
       ↑              ↑
       │              │
   graphs (this repo)      Embodied-AI-Architect
```

### embodied-schemas (`../embodied-schemas`)
Shared Pydantic schemas and factual data catalog. This repo imports:
- `HardwareEntry`, `ChipEntry` - Hardware platform specifications
- `ModelEntry` - ML model specifications
- Constraint tier definitions (latency, power classes)

**Usage:**
```python
from embodied_schemas import HardwareEntry, Registry
from embodied_schemas.hardware import HardwareCapability
```

### Embodied-AI-Architect (`../Embodied-AI-Architect`)
LLM orchestration, agentic tools, CLI. Uses schemas and this repo's analysis tools.

### Data Split with embodied-schemas
| This Repo (graphs) | embodied-schemas |
|-------------------|------------------|
| `ops_per_clock` - Roofline params | Vendor specs (memory, TDP) |
| `theoretical_peaks` - Computed ceilings | Physical specs (weight, dimensions) |
| Calibration data - Measured performance | Environmental specs (temp, IP rating) |
| Operation profiles - GEMM/CONV benchmarks | Interface specs (CSI, USB, PCIe) |
| Efficiency curves | Power profiles and modes |

The `hardware_registry/` directory in this repo contains analysis-specific data that references base hardware specs in `embodied-schemas` via `base_id`.

## Important Notes

- **Shape Propagation**: Always run `ShapeProp` after FX tracing to populate tensor metadata
- **Fusion Partitioning**: The partitioner fuses operations to minimize data movement and maximize compute efficiency
- **Hardware Resource Models**: Add new hardware targets by creating resource models in `hardware/resource_model.py` and mappers in `hardware/mappers/`
- **GPU Naming Convention** (Updated 2025-11-10): All NVIDIA GPU mappers now follow `{Architecture}-{FormFactor}-{Memory}` pattern:
  - Datacenter: `create_b100_sxm6_192gb_mapper`, `create_h100_sxm5_80gb_mapper()`, `create_a100_sxm4_80gb_mapper()`, `create_v100_sxm3_32gb_mapper()`, `create_t4_pcie_16gb_mapper()`
  - Edge: `create_jetson_orin_agx_64gb_mapper()`, `create_jetson_orin_nano_8gb_mapper()`, `create_jetson_thor_128gb_mapper()`
  - Old names deprecated but still work (see `docs/GPU_NAMING_MIGRATION_GUIDE.md`)
- **Hardware Architecture Taxonomy**: See `docs/hardware/architecture_taxonomy.md` for comprehensive guide to execution models:
  - CPU: MIMD Stored Program Machine (multi-core + SIMD)
  - GPU: SIMT Data Parallel (warps of 32 threads lockstep)
  - TPU: Systolic Array / Weight-Stationary Dataflow
  - KPU: MIMD Domain Flow / Spatial Dataflow (stream processing)
  - DSP: VLIW with heterogeneous vector/tensor units
  - DPU: Reconfigurable FPGA tiles (AIE)
  - CGRA: Spatial dataflow with reconfiguration overhead
- **Roofline Model**: Latency estimation considers both compute-bound and memory-bound performance based on arithmetic intensity
- **Package Organization**:
  - `ir/`: Hardware-independent graph structures
  - `transform/`: Graph transformations (partitioning, fusion, tiling)
  - `analysis/`: Performance analysis without graph modification
  - `hardware/`: Hardware modeling and architecture-specific mapping
- don't use Unicode characters