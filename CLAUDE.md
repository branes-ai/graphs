# CLAUDE.md

This file provides guidance to Claude Code when working with the `graphs` repository.

## Mission

This repository is the **quantitative/analytical tool-call layer** of the Embodied AI Architect agentic system. It provides repeatable, confidence-tracked estimates of latency, memory, energy, and throughput for computational graphs running on diverse hardware architectures. The Embodied-AI-Architect LLM orchestrator calls into this repo's estimators to drive hardware selection, model optimization, and deployment planning.

The estimator scope is expanding from DNN-only to the full embodied AI pipeline:
- **DNN models**: CNNs, transformers, detection/segmentation heads
- **Pipeline stages**: perception, planning, control, sensor fusion
- **Linear algebra operators**: Kalman filters, EKFs, PID controllers
- **Compiler transforms**: fusion, tiling, quantization, scheduling
- **Runtime concurrency**: multi-stream, pipeline parallelism, async DMA
- **Hardware efficiency**: latency, memory footprint, energy-delay product

All estimates carry `ConfidenceLevel` metadata (CALIBRATED, INTERPOLATED, THEORETICAL, UNKNOWN) so the agentic system knows how much to trust each result.

## Project Structure

```
graphs/
src/graphs/                     # Main Python package
  core/                         # Graph data structures, confidence tracking
  frontends/                    # Model tracing (PyTorch Dynamo)
  transform/                    # Partitioning, fusion, tiling
  estimation/                   # Performance estimators (roofline, energy, memory, concurrency)
  calibration/                  # Hardware calibration framework + profiles (JSON)
  benchmarks/                   # Calibration microbenchmarks (GEMM, memory, conv2d)
  hardware/                     # Resource models + architecture-specific mappers
    mappers/                    # CPU, GPU, DSP, TPU, KPU, DPU, CGRA, Hailo, DFM
  reporting/                    # Multi-format report generation
  models/                       # Synthetic DNN models (factory functions)
  scripts/                      # Entry point scripts
cli/                            # Command-line tools (30+ tools)
validation/                     # Functional validation (hardware, estimators, energy)
tests/                          # Unit tests (pytest)
experiments/                    # Research experiments (FX, CNN)
workloads/                      # Reference workloads (PyTorch, JAX, TFLite, ONNX)
tools/                          # Utilities (PyTorch, Chrome tracer)
docs/                           # Documentation, session logs, architecture guides
```

**Deprecated paths** (backward-compat shims with warnings):
- `ir/` -> use `core/`
- `analysis/` -> use `estimation/`

## Key Interfaces

### UnifiedAnalyzer (primary entry point)
```python
from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'H100', batch_size=1)
# result: UnifiedAnalysisResult (roofline, energy, memory, concurrency reports)
```

### Hardware Mapper Registry (50+ mappers)
```python
from graphs.hardware.mappers import get_mapper_by_name, list_all_mappers
mapper = get_mapper_by_name('h100_sxm5_80gb')
# Categories: CPU, GPU, DSP, TPU, KPU, DPU, CGRA, Hailo, DFM
```

### Individual Estimators
```python
from graphs.estimation.roofline import RooflineAnalyzer    # latency + bottleneck
from graphs.estimation.energy import EnergyAnalyzer        # 3-component energy
from graphs.estimation.memory import MemoryEstimator       # peak + timeline
from graphs.estimation.concurrency import ConcurrencyAnalyzer
```

### Model Frontends
```python
from graphs.frontends import trace_and_partition
fx_graph, partition_report = trace_and_partition(model, input_tensor)
```

### Confidence Tracking
```python
from graphs.core import ConfidenceLevel, EstimationConfidence
# All descriptors include confidence: CALIBRATED > INTERPOLATED > THEORETICAL > UNKNOWN
```

## Development Commands

```bash
pip install -e .                                    # Install package
python -m pytest tests/                             # Unit tests
python validation/hardware/test_all_hardware.py     # 10-way hardware comparison
python validation/estimators/test_resnet18.py       # Estimator accuracy

# CLI tools (Phase 4.2 unified framework)
./cli/analyze_comprehensive.py --model resnet18 --hardware H100
./cli/analyze_batch.py --model resnet18 --hardware H100 --batch-size 1 4 8 16
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output report.json

# C++ components
cmake -B build -DGRAPHS_ENABLE_TESTS=ON && cmake --build build && cd build && ctest
```

## Conventions

### Adding Hardware Mappers
1. Create mapper class inheriting `HardwareMapper` in `hardware/mappers/`
2. Add `HardwareResourceModel` with peak FLOPS, bandwidth, precision profiles
3. Add factory function: `create_<chip>_<formfactor>_<memory>_mapper()`
4. Register in `hardware/mappers/__init__.py` registry
5. Add validation test in `validation/hardware/`

### Adding Estimators
1. Create estimator in `estimation/` following the Analyzer pattern
2. All estimates must include `EstimationConfidence` with appropriate level
3. Integrate with `UnifiedAnalyzer` if it produces a new report type
4. Add accuracy validation in `validation/estimators/`

### Naming Conventions
- GPU mappers: `{Architecture}-{FormFactor}-{Memory}` (e.g., `h100_sxm5_80gb`)
- Factory functions: `create_<name>_mapper()`
- CLI tools: `analyze_*.py` for analysis, `benchmark_*.py` for benchmarks
- Validation tests: `test_*.py` in `validation/`

### Code Quality
- Do not use Unicode characters in code or output
- All CLI tools support `--output` with auto-detected format (JSON, CSV, MD, text)
- FX-traceable PyTorch models only (use `torch.export` with `symbolic_trace` fallback)
- Always run `ShapeProp` after FX tracing
- Confidence must propagate through the analysis chain

## Multi-Repo Architecture

```
embodied-schemas (shared Pydantic schemas, hardware/model catalog)
       |                    |
   graphs (this repo)    Embodied-AI-Architect (LLM orchestrator)
```

- `embodied-schemas`: vendor specs, physical specs, constraint tiers
- This repo: roofline params, calibration data, efficiency curves, operation profiles
- Hardware entries reference `embodied-schemas` via `base_id`

## Architecture Taxonomy

| Architecture | Execution Model | Key Abstraction |
|-------------|-----------------|-----------------|
| CPU | MIMD + SIMD | Core allocation, vector lanes |
| GPU | SIMT Data Parallel | SM allocation, warp scheduling |
| DSP | VLIW + vector/tensor | Heterogeneous functional units |
| TPU | Systolic Array | Weight-stationary dataflow |
| KPU | Spatial Dataflow | Tile-based stream processing |
| DPU | Reconfigurable FPGA | AIE tile arrays |
| CGRA | Spatial Dataflow | PE mesh with reconfiguration |
