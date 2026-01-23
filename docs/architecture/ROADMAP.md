# Graphs Repository Roadmap

> Roadmap to transform the graphs repository into a high-quality performance and energy estimator with 10-15% accuracy vs real measurements.

## Version Strategy

This project follows [Semantic Versioning](https://semver.org/):

| Milestone | Version | Release Type | Status |
|-----------|---------|--------------|--------|
| M1: Foundation Consolidation | **0.8.0** | Alpha | **COMPLETE** |
| M2: Benchmarking Infrastructure | 0.9.0-alpha | Alpha | Planned |
| M3: Calibration Framework | 0.9.0-beta | Beta | Planned |
| M4: Validation Pipeline | 0.9.0-rc | RC | Planned |
| M5: Hardware Coverage | **1.0.0** | Stable | Planned |
| M6: Frontend Expansion | 1.1.0 | Feature | Future |
| M7: Advanced Analysis | 1.2.0 | Feature | Future |
| M8: Production Readiness | 2.0.0 | Major | Future |

See [CHANGELOG.md](/CHANGELOG.md) for release history.
See [PROPOSAL-001](/docs/proposals/PROPOSAL-001-semver-governance-delegation.md) for versioning and governance details.

## Vision

A rich, capable modeling platform for DNNs and Embodied AI applications that provides trustworthy latency, memory, performance, and energy estimates backed by calibration data.

## Current State

- Graph extraction from PyTorch FX with shape propagation
- Partitioning into subgraphs with fusion optimization
- ~12 hardware mappers (CPU, GPU, TPU, KPU, DSP, DPU, CGRA)
- Basic roofline-based latency and energy estimation
- Simple benchmarking infrastructure in hardware_registry
- Unified analyzer and reporting framework (Phase 4.2)

## Target State

- Comprehensive frontend support (PyTorch FX, ONNX, TFLite)
- 30+ calibrated hardware mappers
- Systematic benchmarking infrastructure
- Calibrated estimation models with 10-15% accuracy
- Validation pipeline with accuracy tracking
- Rich CLI tools: analyze, benchmark, calibrate, validate, compare

---

## Milestone 1: Foundation Consolidation - v0.8.0 COMPLETE

**Goal**: Solidify the existing codebase and establish the infrastructure for calibration-driven estimation.

**Status**: COMPLETE (2026-01-23)

### 1.1 Package Structure Alignment
- [x] Reorganize src/graphs/ to match target architecture
  - [x] Create `core/` package with IR abstractions (renamed from `ir/`)
  - [x] Create `frontends/` package (move FX tracing from scripts)
  - [x] Create `estimation/` package (renamed from `analysis/`)
  - [x] Create `benchmarks/` package (elevated from hardware/calibration/)
  - [x] Create `calibration/` package (elevated from hardware/)
  - [x] Validation remains in `validation/` directory (functional tests)
- [x] Define clear interfaces between packages
- [x] Update imports across codebase
- [x] Add deprecation shims for backward compatibility

### 1.2 Estimation Result with Confidence
- [x] Define `EstimationConfidence` dataclass with confidence levels
- [x] Define `ConfidenceLevel` enum (CALIBRATED, INTERPOLATED, THEORETICAL, UNKNOWN)
- [x] Add confidence fields to all descriptors (Latency, Energy, Memory)
- [x] Propagate confidence through UnifiedAnalyzer
- [x] Display confidence in reports

### 1.3 Registry Infrastructure
- [x] Hardware registry schema defined in calibration/schema.py
- [x] Calibration profile schema implemented
- [x] Benchmark results stored in profiles/
- [x] Registry loaders with validation
- [x] Existing hardware_registry entries migrated

### 1.4 Documentation Alignment
- [x] Reorganize docs/ to match target categories
- [x] Architecture overview in CLAUDE.md
- [x] Updated CLAUDE.md to reflect new structure
- [x] SEMVER versioning strategy documented (PROPOSAL-001)
- [x] Task delegation framework documented

---

## Milestone 2: Benchmarking Infrastructure - v0.9.0-alpha

**Goal**: Build systematic benchmarking capabilities to collect real measurements.

**Status**: Planned

### 2.1 Benchmark Definition Framework
- [ ] Define benchmark specification schema
- [ ] Create microbenchmark definitions for core operators
  - [ ] GEMM (various sizes, precisions)
  - [ ] Conv2d (various configurations)
  - [ ] Attention (various sequence lengths)
  - [ ] Elementwise operations
- [ ] Create workload benchmark definitions
  - [ ] ResNet family
  - [ ] MobileNet family
  - [ ] EfficientNet family
  - [ ] BERT/Transformer models
  - [ ] YOLO models

### 2.2 Benchmark Runners
- [ ] Define runner interface
- [ ] Implement PyTorch CPU runner
- [ ] Implement PyTorch CUDA runner
- [ ] Implement power measurement collector (nvidia-smi)
- [ ] Implement memory profiling collector
- [ ] Add warmup and statistical aggregation

### 2.3 CLI Integration
- [ ] Create `cli/benchmark.py` command
- [ ] Support microbenchmark and workload modes
- [ ] Support output to benchmark registry
- [ ] Add progress reporting

### 2.4 Initial Benchmark Collection
- [ ] Run benchmarks on available hardware
- [ ] Populate benchmark registry with results
- [ ] Document benchmark methodology

---

## Milestone 3: Calibration Framework - v0.9.0-beta

**Goal**: Fit estimation models to real measurements for improved accuracy.

**Status**: Planned

### 3.1 Roofline Calibration
- [ ] Implement roofline parameter fitting
  - [ ] Measure achieved memory bandwidth
  - [ ] Measure achieved compute throughput
  - [ ] Fit efficiency curves by operation type
- [ ] Store calibration in registry
- [ ] Update latency estimator to use calibrated values

### 3.2 Energy Model Calibration
- [ ] Implement energy coefficient fitting
  - [ ] Fit pJ/op for compute operations
  - [ ] Fit pJ/byte for memory operations
  - [ ] Fit static power from idle measurements
- [ ] Store calibration in registry
- [ ] Update energy estimator to use calibrated values

### 3.3 Utilization Factor Calibration
- [ ] Measure actual vs theoretical utilization by op type
- [ ] Fit utilization curves (vs problem size, batch size)
- [ ] Store calibration in registry
- [ ] Update mappers to apply calibrated factors

### 3.4 CLI Integration
- [ ] Create `cli/calibrate.py` command
- [ ] Support fitting from benchmark results
- [ ] Support selective calibration (roofline only, energy only)
- [ ] Generate calibration quality reports

---

## Milestone 4: Validation Pipeline - v0.9.0-rc

**Goal**: Systematically track estimation accuracy and prevent regression.

**Status**: Planned

### 4.1 Validation Framework
- [ ] Define validation test specification
- [ ] Implement estimate vs measurement comparison
- [ ] Calculate accuracy metrics (MAPE, correlation)
- [ ] Generate accuracy reports

### 4.2 Accuracy Tracking
- [ ] Create accuracy dashboard/report
- [ ] Track accuracy by hardware platform
- [ ] Track accuracy by model type
- [ ] Track accuracy by operation type
- [ ] Set accuracy targets and alerts

### 4.3 Regression Testing
- [ ] Define regression test suite
- [ ] Integrate with CI/CD
- [ ] Alert on accuracy degradation
- [ ] Document known accuracy limitations

### 4.4 CLI Integration
- [ ] Create `cli/validate.py` command
- [ ] Support comparison against benchmark registry
- [ ] Generate validation reports
- [ ] Support selective validation (by hardware, by model)

---

## Milestone 5: Hardware Coverage Expansion - v1.0.0

**Goal**: Expand hardware support with calibrated mappers. This milestone marks the first stable release.

**Status**: Planned

### 5.1 GPU Mappers
- [ ] Calibrate H100 mapper
- [ ] Calibrate A100 mapper
- [ ] Calibrate RTX 4090 mapper
- [ ] Add L4/L40 mappers
- [ ] Calibrate Jetson Orin mappers

### 5.2 CPU Mappers
- [ ] Calibrate Intel Xeon mappers (Sapphire Rapids, Emerald Rapids)
- [ ] Calibrate AMD EPYC mappers (Genoa, Bergamo)
- [ ] Calibrate Apple Silicon mappers (M2, M3, M4)
- [ ] Calibrate Ampere Altra mappers

### 5.3 Accelerator Mappers
- [ ] Calibrate TPU v4/v5 mappers (via published data)
- [ ] Calibrate Hailo-8 mapper
- [ ] Add Intel Gaudi mapper
- [ ] Add AWS Inferentia/Trainium mappers
- [ ] Add Qualcomm Cloud AI 100 mapper

### 5.4 Edge Accelerators
- [ ] Calibrate Coral Edge TPU mapper
- [ ] Calibrate Kendryte K210 mapper
- [ ] Add Rockchip NPU mappers
- [ ] Add MediaTek APU mappers

---

## Milestone 6: Frontend Expansion - v1.1.0

**Goal**: Support multiple graph formats beyond PyTorch FX.

**Status**: Future

### 6.1 ONNX Frontend
- [ ] Implement ONNX graph loader
- [ ] Map ONNX ops to internal IR
- [ ] Handle ONNX shape inference
- [ ] Validate against PyTorch FX for same models

### 6.2 TFLite Frontend
- [ ] Implement TFLite flatbuffer parser
- [ ] Map TFLite ops to internal IR
- [ ] Handle TFLite quantization info
- [ ] Validate against reference models

### 6.3 Frontend Abstraction
- [ ] Define common frontend interface
- [ ] Ensure consistent IR output across frontends
- [ ] Add frontend auto-detection

---

## Milestone 7: Advanced Analysis - v1.2.0

**Goal**: Add sophisticated analysis capabilities.

**Status**: Future

### 7.1 Multi-Model Analysis
- [ ] Support pipeline analysis (multiple models in sequence)
- [ ] Support ensemble analysis (models running in parallel)
- [ ] Support dynamic batching analysis

### 7.2 Heterogeneous Execution
- [ ] Support split execution across devices
- [ ] Model data transfer costs between devices
- [ ] Optimize partition placement

### 7.3 Temporal Analysis
- [ ] Model execution timeline
- [ ] Identify pipeline bubbles
- [ ] Model memory reuse across operators

### 7.4 What-If Analysis
- [ ] Support batch size sweeps
- [ ] Support precision sweeps
- [ ] Support hardware comparison
- [ ] Generate optimization recommendations

---

## Milestone 8: Production Readiness - v2.0.0

**Goal**: Make the tool production-ready for external users. Breaking changes allowed (removal of deprecated APIs).

**Status**: Future

### 8.1 Documentation
- [ ] Complete user guide
- [ ] Complete API reference
- [ ] Add tutorials for common use cases
- [ ] Add hardware calibration guide

### 8.2 Testing
- [ ] Achieve 80% code coverage
- [ ] Add integration tests
- [ ] Add performance benchmarks for tool itself

### 8.3 Packaging
- [ ] PyPI package publication
- [ ] Docker container
- [ ] CI/CD for releases

### 8.4 Community
- [ ] Contribution guidelines
- [ ] Issue templates
- [ ] Example notebooks

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Latency estimation accuracy | < 15% MAPE |
| Energy estimation accuracy | < 20% MAPE |
| Hardware platforms calibrated | 30+ |
| Model architectures supported | 50+ |
| Operator types supported | 100+ |
| Frontend formats | 3 (FX, ONNX, TFLite) |

---

## Dependencies and Risks

### Dependencies
- Access to hardware for benchmarking
- Published specifications for some accelerators
- Community contributions for edge platforms

### Risks
- Some hardware may not be accessible for calibration
- Vendor specs may be inaccurate or incomplete
- Dynamic behavior (thermal throttling, etc.) hard to model

### Mitigations
- Use published benchmark data where hardware unavailable
- Document confidence levels clearly
- Support calibration updates as data becomes available

---

---

*Roadmap created: 2025-01-16*
*Last updated: 2026-01-23*
*Current version: 0.8.0 (Milestone 1 complete)*
*Next version: 0.9.0-alpha (Milestone 2)*
