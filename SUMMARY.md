# Graph Characterization & Partitioning: Project Summary

**Last Updated**: 2025-10-21
**Status**: Phase 1 Complete, Phase 2 Starting

---

## Table of Contents

1. [Current Status](#current-status)
2. [What We Built](#what-we-built)
3. [Phase 1: Graph Partitioning & Concurrency (✅ Complete)](#phase-1-graph-partitioning--concurrency--complete)
4. [Fusion-Based Partitioning (✅ Complete)](#fusion-based-partitioning--complete)
5. [Hardware Characterization Pipeline (✅ Production Ready)](#hardware-characterization-pipeline--production-ready)
6. [Enhanced Attention Fusion (📋 Planned)](#enhanced-attention-fusion--planned)
7. [Phase 2: Hardware Mapping (🚧 In Progress)](#phase-2-hardware-mapping--in-progress)
8. [Documentation & Examples](#documentation--examples)
9. [Quick Reference](#quick-reference)

---

## Current Status

### Completed (Phase 1)
- **Graph Partitioning**: Decomposes PyTorch FX graphs into subgraphs
- **Concurrency Analysis**: Multi-level parallelism analysis
- **Fusion-Based Partitioning**: Reduces ops by 1.9-2.1×, saves 20-42% memory
- **Hardware Profiles**: 6 real architectures (CPU/GPU/TPU/KPU)
- **Validation Framework**: 100% pass rate on tested models

### In Progress (Phase 2)
- **Hardware Mapping**: Map fused subgraphs to actual hardware resources
- **Realistic Utilization**: Account for limited parallelism (not 100%!)
- **Latency Correction**: Fix 1000× overly-optimistic estimates

### Planned (Future)
- **Enhanced Attention Fusion**: 8× better transformer fusion
- **Memory Bandwidth Modeling**: Roofline analysis
- **Multi-Device Support**: Pipeline parallelism

---

## What We Built

A comprehensive DNN characterization and partitioning pipeline that:

1. **Analyzes** neural network computation graphs (PyTorch FX)
2. **Partitions** graphs into executable subgraphs with fusion
3. **Characterizes** performance (FLOPs, memory, parallelism, concurrency)
4. **Estimates** realistic hardware utilization and latency
5. **Validates** against theoretical values and real benchmarks

### Core Capabilities

  - **Model Support**: MLP, Conv2D, ResNet, MobileNet, EfficientNet, ViT (any PyTorch model)
  - **Architecture Profiles**: Intel i7, AMD Ryzen 7, H100-PCIe, TPU v4, KPU-T2/T100
  - **Metrics**: FLOPs, MACs, Memory, Tiles, Latency, Energy, Parallelism, Concurrency
  - **Fusion Detection**: Conv+BN+ReLU, Conv+BN, Add+ReLU, Linear+ReLU
  - **Validation**: Within 6% FLOP accuracy, 100% test pass rate
  - **Real Hardware Specs**: Based on actual product specifications

---

## Phase 1: Graph Partitioning & Concurrency

### Problem Identified

**Original pipeline assumed 100% hardware utilization → 1000× too optimistic!**

Example: H100 has 132 SMs, but ResNet-18 at batch=1 only has **12 parallel ops** → only **~20% utilization**

### Solution: Multi-Layer Analysis

#### 1. GraphPartitioner (`graph_partitioner.py`, 800 lines)

Decomposes PyTorch FX graphs into computational subgraphs:

**Capabilities**:
- Extracts operations from FX graph (call_module, call_function, call_method)
- Computes FLOPs/MACs for each operation
- Measures memory traffic (input, output, weights)
- Detects available parallelism (batch, channel, spatial)
- Special handling for depthwise convolutions
- Builds dependency graph (NetworkX)
- Classifies bottlenecks (compute vs memory bound)

**Supported Operations**:
- Standard/Depthwise/Pointwise Conv2d
- Linear layers
- Activations (ReLU, ReLU6, GELU, Hardswish, SiLU)
- Normalization (BatchNorm, LayerNorm)
- Pooling (MaxPool, AvgPool, AdaptiveAvgPool)

#### 2. ConcurrencyAnalyzer (`concurrency_analyzer.py`, 380 lines)

Analyzes parallelism at multiple levels:

**Graph-Level**:
- Execution stages (topological sort)
- Critical path (longest dependency chain)
- Max parallel operations per stage

**Subgraph-Level**:
- Thread count estimation
- Hardware unit requirements
- Parallelism efficiency
- Special handling for depthwise convs

#### 3. Graph Structures (`graph_structures.py`, 600 lines)

Foundation data structures:
- `SubgraphDescriptor`: Complete operation characterization
- `ParallelismDescriptor`: Multi-dimensional parallelism tracking
- `ConcurrencyDescriptor`: Graph-level concurrency metrics

### Validation Results

| Model | FLOPs (G) | Subgraphs | Max Parallel | Critical Path | AI | Characterization |
|-------|-----------|-----------|--------------|---------------|-----|------------------|
| **ResNet-18** | 4.49 | 60 | 12 ops | 9 ops | 31 | Compute-intensive |
| **MobileNet-V2** | 1.91 | 141 | 12 ops | 24 ops | 14 | Memory-bound |
| **EfficientNet-B0** | 2.39 | 214 | 27 ops | 13 ops | 17 | Balanced |

**Key Insights**:
1. Graph-level parallelism limited at batch=1 (12-27 parallel ops)
2. Need batch≥10 to saturate H100 (132 SMs)
3. Arithmetic intensity varies: ResNet (31) vs MobileNet (14)
4. Operation count ≠ complexity (MobileNet: 141 ops, 1.91 GFLOPs)

### Files Created (Phase 1)

**Source Code**:
- `src/graphs/characterize/graph_structures.py` (600 lines)
- `src/graphs/characterize/graph_partitioner.py` (800 lines)
- `src/graphs/characterize/concurrency_analyzer.py` (380 lines)

**Tests**:
- `tests/test_graph_partitioner.py` (170 lines)
- `tests/test_graph_partitioner_general.py` (470 lines)

**Documentation**:
- `docs/GETTING_STARTED.md` (605 lines)
- `docs/graph_partitioner_tutorial.md` (605 lines)
- `docs/graph_partitioner_validation.md` (176 lines)
- `docs/realistic_performance_modeling_plan.md` (1,600 lines)

**Examples**:
- `examples/quick_start_partitioner.py` (127 lines)
- `examples/compare_models.py` (372 lines)
- `examples/README.md` (196 lines)

**Total**: ~5,700 lines

---

## Fusion-Based Partitioning (✅ Complete)

### The Real Problem

Original "partitioner" wasn't actually partitioning - it created **one subgraph per operator**:
- ResNet-18: 60 operators → 60 "subgraphs" (no fusion!)
- No aggregation, no memory traffic reduction

### Solution: Greedy Sequential Fusion

**Algorithm** (`fusion_partitioner.py`, 600 lines):
- Fuse operators sequentially until hitting a boundary
- **Boundaries**: Fork (multiple consumers), Join (multiple inputs), Resource limits

**Fusion Patterns Detected**:
- Conv2d + BatchNorm2d + ReLU
- Conv2d + BatchNorm2d
- Add + ReLU (residual connections)
- Conv2d + BatchNorm2d + ReLU6 (MobileNet)

### Results: Significant Memory Reduction

| Model | Original Ops | Fused Subgraphs | Reduction | Memory Saved |
|-------|--------------|-----------------|-----------|--------------|
| **ResNet-18** | 60 | 32 | 1.9× | 19.6% (19.2 MB) |
| **MobileNet-V2** | 141 | 66 | 2.1× | 42.0% (51.1 MB) |

**Why MobileNet benefits more**: Inverted residual blocks have more sequential operations, creating larger intermediate tensors that now stay in cache.

**Per-Subgraph Examples**:
- Conv+BN+ReLU fusions: 47-63% memory reduction
- MobileNet expansion layers: 63% reduction (9.6 MB stays in cache!)

### Impact

**Before Fusion**:
- 60-141 tiny kernels with unrealistic assumptions
- Intermediate data written to global memory

**After Fusion**:
- 32-66 meaningful execution units
- Can map to hardware (GPU SMs, KPU tiles, TPU arrays)
- 20-42% less memory traffic
- Intermediate tensors stay in L1/registers

### Files Created (Fusion)

**Source Code**:
- `src/graphs/characterize/fusion_partitioner.py` (600 lines)

**Examples/Tests**:
- `examples/test_fusion_partitioner.py` (372 lines)
- `examples/quick_start_partitioner.py` (enhanced with FX analysis)

**Documentation**:
- `docs/GRAPH_PARTITIONING_DESIGN.md`
- `docs/FUSION_ALGORITHM_PROPOSAL.md`
- `docs/FUSION_RESULTS.md`
- `docs/FX_GRAPH_PARTITIONING.md`

---

## Hardware Characterization Pipeline (✅ Production Ready)

### Architecture Profiles

| Architecture | Peak Performance | Memory Bandwidth | Energy | Use Case |
|--------------|------------------|------------------|---------|----------|
| **Intel Core i7** | 1.5 TFLOPS (FP32) | 80 GB/s (DDR5) | 1.004 J/GFLOP | Development |
| **AMD Ryzen 7** | 1.0 TFLOPS (FP32) | 80 GB/s (DDR5) | 1.004 J/GFLOP | Development |
| **H100-PCIe** | 750 TFLOPS (BF16) | 2 TB/s (HBM2e) | 0.501 J/GFLOP | Datacenter |
| **TPU v4** | 275 TFLOPS (BF16) | 1.2 TB/s (HBM2e) | 0.200 J/GFLOP | Cloud |
| **KPU-T2** | 2 TOPS (INT8) | 165 GB/s | 0.100 J/GFLOP | Edge IoT |
| **KPU-T100** | 100 TFLOPS | 1 TB/s (HBM) | 0.100 J/GFLOP | Edge Server |

### Performance Comparison (vs AMD Ryzen 7)

```
H100-PCIe:   1250× faster  (unmatched performance)
TPU v4:       458× faster  (best cloud balance)
KPU-T100:     167× faster  (high-performance edge)
KPU-T2:       3.3× faster  (low-power edge)
Intel Core i7: 1.5× faster  (consumer CPU leader)
AMD Ryzen 7:   1.0× baseline
```

### Energy Efficiency

```
KPU-T100:  0.100 J/GFLOP (10× better than CPU)
KPU-T2:    0.100 J/GFLOP (10× better than CPU)
TPU v4:    0.200 J/GFLOP (5× better than CPU)
H100-PCIe: 0.501 J/GFLOP (2× better than CPU)
CPUs:      1.004 J/GFLOP (baseline)
```

### ResNet Family Validation

| Model | Parameters | FLOPs (Measured) | FLOPs (Theory) | Accuracy | Tiles |
|-------|------------|------------------|----------------|----------|-------|
| **ResNet-18** | 11.69M | 3.79G | 3.59G | ±5.6% | 17 |
| **ResNet-34** | 21.80M | 7.49G | ~7.3G | ±2.6% | 33 |
| **ResNet-50** | 25.56M | 10.80G | ~10.5G | ±2.9% | 49 |

### Throughput Comparison (ResNet Block, 59 GFLOPs)

```
H100-PCIe:  21,100 inferences/sec  (datacenter training/inference)
TPU v4:      7,740 inferences/sec  (cloud-scale inference)
KPU-T100:    2,810 inferences/sec  (robotics, edge servers)
KPU-T2:         56 inferences/sec  (IoT, embedded devices)
Intel Core i7:  25 inferences/sec  (development only)
AMD Ryzen 7:    17 inferences/sec  (development only)
```

### Key Components

**SweepHarness**: Orchestrates batch characterization
**FXGraphWalker**: Core engine, walks graph and computes metrics
**ArchitectureProfile**: Hardware specs (peak FLOPS, energy coefficients)
**FusedOpRegistry**: Pattern matching for operation fusion
**Estimators**: Per-operation FLOP/memory calculation
**TilingStrategy**: Memory hierarchy modeling

### Files Created (Characterization)

**Validation Scripts**:
- `src/graphs/validation/test_conv2d.py`
- `src/graphs/validation/test_resnet18.py`
- `src/graphs/validation/test_resnet_family.py`

**Documentation**:
- `docs/characterization-architecture.md`
- `docs/validation/conv2d_validation_report.md`
- `docs/validation/resnet18_validation_report.md`
- `docs/hardware_characterization_2025-10.md`

**Data**:
- `results/validation/sweep_results.csv` (6 architectures)
- `results/validation/resnet_family_results.csv`

---

## Enhanced Attention Fusion (📋 Planned)

### Current State vs Enhanced

**Current Approach** (limited):
```
LayerNorm → MultiheadAttention (2 ops, 5.7% memory reduction)
```

**Enhanced Approach** (decomposed):
```
LayerNorm → Q/K/V_Proj → Reshape → Transpose → Q@K^T → Scale →
Softmax → Dropout → Scores@V → Transpose → Reshape → Out_Proj → Dropout

15+ operations → 6-8 fused subgraphs (40-60% memory reduction)
```

### Expected Impact

| Model | Current Fusion | Enhanced Fusion | Improvement |
|-------|----------------|-----------------|-------------|
| **ViT-B/16** | 2 ops, 5.7% mem save | 8 groups, 45% mem save | 8× better |
| **Overall** | 1.38× efficiency | 2.0-2.5× efficiency | 1.8× better |

### Implementation Plan (5-6 weeks)

1. **Phase 1: Proof of Concept** (1 week)
   - Create `DecomposedMultiheadAttention` module
   - Validate >30% memory reduction

2. **Phase 2: Custom Tracer** (2 weeks)
   - Implement `DecomposingTracer` class
   - Auto-decompose `nn.MultiheadAttention`
   - Add attention-specific fusion patterns

3. **Phase 3: Parallel Fusion** (1-2 weeks)
   - Fuse parallel operations (Q, K, V projections)
   - Implement `ParallelFusionGroup` concept

4. **Phase 4: Validation** (1 week)
   - Test on all transformer models
   - Benchmark memory and latency

**See**: `docs/ENHANCED_ATTENTION_FUSION_PLAN.md` for details

---

## Phase 2: Hardware Mapping (🚧 In Progress)

### Goals

1. **Map fused subgraphs to hardware resources**:
   - GPU: Map to SM groups (not all 132 SMs!)
   - KPU: Map to tiles
   - TPU: Map to systolic array chunks
   - CPU: Map to cores/vector units

2. **Estimate realistic utilization**:
   - Example: 32 fused subgraphs, max 12 parallel → ~24 SMs active (not 132!)
   - Account for wave quantization
   - Occupancy estimation

3. **Calculate realistic latency**:
   - Effective FLOPS = Peak FLOPS × utilization
   - Memory-bound ops: Use bandwidth model
   - Fused ops: Account for reduced memory traffic

4. **Fix the 1000× latency error**:
   - Target: <30% error vs real benchmarks

### Files to Create

- `src/graphs/characterize/hardware_mapper.py`
- `src/graphs/characterize/gpu_mapper.py`
- `src/graphs/characterize/kpu_mapper.py`
- `src/graphs/characterize/tpu_mapper.py`
- `src/graphs/characterize/cpu_mapper.py`
- `examples/test_hardware_mapping.py`

### Open Questions

1. **SM allocation strategy**: How to distribute fused subgraphs across SMs?
2. **Tile memory constraints**: Do fused subgraphs fit in KPU 256KB scratchpad?
3. **Parallelism mapping**: How to map thread counts to SM allocation?
4. **Latency calculation**: How to combine fused ops' latencies with roofline model?

---

## Documentation & Examples

### User Documentation (4,000+ lines)

1. **GETTING_STARTED.md** (605 lines)
   - 5-minute quick start
   - Understanding the output
   - Key concepts explained
   - 4-week learning path
   - Troubleshooting guide

2. **graph_partitioner_tutorial.md** (605 lines)
   - Tutorial 1: Your first partition
   - Tutorial 2: Subgraph properties
   - Tutorial 3: Understanding concurrency
   - Tutorial 4: Custom validation
   - Tutorial 5: Debugging

3. **graph_partitioner_validation.md** (176 lines)
   - How validation works
   - Universal vs architecture-specific checks
   - Model profiles explained

### Developer Documentation

4. **realistic_performance_modeling_plan.md** (1,600 lines)
   - Phase 1-3 architecture
   - Implementation timeline
   - Component specifications

5. **FUSION_ALGORITHM_PROPOSAL.md**
   - Fusion algorithm design
   - Boundary detection
   - Pattern matching

6. **ENHANCED_ATTENTION_FUSION_PLAN.md**
   - Attention decomposition design
   - 3 implementation approaches
   - Expected results

### Examples & Scripts

**Quick Start**:
```bash
python examples/quick_start_partitioner.py
```

**Model Comparison**:
```bash
python examples/compare_models.py --models resnet18 mobilenet_v2 efficientnet_b0
```

**Fusion Testing**:
```bash
python examples/test_fusion_partitioner.py --model resnet18
python examples/test_fusion_partitioner.py --model mobilenet_v2
```

**Validation**:
```bash
python tests/test_graph_partitioner_general.py resnet18
python tests/test_graph_partitioner_general.py mobilenet_v2 efficientnet_b0
```

---

## Quick Reference

### Key Formulas

**FLOP Estimation**:
```
Convolution:
  FLOPs = Batch × OutChannels × OutHeight × OutWidth ×
          (2 × InChannels × KernelH × KernelW)

Linear:
  FLOPs = Batch × InputDim × OutputDim × 2
```

**Latency Estimation**:
```
Latency = (FLOPs / PeakFLOPS) × SchedulerOverhead ×
          (1 + 0.05 × (Tiles - 1))
```

**Energy Estimation**:
```
Energy = (FLOPs × EnergyPerFLOP + Memory × EnergyPerByte) ×
         (1 + 0.05 × (Tiles - 1))
```

### Common Commands

```bash
# Run characterization (6 architectures)
python src/graphs/scripts/run_characterization.py

# Graph partitioning quick start
python examples/quick_start_partitioner.py

# Fusion partitioning
python examples/test_fusion_partitioner.py --model resnet18

# Model comparison
python examples/compare_models.py --models resnet18 mobilenet_v2

# Validation tests
python src/graphs/validation/test_resnet_family.py

# View results
cat results/validation/sweep_results.csv
cat results/validation/resnet_family_results.csv
```

### Important Files

**Entry Points**:
- `examples/quick_start_partitioner.py` - Start here!
- `src/graphs/scripts/run_characterization.py` - Full sweep

**Core Components**:
- `src/graphs/characterize/graph_partitioner.py` - Graph partitioning
- `src/graphs/characterize/concurrency_analyzer.py` - Concurrency analysis
- `src/graphs/characterize/fusion_partitioner.py` - Fusion-based partitioning
- `src/graphs/characterize/walker.py` - FX graph walker
- `src/graphs/characterize/arch_profiles.py` - Hardware profiles

**Documentation**:
- `docs/GETTING_STARTED.md` - Start here for learning
- `docs/graph_partitioner_tutorial.md` - Hands-on tutorials
- `docs/realistic_performance_modeling_plan.md` - Full architecture

---

## Project Statistics

### Code Volume
- **Source Code**: ~2,400 lines (partitioner, fusion, concurrency)
- **Tests**: ~640 lines (unit tests, validation)
- **Examples**: ~700 lines (demos, comparisons)
- **Documentation**: ~4,000 lines (guides, tutorials, plans)
- **Total**: ~7,700 lines

### Validation Status
- **Test Pass Rate**: 100% on ResNet-18, MobileNet-V2, EfficientNet-B0
- **FLOP Accuracy**: ±6% of theoretical values
- **Coverage**: CNNs (ResNet, MobileNet, EfficientNet), Transformers (ViT)

### Performance Impact
- **Fusion Reduction**: 1.9-2.1× fewer execution units
- **Memory Savings**: 20-42% reduction in global memory traffic
- **Latency Error Identified**: 1000× overly optimistic (now fixing in Phase 2)

---

## Known Limitations

1. **Static shapes only**: Dynamic batch sizes not supported
2. **Limited fusion**: Greedy algorithm may miss optimal fusions
3. **No bandwidth modeling** (Phase 2): Currently assumes compute-bound
4. **Limited operator support**: Conv2d, Linear, activations, norms only
5. **No multi-device**: Pipeline/tensor parallelism not modeled

---

## Roadmap

### ✅ Phase 1: Complete (Weeks 1-2)
- Graph partitioning & concurrency analysis
- Fusion-based partitioning
- Validation framework
- Documentation & examples

### 🚧 Phase 2: In Progress (Weeks 3-4)
- Hardware resource mapping (GPU/KPU/TPU/CPU)
- Realistic utilization estimation
- Corrected latency modeling
- SM/core/tile allocation

### 📋 Phase 3: Planned (Weeks 5-6)
- Memory bandwidth roofline modeling
- Memory-bound operation handling
- Enhanced latency estimation
- Validation against real benchmarks

### 📋 Future Work
- Enhanced attention fusion (Transformers)
- Dynamic batch size support
- Auto-fusion optimization
- Multi-device modeling
- Hardware calibration
- Cost modeling ($/inference)

---

## Conclusion

**Phase 1 Status**: ✅ **Complete and Validated**

The graph characterization and partitioning pipeline provides:

1. **Accurate characterization** of neural network models
2. **Fusion-based partitioning** with 20-42% memory reduction
3. **Multi-level concurrency analysis** (graph + subgraph + hardware)
4. **Foundation for Phase 2** hardware mapping
5. **Production-ready validation** framework

Most importantly, Phase 1 **identified and quantified** the root cause of 1000× latency errors: assuming peak hardware utilization without analyzing actual graph parallelism.

**Phase 2 Focus**: Map 32-66 fused subgraphs to realistic hardware allocation and fix latency estimates.

**Questions?**
- Getting started: `docs/GETTING_STARTED.md`
- Tutorials: `docs/graph_partitioner_tutorial.md`
- Architecture details: `docs/realistic_performance_modeling_plan.md`

---

**For session-by-session progress, see**: `docs/sessions/`
**For daily updates, see**: `CHANGELOG.md`
