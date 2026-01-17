# Hardware Architecture Taxonomy: Execution Models and Programming Paradigms

**Document Version**: 1.0
**Date**: 2025-11-01
**Status**: Reference Architecture

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Classifications](#architectural-classifications)
3. [Programmable ISA Architectures](#programmable-isa-architectures)
4. [Fixed-Function Accelerators](#fixed-function-accelerators)
5. [Reconfigurable Architectures](#reconfigurable-architectures)
6. [Execution Model Comparison](#execution-model-comparison)
7. [Mapper Implementation Strategies](#mapper-implementation-strategies)
8. [Quick Reference Table](#quick-reference-table)

---

## Overview

This document provides a comprehensive taxonomy of the hardware architectures modeled in this repository, organized by their fundamental execution models and programming paradigms. Understanding these distinctions is critical for accurate performance modeling and realistic workload characterization.

### Key Insight

Hardware architectures differ fundamentally in **how they execute computational graphs**:
- **Sequential control flow** (CPU): One instruction stream controlling multiple data
- **Parallel control flow** (GPU): Multiple instruction streams with lockstep execution
- **Dataflow execution** (KPU, CGRA): Data movement drives computation
- **Systolic arrays** (TPU): Weight-stationary dataflow with pipeline propagation

Each paradigm has different performance characteristics, bottlenecks, and optimal workload profiles.

---

## Architectural Classifications

### Flynn's Taxonomy (Extended)

| Architecture | Classification | Control Model | Data Model | Execution Style |
|--------------|----------------|---------------|------------|-----------------|
| **CPU** | MIMD | Multiple independent | Multiple independent | Stored Program |
| **GPU** | SIMT | Single (per warp) | Multiple (lockstep) | Data Parallel |
| **DSP** | VLIW/SIMD | Explicitly parallel | Vector/Scalar | Hybrid |
| **TPU** | Systolic Array | Weight-stationary | Streaming | Dataflow |
| **KPU** | MIMD (Domain Flow) | Data-driven | Streaming | Spatial Dataflow |
| **DPU** | Tile-based VLIW | Per-tile control | Scratchpad | Reconfigurable |
| **CGRA** | Spatial Dataflow | Fabric configuration | Streaming | Reconfigurable |

### Execution Paradigms

#### 1. **Stored Program Machine (von Neumann)**
- **Examples**: CPU (Intel Xeon, AMD EPYC, Ampere Altra)
- **Control**: Sequential instruction stream (MIMD with multi-core)
- **Data**: Unified memory hierarchy (L1/L2/L3/DRAM)
- **Programming**: C/C++/Assembly
- **Characteristic**: Instruction fetches from memory, decoded, executed

#### 2. **SIMT: Single Instruction Multiple Thread**
- **Examples**: GPU (NVIDIA H100, Jetson Orin, AMD MI300)
- **Control**: Warp-level lockstep execution (32 threads/warp)
- **Data**: Thread-private registers + shared memory
- **Programming**: CUDA, HIP, OpenCL
- **Characteristic**: Divergence = serialization, best for data-parallel workloads

#### 3. **VLIW/DSP: Explicitly Parallel Instruction Words**
- **Examples**: DSP (Qualcomm Hexagon, TI C7x)
- **Control**: Compiler-scheduled parallel execution
- **Data**: Vector registers + tensor units
- **Programming**: Intrinsics, auto-vectorized C
- **Characteristic**: Heterogeneous units (scalar/vector/tensor) orchestrated by compiler

#### 4. **Systolic Arrays: Weight-Stationary Dataflow**
- **Examples**: TPU (Google TPU v4), Tensor Cores (NVIDIA), Apple AMX
- **Control**: Weights loaded, activations streamed
- **Data**: Systolic propagation (north→south, west→east)
- **Programming**: XLA, high-level frameworks
- **Characteristic**: High utilization at large batch sizes, pipeline fill/drain overhead

#### 5. **Domain Flow: Data-Driven Spatial Execution**
- **Examples**: KPU (Stillwater KPU-T64/T256/T768)
- **Control**: Data movement drives computation (MIMD tiles)
- **Data**: Near-memory compute with streaming
- **Programming**: Graph compilers
- **Characteristic**: Near-100% utilization even at batch=1, no weight loading bubbles

#### 6. **Reconfigurable Fabrics: Spatial Dataflow**
- **Examples**: CGRA (Plasticine-style), DPU (Xilinx Vitis AI)
- **Control**: Graph mapped to fabric, reconfigured per subgraph
- **Data**: Scratchpad-based, explicit data movement
- **Programming**: High-level synthesis, OpenCL
- **Characteristic**: Reconfiguration overhead, optimal for fixed workloads

---

## Programmable ISA Architectures

### CPU: Multi-Core Stored Program Machine

**Architectural Model**: MIMD (Multiple Instruction, Multiple Data)

**Execution Paradigm**: Sequential control flow with explicit parallelism
- Each core fetches, decodes, and executes instructions independently
- Multi-core provides thread-level parallelism
- SIMD units (AVX-512, NEON) provide data-level parallelism within core
- Memory hierarchy: L1 (per-core) → L2 (per-core/shared) → L3 (shared) → DRAM

**Key Components**:
- **Cores**: 8-192 (consumer: 8-16, datacenter: 64-192)
- **SIMD Width**: 8-16 elements (AVX-512: 16×FP32, ARM NEON: 4×FP32)
- **Special Units**:
  - Intel AMX: 1024 INT8 ops/cycle (16×16 tiles @ 2 GHz = 2 TOPS per core)
  - ARM SVE: Scalable vector length (128-2048 bits)

**Bottlenecks**:
- Memory bandwidth (80-400 GB/s vs 2000 GB/s GPU)
- Branch misprediction on irregular workloads
- Cache thrashing on large working sets

**Best Use Cases**:
- Irregular workloads (sparse matrices, dynamic graphs)
- Small batch sizes (1-4)
- Latency-sensitive applications
- General-purpose compute

**Mapper Strategy**: `CPUMapper`
- Allocate cores based on thread requirements
- Calculate SIMD vectorization efficiency
- Model cache hierarchy (L1/L2/L3 hit rates)
- Account for memory bandwidth limits

**Example Hardware**:
- Intel Xeon Platinum 8480+ (56 cores, AVX-512, AMX)
- AMD EPYC 9754 (128 cores, AVX-512)
- Ampere Altra Max (128 cores, ARM Neoverse N1)

---

### GPU: SIMT Data Parallel Machine

**Architectural Model**: SIMT (Single Instruction, Multiple Thread)

**Execution Paradigm**: Massive data parallelism with lockstep execution
- Streaming Multiprocessors (SMs) execute warps of 32 threads
- All threads in warp execute same instruction (lockstep)
- Thread divergence causes serialization (if/else both branches executed)
- Thousands of concurrent threads (>100K typical)

**Key Components**:
- **SMs**: 20-144 (edge: 20, datacenter: 144)
- **Threads/SM**: 2048 max concurrent
- **Warp Size**: 32 threads (NVIDIA), 64 wavefronts (AMD)
- **Tensor Cores**: Specialized matrix multiply units (FP16/BF16/INT8/FP8)
- **Memory Hierarchy**: L1/Shared (per-SM) → L2 (global) → HBM

**Bottlenecks**:
- Occupancy: Need enough threads to hide latency
- Divergence: Branch divergence serializes execution
- Memory bandwidth: HBM bandwidth (2-3.3 TB/s)
- Kernel launch overhead: ~5-10 µs per kernel

**Best Use Cases**:
- Massive data parallelism (thousands of threads)
- Regular access patterns (coalesced memory)
- Large batch sizes (32+)
- Training and inference

**Mapper Strategy**: `GPUMapper`
- Map threads → warps → SMs
- Account for wave quantization (SM allocation in groups)
- Model occupancy limits (registers, shared memory)
- Add kernel launch overhead for small workloads
- Sequential vs parallel execution modes (batch size dependent)

**Example Hardware**:
- NVIDIA H100 (132 SMs, 16,896 CUDA cores, 528 Tensor Cores)
- NVIDIA Jetson Orin AGX (16 SMs, 2,048 CUDA cores, 64 Tensor Cores)
- AMD MI300X (304 CUs, 19,456 stream processors)

---

### DSP: Heterogeneous Vector/Tensor Processors

**Architectural Model**: VLIW (Very Long Instruction Word) + SIMD

**Execution Paradigm**: Compiler-scheduled parallel execution
- Multiple functional units execute in parallel per cycle
- Compiler extracts instruction-level parallelism (ILP)
- Heterogeneous units: Scalar + Vector + Tensor
- Specialized for signal processing + AI workloads

**Key Components**:
- **Vector Units**: HVX (Hexagon Vector eXtensions) 128-byte vectors
- **Tensor Units**: Dedicated AI accelerators (INT8/INT16)
- **Scalar Units**: Control flow and irregular operations
- **Streaming I/O**: Low-latency sensor integration

**Bottlenecks**:
- Compiler efficiency (ILP extraction)
- Memory bandwidth (integrated SoC)
- Thermal limits (mobile/automotive)

**Best Use Cases**:
- Sensor fusion (camera + radar + LiDAR)
- Always-on AI (mobile, IoT)
- Automotive ADAS
- Edge inference

**Mapper Strategy**: `DSPMapper`
- Route operations to vector vs tensor units
- Model heterogeneous compute (different ops/cycle)
- Account for memory bandwidth sharing (SoC integration)
- Thermal throttling (DVFS)

**Example Hardware**:
- Qualcomm Hexagon 698 (15 TOPS INT8, QRB5165)
- TI C7x DSP (TDA4VM: 8 TOPS, TDA4VH: 32 TOPS)
- CEVA NeuPro-M (20 TOPS INT8, IP core)

---

## Fixed-Function Accelerators

### TPU: Systolic Array Matrix Engines

**Architectural Model**: Weight-Stationary Systolic Array

**Execution Paradigm**: Pipelined matrix multiplication
- **Setup**: Load weights into systolic array (128×128 typical)
- **Stream**: Feed activations north→south
- **Collect**: Results flow west→east
- **Repeat**: Next weight tile (array sits idle during loading)

**Key Components**:
- **MXUs**: Matrix Multiplier Units (2 per TPU v4)
- **Systolic Array**: 128×128 = 16,384 MACs per MXU
- **Pipeline Depth**: 128 cycles (array fills before first output)
- **Vector Units**: Element-wise operations (ReLU, Add) ~10% of systolic throughput

**Bottlenecks**:
- **Low batch sizes**: Array sits idle during weight loading
- **Pipeline fill/drain**: 128-cycle overhead per operation
- **Utilization**: 10-20% at batch=1, 80-90% at batch=64+

**Best Use Cases**:
- Large batch training (batch ≥ 64)
- Matrix-heavy workloads (transformers, MLPs)
- Cloud inference (batching multiple requests)

**Mapper Strategy**: `TPUMapper`
- Route matrix ops to systolic arrays
- Route element-wise ops to vector units
- Model batch size scaling (higher batch = higher utilization)
- Add pipeline fill/drain overhead
- Account for MXU allocation (1-2 MXUs per operation)

**Example Hardware**:
- Google TPU v4 (2 MXUs, 275 TFLOPS BF16)
- Google Coral Edge TPU (4 TOPS INT8, edge variant)
- Apple ANE (Neural Engine, systolic arrays)

---

### KPU: Domain Flow Spatial Architecture

**Architectural Model**: MIMD (Domain Flow)

**Execution Paradigm**: Data-driven spatial computation
- **Stream Processing**: Continuous data flow through tile arrays
- **No Weight Loading Bubbles**: Overlap compute and data movement
- **Small Tiles**: 16×16 PE arrays (256 PEs/tile)
- **High Utilization**: Near 100% even at batch=1

**Key Innovation**: Stream processing eliminates idle time
```
TPU:  [Load Weights] [Compute] [Unload] [Load Weights] [Compute] [Unload]
             ↓           ↓         ↓          ↓           ↓         ↓
           Idle!        Use      Idle!      Idle!        Use      Idle!

KPU:  [Compute][Compute][Compute][Compute][Compute][Compute]
           ↓        ↓        ↓        ↓        ↓        ↓
      Always busy! (100% utilization possible)
```

**Key Components**:
- **Tiles**: 64-768 tiles (T64/T256/T768)
- **Heterogeneous Mix**: 70% INT8, 20% BF16, 10% FP32/Matrix
- **Per-Tile Resources**:
  - 16×16 MAC array (384 GOPS @ 1.5 GHz)
  - FP32 vector engine (16-wide SIMD)
  - 256KB scratchpad (single-cycle access)

**Bottlenecks**:
- Tiling overhead (if data doesn't fit in 256KB scratchpad)
- NoC bandwidth (tile-to-tile communication)

**Best Use Cases**:
- Low batch inference (batch=1-4)
- Edge AI (battery-powered robots, drones)
- Automotive (real-time perception)
- Embodied AI (cost/power/utilization critical)

**Mapper Strategy**: `KPUMapper`
- Analyze tiling requirements (scratchpad constraints)
- Allocate heterogeneous tiles (INT8 vs BF16 vs FP32)
- Model stream processing (continuous utilization)
- Account for tiling overhead (iterations)

**Example Hardware**:
- Stillwater KPU-T64 (64 tiles, 64 TOPS INT8, 6W)
- Stillwater KPU-T256 (256 tiles, 255 TOPS INT8, 30W)
- Stillwater KPU-T768 (768 tiles, 130 TOPS INT8, 60W)

---

## Reconfigurable Architectures

### DPU: FPGA-Based Tile Processors

**Architectural Model**: Reconfigurable Tile Array

**Execution Paradigm**: AIE (Adaptive Intelligence Engine) tiles
- **Tiles**: 64 AIE tiles (estimate for B4096)
- **Scratchpad**: 64KB per tile
- **Reconfigurable**: FPGA fabric allows custom operations
- **Tiling Required**: Operations partitioned to fit in scratchpad

**Key Components**:
- **AIE Tiles**: 64 tiles @ 1.25 GHz
- **MACs**: 4,096 MACs total (B4096 configuration)
- **Precision**: Native INT8 (best performance)
- **Configuration**: One-time FPGA bitstream load (~100ms)

**Bottlenecks**:
- Lower peak performance (7.68 TOPS vs 100+ TOPS)
- Tiling overhead (64KB scratchpad constraint)
- FPGA power consumption

**Best Use Cases**:
- Custom operators (not in standard libraries)
- Inference at edge (moderate performance)
- Mixed-precision workloads
- Prototyping ASICs

**Mapper Strategy**: `DPUMapper`
- Tile-based scratchpad analysis (64KB constraint)
- Model tiling overhead (iterations)
- Account for quantization benefits (INT8 native)

**Example Hardware**:
- Xilinx Versal AI Edge VEK280 (B4096 DPU)
- Xilinx Kria KV260 (B1024/B2304 DPU)

---

### CGRA: Coarse-Grained Reconfigurable Arrays

**Architectural Model**: Spatial Dataflow Fabric

**Execution Paradigm**: Entire subgraph mapped to fabric
- **Spatial Mapping**: Operations placed on PCUs (Pattern Compute Units)
- **Reconfiguration**: Fabric reconfigured for each new subgraph pattern
- **Greedy Place-and-Route**: NP-hard problem, conservative approximation
- **Reconfiguration Overhead**: ~1000 cycles per subgraph

**Key Components**:
- **PCUs**: 32 Pattern Compute Units (medium-grained)
- **Interconnect**: Crossbar switches between PCUs
- **Configuration**: Per-subgraph fabric reconfiguration
- **Memory**: Distributed scratchpads (per-PCU)

**Bottlenecks**:
- Reconfiguration overhead (1000 cycles × #subgraphs)
- Place-and-route efficiency (greedy heuristic)
- Fabric underutilization (spatial efficiency <100%)

**Best Use Cases**:
- Fixed inference graphs (minimal reconfiguration)
- Custom dataflow patterns
- Research (architecture exploration)

**Mapper Strategy**: `CGRAMapper`
- Greedy place-and-route algorithm
- Model reconfiguration overhead (per-subgraph)
- Account for critical path and parallel width
- Partition large subgraphs (fabric overflow)

**Example Hardware**:
- Stanford Plasticine (research prototype)
- SambaNova reconfigurable dataflow units
- Cerebras wafer-scale engine (spatial mapping)

---

## Execution Model Comparison

### Temporal vs Spatial Execution

| Characteristic | Temporal (CPU/GPU/TPU) | Spatial (KPU/CGRA) |
|----------------|------------------------|---------------------|
| **Control** | Sequential instruction issue | Data-driven execution |
| **Reuse** | Same hardware, different data | Different hardware per operation |
| **State** | Stateful (registers, caches) | Stateless (streaming) |
| **Overhead** | Instruction fetch, decode | Reconfiguration (CGRA) |
| **Utilization** | Varies (10-80%) | Near-constant (80-100%) |
| **Best for** | Dynamic workloads | Fixed inference graphs |

### Parallelism Hierarchy

```
Level 0: Instruction-Level (ILP)
  - CPU SIMD: 8-16 elements in parallel
  - DSP VLIW: 4-8 instructions in parallel

Level 1: Thread-Level (TLP)
  - CPU: 2-4 threads/core (SMT)
  - GPU: 2,048 threads/SM (warps)
  - KPU: 256 threads/tile (stream)

Level 2: Multiprocessor-Level
  - CPU: 8-192 cores
  - GPU: 20-144 SMs
  - KPU: 64-768 tiles
  - TPU: 2 MXUs

Level 3: Graph-Level (Task Parallelism)
  - All: Concurrent kernel/subgraph execution
```

### Memory Hierarchy Comparison

| Architecture | L1/Scratchpad | L2/Shared | L3/Global | Main Memory | Bandwidth |
|--------------|---------------|-----------|-----------|-------------|-----------|
| **CPU** | 32KB (per-core) | 512KB-2MB | 32-256MB | 128-512GB (DDR5) | 80-400 GB/s |
| **GPU** | 128KB (per-SM) | 40-60MB | - | 40-80GB (HBM) | 2-3.3 TB/s |
| **TPU** | - | 32-96MB | - | 16-32GB (HBM) | 1.2-2.4 TB/s |
| **KPU** | 256KB (per-tile) | 16-32MB | - | 8-64GB | 128-512 GB/s |
| **DPU** | 64KB (per-tile) | - | - | SoC shared | 50-100 GB/s |
| **DSP** | Vectors | Shared (SoC) | - | SoC shared | 30-80 GB/s |

---

## Mapper Implementation Strategies

### Common Patterns

#### 1. **Thread-to-Hardware Mapping**

**CPU**: `threads → cores × SIMD_width`
```python
cores_needed = ceil(threads / (SIMD_width × threads_per_core))
cores_allocated = min(cores_needed, total_cores)
vectorization_efficiency = threads / (cores_allocated × SIMD_width)
```

**GPU**: `threads → warps → SMs`
```python
warps_needed = ceil(threads / warp_size)
SMs_needed = ceil(warps_needed / warps_per_SM)
SMs_allocated = min(SMs_needed, total_SMs)
occupancy = SMs_allocated / total_SMs
```

**KPU**: `threads → tiles`
```python
tiles_needed = ceil(threads / threads_per_tile)
tiles_allocated = min(tiles_needed, total_tiles)
# Also constrained by tiling (scratchpad constraints)
tiles_needed = max(tiles_needed, tiles_per_iteration_from_tiling)
```

#### 2. **Roofline Performance Model**

All mappers use roofline model for latency estimation:
```python
# Compute-bound latency
compute_time = ops / (peak_ops_per_sec × occupancy)

# Memory-bound latency
memory_time = bytes_transferred / memory_bandwidth

# Actual latency (max of both)
latency = max(compute_time, memory_time)

# Bottleneck classification
if compute_time > memory_time:
    bottleneck = COMPUTE_BOUND
else:
    bottleneck = MEMORY_BOUND
```

#### 3. **Energy Modeling**

Three-component energy model (all mappers):
```python
# Dynamic compute energy
compute_energy = ops × energy_per_op

# Dynamic memory energy
memory_energy = bytes_transferred × energy_per_byte

# Static/leakage energy (modern nanoscale chips)
idle_power = TDP × 0.5  # 50% leakage
idle_energy = idle_power × latency

# Total energy
total_energy = compute_energy + memory_energy + idle_energy
```

#### 4. **Precision-Aware Performance**

Different precisions have different throughput:
```python
# Peak ops/sec by precision
FP64: 1× base
FP32: 2× base (typical)
FP16/BF16: 4× base (Tensor Cores)
INT8: 8× base (Tensor Cores + quantization)
INT4: 16× base (new architectures)
FP8: 4-8× base (H100+)

# Energy scales inversely
FP64: 4× energy/op
FP32: 2× energy/op
FP16/BF16: 1× energy/op
INT8: 0.5× energy/op
```

---

## Quick Reference Table

### Architecture Selection Guide

| Requirement | Best Architecture | Rationale |
|-------------|-------------------|-----------|
| **Batch=1, low latency** | KPU, CPU | High utilization at low batch |
| **Batch=64+, max throughput** | GPU, TPU | Designed for massive parallelism |
| **Fixed graphs, edge inference** | DPU, CGRA | Reconfigurable, power efficient |
| **Dynamic graphs, branching** | CPU | Flexible control flow |
| **Large matmuls (transformers)** | TPU, GPU Tensor Cores | Systolic arrays excel |
| **Sensor fusion, ADAS** | DSP | Heterogeneous compute + I/O |
| **Battery-powered (3-10W)** | KPU-T64, DSP | Highest utilization/watt |
| **Automotive safety (ASIL-D)** | KPU-R2, DSP | Redundancy support |
| **Datacenter training** | GPU (H100, MI300) | Ecosystem + raw performance |
| **Datacenter inference** | TPU v4, GPU, KPU-T768 | Batch throughput |
| **Cost-sensitive edge** | KPU, Coral TPU | Best TOPS/$ |

### Mapper Summary

| Mapper | Primary File | Key Algorithm | Complexity |
|--------|--------------|---------------|------------|
| `CPUMapper` | `cpu.py` | Core allocation + SIMD vectorization | Medium |
| `GPUMapper` | `gpu.py` | Thread→Warp→SM + wave quantization | High |
| `TPUMapper` | `tpu.py` | Systolic array + batch scaling | Medium |
| `KPUMapper` | `kpu.py` | Tile allocation + tiling analysis | High |
| `DPUMapper` | `dpu.py` | Tile allocation + scratchpad tiling | Medium |
| `DSPMapper` | `dsp.py` | Vector/tensor routing | Medium |
| `CGRAMapper` | `cgra.py` | Greedy place-and-route | High |

---

## References

### Academic Papers

**CPU Architecture**:
- Hennessy & Patterson, "Computer Architecture: A Quantitative Approach" (6th ed.)
- Intel® 64 and IA-32 Architectures Optimization Reference Manual

**GPU Architecture**:
- NVIDIA, "NVIDIA Hopper Architecture In-Depth" (2022)
- NVIDIA, "CUDA C++ Programming Guide" (latest)

**Systolic Arrays**:
- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017)
- Google, "Cloud TPU System Architecture" (documentation)

**Domain Flow**:
- Stillwater Supercomputing, "Domain Flow and Streaming Architectures" (dissertation)
- Stillwater, "KPU Architecture Specification" (internal)

**CGRA**:
- Prabhakar et al., "Plasticine: A Reconfigurable Architecture for Parallel Patterns" (ISCA 2017)

### Internal Documentation

- `docs/kpu_architecture.md` - KPU stream processing details
- `docs/hardware/jetson_specifications.md` - NVIDIA Jetson GPU specifications
- `src/graphs/hardware/resource_model.py` - Base hardware abstraction
- `src/graphs/hardware/mappers/` - All mapper implementations

### External Resources

- NVIDIA Developer Documentation: https://docs.nvidia.com/
- Google Cloud TPU Documentation: https://cloud.google.com/tpu/docs
- Xilinx Vitis AI Documentation: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
- ARM Developer: https://developer.arm.com/

---

## Document Maintenance

**Last Updated**: 2025-11-01
**Maintainer**: Architecture Team
**Review Cycle**: Quarterly or when new architectures added

**Change Log**:
- 2025-11-01: Initial version covering all 7 architecture classes
- Next review: 2026-02-01

---

**Questions or corrections?** Please file an issue or contact the architecture team.
