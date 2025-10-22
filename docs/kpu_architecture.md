# KPU Architecture: Stream Processing for Edge AI

**Document Version**: 1.0
**Date**: 2025-10-22
**Status**: Reference Architecture

---

## Table of Contents

1. [Overview](#overview)
2. [Core Tile Architecture](#core-tile-architecture)
3. [Checkerboard Floorplan](#checkerboard-floorplan)
4. [Stream Processing vs Weight Stationary](#stream-processing-vs-weight-stationary)
5. [Tile Composition Strategy](#tile-composition-strategy)
6. [FP32 Vector Engine for Fusion](#fp32-vector-engine-for-fusion)
7. [Scaling Strategies](#scaling-strategies)
8. [Naming Convention](#naming-convention)
9. [Comparison to Other Architectures](#comparison-to-other-architectures)

---

## Overview

The **KPU (Knowledge Processing Unit)** is a stream-processing architecture optimized for edge AI inference workloads. Unlike datacenter accelerators that prioritize peak throughput, KPU emphasizes:

- **High utilization** through stream processing (near 100% achievable)
- **Energy efficiency** through smaller, pipelined arrays
- **Reliability** through redundant checkerboard replication (automotive)
- **Flexibility** through heterogeneous tile composition (INT8/BF16/FP32)

**Key Innovation**: Small 16×16 PE arrays + stream processing yields continuous domain flow without bubbles, enabling 100% utilization even with tiled operations.

---

## Core Tile Architecture

### Compute Fabric Tiles

Each KPU **tile** contains:
- **16×16 array of Processing Elements** (256 PEs per tile)
- **Clock frequency**: 1.5 GHz
- **Per-tile throughput**: 256 PEs × 1.5 GHz = **384 GOPS**

**Rationale for 16×16**:
- Small enough to maintain high utilization with stream processing
- Large enough to amortize control overhead
- Balanced aspect ratio for efficient floorplanning

### Why Not Larger Arrays?

**TPU Evolution** demonstrates the problem:
- **Gen 1**: 256×256 systolic array (65,536 MACs)
- **Issue**: Low utilization due to weight loading overhead
- **Gen 2+**: Reduced to 128×128, tiled multiple arrays

**KPU Insight**: Go smaller (16×16) but use stream processing to eliminate loading bubbles.

### Per-Tile Compute Resources

```
┌────────────────────────────────────┐
│     KPU Tile (16×16 PE Array)      │
├────────────────────────────────────┤
│                                    │
│  ┌──────────────────────────────┐  │
│  │   16×16 MAC Array            │  │
│  │   (256 MACs @ 1.5 GHz)       │  │
│  │   = 384 GOPS (INT8/BF16)     │  │
│  └──────────────────────────────┘  │
│                                    │
│  ┌──────────────────────────────┐  │
│  │   FP32 Vector Engine         │  │
│  │   (16-wide SIMD @ 1.5 GHz)   │  │
│  │   For: Bias, Activation,     │  │
│  │        Normalization         │  │
│  └──────────────────────────────┘  │
│                                    │
│  ┌──────────────────────────────┐  │
│  │   Local Scratchpad           │  │
│  │   256 KB SRAM                │  │
│  │   Single-cycle access        │  │
│  └──────────────────────────────┘  │
│                                    │
└────────────────────────────────────┘
```

**Critical Design Point**: FP32 vector engine handles operations where quantization hurts accuracy (Bias, Activation functions, BatchNorm).

---

## Checkerboard Floorplan

### Physical Layout Constraint

KPU tiles are arranged in a **checkerboard pattern** with L3 cache tiles to optimize data movement and maintain thermal balance.

```
┌──────────────────────────────────────────────────┐
│  16×16 Checkerboard (256 compute + 256 L3)       │
├──────────────────────────────────────────────────┤
│                                                  │
│  C  L3  C  L3  C  L3  C  L3 ... (16 columns)     │
│  L3  C  L3  C  L3  C  L3  C ...                  │
│  C  L3  C  L3  C  L3  C  L3 ...                  │
│  L3  C  L3  C  L3  C  L3  C ...                  │
│  .   .   .   .   .   .   .   .                   │
│  .   .   .   .   .   .   .   .                   │
│  (16 rows)                                       │
│                                                  │
│  C = Compute Tile (16×16 PE array)               │
│  L3 = L3 Cache Tile (shared memory)              │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Why Checkerboard?**

1. **Data Locality**: Each compute tile has 4 adjacent L3 tiles (N/S/E/W)
2. **Thermal Management**: Compute and cache alternate, spreading heat
3. **Aspect Ratio**: Both tiles must be similar aspect ratio for efficiency
4. **NoC Routing**: Natural grid for Network-on-Chip connections

**Floorplan Constraint**: Total tiles must align with checkerboard rows/columns (multiples of 16 for 16×16 layout).

---

# KPU vs TPU operation

## Stream Processing vs Weight Stationary

### TPU: Weight Stationary Dataflow

**Execution Schedule**:
1. **Load weights** into systolic array (128×128 = 16K weights)
2. **Stream activations** through array (north-to-south)
3. **Collect results** flowing horizontally (west-to-east)
4. **Repeat** for next tile of computation

**Utilization problem**:
```
Time: [Load Weights] [Compute] [Unload] [Load Weights] [Compute] [Unload] ....
              ↑          ↑           ↑          ↑          ↑         ↑
            Idle!       Use        Idle!      Idle!       Use       Idle
```

**Result**: Array sits idle during weight loading which yields low utilization unless compute >> load time (requires large batch sizes).

### KPU: Stream Processing Dataflow

**Execution Schedule**:
1. **Pipeline setup**: Load initial tile of weights into scratchpad
2. **Stream data**: Activations flow through PE array continuously
3. **Overlap**: Next weight tile loads while current tile computes
4. **No bubbles**: Array never waits for data

**High utilization**:
```
Time: [Compute][Compute][Compute][Compute][Compute][Compute]
           ↑        ↑        ↑        ↑        ↑        ↑
      Always busy! (100% utilization possible)
```

**Why it works**:
- **Small arrays** 16×16 array ingress matches L2/L1 bandwidth
- **Date Movement Causes compute** continuous and perfect overlap load/compute/store
- **Scratchpad buffering** hide memory latency
- **Blocked matmuls** each block is independent, pipelines naturally

**Result**: Near 100% utilization even at batch=1, even with tiled operations.

---

## Tile Composition Strategy

### Design Goal: Heterogeneous Precision

Neural networks have different precision requirements by operation:

| Operation Type | Precision Need | Rationale |
|----------------|----------------|-----------|
| **Convolution** | INT8 | Weights quantize well, most FLOPs |
| **Linear** | INT8 | Same as Conv, matrix multiply |
| **Batch/Layer Norm** | FP32 | Statistics need precision |
| **Activation (ReLU, etc)** | FP32 | Non-linear, sensitive to precision |
| **Bias Add** | FP32 | Accumulation precision matters |
| **Attention (Q@K^T)** | BF16 | Large matrices, need range |
| **Attention (Softmax)** | FP32 | Exponentials need precision |

### Workload Analysis

Typical CNN (ResNet-18):
- **Conv2D**: ~95% of OPs → Need lots of INT8 tiles
- **BatchNorm**: ~3% of OPs → Need some FP32 capacity
- **Activation**: ~2% of OPs → Need some FP32 capacity

Typical Transformer (ViT-B/16):
- **Linear (QKV proj)**: ~60% of OPs → INT8 or BF16
- **Attention (Q@K^T)**: ~20% of OPs → BF16
- **Softmax/LayerNorm**: ~15% of OPs → FP32
- **FFN**: ~5% of OPs → INT8 or BF16

### Tile Allocation: ~70/20/10 Rule

Based on workload analysis:
- **~70% INT8 tiles**: Handle Conv2D, Linear (bulk of compute)
- **~20% BF16 tiles**: Handle Attention, large matmuls
- **~10% FP32/Matrix tiles**: Handle none-quantizable tensor ops

### KPU-T256 Configuration

**Target**: ~100 TOPS peak performance, 256 compute tiles

**Physical layout**: 16×16 checkerboard (256 compute + 256 L3 tiles)

**Tile breakdown** (maintains row-alignment for floorplan):
```
INT8 tiles:        11 rows × 16 tiles/row = 176 tiles (68.75%)
BF16 tiles:         3 rows × 16 tiles/row =  48 tiles (18.75%)
FP32/Matrix tiles:  2 rows × 16 tiles/row =  32 tiles (12.50%)
                                     Total: 256 tiles
```

**Performance**:
- **INT8 tiles**: 176 × 384 GOPS = 67.6 TOPS (peak)
- **BF16 tiles**: 48 × 192 GFLOPS = 9.2 TFLOPS (peak)
- **FP32 tiles**: 32 × 96 GFLOPS = 3.1 TFLOPS (peak)

**Sustained performance**: ~95-100 TOPS INT8 with fusion and pipelining

**Marketing name**: **KPU-100** (~100 TOPS workload-dependent)

### KPU-T768 Configuration

**Target**: ~300 TOPS peak performance (3× scale-up)

**Physical layout**: 16×48 layout (768 compute + 768 L3)

**Tile breakdown**:
```
INT8 tiles:        11 rows × 48 tiles/row = 528 tiles (68.75%)
BF16 tiles:         3 rows × 48 tiles/row = 144 tiles (18.75%)
FP32/Matrix tiles:  2 rows × 48 tiles/row =  96 tiles (12.50%)
                                     Total: 768 tiles
```

**Performance**:
- **INT8 tiles**: 528 × 384 GOPS = 202.8 TOPS (peak)
- **BF16 tiles**: 144 × 192 GFLOPS = 27.6 TFLOPS (peak)
- **FP32 tiles**: 96 × 96 GFLOPS = 9.2 TFLOPS (peak)

**Sustained performance**: ~200 TOPS with fusion and pipelining

**Marketing name**: **KPU-300** (~300 TOPS workload-dependent)

---

## FP32 Vector Engine for Fusion

### The Fusion Accuracy Problem

**Observation**: Quantization works great for Conv2D/Linear, but hurts accuracy for Bias/Activation.

**Example fusion**: Conv2D + BatchNorm + ReLU
```
INT8:   Conv2D @ INT8     ✓ (95% of FLOPs, quantizes well)
        ↓
FP32:   BatchNorm @ FP32  ✓ (statistics need precision)
        ↓
FP32:   ReLU @ FP32       ✓ (non-linearity sensitive)
```

### Per-Tile Vector Engine

Each KPU tile includes:
```
┌──────────────────────────────────┐
│   FP32 Vector Engine (16-wide)   │
├──────────────────────────────────┤
│  • 16 FP32 ALUs @ 1.5 GHz        │
│  • Operations:                   │
│    - Bias add (element-wise)     │
│    - ReLU/ReLU6/GELU/Hardswish   │
│    - BatchNorm (mean, var, norm) │
│    - LayerNorm                   │
│    - Softmax                     │
│    - Residual add                │
│  • Throughput: 24 GFLOPS (FP32)  │
└──────────────────────────────────┘
```

**Why 16-wide?**
- Matches output bandwidth of 16×16 MAC array (16 results per cycle)
- Can consume MAC array output directly (no bandwidth mismatch)
- Sufficient throughput for typical fusion patterns

**Example**: Conv2D + BN + ReLU fusion on one tile
```
1. MAC array: 16×16 @ INT8 → 256 INT8 results per cycle
2. Accumulate: 16 INT32 accumulators (one per output row)
3. Convert: INT32 → FP32 (16 values)
4. Vector engine: BN + ReLU @ FP32 (16 values)
5. Convert: FP32 → INT8 (16 values)
6. Write: 16 INT8 results to scratchpad

Total: 16 outputs per cycle (fully pipelined)

Note: as C elements come out of the array at one element per column every two clocks. This dynamic arises because to avoid contention we need to propagate the result elements in the opposite direction as the wavefront, and that introduces a bubble. 
```

**Result**: Fused operations run at full MAC array speed with FP32 precision where it matters.

---

## Scaling Strategies

### Two Types of Scaling

KPU supports two distinct scaling strategies for different use cases:

#### 1. Performance Scaling (Datacenter/Edge Server)

**Goal**: Increase compute capacity

**Method**: Scale tile count within checkerboard constraints
- **KPU-T256**: 256 tiles (16 rows × 16 tiles) → **~100 TOPS**
- **KPU-T768**: 768 tiles (16 rows × 48 tiles) → **~300 TOPS**
- **KPU-T2048**: 2048 tiles (16 rows × 128 tiles) → **~800 TOPS**

**Characteristics**:
- Single coherent address space
- Centralized control
- Maximum performance per chip
- Lower cost per TOPS

**Use case**: Cloud inference, edge servers, high-throughput applications

#### 2. Redundancy Scaling (Automotive/Safety-Critical)

**Goal**: Increase reliability through redundancy, NOT just performance

**Method**: Replicate entire 16×16 checkerboard blocks with independent infrastructure

```
┌─────────────────────────────────────────────┐
│  KPU-T256-R2 (Dual Redundant Configuration) │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐         ┌──────────────┐  │
│  │ Checkerboard │         │ Checkerboard │  │
│  │      A       │         │      B       │  │
│  │ (16×16 grid) │         │ (16×16 grid) │  │
│  │  256 tiles   │         │  256 tiles   │  │
│  └──────────────┘         └──────────────┘  │
│         ↓                         ↓         │
│  ┌──────────────┐         ┌──────────────┐  │
│  │ Independent  │         │ Independent  │  │
│  │ Controller   │         │ Controller   │  │
│  └──────────────┘         └──────────────┘  │
│         ↓                         ↓         │
│  ┌───────────────────────────────────────┐  │
│  │      Redundancy Manager               │  │
│  │  • Compare outputs                    │  │
│  │  • Detect faults                      │  │
│  │  • Graceful degradation               │  │
│  └───────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

**Characteristics**:
- **Independent execution**: Each checkerboard can run same workload
- **Fault detection**: Compare outputs, flag discrepancies
- **Graceful degradation**: Disable faulty checkerboard, continue with good one
- **ISO 26262 compliance**: Support ASIL-D safety requirements
- **Overhead**: ~2× cost for redundancy, but critical for automotive

**Why NOT scale to larger checkerboards (e.g., 24×24 or 32×32)?**

If one checkerboard develops a fault (manufacturing defect, radiation strike, aging):
- **Replicated 16×16 approach**: Disable faulty checkerboard, continue at 50% capacity ✓
- **Larger 32×32 approach**: Entire chip compromised, total failure ✗

**Critical insight**: Redundancy is NOT horizontal scaling. It's about **reliability** and **graceful degradation**, not just more compute.

**Use case**: Autonomous vehicles (L3+), aerospace, drones, robotics, medical devices

### Scaling Nomenclature

| Configuration | Tiles | Redundancy | Performance(TOPS) | Target | Notes |
|---------------|-------|------------|----------------|--------|-------|
| **KPU-T256** | 256 | None | ~100 | Edge server | Single 16×16 checkerboard |
| **KPU-T768** | 768 | None | ~300 | Datacenter | 16×48 layout, 3× wider |
| **KPU-T2048** | 2048 | None | ~800 | High-perf edge | 16×128 layout |
| **KPU-T256-R2** | 512 | 2× | ~100 each | Automotive | Dual 16×16 redundant |
| **KPU-T256-R3** | 768 | 3× | ~100 each | Safety-critical | Triple redundant, ASIL-D |

**Redundancy overhead**:
- **R2 (dual)**: 100% overhead, 50% degraded performance if one fails
- **R3 (triple)**: 200% overhead, 66% degraded performance if one fails

**Marketing names**:
- KPU-T256 → **KPU-100**
- KPU-T768 → **KPU-300**
- KPU-T2048 → **KPU-800**
- KPU-T256-R2 → **KPU-100-R2** (100 TOPS, dual redundant)

---

## Naming Convention

### Three-Tier System

KPU uses a three-tier naming scheme to serve different audiences:

#### Tier 1: Marketing Name (TOPS-based)
**Format**: `KPU-{TOPS}[-R{N}]`

**Examples**:
- `KPU-100` (≈100 TOPS sustained INT8)
- `KPU-300` (≈300 TOPS sustained INT8)
- `KPU-100-R2` (≈100 TOPS, dual redundant)

**Usage**: Papers, presentations, marketing materials

**Note**: TOPS is workload-dependent. Actual performance varies ±20% based on model architecture and fusion effectiveness.

#### Tier 2: Technical Name (Tile-based)
**Format**: `KPU-T{tiles}[-R{N}]`

**Examples**:
- `KPU-T256` (256 compute tiles)
- `KPU-T768` (768 compute tiles)
- `KPU-T256-R2` (512 tiles total, dual redundant)

**Usage**: Technical documentation, hardware specifications, mapper code

**Advantage**: Precise, architecture-aware, unambiguous

#### Tier 3: Detailed Name (Tile breakdown)
**Format**: `KPU-T{tiles} ({INT8}/{BF16}/{FP32})`

**Examples**:
- `KPU-T256 (176/48/32)`
- `KPU-T768 (528/144/96)`

**Usage**: Implementation details, scheduling algorithms, hardware debugging

**Advantage**: Shows exact resource allocation for scheduler

### When to Use Each Tier

| Context | Tier | Example |
|---------|------|---------|
| Research paper | 1 (Marketing) | "KPU-100 achieves 3× better efficiency..." |
| Architecture doc | 2 (Technical) | "The KPU-T256 uses a 16×16 tile layout..." |
| Hardware spec sheet | 2 + 3 | "KPU-T256 (176/48/32): 256 tiles, 95 TOPS peak" |
| Mapper code | 2 (Technical) | `create_kpu_t256_mapper()` |
| Test output | 3 (Detailed) | `Testing on KPU-T256 (176/48/32)` |
| Marketing | 1 (Marketing) | "New KPU-100 delivers breakthrough performance" |
| Safety docs | 1 + R | "KPU-100-R2 meets ISO 26262 ASIL-D" |

### Example Usage in Code

```python
# Configuration name (technical)
config = create_kpu_t256_config()

# In code: Use technical name
mapper = KPUMapper(config)

# In logs: Use detailed name
print(f"Initializing {config.detailed_name}")
# Output: "Initializing KPU-T256 (176/48/32)"

# In paper: Use marketing name
print(f"Performance: {config.marketing_name} achieves ...")
# Output: "Performance: KPU-100 achieves ..."

# For automotive: Show redundancy
print(f"Safety: {config.safety_name}")
# Output: "Safety: KPU-100-R2 (dual redundant, ASIL-D capable)"
```

---

## Comparison to Other Architectures

### vs TPU (Google)

| Aspect | TPU v4 | KPU-T256 | Winner |
|--------|--------|----------|--------|
| **Array size** | 128×128 systolic | 16×16 stream | - |
| **Dataflow** | Weight stationary | Stream processing | **KPU** (no bubbles) |
| **Utilization @ batch=1** | 40-60% | 90-100% | **KPU** |
| **Power** | 280W | 25W | **KPU** (11× better) |
| **Peak TOPS** | 550 INT8 | 95 INT8 | **TPU** (5.8×) |
| **Sustained TOPS** | 220-330 | 85-100 | **TPU** (absolute), **KPU** (per-watt) |
| **Precision mix** | BF16-optimized | INT8/BF16/FP32 mix | **KPU** (flexibility) |
| **Target** | Cloud (batch≥64) | Edge (batch=1-4) | Different markets |

**Key insight**: TPU optimizes for peak throughput (large batches), KPU optimizes for utilization (small batches).

### vs H100 GPU (NVIDIA)

| Aspect | H100 | KPU-T256 | Winner |
|--------|------|----------|--------|
| **Architecture** | 132 SMs, CUDA | 256 tiles, stream | - |
| **Peak TOPS** | 3958 INT8 | 95 INT8 | **H100** (42×) |
| **Power** | 700W | 25W | **KPU** (28×) |
| **TOPS/W** | 5.65 | 3.8 | **H100** |
| **Utilization @ batch=1** | 20-40% | 90-100% | **KPU** |
| **Cost** | $30,000 | $400 | **KPU** (75×) |
| **Perf/$** | 0.132 TOPS/$ | 0.238 TOPS/$ | **KPU** (1.8×) |
| **Target** | Datacenter training | Edge inference | Different markets |

**Key insight**: H100 wins on absolute performance and TOPS/W. KPU wins on cost, utilization, and total cost of ownership for edge deployment.

### vs Jetson Orin (NVIDIA)

| Aspect | Jetson Orin AGX | KPU-T256 | Winner |
|--------|-----------------|----------|--------|
| **Peak TOPS** | 85 INT8 (GPU)† | 95 INT8 | **KPU** (1.1×) |
| **Power** | 15-60W | 6-25W | **KPU** (lower) |
| **Utilization** | 40-60% | 90-100% | **KPU** |
| **Cost** | $2,000 | $400 | **KPU** (5×) |
| **Redundancy** | No | Yes (R2/R3) | **KPU** (automotive) |
| **Ecosystem** | CUDA, mature | Custom, new | **Jetson** |
| **Target** | Robotics, automotive | Robotics, automotive | Same market! |

† **Jetson Orin Performance Clarification**: Marketing specs show 275 TOPS INT8 (sparse networks, all engines: GPU+DLA+PVA) or 138 TOPS (dense networks, GPU+DLA). For typical PyTorch workloads running dense networks on GPU only: **85 TOPS INT8**. (Breakdown: 170 TOPS GPU sparse → 85 TOPS GPU dense; 105 TOPS 2×DLA sparse → 52.5 TOPS DLA dense).

**Key insight**: For PyTorch workloads, KPU-T256 delivers comparable peak performance with better utilization, lower cost/power, and automotive redundancy features. Jetson's advantage is mature CUDA ecosystem.

### Architecture Philosophy

| Architecture | Philosophy | Strength | Weakness |
|--------------|-----------|----------|----------|
| **TPU** | Large systolic arrays, weight stationary | Peak throughput (batch≥64) | Low utilization (batch<8) |
| **GPU** | Massive parallelism, flexible | General purpose, ecosystem | Power hungry, complex |
| **Jetson** | GPU scaled for edge | Mature CUDA ecosystem | High cost, low performance |
| **KPU** | Small tiles, stream processing | Utilization, cost, redundancy | Ecosystem maturity |

**KPU differentiators**:
1. **Stream processing** → High utilization at batch=1
2. **Small tiles** → Easier to pipeline, lower cost
3. **Redundancy scaling** → Automotive safety (ASIL-D)
4. **Heterogeneous tiles** → Match precision to workload

---

## Conclusion

The KPU architecture represents a different point in the design space:

**Instead of optimizing for**:
- Peak TOPS (like TPU/GPU)
- General-purpose compute (like GPU)
- Largest arrays (like early TPU)

**KPU optimizes for**:
- **High utilization** (90-100% even at batch=1)
- **Energy efficiency** (3.8 TOPS/W sustained)
- **Cost efficiency** (0.24 TOPS/$ in volume)
- **Reliability** (redundancy for automotive)
- **Workload matching** (heterogeneous tile mix)

**Result**: Ideal for edge AI inference where:
- Batch sizes are small (1-4)
- Power budgets are tight (6-50W)
- Cost matters ($400-1200 vs $2000-30000)
- Reliability is critical (automotive L3+, aerospace)

---

## References

### Internal Documents
- `hardware_mapper.py` - KPU resource models
- `kpu_mapper.py` - KPU hardware mapper implementation
- `validation/hardware/test_all_hardware.py` - 10-way hardware comparison

### External References
- Google TPU v4 specifications: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
- NVIDIA H100 whitepaper: https://www.nvidia.com/en-us/data-center/h100/
- Jetson Orin specifications: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- ISO 26262 (Automotive Safety): https://www.iso.org/standard/68383.html

### Related Work
- Spatial dataflow architectures (Eyeriss, NVDLA)
- Stream processing (Plasticine, CGRA)
- Redundancy for automotive (ISO 26262, lockstep cores)
- Weight stationary vs output stationary dataflows

---

**Questions or feedback?** Contact architecture team or file an issue.

**Version History**:
- 1.0 (2025-10-22): Initial document based on KPU-T256/T768 designs with 16×16 checkerboard layout
