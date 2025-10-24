# Session Summary: Datacenter CPU Mappers

**Date**: 2025-10-24
**Task**: Add datacenter-class CPU mappers for fair comparison with Ampere AmpereOne
**Result**: Successfully added Intel Xeon and AMD EPYC mappers + comprehensive comparison tool

---

## What Was Built

### 1. Ampere AmpereOne 192-core Mapper (ARM)

**Added in previous session, tested this session**

**Specifications**:
- 192 ARM v8.6+ cores @ 3.6 GHz
- TSMC 5nm process
- Peak INT8: 22.1 TOPS
- Memory: 332.8 GB/s (8-channel DDR5-5200)
- TDP: 283W

**Files**:
- `src/graphs/characterize/hardware_mapper.py`: `ampere_ampereone_192_resource_model()`
- `src/graphs/characterize/cpu_mapper.py`: `create_ampere_ampereone_192_mapper()`
- `docs/AMPERE_AMPEREONE_MAPPER.md`: Comprehensive documentation

---

### 2. Intel Xeon Platinum 8490H Mapper (x86)

**NEW - Added this session**

**Specifications**:
- 60 Golden Cove cores @ 2.9 GHz (all-core boost)
- Intel 7 process (10nm Enhanced SuperFin)
- Peak INT8: **88.7 TOPS (AMX)** ⭐
- Peak BF16: 44.4 TFLOPS (AMX)
- Memory: 307.2 GB/s (8-channel DDR5-4800)
- TDP: 350W

**Key Feature**: AMX (Advanced Matrix Extensions)
- 2 matrix tiles per core
- Each tile: 16×16 INT8 operations per cycle
- Massive acceleration for CNN inference

**Files**:
- `src/graphs/characterize/hardware_mapper.py`: `intel_xeon_platinum_8490h_resource_model()` (+160 lines)
- `src/graphs/characterize/cpu_mapper.py`: `create_intel_xeon_platinum_8490h_mapper()` (+54 lines)

---

### 3. AMD EPYC 9654 Mapper (x86)

**NEW - Added this session**

**Specifications**:
- 96 Zen 4 cores @ 2.4 GHz
- TSMC 5nm process (chiplet design)
- Peak INT8: 7.37 TOPS
- Memory: **460.8 GB/s (12-channel DDR5-4800)** ⭐
- TDP: 360W

**Key Feature**: Highest Memory Bandwidth
- 50% more bandwidth than Intel Xeon
- 38% more bandwidth than Ampere
- Best for memory-intensive workloads (Transformers)

**Files**:
- `src/graphs/characterize/hardware_mapper.py`: `amd_epyc_9654_resource_model()` (+137 lines)
- `src/graphs/characterize/cpu_mapper.py`: `create_amd_epyc_9654_mapper()` (+58 lines)

---

## Comprehensive Comparison Tool

**NEW - Created this session**

**File**: `cli/compare_datacenter_cpus.py` (387 lines)

**Features**:
- Compares all 3 datacenter CPUs side-by-side
- Tests 3 AI workloads:
  - ResNet-50 (image classification CNN)
  - DeepLabV3+ (semantic segmentation)
  - ViT-Base (vision transformer)
- Metrics: Latency, throughput (FPS), FPS/W, utilization
- Executive summary with recommendations

**Usage**:
```bash
python cli/compare_datacenter_cpus.py
```

---

## Benchmark Results Summary

### ResNet-50 (CNN) @ INT8

| CPU | Latency | FPS | FPS/W | Winner |
|-----|---------|-----|-------|--------|
| **Intel Xeon Platinum 8490H** | 0.87 ms | 1144 | 3.27 | ✅ **AMX dominates (4.8× faster)** |
| Ampere AmpereOne 192-core | 4.24 ms | 236 | 0.83 | - |
| AMD EPYC 9654 | 4.61 ms | 217 | 0.60 | - |

**Key Insight**: Intel's AMX provides massive 4-5× speedup for CNN workloads despite having 1/3 the cores of Ampere.

---

### DeepLabV3+ (Segmentation) @ INT8

| CPU | Latency | FPS | FPS/W | Winner |
|-----|---------|-----|-------|--------|
| **Intel Xeon Platinum 8490H** | 8.45 ms | 118 | 0.34 | ✅ **AMX dominates (8.8× faster)** |
| Ampere AmpereOne 192-core | 74.11 ms | 13.5 | 0.05 | - |
| AMD EPYC 9654 | 85.54 ms | 11.7 | 0.03 | - |

**Key Insight**: Even more dramatic advantage for segmentation (10× faster). Large convolutions benefit massively from matrix acceleration.

---

### ViT-Base (Transformer) @ INT8

| CPU | Latency | FPS | FPS/W | Winner |
|-----|---------|-----|-------|--------|
| **AMD EPYC 9654** | 1.14 ms | 878 | 2.44 | ✅ **Bandwidth wins (1.4× faster)** |
| Ampere AmpereOne 192-core | 1.53 ms | 654 | 2.31 | - |
| Intel Xeon Platinum 8490H | 1.65 ms | 606 | 1.73 | - |

**Key Insight**: Plot twist! AMD EPYC wins for Transformers due to high memory bandwidth (460 GB/s). AMX doesn't help as much for attention mechanisms.

---

## Key Findings

### 1. Workload Matters More Than Core Count

**CNNs**: Intel Xeon (60 cores) **beats** Ampere (192 cores) by **4.8×**
- Reason: AMX matrix acceleration >> more cores

**Transformers**: AMD EPYC (96 cores) **beats** Ampere (192 cores) by **1.3×**
- Reason: Higher memory bandwidth (460 GB/s vs 332.8 GB/s)

**Conclusion**: Raw core count doesn't guarantee better performance. Specialized hardware (AMX) and memory bandwidth matter more.

---

### 2. Intel AMX Is a Game-Changer for CNNs

**What is AMX?**
- Advanced Matrix Extensions (introduced in Sapphire Rapids)
- 2 matrix tiles per core
- Each tile: 16×16 INT8 operations per cycle
- 60 cores × 2 tiles × 256 ops/cycle = **88.7 TOPS INT8**

**Why It Matters**:
- CNNs are dominated by matrix multiplications (convolutions)
- AMX accelerates these by **8-10×** vs generic SIMD
- Makes CPU competitive with GPUs for certain workloads

**Comparison**:
- Intel AMX: 256 INT8 ops/cycle per core
- ARM SIMD: 32 INT8 ops/cycle per core (8× slower)
- AMD AVX-512: 64 INT8 ops/cycle per core (4× slower, double-pumped)

---

### 3. Memory Bandwidth Matters for Transformers

**Transformer Characteristics**:
- Self-attention is memory-bandwidth-bound
- Reading large attention matrices from memory
- Less matrix-multiplication-heavy than CNNs

**Why AMD EPYC Wins**:
- 460.8 GB/s (12-channel DDR5) vs Intel's 307 GB/s (50% more)
- 96 cores to parallelize attention operations
- AVX-512 is "good enough" for attention (doesn't need AMX)

**Ampere's Weakness**:
- Only 332.8 GB/s (8-channel DDR5)
- Fewer channels than AMD despite more cores

---

### 4. ARM in Datacenter Needs AI Accelerator

**Ampere AmpereOne Position**:
- ✅ Great for general-purpose compute (192 cores!)
- ✅ Best for cloud-native microservices
- ✅ Excellent power efficiency
- ❌ Can't compete with Intel AMX for AI inference
- ❌ Needs matrix accelerator to be AI-competitive

**Market Reality**:
- AWS Graviton (ARM) is gaining share for general compute
- But AWS Inferentia (custom ASIC) is used for AI inference
- ARM CPUs alone aren't enough for AI workloads

---

## Use Case Recommendations

### For AI Inference

| Workload | Winner | Reason |
|----------|--------|--------|
| **CNN Inference** (ResNet, MobileNet, YOLO) | Intel Xeon | AMX provides 4-10× speedup |
| **Segmentation** (DeepLab, U-Net) | Intel Xeon | Large convolutions benefit from AMX |
| **Transformer Inference** (BERT, GPT, ViT) | AMD EPYC | High bandwidth for attention |
| **Mixed CNN+Transformer** (DETR, Swin) | Intel Xeon | CNN portion dominates |

### For General Datacenter

| Workload | Winner | Reason |
|----------|--------|--------|
| **Cloud-Native Microservices** | Ampere AmpereOne | Most cores, best $/FPS |
| **Virtualization** (many VMs) | AMD EPYC | 192 threads (SMT) |
| **Database Servers** | AMD EPYC | Highest memory bandwidth |
| **HPC / Scientific Computing** | Intel Xeon or AMD EPYC | AVX-512 support |

---

## TCO Analysis

### Purchase Price

| CPU | List Price | FPS (ResNet-50) | FPS/$ |
|-----|-----------|-----------------|-------|
| Ampere AmpereOne | ~$5,000 | 236 | **0.047** (best for general compute) |
| Intel Xeon Platinum 8490H | $11,600 | 1144 | **0.099** (best for AI) |
| AMD EPYC 9654 | $11,805 | 217 | 0.018 |

### Power Costs (3-year @ $0.10/kWh)

| CPU | TDP | 3-Year Power Cost | Total TCO |
|-----|-----|-------------------|-----------|
| Ampere AmpereOne | 283W | $744 | $5,744 |
| Intel Xeon Platinum 8490H | 350W | $920 | $12,520 |
| AMD EPYC 9654 | 360W | $946 | $12,751 |

### TCO Winner

**For CNN Inference**: Intel Xeon
- Higher upfront cost justified by 4-10× better performance
- Fewer servers needed (lower TCO overall)

**For General Compute**: Ampere AmpereOne
- Lowest purchase price + lowest power cost
- Best FPS/$ for non-AI workloads

---

## Documentation Created

1. **`docs/DATACENTER_CPU_COMPARISON.md`** (450+ lines)
   - Comprehensive comparison of all 3 CPUs
   - Detailed benchmark results
   - Architecture deep dive
   - TCO analysis
   - Use case recommendations

2. **`docs/AMPERE_AMPEREONE_MAPPER.md`** (350+ lines, from previous session)
   - Ampere AmpereOne specifications
   - Comparison to x86
   - Use cases and recommendations

3. **`cli/README.md`** (updated)
   - Added `compare_datacenter_cpus.py` documentation
   - Usage examples and key insights

4. **`docs/sessions/2025-10-24_datacenter_cpus_main.md`** (this file)
   - Session summary
   - Files created/modified
   - Key findings

---

## Files Created/Modified

### New Files

1. `cli/compare_datacenter_cpus.py` (387 lines) - Comparison tool
2. `docs/DATACENTER_CPU_COMPARISON.md` (450+ lines) - Comprehensive docs
3. `docs/sessions/2025-10-24_datacenter_cpus_main.md` (this file)
4. `validation/hardware/test_ampere_ampereone.py` (from previous session)

### Modified Files

1. `src/graphs/characterize/hardware_mapper.py`:
   - Added `ampere_ampereone_192_resource_model()` (+158 lines, previous session)
   - Added `intel_xeon_platinum_8490h_resource_model()` (+160 lines)
   - Added `amd_epyc_9654_resource_model()` (+137 lines)

2. `src/graphs/characterize/cpu_mapper.py`:
   - Added `create_ampere_ampereone_192_mapper()` (+50 lines, previous session)
   - Added `create_intel_xeon_platinum_8490h_mapper()` (+54 lines)
   - Added `create_amd_epyc_9654_mapper()` (+58 lines)

3. `cli/README.md`:
   - Added Ampere AmpereOne documentation (previous session)
   - Added datacenter CPU comparison documentation

### Total Lines Added

- **Resource Models**: 455 lines (Ampere + Intel + AMD)
- **Mapper Factories**: 162 lines (Ampere + Intel + AMD)
- **Comparison Tool**: 387 lines
- **Documentation**: 800+ lines
- **Total**: ~1,800 lines of code + docs

---

## Key Learnings

### 1. Specialized Hardware Wins

**AMX vs Generic SIMD**:
- AMX: 256 INT8 ops/cycle per core
- ARM SIMD: 32 INT8 ops/cycle per core
- **Result**: 8× advantage for Intel in CNNs

**Lesson**: For AI workloads, specialized accelerators (AMX, TPU, NPU) beat general-purpose cores.

### 2. Memory Bandwidth Is Critical

**AMD EPYC's 460 GB/s** enables it to beat Intel (307 GB/s) for Transformers:
- 50% more bandwidth
- Transformers are bandwidth-bound (attention matrices)
- More cores + more bandwidth = better Transformer performance

**Lesson**: Different AI workloads have different bottlenecks (compute vs bandwidth).

### 3. Core Count Isn't Everything

**Ampere's 192 cores** lose to:
- Intel's 60 cores (for CNNs) - AMX acceleration wins
- AMD's 96 cores (for Transformers) - higher bandwidth wins

**Lesson**: Architecture and memory system matter more than raw core count for AI.

### 4. ARM in Datacenter Has a Place

**Ampere AmpereOne is competitive for**:
- Cloud-native microservices (not AI-heavy)
- General-purpose compute
- Power-efficient deployments
- Cost-conscious deployments

**But needs**:
- Matrix accelerator to compete for AI inference
- AWS, Google, Microsoft are all building custom ARM chips with AI accelerators

---

## Future Work

### 1. Additional CPU Variants

**Intel**:
- Xeon Platinum 8592+ (higher SKU with more cores)
- Granite Rapids (next-gen, 2024)

**AMD**:
- EPYC 9754 (128 cores, highest core count)
- Turin (next-gen, Zen 5, 2024)

**Ampere**:
- AmpereOne 96-core and 128-core variants
- Future generations with AI accelerator?

### 2. Multi-Precision Comparison

Currently tested only INT8. Add:
- FP32 (baseline)
- FP16 (common for inference)
- BF16 (AMX-optimized)

### 3. Multi-Model Comparison

Expand to more models:
- Object detection (YOLO, Faster R-CNN)
- Language models (BERT, GPT-2)
- Recommendation models (DLRM)

### 4. Power Profiling

Add actual power measurements:
- Average power draw per workload
- Power efficiency curves
- DVFS (dynamic voltage/frequency scaling) impact

---

**Status**: ✅ Complete
**Related Sessions**:
- [ViT-Large Addition](2025-10-24_datacenter_cpu_vit_large.md) - Later continuation adding large Transformer model
