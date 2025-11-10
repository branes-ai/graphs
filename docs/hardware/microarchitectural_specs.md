# Comprehensive Micro-Architecture Specifications

**Last Updated:** 2025-11-10
**Purpose:** Complete micro-architectural specifications for all hardware platforms with mappers in the graphs project

This document provides detailed micro-architecture specifications for all SoCs and accelerators supported by the hardware mappers in this project. Specifications include core counts, cache hierarchies, memory subsystems, clock frequencies, process nodes, TDP, and all relevant performance characteristics.

---

## Table of Contents

1. [Datacenter CPUs](#datacenter-cpus)
2. [Datacenter & Edge GPUs](#datacenter--edge-gpus)
3. [TPU Accelerators](#tpu-accelerators)
4. [DSP & Automotive SoCs](#dsp--automotive-socs)
5. [Specialized Accelerators](#specialized-accelerators)
6. [Summary Comparison Tables](#summary-comparison-tables)

---

## Datacenter CPUs

### Intel Xeon Platinum 8490H (Sapphire Rapids)

**File:** `src/graphs/hardware/models/datacenter/intel_xeon_platinum_8490h.py`

#### Core Configuration
- **Cores:** 60 Golden Cove cores
- **Threads:** 120 (HyperThreading/SMT enabled, 2 threads per core)
- **Architecture:** Monolithic die design

#### Cache Hierarchy
- **L1 Data:** 48 KB per core (2.9 MB total)
- **L1 Instruction:** 32 KB per core (1.9 MB total)
- **L2:** 2 MB per core (120 MB total)
- **L3 (LLC):** 112.5 MB shared

#### Clock Frequencies
- **Base:** 2.0 GHz
- **Single-core boost:** 3.5 GHz
- **All-core boost:** 2.9 GHz (sustained)

#### Memory Subsystem
- **Technology:** DDR5-4800
- **Channels:** 8
- **Bandwidth:** 307.2 GB/s (8 × 38.4 GB/s)
- **Capacity:** Up to 4TB

#### SIMD Units
- **AVX-512:** 16 FP32 ops/cycle per core
- **AVX-512 FP16:** 32 ops/cycle per core
- **AMX (Advanced Matrix Extensions):** 2 tiles per core, 16×16 INT8 (256 ops per tile)
- **Peak FP32:** 2.78 TFLOPS @ 2.9 GHz
- **Peak INT8 (AMX):** 88.7 TOPS

#### Process & Power
- **Process Node:** Intel 7 (10nm Enhanced SuperFin)
- **TDP:** 350W
  - Idle: ~80W
  - Dynamic: ~270W

#### Additional Features
- PCIe 5.0: 80 lanes
- CXL 1.1 support
- VNNI (Vector Neural Network Instructions)
- Deep Learning Boost

---

### Intel Xeon Platinum 8592+ (Sapphire Rapids Flagship)

**File:** `src/graphs/hardware/models/datacenter/intel_xeon_platinum_8592plus.py`

#### Core Configuration
- **Cores:** 64 Golden Cove cores (7% more than 8490H)
- **Threads:** 128 (HyperThreading/SMT enabled, 2 threads per core)
- **Architecture:** Monolithic die design

#### Cache Hierarchy
- **L1 Data:** 48 KB per core (3.1 MB total)
- **L1 Instruction:** 32 KB per core (2.0 MB total)
- **L2:** 2 MB per core (128 MB total)
- **L3 (LLC):** 120 MB shared

#### Clock Frequencies
- **Base:** 1.9 GHz
- **Single-core boost:** 3.9 GHz
- **All-core boost:** 3.0 GHz (sustained, higher than 8490H)

#### Memory Subsystem
- **Technology:** DDR5-4800
- **Channels:** 8
- **Bandwidth:** 307.2 GB/s (8 × 38.4 GB/s)
- **Capacity:** Up to 4TB

#### SIMD Units
- **AVX-512:** 16 FP32 ops/cycle per core
- **AVX-512 FP16:** 32 ops/cycle per core
- **AMX:** 2 tiles per core, 16×16 INT8
- **Peak FP32:** 3.07 TFLOPS @ 3.0 GHz
- **Peak INT8 (AMX):** 98.3 TOPS

#### Process & Power
- **Process Node:** Intel 7 (10nm Enhanced SuperFin)
- **TDP:** 350W
  - Idle: ~85W
  - Dynamic: ~265W

#### Additional Features
- PCIe 5.0: 80 lanes
- CXL 1.1 support
- Highest AMX performance in Sapphire Rapids lineup

---

### Intel Xeon Granite Rapids (Xeon 6980P)

**File:** `src/graphs/hardware/models/datacenter/intel_granite_rapids.py`
**Status:** Released 24 September, 2024

#### Core Configuration
- **Cores:** 128 Redwood Cove P-cores (2× core count vs Sapphire Rapids)
- **Threads:** 256 (HyperThreading/SMT enabled, 2 threads per core)
- **Architecture:** Tile-based chiplet design (new for Intel)

#### Cache Hierarchy
- **L1 Data:** 48 KB per core (6.1 MB total)
- **L1 Instruction:** 32 KB per core (4.1 MB total)
- **L2:** 2 MB per core (256 MB total)
- **L3 (LLC):** 320 MB shared (distributed across tiles)

#### Clock Frequencies
- **Base:** 2.0 GHz
- **Single-core boost:** 3.8 GHz (estimated)
- **All-core boost:** 3.2 GHz (estimated)

#### Memory Subsystem
- **Technology:** DDR5-5600 (improved from DDR5-4800)
- **Channels:** 8 (or 12 on HBM SKUs)
- **Bandwidth:** 358.4 GB/s (8 × 44.8 GB/s) or 537.6 GB/s (12-channel)
- **Capacity:** Up to 4TB+

#### SIMD Units
- **AVX-512:** 16 FP32 ops/cycle per core
- **Enhanced AMX:** 2 tiles per core with sparsity acceleration
- **INT4 and FP8 support:** New capabilities
- **Peak FP32:** 6.55 TFLOPS @ 3.2 GHz
- **Peak INT8 (Enhanced AMX):** 209.7 TOPS
- **Peak INT4:** 419.4 TOPS

#### Process & Power
- **Process Node:** Intel 3 (Enhanced FinFET)
- **TDP:** 500W
  - Idle: ~100W
  - Dynamic: ~400W

#### Additional Features
- PCIe 6.0: 96 lanes
- CXL 2.0 support
- UPI for multi-socket
- Structured sparsity acceleration

---

### AMD EPYC 9654 (Genoa)

**File:** `src/graphs/hardware/models/datacenter/amd_epyc_9654.py`

#### Core Configuration
- **Cores:** 96 Zen 4 cores
- **Threads:** 192 (SMT enabled, 2 threads per core)
- **Architecture:** Chiplet design: 12× 8-core CCDs + I/O die

#### Cache Hierarchy
- **L1 Data:** 32 KB per core (3.1 MB total)
- **L1 Instruction:** 32 KB per core (3.1 MB total)
- **L2:** 1 MB per core (96 MB total)
- **L3:** 384 MB shared (32 MB per CCD × 12 CCDs)

#### Clock Frequencies
- **Base:** 2.4 GHz
- **Boost:** 3.7 GHz (single core)

#### Memory Subsystem
- **Technology:** DDR5-4800
- **Channels:** 12
- **Bandwidth:** 460.8 GB/s (12 × 38.4 GB/s)
- **Capacity:** Up to 6TB

#### SIMD Units
- **AVX-512:** Double-pumped (not native 512-bit), effective 8 FP32 ops/cycle
- **AVX2:** For compatibility
- **Peak FP32:** 1.84 TFLOPS @ 2.4 GHz
- **Peak INT8:** 7.37 TOPS
- **Note:** No dedicated AI accelerator (unlike Intel AMX)

#### Process & Power
- **Process Node:** TSMC 5nm
- **TDP:** 360W (can be tuned up to 400W)
  - Idle: ~90W
  - Dynamic: ~270W

#### Additional Features
- PCIe 5.0: 128 lanes
- CXL 1.1+ support
- AVX-512 is double-pumped (slower than native implementation)

---

### AMD EPYC 9754 (Genoa Flagship)

**File:** `src/graphs/hardware/models/datacenter/amd_epyc_9754.py`

#### Core Configuration
- **Cores:** 128 Zen 4 cores (33% more than 9654)
- **Threads:** 256 (SMT enabled, 2 threads per core)
- **Architecture:** Chiplet design: 16× 8-core CCDs + I/O die

#### Cache Hierarchy
- **L1 Data:** 32 KB per core (4 MB total)
- **L1 Instruction:** 32 KB per core (4 MB total)
- **L2:** 1 MB per core (128 MB total)
- **L3:** 512 MB shared (32 MB per CCD × 16 CCDs)

#### Clock Frequencies
- **Base:** 2.25 GHz (slightly lower than 9654 due to higher core count)
- **Boost:** 3.1 GHz (single core)

#### Memory Subsystem
- **Technology:** DDR5-4800
- **Channels:** 12
- **Bandwidth:** 460.8 GB/s (12 × 38.4 GB/s)
- **Capacity:** Up to 6TB

#### SIMD Units
- **AVX-512:** Double-pumped, effective 8 FP32 ops/cycle
- **Peak FP32:** 2.30 TFLOPS @ 2.25 GHz
- **Peak INT8:** 9.22 TOPS
- No dedicated AI accelerator

#### Process & Power
- **Process Node:** TSMC 5nm
- **TDP:** 360W
  - Idle: ~95W
  - Dynamic: ~265W

#### Additional Features
- PCIe 5.0: 128 lanes
- CXL 1.1+ support
- Highest core density (128 cores, 256 threads)

---

### AMD EPYC Turin (Zen 5, EPYC 9965)

**File:** `src/graphs/hardware/models/datacenter/amd_epyc_turin.py`
**Status:** Released 10 October, 2024

#### Core Configuration
- **Cores:** 192 Zen 5 cores (50% more than 9754)
- **Threads:** 384 (SMT enabled, 2 threads per core)
- **Architecture:** Chiplet design: 24× 8-core CCDs + I/O die

#### Cache Hierarchy
- **L1 Data:** 48 KB per core (9.2 MB total, increased from 32 KB)
- **L1 Instruction:** 32 KB per core (6.1 MB total)
- **L2:** 1 MB per core (192 MB total)
- **L3:** 768 MB shared (32 MB per CCD × 24 CCDs)

#### Clock Frequencies
- **Base:** 2.5 GHz (estimated)
- **Boost:** 3.8 GHz (single core, estimated)

#### Memory Subsystem
- **Technology:** DDR5-6000 (improved from DDR5-4800)
- **Channels:** 12
- **Bandwidth:** 576 GB/s (12 × 48 GB/s, 25% more than EPYC 9000)
- **Capacity:** Up to 6TB

#### SIMD Units
- **Native AVX-512:** Improved from double-pumped Zen 4, 8 FP32 ops/cycle
- **Peak FP32:** 3.84 TFLOPS @ 2.5 GHz
- **Peak INT8:** 15.36 TOPS
- **Possible AI matrix accelerator:** Rumored, not confirmed

#### Process & Power
- **Process Node:** TSMC 3nm (N3)
- **TDP:** 500W
  - Idle: ~120W
  - Dynamic: ~380W

#### Additional Features
- PCIe 6.0: 160 lanes
- CXL 2.0 support
- Native (not double-pumped) AVX-512 execution

---

### Ampere AmpereOne 128 (A128-30X)

**File:** `src/graphs/hardware/models/datacenter/ampere_ampereone_128.py`

#### Core Configuration
- **Cores:** 128 Ampere 64-bit ARM v8.6+ cores
- **Threads:** 128 (no SMT, 1 thread per core)
- **Architecture:** Coherent mesh interconnect with distributed snoop filtering

#### Cache Hierarchy
- **L1 Data:** 64 KB per core (8 MB total)
- **L1 Instruction:** 16 KB per core (2 MB total)
- **L2:** 2 MB per core (256 MB total)
- **System Cache (L3-like):** 48 MB shared

#### Clock Frequencies
- **Up to:** 3.6 GHz (consistent across all cores)

#### Memory Subsystem
- **Technology:** DDR5-5200
- **Channels:** 8
- **Bandwidth:** 332.8 GB/s (8 × 41.6 GB/s)
- **Capacity:** Up to 4TB

#### SIMD Units
- **2×128-bit SIMD units per core:** NEON + SVE
- **FP32:** 8 ops/cycle (4 per unit × 2 units)
- **FP16/BF16:** 16 ops/cycle (8 per unit × 2 units)
- **INT8:** 32 ops/cycle (16 per unit × 2 units)
- **Peak FP32:** 3.69 TFLOPS @ 3.6 GHz
- **Peak FP16/BF16:** 7.37 TFLOPS
- **Peak INT8:** 14.75 TOPS

#### Process & Power
- **Process Node:** TSMC 5nm
- **TDP:** 210W
  - Idle: ~40W
  - Dynamic: ~170W

#### Additional Features
- PCIe 5.0: 128 lanes with 32 controllers
- Native FP16/BF16 and INT8/INT16 support
- Ampere AIO (AI Optimizer) for ML frameworks
- 67% of the 192-core flagship

---

### Ampere AmpereOne 192 (A192-32X Flagship)

**File:** `src/graphs/hardware/models/datacenter/ampere_ampereone_192.py`

#### Core Configuration
- **Cores:** 192 Ampere 64-bit ARM v8.6+ cores
- **Threads:** 192 (no SMT, 1 thread per core)
- **Architecture:** Coherent mesh interconnect with distributed snoop filtering

#### Cache Hierarchy
- **L1 Data:** 64 KB per core (12.3 MB total)
- **L1 Instruction:** 16 KB per core (3.1 MB total)
- **L2:** 2 MB per core (384 MB total)
- **System Cache (L3-like):** 64 MB shared

#### Clock Frequencies
- **Up to:** 3.6 GHz (consistent across all cores)

#### Memory Subsystem
- **Technology:** DDR5-5200
- **Channels:** 8
- **Bandwidth:** 332.8 GB/s (8 × 41.6 GB/s)
- **Capacity:** Up to 4TB

#### SIMD Units
- **2×128-bit SIMD units per core:** NEON + SVE
- **FP32:** 8 ops/cycle per core
- **FP16/BF16:** 16 ops/cycle per core
- **INT8:** 32 ops/cycle per core
- **Peak FP32:** 5.53 TFLOPS @ 3.6 GHz
- **Peak FP16/BF16:** 11.06 TFLOPS
- **Peak INT8:** 22.12 TOPS

#### Process & Power
- **Process Node:** TSMC 5nm
- **TDP:** 283W
  - Idle: ~50W
  - Dynamic: ~233W

#### Additional Features
- PCIe 5.0: 128 lanes with 32 controllers
- Native FP16/BF16 and INT8/INT16 support
- Ampere AIO (AI Optimizer)
- Better energy efficiency for AI inference compared to x86

---

## Datacenter GPUs (NVIDIA)

### NVIDIA H100 PCIe (80GB)

**File:** `src/graphs/hardware/models/datacenter/h100_pcie.py`
**Architecture:** Hopper (4th gen, 2022)

#### SM Configuration
- **SMs:** 132 Streaming Multiprocessors
- **CUDA cores per SM:** 128 (doubled from Turing's 64)
- **Total CUDA cores:** 16,896
- **Threads per SM:** 2,048
- **Warps per SM:** 64
- **Warp size:** 32

#### Tensor Core Configuration
- **Tensor cores per SM:** 4 (4th generation)
- **Total Tensor Cores:** 528
- **Ops per clock per TC:** 512 FP16 FMAs
- **Supported precisions:** FP64, FP32, BF16, FP16, FP8 (E4M3/E5M2), INT8
- **Accumulator:** FP32 accumulator for FP16/BF16/FP8

#### Cache Hierarchy
- **L1 per SM:** 256 KB
- **L2 total:** 50 MB
- **Shared memory per SM:** Included in L1 cache

#### Clock Frequencies
- **Boost clock:** 1,980 MHz
- **Sustained clock:** 1,830 MHz (92% of boost)

#### Memory Subsystem
- **Technology:** HBM2e
- **Capacity:** 80 GB
- **Bandwidth:** 2 TB/s (2,000 GB/s)

#### Process & Power
- **TDP:** 350W

#### Performance
- **FP64:** 60 TFLOPS
- **FP32:** 60 TFLOPS (CUDA cores)
- **BF16/FP16:** 750 TFLOPS (Tensor Cores, 12.5× speedup)
- **FP8:** 1.5 PFLOPS (1,500 TFLOPS, 25× speedup)
- **INT8:** 1.5 POPS (1,500 TOPS)

---

### NVIDIA A100 SXM4 (80GB)

**File:** `src/graphs/hardware/models/datacenter/a100_sxm4_80gb.py`
**Architecture:** Ampere (3rd gen, 2020)

#### SM Configuration
- **SMs:** 108 Streaming Multiprocessors
- **CUDA cores per SM:** 128 (doubled from Volta/Turing's 64)
- **Total CUDA cores:** 13,824
- **Threads per SM:** 2,048
- **Warps per SM:** 64
- **Warp size:** 32

#### Tensor Core Configuration
- **Tensor cores per SM:** 4 (3rd generation, reduced from 8 but more capable)
- **Total Tensor Cores:** 432
- **Ops per clock per TC:** 512 (doubled throughput from Volta)
- **Supported precisions:** FP64 (new!), FP32, BF16 (new!), FP16, INT8
- **TF32 and BF16 support:** First GPU to support these formats
- **Accumulator:** FP32

#### Cache Hierarchy
- **L1 per SM:** 192 KB
- **L2 total:** 40 MB

#### Clock Frequencies
- **Boost clock:** 1,410 MHz
- **Sustained clock:** 1,300 MHz (92% of boost)

#### Memory Subsystem
- **Technology:** HBM2e
- **Capacity:** 80 GB
- **Bandwidth:** 2 TB/s (2,000 GB/s)

#### Process & Power
- **TDP:** 400W

#### Performance
- **FP64:** 9.7 TFLOPS (with Tensor Core support - new in Ampere!)
- **FP32:** 19.5 TFLOPS (CUDA cores)
- **TF32:** 156 TFLOPS (Tensor Cores - new format)
- **BF16/FP16:** 312 TFLOPS (Tensor Cores, 16× speedup)
- **INT8:** 624 TOPS (32× speedup)

---

### NVIDIA V100 SXM2 (32GB)

**File:** `src/graphs/hardware/models/datacenter/v100_sxm2.py`
**Architecture:** Volta (1st gen Tensor Cores, 2017)

#### SM Configuration
- **SMs:** 80 Streaming Multiprocessors
- **CUDA cores per SM:** 64
- **Total CUDA cores:** 5,120
- **Threads per SM:** 2,048
- **Warps per SM:** 64
- **Warp size:** 32

#### Tensor Core Configuration
- **Tensor cores per SM:** 8 (1st generation - first ever!)
- **Total Tensor Cores:** 640
- **Ops per clock per TC:** 256 FP16 FMAs
- **Supported precisions:** FP16, INT8 (Tensor Cores only)
- **Accumulator:** FP32

#### Cache Hierarchy
- **L1 per SM:** 128 KB
- **L2 total:** 6 MB

#### Clock Frequencies
- **Boost clock:** 1,530 MHz
- **Sustained clock:** 1,400 MHz (91% of boost)

#### Memory Subsystem
- **Technology:** HBM2
- **Capacity:** 32 GB
- **Bandwidth:** 900 GB/s

#### Process & Power
- **TDP:** 300W

#### Performance
- **FP64:** 7.8 TFLOPS
- **FP32:** 15.7 TFLOPS (CUDA cores)
- **FP16:** 31.4 TFLOPS (CUDA cores, 2× FP32)
- **FP16 (Tensor Cores):** 125 TFLOPS (7.96× speedup)
- **INT8:** 125 TOPS

---

### NVIDIA T4

**File:** `src/graphs/hardware/models/datacenter/t4.py`
**Architecture:** Turing (2nd gen Tensor Cores, 2018)

#### SM Configuration
- **SMs:** 40 Streaming Multiprocessors
- **CUDA cores per SM:** 64
- **Total CUDA cores:** 2,560
- **Threads per SM:** 1,024 (reduced from V100's 2,048)
- **Warps per SM:** 32
- **Warp size:** 32

#### Tensor Core Configuration
- **Tensor cores per SM:** 8 (2nd generation, improved INT8)
- **Total Tensor Cores:** 320
- **Ops per clock per TC:** 256 (similar to Volta)
- **Supported precisions:** FP16, INT8, INT4
- **Inference-optimized**

#### Cache Hierarchy
- **L1 per SM:** 96 KB
- **L2 total:** 4 MB

#### Clock Frequencies
- **Boost clock:** 1,590 MHz
- **Sustained clock:** 1,470 MHz (92% of boost)

#### Memory Subsystem
- **Technology:** GDDR6
- **Capacity:** 16 GB
- **Bandwidth:** 320 GB/s

#### Process & Power
- **TDP:** 70W (inference-optimized for low power!)

#### Performance
- **FP32:** 8.1 TFLOPS (CUDA cores)
- **FP16:** 65 TFLOPS (Tensor Cores, 8× speedup)
- **INT8:** 130 TOPS (16× speedup)
- **INT4:** 260 TOPS (32× speedup)

---
## Edge GPUs (NVIDIA, ARM)

### NVIDIA Jetson Orin Nano (8GB)

**File:** `src/graphs/hardware/models/edge/jetson_orin_nano.py`
**Architecture:** Ampere (2023)

#### SM Configuration
- **SMs:** 16 Streaming Multiprocessors
- **CUDA cores per SM:** 64
- **Total CUDA cores:** 1,024
- **Threads per SM:** 64
- **Warps per SM:** 2
- **Warp size:** 32

#### Tensor Core Configuration
- **Total Tensor Cores:** 32 (from references)
- **Ops per SM per clock:** 512 INT8 ops

#### Cache Hierarchy
- **L1 per SM:** 128 KB
- **L2 total:** 2 MB (half of AGX)

#### Clock Frequencies (Multi-Power Mode)

**7W Mode (battery-powered):**
- Base: 204 MHz
- Boost: 918 MHz
- Sustained: 300 MHz (33% throttle)

**15W Mode (standard edge):**
- Base: 306 MHz
- Boost: 918 MHz
- Sustained: 500 MHz (54% throttle)

#### Memory Subsystem
- **Technology:** LPDDR5
- **Capacity:** 8 GB
- **Bandwidth:** 68 GB/s (original) or 102 GB/s (Super variant)

#### Process & Power
- **Process Node:** 12nm
- **TDP Variants:** 7W / 15W

#### Performance
- **Marketing (Super):** 67 TOPS INT8 (sparse)
- **Marketing (original):** 40 TOPS INT8 (sparse)
- **Dense GPU only:** ~21 TOPS INT8
- **Empirical at 7W:** ~1 TOPS (4-7% of peak)

---

### NVIDIA Jetson Orin AGX

**File:** `src/graphs/hardware/models/edge/jetson_orin_agx.py`
**Architecture:** Ampere (2022)

#### SM Configuration
- **SMs:** 16 Streaming Multiprocessors
- **CUDA cores per SM:** 128 (Ampere architecture)
- **Total CUDA cores:** 2,048
- **Threads per SM:** 64
- **Warps per SM:** 2
- **Warp size:** 32

#### Tensor Core Configuration
- **Tensor cores per SM:** 4
- **Total Tensor Cores:** 64
- **Ops per SM per clock:** 512 INT8 ops
- **Supported precisions:** INT8, FP16, FP32

#### Cache Hierarchy
- **L1 per SM:** 128 KB
- **L2 total:** 4 MB

#### Clock Frequencies (Multi-Power Mode)

**15W Mode (passive cooling):**
- Base: 306 MHz
- Boost: 1,020 MHz
- Sustained: 400 MHz (39% throttle - severe!)

**30W Mode (active fan):**
- Base: 612 MHz
- Boost: 1,150 MHz
- Sustained: 650 MHz (57% throttle)

**60W Mode (max performance):**
- Base: 918 MHz
- Boost: 1,300 MHz
- Sustained: 1,000 MHz (77% throttle)

#### Memory Subsystem
- **Technology:** LPDDR5
- **Capacity:** 64 GB
- **Bandwidth:** 204.8 GB/s

#### Process & Power
- **Process Node:** 8nm
- **TDP Variants:** 15W / 30W / 60W

#### Performance
- **Marketing claim:** 275 TOPS INT8 (sparse, all engines)
- **Dense GPU only:** 85 TOPS INT8 (realistic for PyTorch)
- **Empirical at 15W:** ~1.5 TOPS (2-4% of peak due to thermal throttling!)

---

### NVIDIA Jetson Thor

**File:** `src/graphs/hardware/models/automotive/jetson_thor.py`
**Architecture:** Blackwell (next-gen, 2025+)

#### SM Configuration
- **SMs:** 64 Streaming Multiprocessors (estimated)
- **CUDA cores per SM:** Not specified (Blackwell architecture)
- **Threads per SM:** 128
- **Warps per SM:** 4
- **Warp size:** 32

#### Tensor Core Configuration
- **Ops per SM per clock:** 256 INT8 ops (actual datapath, not sparsity-inflated)

#### Cache Hierarchy
- **L1 per SM:** 256 KB
- **L2 total:** 8 MB

#### Clock Frequencies (Multi-Power Mode)

**30W Mode (typical deployment):**
- Base: 500 MHz
- Boost: 1,300 MHz
- Sustained: 750 MHz (58% of boost)

**60W Mode (high-performance):**
- Base: 800 MHz
- Boost: 1,300 MHz
- Sustained: 1,100 MHz (85% of boost)

**100W Mode (benchtop only):**
- Base: 1,000 MHz
- Boost: 1,300 MHz
- Sustained: 1,250 MHz (96% of boost)

#### Memory Subsystem
- **Technology:** HBM3
- **Capacity:** 128 GB
- **Bandwidth:** 450 GB/s

#### Process & Power
- **Process Node:** 4nm
- **TDP Variants:** 30W / 60W / 100W

#### Performance
- **Marketing claim:** 2,000 TOPS INT8 (includes sparsity!)
- **Actual datapath:** 1,000 TOPS INT8 (dense workloads)
- **Projected empirical at 30W:** ~30 TOPS (3% of peak)

---

### ARM Mali-G78 MP20

**File:** `src/graphs/hardware/models/mobile/arm_mali_g78_mp20.py`
**Architecture:** Valhall (2nd gen, 2020)

#### Core Configuration
- **Shader cores:** 20 (MP20 configuration, max 24)
- **Execution lanes per core:** 16
- **Threads per unit:** 256
- **Warps per unit:** 16
- **Warp size:** 16 (different from NVIDIA's 32!)

#### Tensor Core Configuration
- **None** - graphics GPU, not AI-optimized
- No dedicated tensor cores
- AI inference runs on unified shader cores

#### Cache Hierarchy
- **L1 per core:** 32 KB
- **L2 total:** 2 MB (configurable 512 KB - 2 MB)

#### Clock Frequencies
- **Base:** 400 MHz
- **Boost:** 950 MHz
- **Sustained:** 848 MHz (typical in Google Tensor SoC)

#### Memory Subsystem
- **Technology:** Shared system memory (mobile SoC)
- **Capacity:** Up to 8 GB
- **Bandwidth:** ~40 GB/s (typical for mobile SoC integration)

#### Process & Power
- **TDP:** 3-5W typical @ 848 MHz

#### Performance
- **FP32:** 1.94 TFLOPS (~97 GFLOPS per core)
- **FP16:** 3.88 TFLOPS (2× FP32)
- **INT8:** ~2 TOPS (not optimized for AI)

---

## TPU Accelerators


Tensor Processing Unit (TPU) generations

|                 |  v1  |  v2  |  v3  |  v4  |  v5e |  v5p | v6e (Trillium) | v7 (Ironwood) |
|-----------------|------|------|------|------|------|------|----------------|---------------|
| Date introduced | 2015 | 2017 | 2018 | 2021 | 2023 | 2023 |     2024       |      2025     |
| Process node    |	28 nm |	16 nm |	16 nm |	7 nm | Not listed | Not listed | Not listed | Not listed |
| Die size (mm2)  | 331	  | <625  | <700  | <400 | 300–350 | Not listed | Not listed | Not listed |
| On-chip memory (MiB) | 28 | 32 | 32 (VMEM) + 5 (spMEM) | 128 (CMEM) + 32 (VMEM) + 10 (spMEM) | Not listed | Not listed | Not listed | Not listed |
| Clock speed (MHz) | 700 | 700 | 940 | 1050 | Not listed | 1750 | Not listed | Not listed |
| Memory          | 8 GiB DDR3 | 16 GiB HBM | 32 GiB HBM | 32 GiB HBM | 16 GB HBM | 95 GB HBM | 32 GB | 192 GB HBM |
| Memory bandwidth | 34 GB/s | 600 GB/s | 900 GB/s | 1200 GB/s | 819 GB/s | 2765 GB/s | 1640 GB/s | 7.37 TB/s |
| Thermal design power (W) | 75 | 280 | 220 | 170 | Not listed | Not listed | Not listed | Not listed |
| Computational performance (trillion operations per second) | 23 | 45 | 123 | 275 | 197 (bf16), 393 (int8) | 459 (bf16), 918 (int8) | 918 (bf16), 1836 (int8) | 4614 (fp8) |
| Energy efficiency (teraOPS/W) | 0.31 | 0.16 | 0.56 | 1.62 | Not listed | Not listed | Not listed | 4.7 |

### TPU v1 (2015) - "Inference Pioneer"

**File:** `src/graphs/hardware/models/datacenter/tpu_v1.py`

#### Systolic Array Configuration
- **Array dimensions:** 256×256 (largest among all TPU generations)
- **Total MACs:** 65,536 (256×256)
- **Number of MXUs:** 1
- **Cores per chip:** 1 MXU

#### Matrix Multiply Unit (MXU) Specifications
- **Peak performance:** 92 TOPS INT8
- **Operations per clock:** 131,072 (256×256 × 2 ops)
- **Clock frequency:** 700 MHz

#### Memory Hierarchy
- **Weight Memory:** 8 GiB DDR3 (off-chip)
- **Weight FIFO:** 256 KiB (4 tiles × 64 KiB per tile)
- **Weight tile size:** 64 KiB per tile
- **FIFO depth:** 4 tiles
- **Accumulators:** 4 MiB (double-buffered, 256 elements wide)
- **Unified Buffer:** 24 MiB
- **Memory bandwidth:** 34 GB/s (DDR3)
- **Data path width:** 256 bytes

#### Clock Frequencies
- **Base/sustained:** 700 MHz (single operating point)

#### Supported Precisions
- **INT8 only:** 92 TOPS (native)
- No FP32/FP16/BF16 support
- **Accumulator precision:** INT32
- **Operations per clock:** 131,072 ops/clock @ INT8

#### Process & Power
- **Process node:** Not specified (estimated 28nm era)
- **TDP:** 75W (estimated)
- **Cooling:** Active air

#### Architecture Generation
- 1st generation (2017)
- Inference-only design
- ISCA 2017 paper architecture

#### Key Characteristics
- Roofline knee: ~1,350 ops/byte
- Pipeline fill: 256 cycles
- Interface: PCIe Gen3 x16

---

### TPU v3 (2018) - "HBM & Floating Point"

**File:** `src/graphs/hardware/models/datacenter/tpu_v3.py`

#### Systolic Array Configuration
- **Array dimensions:** 128×128 per MXU
- **Total MACs per MXU:** 16,384
- **Number of MXUs:** 2
- **Total MACs per chip:** 32,768

#### Matrix Multiply Unit (MXU) Specifications
- **Peak performance BF16:** 123 TFLOPS total (61.5 TFLOPS per MXU)
- **Peak performance INT8:** 246 TOPS total (123 TOPS per MXU)
- **Operations per clock per MXU:** 32,768 (128×128 × 2 ops)
- **Clock frequency:** ~940 MHz (estimated)

#### Memory Hierarchy
- **Main memory:** 16 GB HBM (on-chip)
- **Weight tile size:** 32 KiB per tile (smaller than v1)
- **Weight FIFO depth:** 2 tiles
- **L1 cache per MXU:** 16 MB
- **L2 cache total:** 32 MB (shared)
- **Accumulators:** 2 MiB per MXU (128 elements wide)
- **Unified Buffer:** 32 MiB
- **Memory bandwidth:** 900 GB/s

#### Clock Frequencies
- **Base/sustained:** ~940 MHz

#### Supported Precisions
- **FP32:** 61.5 TFLOPS (0.5× relative speedup, not native)
- **BF16:** 123 TFLOPS (1.0× baseline, native)
- **INT8:** 246 TOPS (2.0× relative speedup)
- **BF16 accumulator** for FP operations
- **INT32 accumulator** for INT8

#### Process & Power
- **Process node:** Advanced node (estimated 16nm/12nm)
- **TDP:** 200W (estimated)
- **Cooling:** Active liquid

#### Architecture Generation
- 2nd generation (2018)
- First with HBM and BF16 support
- Training and inference

#### Key Characteristics
- Pipeline fill: 128 cycles (shorter than v1)
- HBM energy: 5 pJ/byte (5× lower than DDR3)
- Smaller arrays for better utilization

---

### TPU v4 (2021) - "Performance Leap"

**File:** `src/graphs/hardware/models/datacenter/tpu_v4.py`

#### Systolic Array Configuration
- **Array dimensions:** 128×128 per MXU
- **Total MACs per MXU:** 16,384
- **Number of MXUs:** 2
- **Total MACs per chip:** 32,768

#### Matrix Multiply Unit (MXU) Specifications
- **Peak performance BF16:** 275 TFLOPS total (137.5 TFLOPS per MXU)
- **Peak performance INT8:** 550 TOPS total (275 TOPS per MXU)
- **Operations per clock per MXU:** 32,768 (128×128 × 2 ops)
- **Clock frequency:** ~1,050 MHz (estimated)

#### Memory Hierarchy
- **Main memory:** 32 GB HBM2e
- **Weight tile size:** 32 KiB per tile
- **Weight FIFO depth:** 2 tiles
- **L1 cache per MXU:** 16 MB
- **L2 cache total:** 32 MB (shared)
- **Accumulators:** 2 MiB per MXU (128 elements wide)
- **Unified Buffer:** 32 MiB
- **Memory bandwidth:** 1.2 TB/s

#### Clock Frequencies
- **Base/sustained:** ~1,050 MHz

#### Supported Precisions
- **FP32:** 137.5 TFLOPS (0.5× relative speedup, not native)
- **BF16:** 275 TFLOPS (1.0× baseline, native)
- **INT8:** 550 TOPS (2.0× relative speedup)
- **Operations per clock:** 65,536 ops/clock @ BF16

#### Process & Power
- **Process node:** Advanced node (estimated 7nm)
- **TDP:** 350W
- **Cooling:** Active liquid

#### Architecture Generation
- 3rd generation (2021)
- Optimized for training and inference
- Very energy efficient

#### Key Characteristics
- Pipeline fill: 128 cycles
- HBM2e energy: 10 pJ/byte
- MAC energy: 0.25 pJ per BF16 MAC

---

### TPU v5p (2023) - "AI Training Powerhouse"

**File:** `src/graphs/hardware/models/datacenter/tpu_v5p.py`

#### Systolic Array Configuration
- **Array dimensions:** 128×128 per MXU
- **Total MACs per MXU:** 16,384
- **Number of MXUs:** 2+ (likely more, not public - conservative estimate)
- **Total MACs per chip:** 32,768+ (conservative)

#### Matrix Multiply Unit (MXU) Specifications
- **Peak performance BF16:** 459 TFLOPS
- **Peak performance FP8:** 918 TOPS (~2× BF16)
- **Peak performance INT8:** 918 TOPS
- **Operations per clock per MXU:** 32,768 (128×128 × 2 ops)
- **Clock frequency:** ~1,100 MHz (estimated)

#### Memory Hierarchy
- **Main memory:** 32 GB HBM3 (estimated)
- **Weight tile size:** 32 KiB per tile
- **Weight FIFO depth:** 2 tiles
- **L1 cache per MXU:** 16 MB
- **L2 cache total:** 32 MB (shared)
- **Accumulators:** 2 MiB per MXU (128 elements wide)
- **Unified Buffer:** 32 MiB
- **Memory bandwidth:** 1.6 TB/s

#### Clock Frequencies
- **Base/sustained:** ~1,100 MHz

#### Supported Precisions
- **FP32:** 229.5 TFLOPS (0.5× relative speedup, not native)
- **BF16:** 459 TFLOPS (1.0× baseline, native)
- **FP8 (E4M3):** 918 TOPS (2.0× relative speedup, NEW)
- **INT8:** 918 TOPS (2.0× relative speedup)
- **FP16 accumulator** for FP8
- **INT32 accumulator** for INT8

#### Process & Power
- **Process node:** Advanced node (estimated 5nm/4nm)
- **TDP:** 400W (estimated)
- **Cooling:** Active liquid

#### Architecture Generation
- 4th generation (2023)
- Latest performance-optimized
- Training and inference focused

#### Key Innovations
- FP8 precision support (transformer optimization)
- Hardware sparsity acceleration (dynamic zero-skipping)
- Improved interconnect (ICI) for multi-chip scaling
- Higher clock speeds vs v4
- HBM3 energy: 8 pJ/byte (lower than HBM2e)
- MAC energy: 0.25 pJ per BF16 MAC

---

### Coral Edge TPU (2019) - "Ultra-Low Power Edge"

**File:** `src/graphs/hardware/models/edge/coral_edge_tpu.py`

#### Systolic Array Configuration
- **Array dimensions:** 64×64 (estimated, not published)
- **Total MACs:** 4,096 (estimated)
- **Number of MXUs:** 1
- **Single small systolic array optimized for edge**

#### Matrix Multiply Unit (MXU) Specifications
- **Peak performance:** 4 TOPS INT8 only
- **Operations per clock:** 8 ops/clock
- **Clock frequency:** 500 MHz (estimated)
- **Efficiency:** 85% (3.4 TOPS effective)

#### Memory Hierarchy
- **Main memory:** Uses host CPU memory (0 on-chip DRAM)
- **Weight tile size:** 4 KiB (tiny tiles for edge)
- **Weight FIFO depth:** 1 tile (minimal buffering)
- **L1 cache:** ~512 KB on-chip SRAM
- **L2 cache:** 0 (uses host memory)
- **Accumulators:** 512 KB (64 elements wide)
- **Unified Buffer:** 512 KB on-chip
- **Memory bandwidth:** ~4 GB/s (USB 3.0/PCIe limited)

#### Clock Frequencies
- **Fixed:** 500 MHz (no DVFS)

#### Supported Precisions
- **INT8 only:** 4 TOPS (only mode available)
- No FP32/FP16/BF16 support
- **Accumulator precision:** INT32
- Requires TensorFlow Lite with full INT8 quantization

#### Process & Power
- **Process node:** 14nm
- **TDP:** 2W peak, 0.5W idle
- **Cooling:** Passive heatsink only
- **Energy efficiency:** 0.6 pJ/op (most efficient!)

#### Architecture Generation
- Edge derivative (2019)
- Inference-only
- IoT/embedded focus

#### Key Characteristics
- Pipeline fill: 64 cycles (short)
- Form factors: USB, M.2, PCIe variants
- BOM cost: ~$25 (die), $75 retail
- Target markets: IoT cameras, drones, embedded vision
- USB 3.0 energy: 20 pJ/byte (bandwidth limited)
- MAC energy: 0.15 pJ per INT8 MAC (very efficient)

---

## DSP & Automotive SoCs

### Qualcomm QCS6490 (Edge AI)

**File:** `src/graphs/hardware/models/edge/qualcomm_qcs6490.py`

#### Core Configuration
- **AI Accelerator:** Hexagon NPU V79 (6th-gen AI Engine)
- **Architecture:** Hexagon DSP with dedicated tensor accelerator
- **Processing Units:** 16 DSP cores with vector and tensor units
- **Operations per cycle:** 500 INT8 ops/unit/clock

#### Memory Hierarchy
- **L1 Cache:** 64 KB per DSP core
- **L2 Cache:** 3 MB total
- **Main Memory:** 8 GB LPDDR4X
- **Memory Bandwidth:** 40 GB/s

#### Clock Frequencies

**5W Mode (Battery):**
- Base: 800 MHz
- Boost: 1.8 GHz
- Sustained: 1.0 GHz (56% throttle)

**10W Mode (Standard):**
- Base: 1.0 GHz
- Boost: 1.8 GHz
- Sustained: 1.5 GHz (83% of boost)

**15W Mode (Max Performance):**
- Base: 1.2 GHz
- Boost: 1.8 GHz
- Sustained: 1.7 GHz (94% of boost)

#### Process & Power
- **Process Node:** 6nm TSMC
- **TDP Range:** 5-15W
- **Cooling:** Passive heatsink (5-10W), active fan (15W)

#### Performance
- **Peak:** 12 TOPS INT8 @ 1.5 GHz sustained
- **Effective (10W):** ~6.6 TOPS (55% efficiency)
- **Effective (15W):** ~8.8 TOPS (65% efficiency)

#### Precision Support
- **INT8:** 12 TOPS peak, native acceleration
- **INT4:** 24 TOPS peak (2× INT8)
- **INT16:** 6 TOPS peak (0.5× INT8)
- **FP16:** 6 TOPS peak, native but slower

---

### Qualcomm QRB5165 (Robotics Platform)

**File:** `src/graphs/hardware/models/edge/qrb5165.py`

#### Core Configuration
- **AI Accelerator:** Hexagon 698 DSP with HVX + HTA
- **CPU:**
  - 1× Kryo 585 Prime (Cortex-A77) @ 2.84 GHz
  - 3× Kryo 585 Gold (Cortex-A77) @ 2.42 GHz
  - 4× Kryo 585 Silver (Cortex-A55) @ 1.81 GHz
- **GPU:** Adreno 650 (~1.0 TFLOPS FP32)
- **DSP Units:** 32 processing elements (4× HVX 1024-bit vector units + HTA)
- **Operations per cycle:** 312 INT8 ops/unit/clock

#### Memory Hierarchy
- **L1 Cache:** 128 KB per DSP unit
- **L2 Cache:** 4 MB shared
- **Main Memory:** 16 GB LPDDR5
- **Memory Bandwidth:** 44 GB/s (quad-channel @ 2750 MHz)

#### Clock Frequencies

**7W Mode (Robotics):**
- Base: 800 MHz
- Boost: 1.5 GHz
- Sustained: 900 MHz (60% throttle)

#### Process & Power
- **Process Node:** 7nm TSMC FinFET
- **TDP:** 7W typical
- **Cooling:** Passive heatsink

#### Performance
- **Peak:** 15 TOPS INT8 @ 1.5 GHz
- **Sustained (7W):** 8.99 TOPS
- **Effective (7W):** ~5.4 TOPS (60% efficiency, 36% of peak)

#### Precision Support
- **INT8:** 15 TOPS peak, native HTA acceleration
- **INT4:** 30 TOPS peak (2× INT8)
- **INT16:** 7.5 TOPS peak (0.5× INT8)
- **FP16:** 3.75 TFLOPS (emulated, not native)

#### BOM Cost (10K units)
- Silicon: $55, Package: $14, Memory: $18
- **Total BOM:** $103
- **Retail:** $278 (2.7× margin)

---

### Qualcomm SA8775P (Snapdragon Ride - Mid-Range)

**File:** `src/graphs/hardware/models/automotive/qualcomm_sa8775p.py`

#### Core Configuration
- **AI Accelerator:** Hexagon DSP with dual HMX (Hexagon Matrix eXtensions)
- **Architecture:** Quad HVX (Vector) + dual HMX (Matrix) co-processors
- **Processing Units:** 32 units (HMX + HVX combined)
- **Operations per cycle:** 500 INT8 ops/unit/clock

#### Memory Hierarchy
- **L1 Cache:** 128 KB per unit
- **L2 Cache:** 8 MB total
- **Main Memory:** 16 GB LPDDR5
- **Memory Bandwidth:** 90 GB/s (automotive LPDDR5)

#### Clock Frequencies

**20W Mode (Passive):**
- Base: 1.0 GHz
- Boost: 2.4 GHz
- Sustained: 1.6 GHz (67% throttle)

**30W Mode (Active):**
- Base: 1.5 GHz
- Boost: 2.4 GHz
- Sustained: 2.0 GHz (83% of boost)

**45W Mode (Max):**
- Base: 1.8 GHz
- Boost: 2.4 GHz
- Sustained: 2.3 GHz (96% of boost)

#### Process & Power
- **Process Node:** 5nm TSMC (ASIL D certified)
- **TDP Range:** 20-45W
- **Cooling:** Passive (20W), active fan (30-45W)

#### Performance
- **Peak:** 32 TOPS INT8 @ 2.0 GHz sustained
- **Effective (20W):** ~6.5 TOPS (50% efficiency)
- **Effective (30W):** ~18.5 TOPS (58% efficiency)
- **Effective (45W):** ~23.9 TOPS (65% efficiency)

#### Precision Support
- **INT8:** 32 TOPS sustained, native HMX
- **INT4:** 64 TOPS
- **INT16:** 16 TOPS
- **FP16:** 16 TOPS, native

#### BOM Cost (10K units, automotive-grade)
- Silicon: $180, Package: $35, Memory: $80
- **Total BOM:** $350
- **Retail:** $770 (2.2× margin)

---

### Qualcomm Snapdragon Ride (High-End Autonomous)

**File:** `src/graphs/hardware/models/automotive/qualcomm_snapdragon_ride.py`

#### Core Configuration
- **AI Accelerator:** Multi-accelerator heterogeneous platform
- **Architecture:** Hexagon DSP cluster + dedicated AI accelerators + GPU compute
- **Processing Units:** 128 heterogeneous compute units
- **Operations per cycle:** 1400-1880 INT8 ops/unit/clock (varies by power mode)

#### Memory Hierarchy
- **L1 Cache:** 256 KB per unit
- **L2 Cache:** 32 MB total (large cache)
- **Main Memory:** 32 GB LPDDR5X (L4/L5 requirement)
- **Memory Bandwidth:** 200 GB/s (LPDDR5X dual-channel)

#### Clock Frequencies

**65W Mode (Highway Pilot):**
- Base: 1.5 GHz
- Boost: 3.0 GHz
- Sustained: 2.2 GHz (73% throttle)

**100W Mode (Urban Driving):**
- Base: 2.0 GHz
- Boost: 3.0 GHz
- Sustained: 2.7 GHz (90% of boost)

**130W Mode (Robotaxi):**
- Base: 2.5 GHz
- Boost: 3.0 GHz
- Sustained: 2.9 GHz (97% of boost)

#### Process & Power
- **Process Node:** 4nm TSMC (ASIL D certified)
- **TDP Range:** 65-130W
- **Cooling:** Active fan (65-100W), liquid cooling (130W)

#### Performance
- **Peak:** 700 TOPS INT8 @ 2.9 GHz (marketed)
- **Sustained (100W):** 552 TOPS
- **Effective (65W):** ~204 TOPS (52% efficiency)
- **Effective (100W):** ~320 TOPS (58% efficiency)
- **Effective (130W):** ~432 TOPS (62% efficiency)

#### Precision Support
- **INT8:** 700 TOPS peak, native AI accelerators
- **INT4:** 1400 TOPS
- **INT16:** 350 TOPS
- **FP16:** 350 TOPS

#### BOM Cost (10K units, automotive-grade)
- Silicon: $380, Package: $85, Memory: $180
- **Total BOM:** $800
- **Retail:** $1,600 (2.0× margin)

---

### TI TDA4VM (Jacinto 7 - Standard)

**File:** `src/graphs/hardware/models/automotive/ti_tda4vm.py`

#### Core Configuration
- **CPU:** 2× Cortex-A72 @ 2.0 GHz + R5F safety cores
- **AI Accelerator:** C7x DSP @ 1.0 GHz with MMA (Matrix Multiply Accelerator)
- **DSP Units:** 32 equivalent processing elements
- **Operations per cycle:** 250 INT8 ops/unit/clock
- **Vector Processing:** 512-bit SIMD

#### Memory Hierarchy
- **L1 Cache:** 48 KB per C7x (32 KB cache + 16 KB SRAM)
- **L2 Cache (MSMC):** 8 MB on-chip SRAM
- **Main Memory:** 8 GB LPDDR4x
- **Memory Bandwidth:** 60 GB/s (dual-channel @ 3733 MT/s)

#### Clock Frequencies

**10W Mode (Front Camera ADAS):**
- Base: 600 MHz
- Boost: 1.0 GHz
- Sustained: 850 MHz (85% of peak)

**20W Mode (Full ADAS System):**
- Base: 700 MHz
- Boost: 1.0 GHz
- Sustained: 950 MHz (95% of peak)

#### Process & Power
- **Process Node:** 16nm FinFET (automotive qualified)
- **TDP Range:** 10-20W
- **Temperature Range:** -40°C to 125°C (AEC-Q100)
- **Safety:** ASIL-D/SIL-3 certified
- **Cooling:** Automotive passive (10W), active (20W)

#### Performance
- **Peak:** 8 TOPS INT8 @ 1.0 GHz
- **C7x DSP:** 80 GFLOPS FP32, 256 GOPS INT16
- **Sustained (10W):** 6.8 TOPS
- **Effective (10W):** ~4.76 TOPS (70% efficiency, 60% of peak)
- **Effective (20W):** ~6.08 TOPS (80% efficiency, 76% of peak)

#### Precision Support
- **INT8:** 8 TOPS peak, native MMA
- **INT16:** 4 TOPS (0.5× INT8)
- **FP32:** 80 GFLOPS (C7x vector, not MMA)
- **FP16:** Emulated via C7x

#### Energy Characteristics
- **FP32:** 2.0 pJ/FLOP (automotive grade)
- **Memory:** 20 pJ/byte (LPDDR4x automotive)
- **INT8:** 0.15× FP32 energy
- **INT16:** 0.25× FP32 energy

---

### TI TDA4VH (Jacinto 7 Very High Performance)

**File:** `src/graphs/hardware/models/automotive/ti_tda4vh.py`

#### Core Configuration
- **CPU:** 8× Cortex-A72 @ 2.0 GHz (4× TDA4VM) + multiple R5F safety cores
- **AI Accelerator:** 4× C7x DSP @ 1.0 GHz with 4× MMAv2
- **DSP Units:** 128 equivalent processing elements (4× TDA4VM)
- **Operations per cycle:** 250 INT8 ops/unit/clock

#### Memory Hierarchy
- **L1 Cache:** 48 KB per C7x
- **L2 Cache:** 16 MB total (larger for multiple accelerators)
- **Main Memory:** 16 GB LPDDR5
- **Memory Bandwidth:** 100 GB/s (LPDDR5 @ 6400 MT/s)

#### Clock Frequencies

**20W Mode (Multi-Camera L2+ ADAS):**
- Base: 700 MHz
- Boost: 1.0 GHz
- Sustained: 850 MHz

**35W Mode (Full L3-4 Autonomy):**
- Base: 800 MHz
- Boost: 1.0 GHz
- Sustained: 950 MHz

#### Process & Power
- **Process Node:** Advanced node (newer than TDA4VM)
- **TDP Range:** 20-35W
- **Safety:** ASIL-D/SIL-3
- **Cooling:** Automotive active (20W), enhanced active (35W)

#### Performance
- **Peak:** 32 TOPS INT8 @ 1.0 GHz (4× TDA4VM)
- **FP32:** 320 GFLOPS (4× C7x DSP)
- **Effective (20W):** ~19.0 TOPS (70% efficiency)
- **Effective (35W):** ~23.7 TOPS (78% efficiency)

#### Precision Support
- **INT8:** 32 TOPS peak, native MMAv2
- **INT16:** 16 TOPS
- **FP32:** 320 GFLOPS

#### Energy Characteristics
- **FP32:** 1.8 pJ/FLOP (10% better than TDA4VM)
- **Memory:** 15 pJ/byte (LPDDR5 more efficient)

---

### TI TDA4AL (Jacinto 7 Advanced Low-Power)

**File:** `src/graphs/hardware/models/automotive/ti_tda4al.py`

#### Core Configuration
- **CPU:** Dual Cortex-A72 @ 2.0 GHz
- **AI Accelerator:** C7x DSP @ 1.0 GHz with MMAv2 (newer than TDA4VM's MMAv1)
- **DSP Units:** 32 equivalent processing elements (same as TDA4VM)
- **Operations per cycle:** 250 INT8 ops/unit/clock

#### Memory Hierarchy
- **L1 Cache:** 48 KB per unit
- **L2 Cache:** 8 MB total
- **Main Memory:** 8 GB LPDDR4x
- **Memory Bandwidth:** 60 GB/s

#### Clock Frequencies

**10W Mode (Front Camera ADAS):**
- Base: 600 MHz
- Boost: 1.0 GHz
- Sustained: 880 MHz (better than TDA4VM's 850 MHz)

**18W Mode (Multi-Camera ADAS):**
- Base: 700 MHz
- Boost: 1.0 GHz
- Sustained: 980 MHz

#### Process & Power
- **Process Node:** Newer node (better power efficiency than TDA4VM)
- **TDP Range:** 10-18W
- **Safety:** ASIL-D/SIL-3
- **Cooling:** Automotive passive (10W), active (18W)

#### Performance
- **Peak:** 8 TOPS INT8 @ 1.0 GHz (same as TDA4VM)
- **FP32:** 80 GFLOPS @ 1.0 GHz
- **Effective (10W):** ~6.0 TOPS (75% efficiency, better than TDA4VM's 70%)
- **Effective (18W):** ~6.4 TOPS (82% efficiency)

#### Key Differences from TDA4VM
- Same AI performance (8 TOPS) but more efficient MMAv2 architecture
- Better power efficiency: 10-18W vs 10-20W
- Higher efficiency factors (75% vs 70% @ 10W)
- Better for newer automotive designs replacing TDA4VM

#### Precision Support
- **INT8:** 8 TOPS peak, native MMAv2
- **INT16:** 4 TOPS
- **FP32:** 80 GFLOPS

#### Energy Characteristics
- **FP32:** 1.8 pJ/FLOP (10% better than TDA4VM)
- **Memory:** 18 pJ/byte

---

### TI TDA4VL (Jacinto 7 Entry-Level)

**File:** `src/graphs/hardware/models/automotive/ti_tda4vl.py`

#### Core Configuration
- **CPU:** 2× Cortex-A72 @ 1.2 GHz (lower than TDA4VM's 2.0 GHz)
- **AI Accelerator:** C7x DSP @ 1.0 GHz with MMAv2
- **DSP Units:** 16 equivalent processing elements (half of TDA4VM)
- **Operations per cycle:** 250 INT8 ops/unit/clock

#### Memory Hierarchy
- **L1 Cache:** 48 KB per unit
- **L2 Cache:** 8 MB total
- **Main Memory:** 4 GB LPDDR4x
- **Memory Bandwidth:** 60 GB/s

#### Clock Frequencies

**7W Mode (Entry-Level ADAS):**
- Base: 500 MHz
- Boost: 1.0 GHz
- Sustained: 750 MHz (75% sustained)

**12W Mode (Multi-Function ADAS):**
- Base: 700 MHz
- Boost: 1.0 GHz
- Sustained: 900 MHz

#### Process & Power
- **Process Node:** Similar to TDA4AL (newer generation)
- **TDP Range:** 7-12W
- **Temperature Range:** -40°C to 125°C
- **Safety:** ASIL-B/C (lower than TDA4VM's ASIL-D)
- **Cooling:** Automotive passive

#### Performance
- **Peak:** 4 TOPS INT8 @ 1.0 GHz (half of TDA4VM)
- **FP32:** 40 GFLOPS @ 1.0 GHz
- **Effective (7W):** ~2.2 TOPS (72% efficiency)
- **Effective (12W):** ~2.8 TOPS (78% efficiency)

#### Key Differences from TDA4VM
- Half the AI performance: 4 TOPS vs 8 TOPS
- Lower CPU frequency: 1.2 GHz vs 2.0 GHz
- Lower power envelope: 7-12W vs 10-20W
- Entry-level safety: ASIL-B/C vs ASIL-D
- Cost-sensitive automotive markets

#### Precision Support
- **INT8:** 4 TOPS peak, native MMAv2
- **INT16:** 2 TOPS
- **FP32:** 40 GFLOPS

#### Energy Characteristics
- **FP32:** 2.0 pJ/FLOP
- **Memory:** 20 pJ/byte

---

## Specialized Accelerators

### KPU-T64 (Edge AI / IoT)

**File:** `src/graphs/hardware/models/accelerators/kpu_t64.py` (also in `edge/`)
**Target Market:** Edge AI, IoT, battery-powered devices

#### Compute Architecture
- **Tiles:** 64 tiles in 8×8 mesh topology
- **Processing Elements:** 1,024 total PEs (16 PEs per tile)
- **Tile Allocation:** 70/20/10 ratio
  - 44 INT8 tiles (69%): Computer vision, object detection
  - 13 BF16 tiles (20%): Sensor fusion, lightweight transformers
  - 7 Matrix tiles (11%): Classification heads, embeddings
- **Execution Model:** Programmable SURE (Spatial Unrolled Resource Execution), token-based spatial dataflow

#### Memory Hierarchy (4-stage)
- **L1 (PE-local):** 4 KiB per PE (64 KiB per tile, 4 MiB total)
  - Read energy: 0.3 pJ/byte
  - Write energy: 0.4 pJ/byte
- **L2 (tile-local):** 32 KiB per tile (2 MiB total)
  - Read energy: 0.8 pJ/byte
  - Write energy: 1.0 pJ/byte
- **L3 (distributed scratchpad):** 256 KiB per tile (16 MiB total)
  - Read energy: 2.0 pJ/byte
  - Write energy: 2.5 pJ/byte
- **DRAM:** DDR4-3200, 25.6 GB/s bandwidth, 2 GB typical
  - Read energy: 10 pJ/byte
  - Write energy: 12 pJ/byte

#### Clock & Performance
- **Frequency:** 800 MHz
- **Performance:**
  - INT8: 1,638 GOPS (1.64 TOPS)
  - BF16: 1,638 GFLOPS (1.64 TFLOPS)
  - FP32: 819 GFLOPS (0.82 TFLOPS, half throughput)

#### Process & Power
- **Process Node:** 22nm
- **TDP Range:** 5-15W
- **Cooling:** Passive (5W), Active fan (15W turbo)
- **Compute Energy:**
  - INT8 MAC: 0.35 pJ
  - BF16 MAC: 0.52 pJ
  - FP32 MAC: 1.05 pJ

#### Precision Support
- INT8 (native, 2 ops/cycle per PE)
- BF16 (native, 2 ops/cycle per PE)
- FP32 (1 op/cycle per PE)
- INT32 accumulator for INT8

#### Interconnect
- **Token payload:** 64 bytes
- **Token signature:** 16 bytes (CAM-like matching, 0.6 pJ)
- **Max tokens in flight:** 1,024 per tile
- **Routing energy:** 0.15 pJ per hop
- **NoC energy:** 0.5 pJ per hop (L3 distributed routing)

#### Data Movement Engines
- **DMA (DRAM ↔ L3):** 4 engines per tile, 1.5 pJ/byte transfer
- **BlockMover (L3 ↔ L2):** 2 engines per tile, 0.8 pJ/byte
- **Streamer (L2 ↔ L1):** 4 engines per tile, 0.3 pJ/byte

#### SURE Program Management
- **Program size:** 4 KiB typical
- **Program cache:** 16 programs cached
- **Load energy:** 50 pJ broadcast
- **Cache hit energy:** 1 pJ

---

### KPU-T256 (Mobile/Robotics)

**File:** `src/graphs/hardware/models/accelerators/kpu_t256.py` (also in `mobile/`)
**Target Market:** Mobile, robotics, autonomous systems

#### Compute Architecture
- **Tiles:** 256 tiles in 16×16 mesh topology
- **Processing Elements:** 4,096 total PEs (16 PEs per tile)
- **Tile Allocation:** 70/20/10 ratio
  - 179 INT8 tiles (70%)
  - 51 BF16 tiles (20%)
  - 26 Matrix tiles (10%)
- **Execution Model:** Token-based spatial dataflow

#### Memory Hierarchy (4-stage)
- **L1:** 4 KiB per PE (256 KiB per tile, 64 MiB total)
  - Read: 0.25 pJ/byte, Write: 0.35 pJ/byte
- **L2:** 32 KiB per tile (8 MiB total)
  - Read: 0.6 pJ/byte, Write: 0.8 pJ/byte
- **L3:** 256 KiB per tile (64 MiB total)
  - Read: 1.5 pJ/byte, Write: 1.8 pJ/byte
- **DRAM:** LPDDR5-6400, 102.4 GB/s bandwidth, 8 GB typical
  - Read: 10 pJ/byte, Write: 12 pJ/byte

#### Clock & Performance
- **Frequency:** 1.2 GHz
- **Performance:**
  - INT8: 9,830 GOPS (9.83 TOPS)
  - BF16: 9,830 GFLOPS (9.83 TFLOPS)
  - FP32: 4,915 GFLOPS (4.92 TFLOPS)

#### Process & Power
- **Process Node:** 16nm / 7nm variants
- **TDP Range:** 25-75W
- **Cooling:** Active fan (25W efficient), Vapor chamber (75W performance)
- **Compute Energy:**
  - INT8 MAC: 0.28 pJ
  - BF16 MAC: 0.42 pJ
  - FP32 MAC: 0.84 pJ

#### Precision Support
Same as T64

#### Interconnect
- Token routing: 0.12 pJ per hop (improved from T64)
- NoC energy: 0.4 pJ per hop

#### Data Movement Engines
- DMA: 1.2 pJ/byte
- BlockMover: 0.6 pJ/byte
- Streamer: 0.25 pJ/byte

---

### KPU-T768 (Automotive/Datacenter)

**File:** `src/graphs/hardware/models/accelerators/kpu_t768.py` (also in `automotive/`)
**Target Market:** Autonomous vehicles L4/L5, edge datacenter

#### Compute Architecture
- **Tiles:** 768 tiles in 24×32 mesh topology
- **Processing Elements:** 12,288 total PEs (16 PEs per tile)
- **Tile Allocation:** 70/20/10 ratio
  - 537 INT8 tiles (70%)
  - 154 BF16 tiles (20%)
  - 77 Matrix tiles (10%)

#### Memory Hierarchy (4-stage)
- **L1:** 4 KiB per PE (768 KiB per tile, 192 MiB total)
  - Read: 0.2 pJ/byte, Write: 0.3 pJ/byte
- **L2:** 32 KiB per tile (24 MiB total)
  - Read: 0.5 pJ/byte, Write: 0.6 pJ/byte
- **L3:** 256 KiB per tile (192 MiB total)
  - Read: 1.2 pJ/byte, Write: 1.5 pJ/byte
- **DRAM:** HBM2, 204.8 GB/s bandwidth, 16 GB typical
  - Read: 5 pJ/byte, Write: 6 pJ/byte (HBM2 lower than DDR4)

#### Clock & Performance
- **Frequency:** 1.5 GHz
- **Performance:**
  - INT8: 36,864 GOPS (36.9 TOPS)
  - BF16: 36,864 GFLOPS (36.9 TFLOPS)
  - FP32: 18,432 GFLOPS (18.4 TFLOPS)

#### Process & Power
- **Process Node:** 7nm / 4nm variants
- **TDP Range:** 75-250W
- **Cooling:** Active liquid cooling
- **Compute Energy:**
  - INT8 MAC: 0.25 pJ
  - BF16 MAC: 0.38 pJ
  - FP32 MAC: 0.75 pJ

#### Interconnect
- Token routing: 0.1 pJ per hop (optimized for larger mesh)
- NoC energy: 0.35 pJ per hop
- Routing distance factor: 1.3 (larger mesh compensation)

#### Data Movement Engines
- DMA: 1.0 pJ/byte (optimized for HBM2)
- BlockMover: 0.5 pJ/byte
- Streamer: 0.2 pJ/byte

---

### Hailo-8 (Computer Vision AI Accelerator)

**File:** `src/graphs/hardware/models/edge/hailo8.py`
**Target Market:** Edge AI cameras, drones, robots, embedded vision

#### Compute Architecture
- **Processing Units:** 32 dataflow units (estimated)
- **Operations per unit:** 500 ops/clock
- **Execution Model:** Structure-driven graph mapping, distributed on-chip memory fabric
- **Specialization:** Network-specific compilation (custom dataflow per model)

#### Memory Hierarchy
- **On-chip memory:** 8 MB total (estimated), distributed SRAM fabric
- **Local memory per unit:** 512 KB (estimated)
- **External DRAM:** None (all-on-chip design, no Von Neumann bottleneck)
- **On-chip bandwidth:** ~200 GB/s (estimated)
- **Memory energy:** 2 pJ/byte (on-chip SRAM)

#### Clock & Performance
- **Frequency:** 1.6 GHz (estimated, fixed frequency)
- **Performance:**
  - INT8: 26 TOPS (marketed, achievable)
  - INT4: 52 TOPS (theoretical, 2× INT8)
  - Realistic sustained: ~22 TOPS INT8 (85% efficiency)

#### Process & Power
- **Process Node:** 16nm TSMC
- **TDP:** 2.5W (single operating point, no DVFS needed)
- **Cooling:** Passive heatsink (small)
- **Efficiency:** 10.4 TOPS/W (best in class)
- **Compute Energy:** 0.5 pJ/FLOP (highly optimized dataflow)

#### Precision Support
- INT8 (native, primary)
- INT4 (2× throughput, not primary use case)

#### BOM Cost (10K units)
- Silicon: $25 (small efficient die)
- Package: $8
- Memory: $0 (all on-chip)
- Assembly: $4
- Thermal: $1
- **Total BOM:** $40
- **Retail:** $160 (M.2 module)

---

### Hailo-10H (Generative AI Edge Accelerator)

**File:** `src/graphs/hardware/models/edge/hailo10h.py`
**Target Market:** Edge Gen AI, on-device LLMs, vision-language models

#### Compute Architecture
- **Processing Units:** 40 dataflow units (more than Hailo-8 for transformers)
- **Operations per unit:** 1,000 INT4 ops/clock, 500 INT8 ops/clock
- **Execution Model:** Enhanced dataflow with KV cache support
- **Specialization:** Transformer-optimized (attention, LayerNorm, SwiGLU)

#### Memory Hierarchy
- **On-chip memory:** 12 MB (larger for KV cache)
- **Local memory per unit:** 512 KB
- **External DRAM:** 4-8 GB LPDDR4X (for model weights + KV cache)
- **LPDDR4X bandwidth:** ~40 GB/s
- **Memory energy:** 15 pJ/byte (LPDDR4X, not on-chip)

#### Clock & Performance
- **Frequency:** 1.0 GHz (lower than Hailo-8, more units)
- **Performance:**
  - INT4: 40 TOPS (primary for LLMs)
  - INT8: 20 TOPS (secondary)
  - Realistic sustained: ~32 TOPS INT4 (80%), ~17 TOPS INT8 (85%)

#### Process & Power
- **Process Node:** 16nm (same as Hailo-8)
- **TDP:** 2.5W (same excellent thermal)
- **Efficiency:** 16 TOPS/W INT4
- **Compute Energy:** 0.55 pJ/FLOP (slightly higher than Hailo-8)

#### Precision Support
- INT4 (primary for LLMs)
- INT8 (secondary)
- FP16 (supported)

#### Real-World Performance
- First token: <1 second (2B parameter LLMs)
- Token generation: 10 tokens/sec sustained
- Stable Diffusion 2.1: <5 seconds per image

#### BOM Cost (10K units, 2025)
- Silicon: $30
- Package: $10
- Memory: $20 (4GB LPDDR4X on-module)
- Assembly: $5
- Thermal: $1
- **Total BOM:** $70
- **Retail:** $240 (estimated)

---

## FPGA

### Xilinx Vitis AI DPU (B4096)

**File:** `src/graphs/hardware/models/accelerators/xilinx_vitis_ai_dpu.py`
**Target Market:** Embodied AI, FPGA-based edge inference

#### Compute Architecture
- **MAC Units:** 4,096 MACs
- **Tiles:** 64 tiles (estimated)
- **Operations per tile:** 64 ops
- **Execution Model:** AIE-ML v1 (2D array of INT8 ALUs), FPGA-based reconfigurable
- **Tile allocation:** 8 vector lanes per tile

#### Memory Hierarchy
- **Scratchpad per tile:** 64 KB (tile-local)
- **L2 cache:** 4 MB (estimated)
- **External DRAM:** DDR4, 50 GB/s bandwidth, 8 GB typical
- **Memory energy:** 15 pJ/byte (similar to GPU, FPGA I/O)

#### Clock & Performance
- **Frequency:** 1.25 GHz (Versal VE2302)
- **Performance:**
  - INT8: 10.24 TOPS theoretical, 7.68 TOPS realistic (75% efficiency)
  - FP16: 1.92 TFLOPS
  - FP32: 0.96 TFLOPS (not native)

#### Process & Power
- **Process Node:** 7nm (Versal ACAP)
- **TDP:** 15-20W (VE2302 edge-optimized)
- **Average Power:** 17.5W
- **Idle Power:** 3W
- **Dynamic Power:** 14.5W
- **Cooling:** Active fan
- **Compute Energy:**
  - INT8 op: 1.89 pJ
  - FP32 equivalent: 7.56 pJ/FLOP

#### Precision Support
- INT8 (native, best performance)
- FP16 (AIE support)
- FP32 (emulated, 0.125× speed)
- INT32 accumulator for INT8

#### Platform
Versal VE2302 ACAP

---

## Research Prototypes

### Stanford Plasticine CGRA (v2)

**File:** `src/graphs/hardware/models/accelerators/stanford_plasticine_cgra.py`
**Target Market:** Research platform, embodied AI

#### Compute Architecture
- **PCUs (Pattern Compute Units):** 32 PCUs
- **MACs per PCU:** 8 (medium granularity)
- **Total MACs:** 256
- **Operations per PCU:** 8 ops
- **Execution Model:** Spatial dataflow, coarse-grained reconfigurable
- **Reconfiguration overhead:** 1,000 cycles (conservative, Achilles heel)

#### Memory Hierarchy
- **Scratchpad per PCU:** 64 KB
- **L2 cache:** 2 MB shared
- **External DRAM:** DDR4, 40 GB/s bandwidth, 4 GB typical
- **Memory energy:** 12 pJ/byte (on-chip network)

#### Clock & Performance
- **Frequency:** 1.0 GHz (typical for CGRAs)
- **Performance:**
  - INT8: 10.24 TOPS theoretical, 6.14 TOPS realistic (60% efficiency)
  - FP16: 1.54 TFLOPS
  - FP32: 0.77 TFLOPS

#### Process & Power
- **Process Node:** 7nm (hypothetical)
- **TDP:** 15W (embodied AI range: 10-25W)
- **Idle Power:** 2W
- **Dynamic Power:** 13W
- **Cooling:** Active fan
- **Compute Energy:**
  - INT8 op: 2.12 pJ
  - FP32 equivalent: 8.48 pJ/FLOP

#### Precision Support
- INT8 (native, best)
- FP16 (0.25× speed)
- FP32 (0.125× speed)
- INT32 accumulator

#### Design Trade-offs
- Medium-grained PCUs (balanced coverage vs overhead)
- Spatial execution (no warp concept)
- High reconfiguration penalty

---

## IP Cores

### Cadence Tensilica Vision Q8 (7th Gen)

**File:** `src/graphs/hardware/models/ip_cores/cadence_vision_q8.py`

#### Compute Architecture
- **Type:** Vision DSP IP core (licensable)
- **DSP Units:** 32 equivalent processing elements
- **SIMD Engine:** 1024-bit wide for vision processing
- **Operations per unit:** 119 INT8 ops/cycle
- **Execution Model:** Heterogeneous (Vector + Scalar units)

#### Memory Hierarchy
- **L1 per unit:** 32 KB
- **L2 cache:** 1 MB shared
- **External memory:** Up to 4 GB, 40 GB/s typical
- **Memory energy:** 12 pJ/byte

#### Clock & Performance
- **Frequency range:** 600 MHz - 1.2 GHz
- **Sustained:** 1.0 GHz
- **Performance:**
  - INT8/INT16: 3.8 TOPS (vision optimized)
  - FP32: 129 GFLOPS
  - FP16: 2× FP32

#### Power
- **Typical:** 0.5-1W @ 1.0 GHz
- **Efficiency:** 3-7 TOPS/W
- **Cooling:** Passive mobile
- **Compute Energy:** 1.5 pJ/FLOP

#### Precision Support
- INT8 (native, 1.0× speed)
- INT16 (native, 1.0× speed)
- FP32 (0.034× relative to INT8)
- FP16 (0.5× FP32)

#### Calibration Status
Estimated (based on Cadence specs)

---

### Synopsys ARC EV7x (4-core)

**File:** `src/graphs/hardware/models/ip_cores/synopsys_arc_ev7x.py`

#### Compute Architecture
- **Type:** Embedded Vision Processor IP (licensable)
- **Cores:** 1-4 cores (scalable), modeled as 4-core
- **Processing Units:** 128 equivalent units
- **VPUs:** 1-4 Vector Processing Units (512-bit wide)
- **DNN Accelerator:** 880-14,080 MACs (scalable)
- **Operations per unit:** 273 INT8 MACs/cycle
- **Execution Model:** Heterogeneous (VPUs + DNN accelerator), ARCv2 RISC ISA base

#### Memory Hierarchy
- **L1 per unit:** 32 KB
- **L2 cache:** 4 MB shared
- **External memory:** Up to 8 GB, 60 GB/s bandwidth
- **Memory energy:** 14 pJ/byte

#### Clock & Performance
- **Frequency range:** 600 MHz - 1.2 GHz
- **Sustained:** 1.0 GHz
- **Performance:**
  - INT8: 35 TOPS (DNN accelerator)
  - INT16: 17.5 TOPS
  - FP32: 8.8 GFLOPS (4 cores × 2.2 GFLOPS)

#### Power
- **Typical:** 3-5W @ 1.0 GHz
- **TDP:** 5W
- **Efficiency:** 7-10 TOPS/W
- **Cooling:** Automotive passive
- **Compute Energy:** 1.4 pJ/FLOP

#### Precision Support
- INT8 (native, DNN accelerator)
- INT16 (native via VPUs)
- INT32 (supported)
- FP32 (via VPU FPU, 0.00025× relative)

#### Process Node
16nm FinFET (typical)

#### Calibration Status
Estimated (based on Synopsys specs)

---

### CEVA NeuPro-M NPM11

**File:** `src/graphs/hardware/models/ip_cores/ceva_neupro_npm11.py`

#### Compute Architecture
- **Type:** Neural Processing IP Core (licensable)
- **NPU Units:** 64 equivalent processing elements
- **Configuration:** Single NeuPro-M engine
- **Operations per unit:** 312 INT8 MACs/cycle
- **Execution Model:** Heterogeneous (Tensor + Vector + Scalar units)

#### Memory Hierarchy
- **L1 per unit:** 64 KB
- **L2 cache:** 2 MB shared
- **External memory:** Up to 8 GB, 50 GB/s typical
- **Memory energy:** 12 pJ/byte

#### Clock & Performance
- **Frequency range:** 800 MHz - 1.25 GHz
- **Sustained:** 1.0 GHz
- **Performance:**
  - INT8: 20 TOPS (primary)
  - INT16: 10 TOPS
  - FP16: 10 TFLOPS
  - INT4: 40 TOPS (2× INT8)

#### Power
- **Typical:** 2W @ 1.0 GHz
- **Efficiency:** ~10 TOPS/W
- **Cooling:** Passive mobile
- **Compute Energy:** 1.2 pJ/FLOP

#### Precision Support
- INT8 (native, 1.0× speed)
- INT16 (native, 0.5× speed)
- FP16 (native, 0.5× speed)
- INT4 (native, 2.0× speed)

#### Calibration Status
Estimated (based on CEVA specs)

#### Process Node
Depends on SoC integration

---

## Conceptual Reference Architectures

### DFM-128 (Data Flow Machine)

**File:** `src/graphs/hardware/models/research/dfm_128.py`

#### Compute Architecture
- **Type:** Classic Data Flow Machine (Jack Dennis, MIT)
- **CAM Slots:** 128 instruction tokens
- **Processing Elements:** 8 PEs (VLIW-like datapath)
  - 4× Integer ALUs (16 INT ops/cycle)
  - 2× FP Units (16 FP32 ops/cycle with FMA)
  - 1× SFU (2 ops/cycle, transcendentals)
  - 1× LSU (2 loads or stores per cycle)
- **Execution Model:** Token-based execution, CAM-based instruction matching
- **Routing Network:** Crossbar (128 CAM → 8 PEs → 128 CAM)

#### Memory Hierarchy
- **L1 Cache:** 64 KB (32 KB I$ + 32 KB D$), 8 KB per PE
- **L2 Cache:** 512 KB unified
- **External memory:** DDR4-3200 single channel, 25.6 GB/s, 16 GB
- **Memory energy:** 15 pJ/byte (DDR4)

#### Clock & Performance
- **Frequency:** 2.0 GHz
- **Performance:**
  - FP32: 64 GFLOPS (with FMA)
  - INT8: 32 GOPS
  - FP16/BF16: 128 GFLOPS

#### Process & Power
- **Process Node:** 7nm (comparable to modern superscalar)
- **TDP Range:** 15W-35W
- **Thermal profiles:** 15W (low power), 25W (nominal), 35W (performance)
- **Cooling:** Passive heatsink (15W), Active fan (25-35W)
- **Compute Energy:** 2.5 pJ/FLOP (similar to modern CPU)

#### Precision Support
- FP32 (native, 1.0× speed)
- FP16/BF16 (2.0× speed)
- INT8 (1.0× relative to INT baseline)
- INT4 (2.0× INT8)
- INT32 accumulator

#### Microarchitectural Features
- **Token Matching:** Fully associative (128-way CAM lookup)
- **Token Operand Slots:** Data or pointers
- **Instruction Ready Queue:** CAM-based ready detection
- **Bypass Network:** Token routing for result forwarding

#### Energy Characteristics (Unique to DFM)
- CAM lookups expensive (associative search across 128 entries)
- Token matching overhead dominates for small workloads
- No instruction fetch/decode (saves energy)
- Routing network energy (crossbar is power-hungry)
- Better efficiency at high ILP

#### Similarity to Modern Processors
- Register renaming → Token operand slots
- Reservation stations → CAM instruction slots
- Issue queue → CAM ready queue
- Execution ports → Processing elements
- Bypass network → Token routing

---

## Summary Comparison Tables

### CPU Comparison

| Platform | Cores | Threads | Process | Base Clock | Boost Clock | L1D/core | L2/core | L3 Total | Memory | Bandwidth | TDP | SIMD Width |
|----------|-------|---------|---------|------------|-------------|----------|---------|----------|--------|-----------|-----|------------|
| **Intel 8490H** | 60 | 120 | Intel 7 | 2.0 GHz | 3.5 GHz | 48 KB | 2 MB | 112.5 MB | DDR5-4800 8ch | 307 GB/s | 350W | AVX-512 + AMX |
| **Intel 8592+** | 64 | 128 | Intel 7 | 1.9 GHz | 3.9 GHz | 48 KB | 2 MB | 120 MB | DDR5-4800 8ch | 307 GB/s | 350W | AVX-512 + AMX |
| **Intel Granite** | 128 | 256 | Intel 3 | 2.0 GHz | 3.8 GHz | 48 KB | 2 MB | 320 MB | DDR5-5600 8-12ch | 358-537 GB/s | 500W | AVX-512 + Enhanced AMX |
| **AMD 9654** | 96 | 192 | TSMC 5nm | 2.4 GHz | 3.7 GHz | 32 KB | 1 MB | 384 MB | DDR5-4800 12ch | 461 GB/s | 360W | Double-pumped AVX-512 |
| **AMD 9754** | 128 | 256 | TSMC 5nm | 2.25 GHz | 3.1 GHz | 32 KB | 1 MB | 512 MB | DDR5-4800 12ch | 461 GB/s | 360W | Double-pumped AVX-512 |
| **AMD Turin** | 192 | 384 | TSMC 3nm | 2.5 GHz | 3.8 GHz | 48 KB | 1 MB | 768 MB | DDR5-6000 12ch | 576 GB/s | 500W | Native AVX-512 |
| **Ampere 128** | 128 | 128 | TSMC 5nm | 3.6 GHz | 3.6 GHz | 64 KB | 2 MB | 48 MB | DDR5-5200 8ch | 333 GB/s | 210W | 2×128-bit NEON+SVE |
| **Ampere 192** | 192 | 192 | TSMC 5nm | 3.6 GHz | 3.6 GHz | 64 KB | 2 MB | 64 MB | DDR5-5200 8ch | 333 GB/s | 283W | 2×128-bit NEON+SVE |

### GPU Comparison

| Platform | Architecture | SMs | CUDA Cores | Tensor Cores | L1/SM | L2 Total | Memory | Bandwidth | TDP | FP32 TFLOPS | INT8 TOPS |
|----------|-------------|-----|------------|--------------|-------|----------|--------|-----------|-----|-------------|-----------|
| **H100 PCIe** | Hopper | 132 | 16,896 | 528 (Gen4) | 256 KB | 50 MB | 80GB HBM2e | 2 TB/s | 350W | 60 | 1,500 |
| **A100 SXM4** | Ampere | 108 | 13,824 | 432 (Gen3) | 192 KB | 40 MB | 80GB HBM2e | 2 TB/s | 400W | 19.5 | 624 |
| **V100 SXM2** | Volta | 80 | 5,120 | 640 (Gen1) | 128 KB | 6 MB | 32GB HBM2 | 900 GB/s | 300W | 15.7 | 125 |
| **T4** | Turing | 40 | 2,560 | 320 (Gen2) | 96 KB | 4 MB | 16GB GDDR6 | 320 GB/s | 70W | 8.1 | 130 |
| **Jetson Orin AGX** | Ampere | 16 | 2,048 | 64 | 128 KB | 4 MB | 64GB LPDDR5 | 205 GB/s | 15-60W | - | 85 (peak) |
| **Jetson Orin Nano** | Ampere | 16 | 1,024 | 32 | 128 KB | 2 MB | 8GB LPDDR5 | 68-102 GB/s | 7-15W | - | 21-40 (peak) |
| **Jetson Thor** | Blackwell | 64 | - | - | 256 KB | 8 MB | 128GB HBM3 | 450 GB/s | 30-100W | - | 1,000 (peak) |
| **Mali-G78 MP20** | Valhall | 20 cores | 320 lanes | None | 32 KB | 2 MB | Shared DRAM | ~40 GB/s | 3-5W | 1.94 | ~2 |

### TPU Comparison

| Platform | Year | Array Size | MACs | Clock | Memory | Bandwidth | BF16 TFLOPS | INT8 TOPS | TDP | Process |
|----------|------|------------|------|-------|--------|-----------|-------------|-----------|-----|---------|
| **TPU v1** | 2017 | 256×256 | 65,536 | 700 MHz | 8GB DDR3 | 34 GB/s | - | 92 | 75W | ~28nm |
| **TPU v3** | 2018 | 128×128×2 | 32,768 | 940 MHz | 16GB HBM | 900 GB/s | 123 | 246 | 200W | ~16nm |
| **TPU v4** | 2021 | 128×128×2 | 32,768 | 1,050 MHz | 32GB HBM2e | 1.2 TB/s | 275 | 550 | 350W | ~7nm |
| **TPU v5p** | 2023 | 128×128×2+ | 32,768+ | 1,100 MHz | 32GB HBM3 | 1.6 TB/s | 459 | 918 | 400W | ~5nm |
| **Coral Edge** | 2019 | 64×64 | 4,096 | 500 MHz | Host mem | 4 GB/s | - | 4 | 2W | 14nm |

### DSP & Automotive Comparison

| Platform | Process | TDP | Peak TOPS (INT8) | Effective TOPS | Memory BW | Target Market |
|----------|---------|-----|------------------|----------------|-----------|---------------|
| **QCS6490** | 6nm | 5-15W | 12 | 6.6-8.8 | 40 GB/s | Edge AI, IoT |
| **QRB5165** | 7nm | 7W | 15 | 5.4 | 44 GB/s | Robotics |
| **SA8775P** | 5nm | 20-45W | 32 | 6.5-23.9 | 90 GB/s | ADAS L2+/L3 |
| **Snapdragon Ride** | 4nm | 65-130W | 700 | 204-432 | 200 GB/s | L3/L4/L5 Autonomous |
| **TDA4VM** | 16nm | 10-20W | 8 | 4.8-6.1 | 60 GB/s | ADAS L2 |
| **TDA4VH** | Advanced | 20-35W | 32 | 19-23.7 | 100 GB/s | ADAS L3-4 |
| **TDA4AL** | Newer | 10-18W | 8 | 6.0-6.4 | 60 GB/s | ADAS L2-3 |
| **TDA4VL** | Newer | 7-12W | 4 | 2.2-2.8 | 60 GB/s | Entry ADAS |

### Accelerator Comparison

| Platform | Tiles/Units | PEs/MACs | Clock (GHz) | INT8 TOPS | Process | TDP (W) | Memory BW | Architecture Type |
|----------|-------------|----------|-------------|-----------|---------|---------|-----------|-------------------|
| **KPU-T64** | 64 tiles | 1,024 PEs | 0.8 | 1.64 | 22nm | 5-15 | 25.6 GB/s (DDR4) | Domain Flow |
| **KPU-T256** | 256 tiles | 4,096 PEs | 1.2 | 9.83 | 16/7nm | 25-75 | 102.4 GB/s (LPDDR5) | Domain Flow |
| **KPU-T768** | 768 tiles | 12,288 PEs | 1.5 | 36.9 | 7/4nm | 75-250 | 204.8 GB/s (HBM2) | Domain Flow |
| **Hailo-8** | 32 units | - | 1.6 | 26 | 16nm | 2.5 | 200 GB/s (on-chip) | Processor Array, all-on-chip |
| **Hailo-10H** | 40 units | - | 1.0 | 20 (40 INT4) | 16nm | 2.5 | 40 GB/s (LPDDR4X) | Processor Array, transformer-opt |
| **DPU-B4096** | 64 tiles | 4,096 MACs | 1.25 | 7.68 | 7nm | 15-20 | 50 GB/s (DDR4) | FPGA reconfigurable |
| **Plasticine-v2** | 32 PCUs | 256 MACs | 1.0 | 6.14 | 7nm | 15 | 40 GB/s | CGRA spatial dataflow |
| **Vision Q8** | 32 units | - | 1.0 | 3.8 | - | 0.5-1 | 40 GB/s | Vision DSP IP |
| **ARC EV7x** | 128 units | 14,080 MACs | 1.0 | 35 | 16nm | 3-5 | 60 GB/s | Vision IP, VPU+DNN |
| **NeuPro NPM11** | 64 units | - | 1.0 | 20 | - | 2 | 50 GB/s | NPU IP, tensor+vector |
| **DFM-128** | 8 PEs | - | 2.0 | 0.032 (INT) | 7nm | 15-35 | 25.6 GB/s (DDR4) | Token-based dataflow |

---

## References

All specifications extracted from hardware model files in:
- `src/graphs/hardware/models/datacenter/`
- `src/graphs/hardware/models/edge/`
- `src/graphs/hardware/models/automotive/`
- `src/graphs/hardware/models/mobile/`
- `src/graphs/hardware/models/accelerators/`
- `src/graphs/hardware/models/ip_cores/`
- `src/graphs/hardware/models/research/`

**Note:** Some specifications (particularly for next-generation and research platforms) are estimates based on architectural trends and available technical literature.
