# NVIDIA Jetson Hardware Specifications Reference

This document contains official hardware specifications for NVIDIA Jetson platforms,
sourced from NVIDIA technical briefs and datasheets (2024-2025).

## Key Architectural Constants

**NVIDIA Ampere Architecture (Jetson Orin family):**
- 128 CUDA cores per SM
- 4 Tensor Cores (3rd generation) per SM
- FP32 FMA: 2 ops per clock per CUDA core

**NVIDIA Blackwell Architecture (Jetson Thor):**
- 128 CUDA cores per SM
- 96 Tensor Cores (5th generation) total
- SM110 architecture

## Jetson Thor (2025 - Blackwell Architecture)

### GPU Configuration
- **Total CUDA cores**: 2,560
- **Total Tensor cores**: 96 (5th gen)
- **Streaming Multiprocessors**: 20 SMs
- **Per SM**: 128 CUDA cores
- **GPU Architecture**: Blackwell (SM110)
- **Graphics Processor Clusters (GPC)**: 3
- **Texture Processor Clusters (TPC)**: 10

### Performance
- Up to 2,070 TFLOPs FP4 (sparse)
- Up to 1,035 TFLOPs FP8 (dense)

### Memory
- 128 GB LPDDR5X

### CPU
- 14× Arm Neoverse V3AE cores (up to 2.6 GHz)

---

## Jetson AGX Orin 64GB (Ampere Architecture)

### GPU Configuration
- **Total CUDA cores**: 2,048
- **Total Tensor cores**: 64 (3rd gen)
- **Streaming Multiprocessors**: 16 SMs
- **Per SM**: 128 CUDA cores, 4 Tensor cores
- **GPU Architecture**: Ampere
- **Graphics Processor Clusters (GPC)**: 2
- **Texture Processor Clusters (TPC)**: 8

### Performance
- Up to 5.3 TFLOPS FP32 (CUDA)
- Up to 170 Sparse TOPS INT8 (Tensor)
- Up to 275 TOPS (combined AI performance)

### Memory
- 64 GB LPDDR5 (204.8 GB/s bandwidth)

### CPU
- 12-core Arm Cortex-A78AE (up to 2.2 GHz)

### Power Modes
- 15W passive cooling
- 30W active cooling
- 60W active cooling

---

## Jetson AGX Orin 32GB (Ampere Architecture)

### GPU Configuration
- **Total CUDA cores**: 1,792
- **Total Tensor cores**: 56 (3rd gen)
- **Streaming Multiprocessors**: 14 SMs
- **Per SM**: 128 CUDA cores, 4 Tensor cores
- **Texture Processor Clusters (TPC)**: 7

### Performance
- Proportionally scaled from 64GB variant

### Memory
- 32 GB LPDDR5

---

## Jetson Orin Nano 8GB (Ampere Architecture)

### GPU Configuration
- **Total CUDA cores**: 1,024
- **Total Tensor cores**: 32 (3rd gen)
- **Streaming Multiprocessors**: 8 SMs
- **Per SM**: 128 CUDA cores, 4 Tensor cores
- **GPU Architecture**: Ampere

### Performance
- Up to 40 TOPS (INT8)

### Memory
- 8 GB LPDDR5

### Power Modes
- 7W mode
- 15W mode

---

## Jetson Orin Nano 4GB (Ampere Architecture)

### GPU Configuration
- **Total CUDA cores**: 512
- **Total Tensor cores**: 16 (3rd gen)
- **Streaming Multiprocessors**: 4 SMs
- **Per SM**: 128 CUDA cores, 4 Tensor cores

### Performance
- Up to 20 TOPS (INT8)

### Memory
- 4 GB LPDDR5

### Power Modes
- 5W mode
- 10W mode

---

## Architecture Comparison: Ampere vs Volta

**Ampere (Jetson Orin):**
- 128 CUDA cores per SM
- 4 Tensor Cores (3rd gen) per SM
- Improved performance per watt
- Enhanced Tensor Core capabilities

**Volta (Jetson AGX Xavier - previous generation):**
- 64 CUDA cores per SM
- 8 Tensor Cores (1st gen) per SM

**Key takeaway**: Ampere doubled the CUDA cores per SM compared to Volta, providing significant performance improvements while maintaining similar power envelopes.

---

## Sources

- NVIDIA Jetson AGX Orin Series Technical Brief v1.2 (July 2022)
- NVIDIA Jetson Orin Nano Series Datasheet DS-11105-001 v1.1
- NVIDIA Jetson Thor Specifications (2025)
- NVIDIA Developer Forums and Technical Documentation

---

## Usage in Hardware Models

When creating hardware resource models for Jetson platforms, ensure:

1. `compute_units` = number of SMs (not total CUDA cores)
2. `cuda_cores_per_sm` = 128 (for Ampere/Blackwell)
3. `tensor_cores_per_sm` = 4 (for Orin family)
4. `ops_per_clock_per_core` = 2.0 (FMA: multiply-add)

**Example calculation for Jetson AGX Orin 64GB:**
- 16 SMs × 128 CUDA cores/SM = 2,048 total CUDA cores ✓
- Peak FP32 throughput = 16 SMs × 128 cores × 2 ops/clock × clock_freq
