# Physics-Based Energy Model - Implementation Status

## Overview

We are refactoring the energy model to be **physics-based** rather than architecture-based:

**OLD MODEL (Wrong)**:
```
energy_per_flop = TDP / peak_FLOPS  # Architecture-dependent
```

**NEW MODEL (Correct)**:
```
total_energy = base_alu_energy + architectural_overhead + memory_energy

base_alu_energy = PROCESS_NODE_ENERGY[nm] × CIRCUIT_TYPE_MULTIPLIER[type]
```

## Key Insight

**The FP32 ALU circuit is the same across all architectures!**
- Same IEEE-754 specification
- Same standard cell logic gates
- Same transistor switching energy (C × V²)
- **Frequency does NOT affect energy per operation** (only power)

Therefore:
- **GPU FP32 ALU @ 7nm**: 1.8 pJ
- **TPU FP32 ALU @ 7nm**: 1.8 pJ (same circuit!)
- **KPU FP32 ALU @ 7nm**: 1.8 pJ (same circuit!)
- **CPU FP32 ALU @ 7nm**: 1.8 pJ (same circuit!)

Differences come from:
1. **Architectural overhead** (control logic, scheduling, coherence)
2. **Memory energy** (data movement through hierarchy)
3. **NOT from the base ALU!**

## Implementation Progress

### ✅ Completed

1. **Energy Constants** (`resource_model.py`)
   - `PROCESS_NODE_ENERGY`: Energy by process node (4nm-28nm)
   - `CIRCUIT_TYPE_MULTIPLIER`: Efficiency by circuit type
   - `get_base_alu_energy()`: Calculate base energy from process + circuit type

2. **ComputeFabric Dataclass** (`resource_model.py`)
   - Multi-fabric support (CUDA + Tensor Cores, INT8 + Matrix tiles, Scalar + SIMD)
   - Per-fabric energy, frequency, throughput
   - Methods: `get_energy_per_op()`, `get_peak_ops_per_sec()`, `get_peak_power()`

3. **HardwareResourceModel Update** (`resource_model.py`)
   - Added `compute_fabrics: Optional[List[ComputeFabric]]` field
   - Backward compatible with legacy `energy_per_flop_fp32`

4. **ARM CPU Model** (`models/datacenter/cpu_arm.py`)
   - Scalar ALU fabric (standard_cell)
   - NEON SIMD fabric (simd_packed, 10% more efficient)
   - Parametrized: `cpu_arm_resource_model(num_cores, process_node_nm, freq, name)`
   - Convenience functions: `jetson_orin_agx_cpu_arm()`, `graviton3_arm()`, etc.

### 🚧 In Progress

5. **Update Existing Hardware Models**
   - [ ] H100: CUDA cores + Tensor Cores
   - [ ] KPU-T256: INT8 tiles + BF16 tiles + Matrix tiles
   - [ ] TPU Edge Pro: Single systolic array (process-node-only energy)
   - [ ] Jetson Orin AGX GPU: CUDA cores + Tensor Cores
   - [ ] x86 CPU: Scalar ALUs (custom_datacenter @ 5GHz) + AVX-512 (simd_packed)

### 📋 TODO

6. **Energy Analyzer Updates** (`analysis/energy.py`)
   - Fabric-aware energy calculation
   - Select fabric based on operation type and precision
   - Sum: base_alu + architectural_overhead + memory

7. **Architectural Energy Models** (`hardware/architectural_energy.py`)
   - `GPUArchitecturalEnergy`: SIMT overhead (~1.4 pJ)
   - `TPUArchitecturalEnergy`: Systolic overhead (~0.3 pJ)
   - `KPUArchitecturalEnergy`: Dataflow overhead (~1.3 pJ)
   - `CPUArchitecturalEnergy`: MIMD overhead (~8 pJ ARM, ~25 pJ x86)

8. **Mapper Integration** (`hardware/mappers/`)
   - Update mappers to select fabric based on operation
   - Pass fabric to energy calculator

9. **Testing & Validation**
   - Verify all 7nm chips have ~1.8 pJ base ALU
   - Verify architectural overhead is separate
   - Test architectural comparison (CPU vs GPU vs TPU vs KPU)

## Energy Model Summary

### Process Node Energy (Standard Cell FP32 ALU)
```
Process Node    Energy (pJ)    Technology
----------------------------------------------
4nm            1.3            TSMC N4/N4P
5nm            1.5            TSMC N5, Samsung 5LPE
7nm            1.8            TSMC N7, Samsung 7LPP
8nm            1.9            Samsung 8LPP
10nm           2.1            Intel 10nm/7
12nm           2.5            TSMC 12FFC
16nm           2.7            TSMC 16FFC
28nm           4.0            TSMC 28HPC+
```

### Circuit Type Multipliers
```
Circuit Type           Multiplier    Rationale
-----------------------------------------------------------------
standard_cell          1.0×          Baseline
tensor_core            0.85×         15% more efficient (amortized control)
simd_packed            0.90×         10% more efficient (packed ops)
custom_datacenter      2.75×         5+ GHz custom circuits
```

### Example Energy Calculations

**GPU H100 @ 5nm:**
- CUDA Core: 1.5 pJ × 1.0 = **1.5 pJ**
- Tensor Core: 1.5 pJ × 0.85 = **1.28 pJ** (15% better)

**KPU-T256 @ 16nm:**
- INT8 Tile: 2.7 pJ × 1.0 = **2.7 pJ**
- Matrix Tile: 2.7 pJ × 0.85 = **2.3 pJ** (15% better)

**ARM CPU @ 8nm:**
- Scalar ALU: 1.9 pJ × 1.0 = **1.9 pJ**
- NEON SIMD: 1.9 pJ × 0.90 = **1.71 pJ** (10% better)

**x86 CPU @ 10nm (5 GHz custom):**
- Scalar ALU: 2.1 pJ × 2.75 = **5.8 pJ** (high-frequency custom circuits)
- AVX-512: 2.1 pJ × 0.90 = **1.9 pJ** (SIMD efficiency)

### Architectural Overhead (Separate from Base ALU)

```
Architecture    Overhead (pJ)    Components
------------------------------------------------------------------
GPU SIMT        ~1.4             Warp scheduling, register file, coherence
TPU Systolic    ~0.3             Weight shift, accumulator access
KPU Dataflow    ~1.3             Token matching, NoC routing, tile control
CPU ARM         ~8               Fetch, decode, rename, predict, coherence
CPU x86         ~25              Higher overhead than ARM (complex CISC)
```

### Total Energy Per Operation

```
Architecture          Base ALU    Arch Overhead    Total
--------------------------------------------------------------
GPU H100 (CUDA)       1.5 pJ     1.4 pJ           2.9 pJ
GPU H100 (Tensor)     1.28 pJ    0.5 pJ           1.8 pJ
TPU Edge Pro          1.8 pJ     0.3 pJ           2.1 pJ
KPU-T256 (INT8)       2.7 pJ     1.3 pJ           4.0 pJ
KPU-T256 (Matrix)     2.3 pJ     0.5 pJ           2.8 pJ
ARM CPU (scalar)      1.9 pJ     8.0 pJ           9.9 pJ
ARM CPU (NEON)        1.71 pJ    4.0 pJ           5.7 pJ
x86 CPU (scalar)      5.8 pJ     25.0 pJ          30.8 pJ
x86 CPU (AVX-512)     1.9 pJ     10.0 pJ          11.9 pJ
```

**Key Observations:**
- ✓ All 7-8nm chips have ~1.8-1.9 pJ base ALU (physically consistent!)
- ✓ TPU is most efficient due to minimal architectural overhead
- ✓ x86 CPUs have highest overhead due to complex CISC architecture
- ✓ Tensor Cores / Matrix tiles are 15% more efficient than standard fabrics

## Next Steps

1. **Update H100 model** with CUDA + Tensor Core fabrics
2. **Update KPU-T256 model** with INT8 + BF16 + Matrix fabrics
3. **Update TPU Edge Pro** to use process-node-only energy (revert my previous "fix")
4. **Implement architectural energy models** in `architectural_energy.py`
5. **Update energy analyzer** to be fabric-aware
6. **Run validation tests** to verify energy values are physically consistent

## Files Modified

- ✅ `src/graphs/hardware/resource_model.py`: Added energy constants, ComputeFabric, compute_fabrics field
- ✅ `src/graphs/hardware/models/datacenter/cpu_arm.py`: New ARM CPU model with scalar + NEON

## Files To Modify

- `src/graphs/hardware/models/datacenter/h100_sxm5_80gb.py`
- `src/graphs/hardware/models/accelerators/kpu_t256.py`
- `src/graphs/hardware/models/edge/tpu_edge_pro.py`
- `src/graphs/hardware/architectural_energy.py`
- `src/graphs/analysis/energy.py`
- `src/graphs/hardware/mappers/gpu.py`
- `src/graphs/hardware/mappers/kpu.py`

---

**Date**: 2025-11-13
**Status**: In Progress (40% complete)
**Next Task**: Update H100 model with multi-fabric support
