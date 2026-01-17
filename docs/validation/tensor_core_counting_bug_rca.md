# Tensor Core Counting Bug - Root Cause Analysis

## Summary
The GPU energy model is counting **individual MACs as "Tensor Core ops"** when it should count **actual Tensor Core operations** (each performing 64 MACs via 4×4×4 matrix multiplication).

**Error magnitude: 128× overcounting of Tensor Core operations!**

---

## The Problem

### What Was Reported:
```
1. COMPUTE UNITS (Tensor Cores vs CUDA Cores)
   • Tensor Core Operations:           0.031 μJ  ( 15.0%)  [104,857 ops]  ← WRONG!
   • CUDA Core Operations:             0.021 μJ  ( 10.0%)  [26,215 ops]   ← WRONG!
```

### What This Implies:
- 104,857 Tensor Core operations × 64 MACs/op = **6,710,848 MACs**
- But the workload only has **65,536 MACs** total!
- **Overcounting by 102×** the actual workload!

---

## Root Cause

### Tensor Core Architecture (NVIDIA Ampere):
- **Each Tensor Core operation performs a 4×4×4 FP16 matrix multiplication**
- This is **64 FMAs (MACs)** per Tensor Core operation
- Energy per Tensor Core operation = 64 MACs × energy_per_MAC

### Current Code (WRONG):
```python
# Line 562 in architectural_energy.py
tensor_core_ops = int(ops * self.tensor_core_utilization)
# ops = 131,072 FLOPs
# tensor_core_ops = 131,072 * 0.8 = 104,857

# This treats individual FLOPs as "Tensor Core operations"
```

**Problem:** The code assumes `ops` represents discrete operations, but:
- `ops` = 131,072 **FLOPs** (or 65,536 **MACs**)
- Code splits: 80% → Tensor Cores, 20% → CUDA Cores
- **But individual FLOPs/MACs are NOT Tensor Core operations!**

---

## Correct Calculation

### Workload:
- 256×256 matrix multiplication
- **65,536 MACs** (131,072 FLOPs)

### Correct Distribution (80% Tensor Core utilization):

**Tensor Cores (80%):**
- MACs handled by Tensor Cores: 65,536 × 0.8 = **52,428 MACs**
- Tensor Core operations: 52,428 / 64 = **819 Tensor Core ops**
- Each Tensor Core op does 4×4×4 = 64 MACs

**CUDA Cores (20%):**
- MACs handled by CUDA Cores: 65,536 × 0.2 = **13,108 MACs**
- CUDA Core operations: **13,108 ops** (1 MAC per op)

**Total:** 52,428 + 13,108 = 65,536 MACs ✓

---

## Comparison

| Metric | Current (WRONG) | Correct | Error |
|--------|----------------|---------|-------|
| Tensor Core ops | 104,857 | 819 | **128× overcounted** |
| CUDA Core ops | 26,215 | 13,108 | 2× overcounted |
| Total MACs implied | 131,072 | 65,536 | 2× overcounted |

---

## Why Energy Calculation is Still "Correct"

Despite the counting error, the **energy calculation happens to be correct** because:

```python
# Current code (line 565-566)
tensor_core_energy = tensor_core_ops * self.tensor_core_mac_energy
                   = 104,857 * 0.3e-12 pJ
                   = 31.46 nJ

# What it should be:
tensor_core_energy = 819 ops * (64 MACs/op * 0.3e-12 pJ/MAC)
                   = 819 * 19.2e-12 pJ
                   = 15.72 nJ  ← DIFFERENT!
```

**Wait, these are different!** Let me recalculate:

Actually, the current calculation:
- 104,857 "ops" × 0.3 pJ = 31.46 nJ

Should be:
- 52,428 MACs × 0.3 pJ/MAC = 15.73 nJ

**So the energy is ALSO wrong! It's 2× too high!**

---

## Impact Analysis

### 1. Reported Operation Counts (WRONG)
- **Tensor Core ops**: 104,857 (should be 819) - **128× overcounted**
- **CUDA Core ops**: 26,215 (should be 13,108) - **2× overcounted**

### 2. Energy Calculation (ALSO WRONG)
- **Tensor Core energy**: 31.46 nJ (should be 15.73 nJ) - **2× too high**
- **CUDA Core energy**: 21.0 nJ (should be 10.5 nJ) - **2× too high**

### 3. Why the 2× Factor?
The `ops` parameter is **FLOPs** (131,072), but should be **MACs** (65,536).
- 1 MAC = 2 FLOPs (multiply + add)
- So everything is being doubled!

---

## The Real Problem: Terminology Confusion

The `ops` parameter in `compute_architectural_energy()` is ambiguous:

```python
def compute_architectural_energy(
    self,
    ops: int,  # ← What does "ops" mean?
    ...
)
```

**What is `ops`?**
1. FLOPs? (floating-point operations)
2. MACs? (multiply-accumulate operations)
3. Instructions?
4. Tensor Core operations?

**Currently:** It's **FLOPs** (131,072), which is 2× the MAC count (65,536)

**Problem:** The code uses `ops` directly as if it's MACs:
```python
tensor_core_ops = int(ops * 0.8)  # Assumes ops = MACs
tensor_core_energy = tensor_core_ops * self.tensor_core_mac_energy  # But mac_energy is per MAC!
```

---

## Fix Strategy

### Option 1: Change `ops` to mean MACs (Recommended)
```python
def compute_architectural_energy(
    self,
    macs: int,  # ← Rename for clarity
    bytes_transferred: int,
    ...
):
    # Distribute MACs across compute units
    tensor_core_macs = int(macs * self.tensor_core_utilization)  # 52,428 MACs
    cuda_core_macs = macs - tensor_core_macs  # 13,108 MACs

    # Calculate Tensor Core operations (64 MACs per op)
    tensor_core_ops = tensor_core_macs // 64  # 819 ops

    # Energy calculation
    tensor_core_energy = tensor_core_macs * self.tensor_core_mac_energy
    cuda_core_energy = cuda_core_macs * self.cuda_core_mac_energy
```

**Display:**
```
• Tensor Core Operations:   15.73 nJ  [819 ops, 52,428 MACs]
• CUDA Core Operations:     10.49 nJ  [13,108 ops, 13,108 MACs]
```

### Option 2: Keep `ops` as FLOPs, convert to MACs internally
```python
def compute_architectural_energy(
    self,
    ops: int,  # FLOPs
    bytes_transferred: int,
    ...
):
    # Convert FLOPs to MACs (1 MAC = 2 FLOPs)
    macs = ops // 2

    # Rest of calculation as in Option 1
```

---

## Recommendation

**Use Option 1:** Rename `ops` → `macs` throughout the codebase for clarity.

**Why?**
1. MACs are the fundamental unit for matrix multiplication hardware
2. Tensor Cores operate on MACs, not FLOPs
3. Energy is specified as "per MAC" in literature
4. Eliminates the 2× confusion factor

---

## Files to Fix

1. `src/graphs/hardware/architectural_energy.py`
   - Rename parameter `ops` → `macs`
   - Fix Tensor Core operation counting
   - Update all 3 energy models (DataParallel, StoredProgram, SystolicArray)

2. `src/graphs/hardware/resource_model.py`
   - Update call site to pass MACs instead of FLOPs

3. `src/graphs/hardware/mappers/gpu.py`
   - Ensure mapper passes MACs, not FLOPs

4. `cli/compare_architectures_energy.py`
   - Update display to show both ops and MACs

---

## Validation

After fix, for 256×256 MLP (65,536 MACs):

```
✓ Tensor Core Operations:   [819 ops, 52,428 MACs]
✓ CUDA Core Operations:      [13,108 ops, 13,108 MACs]
✓ Total MACs:                65,536
✓ Tensor Core utilization:   80%
```

Energy should be:
```
✓ Tensor Core energy:  52,428 × 0.3 pJ = 15.73 nJ
✓ CUDA Core energy:    13,108 × 0.8 pJ = 10.49 nJ
✓ Total compute:       26.22 nJ (vs current 52.44 nJ - 2× reduction!)
```
