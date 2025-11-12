# NVIDIA GPU Memory Hierarchy - Official Nomenclature

**Source:** NVIDIA Ampere Architecture Documentation (A100 GPU)

## Complete Memory Hierarchy (Register File → DRAM)

```
┌─────────────────────────────────────────────────────────┐
│ 1. REGISTER FILE (RF)                                   │
│    • 64K × 32-bit registers per SM (256 KB per SM)     │
│    • Separate from cache hierarchy                      │
│    • Holds operands for CUDA/Tensor cores              │
│    • Per-thread allocation (up to 255 registers/thread) │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. SHARED MEMORY / L1 DATA CACHE / TEXTURE CACHE        │
│    (UNIFIED, configurable carveout)                     │
│    • 192 KB per SM (A100, compute 8.0)                  │
│    • 128 KB per SM (Jetson Orin, compute 8.6)          │
│    • Shared Memory: Programmer-managed scratchpad      │
│    • L1 Data Cache: Hardware-managed                   │
│    • Texture Cache: For texture operations             │
│    • Acts as coalescing buffer for memory accesses     │
│    • Carveout configurable at runtime via              │
│      cudaFuncSetAttribute()                             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. L2 CACHE (Unified, shared across all SMs)           │
│    • 40 MB (A100 GPU)                                   │
│    • 4 MB (Jetson Orin AGX)                            │
│    • ~7× larger than V100                               │
│    • 2.3× L2 read bandwidth vs V100                    │
│    • Persistence control available (L2 cache residency)│
│    • Partitioned crossbar structure                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. HBM2(e) or LPDDR5 DRAM                              │
│    • 40 GB @ 1555 GB/s (A100 40GB)                     │
│    • 80 GB @ 2039 GB/s (A100 80GB)                     │
│    • 64 GB @ 204.8 GB/s (Jetson Orin AGX, LPDDR5)     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Clarifications

### 1. Register File vs L1 Cache
**They are SEPARATE levels.**

NVIDIA documentation explicitly states:
> "asynchronous copy instruction that loads data directly from global memory
> into SM shared memory, eliminating the need for intermediate register
> file (RF) usage"

This confirms:
- **Register File (RF)**: 64K registers per SM (256 KB), separate hardware
- **L1 Data Cache**: Part of unified L1/Shared Memory/Texture cache (128-192 KB per SM)

### 2. Shared Memory vs L1 Cache
**They are UNIFIED in Ampere architecture.**

From NVIDIA docs:
> "The NVIDIA Ampere GPU architecture combines the functionality of the L1
> and texture caches into a unified L1/Texture cache"

> "the portion of the L1 cache dedicated to shared memory (known as the
> carveout) can be selected at runtime"

**Available carveout configurations:**
- 0 KB shared / 128 KB L1 (all cache)
- 8 KB shared / 120 KB L1
- 16 KB shared / 112 KB L1
- 32 KB shared / 96 KB L1
- 64 KB shared / 64 KB L1
- 100 KB shared / 28 KB L1
- 128 KB shared / 0 KB L1 (all scratchpad, Jetson Orin max)
- 164 KB shared / 28 KB L1 (A100 max)

### 3. You Were Partially Right!
Your statement:
> "L1 for the register file and than Shared Mem/L2 so that threads can
> read and write their results to each other efficiently"

**Clarification:**
- **L1 ≠ Register File** (they're separate levels)
- **Shared Memory is part of L1**, not L2!
- The unified structure is: **Shared Memory / L1 Data Cache / Texture Cache**
- **L2** is a separate, larger cache (40 MB) shared across ALL SMs

---

## Correct Terminology for Our Modeling

Based on official NVIDIA nomenclature:

```
Level 1: Register File (RF)
         • 64K × 32-bit registers per SM
         • Per-thread private storage
         • NOT called "L1"

Level 2: Shared Memory / L1 Data Cache (UNIFIED)
         • 128-192 KB per SM
         • Configurable carveout
         • Programmer-managed (Shared Memory)
         • Hardware-managed (L1 Data Cache)

Level 3: L2 Cache
         • 4-40 MB shared across all SMs
         • Hardware-managed
         • Persistence control available

Level 4: HBM2(e) or LPDDR5 DRAM
         • 40-80 GB
         • Off-chip
```

---

## Common Misconceptions (Corrected)

### ❌ WRONG: "L1 is the register file"
**✓ CORRECT:** Register file is a separate level before L1

### ❌ WRONG: "Shared Memory / L2"
**✓ CORRECT:** Shared Memory / L1 (they're unified)

### ❌ WRONG: "L1 → Shared Memory → L2 → DRAM"
**✓ CORRECT:** Register File → (Shared Memory / L1) → L2 → DRAM

---

## Energy Implications for Our Model

Our current model has:
```python
shared_memory_energy_per_byte: float = 0.2e-12     # ✓ Correct
l1_cache_energy_per_byte: float = 0.3e-12          # ✓ Correct (same hardware!)
l2_cache_energy_per_byte: float = 0.8e-12          # ✓ Correct
dram_energy_per_byte: float = 10.0e-12             # ✓ Correct
```

**Issue:** We're modeling Shared Memory and L1 as separate energy components, but they're the **same hardware** with configurable allocation!

**Fix:** Either:
1. Model them as the same energy (0.2-0.3 pJ/byte)
2. Or model them separately but acknowledge they're mutually exclusive (carveout)

---

## Nomenclature for Reporting

For clarity in our comparison tool output:

```
GPU Memory Hierarchy:
  1. Register File (64K regs/SM, 256 KB)
  2. Shared Memory / L1 Cache (unified 128 KB, configurable)
  3. L2 Cache (4 MB, shared across SMs)
  4. DRAM (64 GB LPDDR5)
```

**NOT:**
- ❌ "L1 Cache (registers)"
- ❌ "Shared Memory / L2"
- ❌ "L1 → L2 → Shared Memory"
