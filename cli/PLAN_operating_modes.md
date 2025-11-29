# Operating Mode Energy Model Plan

## Problem Statement

The current energy model conflates different operating scenarios, making it impossible
to fairly compare architectures. Each architecture has optimal operating modes where
the compute machinery works from a specific memory resource.

We need to model **four distinct operating modes** based on where the working set resides:

## Operating Modes

### Mode 1: L1-Resident (On-Chip Fast Memory)

**Scenario**: Kernel working set fits entirely in L1 cache / shared memory / scratchpad

| Architecture | Memory Resource | Typical Size | Energy/Byte |
|--------------|-----------------|--------------|-------------|
| CPU          | L1 D-cache      | 32-48 KB     | ~1 pJ/B     |
| GPU          | Shared Memory   | 128-228 KB   | ~0.25 pJ/B  |
| DSP          | Scratchpad SRAM | 256-512 KB   | ~1 pJ/B     |

**Hit Ratios**: 100% L1 hit (by definition - data fits)

**This is the "best case" for all architectures** - no cache hierarchy traversal.

### Mode 2: L2-Resident

**Scenario**: Working set fits in L2, requires L1 as cache

| Architecture | Memory Resource | Typical Size | Energy/Byte |
|--------------|-----------------|--------------|-------------|
| CPU          | L2 cache        | 256KB - 2MB  | ~2.5 pJ/B   |
| GPU          | L2 cache        | 4-60 MB      | ~0.8 pJ/B   |
| DSP          | N/A (no L2)     | -            | -           |

**Hit Ratios**:
- L1 hit rate: 85-95% (temporal/spatial locality)
- L2 hit rate: 100% (by definition - data fits)

**DSP Note**: DSPs typically don't have L2 - they use DMA to prefetch into scratchpad.
For DSP, L2-resident mode means "data prefetched from DRAM into scratchpad with double-buffering".

### Mode 3: L3-Resident (LLC)

**Scenario**: Working set fits in L3/LLC, requires L1+L2 as cache

| Architecture | Memory Resource | Typical Size | Energy/Byte |
|--------------|-----------------|--------------|-------------|
| CPU          | L3/LLC          | 8-64 MB      | ~5 pJ/B     |
| GPU          | N/A (no L3)     | -            | -           |
| DSP          | N/A (no L3)     | -            | -           |

**Hit Ratios**:
- L1 hit rate: 80-90%
- L2 hit rate: 85-95%
- L3 hit rate: 100% (by definition - data fits)

**GPU/DSP Note**: These architectures don't have L3. For them, this mode doesn't exist -
they go directly to DRAM/HBM.

### Mode 4: DRAM-Resident (Off-Chip Memory)

**Scenario**: Working set exceeds on-chip memory, must stream from DRAM/HBM

| Architecture | Memory Resource | Bandwidth    | Energy/Byte |
|--------------|-----------------|--------------|-------------|
| CPU          | DDR4/DDR5       | 50-100 GB/s  | ~20 pJ/B    |
| GPU          | HBM2/HBM3       | 2-5 TB/s     | ~10 pJ/B    |
| DSP          | LPDDR4/5        | 25-50 GB/s   | ~15 pJ/B    |

**Hit Ratios**:
- L1 hit rate: 70-85%
- L2 hit rate: 80-90%
- L3 hit rate: 85-95% (CPU only)
- DRAM: remainder

**This models the "cold start" or streaming scenario**.

---

## Energy Calculation per Mode

For each mode, the energy calculation follows:

```
E_total = E_instruction + E_compute + E_simt_overhead + E_memory

where:
  E_instruction = E_fetch + E_decode + E_operand_fetch + E_writeback
  E_compute = ops * E_per_op
  E_simt_overhead = (GPU only) thread_mgmt + coherence + sync
  E_memory = sum over hierarchy levels of (accesses * E_per_access)
```

### Memory Energy Calculation by Mode

**Mode 1 (L1-Resident)**:
```
E_memory = bytes * E_L1_per_byte
```

**Mode 2 (L2-Resident)**:
```
E_memory = bytes * L1_hit_rate * E_L1_per_byte
         + bytes * (1 - L1_hit_rate) * E_L2_per_byte
```

**Mode 3 (L3-Resident)**:
```
E_memory = bytes * L1_hit_rate * E_L1_per_byte
         + bytes * (1 - L1_hit_rate) * L2_hit_rate * E_L2_per_byte
         + bytes * (1 - L1_hit_rate) * (1 - L2_hit_rate) * E_L3_per_byte
```

**Mode 4 (DRAM-Resident)**:
```
E_memory = bytes * L1_hit_rate * E_L1_per_byte
         + bytes * (1 - L1_hit_rate) * L2_hit_rate * E_L2_per_byte
         + bytes * (1 - L1_hit_rate) * (1 - L2_hit_rate) * L3_hit_rate * E_L3_per_byte
         + bytes * (1 - L1_hit_rate) * (1 - L2_hit_rate) * (1 - L3_hit_rate) * E_DRAM_per_byte
```

---

## Default Hit Ratios by Mode

| Mode | L1 Hit | L2 Hit | L3 Hit | Notes |
|------|--------|--------|--------|-------|
| L1-Resident | 100% | n/a | n/a | All data in L1 |
| L2-Resident | 90% | 100% | n/a | Good locality |
| L3-Resident | 85% | 90% | 100% | Moderate locality |
| DRAM-Resident | 80% | 85% | 90% | Streaming/cold |

These can be adjusted based on workload characteristics (e.g., matmul has better locality than sparse ops).

---

## SIMT Overhead Scaling by Mode

The GPU SIMT overhead (thread management, coherence, sync) also depends on mode:

**L1-Resident (Shared Memory)**:
- Coherence is minimal (shared memory is per-SM, not coherent)
- Thread management still required
- This is where GPU excels

**L2-Resident and beyond**:
- Full coherence machinery engaged
- L2 is coherent across SMs
- Coherence cost scales with number of warps * cache lines accessed

---

## Architecture Applicability Matrix

| Mode | CPU | GPU | DSP |
|------|-----|-----|-----|
| L1-Resident | Yes (L1 D$) | Yes (Shared Mem) | Yes (Scratchpad) |
| L2-Resident | Yes | Yes | Emulated (DMA prefetch) |
| L3-Resident | Yes | No | No |
| DRAM-Resident | Yes | Yes | Yes |

---

## CLI Interface Design

```bash
# Compare all architectures in L1-resident mode
./cli/validate_architectural_energy.py --mode l1

# Compare in L2-resident mode (with default hit ratios)
./cli/validate_architectural_energy.py --mode l2

# Compare in DRAM-resident mode
./cli/validate_architectural_energy.py --mode dram

# Compare all modes side-by-side
./cli/validate_architectural_energy.py --mode all

# Custom hit ratios
./cli/validate_architectural_energy.py --mode l2 --l1-hit-rate 0.92

# Sweep across modes
./cli/validate_architectural_energy.py --mode-sweep
```

---

## Expected Insights

1. **L1-Resident Mode**: GPU should excel due to large shared memory and efficient register file
2. **L2-Resident Mode**: GPU coherence overhead becomes visible, CPU competitive
3. **L3-Resident Mode**: CPU advantage (only arch with L3), GPU/DSP must go to DRAM
4. **DRAM-Resident Mode**: GPU wins on bandwidth (HBM), but coherence overhead is significant

This framework will reveal:
- Where each architecture is optimal
- The crossover points between architectures
- The true cost of the GPU coherence machinery at different scales

---

## Implementation Steps

1. Add `OperatingMode` enum (L1, L2, L3, DRAM)
2. Add hit ratio parameters to energy model functions
3. Modify `build_*_cycle_energy()` to accept mode parameter
4. Update memory access calculations to use mode-specific hit ratios
5. Add `--mode` CLI argument
6. Create mode comparison table
7. Add mode sweep functionality
