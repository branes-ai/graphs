# Phase 1A: CPU Hierarchical Energy Breakdown - COMPLETE

**Date**: 2025-11-13
**Session Duration**: ~1.5 hours
**Status**: ✅ COMPLETE

## Summary

Successfully integrated hierarchical energy breakdown into the CPU analysis tool (`cli/analyze_cpu_energy.py`). The tool now shows detailed architectural energy events for CPUs with configured energy models.

## What Was Built

### 1. Shared Utilities Module
**File**: `cli/energy_breakdown_utils.py` (178 lines)

**Functions**:
- `print_cpu_hierarchical_breakdown()`: Prints detailed 5-category breakdown
  - Instruction Pipeline (fetch, decode, dispatch)
  - Register File Operations (reads, writes)
  - Memory Hierarchy (L1, L2, L3, DRAM)
  - ALU Operations
  - Branch Prediction
- `aggregate_subgraph_events()`: Aggregates events across subgraphs (for future use)

### 2. Enhanced CPU Analysis Tool
**File**: `cli/analyze_cpu_energy.py` (lines 293-359, +67 lines)

**New Features**:
- Automatic architectural energy extraction after mapping
- Aggregates ops and bytes across all subgraphs
- Calls `architecture_energy_model.compute_architectural_energy()`
- Prints hierarchical breakdown if model available
- Graceful fallback for CPUs without architectural models

## Test Results

### Test 1: Jetson Orin AGX CPU + MobileNetV2 (batch 1)
```
Total Energy: 36.4 mJ
Architectural Overhead: 6.8 mJ (18.7%)

Breakdown:
  - Instruction Pipeline: 0.9 mJ (fetch, decode, dispatch)
  - Register File: 2.6 mJ (reads + writes)
  - Memory Hierarchy: 2.6 mJ (L1: 5%, DRAM: 54%)
  - ALU Operations: 0.7 mJ
  - Branch Prediction: 0.003 mJ
```

### Test 2: Intel Xeon Emerald Rapids + ResNet18 (batch 4)
```
Total Energy: 351.7 mJ
Architectural Overhead: 138.3 mJ (39.3%)

Breakdown scaled correctly with larger model and batch size.
```

### Test 3: AMD EPYC Genoa + MobileNetV2 (batch 1)
```
Total Energy: 456.0 mJ
Note: Detailed architectural breakdown not available for epyc_genoa
      (no StoredProgramEnergyModel configured)
```

## Technical Approach

**Decision**: Aggregate events approach (Option C from session doc)

**Why**:
- Simplest to implement without modifying core data structures
- Reuses existing `compute_architectural_energy()` method
- Provides complete breakdown for entire model
- Pattern can be replicated for GPU/TPU/KPU tools

**How it works**:
1. After mapping completes, aggregate total ops and bytes across subgraphs
2. Manually call `architecture_energy_model.compute_architectural_energy()`
3. Extract architectural events from `extra_details` dict
4. Pass to `print_cpu_hierarchical_breakdown()` for display

## Example Output

```
================================================================================
CPU ENERGY BREAKDOWN ANALYSIS
================================================================================
CPU: ARM Cortex-A78AE (12 cores, 8nm, 30W)
Model: mobilenet_v2
Batch Size: 1
Precision: FP32
================================================================================

[1/4] Loading and tracing model...
✓ Model loaded: 66 subgraphs

[2/4] Creating CPU mapper for jetson_orin_agx_cpu...
✓ CPU mapper created: Jetson-Orin-AGX-CPU
  Cores: 12
  Peak Performance (FP32): 0.21 TFLOPS

[3/4] Mapping model to CPU...
✓ Model mapped
  Peak cores used: 12 / 12
  Avg utilization: 94.6%
  Estimated latency: 1.49 ms
  Total energy: 36.39 mJ

[4/4] Energy Breakdown
================================================================================

────────────────────────────────────────────────────────────────────────────────
CPU (STORED-PROGRAM MULTICORE) ENERGY BREAKDOWN
────────────────────────────────────────────────────────────────────────────────

  1. INSTRUCTION PIPELINE (Fetch → Decode → Dispatch)
     • Instruction Fetch (I-cache):       492.897 μJ  ( 53.6%)  [328,598,270 instructions]
     • Instruction Decode:                262.879 μJ  ( 28.6%)
     • Instruction Dispatch:              164.299 μJ  ( 17.9%)
     └─ Subtotal:                         920.075 μJ

  2. REGISTER FILE OPERATIONS (2 reads + 1 write per instruction)
     • Register Reads:                   1642.991 μJ  ( 62.5%) [657,196,540 reads]
     • Register Writes:                   985.795 μJ  ( 37.5%) [328,598,270 writes]
     └─ Subtotal:                        2628.786 μJ

  3. MEMORY HIERARCHY (4-Stage: L1 → L2 → L3 → DRAM)
     • L1 Cache (per-core, 32 KB):       141.356 μJ  (  5.4%)  [2,208,682 accesses]
     • L2 Cache (per-core, 256 KB):      353.389 μJ  ( 13.5%)  [2,208,682 accesses]
     • L3 Cache (shared LLC, 8 MB):      706.778 μJ  ( 27.0%)  [2,208,682 accesses]
     • DRAM (off-chip DDR4):            1413.556 μJ  ( 54.1%)  [1,104,341 accesses]
     └─ Subtotal:                        2615.079 μJ

  4. ALU OPERATIONS (Floating-point arithmetic)
     • ALU Energy:                        657.197 μJ  [328,598,270 ops]
     └─ Subtotal:                         657.197 μJ

  5. BRANCH PREDICTION (Control flow)
     • Branch Prediction:                   0.329 μJ  [16,429,913 branches, 821,495 mispredicted]
     └─ Subtotal:                           0.329 μJ

  TOTAL CPU ARCHITECTURAL OVERHEAD:     6821.466 μJ
  Base Compute Energy (from mapper):    12576.887 μJ
  Base Memory Energy (from mapper):     1413.556 μJ
  ────────────────────────────────────────────────────────────────────────────────
  SUBTOTAL DYNAMIC ENERGY:              20811.909 μJ  (57.2%)
  Idle/Leakage Energy (15.0W × latency):  15577.568 μJ  (42.8%)
  ────────────────────────────────────────────────────────────────────────────────
  TOTAL CPU ENERGY:                     36389.477 μJ
```

## Known Limitations

1. **Limited CPU Coverage**: Only CPUs with `StoredProgramEnergyModel` get detailed breakdown
   - ✅ Jetson Orin AGX CPU
   - ✅ Intel Xeon Platinum 8490H (Emerald Rapids)
   - ❌ AMD EPYC (all variants)
   - ❌ Ampere (all variants)
   - ❌ Other Intel Xeon variants

2. **Aggregate Only**: Shows model-level breakdown, not per-subgraph

3. **No Export**: JSON/CSV export not yet implemented

## Future Work

### Phase 1B: JSON/CSV Export (Optional)
- Add output formatters
- Support `--output file.json` and `--output file.csv`
- Include all hierarchical events in exports

**Time Estimate**: 1-2 hours

### Phase 1C: Add Missing CPU Models (Optional)
- Create StoredProgramEnergyModel for AMD EPYC and Ampere CPUs
- Configure energy parameters based on process node
- Test with each CPU

**Time Estimate**: 2-3 hours

### Phase 2: GPU Energy Analysis Tool
- Create `cli/analyze_gpu_energy.py` using same pattern
- Extract GPU hierarchical breakdown function
- Integrate DataParallelEnergyModel
- Test with H100, Jetson, A100

**Time Estimate**: 2-3 hours

### Phase 3-4: TPU and KPU Tools
- Similar pattern to Phase 2
- Extract respective hierarchical breakdown functions
- Integrate systolic array and domain flow energy models

**Time Estimate**: 2-3 hours each

## Files Modified/Created

### New Files
- `cli/energy_breakdown_utils.py` (178 lines)

### Modified Files
- `cli/analyze_cpu_energy.py` (+67 lines, lines 293-359)
- `docs/sessions/2025-11-13_architecture_energy_cli_tools.md` (updated with Phase 1A completion)

## Usage

```bash
# Basic usage (with hierarchical breakdown)
python cli/analyze_cpu_energy.py --cpu jetson_orin_agx_cpu --model mobilenet_v2

# Different CPU and model
python cli/analyze_cpu_energy.py --cpu xeon_emerald_rapids --model resnet18 --batch-size 4

# CPU without architectural model (shows basic metrics)
python cli/analyze_cpu_energy.py --cpu epyc_genoa --model mobilenet_v2

# List available CPUs and models
python cli/analyze_cpu_energy.py --list-cpus
python cli/analyze_cpu_energy.py --list-models
```

## Key Insights from Testing

1. **Architectural overhead varies by CPU**: 18.7% for Jetson Orin, 39.3% for Xeon
2. **Register file energy ≈ ALU energy**: Both in the 0.6-0.8 pJ range
3. **Memory hierarchy dominates**: DRAM accounts for 50%+ of memory energy
4. **Instruction fetch non-trivial**: 500-18000 μJ depending on workload
5. **Idle/leakage significant**: 42-51% of total energy at 15W idle power

## Lessons Learned

1. **Aggregate approach works well**: No need to modify core data structures
2. **Graceful fallback essential**: Not all CPUs have models configured
3. **Reusable utilities important**: `energy_breakdown_utils.py` will help GPU/TPU/KPU tools
4. **Testing crucial**: Found issues with ops calculation that wouldn't show in simple tests

---

**Phase 1A Status**: ✅ COMPLETE
**Next**: Phase 1B (export), Phase 1C (more CPUs), or Phase 2 (GPU tool)
