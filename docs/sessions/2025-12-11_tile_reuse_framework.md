# Session Summary: Tile Reuse Framework - Memory-Constrained Blocking

**Date**: 2025-12-11
**Phase**: Research - Tile Reuse Characterization
**Status**: In Progress (needs refinement)

---

## Goals for This Session

1. Continue tile reuse framework implementation from previous session
2. Add memory constraints that trigger blocking decisions
3. Connect memory budgets to tile size optimization

---

## What We Accomplished

### 1. Refactored Tile Representation to 2D (Previous Session)

**Key Insight from User**: Tiles are 2D submatrices, NOT 3D objects.
- A_tile: shape (Tm, Tk) - input activation
- B_tile: shape (Tk, Tn) - weight
- C_tile: shape (Tm, Tn) - output/accumulator

The (Tm, Tk, Tn) notation describes loop bounds, not a single tile.

### 2. Created TileSizeOptimizer (This Session)

**File**: `src/graphs/research/tiling/tile_optimizer.py`

**Purpose**: Determines when memory constraints require blocking and computes optimal tile sizes.

**Key Classes**:
- `TileConstraint`: Captures constraint analysis for a memory level
- `HierarchicalBlocking`: Multi-level blocking scheme (L1, L2, L3 tiles)
- `TileSizeOptimizer`: Main optimizer that determines when blocking is needed

**Core Logic**:
```python
working_set = A_tile + B_tile + C_tile
            = (Tm * Tk * input_bytes) + (Tk * Tn * weight_bytes) + (Tm * Tn * accum_bytes)

if working_set > L2_capacity:
    # Blocking is REQUIRED - must use smaller tiles
```

### 3. Added Integrated Analysis Function

**Function**: `analyze_with_memory_constraints()`

Combines:
1. Memory constraint checking (why blocking is needed)
2. Tile size optimization (what sizes fit)
3. Reuse statistics (how much reuse per tile type)

---

## Key Issues Identified

### Problem: Output Not Useful for Analysis

The current output shows:
- Tile sizes at each level
- Reuse factors (e.g., A reuse 22x, B reuse 22x)
- Iteration counts

**What's Missing**:
1. Per-level reuse tracking (reuse at L1 vs L2 vs L3)
2. Data movement between levels (bytes transferred L3->L2->L1)
3. How tiles actually flow through the hierarchy
4. Energy/latency impact of blocking decisions

---

## Files Created/Modified

### Source Code
- `src/graphs/research/tiling/tile_optimizer.py` (~400 lines) - NEW: Memory-constrained tile optimization
- `src/graphs/research/tiling/__init__.py` - Updated exports

### Key Functions
```python
# Main entry point - shows constraints + reuse
analyze_with_memory_constraints(M=4096, K=2048, N=4096, loop_order=LoopOrder.MNK)

# Just blocking analysis
print_blocking_analysis(M=4096, K=2048, N=4096)

# Just reuse analysis (manual tile sizes)
analyze_tile_reuse(M=4096, K=2048, N=4096, Tm=192, Tk=128, Tn=192)
```

---

## Example Output

```
======================================================================
MEMORY-CONSTRAINED TILE REUSE ANALYSIS
======================================================================

Problem: C[4096, 4096] = A[4096, 2048] @ B[2048, 4096]
Loop order: MNK

----------------------------------------------------------------------
MEMORY CONSTRAINTS (why blocking is needed)
----------------------------------------------------------------------
Full problem working set: 100,663,296 bytes (96.00 MB)
  L1: 262,144 bytes -> BLOCKING REQUIRED
  L2: 2,097,152 bytes -> BLOCKING REQUIRED
  L3: 33,554,432 bytes -> BLOCKING REQUIRED

----------------------------------------------------------------------
SELECTED TILE SIZES (from memory constraints)
----------------------------------------------------------------------
L1 tiles (compute): Tm=192, Tk=128, Tn=192
  A_tile: (192, 128) = 49,152 bytes
  B_tile: (128, 192) = 49,152 bytes
  C_tile: (192, 192) = 147,456 bytes
  Working set: 245,760 bytes

----------------------------------------------------------------------
TILE REUSE STATISTICS (for L1 tiles)
----------------------------------------------------------------------
Tile     Shape           Count      Reuse      Bytes
A        (192, 128)      352        22.0       17,301,504
B        (128, 192)      352        22.0       17,301,504
C        (192, 192)      484        16.0       71,368,704

----------------------------------------------------------------------
SUMMARY
----------------------------------------------------------------------
  Total FLOPs:            68,719,476,736
  Arithmetic intensity:   648.47 FLOPs/byte
  A reuse factor:         22.0x
  B reuse factor:         22.0x
  C accumulations:        16.0x
======================================================================
```

---

## Next Steps

### Immediate (Next Session)
1. [ ] Discuss what output format would actually be useful
2. [ ] Add per-level reuse tracking (L1, L2, L3 separately)
3. [ ] Show data movement between levels
4. [ ] Connect to energy model (pJ per byte at each level)

### Questions to Resolve
1. What visualization of tile flow would be useful?
2. Should we show the 6-loop nest structure explicitly?
3. How to present reuse at different hierarchy levels?

---

## Commands

```bash
# Memory-constrained analysis with reuse
PYTHONPATH=src:$PYTHONPATH python3 -c "
from graphs.research.tiling import analyze_with_memory_constraints, LoopOrder
analyze_with_memory_constraints(M=4096, K=2048, N=4096, loop_order=LoopOrder.MNK)
"

# With specific memory budget (e.g., H100)
PYTHONPATH=src:$PYTHONPATH python3 -c "
from graphs.research.tiling import analyze_with_memory_constraints, LoopOrder, H100_BUDGET
analyze_with_memory_constraints(M=4096, K=2048, N=4096, budget=H100_BUDGET, loop_order=LoopOrder.MNK)
"
```

---

## Related Files

- `src/graphs/research/tiling/block_algebra.py` - 2D tile types (ATile, BTile, CTile)
- `src/graphs/research/tiling/reuse_analyzer.py` - Per-tile-type reuse analysis
- `src/graphs/research/tiling/memory_model.py` - Memory hierarchy and budgets
- `src/graphs/research/tiling/double_buffer.py` - Double-buffering state machine
- `src/graphs/research/tiling/distributed_l3.py` - Distributed L3 topology
- `src/graphs/research/tiling/tile_rotation.py` - Cannon/SUMMA algorithms

---

## Session Notes

### User Feedback
The current implementation doesn't present reuse in a useful way. Need to:
1. Clarify what output format would be productive
2. Focus on data movement and energy, not just counts
3. Show hierarchical structure more clearly

### Technical Debt
1. Reuse analyzer not hierarchy-aware (only computes for one level)
2. No integration with energy model yet
3. Missing visualization of tile flow through levels
