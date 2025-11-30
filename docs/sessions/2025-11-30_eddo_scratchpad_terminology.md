# Session: EDDO Scratchpad Terminology & Consistent Energy Units

**Date**: 2025-11-30
**Focus**: Correcting KPU memory hierarchy terminology and fixing energy unit display

## Summary

This session addressed two main issues:
1. The KPU energy model was incorrectly using cache terminology (L1/L2/L3) when Domain Flow processors use software-managed scratchpad hierarchies (EDDO)
2. The architecture comparison CLI displayed inconsistent energy units (header said "pJ" but values showed "nJ", "uJ")

## Problem Statement

### Issue 1: Incorrect Cache Terminology

The user correctly identified that Domain Flow processors do NOT use hardware-managed cache hierarchies. Instead, they use **EDDO (Explicit Data Distribution and Orchestration)** - software-managed scratchpad hierarchies where:

- Data placement is determined at compile time
- Data is proactively staged before needed (no misses)
- Scratchpads are directly addressed (no tag lookups)
- No coherence protocol (explicit data distribution)
- Deterministic timing (no variable miss latencies)

The previous code was using cache terminology which misrepresented the architecture.

### Issue 2: Inconsistent Energy Units

The comparison table showed:
```
Architecture                        Total (pJ)      Per Op          Relative
CPU (Intel Xeon / AMD EPYC)            187.90 nJ      18.8 pJ       1.00x
GPU (NVIDIA H100 / Jetson)               1.31 uJ     131.2 pJ       6.98x
```

The header said "Total (pJ)" but values showed "nJ" and "uJ" - confusing and bad form.

## Changes Made

### 1. DomainFlowEnergyModel (`architectural_energy.py`)

Complete refactor of the energy model:

- Added comprehensive docstring explaining EDDO memory management
- New scratchpad hierarchy fields:
  - `tile_scratchpad_energy_per_byte` (per-tile local SRAM)
  - `global_scratchpad_energy_per_byte` (shared SRAM)
  - `streaming_buffer_energy_per_byte` (DMA staging)
  - `dram_dma_energy_per_byte` (off-chip via DMA)
- Scratchpad energy modeled at 40-60% of equivalent cache energy
- Updated `compute_architectural_energy()` to use EDDO terminology
- Explanation text now clearly describes EDDO advantages

### 2. CyclePhase Enum (`base.py`)

Added new EDDO-specific phases:
```python
EDDO_TILE_SCRATCHPAD = "eddo_tile_scratchpad"
EDDO_GLOBAL_SCRATCHPAD = "eddo_global_scratchpad"
EDDO_STREAMING_BUFFER = "eddo_streaming_buffer"
EDDO_DMA_SETUP = "eddo_dma_setup"
```

### 3. KPU Cycle Energy Model (`kpu.py`)

Refactored memory access section:
- Uses `CyclePhase.EDDO_TILE_SCRATCHPAD` instead of `MEM_SRAM`
- Uses `CyclePhase.EDDO_GLOBAL_SCRATCHPAD` instead of `MEM_L2`
- Uses `CyclePhase.EDDO_STREAMING_BUFFER` instead of `MEM_L3`
- Added `CyclePhase.EDDO_DMA_SETUP` for DMA descriptor setup
- Added extensive comments explaining EDDO vs cache

### 4. Energy Formatting (`comparison.py`)

Added consistent-scale formatting functions:
```python
ENERGY_SCALES = [
    ('fJ', 1e-3),      # femtojoules
    ('pJ', 1.0),       # picojoules (base)
    ('nJ', 1e3),       # nanojoules
    ('uJ', 1e6),       # microjoules
    ('mJ', 1e9),       # millijoules
    ('J',  1e12),      # joules
]

def determine_common_scale(values_pj: List[float]) -> tuple:
    """Find best unit where min >= 0.1 and max < 100000"""

def format_energy_with_scale(energy_pj: float, divisor: float) -> str:
    """Format energy value using specific scale (no unit suffix)"""
```

### 5. Architecture Comparison CLI (`compare_architecture_energy.py`)

- Updated `format_architecture_comparison_table()` to use consistent units
- Updated TOTAL row in detailed breakdown to use consistent units
- Added "KPU EDDO SCRATCHPADS" section to phase breakdown
- Fixed bugs:
  - `NameError: name 'category' is not defined` -> changed to `comp_set`
  - `AttributeError: 'ArchitectureComparisonSet' object has no attribute 'category'` -> used `getattr()` for compatibility

## Results

### Before (Inconsistent)
```
Architecture                        Total (pJ)      Per Op          Relative
CPU (Intel Xeon / AMD EPYC)            187.90 nJ      18.8 pJ       1.00x
GPU (NVIDIA H100 / Jetson)               1.31 uJ     131.2 pJ       6.98x
```

### After (Consistent)
```
Architecture                        Total (nJ)      Per Op (pJ)     Relative
CPU (Intel Xeon / AMD EPYC)                  398         39.8       1.00x
GPU (NVIDIA H100 / Jetson)                  1474          147       3.70x
TPU (Google TPU v4 / Coral)                  109         10.9       0.27x
KPU (Stillwater Domain Flow)                43.5         4.35       0.11x
```

### New EDDO Section in Phase Breakdown
```
KPU EDDO SCRATCHPADS
------------------------- -------------------- -------------------- --------------------
  Tile Scratchpad                          n/a                  n/a         15.0nJ (34%)
  Global Scratchpad                        n/a                  n/a          6.0nJ (14%)
  Streaming Buffer                         n/a                  n/a           1.5nJ (3%)
  DMA Setup                                n/a                  n/a             6pJ (0%)
```

## Key Technical Insight

**EDDO vs Cache:**

| Aspect | Cache Hierarchy (CPU) | EDDO Scratchpad (KPU) |
|--------|----------------------|----------------------|
| Management | Hardware-managed, automatic | Software-managed, explicit |
| Data placement | Tag-based, reactive | Compiler-directed, proactive |
| Miss handling | Hardware stalls, fetches | No misses - data pre-staged |
| Energy overhead | Tag lookups, coherence | Zero tag energy, no coherence |
| Predictability | Variable latency | Deterministic timing |

This distinction is fundamental to understanding why Domain Flow architectures achieve better energy efficiency - they eliminate the overhead of speculative, reactive memory management.

## Files Modified

1. `src/graphs/hardware/architectural_energy.py` - DomainFlowEnergyModel refactor
2. `src/graphs/hardware/cycle_energy/base.py` - Added EDDO CyclePhase enums
3. `src/graphs/hardware/cycle_energy/kpu.py` - Updated memory section
4. `src/graphs/hardware/cycle_energy/comparison.py` - Added consistent-scale formatting
5. `src/graphs/hardware/cycle_energy/__init__.py` - Exported new functions
6. `cli/compare_architecture_energy.py` - Fixed units and bugs

## Testing

```bash
# Test architecture comparison
./cli/compare_architecture_energy.py --arch-comparison 8nm-x86

# Test sweep mode
./cli/compare_architecture_energy.py --sweep --arch-comparison 8nm-x86

# Run demo
python examples/demo_architectural_energy.py
```

All tests pass and output shows correct EDDO terminology with consistent units.
