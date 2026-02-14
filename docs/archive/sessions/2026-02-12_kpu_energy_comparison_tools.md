# Session Summary: KPU vs GPU Energy Comparison Tools

**Date**: 2026-02-12
**Phase**: KPU Marketing & Analysis Tools
**Status**: Complete

---

## Goals for This Session

1. Create energy comparison graphs between GPU (512 TensorCore MACs) and KPU (systolic array) for matrix multiplication
2. Sweep KPU energy advantage ratio across 5 systolic array sizes (16x16 through 256x256)
3. Diagnose and fix tile boundary energy discontinuities in the model
4. Create a web-native D3.js chart suitable for marketing website

---

## What We Accomplished

### 1. GPU vs KPU Energy Comparison (`tools/energy_comparison_gpu_vs_kpu.py`)

Created a 4-panel matplotlib figure comparing GPU and KPU energy for NxN matmul:

- **Panel 1**: Absolute energy (pJ) on log-log scale
- **Panel 2**: Energy per MAC breakdown (compute, memory access, leakage)
- **Panel 3**: Energy breakdown stacked area
- **Panel 4**: KPU Energy Advantage Ratio (GPU energy / KPU energy)

Energy model (5nm, FP16, weight-stationary dataflow):
- **GPU**: 512 TensorCore MACs, register-file sourced
  - Compute: 0.638 pJ/MAC (5nm FP16 tensor core)
  - Register file: 0.429 pJ/MAC (2 B/MAC at 0.214 pJ/byte)
  - Instruction fetch: 0.156 pJ/MAC (amortized over 64 MACs)
  - Scheduling: 0.040 pJ/MAC
  - Total dynamic: 1.262 pJ/MAC
- **KPU**: 64x64 systolic array, hardwired MAC
  - Compute: 0.15 pJ/MAC (no instruction decode)
  - PE forwarding: 0.05 pJ/MAC (~100um wire at 5nm)
  - Control: 0.015 pJ/MAC (domain-flow distributed CAM)
  - Weight load and activation injection amortized across tiles

### 2. Array Size Sweep (`tools/energy_sweep_array_sizes.py`)

Parameterized KPU energy model sweeping 5 array sizes against fixed GPU baseline:
- 16x16 (256 PEs), 32x32 (1024), 64x64 (4096), 128x128 (16384), 256x256 (65536)
- Shows how larger arrays amortize weight-load and leakage better for large matmuls but suffer worse underutilization for small ones

### 3. Tile Boundary Dip Diagnosis and Fix

**Problem**: Energy advantage ratio showed dips/oscillations at tile boundaries (e.g., N=64 to N=72 for 64x64 array). The ratio would drop before recovering.

**Root Cause** (diagnosed with `tools/diagnose_dip.py`):
- When N crosses the array dimension, `ceil(N/A)^2` tile pairs jumps from 1 to 4
- Each tile streams all N activation rows, so 4 tiles = 4x streaming passes
- Leakage energy (97% of excess) scaled with total execution time across all tiles
- Activation injection energy (3% of excess) also had a discontinuity

**Fix** (two-part):
1. **Per-tile execution model**: Partial tiles use actual dimensions for pipeline fill/drain (`2*(kd-1)` instead of `2*A`), making partial tiles cheaper
2. **Power-gated leakage**: Only active PEs (`kd*nd`) get full leakage (0.08 pJ/unit/ns); inactive PEs get 5% residual through power switches (0.004 pJ/unit/ns). This is consistent with domain-flow architecture that enables deterministic power gating of unused PEs.

Result: Smooth, monotonically increasing curves with no visible dips.

### 4. Web-Native D3.js Chart (`tools/kpu_energy_advantage.html`)

Self-contained HTML page with D3.js v7 for marketing website:
- Title: "KPU Energy Advantage Ratio"
- 5 colored curves (red, orange, green, blue, purple)
- Logarithmic x-axis with only 10^1, 10^2, 10^3 superscript ticks
- Green/red zone fills for "KPU more efficient" / "GPU more efficient"
- Diamond markers at sparse intervals
- Per-curve area fill between line and parity
- Right-edge value annotations (4.3x through 4.8x) with collision avoidance
- No legend, no subtitle, no inline curve labels -- clean marketing style

---

## Key Design Decisions

1. **Power gating residual at 5%**: Conservative estimate; header switches typically achieve 90-99% leakage reduction. This is the key physical mechanism that eliminates tile boundary artifacts.

2. **Per-tile streaming model**: Each tile streams all M=N activation rows, with pipeline fill/drain proportional to actual tile k_dim. This accurately captures partial tile costs.

3. **D3.js over static images**: Web-native SVG rendering scales perfectly across screen sizes and enables future interactivity if needed.

4. **Data embedded in HTML**: Pre-computed ratio values baked into the HTML as JSON arrays rather than computing at load time -- ensures consistency with the validated Python model.

---

## Files Created/Modified

| File | Description |
|------|-------------|
| `tools/energy_comparison_gpu_vs_kpu.py` | 4-panel matplotlib GPU vs KPU comparison |
| `tools/energy_sweep_array_sizes.py` | Array size sweep with per-tile model |
| `tools/diagnose_dip.py` | Tile boundary diagnostic tool |
| `tools/kpu_energy_advantage.html` | D3.js web chart for marketing |
| `energy_comparison_gpu_vs_kpu.png` | Output: 4-panel comparison |
| `energy_comparison_gpu_vs_kpu.svg` | Output: 4-panel comparison (vector) |
| `energy_advantage_array_sweep.png` | Output: array size sweep |
| `energy_advantage_array_sweep.svg` | Output: array size sweep (vector) |

---

## Key Numbers

- Asymptotic KPU advantage (dynamic energy only): ~5.9x
- At N=4096: 16x16 array = 4.3x, 256x256 array = 4.8x
- Crossover points (where KPU becomes more efficient):
  - 16x16: N ~ 4-6
  - 256x256: N ~ 48-64
- GPU dynamic energy/MAC: 1.262 pJ
- KPU dynamic energy/MAC (core only): 0.215 pJ
