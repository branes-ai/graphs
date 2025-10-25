# Recent Changes (Last 3 Months)

**Purpose**: Quick context for AI assistants resuming work. Full history in `CHANGELOG.md`.

**Last Updated**: 2025-10-25

---

## [2025-10-25] - Automotive ADAS Comparison Fix: Data-Driven Recommendations

### Problem Statement

The automotive comparison script (`cli/compare_automotive_adas.py`) was recommending TI TDA4VM for all ADAS use cases, including L3 Highway Pilot, despite:
- **TI TDA4VM performance**: ~6 TOPS INT8 @ 20W
- **L3 requirements**: 300 TOPS minimum (50× gap)
- **L4 requirements**: 1000+ TOPS (166× gap)

**Root cause**: Hardcoded recommendation table (lines 443-455) that ignored all benchmark results.

### Fixed

- **Removed Hardcoded Recommendations** - Replaced with data-driven scoring system
- **Added ADAS Use Case Definitions** - 7 realistic use cases with actual TOPS requirements (3-300 TOPS)
- **Implemented Platform Scoring System** - Multi-factor scoring (50% performance, 20% efficiency, 20% latency, 10% safety)
- **Data-Driven Recommendation Table** - Shows effective TOPS and status indicators
- **Performance Warnings Section** - Automatic warnings when platforms don't meet TOPS requirements
- **Reality-Based Key Findings** - Industry benchmarks (Tesla FSD, Waymo)

### Impact

**Before**: Always recommended TI TDA4VM (6 TOPS) even for L3 (300 TOPS required)
**After**: Data-driven recommendations with clear warnings when insufficient

---

## [2025-10-25] - Hardware Test Suite and Thermal Profile Completion

### Added

- **Comprehensive Hardware Test Suite** (`tests/hardware/`)
  - `test_power_modeling.py` (40+ tests): Validates IDLE_POWER_FRACTION = 0.5 across all 6 mapper types
  - `test_thermal_profiles.py` (25+ tests): Validates thermal_operating_points for all 32 hardware models
  - `run_tests.py`: Custom test runner (pytest-free)
  - `README.md`: Complete documentation

- **Missing Thermal Operating Points Fixed**
  - H100 PCIe: Added 350W TDP, active-air cooling
  - TPU v4: Added 350W TDP, active-liquid cooling

### Validation

- ✅ All 29 hardware tests pass
- ✅ All 32 hardware models have thermal_operating_points
- ✅ Test execution time: <2 seconds

---

## [2025-10-25] - Leakage-Based Power Modeling Phase 2: DSP, DPU, and KPU

### Added

- **Idle Power Modeling Extended to Edge AI Accelerators**
  - DSP Mapper: 8 models (QRB5165, TI TDA4x series, NPU IP cores)
  - DPU Mapper: 1 model (Xilinx Vitis AI DPU)
  - KPU Mapper: 3 models (KPU-T64, T256, T768)
  - All use same 50% idle power model: `P_total = P_idle + P_dynamic` where `P_idle = TDP × 0.5`

- **Thermal Operating Points**
  - Xilinx Vitis AI DPU: 20W TDP (VE2302 edge-optimized)

### Impact

- Total of **6 mappers** now include idle power: GPU, TPU, CPU, DSP, DPU, KPU
- Total of **32 hardware models** covered

---

## [2025-10-25] - Leakage-Based Power Modeling Phase 1: TPU and Datacenter CPUs

### Added

- **Idle Power Modeling** - 50% of TDP consumed at idle due to nanoscale leakage
  - TPUMapper: `IDLE_POWER_FRACTION = 0.5`, `compute_energy_with_idle_power()`
  - CPUMapper: Same implementation for Intel Xeon, AMD EPYC
  - Thermal operating points: All models now include TDP specifications

### Changed

- Energy calculations now include idle power (previously only dynamic power)

---

## [2025-10-24] - Package Reorganization

### Changed

- **Massive Reorganization**: Split `characterize/` into focused packages
  - `ir/`: Intermediate representation (graph structures)
  - `transform/`: Graph transformations (partitioning, fusion, tiling)
  - `analysis/`: Performance analysis (concurrency, roofline)
  - `hardware/`: Hardware modeling and mapping

- **File Size Reduction**: `resource_model.py` from 6096 lines to 759 lines (87.5% reduction)
  - Created `models/` subdirectory with categorical organization
  - 32 hardware models split into separate files
  - Maintained backward compatibility via __init__.py re-exports

---

## Archive Information

**Archival Policy**: Entries older than 3 months moved to `CHANGELOG.md`
**Full History**: See `CHANGELOG.md` for complete project history
**Session Logs**: See `docs/sessions/` for detailed session-by-session work

---

## Quick Context Summary

**Current Focus**: Phase 2 Hardware Mapping
- ✅ Idle power modeling complete (all 6 mappers)
- ✅ Thermal operating points (all 32 models)
- ✅ Hardware test suite (29 tests passing)
- ✅ Automotive ADAS comparison fixed (data-driven)

**Hardware Coverage**:
- 6 mapper types: GPU, TPU, CPU, DSP, DPU, KPU
- 32 hardware models: datacenter (14), edge (4), automotive (5), mobile (1), accelerators (8)

**Key Files**:
- `tests/hardware/`: Test infrastructure
- `cli/compare_automotive_adas.py`: Fixed automotive comparison
- `src/graphs/hardware/`: Resource models and mappers
