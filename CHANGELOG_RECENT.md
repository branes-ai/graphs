# Recent Changes (Last 3 Months)

**Purpose**: Quick context for AI assistants resuming work. Full history in `CHANGELOG.md`.

**Last Updated**: 2025-10-27

---

## [2025-10-27] - CLI Documentation: Comprehensive How-To Guides

### Added

- **CLI Documentation Suite** (`cli/docs/`)
  - 7 comprehensive how-to markdown guides (~17,500 lines total)
  - 50+ usage examples across all tools
  - Step-by-step tutorials for new developers
  - Troubleshooting sections for common issues

- **Core Analysis Tool Guides**
  - `analyze_graph_mapping.md` (4,300+ lines): Complete hardware mapping guide
  - `compare_models.md` (2,400+ lines): Model comparison across hardware
  - `list_hardware_mappers.md` (3,000+ lines): Hardware discovery (35+ models)
  - `discover_models.md` (2,000+ lines): Model discovery (140+ models)

- **Profiling & Partitioning Guides**
  - `profile_graph.md` (1,500+ lines): Hardware-independent profiling
  - `partitioner.md` (800+ lines): Graph partitioning strategies

- **Specialized Comparison Guide**
  - `comparison_tools.md` (3,500+ lines): 5 specialized tools documented
    - Automotive ADAS comparison
    - Datacenter CPU comparison
    - Edge AI platform comparison
    - IP core comparison for SoC integration
    - CPU variant comparison

- **Enhanced CLI README** (`cli/README.md`)
  - Documentation links section
  - Quick reference workflows (4 common patterns)
  - Tool selection guide table
  - Common workflow examples

### Documentation Features

**Consistent Structure:**
- Overview and capabilities
- Quick start (30-second setup)
- Complete command-line reference
- Real-world usage examples
- Output format explanations
- Interpretation guides
- Troubleshooting sections
- Advanced usage patterns
- Cross-tool references

**Coverage:**
- 12 CLI scripts fully documented
- 35+ hardware models explained
- 140+ DNN models covered
- Deployment scenarios (datacenter, edge, automotive, embedded)
- Architecture comparisons (CPU vs GPU vs TPU vs KPU vs DSP)

### Impact

**Developer Experience:**
- New users can get started in minutes (vs hours of code exploration)
- Clear examples for every major use case
- Troubleshooting guides reduce support burden
- Cross-references enable discovery of related tools

**Knowledge Transfer:**
- Complete reference for tool capabilities
- Hardware selection guidance
- Model selection guidance
- Performance optimization tips

---

## [2025-10-26] - Graph Mapping Analysis Tool: Hardware Comparison & Architecture Legend

### Added

- **Hardware Comparison Mode** (`cli/analyze_graph_mapping.py`)
  - `--compare` flag for side-by-side hardware comparison
  - Comprehensive comparison table + detailed subgraph-by-subgraph allocation
  - Performance and energy efficiency rankings

- **Hardware Architecture Legend**
  - Shows compute building block specs (CUDA cores/SM, ops/clock, GOPS)
  - GPU, KPU, CPU, TPU, DSP specifications
  - Memory subsystem details

- **Jetson Specifications Reference** (`docs/hardware/jetson_specifications.md`)
  - Official NVIDIA specs: Thor, AGX Orin, Orin Nano
  - Key: **128 CUDA cores per SM** (Ampere/Blackwell)

- **New Hardware**: KPU-T64, i7-12700K, Ryzen-7-5800X

### Fixed

- **CRITICAL: Jetson AGX Orin** - 32 SMs → 16 SMs (correct official specs)
  - Added microarchitecture fields: `cuda_cores_per_sm=128`, `tensor_cores_per_sm=4`
  - More realistic performance (1.822ms vs 1.159ms)
- **TDP Extraction** - Fallback for legacy hardware without thermal profiles
- **Vendor Naming** - Added "Stillwater" to KPU products

### Impact

**Comparison Insights:**
- Jetson vs KPU-T256: 14.6× faster (was 23× - now more realistic)
- Root cause: KPU tile allocation collapses for low-parallelism subgraphs

---

## [2025-10-25] - Automotive ADAS Comparison Fix: Data-Driven Recommendations

### Fixed

- **Removed Hardcoded Recommendations** - Replaced with data-driven scoring system
- **Added ADAS Use Case Definitions** - 7 realistic use cases with actual TOPS requirements
- **Implemented Platform Scoring** - Multi-factor scoring (50% performance, 20% efficiency, 20% latency, 10% safety)

### Impact

**Before**: Always recommended TI TDA4VM (6 TOPS) even for L3 (300 TOPS required)
**After**: Data-driven recommendations with clear warnings when insufficient

---

## [2025-10-25] - Hardware Test Suite and Thermal Profile Completion

### Added

- **Comprehensive Hardware Test Suite** (`tests/hardware/`)
  - `test_power_modeling.py`: Validates IDLE_POWER_FRACTION = 0.5 across all 6 mapper types
  - `test_thermal_profiles.py`: Validates thermal_operating_points for all 32 hardware models

- **Missing Thermal Operating Points Fixed**
  - H100 PCIe: 350W TDP, active-air cooling
  - TPU v4: 350W TDP, active-liquid cooling

### Validation

- ✅ All 29 hardware tests pass
- ✅ All 32 hardware models have thermal_operating_points

---

## [2025-10-25] - Leakage-Based Power Modeling Phase 2: DSP, DPU, and KPU

### Added

- **Idle Power Modeling Extended to Edge AI Accelerators**
  - DSP, DPU, KPU mappers: 50% idle power model
  - Total of **6 mappers** with idle power: GPU, TPU, CPU, DSP, DPU, KPU
  - Total of **32 hardware models** covered

---

## [2025-10-24] - Package Reorganization

### Changed

- **Massive Reorganization**: Split `characterize/` into focused packages
  - `ir/`: Intermediate representation
  - `transform/`: Graph transformations
  - `analysis/`: Performance analysis
  - `hardware/`: Hardware modeling and mapping

- **File Size Reduction**: `resource_model.py` from 6096 lines to 759 lines (87.5% reduction)

---

## Archive Information

**Archival Policy**: Entries older than 3 months moved to `CHANGELOG.md`
**Full History**: See `CHANGELOG.md` for complete project history
**Session Logs**: See `docs/sessions/` for detailed session-by-session work

---

## Quick Context Summary

**Current Focus**: Phase 2 Hardware Mapping
- ✅ Hardware comparison mode with architecture legend
- ✅ Jetson specifications corrected (critical bug fix)
- ✅ Idle power modeling complete (all 6 mappers)
- ✅ Thermal operating points (all 32 models)
- ✅ Hardware test suite (29 tests passing)
- ✅ Automotive ADAS comparison fixed (data-driven)

**Hardware Coverage**:
- 6 mapper types: GPU, TPU, CPU, DSP, DPU, KPU
- 35 hardware models: datacenter (14), edge (7), automotive (5), mobile (1), accelerators (8)

**Key Files**:
- `cli/analyze_graph_mapping.py`: Hardware comparison tool with architecture legend
- `docs/hardware/jetson_specifications.md`: Official Jetson specs reference
- `tests/hardware/`: Test infrastructure
- `src/graphs/hardware/`: Resource models and mappers
