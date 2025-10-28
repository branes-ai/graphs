# Recent Changes (Last 3 Months)

**Purpose**: Quick context for AI assistants resuming work. Full history in `CHANGELOG.md`.

**Last Updated**: 2025-10-28

---

## [2025-10-28] - Unified Range Selection Across CLI Tools

### Fixed

- **Critical: Off-by-One Bug in Range Selection**
  - `--start 5 --end 10` was showing nodes 6-10 (wrong start) and then 6-9 (wrong end)
  - `--around 10 --context 2` was showing nodes 9-13 instead of 8-12
  - Root cause: User-facing 1-based node numbers treated as 0-based indices
  - Root cause: `--end` was incorrectly decremented (should stay as-is for Python slicing)
  - Fixed in `cli/graph_explorer.py::determine_range()`

- **Variable Shadowing in FusionBasedPartitioner**
  - `total_nodes` used for both graph size and subgraph node count
  - Renamed to `sg_total_nodes` for subgraph context
  - Fixed in `src/graphs/transform/partitioning/fusion_partitioner.py`

### Added

- **Unified Range Selection for `partition_analyzer.py`**
  - Implemented `determine_range()` method (identical to graph_explorer)
  - Added `--start`, `--end`, `--around`, `--context` arguments
  - Updated `visualize_strategy()` to use start/end parameters
  - Now feature-complete with graph_explorer

- **Start/End Support in FusionBasedPartitioner**
  - `visualize_partitioning()`: Changed signature from `max_nodes` to `start/end`
  - `visualize_partitioning_colored()`: Same changes
  - Proper node enumeration with correct display numbering
  - Accurate footer counts for nodes before/after range

### Changed

- **Unified Node Addressing Convention**
  - **1-based numbering**: Node numbers match display output (node #5 = `--start 5`)
  - **Inclusive ranges**: Both start and end are inclusive (`--start 5 --end 10` shows 5,6,7,8,9,10)
  - **Natural language semantics**: "start at 5, end at 10" means exactly that
  - Applied to both `graph_explorer.py` and `partition_analyzer.py`

- **Documentation Updates**
  - `cli/README.md`: Added "Common Conventions" section explaining unified behavior
  - `cli/docs/partition_analyzer.md`: Added range selection section with examples
  - `cli/docs/graph_explorer.md`: Clarified 1-based inclusive behavior
  - All examples updated to show three selection methods

### Impact

**User Experience:**
- Before: `--start 5 --end 10` showed nodes 6-9 (confusing, wrong)
- After: `--start 5 --end 10` shows nodes 5-10 (intuitive, correct)
- "Learn once, use everywhere" - same commands work across all visualization tools

**Testing Results:**
- ✅ `--start 5 --end 10`: Shows nodes 5-10 (6 nodes) correctly
- ✅ `--around 10 --context 2`: Shows nodes 8-12 (5 nodes) correctly
- ✅ `--max-nodes 5`: Shows nodes 1-5 (backward compatible)
- ✅ Both tools show identical behavior

**Key Insight:**
For `--end`, we keep the user value as-is (don't subtract 1) because Python's slice `[start:end]` is already exclusive on end. To show node 10 (1-based), we need slice index 10.

---

## [2025-10-28] - Graph Explorer & Tool Renaming

### Fixed

- **Package Import Structure** (Critical)
  - `pyproject.toml`: Changed from hardcoded package list to automatic discovery
  - `[tool.setuptools.packages.find]` now auto-discovers all 21 packages
  - Fixed `examples/visualize_partitioning.py` imports (removed sys.path manipulation)
  - Impact: Clean package installation with `pip install -e .`

### Added

- **Graph Explorer Tool** (`cli/graph_explorer.py`)
  - Three-level progressive disclosure UX:
    1. **Level 1 (no args)**: Discover 20+ models organized by family
    2. **Level 2 (--model only)**: Comprehensive summary statistics
    3. **Level 3 (--model + range)**: Detailed side-by-side visualization
  - Prevents accidental output floods (large models have 300+ nodes)
  - Flexible range selection: `--start/--end`, `--around/--context`, `--max-nodes`
  - Summary statistics: FLOPs, memory, arithmetic intensity, bottleneck distribution
  - 368 lines of production code

- **Comprehensive Documentation** (`cli/docs/graph_explorer.md`)
  - ~600 lines covering all three modes
  - Real-world examples and workflows
  - Troubleshooting guide
  - Integration with other tools

### Changed

- **Tool Renaming for Clarity**
  - `cli/visualize_partitioning.py` → `cli/graph_explorer.py`
  - `cli/partitioner.py` → `cli/partition_analyzer.py`
  - Rationale: "Explorer" = inspection, "Analyzer" = strategy comparison
  - Updated all documentation and cross-references
  - Class names: `GraphExplorerCLI`, `PartitionAnalyzerCLI`

- **CLI Tool Organization** (`cli/README.md`)
  - Reorganized into logical sections:
    - Discovery Tools: Profiling & Partitioning
    - Core Analysis Tools
    - Specialized Comparisons
  - Natural workflow order: discover → explore → analyze → map

- **Example Script Simplified** (`examples/visualize_partitioning.py`)
  - Reduced to 110-line teaching example
  - Step-by-step API demonstration
  - Clear comments and variations
  - Points to CLI tool for production use

### Impact

**Progressive Disclosure:**
- Before: `--model vit_l_16` would dump 300 nodes → terminal flood
- After: Shows informative summary → user makes informed decision

**Tool Clarity:**
- Before: "visualize_partitioning" + "partitioner" (confusing overlap)
- After: "graph_explorer" + "partition_analyzer" (clear distinction)

**Developer Experience:**
- Summary mode prevents information overload
- Natural workflow: discover → understand → investigate
- Model discovery built-in (no need to remember names)

**Testing Results:**
- ✅ All three levels working (no args, summary, visualization)
- ✅ Range selection working (--max-nodes, --around, --start/--end)
- ✅ All renamed tools tested and working
- ✅ Example script runs successfully

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
