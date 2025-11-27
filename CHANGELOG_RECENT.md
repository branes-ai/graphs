# Recent Changes (Last 3 Months)

**Purpose**: Quick context for AI assistants resuming work. Full history in `CHANGELOG.md`.

**Last Updated**: 2025-11-27

---

## [2025-11-27] - Multi-Calibration Profile Support & Precision Fixes

### Added

- **Multi-Calibration Directory Structure**:
  - Hardware profiles now support multiple calibrations in `calibrations/` subdirectory
  - Calibration filename format: `{power_mode}_{frequency_mhz}MHz_{framework}.json`
  - Examples: `MAXN_625MHz_pytorch.json`, `7W_306MHz_numpy.json`, `performance_4900MHz_numpy.json`
  - New `calibration_filter` parameter in `HardwareProfile.load()` and `registry.get()`
  - Filter by `power_mode`, `freq_mhz`, `framework` - defaults to most recent
  - New `HardwareProfile.list_calibrations()` and `registry.list_calibrations()` methods

- **list_calibrations Script** (`scripts/hardware_db/list_calibrations.py`):
  - List all calibrations across hardware registry
  - Filter by `--hardware`, `--framework`, `--power-mode`
  - `--detail` shows GFLOPS, bandwidth, date for each calibration
  - `--summary` shows statistics by hardware/framework/power mode

- **Migration Script** (`cli/migrate_calibrations.py`):
  - Migrates old `calibration.json` files to new `calibrations/` structure
  - Supports `--dry-run` mode to preview changes

- **TF32 Precision Support**:
  - Added `tf32` to default GPU precision list in calibrator
  - Added `tf32` to `CANONICAL_PRECISION_ORDER` in schema (between fp32 and fp16)
  - TF32 uses FP32 dtype with Tensor Core truncation (19-bit mantissa)

### Fixed

- **GEMM Early Termination Bug**:
  - Root cause: Early termination triggered at size 32 where timing overhead dominates
  - 32x32 GEMM = 65K FLOPs at 0.13ms = 0.5 GFLOPS (below 1.0 threshold)
  - Fix: Added `min_early_termination_size = 256` threshold
  - Early termination only triggers after testing meaningful matrix sizes
  - Result: Jetson CPU GEMM now correctly shows 50+ GFLOPS (was 0.5 GFLOPS)

- **Duplicate Precision in Support Summary**:
  - Root cause: Order of operation processing could add precision to unsupported before supported
  - Fix: Added `actually_unsupported -= actually_supported` cleanup
  - Result: Clean precision support summary without duplicates

- **TF32 Not Displayed in Benchmarks**:
  - Root cause: `tf32` missing from `CANONICAL_PRECISION_ORDER`
  - Display loops using `for p in CANONICAL_ORDER if p in results` silently skipped tf32
  - Fix: Added `tf32` to canonical order list

### Changed

- **Hardware Registry Directory Structure**:
  ```
  hardware_registry/
  ├── cpu/
  │   └── jetson_orin_nano_cpu/
  │       ├── spec.json
  │       └── calibrations/
  │           ├── schedutil_729MHz_numpy.json
  │           └── schedutil_883MHz_numpy.json
  └── gpu/
      └── jetson_orin_nano_gpu/
          ├── spec.json
          └── calibrations/
              └── 7W_306MHz_pytorch.json
  ```

- **No Backward Compatibility**: Old `calibration.json` format not supported; use migration script

### Documentation

- Session log: `docs/sessions/2025-11-27_calibration_multi_profile_support.md`

---

## [2025-11-18] - Hardware Database Schema Consolidation & GPU Auto-Detection

### Added

- **GPU Auto-Detection**:
  - `detect_and_create_gpu_specs()` function in `auto_detect_and_add.py`
  - Detects all NVIDIA GPUs via `nvidia-smi` (cross-platform Windows/Linux/macOS)
  - Auto-infers architecture from model name (Hopper, Ada Lovelace, Ampere, Turing, Pascal, Volta)
  - Detects Tensor Core capability from CUDA compute capability (≥7.0)
  - Creates hardware specs with consolidated 5-block structure
  - New CLI flags: `--detect-gpus` (CPU + GPUs), `--gpus-only` (skip CPU)
  - Support for multiple GPU detection and batch JSON output
  - Auto-detected fields: model name, VRAM size, CUDA capability, driver version, architecture
  - Placeholder fields (require manual entry): SM count, CUDA cores, Tensor cores, RT cores, frequencies, theoretical peaks, cache hierarchy
  - Test script: `test_gpu_detection.py` for quick GPU detection validation

- **Consolidated Hardware Schema** (5-Block Structure):
  - Migrated all hardware specs to new consolidated format
  - Block 1: `system` - Vendor, model, architecture, device type, platform, ISA extensions, special features, TDP, release date, notes
  - Block 2: `core_info` - Cores, threads, frequencies, core clusters (for heterogeneous CPUs/GPUs)
  - Block 3: `memory_subsystem` - Total size, peak bandwidth, memory channels (with detailed channel info)
  - Block 4: `onchip_memory_hierarchy` - Cache levels with structured metadata (level, type, scope, size, associativity, line size)
  - Block 5: `mapper` - Mapper class, config, hints
  - GPU-specific aggregate fields: `total_cuda_cores`, `total_tensor_cores`, `total_sms`, `total_rt_cores`, `cuda_capability`
  - Backward compatibility: `from_dict()` auto-converts old scattered format
  - Forward compatibility: `to_dict()` excludes deprecated fields
  - Updated all Jetson Orin hardware files (Nano CPU/GPU, AGX CPU/GPU)

### Fixed

- **Windows Cache Detection Bug**:
  - Root cause: Stale Python bytecode cache (`.pyc` files) from old detector code
  - Windows was using cached old version even after source updates
  - Solution: `pip install -e .` forces bytecode refresh
  - Windows cache fallback now properly triggers when py-cpuinfo lacks cache SIZE fields
  - Fixed condition from `if not cache_info` to `has_cache_sizes = any(key in cache_info for key in ['l1_dcache_kb', ...])`
  - py-cpuinfo may return metadata (associativity, line size) but not sizes on Windows
  - wmic fallback properly populates L2/L3 cache sizes on Windows
  - `_extract_cache_info()` now builds `cache_levels` array correctly on all platforms
  - Test script: `test_cache_extraction.py` for isolated cache detection testing
  - Test script: `clear_cache_and_test.py` for bytecode cleanup and validation

- **Auto-Detection Schema Compliance**:
  - `auto_detect_and_add.py` was creating old scattered format on Windows
  - Updated to use consolidated 5-block structure
  - Fixed `add_hardware.py` interactive mode to use new schema
  - Fixed `calibrate_hardware.py` preset handling to use new schema
  - Fixed `migrate_presets.py` to use timezone-aware datetimes
  - All HardwareSpec creation now uses consolidated blocks

- **Schema Validation**:
  - Changed `total_size_gb` validation from "must be positive" to "cannot be negative"
  - Allows `total_size_gb: 0` for unknown memory configurations
  - Auto-detection creates minimal `memory_subsystem` when detection fails
  - Ensures `memory_subsystem` always present if bandwidth is known

- **Jetson Orin AGX GPU Spec**:
  - Fixed wrong field names: `cuda_cores` → `total_cuda_cores`, `tensor_cores` → `total_tensor_cores`, `sms` → `total_sms`
  - Fixed wrong SM count: 128 → 16 (correct: 16 SMs × 128 cores/SM = 2048 total CUDA cores)
  - Fixed wrong `total_sms`: 128 → 16
  - Updated notes from "4× Nano" to "2× Nano" (correct performance ratio)
  - Removed duplicate fields from inside cluster definition

### Changed

- **Datetime Handling**:
  - Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')`
  - Updated in: `auto_detect_and_add.py`, `add_hardware.py`, `calibrate_hardware.py`, `migrate_presets.py`
  - All timestamps now timezone-aware (UTC)

- **Auto-Detection Workflow**:
  - Now detects both CPU and GPU when `--detect-gpus` specified
  - Processes multiple hardware specs in single run
  - Batch JSON output: creates separate file per device
  - Improved next-steps guidance for GPU completion (SM count, cores, calibration)
  - Enhanced validation and review output for multiple devices

### Documentation

- Session log: `docs/sessions/2025-11-18_windows_cache_gpu_detection.md`
- Updated CHANGELOG_RECENT.md with schema consolidation and GPU detection

### Notes

- Windows cache detection requires `pip install -e .` after code updates to clear bytecode cache
- GPU auto-detection creates minimal valid specs that require manual completion
- Recommended workflow: `--gpus-only -o .` → manually complete JSON → move to database
- All new hardware specs use consolidated 5-block structure
- Legacy scattered format still supported via `from_dict()` auto-conversion

---

## [2025-11-17] - Hardware Database System (Phases 1-4 Complete)

### Added

- **Hardware Database Foundation** (Phase 1):
  - JSON-based hardware specification database in `hardware_database/`
  - `HardwareSpec` schema with 30+ fields (identity, detection, performance, architecture, metadata)
  - `HardwareDatabase` manager with caching, validation, and query capabilities
  - Directory structure: `cpu/{vendor}/`, `gpu/{vendor}/`, `accelerators/{vendor}/`
  - 9 initial hardware specs migrated from PRESETS (4 CPUs, 4 GPUs, 1 TPU)
  - JSON schema validation (`schema.json`)
  - Files: `src/graphs/hardware/database/schema.py` (210 lines), `manager.py` (180 lines)

- **Cross-Platform Hardware Detection** (Phase 2):
  - `HardwareDetector` class with CPU/GPU auto-detection
  - Cross-platform support via `psutil` and `py-cpuinfo` libraries
  - Platform support: Linux, Windows, macOS (x86_64, aarch64, arm64)
  - Hybrid CPU detection (P-cores + E-cores) using mathematical approach: `E_cores = 2*cores - threads`
  - Pattern matching with confidence scoring (0-100%)
  - Detection tool: `scripts/hardware_db/detect_hardware.py` (280 lines)
  - Dependencies added to `pyproject.toml` and `requirements.txt`
  - File: `src/graphs/hardware/database/detector.py` (450 lines)

- **Management Tools** (Phase 3):
  - `scripts/hardware_db/add_hardware.py` (434 lines): Interactive wizard for adding hardware
  - `scripts/hardware_db/update_hardware.py` (266 lines): Update existing hardware specs
  - `scripts/hardware_db/delete_hardware.py` (69 lines): Remove hardware from database
  - `scripts/hardware_db/improve_patterns.py` (230 lines): Automatic pattern generation
  - `scripts/hardware_db/list_hardware.py` (120 lines): List all hardware with filtering
  - `scripts/hardware_db/query_hardware.py` (150 lines): Query by ID or criteria
  - `scripts/hardware_db/validate_database.py` (100 lines): Schema validation
  - `scripts/hardware_db/migrate_presets.py`: One-time migration from legacy PRESETS
  - Applied pattern improvements to all 9 hardware specs (42 total patterns, avg 4.7 per hardware)

- **Calibration Integration** (Phase 4):
  - Auto-detection integrated with `cli/calibrate_hardware.py`
  - `--preset` flag deprecated (backward compatible with warning)
  - `--id` flag for database lookup
  - Auto-detection now the default mode (no flags required)
  - `scripts/hardware_db/compare_calibration.py` (273 lines): Compare theoretical vs measured performance
  - Auto-identification from calibration filename
  - Efficiency percentages by precision (fp64, fp32, fp16, int8, etc.)
  - Performance recommendations (excellent ≥80%, good ≥50%, moderate ≥20%, low <20%)

### Changed

- **Calibration Workflow** (Breaking Change, Backward Compatible):
  - Old: `./cli/calibrate_hardware.py --preset i7-12700k` (deprecated)
  - New: `./cli/calibrate_hardware.py` (auto-detect) or `./cli/calibrate_hardware.py --id i7_12700k`
  - `--preset` shows deprecation warning but still works
  - Hardware specs now loaded from database instead of hardcoded PRESETS

- **Hardware Detection Method**:
  - Replaced platform-specific subprocess calls with cross-platform libraries
  - psutil: cores/threads/frequency (Windows/Linux/macOS)
  - py-cpuinfo: CPU model/vendor/ISA extensions (Windows/Linux/macOS)
  - Platform-specific code only for specialized features (E-core refinement on Linux)

### Fixed

- **CPU Detection Accuracy**:
  - Was reporting "20 cores, 20 threads" for i7-12700K (incorrect)
  - Now reports "12 cores (8P + 4E), 20 threads" (correct)
  - Root cause: Using `nproc --all` which returns logical CPUs, not physical cores
  - Fixed by using cross-platform psutil and mathematical E-core detection

- **Database Update Method**:
  - `db.update(spec)` was broken (TypeError: unhashable type)
  - Fixed by using `db.add(spec, overwrite=True)` pattern
  - Applied to `update_hardware.py` and `improve_patterns.py`

### Documentation

- `hardware_database/README.md`: Database usage guide with calibration workflow
- `scripts/hardware_db/README.md` (442 lines): Complete management tools guide
- `docs/PHASE1_DATABASE_FOUNDATION.md`: Phase 1 implementation details
- `docs/PHASE2_HARDWARE_DETECTION.md`: Phase 2 cross-platform detection
- `docs/PHASE4_CALIBRATION_INTEGRATION.md`: Phase 4 calibration integration
- `docs/HARDWARE_DATABASE_IMPLEMENTATION.md` (645 lines): Complete implementation summary
- `docs/sessions/2025-11-17_hardware_database_implementation.md`: Session log

### Migration Guide

**For Users:**
```bash
# Old workflow (deprecated)
./cli/calibrate_hardware.py --preset i7-12700k

# New workflow (recommended)
./cli/calibrate_hardware.py  # Auto-detect
./cli/calibrate_hardware.py --id i7_12700k  # Explicit

# Compare results
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/i7_12700k_numpy.json
```

**For Developers:**
```python
# Load hardware database
from graphs.hardware.database import get_database

db = get_database()
db.load_all()

# Query hardware
hw_spec = db.get('i7_12700k')
print(f"{hw_spec.model}: {hw_spec.theoretical_peaks['fp32']} GFLOPS")

# Auto-detect
from graphs.hardware.database.detector import HardwareDetector

detector = HardwareDetector()
results = detector.auto_detect(db)
if results['cpu_matches']:
    match = results['cpu_matches'][0]
    print(f"Detected: {match.matched_spec.model} ({match.confidence*100:.0f}%)")
```

### Statistics

- **Files Created**: 28 total
  - 3 core modules (schema, manager, detector)
  - 9 hardware JSON specs
  - 9 management tools
  - 7 documentation files
- **Lines of Code**: ~2,800 lines
- **Hardware Supported**: 9 specs (4 CPUs, 4 GPUs, 1 TPU)
- **Detection Patterns**: 42 total (improved via automation)
- **Platforms**: Windows, Linux, macOS

### Benefits

- ✅ Eliminates manual preset selection errors
- ✅ Auto-detection removes guesswork (2-command workflow)
- ✅ Maintainable (JSON specs, not Python code)
- ✅ Extensible (easy to add new hardware)
- ✅ Cross-platform (Windows, Linux, macOS)
- ✅ Production-ready (backward compatible, error handling)

### Future Enhancements (Phase 5)

Planned for next release:
1. Export calibration results directly to database
2. Historical calibration tracking
3. Multi-run calibration averaging
4. Thermal/power profiling integration
5. Complete removal of `--preset` flag
6. Web-based visualization

---

## [2025-11-13] - Architecture-Specific Energy Analysis CLI Tools (Phase 1A Complete)

### Added

- **CPU Energy Analysis Tool** (`cli/analyze_cpu_energy.py`, 363 lines):
  - Analyzes DNN models on 9 CPU configurations (Jetson Orin, Intel Xeon, AMD EPYC, Ampere)
  - Command-line interface: `--cpu`, `--model`, `--batch-size`, `--precision`, `--list-cpus`, `--list-models`
  - Complete pipeline: model loading → tracing → partitioning → mapping → energy analysis
  - **Hierarchical energy breakdown** showing 5 architectural categories:
    1. Instruction Pipeline (fetch, decode, dispatch)
    2. Register File Operations (reads, writes)
    3. Memory Hierarchy (L1, L2, L3, DRAM)
    4. ALU Operations
    5. Branch Prediction (misprediction tracking)
  - Graceful fallback for CPUs without architectural energy models
  - Example: `python cli/analyze_cpu_energy.py --cpu jetson_orin_agx_cpu --model mobilenet_v2`

- **Shared Model Factory** (`cli/model_factory.py`, 322 lines):
  - Unified model loading for all architecture-specific tools
  - Supports 30+ built-in torchvision models (ResNet, MobileNet, EfficientNet, ViT, ConvNeXt, etc.)
  - Custom PyTorch model support from file paths
  - Automatic Dynamo tracing and shape propagation
  - Fusion-based partitioning integration
  - Reusable across CPU/GPU/TPU/KPU analysis tools

- **Energy Breakdown Utilities** (`cli/energy_breakdown_utils.py`, 178 lines):
  - Reusable hierarchical breakdown printing functions
  - `print_cpu_hierarchical_breakdown()`: Detailed 5-category CPU energy breakdown
  - `aggregate_subgraph_events()`: Event aggregation across subgraphs
  - Clean separation of printing logic from analysis logic
  - Foundation for GPU/TPU/KPU breakdown functions (Phases 2-4)

### Changed

- **StoredProgramEnergyModel integration**:
  - CPU analysis tool now calls `architecture_energy_model.compute_architectural_energy()` after mapping
  - Aggregates ops and bytes across all subgraphs for model-level breakdown
  - Extracts architectural events from `extra_details` dict
  - Currently supported: Jetson Orin AGX CPU, Intel Xeon Platinum 8490H (Emerald Rapids)
  - Remaining CPUs (AMD EPYC, Ampere) show basic metrics with helpful message

### Technical Details

- **Implementation approach**: Aggregate events (Option C)
  - Manual call to architectural energy model after mapping completes
  - Doesn't modify core data structures (`GraphHardwareAllocation`)
  - Pattern replicable for GPU/TPU/KPU tools in Phases 2-4

- **Test results**:
  - Jetson Orin AGX + MobileNetV2: 36.4 mJ total, 6.8 mJ architectural overhead (18.7%)
  - Intel Xeon + ResNet18 (batch 4): 351.7 mJ total, 138.3 mJ overhead (39.3%)
  - AMD EPYC (no model): Graceful fallback verified

- **Key insights**:
  - Register file energy ≈ ALU energy (both ~0.6-0.8 pJ per op)
  - Memory hierarchy dominates: DRAM accounts for 50%+ of memory energy
  - Architectural overhead varies: 18.7% (Jetson) to 39.3% (Xeon)
  - Idle/leakage energy significant: 42-51% of total at 15W idle power

### Documentation

- `docs/sessions/2025-11-13_architecture_energy_cli_tools.md`: Complete session log with architecture analysis
- `docs/sessions/2025-11-13_phase1a_completion.md`: Phase 1A completion summary with test results

### Roadmap

- **Phase 1B** (Optional): JSON/CSV export for CPU tool (~1-2 hours)
- **Phase 1C** (Optional): Add StoredProgramEnergyModel to AMD EPYC and Ampere CPUs (~2-3 hours)
- **Phase 2**: GPU energy analysis tool (`cli/analyze_gpu_energy.py`) with DataParallelEnergyModel integration (~2-3 hours)
- **Phase 3**: TPU energy analysis tool with SystolicArrayEnergyModel (~2-3 hours)
- **Phase 4**: KPU energy analysis tool with DomainFlowEnergyModel (~2-3 hours)

---

## [2025-11-03] - CLI Script Naming Update

### Changed

- **Promoted unified framework CLI tools to default names**:
  - `analyze_comprehensive_v2.py` → `analyze_comprehensive.py` (Phase 4.2 unified framework)
  - `analyze_batch_v2.py` → `analyze_batch.py` (Phase 4.2 unified framework)
  - Legacy scripts renamed: `analyze_comprehensive.py` → `analyze_comprehensive_v1.py`, `analyze_batch.py` → `analyze_batch_v1.py`
  - **Rationale**: The unified framework versions are production-ready and should be the default user experience

- **Documentation updated**:
  - `CLAUDE.md` - Updated CLI examples to use new names and added power management examples
  - `cli/README.md` - Updated all tool references and added comprehensive "Power Management Analysis" section
  - Script help text updated to reference new names

---

## [2025-11-03] - Mapper-Integrated Pipeline & Enhanced Power Management Reporting

### Key Changes

- **Hardware Mapper Integration**: `UnifiedAnalyzer` now integrates hardware mapper allocation decisions into energy calculation
  - Per-unit static energy calculation based on actual `compute_units_allocated` from mapper
  - Power gating support: `power_gating_enabled` flag models turning off unused units
  - **Impact**: 48× more accurate utilization (36.5% actual vs 0.76% from threads), 61.7% idle energy savings with power gating on ResNet-18

- **Enhanced Energy Reporting**: `EnergyDescriptor` and `EnergyReport` extended with power management fields
  - Per-subgraph: `allocated_units`, `static_energy_allocated_j`, `static_energy_unallocated_j`, `power_gating_savings_j`
  - Aggregate: `average_allocated_units`, `total_power_gating_savings_j`
  - Reports now show "Power Management" section with allocated vs unallocated unit energy breakdown

- **API Changes**:
  - `AnalysisConfig`: New `run_hardware_mapping=True` and `power_gating_enabled=False` flags
  - `UnifiedAnalysisResult`: New `hardware_allocation` field
  - `EnergyAnalyzer`: New `power_gating_enabled` parameter, accepts `hardware_allocation` in `analyze()`
  - Backward compatible: Falls back to old method when no allocation info

- **Files Modified**:
  - `src/graphs/analysis/unified_analyzer.py` - Mapper integration
  - `src/graphs/analysis/energy.py` - Per-unit energy calculation and enhanced reporting
  - `src/graphs/reporting/report_generator.py` - Enhanced text/JSON reports with Power Management section
  - `src/graphs/ir/structures.py` - Compatibility properties for SubgraphDescriptor
  - `cli/analyze_comprehensive.py` - Added `--power-gating` and `--no-hardware-mapping` flags
  - `cli/analyze_batch.py` - Added `--power-gating` and `--no-hardware-mapping` flags
  - `validation/analysis/test_phase1_mapper_integration.py` (NEW)
  - `validation/analysis/test_power_management_reporting.py` (NEW)
  - `docs/designs/functional_energy_composition.md` (NEW) - Comprehensive design document

- **Documentation**:
  - `cli/README.md` - New "Power Management Analysis" section with 5 use cases:
    1. Low-utilization workloads
    2. Edge device power budgeting
    3. Datacenter TCO analysis
    4. Hardware comparison with accurate power
    5. **EDP (Energy-Delay Product) comparison** for Jetson-Orin-AGX vs KPU-T256
  - `CLAUDE.md` - Added power management CLI and Python API examples
  - See full technical details in `CHANGELOG.md`

---

## [2025-10-30] - Mermaid Visualization System (Phases 1-6 Complete)

### Added

- **Mermaid Visualization System (Production Ready)**
  - **Core Generator**: `MermaidGenerator` class in `src/graphs/visualization/mermaid_generator.py` (~750 lines)
    - 5 diagram types: FX graph, partitioned graph, hardware mapping, architecture comparison, bottleneck analysis
    - High-contrast color schemes meeting WCAG AA accessibility standards (4.5:1 minimum contrast)
    - Automatic label sanitization (replaces `[]` with `〈〉` to prevent Mermaid parse errors)
    - Invisible spacer nodes prevent subgraph labels from being covered by internal nodes
    - Scalable with `max_nodes` and `max_subgraphs` parameters

  - **Color Schemes**:
    - **Bottleneck**: Forest Green (#228B22) compute-bound, Crimson Red (#DC143C) memory-bound, Dark Orange (#FF8C00) balanced, Dim Gray (#696969) idle
    - **Utilization**: Dark Green (#006400) >80%, Forest Green 60-80%, Dark Orange 40-60%, Orange (#FFA500) 20-40%, Crimson <20%, Dim Gray 0%
    - **Operation Type**: Dodger Blue (#1E90FF) convolution, Forest Green activation, Goldenrod (#DAA520) normalization, Dark Cyan (#008B8B) element-wise, Medium Gray (#808080) default
    - All colors tested and validated for readability in light/dark themes

  - **ReportGenerator Integration** (`src/graphs/reporting/report_generator.py`):
    - New `include_diagrams` parameter for markdown reports
    - New `diagram_types` parameter to select specific diagram types
    - Automatic diagram embedding in markdown output
    - Seamless integration with existing analysis pipeline

  - **CLI Integration** (`cli/analyze_comprehensive.py`):
    - New `--include-diagrams` flag to enable Mermaid diagrams in markdown output
    - New `--diagram-types` flag to select specific diagram types (partitioned, bottleneck, hardware_mapping)
    - Auto-format detection from file extension
    - Example: `./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output report.md --include-diagrams`

  - **Test Files and Examples** (8 files in `docs/`):
    - `test_fx_graph.md`: Basic FX graph visualization
    - `test_partitioned_bottleneck.md`: Bottleneck-colored subgraphs
    - `test_partitioned_optype.md`: Operation-type colored subgraphs
    - `test_hardware_mapping_h100.md`: H100 GPU resource allocation
    - `test_hardware_mapping_tpu.md`: TPU-v4 MXU allocation
    - `test_architecture_comparison.md`: CPU vs GPU vs TPU side-by-side
    - `test_bottleneck_analysis.md`: Critical path visualization
    - `mermaid_visualization_demo.md`: Comprehensive demo with all diagram types

  - **Documentation** (5 comprehensive guides):
    - `docs/mermaid_visualization_design.md`: Complete design document with all phases
    - `docs/MERMAID_INTEGRATION_COMPLETE.md`: Integration summary and API reference
    - `docs/MERMAID_QUICK_START.md`: 30-second quick start with common use cases
    - `docs/COLOR_CONTRAST_IMPROVEMENTS.md`: Accessibility guide with WCAG compliance details
    - `docs/SUBGRAPH_LABEL_FIX.md`: Technical documentation of label visibility solution

### Fixed

- **Mermaid Parse Errors**:
  - Square brackets `[]` in node labels now replaced with Unicode angle brackets `〈〉`
  - Colons `:` in subgraph labels now replaced with tilde `~`
  - Parentheses `()` in percentage labels now replaced with tilde `~`
  - All special character conflicts resolved

- **Subgraph Label Visibility**:
  - Internal nodes were covering subgraph descriptor text
  - Solution: Added invisible spacer nodes (`fill:none,stroke:none`) at top of each subgraph
  - Spacer creates vertical separation, pushing content nodes below labels
  - All subgraph descriptors now fully readable

- **Color Contrast Issues**:
  - Original light pastel colors (Light Green #90EE90, Light Pink #FFB6C1, Light Yellow #FFFFE0) had poor contrast (1.8:1 to 2.3:1)
  - Replaced with high-contrast colors meeting WCAG AA standards
  - Average contrast improved from 2.3:1 to 5.8:1 (2.5× better)
  - 100% WCAG AA compliance achieved

### Implementation Notes

- **Diagram Generation**:
  - All diagrams use top-down (TD) layout for vertical scalability
  - Subgraphs automatically color-coded based on analysis results
  - Legends generated automatically for each color scheme
  - GitHub-native rendering (no external tools required)

- **Label Sanitization** (`_sanitize_label()` method):
  - Replaces `[` with `〈` (U+3008)
  - Replaces `]` with `〉` (U+3009)
  - Applied to all label generation points (6 locations in code)
  - Prevents Mermaid parser errors while maintaining readability

- **Spacer Node Pattern**:
  ```mermaid
  subgraph SG0["Subgraph 0<br/>Description"]
      SG0_spacer[ ]
      SG0_spacer --> SG0_exec[Content]
  end
  style SG0_spacer fill:none,stroke:none
  ```
  - Invisible spacer takes up vertical space
  - Content node positioned below subgraph label
  - No visual clutter added

- **Integration with Analysis Pipeline**:
  - `UnifiedAnalyzer` → `UnifiedAnalysisResult` → `ReportGenerator` → Markdown with diagrams
  - Diagrams generated on-demand during report generation
  - No changes required to existing analysis code
  - Backward compatible (diagrams optional)

- **Production Usage**:
  ```python
  from graphs.analysis.unified_analyzer import UnifiedAnalyzer
  from graphs.reporting import ReportGenerator

  analyzer = UnifiedAnalyzer()
  result = analyzer.analyze_model('resnet18', 'H100')

  generator = ReportGenerator()
  markdown = generator.generate_markdown_report(
      result,
      include_diagrams=True,
      diagram_types=['partitioned', 'bottleneck']
  )
  ```

- **CLI Usage**:
  ```bash
  # Basic report with diagrams
  ./cli/analyze_comprehensive.py \
      --model resnet18 \
      --hardware H100 \
      --output report.md \
      --include-diagrams

  # Select specific diagrams
  ./cli/analyze_comprehensive.py \
      --model mobilenet_v2 \
      --hardware Jetson-Orin-AGX \
      --output analysis.md \
      --include-diagrams \
      --diagram-types partitioned bottleneck
  ```

### Key Benefits

- **GitHub-Native**: Diagrams render automatically in GitHub markdown (repos, PRs, issues, wikis)
- **Version Control Friendly**: Text-based Mermaid syntax, diffs work properly
- **No External Dependencies**: No Graphviz, image generation, or external services required
- **Accessible**: WCAG AA compliant colors, readable in light/dark themes
- **Scalable**: Handles graphs up to 50+ subgraphs with truncation for larger models
- **Integrated**: Works seamlessly with existing `UnifiedAnalyzer` and `ReportGenerator`
- **Customizable**: Multiple diagram types, color schemes, and display options

### Phases Completed

- ✅ **Phase 1**: Core infrastructure (FX graph, partitioned graph visualization)
- ✅ **Phase 2**: Styling & color coding (3 color schemes, legends)
- ✅ **Phase 3**: Hardware mapping visualization (resource allocation, idle highlighting)
- ✅ **Phase 4**: Architecture comparison (2-3 architectures side-by-side)
- ✅ **Phase 5**: Integration & CLI (ReportGenerator, CLI flags, markdown reports)
- ✅ **Phase 6**: Documentation & polish (5 guides, 8 examples, quick start)

### Status

**Production Ready** - All features implemented, tested with real models (ResNet-18, ResNet-50), documented, and ready for production use.

---

## [2025-10-28] - Phase 3: Energy Estimator Implementation Complete

### Added

- **Energy Estimator (Phase 3.2 Complete)**
  - **Core Algorithm**: `EnergyAnalyzer` class with three-component energy model
    - Compute energy = FLOPs × energy_per_flop
    - Memory energy = bytes_transferred × energy_per_byte
    - Static energy = idle_power × latency (leakage, always-on circuits)
  - **Precision-Aware Energy**: Scaling factors for FP16 (50% compute energy), INT8 (25% compute energy)
  - **TDP Estimation**: Hardware power modeling from thermal profiles or peak operation estimation
  - **Efficiency Metrics**: Energy efficiency, utilization, wasted energy analysis
  - **Optimization Detection**: Identifies opportunities for latency reduction, utilization improvement, quantization
  - **Data Structures**: `EnergyDescriptor`, `EnergyReport`
  - **Files**: `src/graphs/analysis/energy.py` (~450 lines)

- **Integration Tests**: `tests/analysis/test_energy_analyzer.py` (8 tests, all passing)
  - Simple model energy analysis
  - Energy breakdown (compute, memory, static)
  - Energy efficiency calculation
  - GPU vs CPU energy comparison
  - Precision scaling (FP32 vs FP16)
  - Top energy consumers identification
  - Optimization opportunities
  - ResNet-18 validation

- **End-to-End Demo**: `examples/demo_energy_analyzer.py`
  - Analyzes ResNet-18 and MobileNet-V2 on GPU and Edge device
  - Hardware comparison (GPU-A100 vs Edge-Jetson)
  - Precision comparison (FP32 vs FP16)
  - ASCII-art energy breakdown visualization
  - Optimization strategies and key insights

- **Exports**: `EnergyAnalyzer`, `EnergyDescriptor`, `EnergyReport` added to `src/graphs/analysis/__init__.py`

### Implementation Notes

- **Energy Model Components**:
  - **Compute Energy**: Proportional to FLOPs executed
    - Energy = FLOPs × energy_per_flop
    - Precision scaling: FP16 = 0.5×, INT8 = 0.25× of FP32
  - **Memory Energy**: Proportional to data movement
    - Energy = (input_bytes + output_bytes + weight_bytes) × energy_per_byte
    - Memory transfers dominate for memory-bound operations
  - **Static Energy**: Leakage and always-on circuits
    - Energy = idle_power × latency
    - GPUs: ~30% of TDP at idle, CPUs: ~10% of TDP at idle
    - Dominates for small models or long latency

- **TDP (Thermal Design Power) Estimation**:
  - Preferred: Use hardware thermal_operating_points if available
  - Fallback: Estimate from peak_ops_per_sec × energy_per_flop
  - Idle power = TDP × IDLE_POWER_FRACTION (0.3 for GPU, 0.1 for CPU)

- **Energy Efficiency**:
  - Efficiency = dynamic_energy / total_energy
  - Dynamic energy = compute + memory
  - Low efficiency indicates high static power (leakage)

- **Key Observations from Demo**:
  - **ResNet-18 on GPU-A100**: 205 mJ total, 64% static energy, 14.8% efficiency
    - Low efficiency due to short latency (0.56ms) → static power dominates
    - Optimization: Increase batch size to amortize static energy
  - **MobileNet-V2 on GPU-A100**: 219 mJ total, 93% static energy, 5.5% efficiency
    - Extremely low efficiency → designed for mobile, not datacenter
    - More subgraphs (151) → more overhead, longer latency (0.87ms)
  - **ResNet-18 on Edge-Jetson**: 667 mJ total, 44% static energy, 21.1% efficiency
    - Higher total energy but better efficiency (lower idle power)
    - Higher energy per operation (100 pJ/FLOP vs 20 pJ/FLOP)
    - Edge devices: higher energy/op, lower idle power
    - Datacenter GPUs: lower energy/op, higher idle power
  - **Precision Comparison**: FP16 can use more total energy than FP32 in some cases
    - FP16 has 16× higher peak_ops_per_sec (tensor cores)
    - Higher peak performance → higher estimated TDP
    - Trade-off: faster execution vs higher power draw
    - Best for throughput scenarios where high utilization amortizes static power

---

## [2025-10-28] - Phase 3: Roofline Analyzer Implementation Complete

### Added

- **Roofline Analyzer (Phase 3.1 Complete)**
  - **Core Algorithm**: `RooflineAnalyzer` class with roofline model
    - Compute time = FLOPs / peak_FLOPS
    - Memory time = bytes / peak_bandwidth
    - Actual latency = max(compute_time, memory_time) + overhead
    - Bottleneck classification (compute-bound vs memory-bound vs balanced)
  - **Arithmetic Intensity Analysis**: AI = FLOPs / bytes, AI_breakpoint = peak_FLOPS / peak_bandwidth
  - **Hardware-Aware**: Works with any HardwareResourceModel, precision-aware
  - **Utilization Metrics**: FLOP utilization and bandwidth utilization
  - **Data Structures**: `LatencyDescriptor`, `RooflinePoint`, `RooflineReport`
  - **Files**: `src/graphs/analysis/roofline.py` (547 lines)

- **Integration Tests**: `tests/analysis/test_roofline_analyzer.py` (7 tests, all passing)
  - Simple model analysis
  - Compute-bound on high-FLOP hardware
  - Memory-bound on high-bandwidth hardware
  - Arithmetic intensity breakpoint calculation
  - Roofline points generation (for visualization)
  - ResNet-18 validation
  - Latency descriptor formatting

- **End-to-End Demo**: `examples/demo_roofline_analyzer.py`
  - Analyzes ResNet-18 and MobileNet-V2 on GPU and CPU
  - ASCII-art roofline plot visualization
  - Hardware comparison (GPU vs CPU)
  - Bottleneck distribution analysis
  - Key insights and optimization strategies

- **Exports**: `RooflineAnalyzer`, `LatencyDescriptor`, `RooflinePoint`, `RooflineReport` added to `src/graphs/analysis/__init__.py`

### Implementation Notes

- **Roofline Model Formula**:
  - Latency = max(FLOPs/peak_FLOPS, bytes/peak_bandwidth) + overhead
  - If compute_time > memory_time: compute-bound
  - If memory_time > compute_time: memory-bound
  - Otherwise: balanced

- **Arithmetic Intensity (AI)**:
  - AI = FLOPs / total_bytes (inputs + outputs + weights)
  - AI_breakpoint = peak_FLOPS / peak_bandwidth
  - Operations with AI < breakpoint are memory-bound
  - Operations with AI > breakpoint are compute-bound

- **Key Observations from Demo**:
  - ResNet-18 on GPU-A100: 72% memory-bound ops, 0.56ms total
  - MobileNet-V2 on GPU-A100: 82% memory-bound ops, 0.87ms total (more overhead)
  - ResNet-18 on CPU-Xeon: 75% memory-bound ops, 1.18ms total (slower)
  - GPU kernel launch overhead dominates for small models (60-86% of latency)
  - Conv layers with high channels are compute-bound (high AI)
  - Element-wise ops (ReLU, Add) are always memory-bound (low AI)

---

## [2025-10-28] - Phase 3: Memory Estimator Implementation Complete

### Added

- **Memory Estimator (Phase 3.3 Complete)**
  - **Core Algorithm**: `MemoryEstimator` class with execution simulation
    - Topological sort for correct execution order (Kahn's algorithm)
    - Step-by-step memory allocation/deallocation tracking
    - Peak memory detection across entire execution
    - Dead tensor analysis for automatic freeing
  - **Workspace Estimation**: Conv2d im2col buffers, MatMul/Linear transpose buffers, Attention QKV projections
  - **Optimization Detection**:
    - Activation checkpointing (>50% activations)
    - Weight quantization (>30% weights)
    - In-place operations (ReLU, Dropout)
  - **Hardware Fit Analysis**: L2 cache, shared memory, device memory constraints
  - **Data Structures**: `MemoryTimelineEntry`, `MemoryDescriptor`, `MemoryReport`
  - **Files**: `src/graphs/analysis/memory.py` (807 lines total)

- **Integration Tests**: `tests/analysis/test_memory_estimator.py` (8 tests, all passing)
  - Simple sequential models
  - Tensor lifetime tracking
  - Workspace allocation for Conv2d
  - Models with branching (fork/join)
  - Optimization detection
  - ResNet-18 validation
  - Hardware fit analysis
  - Timeline accuracy validation

- **End-to-End Demo**: `examples/demo_memory_estimator.py`
  - Analyzes ResNet-18, MobileNet-V2, ResNet-50
  - Text-based timeline visualization
  - Model comparison table
  - Optimization opportunities summary

- **Exports**: `MemoryEstimator` added to `src/graphs/analysis/__init__.py`

### Fixed

- **Double-Counting Bug**: Workspace was counted in both `activation_memory_bytes` and `workspace_memory_bytes`
  - Fixed by excluding workspace from activation calculation
  - Now: `activation_memory = all non-weight, non-workspace tensors`
- **Dependency Graph Construction**: Built from `SubgraphDescriptor.depends_on` field instead of expecting pre-built graph
- **OperationType Enum**: Removed reference to non-existent `SOFTMAX`, using existing types only

### Implementation Notes

- **Memory Simulation Algorithm**:
  1. Allocate persistent weights upfront (live throughout execution)
  2. For each subgraph in execution order:
     - Allocate output tensor
     - Allocate workspace (temporary)
     - Record timeline entry (peak memory snapshot)
     - Free workspace immediately
     - Free dead input tensors (no longer needed by remaining ops)
  3. Track peak memory across all steps
  4. Analyze for optimization opportunities

- **Key Insight**: Peak memory ≠ sum of all tensors (tensors are freed over time)

- **Timeline Entry Contents**:
  - Total memory (all live tensors including weights)
  - Activation memory (non-weight, non-workspace tensors)
  - Workspace memory (temporary buffers)
  - List of live tensors, allocated tensors, freed tensors

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
