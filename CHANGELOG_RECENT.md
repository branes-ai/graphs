# Recent Changes (Last 3 Months)

**Purpose**: Quick context for AI assistants resuming work. Full history in `CHANGELOG.md`.

**Last Updated**: 2025-11-03

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
