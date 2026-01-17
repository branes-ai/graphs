# Phase 4: Integration & Unified Analysis

**Status**: Planning
**Date**: 2025-10-28
**Goal**: Integrate all Phase 3 analysis capabilities into CLI tools and create comprehensive end-to-end workflows

---

## Overview

Phase 3 delivered four powerful analysis components (Concurrency, Roofline, Energy, Memory), but they exist primarily as standalone Python modules with individual demos. Phase 4 will integrate these capabilities into the CLI tools, create unified analysis workflows, and provide production-quality reporting.

**Key Principle**: Make advanced analysis accessible to users through well-designed CLI tools and comprehensive reports.

---

## Phase 3 Recap: What We Have

### Analysis Components (Complete ✅)

1. **Concurrency Analysis** (`analysis/concurrency.py`)
   - Multi-level parallelism (thread, warp, block)
   - Wave quantization
   - Thread group analysis

2. **Roofline Model** (`analysis/roofline.py`)
   - Arithmetic intensity (AI = FLOPs / bytes)
   - Bottleneck classification (compute vs memory-bound)
   - Hardware-aware latency estimation
   - Utilization metrics

3. **Energy Estimator** (`analysis/energy.py`)
   - Three-component model (compute, memory, static)
   - Precision-aware energy scaling
   - TDP estimation and idle power modeling
   - Efficiency metrics and optimization detection

4. **Memory Estimator** (`analysis/memory.py`)
   - Execution simulation with topological sort
   - Peak memory detection and timeline
   - Workspace estimation
   - Optimization detection (checkpointing, quantization)

### What's Missing

- ❌ CLI integration (only demos exist)
- ❌ Unified analysis workflows (components work independently)
- ❌ Production-quality reporting (JSON, CSV, Markdown)
- ❌ Multi-objective optimization (Pareto frontiers)
- ❌ Advanced visualizations (roofline plots, energy charts, memory timelines)
- ❌ Batch analysis tools (sweep models/hardware/precisions)

---

## Phase 4 Components

### 4.1 CLI Integration (Priority: HIGH)

**Goal**: Bring all Phase 3 analysis into CLI tools

#### 4.1.1 Enhanced `analyze_graph_mapping.py`

**New Flags**:
```bash
--analysis {basic,full,energy,roofline,memory,all}
    basic: Current behavior (allocation only)
    full: All analyses (roofline + energy + memory)
    energy: Energy analysis only
    roofline: Roofline analysis only
    memory: Memory analysis only
    all: Everything including concurrency

--show-timeline          # Show memory timeline
--show-roofline          # ASCII roofline plot
--show-energy-breakdown  # Energy breakdown visualization
--top-consumers N        # Show top N energy/memory consumers
```

**New Output Sections**:
1. **Roofline Analysis**
   - Bottleneck distribution (% compute-bound, % memory-bound)
   - Arithmetic intensity statistics
   - Utilization metrics (FLOP utilization, bandwidth utilization)
   - ASCII roofline plot (optional)

2. **Energy Analysis**
   - Energy breakdown (compute, memory, static)
   - Average power and peak power
   - Energy efficiency
   - Top energy consumers
   - Optimization opportunities

3. **Memory Analysis**
   - Peak memory usage
   - Memory breakdown (weights, activations, workspace)
   - Memory timeline (optional)
   - Hardware fit analysis (L2, shared mem, device mem)
   - Optimization opportunities

**Implementation**:
- File: `cli/analyze_graph_mapping.py`
- Add analysis flags to argparse
- Create analysis workflow function
- Integrate roofline, energy, memory analyzers
- Add output formatting functions
- Estimated effort: 300-400 lines of new code

---

#### 4.1.2 New Tool: `analyze_comprehensive.py`

**Purpose**: All-in-one analysis tool for deep dives

**Features**:
```bash
# Comprehensive single-model analysis
./cli/analyze_comprehensive.py --model resnet18 --hardware H100

# Multi-hardware comparison
./cli/analyze_comprehensive.py --model resnet18 \
    --hardware H100 Jetson-Orin A100 \
    --compare

# Multi-precision comparison
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
    --precision fp32 fp16 int8 \
    --compare-precision

# Generate report
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
    --output-format {text,json,markdown,csv} \
    --output-file report.md
```

**Output Sections**:
1. **Model Summary**
   - Total operations, parameters, layers
   - Input/output shapes
   - Partitioning summary

2. **Hardware Mapping**
   - Resource allocation per subgraph
   - Utilization statistics
   - Bottleneck analysis

3. **Roofline Analysis**
   - Bottleneck distribution
   - AI statistics
   - Roofline plot (ASCII or image)

4. **Energy Analysis**
   - Energy breakdown
   - Power statistics
   - Efficiency metrics
   - Top consumers

5. **Memory Analysis**
   - Peak memory
   - Memory breakdown
   - Timeline visualization
   - Hardware fit

6. **Optimization Opportunities**
   - Ranked by potential savings
   - Specific recommendations
   - Trade-off analysis

**Implementation**:
- File: `cli/analyze_comprehensive.py` (new)
- Estimated effort: 400-500 lines

---

#### 4.1.3 New Tool: `analyze_batch.py`

**Purpose**: Batch analysis for sweeps and experiments

**Features**:
```bash
# Model sweep on single hardware
./cli/analyze_batch.py \
    --models resnet18 resnet50 mobilenet_v2 efficientnet_b0 \
    --hardware H100 \
    --output sweep_results.csv

# Hardware sweep for single model
./cli/analyze_batch.py \
    --model resnet18 \
    --hardware H100 A100 V100 Jetson-Orin TPU-v4 \
    --output hardware_comparison.csv

# Precision sweep
./cli/analyze_batch.py \
    --model resnet18 \
    --hardware H100 \
    --precision fp32 fp16 int8 \
    --output precision_comparison.csv

# Full sweep (models × hardware × precision)
./cli/analyze_batch.py \
    --models resnet18 mobilenet_v2 \
    --hardware H100 Jetson-Orin \
    --precision fp32 fp16 \
    --output full_sweep.csv
```

**Output Formats**:
- CSV: For spreadsheet analysis
- JSON: For programmatic processing
- Markdown: For documentation

**Columns**:
- Model, Hardware, Precision, Batch Size
- Total FLOPs, Total Bytes
- Latency (ms), Throughput (inf/s)
- Energy (mJ), Average Power (W)
- Peak Memory (MB)
- Utilization (%), Efficiency (%)
- Bottleneck Distribution

**Implementation**:
- File: `cli/analyze_batch.py` (new)
- Estimated effort: 300-400 lines

---

### 4.2 Unified Analysis Workflows (Priority: HIGH)

**Goal**: Create comprehensive analysis pipelines that combine all analyzers

#### 4.2.1 Unified Analyzer Module

**File**: `src/graphs/analysis/unified.py`

**Purpose**: Single entry point for all analyses

**API**:
```python
from graphs.analysis.unified import UnifiedAnalyzer

analyzer = UnifiedAnalyzer(
    hardware=gpu_hardware,
    precision=Precision.FP32,
    enable_roofline=True,
    enable_energy=True,
    enable_memory=True,
    enable_concurrency=True
)

# Analyze model
report = analyzer.analyze(
    fx_graph=traced_model,
    partition_report=partition_report
)

# Report contains:
# - partition_report: PartitionReport
# - roofline_report: RooflineReport (optional)
# - energy_report: EnergyReport (optional)
# - memory_report: MemoryReport (optional)
# - concurrency_report: ConcurrencyReport (optional)
```

**Features**:
- Automatic data flow between analyzers
- Roofline latencies feed into energy analyzer
- Memory timeline integrated with energy static power
- Single comprehensive report object

**Implementation**:
- File: `src/graphs/analysis/unified.py` (new)
- Estimated effort: 200-300 lines

---

#### 4.2.2 Report Generation

**File**: `src/graphs/analysis/report.py`

**Purpose**: Generate comprehensive reports in multiple formats

**API**:
```python
from graphs.analysis.report import ReportGenerator

generator = ReportGenerator(unified_report)

# Generate Markdown report
markdown = generator.to_markdown(
    include_roofline_plot=True,
    include_energy_breakdown=True,
    include_memory_timeline=True,
    include_optimization_recommendations=True
)

# Generate JSON report
json_str = generator.to_json(pretty=True)

# Generate CSV summary
csv_str = generator.to_csv()
```

**Report Sections** (Markdown):
1. **Executive Summary**
   - Key metrics at a glance
   - Performance grade (A-F)
   - Top optimization opportunity

2. **Model Information**
   - Architecture details
   - Total operations and parameters
   - Partitioning summary

3. **Performance Analysis**
   - Latency breakdown
   - Throughput estimation
   - Bottleneck identification

4. **Roofline Analysis** (if enabled)
   - Bottleneck distribution chart
   - Arithmetic intensity statistics
   - Roofline plot
   - Utilization analysis

5. **Energy Analysis** (if enabled)
   - Energy breakdown chart
   - Power statistics
   - Efficiency metrics
   - Top energy consumers
   - Optimization opportunities

6. **Memory Analysis** (if enabled)
   - Peak memory usage
   - Memory breakdown chart
   - Timeline visualization
   - Hardware fit analysis
   - Optimization opportunities

7. **Recommendations**
   - Prioritized optimization list
   - Expected savings per optimization
   - Implementation complexity

**Implementation**:
- File: `src/graphs/analysis/report.py` (new)
- Estimated effort: 400-500 lines

---

### 4.3 Enhanced Visualizations (Priority: MEDIUM)

**Goal**: Create better visualizations for analysis results

#### 4.3.1 Roofline Plotter

**File**: `src/graphs/analysis/visualization/roofline_plotter.py`

**Features**:
- ASCII art roofline plot (terminal-friendly)
- matplotlib roofline plot (publication-quality)
- Configurable axes, labels, markers
- Support for multiple models on same plot

**ASCII Example**:
```
Roofline Model - GPU-A100 (FP32)
Peak Performance: 19.5 TFLOPS

  FLOP/s
  |
20T|                                  ████████████
  |                             ███████
  |                        ██████
  |                   █████
10T|              ████
  |         █████
  |    █████
  | ███
5T|██
  +---+---+---+---+---+---+---+---+---+---+---+---+--- AI (FLOP/byte)
  0   1   2   3   4   5   6   7   8   9  10  11  12

  ● = Operations (colored by bottleneck type)
  Red    = Memory-bound (<AI_breakpoint)
  Green  = Compute-bound (>AI_breakpoint)
  Yellow = Balanced
```

**Implementation**:
- Estimated effort: 200-300 lines

---

#### 4.3.2 Energy Breakdown Visualizer

**File**: `src/graphs/analysis/visualization/energy_plotter.py`

**Features**:
- Stacked bar chart (compute/memory/static)
- Pie chart (energy breakdown)
- Waterfall chart (subgraph contributions)
- Top consumers bar chart

**ASCII Example** (already in demo):
```
Compute (35.5%): █████████████████
Memory  (0.6%):
Static  (63.9%): ███████████████████████████████
```

**Implementation**:
- Estimated effort: 200-250 lines

---

#### 4.3.3 Memory Timeline Visualizer

**File**: `src/graphs/analysis/visualization/memory_plotter.py`

**Features**:
- ASCII timeline (terminal-friendly)
- matplotlib timeline (publication-quality)
- Stacked area chart (weights/activations/workspace)
- Annotated with operation names

**ASCII Example**:
```
Memory Timeline - ResNet-18

Peak: 45.2 MB
  |
50|                 ╔═══╗
  |            ╔════╝   ╚════╗
40|       ╔════╝            ╚═══╗
  |  ╔════╝                    ╚════╗
30|══╝                              ╚═══
  |
  +---+---+---+---+---+---+---+---+---+--- Execution Steps
     conv1 bn1 relu maxpool layer1 layer2 layer3 layer4

  Weights:     ████ (11.7 MB, persistent)
  Activations: ████ (33.5 MB, peak)
  Workspace:   ░░░░ (varies)
```

**Implementation**:
- Estimated effort: 250-300 lines

---

### 4.4 Multi-Objective Optimization (Priority: LOW)

**Goal**: Pareto frontier analysis and trade-off exploration

#### 4.4.1 Pareto Analyzer

**File**: `src/graphs/analysis/pareto.py`

**Purpose**: Multi-objective optimization analysis

**Objectives**:
- Latency (minimize)
- Energy (minimize)
- Memory (minimize)
- Accuracy (maximize, if available)

**Use Cases**:
1. **Hardware Selection**: Which hardware for given model?
2. **Precision Selection**: FP32 vs FP16 vs INT8?
3. **Batch Size Selection**: Throughput vs latency vs memory?

**API**:
```python
from graphs.analysis.pareto import ParetoAnalyzer

analyzer = ParetoAnalyzer()

# Add configurations
for hw in [h100, a100, jetson]:
    for prec in [FP32, FP16, INT8]:
        report = analyze(model, hw, prec)
        analyzer.add_config(
            name=f"{hw.name}-{prec.value}",
            latency=report.latency,
            energy=report.energy,
            memory=report.memory
        )

# Find Pareto frontier
frontier = analyzer.compute_pareto_frontier(
    objectives=['latency', 'energy'],
    minimize=True
)

# Print frontier
for config in frontier:
    print(f"{config.name}: {config.latency:.2f}ms, {config.energy:.2f}mJ")

# Visualize trade-offs
analyzer.plot_pareto_frontier(
    x_axis='latency',
    y_axis='energy',
    output='pareto.png'
)
```

**Implementation**:
- Estimated effort: 300-400 lines

---

#### 4.4.2 Trade-Off Explorer CLI

**File**: `cli/explore_tradeoffs.py`

**Purpose**: Interactive trade-off exploration

**Example**:
```bash
# Explore hardware trade-offs
./cli/explore_tradeoffs.py --model resnet18 \
    --hardware H100 A100 V100 Jetson-Orin \
    --objectives latency energy \
    --plot pareto.png

# Explore precision trade-offs
./cli/explore_tradeoffs.py --model resnet18 --hardware H100 \
    --precision fp32 fp16 int8 \
    --objectives latency energy accuracy \
    --plot precision_tradeoff.png

# Explore batch size trade-offs
./cli/explore_tradeoffs.py --model resnet18 --hardware H100 \
    --batch-size 1 2 4 8 16 32 64 \
    --objectives latency throughput energy memory \
    --plot batch_tradeoff.png
```

**Output**:
- Pareto frontier identification
- Dominated configurations (can be eliminated)
- Trade-off recommendations
- Visualization (scatter plot with Pareto frontier highlighted)

**Implementation**:
- Estimated effort: 250-300 lines

---

## Implementation Plan

### Phase 4.1: CLI Integration (Week 1)

**Priority**: HIGH
**Goal**: Make Phase 3 analysis accessible via CLI

**Tasks**:
1. ✅ Plan creation (this document)
2. ⬜ Enhance `analyze_graph_mapping.py` with analysis flags
3. ⬜ Create `analyze_comprehensive.py`
4. ⬜ Create `analyze_batch.py`
5. ⬜ Add tests for new CLI tools
6. ⬜ Update CLI documentation

**Deliverables**:
- Enhanced `analyze_graph_mapping.py` (300 lines added)
- New `analyze_comprehensive.py` (~500 lines)
- New `analyze_batch.py` (~400 lines)
- Integration tests (3-4 tests per tool)
- Updated `cli/README.md`

---

### Phase 4.2: Unified Workflows (Week 2)

**Priority**: HIGH
**Goal**: Streamline analysis with unified API

**Tasks**:
1. ⬜ Create `UnifiedAnalyzer` class
2. ⬜ Implement automatic data flow between analyzers
3. ⬜ Create `ReportGenerator` class
4. ⬜ Implement Markdown report generation
5. ⬜ Implement JSON report generation
6. ⬜ Implement CSV report generation
7. ⬜ Add comprehensive tests
8. ⬜ Create end-to-end demo

**Deliverables**:
- `src/graphs/analysis/unified.py` (~300 lines)
- `src/graphs/analysis/report.py` (~500 lines)
- Integration tests (5-6 tests)
- Demo: `examples/demo_unified_analysis.py`

---

### Phase 4.3: Enhanced Visualizations (Week 3)

**Priority**: MEDIUM
**Goal**: Better visualization of analysis results

**Tasks**:
1. ⬜ Create `RooflinePlotter` (ASCII + matplotlib)
2. ⬜ Create `EnergyPlotter` (charts and breakdowns)
3. ⬜ Create `MemoryPlotter` (timeline visualization)
4. ⬜ Integrate plotters into report generation
5. ⬜ Add tests for visualization functions
6. ⬜ Create visualization examples

**Deliverables**:
- `src/graphs/analysis/visualization/roofline_plotter.py` (~300 lines)
- `src/graphs/analysis/visualization/energy_plotter.py` (~250 lines)
- `src/graphs/analysis/visualization/memory_plotter.py` (~300 lines)
- Visualization tests (3-4 tests)
- Demo: `examples/demo_visualizations.py`

---

### Phase 4.4: Multi-Objective Optimization (Week 4)

**Priority**: LOW
**Goal**: Pareto frontier analysis and trade-off exploration

**Tasks**:
1. ⬜ Create `ParetoAnalyzer` class
2. ⬜ Implement Pareto frontier computation
3. ⬜ Create `explore_tradeoffs.py` CLI tool
4. ⬜ Implement trade-off visualizations
5. ⬜ Add comprehensive tests
6. ⬜ Create end-to-end demo

**Deliverables**:
- `src/graphs/analysis/pareto.py` (~400 lines)
- `cli/explore_tradeoffs.py` (~300 lines)
- Tests (4-5 tests)
- Demo: `examples/demo_pareto_analysis.py`

---

## Success Criteria

### Phase 4.1: CLI Integration
- [ ] All Phase 3 analyzers accessible via CLI flags
- [ ] `analyze_comprehensive.py` generates full reports
- [ ] `analyze_batch.py` enables sweep experiments
- [ ] All CLI tools have comprehensive tests
- [ ] Documentation updated

### Phase 4.2: Unified Workflows
- [ ] `UnifiedAnalyzer` integrates all Phase 3 analyzers
- [ ] Report generation supports text/JSON/markdown/CSV
- [ ] Data flows automatically between analyzers
- [ ] End-to-end demo shows complete workflow
- [ ] All components have tests

### Phase 4.3: Enhanced Visualizations
- [ ] Roofline plots (ASCII and matplotlib)
- [ ] Energy breakdown charts
- [ ] Memory timeline visualizations
- [ ] All visualizations integrated into reports
- [ ] Visualization examples provided

### Phase 4.4: Multi-Objective Optimization
- [ ] Pareto frontier computation
- [ ] Trade-off exploration CLI tool
- [ ] Hardware/precision/batch selection guidance
- [ ] Visualization of trade-offs
- [ ] Tests and demo complete

---

## Estimated Effort

| Component | Lines of Code | Tests | Effort |
|-----------|---------------|-------|--------|
| 4.1 CLI Integration | ~1,200 | 10-12 | 2-3 days |
| 4.2 Unified Workflows | ~800 | 5-6 | 1-2 days |
| 4.3 Enhanced Visualizations | ~850 | 3-4 | 1-2 days |
| 4.4 Multi-Objective Optimization | ~700 | 4-5 | 1-2 days |
| **Total** | **~3,550** | **22-27** | **5-9 days** |

---

## Dependencies

### External Libraries
- **matplotlib**: For publication-quality plots (optional)
- **pandas**: For CSV generation and data manipulation
- **numpy**: For Pareto frontier computation

### Internal Dependencies
- Phase 3 analyzers (all complete ✅)
- Existing CLI tools structure
- Hardware mappers and resource models

---

## Open Questions

1. **Visualization Library**: Use matplotlib (heavy) or keep ASCII-only (lightweight)?
   - Recommendation: Support both (ASCII default, matplotlib optional)

2. **Report Format Priority**: Which format is most important?
   - Recommendation: Text first, then JSON, then Markdown, CSV last

3. **Accuracy Metrics**: How to get model accuracy for Pareto analysis?
   - Recommendation: Make accuracy optional, user-provided via CLI flag

4. **Batch Analysis Parallelization**: Should we parallelize batch sweeps?
   - Recommendation: Yes, use multiprocessing for independent analyses

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Start with Phase 4.1** (CLI Integration) - highest priority
3. **Iterate based on feedback** - adjust priorities as needed
4. **Create session logs** for each phase component
5. **Update CHANGELOG** as components complete

---

## Notes

This plan builds on the solid foundation of Phase 3 to make advanced analysis accessible and practical. The focus is on **usability** and **integration** rather than new analysis capabilities.

**Key Principle**: Every Phase 3 analyzer should be:
- Accessible via CLI
- Well-documented
- Easy to use
- Production-ready

By the end of Phase 4, users will have a complete toolkit for analyzing neural network workloads with professional-grade reports and visualizations.
