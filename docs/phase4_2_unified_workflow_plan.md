# Phase 4.2: Unified Workflows - Design Plan

**Date:** 2025-10-28
**Status:** Planning
**Prerequisites:** Phase 4.1 (CLI Integration) - ✅ Complete

---

## Overview

Phase 4.2 creates a unified analysis and reporting framework that consolidates the Phase 3 analyzers into a single, cohesive workflow. This eliminates code duplication across CLI tools and provides consistent analysis results.

### Current State (After Phase 4.1)

**Problems to Solve:**
1. **Code Duplication**: Each CLI tool (analyze_comprehensive.py, analyze_batch.py, analyze_graph_mapping.py) contains similar analysis orchestration code (~200 lines duplicated)
2. **Inconsistency**: Different tools may produce slightly different results due to implementation variations
3. **Maintenance Burden**: Bugs or improvements must be applied to multiple files
4. **Limited Reusability**: Analysis logic is embedded in CLI tools, not accessible to programmatic users
5. **Report Format Proliferation**: Each tool implements its own JSON/CSV/markdown formatting

**What We'll Build:**
1. **UnifiedAnalyzer**: Single orchestrator for all Phase 3 analyzers
2. **ReportGenerator**: Flexible reporting engine supporting multiple formats
3. **Refactored CLI Tools**: Leverage unified components, reducing code by ~60%

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Tools Layer                         │
│  analyze_comprehensive.py | analyze_batch.py | others...     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Uses
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Unified Workflows                         │
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │  UnifiedAnalyzer     │    │  ReportGenerator     │       │
│  │                      │    │                      │       │
│  │  • Model tracing     │    │  • Text reports      │       │
│  │  • Partitioning      │    │  • JSON export       │       │
│  │  • Analyzer orchest. │    │  • CSV export        │       │
│  │  • Data flow coord.  │    │  • Markdown reports  │       │
│  │  • Result aggregation│    │  • HTML reports      │       │
│  └──────────┬───────────┘    │  • Exec summaries    │       │
│             │                │  • Comparisons       │       │
│             │                └──────────────────────┘       │
└─────────────┼──────────────────────────────────────────────┘
              │
              │ Uses
              ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 3 Analyzers (Existing)                    │
│  RooflineAnalyzer | EnergyAnalyzer | MemoryEstimator |       │
│  ConcurrencyAnalyzer                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 4.2.1: UnifiedAnalyzer

### Purpose
Single entry point for comprehensive model analysis, orchestrating all Phase 3 analyzers with correct data flow and dependencies.

### Location
`src/graphs/analysis/unified_analyzer.py`

### Core Data Structures

```python
@dataclass
class UnifiedAnalysisResult:
    """
    Complete analysis results from all Phase 3 analyzers.

    This is the single source of truth for analysis results,
    ensuring consistency across all tools and use cases.
    """
    # Model metadata
    model_name: str
    display_name: str
    batch_size: int
    precision: Precision

    # Hardware metadata
    hardware_name: str
    hardware_display_name: str
    hardware: HardwareResourceModel

    # Graph structures
    fx_graph: torch.fx.GraphModule
    partition_report: PartitionReport

    # Phase 3 analysis results
    roofline_report: Optional[RooflineReport] = None
    energy_report: Optional[EnergyReport] = None
    memory_report: Optional[MemoryReport] = None
    concurrency_report: Optional[ConcurrencyReport] = None

    # Derived metrics (computed from above)
    total_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    total_energy_mj: float = 0.0
    energy_per_inference_mj: float = 0.0
    peak_memory_mb: float = 0.0
    average_utilization_pct: float = 0.0

    # Timestamps
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary dict for quick overview"""
        ...

    def validate(self) -> List[str]:
        """Validate consistency between reports, return list of warnings"""
        ...


@dataclass
class AnalysisConfig:
    """
    Configuration for unified analysis.

    Allows fine-grained control over which analyses to run,
    useful for performance optimization.
    """
    # Which analyses to run
    run_roofline: bool = True
    run_energy: bool = True
    run_memory: bool = True
    run_concurrency: bool = False  # Optional, expensive

    # Partitioning strategy
    use_fusion_partitioning: bool = True  # vs basic partitioning

    # Analysis options
    estimate_checkpointing_savings: bool = False
    estimate_quantization_savings: bool = False
    detailed_memory_timeline: bool = True

    # Validation
    validate_consistency: bool = True  # Check reports agree with each other
```

### Core Class: UnifiedAnalyzer

```python
class UnifiedAnalyzer:
    """
    Unified orchestrator for comprehensive model analysis.

    Coordinates Phase 3 analyzers with correct data dependencies:
    1. Trace model with FX + shape propagation
    2. Partition graph into subgraphs
    3. Run roofline analysis (latency, bottlenecks)
    4. Run energy analysis (uses latencies from roofline)
    5. Run memory analysis (peak memory, timeline)
    6. Optionally run concurrency analysis

    Usage:
        analyzer = UnifiedAnalyzer()
        result = analyzer.analyze_model(
            model_name='resnet18',
            hardware_name='H100',
            batch_size=1,
            precision=Precision.FP32,
            config=AnalysisConfig()
        )
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize analyzer.

        Args:
            verbose: Print progress messages during analysis
        """
        self.verbose = verbose

    def analyze_model(
        self,
        model_name: str,
        hardware_name: str,
        batch_size: int = 1,
        precision: Precision = Precision.FP32,
        config: Optional[AnalysisConfig] = None,
    ) -> UnifiedAnalysisResult:
        """
        Run comprehensive analysis on a model.

        This is the main entry point. It orchestrates all sub-analyses
        in the correct order with proper data dependencies.

        Args:
            model_name: Model to analyze (e.g., 'resnet18')
            hardware_name: Target hardware (e.g., 'H100')
            batch_size: Batch size for analysis
            precision: Precision to use (FP32, FP16, INT8)
            config: Analysis configuration (uses defaults if None)

        Returns:
            UnifiedAnalysisResult with all analysis reports

        Raises:
            ValueError: If model or hardware not found
            RuntimeError: If analysis fails
        """
        ...

    def analyze_model_with_custom_hardware(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        model_name: str,
        hardware_mapper: HardwareMapper,
        precision: Precision = Precision.FP32,
        config: Optional[AnalysisConfig] = None,
    ) -> UnifiedAnalysisResult:
        """
        Analyze with custom model/hardware (for advanced users).

        Args:
            model: PyTorch model to analyze
            input_tensor: Input tensor for shape propagation
            model_name: Display name for model
            hardware_mapper: Hardware mapper with resource model
            precision: Precision to use
            config: Analysis configuration

        Returns:
            UnifiedAnalysisResult with all analysis reports
        """
        ...

    # Private methods for implementation
    def _trace_and_partition(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        config: AnalysisConfig
    ) -> Tuple[torch.fx.GraphModule, PartitionReport]:
        """Trace model and partition into subgraphs"""
        ...

    def _run_roofline_analysis(
        self,
        partition_report: PartitionReport,
        hardware: HardwareResourceModel,
        precision: Precision
    ) -> RooflineReport:
        """Run roofline analysis for latency and bottlenecks"""
        ...

    def _run_energy_analysis(
        self,
        partition_report: PartitionReport,
        hardware: HardwareResourceModel,
        roofline_report: RooflineReport,
        precision: Precision
    ) -> EnergyReport:
        """Run energy analysis using roofline latencies"""
        ...

    def _run_memory_analysis(
        self,
        partition_report: PartitionReport,
        hardware: HardwareResourceModel,
        config: AnalysisConfig
    ) -> MemoryReport:
        """Run memory analysis for peak usage and timeline"""
        ...

    def _run_concurrency_analysis(
        self,
        partition_report: PartitionReport
    ) -> ConcurrencyReport:
        """Run concurrency analysis for parallelism opportunities"""
        ...

    def _compute_derived_metrics(
        self,
        result: UnifiedAnalysisResult
    ) -> None:
        """Compute derived metrics from analysis reports"""
        ...

    def _validate_result(
        self,
        result: UnifiedAnalysisResult
    ) -> List[str]:
        """Validate consistency between reports"""
        ...
```

### Key Features

1. **Single Entry Point**: One function call for all analyses
2. **Correct Dependencies**: Energy analysis gets latencies from roofline
3. **Flexible Configuration**: Control which analyses run
4. **Validation**: Check consistency between reports
5. **Error Handling**: Clear error messages for common issues
6. **Performance**: Avoid redundant computation

### Example Usage

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.precision import Precision

# Basic usage
analyzer = UnifiedAnalyzer(verbose=True)
result = analyzer.analyze_model(
    model_name='resnet18',
    hardware_name='H100',
    batch_size=1,
    precision=Precision.FP32
)

print(f"Latency: {result.total_latency_ms:.2f} ms")
print(f"Energy: {result.total_energy_mj:.2f} mJ")
print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")

# Custom configuration
config = AnalysisConfig(
    run_concurrency=True,  # Include expensive concurrency analysis
    estimate_checkpointing_savings=True,
    validate_consistency=True
)

result = analyzer.analyze_model(
    model_name='resnet50',
    hardware_name='Jetson-Orin-AGX',
    batch_size=8,
    precision=Precision.FP16,
    config=config
)

# Access individual reports
print(result.roofline_report.format_report())
print(result.energy_report.format_report())
print(result.memory_report.format_report())

# Get executive summary
summary = result.get_executive_summary()
```

---

## Phase 4.2.2: ReportGenerator

### Purpose
Flexible reporting engine that transforms UnifiedAnalysisResult into various output formats with customizable templates.

### Location
`src/graphs/reporting/report_generator.py`

### Core Class: ReportGenerator

```python
class ReportGenerator:
    """
    Flexible report generation from unified analysis results.

    Supports multiple output formats:
    - Text: Human-readable console output
    - JSON: Machine-readable, preserves all data
    - CSV: Spreadsheet-friendly, flattened data
    - Markdown: Documentation-friendly with tables
    - HTML: Rich formatted reports with charts

    Usage:
        generator = ReportGenerator()

        # Text report
        text = generator.generate_text_report(result)
        print(text)

        # JSON export
        json_data = generator.generate_json_report(result)
        with open('report.json', 'w') as f:
            json.dump(json_data, f, indent=2)

        # Comparison report
        comparison = generator.generate_comparison_report([result1, result2, result3])
        print(comparison)
    """

    def __init__(self, style: str = 'default'):
        """
        Initialize report generator.

        Args:
            style: Report style ('default', 'compact', 'detailed')
        """
        self.style = style

    # ===== Single Model Reports =====

    def generate_text_report(
        self,
        result: UnifiedAnalysisResult,
        include_sections: Optional[List[str]] = None,
        show_executive_summary: bool = True
    ) -> str:
        """
        Generate human-readable text report.

        Args:
            result: Analysis result
            include_sections: List of sections to include (all if None)
                             ['executive', 'performance', 'energy', 'memory', 'recommendations']
            show_executive_summary: Include executive summary at top

        Returns:
            Formatted text report
        """
        ...

    def generate_json_report(
        self,
        result: UnifiedAnalysisResult,
        include_raw_reports: bool = True,
        pretty_print: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate JSON report.

        Args:
            result: Analysis result
            include_raw_reports: Include full Phase 3 reports
            pretty_print: Return formatted string vs dict

        Returns:
            JSON string or dict
        """
        ...

    def generate_csv_report(
        self,
        result: UnifiedAnalysisResult,
        include_subgraph_details: bool = False
    ) -> str:
        """
        Generate CSV report (flattened data).

        Args:
            result: Analysis result
            include_subgraph_details: Include per-subgraph rows

        Returns:
            CSV string
        """
        ...

    def generate_markdown_report(
        self,
        result: UnifiedAnalysisResult,
        include_tables: bool = True,
        include_charts: bool = False  # ASCII charts
    ) -> str:
        """
        Generate Markdown report.

        Args:
            result: Analysis result
            include_tables: Include formatted tables
            include_charts: Include ASCII charts (experimental)

        Returns:
            Markdown string
        """
        ...

    def generate_html_report(
        self,
        result: UnifiedAnalysisResult,
        include_interactive_charts: bool = True,
        template: str = 'default'
    ) -> str:
        """
        Generate HTML report with styling and charts.

        Args:
            result: Analysis result
            include_interactive_charts: Include Chart.js visualizations
            template: HTML template to use

        Returns:
            HTML string
        """
        ...

    # ===== Comparison Reports =====

    def generate_comparison_report(
        self,
        results: List[UnifiedAnalysisResult],
        comparison_dimension: str = 'auto',  # 'model', 'hardware', 'batch_size', 'auto'
        format: str = 'text',  # 'text', 'csv', 'markdown', 'html'
        sort_by: str = 'latency'  # 'latency', 'energy', 'throughput', 'efficiency'
    ) -> str:
        """
        Generate comparison report across multiple configurations.

        Args:
            results: List of analysis results to compare
            comparison_dimension: What varies ('model', 'hardware', 'batch_size', or 'auto')
            format: Output format
            sort_by: Metric to sort by

        Returns:
            Formatted comparison report
        """
        ...

    # ===== Executive Summaries =====

    def generate_executive_summary(
        self,
        result: UnifiedAnalysisResult,
        target_audience: str = 'technical'  # 'technical', 'management', 'mixed'
    ) -> str:
        """
        Generate concise executive summary.

        Args:
            result: Analysis result
            target_audience: Adjust detail level and terminology

        Returns:
            Executive summary text
        """
        ...

    # ===== Specialized Reports =====

    def generate_energy_breakdown_report(
        self,
        result: UnifiedAnalysisResult,
        include_chart: bool = True
    ) -> str:
        """Generate detailed energy breakdown report"""
        ...

    def generate_bottleneck_report(
        self,
        result: UnifiedAnalysisResult,
        threshold_pct: float = 10.0
    ) -> str:
        """Generate report focused on performance bottlenecks"""
        ...

    def generate_optimization_recommendations(
        self,
        result: UnifiedAnalysisResult,
        prioritize_by: str = 'impact'  # 'impact', 'difficulty', 'both'
    ) -> str:
        """Generate prioritized optimization recommendations"""
        ...

    # ===== Export Utilities =====

    def save_report(
        self,
        result: UnifiedAnalysisResult,
        output_path: str,
        format: Optional[str] = None  # Auto-detect from extension if None
    ) -> None:
        """
        Save report to file.

        Args:
            result: Analysis result
            output_path: Output file path
            format: Format override (auto-detect from extension if None)
        """
        ...

    def save_comparison_report(
        self,
        results: List[UnifiedAnalysisResult],
        output_path: str,
        **kwargs
    ) -> None:
        """Save comparison report to file"""
        ...
```

### Report Templates

The ReportGenerator will support customizable templates:

**Text Template Structure:**
```
═══════════════════════════════════════════════════════════════
                 {TITLE}
═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────
{executive_summary}

PERFORMANCE ANALYSIS
─────────────────────────────────────────────────────────────────
{performance_metrics}

ENERGY ANALYSIS
─────────────────────────────────────────────────────────────────
{energy_metrics}

MEMORY ANALYSIS
─────────────────────────────────────────────────────────────────
{memory_metrics}

RECOMMENDATIONS
─────────────────────────────────────────────────────────────────
{recommendations}
```

**JSON Structure:**
```json
{
  "metadata": {
    "model": "resnet18",
    "hardware": "H100 SXM5 80GB",
    "batch_size": 1,
    "precision": "FP32",
    "timestamp": "2025-10-28T..."
  },
  "executive_summary": {
    "latency_ms": 5.98,
    "throughput_fps": 167.2,
    "total_energy_mj": 29.4,
    "energy_per_inference_mj": 29.4,
    "peak_memory_mb": 44.7,
    "utilization_pct": 5.5
  },
  "roofline_analysis": { ... },
  "energy_analysis": { ... },
  "memory_analysis": { ... },
  "recommendations": [ ... ]
}
```

**CSV Structure (Flattened):**
```csv
model,hardware,batch_size,precision,latency_ms,throughput_fps,energy_mj,energy_per_inf_mj,peak_mem_mb,utilization_pct
resnet18,H100,1,FP32,5.98,167.2,29.4,29.4,44.7,5.5
```

---

## Phase 4.2.3: Refactor CLI Tools

### Goals
1. Reduce code duplication by 60%
2. Ensure consistent results across tools
3. Simplify maintenance
4. Improve performance (avoid redundant work)

### Changes to analyze_comprehensive.py

**Before (Phase 4.1):**
- ~1,000 lines total
- ~200 lines for analysis orchestration
- ~300 lines for report formatting
- ~500 lines for CLI logic and utilities

**After (Phase 4.2):**
- ~400 lines total (60% reduction in analysis/reporting code)
- Use UnifiedAnalyzer for orchestration
- Use ReportGenerator for all output formats
- CLI logic only

**Code Reduction Example:**
```python
# Before (Phase 4.1): ~200 lines of orchestration
def analyze_model(model_name, hardware_name, ...):
    model, input_tensor = create_model(...)
    hardware_mapper = create_hardware_mapper(...)

    fx_graph = symbolic_trace(model)
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partitioner = GraphPartitioner()
    partition_report = partitioner.partition(fx_graph)

    roofline_analyzer = RooflineAnalyzer(hardware, precision)
    roofline_report = roofline_analyzer.analyze(...)

    energy_analyzer = EnergyAnalyzer(hardware, precision)
    latencies = [lat.actual_latency for lat in roofline_report.latencies]
    energy_report = energy_analyzer.analyze(..., latencies=latencies)

    memory_estimator = MemoryEstimator(hardware)
    memory_report = memory_estimator.estimate_memory(...)

    # ... more analysis code ...

    return ComprehensiveAnalysisReport(...)  # Custom report type

# After (Phase 4.2): ~10 lines
def analyze_model(model_name, hardware_name, ...):
    analyzer = UnifiedAnalyzer(verbose=args.verbose)
    result = analyzer.analyze_model(
        model_name=model_name,
        hardware_name=hardware_name,
        batch_size=batch_size,
        precision=precision
    )
    return result  # Standard UnifiedAnalysisResult
```

```python
# Before (Phase 4.1): ~300 lines of formatting
def generate_text_report(report):
    # Custom formatting code for text output
    output = []
    output.append("=" * 60)
    output.append("COMPREHENSIVE ANALYSIS REPORT")
    output.append("=" * 60)
    # ... 300 more lines ...
    return "\n".join(output)

def generate_json_report(report):
    # Custom JSON serialization
    # ... 100 lines ...

def generate_csv_report(report):
    # Custom CSV formatting
    # ... 100 lines ...

# After (Phase 4.2): ~5 lines
def main():
    result = analyzer.analyze_model(...)

    generator = ReportGenerator(style='detailed')

    if args.format == 'json':
        output = generator.generate_json_report(result)
    elif args.format == 'csv':
        output = generator.generate_csv_report(result)
    else:
        output = generator.generate_text_report(result)

    # Save or print
    if args.output:
        generator.save_report(result, args.output)
    else:
        print(output)
```

### Changes to analyze_batch.py

Similar refactoring:
- Use UnifiedAnalyzer for each configuration
- Use ReportGenerator.generate_comparison_report() for batch comparisons
- Reduce from ~600 lines to ~300 lines

### Changes to analyze_graph_mapping.py

Optional enhancement:
- Can optionally use UnifiedAnalyzer for Phase 3 modes
- Maintains backward compatibility
- Original allocation analysis remains unchanged

---

## Phase 4.2.4: Testing

### Test Coverage

**Unit Tests: test_unified_analyzer.py**
```python
class TestUnifiedAnalyzer:
    def test_basic_analysis(self):
        """Test basic analysis with all components"""

    def test_custom_config(self):
        """Test with custom AnalysisConfig"""

    def test_different_precisions(self):
        """Test FP32, FP16, INT8"""

    def test_batch_sizes(self):
        """Test different batch sizes"""

    def test_error_handling(self):
        """Test error cases (invalid model, hardware)"""

    def test_consistency_validation(self):
        """Test report consistency validation"""
```

**Unit Tests: test_report_generator.py**
```python
class TestReportGenerator:
    def test_text_report(self):
        """Test text report generation"""

    def test_json_report(self):
        """Test JSON report generation and parsing"""

    def test_csv_report(self):
        """Test CSV report generation"""

    def test_markdown_report(self):
        """Test Markdown report generation"""

    def test_comparison_report(self):
        """Test comparison report across configs"""

    def test_executive_summary(self):
        """Test executive summary generation"""

    def test_save_report(self):
        """Test saving reports to files"""
```

**Integration Tests: test_cli_refactored.py**
```python
class TestRefactoredCLITools:
    def test_analyze_comprehensive_consistency(self):
        """Verify analyze_comprehensive.py produces same results as Phase 4.1"""

    def test_analyze_batch_consistency(self):
        """Verify analyze_batch.py produces same results as Phase 4.1"""

    def test_cross_tool_consistency(self):
        """Verify all tools produce consistent results for same config"""
```

---

## Phase 4.2.5: Documentation

### API Documentation

**Location:** `docs/api/unified_analyzer.md`
- Complete API reference for UnifiedAnalyzer
- Usage examples
- Configuration options
- Error handling guide

**Location:** `docs/api/report_generator.md`
- Complete API reference for ReportGenerator
- Report format specifications
- Template customization
- Comparison report examples

### User Guide

**Location:** `docs/guides/unified_workflows.md`
- Conceptual overview
- Common use cases
- Programmatic usage examples
- Migration guide from Phase 4.1

---

## Implementation Timeline

### Phase 4.2.1: UnifiedAnalyzer (Estimated: 2-3 hours)
- [ ] Define data structures (UnifiedAnalysisResult, AnalysisConfig)
- [ ] Implement core UnifiedAnalyzer class
- [ ] Implement private orchestration methods
- [ ] Add validation and error handling
- [ ] Unit tests

### Phase 4.2.2: ReportGenerator (Estimated: 3-4 hours)
- [ ] Define ReportGenerator class
- [ ] Implement text report generation
- [ ] Implement JSON report generation
- [ ] Implement CSV report generation
- [ ] Implement Markdown report generation
- [ ] Implement comparison reports
- [ ] Unit tests

### Phase 4.2.3: Refactor CLI Tools (Estimated: 2-3 hours)
- [ ] Refactor analyze_comprehensive.py
- [ ] Refactor analyze_batch.py
- [ ] Optionally enhance analyze_graph_mapping.py
- [ ] Integration tests

### Phase 4.2.4: Documentation (Estimated: 1-2 hours)
- [ ] API documentation
- [ ] User guide
- [ ] Migration guide
- [ ] Update CLI README

**Total Estimated Time:** 8-12 hours of development

---

## Success Criteria

1. ✅ UnifiedAnalyzer produces consistent results across all use cases
2. ✅ ReportGenerator supports all required formats (text, JSON, CSV, markdown)
3. ✅ CLI tools refactored with 60% code reduction in analysis/reporting logic
4. ✅ All existing tests pass
5. ✅ New unit tests for UnifiedAnalyzer and ReportGenerator (>90% coverage)
6. ✅ Integration tests verify consistency
7. ✅ Documentation complete (API reference + user guide)
8. ✅ Performance: No regression vs Phase 4.1 (same latency for same analysis)

---

## Benefits

### Code Quality
- **60% reduction** in duplicated analysis/reporting code
- **Single source of truth** for analysis logic
- **Consistent results** across all tools
- **Easier maintenance** - fix once, apply everywhere

### User Experience
- **Consistent output** across all tools
- **More output formats** (HTML, improved markdown)
- **Better error messages** (centralized error handling)
- **Programmatic access** for advanced users

### Developer Experience
- **Cleaner CLI code** - focus on CLI logic, not analysis
- **Easier to add new tools** - leverage unified components
- **Better testability** - unit test components independently
- **Clear separation** of concerns (analysis vs reporting)

---

## Dependencies

### Prerequisites
- Phase 4.1 complete (CLI tools working)
- Phase 3 analyzers stable (RooflineAnalyzer, EnergyAnalyzer, MemoryEstimator)

### No New External Dependencies
- Uses only existing packages (PyTorch, dataclasses, typing)
- Optional: `jinja2` for HTML templates (can defer to Phase 4.3)

---

## Risks and Mitigations

**Risk 1:** Breaking changes to CLI tools
- **Mitigation:** Thorough integration tests, maintain backward compatibility

**Risk 2:** Performance regression
- **Mitigation:** Performance tests, avoid redundant work in orchestration

**Risk 3:** Inconsistent results vs Phase 4.1
- **Mitigation:** Cross-validation tests comparing Phase 4.1 vs 4.2 results

---

## Future Extensions (Beyond Phase 4.2)

1. **HTML Reports with Interactive Charts** (Phase 4.3)
   - Leverage Chart.js or plotly for interactive visualizations
   - Roofline plots, energy breakdowns, memory timelines

2. **Custom Report Templates** (Phase 4.3+)
   - User-defined Jinja2 templates
   - Corporate branding support

3. **Report Caching** (Future)
   - Cache UnifiedAnalysisResult to avoid re-analysis
   - Useful for repeated report generation with different formats

4. **Async Analysis** (Future)
   - Parallel analysis of multiple configurations
   - Useful for large batch sweeps

---

## References

- Phase 4 Plan: `docs/PHASE4_INTEGRATION_PLAN.md`
- Phase 4.1 Implementation: `cli/analyze_comprehensive.py`, `cli/analyze_batch.py`
- Phase 3 Analyzers: `src/graphs/analysis/roofline.py`, `energy.py`, `memory.py`
