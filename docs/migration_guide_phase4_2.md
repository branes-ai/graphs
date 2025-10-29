# Migration Guide: Phase 4.1 → Phase 4.2

**Unified Analysis Framework Migration**

This guide helps you migrate from Phase 4.1 tools to the new Phase 4.2 unified framework.

---

## Overview

**Phase 4.2** introduces the unified analysis framework with:
- **UnifiedAnalyzer**: Single orchestrator for all Phase 3 analyzers
- **ReportGenerator**: Flexible reporting engine with multiple formats
- **Refactored CLI tools**: 61.5% code reduction while maintaining all functionality

**Benefits of Migrating:**
- ✅ Simpler API (single entry point for analysis)
- ✅ Consistent output across all tools
- ✅ Better error handling and validation
- ✅ More output formats (text, JSON, CSV, markdown)
- ✅ Easier to maintain (fix bugs once, benefit everywhere)
- ✅ Better performance (optimized orchestration)

---

## CLI Tool Migration

### Quick Reference

| Phase 4.1 Tool | Phase 4.2 Tool | Status |
|----------------|----------------|--------|
| `analyze_comprehensive.py` | `analyze_comprehensive_v2.py` | ✅ Production-ready |
| `analyze_batch.py` | `analyze_batch_v2.py` | ✅ Production-ready |

### analyze_comprehensive.py → analyze_comprehensive_v2.py

**Good News:** The v2 tool is a **drop-in replacement** with identical command-line arguments!

**Before (Phase 4.1):**
```bash
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
  --output results.json --precision fp16 --batch-size 8
```

**After (Phase 4.2):**
```bash
./cli/analyze_comprehensive_v2.py --model resnet18 --hardware H100 \
  --output results.json --precision fp16 --batch-size 8
```

**All arguments work identically:**
- `--model`, `--hardware`, `--precision`, `--batch-size` - same behavior
- `--output`, `--format` - same behavior
- `--quiet`, `--sections`, `--subgraph-details` - same behavior
- `--no-roofline`, `--no-energy`, `--no-memory` - same behavior

**What Changed:**
- 73% less code (962 lines → 262 lines)
- Uses UnifiedAnalyzer internally
- Uses ReportGenerator for all output
- Better error messages
- Slightly faster (optimized orchestration)

**What's the Same:**
- All command-line arguments
- All output formats
- All features and functionality
- Output format and structure

### analyze_batch.py → analyze_batch_v2.py

**Before (Phase 4.1):**
```bash
./cli/analyze_batch.py --model resnet18 --hardware H100 \
  --batch-size 1 4 8 16 --output batch_sweep.csv
```

**After (Phase 4.2):**
```bash
./cli/analyze_batch_v2.py --model resnet18 --hardware H100 \
  --batch-size 1 4 8 16 --output batch_sweep.csv
```

**All features preserved:**
- Batch size sweeps
- Model comparison (`--models`)
- Hardware comparison (multiple `--hardware`)
- Insights generation
- CSV/JSON output

**What Changed:**
- 42% less code (572 lines → 329 lines)
- Uses UnifiedAnalyzer internally
- Better comparison reports
- Cleaner insights formatting

---

## Python API Migration

### Basic Analysis

**Before (Phase 4.1 - Manual Orchestration):**
```python
from torch.fx import symbolic_trace
from graphs.ir.shape_propagation import ShapeProp
from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.roofline_analyzer import RooflineAnalyzer
from graphs.analysis.energy_analyzer import EnergyAnalyzer
from graphs.analysis.memory_estimator import MemoryEstimator
from graphs.hardware.resource_model import get_hardware_model
from graphs.models import get_model

# Get model and hardware
model = get_model('resnet18')
hardware = get_hardware_model('H100')
input_tensor = torch.randn(1, 3, 224, 224)

# Trace and propagate shapes
fx_graph = symbolic_trace(model)
shape_prop = ShapeProp(fx_graph)
shape_prop.propagate(input_tensor)

# Partition graph
partitioner = GraphPartitioner()
partition_report = partitioner.partition(fx_graph)

# Run roofline analysis
roofline_analyzer = RooflineAnalyzer(hardware, precision=Precision.FP32)
roofline_report = roofline_analyzer.analyze(
    partition_report.subgraphs,
    partition_report
)

# Run energy analysis (needs latencies from roofline)
energy_analyzer = EnergyAnalyzer(hardware, precision=Precision.FP32)
latencies = [lat.actual_latency for lat in roofline_report.latencies]
energy_report = energy_analyzer.analyze(
    partition_report.subgraphs,
    partition_report,
    latencies=latencies
)

# Run memory analysis
memory_estimator = MemoryEstimator()
memory_report = memory_estimator.estimate(partition_report.subgraphs)

# Extract metrics manually
total_latency_ms = sum(lat.actual_latency for lat in roofline_report.latencies)
throughput_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0
total_energy_mj = (energy_report.compute_energy_j +
                   energy_report.memory_energy_j +
                   energy_report.static_energy_j) * 1000
peak_memory_mb = memory_report.peak_memory_bytes / 1e6

print(f"Latency: {total_latency_ms:.2f} ms")
print(f"Throughput: {throughput_fps:.1f} fps")
print(f"Energy: {total_energy_mj:.1f} mJ")
print(f"Memory: {peak_memory_mb:.1f} MB")
```

**After (Phase 4.2 - Unified Framework):**
```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from graphs.reporting import ReportGenerator

# Create analyzer
analyzer = UnifiedAnalyzer()

# Run analysis (all in one call)
result = analyzer.analyze_model('resnet18', 'H100')

# Access derived metrics (computed automatically)
print(f"Latency: {result.total_latency_ms:.2f} ms")
print(f"Throughput: {result.throughput_fps:.1f} fps")
print(f"Energy: {result.total_energy_mj:.1f} mJ")
print(f"Memory: {result.peak_memory_mb:.1f} MB")

# Generate report
generator = ReportGenerator()
report = generator.generate_text_report(result)
print(report)
```

**Benefits:**
- **~60 lines → ~10 lines** (6× reduction)
- No manual orchestration needed
- Derived metrics computed automatically
- Consistent results across tools

### Custom Models

**Before (Phase 4.1):**
```python
# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

model = MyModel()
input_tensor = torch.randn(1, 3, 224, 224)

# Manual tracing and analysis (many steps...)
fx_graph = symbolic_trace(model)
# ... (20+ lines of orchestration code)
```

**After (Phase 4.2):**
```python
# Define model (same as before)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

model = MyModel()
input_tensor = torch.randn(1, 3, 224, 224)

# Simplified analysis
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_graph(
    model=model,
    input_tensor=input_tensor,
    model_name='MyCustomModel',
    hardware_name='H100'
)

# Done! All analysis complete
```

### Batch Size Sweeps

**Before (Phase 4.1):**
```python
results = []

for batch_size in [1, 4, 8, 16]:
    # Create new model instance
    model = get_model('resnet18')
    hardware = get_hardware_model('H100')
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Manual tracing
    fx_graph = symbolic_trace(model)
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Manual analysis (20+ lines per iteration)
    partitioner = GraphPartitioner()
    partition_report = partitioner.partition(fx_graph)

    roofline_analyzer = RooflineAnalyzer(hardware, precision=Precision.FP32)
    roofline_report = roofline_analyzer.analyze(
        partition_report.subgraphs,
        partition_report
    )

    # ... more analysis code ...

    results.append({
        'batch_size': batch_size,
        'latency': total_latency_ms,
        # ... manual extraction of metrics
    })

# Manual CSV generation
import csv
with open('results.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['batch_size', 'latency', ...])
    writer.writeheader()
    writer.writerows(results)
```

**After (Phase 4.2):**
```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator

analyzer = UnifiedAnalyzer(verbose=False)
generator = ReportGenerator()

results = []
for batch_size in [1, 4, 8, 16]:
    result = analyzer.analyze_model('resnet18', 'H100', batch_size=batch_size)
    results.append(result)

# Generate CSV automatically
csv_output = generator.generate_comparison_report(results, format='csv')
with open('results.csv', 'w') as f:
    f.write(csv_output)
```

**Benefits:**
- **~100 lines → ~10 lines** (10× reduction)
- No manual metric extraction
- Automatic CSV generation with proper formatting
- Comparison insights included

### Report Generation

**Before (Phase 4.1):**
```python
# Manual report formatting (custom code for each format)
def format_text_report(roofline_report, energy_report, memory_report):
    lines = []
    lines.append("=" * 79)
    lines.append("ANALYSIS REPORT")
    lines.append("=" * 79)
    # ... 100+ lines of formatting code ...
    return "\n".join(lines)

def format_json_report(roofline_report, energy_report, memory_report):
    # ... 50+ lines of JSON serialization ...
    return json.dumps(data, indent=2)

def format_csv_report(roofline_report, energy_report, memory_report):
    # ... 50+ lines of CSV generation ...
    return csv_str

# Generate reports
text_report = format_text_report(roofline_report, energy_report, memory_report)
json_report = format_json_report(roofline_report, energy_report, memory_report)
csv_report = format_csv_report(roofline_report, energy_report, memory_report)
```

**After (Phase 4.2):**
```python
from graphs.reporting import ReportGenerator

generator = ReportGenerator()

# Generate all formats with one call each
text_report = generator.generate_text_report(result)
json_report = generator.generate_json_report(result)
csv_report = generator.generate_csv_report(result)
md_report = generator.generate_markdown_report(result)

# Or save directly
generator.save_report(result, 'report.txt')   # Auto-detects format
generator.save_report(result, 'report.json')
generator.save_report(result, 'report.csv')
generator.save_report(result, 'report.md')
```

**Benefits:**
- No custom formatting code needed
- Consistent format across tools
- Auto-detection from file extension
- Easy to add new formats

---

## Configuration Migration

### Analysis Configuration

**Before (Phase 4.1):**
```python
# Configure each analyzer separately
roofline_analyzer = RooflineAnalyzer(
    hardware,
    precision=Precision.FP16,
    validate=True
)

energy_analyzer = EnergyAnalyzer(
    hardware,
    precision=Precision.FP16,
    include_static=True
)

memory_estimator = MemoryEstimator(
    track_timeline=True
)
```

**After (Phase 4.2):**
```python
from graphs.analysis.unified_analyzer import AnalysisConfig

config = AnalysisConfig(
    run_roofline=True,
    run_energy=True,
    run_memory=True,
    run_concurrency=False,  # Skip expensive analysis
    validate_consistency=True
)

result = analyzer.analyze_model(
    'resnet50',
    'H100',
    batch_size=8,
    precision=Precision.FP16,
    config=config
)
```

**Benefits:**
- Single configuration object
- Clear control over which analyzers run
- Consistent validation across analyzers

---

## Output Format Migration

### JSON Output

**Before (Phase 4.1):**
```python
# Custom JSON structure, different per tool
{
    "roofline": {
        "latencies": [...],
        "bottlenecks": [...]
    },
    "energy": {
        "compute": 10.5,
        "memory": 2.3,
        "static": 5.1
    },
    "memory": {
        "peak_bytes": 55000000
    }
}
```

**After (Phase 4.2):**
```python
# Consistent JSON structure across all tools
{
    "metadata": {
        "model": "ResNet-18",
        "hardware": "H100",
        "batch_size": 1,
        "precision": "FP32",
        "timestamp": "2025-10-28T12:00:00"
    },
    "derived_metrics": {
        "latency_ms": 0.43,
        "throughput_fps": 2318,
        "total_energy_mj": 48.9,
        "peak_memory_mb": 55.0,
        "hardware_utilization": 10.2
    },
    "reports": {
        "roofline": { ... },
        "energy": { ... },
        "memory": { ... }
    }
}
```

**Benefits:**
- Consistent structure across all tools
- Derived metrics at top level for easy access
- Full reports available when needed

### CSV Output

**Before (Phase 4.1):**
```csv
model,latency_ms,energy_mj,memory_mb
resnet18,0.43,48.9,55.0
```

**After (Phase 4.2):**
```csv
model,hardware,batch_size,precision,latency_ms,throughput_fps,energy_mj,energy_per_inf_mj,peak_mem_mb,utilization_pct
ResNet-18,H100-PCIe-80GB,1,FP32,0.43,2317.5,48.89,48.89,55.0,10.2
```

**Benefits:**
- More comprehensive metadata
- Per-inference metrics included
- Consistent column names across tools
- Optional subgraph details

---

## Breaking Changes

### None for CLI Tools

The v2 CLI tools are **100% backward compatible**. No breaking changes!

### Minor Changes for Python API

If you were using internal Phase 3 analyzer APIs directly:

**1. Energy Report Attributes**

```python
# Before
energy_report.total_compute_energy_j
energy_report.total_memory_energy_j
energy_report.total_static_energy_j

# After
energy_report.compute_energy_j  # Removed "total_" prefix
energy_report.memory_energy_j
energy_report.static_energy_j
```

**2. Memory Report Attributes**

```python
# Before
memory_report.timeline

# After
memory_report.memory_timeline  # More explicit name
```

**3. Concurrency Report**

```python
# Before
from graphs.analysis.concurrency import ConcurrencyReport

# After
from graphs.ir.structures import ConcurrencyDescriptor
```

---

## Recommended Migration Path

### Step 1: Try v2 Tools (No Code Changes)

```bash
# Just change the script name
./cli/analyze_comprehensive_v2.py --model resnet18 --hardware H100

# Compare output with v1
./cli/analyze_comprehensive.py --model resnet18 --hardware H100

# Outputs should be nearly identical
```

### Step 2: Update Shell Scripts

```bash
# Before
./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --output results.json

# After (just add _v2)
./cli/analyze_comprehensive_v2.py --model resnet18 --hardware H100 --output results.json
```

### Step 3: Migrate Python Code

```python
# Before
from graphs.analysis.roofline_analyzer import RooflineAnalyzer
# ... many imports and manual orchestration ...

# After
from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from graphs.reporting import ReportGenerator

analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'H100')
```

### Step 4: Test and Validate

```bash
# Run integration tests to ensure everything works
python -m pytest tests/integration/test_unified_workflows.py -v
python -m pytest tests/integration/test_cli_refactored.py -v
```

---

## Compatibility Matrix

| Feature | Phase 4.1 | Phase 4.2 | Notes |
|---------|-----------|-----------|-------|
| **CLI Arguments** | ✅ | ✅ | Identical |
| **Output Formats** | Text, JSON, CSV | Text, JSON, CSV, Markdown | Markdown added |
| **Batch Analysis** | ✅ | ✅ | Improved insights |
| **Model Comparison** | ✅ | ✅ | Better formatting |
| **Hardware Comparison** | ✅ | ✅ | Better formatting |
| **Custom Models** | Manual orchestration | Simple API | Much easier |
| **Report Generation** | Custom code | ReportGenerator | Unified |
| **Code Size** | Large (1534 lines) | Small (591 lines) | 61.5% reduction |

---

## Rollback Plan

If you need to rollback to Phase 4.1 tools:

```bash
# Phase 4.1 tools are still available
./cli/analyze_comprehensive.py --model resnet18 --hardware H100
./cli/analyze_batch.py --model resnet18 --hardware H100 --batch-size 1 4 8

# They continue to work unchanged
```

**Note:** Phase 4.1 tools are considered legacy but will remain available for compatibility.

---

## Getting Help

- **API Documentation**: See `docs/UNIFIED_FRAMEWORK_API.md`
- **CLI Documentation**: See `cli/README.md`
- **Examples**: Check `tests/integration/test_unified_workflows.py`
- **Refactoring Summary**: See `docs/sessions/2025-10-28_phase4_2_refactoring_summary.md`

---

## Summary

**For CLI Users:**
- ✅ Use v2 tools (`*_v2.py`) - they're drop-in replacements
- ✅ Same commands, same arguments, same output
- ✅ Better performance, better error handling

**For Python API Users:**
- ✅ Migrate to UnifiedAnalyzer for simplified orchestration
- ✅ Use ReportGenerator for consistent output formatting
- ✅ 6-10× code reduction while maintaining all features

**Migration Effort:**
- **CLI users**: 5 minutes (just change script name)
- **Python API users**: 30-60 minutes (rewrite orchestration code)
- **Tool developers**: Review `docs/UNIFIED_FRAMEWORK_API.md`

**Phase 4.2 is production-ready and recommended for all new work!**
