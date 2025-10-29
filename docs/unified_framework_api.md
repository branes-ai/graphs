# Unified Analysis Framework API Guide

**Phase 4.2 Developer Documentation**

This guide explains how to use the unified analysis framework programmatically for custom analysis workflows, automation, and integration into other tools.

---

## Overview

The unified framework consists of two main components:

1. **UnifiedAnalyzer**: Orchestrates all Phase 3 analyzers (roofline, energy, memory, concurrency)
2. **ReportGenerator**: Transforms analysis results into multiple output formats

**Benefits:**
- Single point of entry for all analysis
- Consistent results across tools
- Flexible configuration
- Multiple output formats
- Easier to maintain and extend

---

## Quick Start

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from graphs.reporting import ReportGenerator

# Create analyzer
analyzer = UnifiedAnalyzer()

# Run analysis
result = analyzer.analyze_model('resnet18', 'H100')

# Generate report
generator = ReportGenerator()
report = generator.generate_text_report(result)
print(report)
```

---

## UnifiedAnalyzer API

### Basic Usage

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision

# Create analyzer
analyzer = UnifiedAnalyzer(verbose=True)

# Analyze with defaults (FP32, batch=1, all analyzers)
result = analyzer.analyze_model('resnet18', 'H100')

# Analyze with custom configuration
result = analyzer.analyze_model(
    model_name='resnet50',
    hardware_name='Jetson-Orin-AGX',
    batch_size=8,
    precision=Precision.FP16,
    config=AnalysisConfig(
        run_roofline=True,
        run_energy=True,
        run_memory=True,
        run_concurrency=False,  # Skip concurrency for speed
        validate_consistency=True
    )
)
```

### Method: `analyze_model()`

Analyzes a model by name using the model registry.

**Parameters:**
- `model_name` (str): Name of model from registry (e.g., 'resnet18', 'mobilenet_v2')
- `hardware_name` (str): Name of hardware target (e.g., 'H100', 'Jetson-Orin-Nano')
- `batch_size` (int): Batch size (default: 1)
- `precision` (Precision): Precision to use (default: Precision.FP32)
- `config` (AnalysisConfig): Analysis configuration (default: all analyzers enabled)

**Returns:** `UnifiedAnalysisResult` containing all analysis data

**Example:**
```python
result = analyzer.analyze_model(
    model_name='efficientnet_b0',
    hardware_name='KPU-T256',
    batch_size=16,
    precision=Precision.INT8,
    config=AnalysisConfig(
        run_roofline=True,
        run_energy=True,
        run_memory=True,
        run_concurrency=False,
        validate_consistency=False  # Skip for speed in batch operations
    )
)
```

### Method: `analyze_graph()`

Analyzes a PyTorch model directly (not from registry).

**Parameters:**
- `model` (nn.Module): PyTorch model instance
- `input_tensor` (torch.Tensor): Sample input tensor with correct shape/dtype
- `model_name` (str): Display name for the model
- `hardware_name` (str): Name of hardware target
- `batch_size` (int): Batch size (default: 1)
- `precision` (Precision): Precision to use (default: Precision.FP32)
- `config` (AnalysisConfig): Analysis configuration

**Returns:** `UnifiedAnalysisResult`

**Example:**
```python
import torch
import torch.nn as nn

# Define custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))

# Create and analyze
model = MyModel()
input_tensor = torch.randn(1, 3, 224, 224)

result = analyzer.analyze_graph(
    model=model,
    input_tensor=input_tensor,
    model_name='MyCustomModel',
    hardware_name='H100'
)
```

### Method: `analyze_fx_graph()`

Analyzes an already-traced FX graph (for advanced users).

**Parameters:**
- `fx_graph` (torch.fx.GraphModule): FX graph with shape propagation already done
- `model_name` (str): Display name for the model
- `hardware_name` (str): Name of hardware target
- `batch_size` (int): Batch size (default: 1)
- `precision` (Precision): Precision to use (default: Precision.FP32)
- `config` (AnalysisConfig): Analysis configuration

**Returns:** `UnifiedAnalysisResult`

**Example:**
```python
from torch.fx import symbolic_trace
from graphs.ir.shape_propagation import ShapeProp

# Trace model manually
fx_graph = symbolic_trace(model)
shape_prop = ShapeProp(fx_graph)
shape_prop.propagate(input_tensor)

# Analyze traced graph
result = analyzer.analyze_fx_graph(
    fx_graph=fx_graph,
    model_name='ResNet18',
    hardware_name='H100'
)
```

---

## AnalysisConfig

Control which analyzers run and how they behave.

```python
from graphs.analysis.unified_analyzer import AnalysisConfig

config = AnalysisConfig(
    # Which analyzers to run
    run_roofline=True,          # Latency and bottleneck analysis
    run_energy=True,            # Energy consumption analysis
    run_memory=True,            # Memory usage analysis
    run_concurrency=False,      # Parallelism analysis (slow, usually skip)

    # Partitioning strategy
    use_fusion_partitioning=True,  # Use fusion-based partitioning (recommended)

    # Validation
    validate_consistency=True,  # Cross-check analyzer outputs
)
```

**Analyzer Selection:**
- **run_roofline**: Enables roofline model for latency, bottlenecks, utilization
- **run_energy**: Enables three-component energy model (compute, memory, static)
- **run_memory**: Enables peak memory and memory timeline analysis
- **run_concurrency**: Enables parallelism analysis (expensive, usually not needed)

**Partitioning:**
- **use_fusion_partitioning**: Use fusion-based partitioning (creates larger, fused subgraphs)
  - True (default): Better performance, fewer subgraphs
  - False: Unfused partitioning, more subgraphs

**Validation:**
- **validate_consistency**: Cross-check results across analyzers
  - True (default): Verify FLOPs match across analyzers
  - False: Skip validation for speed (batch operations)

---

## UnifiedAnalysisResult

The result object contains all analysis data with convenient derived metrics.

### Metadata

```python
result.model_name           # str: Model name
result.hardware_name        # str: Hardware target name
result.batch_size          # int: Batch size used
result.precision           # Precision: FP32, FP16, or INT8
result.timestamp           # str: ISO format timestamp
```

### Phase 3 Reports

```python
result.roofline_report     # RooflineReport | None
result.energy_report       # EnergyReport | None
result.memory_report       # MemoryReport | None
result.concurrency_report  # ConcurrencyDescriptor | None
```

### Graph Structure

```python
result.partition_report    # PartitionReport: Graph structure
result.num_subgraphs      # int: Number of subgraphs
result.subgraphs          # List[SubgraphDescriptor]: Subgraph details
```

### Derived Metrics

Convenient top-level metrics computed from Phase 3 reports:

```python
# Performance
result.total_latency_ms         # float: Total latency in milliseconds
result.throughput_fps           # float: Throughput in FPS
result.hardware_utilization     # float: Hardware utilization (0-100%)

# Energy
result.total_energy_mj          # float: Total energy per batch (millijoules)
result.energy_per_inference_mj  # float: Energy per single inference
result.compute_energy_mj        # float: Compute energy component
result.memory_energy_mj         # float: Memory energy component
result.static_energy_mj         # float: Static/leakage energy

# Memory
result.peak_memory_mb           # float: Peak memory in megabytes
result.activation_memory_mb     # float: Activation memory
result.weight_memory_mb         # float: Weight memory
result.total_memory_mb          # float: Total memory

# Compute
result.total_flops             # int: Total FLOPs
result.total_gflops            # float: Total GFLOPs
```

### Usage Example

```python
result = analyzer.analyze_model('resnet18', 'H100', batch_size=16)

# Print key metrics
print(f"Model: {result.model_name}")
print(f"Hardware: {result.hardware_name}")
print(f"Batch Size: {result.batch_size}")
print(f"Precision: {result.precision.name}")
print()
print(f"Latency: {result.total_latency_ms:.2f} ms")
print(f"Throughput: {result.throughput_fps:.1f} fps")
print(f"Energy/inference: {result.energy_per_inference_mj:.1f} mJ")
print(f"Peak Memory: {result.peak_memory_mb:.1f} MB")
print(f"Utilization: {result.hardware_utilization:.1f}%")

# Access detailed reports
if result.roofline_report:
    for i, latency in enumerate(result.roofline_report.latencies):
        print(f"Subgraph {i}: {latency.actual_latency:.3f} ms ({latency.bottleneck})")

if result.energy_report:
    print(f"Compute energy: {result.energy_report.compute_energy_j * 1000:.1f} mJ")
    print(f"Memory energy: {result.energy_report.memory_energy_j * 1000:.1f} mJ")
    print(f"Static energy: {result.energy_report.static_energy_j * 1000:.1f} mJ")
```

---

## ReportGenerator API

Transform analysis results into multiple output formats.

### Basic Usage

```python
from graphs.reporting import ReportGenerator

generator = ReportGenerator(style='default')  # default, compact, or detailed

# Generate different formats
text_report = generator.generate_text_report(result)
json_report = generator.generate_json_report(result)
csv_report = generator.generate_csv_report(result)
markdown_report = generator.generate_markdown_report(result)
```

### Text Reports

```python
# Full report with all sections
text = generator.generate_text_report(result)

# Selective sections
text = generator.generate_text_report(
    result,
    include_sections=['executive', 'performance', 'energy'],  # Omit memory
    show_executive_summary=True
)

# Available sections: 'executive', 'performance', 'energy', 'memory', 'recommendations'
```

### JSON Reports

```python
# Full JSON with all data
json_str = generator.generate_json_report(
    result,
    include_raw_reports=True,  # Include full Phase 3 reports
    pretty_print=True          # Format with indentation
)

# Compact JSON (derived metrics only)
json_str = generator.generate_json_report(
    result,
    include_raw_reports=False,
    pretty_print=False
)

# Parse JSON
import json
data = json.loads(json_str)
latency = data['derived_metrics']['latency_ms']
```

### CSV Reports

```python
# Summary CSV (one row)
csv_str = generator.generate_csv_report(result)

# Detailed CSV with per-subgraph rows
csv_str = generator.generate_csv_report(
    result,
    include_subgraph_details=True
)

# Parse CSV
import csv
import io
reader = csv.DictReader(io.StringIO(csv_str))
rows = list(reader)
```

### Markdown Reports

```python
# Full markdown report
md = generator.generate_markdown_report(
    result,
    include_tables=True,  # Include comparison tables
    include_charts=False  # ASCII charts (not yet implemented)
)

# Write to file
with open('report.md', 'w') as f:
    f.write(md)
```

### Comparison Reports

Compare multiple results (batch sweeps, model comparison, hardware comparison).

```python
# Analyze multiple configurations
results = []
for batch_size in [1, 4, 8, 16]:
    result = analyzer.analyze_model('resnet18', 'H100', batch_size=batch_size)
    results.append(result)

# Generate comparison report
comparison = generator.generate_comparison_report(
    results,
    comparison_dimension='auto',  # Auto-detect what varies
    format='text',                # text, csv, json, markdown
    sort_by='latency'            # Sort by metric
)

print(comparison)
```

**Parameters:**
- `results`: List of UnifiedAnalysisResult objects
- `comparison_dimension`: What to compare
  - `'auto'`: Auto-detect (batch size, model, hardware, precision)
  - `'batch_size'`: Compare batch sizes
  - `'model'`: Compare models
  - `'hardware'`: Compare hardware targets
  - `'precision'`: Compare precisions
- `format`: Output format ('text', 'csv', 'json', 'markdown')
- `sort_by`: Sort results by metric ('latency', 'throughput', 'energy', 'memory')

### File I/O

```python
# Save report to file (format auto-detected from extension)
generator.save_report(result, 'report.json')    # JSON
generator.save_report(result, 'report.csv')     # CSV
generator.save_report(result, 'report.md')      # Markdown
generator.save_report(result, 'report.txt')     # Text

# Explicit format
generator.save_report(result, 'output.dat', format='json')

# Load result from JSON
loaded_result = generator.load_result('report.json')
```

---

## Common Workflows

### 1. Batch Size Sweep

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision

analyzer = UnifiedAnalyzer(verbose=False)
generator = ReportGenerator()

# Sweep batch sizes
results = []
batch_sizes = [1, 2, 4, 8, 16, 32]

for batch_size in batch_sizes:
    print(f"Analyzing batch size {batch_size}...")
    result = analyzer.analyze_model(
        'resnet18',
        'H100',
        batch_size=batch_size,
        config=AnalysisConfig(validate_consistency=False)  # Speed up
    )
    results.append(result)

# Generate comparison CSV
csv_output = generator.generate_comparison_report(
    results,
    format='csv',
    sort_by='latency'
)

# Save to file
with open('batch_sweep.csv', 'w') as f:
    f.write(csv_output)

# Find best configuration
best_throughput = max(results, key=lambda r: r.throughput_fps)
best_energy = min(results, key=lambda r: r.energy_per_inference_mj)

print(f"\nBest throughput: batch {best_throughput.batch_size} "
      f"({best_throughput.throughput_fps:.1f} fps)")
print(f"Best energy efficiency: batch {best_energy.batch_size} "
      f"({best_energy.energy_per_inference_mj:.1f} mJ/inference)")
```

### 2. Model Comparison

```python
models = ['resnet18', 'mobilenet_v2', 'efficientnet_b0']
hardware = 'Jetson-Orin-Nano'
batch_size = 8

results = []
for model in models:
    print(f"Analyzing {model}...")
    result = analyzer.analyze_model(model, hardware, batch_size=batch_size)
    results.append(result)

# Generate markdown comparison
md_output = generator.generate_comparison_report(
    results,
    format='markdown',
    sort_by='energy'
)

with open('model_comparison.md', 'w') as f:
    f.write(md_output)
```

### 3. Hardware Comparison

```python
model = 'resnet50'
hardware_targets = ['H100', 'Jetson-Orin-AGX', 'KPU-T256']
batch_size = 1

results = []
for hw in hardware_targets:
    print(f"Analyzing on {hw}...")
    result = analyzer.analyze_model(model, hw, batch_size=batch_size)
    results.append(result)

# Generate text comparison
text_output = generator.generate_comparison_report(
    results,
    format='text',
    sort_by='latency'
)

print(text_output)
```

### 4. Custom Analysis Pipeline

```python
import torch
import torch.nn as nn

# Define custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Analyze
model = MyModel()
input_tensor = torch.randn(1, 3, 224, 224)

result = analyzer.analyze_graph(
    model=model,
    input_tensor=input_tensor,
    model_name='MyCustomCNN',
    hardware_name='H100',
    batch_size=1,
    precision=Precision.FP32
)

# Generate comprehensive report
report = generator.generate_text_report(result)
print(report)

# Save JSON for later analysis
generator.save_report(result, 'custom_model_analysis.json')
```

### 5. Automated Hardware Selection

```python
def find_best_hardware(model_name, batch_size, metric='energy'):
    """Find best hardware for a model based on a metric."""

    hardware_targets = [
        'H100', 'A100', 'Jetson-Orin-AGX', 'Jetson-Orin-Nano',
        'KPU-T256', 'KPU-T64'
    ]

    results = {}
    for hw in hardware_targets:
        try:
            result = analyzer.analyze_model(model_name, hw, batch_size=batch_size)
            results[hw] = result
        except Exception as e:
            print(f"Skipping {hw}: {e}")

    # Find best by metric
    if metric == 'energy':
        best = min(results.items(), key=lambda x: x[1].energy_per_inference_mj)
    elif metric == 'latency':
        best = min(results.items(), key=lambda x: x[1].total_latency_ms)
    elif metric == 'throughput':
        best = max(results.items(), key=lambda x: x[1].throughput_fps)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    hw_name, result = best
    return hw_name, result

# Use it
best_hw, result = find_best_hardware('mobilenet_v2', batch_size=8, metric='energy')
print(f"Best hardware for energy efficiency: {best_hw}")
print(f"Energy per inference: {result.energy_per_inference_mj:.1f} mJ")
```

---

## Error Handling

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer

analyzer = UnifiedAnalyzer()

try:
    result = analyzer.analyze_model('resnet18', 'H100')
except ValueError as e:
    # Invalid model or hardware name
    print(f"Configuration error: {e}")
except RuntimeError as e:
    # Analysis failed (tracing error, etc.)
    print(f"Analysis error: {e}")
except Exception as e:
    # Unexpected error
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
```

**Common Errors:**
- `ValueError`: Invalid model name, hardware name, or configuration
- `RuntimeError`: FX tracing failed, shape propagation failed, or analysis error
- `AttributeError`: Missing required hardware capabilities

---

## Advanced: Custom Hardware

You can analyze with custom hardware targets not in the registry:

```python
from graphs.hardware.resource_model import HardwareResourceModel, Precision
from graphs.hardware.mappers.gpu import GPUMapper

# Define custom hardware
custom_hw = HardwareResourceModel(
    name='Custom-GPU',
    arch_type='GPU',
    peak_tflops_fp32=100.0,
    memory_bandwidth_gb_s=500.0,
    l2_cache_mb=10.0,
    precision_profiles={
        Precision.FP32: 1.0,
        Precision.FP16: 2.0,
        Precision.INT8: 4.0
    },
    tdp_watts=150.0
)

# Create mapper
custom_mapper = GPUMapper(custom_hw)

# Analyze with custom hardware
result = analyzer.analyze_model_with_custom_hardware(
    model_name='resnet18',
    hardware_mapper=custom_mapper,
    batch_size=1,
    precision=Precision.FP32
)
```

---

## Testing

The unified framework includes comprehensive tests:

```bash
# Unit tests
python -m pytest tests/analysis/test_unified_analyzer.py -v
python -m pytest tests/reporting/test_report_generator.py -v

# Integration tests
python -m pytest tests/integration/test_unified_workflows.py -v
python -m pytest tests/integration/test_cli_refactored.py -v
```

---

## Performance Tips

1. **Disable validation for batch operations:**
   ```python
   config = AnalysisConfig(validate_consistency=False)
   ```

2. **Skip concurrency analysis (expensive):**
   ```python
   config = AnalysisConfig(run_concurrency=False)
   ```

3. **Disable verbose output:**
   ```python
   analyzer = UnifiedAnalyzer(verbose=False)
   ```

4. **Reuse analyzer instance:**
   ```python
   analyzer = UnifiedAnalyzer()
   # Reuse for multiple analyses
   result1 = analyzer.analyze_model('resnet18', 'H100')
   result2 = analyzer.analyze_model('resnet50', 'H100')
   ```

---

## Migration from Phase 4.1

If you were using the Phase 4.1 analysis functions directly:

**Before (Phase 4.1):**
```python
from graphs.analysis.roofline_analyzer import RooflineAnalyzer
from graphs.analysis.energy_analyzer import EnergyAnalyzer
from graphs.analysis.memory_estimator import MemoryEstimator

# Manual orchestration
roofline_analyzer = RooflineAnalyzer(hardware, precision)
roofline_report = roofline_analyzer.analyze(subgraphs, partition_report)

energy_analyzer = EnergyAnalyzer(hardware, precision)
latencies = [lat.actual_latency for lat in roofline_report.latencies]
energy_report = energy_analyzer.analyze(subgraphs, partition_report, latencies=latencies)

memory_estimator = MemoryEstimator()
memory_report = memory_estimator.estimate(subgraphs)
```

**After (Phase 4.2):**
```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer

# Simplified orchestration
analyzer = UnifiedAnalyzer()
result = analyzer.analyze_model('resnet18', 'H100')

# Access reports
roofline_report = result.roofline_report
energy_report = result.energy_report
memory_report = result.memory_report

# Or use derived metrics
latency = result.total_latency_ms
energy = result.total_energy_mj
memory = result.peak_memory_mb
```

---

## Further Reading

- **CLI Tools**: See `cli/README.md` for command-line usage
- **Phase 3 Analyzers**: See `docs/sessions/` for analyzer details
- **Integration Tests**: See `tests/integration/` for usage examples
- **Refactoring Summary**: See `docs/sessions/2025-10-28_phase4_2_refactoring_summary.md`

---

## Support

For issues or questions:
- Check `tests/integration/test_unified_workflows.py` for comprehensive examples
- Review CLI tools (`cli/analyze_comprehensive_v2.py`, `cli/analyze_batch_v2.py`) for real-world usage
- File issues on GitHub
