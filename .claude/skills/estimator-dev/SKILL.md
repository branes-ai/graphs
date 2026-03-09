---
name: estimator-dev
description: Guides development of new performance estimators. Use when creating or extending estimators for latency, energy, memory, throughput, or any quantitative analysis module in the estimation/ package.
argument-hint: "[estimator-name] [description]"
---

# New Estimator Development

Develop a new estimator for the graphs analytical framework: $ARGUMENTS

## Mandatory Workflow

### Phase 1: Scope and Interface

Before writing code, define:

1. **What is being estimated?** (latency, energy, memory, throughput, utilization, EDP, etc.)
2. **What inputs does it need?** (PartitionReport, HardwareMapper, RooflineReport, etc.)
3. **What confidence level applies?** Which estimates are CALIBRATED vs THEORETICAL?
4. **How does it compose?** Does it consume other estimator outputs? Does UnifiedAnalyzer need it?

Output a specification:
```
Estimator: <Name>Analyzer
Inputs:  PartitionReport, [other dependencies]
Outputs: <Name>Report (per-subgraph <Name>Descriptor + aggregates)
Confidence: THEORETICAL (no calibration data yet)
Composes with: [which existing estimators it depends on or feeds into]
```

### Phase 2: Descriptor and Report Design

Define the data structures following existing patterns:

1. Create a `<Name>Descriptor` (per-subgraph result) with:
   - All quantitative fields as floats with units documented
   - `confidence: EstimationConfidence` field
   - Named tuple or dataclass (match existing style)

2. Create a `<Name>Report` (aggregate) with:
   - `descriptors: List[<Name>Descriptor]` (per-subgraph)
   - Aggregate totals and derived metrics
   - `model_name`, `hardware_name` metadata

### Phase 3: Implementation

Create `src/graphs/estimation/<name>.py`:

1. Follow the `Analyzer` class pattern from `roofline.py` or `energy.py`
2. Main method: `analyze(partition_report, ...) -> <Name>Report`
3. Per-subgraph analysis in a private method
4. Confidence assignment based on data source
5. No Unicode in code or output

### Phase 4: Integration

1. Add to `estimation/__init__.py` exports
2. If appropriate, integrate with `UnifiedAnalyzer`:
   - Add to `AnalysisConfig` flags
   - Add report field to `UnifiedAnalysisResult`
   - Wire into `analyze_model()` pipeline
3. Add CLI tool in `cli/analyze_<name>.py` if standalone use is valuable

### Phase 5: Validation

1. Create `validation/estimators/test_<name>.py`
2. Test against known models (resnet18, mobilenet_v2)
3. Verify confidence levels are set correctly
4. Cross-check against existing estimators where possible (e.g., energy should be consistent with latency * power)

## Reference: Existing Estimator Interfaces

```python
# Pattern to follow (from roofline.py):
class RooflineAnalyzer:
    def __init__(self, verbose=False): ...
    def analyze(self, partition_report, mapper, precision=Precision.FP32) -> RooflineReport: ...

# Pattern to follow (from energy.py):
class EnergyAnalyzer:
    def __init__(self, verbose=False): ...
    def analyze(self, partition_report, roofline_report, mapper) -> EnergyReport: ...
```
