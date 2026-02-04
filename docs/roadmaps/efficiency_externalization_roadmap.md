# Roadmap: Efficiency Factor Externalization and Validation

## Overview

This roadmap describes the work to externalize hard-coded efficiency derates from `roofline.py` into auditable, calibration-driven data files with proper statistical characterization.

**Goals**:
1. Move efficiency parameters from code to versioned data files
2. Capture statistical properties (mean, variance, confidence intervals)
3. Build tools to measure, audit, and validate efficiency factors
4. Enable per-hardware, per-operation calibration without code changes

**Success Criteria**:
- No hard-coded efficiency values in estimation code
- All efficiency curves have documented calibration sources
- Statistical uncertainty quantified for all parameters
- Validation MAPE < 20% for calibrated hardware

---

## Phase 1: Data Model and Schema (Week 1)

### 1.1 Define Efficiency Curve Schema

**Deliverable**: JSON schema for efficiency calibration files

```json
{
  "$schema": "efficiency_calibration_v1.json",
  "hardware_id": "string",
  "precision": "FP32|FP16|INT8",
  "calibration_metadata": {
    "date": "ISO8601",
    "tool_version": "string",
    "models_used": ["string"],
    "num_samples": "integer"
  },
  "efficiency_curves": {
    "<operation_type>": {
      "data_points": [
        {
          "flops": "number",
          "efficiency_mean": "number",
          "efficiency_std": "number",
          "efficiency_min": "number",
          "efficiency_max": "number",
          "num_observations": "integer",
          "source_subgraphs": ["string"]
        }
      ],
      "fitted_curve": {
        "type": "piecewise_linear|log_linear|sigmoid",
        "parameters": {},
        "r_squared": "number",
        "residual_std": "number"
      }
    }
  }
}
```

**Tasks**:
- [ ] Design JSON schema with statistical fields
- [ ] Define operation type taxonomy (conv2d, conv2d_depthwise, matmul, etc.)
- [ ] Create schema validation utility
- [ ] Document schema in `docs/schemas/efficiency_calibration.md`

### 1.2 Extract Current Hard-Coded Values

**Deliverable**: Initial efficiency curve files for existing hardware

**Tasks**:
- [ ] Extract GPU efficiency curves from `roofline.py` lines 537-827
- [ ] Extract CPU efficiency curves from `roofline.py` lines 788-823
- [ ] Extract bandwidth efficiency curves from `roofline.py` lines 829-933
- [ ] Create `hardware_registry/gpu/jetson_orin_agx/efficiency_curves.json`
- [ ] Create `hardware_registry/cpu/i7-12700K/efficiency_curves.json`
- [ ] Mark all values with `"source": "legacy_hardcoded"` for audit trail

---

## Phase 2: Measurement Infrastructure (Week 2)

### 2.1 Per-Subgraph Efficiency Measurement Tool

**Deliverable**: `cli/measure_efficiency.py`

Measures actual efficiency for each subgraph in a model:
```bash
./cli/measure_efficiency.py --model resnet18 --hardware i7-12700K --device cpu \
    --output measurements/resnet18_i7_efficiency.json
```

**Output per subgraph**:
```json
{
  "subgraph_id": 5,
  "fusion_pattern": "Conv2d_BatchNorm2d_ReLU",
  "operation_type": "conv2d",
  "flops": 231000000,
  "theoretical_peak_flops": 1280000000000,
  "measured_latency_ms": {
    "mean": 0.52,
    "std": 0.03,
    "min": 0.48,
    "max": 0.61,
    "samples": 50
  },
  "achieved_flops": 444230769230,
  "efficiency": {
    "mean": 0.347,
    "std": 0.021,
    "ci_95_lower": 0.312,
    "ci_95_upper": 0.382
  }
}
```

**Tasks**:
- [ ] Extend `TimingInterpreter` to collect multiple samples per node
- [ ] Compute per-subgraph statistics (mean, std, min, max, CI)
- [ ] Map subgraph to operation type using fusion pattern
- [ ] Output structured JSON with full statistical characterization
- [ ] Add `--runs` parameter for configurable sample size (default: 50)

### 2.2 Efficiency Aggregation Tool

**Deliverable**: `cli/aggregate_efficiency.py`

Aggregates measurements across multiple models into efficiency curves:
```bash
./cli/aggregate_efficiency.py \
    --input measurements/resnet18_i7_efficiency.json \
    --input measurements/resnet50_i7_efficiency.json \
    --input measurements/vgg16_i7_efficiency.json \
    --output hardware_registry/cpu/i7-12700K/efficiency_curves.json
```

**Statistical aggregation**:
- Group by operation type and FLOP range (log-scale bins)
- Compute pooled mean and variance across observations
- Fit efficiency curves with uncertainty bounds
- Report goodness-of-fit metrics (R², residual std)

**Tasks**:
- [ ] Implement FLOP binning (logarithmic: 1M, 3M, 10M, 30M, 100M, etc.)
- [ ] Implement pooled variance calculation across models
- [ ] Implement weighted least squares curve fitting
- [ ] Compute confidence intervals for fitted curves
- [ ] Output fitted parameters with statistical metadata

---

## Phase 3: Statistical Analysis Tools (Week 3)

### 3.1 Efficiency Distribution Analyzer

**Deliverable**: `cli/analyze_efficiency_distribution.py`

Analyzes statistical properties of efficiency measurements:
```bash
./cli/analyze_efficiency_distribution.py \
    --calibration hardware_registry/cpu/i7-12700K/efficiency_curves.json \
    --output reports/i7_efficiency_analysis.md
```

**Statistical outputs**:
- Per-operation-type distributions (histograms, Q-Q plots)
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Heteroscedasticity analysis (does variance change with FLOP size?)
- Outlier detection (>3 sigma from fitted curve)
- Cross-model consistency (do different models agree?)

**Tasks**:
- [ ] Implement distribution fitting (normal, log-normal, beta)
- [ ] Implement normality tests with p-values
- [ ] Implement heteroscedasticity tests (Breusch-Pagan)
- [ ] Generate statistical summary tables
- [ ] Generate distribution plots (matplotlib/ASCII fallback)

### 3.2 Uncertainty Propagation in Estimation

**Deliverable**: Modify `RooflineAnalyzer` to propagate uncertainty

Current: Returns point estimate of latency
Proposed: Returns latency with confidence interval

```python
@dataclass
class LatencyEstimate:
    latency_ms: float           # Point estimate
    latency_std_ms: float       # Standard deviation
    ci_95_lower_ms: float       # 95% CI lower bound
    ci_95_upper_ms: float       # 95% CI upper bound
    efficiency_used: float      # Efficiency factor applied
    efficiency_std: float       # Uncertainty in efficiency
```

**Tasks**:
- [ ] Add efficiency uncertainty to `EfficiencyCurve` data structure
- [ ] Implement uncertainty propagation in `_analyze_subgraph()`
- [ ] Propagate variance through sum (total latency = sum of subgraphs)
- [ ] Add confidence intervals to `LatencyDescriptor`
- [ ] Update reports to show uncertainty bounds

---

## Phase 4: Calibration Workflow (Week 4)

### 4.1 Automated Calibration Pipeline

**Deliverable**: `cli/calibrate_hardware.py`

End-to-end calibration for a new hardware target:
```bash
./cli/calibrate_hardware.py --hardware-name "AMD-Ryzen-9-7950X" \
    --device cpu \
    --models resnet18,resnet50,vgg16,mobilenet_v2,vit_b_16 \
    --runs-per-model 50 \
    --output hardware_registry/cpu/amd_ryzen_9_7950x/
```

**Pipeline steps**:
1. Run each model with timing instrumentation
2. Collect per-subgraph efficiency measurements
3. Aggregate into efficiency curves by operation type
4. Fit parametric curves with uncertainty
5. Validate against held-out models
6. Generate calibration report

**Tasks**:
- [ ] Implement model selection for calibration coverage
- [ ] Implement train/validation split for curve fitting
- [ ] Implement automatic operation type detection
- [ ] Generate calibration report with statistics
- [ ] Create default model set for balanced calibration

### 4.2 Calibration Validation Tool

**Deliverable**: `cli/validate_calibration.py`

Validates calibration quality against independent test set:
```bash
./cli/validate_calibration.py \
    --calibration hardware_registry/cpu/i7-12700K/efficiency_curves.json \
    --test-models efficientnet_b0,efficientnet_b1,maxvit_t \
    --device cpu
```

**Outputs**:
- Per-model estimation error
- Per-operation-type error breakdown
- Statistical tests (is error zero-mean? is variance as predicted?)
- Recommendations (which operations need more calibration?)

**Tasks**:
- [ ] Implement leave-one-out cross-validation
- [ ] Implement error decomposition by operation type
- [ ] Implement bias tests (t-test for zero mean error)
- [ ] Implement variance calibration tests (chi-squared)
- [ ] Generate actionable recommendations

---

## Phase 5: Audit and Visualization (Week 5)

### 5.1 Efficiency Curve Visualization Tool

**Deliverable**: `cli/visualize_efficiency.py`

Generates visual reports of efficiency curves:
```bash
./cli/visualize_efficiency.py \
    --calibration hardware_registry/cpu/i7-12700K/efficiency_curves.json \
    --output reports/i7_efficiency_curves.html
```

**Visualizations**:
- Efficiency vs FLOP size (log-log plot) with confidence bands
- Residual plots (measured - predicted)
- Q-Q plots for residual normality
- Comparison across hardware (overlay multiple calibrations)
- Heatmap of operation type x FLOP bin coverage

**Tasks**:
- [ ] Implement matplotlib plotting functions
- [ ] Implement ASCII fallback for terminal display
- [ ] Implement HTML report generation
- [ ] Add comparison mode (multiple hardware on one plot)
- [ ] Add coverage heatmap (which bins have data?)

### 5.2 Calibration Audit Report

**Deliverable**: `cli/audit_calibration.py`

Generates audit report for calibration data:
```bash
./cli/audit_calibration.py --hardware i7-12700K \
    --output reports/i7_calibration_audit.md
```

**Audit contents**:
- Data provenance (which models, when, how many samples)
- Coverage analysis (which operations/sizes have data?)
- Statistical quality (sample sizes, variance estimates)
- Comparison to legacy hard-coded values
- Recommendations for additional calibration

**Tasks**:
- [ ] Implement provenance tracking
- [ ] Implement coverage gap analysis
- [ ] Implement statistical quality metrics
- [ ] Implement legacy comparison report
- [ ] Generate markdown audit report

---

## Phase 6: Integration and Migration (Week 6)

### 6.1 Refactor RooflineAnalyzer

**Deliverable**: Modified `roofline.py` that loads efficiency from data files

**Tasks**:
- [ ] Create `EfficiencyCurveLoader` class
- [ ] Implement fallback chain: calibration file -> legacy hard-coded -> conservative default
- [ ] Remove hard-coded efficiency values from `_get_compute_efficiency_scale()`
- [ ] Remove hard-coded efficiency values from `_get_bandwidth_efficiency_scale()`
- [ ] Add logging when using fallback values
- [ ] Maintain backward compatibility (same API)

### 6.2 Migration and Deprecation

**Deliverable**: Clean migration path from hard-coded to data-driven

**Tasks**:
- [ ] Add deprecation warnings for legacy code paths
- [ ] Create migration guide document
- [ ] Update all CLI tools to use new calibration system
- [ ] Update tests to use calibration files
- [ ] Remove legacy hard-coded values (after validation)

---

## Phase 7: Documentation and Release (Week 7)

### 7.1 User Documentation

**Tasks**:
- [ ] Write calibration workflow guide
- [ ] Write efficiency curve schema documentation
- [ ] Write tool usage examples
- [ ] Update CLAUDE.md with new tools
- [ ] Create troubleshooting guide

### 7.2 Developer Documentation

**Tasks**:
- [ ] Document statistical methods used
- [ ] Document curve fitting algorithms
- [ ] Document uncertainty propagation math
- [ ] Create architecture diagram
- [ ] Write contributing guide for adding new hardware

---

## Summary: Deliverables by Phase

| Phase | Deliverables | Key Statistical Features |
|-------|--------------|-------------------------|
| 1 | Schema, initial data files | Statistical fields in schema |
| 2 | `measure_efficiency.py`, `aggregate_efficiency.py` | Per-sample stats, pooled variance |
| 3 | `analyze_efficiency_distribution.py`, uncertainty propagation | Normality tests, heteroscedasticity, CI |
| 4 | `calibrate_hardware.py`, `validate_calibration.py` | Cross-validation, bias/variance tests |
| 5 | `visualize_efficiency.py`, `audit_calibration.py` | Confidence bands, residual plots |
| 6 | Refactored `roofline.py` | Uncertainty in estimates |
| 7 | Documentation | Statistical methods guide |

---

## Statistical Methods Summary

### Efficiency Measurement Statistics

For each subgraph measurement:
- **Mean**: $\bar{\eta} = \frac{1}{n}\sum_{i=1}^{n} \eta_i$
- **Variance**: $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(\eta_i - \bar{\eta})^2$
- **Standard Error**: $SE = \frac{s}{\sqrt{n}}$
- **95% CI**: $\bar{\eta} \pm 1.96 \cdot SE$

### Pooled Statistics Across Models

For aggregating efficiency across multiple models at same FLOP level:
- **Pooled Mean**: $\bar{\eta}_{pooled} = \frac{\sum_j n_j \bar{\eta}_j}{\sum_j n_j}$
- **Pooled Variance**: $s^2_{pooled} = \frac{\sum_j (n_j - 1)s^2_j + \sum_j n_j(\bar{\eta}_j - \bar{\eta}_{pooled})^2}{\sum_j n_j - 1}$

### Curve Fitting

Weighted least squares with weights $w_i = 1/s^2_i$:
- Minimize: $\sum_i w_i (y_i - f(x_i; \theta))^2$
- Report: $R^2$, residual standard deviation, parameter confidence intervals

### Uncertainty Propagation

For latency = sum of subgraph latencies:
- **Total variance**: $\text{Var}(L_{total}) = \sum_i \text{Var}(L_i)$ (assuming independence)
- **Total CI**: Derived from total variance

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Calibration takes too long | Provide "quick" mode with fewer samples |
| Statistical methods too complex | Provide sensible defaults, hide complexity |
| Backward compatibility breaks | Maintain fallback to legacy values |
| Overfitting to calibration models | Use cross-validation, report generalization error |
| Users don't calibrate | Ship pre-calibrated files for common hardware |
