# Efficiency Calibration Schema

This document describes the JSON schema for efficiency calibration files used in roofline-based latency estimation.

## Overview

Efficiency calibration files capture the relationship between operation size (FLOPs) and hardware efficiency. These curves are used by the roofline model to estimate actual achievable performance as a fraction of theoretical peak.

**Location**: `hardware_registry/<device_type>/<hardware_id>/efficiency_curves.json`

**Examples**:
- `hardware_registry/gpu/jetson_orin_agx/efficiency_curves.json`
- `hardware_registry/cpu/i7-12700K/efficiency_curves.json`

## Schema Version

Current version: `1.0`

## Top-Level Structure

```json
{
  "schema_version": "1.0",
  "hardware_id": "string",
  "hardware_name": "string",
  "device_type": "gpu|cpu|dsp|tpu|kpu",
  "precision": "FP32|FP16|INT8|TF32",
  "calibration_date": "ISO8601 datetime",
  "calibration_tool_version": "string",
  "curves": { ... },
  "bandwidth_curves": { ... },
  "validation": { ... } | null,
  "notes": "string"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version for compatibility checking |
| `hardware_id` | string | Unique identifier for the hardware (e.g., "jetson-orin-agx-50w") |
| `hardware_name` | string | Human-readable hardware name |
| `device_type` | string | One of: "gpu", "cpu", "dsp", "tpu", "kpu" |
| `precision` | string | Numeric precision: "FP32", "FP16", "INT8", "TF32" |
| `calibration_date` | string | ISO8601 timestamp of calibration |
| `calibration_tool_version` | string | Version of tool used for calibration |
| `curves` | object | Compute efficiency curves by operation type |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `bandwidth_curves` | object | Memory bandwidth efficiency curves |
| `validation` | object | Validation results against test models |
| `notes` | string | Free-form notes about calibration |

## Efficiency Curves

Each entry in `curves` maps an operation type to its efficiency curve:

```json
{
  "curves": {
    "<operation_type>": {
      "operation_type": "string",
      "description": "string",
      "data_points": [ ... ],
      "fitted_curve": { ... },
      "min_flops": number,
      "max_flops": number,
      "calibration_models": ["string"],
      "notes": "string"
    }
  }
}
```

### Operation Types

| Operation Type | Description | Typical Range |
|---------------|-------------|---------------|
| `conv2d` | Standard 2D convolution (no BN) | 50M - 10G FLOPs |
| `conv2d_batchnorm` | Conv2D fused with BatchNorm | 10M - 10G FLOPs |
| `conv2d_depthwise` | Depthwise separable convolution | 1M - 100M FLOPs |
| `mbconv` | MBConv-style blocks (EfficientNet) | 5M - 100M FLOPs |
| `matmul` | Matrix multiplication / Linear | 10M - 20G FLOPs |
| `generic` | Generic fallback for untyped ops | 500K - 1G FLOPs |
| `generic_tiny` | Tiny operations (launch dominated) | 500K - 10M FLOPs |

## Data Points

Each data point captures efficiency at a specific FLOP count with statistical properties:

```json
{
  "flops": 231000000,
  "efficiency_mean": 0.73,
  "efficiency_std": 0.146,
  "efficiency_min": 0.584,
  "efficiency_max": 0.876,
  "ci_lower": 0.444,
  "ci_upper": 1.016,
  "num_observations": 1,
  "source": "legacy_hardcoded|measured|interpolated",
  "source_subgraphs": ["ResNet-18 conv layer 231M"],
  "source_models": ["resnet18"]
}
```

### Statistical Fields

| Field | Type | Description |
|-------|------|-------------|
| `flops` | number | FLOP count for this data point |
| `efficiency_mean` | number | Mean efficiency (0.0 to ~5.0) |
| `efficiency_std` | number | Standard deviation |
| `efficiency_min` | number | Minimum observed efficiency |
| `efficiency_max` | number | Maximum observed efficiency |
| `ci_lower` | number | 95% confidence interval lower bound |
| `ci_upper` | number | 95% confidence interval upper bound |
| `num_observations` | integer | Number of measurements averaged |

### Provenance Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Origin: "legacy_hardcoded", "measured", "interpolated" |
| `source_subgraphs` | [string] | Subgraph patterns that contributed to this measurement |
| `source_models` | [string] | Models used for calibration |

### Efficiency Interpretation

Efficiency is the ratio of achieved performance to theoretical peak:

```
efficiency = achieved_flops / theoretical_peak_flops
```

**Efficiency > 1.0**: Indicates the theoretical peak is conservative. This often occurs with very large operations where:
- Caches are fully utilized
- Memory access patterns are optimal
- Tensor cores or specialized units exceed baseline FP32 calculations

**Typical Ranges by Operation Type**:
- Depthwise: 0.02 - 0.03 (severely memory-bound)
- MBConv: 0.025 - 0.10 (mixed small ops)
- Tiny ops: 0.01 - 0.06 (kernel launch dominated)
- Conv2D+BN: 0.06 - 5.0 (size dependent)
- MatMul: 0.20 - 1.5 (size dependent)

## Fitted Curves

Fitted curves provide interpolation between data points:

```json
{
  "fitted_curve": {
    "curve_type": "piecewise_linear|log_linear|sigmoid",
    "breakpoints": [[flops1, eff1], [flops2, eff2], ...],
    "r_squared": 0.95,
    "residual_std": 0.02,
    "max_residual": 0.05
  }
}
```

### Curve Types

| Type | Description | Use Case |
|------|-------------|----------|
| `piecewise_linear` | Linear interpolation in log(FLOPs) space | Default for most operations |
| `log_linear` | Single log-linear fit | Smooth monotonic curves |
| `sigmoid` | S-curve fit | Saturation behavior |

### Interpolation Method

For `piecewise_linear`:
1. Convert FLOP count to log scale: `log_flops = log10(flops)`
2. Find bracketing breakpoints
3. Linear interpolate efficiency between breakpoints
4. Clamp to `[min_flops, max_flops]` range for extrapolation

```python
def interpolate(self, flops: float) -> float:
    log_flops = math.log10(flops)
    for i in range(len(breakpoints) - 1):
        log_bp1 = math.log10(breakpoints[i][0])
        log_bp2 = math.log10(breakpoints[i+1][0])
        if log_bp1 <= log_flops <= log_bp2:
            t = (log_flops - log_bp1) / (log_bp2 - log_bp1)
            return breakpoints[i][1] + t * (breakpoints[i+1][1] - breakpoints[i][1])
    # Extrapolation: clamp to nearest endpoint
    return breakpoints[0][1] if flops < min_flops else breakpoints[-1][1]
```

## Bandwidth Curves

Memory bandwidth efficiency curves follow the same structure as compute curves, but measure bandwidth efficiency:

```json
{
  "bandwidth_curves": {
    "generic": {
      "operation_type": "generic",
      "description": "Generic bandwidth efficiency for memory-bound operations",
      "data_points": [...],
      "fitted_curve": {...}
    }
  }
}
```

Bandwidth efficiency = achieved_bandwidth / theoretical_peak_bandwidth

## Validation Results

Optional validation against held-out models:

```json
{
  "validation": {
    "models_tested": 22,
    "mape": 26.2,
    "max_error": 49.6,
    "breakdown": {
      "excellent": 3,
      "good": 7,
      "fair": 12,
      "poor": 0
    }
  }
}
```

## Example: Complete File

```json
{
  "schema_version": "1.0",
  "hardware_id": "i7-12700K",
  "hardware_name": "Intel Core i7-12700K (Alder Lake)",
  "device_type": "cpu",
  "precision": "FP32",
  "calibration_date": "2026-02-04T16:44:48.680452",
  "calibration_tool_version": "extract_efficiency_curves.py v1.0",
  "curves": {
    "generic": {
      "operation_type": "generic",
      "description": "Generic CPU efficiency - operation size dependent",
      "data_points": [
        {
          "flops": 1000000.0,
          "efficiency_mean": 0.15,
          "efficiency_std": 0.03,
          "efficiency_min": 0.12,
          "efficiency_max": 0.18,
          "ci_lower": 0.0912,
          "ci_upper": 0.2088,
          "num_observations": 1,
          "source": "legacy_hardcoded",
          "source_subgraphs": ["tiny 1M"],
          "source_models": []
        },
        {
          "flops": 100000000.0,
          "efficiency_mean": 0.7,
          "efficiency_std": 0.14,
          "efficiency_min": 0.56,
          "efficiency_max": 0.84,
          "ci_lower": 0.4256,
          "ci_upper": 0.9744,
          "num_observations": 1,
          "source": "legacy_hardcoded",
          "source_subgraphs": ["large 100M"],
          "source_models": []
        }
      ],
      "fitted_curve": {
        "curve_type": "piecewise_linear",
        "breakpoints": [
          [1000000.0, 0.15],
          [10000000.0, 0.25],
          [100000000.0, 0.7],
          [500000000.0, 1.0]
        ],
        "r_squared": 0.0,
        "residual_std": 0.0,
        "max_residual": 0.0
      },
      "min_flops": 1000000.0,
      "max_flops": 1000000000.0,
      "calibration_models": ["mobilenet_v2", "resnet18", "vit_b_16"],
      "notes": "CPU efficiency: MobileNet (9M avg) -> 0.17, ResNet (113M avg) -> 0.85"
    }
  },
  "bandwidth_curves": {},
  "validation": null,
  "notes": "Extracted from roofline.py hard-coded values."
}
```

## Related Tools

| Tool | Purpose |
|------|---------|
| `cli/extract_efficiency_curves.py` | Extract legacy hard-coded values to JSON |
| `cli/measure_efficiency.py` | Measure per-subgraph efficiency (Phase 2) |
| `cli/aggregate_efficiency.py` | Aggregate measurements into curves (Phase 2) |
| `cli/validate_calibration.py` | Validate calibration accuracy (Phase 4) |

## Migration Notes

### From Legacy Hard-Coded Values

Files with `"source": "legacy_hardcoded"` were extracted from `roofline.py` using `extract_efficiency_curves.py`. These values:

1. Have estimated uncertainty (20% relative standard deviation)
2. May have single observations (`num_observations: 1`)
3. Should be replaced with measured values during Phase 2 calibration

### Updating Calibration

To update calibration with measured values:

1. Run `measure_efficiency.py` on target hardware with multiple models
2. Run `aggregate_efficiency.py` to combine measurements
3. Review statistical properties (std, CI) for confidence
4. Update the efficiency_curves.json file
5. Run `validate_calibration.py` to verify accuracy improvement

## References

- [Efficiency Externalization Roadmap](../roadmaps/efficiency_externalization_roadmap.md)
- [Efficiency Derate Analysis](../design/efficiency_derate_analysis.md)
- [Calibration Coverage Analyzer](../design/calibration_coverage_analyzer.md)
