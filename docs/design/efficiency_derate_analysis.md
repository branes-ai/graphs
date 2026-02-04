# Analysis: Hard-Coded Efficiency Derates in Latency Estimation

## Executive Summary

Your concern is well-founded. The current implementation contains **extensive hard-coded efficiency derates** that:

1. Are calibrated primarily on **Jetson Orin AGX 50W** for GPU
2. Have limited calibration data for **CPU** (only i7-12700K with sparse data points)
3. Embed assumptions that may not transfer across hardware
4. Are difficult to validate, maintain, and improve systematically

This document analyzes the current state and proposes a more principled approach.

---

## Current State: Where Are the Hard-Coded Values?

### GPU Efficiency Model (`roofline.py`, lines 537-827)

**Total hard-coded parameters: ~50+**

| Size Range | Pattern | Scale Factor | Calibration Source |
|------------|---------|--------------|-------------------|
| All | Depthwise | 0.03 | Jetson AGX 50W |
| <10M | MBConv | 0.025 | EfficientNet-B0 |
| 10-30M | MBConv | 0.025-0.035 | EfficientNet-B0/B1 |
| 30-50M | MBConv | 0.035-0.055 | EfficientNet-B1/B2 |
| >50M | MBConv | 0.055-0.10 | Extrapolated |
| <1M | Tiny | 0.01 | Assumed |
| 1-10M | Small | 0.02-0.06 | MobileNet-V2 |
| 10-50M | Small-Med | 0.06-0.20 | MobileNet-V2 |
| 50-200M | Conv2D+BN | 0.23-0.54 | ResNet-18 |
| 50-200M | Conv2D only | 0.60-1.13 | ResNet-18 |
| 200-500M | Conv2D+BN | 0.73-0.80 | Extrapolated |
| 500M-5G | Conv2D+BN | 0.80-4.50 | VGG |
| >5G | Conv2D+BN | 4.50-5.00 | VGG |

**BatchNorm fusion factor**: 0.67 (single calibration point: ResNet-18 231M Conv)

### CPU Efficiency Model (`roofline.py`, lines 788-823)

**Total hard-coded parameters: ~8**

| Size Range | Scale Factor | Calibration Source |
|------------|--------------|-------------------|
| <1M | 0.15 | Assumed |
| 1-10M | 0.15-0.25 | Sparse |
| 10-100M | 0.25-0.70 | MobileNet, ResNet |
| >100M | 0.70-1.00 | ViT |

**Comments claim calibration on**:
- MobileNet (9M avg): need ~0.17 scale
- ResNet (113M avg): scale ~0.85 works
- ViT (225M avg): scale ~1.0 works

---

## Validation Results Analysis

From your test run on i7-12700K:

### Pattern 1: Large Convolution Models OVERESTIMATED

| Model | Error | Avg Subgraph Size | Observed Pattern |
|-------|-------|-------------------|------------------|
| VGG-11 | +42.7% | ~100-500M | Large convs |
| VGG-16 | +45.2% | ~100-500M | Large convs |
| VGG-19 | +49.6% | ~100-500M | Large convs |
| ResNet-34 | +37.8% | ~50-230M | Standard convs |
| ResNet-50 | +25.6% | ~50-230M | Bottleneck convs |
| ResNet-101 | +31.9% | ~50-230M | Bottleneck convs |
| ResNet-152 | +35.8% | ~50-230M | Bottleneck convs |

**Diagnosis**: The CPU efficiency model says large ops (>100M) achieve 0.70-1.0 scale, but measured data shows they achieve **better** efficiency (maybe 1.2-1.5x).

### Pattern 2: Efficient Models with Small Ops UNDERESTIMATED

| Model | Error | Avg Subgraph Size | Observed Pattern |
|-------|-------|-------------------|------------------|
| MobileNet-V3-Small | -37.4% | ~2-15M | Depthwise/Pointwise |
| EfficientNet-B0 | -35.8% | ~10-30M | MBConv blocks |
| EfficientNet-B1 | -29.7% | ~15-40M | MBConv blocks |
| MobileNet-V2 | -11.9% | ~5-20M | Depthwise/Pointwise |

**Diagnosis**: The CPU efficiency model says small ops (10-100M) achieve 0.25-0.70 scale, but measured data shows they achieve **worse** efficiency (maybe 0.15-0.40x).

### Pattern 3: Transformer Models ACCURATE

| Model | Error | Avg Subgraph Size | Observed Pattern |
|-------|-------|-------------------|------------------|
| ViT-B/16 | +12.6% | ~225M | MatMul/Linear |
| ViT-B/32 | +8.3% | ~225M | MatMul/Linear |
| ViT-L/16 | +1.3% | ~900M | MatMul/Linear |
| ViT-L/32 | -11.1% | ~900M | MatMul/Linear |

**Diagnosis**: The calibration points used (ViT models) are accurate.

---

## The Fundamental Problem

### Issue 1: Sparse Calibration, Dense Interpolation

The current approach:
1. Measures 3-5 models on target hardware
2. Extracts "representative" efficiency values
3. Creates piecewise-linear interpolation curves
4. Applies to ALL models

**Problem**: The interpolation assumes smooth, monotonic efficiency curves. Reality is more complex:
- **Operation type matters**: Conv2D vs MatMul vs Depthwise have different curves
- **Memory access patterns matter**: Strided vs contiguous
- **Cache effects**: Working set vs L2/L3 size
- **Compiler/library version**: cuDNN, MKL-DNN optimizations vary

### Issue 2: No Feedback Loop

Hard-coded values cannot self-correct. When we discover the values are wrong:
1. Developer manually adjusts curve parameters
2. Fixes one model, may break others
3. No systematic way to know which models are affected
4. Values accumulate as "tech debt"

### Issue 3: Hardware-Specific Values Applied Broadly

The GPU model is calibrated on **Jetson Orin AGX 50W**, but applied to:
- H100 (different architecture, different efficiency curve)
- A100 (different architecture)
- Other Jetson variants (different thermal profiles)

The CPU model is calibrated on **i7-12700K**, but applied to:
- AMD EPYC (different microarchitecture)
- Ampere Altra (ARM, completely different)
- Xeon (different cache hierarchy)

---

## Constructive Proposal: Calibration-Driven Efficiency Model

### Design Principle: Separate Data from Code

**Current** (problematic):
```python
# Hard-coded in roofline.py
if flops < 10e6:
    return 0.15 + 0.10 * t  # Magic numbers embedded in code
```

**Proposed** (calibration-driven):
```python
# Load from hardware-specific calibration file
efficiency_curve = self.calibration.get_efficiency_curve(
    hardware=self.hardware_name,
    operation_type=sg.operation_type,
    precision=self.precision
)
return efficiency_curve.interpolate(sg.flops)
```

### Data Structure: Efficiency Calibration Profile

```json
{
  "hardware": "i7-12700K",
  "precision": "FP32",
  "calibration_date": "2026-02-04",
  "calibration_tool_version": "2.0.0",

  "efficiency_curves": {
    "conv2d": {
      "description": "Standard 2D convolution",
      "calibration_models": ["resnet18", "resnet50", "vgg16"],
      "data_points": [
        {"flops": 1e6, "measured_efficiency": 0.12, "std_dev": 0.03},
        {"flops": 10e6, "measured_efficiency": 0.25, "std_dev": 0.04},
        {"flops": 50e6, "measured_efficiency": 0.55, "std_dev": 0.05},
        {"flops": 100e6, "measured_efficiency": 0.75, "std_dev": 0.06},
        {"flops": 500e6, "measured_efficiency": 1.10, "std_dev": 0.08}
      ],
      "interpolation": "log_linear",
      "extrapolation": "clamp"
    },
    "conv2d_depthwise": {
      "description": "Depthwise separable convolution",
      "calibration_models": ["mobilenet_v2", "efficientnet_b0"],
      "data_points": [
        {"flops": 1e6, "measured_efficiency": 0.05, "std_dev": 0.02},
        {"flops": 5e6, "measured_efficiency": 0.08, "std_dev": 0.02},
        {"flops": 10e6, "measured_efficiency": 0.10, "std_dev": 0.03}
      ]
    },
    "matmul": {
      "description": "Matrix multiplication / Linear layers",
      "calibration_models": ["vit_b_16", "vit_l_16"],
      "data_points": [
        {"flops": 10e6, "measured_efficiency": 0.30, "std_dev": 0.05},
        {"flops": 100e6, "measured_efficiency": 0.70, "std_dev": 0.06},
        {"flops": 1e9, "measured_efficiency": 0.95, "std_dev": 0.05}
      ]
    }
  },

  "validation_results": {
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

### Implementation Phases

#### Phase 1: Extract Current Values to Data Files

Move hard-coded curves to JSON files in `hardware_registry/`:
- `hardware_registry/cpu/i7-12700K/efficiency_curves.json`
- `hardware_registry/gpu/jetson_orin_agx/efficiency_curves.json`

**Benefits**:
- Values visible and auditable
- Easy to update without code changes
- Can version and track changes over time

#### Phase 2: Per-Operation Calibration Tool

Create `cli/calibrate_efficiency.py` that:
1. Runs a model on target hardware
2. Measures per-subgraph actual latency (using TimingInterpreter)
3. Computes achieved efficiency = measured_flops / theoretical_peak
4. Updates efficiency curve data points

**Example workflow**:
```bash
# Calibrate Conv2D efficiency on i7-12700K
./cli/calibrate_efficiency.py --model resnet18,resnet50,vgg16 \
    --hardware i7-12700K --operation conv2d

# Calibrate depthwise efficiency
./cli/calibrate_efficiency.py --model mobilenet_v2,efficientnet_b0 \
    --hardware i7-12700K --operation conv2d_depthwise

# Calibrate MatMul efficiency
./cli/calibrate_efficiency.py --model vit_b_16,vit_l_16 \
    --hardware i7-12700K --operation matmul
```

#### Phase 3: Automatic Curve Fitting

Given calibration data points, automatically fit efficiency curves:
1. Try multiple curve shapes (linear, log-linear, piecewise, sigmoid)
2. Select best fit based on cross-validation error
3. Store fitted parameters with uncertainty bounds

#### Phase 4: Validation Integration

After each calibration run:
1. Run full model validation suite
2. Record MAPE, max error, breakdown
3. Compare against previous calibration
4. Flag regressions automatically

---

## Immediate Actions (Low Risk)

### Action 1: Document Current Assumptions

Add explicit documentation of where each hard-coded value came from:
```python
# CALIBRATION SOURCE: Jetson Orin AGX 50W, 2026-02-04
# MODEL: ResNet-18 Conv2D+BN+ReLU layer (231M FLOPs)
# MEASURED: 720 GFLOPS achieved
# THEORETICAL: 958 GFLOPS peak
# EFFICIENCY: 720/958 = 0.75
BN_FUSION_FACTOR = 0.67  # Source: ResNet-18 with/without BN comparison
```

### Action 2: Create Efficiency Curve Visualization

Tool to plot current efficiency curves vs calibration data:
```bash
./cli/visualize_efficiency_curves.py --hardware i7-12700K
```

Shows:
- Hard-coded curve (what we assume)
- Calibration points (what we measured)
- Validation results (how models perform)
- Gap analysis (where curve is wrong)

### Action 3: Separate Calibration from Inference Code

Refactor `_get_compute_efficiency_scale()` to:
1. Load efficiency curve from configuration
2. Interpolate based on operation characteristics
3. Fall back to conservative defaults if no calibration

---

## Risk Assessment

### If We Do Nothing

- **Accuracy**: Will remain at ~26% MAPE for CPU, acceptable for some use cases
- **Maintenance**: Hard-coded values become increasingly wrong as hardware/libraries evolve
- **Trust**: Users may lose confidence if predictions are consistently wrong
- **Debugging**: Very difficult to diagnose why specific models are wrong

### If We Over-Engineer

- **Complexity**: Calibration system becomes too complex to use
- **Overfitting**: Per-model calibration defeats the purpose of estimation
- **Maintenance burden**: More calibration data to maintain

### Recommended Balance

1. **Keep operation-size-dependent scaling** - this is fundamentally correct
2. **Move parameters to data files** - enables systematic improvement
3. **Add calibration workflow** - enables data-driven updates
4. **Track accuracy over time** - enables regression detection

---

## Conclusion

Your concern about "fudge factors without solid empirical foundation" is valid. The current implementation:

1. **Has empirical foundation** - values were measured on specific hardware
2. **But is fragile** - hard-coded, difficult to update, hardware-specific
3. **And is incomplete** - sparse calibration points, aggressive interpolation

The path forward is:
1. **Don't remove efficiency scaling** - it's necessary and conceptually correct
2. **Externalize the parameters** - move from code to data files
3. **Create calibration workflow** - enable systematic data collection
4. **Track validation results** - measure accuracy over time

The goal is not to eliminate efficiency scaling, but to make it **principled, auditable, and improvable**.
