# Validation Tests

This directory contains validation scripts for the DNN characterization pipeline.

## Purpose

These tests validate that the characterization pipeline produces accurate estimates by comparing measurements against:
- Theoretical FLOP calculations
- Known model architectures (ResNet, MobileNet, EfficientNet)
- Published performance data

## Test Scripts

### `test_conv2d.py`
**Purpose**: Validates Conv2D stack characterization

**What it tests**:
- Conv2d+ReLU fusion pattern detection
- FLOP estimation for convolutional layers
- Shape propagation through sequential convolutions
- Memory usage calculations

**Expected results**:
- ~1.8 GFLOPs for 3-layer Conv2D stack (32×3×64×64 input)
- Non-zero metrics across all architectures

**Run**:
```bash
python src/graphs/validation/test_conv2d.py
```

---

### `test_resnet18.py`
**Purpose**: Deep analysis of ResNet-18 from torchvision

**What it tests**:
- Conv2d+BatchNorm2d+ReLU fusion detection
- Residual block characterization
- Graph structure analysis
- Comparison with theoretical values

**Expected results**:
- 3.79 GFLOPs (within 6% of 3.59 GFLOPs theoretical)
- 11.69M parameters
- 17 fused patterns (Conv+BN+ReLU)

**Run**:
```bash
python src/graphs/validation/test_resnet18.py
```

**Detailed report**: See `docs/validation/resnet18_validation_report.md`

---

### `test_resnet_family.py`
**Purpose**: Compare ResNet-18, ResNet-34, and ResNet-50

**What it tests**:
- Model complexity scaling
- Architecture performance comparison
- Energy efficiency analysis
- Throughput projections

**Expected results**:
- ResNet-18: 3.79 GFLOPs, 11.69M params
- ResNet-34: 7.49 GFLOPs, 21.80M params
- ResNet-50: 10.80 GFLOPs, 25.56M params

**Output**: `resnet_family_results.csv`

**Run**:
```bash
python src/graphs/validation/test_resnet_family.py
```

---

### `test_mobilenet.py`
**Purpose**: Characterize MobileNet family (V2, V3-Small, V3-Large)

**What it tests**:
- Depthwise separable convolution handling
- Conv2d+ReLU6 fusion detection
- Conv2d+Hardswish fusion (MobileNetV3)
- Efficient architecture characterization
- Edge deployment analysis

**Expected results**:
- MobileNet-V2: 1.87 GFLOPs, 3.50M params
- MobileNet-V3-Small: 0.29 GFLOPs, 2.54M params (most efficient)
- MobileNet-V3-Large: 1.17 GFLOPs, 5.48M params

**Output**: `mobilenet_results.csv`

**Run**:
```bash
python src/graphs/validation/test_mobilenet.py
```

---

### `test_efficientnet.py`
**Purpose**: Characterize EfficientNet family (B0, B1, B2, V2-S, V2-M)

**What it tests**:
- MBConv block characterization
- Squeeze-and-Excitation block handling
- Compound scaling validation
- Memory footprint analysis

**Expected results**:
- EfficientNet-B0: 2.35 GFLOPs, 5.29M params
- EfficientNet-B1: 3.41 GFLOPs, 7.79M params
- EfficientNet-B2: 3.96 GFLOPs, 9.11M params
- EfficientNet-V2-S: 17.52 GFLOPs, 21.46M params
- EfficientNet-V2-M: 33.11 GFLOPs, 54.14M params

**Output**: `efficientnet_results.csv`

**Run**:
```bash
python src/graphs/validation/test_efficientnet.py
```

**Detailed report**: See `docs/validation/mobilenet_efficientnet_comparison.md`

---

## Running All Tests

```bash
# From repo root
python src/graphs/validation/test_conv2d.py
python src/graphs/validation/test_resnet18.py
python src/graphs/validation/test_resnet_family.py
python src/graphs/validation/test_mobilenet.py
python src/graphs/validation/test_efficientnet.py
```

Or as Python modules:
```bash
python -m graphs.validation.test_conv2d
python -m graphs.validation.test_resnet18
python -m graphs.validation.test_resnet_family
python -m graphs.validation.test_mobilenet
python -m graphs.validation.test_efficientnet
```

---

## Validation Reports

Detailed validation reports are located in `docs/validation/`:
- `conv2d_validation_report.md` - Conv2D validation details
- `resnet18_validation_report.md` - ResNet-18 deep dive
- `mobilenet_efficientnet_comparison.md` - Comprehensive MobileNet and EfficientNet analysis

---

## Adding New Validation Tests

When adding support for new model architectures (e.g., MobileNet, EfficientNet), create a corresponding validation test:

1. **Create test script**: `test_<model_name>.py`
2. **Include**:
   - Model loading (from torchvision or custom)
   - Expected theoretical values
   - Graph structure analysis
   - Cross-architecture comparison
   - Validation report generation

3. **Template structure**:
```python
#!/usr/bin/env python
"""Validation test for <ModelName>"""

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from graphs.characterize.walker import FXGraphWalker

def main():
    # 1. Load model
    # 2. FX trace + shape propagation
    # 3. Analyze graph structure
    # 4. Characterize across architectures
    # 5. Compare with theoretical values
    # 6. Print results
    pass

if __name__ == "__main__":
    exit(main())
```

4. **Document results**: Create validation report in `docs/validation/`

---

## Expected FLOP Ranges

| Model Type | FLOP Range | Example |
|------------|------------|---------|
| Small MLP | 1-10M | MLP (128→256→64) |
| Ultra-Efficient CNN | 0.2-1G | MobileNet-V3-Small (0.29G) |
| Efficient CNN | 1-3G | MobileNet-V2 (1.87G), EfficientNet-B0 (2.35G) |
| Conv Stack | 1-5G | 3-layer Conv2D |
| Small CNN | 1-5G | ResNet-18 (3.79G), EfficientNet-B1 (3.41G) |
| Medium CNN | 5-15G | ResNet-50 (10.8G), EfficientNet-B2 (3.96G) |
| Large CNN | 15-50G | ResNet-152, EfficientNet-V2-S (17.5G), V2-M (33.1G) |
| Vision Transformer | 10-100G | ViT-Base, ViT-Large |

---

## Validation Checklist

When adding a new test, verify:

- [ ] FX tracing succeeds (no errors)
- [ ] Shape propagation working (all nodes have tensor_meta)
- [ ] Pattern matching finds expected fusions
- [ ] FLOPs within 10% of theoretical values
- [ ] Parameters match PyTorch model
- [ ] Architecture speedups are reasonable (TPU > GPU > KPU > CPU)
- [ ] Energy efficiency trends correct (KPU most efficient)
- [ ] Test produces CSV output (if applicable)
- [ ] Validation report created (for major models)

---

## Troubleshooting

### FX Tracing Fails
**Symptom**: `symbolic_trace()` raises error
**Causes**:
- Dynamic control flow (if/for/while in forward())
- In-place operations
- Non-tensor operations

**Solution**: Use `torch.fx.symbolic_trace()` with custom tracer or refactor model

### Pattern Matching Returns Zero FLOPs
**Symptom**: All metrics are zero
**Causes**:
- Fusion pattern not registered
- Module types don't match expected sequence

**Solution**: Check `fused_ops.py` registry, add missing patterns

### Shape Propagation Fails
**Symptom**: `ShapeProp.propagate()` raises error
**Causes**:
- Incompatible input shape
- Custom operations without shape inference

**Solution**: Ensure input tensor shape matches model expectations

---

## Future Tests to Add

- [x] MobileNetV2/V3 validation ✓
- [x] EfficientNet-B0 through B2, V2-S, V2-M ✓
- [ ] EfficientNet-B3 through B7
- [ ] VGG-16/19 (dense convolutions)
- [ ] Inception-v3 (multi-branch)
- [ ] Vision Transformers (ViT)
- [ ] YOLO/SSD (object detection)
- [ ] U-Net (segmentation)

---

## References

- PyTorch FX documentation: https://pytorch.org/docs/stable/fx.html
- torchvision models: https://pytorch.org/vision/stable/models.html
- FLOP counting: https://github.com/sovrasov/flops-counter.pytorch
