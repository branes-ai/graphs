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

## Running All Tests

```bash
# From repo root
python src/graphs/validation/test_conv2d.py
python src/graphs/validation/test_resnet18.py
python src/graphs/validation/test_resnet_family.py
```

Or as Python modules:
```bash
python -m graphs.validation.test_conv2d
python -m graphs.validation.test_resnet18
python -m graphs.validation.test_resnet_family
```

---

## Validation Reports

Detailed validation reports are located in `docs/validation/`:
- `conv2d_validation_report.md` - Conv2D validation details
- `resnet18_validation_report.md` - ResNet-18 deep dive

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
| Conv Stack | 1-5G | 3-layer Conv2D |
| Small CNN | 1-5G | ResNet-18 |
| Medium CNN | 5-15G | ResNet-50, MobileNetV2 |
| Large CNN | 15-50G | ResNet-152, EfficientNet-B7 |
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

- [ ] MobileNetV1/V2/V3 validation
- [ ] EfficientNet-B0 through B7
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
