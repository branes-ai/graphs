# Estimator Accuracy Validation

This directory validates the accuracy of FLOP and memory estimators against theoretical calculations and published model statistics.

## Test Files

- **`test_conv2d.py`** - Basic Conv2D operation validation
- **`test_resnet18.py`** - ResNet-18 full model characterization
- **`test_resnet_family.py`** - ResNet-18/34/50 comparison
- **`test_mobilenet.py`** - MobileNet-V2 with depthwise convolutions
- **`test_efficientnet.py`** - EfficientNet-B0 validation

## Running Tests

```bash
# Single model
python validation/estimators/test_resnet18.py

# Model family comparison
python validation/estimators/test_resnet_family.py

# All estimator tests
for f in validation/estimators/test_*.py; do python "$f"; done
```

## Validation Criteria

### FLOP Accuracy
- **Target:** Within ±6% of theoretical calculations
- **Method:** Manual FLOP counting for known architectures
- **Status:** ✅ All tested models within 6%

### Memory Accuracy
- **Target:** Within ±10% of actual PyTorch tensors
- **Method:** Compare estimated bytes vs actual tensor sizes
- **Status:** ✅ Validated on ResNet/MobileNet

### Relative Performance
- **Target:** Speedups consistent across hardware
- **Method:** Compare ratios (e.g., ResNet-50 should be ~2.4× ResNet-18)
- **Status:** ✅ Validated in family tests

## Expected Results

### ResNet-18
```
Total FLOPs: ~3.6-3.8 GFLOPs (theoretical: 3.59 G)
Accuracy: Within ±5.6%
Operations: 60 ops → 32 fused subgraphs
Memory reduction: 19.6% (19.2 MB saved)
```

### ResNet Family
```
ResNet-18: 3.79 G FLOPs (±5.6%)
ResNet-34: 7.49 G FLOPs (±2.6%)
ResNet-50: 10.80 G FLOPs (±2.9%)
```

### MobileNet-V2
```
Total FLOPs: ~1.9 GFLOPs
Fusion benefit: 42% memory reduction
Operations: 141 ops → 66 fused subgraphs
```

## Common Issues

**FLOP mismatch >10%:**
- Check if depthwise convolutions handled correctly
- Verify kernel size and stride calculations
- Confirm groups parameter (group > 1 = depthwise)

**Import errors:**
- Ensure sys.path includes repo root
- Check if PyTorch/torchvision installed

**Model not found:**
- Update torchvision: `pip install --upgrade torchvision`
- Check model name spelling

## Theoretical FLOP Calculations

### Standard Convolution
```
FLOPs = Batch × OutChannels × OutHeight × OutWidth ×
        (2 × InChannels × KernelH × KernelW)
```

### Depthwise Convolution
```
FLOPs = Batch × Channels × OutHeight × OutWidth ×
        (2 × KernelH × KernelW)
```

### Linear Layer
```
FLOPs = Batch × InputDim × OutputDim × 2
```

### Batch Normalization
```
FLOPs = Batch × Channels × Height × Width × 4
(mean, var, normalize, scale+shift)
```

## Adding New Model Tests

1. Create `test_<model>.py` file
2. Calculate theoretical FLOPs manually
3. Run characterization
4. Compare and ensure ±6% accuracy
5. Document any special handling needed

Example template:
```python
#!/usr/bin/env python
"""Validation test for <Model>"""

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.characterize.walker import FXGraphWalker
from src.graphs.characterize.arch_profiles import cpu_profile

# Create model
model = ...

# Trace and propagate shapes
traced = symbolic_trace(model)
ShapeProp(traced).propagate(torch.randn(1, 3, 224, 224))

# Characterize
walker = FXGraphWalker(cpu_profile, registry)
result = walker.characterize_graph(traced, {})

# Compare against theoretical FLOPs
theoretical_flops = ...  # Calculate manually
estimated_flops = result['total_flops']
accuracy = abs(estimated_flops - theoretical_flops) / theoretical_flops * 100

print(f"Theoretical: {theoretical_flops/1e9:.2f} G")
print(f"Estimated:   {estimated_flops/1e9:.2f} G")
print(f"Accuracy:    ±{accuracy:.1f}%")

assert accuracy < 6.0, f"Accuracy {accuracy:.1f}% exceeds ±6% threshold"
```

## Success Metrics

Estimator validation passes if:
- ✅ All models within ±6% FLOP accuracy
- ✅ Memory estimates within ±10%
- ✅ No crashes on standard torchvision models
- ✅ Relative speedups consistent (e.g., ResNet-50 ~2.4× ResNet-18)
- ✅ Special operations handled (depthwise conv, grouped conv)

## Known Limitations

1. **Dynamic shapes:** Not supported, static shapes only
2. **Control flow:** if/else in forward() not handled
3. **Custom ops:** May not have estimators registered
4. **Attention:** Transformer attention needs special handling

## Documentation

See also:
- `../hardware/README.md` - Hardware mapper validation
- `../../tests/characterize/` - Unit tests for components
- `../../docs/hardware_characterization_2025-10.md` - Validation results
