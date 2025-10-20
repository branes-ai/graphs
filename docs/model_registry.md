# Model Registry Management

## Overview

The `MODEL_REGISTRY` in `cli/profile_graph.py` and `cli/show_fvcore_table.py` controls which models are available for profiling.

## Current Status

- **Manual registry**: 33 carefully curated models
- **Available in torchvision**: 121 total models
- **FX-traceable**: 80 models (discovered via `cli/discover_models.py`)

## Three Approaches

### 1. Manual Registry (Current - Recommended)

**File**: `cli/profile_graph.py`, `cli/show_fvcore_table.py`

**Pros**:
- Fast startup (no discovery overhead)
- Curated selection of common models
- Known to work well
- Easy to understand

**Cons**:
- Requires manual updates for new models
- Misses some available models (47 currently)

**How to add new models**:
```python
# 1. Test if model is FX-traceable
python cli/discover_models.py --test-model MODEL_NAME

# 2. If successful, add to MODEL_REGISTRY
MODEL_REGISTRY = {
    # ... existing models ...
    'new_model': models.new_model,
}
```

### 2. Automatic Discovery (Available)

**File**: `cli/profile_graph.py` - see `_build_model_registry_auto()` function

**Pros**:
- Always up-to-date with torchvision
- No manual maintenance
- Discovers all 80 FX-traceable models

**Cons**:
- Slower startup (~5-10 seconds to test all models)
- Potential for unexpected failures
- Includes less common models

**How to enable**:
```python
# In profile_graph.py, replace line 44:
# MODEL_REGISTRY = {
#     ...
# }

# With:
MODEL_REGISTRY = _build_model_registry_auto()
```

### 3. Discovery Tool (Hybrid - Recommended for Updates)

**File**: `cli/discover_models.py`

Use this tool periodically to check for new models, then manually add the ones you want.

**Usage**:
```bash
# See summary of available models
python cli/discover_models.py

# Verbose output showing each test
python cli/discover_models.py --verbose

# Generate registry code
python cli/discover_models.py --generate-code

# Test specific model
python cli/discover_models.py --test-model inception_v3
```

## Recommendation

**For most users**: Keep the manual registry (current approach)
- Use `cli/discover_models.py` quarterly to check for new models
- Manually add interesting new models to keep registry focused

**For research/exploration**: Enable automatic discovery
- Replace `MODEL_REGISTRY = {...}` with `MODEL_REGISTRY = _build_model_registry_auto()`
- Accept slower startup for access to all models

## Discovery Results

Current discovery (torchvision 0.x):

```
Total models:          121
FX-traceable:           80
Failed:                  0
Skipped:                41 (detection/segmentation/video/quantized)
```

### Model Families Found (FX-traceable)

| Family | Count | Examples |
|--------|-------|----------|
| ResNet | 5 | resnet18, resnet50, resnet152 |
| EfficientNet | 11 | efficientnet_b0-b7, v2_s/m/l |
| RegNet | 15 | regnet_x/y_400mf to 128gf |
| ViT | 5 | vit_b_16, vit_l_16, vit_h_14 |
| Swin | 6 | swin_t/s/b, swin_v2_t/s/b |
| MobileNet | 3 | mobilenet_v2, v3_large/small |
| VGG | 8 | vgg11-19, with/without BN |
| DenseNet | 4 | densenet121/161/169/201 |
| ConvNeXt | 4 | convnext_tiny/small/base/large |
| Others | 19 | inception, googlenet, shufflenet, etc. |

### Currently Missing from Manual Registry

47 models are FX-traceable but not in manual registry:
- All ResNeXt models (resnext50_32x4d, resnext101_32x8d, etc.)
- Wide ResNets (wide_resnet50_2, wide_resnet101_2)
- Additional EfficientNets (b2, b3, b5, b6, v2 variants)
- Additional RegNets (regnet_x family, larger regnet_y)
- MNASNet variants
- ShuffleNet variants
- Additional Swin variants (swin_v2_*)
- Additional ViT variants (vit_h_14, vit_l_32, vit_b_32)
- MaxViT, Inception v3, GoogleNet
- Additional VGG and SqueezeNet variants
- Additional ConvNeXt (convnext_large)

## How Models are Skipped

The discovery tool skips models with these patterns:
- `rcnn`, `retinanet`, `fcos`, `ssd` - Object detection (complex outputs)
- `deeplabv3`, `fcn`, `lraspp` - Segmentation
- `r3d`, `r2plus1d`, `mc3`, `s3d` - Video (need 5D inputs)
- `mvit`, `swin3d` - Video transformers
- `quantized` - Quantized models (different ops)
- `raft` - Optical flow

## Manual Discovery Commands

```bash
# List all torchvision models
python -c "import torchvision.models as m; print('\n'.join(sorted(m.list_models())))"

# Test if specific model traces
python -c "
from torch.fx import symbolic_trace
import torchvision.models as m
model = m.MODEL_NAME(weights=None)
model.eval()
symbolic_trace(model)
print('✓ Traceable')
"
```

## Updating the Registry

### Option A: Add models individually
1. Run `python cli/discover_models.py`
2. Pick models you want from the output
3. Test: `python cli/discover_models.py --test-model MODEL_NAME`
4. Add to `MODEL_REGISTRY` manually

### Option B: Use generated code
1. Run `python cli/discover_models.py --generate-code`
2. Copy generated `MODEL_REGISTRY = {...}`
3. Replace existing registry in `profile_graph.py` and `show_fvcore_table.py`

### Option C: Enable automatic
1. In `profile_graph.py`, replace:
   ```python
   MODEL_REGISTRY = { ... }
   ```
   with:
   ```python
   MODEL_REGISTRY = _build_model_registry_auto()
   ```

## Performance Comparison

| Approach | Startup Time | Models Available | Maintenance |
|----------|--------------|------------------|-------------|
| Manual (33) | <0.1s | 33 | Quarterly review |
| Manual (80) | <0.1s | 80 | One-time update |
| Automatic | ~5-10s | 80 | Zero |

## Future Considerations

- **Lazy loading**: Load model constructors only when needed
- **Caching**: Cache discovery results to avoid re-testing
- **Versioning**: Track torchvision version and auto-update
- **User additions**: Allow users to register custom models
