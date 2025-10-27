# How to Use: discover_models.py

## Overview

`discover_models.py` tests which torchvision models are FX-traceable and can be used with the graph characterization tools. It scans the torchvision model zoo and generates a registry of compatible models.

**Key Capabilities:**
- Test FX-traceability of all torchvision models
- Filter by model family (ResNet, MobileNet, ViT, etc.)
- Generate MODEL_REGISTRY code for other tools
- Test individual models
- Custom skip patterns

**Target Users:**
- New users discovering available models
- Tool developers maintaining model registries
- Engineers debugging tracing issues

---

## Installation

**Requirements:**
```bash
pip install torch torchvision
```

**Verify Installation:**
```bash
python3 cli/discover_models.py --help
```

---

## Basic Usage

### Discover All Models

```bash
python3 cli/discover_models.py
```

**Output:**
- Summary statistics
- FX-traceable models grouped by family
- Skipped models (detection, segmentation, video)

---

### Verbose Mode

See each model test result:

```bash
python3 cli/discover_models.py --verbose
```

**Output:**
```
Testing models:

✓ resnet18                       - FX-traceable
✓ resnet34                       - FX-traceable
✓ resnet50                       - FX-traceable
✗ fasterrcnn_resnet50_fpn       - TypeError: forward() takes 2 positional...
✓ mobilenet_v2                  - FX-traceable
✓ vit_b_16                      - FX-traceable
...
```

---

### Generate Registry Code

Export Python code for MODEL_REGISTRY:

```bash
python3 cli/discover_models.py --generate-code
```

**Output:**
```python
MODEL_REGISTRY = {
    # Resnet family
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,

    # Mobilenet family
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_small': models.mobilenet_v3_small,

    # Vit family
    'vit_b_16': models.vit_b_16,
    'vit_l_16': models.vit_l_16,
    ...
}
```

**Use Case:** Copy-paste into `profile_graph.py` or custom analysis scripts

---

### Test Single Model

Test a specific model:

```bash
python3 cli/discover_models.py --test-model resnet18
```

**Output:**
```
Testing resnet18...
✓ resnet18                       - FX-traceable

✓ resnet18 is FX-traceable!
```

---

## Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--verbose, -v` | flag | Show detailed test results for each model |
| `--generate-code, -g` | flag | Generate MODEL_REGISTRY Python code |
| `--test-model` | str | Test a single specific model |
| `--skip-patterns` | str[] | Space-separated patterns to skip |

---

## Default Skip Patterns

By default, the following model categories are skipped:

**Detection Models:**
- `rcnn` (Faster R-CNN, Mask R-CNN)
- `retinanet`
- `fcos`
- `ssd`

**Segmentation Models:**
- `deeplabv3`
- `fcn`
- `lraspp`

**Video Models (5D input):**
- `raft` (optical flow)
- `r3d`, `r2plus1d`, `mc3`, `s3d`
- `mvit`, `swin3d`

**Quantized Models:**
- `quantized_*`

**Reason:** These models require special inputs (multiple tensors, 5D inputs, etc.) that don't work with standard FX symbolic tracing.

---

## Custom Skip Patterns

Override default skip patterns:

```bash
python3 cli/discover_models.py --skip-patterns fcos vit ssd
```

**Use Case:** Test specific model families that were previously skipped

---

## Output Sections

### Summary Statistics

```
================================================================================
SUMMARY
================================================================================

FX-traceable:  147 models
  Failed:        23 models
  Skipped:       85 models (detection/segmentation/video/quantized)
```

---

### Models by Family

```
================================================================================
FX-TRACEABLE MODELS BY FAMILY
================================================================================

RESNET (8):
  resnet18
  resnet34
  resnet50
  resnet101
  resnet152
  resnext50_32x4d
  resnext101_32x8d
  wide_resnet50_2

MOBILENET (5):
  mobilenet_v2
  mobilenet_v3_large
  mobilenet_v3_small

EFFICIENTNET (8):
  efficientnet_b0
  efficientnet_b1
  efficientnet_b2
  efficientnet_b3
  efficientnet_b4
  efficientnet_b5
  efficientnet_b6
  efficientnet_b7
  efficientnet_v2_s
  efficientnet_v2_m
  efficientnet_v2_l

VIT (10):
  vit_b_16
  vit_b_32
  vit_l_16
  vit_l_32
  vit_h_14

CONVNEXT (6):
  convnext_tiny
  convnext_small
  convnext_base
  convnext_large

VGG (4):
  vgg11
  vgg13
  vgg16
  vgg19
  vgg11_bn
  vgg13_bn
  vgg16_bn
  vgg19_bn

DENSENET (4):
  densenet121
  densenet161
  densenet169
  densenet201

And more...
```

---

## Common Usage Examples

### Example 1: Quick Model Check

Before using a model in analysis, verify it's FX-traceable:

```bash
python3 cli/discover_models.py --test-model efficientnet_b0
```

---

### Example 2: Discover All Vision Transformers

```bash
python3 cli/discover_models.py --verbose | grep "vit_"
```

---

### Example 3: Update Tool Registry

After a torchvision update, regenerate the registry:

```bash
python3 cli/discover_models.py --generate-code > new_registry.txt
# Review and copy to profile_graph.py
```

---

### Example 4: Debug Tracing Failure

See why a model fails:

```bash
python3 cli/discover_models.py --test-model fasterrcnn_resnet50_fpn --verbose
```

**Output shows error message:**
```
✗ fasterrcnn_resnet50_fpn       - TypeError: forward() takes 2 positional...
```

---

### Example 5: Test ConvNeXt Family Only

```bash
# Skip everything except ConvNeXt
python3 cli/discover_models.py \
  --skip-patterns resnet mobile efficient vit vgg dense shuffle squeeze \
  --verbose | grep convnext
```

---

## FX-Traceability Explained

### What is FX Tracing?

PyTorch FX (`torch.fx`) performs **symbolic tracing** of a model:
- Records operations as a graph
- Propagates tensor shapes
- Enables graph transformations

**Requirement for our tools:** Models must be FX-traceable to use partitioning and hardware mapping.

---

### Why Do Some Models Fail?

**Common Failure Reasons:**

1. **Dynamic Control Flow**
   ```python
   # NOT FX-traceable
   if x.shape[0] > 10:
       return self.path_a(x)
   else:
       return self.path_b(x)
   ```

2. **Multiple Inputs/Outputs**
   ```python
   # Detection models need (images, targets)
   def forward(self, images, targets=None):
       ...
   ```

3. **Python Built-in Types**
   ```python
   # Using Python lists/dicts in forward()
   def forward(self, x):
       results = []
       for layer in self.layers:
           results.append(layer(x))
       return results
   ```

4. **Non-Tensor Operations**
   ```python
   # In-place modifications, assertions, etc.
   assert x.shape[0] == 1
   ```

---

### FX-Traceable Model Characteristics

✓ **Single forward() signature**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```

✓ **Static control flow**
```python
# All branches traced
x = self.conv1(x) if self.use_conv1 else x  # OK if traced with same value
```

✓ **Standard PyTorch operations**
```python
x = torch.relu(x)
x = self.conv(x)
```

---

## Model Categories

### Classification Models (FX-Traceable)

- ResNet family (resnet18, resnet50, etc.)
- MobileNet family (mobilenet_v2, mobilenet_v3_*)
- EfficientNet family (efficientnet_b0-b7, efficientnet_v2_*)
- VGG family (vgg16, vgg19)
- DenseNet family (densenet121, densenet161, etc.)
- Vision Transformers (vit_b_16, vit_l_16, etc.)
- ConvNeXt (convnext_tiny, convnext_small, etc.)
- SqueezeNet, ShuffleNet, etc.

**Total:** 140+ models

---

### Detection Models (NOT FX-Traceable)

- Faster R-CNN, Mask R-CNN
- RetinaNet
- FCOS
- SSD

**Reason:** Require `(images, targets)` input during training

---

### Segmentation Models (NOT FX-Traceable)

- DeepLabV3
- FCN
- LRASPP

**Reason:** Complex multi-scale architectures with dynamic shapes

---

### Video Models (NOT FX-Traceable)

- R3D, R(2+1)D
- MViT, Swin3D

**Reason:** Require 5D input (batch, channels, time, height, width)

---

## Troubleshooting

### Error: "No module named 'torchvision'"

**Solution:**
```bash
pip install torchvision
```

---

### Many Models Fail

**Check torchvision version:**
```bash
python3 -c "import torchvision; print(torchvision.__version__)"
```

**Recommended:** torchvision >= 0.13.0

**Update:**
```bash
pip install --upgrade torchvision
```

---

### Model Missing from Output

**Reason:** Model might be:
1. In a skipped category (detection, segmentation, video)
2. Not yet in torchvision (check version)
3. Actually failing FX tracing

**Debug:**
```bash
python3 cli/discover_models.py --test-model <model_name> --verbose
```

---

## Advanced Usage

### Scan Entire Zoo (No Skipping)

```bash
python3 cli/discover_models.py --skip-patterns "" --verbose
```

**Warning:** This will attempt to trace detection/segmentation models (they will fail)

---

### Integrate into Scripts

```python
import subprocess
import json

# Run discovery
result = subprocess.run(
    ['python3', 'cli/discover_models.py'],
    capture_output=True,
    text=True
)

# Parse output for traceable models
lines = result.stdout.split('\n')
traceable = [
    line.strip()
    for line in lines
    if line.startswith('  ') and not line.startswith(' ' * 3)
]
```

---

### Custom Model Testing

Test your own models:

```python
import torch
from torch.fx import symbolic_trace

class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)

model = MyModel()
try:
    traced = symbolic_trace(model)
    print("✓ Model is FX-traceable!")
except Exception as e:
    print(f"✗ Model NOT traceable: {e}")
```

---

## Generated Registry Code Usage

**After generating code:**

```bash
python3 cli/discover_models.py --generate-code > registry_code.txt
```

**Integrate into tools:**

1. Copy the `MODEL_REGISTRY = {...}` block
2. Paste into `profile_graph.py` or your custom script
3. Use models by name:

```python
from torchvision import models

MODEL_REGISTRY = {
    # ... generated code ...
}

# Load model by name
model_fn = MODEL_REGISTRY['resnet50']
model = model_fn(weights=None)
```

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `profile_graph.py` | Profile discovered models |
| `analyze_graph_mapping.py` | Map models to hardware |
| `compare_models.py` | Compare models across hardware |

---

## Further Reading

- **PyTorch FX Documentation**: https://pytorch.org/docs/stable/fx.html
- **Symbolic Tracing**: `experiments/fx/tutorial/`
- **Architecture Guide**: `CLAUDE.md`

---

## Contact & Feedback

Report issues or request features at the project repository.
