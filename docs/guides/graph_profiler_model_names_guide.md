# Model Names Guide

## How to Find Model Names for `profile_graph.py`

The `--model` argument accepts different types of model identifiers depending on the source.

---

## Quick Reference

| Model Type | Source | Example | How to Find |
|------------|--------|---------|-------------|
| **TorchVision** | PyTorch | `resnet18`, `mobilenet_v2` | `python cli/discover_models.py --list` |
| **HuggingFace** | Transformers | `bert-base-uncased`, `gpt2` | `python cli/discover_transformers.py` |
| **YOLO** | Ultralytics | `yolov8n.pt`, `yolo11m.pt` | File path to `.pt` file |
| **DETR** | HuggingFace | `facebook/detr-resnet-50` | HuggingFace model hub |
| **Custom** | Local | `/path/to/model.pt` | File path |

---

## 1. TorchVision Models (Vision CNNs & Transformers)

### Discovery Tool

```bash
# List all available models
python cli/discover_models.py --list

# OR: Use --list flag on profile_graph.py
python cli/profile_graph.py --list
```

### Common TorchVision Model Names

**ResNet Family:**
```
resnet18, resnet34, resnet50, resnet101, resnet152
```

**MobileNet Family:**
```
mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
```

**EfficientNet Family:**
```
efficientnet_b0, efficientnet_b1, efficientnet_b4, efficientnet_b7
efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
```

**Vision Transformers:**
```
vit_b_16, vit_b_32, vit_l_16, vit_l_32
swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
```

**Other CNNs:**
```
densenet121, densenet161, densenet201
vgg16, vgg19, vgg16_bn, vgg19_bn
alexnet, squeezenet1_0, squeezenet1_1
convnext_tiny, convnext_small, convnext_base
regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf
```

### Usage

```bash
python cli/profile_graph.py --model resnet18
python cli/profile_graph.py --model mobilenet_v2
python cli/profile_graph.py --model vit_b_16
```

---

## 2. HuggingFace Transformer Models (Text & Vision)

### Discovery Tool

```bash
# Discover text transformers
python cli/discover_transformers.py

# Generate usage examples
python cli/discover_transformers.py --generate-examples

# Test specific model
python cli/discover_transformers.py --test-model bert-base-uncased
```

### Common HuggingFace Model Names

**BERT Family (Encoder):**
```
bert-base-uncased
bert-base-cased
bert-large-uncased
distilbert-base-uncased
roberta-base
roberta-large
albert-base-v2
albert-large-v2
xlm-roberta-base
```

**GPT Family (Decoder):**
```
gpt2
gpt2-medium
gpt2-large
distilgpt2
EleutherAI/gpt-neo-125m
EleutherAI/gpt-neo-1.3B
EleutherAI/gpt-neo-2.7B
facebook/opt-125m
facebook/opt-350m
facebook/opt-1.3b
```

**Other Transformers:**
```
google/electra-small-discriminator
google/electra-base-discriminator
microsoft/deberta-v3-small
microsoft/deberta-v3-base
```

**Vision Transformers (DETR):**
```
facebook/detr-resnet-50
facebook/detr-resnet-101
```

### Finding More Models

**Option 1: HuggingFace Model Hub**
- Browse: https://huggingface.co/models
- Filter by task (text classification, generation, etc.)
- Copy the model identifier (e.g., `bert-base-uncased`)

**Option 2: Search on HuggingFace**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Search for models
huggingface-cli search "bert"
```

### Usage

```bash
# BERT
python cli/profile_graph.py --model bert-base-uncased

# GPT-2
python cli/profile_graph.py --model gpt2

# With organization prefix
python cli/profile_graph.py --model EleutherAI/gpt-neo-125m
python cli/profile_graph.py --model facebook/detr-resnet-50
```

---

## 3. YOLO Models (Object Detection)

### Model Files

YOLO models are specified by **file path** to a `.pt` file.

**Common YOLO model files:**
```
yolov8n.pt    # Nano (smallest, fastest)
yolov8s.pt    # Small
yolov8m.pt    # Medium
yolov8l.pt    # Large
yolov8x.pt    # Extra large

yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
```

### Download YOLO Models

**Option 1: Auto-download (Ultralytics)**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads if not present
```

**Option 2: Manual download**
```bash
# Download from GitHub releases
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### Usage

```bash
# YOLO file in current directory
python cli/profile_graph.py --model yolov8n.pt

# YOLO file with path
python cli/profile_graph.py --model /path/to/yolov8m.pt

# Custom input size (common for YOLO)
python cli/profile_graph.py --model yolov8n.pt --input-shape 1 3 640 640
```

---

## 4. Custom Models

### Local PyTorch Models

For custom models, provide the **file path**:

```bash
# Custom .pt file
python cli/profile_graph.py --model /path/to/my_model.pt

# Absolute or relative paths work
python cli/profile_graph.py --model ./models/custom_resnet.pt
python cli/profile_graph.py --model ~/experiments/model_v3.pt
```

### Python API (For nn.Module instances)

```python
from cli.profile_graph import profile_model
import torch.nn as nn

# Define custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))

# Profile it
model = MyModel()
profile_model(model, input_shape=(1, 3, 224, 224), model_name="MyModel")
```

---

## Model Name Format by Source

### TorchVision Format
- **Pattern**: `lowercase_with_underscores`
- **Examples**: `resnet18`, `mobilenet_v2`, `efficientnet_b0`
- **No prefix needed** (loaded from `torchvision.models`)

### HuggingFace Format
- **Pattern**: `lowercase-with-dashes` or `org/model-name`
- **Examples**:
  - Simple: `bert-base-uncased`, `gpt2`, `distilbert-base-cased`
  - With org: `facebook/opt-125m`, `EleutherAI/gpt-neo-1.3B`
- **Case-sensitive** (use exact name from HuggingFace)

### File Paths
- **Pattern**: Relative or absolute path to `.pt` file
- **Examples**: `yolov8n.pt`, `./models/custom.pt`, `/home/user/model.pt`
- **Extension**: Usually `.pt` or `.pth`

---

## Discovery Workflow

### Step 1: Identify Model Type

```
Is it a vision CNN/ViT?     → Use discover_models.py
Is it a text transformer?    → Use discover_transformers.py
Is it YOLO?                  → Find/download .pt file
Is it DETR?                  → Check HuggingFace hub
Is it custom?                → Get file path
```

### Step 2: Find Model Name

**TorchVision:**
```bash
python cli/discover_models.py --list
# OR
python cli/profile_graph.py --list
```

**HuggingFace:**
```bash
python cli/discover_transformers.py --generate-examples
# OR browse https://huggingface.co/models
```

**YOLO:**
```bash
ls *.pt  # Check current directory
# OR download from ultralytics
```

### Step 3: Profile the Model

```bash
python cli/profile_graph.py --model <model_name>
```

---

## Common Issues & Solutions

### Issue: "Unknown model: xyz"
**Solution:** Model not in TorchVision registry.
- Check spelling: `resnet18` not `ResNet18`
- Try discovery tool: `python cli/discover_models.py --test-model xyz`
- If it's HuggingFace, use exact name from hub

### Issue: "Model file not found"
**Solution:** For YOLO/custom models:
- Provide full path: `/home/user/yolov8n.pt`
- Check file exists: `ls -la yolov8n.pt`
- Download if needed: `wget <url>`

### Issue: "transformers not installed"
**Solution:** Install transformers library:
```bash
pip install transformers
```

### Issue: "ultralytics not installed"
**Solution:** Install ultralytics for YOLO:
```bash
pip install ultralytics
```

---

## Quick Command Reference

```bash
# List TorchVision models
python cli/profile_graph.py --list

# Discover transformers
python cli/discover_transformers.py

# Profile TorchVision model
python cli/profile_graph.py --model resnet18

# Profile HuggingFace model
python cli/profile_graph.py --model bert-base-uncased

# Profile YOLO model
python cli/profile_graph.py --model yolov8n.pt

# Profile DETR model
python cli/profile_graph.py --model facebook/detr-resnet-50

# Test if model is supported
python cli/discover_models.py --test-model resnet18
python cli/discover_transformers.py --test-model bert-base-uncased
```

---

## Model Naming Conventions

### TorchVision
- Lowercase with underscores
- Include version: `mobilenet_v2`, `efficientnet_b0`
- Include variant: `vit_b_16` (base, 16x16 patches)

### HuggingFace
- Lowercase with dashes
- Organization prefix optional: `bert-base-uncased` or `google/bert-base-uncased`
- Descriptive: `distilbert` (distilled BERT), `roberta` (RoBERTa)

### YOLO
- Version + size: `yolov8n.pt` (v8 nano)
- Sizes: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (extra large)

---

## Resources

- **TorchVision Models**: https://pytorch.org/vision/stable/models.html
- **HuggingFace Hub**: https://huggingface.co/models
- **YOLO Models**: https://github.com/ultralytics/ultralytics
- **DETR Models**: https://huggingface.co/models?search=detr

---

## Summary

**Finding model names:**
1. **TorchVision**: Run `python cli/profile_graph.py --list`
2. **HuggingFace**: Run `python cli/discover_transformers.py` or browse huggingface.co
3. **YOLO**: Use file path to `.pt` file
4. **Custom**: Use file path

**Format by source:**
- TorchVision: `lowercase_underscore` (e.g., `resnet18`)
- HuggingFace: `lowercase-dash` or `org/model-name` (e.g., `bert-base-uncased`, `facebook/opt-125m`)
- File paths: Relative or absolute (e.g., `yolov8n.pt`, `/path/to/model.pt`)
