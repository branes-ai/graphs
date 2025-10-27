# How to Use: profile_graph.py & profile_graph_with_fvcore.py

## Overview

These tools profile PyTorch models to characterize computational requirements, memory usage, and operation-level resource demands **independent of hardware**.

**profile_graph.py:**
- Hardware-independent graph characterization
- FLOPs, memory, arithmetic intensity per layer
- Bottleneck identification (compute vs memory-bound)

**profile_graph_with_fvcore.py:**
- Compare our FLOP estimates against fvcore library
- Validation and accuracy checking

---

## Quick Start

### Profile a Model

```bash
python3 cli/profile_graph.py --model resnet50
```

### Compare with fvcore

```bash
python3 cli/profile_graph_with_fvcore.py --model resnet50
```

---

## profile_graph.py

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | resnet18 | Model name |
| `--input-shape` | int[] | 1,3,224,224 | Input tensor shape |
| `--output` | str | - | Output file (JSON/CSV) |
| `--verbose` | flag | False | Show per-layer details |

---

### Output Format

```
================================================================================
GRAPH PROFILE: ResNet-50
================================================================================

Model Summary:
  Total Operators:      177
  Total FLOPs:          8.21 G
  Total Memory:         52.4 MB
  Arithmetic Intensity: 156.6 (FLOPs/Byte)

Operation Breakdown:
  Conv2D:      85% (150 ops, 7.8 G FLOPs)
  BatchNorm:   10% (20 ops, 0.2 G FLOPs)
  ReLU:        3% (5 ops, 0.1 G FLOPs)
  Linear:      2% (2 ops, 0.11 G FLOPs)

Bottleneck Analysis:
  Compute-bound:  90% (AI > 50)
  Memory-bound:   10% (AI < 50)

Layer Details (Top 10 by FLOPs):
  conv1:           120 M FLOPs, 2.1 MB, AI=57.1 (compute-bound)
  layer1.0.conv1:  370 M FLOPs, 6.5 MB, AI=56.9 (compute-bound)
  ...
```

---

### Usage Examples

#### Example 1: Profile Custom Input Size

```bash
python3 cli/profile_graph.py \
  --model efficientnet_b4 \
  --input-shape 1,3,380,380
```

#### Example 2: Export to JSON

```bash
python3 cli/profile_graph.py \
  --model mobilenet_v2 \
  --output profile.json
```

**JSON Format:**
```json
{
  "model": "mobilenet_v2",
  "total_flops": 600000000,
  "total_memory": 14200000,
  "layers": [
    {
      "name": "conv1",
      "type": "conv2d",
      "flops": 120000000,
      "memory": 2100000,
      "arithmetic_intensity": 57.1,
      "bottleneck": "compute-bound"
    },
    ...
  ]
}
```

#### Example 3: Verbose Per-Layer Analysis

```bash
python3 cli/profile_graph.py \
  --model resnet18 \
  --verbose
```

---

## profile_graph_with_fvcore.py

### Purpose

Validate our FLOP estimates against the `fvcore` library (Facebook's FLOPs counter).

---

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | resnet18 | Model name |
| `--models` | str | - | Comma-separated model names |

---

### Output Format

```
================================================================================
FLOP COMPARISON: Our Estimator vs fvcore
================================================================================

Model              Our FLOPs    fvcore FLOPs   Difference   Accuracy
--------------------------------------------------------------------------------
resnet18           3.64 G       3.63 G         +0.01 G      99.7%
resnet50           8.21 G       8.19 G         +0.02 G      99.8%
mobilenet_v2       600 M        597 M          +3 M         99.5%
vit_b_16           33.6 G       33.4 G         +0.2 G       99.4%
```

---

### Usage Examples

#### Example 1: Single Model Validation

```bash
python3 cli/profile_graph_with_fvcore.py --model resnet50
```

#### Example 2: Multiple Models

```bash
python3 cli/profile_graph_with_fvcore.py \
  --models "resnet18,resnet50,mobilenet_v2,efficientnet_b0"
```

---

## Interpretation Guide

### Arithmetic Intensity (AI)

**Definition:** FLOPs per Byte of memory traffic

**AI = FLOPs / Memory (bytes)**

**Interpretation:**
- **AI > 100**: Strongly compute-bound (good for GPUs, accelerators)
- **AI 50-100**: Compute-bound (typical CNNs)
- **AI 10-50**: Balanced (need both compute and bandwidth)
- **AI < 10**: Memory-bound (need high bandwidth)

**Examples:**
- Conv2D (3×3, 64 channels): AI ≈ 50-100 (compute-bound)
- Pooling, BatchNorm: AI < 10 (memory-bound)
- MatMul (large): AI > 100 (strongly compute-bound)

---

### Bottleneck Type

**Compute-Bound:**
- Limited by FLOPS
- Benefits from higher clock speeds, more cores/SMs
- Batching improves efficiency

**Memory-Bound:**
- Limited by memory bandwidth
- Benefits from HBM, higher bandwidth
- Precision reduction helps (less data to move)

---

### Operation Distribution

**CNN-Heavy (Conv2D >70%):**
- Choose high-FLOPS hardware (GPUs, TPUs)
- INT8 quantization effective
- Fusion opportunities high

**Transformer-Heavy (MatMul/Linear >50%):**
- Choose high-bandwidth hardware (AMD EPYC, H100)
- Attention mechanisms memory-intensive
- Batching critical

---

## Available Models

**140+ torchvision models supported**

Discover available models:
```bash
python3 cli/discover_models.py
```

Common models:
- ResNet: resnet18, resnet50, resnet101
- MobileNet: mobilenet_v2, mobilenet_v3_*
- EfficientNet: efficientnet_b0 through b7
- Vision Transformers: vit_b_16, vit_l_16
- ConvNeXt: convnext_tiny, convnext_small

---

## Use Cases

### Use Case 1: Model Selection

**Goal:** Choose between ResNet-50 and EfficientNet-B0

```bash
python3 cli/profile_graph.py --model resnet50
python3 cli/profile_graph.py --model efficientnet_b0
```

**Compare:**
- Total FLOPs (latency proxy)
- Total Memory (memory footprint)
- AI (hardware affinity)

---

### Use Case 2: Hardware Affinity

**Goal:** Determine if model is compute-bound or memory-bound

```bash
python3 cli/profile_graph.py --model vit_b_16 --verbose
```

**Look for:**
- AI values per layer
- Percentage compute-bound vs memory-bound
- Operation type distribution

**Decision:**
- Compute-bound → Choose high-FLOPS hardware
- Memory-bound → Choose high-bandwidth hardware

---

### Use Case 3: Validate Custom Models

**Goal:** Profile your own PyTorch model

```python
# my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return self.net(x)

# Add to MODEL_REGISTRY in profile_graph.py
MODEL_REGISTRY['my_model'] = lambda: MyModel()
```

```bash
python3 cli/profile_graph.py --model my_model
```

---

## Troubleshooting

### Error: "Model not found"

**Solution:**
```bash
python3 cli/discover_models.py
```

Check model name spelling and FX-traceability.

---

### FLOP Mismatch with fvcore

**Expected:** ±5-10% variance

**Reasons:**
1. Different counting methodologies
2. We count fused operations differently
3. Some operations counted/skipped differently

**Rule:** <10% difference is acceptable

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `discover_models.py` | Find available models |
| `partitioner.py` | Apply fusion strategies |
| `analyze_graph_mapping.py` | Map to hardware |

---

## Further Reading

- **Graph Partitioner**: `src/graphs/transform/partitioning/`
- **Architecture Guide**: `CLAUDE.md`

---

## Contact & Feedback

Report issues or request features at the project repository.
