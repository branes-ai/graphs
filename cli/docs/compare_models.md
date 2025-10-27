# How to Use: compare_models.py

## Overview

`compare_models.py` analyzes a single DNN model across multiple hardware architectures to determine the best deployment target. It provides operation-level analysis, resource utilization, and performance rankings.

**Key Capabilities:**
- Compare one model across multiple hardware targets
- Analyze operation characteristics (Conv2D, MatMul, etc.)
- Show latency, throughput, and energy efficiency
- Rank hardware by performance metrics
- Filter by deployment scenario (datacenter, edge, embedded)

**Target Users:**
- ML engineers selecting deployment hardware
- System architects planning AI infrastructure
- Product managers evaluating cost/performance trade-offs

---

## Installation

**Requirements:**
```bash
pip install torch torchvision
```

**Verify Installation:**
```bash
python3 cli/compare_models.py --help
```

---

## Basic Usage

### Quick Start

Compare ResNet-50 across all datacenter hardware:

```bash
python3 cli/compare_models.py resnet50 --deployment datacenter
```

### Compare Specific Hardware

```bash
python3 cli/compare_models.py mobilenet_v2 \
  --hardware H100 A100 Jetson-Orin-Nano KPU-T64
```

### Edge Deployment Comparison

```bash
python3 cli/compare_models.py mobilenet_v2 --deployment edge
```

---

## Command-Line Arguments

### Positional Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `model` | str | Model name (required) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--deployment` | str | datacenter | Deployment scenario: datacenter, edge, embedded |
| `--hardware` | str[] | (all) | Specific hardware names (space-separated) |
| `--precision` | str | fp32 | Numeric precision: fp32, fp16, bf16, int8 |
| `--batch-size` | int | 1 | Input batch size |
| `--verbose` | flag | False | Show detailed analysis |

---

## Available Models

### Classification Models

**ResNet Family:**
- `resnet18`, `resnet50`, `resnet101`

**MobileNet Family:**
- `mobilenet_v2`
- `mobilenet_v3_small`, `mobilenet_v3_large`

**EfficientNet Family:**
- `efficientnet_b0`, `efficientnet_b2`, `efficientnet_b4`

**Other CNNs:**
- `vgg16`, `densenet121`, `shufflenet_v2_x1_0`, `squeezenet1_0`

**Vision Transformers:**
- `vit_b_16`, `vit_l_16` (ViT-Base, ViT-Large @ 16×16 patches)
- `vit_b_32`, `vit_l_32` (ViT-Base, ViT-Large @ 32×32 patches)

**ConvNeXt:**
- `convnext_small`, `convnext_base`

---

## Deployment Scenarios

### Datacenter

**Hardware Included:**
- Intel Xeon Platinum 8490H (60-core CPU)
- AMD EPYC 9654 (96-core CPU)
- NVIDIA V100, T4, A100, H100 (GPUs)
- Google TPU v4

**Use Cases:**
- Cloud inference
- High-throughput serving
- Multi-tenant workloads

**Example:**
```bash
python3 cli/compare_models.py resnet50 --deployment datacenter
```

---

### Edge

**Hardware Included:**
- Jetson Orin Nano (7W GPU)
- Stillwater KPU-T64 (6W accelerator)
- Qualcomm QRB5165 (DSP)
- Xilinx DPU (FPGA accelerator)

**Use Cases:**
- Autonomous robots
- Smart cameras
- Drones
- Edge servers

**Example:**
```bash
python3 cli/compare_models.py mobilenet_v2 --deployment edge
```

---

### Embedded

**Hardware Included:**
- Qualcomm QRB5165 (DSP)
- TI TDA4VM (DSP)
- Stillwater KPU-T64 (3W mode)
- CGRA Plasticine

**Use Cases:**
- Battery-powered devices
- IoT sensors
- Wearables
- Low-power always-on inference

**Example:**
```bash
python3 cli/compare_models.py mobilenet_v3_small --deployment embedded
```

---

## Output Format

### Summary Section

```
==============================================================================
MODEL ANALYSIS: ResNet-50
==============================================================================

Model Characteristics:
  Total Subgraphs:        50
  Conv2D Operations:      85% (42 subgraphs)
  Linear Operations:      10% (5 subgraphs)
  Pooling Operations:     5% (3 subgraphs)
```

### Hardware Comparison Table

```
Hardware Comparison (Sorted by Latency)
------------------------------------------------------------------------------
Hardware            Type    Latency   FPS     Power   FPS/W   Energy    Rank
                            (ms)              (W)             (mJ/inf)
------------------------------------------------------------------------------
H100                GPU     0.63      1587    380     4.18    239.5     1
A100                GPU     0.89      1124    320     3.51    284.8     2
TPU-v4              TPU     1.05      952     350     2.72    367.6     3
Jetson-Orin-Nano    GPU     5.45      183     15      12.23   81.9      4
KPU-T64             KPU     4.19      239     6       39.79   25.1      5
QRB5165             DSP     12.34     81      5       16.20   61.7      6
```

### Per-Hardware Details

```
------------------------------------------------------------------------------
NVIDIA H100 (GPU)
------------------------------------------------------------------------------
  Architecture:     HOPPER
  SMs:              132
  Memory:           80 GB (3.35 TB/s)
  TDP:              700W (datacenter), 350W (PCIe)

  Performance:
    Latency:        0.63 ms
    Throughput:     1587 FPS
    Power:          380 W
    Efficiency:     4.18 FPS/W
    Energy/inf:     239.5 mJ

  Utilization:
    Average:        98.2%
    Min:            85.4% (pooling layers)
    Max:            100% (conv2d layers)

  Bottleneck Distribution:
    Compute-bound:  90% of subgraphs
    Memory-bound:   10% of subgraphs
```

---

## Common Usage Examples

### Example 1: Choose Datacenter GPU

**Goal:** Select the best datacenter GPU for ResNet-50 inference

```bash
python3 cli/compare_models.py resnet50 \
  --hardware H100 A100 V100 T4
```

**Key Metrics:**
- **Latency** → Lowest for single-inference
- **FPS** → Highest throughput
- **FPS/W** → Best energy efficiency

**Typical Results:**
- **H100**: Highest throughput (1587 FPS)
- **T4**: Best efficiency (12.5 FPS/W)
- **V100**: Balanced performance/cost

---

### Example 2: Edge Deployment Selection

**Goal:** Choose edge hardware for MobileNet-V2 on a robot

```bash
python3 cli/compare_models.py mobilenet_v2 --deployment edge
```

**Considerations:**
- **Power Budget**: 5-15W typical for mobile robots
- **Latency**: <50ms for real-time control
- **FPS/W**: Maximize battery life

**Typical Results:**
- **KPU-T64 @ 6W**: Best FPS/W (39.79)
- **Jetson Orin Nano @ 7W**: Highest throughput
- **QRB5165**: Lowest power (5W)

---

### Example 3: Vision Transformer Deployment

**Goal:** Analyze ViT-Base across architectures

```bash
python3 cli/compare_models.py vit_b_16 --deployment datacenter
```

**Why Different from CNNs:**
- Higher MatMul operation ratio (75% vs 85% Conv2D)
- More memory-bound (larger tensors)
- Better on high-bandwidth hardware (AMD EPYC, H100)

---

### Example 4: INT8 Quantization Speedup

**Goal:** Quantify INT8 performance gain

```bash
# FP32 baseline
python3 cli/compare_models.py resnet50 --precision fp32

# INT8 quantized
python3 cli/compare_models.py resnet50 --precision int8
```

**Expected Speedup:**
- **GPUs (Tensor Cores)**: 4-8× faster
- **CPUs (AMX/AVX-VNNI)**: 2-4× faster
- **Accelerators (TPU/KPU)**: 2-4× faster

---

### Example 5: Batch Size Impact

**Goal:** Understand batching trade-offs

```bash
python3 cli/compare_models.py resnet50 \
  --hardware H100 \
  --batch-size 1

python3 cli/compare_models.py resnet50 \
  --hardware H100 \
  --batch-size 16
```

**Trade-offs:**
- **Batch=1**: Lowest latency (0.63ms), good for real-time
- **Batch=16**: Higher throughput (12000 FPS), better for offline

---

## Interpretation Guide

### Hardware Rankings

**By Latency (Lower is Better):**
- Best for: Real-time inference, low-latency serving
- Typical Winner: High-end datacenter GPUs (H100, A100)

**By Throughput/FPS (Higher is Better):**
- Best for: Batch inference, high-load serving
- Typical Winner: Same as latency (H100, A100)

**By Energy Efficiency FPS/W (Higher is Better):**
- Best for: Cost-sensitive deployments, sustainability
- Typical Winners: Edge accelerators (KPU-T64), efficient GPUs (T4)

**By Energy per Inference mJ/inf (Lower is Better):**
- Best for: Battery-powered devices, thermal constraints
- Typical Winners: Low-power accelerators (KPU-T64, Coral TPU)

---

### Utilization Metrics

| Utilization | Interpretation | Action |
|-------------|----------------|--------|
| **>90%** | Excellent - hardware well-utilized | ✓ Good match |
| **70-90%** | Good - minor inefficiencies | Consider batching |
| **50-70%** | Fair - some waste | Check memory bandwidth |
| **<50%** | Poor - significant waste | Wrong hardware choice |

---

### Bottleneck Distribution

**Compute-Bound (90%+ of subgraphs):**
- Choose high-FLOPS hardware (H100, TPU)
- INT8 quantization will help significantly
- Batching improves utilization

**Memory-Bound (>30% of subgraphs):**
- Choose high-bandwidth hardware (AMD EPYC 9654, H100 HBM3)
- Reduce precision (less memory traffic)
- Consider data layout optimizations

**Mixed (50/50):**
- Balanced hardware (A100, Jetson Orin)
- Profile carefully to identify optimization opportunities

---

## Hardware-Specific Notes

### NVIDIA GPUs

**Strengths:**
- Excellent for CNNs (Conv2D heavy)
- High utilization (>95%) typical
- Tensor Cores accelerate INT8/FP16

**Weaknesses:**
- High power consumption (250-700W)
- Expensive for edge deployment

**Best For:** Datacenter inference, high-throughput workloads

---

### Google TPU

**Strengths:**
- Excellent for matrix operations (MatMul heavy)
- Fixed-function design → predictable performance
- Good energy efficiency

**Weaknesses:**
- Lower utilization on non-matrix ops (pooling, activation)
- Less flexible than GPUs

**Best For:** Vision Transformers, large batch inference

---

### KPU Accelerators

**Strengths:**
- Exceptional energy efficiency (40+ FPS/W)
- Heterogeneous tiles handle mixed precision
- Low power (6-30W)

**Weaknesses:**
- Requires high parallelism (struggles on late CNN layers)
- Allocation collapse on small feature maps

**Best For:** Early/mid CNN layers, edge inference, battery-powered

---

### CPUs

**Strengths:**
- Programmable (easy deployment)
- No special drivers/runtimes
- Good for small batches

**Weaknesses:**
- Lower peak FLOPS than accelerators
- Higher latency than GPUs

**Best For:** Development, debugging, low-volume inference

---

### DSPs

**Strengths:**
- Integrated in mobile SoCs (no separate chip)
- Low power (2-5W)
- Good for signal processing + AI hybrid workloads

**Weaknesses:**
- Lower performance than dedicated accelerators
- Limited memory bandwidth

**Best For:** Mobile devices, automotive ADAS, always-on inference

---

## Troubleshooting

### Error: "Model not found"

**Solution:** Check available models:
```bash
python3 cli/discover_models.py
```

### Very Low Utilization (<30%)

**Root Causes:**
1. **Wrong hardware choice** - model doesn't match architecture
2. **Insufficient parallelism** - batch size too small
3. **Memory-bound** - operations are bandwidth-limited

**Solutions:**
- Increase batch size (`--batch-size 16`)
- Choose different hardware category
- Use INT8 precision (reduces memory traffic)

### Unexpected Hardware Winner

**Example:** "Why is KPU-T64 faster than Jetson on ResNet-50?"

**Answer:** Check per-layer performance (use `--verbose`):
- KPU may excel on early layers but collapse on late layers
- Overall winner depends on model characteristics
- Use detailed analysis to understand bottlenecks

---

## Advanced Usage

### Compare Specific Hardware Mix

Mix hardware categories:

```bash
python3 cli/compare_models.py resnet50 \
  --hardware H100 Xeon-8490H KPU-T256 TPU-v4
```

### Verbose Mode for Debugging

See per-subgraph analysis:

```bash
python3 cli/compare_models.py mobilenet_v2 \
  --hardware Jetson-Orin-Nano --verbose
```

Output includes:
- Per-subgraph allocation details
- Utilization breakdown
- Bottleneck identification

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `analyze_graph_mapping.py` | Detailed per-subgraph analysis for one model |
| `list_hardware_mappers.py` | Discover available hardware |
| `discover_models.py` | Find FX-traceable models |
| `profile_graph.py` | Hardware-independent graph profiling |

---

## Further Reading

- **Hardware Comparison**: `cli/docs/analyze_graph_mapping.md`
- **Hardware Discovery**: `cli/docs/list_hardware_mappers.md`
- **Architecture Guide**: `CLAUDE.md`

---

## Contact & Feedback

Report issues or request features at the project repository.
