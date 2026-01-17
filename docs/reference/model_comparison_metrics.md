# Model Comparison Metrics

## Overview

When comparing DNN models, different metrics reveal different aspects of model efficiency and characteristics. This guide explains what each metric means and when to use it.

## Quick Reference

```bash
# Compare models
python cli/compare_models.py resnet18 mobilenet_v2 efficientnet_b0

# Sort by different metrics
python cli/compare_models.py MODEL1 MODEL2 --sort-by [params|flops|macs|memory|efficiency|ai]
```

## Metric Categories

### 1. Model Size

**Parameters**
- **What**: Total number of learnable weights/biases
- **Units**: Typically shown as K/M (thousands/millions)
- **Interpretation**:
  - Affects storage cost (model file size)
  - Affects memory needed to load model
  - Does NOT directly indicate compute cost
- **Example**: ResNet-18 has 11.69M parameters = ~45 MB (at float32)

**Trainable Parameters**
- **What**: Parameters that are updated during training
- **Why**: Some models have frozen layers (transfer learning)
- **Usually**: Same as total parameters for standard models

**Model Size (MB)**
- **What**: Disk space needed for model weights
- **Calculation**: Parameters × 4 bytes (float32)
- **Note**: Can be reduced via quantization (int8 = 1 byte)

**Layers**
- **What**: Number of operations in the computational graph
- **Interpretation**: Indicates model depth
- **Note**: Includes all ops (conv, relu, add, etc.)

**Max Width**
- **What**: Maximum number of channels in any layer
- **Interpretation**: Indicates "wideness" of network
- **Example**: ResNet-18 goes up to 512 channels, MobileNet-V2 up to 1280

### 2. Computational Cost

**MACs (Multiply-Accumulate Operations)**
- **What**: Number of multiply-add pairs
- **Units**: K/M/G (thousands/millions/billions)
- **Interpretation**:
  - Primary measure of compute cost for CNNs
  - 1 convolution MAC = `C_in × K_h × K_w` operations per output pixel
- **Example**: ResNet-18 = 1.81 GMACs
- **Note**: Industry standard for CNN complexity

**FLOPs (Floating-Point Operations)**
- **What**: Total arithmetic operations
- **Relationship**: FLOPs ≈ 2 × MACs (for convolutions)
  - Each MAC = 1 multiply + 1 add = 2 FLOPs
  - Other ops (BatchNorm, ReLU) add additional FLOPs
- **Interpretation**: Total compute regardless of operation type

**FLOPs per Parameter**
- **What**: How many operations does each parameter perform?
- **Calculation**: Total FLOPs / Total Parameters
- **Interpretation**:
  - **High ratio** (>300): Parameters are reused heavily (e.g., ResNet)
  - **Low ratio** (<200): Parameters underutilized (e.g., EfficientNet)
  - Indicates computational efficiency of architecture
- **Example**:
  - ResNet-18: 311 FLOPs/param (each weight used ~155 times)
  - EfficientNet-B0: 152 FLOPs/param (more params, less reuse)

**Params per MAC**
- **What**: Storage cost per unit of computation
- **Interpretation**:
  - **Low** (<0.01): Compute-heavy, few weights (efficient storage)
  - **High** (>0.02): Memory-heavy, many weights per operation

### 3. Memory Traffic

**Total Memory**
- **What**: Sum of all data movement (input + output + weights)
- **Units**: MB/GB
- **Interpretation**:
  - Measures memory bandwidth requirements
  - High memory = potential bottleneck on memory-limited hardware
- **Note**: Differs from peak activation memory

**Input Memory**
- **What**: Bytes read from input tensors
- **Interpretation**: Data loaded from previous layers

**Output Memory**
- **What**: Bytes written to output tensors
- **Interpretation**: Data produced for next layers

**Weight Memory**
- **What**: Bytes read from parameters
- **Interpretation**: Model weight accesses during forward pass

**Bytes per FLOP**
- **What**: Memory traffic per unit of computation
- **Calculation**: Total Memory / Total FLOPs
- **Interpretation**:
  - **Low** (<0.1): Compute-bound, efficient memory usage
  - **High** (>0.3): Memory-bound, lots of data movement

### 4. Arithmetic Intensity (AI)

**Average AI**
- **What**: FLOPs per byte of memory traffic
- **Units**: FLOPs/byte
- **Interpretation**:
  - **High AI (>50)**: Compute-bound operations
    - Limited by GPU/CPU compute throughput
    - Can saturate ALUs
    - Example: Large matrix multiplies
  - **Medium AI (10-50)**: Balanced
    - Both compute and memory matter
  - **Low AI (<10)**: Memory-bound operations
    - Limited by memory bandwidth
    - Cannot saturate compute units
    - Example: Depthwise convolutions, BatchNorm
- **Critical**: Determines hardware bottleneck

**Compute-Bound %**
- **What**: Percentage of operations with AI > 50
- **Interpretation**:
  - High %: Model is compute-limited on most hardware
  - Low %: Model will be memory-bandwidth limited

**Memory-Bound %**
- **What**: Percentage of operations with AI < 10
- **Interpretation**:
  - High %: Model needs high memory bandwidth
  - Indicates potential for optimization (fusion, tiling)

**Classification**
- **Compute-bound**: Avg AI > 50 (limited by FLOPS)
- **Balanced**: Avg AI 10-50 (both matter)
- **Memory-bound**: Avg AI < 10 (limited by bandwidth)

## Roofline Analysis Context

The Arithmetic Intensity is key to roofline analysis:

```
Performance (FLOPS) = min(
    Peak Compute,
    Arithmetic Intensity × Memory Bandwidth
)
```

**Example Hardware** (NVIDIA A100):
- Peak Compute: 19.5 TFLOPS (FP32)
- Memory Bandwidth: 1,555 GB/s
- Ridge Point: 19.5 TFLOPS / 1,555 GB/s = 12.5 FLOPs/byte

**Interpretation**:
- Operations with AI < 12.5: Memory-bound (bottlenecked by bandwidth)
- Operations with AI > 12.5: Compute-bound (bottlenecked by ALUs)

## Efficiency Rankings

The comparison tool shows:

1. **Smallest/Largest Model**: Parameter efficiency
2. **Least/Most FLOPs**: Computational efficiency
3. **Least/Most Memory**: Memory efficiency
4. **Compute per Param**: Architectural efficiency
5. **Highest/Lowest AI**: Hardware utilization characteristics

## Comparison Examples

### Mobile Architectures (Low Resource)

```bash
python cli/compare_models.py mobilenet_v2 mobilenet_v3_small squeezenet1_0 efficientnet_b0
```

**Key Metrics**:
- Parameters (minimize for storage)
- MACs (minimize for battery)
- Memory (minimize for edge devices)

### High-Performance Servers (Maximize Accuracy)

```bash
python cli/compare_models.py resnet50 resnet101 resnet152 convnext_base
```

**Key Metrics**:
- FLOPs/param (computational efficiency)
- AI (hardware utilization)
- Throughput (images/sec - if available)

### Transformer vs CNN

```bash
python cli/compare_models.py resnet50 vit_b_16 swin_t
```

**Expected Differences**:
- ViT: Higher params, lower AI (attention is memory-bound)
- ResNet: Fewer params, higher AI (convolutions are compute-bound)
- Swin: Hybrid characteristics

## Interpreting Results

### Case Study: ResNet-18 vs MobileNet-V2

| Metric | ResNet-18 | MobileNet-V2 | Winner | Why |
|--------|-----------|--------------|--------|-----|
| Parameters | 11.69M | 3.50M | MobileNet | 3.3× smaller model |
| MACs | 1.81G | 300M | MobileNet | 6× less compute |
| FLOPs/param | 311 | 183 | ResNet | Better parameter reuse |
| Memory | 117 MB | 173 MB | ResNet | Less data movement |
| Avg AI | 24.76 | 4.68 | ResNet | More compute-bound |

**Conclusion**:
- **MobileNet-V2**: Better for mobile (small, fast)
- **ResNet-18**: Better for GPUs (higher AI, better utilization)

### Trade-offs

**High FLOPs/param** (e.g., ResNet)
- ✅ Good: Efficient use of parameters
- ✅ Good: Better for compute-heavy hardware
- ❌ Bad: More total computation

**High Memory Traffic** (e.g., MobileNet)
- ✅ Good: Fewer total FLOPs
- ❌ Bad: Memory bandwidth bottleneck
- ❌ Bad: Harder to optimize

**High Arithmetic Intensity**
- ✅ Good: GPU/TPU friendly
- ✅ Good: Saturates compute units
- ❌ Bad: Requires more FLOPs

**Low Arithmetic Intensity**
- ✅ Good: Fewer FLOPs
- ❌ Bad: Memory bandwidth limited
- ❌ Bad: Poor hardware utilization

## Recommended Comparisons

### By Use Case

**Mobile/Edge**:
```bash
python cli/compare_models.py mobilenet_v2 mobilenet_v3_small squeezenet1_0 efficientnet_b0 mnasnet1_0
```
Sort by: `--sort-by macs` or `--sort-by params`

**Server/GPU**:
```bash
python cli/compare_models.py resnet50 resnet101 densenet121 convnext_base
```
Sort by: `--sort-by efficiency` or `--sort-by ai`

**Research/Exploration**:
```bash
python cli/compare_models.py resnet50 vit_b_16 swin_t convnext_base efficientnet_b4
```
Sort by: Any metric to understand trade-offs

### By Architecture Family

**ResNet Family**:
```bash
python cli/compare_models.py resnet18 resnet34 resnet50 resnet101 resnet152
```
Shows scaling trends: parameter growth vs compute growth

**EfficientNet Family**:
```bash
python cli/compare_models.py efficientnet_b0 efficientnet_b1 efficientnet_b4 efficientnet_b7
```
Shows compound scaling (width + depth + resolution)

**Vision Transformers**:
```bash
python cli/compare_models.py vit_b_16 vit_b_32 vit_l_16
```
Shows patch size and model size effects

## Advanced Metrics (Future Work)

Consider adding:
- **Latency** (ms per image) - from actual hardware measurements
- **Throughput** (images/sec) - from benchmarking
- **Energy** (mJ per inference) - from power measurements
- **Accuracy** (Top-1/Top-5) - from validation set
- **Memory Peak** (max activation size) - from profiling
- **Cache Efficiency** - from hardware counters
- **Parallelism** - layer-wise vs data parallelism opportunities

These require actual hardware profiling or accuracy measurements.

## Summary

**Choose metrics based on constraints**:
- **Storage limited**: Minimize parameters
- **Compute limited**: Minimize MACs/FLOPs
- **Memory limited**: Minimize memory traffic
- **Latency critical**: Minimize layers (depth)
- **Throughput critical**: Maximize AI (hardware utilization)

**The "best" model depends on your deployment target and constraints.**
