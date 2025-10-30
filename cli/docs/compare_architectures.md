# compare_architectures.py - Multi-Architecture Comparison Tool

**Unified comparison of DNN models across hardware architectures with hierarchical drill-down.**

Compare energy, latency, memory, and utilization across CPU, GPU, TPU, and KPU architectures to understand:
- Which architecture is best for your workload
- **WHY** one architecture is more energy efficient
- Trade-offs between energy and performance
- Architectural differences in resource contention management

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Features](#core-features)
3. [Three Comparison Levels](#three-comparison-levels)
4. [Understanding the Output](#understanding-the-output)
5. [Architecture Progression](#architecture-progression)
6. [Use Cases](#use-cases)
7. [Advanced Options](#advanced-options)
8. [Future Enhancements](#future-enhancements)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Compare ResNet-18 across all architectures:**
```bash
./cli/compare_architectures.py --model resnet18
```

**Output:**
```
================================================================================
Architecture Comparison: resnet18
Batch Size: 1, Precision: fp32
================================================================================

Recommendations:
  Best for Energy:      TPU
  Best for Latency:     GPU
  Best for Throughput:  GPU
  Best Balance:         KPU

Architecture    Energy       Latency      Memory       Util%    vs GPU
--------------------------------------------------------------------------------
CPU             125.30 mJ    2.15 ms      512.00 MB    12.5%    2.56× energy
GPU             48.89 mJ     431.50 µs    54.97 MB     18.2%    baseline     ⭐ (speed)
TPU             15.42 mJ     1.20 ms      256.00 MB    85.0%    0.32× energy  ⭐ (energy)
KPU             62.18 mJ     1.85 ms      128.00 MB    67.3%    1.27× energy

Key Insights:
  • TPU is 3.2× more energy efficient than GPU
  • GPU is 2.8× faster than TPU
  • Trade-off: TPU wins energy, GPU wins latency
  • KPU offers best energy-latency balance with programmability
```

**30-second workflow:**
```bash
# 1. Compare all architectures (summary)
./cli/compare_architectures.py --model resnet18

# 2. Drill down: detailed GPU analysis
./cli/compare_architectures.py --model resnet18 --level detailed --architecture GPU

# 3. Explain difference: why is TPU more efficient?
./cli/compare_architectures.py --model resnet18 --explain-difference GPU TPU --metric energy
```

---

## Core Features

### ✅ Available Now

1. **Multi-Architecture Comparison**
   - CPU (Intel Xeon Platinum 8490H)
   - GPU (NVIDIA H100)
   - TPU (Google TPU v4)
   - KPU (Stillwater KPU-T256)

2. **Hierarchical Drill-Down**
   - Level 0 (summary): Executive summary with recommendations
   - Level 1 (detailed): Per-architecture energy/latency breakdowns
   - Level 2 (subgraph): Per-layer comparison across architectures

3. **Educational Explanations**
   - Not just "what" but "WHY" one architecture is better
   - Architectural difference explanations
   - Trade-off analysis

4. **Flexible Configuration**
   - Any batch size (1, 8, 16, 32, 64, ...)
   - Multiple precisions (FP32, FP16, BF16, INT8, ...)
   - Select specific architectures to compare

### 🚧 Planned Enhancements

- Architectural energy event breakdowns (instruction fetch, coherence, etc.)
- Export formats: JSON, CSV, HTML with interactive charts
- Cost analysis ($/inference based on cloud pricing)
- Batch sweep mode (automatically compare across batch sizes)
- Custom model support (beyond torchvision)

---

## Three Comparison Levels

### Level 0: Summary (Default)

**Executive summary with recommendations and key insights.**

```bash
./cli/compare_architectures.py --model resnet18
```

**Shows:**
- Recommendations (best for energy, latency, throughput, balance)
- Comparison table (energy, latency, memory, utilization)
- Key insights with quantified differences
- Next steps (drill down options)

**Use when:** You need quick answers about which architecture to use.

---

### Level 1: Detailed

**Deep dive into a single architecture's energy and performance breakdown.**

```bash
./cli/compare_architectures.py --model resnet18 \
    --level detailed --architecture GPU
```

**Output:**
```
================================================================================
Detailed Analysis: GPU
================================================================================

Energy Breakdown:
  Total Energy:          48.89 mJ
    ├─ Compute:          18.52 mJ (37.9%)
    └─ Memory:           30.37 mJ (62.1%)

Performance:
  Latency:               431.50 µs
  Throughput:            2,317 inferences/sec
  Utilization:           18.2%

Memory:
  Peak Memory:           54.97 MB
  Activation Memory:     38.24 MB
  Weight Memory:         16.73 MB

Bottleneck Analysis:
  Compute-bound:         12 subgraphs (60%)
  Memory-bound:          8 subgraphs (40%)
  Total subgraphs:       20
```

**Use when:** You need to understand a specific architecture's behavior in detail.

---

### Level 2: Subgraph Comparison

**Per-layer comparison showing which architecture wins for each operation.**

```bash
./cli/compare_architectures.py --model resnet18 \
    --level subgraph
```

**Output:**
```
================================================================================
Subgraph-Level Comparison: resnet18
================================================================================

ID   Operation             CPU          GPU          TPU          KPU          Winner
----------------------------------------------------------------------------------------
0    conv2d_64ch          15.32 µJ     3.84 µJ      0.52 µJ      2.18 µJ      TPU
1    relu                 0.18 µJ      1.25 µJ      0.62 µJ      0.35 µJ      CPU
2    maxpool              2.45 µJ      4.82 µJ      1.15 µJ      1.98 µJ      TPU
3    conv2d_128ch         28.64 µJ     6.12 µJ      0.98 µJ      4.35 µJ      TPU
...

Summary:
  TPU: 15 subgraphs (best) - Dominates Conv2D operations
  CPU: 3 subgraphs (best) - Wins simple element-wise ops
  KPU: 2 subgraphs (best) - Competitive on medium workloads
```

**Use when:** You need to understand which operations favor which architecture.

---

## Understanding the Output

### Metrics Explained

| Metric | Description | Lower is Better? |
|--------|-------------|------------------|
| **Energy** | Total energy per inference (Joules or mJ) | ✅ Yes |
| **Latency** | Time per inference (seconds, ms, or µs) | ✅ Yes |
| **Memory** | Peak memory usage (MB or GB) | ✅ Yes |
| **Util%** | Hardware utilization (0-100%) | ❌ No (higher = better) |
| **Throughput** | Inferences per second | ❌ No (higher = better) |

### Winner Categories

- **Energy Winner** - Lowest energy per inference (best for battery life, operating cost)
- **Latency Winner** - Fastest inference (best for real-time applications)
- **Throughput Winner** - Highest inferences/sec (best for batch processing)
- **Memory Winner** - Lowest peak memory (best for constrained devices)
- **Balance Winner** - Best energy × latency product (overall efficiency)

### Reading Ratios

```
vs GPU
------
0.32×  - This architecture uses 0.32× the energy (3.1× more efficient)
2.56×  - This architecture uses 2.56× the energy (2.6× less efficient)
```

---

## Architecture Progression

The tool compares four distinct resource contention management approaches:

```
CPU (Sequential/Modest Parallelism)
    ↓ Add massive SIMT parallelism + coherence machinery
GPU (Data Parallel SIMT)
    ↓ Eliminate instruction fetch, use fixed spatial schedule
TPU (Systolic Array - Fixed Function)
    ↓ Add programmability with domain tracking
KPU (Domain Flow Architecture - Programmable Spatial)
```

### Key Architectural Differences

| Architecture | Parallelism | Instruction Fetch | Coherence Overhead | Energy Efficiency |
|--------------|-------------|-------------------|-------------------|-------------------|
| **CPU** | 8-16 cores | Yes (per op) | Low | Baseline |
| **GPU** | 270K threads (SIMT) | Yes (per warp) | **High** | 0.4-0.5× CPU (worse!) |
| **TPU** | Systolic array | **No** (preloaded) | None | **5-10× GPU** |
| **KPU** | Programmable spatial | **No** (domain tracking) | None | **2-3× GPU** |

**Why GPU can be worse than CPU for small batches:**
- GPU coherence machinery to manage 270K threads costs more energy than it saves
- Only beneficial when batch size is large enough to amortize coherence overhead
- Rule of thumb: GPU wins at batch ≥ 8-16

**Why TPU/KPU are so efficient:**
- Eliminate instruction fetch overhead (biggest energy cost in von Neumann architectures)
- Spatial data flows remove memory contention
- Pre-designed or domain-tracked schedules avoid runtime arbitration

---

## Use Cases

### Use Case 1: Deployment Architecture Selection

**Goal:** Choose hardware for production deployment

```bash
# Compare for typical inference workload
./cli/compare_architectures.py --model resnet18 --batch-size 1

# If deploying to edge (constrained power budget)
→ Choose: TPU or KPU (lowest energy)

# If deploying to cloud (latency-critical)
→ Choose: GPU (lowest latency)

# If need programmability (changing models)
→ Choose: KPU (programmable, still efficient)
```

---

### Use Case 2: Understanding Energy vs Latency Trade-offs

**Goal:** Quantify the energy-latency trade-off

```bash
# Summary comparison
./cli/compare_architectures.py --model resnet18

# Explain the difference
./cli/compare_architectures.py --model resnet18 \
    --explain-difference GPU TPU --metric energy
```

**Output shows:**
```
Why is GPU different from TPU?

GPU uses 3.2× MORE energy than TPU

Energy Breakdown:
  GPU: 48.89 mJ
    ├─ Compute:  18.52 mJ
    └─ Memory:   30.37 mJ

  TPU: 15.42 mJ
    ├─ Compute:  2.78 mJ
    └─ Memory:   12.64 mJ

Architectural Differences:
  TPU is more energy efficient because it uses a different
  resource contention management strategy.

  TPU eliminates instruction fetch and memory contention overhead
  that dominates GPU energy consumption.
```

---

### Use Case 3: Model Optimization Insights

**Goal:** Identify which layers are inefficient on current hardware

```bash
# Subgraph-level analysis
./cli/compare_architectures.py --model resnet18 \
    --level subgraph \
    --architectures GPU TPU
```

**Insights:**
- If TPU wins most Conv2D layers → Consider TPU deployment
- If CPU wins element-wise ops → GPU/TPU overhead not worth it for simple ops
- If mixed results → Hybrid approach or KPU (programmable) may be best

---

### Use Case 4: Batch Size Impact

**Goal:** Understand how batch size affects architecture choice

```bash
# Small batch (edge inference)
./cli/compare_architectures.py --model resnet18 --batch-size 1

# Large batch (datacenter training/inference)
./cli/compare_architectures.py --model resnet18 --batch-size 64
```

**Expected observations:**
- Batch=1: CPU/TPU/KPU win (GPU coherence overhead not amortized)
- Batch=64: GPU wins (coherence cost amortized over large batch)

---

## Advanced Options

### Select Specific Architectures

```bash
# Compare only GPU and TPU
./cli/compare_architectures.py --model resnet18 \
    --architectures GPU TPU
```

### Change Batch Size

```bash
# Large batch comparison
./cli/compare_architectures.py --model resnet18 \
    --batch-size 32
```

### Change Precision

```bash
# Mixed-precision inference
./cli/compare_architectures.py --model resnet18 \
    --precision FP16

# INT8 quantized inference
./cli/compare_architectures.py --model resnet18 \
    --precision INT8
```

### Explain Specific Metrics

```bash
# Explain latency difference
./cli/compare_architectures.py --model resnet18 \
    --explain-difference GPU TPU --metric latency

# Explain memory difference
./cli/compare_architectures.py --model resnet18 \
    --explain-difference GPU CPU --metric memory
```

### Save Output

```bash
# Save to text file
./cli/compare_architectures.py --model resnet18 \
    --output comparison.txt

# Future: Export to JSON/CSV/HTML (planned)
./cli/compare_architectures.py --model resnet18 \
    --output comparison.json  # Planned
```

---

## Comparison with Other Tools

| Tool | Purpose | Architectures | Drill-Down | Educational |
|------|---------|---------------|------------|-------------|
| **compare_architectures.py** | Multi-arch comparison | 4 (CPU/GPU/TPU/KPU) | ✅ 3 levels | ✅ WHY explanations |
| **analyze_comprehensive.py** | Single-arch deep dive | 1 (specified) | ✅ Very detailed | ⚠️ Technical |
| **compare_models.py** | Model comparison | 1 arch, N models | ❌ Summary only | ❌ No |
| **analyze_batch.py** | Batch size sweep | 1 arch | ✅ Per-batch | ⚠️ Technical |

**When to use compare_architectures.py:**
- Choosing deployment hardware
- Understanding architecture trade-offs
- Educational purposes (learning about different architectures)
- Quick what-if scenarios ("should I use GPU or TPU?")

**When to use other tools:**
- `analyze_comprehensive.py`: Deep analysis of single architecture
- `compare_models.py`: Compare different models on same hardware
- `analyze_batch.py`: Optimize batch size for single architecture

---

## Future Enhancements

### Planned Features

1. **Architectural Energy Breakdowns** (Next)
   - Show instruction fetch, coherence, domain tracking energy
   - Visualize where energy goes for each architecture
   - Quantify specific architectural overheads

2. **Export Formats**
   - JSON: Structured data for programmatic analysis
   - CSV: Spreadsheet import for custom analysis
   - HTML: Interactive charts with Plotly

3. **Cost Analysis**
   - $/inference based on cloud pricing
   - TCO comparison (hardware cost + energy cost)
   - Break-even analysis

4. **Batch Sweep Mode**
   - Automatic comparison across batch sizes [1, 4, 8, 16, 32, 64]
   - Find optimal batch size per architecture
   - Visualize batch size impact

5. **Custom Model Support**
   - Support for user-provided PyTorch modules
   - Not limited to torchvision models

6. **Interactive Visualization**
   - Terminal heatmaps (best arch per layer)
   - Energy breakdown charts
   - Roofline overlays

---

## Troubleshooting

### Model Not Found

**Error:**
```
ValueError: Model 'my_model' not found in torchvision.models
```

**Solution:**
Currently only torchvision models are supported. Use models like:
- `resnet18`, `resnet50`, `resnet152`
- `mobilenet_v2`, `mobilenet_v3_small`
- `efficientnet_b0`, `efficientnet_v2_s`
- `vit_b_16`, `swin_t`

**Future:** Custom model support coming soon.

---

### Architecture Not Available

**Error:**
```
Warning: Failed to create mapper for KPU: ...
```

**Solution:**
Some architectures require specific hardware configurations. Check available mappers:
```bash
./cli/list_hardware_mappers.py
```

---

### Low Utilization Warning

**Output shows 5-20% utilization for GPU**

**This is expected for:**
- Small models (ResNet-18, MobileNet) at batch=1
- Limited parallelism doesn't saturate 132 SMs
- GPU designed for large batches (32-128)

**To improve GPU utilization:**
```bash
# Increase batch size
./cli/compare_architectures.py --model resnet18 --batch-size 32
```

---

### Very High Energy for GPU

**GPU shows more energy than CPU at batch=1**

**This is correct!**
- GPU coherence machinery (managing 270K threads) costs energy
- At small batches, overhead >> computation benefit
- GPU is optimized for large batches, not single inference

**See educational explanation:**
```bash
./cli/compare_architectures.py --model resnet18 \
    --explain-difference CPU GPU --metric energy
```

---

## Examples

### Example 1: Edge Deployment Decision

**Scenario:** Deploy ResNet-18 for real-time video analysis on edge device (15W power budget)

```bash
# Compare with realistic edge batch size
./cli/compare_architectures.py --model resnet18 --batch-size 1 \
    --architectures CPU GPU TPU KPU
```

**Decision criteria:**
1. Energy < 100 mJ per inference (stay within power budget)
2. Latency < 50 ms (real-time requirement)
3. Memory < 512 MB (edge device constraint)

**Expected result:** TPU or KPU winner (low energy, fits constraints)

---

### Example 2: Cloud Cost Optimization

**Scenario:** Reduce cloud inference costs for high-volume service

```bash
# Large batch (datacenter workload)
./cli/compare_architectures.py --model resnet18 --batch-size 64 \
    --architectures GPU TPU
```

**Decision criteria:**
1. $/1M inferences (minimize cost)
2. Throughput > 10K inferences/sec (handle load)

**Expected result:** GPU likely wins at large batch (amortizes overhead)

---

### Example 3: Research & Education

**Scenario:** Teach students about hardware architecture trade-offs

```bash
# Summary comparison
./cli/compare_architectures.py --model resnet18

# Deep dive on GPU
./cli/compare_architectures.py --model resnet18 \
    --level detailed --architecture GPU

# Explain architectural differences
./cli/compare_architectures.py --model resnet18 \
    --explain-difference GPU TPU --metric energy
```

**Learning outcomes:**
- Understand CPU → GPU → TPU → KPU progression
- Quantify energy-latency trade-offs
- Learn why systolic arrays are efficient

---

## Related Documentation

- [analyze_comprehensive.py](analyze_comprehensive.md) - Deep analysis of single architecture
- [compare_models.py](compare_models.md) - Compare models on same hardware
- [analyze_batch.py](analyze_batch.md) - Batch size optimization
- [Hardware Mappers](list_hardware_mappers.md) - Available architectures

---

## Contributing

Found an issue or have a suggestion? Please report it at:
https://github.com/anthropics/claude-code/issues

---

**Last Updated:** 2025-10-30 (Phase 4.2 - Unified Framework)
