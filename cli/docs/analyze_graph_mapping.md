# How to Use: analyze_graph_mapping.py

## Overview

`analyze_graph_mapping.py` is a comprehensive graph mapping analysis tool that shows how computational graphs are partitioned and mapped onto hardware resources. It provides detailed insight into resource allocation, utilization, bottlenecks, and optimization opportunities.

**Key Capabilities:**
- Partition computational graphs into fused subgraphs
- Map subgraphs to specific hardware architectures
- Estimate power consumption and latency per subgraph
- Compare multiple hardware targets side-by-side
- Display hardware architecture building blocks (CUDA cores/SM, tiles, etc.)
- Identify performance bottlenecks and low utilization

**Target Users:**
- Compiler engineers optimizing graph mappings
- Hardware designers evaluating architectures
- ML engineers selecting deployment targets
- System architects planning AI infrastructure

---

## Installation

**Requirements:**
```bash
pip install torch torchvision
```

**Verify Installation:**
```bash
python3 cli/analyze_graph_mapping.py --help
```

---

## Basic Usage

### Single Hardware Analysis

Analyze how a model maps to a single hardware target:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet18 \
  --hardware H100
```

**Output Sections:**
1. **Model Information**: Name, batch size, precision
2. **Graph Partitioning**: Total subgraphs, FLOPs, memory
3. **Hardware Specifications**: Architecture, peak GOPS, bandwidth, TDP
4. **Hardware Building Blocks**: CUDA cores/SM, ops/clock, total compute
5. **Subgraph Table**: Per-subgraph allocation, utilization, latency, power
6. **Execution Summary**: Total latency, average power, FPS, energy/inference

### Hardware Comparison

Compare the same model across multiple hardware targets:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --compare "H100,Jetson-Orin-AGX,KPU-T256"
```

**Comparison Output:**
1. **Hardware Architecture Legend**: Building block specs for each target
2. **Comparison Table**: 20+ metrics side-by-side
3. **Detailed Subgraph Comparison**: Allocation patterns across hardware
4. **Performance Ranking**: Sorted by latency
5. **Energy Efficiency Ranking**: Sorted by mJ/inference

---

## Command-Line Arguments

### Required Arguments (One Of)

| Argument | Type | Description |
|----------|------|-------------|
| `--hardware` | str | Single hardware target name |
| `--compare` | str | Comma-separated hardware names for comparison |

**Important:** You must specify either `--hardware` OR `--compare`, but not both.

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | resnet18 | Model name (see Available Models) |
| `--batch-size` | int | 1 | Input batch size |
| `--precision` | str | fp32 | Numeric precision: fp32, fp16, bf16, int8 |

### Power/Thermal Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--thermal-profile` | str | - | Target power budget (e.g., "30W", "7W") |

**Note:** If `--thermal-profile` is not specified, the tool uses the hardware's default TDP.

### Output Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | flag | False | Enable detailed debug output |

---

## Available Hardware

### Datacenter GPUs
- `H100` - NVIDIA H100 SXM5 (80 GB, 700W TDP)
- `H100-PCIe` - NVIDIA H100 PCIe (80 GB, 350W TDP)
- `A100` - NVIDIA A100 SXM4 (80 GB, 400W TDP)
- `A100-PCIe` - NVIDIA A100 PCIe (40 GB, 250W TDP)
- `V100` - NVIDIA V100 (32 GB, 300W TDP)
- `T4` - NVIDIA T4 (16 GB, 70W TDP)

### Edge GPUs
- `Jetson-Orin-AGX` - NVIDIA Jetson AGX Orin (64 GB, 15W-60W)
- `Jetson-Orin-Nano` - NVIDIA Jetson Orin Nano (4-8 GB, 7W-15W)
- `Jetson-Thor` - NVIDIA Jetson Thor (Blackwell 2025)

### Datacenter CPUs
- `Xeon-8490H` - Intel Xeon Platinum 8490H (60-core, 350W)
- `Xeon-8592plus` - Intel Xeon Platinum 8592+ (64-core, 350W)
- `Granite-Rapids` - Intel Granite Rapids (128-core, 500W)
- `EPYC-9654` - AMD EPYC 9654 (96-core, 360W)
- `EPYC-9754` - AMD EPYC 9754 (128-core, 360W)
- `EPYC-Turin` - AMD EPYC Turin (128-core, 500W)
- `AmpereOne-192` - Ampere AmpereOne (192-core, 350W)
- `AmpereOne-128` - Ampere AmpereOne (128-core, 250W)

### Consumer CPUs
- `i7-12700K` - Intel Core i7-12700K (12-core, 125W)
- `Ryzen-7-5800X` - AMD Ryzen 7 5800X (8-core, 105W)

### TPU Accelerators
- `TPU-v4` - Google TPU v4 (350W TDP)
- `Coral-Edge-TPU` - Google Coral Edge TPU (2W TDP)

### KPU Accelerators (Stillwater)
- `KPU-T64` - Stillwater KPU-T64 (64 tiles, 6W)
- `KPU-T256` - Stillwater KPU-T256 (256 tiles, 30W)
- `KPU-T768` - Stillwater KPU-T768 (768 tiles, 100W)

### DSP Processors
- `QRB5165` - Qualcomm QRB5165 (Hexagon 698 DSP)
- `TI-TDA4VM` - Texas Instruments TDA4VM (C7x DSP)

### DPU/FPGA Accelerators
- `DPU-Vitis-AI` - Xilinx Vitis AI DPU

### CGRA Accelerators
- `Plasticine-V2` - Plasticine-style CGRA

**Check Available Hardware:**
```bash
python3 cli/list_hardware_mappers.py
```

---

## Available Models

### ResNet Family
- `resnet18`, `resnet34`, `resnet50`, `resnet101`

### MobileNet Family
- `mobilenet_v2`
- `mobilenet_v3_small`, `mobilenet_v3_large`

### EfficientNet Family
- `efficientnet_b0`, `efficientnet_b1`

### VGG Family
- `vgg16`

**Discover More Models:**
```bash
python3 cli/discover_models.py
```

---

## Common Usage Examples

### Example 1: Basic Model Analysis

Analyze ResNet-18 on H100:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet18 \
  --hardware H100
```

**Use Case:** Understand how ResNet-18 uses H100 SMs, where bottlenecks occur, and overall performance.

---

### Example 2: Edge Deployment Analysis

Analyze MobileNet-V2 on Jetson Orin Nano at 7W:

```bash
python3 cli/analyze_graph_mapping.py \
  --model mobilenet_v2 \
  --hardware Jetson-Orin-Nano \
  --thermal-profile 7W
```

**Use Case:** Evaluate if MobileNet-V2 can run efficiently on battery-powered edge devices.

---

### Example 3: Batch Size Impact

Compare batch=1 vs batch=16:

```bash
# Batch 1
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --hardware H100 \
  --batch-size 1

# Batch 16
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --hardware H100 \
  --batch-size 16
```

**Use Case:** Understand how batching improves SM utilization and throughput.

---

### Example 4: Precision Comparison

Compare FP32 vs INT8 performance:

```bash
# FP32 (default)
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --hardware H100

# INT8
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --hardware H100 \
  --precision int8
```

**Use Case:** Quantify the speedup from reduced precision inference.

---

### Example 5: Hardware Comparison

Compare Jetson AGX Orin vs KPU-T256 at 30W:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet18 \
  --compare "Jetson-Orin-AGX,KPU-T256" \
  --batch-size 1 \
  --thermal-profile 30W
```

**Output Highlights:**
- Hardware architecture legend showing building blocks
- Side-by-side comparison of 20+ metrics
- Per-subgraph allocation patterns
- Performance and efficiency rankings

**Use Case:** Choose between GPU and KPU accelerator for edge AI applications.

---

### Example 6: Multi-Hardware Datacenter Comparison

Compare datacenter accelerators:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --compare "H100,A100,TPU-v4,KPU-T768"
```

**Use Case:** Select the best datacenter accelerator for CNN inference workloads.

---

### Example 7: CPU vs GPU vs Accelerator

Compare different architecture classes:

```bash
python3 cli/analyze_graph_mapping.py \
  --model mobilenet_v2 \
  --compare "Xeon-8490H,H100,TPU-v4,KPU-T256"
```

**Use Case:** Understand trade-offs between programmable ISAs (CPU/GPU) and fixed-function accelerators (TPU/KPU).

---

## Understanding the Output

### Subgraph Table Columns

| Column | Description |
|--------|-------------|
| **ID** | Subgraph index (0-based) |
| **Ops** | Fused operation types (conv2d+bn+relu) |
| **FLOPs** | Floating-point operations (G = billions) |
| **Bytes** | Memory traffic (MB) |
| **AI** | Arithmetic Intensity (FLOPs/Byte) |
| **Bottleneck** | compute-bound or memory-bound |
| **Units** | Hardware units allocated (SMs, tiles, cores) |
| **Util%** | Compute utilization percentage |
| **Latency** | Execution time (ms) |
| **Power** | Power consumption (W) |

### Hardware Building Blocks Section

Shows the compute microarchitecture:

```
Jetson-Orin-AGX (GPU):
  Total Units: 16 SMs
  Architecture:
    - 128 CUDA cores per SM
    - 2.0 ops/clock/core (FMA)
    - 0.65 GHz clock (sustained)
    → 166.4 GOPS per SM
    → 2662.4 GOPS total (16 SMs)
    - 4 Tensor Cores per SM (matrix ops)
  Memory:
    - Bandwidth: 204.8 GB/s
    - L1 per unit: 128 KB
    - L2 total: 4.0 MB
    - Main memory: 64.0 GB
```

**Why This Matters:**
- **166.4 GOPS per SM**: This is the max performance if all 128 CUDA cores are busy
- **Utilization %**: If you see 50% util, only 64 CUDA cores are active
- **Bottleneck Identification**: Low util + compute-bound → insufficient parallelism

---

## Interpretation Guide

### High Utilization (>90%)
- ✓ Good: Hardware is well-utilized
- Watch: May be power-limited if at max TDP

### Medium Utilization (50-90%)
- Moderate: Some inefficiency
- Consider: Batching, fusion, or different hardware

### Low Utilization (<50%)
- ⚠ Problem: Wasting hardware resources
- Root Causes:
  - Insufficient parallelism (small feature maps)
  - Memory-bound operations (high bandwidth requirements)
  - Poor graph partitioning (small subgraphs)

### Bottleneck Type

| Type | Meaning | Optimization |
|------|---------|--------------|
| **compute-bound** | Limited by compute units | Good for high-FLOPS hardware |
| **memory-bound** | Limited by bandwidth | Need high-bandwidth systems |

**Rule of Thumb:** Arithmetic Intensity > 50 → compute-bound, AI < 50 → memory-bound

---

## Performance Optimization Tips

### Increase Utilization

1. **Increase Batch Size**
   ```bash
   --batch-size 16  # or 32, 64
   ```
   - More parallelism across samples
   - Better SM/tile occupancy

2. **Change Precision**
   ```bash
   --precision int8
   ```
   - Higher throughput (4× TOPS for INT8 vs FP32)
   - May improve utilization on tensor cores

3. **Choose Matching Hardware**
   - High-parallelism workloads → GPU, TPU, large KPU
   - Low-parallelism workloads → CPU, small KPU
   - Memory-bound → High-bandwidth CPU (AMD EPYC)
   - Compute-bound → High-FLOPS GPU (H100) or accelerator

### Reduce Power Consumption

1. **Use Thermal Profiles**
   ```bash
   --thermal-profile 30W
   ```
   - Clocks down to meet power budget
   - Trades performance for efficiency

2. **Choose Efficient Architecture**
   - KPU accelerators: Best TOPS/W for dataflow workloads
   - TPU: Good for matrix-heavy ops
   - Edge GPUs: Balanced performance/power

### Improve Latency

1. **Use High-Performance Hardware**
   - H100 > A100 > V100 for datacenter
   - Jetson Orin AGX > Orin Nano for edge

2. **Optimize Batch Size**
   - Batch=1 for lowest latency
   - Larger batches increase throughput but latency

3. **Check for Allocation Collapse**
   - Look for subgraphs with only 1-2 units allocated
   - Consider CPU offload for these subgraphs

---

## Hardware Architecture Notes

### GPUs (NVIDIA)

**Building Block:** Streaming Multiprocessor (SM)
- 128 CUDA cores per SM (Ampere, Blackwell)
- 64 CUDA cores per SM (Volta)
- 2.0 FP32 ops/clock/core (FMA instruction)

**Allocation:** Round up to nearest SM, max utilization per SM

### KPU Accelerators (Stillwater)

**Building Block:** Heterogeneous Tiles
- KPU-T64: 48 INT8 tiles + 12 BF16 tiles + 4 Matrix tiles
- KPU-T256: 179 INT8 + 45 BF16 + 32 Matrix tiles
- Tile selection based on operation type and precision

**Allocation:** Round up to nearest tile group

**Performance Characteristic:** Requires high parallelism
- Excels at early CNN layers (high parallelism)
- Struggles with late layers (low parallelism) → allocation collapse

### TPU (Google)

**Building Block:** Systolic Array Tiles
- 128×128 matrix units per tile
- Fixed-function matrix multiplication

**Allocation:** Fixed tile count, utilization varies

### CPU (Intel, AMD, Ampere)

**Building Block:** CPU Core
- Intel: AVX-512 SIMD (16 FP32 ops/cycle)
- AMD: AVX2 SIMD (8 FP32 ops/cycle)
- Ampere: ARM NEON (4 FP32 ops/cycle)

**Allocation:** All cores allocated, utilization computed

### DSP (Qualcomm, TI)

**Building Block:** Vector Unit
- Qualcomm HVX: 1024-bit vector ops
- TI C7x: 512-bit vector ops

**Allocation:** Vector units per operation

---

## Troubleshooting

### Error: "Unknown hardware: X"

**Solution:** Check available hardware names:
```bash
python3 cli/list_hardware_mappers.py
```

Use exact name including hyphens (e.g., `Jetson-Orin-AGX`, not `JetsonOrinAGX`).

---

### Error: "Unknown model: X"

**Solution:** Check available models:
```bash
python3 cli/discover_models.py
```

Model names are case-insensitive but must match exactly.

---

### Low Utilization on KPU

**Root Cause:** KPU requires high parallelism. Late CNN layers have small feature maps.

**Solutions:**
- Increase batch size
- Use larger models (more channels)
- Consider hybrid deployment (KPU for early layers, CPU for late layers)

---

### Comparison Table Too Wide

**Issue:** 3+ hardware comparison creates wide tables (>200 columns)

**Workaround:** Use vertical format (planned for future release) or run pairwise comparisons.

---

## Advanced Usage

### Custom Thermal Profiles

Create custom power budgets:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --hardware Jetson-Orin-AGX \
  --thermal-profile 30W
```

The mapper adjusts clocks to meet the power budget.

---

### Scripting and Automation

Run batch analyses:

```bash
for model in resnet18 resnet50 mobilenet_v2; do
  for hw in H100 Jetson-Orin-AGX KPU-T256; do
    python3 cli/analyze_graph_mapping.py \
      --model $model \
      --hardware $hw > results_${model}_${hw}.txt
  done
done
```

---

### Export for Further Analysis

Redirect output to files:

```bash
python3 cli/analyze_graph_mapping.py \
  --model resnet50 \
  --compare "H100,A100,TPU-v4" > comparison_report.txt
```

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `list_hardware_mappers.py` | Discover available hardware |
| `compare_models.py` | Compare different models on same hardware |
| `discover_models.py` | Find FX-traceable models |
| `profile_graph.py` | Hardware-independent graph profiling |

---

## Further Reading

- **Session Log**: `docs/sessions/2025-10-26_hardware_comparison_and_jetson_fix.md`
- **Jetson Specs**: `docs/hardware/jetson_specifications.md`
- **Hardware Comparison**: `docs/DATACENTER_CPU_COMPARISON.md`
- **Architecture Guide**: `CLAUDE.md`

---

## Contact & Feedback

Report issues or request features at the project repository.
