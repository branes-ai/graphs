# How to Use: graph_explorer.py

## Overview

`graph_explorer.py` is an interactive tool for exploring PyTorch FX computational graphs. It provides three progressive modes (discovery → summary → detailed visualization) to help you understand model structure, operation characteristics, and graph properties without being overwhelmed by output.

**Key Capabilities:**
- Side-by-side FX graph and partition visualization
- Flexible range selection (start/end, around, max-nodes)
- Multiple model support (20+ torchvision models)
- Shows operation details, FLOPs, arithmetic intensity, bottlenecks
- Partition reasoning displayed
- Export to file for sharing and documentation

**Target Users:**
- Compiler engineers debugging graph transformations
- ML engineers understanding model structure
- Performance engineers identifying optimization opportunities
- Researchers studying fusion patterns

---

## Quick Start

The tool has three modes of operation:

**Level 1: Discover Models (no arguments)**
```bash
# Show available models organized by family
python3 cli/graph_explorer.py
```

**Level 2: Model Summary (model only)**
```bash
# Get comprehensive summary statistics without overwhelming output
python3 cli/graph_explorer.py --model resnet18
```

**Level 3: Visualization (model + range)**
```bash
# Visualize specific sections
python3 cli/graph_explorer.py --model resnet18 --max-nodes 20
python3 cli/graph_explorer.py --model resnet18 --around 35 --context 10
```

**Design Philosophy:** Start high-level (summary) → drill down (visualization) to prevent accidental output floods.

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | resnet18 | Model name from torchvision |
| `--input-shape` | str | 1,3,224,224 | Input tensor shape (comma-separated) |
| `--start` | int | None | Start node index (0-based, inclusive) |
| `--end` | int | None | End node index (exclusive) |
| `--around` | int | None | Center node for context view |
| `--context` | int | 10 | Nodes before/after center (with --around) |
| `--max-nodes` | int | None | Max nodes from start (backward compatible) |
| `--output` | str | None | Save visualization to file |

### Range Selection Methods

**Method 1: Explicit Range (--start, --end)**
- Best for: Systematic exploration of graph sections
- Example: `--start 20 --end 50` shows nodes 20-49

**Method 2: Context View (--around, --context)**
- Best for: Debugging specific nodes
- Example: `--around 35 --context 10` shows nodes 25-45

**Method 3: Max Nodes (--max-nodes)**
- Best for: Quick overview from beginning
- Example: `--max-nodes 50` shows first 50 nodes

**Note:** Only use one method at a time. Methods are mutually exclusive.

---

## Three Modes of Operation

The tool progressively discloses information to prevent overwhelming output:

### Mode 1: Model Discovery (No Arguments)

**Invocation:**
```bash
python3 cli/graph_explorer.py
```

**Output:**
- Error message: "Please specify a model with --model"
- List of 20+ supported models organized by family
- Usage examples

**Use Case:** Discover what models are available

---

### Mode 2: Model Summary (Model Only)

**Invocation:**
```bash
python3 cli/graph_explorer.py --model MODEL_NAME
```

**Output:**
- Total FX nodes and subgraphs
- Total FLOPs, MACs, memory traffic
- Arithmetic intensity (average and range)
- Bottleneck distribution (compute-bound, bandwidth-bound, etc.)
- Operation type distribution (top 10)
- Partition reason distribution
- Guidance on how to visualize specific sections

**Use Case:** Understand model characteristics before diving into visualization

**Example Output:**
```
MODEL SUMMARY: resnet18
================================================================================

Total FX Nodes:        71
Partitioned Subgraphs: 68
Nodes Not Partitioned: 3

Total FLOPs:           3.64 GFLOPs
Total MACs:            1814.07 M
Total Memory Traffic:  116.95 MB

Arithmetic Intensity:
  Average: 24.8 FLOPs/byte
  Range:   0.1 - 166.0 FLOPs/byte

Bottleneck Distribution:
  balanced            :   9 ( 13.2%)
  bandwidth_bound     :  46 ( 67.6%)
  compute_bound       :  13 ( 19.1%)

Operation Type Distribution:
  batchnorm           :  20 ( 29.4%)
  conv2d              :  17 ( 25.0%)
  relu                :  17 ( 25.0%)
  ...
```

**Why Summary Mode Exists:**
- Large models (e.g., ViT-L) have 300+ nodes
- Prevents accidental output floods
- Provides context before drilling down
- Shows what's interesting to investigate

---

### Mode 3: Visualization (Model + Range or Output)

**Invocation:**
```bash
# With range selection
python3 cli/graph_explorer.py --model MODEL_NAME --max-nodes 20
python3 cli/graph_explorer.py --model MODEL_NAME --start 10 --end 30
python3 cli/graph_explorer.py --model MODEL_NAME --around 35 --context 10

# With output file (saves full visualization)
python3 cli/graph_explorer.py --model MODEL_NAME --output viz.txt
```

**Output:**
- Side-by-side FX graph and partition visualization
- Node-by-node details (operation, FLOPs, arithmetic intensity, bottlenecks)
- Partition reasoning for each subgraph
- Usage tips

**Use Case:** Detailed investigation of specific graph sections

---

### Typical Workflow

```bash
# Step 1: See what models are available
./cli/graph_explorer.py
# → Choose: resnet18, mobilenet_v2, vit_b_16, etc.

# Step 2: Get model summary
./cli/graph_explorer.py --model efficientnet_b0
# → See: 347 nodes, 125 subgraphs, bottleneck distribution

# Step 3: Investigate interesting sections
./cli/graph_explorer.py --model efficientnet_b0 --around 100 --context 15
# → Visualize: nodes 85-115 in detail
```

---

## Supported Models

### CNNs (Convolutional Networks)
- **ResNet family**: resnet18, resnet34, resnet50, resnet101, resnet152
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **EfficientNet**: efficientnet_b0 through efficientnet_b4
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base

### Transformers
- **Vision Transformer (ViT)**: vit_b_16, vit_b_32, vit_l_16
- **Swin Transformer**: swin_t, swin_s, swin_b

---

## Usage Examples

### Example 1: Discover and Understand a Model

**Scenario:** You want to analyze a model but don't know which one to use

```bash
# Step 1: See available models
python3 cli/graph_explorer.py

# Step 2: Get summary for ResNet-18
python3 cli/graph_explorer.py --model resnet18

# Step 3: Visualize interesting section (first 20 nodes)
python3 cli/graph_explorer.py --model resnet18 --max-nodes 20
```

**Step 2 Output (Summary):**
```
MODEL SUMMARY: resnet18
================================================================================

Total FX Nodes:        71
Partitioned Subgraphs: 68

Total FLOPs:           3.64 GFLOPs
Total Memory Traffic:  116.95 MB

Arithmetic Intensity:
  Average: 24.8 FLOPs/byte

Bottleneck Distribution:
  bandwidth_bound     :  46 ( 67.6%)  ← Most ops are bandwidth-bound
  compute_bound       :  13 ( 19.1%)
  balanced            :   9 ( 13.2%)

Operation Type Distribution:
  batchnorm           :  20 ( 29.4%)  ← Lots of BatchNorm
  conv2d              :  17 ( 25.0%)  ← Standard convolutions
  relu                :  17 ( 25.0%)

NEXT STEPS: Visualize Specific Sections
[... guidance on --max-nodes, --start/--end, --around ...]
```

**Use Case:** Progressive discovery - models → summary → detailed visualization

---

### Example 2: Investigate Specific Node

**Scenario:** Debugging partitioning decision around node 35

```bash
python3 cli/graph_explorer.py \
  --model resnet18 \
  --around 35 \
  --context 5
```

**What It Shows:**
- Nodes 30-40 (35 ± 5)
- Useful for understanding local partitioning decisions
- See fusion candidates and bottleneck changes

**Use Case:** Understanding why a specific node wasn't fused

---

### Example 3: Explore Specific Range

**Scenario:** Examine ResNet-18 layer2 (nodes 20-40)

```bash
python3 cli/graph_explorer.py \
  --model resnet18 \
  --start 20 \
  --end 40
```

**Use Case:** Studying specific architectural blocks (e.g., residual blocks)

---

### Example 4: Compare Different Models

**Scenario:** Understand MobileNetV2 vs ResNet-18 partitioning

```bash
# MobileNetV2 (inverted residuals)
python3 cli/graph_explorer.py \
  --model mobilenet_v2 \
  --max-nodes 30 \
  --output mobilenet_v2_viz.txt

# ResNet-18 (standard residuals)
python3 cli/graph_explorer.py \
  --model resnet18 \
  --max-nodes 30 \
  --output resnet18_viz.txt

# Compare files
diff mobilenet_v2_viz.txt resnet18_viz.txt
```

**Key Differences:**
- MobileNetV2: Depthwise + Pointwise convs (different arithmetic intensity)
- ResNet-18: Standard convs (higher compute intensity)

---

### Example 5: Vision Transformer Structure

**Scenario:** Understand ViT attention block partitioning

```bash
python3 cli/graph_explorer.py \
  --model vit_b_16 \
  --start 50 \
  --end 100
```

**What You'll See:**
- Multi-head attention operations
- LayerNorm operations
- MLP blocks
- Different bottleneck patterns than CNNs (more memory-bound)

**Use Case:** Understanding transformer-specific partitioning

---

### Example 6: Full Graph Documentation

**Scenario:** Generate complete graph visualization for documentation

```bash
python3 cli/graph_explorer.py \
  --model efficientnet_b0 \
  --output efficientnet_b0_full.txt

# View in pager
less efficientnet_b0_full.txt

# Or filter for specific info
grep "Bottleneck: compute_bound" efficientnet_b0_full.txt
```

**Use Case:**
- Documentation and reporting
- Sharing with team
- Archiving partitioning decisions

---

### Example 7: Custom Input Shape

**Scenario:** Analyze partitioning for different input resolution

```bash
# Standard resolution (224×224)
python3 cli/graph_explorer.py \
  --model resnet18 \
  --input-shape 1,3,224,224 \
  --max-nodes 10

# High resolution (512×512)
python3 cli/graph_explorer.py \
  --model resnet18 \
  --input-shape 1,3,512,512 \
  --max-nodes 10
```

**What Changes:**
- FLOPs scale with spatial dimensions
- Arithmetic intensity may change
- Memory traffic increases

---

## Output Format

### Visualization Structure

```
==================================================================================================================
GRAPH PARTITIONING VISUALIZATION
==================================================================================================================

FX Graph (Execution Order)                            Partitioned Subgraphs
--------------------------------------------------    ------------------------------------------------------------

[node_num]. [node_type] node_name                     >> Subgraph #N: subgraph_name
   Operation details                                     Type, FLOPs, Arithmetic Intensity
                                                         Bottleneck, Partition Reason
```

### Node Types

| Type | Description | Example |
|------|-------------|---------|
| `placeholder` | Graph input | `x` |
| `call_module` | Module invocation | `conv1`, `bn1` |
| `call_function` | Function call | `add`, `mul` |
| `call_method` | Method call | `flatten` |
| `get_attr` | Attribute access | `weight` |
| `output` | Graph output | `output` |

### Subgraph Information

**Line 1:** Subgraph name and ID
**Line 2:** Operation type, FLOPs
**Line 3:** Arithmetic intensity, bottleneck type
**Line 4:** Partition reason, fusion candidates

---

## Interpretation Guide

### Understanding Arithmetic Intensity

**Arithmetic Intensity (AI) = FLOPs / Memory Traffic (bytes)**

| AI Value | Category | Typical Operations | Hardware Preference |
|----------|----------|-------------------|-------------------|
| > 100 | Very High | Large conv kernels | GPU (compute-bound) |
| 50-100 | High | Standard convs | GPU/TPU |
| 10-50 | Moderate | Small convs, matmul | Balanced hardware |
| 1-10 | Low | Pointwise conv, elementwise | Memory bandwidth critical |
| < 1 | Very Low | BatchNorm, activation | Memory-bound |

**Implications:**
- **High AI**: Focus on compute throughput (GPU cores, TPU systolic arrays)
- **Low AI**: Focus on memory bandwidth (HBM, on-chip SRAM)

---

### Bottleneck Types

| Bottleneck | Meaning | Optimization Strategy |
|------------|---------|----------------------|
| `compute_bound` | Limited by FLOPs | Add compute units, increase clock |
| `bandwidth_bound` | Limited by memory BW | Increase bandwidth, fusion |
| `memory_bound` | Limited by memory size | Increase cache, tiling |
| `balanced` | Well-balanced | General optimization |

**Key Insight:** Fusion is most effective when combining compute_bound ops with bandwidth_bound ops.

---

### Partition Reasons

| Reason | Meaning | Action |
|--------|---------|--------|
| `fusion_opportunity` | Could be fused | Consider fusion policy |
| `operation_boundary` | Natural boundary | Expected |
| `data_dependency` | Data forces split | Unavoidable |
| `fusion_incompatible` | Can't fuse | Expected (e.g., pooling) |
| `memory_limit_exceeded` | Too large to fuse | Reduce fusion threshold |

**Fusion Opportunity:**
- Indicates potential optimization
- Shows number of fusion candidates
- Review fusion policy if many opportunities missed

---

### Fusion Patterns

**Common Fusion Patterns (CNNs):**
- Conv → BatchNorm → ReLU (3 ops → 1 subgraph)
- Depthwise Conv → Pointwise Conv (2 ops → 1 subgraph)
- MatMul → Bias → Activation (3 ops → 1 subgraph)

**Limited Fusion (Transformers):**
- Attention operations less fusible
- More memory-bound operations
- Smaller subgraphs typical

---

## Troubleshooting

### Issue: "Model not supported"

**Error:**
```
Error: Model 'my_custom_model' not supported
Supported models: resnet18, mobilenet_v2, ...
```

**Solution:**
- Use one of the 20+ supported torchvision models
- Or use the Python API directly (see `examples/graph_explorer.py`)

---

### Issue: Cannot Use Multiple Range Methods

**Error:**
```
Error: Cannot use multiple range selection methods simultaneously
Choose one: --start/--end, --around/--context, or --max-nodes
```

**Solution:**
```bash
# Wrong (multiple methods)
python3 cli/graph_explorer.py --model resnet18 --start 10 --around 20

# Correct (one method)
python3 cli/graph_explorer.py --model resnet18 --around 20 --context 5
```

---

### Issue: Invalid Input Shape

**Error:**
```
Error: Invalid input shape '1-3-224-224'
Expected format: 1,3,224,224
```

**Solution:**
- Use commas (not hyphens or spaces)
- Format: `batch,channels,height,width`
- Example: `--input-shape 1,3,512,512`

---

### Issue: Range Selection Not Showing Expected Nodes

**Note in Output:**
```
Note: Displaying from beginning to node 50
      (Range selection starting at node 7 not yet fully supported)
```

**Explanation:**
- Current implementation shows nodes from beginning up to end
- Starting at arbitrary offset not yet implemented
- Use `--around` for investigating middle sections

**Workaround:**
```bash
# To see nodes 30-50, use:
python3 cli/graph_explorer.py --model resnet18 --around 40 --context 10
```

---

### Issue: Output Too Long for Terminal

**Problem:** Visualization scrolls off screen

**Solutions:**

1. **Limit nodes:**
```bash
python3 cli/graph_explorer.py --model resnet18 --max-nodes 20
```

2. **Use pager:**
```bash
python3 cli/graph_explorer.py --model resnet18 | less
```

3. **Save to file:**
```bash
python3 cli/graph_explorer.py --model resnet18 --output viz.txt
less viz.txt
```

---

## Advanced Usage

### Scripted Analysis

**Goal:** Analyze multiple models automatically

```bash
#!/bin/bash
# analyze_all_resnets.sh

for model in resnet18 resnet34 resnet50 resnet101 resnet152; do
  echo "Analyzing $model..."
  python3 cli/graph_explorer.py \
    --model $model \
    --max-nodes 50 \
    --output "${model}_viz.txt"
done

# Generate summary
grep "Total FX nodes:" *_viz.txt > summary.txt
grep "Partitioned subgraphs:" *_viz.txt >> summary.txt
```

---

### Filtering Specific Information

**Goal:** Extract only compute-bound operations

```bash
python3 cli/graph_explorer.py \
  --model resnet18 \
  --output full_viz.txt

grep "compute_bound" full_viz.txt
```

**Goal:** Find all fusion opportunities

```bash
python3 cli/graph_explorer.py \
  --model mobilenet_v2 \
  --output full_viz.txt

grep "fusion_opportunity" full_viz.txt | wc -l
```

---

### Comparing Architectures

**Goal:** Compare CNN vs Transformer bottleneck distribution

```bash
# CNN bottlenecks
python3 cli/graph_explorer.py --model resnet50 --output resnet50.txt
grep "Bottleneck:" resnet50.txt | sort | uniq -c

# Transformer bottlenecks
python3 cli/graph_explorer.py --model vit_b_16 --output vit_b_16.txt
grep "Bottleneck:" vit_b_16.txt | sort | uniq -c
```

**Expected Results:**
- **CNNs**: More `compute_bound` (convolutions)
- **Transformers**: More `bandwidth_bound` (attention, layer norm)

---

### Integration with Analysis Pipeline

**Goal:** Visualize → Analyze → Report

```bash
# Step 1: Visualize
python3 cli/graph_explorer.py \
  --model efficientnet_b0 \
  --output step1_viz.txt

# Step 2: Profile
python3 cli/profile_graph.py \
  --model efficientnet_b0 \
  --output step2_profile.json

# Step 3: Analyze mapping
python3 cli/analyze_graph_mapping.py \
  --model efficientnet_b0 \
  --hardware jetson-orin-nano \
  --output step3_mapping.txt

# Step 4: Generate report
cat step1_viz.txt step2_profile.json step3_mapping.txt > final_report.txt
```

---

### Custom Visualization Focus

**Goal:** Show only interesting sections (high FLOP operations)

```bash
# Generate full visualization
python3 cli/graph_explorer.py --model resnet50 --output full.txt

# Extract high-FLOP operations (>100M FLOPs)
awk '/FLOPs:/ {
  if ($2 ~ /^[1-9][0-9]{2,}\./ || $2 ~ /[1-9]\.[0-9]+G/) print
}' full.txt
```

---

## Related Tools

### Complementary Tools

**Before visualization:**
- **[discover_models.py](discover_models.md)** - Find FX-traceable models
- **[profile_graph.py](profile_graph.md)** - Understand model characteristics

**After visualization:**
- **[partition_analyzer.py](partition_analyzer.md)** - Apply partitioning strategies, quantify benefits
- **[analyze_graph_mapping.py](analyze_graph_mapping.md)** - Map to specific hardware

**For comparison:**
- **[compare_models.py](compare_models.md)** - Compare multiple models across hardware

---

### Workflow Integration

**Typical Workflow:**

1. **Discover:** `discover_models.py` → Find traceable models
2. **Profile:** `profile_graph.py` → Understand FLOPs, memory, AI
3. **Visualize:** `graph_explorer.py` → See partitioning decisions
4. **Partition:** `partition_analyzer.py` → Apply and quantify strategies
5. **Map:** `analyze_graph_mapping.py` → Map to target hardware

---

## Further Reading

### Documentation
- **Package Organization**: `CLAUDE.md` - Architecture overview
- **Partition Structures**: `src/graphs/ir/structures.py` - Data structures
- **Graph Partitioner**: `src/graphs/transform/partitioning/graph_partition_analyzer.py` - Implementation

### Examples
- **API Example**: `examples/graph_explorer.py` - Simple Python API usage
- **Partitioning Demo**: `examples/demo_fusion_comparison.py` - Fusion benefits

### Related Sessions
- **2025-10-19**: Graph partitioning implementation
- **2025-10-20**: Fusion partitioning strategy
- **2025-10-24**: Package reorganization

---

## Summary

`graph_explorer.py` is a powerful debugging and analysis tool with progressive disclosure:

**Three Modes:**
1. **Model Discovery** (no args) - List 20+ available models
2. **Model Summary** (--model) - Statistics without visualization flood
3. **Visualization** (--model + range/output) - Detailed side-by-side view

**Key Features:**
- ✓ Progressive disclosure prevents output floods
- ✓ Comprehensive summary statistics (FLOPs, bottlenecks, operation distribution)
- ✓ Flexible range selection (start/end, around, max-nodes)
- ✓ 20+ model support (CNNs, Transformers)
- ✓ Detailed operation metrics (FLOPs, AI, bottlenecks)
- ✓ Partition reasoning displayed
- ✓ Export to file

**Best Practices:**
1. **Start with summary mode** to understand model characteristics
2. **Use summary statistics** to identify interesting sections
3. **Then drill down** with --around or --max-nodes
4. **Export full visualization to file** for large models
5. **Combine with other tools**:
   - `partition_analyzer.py` for strategy comparison
   - `analyze_graph_mapping.py` for hardware-specific analysis

**Typical Workflow:**
```bash
# 1. Discover models
./cli/graph_explorer.py

# 2. Get summary
./cli/graph_explorer.py --model efficientnet_b0

# 3. Investigate specific sections
./cli/graph_explorer.py --model efficientnet_b0 --around 100 --context 15
```

**Next Steps:**
- Try the three modes in order
- Use summary mode to understand model characteristics
- Drill down to interesting sections
- Compare different architectures
- Integrate into your analysis workflow
