# How to Use: partition_analyzer.py

## Overview

`partition_analyzer.py` analyzes and compares different partitioning strategies for FX computational graphs. It quantifies the benefits of operator fusion by comparing unfused (baseline) vs fusion strategies, showing metrics like subgraph reduction and data movement savings.

**Key Capabilities:**
- Partition graphs using different strategies (unfused vs fusion)
- Compare partitioning strategies side-by-side
- Visualize graph structure and partitions
- Quantify benefits of fusion (reduced subgraphs, memory traffic)

**Target Users:**
- Compiler engineers evaluating fusion strategies
- Researchers studying graph transformations
- ML engineers understanding model structure

---

## Quick Start

```bash
# Compare all strategies on ResNet-18
python3 cli/partition_analyzer.py --model resnet18 --strategy all --compare

# Visualize fusion partitioning
python3 cli/partition_analyzer.py --model mobilenet_v2 --strategy fusion --visualize
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | resnet18 | Model name |
| `--strategy` | str | all | Partitioning strategy: unfused, fusion, all |
| `--compare` | flag | False | Show side-by-side comparison |
| `--quantify` | flag | False | Show detailed metrics |
| `--visualize` | flag | False | Show graph visualization |
| `--input-shape` | int[] | 1,3,224,224 | Input tensor shape |

### Range Selection (for visualization)

**Note:** Node numbers are **1-based** (matching display output) and ranges are **inclusive** on both ends.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start` | int | None | Start node (1-based, inclusive) |
| `--end` | int | None | End node (1-based, inclusive) |
| `--around` | int | None | Center node for context view |
| `--context` | int | 10 | Nodes before/after center (with --around) |
| `--max-nodes` | int | 20 | Max nodes from start (backward compatible) |

**Range Selection Priority:**
1. `--around` with `--context` (highest priority)
2. `--start` and/or `--end`
3. `--max-nodes` (default: first 20 nodes)

---

## Partitioning Strategies

### Unfused (Baseline)

One subgraph per operator:
- No fusion
- Maximum subgraph count
- Maximum data movement
- Useful as baseline for comparison

### Fusion

Aggregate operators to minimize data movement:
- Fuses compatible operations (conv+bn+relu)
- Reduces subgraph count (5-10×)
- Minimizes intermediate memory traffic
- Production-ready strategy

---

## Usage Examples

### Example 1: Basic Comparison

```bash
python3 cli/partition_analyzer.py \
  --model resnet50 \
  --strategy all \
  --compare
```

**Output:**
```
================================================================================
PARTITIONING COMPARISON: ResNet-50
================================================================================

Unfused Strategy:
  Subgraphs:        177
  Total FLOPs:      8.21 G
  Total Memory:     52.4 MB
  Data Movement:    high (every operator boundary)

Fusion Strategy:
  Subgraphs:        20
  Total FLOPs:      8.21 G (same)
  Total Memory:     52.4 MB (same)
  Data Movement:    low (fused operator groups)

Fusion Benefits:
  Subgraph Reduction:  88.7% (177 → 20)
  Data Movement:       ~80% reduction
  Kernel Launch:       ~88% reduction
```

---

### Example 2: Visualize Fusion

```bash
# Show first 30 nodes
python3 cli/partition_analyzer.py \
  --model mobilenet_v2 \
  --strategy fusion \
  --visualize \
  --max-nodes 30

# Show specific range (nodes 10-25, inclusive)
python3 cli/partition_analyzer.py \
  --model mobilenet_v2 \
  --strategy fusion \
  --visualize \
  --start 10 --end 25

# Investigate around node 15 (±5 nodes)
python3 cli/partition_analyzer.py \
  --model mobilenet_v2 \
  --strategy fusion \
  --visualize \
  --around 15 --context 5
```

**Output:** ASCII graph showing fused subgraphs with node-by-node details

---

### Example 3: Quantify Fusion Benefits

```bash
python3 cli/partition_analyzer.py \
  --model efficientnet_b0 \
  --strategy fusion \
  --quantify
```

**Output:**
- Per-subgraph FLOPs
- Per-subgraph memory
- Fusion decisions
- Bottleneck analysis

---

## Interpretation

### Subgraph Reduction

**High Reduction (>80%):**
- ✓ Good: Significant fusion opportunities
- Typical for CNNs (Conv+BN+ReLU patterns)

**Low Reduction (<50%):**
- May indicate complex architecture
- Transformers have less fusion opportunity (attention blocks)

### Data Movement

**Fusion Benefits:**
- Eliminates intermediate tensor writes
- Reduces DRAM traffic (40-80%)
- Improves cache utilization

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `profile_graph.py` | Profile graph characteristics |
| `analyze_graph_mapping.py` | Map partitioned graph to hardware |

---

## Further Reading

- **Fusion Partitioner**: `src/graphs/transform/partitioning/fusion_partition_analyzer.py`
- **Architecture Guide**: `CLAUDE.md`
