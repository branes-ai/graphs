# How to Use: partitioner.py

## Overview

`partitioner.py` applies graph partitioning strategies to computational graphs, comparing different fusion approaches and visualizing the results.

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
python3 cli/partitioner.py --model resnet18 --strategy all --compare

# Visualize fusion partitioning
python3 cli/partitioner.py --model mobilenet_v2 --strategy fusion --visualize
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
| `--max-nodes` | int | 50 | Max nodes in visualization |
| `--input-shape` | int[] | 1,3,224,224 | Input tensor shape |

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
python3 cli/partitioner.py \
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
python3 cli/partitioner.py \
  --model mobilenet_v2 \
  --strategy fusion \
  --visualize \
  --max-nodes 30
```

**Output:** ASCII graph showing fused subgraphs

---

### Example 3: Quantify Fusion Benefits

```bash
python3 cli/partitioner.py \
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

- **Fusion Partitioner**: `src/graphs/transform/partitioning/fusion_partitioner.py`
- **Architecture Guide**: `CLAUDE.md`
