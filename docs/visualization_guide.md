# Fusion Partitioner Visualization Guide

This guide covers the enhanced visualization capabilities of the FusionBasedPartitioner.

## Overview

The fusion partitioner now supports three visualization modes:
1. **Color-Coded Terminal Output** - Color codes subgraphs by bottleneck type
2. **DOT/Graphviz Export** - Export fusion graph for external visualization tools
3. **ASCII Fallback** - Automatic terminal capability detection with graceful degradation

## Features

### 1. Color-Coded Visualization

Subgraphs are color-coded based on their performance bottleneck type:

| Bottleneck Type | Color | Meaning |
|----------------|-------|---------|
| **Compute-Bound** | 🟢 Green | Good for GPU/TPU acceleration |
| **Balanced** | 🔵 Cyan | Good utilization of both compute and memory |
| **Memory-Bound** | 🟡 Yellow | May benefit from additional fusion |
| **Bandwidth-Bound** | 🔴 Red | Limited by memory bandwidth |

#### Usage

```bash
# Color-coded visualization
python cli/partitioner.py --model resnet18 --strategy fusion --visualize --color

# Limit number of nodes shown
python cli/partitioner.py --model resnet18 --strategy fusion --visualize --color --max-nodes 20
```

#### Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUSION-BASED PARTITIONING (Color-Coded by Bottleneck)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

────────────────────────────────────────────────────────
LEGEND: Bottleneck Types
────────────────────────────────────────────────────────
  COMPUTE_BOUND          - Compute-bound (good for GPU/TPU)
  BALANCED               - Balanced (good utilization)
  MEMORY_BOUND           - Memory-bound (may need fusion)
  BANDWIDTH_BOUND        - Bandwidth-bound (memory bandwidth limited)
────────────────────────────────────────────────────────

FX Graph                                   Fused Subgraphs (Color-Coded)
─────────────────────────────────────    ──────────────────────────────────────

1. [call_module] conv1                    ┌─ SUBGRAPH #1 ──────────────
   Conv2d                                 │  Pattern: Conv2d_BatchNorm2d_ReLU
                                          │  Operators: 3
                                          │  Type: COMPUTE_BOUND  (in green)
                                          │
                                          │  ├ conv1 (Conv2d)

2. [call_module] bn1                      │  ├ bn1 (BatchNorm2d)
   BatchNorm2d

3. [call_module] relu                     │  ├ relu (ReLU)
   ReLU                                   │
                                          │  Compute: 236.03MFLOPs
                                          │  Memory: 3.85MB
                                          │  Saved: 6.42MB (62.5%)
                                          │  AI: 61.3 FLOPs/byte  (in green)
                                          └─────────────────────────────
```

### 2. DOT/Graphviz Export

Export the fusion graph to DOT format for visualization with Graphviz tools.

#### Usage

```bash
# Export to DOT file
python cli/partitioner.py --model resnet18 --strategy fusion --export-dot fusion_graph.dot

# Generate PNG visualization
dot -Tpng fusion_graph.dot -o fusion_graph.png

# Generate SVG (scalable)
dot -Tsvg fusion_graph.dot -o fusion_graph.svg

# Generate PDF
dot -Tpdf fusion_graph.dot -o fusion_graph.pdf
```

#### DOT Graph Features

- **Nodes**: Colored boxes representing fused subgraphs
  - Green: Compute-bound operations
  - Cyan: Balanced operations
  - Yellow: Memory-bound operations
  - Red: Bandwidth-bound operations

- **Node Labels**: Include:
  - Subgraph ID
  - Fusion pattern (e.g., "Conv2d_BatchNorm2d_ReLU")
  - Number of operators
  - FLOPs (in GFLOPs)
  - Arithmetic intensity
  - Bottleneck type

- **Edges**: Gray arrows showing data dependencies between subgraphs

- **Legend**: Built-in legend showing color meanings

#### Example DOT Output

```dot
digraph FusionGraph {
    rankdir=TB;
    node [shape=box, style=rounded];
    edge [color=gray];

    sg_0 [label="Subgraph 0\nPattern: Conv2d_BatchNorm2d_ReLU\nOps: 3\nFLOPs: 0.24G\nAI: 61.3\nType: compute_bound",
          fillcolor="#228B22", style="filled,rounded"];

    sg_1 [label="Subgraph 1\nPattern: Unfused\nOps: 1\nFLOPs: 0.00G\nAI: 0.0\nType: bandwidth_bound",
          fillcolor="#DC143C", style="filled,rounded"];

    sg_0 -> sg_1;  // Data dependency
    ...
}
```

### 3. Terminal Capability Detection

The visualization system automatically detects terminal capabilities and adjusts output accordingly.

#### Detection Levels

1. **TRUECOLOR** - 24-bit color support
2. **COLOR** - ANSI color support (most modern terminals)
3. **UTF8** - UTF-8 box drawing characters (no color)
4. **BASIC** - ASCII-only (maximum compatibility)

#### Automatic Fallback

```
Terminal Type          → Output Mode
─────────────────────────────────────
Modern terminal        → UTF-8 + Color
iTerm2/Terminal.app    → UTF-8 + Color
VS Code terminal       → UTF-8 + Color
Output redirection     → ASCII only
NO_COLOR env set       → UTF-8 no color
--no-color flag        → UTF-8 no color
Legacy terminal        → ASCII fallback
```

#### Manual Control

```bash
# Force color on (even if piped)
python cli/partitioner.py --model resnet18 --strategy fusion --visualize --color

# Force color off (ASCII + UTF-8)
python cli/partitioner.py --model resnet18 --strategy fusion --visualize --no-color

# Disable all special characters (for logging)
NO_COLOR=1 python cli/partitioner.py --model resnet18 --strategy fusion --visualize
```

#### Box Drawing Characters

**UTF-8 Mode:**
```
┌─ SUBGRAPH #1 ──────────────
│  Pattern: Conv2d_BatchNorm2d
│  ├ conv (Conv2d)
│  ├ bn (BatchNorm2d)
└─────────────────────────────
```

**ASCII Mode:**
```
+- SUBGRAPH #1 --------------
|  Pattern: Conv2d_BatchNorm2d
|  + conv (Conv2d)
|  + bn (BatchNorm2d)
+-----------------------------
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--visualize` | Enable partitioning visualization |
| `--color` | Use color-coded visualization (fusion only) |
| `--no-color` | Disable colors (use UTF-8 box drawing only) |
| `--export-dot FILE` | Export to DOT/Graphviz format |
| `--max-nodes N` | Limit visualization to N nodes |

## Examples

### Example 1: Quick Visual Inspection

```bash
# Color-coded overview with legend
python cli/partitioner.py \
    --model resnet18 \
    --strategy fusion \
    --visualize \
    --color \
    --max-nodes 15
```

### Example 2: Full Graph Visualization

```bash
# Export full graph for detailed analysis
python cli/partitioner.py \
    --model efficientnet_b0 \
    --strategy fusion \
    --export-dot efficientnet_fusion.dot

# Generate high-res PNG
dot -Tpng -Gdpi=300 efficientnet_fusion.dot -o efficientnet_fusion.png
```

### Example 3: Comparative Analysis

```bash
# Generate DOT files for multiple models
for model in resnet18 mobilenet_v2 efficientnet_b0; do
    python cli/partitioner.py \
        --model $model \
        --strategy fusion \
        --export-dot "${model}_fusion.dot"

    dot -Tsvg "${model}_fusion.dot" -o "${model}_fusion.svg"
done
```

### Example 4: CI/CD Compatible Output

```bash
# ASCII-only output for logs
python cli/partitioner.py \
    --model resnet50 \
    --strategy fusion \
    --visualize \
    --no-color \
    > fusion_report.txt
```

## Programmatic Usage

You can also use the visualization features programmatically:

```python
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.characterize.fusion_partitioner import FusionBasedPartitioner

# Load and trace model
model = models.resnet18(weights=None)
model.eval()

input_tensor = torch.randn(1, 3, 224, 224)
fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)

# Partition with fusion
partitioner = FusionBasedPartitioner()
report = partitioner.partition(fx_graph)

# Color-coded visualization
viz = partitioner.visualize_partitioning_colored(
    fx_graph,
    max_nodes=20,
    use_color=True  # or None for auto-detect
)
print(viz)

# Export to DOT
partitioner.export_to_graphviz(fx_graph, "resnet18_fusion.dot")
```

## Color Scheme

### ANSI Colors Used

```python
COMPUTE_BOUND     = GREEN      # #228B22 / ANSI Green
BALANCED          = CYAN       # #87CEEB / ANSI Cyan
MEMORY_BOUND      = YELLOW     # #FFD700 / ANSI Yellow
BANDWIDTH_BOUND   = RED        # #DC143C / ANSI Red
```

### Interpreting Colors

**🟢 Green (Compute-Bound)**
- High arithmetic intensity (>50 FLOPs/byte)
- Ideal for GPU/TPU acceleration
- Well-suited for parallel execution
- Example: Large Conv2d operations

**🔵 Cyan (Balanced)**
- Moderate arithmetic intensity (10-50 FLOPs/byte)
- Good balance of compute and memory
- Efficient hardware utilization
- Example: Mid-sized convolutions

**🟡 Yellow (Memory-Bound)**
- Lower arithmetic intensity (1-10 FLOPs/byte)
- May benefit from additional fusion
- Consider memory optimization
- Example: BatchNorm, small convolutions

**🔴 Red (Bandwidth-Bound)**
- Very low arithmetic intensity (<1 FLOPs/byte)
- Limited by memory bandwidth
- Minimal computation relative to data movement
- Example: Pooling, element-wise operations

## Tips and Best Practices

1. **Use DOT export for large graphs** - Terminal visualization is limited to ~100 nodes for readability

2. **Leverage color coding** - Quickly identify optimization opportunities by looking for red/yellow subgraphs

3. **Compare before/after** - Export DOT graphs before and after optimization to visualize improvements

4. **Check terminal support** - Modern terminals (iTerm2, VS Code, Windows Terminal) have best support

5. **Use SVG for documentation** - SVG export is scalable and perfect for documentation

6. **Combine with balance analysis** - Use `--analyze-balance` with `--color` for comprehensive insights

## Troubleshooting

### Colors Not Showing

```bash
# Check if TERM is set
echo $TERM

# Try forcing color on
python cli/partitioner.py --model resnet18 --strategy fusion --visualize --color
```

### Box Drawing Characters Broken

```bash
# Check terminal encoding
echo $LANG

# Force ASCII mode
python cli/partitioner.py --model resnet18 --strategy fusion --visualize --no-color
```

### DOT File Not Generating Image

```bash
# Install Graphviz
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Windows:
# Download from https://graphviz.org/download/

# Verify installation
dot -V
```

## Environment Variables

| Variable | Effect |
|----------|--------|
| `NO_COLOR` | Disable all colors |
| `TERM=dumb` | Force basic ASCII mode |
| `LANG=*.UTF-8` | Enable UTF-8 box drawing |

## Future Enhancements

Potential future additions:
- Interactive HTML visualization
- D3.js graph export
- Mermaid diagram format
- Comparison diff visualization
- Performance heatmaps
- Timeline/Gantt chart view

## References

- [Graphviz Documentation](https://graphviz.org/documentation/)
- [DOT Language](https://graphviz.org/doc/info/lang.html)
- [ANSI Color Codes](https://en.wikipedia.org/wiki/ANSI_escape_code)
- [Unicode Box Drawing](https://en.wikipedia.org/wiki/Box-drawing_character)
