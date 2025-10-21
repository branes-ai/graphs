# Hierarchical Module Table

## Overview

The hierarchical table formatter provides a fvcore-style table view showing module structure, parameters, compute (MACs/FLOPs), and **memory usage** (new!).

## Features

✅ **Hierarchical structure** - Shows module organization like fvcore
✅ **Parameter counts** - Total parameters per module with shapes
✅ **Compute metrics** - MACs or FLOPs (selectable)
✅ **Memory column** - Total memory required (input + output + weights)
✅ **Aggregated statistics** - Parent modules show sum of children

## Usage

```bash
# Show ResNet-18 with MACs
python cli/show_table.py --model resnet18

# Show with FLOPs instead of MACs
python cli/show_table.py --model resnet18 --flops

# Other models
python cli/show_table.py --model resnet50
python cli/show_table.py --model mobilenet_v2
```

## Example Output

```
| Module                         | #Parameters          | MACs         | Memory       |
|:-----------------------------|:-------------------|:-----------|:-----------|
| model                          | 11.690M              | 1.814G       | 116.95MB     |
| layer1                         | 147.968K             | 462.422M     | 16.65MB      |
|  layer1.0                      | 73.984K              | 231.211M     | 8.33MB       |
|   bn1                          | 128                  |              | 1.61MB       |
|    layer1.0.bn1.weight         | (64,)                |              |              |
|    layer1.0.bn1.bias           | (64,)                |              |              |
|   conv1                        | 36.864K              | 115.606M     | 1.75MB       |
|    layer1.0.conv1.weight       | (64, 64, 3, 3)       |              |              |
|   conv2                        | 36.864K              | 115.606M     | 1.75MB       |
|    layer1.0.conv2.weight       | (64, 64, 3, 3)       |              |              |
|  layer1.1                      | 73.984K              | 231.211M     | 8.33MB       |
|   ...                          |                      |              |              |
| layer2                         | 525.568K             | 411.042M     | 12.54MB      |
|  layer2.0                      | 230.144K             | 179.831M     | 7.35MB       |
|   downsample                   | 8.448K               | 6.423M       | 2.04MB       |
|    downsample.0                | 8.192K               | 6.423M       | 1.24MB       |
|   ...                          |                      |              |              |
```

## Comparison with FVCore

| Feature | FVCore | Our Tool |
|---------|--------|----------|
| Hierarchical structure | ✓ | ✓ |
| Parameter counts | ✓ | ✓ |
| Parameter shapes | ✓ | ✓ |
| FLOPs/MACs | ✓ | ✓ (both) |
| Memory usage | ✗ | ✓ **NEW!** |
| Shows activations | ✗ | ✓ |
| Shows add operations | ✗ | ✓ |

## Memory Column Details

The **Memory** column shows total memory traffic for each operation:
- **Input bytes**: Memory read from input tensors
- **Output bytes**: Memory written to output tensors
- **Weight bytes**: Memory read from parameters (weights, bias, etc.)

**Total Memory** = Input + Output + Weights

This is useful for:
- Understanding memory bandwidth requirements
- Calculating arithmetic intensity (FLOPs/byte)
- Identifying memory-bound vs compute-bound operations
- Optimizing for memory hierarchy (cache, DRAM, etc.)

### Example

For a Conv2d layer:
- Input: 1×64×56×56×4 bytes = 803 KB
- Output: 1×64×56×56×4 bytes = 803 KB
- Weights: 64×64×3×3×4 bytes = 147 KB
- **Total: 1.75 MB**

## Hierarchy Construction

The table reconstructs PyTorch module hierarchy from FX graph nodes:

### Node Name Conversion
```
FX node name         →  Module path
"conv1"              →  "conv1"
"layer1_0_conv1"     →  "layer1.0.conv1"
"layer2_0_downsample_0" →  "layer2.0.downsample.0"
```

### Parent Aggregation

Parent modules are created by aggregating children:
```
layer1 = layer1.0 + layer1.1
layer1.0 = conv1 + bn1 + conv2 + bn2 + relu
```

This happens automatically - parent statistics are computed from children.

## Programmatic Usage

```python
from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.table_formatter import format_partition_table
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

# Trace model
fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)

# Partition
partitioner = GraphPartitioner()
report = partitioner.partition(fx_graph)

# Format as table
table = format_partition_table(fx_graph, report, show_macs=True)
print(table)
```

## Formatting Options

### Show MACs (default)
```python
table = format_partition_table(fx_graph, report, show_macs=True)
```

### Show FLOPs
```python
table = format_partition_table(fx_graph, report, show_macs=False)
```

## Column Formats

| Column | Format | Examples |
|--------|--------|----------|
| #Parameters | Auto-scaled | 128, 9.408K, 11.690M |
| MACs/FLOPs | Auto-scaled | 115.606M, 1.814G |
| Memory | Auto-scaled | 1.75MB, 116.95MB |
| Shapes | Tuple notation | (64,), (64, 3, 7, 7) |

## ResNet-18 Statistics

```
Total parameters: 11.69M
Total FLOPs: 3.644 GFLOPs
Total MACs: 1.814 GMACs
Total memory: 116.95 MB
```

### Layer Breakdown
| Layer | Parameters | MACs | Memory |
|-------|------------|------|--------|
| conv1 | 9.408K | 118.014M | 3.85MB |
| layer1 | 147.968K | 462.422M | 16.65MB |
| layer2 | 525.568K | 411.042M | 12.54MB |
| layer3 | 2.100M | 411.042M | 10.96MB |
| layer4 | 8.394M | 411.042M | 10.48MB |
| fc | 513.000K | 512.000K | 2.06MB |

## Implementation Details

### Key Components

1. **`HierarchicalTableFormatter`** (`table_formatter.py`)
   - Builds module hierarchy from FX graph
   - Aggregates statistics for parent modules
   - Formats as markdown table

2. **`ModuleStats`** dataclass
   - Stores statistics for each module
   - Includes parameters, FLOPs, MACs, memory
   - Tracks hierarchy level for indentation

3. **Aggregation Algorithm**
   - Identifies parent paths from module names
   - Creates parents in depth-first order (deepest first)
   - Sums child statistics to create parent entries

### Sorting Strategy

Modules are sorted hierarchically:
1. **model** comes first (root)
2. Top-level modules sorted alphabetically
3. Within each parent, children sorted by index then name
4. Numeric indices sort numerically (0, 1, 2, not 0, 1, 10)

### Depth-First Aggregation

Parent modules must be created deepest-first:
```
1. Create layer1.0 from layer1.0.conv1, layer1.0.bn1, etc.
2. Create layer1 from layer1.0 and layer1.1
3. Create model from all top-level modules
```

This ensures parent statistics are correctly aggregated.

## Files

- **`src/graphs/characterize/table_formatter.py`** - Main formatter implementation
- **`cli/show_table.py`** - CLI tool to display tables
- **`cli/debug_table.py`** - Debug tool for hierarchy construction

## Future Enhancements

Potential additions:
- [ ] Export to CSV/JSON
- [ ] Filter by operation type
- [ ] Color coding by memory/compute intensity
- [ ] Comparison mode (multiple models side-by-side)
- [ ] Show arithmetic intensity per module
- [ ] Highlight bottlenecks
- [ ] Custom sorting options
