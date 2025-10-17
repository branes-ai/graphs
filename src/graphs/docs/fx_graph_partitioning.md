# FX Graph Partitioning: What Gets Partitioned?

## Overview

The GraphPartitioner analyzes PyTorch FX graphs and creates subgraphs for performance analysis. This document explains what gets partitioned and what doesn't.

## FX Graph Node Types

PyTorch FX graphs contain several node types:

### 1. `call_module` Nodes ✅ PARTITIONED

These represent calls to `nn.Module` instances:
- `Conv2d`, `Linear` - Convolution and linear layers
- `BatchNorm2d`, `LayerNorm` - Normalization layers
- `ReLU`, `GELU`, `Hardswish` - Activation functions
- `MaxPool2d`, `AvgPool2d` - Pooling layers

**Status**: ✅ Fully supported in Phase 1

**Example (ResNet-18)**:
```
60 call_module nodes:
  - Conv2d: 20 nodes
  - BatchNorm2d: 20 nodes
  - ReLU: 17 nodes
  - MaxPool2d: 1 node
  - AdaptiveAvgPool2d: 1 node
  - Linear: 1 node

→ Creates 60 subgraphs
```

### 2. `call_function` Nodes ⚠️ NOT PARTITIONED (Phase 1)

These represent calls to PyTorch functions:
- `torch.add` - Element-wise addition (residual connections)
- `torch.cat` - Tensor concatenation
- `torch.flatten` - Flatten tensors
- `F.relu`, `F.softmax` - Functional API operations

**Status**: ⚠️ Not partitioned in Phase 1

**Impact**: These operations are NOT included in FLOPs/memory analysis

**Example (ResNet-18)**:
```
9 call_function nodes:
  - torch.add: 8 nodes (residual connections)
  - torch.flatten: 1 node

→ Creates 0 subgraphs (ignored)
```

**Why not partitioned?**
- Phase 1 focuses on major compute operations (convolutions, linear layers)
- `call_function` operations typically have minimal FLOPs (e.g., element-wise add)
- Will be added in Phase 2 for completeness

### 3. `call_method` Nodes ⚠️ NOT PARTITIONED (Phase 1)

These represent calls to tensor methods:
- `.view()` - Reshape tensor
- `.reshape()` - Reshape tensor
- `.permute()` - Permute dimensions
- `.squeeze()`, `.unsqueeze()` - Dimension manipulation

**Status**: ⚠️ Not partitioned in Phase 1

**Impact**: These operations are NOT included in FLOPs/memory analysis

**Why not partitioned?**
- These are typically zero-FLOP operations (just metadata changes)
- May involve memory copies (will be analyzed in Phase 2)

### 4. `placeholder` and `output` Nodes

- `placeholder`: Input to the model
- `output`: Output of the model

**Status**: Not partitioned (not operations)

## What This Means for Your Analysis

### ResNet-18 Example

**FX Graph**: 71 total nodes
- 60 call_module (layers)
- 9 call_function (torch.add, flatten)
- 1 placeholder (input)
- 1 output

**Subgraphs Created**: 60 subgraphs (from call_module nodes)

**Not Analyzed**: 9 call_function nodes
- 8× `torch.add` (residual connections): ~5 million FLOPs total (negligible vs 4.49 GFLOPs)
- 1× `torch.flatten`: 0 FLOPs (just reshape)

**Impact**: Missing ~0.005 GFLOPs (0.1% of total) - acceptable for Phase 1

### MobileNet-V2 Example

**FX Graph**: ~150 total nodes
- 141 call_module (layers)
- ~9 call_function (torch.add, etc.)

**Subgraphs Created**: 141 subgraphs

**Not Analyzed**: ~9 call_function nodes (residual connections)

### Functional vs Module API

Some models use the **functional API** instead of modules:

```python
# Module API (PARTITIONED ✅)
self.relu = nn.ReLU()
x = self.relu(x)  # → call_module node

# Functional API (NOT PARTITIONED ⚠️)
x = F.relu(x)  # → call_function node
```

**Recommendation**: If your model uses functional API heavily, FLOPs may be underestimated.

## How to Check What's Being Partitioned

Run the quick start script to see the breakdown:

```bash
python examples/quick_start_partitioner.py
```

Look for this section:

```
FX NODE → SUBGRAPH MAPPING
==========================
FX Graph has 58 call_module nodes
Created 60 subgraphs from call_module nodes

Nodes NOT converted to subgraphs:
  call_function nodes: 9 nodes
    - add: 8
    - flatten: 1
```

## Phase 2 Enhancements (Future)

Phase 2 will add support for:

1. **call_function nodes**: torch.add, torch.cat, etc.
2. **Fusion detection**: Conv+BN+ReLU fused patterns
3. **Memory operations**: View/reshape memory copies
4. **Functional API**: F.relu, F.softmax, etc.

## FAQs

### Q: Why are my FLOPs lower than expected?

**A**: Check if your model uses functional API (`F.relu`) instead of module API (`nn.ReLU`). Functional calls are not partitioned in Phase 1.

**Solution**: Convert to module API or wait for Phase 2 support.

### Q: Are residual connections (torch.add) included?

**A**: No, not in Phase 1. However, their FLOP contribution is typically <0.1% of total.

**Example**: ResNet-18 has 8× `torch.add` operations
- Each add: ~200K elements → 200K FLOPs
- Total: ~1.6M FLOPs
- vs Total model: 4.49 GFLOPs
- Impact: 0.036% of total (negligible)

### Q: Will Phase 2 fix this?

**A**: Yes, Phase 2 will add support for call_function and call_method nodes.

### Q: How do I know if my model is well-supported?

Run the quick start script and check:
```
Created X subgraphs from call_module nodes
Total: Y nodes not partitioned
```

If `Y` is small (< 10% of total nodes) and mostly torch.add/flatten, you're fine.

If `Y` is large or includes compute operations (F.relu, F.conv2d), Phase 1 results may be incomplete.

## Summary Table

| Node Type | Examples | Phase 1 Support | Typical FLOP Impact |
|-----------|----------|-----------------|---------------------|
| call_module | Conv2d, Linear, ReLU (module) | ✅ Full | 99%+ of FLOPs |
| call_function | torch.add, F.relu, flatten | ⚠️ None | <1% of FLOPs |
| call_method | .view(), .reshape() | ⚠️ None | 0% (metadata) |
| placeholder | Model input | N/A | N/A |
| output | Model output | N/A | N/A |

## Recommendations

1. **Check your model**: Run `quick_start_partitioner.py` to see what's being partitioned
2. **Module API**: Prefer `nn.ReLU()` over `F.relu()` for better Phase 1 support
3. **Validate results**: Run validation tests to ensure FLOP counts are in expected range
4. **Phase 2**: If you need comprehensive analysis of all operations, wait for Phase 2 or contribute!

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [Graph Partitioner Tutorial](graph_partitioner_tutorial.md)
- [Realistic Performance Modeling Plan](realistic_performance_modeling_plan.md)
