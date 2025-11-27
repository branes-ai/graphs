# Fusion-Based Partitioning Algorithm: Concrete Proposal

## Simplest Effective Algorithm

Based on the constraint that we want **coarse-grained subgraphs** suitable for hardware mapping, here's a simple but effective algorithm:

## Algorithm: Greedy Sequential Fusion with Residual Awareness

### Core Idea

**Fuse operators sequentially until hitting a "boundary"**

**Boundaries** (stop fusion):
1. **Fork point**: Output tensor used by multiple consumers (e.g., residual branch)
2. **Join point**: Operation has multiple inputs (e.g., Add for residual, Mul for SE blocks)
3. **Shape change**: Significant reshape/dimension change
4. **Resource limit**: Fused subgraph exceeds memory/compute budget

### Pseudocode

```python
def partition_graph(fx_graph):
    """
    Greedy fusion algorithm

    Start from inputs, fuse operators sequentially until hitting a boundary
    """

    visited = set()
    subgraphs = []

    # Topological sort to ensure correct dependencies
    nodes = topological_sort(fx_graph)

    for start_node in nodes:
        if start_node in visited:
            continue

        # Start a new subgraph
        subgraph = [start_node]
        visited.add(start_node)
        current = start_node

        # Keep fusing while possible
        while True:
            next_node = get_fusible_successor(current, fx_graph, visited)

            if next_node is None:
                # Hit a boundary, stop fusion
                break

            subgraph.append(next_node)
            visited.add(next_node)
            current = next_node

        subgraphs.append(Subgraph(subgraph))

    return subgraphs


def get_fusible_successor(node, graph, visited):
    """
    Check if node has a fusible successor

    Returns next node to fuse, or None if fusion should stop
    """

    # Get all consumers of this node's output
    consumers = get_consumers(node, graph)

    # Boundary 1: Multiple consumers (fork point)
    if len(consumers) > 1:
        return None  # Stop fusion (e.g., residual branch)

    # Boundary 2: No consumers (end of graph)
    if len(consumers) == 0:
        return None

    next_node = consumers[0]

    # Boundary 3: Already visited (cyclic dependency)
    if next_node in visited:
        return None

    # Boundary 4: Multiple inputs (join point)
    if len(get_inputs(next_node, graph)) > 1:
        return None  # Stop before join (e.g., Add)

    # Boundary 5: Not fusible operation
    if not is_fusible(node, next_node):
        return None

    # Boundary 6: Resource constraints
    if would_exceed_resources(node, next_node):
        return None

    return next_node


def is_fusible(node1, node2):
    """
    Check if two nodes can be fused

    Fusion patterns:
    - Conv2d → BatchNorm2d ✓
    - BatchNorm2d → ReLU ✓
    - Conv2d → ReLU ✓
    - Linear → GELU ✓
    - Any element-wise chain ✓
    """

    fusible_patterns = [
        ('Conv2d', 'BatchNorm2d'),
        ('BatchNorm2d', 'ReLU'),
        ('Conv2d', 'ReLU'),
        ('Linear', 'BatchNorm1d'),
        ('Linear', 'GELU'),
        ('Linear', 'ReLU'),
        # Element-wise can fuse with anything
        ('*', 'ReLU'),
        ('*', 'GELU'),
        ('Add', 'ReLU'),
        ('Mul', 'ReLU')
    ]

    for pattern in fusible_patterns:
        if matches_pattern(node1, node2, pattern):
            return True

    return False
```

## Example: ResNet-18 Residual Block

### Input Graph:
```
┌─────────────┐
│   Input     │
└──────┬──────┘
       │ (fork)
       ├──────────────────────────┐
       │                          │ (residual)
   ┌───▼───┐                      │
   │Conv 1 │                      │
   └───┬───┘                      │
   ┌───▼───┐                      │
   │  BN 1 │                      │
   └───┬───┘                      │
   ┌───▼───┐                      │
   │ ReLU 1│                      │
   └───┬───┘                      │
   ┌───▼───┐                      │
   │Conv 2 │                      │
   └───┬───┘                      │
   ┌───▼───┐                      │
   │  BN 2 │                      │
   └───┬───┘                      │
       │                          │
   ┌───▼───┐ (join)               │
   │  Add  │◄─────────────────────┘
   └───┬───┘
   ┌───▼───┐
   │ ReLU  │
   └───────┘
```

### Fusion Process:

**Iteration 1** (start at Input):
- Input has 2 consumers → STOP (fork boundary)
- Subgraph 1: [Input]

**Iteration 2** (start at Conv1):
- Conv1 → BN1: Fusible ✓
- BN1 → ReLU1: Fusible ✓
- ReLU1 → Conv2: Fusible ✓
- Conv2 → BN2: Fusible ✓
- BN2 → Add: Add has 2 inputs → STOP (join boundary)
- Subgraph 2: [Conv1, BN1, ReLU1, Conv2, BN2]

**Iteration 3** (residual branch):
- This is just a passthrough, no ops
- (or could be downsample conv)

**Iteration 4** (start at Add):
- Add → ReLU: Fusible ✓
- ReLU → end
- Subgraph 3: [Add, ReLU]

### Result:
```
Original: 9 operator nodes
Fused: 3 subgraphs
  - Subgraph 1: Input (passthrough)
  - Subgraph 2: Conv1+BN1+ReLU1+Conv2+BN2 (fused)
  - Subgraph 3: Add+ReLU (fused)

Reduction: 9 → 3 operators (3× reduction)
```

## Expected Results by Model

### ResNet-18

**Before**:
- 60 operator nodes
- 59 data transfers

**After**:
- ~20 subgraphs
  - 1× Input
  - 1× Initial Conv+BN+ReLU
  - 1× MaxPool
  - 16× Residual path (Conv+BN+ReLU+Conv+BN per block, 8 blocks × 2)
  - 8× Residual join (Add+ReLU, 8 blocks)
  - 3× Downsample (Conv+BN)
  - 1× Pool+Flatten+Linear
- ~19 data transfers

**Reduction**: 60 → 20 nodes (3× reduction)

### MobileNet-V2

**Before**:
- 141 operator nodes

**After**:
- ~35 subgraphs
  - 1× Initial Conv+BN+ReLU
  - 17× Depthwise Conv+BN+ReLU (fused)
  - 17× Pointwise Conv+BN (fused)
  - ~17× Residual Add (separate, can't fuse due to branching)
  - 1× Final Conv+Pool+Linear

**Reduction**: 141 → 35 nodes (4× reduction)

### EfficientNet-B0

**Before**:
- 214 operator nodes

**After**:
- ~50-60 subgraphs (more complex due to SE blocks)

**Reduction**: 214 → 55 nodes (3.9× reduction)

## Hardware Mapping

After fusion, subgraphs map naturally to hardware:

### GPU (CUDA):
```python
# Each fused subgraph → 1 CUDA kernel
Subgraph: Conv+BN+ReLU
→ Launch 1 fused kernel
→ Intermediate BN output stays in registers
→ Only read Conv input, write ReLU output to global memory

vs Current (unfused):
→ Launch 3 kernels (Conv, BN, ReLU)
→ Conv writes to global memory
→ BN reads from global, writes to global
→ ReLU reads from global, writes to global
→ 2× extra global memory round-trips
```

### TPU:
```python
Subgraph: Conv+BN+ReLU
→ Matrix multiplication on systolic array
→ Pipeline BN and ReLU as post-processing
→ Intermediate stays in SRAM/on-chip

vs Current:
→ 3 separate operations
→ Each requires SRAM ↔ DRAM transfer
```

### KPU Tile:
```python
Subgraph: Entire residual block
→ Load input to scratchpad (256KB)
→ Execute all ops (Conv+BN+ReLU+Conv+BN+Add+ReLU)
→ Write output to DRAM
→ Intermediate data never leaves scratchpad

vs Current:
→ Each op loads/stores to DRAM
→ 60× memory bandwidth vs 1×
```

## Implementation Complexity

### Minimal Version (1-2 days):
```python
1. Implement fusible pattern matching
2. Implement greedy fusion algorithm
3. Test on ResNet-18

Lines of code: ~300-400
```

### Full Version (1 week):
```python
1. All fusion patterns (Conv+BN+ReLU, Depthwise+Pointwise, etc.)
2. Resource-aware fusion (memory budget)
3. Hardware-specific partitioning hints
4. Visualization of fused graph

Lines of code: ~1000-1500
```

## Validation

How to know it's working:

```python
# Before fusion
assert len(subgraphs) == 60  # One per operator

# After fusion
assert len(subgraphs) < 30  # At least 2× reduction
assert all(sg.num_ops >= 1 for sg in subgraphs)  # Each has ops

# Check fusion patterns
conv_bn_relu_fused = [sg for sg in subgraphs
                      if sg.fusion_type == 'Conv_BN_ReLU']
assert len(conv_bn_relu_fused) > 10  # ResNet has many of these

# Check data movement reduction
unfused_transfers = 59  # Between 60 ops
fused_transfers = len(subgraphs) - 1
assert fused_transfers < unfused_transfers * 0.5  # At least 50% reduction
```

## Questions

1. **Should Add+ReLU be a separate subgraph or merged?**
   - Separate: Easier to see residual structure
   - Merged with preceding: Larger subgraphs, less overhead
   - **Recommendation**: Keep separate for clarity

2. **How to handle downsample branches?**
   - These are 1×1 Conv+BN on residual path
   - Currently separate subgraph
   - Could fuse with main path at Add
   - **Recommendation**: Keep separate initially

3. **Resource budgets?**
   - GPU: ~100K threads per subgraph (avoid register pressure)
   - TPU: Fits in SRAM (~32MB)
   - KPU: Fits in tile scratchpad (256KB)
   - **Recommendation**: Start without limits, add later

## Next Steps

1. **Implement greedy fusion algorithm** (minimal version)
2. **Test on ResNet-18** (should get ~20 subgraphs from 60 ops)
3. **Add visualization** (show fused subgraphs)
4. **Extend to MobileNet** (test depthwise fusion)
5. **Add resource budgets** (memory-aware partitioning)

Shall I implement this?
