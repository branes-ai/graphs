# Graph Partitioning Design: Operator Fusion and Aggregation

## Problem Statement

**Current Issue**: The "partitioner" just creates one subgraph per operator node - this is NOT partitioning.

**True Goal**: Aggregate multiple operator nodes into coarse-grained subgraphs that:
- Minimize data movement between hardware units (cores, warps, SMs, tiles)
- Maximize data reuse within a subgraph
- Can be efficiently scheduled on hardware
- Reduce kernel launch overhead

## Example: ResNet-18 Layer

### Current (Incorrect) Approach:
```
Conv2d(64, 64, 3x3) → Subgraph 1
BatchNorm2d(64)     → Subgraph 2
ReLU()              → Subgraph 3
```
**Problem**: 3 separate kernel launches, intermediate activations written to/read from global memory 2x

### Correct Partitioning:
```
Subgraph: Conv2d + BatchNorm2d + ReLU (fused)
```
**Benefit**: 1 kernel launch, intermediate activations stay in registers/shared memory

### ResNet Residual Block:
```
Input
  ↓
Conv3x3 + BN + ReLU  ← Subgraph 1 (fused)
  ↓
Conv3x3 + BN         ← Subgraph 2 (fused)
  ↓
Add (+ residual)     ← Merge with Subgraph 2 or separate?
  ↓
ReLU                 ← Fuse with Add?
  ↓
Output
```

**Question**: Should we create:
- Option A: 2 large subgraphs (both conv paths) + 1 add+relu
- Option B: 1 massive subgraph for entire block
- Option C: 3 medium subgraphs with clear data dependencies

## Partitioning Algorithm Design

### Phase 1: Pattern-Based Fusion

Detect and fuse common patterns:

#### 1. **Vertical Fusion** (Sequential operators)
Fuse operators that execute sequentially with compatible shapes:

```python
Patterns:
- Conv2d → BatchNorm2d → ReLU  (very common)
- Conv2d → ReLU
- Linear → BatchNorm1d → ReLU
- Linear → GELU (Transformers)
- DepthwiseConv2d → PointwiseConv2d (MobileNet)
```

**Constraints**:
- Output of Op1 feeds only into Op2 (no branching)
- Shapes compatible (no reshape/view in between)
- Intermediate tensor not needed elsewhere

**Benefits**:
- Eliminates intermediate memory writes
- Reduces kernel launch overhead
- Data stays in L1/registers

#### 2. **Horizontal Fusion** (Parallel operators)
Fuse operators that can execute in parallel:

```python
# Inception-style multi-branch
Input → [Conv1x1, Conv3x3, Conv5x5] → Concat

# Could fuse into single subgraph if:
# - All branches independent
# - Results concatenated
# - Hardware has enough resources
```

**Constraints**:
- Operators independent (no dependencies)
- Same input tensor
- Outputs used together (e.g., concatenated)

**Benefits**:
- Amortize input loading
- Better hardware utilization

#### 3. **Element-wise Fusion**
Fuse element-wise operations:

```python
# Residual connection
Conv3x3 + BN → Add(x, residual) → ReLU

# Can fuse Add + ReLU
# Can potentially fuse BN + Add + ReLU
```

**Constraints**:
- Element-wise operations (add, mul, relu, etc.)
- Same shape tensors

**Benefits**:
- Minimal memory bandwidth
- Very cheap to fuse

### Phase 2: Dependency-Based Partitioning

After fusion, partition the remaining graph:

#### Strategy 1: **Minimize Edge Cuts** (Data Movement)

Use graph partitioning algorithms (e.g., METIS, KaHyPar):

```python
def partition_graph(fused_graph, num_partitions):
    """
    Minimize edge cuts = minimize data transfer between partitions

    Edge weight = size of tensor transferred
    Node weight = compute cost (FLOPs)
    """

    # Build weighted graph
    for edge in fused_graph.edges:
        edge.weight = tensor_size(edge.tensor)

    for node in fused_graph.nodes:
        node.weight = compute_flops(node)

    # Partition to minimize cuts while balancing compute
    partitions = metis_partition(
        fused_graph,
        num_partitions,
        balance_constraint=0.1  # 10% imbalance allowed
    )

    return partitions
```

#### Strategy 2: **Layer-Based Partitioning** (Architectural)

Partition based on architectural structure:

```python
# ResNet: Each residual block = 1 subgraph
# MobileNet: Each inverted residual block = 1 subgraph
# Transformer: Each transformer layer = 1 subgraph

def partition_by_architecture(model):
    if is_resnet(model):
        return partition_residual_blocks(model)
    elif is_mobilenet(model):
        return partition_inverted_residual_blocks(model)
    elif is_transformer(model):
        return partition_transformer_layers(model)
    else:
        return partition_by_fusion(model)
```

#### Strategy 3: **Resource-Constrained Partitioning**

Partition based on hardware constraints:

```python
def partition_by_resources(fused_graph, hardware):
    """
    Ensure each subgraph fits in hardware resources

    Constraints:
    - Memory: Subgraph working set < cache/scratchpad size
    - Compute: Subgraph FLOPs < reasonable kernel size
    - Parallelism: Subgraph has enough parallelism for hardware
    """

    subgraphs = []
    current_subgraph = []
    current_memory = 0
    current_flops = 0

    for node in fused_graph.topological_sort():
        node_memory = estimate_memory(node)
        node_flops = estimate_flops(node)

        if (current_memory + node_memory > hardware.cache_size or
            current_flops + node_flops > hardware.max_kernel_flops):
            # Start new subgraph
            subgraphs.append(current_subgraph)
            current_subgraph = [node]
            current_memory = node_memory
            current_flops = node_flops
        else:
            # Add to current subgraph
            current_subgraph.append(node)
            current_memory += node_memory
            current_flops += node_flops

    return subgraphs
```

## Concrete Algorithm Proposal

### Algorithm: Hierarchical Fusion + Partitioning

```python
class TrueGraphPartitioner:
    """
    Phase 1: Fuse operators based on patterns
    Phase 2: Partition fused graph based on dependencies and resources
    """

    def partition(self, fx_graph, hardware_profile):
        # Step 1: Pattern-based fusion
        fused_graph = self.apply_fusion_patterns(fx_graph)

        # Step 2: Resource-aware partitioning
        subgraphs = self.partition_by_resources(fused_graph, hardware_profile)

        # Step 3: Validate and optimize
        subgraphs = self.optimize_partitions(subgraphs)

        return subgraphs

    def apply_fusion_patterns(self, fx_graph):
        """Apply fusion patterns in priority order"""

        graph = fx_graph.copy()

        # Priority 1: Conv + BN + ReLU (most common, highest benefit)
        graph = self.fuse_conv_bn_relu(graph)

        # Priority 2: Conv + BN
        graph = self.fuse_conv_bn(graph)

        # Priority 3: Element-wise chains (Add + ReLU, etc.)
        graph = self.fuse_elementwise_chains(graph)

        # Priority 4: Linear + activation
        graph = self.fuse_linear_activation(graph)

        # Priority 5: Depthwise + Pointwise (MobileNet)
        graph = self.fuse_depthwise_pointwise(graph)

        return graph

    def fuse_conv_bn_relu(self, graph):
        """
        Pattern: Conv2d → BatchNorm2d → ReLU

        Constraints:
        - Conv output feeds only into BN
        - BN output feeds only into ReLU
        - No intermediate tensors needed elsewhere
        """

        for conv_node in graph.find_nodes('Conv2d'):
            # Check if followed by BN
            bn_node = self.get_single_user(conv_node)
            if not bn_node or bn_node.type != 'BatchNorm2d':
                continue

            # Check if BN followed by ReLU
            relu_node = self.get_single_user(bn_node)
            if not relu_node or relu_node.type != 'ReLU':
                continue

            # Fuse into single node
            fused_node = FusedNode(
                name=f"{conv_node.name}_bn_relu",
                ops=[conv_node, bn_node, relu_node],
                fusion_type="Conv_BN_ReLU"
            )

            graph.replace_nodes([conv_node, bn_node, relu_node], fused_node)

        return graph

    def partition_by_resources(self, fused_graph, hardware):
        """
        Create subgraphs that fit hardware constraints

        Example for GPU:
        - Each subgraph should map to 1-4 SMs
        - Memory footprint < L2 cache (4-40 MB)
        - Enough parallelism to saturate SMs

        Example for KPU tile:
        - Memory footprint < tile scratchpad (256 KB)
        - Minimize inter-tile communication
        """

        if hardware.type == 'GPU':
            return self.partition_for_gpu(fused_graph, hardware)
        elif hardware.type == 'TPU':
            return self.partition_for_tpu(fused_graph, hardware)
        elif hardware.type == 'KPU':
            return self.partition_for_kpu(fused_graph, hardware)
        else:
            return self.partition_generic(fused_graph, hardware)
```

## Expected Results for ResNet-18

### Current (Wrong) Approach:
```
60 subgraphs (one per operator)
- 20× Conv2d
- 20× BatchNorm2d
- 17× ReLU
- 3× other
```

### After Fusion:
```
~21 subgraphs (fused operators)
- 17× Conv+BN+ReLU (fused)
- 3× Conv+BN (downsample branches)
- 1× other operations
```

### After Partitioning (by residual blocks):
```
~9 subgraphs (coarse-grained)
- 1× Initial conv + pool
- 8× Residual blocks (each contains 2 fused Conv+BN+ReLU + Add)
- 1× Final pool + FC
```

**Benefits**:
- Reduced from 60 → 9 subgraphs
- Each subgraph is a meaningful unit for scheduling
- Minimized data movement (intermediate activations stay local)
- Easier hardware mapping (1 residual block → 1 tile/SM group)

## Hardware Mapping Examples

### GPU Mapping:
```
Subgraph = Residual Block
→ Maps to 2-4 SMs
→ Intermediate activations stay in L1/shared memory
→ Only input/output tensors touch global memory
```

### TPU Mapping:
```
Subgraph = Fused Conv+BN+ReLU
→ Maps to systolic array
→ Matrix multiplication (Conv) pipelined with BN+ReLU
→ Minimizes SRAM ↔ DRAM transfers
```

### KPU Tile Mapping:
```
Subgraph = Layer or block that fits in tile scratchpad
→ Maps to single tile
→ All intermediate data stays in 256KB scratchpad
→ Only fetch input, write output
```

## Implementation Priority

### Phase 1: Basic Fusion (Week 1)
1. Implement Conv+BN+ReLU fusion
2. Implement Conv+BN fusion
3. Implement Add+ReLU fusion (residual connections)
4. Test on ResNet-18, MobileNet-V2

### Phase 2: Advanced Patterns (Week 2)
1. Depthwise+Pointwise fusion (MobileNet)
2. Multi-head attention fusion (Transformers)
3. Horizontal fusion (Inception blocks)

### Phase 3: Resource-Aware Partitioning (Week 3)
1. GPU partitioning (SM groups)
2. TPU partitioning (systolic array chunks)
3. KPU partitioning (tile memory constraints)

### Phase 4: Optimization (Week 4)
1. Graph cut minimization algorithms
2. Balance compute across partitions
3. Recomputation vs memory trade-offs

## Key Metrics

After true partitioning, we should see:

```
Before (current):
  Subgraphs: 60
  Avg subgraph size: 1 operator
  Data transfers: 59 (between every operator)

After (fused + partitioned):
  Subgraphs: 9-21 (depending on strategy)
  Avg subgraph size: 3-7 operators
  Data transfers: 8-20 (only between subgraphs)

Reduction:
  Kernel launches: 60 → 9 (6.7× fewer)
  Global memory traffic: ~50% reduction
  Scheduling overhead: ~80% reduction
```

## Questions for Design

1. **Granularity**: How coarse should subgraphs be?
   - Fine (fused ops): Better flexibility, more scheduling options
   - Coarse (blocks): Less overhead, easier hardware mapping

2. **Residual Connections**: How to handle?
   - Option A: Separate subgraph for Add+ReLU
   - Option B: Merge with preceding Conv+BN
   - Option C: Merge with following operations

3. **Hardware-Specific**: Should partitioning be hardware-aware?
   - Yes: Optimal for specific hardware
   - No: More portable across platforms

4. **Dynamic Shapes**: How to handle models with dynamic control flow?
   - Static partitioning (based on sample trace)
   - Dynamic partitioning (runtime)

## Next Steps

1. Implement fusion pattern detector
2. Build fused graph representation
3. Implement basic partitioning (residual blocks for ResNet)
4. Test and validate on ResNet-18
5. Extend to MobileNet, EfficientNet
6. Add hardware-aware partitioning

Would you like me to implement this?
