# Fusion Graph Visualization Enhancement Plan

## Overview

Enhance `cli/partitioner.py` to visualize fusion-based graph partitioning, showing how the partitioner groups multiple operators into coarse-grained fused subgraphs.

## Current State

### What Exists
- ✅ `GraphPartitioner.visualize_partitioning()` - Shows side-by-side view of:
  - **Left**: FX graph nodes in execution order
  - **Right**: Partitioned subgraphs (1-to-1 mapping with nodes)
- ✅ `cli/partitioner.py` - CLI tool that:
  - Applies partitioning strategies (unfused, fusion)
  - Compares strategies quantitatively
  - Calls `visualize_partitioning()` for "unfused" strategy only

### What's Missing
- ❌ `FusionBasedPartitioner.visualize_partitioning()` - No visualization method
- ❌ Fusion-specific visualization showing:
  - Multiple FX nodes grouped into single fused subgraphs
  - Visual indication of fusion boundaries
  - Fusion patterns (e.g., "Conv_BN_ReLU")
  - Balance analysis across the graph

## Design Goals

### 1. Consistent with Existing Visualization
- Maintain side-by-side layout: FX Graph (left) | Fused Subgraphs (right)
- Use similar formatting and alignment
- Keep column widths and structure compatible

### 2. Clearly Show Fusion Grouping
- **Visual Grouping**: Use box-drawing characters to show which FX nodes are fused together
  ```
  FX Graph                          Fused Subgraphs
  ----------------                  ---------------------------
  1. [call_module] conv1      ┌──   SUBGRAPH #1 (Conv_BN_ReLU)
     Conv2d(3->64)            │     • conv1 (Conv2d)
                              │     • bn1 (BatchNorm2d)
  2. [call_module] bn1        │     • relu (ReLU)
     BatchNorm2d(64)          │
                              │     Compute: 1.2 GMACs, 2.4 GFLOPs
  3. [call_function] relu     │     Memory: 24 MB → 8 MB (fused)
     ReLU                     └──   AI: 100 FLOPs/byte [COMPUTE]
  ```

### 3. Show Fusion Benefits
- **Data Movement Reduction**: Show memory savings from fusion
  - Before: X MB (unfused, intermediate tensors written to memory)
  - After: Y MB (fused, intermediates stay in cache)
  - Reduction: (X-Y)/X%
- **Arithmetic Intensity**: Compute-to-memory ratio improves with fusion
- **Kernel Count**: Fewer kernel launches

### 4. Enable Balance Analysis
- Show distribution of subgraph sizes (# operators per fused group)
- Identify large/small fused regions
- Highlight potential imbalances:
  - Very small fusions (missed opportunities?)
  - Very large fusions (too greedy?)

## Implementation Plan

### Phase 1: Store Fusion Results in FusionBasedPartitioner

**File**: `src/graphs/characterize/fusion_partitioner.py`

**Changes**:
```python
class FusionBasedPartitioner:
    def __init__(self):
        self.next_subgraph_id = 0
        self.fused_subgraphs: List[FusedSubgraph] = []  # ADD THIS
        self.fx_graph_cached = None  # Store fx_graph for visualization

    def partition(self, fx_graph: GraphModule) -> FusionReport:
        # ... existing code ...

        # After creating fused_subgraphs:
        self.fused_subgraphs = fused_subgraphs  # STORE FOR VISUALIZATION
        self.fx_graph_cached = fx_graph

        report = self._generate_report(fused_subgraphs, len(nodes))
        return report
```

**Rationale**: Like `GraphPartitioner.subgraphs`, store results for visualization access

---

### Phase 2: Implement FusionBasedPartitioner.visualize_partitioning()

**File**: `src/graphs/characterize/fusion_partitioner.py`

**Method Signature**:
```python
def visualize_partitioning(self, fx_graph: GraphModule, max_nodes: Optional[int] = None) -> str:
    """
    Create side-by-side visualization of FX graph and fused subgraphs.

    Shows:
    - Left: Original FX nodes in execution order
    - Right: Fused subgraphs with visual grouping

    Args:
        fx_graph: The FX graph that was partitioned
        max_nodes: Maximum number of nodes to show (None for all)

    Returns:
        String containing formatted visualization
    """
```

**Algorithm**:
1. **Build node-to-subgraph mapping**:
   ```python
   # Map each FX node_id to the FusedSubgraph it belongs to
   node_to_fused_subgraph = {}
   for fused_sg in self.fused_subgraphs:
       for node_id in fused_sg.node_ids:
           node_to_fused_subgraph[node_id] = fused_sg
   ```

2. **Process nodes in execution order**:
   ```python
   all_nodes = list(fx_graph.graph.nodes)
   if max_nodes:
       all_nodes = all_nodes[:max_nodes]
   ```

3. **Track fusion boundaries**:
   ```python
   current_subgraph_id = None
   subgraph_start = False
   subgraph_end = False
   nodes_in_current = 0
   total_nodes_in_subgraph = 0
   ```

4. **For each node, determine position in fused subgraph**:
   ```python
   for idx, node in enumerate(all_nodes):
       node_id = str(id(node))
       fused_sg = node_to_fused_subgraph.get(node_id)

       if fused_sg:
           # Is this the first node in the subgraph?
           if fused_sg.subgraph_id != current_subgraph_id:
               subgraph_start = True
               current_subgraph_id = fused_sg.subgraph_id
               nodes_in_current = 0
               total_nodes_in_subgraph = len(fused_sg.node_ids)

           nodes_in_current += 1

           # Is this the last node in the subgraph?
           if nodes_in_current == total_nodes_in_subgraph:
               subgraph_end = True
           else:
               subgraph_end = False
       else:
           # Node not in any fused subgraph
           current_subgraph_id = None
   ```

5. **Format visualization with grouping indicators**:
   ```python
   # Left side: FX node (same as unfused)
   left_lines = self._format_fx_node(node, fx_graph, idx)

   # Right side: Show fusion grouping
   if subgraph_start:
       # Top of fused subgraph box
       right_lines.append(f"┌─ SUBGRAPH #{fused_sg.subgraph_id} ────────")
       right_lines.append(f"│  Pattern: {fused_sg.fusion_pattern}")
       right_lines.append(f"│  Operators: {fused_sg.num_operators}")
       right_lines.append(f"│")

   if fused_sg:
       # Middle of fused subgraph
       right_lines.append(f"│  • {node.name} ({op_type})")

   if subgraph_end:
       # Bottom of fused subgraph box with metrics
       right_lines.append(f"│")
       right_lines.append(f"│  Compute: {format_flops(fused_sg.total_flops)}")
       right_lines.append(f"│  Memory: {format_bytes(external)} (fused)")
       right_lines.append(f"│  Saved: {format_bytes(internal)} internal")
       right_lines.append(f"│  AI: {fused_sg.arithmetic_intensity:.1f} [{bottleneck}]")
       right_lines.append(f"└──────────────────────────────────")
   ```

**Helper Methods**:
```python
def _format_fx_node(self, node, graph: GraphModule, idx: int) -> List[str]:
    """Format FX node for left column (reuse from GraphPartitioner)"""
    # Similar to GraphPartitioner implementation
    pass

def _format_fused_subgraph_header(self, fused_sg: FusedSubgraph) -> List[str]:
    """Format header for a fused subgraph"""
    pass

def _format_fused_operator(self, node, is_first: bool, is_last: bool) -> List[str]:
    """Format a single operator within a fused subgraph"""
    pass

def _format_fused_subgraph_footer(self, fused_sg: FusedSubgraph) -> List[str]:
    """Format footer with metrics for a fused subgraph"""
    pass
```

---

### Phase 3: Update cli/partitioner.py to Enable Fusion Visualization

**File**: `cli/partitioner.py`

**Changes**:
```python
def visualize_strategy(self, strategy: str, max_nodes: int = 20):
    """Visualize partitioning for a strategy"""
    if strategy not in self.results:
        print(f"Error: No results for strategy '{strategy}'")
        return

    result = self.results[strategy]
    partitioner = result['partitioner']

    print("\n" + "=" * 80)
    print(f"VISUALIZATION: {strategy.upper()} STRATEGY")
    print("=" * 80)
    print()

    # Check if partitioner has visualization method
    if hasattr(partitioner, 'visualize_partitioning'):
        viz = partitioner.visualize_partitioning(self.fx_graph, max_nodes=max_nodes)
        print(viz)
    else:
        # REMOVE THIS ELSE BLOCK - both strategies now support visualization
        print(f"Visualization not available for '{strategy}' strategy")
```

**Testing Command**:
```bash
# Test fusion visualization
python cli/partitioner.py --model resnet18 --strategy fusion --visualize

# Compare both strategies with visualization
python cli/partitioner.py --model mobilenet_v2 --strategy all --compare --visualize
```

---

### Phase 4: Add Balance Analysis

**File**: `src/graphs/characterize/fusion_partitioner.py`

**Add method**:
```python
def analyze_balance(self) -> str:
    """
    Analyze balance of fusion partitioning

    Returns formatted report showing:
    - Distribution of fusion sizes
    - Largest/smallest fused subgraphs
    - Potential issues (very small or very large fusions)
    """
    if not self.fused_subgraphs:
        return "No fused subgraphs to analyze"

    lines = []
    lines.append("=" * 80)
    lines.append("FUSION BALANCE ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Compute distribution
    sizes = [sg.num_operators for sg in self.fused_subgraphs]

    lines.append(f"Total Fused Subgraphs: {len(self.fused_subgraphs)}")
    lines.append(f"Fusion Size Distribution:")
    lines.append(f"  Min: {min(sizes)} operators")
    lines.append(f"  Max: {max(sizes)} operators")
    lines.append(f"  Avg: {sum(sizes) / len(sizes):.1f} operators")
    lines.append(f"  Median: {sorted(sizes)[len(sizes)//2]} operators")
    lines.append("")

    # Histogram
    from collections import Counter
    histogram = Counter(sizes)
    lines.append("Fusion Size Histogram:")
    for size in sorted(histogram.keys()):
        count = histogram[size]
        bar = "█" * min(count, 50)  # Limit bar width
        lines.append(f"  {size:3d} ops: {bar} ({count})")
    lines.append("")

    # Identify extremes
    single_op_count = sum(1 for sg in self.fused_subgraphs if sg.num_operators == 1)
    large_fusions = [sg for sg in self.fused_subgraphs if sg.num_operators > 10]

    if single_op_count > 0:
        pct = single_op_count / len(self.fused_subgraphs) * 100
        lines.append(f"⚠️  Single-operator subgraphs: {single_op_count} ({pct:.1f}%)")
        lines.append("    → Potential fusion opportunities missed")
        lines.append("")

    if large_fusions:
        lines.append(f"⚠️  Large fusions (>10 ops): {len(large_fusions)}")
        for sg in large_fusions[:3]:  # Show top 3
            lines.append(f"    → Subgraph #{sg.subgraph_id}: {sg.num_operators} ops ({sg.fusion_pattern})")
        lines.append("")

    # Show top fusion patterns
    lines.append("Top Fusion Patterns:")
    patterns = defaultdict(int)
    for sg in self.fused_subgraphs:
        patterns[sg.fusion_pattern] += 1

    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(self.fused_subgraphs) * 100
        lines.append(f"  {pattern:<30} {count:4d} ({pct:5.1f}%)")

    lines.append("=" * 80)

    return "\n".join(lines)
```

**Add to cli/partitioner.py**:
```python
# In run() method, after visualization:
if args.analyze_balance and 'fusion' in self.results:
    partitioner = self.results['fusion']['partitioner']
    if hasattr(partitioner, 'analyze_balance'):
        print()
        print(partitioner.analyze_balance())
```

**Add CLI argument**:
```python
parser.add_argument('--analyze-balance', action='store_true',
                   help='Analyze fusion balance (requires --strategy fusion)')
```

---

### Phase 5: Side-by-Side Comparison Visualization

**File**: `cli/partitioner.py`

**Add method**:
```python
def visualize_comparison(self, max_nodes: int = 20):
    """
    Show side-by-side comparison of unfused vs fusion strategies

    Layout:
    FX Graph | Unfused Subgraphs | Fused Subgraphs
    """
    if 'unfused' not in self.results or 'fusion' not in self.results:
        print("Need both 'unfused' and 'fusion' results to compare")
        return

    print("\n" + "=" * 120)
    print("SIDE-BY-SIDE PARTITIONING COMPARISON")
    print("=" * 120)
    print()

    # Three columns: FX Graph | Unfused | Fused
    col_width = 40
    header = f"{'FX Graph':<{col_width}}{'Unfused Partitioning':<{col_width}}{'Fusion Partitioning':<{col_width}}"
    print(header)
    print("-" * col_width + "-" * col_width + "-" * col_width)

    # ... implementation ...
```

**CLI flag**:
```python
parser.add_argument('--compare-visual', action='store_true',
                   help='Show side-by-side visual comparison (requires --strategy all)')
```

---

## Testing Strategy

### Test Cases

1. **Simple Sequential Chain** (MobileNetV2 blocks):
   - Expected: Conv→BN→ReLU fusions
   - Validate: No single-op subgraphs for fusible ops

2. **Branching (ResNet skip connections)**:
   - Expected: Fusion stops at branches
   - Validate: Fork/join boundaries respected

3. **Complex Patterns (EfficientNet)**:
   - Expected: Depthwise→Pointwise→BN→ReLU patterns
   - Validate: Fusion respects operation type boundaries

4. **Large Models** (ResNet-50):
   - Expected: Balanced fusion distribution
   - Validate: No extreme imbalances

### Validation Checks

```bash
# Test basic fusion visualization
python cli/partitioner.py --model mobilenet_v2 --strategy fusion --visualize --max-nodes 30

# Test balance analysis
python cli/partitioner.py --model resnet18 --strategy fusion --analyze-balance

# Test comparison
python cli/partitioner.py --model efficientnet_b0 --strategy all --compare --visualize

# Test side-by-side comparison
python cli/partitioner.py --model resnet50 --strategy all --compare-visual --max-nodes 15
```

---

## Example Output

### Fusion Visualization Example
```
================================================================================
GRAPH PARTITIONING VISUALIZATION (FUSION STRATEGY)
================================================================================

FX Graph (Execution Order)                       Fused Subgraphs
--------------------------------------------------    ---------------------------
1. [call_module] conv1                        ┌──   SUBGRAPH #1
   Conv2d(3->64, (7,7), stride=2)             │     Pattern: Conv_BN_ReLU
                                              │     Operators: 3
2. [call_module] bn1                          │
   BatchNorm2d(64)                            │     • conv1 (Conv2d 3->64)
                                              │     • bn1 (BatchNorm2d)
3. [call_function] relu                       │     • relu (ReLU)
   ReLU                                       │
                                              │     Compute: 118.01 MMACs, 236.03 MFLOPs
                                              │     Memory: 3.21 MB (external)
                                              │     Saved: 0.80 MB (internal, fused)
                                              │     AI: 73.5 FLOPs/byte [COMPUTE]
                                              └──   Reduction: 20.0%

4. [call_module] maxpool                      ┌──   SUBGRAPH #2
   MaxPool2d((3,3), stride=2)                 │     Pattern: MaxPool (single op)
                                              │     Operators: 1
                                              │
                                              │     • maxpool (MaxPool2d)
                                              │
                                              │     Compute: 0 MMACs, 0 MFLOPs
                                              │     Memory: 0.20 MB (external)
                                              │     AI: 0.0 FLOPs/byte [MEMORY]
                                              └──

5. [call_module] layer1.0.conv1               ┌──   SUBGRAPH #3
   Conv2d(64->64, (3,3))                      │     Pattern: Conv_BN_ReLU_Conv_BN
                                              │     Operators: 5
6. [call_module] layer1.0.bn1                 │
   BatchNorm2d(64)                            │     • layer1.0.conv1 (Conv2d)
                                              │     • layer1.0.bn1 (BatchNorm2d)
7. [call_function] layer1.0.relu              │     • layer1.0.relu (ReLU)
   ReLU                                       │     • layer1.0.conv2 (Conv2d)
                                              │     • layer1.0.bn2 (BatchNorm2d)
8. [call_module] layer1.0.conv2               │
   Conv2d(64->64, (3,3))                      │     Compute: 236.03 MMACs, 472.07 MFLOPs
                                              │     Memory: 2.10 MB (external)
9. [call_module] layer1.0.bn2                 │     Saved: 1.60 MB (internal, fused)
   BatchNorm2d(64)                            │     AI: 224.8 FLOPs/byte [COMPUTE]
                                              └──   Reduction: 43.2%

...

================================================================================
Total FX nodes: 174
Fused subgraphs: 52  (3.3× reduction from 174 ops)
Average fusion size: 3.3 operators/subgraph
Data movement reduction: 35.7%
================================================================================
```

---

## Success Criteria

- ✅ Fusion visualization matches quality of unfused visualization
- ✅ Visual grouping clearly shows which operators are fused
- ✅ Fusion patterns and metrics are prominently displayed
- ✅ Balance analysis identifies potential issues
- ✅ Side-by-side comparison makes tradeoffs obvious
- ✅ Works with all test models (ResNet, MobileNet, EfficientNet)
- ✅ Documentation updated with examples

---

## Future Enhancements

1. **Interactive Mode**: Allow user to adjust fusion boundaries
2. **Color Coding**: Use ANSI colors to highlight compute vs memory-bound subgraphs
3. **Export to Graphviz**: Generate DOT format for graph visualization tools
4. **Profiling Integration**: Overlay actual runtime measurements
5. **Hardware Targeting**: Show different fusion strategies for different targets (CPU/GPU/TPU)

---

## Timeline Estimate

- **Phase 1** (Store fusion results): 1 hour
- **Phase 2** (Implement visualization): 4-6 hours
- **Phase 3** (Update CLI): 1 hour
- **Phase 4** (Balance analysis): 2-3 hours
- **Phase 5** (Side-by-side comparison): 2-3 hours
- **Testing & Documentation**: 2-3 hours

**Total**: ~15-20 hours of focused development

---

## Dependencies

- PyTorch FX graph structure remains stable
- Box-drawing characters supported in terminal (UTF-8)
- No new external dependencies required

---

## Alternatives Considered

1. **Graphviz Visualization**: Would be prettier but requires external dependency
   - Decision: Start with text-based, add Graphviz export later

2. **HTML Output**: Could use web browser for richer UI
   - Decision: Keep CLI-first, maintain consistency with existing tools

3. **Separate Tool**: Create new visualization script instead of enhancing partitioner.py
   - Decision: Keep everything in partitioner.py for consistency

---

## Notes

- The visualization must scale to large graphs (100+ nodes)
- Use `max_nodes` parameter to control output length
- Consider adding paging/scrolling for very large outputs
- Box-drawing characters might not render in all terminals - provide ASCII fallback
