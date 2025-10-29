# Enhanced Attention Fusion - Design Plan

## Current State vs. Enhanced Fusion

### Current Approach

**What We Fuse Now**:
```
LayerNorm → MultiheadAttention (2 ops)
```

**Problem**: MultiheadAttention is a black box containing ~10-15 operations internally, so we're only getting 1 fusion boundary instead of potentially 10+.

### Enhanced Approach

**Break Down MultiheadAttention** into constituent operations:
```
LayerNorm → QKV_Projections → Split_Heads → Attention_Compute → Concat_Heads → Output_Projection → Dropout
```

This would create 7-10 fusible operations instead of 1, potentially achieving 5-7× better fusion for attention blocks.

---

## Multi-Head Attention Internals

### Standard Multi-Head Attention Structure

```python
class MultiheadAttention(nn.Module):
    def forward(self, query, key, value):
        # 1. QKV Projections (3 Linear layers)
        Q = self.q_proj(query)      # (batch, seq_len, embed_dim)
        K = self.k_proj(key)        # (batch, seq_len, embed_dim)
        V = self.v_proj(value)      # (batch, seq_len, embed_dim)

        # 2. Split into heads
        Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)

        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)
        # scores: (batch, num_heads, seq_len, seq_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout)

        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, num_heads, seq_len, head_dim)

        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, embed_dim)

        # 5. Output projection
        output = self.out_proj(attn_output)

        return output
```

### Operations Breakdown

| Step | Operation | Type | Fusible With |
|------|-----------|------|--------------|
| 0 | LayerNorm | Normalization | ✅ Q_Proj |
| 1 | Q_Projection | Linear | ✅ K_Proj (parallel) |
| 2 | K_Projection | Linear | ✅ V_Proj (parallel) |
| 3 | V_Projection | Linear | ✅ Reshape |
| 4 | Reshape Q,K,V | View/Permute | ✅ Transpose |
| 5 | Transpose Q,K,V | Permute | ✅ Matmul |
| 6 | Q @ K^T | Matmul | ✅ Scale |
| 7 | Scale (/√d) | Mul | ✅ Softmax |
| 8 | Softmax | Activation | ✅ Dropout |
| 9 | Dropout (attn) | Regularization | ✅ Matmul |
| 10 | Scores @ V | Matmul | ✅ Transpose |
| 11 | Transpose | Permute | ✅ Reshape |
| 12 | Reshape (concat) | View | ✅ Out_Proj |
| 13 | Out_Projection | Linear | ✅ Dropout |
| 14 | Dropout (output) | Regularization | ✅ Add (residual) |

**Total**: 15 operations (vs 1 currently!)

---

## Fusion Opportunities

### Fusible Chains

#### Chain 1: QKV Projections (3 parallel ops)
```
Input
  ├─→ Linear (Q_proj) ──┐
  ├─→ Linear (K_proj) ──┼─→ [fused kernel]
  └─→ Linear (V_proj) ──┘
```
**Benefit**: 3 separate kernels → 1 fused kernel
**Memory Save**: 2 intermediate tensors eliminated

#### Chain 2: Attention Computation
```
Q @ K^T → Scale → Softmax → Dropout → @ V
```
**Benefit**: 5 operations → 1-2 fused kernels
**Memory Save**: 4 intermediate tensors eliminated
**Challenge**: Matmul operations are compute-intensive, may not fuse well

#### Chain 3: Output Path
```
Transpose → Reshape → Linear (out_proj) → Dropout
```
**Benefit**: 4 operations → 1-2 fused kernels
**Memory Save**: 3 intermediate tensors eliminated

### Expected Fusion Efficiency

**Current**:
```
Attention Block: 2 ops (LayerNorm → MultiheadAttention)
```

**Enhanced**:
```
Attention Block: 5-7 fused chains:
  1. LayerNorm → QKV_Projections (4 ops)
  2. Reshape → Transpose (2 ops)
  3. Matmul → Scale → Softmax (3 ops)
  4. Dropout → Matmul (2 ops)
  5. Transpose → Reshape → Out_Proj (3 ops)
  6. Dropout → Add (2 ops)
```

**Improvement**: 2 ops → 6-7 fused subgraphs (but 16 total ops covered)

**Memory Reduction**: Expected 40-60% for attention blocks (vs current 5.7%)

---

## Implementation Approaches

### Approach 1: Custom FX Tracer (Recommended)

**Strategy**: Create custom tracer that decomposes MultiheadAttention

```python
class DecomposingTracer(torch.fx.Tracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)

        # Post-process: decompose MultiheadAttention nodes
        for node in graph.nodes:
            if node.op == 'call_module':
                module = root.get_submodule(node.target)
                if isinstance(module, nn.MultiheadAttention):
                    self._decompose_attention(graph, node, module)

        return graph

    def _decompose_attention(self, graph, attn_node, attn_module):
        """Replace single MultiheadAttention node with decomposed operations"""

        with graph.inserting_before(attn_node):
            # Insert Q, K, V projections
            q_proj = graph.call_module(f"{attn_node.target}.q_proj", ...)
            k_proj = graph.call_module(f"{attn_node.target}.k_proj", ...)
            v_proj = graph.call_module(f"{attn_node.target}.v_proj", ...)

            # Insert reshape/transpose for multi-head
            q_reshaped = graph.call_function(torch.reshape, ...)
            # ... etc

            # Insert attention computation
            scores = graph.call_function(torch.matmul, ...)
            scaled = graph.call_function(torch.mul, ...)
            attn_weights = graph.call_function(F.softmax, ...)
            # ... etc

        # Replace uses of original attn_node with final output
        attn_node.replace_all_uses_with(final_output)
        graph.erase_node(attn_node)
```

**Pros**:
- Clean separation of concerns
- Reusable across models
- Can handle different attention variants

**Cons**:
- Requires understanding PyTorch FX graph manipulation
- Need to handle edge cases (causal masks, key padding, etc.)

**Effort**: 2-3 days

---

### Approach 2: Decomposed Attention Module

**Strategy**: Use explicit decomposed attention implementation

```python
class DecomposedMultiheadAttention(nn.Module):
    """Functionally identical to nn.MultiheadAttention but with explicit ops"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Explicitly separate projections (instead of packed weights)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape

        # QKV projections (will be separate nodes in FX graph)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head (will be separate nodes)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation (separate nodes)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads (separate nodes)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output
```

**Usage**:
```python
# Replace model's attention with decomposed version
model = models.vit_b_16(weights=None)
for name, module in model.named_modules():
    if isinstance(module, nn.MultiheadAttention):
        # Replace with DecomposedMultiheadAttention
        decomposed = DecomposedMultiheadAttention(...)
        # ... copy weights ...
        setattr(parent_module, child_name, decomposed)
```

**Pros**:
- Simple and explicit
- Easy to understand and debug
- Can trace with standard torch.fx.symbolic_trace

**Cons**:
- Requires model modification
- Need separate implementation for each attention variant
- More verbose

**Effort**: 1-2 days

---

### Approach 3: Torch FX Decomposition (Future)

**Strategy**: Use PyTorch's built-in decomposition framework (torch.fx.experimental.proxy_tensor)

```python
from torch.fx.experimental.proxy_tensor import make_fx

# Define decomposition for MultiheadAttention
@torch.fx.wrap
def decompose_attention(query, key, value, attn_module):
    # Explicit decomposition logic
    ...

# Use make_fx with decomposition
decomposed_graph = make_fx(model, decomposition_table={
    nn.MultiheadAttention: decompose_attention
})(input)
```

**Pros**:
- Official PyTorch approach
- Framework handles complexity
- Future-proof

**Cons**:
- Still experimental (as of PyTorch 2.x)
- Limited documentation
- May change in future versions

**Effort**: 3-4 days (learning curve)

---

## Proposed Fusible Patterns (Enhanced)

### New Patterns to Add

```python
# Attention QKV projections
('LayerNorm', 'Linear'),  # Already have this
('Linear', 'Linear'),      # K_proj after Q_proj (if sequential)

# Attention computation
('matmul', 'mul'),         # Q@K^T → scale
('mul', 'softmax'),        # scale → softmax
('softmax', 'Dropout'),    # softmax → dropout
('Dropout', 'matmul'),     # dropout → scores@V

# Reshape/transpose chains
('reshape', 'transpose'),
('transpose', 'reshape'),
('reshape', 'Linear'),     # concat → out_proj

# Full attention output
('Linear', 'Dropout'),     # out_proj → dropout (already have)
('Dropout', 'add'),        # dropout → residual (already have)
```

### Pattern Recognition Challenges

**Issue 1: Parallel Operations**

Q, K, V projections happen in parallel but fusion expects sequential:
```
       Input
      /  |  \
    Q   K   V  ← Parallel, not sequential!
```

**Solution**: Add support for "parallel fusion groups":
```python
class ParallelFusionGroup:
    """Group of operations that execute in parallel"""
    operations: List[Node]
    can_fuse_together: bool  # If True, can fuse into single kernel
```

**Issue 2: Branching for Multi-Head**

After projections, tensors split into heads:
```
Q (batch, seq, embed)
  ↓
Q (batch, seq, heads, head_dim)  ← Reshape
  ↓
Q (batch, heads, seq, head_dim)  ← Transpose
```

This creates multiple consumers, which current fusion stops at.

**Solution**: Allow fusion through reshape/transpose if they're just views (no memory copy).

---

## Expected Results

### Current ViT-B/16 Attention Block

```
Operations: 2
  - LayerNorm
  - MultiheadAttention

Subgraphs: 1
Memory Reduction: 5.7%
```

### Enhanced ViT-B/16 Attention Block

```
Operations: 16 (decomposed)
  - LayerNorm
  - Linear (Q_proj)
  - Linear (K_proj)
  - Linear (V_proj)
  - Reshape (Q)
  - Transpose (Q)
  - Reshape (K)
  - Transpose (K)
  - Reshape (V)
  - Transpose (V)
  - Matmul (Q@K^T)
  - Mul (scale)
  - Softmax
  - Dropout
  - Matmul (scores@V)
  - Transpose (concat prep)
  - Reshape (concat)
  - Linear (out_proj)
  - Dropout (output)

Fused Subgraphs: 6-8
  1. LayerNorm → Q_proj → K_proj → V_proj (4 ops)
  2. Reshape → Transpose (×3, may fuse differently) (6 ops)
  3. Matmul → Scale → Softmax → Dropout (4 ops)
  4. Matmul (1 op, large operation)
  5. Transpose → Reshape → Out_proj (3 ops)
  6. Dropout → Add (2 ops, with residual)

Memory Reduction: 40-60% (vs 5.7% currently)
```

### Per-Model Impact

| Model | Current Attn Fusion | Enhanced Attn Fusion | Improvement |
|-------|-------------------|-------------------|-------------|
| **ViT-B/16** | 2 ops, 5.7% mem save | 8 fused groups, 45% mem save | 8× better |
| **Swin-T** | N/A (custom attn) | Would need decomposition | New capability |
| **Overall** | 1.38× efficiency | Est. 2.0-2.5× efficiency | 1.8× better |

---

## Implementation Plan

### Phase 1: Proof of Concept (1 week)

**Goal**: Demonstrate decomposed attention fusion on toy example

**Tasks**:
1. Create `DecomposedMultiheadAttention` module
2. Build simple ViT block with decomposed attention
3. Trace and partition with existing fusion strategy
4. Measure memory reduction improvement

**Success Criteria**: Achieve >30% memory reduction in attention block (vs current 5.7%)

### Phase 2: Custom Tracer (2 weeks)

**Goal**: Automatic decomposition of standard MultiheadAttention

**Tasks**:
1. Implement `DecomposingTracer` class
2. Add decomposition logic for `nn.MultiheadAttention`
3. Handle edge cases (masks, padding, batched inputs)
4. Add new fusible patterns for attention components
5. Test on ViT and Swin models

**Success Criteria**:
- Automatic decomposition works for all ViT variants
- Fusion efficiency improves from 1.38× to 2.0×+
- No regression in accuracy

### Phase 3: Parallel Fusion Support (1-2 weeks)

**Goal**: Fuse parallel operations (Q, K, V projections)

**Tasks**:
1. Extend fusion algorithm to recognize parallel patterns
2. Implement `ParallelFusionGroup` concept
3. Add support for fusing parallel Linear operations
4. Update fusion metrics to account for parallel groups

**Success Criteria**:
- Q, K, V projections fuse together
- Memory reduction improves by additional 10-15%

### Phase 4: Validation and Optimization (1 week)

**Goal**: Validate correctness and optimize performance

**Tasks**:
1. Test on all supported transformer models
2. Benchmark memory usage and execution time
3. Compare with unfused baseline
4. Document fusion patterns and results

**Success Criteria**:
- All tests pass
- Memory reduction 40-60% for attention blocks
- Overall transformer fusion efficiency 2.0-2.5×

**Total Effort**: 5-6 weeks

---

## Challenges and Risks

### Challenge 1: FX Graph Complexity

**Issue**: Decomposed attention creates 15+ nodes per block

**Impact**: Graph becomes harder to visualize and debug

**Mitigation**:
- Add hierarchical visualization (collapse/expand subgraphs)
- Provide summary statistics
- Keep original (non-decomposed) tracing as option

### Challenge 2: Attention Variants

**Issue**: Different models use different attention mechanisms:
- Standard multi-head (ViT)
- Shifted window (Swin)
- Cross attention (DETR)
- Sparse attention (BigBird)

**Impact**: Need separate decomposition for each variant

**Mitigation**:
- Start with standard multi-head (covers most models)
- Add variants incrementally
- Provide extension API for custom attention

### Challenge 3: Performance Regression

**Issue**: Decomposition might expose inefficiencies

**Impact**: Could be slower than optimized nn.MultiheadAttention

**Mitigation**:
- Benchmark carefully
- Keep fusion optional (flag to enable/disable)
- Profile to identify bottlenecks

### Challenge 4: Numerical Stability

**Issue**: Different operation order might affect precision

**Impact**: Could cause slight accuracy differences

**Mitigation**:
- Validate outputs match original (within epsilon)
- Use torch.allclose() for testing
- Document any precision trade-offs

---

## Alternative: Kernel Fusion (Longer Term)

**Instead of graph-level fusion**, could implement actual kernel fusion:

```python
@torch.jit.script
def fused_attention_forward(Q, K, V, num_heads, dropout_p):
    """Fused CUDA kernel for multi-head attention"""
    # All operations in single kernel
    # - Reshape Q,K,V
    # - Compute Q@K^T
    # - Scale, softmax, dropout
    # - Compute scores@V
    # - Reshape output
    ...
```

**Benefits**:
- True kernel fusion (not just graph fusion)
- Maximum performance
- No intermediate tensors

**Drawbacks**:
- Requires CUDA expertise
- Complex to implement
- Maintenance burden
- Already exists (Flash Attention, xFormers)

**Recommendation**: Focus on graph-level fusion first, leverage existing kernel fusion libraries where available.

---

## Conclusion

Enhanced attention fusion would provide substantial benefits:

**Quantitative**:
- 8× better memory reduction for attention (5.7% → 45%)
- 1.8× better overall transformer fusion (1.38× → 2.5×)
- 40-60% fewer kernel launches for attention blocks

**Qualitative**:
- Better understanding of attention bottlenecks
- More optimization opportunities
- Applicable to all transformer variants

**Recommendation**: Implement Phase 1 (Proof of Concept) first to validate the approach, then proceed to Phase 2 (Custom Tracer) if results are promising.

The investment (5-6 weeks) is justified by the significant improvement in transformer fusion efficiency.
