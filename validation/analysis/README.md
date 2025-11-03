# Analysis Validation Tests

This directory contains validation tests for performance analysis capabilities, focusing on EDP (Energy-Delay Product) breakdown and hierarchical analysis.

## Test Organization

### Subgraph-Level EDP Tests

**Purpose:** Validate Phase 1 (subgraph-level) hierarchical EDP breakdown on core DNN building blocks.

**Test Files:**

1. **`test_building_blocks_simple.py`** ✅ **Working**
   - Tests using `UnifiedAnalyzer` directly with custom models
   - Validates subgraph-level EDP on 4 core building blocks:
     - MLP (Linear → ReLU → Linear → ReLU)
     - Conv2D (Conv → BN → ReLU)
     - ResNet Block (Conv → BN → ReLU → Conv → BN + residual → ReLU)
     - Attention Head (Q, K, V projections → attention)
   - Hardware: KPU T256
   - Status: All tests pass ✅

2. **`test_building_blocks_edp.py`** ⚠️ **Future Enhancement**
   - Tests using `ArchitectureComparator` API (official API)
   - Multiple architectures: GPU, KPU, TPU, CPU
   - Status: Requires ArchitectureComparator enhancement to support custom models
   - Future: When ArchitectureComparator supports custom models, use this test

**Relationship:** `test_building_blocks_simple.py` is the current working test. `test_building_blocks_edp.py` shows the intended API pattern for future use.

---

## Running Tests

### Run All Analysis Validation Tests

```bash
pytest validation/analysis/ -v
```

### Run Specific Test

```bash
# Current working test
python validation/analysis/test_building_blocks_simple.py

# Future test (when ArchitectureComparator supports custom models)
python validation/analysis/test_building_blocks_edp.py
```

### Pytest Integration

```bash
# Run with pytest
pytest validation/analysis/test_building_blocks_simple.py -v

# Run with coverage
pytest validation/analysis/ -v --cov=src/graphs/analysis
```

---

## Test Validation Criteria

Each building block test validates:

1. ✅ **Subgraph count matches model structure**
   - MLP: 4 subgraphs (fc1, relu1, fc2, relu2)
   - Conv2D: 3 subgraphs (conv, bn, relu)
   - ResNet Block: 7 subgraphs (conv1, bn1, relu1, conv2, bn2, add, relu2)
   - Attention: 4 subgraphs (q_proj, k_proj, v_proj, out_proj)

2. ✅ **EDP fractions sum to 1.0**
   - Normalized fractions should sum to 1.0 within tolerance

3. ✅ **Component EDPs decompose correctly**
   - Compute EDP + Memory EDP + Static EDP = Total EDP

4. ✅ **Cumulative distribution shows concentration**
   - Top few subgraphs should dominate (80/20 rule)

5. ✅ **Bottleneck analysis present**
   - Each subgraph shows bottleneck type (compute_bound, memory_bound, bandwidth_bound)

---

## Key Findings from Validation

### Static Energy Dominance (75-76%)

**Critical Finding:** Static (leakage) energy dominates EDP across all building blocks.

**Implication:** Latency reduction has outsized impact on EDP optimization.

### Pareto Principle (80/20 Rule)

**Finding:** Top 2-3 subgraphs account for 50-80% of total EDP.

**Examples:**
- MLP: Top 1 subgraph = 50%, Top 2 = 80%
- ResNet Block: Top 3 = 50%, Top 5 = 80%

### Memory-Bound Operations

**Finding:** All subgraphs show `bandwidth_bound` bottleneck for small models (batch=1).

**Why:** Low arithmetic intensity + high compute capability → memory bottleneck

### Expected Results per Building Block

#### MLP
```
Top Subgraph: fc1 (79.9% of total EDP)
- Linear layers dominate
- ReLU operations negligible (0.0%)
- Static energy: 76.1%
```

#### Conv2D
```
Top Subgraph: bn (42.2%)
- Surprising: BN > Conv due to small input size
- Conv: 15.8%
- ReLU: 42.0%
- Static energy: 76.1%
```

#### ResNet Block
```
Top Subgraph: add (21.9%)
- Residual connection surprisingly high
- Conv1/Conv2: 19.5% each
- More evenly distributed
- Static energy: 76.2%
```

#### Attention Head
```
Top Subgraph: q_proj (25.0%)
- All 4 projections equal (25% each)
- Note: Matmul/softmax not visible as separate subgraphs
- Static energy: 75.2%
```

---

## Known Limitations (Phase 1)

### 1. Architectural Overhead Not Distributed

**Issue:** Subgraph EDP sums differ from model-level EDP by ~96%

**Cause:** Architectural overhead applied at model level, not distributed to subgraphs

**Status:** Expected behavior for Phase 1, addressed in Phase 2

### 2. Attention Mechanism Visibility

**Issue:** Attention head only shows Linear projections, missing matmul/softmax

**Cause:** FX tracing represents attention as primitive ops, partitioner fuses them

**Impact:** Cannot analyze attention mechanism's core operations

**Future:** Phase 2 operator-level breakdown may reveal these

### 3. Small Model Scale

**Issue:** All operations show as memory-bound (batch=1, small inputs)

**Impact:** Cannot validate compute-bound behavior

**Future:** Test on larger models (ResNet-50, BERT-base)

---

## CI Integration

These tests are run as part of the CI pipeline in `.github/workflows/ci.yml`:

```yaml
- name: Run analysis validation tests
  run: |
    pytest validation/analysis/ -v -k "simple" || echo "⚠️  Analysis validation tests (non-blocking)"
```

**Note:** Currently non-blocking to allow iterative development. Will become blocking once ArchitectureComparator supports custom models.

---

## Future Tests (Phase 2)

### Operator-Level EDP Tests

When Phase 2 (operator-level EDP) is implemented, add:

1. **`test_operator_edp_breakdown.py`**
   - Validate operator-level decomposition within subgraphs
   - Test architectural modifiers (CPU/GPU vs TPU/KPU)
   - Quantify fusion benefits

2. **`test_architectural_modifiers.py`**
   - Validate modifier tables for all operator types
   - Test modifier application logic
   - Verify alignment of subgraph EDP sums with model-level EDP

3. **`test_fusion_benefits.py`**
   - Compare fused vs separate execution scenarios
   - Quantify EDP savings from fusion
   - Architecture-specific fusion analysis

---

## Documentation References

- **Implementation:** `src/graphs/analysis/architecture_comparator.py:802-1006`
- **Design:** `docs/designs/hierarchical_edp_breakdown.md`
- **Session Log:** `docs/sessions/2025-11-03_building_blocks_validation.md`
- **Phase 1 Summary:** `docs/sessions/2025-11-03_phase1_subgraph_edp.md`

---

## Contributing

When adding new analysis validation tests:

1. Follow the pattern in `test_building_blocks_simple.py`
2. Validate all 5 criteria listed above
3. Document expected results
4. Add to CI configuration
5. Update this README with findings

---

## Support

For questions or issues:
- Check design document: `docs/designs/hierarchical_edp_breakdown.md`
- Review session logs: `docs/sessions/2025-11-03_*.md`
- Open issue at: https://github.com/anthropics/graphs/issues
