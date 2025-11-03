# Test Organization and CI Integration

**Date:** 2025-11-03
**Status:** ✅ Complete

---

## Problem

Three test files were located in the project root directory instead of properly organized in the test structure:

1. `test_building_blocks_simple.py` - Subgraph-level EDP validation (working)
2. `test_building_blocks_edp.py` - ArchitectureComparator API test (future)
3. `test_mermaid_visualization.py` - Mermaid diagram generation test

These files were not:
- Part of the organized test structure
- Integrated into CI pipeline
- Discoverable by pytest
- Following project conventions

---

## Solution

### File Moves

**Analysis Validation Tests** → `validation/analysis/`:
```bash
test_building_blocks_simple.py  → validation/analysis/test_building_blocks_simple.py
test_building_blocks_edp.py     → validation/analysis/test_building_blocks_edp.py
```

**Visualization Tests** → `tests/visualization/`:
```bash
test_mermaid_visualization.py   → tests/visualization/test_mermaid_visualization.py
```

---

## Directory Structure Created

### `validation/analysis/` (NEW)

**Purpose:** Validation tests for performance analysis capabilities (EDP breakdown, hierarchical analysis)

**Contents:**
```
validation/analysis/
├── README.md                          # Documentation and test guide
├── test_building_blocks_simple.py     # ✅ Working: UnifiedAnalyzer approach
└── test_building_blocks_edp.py        # ⚠️  Future: ArchitectureComparator approach
```

**README Created:** Comprehensive guide covering:
- Test purpose and organization
- Running instructions
- Validation criteria
- Expected results per building block
- Key findings (static energy dominance, 80/20 rule)
- Known limitations
- Future Phase 2 tests

---

### `tests/visualization/` (NEW)

**Purpose:** Unit tests for visualization functionality

**Contents:**
```
tests/visualization/
└── test_mermaid_visualization.py      # Mermaid diagram generation tests
```

---

## CI Integration

### Updated `.github/workflows/ci.yml`

Added analysis validation tests to the test job (line 90):

```yaml
- name: Run validation tests (smoke tests only)
  run: |
    # Run quick validation tests, skip slow ones
    pytest validation/hardware/ -v -k "not slow" || echo "⚠️  Some validation tests failed (non-blocking)"
    pytest validation/estimators/test_conv2d.py -v || echo "⚠️  Estimator tests failed (non-blocking)"
    pytest validation/analysis/test_building_blocks_simple.py -v || echo "⚠️  Analysis validation tests failed (non-blocking)"
```

**Status:** Non-blocking (allows iterative development, will become blocking in future)

---

## Test Organization Pattern

### Directory Structure

```
graphs/
├── tests/                              # Unit tests (blocking in CI)
│   ├── analysis/                       # Analysis module unit tests
│   ├── hardware/                       # Hardware module unit tests
│   ├── ir/                             # IR structures unit tests
│   ├── transform/                      # Transform module unit tests
│   │   └── partitioning/               # Partitioning tests
│   ├── visualization/                  # Visualization tests (NEW)
│   ├── cli/                            # CLI tool tests
│   ├── reporting/                      # Report generation tests
│   └── integration/                    # Integration tests
│
└── validation/                         # Functional validation (non-blocking)
    ├── hardware/                       # Hardware mapper validation
    ├── estimators/                     # Estimator accuracy validation
    └── analysis/                       # Analysis capability validation (NEW)
```

### Test Type Guidelines

**Unit Tests (`tests/`):**
- Fast, focused, isolated
- Test single functions/classes
- Run with coverage
- Blocking in CI

**Validation Tests (`validation/`):**
- End-to-end functionality
- Multi-component integration
- Validate against expected behavior
- Non-blocking in CI (currently)

---

## Running Tests

### Run All Tests
```bash
# Unit tests with coverage
pytest tests/ -v --cov=src/graphs

# All validation tests
pytest validation/ -v

# Specific validation suite
pytest validation/analysis/ -v
```

### Run Specific Tests
```bash
# Analysis validation (working)
python validation/analysis/test_building_blocks_simple.py

# Visualization test
python tests/visualization/test_mermaid_visualization.py

# Pytest discovery
pytest validation/analysis/test_building_blocks_simple.py -v
pytest tests/visualization/test_mermaid_visualization.py -v
```

### CI Simulation (Local)
```bash
# Replicate CI test job
pytest tests/ -v --cov=src/graphs --cov-report=term-missing
pytest validation/hardware/ -v -k "not slow"
pytest validation/estimators/test_conv2d.py -v
pytest validation/analysis/test_building_blocks_simple.py -v
```

---

## Validation Test Results

### `test_building_blocks_simple.py` ✅

**Status:** All 4 building blocks pass

**Results:**
```
Test Results:
  MLP                  ✅ PASS
  Conv2D               ✅ PASS
  ResNet               ✅ PASS
  Attention            ✅ PASS

✅ ALL BUILDING BLOCKS PASSED!
```

**Key Findings:**
- Static energy dominates (75-76%) across all building blocks
- Top 2-3 subgraphs account for 50-80% of total EDP
- All operations memory-bound (small models, batch=1)
- Operator fusion impact hidden at subgraph level

**Documentation:** `validation/analysis/README.md` contains complete analysis

---

### `test_building_blocks_edp.py` ⚠️

**Status:** Future enhancement needed

**Issue:** Requires `ArchitectureComparator` to support custom models (currently expects torchvision model names)

**Future:** When ArchitectureComparator supports custom models, this test will:
- Use official API (`ArchitectureComparator.get_subgraph_edp_breakdown()`)
- Test multiple architectures (CPU, GPU, TPU, KPU)
- Replace `test_building_blocks_simple.py` as the primary test

---

### `test_mermaid_visualization.py` ✅

**Status:** Working

**Purpose:** Validates Mermaid diagram generation for graph visualization

**Tests:**
- FX tracing and partitioning
- Mermaid diagram generation
- Multiple visualization formats
- Markdown file output

---

## Benefits of Reorganization

### 1. Discoverability
- Tests now follow standard pytest discovery patterns
- Organized by module/capability
- Easy to find related tests

### 2. CI Integration
- Tests run automatically on push/PR
- Non-blocking allows iterative development
- Clear separation: unit tests (blocking) vs validation (non-blocking)

### 3. Documentation
- README in validation/analysis/ explains purpose and findings
- Test files self-documenting with clear structure
- Session logs track implementation and results

### 4. Maintainability
- Clear ownership: tests/ mirrors src/graphs/ structure
- Easy to add new tests in appropriate locations
- Consistent patterns across test types

---

## Future Enhancements

### Phase 2 Tests (Operator-Level EDP)

When Phase 2 is implemented, add to `validation/analysis/`:

1. **`test_operator_edp_breakdown.py`**
   - Validate operator-level decomposition
   - Test architectural modifiers
   - Verify EDP alignment

2. **`test_architectural_modifiers.py`**
   - Modifier tables for all operator types
   - Modifier application logic
   - Architecture-specific behavior

3. **`test_fusion_benefits.py`**
   - Fused vs separate execution comparison
   - Quantify EDP savings
   - Architecture-specific fusion analysis

### ArchitectureComparator Enhancement

**Goal:** Support custom models in `ArchitectureComparator.analyze_all()`

**Impact:**
- Enable `test_building_blocks_edp.py` to work
- Consolidate tests to single official API
- Remove workaround in `test_building_blocks_simple.py`

**Current Limitation:**
```python
# Current: ArchitectureComparator expects torchvision model names
comparator = ArchitectureComparator(
    model_name='resnet18',  # Must be in torchvision.models
    ...
)

# Future: Support custom models
comparator = ArchitectureComparator(
    model=custom_model,     # Any nn.Module
    input_tensor=input_tensor,
    ...
)
```

---

## Files Modified

### Created
1. **`validation/analysis/README.md`** - Comprehensive test documentation
2. **`docs/sessions/2025-11-03_test_organization.md`** (this document)

### Moved
1. `test_building_blocks_simple.py` → `validation/analysis/`
2. `test_building_blocks_edp.py` → `validation/analysis/`
3. `test_mermaid_visualization.py` → `tests/visualization/`

### Modified
1. **`.github/workflows/ci.yml`** (line 90) - Added analysis validation test

---

## Verification

### Tests Run Successfully in New Locations

```bash
$ cd validation/analysis && python test_building_blocks_simple.py
====================================================================================================
PROGRESSIVE BUILDING BLOCK TESTING
Testing subgraph-level EDP breakdown on core DNN building blocks
====================================================================================================

TEST 1: MLP (Linear → ReLU → Linear → ReLU)
Expected: 4 subgraphs, Linear layers should dominate
...
✅ ALL BUILDING BLOCKS PASSED!
```

### Pytest Discovery Works

```bash
$ pytest validation/analysis/ -v
...
validation/analysis/test_building_blocks_simple.py::TestBuildingBlocks::test_mlp PASSED
validation/analysis/test_building_blocks_simple.py::TestBuildingBlocks::test_conv2d PASSED
...
```

### No Stray Test Files in Root

```bash
$ ls -la *.py
-rw-rw-r-- 1 stillwater stillwater 1633 Oct 17 06:58 how-to-run-characterization.py
```

Only `how-to-run-characterization.py` remains (documentation script, not a test).

---

## Summary

✅ **Test Organization Complete:**
- All tests moved to appropriate locations
- New directories created with documentation
- CI integration added
- Tests verified to work in new locations
- No stray test files in root

✅ **Directory Structure Clear:**
- `tests/` - Unit tests (blocking)
- `validation/` - Functional validation (non-blocking)
- Organized by module/capability

✅ **Documentation Complete:**
- `validation/analysis/README.md` - Comprehensive test guide
- This session document - Reorganization details

✅ **CI Integration:**
- Analysis validation tests added to pipeline
- Non-blocking to allow iterative development
- Will become blocking as tests mature

**Ready for Phase 2 implementation with proper test infrastructure!**
