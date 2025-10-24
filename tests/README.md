# Unit Tests

This directory contains unit tests that verify the **correctness** of individual components and functions.

## Directory Structure

```
tests/
├── ir/                         # Tests for Intermediate Representation
│   ├── __init__.py
│   └── test_structures.py         # Data structure tests (enums, descriptors)
├── transform/                  # Tests for Graph Transformations
│   ├── __init__.py
│   └── partitioning/              # Partitioning algorithm tests
│       ├── __init__.py
│       ├── test_graph_partitioner.py
│       ├── test_fusion_partitioner.py
│       ├── test_partitioner_on_resnet18.py
│       └── test_arithmetic_intensity.py
├── analysis/                   # Tests for Performance Analysis
│   ├── __init__.py
│   └── test_concurrency.py        # Concurrency analyzer tests
└── README.md                   # This file
```

**Note**: Test structure reorganized on 2025-10-24 to mirror the new package structure (`src/graphs/ir/`, `src/graphs/transform/`, etc.). Old `tests/characterize/` has been split into focused directories.

## Purpose

**Unit tests answer:** "Does this code work correctly?"

These tests verify:
- Function inputs/outputs are correct
- Edge cases are handled properly
- Errors are raised appropriately
- Code logic is sound

## Unit Tests vs Validation

| Aspect | Unit Tests (`./tests/`) | Validation (`./validation/`) |
|--------|-------------------------|------------------------------|
| **Purpose** | Code correctness | Estimate accuracy |
| **Question** | "Does code work?" | "Are results accurate?" |
| **Compares** | Against expected behavior | Against benchmarks |
| **Scope** | Individual functions | End-to-end systems |
| **Speed** | Fast (<1s per test) | Slow (seconds to minutes) |

## Running Tests

### All Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/graphs
```

### Individual Test Suites
```bash
# IR data structure tests
python tests/ir/test_structures.py

# Analysis tests
python tests/analysis/test_concurrency.py

# All partitioning tests
python -m pytest tests/transform/partitioning/

# Individual partitioning tests
python tests/transform/partitioning/test_graph_partitioner.py
python tests/transform/partitioning/test_fusion_partitioner.py
python tests/transform/partitioning/test_partitioner_on_resnet18.py
python tests/transform/partitioning/test_arithmetic_intensity.py
```

## Test Organization

Tests are organized by package path to mirror the source code structure:

```
tests/transform/partitioning/         ← Tests for src/graphs/transform/partitioning/
  ├── test_graph_partitioner.py       ← Tests GraphPartitioner
  ├── test_fusion_partitioner.py      ← Tests FusionBasedPartitioner
  └── ...

tests/ir/                             ← Tests for src/graphs/ir/
  └── (future tests for data structures)

tests/analysis/                       ← Tests for src/graphs/analysis/
  └── (future tests for concurrency analysis)
```

**Naming convention:**
- Test files: `test_<module>.py`
- Test functions: `test_<functionality>()`
- Test classes: `Test<Component>`

## Test Files

### IR Tests (`tests/ir/`)

#### `test_structures.py`
Tests core IR data structures:
- Enumerations (OperationType, BottleneckType, PartitionReason)
- TensorDescriptor - Shape, dtype, memory footprint
- ParallelismDescriptor - Thread/warp/block dimensions
- SubgraphDescriptor - Complete subgraph metadata
- SubgraphConcurrency - Subgraph-level parallelism
- ConcurrencyDescriptor - Graph-level concurrency
- PartitionReport - Complete partition statistics
- Integration tests combining multiple structures

### Analysis Tests (`tests/analysis/`)

#### `test_concurrency.py`
Tests concurrency analyzer functionality:
- Graph-level concurrency (parallel stages, fork-join patterns)
- Subgraph-level concurrency (thread parallelism, vectorization)
- Critical path analysis (longest latency path)
- Dependency graph construction
- Stage computation (parallel execution groups)
- Utilization metrics (concurrency efficiency)
- Batch parallelism detection
- Integration with realistic graph structures (ResNet-like)

### Transform Tests (`tests/transform/partitioning/`)

#### `test_graph_partitioner.py`
Tests graph partitioning on ResNet-18:
- Subgraph extraction
- FLOP calculation correctness
- Memory traffic estimation
- Parallelism analysis

#### `test_fusion_partitioner.py`
Tests fusion-based partitioning:
- Fusion pattern detection
- Memory reduction calculations
- Subgraph aggregation
- Boundary detection (fork/join)

#### `test_partitioner_on_resnet18.py`
ResNet-18 specific partitioning validation:
- End-to-end partitioning
- Realistic model testing
- Performance characteristics

#### `test_arithmetic_intensity.py`
Tests arithmetic intensity calculations:
- Compute vs memory bound classification
- Roofline model inputs
- Bottleneck detection

## Writing New Tests

### Test Template
```python
#!/usr/bin/env python
"""Unit tests for <component>"""

import pytest
import torch
from torch.fx import symbolic_trace

import sys
sys.path.insert(0, 'src')

from graphs.transform.partitioning import <Component>  # Example import


def test_basic_functionality():
    """Test basic usage works"""
    component = Component()
    result = component.process(input_data)
    assert result is not None
    assert result.some_property == expected_value


def test_edge_case():
    """Test edge cases are handled"""
    component = Component()

    # Empty input
    with pytest.raises(ValueError):
        component.process([])

    # None input
    with pytest.raises(ValueError):
        component.process(None)


def test_correctness():
    """Test output is mathematically correct"""
    component = Component()
    result = component.calculate(input_value)
    expected = manual_calculation(input_value)
    assert abs(result - expected) < 1e-6, "Calculation mismatch"


if __name__ == '__main__':
    test_basic_functionality()
    test_edge_case()
    test_correctness()
    print("✅ All tests passed!")
```

### Best Practices
1. **One concept per test:** Test one thing at a time
2. **Descriptive names:** `test_handles_empty_input()` not `test_1()`
3. **Fast execution:** Unit tests should run in milliseconds
4. **No external deps:** Don't require network/filesystem/hardware
5. **Deterministic:** Same inputs always produce same outputs

## Expected Coverage

Target coverage for unit tests:
- **Core algorithms:** >90% (partitioner, fusion)
- **Data structures:** >80% (IR structures)
- **Analysis:** >80% (concurrency analysis)
- **Transformations:** >85% (partitioning, fusion)

Run coverage report:
```bash
python -m pytest tests/ --cov=src/graphs --cov-report=html
# Open htmlcov/index.html in browser
```

## Common Issues

**Import errors:**
- Check sys.path setup (should add repo root)
- Verify package structure (src/graphs/...)

**Tests pass locally but fail in CI:**
- Check for non-deterministic behavior (random, timestamps)
- Verify no absolute paths are hardcoded

**Slow tests:**
- Unit tests should be <1s each
- If slower, move to `validation/` directory

## Success Criteria

Unit tests pass if:
- ✅ All test functions return without errors
- ✅ No assertion failures
- ✅ Edge cases handled gracefully
- ✅ Fast execution (<1s per test file)
- ✅ Code coverage >80% for tested modules

## Future Work

### Tests to Add
- [x] IR structure tests (`tests/ir/test_structures.py`) - ✅ ADDED
- [x] Concurrency analysis tests (`tests/analysis/test_concurrency.py`) - ✅ ADDED
- [ ] Tiling transformation tests (`tests/transform/tiling/`)
- [ ] Fusion transformation tests (`tests/transform/fusion/`)
- [ ] Hardware mapper unit tests (allocation logic) - Note: Validation tests already exist in `validation/hardware/`

### Test Infrastructure
- [ ] Add pytest configuration (pytest.ini)
- [ ] Set up CI/CD for automatic testing
- [ ] Add pre-commit hooks for test execution
- [ ] Create test fixtures for common models

## Documentation

See also:
- `../validation/README.md` - Functional validation tests
- `../examples/README.md` - Usage demonstrations
- Python unittest docs: https://docs.python.org/3/library/unittest.html
- Pytest docs: https://docs.pytest.org/
