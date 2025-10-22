# Unit Tests

This directory contains unit tests that verify the **correctness** of individual components and functions.

## Directory Structure

```
tests/
├── characterize/               # Tests for characterization components
│   ├── test_graph_partitioner.py
│   ├── test_graph_partitioner_general.py
│   ├── test_fusion_partitioner.py
│   └── test_arithmetic_intensity.py
└── README.md                   # This file
```

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
python -m pytest tests/ --cov=src/graphs/characterize
```

### Individual Test Files
```bash
# Graph partitioner tests
python tests/characterize/test_graph_partitioner.py
python tests/characterize/test_graph_partitioner_general.py

# Fusion partitioner tests
python tests/characterize/test_fusion_partitioner.py

# Arithmetic intensity tests
python tests/characterize/test_arithmetic_intensity.py
```

## Test Organization

Tests are organized by package path to mirror the source code structure:

```
tests/characterize/              ← Tests for src/graphs/characterize/
  ├── test_graph_partitioner.py  ← Tests graph_partitioner.py
  ├── test_fusion_partitioner.py ← Tests fusion_partitioner.py
  └── ...
```

**Naming convention:**
- Test files: `test_<module>.py`
- Test functions: `test_<functionality>()`
- Test classes: `Test<Component>`

## Test Files

### `test_graph_partitioner.py`
Tests graph partitioning on ResNet-18:
- Subgraph extraction
- FLOP calculation correctness
- Memory traffic estimation
- Parallelism analysis

### `test_graph_partitioner_general.py`
Universal tests on multiple models:
- ResNet-18, MobileNet-V2, EfficientNet-B0
- Validates consistency across architectures
- Tests edge cases (batch sizes, input shapes)

### `test_fusion_partitioner.py`
Tests fusion-based partitioning:
- Fusion pattern detection
- Memory reduction calculations
- Subgraph aggregation
- Boundary detection (fork/join)

### `test_arithmetic_intensity.py`
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
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.characterize.<module> import <Component>


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
- **Core algorithms:** >90% (partitioner, fusion, walker)
- **Utilities:** >80% (formatters, helpers)
- **Mappers:** >70% (hardware-specific logic)

Run coverage report:
```bash
python -m pytest tests/ --cov=src/graphs/characterize --cov-report=html
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
- [ ] Hardware mapper unit tests (allocation logic)
- [ ] Precision profile tests
- [ ] Clock domain DVFS tests
- [ ] Tiling strategy tests
- [ ] Energy model tests

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
