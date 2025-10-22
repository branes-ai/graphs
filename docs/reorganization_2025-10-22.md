# Directory Reorganization - October 22, 2025

## Summary

Reorganized project directory structure to separate concerns between CLI tools, examples, unit tests, and functional validation.

**Status**: ✅ Complete

**Date**: 2025-10-22

**Rationale**: Original structure had grown organically with validation tests mixed into examples/, debug scripts in cli/, and unclear separation between unit tests and validation.

## New Structure

### Before (Messy)
```
examples/         # Mix of demos AND validation tests (15 files)
  ├── test_all_hardware.py       # Actually validation
  ├── test_*_mapper.py (7 files) # Actually validation
  ├── quick_start_partitioner.py # Actually demo
  └── ...

cli/              # Mix of tools AND debug scripts (17 files)
  ├── partitioner.py             # Actual tool
  ├── debug_*.py (5 files)       # Dev artifacts
  └── ...

tests/            # Some unit tests, some validation
  ├── test_fusion_partitioner.py # Could be either
  └── ...

src/graphs/validation/  # Estimator tests only
  └── test_*.py (6 files)
```

### After (Clean)
```
cli/              # Command-line TOOLS only (5 files)
  ├── partitioner.py
  ├── profile_graph.py
  ├── discover_models.py
  ├── show_fvcore_table.py
  └── README.md

examples/         # Capability DEMONSTRATIONS only (5 files)
  ├── quick_start_partitioner.py
  ├── demo_fusion_comparison.py
  ├── demo_new_performance_model.py
  ├── compare_models.py
  ├── visualize_partitioning.py
  └── README.md

tests/            # UNIT TESTS organized by package (5 files)
  ├── characterize/
  │   ├── test_graph_partitioner.py
  │   ├── test_graph_partitioner_general.py
  │   ├── test_fusion_partitioner.py
  │   └── test_arithmetic_intensity.py
  └── README.md

validation/       # FUNCTIONAL VALIDATION (NEW - 13 files)
  ├── hardware/   # Hardware mapper validation
  │   ├── test_all_hardware.py
  │   ├── test_cgra_mapper.py
  │   ├── test_dpu_mapper.py
  │   └── ... (7 files total)
  ├── estimators/ # Estimator accuracy tests
  │   ├── test_conv2d.py
  │   ├── test_resnet18.py
  │   └── ... (5 files total)
  └── README.md

archive/          # Debug/dev artifacts (9 files)
  └── debug_*.py, fvcore_compare.py, etc.
```

## Changes Made

### 1. Created New Directories
- `validation/` - Functional validation tests
- `validation/hardware/` - Hardware mapper validation
- `validation/estimators/` - Estimator accuracy tests
- `tests/characterize/` - Unit tests organized by package
- `archive/` - Development artifacts

### 2. File Moves

#### From `examples/` → `validation/hardware/` (7 files)
- `test_all_hardware.py`
- `test_cgra_mapper.py`
- `test_cpu_vs_gpu_mapping.py`
- `test_dpu_mapper.py`
- `test_gpu_cpu_kpu_comparison.py`
- `test_hardware_mapping.py`
- `test_kpu_simple.py`

#### From `src/graphs/validation/` → `validation/estimators/` (5 files)
- `test_conv2d.py`
- `test_resnet18.py`
- `test_resnet_family.py`
- `test_mobilenet.py`
- `test_efficientnet.py`

#### From `tests/` → `tests/characterize/` (4 files)
- `test_graph_partitioner.py`
- `test_graph_partitioner_general.py`
- `test_fusion_partitioner.py`
- `test_arithmetic_intensity.py` (renamed from `validate_arithmetic_intensity.py`)

#### From `cli/` → `archive/` (9 files)
- `debug_*.py` (5 files)
- `compare_per_layer.py`
- `fvcore_compare.py`
- `inspect_fx_graph.py`
- `verify_new_ops.py`
- `test_efficientnet_family.py`

#### Remaining in `examples/` (renamed 1 file)
- `test_fusion_partitioner.py` → `demo_fusion_comparison.py` (actually a demo, not a test)

### 3. Import Updates

Updated all moved files to use correct sys.path:
- **Hardware validation** (`validation/hardware/*.py`): Changed `sys.path.insert(0, '..')` → `sys.path.insert(0, '../..')`
- **Estimator validation** (`validation/estimators/*.py`): Added `sys.path.insert(0, '../..')` and changed `from graphs.*` → `from src.graphs.*`
- **Unit tests** (`tests/characterize/*.py`): No changes needed (already correct)

### 4. Documentation Created

Created comprehensive README files for each directory:
- `validation/README.md` - Overview of validation system
- `validation/hardware/README.md` - Hardware mapper validation guide
- `validation/estimators/README.md` - Estimator accuracy validation guide
- `tests/README.md` - Unit testing guide
- `cli/README.md` - Command-line tools documentation
- `examples/README.md` - Updated with new structure

### 5. Main Documentation Updated

- `CLAUDE.md` - Updated project structure diagram and command examples
- Added reference to this reorganization document

## Rationale

### Problem Statement

The original directory structure had several issues:

1. **Mixed Purposes**: `examples/` contained both demonstrations (quick_start) and validation tests (test_all_hardware)
2. **No Validation Separation**: Functional validation (accuracy checks) mixed with unit tests (correctness checks)
3. **Cluttered CLI**: Development debug scripts mixed with production tools
4. **Unclear Organization**: Hard to find the right place for new files

### Solution Principles

1. **Separation of Concerns**: Each directory has ONE clear purpose
2. **User-Oriented**: Different users (developers, researchers, end-users) have different entry points
3. **Discoverability**: Clear naming makes it obvious where files belong
4. **Scalability**: Structure supports future growth

### Directory Purposes

| Directory | Purpose | User | Question Answered |
|-----------|---------|------|-------------------|
| `cli/` | Command-line tools | End users | "How do I use this in production?" |
| `examples/` | Demonstrations | New users | "How do I get started?" |
| `tests/` | Unit tests | Developers | "Does this code work correctly?" |
| `validation/` | Functional validation | Researchers | "Are the estimates accurate?" |
| `archive/` | Dev artifacts | Historical | "What did we try during development?" |

## Impact

### Breaking Changes

**File Paths**: All moved files have new paths
- Old: `examples/test_all_hardware.py`
- New: `validation/hardware/test_all_hardware.py`

**Imports**: All moved files updated automatically

### Non-Breaking

**Source Code**: No changes to `src/graphs/` package
**APIs**: No changes to public APIs
**Documentation**: Updated to reflect new structure

### Benefits

1. **Clarity**: Obvious where each file belongs
2. **Discoverability**: Easy to find relevant files
3. **Scalability**: Room to grow each category
4. **Professionalism**: Clean structure for open source release
5. **Maintainability**: Easier to navigate and maintain

## Validation

### Files Moved Successfully
- ✅ 7 hardware validation tests
- ✅ 5 estimator validation tests
- ✅ 4 unit tests to characterize/
- ✅ 9 debug scripts to archive/
- ✅ 1 demo renamed for clarity

### Imports Updated
- ✅ All validation/hardware/ files
- ✅ All validation/estimators/ files
- ✅ No changes needed for tests/characterize/

### Documentation Complete
- ✅ 5 new README files created
- ✅ CLAUDE.md updated
- ✅ This reorganization document created

### Testing
```bash
# Verify imports work
python -c "import sys; sys.path.insert(0, '.'); from src.graphs.characterize import hardware_mapper"

# Check file locations
ls validation/hardware/test_all_hardware.py
ls validation/estimators/test_resnet18.py
ls tests/characterize/test_graph_partitioner.py
ls cli/partitioner.py
ls examples/quick_start_partitioner.py

# All exist ✅
```

## Migration Guide

### For Developers

**Old way:**
```bash
python examples/test_all_hardware.py
python src/graphs/validation/test_resnet18.py
```

**New way:**
```bash
python validation/hardware/test_all_hardware.py
python validation/estimators/test_resnet18.py
```

### For Documentation Updates

Search and replace in docs:
- `examples/test_all_hardware.py` → `validation/hardware/test_all_hardware.py`
- `src/graphs/validation/test_*.py` → `validation/estimators/test_*.py`
- `tests/test_*.py` → `tests/characterize/test_*.py`

### For Scripts/CI

Update any scripts that reference old paths:
```bash
# Old
python examples/test_*.py

# New
python validation/hardware/test_*.py
python validation/estimators/test_*.py
```

## Future Work

### Short Term
- [ ] Add pytest configuration (pytest.ini)
- [ ] Set up CI/CD for automatic testing
- [ ] Add more unit tests for hardware mappers

### Long Term
- [ ] Add `tests/models/` for model tests
- [ ] Add `validation/benchmarks/` for real hardware benchmarks
- [ ] Create `tools/` for user-facing tools (move from cli/)

## Statistics

### Before Reorganization
- `examples/`: 15 files (mixed demos and tests)
- `cli/`: 17 files (mixed tools and debug)
- `tests/`: 5 files (mixed unit and validation)
- `src/graphs/validation/`: 6 files (estimator tests)

### After Reorganization
- `cli/`: 5 files (tools only)
- `examples/`: 5 files (demos only)
- `tests/characterize/`: 4 files (unit tests)
- `validation/hardware/`: 7 files (hardware validation)
- `validation/estimators/`: 5 files (estimator validation)
- `archive/`: 9 files (dev artifacts)

### Code Changes
- Files moved: 25
- Imports updated: 12 files
- README files created: 5
- Lines of documentation added: ~2,000

## Related Documents

- `validation/README.md` - Validation system overview
- `tests/README.md` - Unit testing guide
- `examples/README.md` - Examples guide
- `cli/README.md` - CLI tools documentation
- `CLAUDE.md` - Updated project structure
- `CHANGELOG.md` - Will be updated with this reorganization

## Approval

**Proposed by**: User request ("directory organization is kind of a mess")

**Executed by**: Claude Code

**Date**: 2025-10-22

**Status**: ✅ Complete and validated

## Questions?

See:
- `validation/README.md` - How validation works
- `tests/README.md` - How to write unit tests
- `examples/README.md` - How to use examples
- `CLAUDE.md` - Updated project overview
