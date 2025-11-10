# Session Log: KPU Model Duplication Cleanup

**Date**: 2025-11-10  
**Duration**: 5 minutes  
**Objective**: Remove duplicate KPU model files across deployment directories

---

## Problem Identified

Duplicate KPU model files existed across different deployment directories:

```
./src/graphs/hardware/models/accelerators/kpu_t768.py (314 lines) ✓ Canonical
./src/graphs/hardware/models/automotive/kpu_t768.py (207 lines)   ✗ Duplicate

./src/graphs/hardware/models/accelerators/kpu_t256.py (341 lines) ✓ Canonical
./src/graphs/hardware/models/mobile/kpu_t256.py (206 lines)       ✗ Duplicate

./src/graphs/hardware/models/accelerators/kpu_t64.py (332 lines)  ✓ Canonical
./src/graphs/hardware/models/edge/kpu_t64.py (204 lines)          ✗ Duplicate
```

**All had the same function name**: `kpu_t{size}_resource_model()`

---

## Analysis

### Current Usage

**Main Package** (`models/__init__.py`):
- Imported from `accelerators/` versions (newer, more detailed)

**Test Files**:
- `test_kpu_tile_energy.py` - imported from `automotive/`, `mobile/`, `edge/`
- `test_tpu_vs_kpu_energy_breakdown.py` - imported from `automotive/`

### Root Cause

Incomplete migration from October 2024 refactoring:
- `accelerators/` versions created during package reorganization
- Old deployment-specific versions not removed
- Led to confusion and potential import conflicts

### Differences

**Accelerators versions** (canonical):
- 314-341 lines
- More detailed documentation
- Includes thermal operating points
- Includes BOM cost profiles
- Complete performance characteristics

**Deployment-specific versions** (old):
- 204-207 lines
- Simpler documentation
- Missing thermal profiles
- Missing cost models
- Less comprehensive

---

## Solution Applied

**Option 1: Remove Old Deployment-Specific Files** ✅

Removed duplicate files:
```bash
rm src/graphs/hardware/models/automotive/kpu_t768.py
rm src/graphs/hardware/models/mobile/kpu_t256.py
rm src/graphs/hardware/models/edge/kpu_t64.py
```

Updated test imports:
```python
# Before:
from graphs.hardware.models.edge.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.mobile.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.automotive.kpu_t768 import kpu_t768_resource_model

# After:
from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.accelerators.kpu_t768 import kpu_t768_resource_model
```

---

## Files Modified

### Deleted (3 files):
- `src/graphs/hardware/models/automotive/kpu_t768.py`
- `src/graphs/hardware/models/mobile/kpu_t256.py`
- `src/graphs/hardware/models/edge/kpu_t64.py`

### Updated (2 files):
- `tests/hardware/test_kpu_tile_energy.py` - Updated 3 imports
- `tests/hardware/test_tpu_vs_kpu_energy_breakdown.py` - Updated 1 import

---

## Testing

### Import Verification
```python
from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.accelerators.kpu_t768 import kpu_t768_resource_model

# ✅ All imports work
# ✅ Model names: "Stillwater KPU-T64", "Stillwater KPU-T256", "Stillwater KPU-T768"
```

### Duplicate Check
```bash
grep -r "from.*automotive.*kpu_t768" --include="*.py" .
grep -r "from.*mobile.*kpu_t256" --include="*.py" .
grep -r "from.*edge.*kpu_t64" --include="*.py" .

# ✅ No results (all old references removed)
```

---

## Benefits

1. **Single Source of Truth**: Only one canonical version of each KPU model
2. **Eliminates Confusion**: Clear where KPU models live (`accelerators/`)
3. **Better Documentation**: Canonical versions have more complete specs
4. **Consistent Imports**: All code uses same import path
5. **Reduced Maintenance**: Only one file to update per model

---

## Rationale

**Why accelerators/ instead of deployment-specific?**

1. **KPU is hardware-agnostic**: Same silicon can be deployed in automotive, mobile, edge, or datacenter
2. **Deployment is use-case, not hardware**: T768 specs don't change based on where it's deployed
3. **Consistent with other accelerators**: TPU, DPU, CGRA all in `accelerators/`
4. **Follows refactoring pattern**: October 2024 reorganization established `accelerators/` as canonical location

**Deployment categorization should be done at mapper level, not model level:**
- Hardware specs (model): architecture-dependent
- Deployment constraints (mapper): use-case-dependent

---

## Impact

**Zero Breaking Changes**:
- Main package already used `accelerators/` versions
- Only test files needed updates
- All tests still pass
- No user-facing API changes

**Cleanup Complete**:
- 3 duplicate files removed
- 2 test files updated
- All imports verified working
- No remaining references to old paths

---

## Conclusion

Successfully eliminated KPU model duplication by:
- Removing 3 older deployment-specific files
- Updating 2 test files to use canonical `accelerators/` versions
- Establishing single source of truth for KPU models

This cleanup aligns with the October 2024 package reorganization and makes the codebase more maintainable.

---

**Status**: ✅ Complete  
**Files Changed**: 5 (3 deleted, 2 updated)  
**Tests**: ✅ All passing  
**Next Action**: User will commit changes
