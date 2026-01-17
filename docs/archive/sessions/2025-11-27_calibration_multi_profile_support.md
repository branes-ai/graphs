# Session Summary: Multi-Calibration Profile Support & Bug Fixes

**Date**: 2025-11-27
**Duration**: ~4 hours
**Phase**: Hardware Calibration Infrastructure
**Status**: Complete

---

## Goals for This Session

1. Implement calibrations subdirectory structure for multiple calibrations per hardware profile
2. Fix GEMM early termination bug causing poor results on small matrices
3. Add TF32 precision support to GPU calibration
4. Fix duplicate precision bug in supported/unsupported lists
5. Add `list_calibrations` script for hardware registry management

---

## What We Accomplished

### 1. Multi-Calibration Directory Structure

**Description**: Restructured hardware registry to support multiple calibrations per hardware profile, with filenames encoding power mode, frequency, and framework.

**Implementation**:
- Modified: `src/graphs/hardware/registry/profile.py`
  - Added `_make_calibration_filename()` - generates `{power_mode}_{freq}MHz_{framework}.json`
  - Added `_parse_calibration_filename()` - parses calibration filenames
  - Updated `save()` to save to `calibrations/` subdirectory
  - Updated `load()` with `calibration_filter` parameter for selecting specific calibrations
  - Added `_load_calibration()` method for filtered/most-recent selection
  - Added `list_calibrations()` class method

- Modified: `src/graphs/hardware/registry/registry.py`
  - Updated docstring with new directory structure
  - Added `calibration_filter` parameter to `get()` method
  - Added `list_calibrations()` method

- Created: `cli/migrate_calibrations.py` - migration script for old calibration.json files

**New Directory Structure**:
```
hardware_registry/
├── cpu/
│   └── jetson_orin_nano_cpu/
│       ├── spec.json
│       └── calibrations/
│           ├── schedutil_729MHz_numpy.json
│           ├── schedutil_883MHz_numpy.json
│           └── performance_960MHz_pytorch.json
└── gpu/
    └── jetson_orin_nano_gpu/
        ├── spec.json
        └── calibrations/
            ├── 7W_306MHz_pytorch.json
            ├── 15W_625MHz_pytorch.json
            └── MAXN_625MHz_pytorch.json
```

### 2. Fixed GEMM Early Termination Bug

**Description**: GEMM benchmarks were only testing size 32 and stopping, showing incorrect 0.3-0.5 GFLOPS results.

**Root Cause**: Early termination logic triggered at size 32 where timing overhead dominates, giving artificially low GFLOPS that fell below the 1.0 GFLOPS threshold.

**Fix**:
- Added `min_early_termination_size = 256` threshold
- Early termination only triggers after testing size 256+
- Applied to both `benchmarks/numpy/blas_bench.py` and `benchmarks/pytorch/blas_bench.py`

**Result**: Jetson CPU now correctly shows 50+ GFLOPS for GEMM instead of 0.5 GFLOPS.

### 3. Added TF32 Precision Support

**Description**: TF32 (TensorFloat-32) was not being tested or displayed in GPU calibration.

**Root Causes**:
1. `tf32` missing from default GPU precision list in `calibrator.py`
2. `tf32` missing from `CANONICAL_PRECISION_ORDER` in `schema.py`

**Fixes**:
- Added `tf32` to GPU precision list: `['fp64', 'fp32', 'tf32', 'fp16', 'bf16', ...]`
- Added `tf32` to `CANONICAL_PRECISION_ORDER` between `fp32` and `fp16`

### 4. Fixed Duplicate Precision Bug

**Description**: Precision Support Summary showed some precisions (int64, int32, int16, int8) in both supported and unsupported lists.

**Root Cause**: Order of operation processing could add precision to unsupported before it was added to supported by a later operation.

**Fix**: Added final cleanup `actually_unsupported -= actually_supported` before building precision matrix.

### 5. Added list_calibrations Script

**Description**: New script to list and filter calibrations in the hardware registry.

**Created**: `scripts/hardware_db/list_calibrations.py`

**Features**:
- List all calibrations across registry
- Filter by `--hardware`, `--framework`, `--power-mode`
- `--detail` shows performance metrics (GFLOPS, bandwidth, date)
- `--summary` shows statistics by hardware/framework/power mode

**Example Output**:
```
jetson_orin_nano_cpu (Jetson Orin Nano (CPU))
----------------------------------------------------------------------
  - schedutil       729 MHz      numpy
  - schedutil       883 MHz      numpy
  - performance     960 MHz      pytorch
```

---

## Key Insights

1. **Early Termination Needs Size Guard**: Small matrix benchmarks have high overhead, making absolute GFLOPS thresholds unreliable. Must reach minimum computational size before applying performance thresholds.

2. **Canonical Ordering Lists Must Be Complete**: Display loops using `for p in CANONICAL_ORDER if p in results` silently skip precisions not in the canonical list.

3. **Set Operations for Cleanup**: Using `set -= set` is a clean way to handle edge cases in supported/unsupported classification.

---

## Files Created/Modified

### Source Code
- `src/graphs/hardware/registry/profile.py` - Multi-calibration support (~100 lines added)
- `src/graphs/hardware/registry/registry.py` - Registry updates (~30 lines added)
- `src/graphs/hardware/calibration/calibrator.py` - TF32 support, duplicate fix (~10 lines)
- `src/graphs/hardware/calibration/schema.py` - Added tf32 to canonical order (1 line)
- `src/graphs/hardware/calibration/benchmarks/numpy/blas_bench.py` - Early termination fix (~10 lines)
- `src/graphs/hardware/calibration/benchmarks/pytorch/blas_bench.py` - Early termination fix (~10 lines)

### CLI Tools
- `cli/migrate_calibrations.py` (new, ~100 lines) - Migration script
- `scripts/hardware_db/list_calibrations.py` (new, ~150 lines) - List calibrations

### Documentation
- `docs/sessions/2025-11-27_calibration_multi_profile_support.md` - This session log

**Total**: ~400 lines of code/docs

---

## Validation/Testing

### Tests Run
- Migration script dry-run: Pass - correctly identified 2 calibrations to migrate
- Migration script live: Pass - migrated to new structure
- Registry load: Pass - loads profiles with new calibrations/ structure
- list_calibrations script: Pass - correctly lists and filters calibrations

### Validation Results
- Jetson CPU GEMM: 54.7 GFLOPS (was incorrectly showing 0.5 GFLOPS)
- i7-12700K GEMM: 637.4 GFLOPS (correct)

---

## Challenges & Solutions

### Challenge 1: GEMM Early Termination at Size 32
**Issue**: 32x32 GEMM = 65K FLOPs at 0.13ms = 0.5 GFLOPS, below 1.0 threshold

**Solution**: Added `min_early_termination_size = 256` (33M FLOPs) where timing is reliable

### Challenge 2: TF32 Not Displayed
**Issue**: TF32 tested but not shown in output

**Solution**: Added `tf32` to `CANONICAL_PRECISION_ORDER` - display loops filter by this list

### Challenge 3: Duplicate Precisions in Summary
**Issue**: int64/32/16/8 appeared in both supported and unsupported

**Solution**: Set difference cleanup before building precision matrix

---

## Next Steps

### Immediate
1. [ ] Run full GPU calibration with TF32 support
2. [ ] Verify TF32 performance is between FP32 and FP16 on Tensor Cores
3. [ ] Test multi-power-mode calibration workflow on Jetson

### Short Term
1. [ ] Add calibration comparison tool (compare MAXN vs 7W performance)
2. [ ] Add calibration selection to analysis CLI (--calibration-filter)

---

## Code Snippets / Examples

### New Calibration Filename Format
```python
def _make_calibration_filename(calibration: HardwareCalibration) -> str:
    """Format: {power_mode}_{frequency_mhz}MHz_{framework}.json"""
    # Examples:
    #   MAXN_625MHz_pytorch.json
    #   7W_306MHz_numpy.json
    #   performance_4900MHz_numpy.json
```

### Using Calibration Filter
```python
# Load profile with specific calibration
profile = registry.get('jetson_orin_nano_gpu', calibration_filter={
    'power_mode': 'MAXN',
    'framework': 'pytorch'
})

# List all calibrations for a profile
cals = registry.list_calibrations('jetson_orin_nano_cpu')
for cal in cals:
    print(f"{cal['power_mode']}_{cal['freq_mhz']}MHz_{cal['framework']}")
```

### List Calibrations CLI
```bash
# List all calibrations
python scripts/hardware_db/list_calibrations.py

# Filter by hardware
python scripts/hardware_db/list_calibrations.py --hardware jetson_orin_nano_cpu

# Show detailed info
python scripts/hardware_db/list_calibrations.py --detail

# Show summary statistics
python scripts/hardware_db/list_calibrations.py --summary
```

---

## Session Notes

### Decisions Made
1. No backward compatibility for old calibration.json - user requested clean break
2. Most recent calibration selected by default when no filter specified
3. TF32 placed between FP32 and FP16 in canonical order (reflects precision hierarchy)

### Related Sessions
- [2025-11-17_hardware_database_implementation.md](2025-11-17_hardware_database_implementation.md) - Original hardware database
- [2025-11-26_hardware_schema_cleanup.md](2025-11-26_hardware_schema_cleanup.md) - Schema consolidation
