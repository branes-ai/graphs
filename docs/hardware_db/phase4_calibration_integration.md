# Phase 4: Calibration Integration - Implementation Summary

**Date**: 2025-11-17
**Status**: ✅ Complete

## Overview

Phase 4 integrates the hardware database with the calibration system, replacing the hardcoded `--preset` flag with automatic hardware detection and database lookups. This eliminates the bug-prone manual preset selection and provides a seamless workflow from detection to calibration to performance comparison.

## Key Changes

### 1. Calibration CLI Refactoring (`cli/calibrate_hardware.py`)

**Removed:**
- Required `--preset` flag (now deprecated)
- Hardcoded PRESETS dependency for hardware selection

**Added:**
- Auto-detection as the default mode
- `--id` flag to specify hardware from database
- Hardware database integration
- Backward compatibility with `--preset` (deprecated with warning)

**New Usage Patterns:**

```bash
# Auto-detect and calibrate (default)
./cli/calibrate_hardware.py

# Calibrate specific hardware from database
./cli/calibrate_hardware.py --id i7_12700k
./cli/calibrate_hardware.py --id h100_sxm5

# Quick calibration
./cli/calibrate_hardware.py --quick

# Specific operations
./cli/calibrate_hardware.py --operations blas
./cli/calibrate_hardware.py --operations dot,gemm

# Legacy preset mode (deprecated)
./cli/calibrate_hardware.py --preset i7-12700k
```

### 2. Auto-Detection Integration

**New Function: `auto_detect_hardware()`**

Automatically detects current hardware and matches to database:

```python
def auto_detect_hardware(db: HardwareDatabase):
    """
    Auto-detect current hardware and match to database.

    Returns:
        tuple: (matched_spec, confidence) or (None, 0.0) if no match
    """
```

**Detection Flow:**
1. Detects CPU using cross-platform libraries (psutil, py-cpuinfo)
2. Detects GPUs using nvidia-smi or PyTorch CUDA
3. Matches detected hardware against database patterns
4. Returns best match with confidence score
5. Prompts user if confidence < 50%

**Example Output:**

```
================================================================================
Auto-Detecting Hardware
================================================================================
CPU:      12th Gen Intel(R) Core(TM) i7-12700K
Vendor:   Intel
Cores:    12 cores, 20 threads
          (8P + 4E cores)

✓ Matched to database: i7_12700k
  Confidence: 100%
  Device: CPU
```

### 3. Database-Driven Calibration

**Hardware Spec Extraction:**

```python
# Extract calibration parameters from hardware spec
hardware_name = hardware_spec.model
device = 'cuda' if hardware_spec.device_type == 'gpu' else hardware_spec.device_type
theoretical_peaks = hardware_spec.theoretical_peaks
peak_bandwidth = hardware_spec.peak_bandwidth_gbps
```

**Benefits:**
- No hardcoded presets
- Centralized hardware specs
- Easy to add new hardware
- Automatic updates when database changes

### 4. Comparison Tool (`scripts/hardware_db/compare_calibration.py`)

New tool to compare calibrated results vs theoretical performance:

```bash
# Auto-identify hardware from calibration
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/i7_12700k_numpy.json

# Specify hardware explicitly
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/h100_sxm5_pytorch.json \
    --id h100_sxm5

# Detailed operation breakdown
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/i7_12700k_numpy.json \
    --verbose
```

**Output Features:**
- Memory bandwidth comparison (theoretical vs measured)
- Compute performance by precision
- Efficiency percentages
- Detailed operation breakdown (verbose mode)
- Performance assessment with recommendations

**Example Output:**

```
================================================================================
Theoretical vs Calibrated Performance Comparison
================================================================================

Hardware: Intel-i7-12700K (i7_12700k)
Vendor:   Intel
Type:     CPU

Memory Bandwidth
--------------------------------------------------------------------------------
Theoretical: 75.0 GB/s
Measured:    68.2 GB/s (90.9% efficiency)

Compute Performance by Precision
--------------------------------------------------------------------------------
Precision        Theoretical        Measured   Efficiency
--------------------------------------------------------------------------------
fp64            360.0 GFLOPS     54.8 GFLOPS        15.2%
fp32            720.0 GFLOPS     21.3 GFLOPS         3.0%
int64            360.0 GIOPS       4.9 GIOPS         1.4%
int32            360.0 GIOPS       4.3 GIOPS         1.2%
int16            720.0 GIOPS       4.3 GIOPS         0.6%
int8            1440.0 GIOPS       5.4 GIOPS         0.4%

================================================================================
Summary
================================================================================
FP32 Efficiency: 3.0%
  ⚠ Low performance (<20% of theoretical)
    Consider:
    - Using optimized BLAS library (MKL, OpenBLAS)
    - Enabling compiler optimizations
    - Checking thermal throttling
```

## Complete Workflow Example

### 1. Auto-Detect and Calibrate

```bash
# Let the system detect your hardware and run calibration
./cli/calibrate_hardware.py --quick --operations blas
```

Output:
```
================================================================================
Auto-Detecting Hardware
================================================================================
CPU:      12th Gen Intel(R) Core(TM) i7-12700K
Vendor:   Intel
Cores:    12 cores, 20 threads
          (8P + 4E cores)

✓ Matched to database: i7_12700k
  Confidence: 100%
  Device: CPU

================================================================================
EXECUTION DEVICE
================================================================================
  Requested device: CPU
  Actual device:    CPU
  Framework:        NUMPY

[... calibration runs ...]

================================================================================
Calibration Complete!
================================================================================

Calibration file: src/graphs/hardware/calibration/profiles/i7_12700k_numpy.json

Next steps:
  1. Review the calibration results above
  2. Use this calibration in your analysis:
     ./cli/analyze_comprehensive.py --model resnet18 \
         --hardware i7_12700k \
         --calibration src/graphs/hardware/calibration/profiles/i7_12700k_numpy.json

  3. Or export calibration to database:
     python scripts/hardware_db/update_hardware.py --id i7_12700k \
         --field calibration_file --value src/graphs/hardware/calibration/profiles/i7_12700k_numpy.json
```

### 2. Compare Results vs Theoretical

```bash
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/i7_12700k_numpy.json \
    --verbose
```

### 3. Add New Hardware (if auto-detection failed)

```bash
# Detect and export
python scripts/hardware_db/detect_hardware.py --export my_system.json

# Add to database
python scripts/hardware_db/add_hardware.py --from-detection my_system.json

# Calibrate
./cli/calibrate_hardware.py --id my_system
```

## Migration from --preset to Database

### Old Way (Deprecated)

```bash
./cli/calibrate_hardware.py --preset i7-12700k
```

Warning shown:
```
⚠ WARNING: --preset is deprecated
  Use --id <hardware_id> or auto-detection instead
  Falling back to legacy preset mode...
```

### New Way

```bash
# Auto-detect (recommended)
./cli/calibrate_hardware.py

# Or specify from database
./cli/calibrate_hardware.py --id i7_12700k
```

## Backward Compatibility

**Preserved:**
- `--preset` still works (with deprecation warning)
- All PRESETS converted to database format during migration
- Existing calibration files work unchanged
- Output file format unchanged

**Migration Path:**
1. Existing presets migrated to database (Phase 1)
2. `--preset` marked deprecated but functional (Phase 4)
3. Future: Remove `--preset` entirely (Phase 5)

## Technical Implementation Details

### Hardware Spec to Calibration Parameters

```python
# Hardware spec from database
hardware_spec = HardwareSpec(
    id="i7_12700k",
    model="Intel-i7-12700K",
    device_type="cpu",
    theoretical_peaks={
        'fp64': 360.0,
        'fp32': 720.0,
        'int64': 360.0,
        'int32': 360.0,
        'int16': 720.0,
        'int8': 1440.0,
    },
    peak_bandwidth_gbps=75.0,
    ...
)

# Extracted for calibration
hardware_name = hardware_spec.model  # "Intel-i7-12700K"
device = 'cpu'  # from hardware_spec.device_type
theoretical_peaks = hardware_spec.theoretical_peaks  # dict
peak_bandwidth = hardware_spec.peak_bandwidth_gbps  # 75.0
```

### Auto-Detection Logic

```python
# Load database
db = get_database()
db.load_all()

# Auto-detect
hardware_spec, confidence = auto_detect_hardware(db)

if not hardware_spec:
    print("Auto-detection failed. Please use --id to specify hardware.")
    return 1

if confidence < 0.5:
    print(f"⚠ WARNING: Low confidence match ({confidence*100:.0f}%)")
    response = input("Continue with this hardware? (yes/no): ")
    if response not in ['yes', 'y']:
        return 1
```

### Comparison Algorithm

```python
# Compare each precision
for prec in ['fp64', 'fp32', 'fp16', 'int64', 'int32', 'int16', 'int8']:
    theoretical = hw_spec.theoretical_peaks.get(prec, 0.0)
    measured = measured_peaks.get(prec, 0.0)

    if theoretical > 0 and measured > 0:
        efficiency = (measured / theoretical) * 100
        print(f"{prec}: {efficiency:.1f}% efficiency")
```

## Files Modified/Created

### Modified Files
- `cli/calibrate_hardware.py` (+150 lines, modified main flow)
  - Added `auto_detect_hardware()` function
  - Made `--preset` optional with deprecation
  - Added `--id` flag for database lookup
  - Integrated database loading and matching

### Created Files
- `scripts/hardware_db/compare_calibration.py` (273 lines)
  - Theoretical vs calibrated comparison
  - Auto-identification from calibration name
  - Detailed and summary views
  - Performance recommendations

### Documentation
- `docs/PHASE4_CALIBRATION_INTEGRATION.md` (this file)

## Integration with Previous Phases

**Phase 1 (Database Foundation):**
- Uses database schema and manager
- Loads specs from JSON files

**Phase 2 (Hardware Detection):**
- Uses `HardwareDetector` for auto-detection
- Uses cross-platform detection (psutil, py-cpuinfo)
- Matches detected hardware to database patterns

**Phase 3 (Management Tools):**
- Suggests using `add_hardware.py` if detection fails
- Can export calibration results to database
- Uses improved detection patterns

## Known Limitations

1. **Auto-Detection Accuracy:**
   - Depends on detection pattern quality
   - May fail for new/unlisted hardware
   - Solution: Use `--id` flag or add to database

2. **Calibration Comparison:**
   - Assumes calibration file naming matches database
   - May need manual `--id` for custom filenames
   - Solution: Use explicit `--id` flag

3. **Legacy Preset Support:**
   - Still functional but deprecated
   - Will be removed in future release
   - Solution: Migrate to `--id` or auto-detection

## Performance Recommendations

The comparison tool provides actionable recommendations:

- **≥80% efficiency:** Excellent performance
- **≥50% efficiency:** Good performance
- **≥20% efficiency:** Moderate performance
- **<20% efficiency:** Low performance, suggests:
  - Using optimized BLAS library (MKL, OpenBLAS)
  - Enabling compiler optimizations
  - Checking thermal throttling

## Future Enhancements (Phase 5)

Planned improvements:
1. Export calibration results directly to database
2. Historical calibration tracking
3. Multi-run calibration averaging
4. Thermal/power profiling integration
5. Complete removal of `--preset` flag
6. Web-based visualization of comparison results

## Conclusion

Phase 4 successfully integrates the hardware database with calibration, providing a seamless workflow from detection to calibration to performance analysis. The automatic detection eliminates manual preset selection errors, while maintaining backward compatibility with existing workflows.

Key Benefits:
- ✅ Auto-detection replaces manual preset selection
- ✅ Database-driven calibration (no hardcoded specs)
- ✅ Comparison tool for theoretical vs measured performance
- ✅ Backward compatible with existing workflows
- ✅ Clear migration path from presets to database

The system is now production-ready for hardware calibration with automatic hardware identification and performance validation.
