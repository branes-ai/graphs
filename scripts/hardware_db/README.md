# Hardware Database Management Scripts

This directory contains management tools for the hardware database. These scripts provide detection, querying, validation, and calibration integration for hardware specifications.

## Quick Start

### Auto-Detect and Calibrate Your Hardware

The fastest way to get started is to auto-detect your hardware and run calibration:

```bash
# Auto-detect hardware and calibrate (default mode)
./cli/calibrate_hardware.py --quick

# Compare calibrated results vs theoretical performance
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/<your_hardware>_numpy.json
```

## Detection Tools

### `detect_hardware.py`

Auto-detect current hardware and match to database.

```bash
# Basic detection
python scripts/hardware_db/detect_hardware.py

# Verbose output with all details
python scripts/hardware_db/detect_hardware.py --verbose

# Export detection results to JSON
python scripts/hardware_db/detect_hardware.py --export detection.json
```

**Cross-Platform Support:**
- Linux (x86_64, aarch64): Full support
- Windows (x86_64): Full support via psutil/py-cpuinfo
- macOS (x86_64, arm64): Full support via psutil/py-cpuinfo

**Dependencies:**
```bash
pip install psutil py-cpuinfo
```

## Database Management

### `list_hardware.py`

List all hardware in the database.

```bash
# List all hardware
python scripts/hardware_db/list_hardware.py

# Show database statistics
python scripts/hardware_db/list_hardware.py --stats

# Filter by vendor
python scripts/hardware_db/list_hardware.py --vendor Intel

# Filter by type and platform
python scripts/hardware_db/list_hardware.py --device-type gpu --platform x86_64
```

### `query_hardware.py`

Query specific hardware by ID or search criteria.

```bash
# Get specific hardware by ID
python scripts/hardware_db/query_hardware.py --id i7_12700k

# Search by vendor
python scripts/hardware_db/query_hardware.py --vendor Intel

# Show detailed information
python scripts/hardware_db/query_hardware.py --id h100_sxm5 --detail

# Export to JSON
python scripts/hardware_db/query_hardware.py --id h100_sxm5 --export h100.json
```

### `validate_database.py`

Validate hardware specs against schema.

```bash
# Validate all specs
python scripts/hardware_db/validate_database.py

# Strict mode (warnings as errors)
python scripts/hardware_db/validate_database.py --strict
```

## Adding and Updating Hardware

### `add_hardware.py`

Add new hardware to the database using interactive wizard or JSON file.

```bash
# Interactive wizard (recommended for first-time users)
python scripts/hardware_db/add_hardware.py

# From JSON file
python scripts/hardware_db/add_hardware.py --from-file hardware.json

# From detection results
python scripts/hardware_db/detect_hardware.py --export detection.json
python scripts/hardware_db/add_hardware.py --from-detection detection.json
```

**Workflow for New Hardware:**
1. Detect hardware: `python scripts/hardware_db/detect_hardware.py --export my_hw.json`
2. Add to database: `python scripts/hardware_db/add_hardware.py --from-detection my_hw.json`
3. Calibrate: `./cli/calibrate_hardware.py --id my_hw`
4. Compare: `python scripts/hardware_db/compare_calibration.py --calibration profiles/my_hw_numpy.json`

### `update_hardware.py`

Update existing hardware specifications.

```bash
# Interactive update (prompts for each field)
python scripts/hardware_db/update_hardware.py --id i7_12700k --interactive

# Update single field
python scripts/hardware_db/update_hardware.py --id h100_sxm5 \
  --field architecture --value "Hopper"

# Update detection patterns (comma-separated)
python scripts/hardware_db/update_hardware.py --id h100_sxm5 \
  --field detection_patterns --value "H100,NVIDIA H100,H100.*80GB"

# Update theoretical peaks (JSON)
python scripts/hardware_db/update_hardware.py --id h100_sxm5 \
  --field theoretical_peaks --value '{"fp32": 67000.0, "fp16": 134000.0}'
```

### `delete_hardware.py`

Remove hardware from the database.

```bash
# Delete with confirmation
python scripts/hardware_db/delete_hardware.py --id old_hardware

# Skip confirmation (for scripts)
python scripts/hardware_db/delete_hardware.py --id old_hardware --yes
```

## Calibration Integration (Phase 4)

### `compare_calibration.py`

Compare measured calibration results against theoretical hardware specs.

```bash
# Auto-identify hardware from calibration filename
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/i7_12700k_numpy.json

# Specify hardware explicitly
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/h100_sxm5_pytorch.json \
    --id h100_sxm5

# Verbose mode with per-operation breakdown
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/i7_12700k_numpy.json \
    --verbose
```

**Output:**
- Memory bandwidth comparison (theoretical vs measured)
- Compute performance by precision (fp64, fp32, int32, etc.)
- Efficiency percentages
- Performance recommendations

**Example:**
```
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
int32            360.0 GIOPS       4.3 GIOPS         1.2%
```

### Complete Calibration Workflow

1. **Auto-detect and calibrate:**
   ```bash
   ./cli/calibrate_hardware.py --quick --operations blas
   ```

2. **Compare results:**
   ```bash
   python scripts/hardware_db/compare_calibration.py \
       --calibration src/graphs/hardware/calibration/profiles/<hardware>_numpy.json \
       --verbose
   ```

3. **Optional: Export to database:**
   ```bash
   python scripts/hardware_db/update_hardware.py --id <hardware_id> \
       --field calibration_file \
       --value src/graphs/hardware/calibration/profiles/<hardware>_numpy.json
   ```

## Advanced Tools

### `improve_patterns.py`

Automatically improve detection patterns for better hardware matching.

```bash
# Preview pattern improvements (dry run)
python scripts/hardware_db/improve_patterns.py --dry-run

# Apply improvements to all hardware
python scripts/hardware_db/improve_patterns.py

# Improve specific hardware
python scripts/hardware_db/improve_patterns.py --id h100_sxm5
```

**Pattern Generation Logic:**
- Intel CPUs: Generation prefix, Core tier, model number
- AMD CPUs: Ryzen tier, model number
- NVIDIA GPUs: Architecture, model, memory size
- Ampere CPUs: Altra family variants

### `migrate_presets.py`

Migrate legacy PRESETS from `cli/calibrate_hardware.py` to database format.

```bash
# Migrate all PRESETS
python scripts/hardware_db/migrate_presets.py

# Dry run to preview changes
python scripts/hardware_db/migrate_presets.py --dry-run
```

## Common Workflows

### New User Setup

```bash
# 1. Auto-detect your hardware
python scripts/hardware_db/detect_hardware.py --verbose

# 2. If detected, calibrate immediately
./cli/calibrate_hardware.py --quick

# 3. Compare results
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/<detected_hardware>_numpy.json
```

### Adding Unsupported Hardware

```bash
# 1. Detect and export
python scripts/hardware_db/detect_hardware.py --export my_system.json

# 2. Add to database (wizard will guide you)
python scripts/hardware_db/add_hardware.py --from-detection my_system.json

# 3. Calibrate with new database entry
./cli/calibrate_hardware.py --id my_system

# 4. Compare results
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/my_system_numpy.json
```

### Database Maintenance

```bash
# Validate all specs
python scripts/hardware_db/validate_database.py --strict

# Improve detection patterns
python scripts/hardware_db/improve_patterns.py

# Show database statistics
python scripts/hardware_db/list_hardware.py --stats

# Update hardware spec
python scripts/hardware_db/update_hardware.py --id <hardware_id> --interactive
```

## Migration from Legacy --preset

**Old Way (Deprecated):**
```bash
./cli/calibrate_hardware.py --preset i7-12700k
```

**New Way:**
```bash
# Auto-detect (recommended)
./cli/calibrate_hardware.py

# Or specify from database
./cli/calibrate_hardware.py --id i7_12700k
```

The `--preset` flag is deprecated and will be removed in a future release. All presets have been migrated to the database.

## Performance Recommendations

The comparison tool provides actionable recommendations based on efficiency:

- **≥80% efficiency:** Excellent performance
- **≥50% efficiency:** Good performance
- **≥20% efficiency:** Moderate performance
- **<20% efficiency:** Low performance, check:
  - BLAS library (MKL, OpenBLAS)
  - Compiler optimizations
  - Thermal throttling

## See Also

- `hardware_database/README.md` - Database structure and schema
- `docs/PHASE4_CALIBRATION_INTEGRATION.md` - Calibration integration details
- `cli/README.md` - Calibration CLI documentation
