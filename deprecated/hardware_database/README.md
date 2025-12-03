# Hardware Database

This directory contains hardware specifications for CPUs, GPUs, and accelerators.
Each hardware device is described in a JSON file following the HardwareSpec schema.

## Directory Structure

```
hardware_database/
├── schema.json                  # JSON Schema for validation
├── README.md                    # This file
├── efficiency_calibration.md    # Calibration documentation
├── cpu/
│   ├── intel/
│   │   └── intel_12th_gen_intelr_coretm_i7_12700k.json
│   ├── amd/
│   │   ├── amd_amd_ryzen_7_2700x_eight_core_processor.json
│   │   └── amd_ryzen_7_2700x_eight_core_processor.json
│   ├── ampere/                  # (empty - placeholder)
│   ├── ampere_computing/
│   │   └── ampere_altra_max.json
│   └── nvidia/
│       ├── jetson_orin_agx_cpu.json
│       └── jetson_orin_nano_cpu.json
├── gpu/
│   ├── nvidia/
│   │   ├── h100_sxm5.json
│   │   ├── jetson_orin_agx_gpu.json
│   │   ├── jetson_orin_nano_gpu.json
│   │   └── nvidia_geforce_gtx_1070.json
│   └── amd/                     # (empty - placeholder)
└── accelerators/
    ├── google/                  # (empty - placeholder)
    └── xilinx/                  # (empty - placeholder)
```

## Hardware Spec Format

Each JSON file contains a complete hardware specification:

```json
{
  "id": "intel_i7_12700k",
  "vendor": "Intel",
  "model": "Core i7-12700K",
  "architecture": "Alder Lake",
  "device_type": "cpu",
  "platform": "x86_64",

  "detection_patterns": [
    "12th Gen Intel.*Core.*i7-12700K",
    "Intel.*i7-12700K"
  ],

  "cores": 12,
  "threads": 20,
  "base_frequency_ghz": 3.6,
  "boost_frequency_ghz": 5.0,

  "memory_type": "DDR5",
  "peak_bandwidth_gbps": 75.0,

  "isa_extensions": ["AVX2", "AVX512", "FMA3", "VNNI"],

  "theoretical_peaks": {
    "fp64": 360.0,
    "fp32": 720.0,
    "int64": 360.0,
    "int32": 360.0,
    "int16": 720.0,
    "int8": 1440.0
  },

  "mapper_class": "CPUMapper",
  "mapper_config": {
    "simd_width": 256
  }
}
```

See `schema.json` for the complete specification.

## Requirements

Hardware detection requires cross-platform Python libraries:

```bash
pip install psutil py-cpuinfo
```

Or install the full package:

```bash
pip install -e .
```

These libraries provide:
- **psutil**: CPU cores/threads/frequency (Windows/Linux/macOS)
- **py-cpuinfo**: CPU model, vendor, ISA extensions (Windows/Linux/macOS)

## Quick Start

### Auto-Detect and Calibrate (Recommended Workflow)

```bash
# 1. Auto-detect and calibrate your hardware
./cli/calibrate_hardware.py --quick

# 2. Compare calibrated vs theoretical performance
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/<your_hardware>_numpy.json
```

This workflow automatically detects your hardware, matches it to the database, and runs calibration. No manual preset selection required!

## Usage

### Detect Current Hardware

```bash
# Auto-detect hardware and match to database
python scripts/hardware_db/detect_hardware.py

# Show detailed detection info
python scripts/hardware_db/detect_hardware.py --verbose

# Export detection results
python scripts/hardware_db/detect_hardware.py --export detection.json
```

### List All Hardware

```bash
# List all hardware
python scripts/hardware_db/list_hardware.py

# Show database statistics
python scripts/hardware_db/list_hardware.py --stats

# Filter by vendor/type/platform
python scripts/hardware_db/list_hardware.py --vendor Intel
python scripts/hardware_db/list_hardware.py --device-type gpu --platform x86_64
```

### Query Hardware

```bash
# Get specific hardware by ID
python scripts/hardware_db/query_hardware.py --id intel_i7_12700k

# Search by vendor
python scripts/hardware_db/query_hardware.py --vendor Intel

# Search by device type and platform
python scripts/hardware_db/query_hardware.py --device-type gpu --platform x86_64

# Show detailed information
python scripts/hardware_db/query_hardware.py --id h100_sxm5 --detail

# Export to JSON
python scripts/hardware_db/query_hardware.py --id h100_sxm5 --export h100.json
```

### Validate Database

```bash
# Validate all specs
python scripts/hardware_db/validate_database.py

# Strict mode (warnings as errors)
python scripts/hardware_db/validate_database.py --strict
```

### Add New Hardware

```bash
# Interactive wizard (recommended for first-time users)
python scripts/hardware_db/add_hardware.py

# From JSON file
python scripts/hardware_db/add_hardware.py --from-file hardware.json

# From detection results
python scripts/hardware_db/detect_hardware.py --export detection.json
python scripts/hardware_db/add_hardware.py --from-detection detection.json
```

### Update Existing Hardware

```bash
# Interactive update (prompts for each field)
python scripts/hardware_db/update_hardware.py --id i7_12700k --interactive

# Update single field
python scripts/hardware_db/update_hardware.py --id h100_sxm5 \
  --field architecture --value "Hopper"

# Update detection patterns
python scripts/hardware_db/update_hardware.py --id h100_sxm5 \
  --field detection_patterns --value "H100,NVIDIA H100,H100.*80GB"
```

### Delete Hardware

```bash
# Delete with confirmation
python scripts/hardware_db/delete_hardware.py --id old_hardware

# Skip confirmation (for scripts)
python scripts/hardware_db/delete_hardware.py --id old_hardware --yes
```

### Improve Detection Patterns

```bash
# Preview pattern improvements (dry run)
python scripts/hardware_db/improve_patterns.py --dry-run

# Apply improvements to all hardware
python scripts/hardware_db/improve_patterns.py

# Improve specific hardware
python scripts/hardware_db/improve_patterns.py --id h100_sxm5
```

### Migrate Existing Presets

```bash
# Migrate PRESETS from cli/calibrate_hardware.py
python scripts/hardware_db/migrate_presets.py

# Dry run to preview
python scripts/hardware_db/migrate_presets.py --dry-run
```

### Calibration Integration (Phase 4)

The hardware database is now integrated with the calibration system. Instead of using the deprecated `--preset` flag, calibration automatically detects your hardware and uses database specs.

```bash
# Auto-detect and calibrate (default mode)
./cli/calibrate_hardware.py

# Quick calibration
./cli/calibrate_hardware.py --quick

# Calibrate specific hardware from database
./cli/calibrate_hardware.py --id i7_12700k
./cli/calibrate_hardware.py --id h100_sxm5

# Specific operations
./cli/calibrate_hardware.py --operations blas
./cli/calibrate_hardware.py --operations dot,gemm

# Legacy preset mode (deprecated, shows warning)
./cli/calibrate_hardware.py --preset i7-12700k
```

### Compare Calibration vs Theoretical Performance

After calibration, compare measured results against theoretical specs:

```bash
# Auto-identify hardware from calibration filename
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/i7_12700k_numpy.json

# Specify hardware explicitly
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/h100_sxm5_pytorch.json \
    --id h100_sxm5

# Verbose mode with operation breakdown
python scripts/hardware_db/compare_calibration.py \
    --calibration profiles/i7_12700k_numpy.json \
    --verbose
```

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
int32            360.0 GIOPS       4.3 GIOPS         1.2%

Summary
================================================================================
FP32 Efficiency: 3.0%
  ⚠ Low performance (<20% of theoretical)
    Consider:
    - Using optimized BLAS library (MKL, OpenBLAS)
    - Enabling compiler optimizations
    - Checking thermal throttling
```

## Auto-Detection

Hardware is auto-detected using the `HardwareDetector` class with cross-platform support:

### Platform Support
- **Linux**: Full support (x86_64, aarch64)
- **Windows**: Full support (x86_64) - GPU detection via nvidia-smi
- **macOS**: Full support (x86_64, arm64)

### Detection Methods

1. **CPU Detection**:
   - **Linux**: `/proc/cpuinfo` parsing, `nproc` for core count
   - **Windows**: `wmic cpu` command, `platform` module
   - **macOS**: `sysctl` commands for CPU info
   - Detects: model, vendor, architecture, cores, threads, frequency, ISA extensions

2. **GPU Detection** (NVIDIA focus):
   - **All platforms**: `nvidia-smi` (primary method)
   - **Fallback**: PyTorch CUDA (`torch.cuda.get_device_properties`)
   - Detects: model, vendor, memory, CUDA capability, driver version

3. **Pattern Matching**: Detected strings are matched against `detection_patterns` using regex

### Example Detection Patterns

```json
{
  "detection_patterns": [
    "12th Gen Intel.*Core.*i7-12700K",
    "Intel.*i7-12700K"
  ]
}
```

Pattern types:
- **Regex**: `"12th Gen Intel.*Core.*i7-12700K"` - Matches Intel i7-12700K with variations
- **Substring**: `"H100"` - Simple substring match
- **Exact**: `"NVIDIA H100 SXM5 80GB"` - Exact model string

### Using the Detector

```bash
# Auto-detect current hardware
python scripts/hardware_db/detect_hardware.py

# Verbose output with all details
python scripts/hardware_db/detect_hardware.py --verbose

# Export detection results
python scripts/hardware_db/detect_hardware.py --export detection.json
```

## Field Descriptions

### Required Fields

- **id**: Unique identifier (lowercase, underscores)
- **vendor**: Manufacturer name
- **model**: Full model name
- **device_type**: cpu, gpu, tpu, kpu, dpu, cgra
- **platform**: x86_64, aarch64, arm64
- **theoretical_peaks**: At least `fp32` must be specified
- **peak_bandwidth_gbps**: Memory bandwidth
- **mapper_class**: CPUMapper, GPUMapper, etc.

### Optional Fields

- **detection_patterns**: Regex for auto-detection
- **cores**, **threads**: CPU configuration
- **cuda_cores**, **tensor_cores**: GPU specifics
- **isa_extensions**: AVX2, NEON, SVE, etc.
- **cache sizes**: L1, L2, L3
- **power**: TDP, max power
- **metadata**: Release date, URLs, notes

## Contributing Hardware Specs

1. Create a JSON file following the schema
2. Place in appropriate directory (cpu/vendor/, gpu/vendor/, etc.)
3. Validate: `python scripts/hardware_db/validate_database.py`
4. Test detection: `python scripts/hardware_db/detect_hardware.py`
5. Submit pull request

## Data Sources

Hardware specs should be sourced from:
1. **Manufacturer datasheets** (preferred)
2. **Measured calibration results**
3. **Community benchmarks** (clearly marked)

Mark the `data_source` field accordingly.

## Platform Support

The database supports hardware across multiple platforms:
- **Linux** (x86_64, aarch64): Full support
- **Windows** (x86_64): Full support (Phase 2)
- **macOS** (x86_64, arm64): Full support (Phase 2)

Note: Detection patterns may vary by OS. Test on target platforms.

## Version Control

- All hardware specs are version controlled in git
- `last_updated` field tracks modification time
- Breaking schema changes require migration scripts

## See Also

- `src/graphs/hardware/database/schema.py` - Python schema definition
- `src/graphs/hardware/database/manager.py` - Database manager
- `scripts/hardware_db/` - Management tools
