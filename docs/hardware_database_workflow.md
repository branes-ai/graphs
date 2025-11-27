# Hardware Database Workflow

This document describes the workflow for creating, calibrating, and managing hardware database entries in the graphs project.

## Overview

The hardware database replaces hardcoded hardware presets with a JSON-based system that supports:
- Auto-detection of CPU and GPU hardware
- **Board-level detection** for embedded/SoC devices (Jetson, Raspberry Pi, Qualcomm)
- Pattern matching to identify hardware
- Theoretical performance specifications from datasheets
- Empirical calibration data from benchmarks
- Multiple power modes (for edge devices like Jetson)

## Board Detection (Embedded Devices)

For embedded devices like NVIDIA Jetson, the CPU and GPU names from system detection are generic (e.g., "ARMv8 Processor", "Orin (nvgpu)"). The hardware database uses **board-level detection** to identify these devices:

1. Reads `/proc/device-tree/model` for the board name
2. Reads `/etc/nv_tegra_release` for NVIDIA Tegra info
3. Uses composite signals (CUDA capability, core count) for matching
4. Maps the board to its component CPU and GPU specs

This allows automatic identification of Jetson Orin Nano, Jetson AGX Orin, and similar devices.

## Directory Structure

```
hardware_database/
├── cpu/
│   ├── intel/
│   │   └── intel_12th_gen_intelr_coretm_i7_12700k.json
│   ├── amd/
│   │   └── amd_ryzen_7_2700x_eight_core_processor.json
│   ├── ampere_computing/
│   │   └── ampere_altra_max.json
│   └── nvidia/
│       ├── jetson_orin_nano_cpu.json
│       └── jetson_orin_agx_cpu.json
├── gpu/
│   └── nvidia/
│       ├── h100_sxm5.json
│       ├── nvidia_geforce_gtx_1070.json
│       ├── jetson_orin_nano_gpu.json
│       └── jetson_orin_agx_gpu.json
├── boards/                              # Board/SoC definitions (NEW)
│   └── nvidia/
│       ├── jetson_orin_nano.json        # Maps to CPU + GPU components
│       └── jetson_orin_agx.json
└── schema.json
```

## Workflow Steps

### 1. Detect Hardware

First, detect what hardware is available on your system:

```bash
python scripts/hardware_db/detect_hardware.py
```

This shows:
- CPU model, vendor, cores, threads, cache hierarchy
- GPU model, memory, compute capability (if NVIDIA GPU present)
- Matches against existing database entries

### 2. Add New Hardware

#### Option A: Auto-detect and Add (Recommended)

For hardware not yet in the database:

```bash
# Interactive mode - prompts for bandwidth and performance
python scripts/hardware_db/auto_detect_and_add.py

# Write to file for manual review (recommended)
python scripts/hardware_db/auto_detect_and_add.py \
    --bandwidth 51.2 \
    --fp32-gflops 115.2 \
    -o my_cpu.json

# Include GPU detection
python scripts/hardware_db/auto_detect_and_add.py --detect-gpus -o .

# With calibration benchmarks (measures actual performance)
python scripts/hardware_db/auto_detect_and_add.py --with-calibration -o my_cpu.json
```

#### Option B: Add from Existing JSON

If you have a pre-created JSON specification:

```bash
python scripts/hardware_db/add_hardware.py --from-file my_hardware.json
```

#### Option C: Manual Creation

Create a JSON file following the schema in `hardware_database/schema.json`. Key fields:

```json
{
  "id": "vendor_model_name",
  "system": {
    "vendor": "Intel",
    "model": "Core i7-12700K",
    "architecture": "Alder Lake",
    "device_type": "cpu",
    "platform": "x86_64"
  },
  "core_info": {
    "total_cores": 12,
    "threads_per_core": 1,
    "p_cores": 8,
    "e_cores": 4
  },
  "memory_subsystem": {
    "peak_bandwidth_gbps": 89.6
  },
  "theoretical_peaks": {
    "fp32": 720.0,
    "fp64": 360.0
  },
  "detection_patterns": [
    "12th Gen Intel.*i7-12700K",
    "i7-12700K"
  ],
  "mapper": {
    "mapper_class": "CPUMapper"
  }
}
```

### 3. Run Calibration

Calibration measures actual hardware performance through benchmarks:

```bash
# Auto-detect hardware and calibrate
./cli/calibrate_hardware.py

# Calibrate specific hardware from database
./cli/calibrate_hardware.py --id intel_12th_gen_intelr_coretm_i7_12700k

# Quick calibration (fewer sizes/trials)
./cli/calibrate_hardware.py --quick

# Specific benchmarks only
./cli/calibrate_hardware.py --operations blas      # BLAS levels 1-3
./cli/calibrate_hardware.py --operations stream    # Memory bandwidth
./cli/calibrate_hardware.py --operations blas,stream  # Both (default)

# Custom output location
./cli/calibrate_hardware.py --output profiles/my_hardware.json
```

Calibration output is saved to:
```
src/graphs/hardware/calibration/profiles/<hardware_id>_<framework>.json
```

### 4. Export Calibration to Database

After calibration, export results back to the hardware database:

```bash
# Export single calibration file
python scripts/hardware_db/export_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/intel_12th_gen_intelr_coretm_i7_12700k_numpy.json \
    --hardware-id intel_12th_gen_intelr_coretm_i7_12700k

# Export with power mode (for edge devices)
python scripts/hardware_db/export_calibration.py \
    --calibration profiles/jetson_orin_nano_gpu_pytorch.json \
    --hardware-id jetson_orin_nano_gpu \
    --power-mode 25W

# Export all calibrations from a directory
python scripts/hardware_db/export_calibration.py \
    --calibration-dir src/graphs/hardware/calibration/profiles/nvidia/jetson_orin_nano

# Dry run (preview without writing)
python scripts/hardware_db/export_calibration.py \
    --calibration <path> --dry-run
```

### 5. Compare Calibration vs Theoretical

Analyze how measured performance compares to datasheet specifications:

```bash
python scripts/hardware_db/compare_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/intel_12th_gen_intelr_coretm_i7_12700k_numpy.json \
    --id intel_12th_gen_intelr_coretm_i7_12700k \
    --verbose
```

This shows efficiency percentages (measured / theoretical) for each precision and operation type.

## Management Commands

### List Hardware

```bash
# List all hardware
python scripts/hardware_db/list_hardware.py

# Filter by type
python scripts/hardware_db/list_hardware.py --device-type gpu
```

### Query Hardware

```bash
# Query by ID
python scripts/hardware_db/query_hardware.py --id intel_12th_gen_intelr_coretm_i7_12700k --detail

# Filter by vendor
python scripts/hardware_db/query_hardware.py --vendor NVIDIA

# Filter by platform
python scripts/hardware_db/query_hardware.py --platform aarch64

# Export to JSON
python scripts/hardware_db/query_hardware.py --vendor Intel --export intel_hardware.json
```

### Update Hardware

```bash
# Update specific field
python scripts/hardware_db/update_hardware.py \
    --id intel_12th_gen_intelr_coretm_i7_12700k \
    --field theoretical_peaks.fp32 \
    --value 750.0

# Interactive mode
python scripts/hardware_db/update_hardware.py \
    --id intel_12th_gen_intelr_coretm_i7_12700k \
    --interactive
```

### Delete Hardware

```bash
# With confirmation prompt
python scripts/hardware_db/delete_hardware.py --id old_hardware_id

# Skip confirmation
python scripts/hardware_db/delete_hardware.py --id old_hardware_id --yes
```

### Validate Database

```bash
python scripts/hardware_db/validate_database.py
```

Checks:
- JSON schema compliance
- Required fields present
- Detection patterns validity
- Theoretical peaks consistency

### Improve Detection Patterns

```bash
# Show suggestions for all hardware
python scripts/hardware_db/improve_patterns.py --dry-run

# Apply improvements for specific hardware
python scripts/hardware_db/improve_patterns.py --id intel_12th_gen_intelr_coretm_i7_12700k
```

### Database Summary

```bash
python scripts/hardware_db/summarize_database.py
```

Shows:
- Hardware count by device type, vendor, platform
- Performance summary tables (GFLOPS, bandwidth)
- Missing calibration data

## Typical Workflows

### Adding a New Desktop CPU

```bash
# 1. Detect and create spec file
python scripts/hardware_db/auto_detect_and_add.py \
    --bandwidth 89.6 \
    --fp32-gflops 720 \
    -o my_cpu.json

# 2. Review and edit my_cpu.json if needed

# 3. Add to database
python scripts/hardware_db/add_hardware.py --from-file my_cpu.json

# 4. Run calibration
./cli/calibrate_hardware.py --id <new_hardware_id>

# 5. Export calibration results
python scripts/hardware_db/export_calibration.py \
    --calibration src/graphs/hardware/calibration/profiles/<id>_numpy.json
```

### Adding a Jetson Device (Multiple Power Modes)

```bash
# 1. On the Jetson device, detect hardware
python scripts/hardware_db/auto_detect_and_add.py --detect-gpus -o .

# 2. Add both CPU and GPU specs to database
python scripts/hardware_db/add_hardware.py --from-file jetson_orin_nano_cpu.json
python scripts/hardware_db/add_hardware.py --from-file jetson_orin_nano_gpu.json

# 3. Calibrate at each power mode
sudo nvpmodel -m 0  # 25W mode
./cli/calibrate_hardware.py --id jetson_orin_nano_gpu \
    --output profiles/jetson_orin_nano_gpu_25W.json

sudo nvpmodel -m 1  # 15W mode
./cli/calibrate_hardware.py --id jetson_orin_nano_gpu \
    --output profiles/jetson_orin_nano_gpu_15W.json

# 4. Export each calibration with power mode
python scripts/hardware_db/export_calibration.py \
    --calibration profiles/jetson_orin_nano_gpu_25W.json \
    --power-mode 25W

python scripts/hardware_db/export_calibration.py \
    --calibration profiles/jetson_orin_nano_gpu_15W.json \
    --power-mode 15W
```

### Adding a New Embedded Board (Qualcomm, Stillwater, etc.)

For new embedded devices where CPU/GPU names are generic:

```bash
# 1. On the device, check what detection signals are available
cat /proc/device-tree/model           # Board name
cat /proc/device-tree/compatible      # Compatible strings
cat /etc/nv_tegra_release             # NVIDIA Tegra info (if applicable)

# 2. Create CPU and GPU hardware specs
python scripts/hardware_db/auto_detect_and_add.py --detect-gpus -o .

# 3. Create a board definition file
# hardware_database/boards/<vendor>/<board_name>.json
```

Board definition example:

```json
{
  "id": "my_board",
  "type": "board",
  "vendor": "Qualcomm",
  "model": "Snapdragon Dev Kit",
  "family": "Snapdragon",
  "soc": "SM8550",
  "platform": "aarch64",

  "detection": {
    "device_tree_model": ["Qualcomm Snapdragon Dev Kit"],
    "compatible_strings": ["qcom,sm8550"],
    "cuda_capability": null,
    "cpu_cores": [8],
    "gpu_model_patterns": ["Adreno.*740"],
    "cpu_model_patterns": ["Cortex-A", "Kryo"]
  },

  "components": {
    "cpu": "snapdragon_sm8550_cpu",
    "gpu": "snapdragon_sm8550_gpu"
  }
}
```

The board definition links detection signals to the component hardware specs.

### Updating After Hardware Changes

```bash
# 1. Re-run calibration
./cli/calibrate_hardware.py --id <hardware_id>

# 2. Compare against previous/theoretical
python scripts/hardware_db/compare_calibration.py \
    --calibration <new_calibration.json> \
    --verbose

# 3. Export updated calibration
python scripts/hardware_db/export_calibration.py \
    --calibration <new_calibration.json>
```

## Using Hardware in Analysis

Once hardware is in the database and calibrated:

```bash
# Comprehensive model analysis
./cli/analyze_comprehensive.py --model resnet18 --hardware intel_12th_gen_intelr_coretm_i7_12700k

# With specific calibration file
./cli/analyze_comprehensive.py --model resnet18 \
    --hardware intel_12th_gen_intelr_coretm_i7_12700k \
    --calibration src/graphs/hardware/calibration/profiles/intel_12th_gen_intelr_coretm_i7_12700k_numpy.json

# Batch size analysis
./cli/analyze_batch.py --model resnet50 \
    --hardware jetson_orin_nano_gpu \
    --batch-size 1 4 8 16
```

## Script Reference

| Script | Purpose |
|--------|---------|
| `detect_hardware.py` | Detect CPU/GPU and match to database |
| `auto_detect_and_add.py` | Auto-detect and create hardware spec |
| `add_hardware.py` | Add hardware from JSON file |
| `list_hardware.py` | List all hardware in database |
| `query_hardware.py` | Query/filter hardware |
| `update_hardware.py` | Update hardware fields |
| `delete_hardware.py` | Delete hardware from database |
| `validate_database.py` | Validate database integrity |
| `summarize_database.py` | Show database overview |
| `improve_patterns.py` | Suggest better detection patterns |
| `compare_calibration.py` | Compare measured vs theoretical |
| `export_calibration.py` | Export calibration to database |
