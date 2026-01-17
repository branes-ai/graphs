# Session Log: Power Profile Comparison Tool

**Date**: 2025-11-28
**Focus**: Creating CLI tool for comparing power profiles across multiple calibrations

## Overview

This session focused on creating and refining a CLI utility (`cli/compare_power_profiles.py`) to compare throughput and efficiency across different power profiles for hardware with multiple calibrations (e.g., Jetson GPU with 7W, 15W, 25W, MAXN modes).

## Changes Made

### New CLI Tool: `cli/compare_power_profiles.py`

Created a comprehensive power profile comparison tool with the following features:

**Table Organization (Three Segments)**:
1. **Speed of Light**: Theoretical peak scaled by frequency ratio
   - Frequency (MHz)
   - Peak compute (GFLOPS/GOPS)
   - Peak bandwidth (GB/s)

2. **Measured**: BLAS workload performance
   - AXPY (Level 1) - throughput and bandwidth
   - GEMV (Level 2) - throughput and bandwidth
   - GEMM (Level 3) - throughput and bandwidth

3. **Efficiency**: Power metrics
   - Watts (with `~` prefix for estimated values)
   - GFLOPS/W (using GEMM as compute-bound reference)

**Per-Precision Tables**:
- FP32, TF32, BF16, FP16, INT8
- Each precision shows its own theoretical peak and measured results

**Data Extraction**:
- Reads from `precision_results` field in OperationCalibration
- Extracts both `measured_gops` and `achieved_bandwidth_gbps` per precision

**Power Estimation**:
- Parses wattage from power mode names (e.g., "7W", "15W", "25W")
- Estimates unknown modes (e.g., MAXN_SUPER) using frequency-proportional scaling

### Header Alignment Fix

Fixed alignment issues where segment headers didn't match column widths:

**Before**: Headers were misaligned with data columns
```
                  Speed of Light      |                     Measured                      |    Efficiency
Power Mode     Freq       Peak     BW |     AXPY     BW |     GEMV     BW |     GEMM     BW |  Watts   GFLOPS/W
```

**After**: Headers properly centered over their columns
```
---------------------------------------------------------------------------------------------------------------
                  Speed of Light      |                      Measured                       |    Efficiency
Power Mode     Freq       Peak     BW |     AXPY     BW |     GEMV     BW |     GEMM     BW |  Watts   GFLOPS/W
                MHz     GFLOPS   GB/s |   GFLOPS   GB/s |   GFLOPS   GB/s |   GFLOPS   GB/s |
---------------------------------------------------------------------------------------------------------------
```

**Solution**: Build column strings first, then use `len()` to calculate exact widths for centering segment labels.

## Key Data Structures

```python
@dataclass
class PowerProfileSummary:
    power_mode: str
    freq_mhz: int
    framework: str
    calibration_date: str
    peak_compute_gflops: float
    peak_bandwidth_gbps: float
    measured_bandwidth_gbps: float
    axpy_result: Optional[WorkloadResult] = None
    gemv_result: Optional[WorkloadResult] = None
    gemm_result: Optional[WorkloadResult] = None
    estimated_watts: Optional[float] = None
    # Per-precision best GFLOPS for each workload: {precision: {workload: (gflops, bandwidth)}}
    precision_workload_data: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
```

## Usage Examples

```bash
# Compare all power profiles for Jetson GPU
./cli/compare_power_profiles.py --id jetson_orin_nano_gpu

# List hardware with multiple calibrations
./cli/compare_power_profiles.py --list

# Show detailed per-operation breakdown
./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --detailed

# Export to CSV
./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --csv power_comparison.csv
```

## Sample Output

```
========================================================================================================================
Power Profile Comparison: NVIDIA Jetson Orin Nano (GPU)
========================================================================================================================
Framework: pytorch

Spec Peaks @ 918.0 MHz:
  FP32: 1,880 GFLOPS | FP16: 7,520 GFLOPS | BF16: 7,520 GFLOPS | TF32: 3,760 GFLOPS | INT8: 15,040 GOPS
  Memory Bandwidth: 68.0 GB/s

[BF16] Spec peak @ 918 MHz: 7,520 GFLOPS, 68 GB/s
---------------------------------------------------------------------------------------------------------------
                  Speed of Light      |                      Measured                       |    Efficiency
Power Mode     Freq       Peak     BW |     AXPY     BW |     GEMV     BW |     GEMM     BW |  Watts   GFLOPS/W
                MHz     GFLOPS   GB/s |   GFLOPS   GB/s |   GFLOPS   GB/s |   GFLOPS   GB/s |
---------------------------------------------------------------------------------------------------------------
7W              306       2507     68 |      4.9   14.7 |     18.0   18.0 |   1911.2    2.8 |    7.0      273.0
15W             612       5013     68 |      8.1   24.4 |     15.3   15.3 |   3128.4    4.6 |   15.0      208.6
25W             918       7520     68 |     11.9   35.6 |     21.6   21.6 |   3749.3    5.5 |   25.0      150.0
MAXN_SUPER     1020       8356     68 |     12.0   36.0 |     27.8   27.9 |   2772.9    4.1 | ~ 27.8       99.8

Scaling Analysis (Best GEMM):
------------------------------------------------------------
  7W -> 15W: 2.1x power, 1.5x performance, efficiency -27.8%
  7W -> 25W: 3.6x power, 1.9x performance, efficiency -48.1%
  7W -> MAXN_SUPER: 4.0x power, 1.5x performance, efficiency -61.3%
```

## Observations from Jetson Orin Nano Data

1. **Best GFLOPS/W at lowest power mode**: 7W mode achieves 273 GFLOPS/W (BF16 GEMM), much higher than 25W mode (150 GFLOPS/W)
2. **Diminishing returns at higher power**: Performance scales sublinearly with power
3. **MAXN_SUPER anomaly**: Lower GEMM throughput than 25W mode despite higher frequency - possible thermal throttling
4. **Memory-bound workloads (AXPY)**: Bandwidth scales roughly linearly with frequency

## Files Modified

- `cli/compare_power_profiles.py` - New CLI tool (700 lines)

## Related Files

- `src/graphs/hardware/calibration/schema.py` - HardwareCalibration and PrecisionResult schemas
- `hardware_registry/gpu/jetson_orin_nano_gpu/` - Calibration data directory
