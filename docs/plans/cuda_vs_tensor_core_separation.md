# Plan: Separating CUDA Core vs Tensor Core Performance

**Date**: 2025-11-26
**Status**: Phase 1 Complete, Phase 2 In Progress

## Phase 1 Status: COMPLETE

Phase 1 (Clock Measurement) has been implemented:
- `src/graphs/hardware/calibration/gpu_clock.py` - GPU clock query module
- Calibrator now captures GPU clock during CUDA calibration
- CalibrationSummary extended with `measured_clock_mhz` and `gpu_clock` fields
- export_calibration.py updated to extract and display clock data

## Precision Naming Convention: IMPLEMENTED

Added proper precision taxonomy to `src/graphs/hardware/database/schema.py`:
- **fp64/fp32/fp16/fp8/fp4**: IEEE 754 floating-point formats
- **tf32**: NVIDIA TensorFloat-32 (19-bit, NOT 32-bit!) - Tensor Cores only
- **bf16**: Google Brain Float16 (16-bit)
- **int64/int32/int16/int8/int4**: Integer formats

Key insight: NVIDIA's "FP32 Tensor Core" performance is actually TF32 (19-bit).

## Problem Statement

### Issue 1: CUDA Cores vs Tensor Cores Not Distinguished

Currently, `theoretical_peaks` and `calibration_summary.measured_peaks` conflate two fundamentally different compute units:

| Precision | Current theoretical_peaks | What it represents | Actual Jetson Orin Nano capability |
|-----------|---------------------------|--------------------|------------------------------------|
| fp32 | 640 GFLOPS | CUDA cores only | CUDA: 1331 GFLOPS @1.3GHz, TC: ~5300 GFLOPS |
| fp16 | 1280 GFLOPS | Unclear mix | CUDA: 2662 GFLOPS @1.3GHz, TC: ~10650 GFLOPS |

The measured FP32 (1199 GFLOPS) exceeds the CUDA-core theoretical (640 GFLOPS) because:
1. PyTorch matmul uses Tensor Cores for FP32 on Ampere (TF32 mode)
2. The theoretical was calculated at 625MHz, not 1.3GHz

### Issue 2: Clock Frequency Not Captured During Calibration

The calibration profiles don't record the actual GPU clock frequency during benchmarks:
- Jetson 7W mode: ~650 MHz
- Jetson 15W mode: ~900 MHz
- Jetson 25W mode: ~1.3 GHz

Without knowing the actual clock, we can't:
1. Calculate accurate efficiency (measured / theoretical at that clock)
2. Normalize results across power modes
3. Predict performance at different clock speeds

## Jetson Orin Nano GPU Specifications

From NVIDIA documentation:
- **SMs**: 8
- **CUDA Cores per SM**: 128 (total: 1024)
- **Tensor Cores per SM**: 4 (total: 32, Gen 3)
- **FP32 units per SM**: 64 (half of CUDA cores can do FP32)
- **Clock Range**: 306 MHz (min) to 1.3 GHz (max)

### Theoretical Peak Calculations

**CUDA Core Performance** (traditional FMA operations):
```
FP32: 1024 cores × 2 ops/core/clock × clock_ghz = 2048 × clock_ghz GFLOPS
      @625MHz:  1,280 GFLOPS
      @1.0GHz:  2,048 GFLOPS
      @1.3GHz:  2,662 GFLOPS

FP64: 64 cores × 2 ops/core/clock × clock_ghz = 128 × clock_ghz GFLOPS
      @1.3GHz:  166 GFLOPS

FP16: 1024 cores × 4 ops/core/clock × clock_ghz = 4096 × clock_ghz GFLOPS (packed)
      @1.3GHz:  5,325 GFLOPS
```

**Tensor Core Performance** (matrix operations, Gen 3):
```
FP16/BF16: 32 TCs × 256 ops/TC/clock × clock_ghz = 8192 × clock_ghz GFLOPS
           @1.3GHz:  10,650 GFLOPS

TF32: 32 TCs × 128 ops/TC/clock × clock_ghz = 4096 × clock_ghz GFLOPS
      @1.3GHz:  5,325 GFLOPS
      NOTE: This is what NVIDIA misleadingly calls "FP32 Tensor Core" performance.
            TF32 is actually a 19-bit format (1+8+10), not IEEE FP32 (32-bit).

INT8: 32 TCs × 512 ops/TC/clock × clock_ghz = 16384 × clock_ghz GIOPS
      @1.3GHz:  21,300 GIOPS
```

## Proposed Solution

### Part A: Extend theoretical_peaks Schema

Replace flat `theoretical_peaks` with structured `compute_unit_peaks`:

```json
{
  "compute_unit_peaks": {
    "cuda_cores": {
      "reference_clock_ghz": 1.3,
      "fp64": 166.0,
      "fp32": 2662.0,
      "fp16": 5325.0,
      "int32": 2662.0,
      "int16": 5325.0,
      "int8": 10650.0
    },
    "tensor_cores": {
      "reference_clock_ghz": 1.3,
      "generation": 3,
      "tf32": 5325.0,
      "fp16": 10650.0,
      "bf16": 10650.0,
      "int8": 21300.0,
      "int4": 42600.0
    }
  },
  "theoretical_peaks": {
    "// DEPRECATED": "Use compute_unit_peaks instead",
    "fp32": 2662.0,
    "fp16": 10650.0,
    "...": "backward compatibility, use max of cuda/tensor"
  }
}
```

### Part B: Extend Calibration to Capture Compute Unit + Clock

#### B.1: Add Clock Measurement to Calibration Metadata

```json
{
  "metadata": {
    "hardware_name": "NVIDIA-Jetson-Orin-Nano-GPU",
    "calibration_date": "2025-11-16T10:04:26.791492",
    "...": "existing fields",

    "gpu_clock": {
      "measured_sm_clock_mhz": 1300,
      "measured_mem_clock_mhz": 3200,
      "power_mode": "MAXN",
      "nvpmodel_mode": 0,
      "measurement_method": "nvidia-smi"
    }
  }
}
```

#### B.2: Track Compute Unit Used Per Operation

```json
{
  "operation_profiles": {
    "matmul_device=cuda_matrix_size=4096_large": {
      "operation_type": "matmul",
      "compute_unit_used": "tensor_cores",
      "precision_results": {
        "fp32": {
          "precision": "fp32",
          "compute_mode": "tf32",
          "compute_unit": "tensor_cores",
          "measured_gops": 1199.0,
          "theoretical_at_clock": 5325.0,
          "efficiency": 0.225,
          "sm_clock_mhz": 1300,
          "...": "existing fields"
        },
        "fp16": {
          "precision": "fp16",
          "compute_unit": "tensor_cores",
          "measured_gops": 7324.9,
          "theoretical_at_clock": 10650.0,
          "efficiency": 0.688,
          "sm_clock_mhz": 1300
        }
      }
    }
  }
}
```

### Part C: Extend CalibrationSummary Schema

```python
@dataclass
class ComputeUnitMeasurement:
    """Performance measurement for a specific compute unit type."""
    compute_unit: str  # "cuda_cores", "tensor_cores"
    measured_peaks: Dict[str, float]
    efficiency: Dict[str, float]
    reference_clock_mhz: int
    measured_clock_mhz: int

@dataclass
class CalibrationSummary:
    # Existing fields...

    # NEW: Per-compute-unit measurements
    by_compute_unit: Optional[Dict[str, ComputeUnitMeasurement]] = None
    """
    Example:
    {
        "cuda_cores": {
            "measured_peaks": {"fp32": 2500.0, "fp64": 150.0},
            "efficiency": {"fp32": 0.94, "fp64": 0.90},
            "reference_clock_mhz": 1300,
            "measured_clock_mhz": 1287
        },
        "tensor_cores": {
            "measured_peaks": {"tf32": 4800.0, "fp16": 9500.0, "int8": 19000.0},
            "efficiency": {"tf32": 0.90, "fp16": 0.89, "int8": 0.89},
            "reference_clock_mhz": 1300,
            "measured_clock_mhz": 1287
        }
    }
    """

    # NEW: Clock during calibration
    measured_clock_mhz: Optional[int] = None
    reference_clock_mhz: Optional[int] = None
```

### Part D: Calibration Tool Enhancements

#### D.1: Add GPU Clock Query

```python
def get_gpu_clock_info() -> Dict:
    """Query current GPU clock frequencies."""
    # For NVIDIA GPUs via nvidia-smi or pynvml
    # For Jetson via tegrastats or jtop

    # Returns:
    # {
    #     "sm_clock_mhz": 1300,
    #     "mem_clock_mhz": 3200,
    #     "power_mode": "MAXN",
    #     "temperature_c": 45
    # }
```

#### D.2: Add Compute Unit Selection Tests

```python
def benchmark_cuda_cores_fp32(size: int) -> BenchmarkResult:
    """FP32 benchmark that avoids Tensor Cores."""
    # Use operations that don't trigger Tensor Core path:
    # - Element-wise operations (add, mul)
    # - Small matrix operations
    # - torch.backends.cuda.matmul.allow_tf32 = False

def benchmark_tensor_cores_fp32(size: int) -> BenchmarkResult:
    """FP32 benchmark using Tensor Cores (TF32 mode)."""
    # Large matmul with TF32 enabled
    # torch.backends.cuda.matmul.allow_tf32 = True
```

## Implementation Phases

### Phase 1: Clock Measurement (2-3 hours)
1. Add `get_gpu_clock_info()` to calibration tool
2. Store clock in calibration profile metadata
3. Test on Jetson at 7W, 15W, 25W modes
4. Update export_calibration.py to include clock data

### Phase 2: Schema Extension (3-4 hours)
1. Add `ComputeUnitPeaks` dataclass to schema.py
2. Add `compute_unit_peaks` field to HardwareSpec
3. Update Jetson hardware_database JSON with correct values
4. Maintain backward compatibility with `theoretical_peaks`

### Phase 3: Calibration Enhancement (4-6 hours)
1. Add CUDA-core-only benchmarks (disable TF32)
2. Add Tensor Core explicit benchmarks
3. Update calibration profile schema
4. Update export_calibration.py to extract per-unit data

### Phase 4: Reporting Updates (2-3 hours)
1. Update summarize_database.py to show per-unit comparison
2. Add `--compare cuda-vs-tensor` mode
3. Update detailed view to show clock and compute unit breakdown

## Correct Jetson Orin Nano Values

After implementation, the database should show:

```
NVIDIA Jetson Orin Nano (GPU) @ 1.3 GHz

CUDA CORE PERFORMANCE (Theoretical)
  fp64:    166 GFLOPS
  fp32:  2,662 GFLOPS
  fp16:  5,325 GFLOPS
  int32: 2,662 GIOPS
  int8: 10,650 GIOPS

TENSOR CORE PERFORMANCE (Theoretical)
  tf32:       5,325 GFLOPS (19-bit, what NVIDIA calls "FP32 Tensor")
  fp16:      10,650 GFLOPS
  bf16:      10,650 GFLOPS
  int8:      21,300 GIOPS
  int4:      42,600 GIOPS

CALIBRATED @ 25W (1.3 GHz, PyTorch)
  Compute Unit    Prec    Theoretical  Measured  Efficiency
  tensor_cores    tf32       5,325      4,800      90.1%
  tensor_cores    fp16      10,650      9,500      89.2%
  cuda_cores      fp32       2,662      2,500      93.9%
  cuda_cores      fp64         166        150      90.4%

  Memory BW: 57.8 / 68.0 GB/s (85.0%)
```

## Questions for Discussion

1. **Backward compatibility**: Keep flat `theoretical_peaks` as "best available" (max of cuda/tensor)?

2. ~~**Precision naming**: Use `fp32_tf32` for Tensor Core FP32, or just note compute_unit separately?~~
   **RESOLVED**: Use `tf32` for Tensor Core "FP32" operations. TF32 is a distinct 19-bit format.

3. **Mixed workloads**: How to handle workloads that use both CUDA and Tensor cores?

4. **Clock normalization**: Should we normalize all measurements to a reference clock for comparison?

5. **Power mode naming**: Use wattage (25W) or NVIDIA mode names (MAXN, 15W, 10W)?
