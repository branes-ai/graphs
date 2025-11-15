# Multi-Precision Hardware Calibration - Implementation Plan

## Overview

Enhance the hardware calibration framework to benchmark operations across all precision types (INT8, INT16, INT32, FP8, FP16, FP32, FP64, BF16) and report **FAIL** when hardware doesn't support a particular precision.

**Goal**: Create comprehensive precision compatibility matrix for each hardware platform to enable accurate performance estimation per precision.

---

## Current State Analysis

### Existing Precision Types (from `resource_model.py:45-58`)
```python
class Precision(Enum):
    FP64 = "fp64"          # IEEE Double Precision
    FP32 = "fp32"          # IEEE Single Precision
    FP16 = "fp16"          # IEEE Half Precision
    FP8 = "fp8"            # Generic FP8
    FP8_E4M3 = "fp8_e4m3"  # NVIDIA H100 format
    FP8_E5M2 = "fp8_e5m2"  # Alternative format
    FP4 = "fp4"            # 4-bit floating point
    BF16 = "bf16"          # Brain Float16
    INT32 = "int32"        # 32-bit integer
    INT16 = "int16"        # 16-bit integer
    INT8 = "int8"          # 8-bit integer
    INT4 = "int4"          # 4-bit integer
```

### Current Benchmark Limitations
1. **matmul_bench.py**: Only tests `np.float32` (line 23, 37)
2. **No precision detection**: Doesn't check if hardware supports a precision
3. **No failure handling**: Would crash on unsupported dtype
4. **No per-precision storage**: Schema doesn't track precision-specific results

---

## Implementation Plan

### Phase 1: Schema Extensions

#### 1.1 Add PrecisionTestResult

```python
# schema.py - NEW

@dataclass
class PrecisionTestResult:
    """
    Result of testing a single precision on hardware.

    Captures whether the precision is supported and its performance.
    """
    precision: str  # Precision enum value

    # Support status
    supported: bool  # True if hardware can run this precision
    failure_reason: Optional[str] = None  # Why it failed (if supported=False)

    # Performance (only if supported=True)
    measured_gflops: Optional[float] = None
    efficiency: Optional[float] = None  # vs theoretical peak for this precision
    mean_latency_ms: Optional[float] = None

    # Compared to FP32 baseline
    speedup_vs_fp32: Optional[float] = None  # e.g., FP16 = 2.0× faster

    # Test configuration
    test_size: int  # Matrix size (for matmul)
    num_trials: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PrecisionTestResult':
        return cls(**data)
```

#### 1.2 Update OperationCalibration

```python
# schema.py - MODIFIED

@dataclass
class OperationCalibration:
    # ... existing fields ...

    # NEW: Per-precision results
    precision_results: Dict[str, PrecisionTestResult] = field(default_factory=dict)

    # DEPRECATED (use precision_results instead):
    # measured_gflops: float  # Keep for backward compatibility
```

#### 1.3 Add PrecisionCapabilityMatrix

```python
# schema.py - NEW

@dataclass
class PrecisionCapabilityMatrix:
    """
    Hardware precision support matrix.

    Summarizes which precisions are supported across all operations.
    """
    hardware_name: str

    # Per-precision support status
    supported_precisions: List[str]  # ["fp32", "fp16", "int8"]
    unsupported_precisions: List[str]  # ["fp64", "fp8_e4m3"]

    # Per-precision peak performance (if supported)
    peak_gflops_by_precision: Dict[str, float]

    # Speedup ratios (relative to FP32 baseline)
    speedup_vs_fp32: Dict[str, float]  # {"fp16": 2.0, "int8": 4.0}

    def to_dict(self) -> Dict:
        return asdict(self)
```

#### 1.4 Update HardwareCalibration

```python
# schema.py - MODIFIED

@dataclass
class HardwareCalibration:
    # ... existing fields ...

    # NEW: Precision capability summary
    precision_matrix: Optional[PrecisionCapabilityMatrix] = None
```

---

### Phase 2: Precision Detection Framework

#### 2.1 Create precision_detector.py

```python
# calibration/precision_detector.py - NEW FILE

"""
Detect hardware support for different numerical precisions.
"""

import numpy as np
import sys
from typing import List, Tuple
from ..resource_model import Precision


def detect_numpy_precision_support() -> List[Precision]:
    """
    Detect which precisions NumPy/hardware can actually execute.

    Returns:
        List of supported Precision enums
    """
    supported = []

    # Test each precision
    precision_tests = [
        (Precision.FP64, np.float64),
        (Precision.FP32, np.float32),
        (Precision.FP16, np.float16),
        (Precision.BF16, None),  # Requires special handling
        (Precision.INT32, np.int32),
        (Precision.INT16, np.int16),
        (Precision.INT8, np.int8),
    ]

    for precision, numpy_dtype in precision_tests:
        if numpy_dtype is None:
            # Special cases (BF16, FP8) require PyTorch/TensorFlow
            continue

        try:
            # Try to create and multiply small arrays
            A = np.random.rand(16, 16).astype(numpy_dtype)
            B = np.random.rand(16, 16).astype(numpy_dtype)
            C = A @ B

            # Check for NaN/Inf (indicates failure)
            if not np.all(np.isfinite(C)):
                continue

            supported.append(precision)

        except (TypeError, ValueError, RuntimeError):
            # Precision not supported
            continue

    return supported


def detect_pytorch_precision_support(device='cpu') -> List[Precision]:
    """
    Detect which precisions PyTorch supports on given device.

    Returns:
        List of supported Precision enums
    """
    try:
        import torch
    except ImportError:
        return []

    supported = []

    precision_tests = [
        (Precision.FP64, torch.float64),
        (Precision.FP32, torch.float32),
        (Precision.FP16, torch.float16),
        (Precision.BF16, torch.bfloat16),
        (Precision.INT32, torch.int32),
        (Precision.INT16, torch.int16),
        (Precision.INT8, torch.int8),
    ]

    # FP8 only on CUDA with compute capability >= 8.9 (H100)
    if device == 'cuda' and torch.cuda.is_available():
        cuda_cap = torch.cuda.get_device_capability()
        if cuda_cap[0] >= 8 and cuda_cap[1] >= 9:
            # H100 supports FP8
            if hasattr(torch, 'float8_e4m3fn'):
                precision_tests.append((Precision.FP8_E4M3, torch.float8_e4m3fn))
            if hasattr(torch, 'float8_e5m2'):
                precision_tests.append((Precision.FP8_E5M2, torch.float8_e5m2))

    for precision, torch_dtype in precision_tests:
        try:
            A = torch.randn(16, 16, dtype=torch_dtype, device=device)
            B = torch.randn(16, 16, dtype=torch_dtype, device=device)
            C = torch.matmul(A, B)

            # Check for NaN/Inf
            if not torch.all(torch.isfinite(C)):
                continue

            supported.append(precision)

        except (TypeError, RuntimeError):
            continue

    return supported


def get_precision_capabilities(device='cpu') -> Tuple[List[Precision], List[Precision]]:
    """
    Get comprehensive precision support for hardware.

    Returns:
        (supported_precisions, unsupported_precisions)
    """
    # Try PyTorch first (more comprehensive)
    pytorch_supported = detect_pytorch_precision_support(device)

    if pytorch_supported:
        supported = pytorch_supported
    else:
        # Fallback to NumPy
        supported = detect_numpy_precision_support()

    # Determine unsupported
    all_precisions = [
        Precision.FP64, Precision.FP32, Precision.FP16, Precision.BF16,
        Precision.FP8_E4M3, Precision.FP8_E5M2,
        Precision.INT32, Precision.INT16, Precision.INT8, Precision.INT4
    ]

    unsupported = [p for p in all_precisions if p not in supported]

    return supported, unsupported
```

---

### Phase 3: Multi-Precision Matmul Benchmark

#### 3.1 Update matmul_bench.py

```python
# benchmarks/matmul_bench.py - MAJOR REFACTOR

from ..schema import PrecisionTestResult
from ..precision_detector import get_precision_capabilities
from ...resource_model import Precision


def benchmark_matmul_precision(
    N: int,
    precision: Precision,
    num_trials: int = 10,
    num_warmup: int = 3,
    device: str = 'cpu'
) -> PrecisionTestResult:
    """
    Benchmark matrix multiplication at a specific precision.

    Args:
        N: Matrix dimension
        precision: Target precision to test
        num_trials: Number of measurement runs
        num_warmup: Number of warmup runs
        device: 'cpu' or 'cuda'

    Returns:
        PrecisionTestResult with performance or failure reason
    """
    # Map precision enum to numpy/torch dtype
    dtype_map = {
        Precision.FP64: (np.float64, 'torch.float64'),
        Precision.FP32: (np.float32, 'torch.float32'),
        Precision.FP16: (np.float16, 'torch.float16'),
        Precision.BF16: (None, 'torch.bfloat16'),  # PyTorch only
        Precision.INT32: (np.int32, 'torch.int32'),
        Precision.INT16: (np.int16, 'torch.int16'),
        Precision.INT8: (np.int8, 'torch.int8'),
    }

    if precision not in dtype_map:
        return PrecisionTestResult(
            precision=precision.value,
            supported=False,
            failure_reason=f"Precision {precision.value} not in benchmark dtype map",
            test_size=N,
            num_trials=0
        )

    numpy_dtype, torch_dtype_str = dtype_map[precision]

    # Try PyTorch if available and needed
    try:
        import torch
        use_pytorch = True

        # Get torch dtype
        torch_dtype = eval(torch_dtype_str)

    except (ImportError, AttributeError):
        use_pytorch = False

        # Fallback to NumPy
        if numpy_dtype is None:
            return PrecisionTestResult(
                precision=precision.value,
                supported=False,
                failure_reason=f"{precision.value} requires PyTorch (not installed)",
                test_size=N,
                num_trials=0
            )

    # Run benchmark
    try:
        if use_pytorch and (device == 'cuda' or numpy_dtype is None):
            result = _benchmark_pytorch_matmul(N, torch_dtype, device, num_trials, num_warmup)
        else:
            result = _benchmark_numpy_matmul(N, numpy_dtype, num_trials, num_warmup)

        return PrecisionTestResult(
            precision=precision.value,
            supported=True,
            failure_reason=None,
            measured_gflops=result['gflops'],
            efficiency=result.get('efficiency', None),
            mean_latency_ms=result['mean_latency_ms'],
            speedup_vs_fp32=result.get('speedup_vs_fp32', None),
            test_size=N,
            num_trials=num_trials
        )

    except Exception as e:
        return PrecisionTestResult(
            precision=precision.value,
            supported=False,
            failure_reason=f"Runtime error: {str(e)}",
            test_size=N,
            num_trials=0
        )


def _benchmark_numpy_matmul(N: int, dtype, num_trials: int, num_warmup: int) -> Dict:
    """Helper: NumPy benchmark"""
    A = np.random.rand(N, N).astype(dtype)
    B = np.random.rand(N, N).astype(dtype)

    # Warmup
    for _ in range(num_warmup):
        C = A @ B

    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time_ms = np.mean(times)
    flops = 2.0 * N * N * N
    gflops = flops / (mean_time_ms / 1000.0) / 1e9

    return {
        'mean_latency_ms': mean_time_ms,
        'gflops': gflops,
    }


def _benchmark_pytorch_matmul(N: int, dtype, device: str, num_trials: int, num_warmup: int) -> Dict:
    """Helper: PyTorch benchmark"""
    import torch

    A = torch.randn(N, N, dtype=dtype, device=device)
    B = torch.randn(N, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(num_warmup):
        C = torch.matmul(A, B)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.matmul(A, B)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time_ms = np.mean(times)
    flops = 2.0 * N * N * N
    gflops = flops / (mean_time_ms / 1000.0) / 1e9

    return {
        'mean_latency_ms': mean_time_ms,
        'gflops': gflops,
    }


def calibrate_matmul_all_precisions(
    sizes: List[int] = [1024, 2048],
    precisions: Optional[List[Precision]] = None,
    theoretical_peaks: Optional[Dict[str, float]] = None,
    device: str = 'cpu',
    num_trials: int = 10
) -> List[OperationCalibration]:
    """
    Calibrate matmul across all precisions.

    Args:
        sizes: Matrix sizes to test
        precisions: List of precisions to test (None = detect automatically)
        theoretical_peaks: Theoretical GFLOPS per precision
        device: 'cpu' or 'cuda'
        num_trials: Trials per test

    Returns:
        List of OperationCalibration, one per size, with precision_results populated
    """
    # Auto-detect supported precisions if not specified
    if precisions is None:
        supported, unsupported = get_precision_capabilities(device)
        precisions = supported + unsupported  # Test all to record failures

    calibrations = []

    for N in sizes:
        print(f"\nCalibrating matmul {N}×{N} across {len(precisions)} precisions...")

        # Test each precision
        precision_results = {}
        fp32_latency = None

        for precision in precisions:
            print(f"  Testing {precision.value:12s}...", end=" ", flush=True)

            result = benchmark_matmul_precision(N, precision, num_trials, device=device)

            if result.supported:
                print(f"✓ {result.measured_gflops:.1f} GFLOPS")

                # Track FP32 for speedup calculations
                if precision == Precision.FP32:
                    fp32_latency = result.mean_latency_ms
            else:
                print(f"✗ FAIL: {result.failure_reason}")

            precision_results[precision.value] = result

        # Calculate speedups relative to FP32
        if fp32_latency:
            for prec_name, result in precision_results.items():
                if result.supported and result.mean_latency_ms:
                    result.speedup_vs_fp32 = fp32_latency / result.mean_latency_ms

        # Create OperationCalibration
        calibration = OperationCalibration(
            operation_type=OperationType.MATMUL.value,
            precision_results=precision_results,

            # Legacy fields (use FP32 or best precision)
            measured_gflops=precision_results.get('fp32', precision_results[list(precision_results.keys())[0]]).measured_gflops or 0.0,
            efficiency=0.0,  # Computed per precision now
            achieved_bandwidth_gbps=0.0,
            memory_bound=False,
            compute_bound=False,
            arithmetic_intensity=0.0,
            batch_size=1,
            input_shape=(N, N),
            output_shape=(N, N),
            mean_latency_ms=0.0,
            std_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            num_trials=num_trials,
            extra_params={'matrix_size': N, 'device': device}
        )

        calibrations.append(calibration)

    return calibrations
```

---

### Phase 4: Calibrator Integration

#### 4.1 Update calibrator.py

```python
# calibrator.py - MODIFIED

from .precision_detector import get_precision_capabilities
from .benchmarks.matmul_bench import calibrate_matmul_all_precisions


def calibrate_hardware(
    hardware_name: str,
    theoretical_peak_gflops: float,
    theoretical_bandwidth_gbps: float,
    device: str = 'cpu',  # NEW PARAMETER
    precisions: Optional[List[str]] = None,  # NEW: specific precisions to test
    # ... existing parameters ...
) -> HardwareCalibration:
    """
    Run complete hardware calibration with multi-precision support.
    """
    # ... existing setup ...

    # NEW: Detect precision capabilities
    print("\n" + "=" * 80)
    print("Precision Capability Detection")
    print("=" * 80)

    supported, unsupported = get_precision_capabilities(device)

    print(f"\nSupported precisions ({len(supported)}):")
    for p in supported:
        print(f"  ✓ {p.value}")

    if unsupported:
        print(f"\nUnsupported precisions ({len(unsupported)}):")
        for p in unsupported:
            print(f"  ✗ {p.value}")

    # NEW: Multi-precision matmul calibration
    if 'matmul' in operations:
        print("\n" + "=" * 80)
        print("Matrix Multiplication Calibration (Multi-Precision)")
        print("=" * 80)

        matmul_cals = calibrate_matmul_all_precisions(
            sizes=[1024, 2048],
            precisions=supported + unsupported,  # Test all
            device=device,
            num_trials=5 if quick else 10
        )

        calibration.operations.extend(matmul_cals)

    # NEW: Build precision capability matrix
    precision_matrix = PrecisionCapabilityMatrix(
        hardware_name=hardware_name,
        supported_precisions=[p.value for p in supported],
        unsupported_precisions=[p.value for p in unsupported],
        peak_gflops_by_precision={},  # Populated from matmul results
        speedup_vs_fp32={}  # Populated from matmul results
    )

    # Extract peak GFLOPS per precision from matmul results
    for cal in calibration.operations:
        if cal.operation_type == 'matmul':
            for prec_name, prec_result in cal.precision_results.items():
                if prec_result.supported:
                    # Track best GFLOPS for each precision
                    current_best = precision_matrix.peak_gflops_by_precision.get(prec_name, 0.0)
                    precision_matrix.peak_gflops_by_precision[prec_name] = max(
                        current_best,
                        prec_result.measured_gflops or 0.0
                    )

                    # Track speedup
                    if prec_result.speedup_vs_fp32:
                        precision_matrix.speedup_vs_fp32[prec_name] = prec_result.speedup_vs_fp32

    calibration.precision_matrix = precision_matrix

    return calibration
```

---

### Phase 5: CLI Updates

#### 5.1 Update calibrate_hardware.py

```python
# cli/calibrate_hardware.py - MODIFIED

def display_calibration_results(cal: HardwareCalibration):
    """Display calibration with precision matrix"""
    # ... existing display code ...

    # NEW: Precision capability matrix
    if cal.precision_matrix:
        print("\n" + "=" * 80)
        print("Precision Capability Matrix")
        print("=" * 80)

        print("\nSupported Precisions:")
        print(f"{'Precision':<12} {'Peak GFLOPS':>12} {'Speedup vs FP32':>15}")
        print("-" * 80)

        for prec in cal.precision_matrix.supported_precisions:
            gflops = cal.precision_matrix.peak_gflops_by_precision.get(prec, 0.0)
            speedup = cal.precision_matrix.speedup_vs_fp32.get(prec, 1.0)
            print(f"{prec:<12} {gflops:>12.1f} {speedup:>14.2f}×")

        if cal.precision_matrix.unsupported_precisions:
            print("\nUnsupported Precisions:")
            for prec in cal.precision_matrix.unsupported_precisions:
                print(f"  ✗ {prec}")
```

---

### Phase 6: Reporting & Visualization

#### 6.1 Create precision_report.py

```python
# reporting/precision_report.py - NEW FILE

"""
Generate precision compatibility reports and visualizations.
"""

def generate_precision_matrix_markdown(cal: HardwareCalibration) -> str:
    """
    Generate markdown table showing precision support matrix.
    """
    md = f"# Precision Capability Matrix: {cal.metadata.hardware_name}\n\n"

    md += "## Summary\n\n"
    md += f"- **Supported**: {len(cal.precision_matrix.supported_precisions)} precisions\n"
    md += f"- **Unsupported**: {len(cal.precision_matrix.unsupported_precisions)} precisions\n\n"

    md += "## Detailed Results\n\n"
    md += "| Precision | Status | Peak GFLOPS | Speedup vs FP32 | Notes |\n"
    md += "|-----------|--------|-------------|-----------------|-------|\n"

    all_precisions = cal.precision_matrix.supported_precisions + cal.precision_matrix.unsupported_precisions

    for prec in sorted(all_precisions):
        if prec in cal.precision_matrix.supported_precisions:
            status = "✓ PASS"
            gflops = cal.precision_matrix.peak_gflops_by_precision.get(prec, 0.0)
            speedup = cal.precision_matrix.speedup_vs_fp32.get(prec, 1.0)
            notes = f"{speedup:.2f}× faster" if speedup > 1.0 else ""
            md += f"| {prec:<9} | {status:6} | {gflops:>11.1f} | {speedup:>14.2f}× | {notes} |\n"
        else:
            md += f"| {prec:<9} | ✗ FAIL | {'N/A':>11} | {'N/A':>14} | Not supported |\n"

    return md
```

---

## Testing Strategy

### Unit Tests

```python
# tests/calibration/test_precision_detector.py - NEW

def test_fp32_always_supported():
    """FP32 should be supported on all platforms"""
    supported, unsupported = get_precision_capabilities('cpu')
    assert Precision.FP32 in supported

def test_int8_cpu_support():
    """INT8 should work on CPU via NumPy"""
    supported, _ = get_precision_capabilities('cpu')
    assert Precision.INT8 in supported

def test_fp8_requires_h100():
    """FP8 should only be supported on H100+ GPUs"""
    # Skip if no GPU
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    cuda_cap = torch.cuda.get_device_capability()
    supported, unsupported = get_precision_capabilities('cuda')

    if cuda_cap >= (8, 9):
        # H100+
        assert Precision.FP8_E4M3 in supported
    else:
        # Older GPU
        assert Precision.FP8_E4M3 in unsupported
```

### Validation Tests

```python
# validation/empirical/test_multi_precision.py - NEW

def test_jetson_orin_precision_matrix():
    """Validate Jetson Orin Nano precision support"""
    # Run calibration
    cal = calibrate_hardware(
        "Jetson-Orin-Nano",
        theoretical_peak_gflops=1000,
        theoretical_bandwidth_gbps=68,
        device='cuda'
    )

    # Expected support
    assert 'fp32' in cal.precision_matrix.supported_precisions
    assert 'fp16' in cal.precision_matrix.supported_precisions
    assert 'int8' in cal.precision_matrix.supported_precisions

    # Expected failures (no FP8 on Orin)
    assert 'fp8_e4m3' in cal.precision_matrix.unsupported_precisions
```

---

## Expected Hardware Results

### Jetson Orin Nano (ARM Cortex-A78AE + Ampere GPU)

| Precision | Status | Expected Result |
|-----------|--------|-----------------|
| FP64 | ✗ FAIL | No FP64 on Ampere GPU |
| FP32 | ✓ PASS | ~300 GFLOPS |
| FP16 | ✓ PASS | ~600 GFLOPS (2× speedup) |
| BF16 | ✗ FAIL | Not on Ampere GPU (Ada+) |
| FP8 | ✗ FAIL | Hopper only (H100) |
| INT32 | ✓ PASS | ~100 GFLOPS |
| INT16 | ✓ PASS | ~200 GFLOPS |
| INT8 | ✓ PASS | ~1200 GFLOPS (4× speedup, Tensor Cores) |

### Intel i7-12700K (Alder Lake, AVX-512)

| Precision | Status | Expected Result |
|-----------|--------|-----------------|
| FP64 | ✓ PASS | ~360 GFLOPS |
| FP32 | ✓ PASS | ~720 GFLOPS |
| FP16 | ✓ PASS | ~720 GFLOPS (emulated, no speedup) |
| BF16 | ✗ FAIL | No native support |
| FP8 | ✗ FAIL | No native support |
| INT32 | ✓ PASS | ~360 GFLOPS |
| INT16 | ✓ PASS | ~720 GFLOPS (VNNI) |
| INT8 | ✓ PASS | ~1440 GOPS (2× via VNNI) |

### NVIDIA H100 SXM5

| Precision | Status | Expected Result |
|-----------|--------|-----------------|
| FP64 | ✓ PASS | 34 TFLOPS |
| FP32 | ✓ PASS | 67 TFLOPS |
| FP16 | ✓ PASS | 1979 TFLOPS (Tensor Cores) |
| BF16 | ✓ PASS | 1979 TFLOPS (Tensor Cores) |
| FP8 | ✓ PASS | 3958 TFLOPS (Tensor Cores) |
| INT8 | ✓ PASS | 3958 TOPS (Tensor Cores) |

---

## Usage Examples

### Example 1: Quick CPU Calibration

```bash
# Run multi-precision calibration on CPU
./cli/calibrate_hardware.py --preset i7-12700k --device cpu

# Output:
# Precision Capability Detection
# ================================================================================
# Supported precisions (6):
#   ✓ fp64
#   ✓ fp32
#   ✓ fp16
#   ✓ int32
#   ✓ int16
#   ✓ int8
#
# Unsupported precisions (4):
#   ✗ bf16
#   ✗ fp8_e4m3
#   ✗ fp8_e5m2
#   ✗ int4
#
# Matrix Multiplication Calibration (Multi-Precision)
# ================================================================================
#
# Calibrating matmul 1024×1024 across 10 precisions...
#   Testing fp64       ... ✓ 380.5 GFLOPS
#   Testing fp32       ... ✓ 741.2 GFLOPS
#   Testing fp16       ... ✓ 740.8 GFLOPS
#   Testing bf16       ... ✗ FAIL: bf16 requires PyTorch (not installed)
#   Testing int32      ... ✓ 370.1 GFLOPS
#   Testing int16      ... ✓ 722.3 GFLOPS
#   Testing int8       ... ✓ 1405.6 GOPS
#   Testing fp8_e4m3   ... ✗ FAIL: Precision fp8_e4m3 not in benchmark dtype map
#   Testing fp8_e5m2   ... ✗ FAIL: Precision fp8_e5m2 not in benchmark dtype map
#   Testing int4       ... ✗ FAIL: Precision int4 not in benchmark dtype map
```

### Example 2: GPU Calibration on Jetson

```bash
# Run on CUDA device
python cli/calibrate_hardware.py --preset jetson-orin-nano --device cuda

# Save results
python cli/calibrate_hardware.py --preset jetson-orin-nano --device cuda \
    --output profiles/jetson_orin_nano_multi_precision.json
```

### Example 3: Custom Precision Selection

```bash
# Only test FP32, FP16, INT8 (skip others)
./cli/calibrate_hardware.py --preset jetson-orin-nano \
    --device cuda \
    --precisions fp32,fp16,int8
```

---

## Deliverables

### Code Files (New)
1. `src/graphs/hardware/calibration/precision_detector.py` (~300 lines)
2. `src/graphs/reporting/precision_report.py` (~200 lines)
3. `tests/calibration/test_precision_detector.py` (~150 lines)
4. `validation/empirical/test_multi_precision.py` (~200 lines)

### Code Files (Modified)
1. `src/graphs/hardware/calibration/schema.py` (+150 lines)
2. `src/graphs/hardware/calibration/benchmarks/matmul_bench.py` (+400 lines)
3. `src/graphs/hardware/calibration/calibrator.py` (+100 lines)
4. `cli/calibrate_hardware.py` (+80 lines)

### Documentation
1. This plan: `docs/MULTI_PRECISION_CALIBRATION_PLAN.md`
2. Updated: `src/graphs/hardware/calibration/README.md`
3. New: `docs/PRECISION_COMPATIBILITY_MATRICES.md`

### Generated Profiles
1. `profiles/i7_12700k_multi_precision.json`
2. `profiles/jetson_orin_nano_multi_precision.json`
3. `profiles/h100_multi_precision.json` (if hardware available)

---

## Timeline

| Phase | Effort | Duration |
|-------|--------|----------|
| 1. Schema extensions | 4 hours | Day 1 |
| 2. Precision detection | 6 hours | Day 1-2 |
| 3. Matmul benchmark | 8 hours | Day 2-3 |
| 4. Calibrator integration | 4 hours | Day 3 |
| 5. CLI updates | 3 hours | Day 3 |
| 6. Reporting | 3 hours | Day 4 |
| 7. Testing | 4 hours | Day 4 |
| 8. Documentation | 2 hours | Day 4 |
| **Total** | **34 hours** | **4 days** |

---

## Open Questions

1. **Should we test mixed-precision operations?** (e.g., INT8 matmul with FP32 accumulation)
2. **How to handle quantization schemes?** (symmetric vs asymmetric, per-tensor vs per-channel)
3. **Should we calibrate tensor cores separately from CUDA cores?**
4. **Do we need per-operation theoretical peaks?** (e.g., H100 FP8 matmul = 3958 TFLOPS, but FP8 elementwise = 67 TFLOPS)

---

## Success Criteria

- [x] Plan documented
- [ ] Schema supports per-precision results
- [ ] Precision detection works on CPU + GPU
- [ ] Matmul benchmark tests all 10+ precisions
- [ ] FAIL status correctly reported for unsupported precisions
- [ ] Calibration profiles include precision matrix
- [ ] CLI displays precision compatibility table
- [ ] Validated on 3+ hardware platforms
- [ ] Documentation complete

---

## Next Steps

1. **Review this plan** - Get stakeholder approval
2. **Implement Phase 1** - Schema extensions (4 hours)
3. **Validate on Jetson** - Run initial multi-precision test
4. **Iterate based on findings** - Adjust plan as needed
