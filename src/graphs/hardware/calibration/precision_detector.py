"""
Precision Detection Framework

Detects which numerical precisions are supported on the current hardware.
Tests both NumPy and PyTorch dtypes to determine hardware capabilities.
"""

import numpy as np
import sys
from typing import List, Tuple, Optional


# Import Precision enum from resource_model
try:
    from ..resource_model import Precision
except ImportError:
    # Fallback: define locally if import fails
    from enum import Enum
    class Precision(Enum):
        FP64 = "fp64"
        FP32 = "fp32"
        TF32 = "tf32"  # NVIDIA TensorFloat-32, Tensor Cores only
        FP16 = "fp16"
        BF16 = "bf16"
        FP8_E4M3 = "fp8_e4m3"
        FP8_E5M2 = "fp8_e5m2"
        INT64 = "int64"
        INT32 = "int32"
        INT16 = "int16"
        INT8 = "int8"
        INT4 = "int4"


def test_numpy_precision(precision: Precision, numpy_dtype) -> Tuple[bool, Optional[str]]:
    """
    Test if a precision works with NumPy.

    Args:
        precision: Precision enum
        numpy_dtype: NumPy dtype to test (e.g., np.float32)

    Returns:
        (supported, failure_reason)
    """
    try:
        # Create small test arrays (use appropriate RNG for dtype)
        if precision in [Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8, Precision.INT4]:
            # Integer types: use randint
            A = np.random.randint(-10, 10, size=(16, 16), dtype=numpy_dtype)
            B = np.random.randint(-10, 10, size=(16, 16), dtype=numpy_dtype)
        else:
            # Floating-point types: use rand
            A = np.random.rand(16, 16).astype(numpy_dtype)
            B = np.random.rand(16, 16).astype(numpy_dtype)

        # Test matmul
        C = A @ B

        # Check for NaN/Inf (only for floating-point types)
        if precision not in [Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8, Precision.INT4]:
            if not np.all(np.isfinite(C)):
                return False, "Matmul produced NaN/Inf"

        return True, None

    except (TypeError, ValueError, RuntimeError) as e:
        return False, f"NumPy error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}"


def test_pytorch_precision(precision: Precision, torch_dtype, device: str = 'cpu') -> Tuple[bool, Optional[str]]:
    """
    Test if a precision works with PyTorch.

    Args:
        precision: Precision enum
        torch_dtype: PyTorch dtype to test
        device: 'cpu' or 'cuda'

    Returns:
        (supported, failure_reason)
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed"

    try:
        # Create small test tensors (use appropriate RNG for dtype)
        if precision in [Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8, Precision.INT4]:
            # Integer types: use randint
            A = torch.randint(-10, 10, (16, 16), dtype=torch_dtype, device=device)
            B = torch.randint(-10, 10, (16, 16), dtype=torch_dtype, device=device)
        else:
            # Floating-point types: use randn
            A = torch.randn(16, 16, dtype=torch_dtype, device=device)
            B = torch.randn(16, 16, dtype=torch_dtype, device=device)

        # Test matmul
        C = torch.matmul(A, B)

        # Sync if CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        # Check for NaN/Inf (only for floating-point types)
        if precision not in [Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8, Precision.INT4]:
            if not torch.all(torch.isfinite(C)):
                return False, "Matmul produced NaN/Inf"

        return True, None

    except (TypeError, RuntimeError) as e:
        return False, f"PyTorch error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}"


def detect_numpy_precisions() -> Tuple[List[Precision], List[Precision]]:
    """
    Detect which precisions NumPy supports.

    Returns:
        (supported, unsupported)
    """
    supported = []
    unsupported = []

    tests = [
        (Precision.FP64, np.float64),
        (Precision.FP32, np.float32),
        (Precision.FP16, np.float16),
        (Precision.INT64, np.int64),
        (Precision.INT32, np.int32),
        (Precision.INT16, np.int16),
        (Precision.INT8, np.int8),
    ]

    for precision, dtype in tests:
        is_supported, reason = test_numpy_precision(precision, dtype)
        if is_supported:
            supported.append(precision)
        else:
            unsupported.append(precision)

    # BF16, FP8 not in NumPy (PyTorch only)
    unsupported.extend([Precision.BF16, Precision.FP8_E4M3, Precision.FP8_E5M2])

    return supported, unsupported


def detect_pytorch_precisions(device: str = 'cpu') -> Tuple[List[Precision], List[Precision]]:
    """
    Detect which precisions PyTorch supports on given device.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        (supported, unsupported)
    """
    try:
        import torch
    except ImportError:
        return [], []

    supported = []
    unsupported = []

    # Basic precisions
    tests = [
        (Precision.FP64, torch.float64),
        (Precision.FP32, torch.float32),
        (Precision.FP16, torch.float16),
        (Precision.BF16, torch.bfloat16),
        (Precision.INT64, torch.int64),
        (Precision.INT32, torch.int32),
        (Precision.INT16, torch.int16),
        (Precision.INT8, torch.int8),
    ]

    for precision, dtype in tests:
        is_supported, reason = test_pytorch_precision(precision, dtype, device)
        if is_supported:
            supported.append(precision)
        else:
            unsupported.append(precision)

    # FP8 only on CUDA with compute capability >= 8.9 (H100)
    if device == 'cuda' and torch.cuda.is_available():
        try:
            cuda_cap = torch.cuda.get_device_capability()
            if cuda_cap[0] >= 8 and cuda_cap[1] >= 9:
                # H100+ supports FP8
                if hasattr(torch, 'float8_e4m3fn'):
                    is_supported, _ = test_pytorch_precision(Precision.FP8_E4M3, torch.float8_e4m3fn, device)
                    if is_supported:
                        supported.append(Precision.FP8_E4M3)
                    else:
                        unsupported.append(Precision.FP8_E4M3)
                else:
                    unsupported.append(Precision.FP8_E4M3)

                if hasattr(torch, 'float8_e5m2'):
                    is_supported, _ = test_pytorch_precision(Precision.FP8_E5M2, torch.float8_e5m2, device)
                    if is_supported:
                        supported.append(Precision.FP8_E5M2)
                    else:
                        unsupported.append(Precision.FP8_E5M2)
                else:
                    unsupported.append(Precision.FP8_E5M2)
            else:
                unsupported.extend([Precision.FP8_E4M3, Precision.FP8_E5M2])
        except:
            unsupported.extend([Precision.FP8_E4M3, Precision.FP8_E5M2])
    else:
        unsupported.extend([Precision.FP8_E4M3, Precision.FP8_E5M2])

    return supported, unsupported


def get_precision_capabilities(device: str = 'cpu') -> Tuple[List[Precision], List[Precision]]:
    """
    Get comprehensive precision support for hardware.

    Tries PyTorch first (more comprehensive), falls back to NumPy.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        (supported_precisions, unsupported_precisions)
    """
    # Try PyTorch first (has more precision types)
    pytorch_supported, pytorch_unsupported = detect_pytorch_precisions(device)

    if pytorch_supported:
        # PyTorch available - use its results
        return pytorch_supported, pytorch_unsupported
    else:
        # Fallback to NumPy
        return detect_numpy_precisions()


def print_precision_capabilities(device: str = 'cpu'):
    """
    Print precision capabilities for debugging.

    Args:
        device: 'cpu' or 'cuda'
    """
    supported, unsupported = get_precision_capabilities(device)

    print(f"\nPrecision Detection (device={device}):")
    print("=" * 60)

    if supported:
        print(f"\nSupported ({len(supported)}):")
        for p in supported:
            print(f"  ✓ {p.value}")

    if unsupported:
        print(f"\nUnsupported ({len(unsupported)}):")
        for p in unsupported:
            print(f"  ✗ {p.value}")


if __name__ == "__main__":
    # Test precision detection
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    print_precision_capabilities(args.device)
