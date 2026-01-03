#!/usr/bin/env python3
"""
Hardware Calibration Benchmark

Calibrates hardware performance and outputs calibration.json for the hardware registry.
Requires the graphs package for hardware auto-detection and registry matching.

Works on:
- Windows (native Python with graphs package)
- Linux
- macOS

Requirements:
    pip install torch numpy psutil
    pip install -e /path/to/graphs  # Or PYTHONPATH=src

Usage:
    # CPU-only calibration (recommended to avoid iGPU contamination)
    python portable_calibration.py --cpu-only

    # Auto-detect all devices
    python portable_calibration.py --all-devices

    # Specific device
    python portable_calibration.py --device cuda:0
    python portable_calibration.py --device rocm:0
    python portable_calibration.py --device directml

    # Quick calibration (fewer sizes)
    python portable_calibration.py --quick

    # Specify output file (default: <hardware_id>_calibration.json)
    python portable_calibration.py --output my_calibration.json

Author: Generated for graphs hardware registry
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_warmup: int = 3
    num_trials: int = 10
    min_time_seconds: float = 0.1  # Minimum time per measurement
    quick_mode: bool = False
    ultra_quick: bool = False  # Minimal sizes for testing
    cpu_only: bool = False

    # Matrix sizes for GEMM benchmarks (M=N=K)
    gemm_sizes: List[int] = field(default_factory=lambda: [
        128, 256, 512, 1024, 2048, 4096, 8192
    ])

    # Quick mode uses fewer sizes
    gemm_sizes_quick: List[int] = field(default_factory=lambda: [
        256, 1024, 2048
    ])

    # Ultra-quick mode for testing
    gemm_sizes_ultra: List[int] = field(default_factory=lambda: [
        256, 512
    ])

    # Memory sizes for bandwidth tests (in MB)
    memory_sizes_mb: List[int] = field(default_factory=lambda: [
        1, 4, 16, 64, 256
    ])

    memory_sizes_quick: List[int] = field(default_factory=lambda: [
        4, 64
    ])

    # Precisions to test
    precisions: List[str] = field(default_factory=lambda: [
        'fp32', 'fp16', 'bf16', 'int8'
    ])

    precisions_quick: List[str] = field(default_factory=lambda: [
        'fp32', 'fp16'
    ])

    def get_gemm_sizes(self) -> List[int]:
        if self.ultra_quick:
            return self.gemm_sizes_ultra
        return self.gemm_sizes_quick if self.quick_mode else self.gemm_sizes

    def get_memory_sizes(self) -> List[int]:
        if self.ultra_quick or self.quick_mode:
            return self.memory_sizes_quick
        return self.memory_sizes_mb

    def get_precisions(self) -> List[str]:
        if self.ultra_quick:
            return self.precisions_quick
        return self.precisions


# ============================================================================
# Hardware Detection
# ============================================================================

def normalize_hardware_id(name: str) -> str:
    """
    Convert hardware name to normalized ID matching registry convention.

    Example: "AMD Ryzen 9 8945HS with Radeon 780M Graphics"
          -> "amd_ryzen_9_8945hs_w_radeon_780m_graphics"
    """
    import re
    # Lowercase
    hw_id = name.lower()
    # Replace "with" with "w"
    hw_id = hw_id.replace(' with ', '_w_')
    # Replace special chars with underscore
    hw_id = re.sub(r'[^a-z0-9]+', '_', hw_id)
    # Remove leading/trailing underscores
    hw_id = hw_id.strip('_')
    # Collapse multiple underscores
    hw_id = re.sub(r'_+', '_', hw_id)
    return hw_id


def detect_cpu_info() -> Dict[str, Any]:
    """Detect CPU information."""
    import psutil

    info = {
        'name': platform.processor() or 'Unknown',
        'architecture': platform.machine(),
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'frequency_mhz': None,
        'hardware_id': None,  # Will be set after name detection
    }

    # Try to get CPU frequency
    try:
        freq = psutil.cpu_freq()
        if freq:
            info['frequency_mhz'] = freq.current
            info['frequency_max_mhz'] = freq.max
    except Exception:
        pass

    # Try to get better CPU name on Windows
    if platform.system() == 'Windows':
        try:
            import subprocess
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True
            )
            lines = [l.strip() for l in result.stdout.split('\n') if l.strip() and l.strip() != 'Name']
            if lines:
                info['name'] = lines[0]
        except Exception:
            pass

    # Try to get better CPU name on Linux
    elif platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['name'] = line.split(':')[1].strip()
                        break
        except Exception:
            pass

    # Generate hardware ID from name
    info['hardware_id'] = normalize_hardware_id(info['name'])

    return info


def detect_gpu_info() -> List[Dict[str, Any]]:
    """Detect available GPUs."""
    gpus = []

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                name = props.name
                gpus.append({
                    'index': i,
                    'name': name,
                    'type': 'cuda',
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessors': props.multi_processor_count,
                    'hardware_id': normalize_hardware_id(name),
                })
    except Exception:
        pass

    # Try PyTorch ROCm (AMD)
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip:
            # ROCm uses same API as CUDA
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    name = props.name
                    gpus.append({
                        'index': i,
                        'name': name,
                        'type': 'rocm',
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_units': props.multi_processor_count,
                        'hardware_id': normalize_hardware_id(name),
                    })
    except Exception:
        pass

    return gpus


def detect_npu_info() -> List[Dict[str, Any]]:
    """Detect available NPUs (DirectML, XDNA, etc.)."""
    npus = []

    # Try DirectML (Windows)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            # DirectML doesn't expose NPU name directly, use generic
            name = 'DirectML NPU'
            npus.append({
                'index': 0,
                'name': name,
                'type': 'directml',
                'hardware_id': 'directml_npu',
                'notes': 'Detected via ONNX Runtime DirectML provider'
            })
    except Exception:
        pass

    return npus


def detect_all_devices(cpu_only: bool = False) -> Dict[str, Any]:
    """Detect all available compute devices."""
    result = {
        'cpu': detect_cpu_info(),
        'gpus': [],
        'npus': [],
        'platform': {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
        }
    }

    if not cpu_only:
        result['gpus'] = detect_gpu_info()
        result['npus'] = detect_npu_info()

    # Add PyTorch/NumPy versions
    try:
        import torch
        result['platform']['pytorch_version'] = torch.__version__
        result['platform']['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        pass

    try:
        import numpy as np
        result['platform']['numpy_version'] = np.__version__
    except ImportError:
        pass

    return result


def match_hardware_to_registry(device_info: Dict[str, Any], device_type: str = 'cpu') -> Tuple[str, str, float]:
    """
    Match detected hardware against the graphs hardware registry.

    Args:
        device_info: Detected device information
        device_type: 'cpu', 'gpu', or 'npu'

    Returns:
        Tuple of (hardware_id, hardware_name, confidence)

    Raises:
        RuntimeError: If graphs package not available or no match found
    """
    try:
        from graphs.hardware.calibration.registry_sync import get_registry
    except ImportError:
        raise RuntimeError(
            "graphs package not installed. This script requires the graphs package for hardware auto-detection.\n"
            "Install with: pip install -e /path/to/graphs\n"
            "Or run from the graphs directory with: PYTHONPATH=src python tools/portable_calibration.py"
        )

    # Get detected hardware name
    if device_type == 'cpu':
        detected_name = device_info['cpu']['name']
    elif device_type == 'gpu':
        if device_info['gpus']:
            detected_name = device_info['gpus'][0]['name']
        else:
            raise RuntimeError("No GPU detected")
    else:
        detected_name = 'Unknown'

    # Load hardware registry
    registry = get_registry()
    entries = registry.list_hardware(device_type=device_type)

    if not entries:
        raise RuntimeError(
            f"No {device_type} entries found in hardware registry.\n"
            f"Check: ls hardware_registry/{device_type}/"
        )

    # Find best match by name similarity
    def compute_similarity(detected: str, registry_name: str, registry_id: str) -> float:
        """Compute string similarity score."""
        import re
        # Normalize strings
        detected_norm = re.sub(r'[^a-z0-9]', '', detected.lower())
        name_norm = re.sub(r'[^a-z0-9]', '', registry_name.lower())
        id_norm = re.sub(r'[^a-z0-9]', '', registry_id.lower())

        # Check for substring matches
        if detected_norm in name_norm or name_norm in detected_norm:
            return 0.9
        if detected_norm in id_norm or id_norm in detected_norm:
            return 0.85

        # Check key component matches (model numbers, etc.)
        detected_parts = set(re.findall(r'[a-z]+|\d+', detected.lower()))
        name_parts = set(re.findall(r'[a-z]+|\d+', registry_name.lower()))
        id_parts = set(re.findall(r'[a-z]+|\d+', registry_id.lower()))

        # Calculate Jaccard similarity
        all_registry_parts = name_parts | id_parts
        if not all_registry_parts:
            return 0.0

        intersection = detected_parts & all_registry_parts
        union = detected_parts | all_registry_parts

        return len(intersection) / len(union) if union else 0.0

    best_match = None
    best_score = 0.0

    for entry in entries:
        score = compute_similarity(detected_name, entry.model, entry.id)
        if score > best_score:
            best_score = score
            best_match = entry

    # Require minimum confidence
    MIN_CONFIDENCE = 0.3
    if best_match is None or best_score < MIN_CONFIDENCE:
        available = [f"  - {e.id}: {e.model}" for e in entries[:10]]
        raise RuntimeError(
            f"No hardware match found in registry for: {detected_name}\n"
            f"Device type: {device_type}\n"
            f"Best match score: {best_score*100:.0f}% (minimum: {MIN_CONFIDENCE*100:.0f}%)\n\n"
            f"Available {device_type} entries:\n" + "\n".join(available) + "\n\n"
            "Options:\n"
            "  1. Add your hardware to the registry first\n"
            "  2. Check: python cli/calibration_coverage.py"
        )

    return (best_match.id, best_match.model, best_score)


# ============================================================================
# Benchmarks
# ============================================================================

def benchmark_gemm_pytorch(
    size: int,
    dtype: str,
    device: str,
    config: BenchmarkConfig
) -> Dict[str, float]:
    """
    Run GEMM benchmark using PyTorch.

    Returns:
        Dict with 'gflops', 'time_ms', 'size'
    """
    import torch

    # Map dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'int8': torch.int8,
    }

    torch_dtype = dtype_map.get(dtype, torch.float32)

    # Check if dtype is supported on device
    if device == 'cpu' and dtype == 'bf16':
        # BF16 on CPU may not be supported on all hardware
        try:
            test = torch.randn(2, 2, dtype=torch.bfloat16)
        except Exception:
            return {'gflops': 0, 'time_ms': 0, 'size': size, 'skipped': True}

    # Create matrices
    M, N, K = size, size, size

    try:
        if dtype == 'int8':
            # INT8 GEMM uses different API
            A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
            B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        else:
            A = torch.randn(M, K, dtype=torch_dtype, device=device)
            B = torch.randn(K, N, dtype=torch_dtype, device=device)
    except Exception as e:
        return {'gflops': 0, 'time_ms': 0, 'size': size, 'skipped': True, 'error': str(e)}

    # Define the operation
    def run_gemm():
        if dtype == 'int8':
            # Use matmul which works for int8
            return torch.matmul(A.float(), B.float()).to(torch.int32)
        else:
            return torch.matmul(A, B)

    # Warmup
    for _ in range(config.num_warmup):
        _ = run_gemm()
        if device != 'cpu':
            torch.cuda.synchronize() if 'cuda' in device else None

    # Benchmark
    times = []
    for _ in range(config.num_trials):
        if device != 'cpu':
            torch.cuda.synchronize() if 'cuda' in device else None

        start = time.perf_counter()
        _ = run_gemm()

        if device != 'cpu':
            torch.cuda.synchronize() if 'cuda' in device else None

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Calculate GFLOPS
    # GEMM: 2*M*N*K operations (multiply-add = 2 ops)
    flops = 2 * M * N * K
    avg_time = sum(times) / len(times)
    min_time = min(times)

    gflops = flops / min_time / 1e9  # Use best time for peak
    avg_gflops = flops / avg_time / 1e9

    return {
        'size': size,
        'gflops': gflops,
        'avg_gflops': avg_gflops,
        'time_ms': min_time * 1000,
        'avg_time_ms': avg_time * 1000,
        'flops': flops,
    }


def benchmark_memory_pytorch(
    size_mb: int,
    device: str,
    config: BenchmarkConfig
) -> Dict[str, float]:
    """
    Run memory bandwidth benchmark using PyTorch.

    Implements STREAM-like copy operation: C = A

    Returns:
        Dict with 'bandwidth_gbps', 'time_ms', 'size_mb'
    """
    import torch

    # Calculate number of elements for given size
    num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32

    try:
        A = torch.randn(num_elements, dtype=torch.float32, device=device)
        B = torch.empty_like(A)
    except Exception as e:
        return {'bandwidth_gbps': 0, 'time_ms': 0, 'size_mb': size_mb, 'error': str(e)}

    def run_copy():
        B.copy_(A)
        return B

    # Warmup
    for _ in range(config.num_warmup):
        _ = run_copy()
        if device != 'cpu':
            torch.cuda.synchronize() if 'cuda' in device else None

    # Benchmark
    times = []
    for _ in range(config.num_trials):
        if device != 'cpu':
            torch.cuda.synchronize() if 'cuda' in device else None

        start = time.perf_counter()
        _ = run_copy()

        if device != 'cpu':
            torch.cuda.synchronize() if 'cuda' in device else None

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Calculate bandwidth
    # Copy: read A + write B = 2 * size bytes transferred
    bytes_transferred = 2 * num_elements * 4  # float32 = 4 bytes
    min_time = min(times)

    bandwidth_gbps = bytes_transferred / min_time / 1e9

    return {
        'size_mb': size_mb,
        'bandwidth_gbps': bandwidth_gbps,
        'time_ms': min_time * 1000,
        'bytes_transferred': bytes_transferred,
    }


def benchmark_device(
    device: str,
    device_name: str,
    config: BenchmarkConfig
) -> Dict[str, Any]:
    """
    Run full benchmark suite on a device.

    Returns:
        Calibration data dictionary
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {device_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    results = {
        'device': device,
        'device_name': device_name,
        'gemm_results': {},
        'memory_results': [],
        'best_gflops': {},
        'best_bandwidth_gbps': 0,
    }

    sizes = config.get_gemm_sizes()
    precisions = config.get_precisions()
    memory_sizes = config.get_memory_sizes()

    # GEMM benchmarks for each precision
    for precision in precisions:
        print(f"\n  {precision.upper()} GEMM benchmarks:")
        results['gemm_results'][precision] = []
        best_gflops = 0

        for size in sizes:
            result = benchmark_gemm_pytorch(size, precision, device, config)
            results['gemm_results'][precision].append(result)

            if result.get('skipped'):
                print(f"    {size:5d}x{size:<5d}: SKIPPED")
                continue

            gflops = result['gflops']
            best_gflops = max(best_gflops, gflops)

            print(f"    {size:5d}x{size:<5d}: {gflops:8.2f} GFLOPS ({result['time_ms']:.2f} ms)")

        results['best_gflops'][precision] = best_gflops
        print(f"    Peak {precision.upper()}: {best_gflops:.2f} GFLOPS")

    # Memory bandwidth benchmarks
    print(f"\n  Memory bandwidth benchmarks:")
    for size_mb in memory_sizes:
        result = benchmark_memory_pytorch(size_mb, device, config)
        results['memory_results'].append(result)

        if 'error' in result:
            print(f"    {size_mb:4d} MB: ERROR - {result['error']}")
            continue

        bw = result['bandwidth_gbps']
        results['best_bandwidth_gbps'] = max(results['best_bandwidth_gbps'], bw)
        print(f"    {size_mb:4d} MB: {bw:8.2f} GB/s")

    print(f"    Peak bandwidth: {results['best_bandwidth_gbps']:.2f} GB/s")

    return results


# ============================================================================
# DirectML NPU Benchmarks (Windows only)
# ============================================================================

def benchmark_npu_directml(config: BenchmarkConfig) -> Optional[Dict[str, Any]]:
    """
    Run NPU benchmarks using DirectML via ONNX Runtime.

    Returns:
        Calibration data dictionary or None if not available
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  ONNX Runtime not installed. Install with: pip install onnxruntime-directml")
        return None

    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' not in providers:
        print("  DirectML provider not available")
        return None

    print(f"\n{'='*60}")
    print("Benchmarking: DirectML NPU")
    print(f"{'='*60}")

    results = {
        'device': 'directml',
        'device_name': 'DirectML NPU',
        'gemm_results': {},
        'best_gflops': {},
    }

    sizes = config.get_gemm_sizes()

    # Create a simple GEMM model using ONNX
    def create_gemm_model(M, N, K, dtype='float16'):
        """Create a simple ONNX model for GEMM."""
        import onnx
        from onnx import helper, TensorProto

        # Map dtype
        dtype_map = {
            'float32': TensorProto.FLOAT,
            'float16': TensorProto.FLOAT16,
        }
        onnx_dtype = dtype_map.get(dtype, TensorProto.FLOAT16)

        # Create inputs
        A = helper.make_tensor_value_info('A', onnx_dtype, [M, K])
        B = helper.make_tensor_value_info('B', onnx_dtype, [K, N])
        C = helper.make_tensor_value_info('C', onnx_dtype, [M, N])

        # Create MatMul node
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['A', 'B'],
            outputs=['C']
        )

        # Create graph
        graph = helper.make_graph(
            [matmul_node],
            'gemm_graph',
            [A, B],
            [C]
        )

        # Create model
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        return model.SerializeToString()

    # Test FP16 (most common for NPUs)
    precision = 'fp16'
    print(f"\n  {precision.upper()} GEMM benchmarks (DirectML):")
    results['gemm_results'][precision] = []
    best_gflops = 0

    for size in sizes:
        try:
            # Create model
            model_bytes = create_gemm_model(size, size, size, 'float16')

            # Create session with DirectML
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(
                model_bytes,
                sess_options,
                providers=['DmlExecutionProvider']
            )

            # Create inputs
            np_dtype = np.float16
            A = np.random.randn(size, size).astype(np_dtype)
            B = np.random.randn(size, size).astype(np_dtype)

            # Warmup
            for _ in range(config.num_warmup):
                _ = session.run(None, {'A': A, 'B': B})

            # Benchmark
            times = []
            for _ in range(config.num_trials):
                start = time.perf_counter()
                _ = session.run(None, {'A': A, 'B': B})
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            # Calculate GFLOPS
            flops = 2 * size * size * size
            min_time = min(times)
            gflops = flops / min_time / 1e9
            best_gflops = max(best_gflops, gflops)

            results['gemm_results'][precision].append({
                'size': size,
                'gflops': gflops,
                'time_ms': min_time * 1000,
            })

            print(f"    {size:5d}x{size:<5d}: {gflops:8.2f} GFLOPS ({min_time*1000:.2f} ms)")

        except Exception as e:
            print(f"    {size:5d}x{size:<5d}: ERROR - {str(e)[:50]}")
            results['gemm_results'][precision].append({
                'size': size,
                'gflops': 0,
                'error': str(e),
            })

    results['best_gflops'][precision] = best_gflops
    print(f"    Peak {precision.upper()}: {best_gflops:.2f} GFLOPS")

    return results


# ============================================================================
# Calibration Output
# ============================================================================

def generate_calibration_json(
    device_results: Dict[str, Any],
    device_info: Dict[str, Any],
    matched_hw: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate calibration.json in hardware registry format.

    Args:
        device_results: Benchmark results
        device_info: Detected device information
        matched_hw: Matched hardware from registry (hardware_id, hardware_name, confidence)
    """
    hardware_id = matched_hw['hardware_id']

    # Build metadata
    metadata = {
        'hardware_id': hardware_id,
        'registry_match_confidence': matched_hw['confidence'],
        'hardware_name': device_results.get('device_name', 'Unknown'),
        'calibration_date': datetime.now().isoformat(),
        'calibration_tool_version': '2.0.0-portable',
        'cpu_model': device_info['cpu']['name'],
        'cpu_count': device_info['cpu']['logical_cores'],
        'python_version': device_info['platform']['python_version'],
        'pytorch_version': device_info['platform'].get('pytorch_version', 'N/A'),
        'numpy_version': device_info['platform'].get('numpy_version', 'N/A'),
        'num_warmup_runs': 3,
        'num_measurement_runs': 10,
        'device_type': device_results['device'],
        'platform_os': device_info['platform']['os'],
        'platform_architecture': device_info['cpu']['architecture'],
        'framework': 'pytorch',
    }

    # Extract best measurements
    best_gflops = device_results.get('best_gflops', {})
    best_fp32 = best_gflops.get('fp32', 0)
    best_fp16 = best_gflops.get('fp16', 0)
    best_int8 = best_gflops.get('int8', 0)

    # Calculate averages
    all_gflops = []
    for precision, results in device_results.get('gemm_results', {}).items():
        for r in results:
            if r.get('gflops', 0) > 0:
                all_gflops.append(r['gflops'])

    avg_gflops = sum(all_gflops) / len(all_gflops) if all_gflops else 0

    # Build operation profiles
    operation_profiles = {}
    for precision, results in device_results.get('gemm_results', {}).items():
        for r in results:
            if r.get('skipped') or r.get('error'):
                continue

            key = f"gemm_precision={precision}_size={r['size']}_device={device_results['device']}"
            operation_profiles[key] = {
                'operation_type': 'gemm',
                'precision': precision,
                'size': r['size'],
                'gflops': r['gflops'],
                'time_ms': r['time_ms'],
                'flops': r.get('flops', 2 * r['size']**3),
            }

    # Build calibration data
    calibration = {
        'metadata': metadata,
        'theoretical_peak_gflops': 0,  # Will be filled from registry
        'theoretical_bandwidth_gbps': 0,  # Will be filled from registry
        'best_measured_gflops': max(best_gflops.values()) if best_gflops else 0,
        'avg_measured_gflops': avg_gflops,
        'measured_bandwidth_gbps': device_results.get('best_bandwidth_gbps', 0),
        'per_precision_peaks': {
            'fp32': best_fp32,
            'fp16': best_fp16,
            'bf16': best_gflops.get('bf16', 0),
            'int8': best_int8,
        },
        'operation_profiles': operation_profiles,
        'calibration_runs': [],  # Full benchmark data
    }

    # Add detailed runs
    for precision, results in device_results.get('gemm_results', {}).items():
        for r in results:
            if r.get('skipped') or r.get('error'):
                continue
            calibration['calibration_runs'].append({
                'operation': 'gemm',
                'precision': precision,
                'size': r['size'],
                'measured_gflops': r['gflops'],
                'time_ms': r['time_ms'],
            })

    return calibration


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Portable Hardware Calibration Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--cpu-only', action='store_true',
                        help='Only benchmark CPU (avoids iGPU contamination)')
    parser.add_argument('--device', type=str, default=None,
                        help='Specific device to benchmark (cpu, cuda:0, rocm:0, directml)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with fewer sizes')
    parser.add_argument('--ultra-quick', action='store_true',
                        help='Ultra-quick mode for testing (minimal sizes)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: <hardware_id>_calibration.json)')
    parser.add_argument('--all-devices', action='store_true',
                        help='Benchmark all detected devices')

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        quick_mode=args.quick,
        ultra_quick=args.ultra_quick,
        cpu_only=args.cpu_only,
    )

    # Ultra-quick also reduces trials
    if args.ultra_quick:
        config.num_trials = 3
        config.num_warmup = 1

    print("="*60)
    print("Portable Hardware Calibration Benchmark")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    # Detect all devices
    print("\nDetecting hardware...")
    device_info = detect_all_devices(cpu_only=args.cpu_only)

    print(f"\nCPU: {device_info['cpu']['name']}")
    print(f"     {device_info['cpu']['physical_cores']} cores, {device_info['cpu']['logical_cores']} threads")

    if device_info['gpus']:
        for gpu in device_info['gpus']:
            print(f"GPU: {gpu['name']} ({gpu['type']}, {gpu.get('memory_gb', 0):.1f} GB)")

    if device_info['npus']:
        for npu in device_info['npus']:
            print(f"NPU: {npu['name']} ({npu['type']})")

    # Determine which devices to benchmark
    devices_to_benchmark = []

    if args.device:
        # Specific device requested
        device_type = 'gpu' if 'cuda' in args.device or 'rocm' in args.device else 'cpu' if args.device == 'cpu' else 'npu'
        devices_to_benchmark.append((args.device, args.device, device_type))
    elif args.cpu_only:
        # CPU only
        devices_to_benchmark.append(('cpu', device_info['cpu']['name'], 'cpu'))
    elif args.all_devices:
        # All devices
        devices_to_benchmark.append(('cpu', device_info['cpu']['name'], 'cpu'))
        for gpu in device_info['gpus']:
            device_str = f"{gpu['type']}:{gpu['index']}" if gpu['type'] != 'cuda' else f"cuda:{gpu['index']}"
            devices_to_benchmark.append((device_str, gpu['name'], 'gpu'))
        for npu in device_info['npus']:
            devices_to_benchmark.append((npu['type'], npu['name'], 'npu'))
    else:
        # Default: CPU only to avoid iGPU contamination
        print("\n** Running CPU-only by default to avoid iGPU contamination **")
        print("   Use --device cuda:0 for GPU or --all-devices for all")
        devices_to_benchmark.append(('cpu', device_info['cpu']['name'], 'cpu'))

    # Match hardware against registry - REQUIRED
    print("\nMatching hardware to registry...")
    matched_hardware = {}
    for device, device_name, device_type in devices_to_benchmark:
        try:
            hw_id, hw_name, confidence = match_hardware_to_registry(device_info, device_type)
            matched_hardware[device] = {
                'hardware_id': hw_id,
                'hardware_name': hw_name,
                'confidence': confidence,
            }
            print(f"  {device_type.upper()}: Matched to '{hw_id}' ({confidence*100:.0f}% confidence)")
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            return 1

    print()

    # Run benchmarks
    all_results = {}

    for device, device_name, device_type in devices_to_benchmark:
        if device == 'directml':
            result = benchmark_npu_directml(config)
            if result:
                all_results[device] = result
        else:
            # Normalize device string for PyTorch
            torch_device = device
            if device.startswith('rocm:'):
                torch_device = f"cuda:{device.split(':')[1]}"  # ROCm uses cuda API

            result = benchmark_device(torch_device, device_name, config)
            all_results[device] = result

    # Generate output
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for device, results in all_results.items():
        print(f"\n{results['device_name']}:")
        for precision, gflops in results.get('best_gflops', {}).items():
            if gflops > 0:
                print(f"  Peak {precision.upper()}: {gflops:.2f} GFLOPS")
        if results.get('best_bandwidth_gbps', 0) > 0:
            print(f"  Peak Bandwidth: {results['best_bandwidth_gbps']:.2f} GB/s")

    # Save calibration for primary device
    primary_device = devices_to_benchmark[0][0]
    primary_device_type = devices_to_benchmark[0][2]
    primary_results = all_results.get(primary_device, list(all_results.values())[0] if all_results else {})
    primary_matched = matched_hardware[primary_device]

    calibration = generate_calibration_json(
        primary_results,
        device_info,
        primary_matched,
    )

    # Get hardware_id from matched hardware
    hardware_id = primary_matched['hardware_id']

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"{hardware_id}_calibration.json")

    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: {output_path}")
    print(f"Hardware ID: {hardware_id}")
    print(f"Match confidence: {primary_matched['confidence']*100:.0f}%")
    print("\nTo integrate with hardware registry:")
    print(f"  1. Copy to: hardware_registry/{primary_device_type}/{hardware_id}/calibration.json")
    print("  2. Run: python cli/calibration_coverage.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
