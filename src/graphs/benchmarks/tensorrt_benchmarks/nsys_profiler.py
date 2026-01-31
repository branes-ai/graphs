"""
Nsight Systems profiling for DLA/GPU utilization measurement.

Captures per-resource busy times by running TensorRT inference under
nsys and parsing the resulting trace. This enables computing the
Utilization (U) component of XUE metrics.

Requirements:
    - nsys CLI available on PATH (installed with JetPack or CUDA toolkit)
    - TensorRT engine or ONNX model to profile
"""

import os
import csv
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any


def check_nsys_available() -> bool:
    """Check if nsys CLI is available."""
    try:
        result = subprocess.run(
            ['nsys', '--version'],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_nsys_version() -> Optional[str]:
    """Get nsys version string."""
    try:
        result = subprocess.run(
            ['nsys', '--version'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Output like: "NVIDIA Nsight Systems version 2023.2.1.122-32377109v0"
            for line in result.stdout.splitlines():
                if 'version' in line.lower():
                    return line.strip()
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _write_inference_script(
    onnx_path: str,
    input_shape: Tuple[int, ...],
    dla_core: int,
    precision: str,
    gpu_fallback: bool,
    iterations: int,
    script_path: str,
):
    """Write a standalone Python script that runs TRT inference.

    nsys profiles this script as a subprocess to capture GPU/DLA traces.
    """
    script = f'''#!/usr/bin/env python3
"""Auto-generated TRT inference script for nsys profiling."""
import sys
sys.path.insert(0, {str(Path(__file__).parent.parent.parent.parent)!r})

from graphs.benchmarks.tensorrt_benchmarks.trt_utils import (
    build_engine_from_onnx, time_engine,
)

engine = build_engine_from_onnx(
    {onnx_path!r},
    dla_core={dla_core},
    precision={precision!r},
    gpu_fallback={gpu_fallback},
    input_shape={input_shape!r},
)

# Warmup
time_engine(engine, warmup=5, iterations=5)

# Timed iterations (these are what nsys captures)
time_engine(engine, warmup=0, iterations={iterations})
'''
    with open(script_path, 'w') as f:
        f.write(script)


def profile_with_nsys(
    onnx_path: str,
    input_shape: Tuple[int, ...],
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    iterations: int = 50,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Run TRT inference under nsys and return path to .sqlite trace.

    Args:
        onnx_path: Path to ONNX model file.
        input_shape: Input tensor shape including batch dimension.
        dla_core: DLA core to target (-1 for GPU-only).
        precision: "fp16" or "int8".
        gpu_fallback: Allow GPU fallback for unsupported layers.
        iterations: Number of inference iterations to profile.
        output_dir: Directory for trace output (uses temp if None).

    Returns:
        Path to .sqlite trace file, or None if profiling failed.
    """
    if not check_nsys_available():
        raise RuntimeError("nsys not found. Install NVIDIA Nsight Systems.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='nsys_dla_')

    trace_base = os.path.join(output_dir, 'dla_trace')
    script_path = os.path.join(output_dir, 'run_inference.py')

    _write_inference_script(
        onnx_path, input_shape, dla_core, precision,
        gpu_fallback, iterations, script_path,
    )

    # Run nsys profile
    # --trace=cuda,nvtx captures CUDA API + kernel launches
    # --export=sqlite gives us a queryable database
    cmd = [
        'nsys', 'profile',
        '--trace=cuda,nvtx',
        '--export=sqlite',
        f'--output={trace_base}',
        '--force-overwrite=true',
        'python3', script_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=300,  # 5 min max
        )
        if result.returncode != 0:
            print(f"nsys profile failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("nsys profile timed out (5 min)")
        return None

    sqlite_path = trace_base + '.sqlite'
    if not os.path.exists(sqlite_path):
        # nsys may add a numeric suffix
        for candidate in Path(output_dir).glob('dla_trace*.sqlite'):
            sqlite_path = str(candidate)
            break

    if os.path.exists(sqlite_path):
        return sqlite_path

    return None


def parse_nsys_trace(trace_path: str) -> Dict[str, Any]:
    """Parse nsys .sqlite trace and extract per-resource busy times.

    Args:
        trace_path: Path to .sqlite trace file from nsys.

    Returns:
        Dict with:
            gpu_busy_ms: Total GPU kernel execution time (ms)
            dla_busy_ms: Total DLA submission time (ms) (if available)
            total_wall_ms: Wall-clock duration of profiled region (ms)
            gpu_kernel_count: Number of GPU kernels
            gpu_kernels: List of (name, duration_ms) for top kernels
    """
    result = {
        'gpu_busy_ms': 0.0,
        'dla_busy_ms': 0.0,
        'total_wall_ms': 0.0,
        'gpu_kernel_count': 0,
        'gpu_kernels': [],
    }

    if not os.path.exists(trace_path):
        return result

    try:
        conn = sqlite3.connect(trace_path)
        cursor = conn.cursor()

        # Get list of tables to understand trace structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # GPU kernel execution times from CUPTI_ACTIVITY_KIND_KERNEL
        if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
            cursor.execute("""
                SELECT shortName, SUM(end - start) as total_ns,
                       COUNT(*) as count
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                GROUP BY shortName
                ORDER BY total_ns DESC
            """)
            rows = cursor.fetchall()
            total_gpu_ns = 0
            kernel_count = 0
            kernels = []
            for name, total_ns, count in rows:
                total_gpu_ns += total_ns
                kernel_count += count
                kernels.append((name, total_ns / 1e6, count))  # name, ms, count

            result['gpu_busy_ms'] = total_gpu_ns / 1e6
            result['gpu_kernel_count'] = kernel_count
            result['gpu_kernels'] = kernels[:20]  # Top 20

        # DLA submissions - look for DLA-related activity
        # On Jetson, DLA work may appear as NVDLA or custom activity
        for table in tables:
            if 'DLA' in table.upper() or 'NVDLA' in table.upper():
                try:
                    cursor.execute(f"""
                        SELECT SUM(end - start) as total_ns
                        FROM "{table}"
                    """)
                    row = cursor.fetchone()
                    if row and row[0]:
                        result['dla_busy_ms'] += row[0] / 1e6
                except sqlite3.OperationalError:
                    pass  # Table may not have start/end columns

        # Wall clock from first to last activity
        # Use CUDA runtime API calls as timeline bounds
        if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
            cursor.execute("""
                SELECT MIN(start), MAX(end)
                FROM CUPTI_ACTIVITY_KIND_RUNTIME
            """)
            row = cursor.fetchone()
            if row and row[0] and row[1]:
                result['total_wall_ms'] = (row[1] - row[0]) / 1e6

        conn.close()

    except sqlite3.Error as e:
        print(f"Error parsing nsys trace: {e}")

    return result


def parse_nsys_csv(trace_path: str) -> Dict[str, Any]:
    """Alternative parser using nsys stats CSV export.

    Falls back to this if sqlite parsing is insufficient.
    """
    result = {
        'gpu_busy_ms': 0.0,
        'dla_busy_ms': 0.0,
        'total_wall_ms': 0.0,
        'gpu_kernel_count': 0,
    }

    try:
        # Export GPU kernel summary
        csv_path = trace_path.replace('.sqlite', '_kernels.csv')
        subprocess.run(
            [
                'nsys', 'stats',
                '--report', 'cuda_gpu_kern_sum',
                '--format', 'csv',
                f'--output={csv_path}',
                trace_path.replace('.sqlite', '.nsys-rep'),
            ],
            capture_output=True, text=True, timeout=60,
        )

        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # nsys CSV has columns: Time(%), Total Time (ns), Instances, ...
                    total_ns = float(row.get('Total Time (ns)', 0))
                    instances = int(row.get('Instances', 0))
                    result['gpu_busy_ms'] += total_ns / 1e6
                    result['gpu_kernel_count'] += instances
            os.unlink(csv_path)

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"nsys stats export failed: {e}")

    return result


def compute_utilization(
    gpu_busy_ms: float,
    dla_busy_ms: float,
    total_latency_ms: float,
    num_resources: int = 2,
) -> float:
    """Compute utilization U = sum(busy) / (latency * nResources).

    Args:
        gpu_busy_ms: GPU kernel execution time in ms.
        dla_busy_ms: DLA execution time in ms.
        total_latency_ms: Total wall-clock inference latency in ms.
        num_resources: Number of compute resources (DLA + GPU = 2).

    Returns:
        Utilization as fraction (0.0 to 1.0).
    """
    if total_latency_ms <= 0 or num_resources <= 0:
        return 0.0

    total_busy = gpu_busy_ms + dla_busy_ms
    return min(total_busy / (total_latency_ms * num_resources), 1.0)


def profile_model_utilization(
    onnx_path: str,
    input_shape: Tuple[int, ...],
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    total_latency_ms: float = 0.0,
    iterations: int = 50,
) -> Dict[str, Any]:
    """Profile a model and compute utilization metrics.

    Convenience function that runs nsys profiling and computes U.

    Args:
        onnx_path: Path to ONNX model.
        input_shape: Input shape including batch.
        dla_core: DLA core to target.
        precision: "fp16" or "int8".
        gpu_fallback: Allow GPU fallback.
        total_latency_ms: Pre-measured inference latency per iteration.
        iterations: Iterations to profile.

    Returns:
        Dict with gpu_busy_ms, dla_busy_ms, utilization, and trace details.
    """
    trace_path = profile_with_nsys(
        onnx_path, input_shape,
        dla_core=dla_core, precision=precision,
        gpu_fallback=gpu_fallback, iterations=iterations,
    )

    if trace_path is None:
        return {
            'gpu_busy_ms': None,
            'dla_busy_ms': None,
            'utilization': None,
            'error': 'nsys profiling failed',
        }

    trace_data = parse_nsys_trace(trace_path)

    # Compute per-iteration busy times
    if iterations > 0:
        gpu_per_iter = trace_data['gpu_busy_ms'] / iterations
        dla_per_iter = trace_data['dla_busy_ms'] / iterations
    else:
        gpu_per_iter = 0.0
        dla_per_iter = 0.0

    # Use measured latency if provided, otherwise derive from trace
    latency = total_latency_ms
    if latency <= 0 and trace_data['total_wall_ms'] > 0:
        latency = trace_data['total_wall_ms'] / iterations

    # Determine number of active resources
    num_resources = 1  # At least GPU
    if dla_core >= 0:
        num_resources = 2  # DLA + GPU

    utilization = compute_utilization(
        gpu_per_iter, dla_per_iter, latency, num_resources,
    )

    # Clean up trace files
    trace_dir = os.path.dirname(trace_path)
    try:
        for f in Path(trace_dir).glob('dla_trace*'):
            f.unlink()
        for f in Path(trace_dir).glob('run_inference.py'):
            f.unlink()
        os.rmdir(trace_dir)
    except OSError:
        pass

    return {
        'gpu_busy_ms': gpu_per_iter,
        'dla_busy_ms': dla_per_iter,
        'utilization': utilization,
        'gpu_kernel_count': trace_data['gpu_kernel_count'],
        'top_kernels': trace_data.get('gpu_kernels', [])[:5],
    }
