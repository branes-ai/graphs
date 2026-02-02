"""
NumPy-based CPU Benchmarks

Pure NumPy implementations for CPU performance characterization.
Important for understanding performance of signal processing and sensor fusion
operators in Embodied AI applications that use NumPy.

These benchmarks:
- Run on CPU only (NumPy has no GPU support)
- Reflect real-world NumPy performance in production applications
- Are useful for CPU calibration and understanding CPU bottlenecks
"""

from .matmul_bench import calibrate_matmul_numpy
from .memory_bench import calibrate_memory_bandwidth_numpy, calibrate_stream_bandwidth_numpy
from .blas_bench import calibrate_blas_suite_numpy
from .multicore_stream import benchmark_multicore_stream

__all__ = [
    'calibrate_matmul_numpy',
    'calibrate_memory_bandwidth_numpy',
    'calibrate_stream_bandwidth_numpy',
    'calibrate_blas_suite_numpy',
    'benchmark_multicore_stream',
]
