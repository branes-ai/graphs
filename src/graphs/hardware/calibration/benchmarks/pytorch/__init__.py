"""
PyTorch-based Benchmarks (CPU or GPU)

PyTorch implementations that can leverage GPU acceleration for performance
characterization. These benchmarks:
- Can run on CPU or CUDA devices
- Use optimized CUDA kernels when available
- Reflect PyTorch/DL framework performance (not NumPy signal processing)
- Are essential for GPU calibration

For Embodied AI applications using PyTorch models, these benchmarks represent
actual inference performance. For signal processing/sensor fusion using NumPy,
use the numpy/ benchmarks instead.
"""

from .matmul_bench import calibrate_matmul_pytorch
from .memory_bench import calibrate_memory_bandwidth_pytorch

__all__ = [
    'calibrate_matmul_pytorch',
    'calibrate_memory_bandwidth_pytorch',
]
