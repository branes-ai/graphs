"""
GPU Clock Query Module

Provides functions to query current GPU clock frequencies for accurate
calibration and efficiency calculations.

Supports:
- NVIDIA GPUs via nvidia-smi
- NVIDIA Jetson via tegrastats/jtop
- Fallback to PyTorch CUDA properties

Usage:
    from graphs.hardware.calibration.gpu_clock import get_gpu_clock_info

    clock_info = get_gpu_clock_info()
    print(f"SM Clock: {clock_info['sm_clock_mhz']} MHz")
    print(f"Memory Clock: {clock_info['mem_clock_mhz']} MHz")
"""

import subprocess
import re
import os
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class GPUClockInfo:
    """GPU clock frequency information."""

    # Current clock frequencies
    sm_clock_mhz: Optional[int] = None
    """Current SM/Graphics clock in MHz."""

    mem_clock_mhz: Optional[int] = None
    """Current memory clock in MHz."""

    # Maximum clock frequencies
    max_sm_clock_mhz: Optional[int] = None
    """Maximum SM/Graphics clock in MHz."""

    max_mem_clock_mhz: Optional[int] = None
    """Maximum memory clock in MHz."""

    # Power and thermal
    power_draw_watts: Optional[float] = None
    """Current power draw in Watts."""

    power_limit_watts: Optional[float] = None
    """Power limit in Watts."""

    temperature_c: Optional[int] = None
    """GPU temperature in Celsius."""

    # Jetson-specific
    nvpmodel_mode: Optional[int] = None
    """Jetson NVPModel power mode (0=MAXN, 1=15W, 2=10W, etc.)."""

    power_mode_name: Optional[str] = None
    """Human-readable power mode name (e.g., 'MAXN', '15W', '10W')."""

    # Query metadata
    query_method: str = "unknown"
    """Method used to query clocks: 'nvidia-smi', 'tegrastats', 'pynvml', 'torch'."""

    query_success: bool = False
    """Whether clock query was successful."""

    error_message: Optional[str] = None
    """Error message if query failed."""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'GPUClockInfo':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _query_nvidia_smi() -> Optional[GPUClockInfo]:
    """
    Query GPU clocks using nvidia-smi.

    Works on datacenter and desktop NVIDIA GPUs.
    May have limited functionality on Jetson.
    """
    try:
        # Query current clocks, max clocks, power, and temperature
        cmd = [
            'nvidia-smi',
            '--query-gpu=clocks.current.sm,clocks.current.memory,'
            'clocks.max.sm,clocks.max.memory,'
            'power.draw,power.limit,temperature.gpu',
            '--format=csv,noheader,nounits'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            return None

        # Parse CSV output
        line = result.stdout.strip()
        if not line:
            return None

        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            return None

        def safe_int(s):
            try:
                return int(float(s))
            except (ValueError, TypeError):
                return None

        def safe_float(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        return GPUClockInfo(
            sm_clock_mhz=safe_int(parts[0]),
            mem_clock_mhz=safe_int(parts[1]),
            max_sm_clock_mhz=safe_int(parts[2]),
            max_mem_clock_mhz=safe_int(parts[3]),
            power_draw_watts=safe_float(parts[4]),
            power_limit_watts=safe_float(parts[5]),
            temperature_c=safe_int(parts[6]),
            query_method='nvidia-smi',
            query_success=True
        )

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return GPUClockInfo(
            query_method='nvidia-smi',
            query_success=False,
            error_message=str(e)
        )


def _query_jetson_nvpmodel() -> Optional[Dict]:
    """
    Query Jetson power mode using nvpmodel.

    Returns dict with nvpmodel_mode and power_mode_name.
    """
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        # Parse output like: "NV Power Mode: MAXN"
        output = result.stdout
        mode_match = re.search(r'NV Power Mode:\s*(\S+)', output)
        id_match = re.search(r'(\d+)', output)

        if mode_match:
            return {
                'power_mode_name': mode_match.group(1),
                'nvpmodel_mode': int(id_match.group(1)) if id_match else None
            }

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def _query_jetson_tegrastats() -> Optional[GPUClockInfo]:
    """
    Query Jetson GPU clocks using tegrastats.

    tegrastats provides real-time GPU frequency on Jetson platforms.
    """
    try:
        # tegrastats outputs continuously, so we just get one sample
        result = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True,
            text=True,
            timeout=1
        )

        # tegrastats may timeout (expected) but still have output
        output = result.stdout or result.stderr

        if not output:
            return None

        # Parse GR3D_FREQ line like: GR3D_FREQ 76%@1300
        # or newer format: GR3D 0%@1300
        freq_match = re.search(r'GR3D(?:_FREQ)?\s+\d+%@(\d+)', output)

        info = GPUClockInfo(
            query_method='tegrastats',
            query_success=bool(freq_match)
        )

        if freq_match:
            info.sm_clock_mhz = int(freq_match.group(1))

        # Try to get nvpmodel info
        nvp = _query_jetson_nvpmodel()
        if nvp:
            info.nvpmodel_mode = nvp.get('nvpmodel_mode')
            info.power_mode_name = nvp.get('power_mode_name')

        return info

    except subprocess.TimeoutExpired:
        # Expected - tegrastats runs continuously
        # Try to parse any output we got
        return None
    except (FileNotFoundError, Exception) as e:
        return GPUClockInfo(
            query_method='tegrastats',
            query_success=False,
            error_message=str(e)
        )


def _query_jetson_sysfs() -> Optional[GPUClockInfo]:
    """
    Query Jetson GPU clocks from sysfs.

    This is a reliable fallback that works on all Jetson platforms.
    """
    try:
        info = GPUClockInfo(query_method='sysfs')

        # Try different sysfs paths for different Jetson models
        clock_paths = [
            '/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq',  # Orin
            '/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq',   # Xavier
            '/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq',  # Orin alt
        ]

        max_clock_paths = [
            '/sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq',
            '/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq',
            '/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq',
        ]

        # Try to read current frequency
        for path in clock_paths:
            if os.path.exists(path):
                with open(path) as f:
                    freq_hz = int(f.read().strip())
                    info.sm_clock_mhz = freq_hz // 1_000_000
                    info.query_success = True
                break

        # Try to read max frequency
        for path in max_clock_paths:
            if os.path.exists(path):
                with open(path) as f:
                    freq_hz = int(f.read().strip())
                    info.max_sm_clock_mhz = freq_hz // 1_000_000
                break

        # Get nvpmodel info
        nvp = _query_jetson_nvpmodel()
        if nvp:
            info.nvpmodel_mode = nvp.get('nvpmodel_mode')
            info.power_mode_name = nvp.get('power_mode_name')

        return info if info.query_success else None

    except Exception as e:
        return GPUClockInfo(
            query_method='sysfs',
            query_success=False,
            error_message=str(e)
        )


def _query_torch_cuda() -> Optional[GPUClockInfo]:
    """
    Query GPU info using PyTorch CUDA.

    This is a fallback that provides limited clock info.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        # PyTorch doesn't provide current clock, only some device properties
        # We can estimate from device properties
        info = GPUClockInfo(
            query_method='torch',
            query_success=True
        )

        # Note: torch doesn't give current clock, but we can try nvidia-smi first
        # This is mainly for device validation

        return info

    except Exception as e:
        return GPUClockInfo(
            query_method='torch',
            query_success=False,
            error_message=str(e)
        )


def _is_jetson() -> bool:
    """Check if running on NVIDIA Jetson platform."""
    # Check for Jetson-specific files
    jetson_indicators = [
        '/etc/nv_tegra_release',
        '/sys/devices/gpu.0',
        '/usr/bin/tegrastats',
    ]

    for indicator in jetson_indicators:
        if os.path.exists(indicator):
            return True

    # Check platform
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            if 'jetson' in model or 'orin' in model:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    return False


def get_gpu_clock_info(device_index: int = 0) -> GPUClockInfo:
    """
    Query current GPU clock frequencies.

    Tries multiple methods in order of preference:
    1. nvidia-smi (datacenter/desktop GPUs)
    2. Jetson sysfs (Jetson platforms)
    3. tegrastats (Jetson platforms, alternative)
    4. PyTorch CUDA (fallback)

    Args:
        device_index: GPU device index (default 0)

    Returns:
        GPUClockInfo with available clock data

    Example:
        >>> info = get_gpu_clock_info()
        >>> print(f"SM Clock: {info.sm_clock_mhz} MHz")
        >>> print(f"Power Mode: {info.power_mode_name}")
    """
    is_jetson = _is_jetson()

    # For Jetson, prefer sysfs first (most reliable)
    if is_jetson:
        info = _query_jetson_sysfs()
        if info and info.query_success:
            return info

        # Try tegrastats as backup
        info = _query_jetson_tegrastats()
        if info and info.query_success:
            return info

    # Try nvidia-smi (works on most NVIDIA GPUs)
    info = _query_nvidia_smi()
    if info and info.query_success:
        # On Jetson, add nvpmodel info if available
        if is_jetson:
            nvp = _query_jetson_nvpmodel()
            if nvp:
                info.nvpmodel_mode = nvp.get('nvpmodel_mode')
                info.power_mode_name = nvp.get('power_mode_name')
        return info

    # Fallback to PyTorch
    info = _query_torch_cuda()
    if info and info.query_success:
        return info

    # Return failure info
    return GPUClockInfo(
        query_method='none',
        query_success=False,
        error_message='No GPU clock query method available'
    )


def get_jetson_power_mode() -> Optional[str]:
    """
    Get current Jetson power mode name.

    Returns:
        Power mode name (e.g., 'MAXN', '15W', '10W') or None

    Example:
        >>> mode = get_jetson_power_mode()
        >>> print(f"Current mode: {mode}")  # e.g., "MAXN"
    """
    nvp = _query_jetson_nvpmodel()
    return nvp.get('power_mode_name') if nvp else None


def estimate_theoretical_peak(
    cuda_cores: int,
    tensor_cores: int,
    clock_mhz: int,
    precision: str = 'fp32'
) -> Dict[str, float]:
    """
    Estimate theoretical peak performance at given clock.

    Uses correct precision naming convention:
    - fp64/fp32/fp16: IEEE 754 floating-point (CUDA cores)
    - tf32: NVIDIA TensorFloat-32, 19-bit (1+8+10), Tensor Cores only
    - bf16: Brain Float16, 16-bit (1+8+7)

    IMPORTANT: NVIDIA markets "FP32 Tensor Core" performance, but Tensor Cores
    actually use TF32 (19-bit) for these operations, not IEEE FP32 (32-bit).
    This function correctly reports tf32 for Tensor Core "FP32" operations.

    Args:
        cuda_cores: Number of CUDA cores
        tensor_cores: Number of Tensor Cores
        clock_mhz: GPU clock frequency in MHz
        precision: Precision to calculate for:
            - 'fp64': IEEE double precision (CUDA cores, 1/32 rate on Ampere)
            - 'fp32': IEEE single precision (CUDA cores only)
            - 'fp16': IEEE half precision
            - 'tf32': NVIDIA TensorFloat-32 (Tensor Cores, 19-bit)
            - 'bf16': Brain Float16
            - 'int32', 'int8', 'int4': Integer precisions

    Returns:
        Dict with performance estimates:
            - 'cuda_core_gflops': CUDA core theoretical peak
            - 'tensor_core_gflops': Tensor Core theoretical peak
            - 'clock_mhz': Input clock frequency
            - 'precision': Input precision
            - 'precision_bits': Actual bit width used in computation

    Example:
        >>> # For IEEE FP32 (CUDA cores only):
        >>> peaks = estimate_theoretical_peak(1024, 32, 1300, 'fp32')
        >>> print(f"CUDA cores FP32: {peaks['cuda_core_gflops']:.0f} GFLOPS")
        2662 GFLOPS

        >>> # For TF32 (Tensor Cores, what NVIDIA calls "FP32 Tensor"):
        >>> peaks = estimate_theoretical_peak(1024, 32, 1300, 'tf32')
        >>> print(f"Tensor cores TF32: {peaks['tensor_core_gflops']:.0f} GFLOPS")
        5325 GFLOPS
    """
    clock_ghz = clock_mhz / 1000.0

    # Precision bit widths (computation bits, not storage)
    precision_bits = {
        'fp64': 64,
        'fp32': 32,
        'tf32': 19,  # NOT 32! NVIDIA's misleading name
        'bf16': 16,
        'fp16': 16,
        'fp8': 8,
        'int64': 64,
        'int32': 32,
        'int16': 16,
        'int8': 8,
        'int4': 4,
    }

    # CUDA core calculations (FMA = 2 ops per cycle)
    # These are traditional SIMT operations on IEEE formats
    cuda_core_peak = {
        'fp64': cuda_cores * 2 * clock_ghz / 32,  # 1/32 FP64 rate on Ampere
        'fp32': cuda_cores * 2 * clock_ghz,       # IEEE FP32
        'fp16': cuda_cores * 4 * clock_ghz,       # 2x FP32 with packed FP16
        'bf16': cuda_cores * 4 * clock_ghz,       # 2x FP32 with packed BF16
        'int32': cuda_cores * 2 * clock_ghz,
        'int16': cuda_cores * 4 * clock_ghz,
        'int8': cuda_cores * 4 * clock_ghz,
        # Note: tf32 is NOT available on CUDA cores, only Tensor Cores
        'tf32': 0.0,
    }

    # Tensor Core calculations (Gen 3 Ampere, Gen 4 Hopper)
    # Per TC per cycle (Ampere Gen 3):
    #   - TF32: 128 FMA ops (what NVIDIA misleadingly calls "FP32")
    #   - FP16: 256 FMA ops
    #   - BF16: 256 FMA ops
    #   - INT8: 512 ops
    #   - INT4: 1024 ops
    tc_ops_per_cycle = {
        'tf32': 128,    # 19-bit TensorFloat-32 (NOT IEEE FP32!)
        'fp16': 256,    # IEEE FP16
        'bf16': 256,    # Brain Float16
        'int8': 512,
        'int4': 1024,
        # Note: fp32/fp64 not directly supported on Tensor Cores
        # fp32 inputs are truncated to tf32 (19-bit) for Tensor Core ops
        'fp32': 0.0,    # Use tf32 instead
        'fp64': 0.0,    # Not supported on Tensor Cores
    }

    tc_peak = tc_ops_per_cycle.get(precision, 0) * tensor_cores * clock_ghz

    return {
        'cuda_core_gflops': cuda_core_peak.get(precision, 0),
        'tensor_core_gflops': tc_peak,
        'clock_mhz': clock_mhz,
        'precision': precision,
        'precision_bits': precision_bits.get(precision, 0),
    }


# Convenience function for direct use
def print_gpu_clock_info():
    """Print current GPU clock information to stdout."""
    info = get_gpu_clock_info()

    print("\n" + "=" * 50)
    print("GPU CLOCK INFORMATION")
    print("=" * 50)

    if not info.query_success:
        print(f"Query failed: {info.error_message}")
        print(f"Method tried: {info.query_method}")
        return

    print(f"Query method: {info.query_method}")
    print()

    if info.sm_clock_mhz:
        print(f"SM Clock:     {info.sm_clock_mhz} MHz", end="")
        if info.max_sm_clock_mhz:
            pct = info.sm_clock_mhz / info.max_sm_clock_mhz * 100
            print(f" / {info.max_sm_clock_mhz} MHz ({pct:.0f}%)")
        else:
            print()

    if info.mem_clock_mhz:
        print(f"Memory Clock: {info.mem_clock_mhz} MHz", end="")
        if info.max_mem_clock_mhz:
            pct = info.mem_clock_mhz / info.max_mem_clock_mhz * 100
            print(f" / {info.max_mem_clock_mhz} MHz ({pct:.0f}%)")
        else:
            print()

    if info.power_draw_watts:
        print(f"Power:        {info.power_draw_watts:.1f} W", end="")
        if info.power_limit_watts:
            print(f" / {info.power_limit_watts:.1f} W limit")
        else:
            print()

    if info.temperature_c:
        print(f"Temperature:  {info.temperature_c}°C")

    if info.power_mode_name:
        print(f"Power Mode:   {info.power_mode_name}", end="")
        if info.nvpmodel_mode is not None:
            print(f" (nvpmodel {info.nvpmodel_mode})")
        else:
            print()

    print()


if __name__ == '__main__':
    print_gpu_clock_info()
