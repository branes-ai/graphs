"""
Hardware and Software Auto-Detection Module

Provides self-organizing hardware identification and software stack versioning
for reproducible, comparable calibration runs.

Key Principles:
1. NO user-provided hardware IDs - everything is auto-detected
2. Deterministic fingerprints - same hardware = same fingerprint
3. Complete software stack capture - enables regression analysis
4. Append-only storage - every run creates a new record

Usage:
    from graphs.hardware.calibration.auto_detect import (
        HardwareIdentity, SoftwareStack, CalibrationContext
    )

    # Auto-detect everything
    context = CalibrationContext.detect()

    print(f"Hardware: {context.hardware.cpu_model}")
    print(f"HW Fingerprint: {context.hardware.fingerprint}")
    print(f"SW Fingerprint: {context.software.fingerprint}")
"""

import hashlib
import os
import platform
import re
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# HARDWARE IDENTITY
# =============================================================================

@dataclass
class CPUIdentity:
    """Auto-detected CPU identity."""
    model: str                      # "Intel(R) Core(TM) i7-12700K"
    vendor: str                     # "GenuineIntel", "AuthenticAMD"
    family: int                     # CPU family
    model_id: int                   # CPU model number
    stepping: int                   # Silicon stepping/revision
    microcode: str                  # Microcode version (e.g., "0x2c")
    cores_physical: int             # Physical cores
    cores_logical: int              # Logical cores (with HT/SMT)
    base_freq_mhz: int              # Base frequency
    max_freq_mhz: int               # Max turbo frequency
    cache_l1d_kb: int = 0           # L1 data cache
    cache_l1i_kb: int = 0           # L1 instruction cache
    cache_l2_kb: int = 0            # L2 cache
    cache_l3_kb: int = 0            # L3 cache
    flags: List[str] = field(default_factory=list)  # CPU flags (avx, avx2, avx512, etc.)

    @classmethod
    def detect(cls) -> 'CPUIdentity':
        """Auto-detect CPU identity from system."""
        info = {
            'model': 'Unknown CPU',
            'vendor': 'Unknown',
            'family': 0,
            'model_id': 0,
            'stepping': 0,
            'microcode': '0x0',
            'cores_physical': 1,
            'cores_logical': 1,
            'base_freq_mhz': 0,
            'max_freq_mhz': 0,
            'cache_l1d_kb': 0,
            'cache_l1i_kb': 0,
            'cache_l2_kb': 0,
            'cache_l3_kb': 0,
            'flags': [],
        }

        # Try /proc/cpuinfo (Linux)
        cpuinfo_path = Path('/proc/cpuinfo')
        if cpuinfo_path.exists():
            info.update(cls._parse_proc_cpuinfo(cpuinfo_path))

        # Try lscpu for additional info
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info.update(cls._parse_lscpu(result.stdout))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try psutil for core counts
        try:
            import psutil
            info['cores_physical'] = psutil.cpu_count(logical=False) or info['cores_physical']
            info['cores_logical'] = psutil.cpu_count(logical=True) or info['cores_logical']
        except ImportError:
            pass

        # Try py-cpuinfo for detailed info
        try:
            import cpuinfo
            cpu_data = cpuinfo.get_cpu_info()
            if cpu_data:
                info['model'] = cpu_data.get('brand_raw', info['model'])
                info['vendor'] = cpu_data.get('vendor_id_raw', info['vendor'])
                info['flags'] = cpu_data.get('flags', info['flags'])
                if 'hz_advertised_friendly' in cpu_data:
                    hz_str = cpu_data['hz_advertised_friendly']
                    if 'GHz' in hz_str:
                        try:
                            ghz = float(hz_str.replace('GHz', '').strip())
                            info['base_freq_mhz'] = int(ghz * 1000)
                        except ValueError:
                            pass
        except ImportError:
            pass

        return cls(**info)

    @staticmethod
    def _parse_proc_cpuinfo(path: Path) -> Dict[str, Any]:
        """Parse /proc/cpuinfo for CPU details."""
        result = {}
        try:
            content = path.read_text()
            lines = content.split('\n')

            for line in lines:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if key == 'model name':
                    result['model'] = value
                elif key == 'vendor_id':
                    result['vendor'] = value
                elif key == 'cpu family':
                    result['family'] = int(value)
                elif key == 'model':
                    result['model_id'] = int(value)
                elif key == 'stepping':
                    result['stepping'] = int(value)
                elif key == 'microcode':
                    result['microcode'] = value
                elif key == 'flags':
                    result['flags'] = value.split()
                elif key == 'cache size':
                    # Usually L3 cache
                    match = re.match(r'(\d+)\s*KB', value, re.IGNORECASE)
                    if match:
                        result['cache_l3_kb'] = int(match.group(1))

        except Exception:
            pass

        return result

    @staticmethod
    def _parse_lscpu(output: str) -> Dict[str, Any]:
        """Parse lscpu output for CPU details."""
        result = {}
        try:
            for line in output.split('\n'):
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                if 'core(s) per socket' in key:
                    result['cores_physical'] = int(value)
                elif 'thread(s) per core' in key:
                    threads_per_core = int(value)
                    if 'cores_physical' in result:
                        result['cores_logical'] = result['cores_physical'] * threads_per_core
                elif 'cpu max mhz' in key:
                    result['max_freq_mhz'] = int(float(value))
                elif 'cpu mhz' in key and 'max' not in key and 'min' not in key:
                    result['base_freq_mhz'] = int(float(value))
                elif 'l1d cache' in key:
                    match = re.search(r'(\d+)', value)
                    if match:
                        result['cache_l1d_kb'] = int(match.group(1))
                elif 'l1i cache' in key:
                    match = re.search(r'(\d+)', value)
                    if match:
                        result['cache_l1i_kb'] = int(match.group(1))
                elif 'l2 cache' in key:
                    match = re.search(r'(\d+)', value)
                    if match:
                        result['cache_l2_kb'] = int(match.group(1))
                elif 'l3 cache' in key:
                    match = re.search(r'(\d+)', value)
                    if match:
                        result['cache_l3_kb'] = int(match.group(1))

        except Exception:
            pass

        return result

    def estimate_theoretical_peak_gflops(self, precision: str = "fp32") -> float:
        """
        Estimate theoretical peak GFLOPS based on CPU specs.

        Uses core count, max frequency, and SIMD capabilities to estimate
        the theoretical peak compute throughput.

        Args:
            precision: Precision to estimate for ('fp32', 'fp64', 'int8', etc.)

        Returns:
            Estimated peak GFLOPS (or GOPS for integer).
        """
        # Determine SIMD width from flags
        # FMA = Fused Multiply-Add, counts as 2 ops (multiply + add)
        # Most modern CPUs have 2 FMA units per core
        flags_set = set(self.flags) if self.flags else set()

        # FP32 ops per cycle per core
        if 'avx512f' in flags_set:
            # AVX-512: 512-bit = 16 FP32, x2 for FMA, x2 units = 64 ops/cycle
            # But many CPUs downclock with AVX-512, use conservative estimate
            fp32_ops_per_cycle = 32  # Conservative: 1 FMA unit
        elif 'avx2' in flags_set and 'fma' in flags_set:
            # AVX2 + FMA: 256-bit = 8 FP32, x2 for FMA, x2 units = 32 ops/cycle
            fp32_ops_per_cycle = 32
        elif 'avx2' in flags_set:
            # AVX2 without FMA: 256-bit = 8 FP32, x2 units = 16 ops/cycle
            fp32_ops_per_cycle = 16
        elif 'avx' in flags_set:
            # AVX: 256-bit = 8 FP32, typically 1 unit = 8 ops/cycle
            fp32_ops_per_cycle = 8
        elif 'sse4_1' in flags_set or 'sse4_2' in flags_set:
            # SSE4: 128-bit = 4 FP32
            fp32_ops_per_cycle = 4
        else:
            # Fallback: assume basic SSE (128-bit)
            fp32_ops_per_cycle = 4

        # Adjust for precision
        if precision == 'fp64':
            ops_per_cycle = fp32_ops_per_cycle // 2  # Half the width
        elif precision in ('fp16', 'bf16'):
            ops_per_cycle = fp32_ops_per_cycle * 2  # Double the width
        elif precision == 'int8':
            ops_per_cycle = fp32_ops_per_cycle * 4  # 4x the width
        elif precision == 'int32':
            ops_per_cycle = fp32_ops_per_cycle  # Same as FP32
        else:
            ops_per_cycle = fp32_ops_per_cycle  # Default to FP32

        # Use max frequency if available, otherwise base
        freq_ghz = (self.max_freq_mhz or self.base_freq_mhz or 3000) / 1000.0

        # Calculate peak
        # Note: Using physical cores only (logical cores share execution units)
        peak_gflops = self.cores_physical * ops_per_cycle * freq_ghz

        return peak_gflops

    def estimate_theoretical_bandwidth_gbps(self) -> float:
        """
        Estimate theoretical memory bandwidth based on memory type.

        This is a rough estimate - actual bandwidth depends on memory
        configuration which we may not fully detect.

        Returns:
            Estimated memory bandwidth in GB/s.
        """
        # Common memory bandwidths (rough estimates)
        # DDR4-3200: ~50 GB/s per channel, typically 2 channels = 100 GB/s
        # DDR5-4800: ~75 GB/s per channel, typically 2 channels = 150 GB/s
        # This is a conservative estimate for desktop systems
        return 100.0  # Default to ~100 GB/s (DDR4 dual-channel)


@dataclass
class GPUIdentity:
    """Auto-detected GPU identity."""
    model: str                      # "NVIDIA GeForce RTX 4090"
    vendor: str                     # "NVIDIA", "AMD", "Intel"
    pci_id: str                     # "10de:2684" (vendor:device)
    vbios_version: str              # "95.02.18.40.84"
    memory_mb: int                  # GPU memory in MB
    compute_capability: str         # "8.9" (NVIDIA) or equivalent
    driver_version: str             # "560.35.03"
    cuda_cores: int = 0             # CUDA cores (NVIDIA)
    tensor_cores: int = 0           # Tensor cores
    sm_count: int = 0               # Streaming multiprocessors
    clock_mhz: int = 0              # GPU clock (boost) in MHz
    memory_clock_mhz: int = 0       # Memory clock in MHz
    memory_bus_width: int = 0       # Memory bus width in bits

    @classmethod
    def detect(cls) -> Optional['GPUIdentity']:
        """Auto-detect GPU identity. Returns None if no GPU found."""
        # Try NVIDIA first
        gpu = cls._detect_nvidia()
        if gpu:
            return gpu

        # Try AMD ROCm
        gpu = cls._detect_amd()
        if gpu:
            return gpu

        # Try Intel
        gpu = cls._detect_intel()
        if gpu:
            return gpu

        return None

    @classmethod
    def _detect_nvidia(cls) -> Optional['GPUIdentity']:
        """Detect NVIDIA GPU using nvidia-smi."""
        try:
            # Get basic info including clocks and memory bus width
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,pci.bus_id,memory.total,driver_version,vbios_version,clocks.max.graphics,clocks.max.memory,memory.bus_width',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

            line = result.stdout.strip().split('\n')[0]  # First GPU
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                return None

            model = parts[0]
            pci_bus = parts[1]
            memory_mb = int(float(parts[2]))
            driver_version = parts[3]
            vbios_version = parts[4]

            # Parse clock and memory info (may not be available on all GPUs)
            clock_mhz = 0
            memory_clock_mhz = 0
            memory_bus_width = 0
            if len(parts) >= 6 and parts[5] not in ('[Not Supported]', '[N/A]', ''):
                try:
                    clock_mhz = int(float(parts[5]))
                except ValueError:
                    pass
            if len(parts) >= 7 and parts[6] not in ('[Not Supported]', '[N/A]', ''):
                try:
                    memory_clock_mhz = int(float(parts[6]))
                except ValueError:
                    pass
            if len(parts) >= 8 and parts[7] not in ('[Not Supported]', '[N/A]', ''):
                try:
                    memory_bus_width = int(float(parts[7]))
                except ValueError:
                    pass

            # Get PCI ID
            pci_id = cls._get_nvidia_pci_id(pci_bus)

            # Get compute capability
            compute_cap = cls._get_nvidia_compute_capability()

            # Get CUDA cores and SM count
            cuda_cores, sm_count, tensor_cores = cls._get_nvidia_cuda_info()

            return cls(
                model=model,
                vendor='NVIDIA',
                pci_id=pci_id,
                vbios_version=vbios_version,
                memory_mb=memory_mb,
                compute_capability=compute_cap,
                driver_version=driver_version,
                cuda_cores=cuda_cores,
                tensor_cores=tensor_cores,
                sm_count=sm_count,
                clock_mhz=clock_mhz,
                memory_clock_mhz=memory_clock_mhz,
                memory_bus_width=memory_bus_width,
            )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    @staticmethod
    def _get_nvidia_pci_id(pci_bus: str) -> str:
        """Get PCI vendor:device ID for NVIDIA GPU."""
        try:
            # pci_bus is like "00000000:01:00.0"
            # Extract the bus:device.function part
            parts = pci_bus.split(':')
            if len(parts) >= 2:
                bus_device = ':'.join(parts[-2:])
                # Read from sysfs
                vendor_path = Path(f'/sys/bus/pci/devices/{pci_bus}/vendor')
                device_path = Path(f'/sys/bus/pci/devices/{pci_bus}/device')
                if vendor_path.exists() and device_path.exists():
                    vendor = vendor_path.read_text().strip().replace('0x', '')
                    device = device_path.read_text().strip().replace('0x', '')
                    return f"{vendor}:{device}"
        except Exception:
            pass
        return "10de:0000"  # NVIDIA vendor ID, unknown device

    @staticmethod
    def _get_nvidia_compute_capability() -> str:
        """Get NVIDIA compute capability."""
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                return f"{major}.{minor}"
        except Exception:
            pass

        # Fallback: try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "0.0"

    @staticmethod
    def _get_nvidia_cuda_info() -> Tuple[int, int, int]:
        """Get CUDA cores, SM count, and tensor cores."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                sm_count = props.multi_processor_count
                # Estimate CUDA cores and tensor cores based on compute capability
                major, minor = torch.cuda.get_device_capability(0)

                # CUDA cores per SM and tensor cores per SM by architecture
                arch_info = {
                    # (major, minor): (cuda_cores_per_sm, tensor_cores_per_sm)
                    (7, 0): (64, 8),    # Volta V100
                    (7, 5): (64, 8),    # Turing
                    (8, 0): (64, 4),    # Ampere GA100 (A100)
                    (8, 6): (128, 4),   # Ampere GA10x (RTX 30xx)
                    (8, 7): (128, 4),   # Ampere GA10x (Jetson Orin)
                    (8, 9): (128, 4),   # Ada Lovelace (RTX 40xx)
                    (9, 0): (128, 4),   # Hopper (H100)
                }
                cores_per_sm, tc_per_sm = arch_info.get((major, minor), (64, 4))

                cuda_cores = sm_count * cores_per_sm
                tensor_cores = sm_count * tc_per_sm

                return cuda_cores, sm_count, tensor_cores
        except Exception:
            pass
        return 0, 0, 0

    def estimate_theoretical_peak_gflops(self, precision: str = "fp32") -> float:
        """
        Estimate theoretical peak GFLOPS/TOPS for GPU.

        Uses CUDA cores, tensor cores, and clock frequency to estimate peak.

        Args:
            precision: Precision to estimate for ('fp32', 'fp16', 'bf16', 'tf32', 'int8', 'fp64')

        Returns:
            Estimated peak GFLOPS (or GOPS/TOPS for integer/tensor ops).
        """
        if self.vendor != 'NVIDIA':
            # For non-NVIDIA GPUs, return conservative estimate
            return 0.0

        # Use detected clock or estimate from model
        clock_ghz = self.clock_mhz / 1000.0 if self.clock_mhz > 0 else self._estimate_clock_ghz()

        # Parse compute capability
        try:
            major, minor = map(int, self.compute_capability.split('.'))
        except (ValueError, AttributeError):
            major, minor = 8, 0  # Default to Ampere

        # Calculate peak based on precision and architecture
        if precision == 'fp64':
            # FP64: 2 ops/cycle per CUDA core (FMA) but typically 1/2 or 1/32 rate
            if major >= 8:
                # Ampere/Ada/Hopper: 1/64 rate for consumer, 1/2 for datacenter (A100, H100)
                if 'A100' in self.model or 'H100' in self.model or 'A30' in self.model:
                    ops_per_cycle = self.cuda_cores * 1.0  # 1/2 rate with FMA
                else:
                    ops_per_cycle = self.cuda_cores * (2 / 64)  # 1/64 rate
            else:
                ops_per_cycle = self.cuda_cores * 0.5  # 1/2 rate typical

        elif precision == 'fp32':
            # FP32: 2 ops/cycle per CUDA core (FMA)
            ops_per_cycle = self.cuda_cores * 2

        elif precision in ('fp16', 'bf16'):
            # FP16/BF16: 2x rate on CUDA cores + tensor cores
            # Tensor cores: ~256 ops/cycle for FP16 per TC (4x4x4 matrix)
            cuda_ops = self.cuda_cores * 4  # 2x rate with FMA
            tensor_ops = self.tensor_cores * 256 if self.tensor_cores > 0 else 0
            ops_per_cycle = cuda_ops + tensor_ops

        elif precision == 'tf32':
            # TF32 (Ampere+): Tensor cores only, ~128 ops/cycle per TC
            if major >= 8 and self.tensor_cores > 0:
                ops_per_cycle = self.tensor_cores * 128
            else:
                ops_per_cycle = self.cuda_cores * 2  # Fallback to FP32

        elif precision == 'int8':
            # INT8: 4x rate on CUDA cores + tensor cores
            cuda_ops = self.cuda_cores * 8
            tensor_ops = self.tensor_cores * 512 if self.tensor_cores > 0 else 0
            ops_per_cycle = cuda_ops + tensor_ops

        elif precision == 'int4':
            # INT4: 8x rate, tensor cores only on newer architectures
            if major >= 8 and self.tensor_cores > 0:
                ops_per_cycle = self.tensor_cores * 1024
            else:
                ops_per_cycle = self.cuda_cores * 16

        else:
            # Default to FP32
            ops_per_cycle = self.cuda_cores * 2

        peak_gflops = ops_per_cycle * clock_ghz
        return peak_gflops

    def _estimate_clock_ghz(self) -> float:
        """Estimate boost clock from model name if not detected."""
        model_lower = self.model.lower()

        # Common NVIDIA GPU clocks (boost clock in GHz)
        clock_estimates = {
            'h100': 1.98,
            'h200': 1.98,
            'a100': 1.41,
            'a30': 1.44,
            'v100': 1.53,
            'rtx 4090': 2.52,
            'rtx 4080': 2.51,
            'rtx 4070': 2.48,
            'rtx 3090': 1.70,
            'rtx 3080': 1.71,
            'rtx 3070': 1.73,
            'rtx 2080': 1.80,
            't4': 1.59,
            'orin': 1.30,
        }

        for model_key, clock in clock_estimates.items():
            if model_key in model_lower:
                return clock

        return 1.5  # Conservative default

    def estimate_theoretical_bandwidth_gbps(self) -> float:
        """
        Estimate theoretical memory bandwidth in GB/s.

        Uses memory clock and bus width if available, otherwise estimates from model.
        """
        if self.memory_clock_mhz > 0 and self.memory_bus_width > 0:
            # Bandwidth = memory_clock * bus_width * 2 (DDR) / 8 (bits to bytes)
            # For GDDR6X: effective rate is 2x (PAM4)
            # For HBM: effective rate is already accounted for
            if 'hbm' in self.model.lower() or 'a100' in self.model.lower() or 'h100' in self.model.lower():
                # HBM memory: simpler calculation
                bandwidth_gbps = (self.memory_clock_mhz * 2 * self.memory_bus_width) / (8 * 1000)
            else:
                # GDDR6/GDDR6X: 2x for DDR
                bandwidth_gbps = (self.memory_clock_mhz * 2 * self.memory_bus_width) / (8 * 1000)
            return bandwidth_gbps

        # Estimate from model name
        return self._estimate_bandwidth_gbps()

    def _estimate_bandwidth_gbps(self) -> float:
        """Estimate memory bandwidth from model name."""
        model_lower = self.model.lower()

        bandwidth_estimates = {
            'h100 sxm': 3350,   # HBM3
            'h100 pcie': 2000,  # HBM2e
            'h200': 4800,       # HBM3e
            'a100 sxm': 2039,   # HBM2e
            'a100 pcie': 1935,  # HBM2e
            'a30': 933,         # HBM2
            'v100': 900,        # HBM2
            'rtx 4090': 1008,   # GDDR6X
            'rtx 4080': 717,    # GDDR6X
            'rtx 4070': 504,    # GDDR6X
            'rtx 3090': 936,    # GDDR6X
            'rtx 3080': 760,    # GDDR6X
            'rtx 3070': 448,    # GDDR6
            't4': 320,          # GDDR6
            'orin': 204,        # LPDDR5
        }

        for model_key, bw in bandwidth_estimates.items():
            if model_key in model_lower:
                return float(bw)

        # Conservative default
        return 500.0

    @classmethod
    def _detect_amd(cls) -> Optional['GPUIdentity']:
        """Detect AMD GPU using rocm-smi."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname', '--showmeminfo', 'vram', '--showdriverversion'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Parse rocm-smi output
                # This is simplified - real implementation would parse properly
                return cls(
                    model='AMD GPU',
                    vendor='AMD',
                    pci_id='1002:0000',
                    vbios_version='unknown',
                    memory_mb=0,
                    compute_capability='gfx0',
                    driver_version='unknown',
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    @classmethod
    def _detect_intel(cls) -> Optional['GPUIdentity']:
        """Detect Intel GPU."""
        # Intel GPU detection is more complex, skip for now
        return None


@dataclass
class MemoryIdentity:
    """Auto-detected system memory identity."""
    total_gb: float                 # Total RAM in GB
    type: str                       # "DDR5", "DDR4", "LPDDR5"
    speed_mhz: int                  # Memory speed
    channels: int                   # Number of memory channels

    @classmethod
    def detect(cls) -> 'MemoryIdentity':
        """Auto-detect memory configuration."""
        total_gb = 0.0
        mem_type = "Unknown"
        speed_mhz = 0
        channels = 1

        # Get total memory
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback to /proc/meminfo
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            total_gb = kb / (1024**2)
                            break
            except Exception:
                pass

        # Try dmidecode for memory details (requires root)
        try:
            result = subprocess.run(
                ['sudo', 'dmidecode', '-t', 'memory'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Type:' in line and 'DDR' in line:
                        mem_type = line.split(':')[1].strip()
                    elif 'Speed:' in line and 'MHz' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            speed_mhz = int(match.group(1))
        except Exception:
            pass

        return cls(
            total_gb=round(total_gb, 1),
            type=mem_type,
            speed_mhz=speed_mhz,
            channels=channels,
        )


@dataclass
class HardwareIdentity:
    """
    Complete auto-detected hardware identity.

    The fingerprint is deterministic and stable across:
    - Reboots
    - OS reinstalls
    - Driver updates

    The fingerprint changes when:
    - Different physical hardware
    - BIOS/VBIOS update (silicon-level change)
    - Microcode update
    """
    fingerprint: str                # SHA256 hash of hardware identity
    cpu: CPUIdentity
    gpu: Optional[GPUIdentity]
    memory: MemoryIdentity
    hostname: str                   # For reference only, not in fingerprint

    @classmethod
    def detect(cls) -> 'HardwareIdentity':
        """Auto-detect complete hardware identity."""
        cpu = CPUIdentity.detect()
        gpu = GPUIdentity.detect()
        memory = MemoryIdentity.detect()
        hostname = platform.node()

        # Generate fingerprint from stable hardware characteristics
        fingerprint = cls._generate_fingerprint(cpu, gpu, memory)

        return cls(
            fingerprint=fingerprint,
            cpu=cpu,
            gpu=gpu,
            memory=memory,
            hostname=hostname,
        )

    @staticmethod
    def _generate_fingerprint(
        cpu: CPUIdentity,
        gpu: Optional[GPUIdentity],
        memory: MemoryIdentity
    ) -> str:
        """
        Generate deterministic hardware fingerprint.

        Includes:
        - CPU model, stepping, microcode (silicon identity)
        - GPU PCI ID, VBIOS (silicon identity)
        - Memory size (system configuration)

        Does NOT include:
        - Frequencies (can change with DVFS)
        - Driver versions (software, not hardware)
        - Cache sizes (derived from model)
        """
        components = [
            cpu.model,
            cpu.vendor,
            str(cpu.stepping),
            cpu.microcode,
            str(cpu.cores_physical),
        ]

        if gpu:
            components.extend([
                gpu.pci_id,
                gpu.vbios_version,
                str(gpu.memory_mb),
            ])
        else:
            components.append("no_gpu")

        components.append(str(int(memory.total_gb)))

        fingerprint_str = "|".join(components)
        full_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        return full_hash[:16]  # 16 chars is enough for uniqueness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fingerprint': self.fingerprint,
            'hostname': self.hostname,
            'cpu': asdict(self.cpu),
            'gpu': asdict(self.gpu) if self.gpu else None,
            'memory': asdict(self.memory),
        }


# =============================================================================
# SOFTWARE STACK
# =============================================================================

@dataclass
class SoftwareStack:
    """
    Auto-detected software stack versions.

    Captures everything that could affect benchmark results:
    - OS and kernel
    - Drivers (GPU, etc.)
    - Runtimes (CUDA, ROCm)
    - Frameworks (PyTorch, NumPy)
    - Libraries (cuDNN, MKL, OpenBLAS)
    """
    fingerprint: str                # SHA256 hash of software stack

    # OS
    os_name: str                    # "Linux", "Windows", "Darwin"
    os_release: str                 # "6.8.0-90-generic"
    os_distro: str                  # "Ubuntu 24.04.1 LTS"

    # GPU Driver
    gpu_driver_version: str         # "560.35.03" or "N/A"

    # CUDA ecosystem
    cuda_version: str               # "12.4" or "N/A"
    cuda_runtime_version: str       # "12.4" or "N/A"
    cudnn_version: str              # "8.9.7" or "N/A"
    cublas_version: str             # "12.4.2" or "N/A"

    # ROCm ecosystem
    rocm_version: str               # "6.0" or "N/A"

    # Python ecosystem
    python_version: str             # "3.11.14"
    pytorch_version: str            # "2.4.0" or "N/A"
    numpy_version: str              # "2.0.1"

    # BLAS backends
    mkl_version: str                # "2024.1" or "N/A"
    openblas_version: str           # "0.3.27" or "N/A"
    blas_library: str               # "MKL", "OpenBLAS", "cuBLAS"

    # Compiler
    gcc_version: str                # "13.2.0" or "N/A"
    nvcc_version: str               # "12.4" or "N/A"

    @classmethod
    def detect(cls) -> 'SoftwareStack':
        """Auto-detect complete software stack."""
        # OS info
        os_name = platform.system()
        os_release = platform.release()
        os_distro = cls._get_os_distro()

        # GPU driver
        gpu_driver = cls._get_gpu_driver_version()

        # CUDA ecosystem
        cuda_ver = cls._get_cuda_version()
        cuda_runtime = cls._get_cuda_runtime_version()
        cudnn_ver = cls._get_cudnn_version()
        cublas_ver = cls._get_cublas_version()

        # ROCm
        rocm_ver = cls._get_rocm_version()

        # Python ecosystem
        python_ver = platform.python_version()
        pytorch_ver = cls._get_pytorch_version()
        numpy_ver = cls._get_numpy_version()

        # BLAS
        mkl_ver = cls._get_mkl_version()
        openblas_ver = cls._get_openblas_version()
        blas_lib = cls._detect_blas_library()

        # Compiler
        gcc_ver = cls._get_gcc_version()
        nvcc_ver = cls._get_nvcc_version()

        # Generate fingerprint
        fingerprint = cls._generate_fingerprint(
            os_release, gpu_driver, cuda_ver, pytorch_ver, numpy_ver
        )

        return cls(
            fingerprint=fingerprint,
            os_name=os_name,
            os_release=os_release,
            os_distro=os_distro,
            gpu_driver_version=gpu_driver,
            cuda_version=cuda_ver,
            cuda_runtime_version=cuda_runtime,
            cudnn_version=cudnn_ver,
            cublas_version=cublas_ver,
            rocm_version=rocm_ver,
            python_version=python_ver,
            pytorch_version=pytorch_ver,
            numpy_version=numpy_ver,
            mkl_version=mkl_ver,
            openblas_version=openblas_ver,
            blas_library=blas_lib,
            gcc_version=gcc_ver,
            nvcc_version=nvcc_ver,
        )

    @staticmethod
    def _generate_fingerprint(*components: str) -> str:
        """Generate software stack fingerprint."""
        fingerprint_str = "|".join(components)
        full_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        return full_hash[:16]

    @staticmethod
    def _get_os_distro() -> str:
        """Get OS distribution name."""
        try:
            # Try /etc/os-release (Linux)
            os_release = Path('/etc/os-release')
            if os_release.exists():
                content = os_release.read_text()
                for line in content.split('\n'):
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=')[1].strip('"')
        except Exception:
            pass
        return platform.platform()

    @staticmethod
    def _get_gpu_driver_version() -> str:
        """Get GPU driver version."""
        # Try NVIDIA
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Try AMD
        try:
            result = subprocess.run(
                ['rocm-smi', '--showdriverversion'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Driver' in line:
                        return line.split(':')[-1].strip()
        except Exception:
            pass

        return "N/A"

    @staticmethod
    def _get_cuda_version() -> str:
        """Get CUDA version."""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass

        # Try from PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda or "N/A"
        except Exception:
            pass

        return "N/A"

    @staticmethod
    def _get_cuda_runtime_version() -> str:
        """Get CUDA runtime version."""
        try:
            import torch
            if torch.cuda.is_available():
                return str(torch.cuda.runtime.get_version())
        except Exception:
            pass
        return "N/A"

    @staticmethod
    def _get_cudnn_version() -> str:
        """Get cuDNN version."""
        try:
            import torch
            if torch.cuda.is_available() and torch.backends.cudnn.is_available():
                return str(torch.backends.cudnn.version())
        except Exception:
            pass
        return "N/A"

    @staticmethod
    def _get_cublas_version() -> str:
        """Get cuBLAS version."""
        # cuBLAS version typically matches CUDA version
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda or "N/A"
        except Exception:
            pass
        return "N/A"

    @staticmethod
    def _get_rocm_version() -> str:
        """Get ROCm version."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "N/A"

    @staticmethod
    def _get_pytorch_version() -> str:
        """Get PyTorch version."""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return "N/A"

    @staticmethod
    def _get_numpy_version() -> str:
        """Get NumPy version."""
        try:
            import numpy
            return numpy.__version__
        except ImportError:
            return "N/A"

    @staticmethod
    def _get_mkl_version() -> str:
        """Get MKL version if available."""
        try:
            import numpy
            config = numpy.__config__
            if hasattr(config, 'get_info'):
                mkl_info = config.get_info('blas_mkl_info')
                if mkl_info:
                    return mkl_info.get('define_macros', [('', 'unknown')])[0][1]
        except Exception:
            pass

        # Try numpy.show_config() parsing
        try:
            import numpy
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            numpy.show_config()
            config_str = sys.stdout.getvalue()
            sys.stdout = old_stdout

            if 'mkl' in config_str.lower():
                match = re.search(r'mkl.*?(\d+\.\d+\.\d+)', config_str, re.IGNORECASE)
                if match:
                    return match.group(1)
                return "detected"
        except Exception:
            pass

        return "N/A"

    @staticmethod
    def _get_openblas_version() -> str:
        """Get OpenBLAS version if available."""
        try:
            import numpy
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            numpy.show_config()
            config_str = sys.stdout.getvalue()
            sys.stdout = old_stdout

            if 'openblas' in config_str.lower():
                match = re.search(r'openblas.*?(\d+\.\d+\.\d+)', config_str, re.IGNORECASE)
                if match:
                    return match.group(1)
                return "detected"
        except Exception:
            pass

        return "N/A"

    @staticmethod
    def _detect_blas_library() -> str:
        """Detect which BLAS library is being used."""
        try:
            import numpy
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            numpy.show_config()
            config_str = sys.stdout.getvalue().lower()
            sys.stdout = old_stdout

            if 'mkl' in config_str:
                return "MKL"
            elif 'openblas' in config_str:
                return "OpenBLAS"
            elif 'blis' in config_str:
                return "BLIS"
            elif 'atlas' in config_str:
                return "ATLAS"
        except Exception:
            pass

        return "Unknown"

    @staticmethod
    def _get_gcc_version() -> str:
        """Get GCC version."""
        try:
            result = subprocess.run(
                ['gcc', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return "N/A"

    @staticmethod
    def _get_nvcc_version() -> str:
        """Get NVCC version."""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# =============================================================================
# CALIBRATION CONTEXT
# =============================================================================

@dataclass
class EnvironmentalContext:
    """Runtime environmental conditions."""
    power_mode: str                 # "TDP", "15W", "30W", "MAXN"
    cpu_governor: str               # "performance", "powersave"
    cpu_freq_mhz: int               # Current CPU frequency
    gpu_power_limit_w: int          # Current GPU power limit (0 if no GPU)
    gpu_freq_mhz: int               # Current GPU frequency (0 if no GPU)
    thermal_state: str              # "cool", "warm", "throttled"
    ambient_temp_c: float           # Ambient temperature if available
    system_load_pct: float          # System load percentage

    @classmethod
    def detect(cls) -> 'EnvironmentalContext':
        """Detect current environmental conditions."""
        return cls(
            power_mode=cls._detect_power_mode(),
            cpu_governor=cls._detect_cpu_governor(),
            cpu_freq_mhz=cls._detect_cpu_freq(),
            gpu_power_limit_w=cls._detect_gpu_power_limit(),
            gpu_freq_mhz=cls._detect_gpu_freq(),
            thermal_state=cls._detect_thermal_state(),
            ambient_temp_c=0.0,  # Usually not available
            system_load_pct=cls._detect_system_load(),
        )

    @staticmethod
    def _detect_power_mode() -> str:
        """Detect power mode."""
        # Check for Jetson power modes
        try:
            result = subprocess.run(
                ['nvpmodel', '-q'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Power Mode' in line:
                        return line.split(':')[-1].strip()
        except Exception:
            pass

        # Default to TDP
        return "TDP"

    @staticmethod
    def _detect_cpu_governor() -> str:
        """Detect CPU frequency governor."""
        try:
            gov_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
            if gov_path.exists():
                return gov_path.read_text().strip()
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _detect_cpu_freq() -> int:
        """Detect current CPU frequency."""
        try:
            freq_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
            if freq_path.exists():
                khz = int(freq_path.read_text().strip())
                return khz // 1000  # Convert to MHz
        except Exception:
            pass
        return 0

    @staticmethod
    def _detect_gpu_power_limit() -> int:
        """Detect GPU power limit."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(float(result.stdout.strip()))
        except Exception:
            pass
        return 0

    @staticmethod
    def _detect_gpu_freq() -> int:
        """Detect current GPU frequency."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0

    @staticmethod
    def _detect_thermal_state() -> str:
        """Detect thermal state."""
        try:
            # Check CPU temperature
            for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
                for temp_file in hwmon.glob('temp*_input'):
                    try:
                        temp_mc = int(temp_file.read_text().strip())
                        temp_c = temp_mc / 1000
                        if temp_c > 80:
                            return "throttled"
                        elif temp_c > 60:
                            return "warm"
                        else:
                            return "cool"
                    except Exception:
                        continue
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _detect_system_load() -> float:
        """Detect system load percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            pass

        try:
            with open('/proc/loadavg') as f:
                load_1min = float(f.read().split()[0])
                # Normalize by CPU count
                import os
                cpu_count = os.cpu_count() or 1
                return min(100.0, (load_1min / cpu_count) * 100)
        except Exception:
            pass

        return 0.0


@dataclass
class CalibrationContext:
    """
    Complete context for a calibration run.

    Combines hardware identity, software stack, and environmental conditions.
    This is everything needed to understand and reproduce a calibration result.
    """
    run_id: str                     # UUID for this specific run
    timestamp: datetime             # When this run started
    hardware: HardwareIdentity
    software: SoftwareStack
    environment: EnvironmentalContext

    @classmethod
    def detect(cls) -> 'CalibrationContext':
        """Auto-detect complete calibration context."""
        return cls(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            hardware=HardwareIdentity.detect(),
            software=SoftwareStack.detect(),
            environment=EnvironmentalContext.detect(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp.isoformat(),
            'timestamp_unix': int(self.timestamp.timestamp()),
            'hardware': self.hardware.to_dict(),
            'software': self.software.to_dict(),
            'environment': asdict(self.environment),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "CALIBRATION CONTEXT",
            "=" * 70,
            "",
            f"Run ID:     {self.run_id}",
            f"Timestamp:  {self.timestamp.isoformat()}",
            "",
            "HARDWARE IDENTITY",
            "-" * 40,
            f"Fingerprint: {self.hardware.fingerprint}",
            f"CPU:         {self.hardware.cpu.model}",
            f"  Stepping:  {self.hardware.cpu.stepping}",
            f"  Microcode: {self.hardware.cpu.microcode}",
            f"  Cores:     {self.hardware.cpu.cores_physical}P / {self.hardware.cpu.cores_logical}L",
        ]

        if self.hardware.gpu:
            gpu = self.hardware.gpu
            lines.extend([
                f"GPU:         {gpu.model}",
                f"  PCI ID:    {gpu.pci_id}",
                f"  VBIOS:     {gpu.vbios_version}",
                f"  Memory:    {gpu.memory_mb} MB",
                f"  CUDA Cores: {gpu.cuda_cores}, Tensor Cores: {gpu.tensor_cores}",
                f"  Compute:   {gpu.compute_capability}",
                f"  Clock:     {gpu.clock_mhz} MHz",
            ])
            # Show estimated peaks
            peak_fp32 = gpu.estimate_theoretical_peak_gflops("fp32")
            peak_fp16 = gpu.estimate_theoretical_peak_gflops("fp16")
            bandwidth = gpu.estimate_theoretical_bandwidth_gbps()
            lines.extend([
                f"  Peak FP32: {peak_fp32:.0f} GFLOPS",
                f"  Peak FP16: {peak_fp16:.0f} GFLOPS",
                f"  Bandwidth: {bandwidth:.0f} GB/s",
            ])
        else:
            lines.append("GPU:         None detected")

        lines.extend([
            f"Memory:      {self.hardware.memory.total_gb:.1f} GB {self.hardware.memory.type}",
            "",
            "SOFTWARE STACK",
            "-" * 40,
            f"Fingerprint: {self.software.fingerprint}",
            f"OS:          {self.software.os_distro}",
            f"Kernel:      {self.software.os_release}",
            f"GPU Driver:  {self.software.gpu_driver_version}",
            f"CUDA:        {self.software.cuda_version}",
            f"PyTorch:     {self.software.pytorch_version}",
            f"NumPy:       {self.software.numpy_version}",
            f"BLAS:        {self.software.blas_library}",
            "",
            "ENVIRONMENT",
            "-" * 40,
            f"Power Mode:  {self.environment.power_mode}",
            f"Governor:    {self.environment.cpu_governor}",
            f"CPU Freq:    {self.environment.cpu_freq_mhz} MHz",
            f"Thermal:     {self.environment.thermal_state}",
            f"Load:        {self.environment.system_load_pct:.1f}%",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_all() -> CalibrationContext:
    """
    Auto-detect complete calibration context.

    This is the main entry point for the auto-detection system.
    """
    return CalibrationContext.detect()


def print_context():
    """Print detected calibration context to stdout."""
    context = detect_all()
    print(context.summary())


if __name__ == "__main__":
    # Test auto-detection
    print_context()
