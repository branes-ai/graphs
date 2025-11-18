"""
Hardware Detection Module

Cross-platform hardware detection for CPU, GPU, and platform identification.
Supports Linux, Windows (x86_64), and macOS.

Uses cross-platform libraries where possible:
- psutil: CPU cores/threads/frequency
- py-cpuinfo: CPU model, vendor, ISA extensions
- Platform-specific code only for specialized features (P-core/E-core detection)
"""

import re
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

from .schema import HardwareSpec


@dataclass
class MatchResult:
    """Result of matching detected hardware to database spec"""
    detected_string: str
    matched_spec: 'HardwareSpec'
    confidence: float


@dataclass
class DetectedCPU:
    """CPU detection result"""
    model_name: str
    vendor: str
    architecture: str
    cores: int
    threads: int
    base_frequency_ghz: Optional[float] = None
    boost_frequency_ghz: Optional[float] = None
    e_cores: Optional[int] = None  # Efficiency cores (for hybrid CPUs)
    isa_extensions: List[str] = field(default_factory=list)

    # Cache information (on-chip memory hierarchy)
    # Simple fields for backward compatibility
    l1_dcache_kb: Optional[int] = None
    l1_icache_kb: Optional[int] = None
    l2_cache_kb: Optional[int] = None
    l3_cache_kb: Optional[int] = None
    l1_cache_line_size_bytes: Optional[int] = None
    l2_cache_line_size_bytes: Optional[int] = None
    l3_cache_line_size_bytes: Optional[int] = None
    l1_dcache_associativity: Optional[int] = None
    l1_icache_associativity: Optional[int] = None
    l2_cache_associativity: Optional[int] = None
    l3_cache_associativity: Optional[int] = None

    # Structured cache_levels
    cache_levels: Optional[List[Dict]] = None


@dataclass
class DetectedGPU:
    """GPU detection result"""
    model_name: str
    vendor: str
    memory_gb: Optional[int] = None
    cuda_capability: Optional[str] = None
    driver_version: Optional[str] = None


@dataclass
class DetectedMemoryChannel:
    """Detected memory channel/DIMM information"""
    name: str
    type: str  # "ddr4", "ddr5", "lpddr5x", etc.
    size_gb: float
    speed_mts: int  # MT/s (e.g., 5600)
    rank_count: Optional[int] = None
    ecc_enabled: Optional[bool] = None
    physical_position: Optional[int] = None
    locator: Optional[str] = None  # e.g., "Channel 0", "DIMM_A1"


@dataclass
class DetectedMemory:
    """Memory detection result"""
    total_gb: float
    channels: List[DetectedMemoryChannel] = field(default_factory=list)
    numa_nodes: Optional[int] = None


class HardwareDetector:
    """
    Cross-platform hardware detector.

    Detects CPU, GPU, and platform information on Linux, Windows, and macOS.
    """

    def __init__(self):
        self.os_type = self._detect_os()
        self.platform_arch = self._detect_platform()

    def _detect_os(self) -> str:
        """Detect operating system (linux, windows, macos)"""
        sys_name = platform.system().lower()

        if sys_name == 'linux':
            return 'linux'
        elif sys_name == 'windows':
            return 'windows'
        elif sys_name == 'darwin':
            return 'macos'
        else:
            return 'unknown'

    def _detect_platform(self) -> str:
        """Detect platform architecture (x86_64, aarch64, arm64)"""
        machine = platform.machine().lower()

        # Normalize architecture names
        if machine in ['x86_64', 'amd64', 'x64']:
            return 'x86_64'
        elif machine in ['aarch64', 'arm64']:
            # Prefer aarch64 for Linux, arm64 for macOS
            if self.os_type == 'macos':
                return 'arm64'
            else:
                return 'aarch64'
        else:
            return machine

    # ========================================================================
    # CPU Detection
    # ========================================================================

    def detect_cpu(self) -> Optional[DetectedCPU]:
        """
        Detect CPU information (cross-platform).

        Uses psutil + py-cpuinfo for cross-platform detection,
        with platform-specific code for specialized features.

        Returns:
            DetectedCPU instance or None if detection fails
        """
        # Try cross-platform libraries first
        if PSUTIL_AVAILABLE and CPUINFO_AVAILABLE:
            try:
                return self._detect_cpu_crossplatform()
            except Exception as e:
                print(f"Warning: Cross-platform CPU detection failed: {e}")
                # Fall through to platform-specific detection

        # Fall back to platform-specific detection
        if self.os_type == 'linux':
            return self._detect_cpu_linux()
        elif self.os_type == 'windows':
            return self._detect_cpu_windows()
        elif self.os_type == 'macos':
            return self._detect_cpu_macos()
        else:
            return None

    def _detect_cpu_crossplatform(self) -> Optional[DetectedCPU]:
        """
        Detect CPU using cross-platform libraries (psutil + py-cpuinfo).

        Works on Linux, Windows, and macOS.
        """
        # Get CPU info from py-cpuinfo
        cpu_info = cpuinfo.get_cpu_info()

        # Extract basic info
        model_name = cpu_info.get('brand_raw', 'Unknown CPU')
        vendor_id = cpu_info.get('vendor_id_raw', 'Unknown')

        # Map vendor_id to vendor name
        vendor_map = {
            'GenuineIntel': 'Intel',
            'AuthenticAMD': 'AMD',
            'ARM': 'ARM',
            'APM': 'Ampere Computing',
        }
        vendor = vendor_map.get(vendor_id, vendor_id)

        # Extract architecture
        architecture = self._extract_architecture(model_name, vendor)

        # Get cores and threads from psutil (cross-platform)
        cores = psutil.cpu_count(logical=False) or 1
        threads = psutil.cpu_count(logical=True) or cores

        # Get frequency from psutil
        freq = psutil.cpu_freq()
        base_frequency_ghz = None
        boost_frequency_ghz = None
        if freq:
            # Use current frequency as base
            if freq.current:
                base_frequency_ghz = freq.current / 1000.0
            # Use max frequency as boost
            if freq.max:
                boost_frequency_ghz = freq.max / 1000.0

        # On Linux, try to get more accurate frequencies from lscpu
        if self.os_type == 'linux':
            try:
                result = subprocess.run(
                    ['lscpu'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'CPU max MHz:' in line:
                            max_mhz = float(line.split(':')[1].strip())
                            boost_frequency_ghz = max_mhz / 1000.0
                        elif 'CPU min MHz:' in line and base_frequency_ghz is None:
                            min_mhz = float(line.split(':')[1].strip())
                            # Don't use min as base; keep current from psutil
            except Exception:
                pass  # Fallback to psutil values

        # Extract ISA extensions from flags
        flags = cpu_info.get('flags', [])
        isa_extensions = self._extract_isa_extensions(flags, vendor)

        # Detect E-cores for hybrid CPUs (platform-specific)
        e_cores = self._detect_e_cores(model_name, vendor, cores, threads)

        # Extract cache information from cpuinfo
        cache_info = self._extract_cache_info(cpu_info)

        return DetectedCPU(
            model_name=model_name,
            vendor=vendor,
            architecture=architecture,
            cores=cores,
            threads=threads,
            base_frequency_ghz=base_frequency_ghz,
            boost_frequency_ghz=boost_frequency_ghz,
            e_cores=e_cores,
            isa_extensions=isa_extensions,
            **cache_info
        )

    def _detect_e_cores(
        self,
        model_name: str,
        vendor: str,
        cores: int,
        threads: int
    ) -> Optional[int]:
        """
        Detect number of E-cores for hybrid CPUs.

        Intel 12th gen+ (Alder Lake, Raptor Lake) have P-cores + E-cores.
        E-cores don't have hyperthreading, P-cores do.

        Args:
            model_name: CPU model name
            vendor: CPU vendor
            cores: Total physical cores
            threads: Total hardware threads

        Returns:
            Number of E-cores, or None if not a hybrid CPU
        """
        # Only Intel 12th gen+ have hybrid architecture
        if vendor != 'Intel':
            return None

        model_lower = model_name.lower()

        # Check if this is a hybrid CPU
        is_alder_lake = '12th gen' in model_lower or 'i9-12' in model_lower or 'i7-12' in model_lower or 'i5-12' in model_lower
        is_raptor_lake = '13th gen' in model_lower or 'i9-13' in model_lower or 'i7-13' in model_lower or 'i5-13' in model_lower
        is_raptor_lake_refresh = '14th gen' in model_lower or 'i9-14' in model_lower or 'i7-14' in model_lower or 'i5-14' in model_lower

        if not (is_alder_lake or is_raptor_lake or is_raptor_lake_refresh):
            return None

        # For hybrid CPUs: P-cores have HT (2 threads), E-cores don't (1 thread)
        # threads = P_cores * 2 + E_cores * 1
        # cores = P_cores + E_cores
        #
        # Solving: P_cores = threads - cores
        #          E_cores = cores - P_cores = 2*cores - threads

        p_cores = threads - cores
        e_cores = 2 * cores - threads

        # Sanity check
        if e_cores < 0 or p_cores < 0:
            return None

        # Platform-specific refinement
        if self.os_type == 'linux':
            # Try to get more accurate count from lscpu
            try:
                e_cores_lscpu = self._detect_e_cores_linux()
                if e_cores_lscpu is not None:
                    return e_cores_lscpu
            except:
                pass

        return e_cores if e_cores > 0 else None

    def _detect_e_cores_linux(self) -> Optional[int]:
        """
        Detect E-cores on Linux using lscpu topology.

        Returns:
            Number of E-cores, or None if detection fails
        """
        try:
            # lscpu -e shows per-core topology including max MHz
            # E-cores typically have lower max frequency than P-cores
            output = subprocess.check_output(['lscpu', '-e'], text=True)

            # Parse output to find cores with different max frequencies
            # This is heuristic but works for Alder Lake/Raptor Lake
            core_freqs = {}
            for line in output.split('\n')[1:]:  # Skip header
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    core_id = parts[1]
                    max_mhz = parts[4]
                    if max_mhz != '-':
                        core_freqs[core_id] = float(max_mhz)

            if not core_freqs:
                return None

            # Find frequency threshold (E-cores have significantly lower max freq)
            freqs = sorted(set(core_freqs.values()))
            if len(freqs) < 2:
                return None  # Not a hybrid CPU

            # Count cores with lower frequency (E-cores)
            threshold = (freqs[0] + freqs[-1]) / 2
            e_core_count = sum(1 for freq in core_freqs.values() if freq < threshold)

            return e_core_count if e_core_count > 0 else None

        except Exception:
            return None

    def _detect_cpu_linux(self) -> Optional[DetectedCPU]:
        """Detect CPU on Linux via /proc/cpuinfo"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()

            # Extract model name
            model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
            model_name = model_match.group(1).strip() if model_match else 'Unknown CPU'

            # Extract vendor
            vendor_match = re.search(r'vendor_id\s*:\s*(.+)', cpuinfo)
            vendor_id = vendor_match.group(1).strip() if vendor_match else 'Unknown'

            # Map vendor_id to vendor name
            vendor_map = {
                'GenuineIntel': 'Intel',
                'AuthenticAMD': 'AMD',
                'ARM': 'ARM',
                'APM': 'Ampere Computing',
            }
            vendor = vendor_map.get(vendor_id, vendor_id)

            # Extract architecture from model name
            architecture = self._extract_architecture(model_name, vendor)

            # Count physical cores and hardware threads
            # Use lscpu for accurate hybrid CPU detection (P-cores + E-cores)
            try:
                lscpu_output = subprocess.check_output(['lscpu'], text=True)

                # Parse lscpu output
                cores_per_socket = None
                sockets = 1
                threads_per_core = None
                total_cpus = None

                for line in lscpu_output.split('\n'):
                    if line.startswith('Core(s) per socket:'):
                        cores_per_socket = int(line.split(':')[1].strip())
                    elif line.startswith('Socket(s):'):
                        sockets = int(line.split(':')[1].strip())
                    elif line.startswith('Thread(s) per core:'):
                        threads_per_core = int(line.split(':')[1].strip())
                    elif line.startswith('CPU(s):'):
                        # First CPU(s) line is total logical CPUs
                        if total_cpus is None:
                            total_cpus = int(line.split(':')[1].strip())

                if cores_per_socket and sockets:
                    cores = cores_per_socket * sockets
                else:
                    # Fallback: count unique (physical_id, core_id) pairs
                    cores = self._count_physical_cores_from_cpuinfo(cpuinfo)

                # Get actual hardware thread count (handles hybrid CPUs correctly)
                if total_cpus:
                    threads = total_cpus
                else:
                    # Fallback to counting processor entries
                    threads = len(re.findall(r'^processor\s*:', cpuinfo, re.MULTILINE))

            except (subprocess.CalledProcessError, FileNotFoundError):
                # lscpu not available, fall back to cpuinfo parsing
                cores = self._count_physical_cores_from_cpuinfo(cpuinfo)
                threads = len(re.findall(r'^processor\s*:', cpuinfo, re.MULTILINE))

            # Extract frequency (in MHz, convert to GHz)
            freq_match = re.search(r'cpu MHz\s*:\s*(\d+\.?\d*)', cpuinfo)
            base_frequency_ghz = None
            if freq_match:
                base_frequency_ghz = float(freq_match.group(1)) / 1000.0

            # Extract flags (ISA extensions)
            flags_match = re.search(r'flags\s*:\s*(.+)', cpuinfo)
            isa_extensions = []
            if flags_match:
                flags = flags_match.group(1).split()
                isa_extensions = self._extract_isa_extensions(flags, vendor)

            return DetectedCPU(
                model_name=model_name,
                vendor=vendor,
                architecture=architecture,
                cores=cores,
                threads=threads,
                base_frequency_ghz=base_frequency_ghz,
                isa_extensions=isa_extensions
            )

        except Exception as e:
            print(f"Warning: CPU detection failed on Linux: {e}")
            return None

    def _detect_cpu_windows(self) -> Optional[DetectedCPU]:
        """Detect CPU on Windows via wmic and platform module"""
        try:
            # Use wmic to get CPU information
            cmd = 'wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed /format:list'
            output = subprocess.check_output(cmd, shell=True, text=True)

            # Parse wmic output
            model_name = 'Unknown CPU'
            cores = 0
            threads = 0
            base_frequency_ghz = None

            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('Name='):
                    model_name = line.split('=', 1)[1].strip()
                elif line.startswith('NumberOfCores='):
                    cores = int(line.split('=')[1].strip())
                elif line.startswith('NumberOfLogicalProcessors='):
                    threads = int(line.split('=')[1].strip())
                elif line.startswith('MaxClockSpeed='):
                    # MaxClockSpeed is in MHz
                    freq_mhz = int(line.split('=')[1].strip())
                    base_frequency_ghz = freq_mhz / 1000.0

            # Detect vendor from model name
            vendor = 'Unknown'
            if 'Intel' in model_name:
                vendor = 'Intel'
            elif 'AMD' in model_name:
                vendor = 'AMD'
            elif 'Ampere' in model_name or 'Altra' in model_name:
                vendor = 'Ampere Computing'

            # Extract architecture
            architecture = self._extract_architecture(model_name, vendor)

            # Detect ISA extensions (Windows-specific approach)
            isa_extensions = self._detect_isa_extensions_windows(vendor)

            return DetectedCPU(
                model_name=model_name,
                vendor=vendor,
                architecture=architecture,
                cores=cores,
                threads=threads,
                base_frequency_ghz=base_frequency_ghz,
                isa_extensions=isa_extensions
            )

        except Exception as e:
            print(f"Warning: CPU detection failed on Windows: {e}")
            return None

    def _detect_cpu_macos(self) -> Optional[DetectedCPU]:
        """Detect CPU on macOS via sysctl"""
        try:
            # Get CPU brand string
            model_name = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                text=True
            ).strip()

            # Get core counts
            cores = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.physicalcpu'],
                text=True
            ).strip())

            threads = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.logicalcpu'],
                text=True
            ).strip())

            # Get frequency (in Hz, convert to GHz)
            try:
                freq_hz = int(subprocess.check_output(
                    ['sysctl', '-n', 'hw.cpufrequency'],
                    text=True
                ).strip())
                base_frequency_ghz = freq_hz / 1e9
            except:
                base_frequency_ghz = None

            # Detect vendor
            vendor = 'Unknown'
            if 'Intel' in model_name:
                vendor = 'Intel'
            elif 'AMD' in model_name:
                vendor = 'AMD'
            elif 'Apple' in model_name or 'M1' in model_name or 'M2' in model_name or 'M3' in model_name:
                vendor = 'Apple'

            # Extract architecture
            architecture = self._extract_architecture(model_name, vendor)

            # Get ISA extensions (macOS-specific)
            isa_extensions = []
            try:
                features = subprocess.check_output(
                    ['sysctl', '-n', 'machdep.cpu.features'],
                    text=True
                ).strip().split()
                isa_extensions = self._extract_isa_extensions(features, vendor)
            except:
                pass

            return DetectedCPU(
                model_name=model_name,
                vendor=vendor,
                architecture=architecture,
                cores=cores,
                threads=threads,
                base_frequency_ghz=base_frequency_ghz,
                isa_extensions=isa_extensions
            )

        except Exception as e:
            print(f"Warning: CPU detection failed on macOS: {e}")
            return None

    def _count_physical_cores_from_cpuinfo(self, cpuinfo: str) -> int:
        """
        Count physical cores from /proc/cpuinfo.

        For hybrid CPUs (P-cores + E-cores), this correctly counts total physical cores.
        Counts unique (physical_id, core_id) pairs.
        """
        cores_seen = set()

        # Parse cpuinfo line by line, tracking processor blocks
        physical_id = 0
        core_id = None

        for line in cpuinfo.split('\n'):
            line = line.strip()

            if line.startswith('physical id'):
                physical_id = int(line.split(':')[1].strip())
            elif line.startswith('core id'):
                core_id = int(line.split(':')[1].strip())
                # Record this (physical_id, core_id) pair
                cores_seen.add((physical_id, core_id))

        # If no core id information, fall back to counting unique processor IDs
        if not cores_seen:
            processors = len(re.findall(r'^processor\s*:', cpuinfo, re.MULTILINE))
            return processors

        return len(cores_seen)

    def _extract_cache_info(self, cpu_info: Dict) -> Dict[str, Optional[int]]:
        """
        Extract cache information from cpuinfo dictionary.

        Args:
            cpu_info: Dictionary from cpuinfo.get_cpu_info()

        Returns:
            Dictionary with cache fields in KB and bytes
        """
        cache_info = {}

        # Extract cache sizes (convert from bytes to KB)
        if 'l1_data_cache_size' in cpu_info:
            cache_info['l1_dcache_kb'] = cpu_info['l1_data_cache_size'] // 1024
        if 'l1_instruction_cache_size' in cpu_info:
            cache_info['l1_icache_kb'] = cpu_info['l1_instruction_cache_size'] // 1024
        if 'l2_cache_size' in cpu_info:
            cache_info['l2_cache_kb'] = cpu_info['l2_cache_size'] // 1024
        if 'l3_cache_size' in cpu_info:
            cache_info['l3_cache_kb'] = cpu_info['l3_cache_size'] // 1024

        # Extract cache line sizes (already in bytes from cpuinfo)
        # Note: cpuinfo typically only provides L2 cache line size
        if 'l2_cache_line_size' in cpu_info:
            cache_info['l2_cache_line_size_bytes'] = cpu_info['l2_cache_line_size']
            # Assume L1 and L3 have same cache line size (typical for most CPUs)
            cache_info['l1_cache_line_size_bytes'] = cpu_info['l2_cache_line_size']
            cache_info['l3_cache_line_size_bytes'] = cpu_info['l2_cache_line_size']

        # Extract cache associativity
        if 'l2_cache_associativity' in cpu_info:
            cache_info['l2_cache_associativity'] = cpu_info['l2_cache_associativity']

        # Try to extract L1 and L3 associativity if available
        if 'l1_cache_associativity' in cpu_info:
            # Apply to both dcache and icache (most CPUs have same associativity)
            cache_info['l1_dcache_associativity'] = cpu_info['l1_cache_associativity']
            cache_info['l1_icache_associativity'] = cpu_info['l1_cache_associativity']
        if 'l3_cache_associativity' in cpu_info:
            cache_info['l3_cache_associativity'] = cpu_info['l3_cache_associativity']

        # Build cache_levels structure
        cache_levels = []

        # L1 dcache - per_core
        if cache_info.get('l1_dcache_kb'):
            cache_levels.append({
                "name": "L1 dcache",
                "level": 1,
                "cache_type": "data",
                "scope": "per_core",
                "size_per_unit_kb": cache_info['l1_dcache_kb'],
                "associativity": cache_info.get('l1_dcache_associativity'),
                "line_size_bytes": cache_info.get('l1_cache_line_size_bytes')
            })

        # L1 icache - per_core
        if cache_info.get('l1_icache_kb'):
            cache_levels.append({
                "name": "L1 icache",
                "level": 1,
                "cache_type": "instruction",
                "scope": "per_core",
                "size_per_unit_kb": cache_info['l1_icache_kb'],
                "associativity": cache_info.get('l1_icache_associativity'),
                "line_size_bytes": cache_info.get('l1_cache_line_size_bytes')
            })

        # L2 - per_core (modern CPUs)
        if cache_info.get('l2_cache_kb'):
            cache_levels.append({
                "name": "L2",
                "level": 2,
                "cache_type": "unified",
                "scope": "per_core",
                "size_per_unit_kb": cache_info['l2_cache_kb'],
                "associativity": cache_info.get('l2_cache_associativity'),
                "line_size_bytes": cache_info.get('l2_cache_line_size_bytes')
            })

        # L3 - shared (LLC)
        if cache_info.get('l3_cache_kb'):
            cache_levels.append({
                "name": "L3",
                "level": 3,
                "cache_type": "unified",
                "scope": "shared",
                "total_size_kb": cache_info['l3_cache_kb'],
                "associativity": cache_info.get('l3_cache_associativity'),
                "line_size_bytes": cache_info.get('l3_cache_line_size_bytes')
            })

        if cache_levels:
            cache_info['cache_levels'] = cache_levels

        return cache_info

    def _extract_architecture(self, model_name: str, vendor: str) -> str:
        """Extract CPU architecture from model name"""
        model_lower = model_name.lower()

        # Intel architectures
        if vendor == 'Intel':
            if '13th gen' in model_lower or 'i9-13' in model_lower or 'i7-13' in model_lower or 'i5-13' in model_lower:
                return 'Raptor Lake'
            elif '12th gen' in model_lower or 'i9-12' in model_lower or 'i7-12' in model_lower or 'i5-12' in model_lower:
                return 'Alder Lake'
            elif '11th gen' in model_lower or 'i9-11' in model_lower or 'i7-11' in model_lower:
                return 'Rocket Lake'
            elif '10th gen' in model_lower or 'i9-10' in model_lower or 'i7-10' in model_lower:
                return 'Comet Lake'
            elif 'xeon' in model_lower and 'scalable' in model_lower:
                if '4th' in model_lower:
                    return 'Sapphire Rapids'
                elif '3rd' in model_lower:
                    return 'Ice Lake'
                elif '2nd' in model_lower:
                    return 'Cascade Lake'

        # AMD architectures
        elif vendor == 'AMD':
            if 'ryzen 9 7' in model_lower or 'ryzen 7 7' in model_lower or 'ryzen 5 7' in model_lower:
                return 'Zen 4'
            elif 'ryzen 9 5' in model_lower or 'ryzen 7 5' in model_lower or 'ryzen 5 5' in model_lower:
                return 'Zen 3'
            elif 'ryzen 9 3' in model_lower or 'ryzen 7 3' in model_lower:
                return 'Zen 2'
            elif 'epyc' in model_lower:
                if '9' in model_lower[:20]:  # 9xx4 series
                    return 'Zen 4'
                elif '7' in model_lower[:20]:  # 7xx3 series
                    return 'Zen 3'

        # Ampere
        elif vendor == 'Ampere Computing':
            if 'altra' in model_lower:
                return 'Neoverse N1'

        # Apple Silicon
        elif vendor == 'Apple':
            if 'm3' in model_lower:
                return 'Apple M3'
            elif 'm2' in model_lower:
                return 'Apple M2'
            elif 'm1' in model_lower:
                return 'Apple M1'

        return 'Unknown'

    def _extract_isa_extensions(self, flags: List[str], vendor: str) -> List[str]:
        """Extract relevant ISA extensions from CPU flags"""
        extensions = []
        flags_lower = [f.lower() for f in flags]

        # x86 extensions
        if vendor in ['Intel', 'AMD']:
            if 'avx512f' in flags_lower or 'avx512' in flags_lower:
                extensions.append('AVX512')
            if 'avx2' in flags_lower:
                extensions.append('AVX2')
            if 'avx' in flags_lower and 'AVX2' not in extensions:
                extensions.append('AVX')
            if 'fma' in flags_lower or 'fma3' in flags_lower:
                extensions.append('FMA3')
            if 'sse4_2' in flags_lower or 'sse4.2' in flags_lower:
                extensions.append('SSE4.2')
            if 'vnni' in flags_lower or 'avx_vnni' in flags_lower:
                extensions.append('VNNI')
            if 'amx' in flags_lower or 'amx_tile' in flags_lower:
                extensions.append('AMX')

        # ARM extensions
        elif vendor in ['ARM', 'Apple', 'Ampere Computing']:
            if 'neon' in flags_lower:
                extensions.append('NEON')
            if 'sve' in flags_lower:
                extensions.append('SVE')
            if 'sve2' in flags_lower:
                extensions.append('SVE2')
            if 'bf16' in flags_lower:
                extensions.append('BF16')
            if 'i8mm' in flags_lower:
                extensions.append('I8MM')

        return extensions

    def _detect_isa_extensions_windows(self, vendor: str) -> List[str]:
        """
        Detect ISA extensions on Windows.

        This is more limited than Linux since we can't easily read CPU flags.
        Use cpuinfo library if available, otherwise make educated guesses.
        """
        extensions = []

        try:
            # Try using cpuinfo library if available
            import cpuinfo
            info = cpuinfo.get_cpu_info()

            flags = info.get('flags', [])
            return self._extract_isa_extensions(flags, vendor)

        except ImportError:
            # cpuinfo not available, make conservative assumptions based on vendor
            # Modern Intel/AMD CPUs on x86_64 will have at least these
            if vendor in ['Intel', 'AMD']:
                extensions = ['AVX2', 'FMA3', 'SSE4.2']

            return extensions

    # ========================================================================
    # GPU Detection
    # ========================================================================

    def detect_gpu(self) -> List[DetectedGPU]:
        """
        Detect GPU information (cross-platform).

        Supports NVIDIA GPUs via nvidia-smi and PyTorch CUDA.

        Returns:
            List of DetectedGPU instances (empty if no GPUs found)
        """
        gpus = []

        # Try nvidia-smi first (works on Linux and Windows)
        nvidia_gpus = self._detect_nvidia_smi()
        gpus.extend(nvidia_gpus)

        # Try PyTorch CUDA as fallback/supplement
        if not gpus:
            torch_gpus = self._detect_pytorch_cuda()
            gpus.extend(torch_gpus)

        return gpus

    def _detect_nvidia_smi(self) -> List[DetectedGPU]:
        """Detect NVIDIA GPUs via nvidia-smi (Linux and Windows)"""
        gpus = []

        try:
            # nvidia-smi command works on both Linux and Windows
            # On Windows, nvidia-smi.exe is in C:\Windows\System32 or CUDA bin
            cmd = [
                'nvidia-smi',
                '--query-gpu=name,memory.total,compute_cap,driver_version',
                '--format=csv,noheader,nounits'
            ]

            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)

            for line in output.strip().split('\n'):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    model_name = parts[0]
                    memory_mb = int(float(parts[1]))
                    memory_gb = memory_mb // 1024
                    cuda_capability = parts[2]
                    driver_version = parts[3]

                    gpus.append(DetectedGPU(
                        model_name=model_name,
                        vendor='NVIDIA',
                        memory_gb=memory_gb,
                        cuda_capability=cuda_capability,
                        driver_version=driver_version
                    ))

        except (subprocess.CalledProcessError, FileNotFoundError):
            # nvidia-smi not found or failed
            pass

        return gpus

    def _detect_pytorch_cuda(self) -> List[DetectedGPU]:
        """Detect NVIDIA GPUs via PyTorch CUDA (all platforms)"""
        gpus = []

        try:
            import torch

            if not torch.cuda.is_available():
                return gpus

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)

                model_name = props.name
                memory_gb = props.total_memory // (1024 ** 3)
                cuda_capability = f"{props.major}.{props.minor}"

                gpus.append(DetectedGPU(
                    model_name=model_name,
                    vendor='NVIDIA',
                    memory_gb=memory_gb,
                    cuda_capability=cuda_capability,
                    driver_version=None  # Not available via PyTorch
                ))

        except ImportError:
            # PyTorch not installed
            pass

        return gpus

    # ========================================================================
    # Database Matching
    # ========================================================================

    def match_cpu_to_database(self, cpu: DetectedCPU, db) -> List[MatchResult]:
        """
        Match detected CPU against hardware database.

        Args:
            cpu: DetectedCPU instance
            db: HardwareDatabase instance

        Returns:
            List of MatchResult sorted by confidence
        """
        results = []

        # Search by vendor and device_type=cpu
        candidates = db.search(vendor=cpu.vendor, device_type='cpu')

        for spec in candidates:
            confidence = self._calculate_match_confidence(
                cpu.model_name,
                spec.detection_patterns
            )

            if confidence > 0.0:
                results.append(MatchResult(
                    detected_string=cpu.model_name,
                    matched_spec=spec,
                    confidence=confidence
                ))

        # Sort by confidence (highest first)
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def match_gpu_to_database(self, gpu: DetectedGPU, db) -> List[MatchResult]:
        """
        Match detected GPU against hardware database.

        Args:
            gpu: DetectedGPU instance
            db: HardwareDatabase instance

        Returns:
            List of MatchResult sorted by confidence
        """
        results = []

        # Search by vendor and device_type=gpu
        candidates = db.search(vendor=gpu.vendor, device_type='gpu')

        for spec in candidates:
            confidence = self._calculate_match_confidence(
                gpu.model_name,
                spec.detection_patterns
            )

            if confidence > 0.0:
                results.append(MatchResult(
                    detected_string=gpu.model_name,
                    matched_spec=spec,
                    confidence=confidence
                ))

        # Sort by confidence (highest first)
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def _calculate_match_confidence(
        self,
        detected_string: str,
        patterns: List[str]
    ) -> float:
        """
        Calculate confidence score for pattern matching.

        Args:
            detected_string: Detected hardware string
            patterns: List of regex patterns from database

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not patterns:
            return 0.0

        detected_lower = detected_string.lower()

        for pattern in patterns:
            try:
                # Try regex match
                if re.search(pattern, detected_string, re.IGNORECASE):
                    return 1.0  # Exact pattern match
            except re.error:
                # Not a regex, try substring match
                if pattern.lower() in detected_lower:
                    return 0.9  # Substring match

        # No match
        return 0.0

    # ========================================================================
    # Memory Detection
    # ========================================================================

    def detect_memory(self) -> Optional[DetectedMemory]:
        """
        Detect system memory configuration.

        On Linux, uses dmidecode to get detailed DIMM information.
        On other platforms, falls back to psutil for total memory only.

        Returns:
            DetectedMemory instance or None if detection fails
        """
        if self.os_type == 'linux':
            return self._detect_memory_linux()
        else:
            return self._detect_memory_fallback()

    def _detect_memory_linux(self) -> Optional[DetectedMemory]:
        """Detect memory using dmidecode on Linux (requires sudo)"""
        try:
            # Run dmidecode to get memory info
            result = subprocess.run(
                ['sudo', 'dmidecode', '-t', 'memory'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                # Fall back to psutil if dmidecode fails
                return self._detect_memory_fallback()

            output = result.stdout

            # Parse dmidecode output
            channels = []
            current_dimm = {}
            channel_idx = 0

            for line in output.split('\n'):
                line = line.strip()

                # Size field
                if line.startswith('Size:'):
                    size_str = line.split(':', 1)[1].strip()
                    if 'MB' in size_str:
                        size_mb = int(size_str.split()[0])
                        current_dimm['size_gb'] = size_mb / 1024.0
                    elif 'GB' in size_str:
                        size_gb = int(size_str.split()[0])
                        current_dimm['size_gb'] = size_gb
                    elif 'No Module Installed' not in size_str:
                        # Skip empty slots
                        continue

                # Type field
                elif line.startswith('Type:') and 'size_gb' in current_dimm:
                    mem_type = line.split(':', 1)[1].strip().lower()
                    if mem_type != 'unknown':
                        current_dimm['type'] = mem_type

                # Speed field (in MT/s)
                elif line.startswith('Configured Memory Speed:') and 'size_gb' in current_dimm:
                    speed_str = line.split(':', 1)[1].strip()
                    if 'MT/s' in speed_str:
                        speed = int(speed_str.split()[0])
                        current_dimm['speed_mts'] = speed

                # Locator field
                elif line.startswith('Locator:') and 'size_gb' in current_dimm:
                    locator = line.split(':', 1)[1].strip()
                    current_dimm['locator'] = locator

                # Rank field
                elif line.startswith('Rank:') and 'size_gb' in current_dimm:
                    rank_str = line.split(':', 1)[1].strip()
                    if rank_str.isdigit():
                        current_dimm['rank_count'] = int(rank_str)

                # ECC detection
                elif line.startswith('Error Correction Type:'):
                    ecc_type = line.split(':', 1)[1].strip()
                    current_dimm['ecc_enabled'] = (ecc_type != 'None')

                # End of a memory device entry - save it
                elif line == '' and current_dimm and 'size_gb' in current_dimm and 'type' in current_dimm:
                    # Create a DetectedMemoryChannel
                    name = current_dimm.get('locator', f'Channel {channel_idx}')
                    channel = DetectedMemoryChannel(
                        name=name,
                        type=current_dimm['type'],
                        size_gb=current_dimm['size_gb'],
                        speed_mts=current_dimm.get('speed_mts', 0),
                        rank_count=current_dimm.get('rank_count'),
                        ecc_enabled=current_dimm.get('ecc_enabled'),
                        physical_position=channel_idx,
                        locator=current_dimm.get('locator')
                    )
                    channels.append(channel)
                    channel_idx += 1
                    current_dimm = {}

            # Calculate total memory
            total_gb = sum(ch.size_gb for ch in channels)

            if total_gb > 0 and channels:
                return DetectedMemory(
                    total_gb=total_gb,
                    channels=channels
                )
            else:
                # Fall back if dmidecode parsing failed
                return self._detect_memory_fallback()

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # dmidecode not available or failed, fall back
            return self._detect_memory_fallback()

    def _detect_memory_fallback(self) -> Optional[DetectedMemory]:
        """Fallback memory detection using psutil (total memory only)"""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            mem_info = psutil.virtual_memory()
            total_gb = mem_info.total / (1024 ** 3)

            return DetectedMemory(
                total_gb=total_gb,
                channels=[]  # No detailed channel info available
            )
        except Exception:
            return None

    # ========================================================================
    # Full Auto-Detection
    # ========================================================================

    def auto_detect(self, db) -> Dict[str, Any]:
        """
        Auto-detect all hardware and match against database.

        Args:
            db: HardwareDatabase instance

        Returns:
            Dictionary with detection results:
            {
                'os': str,
                'platform': str,
                'cpu': DetectedCPU or None,
                'cpu_matches': List[HardwareDetectionResult],
                'gpus': List[DetectedGPU],
                'gpu_matches': List[List[HardwareDetectionResult]]
            }
        """
        results = {
            'os': self.os_type,
            'platform': self.platform_arch,
            'cpu': None,
            'cpu_matches': [],
            'gpus': [],
            'gpu_matches': []
        }

        # Detect CPU
        cpu = self.detect_cpu()
        if cpu:
            results['cpu'] = cpu
            results['cpu_matches'] = self.match_cpu_to_database(cpu, db)

        # Detect GPUs
        gpus = self.detect_gpu()
        results['gpus'] = gpus

        for gpu in gpus:
            matches = self.match_gpu_to_database(gpu, db)
            results['gpu_matches'].append(matches)

        return results
