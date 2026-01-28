"""
CPU Clock Query Module

Provides functions to query current CPU clock frequencies for accurate
calibration and efficiency calculations.

Supports:
- Linux: /proc/cpuinfo, /sys/devices/system/cpu/
- macOS: sysctl
- Cross-platform: psutil (fallback)

Usage:
    from graphs.hardware.calibration.cpu_clock import get_cpu_clock_info

    clock_info = get_cpu_clock_info()
    print(f"Current Freq: {clock_info.current_freq_mhz} MHz")
    print(f"Max Freq: {clock_info.max_freq_mhz} MHz")
"""

import os
import re
import subprocess
import platform
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class CPUClockInfo:
    """CPU clock frequency information."""

    # Current clock frequencies (MHz)
    current_freq_mhz: Optional[float] = None
    """Current CPU frequency in MHz (average across cores)."""

    min_freq_mhz: Optional[float] = None
    """Minimum CPU frequency in MHz (scaling_min_freq)."""

    max_freq_mhz: Optional[float] = None
    """Maximum CPU frequency in MHz (scaling_max_freq)."""

    base_freq_mhz: Optional[float] = None
    """Base/nominal CPU frequency in MHz (from cpuinfo_max_freq or spec)."""

    # Per-core frequencies (for heterogeneous CPUs like big.LITTLE or P/E cores)
    per_core_freq_mhz: List[float] = field(default_factory=list)
    """Per-core frequencies in MHz (index = core number)."""

    # Frequency scaling
    governor: Optional[str] = None
    """CPU frequency governor (e.g., 'performance', 'powersave', 'ondemand')."""

    driver: Optional[str] = None
    """CPU frequency scaling driver (e.g., 'intel_pstate', 'acpi-cpufreq')."""

    # Turbo/boost state
    turbo_enabled: Optional[bool] = None
    """Whether turbo boost is enabled."""

    # Jetson-specific (nvpmodel)
    nvpmodel_mode: Optional[int] = None
    """Jetson NVPModel power mode (0=MAXN, 1=15W, etc.)."""

    power_mode_name: Optional[str] = None
    """Human-readable power mode name (e.g., 'MAXN', '15W', '25W')."""

    # Query metadata
    query_method: str = "unknown"
    """Method used to query clocks: 'sysfs', 'cpuinfo', 'sysctl', 'psutil'."""

    query_success: bool = False
    """Whether clock query was successful."""

    error_message: Optional[str] = None
    """Error message if query failed."""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != []:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'CPUClockInfo':
        """Create from dictionary."""
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def _query_linux_sysfs() -> Optional[CPUClockInfo]:
    """
    Query CPU frequencies from Linux sysfs.

    Most reliable method on Linux systems with cpufreq support.
    """
    try:
        info = CPUClockInfo(query_method='sysfs')
        cpu_base = '/sys/devices/system/cpu'

        if not os.path.exists(cpu_base):
            return None

        # Collect per-core frequencies
        per_core_freqs = []
        core_id = 0

        while True:
            core_path = f'{cpu_base}/cpu{core_id}/cpufreq'
            if not os.path.exists(core_path):
                break

            # Read current frequency (in kHz)
            cur_freq_path = f'{core_path}/scaling_cur_freq'
            if os.path.exists(cur_freq_path):
                with open(cur_freq_path) as f:
                    freq_khz = int(f.read().strip())
                    per_core_freqs.append(freq_khz / 1000.0)  # Convert to MHz

            core_id += 1

        if not per_core_freqs:
            return None

        info.per_core_freq_mhz = per_core_freqs
        info.current_freq_mhz = sum(per_core_freqs) / len(per_core_freqs)

        # Read min/max frequencies from first core (typically same for all)
        core0_path = f'{cpu_base}/cpu0/cpufreq'

        if os.path.exists(f'{core0_path}/scaling_min_freq'):
            with open(f'{core0_path}/scaling_min_freq') as f:
                info.min_freq_mhz = int(f.read().strip()) / 1000.0

        if os.path.exists(f'{core0_path}/scaling_max_freq'):
            with open(f'{core0_path}/scaling_max_freq') as f:
                info.max_freq_mhz = int(f.read().strip()) / 1000.0

        if os.path.exists(f'{core0_path}/cpuinfo_max_freq'):
            with open(f'{core0_path}/cpuinfo_max_freq') as f:
                info.base_freq_mhz = int(f.read().strip()) / 1000.0

        if os.path.exists(f'{core0_path}/scaling_governor'):
            with open(f'{core0_path}/scaling_governor') as f:
                info.governor = f.read().strip()

        if os.path.exists(f'{core0_path}/scaling_driver'):
            with open(f'{core0_path}/scaling_driver') as f:
                info.driver = f.read().strip()

        # Check turbo/boost state
        # Intel pstate
        intel_turbo_path = '/sys/devices/system/cpu/intel_pstate/no_turbo'
        if os.path.exists(intel_turbo_path):
            with open(intel_turbo_path) as f:
                # no_turbo = 1 means turbo disabled
                info.turbo_enabled = f.read().strip() == '0'

        # Generic boost
        boost_path = '/sys/devices/system/cpu/cpufreq/boost'
        if info.turbo_enabled is None and os.path.exists(boost_path):
            with open(boost_path) as f:
                info.turbo_enabled = f.read().strip() == '1'

        info.query_success = True
        return info

    except Exception as e:
        return CPUClockInfo(
            query_method='sysfs',
            query_success=False,
            error_message=str(e)
        )


def _query_linux_cpuinfo() -> Optional[CPUClockInfo]:
    """
    Query CPU frequencies from /proc/cpuinfo.

    Fallback for systems without cpufreq sysfs.
    """
    try:
        info = CPUClockInfo(query_method='cpuinfo')

        if not os.path.exists('/proc/cpuinfo'):
            return None

        with open('/proc/cpuinfo') as f:
            content = f.read()

        # Parse "cpu MHz" lines
        import re
        freq_matches = re.findall(r'cpu MHz\s*:\s*([\d.]+)', content)

        if not freq_matches:
            return None

        per_core_freqs = [float(f) for f in freq_matches]
        info.per_core_freq_mhz = per_core_freqs
        info.current_freq_mhz = sum(per_core_freqs) / len(per_core_freqs)
        info.query_success = True

        return info

    except Exception as e:
        return CPUClockInfo(
            query_method='cpuinfo',
            query_success=False,
            error_message=str(e)
        )


def _query_macos_sysctl() -> Optional[CPUClockInfo]:
    """
    Query CPU frequencies on macOS using sysctl.

    Note: macOS doesn't expose current frequency, only nominal/max.
    """
    try:
        info = CPUClockInfo(query_method='sysctl')

        # Query CPU frequency
        result = subprocess.run(
            ['sysctl', '-n', 'hw.cpufrequency_max'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            freq_hz = int(result.stdout.strip())
            info.max_freq_mhz = freq_hz / 1_000_000.0
            info.base_freq_mhz = info.max_freq_mhz
            # macOS doesn't report current frequency
            info.current_freq_mhz = info.max_freq_mhz
            info.query_success = True

        return info if info.query_success else None

    except Exception as e:
        return CPUClockInfo(
            query_method='sysctl',
            query_success=False,
            error_message=str(e)
        )


def _query_windows_power_plan() -> Optional[str]:
    """
    Query Windows power plan name.

    Returns power plan name like 'High performance', 'Balanced', 'Power saver'.
    """
    if platform.system() != 'Windows':
        return None

    try:
        # Use powercfg to get active power plan
        result = subprocess.run(
            ['powercfg', '/getactivescheme'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout:
            # Output format: "Power Scheme GUID: <guid>  (High performance)"
            # Extract the name in parentheses
            import re
            match = re.search(r'\(([^)]+)\)', result.stdout)
            if match:
                plan_name = match.group(1)
                # Normalize: "High performance" -> "HighPerformance"
                return plan_name.replace(' ', '')

        return None
    except Exception:
        return None


def _query_psutil() -> Optional[CPUClockInfo]:
    """
    Query CPU frequencies using psutil (cross-platform fallback).
    """
    try:
        import psutil

        info = CPUClockInfo(query_method='psutil')

        # Get frequency info
        freq = psutil.cpu_freq(percpu=False)
        if freq:
            info.current_freq_mhz = freq.current
            info.min_freq_mhz = freq.min if freq.min else None
            info.max_freq_mhz = freq.max if freq.max else None
            info.query_success = True

        # Try per-cpu frequencies
        per_cpu_freq = psutil.cpu_freq(percpu=True)
        if per_cpu_freq:
            info.per_core_freq_mhz = [f.current for f in per_cpu_freq]

        # On Windows, try to get power plan as "governor" equivalent
        if platform.system() == 'Windows' and info.governor is None:
            power_plan = _query_windows_power_plan()
            if power_plan:
                info.governor = power_plan

        return info if info.query_success else None

    except ImportError:
        return None
    except Exception as e:
        return CPUClockInfo(
            query_method='psutil',
            query_success=False,
            error_message=str(e)
        )


def _query_jetson_nvpmodel() -> Optional[Dict]:
    """
    Query Jetson power mode using nvpmodel.

    On Jetson systems, nvpmodel controls both CPU and GPU power limits.

    Returns:
        Dict with nvpmodel_mode and power_mode_name, or None if not available.
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

        # Parse output like: "NV Power Mode: MAXN" or "NV Power Mode: 15W"
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


def get_cpu_clock_info() -> CPUClockInfo:
    """
    Query current CPU clock frequencies.

    Tries multiple methods in order of preference based on platform:
    - Linux: sysfs -> cpuinfo -> psutil
    - macOS: sysctl -> psutil
    - Windows/Other: psutil

    On Jetson systems, also queries nvpmodel for power mode info.

    Returns:
        CPUClockInfo with available clock data

    Example:
        >>> info = get_cpu_clock_info()
        >>> print(f"Current: {info.current_freq_mhz:.0f} MHz")
        >>> print(f"Max: {info.max_freq_mhz:.0f} MHz")
        >>> print(f"Governor: {info.governor}")
        >>> print(f"Power Mode: {info.power_mode_name}")  # On Jetson
    """
    system = platform.system()
    info = None

    if system == 'Linux':
        # Try sysfs first (most detailed)
        info = _query_linux_sysfs()
        if not (info and info.query_success):
            # Try /proc/cpuinfo
            info = _query_linux_cpuinfo()

    elif system == 'Darwin':
        # macOS
        info = _query_macos_sysctl()

    # Fallback to psutil (cross-platform)
    if not (info and info.query_success):
        info = _query_psutil()

    # If still no success, return failure
    if not (info and info.query_success):
        return CPUClockInfo(
            query_method='none',
            query_success=False,
            error_message='No CPU clock query method available'
        )

    # On Linux, check for Jetson nvpmodel (controls both CPU and GPU power modes)
    if system == 'Linux':
        nvp = _query_jetson_nvpmodel()
        if nvp:
            info.nvpmodel_mode = nvp.get('nvpmodel_mode')
            info.power_mode_name = nvp.get('power_mode_name')

    return info


def print_cpu_clock_info():
    """Print current CPU clock information to stdout."""
    info = get_cpu_clock_info()

    print("\n" + "=" * 50)
    print("CPU CLOCK INFORMATION")
    print("=" * 50)

    if not info.query_success:
        print(f"Query failed: {info.error_message}")
        print(f"Method tried: {info.query_method}")
        return

    print(f"Query method: {info.query_method}")
    print()

    if info.current_freq_mhz:
        print(f"Current Freq: {info.current_freq_mhz:.0f} MHz", end="")
        if info.max_freq_mhz:
            pct = info.current_freq_mhz / info.max_freq_mhz * 100
            print(f" ({pct:.0f}% of max)")
        else:
            print()

    if info.min_freq_mhz and info.max_freq_mhz:
        print(f"Freq Range:   {info.min_freq_mhz:.0f} - {info.max_freq_mhz:.0f} MHz")

    if info.base_freq_mhz:
        print(f"Base Freq:    {info.base_freq_mhz:.0f} MHz")

    if info.governor:
        print(f"Governor:     {info.governor}")

    if info.driver:
        print(f"Driver:       {info.driver}")

    if info.turbo_enabled is not None:
        print(f"Turbo Boost:  {'Enabled' if info.turbo_enabled else 'Disabled'}")

    if info.per_core_freq_mhz and len(info.per_core_freq_mhz) > 1:
        min_core = min(info.per_core_freq_mhz)
        max_core = max(info.per_core_freq_mhz)
        if max_core - min_core > 100:  # Show spread if significant
            print(f"Core Spread:  {min_core:.0f} - {max_core:.0f} MHz")

    print()


if __name__ == '__main__':
    print_cpu_clock_info()
