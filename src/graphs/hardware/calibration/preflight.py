"""
Pre-flight Checks for Hardware Calibration

Validates system state before running calibration benchmarks to ensure
results are meaningful and representative of peak performance.

Checks include:
- CPU frequency governor (should be 'performance')
- CPU frequency vs max (should be > 90%)
- GPU power state and clocks
- System load (should be minimal)
- Thermal state (not throttling)

Usage:
    from graphs.hardware.calibration.preflight import run_preflight_checks

    result = run_preflight_checks(device='cpu')
    if not result.passed:
        print(result.format_report())
        if not force:
            sys.exit(1)
"""

import os
import platform
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class CheckStatus(Enum):
    """Status of a pre-flight check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    status: CheckStatus
    message: str
    current_value: Optional[str] = None
    expected_value: Optional[str] = None
    fix_command: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.PASSED, CheckStatus.SKIPPED)

    @property
    def symbol(self) -> str:
        return {
            CheckStatus.PASSED: "✓",
            CheckStatus.WARNING: "⚠",
            CheckStatus.FAILED: "✗",
            CheckStatus.SKIPPED: "○",
        }[self.status]


@dataclass
class PreflightReport:
    """Complete pre-flight check report."""
    checks: List[CheckResult] = field(default_factory=list)
    device: str = "cpu"
    timestamp: str = ""

    @property
    def passed(self) -> bool:
        """True if all checks passed (warnings are OK)."""
        return all(c.status != CheckStatus.FAILED for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """True if any checks have warnings."""
        return any(c.status == CheckStatus.WARNING for c in self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        """List of failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAILED]

    @property
    def warning_checks(self) -> List[CheckResult]:
        """List of warning checks."""
        return [c for c in self.checks if c.status == CheckStatus.WARNING]

    def add(self, check: CheckResult):
        """Add a check result."""
        self.checks.append(check)

    def format_report(self) -> str:
        """Format the report for display."""
        lines = []
        lines.append("=" * 70)
        lines.append("PRE-FLIGHT CHECKS")
        lines.append("=" * 70)
        lines.append("")

        # Group by status
        for check in self.checks:
            status_str = f"{check.symbol} {check.name}: {check.message}"
            lines.append(status_str)

            if check.current_value and check.expected_value:
                lines.append(f"    Current:  {check.current_value}")
                lines.append(f"    Expected: {check.expected_value}")

            if check.fix_command and check.status == CheckStatus.FAILED:
                lines.append(f"    Fix: {check.fix_command}")

        lines.append("")

        # Summary
        if self.passed:
            if self.has_warnings:
                lines.append("RESULT: PASSED with warnings")
                lines.append("  Calibration will proceed, but results may not represent peak performance.")
            else:
                lines.append("RESULT: PASSED")
                lines.append("  System is ready for calibration.")
        else:
            lines.append("RESULT: FAILED")
            lines.append("  Calibration aborted. Fix issues above or use --force to override.")
            lines.append("")
            lines.append("  Fix commands:")
            for check in self.failed_checks:
                if check.fix_command:
                    lines.append(f"    {check.fix_command}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'device': self.device,
            'timestamp': self.timestamp,
            'passed': self.passed,
            'has_warnings': self.has_warnings,
            'checks': [
                {
                    'name': c.name,
                    'status': c.status.value,
                    'message': c.message,
                    'current_value': c.current_value,
                    'expected_value': c.expected_value,
                }
                for c in self.checks
            ]
        }


def _check_cpu_governor() -> CheckResult:
    """Check CPU frequency governor."""
    try:
        # Read governor from first CPU
        governor_path = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
        if not os.path.exists(governor_path):
            return CheckResult(
                name="CPU Governor",
                status=CheckStatus.SKIPPED,
                message="cpufreq not available (VM or container?)"
            )

        with open(governor_path) as f:
            governor = f.read().strip()

        if governor == 'performance':
            return CheckResult(
                name="CPU Governor",
                status=CheckStatus.PASSED,
                message="performance",
                current_value=governor,
                expected_value="performance"
            )
        elif governor in ('powersave', 'conservative'):
            return CheckResult(
                name="CPU Governor",
                status=CheckStatus.FAILED,
                message=f"{governor} (low performance mode)",
                current_value=governor,
                expected_value="performance",
                fix_command="sudo cpupower frequency-set -g performance"
            )
        else:
            # ondemand, schedutil are OK but not ideal
            return CheckResult(
                name="CPU Governor",
                status=CheckStatus.WARNING,
                message=f"{governor} (dynamic - may vary)",
                current_value=governor,
                expected_value="performance",
                fix_command="sudo cpupower frequency-set -g performance"
            )

    except Exception as e:
        return CheckResult(
            name="CPU Governor",
            status=CheckStatus.SKIPPED,
            message=f"Could not read: {e}"
        )


def _check_cpu_frequency() -> CheckResult:
    """Check current CPU frequency vs maximum.

    Note: With intel_pstate driver in performance mode, idle frequency may be low
    but will ramp up under load. We check the governor separately and only fail
    here if governor is NOT performance and frequency is low.
    """
    try:
        from .cpu_clock import get_cpu_clock_info

        info = get_cpu_clock_info()
        if not info.query_success or not info.current_freq_mhz:
            return CheckResult(
                name="CPU Frequency",
                status=CheckStatus.SKIPPED,
                message="Could not query CPU frequency"
            )

        current = info.current_freq_mhz
        max_freq = info.max_freq_mhz or info.base_freq_mhz
        governor = info.governor

        if not max_freq:
            return CheckResult(
                name="CPU Frequency",
                status=CheckStatus.WARNING,
                message=f"{current:.0f} MHz (max unknown)",
                current_value=f"{current:.0f} MHz",
                expected_value="Unknown"
            )

        pct = current / max_freq * 100

        # Check if using intel_pstate driver
        driver = info.driver or ""
        using_intel_pstate = "intel_pstate" in driver.lower() or "pstate" in driver.lower()

        # With intel_pstate + performance governor, low idle frequency is expected
        # The CPU will ramp up under load - this is fine
        if using_intel_pstate and governor == "performance":
            if pct >= 90:
                return CheckResult(
                    name="CPU Frequency",
                    status=CheckStatus.PASSED,
                    message=f"{current:.0f} MHz ({pct:.0f}% of max)",
                    current_value=f"{current:.0f} MHz",
                    expected_value=f">= {max_freq * 0.9:.0f} MHz (90%)"
                )
            else:
                # Low idle freq with intel_pstate + performance is OK
                return CheckResult(
                    name="CPU Frequency",
                    status=CheckStatus.PASSED,
                    message=f"{current:.0f} MHz idle (intel_pstate will boost under load)",
                    current_value=f"{current:.0f} MHz (idle)",
                    expected_value=f"Up to {max_freq:.0f} MHz under load"
                )

        # Standard frequency check for other configurations
        if pct >= 90:
            return CheckResult(
                name="CPU Frequency",
                status=CheckStatus.PASSED,
                message=f"{current:.0f} MHz ({pct:.0f}% of max)",
                current_value=f"{current:.0f} MHz",
                expected_value=f">= {max_freq * 0.9:.0f} MHz (90%)"
            )
        elif pct >= 70:
            return CheckResult(
                name="CPU Frequency",
                status=CheckStatus.WARNING,
                message=f"{current:.0f} MHz ({pct:.0f}% of max) - below optimal",
                current_value=f"{current:.0f} MHz ({pct:.0f}%)",
                expected_value=f">= {max_freq * 0.9:.0f} MHz (90%)",
                fix_command="Reduce system load or set governor to 'performance'"
            )
        else:
            return CheckResult(
                name="CPU Frequency",
                status=CheckStatus.FAILED,
                message=f"{current:.0f} MHz ({pct:.0f}% of max) - severely throttled",
                current_value=f"{current:.0f} MHz ({pct:.0f}%)",
                expected_value=f">= {max_freq * 0.9:.0f} MHz (90%)",
                fix_command="sudo cpupower frequency-set -g performance"
            )

    except Exception as e:
        return CheckResult(
            name="CPU Frequency",
            status=CheckStatus.SKIPPED,
            message=f"Could not check: {e}"
        )


def _check_cpu_turbo() -> CheckResult:
    """Check if turbo boost is enabled."""
    try:
        # Check Intel pstate
        no_turbo_path = '/sys/devices/system/cpu/intel_pstate/no_turbo'
        if os.path.exists(no_turbo_path):
            with open(no_turbo_path) as f:
                no_turbo = f.read().strip()

            if no_turbo == '0':
                return CheckResult(
                    name="Turbo Boost",
                    status=CheckStatus.PASSED,
                    message="Enabled"
                )
            else:
                return CheckResult(
                    name="Turbo Boost",
                    status=CheckStatus.WARNING,
                    message="Disabled (reduced peak performance)",
                    current_value="Disabled",
                    expected_value="Enabled",
                    fix_command="echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
                )

        # Check generic boost
        boost_path = '/sys/devices/system/cpu/cpufreq/boost'
        if os.path.exists(boost_path):
            with open(boost_path) as f:
                boost = f.read().strip()

            if boost == '1':
                return CheckResult(
                    name="Turbo Boost",
                    status=CheckStatus.PASSED,
                    message="Enabled"
                )
            else:
                return CheckResult(
                    name="Turbo Boost",
                    status=CheckStatus.WARNING,
                    message="Disabled",
                    current_value="Disabled",
                    expected_value="Enabled"
                )

        return CheckResult(
            name="Turbo Boost",
            status=CheckStatus.SKIPPED,
            message="Could not determine (not Intel pstate)"
        )

    except Exception as e:
        return CheckResult(
            name="Turbo Boost",
            status=CheckStatus.SKIPPED,
            message=f"Could not check: {e}"
        )


def _check_system_load() -> CheckResult:
    """Check system CPU load."""
    try:
        import psutil
        load = psutil.cpu_percent(interval=0.5)

        if load < 5:
            return CheckResult(
                name="System Load",
                status=CheckStatus.PASSED,
                message=f"{load:.1f}% (idle)",
                current_value=f"{load:.1f}%",
                expected_value="< 5%"
            )
        elif load < 20:
            return CheckResult(
                name="System Load",
                status=CheckStatus.WARNING,
                message=f"{load:.1f}% (some background activity)",
                current_value=f"{load:.1f}%",
                expected_value="< 5%"
            )
        else:
            return CheckResult(
                name="System Load",
                status=CheckStatus.FAILED,
                message=f"{load:.1f}% (high - will affect results)",
                current_value=f"{load:.1f}%",
                expected_value="< 5%",
                fix_command="Close other applications or wait for background tasks"
            )

    except ImportError:
        return CheckResult(
            name="System Load",
            status=CheckStatus.SKIPPED,
            message="psutil not installed"
        )
    except Exception as e:
        return CheckResult(
            name="System Load",
            status=CheckStatus.SKIPPED,
            message=f"Could not check: {e}"
        )


def _check_thermal_throttling() -> CheckResult:
    """Check if CPU is thermally throttling."""
    try:
        # Try to read CPU temperature
        temp_paths = [
            '/sys/class/thermal/thermal_zone0/temp',
            '/sys/class/hwmon/hwmon0/temp1_input',
            '/sys/class/hwmon/hwmon1/temp1_input',
        ]

        temp_c = None
        for path in temp_paths:
            if os.path.exists(path):
                with open(path) as f:
                    temp_c = int(f.read().strip()) / 1000
                break

        if temp_c is None:
            return CheckResult(
                name="Thermal State",
                status=CheckStatus.SKIPPED,
                message="Could not read temperature"
            )

        if temp_c < 70:
            return CheckResult(
                name="Thermal State",
                status=CheckStatus.PASSED,
                message=f"{temp_c:.0f}°C (cool)",
                current_value=f"{temp_c:.0f}°C",
                expected_value="< 80°C"
            )
        elif temp_c < 85:
            return CheckResult(
                name="Thermal State",
                status=CheckStatus.WARNING,
                message=f"{temp_c:.0f}°C (warm - may throttle under load)",
                current_value=f"{temp_c:.0f}°C",
                expected_value="< 80°C"
            )
        else:
            return CheckResult(
                name="Thermal State",
                status=CheckStatus.FAILED,
                message=f"{temp_c:.0f}°C (hot - likely throttling)",
                current_value=f"{temp_c:.0f}°C",
                expected_value="< 80°C",
                fix_command="Allow system to cool down before calibrating"
            )

    except Exception as e:
        return CheckResult(
            name="Thermal State",
            status=CheckStatus.SKIPPED,
            message=f"Could not check: {e}"
        )


def _check_gpu_power_state() -> CheckResult:
    """Check GPU power state (NVIDIA).

    For Jetson platforms, all standard power profiles (7W, 15W, 25W, MAXN, etc.)
    are valid calibration configurations. Calibrating at a specific power profile
    is intentional - it measures performance at that power envelope.

    Only unknown/unexpected power modes trigger a warning.
    """
    try:
        from .gpu_clock import get_gpu_clock_info

        info = get_gpu_clock_info()
        if not info.query_success:
            return CheckResult(
                name="GPU Power State",
                status=CheckStatus.SKIPPED,
                message="Could not query GPU"
            )

        # Check Jetson power mode
        if info.power_mode_name:
            mode_upper = info.power_mode_name.upper()

            # All known Jetson power profiles are valid for calibration
            # Each profile represents a specific power/performance envelope
            # - MAXN/MAX: Maximum performance (no power limit)
            # - 7W, 10W, 15W, 25W, 30W, 50W, 60W: Power-limited profiles
            known_jetson_modes = {
                'MAXN', 'MAX',  # Maximum performance modes
                '7W', '10W', '15W', '20W', '25W', '30W', '50W', '60W',  # Power profiles
            }

            if mode_upper in known_jetson_modes:
                # Indicate if this is max performance or a power-limited profile
                if mode_upper in ('MAXN', 'MAX'):
                    desc = "maximum performance"
                else:
                    desc = f"power-limited profile"
                return CheckResult(
                    name="GPU Power Mode",
                    status=CheckStatus.PASSED,
                    message=f"{info.power_mode_name} ({desc})"
                )
            else:
                # Unknown power mode - warn but don't fail
                return CheckResult(
                    name="GPU Power Mode",
                    status=CheckStatus.WARNING,
                    message=f"{info.power_mode_name} (unknown mode)",
                    current_value=info.power_mode_name,
                    expected_value="Known mode (MAXN, 7W, 15W, 25W, etc.)",
                    fix_command="sudo nvpmodel -m 0  # for MAXN mode"
                )

        # Check SM clock percentage
        if info.sm_clock_mhz and info.max_sm_clock_mhz:
            pct = info.sm_clock_mhz / info.max_sm_clock_mhz * 100
            if pct >= 90:
                return CheckResult(
                    name="GPU Clock",
                    status=CheckStatus.PASSED,
                    message=f"{info.sm_clock_mhz} MHz ({pct:.0f}% of max)"
                )
            elif pct >= 70:
                return CheckResult(
                    name="GPU Clock",
                    status=CheckStatus.WARNING,
                    message=f"{info.sm_clock_mhz} MHz ({pct:.0f}% of max)",
                    current_value=f"{info.sm_clock_mhz} MHz",
                    expected_value=f">= {info.max_sm_clock_mhz * 0.9:.0f} MHz"
                )
            else:
                return CheckResult(
                    name="GPU Clock",
                    status=CheckStatus.FAILED,
                    message=f"{info.sm_clock_mhz} MHz ({pct:.0f}% of max) - throttled",
                    current_value=f"{info.sm_clock_mhz} MHz",
                    expected_value=f">= {info.max_sm_clock_mhz * 0.9:.0f} MHz"
                )

        return CheckResult(
            name="GPU Power State",
            status=CheckStatus.PASSED,
            message=f"SM Clock: {info.sm_clock_mhz} MHz"
        )

    except Exception as e:
        return CheckResult(
            name="GPU Power State",
            status=CheckStatus.SKIPPED,
            message=f"Could not check: {e}"
        )


def run_preflight_checks(device: str = 'cpu') -> PreflightReport:
    """
    Run all pre-flight checks for the specified device.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        PreflightReport with all check results
    """
    from datetime import datetime

    report = PreflightReport(
        device=device,
        timestamp=datetime.now().isoformat()
    )

    # CPU checks (always run - needed for both CPU and GPU calibration)
    report.add(_check_cpu_governor())
    report.add(_check_cpu_frequency())
    report.add(_check_cpu_turbo())
    report.add(_check_system_load())
    report.add(_check_thermal_throttling())

    # GPU checks (only for CUDA device)
    if device == 'cuda':
        report.add(_check_gpu_power_state())

    return report


def require_performance_mode(device: str = 'cpu', force: bool = False) -> PreflightReport:
    """
    Run pre-flight checks and raise if not in performance mode.

    Args:
        device: 'cpu' or 'cuda'
        force: If True, return report even if checks fail (don't raise)

    Returns:
        PreflightReport

    Raises:
        RuntimeError if checks fail and force=False
    """
    report = run_preflight_checks(device)

    print(report.format_report())

    if not report.passed and not force:
        raise RuntimeError(
            "Pre-flight checks failed. System is not in performance mode.\n"
            "Use --force to override (results will be flagged as non-representative)."
        )

    return report


if __name__ == '__main__':
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    report = run_preflight_checks(device)
    print(report.format_report())
    sys.exit(0 if report.passed else 1)
