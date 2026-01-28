#!/usr/bin/env python
"""
Power Profile Comparison CLI

Compare throughput and efficiency across different power profiles for hardware
with multiple calibrations (e.g., Jetson with 7W, 15W, 25W, MAXN modes).

Usage:
    # Compare all power profiles for a specific hardware
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu

    # List all hardware with multiple calibrations
    ./cli/compare_power_profiles.py --list

    # Show detailed per-operation breakdown
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --detailed

    # Export to CSV
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --csv output.csv

    # Filter by framework
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --framework pytorch
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.registry import get_registry, HardwareProfile
from graphs.calibration.schema import HardwareCalibration


@dataclass
class WorkloadResult:
    """Result for a specific workload (AXPY, GEMV, GEMM)."""
    best_gflops: float
    best_precision: str
    best_size: str


@dataclass
class PowerProfileSummary:
    """Summary of a single power profile calibration."""
    power_mode: str
    freq_mhz: int
    framework: str
    calibration_date: str

    # Peak specs (from calibration metadata)
    peak_compute_gflops: float
    peak_bandwidth_gbps: float

    # Measured bandwidth
    measured_bandwidth_gbps: float

    # Workload-specific results
    axpy_result: Optional[WorkloadResult] = None
    gemv_result: Optional[WorkloadResult] = None
    gemm_result: Optional[WorkloadResult] = None

    # Power estimation (from power mode name if available)
    estimated_watts: Optional[float] = None

    # Per-operation data for detailed view
    operation_data: Dict[str, Tuple[float, str]] = field(default_factory=dict)  # op_name -> (gflops, precision)

    # Per-precision best GFLOPS for each workload: {precision: {workload: (gflops, bandwidth)}}
    precision_workload_data: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)

    def get_efficiency(self, workload: str) -> Optional[float]:
        """Get efficiency for a specific workload."""
        result = getattr(self, f"{workload}_result", None)
        if result and self.peak_compute_gflops > 0:
            return result.best_gflops / self.peak_compute_gflops
        return None

    def get_gflops_per_watt(self, workload: str) -> Optional[float]:
        """Get GFLOPS/Watt for a specific workload."""
        result = getattr(self, f"{workload}_result", None)
        if result and self.estimated_watts and self.estimated_watts > 0:
            return result.best_gflops / self.estimated_watts
        return None


def extract_watts_from_power_mode(power_mode: str) -> Optional[float]:
    """Extract wattage from power mode name like '7W', '15W', '25W'."""
    match = re.match(r'^(\d+(?:\.\d+)?)\s*W$', power_mode, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def estimate_power_for_unknown_modes(summaries: List['PowerProfileSummary']) -> None:
    """Estimate power for modes without explicit wattage (e.g., MAXN_SUPER).

    Uses frequency-proportional scaling from the nearest known power mode.
    Power scales roughly with frequency (P ~ f * V^2, and V ~ f for DVFS).
    """
    # Find summaries with known watts, sorted by frequency
    known = [(s.freq_mhz, s.estimated_watts, s) for s in summaries if s.estimated_watts]
    unknown = [s for s in summaries if not s.estimated_watts]

    if not known or not unknown:
        return

    known.sort(key=lambda x: x[0])

    for s in unknown:
        # Find nearest known power mode by frequency
        nearest = min(known, key=lambda x: abs(x[0] - s.freq_mhz))
        nearest_freq, nearest_watts, _ = nearest

        if nearest_freq > 0 and nearest_watts:
            # Estimate power proportional to frequency ratio
            # This is a simplification; actual power scales ~f^3 for DVFS
            # but we use linear as a conservative estimate
            freq_ratio = s.freq_mhz / nearest_freq
            s.estimated_watts = nearest_watts * freq_ratio


def parse_operation_name(op_name: str) -> Dict[str, str]:
    """Parse operation name to extract components like precision, size, operation type."""
    result = {}

    # Extract key=value pairs
    for part in op_name.split('_'):
        if '=' in part:
            key, value = part.split('=', 1)
            result[key] = value

    # Determine workload type
    if 'axpy' in op_name.lower():
        result['workload'] = 'axpy'
    elif 'gemv' in op_name.lower():
        result['workload'] = 'gemv'
    elif 'gemm' in op_name.lower():
        result['workload'] = 'gemm'

    return result


def find_best_for_workload(operation_profiles: Dict, workload: str) -> Optional[WorkloadResult]:
    """Find the best performing operation for a given workload type."""
    best_gflops = 0.0
    best_precision = ""
    best_size = ""

    for op_name, op_cal in operation_profiles.items():
        if not op_cal.measured_gflops:
            continue

        parsed = parse_operation_name(op_name)
        if parsed.get('workload') != workload:
            continue

        if op_cal.measured_gflops > best_gflops:
            best_gflops = op_cal.measured_gflops
            best_precision = parsed.get('precision', 'unknown')
            best_size = parsed.get('size', 'unknown')

    if best_gflops > 0:
        return WorkloadResult(
            best_gflops=best_gflops,
            best_precision=best_precision,
            best_size=best_size
        )
    return None


def get_peak_for_precision(profile: HardwareProfile, precision: str) -> float:
    """Get the theoretical peak GFLOPS for a given precision."""
    peaks = profile.theoretical_peaks

    # Map precision names to spec keys
    precision_map = {
        'fp64': 'fp64',
        'fp32': 'fp32',
        'fp16': 'fp16',
        'bf16': 'bf16',
        'tf32': 'tf32',
        'int8': 'int8',
        'int16': 'int16',
        'int32': 'int32',
        'int64': 'int64',
    }

    key = precision_map.get(precision.lower(), 'fp32')
    return peaks.get(key, peaks.get('fp32', 0))


def load_calibration_summary(cal_path: Path, cal_info: Dict, profile: HardwareProfile) -> PowerProfileSummary:
    """Load a calibration file and extract summary metrics."""
    calibration = HardwareCalibration.load(cal_path)

    # Get power mode from calibration metadata if available
    power_mode = cal_info['power_mode']
    if calibration.metadata.gpu_clock and calibration.metadata.gpu_clock.power_mode_name:
        power_mode = calibration.metadata.gpu_clock.power_mode_name

    # Get peak specs - use calibration metadata if available, else from profile
    freq_mhz = cal_info['freq_mhz']

    # Calculate frequency ratio for scaling peaks
    # The spec has peak at boost clock, scale for actual frequency
    boost_clock = profile.boost_clock_mhz or profile.base_clock_mhz or freq_mhz
    freq_ratio = freq_mhz / boost_clock if boost_clock > 0 else 1.0

    peak_bandwidth = profile.peak_bandwidth_gbps

    # Find best for each workload
    axpy_result = find_best_for_workload(calibration.operation_profiles, 'axpy')
    gemv_result = find_best_for_workload(calibration.operation_profiles, 'gemv')
    gemm_result = find_best_for_workload(calibration.operation_profiles, 'gemm')

    # Get peak compute based on GEMM precision (the most relevant workload)
    # This ensures efficiency is calculated against the correct precision peak
    if gemm_result:
        peak_at_boost = get_peak_for_precision(profile, gemm_result.best_precision)
    else:
        # Fallback to FP32 if no GEMM result
        peak_at_boost = profile.theoretical_peaks.get('fp32', 0)

    # Scale by frequency ratio
    peak_compute_at_freq = peak_at_boost * freq_ratio

    # Collect all operation data for detailed view
    operation_data = {}
    for op_name, op_cal in calibration.operation_profiles.items():
        if op_cal.measured_gflops:
            parsed = parse_operation_name(op_name)
            precision = parsed.get('precision', 'unknown')
            operation_data[op_name] = (op_cal.measured_gflops, precision)

    # Extract per-precision best GFLOPS and BW for each workload from precision_results
    # Data structure: {precision: {workload: (gflops, bandwidth)}}
    precision_workload_data: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for op_name, op_cal in calibration.operation_profiles.items():
        # Determine workload type
        if 'axpy' in op_name.lower():
            workload = 'axpy'
        elif 'gemv' in op_name.lower():
            workload = 'gemv'
        elif 'gemm' in op_name.lower():
            workload = 'gemm'
        else:
            continue  # Skip non-BLAS operations

        # Extract from precision_results if available
        if hasattr(op_cal, 'precision_results') and op_cal.precision_results:
            for prec_name, prec_result in op_cal.precision_results.items():
                if prec_result.supported and prec_result.measured_gops:
                    if prec_name not in precision_workload_data:
                        precision_workload_data[prec_name] = {}
                    # Keep track of the best GFLOPS for this precision/workload combo
                    current_best_gops, current_best_bw = precision_workload_data[prec_name].get(workload, (0, 0))
                    if prec_result.measured_gops > current_best_gops:
                        bw = prec_result.achieved_bandwidth_gbps if hasattr(prec_result, 'achieved_bandwidth_gbps') else 0
                        precision_workload_data[prec_name][workload] = (prec_result.measured_gops, bw)

    return PowerProfileSummary(
        power_mode=power_mode,
        freq_mhz=freq_mhz,
        framework=cal_info['framework'],
        calibration_date=calibration.metadata.calibration_date,
        peak_compute_gflops=peak_compute_at_freq,
        peak_bandwidth_gbps=peak_bandwidth,
        measured_bandwidth_gbps=calibration.measured_bandwidth_gbps,
        axpy_result=axpy_result,
        gemv_result=gemv_result,
        gemm_result=gemm_result,
        estimated_watts=extract_watts_from_power_mode(power_mode),
        operation_data=operation_data,
        precision_workload_data=precision_workload_data,
    )


def find_best_for_workload_and_precision(operation_profiles: Dict, workload: str, precision: str) -> Optional[float]:
    """Find the best performing operation for a given workload type and precision."""
    best_gflops = 0.0

    for op_name, op_cal in operation_profiles.items():
        if not op_cal.measured_gflops:
            continue

        parsed = parse_operation_name(op_name)
        if parsed.get('workload') != workload:
            continue
        if parsed.get('precision', '').lower() != precision.lower():
            continue

        if op_cal.measured_gflops > best_gflops:
            best_gflops = op_cal.measured_gflops

    return best_gflops if best_gflops > 0 else None


def print_power_profile_table(profile: HardwareProfile, summaries: List[PowerProfileSummary]):
    """Print a comparison table of power profiles."""
    print(f"\n{'='*120}")
    print(f"Power Profile Comparison: {profile.vendor} {profile.model}")
    print(f"{'='*120}")

    # Show framework(s) used
    frameworks = sorted(set(s.framework for s in summaries))
    print(f"Framework: {', '.join(frameworks)}")
    print()

    # Show spec peaks for reference
    peaks = profile.theoretical_peaks
    print(f"Spec Peaks @ {profile.boost_clock_mhz or profile.base_clock_mhz} MHz:")
    print(f"  FP32: {peaks.get('fp32', 0):,.0f} GFLOPS | FP16: {peaks.get('fp16', 0):,.0f} GFLOPS | "
          f"BF16: {peaks.get('bf16', 0):,.0f} GFLOPS | TF32: {peaks.get('tf32', 0):,.0f} GFLOPS | "
          f"INT8: {peaks.get('int8', 0):,.0f} GOPS")
    print(f"  Memory Bandwidth: {profile.peak_bandwidth_gbps:,.1f} GB/s")
    print()

    # Sort by frequency ascending
    summaries = sorted(summaries, key=lambda s: s.freq_mhz)

    # Estimate power for modes without explicit wattage
    estimate_power_for_unknown_modes(summaries)

    # Precisions to report (in order)
    precisions_to_report = ['fp32', 'tf32', 'bf16', 'fp16', 'int8']

    for precision in precisions_to_report:
        # Check if this precision is supported (has a non-zero peak)
        peak_at_boost = peaks.get(precision, 0)
        if peak_at_boost == 0:
            print(f"[{precision.upper()}] Not supported by this hardware")
            print()
            continue

        # Check if we have any data for this precision using precision_workload_data
        has_data = False
        for s in summaries:
            if precision in s.precision_workload_data:
                has_data = True
                break

        if not has_data:
            print(f"[{precision.upper()}] No calibration data available")
            print()
            continue

        # Print table for this precision with three segments:
        # 1. Speed of Light (theoretical): Freq, Peak GOPS, Peak BW
        # 2. Measured: AXPY (GOPS/BW), GEMV (GOPS/BW), GEMM (GOPS/BW)
        # 3. Efficiency: Watts, GOPS/W
        unit = "GOPS" if precision == 'int8' else "GFLOPS"
        peak_bw = profile.peak_bandwidth_gbps

        print(f"[{precision.upper()}] Spec peak @ {profile.boost_clock_mhz or profile.base_clock_mhz:.0f} MHz: "
              f"{peak_at_boost:,.0f} {unit}, {peak_bw:.0f} GB/s")

        # Column widths (must match data row formatting exactly)
        # Data row: f"{s.power_mode:<12} {s.freq_mhz:>6} {peak_at_freq:>10.0f} {peak_bw:>6.0f} | "
        #           f"{axpy_gops_str} {axpy_bw_str} | {gemv_gops_str} {gemv_bw_str} | {gemm_gops_str} {gemm_bw_str} | "
        #           f"{watts_str} {gpw_str}"
        # where: axpy_gops_str = f"{val:>8.1f}", axpy_bw_str = f"{val:>6.1f}", etc.

        # Speed of Light segment: Freq(6) + space + Peak(10) + space + BW(6) = 24 chars
        # Measured segment: [AXPY(8) + space + BW(6) + " | "] * 3 - last " | " = 3*(8+1+6+3) - 3 = 51 chars
        # Efficiency segment: Watts(6) + space + GPW(10) = 17 chars

        sol_cols = f"{'Freq':>6} {'Peak':>10} {'BW':>6}"  # 24 chars
        # Measured has 3 workloads, each: 8 + 1 + 6 = 15, plus " | " between = 15*3 + 2*3 = 51
        meas_cols = f"{'AXPY':>8} {'BW':>6} | {'GEMV':>8} {'BW':>6} | {'GEMM':>8} {'BW':>6}"  # 51 chars
        eff_cols = f"{'Watts':>6} {unit + '/W':>10}"  # 17 chars

        # Calculate segment header widths to match column widths
        sol_width = len(sol_cols)  # 24
        meas_width = len(meas_cols)  # 51
        eff_width = len(eff_cols)  # 17

        total_width = 12 + 1 + sol_width + 3 + meas_width + 3 + eff_width  # 12 + 1 + 24 + 3 + 51 + 3 + 17 = 111
        print("-" * total_width)

        # Header row 1 - segment labels (centered over their columns)
        print(f"{'':12} {'Speed of Light':^{sol_width}} | {'Measured':^{meas_width}} | {'Efficiency':^{eff_width}}")

        # Header row 2 - column names
        print(f"{'Power Mode':<12} {sol_cols} | {meas_cols} | {eff_cols}")

        # Header row 3 - units
        unit_short = unit[:6] if len(unit) > 6 else unit
        unit_row_sol = f"{'MHz':>6} {unit_short:>10} {'GB/s':>6}"
        unit_row_meas = f"{unit_short:>8} {'GB/s':>6} | {unit_short:>8} {'GB/s':>6} | {unit_short:>8} {'GB/s':>6}"
        unit_row_eff = f"{'':>6} {'':>10}"
        print(f"{'':12} {unit_row_sol} | {unit_row_meas} | {unit_row_eff}")
        print("-" * total_width)

        for s in summaries:
            # Calculate speed of light at this frequency
            boost_clock = profile.boost_clock_mhz or profile.base_clock_mhz or s.freq_mhz
            freq_ratio = s.freq_mhz / boost_clock if boost_clock > 0 else 1.0
            peak_at_freq = peak_at_boost * freq_ratio
            # Note: BW doesn't scale with GPU frequency (it's memory frequency dependent)
            # but we show the spec peak BW for comparison

            # Get workload results for this precision from precision_workload_data
            prec_data = s.precision_workload_data.get(precision, {})
            axpy_data = prec_data.get('axpy', (None, None))
            gemv_data = prec_data.get('gemv', (None, None))
            gemm_data = prec_data.get('gemm', (None, None))

            # Unpack (gflops, bw) tuples
            axpy_gflops, axpy_bw = axpy_data if isinstance(axpy_data, tuple) else (axpy_data, None)
            gemv_gflops, gemv_bw = gemv_data if isinstance(gemv_data, tuple) else (gemv_data, None)
            gemm_gflops, gemm_bw = gemm_data if isinstance(gemm_data, tuple) else (gemm_data, None)

            # Format measured values (GOPS and BW for each workload)
            axpy_gops_str = f"{axpy_gflops:>8.1f}" if axpy_gflops else f"{'N/A':>8}"
            axpy_bw_str = f"{axpy_bw:>6.1f}" if axpy_bw else f"{'N/A':>6}"
            gemv_gops_str = f"{gemv_gflops:>8.1f}" if gemv_gflops else f"{'N/A':>8}"
            gemv_bw_str = f"{gemv_bw:>6.1f}" if gemv_bw else f"{'N/A':>6}"
            gemm_gops_str = f"{gemm_gflops:>8.1f}" if gemm_gflops else f"{'N/A':>8}"
            gemm_bw_str = f"{gemm_bw:>6.1f}" if gemm_bw else f"{'N/A':>6}"

            # Watts (with ~ prefix if estimated)
            if s.estimated_watts:
                original_watts = extract_watts_from_power_mode(s.power_mode)
                if original_watts:
                    watts_str = f"{s.estimated_watts:>6.1f}"
                else:
                    watts_str = f"~{s.estimated_watts:>5.1f}"
            else:
                watts_str = f"{'N/A':>6}"

            # GFLOPS/W (use GEMM as it's the compute-bound reference)
            if gemm_gflops and s.estimated_watts and s.estimated_watts > 0:
                gpw = gemm_gflops / s.estimated_watts
                gpw_str = f"{gpw:>10.1f}"
            else:
                gpw_str = f"{'N/A':>10}"

            print(f"{s.power_mode:<12} {s.freq_mhz:>6} {peak_at_freq:>10.0f} {peak_bw:>6.0f} | "
                  f"{axpy_gops_str} {axpy_bw_str} | {gemv_gops_str} {gemv_bw_str} | {gemm_gops_str} {gemm_bw_str} | "
                  f"{watts_str} {gpw_str}")

        print()

    # Scaling analysis (use best GEMM across all precisions)
    watt_summaries = [s for s in summaries if s.estimated_watts and s.gemm_result]
    if len(watt_summaries) >= 2:
        print("Scaling Analysis (Best GEMM):")
        print("-" * 60)
        base = watt_summaries[0]
        for s in watt_summaries[1:]:
            power_ratio = s.estimated_watts / base.estimated_watts if base.estimated_watts else 1
            perf_ratio = s.gemm_result.best_gflops / base.gemm_result.best_gflops if base.gemm_result.best_gflops else 1
            base_gpw = base.gemm_result.best_gflops / base.estimated_watts if base.estimated_watts else 0
            s_gpw = s.gemm_result.best_gflops / s.estimated_watts if s.estimated_watts else 0
            efficiency_change = ((s_gpw / base_gpw) - 1) * 100 if base_gpw and s_gpw else 0

            print(f"  {base.power_mode} -> {s.power_mode}: "
                  f"{power_ratio:.1f}x power, {perf_ratio:.1f}x performance, "
                  f"efficiency {efficiency_change:+.1f}%")
        print()


def print_detailed_operations(summaries: List[PowerProfileSummary]):
    """Print detailed per-operation comparison."""
    print("\nPer-Operation Performance (GFLOPS):")
    print("=" * 140)

    # Sort summaries by frequency
    summaries = sorted(summaries, key=lambda s: s.freq_mhz)

    # Collect all operations
    all_ops = set()
    for s in summaries:
        all_ops.update(s.operation_data.keys())

    if not all_ops:
        print("  No per-operation data available.")
        return

    # Group by workload type
    workloads = {'axpy': [], 'gemv': [], 'gemm': [], 'other': []}
    for op in all_ops:
        parsed = parse_operation_name(op)
        wl = parsed.get('workload', 'other')
        if wl in workloads:
            workloads[wl].append(op)
        else:
            workloads['other'].append(op)

    # Header
    header = f"{'Operation':<50} {'Prec':<8}"
    for s in summaries:
        header += f" {s.power_mode:>12}"
    print(header)
    print("-" * 140)

    for wl_name in ['axpy', 'gemv', 'gemm', 'other']:
        ops = sorted(workloads[wl_name])
        if not ops:
            continue

        print(f"\n{wl_name.upper()}:")
        for op in ops:
            # Extract precision and size for display
            parsed = parse_operation_name(op)
            precision = parsed.get('precision', '')
            size = parsed.get('size', '')

            # Shorten operation name for display
            display_name = f"  size={size}" if size else op[:48]

            row = f"{display_name:<50} {precision:<8}"
            for s in summaries:
                val, _ = s.operation_data.get(op, (None, None))
                if val:
                    row += f" {val:>12,.1f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    print()


def export_csv(profile: HardwareProfile, summaries: List[PowerProfileSummary], output_path: Path):
    """Export comparison data to CSV."""
    # Sort by frequency
    summaries = sorted(summaries, key=lambda s: s.freq_mhz)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Hardware ID', 'Vendor', 'Model',
            'Power Mode', 'Frequency MHz', 'Framework', 'Calibration Date',
            'Peak Compute GFLOPS', 'Peak Bandwidth GB/s', 'Measured Bandwidth GB/s',
            'AXPY GFLOPS', 'AXPY Precision', 'AXPY Size',
            'GEMV GFLOPS', 'GEMV Precision', 'GEMV Size',
            'GEMM GFLOPS', 'GEMM Precision', 'GEMM Size', 'GEMM Efficiency',
            'Estimated Watts', 'GEMM GFLOPS/Watt'
        ])

        # Data rows
        for s in summaries:
            gemm_eff = s.get_efficiency('gemm')
            gemm_gpw = s.get_gflops_per_watt('gemm')

            writer.writerow([
                profile.id, profile.vendor, profile.model,
                s.power_mode, s.freq_mhz, s.framework, s.calibration_date,
                s.peak_compute_gflops, s.peak_bandwidth_gbps, s.measured_bandwidth_gbps,
                s.axpy_result.best_gflops if s.axpy_result else '',
                s.axpy_result.best_precision if s.axpy_result else '',
                s.axpy_result.best_size if s.axpy_result else '',
                s.gemv_result.best_gflops if s.gemv_result else '',
                s.gemv_result.best_precision if s.gemv_result else '',
                s.gemv_result.best_size if s.gemv_result else '',
                s.gemm_result.best_gflops if s.gemm_result else '',
                s.gemm_result.best_precision if s.gemm_result else '',
                s.gemm_result.best_size if s.gemm_result else '',
                gemm_eff if gemm_eff else '',
                s.estimated_watts or '',
                gemm_gpw if gemm_gpw else ''
            ])

    print(f"Exported to: {output_path}")


def list_hardware_with_calibrations(registry):
    """List all hardware that has multiple calibrations."""
    print("\nHardware with Multiple Power Profiles:")
    print("=" * 70)

    profiles = registry.list_all()
    found = False

    for pid in sorted(profiles):
        profile = registry.get(pid)
        # Get calibrations
        profile_dir = registry.path / profile.device_type / pid
        calibrations = HardwareProfile.list_calibrations(profile_dir)

        if len(calibrations) > 1:
            found = True
            power_modes = set(c['power_mode'] for c in calibrations)
            frameworks = set(c['framework'] for c in calibrations)
            print(f"\n  {pid}")
            print(f"    {profile.vendor} {profile.model}")
            print(f"    Calibrations: {len(calibrations)}")
            print(f"    Power Modes: {', '.join(sorted(power_modes))}")
            print(f"    Frameworks: {', '.join(sorted(frameworks))}")

    if not found:
        print("\n  No hardware found with multiple calibrations.")
        print("  Run calibrations at different power modes to compare.")

    print()


def compare_power_profiles(hardware_id: str, framework: Optional[str] = None,
                          detailed: bool = False, csv_output: Optional[Path] = None):
    """Compare power profiles for a specific hardware."""
    registry = get_registry()
    profile = registry.get(hardware_id)

    if not profile:
        print(f"Error: Hardware '{hardware_id}' not found in registry.")
        print(f"Use --list to see available hardware.")
        return 1

    # Get calibrations directory
    profile_dir = registry.path / profile.device_type / hardware_id
    calibrations = HardwareProfile.list_calibrations(profile_dir)

    if not calibrations:
        print(f"Error: No calibrations found for '{hardware_id}'.")
        print(f"Run calibration first: ./cli/calibrate.py --id {hardware_id}")
        return 1

    # Filter by framework if specified
    if framework:
        calibrations = [c for c in calibrations if c['framework'].lower() == framework.lower()]
        if not calibrations:
            print(f"Error: No calibrations found for framework '{framework}'.")
            return 1

    # Load summaries
    summaries = []
    for cal_info in calibrations:
        try:
            cal_path = Path(cal_info['path'])
            summary = load_calibration_summary(cal_path, cal_info, profile)
            summaries.append(summary)
        except Exception as e:
            print(f"Warning: Failed to load {cal_info['path']}: {e}")

    if not summaries:
        print("Error: Failed to load any calibration data.")
        return 1

    # Print comparison table
    print_power_profile_table(profile, summaries)

    # Print detailed operations if requested
    if detailed:
        print_detailed_operations(summaries)

    # Export CSV if requested
    if csv_output:
        export_csv(profile, summaries, csv_output)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare power profiles for hardware with multiple calibrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare all power profiles for Jetson GPU
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu

    # List hardware with multiple calibrations
    ./cli/compare_power_profiles.py --list

    # Show detailed per-operation breakdown
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --detailed

    # Export to CSV
    ./cli/compare_power_profiles.py --id jetson_orin_nano_gpu --csv power_comparison.csv
        """
    )

    parser.add_argument('--id', type=str, help='Hardware ID to compare')
    parser.add_argument('--list', action='store_true', help='List hardware with multiple calibrations')
    parser.add_argument('--detailed', action='store_true', help='Show per-operation breakdown')
    parser.add_argument('--framework', type=str, choices=['numpy', 'pytorch'],
                       help='Filter by framework')
    parser.add_argument('--csv', type=Path, help='Export results to CSV file')

    args = parser.parse_args()

    if args.list:
        registry = get_registry()
        list_hardware_with_calibrations(registry)
        return 0

    if not args.id:
        parser.print_help()
        print("\nError: --id required (or use --list to see available hardware)")
        return 1

    return compare_power_profiles(
        hardware_id=args.id,
        framework=args.framework,
        detailed=args.detailed,
        csv_output=args.csv
    )


if __name__ == "__main__":
    sys.exit(main())
