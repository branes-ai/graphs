#!/usr/bin/env python
"""
Show Calibration Efficiency

Display calibration measurements as percentage of theoretical peak performance.

Usage:
    ./cli/show_calibration_efficiency.py --id jetson_orin_nano_gpu
    ./cli/show_calibration_efficiency.py --id jetson_orin_nano_gpu --power-mode 25W
    ./cli/show_calibration_efficiency.py --all
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.registry import get_registry


def format_gflops(val: float, is_int: bool = False) -> str:
    """Format GFLOPS/GOPS value with appropriate precision."""
    suffix = "GOPS" if is_int else "GFLOPS"
    tera_suffix = "TOPS" if is_int else "TFLOPS"

    if val >= 1000:
        return f"{val/1000:.2f} {tera_suffix}"
    elif val >= 100:
        return f"{val:.0f} {suffix}"
    elif val >= 10:
        return f"{val:.1f} {suffix}"
    else:
        return f"{val:.2f} {suffix}"


def format_efficiency(measured: float, theoretical: float) -> str:
    """Format efficiency as percentage with color indicator."""
    if theoretical <= 0:
        return "N/A"

    pct = (measured / theoretical) * 100

    if pct > 110:
        return f"{pct:6.1f}% ⚠"  # Above theoretical - need to check specs
    elif pct >= 80:
        return f"{pct:6.1f}% ✓"  # Excellent
    elif pct >= 50:
        return f"{pct:6.1f}%"    # Good
    elif pct >= 20:
        return f"{pct:6.1f}% ↓"  # Below optimal
    else:
        return f"{pct:6.1f}% ↓↓" # Very low


def show_calibration_efficiency(profile, calibration, power_mode: str = None):
    """Display efficiency report for a calibration."""
    print(f"\n{'='*80}")
    print(f"Hardware: {profile.model}")
    print(f"Profile:  {profile.id}")
    if power_mode:
        print(f"Power Mode: {power_mode}")
    if calibration.metadata.gpu_clock:
        gc = calibration.metadata.gpu_clock
        print(f"GPU Clock: {gc.sm_clock_mhz} MHz")
        if gc.power_mode_name:
            print(f"Power Mode: {gc.power_mode_name}")
    print(f"{'='*80}\n")

    # Get theoretical peaks from profile
    theoretical = profile.theoretical_peaks

    # Show precision matrix results
    if calibration.precision_matrix:
        pm = calibration.precision_matrix

        print("BLAS Performance (GEMM - Best Achieved):")
        print("-" * 80)
        print(f"{'Precision':<10} {'Measured':>15} {'Theoretical':>15} {'Efficiency':>12} {'Status'}")
        print("-" * 80)

        # Get per-precision peaks from calibration
        measured_peaks = pm.peak_gflops_by_precision

        for prec in ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'int8', 'int16', 'int32']:
            theo = theoretical.get(prec, 0)
            meas = measured_peaks.get(prec, 0)
            is_int = prec.startswith('int')

            if theo > 0 or meas > 0:
                theo_str = format_gflops(theo, is_int) if theo > 0 else "N/A"
                meas_str = format_gflops(meas, is_int) if meas > 0 else "N/A"
                eff_str = format_efficiency(meas, theo) if theo > 0 and meas > 0 else "N/A"

                # Determine status
                if meas == 0:
                    status = "Not measured"
                elif theo == 0:
                    status = "No spec"
                elif meas / theo > 1.1:
                    status = "Check spec!"
                elif meas / theo >= 0.8:
                    status = "Excellent"
                elif meas / theo >= 0.5:
                    status = "Good"
                elif meas / theo >= 0.2:
                    status = "Suboptimal"
                else:
                    status = "Very low"

                print(f"{prec:<10} {meas_str:>15} {theo_str:>15} {eff_str:>12} {status}")

        print()

    # Show memory bandwidth
    print("Memory Bandwidth (STREAM):")
    print("-" * 80)
    theo_bw = profile.peak_bandwidth_gbps
    meas_bw = calibration.measured_bandwidth_gbps
    bw_eff = calibration.bandwidth_efficiency

    print(f"{'Measured:':<20} {meas_bw:.1f} GB/s")
    print(f"{'Theoretical:':<20} {theo_bw:.1f} GB/s")
    print(f"{'Efficiency:':<20} {bw_eff*100:.1f}%")
    print()

    # Show summary
    print("Summary:")
    print("-" * 80)
    print(f"{'Best Compute Efficiency:':<30} {calibration.best_efficiency*100:.1f}%")
    print(f"{'Average Compute Efficiency:':<30} {calibration.avg_efficiency*100:.1f}%")
    print(f"{'Worst Compute Efficiency:':<30} {calibration.worst_efficiency*100:.1f}%")
    print(f"{'Memory Bandwidth Efficiency:':<30} {bw_eff*100:.1f}%")
    print()


def show_summary_table(registry):
    """Display a compact summary table of all registry records."""
    print()
    print("Hardware Registry Summary")
    print("=" * 120)

    # Header
    header = (
        f"{'ID':<35} {'Type':<5} {'TDP':>6} "
        f"{'Peak Compute':>14} {'Peak BW':>10} {'Best Prec':<8} {'Calibrated'}"
    )
    print(header)
    print("-" * 120)

    for hw_id in sorted(registry.list_all()):
        profile = registry.get(hw_id)
        if not profile:
            continue

        # Device type
        dev_type = profile.device_type[:5].upper() if profile.device_type else "?"

        # TDP
        tdp = f"{profile.tdp_watts:.0f}W" if profile.tdp_watts else "N/A"

        # Peak compute - find best precision
        peaks = profile.theoretical_peaks or {}
        best_prec = None
        best_value = 0
        for prec, val in peaks.items():
            if val > best_value:
                best_value = val
                best_prec = prec

        if best_value >= 1000:
            compute_str = f"{best_value/1000:.1f} TFLOPS"
        elif best_value > 0:
            compute_str = f"{best_value:.0f} GFLOPS"
        else:
            compute_str = "N/A"

        best_prec_str = best_prec.upper() if best_prec else "N/A"

        # Peak bandwidth
        bw = profile.peak_bandwidth_gbps
        if bw and bw > 0:
            bw_str = f"{bw:.0f} GB/s"
        else:
            bw_str = "N/A"

        # Check if calibrated
        calibrations = registry.list_calibrations(hw_id)
        if calibrations:
            cal_str = f"Yes ({len(calibrations)})"
        else:
            cal_str = "No"

        print(f"{hw_id:<35} {dev_type:<5} {tdp:>6} {compute_str:>14} {bw_str:>10} {best_prec_str:<8} {cal_str}")

    print("-" * 120)
    print(f"Total: {len(registry.list_all())} hardware profiles")
    print()


def format_compute(value: float) -> str:
    """Format compute value (GFLOPS/TFLOPS)."""
    if value <= 0:
        return "N/A"
    elif value >= 1000:
        return f"{value/1000:.2f}T"
    elif value >= 100:
        return f"{value:.0f}G"
    else:
        return f"{value:.1f}G"


def format_bandwidth(value: float) -> str:
    """Format bandwidth value (GB/s)."""
    if value <= 0:
        return "N/A"
    elif value >= 1000:
        return f"{value/1000:.2f}T"
    else:
        return f"{value:.0f}G"


def format_efficiency_pct(value: float) -> str:
    """Format efficiency as percentage (for table display)."""
    if value is None or value <= 0:
        return "N/A"
    return f"{value*100:.1f}%"


def format_compute_short(value: float) -> str:
    """Format compute value in short form (for compact tables)."""
    if value <= 0:
        return "-"
    elif value >= 1000:
        return f"{value/1000:.1f}T"
    elif value >= 100:
        return f"{value:.0f}G"
    else:
        return f"{value:.0f}G"


def format_bandwidth_short(value: float) -> str:
    """Format bandwidth value in short form (for compact tables)."""
    if value is None or value <= 0:
        return "-"
    elif value >= 1000:
        return f"{value/1000:.0f}T"
    else:
        return f"{value:.0f}G"


def format_eff_short(value: float) -> str:
    """Format efficiency as short percentage (for compact tables)."""
    if value is None or value <= 0:
        return "-"
    return f"{value*100:.0f}%"


def extract_stream_bandwidth_by_size(calibration) -> dict:
    """Extract STREAM bandwidth measurements grouped by buffer size.

    Returns dict with:
        - 'cache_bw': bandwidth from smallest buffer (cache-bound)
        - 'dram_bw': bandwidth from largest buffer (DRAM-bound)
        - 'min_size_mb': smallest buffer size tested
        - 'max_size_mb': largest buffer size tested
    """
    if not calibration.operation_profiles:
        return {'cache_bw': 0, 'dram_bw': 0, 'min_size_mb': 0, 'max_size_mb': 0}

    # Collect all STREAM results with their sizes
    stream_results = []
    for key, profile in calibration.operation_profiles.items():
        op_type = profile.operation_type
        if op_type in ('stream_copy', 'stream_scale', 'stream_add', 'stream_triad'):
            size_mb = profile.extra_params.get('size_mb', 0)
            bw = profile.achieved_bandwidth_gbps or 0
            if size_mb > 0 and bw > 0:
                stream_results.append({
                    'size_mb': size_mb,
                    'bandwidth': bw,
                    'kernel': profile.extra_params.get('kernel', op_type),
                })

    if not stream_results:
        return {'cache_bw': 0, 'dram_bw': 0, 'min_size_mb': 0, 'max_size_mb': 0}

    # Group by size and find min/max sizes
    by_size = {}
    for r in stream_results:
        size = r['size_mb']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r['bandwidth'])

    sizes = sorted(by_size.keys())
    min_size = sizes[0]
    max_size = sizes[-1]

    # Use max bandwidth at each size (across all kernels)
    cache_bw = max(by_size[min_size])
    dram_bw = max(by_size[max_size])

    return {
        'cache_bw': cache_bw,
        'dram_bw': dram_bw,
        'min_size_mb': min_size,
        'max_size_mb': max_size,
    }


def show_efficiency_table(registry, precision_filter: str = None):
    """Display a compact efficiency table for all calibrated hardware.

    Args:
        registry: Hardware registry
        precision_filter: Optional precision to report on (e.g., 'fp32', 'fp16')
    """
    print()
    if precision_filter:
        print(f"Calibration Efficiency Summary (Precision: {precision_filter.upper()})")
    else:
        print("Calibration Efficiency Summary (Best Precision)")

    # Column widths - ID must be full length (40 chars) to be usable with --id
    # ID=40, Mode=11, Freq=11, Frmwk=7, Prec=4 = 73 + spaces = 77
    # Compute: Peak=6, Meas=6, Effic=6 = 18 + spaces = 20
    # On-Chip: Peak=4, Meas=5, Size=4, Effic=6 = 19 + spaces = 22
    # Off-Chip: Peak=4, Meas=5, Size=4, Effic=6 = 19 + spaces = 22
    # Separators: 3 x " | " = 9
    # Total line width: 77 + 20 + 22 + 22 + 9 = 150

    total_width = 150
    id_section = 77      # ID through Prec
    compute_section = 20
    onchip_section = 22
    offchip_section = 22

    print("=" * total_width)

    # Line 1: Group labels
    header1 = (
        f"{'ID':<40} {'Mode':<11} {'Freq MHz':<11} {'Frmwk':<7} {'Prec':<4}| "
        f"{'Compute':^{compute_section}}| "
        f"{'On-Chip BW (L3)':^{onchip_section}}| "
        f"{'Off-Chip BW (DRAM)':^{offchip_section}}"
    )
    # Line 2: Column headers
    header2 = (
        f"{'':<40} {'':<11} {'(cal/max)':<11} {'':<7} {'':<4}| "
        f"{'Peak':>6} {'Meas':>6} {'Eff':>6}| "
        f"{'Pk':>4} {'Meas':>5} {'Sz':>4} {'Eff':>6}| "
        f"{'Pk':>4} {'Meas':>5} {'Sz':>4} {'Eff':>6}"
    )
    print(header1)
    print(header2)
    print("-" * id_section + "+" + "-" * (compute_section + 1) + "+" + "-" * (onchip_section + 1) + "+" + "-" * (offchip_section + 1))

    rows = []

    for hw_id in sorted(registry.list_all()):
        profile = registry.get(hw_id)
        if not profile:
            continue

        calibrations = registry.list_calibrations(hw_id)
        if not calibrations:
            continue

        # Get theoretical peaks from profile
        theoretical_peaks = profile.theoretical_peaks or {}
        theoretical_bw = profile.peak_bandwidth_gbps or 0

        for cal_info in calibrations:
            cal_filter = {
                'power_mode': cal_info['power_mode'],
                'freq_mhz': cal_info['freq_mhz'],
                'framework': cal_info['framework'],
            }
            full_profile = registry.get(hw_id, calibration_filter=cal_filter)

            if not full_profile or not full_profile.calibration:
                continue

            cal = full_profile.calibration
            pm = cal.precision_matrix

            # Power mode
            power_mode = cal_info['power_mode'][:11] if cal_info['power_mode'] else "default"

            # Frequency - get from calibration metadata
            freq_str = "N/A"
            if cal.metadata:
                if cal.metadata.gpu_clock:
                    gc = cal.metadata.gpu_clock
                    cal_freq = gc.sm_clock_mhz or 0
                    max_freq = gc.max_sm_clock_mhz or cal_freq
                    if cal_freq > 0:
                        freq_str = f"{cal_freq}/{max_freq}"
                elif cal.metadata.cpu_clock:
                    cc = cal.metadata.cpu_clock
                    cal_freq = int(cc.current_freq_mhz) if cc.current_freq_mhz else 0
                    max_freq = int(cc.max_freq_mhz) if cc.max_freq_mhz else cal_freq
                    if cal_freq > 0:
                        freq_str = f"{cal_freq}/{max_freq}"

            # Framework
            framework = cal_info['framework'][:7] if cal_info['framework'] else "?"

            # Determine which precision to report
            if precision_filter:
                # Use specified precision
                target_prec = precision_filter.lower()
                measured = pm.peak_gflops_by_precision.get(target_prec, 0) if pm and pm.peak_gflops_by_precision else 0
                theoretical = theoretical_peaks.get(target_prec, 0)

                # Skip if no data for this precision
                if measured <= 0 and theoretical <= 0:
                    continue
            else:
                # Find best measured precision
                target_prec = None
                measured = 0
                if pm and pm.peak_gflops_by_precision:
                    for prec, val in pm.peak_gflops_by_precision.items():
                        if val > measured:
                            measured = val
                            target_prec = prec

                theoretical = theoretical_peaks.get(target_prec, 0) if target_prec else 0

            # Compute efficiency
            comp_eff = measured / theoretical if theoretical > 0 and measured > 0 else 0

            # Extract on-chip (cache) and off-chip (DRAM) bandwidth
            stream_data = extract_stream_bandwidth_by_size(cal)
            cache_bw = stream_data['cache_bw']
            dram_bw = stream_data['dram_bw']
            cache_size_mb = stream_data['min_size_mb']
            dram_size_mb = stream_data['max_size_mb']

            # Cache (L3) peak bandwidth - not in registry, so use None
            # TODO: Add l3_cache_bandwidth_gbps to hardware registry
            cache_peak = None

            # Cache efficiency - can't calculate without theoretical peak
            cache_eff = None

            # DRAM efficiency - compare against theoretical DRAM bandwidth
            dram_eff = dram_bw / theoretical_bw if theoretical_bw > 0 and dram_bw > 0 else 0

            rows.append({
                'id': hw_id,  # Full ID - no truncation
                'mode': power_mode[:11],
                'freq': freq_str[:11],
                'framework': framework[:7],
                'precision': target_prec.upper() if target_prec else "N/A",
                'comp_peak': theoretical,
                'comp_meas': measured,
                'comp_eff': comp_eff,
                'cache_peak': cache_peak,
                'cache_bw': cache_bw,
                'cache_size_mb': cache_size_mb,
                'cache_eff': cache_eff,
                'dram_peak': theoretical_bw,
                'dram_bw': dram_bw,
                'dram_size_mb': dram_size_mb,
                'dram_eff': dram_eff,
            })

    # Sort by ID then power mode
    rows.sort(key=lambda x: (x['id'], x['mode']))

    for row in rows:
        # Format buffer sizes (compact)
        cache_size_str = f"{row['cache_size_mb']}M" if row['cache_size_mb'] > 0 else "-"
        dram_size_str = f"{row['dram_size_mb']}M" if row['dram_size_mb'] > 0 else "-"

        # Format cache peak (N/A if not available)
        cache_peak_str = format_bandwidth_short(row['cache_peak']) if row['cache_peak'] else "-"
        cache_eff_str = format_eff_short(row['cache_eff']) if row['cache_eff'] else "-"

        line = (
            f"{row['id']:<40} {row['mode']:<11} {row['freq']:<11} {row['framework']:<7} {row['precision']:<4}| "
            f"{format_compute_short(row['comp_peak']):>6} {format_compute_short(row['comp_meas']):>6} "
            f"{format_eff_short(row['comp_eff']):>6}| "
            f"{cache_peak_str:>4} {format_bandwidth_short(row['cache_bw']):>5} {cache_size_str:>4} {cache_eff_str:>6}| "
            f"{format_bandwidth_short(row['dram_peak']):>4} {format_bandwidth_short(row['dram_bw']):>5} "
            f"{dram_size_str:>4} {format_eff_short(row['dram_eff']):>6}"
        )
        print(line)

    print("-" * 77 + "+" + "-" * 21 + "+" + "-" * 23 + "+" + "-" * 23)
    print(f"Total: {len(rows)} calibration records")
    print()
    print("Legend: G = GFLOPS or GB/s, T = TFLOPS or TB/s")
    print("        On-Chip BW = L3 cache (smallest buffer), Off-Chip BW = DRAM (largest buffer)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show calibration efficiency (percentage of theoretical peak)"
    )
    parser.add_argument(
        "--id",
        help="Hardware ID (e.g., jetson_orin_nano_gpu)"
    )
    parser.add_argument(
        "--power-mode", "-p",
        help="Filter by power mode (e.g., 25W, MAXN_SUPER)"
    )
    parser.add_argument(
        "--framework", "-f",
        help="Filter by framework (numpy, pytorch)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all calibrations"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available hardware profiles"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show compact summary table of all registry records"
    )
    parser.add_argument(
        "--table", "-t",
        action="store_true",
        help="Show compact efficiency table for calibrated hardware"
    )
    parser.add_argument(
        "--precision",
        help="Precision to report on (e.g., fp32, fp16, bf16, int8). Default: best precision"
    )

    args = parser.parse_args()

    # Load registry
    registry = get_registry()
    count = registry.load_all()

    if args.summary:
        show_summary_table(registry)
        return 0

    if args.table:
        show_efficiency_table(registry, precision_filter=args.precision)
        return 0

    if args.list:
        print(f"Available hardware profiles ({count} loaded):\n")
        for hw_id in sorted(registry.list_all()):
            profile = registry.get(hw_id)
            cals = registry.list_calibrations(hw_id)
            cal_str = f"({len(cals)} calibrations)" if cals else "(no calibrations)"
            print(f"  {hw_id:<40} {cal_str}")
        return 0

    if not args.id and not args.all:
        parser.print_help()
        print("\nUse --list to see available hardware profiles")
        print("Use --id <id> to show efficiency for a specific profile")
        print("Use --all to show all calibrations")
        return 1

    # Get hardware IDs to process
    if args.all:
        hardware_ids = registry.list_all()
    else:
        hardware_ids = [args.id]

    found_any = False

    for hw_id in hardware_ids:
        profile = registry.get(hw_id)
        if not profile:
            if not args.all:
                print(f"Error: Hardware ID '{hw_id}' not found in registry")
                return 1
            continue

        # Get calibrations
        calibrations = registry.list_calibrations(hw_id)
        if not calibrations:
            if not args.all:
                print(f"No calibrations found for {hw_id}")
            continue

        for cal_info in calibrations:
            # Apply filters
            if args.power_mode and cal_info['power_mode'].upper() != args.power_mode.upper():
                continue
            if args.framework and cal_info['framework'].lower() != args.framework.lower():
                continue

            # Load full calibration
            cal_filter = {
                'power_mode': cal_info['power_mode'],
                'freq_mhz': cal_info['freq_mhz'],
                'framework': cal_info['framework'],
            }
            full_profile = registry.get(hw_id, calibration_filter=cal_filter)

            if full_profile and full_profile.calibration:
                found_any = True
                show_calibration_efficiency(
                    full_profile,
                    full_profile.calibration,
                    power_mode=cal_info['power_mode']
                )

    if not found_any:
        print("No matching calibrations found.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
