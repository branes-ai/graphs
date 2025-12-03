#!/usr/bin/env python
"""
Show TOPS/W (Tera Operations Per Second per Watt) from Calibration Data

This script calculates the energy efficiency metric TOPS/W for different
hardware and precisions using empirical calibration measurements.

TOPS/W = Measured_GOPS / (Power_W * 1000)
       = Measured_TFLOPS / (Power_W * 2)  [since 1 MAC = 2 FLOPS]

Usage:
    ./cli/show_tops_per_watt.py                    # Show all calibrated hardware
    ./cli/show_tops_per_watt.py --id jetson_orin_nano_gpu
    ./cli/show_tops_per_watt.py --id jetson_orin_nano_gpu --power-mode 25W
    ./cli/show_tops_per_watt.py --compare          # Side-by-side comparison
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.registry import get_registry
from graphs.hardware.technology_profile import (
    get_process_base_energy_pj,
    CIRCUIT_TYPE_MULTIPLIER,
)


# =============================================================================
# Hardware Technology Specifications
# =============================================================================
# Maps hardware IDs to their process node and circuit type for ALU energy calc

HARDWARE_TECH_SPECS = {
    # Intel CPUs - Intel 7 process (marketing 10nm, actual ~10nm)
    'intel_12th_gen_intelr_coretm_i7_12700k': {
        'process_nm': 10,  # Intel 7
        'circuit_type': 'x86_performance',
        'compute_unit': 'avx512',  # AVX-512 SIMD
    },

    # AMD CPUs
    'amd_ryzen_7_2700x_eight_core_processor': {
        'process_nm': 12,  # GlobalFoundries 12LP
        'circuit_type': 'x86_performance',
        'compute_unit': 'avx2',
    },
    'ryzen_7_8845hs_w_radeon_780m_graphics': {
        'process_nm': 4,  # TSMC N4
        'circuit_type': 'x86_performance',
        'compute_unit': 'avx512',
    },
    'ryzen_9_8945hs_w_radeon_780m_graphics': {
        'process_nm': 4,  # TSMC N4
        'circuit_type': 'x86_performance',
        'compute_unit': 'avx512',
    },

    # Jetson Orin Nano - Samsung 8nm
    'jetson_orin_nano_cpu': {
        'process_nm': 8,  # Samsung 8LPP
        'circuit_type': 'arm_efficiency',
        'compute_unit': 'neon',  # ARM NEON
    },
    'jetson_orin_nano_gpu': {
        'process_nm': 8,  # Samsung 8LPP
        'circuit_type': 'tensor_core',  # Ampere Tensor Cores
        'compute_unit': 'tensor_core',
    },
}


# Precision energy scaling factors (relative to FP32 = 1.0)
# Based on datapath width and circuit complexity
PRECISION_ENERGY_FACTOR = {
    'fp64': 2.0,    # Double width datapath, more complex
    'fp32': 1.0,    # Baseline
    'tf32': 0.6,    # Reduced mantissa (10 bits vs 23)
    'fp16': 0.5,    # Half width datapath
    'bf16': 0.5,    # Half width (same as FP16 for energy)
    'int8': 0.25,   # Quarter width, simpler ALU
    'int16': 0.5,   # Half width
    'int32': 1.0,   # Same as FP32 for integer
    'int64': 2.0,   # Double width
}


def get_clock_mhz_from_calibration(calibration) -> Optional[float]:
    """Extract clock frequency from calibration metadata."""
    if calibration.metadata.gpu_clock:
        return calibration.metadata.gpu_clock.sm_clock_mhz
    elif calibration.metadata.cpu_clock:
        return calibration.metadata.cpu_clock.current_freq_mhz
    return None


def calculate_theoretical_at_clock(profile, clock_mhz: float) -> Dict[str, float]:
    """
    Calculate theoretical peaks at a specific clock frequency.

    If profile has ops_per_clock, use: theoretical = ops_per_clock * clock_mhz / 1000
    Otherwise fall back to static theoretical_peaks.

    Returns:
        Dict of precision -> GOPS/GFLOPS at the given clock
    """
    # Try to get ops_per_clock from profile (may need to load from raw spec)
    ops_per_clock = getattr(profile, 'ops_per_clock', None)

    if ops_per_clock and clock_mhz:
        # Calculate theoretical at this clock
        result = {}
        for prec, ops in ops_per_clock.items():
            if ops > 0:
                # ops_per_clock * MHz = ops/sec in millions = GOPS (since MHz = 10^6)
                result[prec] = ops * clock_mhz / 1000.0  # Convert to GOPS
            else:
                result[prec] = 0.0
        return result

    # Fall back to static theoretical_peaks
    return profile.theoretical_peaks


def get_alu_tops_per_watt(hw_id: str, precision: str) -> Optional[Tuple[float, str]]:
    """
    Calculate theoretical ALU-only TOPS/W from first principles.

    Returns:
        Tuple of (alu_tops_per_watt, explanation) or None if unknown hardware
    """
    spec = HARDWARE_TECH_SPECS.get(hw_id)
    if not spec:
        return None

    # Get base FP32 energy for this process node
    base_energy_pj = get_process_base_energy_pj(spec['process_nm'])

    # Apply circuit type multiplier
    circuit_mult = CIRCUIT_TYPE_MULTIPLIER.get(spec['circuit_type'], 1.0)

    # Apply precision scaling
    prec_factor = PRECISION_ENERGY_FACTOR.get(precision, 1.0)

    # Total ALU energy per MAC in picojoules
    alu_energy_pj = base_energy_pj * circuit_mult * prec_factor

    # Convert to TOPS/W:
    # 1 pJ/op = 1e-12 J/op
    # At 1 TOPS = 1e12 ops/s
    # Power = 1e12 ops/s * 1e-12 J/op = 1 W
    # So ALU_TOPS_per_W = 1 / alu_energy_pj
    alu_tops_per_w = 1.0 / alu_energy_pj

    explanation = (f"{spec['process_nm']}nm * {circuit_mult:.2f}x ({spec['circuit_type']}) "
                   f"* {prec_factor:.2f}x ({precision}) = {alu_energy_pj:.2f} pJ/MAC")

    return alu_tops_per_w, explanation


# Known TDP values for hardware (since not all profiles have TDP)
KNOWN_TDP_WATTS = {
    # Intel CPUs
    'intel_12th_gen_intelr_coretm_i7_12700k': 125,  # PL1=125W, PL2=190W

    # AMD CPUs
    'amd_ryzen_7_2700x_eight_core_processor': 105,
    'ryzen_7_8845hs_w_radeon_780m_graphics': 45,    # Mobile
    'ryzen_9_8945hs_w_radeon_780m_graphics': 45,    # Mobile

    # Jetson - use power mode
    'jetson_orin_nano_cpu': {
        '7W': 7,
        '15W': 15,
        '25W': 25,
        'MAXNSUPER': 25,
        'schedutil': 15,  # Default assumption
    },
    'jetson_orin_nano_gpu': {
        '7W': 7,
        '15W': 15,
        '25W': 25,
        'MAXNSUPER': 25,
    },
}


def get_tdp_watts(hw_id: str, power_mode: str = None) -> Optional[float]:
    """Get TDP in watts for hardware, considering power mode."""
    tdp = KNOWN_TDP_WATTS.get(hw_id)

    if tdp is None:
        return None

    if isinstance(tdp, dict):
        if power_mode and power_mode in tdp:
            return tdp[power_mode]
        # Return first available
        return list(tdp.values())[0] if tdp else None

    return tdp


def format_tops(gflops: float, is_int: bool = False) -> str:
    """Format GFLOPS as TOPS (1 MAC = 2 FLOPS for float, 1 for int)."""
    if is_int:
        tops = gflops / 1000  # GOPS -> TOPS
    else:
        tops = gflops / 1000 / 2  # GFLOPS -> TOPS (2 FLOPS per MAC)

    if tops >= 1:
        return f"{tops:.2f} TOPS"
    elif tops >= 0.001:
        return f"{tops*1000:.1f} GOPS"
    else:
        return f"{tops*1e6:.1f} MOPS"


def format_tops_per_watt(gflops: float, tdp_w: float, is_int: bool = False) -> str:
    """Calculate and format TOPS/W."""
    if is_int:
        tops = gflops / 1000  # GOPS -> TOPS
    else:
        tops = gflops / 1000 / 2  # GFLOPS -> TOPS

    tops_per_watt = tops / tdp_w

    if tops_per_watt >= 1:
        return f"{tops_per_watt:.2f} TOPS/W"
    elif tops_per_watt >= 0.001:
        return f"{tops_per_watt*1000:.1f} GOPS/W"
    else:
        return f"{tops_per_watt*1e6:.1f} MOPS/W"


def show_tops_per_watt(profile, calibration, power_mode: str = None, framework: str = None):
    """Display TOPS/W report for a calibration."""
    hw_id = profile.id
    tdp_w = get_tdp_watts(hw_id, power_mode)

    print()
    print("=" * 90)
    print(f"Hardware: {profile.model}")
    print(f"Profile:  {hw_id}")

    if framework:
        print(f"Framework: {framework}")

    if power_mode:
        print(f"Power Mode: {power_mode}")

    if calibration.metadata.gpu_clock:
        gc = calibration.metadata.gpu_clock
        print(f"Clock: {gc.sm_clock_mhz} MHz")
    elif calibration.metadata.cpu_clock:
        cc = calibration.metadata.cpu_clock
        print(f"Clock: {cc.current_freq_mhz:.0f} MHz")

    if tdp_w:
        print(f"TDP: {tdp_w} W")
    else:
        print("TDP: Unknown (cannot calculate TOPS/W)")

    print("=" * 90)
    print()

    if not calibration.precision_matrix:
        print("No precision matrix data available.")
        return

    pm = calibration.precision_matrix
    measured_peaks = pm.peak_gflops_by_precision

    # Get clock frequency from calibration and calculate theoretical at that clock
    clock_mhz = get_clock_mhz_from_calibration(calibration)
    if profile.ops_per_clock and clock_mhz:
        theoretical = calculate_theoretical_at_clock(profile, clock_mhz)
    else:
        theoretical = profile.theoretical_peaks

    # Check if we have tech specs for ALU-only calculation
    has_tech_specs = hw_id in HARDWARE_TECH_SPECS

    # Header
    print(f"{'Precision':<10} {'Measured':>14} {'Theoretical':>14} {'Efficiency':>12}", end="")
    if tdp_w:
        print(f" {'Sys TOPS/W':>12}", end="")
        if has_tech_specs:
            print(f" {'ALU TOPS/W':>12} {'Overhead':>10}")
        else:
            print()
    else:
        print()
    header_width = 120 if (tdp_w and has_tech_specs) else (90 if tdp_w else 54)
    print("-" * header_width)

    # Order precisions by typical importance
    precision_order = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'int8', 'int16', 'int32', 'int64']

    for prec in precision_order:
        theo = theoretical.get(prec, 0)
        meas = measured_peaks.get(prec, 0)

        if theo == 0 and meas == 0:
            continue

        is_int = prec.startswith('int')

        # Format measured
        if meas > 0:
            meas_str = format_tops(meas, is_int)
        else:
            meas_str = "N/A"

        # Format theoretical
        if theo > 0:
            theo_str = format_tops(theo, is_int)
        else:
            theo_str = "N/A"

        # Efficiency
        if theo > 0 and meas > 0:
            eff = meas / theo * 100
            eff_str = f"{eff:.1f}%"
        else:
            eff_str = "N/A"

        # System TOPS/W (measured)
        if tdp_w and meas > 0:
            if is_int:
                sys_tops = meas / 1000  # GOPS -> TOPS
            else:
                sys_tops = meas / 1000 / 2  # GFLOPS -> TOPS (2 FLOPS per MAC)
            sys_tpw = sys_tops / tdp_w
            sys_tpw_str = f"{sys_tpw:.4f}"
        else:
            sys_tpw = 0
            sys_tpw_str = "N/A"

        # ALU-only TOPS/W and overhead ratio
        if has_tech_specs and tdp_w and meas > 0:
            alu_result = get_alu_tops_per_watt(hw_id, prec)
            if alu_result:
                alu_tpw, _ = alu_result
                alu_tpw_str = f"{alu_tpw:.2f}"
                # Overhead = ALU_TOPS/W / System_TOPS/W
                # This tells us: "ALU alone could do Nx more TOPS/W than system achieves"
                if sys_tpw > 0:
                    # Only show overhead if architecture supports this precision (has theoretical)
                    if theo > 0:
                        overhead = alu_tpw / sys_tpw
                        overhead_str = f"{overhead:.0f}x"
                    else:
                        # No theoretical = architecture doesn't natively support this precision
                        overhead_str = "-"
                else:
                    overhead_str = "-"
            else:
                alu_tpw_str = "-"
                overhead_str = "-"
        else:
            alu_tpw_str = "-"
            overhead_str = "-"

        print(f"{prec:<10} {meas_str:>14} {theo_str:>14} {eff_str:>12}", end="")
        if tdp_w:
            print(f" {sys_tpw_str:>12}", end="")
            if has_tech_specs:
                print(f" {alu_tpw_str:>12} {overhead_str:>10}")
            else:
                print()
        else:
            print()

    print()

    # Summary with ALU overhead analysis
    if tdp_w:
        print("Key Metrics:")
        print("-" * 70)

        # Best FP32 TOPS/W
        fp32_meas = measured_peaks.get('fp32', 0)
        if fp32_meas > 0:
            fp32_tops = fp32_meas / 1000 / 2
            fp32_tpw = fp32_tops / tdp_w
            print(f"  FP32: {fp32_tops:.3f} TOPS @ {tdp_w}W = {fp32_tpw:.4f} TOPS/W (system)")
            if has_tech_specs:
                alu_result = get_alu_tops_per_watt(hw_id, 'fp32')
                if alu_result:
                    alu_tpw, explanation = alu_result
                    overhead = alu_tpw / fp32_tpw if fp32_tpw > 0 else 0
                    print(f"         ALU-only: {alu_tpw:.2f} TOPS/W -> {overhead:.0f}x system overhead")
                    print(f"         ({explanation})")

        # Best FP16/BF16 TOPS/W
        fp16_meas = max(measured_peaks.get('fp16', 0), measured_peaks.get('bf16', 0))
        best_fp16_prec = 'bf16' if measured_peaks.get('bf16', 0) >= measured_peaks.get('fp16', 0) else 'fp16'
        if fp16_meas > 0:
            fp16_tops = fp16_meas / 1000 / 2
            fp16_tpw = fp16_tops / tdp_w
            print(f"  {best_fp16_prec.upper()}: {fp16_tops:.3f} TOPS @ {tdp_w}W = {fp16_tpw:.4f} TOPS/W (system)")
            if has_tech_specs:
                alu_result = get_alu_tops_per_watt(hw_id, best_fp16_prec)
                if alu_result:
                    alu_tpw, explanation = alu_result
                    overhead = alu_tpw / fp16_tpw if fp16_tpw > 0 else 0
                    print(f"         ALU-only: {alu_tpw:.2f} TOPS/W -> {overhead:.0f}x system overhead")

        # Best INT8 TOPS/W
        int8_meas = measured_peaks.get('int8', 0)
        if int8_meas > 0:
            int8_tops = int8_meas / 1000
            int8_tpw = int8_tops / tdp_w
            print(f"  INT8: {int8_tops:.3f} TOPS @ {tdp_w}W = {int8_tpw:.4f} TOPS/W (system)")
            if has_tech_specs:
                alu_result = get_alu_tops_per_watt(hw_id, 'int8')
                if alu_result:
                    alu_tpw, explanation = alu_result
                    overhead = alu_tpw / int8_tpw if int8_tpw > 0 else 0
                    print(f"         ALU-only: {alu_tpw:.2f} TOPS/W -> {overhead:.0f}x system overhead")

        print()


def show_comparison(registry):
    """Show side-by-side TOPS/W comparison for all hardware."""
    print()
    print("=" * 100)
    print("TOPS/W COMPARISON ACROSS HARDWARE")
    print("=" * 100)
    print()

    # Collect data
    data = []

    for hw_id in sorted(registry.list_all()):
        calibrations = registry.list_calibrations(hw_id)
        if not calibrations:
            continue

        # Sort by clock frequency for consistent ordering
        calibrations = sorted(calibrations, key=lambda c: c.get('freq_mhz', 0))

        for cal_info in calibrations:
            cal_filter = {
                'power_mode': cal_info['power_mode'],
                'freq_mhz': cal_info['freq_mhz'],
                'framework': cal_info['framework'],
            }
            full = registry.get(hw_id, calibration_filter=cal_filter)

            if not full or not full.calibration or not full.calibration.precision_matrix:
                continue

            cal = full.calibration
            pm = cal.precision_matrix
            tdp_w = get_tdp_watts(hw_id, cal_info['power_mode'])

            if not tdp_w:
                continue

            measured = pm.peak_gflops_by_precision

            # Get key metrics
            fp32_gflops = measured.get('fp32', 0)
            fp16_gflops = max(measured.get('fp16', 0), measured.get('bf16', 0))
            int8_gops = measured.get('int8', 0)

            fp32_tops = fp32_gflops / 1000 / 2 if fp32_gflops else 0
            fp16_tops = fp16_gflops / 1000 / 2 if fp16_gflops else 0
            int8_tops = int8_gops / 1000 if int8_gops else 0

            data.append({
                'name': f"{full.model} ({cal_info['power_mode']})",
                'hw_id': hw_id,
                'power_mode': cal_info['power_mode'],
                'framework': cal_info['framework'],
                'tdp_w': tdp_w,
                'fp32_tops': fp32_tops,
                'fp32_tpw': fp32_tops / tdp_w if fp32_tops else 0,
                'fp16_tops': fp16_tops,
                'fp16_tpw': fp16_tops / tdp_w if fp16_tops else 0,
                'int8_tops': int8_tops,
                'int8_tpw': int8_tops / tdp_w if int8_tops else 0,
            })

    if not data:
        print("No calibration data with known TDP found.")
        return

    # Print FP32 comparison
    print("FP32 Performance:")
    print("-" * 115)
    print(f"{'Hardware':<45} {'Framework':>10} {'TDP':>6} {'TOPS':>10} {'TOPS/W':>12} {'W/TOPS':>10}")
    print("-" * 115)

    for d in sorted(data, key=lambda x: -x['fp32_tpw']):
        if d['fp32_tops'] > 0:
            w_per_tops = d['tdp_w'] / d['fp32_tops']
            print(f"{d['name']:<45} {d['framework']:>10} {d['tdp_w']:>5}W {d['fp32_tops']:>9.3f} {d['fp32_tpw']:>11.4f} {w_per_tops:>9.1f}")

    print()

    # Print FP16/BF16 comparison
    print("FP16/BF16 Performance:")
    print("-" * 115)
    print(f"{'Hardware':<45} {'Framework':>10} {'TDP':>6} {'TOPS':>10} {'TOPS/W':>12} {'W/TOPS':>10}")
    print("-" * 115)

    for d in sorted(data, key=lambda x: -x['fp16_tpw']):
        if d['fp16_tops'] > 0:
            w_per_tops = d['tdp_w'] / d['fp16_tops']
            print(f"{d['name']:<45} {d['framework']:>10} {d['tdp_w']:>5}W {d['fp16_tops']:>9.3f} {d['fp16_tpw']:>11.4f} {w_per_tops:>9.1f}")

    print()

    # Print INT8 comparison (if any)
    has_int8 = any(d['int8_tops'] > 0 for d in data)
    if has_int8:
        print("INT8 Performance:")
        print("-" * 115)
        print(f"{'Hardware':<45} {'Framework':>10} {'TDP':>6} {'TOPS':>10} {'TOPS/W':>12} {'W/TOPS':>10}")
        print("-" * 115)

        for d in sorted(data, key=lambda x: -x['int8_tpw']):
            if d['int8_tops'] > 0:
                w_per_tops = d['tdp_w'] / d['int8_tops']
                print(f"{d['name']:<45} {d['framework']:>10} {d['tdp_w']:>5}W {d['int8_tops']:>9.3f} {d['int8_tpw']:>11.4f} {w_per_tops:>9.1f}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Show TOPS/W from calibration data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--id", help="Hardware ID")
    parser.add_argument("--power-mode", "-p", help="Filter by power mode")
    parser.add_argument("--framework", "-f", help="Filter by framework")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Show side-by-side comparison")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available hardware")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all calibrations")

    args = parser.parse_args()

    registry = get_registry()
    count = registry.load_all()

    if args.list:
        print(f"Available hardware profiles ({count} loaded):\n")
        for hw_id in sorted(registry.list_all()):
            profile = registry.get(hw_id)
            cals = registry.list_calibrations(hw_id)
            tdp = get_tdp_watts(hw_id)
            tdp_str = f"{tdp}W" if tdp else "TDP?"
            cal_str = f"({len(cals)} calibrations)" if cals else "(no calibrations)"
            print(f"  {hw_id:<50} {tdp_str:>6} {cal_str}")
        return 0

    if args.compare:
        show_comparison(registry)
        return 0

    if not args.id and not args.all:
        parser.print_help()
        print("\nUse --list to see available hardware")
        print("Use --compare for side-by-side comparison")
        return 1

    # Process specific hardware or all
    hardware_ids = [args.id] if args.id else registry.list_all()

    for hw_id in hardware_ids:
        profile = registry.get(hw_id)
        if not profile:
            if args.id:
                print(f"Error: Hardware ID '{hw_id}' not found")
                return 1
            continue

        calibrations = registry.list_calibrations(hw_id)
        if not calibrations:
            continue

        # Sort calibrations by clock frequency (ascending) for progressive display
        calibrations = sorted(calibrations, key=lambda c: c.get('freq_mhz', 0))

        for cal_info in calibrations:
            # Apply filters
            if args.power_mode and cal_info['power_mode'].upper() != args.power_mode.upper():
                continue
            if args.framework and cal_info['framework'].lower() != args.framework.lower():
                continue

            cal_filter = {
                'power_mode': cal_info['power_mode'],
                'freq_mhz': cal_info['freq_mhz'],
                'framework': cal_info['framework'],
            }
            full = registry.get(hw_id, calibration_filter=cal_filter)

            if full and full.calibration:
                show_tops_per_watt(full, full.calibration, cal_info['power_mode'], cal_info['framework'])

    return 0


if __name__ == "__main__":
    sys.exit(main())
