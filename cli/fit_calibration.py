#!/usr/bin/env python
"""
Calibration Fitting CLI Tool

Command-line interface for fitting hardware performance models from benchmark
measurements. This tool takes benchmark results and fits:
- Roofline parameters (bandwidth, compute ceilings)
- Energy coefficients (pJ/op, pJ/byte, static power)
- Utilization curves (vs problem size)

This is complementary to cli/calibrate.py which runs the benchmarks.
Use this tool after collecting benchmark results to fit the models.

Calibration Modes:
    roofline    - Fit roofline parameters (bandwidth, compute ceilings)
    energy      - Fit energy coefficients (pJ/op, pJ/byte, static power)
    utilization - Fit utilization curves (vs problem size)
    all         - Run all calibration modes

Usage:
    # Fit roofline from benchmark results
    ./cli/fit_calibration.py --mode roofline --input results.json

    # Fit energy model from power measurements
    ./cli/fit_calibration.py --mode energy --input results.json --peak-gflops 50000

    # Fit utilization curves
    ./cli/fit_calibration.py --mode utilization --input results.json --peak-gflops 50000

    # Run full calibration fitting
    ./cli/fit_calibration.py --mode all --input results.json --peak-gflops 50000

    # Output calibration profile
    ./cli/fit_calibration.py --mode all --input results.json --output profile.json

    # Generate quality report
    ./cli/fit_calibration.py --mode all --input results.json --report quality.md

    # Update existing profile with new data
    ./cli/fit_calibration.py --mode roofline --update existing.json --input new_results.json

    # Specify hardware specifications
    ./cli/fit_calibration.py --mode all --input results.json \\
        --peak-gflops 50000 --peak-bandwidth 2000 --device-name "NVIDIA H100"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.benchmarks.schema import BenchmarkResult, TimingStats
from graphs.calibration.roofline_fitter import (
    RooflineFitter,
    RooflineParameters,
    fit_roofline,
)
from graphs.calibration.energy_fitter import (
    EnergyFitter,
    EnergyCoefficients,
    fit_energy_model,
)
from graphs.calibration.utilization_fitter import (
    UtilizationFitter,
    UtilizationProfile,
    fit_utilization,
)


def load_benchmark_results(path: Path) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    results = []

    # Handle different JSON structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if 'results' in data:
            items = data['results']
        else:
            items = [data]
    else:
        items = []

    for item in items:
        try:
            result = BenchmarkResult.from_dict(item)
            results.append(result)
        except (KeyError, TypeError) as e:
            print(f"Warning: Skipping invalid result: {e}", file=sys.stderr)

    return results


def save_profile(profile: Dict[str, Any], path: Path) -> None:
    """Save calibration profile to file."""
    suffix = path.suffix.lower()

    if suffix == '.yaml' or suffix == '.yml':
        import yaml
        with open(path, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    else:
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2)


def load_profile(path: Path) -> Dict[str, Any]:
    """Load existing calibration profile."""
    suffix = path.suffix.lower()

    with open(path, 'r') as f:
        if suffix == '.yaml' or suffix == '.yml':
            import yaml
            return yaml.safe_load(f)
        else:
            return json.load(f)


def fit_roofline_mode(
    results: List[BenchmarkResult],
    peak_bandwidth_gbps: Optional[float] = None,
    peak_gflops: Optional[float] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run roofline calibration."""
    if not quiet:
        print("Fitting roofline parameters...")

    fitter = RooflineFitter(
        theoretical_bandwidth_gbps=peak_bandwidth_gbps,
        theoretical_compute_gflops=peak_gflops,
    )

    for result in results:
        fitter.add_result(result)

    if not fitter.can_fit():
        print("Warning: Insufficient data for roofline fitting", file=sys.stderr)
        return {'error': 'Insufficient data for roofline fitting'}

    params = fitter.fit()

    if not quiet:
        print(f"  Achieved bandwidth: {params.achieved_bandwidth_gbps:.1f} GB/s")
        print(f"  Achieved compute: {params.achieved_compute_gflops:.1f} GFLOPS")
        print(f"  Ridge point: {params.ridge_point:.2f} FLOP/byte")

        if params.bandwidth_efficiency > 0:
            print(f"  Bandwidth efficiency: {params.bandwidth_efficiency:.1%}")
        if params.compute_efficiency > 0:
            print(f"  Compute efficiency: {params.compute_efficiency:.1%}")

    return params.to_dict()


def fit_energy_mode(
    results: List[BenchmarkResult],
    idle_power_watts: Optional[float] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run energy coefficient calibration."""
    if not quiet:
        print("Fitting energy coefficients...")

    # Filter results with power data
    power_results = [r for r in results if r.avg_power_watts is not None]

    if len(power_results) < 3:
        print("Warning: Insufficient power measurements for energy fitting", file=sys.stderr)
        return {'error': 'Insufficient power measurements'}

    try:
        coefficients = fit_energy_model(power_results, idle_power_watts)

        if not quiet:
            print(f"  Compute: {coefficients.compute_pj_per_op:.3f} pJ/op")
            print(f"  Memory: {coefficients.memory_pj_per_byte:.3f} pJ/byte")
            print(f"  Static power: {coefficients.static_power_watts:.1f} W")

            if coefficients.fit_quality:
                print(f"  Fit quality: {coefficients.fit_quality.value}")

        return coefficients.to_dict()

    except ValueError as e:
        print(f"Warning: Energy fitting failed: {e}", file=sys.stderr)
        return {'error': str(e)}


def fit_utilization_mode(
    results: List[BenchmarkResult],
    peak_gflops: float,
    peak_bandwidth_gbps: float = 0.0,
    peak_by_precision: Optional[Dict[str, float]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run utilization curve calibration."""
    if not quiet:
        print("Fitting utilization curves...")

    if peak_gflops <= 0:
        print("Warning: Peak GFLOPS required for utilization fitting", file=sys.stderr)
        return {'error': 'Peak GFLOPS required'}

    fitter = UtilizationFitter(
        peak_compute_gflops=peak_gflops,
        peak_bandwidth_gbps=peak_bandwidth_gbps,
        peak_compute_by_precision=peak_by_precision,
    )

    for result in results:
        fitter.add_result(result)

    if not fitter.can_fit():
        print("Warning: Insufficient data for utilization fitting", file=sys.stderr)
        return {'error': 'Insufficient data for utilization fitting'}

    profile = fitter.fit()

    # Print summary
    if not quiet:
        for op, prec_curves in profile.curves.items():
            for prec, curve_result in prec_curves.items():
                metrics = curve_result.metrics
                print(f"  {op}/{prec}: {metrics.num_data_points} points, "
                      f"R^2={metrics.r_squared:.3f}, "
                      f"util=[{metrics.min_utilization:.2f}, {metrics.max_utilization:.2f}]")

    return profile.to_dict()


def generate_report(
    roofline: Optional[Dict] = None,
    energy: Optional[Dict] = None,
    utilization: Optional[Dict] = None,
    device_name: str = "",
) -> str:
    """Generate a quality report in markdown format."""
    lines = []
    lines.append("# Calibration Quality Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    if device_name:
        lines.append(f"Device: {device_name}")
    lines.append("")

    if roofline and 'error' not in roofline:
        lines.append("## Roofline Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Achieved Bandwidth | {roofline.get('achieved_bandwidth_gbps', 0):.1f} GB/s |")
        lines.append(f"| Achieved Compute | {roofline.get('achieved_compute_gflops', 0):.1f} GFLOPS |")
        lines.append(f"| Ridge Point | {roofline.get('ridge_point', 0):.2f} FLOP/byte |")
        lines.append(f"| Bandwidth Efficiency | {roofline.get('bandwidth_efficiency', 0):.1%} |")
        lines.append(f"| Compute Efficiency | {roofline.get('compute_efficiency', 0):.1%} |")
        lines.append("")

        if roofline.get('bandwidth_fit'):
            bw_fit = roofline['bandwidth_fit']
            lines.append("### Bandwidth Fit Quality")
            lines.append(f"- Data points: {bw_fit.get('num_data_points', 0)}")
            lines.append(f"- R-squared: {bw_fit.get('r_squared', 0):.4f}")
            lines.append(f"- Quality: {bw_fit.get('quality', 'unknown')}")
            lines.append("")

        if roofline.get('compute_fit'):
            c_fit = roofline['compute_fit']
            lines.append("### Compute Fit Quality")
            lines.append(f"- Data points: {c_fit.get('num_data_points', 0)}")
            lines.append(f"- R-squared: {c_fit.get('r_squared', 0):.4f}")
            lines.append(f"- Quality: {c_fit.get('quality', 'unknown')}")
            lines.append("")

    if energy and 'error' not in energy:
        lines.append("## Energy Coefficients")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Compute Energy | {energy.get('compute_pj_per_op', 0):.3f} pJ/op |")
        lines.append(f"| Memory Energy | {energy.get('memory_pj_per_byte', 0):.3f} pJ/byte |")
        lines.append(f"| Static Power | {energy.get('static_power_watts', 0):.1f} W |")
        lines.append("")

        if energy.get('fit_metrics'):
            metrics = energy['fit_metrics']
            lines.append("### Fit Quality")
            lines.append(f"- Data points: {metrics.get('num_data_points', 0)}")
            lines.append(f"- R-squared: {metrics.get('r_squared', 0):.4f}")
            lines.append(f"- RMSE: {metrics.get('rmse', 0):.4f}")
            lines.append("")

    if utilization and 'error' not in utilization:
        lines.append("## Utilization Curves")
        lines.append("")

        curves = utilization.get('curves', {})
        if curves:
            lines.append("| Operation | Precision | Points | R^2 | Min Util | Max Util |")
            lines.append("|-----------|-----------|--------|-----|----------|----------|")

            for op, prec_curves in curves.items():
                for prec, curve_data in prec_curves.items():
                    metrics = curve_data.get('metrics', {})
                    lines.append(
                        f"| {op} | {prec} | "
                        f"{metrics.get('num_data_points', 0)} | "
                        f"{metrics.get('r_squared', 0):.3f} | "
                        f"{metrics.get('min_utilization', 0):.2f} | "
                        f"{metrics.get('max_utilization', 0):.2f} |"
                    )
            lines.append("")

    if not any([roofline, energy, utilization]):
        lines.append("No calibration data available.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calibration Fitting CLI - Fit hardware models from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        '--mode', '-m',
        choices=['roofline', 'energy', 'utilization', 'all'],
        default='all',
        help="Calibration mode (default: all)",
    )

    # Input/Output
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help="Input benchmark results file (JSON)",
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help="Output calibration profile (JSON or YAML)",
    )
    parser.add_argument(
        '--report', '-r',
        type=Path,
        help="Generate quality report (Markdown)",
    )
    parser.add_argument(
        '--update', '-u',
        type=Path,
        help="Update existing profile with new data",
    )

    # Hardware specifications
    parser.add_argument(
        '--peak-gflops',
        type=float,
        default=0.0,
        help="Theoretical peak GFLOPS (FP32)",
    )
    parser.add_argument(
        '--peak-bandwidth',
        type=float,
        default=0.0,
        help="Theoretical peak bandwidth (GB/s)",
    )
    parser.add_argument(
        '--device-name',
        type=str,
        default="",
        help="Hardware device name",
    )
    parser.add_argument(
        '--idle-power',
        type=float,
        help="Idle power in watts (for energy calibration)",
    )

    # Precision-specific peaks
    parser.add_argument(
        '--peak-fp16',
        type=float,
        help="Peak GFLOPS for FP16",
    )
    parser.add_argument(
        '--peak-int8',
        type=float,
        help="Peak GFLOPS for INT8",
    )

    # Verbosity
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Load benchmark results
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    results = load_benchmark_results(args.input)
    if not results:
        print(f"Error: No valid benchmark results in {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loaded {len(results)} benchmark results")

    # Load existing profile if updating
    existing_profile = {}
    if args.update and args.update.exists():
        existing_profile = load_profile(args.update)
        if not args.quiet:
            print(f"Loaded existing profile from {args.update}")

    # Build precision map
    peak_by_precision = {}
    if args.peak_gflops > 0:
        peak_by_precision['fp32'] = args.peak_gflops
    if args.peak_fp16:
        peak_by_precision['fp16'] = args.peak_fp16
    if args.peak_int8:
        peak_by_precision['int8'] = args.peak_int8

    # Run calibration
    profile = existing_profile.copy()
    profile['metadata'] = {
        'device_name': args.device_name or existing_profile.get('metadata', {}).get('device_name', ''),
        'created_at': existing_profile.get('metadata', {}).get('created_at', datetime.now().isoformat()),
        'updated_at': datetime.now().isoformat(),
        'peak_gflops': args.peak_gflops,
        'peak_bandwidth_gbps': args.peak_bandwidth,
    }

    roofline_result = None
    energy_result = None
    utilization_result = None

    if args.mode in ['roofline', 'all']:
        roofline_result = fit_roofline_mode(
            results,
            peak_bandwidth_gbps=args.peak_bandwidth if args.peak_bandwidth > 0 else None,
            peak_gflops=args.peak_gflops if args.peak_gflops > 0 else None,
            quiet=args.quiet,
        )
        if 'error' not in roofline_result:
            profile['roofline'] = roofline_result

    if args.mode in ['energy', 'all']:
        energy_result = fit_energy_mode(results, args.idle_power, quiet=args.quiet)
        if 'error' not in energy_result:
            profile['energy'] = energy_result

    if args.mode in ['utilization', 'all']:
        if args.peak_gflops > 0:
            utilization_result = fit_utilization_mode(
                results,
                peak_gflops=args.peak_gflops,
                peak_bandwidth_gbps=args.peak_bandwidth,
                peak_by_precision=peak_by_precision if peak_by_precision else None,
                quiet=args.quiet,
            )
            if 'error' not in utilization_result:
                profile['utilization'] = utilization_result
        elif not args.quiet:
            print("Skipping utilization fitting (--peak-gflops required)")

    # Save profile
    if args.output:
        save_profile(profile, args.output)
        if not args.quiet:
            print(f"Saved calibration profile to {args.output}")

    # Generate report
    if args.report:
        report = generate_report(
            roofline=roofline_result,
            energy=energy_result,
            utilization=utilization_result,
            device_name=args.device_name,
        )
        with open(args.report, 'w') as f:
            f.write(report)
        if not args.quiet:
            print(f"Saved quality report to {args.report}")

    # Print summary if no output specified
    if not args.output and not args.report and not args.quiet:
        print("\nCalibration complete. Use --output to save profile.")


if __name__ == "__main__":
    main()
