#!/usr/bin/env python3
"""
Product Trajectory Analysis CLI

Analyzes post-silicon dynamics: how hardware efficiency improves over time
as software stacks (drivers, frameworks, runtimes) mature.

Usage:
    # Show efficiency trajectory for current hardware
    ./cli/analyze_product_trajectory.py

    # Show trajectory for specific hardware fingerprint
    ./cli/analyze_product_trajectory.py --hardware abc123def456

    # Compare multiple products
    ./cli/analyze_product_trajectory.py --compare hw1 hw2 hw3

    # Show driver impact analysis
    ./cli/analyze_product_trajectory.py --driver-impact

    # Detect and report regressions
    ./cli/analyze_product_trajectory.py --detect-regressions --threshold 3.0

    # Time-to-maturity analysis
    ./cli/analyze_product_trajectory.py --maturity-analysis

    # Export data for external visualization
    ./cli/analyze_product_trajectory.py --export trajectory.csv
    ./cli/analyze_product_trajectory.py --export trajectory.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.calibration.calibration_db import CalibrationDB, CalibrationRun, RegressionAlert
from graphs.calibration.auto_detect import detect_all, CalibrationContext


# =============================================================================
# ASCII VISUALIZATION
# =============================================================================

def render_ascii_trajectory_plot(
    runs: List[CalibrationRun],
    metric: str = "peak_measured_gops",
    width: int = 60,
    height: int = 15,
    title: Optional[str] = None
) -> str:
    """
    Render an ASCII art plot of metric over time.

    Args:
        runs: List of calibration runs, sorted by time.
        metric: Which metric to plot.
        width: Plot width in characters.
        height: Plot height in lines.
        title: Optional plot title.

    Returns:
        Multi-line string with ASCII plot.
    """
    if not runs:
        return "No data to plot."

    # Extract data points
    values = [getattr(r, metric, 0) for r in runs]
    timestamps = [datetime.fromisoformat(r.timestamp) for r in runs]

    if not values or max(values) == 0:
        return f"No valid {metric} data to plot."

    # Calculate ranges
    min_val = min(v for v in values if v > 0) * 0.9 if any(v > 0 for v in values) else 0
    max_val = max(values) * 1.1

    if max_val == min_val:
        max_val = min_val + 1

    # Time range
    time_start = timestamps[0]
    time_end = timestamps[-1]
    time_range = (time_end - time_start).total_seconds()
    if time_range == 0:
        time_range = 1

    # Build plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for i, (val, ts) in enumerate(zip(values, timestamps)):
        if val <= 0:
            continue

        # Calculate position
        x = int((ts - time_start).total_seconds() / time_range * (width - 1))
        y = height - 1 - int((val - min_val) / (max_val - min_val) * (height - 1))

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        # Mark point
        if i == len(values) - 1:
            grid[y][x] = '*'  # Latest point
        else:
            grid[y][x] = 'o'

        # Draw connecting line to previous point
        if i > 0:
            prev_val = values[i - 1]
            prev_ts = timestamps[i - 1]
            prev_x = int((prev_ts - time_start).total_seconds() / time_range * (width - 1))
            prev_y = height - 1 - int((prev_val - min_val) / (max_val - min_val) * (height - 1))
            prev_x = max(0, min(width - 1, prev_x))
            prev_y = max(0, min(height - 1, prev_y))

            # Simple line drawing
            if x > prev_x:
                for lx in range(prev_x + 1, x):
                    # Interpolate y
                    t = (lx - prev_x) / (x - prev_x) if x != prev_x else 0
                    ly = int(prev_y + t * (y - prev_y))
                    ly = max(0, min(height - 1, ly))
                    if grid[ly][lx] == ' ':
                        grid[ly][lx] = '-'

    # Build output
    lines = []

    # Title
    if title:
        lines.append(f"    {title}")
        lines.append("")

    # Y-axis label formatting
    if max_val >= 1000:
        y_fmt = lambda v: f"{v/1000:.1f}K"
    else:
        y_fmt = lambda v: f"{v:.1f}"

    # Add Y axis and grid
    for i, row in enumerate(grid):
        # Y-axis value
        y_val = max_val - i * (max_val - min_val) / (height - 1)
        y_label = y_fmt(y_val)
        line = f"{y_label:>8} |{''.join(row)}|"
        lines.append(line)

    # X-axis
    lines.append(f"{'':>8} +{'-' * width}+")

    # X-axis labels
    start_label = time_start.strftime("%Y-%m-%d")
    end_label = time_end.strftime("%Y-%m-%d")
    mid_label = (time_start + (time_end - time_start) / 2).strftime("%m-%d")
    x_labels = f"{'':>9}{start_label}{mid_label:^{width-20}}{end_label}"
    lines.append(x_labels)

    # Legend
    lines.append("")
    lines.append(f"    o = data point, * = latest measurement")
    lines.append(f"    Metric: {metric}")

    return '\n'.join(lines)


def render_efficiency_trajectory(
    runs: List[CalibrationRun],
    hw_name: str,
    width: int = 60,
    height: int = 12
) -> str:
    """Render efficiency trajectory with software version annotations."""
    title = f"Efficiency vs Time: {hw_name}"
    return render_ascii_trajectory_plot(
        runs,
        metric="peak_measured_gops",
        width=width,
        height=height,
        title=title
    )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_maturity(db: CalibrationDB, hardware_fps: Optional[List[str]] = None) -> str:
    """
    Generate time-to-maturity analysis table.

    Shows how long each product took to reach efficiency milestones.
    """
    if hardware_fps is None:
        hardware_fps = db.get_all_hardware()

    if not hardware_fps:
        return "No hardware data found."

    lines = [
        "",
        "=" * 78,
        "TIME-TO-MATURITY ANALYSIS",
        "=" * 78,
        "",
        "How long (days) to reach efficiency milestones from first calibration:",
        "",
        f"{'Product':<30} {'70%':>8} {'80%':>8} {'90%':>8} {'Current':>10}",
        "-" * 78,
    ]

    for hw_fp in hardware_fps:
        summary = db.get_hardware_summary(hw_fp)
        if not summary:
            continue

        cpu_model = summary.get('cpu_model', 'Unknown')[:28]
        trajectory = db.get_trajectory(hw_fp)

        if not trajectory:
            continue

        first_time = datetime.fromisoformat(trajectory[0].timestamp)
        current_eff = trajectory[-1].efficiency * 100

        # Find time to each milestone
        milestones = {}
        for threshold in [0.7, 0.8, 0.9]:
            delta = db.get_time_to_milestone(hw_fp, threshold)
            if delta:
                milestones[threshold] = delta.days
            else:
                milestones[threshold] = None

        # Format row
        def fmt_days(d):
            return f"Day {d}" if d is not None else "N/A"

        lines.append(
            f"{cpu_model:<30} "
            f"{fmt_days(milestones.get(0.7)):>8} "
            f"{fmt_days(milestones.get(0.8)):>8} "
            f"{fmt_days(milestones.get(0.9)):>8} "
            f"{current_eff:>9.1f}%"
        )

    # Assessment section
    lines.extend([
        "",
        "Assessment Legend:",
        "  - Day 0-30 to 90%: Excellent software support",
        "  - Day 30-90 to 90%: Good software support",
        "  - Day 90+ to 90%: Slow optimization",
        "  - N/A at 90%: Concerning - may not reach target",
    ])

    return '\n'.join(lines)


def analyze_driver_impact(
    db: CalibrationDB,
    hardware_fp: str,
    precision: str = "fp32"
) -> str:
    """
    Analyze impact of driver/software changes on performance.
    """
    groups = db.get_software_impact(hardware_fp, precision, "software_fingerprint")

    if not groups:
        return f"No data for hardware {hardware_fp[:8]}..."

    lines = [
        "",
        "=" * 78,
        f"SOFTWARE IMPACT ANALYSIS: {hardware_fp[:8]}...",
        "=" * 78,
        "",
        f"{'SW Fingerprint':<18} {'Driver':<12} {'PyTorch':<10} {'GOPS':>10} {'BW (GB/s)':>10}",
        "-" * 78,
    ]

    # Sort groups by timestamp of first run
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: datetime.fromisoformat(kv[1][0].timestamp)
    )

    for sw_fp, runs in sorted_groups:
        # Average performance across runs with this software
        avg_gops = sum(r.peak_measured_gops for r in runs) / len(runs)
        avg_bw = sum(r.stream_best_gbps for r in runs) / len(runs)

        # Get software versions from first run
        first_run = runs[0]
        driver = first_run.gpu_driver_version[:10] if first_run.gpu_driver_version != "N/A" else "N/A"
        pytorch = first_run.pytorch_version[:8]

        lines.append(
            f"{sw_fp[:16]:<18} "
            f"{driver:<12} "
            f"{pytorch:<10} "
            f"{avg_gops:>10.1f} "
            f"{avg_bw:>10.1f}"
        )

    # Calculate improvement
    if len(sorted_groups) >= 2:
        first_sw, first_runs = sorted_groups[0]
        last_sw, last_runs = sorted_groups[-1]

        first_gops = sum(r.peak_measured_gops for r in first_runs) / len(first_runs)
        last_gops = sum(r.peak_measured_gops for r in last_runs) / len(last_runs)

        improvement = (last_gops - first_gops) / first_gops * 100 if first_gops > 0 else 0

        lines.extend([
            "",
            f"Total improvement from software updates: {improvement:+.1f}%",
            f"  First SW stack: {first_gops:.1f} GOPS",
            f"  Latest SW stack: {last_gops:.1f} GOPS",
        ])

    return '\n'.join(lines)


def analyze_regressions(
    db: CalibrationDB,
    threshold_pct: float = 5.0
) -> str:
    """
    Detailed regression analysis report.
    """
    alerts = db.detect_regressions(threshold_pct=threshold_pct)

    lines = [
        "",
        "=" * 78,
        f"REGRESSION DETECTION (threshold: {threshold_pct}%)",
        "=" * 78,
        "",
    ]

    if not alerts:
        lines.append("No regressions detected.")
        return '\n'.join(lines)

    lines.append(f"Found {len(alerts)} regression(s):")
    lines.append("")

    for i, alert in enumerate(alerts, 1):
        lines.extend([
            f"--- Regression #{i} ---",
            f"Hardware:   {alert.hardware_fingerprint[:8]}... ({alert.previous_run.cpu_model[:30]})",
            f"Precision:  {alert.precision}",
            f"Metric:     {alert.metric}",
            f"",
            f"Previous:   {alert.previous_value:.2f}",
            f"  Date:     {alert.previous_run.timestamp[:10]}",
            f"  SW Stack: {alert.previous_run.software_fingerprint[:8]}...",
            f"  Driver:   {alert.previous_run.gpu_driver_version}",
            f"  PyTorch:  {alert.previous_run.pytorch_version}",
            f"",
            f"Current:    {alert.current_value:.2f}",
            f"  Date:     {alert.current_run.timestamp[:10]}",
            f"  SW Stack: {alert.current_run.software_fingerprint[:8]}...",
            f"  Driver:   {alert.current_run.gpu_driver_version}",
            f"  PyTorch:  {alert.current_run.pytorch_version}",
            f"",
            f"Regression: -{alert.regression_pct:.1f}%",
            "",
            "Likely cause: Software stack change",
            "Recommendation: Investigate driver/framework update",
            "",
        ])

    return '\n'.join(lines)


def compare_products(
    db: CalibrationDB,
    hardware_fps: List[str],
    metric: str = "peak_measured_gops"
) -> str:
    """
    Side-by-side comparison of multiple products' trajectories.
    """
    if len(hardware_fps) < 2:
        return "Need at least 2 hardware fingerprints to compare."

    lines = [
        "",
        "=" * 78,
        "CROSS-PRODUCT COMPARISON",
        "=" * 78,
        "",
    ]

    # Build comparison table
    header = f"{'Metric':<25}"
    for hw_fp in hardware_fps:
        header += f" {hw_fp[:10]:>12}"
    lines.append(header)
    lines.append("-" * (25 + 13 * len(hardware_fps)))

    # Get data for each product
    product_data = {}
    for hw_fp in hardware_fps:
        trajectory = db.get_trajectory(hw_fp)
        if trajectory:
            product_data[hw_fp] = {
                'first': trajectory[0],
                'latest': trajectory[-1],
                'count': len(trajectory),
                'improvement_rate': db.get_improvement_rate(hw_fp),
            }

    # Rows
    metrics = [
        ("CPU Model", lambda d: d['latest'].cpu_model[:12] if d else "N/A"),
        ("First GOPS", lambda d: f"{d['first'].peak_measured_gops:.1f}" if d else "N/A"),
        ("Latest GOPS", lambda d: f"{d['latest'].peak_measured_gops:.1f}" if d else "N/A"),
        ("First BW (GB/s)", lambda d: f"{d['first'].stream_best_gbps:.1f}" if d else "N/A"),
        ("Latest BW (GB/s)", lambda d: f"{d['latest'].stream_best_gbps:.1f}" if d else "N/A"),
        ("Calibration Runs", lambda d: str(d['count']) if d else "0"),
        ("Improvement (%/month)", lambda d: f"{d['improvement_rate']:.1f}" if d and d['improvement_rate'] else "N/A"),
    ]

    for metric_name, extractor in metrics:
        row = f"{metric_name:<25}"
        for hw_fp in hardware_fps:
            data = product_data.get(hw_fp)
            value = extractor(data)
            row += f" {value:>12}"
        lines.append(row)

    return '\n'.join(lines)


def export_trajectory_csv(
    db: CalibrationDB,
    hardware_fp: Optional[str],
    output_path: str
) -> bool:
    """Export trajectory data to CSV for external visualization."""
    try:
        if hardware_fp:
            runs = db.get_trajectory(hardware_fp)
        else:
            # Export all runs
            cursor = db.conn.cursor()
            cursor.execute("SELECT * FROM calibration_runs ORDER BY timestamp_unix")
            runs = [CalibrationRun.from_dict(dict(row)) for row in cursor.fetchall()]

        if not runs:
            print(f"No data to export.")
            return False

        # CSV headers
        headers = [
            "run_id", "timestamp", "hardware_fingerprint", "software_fingerprint",
            "cpu_model", "gpu_model", "precision", "device",
            "peak_measured_gops", "stream_best_gbps", "efficiency",
            "gpu_driver_version", "pytorch_version", "numpy_version",
            "cpu_governor", "power_mode", "thermal_state"
        ]

        with open(output_path, 'w') as f:
            f.write(','.join(headers) + '\n')

            for run in runs:
                values = [
                    run.run_id,
                    run.timestamp,
                    run.hardware_fingerprint,
                    run.software_fingerprint,
                    f'"{run.cpu_model}"',
                    f'"{run.gpu_model}"',
                    run.precision,
                    run.device,
                    str(run.peak_measured_gops),
                    str(run.stream_best_gbps),
                    str(run.efficiency),
                    run.gpu_driver_version,
                    run.pytorch_version,
                    run.numpy_version,
                    run.cpu_governor,
                    run.power_mode,
                    run.thermal_state,
                ]
                f.write(','.join(values) + '\n')

        print(f"Exported {len(runs)} records to {output_path}")
        return True

    except Exception as e:
        print(f"Export error: {e}")
        return False


def export_trajectory_json(
    db: CalibrationDB,
    hardware_fp: Optional[str],
    output_path: str
) -> bool:
    """Export trajectory data to JSON for external visualization."""
    try:
        data = {
            'export_date': datetime.now(timezone.utc).isoformat(),
            'summary': db.get_summary(),
            'trajectories': {},
        }

        if hardware_fp:
            hardware_fps = [hardware_fp]
        else:
            hardware_fps = db.get_all_hardware()

        for hw_fp in hardware_fps:
            trajectory = db.get_trajectory(hw_fp)
            if trajectory:
                data['trajectories'][hw_fp] = {
                    'hardware_summary': db.get_hardware_summary(hw_fp),
                    'runs': [run.to_dict() for run in trajectory],
                    'improvement_rate': db.get_improvement_rate(hw_fp),
                }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(data['trajectories'])} trajectories to {output_path}")
        return True

    except Exception as e:
        print(f"Export error: {e}")
        return False


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze post-silicon product efficiency trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show trajectory for current hardware
  %(prog)s

  # Show trajectory for specific hardware
  %(prog)s --hardware abc123def456

  # Compare multiple products
  %(prog)s --compare hw1 hw2 hw3

  # Detect regressions
  %(prog)s --detect-regressions --threshold 3.0

  # Time-to-maturity analysis
  %(prog)s --maturity-analysis

  # Driver impact analysis
  %(prog)s --driver-impact

  # Export for visualization
  %(prog)s --export trajectory.csv
  %(prog)s --export trajectory.json
        """
    )

    # Target selection
    parser.add_argument(
        "--hardware", "-H",
        help="Hardware fingerprint to analyze (default: auto-detect current)"
    )
    parser.add_argument(
        "--compare", "-c",
        nargs="+",
        metavar="HW_FP",
        help="Compare multiple hardware fingerprints"
    )

    # Analysis modes
    parser.add_argument(
        "--maturity-analysis", "-m",
        action="store_true",
        help="Show time-to-maturity analysis"
    )
    parser.add_argument(
        "--driver-impact", "-d",
        action="store_true",
        help="Show driver/software impact analysis"
    )
    parser.add_argument(
        "--detect-regressions", "-r",
        action="store_true",
        help="Detect and report performance regressions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0)"
    )

    # Output options
    parser.add_argument(
        "--export", "-e",
        metavar="FILE",
        help="Export trajectory data to CSV or JSON"
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help="Precision to analyze (default: fp32)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device type (default: cpu)"
    )
    parser.add_argument(
        "--list-hardware", "-l",
        action="store_true",
        help="List all hardware fingerprints in database"
    )

    # Database
    parser.add_argument(
        "--db",
        default=str(Path(__file__).parent.parent / "results" / "calibration_db" / "calibrations_v2.db"),
        help="Path to calibration database"
    )

    # Display options
    parser.add_argument(
        "--width",
        type=int,
        default=60,
        help="Plot width in characters (default: 60)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=12,
        help="Plot height in lines (default: 12)"
    )

    args = parser.parse_args()

    # Auto-detect current hardware if needed
    context = None
    if not args.hardware and not args.compare and not args.list_hardware:
        print("Auto-detecting hardware...")
        context = detect_all()
        args.hardware = context.hardware.fingerprint
        print(f"Hardware fingerprint: {args.hardware}")
        print(f"CPU: {context.hardware.cpu.model}")
        print()

    # Open database
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        print("Run benchmark_sweep.py first to create calibration data.")
        return 1

    db = CalibrationDB(str(db_path))

    try:
        # List hardware
        if args.list_hardware:
            print("\n" + "=" * 60)
            print("HARDWARE IN DATABASE")
            print("=" * 60 + "\n")

            for hw_fp in db.get_all_hardware():
                summary = db.get_hardware_summary(hw_fp)
                cpu_model = summary.get('cpu_model', 'Unknown')
                runs = summary.get('total_runs', 0)
                print(f"  {hw_fp}  {cpu_model[:40]:<40}  ({runs} runs)")

            return 0

        # Export
        if args.export:
            export_path = args.export
            if export_path.endswith('.csv'):
                export_trajectory_csv(db, args.hardware, export_path)
            elif export_path.endswith('.json'):
                export_trajectory_json(db, args.hardware, export_path)
            else:
                print(f"Unknown export format. Use .csv or .json extension.")
                return 1
            return 0

        # Compare products
        if args.compare:
            print(compare_products(db, args.compare))
            return 0

        # Regression detection
        if args.detect_regressions:
            print(analyze_regressions(db, args.threshold))
            return 0

        # Maturity analysis
        if args.maturity_analysis:
            hardware_list = [args.hardware] if args.hardware else None
            print(analyze_maturity(db, hardware_list))
            return 0

        # Driver impact
        if args.driver_impact:
            if not args.hardware:
                print("Error: --driver-impact requires --hardware")
                return 1
            print(analyze_driver_impact(db, args.hardware, args.precision))
            return 0

        # Default: show trajectory for current/specified hardware
        if args.hardware:
            trajectory = db.get_trajectory(
                args.hardware,
                args.precision,
                args.device
            )

            if not trajectory:
                print(f"No calibration data for hardware {args.hardware[:8]}...")
                print("Run benchmark_sweep.py to calibrate this hardware.")
                return 1

            # Get hardware name
            hw_name = trajectory[0].cpu_model[:30]

            print("\n" + "=" * 78)
            print(f"EFFICIENCY TRAJECTORY: {args.hardware[:8]}...")
            print("=" * 78)

            # ASCII plot
            print()
            print(render_efficiency_trajectory(
                trajectory,
                hw_name,
                width=args.width,
                height=args.height
            ))

            # Summary table
            print()
            print("-" * 78)
            print(f"{'Date':<12} {'SW Stack':<12} {'GOPS':>12} {'BW (GB/s)':>12} {'Eff':>10}")
            print("-" * 78)

            for run in trajectory:
                date = run.timestamp[:10]
                sw = run.software_fingerprint[:10]
                gops = run.peak_measured_gops
                bw = run.stream_best_gbps
                eff = run.efficiency * 100

                print(f"{date:<12} {sw:<12} {gops:>12.1f} {bw:>12.1f} {eff:>9.1f}%")

            # Improvement rate
            rate = db.get_improvement_rate(args.hardware, args.precision)
            if rate is not None:
                print()
                print(f"Improvement rate: {rate:+.2f}%/month")

            # Time to milestones
            print()
            print("Milestones:")
            for threshold in [0.7, 0.8, 0.9]:
                delta = db.get_time_to_milestone(args.hardware, threshold, args.precision)
                if delta:
                    print(f"  {threshold*100:.0f}% efficiency: Day {delta.days}")
                else:
                    print(f"  {threshold*100:.0f}% efficiency: Not reached")

            return 0

        # No specific action - show help
        parser.print_help()
        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
