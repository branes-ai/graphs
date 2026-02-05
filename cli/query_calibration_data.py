#!/usr/bin/env python3
"""
Query Calibration Data

Query and analyze raw calibration measurements in calibration_data/.

Directory structure (new with precision):
    calibration_data/<hardware_id>/<precision>/measurements/
    calibration_data/<hardware_id>/<precision>/efficiency_curves.json

Legacy structure (still supported):
    calibration_data/<hardware_id>/measurements/
    calibration_data/<hardware_id>/efficiency_curves.json

Usage:
    # List all hardware and summary
    ./cli/query_calibration_data.py --list

    # Show efficiency curves for a hardware target
    ./cli/query_calibration_data.py --id jetson_orin_agx_50w --curves
    ./cli/query_calibration_data.py --id jetson_orin_agx_50w --precision fp16 --curves

    # Compare efficiency across hardware targets
    ./cli/query_calibration_data.py --compare

    # Show measurements for a specific model
    ./cli/query_calibration_data.py --id jetson_orin_agx_50w --model resnet18

    # Export summary to CSV
    ./cli/query_calibration_data.py --compare --output summary.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

repo_root = Path(__file__).parent.parent
calibration_dir = repo_root / "calibration_data"

# Supported precisions
PRECISIONS = ['fp32', 'fp16', 'bf16', 'tf32', 'int8']


def format_flops(flops: float) -> str:
    """Format FLOPs with appropriate unit."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.1f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.0f}K"
    else:
        return f"{flops:.0f}"


def find_calibration_entries() -> List[Tuple[str, str, Path]]:
    """Find all calibration entries (hardware_id, precision, path).

    Handles both new structure (hardware/precision/) and legacy (hardware/).

    Returns:
        List of (hardware_id, precision, calibration_dir) tuples
    """
    entries = []

    if not calibration_dir.exists():
        return entries

    for hw_dir in sorted(calibration_dir.iterdir()):
        if not hw_dir.is_dir():
            continue

        hw_id = hw_dir.name

        # Check for new structure: hardware_id/precision/
        found_precision_dirs = False
        for precision in PRECISIONS:
            prec_dir = hw_dir / precision
            if prec_dir.is_dir():
                # Check if it has measurements or curves
                if (prec_dir / "measurements").exists() or (prec_dir / "efficiency_curves.json").exists():
                    entries.append((hw_id, precision, prec_dir))
                    found_precision_dirs = True

        # Check for legacy structure: hardware_id/measurements/
        if not found_precision_dirs:
            if (hw_dir / "measurements").exists() or (hw_dir / "efficiency_curves.json").exists():
                # Infer precision from efficiency_curves.json if available
                precision = "fp32"  # default
                curves_file = hw_dir / "efficiency_curves.json"
                if curves_file.exists():
                    try:
                        with open(curves_file) as f:
                            curves = json.load(f)
                            precision = curves.get("precision", "FP32").lower()
                    except:
                        pass
                entries.append((hw_id, precision, hw_dir))

    return entries


def load_hardware_data(hardware_id: str, precision: str = "fp32") -> Dict[str, Any]:
    """Load all data for a hardware target and precision."""
    # Try new structure first
    hw_dir = calibration_dir / hardware_id / precision
    if not hw_dir.exists():
        # Try legacy structure
        hw_dir = calibration_dir / hardware_id
        if not hw_dir.exists():
            return None

    data = {
        "hardware_id": hardware_id,
        "precision": precision,
        "measurements": [],
        "efficiency_curves": None,
        "calibration_report": None,
    }

    # Load measurements
    measurements_dir = hw_dir / "measurements"
    if measurements_dir.exists():
        for json_file in sorted(measurements_dir.glob("*.json")):
            with open(json_file) as f:
                data["measurements"].append(json.load(f))

    # Load efficiency curves
    curves_file = hw_dir / "efficiency_curves.json"
    if curves_file.exists():
        with open(curves_file) as f:
            data["efficiency_curves"] = json.load(f)

    return data


def list_hardware():
    """List all hardware targets and their calibration status."""
    print("\n" + "=" * 90)
    print("  CALIBRATION DATA SUMMARY")
    print("=" * 90)

    entries = find_calibration_entries()

    if not entries:
        print("  No calibration data found")
        return

    headers = ["Hardware ID", "Precision", "Models", "Op Types", "Data Points", "Date"]
    print(f"\n  {headers[0]:<25} {headers[1]:<10} {headers[2]:>7} {headers[3]:>9} {headers[4]:>12} {headers[5]:<12}")
    print("  " + "-" * 82)

    for hw_id, precision, cal_dir in entries:
        measurements_dir = cal_dir / "measurements"
        curves_file = cal_dir / "efficiency_curves.json"

        model_count = len(list(measurements_dir.glob("*.json"))) if measurements_dir.exists() else 0

        op_types = 0
        data_points = 0
        date = "-"

        if curves_file.exists():
            with open(curves_file) as f:
                curves = json.load(f)
                op_types = len(curves.get("curves", {}))
                for curve in curves.get("curves", {}).values():
                    data_points += len(curve.get("data_points", []))
                cal_date = curves.get("calibration_date", "")
                if cal_date:
                    date = cal_date[:10]

        print(f"  {hw_id:<25} {precision:<10} {model_count:>7} {op_types:>9} {data_points:>12} {date:<12}")

    print()


def show_curves(hardware_id: str, precision: str = "fp32"):
    """Show efficiency curves for a hardware target."""
    data = load_hardware_data(hardware_id, precision)
    if not data:
        print(f"No data found for {hardware_id} ({precision})")
        return

    curves = data.get("efficiency_curves")
    if not curves:
        print(f"No efficiency curves found for {hardware_id} ({precision})")
        return

    print(f"\n{'=' * 80}")
    print(f"  EFFICIENCY CURVES: {hardware_id}")
    print(f"{'=' * 80}")
    print(f"  Device: {curves.get('device_type', 'unknown')}")
    print(f"  Precision: {curves.get('precision', precision.upper())}")
    print(f"  Date: {curves.get('calibration_date', 'unknown')[:10]}")

    for op_type, curve_data in sorted(curves.get("curves", {}).items()):
        print(f"\n  {op_type}:")
        print(f"    {'FLOPS':>12} {'Efficiency':>10} {'Std':>8} {'Samples':>8} {'Models':>8}")
        print(f"    {'-' * 50}")

        for dp in curve_data.get("data_points", []):
            flops = format_flops(dp["flops"])
            eff = f"{dp['efficiency_mean']:.3f}"
            std = f"{dp['efficiency_std']:.3f}"
            samples = dp.get("num_samples", dp.get("num_observations", 0))
            models = len(dp.get("source_models", []))
            print(f"    {flops:>12} {eff:>10} {std:>8} {samples:>8} {models:>8}")

    print()


def show_model(hardware_id: str, model_name: str, precision: str = "fp32"):
    """Show measurements for a specific model."""
    data = load_hardware_data(hardware_id, precision)
    if not data:
        print(f"No data found for {hardware_id} ({precision})")
        return

    measurement = None
    for m in data["measurements"]:
        if m.get("model") == model_name:
            measurement = m
            break

    if not measurement:
        print(f"No measurement found for {model_name} on {hardware_id} ({precision})")
        return

    print(f"\n{'=' * 80}")
    print(f"  MODEL: {model_name} on {hardware_id}")
    print(f"{'=' * 80}")
    print(f"  Device: {measurement.get('device', 'unknown')}")
    print(f"  Precision: {measurement.get('precision', precision.upper())}")
    print(f"  Theoretical Peak: {format_flops(measurement.get('theoretical_peak_flops', 0))}FLOPS")
    print(f"  Date: {measurement.get('measurement_date', 'unknown')[:10]}")

    print(f"\n  {'SG':<4} {'Pattern':<28} {'OpType':<16} {'FLOPs':>10} {'Eff':>8} {'Lat(ms)':>10}")
    print(f"  {'-' * 85}")

    for sg in measurement.get("subgraphs", []):
        sg_id = sg["subgraph_id"]
        pattern = sg["fusion_pattern"][:27]
        op_type = sg["operation_type"][:15]
        flops = format_flops(sg["flops"])
        eff = f"{sg['efficiency']['mean']:.4f}"
        lat = f"{sg['measured_latency']['mean']:.3f}"
        print(f"  {sg_id:<4} {pattern:<28} {op_type:<16} {flops:>10} {eff:>8} {lat:>10}")

    print()


def compare_hardware(precision_filter: str = None):
    """Compare efficiency across hardware targets."""
    print(f"\n{'=' * 100}")
    print("  HARDWARE COMPARISON")
    print(f"{'=' * 100}")

    entries = find_calibration_entries()

    if not entries:
        print("  No calibration data found")
        return

    # Filter by precision if specified
    if precision_filter:
        entries = [(hw, prec, path) for hw, prec, path in entries if prec == precision_filter]
        if not entries:
            print(f"  No calibration data found for precision {precision_filter}")
            return

    # Collect data
    hw_data = {}
    all_op_types = set()

    for hw_id, precision, cal_dir in entries:
        curves_file = cal_dir / "efficiency_curves.json"

        if curves_file.exists():
            with open(curves_file) as f:
                curves = json.load(f)
                key = f"{hw_id}/{precision}"
                hw_data[key] = curves
                all_op_types.update(curves.get("curves", {}).keys())

    if not hw_data:
        print("  No efficiency curves found")
        return

    # Print comparison by operation type
    for op_type in sorted(all_op_types):
        print(f"\n  {op_type}:")
        print(f"    {'Hardware':<30} {'Prec':<6} {'Peak Eff':>10} {'@ FLOPs':>12} {'Data Pts':>10}")
        print(f"    {'-' * 72}")

        for key, curves in sorted(hw_data.items()):
            hw_id, precision = key.rsplit('/', 1)
            curve = curves.get("curves", {}).get(op_type)
            if not curve:
                continue

            # Find peak efficiency
            peak_eff = 0
            peak_flops = 0
            for dp in curve.get("data_points", []):
                if dp["efficiency_mean"] > peak_eff:
                    peak_eff = dp["efficiency_mean"]
                    peak_flops = dp["flops"]

            data_pts = len(curve.get("data_points", []))
            print(f"    {hw_id:<30} {precision:<6} {peak_eff:>10.3f} {format_flops(peak_flops):>12} {data_pts:>10}")

    print()


def export_csv(output_path: str, precision_filter: str = None):
    """Export comparison data to CSV."""
    entries = find_calibration_entries()

    if precision_filter:
        entries = [(hw, prec, path) for hw, prec, path in entries if prec == precision_filter]

    rows = []

    for hw_id, precision, cal_dir in entries:
        curves_file = cal_dir / "efficiency_curves.json"

        if curves_file.exists():
            with open(curves_file) as f:
                curves = json.load(f)

            for op_type, curve in curves.get("curves", {}).items():
                for dp in curve.get("data_points", []):
                    rows.append({
                        "hardware_id": hw_id,
                        "precision": precision,
                        "device_type": curves.get("device_type", ""),
                        "operation_type": op_type,
                        "flops": dp["flops"],
                        "efficiency_mean": dp["efficiency_mean"],
                        "efficiency_std": dp["efficiency_std"],
                        "num_observations": dp.get("num_observations", 0),
                        "num_samples": dp.get("num_samples", 0),
                        "source_models": ",".join(dp.get("source_models", [])),
                    })

    with open(output_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Query calibration data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --id jetson_orin_agx_50w --curves
  %(prog)s --id jetson_orin_agx_50w --precision fp16 --curves
  %(prog)s --id jetson_orin_agx_50w --model resnet18
  %(prog)s --compare
  %(prog)s --compare --precision fp16
  %(prog)s --compare --output summary.csv
""")
    parser.add_argument('--list', action='store_true',
                        help='List all hardware and summary')
    parser.add_argument('--id', type=str,
                        help='Hardware ID to query')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=PRECISIONS,
                        help='Precision to query (default: fp32)')
    parser.add_argument('--curves', action='store_true',
                        help='Show efficiency curves')
    parser.add_argument('--model', type=str,
                        help='Show measurements for a specific model')
    parser.add_argument('--compare', action='store_true',
                        help='Compare efficiency across hardware')
    parser.add_argument('--output', type=str,
                        help='Export to CSV file')

    args = parser.parse_args()

    if args.output and args.compare:
        export_csv(args.output, args.precision if args.precision != 'fp32' else None)
        return 0

    if args.list or (not args.id and not args.compare):
        list_hardware()
        return 0

    if args.compare:
        compare_hardware(args.precision if args.precision != 'fp32' else None)
        return 0

    if args.id:
        if args.model:
            show_model(args.id, args.model, args.precision)
        elif args.curves:
            show_curves(args.id, args.precision)
        else:
            show_curves(args.id, args.precision)
        return 0

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
