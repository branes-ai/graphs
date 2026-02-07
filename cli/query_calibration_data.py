#!/usr/bin/env python3
"""
Query Calibration Data

Query and analyze raw calibration measurements in calibration_data/.

Uses GroundTruthLoader for measurement data and reads efficiency curves
directly from the file system.

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
from pathlib import Path
from typing import Dict, List, Optional, Any

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.calibration.ground_truth import GroundTruthLoader, _normalize_precision

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


def _find_curves_file(hardware_id: str, precision: str) -> Optional[Path]:
    """Find efficiency_curves.json for a hardware/precision combo."""
    # v2.0 canonical: calibration_data/<hw>/<prec>/efficiency_curves.json
    p1 = calibration_dir / hardware_id / precision / "efficiency_curves.json"
    if p1.exists():
        return p1
    # Legacy: calibration_data/<hw>/efficiency_curves.json
    p2 = calibration_dir / hardware_id / "efficiency_curves.json"
    if p2.exists():
        return p2
    return None


def _find_all_curves_entries() -> List[Dict[str, Any]]:
    """Find all (hardware_id, precision, curves_path) combos with efficiency curves."""
    entries = []
    if not calibration_dir.exists():
        return entries

    for hw_dir in sorted(calibration_dir.iterdir()):
        if not hw_dir.is_dir():
            continue
        hw_id = hw_dir.name

        # Check precision subdirs for curves
        found = False
        for precision in PRECISIONS:
            curves_path = hw_dir / precision / "efficiency_curves.json"
            if curves_path.exists():
                entries.append({
                    'hardware_id': hw_id,
                    'precision': precision,
                    'curves_path': curves_path,
                })
                found = True

        # Legacy: curves at top level
        if not found:
            curves_path = hw_dir / "efficiency_curves.json"
            if curves_path.exists():
                precision = "fp32"
                try:
                    with open(curves_path) as f:
                        curves = json.load(f)
                        precision = curves.get("precision", "FP32").lower()
                except Exception:
                    pass
                entries.append({
                    'hardware_id': hw_id,
                    'precision': precision,
                    'curves_path': curves_path,
                })

    return entries


def list_hardware():
    """List all hardware targets and their calibration status."""
    loader = GroundTruthLoader(calibration_dir)
    hw_ids = loader.list_hardware()
    curves_entries = _find_all_curves_entries()

    # Merge: measurement data + curves data
    # Build a set of all (hw_id, precision) combos from both sources
    all_combos = {}

    # From measurements
    for hw_id in hw_ids:
        configs = loader.list_configurations(hw_id)
        by_precision = {}
        for cfg in configs:
            prec = cfg['precision']
            by_precision.setdefault(prec, []).append(cfg)
        for prec, cfgs in by_precision.items():
            key = (hw_id, prec)
            all_combos.setdefault(key, {'model_count': 0, 'op_types': 0, 'data_points': 0, 'date': '-'})
            all_combos[key]['model_count'] = len(cfgs)

    # From curves
    for entry in curves_entries:
        key = (entry['hardware_id'], entry['precision'])
        all_combos.setdefault(key, {'model_count': 0, 'op_types': 0, 'data_points': 0, 'date': '-'})
        try:
            with open(entry['curves_path']) as f:
                curves = json.load(f)
                all_combos[key]['op_types'] = len(curves.get("curves", {}))
                for curve in curves.get("curves", {}).values():
                    all_combos[key]['data_points'] += len(curve.get("data_points", []))
                cal_date = curves.get("calibration_date", "")
                if cal_date:
                    all_combos[key]['date'] = cal_date[:10]
        except Exception:
            pass

    if not all_combos:
        print("\n  No calibration data found")
        return

    # Calculate column width
    hw_id_width = max(len("Hardware ID"), max(len(k[0]) for k in all_combos))
    total_width = hw_id_width + 60

    print("\n" + "=" * total_width)
    print("  CALIBRATION DATA SUMMARY")
    print("=" * total_width)

    headers = ["Hardware ID", "Precision", "Models", "Op Types", "Data Points", "Date"]
    print(f"\n  {headers[0]:<{hw_id_width}} {headers[1]:<10} {headers[2]:>7} {headers[3]:>9} {headers[4]:>12} {headers[5]:<12}")
    print("  " + "-" * (total_width - 2))

    for (hw_id, precision), info in sorted(all_combos.items()):
        print(f"  {hw_id:<{hw_id_width}} {precision:<10} {info['model_count']:>7} "
              f"{info['op_types']:>9} {info['data_points']:>12} {info['date']:<12}")

    print()


def show_curves(hardware_id: str, precision: str = "fp32"):
    """Show efficiency curves for a hardware target."""
    curves_path = _find_curves_file(hardware_id, precision)
    if not curves_path:
        print(f"No efficiency curves found for {hardware_id} ({precision})")
        return

    with open(curves_path) as f:
        curves = json.load(f)

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
    loader = GroundTruthLoader(calibration_dir)

    try:
        record = loader.load(hardware_id, model_name, precision, batch_size=1)
    except FileNotFoundError:
        print(f"No measurement found for {model_name} on {hardware_id} ({precision})")
        return

    print(f"\n{'=' * 80}")
    print(f"  MODEL: {model_name} on {hardware_id}")
    print(f"{'=' * 80}")
    print(f"  Device: {record.device}")
    print(f"  Precision: {record.precision}")
    print(f"  Batch Size: {record.batch_size}")
    print(f"  Theoretical Peak: {format_flops(record.theoretical_peak_flops)}FLOPS")
    print(f"  Date: {record.measurement_date[:10]}")

    if record.model_summary:
        ms = record.model_summary
        print(f"  Total Latency: {ms.total_latency_ms:.3f} ms")
        print(f"  Throughput: {ms.throughput_fps:.1f} FPS")

    print(f"\n  {'SG':<4} {'Pattern':<28} {'OpType':<16} {'FLOPs':>10} {'Eff':>8} {'Lat(ms)':>10}")
    print(f"  {'-' * 85}")

    for sg in record.subgraphs:
        pattern = sg.fusion_pattern[:27]
        op_type = sg.operation_type[:15]
        flops = format_flops(sg.flops)
        eff = f"{sg.efficiency.mean:.4f}"
        lat = f"{sg.measured_latency.mean:.3f}"
        print(f"  {sg.subgraph_id:<4} {pattern:<28} {op_type:<16} {flops:>10} {eff:>8} {lat:>10}")

    print()


def compare_hardware(precision_filter: str = None):
    """Compare efficiency across hardware targets."""
    print(f"\n{'=' * 100}")
    print("  HARDWARE COMPARISON")
    print(f"{'=' * 100}")

    entries = _find_all_curves_entries()

    if not entries:
        print("  No calibration data found")
        return

    # Filter by precision if specified
    if precision_filter:
        entries = [e for e in entries if e['precision'] == precision_filter]
        if not entries:
            print(f"  No calibration data found for precision {precision_filter}")
            return

    # Collect data
    hw_data = {}
    all_op_types = set()

    for entry in entries:
        try:
            with open(entry['curves_path']) as f:
                curves = json.load(f)
                key = f"{entry['hardware_id']}/{entry['precision']}"
                hw_data[key] = curves
                all_op_types.update(curves.get("curves", {}).keys())
        except Exception:
            pass

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
    entries = _find_all_curves_entries()

    if precision_filter:
        entries = [e for e in entries if e['precision'] == precision_filter]

    rows = []

    for entry in entries:
        try:
            with open(entry['curves_path']) as f:
                curves = json.load(f)
        except Exception:
            continue

        for op_type, curve in curves.get("curves", {}).items():
            for dp in curve.get("data_points", []):
                rows.append({
                    "hardware_id": entry['hardware_id'],
                    "precision": entry['precision'],
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
