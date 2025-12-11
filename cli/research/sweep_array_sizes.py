#!/usr/bin/env python
"""
Systolic Array Size Sweep Tool

Sweep systolic array sizes and analyze utilization across DNN workloads.

Usage:
    python cli/research/sweep_array_sizes.py --input shapes.parquet --output utilization/
    python cli/research/sweep_array_sizes.py --sizes 16 32 64 128 --precision BF16
    python cli/research/sweep_array_sizes.py --input shapes.parquet --find-optimal
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

from graphs.research.shape_collection import ShapeDatabase
from graphs.research.systolic import (
    ARRAY_SIZES,
    PRECISIONS,
    ArraySizeSweeper,
    sweep_by_class,
    analyze_size_sensitivity,
)
from graphs.research.systolic.visualization import (
    plot_utilization_vs_array_size,
    plot_utilization_by_class,
    plot_utilization_heatmap,
    plot_utilization_histogram_grid,
    plot_optimal_size_analysis,
    plot_size_comparison_table,
    generate_all_utilization_plots,
)


def main():
    parser = argparse.ArgumentParser(
        description='Sweep systolic array sizes and analyze utilization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full sweep and generate all plots
    python cli/research/sweep_array_sizes.py --input shapes.parquet --output utilization/

    # Sweep specific array sizes
    python cli/research/sweep_array_sizes.py --input shapes.parquet --sizes 16 32 64 128

    # Find optimal array size
    python cli/research/sweep_array_sizes.py --input shapes.parquet --find-optimal

    # Analyze by DNN class
    python cli/research/sweep_array_sizes.py --input shapes.parquet --by-class --output utilization/
        """,
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input shape database (.parquet, .csv, or .json)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='utilization',
        help='Output directory for results (default: utilization/)',
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=ARRAY_SIZES,
        help=f'Array sizes to sweep (default: {ARRAY_SIZES})',
    )
    parser.add_argument(
        '--precision', '-p',
        type=str,
        default='BF16',
        choices=PRECISIONS,
        help='Precision for analysis (default: BF16)',
    )
    parser.add_argument(
        '--all-precisions',
        action='store_true',
        help='Sweep all precisions (FP32, BF16, INT8)',
    )
    parser.add_argument(
        '--by-class',
        action='store_true',
        help='Analyze separately by DNN class',
    )
    parser.add_argument(
        '--find-optimal',
        action='store_true',
        help='Find and report optimal array size',
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='pdf',
        choices=['pdf', 'svg', 'png'],
        help='Output format for plots (default: pdf)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress',
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Export results to CSV file',
    )

    args = parser.parse_args()

    # Load database
    print(f"Loading shapes from {args.input}...")
    db = ShapeDatabase.load(args.input)

    stats = db.get_statistics()
    print(f"Loaded {stats['total_records']} records ({stats['matmul_ops']} matmul ops)")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine precisions to sweep
    precisions = PRECISIONS if args.all_precisions else [args.precision]

    # Progress callback
    def progress(size, prec):
        if args.verbose:
            print(f"  Analyzing {size}x{size} {prec}...")

    # Run sweep
    print(f"Sweeping array sizes: {args.sizes}")
    print(f"Precisions: {precisions}")
    print()

    sweeper = ArraySizeSweeper(array_sizes=args.sizes, precisions=precisions)
    results = sweeper.sweep(db, progress_callback=progress if args.verbose else None)

    # Print summary
    print()
    print("=" * 70)
    print("UTILIZATION SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Size':>6} {'Precision':>10} {'Mean':>8} {'Weighted':>10} {'Median':>8} {'>50%':>8} {'>75%':>8}")
    print("-" * 70)

    for size in args.sizes:
        for prec in precisions:
            key = (size, prec)
            if key in results:
                r = results[key]
                pct_50 = r.shapes_above_50pct / r.total_shapes * 100 if r.total_shapes > 0 else 0
                pct_75 = r.shapes_above_75pct / r.total_shapes * 100 if r.total_shapes > 0 else 0
                print(f"{size:>6} {prec:>10} {r.mean_utilization:>8.2%} "
                      f"{r.weighted_mean_utilization:>10.2%} {r.median_utilization:>8.2%} "
                      f"{pct_50:>7.1f}% {pct_75:>7.1f}%")

    # Find optimal size
    if args.find_optimal:
        print()
        print("=" * 70)
        print("OPTIMAL ARRAY SIZE ANALYSIS")
        print("=" * 70)
        print()

        for prec in precisions:
            opt_size, opt_util = sweeper.find_optimal_size(
                results, metric='weighted_mean_utilization', precision=prec
            )
            print(f"{prec}: Optimal size = {opt_size}x{opt_size} "
                  f"(weighted utilization = {opt_util:.2%})")

    # Analyze by class
    if args.by_class:
        print()
        print("=" * 70)
        print("UTILIZATION BY DNN CLASS")
        print("=" * 70)
        print()

        class_results = sweep_by_class(db, args.sizes, args.precision)

        for dnn_class, size_results in class_results.items():
            if size_results:
                best_size = max(size_results.keys(),
                               key=lambda s: size_results[s].weighted_mean_utilization)
                best_util = size_results[best_size].weighted_mean_utilization
                print(f"{dnn_class}: Best size = {best_size}x{best_size} "
                      f"(util = {best_util:.2%})")

        # Generate class comparison plot
        plot_utilization_by_class(
            class_results,
            str(output_dir / 'utilization_by_class'),
            format=args.format,
        )

    # Generate plots
    print()
    print("Generating plots...")

    generate_all_utilization_plots(
        results,
        str(output_dir),
        precision=args.precision,
        format=args.format,
    )

    # Export to CSV
    if args.csv:
        print(f"Exporting results to {args.csv}...")
        sweeper.to_csv(results, args.csv)

    print()
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
