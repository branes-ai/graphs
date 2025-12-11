#!/usr/bin/env python
"""
Shape Analysis and Visualization Tool

Analyze tensor shape distributions and generate publication-ready plots.

Usage:
    python cli/research/analyze_shapes.py --input shapes.parquet --output plots/
    python cli/research/analyze_shapes.py --input shapes.parquet --heatmaps --format pdf
    python cli/research/analyze_shapes.py --input shapes.parquet --by-class
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

from graphs.research.shape_collection import ShapeDatabase
from graphs.research.visualization import (
    plot_dimension_histograms,
    plot_dimension_by_model_family,
    plot_mn_heatmap,
    plot_mk_heatmap,
    plot_kn_heatmap,
    setup_publication_style,
    generate_latex_table,
)
from graphs.research.visualization.distributions import (
    plot_cumulative_distribution,
    plot_op_type_breakdown,
    plot_dimension_scatter,
)
from graphs.research.visualization.heatmaps import (
    plot_all_dimension_heatmaps,
    plot_mn_heatmap_by_class,
    plot_dimension_vs_array_size,
)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze shape distributions and generate plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all standard plots
    python cli/research/analyze_shapes.py --input shapes.parquet --output plots/

    # Generate heatmaps only
    python cli/research/analyze_shapes.py --input shapes.parquet --output plots/ --heatmaps

    # Generate histograms by DNN class
    python cli/research/analyze_shapes.py --input shapes.parquet --output plots/ --by-class

    # Export statistics to LaTeX
    python cli/research/analyze_shapes.py --input shapes.parquet --latex-tables tables/
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
        required=True,
        help='Output directory for plots',
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='pdf',
        choices=['pdf', 'svg', 'png'],
        help='Output format (default: pdf)',
    )
    parser.add_argument(
        '--by-class',
        action='store_true',
        help='Generate separate plots for each DNN class',
    )
    parser.add_argument(
        '--heatmaps',
        action='store_true',
        help='Generate dimension heatmaps',
    )
    parser.add_argument(
        '--histograms',
        action='store_true',
        help='Generate dimension histograms',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all plot types',
    )
    parser.add_argument(
        '--latex-tables',
        type=str,
        default=None,
        help='Output directory for LaTeX tables',
    )
    parser.add_argument(
        '--log-scale',
        action='store_true',
        default=True,
        help='Use log scale for axes (default: True)',
    )
    parser.add_argument(
        '--no-log-scale',
        action='store_false',
        dest='log_scale',
        help='Disable log scale',
    )

    args = parser.parse_args()

    # Load database
    print(f"Loading shapes from {args.input}...")
    db = ShapeDatabase.load(args.input)

    stats = db.get_statistics()
    print(f"Loaded {stats['total_records']} records from {stats['unique_models']} models")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to plot
    do_all = args.all or (not args.heatmaps and not args.histograms)
    do_heatmaps = args.heatmaps or do_all
    do_histograms = args.histograms or do_all

    # Generate plots
    print("Generating plots...")

    if do_histograms:
        print("  - Dimension histograms")
        plot_dimension_histograms(
            db,
            str(output_dir / 'dimension_histograms'),
            by_class=args.by_class,
            log_scale=args.log_scale,
            format=args.format,
        )

        print("  - Dimension box plots by family")
        plot_dimension_by_model_family(
            db,
            str(output_dir / 'dimension_by_family'),
            format=args.format,
        )

        print("  - Cumulative distributions")
        plot_cumulative_distribution(
            db,
            str(output_dir / 'cumulative_distribution'),
            by_class=args.by_class,
            format=args.format,
        )

        print("  - Operation type breakdown")
        plot_op_type_breakdown(
            db,
            str(output_dir / 'op_type_breakdown'),
            by_class=args.by_class,
            format=args.format,
        )

    if do_heatmaps:
        print("  - (M, N) heatmap")
        plot_mn_heatmap(
            db,
            str(output_dir / 'mn_heatmap'),
            log_scale=args.log_scale,
            format=args.format,
        )

        print("  - (M, K) heatmap")
        plot_mk_heatmap(
            db,
            str(output_dir / 'mk_heatmap'),
            log_scale=args.log_scale,
            format=args.format,
        )

        print("  - (K, N) heatmap")
        plot_kn_heatmap(
            db,
            str(output_dir / 'kn_heatmap'),
            log_scale=args.log_scale,
            format=args.format,
        )

        if args.by_class:
            print("  - (M, N) heatmaps by class")
            plot_mn_heatmap_by_class(
                db,
                str(output_dir / 'mn_heatmap_by_class'),
                log_scale=args.log_scale,
                format=args.format,
            )

        print("  - Dimension vs array size analysis")
        for dim in ['M', 'N']:
            plot_dimension_vs_array_size(
                db,
                str(output_dir / f'{dim.lower()}_vs_array_size'),
                dimension=dim,
                format=args.format,
            )

    # Generate LaTeX tables
    if args.latex_tables:
        latex_dir = Path(args.latex_tables)
        latex_dir.mkdir(parents=True, exist_ok=True)

        print(f"  - LaTeX tables to {latex_dir}")

        # Statistics summary table
        from graphs.research.visualization.publication import generate_latex_summary_table
        generate_latex_summary_table(
            stats,
            output_path=str(latex_dir / 'shape_statistics.tex'),
            caption='Tensor Shape Statistics Summary',
            label='tab:shape_stats',
        )

        # Per-class statistics
        try:
            import pandas as pd

            df = db.get_matmul_dimensions()
            if not df.empty:
                class_stats = df.groupby('model_class').agg({
                    'M': ['min', 'max', 'mean', 'median'],
                    'K': ['min', 'max', 'mean', 'median'],
                    'N': ['min', 'max', 'mean', 'median'],
                }).round(0)
                class_stats.to_latex(
                    str(latex_dir / 'per_class_statistics.tex'),
                    caption='Matmul Dimensions by DNN Class',
                    label='tab:per_class_stats',
                )
        except ImportError:
            pass

    print()
    print(f"Plots saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
