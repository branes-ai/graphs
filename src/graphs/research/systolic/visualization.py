"""
Systolic Array Utilization Visualization

Publication-ready plots for utilization analysis.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from graphs.research.systolic.sweep import SweepResult, ARRAY_SIZES
from graphs.research.visualization.publication import (
    setup_publication_style,
    save_figure,
    add_grid_lines,
    get_color,
    get_color_sequence,
    COLORS,
)


def plot_utilization_vs_array_size(
    results: Dict[Tuple[int, str], SweepResult],
    output_path: str,
    precision: str = 'BF16',
    metric: str = 'weighted_mean_utilization',
    show_percentiles: bool = True,
    format: str = 'pdf',
) -> None:
    """
    Line plot of utilization vs array size.

    Shows how larger arrays lead to lower utilization for small shapes.

    Args:
        results: Sweep results dictionary
        output_path: Output path
        precision: Precision to plot
        metric: Metric to plot
        show_percentiles: Show p10/p90 bands
        format: Output format
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    setup_publication_style()

    # Extract data
    sizes = []
    values = []
    p10_values = []
    p90_values = []

    for size in ARRAY_SIZES:
        key = (size, precision)
        if key in results:
            sizes.append(size)
            values.append(getattr(results[key], metric, 0.0))
            p10_values.append(results[key].p10_utilization)
            p90_values.append(results[key].p90_utilization)

    if not sizes:
        print("No results found for precision:", precision)
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Main line
    ax.plot(sizes, values, 'o-', color=COLORS['blue'], linewidth=2,
           markersize=6, label=metric.replace('_', ' ').title())

    if show_percentiles:
        # P10-P90 band
        ax.fill_between(sizes, p10_values, p90_values,
                       alpha=0.2, color=COLORS['blue'], label='P10-P90 range')

    ax.set_xlabel('Systolic Array Size')
    ax.set_ylabel('Utilization')
    ax.set_title(f'Utilization vs Array Size ({precision})')
    ax.set_ylim(0, 1.0)

    # Mark common sizes
    for size in [32, 64, 128]:
        if size in sizes:
            idx = sizes.index(size)
            ax.axvline(size, color='gray', linestyle='--', alpha=0.3)
            ax.annotate(f'{values[idx]:.2f}',
                       xy=(size, values[idx]),
                       xytext=(size + 5, values[idx] + 0.05),
                       fontsize=8)

    ax.legend(loc='upper right', fontsize=8)
    add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_utilization_by_class(
    class_results: Dict[str, Dict[int, SweepResult]],
    output_path: str,
    metric: str = 'weighted_mean_utilization',
    format: str = 'pdf',
) -> None:
    """
    Line plot comparing utilization across DNN classes.

    Args:
        class_results: Results from sweep_by_class()
        output_path: Output path
        metric: Metric to plot
        format: Output format
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    setup_publication_style()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for dnn_class, size_results in class_results.items():
        sizes = sorted(size_results.keys())
        values = [getattr(size_results[s], metric, 0.0) for s in sizes]

        ax.plot(sizes, values, 'o-', color=get_color(dnn_class),
               linewidth=1.5, markersize=5, label=dnn_class)

    ax.set_xlabel('Systolic Array Size')
    ax.set_ylabel('Utilization')
    ax.set_title('Utilization by DNN Class')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=8)
    add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_utilization_heatmap(
    results: Dict[Tuple[int, str], SweepResult],
    output_path: str,
    precisions: List[str] = ['FP32', 'BF16', 'INT8'],
    metric: str = 'weighted_mean_utilization',
    format: str = 'pdf',
) -> None:
    """
    Heatmap: array_size x precision, colored by utilization.

    Args:
        results: Sweep results dictionary
        output_path: Output path
        precisions: Precisions to include
        metric: Metric to visualize
        format: Output format
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    setup_publication_style()

    # Build matrix
    sizes = [s for s in ARRAY_SIZES if any((s, p) in results for p in precisions)]
    matrix = []

    for prec in precisions:
        row = []
        for size in sizes:
            key = (size, prec)
            if key in results:
                row.append(getattr(results[key], metric, 0.0))
            else:
                row.append(0.0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 3))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(precisions)))
    ax.set_yticklabels(precisions)

    ax.set_xlabel('Array Size')
    ax.set_ylabel('Precision')
    ax.set_title('Utilization by Array Size and Precision')

    # Annotate cells
    for i in range(len(precisions)):
        for j in range(len(sizes)):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label='Utilization')

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_utilization_histogram_grid(
    results: Dict[Tuple[int, str], SweepResult],
    output_path: str,
    array_sizes: List[int] = [16, 32, 64, 128],
    precision: str = 'BF16',
    format: str = 'pdf',
) -> None:
    """
    Grid of histograms showing utilization distribution per array size.

    Args:
        results: Sweep results dictionary
        output_path: Output path
        array_sizes: Array sizes to show
        precision: Precision to filter
        format: Output format
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    setup_publication_style()

    n_plots = len(array_sizes)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)

    for idx, size in enumerate(array_sizes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        key = (size, precision)
        if key not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        result = results[key]

        # Plot histogram
        bins = result.histogram_bins
        counts = result.utilization_histogram

        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]
        ax.bar(bin_centers, counts, width=0.08, color=COLORS['blue'],
              edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Utilization')
        ax.set_ylabel('Count')
        ax.set_title(f'{size}x{size} Array')
        ax.set_xlim(0, 1)

        # Add mean line
        ax.axvline(result.mean_utilization, color='red', linestyle='--',
                  linewidth=1, label=f'Mean: {result.mean_utilization:.2f}')

        add_grid_lines(ax)

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_optimal_size_analysis(
    results: Dict[Tuple[int, str], SweepResult],
    output_path: str,
    precision: str = 'BF16',
    format: str = 'pdf',
) -> None:
    """
    Plot showing trade-off between array size and utilization.

    Helps identify "sweet spot" array sizes.

    Args:
        results: Sweep results dictionary
        output_path: Output path
        precision: Precision
        format: Output format
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")

    setup_publication_style()

    sizes = []
    weighted_utils = []
    pct_above_50 = []
    pct_above_75 = []

    for size in ARRAY_SIZES:
        key = (size, precision)
        if key in results:
            result = results[key]
            sizes.append(size)
            weighted_utils.append(result.weighted_mean_utilization)
            total = result.total_shapes
            pct_above_50.append(result.shapes_above_50pct / total * 100 if total > 0 else 0)
            pct_above_75.append(result.shapes_above_75pct / total * 100 if total > 0 else 0)

    if not sizes:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left plot: Utilization
    ax1.plot(sizes, weighted_utils, 'o-', color=COLORS['blue'], linewidth=2, markersize=6)
    ax1.set_xlabel('Array Size')
    ax1.set_ylabel('Weighted Mean Utilization')
    ax1.set_title('Utilization vs Array Size')
    ax1.set_ylim(0, 1.0)
    add_grid_lines(ax1)

    # Find optimal size (highest utilization)
    best_idx = weighted_utils.index(max(weighted_utils))
    ax1.scatter([sizes[best_idx]], [weighted_utils[best_idx]], color='red', s=100, zorder=5,
               label=f'Best: {sizes[best_idx]}x{sizes[best_idx]}')
    ax1.legend(loc='lower left', fontsize=8)

    # Right plot: Percentage above thresholds
    ax2.plot(sizes, pct_above_50, 'o-', color=COLORS['green'], linewidth=2, markersize=6, label='>50%')
    ax2.plot(sizes, pct_above_75, 's-', color=COLORS['orange'], linewidth=2, markersize=6, label='>75%')
    ax2.set_xlabel('Array Size')
    ax2.set_ylabel('Percentage of Operations')
    ax2.set_title('Operations Above Utilization Threshold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower left', fontsize=8)
    add_grid_lines(ax2)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_size_comparison_table(
    results: Dict[Tuple[int, str], SweepResult],
    output_path: str,
    precision: str = 'BF16',
    format: str = 'pdf',
) -> None:
    """
    Create a visual table comparing key metrics across array sizes.

    Args:
        results: Sweep results dictionary
        output_path: Output path
        precision: Precision
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    # Build comparison data
    data = []
    for size in ARRAY_SIZES:
        key = (size, precision)
        if key in results:
            r = results[key]
            data.append({
                'Size': f'{size}x{size}',
                'Mean': f'{r.mean_utilization:.2f}',
                'Weighted': f'{r.weighted_mean_utilization:.2f}',
                'Median': f'{r.median_utilization:.2f}',
                'P10': f'{r.p10_utilization:.2f}',
                'P90': f'{r.p90_utilization:.2f}',
                '>50%': f'{r.shapes_above_50pct / r.total_shapes * 100:.0f}%' if r.total_shapes > 0 else '-',
                '>75%': f'{r.shapes_above_75pct / r.total_shapes * 100:.0f}%' if r.total_shapes > 0 else '-',
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, len(data) * 0.4 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title(f'Utilization Comparison ({precision})', fontsize=11, pad=20)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def generate_all_utilization_plots(
    results: Dict[Tuple[int, str], SweepResult],
    output_dir: str,
    precision: str = 'BF16',
    format: str = 'pdf',
) -> None:
    """
    Generate all utilization visualization plots.

    Args:
        results: Sweep results dictionary
        output_dir: Output directory
        precision: Precision
        format: Output format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_utilization_vs_array_size(
        results, str(output_dir / 'utilization_vs_size'), precision, format=format
    )

    plot_utilization_heatmap(
        results, str(output_dir / 'utilization_heatmap'), format=format
    )

    plot_utilization_histogram_grid(
        results, str(output_dir / 'utilization_histograms'), precision=precision, format=format
    )

    plot_optimal_size_analysis(
        results, str(output_dir / 'optimal_size_analysis'), precision, format=format
    )

    plot_size_comparison_table(
        results, str(output_dir / 'size_comparison_table'), precision, format=format
    )
