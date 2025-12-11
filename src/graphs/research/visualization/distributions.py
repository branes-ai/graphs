"""
Shape Distribution Plots

Publication-ready histograms and box plots for M, K, N dimension analysis.
"""

from typing import List, Optional, Dict
from pathlib import Path
import math

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

from graphs.research.shape_collection.database import ShapeDatabase
from graphs.research.visualization.publication import (
    setup_publication_style,
    get_color,
    get_color_sequence,
    save_figure,
    create_figure,
    add_grid_lines,
    COLORS,
)


def plot_dimension_histograms(
    db: ShapeDatabase,
    output_path: str,
    dimensions: List[str] = ['M', 'K', 'N'],
    by_class: bool = True,
    log_scale: bool = True,
    bins: int = 50,
    format: str = 'pdf',
    dpi: int = 300,
) -> None:
    """
    Generate histograms of M, K, N dimensions.

    Creates:
    - Overall distribution across all models
    - Per-class distributions (CNN vs Transformer) if by_class=True
    - Log-scale versions for wide ranges

    Args:
        db: ShapeDatabase with records
        output_path: Output path (without extension)
        dimensions: Which dimensions to plot
        by_class: Whether to separate by DNN class
        log_scale: Use log scale for x-axis
        bins: Number of histogram bins
        format: Output format ('pdf', 'svg', 'png')
        dpi: Resolution for PNG
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    # Get matmul dimensions DataFrame
    df = db.get_matmul_dimensions()
    if df.empty:
        print("No matmul operations found in database")
        return

    # Filter to rows with valid dimensions
    df = df[(df['M'] > 0) & (df['K'] > 0) & (df['N'] > 0)]

    if by_class:
        classes = df['model_class'].unique()
        n_classes = len(classes)

        # Create figure with subplots: dimensions x classes
        fig, axes = plt.subplots(
            len(dimensions), n_classes,
            figsize=(3.5 * n_classes, 2.5 * len(dimensions)),
            squeeze=False,
        )

        for i, dim in enumerate(dimensions):
            for j, cls in enumerate(classes):
                ax = axes[i, j]
                data = df[df['model_class'] == cls][dim].values

                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes)
                    continue

                # Use log-spaced bins if log_scale
                if log_scale and data.min() > 0:
                    log_bins = np.logspace(
                        np.log10(max(1, data.min())),
                        np.log10(data.max()),
                        bins,
                    )
                    ax.hist(data, bins=log_bins, color=get_color(cls),
                           alpha=0.7, edgecolor='white', linewidth=0.5)
                    ax.set_xscale('log')
                else:
                    ax.hist(data, bins=bins, color=get_color(cls),
                           alpha=0.7, edgecolor='white', linewidth=0.5)

                # Labels
                if i == 0:
                    ax.set_title(cls)
                if j == 0:
                    ax.set_ylabel(f'{dim} Count')
                if i == len(dimensions) - 1:
                    ax.set_xlabel(f'{dim} Value')

                add_grid_lines(ax)

                # Add statistics annotation
                median = np.median(data)
                mean = np.mean(data)
                ax.axvline(median, color='red', linestyle='--', linewidth=1,
                          label=f'Median: {median:.0f}')

        plt.tight_layout()
        save_figure(fig, output_path + '_by_class', formats=[format])
        plt.close(fig)

    # Create overall distribution plot
    fig, axes = plt.subplots(1, len(dimensions), figsize=(3.5 * len(dimensions), 2.5))
    if len(dimensions) == 1:
        axes = [axes]

    for i, dim in enumerate(dimensions):
        ax = axes[i]
        data = df[dim].values

        if log_scale and data.min() > 0:
            log_bins = np.logspace(
                np.log10(max(1, data.min())),
                np.log10(data.max()),
                bins,
            )
            ax.hist(data, bins=log_bins, color=COLORS['blue'],
                   alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.set_xscale('log')
        else:
            ax.hist(data, bins=bins, color=COLORS['blue'],
                   alpha=0.7, edgecolor='white', linewidth=0.5)

        ax.set_xlabel(f'{dim} Dimension')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {dim}')

        # Add statistics
        median = np.median(data)
        ax.axvline(median, color='red', linestyle='--', linewidth=1)
        ax.text(0.95, 0.95, f'Median: {median:.0f}\nN: {len(data)}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path + '_overall', formats=[format])
    plt.close(fig)


def plot_dimension_by_model_family(
    db: ShapeDatabase,
    output_path: str,
    families: Optional[List[str]] = None,
    dimensions: List[str] = ['M', 'K', 'N'],
    format: str = 'pdf',
) -> None:
    """
    Box plots showing dimension distributions per model family.

    Args:
        db: ShapeDatabase with records
        output_path: Output path (without extension)
        families: List of model families to include (None for auto-detect)
        dimensions: Which dimensions to plot
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        print("No matmul operations found")
        return

    # Auto-detect families if not specified
    if families is None:
        # Get most common model name prefixes
        model_counts = df['model_name'].value_counts()
        families = []
        seen_families = set()

        for model in model_counts.index[:20]:  # Top 20 models
            family = model.split('_')[0].split('-')[0].lower()
            if family not in seen_families:
                seen_families.add(family)
                families.append(family)

        families = families[:8]  # Limit to 8 for readability

    # Extract family from model name
    def get_family(model_name):
        name_lower = model_name.lower()
        for f in families:
            if name_lower.startswith(f):
                return f.capitalize()
        return 'Other'

    df['family'] = df['model_name'].apply(get_family)
    df = df[df['family'] != 'Other']

    if df.empty:
        print("No matching families found")
        return

    # Create box plots
    fig, axes = plt.subplots(1, len(dimensions), figsize=(3.5 * len(dimensions), 3.5))
    if len(dimensions) == 1:
        axes = [axes]

    unique_families = sorted(df['family'].unique())
    colors = get_color_sequence(len(unique_families))

    for i, dim in enumerate(dimensions):
        ax = axes[i]

        # Prepare data for box plot
        data_by_family = [df[df['family'] == f][dim].values for f in unique_families]

        bp = ax.boxplot(
            data_by_family,
            labels=unique_families,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('Model Family')
        ax.set_ylabel(f'{dim} Value')
        ax.set_title(f'{dim} by Model Family')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)

        add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_dimension_scatter(
    db: ShapeDatabase,
    output_path: str,
    x_dim: str = 'M',
    y_dim: str = 'N',
    color_by: str = 'model_class',
    log_scale: bool = True,
    format: str = 'pdf',
) -> None:
    """
    Scatter plot of two dimensions, colored by category.

    Args:
        db: ShapeDatabase
        output_path: Output path
        x_dim: X-axis dimension
        y_dim: Y-axis dimension
        color_by: Column to color by ('model_class' or 'op_type')
        log_scale: Use log scale
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    fig, ax = create_figure(width=4, height=3.5)

    categories = df[color_by].unique()
    colors = {cat: get_color(cat) if cat in COLORS else c
             for cat, c in zip(categories, get_color_sequence(len(categories)))}

    for cat in categories:
        subset = df[df[color_by] == cat]
        ax.scatter(
            subset[x_dim], subset[y_dim],
            c=colors[cat], label=cat, alpha=0.6, s=20,
        )

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel(f'{x_dim} Dimension')
    ax.set_ylabel(f'{y_dim} Dimension')
    ax.set_title(f'{x_dim} vs {y_dim} by {color_by}')
    ax.legend(loc='best', fontsize=8)

    add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_cumulative_distribution(
    db: ShapeDatabase,
    output_path: str,
    dimensions: List[str] = ['M', 'K', 'N'],
    by_class: bool = True,
    format: str = 'pdf',
) -> None:
    """
    Cumulative distribution function (CDF) plots for dimensions.

    Useful for understanding what fraction of operations are below certain thresholds.

    Args:
        db: ShapeDatabase
        output_path: Output path
        dimensions: Dimensions to plot
        by_class: Separate by DNN class
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    fig, axes = plt.subplots(1, len(dimensions), figsize=(3.5 * len(dimensions), 2.5))
    if len(dimensions) == 1:
        axes = [axes]

    for i, dim in enumerate(dimensions):
        ax = axes[i]

        if by_class:
            classes = df['model_class'].unique()
            for cls in classes:
                data = np.sort(df[df['model_class'] == cls][dim].values)
                cdf = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, cdf, label=cls, color=get_color(cls), linewidth=1.5)
            ax.legend(loc='lower right', fontsize=8)
        else:
            data = np.sort(df[dim].values)
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, color=COLORS['blue'], linewidth=1.5)

        ax.set_xscale('log')
        ax.set_xlabel(f'{dim} Dimension')
        ax.set_ylabel('Cumulative Fraction')
        ax.set_title(f'CDF of {dim}')
        ax.set_ylim(0, 1)

        # Add reference lines
        for threshold in [0.5, 0.9, 0.99]:
            ax.axhline(threshold, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

        add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_op_type_breakdown(
    db: ShapeDatabase,
    output_path: str,
    by_class: bool = True,
    format: str = 'pdf',
) -> None:
    """
    Bar chart showing operation type distribution.

    Args:
        db: ShapeDatabase
        output_path: Output path
        by_class: Separate by DNN class
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    # Filter to matmul ops
    matmul_db = db.filter_matmul_ops()
    df = matmul_db.to_dataframe()

    if df.empty:
        return

    if by_class:
        # Grouped bar chart
        pivot = df.groupby(['model_class', 'op_type']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 3.5))

        pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('DNN Class')
        ax.set_ylabel('Count')
        ax.set_title('Operation Types by DNN Class')
        ax.legend(title='Op Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=0)

    else:
        # Simple bar chart
        op_counts = df['op_type'].value_counts()

        fig, ax = plt.subplots(figsize=(5, 3))

        colors = get_color_sequence(len(op_counts))
        ax.bar(op_counts.index, op_counts.values, color=colors,
              edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Operation Type')
        ax.set_ylabel('Count')
        ax.set_title('Operation Type Distribution')
        ax.tick_params(axis='x', rotation=45)

    add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)
