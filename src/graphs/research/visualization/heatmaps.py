"""
2D Heatmap Visualizations

Heatmaps showing (M,N), (M,K), (K,N) dimension pair frequencies
for understanding systolic array design space.
"""

from typing import Optional, List, Tuple
from pathlib import Path
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
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
    save_figure,
    add_grid_lines,
    COLORS,
)


def _create_log_bins(data: np.ndarray, n_bins: int) -> np.ndarray:
    """Create log-spaced bins for histogram."""
    data_min = max(1, data.min())
    data_max = data.max()

    if data_max <= data_min:
        return np.array([data_min, data_min + 1])

    return np.logspace(np.log10(data_min), np.log10(data_max), n_bins + 1)


def plot_mn_heatmap(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 30,
    cmap: str = 'YlOrRd',
    format: str = 'pdf',
    title: str = '(M, N) Dimension Frequency',
) -> None:
    """
    Generate (M, N) pair frequency heatmap.

    Shows which matrix dimensions are common in real DNNs.
    Critical for understanding systolic array design space.

    Args:
        db: ShapeDatabase
        output_path: Output path (without extension)
        log_scale: Use log scale for axes
        n_bins: Number of bins per axis
        cmap: Colormap name
        format: Output format
        title: Plot title
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        print("No matmul operations found")
        return

    # Filter valid data
    df = df[(df['M'] > 0) & (df['N'] > 0)]

    M_data = df['M'].values
    N_data = df['N'].values

    fig, ax = plt.subplots(figsize=(5, 4))

    if log_scale:
        # Log-spaced bins
        M_bins = _create_log_bins(M_data, n_bins)
        N_bins = _create_log_bins(N_data, n_bins)

        # 2D histogram
        H, M_edges, N_edges = np.histogram2d(M_data, N_data, bins=[M_bins, N_bins])

        # Use log scale for color if counts vary widely
        if H.max() > 10 * H[H > 0].min():
            norm = mcolors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max())
        else:
            norm = None

        im = ax.pcolormesh(M_edges, N_edges, H.T, cmap=cmap, norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        H, M_edges, N_edges = np.histogram2d(M_data, N_data, bins=n_bins)
        im = ax.pcolormesh(M_edges, N_edges, H.T, cmap=cmap)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Count')

    ax.set_xlabel('M (Output Rows)')
    ax.set_ylabel('N (Output Columns)')
    ax.set_title(title)

    # Add reference lines for common array sizes
    array_sizes = [32, 64, 128, 256]
    for size in array_sizes:
        if M_data.min() < size < M_data.max():
            ax.axvline(size, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        if N_data.min() < size < N_data.max():
            ax.axhline(size, color='white', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_mk_heatmap(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 30,
    cmap: str = 'YlOrRd',
    format: str = 'pdf',
) -> None:
    """
    Generate (M, K) pair frequency heatmap.

    Shows output rows vs reduction dimension patterns.

    Args:
        db: ShapeDatabase
        output_path: Output path
        log_scale: Use log scale
        n_bins: Number of bins
        cmap: Colormap
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    df = df[(df['M'] > 0) & (df['K'] > 0)]

    M_data = df['M'].values
    K_data = df['K'].values

    fig, ax = plt.subplots(figsize=(5, 4))

    if log_scale:
        M_bins = _create_log_bins(M_data, n_bins)
        K_bins = _create_log_bins(K_data, n_bins)
        H, M_edges, K_edges = np.histogram2d(M_data, K_data, bins=[M_bins, K_bins])

        norm = mcolors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max()) if H.max() > 10 else None
        im = ax.pcolormesh(M_edges, K_edges, H.T, cmap=cmap, norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        H, M_edges, K_edges = np.histogram2d(M_data, K_data, bins=n_bins)
        im = ax.pcolormesh(M_edges, K_edges, H.T, cmap=cmap)

    fig.colorbar(im, ax=ax, label='Count')

    ax.set_xlabel('M (Output Rows)')
    ax.set_ylabel('K (Reduction Dimension)')
    ax.set_title('(M, K) Dimension Frequency')

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_kn_heatmap(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 30,
    cmap: str = 'YlOrRd',
    format: str = 'pdf',
) -> None:
    """
    Generate (K, N) pair frequency heatmap.

    Shows weight matrix dimension patterns (K x N).

    Args:
        db: ShapeDatabase
        output_path: Output path
        log_scale: Use log scale
        n_bins: Number of bins
        cmap: Colormap
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    df = df[(df['K'] > 0) & (df['N'] > 0)]

    K_data = df['K'].values
    N_data = df['N'].values

    fig, ax = plt.subplots(figsize=(5, 4))

    if log_scale:
        K_bins = _create_log_bins(K_data, n_bins)
        N_bins = _create_log_bins(N_data, n_bins)
        H, K_edges, N_edges = np.histogram2d(K_data, N_data, bins=[K_bins, N_bins])

        norm = mcolors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max()) if H.max() > 10 else None
        im = ax.pcolormesh(K_edges, N_edges, H.T, cmap=cmap, norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        H, K_edges, N_edges = np.histogram2d(K_data, N_data, bins=n_bins)
        im = ax.pcolormesh(K_edges, N_edges, H.T, cmap=cmap)

    fig.colorbar(im, ax=ax, label='Count')

    ax.set_xlabel('K (Reduction Dimension)')
    ax.set_ylabel('N (Output Columns)')
    ax.set_title('(K, N) Weight Matrix Dimensions')

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_all_dimension_heatmaps(
    db: ShapeDatabase,
    output_dir: str,
    log_scale: bool = True,
    n_bins: int = 30,
    format: str = 'pdf',
) -> None:
    """
    Generate all three dimension pair heatmaps.

    Args:
        db: ShapeDatabase
        output_dir: Output directory
        log_scale: Use log scale
        n_bins: Number of bins
        format: Output format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mn_heatmap(db, str(output_dir / 'mn_heatmap'), log_scale, n_bins, format=format)
    plot_mk_heatmap(db, str(output_dir / 'mk_heatmap'), log_scale, n_bins, format=format)
    plot_kn_heatmap(db, str(output_dir / 'kn_heatmap'), log_scale, n_bins, format=format)


def plot_mn_heatmap_by_class(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 25,
    format: str = 'pdf',
) -> None:
    """
    Generate (M, N) heatmaps separated by DNN class.

    Creates a grid of heatmaps, one for each class.

    Args:
        db: ShapeDatabase
        output_path: Output path
        log_scale: Use log scale
        n_bins: Number of bins
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    df = df[(df['M'] > 0) & (df['N'] > 0)]

    classes = sorted(df['model_class'].unique())
    n_classes = len(classes)

    if n_classes == 0:
        return

    # Determine grid layout
    n_cols = min(3, n_classes)
    n_rows = math.ceil(n_classes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    # Global bins for consistent comparison
    M_all = df['M'].values
    N_all = df['N'].values

    if log_scale:
        M_bins = _create_log_bins(M_all, n_bins)
        N_bins = _create_log_bins(N_all, n_bins)
    else:
        M_bins = np.linspace(M_all.min(), M_all.max(), n_bins + 1)
        N_bins = np.linspace(N_all.min(), N_all.max(), n_bins + 1)

    # Find global max for consistent color scale
    global_max = 0
    for cls in classes:
        cls_df = df[df['model_class'] == cls]
        H, _, _ = np.histogram2d(cls_df['M'].values, cls_df['N'].values, bins=[M_bins, N_bins])
        global_max = max(global_max, H.max())

    for idx, cls in enumerate(classes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        cls_df = df[df['model_class'] == cls]
        H, _, _ = np.histogram2d(cls_df['M'].values, cls_df['N'].values, bins=[M_bins, N_bins])

        if log_scale:
            norm = mcolors.LogNorm(vmin=1, vmax=global_max)
            im = ax.pcolormesh(M_bins, N_bins, H.T, cmap='YlOrRd', norm=norm)
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            im = ax.pcolormesh(M_bins, N_bins, H.T, cmap='YlOrRd', vmin=0, vmax=global_max)

        ax.set_xlabel('M')
        ax.set_ylabel('N')
        ax.set_title(f'{cls} (n={len(cls_df)})')

    # Hide unused axes
    for idx in range(n_classes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Count', shrink=0.8)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_mn_heatmap_annotated(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 30,
    cmap: str = 'YlOrRd',
    format: str = 'pdf',
) -> None:
    """
    Generate annotated (M, N) heatmap with callouts for key observations.
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    df = df[(df['M'] > 0) & (df['N'] > 0)]
    M_data = df['M'].values
    N_data = df['N'].values

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_scale:
        M_bins = _create_log_bins(M_data, n_bins)
        N_bins = _create_log_bins(N_data, n_bins)
        H, M_edges, N_edges = np.histogram2d(M_data, N_data, bins=[M_bins, N_bins])
        norm = mcolors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max()) if H.max() > 10 else None
        im = ax.pcolormesh(M_edges, N_edges, H.T, cmap=cmap, norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        H, M_edges, N_edges = np.histogram2d(M_data, N_data, bins=n_bins)
        im = ax.pcolormesh(M_edges, N_edges, H.T, cmap=cmap)

    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('M (Output Rows)', fontsize=11)
    ax.set_ylabel('N (Output Columns)', fontsize=11)
    ax.set_title('(M, N) Output Tensor Dimensions', fontsize=12, fontweight='bold')

    # Annotation style
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black')

    # 1. Hotspot at M~196 (14x14 feature maps)
    ax.annotate('14x14 feature maps\n(42% of ops)',
                xy=(196, 512), xytext=(20, 1500),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # 2. M=1 column (FC layers batch=1)
    ax.annotate('M=1: FC layers\nwith batch=1\n(16% of ops)',
                xy=(1, 100), xytext=(3, 30),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # 3. Vertical banding - M values cluster at specific spatial sizes
    # Place callout box in center-bottom, arrows originate from top of box
    bbox_center_x = 150  # center of box in data coordinates
    bbox_y = 6  # box position

    # Draw the text box without arrow
    ax.text(bbox_center_x, bbox_y,
            'Vertical bands: M clusters\nat spatial resolutions\n49, 196, 784, 3136\n(7x7, 14x14, 28x28, 56x56)',
            fontsize=9, ha='center', va='bottom', bbox=bbox_props)

    # Draw arrows from just above the box to each vertical band
    arrow_start_y = 15  # even closer to the text box
    for m_val in [49, 196, 784, 3136]:
        ax.annotate('', xy=(m_val, 35), xytext=(bbox_center_x, arrow_start_y),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                  color='black', alpha=0.7))

    # 4. Large M (early layers) - point to top-right of the data square
    ax.annotate('Early layers:\n224x224, 112x112\nfeature maps',
                xy=(15000, 170), xytext=(3500, 1500),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='black'))

    # Add array size reference lines (vertical) for M dimension
    for size in [32, 64, 128]:
        ax.axvline(size, color='blue', linestyle=':', alpha=0.5, linewidth=1)

    # Add channel count reference lines (horizontal) for N dimension
    for size in [32, 64, 128, 256, 512, 1024]:
        ax.axhline(size, color='blue', linestyle=':', alpha=0.5, linewidth=1)

    # Label N channel lines on the right margin
    ax.text(M_data.max() * 1.3, 1024, '1024', fontsize=7, color='blue', va='center')
    ax.text(M_data.max() * 1.3, 512, '512', fontsize=7, color='blue', va='center')
    ax.text(M_data.max() * 1.3, 256, '256', fontsize=7, color='blue', va='center')
    ax.text(M_data.max() * 1.3, 128, '128', fontsize=7, color='blue', va='center')
    ax.text(M_data.max() * 1.3, 64, '64', fontsize=7, color='blue', va='center')
    ax.text(M_data.max() * 1.3, 32, '32', fontsize=7, color='blue', va='center')

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_mk_heatmap_annotated(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 30,
    cmap: str = 'YlOrRd',
    format: str = 'pdf',
) -> None:
    """
    Generate annotated (M, K) heatmap with callouts for key observations.
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    df = df[(df['M'] > 0) & (df['K'] > 0)]
    M_data = df['M'].values
    K_data = df['K'].values

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_scale:
        M_bins = _create_log_bins(M_data, n_bins)
        K_bins = _create_log_bins(K_data, n_bins)
        H, M_edges, K_edges = np.histogram2d(M_data, K_data, bins=[M_bins, K_bins])
        norm = mcolors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max()) if H.max() > 10 else None
        im = ax.pcolormesh(M_edges, K_edges, H.T, cmap=cmap, norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        H, M_edges, K_edges = np.histogram2d(M_data, K_data, bins=n_bins)
        im = ax.pcolormesh(M_edges, K_edges, H.T, cmap=cmap)

    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('M (Output Rows)', fontsize=11)
    ax.set_ylabel('K (Reduction Dimension)', fontsize=11)
    ax.set_title('(M, K) Input Activation Dimensions', fontsize=12, fontweight='bold')

    # Annotation style
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black')

    # 1. Vertical bands at spatial sizes - vertically oriented text, arrows counterclockwise
    bbox_center_x = 8000
    bbox_y = 4500

    # Draw the text box - vertically oriented text (one item per line)
    ax.text(bbox_center_x, bbox_y,
            'Vertical bands:\n(feature map\nsizes)\n7x7\n14x14\n28x28\n56x56',
            fontsize=9, ha='left', va='center', bbox=bbox_props)

    # Draw arrows from left of the box to top of each vertical band (49, 196, 784, 3136)
    # Counterclockwise flow: positive rad curves left
    arrow_start_x = 7000
    for m_val, arrow_y in [(49, 5500), (196, 5000), (784, 4500), (3136, 4000)]:
        ax.annotate('', xy=(m_val, arrow_y), xytext=(arrow_start_x, bbox_y),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15',
                                  color='black', alpha=0.7))

    # 2. K=9 band (depthwise 3x3)
    ax.annotate('K=9: depthwise\n3x3 convolutions',
                xy=(200, 9), xytext=(1000, 15),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # 2b. K=49 band (7x7 kernels)
    ax.annotate('K=49: 7x7 kernels\n(90 ops)',
                xy=(200, 49), xytext=(1000, 70),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # 3. Inverse relationship - move left to avoid colorbar, positioned between data points
    ax.annotate('CNN funnel:\nlarge spatial -> small K\nsmall spatial -> large K',
                xy=(3000, 100), xytext=(4000, 250),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='black'))

    # 4. Large K region (late layers)
    ax.annotate('Late layers:\nhigh channel count\n(K > 1000)',
                xy=(49, 2000), xytext=(10, 8000),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_kn_heatmap_annotated(
    db: ShapeDatabase,
    output_path: str,
    log_scale: bool = True,
    n_bins: int = 30,
    cmap: str = 'YlOrRd',
    format: str = 'pdf',
) -> None:
    """
    Generate annotated (K, N) heatmap with callouts for key observations.
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    df = df[(df['K'] > 0) & (df['N'] > 0)]
    K_data = df['K'].values
    N_data = df['N'].values

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_scale:
        K_bins = _create_log_bins(K_data, n_bins)
        N_bins = _create_log_bins(N_data, n_bins)
        H, K_edges, N_edges = np.histogram2d(K_data, N_data, bins=[K_bins, N_bins])
        norm = mcolors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max()) if H.max() > 10 else None
        im = ax.pcolormesh(K_edges, N_edges, H.T, cmap=cmap, norm=norm)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        H, K_edges, N_edges = np.histogram2d(K_data, N_data, bins=n_bins)
        im = ax.pcolormesh(K_edges, N_edges, H.T, cmap=cmap)

    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('K (Reduction Dimension)', fontsize=11)
    ax.set_ylabel('N (Output Columns)', fontsize=11)
    ax.set_title('(K, N) Weight Matrix Dimensions', fontsize=12, fontweight='bold')

    # Annotation style
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black')

    # 1. Diagonal correlation
    ax.annotate('Diagonal trend:\nK and N scale together\nin deeper networks',
                xy=(300, 300), xytext=(30, 1500),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # 2. Transformer region
    ax.annotate('Transformer signature:\nK=768-1024, N=768-3072\n(attention + MLP)',
                xy=(768, 1024), xytext=(2000, 2500),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # 3. Dense mid-range region - move bubble to right side to avoid overlap
    ax.annotate('CNN sweet spot:\nK=64-512, N=128-512\n(most conv layers)',
                xy=(150, 200), xytext=(400, 60),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='black'))

    # 4. Small weights region - keep on left side
    ax.annotate('Small weights:\nearly layers,\ndepthwise conv',
                xy=(20, 40), xytext=(5, 150),
                fontsize=9, ha='left', bbox=bbox_props,
                arrowprops=arrow_props)

    # Add weight size contours (approximate)
    # 1MB = 500K elements at BF16, so K*N = 500K
    # Add explanation label for diagonal lines
    ax.text(3500, 15, 'Diagonal lines:\nweight size at BF16',
            fontsize=8, ha='left', va='bottom', color='blue', alpha=0.8)

    for size_mb, label in [(0.01, '10KB'), (0.1, '100KB'), (1, '1MB'), (4, '4MB')]:
        elements = size_mb * 1024 * 1024 / 2  # BF16 = 2 bytes
        k_range = np.logspace(1, 4, 100)
        n_line = elements / k_range
        valid = (n_line >= N_data.min()) & (n_line <= N_data.max())
        if valid.any():
            ax.plot(k_range[valid], n_line[valid], 'b--', alpha=0.4, linewidth=1)
            # Label at midpoint
            mid_idx = len(k_range[valid]) // 2
            ax.text(k_range[valid][mid_idx], n_line[valid][mid_idx] * 1.2,
                   label, fontsize=8, color='blue', alpha=0.7)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)


def plot_all_annotated_heatmaps(
    db: ShapeDatabase,
    output_dir: str,
    format: str = 'pdf',
) -> None:
    """
    Generate all three annotated dimension heatmaps.

    Args:
        db: ShapeDatabase
        output_dir: Output directory
        format: Output format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mn_heatmap_annotated(db, str(output_dir / 'mn_heatmap_annotated'), format=format)
    plot_mk_heatmap_annotated(db, str(output_dir / 'mk_heatmap_annotated'), format=format)
    plot_kn_heatmap_annotated(db, str(output_dir / 'kn_heatmap_annotated'), format=format)


def plot_dimension_vs_array_size(
    db: ShapeDatabase,
    output_path: str,
    array_sizes: List[int] = [16, 32, 64, 128],
    dimension: str = 'M',
    format: str = 'pdf',
) -> None:
    """
    Plot histogram showing what fraction of operations fit within different array sizes.

    Args:
        db: ShapeDatabase
        output_path: Output path
        array_sizes: Array sizes to mark
        dimension: Which dimension to analyze
        format: Output format
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")

    setup_publication_style()

    df = db.get_matmul_dimensions()
    if df.empty:
        return

    data = df[dimension].values
    data = data[data > 0]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Plot histogram
    bins = _create_log_bins(data, 50)
    ax.hist(data, bins=bins, alpha=0.7, color=COLORS['blue'], edgecolor='white', linewidth=0.5)
    ax.set_xscale('log')

    # Mark array sizes and compute fractions
    y_max = ax.get_ylim()[1]
    for i, size in enumerate(array_sizes):
        fraction = (data <= size).sum() / len(data)
        ax.axvline(size, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(size * 1.1, y_max * (0.9 - i * 0.1),
               f'{size}: {fraction:.1%}',
               fontsize=8, color='red')

    ax.set_xlabel(f'{dimension} Dimension')
    ax.set_ylabel('Count')
    ax.set_title(f'{dimension} Distribution vs Array Sizes')

    add_grid_lines(ax)

    plt.tight_layout()
    save_figure(fig, output_path, formats=[format])
    plt.close(fig)
