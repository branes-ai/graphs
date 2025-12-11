"""
Publication-Ready Visualization Utilities

Configure matplotlib for publication-quality plots and generate LaTeX tables.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Publication-quality color palettes
COLORS = {
    # Primary palette (colorblind-friendly)
    'blue': '#0077BB',
    'orange': '#EE7733',
    'green': '#009988',
    'red': '#CC3311',
    'purple': '#AA3377',
    'cyan': '#33BBEE',
    'gray': '#BBBBBB',

    # DNN class colors
    'CNN': '#0077BB',
    'Encoder': '#EE7733',
    'Decoder': '#009988',
    'FullTransformer': '#CC3311',
    'Unknown': '#BBBBBB',
}

# Color sequence for multiple categories
COLOR_SEQUENCE = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#AA3377', '#33BBEE']


def setup_publication_style(
    font_family: str = 'serif',
    font_size: int = 10,
    figure_width: float = 3.5,
    figure_height: float = 2.5,
    dpi: int = 300,
    use_latex: bool = False,
) -> None:
    """
    Configure matplotlib for publication-quality plots.

    Args:
        font_family: Font family ('serif' for papers, 'sans-serif' for slides)
        font_size: Base font size
        figure_width: Default figure width in inches (3.5 for single column)
        figure_height: Default figure height in inches
        dpi: Resolution for raster output
        use_latex: Use LaTeX for text rendering (requires LaTeX installation)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for setup_publication_style()")

    plt.rcParams.update({
        # Font settings
        'font.family': font_family,
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'legend.title_fontsize': font_size,

        # Figure settings
        'figure.figsize': (figure_width, figure_height),
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Axes settings
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Tick settings
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',

        # LaTeX settings
        'text.usetex': use_latex,
    })

    # Set serif fonts for publication
    if font_family == 'serif':
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def setup_presentation_style(
    font_size: int = 14,
    figure_width: float = 8,
    figure_height: float = 5,
) -> None:
    """
    Configure matplotlib for presentation slides.

    Args:
        font_size: Base font size (larger for readability)
        figure_width: Figure width in inches
        figure_height: Figure height in inches
    """
    setup_publication_style(
        font_family='sans-serif',
        font_size=font_size,
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=150,
        use_latex=False,
    )


def get_color(category: str) -> str:
    """
    Get color for a category.

    Args:
        category: Category name (DNN class or custom)

    Returns:
        Hex color code
    """
    return COLORS.get(category, COLORS['gray'])


def get_color_sequence(n: int) -> List[str]:
    """
    Get a sequence of n distinct colors.

    Args:
        n: Number of colors needed

    Returns:
        List of hex color codes
    """
    if n <= len(COLOR_SEQUENCE):
        return COLOR_SEQUENCE[:n]
    else:
        # Cycle through colors if more are needed
        return [COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i in range(n)]


def generate_latex_table(
    data: 'pd.DataFrame',
    output_path: Optional[str] = None,
    caption: str = '',
    label: str = '',
    column_format: Optional[str] = None,
    escape: bool = True,
    float_format: str = '%.2f',
) -> str:
    """
    Generate LaTeX table from DataFrame.

    Args:
        data: pandas DataFrame
        output_path: Optional path to save .tex file
        caption: Table caption
        label: LaTeX label for referencing
        column_format: Column alignment (e.g., 'lrr')
        escape: Escape special LaTeX characters
        float_format: Format string for floats

    Returns:
        LaTeX table string
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for generate_latex_table()")

    # Generate column format if not provided
    if column_format is None:
        column_format = 'l' + 'r' * (len(data.columns) - 1)

    # Generate LaTeX
    latex = data.to_latex(
        index=False,
        escape=escape,
        float_format=lambda x: float_format % x,
        column_format=column_format,
    )

    # Wrap in table environment with caption and label
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}
"""

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(full_latex)

    return full_latex


def generate_latex_summary_table(
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
    caption: str = 'Shape Statistics Summary',
    label: str = 'tab:shape_stats',
) -> str:
    """
    Generate LaTeX table from statistics dictionary.

    Args:
        stats: Statistics dictionary from ShapeDatabase.get_statistics()
        output_path: Optional path to save .tex file
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for generate_latex_summary_table()")

    # Extract key statistics
    rows = []

    # Overall counts
    rows.append(['Total Records', stats.get('total_records', 0)])
    rows.append(['Matmul Operations', stats.get('matmul_ops', 0)])
    rows.append(['Unique Models', stats.get('unique_models', 0)])

    # M, K, N statistics
    for dim in ['M', 'K', 'N']:
        dim_stats = stats.get(f'{dim}_stats', {})
        if dim_stats:
            rows.append([f'{dim} Min', dim_stats.get('min', 0)])
            rows.append([f'{dim} Max', dim_stats.get('max', 0)])
            rows.append([f'{dim} Mean', f"{dim_stats.get('mean', 0):.1f}"])
            rows.append([f'{dim} Median', dim_stats.get('median', 0)])

    df = pd.DataFrame(rows, columns=['Metric', 'Value'])

    return generate_latex_table(
        df,
        output_path=output_path,
        caption=caption,
        label=label,
        column_format='lr',
    )


def save_figure(
    fig: 'plt.Figure',
    path: str,
    formats: List[str] = ['pdf', 'svg', 'png'],
    dpi: int = 300,
) -> None:
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib Figure
        path: Base path (without extension)
        formats: List of formats to save
        dpi: Resolution for raster formats
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for save_figure()")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(
            path.with_suffix(f'.{fmt}'),
            format=fmt,
            dpi=dpi if fmt == 'png' else None,
            bbox_inches='tight',
            pad_inches=0.05,
        )


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: float = 3.5,
    height: float = 2.5,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple:
    """
    Create figure with publication settings.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        width: Figure width per column in inches
        height: Figure height per row in inches
        sharex: Share x-axis across subplots
        sharey: Share y-axis across subplots

    Returns:
        (fig, axes) tuple
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for create_figure()")

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width * ncols, height * nrows),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    # Return single axis if 1x1
    if nrows == 1 and ncols == 1:
        return fig, axes[0, 0]
    elif nrows == 1:
        return fig, axes[0]
    elif ncols == 1:
        return fig, axes[:, 0]
    else:
        return fig, axes


def add_grid_lines(ax, alpha: float = 0.3) -> None:
    """Add subtle grid lines to axis."""
    if HAS_MATPLOTLIB:
        ax.grid(True, alpha=alpha, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)


def format_large_number(x: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if abs(x) >= 1e9:
        return f'{x/1e9:.1f}B'
    elif abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'


def format_axis_labels(ax, x_label: str = '', y_label: str = '') -> None:
    """Set axis labels with proper formatting."""
    if HAS_MATPLOTLIB:
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
