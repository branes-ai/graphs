"""
Visualization Module

Publication-ready shape distribution plots and heatmaps.

Functions:
    plot_dimension_histograms: M, K, N histograms overall and by DNN class
    plot_dimension_by_model_family: Box plots per model family
    plot_mn_heatmap: (M, N) pair frequency heatmap
    plot_mk_heatmap: (M, K) patterns
    plot_kn_heatmap: (K, N) weight patterns
    setup_publication_style: Configure matplotlib for publication
    generate_latex_table: DataFrame to LaTeX table
"""

from graphs.research.visualization.distributions import (
    plot_dimension_histograms,
    plot_dimension_by_model_family,
)
from graphs.research.visualization.heatmaps import (
    plot_mn_heatmap,
    plot_mk_heatmap,
    plot_kn_heatmap,
)
from graphs.research.visualization.publication import (
    setup_publication_style,
    generate_latex_table,
)

__all__ = [
    'plot_dimension_histograms',
    'plot_dimension_by_model_family',
    'plot_mn_heatmap',
    'plot_mk_heatmap',
    'plot_kn_heatmap',
    'setup_publication_style',
    'generate_latex_table',
]
