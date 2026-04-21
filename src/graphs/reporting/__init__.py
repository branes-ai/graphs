"""
Report Generation Framework

Transforms unified analysis results into various output formats, including
the micro-architectural model delivery report (Branes-branded HTML + JSON
+ PowerPoint).
"""

from graphs.reporting.report_generator import ReportGenerator
from graphs.reporting.microarch_schema import (
    CONFIDENCE_LADDER,
    LayerPanel,
    MicroarchReport,
    REPORT_LAYERS_IN_ORDER,
    empty_report,
)
from graphs.reporting.microarch_html_template import (
    BRAND_LOGO_RELATIVE,
    render_sku_page,
    render_comparison_page,
    render_index_page,
)

__all__ = [
    'ReportGenerator',
    'CONFIDENCE_LADDER',
    'LayerPanel',
    'MicroarchReport',
    'REPORT_LAYERS_IN_ORDER',
    'empty_report',
    'BRAND_LOGO_RELATIVE',
    'render_sku_page',
    'render_comparison_page',
    'render_index_page',
]
