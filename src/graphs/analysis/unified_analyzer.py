"""
Deprecated: Use graphs.estimation.unified_analyzer instead.
This module will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.analysis.unified_analyzer is deprecated. "
    "Use graphs.estimation.unified_analyzer instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

from graphs.estimation.unified_analyzer import *
